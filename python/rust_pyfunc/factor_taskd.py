#!/usr/bin/env python3
"""factor_taskd — 因子计算任务守护进程（FastAPI）

长期运行的后台服务，管理因子计算任务的提交、执行、取消和监控。
通过 SQLite 持久化任务状态，通过 subprocess 管理子进程生命周期。

REST API:
  POST   /api/tasks                    — 提交任务（带 request_id 时同键重试原子返回同一任务）
  GET    /api/tasks                    — 列出所有任务
  GET    /api/tasks/{id}               — 查看任务详情
  POST   /api/tasks/{id}/cancel        — 取消任务
  POST   /api/tasks/{id}/adjust-njobs  — 调节并行数
  GET    /api/tasks/{id}/log           — 获取任务日志
  DELETE /api/tasks/{id}               — 删除记录（不停止运行）

任务记录里记的东西（执行后可见）：
  script_sha256  提交那一刻脚本内容的 sha256；执行前会重算，不一致就拒绝执行
  run_env        真实解释器、工作目录、worker 二进制指纹、脚本哈希、传给脚本的日期
  dates_honored  起止日期是否真的被脚本读了（脚本不读就是 false，不谎报「已按该范围计算」）
  request_id     提交方给的请求唯一键，同键重试返回同一任务
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import sqlite3
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ── 配置 ──────────────────────────────────────────────
DEFAULT_DB_PATH = os.path.expanduser("~/.factor_taskd/tasks.db")
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 9099
DEFAULT_NJOBS = 50
WATCH_DIR = "/home/chenzongwei/pythoncode"

app = FastAPI(title="factor_taskd", version="1.0.0")

# ── SQLite 初始化 ─────────────────────────────────────
_db_path: str = DEFAULT_DB_PATH
_db_lock = threading.Lock()


def get_db() -> sqlite3.Connection:
    db = sqlite3.connect(_db_path)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")
    return db


def init_db(db_path: str) -> None:
    global _db_path
    _db_path = db_path
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    db = get_db()
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            script_path TEXT NOT NULL,
            n_jobs INTEGER DEFAULT 50,
            start_date INTEGER,
            end_date INTEGER,
            status TEXT DEFAULT 'pending',
            pid INTEGER,
            version TEXT,
            backup_prefix TEXT,
            log_path TEXT,
            created_at TEXT,
            started_at TEXT,
            finished_at TEXT,
            error TEXT,
            progress_snapshot TEXT
        )
    """
    )
    db.commit()
    # 兼容旧数据库：添加 progress_snapshot 列（若已存在则忽略）
    try:
        db.execute("ALTER TABLE tasks ADD COLUMN progress_snapshot TEXT")
        db.commit()
    except sqlite3.OperationalError:
        pass
    # 兼容旧数据库：添加 priority 列（0=最高，默认2；调度器按 优先级+id 排队）
    try:
        db.execute("ALTER TABLE tasks ADD COLUMN priority INTEGER DEFAULT 2")
        db.commit()
    except sqlite3.OperationalError:
        pass
    # 兼容旧数据库：提交方请求唯一键（同键重试原子返回同一任务）
    try:
        db.execute("ALTER TABLE tasks ADD COLUMN request_id TEXT")
        db.commit()
    except sqlite3.OperationalError:
        pass
    # 兼容旧数据库：提交那一刻脚本内容的 sha256（执行前重算，不一致拒绝执行）
    try:
        db.execute("ALTER TABLE tasks ADD COLUMN script_sha256 TEXT")
        db.commit()
    except sqlite3.OperationalError:
        pass
    # 兼容旧数据库：执行环境登记（解释器 / 工作目录 / worker 指纹 / 脚本哈希 / 日期投递）
    try:
        db.execute("ALTER TABLE tasks ADD COLUMN run_env TEXT")
        db.commit()
    except sqlite3.OperationalError:
        pass
    # 兼容旧数据库：本任务用的控制文件路径
    try:
        db.execute("ALTER TABLE tasks ADD COLUMN control_path TEXT")
        db.commit()
    except sqlite3.OperationalError:
        pass
    # 请求唯一键唯一索引：并发同键提交由 SQLite 兜底，只可能存在一行。
    # 用部分索引（request_id IS NOT NULL），旧行与不带键的提交不受影响。
    db.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_tasks_request_id"
        " ON tasks(request_id) WHERE request_id IS NOT NULL"
    )
    db.commit()
    db.close()


# ── Pydantic 模型 ────────────────────────────────────
class TaskSubmit(BaseModel):
    script_path: str
    n_jobs: int = DEFAULT_NJOBS
    start_date: int | None = None
    end_date: int | None = None
    priority: int = 2  # 0=最高；调度器按 priority ASC, id ASC 排队
    request_id: str | None = None  # 提交方生成的请求唯一键；同键重试返回同一任务
    script_sha256: str | None = None  # 提交方算的脚本 sha256；与守护进程算的不一致就拒绝


class NjobsAdjust(BaseModel):
    n_jobs: int


# ── 子进程管理 ───────────────────────────────────────
_active_processes: dict[int, subprocess.Popen] = {}
_active_pids_lock = threading.Lock()


# ── 任务名中文化校验 ─────────────────────────────────
# 任务名 = 脚本文件名去掉扩展名。规范：主体必须为中文（可带英文/数字后缀）。
# 规则：中文字符数 >= 1，且 >= 英文/数字字符数的一半（即中文至少占文字内容 1/3），
# 如「挂单猫0701」「交友软件_v2」合规；「run_urgency_v2_baseline」不含中文，拒绝。
# 下划线/连字符/空格/括号等分隔符不计数（允许但不算主体）。
# 例外：hm 系列因子约定命名，四种形式同样合规：
#   ① hm+数字        （如 hm100、hm89、hm134）
#   ② hm+数字+_ind   （如 hm100_ind、hm134_ind、hm89_ind）
#   ③ hm+数字+_ssm   （如 hm100_ssm，双边单调度 SSM 版本）
#   ④ hm+数字+_ssm+序号（如 hm100_ssm2，SSM 选法的第 2 版）
_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_ASCII_WORD_RE = re.compile(r"[A-Za-z0-9]")
_HM_NAME_RE = re.compile(r"^hm\d+(_ind|_ssm\d*)?$", re.IGNORECASE)


def validate_chinese_name(name: str) -> str | None:
    """校验任务名以中文为主体（例外：hm+数字 / hm+数字_ind / hm+数字_ssm[序号]）。返回 None=合规，否则返回不合规原因。"""
    if _HM_NAME_RE.match(name):
        return None
    cjk = len(_CJK_RE.findall(name))
    ascii_word = len(_ASCII_WORD_RE.findall(name))
    if cjk < 1:
        return f"名称「{name}」不含任何中文字符"
    if cjk * 2 < ascii_word:
        return (
            f"名称「{name}」中文 {cjk} 个 < 英文/数字 {ascii_word} 个的一半，"
            "主体不是中文"
        )
    return None


def _extract_version(script_path: str) -> str:
    """从脚本文件名提取版本名"""
    name = os.path.splitext(os.path.basename(script_path))[0]
    return name


def _build_backup_prefix(version: str) -> str:
    """构建 backup 文件前缀"""
    return f"backup_{version}.bin"


def _find_first_backup_file(version: str) -> str | None:
    """在当前目录或脚本目录查找 backup 文件"""
    pattern = f"backup_{version}.bin"
    for d in (os.getcwd(), WATCH_DIR, os.path.dirname(WATCH_DIR)):
        candidate = os.path.join(d, pattern)
        if os.path.exists(candidate):
            return candidate
        # 也检查 .progress.jsonl 和 .control
        for ext in (".progress.jsonl", ".control"):
            path = candidate + ext
            if os.path.exists(path):
                return candidate
    return f"./{pattern}"


def _update_task(task_id: int, **fields) -> None:
    """UPDATE tasks 的薄封装。列名由调用方在代码里写死，不来自外部输入。"""
    if not fields:
        return
    assignments = ", ".join(f"{name}=?" for name in fields)
    with _db_lock:
        db = get_db()
        db.execute(f"UPDATE tasks SET {assignments} WHERE id=?", (*fields.values(), task_id))
        db.commit()
        db.close()


def _fail_task(task_id: int, error: str, run_env: dict | None = None) -> None:
    """没跑起来就算失败：写清原因，并把已经拿到的环境信息登记下来。"""
    fields: dict = {
        "status": "failed",
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "error": error,
    }
    if run_env is not None:
        fields["run_env"] = json.dumps(run_env, ensure_ascii=False)
    _update_task(task_id, **fields)


# ── 执行环境登记与脚本指纹 ───────────────────────────


def _sha256_file(path: str) -> str | None:
    """文件的 sha256（十六进制）；读不到返回 None。"""
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


def _file_fingerprint(path: str | None) -> dict:
    """文件指纹：路径、大小、修改时间、sha256。文件不在时 exists=False，不抛错。"""
    facts: dict = {"path": path, "exists": False, "bytes": None, "mtime": None, "sha256": None}
    if not path:
        return facts
    try:
        info = os.stat(path)
    except OSError:
        return facts
    facts["exists"] = True
    facts["bytes"] = info.st_size
    facts["mtime"] = datetime.fromtimestamp(info.st_mtime).astimezone().isoformat()
    facts["sha256"] = _sha256_file(path)
    return facts


def _probe_rust_runtime(script_dir: str) -> dict:
    """问目标解释器：rust_pyfunc 装在哪、worker 二进制解析到哪个文件。

    脚本是用 [sys.executable, script] 起的，解释器只有一个答案；
    worker 由 rust_pyfunc 自己解析（安装位置未必是本文件所在目录），所以在子进程里问一次，
    拿不到就退回本文件旁边的部署位置。
    """
    code = (
        "import json\n"
        "info = {}\n"
        "try:\n"
        "    import rust_pyfunc as rp\n"
        "    info['rustPyfuncFile'] = getattr(rp, '__file__', None)\n"
        "    locate = getattr(rp, 'locate_worker_binary', None)\n"
        "    info['worker'] = str(locate()) if callable(locate) else None\n"
        "except Exception as exc:\n"
        "    info['error'] = f'{type(exc).__name__}: {exc}'\n"
        "print(json.dumps(info))\n"
    )
    fallback_worker = str(Path(__file__).resolve().with_name("rust_pyfunc_worker"))
    probed: dict = {"error": "rust_pyfunc 探测没跑起来"}
    try:
        out = subprocess.run(
            [sys.executable, "-c", code],
            cwd=script_dir,
            capture_output=True,
            text=True,
            timeout=60,
        )
        probed = json.loads(out.stdout.strip().splitlines()[-1])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        probed = {"error": "探测 rust_pyfunc 失败（超时或输出不可解析）"}
    return {
        "python": sys.executable,
        "pythonVersion": sys.version.split()[0],
        "rustPyfuncFile": probed.get("rustPyfuncFile"),
        "workerPath": probed.get("worker") or fallback_worker,
        "workerProbeError": probed.get("error"),
    }


def _script_reads_env_dates(script_path: str) -> dict:
    """脚本源码里有没有读 FACTOR_START_DATE / FACTOR_END_DATE。"""
    try:
        with open(script_path, encoding="utf-8", errors="replace") as f:
            source = f.read()
    except OSError:
        return {"start": None, "end": None}
    return {"start": "FACTOR_START_DATE" in source, "end": "FACTOR_END_DATE" in source}


def _build_run_env(
    script_path: str,
    script_sha256: str | None,
    n_jobs: int,
    start_date: int | None,
    end_date: int | None,
) -> dict:
    """执行环境登记：真实解释器、工作目录、worker 指纹、脚本哈希、日期投递情况。"""
    script_dir = os.path.dirname(os.path.abspath(script_path))
    runtime = _probe_rust_runtime(script_dir)
    reads = _script_reads_env_dates(script_path)
    delivered: dict = {}
    read_flags: dict = {}
    if start_date is not None:
        delivered["FACTOR_START_DATE"] = str(start_date)
        read_flags["FACTOR_START_DATE"] = reads["start"]
    if end_date is not None:
        delivered["FACTOR_END_DATE"] = str(end_date)
        read_flags["FACTOR_END_DATE"] = reads["end"]
    if not delivered:
        honored: bool | None = None
        note = "本次没有传起止日期，脚本按自己的默认范围跑"
    elif all(read_flags.values()):
        honored = True
        note = "脚本里读了 FACTOR_START_DATE / FACTOR_END_DATE，区间按投递值生效"
    else:
        honored = False
        missed = "、".join(name for name, ok in read_flags.items() if not ok)
        note = (
            f"脚本里没有出现 {missed}，这两个环境变量传下去了但脚本不读，"
            "不能认为它按这个区间计算（跑的多半是脚本自己的默认范围）"
        )
    return {
        "recordedAt": datetime.now(timezone.utc).isoformat(),
        "interpreter": runtime["python"],
        "interpreterVersion": runtime["pythonVersion"],
        "cwd": script_dir,
        "rustPyfuncFile": runtime["rustPyfuncFile"],
        "rustPyfuncWorker": _file_fingerprint(runtime["workerPath"]),
        "rustPyfuncProbeError": runtime["workerProbeError"],
        "scriptPath": os.path.abspath(script_path),
        "scriptSha256AtSubmit": script_sha256,
        "envDelivered": {"FACTOR_N_JOBS": str(n_jobs), **delivered},
        "dates": {
            "requested": {"startDate": start_date, "endDate": end_date},
            "delivered": delivered or None,
            "scriptReads": read_flags or None,
            "honored": honored,
            "note": note,
        },
    }


def _run_task(
    task_id: int,
    script_path: str,
    n_jobs: int,
    start_date: int | None,
    end_date: int | None,
    script_sha256: str | None = None,
    backup_prefix: str | None = None,
) -> None:
    """在后台线程中执行任务。

    执行前的两道闸：
    ① 脚本指纹：提交时算的 sha256 与此刻文件内容不一致（排队期间被改过）就不执行，
       状态标 failed 并把两个哈希写进任务记录，绝不静默按新内容跑。
    ② 日期：起止日期用 FACTOR_START_DATE / FACTOR_END_DATE 传给脚本，同时静态检查脚本
       有没有读这两个变量，结论记在 run_env.dates.honored（脚本不读就是 false，不谎报）。
    """
    version = _extract_version(script_path)
    # 日志按任务 id 隔离：同名脚本重跑各写各的日志，不再互相覆盖（旧格式 <version>.log 不动）
    log_path = os.path.expanduser(f"~/.factor_taskd/logs/{task_id}_{version}.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    backup_prefix = backup_prefix or _build_backup_prefix(version)
    # 用 Popen(cwd=...) 切目录，禁止 os.chdir —— 会污染 daemon 全局 cwd
    script_dir = os.path.dirname(os.path.abspath(script_path))

    script_sha256_now = _sha256_file(script_path)

    # ① 脚本冻结闸：文件没了、或内容与提交时不一致 → 拒绝执行
    if not os.path.isfile(script_path):
        _fail_task(
            task_id,
            f"执行前脚本文件不存在：{script_path}",
            {
                "recordedAt": datetime.now(timezone.utc).isoformat(),
                "interpreter": sys.executable,
                "interpreterVersion": sys.version.split()[0],
                "cwd": script_dir,
                "scriptPath": os.path.abspath(script_path),
                "scriptSha256AtSubmit": script_sha256,
                "scriptSha256AtStart": None,
                "refused": True,
            },
        )
        return
    if script_sha256 is not None and script_sha256_now != script_sha256:
        _fail_task(
            task_id,
            "脚本内容在提交后被修改，拒绝执行（不会静默按新内容跑）："
            f"提交时 sha256={script_sha256}，执行前 sha256={script_sha256_now}。"
            "要跑新内容请重新提交，并在任务跑完前不要再改这个文件。",
            {
                "recordedAt": datetime.now(timezone.utc).isoformat(),
                "interpreter": sys.executable,
                "interpreterVersion": sys.version.split()[0],
                "cwd": script_dir,
                "scriptPath": os.path.abspath(script_path),
                "scriptSha256AtSubmit": script_sha256,
                "scriptSha256AtStart": script_sha256_now,
                "refused": True,
            },
        )
        return

    cmd = [sys.executable, script_path]
    env = os.environ.copy()
    # 通过环境变量传递 n_jobs，脚本读取后覆盖硬编码值
    env["FACTOR_N_JOBS"] = str(n_jobs)
    # 起止日期同样通过环境变量传下去；脚本读不读由它自己决定，读没读记进 run_env
    if start_date is not None:
        env["FACTOR_START_DATE"] = str(start_date)
    if end_date is not None:
        env["FACTOR_END_DATE"] = str(end_date)

    run_env = _build_run_env(script_path, script_sha256, n_jobs, start_date, end_date)
    run_env["scriptSha256AtStart"] = script_sha256_now

    # 标记为 running（同时登记这次真正用的日志、backup 前缀与执行环境）
    _update_task(
        task_id,
        status="running",
        pid=0,
        started_at=datetime.now(timezone.utc).isoformat(),
        log_path=log_path,
        backup_prefix=backup_prefix,
        run_env=json.dumps(run_env, ensure_ascii=False),
    )

    try:
        # 日志文件只属于本任务，用 O_APPEND 追加（不再 O_TRUNC）：多进程/多 worker 并发写
        # 同一日志时按追加原子写入，避免共享 file offset 互相覆盖、截断多字节 UTF-8。
        log_fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            process = subprocess.Popen(
                cmd,
                stdout=log_fd,
                stderr=subprocess.STDOUT,
                env=env,
                cwd=script_dir,
                preexec_fn=os.setsid if hasattr(os, "setsid") else None,
            )
        finally:
            os.close(log_fd)

        # 注册进程
        with _active_pids_lock:
            _active_processes[task_id] = process

        # 更新 PID
        _update_task(task_id, pid=process.pid)

        process.wait()
        returncode = process.returncode

        # 检查是否被取消
        with _db_lock:
            db = get_db()
            row = db.execute("SELECT status FROM tasks WHERE id=?", (task_id,)).fetchone()
            db.close()
        if row and row["status"] == "cancelled":
            return

        # 标记完成，并快照进度
        new_status = "completed" if returncode == 0 else "failed"
        error_msg = None if returncode == 0 else f"进程退出码: {returncode}"
        progress_snapshot = _read_progress(backup_prefix, task_id, script_dir)
        _update_task(
            task_id,
            status=new_status,
            finished_at=datetime.now(timezone.utc).isoformat(),
            error=error_msg,
            progress_snapshot=json.dumps(progress_snapshot) if progress_snapshot else None,
        )

    except Exception as e:
        progress_snapshot = _read_progress(backup_prefix, task_id, script_dir)
        _update_task(
            task_id,
            status="failed",
            finished_at=datetime.now(timezone.utc).isoformat(),
            error=str(e),
            progress_snapshot=json.dumps(progress_snapshot) if progress_snapshot else None,
        )
    finally:
        with _active_pids_lock:
            _active_processes.pop(task_id, None)


# ── 任务队列与并发调度 ─────────────────────────────────
# _max_concurrent：全局同时运行的【任务】数上限（每个任务内部还有自己的 n_jobs 并行度）。
# None/<=0 = 不限制（默认，行为与旧版一致：提交即运行）。
# 持久化在 ~/.factor_taskd/config.json，daemon 重启后依然生效。
CONFIG_PATH = os.path.expanduser("~/.factor_taskd/config.json")
_max_concurrent: int | None = None


def _load_config() -> None:
    global _max_concurrent
    try:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            cfg = json.load(f)
        _max_concurrent = cfg.get("max_concurrent") or None
    except OSError:
        _max_concurrent = None


def _save_config() -> None:
    try:
        os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump({"max_concurrent": _max_concurrent}, f)
    except OSError:
        pass


def _queue_state() -> dict:
    """当前队列状态：运行中/排队中数量 + 全局并发上限。"""
    with _db_lock:
        db = get_db()
        running = db.execute(
            "SELECT COUNT(*) FROM tasks WHERE status='running'"
        ).fetchone()[0]
        pending = db.execute(
            "SELECT COUNT(*) FROM tasks WHERE status='pending'"
        ).fetchone()[0]
        db.close()
    return {
        "max_concurrent": _max_concurrent,
        "running": running,
        "pending": pending,
    }


def _scheduler_tick() -> None:
    """调度一轮：空闲配额内，按 优先级(高→低) + 提交先后 启动排队任务。

    启动前先原子占用（UPDATE ... WHERE status='pending' 且 rowcount==1 才启动），
    避免 tick 重入/并发时同一任务被重复启动；取消竞态也由该条件挡住。
    """
    with _db_lock:
        db = get_db()
        pending = db.execute(
            "SELECT id, script_path, n_jobs, start_date, end_date, script_sha256, backup_prefix"
            " FROM tasks WHERE status='pending' ORDER BY priority ASC, id ASC"
        ).fetchall()
        running_n = db.execute(
            "SELECT COUNT(*) FROM tasks WHERE status='running'"
        ).fetchone()[0]
        db.close()
    limit = _max_concurrent
    slots = len(pending) if (limit is None or limit <= 0) else max(0, limit - running_n)
    for row in pending[:slots]:
        with _db_lock:
            db = get_db()
            cur = db.execute(
                "UPDATE tasks SET status='running' WHERE id=? AND status='pending'",
                (row["id"],),
            )
            db.commit()
            db.close()
        if cur.rowcount != 1:
            continue  # 已被取消/他人占用，跳过
        threading.Thread(
            target=_run_task,
            args=(
                row["id"],
                row["script_path"],
                row["n_jobs"],
                row["start_date"],
                row["end_date"],
                row["script_sha256"],
                row["backup_prefix"],
            ),
            daemon=True,
        ).start()


# 周期性对账的间隔与最小年龄门槛（见 _scheduler_loop 里的说明）。
# 最小年龄要明显大于 launch_task「先写 running/pid=0、再 Popen、再填 pid」这段窗口。
_RECONCILE_INTERVAL_SEC = 30.0
_RECONCILE_MIN_AGE_SEC = 300.0


def _scheduler_loop() -> None:
    """常驻调度线程：每 1 秒补一轮，覆盖 daemon 重启后的 pending 恢复
    与运行中任务退出后的补位（运行中的任务退出 → 下一 tick 空出配额 → 排队任务顶上）。"""
    last_reconcile = 0.0
    while True:
        try:
            # 周期性对账：启动时那次只能处理「重启那一刻进程就已经没了」的行。
            # 上一个 daemon 启动、重启时进程还活着、之后才退出的任务（本进程没有它的
            # Popen 句柄，wait() 不会被触发）只能靠这里补上，否则那行永远停在 running。
            now = time.time()
            if now - last_reconcile >= _RECONCILE_INTERVAL_SEC:
                last_reconcile = now
                _reconcile_running_tasks(min_age_sec=_RECONCILE_MIN_AGE_SEC)
            _scheduler_tick()
        except Exception:
            pass
        time.sleep(1)


# ── 任务记录读取（列表 / 详情 / 提交回执共用） ────────


def _decorate_task(task: dict, include_progress: bool = True) -> dict:
    """把一行任务补齐成 API 返回的样子：进度、进程是否活着、run_env、日期是否生效。

    必须在 _db_lock 之外调用：_is_process_active 在内存表未命中时会去取 _db_lock，
    而 _db_lock 不可重入，持锁调用会自死锁（见 list_tasks 的说明）。
    """
    if include_progress and task.get("backup_prefix"):
        search_dir = os.path.dirname(task["script_path"])
        if task.get("status") in ("cancelled", "completed", "failed") and task.get(
            "progress_snapshot"
        ):
            try:
                task["progress"] = json.loads(task["progress_snapshot"])
            except json.JSONDecodeError:
                task["progress"] = None
        else:
            task["progress"] = _read_progress(
                task["backup_prefix"], task["id"], search_dir
            )
    raw_env = task.get("run_env")
    run_env = None
    if isinstance(raw_env, str) and raw_env:
        try:
            run_env = json.loads(raw_env)
        except json.JSONDecodeError:
            run_env = {"parseError": raw_env[:200]}
    task["run_env"] = run_env
    dates = run_env.get("dates") if isinstance(run_env, dict) else None
    honored = dates.get("honored") if isinstance(dates, dict) else None
    # 起止日期是否真的被脚本读了：没传=null，传了且脚本不读=false（不谎报已按该范围计算）
    task["dates_honored"] = honored if isinstance(honored, bool) else None
    task["active"] = _is_process_active(task["id"])
    task["name_valid"] = validate_chinese_name(task.get("name", "")) is None
    return task


def _task_row(task_id: int) -> dict | None:
    with _db_lock:
        db = get_db()
        row = db.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
        db.close()
    return dict(row) if row else None


def _task_by_request_id(request_id: str) -> dict | None:
    with _db_lock:
        db = get_db()
        row = db.execute(
            "SELECT * FROM tasks WHERE request_id=?", (request_id,)
        ).fetchone()
        db.close()
    return dict(row) if row else None


def _controller_path(script_dir: str, backup_prefix: str) -> str:
    """脚本侧控制文件路径。

    这个路径由 Rust 引擎按 <backup_file>.control 写死（见 src/parallel_computing*.rs），
    backup_file 由脚本自己传给引擎，守护进程只能照同一个约定拼。改成按任务 id 命名会让
    「运行中调并行数」彻底失效，所以这里保持引擎约定；同名脚本并发时由 adjust_njobs
    拒绝调整（两个任务共用一个控制文件，改了会互相影响）。
    """
    return os.path.join(script_dir, f"{backup_prefix}.control")


def _prefix_of(row: dict) -> str:
    return row.get("backup_prefix") or _build_backup_prefix(
        _extract_version(row["script_path"])
    )


def _other_active_tasks_with_prefix(backup_prefix: str, exclude: int) -> list[dict]:
    """还活着（排队/运行中）且共用同一个 backup 前缀的其它任务。"""
    with _db_lock:
        db = get_db()
        rows = db.execute(
            "SELECT id, name, script_path, backup_prefix, status FROM tasks"
            " WHERE status IN ('pending', 'running') AND id != ?",
            (exclude,),
        ).fetchall()
        db.close()
    return [
        {"id": row["id"], "name": row["name"], "status": row["status"]}
        for row in rows
        if _prefix_of(dict(row)) == backup_prefix
    ]


# ── API 端点 ─────────────────────────────────────────


@app.post("/api/tasks")
def submit_task(body: TaskSubmit):
    """提交新的因子计算任务。

    request_id 是提交方生成的请求唯一键：同键重试（并发提交、网络重试、响应丢失后重发）
    原子返回第一次建的那个任务，不会重复建。它跟插件侧「同内容 30 分钟内重复提交提醒」
    是两件事：前者是同一次提交动作的重试，后者是两次不同的提交动作。
    """
    request_id = (body.request_id or "").strip() or None
    if request_id:
        existing = _task_by_request_id(request_id)
        if existing is not None:
            if os.path.abspath(body.script_path) != existing["script_path"]:
                raise HTTPException(
                    409,
                    f"request_id「{request_id}」已经用在另一个脚本上（任务 "
                    f"{existing['id']}：{existing['script_path']}），拒绝混用；"
                    "重试请用原脚本，想提交别的脚本请换一个 request_id",
                )
            _scheduler_tick()
            return {**_decorate_task(existing), "deduplicated": True}

    script_path = os.path.abspath(body.script_path)
    if not os.path.isfile(script_path):
        raise HTTPException(400, f"脚本文件不存在: {script_path}")

    version = _extract_version(script_path)

    # 强制检查：任务名（= 脚本文件名）主体必须为中文（可含数字/英文），
    # 例外：hm 系列因子约定命名——hm+数字（如 hm100）、hm+数字+_ind（如 hm100_ind）。
    # 这是任务管理系统的硬性规范，CLI/Web 前端只是友好提示，此处才是权威拦截。
    name_err = validate_chinese_name(version)
    if name_err:
        raise HTTPException(
            400,
            f"任务名称不合规：{name_err}。任务名称必须以中文为主体"
            "（可含数字/英文），请将脚本改名为如「交友软件_v2.py」"
            "「网络社交因子.py」后再提交；hm+数字（如 hm100）"
            "或 hm+数字_ind（如 hm100_ind）亦可",
        )

    # 提交前语法预检：语法错误在任务系统里只会让任务秒失败并污染历史记录，
    # 先用 compile() 内存编译拦截（不写 __pycache__，无文件系统权限问题），
    # 错误立即反馈给提交者。
    try:
        with open(script_path, encoding="utf-8", errors="replace") as f:
            compile(f.read(), script_path, "exec")
    except (SyntaxError, ValueError) as e:
        raise HTTPException(400, f"脚本语法错误，拒绝提交：\n{e}")

    # 提交那一刻的脚本指纹：执行前会重算，不一致就拒绝执行（见 _run_task）
    script_sha256 = _sha256_file(script_path)
    if script_sha256 is None:
        raise HTTPException(400, f"脚本读不出来，算不出 sha256：{script_path}")
    if body.script_sha256 and body.script_sha256 != script_sha256:
        raise HTTPException(
            400,
            "脚本内容与提交方给的指纹不一致（提交过程中文件被改过）：提交方 "
            f"{body.script_sha256}，守护进程 {script_sha256}。请确认文件稳定后重新提交",
        )

    now = datetime.now(timezone.utc).isoformat()

    task_id: int | None = None
    with _db_lock:
        db = get_db()
        try:
            cursor = db.execute(
                """INSERT INTO tasks
                   (name, script_path, n_jobs, start_date, end_date,
                    status, priority, version, created_at, request_id, script_sha256)
                   VALUES (?, ?, ?, ?, ?, 'pending', ?, ?, ?, ?, ?)""",
                (
                    version,
                    script_path,
                    body.n_jobs,
                    body.start_date,
                    body.end_date,
                    body.priority,
                    version,
                    now,
                    request_id,
                    script_sha256,
                ),
            )
            task_id = cursor.lastrowid
            db.commit()
        except sqlite3.IntegrityError:
            # 并发同键：唯一索引挡住了第二条，回头取第一条
            db.rollback()
            task_id = None
        db.close()

    if task_id is None:
        existing = _task_by_request_id(request_id) if request_id else None
        if existing is None:
            raise HTTPException(500, "任务写库失败（request_id 冲突但找不到已有任务）")
        _scheduler_tick()
        return {**_decorate_task(existing), "deduplicated": True}

    # 不自行启动线程：入队后交给调度器（遵守全局并发上限与优先级）。
    # 无上限时调度器会立即启动，行为与旧版一致。
    _scheduler_tick()

    row = _task_row(task_id)
    return {**_decorate_task(row if row else {"id": task_id, "name": version, "status": "pending"}), "deduplicated": False}


@app.get("/api/tasks")
def list_tasks():
    """列出所有任务，含进度信息。

    进度解析(_read_progress)与进程探活(_is_process_active)必须在 _db_lock
    临界区外执行：_is_process_active 在 _active_processes 未命中时会回退
    获取 _db_lock，而 _db_lock 是非重入 threading.Lock——在持锁期间再次获取
    会自死锁，该线程永久持有 _db_lock，所有读请求随之挂死、threadpool worker
    逐个被耗尽。早期把整个循环放在锁内导致了此故障。
    """
    with _db_lock:
        db = get_db()
        rows = db.execute(
            "SELECT * FROM tasks ORDER BY created_at DESC LIMIT 100"
        ).fetchall()
        db.close()
    tasks = []
    for row in rows:
        # 统一走 _decorate_task：进度、探活、run_env 解析、日期是否生效、名称是否合规
        tasks.append(_decorate_task(dict(row)))
    return tasks


@app.get("/api/tasks/{task_id}")
def get_task(task_id: int):
    with _db_lock:
        db = get_db()
        row = db.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
        db.close()
    if not row:
        raise HTTPException(404, "任务不存在")
    return _decorate_task(dict(row))


# 代码接口上限：脚本一般 <1MB；超大文件截断返回并标记 truncated
MAX_CODE_BYTES = 5 * 1024 * 1024


@app.get("/api/tasks/{task_id}/code")
def get_task_code(task_id: int):
    """获取任务脚本的完整代码（供前端「点击任务名 → 代码弹窗」使用）"""
    with _db_lock:
        db = get_db()
        row = db.execute(
            "SELECT name, script_path FROM tasks WHERE id=?", (task_id,)
        ).fetchone()
        db.close()
    if not row:
        raise HTTPException(404, "任务不存在")
    path = row["script_path"]
    if not path or not os.path.isfile(path):
        return {
            "name": row["name"],
            "script_path": path or "",
            "content": "",
            "error": "脚本文件不存在或已被删除",
        }
    size = os.path.getsize(path)
    truncated = size > MAX_CODE_BYTES
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read(MAX_CODE_BYTES)
    return {
        "name": row["name"],
        "script_path": path,
        "content": content,
        "size": size,
        "lines": content.count("\n") + 1,
        "mtime": datetime.fromtimestamp(os.path.getmtime(path))
        .astimezone()
        .isoformat(),
        "truncated": truncated,
    }


@app.post("/api/tasks/{task_id}/cancel")
def cancel_task(task_id: int):
    """取消运行中的任务"""
    with _db_lock:
        db = get_db()
        row = db.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
        if not row:
            db.close()
            raise HTTPException(404, "任务不存在")

        # 快照当前进度后标记为取消
        progress = None
        if row["status"] == "pending":
            # 排队中（尚未启动子进程）：直接改状态即可，无进程可杀
            db.execute(
                "UPDATE tasks SET status='cancelled', finished_at=? WHERE id=?",
                (datetime.now(timezone.utc).isoformat(), task_id),
            )
            db.commit()
            db.close()
            return {"status": "cancelled"}
        if row["backup_prefix"]:
            progress = _read_progress(row["backup_prefix"], task_id)
        db.execute(
            "UPDATE tasks SET status='cancelled', finished_at=?, progress_snapshot=? WHERE id=?",
            (
                datetime.now(timezone.utc).isoformat(),
                json.dumps(progress) if progress else None,
                task_id,
            ),
        )
        db.commit()
        db.close()

    # 终止进程
    pid_from_db = row["pid"] if row and row["pid"] else 0
    script_path = row["script_path"] if row else ""
    with _active_pids_lock:
        proc = _active_processes.get(task_id)
        if proc:
            try:
                pgid = os.getpgid(proc.pid)
                os.killpg(pgid, signal.SIGTERM)
                for _ in range(30):
                    if proc.poll() is not None:
                        break
                    time.sleep(0.1)
                if proc.poll() is None:
                    os.killpg(pgid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        elif pid_from_db > 0:
            # 防 PID 复用误杀：daemon 重启过之后 DB 里的 pid 可能已被系统上其他进程复用，
            # 此时 killpg 会误杀无辜进程（本类服务历史上被信号杀掉的潜在元凶之一）。
            # kill 前先核对 /proc/<pid>/cmdline 确实是本任务的脚本；不是则不动它。
            if script_path and not _pid_belongs_to_script(pid_from_db, script_path):
                return {"status": "cancelled", "warn": "taskd 重启后 pid 已被复用，未发送信号"}
            try:
                pgid = os.getpgid(pid_from_db)
                os.killpg(pgid, signal.SIGTERM)
                for _ in range(30):
                    try:
                        os.kill(pid_from_db, 0)
                    except ProcessLookupError:
                        break
                    time.sleep(0.1)
                else:
                    os.killpg(pgid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError, OSError):
                pass

    return {"status": "cancelled"}


@app.post("/api/tasks/{task_id}/adjust-njobs")
def adjust_njobs(task_id: int, body: NjobsAdjust):
    """调节运行中任务的并行数。

    控制文件路径由 Rust 引擎固定成 <backup_file>.control（backup_file 是脚本自己传的），
    守护进程改不了。于是同名脚本同时跑两个任务时，两个任务共用一个控制文件——这时调整
    会同时改变另一个任务的并行数，属于静默串台，直接拒绝（409）而不是装作没发生。
    """
    if body.n_jobs < 1:
        raise HTTPException(400, "n_jobs 必须大于 0")

    with _db_lock:
        db = get_db()
        row = db.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
        db.close()

    if not row:
        raise HTTPException(404, "任务不存在")

    if row["status"] == "pending":
        # 排队中尚未启动：只更新 DB 值，调度器启动时自然生效
        _update_task(task_id, n_jobs=body.n_jobs)
        return {"status": "adjusted", "n_jobs": body.n_jobs, "queued": True}

    backup_prefix = _prefix_of(dict(row))
    script_dir = os.path.dirname(os.path.abspath(row["script_path"]))
    control_path = _controller_path(script_dir, backup_prefix)

    others = _other_active_tasks_with_prefix(backup_prefix, exclude=task_id)
    if others:
        names = "、".join(f"任务 {o['id']}（{o['name']}，{o['status']}）" for o in others)
        raise HTTPException(
            409,
            f"同名脚本还有 {names} 没结束，它们和本任务共用控制文件 {control_path}"
            "（路径由 Rust 引擎按 <backup>.control 固定），改并行数会同时改到别的任务。"
            "等它们结束再调，或取消后重新提交。",
        )

    control_data = {"n_jobs": body.n_jobs}
    # 原子写：先写临时文件再 replace，脚本不会读到写了一半的 json
    tmp_path = f"{control_path}.tmp{os.getpid()}"
    try:
        with open(tmp_path, "w") as f:
            json.dump(control_data, f)
        os.replace(tmp_path, control_path)
    except OSError as e:
        raise HTTPException(500, f"写入控制文件失败: {e}")

    # 更新数据库中的 n_jobs，并把这次用的控制文件路径记进任务记录
    _update_task(task_id, n_jobs=body.n_jobs, control_path=control_path)

    return {"status": "adjusted", "n_jobs": body.n_jobs, "control_path": control_path}


class ConfigUpdate(BaseModel):
    max_concurrent: int | None = None


@app.get("/api/config")
def get_config():
    """队列与并发配置：全局任务并发上限 + 运行/排队计数"""
    return _queue_state()


@app.post("/api/config")
def set_config(body: ConfigUpdate):
    """运行时调整全局并发任务上限（None/<=0 = 不限制）。持久化，重启后生效。"""
    global _max_concurrent
    _max_concurrent = (
        int(body.max_concurrent) if body.max_concurrent and body.max_concurrent > 0 else None
    )
    _save_config()
    _scheduler_tick()  # 提高/取消上限后立刻补位
    return _queue_state()


class PriorityAdjust(BaseModel):
    priority: int = 2


@app.post("/api/tasks/{task_id}/priority")
def adjust_task_priority(task_id: int, body: PriorityAdjust):
    """调整任务优先级（0=最高）；对排队任务立即重新排序，调度器下一轮补位。"""
    if body.priority not in (0, 1, 2):
        raise HTTPException(400, "priority 必须为 0/1/2（0=最高）")
    with _db_lock:
        db = get_db()
        cur = db.execute(
            "UPDATE tasks SET priority=? WHERE id=?", (body.priority, task_id)
        )
        db.commit()
        db.close()
    if cur.rowcount == 0:
        raise HTTPException(404, "任务不存在")
    _scheduler_tick()
    return {"status": "ok", "priority": body.priority}


def _read_log_file(file_path: str, offset: int, limit: int) -> dict:
    """读取日志文件并分页。

    多进程并发写日志时可能出现截断的多字节 UTF-8（中间字节裸露），
    必须 errors='replace'，否则 readlines() 抛 UnicodeDecodeError → API 500，
    前端代理再误报成「日志服务不可用」。
    """
    if not file_path or not os.path.isfile(file_path):
        return {"log": "", "total_lines": 0}
    try:
        with open(file_path, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        total = len(lines)
        end = max(0, total - offset)
        start = max(0, end - limit)
        if start >= end:
            return {"log": "", "total_lines": total}
        return {"log": "".join(lines[start:end]), "total_lines": total}
    except OSError:
        return {"log": "", "total_lines": 0}


@app.get("/api/tasks/{task_id}/log")
def get_task_log(task_id: int, offset: int = 0, limit: int = 1000):
    """获取任务主日志（factor_taskd 子进程的 stdout/stderr）"""
    with _db_lock:
        db = get_db()
        row = db.execute("SELECT log_path FROM tasks WHERE id=?", (task_id,)).fetchone()
        db.close()
    if not row:
        raise HTTPException(404, "任务不存在")
    return _read_log_file(row["log_path"], offset, limit)


@app.get("/api/tasks/{task_id}/progress")
def get_task_progress(task_id: int):
    """解析任务日志最后一行 📊 进度，返回结构化进度数据。"""
    import re

    with _db_lock:
        db = get_db()
        row = db.execute(
            "SELECT log_path, status FROM tasks WHERE id=?", (task_id,)
        ).fetchone()
        db.close()
    if not row:
        raise HTTPException(404, "任务不存在")
    try:
        with open(row["log_path"], encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except OSError:
        return {"status": row["status"], "progress_pct": None}
    # 从末尾找最后一条 📊 进度行
    for line in reversed(lines):
        m = re.search(
            r"进度:\s*(\d+)/(\d+)\s*\(([0-9.]+)%\).*?速度:\s*(\d+)/s.*?已用:\s*(\S+).*?预估剩余\(全局\):\s*(\S+)",
            line,
        )
        if m:
            return {
                "status": row["status"],
                "completed": int(m.group(1)),
                "total": int(m.group(2)),
                "progress_pct": float(m.group(3)),
                "speed": int(m.group(4)),
                "elapsed": m.group(5),
                "eta": m.group(6),
            }
    # 找 Step 标记
    for line in reversed(lines):
        if "Step" in line and ":" in line:
            return {
                "status": row["status"],
                "step": line.strip()[:80],
                "progress_pct": None,
            }
    return {"status": row["status"], "progress_pct": None}


@app.get("/api/tasks/{task_id}/subprocess-log")
def get_subprocess_log(task_id: int, offset: int = 0, limit: int = 1000):
    """获取子进程日志（Rust 计算引擎的 debug_log）"""
    with _db_lock:
        db = get_db()
        row = db.execute(
            "SELECT script_path, backup_prefix FROM tasks WHERE id=?", (task_id,)
        ).fetchone()
        db.close()
    if not row:
        raise HTTPException(404, "任务不存在")
    if not row["backup_prefix"]:
        return {"log": "", "total_lines": 0}
    file_path = os.path.join(
        os.path.dirname(row["script_path"]), f"{row['backup_prefix']}.log"
    )
    return _read_log_file(file_path, offset, limit)


@app.delete("/api/tasks/{task_id}")
def delete_task(task_id: int):
    """删除任务记录（不停止运行）"""
    with _db_lock:
        db = get_db()
        db.execute("DELETE FROM tasks WHERE id=?", (task_id,))
        db.commit()
        db.close()
    return {"status": "deleted"}


# ── 辅助 ─────────────────────────────────────────────


def _read_progress(
    backup_prefix: str, task_id: int, search_dir: str | None = None
) -> dict | None:
    """读取 backup_<ver>.bin.progress.jsonl 的进度信息"""
    # 按优先级搜索：指定目录 > 当前目录 > ./ 前缀
    candidates = []
    if search_dir:
        candidates.append(os.path.join(search_dir, f"{backup_prefix}.progress.jsonl"))
    candidates.append(f"{backup_prefix}.progress.jsonl")
    if not backup_prefix.startswith("./"):
        candidates.append(f"./{backup_prefix}.progress.jsonl")
    progress_path = None
    for p in candidates:
        if os.path.isfile(p):
            progress_path = p
            break
    if progress_path is None:
        return None
    try:
        with open(progress_path) as f:
            lines = f.readlines()
        records = []
        for line in lines:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        if not records:
            return None
        return {
            "records": records,
            "latest": records[-1],
            "is_running": records[-1].get("status") != "completed",
        }
    except OSError:
        return None


def _pid_belongs_to_script(pid: int, script_path: str) -> bool:
    """核对 pid 对应的进程命令行中确实包含任务脚本路径。

    用于 cancel_task 的 DB pid 回退分支：daemon 重启后内存表为空，
    只能凭 DB 里持久化的 pid 找进程；若该 pid 已被操作系统复用于
    其他进程，直接 killpg 会误杀无辜进程。核验不通过即视为不可信。
    """
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            cmdline = f.read().replace(b"\x00", b" ").decode(errors="replace")
        return script_path in cmdline
    except OSError:
        return False


def _task_process_alive(task_id: int, pid: int, script_path: str) -> bool:
    """任务的子进程是否真的还在跑（防 PID 复用）。

    daemon 重启后内存里的 _active_processes 是空的，只能凭 DB 里持久化的 pid
    判断。pid 在重启后可能已被系统分给别的进程，所以不能只 os.kill(pid, 0)：
    先核对 /proc/<pid>/cmdline 确实是本任务的脚本（口径同 cancel_task 的
    「防 PID 复用误杀」），再探活，两者都满足才算活着。
    """
    with _active_pids_lock:
        proc = _active_processes.get(task_id)
    if proc is not None:
        return proc.poll() is None
    if pid <= 0:
        return False
    if not _pid_belongs_to_script(pid, script_path):
        return False
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError, OSError):
        return False


def _reconcile_running_tasks(min_age_sec: float = 0.0) -> list[tuple[int, int]]:
    """对账：把 status='running' 但进程已消失的行标成 failed。

    daemon 重启会丢掉内存里的 _active_processes，而重启恢复只扫 pending，
    于是「进程早就跑完/消失了」的行会永远停在 running。并发上限为 null 时
    只是显示错误；一旦设了 max_concurrent，这行会永久占住一个配额，队列卡死。
    这里按 pid + /proc cmdline 逐行核对，只改真正没有进程的行，并补 finished_at。
    UPDATE 带 status='running' 条件，重复调用不会碰 completed/failed/cancelled 的行。

    min_age_sec：只对「启动至今超过这么多秒」的行做判断。调度线程周期调用时必须给一个
    正值——launch_task 是先写 status='running'/pid=0、再 Popen、最后才把 pid 和内存句柄
    填上，中间有个很短的窗口期，pid=0 且不在内存表里，会被误判成「进程已消失」。
    启动时调用（没有任何任务在启动）用默认 0 即可。

    返回被改掉的行 [(task_id, pid), ...]（进程仍活着的行不动，保持 running）。
    """
    now = datetime.now(timezone.utc)
    with _db_lock:
        db = get_db()
        rows = db.execute(
            "SELECT id, pid, script_path, started_at FROM tasks WHERE status='running'"
        ).fetchall()
        db.close()
    reconciled: list[tuple[int, int]] = []
    for row in rows:
        if min_age_sec > 0 and row["started_at"]:
            try:
                age = (now - datetime.fromisoformat(row["started_at"])).total_seconds()
            except ValueError:
                age = min_age_sec
            if age < min_age_sec:
                continue
        pid = row["pid"] or 0
        # 探活在 _db_lock 外做：与 list_tasks 的注释同理，避免持锁期间再取锁
        if _task_process_alive(row["id"], pid, row["script_path"]):
            continue
        with _db_lock:
            db = get_db()
            cur = db.execute(
                "UPDATE tasks SET status='failed', finished_at=?, error=?"
                " WHERE id=? AND status='running'",
                (
                    datetime.now(timezone.utc).isoformat(),
                    f"守护进程重启后进程已消失（pid={pid}），标记为 failed",
                    row["id"],
                ),
            )
            db.commit()
            db.close()
        if cur.rowcount == 1:
            reconciled.append((row["id"], pid))
    return reconciled


def _is_process_active(task_id: int) -> bool:
    with _active_pids_lock:
        proc = _active_processes.get(task_id)
    if proc is not None:
        return proc.poll() is None
    # daemon 重启后内存表为空，回退查 DB pid 是否仍存活
    with _db_lock:
        db = get_db()
        row = db.execute(
            "SELECT pid, status FROM tasks WHERE id=?", (task_id,)
        ).fetchone()
        db.close()
    if not row or row["status"] != "running":
        return False
    pid = row["pid"] or 0
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError, OSError):
        return False


# ── 入口 ─────────────────────────────────────────────


def main():
    global _max_concurrent
    parser = argparse.ArgumentParser(description="factor_taskd — 因子计算任务守护进程")
    parser.add_argument(
        "--host", default=DEFAULT_HOST, help=f"监听地址 (默认: {DEFAULT_HOST})"
    )
    parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT, help=f"监听端口 (默认: {DEFAULT_PORT})"
    )
    parser.add_argument(
        "--db", default=DEFAULT_DB_PATH, help=f"SQLite 数据库路径 (默认: {DEFAULT_DB_PATH})"
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=None,
        help="全局同时运行的任务数上限（0/缺省=不限制；也可用 /api/config 运行时调整并持久化）",
    )
    parser.add_argument("--reload", action="store_true", help="启用热重载（开发用）")
    args = parser.parse_args()

    _load_config()
    if args.max_concurrent is not None:
        _max_concurrent = args.max_concurrent if args.max_concurrent > 0 else None
        _save_config()

    init_db(args.db)
    # 启动对账：重启前遗留的 running 行若进程已消失，标成 failed。
    # 必须在调度线程启动前做，否则这些行会先按 running 占掉并发配额。
    _reconcile_running_tasks()
    # 启动队列调度线程：重启后 pending 任务自动恢复调度
    threading.Thread(target=_scheduler_loop, daemon=True).start()
    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
