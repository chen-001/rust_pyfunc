#!/usr/bin/env python3
"""factor_taskd — 因子计算任务守护进程（FastAPI）

长期运行的后台服务，管理因子计算任务的提交、执行、取消和监控。
通过 SQLite 持久化任务状态，通过 subprocess 管理子进程生命周期。

REST API:
  POST   /api/tasks                    — 提交任务
  GET    /api/tasks                    — 列出所有任务
  GET    /api/tasks/{id}               — 查看任务详情
  POST   /api/tasks/{id}/cancel        — 取消任务
  POST   /api/tasks/{id}/adjust-njobs  — 调节并行数
  GET    /api/tasks/{id}/log           — 获取任务日志
  DELETE /api/tasks/{id}               — 删除记录（不停止运行）
"""
from __future__ import annotations

import argparse
import json
import os
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
    db.close()


# ── Pydantic 模型 ────────────────────────────────────
class TaskSubmit(BaseModel):
    script_path: str
    n_jobs: int = DEFAULT_NJOBS
    start_date: int | None = None
    end_date: int | None = None


class NjobsAdjust(BaseModel):
    n_jobs: int


# ── 子进程管理 ───────────────────────────────────────
_active_processes: dict[int, subprocess.Popen] = {}
_active_pids_lock = threading.Lock()


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


def _run_task(
    task_id: int,
    script_path: str,
    n_jobs: int,
    start_date: int | None,
    end_date: int | None,
) -> None:
    """在后台线程中执行任务"""
    version = _extract_version(script_path)
    log_path = os.path.expanduser(f"~/.factor_taskd/logs/{version}.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    backup_prefix = _build_backup_prefix(version)

    # 切换到脚本所在目录，确保相对路径（如 ./backup_*.bin）正确解析
    script_dir = os.path.dirname(os.path.abspath(script_path))
    os.chdir(script_dir)

    cmd = [sys.executable, script_path]
    env = os.environ.copy()
    # 通过环境变量传递 n_jobs，脚本读取后覆盖硬编码值
    env["FACTOR_N_JOBS"] = str(n_jobs)

    # 标记为 running
    with _db_lock:
        db = get_db()
        db.execute(
            "UPDATE tasks SET status='running', pid=?, started_at=?, log_path=?, backup_prefix=? WHERE id=?",
            (
                0,
                datetime.now(timezone.utc).isoformat(),
                log_path,
                backup_prefix,
                task_id,
            ),
        )
        db.commit()
        db.close()

    try:
        with open(log_path, "w") as log_f:
            process = subprocess.Popen(
                cmd,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                env=env,
                preexec_fn=os.setsid if hasattr(os, "setsid") else None,
            )

        # 注册进程
        with _active_pids_lock:
            _active_processes[task_id] = process

        # 更新 PID
        with _db_lock:
            db = get_db()
            db.execute("UPDATE tasks SET pid=? WHERE id=?", (process.pid, task_id))
            db.commit()
            db.close()

        process.wait()
        returncode = process.returncode

        # 检查是否被取消
        with _db_lock:
            db = get_db()
            row = db.execute(
                "SELECT status FROM tasks WHERE id=?", (task_id,)
            ).fetchone()
            if row and row["status"] == "cancelled":
                db.close()
                return

        # 标记完成，并快照进度
        new_status = "completed" if returncode == 0 else "failed"
        error_msg = None if returncode == 0 else f"进程退出码: {returncode}"
        progress_snapshot = _read_progress(backup_prefix, task_id)
        with _db_lock:
            db = get_db()
            db.execute(
                "UPDATE tasks SET status=?, finished_at=?, error=?, progress_snapshot=? WHERE id=?",
                (
                    new_status,
                    datetime.now(timezone.utc).isoformat(),
                    error_msg,
                    json.dumps(progress_snapshot) if progress_snapshot else None,
                    task_id,
                ),
            )
            db.commit()
            db.close()

    except Exception as e:
        progress_snapshot = _read_progress(backup_prefix, task_id) if backup_prefix else None
        with _db_lock:
            db = get_db()
            db.execute(
                "UPDATE tasks SET status='failed', finished_at=?, error=?, progress_snapshot=? WHERE id=?",
                (
                    datetime.now(timezone.utc).isoformat(),
                    str(e),
                    json.dumps(progress_snapshot) if progress_snapshot else None,
                    task_id,
                ),
            )
            db.commit()
            db.close()
    finally:
        with _active_pids_lock:
            _active_processes.pop(task_id, None)


# ── API 端点 ─────────────────────────────────────────


@app.post("/api/tasks")
def submit_task(body: TaskSubmit):
    """提交新的因子计算任务"""
    script_path = os.path.abspath(body.script_path)
    if not os.path.isfile(script_path):
        raise HTTPException(400, f"脚本文件不存在: {script_path}")

    version = _extract_version(script_path)
    now = datetime.now(timezone.utc).isoformat()

    with _db_lock:
        db = get_db()
        cursor = db.execute(
            """INSERT INTO tasks
               (name, script_path, n_jobs, start_date, end_date,
                status, version, created_at)
               VALUES (?, ?, ?, ?, ?, 'pending', ?, ?)""",
            (
                version,
                script_path,
                body.n_jobs,
                body.start_date,
                body.end_date,
                version,
                now,
            ),
        )
        task_id = cursor.lastrowid
        db.commit()
        db.close()

    # 启动后台线程执行
    t = threading.Thread(
        target=_run_task,
        args=(task_id, script_path, body.n_jobs, body.start_date, body.end_date),
        daemon=True,
    )
    t.start()

    return {"id": task_id, "status": "pending", "name": version}


@app.get("/api/tasks")
def list_tasks():
    """列出所有任务，含进度信息"""
    tasks = []
    with _db_lock:
        db = get_db()
        rows = db.execute(
            "SELECT * FROM tasks ORDER BY created_at DESC LIMIT 100"
        ).fetchall()
        for row in rows:
            task = dict(row)
            if task.get("backup_prefix"):
                search_dir = os.path.dirname(task["script_path"])
                if task["status"] in ("cancelled", "completed", "failed") and task.get("progress_snapshot"):
                    task["progress"] = json.loads(task["progress_snapshot"])
                else:
                    task["progress"] = _read_progress(task["backup_prefix"], task["id"], search_dir)
            task["active"] = _is_process_active(task["id"])
            tasks.append(task)
        db.close()
    return tasks


@app.get("/api/tasks/{task_id}")
def get_task(task_id: int):
    with _db_lock:
        db = get_db()
        row = db.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
        db.close()
    if not row:
        raise HTTPException(404, "任务不存在")
    task = dict(row)
    if task.get("backup_prefix"):
        search_dir = os.path.dirname(task["script_path"])
        if task["status"] in ("cancelled", "completed", "failed") and task.get("progress_snapshot"):
            task["progress"] = json.loads(task["progress_snapshot"])
        else:
            task["progress"] = _read_progress(task["backup_prefix"], task_id, search_dir)
    task["active"] = _is_process_active(task_id)
    return task


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
        if row.get("backup_prefix"):
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
    """调节运行中任务的并行数"""
    if body.n_jobs < 1:
        raise HTTPException(400, "n_jobs 必须大于 0")

    with _db_lock:
        db = get_db()
        row = db.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
        db.close()

    if not row:
        raise HTTPException(404, "任务不存在")

    backup_prefix = row["backup_prefix"]
    if not backup_prefix:
        raise HTTPException(400, "任务尚未生成 backup 文件前缀")

    script_dir = os.path.dirname(os.path.abspath(row["script_path"]))
    control_path = os.path.join(script_dir, f"{backup_prefix}.control")
    control_data = {"n_jobs": body.n_jobs}
    try:
        with open(control_path, "w") as f:
            json.dump(control_data, f)
    except OSError as e:
        raise HTTPException(500, f"写入控制文件失败: {e}")

    # 更新数据库中的 n_jobs
    with _db_lock:
        db = get_db()
        db.execute("UPDATE tasks SET n_jobs=? WHERE id=?", (body.n_jobs, task_id))
        db.commit()
        db.close()

    return {"status": "adjusted", "n_jobs": body.n_jobs}


def _read_log_file(file_path: str, offset: int, limit: int) -> dict:
    """读取日志文件并分页"""
    if not file_path or not os.path.isfile(file_path):
        return {"log": "", "total_lines": 0}
    try:
        with open(file_path) as f:
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


def _read_progress(backup_prefix: str, task_id: int, search_dir: str | None = None) -> dict | None:
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


def _is_process_active(task_id: int) -> bool:
    with _active_pids_lock:
        proc = _active_processes.get(task_id)
        if proc is None:
            return False
        return proc.poll() is None


# ── 入口 ─────────────────────────────────────────────


def main():
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
    parser.add_argument("--reload", action="store_true", help="启用热重载（开发用）")
    args = parser.parse_args()

    init_db(args.db)
    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
