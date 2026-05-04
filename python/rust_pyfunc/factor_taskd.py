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
            error TEXT
        )
    """
    )
    db.commit()
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

    cmd = [sys.executable, script_path]
    # 如果脚本接受参数，可以在这里添加
    env = os.environ.copy()

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

        # 标记完成
        new_status = "completed" if returncode == 0 else "failed"
        error_msg = None if returncode == 0 else f"进程退出码: {returncode}"
        with _db_lock:
            db = get_db()
            db.execute(
                "UPDATE tasks SET status=?, finished_at=?, error=? WHERE id=?",
                (
                    new_status,
                    datetime.now(timezone.utc).isoformat(),
                    error_msg,
                    task_id,
                ),
            )
            db.commit()
            db.close()

    except Exception as e:
        with _db_lock:
            db = get_db()
            db.execute(
                "UPDATE tasks SET status='failed', finished_at=?, error=? WHERE id=?",
                (datetime.now(timezone.utc).isoformat(), str(e), task_id),
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
            # 尝试读取进度信息
            if task.get("backup_prefix"):
                progress = _read_progress(task["backup_prefix"], task["id"])
                task["progress"] = progress
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
        task["progress"] = _read_progress(task["backup_prefix"], task_id)
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

        db.execute(
            "UPDATE tasks SET status='cancelled', finished_at=? WHERE id=?",
            (datetime.now(timezone.utc).isoformat(), task_id),
        )
        db.commit()
        db.close()

    # 终止进程
    with _active_pids_lock:
        proc = _active_processes.get(task_id)
        if proc:
            try:
                pgid = os.getpgid(proc.pid)
                os.killpg(pgid, signal.SIGTERM)
                # 等 3 秒再强制杀
                for _ in range(30):
                    if proc.poll() is not None:
                        break
                    time.sleep(0.1)
                if proc.poll() is None:
                    os.killpg(pgid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
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

    control_path = f"./{backup_prefix}.control"
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


@app.get("/api/tasks/{task_id}/log")
def get_task_log(task_id: int):
    """获取任务日志"""
    with _db_lock:
        db = get_db()
        row = db.execute("SELECT log_path FROM tasks WHERE id=?", (task_id,)).fetchone()
        db.close()
    if not row:
        raise HTTPException(404, "任务不存在")
    log_path = row["log_path"]
    if not log_path or not os.path.isfile(log_path):
        return {"log": ""}
    try:
        with open(log_path) as f:
            # 只读取最后 1000 行
            lines = f.readlines()
            tail = lines[-1000:] if len(lines) > 1000 else lines
            return {"log": "".join(tail)}
    except OSError:
        return {"log": ""}


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


def _read_progress(backup_prefix: str, task_id: int) -> dict | None:
    """读取 backup_<ver>.bin.progress.jsonl 的进度信息"""
    progress_path = f"{backup_prefix}.progress.jsonl"
    if not os.path.isfile(progress_path):
        # 也检查 ./backup_<ver>.bin.progress.jsonl
        progress_path = (
            f"./{backup_prefix}.progress.jsonl"
            if not backup_prefix.startswith("./")
            else progress_path
        )
        if not os.path.isfile(progress_path):
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
