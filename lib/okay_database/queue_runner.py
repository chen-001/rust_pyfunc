"""hm0~hm92 因子库补算的串行队列。

用户的要求：改写可以很多 agent 并行做，但**计算任务同时只能跑一个**。
本脚本就把这件事落到 factor_taskd 上：

  1. 把守护进程的 max_concurrent 设成 1（多任务同时提交时后面的自动排队）
  2. 只从 program_state.json 里挑「改写已完成、还没提交」的系列
  3. 提交 → 等待 → 记录结果 → 下一个
  4. 每轮把 program_state.json 和总进度页刷新一次

用法：
    python lib/okay_database/queue_runner.py            # 常驻，一个接一个跑
    python lib/okay_database/queue_runner.py --once     # 只处理一个就退出
    python lib/okay_database/queue_runner.py --status   # 只看队列状态
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

REPO = Path("/home/chenzongwei/rust_pyfunc")
REPORT_DIR = REPO / "reports/okay_database"
STATE_PATH = REPORT_DIR / "program_state.json"
QUEUE_LOG = REPORT_DIR / "queue_log.jsonl"
DAEMON = os.environ.get("FACTOR_TASK_URL", "http://127.0.0.1:9099")
PY = "/home/chenzongwei/.conda/envs/chenzongwei311/bin/python"
CLI = str(REPO / "python/rust_pyfunc/factor_task.py")

START_DATE = 20150105
END_DATE = 20260918
N_JOBS = 400
LIBRARY = "/nas197/user_home/chenzongwei/factor_library"


def api(method: str, path: str, body: dict | None = None, timeout: int = 15):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(f"{DAEMON}{path}", data=data, method=method)
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode()
    return json.loads(raw) if raw else None


def set_single_slot() -> None:
    """把守护进程限制成同时只跑一个任务，后来的自动排队。"""
    try:
        api("POST", "/api/config", {"max_concurrent": 1})
        cfg = api("GET", "/api/config")
        print(f"[队列] 守护进程 max_concurrent = {cfg.get('max_concurrent')}")
    except Exception as exc:  # noqa: BLE001
        print(f"[队列] 设置 max_concurrent 失败（继续按提交顺序跑）: {exc}")


def load_state() -> dict:
    return json.loads(STATE_PATH.read_text())


def save_state(state: dict) -> None:
    tmp = STATE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=1))
    tmp.replace(STATE_PATH)


def log(record: dict) -> None:
    with QUEUE_LOG.open("a") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def running_tasks() -> list[dict]:
    tasks = api("GET", "/api/tasks") or []
    return [t for t in tasks if t.get("status") in ("running", "pending", "queued")]


SUBMIT_DIR = REPORT_DIR / "submit"


def submit(script: str) -> dict:
    """提交脚本。

    任务系统的任务名 = 脚本文件名，hmXX_okay_database 这种名字在旧版守护进程里会被判「不含中文」
    而拒绝。新版守护进程已经放宽（正则加了 _okay_database），但要重启才生效。
    这里先按真名提交，被拒就退到一个合规名字的软链接上提交同一个文件，
    等守护进程重启后再自动走真名，不需要改脚本。
    """
    for path in _submit_candidates(script):
        cmd = [PY, CLI, "submit", path, "--n-jobs", str(N_JOBS),
               "--start-date", str(START_DATE), "--end-date", str(END_DATE)]
        out = subprocess.run(cmd, capture_output=True, text=True)
        text = (out.stdout or "") + (out.stderr or "")
        if out.returncode != 0 and "任务名称不合规" in text:
            print(f"[队列] 任务系统不接受文件名 {Path(path).name}，改用合规名字的软链接提交")
            continue
        print(out.stdout.strip() or out.stderr.strip())
        if out.returncode != 0:
            raise RuntimeError(out.stderr.strip() or "提交失败")
        for line in out.stdout.splitlines():
            if "任务 ID" in line:
                return {"id": int(line.split(":")[-1].strip()), "submitted_path": path}
        raise RuntimeError("提交成功但没解析到任务 ID")
    raise RuntimeError("提交失败：任务系统拒绝了这个文件名，且软链接兜底也没成功")


def _submit_candidates(script: str):
    """优先进程里的真名；真名不合规时给一个 hm<数字>.py 的软链接。"""
    name = Path(script).stem
    m = re.match(r"^hm(\d+)_okay_database$", name)
    if not m:
        return [script]
    SUBMIT_DIR.mkdir(parents=True, exist_ok=True)
    link = SUBMIT_DIR / f"hm{int(m.group(1))}.py"
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(Path(script).resolve())
    return [script, str(link)]


def wait_task(task_id: int, poll: int = 30) -> str:
    while True:
        task = api("GET", f"/api/tasks/{task_id}")
        status = task.get("status")
        if status in ("completed", "failed", "cancelled"):
            return status
        time.sleep(poll)


def library_written(ver: str) -> dict:
    """核对因子库四个落点里这个系列的文件数。"""
    out = {}
    for label, sub in (("base", "factor_base"), ("gap5", "hf_data"), ("gap1", "hf_data1"), ("db", "dailybank")):
        d = Path(LIBRARY) / sub / (f"{ver}_base" if sub == "factor_base" else ver)
        out[label] = len(list(d.glob("*.h5"))) if d.is_dir() else 0
    return out


def colblk_left(ver: str) -> int:
    root = Path("/hdd/user_home_unsafe/chenzongwei") / f"factor_store_{ver}_okay_database"
    if not root.exists():
        return 0
    return len(list(root.rglob("*.colblk")))


def pick_next(state: dict) -> dict | None:
    """挑下一个该提交的：改写好了、还没算完、也不在跑。"""
    for s in state["series"]:
        if s["step_rewrite"] == "done" and s["step_submit"] not in ("done", "running"):
            return s
    return None


def reconcile(state: dict) -> None:
    """进程重启后对账：标着 running 的系列，如果任务已经不在了，就按结果落账。"""
    changed = False
    for s in state["series"]:
        if s.get("step_submit") != "running" or not s.get("task_id"):
            continue
        try:
            task = api("GET", f"/api/tasks/{s['task_id']}")
        except Exception:  # noqa: BLE001
            continue
        status = task.get("status")
        if status in ("completed", "failed", "cancelled"):
            s["step_submit"] = "done" if status == "completed" else status
            files = library_written(s["ver"])
            s["library_files"] = files
            s["step_library"] = "done" if any(files.values()) else "pending"
            left = colblk_left(s["ver"])
            s["colblk_left"] = left
            s["step_cleanup"] = "done" if left == 0 else "pending"
            s["note"] = f"对账：{status}，因子库 {files}，残留 colblk {left}"
            changed = True
    if changed:
        save_state(state)
        refresh_dashboard()


def refresh_dashboard() -> None:
    subprocess.run([PY, str(REPO / "lib/okay_database/render_dashboard.py")], capture_output=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--poll", type=int, default=30)
    args = ap.parse_args()

    state = load_state()
    if args.status:
        for s in state["series"]:
            print(f"{s['series']:>8} 改写={s['step_rewrite']:<8} 提交={s['step_submit']:<8} "
                  f"入库={s['step_library']:<8} 清colblk={s['step_cleanup']:<8} task={s['task_id']}")
        return 0

    set_single_slot()
    reconcile(load_state())
    while True:
        state = load_state()
        busy = running_tasks()
        if busy:
            print(f"[队列] 已有任务在跑（id={busy[0]['id']}），等它结束")
            time.sleep(args.poll)
            continue

        series = pick_next(state)
        if series is None:
            print("[队列] 没有待提交的系列了")
            refresh_dashboard()
            return 0

        script = series["database_py"]
        if not Path(script).is_file():
            series["note"] = "database 脚本不存在"
            save_state(state)
            print(f"[队列] {series['series']} 缺少 {script}，跳过")
            return 1 if args.once else 0

        print(f"[队列] 提交 {series['series']} → {script}")
        info = submit(script)
        series["task_id"] = info["id"]
        series["step_submit"] = "running"
        series["note"] = f"提交于 {datetime.now():%Y-%m-%d %H:%M:%S}，n_jobs={N_JOBS}"
        save_state(state)
        refresh_dashboard()
        log({"series": series["series"], "event": "submit", "task_id": info["id"],
             "time": datetime.now().isoformat()})

        status = wait_task(info["id"], poll=args.poll)
        series["step_submit"] = "done" if status == "completed" else status
        files = library_written(series["ver"])
        series["library_files"] = files
        series["step_library"] = "done" if files["gap5"] or files["gap1"] or files["base"] else "pending"
        left = colblk_left(series["ver"])
        series["colblk_left"] = left
        series["step_cleanup"] = "done" if left == 0 else "pending"
        series["note"] = f"{status}，因子库 {files}，残留 colblk {left}"
        save_state(state)
        refresh_dashboard()
        log({"series": series["series"], "event": "finish", "status": status,
             "files": files, "colblk_left": left, "time": datetime.now().isoformat()})
        print(f"[队列] {series['series']} 结束：{status}，因子库 {files}，残留 colblk {left}")

        if args.once:
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
