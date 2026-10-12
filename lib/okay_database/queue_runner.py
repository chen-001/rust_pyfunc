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
import fcntl
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scan_series import kind_from_database_py  # noqa: E402

REPO = Path("/home/chenzongwei/rust_pyfunc")
REPORT_DIR = REPO / "reports/okay_database"
STATE_PATH = REPORT_DIR / "program_state.json"
QUEUE_LOG = REPORT_DIR / "queue_log.jsonl"
DAEMON = os.environ.get("FACTOR_TASK_URL", "http://127.0.0.1:9099")
PY = "/home/chenzongwei/.conda/envs/chenzongwei311/bin/python"
CLI = str(REPO / "python/rust_pyfunc/factor_task.py")

START_DATE = 20150105
END_DATE = 20260918
N_JOBS = 200
LIBRARY = "/nas197/user_home/chenzongwei/factor_library"

# 因子库统计的脚本与产物。界面状态条上的「因子库 · 数据 X/5614」取自这里的 coverage.json，
# 那份文件是快照，不主动刷就不动——2026-10-08 的现象（hm21 16:01 写完，界面还显示 15:15 的
# 旧数字 1349）就是这么来的。队列每完成一个系列在这里挂钩子，让数字自动跟上。
LIBRARY_REPO = Path("/home/chenzongwei/dsh-factor-research")
LIBRARY_CLI = LIBRARY_REPO / "lib/factor_library.py"
COVERAGE_JSON = LIBRARY_REPO / "factor_library/coverage.json"
REFRESH_LOCK = REPORT_DIR / "claims" / ".library_refresh.lock"
REFRESH_TIMEOUT = 1800  # 四个步骤正常两分钟左右，给到半小时只为止损
REFRESH_SAMPLE = f"{START_DATE}-{END_DATE}"


def api(method: str, path: str, body: dict | None = None, timeout: int = 30, retries: int = 6):
    """守护进程偶发超时（计算任务把机器压满时）不该让常驻队列整个退出，这里自动重试。"""
    last = None
    for attempt in range(retries):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(f"{DAEMON}{path}", data=data, method=method)
        req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode()
            return json.loads(raw) if raw else None
        except Exception as exc:  # noqa: BLE001
            last = exc
            print(f"[队列] 守护进程无响应（第 {attempt + 1}/{retries} 次）：{exc}")
            time.sleep(min(5 * (attempt + 1), 30))
    raise RuntimeError(f"守护进程连续 {retries} 次无响应: {last}")


def set_single_slot() -> None:
    """把守护进程限制成同时只跑一个任务，后来的自动排队。"""
    try:
        api("POST", "/api/config", {"max_concurrent": 1})
        cfg = api("GET", "/api/config")
        print(f"[队列] 守护进程 max_concurrent = {cfg.get('max_concurrent')}")
    except Exception as exc:  # noqa: BLE001
        print(f"[队列] 设置 max_concurrent 失败（继续按提交顺序跑）: {exc}")


# 和 claim_series.py 共用同一把文件锁：program_state.json 是多方共写的，
# 不加锁时队列整体覆盖会把 worker 刚写进去的完成记录（含 note）吃掉。
_STATE_LOCK = REPORT_DIR / "claims" / ".state.lock"


def load_state() -> dict:
    return json.loads(STATE_PATH.read_text())


def save_state(state: dict) -> None:
    _STATE_LOCK.parent.mkdir(parents=True, exist_ok=True)
    with _STATE_LOCK.open("a") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        tmp = STATE_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, ensure_ascii=False, indent=1))
        tmp.replace(STATE_PATH)


def update_series(series_name: str, **fields) -> dict:
    """只改某个系列的字段，读-改-写全程持锁，避免覆盖别人的记录。"""
    _STATE_LOCK.parent.mkdir(parents=True, exist_ok=True)
    with _STATE_LOCK.open("a") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        state = json.loads(STATE_PATH.read_text())
        target = None
        for item in state["series"]:
            if item["series"] == series_name:
                target = item
                item.update(fields)
                break
        tmp = STATE_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, ensure_ascii=False, indent=1))
        tmp.replace(STATE_PATH)
    return state


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


def kind_conflict(series: dict) -> str | None:
    """清单里的 kind 与生产脚本 pipeline_groups 的 kind 不一致时返回说明，否则 None。

    hm074 那类坑：旧研究脚本是逐股形态，Rust 改写换成了横截面，清单里还留着旧的 kind。
    提交前对不上就在这里停下并记账，不靠人记得；由生产脚本（真源）说了算。
    """
    declared = series.get("kind")
    script = series.get("database_py") or ""
    actual = kind_from_database_py(Path(script))
    if not declared or actual is None:
        return None
    if actual != declared:
        return (f"kind 与生产脚本不一致：清单写的是 {declared}，{Path(script).name} 的 pipeline_groups "
                f"是 {actual}；先把两边改一致再提交")
    return None


def wait_task(task_id: int, poll: int = 30) -> str:
    while True:
        try:
            task = api("GET", f"/api/tasks/{task_id}")
        except Exception as exc:  # noqa: BLE001
            print(f"[队列] 查任务 {task_id} 暂时失败，继续等：{exc}")
            time.sleep(poll)
            continue
        status = task.get("status")
        if status in ("completed", "failed", "cancelled"):
            return status
        time.sleep(poll)


def library_ver(series: dict) -> str:
    """因子库里的版本名统一用 hm<编号>（hm013 → hm13），不用旧的 research ver。

    旧 okay 文件里的 ver 是研究代号（hide_order_depth、hot_change1 这类），
    因子库要按系列编号存放，这样 89 个系列摆在库里一眼能对上。
    """
    num = int(series["series"][2:])
    return f"hm{num}"


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
        if s.get("recheck") == "pending":
            continue  # 完成记录是重建来的、验证结论没被记下，先别提交计算
        # 只挑从没提交过的。失败/取消的不能自动重提：那会把同一个错误重复跑满几十分钟，
        # hm1 连跑 18 次失败就是这么来的（每次 46 分钟全量 + 最后一步抛异常）。
        # 失败的要先有人查清原因、改完，再把 step_submit 手改回 "pending" 才会重排。
        if s["step_rewrite"] == "done" and s["step_submit"] == "pending":
            return s
    return None


def reconcile(state: dict) -> None:
    """进程重启后对账：标着 running 的系列，如果任务已经不在了，就按结果落账。"""
    changed = False
    completed: list[str] = []
    for s in state["series"]:
        if s.get("step_submit") != "running" or not s.get("task_id"):
            continue
        try:
            task = api("GET", f"/api/tasks/{s['task_id']}")
        except Exception:  # noqa: BLE001
            continue
        status = task.get("status")
        if status in ("completed", "failed", "cancelled"):
            files = library_written(library_ver(s))
            left = colblk_left(library_ver(s))
            update_series(s["series"],
                          step_submit="done" if status == "completed" else status,
                          library_files=files,
                          step_library="done" if any(files.values()) else "pending",
                          colblk_left=left,
                          step_cleanup="done" if left == 0 else "pending",
                          note=f"对账：{status}，因子库 {files}，残留 colblk {left}")
            if status == "completed":
                completed.append(s["series"])
            changed = True
    if changed:
        refresh_dashboard()
    # 对账落账的完成也要刷新统计，否则队列重启后这批数字同样不跟手
    if completed:
        on_series_completed(completed)


def refresh_dashboard() -> None:
    subprocess.run([PY, str(REPO / "lib/okay_database/render_dashboard.py")], capture_output=True)


def library_with_data() -> int | None:
    """当前 coverage.json 里 gap1/gap5 口径的有数据条数，只用来记「刷之前是多少」。"""
    try:
        data = json.loads(COVERAGE_JSON.read_text())
        return data.get("summary", {}).get("gap", {}).get("withData")
    except Exception:  # noqa: BLE001
        return None


def refresh_library_stats(reason: str) -> dict:
    """把因子库的四个统计产物刷一遍（scan → coverage → backfill-plan → common-sample）。

    这是「界面数字不跟着补算走」的根治点。status 里的 withData 读的是 coverage.json，
    那是快照，以前只有人记得手动跑 coverage 才会更新：hm021 是 16:01 写完的，而最后一次
    手动跑是 15:15，那份快照里根本没有 hm021，界面就一直显示旧数字。

    刷新失败绝不影响补算主线，所有异常和超时都收成返回值，由调用方记进 queue_log。
    两处加锁：factor_library.py 自己那把锁管写入（排队、原子替换），这里这把锁只是
    避免两个队列进程同时刷一遍白烧 CPU。全程约两分钟，而下一个系列本来就是几小时，
    串行等着没有代价。
    """
    before = library_with_data()
    REFRESH_LOCK.parent.mkdir(parents=True, exist_ok=True)
    handle = REFRESH_LOCK.open("a")
    try:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            print("[队列] 另一次因子库刷新正在跑，这次跳过")
            return {"ok": True, "skipped": "refresh-busy", "reason": reason,
                    "withDataBefore": before, "withDataAfter": before}
        t0 = time.time()
        proc = subprocess.run(
            [PY, str(LIBRARY_CLI), "refresh", "--root", "/home/chenzongwei/pythoncode",
             "--sample", REFRESH_SAMPLE, "--json"],
            capture_output=True, text=True, timeout=REFRESH_TIMEOUT)
        text = (proc.stdout or "").strip()
        payload = {}
        if text:
            try:
                payload = json.loads(text.splitlines()[-1])
            except ValueError:
                payload = {}
        after = payload.get("withData")
        if after is None:
            after = library_with_data()
        info = {
            "ok": proc.returncode == 0,
            "reason": reason,
            "seconds": round(time.time() - t0, 1),
            "withDataBefore": before,
            "withDataAfter": after,
            "steps": payload.get("steps"),
            "failedSteps": payload.get("failedSteps"),
            "artifacts": payload.get("artifacts"),
        }
        if proc.returncode != 0:
            info["error"] = (payload.get("errors")
                             or (proc.stderr or "").strip()[-300:] or "刷新退出码非零")
        return info
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "reason": reason, "withDataBefore": before,
                "withDataAfter": library_with_data(), "error": repr(exc)}
    finally:
        try:
            fcntl.flock(handle, fcntl.LOCK_UN)
        finally:
            handle.close()


def on_series_completed(series_names: list[str]) -> None:
    """一个系列补算完成后的收尾。内部全部兜住异常，绝不让它影响提交下一条的主线。"""
    name = "、".join(series_names)
    try:
        print(f"[队列] {name} 完成，刷新因子库统计（coverage 约一两分钟）")
        info = refresh_library_stats(f"series-completed:{name}")
    except Exception as exc:  # noqa: BLE001
        info = {"ok": False, "reason": f"series-completed:{name}", "error": repr(exc)}
    try:
        log({"series": series_names, "event": "library-refresh",
             "time": datetime.now().isoformat(), **info})
    except Exception as exc:  # noqa: BLE001
        print(f"[队列] 刷新记录没写进 queue_log：{exc}")
    if info.get("ok"):
        print(f"[队列] 因子库统计已更新：有数据 {info.get('withDataBefore')} → "
              f"{info.get('withDataAfter')}（{info.get('seconds')} 秒）")
    else:
        print(f"[队列] 因子库统计刷新失败（不影响补算）：{info.get('error')}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--refresh-only", action="store_true",
                    help="只跑一次因子库统计刷新就退出（验证挂钩子用，不碰补算）")
    ap.add_argument("--tick", action="store_true",
                    help="给系统定时任务用的一次性动作：先对账已结束的任务，再看有没有在跑；"
                         "都没有就提交下一个待补的系列，然后立刻退出，不等它跑完")
    ap.add_argument("--poll", type=int, default=30)
    args = ap.parse_args()

    if args.refresh_only:
        on_series_completed(["手动触发"])
        return 0

    if args.tick:
        # 这个模式的存在理由：常驻进程会随对话结束一起消失，补算就断在「上一个算完了、
        # 下一个没人提交」这个缝上。让系统定时任务每几分钟叫一次这个模式，
        # 每次只做一件事就退出，就不依赖任何一直开着的进程了。
        state = load_state()
        reconcile(state)                      # 把已结束、还没记账的补上，并刷新统计
        if running_tasks():                   # 有任务在跑或在排队，这轮什么都不做
            return 0
        series = pick_next(load_state())
        if series is None:
            return 0
        script = series["database_py"]
        if not Path(script).is_file():
            update_series(series["series"], note="database 脚本不存在，无法提交")
            return 1
        conflict = kind_conflict(series)
        if conflict:
            update_series(series["series"], note=conflict)
            log({"series": series["series"], "event": "kind-conflict",
                 "time": datetime.now().isoformat()})
            print(f"[tick] 拒绝提交 {series['series']}：{conflict}")
            return 1
        info = submit(script)
        update_series(series["series"], task_id=info["id"], step_submit="running",
                      note=f"提交于 {datetime.now():%Y-%m-%d %H:%M:%S}，n_jobs={N_JOBS}")
        log({"series": series["series"], "event": "submit", "task_id": info["id"],
             "time": datetime.now().isoformat(), "by": "tick"})
        refresh_dashboard()
        print(f"[tick] 已提交 {series['series']}（任务 {info['id']}）")
        return 0

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
        # 每轮都对账一次：别的队列进程/上次退出留下的 running 状态都能补上
        reconcile(state)
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

        conflict = kind_conflict(series)
        if conflict:
            series["note"] = conflict
            save_state(state)
            log({"series": series["series"], "event": "kind-conflict",
                 "time": datetime.now().isoformat()})
            print(f"[队列] 拒绝提交 {series['series']}：{conflict}")
            return 1
        print(f"[队列] 提交 {series['series']} → {script}")
        info = submit(script)
        update_series(series["series"], task_id=info["id"], step_submit="running",
                      note=f"提交于 {datetime.now():%Y-%m-%d %H:%M:%S}，n_jobs={N_JOBS}")
        refresh_dashboard()
        log({"series": series["series"], "event": "submit", "task_id": info["id"],
             "time": datetime.now().isoformat()})

        status = wait_task(info["id"], poll=args.poll)
        files = library_written(library_ver(series))
        left = colblk_left(library_ver(series))
        update_series(series["series"],
                      step_submit="done" if status == "completed" else status,
                      library_files=files,
                      step_library="done" if any(files.values()) else "pending",
                      colblk_left=left,
                      step_cleanup="done" if left == 0 else "pending",
                      note=f"{status}，因子库 {files}，残留 colblk {left}")
        state = load_state()
        refresh_dashboard()
        log({"series": series["series"], "event": "finish", "status": status,
             "files": files, "colblk_left": left, "time": datetime.now().isoformat()})
        print(f"[队列] {series['series']} 结束：{status}，因子库 {files}，残留 colblk {left}")

        # 只有真算完才刷新统计。失败/取消的没往库里写新东西，刷了数字也不动，
        # 只会白烧两分钟并把「没涨」记进日志，看着像又一次故障。
        if status == "completed":
            on_series_completed([series["series"]])

        if args.once:
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
