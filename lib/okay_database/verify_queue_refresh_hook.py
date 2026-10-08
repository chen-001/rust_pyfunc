"""验证「队列完成一个系列 → 因子库统计自动刷新」这条链是通的。

为什么要有这个脚本：那条链的正常触发条件是「某个系列补算跑完」，而一个系列要几小时，
不能靠等真实系列来验收。所以这里把队列的几处外部依赖换掉（提交、等任务、状态文件、
日志文件都换成临时的），但**完成分支本身与刷新函数用的是真代码**：
main() → wait_task 返回 completed → update_series → on_series_completed → refresh_library_stats
→ factor_library.py refresh → coverage.json 换新 → status 的 withData。

两个场景：
  A. 任务 completed：必须触发刷新，coverage.json 的 mtime 与 generatedAt 变新，
     queue_log 里出现一条 library-refresh。
  B. 任务 failed：不许触发刷新（没往库里写新东西，刷了数字也不动，白烧两分钟）。

跑法：
  /home/chenzongwei/.conda/envs/chenzongwei311/bin/python \
      lib/okay_database/verify_queue_refresh_hook.py

退出码 0 通过，1 不通过。全程约两三分钟（刷新里 coverage 那一步就要一分钟）。
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import queue_runner as qr  # noqa: E402

COVERAGE = qr.COVERAGE_JSON
FAKE_SERIES = "hm99"


def coverage_state() -> dict:
    """coverage.json 的三大件：mtime、generatedAt、gap 口径的 withData。"""
    try:
        stat = COVERAGE.stat()
        data = json.loads(COVERAGE.read_text())
    except (OSError, ValueError) as exc:
        return {"exists": False, "error": repr(exc)}
    return {
        "exists": True,
        "mtime": stat.st_mtime,
        "generatedAt": (data.get("meta") or {}).get("generatedAt"),
        "withData": data.get("summary", {}).get("gap", {}).get("withData"),
    }


def run_scenario(tmp: Path, status: str) -> dict:
    """跑一次 main(--once)，让那个假系列走到指定的结果。"""
    tmp.mkdir(parents=True, exist_ok=True)
    state = {
        "series": [{
            "series": FAKE_SERIES,
            "step_rewrite": "done",
            "step_submit": "pending",
            "task_id": None,
            "database_py": str(tmp / "hm99_okay_database.py"),
            "note": "",
        }],
    }
    (tmp / "hm99_okay_database.py").write_text("# 空壳，只为通过「脚本存在」这一关\n")
    state_path = tmp / "program_state.json"
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=1))
    queue_log = tmp / "queue_log.jsonl"
    if queue_log.exists():
        queue_log.unlink()

    # 只换掉对外的几处：状态文件、日志、锁；提交与等任务换成假的。
    # on_series_completed / refresh_library_stats / 完成分支本身都是真代码。
    qr.STATE_PATH = state_path
    qr.QUEUE_LOG = queue_log
    qr.REFRESH_LOCK = tmp / ".library_refresh.lock"
    qr._STATE_LOCK = tmp / ".state.lock"
    qr.SUBMIT_DIR = tmp / "submit"
    qr.set_single_slot = lambda: None
    qr.running_tasks = lambda: []
    qr.submit = lambda script: {"id": 424242, "submitted_path": script}
    qr.wait_task = lambda task_id, poll=30: status
    qr.refresh_dashboard = lambda: None  # 总进度页跟这条链无关，不在这里重画

    # queue_runner.main() 是从 sys.argv 读参数的，这里临时换掉
    saved_argv = sys.argv
    sys.argv = ["queue_runner.py", "--once"]
    try:
        rc = qr.main()
    finally:
        sys.argv = saved_argv

    saved = json.loads(state_path.read_text())["series"][0]
    events = []
    if queue_log.exists():
        events = [json.loads(line) for line in queue_log.read_text().splitlines() if line.strip()]
    return {"rc": rc, "saved": saved, "events": events}


def main() -> int:
    print(f"因子库产物：{COVERAGE}")
    tmp_root = Path(tempfile.mkdtemp(prefix="queue-refresh-hook-"))
    problems: list[str] = []
    try:
        # ── 场景 A：任务成功完成，必须刷新 ──────────────────────
        before = coverage_state()
        print(f"\n[A] 刷新前 coverage：mtime={before.get('mtime')} "
              f"generatedAt={before.get('generatedAt')} withData={before.get('withData')}")
        a = run_scenario(tmp_root / "a", "completed")
        after = coverage_state()
        print(f"[A] 刷新后 coverage：mtime={after.get('mtime')} "
              f"generatedAt={after.get('generatedAt')} withData={after.get('withData')}")

        if a["saved"].get("step_submit") != "done":
            problems.append(f"[A] 完成分支没落账：step_submit={a['saved'].get('step_submit')}")
        refresh_events = [e for e in a["events"] if e.get("event") == "library-refresh"]
        if not refresh_events:
            problems.append("[A] 完成之后没有 library-refresh 事件，挂钩子没生效")
        else:
            ev = refresh_events[0]
            print(f"[A] library-refresh：before={ev.get('withDataBefore')} "
                  f"after={ev.get('withDataAfter')} 秒={ev.get('seconds')} steps={ev.get('steps')}")
            if not ev.get("ok"):
                problems.append(f"[A] 刷新报失败：{ev.get('error')}")
        if after.get("generatedAt") == before.get("generatedAt"):
            problems.append(f"[A] coverage.json 的 generatedAt 没变，还是 {after.get('generatedAt')}")
        if not (after.get("mtime") or 0) > (before.get("mtime") or 0):
            problems.append("[A] coverage.json 的 mtime 没变新")
        if after.get("withData") != before.get("withData"):
            print(f"[A] 注意：withData 从 {before.get('withData')} 变成 {after.get('withData')}"
                  "（这轮没有新系列完成，正常情况下不变）")

        # ── 场景 B：任务失败，不许刷新（负对照）────────────────
        b_before = coverage_state()
        b = run_scenario(tmp_root / "b", "failed")
        b_after = coverage_state()
        b_events = [e for e in b["events"] if e.get("event") == "library-refresh"]
        if b_events:
            problems.append("[B] 任务失败却触发了刷新，失败路径不该刷")
        if b["saved"].get("step_submit") != "failed":
            problems.append(f"[B] 失败没落账：step_submit={b['saved'].get('step_submit')}")
        if b_after.get("mtime") != b_before.get("mtime"):
            problems.append("[B] 任务失败却把 coverage.json 重写了")
        print(f"[B] 任务 failed：library-refresh 事件 {len(b_events)} 条，"
              f"coverage mtime {'没变' if b_after.get('mtime') == b_before.get('mtime') else '变了'}")
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    if problems:
        print("\n不通过：")
        for item in problems:
            print(f"  {item}")
        return 1
    print("\n通过：完成一个系列会触发刷新，coverage.json 换新且 status 能读到；失败路径不刷。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
