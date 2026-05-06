#!/usr/bin/env python3
"""factor-task — 因子计算任务 CLI 工具

通过 HTTP 与 factor_taskd 守护进程通信，管理因子计算任务。

用法:
  factor-task submit <script.py> [--n-jobs N] [--start-date YYYYMMDD] [--end-date YYYYMMDD]
  factor-task list
  factor-task show <id>
  factor-task cancel <id>
  factor-task adjust <id> <n_jobs>
  factor-task log <id>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

DAEMON_URL = os.environ.get("FACTOR_TASK_URL", "http://127.0.0.1:9099")
TIMEOUT = 10


def _request(method: str, path: str, body: dict | None = None) -> dict:
    url = f"{DAEMON_URL}{path}"
    data = json.dumps(body).encode() if body else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        err = e.read().decode()
        print(f"❌ HTTP {e.code}: {err}", file=sys.stderr)
        sys.exit(1)
    except urllib.error.URLError as e:
        print(f"❌ 连接失败 (确保 factor_taskd 正在运行): {e.reason}", file=sys.stderr)
        sys.exit(1)


def cmd_submit(args):
    body = {
        "script_path": os.path.abspath(args.script),
        "n_jobs": args.n_jobs or 50,
    }
    if args.start_date:
        body["start_date"] = args.start_date
    if args.end_date:
        body["end_date"] = args.end_date

    resp = _request("POST", "/api/tasks", body)
    print(f"✅ 任务已提交")
    print(f"   任务 ID: {resp['id']}")
    print(f"   名称:    {resp['name']}")
    print(f"   状态:    {resp['status']}")
    print(f"   查看:    factor-task show {resp['id']}")


def cmd_list(args):
    tasks = _request("GET", "/api/tasks")
    if not tasks:
        print("📭 暂无任务")
        return
    print(f"{'ID':>4}  {'名称':<20} {'状态':<12} {'n_jobs':<6} {'PID':<7} {'进度':<10}")
    print("-" * 70)
    for t in tasks:
        progress = ""
        if t.get("progress") and t["progress"].get("latest"):
            latest = t["progress"]["latest"]
            if "collected" in latest and "total_tasks" in latest:
                pct = (
                    latest["collected"] / latest["total_tasks"] * 100
                    if latest["total_tasks"]
                    else 0
                )
                progress = f"{latest['collected']}/{latest['total_tasks']}({pct:.0f}%)"
        pid = t.get("pid") or ""
        n_jobs = t.get("n_jobs") or ""
        status = t.get("status", "unknown")
        print(
            f"{t['id']:>4}  {t.get('name',''):<20} {status:<12} {str(n_jobs):<6} {str(pid):<7} {progress:<10}"
        )


def cmd_show(args):
    task = _request("GET", f"/api/tasks/{args.id}")
    print(f"任务 ID:      {task['id']}")
    print(f"名称:         {task.get('name', '-')}")
    print(f"脚本路径:     {task.get('script_path', '-')}")
    print(f"状态:         {task.get('status', '-')}")
    print(f"PID:          {task.get('pid', '-')}")
    print(f"n_jobs:       {task.get('n_jobs', '-')}")
    print(f"起止日期:     {task.get('start_date', '-')} ~ {task.get('end_date', '-')}")
    print(f"创建时间:     {task.get('created_at', '-')}")
    print(f"开始时间:     {task.get('started_at', '-')}")
    print(f"完成时间:     {task.get('finished_at', '-')}")
    print(f"日志路径:     {task.get('log_path', '-')}")
    print(f"运行中:       {task.get('active', False)}")
    if task.get("error"):
        print(f"错误:         {task['error']}")

    progress = task.get("progress")
    if progress and progress.get("records"):
        records = progress["records"]
        latest = progress.get("latest", {})
        if "collected" in latest and "total_tasks" in latest:
            pct = (
                latest["collected"] / latest["total_tasks"] * 100
                if latest["total_tasks"]
                else 0
            )
            print(
                f"进度:         {latest['collected']}/{latest['total_tasks']} ({pct:.1f}%)"
            )
        if len(records) > 1:
            last = records[-1]
            print(
                f"最后备份:     {last.get('batch', '-')}/{last.get('total_batches', '-')}"
            )


def cmd_cancel(args):
    resp = _request("POST", f"/api/tasks/{args.id}/cancel")
    print(f"✅ 任务 {args.id} 已取消")


def cmd_adjust(args):
    body = {"n_jobs": args.n_jobs}
    resp = _request("POST", f"/api/tasks/{args.id}/adjust-njobs", body)
    print(f"✅ 任务 {args.id} n_jobs 已调整为 {args.n_jobs}")


def cmd_log(args):
    resp = _request("GET", f"/api/tasks/{args.id}/log")
    log = resp.get("log", "")
    if log:
        print(log)
    else:
        print("📭 日志为空")


def main():
    parser = argparse.ArgumentParser(description="因子计算任务 CLI 工具")
    parser.add_argument(
        "--url", default=DAEMON_URL, help=f"daemon 地址 (默认: {DAEMON_URL})"
    )
    sub = parser.add_subparsers(dest="command")

    p_submit = sub.add_parser("submit", help="提交新任务")
    p_submit.add_argument("script", help="Python 脚本路径")
    p_submit.add_argument("--n-jobs", type=int, default=None, help="并行数")
    p_submit.add_argument("--start-date", type=int, help="起始日期 (YYYYMMDD)")
    p_submit.add_argument("--end-date", type=int, help="结束日期 (YYYYMMDD)")
    p_submit.set_defaults(func=cmd_submit)

    p_list = sub.add_parser("list", help="列出所有任务")
    p_list.set_defaults(func=cmd_list)

    p_show = sub.add_parser("show", help="查看任务详情")
    p_show.add_argument("id", type=int, help="任务 ID")
    p_show.set_defaults(func=cmd_show)

    p_cancel = sub.add_parser("cancel", help="取消任务")
    p_cancel.add_argument("id", type=int, help="任务 ID")
    p_cancel.set_defaults(func=cmd_cancel)

    p_adjust = sub.add_parser("adjust", help="调节并行数")
    p_adjust.add_argument("id", type=int, help="任务 ID")
    p_adjust.add_argument("n_jobs", type=int, help="新的并行数")
    p_adjust.set_defaults(func=cmd_adjust)

    p_log = sub.add_parser("log", help="查看任务日志")
    p_log.add_argument("id", type=int, help="任务 ID")
    p_log.set_defaults(func=cmd_log)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    import factor_task  # noqa: F811
    factor_task.DAEMON_URL = args.url

    args.func(args)


if __name__ == "__main__":
    main()
