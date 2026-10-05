"""按审阅名单清理 factor_store 的 colblk 文件，并实时生成进度页。

- 名单来源：reports/okay_database/colblk_delete_list.json（由 20261005 审阅名单的「1-可删名单(逐文件)」导出）
- 删除方式：ionice -c2 -n7 nice -n 19 rm -f（尽最大努力的最低 IO 优先级）。
  不用 -c3 空闲级别：实测被别的磁盘活动饿住，一个 unlink 停在 D 状态十几分钟没动静。
- 保护：路径里带 _accepted 的入选原件一律跳过（体积只有 0.16 GB，留着零风险）
- 进度页：reports/okay_database/colblk_cleanup.html，每删完一个文件重写一次，页面 5 秒自刷新

用法：
    python lib/okay_database/cleanup_colblk.py            # 真删
    python lib/okay_database/cleanup_colblk.py --dry-run  # 只统计不删
"""
from __future__ import annotations

import argparse
import html
import json
import os
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path

REPO = Path("/home/chenzongwei/rust_pyfunc")
REPORT_DIR = REPO / "reports/okay_database"
DELETE_LIST = REPORT_DIR / "colblk_delete_list.json"
STATE_PATH = REPORT_DIR / "colblk_cleanup_state.json"
HTML_PATH = REPORT_DIR / "colblk_cleanup.html"

# 盘口：用户看的是 /hdd/user_home_unsafe/chenzongwei 下的剩余可用/总空间，它是 /hdd 这个文件系统的一部分
WATCH_DIR = "/hdd/user_home_unsafe/chenzongwei"
GUARD_SUBSTR = "_accepted"


def disk_stat(path: str) -> dict:
    usage = shutil.disk_usage(path)
    return {
        "path": path,
        "total": usage.total,
        "used": usage.used,
        "free": usage.free,
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB", "PB"):
        if abs(n) < 1024 or unit == "PB":
            return f"{n:,.2f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024
    return f"{n:.2f} PB"


def load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {}


def save_state(state: dict) -> None:
    tmp = STATE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=1))
    tmp.replace(STATE_PATH)


def write_html(state: dict) -> None:
    items = state["items"]
    total_n = len(items)
    total_b = state["total_bytes"]
    done = [i for i in items if i["status"] == "deleted"]
    failed = [i for i in items if i["status"] == "failed"]
    skipped = [i for i in items if i["status"] == "skipped"]
    freed = sum(i["size"] for i in done)
    pct = (freed / total_b * 100) if total_b else 0.0

    before, after = state.get("disk_before"), state.get("disk_after")
    now = disk_stat(WATCH_DIR)

    # 按 store 汇总
    stores: dict[str, dict] = {}
    for i in items:
        s = stores.setdefault(i["store"], {"n": 0, "size": 0, "done": 0, "freed": 0, "series": i["series"]})
        s["n"] += 1
        s["size"] += i["size"]
        if i["status"] == "deleted":
            s["done"] += 1
            s["freed"] += i["size"]

    rows = []
    for name, s in sorted(stores.items(), key=lambda kv: -kv[1]["size"]):
        p = (s["freed"] / s["size"] * 100) if s["size"] else 0
        rows.append(
            "<tr>"
            f"<td class='mono'>{html.escape(name)}</td>"
            f"<td class='mono'>{html.escape(str(s['series'] or '-'))}</td>"
            f"<td class='num'>{s['done']}/{s['n']}</td>"
            f"<td class='num'>{human(s['size'])}</td>"
            f"<td class='num'>{human(s['freed'])}</td>"
            f"<td><div class='mini'><span style='width:{p:.1f}%'></span></div></td>"
            "</tr>"
        )

    logs = "".join(f"<div class='log'>{html.escape(line)}</div>" for line in state.get("log", [])[-40:])

    disk_rows = ""
    for label, st in (("清理前", before), ("清理后", after), ("此刻", now)):
        if not st:
            continue
        disk_rows += (
            "<tr>"
            f"<td>{label}</td>"
            f"<td class='mono'>{st['time']}</td>"
            f"<td class='num'>{human(st['total'])}</td>"
            f"<td class='num'>{human(st['used'])}</td>"
            f"<td class='num strong'>{human(st['free'])}</td>"
            "</tr>"
        )

    delta = ""
    if before and after:
        delta = human(after["free"] - before["free"])

    status_text = {"running": "进行中", "done": "已完成", "dry-run": "演练（未删除）"}.get(
        state.get("status", "running"), state.get("status", "running")
    )

    page = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>colblk 清理进度</title>
<meta http-equiv="refresh" content="5">
<style>
 body{{margin:0;padding:24px 28px;background:#0f1216;color:#e6e9ef;
      font-family:-apple-system,"PingFang SC","Microsoft YaHei",sans-serif}}
 h1{{font-size:20px;margin:0 0 4px}} .sub{{color:#8b93a1;font-size:13px;margin-bottom:18px}}
 .cards{{display:flex;gap:14px;flex-wrap:wrap;margin-bottom:20px}}
 .card{{background:#171b21;border:1px solid #232a33;border-radius:10px;padding:14px 18px;min-width:170px}}
 .card .k{{font-size:12px;color:#8b93a1;margin-bottom:6px}} .card .v{{font-size:20px;font-weight:600}}
 .bar{{height:14px;background:#232a33;border-radius:7px;overflow:hidden;margin:6px 0 22px}}
 .bar span{{display:block;height:100%;background:linear-gradient(90deg,#2e7d6f,#43c59e);width:{pct:.2f}%}}
 table{{width:100%;border-collapse:collapse;font-size:13px;margin-bottom:22px}}
 th,td{{padding:7px 10px;border-bottom:1px solid #232a33;text-align:left}}
 th{{color:#8b93a1;font-weight:500;font-size:12px}}
 td.num{{text-align:right;font-variant-numeric:tabular-nums}} .strong{{color:#43c59e;font-weight:600}}
 .mono{{font-family:ui-monospace,Menlo,monospace;font-size:12px}}
 .mini{{height:8px;background:#232a33;border-radius:4px;overflow:hidden;min-width:90px}}
 .mini span{{display:block;height:100%;background:#43c59e}}
 .log{{font-family:ui-monospace,Menlo,monospace;font-size:11.5px;color:#9aa4b2;line-height:1.55}}
 .badge{{display:inline-block;padding:2px 9px;border-radius:10px;font-size:12px;background:#1d3a33;color:#43c59e}}
 .badge.run{{background:#3a3220;color:#e0b556}}
 h2{{font-size:15px;margin:22px 0 10px;color:#c7cfdb}}
</style></head><body>
<h1>factor_store colblk 清理进度</h1>
<div class="sub">清单来源：20261005 colblk 清理审阅名单的「可删」表 ·
 状态 <span class="badge {'run' if state.get('status')=='running' else ''}">{status_text}</span> ·
 更新于 {now['time']}（页面每 5 秒自刷新）</div>

<div class="cards">
 <div class="card"><div class="k">已释放</div><div class="v">{human(freed)}</div></div>
 <div class="card"><div class="k">计划释放</div><div class="v">{human(total_b)}</div></div>
 <div class="card"><div class="k">文件进度</div><div class="v">{len(done)}/{total_n}</div></div>
 <div class="card"><div class="k">失败 / 跳过</div><div class="v">{len(failed)} / {len(skipped)}</div></div>
 <div class="card"><div class="k">本文件系统可用</div><div class="v">{human(now['free'])}</div></div>
</div>
<div class="bar"><span></span></div>

<h2>{WATCH_DIR} 所在文件系统的空间</h2>
<table><tr><th>时点</th><th>时刻</th><th>总空间</th><th>已用</th><th>剩余可用</th></tr>{disk_rows}</table>
<div class="sub">{'本次清理净增可用空间 ' + delta if delta else '清理结束后这里会显示净增空间'}</div>

<h2>按系列 store 的进度</h2>
<table><tr><th>store 目录</th><th>归属系列</th><th>文件</th><th>计划</th><th>已释放</th><th>进度</th></tr>{''.join(rows)}</table>

<h2>执行日志</h2>
{logs or "<div class='log'>等待开始…</div>"}
</body></html>"""
    tmp = HTML_PATH.with_suffix(".tmp")
    tmp.write_text(page)
    tmp.replace(HTML_PATH)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--keep-accepted", dest="keep_accepted", action="store_true", default=True)
    args = ap.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    raw = json.loads(DELETE_LIST.read_text())
    previous = load_state()
    prev_status = {i["path"]: i for i in previous.get("items", [])} if previous else {}
    items = []
    for r in raw:
        keep = args.keep_accepted and GUARD_SUBSTR in r["path"]
        old = prev_status.get(r["path"], {})
        status = old.get("status") if old.get("status") in ("deleted", "failed") else (
            "skipped" if keep else "pending"
        )
        items.append(
            {
                "store": r["store"],
                "path": r["path"],
                "size": r["size"],
                "series": r["series"],
                "status": status,
                "note": old.get("note", "") or ("入选原件，保留" if keep else ""),
            }
        )
    todo = [i for i in items if i["status"] == "pending"]
    total_b = sum(i["size"] for i in items if i["status"] != "skipped")

    state = {
        "started": previous.get("started") or datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "resumed": datetime.now().strftime("%Y-%m-%d %H:%M:%S") if previous else None,
        "status": "dry-run" if args.dry_run else "running",
        "dry_run": args.dry_run,
        "disk_before": (previous.get("disk_before") if previous else None) or disk_stat(WATCH_DIR),
        "disk_after": None,
        "items": items,
        "total_bytes": total_b,
        "log": previous.get("log", []) if previous else [],
    }
    log = state["log"]
    already = sum(1 for i in items if i["status"] == "deleted")
    log.append(
        f"[{datetime.now():%H:%M:%S}] 计划删除 {len(todo) + already} 个 colblk，合计 {human(total_b)}；"
        f"已删 {already}，本次待删 {len(todo)}；跳过 {sum(1 for i in items if i['status'] == 'skipped')} 个（入选原件）"
    )
    log.append(f"[{datetime.now():%H:%M:%S}] 清理前可用空间 {human(state['disk_before']['free'])} / 总 {human(state['disk_before']['total'])}")
    save_state(state)
    write_html(state)
    print("\n".join(log))

    if args.dry_run:
        state["status"] = "done"
        state["disk_after"] = disk_stat(WATCH_DIR)
        save_state(state)
        write_html(state)
        return 0

    freed = 0
    for idx, item in enumerate(todo, 1):
        path = item["path"]
        try:
            if not os.path.isfile(path):
                item["status"] = "skipped"
                item["note"] = "文件已不存在"
            else:
                # ionice 最低优先级 + nice 19，尽量不干扰同时在跑的计算任务
                r = subprocess.run(
                    ["ionice", "-c2", "-n7", "nice", "-n", "19", "rm", "-f", "--", path],
                    capture_output=True,
                    text=True,
                )
                if r.returncode == 0:
                    item["status"] = "deleted"
                    freed += item["size"]
                    log.append(f"[{datetime.now():%H:%M:%S}] {idx}/{len(todo)} 已删 {human(item['size'])} {path}")
                else:
                    item["status"] = "failed"
                    item["note"] = (r.stderr or "").strip()[:200]
                    log.append(f"[{datetime.now():%H:%M:%S}] 删除失败 {path} :: {item['note']}")
        except Exception as exc:  # noqa: BLE001
            item["status"] = "failed"
            item["note"] = str(exc)[:200]
            log.append(f"[{datetime.now():%H:%M:%S}] 异常 {path} :: {item['note']}")
        save_state(state)
        write_html(state)

    state["status"] = "done"
    state["finished"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    state["disk_after"] = disk_stat(WATCH_DIR)
    log.append(
        f"[{datetime.now():%H:%M:%S}] 完成。已释放 {human(freed)}；清理后可用空间 "
        f"{human(state['disk_after']['free'])} / 总 {human(state['disk_after']['total'])}"
    )
    save_state(state)
    write_html(state)
    print("\n".join(log[-6:]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
