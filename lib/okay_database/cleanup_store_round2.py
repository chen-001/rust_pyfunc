"""第二期清理：把 hm94~hm110 遗留的 24 个 factor_store 目录清空，外加一个隐藏归档。

第一期只删了 factors.colblk（11.42 TB），目录里剩下的 factors.proj / proj.delta 是同量级的
另一份副本，这一期一并清掉。

保留规则（每一期都遵守）：
- 路径里含 `_accepted` 的入选原件子库，一律保留
- factor_groups.json 及其备份，保留（记录这个 store 里原来有哪些组，只有几 KB）
- cross_yhyb/daily_compare.parquet、柜员游戏01/hmokay4_repair_origin.json 这类小文件保留
- factor_store_hm29_okay_database 等正在被补算任务使用的临时库，不在名单里，绝不触碰

删除方式：ionice -c2 -n7 nice -n 19 rm -f，尽最大努力的最低 IO 优先级，避免和在跑的补算抢盘。

用法：
    python lib/okay_database/cleanup_store_round2.py --dry-run
    python lib/okay_database/cleanup_store_round2.py
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
STATE_PATH = REPORT_DIR / "store_cleanup_round2_state.json"
HTML_PATH = REPORT_DIR / "store_cleanup_round2.html"

BASE = Path("/hdd/user_home_unsafe/chenzongwei")

STORES = [
    "日内CAPM升级版-多因子模型", "urgency_v1", "为人画像", "蹭热点达人", "hot_stock_pool_v1_fix",
    "hm95", "切换时刻贡献", "hot_stock_pool_v2", "hm100", "柜员游戏退火版挂单失衡修复",
    "柜员游戏三层01", "柜员游戏退火版挂单失衡", "hm108", "柜员游戏01", "cross_yhyb",
    "可观测挂单协调骤降", "yupei_dist", "pair_exp_A_4x50", "pair_exp_B_6x50", "pair_exp_C_7x50",
    "pair_exp_D_8x50", "pair_exp_E_4x100", "pair_exp_F_8x25", "long_order_v1",
]
# 第二期追加：hm97 晋升正式版之前有意留的旧副本，已经确认官方 h5 比它新
ARCHIVE = ".hmokay4_archive_hm97_before_nan_promotion/factor_store_柜员游戏01"

KEEP_SUBSTR = ("_accepted",)
KEEP_NAMES = ("factor_groups.json", "factor_groups.json.bak_20260913",
              "daily_compare.parquet", "hmokay4_repair_origin.json")


def human(n: float) -> str:
    n = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB", "PB"):
        if abs(n) < 1024 or unit == "PB":
            return f"{n:,.2f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024
    return f"{n:.2f} PB"


def disk_free() -> int:
    return shutil.disk_usage(BASE).free


def collect(root: Path, store_label: str) -> tuple[list[dict], list[dict]]:
    """走一遍目录，返回（要删的文件、要保留的文件）。只读元数据。"""
    todo, keep = [], []
    for dirpath, _dirnames, filenames in os.walk(root):
        rel = os.path.relpath(dirpath, root)
        if any(s in rel for s in KEEP_SUBSTR):
            for f in filenames:
                p = Path(dirpath) / f
                try:
                    keep.append({"path": str(p), "size": p.stat().st_size, "why": "入选原件"})
                except OSError:
                    pass
            continue
        for f in filenames:
            p = Path(dirpath) / f
            try:
                size = p.stat().st_size
            except OSError:
                continue
            if f in KEEP_NAMES or f.endswith(".json"):
                keep.append({"path": str(p), "size": size, "why": "元数据"})
            else:
                todo.append({"store": store_label, "path": str(p), "size": size, "status": "pending"})
    return todo, keep


def save_state(state: dict) -> None:
    tmp = STATE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=1))
    tmp.replace(STATE_PATH)


def write_html(state: dict) -> None:
    items = state["items"]
    done = [i for i in items if i["status"] == "deleted"]
    failed = [i for i in items if i["status"] == "failed"]
    freed = sum(i["size"] for i in done)
    total = state["total_bytes"]
    pct = freed / total * 100 if total else 0
    now_free = disk_free()

    per: dict[str, dict] = {}
    for i in items:
        s = per.setdefault(i["store"], {"n": 0, "size": 0, "done": 0, "freed": 0})
        s["n"] += 1
        s["size"] += i["size"]
        if i["status"] == "deleted":
            s["done"] += 1
            s["freed"] += i["size"]
    rows = []
    for name, s in sorted(per.items(), key=lambda kv: -kv[1]["size"]):
        p = s["freed"] / s["size"] * 100 if s["size"] else 0
        rows.append(
            "<tr>"
            f"<td class='mono'>{html.escape(name)}</td>"
            f"<td class='num'>{s['done']}/{s['n']}</td>"
            f"<td class='num'>{human(s['size'])}</td>"
            f"<td class='num'>{human(s['freed'])}</td>"
            f"<td><div class='mini'><span style='width:{p:.1f}%'></span></div></td>"
            "</tr>"
        )
    logs = "".join(f"<div class='log'>{html.escape(x)}</div>" for x in state.get("log", [])[-50:])
    status = {"running": "进行中", "done": "已完成", "dry-run": "演练（未删除）"}.get(
        state.get("status"), state.get("status"))

    page = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>factor_store 第二期清理进度</title>
<meta http-equiv="refresh" content="5">
<style>
 body{{margin:0;padding:24px 28px;background:#0f1216;color:#e6e9ef;
      font-family:-apple-system,"PingFang SC","Microsoft YaHei",sans-serif}}
 h1{{font-size:20px;margin:0 0 4px}} h2{{font-size:15px;margin:24px 0 10px;color:#c7cfdb}}
 .sub{{color:#8b93a1;font-size:13px;margin-bottom:18px}}
 .cards{{display:flex;gap:14px;flex-wrap:wrap;margin-bottom:18px}}
 .card{{background:#171b21;border:1px solid #232a33;border-radius:10px;padding:14px 18px;min-width:170px}}
 .card .k{{font-size:12px;color:#8b93a1;margin-bottom:6px}} .card .v{{font-size:20px;font-weight:600}}
 .bar{{height:14px;background:#232a33;border-radius:7px;overflow:hidden;margin:6px 0 22px}}
 .bar span{{display:block;height:100%;background:linear-gradient(90deg,#2e7d6f,#43c59e);width:{pct:.2f}%}}
 table{{width:100%;border-collapse:collapse;font-size:13px;margin-bottom:18px}}
 th,td{{padding:7px 10px;border-bottom:1px solid #232a33;text-align:left}}
 th{{color:#8b93a1;font-weight:500;font-size:12px}}
 td.num{{text-align:right;font-variant-numeric:tabular-nums}} .strong{{color:#43c59e;font-weight:600}}
 .mono{{font-family:ui-monospace,Menlo,monospace;font-size:12px}}
 .mini{{height:8px;background:#232a33;border-radius:4px;overflow:hidden;min-width:90px}}
 .mini span{{display:block;height:100%;background:#43c59e}}
 .log{{font-family:ui-monospace,Menlo,monospace;font-size:11.5px;color:#9aa4b2;line-height:1.55}}
 .scroll{{max-height:420px;overflow:auto;border:1px solid #232a33;border-radius:8px;padding:10px}}
 .badge{{display:inline-block;padding:2px 9px;border-radius:10px;font-size:12px;background:#1d3a33;color:#43c59e}}
 .badge.run{{background:#3a3220;color:#e0b556}}
</style></head><body>
<h1>factor_store 第二期清理进度</h1>
<div class="sub">24 个 hm94~hm110 遗留 store + 1 个隐藏归档 ·
 状态 <span class="badge {'run' if state.get('status')=='running' else ''}">{status}</span> ·
 {datetime.now():%Y-%m-%d %H:%M:%S}（每 5 秒自刷新）</div>

<div class="cards">
 <div class="card"><div class="k">已释放</div><div class="v">{human(freed)}</div></div>
 <div class="card"><div class="k">计划释放</div><div class="v">{human(total)}</div></div>
 <div class="card"><div class="k">文件进度</div><div class="v">{len(done)}/{len(items)}</div></div>
 <div class="card"><div class="k">失败</div><div class="v">{len(failed)}</div></div>
 <div class="card"><div class="k">/hdd 当前可用</div><div class="v">{human(now_free)}</div></div>
</div>
<div class="bar"><span></span></div>
<div class="sub">清理前可用 {human(state['free_before'])} · 计划保留（入选原件与元数据）{len(state.get('keep',[]))} 个文件、
{human(sum(k['size'] for k in state.get('keep',[])))}</div>

<h2>按 store 的进度</h2>
<table><tr><th>store 目录</th><th>文件</th><th>计划释放</th><th>已释放</th><th>进度</th></tr>{''.join(rows)}</table>

<h2>执行日志</h2>
<div class="scroll">{logs or "<div class='log'>等待开始…</div>"}</div>
</body></html>"""
    tmp = HTML_PATH.with_suffix(".tmp")
    tmp.write_text(page)
    tmp.replace(HTML_PATH)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    roots = [(f"factor_store_{s}", BASE / f"factor_store_{s}") for s in STORES]
    roots.append((ARCHIVE, BASE / ARCHIVE))

    items, keep, scan_log = [], [], []
    for label, root in roots:
        if not root.is_dir():
            scan_log.append(f"跳过（不存在）{root}")
            continue
        todo, kept = collect(root, label)
        items.extend(todo)
        keep.extend(kept)
        scan_log.append(f"扫描 {label}：待删 {len(todo)} 个文件 {human(sum(t['size'] for t in todo))}，"
                        f"保留 {len(kept)} 个")
    # 大文件先删：既能最快看到空间变化，也让可能卡顿的大文件早点过去
    items.sort(key=lambda i: -i["size"])
    total = sum(i["size"] for i in items)

    state = {
        "started": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": "dry-run" if args.dry_run else "running",
        "free_before": disk_free(),
        "free_after": None,
        "items": items,
        "keep": keep,
        "total_bytes": total,
        "log": scan_log,
    }
    log = state["log"]
    log.append(f"[{datetime.now():%H:%M:%S}] 待删 {len(items)} 个文件，合计 {human(total)}；"
               f"保留 {len(keep)} 个文件 {human(sum(k['size'] for k in keep))}")
    log.append(f"[{datetime.now():%H:%M:%S}] 清理前 /hdd 可用 {human(state['free_before'])}")
    save_state(state)
    write_html(state)

    if args.dry_run:
        state["status"] = "done"
        state["free_after"] = disk_free()
        save_state(state)
        write_html(state)
        print("\n".join(log))
        return 0

    freed = 0
    for idx, item in enumerate(items, 1):
        p = item["path"]
        try:
            if not os.path.isfile(p):
                item["status"] = "deleted"
            else:
                r = subprocess.run(
                    ["ionice", "-c2", "-n7", "nice", "-n", "19", "rm", "-f", "--", p],
                    capture_output=True, text=True,
                )
                if r.returncode == 0:
                    item["status"] = "deleted"
                    freed += item["size"]
                else:
                    item["status"] = "failed"
                    log.append(f"[{datetime.now():%H:%M:%S}] 删除失败 {p} :: {(r.stderr or '').strip()[:160]}")
        except Exception as exc:  # noqa: BLE001
            item["status"] = "failed"
            log.append(f"[{datetime.now():%H:%M:%S}] 异常 {p} :: {str(exc)[:160]}")
        if idx % 20 == 0 or item["size"] > 50 * 1024 ** 3:
            log.append(f"[{datetime.now():%H:%M:%S}] {idx}/{len(items)} 已删 {human(freed)} :: {p}")
        if idx % 10 == 0 or item["size"] > 50 * 1024 ** 3:
            save_state(state)
            write_html(state)

    # 清掉空目录（只删空目录，rmdir 不会碰还有东西的目录）
    removed_dirs = 0
    for label, root in reversed(roots):
        if not root.is_dir():
            continue
        for dirpath, dirnames, filenames in os.walk(root, topdown=False):
            if dirpath == str(root):
                continue
            try:
                os.rmdir(dirpath)
                removed_dirs += 1
            except OSError:
                pass
        try:
            os.rmdir(root)
            removed_dirs += 1
        except OSError:
            log.append(f"[{datetime.now():%H:%M:%S}] {label} 还有保留文件，目录留着")
    log.append(f"[{datetime.now():%H:%M:%S}] 清掉空目录 {removed_dirs} 个")

    state["status"] = "done"
    state["finished"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    state["free_after"] = disk_free()
    log.append(f"[{datetime.now():%H:%M:%S}] 完成。已释放 {human(freed)}；"
               f"当前 /hdd 可用 {human(state['free_after'])}（清理前 {human(state['free_before'])}）")
    save_state(state)
    write_html(state)
    print("\n".join(log[-8:]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
