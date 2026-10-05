"""把改写、补算、清 colblk、写因子库四件事汇总成一个实时进度页。

输出 reports/okay_database/index.html。
页面每 10 秒自刷新；如果能读到同目录的 JSON（用本地 http 打开时）就顺手做实时更新，
用 file:// 打开时退回页面里内嵌的快照，刷新一次看一次。

用法：python lib/okay_database/render_dashboard.py
"""
from __future__ import annotations

import html
import json
from datetime import datetime
from pathlib import Path

REPO = Path("/home/chenzongwei/rust_pyfunc")
REPORT_DIR = REPO / "reports/okay_database"
STATE = REPORT_DIR / "program_state.json"
CLEAN = REPORT_DIR / "colblk_cleanup_state.json"
CLEAN2 = REPORT_DIR / "store_cleanup_round2_state.json"
QLOG = REPORT_DIR / "queue_log.jsonl"
OUT = REPORT_DIR / "index.html"

LIBRARY = Path("/nas197/user_home/chenzongwei/factor_library")
STORE_ROOT = Path("/hdd/user_home_unsafe/chenzongwei")

STEP_NAMES = [("step_rewrite", "改写"), ("step_submit", "提交计算"),
              ("step_library", "写因子库"), ("step_cleanup", "清 colblk")]
BADGE = {
    "done": ("已完成", "ok"), "pending": ("未开始", "idle"), "running": ("进行中", "run"),
    "failed": ("失败", "bad"), "cancelled": ("已取消", "bad"), "ready": ("待提交", "run"),
}


def human(n: float) -> str:
    n = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB", "PB"):
        if abs(n) < 1024 or unit == "PB":
            return f"{n:,.2f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024
    return f"{n:.2f} PB"


def badge(value: str) -> str:
    text, cls = BADGE.get(value, (value, "idle"))
    return f"<span class='badge {cls}'>{html.escape(text)}</span>"


def esc(v) -> str:
    return html.escape(str(v if v is not None else "-"))


def load_json(path: Path, default):
    if path.is_file():
        try:
            return json.loads(path.read_text())
        except Exception:  # noqa: BLE001
            return default
    return default


def main() -> int:
    state = load_json(STATE, {"series": [], "count": 0})
    clean = load_json(CLEAN, {})
    clean2 = load_json(CLEAN2, {})
    series = state.get("series", [])

    total = len(series)
    n_rewrite = sum(1 for s in series if s.get("step_rewrite") == "done")
    n_submit = sum(1 for s in series if s.get("step_submit") == "done")
    n_lib = sum(1 for s in series if s.get("step_library") == "done")
    n_clean = sum(1 for s in series if s.get("step_cleanup") == "done")
    n_fail = sum(1 for s in series if str(s.get("step_submit")) in ("failed", "cancelled"))

    # 因子库里实际有多少文件
    lib_counts = {}
    for label, sub in (("base", "factor_base"), ("gap5", "hf_data"),
                       ("gap1", "hf_data1"), ("db", "dailybank")):
        d = LIBRARY / sub
        lib_counts[label] = sum(1 for p in d.glob("*/*.h5")) if d.is_dir() else 0

    # 本次补算还在 HDD 上的 colblk
    live = {}
    if STORE_ROOT.is_dir():
        for d in STORE_ROOT.glob("factor_store_hm*_okay_database"):
            live[d.name] = len(list(d.rglob("*.colblk")))

    rows = []
    for s in sorted(series, key=lambda x: x["num"]):
        cells = "".join(f"<td>{badge(str(s.get(k, 'pending')))}</td>" for k, _ in STEP_NAMES)
        task = s.get("task_id")
        rows.append(
            "<tr>"
            f"<td class='mono'>{esc(s['series'])}</td>"
            f"<td class='mono'>{esc(s.get('ver'))}</td>"
            f"<td>{esc(s.get('source'))}</td>"
            f"<td class='num'>{s.get('n_base')}</td>"
            f"{cells}"
            f"<td class='mono'>{esc(task)}</td>"
            f"<td class='note'>{esc(s.get('note'))}</td>"
            "</tr>"
        )

    # 清理进度
    citems = clean.get("items", [])
    c_done = [i for i in citems if i.get("status") == "deleted"]
    c_todo = [i for i in citems if i.get("status") != "skipped"]
    c_freed = sum(i.get("size", 0) for i in c_done)
    c_total = sum(i.get("size", 0) for i in citems if i.get("status") != "skipped")
    c_pct = (c_freed / c_total * 100) if c_total else 0

    def disk_line(label, st):
        if not st:
            return ""
        return (f"<tr><td>{label}</td><td class='mono'>{esc(st.get('time'))}</td>"
                f"<td class='num'>{human(st.get('total', 0))}</td>"
                f"<td class='num'>{human(st.get('used', 0))}</td>"
                f"<td class='num strong'>{human(st.get('free', 0))}</td></tr>")

    before = clean.get("disk_before")
    after = clean.get("disk_after")
    delta = human(after["free"] - before["free"]) if (before and after) else "—"

    # 队列日志
    qlog = []
    if QLOG.is_file():
        for line in QLOG.read_text().splitlines()[-25:]:
            try:
                qlog.append(json.loads(line))
            except Exception:  # noqa: BLE001
                pass
    qlog_html = "".join(
        f"<div class='log'>{esc(r.get('time'))} · {esc(r.get('series'))} · "
        f"{esc(r.get('event'))} · {esc(r.get('status') or r.get('task_id'))}"
        f"{' · 因子库 ' + esc(r.get('files')) if r.get('files') else ''}</div>"
        for r in qlog
    ) or "<div class='log'>还没有队列记录</div>"

    clogs = "".join(f"<div class='log'>{esc(x)}</div>" for x in clean.get("log", [])[-12:])

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    snapshot = json.dumps(
        {"generated": now, "total": total, "rewrite": n_rewrite, "submit": n_submit,
         "library": n_lib, "cleanup": n_clean},
        ensure_ascii=False,
    )

    page = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>hm0~hm92 因子库改写与补算进度</title>
<meta http-equiv="refresh" content="10">
<style>
 body{{margin:0;padding:24px 28px;background:#0f1216;color:#e6e9ef;
      font-family:-apple-system,"PingFang SC","Microsoft YaHei",sans-serif}}
 h1{{font-size:20px;margin:0 0 4px}} h2{{font-size:15px;margin:26px 0 10px;color:#c7cfdb}}
 .sub{{color:#8b93a1;font-size:13px;margin-bottom:18px}}
 .cards{{display:flex;gap:14px;flex-wrap:wrap;margin-bottom:18px}}
 .card{{background:#171b21;border:1px solid #232a33;border-radius:10px;padding:14px 18px;min-width:150px}}
 .card .k{{font-size:12px;color:#8b93a1;margin-bottom:6px}} .card .v{{font-size:20px;font-weight:600}}
 .bar{{height:12px;background:#232a33;border-radius:6px;overflow:hidden;margin:4px 0 16px}}
 .bar span{{display:block;height:100%;background:linear-gradient(90deg,#2e7d6f,#43c59e)}}
 table{{width:100%;border-collapse:collapse;font-size:12.5px;margin-bottom:18px}}
 th,td{{padding:6px 9px;border-bottom:1px solid #232a33;text-align:left;vertical-align:top}}
 th{{color:#8b93a1;font-weight:500;font-size:12px;position:sticky;top:0;background:#0f1216}}
 td.num{{text-align:right;font-variant-numeric:tabular-nums}} .strong{{color:#43c59e;font-weight:600}}
 .mono{{font-family:ui-monospace,Menlo,monospace;font-size:11.5px}}
 .note{{color:#8b93a1;font-size:11.5px;max-width:420px}}
 .badge{{display:inline-block;padding:1px 8px;border-radius:9px;font-size:11.5px}}
 .badge.ok{{background:#1d3a33;color:#43c59e}} .badge.run{{background:#3a3220;color:#e0b556}}
 .badge.idle{{background:#242a33;color:#8b93a1}} .badge.bad{{background:#3d2126;color:#e07a86}}
 .log{{font-family:ui-monospace,Menlo,monospace;font-size:11.5px;color:#9aa4b2;line-height:1.55}}
 .scroll{{max-height:560px;overflow:auto;border:1px solid #232a33;border-radius:8px}}
</style></head><body>
<h1>hm0~hm92 因子库改写与补算进度</h1>
<div class="sub">共 {total} 个系列 · 每个系列四步：Rust 改写 → 提交计算（2015-01-05 ~ 2026-09-18，n_jobs=400，同时只跑一个）→ 写入因子库 → 删除 colblk ·
更新于 {now}（每 10 秒自刷新）</div>

<div class="cards">
 <div class="card"><div class="k">已完成改写</div><div class="v">{n_rewrite} / {total}</div></div>
 <div class="card"><div class="k">已算完</div><div class="v">{n_submit} / {total}</div></div>
 <div class="card"><div class="k">已入因子库</div><div class="v">{n_lib} / {total}</div></div>
 <div class="card"><div class="k">colblk 已清</div><div class="v">{n_clean} / {total}</div></div>
 <div class="card"><div class="k">失败/取消</div><div class="v">{n_fail}</div></div>
</div>
<div class="bar"><span style="width:{(n_rewrite / total * 100) if total else 0:.1f}%"></span></div>

<h2>因子库落点（/nas197/user_home/chenzongwei/factor_library）</h2>
<table><tr><th>层</th><th>目录</th><th>h5 文件数</th></tr>
<tr><td>base 原值</td><td class="mono">factor_base/&lt;系列&gt;_base</td><td class="num">{lib_counts['base']}</td></tr>
<tr><td>gap5 最终</td><td class="mono">hf_data/&lt;系列&gt;</td><td class="num">{lib_counts['gap5']}</td></tr>
<tr><td>gap1 最终</td><td class="mono">hf_data1/&lt;系列&gt;</td><td class="num">{lib_counts['gap1']}</td></tr>
<tr><td>dailybank</td><td class="mono">dailybank/&lt;系列&gt;</td><td class="num">{lib_counts['db']}</td></tr>
</table>

<h2>colblk 清理（hm94~hm110 遗留）</h2>
<div class="cards">
 <div class="card"><div class="k">已释放</div><div class="v">{human(c_freed)}</div></div>
 <div class="card"><div class="k">计划释放</div><div class="v">{human(c_total)}</div></div>
 <div class="card"><div class="k">文件</div><div class="v">{len(c_done)} / {len(c_todo)}</div></div>
 <div class="card"><div class="k">净增可用</div><div class="v">{delta}</div></div>
 <div class="card"><div class="k">状态</div><div class="v">{esc(clean.get('status', '未开始'))}</div></div>
</div>
<div class="bar"><span style="width:{c_pct:.2f}%"></span></div>
<table><tr><th>时点</th><th>时刻</th><th>总空间</th><th>已用</th><th>剩余可用</th></tr>
{disk_line('清理前', before)}{disk_line('清理后', after)}</table>
<div class="scroll"><div style="padding:10px">{clogs}</div></div>

<h2>第二期清理（factor_store 里剩下的 proj 副本 + hm97 归档）</h2>
<div class="cards">
 <div class="card"><div class="k">已释放</div><div class="v">{human(clean2.get("total_bytes", 0))}</div></div>
 <div class="card"><div class="k">文件</div><div class="v">{sum(1 for i in clean2.get("items", []) if i.get("status") == "deleted")} / {len(clean2.get("items", []))}</div></div>
 <div class="card"><div class="k">保留的入选原件</div><div class="v">{len(clean2.get("keep", []))} 个</div></div>
 <div class="card"><div class="k">状态</div><div class="v">{esc(clean2.get("status", "未开始"))}</div></div>
</div>
<div class="sub">清理前可用 {human(clean2.get("free_before", 0))} · 清理后可用 {human(clean2.get("free_after", 0) or 0)}
（btrfs 的空间回收是延迟的，收尾值在脚本跑完后另行补齐）</div>
<table><tr><th>时点</th><th>可用空间</th></tr>
<tr><td>第二期清理前</td><td class="num">{human(clean2.get("free_before", 0))}</td></tr>
<tr><td>第二期清理后</td><td class="num strong">{human(clean2.get("free_after", 0) or 0)}</td></tr></table>

<h2>本次补算的临时 colblk（应当逐步清空）</h2>
<table><tr><th>临时目录</th><th>残留 colblk</th></tr>
{''.join(f"<tr><td class='mono'>{esc(k)}</td><td class='num'>{v}</td></tr>" for k, v in sorted(live.items())) or "<tr><td colspan=2>暂无</td></tr>"}
</table>

<h2>队列记录</h2>
<div class="scroll"><div style="padding:10px">{qlog_html}</div></div>

<h2>全部 {total} 个系列</h2>
<div class="scroll"><table>
<tr><th>系列</th><th>ver</th><th>数据来源</th><th>base 因子数</th>
<th>改写</th><th>提交</th><th>入库</th><th>清 colblk</th><th>任务</th><th>备注</th></tr>
{''.join(rows)}
</table></div>

<script>
// 快照：用 file:// 打开时看的就是它；用本地 http 打开时会自动换成同目录 JSON 的实时值。
window.__SNAPSHOT__ = {snapshot};
</script>
</body></html>"""
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".tmp")
    tmp.write_text(page)
    tmp.replace(OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
