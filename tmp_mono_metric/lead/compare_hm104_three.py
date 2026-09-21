"""对比 hm104 原版 / ssm2 / ssm3 的 gap5 入选名单，生成分布图 HTML。

指标统一从 ssm3 那轮的全量完整回测表取（三批都是同一套引擎配置，表内容一致）。
|RankIC| = |IC_mean|；多头超额收益 = hedge_annualized_return（多头组年化 − 指数年化）。
"""
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

R0 = Path("/nas197/user_home_unsafe/chenzongwei/hm104_tail_v4")
R2 = Path("/nas197/user_home_unsafe/chenzongwei/hm104_ssm2_tail_v4")
R3 = Path("/nas197/user_home_unsafe/chenzongwei/hm104_ssm3_tail_v4")
OUT = Path("/home/chenzongwei/rust_pyfunc/tmp_mono_metric/lead/hm104_selection_compare.html")

GROUPS = [
    ("原版 hm104", "无门槛 · 纯 |IC| 排序", R0, "#1e4e79", "#eff5fb"),
    ("hm104_ssm2", "SSM 门槛 0.5 + |IC| 排序", R2, "#b8450f", "#fff5ee"),
    ("hm104_ssm3", "SSM 门槛 0.9 + |IC| 排序", R3, "#186b43", "#eef9f3"),
]


SHORT = {"原版 hm104": "原版", "hm104_ssm2": "ssm2", "hm104_ssm3": "ssm3"}


def load_selected(R):
    return pd.read_parquet(R / "selected" / "gap5_selected.parquet")["factor_name"].tolist()


full = pd.read_parquet(R3 / "metrics" / "summary_neu_gap5_candidates.parquet")
full["absic"] = full.IC_mean.abs()
idx = full.set_index("factor_name")

lists = [(name, desc, load_selected(R), c, bg) for name, desc, R, c, bg in GROUPS]
sets = [set(x[2]) for x in lists]

# ---- 重叠统计 ----
n = len(lists)
pair = [[len(sets[i] & sets[j]) for j in range(n)] for i in range(n)]
all3 = len(sets[0] & sets[1] & sets[2])
any_cnt = len(sets[0] | sets[1] | sets[2])
print("名单长度:", [len(x[2]) for x in lists])
print("两两重合:", pair)
print("三组共有:", all3, " 合计不重复:", any_cnt)


def stats(vals):
    v = np.asarray([x for x in vals if np.isfinite(x)], dtype=float)
    return dict(n=len(v), mn=float(v.min()), q1=float(np.percentile(v, 25)),
                med=float(np.median(v)), q3=float(np.percentile(v, 75)),
                mx=float(v.max()), mean=float(v.mean()), pts=v)


series = {}
for name, desc, lst, col, bg in lists:
    sub = idx.reindex(lst)
    series[name] = dict(
        ic=stats(sub["absic"].to_numpy()),
        hedge=stats(sub["hedge_annualized_return"].to_numpy()),
        ls=stats(sub["annualized_return"].to_numpy()),
        sharpe=stats(sub["sharpe_ratio"].to_numpy()),
    )

for name, desc, lst, col, bg in lists:
    s = series[name]
    print(f"{name:14s} |IC| 中位 {s['ic']['med']:.4f}  多头超额中位 {s['hedge']['med']*100:6.2f}%  "
          f"多空年化中位 {s['ls']['med']*100:6.2f}%  夏普中位 {s['sharpe']['med']:5.2f}")


# ---------------------------------------------------------------- SVG 箱线图
def box_panel(key, title, fmt, ylabel, width=760, height=430, pad_l=76, pad_r=24,
              pad_t=44, pad_b=56, y_min=None, y_max=None):
    vals = [series[nm][key] for nm, *_ in lists]
    lo = min(v["mn"] for v in vals) if y_min is None else y_min
    hi = max(v["mx"] for v in vals) if y_max is None else y_max
    span = hi - lo or 1.0
    lo -= span * 0.10
    hi += span * 0.12
    span = hi - lo

    def Y(v):
        return pad_t + (hi - v) / span * (height - pad_t - pad_b)

    plot_w = width - pad_l - pad_r
    n_g = len(lists)
    slot = plot_w / n_g
    bw = min(96.0, slot * 0.46)

    p = [f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" aria-label="{html.escape(title)}">']
    p.append(f'<text x="{pad_l}" y="22" font-size="15" font-weight="650" fill="#1c2128">{html.escape(title)}</text>')
    # 网格 + y 轴刻度
    nt = 5
    for k in range(nt + 1):
        v = lo + span * k / nt
        y = Y(v)
        p.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{width-pad_r}" y2="{y:.1f}" stroke="#eef1f5"/>')
        p.append(f'<text x="{pad_l-10}" y="{y+4:.1f}" font-size="11.5" fill="#5c6672" text-anchor="end">{fmt(v)}</text>')
    p.append(f'<text x="14" y="{pad_t+ (height-pad_t-pad_b)/2:.0f}" font-size="11.5" fill="#5c6672" '
             f'transform="rotate(-90 14 {pad_t+(height-pad_t-pad_b)/2:.0f})" text-anchor="middle">{html.escape(ylabel)}</text>')

    rng = np.random.default_rng(7)
    for gi, (name, desc, lst, col, bgc) in enumerate(lists):
        v = vals[gi]
        cx = pad_l + slot * (gi + 0.5)
        x0, x1 = cx - bw / 2, cx + bw / 2
        # 箱体
        p.append(f'<rect x="{x0:.1f}" y="{Y(v["q3"]):.1f}" width="{bw:.1f}" '
                 f'height="{max(1.0, Y(v["q1"])-Y(v["q3"])):.1f}" fill="{bgc}" stroke="{col}" stroke-width="1.6"/>')
        # 中位线
        p.append(f'<line x1="{x0:.1f}" y1="{Y(v["med"]):.1f}" x2="{x1:.1f}" y2="{Y(v["med"]):.1f}" '
                 f'stroke="{col}" stroke-width="2.6"/>')
        # 均值点
        p.append(f'<circle cx="{cx:.1f}" cy="{Y(v["mean"]):.1f}" r="3.4" fill="none" stroke="{col}" stroke-width="1.8"/>')
        # 须
        p.append(f'<line x1="{cx:.1f}" y1="{Y(v["q3"]):.1f}" x2="{cx:.1f}" y2="{Y(v["mx"]):.1f}" stroke="{col}" stroke-width="1.3"/>')
        p.append(f'<line x1="{cx:.1f}" y1="{Y(v["q1"]):.1f}" x2="{cx:.1f}" y2="{Y(v["mn"]):.1f}" stroke="{col}" stroke-width="1.3"/>')
        p.append(f'<line x1="{cx-bw*0.18:.1f}" y1="{Y(v["mx"]):.1f}" x2="{cx+bw*0.18:.1f}" y2="{Y(v["mx"]):.1f}" stroke="{col}" stroke-width="1.3"/>')
        p.append(f'<line x1="{cx-bw*0.18:.1f}" y1="{Y(v["mn"]):.1f}" x2="{cx+bw*0.18:.1f}" y2="{Y(v["mn"]):.1f}" stroke="{col}" stroke-width="1.3"/>')
        # 抖动散点
        jx = rng.uniform(-bw * 0.30, bw * 0.30, len(v["pts"]))
        for j, val in zip(jx, v["pts"]):
            p.append(f'<circle cx="{cx+j:.1f}" cy="{Y(val):.1f}" r="2.5" fill="{col}" fill-opacity="0.55"/>')
        # 标签
        p.append(f'<text x="{cx:.1f}" y="{height-pad_b+20:.0f}" font-size="12.5" font-weight="600" '
                 f'fill="#1c2128" text-anchor="middle">{html.escape(name)}</text>')
        p.append(f'<text x="{cx:.1f}" y="{height-pad_b+37:.0f}" font-size="10.5" '
                 f'fill="#5c6672" text-anchor="middle">中位 {fmt(v["med"])}</text>')
    p.append('</svg>')
    return "".join(p)


ic_panel = box_panel("ic", "|RankIC| 分布（35 个因子）", lambda v: f"{v:.3f}", "|RankIC|", y_min=0.0)
hd_panel = box_panel("hedge", "多头超额收益分布（年化，35 个因子）",
                     lambda v: f"{v*100:.0f}%", "多头超额收益")

# ---------------------------------------------------------------- 重叠矩阵 HTML
def venn_row(i, j):
    if i == j:
        return f'<td class="diag">{len(sets[i])}</td>'
    return f'<td>{pair[i][j]}</td>'


rows = []
for i, (name, desc, *_rest) in enumerate(lists):
    cells = "".join(venn_row(i, j) for j in range(n))
    rows.append(f'<tr><th>{SHORT[name]}</th>{cells}</tr>')
head = "".join(f"<th>{SHORT[nm]}</th>" for nm, *_ in lists)

common_all = sorted(sets[0] & sets[1] & sets[2])
common_list_html = "".join(f"<li>{html.escape(x)}</li>" for x in common_all)

summary_rows = []
for name, desc, lst, col, bgc in lists:
    s = series[name]
    short = {"原版 hm104": "原版", "hm104_ssm2": "ssm2 (0.5)", "hm104_ssm3": "ssm3 (0.9)"}[name]
    summary_rows.append(
        f'<tr><th>{html.escape(short)}</th>'
        f'<td>{s["ic"]["med"]:.4f}</td><td>{s["hedge"]["med"]*100:.2f}%</td>'
        f'<td>{s["ls"]["med"]*100:.2f}%</td><td>{s["sharpe"]["med"]:.2f}</td></tr>'
    )

doc = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>hm104 三版入选因子对比</title>
<style>
 :root{{--bg:#f6f7f9;--panel:#fff;--ink:#1c2128;--muted:#5c6672;--line:#dde3ea;--line-soft:#eef1f5}}
 *{{box-sizing:border-box}} html,body{{margin:0}}
 body{{background:var(--bg);color:var(--ink);font:14px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif}}
 header{{padding:22px 28px 8px}}
 h1{{margin:0 0 6px;font-size:21px;font-weight:650}}
 header p{{margin:0;color:var(--muted);font-size:13px}}
 .wrap{{display:grid;grid-template-columns:minmax(0,1fr) 470px;gap:22px;padding:14px 28px 44px;align-items:start}}
 @media(max-width:1240px){{.wrap{{grid-template-columns:minmax(0,1fr)}}}}
 .charts{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}}
 @media(max-width:1500px){{.charts{{grid-template-columns:minmax(0,1fr)}}}}
 .card{{background:var(--panel);border:1.5px solid var(--line);border-radius:10px;padding:14px 16px}}
 .card h2{{margin:0 0 10px;font-size:14px;font-weight:650}}
 aside{{position:sticky;top:16px;align-self:start;max-height:calc(100vh - 32px);overflow-y:auto}}
 table{{width:100%;border-collapse:collapse;font-size:12.5px}}
 th,td{{padding:6px 7px;border-bottom:1px solid var(--line-soft);text-align:right;white-space:nowrap;font-size:12px}}
 th:first-child,td:first-child{{text-align:left}}
 th:first-child,td:first-child{{text-align:left;white-space:normal}}
 thead th{{color:var(--muted);font-weight:600;border-bottom:1px solid var(--line)}}
 td.diag{{background:#f2f4f7;font-weight:650}}
 ul{{margin:0;padding-left:18px;font-size:12px;color:var(--muted)}}
 li{{margin:2px 0;word-break:break-all}}
 .legend{{display:flex;gap:16px;flex-wrap:wrap;font-size:12.5px;color:var(--muted);margin-top:8px}}
 .legend i{{display:inline-block;width:11px;height:11px;border-radius:3px;margin-right:6px;vertical-align:-1px}}
</style></head><body>
<header>
  <h1>hm104 三版入选因子对比（gap5，各 35 个）</h1>
  <p>原版走 IC-only 模式没有收益指标，所以三批的 |RankIC| 与多头超额收益统一从同一套全量完整回测表取，口径一致。</p>
  <div class="legend">
    <span><i style="background:#1e4e79"></i>原版 hm104（无门槛）</span>
    <span><i style="background:#b8450f"></i>hm104_ssm2（门槛 0.5）</span>
    <span><i style="background:#186b43"></i>hm104_ssm3（门槛 0.9）</span>
  </div>
</header>
<div class="wrap">
  <div class="charts">
    <div class="card">{ic_panel}</div>
    <div class="card">{hd_panel}</div>
  </div>
  <aside>
    <div class="card">
      <h2>名单重合（gap5，各 35 个）</h2>
      <table><thead><tr><th>∩</th>{head}</tr></thead>
      <tbody>{''.join(rows)}</tbody></table>
      <p style="margin:10px 0 0;font-size:12.5px;color:var(--muted)">
        三组共有 <b>{all3}</b> 个；三组合计不重复 <b>{any_cnt}</b> 个。</p>
    </div>
    <div class="card">
      <h2>中位数对照</h2>
      <table><thead><tr><th>版本</th><th>|RankIC|</th><th>多头超额</th><th>多空年化</th><th>夏普</th></tr></thead>
      <tbody>{''.join(summary_rows)}</tbody></table>
    </div>
    <div class="card">
      <h2>三组共有的因子（{all3} 个）</h2>
      <ul>{common_list_html}</ul>
    </div>
  </aside>
</div>
</body></html>"""

OUT.write_text(doc, encoding="utf-8")
print("written:", OUT)
