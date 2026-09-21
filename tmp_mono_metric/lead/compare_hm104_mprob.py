"""hm104 上对比 mprob 版与 ssm2 / ssm3 / 原版的入选名单。

指标统一从 hm104_mprob 那轮的完整回测表取（同一套引擎配置、同一批因子，
所有名单的 |RankIC| 与多头超额收益都在这一张表里，口径一致）。

名单来源：
- 原版 hm104        : hm104_tail_v4/selected/gap5_selected.parquet（IC-only，无门槛）
- hm104_ssm2        : SSM >= 0.5 + |IC| 排序
- hm104_ssm3        : SSM >= 0.9 + |IC| 排序
- hm104_mprob(门槛) : 本轮引擎实际入选（MPROB >= 0.315 + |IC| 排序 + 四通道合并去重）
- mprob-only top35  : 只用 MPROB 降序排序 + 相关性去重，取前 35
- |IC| top35 / 多头超额 top35 : 全表直接排序，作为参照集

去重说明：greedy 相关性去重是按顺序扫描、逐个与已选比较，所以「前 35 个」只取决于
候选序列的前缀。这里对候选序列先截前 PREFIX 个再去重，结果与在完整序列上去重后取前 35
逐位相同（PREFIX 远大于 35，足够），但快几个数量级。
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import rust_pyfunc as rp

NAS = Path("/nas197/user_home_unsafe/chenzongwei")
R_M = NAS / "hm104_mprob_tail_v4"
R0 = NAS / "hm104_tail_v4"
R2 = NAS / "hm104_ssm2_tail_v4"
R3 = NAS / "hm104_ssm3_tail_v4"
CUT_NUM = 35
PREFIX = 6000


def load_wide(R, stem, keep=None):
    m = np.load(R / "ic_ts" / f"{stem}.npy", mmap_mode="r")
    names = json.loads((R / "ic_ts" / f"{stem}_names.json").read_text(encoding="utf-8"))
    pos = {n: i for i, n in enumerate(names)}
    if keep is None:
        return pd.DataFrame(np.asarray(m), columns=names)
    idx = np.array([pos[n] for n in keep if n in pos])
    return pd.DataFrame(np.asarray(m[:, idx]), columns=[n for n in keep if n in pos])


def corr_select(ordered_names, ic_wide, th, prefix=PREFIX):
    """前缀截断的 greedy 相关性去重（结果与完整序列上前 35 个相同）。"""
    ordered_names = list(ordered_names)[:prefix]
    if not ordered_names:
        return []
    o = np.ascontiguousarray(ic_wide.loc[:, ordered_names].to_numpy(dtype=np.float32, copy=False).T)
    return [ordered_names[i] for i in rp.tail_v2_select_by_ic_corr_abs_f32(o, float(th))]


full = pd.read_parquet(R_M / "metrics" / "summary_neu_gap5_candidates.parquet")
full["absic"] = full.IC_mean.abs()
pool_names = full[full.ratio_mean >= 0.5].factor_name.tolist()
print(f"hm104_mprob 全表 {full.shape}，MPROB 非空 {np.isfinite(full.MPROB).sum()}，"
      f"覆盖 >=0.5 的池子 {len(pool_names)} 个")
ic_neu = load_wide(R_M, "ic_neu_gap5", keep=pool_names)
print(f"ic_neu_gap5 子表 {ic_neu.shape}")

# ---- 交叉校验：mprob 轮与 ssm3 轮的既有列是否一致 ----
try:
    f3 = pd.read_parquet(R3 / "metrics" / "summary_neu_gap5_candidates.parquet")
    j = full.set_index("factor_name")[["IC_mean", "hedge_annualized_return"]].join(
        f3.set_index("factor_name")[["IC_mean", "hedge_annualized_return"]],
        lsuffix="_m", rsuffix="_s", how="inner")
    print(f"[交叉校验] 与 ssm3 轮共有 {len(j)} 个因子，"
          f"|ΔIC|max={np.nanmax(np.abs(j.IC_mean_m - j.IC_mean_s)):.3e}，"
          f"|Δ多头超额|max={np.nanmax(np.abs(j.hedge_annualized_return_m - j.hedge_annualized_return_s)):.3e}")
except Exception as e:  # noqa: BLE001
    print(f"[交叉校验] 跳过：{e}")

# ---- 名单 ----
lists: dict[str, list[str]] = {}


def add(name, names_):
    lists[name] = list(names_)
    print(f"  {name:<30} {len(names_)} 个")


print("\n[名单]")
add("原版 hm104(IC-only)", pd.read_parquet(R0 / "selected" / "gap5_selected.parquet").factor_name)
add("hm104_ssm2(SSM>=0.5)", pd.read_parquet(R2 / "selected" / "gap5_selected.parquet").factor_name)
add("hm104_ssm3(SSM>=0.9)", pd.read_parquet(R3 / "selected" / "gap5_selected.parquet").factor_name)
sel_m = R_M / "selected" / "gap5_selected.parquet"
if sel_m.exists():
    add("hm104_mprob(MPROB>=0.315)", pd.read_parquet(sel_m).factor_name)
else:
    print("  hm104_mprob(MPROB>=0.315)  —— 引擎选择阶段尚未落盘，本次跳过")

for th in (0.5, 0.8):
    pool = full[full.ratio_mean >= 0.5].sort_values("MPROB", ascending=False)
    add(f"mprob-only top35(去重{th})", corr_select(pool.factor_name.tolist(), ic_neu, th)[:CUT_NUM])

add("|IC| top35", full.sort_values("absic", ascending=False).factor_name.head(CUT_NUM))
add("多头超额 top35", full.sort_values("hedge_annualized_return", ascending=False).factor_name.head(CUT_NUM))

# ---- 每个名单的指标 ----
rows = []
for name, lst in lists.items():
    sub = full[full.factor_name.isin(lst)]
    if not len(sub):
        continue
    rows.append(dict(
        名单=name, n=len(sub),
        IC中位=sub.absic.median(), IC均值=sub.absic.mean(), IC最小=sub.absic.min(),
        多头超额中位=sub.hedge_annualized_return.median(),
        多头超额均值=sub.hedge_annualized_return.mean(),
        多头超额最小=sub.hedge_annualized_return.min(),
        多空年化中位=sub.annualized_return.median(),
        夏普中位=sub.sharpe_ratio.median(),
        MPROB中位=sub.MPROB.median(), SSM中位=sub.SSM.median(),
    ))
tab = pd.DataFrame(rows)
pd.set_option("display.width", 250)
print("\n" + "=" * 130)
print("[各名单的 |RankIC| 与多头超额收益]（多头超额 = hedge_annualized_return = 多头组年化 − 指数年化）")
print(tab.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

# ---- 两两重合 ----
KEYS = list(lists)
sets = {k: set(v) for k, v in lists.items()}
short = {k: k.split("(")[0][:16] for k in KEYS}
print("\n" + "=" * 130)
print("[两两重合的因子个数]")
print(f"{'':<20}" + "".join(f"{short[k]:>18}" for k in KEYS))
for a in KEYS:
    print(f"{short[a]:<20}" + "".join(f"{len(sets[a] & sets[b]):>18}" for b in KEYS))

print("\n[与 mprob-only top35(去重0.5) 的重合]")
base = sets.get("mprob-only top35(去重0.5)", set())
for k in KEYS:
    if not k.startswith("mprob-only"):
        print(f"  {k:<32} {len(base & sets[k]):>3} / {len(base)}")

# ---- MPROB 排名深度扫描 ----
print("\n" + "=" * 130)
print("[MPROB 排名深度扫描]（取 MPROB 最高的前 N 个 → 相关性去重 0.5 → 最多 35 个）")
scan = []
for n_ in (20, 35, 50, 100, 200, 500, 1000, 2000, 5000):
    pool = full[full.ratio_mean >= 0.5].sort_values("MPROB", ascending=False).head(n_)
    chosen = corr_select(pool.factor_name.tolist(), ic_neu, 0.5)[:CUT_NUM]
    sub = full[full.factor_name.isin(chosen)]
    scan.append(dict(
        前N=n_, 去重后=len(chosen),
        IC中位=sub.absic.median() if len(sub) else np.nan,
        多头超额中位=sub.hedge_annualized_return.median() if len(sub) else np.nan,
        多空年化中位=sub.annualized_return.median() if len(sub) else np.nan,
        夏普中位=sub.sharpe_ratio.median() if len(sub) else np.nan,
        MPROB中位=sub.MPROB.median() if len(sub) else np.nan,
        SSM中位=sub.SSM.median() if len(sub) else np.nan,
        与原版重合=len(set(chosen) & sets["原版 hm104(IC-only)"]),
    ))
print(pd.DataFrame(scan).to_string(index=False, float_format=lambda v: f"{v:.4f}"))

n_gate = int((full.ratio_mean >= 0.5).sum())
print(f"\n[MPROB 门槛是否咬得住] 覆盖 >=0.5 的池子 {n_gate} 个，其中 MPROB >= 0.315 的有 "
      f"{int(((full.ratio_mean >= 0.5) & (full.MPROB >= 0.315)).sum())} 个；"
      f"按 |IC| 排序的前 35 个里 MPROB 中位 {full.nlargest(35, 'absic').MPROB.median():.3f}")

print("\n[MPROB / SSM 分布]")
print("MPROB:" + full.MPROB.describe(percentiles=[.5, .9, .95, .99]).to_string().replace("\n", "\n       "))
print("SSM:  " + full.SSM.describe(percentiles=[.5, .9, .95, .99]).to_string().replace("\n", "\n      "))

# ---------------------------------------------------------------- HTML
import html as _html

OUT = Path("/home/chenzongwei/rust_pyfunc/tmp_mono_metric/lead/hm104_mprob_compare.html")
COLORS = ["#1e4e79", "#b8450f", "#186b43", "#7a3ea8", "#0f6f8a", "#8a6d0f", "#4a5568", "#a11b4a"]


def _box(key, title, fmt, ylabel, width=780, height=430):
    vals = [full[full.factor_name.isin(lists[k])][key].to_numpy(dtype=float) for k in KEYS]
    vals = [v[np.isfinite(v)] for v in vals]
    lo = min(v.min() for v in vals if len(v))
    hi = max(v.max() for v in vals if len(v))
    span = (hi - lo) or 1.0
    lo -= span * 0.10
    hi += span * 0.12
    span = hi - lo
    pad_l, pad_r, pad_t, pad_b = 80, 24, 44, 82
    slot = (width - pad_l - pad_r) / len(KEYS)
    bw = min(74.0, slot * 0.42)
    p = [f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" aria-label="{_html.escape(title)}">',
         f'<text x="{pad_l}" y="22" font-size="15" font-weight="650" fill="#1c2128">{_html.escape(title)}</text>']

    def Y(v):
        return pad_t + (hi - v) / span * (height - pad_t - pad_b)

    for k in range(6):
        v = lo + span * k / 5
        y = Y(v)
        p.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{width-pad_r}" y2="{y:.1f}" stroke="#eef1f5"/>')
        p.append(f'<text x="{pad_l-10}" y="{y+4:.1f}" font-size="11.5" fill="#5c6672" text-anchor="end">{fmt(v)}</text>')
    p.append(f'<text x="14" y="{pad_t+(height-pad_t-pad_b)/2:.0f}" font-size="11.5" fill="#5c6672" '
             f'transform="rotate(-90 14 {pad_t+(height-pad_t-pad_b)/2:.0f})" text-anchor="middle">{_html.escape(ylabel)}</text>')
    rng = np.random.default_rng(7)
    for gi, (k, v) in enumerate(zip(KEYS, vals)):
        if not len(v):
            continue
        col = COLORS[gi % len(COLORS)]
        cx = pad_l + slot * (gi + 0.5)
        x0, x1 = cx - bw / 2, cx + bw / 2
        q1, q3, med, mn, mx = np.percentile(v, [25, 75, 50, 0, 100])
        p.append(f'<rect x="{x0:.1f}" y="{Y(q3):.1f}" width="{bw:.1f}" height="{max(1.0, Y(q1)-Y(q3)):.1f}" '
                 f'fill="{col}18" stroke="{col}" stroke-width="1.6"/>')
        p.append(f'<line x1="{x0:.1f}" y1="{Y(med):.1f}" x2="{x1:.1f}" y2="{Y(med):.1f}" stroke="{col}" stroke-width="2.6"/>')
        p.append(f'<circle cx="{cx:.1f}" cy="{Y(v.mean()):.1f}" r="3.4" fill="none" stroke="{col}" stroke-width="1.8"/>')
        p.append(f'<line x1="{cx:.1f}" y1="{Y(q3):.1f}" x2="{cx:.1f}" y2="{Y(mx):.1f}" stroke="{col}" stroke-width="1.3"/>')
        p.append(f'<line x1="{cx:.1f}" y1="{Y(q1):.1f}" x2="{cx:.1f}" y2="{Y(mn):.1f}" stroke="{col}" stroke-width="1.3"/>')
        for jx, val in zip(rng.uniform(-bw * 0.30, bw * 0.30, len(v)), v):
            p.append(f'<circle cx="{cx+jx:.1f}" cy="{Y(val):.1f}" r="2.3" fill="{col}" fill-opacity="0.5"/>')
        lab = k if len(k) <= 18 else k[:17] + "…"
        p.append(f'<text x="{cx:.1f}" y="{height-pad_b+20:.0f}" font-size="11.5" font-weight="600" fill="#1c2128" '
                 f'text-anchor="middle">{_html.escape(lab)}</text>')
        p.append(f'<text x="{cx:.1f}" y="{height-pad_b+37:.0f}" font-size="10.5" fill="#5c6672" '
                 f'text-anchor="middle">中位 {fmt(med)}</text>')
    p.append('</svg>')
    return "".join(p)


summary_rows = "".join(
    f'<tr><th>{_html.escape(r["名单"])}</th><td>{int(r["n"])}</td>'
    f'<td>{r["IC中位"]:.4f}</td><td>{r["IC均值"]:.4f}</td><td>{r["IC最小"]:.4f}</td>'
    f'<td>{r["多头超额中位"]*100:.2f}%</td><td>{r["多头超额均值"]*100:.2f}%</td>'
    f'<td>{r["多头超额最小"]*100:.2f}%</td><td>{r["多空年化中位"]*100:.2f}%</td>'
    f'<td>{r["夏普中位"]:.2f}</td><td>{r["MPROB中位"]:.3f}</td><td>{r["SSM中位"]:.3f}</td></tr>'
    for r in rows)

scan_rows = "".join(
    f'<tr><th>MPROB 前 {r["前N"]}</th><td>{r["去重后"]}</td>'
    f'<td>{r["IC中位"]:.4f}</td><td>{r["多头超额中位"]*100:.2f}%</td>'
    f'<td>{r["多空年化中位"]*100:.2f}%</td><td>{r["夏普中位"]:.2f}</td>'
    f'<td>{r["MPROB中位"]:.3f}</td><td>{r["SSM中位"]:.3f}</td>'
    f'<td>{r["与原版重合"]}</td></tr>' for r in scan)

mk = "mprob-only top35(去重0.5)"
detail = full[full.factor_name.isin(lists[mk])].sort_values("MPROB", ascending=False)
detail_rows = "".join(
    f'<tr><th class="nm">{_html.escape(x.factor_name)}</th><td>{x.IC_mean:+.4f}</td><td>{abs(x.IC_mean):.4f}</td>'
    f'<td>{x.hedge_annualized_return*100:+.2f}%</td><td>{x.annualized_return*100:+.2f}%</td>'
    f'<td>{x.MPROB:.3f}</td><td>{x.SSM:.3f}</td></tr>'
    for x in detail.itertuples())

head_cells = "".join(f'<th>{_html.escape(k.split("(")[0])}</th>' for k in KEYS)
overlap_rows = "".join(
    f'<tr><th>{_html.escape(KEYS[i].split("(")[0])}</th>'
    + "".join(f'<td class="{"diag" if i == j else ""}">{len(sets[KEYS[i]] & sets[KEYS[j]])}</td>'
              for j in range(len(KEYS))) + '</tr>' for i in range(len(KEYS)))

doc = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>hm104 MPROB 与 SSM 入选对比</title>
<style>
 :root{{--bg:#f6f7f9;--panel:#fff;--ink:#1c2128;--muted:#5c6672;--line:#dde3ea;--soft:#eef1f5}}
 *{{box-sizing:border-box}} html,body{{margin:0}}
 body{{background:var(--bg);color:var(--ink);font:14px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif}}
 header{{padding:22px 28px 6px}} h1{{margin:0 0 6px;font-size:21px;font-weight:650}}
 header p{{margin:0;color:var(--muted);font-size:13px}}
 .wrap{{display:grid;grid-template-columns:minmax(0,1fr) 620px;gap:22px;padding:14px 28px 44px;align-items:start}}
 @media(max-width:1500px){{.wrap{{grid-template-columns:minmax(0,1fr)}}}}
 .card{{background:var(--panel);border:1.5px solid var(--line);border-radius:10px;padding:14px 16px;margin-bottom:18px}}
 .card h2{{margin:0 0 10px;font-size:14px;font-weight:650}}
 aside{{position:sticky;top:16px;align-self:start;max-height:calc(100vh - 32px);overflow-y:auto}}
 table{{width:100%;border-collapse:collapse;font-size:12px}}
 th,td{{padding:5px 7px;border-bottom:1px solid var(--soft);text-align:right;white-space:nowrap}}
 th:first-child,td:first-child{{text-align:left}}
 thead th{{color:var(--muted);font-weight:600;border-bottom:1px solid var(--line)}}
 td.diag{{background:#f2f4f7;font-weight:650}}
 th.nm{{white-space:normal;word-break:break-all;font-weight:400;font-size:11px;max-width:520px}}
 .scroll{{overflow-x:auto}}
 .legend{{display:flex;gap:14px;flex-wrap:wrap;font-size:12px;color:var(--muted);margin-top:8px}}
 .legend i{{display:inline-block;width:11px;height:11px;border-radius:3px;margin-right:6px;vertical-align:-1px}}
</style></head><body>
<header>
  <h1>hm104：MPROB 门槛版 / mprob-only top35 与 SSM 两版对比（gap5）</h1>
  <p>|RankIC| 与多头超额收益统一取 hm104_mprob 那轮的完整回测表（21968 个因子同一口径）；
     多头超额 = hedge_annualized_return（多头组年化 − 指数年化）。名单一律为去重后的 35 个。</p>
  <div class="legend">{''.join(f'<span><i style="background:{COLORS[i%len(COLORS)]}"></i>{_html.escape(k)}</span>' for i, k in enumerate(KEYS))}</div>
</header>
<div class="wrap">
 <main>
  <div class="card"><h2>|RankIC| 分布</h2>{_box("absic", "|RankIC| 分布（各名单）", lambda v: f"{v:.3f}", "|RankIC|")}</div>
  <div class="card"><h2>多头超额收益分布（年化）</h2>{_box("hedge_annualized_return", "多头超额收益分布（年化）", lambda v: f"{v*100:.0f}%", "多头超额")}</div>
  <div class="card"><h2>mprob-only top35（MPROB 降序 + 相关性去重 0.5）逐因子明细</h2>
    <div class="scroll"><table><thead><tr><th>因子名</th><th>IC_mean</th><th>|RankIC|</th><th>多头超额</th><th>多空年化</th><th>MPROB</th><th>SSM</th></tr></thead>
    <tbody>{detail_rows}</tbody></table></div></div>
  <div class="card"><h2>MPROB 排名深度扫描（取 MPROB 前 N → 去重 0.5 → 最多 35）</h2>
    <div class="scroll"><table><thead><tr><th>深度</th><th>去重后</th><th>|IC| 中位</th><th>多头超额中位</th><th>多空年化中位</th><th>夏普中位</th><th>MPROB 中位</th><th>SSM 中位</th><th>与原版重合</th></tr></thead>
    <tbody>{scan_rows}</tbody></table></div></div>
 </main>
 <aside>
  <div class="card"><h2>各名单指标汇总</h2>
   <div class="scroll"><table><thead><tr><th>名单</th><th>n</th><th>|IC| 中位</th><th>|IC| 均值</th><th>|IC| 最小</th>
   <th>多头超额中位</th><th>均值</th><th>最小</th><th>多空年化中位</th><th>夏普中位</th><th>MPROB 中位</th><th>SSM 中位</th></tr></thead>
   <tbody>{summary_rows}</tbody></table></div></div>
  <div class="card"><h2>两两重合的因子个数</h2>
   <div class="scroll"><table><thead><tr><th></th>{head_cells}</tr></thead><tbody>{overlap_rows}</tbody></table></div></div>
 </aside>
</div></body></html>"""
OUT.write_text(doc, encoding="utf-8")
print(f"\n✅ HTML 报告: {OUT}")
sys.stdout.flush()
