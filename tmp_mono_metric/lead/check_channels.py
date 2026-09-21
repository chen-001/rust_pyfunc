"""确认：另外三个通道（neu_ret / raw_ic / raw_ret）对最终入选名单到底有没有影响。

对照 A：只用 neu_ic 通道（门槛 → |IC| 降序 → 去重 0.5 → 再取 35）
对照 B：完整四通道（与线上 select_tail_v2_factors 逐项一致的那套）
用 hm104_ssm3 的缓存，门槛 0.9。
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import rust_pyfunc as rp

R = Path("/nas197/user_home_unsafe/chenzongwei/hm104_ssm3_tail_v4")
COVER_RATE = 0.5
RET_POINT_NEU_GAP5, RET_POINT_GAP5, IC_POINT_GAP5 = 0.055, 0.1, 0.03
IC_POINT_NEU_GAP5, CORR_POINT_NEU, CORR_POINT = 0.01, 0.5, 0.8
CUT_NUM, CUT2_RATE, GATE = 35, 1.0, 0.9


def load_wide(stem):
    m = np.load(R / "ic_ts" / f"{stem}.npy", allow_pickle=False)
    names = json.loads((R / "ic_ts" / f"{stem}_names.json").read_text(encoding="utf-8"))
    dates = np.load(R / "ic_ts" / f"{stem}_dates.npy", allow_pickle=False)
    return pd.DataFrame(m, index=dates[: m.shape[0]], columns=names)


def corr_select(names, ic_wide, th):
    if not names:
        return []
    o = np.ascontiguousarray(ic_wide.loc[:, names].to_numpy(dtype=np.float32, copy=False).T)
    return [names[i] for i in rp.tail_v2_select_by_ic_corr_abs_f32(o, float(th))]


def select_by_metric(df, ic_wide, metric, minv, cp):
    if metric == "IC_mean":
        r = df.assign(metric_abs=df[metric].abs()).sort_values("metric_abs", ascending=False)
        f = r[r.metric_abs >= minv]
    else:
        f = df.sort_values(metric, ascending=False)
        f = f[f[metric] >= minv]
    names = f.factor_name.tolist()
    chosen = corr_select(names, ic_wide, cp)
    return f[f.factor_name.isin(chosen)].copy()


sn = pd.read_parquet(R / "metrics" / "summary_neu_gap5_candidates.parquet")
sr = pd.read_parquet(R / "metrics" / "summary_rolled_gap5_candidates.parquet")
ineu, iraw = load_wide("ic_neu_gap5"), load_wide("ic_rolled_gap5")
actual = pd.read_parquet(R / "selected" / "gap5_selected.parquet")["factor_name"].tolist()

neu = sn[(sn.ratio_mean >= COVER_RATE) & (sn["SSM"] >= GATE)].copy()
raw = sr[sr.ratio_mean >= COVER_RATE].copy()

neu_ret = select_by_metric(neu, ineu, "hedge_annualized_return", RET_POINT_NEU_GAP5, CORR_POINT_NEU)
neu_ic = select_by_metric(neu, ineu, "IC_mean", IC_POINT_NEU_GAP5, CORR_POINT_NEU)
raw_ret = select_by_metric(raw, iraw, "hedge_annualized_return", RET_POINT_GAP5, CORR_POINT)
raw_ic = select_by_metric(raw, iraw, "IC_mean", IC_POINT_GAP5, CORR_POINT)
nr, ni, rr, ri = (neu_ret.factor_name.tolist(), neu_ic.factor_name.tolist(),
                  raw_ret.factor_name.tolist(), raw_ic.factor_name.tolist())

# ---- 对照 A：只用 neu_ic 通道 ----
A = corr_select(ni, ineu, CORR_POINT)[:CUT_NUM]

# ---- 对照 B：完整四通道 ----
ordered = ni + [n for n in ri if n not in ni] + \
          [n for n in nr if n not in ni and n not in ri] + \
          [n for n in rr if n not in ni and n not in ri and n not in nr]
chosen = corr_select(ordered, ineu, CORR_POINT)
cut1 = 0
cut2 = int(CUT_NUM * CUT2_RATE)
n1 = [n for n in chosen if n in nr]
n2 = [n for n in chosen if n in ni and n not in n1]
B = n1[:cut1] + n2[:cut2]

print("=" * 100)
print(f"[门槛 {GATE}] neu_ic 初筛 {len(ni)} 个   neu_ret 初筛 {len(nr)} 个   raw_ic {len(ri)} 个   raw_ret {len(rr)} 个")
print(f"  对照 B（四通道）与实际入选逐项一致: {B == actual}")
print(f"  对照 A（只用 neu_ic）选出 {len(A)} 个，与对照 B 逐项一致: {A == B}")
onlyB = [n for n in B if n not in A]
onlyA = [n for n in A if n not in B]
print(f"  只在四通道里出现的: {len(onlyB)} 个；只在 neu_ic-only 里出现的: {len(onlyA)} 个")
if onlyB or onlyA:
    print()
    print("  差异因子的归因（在哪些通道的初筛池里）:")
    for n in (onlyB + onlyA)[:12]:
        tags = []
        if n in ni: tags.append("neu_ic")
        if n in nr: tags.append("neu_ret")
        if n in ri: tags.append("raw_ic")
        if n in rr: tags.append("raw_ret")
        which = "四通道独有" if n in onlyB else "neu_ic-only 独有"
        print(f"    [{which}] {n[:70]}  属于 {tags}")
