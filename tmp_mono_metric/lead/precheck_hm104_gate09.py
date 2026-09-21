"""hm104 门槛 0.9 的离线预检：用 hm104_ssm2 这轮的缓存复现完整筛选（四通道 + 两道去重），
确认 0.9 时能不能凑满 35 个，以及凑满的话质量如何。不跑回测。
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import rust_pyfunc as rp

R = Path("/nas197/user_home_unsafe/chenzongwei/hm104_ssm2_tail_v4")

COVER_RATE = 0.5
RET_POINT_NEU_GAP5 = 0.055
RET_POINT_GAP5 = 0.1
IC_POINT_GAP5 = 0.03
IC_POINT_NEU_GAP5 = 0.01
CORR_POINT_NEU = 0.5
CORR_POINT = 0.8
CUT_NUM = 35
CUT1_RATE = None
CUT2_RATE = 1.0
RAW_RET_RATE = None
RAW_IC_RATE = None


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


def select(sn, sr, ineu, iraw, gate):
    neu = sn[sn.ratio_mean >= COVER_RATE].copy()
    raw = sr[sr.ratio_mean >= COVER_RATE].copy()
    if gate is not None:
        neu = neu[neu["SSM"] >= gate].copy()
    nr = select_by_metric(neu, ineu, "hedge_annualized_return", RET_POINT_NEU_GAP5, CORR_POINT_NEU)
    ni = select_by_metric(neu, ineu, "IC_mean", IC_POINT_NEU_GAP5, CORR_POINT_NEU)
    rr = select_by_metric(raw, iraw, "hedge_annualized_return", RET_POINT_GAP5, CORR_POINT)
    ri = select_by_metric(raw, iraw, "IC_mean", IC_POINT_GAP5, CORR_POINT)
    a, b, c, d = (nr.factor_name.tolist(), ni.factor_name.tolist(),
                  rr.factor_name.tolist(), ri.factor_name.tolist())
    ordered = b + [n for n in d if n not in b] + \
              [n for n in a if n not in b and n not in d] + \
              [n for n in c if n not in b and n not in d and n not in a]
    chosen = corr_select(ordered, ineu, CORR_POINT)
    if len(chosen) <= CUT_NUM:
        return chosen, len(ni), len(ordered)
    cut1 = 0 if CUT1_RATE is None else int(CUT_NUM * CUT1_RATE)
    cut2 = 0 if CUT2_RATE is None else int(CUT_NUM * CUT2_RATE)
    rrc = 0 if RAW_RET_RATE is None else int(CUT_NUM * RAW_RET_RATE)
    ric = 0 if RAW_IC_RATE is None else int(CUT_NUM * RAW_IC_RATE)
    n1 = [n for n in chosen if n in a]
    n2 = [n for n in chosen if n in b and n not in n1]
    n3 = [n for n in chosen if n in c and n not in n1 and n not in n2]
    n4 = [n for n in chosen if n in d and n not in n1 and n not in n2 and n not in n3]
    return n1[:cut1] + n2[:cut2] + n3[:rrc] + n4[:ric], len(ni), len(ordered)


sn = pd.read_parquet(R / "metrics" / "summary_neu_gap5_candidates.parquet")
sr = pd.read_parquet(R / "metrics" / "summary_rolled_gap5_candidates.parquet")
ineu, iraw = load_wide("ic_neu_gap5"), load_wide("ic_rolled_gap5")
sn["absic"] = sn.IC_mean.abs()
actual2 = pd.read_parquet(R / "selected" / "gap5_selected.parquet")["factor_name"].tolist()

print("=" * 104)
print("[校验] 门槛 0.5 复现 vs hm104_ssm2 实际入选")
rep, nni, nordered = select(sn, sr, ineu, iraw, 0.5)
print(f"  实际 {len(actual2)} / 复现 {len(rep)}   逐项一致: {rep == actual2}   集合相同: {set(rep)==set(actual2)}")
print()
print("[预检] 门槛 0.9")
s9, nni9, nordered9 = select(sn, sr, ineu, iraw, 0.9)
sub9 = sn[sn.factor_name.isin(s9)]
print(f"  中性化池(SSM>=0.9) = {int((sn.ratio_mean>=COVER_RATE).sum() and (sn[(sn.ratio_mean>=COVER_RATE)&(sn.SSM>=0.9)]).shape[0])}")
print(f"  neu_ic 通道初筛出 {nni9} 个，合并候选 {nordered9} 个，最终入选 {len(s9)} 个")
if len(s9):
    print(f"  SSM: 最小 {sub9.SSM.min():.3f} 中位 {sub9.SSM.median():.3f}")
    print(f"  |IC| 中位 {sub9.absic.median():.4f}   多空年化中位 {sub9.annualized_return.median()*100:.2f}%   夏普中位 {sub9.sharpe_ratio.median():.2f}")
print(f"  凑满 35 个: {len(s9)==35}")
