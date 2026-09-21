"""hm104 门槛扫描（含两道相关性去重，复现已与实际入选名单逐项一致）。

原始值通道与门槛无关，先算一次缓存复用。
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import rust_pyfunc as rp

R = Path("/nas197/user_home_unsafe/chenzongwei/hm104_ssm2_tail_v4")
COVER_RATE = 0.5
RET_POINT_NEU_GAP5, RET_POINT_GAP5, IC_POINT_GAP5 = 0.055, 0.1, 0.03
IC_POINT_NEU_GAP5, CORR_POINT_NEU, CORR_POINT = 0.01, 0.5, 0.8
CUT_NUM, CUT2_RATE = 35, 1.0


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
sn["absic"] = sn.IC_mean.abs()

neu_all = sn[sn.ratio_mean >= COVER_RATE].copy()
raw_all = sr[sr.ratio_mean >= COVER_RATE].copy()
# 原始值两个通道与门槛无关，只算一次
raw_ret = select_by_metric(raw_all, iraw, "hedge_annualized_return", RET_POINT_GAP5, CORR_POINT)
raw_ic = select_by_metric(raw_all, iraw, "IC_mean", IC_POINT_GAP5, CORR_POINT)
rr_names, ri_names = raw_ret.factor_name.tolist(), raw_ic.factor_name.tolist()
print(f"原始值通道缓存完成: raw_ret {len(rr_names)} 个, raw_ic {len(ri_names)} 个", flush=True)

print("=" * 108)
print("[hm104 门槛扫描] 含两道相关性去重（与实际筛选逐项一致），收益指标从 hm104_ssm2 全量回测表取")
print(f"  {'门槛':>7}{'中性化池':>9}{'neu_ic初筛':>10}{'最终入选':>9}{'|IC|中位':>10}{'多空年化中位':>13}{'夏普中位':>10}{'SSM中位':>9}")
for g in [None, 0.0, 0.098, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9]:
    neu = neu_all if g is None else neu_all[neu_all["SSM"] >= g].copy()
    neu_ret = select_by_metric(neu, ineu, "hedge_annualized_return", RET_POINT_NEU_GAP5, CORR_POINT_NEU)
    neu_ic = select_by_metric(neu, ineu, "IC_mean", IC_POINT_NEU_GAP5, CORR_POINT_NEU)
    nr, ni = neu_ret.factor_name.tolist(), neu_ic.factor_name.tolist()
    ordered = ni + [n for n in ri_names if n not in ni] + \
              [n for n in nr if n not in ni and n not in ri_names] + \
              [n for n in rr_names if n not in ni and n not in ri_names and n not in nr]
    chosen = corr_select(ordered, ineu, CORR_POINT)
    if len(chosen) <= CUT_NUM:
        final = chosen
    else:
        n1 = [n for n in chosen if n in nr]
        n2 = [n for n in chosen if n in ni and n not in n1]
        n3 = [n for n in chosen if n in rr_names and n not in n1 and n not in n2]
        n4 = [n for n in chosen if n in ri_names and n not in n1 and n not in n2 and n not in n3]
        final = n2[:int(CUT_NUM * CUT2_RATE)]
    sub = sn[sn.factor_name.isin(final)]
    lab = "无门槛" if g is None else f"{g:.3f}"
    print(f"  {lab:>7}{len(neu):>9}{len(ni):>10}{len(final):>9}{sub.absic.median():>10.4f}"
          f"{sub.annualized_return.median() * 100:>12.2f}%{sub.sharpe_ratio.median():>10.2f}{sub.SSM.median():>9.3f}", flush=True)
