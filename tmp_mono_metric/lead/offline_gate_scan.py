"""离线复现 use_ssm=True 的完整筛选（四通道 + 两道相关性去重），用于扫描 SSM 门槛。

先与实际入选名单对齐验证，再扫门槛。不跑回测。
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import rust_pyfunc as rp

R = Path("/nas197/user_home_unsafe/chenzongwei/hm100_ssm2_tail_v4")

# hm100_ssm2.py 的 selection_kwargs（未显式给出的用函数默认值）
COVER_RATE = 0.5
RET_POINT_NEU_GAP5 = 0.055      # 默认
RET_POINT_GAP5 = 0.1            # 默认
IC_POINT_GAP5 = 0.03            # 默认
IC_POINT_NEU_GAP5 = 0.01
CORR_POINT_NEU = 0.5
CORR_POINT = 0.8
CUT_NUM = 35
CUT1_RATE = None
CUT2_RATE = 1.0
RAW_RET_RATE = None
RAW_IC_RATE = None


def load_wide(stem: str) -> pd.DataFrame:
    m = np.load(R / "ic_ts" / f"{stem}.npy", allow_pickle=False)
    names = json.loads((R / "ic_ts" / f"{stem}_names.json").read_text(encoding="utf-8"))
    dates = np.load(R / "ic_ts" / f"{stem}_dates.npy", allow_pickle=False)
    return pd.DataFrame(m, index=dates[: m.shape[0]], columns=names)


def corr_select(sorted_names, ic_wide, threshold):
    if not sorted_names:
        return []
    ordered_ic = np.ascontiguousarray(
        ic_wide.loc[:, sorted_names].to_numpy(dtype=np.float32, copy=False).T
    )
    pos = rp.tail_v2_select_by_ic_corr_abs_f32(ordered_ic, float(threshold))
    return [sorted_names[i] for i in pos]


def select_by_metric(df, ic_wide, metric, min_value, corr_point):
    if metric == "IC_mean":
        ranked = df.assign(metric_abs=df[metric].abs()).sort_values(
            "metric_abs", ascending=False, kind="stable"
        )
        filtered = ranked[ranked.metric_abs >= min_value]
    else:
        filtered = df.sort_values(metric, ascending=False, kind="stable")
        filtered = filtered[filtered[metric] >= min_value]
    ordered = filtered.factor_name.tolist()
    chosen = corr_select(ordered, ic_wide, corr_point)
    return filtered[filtered.factor_name.isin(chosen)].copy()


def select(summ_neu, summ_raw, ic_neu, ic_raw, ssm_gate, cut_num=CUT_NUM):
    gap_neu = summ_neu[summ_neu.ratio_mean >= COVER_RATE].copy()
    gap_raw = summ_raw[summ_raw.ratio_mean >= COVER_RATE].copy()
    if ssm_gate is not None:
        gap_neu = gap_neu[gap_neu["SSM"] >= ssm_gate].copy()

    neu_ret = select_by_metric(gap_neu, ic_neu, "hedge_annualized_return",
                               RET_POINT_NEU_GAP5, CORR_POINT_NEU)
    neu_ic = select_by_metric(gap_neu, ic_neu, "IC_mean", IC_POINT_NEU_GAP5, CORR_POINT_NEU)
    raw_ret = select_by_metric(gap_raw, ic_raw, "hedge_annualized_return",
                               RET_POINT_GAP5, CORR_POINT)
    raw_ic = select_by_metric(gap_raw, ic_raw, "IC_mean", IC_POINT_GAP5, CORR_POINT)

    nr, ni, rr, ri = (neu_ret.factor_name.tolist(), neu_ic.factor_name.tolist(),
                      raw_ret.factor_name.tolist(), raw_ic.factor_name.tolist())
    ordered = ni + [n for n in ri if n not in ni] + \
              [n for n in nr if n not in ni and n not in ri] + \
              [n for n in rr if n not in ni and n not in ri and n not in nr]
    chosen = corr_select(ordered, ic_neu, CORR_POINT)
    if len(chosen) <= cut_num:
        return chosen
    cut1 = 0 if CUT1_RATE is None else int(cut_num * CUT1_RATE)
    cut2 = 0 if CUT2_RATE is None else int(cut_num * CUT2_RATE)
    raw_ret_cut = 0 if RAW_RET_RATE is None else int(cut_num * RAW_RET_RATE)
    raw_ic_cut = 0 if RAW_IC_RATE is None else int(cut_num * RAW_IC_RATE)
    neu_ret_names = [n for n in chosen if n in nr]
    neu_ic_names = [n for n in chosen if n in ni and n not in neu_ret_names]
    raw_ret_names = [n for n in chosen if n in rr and n not in neu_ret_names and n not in neu_ic_names]
    raw_ic_names = [n for n in chosen if n in ri and n not in neu_ret_names
                    and n not in neu_ic_names and n not in raw_ret_names]
    return (neu_ret_names[:cut1] + neu_ic_names[:cut2]
            + raw_ret_names[:raw_ret_cut] + raw_ic_names[:raw_ic_cut])


summ_neu = pd.read_parquet(R / "metrics" / "summary_neu_gap5_candidates.parquet")
summ_raw = pd.read_parquet(R / "metrics" / "summary_rolled_gap5_candidates.parquet")
ic_neu = load_wide("ic_neu_gap5")
ic_raw = load_wide("ic_rolled_gap5")
actual = pd.read_parquet(R / "selected" / "gap5_selected.parquet")["factor_name"].tolist()

print("=" * 110)
print("[复现校验] 门槛 0.098")
rep = select(summ_neu, summ_raw, ic_neu, ic_raw, 0.098)
print(f"  实际 {len(actual)} 个 / 复现 {len(rep)} 个")
print(f"  逐项完全一致: {rep == actual}")
print(f"  集合相同: {set(rep) == set(actual)}")
if set(rep) != set(actual):
    print(f"  实际独有 {len(set(actual)-set(rep))} 个，复现独有 {len(set(rep)-set(actual))} 个")
    for n in list(set(actual) - set(rep))[:3]:
        print(f"    实际独有: {n[:78]}")
    for n in list(set(rep) - set(actual))[:3]:
        print(f"    复现独有: {n[:78]}")

print()
print("=" * 110)
print("[门槛扫描] 入选 35 个的质量（收益指标从 ssm2 全量回测表取）")
summ_neu["absic"] = summ_neu.IC_mean.abs()
print(f"  {'门槛':>7}{'中性化池':>10}{'入选':>6}{'|IC|中位':>10}{'多空年化中位':>13}{'夏普中位':>10}{'SSM中位':>9}{'SSM最小':>9}")
rows = {}
for g in [None, 0.0, 0.098, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    pool = summ_neu[summ_neu.ratio_mean >= COVER_RATE]
    if g is not None:
        pool = pool[pool["SSM"] >= g]
    s = select(summ_neu, summ_raw, ic_neu, ic_raw, g)
    rows[g] = s
    sub = summ_neu[summ_neu.factor_name.isin(s)]
    lab = "无门槛" if g is None else f"{g:.3f}"
    print(f"  {lab:>7}{len(pool):>10}{len(s):>6}{sub.absic.median():>10.4f}"
          f"{sub.annualized_return.median() * 100:>12.2f}%{sub.sharpe_ratio.median():>10.2f}"
          f"{sub.SSM.median():>9.3f}{sub.SSM.min():>9.3f}")

print()
print("=" * 110)
print("[与无门槛（原版纯 |IC| 选）的重合]")
base = set(rows[None])
for g in [0.0, 0.098, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    print(f"  门槛 {g:.3f}: 重合 {len(set(rows[g]) & base)}/35")
