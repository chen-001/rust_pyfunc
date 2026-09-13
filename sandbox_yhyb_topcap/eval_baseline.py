# -*- coding: utf-8 -*-
"""补充分析 1：同一宇宙、同一时段的「活跃度基线」IC + 逐日 IC + 特异性过滤前后对比。
口径同 eval_ic.py：gap1 = close.shift(-1)/close-1（D 行 = D→D+1 前瞻）。
"""
import json
import numpy as np
import pandas as pd
from pure_ocean_breeze.jason.data.read_data import read_daily
from scipy import stats

HERE = "/home/chenzongwei/rust_pyfunc/sandbox_yhyb_topcap"
DATES = [20240104, 20240603, 20241008, 20260105, 20260716]
SUF = lambda c: c + (".SH" if c.startswith(("6", "9")) else ".SZ")

close = read_daily(close=1)
gap1 = close.shift(-1) / close - 1
amount = pd.read_parquet("/home/chenzongwei/database/daily_data/amounts.parquet")
flow_cap = pd.read_parquet("/home/chenzongwei/database/daily_data/flow_caps.parquet")
tc = read_daily(total_cap=1)

def sp(x, y):
    m = pd.notna(x) & pd.notna(y) & np.isfinite(x) & np.isfinite(y)
    if m.sum() < 30:
        return np.nan
    return stats.spearmanr(x[m], y[m]).statistic


daily = []
for d in DATES:
    dd = pd.Timestamp(f"{d // 10000}-{(d // 100) % 100:02d}-{d % 100:02d}")
    o = json.load(open(f"{HERE}/out_{d}.json"))
    idx = [SUF(c) for c in o["codes"]]
    v = np.array(o["vals"], dtype=float)
    df = pd.DataFrame(v, index=idx, columns=[f"{e}__{f}" for e in o["events"] for f in o["factor_names"]])
    g1 = gap1.loc[dd].reindex(idx)
    am = amount.loc[dd].reindex(idx)
    fc = flow_cap.loc[dd].reindex(idx)
    tcv = tc.loc[dd].reindex(idx)
    row = dict(
        date=str(dd.date()),
        ic_amount=sp(np.log(am + 1), g1),
        ic_flowcap=sp(np.log(fc + 1), g1),
        ic_totalcap=sp(np.log(tcv + 1), g1),
        ic_rate_bbm=sp(df["big_buy_m__rate"], g1),
        ic_rate_jump=sp(df["jump__rate"], g1),
        ic_rate_ice=sp(df["ice__rate"], g1),
        # 特异性过滤前后（big_buy_m）
        ic_raw_bbm=sp(df["big_buy_m__strength_raw_mean"], g1),
        ic_top5_bbm=sp(df["big_buy_m__top5_strength"], g1),
        ic_top10_bbm=sp(df["big_buy_m__top10_strength"], g1),
        ic_samespec_bbm=sp(df["big_buy_m__top5_spec_mean"], g1),
        ic_sameind_bbm=sp(df["big_buy_m__same_ind_strong_ratio"], g1),
        ic_sameind5_bbm=sp(df["big_buy_m__same_ind_top5_strength"], g1),
        # jump
        ic_raw_jump=sp(df["jump__strength_raw_mean"], g1),
        ic_top5_jump=sp(df["jump__top5_strength"], g1),
        ic_rmedf_jump=sp(df["jump__rmed_f_mean"], g1),
        ic_rhitf_jump=sp(df["jump__rhit_f_mean"], g1),
        ic_medb_jump=sp(df["jump__med_b_mean"], g1),
        # ice
        ic_top5_ice=sp(df["ice__top5_strength"], g1),
        ic_sameind_ice=sp(df["ice__same_ind_strong_ratio"], g1),
    )
    daily.append(row)

dd = pd.DataFrame(daily)
pd.set_option("display.width", 250)
print("=== 逐日 IC（Spearman vs gap1） ===")
print(dd.to_string(index=False))
print()
print("=== 活跃度/耦合 基线汇总（5 天均值） ===")
for c in dd.columns:
    if c == "date":
        continue
    s = dd[c].dropna()
    print(f"{c:26s} n={len(s)} mean={s.mean():+.4f} mean|.|={s.abs().mean():.4f} icir={s.mean()/s.std():+.2f} pos={ (s>0).mean():.2f}")
dd.to_csv(f"{HERE}/daily_ic_detail.csv", index=False)
