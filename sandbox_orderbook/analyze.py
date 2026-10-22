# -*- coding: utf-8 -*-
"""盘口深度一期：过滤、对账、描述统计。"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import h5py

SANDBOX = Path("/home/chenzongwei/rust_pyfunc/sandbox_orderbook")
DATES = [20240205, 20240329, 20240813, 20240930, 20241008, 20250407, 20251127, 20260311]
ROOT = "/ssd_data/data/vars"

def maps(d):
    cal = [int(x) for x in open(f"{ROOT}/{d}/calendar_map.csv").read().splitlines()[1:]]
    sym = [s.split(",")[0] for s in open(f"{ROOT}/{d}/symbol_map.csv").read().splitlines()[1:]]
    return cal, sym

def field(d, f, start, end):
    cal, sym = maps(d)
    i0 = next(i for i, x in enumerate(cal) if x >= start)
    i1 = next(i for i, x in enumerate(cal) if x > end)
    with h5py.File(f"{ROOT}/{d}/{f}.h5", "r") as h:
        a = h["data"][i0:i1, : len(sym)]
    return pd.DataFrame(a, index=cal[i0:i1], columns=sym)

df = pd.concat([pd.DataFrame(json.load(open(SANDBOX / f"out/{d}.json"))["stocks"]).assign(date=d) for d in DATES])
df["code"] = df["code"].astype(str).str.zfill(6)

cal = maps("DailyPrice")[0]
i_end = max(i for i, x in enumerate(cal) if x <= 20260311)
i_start = next(i for i, x in enumerate(cal) if x >= 20230801)
S = slice(cal[i_start], cal[i_end])

close = field("DailyPrice", "S_DQ_CLOSE", S.start, S.stop)
st = field("DailyPrice", "S_DQ_TRADESTATUS", S.start, S.stop)
hi = field("DailyPrice", "S_DQ_HIGH", S.start, S.stop)
lo = field("DailyPrice", "S_DQ_LOW", S.start, S.stop)
amt_d = field("DailyPrice", "S_DQ_AMOUNT", S.start, S.stop)
mv = field("DailyDerived", "S_DQ_MV", S.start, S.stop)

# 上市满 60 个交易日的掩码：往前数第 60 个交易日之前已有收盘价
idx = list(close.index)
age_ok = {}
for d in DATES:
    j = idx.index(d)
    prior = close.iloc[max(0, j - 60): j]
    age_ok[d] = (prior.notna().sum() >= 60)
    print(d, "上市满60日股票数", int(age_ok[d].sum()))

df["mv"] = [mv.loc[d, c] if (d in mv.index and c in mv.columns) else np.nan for d, c in zip(df.date, df.code)]
df["turnover_daily"] = [amt_d.loc[d, c] * 1000 if (d in amt_d.index and c in amt_d.columns) else np.nan for d, c in zip(df.date, df.code)]
df["status"] = [st.loc[d, c] if (d in st.index and c in st.columns) else np.nan for d, c in zip(df.date, df.code)]
df["high_d"] = [hi.loc[d, c] if (d in hi.index and c in hi.columns) else np.nan for d, c in zip(df.date, df.code)]
df["low_d"] = [lo.loc[d, c] if (d in lo.index and c in lo.columns) else np.nan for d, c in zip(df.date, df.code)]
df["age_ok"] = [bool(age_ok[d].get(c, False)) for d, c in zip(df.date, df.code)]

n0 = len(df)
f = df[(df.status == 1) & (df.high_d != df.low_d) & df.age_ok & df.turnover_daily.notna()]
print("原始行", n0, "过滤后", len(f), "剔除", n0 - len(f))
print(f.groupby("date").size().to_string())

# 对账：快照最后一行的累计成交额 vs 日频
f = f.assign(ratio=f.turnover / f.turnover_daily)
print("对账 turnover/日频: 中位 %.4f  P10 %.4f  P90 %.4f  落在 0.98-1.02 的比例 %.3f" % (
    f.ratio.median(), f.ratio.quantile(.1), f.ratio.quantile(.9), ((f.ratio > .98) & (f.ratio < 1.02)).mean()))

# 深度能撑几分钟的成交
f = f.assign(mins=f.amt50 / (f.turnover_daily / 240))

def q(s, name, fmt="{:,.3f}"):
    s = s.dropna()
    print(f"{name:16s} N={len(s):5d}  P10={fmt.format(s.quantile(.1))}  中位={fmt.format(s.median())}  P90={fmt.format(s.quantile(.9))}")

print("\n=== 全部 8 天合并 ===")
q(f.n_snap, "快照数", "{:,.0f}")
q(f.gap50, "快照间隔秒")
q(f.amt50 / 1e4, "十档金额万元")
q(f.sh50 / 1e4, "十档股数万股")
q(f.l1, "一档占十档")
q(f.l5, "前五档占十档")
q(f.ba, "买盘/卖盘")
q(f.spr50, "价差元", "{:,.4f}")
q(f.mins, "能撑分钟数")

print("\n=== 快照数分布 ===")
print((f.n_snap // 500 * 500).value_counts().sort_index().tail(12).to_string())
print("等于或超过 4800 的比例", float((f.n_snap >= 4800).mean()))

print("\n=== 分日 ===")
for d, g in f.groupby("date"):
    print(d, f"N={len(g):5d} 快照中位 {g.n_snap.median():7.0f} 十档金额中位 {g.amt50.median()/1e4:9,.0f}万 一档占比 {g.l1.median():.3f} 买/卖 {g.ba.median():.2f} 撑 {g.mins.median():.2f} 分钟")

print("\n=== 按流通市值三档 ===")
f = f.assign(size=pd.qcut(f.mv, [0, 1/3, 2/3, 1], labels=["小", "中", "大"]))
for s, g in f.groupby("size", observed=True):
    print(s, f"N={len(g):5d} 快照 {g.n_snap.median():6.0f} 十档金额 {g.amt50.median()/1e4:9,.0f}万 股数 {g.sh50.median()/1e4:8,.0f}万 一档 {g.l1.median():.3f} 五档 {g.l5.median():.3f} 买/卖 {g.ba.median():.2f} 撑 {g.mins.median():.2f} 分钟")

f.to_parquet(SANDBOX / "out/panel.parquet")
print("\n已存", SANDBOX / "out/panel.parquet", f.shape)
