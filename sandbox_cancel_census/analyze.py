# -*- coding: utf-8 -*-
"""撤单普查（第 03 期）：日频过滤、对账、描述统计、出图用的中间表。"""
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

SB = Path("/home/chenzongwei/rust_pyfunc/sandbox_cancel_census")
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


def daily(dates=None):
    cal = maps("DailyPrice")[0]
    i_end = max(i for i, x in enumerate(cal) if x <= max(dates or DATES))
    i_start = next(i for i, x in enumerate(cal) if x >= 20230801)
    S = slice(cal[i_start], cal[i_end])
    out = {}
    for name, f, key in [("close", "DailyPrice", "S_DQ_CLOSE"), ("pre", "DailyPrice", "S_DQ_PRECLOSE"),
                         ("st", "DailyPrice", "S_DQ_TRADESTATUS"), ("hi", "DailyPrice", "S_DQ_HIGH"),
                         ("lo", "DailyPrice", "S_DQ_LOW"), ("amt", "DailyPrice", "S_DQ_AMOUNT"),
                         ("vol", "DailyPrice", "S_DQ_VOLUME"), ("mv", "DailyDerived", "S_DQ_MV")]:
        out[name] = field(f, key, S.start, S.stop)
    return out


def load():
    dates = [d for d in DATES if (SB / f"out/{d}.json").exists()]
    df = pd.concat([pd.DataFrame(json.load(open(SB / f"out/{d}.json"))["stocks"]).assign(date=d) for d in dates])
    df["code"] = df["code"].astype(str).str.zfill(6)
    D = daily(dates)
    idx = list(D["close"].index)
    age = {}
    for d in dates:
        j = idx.index(d)
        age[d] = D["close"].iloc[max(0, j - 60): j].notna().sum() >= 60
    for k in ["close", "pre", "st", "hi", "lo", "amt", "vol", "mv"]:
        tab = D[k]
        df[k] = [tab.loc[d, c] if (d in tab.index and c in tab.columns) else np.nan
                 for d, c in zip(df["date"], df["code"])]
    df["age_ok"] = [bool(age[d].get(c, False)) for d, c in zip(df["date"], df["code"])]
    df["tvol"] = df.tvol_b + df.tvol_s
    df["cvol"] = df.cvol_b + df.cvol_s
    df["cn"] = df.cn_b + df.cn_s
    df["tn"] = df.tn_b + df.tn_s
    df["cancel_share"] = df.cvol / (df.cvol + df.tvol)
    df["ret"] = df.close / df.pre - 1
    return df


if __name__ == "__main__":
    df = load()
    print("原始行", len(df))
    f = df[(df.st == 1) & (df.hi != df.lo) & df.age_ok & df.amt.notna()].copy()
    print("过滤后", len(f))
    print(f.groupby("date").size().to_string())
    f["amt_us"] = f.amt * 1000
    f["ratio_amt"] = f.tamt / f.amt_us
    f["ratio_vol"] = f.tvol / (f.vol * 100)
    print("\n对账 成交额比（逐笔/日频）")
    print(f.groupby("date")["ratio_amt"].describe(percentiles=[.05, .5, .95]).round(4).to_string())
    print("\n对账 成交量比（逐笔/日频）")
    print(f.groupby("date")["ratio_vol"].describe(percentiles=[.05, .5, .95]).round(4).to_string())
    print("\n撤单占比分布")
    print(f.groupby("date")["cancel_share"].describe(percentiles=[.1, .25, .5, .75, .9]).round(4).to_string())
