# -*- coding: utf-8 -*-
"""成交的节奏（第 04 期）：日频过滤、对账、图的中间量。

两个计数单位
- 记录：文件里一行，图一图二用它，描述文件长什么样
- 事件：同一时间戳（10 毫秒刻度）算一次，图三图四用它
"""
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

SB = Path("/home/chenzongwei/rust_pyfunc/sandbox_trade_rhythm")
DATES = [20240205, 20240329, 20240813, 20240930, 20241008, 20250407, 20251127, 20260311]
ROOT = "/ssd_data/data/vars"
T = 14220


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


def daily(dates):
    cal = maps("DailyPrice")[0]
    i_end = max(i for i, x in enumerate(cal) if x <= max(dates))
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
    dates = [d for d in DATES if (SB / f"out/{d}.json").exists() and (SB / f"out/{d}.json").stat().st_size > 100000]
    rows, hist, lz = [], [], []
    mk = {}
    for d in dates:
        o = json.load(open(SB / f"out/{d}.json"))
        m = o["market"]
        mk[d] = m
        for r in o["recs"]:
            mean_e = r["e"] / T
            cov = r["sem"] / T - mean_e * m["mean"]
            r2 = cov * cov / ((r["sx2"] / T - mean_e ** 2) * m["var"]) if r["sx2"] > 0 else np.nan
            rows.append({"code": r["code"], "s": r["s"], "e": r["e"], "nc": r["nc"],
                         "amt": r["amt"], "tamt": r["amt"], "vol": r["vol"], "tvol": r["vol"],
                         "fano": r["fano"], "fano_e": r["fano_e"], "r2": r2, "date": d})
            hist.append(r["hist"])
            lz.append(r["lz"])
    df = pd.DataFrame(rows)
    df["code"] = df["code"].astype(str).str.zfill(6)
    A = dict(H=np.array(hist, float), LZ=np.array(lz, float), MK=mk)
    D = daily(dates)
    idx = list(D["close"].index)
    for k in ["close", "pre", "st", "hi", "lo", "amt", "vol", "mv"]:
        tab = D[k]
        df[k] = [tab.loc[d, c] if (d in tab.index and c in tab.columns) else np.nan
                 for d, c in zip(df["date"], df["code"])]
    age = {}
    for d in dates:
        j = idx.index(d)
        age[d] = D["close"].iloc[max(0, j - 60): j].notna().sum() >= 60
    df["age_ok"] = [bool(age[d].get(c, False)) for d, c in zip(df["date"], df["code"])]
    df["ret"] = df.close / df.pre - 1
    df["lam"] = df.s / T
    f = (df.st == 1) & (df.hi != df.lo) & df.age_ok & df.amt.notna()
    return df[f].reset_index(drop=True), {k: (v[f.values] if isinstance(v, np.ndarray) else v) for k, v in A.items()}


if __name__ == "__main__":
    df, A = load()
    pd.set_option("display.width", 200)
    print("过滤后", len(df), df.groupby("date").size().to_dict())
    print("\n对账 逐笔成交额 / 日频")
    print(df.groupby("date").apply(lambda g: (g.tamt / (g.amt * 1000)).quantile([.05, .5, .95]),
                                   include_groups=False).round(6).to_string())
    print("\n每秒事件数中位 / 记录数中位")
    print(df.groupby("date")[["lam", "s"]].median().round(2).to_string())
    print("\n每秒事件数方差除以均值（随机=1）")
    print(df.groupby("date")["fano"].describe(percentiles=[.1, .5, .9]).round(2).to_string())
    print("\n与大盘同秒的相关系数平方")
    print(df.groupby("date")["r2"].describe(percentiles=[.1, .5, .9]).round(3).to_string())
    print("\n整体 fano 分位", np.round(df.fano.quantile([.1, .25, .5, .75, .9]).values, 2))
    print("整体 r2 中位", round(df.r2.median(), 3))
