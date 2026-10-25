# -*- coding: utf-8 -*-
"""成交的节奏（第 04 期）：日频过滤、对账、四个图的中间量。"""
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
    rows, hist, hist_s, lz, lz_s = [], [], [], [], []
    for d in dates:
        o = json.load(open(SB / f"out/{d}.json"))
        for r in o["recs"]:
            rows.append({k: r[k] for k in ("code", "s", "c", "s2", "nc", "nauc", "noth", "amt", "amt_c", "vol",
                                          "d_op", "d_sp", "d_ou", "d_su", "d_mc", "maxcs", "maxcsm")} | {"date": d})
            hist.append(r["hist"])
            hist_s.append(r["hist_s"])
            lz.append(r["lz"])
            lz_s.append(r["lz_s"])
    df = pd.DataFrame(rows)
    df["tamt"] = df["amt"]      # 逐笔加总的成交额
    df["tvol"] = df["vol"]
    df["code"] = df["code"].astype(str).str.zfill(6)
    A = dict(H=np.array(hist, float), HS=np.array(hist_s, float),
             LZ=np.array(lz, float), LZS=np.array(lz_s, float))
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
    df["ratio_p"] = df.d_op / df.d_sp
    df["ratio_u"] = df.d_ou / df.d_su
    df["common"] = 1 - (df.ratio_p - 1) / (df.ratio_u - 1)
    df["ret"] = df.close / df.pre - 1
    df["lam"] = df.s / T
    f = (df.st == 1) & (df.hi != df.lo) & df.age_ok & df.amt.notna()
    return df[f].reset_index(drop=True), {k: v[f.values] for k, v in A.items()}


if __name__ == "__main__":
    df, A = load()
    pd.set_option("display.width", 200)
    print("过滤后", len(df), df.groupby("date").size().to_dict())
    df["ratio_amt"] = df.tamt / (df.amt * 1000)
    df["ratio_vol"] = df.tvol / (df.vol * 100)
    print("\n对账 逐笔成交额 / 日频")
    print(df.groupby("date")["ratio_amt"].describe(percentiles=[.05, .5, .95]).round(4).to_string())
    print("\n对账 逐笔成交量 / 日频")
    print(df.groupby("date")["ratio_vol"].describe(percentiles=[.05, .5, .95]).round(4).to_string())
    print("\n纯成团倍数 ratio_p")
    print(df.groupby("date")["ratio_p"].describe(percentiles=[.1, .5, .9]).round(2).to_string())
    print("\n去作息前 ratio_u 中位")
    print(df.groupby("date")["ratio_u"].median().round(2).to_string())
    print("\n共同作息解释的份额 中位")
    print(df.groupby("date")["common"].median().round(3).to_string())
    print("\n每秒成交笔数均值 lam 中位")
    print(df.groupby("date")["lam"].median().round(2).to_string())
    print("\n一秒内最多簇数 实际/对照 中位")
    print(df.groupby("date")[["maxcs", "maxcsm"]].median().round(1).to_string())
    print("\n整体 ratio_p 分位", np.round(df.ratio_p.quantile([.1, .25, .5, .75, .9]).values, 2),
          " 均值", round(df.ratio_p.mean(), 2))
