# -*- coding: utf-8 -*-
"""第 06 期：成交落在每分钟的第几秒。加载、过滤、对账、图的中间量。

口径
- 记录 = flag 66/83 且 turnover>0，时间戳用交易所的 exchtime
- 连续竞价 9:30:00-11:30:00 与 13:00:00-14:57:00，两端都算，共 14,222 秒
- 秒位 = 秒字段 0-59；第 0 秒一天出现 239 次，其余 237 次
- 刻度 = 时间戳截到 10 毫秒，同刻度多条算一次撮合
"""
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

SB = Path("/home/chenzongwei/rust_pyfunc/sandbox_second_census")
DATES = [20240205, 20240329, 20240813, 20240930, 20241008, 20250407, 20251127, 20260311]
ROOT = "/ssd_data/data/vars"
N_SS = 60


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
    recs, rows = [], []
    meta = None
    for d in dates:
        o = json.load(open(SB / f"out/{d}.json"))
        meta = o["meta"]
        for r in o["recs"]:
            recs.append(r)
            rows.append({"code": r["code"], "date": d, "s": r["s"], "e": r["e"], "tamt": r["amt"], "tvol": r["vol"]})
    df = pd.DataFrame(rows)
    df["code"] = df["code"].astype(str).str.zfill(6)
    n = len(df)
    A = {}
    for k, dt in [("cnt", np.int32), ("tick", np.int32), ("amt60", np.float64),
                  ("vol60", np.float64), ("cnt2", np.float64), ("amt2", np.float64),
                  ("share1", np.float64), ("share2", np.float64)]:
        M = np.zeros((n, N_SS), dt)
        for i, r in enumerate(recs):
            M[i] = r[k]
        A[k] = M
    S = np.zeros((n, 4, N_SS), np.int32)
    for i, r in enumerate(recs):
        S[i] = r["seg"]
    A["seg"] = S
    A["occ"] = np.array(meta["occ"], float)
    A["occ_seg"] = np.array(meta["occ_seg"], float)
    A["seg_s"] = np.array(meta["seg_s"], float)
    A["n_sec"] = meta["n_sec"]

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
    f = ((df.st == 1) & (df.hi != df.lo) & df.age_ok & df.amt.notna()).values
    df = df[f].reset_index(drop=True)
    A = {k: (v[f] if isinstance(v, np.ndarray) and v.shape[0] == f.shape[0] else v) for k, v in A.items()}
    # 活跃度五分位：每天按记录数切，每组占当天两成
    df["q"] = 0
    for _, g in df.groupby("date"):
        df.loc[g.index, "q"] = pd.qcut(g.s, 5, labels=False, duplicates="drop")
    return df, A


def curves(df, A):
    """每股票日 60 点的相对强度、刻度相对强度、每笔金额倍数、分钟间波动比。"""
    occ = A["occ"]
    n_sec = A["n_sec"]
    s = df.s.values[:, None]
    base = s / n_sec
    lam = A["cnt"] / occ
    inten = lam / base
    lam_t = A["tick"] / occ
    base_t = df.e.values[:, None] / n_sec
    inten_t = lam_t / base_t
    amt_all = A["amt60"].sum(1, keepdims=True)
    per_amt = np.divide(A["amt60"], A["cnt"], out=np.full_like(A["amt60"], np.nan), where=A["cnt"] > 0)
    per_amt0 = amt_all / s
    amt_mult = per_amt / per_amt0
    mean = lam
    var = np.maximum(A["cnt2"] / occ - mean ** 2, 0)
    fano = np.divide(var, mean, out=np.full_like(var, np.nan), where=mean > 0)
    disp = np.sqrt(fano)
    m1 = A["share1"] / occ
    v1 = np.maximum(A["share2"] / occ - m1 ** 2, 0)
    share_cv = np.sqrt(v1) / m1
    return {"inten": inten, "inten_t": inten_t, "amt_mult": amt_mult, "disp": disp,
            "share_cv": share_cv, "lam": lam}


def seg_curves(df, A):
    seg_s = A["seg_s"]
    occ_seg = A["occ_seg"]
    tot = A["seg"].sum(2, keepdims=True)
    base = tot / seg_s[None, :, None]
    lam = A["seg"] / occ_seg[None, :, :]
    return lam / np.where(base > 0, base, np.nan)


if __name__ == "__main__":
    df, A = load()
    pd.set_option("display.width", 220)
    print("过滤后", len(df), df.groupby("date").size().to_dict())
    df["rat"] = df.tamt / (df.amt * 1000)
    print("\n对账 逐笔成交额 / 日频")
    print(df.groupby("date")["rat"].quantile([.05, .5, .95]).round(6).to_string())
    C = curves(df, A)
    med = np.median(C["inten"], 0)
    print("\n图一 相对强度 0-9 秒中位", np.round(med[:10], 3))
    print("图一 最小三个", np.round(np.sort(med)[:3], 3), "最大三个", np.round(np.sort(med)[-3:], 3))
    print("\n图二 波动比 0-9 秒中位", np.round(np.median(C["disp"], 0)[:10], 2))
    print("图二 分位", np.round(np.nanquantile(C["disp"], [.1, .5, .9]), 2))
    print("\n图三 每笔金额倍数 0-9 秒中位", np.round(np.median(C["amt_mult"], 0)[:10], 3))
    print("\n每秒笔数中位", round(float(np.median(C["lam"])), 2))
