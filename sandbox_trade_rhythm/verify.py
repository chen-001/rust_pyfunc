# -*- coding: utf-8 -*-
"""小样本核对：用 pandas 独立重算 Rust 的每个数，逐只对比。

口径（与 Rust 一致）
- 记录 = flag 66/83 且 turnover>0
- 事件 = 同一时间戳（截到 10 毫秒）算一次
- 连续竞价 9:30-11:30 与 13:00-14:57，逐秒 14,220 桶
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

SB = Path("/home/chenzongwei/rust_pyfunc/sandbox_trade_rhythm")
DATE = 20240329
T_AM, T_PM = 7200, 7020
T = T_AM + T_PM
NB = 33


def sidx(sec):
    i = np.full(len(sec), -1, dtype=np.int64)
    am = (sec >= 34200) & (sec <= 41400)
    pm = (sec >= 46800) & (sec <= 53820)
    i[am] = np.minimum(sec[am] - 34200, T_AM - 1)
    i[pm] = T_AM + np.minimum(sec[pm] - 46800, T_PM - 1)
    return i


def lorenz(cnt):
    v = np.sort(cnt)[::-1].astype(float)
    tot = v.sum()
    out = []
    for i in range(201):
        target = i / 200 * T
        k = min(int(np.floor(target)), T)
        frac = target - np.floor(target)
        extra = v[k] * frac if k < T else 0.0
        out.append((v[:k].sum() + extra) / tot)
    return np.array(out)


def main():
    out = json.load(open(SB / "out/tiny.json"))
    codes = [r["code"] for r in out["recs"]]
    ecnt = np.zeros((len(codes), T))
    rows = {}
    for i, c in enumerate(codes):
        d = pd.read_csv(f"/ssd_data/stock/{DATE}/transaction/{c}_{DATE}_transaction.csv",
                        usecols=["exchtime", "turnover", "flag"])
        d = d[(d.flag.isin([66, 83])) & (d.turnover > 0)]
        us = np.asarray((d.exchtime.values + 8 * 3600 * 10**6) % (86400 * 10**6), dtype=np.int64)
        idx = sidx(us // 10**6)
        keep = idx >= 0
        us, idx = us[keep], idx[keep]
        rc = np.bincount(idx, minlength=T)
        tick = us // 10**4
        first = np.r_[True, tick[1:] != tick[:-1]]
        ec = np.bincount(idx[first], minlength=T)
        ecnt[i] = ec
        rows[c] = dict(s=int(keep.sum()), e=int(first.sum()),
                       hist=np.bincount(np.minimum(rc, NB - 1), minlength=NB),
                       lz=lorenz(rc), fano=float(ec.var() / ec.mean()))
    mkt = ecnt.sum(0)

    bad = 0
    for i, r in enumerate(out["recs"]):
        c = r["code"]
        v = rows[c]
        sem = float((ecnt[i] * mkt).sum())
        r2_py = np.corrcoef(ecnt[i], mkt)[0, 1] ** 2
        mean_e = r["e"] / T
        mmean, mvar = mkt.mean(), mkt.var()
        r2_rust = ((r["sem"] - T * mmean * mean_e) / np.sqrt((r["sx2"] - T * mean_e ** 2) * (mvar * T))) ** 2
        chk = {"s": r["s"] - v["s"], "e": r["e"] - v["e"], "fano": r["fano"] - v["fano"],
               "sem": r["sem"] - sem, "hist": int(np.abs(np.array(r["hist"]) - v["hist"]).sum()),
               "lz": float(np.abs(np.array(r["lz"]) - v["lz"]).max()), "r2": r2_rust - r2_py}
        worst = max(abs(x) for x in chk.values())
        if worst > 1e-3:
            bad += 1
            print(c, {k: round(vv, 6) if isinstance(vv, float) else vv for k, vv in chk.items()})
    print(f"核对 {len(codes)} 只，不一致 {bad} 只")


if __name__ == "__main__":
    main()
