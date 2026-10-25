# -*- coding: utf-8 -*-
"""小样本核对：用 pandas 独立重算一遍 Rust 的每个数，逐只对比。

口径（与 Rust 一致）
- 成交 = flag 66/83 且 turnover>0
- 连续竞价 = 9:30:00-11:30:00 与 13:00:00-14:57:00，逐秒 14,220 桶
- 簇 = 同一毫秒（exchtime 取到毫秒）
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SB = Path("/home/chenzongwei/rust_pyfunc/sandbox_trade_rhythm")
DATE = 20240329
LIMIT = 30
T_AM, T_PM = 7200, 7020
T = T_AM + T_PM
NB = 33


def bucket(sec):
    idx = np.full(len(sec), -1, dtype=np.int64)
    am = (sec >= 34200) & (sec <= 41400)
    pm = (sec >= 46800) & (sec <= 53820)
    idx[am] = np.minimum(sec[am] - 34200, T_AM - 1)
    idx[pm] = T_AM + np.minimum(sec[pm] - 46800, T_PM - 1)
    return idx


def load_one(code):
    p = f"/ssd_data/stock/{DATE}/transaction/{code}_{DATE}_transaction.csv"
    d = pd.read_csv(p, usecols=["exchtime", "turnover", "flag"])
    d = d[(d.flag.isin([66, 83])) & (d.turnover > 0)]
    day_us = (d.exchtime.values + 8 * 3600 * 10**6) % (86400 * 10**6)
    sec = day_us // 10**6
    ms = day_us // 10**4
    return sec, ms, d.turnover.values


def main():
    out = json.load(open(SB / f"out/tiny_{DATE}.json"))
    codes = [r["code"] for r in out["recs"]]
    cnt = np.zeros((len(codes), T), dtype=np.int64)
    rows = {}
    for i, c in enumerate(codes):
        sec, ms, tn = load_one(c)
        idx = bucket(sec)
        keep = idx >= 0
        h = np.bincount(idx[keep], minlength=T)
        cnt[i] = h
        m_keep = ms[keep]
        # 同一个毫秒相邻的算一簇
        newc = np.r_[True, m_keep[1:] != m_keep[:-1]]
        size = np.diff(np.r_[np.where(newc)[0], len(m_keep)])
        rows[c] = dict(s=int(keep.sum()), c=int(newc.sum()), s2=int((size.astype(np.int64) ** 2).sum()), hist=np.bincount(np.minimum(h, NB - 1), minlength=NB))

    tot = cnt.sum(axis=0).astype(float)
    p = tot / tot.sum()
    p2 = float((p ** 2).sum())

    bad = 0
    for i, r in enumerate(out["recs"]):
        c = r["code"]
        v = rows[c]
        x = cnt[i].astype(float)
        s = v["s"]
        d_op = float((x ** 2).sum() - 2 * s * (x * p).sum() + s * s * p2) / s
        d_ou = float((x ** 2).sum() - s * s / T) / s
        d_sp = v["s2"] * (1 - p2) / s
        d_su = v["s2"] * (1 - 1 / T) / s
        chk = dict(s=r["s"] - v["s"], c=r["c"] - v["c"], s2=r["s2"] - v["s2"],
                   d_op=r["d_op"] - d_op, d_ou=r["d_ou"] - d_ou,
                   d_sp=r["d_sp"] - d_sp, d_su=r["d_su"] - d_su,
                   hist=int(np.abs(np.array(r["hist"]) - v["hist"]).sum()),
                   d_mc=r["d_mc"] - d_sp)
        worst = max(abs(chk["s"]), abs(chk["c"]), abs(chk["s2"]), abs(chk["d_op"]), abs(chk["d_ou"]), abs(chk["d_sp"]), abs(chk["d_su"]), chk["hist"])
        if worst > 1e-9:
            bad += 1
            print(c, {k: round(vv, 6) if isinstance(vv, float) else vv for k, vv in chk.items()})
    print(f"核对 {len(codes)} 只，不一致 {bad} 只；p2={p2:.6e}，T_eff={1/p2:.0f}")
    print("比值样例:", [(r["code"], round(r["d_op"] / r["d_sp"], 3), round(r["d_ou"] / r["d_su"], 3)) for r in out["recs"][:5]])


if __name__ == "__main__":
    main()
