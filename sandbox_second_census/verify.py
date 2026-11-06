# -*- coding: utf-8 -*-
"""小样本核对：用 pandas 独立重算 Rust 的每个数，逐只对比。

口径与 Rust 一致
- 记录 = flag 66/83 且 turnover>0
- 时段 9:30:00-11:30:00 与 13:00:00-14:57:00，两端都算，共 14,222 秒
- 秒位 = 秒字段 0-59
- 刻度 = 时间戳截到 10 毫秒，同刻度多条算一次撮合
- 四段 = 开盘 5 分钟 / 上午其余 / 午后 / 最后 5 分钟
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

SB = Path("/home/chenzongwei/rust_pyfunc/sandbox_second_census")
DATE = 20240329
AM = 7201
N_SEC = 14222
N_SS = 60


def idx_of(sec):
    i = np.full(len(sec), -1, dtype=np.int64)
    am = (sec >= 34200) & (sec <= 41400)
    pm = (sec >= 46800) & (sec <= 53820)
    i[am] = sec[am] - 34200
    i[pm] = AM + sec[pm] - 46800
    return i


def seg_of(i):
    return np.where(i < 300, 0, np.where(i < AM, 1, np.where(i < AM + 6720, 2, 3)))


def check(r):
    c = r["code"]
    d = pd.read_csv(f"/ssd_data/stock/{DATE}/transaction/{c}_{DATE}_transaction.csv",
                    usecols=["exchtime", "turnover", "volume", "flag"])
    d = d[(d.flag.isin([66, 83])) & (d.turnover > 0)]
    us = (d.exchtime.values.astype(np.int64) + 8 * 3600 * 10**6) % (86400 * 10**6)
    sec = us // 10**6
    tick = us // 10**4
    i = idx_of(sec)
    keep = i >= 0
    i, us, tick, amt, vol = i[keep], us[keep], tick[keep], d.turnover.values[keep], d.volume.values[keep]
    amt_all, vol_all = d.turnover.values.sum(), d.volume.values.sum()
    first = np.r_[True, tick[1:] != tick[:-1]]
    c_s = np.bincount(i, minlength=N_SEC)
    ii0 = np.arange(N_SEC)
    mm_all = np.where(ii0 < AM, np.minimum(ii0 // 60, 119), 120 + np.minimum((ii0 - AM) // 60, 116))
    tot = np.bincount(mm_all, weights=c_s, minlength=237)
    y = c_s / tot[mm_all]
    ii = np.arange(N_SEC)
    ss_of_i = ((np.where(ii < AM, 34200, 46800) + np.where(ii < AM, ii, ii - AM)) % 60).astype(np.int64)
    ss = ((np.where(i < AM, 34200, 46800) + np.where(i < AM, i, i - AM)) % 60).astype(np.int64)
    g = seg_of(i)

    out = {
        "s": len(i),
        "e": int(first.sum()),
        "amt": amt_all,
        "vol": vol_all,
        "cnt": np.bincount(ss, minlength=N_SS),
        "tick": np.bincount(ss[first], minlength=N_SS),
        "amt60": np.bincount(ss, weights=amt, minlength=N_SS),
        "vol60": np.bincount(ss, weights=vol, minlength=N_SS),
        "cnt2": np.bincount(ss_of_i, weights=np.bincount(i, minlength=N_SEC).astype(float) ** 2, minlength=N_SS),
        "amt2": np.bincount(ss_of_i, weights=np.bincount(i, weights=amt, minlength=N_SEC) ** 2, minlength=N_SS),
        "share1": np.bincount(ss_of_i, weights=y, minlength=N_SS),
        "share2": np.bincount(ss_of_i, weights=y ** 2, minlength=N_SS),
        "seg": np.stack([np.bincount(ss[g == k], minlength=N_SS) for k in range(4)]),
    }
    out["cnt2"] = np.rint(out["cnt2"]).astype(np.int64)
    return out


def main():
    o = json.load(open(SB / "out/tiny.json"))
    worst = 0.0
    for r in o["recs"]:
        t = check(r)
        for k in ["s", "e", "amt", "vol"]:
            a, b = float(r[k]), float(t[k])
            worst = max(worst, abs(a - b) / max(abs(b), 1.0))
        for k in ["cnt", "tick", "amt60", "vol60", "cnt2", "amt2", "share1", "share2"]:
            a = np.asarray(r[k], dtype=float)
            b = np.asarray(t[k], dtype=float)
            worst = max(worst, np.max(np.abs(a - b) / np.maximum(np.abs(b), 1.0)))
        a = np.asarray(r["seg"], dtype=float)
        b = t["seg"].astype(float)
        worst = max(worst, np.max(np.abs(a - b) / np.maximum(np.abs(b), 1.0)))
    print(f"核对股票数 {len(o['recs'])}，最大相对误差 {worst:.3e}")


if __name__ == "__main__":
    main()
