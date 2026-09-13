# -*- coding: utf-8 -*-
"""诊断：核对 sandbox 事件检测与对级匹配是否合理（抽样一对头部股）。"""
import pandas as pd
import numpy as np

DATE = 20240104
A, B = "000001", "600519"


def read(code):
    p = f"/ssd_data/stock/{DATE}/transaction/{code}_{DATE}_transaction.csv"
    df = pd.read_csv(p, usecols=["exchtime", "price", "volume", "turnover", "flag"])
    # 与 fast_csv_reader 相同的 adjust_afternoon
    off = 8 * 3600
    day = ((df.exchtime // 1_000_000) + off) % 86400
    t = df.exchtime + off * 1_000_000
    m = (day >= 9 * 3600 + 30 * 60) & (day <= 11 * 3600 + 30 * 60)
    pm = (day >= 13 * 3600) & (day <= 14 * 3600 + 57 * 60)
    t = pd.Series(np.where(m, t, np.where(pm, t - 90 * 60 * 1_000_000, np.nan)))
    df["t"] = t
    df = df.dropna().sort_values("t")
    return df


def detect(df):
    amt = df.turnover.values
    n = len(amt)
    inner_m = np.sort(amt)[int(n * 0.40)]
    inner_l = np.sort(amt)[int(n * 0.90)]
    is_l = amt >= inner_l
    is_m = (amt >= inner_m) & (amt < inner_l)
    is_s = amt < inner_m
    t = df.t.values.astype(np.int64)
    p = df.price.values
    f = df.flag.values
    ev = {}
    cand_m = t[(is_m) & (f == 66)]
    cand_s = t[(is_s) & (f == 66)]
    cand_l_sell = t[(is_l) & (f == 83)]
    ev["big_buy_m"] = cand_m
    ev["big_buy_s"] = cand_s
    ev["big_sell_l"] = cand_l_sell
    # sweep
    tb = t[(is_l) & (f == 66)]
    sw = 1200 * 10**6
    sweep = []
    j = 0
    for i in range(len(tb)):
        while tb[j] <= tb[i] - sw:
            j += 1
        if i - j + 1 >= 2:
            sweep.append(tb[i])
    ev["sweep_buy"] = np.array(sweep, dtype=np.int64)
    # ice
    idx = np.where(is_l)[0]
    order = np.lexsort((t[idx], p[idx]))  # 先价格后时间
    idx2 = idx[order]
    ice = []
    s = 0
    while s < len(idx2):
        e = s + 1
        while e < len(idx2) and p[idx2[e]] == p[idx2[s]]:
            e += 1
        if e - s >= 2:
            j = s
            for i in range(s, e):
                while t[idx2[j]] <= t[idx2[i]] - 180 * 10**6:
                    j += 1
                if i - j + 1 >= 2:
                    ice.append(t[idx2[i]])
        s = e
    ev["ice"] = np.array(ice, dtype=np.int64)
    # jump
    dp = np.abs(np.diff(p))
    thr = np.sort(dp)[int(n * 0.99)]
    jm = t[1:][np.abs(np.diff(p)) > thr]
    ev["jump"] = jm
    return ev, dict(n=n, inner_m=inner_m, inner_l=inner_l)


da = read(A)
db = read(B)
ea, ma = detect(da)
eb, mb = detect(db)
print(f"A={A} trades={ma['n']} inner_m={ma['inner_m']:.0f} inner_l={ma['inner_l']:.0f}")
print(f"B={B} trades={mb['n']} inner_m={mb['inner_m']:.0f} inner_l={mb['inner_l']:.0f}")
for k in ea:
    print(f"{k:12s} A={len(ea[k]):5d} B={len(eb[k]):5d}")

# 50 笔配对抽样 fwd 距离统计（A 事件 -> B 下一事件）
def pair_gaps(ta, tb, cap=1_000_000):
    ta = np.sort(ta)
    tb = np.sort(tb)
    j = 0
    gaps = []
    for a in ta:
        while j < len(tb) and tb[j] <= a:
            j += 1
        if j < len(tb):
            gaps.append(tb[j] - a)
    g = np.array(gaps)
    return g

for k in ea:
    g = pair_gaps(ea[k], eb[k])
    if len(g):
        hit = (g <= 60e6).mean()
        print(f"{k:12s} fwd n={len(g):5d} hit60={hit:.3f} med={np.median(g)/1e6:.1f}s mean={g.mean()/1e6:.1f}s")
    g2 = pair_gaps(eb[k], ea[k])
    if len(g2):
        hit2 = (g2 <= 60e6).mean()
        print(f"{'':12s} bwd(=B->A) n={len(g2):5d} hit60={hit2:.3f} med={np.median(g2)/1e6:.1f}s mean={g2.mean()/1e6:.1f}s")
