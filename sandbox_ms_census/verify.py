# -*- coding: utf-8 -*-
"""小样本核对：pandas 独立重算 Rust 的确定性统计量，并与日频成交额对账。"""
import numpy as np, pandas as pd
from load import load_day, daily, DATES
D = daily()
for d in [20240329, 20251127]:
    m, A = load_day(d, "s30")
    bad = {}; worst = 0
    ratios = []
    for k, c in enumerate(m["codes"]):
        t = pd.read_csv(f"/ssd_data/stock/{d}/transaction/{c}_{d}_transaction.csv")
        t = t[t.flag.isin([66, 83]) & (t.turnover > 0)].copy()
        ratios.append(t.turnover.sum() / 1000 / D["amt"].loc[d, c])
        cs = np.round(((t.exchtime.astype(float).values + 8 * 3600e6) % 86400e6) / 1e4).astype(np.int64)
        am = (cs >= 3420000) & (cs < 4140000); pm = (cs >= 4680000) & (cs < 5382000)
        idx = np.where(am, cs - 3420000, cs - 4680000 + 720000)
        t = t[am | pm].copy(); t["i"] = idx[am | pm]
        t["oid"] = np.where(t.flag == 66, t.bid_order, t.ask_order).astype(np.int64)
        def cmp(n, x):
            global worst
            y = A[n][k]; e = np.abs(np.asarray(x, float) - y.astype(float)).max()
            worst = max(worst, e)
            if e > 0: bad[n] = bad.get(n, 0) + 1
        cell = t.i.values % 100
        cmp("rec", np.bincount(cell, minlength=100))
        ev = t.groupby(["i", "flag", "oid"], sort=False).agg(vol=("volume", "sum")).reset_index().sort_values("i", kind="stable")
        I = ev.i.values; cmp("ev", np.bincount(I % 100, minlength=100))
        cmp("ev0", np.bincount(I[(I // 100) % 60 == 0] % 100, minlength=100))
        cmp("evx", np.bincount(I[I >= 30000] % 100, minlength=100))
        cmp("clk", np.bincount(((I // 100) % 3) * 100 + I % 100, minlength=300))
        # 间隔：跨午休不算
        sess = I < 720000; g = np.diff(I); same = sess[1:] == sess[:-1]
        cmp("gap", np.bincount(g[same & (g < 500)], minlength=500))
        edges = m["gap_edges"]; b = np.digitize(g, edges) - 1
        v = ev.vol.values; dr = ev.flag.values
        ok = same & (g < 6000)
        cmp("pair", np.bincount(b[ok], minlength=13))
        sd = dr[1:] == dr[:-1]; sdv = sd & (v[1:] == v[:-1])
        cmp("sd", np.bincount(b[ok & sd], minlength=13)); cmp("sdv", np.bincount(b[ok & sdv], minlength=13))
        # 快照
        s = pd.read_csv(f"/ssd_data/stock/{d}/market_data/{c}_{d}_market_data.csv", usecols=["time"]).time.values
        sc = (s // 10000000 * 3600 + s // 100000 % 100 * 60 + s // 1000 % 100) * 100 + s % 1000 // 10
        a2 = (sc >= 3420000) & (sc < 4140000); p2 = (sc >= 4680000) & (sc < 5382000)
        S = np.unique(np.where(a2, sc - 3420000, sc - 4680000 + 720000)[a2 | p2])
        occ = np.zeros(301, int); snap = np.zeros(300, int)
        for a, bb in zip(S[:-1], S[1:]):
            if (a < 720000) == (bb < 720000):
                occ[:min(bb - a, 300)] += 1
        j = np.searchsorted(S, I, side="right") - 1
        for e, p in zip(I, j):
            if p < 0 or p + 1 >= len(S): continue
            if (S[p] < 720000) != (S[p + 1] < 720000) or (S[p] < 720000) != (e < 720000): continue
            dd = e - S[p]
            if dd < 300: snap[dd] += 1
        cmp("occ", occ[:300]); cmp("snap", snap)
        sc2 = A["scal"][k]
        assert sc2[0] == len(t) and sc2[1] == len(ev) and sc2[2] == t.i.nunique(), (c, sc2[:3], len(t), len(ev))
    S = A["scal"]
    print(d, "stocks", len(m["codes"]), "worst abs diff", worst, "bad", bad)
    print("  amount ratio vs daily  min/med/max", np.min(ratios), np.median(ratios), np.max(ratios))
    print("  unsorted", S[:, 8].sum(), "edge-second records", S[:, 9].sum(), "nsnap med", np.median(S[:, 4]), "rec/ev", S[:, 0].sum() / S[:, 1].sum(),
          "ticks>=2 orders share", S[:, 3].sum() / S[:, 2].sum())
    print("  snap phases (first 8):", S[:8, 5:8].tolist(), "codes", m["codes"][:8])
