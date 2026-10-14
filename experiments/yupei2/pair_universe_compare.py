#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""同一套统计量在「股票白名单」与「未过滤」两个口径下的逐项对照。

两个口径唯一允许的差别是：未过滤版本把基金/债券也当成 X 的候选（也能当配对的一端），
过滤版本在读数据与建 X 候选集之前就把它们踢掉。为了公平，下面所有配对层面的统计量
都只在「股票-股票」这个共同子集上算，这样差值就只来自「谁有资格当 X」。

用法:
  python experiments/yupei2/pair_universe_compare.py <未过滤目录> <过滤目录> <date> [<date2>]
"""
import os
import sys
import json
import struct
import numpy as np
from scipy.stats import spearmanr

BACKUP = "/hdd/user_home_unsafe/chenzongwei/yupei_dist_backup"
SYMBOL_MAP = "/ssd_data/data/basic_info/symbol_map.csv"


def read_codes(p):
    return [l.strip() for l in open(p) if l.strip()]


def load(root, date, sm):
    dd = os.path.join(root, str(date))
    codes = read_codes(os.path.join(dd, "codes.txt"))
    n = len(codes)

    def rd(nm):
        return np.fromfile(os.path.join(dd, nm), dtype="<f4").reshape(n, n)

    o = {"codes": codes, "n": n, "G": rd("G.bin"), "Z": rd("Z.bin"), "ZS": rd("ZS.bin"),
         "RU": rd("RHO_U.bin"), "X0": rd("X0.bin").astype(np.int64),
         "rho": rd("rho.bin"), "GS": rd("GSTAR.bin")}
    o["E"] = np.fromfile(os.path.join(dd, "E.bin"), dtype="<f8")
    o["is_stock"] = np.array([c in sm for c in codes])
    raw = open(os.path.join(BACKUP, str(date), "stats.bin"), "rb").read()
    m = struct.unpack_from("<Q", raw, 0)[0]
    o["nt"] = np.frombuffer(raw, dtype="<f8", count=m * 14, offset=8).reshape(m, 14)[:, 0]
    return o


def stock_pair_mask(A):
    iu = np.triu_indices(A["n"], 1)
    m = A["is_stock"][iu[0]] & A["is_stock"][iu[1]]
    return iu, m


def stats(A, iu, m, tag):
    n = A["n"]
    ib, ic = iu[0][m], iu[1][m]
    x0 = A["X0"][ib, ic]
    valid = A["is_stock"][np.clip(x0, 0, n - 1)] & (A["RU"][ib, ic] != 0)
    ib, ic, x0 = ib[valid], ic[valid], x0[valid]

    G = A["G"][ib, ic].astype(np.float64)
    Z = A["Z"][ib, ic].astype(np.float64)
    ZS = A["ZS"][ib, ic].astype(np.float64)
    RU = A["RU"][ib, ic].astype(np.float64)
    p = A["rho"][x0, ib].astype(np.float64)
    q = A["rho"][x0, ic].astype(np.float64)
    r = A["rho"][ib, ic].astype(np.float64)
    eb, ec = np.sqrt(A["E"][ib]), np.sqrt(A["E"][ic])
    single = np.maximum(p, q)

    # 条件冗余残差：按 (p,q) 分位网格估 r 的条件均值，再看 G 比「同剖面的典型冗余」高多少
    nb = 24
    bp = np.quantile(p, np.linspace(0, 1, nb + 1))
    bq = np.quantile(q, np.linspace(0, 1, nb + 1))
    bp[0] -= 1e-9
    bq[0] -= 1e-9
    Ip = np.clip(np.searchsorted(bp, p) - 1, 0, nb - 1)
    Iq = np.clip(np.searchsorted(bq, q) - 1, 0, nb - 1)
    rhat = np.zeros_like(r)
    for i in range(nb):
        for j in range(nb):
            s = (Ip == i) & (Iq == j)
            if s.any():
                rhat[s] = r[s].mean()
    D = np.sqrt(eb * eb + ec * ec + 2.0 * rhat * eb * ec)
    Gpred = (eb * p + ec * q) / D - single
    ex = G - Gpred

    cnt = np.bincount(x0, minlength=n)
    srt = np.sort(cnt)[::-1]
    return {
        "口径": tag,
        "有效配对数": int(len(G)),
        "G_p10_50_90": [round(float(np.percentile(G, x)), 5) for x in (10, 50, 90)],
        "Z_gt2_share": round(float((Z > 2).mean()), 5),
        "ZS_gt2_share": round(float((ZS > 2).mean()), 5),
        "X0_被选中的股票数": int((cnt > 0).sum()),
        "X0_前10名占比": round(float(srt[:10].sum() / len(G)), 4),
        "X0_第一名": A["codes"][int(np.argmax(cnt))],
        "X0_第一名占比": round(float(cnt.max() / len(G)), 4),
        "Spearman_G_vs_log笔数对": round(float(spearmanr(G, np.log10(
            np.maximum(A["nt"][ib], 1) * np.maximum(A["nt"][ic], 1))).statistic), 4),
        "剖面能解释的G方差占比": round(float(1.0 - ex.var() / G.var()), 4),
        "残差std比G的std": round(float(ex.std() / G.std()), 4),
        "残差与G的Spearman": round(float(spearmanr(ex, G).statistic), 4),
        "残差与rhat减r的Spearman": round(float(spearmanr(rhat - r, ex).statistic), 4),
    }


def main():
    un_root, so_root, date = sys.argv[1], sys.argv[2], int(sys.argv[3])
    sm = set(l.split(",")[0] for l in open(SYMBOL_MAP).read().splitlines()[1:] if l)
    A = load(un_root, date, sm)
    B = load(so_root, date, sm)
    assert A["codes"] == B["codes"], "两个口径的 codes.txt 行序必须一致才能逐对比较"

    iu, m = stock_pair_mask(A)
    ra = stats(A, iu, m, "未过滤（基金债券也能当 X 和配对的一端）")
    rb = stats(B, iu, m, "股票白名单")

    print("=" * 78)
    print(f"{date}  两侧都只在「股票-股票」这一个共同配对子集上统计")
    print("=" * 78)
    keys = [k for k in ra if k != "口径"]
    w = max(len(k) for k in keys)
    print(f"{'统计量'.ljust(w)}   {'未过滤':>14}   {'股票白名单':>14}   变化")
    for k in keys:
        va, vb = ra[k], rb[k]
        if isinstance(va, list):
            d = f"{va} → {vb}"
        elif isinstance(va, float):
            d = f"{vb - va:+.4f}"
        else:
            d = "—" if va == vb else f"{va} → {vb}"
        print(f"{k.ljust(w)}   {str(va):>14}   {str(vb):>14}   {d}")

    # X0 选票的重合度：同一对股票，两个口径选的 X 一样吗
    ib, ic = iu[0][m], iu[1][m]
    xa, xb = A["X0"][ib, ic], B["X0"][ib, ic]
    ok = A["is_stock"][np.clip(xa, 0, A["n"] - 1)] & (A["RU"][ib, ic] != 0)
    same = (xa[ok] == xb[ok]).mean()
    print(f"\n同一对股票、两个口径选出同一个 X0 的比例：{same:.4f}")
    print("（这个比例低不代表算错，它是这次对照的核心：换掉 X 的候选集，选谁当 X0 大面积改变）")

    out = {"date": date, "未过滤": ra, "股票白名单": rb, "X0一致率": float(same)}
    if len(sys.argv) > 4:
        d2 = int(sys.argv[4])
        A2, B2 = load(un_root, d2, sm), load(so_root, d2, sm)
        iu2, m2 = stock_pair_mask(A2)
        ra2 = stats(A2, iu2, m2, "未过滤")
        rb2 = stats(B2, iu2, m2, "股票白名单")
        out[str(d2)] = {"未过滤": ra2, "股票白名单": rb2}
        print(f"\n{d2} 的对照也已写入 JSON（跨日稳定性见下）")
        # 跨日
        ia = {c: i for i, c in enumerate(A["codes"])}
        ia2 = {c: i for i, c in enumerate(A2["codes"])}
        common = sorted(set(A["codes"]) & set(A2["codes"]))
        ca = np.array([ia[c] for c in common])
        ca2 = np.array([ia2[c] for c in common])
        okc = np.array([c in sm for c in common])
        iu3 = np.triu_indices(len(common), 1)
        mc = okc[iu3[0]] & okc[iu3[1]]
        for tag, M1, M2, key in (("未过滤", "Z", "Z", "Z"), ("股票白名单", "Z", "Z", "Z")):
            pass
        def cross(M, key):
            a = M["ZS"][np.ix_(ca, ca)][iu3][mc]
            b = M["ZS"][np.ix_(ca2, ca2)][iu3][mc]
            return round(float(spearmanr(a, b).statistic), 4)
        out["跨日Spearman_ZS_未过滤"] = cross(A, "ZS")
        out["跨日Spearman_ZS_白名单"] = cross(B, "ZS")
        print(f"跨日 Spearman(ZS)  未过滤 {out['跨日Spearman_ZS_未过滤']}  "
              f"白名单 {out['跨日Spearman_ZS_白名单']}")

    p = f"/tmp/yupei2_universe_compare_{date}.json"
    with open(p, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("写入", p)


if __name__ == "__main__":
    main()
