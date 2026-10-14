#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""配对互补矩阵的「条件冗余残差」——修正 G 的可检验性。

问题：G(B,C) = rho(X0,U_BC) - max(rho(X0,B),rho(X0,C)) 是 (p, q, r, eB, eC) 的确定函数，
其中 p=rho(X0,B)、q=rho(X0,C)、r=rho(B,C)。所以按活跃度分层做的那个 Z 只是把数据
标准化了一遍，不构成检验：空模型抽的配对本就来自同一个分布，Z 的均值必然为 0、
标准差必然为 1。真正能做检验的对照是「同相似度剖面的配对，通常有多冗余」：

    r_hat(p,q) = E[ r | p, q ]        （在 (p,q) 的分位网格里估）
    G_pred     = rho(X0,U) 在 r 取 r_hat 时的取值  - max(p,q)
    excess     = G - G_pred

excess 的符号等价于 sign(r_hat(p,q) - r)：这对伙伴各自跟 X0 都挺像，
但彼此之间比同剖面的其他配对更不像 —— 这才是用户那句「两种不同的承接方式」的可检验版本。

输出: /tmp/yupei2_cond_excess.json

用法: python experiments/yupei2/pair_conditional.py <pair_out_dir> <date> [<date2>]
"""
import os
import sys
import json
import numpy as np
from scipy.stats import spearmanr

SYMBOL_MAP = "/ssd_data/data/basic_info/symbol_map.csv"


def load(root, date, sm):
    dd = os.path.join(root, str(date))
    codes = [l.strip() for l in open(os.path.join(dd, "codes.txt")) if l.strip()]
    n = len(codes)

    def rd(nm):
        return np.fromfile(os.path.join(dd, nm), dtype="<f4").reshape(n, n)

    o = {"date": date, "codes": codes, "n": n, "rho": rd("rho.bin"), "G": rd("G.bin"),
         "RU": rd("RHO_U.bin"), "SG": rd("SINGLE.bin"), "X0": rd("X0.bin").astype(np.int64),
         "ZS": rd("ZS.bin")}
    o["E"] = np.fromfile(os.path.join(dd, "E.bin"), dtype="<f8")
    o["is_stock"] = np.array([c in sm for c in codes])
    return o


def stock_pairs(A):
    n = A["n"]
    iu = np.triu_indices(n, 1)
    m = A["is_stock"][iu[0]] & A["is_stock"][iu[1]]
    return iu[0][m], iu[1][m]


def rho_of(p, q, eb, ec, r):
    return (eb * p + ec * q) / np.sqrt(eb * eb + ec * ec + 2.0 * r * eb * ec)


def tri_resid(A, ib, ic, nb=24):
    """在给定的配对下标 (ib, ic) 上算条件冗余残差（三角残差）。
    只保留 B、C、X0 三者都是股票的配对。返回 (残差, G, r_hat - r)。"""
    keep = A["is_stock"][ib] & A["is_stock"][ic]
    ib, ic = ib[keep], ic[keep]
    x0 = A["X0"][ib, ic]
    keep = A["is_stock"][np.clip(x0, 0, A["n"] - 1)] & (A["RU"][ib, ic] != 0)
    ib, ic, x0 = ib[keep], ic[keep], x0[keep]
    p = A["rho"][x0, ib]
    q = A["rho"][x0, ic]
    r = A["rho"][ib, ic]
    eb = np.sqrt(A["E"][ib])
    ec = np.sqrt(A["E"][ic])
    G = A["RU"][ib, ic] - np.maximum(p, q)
    resid = float(np.abs(rho_of(p, q, eb, ec, r) - A["RU"][ib, ic]).max())
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
    Gpred = rho_of(p, q, eb, ec, rhat) - np.maximum(p, q)
    return G - Gpred, G, rhat - r, resid, (ib, ic)


def excess(A, ib, ic, nb=24):
    ex, G, tri, resid, _ = tri_resid(A, ib, ic, nb)
    stat = {
        "max_abs_identity_residual": resid,
        "n_pairs_used": int(len(ex)),
        "G_p10_50_90": [float(np.percentile(G, x)) for x in (10, 50, 90)],
        "excess_p10_50_90": [float(np.percentile(ex, x)) for x in (10, 50, 90)],
        "excess_positive_share": float((ex > 0).mean()),
        "std_ratio_excess_over_G": float(ex.std() / G.std()),
        "var_share_of_G_not_explained_by_profile": float(ex.var() / G.var()),
        "spearman_excess_vs_G": float(spearmanr(ex, G).statistic),
        "spearman_excess_vs_tri": float(spearmanr(ex, tri).statistic),
    }
    return ex, tri, stat


def main():
    root, date = sys.argv[1], int(sys.argv[2])
    sm = set(l.split(",")[0] for l in open(SYMBOL_MAP).read().splitlines()[1:] if l)
    A = load(root, date, sm)
    ib, ic = stock_pairs(A)
    ex_a, tri_a, stat = excess(A, ib, ic)
    out = {"date": date, "n_pairs": int(len(ib)), "stats": stat}
    for k, v in stat.items():
        print(f"  {k}: {v}")

    if len(sys.argv) > 3:
        date2 = int(sys.argv[3])
        B = load(root, date2, sm)
        ib2, ic2 = stock_pairs(B)
        ex_b, tri_b, stat2 = excess(B, ib2, ic2)
        out["date2"] = date2
        out["stats2"] = stat2
        # 两日在同一批共同股票上重算三角残差，再逐对比
        common = sorted(set(A["codes"]) & set(B["codes"]))
        ca = np.array([A["codes"].index(c) for c in common])
        cb = np.array([B["codes"].index(c) for c in common])
        iu2 = np.triu_indices(len(common), 1)
        ok = np.array([c in sm for c in common])
        m2 = ok[iu2[0]] & ok[iu2[1]]
        ta = tri_resid(A, ca[iu2[0]][m2], ca[iu2[1]][m2])[2]
        tb = tri_resid(B, cb[iu2[0]][m2], cb[iu2[1]][m2])[2]
        ga = tri_resid(A, ca[iu2[0]][m2], ca[iu2[1]][m2])[1]
        gb = tri_resid(B, cb[iu2[0]][m2], cb[iu2[1]][m2])[1]
        out["cross_day"] = {
            "common_stocks": len(common), "n_pairs": int(len(ta)),
            "spearman_triangle_resid": float(spearmanr(ta, tb).statistic),
            "spearman_G": float(spearmanr(ga, gb).statistic),
        }
        print(f"  跨日（共同股票 {len(common)}，共同配对 {len(ta)}）: "
              f"三角残差 Spearman = {out['cross_day']['spearman_triangle_resid']:.4f}, "
              f"G Spearman = {out['cross_day']['spearman_G']:.4f}")

    p = f"/tmp/yupei2_cond_excess_{date}.json"
    with open(p, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("写入", p)


if __name__ == "__main__":
    main()
