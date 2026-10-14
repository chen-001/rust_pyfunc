#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""测试「把 X0 换成补缺口最大的那只」（X0* = argmax[rho(X,U_BC) - max(rho(X,B),rho(X,C))]）。

六个方面：
  T1 分散度：X0 与 X0* 各用到多少只股票，集中度
  T2 逐对跨日：同一对股票两天是不是选了同一只 X；Z 与 ZS 的跨日 Spearman
  T3 尾部对齐：各自取前 0.5%/1%/2%/5% 的配对集合做 Jaccard（抵消两路尾部厚薄不同的干扰）
  T4 X0* 到底在挑什么：在 X0 与 X0* 上分别看 |p-q|（两个伙伴的平衡度）与 rho(X,U)
  T5 条件冗余分解：G 与 GSTAR 各自有多少方差能被 (p,q) 剖面解释掉
  T6 与活跃度的关系：Z、ZS 各自与 log 成交笔数对的 Spearman

用法: python experiments/yupei2/pair_x0star_test.py <pair_out_dir> <date1> <date2>
"""
import os
import sys
import json
import numpy as np
from scipy.stats import spearmanr

SM = set(l.split(",")[0]
         for l in open("/ssd_data/data/basic_info/symbol_map.csv").read().splitlines()[1:] if l)


def load(root, date):
    dd = os.path.join(root, str(date))
    codes = [l.strip() for l in open(os.path.join(dd, "codes.txt")) if l.strip()]
    n = len(codes)
    o = {"codes": codes, "n": n,
         "is_stock": np.array([c in SM for c in codes])}
    for nm in ("rho", "G", "GSTAR", "Z", "ZS", "X0", "X0STAR", "RHO_U", "RHO_US", "SINGLE", "SINGLE_S"):
        o[nm] = np.fromfile(os.path.join(dd, nm + ".bin"), dtype="<f4").reshape(n, n)
    o["X0"] = o["X0"].astype(np.int64)
    o["X0STAR"] = o["X0STAR"].astype(np.int64)
    return o


def pairs_of(A):
    n = A["n"]
    iu = np.triu_indices(n, 1)
    m = A["is_stock"][iu[0]] & A["is_stock"][iu[1]]
    return iu[0][m], iu[1][m]


def profile_resid(Gv, p, q, eb, ec, single, nb=24):
    """把 G 对 (p,q) 剖面做条件平均后的残差。返回残差与「剖面能解释的方差占比」。"""
    bp = np.quantile(p, np.linspace(0, 1, nb + 1))
    bq = np.quantile(q, np.linspace(0, 1, nb + 1))
    bp[0] -= 1e-9
    bq[0] -= 1e-9
    bk = (np.clip(np.searchsorted(bp, p) - 1, 0, nb - 1) * nb
          + np.clip(np.searchsorted(bq, q) - 1, 0, nb - 1))
    return bk


def rhat_of(bk, r, nb=24):
    cnt = np.bincount(bk, minlength=nb * nb).astype(np.float64)
    sg = np.bincount(bk, weights=r, minlength=nb * nb)
    return np.where(cnt > 0, sg / np.maximum(cnt, 1), 0.0)[bk]


def main():
    root, d1, d2 = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    A = load(root, d1)
    B = load(root, d2)
    n = A["n"]
    out = {"date_pair": [d1, d2]}

    # ---------------- T1 分散度 ----------------
    print("=" * 76)
    print("T1 分散度（X0 按 rho 最大挑，X0* 按补缺口最大挑）")
    print("=" * 76)
    t1 = {}
    for tag, M in (("X0", A["X0"]), ("X0star", A["X0STAR"])):
        ib, ic = pairs_of(A)
        v = M[ib, ic]
        cnt = np.bincount(v, minlength=n)
        srt = np.sort(cnt)[::-1]
        t1[tag] = {"被选中的股票数": int((cnt > 0).sum()),
                   "前10名占比": round(float(srt[:10].sum() / len(v)), 4),
                   "前100名占比": round(float(srt[:100].sum() / len(v)), 4),
                   "第一名": A["codes"][int(np.argmax(cnt))],
                   "第一名占比": round(float(cnt.max() / len(v)), 4)}
        print(f"  {tag:<7} 用到 {t1[tag]['被选中的股票数']:>5} 只  前10名占 {t1[tag]['前10名占比']:.3f}  "
              f"前100名占 {t1[tag]['前100名占比']:.3f}  第一名 {t1[tag]['第一名']} 占 {t1[tag]['第一名占比']:.3f}")
    out["T1"] = t1

    # ---------------- 共同配对 ----------------
    common = sorted(set(A["codes"]) & set(B["codes"]))
    ia = {c: i for i, c in enumerate(A["codes"])}
    ibx = {c: i for i, c in enumerate(B["codes"])}
    ca = np.array([ia[c] for c in common])
    cb = np.array([ibx[c] for c in common])
    ok = np.array([c in SM for c in common])
    iu = np.triu_indices(len(common), 1)
    m = ok[iu[0]] & ok[iu[1]]
    r0, r1 = iu[0][m], iu[1][m]
    print(f"\n共同股票 {len(common)} 只，共同股票对 {len(r0):,} 对")

    # ---------------- T2 逐对跨日 ----------------
    print()
    print("=" * 76)
    print("T2 逐对跨日：同一对股票两天是不是选了同一只 X")
    print("=" * 76)
    t2 = {}
    for tag, key in (("X0", "X0"), ("X0star", "X0STAR")):
        a = A[key][ca][:, ca][r0, r1]
        b = B[key][cb][:, cb][r0, r1]
        same = float((a == b).mean())
        t2[tag + "_一致率"] = round(same, 4)
        print(f"  {tag:<7} 两天选出同一只的比例 {same:.4f}")
    for tag, key in (("Z", "Z"), ("ZS", "ZS"), ("G", "G"), ("GSTAR", "GSTAR"), ("rho", "rho")):
        a = A[key][ca][:, ca][r0, r1]
        b = B[key][cb][:, cb][r0, r1]
        r = float(spearmanr(a, b).statistic)
        t2[tag + "_跨日Spearman"] = round(r, 4)
        print(f"  {tag:<7} 跨日 Spearman {r:.4f}")
    out["T2"] = t2

    # ---------------- T3 尾部对齐 ----------------
    print()
    print("=" * 76)
    print("T3 尾部对齐：各自取前 k% 的配对，看两天重合多少（Jaccard）")
    print("=" * 76)
    t3 = {}
    ij = np.array([r0, r1]).T
    for tag, key in (("Z", "Z"), ("ZS", "ZS")):
        a = A[key][ca][:, ca][r0, r1]
        b = B[key][cb][:, cb][r0, r1]
        row = {}
        for q in (0.5, 1.0, 2.0, 5.0):
            ha = set(map(tuple, ij[a >= np.percentile(a, 100 - q)]))
            hb = set(map(tuple, ij[b >= np.percentile(b, 100 - q)]))
            j = len(ha & hb) / max(len(ha | hb), 1)
            row[f"前{q}%"] = round(float(j), 4)
        t3[tag] = row
        print(f"  {tag:<4} " + "  ".join(f"前{k} {v:.4f}" for k, v in row.items()))
    out["T3"] = t3

    # ---------------- T4 X0* 在挑什么 ----------------
    print()
    print("=" * 76)
    print("T4 X0* 到底在挑什么")
    print("=" * 76)
    t4 = {}
    for tag, kx, kru, ks in (("X0", "X0", "RHO_U", "SINGLE"), ("X0star", "X0STAR", "RHO_US", "SINGLE_S")):
        ib, ic = pairs_of(A)
        x = A[kx][ib, ic]
        p = A["rho"][x, ib].astype(np.float64)
        q = A["rho"][x, ic].astype(np.float64)
        ru = A[kru][ib, ic].astype(np.float64)
        sg = A[ks][ib, ic].astype(np.float64)
        t4[tag] = {
            "两个伙伴相似度之差 |p-q| 中位": round(float(np.median(np.abs(p - q))), 4),
            "rho(X,U) 中位": round(float(np.median(ru)), 4),
            "被选中的 X 的 max(p,q) 中位": round(float(np.median(sg)), 4),
            "增益 G 中位": round(float(np.median(ru - sg)), 5),
        }
        print(f"  {tag:<7} |p-q| 中位 {t4[tag]['两个伙伴相似度之差 |p-q| 中位']:.4f}   "
              f"rho(X,U) 中位 {t4[tag]['rho(X,U) 中位']:.4f}   "
              f"max(p,q) 中位 {t4[tag]['被选中的 X 的 max(p,q) 中位']:.4f}   "
              f"增益中位 {t4[tag]['增益 G 中位']:.5f}")
    out["T4"] = t4

    # ---------------- T5 条件冗余分解 ----------------
    print()
    print("=" * 76)
    print("T5 条件冗余分解：G 与 GSTAR 各自有多少方差能被 (p,q) 剖面解释掉")
    print("=" * 76)
    eb = np.sqrt(np.fromfile(os.path.join(root, str(d1), "E.bin"), dtype="<f8"))
    t5 = {}
    ib, ic = pairs_of(A)
    r = A["rho"][ib, ic].astype(np.float64)
    e1 = eb[ib]
    e2 = eb[ic]
    for tag, kx, ktarget, ks in (("G", "X0", "RHO_U", "SINGLE"), ("GSTAR", "X0STAR", "RHO_US", "SINGLE_S")):
        x = A[kx][ib, ic]
        p = A["rho"][x, ib].astype(np.float64)
        q = A["rho"][x, ic].astype(np.float64)
        g = A[ktarget][ib, ic].astype(np.float64) - A[ks][ib, ic].astype(np.float64)
        bk = profile_resid(g, p, q, e1, e2, None)
        rh = rhat_of(bk, r)
        pred = (e1 * p + e2 * q) / np.sqrt(e1 * e1 + e2 * e2 + 2 * rh * e1 * e2) - np.maximum(p, q)
        res = g - pred
        t5[tag] = {"G 的标准差": round(float(g.std()), 5),
                   "残差的标准差": round(float(res.std()), 5),
                   "剖面能解释的方差占比": round(float(1 - res.var() / g.var()), 4)}
        print(f"  {tag:<6} G 的 std {t5[tag]['G 的标准差']:.5f}   残差 std {t5[tag]['残差的标准差']:.5f}   "
              f"剖面解释掉 {t5[tag]['剖面能解释的方差占比']*100:.2f}% 的方差")
    out["T5"] = t5

    # ---------------- T6 与活跃度的关系 ----------------
    print()
    print("=" * 76)
    print("T6 与活跃度的关系：Z / ZS 与 log 成交笔数对的 Spearman")
    print("=" * 76)
    import struct
    raw = open(os.path.join("/hdd/user_home_unsafe/chenzongwei/yupei_dist_backup", str(d1), "stats.bin"), "rb").read()
    mm = struct.unpack_from("<Q", raw, 0)[0]
    nt = np.frombuffer(raw, dtype="<f8", count=mm * 14, offset=8).reshape(mm, 14)[:, 0]
    lnt = np.log10(np.maximum(nt[ib], 1) * np.maximum(nt[ic], 1))
    t6 = {}
    for tag, key in (("Z", "Z"), ("ZS", "ZS")):
        v = A[key][ib, ic].astype(np.float64)
        t6[tag] = round(float(spearmanr(v, lnt).statistic), 4)
        print(f"  {tag:<4} Spearman {t6[tag]:.4f}")
    out["T6"] = t6

    p = f"/tmp/yupei2_x0star_test_{d1}_{d2}.json"
    with open(p, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("\n写入", p)


if __name__ == "__main__":
    main()
