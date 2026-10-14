#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""截面相关性：G 构造的因子 vs「解释 G 的那个剖面」构造的因子。

背景。主会话此前说过「G 的 99.1% 方差能被 (p,q) 剖面解释掉」，那是配对层面的陈述。
用户指出做因子是在截面上比，配对层面被解释掉不等于截面因子没有增量。
所以这里直接算截面上的 Spearman。

三个东西，全部按同一批配对、同一批股票聚合成每股一个数：
  F_G     用 G 本身
  F_pred  用 G_pred，也就是把每对的 r 换成它所在 (p,q) 小格的典型 r 之后重算的 G
  F_res   用 G - G_pred，也就是扣掉剖面以后剩下的那一层
再额外给两个「纯剖面」的对照：
  F_maxrho  每股的 rho 行最大值（现有 r2_best_single 那条）
  F_basket  rho[X][U_S] 的贪心篮子 8 只版本（现有 r3_basket_rho8 那条）

聚合方式各给三种，因为不同聚合答出来的可能不一样：
  agg=A  X0 落在这只股票上时，那些配对的平均值（explain_gain_mean 的定义）
  agg=B  这只股票当伙伴时的行最大值（bestcomp_gain 的定义）
  agg=C  这只股票当伙伴时的行平均值

用法: python experiments/yupei2/pair_xs_corr.py <pair_out_dir> <date> [second]
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
    o = {"codes": codes, "n": n, "is_stock": np.array([c in SM for c in codes])}
    for nm in ("rho", "G", "GSTAR", "Z", "ZS", "X0", "X0STAR", "RHO_U", "RHO_US", "SINGLE", "SINGLE_S"):
        o[nm] = np.fromfile(os.path.join(dd, nm + ".bin"), dtype="<f4").reshape(n, n)
    o["E"] = np.fromfile(os.path.join(dd, "E.bin"), dtype="<f8")
    o["X0"] = o["X0"].astype(np.int64)
    o["X0STAR"] = o["X0STAR"].astype(np.int64)
    return o


def agg_x0(idx, val, n):
    """按 X0 聚合：每股在它被选中的那些配对上的平均值与个数。"""
    cnt = np.bincount(idx, minlength=n).astype(np.float64)
    s = np.bincount(idx, weights=val, minlength=n)
    return np.where(cnt > 0, s / np.maximum(cnt, 1), np.nan), cnt


def agg_row(M, ok, n):
    """按行聚合：每股与伙伴之间的最大值和平均值（跳过自己、跳过非个股）。"""
    A = np.array(M, dtype=np.float64, copy=True)
    np.fill_diagonal(A, -np.inf)
    A[:, ~ok] = -np.inf
    mx = A.max(axis=1)
    mx[~ok] = np.nan
    B = np.array(M, dtype=np.float64, copy=True)
    np.fill_diagonal(B, np.nan)
    B[:, ~ok] = np.nan
    B[~ok, :] = np.nan
    mn = np.nanmean(B, axis=1)
    return mx, mn


def build(A, path):
    """按给定路径（first 用 X0/RHO_U/SINGLE，second 用 X0STAR/RHO_US/SINGLE_S）
    造出 F_G / F_pred / F_res 三套，各含三种聚合。"""
    n = A["n"]
    ok = A["is_stock"]
    iu = np.triu_indices(n, 1)
    m = ok[iu[0]] & ok[iu[1]]
    ib, ic = iu[0][m], iu[1][m]
    kx = "X0" if path == "first" else "X0STAR"
    kt = "RHO_U" if path == "first" else "RHO_US"
    ks = "SINGLE" if path == "first" else "SINGLE_S"
    x = A[kx][ib, ic]
    p = A["rho"][x, ib].astype(np.float64)
    q = A["rho"][x, ic].astype(np.float64)
    r = A["rho"][ib, ic].astype(np.float64)
    e1 = np.sqrt(A["E"][ib])
    e2 = np.sqrt(A["E"][ic])
    Gv = A[kt][ib, ic].astype(np.float64) - A[ks][ib, ic].astype(np.float64)

    # 条件冗余预测：按 (p,q) 24x24 分位网格取 r 的格内均值
    nb = 24
    bp = np.quantile(p, np.linspace(0, 1, nb + 1))
    bq = np.quantile(q, np.linspace(0, 1, nb + 1))
    bp[0] -= 1e-9
    bq[0] -= 1e-9
    bk = (np.clip(np.searchsorted(bp, p) - 1, 0, nb - 1) * nb
          + np.clip(np.searchsorted(bq, q) - 1, 0, nb - 1))
    cnt = np.bincount(bk, minlength=nb * nb).astype(np.float64)
    sg = np.bincount(bk, weights=r, minlength=nb * nb)
    rh = np.where(cnt > 0, sg / np.maximum(cnt, 1), 0.0)[bk]
    pred = (e1 * p + e2 * q) / np.sqrt(e1 * e1 + e2 * e2 + 2 * rh * e1 * e2) - np.maximum(p, q)
    res = Gv - pred

    out = {}
    # agg A：按 X0 聚合
    fg, cA = agg_x0(x, Gv, n)
    fp, _ = agg_x0(x, pred, n)
    fr, _ = agg_x0(x, res, n)
    out["A"] = {"F_G": fg, "F_pred": fp, "F_res": fr, "覆盖股票数": int((cA > 0).sum())}
    # agg B/C：把逐对值放回对称矩阵再按行聚合
    def to_mat(idx_a, idx_b, val):
        M = np.zeros((n, n), dtype=np.float32)
        M[idx_a, idx_b] = val
        M[idx_b, idx_a] = val
        return M
    Mg = to_mat(ib, ic, Gv.astype(np.float32))
    Mp = to_mat(ib, ic, pred.astype(np.float32))
    Mr = to_mat(ib, ic, res.astype(np.float32))
    mxg, mng = agg_row(Mg, ok, n)
    mxp, mnp = agg_row(Mp, ok, n)
    mxr, mnr = agg_row(Mr, ok, n)
    out["B"] = {"F_G": mxg, "F_pred": mxp, "F_res": mxr}
    out["C"] = {"F_G": mng, "F_pred": mnp, "F_res": mnr}
    # 纯剖面对照
    Mx = np.array(A["rho"], dtype=np.float32, copy=True)
    np.fill_diagonal(Mx, -np.inf)
    Mx[:, ~ok] = -np.inf
    out["纯剖面"] = {"F_maxrho": Mx.max(axis=1)}
    return out


def cor(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 50:
        return float("nan"), int(m.sum())
    return float(spearmanr(a[m], b[m]).statistic), int(m.sum())


def show(name, d):
    print(f"  --- {name} ---")
    for agg in ("A", "B", "C"):
        if agg not in d:
            continue
        g = d[agg]["F_G"]
        for other in ("F_pred", "F_res"):
            c, k = cor(g, d[agg][other])
            print(f"    {agg} 聚合：截面 Spearman(F_G, {other[2:]}) = {c:.4f}   n={k}")
    for agg in ("A", "B", "C"):
        if agg not in d:
            continue
        c, k = cor(d[agg]["F_res"], d["纯剖面"]["F_maxrho"])
        print(f"    {agg} 聚合：截面 Spearman(F_res, 纯剖面 maxrho) = {c:.4f}   n={k}")


def main():
    root, date = sys.argv[1], int(sys.argv[2])
    A = load(root, date)
    print(f"=== {date}  股票 {int(A['is_stock'].sum())} 只 ===")
    for path in ("first", "second"):
        d = build(A, path)
        print()
        print(f"【{'第一路 X0（rho 最大）' if path=='first' else '第二路 X0*（补缺口最大）'}】")
        show(path, d)
    return 0


if __name__ == "__main__":
    sys.exit(main())
