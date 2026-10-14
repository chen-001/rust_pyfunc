#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""玉佩2 实证：Q1 求和算子的可加性 / Q2 取max近邻覆盖的互补增益 / 备份矩阵对齐

用真实逐笔数据回答三个可判定的问题（全部数字直接打印，结尾存 pkl）：

Q1  把 B 与 C 的成交按时间拼成 UBC，S(A,UBC) 是否严格等于 S(A,B)+S(A,C)？
Q2  把逐笔"求和"换成"近邻覆盖"（Gmax 取 max），拼接后的覆盖增益能带来什么？
    最优 C 的增益相对随机 C、按活动量匹配的随机 C 有多大？是"互补对"还是"挑最大的"？
ALIGN  自己的算子能否对齐备份矩阵 /hdd/.../yupei_dist_backup/20241231/mats/*.bin

口径（与 src/yupei_dist/matrix_stage.rs 逐条对应）：
  - 10ms 桶，BUCKET_US=10000；桶基准 = exchtime 相对当日 09:30 的微秒差（下午已前移 90 分钟）
  - S_dir[i][j] = Σ_k u_i[k]·( Σ_{l<k} u_j[l]·d^{k-l} + 0.5·u_j[k] ), d = exp(-0.01/τ)
    即 codes[j] 的成交在前、codes[i] 的成交在后；同桶按 50/50 折半
  - 每笔成交权重：cnt w=1；vol w=sqrt(volume)
  - 生产口径还砍掉最后一段：N_BLOCKS = 1422000//4096 = 347 块，桶 ≥ 1421312 不参与
  - MIN_TRADES=200 的过滤在脚本外（直接用备份的 codes.txt 名单，它已是过滤后的名单）

运行：
  /home/chenzongwei/.conda/envs/chenzongwei311/bin/python experiments/yupei2/empirics.py [--no-cache]
"""
import os
import sys
import time
import pickle
import datetime
import numpy as np
import scipy.sparse as sp
from scipy.stats import spearmanr
import rust_pyfunc as rp

# ---------------------------------------------------------------- 常量与口径
DATE = 20241231
BACKUP = "/hdd/user_home_unsafe/chenzongwei/yupei_dist_backup/20241231"
BUCKET_US = 10_000
BLOCK_BUCKETS = 4096
N_BUCKETS = 1_422_000
N_BLOCKS = N_BUCKETS // BLOCK_BUCKETS          # 347
B_TOTAL = N_BLOCKS * BLOCK_BUCKETS             # 1421312（末段 688 桶被生产实现丢弃）
CACHE_DIR = "/tmp"
OUT_PKL = "/tmp/yupei2_empirics_result.pkl"

ALIGN_MATS = ["cnt_t005", "cnt_t02", "cnt_t1", "vol_t02", "vol_t05", "vol_t1"]
ALIGN_N = 30                                   # 对齐用 codes.txt 前 30 只
SAMPLE_STEP = 10                               # 主样本：codes.txt 每 10 只取 1 只
Q2_NA = 40                                     # Q2 的 A 股数量
Q2_K = 5                                       # 每只 A 取几个伙伴 B
Q2_R = 200                                     # 随机对照抽样次数
Q1_TRIPLES = 40


def day_start_us(date):
    """与 compute.rs 的 day_start_us_of 一致：UTC 零点 + 9.5h（配合 +8h 偏移，9:30 开盘 = 0）"""
    y, m, d = date // 10000, date // 100 % 100, date % 100
    days = (datetime.date(y, m, d) - datetime.date(1970, 1, 1)).days
    return days * 86400 * 1_000_000 + (9 * 3600 + 30 * 60) * 1_000_000


DSU = day_start_us(DATE)


# ---------------------------------------------------------------- 读数据
def load_day(codes, date=DATE, use_cache=True):
    """读逐笔 → 每只股票 (桶, u_cnt, u_vol) 稀疏单元表 + 成交笔数

    与生产一致：read_trade_fast_inner(code, date, with_retreat=false, with_afternoon_adjust=true)
    Python 侧对应 rp.read_trade_fast_us(code, date, 0, True)，第 0 列是精确 time_us。
    """
    cache = os.path.join(CACHE_DIR, f"yupei2_cells_{date}_{len(codes)}_{codes[0]}.npz")
    if use_cache and os.path.exists(cache):
        z = np.load(cache)
        if list(z["codes"]) == list(codes):
            print(f"  [cache] {cache}")
            return dict(kabs=z["kabs"], stk=z["stk"], ucnt=z["ucnt"], uvol=z["uvol"],
                        ntrades=z["ntrades"],
                        blk=z["kabs"] // BLOCK_BUCKETS, kloc=z["kabs"] % BLOCK_BUCKETS)
    blk, stk, kabs, ucnt, uvol, ntrades = [], [], [], [], [], []
    for i, c in enumerate(codes):
        arr = rp.read_trade_fast_us(c, date, 0, True)
        ntrades.append(len(arr))
        t = arr[:, 0].astype(np.int64)
        off = t - DSU
        b = off // BUCKET_US
        keep = (off >= 0) & (b < B_TOTAL)          # 生产实现只处理前 347 块
        b = b[keep]
        v = np.maximum(arr[keep, 3], 0.0)
        kb = np.bincount(b)
        vb = np.bincount(b, weights=np.sqrt(v))
        nz = np.nonzero(kb)[0]
        kabs.append(nz)
        stk.append(np.full(len(nz), i))
        ucnt.append(kb[nz])
        uvol.append(vb[nz])
    D = dict(kabs=np.concatenate(kabs), stk=np.concatenate(stk),
             ucnt=np.concatenate(ucnt), uvol=np.concatenate(uvol),
             ntrades=np.array(ntrades, dtype=np.float64))
    D["blk"] = D["kabs"] // BLOCK_BUCKETS
    D["kloc"] = D["kabs"] % BLOCK_BUCKETS
    if use_cache:
        np.savez(cache, codes=np.array(codes), kabs=D["kabs"], stk=D["stk"],
                 ucnt=D["ucnt"], uvol=D["uvol"], ntrades=D["ntrades"])
    return D


# ---------------------------------------------------------------- 求和算子
def sum_operator(D, N, ufield="ucnt", tau=1.0, dtype=np.float64):
    """S_dir[i][j] = Σ_k u_i[k]·g_j[k]，g_j[k] = Σ_{l<k} u_j[l]·d^{k-l} + 0.5·u_j[k]

    分块 + 指数重标度：块内 CU[m] = Σ_{j<m} x_j·d^{-j}，则
      g = 0.5X + (acc + CU)·d^m ，acc 跨块按 acc ← (acc + Σ_j x_j d^{-j})·d^B 推进。
    d^{-B} 必须不溢出：B 上限按 dtype 的指数范围取（f64→1e150，f32→1e30）。
    """
    mu = 0.01 / tau
    d = float(np.exp(-mu))
    lim = 150.0 if dtype == np.float64 else 30.0
    Bint = int(min(BLOCK_BUCKETS, max(64, np.floor(lim * np.log(10) / mu))))
    order = np.argsort(D["kabs"], kind="stable")
    kabs = D["kabs"][order]
    stk = D["stk"][order]
    u = D[ufield][order].astype(dtype)
    bedges = np.append(np.arange(0, B_TOTAL, Bint), B_TOTAL)
    cedges = np.searchsorted(kabs, bedges)
    S = np.zeros((N, N))
    acc = np.zeros(N, dtype=dtype)
    dpos = (d ** np.arange(Bint)).astype(dtype)
    dinv = (1.0 / (d ** np.arange(Bint))).astype(dtype)
    dB = dtype(d ** Bint)
    for bi in range(len(bedges) - 1):
        a, b = cedges[bi], cedges[bi + 1]
        if a == b:
            acc = acc * dB
            continue
        loc = kabs[a:b] - bedges[bi]
        X = np.zeros((Bint, N), dtype=dtype)
        X[loc, stk[a:b]] = u[a:b]
        Xd = X * dinv[:, None]
        Cs = np.cumsum(Xd, axis=0, dtype=dtype)
        G = 0.5 * X + (acc + (Cs - Xd)) * dpos[:, None]
        rows = np.unique(loc)
        Xs = sp.csr_matrix((u[a:b], (np.searchsorted(rows, loc), stk[a:b])), shape=(len(rows), N))
        S += np.asarray(Xs.T @ G[rows], dtype=np.float64)
        acc = (acc + Cs[-1]) * dB
    return S


# ---------------------------------------------------------------- 取 max 的覆盖场
def prep_gmax(D, N, ufield, tau):
    """每只股票 (升序桶, 前缀最大 A_j = ln u + mu·k 的累积最大)

    于是 Gmax_b[k] = max_{l<=k} u_b[l]·d^{k-l} = exp(prefixmax(k) - mu·k)。
    对数域是为了避免 d^{-k} 在整日尺度上溢出。
    """
    mu = 0.01 / tau
    arrs = []
    for i in range(N):
        m = D["stk"] == i
        k = D["kabs"][m]
        u = D[ufield][m].astype(np.float64)
        with np.errstate(divide="ignore"):
            A = np.log(u) + mu * k
        arrs.append((k, np.maximum.accumulate(A)))
    return mu, arrs


def gmax_query(arr, ks, mu):
    k, P = arr
    idx = np.searchsorted(k, ks, side="right") - 1
    out = np.zeros(len(ks))
    v = idx >= 0
    if v.any():
        out[v] = np.exp(P[idx[v]] - mu * ks[v])
    return out


def gather_gmax(arrs, ks, mu, cols=None):
    idxs = list(range(len(arrs))) if cols is None else list(cols)
    out = np.empty((len(ks), len(idxs)))
    for j, c in enumerate(idxs):
        out[:, j] = gmax_query(arrs[c], ks, mu)
    return out


# ---------------------------------------------------------------- ALIGN：对齐备份矩阵
def run_alignment(codes, D30, D397):
    print("=" * 100)
    print(f"ALIGN  与备份矩阵对齐（{BACKUP}/mats/*.bin, 行主序 N×N f32, 列序=codes.txt）")
    print("-" * 100)
    out = []
    for nm in ALIGN_MATS:
        if nm.startswith("cnt"):
            fld, tau = "ucnt", {"cnt_t005": .05, "cnt_t02": .2, "cnt_t1": 1.0}[nm]
        else:
            fld, tau = "uvol", {"vol_t02": .2, "vol_t05": .5, "vol_t1": 1.0}[nm]
        S = sum_operator(D30, ALIGN_N, fld, tau)
        M = np.memmap(f"{BACKUP}/mats/{nm}.bin", dtype=np.float32, mode="r", shape=(3964, 3964))
        bk = np.array(M[:ALIGN_N, :ALIGN_N], dtype=np.float64)
        rel = np.abs(S - bk) / np.maximum(bk, 1e-30)
        out.append((nm, rel.max(), np.median(rel), np.abs(S - bk).max(), bk.max()))
        print(f"  {nm:10s} n=30×30  max相对误差={rel.max():.3e}  中位={np.median(rel):.3e}"
              f"  最大绝对误差={np.abs(S-bk).max():.3e}  峰值={bk.max():.3e}")
    # 抽样对齐：主样本 397 只（codes.txt 每 10 只取 1 只）
    idx = np.array(list(range(0, len(codes), SAMPLE_STEP)))[:D397["N"]]
    S397 = {k: D397["S"][k] for k in ("cnt_t005", "cnt_t1", "vol_t1")}
    for nm, S in S397.items():
        M = np.memmap(f"{BACKUP}/mats/{nm}.bin", dtype=np.float32, mode="r", shape=(3964, 3964))
        bk = np.array(M[np.ix_(idx, idx)], dtype=np.float64)
        rel = np.abs(S - bk) / np.maximum(bk, 1e-30)
        big = bk > 1e5
        print(f"  {nm:10s} n={len(idx)}² max相对误差={rel.max():.3e} 中位={np.median(rel):.3e}"
              f"  corr={np.corrcoef(S.ravel(), bk.ravel())[0,1]:.10f}"
              f"  （backup>1e5 的子集 max相对误差={rel[big].max():.3e}, n={big.sum()}）")
    return out


# ---------------------------------------------------------------- Q1：可加性
def run_q1(D, codes, N):
    print("=" * 100)
    print("Q1  求和算子的可加性：S(A, UBC) ?= S(A,B) + S(A,C)   （UBC = B 与 C 的成交按时间拼接）")
    print("-" * 100)
    rng = np.random.default_rng(20241231)
    triples = []
    while len(triples) < Q1_TRIPLES:
        a, b, c = rng.choice(N, 3, replace=False)
        triples.append((int(a), int(b), int(c)))

    def cells_of(i, fld):
        m = D["stk"] == i
        return D["kabs"][m], D[fld][m].astype(np.float64)

    kabs, stk, ucnt, uvol = [], [], [], []
    for (a, b, c) in triples:
        ka, ua = cells_of(a, "ucnt"); kb, ub = cells_of(b, "ucnt"); kc, uc = cells_of(c, "ucnt")
        va = cells_of(a, "uvol")[1]; vb = cells_of(b, "uvol")[1]; vc = cells_of(c, "uvol")[1]
        km = np.concatenate([kb, kc])
        ku, inv = np.unique(km, return_inverse=True)
        um = np.bincount(inv, weights=np.concatenate([ub, uc]))
        vm = np.bincount(inv, weights=np.concatenate([vb, vc]))
        for k, cu, cv in [(ka, ua, va), (kb, ub, vb), (kc, uc, vc), (ku, um, vm)]:
            kabs.append(k); stk.append(np.full(len(k), len(stk))); ucnt.append(cu); uvol.append(cv)
    DQ = dict(kabs=np.concatenate(kabs), stk=np.concatenate(stk),
              ucnt=np.concatenate(ucnt), uvol=np.concatenate(uvol),
              ntrades=np.zeros(4 * Q1_TRIPLES))
    res = {}
    for fld in ("ucnt", "uvol"):
        for tau in (1.0, 0.05):
            S = sum_operator(DQ, 4 * Q1_TRIPLES, fld, tau)
            e_in, e_out, e_self = [], [], []
            for i in range(Q1_TRIPLES):
                ia, ib, ic, im = 4 * i, 4 * i + 1, 4 * i + 2, 4 * i + 3
                e_in.append(abs(S[ia, im] - (S[ia, ib] + S[ia, ic])) / abs(S[ia, ib] + S[ia, ic]))
                e_out.append(abs(S[im, ia] - (S[ib, ia] + S[ic, ia])) / abs(S[ib, ia] + S[ic, ia]))
                r3 = S[ib, ib] + S[ic, ic] + S[ib, ic] + S[ic, ib]
                e_self.append(abs(S[im, im] - r3) / abs(r3))
            res[(fld, tau)] = (max(e_in), max(e_out), max(e_self))
            print(f"  {fld:5s} τ={tau:<4} S(A,UBC)-[S(A,B)+S(A,C)] 最大相对误差={max(e_in):.3e}"
                  f" | 反向 S(UBC,A)-[S(B,A)+S(C,A)]={max(e_out):.3e}"
                  f" | 自项 S(UBC,UBC)-[S(B,B)+S(C,C)+S(B,C)+S(C,B)]={max(e_self):.3e}")
    # 举例
    fld, tau = "ucnt", 1.0
    S = sum_operator(DQ, 4 * Q1_TRIPLES, fld, tau)
    print("  举例（cnt, τ=1s）:")
    for i in range(4):
        ia, ib, ic, im = 4 * i, 4 * i + 1, 4 * i + 2, 4 * i + 3
        a, b, c = triples[i]
        print(f"    A={codes[a]} B={codes[b]} C={codes[c]}:  S(A,B)={S[ia,ib]:.6g} S(A,C)={S[ia,ic]:.6g}"
              f" S(A,UBC)={S[ia,im]:.6g} 和={S[ia,ib]+S[ia,ic]:.6g}")
    return res, triples


# ---------------------------------------------------------------- Q2：互补增益
def gmax_share_of_last_trade(D, N, ufield, tau, n_query=20000, seed=3):
    """取max 场 Gmax_b[k] 有多少比例就等于"最近一次成交的强度按衰减折算"

    Gmax_b[k] = u_b[l*]·d^{k-l*}，l* 是让 u·d^{k-l} 最大的那个桶。若 l* 恰好是 k 之前最后一个
    有成交的桶，这个 max 场就退化成"谁最近在放量"，跟"覆盖/互补"无关。返回命中比例。
    """
    mu = 0.01 / tau
    rng = np.random.default_rng(seed)
    tot = hit = 0
    for i in range(N):
        m = D["stk"] == i
        k = D["kabs"][m]
        u = D[ufield][m].astype(np.float64)
        P = np.maximum.accumulate(np.log(u) + mu * k)
        ks = np.sort(rng.integers(0, B_TOTAL, size=n_query))
        idx = np.searchsorted(k, ks, side="right") - 1
        v = idx >= 0
        last_log = np.log(u[idx[v]]) + mu * k[idx[v]]
        hit += int((P[idx[v]] == last_log).sum())
        tot += int(v.sum())
    return hit / tot


def run_coverage(D, S, arrs, mu, ufield, N, codes):
    """覆盖数：A 自身活跃度 = Σ_k u_A[k]，把候选 c 按 cov(A,c) 降序累加，
    问达到 50%/80%/90% 需要几个伙伴。

    两种口径都给：
      n*   = 简单的 cov 累加（线性叠加）
      or*  = 真 OR 覆盖 Σ_k u_A[k]·max_{已选伙伴} Gmax_c[k]（顺序取伙伴，只加到 95% 为止）
    """
    import pandas as pd
    rows = []
    for A in range(N):
        m = D["stk"] == A
        ka = D["kabs"][m]
        w = D[ufield][m].astype(np.float64)
        target = float(w.sum())
        Gm = gather_gmax(arrs, ka, mu)
        cov = Gm.T @ w
        order = [int(c) for c in np.argsort(-cov) if c != A]
        cs = np.cumsum(cov[order])

        def need(arr, frac):
            i = np.searchsorted(arr, frac * target)
            return (i + 1) if i < len(arr) else np.nan

        # 真 OR：逐个把伙伴并进去
        running = Gm[:, order[0]].copy()
        orc = [float(w @ running)]
        for c in order[1:]:
            if orc[-1] >= 0.95 * target:
                break
            running = np.maximum(running, Gm[:, c])
            orc.append(float(w @ running))
        orc = np.array(orc)
        rows.append(dict(A=A, code=codes[A], ntrades=float(D["ntrades"][A]), self_act=target,
                         n50=need(cs, .5), n80=need(cs, .8), n90=need(cs, .9),
                         or50=need(orc, .5), or80=need(orc, .8), or90=need(orc, .9),
                         n_used_or=len(orc), top_cov_share=float(cov[order[0]] / target)))
    return pd.DataFrame(rows)


def report_coverage(d, tag):
    import pandas as pd
    q = d[["n50", "n80", "n90", "or50", "or80", "or90"]].quantile([0.1, 0.5, 0.9]).round(1)
    print(f"  [{tag}] 覆盖数分布（10/50/90 分位）")
    print("    " + q.to_string().replace("\n", "\n    "))
    print(f"    未达到 90% 的 A 数: 累加口径 {int(d.n90.isna().sum())} 只，OR 口径 {int(d.or90.isna().sum())} 只"
          f"（OR 口径最多只加到 95%，列数上限 {int(d.n_used_or.max())}）")
    for k in ["n50", "n80", "n90", "or50", "or80", "or90"]:
        if d[k].nunique(dropna=True) <= 1:
            print(f"    覆盖数 {k} = {d[k].dropna().iloc[0]:.0f}（所有 A 都一样，没有分散 → 相关系数无定义）")
        else:
            rho = spearmanr(d[k], d.ntrades).statistic
            print(f"    覆盖数 {k} 与当天成交笔数的 spearman rho = {rho:+.3f}")
    print(f"    自身活跃度（Σ u_A）与笔数的 spearman = {spearmanr(d.self_act, d.ntrades).statistic:+.3f}"
          f"；最大伙伴覆盖占自身活跃度的比例 中位 {d.top_cov_share.median():.2f}"
          f"（10/90 分位 {d.top_cov_share.quantile(.1):.2f}/{d.top_cov_share.quantile(.9):.2f}）")
    print(f"    这个比例与笔数的 spearman = {spearmanr(d.top_cov_share, d.ntrades).statistic:+.3f}")
    print(f"    覆盖数的分散度（90 分位 / 10 分位）: n90={d.n90.quantile(.9)/max(d.n90.quantile(.1),1e-9):.1f}"
          f", or90={d.or90.quantile(.9)/max(d.or90.quantile(.1),1e-9):.1f}")


def coverage_matrix(D, N, ufield, tau, dtype=np.float64):
    """加截断的覆盖矩阵 M[a,c] = Σ_k u_a[k]·min(1, Gmax_c[k])

    Gmax_c[k] = max_{l<=k} u_c[l]·d^{k-l}，分块 + 重标度：
      块内 cm[m] = max_{j<=m} x_j·d^{-j}，Gmax = d^m·max(carry, cm[m])，carry ← d^B·max(carry, cm[-1])
    与 sum_operator 同一套分块，所以量级、口径完全一致。
    """
    mu = 0.01 / tau
    d = float(np.exp(-mu))
    lim = 150.0 if dtype == np.float64 else 30.0
    Bint = int(min(BLOCK_BUCKETS, max(64, np.floor(lim * np.log(10) / mu))))
    order = np.argsort(D["kabs"], kind="stable")
    kabs = D["kabs"][order]
    stk = D["stk"][order]
    u = D[ufield][order].astype(dtype)
    bedges = np.append(np.arange(0, B_TOTAL, Bint), B_TOTAL)
    cedges = np.searchsorted(kabs, bedges)
    M = np.zeros((N, N))
    carry = np.zeros(N, dtype=dtype)
    dpos = (d ** np.arange(Bint)).astype(dtype)
    dinv = (1.0 / (d ** np.arange(Bint))).astype(dtype)
    dB = dtype(d ** Bint)
    for bi in range(len(bedges) - 1):
        a, b = cedges[bi], cedges[bi + 1]
        if a == b:
            carry = carry * dB
            continue
        loc = kabs[a:b] - bedges[bi]
        X = np.zeros((Bint, N), dtype=dtype)
        X[loc, stk[a:b]] = u[a:b]
        cm = np.maximum.accumulate(X * dinv[:, None], axis=0)
        Gm = np.maximum(cm, carry[None, :]) * dpos[:, None]
        carry = np.maximum(carry, cm[-1]) * dB
        H = np.minimum(1.0, Gm)
        rows = np.unique(loc)
        Xs = sp.csr_matrix((u[a:b], (np.searchsorted(rows, loc), stk[a:b])), shape=(len(rows), N))
        M += np.asarray(Xs.T @ H[rows], dtype=np.float64)
    return M


def run_saturation(D, M, arrs, mu, ufield, N, codes, A_gain, R=200, seed=17, topM=50):
    """加截断后的覆盖：v(A,c)=M[a,c]/Σ_k u_A[k]；覆盖数 = 按 v 降序贪心并伙伴到 V=50/80/90%

    A_gain 里的 A 做完整的"最优 C"搜索（全候选池），其余 A 只做 v 与覆盖数
    （v_best ≥ 0.9 时一步就到 n50=n80=n90=1，不用再算）。
    """
    import pandas as pd
    rng = np.random.default_rng(seed)
    nt = D["ntrades"]
    target = np.bincount(D["stk"], weights=D[ufield], minlength=N)
    v_all = M / target[:, None]
    rows_v, rows_g = [], []
    for A in range(N):
        va = v_all[A].copy()
        va[A] = -np.inf
        B = int(np.argmax(va))
        vbest = float(va[B])
        pool_v = [c for c in range(N) if c != A]
        rec = dict(A=A, code=codes[A], ntrades=float(nt[A]), B=B, B_code=codes[B],
                   v_best=vbest, v_med_all=float(np.median(va[pool_v])),
                   n50=np.nan, n80=np.nan, n90=np.nan, n_greedy_used=0)
        if vbest >= 0.9:
            rec.update(n50=1, n80=1, n90=1, n_greedy_used=1)     # 一个伙伴就过 90%
        else:
            m = D["stk"] == A
            ka = D["kabs"][m]
            w = D[ufield][m].astype(np.float64)
            ranked = [int(c) for c in np.argsort(-va) if c != A][:topM]
            H = gather_gmax(arrs, ka, mu, cols=ranked)
            hit = np.minimum(1.0, H)
            running = np.zeros(len(ka))
            need = {}
            for j in range(len(ranked)):
                running = np.maximum(running, hit[:, j])
                V = float(w @ running) / float(w.sum())
                for frac in (0.5, 0.8, 0.9):
                    if frac not in need and V >= frac:
                        need[frac] = j + 1
                if len(need) == 3:
                    break
            rec.update(n50=need.get(0.5, np.nan), n80=need.get(0.8, np.nan),
                       n90=need.get(0.9, np.nan), n_greedy_used=j + 1)
        rows_v.append(rec)
        if A not in A_gain:
            continue
        # ---- 最优 C：B = v 最大的伙伴，C 搜全池 ----
        m = D["stk"] == A
        ka = D["kabs"][m]
        w = D[ufield][m].astype(np.float64)
        hitT = np.ascontiguousarray(np.minimum(1.0, gather_gmax(arrs, ka, mu)).T)   # (N, |K_A|)
        v = (hitT @ w) / float(w.sum())
        v_err = float(np.max(np.abs(v - v_all[A])))          # 两种实现的自检
        B = int(np.argmax(np.where(np.arange(N) == A, -np.inf, v)))
        hB = hitT[B]
        Vbc = (np.maximum(hB[None, :], hitT) @ w) / float(w.sum())
        pos = Vbc - v[B]
        gain = np.minimum(pos, pos - (v - v[B]))              # = V - max(v(A,B), v(A,C))
        gain[[A, B]] = -np.inf
        pool = np.array([c for c in range(N) if c not in (A, B)])
        Cstar = int(np.argmax(gain))
        vmax_in_pool = int(pool[np.argmax(v[pool])])
        rc = rng.choice(pool, size=R, replace=False)
        am = _nearest(nt, nt[Cstar], pool, R, {Cstar})
        rows_g.append(dict(A=A, code=codes[A], ntrades=float(nt[A]), B=B, Cstar=Cstar,
                           B_code=codes[B], C_code=codes[Cstar],
                           v_B=float(v[B]), v_Cstar=float(v[Cstar]), gain_best=float(gain[Cstar]),
                           rand_med=float(np.median(gain[rc])),
                           rand_p90=float(np.percentile(gain[rc], 90)),
                           act_med=float(np.median(gain[am])),
                           act_p90=float(np.percentile(gain[am], 90)),
                           same_as_vmax=int(Cstar == vmax_in_pool), v_check=v_err))
    return pd.DataFrame(rows_v), pd.DataFrame(rows_g)


def report_saturation(dv, dg, tag):
    print(f"  [{tag}] 1) v(A, c_best) 的横截面分布: 中位 {dv.v_best.median():.4f}"
          f"（10/90 分位 {dv.v_best.quantile(.1):.4f} / {dv.v_best.quantile(.9):.4f}）"
          f"；与当天成交笔数 spearman = {spearmanr(dv.v_best, dv.ntrades).statistic:+.3f}")
    print(f"          候选池里 v 的中位数 = {dv.v_med_all.median():.4f}"
          f"（10/90 分位 {dv.v_med_all.quantile(.1):.4f} / {dv.v_med_all.quantile(.9):.4f}）")
    for k, frac in [("n50", 50), ("n80", 80), ("n90", 90)]:
        if dv[k].nunique(dropna=True) <= 1:
            print(f"      2) 覆盖数 n{frac} = {dv[k].dropna().iloc[0]:.0f}（所有 A 都一样，无分散）")
        else:
            print(f"      2) 覆盖数 n{frac}: 10/50/90 分位 = {dv[k].quantile(.1):.0f} /"
                  f" {dv[k].median():.0f} / {dv[k].quantile(.9):.0f}"
                  f"；与成交笔数 spearman = {spearmanr(dv[k].fillna(99), dv.ntrades).statistic:+.3f}")
    print(f"      3) 最优 C 增益: 中位 {dg.gain_best.median():.4f}（10/90 分位"
          f" {dg.gain_best.quantile(.1):.4f} / {dg.gain_best.quantile(.9):.4f}）"
          f" | 随机 C p90 中位 {dg.rand_p90.median():.4f} | 笔数匹配 C p90 中位 {dg.act_p90.median():.4f}"
          f" | 最优/随机p90 中位 {dg.gain_best.div(dg.rand_p90.clip(lower=1e-9)).median():.2f}"
          f" | 最优/匹配p90 中位 {dg.gain_best.div(dg.act_p90.clip(lower=1e-9)).median():.2f}")
    print(f"         最优 C 就是 v 最大者的比例 = {dg.same_as_vmax.mean():.3f}"
          f"（v_B 中位 {dg.v_B.median():.4f}，最优 C 的 v 中位 {dg.v_Cstar.median():.4f}）")
    print(f"      4) 增益与 A 当天成交笔数的 spearman = {spearmanr(dg.gain_best, dg.ntrades).statistic:+.3f}")


def _nearest(vals, target, pool_idx, n, exclude):
    v = vals[pool_idx]
    order = np.argsort(np.abs(v - target))
    sel = [pool_idx[j] for j in order if pool_idx[j] not in exclude]
    return np.array(sel[:n])


def run_q2_one(D, S, arrs, mu, ufield, N, A_set, codes, K=Q2_K, R=Q2_R, seed=7,
               partner="topS", tag=""):
    """对每只 A：按 S 取最强的 K 个伙伴 B；对每个 B 搜使 cov(A,UBC)-max(cov(A,B),cov(A,C)) 最大的 C

    cov(A,B) = Σ_k u_A[k]·Gmax_b[k]，Gmax_b[k] = max_{l<=k} u_b[l]·d^{k-l}（同桶计入 l=k）
    """
    rng = np.random.default_rng(seed)
    nt = D["ntrades"]
    rows = []
    for A in A_set:
        m = D["stk"] == A
        ka = D["kabs"][m]
        w = D[ufield][m].astype(np.float64)
        Gm = gather_gmax(arrs, ka, mu)                 # (|K_A|, N)  Gmax_c[k]
        cov = Gm.T @ w                                 # cov(A,c)
        srow = S[A].copy(); srow[A] = -np.inf
        if partner == "topS":
            Bs = [int(x) for x in np.argsort(-srow)[:K]]
        else:
            rb = np.random.default_rng(seed + 1000 + int(A))
            Bs = [int(x) for x in rb.choice([c for c in range(N) if c != A], size=K, replace=False)]
        for B in Bs:
            gB = Gm[:, B]
            gain = w @ np.maximum(0.0, Gm - gB[:, None])          # = cov(A,UBC) - cov(A,B)
            negsum = cov[B] + gain - cov                          # = Σ w·max(0, Gb-Gc)
            gain1 = np.minimum(gain, negsum)                      # = cov(A,UBC) - max(cov(A,B),cov(A,C))
            gain[[A, B]] = -np.inf; gain1[[A, B]] = -np.inf
            pool = np.array([c for c in range(N) if c not in (A, B)])
            Cstar = int(np.argmax(gain)); gs = float(gain[Cstar])
            r = dict(A=A, B=B, Cstar=Cstar, A_code=codes[A], B_code=codes[B], C_code=codes[Cstar],
                     covB=float(cov[B]), covCstar=float(cov[Cstar]), gain_best=gs,
                     gain1_best=float(gain1[Cstar]), nt_Cstar=float(nt[Cstar]))
            rc = rng.choice(pool, size=R, replace=False)
            r["rand_med"] = float(np.median(gain[rc]))
            r["rand_p90"] = float(np.percentile(gain[rc], 90))
            r["rand_max"] = float(gain[rc].max())
            r["rand_ge_best"] = int((gain[rc] >= gs).sum())
            am = _nearest(nt, nt[Cstar], pool, R, {Cstar})
            r["act_med"] = float(np.median(gain[am]))
            r["act_p90"] = float(np.percentile(gain[am], 90))
            order = pool[np.argsort(-cov[pool])]
            r["Cmaxcov"] = int(order[0])
            r["same_as_maxcov"] = int(Cstar == int(order[0]))
            # 特异性：有多少候选能做到与最优增益相差很小（>0 就说明"最优"不唯一）
            gp = gain[pool]
            r["n_ge_99pct"] = int((gp >= 0.99 * gs).sum())
            r["n_ge_90pct"] = int((gp >= 0.90 * gs).sum())
            r["n_pool"] = len(pool)
            alt = [c for c in order[:11] if c != Cstar][:9]        # cov 排名 2..10
            r["n_alt"] = len(alt)
            if alt:
                tg = gain[alt]
                r["covrank2_10_med"] = float(np.median(tg))
                r["covrank2_10_max"] = float(tg.max())
                r["covrank_ratio_med"] = r["covrank2_10_med"] / gs
                r["covrank_ratio_max"] = float(tg.max() / gs)
            r["spearman"] = float(spearmanr(gain[pool], cov[pool]).statistic)
            r["rel_best_over_covB"] = gs / cov[B]
            r["rel_best_over_covC"] = gs / cov[Cstar]
            rows.append(r)
    import pandas as pd
    d = pd.DataFrame(rows)
    d.attrs["tag"] = tag
    return d


def summarize(d):
    return dict(
        n=len(d),
        最优C等于最大covC的比例=d.same_as_maxcov.mean(),
        gain与cov的spearman中位=d.spearman.median(),
        最优增益中位=d.gain_best.median(),
        随机C增益中位=d.rand_med.median(),
        随机C增益p90=d.rand_p90.median(),
        活动量匹配C增益p90=d.act_p90.median(),
        最优比随机p90=d.gain_best.div(d.rand_p90).median(),
        最优比活动量匹配p90=d.gain_best.div(d.act_p90).median(),
        次优cov候选增益比中位=d.covrank_ratio_med.median(),
        次优cov候选增益比最大=d.covrank_ratio_max.median(),
        与最优增益差1pct内的候选数中位=d.n_ge_99pct.median(),
        与最优增益差10pct内的候选数中位=d.n_ge_90pct.median(),
        候选池大小=d.n_pool.median(),
        最优增益比covB中位=d.rel_best_over_covB.median(),
        最优增益比covC中位=d.rel_best_over_covC.median(),
        任务口径gain1比posgain中位=d.gain1_best.div(d.gain_best).median(),
    )


def main():
    use_cache = "--no-cache" not in sys.argv
    t_all = time.time()
    codes = [l.strip() for l in open(os.path.join(BACKUP, "codes.txt")) if l.strip()]
    print(f"备份名单 codes.txt: {len(codes)} 只；日期 {DATE}；桶制 {BUCKET_US}us；"
          f"生产只处理前 {N_BLOCKS} 块（桶 < {B_TOTAL}）")

    # ---- 载入主样本 397 只 ----
    sel = list(range(0, len(codes), SAMPLE_STEP))
    codesN = [codes[i] for i in sel]
    print(f"主样本（每 {SAMPLE_STEP} 只取 1）: {len(codesN)} 只")
    t0 = time.time(); DN = load_day(codesN, use_cache=use_cache)
    print(f"  读逐笔 {time.time()-t0:.1f}s，单元数 {len(DN['kabs'])}，"
          f"成交笔数 中位 {np.median(DN['ntrades']):.0f} 最小 {DN['ntrades'].min():.0f}")
    N = len(codesN)

    # ---- 4 个 (权重族, τ) 的求和矩阵 + 取max 场 ----
    CFG = [("cnt_t005", "ucnt", 0.05), ("cnt_t1", "ucnt", 1.0),
           ("vol_t005", "uvol", 0.05), ("vol_t1", "uvol", 1.0)]
    COV_ONLY = "--coverage-only" in sys.argv
    SAT_ONLY = "--sat-only" in sys.argv
    if COV_ONLY:
        CFG = [c for c in CFG if c[0] in ("cnt_t1", "vol_t1")]
    elif SAT_ONLY:
        CFG = [c for c in CFG if c[0] in ("cnt_t1", "cnt_t005")]
    S = {}; G = {}
    for nm, fld, tau in CFG:
        t0 = time.time(); S[nm] = sum_operator(DN, N, fld, tau)
        mu, arrs = prep_gmax(DN, N, fld, tau)
        G[nm] = (mu, arrs)
        print(f"  {nm:9s} 求和矩阵 + 取max场  {time.time()-t0:.1f}s")

    # ---- COVERAGE：达到自身活跃度 50/80/90% 需要几个伙伴 ----
    print("=" * 100)
    print("COV  覆盖数：把候选按 cov(A,c) 降序累加，达到 A 自身活跃度 Σ_k u_A[k] 的 50/80/90% 需要几个伙伴")
    print("-" * 100)
    COV = {}
    for nm in (() if SAT_ONLY else ("cnt_t1", "vol_t1")):
        spec = [c for c in CFG if c[0] == nm][0]
        fld, tau = spec[1], spec[2]
        mu, arrs = G[nm]
        t0 = time.time()
        COV[nm] = run_coverage(DN, S[nm], arrs, mu, fld, N, codesN)
        report_coverage(COV[nm], nm)
        print(f"  （{nm} 用时 {time.time()-t0:.1f}s）")
    if COV_ONLY:
        pickle.dump(dict(COV=COV, S={k: np.asarray(v) for k, v in S.items()},
                         codesN=codesN, agg=None, Q2=None, Q1=None, triples=None, align=None),
                    open(OUT_PKL, "wb"))
        print(f"\n结果已存 {OUT_PKL}；总用时 {time.time()-t_all:.1f}s（--coverage-only）")
        return

    # ---- SATURATION：每笔 A 成交最多被接住一次 ----
    print("=" * 100)
    print("SAT  加截断的覆盖：hit_c[k]=min(1,Gmax_c[k])；v(A,c)=Σ_k u_A[k]·hit_c[k]/Σ_k u_A[k]")
    print("-" * 100)
    SATV, SATG = {}, {}
    A40g = [int(x) for x in np.random.default_rng(20241231).choice(N, Q2_NA, replace=False)]
    for nm in ("cnt_t1", "cnt_t005"):
        spec = [c for c in CFG if c[0] == nm][0]
        fld, tau = spec[1], spec[2]
        mu, arrs = G[nm]
        t0 = time.time()
        Mmat = coverage_matrix(DN, N, fld, tau)
        SATV[nm], SATG[nm] = run_saturation(DN, Mmat, arrs, mu, fld, N, codesN, set(A40g))
        report_saturation(SATV[nm], SATG[nm], nm)
        print(f"  （{nm} 用时 {time.time()-t0:.1f}s）")
    if SAT_ONLY:
        pickle.dump(dict(SATV=SATV, SATG=SATG, S={k: np.asarray(v) for k, v in S.items()},
                         codesN=codesN, agg=None, Q2=None, Q1=None, triples=None, align=None, COV=None),
                    open(OUT_PKL, "wb"))
        print(f"\n结果已存 {OUT_PKL}；总用时 {time.time()-t_all:.1f}s（--sat-only）")
        return

    # ---- ALIGN ----
    codes30 = codes[:ALIGN_N]
    D30 = load_day(codes30, use_cache=use_cache)
    align = run_alignment(codes, D30, dict(N=N, S=S))

    # ---- Q1 ----
    q1, triples = run_q1(DN, codesN, N)

    # ---- Q2 ----
    print("=" * 100)
    print(f"Q2  取max近邻覆盖的互补增益（A 取 {Q2_NA} 只，每只 {Q2_K} 个伙伴 B，对照 {Q2_R} 次抽样）")
    print("-" * 100)
    A40 = [int(x) for x in np.random.default_rng(20241231).choice(N, Q2_NA, replace=False)]
    print("  Gmax 场退化程度（等于『最近一次成交的强度按衰减折算』的比例，随机 2 万桶/只）:")
    for nm, fld, tau in CFG:
        print(f"    {nm:9s} {gmax_share_of_last_trade(DN, N, fld, tau):.3f}")
    Q2 = {}
    for nm, fld, tau in CFG:
        mu, arrs = G[nm]
        t0 = time.time()
        Q2[nm] = run_q2_one(DN, S[nm], arrs, mu, fld, N, A40, codesN, tag=nm)
        print(f"  {nm:9s} {time.time()-t0:.1f}s")
    mu, arrs = G["cnt_t1"]
    Q2["cnt_t1_randB"] = run_q2_one(DN, S["cnt_t1"], arrs, mu, "ucnt", N, A40, codesN,
                                    partner="rand", tag="cnt_t1_randB")

    import pandas as pd
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 60)

    print("-" * 100)
    print("Q2 汇总（每格是 40 只 A × 5 个 B = 200 行的中位数）")
    rows = []
    for k, d in Q2.items():
        s = summarize(d); s = {"口径": k, **s}
        rows.append(s)
    agg = pd.DataFrame(rows)
    print(agg.round(4).to_string(index=False))

    d = Q2["cnt_t1"]
    cols = ["A_code", "B_code", "C_code", "covB", "covCstar", "gain_best", "gain1_best",
            "rand_med", "rand_p90", "act_p90", "rel_best_over_covB", "rel_best_over_covC"]
    print("\ncnt_t1 具体三元组（前 8 行）:")
    print(d[cols].head(8).round(4).to_string(index=False))
    print("\ncnt_t1 增益最大的 5 行:")
    print(d.sort_values("gain_best", ascending=False)[cols].head(5).round(4).to_string(index=False))
    print("\ncnt_t1_randB（B 改成随机伙伴）前 5 行: 注意最优 C 仍是覆盖最大的那只")
    print(Q2["cnt_t1_randB"][cols].head(5).round(4).to_string(index=False))
    print("\nQ2 特异性：每个 (A,B) 的候选池里，增益能达到最优增益 99% / 90% 的候选有多少只")
    for k, dd in Q2.items():
        print(f"  {k:12s} 池均={dd.n_pool.mean():.0f} 只 | 99%以内 中位={dd.n_ge_99pct.median():.0f} 只"
              f"（有并列的行占比 {(dd.n_ge_99pct > 0).mean():.2f}）"
              f" | 90%以内 中位={dd.n_ge_90pct.median():.0f} 只")

    pickle.dump(dict(S=S, G=None, Q1=q1, triples=triples, Q2=Q2, align=align, agg=agg,
                     COV=COV, SATV=SATV, SATG=SATG, codesN=codesN, A40=A40,
                     S_big={k: np.asarray(v) for k, v in S.items()}),
                open(OUT_PKL, "wb"))
    print(f"\n结果已存 {OUT_PKL}；总用时 {time.time()-t_all:.1f}s")


if __name__ == "__main__":
    main()
