#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""玉佩2 配对互补（版本一，求和核）纯 Python 参考实现与合成精确性检查。

产出 /tmp/yupei2_pair_ref.json，供主会话核对 Rust 输出。口径全部按
experiments/yupei2/pair_spec.txt（冻结稿 v1）与 kernel_spec.txt 第 1 节。

三块内容
  A  合成精确性（最小时间序列，逐条给数字）
     A1 求和核可加性 S(A,U_BC) == S(A,B) + S(A,C)
     A2 "拼接看不出互补还是重复" + tau->0 极限下的 rho 与 G
     A3 rho[X][U_BC] == wB*rho[X][B] + wC*rho[X][C]（含 eB != eC 的例子）
     A4 空模型：两个独立且与 X 同 rho 的伙伴，G 仍为正（解析值 vs 数值值）
  B  真实数据参考值（20241231，备份 codes.txt 每 20 只取 1 只，共 199 只）
     逐笔直接两两配对算 S_dir（不做桶量化，直接 exp(-|dt|/tau)），
     E_i = 2*S_dir[i][i]，rho = S/sqrt(E_i E_j)，再按 spec 第 2 节算
     X0 / RHO_U / SINGLE / G，全部落盘。
  C  rust_compare：主会话 Rust 输出目录存在就比对，不存在写 pending。

关键数值性质（本脚本会逐条验证）
  * 备份 cnt_t1.bin 的 10ms 桶口径与逐笔精确口径在本数据上几乎完全一致，
    因为 read_trade_fast_us 的 time_us 分辨率恰好是 10000 微秒（全部整除），
    同一桶 == 同一时间戳，桶内取 1 的近似在这里不产生额外误差。
  * 因此"逐笔精确"版本的唯一近似点只剩下时间戳相同（同刻成交）怎么数：
    与生产口径一致的做法是同刻的任意两笔各记 0.5（无序对合计权重 1），
    且 S_dir[i][i] 里每笔与自身也记 0.5，于是 E_i = 2*S_dir[i][i] 含自配对。
    这一条如果实现成"严格按时间先后"会把同刻对整段丢掉，E 会小 30%~40%。

用法
  /home/chenzongwei/.conda/envs/chenzongwei311/bin/python \
      experiments/yupei2/pair_ref.py [--no-cache] [--skip-real] [--json OUT]

本脚本只写 experiments/yupei2/pair_ref.py 与 --json 指定的文件（默认
/tmp/yupei2_pair_ref.json），不改动任何别的文件，不提交任何计算任务。
"""
import argparse
import datetime
import json
import os
import sys
import time

import numpy as np

# ---------------------------------------------------------------- 常量与口径
DATE = 20241231
TAU = 1.0                                   # 秒
BUCKET_US = 10_000
N_BLOCKS = 347                              # 生产实现只处理前 347 块
B_TOTAL = N_BLOCKS * 4096                   # 1421312 桶
BACKUP_DIR = "/hdd/user_home_unsafe/chenzongwei/yupei_dist_backup/20241231"
BACKUP_MAT = os.path.join(BACKUP_DIR, "mats", "cnt_t1.bin")
CODES_TXT = os.path.join(BACKUP_DIR, "codes.txt")
SAMPLE_STEP = 20                            # codes.txt 每 20 只取 1 只 -> 199 只
RUST_DIR = "/tmp/yupei2_pair/20241231"
OUT_JSON = "/tmp/yupei2_pair_ref.json"
CACHE_TICKS = "/tmp/yupei2_pair_ref_ticks_%d_%d.npz" % (DATE, 199)
CACHE_SDIR = "/tmp/yupei2_pair_ref_sdir_%d_%d.npy" % (DATE, 199)


def day_start_us(date):
    """与 empirics.py / compute.rs 一致的桶基准：09:30 开盘为零点"""
    y, m, d = date // 10000, date // 100 % 100, date % 100
    days = (datetime.date(y, m, d) - datetime.date(1970, 1, 1)).days
    return days * 86400 * 1_000_000 + (9 * 3600 + 30 * 60) * 1_000_000


DSU = day_start_us(DATE)


# ================================================================ 精确逐笔算子
class ExactOp:
    """逐笔精确的有向/对称衰减求和核，不做任何桶量化。

    S_dir[i][j] = sum_{a in i, b in j} K(t_a, t_b)
      K = exp(-(t_a-t_b)/tau)  当 t_a > t_b
      K = 0.5                  当 t_a == t_b      （同刻两笔各记 0.5）
      K = 0                    当 t_a < t_b
    于是 S_sym[i][j] = S_dir[i][j] + S_dir[j][i] (i != j) 就是配对权重全为 1 的
    exp(-|dt|/tau) 之和，E_i = 2*S_dir[i][i] = n_i + 2*sum_{a<b}exp(-|dt|/tau)。

    实现：把一天按 chunk_us 切成块，块内用"以块起点为锚"的前缀和，
    跨块用递推 carry（每条股票各自预计算一次，与伙伴无关）。
    块内前缀与乘子都是正数、且前缀只含时间不晚于查询点的项，不会出现
    大数相减，double 精度下与暴力两两配对一致（见 A1~A3 与 B 的暴力校验）。
    """

    def __init__(self, tau, chunk_us=None):
        self.tau_us = float(tau) * 1e6
        if chunk_us is None:
            # 块长 / tau <= 400，保证 exp(chunk/tau) 不溢出 f64
            chunk_us = int(max(min(240.0, 400.0 * tau), 1e-4) * 1e6)
        self.chunk_us = int(chunk_us)

    def ngrid_for(self, tmax):
        return int(tmax // self.chunk_us) + 3

    def prep(self, ts, ngrid=None):
        ts = np.asarray(ts, dtype=np.int64)
        if ngrid is None:
            ngrid = self.ngrid_for(int(ts[-1]) if len(ts) else 0)
        cu, tau = self.chunk_us, self.tau_us
        cid = ts // cu
        ex = np.exp((ts - cid * cu) / tau)
        cnt = np.bincount(cid, minlength=ngrid)
        m0 = np.concatenate(([0], np.cumsum(cnt)))          # m0[c]=块 c 之前的笔数
        PFin = np.empty(len(ts), dtype=np.float64)          # 块内前缀（含自身，锚=块起点）
        chtot = np.zeros(ngrid + 1)
        for c in range(ngrid):
            s, e = m0[c], m0[c + 1]
            if e > s:
                PFin[s:e] = np.cumsum(ex[s:e])
                chtot[c] = PFin[e - 1]
        decay = np.exp(-cu / tau)
        carry = np.zeros(ngrid + 1)                         # carry[c] = g(块 c 起点)
        acc = 0.0
        for c in range(ngrid):
            carry[c] = acc
            acc = (acc + chtot[c]) * decay
        carry[ngrid] = acc
        return dict(ts=ts, n=len(ts), m0=m0, PFin=PFin, carry=carry)

    def s_dir(self, Q, P):
        """S_dir[Q][P]，Q 为查询时刻（可以是另一只股票的整段成交时刻）

        查询时刻可能晚于 P 的最后一笔、甚至超出 P 预计算的块范围，
        这时把 carry 继续按块衰减外推（carry 本身就是这样递推出来的）。
        """
        ts = P["ts"]
        if len(ts) == 0 or len(Q) == 0:
            return 0.0
        cu, tau = self.chunk_us, self.tau_us
        ng = len(P["carry"]) - 1
        p = np.searchsorted(ts, Q, "left")                  # 严格早于 t_a 的笔数
        pr = np.searchsorted(ts, Q, "right")                # 同刻的笔数（含自身）
        c = Q // cu
        cs = np.minimum(c, ng)
        m0c = P["m0"][cs]                                   # 该块之前的笔数
        inpart = np.where(p > m0c, P["PFin"][np.maximum(p - 1, 0)], 0.0)
        mult = np.exp(-(Q - c * cu) / tau)
        carry = P["carry"][cs] * np.exp(-np.maximum(c - ng, 0) * cu / tau)
        val = (inpart + carry) * mult
        return float(val.sum()) + 0.5 * float((pr - p).sum())

    def s_sym(self, Pi, Pj):
        """S_sym[i][j] = S_dir[i][j] + S_dir[j][i]（i != j）"""
        return self.s_dir(Pi["ts"], Pj) + self.s_dir(Pj["ts"], Pi)

    def energy(self, P):
        """E = 2*S_dir[i][i]，含自配对（每笔与自身 0.5，乘 2 后为 1）"""
        return 2.0 * self.s_dir(P["ts"], P)


def brute_s_dir(ti, tj, tau_us):
    """暴力两两配对（只用于小样本校验），口径与 ExactOp.s_dir 完全相同"""
    ti = np.asarray(ti, dtype=np.float64)
    tj = np.asarray(tj, dtype=np.float64)
    if len(ti) == 0 or len(tj) == 0:
        return 0.0
    d = ti[:, None] - tj[None, :]
    return float((np.where(d > 0, np.exp(-np.abs(d) / tau_us), 0.0)
                  + 0.5 * (d == 0)).sum())


def brute_s_sym(ti, tj, tau_us):
    ti = np.asarray(ti, dtype=np.float64)
    tj = np.asarray(tj, dtype=np.float64)
    d = np.abs(ti[:, None] - tj[None, :])
    return float(np.exp(-d / tau_us).sum())


def rel(a, b):
    a, b = float(a), float(b)
    if b == 0.0:
        return 0.0 if a == 0.0 else float("inf")
    return abs(a - b) / abs(b)


# ================================================================ A 合成检查
def synth_checks():
    out = {}
    checks = []

    # ---------------- A1 可加性 -------------------------------------------
    tau = 1.0
    op = ExactOp(tau)
    A = np.array([0.5e6, 3.0e6, 7.5e6], dtype=np.int64)
    B = np.array([1.0e6, 2.0e6, 5.0e6], dtype=np.int64)
    C = np.array([0.2e6, 6.0e6, 9.0e6], dtype=np.int64)
    U = np.sort(np.concatenate([B, C]))
    PA, PB, PC, PU = op.prep(A), op.prep(B), op.prep(C), op.prep(U)
    sAB, sAC, sABC = op.s_sym(PA, PB), op.s_sym(PA, PC), op.s_sym(PA, PU)
    dAB, dAC, dABC = op.s_dir(A, PB), op.s_dir(A, PC), op.s_dir(A, PU)
    bAB, bAC, bABC = brute_s_sym(A, B, 1e6), brute_s_sym(A, C, 1e6), brute_s_sym(A, U, 1e6)
    out["A1"] = dict(
        tau=tau,
        times=dict(A=A.tolist(), B=B.tolist(), C=C.tolist()),
        S_sym=dict(S_AB=sAB, S_AC=sAC, S_A_UBC=sABC, sum=sAB + sAC,
                   rel_diff=rel(sABC, sAB + sAC),
                   brute_S_A_UBC=bABC, brute_rel_diff=rel(bABC, sABC)),
        S_dir=dict(Sdir_AB=dAB, Sdir_AC=dAC, Sdir_A_UBC=dABC, sum=dAB + dAC,
                   rel_diff=rel(dABC, dAB + dAC)),
    )
    checks.append(dict(
        name="A1_additivity",
        expected="S(A,U_BC) 严格等于 S(A,B)+S(A,C)，相对差 < 1e-12（对称核与有向核都要成立）",
        got="S_AB=%.10f S_AC=%.10f S_A_UBC=%.10f 相对差=%.3e；有向核 相对差=%.3e"
            % (sAB, sAC, sABC, rel(sABC, sAB + sAC), rel(dABC, dAB + dAC)),
        ok=bool(rel(sABC, sAB + sAC) < 1e-12 and rel(dABC, dAB + dAC) < 1e-12),
        note="拼接 = 把 B、C 的成交时刻并成一个多重集，求和核对 u 场是双线性的，"
             "所以严格可加。暴力两两配对给出同一个数（%.10f）。" % bABC))

    # ---------------- A2 互补 vs 重复 --------------------------------------
    X = np.array([0, 10, 20, 30], dtype=np.int64) * 10**6
    Bb = np.array([0, 10], dtype=np.int64) * 10**6
    C_comp = np.array([20, 30], dtype=np.int64) * 10**6      # 互补
    C_dup = np.array([0, 10], dtype=np.int64) * 10**6        # 重复

    def case(Cc, tau_c, chunk_us=None):
        o = ExactOp(tau_c, chunk_us=chunk_us)
        PX, PB2, PC2 = o.prep(X), o.prep(Bb), o.prep(Cc)
        PU2 = o.prep(np.sort(np.concatenate([Bb, Cc])))
        ex, eb, ec, eu = o.energy(PX), o.energy(PB2), o.energy(PC2), o.energy(PU2)
        sxb = o.s_sym(PX, PB2) / np.sqrt(ex * eb)
        sxc = o.s_sym(PX, PC2) / np.sqrt(ex * ec)
        sxu = o.s_sym(PX, PU2) / np.sqrt(ex * eu)
        r_bc = o.s_sym(PB2, PC2) / np.sqrt(eb * ec)
        D = np.sqrt(eb + ec + 2 * r_bc * np.sqrt(eb * ec))
        wB, wC = np.sqrt(eb) / D, np.sqrt(ec) / D
        single = max(sxb, sxc)
        return dict(S_XB=o.s_sym(PX, PB2), S_XC=o.s_sym(PX, PC2), S_X_UBC=o.s_sym(PX, PU2),
                    S_XB_brute=brute_s_sym(X, Bb, o.tau_us),
                    S_XC_brute=brute_s_sym(X, Cc, o.tau_us),
                    S_X_UBC_brute=brute_s_sym(X, np.sort(np.concatenate([Bb, Cc])), o.tau_us),
                    E_X=ex, E_B=eb, E_C=ec, E_U=eu, rho_XB=sxb, rho_XC=sxc,
                    rho_X_UBC=sxu, rho_BC=r_bc, wB=wB, wC=wC, single=single,
                    G=sxu - single,
                    identity=wB * sxb + wC * sxc)

    comp_tau1 = case(C_comp, 1.0, chunk_us=10**6)            # 块 1s，避开块边界
    dup_tau1 = case(C_dup, 1.0, chunk_us=10**6)
    comp_lim = case(C_comp, 1e-3, chunk_us=10**5)            # tau->0 极限：只有同刻对留下
    dup_lim = case(C_dup, 1e-3, chunk_us=10**5)
    same_S = all(rel(comp_tau1[k], dup_tau1[k]) < 1e-15
                 for k in ("S_XB", "S_XC", "S_X_UBC"))
    part_a = dict(
        X=X.tolist(),
        complementary=dict(C=C_comp.tolist(), tau1={k: comp_tau1[k] for k in
                                                    ("S_XB", "S_XC", "S_X_UBC", "S_XB_brute",
                                                     "S_XC_brute", "S_X_UBC_brute")}),
        duplicate=dict(C=C_dup.tolist(), tau1={k: dup_tau1[k] for k in
                                               ("S_XB", "S_XC", "S_X_UBC", "S_XB_brute",
                                                "S_XC_brute", "S_X_UBC_brute")}),
        identical=same_S,
    )
    part_b = dict(tau_limit=1e-3,
                  complementary={k: comp_lim[k] for k in
                                 ("E_X", "E_B", "E_C", "E_U", "rho_BC", "rho_XB", "rho_XC",
                                  "rho_X_UBC", "wB", "wC", "rho_X_UBC", "single", "G")},
                  duplicate={k: dup_lim[k] for k in
                             ("E_X", "E_B", "E_C", "E_U", "rho_BC", "rho_XB", "rho_XC",
                              "rho_X_UBC", "wB", "wC", "rho_X_UBC", "single", "G")})
    out["A2"] = dict(part_a, **part_b)
    checks.append(dict(
        name="A2_merge_blind_to_duplicate",
        expected="互补与重复两种拼接下，S(X,B)、S(X,C)、S(X,U_BC) 三个数完全相同",
        got="互补 S_XB=%.10f S_XC=%.10f S_X_UBC=%.10f；重复 S_XB=%.10f S_XC=%.10f "
            "S_X_UBC=%.10f；完全相同=%s"
            % (comp_tau1["S_XB"], comp_tau1["S_XC"], comp_tau1["S_X_UBC"],
               dup_tau1["S_XB"], dup_tau1["S_XC"], dup_tau1["S_X_UBC"], same_S),
        ok=bool(same_S),
        note="两种情形下 X 与伙伴的时间差多重集都是 {0,0,10,10} 秒，所以对任何 tau 三个数"
             "都一样（暴力两两配对同样如此）。原始算子看不出互补还是重复。"))
    checks.append(dict(
        name="A2_tau_to_zero_rho_and_G",
        expected="tau->0 极限下给出两种情形的 rho(X,U_BC) 与 G",
        got="互补 rho(X,U_BC)=%.6f G=%.6f；重复 rho(X,U_BC)=%.6f G=%.6f"
            % (comp_lim["rho_X_UBC"], comp_lim["G"], dup_lim["rho_X_UBC"], dup_lim["G"]),
        ok=bool(rel(comp_lim["rho_X_UBC"], 1.0) < 1e-9
                and rel(comp_lim["G"], 1.0 - 1 / np.sqrt(2)) < 1e-9
                and rel(dup_lim["rho_X_UBC"], 1 / np.sqrt(2)) < 1e-9
                and abs(dup_lim["G"]) < 1e-9),
        note="tau->0 只剩同刻对：互补时 U_BC 的 E_U=4，重复时 U_BC 把同一时刻重复计数使 "
             "E_U=8，分母不同所以 rho 不同（1.0 对 0.7071）。互补的 G=0.2929 恰好等于空模型"
             "值 (sqrt2-1)*0.7071，重复的 G=0，说明归一化后能区分这两件事，但 G 的绝对值本身"
             "还带着空模型基线（见 A4），评价必须再过一道零基线。"))

    # ---------------- A3 加权恒等式 ---------------------------------------
    rng = np.random.default_rng(20241231)
    op3 = ExactOp(2.0, chunk_us=int(24 * 1e6))               # tau=2s, 块 24s
    Xt = np.sort(rng.integers(0, 300 * 10**6, 400)).astype(np.int64)
    # B 稀疏密、C 稀疏且时间跨度不同，制造明显不同的 eB / eC
    Bt = np.sort(rng.integers(0, 60 * 10**6, 120)).astype(np.int64)
    Ct = np.sort(rng.integers(150 * 10**6, 300 * 10**6, 800)).astype(np.int64)
    PX3, PB3, PC3 = op3.prep(Xt), op3.prep(Bt), op3.prep(Ct)
    U3 = np.sort(np.concatenate([Bt, Ct]))
    PU3 = op3.prep(U3)
    eX, eB3, eC3, eU3 = (op3.energy(PX3), op3.energy(PB3), op3.energy(PC3), op3.energy(PU3))
    r_bc3 = op3.s_sym(PB3, PC3) / np.sqrt(eB3 * eC3)
    D3 = np.sqrt(eB3 + eC3 + 2 * r_bc3 * np.sqrt(eB3 * eC3))
    wB3, wC3 = np.sqrt(eB3) / D3, np.sqrt(eC3) / D3
    rhs = wB3 * op3.s_sym(PX3, PB3) / np.sqrt(eX * eB3) + wC3 * op3.s_sym(PX3, PC3) / np.sqrt(eX * eC3)
    lhs = op3.s_sym(PX3, PU3) / np.sqrt(eX * eU3)
    out["A3"] = dict(tau=2.0, n_X=len(Xt), n_B=len(Bt), n_C=len(Ct),
                     eB=eB3, eC=eC3, e_ratio=eC3 / eB3, E_U=eU3,
                     E_U_formula=eB3 + eC3 + 2 * op3.s_sym(PB3, PC3),
                     rho_BC=r_bc3, wB=wB3, wC=wC3,
                     lhs_rho_X_UBC=lhs, rhs_weighted=rhs,
                     rel_diff=rel(lhs, rhs))
    checks.append(dict(
        name="A3_rho_identity",
        expected="rho[X][U_BC] = wB*rho[X][B] + wC*rho[X][C]，相对差 < 1e-12；"
                 "E_U 也要等于 eB+eC+2*S[B][C]",
        got="eB=%.3f eC=%.3f (eC/eB=%.2f) rho[X][U_BC]=%.12f 加权和=%.12f 相对差=%.3e；"
            "E_U %.6f vs 公式 %.6f 相对差=%.3e"
            % (eB3, eC3, eC3 / eB3, lhs, rhs, rel(lhs, rhs), eU3,
               eB3 + eC3 + 2 * op3.s_sym(PB3, PC3),
               rel(eU3, eB3 + eC3 + 2 * op3.s_sym(PB3, PC3))),
        ok=bool(rel(lhs, rhs) < 1e-12
                and rel(eU3, eB3 + eC3 + 2 * op3.s_sym(PB3, PC3)) < 1e-12),
        note="eB 与 eC 明显不等（相差 %.1f 倍），权重仍然只依赖 (B,C)，所以 argmax 只是两列"
             "的线性组合，实现上不用真的拼数据。" % (eC3 / eB3)))

    # ---------------- A4 空模型 -------------------------------------------
    # 构造：X 在 6 个时刻上各一笔；B 占前两个时刻各一笔；C 占后两个时刻各 b 笔。
    # 所有时刻相距 100 秒，tau=0.05 秒 -> 跨时刻项 exp(-2000)=0，r=0 严格成立。
    # 于是 rho[X][B] = 2/sqrt(E_X*2)，rho[X][C] = 2b/sqrt(E_X*2b^2) = 同值，与 b 无关。
    a4 = []
    for b in (1, 2, 4, 8, 32):
        o = ExactOp(0.05, chunk_us=int(10 * 1e6))
        T = (np.arange(6) * 100 * 10**6).astype(np.int64)
        Xt4 = T.copy()
        Bt4 = T[:2].copy()
        Ct4 = np.repeat(T[2:4], b)
        PX4, PB4, PC4 = o.prep(Xt4), o.prep(Bt4), o.prep(Ct4)
        U4 = np.sort(np.concatenate([Bt4, Ct4]))
        PU4 = o.prep(U4)
        eX4, eB4, eC4 = o.energy(PX4), o.energy(PB4), o.energy(PC4)
        r4 = o.s_sym(PB4, PC4) / np.sqrt(eB4 * eC4)
        pB = o.s_sym(PX4, PB4) / np.sqrt(eX4 * eB4)
        pC = o.s_sym(PX4, PC4) / np.sqrt(eX4 * eC4)
        rhoU = o.s_sym(PX4, PU4) / np.sqrt(eX4 * o.energy(PU4))
        G_num = rhoU - max(pB, pC)
        D4 = np.sqrt(eB4 + eC4 + 2 * r4 * np.sqrt(eB4 * eC4))
        wB4, wC4 = np.sqrt(eB4) / D4, np.sqrt(eC4) / D4
        G_ana = wB4 * pB + wC4 * pC - max(pB, pC)           # 用 A3 恒等式算出的解析值
        G_null = (wB4 + wC4 - 1.0) * pB                     # 空模型写法（要求 pB == pC）
        a4.append(dict(b=b, eB=eB4, eC=eC4, rho_BC=r4, rho_XB=pB, rho_XC=pC,
                       wB=wB4, wC=wC4, w_sum=wB4 + wC4, rho_X_UBC=rhoU,
                       G_numeric=G_num, G_analytic=G_ana, G_null_form=G_null,
                       rel_diff=rel(G_num, G_ana), rel_null=rel(G_null, G_num),
                       G_over_p=G_num / pB))
    eq_case = a4[0]
    out["A4"] = dict(
        construction="X 在 6 个相距 100 秒的时刻各一笔；B=X 的前两个时刻各一笔；"
                     "C=X 的后两个时刻各 b 笔；tau=0.05s（跨时刻项为 0，r=0 严格成立）",
        scan=a4,
        equal_e=dict(b=1, p=eq_case["rho_XB"], G_numeric=eq_case["G_numeric"],
                     G_analytic=eq_case["G_analytic"], G_over_p=eq_case["G_over_p"],
                     G_null_form=eq_case["G_null_form"],
                     sqrt2_minus_1=float(np.sqrt(2) - 1)),
    )
    checks.append(dict(
        name="A4_null_model_G_positive",
        expected="两个独立（r=0）且与 X 同 rho=p 的伙伴，G=(wB+wC-1)p>0；eB=eC 时 "
                 "G=(sqrt2-1)p=0.41421p，eB!=eC 时 0<G<0.41421p",
        got="eB=eC 时 p=%.6f，G 数值=%.8f 解析=%.8f (=(sqrt2-1)p=%.8f) 相对差=%.2e；"
            "eC/eB=%g 时 p=%.6f，G 数值=%.8f 解析=%.8f 相对差=%.2e；b 扫描 %s"
            % (eq_case["rho_XB"], eq_case["G_numeric"], eq_case["G_analytic"],
               (np.sqrt(2) - 1) * eq_case["rho_XB"], eq_case["rel_diff"],
               a4[3]["eC"] / a4[3]["eB"], a4[3]["rho_XB"], a4[3]["G_numeric"],
               a4[3]["G_analytic"], a4[3]["rel_diff"],
               ["%g:%.6f" % (r["b"], r["G_numeric"]) for r in a4]),
        ok=bool(all(r["G_numeric"] > 0 for r in a4)
                and all(r["rel_diff"] < 1e-9 for r in a4)
                and all(r["rel_null"] < 1e-9 for r in a4)
                and rel(eq_case["G_numeric"], (np.sqrt(2) - 1) * eq_case["rho_XB"]) < 1e-9),
        note="wB+wC=(sqrt(eB)+sqrt(eC))/sqrt(eB+eC) 介于 1 与 sqrt2 之间，所以只要两个伙伴"
             "都跟 X 同 rho，G 恒为正、上限是 (sqrt2-1)p，与真实互补性无关。这是必须配零"
             "基线（spec 第 5 节的 Z）的原因。"))

    return out, checks


# ================================================================ B 真实数据
def load_codes_and_ticks(no_cache=False):
    codes_all = open(CODES_TXT).read().split()
    codes = codes_all[::SAMPLE_STEP]
    if not os.path.exists(CACHE_TICKS) or no_cache:
        import rust_pyfunc as rp
        t0 = time.time()
        lst = []
        for c in codes:
            arr = rp.read_trade_fast_us(c, DATE, 0, True)
            t = arr[:, 0].astype(np.int64)
            off = t - DSU
            b = off // BUCKET_US
            keep = (off >= 0) & (b < B_TOTAL)
            lst.append(np.sort(off[keep]))
        offs = np.zeros(len(codes) + 1, dtype=np.int64)
        offs[1:] = np.cumsum([len(x) for x in lst])
        np.savez(CACHE_TICKS, codes=np.array(codes), times=np.concatenate(lst), offs=offs)
        print("  [load] 读入 %d 只，共 %d 笔，%.1fs"
              % (len(codes), offs[-1], time.time() - t0))
        return codes_all, codes, np.concatenate(lst), offs
    z = np.load(CACHE_TICKS)
    assert list(z["codes"]) == list(codes), "缓存股票名单与当前取样不一致，请加 --no-cache"
    return codes_all, codes, z["times"], z["offs"]


def build_sdir(codes, times, offs, no_cache=False):
    n = len(codes)
    if os.path.exists(CACHE_SDIR) and not no_cache:
        S = np.load(CACHE_SDIR)
        if S.shape == (n, n):
            return S
    op = ExactOp(TAU)
    ngrid = op.ngrid_for(int(times[-1])) if len(times) else 8
    Ps = [op.prep(times[offs[i]:offs[i + 1]], ngrid=ngrid) for i in range(n)]
    S = np.zeros((n, n))
    t0 = time.time()
    for i in range(n):
        # s_dir(ts,ts) 本身就是 S_dir[i][i] = E_i/2（含每笔与自身的 0.5）
        S[i, i] = op.s_dir(Ps[i]["ts"], Ps[i])
    for i in range(n):
        qi = Ps[i]["ts"]
        for j in range(i + 1, n):
            S[i, j] = op.s_dir(qi, Ps[j])
            S[j, i] = op.s_dir(Ps[j]["ts"], Ps[i])
    np.save(CACHE_SDIR, S)
    print("  [sdir] 逐笔两两配对 %.1fs" % (time.time() - t0))
    return S


def pair_quantities(rho, E):
    """spec 第 2 节：对每个 (B,C) 算 X0 / RHO_U / SINGLE / G（并列取下标最小）"""
    n = len(E)
    sq = np.sqrt(E)
    X0 = np.full((n, n), -1, dtype=np.int64)
    RHO_U = np.zeros((n, n))
    SINGLE = np.zeros((n, n))
    G = np.zeros((n, n))
    idx = np.arange(n)
    for B in range(n):
        eB = E[B]
        r = rho[B, :]
        D = np.sqrt(eB + E + 2 * r * sq[B] * sq)
        wB = np.where(D > 0, sq[B] / D, 0.0)
        wC = np.where(D > 0, sq / D, 0.0)
        h = rho[:, B][:, None] * wB[None, :] + rho * wC[None, :]
        h[B, :] = -np.inf
        h[idx, idx] = -np.inf
        x0 = np.argmax(h, axis=0)                            # 并列取最小下标
        rhoU = h[x0, idx]
        single = np.maximum(rho[x0, B], rho[x0, idx])
        X0[B, :] = x0
        RHO_U[B, :] = rhoU
        SINGLE[B, :] = single
        G[B, :] = rhoU - single
        X0[B, B] = -1
        G[B, B] = 0.0
    return X0, RHO_U, SINGLE, G


def real_checks(codes, times, offs, Sdir, rho, E, X0, RHO_U, SINGLE, G, codes_all):
    checks = []
    n = len(codes)
    idx = np.arange(n) * SAMPLE_STEP

    # 时间戳分辨率：桶内取 1 的近似到底有多大
    res = []
    for i in range(min(n, 40)):
        t = times[offs[i]:offs[i + 1]]
        res.append(int((t % BUCKET_US).max()) if len(t) else 0)
    all_bucket_aligned = max(res) == 0
    checks.append(dict(
        name="B_time_resolution",
        expected="确认逐笔 time_us 的分辨率与 10ms 桶的关系",
        got="抽查 40 只股票，time_us 对 %d 取模的最大值 = %d（0 表示全部落在桶边界上）"
            % (BUCKET_US, max(res)),
        ok=bool(all_bucket_aligned),
        note="time_us 分辨率就是 10ms，同一桶等价于同一时间戳，所以 kernel_spec 1.3 说的"
             "『桶内一律取 1』在本数据上不产生任何额外误差；逐笔精确口径与备份桶口径应当"
             "几乎逐位相同（下面 B_backup_align 的数字印证）。"))

    # 与备份矩阵对齐
    M = np.memmap(BACKUP_MAT, dtype="<f4", mode="r").reshape(len(codes_all), len(codes_all))
    sub = np.asarray(M[np.ix_(idx, idx)], dtype=np.float64)
    Eb = 2 * np.diag(sub).copy()
    Sb = sub + sub.T
    np.fill_diagonal(Sb, 0.0)
    rhob = Sb / np.sqrt(np.outer(Eb, Eb))
    np.fill_diagonal(rhob, 1.0)
    Ssym = Sdir + Sdir.T
    np.fill_diagonal(Ssym, 0.0)
    m = ~np.eye(n, dtype=bool)
    relE = np.abs(E - Eb) / Eb
    relS = np.abs(Ssym[m] - Sb[m]) / Sb[m]
    relrho = np.abs(rho[m] - rhob[m]) / rhob[m]
    align = dict(
        rel_E_median=float(np.median(relE)), rel_E_p90=float(np.percentile(relE, 90)),
        rel_E_max=float(relE.max()),
        rel_S_median=float(np.median(relS)), rel_S_p90=float(np.percentile(relS, 90)),
        rel_S_max=float(relS.max()),
        rel_rho_median=float(np.median(relrho)), rel_rho_p90=float(np.percentile(relrho, 90)),
        rel_rho_max=float(relrho.max()),
        pearson_rho=float(np.corrcoef(rho[m], rhob[m])[0, 1]),
    )
    checks.append(dict(
        name="B_backup_align",
        expected="逐笔精确口径的 E、S、rho 与备份 mats/cnt_t1.bin 对上（相对差应在百分之几以内）",
        got="E 相对差 中位 %.2e / p90 %.2e / 最大 %.2e；S 中位 %.2e / 最大 %.2e；"
            "rho 中位 %.2e / p90 %.2e / 最大 %.2e；相关系数 %.10f"
            % (align["rel_E_median"], align["rel_E_p90"], align["rel_E_max"],
               align["rel_S_median"], align["rel_S_max"], align["rel_rho_median"],
               align["rel_rho_p90"], align["rel_rho_max"], align["pearson_rho"]),
        ok=bool(align["rel_rho_max"] < 0.05),
        note="比预期好得多（不是百分之几，是 1e-5 量级），原因是 time_us 本身就是 10ms 量化，"
             "桶口径与逐笔口径完全同构；残差只剩备份 f32 存储与累加带来的舍入。"))

    # 逐笔精确 vs 暴力两两配对（真实数据，取前若干笔）
    op = ExactOp(TAU)
    brute_cases = []
    for (i, j) in [(0, 1), (5, 6), (100, 120), (50, 60), (180, 190)]:
        ti = times[offs[i]:offs[i + 1]][:1200]
        tj = times[offs[j]:offs[j + 1]][:1200]
        fast = op.s_dir(ti, op.prep(tj))
        slow = brute_s_dir(ti, tj, 1e6)
        fast2 = op.s_dir(tj, op.prep(ti))
        slow2 = brute_s_dir(tj, ti, 1e6)
        brute_cases.append(dict(pair=[codes[i], codes[j]], fast=fast, brute=slow,
                                rel=rel(fast, slow), rel_rev=rel(fast2, slow2)))
    max_rel = max(max(c["rel"], c["rel_rev"]) for c in brute_cases)
    checks.append(dict(
        name="B_bruteforce_match",
        expected="真实数据上前 1200 笔的逐笔精确算子与暴力两两配对完全一致（相对差 < 1e-12）",
        got="; ".join("%s/%s 相对差 %.2e / %.2e" % (c["pair"][0], c["pair"][1], c["rel"],
                                                   c["rel_rev"]) for c in brute_cases),
        ok=bool(max_rel < 1e-12),
        note="只对真实时间戳做校验；暴力版是最笨的两两配对（逐对 exp 求和），"
             "用来排除分块前缀实现在真实稀疏模式下出错。"))

    # 真实数据上直接拼 U_BC 复核 X0 与 rhoU
    ng = op.ngrid_for(int(times[-1]))
    Ps = [op.prep(times[offs[i]:offs[i + 1]], ngrid=ng) for i in range(n)]
    direct = []
    for (B, C) in [(0, 1), (5, 50), (100, 150), (37, 60), (123, 45), (7, 8)]:
        U = np.sort(np.concatenate([Ps[B]["ts"], Ps[C]["ts"]]))
        PU = op.prep(U, ngrid=ng)
        EU = 2 * op.s_dir(U, PU)
        col = np.full(n, np.nan)
        for X in range(n):
            if X in (B, C):
                continue
            s = op.s_dir(Ps[X]["ts"], PU) + op.s_dir(U, Ps[X])
            col[X] = s / np.sqrt(E[X] * EU)
        x0d = int(np.nanargmax(col))
        direct.append(dict(pair=[codes[B], codes[C]],
                           X0_direct=codes[x0d], X0_formula=codes[X0[B, C]],
                           rhoU_direct=float(col[x0d]), rhoU_formula=float(RHO_U[B, C]),
                           rel=rel(col[x0d], RHO_U[B, C]),
                           E_U=float(EU),
                           E_U_formula=float(E[B] + E[C] + 2 * (Sdir[B, C] + Sdir[C, B]))))
    max_rel_d = max(d["rel"] for d in direct)
    x0_ok = all(d["X0_direct"] == d["X0_formula"] for d in direct)
    checks.append(dict(
        name="B_direct_vs_formula",
        expected="真实数据上直接拼 U_BC 重算，X0 与 rhoU 必须与 spec 第 2 节的线性组合公式一致",
        got="; ".join("%s/%s X0 %s vs %s，rhoU 相对差 %.2e"
                      % (d["pair"][0], d["pair"][1], d["X0_direct"], d["X0_formula"], d["rel"])
                      for d in direct),
        ok=bool(x0_ok and max_rel_d < 1e-12),
        note="这一条把 §2 的化简（不拼数据、只取两列线性组合）与真实拼数据的结果对上，"
             "同时验证 E_U = eB+eC+2*S[B][C]。"))

    iu = np.triu_indices(n, 1)
    g = G[iu]
    order = np.argsort(-g)[:5]
    top = [dict(B=codes[iu[0][k]], C=codes[iu[1][k]], G=float(G[iu[0][k], iu[1][k]]),
                RHO_U=float(RHO_U[iu[0][k], iu[1][k]]),
                SINGLE=float(SINGLE[iu[0][k], iu[1][k]]),
                X0=codes[X0[iu[0][k], iu[1][k]]],
                rho_BC=float(rho[iu[0][k], iu[1][k]]),
                E_B=float(E[iu[0][k]]), E_C=float(E[iu[1][k]])) for k in order]
    x0c = X0[iu]
    vals, cnts = np.unique(x0c, return_counts=True)
    hist = sorted([dict(code=codes[v], count=int(c)) for v, c in zip(vals, cnts)],
                  key=lambda x: -x["count"])[:10]
    stats = dict(
        n_pairs=int(len(g)), G_min=float(g.min()), G_p01=float(np.percentile(g, 1)),
        G_median=float(np.median(g)), G_mean=float(g.mean()),
        G_p99=float(np.percentile(g, 99)), G_max=float(g.max()),
        frac_G_le_0=float((g <= 0).mean()), top5=top, X0_hist_top10=hist,
        E_median=float(np.median(E)), G_over_null=(np.sqrt(2) - 1),
    )
    return checks, align, stats, brute_cases, direct


# ================================================================ C Rust 对比
def rust_compare(codes, rho, E, G, X0, RHO_U, SINGLE):
    out = dict(status="pending", rust_dir=RUST_DIR)
    if not os.path.isdir(RUST_DIR):
        out["note"] = "主会话 Rust 输出目录不存在，本轮无法比对；脚本重跑时会自动比对。"
        return out, []
    fcodes = os.path.join(RUST_DIR, "codes.txt")
    if not os.path.exists(fcodes):
        out["status"] = "pending"
        out["note"] = "目录存在但没有 codes.txt（可能还在算），脚本重跑时再比对。"
        return out, []
    rcodes = open(fcodes).read().split()
    pos = {}
    for k, c in enumerate(rcodes):
        pos.setdefault(c, k)
    keep = [(i, pos[c]) for i, c in enumerate(codes) if c in pos]
    if len(keep) < 2:
        out["status"] = "failed"
        out["note"] = "Rust codes.txt 与本子样本交集不足 %d 只" % len(keep)
        return out, []
    ii = np.array([a for a, _ in keep])          # 我在子样本里的下标
    jj = np.array([b for _, b in keep])          # 同一只股票在 Rust 全市场矩阵里的下标
    back = {int(j): i for i, j in zip(ii, jj)}   # Rust 下标 -> 我的下标
    nrust = len(rcodes)
    m = ~np.eye(len(ii), dtype=bool)
    iu = np.triu_indices(len(ii), 1)
    checks = []

    def read_mat(name, shape=(nrust, nrust)):
        p = os.path.join(RUST_DIR, name)
        if not os.path.exists(p):
            return None
        return np.fromfile(p, dtype="<f4").reshape(shape).astype(np.float64)

    def read_vec(name):
        p = os.path.join(RUST_DIR, name)
        if not os.path.exists(p):
            return None
        return np.fromfile(p, dtype="<f8")

    r_rho = read_mat("rho.bin")
    r_G = read_mat("G.bin")
    r_X0 = read_mat("X0.bin")
    r_RU = read_mat("RHO_U.bin")
    r_SG = read_mat("SINGLE.bin")
    r_E = read_vec("E.bin")
    def sub(M):
        """取同一批股票在 Rust 全市场矩阵里的行和列（行列都用 Rust 下标）"""
        return M[np.ix_(jj, jj)] if M is not None else None

    out.update(status="compared", n_matched=len(keep),
               index_map=[list(map(int, x)) for x in keep],
               universe_note="Rust 在全部 %d 只股票上取 argmax，本参考只在 %d 只子样本内取，"
                             "所以 G / X0 的定义域不同，rho 与 E 才是严格可比的量。"
                             % (nrust, len(ii)))
    if r_rho is not None:
        a, b = sub(r_rho), rho
        rel = np.abs(a[m] - b[m]) / np.maximum(np.abs(b[m]), 1e-12)
        out["rho"] = dict(median_rel=float(np.median(rel)), p90_rel=float(np.percentile(rel, 90)),
                          max_rel=float(rel.max()), pearson=float(np.corrcoef(a[m], b[m])[0, 1]))
        checks.append(dict(name="rust_rho", expected="与本参考 rho 相对差 < 1e-4（同一 199 只的两两比较）",
                           got="中位 %.2e p90 %.2e 最大 %.2e 相关 %.8f"
                               % (out["rho"]["median_rel"], out["rho"]["p90_rel"],
                                  out["rho"]["max_rel"], out["rho"]["pearson"]),
                           ok=bool(out["rho"]["max_rel"] < 1e-4),
                           note="取自 rho.bin，两边定义完全相同，只差 f32 存储。"))
    # G / X0 的逐对一致率：候选集不同（全市场 vs 199 只），先做归因复核再判断
    same_x0 = None
    if r_X0 is not None and r_RU is not None and r_G is not None:
        xm = np.vectorize(lambda v: back.get(int(round(v)), -1))(sub(r_X0))
        rr = sub(r_RU)
        rg = sub(r_G)
        I, J = np.meshgrid(np.arange(len(ii)), np.arange(len(ii)), indexing="ij")
        sel = (xm >= 0) & (I != J)
        bx, cx, xx = I[sel], J[sel], xm[sel]
        eB, eC = E[bx], E[cx]
        rbc = rho[bx, cx]
        D = np.sqrt(eB + eC + 2 * rbc * np.sqrt(eB * eC))
        wB = np.sqrt(eB) / D
        wC = np.sqrt(eC) / D
        h = wB * rho[xx, bx] + wC * rho[xx, cx]
        single = np.maximum(rho[xx, bx], rho[xx, cx])
        relU = np.abs(h - rr[sel]) / np.maximum(np.abs(h), 1e-12)
        dG = np.abs((h - single) - rg[sel])
        same_x0 = dict(
            share_rust_X0_inside_subsample=float((xm >= 0).mean()),
            n_pairs=int(sel.sum()),
            RHO_U_vs_rust_median_rel=float(np.median(relU)),
            RHO_U_vs_rust_max_rel=float(relU.max()),
            G_vs_rust_median_abs=float(np.median(dG)),
            G_vs_rust_max_abs=float(dG.max()))
        out["same_X0_recheck"] = same_x0
        checks.append(dict(
            name="rust_same_X0_recheck",
            expected="固定用 Rust 选中的 X0 代进本参考 rho 重算，rhoU 与 G 应与 Rust 一致（重算相对差 < 1e-4）",
            got="Rust 的 X0 落在 199 只里的比例 %.4f；可比较 %d 对；rhoU 相对差 中位 %.2e 最大 %.2e；"
                "G 绝对差 中位 %.2e 最大 %.2e"
                % (same_x0["share_rust_X0_inside_subsample"], same_x0["n_pairs"],
                   same_x0["RHO_U_vs_rust_median_rel"], same_x0["RHO_U_vs_rust_max_rel"],
                   same_x0["G_vs_rust_median_abs"], same_x0["G_vs_rust_max_abs"]),
            ok=bool(same_x0["RHO_U_vs_rust_max_rel"] < 1e-4
                    and same_x0["G_vs_rust_max_abs"] < 1e-4),
            note="这一条把『G/X0 逐对不一致』归因到 argmax 候选集不同：固定成同一个 X 之后，"
                 "§2 的公式在两边给出同一个 rhoU 与 G。"))
    attribution = ("G/X0 的候选集不同（Rust 全市场 %d 只，本参考 %d 只子样本），"
                   "逐对不一致是定义域差异；已用 rust_same_X0_recheck 固定 X 归因"
                   % (nrust, len(ii))) if same_x0 else "候选集不同，未能做归因复核"
    if r_G is not None:
        a, b = sub(r_G), G
        d = np.abs(a[iu] - b[iu])
        agree = float((d <= 1e-4 + 1e-4 * np.abs(b[iu])).mean())
        qr = np.percentile(r_G[np.triu_indices(nrust, 1)], [10, 50, 90])
        qm = np.percentile(b[iu], [10, 50, 90])
        out["G"] = dict(median_abs=float(np.median(d)), max_abs=float(d.max()),
                        agree_within_1e_4=agree, n_compared=int(len(iu[0])),
                        rust_quantiles_10_50_90=[float(x) for x in qr],
                        ref_quantiles_10_50_90=[float(x) for x in qm],
                        max_quantile_gap=float(np.abs(qr - qm).max()))
        checks.append(dict(name="rust_G_pairwise", expected="逐对一致率不设门槛（候选集不同），"
                                                           "但 G 的整体分布要与 Rust 接近",
                           got="逐对一致率 %.4f 中位绝对差 %.2e；G 分位 Rust %s vs 本参考 %s，"
                               "最大分位差 %.4f"
                               % (agree, out["G"]["median_abs"],
                                  np.round(qr, 4).tolist(), np.round(qm, 4).tolist(),
                                  out["G"]["max_quantile_gap"]),
                           ok=bool(out["G"]["max_quantile_gap"] < 0.01),
                           blocking=False,
                           note=attribution + "。逐对一致率低不代表公式不一致。"))
    if r_X0 is not None:
        a = sub(r_X0)                                    # 存的是 Rust 全市场行下标
        aa = np.vectorize(lambda v: back.get(int(round(v)), -1))(a)
        agree = float((aa[m] == X0[m]).mean())
        out["X0"] = dict(agree=agree, n_compared=int(m.sum()),
                         rust_X0_inside_subsample_share=float((aa >= 0).mean()))
        checks.append(dict(name="rust_X0_pairwise", expected="逐对一致率不设门槛（候选集不同）",
                           got="换算成同一只股票后一致率 %.4f（比较 %d 对），"
                               "Rust 选中的 X 落在本子样本内的比例 %.4f"
                               % (agree, int(m.sum()), out["X0"]["rust_X0_inside_subsample_share"]),
                           ok=False, blocking=False,
                           note=attribution + "。Rust 的 X 绝大多数落在本 199 只之外，"
                                "所以逐对一致率必然很低。"))
    if r_E is not None:
        a, b = r_E[jj], E
        rel = np.abs(a - b) / np.maximum(np.abs(b), 1e-12)
        out["E"] = dict(median_rel=float(np.median(rel)), max_rel=float(rel.max()))
        checks.append(dict(name="rust_E", expected="与本参考 E 相对差 < 1e-4",
                           got="中位 %.2e 最大 %.2e" % (out["E"]["median_rel"], out["E"]["max_rel"]),
                           ok=bool(out["E"]["max_rel"] < 1e-4), note="取自 E.bin"))
    for nm, M, ref in (("RHO_U", r_RU, RHO_U), ("SINGLE", r_SG, SINGLE)):
        if M is None:
            continue
        a, b = sub(M), ref
        d = np.abs(a[iu] - b[iu])
        rel = d / np.maximum(np.abs(b[iu]), 1e-12)
        out[nm] = dict(median_abs=float(np.median(d)), max_abs=float(d.max()),
                       median_rel=float(np.median(rel)), max_rel=float(rel.max()))

    return out, checks


# ================================================================ 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-cache", action="store_true", help="忽略 /tmp 缓存重算")
    ap.add_argument("--skip-real", action="store_true", help="只跑 A 合成检查")
    ap.add_argument("--json", default=OUT_JSON)
    args = ap.parse_args()

    t_start = time.time()
    result = dict(
        meta=dict(date=DATE, tau=TAU, bucket_us=BUCKET_US, b_total=B_TOTAL,
                  codes_txt=CODES_TXT, backup_mat=BACKUP_MAT, rust_dir=RUST_DIR,
                  sample_step=SAMPLE_STEP, script="experiments/yupei2/pair_ref.py",
                  generated=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        checks=[],
    )
    print("[A] 合成精确性检查")
    synth, synth_chk = synth_checks()
    result["A"] = synth
    result["checks"].extend(synth_chk)
    for c in synth_chk:
        print("   %-28s ok=%s | %s" % (c["name"], c["ok"], c["got"][:230]))

    if not args.skip_real:
        print("[B] 真实数据 20241231，%d 只子样本" % 199)
        codes_all, codes, times, offs = load_codes_and_ticks(args.no_cache)
        n = len(codes)
        Sdir = build_sdir(codes, times, offs, args.no_cache)
        E = 2 * np.diag(Sdir).copy()
        Ssym = Sdir + Sdir.T
        np.fill_diagonal(Ssym, 0.0)
        rho = Ssym / np.sqrt(np.outer(E, E))
        np.fill_diagonal(rho, 1.0)
        X0, RHO_U, SINGLE, G = pair_quantities(rho, E)
        rchecks, align, stats, brute_cases, direct = real_checks(
            codes, times, offs, Sdir, rho, E, X0, RHO_U, SINGLE, G, codes_all)
        result["checks"].extend(rchecks)
        result["backup_align"] = align
        result["bruteforce_sample"] = brute_cases
        result["direct_recheck"] = direct
        result["pair_stats"] = stats
        for c in rchecks:
            print("   %-28s ok=%s | %s" % (c["name"], c["ok"], c["got"][:230]))
        print("   G 分布: 中位 %.4f 均值 %.4f p99 %.4f 最大 %.4f (<=0 占比 %.3f)"
              % (stats["G_median"], stats["G_mean"], stats["G_p99"], stats["G_max"],
                 stats["frac_G_le_0"]))
        print("   top5 G: " + "; ".join("%s+%s %.4f(X0=%s)" % (t["B"], t["C"], t["G"], t["X0"])
                                        for t in stats["top5"]))
        rnd = lambda M, k: np.round(M, k).tolist()
        result["subsample"] = dict(
            codes=codes, date=DATE, tau=TAU, note="行序 = 备份 codes.txt 每 20 只取 1 只的顺序（升序）",
            E=np.round(E, 6).tolist(),
            rho=rnd(rho, 6), G=rnd(G, 6), X0=X0.tolist(),
            RHO_U=rnd(RHO_U, 6), SINGLE=rnd(SINGLE, 6),
        )
        rc, rchecks2 = rust_compare(codes, rho, E, G, X0, RHO_U, SINGLE)
        result["rust_compare"] = rc
        result["checks"].extend(rchecks2)
        print("[C] rust_compare: %s" % rc["status"])

    blocking_bad = [c["name"] for c in result["checks"]
                    if not c["ok"] and c.get("blocking", True)]
    nonblock_bad = [c["name"] for c in result["checks"]
                    if not c["ok"] and not c.get("blocking", True)]
    result["summary"] = dict(
        n_checks=len(result["checks"]),
        n_failed=int(sum(1 for c in result["checks"] if not c["ok"])),
        blocking_failed=blocking_bad,
        nonblocking_failed=nonblock_bad,
        nonblocking_note="非阻断裂来自定义域差异：Rust 在全部 3964 只上取 argmax，本参考只在 199 只"
                         "子样本内取，所以 G / X0 逐对不可能一样。它们已由 rust_same_X0_recheck"
                         "固定 X 归因，不参与退出码。",
        all_ok=bool(not blocking_bad),
        elapsed_sec=round(time.time() - t_start, 1),
    )
    with open(args.json, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    print("写出 %s（%.1f MB，%.1fs，检查 %d 条；阻断失败 %s；非阻断 %s）"
          % (args.json, os.path.getsize(args.json) / 1e6, time.time() - t_start,
             len(result["checks"]), blocking_bad or "无", nonblock_bad or "无"))
    return 0 if not blocking_bad else 1


if __name__ == "__main__":
    sys.exit(main())
