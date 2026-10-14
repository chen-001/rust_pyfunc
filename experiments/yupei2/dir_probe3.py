#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""玉佩2 方向性降维 probe3：时间域（领先时标谱）与集合结构。

口径（与 src/yupei_dist/matrix_stage.rs 实测一致，不按注释读）：
  S[i][j] = codes[j] 的成交在前（领先）、codes[i] 在后（跟随）的衰减交互能量
  L[i][j] = S[j][i] = "i 领先 j" 能量；W[i][j] = S[i][j] + S[j][i]；q = L / W

S1 领先时标谱：对每对 (i,j) 用 cnt 族 9 个 τ 的 S_ij(τ) 对 x=1/τ、y=log S_ij(τ) 做最小二乘
      ℓ_ij = −slope = 能量加权的平均滞后（秒，别人领先我时我的响应滞后）
      ℓ_ji 同理（我领先别人时别人的响应滞后）
      ts_follow_i = Σ_j W_ij·ℓ_ij / Σ W_ij ,  ts_lead_i = Σ_j W_ij·ℓ_ji / Σ W_ij
      只在按 W 取的前 200 个伙伴上聚合
S2 跨 τ 方向一致性：9 个 τ 上 q>0.5 的符号，与多数符号一致的比例
S3 方向 × 同向/对盘 2×2：flow_same_t1 / flow_opp_t1 上的领先份额与占比
S4 集合重叠：互动集合（W 前 50）与领先集合（L 前 50）的 Jaccard；领先集合与跟随集合的 Jaccard

运行：
  /home/chenzongwei/.conda/envs/chenzongwei311/bin/python experiments/yupei2/dir_probe3.py [--smoke]
"""
import gc
import json
import os
import sys
import time

import numpy as np
import rust_pyfunc as rp
from scipy.stats import rankdata, spearmanr

BACKUP = "/hdd/user_home_unsafe/chenzongwei/yupei_dist_backup"
DATE_A, DATE_B = 20241230, 20241231
CACHE = "/tmp/yupei2_probe_cache"
OUT_JSON = "/home/chenzongwei/rust_pyfunc/experiments/yupei2_dir_reduce.probe3.json"

# cnt 族 9 个 τ（秒）
TAU_NAMES = ["cnt_t005", "cnt_t01", "cnt_t02", "cnt_t05", "cnt_t1",
             "cnt_t3", "cnt_t5", "cnt_t10", "cnt_t30"]
TAU_SEC = np.array([0.05, 0.1, 0.2, 0.5, 1.0, 3.0, 5.0, 10.0, 30.0])
SMALL = TAU_SEC <= 1.0 + 1e-9           # 只用 0.05~1 的对照子集
X = 1.0 / TAU_SEC
K = 200                                  # 按 W 取前 K 个伙伴
SMOKE_N = 400
TINY = 1e-30

STRENGTH_KEEP = 50                       # S4 的集合大小

# 与上一轮缓存里已有的对照口径（cnt_t1 上算的）
REF_STATS = {"lead_ratio": "st_soft_a1", "lead_breadth": "st_lead_breadth", "q_p90": "st_q_p90"}

NEW_STATS = [
    "ts_follow", "ts_lead", "ts_ratio",
    "ts_follow_small", "ts_lead_small", "ts_ratio_small",
    "tau_consistency",
    "same_lead_share", "opp_lead_share", "same_share_of_total",
    "jaccard_w_lead", "jaccard_lead_follow",
]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def read_codes(date):
    return [l.strip() for l in open(f"{BACKUP}/{date}/codes.txt") if l.strip()]


def read_mat(date, mat):
    return np.ascontiguousarray(rp.yupei_dist_backup_read_matrix(BACKUP, date, mat), dtype=np.float64)


def topk_idx(mat, k):
    """每行按值取前 k 大的列下标（含可能的自己，调用方自行过滤）。"""
    return np.argpartition(-mat, k, axis=1)[:, :k].astype(np.int32)


def day_stats(date, smoke=False):
    t0 = time.time()
    codes = read_codes(date)
    n = len(codes)
    if smoke:
        n = min(n, SMOKE_N)
        codes = codes[:n]
    st_codes, stats_bin = rp.yupei_dist_backup_read_stats(BACKUP, date)
    assert list(st_codes)[:n] == codes, "stats 行序与 codes.txt 不一致"
    n_trades = np.asarray(stats_bin[:n, 0], dtype=np.float64)

    M = read_mat(date, "cnt_t1")[:n, :n]
    MT = np.ascontiguousarray(M.T)
    W = M + MT
    np.fill_diagonal(W, 0.0)
    strength = W.sum(axis=1)
    log(f"{date} cnt_t1 读入 {n}x{n}，正准备取前 {K} 伙伴")

    idx = topk_idx(W, K)
    rows = np.broadcast_to(np.arange(n)[:, None], idx.shape)
    Wsel = np.take_along_axis(W, idx, axis=1)
    valid = (Wsel > 0.0) & (idx != rows)

    # ---- S1: 9 个 τ 的最小二乘充分统计量（形状 N×K）----
    def new_acc():
        return [np.zeros((n, K)) for _ in range(6)]   # n, sx, sxx, sy, sxy, syy

    accF, accL, accFs, accLs = new_acc(), new_acc(), new_acc(), new_acc()
    n_pos = np.zeros((n, K))       # S2: q>0.5 的 τ 个数
    n_sgn = np.zeros((n, K))       # S2: 参与计数的 τ 个数

    for t, name in enumerate(TAU_NAMES):
        Mt = read_mat(date, name)[:n, :n]
        A = np.take_along_axis(Mt, idx, axis=1)      # S[i][j] = j 领先 i
        B = Mt[idx, rows]                            # S[j][i] = i 领先 j
        x = X[t]
        for acc, Y in ((accF, A), (accL, B)):
            ok = (Y > 0.0) & valid
            acc[0] += ok
            acc[1] += np.where(ok, x, 0.0)
            acc[2] += np.where(ok, x * x, 0.0)
            yy = np.where(ok, np.log(np.maximum(Y, TINY)), 0.0)
            acc[3] += yy
            acc[4] += yy * x
            acc[5] += yy * yy
        if SMALL[t]:
            for acc, Y in ((accFs, A), (accLs, B)):
                ok = (Y > 0.0) & valid
                acc[0] += ok
                acc[1] += np.where(ok, x, 0.0)
                acc[2] += np.where(ok, x * x, 0.0)
                yy = np.where(ok, np.log(np.maximum(Y, TINY)), 0.0)
                acc[3] += yy
                acc[4] += yy * x
                acc[5] += yy * yy
        both = valid & ((A > 0.0) | (B > 0.0))
        n_sgn += both
        n_pos += both & (B > A)
        del Mt, A, B
        gc.collect()
    log(f"{date} 9 个 τ 扫描完成（{time.time()-t0:.1f}s）")

    def fit(acc, min_n=5):
        cnt, sx, sxx, sy, sxy, syy = acc
        den = cnt * sxx - sx * sx
        with np.errstate(divide="ignore", invalid="ignore"):
            slope = np.where(den > 0, (cnt * sxy - sx * sy) / den, np.nan)
            num = (cnt * sxy - sx * sy) ** 2
            r2den = den * (cnt * syy - sy * sy)
            r2 = np.where(r2den > 0, num / r2den, np.nan)
        ok = (cnt >= min_n) & valid
        slope = np.where(ok, slope, np.nan)
        r2 = np.where(ok, r2, np.nan)
        return -slope, r2          # ℓ = −slope

    ellF, r2F = fit(accF)
    ellL, r2L = fit(accL)
    ellFs, r2Fs = fit(accFs, min_n=4)
    ellLs, r2Ls = fit(accLs, min_n=4)

    def waggregate(ell):
        w = np.where(np.isfinite(ell), Wsel, 0.0)
        e = np.where(np.isfinite(ell), ell, 0.0)
        s = w.sum(axis=1)
        return np.where(s > 0, (w * e).sum(axis=1) / s, np.nan)

    ts_follow = waggregate(ellF)
    ts_lead = waggregate(ellL)
    with np.errstate(divide="ignore", invalid="ignore"):
        ts_ratio = np.where(ts_lead > 0, ts_follow / ts_lead, np.nan)
    ts_follow_s = waggregate(ellFs)
    ts_lead_s = waggregate(ellLs)
    with np.errstate(divide="ignore", invalid="ignore"):
        ts_ratio_s = np.where(ts_lead_s > 0, ts_follow_s / ts_lead_s, np.nan)
    log(f"{date} S1 完成 ts_follow 中位 {np.nanmedian(ts_follow):.4f}s "
        f"ts_lead 中位 {np.nanmedian(ts_lead):.4f}s")

    # ---- S2: 跨 τ 符号一致性 ----
    with np.errstate(divide="ignore", invalid="ignore"):
        agree = np.where(n_sgn >= 5,
                         np.maximum(n_pos, n_sgn - n_pos) / np.maximum(n_sgn, 1), np.nan)
    # 逐行 nanmean（整行都是 nan 时也要给出 nan，不用 np.nanmean 免得告警）
    tau_consistency = np.full(n, np.nan)
    for i in range(n):
        v = agree[i][np.isfinite(agree[i])]
        if v.size:
            tau_consistency[i] = v.mean()

    # ---- 方向口径自检：强领先的一对，其"领先方向"的 ℓ 应该更小、"反向"的 ℓ 更大 ----
    q_slot = np.where(Wsel > 0, np.divide(M[idx, rows], Wsel, out=np.full_like(Wsel, 0.5),
                                        where=Wsel > 0), 0.5)
    ok_slot = np.isfinite(ellF) & np.isfinite(ellL) & valid
    qc = q_slot[ok_slot]
    r_ellF = float(spearmanr(qc, ellF[ok_slot]).statistic)
    r_ellL = float(spearmanr(qc, ellL[ok_slot]).statistic)
    r_ellFL = float(spearmanr(ellF[ok_slot], ellL[ok_slot]).statistic)
    # 逐对检验：方向份额明显偏向一侧的那些对，能量大的一侧时标应该更短
    grp = {}
    for tag, m in (("q_lt_035=j领先i强", qc < 0.35), ("q_gt_065=i领先j强", qc > 0.65)):
        if m.sum() < 100:
            grp[tag] = {"n": int(m.sum())}
            continue
        ef, el = ellF[ok_slot][m], ellL[ok_slot][m]
        grp[tag] = {
            "n": int(m.sum()),
            "ell_follow_mean": float(np.mean(ef)), "ell_lead_mean": float(np.mean(el)),
            "ell_follow_median": float(np.median(ef)), "ell_lead_median": float(np.median(el)),
            "frac_follow_shorter": float(np.mean(ef < el)),
        }

    # ---- S4: 集合重叠 ----
    idxW = topk_idx(W, STRENGTH_KEEP)
    idxLead = topk_idx(MT, STRENGTH_KEEP)
    idxFollow = topk_idx(M, STRENGTH_KEEP)
    jac_wl = np.full(n, np.nan)
    jac_lf = np.full(n, np.nan)
    for i in range(n):
        sw = set(idxW[i].tolist()) - {i}
        sl = set(idxLead[i].tolist()) - {i}
        sf = set(idxFollow[i].tolist()) - {i}
        if sw and sl:
            jac_wl[i] = len(sw & sl) / len(sw | sl)
        if sl and sf:
            jac_lf[i] = len(sl & sf) / len(sl | sf)
    log(f"{date} S4 完成 jaccard(W,lead) 中位 {np.nanmedian(jac_wl):.3f}")

    # ---- S3: 同向 / 对盘 2×2 ----
    same = read_mat(date, "flow_same_t1")[:n, :n]
    opp = read_mat(date, "flow_opp_t1")[:n, :n]
    for mat in (same, opp):
        np.fill_diagonal(mat, 0.0)
    s_lead = same.sum(axis=0)      # Σ_j S_same[j][i] = 同向关系里我领先
    s_follow = same.sum(axis=1)
    o_lead = opp.sum(axis=0)
    o_follow = opp.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        same_lead_share = np.where(s_lead + s_follow > 0,
                                   s_lead / (s_lead + s_follow), np.nan)
        opp_lead_share = np.where(o_lead + o_follow > 0,
                                  o_lead / (o_lead + o_follow), np.nan)
        same_share_of_total = np.where((s_lead + s_follow + o_lead + o_follow) > 0,
                                       (s_lead + s_follow) /
                                       (s_lead + s_follow + o_lead + o_follow), np.nan)
    del same, opp
    gc.collect()

    # ---- 对照口径（沿用上一轮缓存，cnt_t1 口径）----
    cpath = f"{CACHE}/{date}_cnt_t1_v2.npz"
    z = np.load(cpath, allow_pickle=True)
    assert [str(c) for c in z["codes"]][:n] == codes, "缓存 codes 与本次不一致"
    ref = {k: np.asarray(z[v], dtype=np.float64)[:n] for k, v in REF_STATS.items()}
    ref_strength = np.asarray(z["strength"], dtype=np.float64)[:n]

    out = {
        "codes": codes,
        "strength": strength,
        "n_trades": n_trades,
        "ref": ref,
        "stats": {
            "ts_follow": ts_follow, "ts_lead": ts_lead, "ts_ratio": ts_ratio,
            "ts_follow_small": ts_follow_s, "ts_lead_small": ts_lead_s,
            "ts_ratio_small": ts_ratio_s,
            "tau_consistency": tau_consistency,
            "same_lead_share": same_lead_share, "opp_lead_share": opp_lead_share,
            "same_share_of_total": same_share_of_total,
            "jaccard_w_lead": jac_wl, "jaccard_lead_follow": jac_lf,
        },
        "diag": {
            "r2_median_full": float(np.nanmedian(r2F[np.isfinite(r2F)])),
            "r2_median_full_lead": float(np.nanmedian(r2L[np.isfinite(r2L)])),
            "r2_median_small": float(np.nanmedian(r2Fs[np.isfinite(r2Fs)])),
            "n_pairs_fit": int(np.isfinite(ellF).sum()),
            "ell_median_full": float(np.nanmedian(ellF[np.isfinite(ellF)])),
            "ell_median_small": float(np.nanmedian(ellFs[np.isfinite(ellFs)])),
            "strength_ref_max_absdiff": float(np.nanmax(np.abs(strength - ref_strength))),
            "orientation_check": {
                "spearman_q_vs_ell_follow": r_ellF,
                "spearman_q_vs_ell_lead": r_ellL,
                "spearman_ell_follow_vs_ell_lead": r_ellFL,
                "by_group": grp,
                "expect": "q_ij 是“i 领先 j”的份额；j 领先 i 越紧，ℓ(i 落后 j) 越小、ℓ(j 落后 i) 越大，"
                          "所以预期前一项为正、后一项为负。符号不符说明口径读反了。",
            },
        },
    }
    del M, MT, W, idx, Wsel, valid
    gc.collect()
    log(f"{date} 全部统计量完成（总 {time.time()-t0:.1f}s）")
    return out


def sp(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 30:
        return None
    r = spearmanr(a[m], b[m]).statistic
    return None if not np.isfinite(r) else float(r)


def _rank(v):
    v = np.asarray(v, dtype=np.float64)
    out = np.full(v.shape, np.nan)
    m = np.isfinite(v)
    out[m] = rankdata(v[m])
    return out


def sp_neutral(a, b, ctrl_a, ctrl_b):
    """各自对控制变量（秩）回归取残差后的跨日秩相关。"""
    ra, rb = _rank(a), _rank(b)
    ca = [_rank(c) for c in ctrl_a]
    cb = [_rank(c) for c in ctrl_b]
    m = np.isfinite(ra) & np.isfinite(rb)
    for c in ca + cb:
        m &= np.isfinite(c)
    if m.sum() < 30:
        return None

    def resid(y, ctrls):
        Xm = np.column_stack([np.ones(int(m.sum()))] + [c[m] for c in ctrls])
        beta, *_ = np.linalg.lstsq(Xm, y, rcond=None)
        return y - Xm @ beta

    ea, eb = resid(ra[m], ca), resid(rb[m], cb)
    r = spearmanr(ea, eb).statistic
    return None if not np.isfinite(r) else float(r)


def dist(v):
    v = np.asarray(v, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {}
    q1, q3 = np.percentile(v, [25, 75])
    return {"n": int(v.size), "median": float(np.median(v)), "iqr": float(q3 - q1),
            "std": float(v.std(ddof=1)) if v.size > 1 else None,
            "mean": float(v.mean()),
            "frac_nonzero": float(np.count_nonzero(v) / v.size)}


def main():
    smoke = "--smoke" in sys.argv
    log(f"开始 probe3（smoke={smoke}）")
    A = day_stats(DATE_A, smoke)
    B = day_stats(DATE_B, smoke)

    common = sorted(set(A["codes"]) & set(B["codes"]))
    ia = {c: i for i, c in enumerate(A["codes"])}
    ib = {c: i for i, c in enumerate(B["codes"])}
    a_idx = np.array([ia[c] for c in common])
    b_idx = np.array([ib[c] for c in common])
    log(f"两天代码交集 {len(common)} 只")

    all_stats = NEW_STATS + ["ref_lead_ratio", "ref_lead_breadth", "ref_q_p90"]
    summary = {}
    rank_rows = []
    for s in all_stats:
        if s.startswith("ref_"):
            ka = s[4:]
            va = A["ref"][ka][a_idx]
            vb = B["ref"][ka][b_idx]
        else:
            va = A["stats"][s][a_idx]
            vb = B["stats"][s][b_idx]
        r = sp(va, vb)
        rn = sp_neutral(va, vb, [A["strength"][a_idx], A["n_trades"][a_idx]],
                        [B["strength"][b_idx], B["n_trades"][b_idx]])
        rn_size = sp_neutral(va, vb, [A["strength"][a_idx]], [B["strength"][b_idx]])
        rho_str = max([abs(x) for x in (sp(va, A["strength"][a_idx]), sp(vb, B["strength"][b_idx])) if x is not None] or [np.nan])
        rho_ntr = max([abs(x) for x in (sp(va, A["n_trades"][a_idx]), sp(vb, B["n_trades"][b_idx])) if x is not None] or [np.nan])
        rho_ref = {}
        for ref_name in REF_STATS:
            xs = [sp(va, A["ref"][ref_name][a_idx]), sp(vb, B["ref"][ref_name][b_idx])]
            xs = [abs(x) for x in xs if x is not None]
            rho_ref[ref_name] = max(xs) if xs else None
        summary[s] = {
            "cross_day_spearman": r,
            "cross_day_spearman_size_neutral": rn_size,
            "cross_day_spearman_double_neutral": rn,
            "max_abs_rho_with_strength": rho_str,
            "max_abs_rho_with_ntrades": rho_ntr,
            "max_abs_rho_with_ref": rho_ref,
            "dist_A": dist(va), "dist_B": dist(vb),
        }
        rank_rows.append((s, r, rn_size, rn, rho_str, rho_ref))
        log(f"  {s:22s} 跨日={r if r is None else round(r,3)} 去规模={rn_size if rn_size is None else round(rn_size,3)}")

    stat_rank = sorted([r for r in rank_rows if r[1] is not None], key=lambda x: -x[1])

    # ---- 同一天内部的相互相关（判断新口径之间是不是同一个东西）----
    def within(a, b):
        return sp(A["stats"][a][a_idx], A["stats"][b][a_idx])
    pairs = [("ts_follow", "ts_lead"), ("ts_follow", "ts_follow_small"),
             ("ts_lead", "ts_lead_small"), ("ts_follow", "ts_ratio"),
             ("same_lead_share", "opp_lead_share"),
             ("same_lead_share", "same_share_of_total"),
             ("jaccard_w_lead", "jaccard_lead_follow"),
             ("jaccard_lead_follow", "ts_ratio"),
             ("tau_consistency", "ts_ratio")]
    within_day = {f"{a}~{b}": within(a, b) for a, b in pairs}
    log("同天内部相关: " + ", ".join(f"{k}={None if v is None else round(v,3)}"
                                    for k, v in within_day.items()))
    doc = {
        "task": "task-7 方向的时间域与集合结构降维",
        "scope": {
            "dates": [DATE_A, DATE_B],
            "matrices": {"S1_S2_S4": TAU_NAMES + ["cnt_t1(参考 W)"],
                         "S3": ["flow_same_t1", "flow_opp_t1"]},
            "taus_sec": TAU_SEC.tolist(),
            "top_k_partners": K,
            "set_size_S4": STRENGTH_KEEP,
            "smoke": smoke,
            "definitions": {
                "ell_ij": "对 x=1/τ、y=log S[i][j](τ) 做最小二乘，ℓ_ij = −slope，含义是 j 领先 i 时 i 的平均响应滞后（秒）",
                "ts_follow": "Σ_j W_ij·ℓ_ij / Σ_j W_ij，W 前 200 伙伴",
                "ts_lead": "Σ_j W_ij·ℓ_ji / Σ_j W_ij",
                "ts_ratio": "ts_follow / ts_lead",
                "tau_consistency": "9 个 τ 上 q>0.5 的符号，与多数符号一致的比例，伙伴等权",
                "same_lead_share": "Σ_j S_same[j][i] / Σ_j (S_same[i][j]+S_same[j][i])",
                "jaccard_w_lead": "互动集合（W 前 50）与领先集合（L 前 50）的 Jaccard",
                "jaccard_lead_follow": "领先集合与跟随集合（S 行前 50）的 Jaccard",
            },
        },
        "coverage": {
            "n_codes": {"A": len(A["codes"]), "B": len(B["codes"]), "common": len(common)},
            "diag_A": A["diag"], "diag_B": B["diag"],
        },
        "stat_summary": summary,
        "cross_day_rank": [
            {"stat": s, "cross_day": r, "size_neutral": rn1, "double_neutral": rn2,
             "max_abs_rho_strength": rs, "max_abs_rho_ref": rr}
            for (s, r, rn1, rn2, rs, rr) in stat_rank
        ],
        "within_day_corr_A": within_day,
        "orientation_check": {"A": A["diag"]["orientation_check"],
                              "B": B["diag"]["orientation_check"]},
        "caveats": [
            "同桶折半项：同一 10 毫秒桶内的配对在核里按 e^0=1 计（再乘 0.5），相当于 δ=0，"
            "会把 ℓ 往小拉。τ=0.05 时同桶项占该对能量的 11%~14%（上一轮实测），所以 ℓ 是平均滞后的下界。",
            "桶离散化：衰减按整桶数推进，δ 实际被量化到 10 毫秒的整数倍，ℓ 的分辨率是 10 毫秒。",
            "线性假设：S(τ)=A·E[e^{−δ/τ}] 只在 δ 分布较窄时近似等于 A·e^{−ℓ/τ}；"
            "大 τ 端被总数饱和、小 τ 端被同桶地板顶住，两端都会把拟合压弯。看 R² 与只用 0.05~1 的对照子集。",
            "S3 的同向矩阵把“一起买”和“一起卖”相加在一起（flow 的 P 场与 M 场各自非负），"
            "所以 same_lead_share 说的是“同向成交里谁先动”，分不出买侧还是卖侧；对盘矩阵同理，"
            "两侧都是幅度，不含正负号。",
            "只有 20241230/20241231 两天矩阵，跨日秩相关由这一对相邻交易日支撑。",
        ],
    }
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON + (".smoke" if smoke else ""), "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=1)
    log(f"写出 {OUT_JSON}{'.smoke' if smoke else ''}")
    return doc


if __name__ == "__main__":
    main()
