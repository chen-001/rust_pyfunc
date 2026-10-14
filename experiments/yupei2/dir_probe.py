"""方向性降维候选统计量的机制样本探测（task-4）。

口径（已实测确认，不按源码注释读）：
    S[i][j] = "列 j 领先、行 i 跟随"的衰减交互能量（10ms 桶）
    L[i][j] = S[j][i]                        = "i 领先 j"
    W[i][j] = S[i][j] + S[j][i]              = 对称强度（与方向无关）
    q[i][j] = L/W = S[j][i]/(S[i][j]+S[j][i]) ∈ [0,1]，q=0.5 表示这一对没有方向

对 cnt_t005 / cnt_t1 / cnt_t30 / vol_t1 四个矩阵，逐股票算 8 个候选统计量
（A~H），并给出截面分布、与强度/笔数的秩相关、两天交集上的跨日秩相关。

只读数据。矩阵逐个读、算完释放，避免同时持有多个 3964×3964 的 float64。
中间结果缓存到 /tmp/yupei2_probe_cache，重跑不重复读大矩阵。
"""
import gc
import json
import os

import numpy as np
import rust_pyfunc as rp
from scipy.stats import spearmanr

BACKUP = "/hdd/user_home_unsafe/chenzongwei/yupei_dist_backup"
DATE_A, DATE_B = 20241230, 20241231
MATS = ["cnt_t005", "cnt_t1", "cnt_t30", "vol_t1"]
TOP_K = 20
EPS = 1e-6
CACHE = "/tmp/yupei2_probe_cache"
OUT_JSON = "/home/chenzongwei/rust_pyfunc/experiments/yupei2_dir_reduce.probe.json"
OUT_JSON2 = "/home/chenzongwei/rust_pyfunc/experiments/yupei2_dir_reduce.probe2.json"

STATS = [
    "q_top20_equal",
    "q_top20_wmean",
    "q_all_wmean",
    "lead_breadth",
    "net_lead_raw",
    "logit_top20",
    "strong_weak_gap",
    "top20_lead_share",
]
# 每个统计量在"没有方向"时的取值，用于算偏离零假设的比例
NULL_VALUE = {
    "q_top20_equal": 0.5,
    "q_top20_wmean": 0.5,
    "q_all_wmean": 0.5,
    "lead_breadth": 0.5,
    "net_lead_raw": 0.0,
    "logit_top20": 0.0,
    "strong_weak_gap": 0.0,
    "top20_lead_share": 0.0,
}

# ---- task-6 追加：软权重曲线与更稳的聚合口径 ----
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]


def alpha_key(a):
    return "soft_a" + ("%g" % a).replace(".", "")


ALPHA_STATS = [alpha_key(a) for a in ALPHAS]
SWEEP_STATS = ALPHA_STATS + [
    "q_top50pct_equal",
    "sign_z",
    "q_p75",
    "q_p90",
]
NULL_VALUE.update({alpha_key(a): 0.5 for a in ALPHAS})
NULL_VALUE.update({
    "q_top50pct_equal": 0.5,
    "sign_z": 0.0,
    "q_p75": 0.5,
    "q_p90": 0.5,
})
ALL_STATS = STATS + SWEEP_STATS


def read_codes(date):
    p = f"{BACKUP}/{date}/codes.txt"
    return [l.strip() for l in open(p) if l.strip()]


def compute_day(date, mat):
    """返回 dict(codes, stats, strength, n_trades)；带 /tmp 缓存。

    缓存文件名带版本后缀：task-4 那版（task4）只存 8 个统计量，task-6 追加软权重曲线
    与分位数口径后升到 v2。旧缓存无法推出新统计量（要 W^alpha 与 q 的分位数），
    所以 v2 会重读一次矩阵，之后一直命中缓存。
    """
    os.makedirs(CACHE, exist_ok=True)
    cache = f"{CACHE}/{date}_{mat}_v2.npz"
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        return {
            "codes": [str(c) for c in z["codes"]],
            "stats": {k: z["st_" + k] for k in ALL_STATS},
            "strength": z["strength"],
            "n_trades": z["n_trades"],
            "from_cache": True,
        }

    codes = read_codes(date)
    n = len(codes)
    S = np.ascontiguousarray(rp.yupei_dist_backup_read_matrix(BACKUP, date, mat), dtype=np.float64)
    assert S.shape == (n, n), f"{date} {mat} 矩阵形状 {S.shape} 与 codes {n} 不符"
    ST = np.ascontiguousarray(S.T)
    gc.collect()

    st_codes, stats_bin = rp.yupei_dist_backup_read_stats(BACKUP, date)
    assert list(st_codes) == codes, f"{date} stats 行序与 codes.txt 不一致"
    n_trades = np.asarray(stats_bin[:, 0], dtype=np.float64)

    res = {k: np.full(n, np.nan) for k in ALL_STATS}
    strength = np.empty(n)
    w_tmp = np.empty(n)
    q_tmp = np.empty(n)


    for i in range(n):
        s = S[i]
        st = ST[i]
        np.add(s, st, out=w_tmp)
        w_tmp[i] = 0.0                       # 自己不算伙伴
        n_pos = int(np.count_nonzero(w_tmp > 0.0))
        strength[i] = w_tmp.sum()

        if n_pos == 0:
            continue
        np.divide(st, w_tmp, out=q_tmp, where=w_tmp > 0.0)
        q_tmp[w_tmp <= 0.0] = 0.5            # 没有联动 = 没有方向

        # 前 K 个伙伴按 W 取（不含自己）
        k = TOP_K if n_pos >= TOP_K else n_pos
        idx = np.argpartition(w_tmp, -k)[-k:]
        qk = q_tmp[idx]
        wk = w_tmp[idx]

        res["q_top20_equal"][i] = qk.mean()
        wsum = wk.sum()
        res["q_top20_wmean"][i] = float((wk * qk).sum() / wsum) if wsum > 0 else np.nan
        # C: 全部伙伴（排除自己，对角是自配对能量，不能算作伙伴）的 W 加权 q
        st_sum_excl = float(st.sum() - st[i])
        res["q_all_wmean"][i] = st_sum_excl / strength[i] if strength[i] > 0 else np.nan
        res["lead_breadth"][i] = float(np.count_nonzero(q_tmp > 0.5)) / (n - 1)
        res["net_lead_raw"][i] = float(st.sum() - s.sum())
        qc = np.clip(qk, EPS, 1.0 - EPS)
        res["logit_top20"][i] = float(np.log(qc / (1.0 - qc)).mean())
        res["top20_lead_share"][i] = float(np.count_nonzero(qk > 0.6)) / k

        # ---- task-6: 只在 W>0 的伙伴上做的聚合 ----
        vidx = np.flatnonzero(w_tmp > 0.0)
        nv = vidx.size
        if nv >= 10:
            wv = w_tmp[vidx]
            qv = q_tmp[vidx]
            # 软权重曲线 Σ W^alpha q / Σ W^alpha
            for a in ALPHAS:
                wa = wv if a == 1.0 else (np.ones(nv) if a == 0.0 else wv ** a)
                res[alpha_key(a)][i] = float((wa * qv).sum() / wa.sum())
            # 半截断: W 排名前 50% 等权
            k50 = max(1, nv // 2)
            part50 = np.argpartition(wv, nv - k50)
            res["q_top50pct_equal"][i] = float(qv[part50[nv - k50:]].mean())
            # 符号检验 z
            n_lead = int(np.count_nonzero(qv > 0.5))
            n_follow = int(np.count_nonzero(qv < 0.5))
            if n_lead + n_follow > 0:
                res["sign_z"][i] = float((n_lead - n_follow) / np.sqrt(n_lead + n_follow))
            # 分位数口径
            res["q_p75"][i] = float(np.percentile(qv, 75))
            res["q_p90"][i] = float(np.percentile(qv, 90))
            # 强伙伴 vs 弱伙伴的领先份额差（各取 20%）
            kk = max(1, nv // 5)
            part = np.argpartition(wv, nv - kk)
            hi = vidx[part[nv - kk:]]
            lo = vidx[part[:kk]]
            res["strong_weak_gap"][i] = float(q_tmp[hi].mean() - q_tmp[lo].mean())

    del S, ST
    gc.collect()

    np.savez_compressed(
        cache,
        codes=np.array(codes),
        strength=strength,
        n_trades=n_trades,
        **{"st_" + k: res[k] for k in ALL_STATS},
    )
    return {"codes": codes, "stats": res, "strength": strength, "n_trades": n_trades,
            "from_cache": False}


def sp(a, b):
    """秩相关，自动剔除任一侧非有限值；样本不足 30 返回 None。"""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 30:
        return None
    r = spearmanr(a[m], b[m]).statistic
    return None if not np.isfinite(r) else float(r)


def _rank(v):
    from scipy.stats import rankdata
    v = np.asarray(v, dtype=np.float64)
    out = np.full(v.shape, np.nan)
    m = np.isfinite(v)
    out[m] = rankdata(v[m])
    return out


def sp_size_neutral(a, b, size_a, size_b, extra_a=None, extra_b=None):
    """先各自对当日规模（对称强度）的秩做线性回归取残差，再算残差的跨日秩相关。
    用来判断某个口径的跨日稳定是不是只来自"规模本身很稳"。
    extra_a/extra_b 给定时一并当作控制变量（例如当日笔数的秩）。"""
    ra, rb = _rank(a), _rank(b)
    ctrl_a = [_rank(size_a)]
    ctrl_b = [_rank(size_b)]
    if extra_a is not None:
        ctrl_a.append(_rank(extra_a))
        ctrl_b.append(_rank(extra_b))
    m = np.isfinite(ra) & np.isfinite(rb)
    for c in ctrl_a + ctrl_b:
        m &= np.isfinite(c)
    if m.sum() < 30:
        return None
    def resid(y, ctrls):
        X = np.column_stack([np.ones(m.sum())] + [c[m] for c in ctrls])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        return y - X @ beta
    ea = resid(ra[m], ctrl_a)
    eb = resid(rb[m], ctrl_b)
    r = spearmanr(ea, eb).statistic
    return None if not np.isfinite(r) else float(r)


def _dist(v):
    v = np.asarray(v, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {}
    q1, q3 = np.percentile(v, [25, 75])
    return {
        "n": int(v.size),
        "median": float(np.median(v)),
        "iqr": float(q3 - q1),
        "std": float(v.std(ddof=1)) if v.size > 1 else None,
        "mean": float(v.mean()),
    }


def build_probe2(days):
    """task-6: 软权重曲线 + 半截断 + 符号 z + 分位数口径。"""
    ref_stats = ["q_top20_equal", "q_top50pct_equal", "sign_z", "q_p75", "q_p90",
                 "q_all_wmean", "lead_breadth", "net_lead_raw"] + ALPHA_STATS
    out = {
        "task": "task-6 权重曲线与更稳聚合口径的探测",
        "scope": {
            "dates": [DATE_A, DATE_B],
            "matrices": MATS,
            "alphas": ALPHAS,
            "definitions": {
                "soft_alpha": "Σ_j W^alpha·q / Σ_j W^alpha，j 只取 W>0 且非自己的伙伴；"
                              "alpha=0 为等权，alpha=1 与现有 net_lead/strength（lead_ratio）等价",
                "q_top50pct_equal": "W 排名前 50% 的伙伴上 q 的等权平均",
                "sign_z": "(n_lead − n_follow)/sqrt(n_lead+n_follow)，n_lead=#{q>0.5}, n_follow=#{q<0.5}",
                "q_p75_q_p90": "W>0 伙伴上 q 的 75/90 分位",
                "lead_ratio_ref": "本次脚本自己实现的 soft_alpha=1（= 净领先/强度），不是生产 parquet",
                "lead_breadth_ref": "本次脚本自己实现的 #(q>0.5)/(N−1)",
            },
        },
        "alpha_curve": {},
        "alpha_peaks": {},
        "stat_summary": {},
        "stat_ranking_mean_over_matrices": [],
        "existing_overlap": {},
        "conclusion": {},
    }

    matrix_ctx = {}
    for mat in MATS:
        ca = days[(DATE_A, mat)]["codes"]
        cb = days[(DATE_B, mat)]["codes"]
        common = sorted(set(ca) & set(cb))
        ia = {c: i for i, c in enumerate(ca)}
        ib = {c: i for i, c in enumerate(cb)}
        matrix_ctx[mat] = (common, np.array([ia[c] for c in common]),
                           np.array([ib[c] for c in common]))

    for mat in MATS:
        common, idxa, idxb = matrix_ctx[mat]
        stA, stB = days[(DATE_A, mat)]["stats"], days[(DATE_B, mat)]["stats"]
        strA, strB = days[(DATE_A, mat)]["strength"], days[(DATE_B, mat)]["strength"]
        ntrA, ntrB = days[(DATE_A, mat)]["n_trades"], days[(DATE_B, mat)]["n_trades"]
        lrA, lrB = stA["soft_a1"], stB["soft_a1"]          # 现有 lead_ratio 口径
        brA, brB = stA["lead_breadth"], stB["lead_breadth"]

        curve = []
        for a in ALPHAS:
            k = alpha_key(a)
            rho = sp(stA[k][idxa], stB[k][idxb])
            curve.append({
                "alpha": a,
                "stat": k,
                "cross_day_spearman": rho,
                "spearman_with_strength_A": sp(strA, stA[k]),
                "spearman_with_ntrades_A": sp(ntrA, stA[k]),
                "spearman_with_lead_ratio_A": sp(lrA, stA[k]),
                "spearman_with_lead_breadth_A": sp(brA, stA[k]),
                **_dist(stA[k]),
            })
        out["alpha_curve"][mat] = curve
        best = max(curve, key=lambda r: (r["cross_day_spearman"] is not None,
                                         r["cross_day_spearman"] or -9))
        out["alpha_peaks"][mat] = {
            "peak_alpha": best["alpha"],
            "peak_cross_day_spearman": best["cross_day_spearman"],
            "curve_cross_day": {("%g" % r["alpha"]): r["cross_day_spearman"] for r in curve},
            "monotone_note": "跨日秩相关随 alpha 的变化见 curve_cross_day",
        }

        for st in ref_stats:
            out["stat_summary"][f"{mat}|{st}"] = {
                "matrix": mat, "stat": st,
                "cross_day_spearman": sp(stA[st][idxa], stB[st][idxb]),
                "cross_day_spearman_size_neutralized": sp_size_neutral(
                    stA[st][idxa], stB[st][idxb], strA[idxa], strB[idxb]),
                "cross_day_spearman_size_ntrades_neutralized": sp_size_neutral(
                    stA[st][idxa], stB[st][idxb], strA[idxa], strB[idxb],
                    extra_a=ntrA[idxa], extra_b=ntrB[idxb]),
                "spearman_with_strength_A": sp(strA, stA[st]),
                "spearman_with_strength_B": sp(strB, stB[st]),
                "spearman_with_ntrades_A": sp(ntrA, stA[st]),
                "spearman_with_ntrades_B": sp(ntrB, stB[st]),
                "spearman_with_lead_ratio_A": sp(lrA, stA[st]),
                "spearman_with_lead_ratio_B": sp(lrB, stB[st]),
                "spearman_with_lead_breadth_A": sp(brA, stA[st]),
                "spearman_with_lead_breadth_B": sp(brB, stB[st]),
                "distribution_A": _dist(stA[st]),
                "distribution_B": _dist(stB[st]),
            }

    # ---- 跨矩阵汇总与结论 ----
    by_stat = {}
    for key, v in out["stat_summary"].items():
        by_stat.setdefault(v["stat"], []).append(v)
    rows = []
    for st, vs in by_stat.items():
        cd = [v["cross_day_spearman"] for v in vs if v["cross_day_spearman"] is not None]
        cdn = [v["cross_day_spearman_size_neutralized"] for v in vs
               if v["cross_day_spearman_size_neutralized"] is not None]
        cdn2 = [v["cross_day_spearman_size_ntrades_neutralized"] for v in vs
                if v["cross_day_spearman_size_ntrades_neutralized"] is not None]
        size = [abs(v["spearman_with_strength_A"] or 0) for v in vs] + \
               [abs(v["spearman_with_ntrades_A"] or 0) for v in vs]
        dup = [abs(v["spearman_with_lead_ratio_A"] or 0) for v in vs] + \
              [abs(v["spearman_with_lead_breadth_A"] or 0) for v in vs]
        rows.append({
            "stat": st,
            "mean_cross_day_spearman": float(np.mean(cd)) if cd else None,
            "mean_cross_day_spearman_size_neutralized": float(np.mean(cdn)) if cdn else None,
            "mean_cross_day_spearman_double_neutralized": float(np.mean(cdn2)) if cdn2 else None,
            "min_cross_day": float(np.min(cd)) if cd else None,
            "max_cross_day": float(np.max(cd)) if cd else None,
            "max_abs_rho_with_size": float(np.max(size)) if size else None,
            "max_abs_rho_with_existing": float(np.max(dup)) if dup else None,
            "per_matrix_cross_day": {v["matrix"]: v["cross_day_spearman"] for v in vs},
            "per_matrix_cross_day_neutralized": {
                v["matrix"]: v["cross_day_spearman_size_neutralized"] for v in vs},
        })
    rows.sort(key=lambda r: -(r["mean_cross_day_spearman"] or -9))
    out["stat_ranking_mean_over_matrices"] = rows

    def passes(r, cd_min=0.35, size_max=0.35, dup_max=0.90):
        return (r["mean_cross_day_spearman"] is not None
                and r["mean_cross_day_spearman"] >= cd_min
                and r["max_abs_rho_with_size"] <= size_max
                and r["max_abs_rho_with_existing"] <= dup_max)

    ok = [r for r in rows if passes(r)]
    ok_neutral = [r for r in rows
                  if (r["mean_cross_day_spearman_size_neutralized"] or -9) >= 0.25
                  and r["max_abs_rho_with_existing"] <= 0.90]
    out["conclusion"] = {
        "rule": "三条判据：四矩阵平均跨日秩相关 ≥ 0.35；与强度/笔数的|秩相关| ≤ 0.35；"
                "与现有 lead_ratio / lead_breadth 的|秩相关| ≤ 0.90。取满足三条中跨日最高者。",
        "passing": [r["stat"] for r in ok],
        "best": ok[0]["stat"] if ok else None,
        "best_detail": ok[0] if ok else None,
        "relaxed_rule": "把第一条换成「去掉规模成分后的跨日秩相关 ≥ 0.25」且第二条不看"
                        "（因为去掉规模后本来就不再看规模相关），仍要求与现有口径 ≤ 0.90。",
        "passing_relaxed": [r["stat"] for r in ok_neutral],
        "best_relaxed": ok_neutral[0]["stat"] if ok_neutral else None,
        "best_relaxed_detail": ok_neutral[0] if ok_neutral else None,
        "ranking": [{"stat": r["stat"], "mean_cross_day": r["mean_cross_day_spearman"],
                     "mean_cross_day_size_neutralized":
                         r["mean_cross_day_spearman_size_neutralized"],
                     "max_abs_rho_size": r["max_abs_rho_with_size"],
                     "max_abs_rho_existing": r["max_abs_rho_with_existing"]} for r in rows],
        "if_none_passes": "没有口径同时满足三条时，说明该矩阵体系里没有既稳又新的方向口径，"
                          "下一步应该换构造（换权重设计、换 τ 组合、换统计量族）而不是换聚合权重。",
    }
    out["notes"] = {
        "why_reread": "task-6 的软权重曲线与分位数口径需要 W^alpha 与 q 的分位数，"
                      "task-4 的缓存里没有这些量，无法从缓存推出，所以升到 _v2 缓存并重读一次矩阵，"
                      "之后所有重跑都命中缓存。",
        "diagonal": "所有口径都把对角（自配对能量）排除在伙伴之外。",
        "convention": "S[i][j]=列领先行，所以 q=S[j][i]/(S[i][j]+S[j][i]) 是 i 领先 j 的份额。",
        "limitation": "只有 20241230/20241231 两天，跨日秩相关由这一对相邻交易日支撑，"
                      "是机制样本层面的结论。",
    }
    return out


def main():
    days = {}
    for date in (DATE_A, DATE_B):
        for mat in MATS:
            days[(date, mat)] = compute_day(date, mat)
            print(f"算完 {date} {mat} n={len(days[(date, mat)]['codes'])} "
                  f"cache={days[(date, mat)]['from_cache']}", flush=True)

    # 对照：现有 dir_cnt_t1_net_lead / lead_cnt_t1_breadth（同一天，自己实现）
    ref = {}
    for date in (DATE_A, DATE_B):
        d = days[(date, "cnt_t1")]
        ref[date] = {
            "dir_cnt_t1_net_lead": d["stats"]["net_lead_raw"],
            "lead_cnt_t1_breadth": d["stats"]["lead_breadth"],
        }

    out = {
        "task": "task-4 方向性降维候选统计量机制样本探测",
        "scope": {
            "dates": [DATE_A, DATE_B],
            "matrices": MATS,
            "convention": "S[i][j] = 列 j 领先、行 i 跟随; L[i][j]=S[j][i]; "
                          "W=S+S^T; q=L/W ∈[0,1], 0.5=无方向",
            "top_k": TOP_K,
            "matrix_dir": BACKUP,
            "cache_dir": CACHE,
        },
        "coverage": {},
        "cross_day_stability_rank": [],
        "per_stat": {},
        "code_intersection": {},
    }

    for mat in MATS:
        ca = days[(DATE_A, mat)]["codes"]
        cb = days[(DATE_B, mat)]["codes"]
        common = sorted(set(ca) & set(cb))
        ia = {c: i for i, c in enumerate(ca)}
        ib = {c: i for i, c in enumerate(cb)}
        idxa = np.array([ia[c] for c in common])
        idxb = np.array([ib[c] for c in common])
        out["code_intersection"][mat] = {
            "n_codes_A": len(ca), "n_codes_B": len(cb), "n_common": len(common),
            "only_A": len(set(ca) - set(cb)), "only_B": len(set(cb) - set(ca)),
        }
        out["coverage"][mat] = {"n_codes_per_day": {"A": len(ca), "B": len(cb)},
                                "n_common": len(common)}
        # 参照系: 对称强度与当日笔数本身的跨日秩相关（判断 0.5 级别的稳定性算好还是差）
        out["coverage"][mat]["reference_cross_day"] = {
            "strength_sum_W": sp(days[(DATE_A, mat)]["strength"][idxa],
                                 days[(DATE_B, mat)]["strength"][idxb]),
            "n_trades": sp(days[(DATE_A, mat)]["n_trades"][idxa],
                           days[(DATE_B, mat)]["n_trades"][idxb]),
        }

        for stat in STATS:
            va = days[(DATE_A, mat)]["stats"][stat]
            vb = days[(DATE_B, mat)]["stats"][stat]
            x = va[idxa]
            y = vb[idxb]
            finite = np.isfinite(x) & np.isfinite(y)
            rho = sp(x, y)

            row = {
                "matrix": mat, "stat": stat,
                "cross_day_spearman": rho,
                "n_common_total": int(len(common)),
                "n_finite_both": int(finite.sum()),
                "median_A": float(np.nanmedian(x)) if finite.sum() else None,
                "median_B": float(np.nanmedian(y)) if finite.sum() else None,
            }
            out["cross_day_stability_rank"].append(row)

            # 分布用两天合并（两天同一天量纲一致，各自算完取平均更稳）
            def dist(v):
                v = v[np.isfinite(v)]
                if v.size == 0:
                    return {}
                q1, q3 = np.percentile(v, [25, 75])
                return {
                    "n": int(v.size),
                    "median": float(np.median(v)),
                    "iqr": float(q3 - q1),
                    "std": float(v.std(ddof=1)) if v.size > 1 else None,
                    "mean": float(v.mean()),
                    "frac_nonzero": float(np.count_nonzero(v) / v.size),
                    "frac_neq_null": float(
                        np.count_nonzero(np.abs(v - NULL_VALUE[stat]) > 1e-12) / v.size
                    ),
                    "null_value": NULL_VALUE[stat],
                }

            per = {
                "matrix": mat,
                "distribution_A": dist(va),
                "distribution_B": dist(vb),
                "spearman_with_strength_A": sp(days[(DATE_A, mat)]["strength"], va),
                "spearman_with_strength_B": sp(days[(DATE_B, mat)]["strength"], vb),
                "spearman_with_ntrades_A": sp(days[(DATE_A, mat)]["n_trades"], va),
                "spearman_with_ntrades_B": sp(days[(DATE_B, mat)]["n_trades"], vb),
                "spearman_with_dir_cnt_t1_net_lead_A": sp(
                    ref[DATE_A]["dir_cnt_t1_net_lead"], va),
                "spearman_with_dir_cnt_t1_net_lead_B": sp(
                    ref[DATE_B]["dir_cnt_t1_net_lead"], vb),
                "spearman_with_lead_cnt_t1_breadth_A": sp(
                    ref[DATE_A]["lead_cnt_t1_breadth"], va),
                "spearman_with_lead_cnt_t1_breadth_B": sp(
                    ref[DATE_B]["lead_cnt_t1_breadth"], vb),
            }
            out["per_stat"][f"{mat}|{stat}"] = per

    out["cross_day_stability_rank"].sort(
        key=lambda r: (r["cross_day_spearman"] is None, -(r["cross_day_spearman"] or 0))
    )
    # 每个统计量在四个矩阵上的跨日稳定性平均，按它排名
    by_stat = {}
    for r in out["cross_day_stability_rank"]:
        by_stat.setdefault(r["stat"], []).append(r["cross_day_spearman"])
    out["stat_ranking_mean_over_matrices"] = sorted(
        [
            {
                "stat": k,
                "mean_cross_day_spearman": float(np.mean([v for v in vs if v is not None])),
                "min": float(np.nanmin(vs)),
                "max": float(np.nanmax(vs)),
                "per_matrix": {r["matrix"]: r["cross_day_spearman"]
                               for r in out["cross_day_stability_rank"] if r["stat"] == k},
            }
            for k, vs in by_stat.items()
            if any(v is not None for v in vs)
        ],
        key=lambda d: -d["mean_cross_day_spearman"],
    )
    out["notes"] = {
        "cross_day_reading": "跨日秩相关是本次最关键的数。同一天的两个统计量相关再高，跨日掉到 0 就说明方向是噪声。",
        "reference_frame": "coverage.<matrix>.reference_cross_day 给了对称强度 ΣW 与当日笔数自己的跨日秩相关，"
                           "用来说明 0.4~0.5 这个水平在当天数据里算好还是差。",
        "net_lead_raw_scale": "E 是原始量纲（量级可达 1e7 以上），跨日秩相关不受量纲影响。",
        "algebra": "C = ΣL/ΣW，E = ΣL − ΣS，ΣW = ΣS + ΣL，所以 C = 0.5 + E/(2·ΣW)。"
                   "C 与 E 不是单调关系，差别全在是否用 ΣW 归一化。",
        "comparison_factors": "spearman_with_dir_cnt_t1_net_lead / spearman_with_lead_cnt_t1_breadth "
                              "用的是本次脚本自己实现的 E 与 D（A/B 两天各算一次），"
                              "不是与生产 parquet 的 dir_cnt_t1_net_lead / lead_cnt_t1_breadth 对比，"
                              "只用于口径对齐与内部一致性核对。",
        "diagonal": "矩阵对角是每只股票与自己配对的能量，非零且很大（20241231 实测 cnt_t1 对角中位数 1.9e5、"
                    "最大 1.1e8；vol_t1 中位数 1.1e8、最大 4.8e11），与 src/yupei_dist/matrix_stage.rs "
                    "头部注释写的『对角 0』不符。本脚本所有统计量都把对角排除在伙伴之外；"
                    "净领先类（E）里对角在 ΣL 与 ΣS 中各出现一次，自动抵消。"
                    "生产侧的 row_sum_dir / col_sum_dir 是包含对角的，ctx.symmetric 是排除对角的，两者口径不同。",
        "degenerate_stats": "H（top20_lead_share，阈值 q>0.6）在 cnt_t1 上只有 1.3% 的股票非零、"
                            "cnt_t30 上只有 0.3%，实际是个常数零，不可用；"
                            "cnt_t005 与 vol_t1 分别为 14.7% 与 13.9%。",
        "projection_5day": "按 Spearman-Brown（k 日平均、日间秩相关 ρ 时的可靠性 kρ/(1+(k-1)ρ)）估算，"
                           "ρ=0.42 的 q_all_wmean 取 5 日均值约为 0.78，取 10 日约为 0.88。"
                           "这是算式推算，不是实测，实测要先按日算再滚动。",
        "limitation": "只有 20241230/20241231 两天有矩阵备份，跨日稳定性只由这一对相邻交易日支撑，"
                      "结论是机制样本层面的，不能当成全历史统计。",
    }
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("写出:", OUT_JSON)

    print("\n=== 跨日稳定性排名（20241230 vs 20241231，按代码交集）===")
    for r in out["cross_day_stability_rank"]:
        rho = r["cross_day_spearman"]
        print(f"{rho if rho is None else round(rho,4):>8}  {r['matrix']:9s} {r['stat']:20s} "
              f"交集 {r['n_common_total']} 两天都有值 {r['n_finite_both']}")

    print("\n=== 按统计量平均（四个矩阵）===")
    for d in out["stat_ranking_mean_over_matrices"]:
        print(f"{d['mean_cross_day_spearman']:>8.4f}  {d['stat']:20s} "
              f"min={d['min']:.4f} max={d['max']:.4f}")

    print("\n=== 参照系（对称强度 ΣW 与当日笔数的跨日秩相关）===")
    for mat in MATS:
        r = out["coverage"][mat]["reference_cross_day"]
        print(f"{mat:9s} strength={r['strength_sum_W']:.4f}  n_trades={r['n_trades']:.4f}")

    # ---- task-6: 权重曲线与更稳口径 ----
    p2 = build_probe2(days)
    with open(OUT_JSON2, "w", encoding="utf-8") as f:
        json.dump(p2, f, ensure_ascii=False, indent=1)
    print("写出:", OUT_JSON2)

    print("\n=== alpha 曲线（跨日秩相关）===")
    for mat in MATS:
        c = p2["alpha_peaks"][mat]
        line = "  ".join(f"a={k}:{('%s' % round(v,4)) if v is not None else 'NA'}"
                         for k, v in c["curve_cross_day"].items())
        print(f"{mat:9s} {line}  峰值 alpha={c['peak_alpha']} ({c['peak_cross_day_spearman']:.4f})")

    print("\n=== 口径汇总（四矩阵平均跨日；去掉规模后的跨日；与规模/与现有口径的最大|秩相关|）===")
    for r in p2["stat_ranking_mean_over_matrices"]:
        print(f"{r['mean_cross_day_spearman']:>8.4f}  nu={r['mean_cross_day_spearman_size_neutralized']:>7.4f}  "
              f"{r['stat']:20s} size<= {r['max_abs_rho_with_size']:.3f}  "
              f"existing<= {r['max_abs_rho_with_existing']:.3f}  "
              f"{ {k: (round(v,3) if v is not None else None) for k, v in r['per_matrix_cross_day'].items()} }")

    print("\n=== 结论 ===")
    print("满足三条判据的口径:", p2["conclusion"]["passing"])
    print("最优:", p2["conclusion"]["best"])
    print("放宽判据（去掉规模成分后仍稳、且与现有口径不重复）:", p2["conclusion"]["passing_relaxed"])
    print("最优（放宽）:", p2["conclusion"]["best_relaxed"])


if __name__ == "__main__":
    main()
