#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""玉佩2 配对互补网络：机制层面的诊断（回答假设卡上那四条竞争解释）。

读 sandbox_yupei_pair 的输出目录（schema 见 experiments/yupei2/pair_spec.txt 第 4 节），
只做机制判定，不看收益。降维因子的构造与跨日/老因子对照在 pair_reduce.py 里，不在这里重复。

运行:
  python experiments/yupei2/pair_net_diag.py <pair_out_dir> <date> [<pair_out_dir2> <date2>]
"""
import os
import sys
import json
import struct
import numpy as np
from scipy.stats import spearmanr

BACKUP = "/hdd/user_home_unsafe/chenzongwei/yupei_dist_backup"


def read_f32(path, n=None):
    a = np.fromfile(path, dtype="<f4")
    if n is not None:
        assert a.size == n * n, f"{path} 有 {a.size} 个元素, 期望 {n*n}"
        a = a.reshape(n, n)
    return a


def read_f64(path):
    return np.fromfile(path, dtype="<f8")


def read_codes(path):
    with open(path) as f:
        return [l.strip() for l in f if l.strip()]


def read_stats_ntrades(path):
    raw = open(path, "rb").read()
    n = struct.unpack_from("<Q", raw, 0)[0]
    a = np.frombuffer(raw, dtype="<f8", count=n * 14, offset=8).reshape(n, 14)
    return a[:, 0], a[:, 2]          # 成交笔数, 总成交量


def read_industry(day_dir, codes):
    """industry.bin: u64 n + n×i16。n 是全市场 symbol_map 的行数（20241231 是 5770），
    不是备份 codes.txt 的行数，顺序对应 /ssd_data/data/basic_info/symbol_map.csv 的 pos 列。
    所以必须靠 pos 映射回 codes，不能直接对齐。"""
    p = os.path.join(day_dir, "industry.bin")
    if not os.path.exists(p):
        return None
    raw = open(p, "rb").read()
    m = struct.unpack_from("<Q", raw, 0)[0]
    arr = np.frombuffer(raw, dtype="<i2", count=m, offset=8).astype(np.int32)
    pos = {}
    with open("/ssd_data/data/basic_info/symbol_map.csv") as f:
        next(f)
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 2:
                pos[parts[0]] = int(parts[1])
    return np.array([arr[pos[c]] if c in pos and pos[c] < m else -1 for c in codes], dtype=np.int32)


def load(day_dir, date):
    codes = read_codes(os.path.join(day_dir, "codes.txt"))
    n = len(codes)
    d = {
        "date": date, "codes": codes, "n": n,
        "G": read_f32(os.path.join(day_dir, "G.bin"), n),
        "Z": read_f32(os.path.join(day_dir, "Z.bin"), n),
        "RHO_U": read_f32(os.path.join(day_dir, "RHO_U.bin"), n),
        "SINGLE": read_f32(os.path.join(day_dir, "SINGLE.bin"), n),
        "X0": read_f32(os.path.join(day_dir, "X0.bin"), n),
        "GSTAR": read_f32(os.path.join(day_dir, "GSTAR.bin"), n),
        "ZS": read_f32(os.path.join(day_dir, "ZS.bin"), n),
        "X0STAR": read_f32(os.path.join(day_dir, "X0STAR.bin"), n),
        "rho": read_f32(os.path.join(day_dir, "rho.bin"), n),
    }
    bak = os.path.join(BACKUP, str(date))
    d["ntrades"], d["total_vol"] = read_stats_ntrades(os.path.join(bak, "stats.bin"))
    d["ind"] = read_industry(bak, codes)
    return d


def upper_tri(M):
    iu = np.triu_indices(M.shape[0], 1)
    return M[iu]


def main():
    dd, date = sys.argv[1], int(sys.argv[2])
    A = load(os.path.join(dd, str(date)), date)
    n = A["n"]
    print(f"=== {date}  n={n} ===")

    # 只统计"股票-股票"配对。非股票（基金/债券）那 280 只在 --stocks-only 口径下
    # 整个行列都不参与计算，输出里是 0，如果混进来会把分位数和占比带偏。
    # 白名单口径必须和 Rust 那边一致（symbol_map.csv 的 symbol 列）。
    sm = set()
    with open("/ssd_data/data/basic_info/symbol_map.csv") as f:
        next(f)
        for line in f:
            p = line.strip().split(",")
            if p:
                sm.add(p[0])
    keep_stock = np.array([c in sm for c in A["codes"]])
    iu_all = np.triu_indices(n, 1)
    m_stock = keep_stock[iu_all[0]] & keep_stock[iu_all[1]]

    iu = np.triu_indices(n, 1)
    G = A["G"][iu][m_stock]
    Z = A["Z"][iu][m_stock]
    RU = A["RHO_U"][iu][m_stock]
    SG = A["SINGLE"][iu][m_stock]
    ZS = A["ZS"][iu][m_stock]
    GS = A["GSTAR"][iu][m_stock]
    RC = A["rho"][iu][m_stock]             # rho(B,C)
    nt_b = A["ntrades"][iu[0]][m_stock]
    nt_c = A["ntrades"][iu[1]][m_stock]
    ib_all, ic_all = iu[0][m_stock], iu[1][m_stock]
    out = {"date": date, "n": n, "n_stock_stock_pairs": int(m_stock.sum()),
           "n_all_pairs": int(len(m_stock))}
    print(f"[前] 股票-股票配对 {out['n_stock_stock_pairs']} / 全部 {out['n_all_pairs']}")

    # ---------- 0 基本分布 ----------
    def qs(v):
        return [float(np.percentile(v, p)) for p in (1, 10, 50, 90, 99)]
    out["G_quantiles_p1_10_50_90_99"] = qs(G)
    out["Z_quantiles_p1_10_50_90_99"] = qs(Z)
    out["Z_gt2_share"] = float((Z > 2).mean())
    out["Z_gt3_share"] = float((Z > 3).mean())
    out["ZS_gt2_share"] = float((ZS > 2).mean())
    out["ZS_gt3_share"] = float((ZS > 3).mean())
    out["RHO_U_quantiles"] = qs(RU)
    out["RHO_U_gt_0.5_share"] = float((RU > 0.5).mean())
    print(f"[0] G 分位 p1/10/50/90/99 = {out['G_quantiles_p1_10_50_90_99']}")
    print(f"[0] Z 分位 = {out['Z_quantiles_p1_10_50_90_99']}, Z>2 占比 {out['Z_gt2_share']:.4f}, Z>3 {out['Z_gt3_share']:.4f}")
    print(f"[0] ZS>2 占比 {out['ZS_gt2_share']:.4f}, ZS>3 {out['ZS_gt3_share']:.4f}")
    print(f"[0] rho(X0,U) 分位 = {out['RHO_U_quantiles']}, 大于 0.5 的占比 {out['RHO_U_gt_0.5_share']:.4f}")

    # ---------- 0b X0 集中度（用户原规则最要命的结构问题）----------
    x0 = A["X0"][iu][m_stock].astype(int)
    x0s = A["X0STAR"][iu][m_stock].astype(int)
    for tag, v in (("X0", x0), ("X0star", x0s)):
        cnt = np.bincount(v, minlength=n)
        srt = np.sort(cnt)[::-1]
        out[f"{tag}_n_distinct"] = int((cnt > 0).sum())
        out[f"{tag}_top10_share"] = float(srt[:10].sum() / len(v))
        out[f"{tag}_top100_share"] = float(srt[:100].sum() / len(v))
        out[f"{tag}_top5"] = [[A["codes"][i], int(cnt[i])] for i in np.argsort(-cnt)[:5]]
        print(f"[0b] {tag}: 用到 {out[f'{tag}_n_distinct']} 只, 前10占 {out[f'{tag}_top10_share']:.3f}, "
              f"前100占 {out[f'{tag}_top100_share']:.3f}, top5 {out[f'{tag}_top5']}")

    # ---------- 解释一：活跃度影子 ----------
    lg = np.log10(np.maximum(nt_b, 1) * np.maximum(nt_c, 1))
    r_g_nt, _ = spearmanr(G, lg)
    r_z_nt, _ = spearmanr(Z, lg)
    out["spearman_G_vs_logpair_trades"] = float(r_g_nt)
    out["spearman_Z_vs_logpair_trades"] = float(r_z_nt)
    # 在活跃度 10x10 层内再看 Z 的分散度（空模型分层后应当已经平掉）。
    # 这里必须用 Rust 那套切点：按全市场 ntrades 升序后取下标 k*n/10，
    # 用 np.percentile 在配对子集上另切一套会出现层与层错位。
    cuts = np.array([np.sort(A["ntrades"])[(k * n) // 10] for k in range(1, 10)])
    dec_b = np.digitize(nt_b, cuts)
    dec_c = np.digitize(nt_c, cuts)
    strat = dec_b * 10 + dec_c
    med = np.array([np.median(Z[strat == s]) if (strat == s).sum() > 100 else np.nan
                    for s in range(100)])
    out["Z_stratum_median_min_max"] = [float(np.nanmin(med)), float(np.nanmax(med))]
    out["spearman_G_vs_Z"] = float(spearmanr(G, Z).statistic)
    print(f"[1] Spearman(G, log 笔数对) = {r_g_nt:.4f}; Spearman(Z, log 笔数对) = {r_z_nt:.4f}")
    print(f"[1] 分层 Z 中位数范围 [{np.nanmin(med):.3f}, {np.nanmax(med):.3f}]（应当接近 0）")

    # ---------- 解释二：行业 ----------
    ind = A["ind"]
    if ind is not None:
        ok = (ind[ib_all] >= 0) & (ind[ic_all] >= 0)
        same = (ind[ib_all] == ind[ic_all])
        base = float(same[ok].mean())
        out["industry_known_share"] = float(ok.mean())
        out["same_industry_share_baseline"] = base
        rows = {}
        for thr in (1.0, 2.0, 3.0, 5.0):
            m = ok & (Z > thr)
            rows[f"Z>{thr}"] = [int(m.sum()), float(same[m].mean()) if m.sum() else float("nan")]
        out["same_industry_share_by_Z"] = rows
        print(f"[2] 同行业基线占比 {base:.4f}（已知行业 {out['industry_known_share']:.3f}）")
        for k, v in rows.items():
            print(f"[2]   {k}: 配对数 {v[0]}, 同行业占比 {v[1]:.4f}")

    # ---------- 解释四：G 是不是 rho(B,C) 的单调函数 ----------
    r_g_rc, _ = spearmanr(G, RC)
    out["spearman_G_vs_rhoBC"] = float(r_g_rc)
    out["spearman_Z_vs_rhoBC"] = float(spearmanr(Z, RC).statistic)
    bins = np.percentile(RC, np.arange(0, 101, 10))
    per = []
    for k in range(10):
        m = (RC >= bins[k]) & (RC <= bins[k + 1]) if k == 9 else (RC >= bins[k]) & (RC < bins[k + 1])
        if m.sum() < 100:
            per.append([k, float("nan"), float("nan"), int(m.sum())])
            continue
        per.append([k, float(np.percentile(G[m], 10)), float(np.percentile(G[m], 90)), int(m.sum())])
    out["G_by_rhoBC_decile_p10_p90"] = per
    print(f"[4] Spearman(G, rho(B,C)) = {r_g_rc:.4f}, Spearman(Z, rho(B,C)) = {out['spearman_Z_vs_rhoBC']:.4f}")
    for p in per:
        print(f"[4]   rho(B,C) 第 {p[0]} 档: G p10={p[1]:.4f} p90={p[2]:.4f} n={p[3]}")

    # ---------- 解释三：X0 替换后的增益保留率 ----------
    # 高 Z 配对里，把 X0 换成它在 rho 网络中的第一名邻居，看 G 保留多少
    topm = Z > 3.0
    tb, tc = ib_all[topm], ic_all[topm]
    take = min(len(tb), 4000)
    sel = np.linspace(0, len(tb) - 1, take).astype(int) if len(tb) > take else np.arange(len(tb))
    tb, tc = tb[sel], tc[sel]
    keep = []
    for b, c in zip(tb, tc):
        x0 = int(A["X0"][b, c])
        if x0 >= n:
            continue
        col = A["rho"][x0].copy()
        col[x0] = -2.0
        x1 = int(np.argmax(col))
        ru0 = A["RHO_U"][b, c]
        keep.append((ru0, float(max(A["rho"][x1, b], A["rho"][x1, c]))))
    if keep:
        k = np.array(keep)
        keep_ratio = (k[:, 1] / np.maximum(k[:, 0], 1e-9))
        out["X0_top_neighbor_single_ratio_p10_50_90"] = [float(np.percentile(keep_ratio, p))
                                                         for p in (10, 50, 90)]
        print(f"[3] 高 Z 配对：把 X0 换成它最强邻居后，max 单只相容度 / 原 rho(X0,U) 的分位 "
              f"{out['X0_top_neighbor_single_ratio_p10_50_90']}")

    # ---------- 头部配对的股票集中度 ----------
    order = np.argsort(-Z)
    top = order[:2000]
    stocks = np.unique(np.concatenate([ib_all[top], ic_all[top]]))
    out["top2000Z_distinct_stocks"] = int(len(stocks))
    print(f"[5] Z 最高 2000 对涉及 {len(stocks)} 只不同股票（共 {n} 只）")

    # ---------- 跨日稳定性 ----------
    if len(sys.argv) > 4:
        dd2, date2 = sys.argv[3], int(sys.argv[4])
        B = load(os.path.join(dd2, str(date2)), date2)
        common = sorted(set(A["codes"]) & set(B["codes"]))
        ia = {c: i for i, c in enumerate(A["codes"])}
        ib = {c: i for i, c in enumerate(B["codes"])}
        idx_a = np.array([ia[c] for c in common])
        idx_b = np.array([ib[c] for c in common])
        Za = A["Z"][np.ix_(idx_a, idx_a)]
        Zb = B["Z"][np.ix_(idx_b, idx_b)]
        iu2 = np.triu_indices(len(common), 1)
        za, zb = Za[iu2], Zb[iu2]
        out["cross_day"] = {
            "date_pair": [date, date2], "common_stocks": len(common),
            "spearman_Z": float(spearmanr(za, zb).statistic),
            "spearman_G": float(spearmanr(A["G"][np.ix_(idx_a, idx_a)][iu2],
                                          B["G"][np.ix_(idx_b, idx_b)][iu2]).statistic),
            "spearman_rho": float(spearmanr(A["rho"][np.ix_(idx_a, idx_a)][iu2],
                                            B["rho"][np.ix_(idx_b, idx_b)][iu2]).statistic),
        }
        ij = np.array(np.triu_indices(len(common), 1)).T
        ja = {}
        for thr in (2.0, 3.0):
            ta = set(map(tuple, ij[za > thr]))
            tb2 = set(map(tuple, ij[zb > thr]))
            ja[thr] = float(len(ta & tb2) / max(len(ta | tb2), 1))
        out["cross_day"]["jaccard_Zgt2"] = ja[2.0]
        out["cross_day"]["jaccard_Zgt3"] = ja[3.0]
        # ZS（最大补缺口口径）的跨日：这是用户原规则之外我加的那一路
        zsa = A["ZS"][np.ix_(idx_a, idx_a)][iu2]
        zsb = B["ZS"][np.ix_(idx_b, idx_b)][iu2]
        out["cross_day"]["spearman_ZS"] = float(spearmanr(zsa, zsb).statistic)
        for thr in (2.0, 3.0):
            ta = set(map(tuple, ij[zsa > thr]))
            tb2 = set(map(tuple, ij[zsb > thr]))
            out["cross_day"]["jaccard_ZSgt%d" % int(thr)] = float(len(ta & tb2) / max(len(ta | tb2), 1))
        print(f"[6] 跨日 共同股票 {len(common)}: Spearman(Z)={out['cross_day']['spearman_Z']:.4f} "
              f"Spearman(ZS)={out['cross_day']['spearman_ZS']:.4f} "
              f"Spearman(G)={out['cross_day']['spearman_G']:.4f} "
              f"Spearman(rho)={out['cross_day']['spearman_rho']:.4f} "
              f"Z>3 Jaccard={out['cross_day']['jaccard_Zgt3']:.4f} "
              f"ZS>3 Jaccard={out['cross_day']['jaccard_ZSgt3']:.4f}")

    with open(f"/tmp/yupei2_pair_diag_{date}.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"写入 /tmp/yupei2_pair_diag_{date}.json")


if __name__ == "__main__":
    main()
