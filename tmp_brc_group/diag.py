"""诊断：raw / neu 两个通道都复算，和两张 summary 表对比。"""
import numpy as np
import pandas as pd
import rust_pyfunc as rp

import repro as R

dates, stocks = R.load_axes()
ret, ret_sum, restrict, index = R.load_returns()
info = rp.factor_store_v5_info(R.STORE)
names = list(info["factor_names"])
col = {n: i for i, n in enumerate(names)}
sd = np.asarray(rp.factor_store_v5_template(R.STORE)["dates"])
off = int(np.searchsorted(sd, dates[0]))
T, N = dates.size, stocks.size
dates_list = dates.astype(np.int32).tolist()
stocks_list = [str(s) for s in stocks]


def load(name):
    base, is_fold, slot = R.parse_name(name)
    rec = rp.factor_store_v5_read_factor(R.STORE, col[base])
    di = np.asarray(rec["date_id"], dtype=np.int64) - off
    ci = np.asarray(rec["code_id"], dtype=np.int64)
    v = np.asarray(rec["factor"], dtype=np.float32)
    keep = (di >= 0) & (di < T)
    F = np.full((T, N), np.nan, dtype=np.float32)
    F[di[keep], ci[keep]] = v[keep]
    if is_fold:
        F = R.fold_values(F)
    return F, slot, base, is_fold


def ic_stats(surf, dates, gap=5):
    has = np.isfinite(surf).any(axis=1)
    eff = [r for r in range(1, dates.size) if dates[r] > R.BACKTEST_START and has[r - 1]]
    ics = []
    held = eff[0] - 1
    for local_t, r in enumerate(eff):
        if local_t % gap == 0:
            held = r - 1
        sig = surf[held]
        rs = restrict[held]
        mask = np.isfinite(sig) & np.isfinite(ret[r]) & np.isfinite(rs) & (rs == np.float32(0.0))
        if (local_t + 1) % gap:
            continue
        s = sig[mask]
        f = ret_sum[r][mask]
        n = s.size
        if n < 2:
            continue
        sr = R.avg_rank(s)
        fr = R.avg_rank(f)
        ics.append(1.0 - 6.0 * ((sr - fr) ** 2).sum() / (n * (n * n - 1.0)))
    return float(np.mean(ics)), len(ics)


rolled = pd.read_parquet(
    "/nas197/user_home_unsafe/chenzongwei/hm104_brc_tail_v4/metrics/summary_rolled_gap5_candidates.parquet")
neu = pd.read_parquet(
    "/nas197/user_home_unsafe/chenzongwei/hm104_brc_tail_v4/metrics/summary_neu_gap5_candidates.parquet")
rolled = rolled[rolled["stage"] == "rolled"].set_index("factor_name")
neu = neu[neu["stage"] == "neu"].set_index("factor_name")

for name in ["x15y10_ba_cold_f34_mean_smooth_1",
             "hotpool_ext_ind_rel_x15y10_ba_cold_rk_vol_autocorr1_smooth_1"]:
    F, slot, base, is_fold = load(name)
    block = rp.tail_v5_rank_fill_roll_block_f32(F, restrict, R.WINDOWS)
    surf = np.array(block[:, :, slot], dtype=np.float32, copy=True)
    del block, F
    ic_r, nd_r = ic_stats(surf, dates)
    g_r, eff, ratio = R.group_returns(surf, ret, restrict, dates)
    m_r = R.metrics_from_groups(g_r, eff, index)
    nn = rp.tail_v5_neutralize_block_exact(R.STYLE, dates_list, stocks_list, surf[:, :, None], True, 12, False)
    nn = np.array(nn[:, :, 0], dtype=np.float32, copy=True)
    ic_n, nd_n = ic_stats(nn, dates)
    g_n, eff_n, ratio_n = R.group_returns(nn, ret, restrict, dates)
    m_n = R.metrics_from_groups(g_n, eff_n, index)
    tr, tn = rolled.loc[name], neu.loc[name]
    print(f"\n### {name}  slot={slot} base={base} fold={is_fold}")
    print(f"  raw: IC calc={ic_r:.6f} tbl={tr['IC_mean']:.6f} d={abs(ic_r-tr['IC_mean']):.2e} nd={nd_r}/{int(tr['date_size'])}")
    print(f"  raw: ls calc={m_r['ls_mean250']:.6f} tbl={tr['annualized_return']:.6f} | "
          f"hedge calc={m_r['hedge_mean250']:.6f} tbl={tr['hedge_annualized_return']:.6f}")
    print(f"  neu: IC calc={ic_n:.6f} tbl={tn['IC_mean']:.6f} d={abs(ic_n-tn['IC_mean']):.2e} nd={nd_n}/{int(tn['date_size'])}")
    print(f"  neu: ls calc={m_n['ls_mean250']:.6f} tbl={tn['annualized_return']:.6f} | "
          f"hedge calc={m_n['hedge_mean250']:.6f} tbl={tn['hedge_annualized_return']:.6f}")
    print("  neu group_sum:", np.round(m_n["group_sum"], 6))
