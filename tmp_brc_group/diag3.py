"""诊断 3：对 6 个未对上的因子，比对 neu 通道 IC（surface 排序是否一致）。"""
import numpy as np
import pandas as pd
import rust_pyfunc as rp

import repro as R
from diag2 import load_industry  # noqa

FAIL = ["hotpool_ext_ind_rel_x15y10_ba_hot_rk_vol_autocorr1_abs_fold_mean_smooth_5",
        "hotpool_ext_indpool_x60y3_ba_hot_f31_mean_max_smooth_10",
        "x15y10_ba_cold_f00_corr_f28_mean_smooth_10",
        "x15y10_ba_cold_f06_corr_f36_mean_smooth_5",
        "x15y10_ba_cold_f17_mean_smooth_1",
        "x60y3_buy_hot_f08_mean_mean_smooth_10"]

dates, stocks = R.load_axes()
ret, ret_sum, restrict, index = R.load_returns()
info = rp.factor_store_v5_info(R.STORE)
col = {n: i for i, n in enumerate(list(info["factor_names"]))}
sd = np.asarray(rp.factor_store_v5_template(R.STORE)["dates"])
off = int(np.searchsorted(sd, dates[0]))
T, N = dates.size, stocks.size
dates_list = dates.astype(np.int32).tolist()
stocks_list = [str(s) for s in stocks]
industry = load_industry(dates, stocks)
shared = rp.neutralize_std_precompute_py(industry, restrict, R.STYLE, dates_list, stocks_list)

neu_tbl = pd.read_parquet(R.SUMMARY)
neu_tbl = neu_tbl[(neu_tbl["stage"] == "neu") & (neu_tbl["gap"] == 5)].set_index("factor_name")


def ic_gap5(surf, dates, ret, ret_sum, restrict, gap=5):
    has = np.isfinite(surf).any(axis=1)
    eff = [r for r in range(1, dates.size) if dates[r] > R.BACKTEST_START and has[r - 1]]
    ics, held = [], eff[0] - 1
    for local_t, r in enumerate(eff):
        if local_t % gap == 0:
            held = r - 1
        if (local_t + 1) % gap:
            continue
        sig = surf[held]
        rs = restrict[held]
        mask = np.isfinite(sig) & np.isfinite(ret[r]) & np.isfinite(rs) & (rs == np.float32(0.0))
        s, f = sig[mask], ret_sum[r][mask]
        n = s.size
        if n < 2:
            continue
        sr, fr = R.avg_rank(s), R.avg_rank(f)
        ics.append(1.0 - 6.0 * ((sr - fr) ** 2).sum() / (n * (n * n - 1.0)))
    return float(np.mean(ics))


def ordinal_rank(x):
    o = np.argsort(x, kind="stable")
    r = np.empty(x.size, dtype=np.float64)
    r[o] = np.arange(1, x.size + 1)
    return r


def ic_ordinal(surf, dates, ret, ret_sum, restrict, gap=5):
    """用 ordinal 秩（引擎 rk_ordinal / walk_buf 的那套）算 IC。"""
    has = np.isfinite(surf).any(axis=1)
    eff = [r for r in range(1, dates.size) if dates[r] > R.BACKTEST_START and has[r - 1]]
    ics, held = [], eff[0] - 1
    for local_t, r in enumerate(eff):
        if local_t % gap == 0:
            held = r - 1
        if (local_t + 1) % gap:
            continue
        sig = surf[held]
        rs = restrict[held]
        mask = np.isfinite(sig) & np.isfinite(ret[r]) & np.isfinite(rs) & (rs == np.float32(0.0))
        s, f = sig[mask], ret_sum[r][mask]
        n = s.size
        if n < 2:
            continue
        d = ordinal_rank(f) - ordinal_rank(s)
        ics.append(1.0 - 6.0 * (d * d).sum() / (n * (n * n - 1.0)))
    return float(np.mean(ics))


for name in FAIL:
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
    block = rp.tail_v5_rank_fill_roll_block_f32(F, restrict, R.WINDOWS)
    surf = np.array(block[:, :, slot], dtype=np.float32, copy=True)
    del block, F
    nn = rp.neutralize_std_block_with_shared(surf[:, :, None], shared, False)
    nn = np.array(nn[:, :, 0], dtype=np.float32, copy=True)
    tbl = neu_tbl.loc[name]
    print(f"\n### {name}  slot={slot}")
    print(f"  IC avg-rank calc={ic_gap5(nn, dates, ret, ret_sum, restrict):.8f} tbl={tbl['IC_mean']:.8f}")
    print(f"  IC ordinal  calc={ic_ordinal(nn, dates, ret, ret_sum, restrict):.8f}")
    g, eff, ratio = R.group_returns(nn, ret, restrict, dates)
    m = R.metrics_from_groups(g, eff, index)
    print(f"  ls calc={m['ls_mean250']:.8f} tbl={tbl['annualized_return']:.8f} diff={m['ls_mean250']-tbl['annualized_return']:.3e}")
