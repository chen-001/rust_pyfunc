"""诊断 4：对 6 个未对上因子，比较 RAW 通道（无中性化）是否也对不上。

若 raw 通道逐位对上 → 问题在中性化分支；若 raw 也差 → 问题在分组循环。
"""
import numpy as np
import pandas as pd
import rust_pyfunc as rp

import repro as R

FAIL = ["hotpool_ext_ind_rel_x15y10_ba_hot_rk_vol_autocorr1_abs_fold_mean_smooth_5",
        "hotpool_ext_indpool_x60y3_ba_hot_f31_mean_max_smooth_10",
        "x15y10_ba_cold_f00_corr_f28_mean_smooth_10",
        "x15y10_ba_cold_f06_corr_f36_mean_smooth_5",
        "x15y10_ba_cold_f17_mean_smooth_1",
        "x60y3_buy_hot_f08_mean_mean_smooth_10",
        # 两个自检已对上的做对照
        "hotpool_ext_ind_rel_x15y10_ba_cold_rk_vol_autocorr1_smooth_1",
        "x15y10_ba_cold_f34_mean_smooth_1"]

dates, stocks = R.load_axes()
ret, ret_sum, restrict, index = R.load_returns()
info = rp.factor_store_v5_info(R.STORE)
col = {n: i for i, n in enumerate(list(info["factor_names"]))}
sd = np.asarray(rp.factor_store_v5_template(R.STORE)["dates"])
off = int(np.searchsorted(sd, dates[0]))
T, N = dates.size, stocks.size

raw_tbl = pd.read_parquet(
    "/nas197/user_home_unsafe/chenzongwei/hm104_brc_tail_v4/metrics/summary_rolled_gap5_candidates.parquet")
raw_tbl = raw_tbl[raw_tbl["stage"] == "rolled"].set_index("factor_name")

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
    g, eff, ratio = R.group_returns(surf, ret, restrict, dates)
    m = R.metrics_from_groups(g, eff, index)
    if name not in raw_tbl.index:
        print(f"{name[:58]:58s} RAW 表里没有这一行（raw 通道被筛掉），跳过")
        continue
    tr = raw_tbl.loc[name]
    print(f"{name[:58]:58s} ls={m['ls_mean250']:.8f} tbl={tr['annualized_return']:.8f} "
          f"d={m['ls_mean250']-tr['annualized_return']:+.2e} | "
          f"hedge d={m['hedge_mean250']-tr['hedge_annualized_return']:+.2e} | "
          f"ds={m['date_size']}/{int(tr['date_size'])}")
