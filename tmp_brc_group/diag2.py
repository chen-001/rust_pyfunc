"""诊断 2：用 neutralize_std_* （引擎 v7/v8 实际走的中性化）重算 neu 通道。"""
import numpy as np
import pandas as pd
import rust_pyfunc as rp

import repro as R

INDUSTRY_CSV = "/nas197/binary/stock/sz_alpha/csv/vars/Base/SW_IND_CODE.csv"


def load_industry(dates, stocks):
    df = pd.read_csv(INDUSTRY_CSV, index_col=0).T
    df.index = pd.to_datetime(df.index)
    ds = [str(int(d)) for d in dates]
    out = np.full((len(dates), len(stocks)), np.nan, dtype=np.float64)
    si = pd.Index(stocks)
    for i, d in enumerate(ds):
        ts = pd.Timestamp(f"{d[:4]}-{d[4:6]}-{d[6:8]}")
        if ts in df.index:
            out[i] = df.loc[ts].reindex(si).to_numpy(dtype=np.float64)
    return out


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
print("industry 覆盖:", np.isfinite(industry).mean())

rolled = pd.read_parquet(
    "/nas197/user_home_unsafe/chenzongwei/hm104_brc_tail_v4/metrics/summary_rolled_gap5_candidates.parquet")
neu = pd.read_parquet(
    "/nas197/user_home_unsafe/chenzongwei/hm104_brc_tail_v4/metrics/summary_neu_gap5_candidates.parquet")
rolled = rolled[rolled["stage"] == "rolled"].set_index("factor_name")
neu = neu[neu["stage"] == "neu"].set_index("factor_name")

for name in ["x15y10_ba_cold_f34_mean_smooth_1",
             "hotpool_ext_ind_rel_x15y10_ba_cold_rk_vol_autocorr1_smooth_1"]:
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
    nn = rp.neutralize_std_block_py(surf[:, :, None], industry, restrict, R.STYLE,
                                    dates_list, stocks_list, False)
    nn = np.array(nn[:, :, 0], dtype=np.float32, copy=True)
    g, eff, ratio = R.group_returns(nn, ret, restrict, dates)
    m = R.metrics_from_groups(g, eff, index)
    tn = neu.loc[name]
    print(f"\n### {name}")
    print(f"  neu(neutralize_std) ls={m['ls_mean250']:.6f} tbl={tn['annualized_return']:.6f} | "
          f"hedge={m['hedge_mean250']:.6f} tbl={tn['hedge_annualized_return']:.6f} | "
          f"date_size={m['date_size']} tbl={int(tn['date_size'])} ratio={np.nanmean(ratio):.6f} tbl={tn['ratio_mean']:.6f}")
