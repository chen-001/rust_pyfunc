"""Real-data plumbing for the BRC crosscheck.

Replicates the engine's day-selection rules (read from the tail_v5 engine loop,
NOT from its BRC code):

  dates axis  = meta/dates.npy (2430) == store template rows 244..2673, 1:1 with
                ret_sum_gap5.npy rows.
  eff         = [raw_eff_idx for raw_eff_idx in 1..n_dates-1
                 if dates[raw_eff_idx] > backtest_start
                 and factor[raw_eff_idx - 1] has at least one finite value]
  local_t     enumerates eff.
  gap5 held   = when local_t % 5 == 0 -> held_row = raw_eff_idx - 1
  gap5 grid   = when (local_t + 1) % 5 == 0  (local_t = 4, 9, 14, ...)
  day inputs  = signal F[held_row, :], ret_gap5[raw_eff_idx, :] for the mask,
                ret_sum_gap5[raw_eff_idx, :] as the future value fed to BRC,
                restrict restrict[held_row, :]
  valid mask  = isfinite(signal) & isfinite(ret_gap5) & isfinite(restrict) & restrict == 0

  The mask/value split (mask on ret_gap5, value from ret_sum_gap5) is what the
  engine's day loop does; the two arrays are not identical.
"""

import numpy as np
import rust_pyfunc as rp

from brc_ref import avg_rank_ascending

STORE = "/hdd/user_home_unsafe/chenzongwei/factor_store_hot_stock_pool_v1_fix"
BACKTEST = ("/home/chenzongwei/pythoncode/_tail_v2_shared/backtest_inputs/"
            "000905_20160104_20251231_9193_8c19aea9f4e01583")
META = "/nas197/user_home_unsafe/chenzongwei/hm104_ssm3_tail_v4/meta"
BACKTEST_START = 20160101
GAP = 5


def load_axes():
    dates = np.load(META + "/dates.npy")
    stocks = np.load(META + "/stocks.npy", allow_pickle=True).astype(str)
    return dates, stocks


def load_returns():
    """(ret_gap5, ret_sum_gap5, restrict) -- all (n_dates, n_stocks)."""
    ret = np.load(BACKTEST + "/ret_gap5.npy")
    ret_sum = np.load(BACKTEST + "/ret_sum_gap5.npy")
    restrict = np.load(BACKTEST + "/restrict.npy")
    return ret, ret_sum, restrict


def store_row_offset(dates):
    """Store template row of dates[0]; the run's axis is a contiguous slice."""
    sd = np.asarray(rp.factor_store_v5_template(STORE)["dates"])
    off = int(np.searchsorted(sd, dates[0]))
    assert (sd[off : off + dates.size] == dates).all(), "store axis != run axis"
    return off


def load_factor_matrix(col_idx, dates):
    """Factor as (n_dates, n_stocks) float32 with NaN for missing records."""
    stocks = np.load(META + "/stocks.npy", allow_pickle=True).astype(str)
    off = store_row_offset(dates)
    rec = rp.factor_store_v5_read_factor(STORE, col_idx)
    di = rec["date_id"].astype(np.int64) - off
    ci = rec["code_id"].astype(np.int64)
    v = rec["factor"].astype(np.float32)
    keep = (di >= 0) & (di < dates.size)
    di, ci, v = di[keep], ci[keep], v[keep]
    F = np.full((dates.size, stocks.size), np.nan, dtype=np.float32)
    F[di, ci] = v
    return F


def rank_fill_row(f_row, restrict_row):
    """Engine preprocessing for one day: src/tail_v5_pipeline.rs
    rank_and_fill_missing_cross_sectional_median, whose output is slot 0
    (`_smooth_1`) of tail_rank_fill_roll_block_f32 -- what the engine feeds BRC.

    1. average rank of the finite values only (NaN excluded, ties -> mean rank);
    2. stocks whose factor is NaN but restrict == 0 get rank (valid_count + 1) / 2;
       the rest stay NaN;
    3. no second rank.
    """
    f = np.asarray(f_row, dtype=np.float32)
    out = np.full(f.size, np.nan, dtype=np.float32)
    valid = np.isfinite(f)
    cnt = int(valid.sum())
    if cnt == 0:
        return out
    out[valid] = avg_rank_ascending(f[valid].astype(np.float64)).astype(np.float32)
    restr = np.asarray(restrict_row, dtype=np.float32)
    out[(~valid) & (restr == np.float32(0.0))] = np.float32((cnt + 1) / 2.0)
    return out


def day_cross_section_rankfill(F, ret, ret_sum, restrict, eff, local_t):
    """Same day selection as day_cross_section, but the signal is the engine's
    rank-filled `_smooth_1` variant instead of the raw store value."""
    r = eff[local_t]
    h = held_row(eff, local_t)
    signal = rank_fill_row(F[h], restrict[h])
    restr = restrict[h]
    mask = (
        np.isfinite(signal)
        & np.isfinite(ret[r])
        & np.isfinite(restr)
        & (restr == 0.0)
    )
    return (
        signal[mask].astype(np.float64),
        ret_sum[r][mask].astype(np.float64),
        h,
    )


def effective_days(F, dates, backtest_start=BACKTEST_START):
    """Engine's eff list: raw rows 1..n-1, date after backtest_start, row-1 not all-nan."""
    has = np.isfinite(F).any(axis=1)
    return [
        r
        for r in range(1, dates.size)
        if dates[r] > backtest_start and has[r - 1]
    ]


def grid_days(eff):
    """(local_t, raw_eff_idx) for gap5 IC sampling days."""
    return [(t, r) for t, r in enumerate(eff) if (t + 1) % GAP == 0]


def held_row(eff, local_t, gap=GAP):
    """Engine's gap5 held row: refreshed at local_t % gap == 0, value raw_eff_idx - 1."""
    return eff[local_t - (local_t % gap)] - 1


def day_cross_section(F, ret, ret_sum, restrict, eff, local_t):
    """(signal, future, held_row) exactly as the engine's day loop filters them.

    mask: finite signal & finite ret_gap5 & finite restrict & restrict == 0
    future value: ret_sum_gap5 (same rows as the mask)
    """
    r = eff[local_t]
    h = held_row(eff, local_t)
    signal = F[h]
    restr = restrict[h]
    mask = (
        np.isfinite(signal)
        & np.isfinite(ret[r])
        & np.isfinite(restr)
        & (restr == 0.0)
    )
    return (
        signal[mask].astype(np.float64),
        ret_sum[r][mask].astype(np.float64),
        h,
    )
