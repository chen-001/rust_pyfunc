import os


import numpy as np
import pandas as pd

from .pandas_correlation import fast_correlation_matrix_v2_df
from .pandas_corrwith import corrwith
from .rust_pyfunc import (
    calculate_binned_entropy_1d,
    calculate_lyapunov_exponent,
    find_max_range_product,
    lz_complexity,
    trend_2d,
)

_STOCK_ROOTS = ["/ssd_data/stock", "/nas197/binary/stock/sz_alpha/stock"]
_TRADE_EPOCH_OFFSET = pd.Timestamp("1970-01-01 08:00:00")

__all__ = [
    "adjust_afternoon",
    "read_trade",
    "read_market",
    "read_market_pair",
    "get_features_factors_single",
    "get_features_factors",
    "get_features_names",
]


def _resolve_stock_path(date: int, subdir: str, filename: str) -> str:
    if env := os.environ.get("RUST_PYFUNC_LEVEL2_PATH"):
        return os.path.join(env, str(date), subdir, filename)
    for root in _STOCK_ROOTS:
        path = os.path.join(root, str(date), subdir, filename)
        if os.path.exists(path):
            return path
    return os.path.join(_STOCK_ROOTS[0], str(date), subdir, filename)


def _to_exchange_timestamp(series: pd.Series) -> pd.Series:
    return pd.to_timedelta(series / 1e6, unit="s") + _TRADE_EPOCH_OFFSET


def adjust_afternoon(df: pd.DataFrame, only_inday: int = 1) -> pd.DataFrame:
    start = "09:30:00" if only_inday else "09:00:00"
    end = "14:57:00" if only_inday else "15:00:00"
    if df.index.name == "exchtime":
        df1 = df.between_time(start, "11:30:00")
        df2 = df.between_time("13:00:00", end)
        df2.index = df2.index - pd.Timedelta(minutes=90)
        df = pd.concat([df1, df2])
    elif "exchtime" in df.columns:
        df1 = df.set_index("exchtime").between_time(start, "11:30:00")
        df2 = df.set_index("exchtime").between_time("13:00:00", end)
        df2.index = df2.index - pd.Timedelta(minutes=90)
        df = pd.concat([df1, df2]).reset_index()
    return df


def read_trade(symbol: str, date: int, with_retreat: int = 0) -> pd.DataFrame:
    file_name = f"{symbol}_{date}_transaction.csv"
    file_path = _resolve_stock_path(date, "transaction", file_name)
    df = pd.read_csv(
        file_path,
        dtype={"symbol": str},
        usecols=[
            "exchtime",
            "price",
            "volume",
            "turnover",
            "flag",
            "index",
            "localtime",
            "ask_order",
            "bid_order",
        ],
        memory_map=True,
        engine="c",
        low_memory=False,
    )
    if not with_retreat:
        df = df[df.flag != 32]
    df.exchtime = _to_exchange_timestamp(df.exchtime)
    return df


def read_market(symbol: str, date: int, with_high_low_limited: int = 0) -> pd.DataFrame:
    file_name = f"{symbol}_{date}_market_data.csv"
    file_path = _resolve_stock_path(date, "market_data", file_name)
    df = pd.read_csv(
        file_path,
        dtype={"symbol": str},
        memory_map=True,
        engine="c",
        low_memory=False,
    )
    df.exchtime = _to_exchange_timestamp(df.exchtime)
    if not with_high_low_limited:
        df = df[(df.ask_prc1 != 0) & (df.bid_prc1 != 0)]
    return df


def read_market_pair(symbol: str, date: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = read_market(symbol, date)
    df = df[df.last_prc != 0]

    ask_prc_cols = [f"ask_prc{i}" for i in range(1, 11)]
    ask_vol_cols = [f"ask_vol{i}" for i in range(1, 11)]
    asks = pd.concat(
        [
            pd.melt(
                df[ask_prc_cols + ["exchtime"]],
                id_vars=["exchtime"],
                value_name="price",
            )
            .rename(columns={"variable": "number"})
            .set_index("exchtime"),
            pd.melt(
                df[ask_vol_cols + ["exchtime"]],
                id_vars=["exchtime"],
                value_name="vol",
            )
            .drop(columns=["variable"])
            .set_index("exchtime"),
        ],
        axis=1,
    )
    asks = asks[asks.price != 0]
    asks.number = asks.number.str.slice(7).astype(int)
    asks = asks.reset_index().sort_values(by=["exchtime", "number"]).reset_index(drop=True)

    bid_prc_cols = [f"bid_prc{i}" for i in range(1, 11)]
    bid_vol_cols = [f"bid_vol{i}" for i in range(1, 11)]
    bids = pd.concat(
        [
            pd.melt(
                df[bid_prc_cols + ["exchtime"]],
                id_vars=["exchtime"],
                value_name="price",
            )
            .rename(columns={"variable": "number"})
            .set_index("exchtime"),
            pd.melt(
                df[bid_vol_cols + ["exchtime"]],
                id_vars=["exchtime"],
                value_name="vol",
            )
            .drop(columns=["variable"])
            .set_index("exchtime"),
        ],
        axis=1,
    )
    bids = bids[bids.price != 0]
    bids.number = bids.number.str.slice(7).astype(int)
    bids = bids.reset_index().sort_values(by=["exchtime", "number"]).reset_index(drop=True)
    return asks, bids


def _calc_lyapunov(series: pd.Series) -> float:
    try:
        return calculate_lyapunov_exponent(series.to_numpy(float))["lyapunov_exponent"]
    except Exception:
        return np.nan


def _calc_lz_complexity(series: pd.Series) -> float:
    try:
        return lz_complexity(series.to_numpy(float), [0.33, 0.66])
    except Exception:
        return np.nan


def _calc_entropy(series: pd.Series) -> float:
    try:
        n_bins = int(np.ceil(np.log2(len(series)))) + 1
        return calculate_binned_entropy_1d(series.to_numpy(float), n_bins=n_bins)
    except Exception:
        return np.nan


def _calc_max_range_product(series: pd.Series) -> float:
    try:
        x, y, _ = find_max_range_product(series.to_numpy(float))
        return abs(x - y) / series.shape[0]
    except Exception:
        return np.nan


def get_features_factors_single(
    df: pd.DataFrame,
    with_abs: bool = False,
    with_max_min: bool = False,
    with_percentiles: bool = True,
    with_lag_autocorr: int = 1,
    with_threshold_counts: bool = True,
    with_period_compare: bool = True,
    with_lyapunov_exponent: bool = True,
    with_complexity: bool = True,
) -> tuple[list[pd.Series], list[str]]:
    res: list[pd.Series] = []
    names: list[str] = []

    means = df.mean()
    medians = df.median()
    stds = df.std()
    skews = df.skew()
    kurts = df.kurt()

    res.extend([means, medians, stds, skews, kurts])
    names.extend(["mean", "median", "std", "skew", "kurt"])

    if with_max_min:
        maxs = df.max()
        mins = df.min()
        ranges = maxs - mins
        res.extend([maxs, mins, ranges])
        names.extend(["max", "min", "range"])

    if with_percentiles:
        p5s = df.quantile(0.05)
        p25s = df.quantile(0.25)
        p75s = df.quantile(0.75)
        p95s = df.quantile(0.95)
        iqrs = p75s - p25s
        cvs = stds / (means.abs() + 1e-8)
        res.extend([p5s, p25s, p75s, p95s, iqrs, cvs])
        names.extend(["p5", "p25", "p75", "p95", "iqr", "cv"])

    n_lags = min(max(1, with_lag_autocorr), 3)
    reset_df = df.reset_index(drop=True)
    for lag in range(1, n_lags + 1):
        ac = corrwith(reset_df, reset_df.shift(lag), 0, use_single_thread=True)
        res.append(ac)
        names.append(f"autocorr{lag}")
        res.append(ac.abs())
        names.append(f"autocorr{lag}_abs")

    trends = trend_2d(df.to_numpy(float), 0)
    trends_series = pd.Series(trends, index=df.columns)
    res.append(trends_series)
    names.append("trend")

    if with_period_compare:
        n_rows = len(df)
        split_point = n_rows // 3
        first_period_means = df.iloc[:split_point].mean()
        last_period_means = df.iloc[-split_point:].mean()
        period_diffs = last_period_means - first_period_means
        period_ratios = last_period_means / (first_period_means.abs() + 1e-8)
        res.extend([period_diffs, period_ratios])
        names.extend(["period_diff", "period_ratio"])

    if with_threshold_counts:
        p90s = df.quantile(0.90)
        p10s = df.quantile(0.10)
        mean_above_p90 = df[df > p90s].mean().fillna(0)
        mean_below_p10 = df[df < p10s].mean().fillna(0)
        res.extend([mean_above_p90, mean_below_p10])
        names.extend(["mean_above_p90", "mean_below_p10"])

    if with_abs:
        res.extend([means.abs(), medians.abs(), stds.abs(), skews.abs(), kurts.abs()])
        names.extend(["mean_abs", "median_abs", "std_abs", "skew_abs", "kurt_abs"])
        if with_max_min:
            res.extend([maxs.abs(), mins.abs()])
            names.extend(["max_abs", "min_abs"])
        if with_percentiles:
            res.extend([p5s.abs(), p25s.abs(), p75s.abs(), p95s.abs(), iqrs.abs(), cvs.abs()])
            names.extend(["p5_abs", "p25_abs", "p75_abs", "p95_abs", "iqr_abs", "cv_abs"])
        res.append(trends_series.abs())
        names.append("trend_abs")
        if with_period_compare:
            res.extend([period_diffs.abs(), period_ratios.abs()])
            names.extend(["period_diff_abs", "period_ratio_abs"])

    if with_lyapunov_exponent:
        res.append(df.apply(_calc_lyapunov))
        names.append("lyapunov")

    if with_complexity:
        res.extend(
            [
                df.apply(_calc_lz_complexity),
                df.apply(_calc_entropy),
                df.apply(_calc_max_range_product),
            ]
        )
        names.extend(["lz_complexity", "entropy_1d", "max_range_product"])

    return res, names


def get_features_factors(
    df: pd.DataFrame,
    with_abs: bool = False,
    with_max_min: bool = False,
    with_corr: bool = True,
    with_percentiles: bool = True,
    with_lag_autocorr: int = 1,
    with_threshold_counts: bool = True,
    with_period_compare: bool = True,
    with_lyapunov_exponent: bool = True,
    with_complexity: bool = True,
    append_for_corr: pd.DataFrame | None = None,
) -> tuple[list[float], list[str]]:
    means = df.mean()
    medians = df.median()
    stds = df.std()
    skews = df.skew()
    kurts = df.kurt()

    if with_abs:
        means_abs = means.abs()
        medians_abs = medians.abs()
        stds_abs = stds.abs()
        skews_abs = skews.abs()
        kurts_abs = kurts.abs()

    if with_max_min:
        maxs = df.max()
        mins = df.min()
        ranges = maxs - mins
        if with_abs:
            maxs_abs = maxs.abs()
            mins_abs = mins.abs()

    if with_percentiles:
        p5s = df.quantile(0.05)
        p25s = df.quantile(0.25)
        p75s = df.quantile(0.75)
        p95s = df.quantile(0.95)
        iqrs = p75s - p25s
        cvs = stds / (means.abs() + 1e-8)
        if with_abs:
            p5s_abs = p5s.abs()
            p25s_abs = p25s.abs()
            p75s_abs = p75s.abs()
            p95s_abs = p95s.abs()
            iqrs_abs = iqrs.abs()
            cvs_abs = cvs.abs()

    n_lags = min(max(1, with_lag_autocorr), 3)
    autocorrs: list[pd.Series] = []
    autocorrs_abs: list[pd.Series] = []
    reset_df = df.reset_index(drop=True)
    for lag in range(1, n_lags + 1):
        ac = corrwith(reset_df, reset_df.shift(lag), 0, use_single_thread=True)
        autocorrs.append(ac)
        autocorrs_abs.append(ac.abs())

    trends = trend_2d(df.to_numpy(float), 0)
    if with_abs:
        trends_abs = [abs(i) for i in trends]

    if with_period_compare:
        n_rows = len(df)
        split_point = n_rows // 3
        first_period_means = df.iloc[:split_point].mean()
        last_period_means = df.iloc[-split_point:].mean()
        period_diffs = last_period_means - first_period_means
        period_ratios = last_period_means / (first_period_means.abs() + 1e-8)
        if with_abs:
            period_diffs_abs = period_diffs.abs()
            period_ratios_abs = period_ratios.abs()

    if with_threshold_counts:
        p90s = df.quantile(0.90)
        p10s = df.quantile(0.10)
        mean_above_p90 = df[df > p90s].mean().fillna(0)
        mean_below_p10 = df[df < p10s].mean().fillna(0)

    res: list[float] = []
    names: list[str] = []
    col_names = df.columns.tolist()

    def append_results(series_list: list[pd.Series], name_suffixes: list[str]) -> None:
        for series, suffix in zip(series_list, name_suffixes):
            res.extend(series.tolist())
            names.extend([f"{col}_{suffix}" for col in col_names])

    append_results([means, medians, stds, skews, kurts], ["mean", "median", "std", "skew", "kurt"])

    if with_max_min:
        append_results([maxs, mins, ranges], ["max", "min", "range"])

    if with_percentiles:
        append_results([p5s, p25s, p75s, p95s, iqrs, cvs], ["p5", "p25", "p75", "p95", "iqr", "cv"])

    for lag_idx, (ac, ac_abs) in enumerate(zip(autocorrs, autocorrs_abs), start=1):
        append_results([ac, ac_abs], [f"autocorr{lag_idx}", f"autocorr{lag_idx}_abs"])

    res.extend(trends)
    names.extend([f"{col}_trend" for col in col_names])

    if with_period_compare:
        append_results([period_diffs, period_ratios], ["period_diff", "period_ratio"])

    if with_threshold_counts:
        append_results([mean_above_p90, mean_below_p10], ["mean_above_p90", "mean_below_p10"])

    if with_corr:
        df0 = pd.concat([df, append_for_corr], axis=1) if append_for_corr is not None else df.copy()
        corrs_matrix = fast_correlation_matrix_v2_df(df0, max_workers=1)
        n = corrs_matrix.shape[0]
        i_idx, j_idx = np.triu_indices(n, 1)
        row_names = corrs_matrix.index[i_idx]
        col_names_corr = corrs_matrix.columns[j_idx]
        corr_values = corrs_matrix.to_numpy()[i_idx, j_idx]

        corr_names = [f"{row}_corr_{col}" for row, col in zip(row_names, col_names_corr)]
        res.extend(corr_values.tolist())
        names.extend(corr_names)

        if with_abs:
            res.extend(np.abs(corr_values).tolist())
            names.extend([f"{name}_abs" for name in corr_names])

    if with_abs:
        append_results(
            [means_abs, medians_abs, stds_abs, skews_abs, kurts_abs],
            ["mean_abs", "median_abs", "std_abs", "skew_abs", "kurt_abs"],
        )

        if with_max_min:
            append_results([maxs_abs, mins_abs], ["max_abs", "min_abs"])

        if with_percentiles:
            append_results(
                [p5s_abs, p25s_abs, p75s_abs, p95s_abs, iqrs_abs, cvs_abs],
                ["p5_abs", "p25_abs", "p75_abs", "p95_abs", "iqr_abs", "cv_abs"],
            )

        res.extend(trends_abs)
        names.extend([f"{col}_trend_abs" for col in col_names])

        if with_period_compare:
            append_results([period_diffs_abs, period_ratios_abs], ["period_diff_abs", "period_ratio_abs"])

    if with_lyapunov_exponent:
        append_results([df.apply(_calc_lyapunov)], ["lyapunov"])

    if with_complexity:
        append_results(
            [
                df.apply(_calc_lz_complexity),
                df.apply(_calc_entropy),
                df.apply(_calc_max_range_product),
            ],
            ["lz_complexity", "entropy_1d", "max_range_product"],
        )

    return res, names


def get_features_names(
    col_names: list,
    with_abs: bool = False,
    with_max_min: bool = False,
    with_corr: bool = True,
    with_percentiles: bool = True,
    with_lag_autocorr: int = 1,
    with_threshold_counts: bool = True,
    with_period_compare: bool = True,
    with_lyapunov_exponent: bool = True,
    with_complexity: bool = True,
    append_for_corr_cols: list | None = None,
) -> list[str]:
    names: list[str] = []

    def append_names(suffixes: list[str]) -> None:
        for suffix in suffixes:
            names.extend([f"{col}_{suffix}" for col in col_names])

    append_names(["mean", "median", "std", "skew", "kurt"])

    if with_max_min:
        append_names(["max", "min", "range"])

    if with_percentiles:
        append_names(["p5", "p25", "p75", "p95", "iqr", "cv"])

    n_lags = min(max(1, with_lag_autocorr), 3)
    for lag_num in range(1, n_lags + 1):
        append_names([f"autocorr{lag_num}", f"autocorr{lag_num}_abs"])

    names.extend([f"{col}_trend" for col in col_names])

    if with_period_compare:
        append_names(["period_diff", "period_ratio"])

    if with_threshold_counts:
        append_names(["mean_above_p90", "mean_below_p10"])

    if with_corr:
        all_corr_cols = list(col_names) + list(append_for_corr_cols) if append_for_corr_cols is not None else list(col_names)
        n = len(all_corr_cols)
        corr_names = []
        for i in range(n):
            for j in range(i + 1, n):
                corr_names.append(f"{all_corr_cols[i]}_corr_{all_corr_cols[j]}")
        names.extend(corr_names)

        if with_abs:
            names.extend([f"{name}_abs" for name in corr_names])

    if with_abs:
        append_names(["mean_abs", "median_abs", "std_abs", "skew_abs", "kurt_abs"])

        if with_max_min:
            append_names(["max_abs", "min_abs"])

        if with_percentiles:
            append_names(["p5_abs", "p25_abs", "p75_abs", "p95_abs", "iqr_abs", "cv_abs"])

        names.extend([f"{col}_trend_abs" for col in col_names])

        if with_period_compare:
            append_names(["period_diff_abs", "period_ratio_abs"])

    if with_lyapunov_exponent:
        append_names(["lyapunov"])

    if with_complexity:
        append_names(["lz_complexity", "entropy_1d", "max_range_product"])

    return names
