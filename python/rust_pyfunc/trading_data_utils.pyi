from __future__ import annotations

from typing import List, Tuple

import pandas as pd


def adjust_afternoon(df: pd.DataFrame, only_inday: int = 1) -> pd.DataFrame: ...


def read_trade(symbol: str, date: int, with_retreat: int = 0) -> pd.DataFrame: ...


def read_market(
    symbol: str,
    date: int,
    with_high_low_limited: int = 0,
) -> pd.DataFrame: ...


def read_market_pair(symbol: str, date: int) -> Tuple[pd.DataFrame, pd.DataFrame]: ...


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
) -> Tuple[List[pd.Series], List[str]]: ...


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
) -> Tuple[List[float], List[str]]: ...


def get_features_names(
    col_names: List[str],
    with_abs: bool = False,
    with_max_min: bool = False,
    with_corr: bool = True,
    with_percentiles: bool = True,
    with_lag_autocorr: int = 1,
    with_threshold_counts: bool = True,
    with_period_compare: bool = True,
    with_lyapunov_exponent: bool = True,
    with_complexity: bool = True,
    append_for_corr_cols: List[str] | None = None,
) -> List[str]: ...
