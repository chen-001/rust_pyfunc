from __future__ import annotations

from functools import lru_cache
from typing import List, Tuple

import numpy as np
import pandas as pd


THEME_FEATURE_EXPANSION_MINUTE_FIELDS = [
    "open",
    "close",
    "high",
    "low",
    "amount",
    "volume",
    "act_buy_amount_sum",
    "act_sell_amount_sum",
    "act_buy_vol_sum",
    "act_sell_vol_sum",
    "act_buy_count_sum",
    "act_sell_count_sum",
    "up_tick_count",
    "down_tick_count",
    "bid_size_1_mean",
    "ask_size_1_mean",
    "bid_size_6_mean",
    "ask_size_6_mean",
    "bid_size_10_mean",
    "ask_size_10_mean",
    "bid_vwap_mean",
    "ask_vwap_mean",
    "ask_vwap3",
    "ask_vwap5",
    "ask_vwap10",
    "bid_vwap3",
    "bid_vwap5",
    "bid_vwap10",
    "bid_vol1",
    "ask_vol1",
    "bid_vol2",
    "ask_vol2",
    "bid_vol3",
    "ask_vol3",
]


THEME_FEATURE_VECTOR_DIMENSIONS = {
    "ret_vs_theme_path_8": 8,
    "flow_vs_theme_path_8": 8,
    "amt_vs_theme_path_8": 8,
    "depth_vs_theme_path_8": 8,
    "ret_dct_6": 6,
    "flow_dct_6": 6,
    "joint_latent_6": 6,
    "soft_assign_dists_5": 5,
    "soft_assign_probs_5": 5,
    "theme_gap_vector_3": 3,
    "open_close_ret_vs_theme_4": 4,
    "open_close_path_vs_theme_6": 6,
    "open_close_center_margin_4": 4,
    "flow_to_price_transmission_6": 6,
    "open_signal_close_confirm_4": 4,
    "capital_expression_pricing_confirmation_8": 8,
}


@lru_cache(maxsize=1)
def _get_dw():
    import design_whatever as dw

    return dw


def _get_trading_dates(start_date: int, end_date: int) -> List[int]:
    td = _get_dw().td
    start_pos = td.get_loc(start_date)
    end_pos = td.get_loc(end_date)
    return list(td.trading_days[start_pos : end_pos + 1])


def _last_n_trading_date(end_date: int, n_date: int) -> int:
    td = _get_dw().td
    pos = td.get_loc(end_date) - n_date
    return td.trading_days[pos]


def prepare_minute_data_for_theme_feature_expansion(
    date: int,
    lookback: int = 2,
) -> Tuple[np.ndarray, List[int], np.ndarray]:
    start_date = _last_n_trading_date(date, lookback)
    dates = _get_trading_dates(start_date, date)
    dw = _get_dw()
    all_data = {
        field: dw.read_minute_data(field, dates[0], dates[-1])
        for field in THEME_FEATURE_EXPANSION_MINUTE_FIELDS
    }

    symbols = None
    for d in dates:
        day_symbols = all_data["close"].loc[d].columns
        symbols = day_symbols if symbols is None else symbols.intersection(day_symbols)
    symbols = symbols.sort_values()

    n_days = len(dates)
    n_minutes = all_data["close"].loc[dates[0]].shape[0]
    n_stocks = len(symbols)
    n_fields = len(THEME_FEATURE_EXPANSION_MINUTE_FIELDS)
    minute_data = np.zeros((n_days, n_minutes, n_stocks, n_fields), dtype=np.float64)

    for d_idx, d in enumerate(dates):
        for f_idx, field in enumerate(THEME_FEATURE_EXPANSION_MINUTE_FIELDS):
            minute_data[d_idx, :, :, f_idx] = all_data[field].loc[d][symbols].values

    return minute_data, dates, symbols.values


def compute_theme_feature_expansion_from_minute_raw(
    minute_data: np.ndarray,
    k: int = 30,
    summary_dim: int = 17,
    n_threads: int = 8,
):
    import rust_pyfunc as rp

    return rp.compute_theme_feature_expansion_from_minute(
        minute_data=minute_data,
        k=k,
        summary_dim=summary_dim,
        n_threads=n_threads,
    )


def compute_theme_feature_expansion_from_date(
    date: int,
    lookback: int = 2,
    k: int = 30,
    summary_dim: int = 17,
    n_threads: int = 8,
):
    minute_data, dates, symbols = prepare_minute_data_for_theme_feature_expansion(
        date=date,
        lookback=lookback,
    )
    vector_arrays, scalar_arrays, vector_names, scalar_names = (
        compute_theme_feature_expansion_from_minute_raw(
            minute_data=minute_data,
            k=k,
            summary_dim=summary_dim,
            n_threads=n_threads,
        )
    )
    vector_frames = [
        pd.DataFrame(
            arr,
            index=symbols,
            columns=[f"{name}_{i}" for i in range(THEME_FEATURE_VECTOR_DIMENSIONS[name])],
        )
        for name, arr in zip(vector_names, vector_arrays)
    ]
    scalar_series = [
        pd.Series(arr, index=symbols, name=name)
        for name, arr in zip(scalar_names, scalar_arrays)
    ]
    return vector_frames, scalar_series, vector_names, scalar_names, symbols, dates
