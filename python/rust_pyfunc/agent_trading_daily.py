from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .rust_pyfunc import (
    compute_agent_trading_features,
    compute_agent_trading_features_multi,
    get_agent_feature_names,
    get_agent_time_grid_feature_names,
    get_agent_param_axis_feature_names,
)


_QUANTILE_PCTS: NDArray[np.float64] = np.asarray([5.0, 25.0, 50.0, 75.0, 95.0], dtype=np.float64)
_QUANTILE_NAMES: List[str] = ["q05", "q25", "q50", "q75", "q95"]
_ENTROPY_BINS: int = 32
_AGG_OPS: List[str] = [
    "mean",
    "std",
    "skew",
    "kurt",
    "q05",
    "q25",
    "q50",
    "q75",
    "q95",
    "acf1",
    "acf5",
    "trend_coef",
    "trend_t",
    "entropy",
    "mdd_like",
]


def _to_2d_float_array(values: NDArray[np.float64]) -> NDArray[np.float64]:
    arr = np.asarray(values, dtype=np.float64)
    arr2d = arr.reshape(-1, 1) if arr.ndim == 1 else arr
    return np.where(np.isfinite(arr2d), arr2d, np.nan)


def _acf_lag(x: NDArray[np.float64], lag: int) -> float:
    valid = x[~np.isnan(x)]
    if valid.size <= lag + 1:
        return 0.0
    x0 = valid[:-lag]
    x1 = valid[lag:]
    x0c = x0 - x0.mean()
    x1c = x1 - x1.mean()
    den = np.sqrt(np.sum(x0c * x0c) * np.sum(x1c * x1c))
    return 0.0 if den <= 1e-12 else float(np.sum(x0c * x1c) / den)


def _trend_coef_and_t(x: NDArray[np.float64]) -> tuple[float, float]:
    idx = np.flatnonzero(~np.isnan(x))
    if idx.size < 3:
        return 0.0, 0.0
    y = x[idx]
    xv = idx.astype(np.float64)
    x_centered = xv - xv.mean()
    sxx = float(np.sum(x_centered * x_centered))
    if sxx <= 1e-12:
        return 0.0, 0.0
    y_centered = y - y.mean()
    slope = float(np.sum(x_centered * y_centered) / sxx)
    intercept = float(y.mean() - slope * xv.mean())
    resid = y - (slope * xv + intercept)
    dof = idx.size - 2
    if dof <= 0:
        return slope, 0.0
    sigma2 = float(np.sum(resid * resid) / dof)
    se = np.sqrt(sigma2 / sxx)
    t_val = 0.0 if se <= 1e-12 else float(slope / se)
    return slope, t_val


def _entropy_norm(x: NDArray[np.float64]) -> float:
    valid = x[np.isfinite(x)]
    if valid.size < 2:
        return 0.0
    lo = float(valid.min())
    hi = float(valid.max())
    span = hi - lo
    if span <= 1e-12:
        return 0.0
    bins_idx = np.minimum(
        ((valid - lo) / (span + 1e-12) * (_ENTROPY_BINS - 1)).astype(np.int64),
        _ENTROPY_BINS - 1,
    )
    hist = np.bincount(bins_idx, minlength=_ENTROPY_BINS).astype(np.float64)
    probs = hist[hist > 0.0] / valid.size
    ent = -np.sum(probs * np.log(probs))
    return float(ent / np.log(_ENTROPY_BINS))


def _mdd_like(x: NDArray[np.float64]) -> float:
    valid = x[~np.isnan(x)]
    if valid.size == 0:
        return 0.0
    running_peak = np.maximum.accumulate(valid)
    return float(np.max(running_peak - valid))


def _aggregate_matrix_to_scalars(
    matrix: NDArray[np.float64],
    column_names: List[str],
    level: str,
    topic: str,
) -> Dict[str, float]:
    arr = _to_2d_float_array(matrix)
    n_rows, n_cols = arr.shape
    if len(column_names) != n_cols:
        return {}

    cnt = np.sum(~np.isnan(arr), axis=0).astype(np.float64)
    safe_cnt = np.where(cnt > 0.0, cnt, 1.0)

    mean = np.nansum(arr, axis=0) / safe_cnt
    centered = np.where(np.isnan(arr), 0.0, arr - mean)
    m2 = np.sum(centered * centered, axis=0) / safe_cnt
    std = np.sqrt(m2)
    m3 = np.sum(centered * centered * centered, axis=0) / safe_cnt
    m4 = np.sum(centered * centered * centered * centered, axis=0) / safe_cnt

    skew = np.where(std > 1e-12, m3 / (std * std * std), 0.0)
    kurt = np.where(std > 1e-12, m4 / (std * std * std * std) - 3.0, 0.0)

    q = np.nanpercentile(arr, _QUANTILE_PCTS, axis=0) if n_rows > 0 else np.zeros((5, n_cols))
    q = np.nan_to_num(q, nan=0.0, posinf=0.0, neginf=0.0)

    result: Dict[str, float] = {}
    for j, metric in enumerate(column_names):
        prefix = f"FDAY_{level}_{topic}_{metric}"

        result[f"{prefix}_mean"] = float(np.nan_to_num(mean[j], nan=0.0, posinf=0.0, neginf=0.0))
        result[f"{prefix}_std"] = float(np.nan_to_num(std[j], nan=0.0, posinf=0.0, neginf=0.0))
        result[f"{prefix}_skew"] = float(np.nan_to_num(skew[j], nan=0.0, posinf=0.0, neginf=0.0))
        result[f"{prefix}_kurt"] = float(np.nan_to_num(kurt[j], nan=0.0, posinf=0.0, neginf=0.0))

        for qi, qn in enumerate(_QUANTILE_NAMES):
            result[f"{prefix}_{qn}"] = float(q[qi, j])

        col = arr[:, j]
        result[f"{prefix}_acf1"] = _acf_lag(col, 1)
        result[f"{prefix}_acf5"] = _acf_lag(col, 5)
        trend_coef, trend_t = _trend_coef_and_t(col)
        result[f"{prefix}_trend_coef"] = trend_coef
        result[f"{prefix}_trend_t"] = trend_t
        result[f"{prefix}_entropy"] = _entropy_norm(col)
        result[f"{prefix}_mdd_like"] = _mdd_like(col)

    return result


def _expand_daily_columns(level: str, topic: str, metrics: List[str]) -> List[str]:
    return [f"FDAY_{level}_{topic}_{metric}_{op}" for metric in metrics for op in _AGG_OPS]


def get_agent_daily_feature_column_groups_for_get_features_factors(
    include_timestamp_metrics: bool = False,
) -> Dict[str, List[str]]:
    event_metrics = list(get_agent_feature_names())
    grid_metrics = list(get_agent_time_grid_feature_names())
    param_metrics = list(get_agent_param_axis_feature_names())

    if not include_timestamp_metrics:
        event_metrics = [m for m in event_metrics if m != "timestamp"]
        grid_metrics = [m for m in grid_metrics if m != "grid_timestamp"]
        param_metrics = [m for m in param_metrics if m != "timestamp"]

    l0_event = _expand_daily_columns(level="L0", topic="event", metrics=event_metrics)
    l1_grid = _expand_daily_columns(level="L1", topic="grid", metrics=grid_metrics)
    l8_param = _expand_daily_columns(level="L8", topic="param", metrics=param_metrics)

    return {
        "L0_event": l0_event,
        "L1_grid": l1_grid,
        "L8_param": l8_param,
        "all": l0_event + l1_grid + l8_param,
    }


def pack_agent_daily_feature_table_for_get_features_factors(
    daily_table: pd.DataFrame,
    include_levels: List[str] | None = None,
    include_timestamp_metrics: bool = False,
) -> tuple[pd.DataFrame, Dict[str, List[str]]]:
    groups = get_agent_daily_feature_column_groups_for_get_features_factors(
        include_timestamp_metrics=include_timestamp_metrics
    )

    levels = ["L0", "L1", "L8"] if include_levels is None else include_levels
    selected_group_keys = [f"{lvl}_event" for lvl in levels if lvl == "L0"] + [
        f"{lvl}_grid" for lvl in levels if lvl == "L1"
    ] + [f"{lvl}_param" for lvl in levels if lvl == "L8"]

    selected_columns: List[str] = []
    for gk in selected_group_keys:
        selected_columns.extend(groups.get(gk, []))

    available_columns = [c for c in selected_columns if c in daily_table.columns]
    selected_df = daily_table.loc[:, available_columns]

    return selected_df, {
        "selected": available_columns,
        "L0_event": [c for c in groups["L0_event"] if c in daily_table.columns],
        "L1_grid": [c for c in groups["L1_grid"] if c in daily_table.columns],
        "L8_param": [c for c in groups["L8_param"] if c in daily_table.columns],
    }


def aggregate_agent_feature_triplets_to_daily_table(
    event_features: NDArray[np.float64],
    event_feature_names: List[str],
    time_grid_features: NDArray[np.float64],
    time_grid_feature_names: List[str],
    param_axis_features: NDArray[np.float64],
    param_axis_feature_names: List[str],
) -> pd.DataFrame:
    event_scalars = _aggregate_matrix_to_scalars(
        matrix=event_features,
        column_names=event_feature_names,
        level="L0",
        topic="event",
    )
    grid_scalars = _aggregate_matrix_to_scalars(
        matrix=time_grid_features,
        column_names=time_grid_feature_names,
        level="L1",
        topic="grid",
    )
    param_scalars = _aggregate_matrix_to_scalars(
        matrix=param_axis_features,
        column_names=param_axis_feature_names,
        level="L8",
        topic="param",
    )

    merged: Dict[str, float] = {}
    merged.update(event_scalars)
    merged.update(grid_scalars)
    merged.update(param_scalars)
    return pd.DataFrame([merged], dtype=np.float64)


def compute_agent_trading_daily_feature_table(
    market_timestamps: NDArray[np.int64],
    market_prices: NDArray[np.float64],
    market_volumes: NDArray[np.float64],
    market_turnovers: NDArray[np.float64],
    market_flags: NDArray[np.int32],
    bid_order_ids: NDArray[np.int64],
    ask_order_ids: NDArray[np.int64],
    agent_market_indices: NDArray[np.uint64],
    agent_directions: NDArray[np.int32],
    agent_volumes: NDArray[np.float64],
    window_ms: int = 10000,
    grid_ms: int = 1000,
) -> pd.DataFrame:
    (
        event_features,
        event_feature_names,
        time_grid_features,
        time_grid_feature_names,
        param_axis_features,
        param_axis_feature_names,
    ) = compute_agent_trading_features(
        market_timestamps,
        market_prices,
        market_volumes,
        market_turnovers,
        market_flags,
        bid_order_ids,
        ask_order_ids,
        agent_market_indices,
        agent_directions,
        agent_volumes,
        window_ms=window_ms,
        grid_ms=grid_ms,
    )
    return aggregate_agent_feature_triplets_to_daily_table(
        event_features=event_features,
        event_feature_names=event_feature_names,
        time_grid_features=time_grid_features,
        time_grid_feature_names=time_grid_feature_names,
        param_axis_features=param_axis_features,
        param_axis_feature_names=param_axis_feature_names,
    )


def compute_agent_trading_daily_feature_table_multi(
    market_timestamps: NDArray[np.int64],
    market_prices: NDArray[np.float64],
    market_volumes: NDArray[np.float64],
    market_turnovers: NDArray[np.float64],
    market_flags: NDArray[np.int32],
    bid_order_ids: NDArray[np.int64],
    ask_order_ids: NDArray[np.int64],
    all_agent_market_indices: List[NDArray[np.uint64]],
    all_agent_directions: List[NDArray[np.int32]],
    all_agent_volumes: List[NDArray[np.float64]],
    target_agent_idx: int,
    window_ms: int = 10000,
    grid_ms: int = 1000,
    param_window_ms: int = 2000,
    all_agent_params: List[float] | None = None,
) -> pd.DataFrame:
    (
        event_features,
        event_feature_names,
        time_grid_features,
        time_grid_feature_names,
        param_axis_features,
        param_axis_feature_names,
    ) = compute_agent_trading_features_multi(
        market_timestamps,
        market_prices,
        market_volumes,
        market_turnovers,
        market_flags,
        bid_order_ids,
        ask_order_ids,
        all_agent_market_indices,
        all_agent_directions,
        all_agent_volumes,
        target_agent_idx=target_agent_idx,
        window_ms=window_ms,
        grid_ms=grid_ms,
        param_window_ms=param_window_ms,
        all_agent_params=all_agent_params,
    )
    return aggregate_agent_feature_triplets_to_daily_table(
        event_features=event_features,
        event_feature_names=event_feature_names,
        time_grid_features=time_grid_features,
        time_grid_feature_names=time_grid_feature_names,
        param_axis_features=param_axis_features,
        param_axis_feature_names=param_axis_feature_names,
    )


def compute_agent_trading_daily_feature_table_for_get_features_factors(
    market_timestamps: NDArray[np.int64],
    market_prices: NDArray[np.float64],
    market_volumes: NDArray[np.float64],
    market_turnovers: NDArray[np.float64],
    market_flags: NDArray[np.int32],
    bid_order_ids: NDArray[np.int64],
    ask_order_ids: NDArray[np.int64],
    agent_market_indices: NDArray[np.uint64],
    agent_directions: NDArray[np.int32],
    agent_volumes: NDArray[np.float64],
    window_ms: int = 10000,
    grid_ms: int = 1000,
    include_levels: List[str] | None = None,
    include_timestamp_metrics: bool = False,
) -> tuple[pd.DataFrame, Dict[str, List[str]]]:
    daily_table = compute_agent_trading_daily_feature_table(
        market_timestamps=market_timestamps,
        market_prices=market_prices,
        market_volumes=market_volumes,
        market_turnovers=market_turnovers,
        market_flags=market_flags,
        bid_order_ids=bid_order_ids,
        ask_order_ids=ask_order_ids,
        agent_market_indices=agent_market_indices,
        agent_directions=agent_directions,
        agent_volumes=agent_volumes,
        window_ms=window_ms,
        grid_ms=grid_ms,
    )
    return pack_agent_daily_feature_table_for_get_features_factors(
        daily_table=daily_table,
        include_levels=include_levels,
        include_timestamp_metrics=include_timestamp_metrics,
    )


def compute_agent_trading_daily_feature_table_multi_for_get_features_factors(
    market_timestamps: NDArray[np.int64],
    market_prices: NDArray[np.float64],
    market_volumes: NDArray[np.float64],
    market_turnovers: NDArray[np.float64],
    market_flags: NDArray[np.int32],
    bid_order_ids: NDArray[np.int64],
    ask_order_ids: NDArray[np.int64],
    all_agent_market_indices: List[NDArray[np.uint64]],
    all_agent_directions: List[NDArray[np.int32]],
    all_agent_volumes: List[NDArray[np.float64]],
    target_agent_idx: int,
    window_ms: int = 10000,
    grid_ms: int = 1000,
    param_window_ms: int = 2000,
    all_agent_params: List[float] | None = None,
    include_levels: List[str] | None = None,
    include_timestamp_metrics: bool = False,
) -> tuple[pd.DataFrame, Dict[str, List[str]]]:
    daily_table = compute_agent_trading_daily_feature_table_multi(
        market_timestamps=market_timestamps,
        market_prices=market_prices,
        market_volumes=market_volumes,
        market_turnovers=market_turnovers,
        market_flags=market_flags,
        bid_order_ids=bid_order_ids,
        ask_order_ids=ask_order_ids,
        all_agent_market_indices=all_agent_market_indices,
        all_agent_directions=all_agent_directions,
        all_agent_volumes=all_agent_volumes,
        target_agent_idx=target_agent_idx,
        window_ms=window_ms,
        grid_ms=grid_ms,
        param_window_ms=param_window_ms,
        all_agent_params=all_agent_params,
    )
    return pack_agent_daily_feature_table_for_get_features_factors(
        daily_table=daily_table,
        include_levels=include_levels,
        include_timestamp_metrics=include_timestamp_metrics,
    )
