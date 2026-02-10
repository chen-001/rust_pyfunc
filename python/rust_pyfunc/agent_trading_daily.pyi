from typing import Dict, List

import pandas as pd
import numpy as np
from numpy.typing import NDArray


def aggregate_agent_feature_triplets_to_daily_table(
    event_features: NDArray[np.float64],
    event_feature_names: List[str],
    time_grid_features: NDArray[np.float64],
    time_grid_feature_names: List[str],
    param_axis_features: NDArray[np.float64],
    param_axis_feature_names: List[str],
) -> pd.DataFrame: ...


def get_agent_daily_feature_column_groups_for_get_features_factors(
    include_timestamp_metrics: bool = False,
) -> Dict[str, List[str]]: ...


def pack_agent_daily_feature_table_for_get_features_factors(
    daily_table: pd.DataFrame,
    include_levels: List[str] | None = None,
    include_timestamp_metrics: bool = False,
) -> tuple[pd.DataFrame, Dict[str, List[str]]]: ...


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
) -> pd.DataFrame: ...


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
) -> pd.DataFrame: ...


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
) -> tuple[pd.DataFrame, Dict[str, List[str]]]: ...


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
) -> tuple[pd.DataFrame, Dict[str, List[str]]]: ...
