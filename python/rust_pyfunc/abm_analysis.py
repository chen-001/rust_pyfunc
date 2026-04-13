from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Tuple

from .rust_pyfunc import compute_agent_trading_features_multi
from .agent_trading_daily import compute_agent_trading_daily_feature_table_multi
from .trading_data_utils import adjust_afternoon, read_trade, get_features_factors


def _pair_window_same_direction_rate(
    ts_a: np.ndarray,
    dir_a: np.ndarray,
    ts_b: np.ndarray,
    dir_b: np.ndarray,
    window_ns: int,
    forward_only: bool,
) -> float:
    n_a = ts_a.shape[0]
    n_b = ts_b.shape[0]
    if n_a == 0 or n_b == 0:
        return 0.0

    buy_prefix = np.zeros(n_b + 1, dtype=np.int64)
    sell_prefix = np.zeros(n_b + 1, dtype=np.int64)
    buy_prefix[1:] = np.cumsum((dir_b == 66).astype(np.int64))
    sell_prefix[1:] = np.cumsum((dir_b == 83).astype(np.int64))

    left = 0
    right = 0
    hit = 0

    for i in range(n_a):
        t = ts_a[i]
        if forward_only:
            start = t + 1
            end = t + window_ns
        else:
            half = window_ns // 2
            start = t - half
            end = t + half

        while left < n_b and ts_b[left] < start:
            left += 1
        if right < left:
            right = left
        while right < n_b and ts_b[right] <= end:
            right += 1

        if left >= right:
            continue

        if dir_a[i] == 66:
            cnt = buy_prefix[right] - buy_prefix[left]
        elif dir_a[i] == 83:
            cnt = sell_prefix[right] - sell_prefix[left]
        else:
            cnt = 0

        if cnt > 0:
            hit += 1

    return float(hit / n_a)


def _extract_all_agent_arrays(sim_results: List[Dict[str, Any]]) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    all_agent_market_indices: List[np.ndarray] = [
        r["market_indices"].astype(np.uint64) for r in sim_results
    ]
    all_agent_directions: List[np.ndarray] = [
        r["directions"].astype(np.int32) for r in sim_results
    ]
    all_agent_volumes: List[np.ndarray] = [
        r["volumes"].astype(np.float64) for r in sim_results
    ]
    return all_agent_market_indices, all_agent_directions, all_agent_volumes


def _vector_df_to_scalar(df: pd.DataFrame, factor_cfg: Dict[str, Any]) -> Tuple[List[float], List[str]]:
    vals, names = get_features_factors(
        df,
        with_abs=factor_cfg["with_abs"],
        with_max_min=factor_cfg["with_max_min"],
        with_corr=factor_cfg["with_corr"],
        with_percentiles=factor_cfg["with_percentiles"],
        with_lag_autocorr=factor_cfg["with_lag_autocorr"],
        with_threshold_counts=factor_cfg["with_threshold_counts"],
        with_period_compare=factor_cfg["with_period_compare"],
        with_lyapunov_exponent=factor_cfg["with_lyapunov_exponent"],
        with_complexity=factor_cfg["with_complexity"],
    )
    return vals, names


def _compute_target_two_part(
    market: Dict[str, np.ndarray],
    all_agent_market_indices: List[np.ndarray],
    all_agent_directions: List[np.ndarray],
    all_agent_volumes: List[np.ndarray],
    all_agent_params: List[float],
    target_agent_idx: int,
    rust_cfg: Dict[str, int],
    factor_cfg: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    event_arr, event_cols, grid_arr, grid_cols, param_arr, param_cols = compute_agent_trading_features_multi(
        market["market_timestamps"],
        market["market_prices"],
        market["market_volumes"],
        market["market_turnovers"],
        market["market_flags"],
        market["bid_order_ids"],
        market["ask_order_ids"],
        all_agent_market_indices,
        all_agent_directions,
        all_agent_volumes,
        target_agent_idx=target_agent_idx,
        window_ms=rust_cfg["window_ms"],
        grid_ms=rust_cfg["grid_ms"],
        param_window_ms=rust_cfg["param_window_ms"],
        all_agent_params=all_agent_params,
    )

    event_df = pd.DataFrame(event_arr, columns=[f"EV_{c}" for c in event_cols]).replace([np.inf, -np.inf], np.nan)
    grid_df = pd.DataFrame(grid_arr, columns=[f"GRID_{c}" for c in grid_cols]).replace([np.inf, -np.inf], np.nan)
    param_df = pd.DataFrame(param_arr, columns=[f"PARAM_{c}" for c in param_cols]).replace([np.inf, -np.inf], np.nan)

    ev_vals, ev_names = _vector_df_to_scalar(event_df, factor_cfg)
    grid_vals, grid_names = _vector_df_to_scalar(grid_df, factor_cfg)
    param_vals, param_names = _vector_df_to_scalar(param_df, factor_cfg)

    vector_to_scalar_df = pd.DataFrame(
        [
            {
                **{f"V2S_{k}": v for k, v in zip(ev_names, ev_vals)},
                **{f"V2S_{k}": v for k, v in zip(grid_names, grid_vals)},
                **{f"V2S_{k}": v for k, v in zip(param_names, param_vals)},
            }
        ]
    )

    direct_scalar_df = compute_agent_trading_daily_feature_table_multi(
        market_timestamps=market["market_timestamps"],
        market_prices=market["market_prices"],
        market_volumes=market["market_volumes"],
        market_turnovers=market["market_turnovers"],
        market_flags=market["market_flags"],
        bid_order_ids=market["bid_order_ids"],
        ask_order_ids=market["ask_order_ids"],
        all_agent_market_indices=all_agent_market_indices,
        all_agent_directions=all_agent_directions,
        all_agent_volumes=all_agent_volumes,
        target_agent_idx=target_agent_idx,
        window_ms=rust_cfg["window_ms"],
        grid_ms=rust_cfg["grid_ms"],
        param_window_ms=rust_cfg["param_window_ms"],
        all_agent_params=all_agent_params,
    )
    return vector_to_scalar_df, direct_scalar_df


def _calc_global_interaction_features(
    sim_results: List[Dict[str, Any]],
    market_timestamps: np.ndarray,
    lookback_ms_list: List[int],
    per_agent_direct_scalar_df: pd.DataFrame,
    sync_window_ms: int,
    lead_window_ms: int,
    ns_per_sec: int,
) -> pd.DataFrame:
    def _direction_sign(directions: np.ndarray) -> np.ndarray:
        return np.where(directions == 66, 1.0, np.where(directions == 83, -1.0, 0.0))
    n_agents = len(sim_results)
    trade_counts = np.asarray([int(r["n_trades"]) for r in sim_results], dtype=np.float64)
    buy_ratios = np.asarray(
        [
            float((r["directions"] == 66).sum() / max(1, len(r["directions"])))
            for r in sim_results
        ],
        dtype=np.float64,
    )
    signed_flows = np.asarray(
        [
            float((_direction_sign(r["directions"].astype(np.int32)) * r["volumes"].astype(np.float64)).sum())
            for r in sim_results
        ],
        dtype=np.float64,
    )

    per_agent_ts = [
        market_timestamps[r["market_indices"].astype(np.uint64)].astype(np.int64)
        for r in sim_results
    ]
    per_agent_dir = [r["directions"].astype(np.int32) for r in sim_results]

    sync_vals: List[float] = []
    lead_vals: List[float] = []
    for i in range(n_agents):
        for j in range(i + 1, n_agents):
            sync_ij = _pair_window_same_direction_rate(
                per_agent_ts[i],
                per_agent_dir[i],
                per_agent_ts[j],
                per_agent_dir[j],
                window_ns=sync_window_ms * 1_000_000,
                forward_only=False,
            )
            sync_ji = _pair_window_same_direction_rate(
                per_agent_ts[j],
                per_agent_dir[j],
                per_agent_ts[i],
                per_agent_dir[i],
                window_ns=sync_window_ms * 1_000_000,
                forward_only=False,
            )
            sync_vals.extend([sync_ij, sync_ji])

            lead_ij = _pair_window_same_direction_rate(
                per_agent_ts[i],
                per_agent_dir[i],
                per_agent_ts[j],
                per_agent_dir[j],
                window_ns=lead_window_ms * 1_000_000,
                forward_only=True,
            )
            lead_ji = _pair_window_same_direction_rate(
                per_agent_ts[j],
                per_agent_dir[j],
                per_agent_ts[i],
                per_agent_dir[i],
                window_ns=lead_window_ms * 1_000_000,
                forward_only=True,
            )
            lead_vals.append(abs(lead_ij - lead_ji))

    active_count_by_sec: Dict[int, int] = {}
    for ts in per_agent_ts:
        sec_bins = np.unique(ts // ns_per_sec)
        for b in sec_bins:
            active_count_by_sec[int(b)] = active_count_by_sec.get(int(b), 0) + 1
    active_agents = np.asarray(list(active_count_by_sec.values()), dtype=np.float64)

    params = np.asarray(lookback_ms_list, dtype=np.float64)
    metric_col = "FDAY_L0_event_cumulative_return_mean"
    if metric_col in per_agent_direct_scalar_df.columns:
        y = per_agent_direct_scalar_df[metric_col].astype(np.float64).values
        x_mean = params.mean()
        y_mean = y.mean()
        num = float(np.sum((params - x_mean) * (y - y_mean)))
        den = float(np.sum((params - x_mean) ** 2))
        param_ret_slope = 0.0 if den <= 1e-12 else num / den
    else:
        param_ret_slope = 0.0

    return pd.DataFrame(
        [
            {
                "GLOB_n_agents": float(n_agents),
                "GLOB_trade_count_mean": float(trade_counts.mean()),
                "GLOB_trade_count_std": float(trade_counts.std()),
                "GLOB_buy_ratio_mean": float(buy_ratios.mean()),
                "GLOB_buy_ratio_std": float(buy_ratios.std()),
                "GLOB_signed_flow_dispersion": float(signed_flows.std()),
                "GLOB_sync_mean": float(np.mean(sync_vals) if len(sync_vals) else 0.0),
                "GLOB_sync_std": float(np.std(sync_vals) if len(sync_vals) else 0.0),
                "GLOB_sync_max": float(np.max(sync_vals) if len(sync_vals) else 0.0),
                "GLOB_lead_lag_asym_mean": float(np.mean(lead_vals) if len(lead_vals) else 0.0),
                "GLOB_active_agents_mean": float(active_agents.mean() if active_agents.size else 0.0),
                "GLOB_active_agents_std": float(active_agents.std() if active_agents.size else 0.0),
                "GLOB_active_agents_max": float(active_agents.max() if active_agents.size else 0.0),
                "GLOB_param_return_slope": float(param_ret_slope),
            }
        ]
    )


def _df_tolist(df: pd.DataFrame, drop_cols: List[str]) -> tuple[list]:
    df0 = df.drop(columns=drop_cols)
    names0 = df0.columns.tolist()
    res = []
    names = []
    for c in range(int(df.shape[0])):
        row = df0.iloc[c, :].tolist()
        res.append(row)
        agent_name = df.agent_name.iloc[c]
        names.append([agent_name + "_" + n for n in names0])
    res = sum(res, [])
    names = sum(names, [])
    return res, names


def abm_step_two(
    code: str,
    date: int,
    func,
    MOMENTUM_AGENT_SPEC: Dict[str, Any],
    RUST_FEATURE_CONFIG: Dict[str, int],
    VECTOR_TO_SCALAR_CONFIG: Dict[str, Any],
    GLOBAL_FEATURE_CONFIG: Dict[str, Any],
    NS_PER_SEC: int = 1_000_000_000,
) -> Tuple[List[float], List[str], Dict[str, Any]]:

    trade_data = adjust_afternoon(read_trade(code, date))
    market = {
        "market_timestamps": trade_data["exchtime"].astype(np.int64).values,
        "market_prices": trade_data["price"].astype(np.float64).values,
        "market_volumes": trade_data["volume"].astype(np.float64).values,
        "market_turnovers": trade_data["turnover"].astype(np.float64).values,
        "market_flags": trade_data["flag"].astype(np.int32).values,
        "bid_order_ids": trade_data["bid_order"].astype(np.int64).values,
        "ask_order_ids": trade_data["ask_order"].astype(np.int64).values,
    }

    sim_results = func(market, MOMENTUM_AGENT_SPEC)
    all_agent_market_indices, all_agent_directions, all_agent_volumes = _extract_all_agent_arrays(sim_results)
    try:
        all_agent_params: List[float] = [float(v) for v in MOMENTUM_AGENT_SPEC["lookback_ms_list"]]
    except Exception:
        all_agent_params = None

    per_agent_v2s_rows: List[pd.DataFrame] = []
    per_agent_direct_rows: List[pd.DataFrame] = []
    for idx in range(len(sim_results)):
        v2s_df, direct_df = _compute_target_two_part(
            market=market,
            all_agent_market_indices=all_agent_market_indices,
            all_agent_directions=all_agent_directions,
            all_agent_volumes=all_agent_volumes,
            all_agent_params=all_agent_params,
            target_agent_idx=idx,
            rust_cfg=RUST_FEATURE_CONFIG,
            factor_cfg=VECTOR_TO_SCALAR_CONFIG,
        )
        v2s_df.insert(0, "agent_name", MOMENTUM_AGENT_SPEC["agent_names"][idx])
        v2s_df.insert(0, "agent_idx", idx)
        per_agent_v2s_rows.append(v2s_df)

        direct_df.insert(0, "agent_name", MOMENTUM_AGENT_SPEC["agent_names"][idx])
        direct_df.insert(0, "agent_idx", idx)
        per_agent_direct_rows.append(direct_df)

    per_agent_vector_to_scalar_df = pd.concat(per_agent_v2s_rows, axis=0, ignore_index=True)
    per_agent_direct_scalar_df = pd.concat(per_agent_direct_rows, axis=0, ignore_index=True)

    global_interaction_df = _calc_global_interaction_features(
        sim_results=sim_results,
        market_timestamps=market["market_timestamps"],
        lookback_ms_list=MOMENTUM_AGENT_SPEC.get("lookback_ms_list", list(range(len(MOMENTUM_AGENT_SPEC["agent_names"])))),
        per_agent_direct_scalar_df=per_agent_direct_scalar_df,
        sync_window_ms=GLOBAL_FEATURE_CONFIG["sync_window_ms"],
        lead_window_ms=GLOBAL_FEATURE_CONFIG["lead_window_ms"],
        ns_per_sec=NS_PER_SEC,
    ).drop(columns=GLOBAL_FEATURE_CONFIG["drop_cols"])

    sim_summary_df = pd.DataFrame(
        {
            "agent_idx": np.arange(len(sim_results), dtype=np.int64),
            "agent_name": MOMENTUM_AGENT_SPEC["agent_names"],
            "lookback_ms": MOMENTUM_AGENT_SPEC.get("lookback_ms_list", list(range(len(MOMENTUM_AGENT_SPEC["agent_names"])))),
            "n_trades": [int(r["n_trades"]) for r in sim_results],
            "total_buy_volume": [float(r["total_buy_volume"]) for r in sim_results],
            "total_sell_volume": [float(r["total_sell_volume"]) for r in sim_results],
            "final_position": [float(r["final_position"]) for r in sim_results],
        }
    )

    res_dict = {
        "per_agent_vector_to_scalar_df": per_agent_vector_to_scalar_df,
        "per_agent_direct_scalar_df": per_agent_direct_scalar_df,
        "global_interaction_df": global_interaction_df,
        "sim_summary_df": sim_summary_df,
        "sim_results": sim_results,
    }

    res_sim_summary, names_sim_summary = _df_tolist(res_dict["sim_summary_df"], ["agent_idx", "agent_name", "lookback_ms"])
    res_per_agent, names_per_agent = _df_tolist(res_dict["per_agent_vector_to_scalar_df"], ["agent_idx", "agent_name"])
    res_global_interaction, names_global_interaction = (
        res_dict["global_interaction_df"].iloc[0, :].tolist(),
        res_dict["global_interaction_df"].columns.tolist(),
    )
    res0 = res_sim_summary + res_per_agent + res_global_interaction
    names = names_sim_summary + names_per_agent + names_global_interaction
    return res0, names, res_dict


def abm_generate_names(agent_names: List[str]) -> List[str]:
    basic_suffixes = [
        '_n_trades',
        '_total_buy_volume',
        '_total_sell_volume',
        '_final_position'
    ]

    ev_metrics = [
        'timestamp', 'post_trade_price_trend', 'order_id_diff', 'pre_trade_count',
        'post_trade_count', 'price_deviation', 'pre_trade_volatility', 'post_trade_volatility',
        'sync_rate_same_direction', 'max_drawdown', 'max_drawdown_duration', 'profit_concentration',
        'gain_loss_ratio', 'avg_holding_duration', 'exposure_utilization', 'trading_density',
        'sharpe_ratio', 'sortino_ratio', 'calmar_ratio', 'cumulative_return', 'excess_return',
        'current_position', 'cumulative_volume', 'inter_trade_dt_ns', 'signed_volume',
        'direction_flip', 'run_length_same_direction', 'volume_participation',
        'impact_inst_1s', 'impact_inst_5s', 'impact_reversion_5s', 'burstiness_running'
    ]

    grid_metrics = [
        'grid_timestamp', 'trade_count', 'trade_density', 'signed_volume_sum',
        'buy_ratio', 'flip_rate', 'impact_inst_1s_mean', 'impact_inst_5s_mean',
        'position_last', 'cum_return_delta', 'cum_volume_delta'
    ]

    param_metrics = [
        'timestamp', 'crowding_index_same_direction', 'active_agent_ratio',
        'param_dispersion_trade_count', 'param_dispersion_signed_flow',
        'minority_score', 'lead_lag_score', 'param_slope_signed_flow',
        'param_curvature_signed_flow', 'param_monotonicity_signed_flow',
        'param_turning_points_signed_flow', 'response_nonlinearity', 'target_flow_rank'
    ]

    stats = [
        'mean', 'median', 'std', 'skew', 'kurt', 'p5', 'p25', 'p75', 'p95',
        'iqr', 'cv', 'autocorr1', 'autocorr1_abs', 'trend', 'period_diff',
        'period_ratio', 'lz_complexity', 'entropy_1d', 'max_range_product'
    ]

    glob_features = [
        'GLOB_trade_count_mean', 'GLOB_trade_count_std',
        'GLOB_buy_ratio_mean', 'GLOB_buy_ratio_std',
        'GLOB_signed_flow_dispersion',
        'GLOB_sync_mean', 'GLOB_sync_std', 'GLOB_sync_max',
        'GLOB_lead_lag_asym_mean',
        'GLOB_active_agents_mean', 'GLOB_active_agents_std',
        'GLOB_param_return_slope'
    ]

    names = []

    for agent in agent_names:
        for suffix in basic_suffixes:
            names.append(f'{agent}{suffix}')

    for agent in agent_names:
        for stat in stats:
            for metric in ev_metrics:
                names.append(f'{agent}_V2S_EV_{metric}_{stat}')
        for stat in stats:
            for metric in grid_metrics:
                names.append(f'{agent}_V2S_GRID_{metric}_{stat}')
        for stat in stats:
            for metric in param_metrics:
                names.append(f'{agent}_V2S_PARAM_{metric}_{stat}')

    names.extend(glob_features)

    return names
