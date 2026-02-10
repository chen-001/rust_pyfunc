//! Agent交易特征计算模块 - Python绑定

use super::core::*;
use super::types::*;
use ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1};
use pyo3::prelude::*;
use std::cmp::Ordering;

const NS_PER_MS: i64 = 1_000_000;
const NS_PER_SEC: i64 = 1_000_000_000;

type FeatureOutputTuple = (
    Py<PyArray2<f64>>,
    Vec<String>,
    Py<PyArray2<f64>>,
    Vec<String>,
    Py<PyArray2<f64>>,
    Vec<String>,
);

fn detect_timestamp_factor_to_ns(timestamps: &[i64]) -> i64 {
    if timestamps.len() < 2 {
        return NS_PER_MS;
    }

    let min_ts = *timestamps.iter().min().unwrap_or(&0);
    let max_ts = *timestamps.iter().max().unwrap_or(&0);
    let range = max_ts.saturating_sub(min_ts);

    let mut min_positive_diff = i64::MAX;
    for w in timestamps.windows(2).take(20_000) {
        let diff = w[1].saturating_sub(w[0]);
        if diff > 0 && diff < min_positive_diff {
            min_positive_diff = diff;
        }
    }

    if range >= 100_000_000_000 || min_positive_diff >= NS_PER_MS {
        1
    } else {
        NS_PER_MS
    }
}

fn normalize_timestamps_to_ns(timestamps: &[i64]) -> Vec<i64> {
    let factor = detect_timestamp_factor_to_ns(timestamps);
    if factor == 1 {
        timestamps.to_vec()
    } else {
        timestamps
            .iter()
            .map(|t| t.saturating_mul(factor))
            .collect::<Vec<_>>()
    }
}

fn validate_market_lengths(
    n_market: usize,
    market_prices: &[f64],
    market_volumes: &[f64],
    market_turnovers: &[f64],
    market_flags: &[i32],
    bid_order_ids: &[i64],
    ask_order_ids: &[i64],
) -> PyResult<()> {
    if market_prices.len() != n_market
        || market_volumes.len() != n_market
        || market_turnovers.len() != n_market
        || market_flags.len() != n_market
        || bid_order_ids.len() != n_market
        || ask_order_ids.len() != n_market
    {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "市场数组长度不一致：timestamps/prices/volumes/turnovers/flags/bid_order_ids/ask_order_ids 必须等长",
        ));
    }
    Ok(())
}

fn direction_sign(direction: i32) -> f64 {
    match direction {
        66 => 1.0,
        83 => -1.0,
        _ => 0.0,
    }
}

fn std_slice(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let var = values
        .iter()
        .map(|v| {
            let d = v - mean;
            d * d
        })
        .sum::<f64>()
        / values.len() as f64;
    var.sqrt()
}

fn linear_slope(xs: &[f64], ys: &[f64]) -> f64 {
    if xs.len() < 2 || xs.len() != ys.len() {
        return 0.0;
    }
    let n = xs.len() as f64;
    let mean_x = xs.iter().sum::<f64>() / n;
    let mean_y = ys.iter().sum::<f64>() / n;

    let mut num = 0.0;
    let mut den = 0.0;
    for i in 0..xs.len() {
        let dx = xs[i] - mean_x;
        num += dx * (ys[i] - mean_y);
        den += dx * dx;
    }

    if den.abs() < 1e-12 {
        0.0
    } else {
        num / den
    }
}

fn curvature_mean_second_derivative(xs: &[f64], ys: &[f64]) -> f64 {
    if xs.len() < 3 || xs.len() != ys.len() {
        return 0.0;
    }

    let mut sum = 0.0;
    let mut cnt = 0usize;
    for i in 1..xs.len() - 1 {
        let h1 = xs[i] - xs[i - 1];
        let h2 = xs[i + 1] - xs[i];
        if h1.abs() < 1e-12 || h2.abs() < 1e-12 {
            continue;
        }

        let d1 = (ys[i] - ys[i - 1]) / h1;
        let d2 = (ys[i + 1] - ys[i]) / h2;
        let h = (xs[i + 1] - xs[i - 1]) * 0.5;
        if h.abs() < 1e-12 {
            continue;
        }

        sum += (d2 - d1) / h;
        cnt += 1;
    }

    if cnt == 0 {
        0.0
    } else {
        sum / cnt as f64
    }
}

fn monotonicity_and_turning_points(ys: &[f64]) -> (f64, f64) {
    if ys.len() < 2 {
        return (0.0, 0.0);
    }

    let mut pos = 0usize;
    let mut neg = 0usize;
    let mut nonzero = 0usize;
    let mut turning = 0usize;
    let mut prev_sign = 0i32;

    for i in 1..ys.len() {
        let diff = ys[i] - ys[i - 1];
        let sign = if diff > 1e-12 {
            1
        } else if diff < -1e-12 {
            -1
        } else {
            0
        };

        if sign == 0 {
            continue;
        }

        nonzero += 1;
        if sign > 0 {
            pos += 1;
        } else {
            neg += 1;
        }

        if prev_sign != 0 && sign != prev_sign {
            turning += 1;
        }
        prev_sign = sign;
    }

    if nonzero == 0 {
        (0.0, 0.0)
    } else {
        (pos.max(neg) as f64 / nonzero as f64, turning as f64)
    }
}

fn build_market_trades(
    market_ts_ns: &[i64],
    market_pr: &[f64],
    market_vo: &[f64],
    market_to: &[f64],
    market_fl: &[i32],
    bid_ids: &[i64],
    ask_ids: &[i64],
) -> Vec<MarketTrade> {
    let mut market_trades: Vec<MarketTrade> = Vec::with_capacity(market_ts_ns.len());
    for i in 0..market_ts_ns.len() {
        market_trades.push(MarketTrade {
            timestamp: market_ts_ns[i],
            price: market_pr[i],
            volume: market_vo[i],
            turnover: market_to[i],
            flag: market_fl[i],
            bid_order_id: bid_ids[i],
            ask_order_id: ask_ids[i],
        });
    }
    market_trades
}

fn build_agent_trades(
    agent_indices: &[usize],
    agent_dirs: &[i32],
    agent_vols: &[f64],
    market_trades: &[MarketTrade],
    agent_hint: &str,
) -> PyResult<Vec<AgentTrade>> {
    let n_agent = agent_indices.len();
    if n_agent == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "{}交易数据为空",
            agent_hint
        )));
    }

    if agent_dirs.len() != n_agent || agent_vols.len() != n_agent {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "{}的 market_indices/directions/volumes 长度必须一致",
            agent_hint
        )));
    }

    let mut agent_trades: Vec<AgentTrade> = Vec::with_capacity(n_agent);
    for i in 0..n_agent {
        let market_idx = agent_indices[i];
        if market_idx >= market_trades.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "{}交易索引 {} 超出市场数据范围",
                agent_hint, market_idx
            )));
        }

        if i > 0 && market_idx < agent_indices[i - 1] {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "{}的 market_indices 必须按升序排列",
                agent_hint
            )));
        }

        agent_trades.push(AgentTrade {
            timestamp: market_trades[market_idx].timestamp,
            market_trade_idx: market_idx,
            direction: agent_dirs[i],
            volume: agent_vols[i],
            price: market_trades[market_idx].price,
        });
    }

    Ok(agent_trades)
}

fn compute_forward_signed_impact(
    agent_trades: &[AgentTrade],
    market_trades: &[MarketTrade],
    horizon_ns: i64,
) -> Vec<f64> {
    let mut impacts = vec![0.0; agent_trades.len()];
    if agent_trades.is_empty() || market_trades.is_empty() {
        return impacts;
    }

    let n_market = market_trades.len();
    let mut ptr = 0usize;

    for i in 0..agent_trades.len() {
        let trade = &agent_trades[i];
        let min_ptr = trade.market_trade_idx.saturating_add(1);
        if ptr < min_ptr {
            ptr = min_ptr;
        }

        let target_time = trade.timestamp.saturating_add(horizon_ns);
        while ptr < n_market && market_trades[ptr].timestamp < target_time {
            ptr += 1;
        }

        if ptr >= n_market || trade.price.abs() < 1e-12 {
            impacts[i] = 0.0;
            continue;
        }

        let future_price = market_trades[ptr].price;
        let ret = (future_price - trade.price) / trade.price;
        impacts[i] = ret * direction_sign(trade.direction);
    }

    impacts
}

fn get_base_feature_names_internal() -> Vec<String> {
    vec![
        "timestamp".to_string(),
        "post_trade_price_trend".to_string(),
        "order_id_diff".to_string(),
        "pre_trade_count".to_string(),
        "post_trade_count".to_string(),
        "price_deviation".to_string(),
        "pre_trade_volatility".to_string(),
        "post_trade_volatility".to_string(),
        "sync_rate_same_direction".to_string(),
        "max_drawdown".to_string(),
        "max_drawdown_duration".to_string(),
        "profit_concentration".to_string(),
        "gain_loss_ratio".to_string(),
        "avg_holding_duration".to_string(),
        "exposure_utilization".to_string(),
        "trading_density".to_string(),
        "sharpe_ratio".to_string(),
        "sortino_ratio".to_string(),
        "calmar_ratio".to_string(),
        "cumulative_return".to_string(),
        "excess_return".to_string(),
        "current_position".to_string(),
        "cumulative_volume".to_string(),
    ]
}

fn get_fast_event_feature_names_internal() -> Vec<String> {
    vec![
        "inter_trade_dt_ns".to_string(),
        "signed_volume".to_string(),
        "direction_flip".to_string(),
        "run_length_same_direction".to_string(),
        "volume_participation".to_string(),
        "impact_inst_1s".to_string(),
        "impact_inst_5s".to_string(),
        "impact_reversion_5s".to_string(),
        "burstiness_running".to_string(),
    ]
}

fn get_time_grid_feature_names_internal() -> Vec<String> {
    vec![
        "grid_timestamp".to_string(),
        "trade_count".to_string(),
        "trade_density".to_string(),
        "signed_volume_sum".to_string(),
        "buy_ratio".to_string(),
        "flip_rate".to_string(),
        "impact_inst_1s_mean".to_string(),
        "impact_inst_5s_mean".to_string(),
        "position_last".to_string(),
        "cum_return_delta".to_string(),
        "cum_volume_delta".to_string(),
    ]
}

fn get_param_axis_feature_names_internal() -> Vec<String> {
    vec![
        "timestamp".to_string(),
        "crowding_index_same_direction".to_string(),
        "active_agent_ratio".to_string(),
        "param_dispersion_trade_count".to_string(),
        "param_dispersion_signed_flow".to_string(),
        "minority_score".to_string(),
        "lead_lag_score".to_string(),
        "param_slope_signed_flow".to_string(),
        "param_curvature_signed_flow".to_string(),
        "param_monotonicity_signed_flow".to_string(),
        "param_turning_points_signed_flow".to_string(),
        "response_nonlinearity".to_string(),
        "target_flow_rank".to_string(),
    ]
}

fn build_event_feature_array(
    base_features: &[AgentFeatures],
    agent_trades: &[AgentTrade],
    market_trades: &[MarketTrade],
) -> (Array2<f64>, Vec<String>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let base_names = get_base_feature_names_internal();
    let extra_names = get_fast_event_feature_names_internal();

    let mut all_names = Vec::with_capacity(base_names.len() + extra_names.len());
    all_names.extend(base_names);
    all_names.extend(extra_names);

    let n = base_features.len();
    let n_cols = all_names.len();
    let mut result = Array2::<f64>::zeros((n, n_cols));

    let impact_1s = compute_forward_signed_impact(agent_trades, market_trades, NS_PER_SEC);
    let impact_5s = compute_forward_signed_impact(agent_trades, market_trades, 5 * NS_PER_SEC);

    let mut signed_volume_series = vec![0.0; n];
    let mut direction_flip_series = vec![0.0; n];
    let mut inter_trade_dt_series = vec![0.0; n];
    let mut burstiness_series = vec![0.0; n];
    let mut impact_reversion_series = vec![0.0; n];

    let mut dt_count = 0.0;
    let mut dt_mean = 0.0;
    let mut dt_m2 = 0.0;

    let mut run_len = 0.0;
    let mut prev_dir = 0i32;

    for i in 0..n {
        let base_row = base_features[i].to_vec();
        for j in 0..base_row.len() {
            result[[i, j]] = base_row[j];
        }

        let trade = &agent_trades[i];
        let sign = direction_sign(trade.direction);

        let inter_trade_dt_ns = if i == 0 {
            0.0
        } else {
            (trade.timestamp - agent_trades[i - 1].timestamp).max(0) as f64
        };
        inter_trade_dt_series[i] = inter_trade_dt_ns;

        if i > 0 {
            dt_count += 1.0;
            let delta = inter_trade_dt_ns - dt_mean;
            dt_mean += delta / dt_count;
            dt_m2 += delta * (inter_trade_dt_ns - dt_mean);
        }

        let dt_std = if dt_count > 1.0 {
            (dt_m2 / dt_count).sqrt()
        } else {
            0.0
        };
        let burstiness = if (dt_std + dt_mean).abs() < 1e-12 {
            0.0
        } else {
            (dt_std - dt_mean) / (dt_std + dt_mean)
        };
        burstiness_series[i] = burstiness;

        let direction_flip = if i == 0 || trade.direction == prev_dir {
            0.0
        } else {
            1.0
        };
        direction_flip_series[i] = direction_flip;

        run_len = if i == 0 || trade.direction != prev_dir {
            1.0
        } else {
            run_len + 1.0
        };

        let signed_volume = sign * trade.volume;
        signed_volume_series[i] = signed_volume;

        let market_volume = market_trades[trade.market_trade_idx].volume;
        let volume_participation = if market_volume.abs() < 1e-12 {
            0.0
        } else {
            trade.volume / market_volume
        };

        let impact1 = impact_1s[i];
        let impact5 = impact_5s[i];
        let impact_reversion = if impact1.abs() < 1e-12 {
            0.0
        } else {
            (impact1 - impact5) / impact1
        };
        impact_reversion_series[i] = impact_reversion;

        let mut col = 23;
        result[[i, col]] = inter_trade_dt_ns;
        col += 1;
        result[[i, col]] = signed_volume;
        col += 1;
        result[[i, col]] = direction_flip;
        col += 1;
        result[[i, col]] = run_len;
        col += 1;
        result[[i, col]] = volume_participation;
        col += 1;
        result[[i, col]] = impact1;
        col += 1;
        result[[i, col]] = impact5;
        col += 1;
        result[[i, col]] = impact_reversion;
        col += 1;
        result[[i, col]] = burstiness;

        prev_dir = trade.direction;
    }

    (
        result,
        all_names,
        signed_volume_series,
        direction_flip_series,
        impact_1s,
        impact_5s,
        impact_reversion_series,
    )
}

fn build_time_grid_feature_array(
    agent_trades: &[AgentTrade],
    base_features: &[AgentFeatures],
    signed_volume_series: &[f64],
    direction_flip_series: &[f64],
    impact_1s: &[f64],
    impact_5s: &[f64],
    grid_ns: i64,
) -> (Array2<f64>, Vec<String>) {
    let names = get_time_grid_feature_names_internal();
    if agent_trades.is_empty() {
        return (Array2::<f64>::zeros((0, names.len())), names);
    }

    let start_raw = agent_trades[0].timestamp;
    let end_time = agent_trades[agent_trades.len() - 1].timestamp;
    let start_time = start_raw - start_raw.rem_euclid(grid_ns);
    let n_bins = ((end_time - start_time) / grid_ns + 1) as usize;

    let mut trade_count = vec![0usize; n_bins];
    let mut signed_volume_sum = vec![0.0; n_bins];
    let mut buy_count = vec![0usize; n_bins];
    let mut flip_count = vec![0.0; n_bins];
    let mut impact_1s_sum = vec![0.0; n_bins];
    let mut impact_5s_sum = vec![0.0; n_bins];

    let mut position_last = vec![0.0; n_bins];
    let mut cum_return_last = vec![0.0; n_bins];
    let mut cum_volume_last = vec![0.0; n_bins];

    let mut has_position = vec![false; n_bins];
    let mut has_return = vec![false; n_bins];
    let mut has_volume = vec![false; n_bins];

    for i in 0..agent_trades.len() {
        let bin = ((agent_trades[i].timestamp - start_time) / grid_ns) as usize;

        trade_count[bin] += 1;
        signed_volume_sum[bin] += signed_volume_series[i];
        impact_1s_sum[bin] += impact_1s[i];
        impact_5s_sum[bin] += impact_5s[i];
        flip_count[bin] += direction_flip_series[i];

        if agent_trades[i].direction == 66 {
            buy_count[bin] += 1;
        }

        position_last[bin] = base_features[i].current_position;
        cum_return_last[bin] = base_features[i].cumulative_return;
        cum_volume_last[bin] = base_features[i].cumulative_volume;

        has_position[bin] = true;
        has_return[bin] = true;
        has_volume[bin] = true;
    }

    let mut result = Array2::<f64>::zeros((n_bins, names.len()));
    let mut prev_position = 0.0;
    let mut prev_cum_return = 0.0;
    let mut prev_cum_volume = 0.0;

    for b in 0..n_bins {
        let bin_ts = start_time + b as i64 * grid_ns;

        if has_position[b] {
            prev_position = position_last[b];
        } else {
            position_last[b] = prev_position;
        }

        let cur_cum_return = if has_return[b] {
            cum_return_last[b]
        } else {
            prev_cum_return
        };
        let cur_cum_volume = if has_volume[b] {
            cum_volume_last[b]
        } else {
            prev_cum_volume
        };

        let trade_count_f = trade_count[b] as f64;
        let trade_density = trade_count_f * NS_PER_SEC as f64 / grid_ns as f64;
        let buy_ratio = if trade_count[b] == 0 {
            0.0
        } else {
            buy_count[b] as f64 / trade_count_f
        };
        let flip_rate = if trade_count[b] <= 1 {
            0.0
        } else {
            flip_count[b] / (trade_count[b] - 1) as f64
        };
        let impact_1s_mean = if trade_count[b] == 0 {
            0.0
        } else {
            impact_1s_sum[b] / trade_count_f
        };
        let impact_5s_mean = if trade_count[b] == 0 {
            0.0
        } else {
            impact_5s_sum[b] / trade_count_f
        };

        let cum_return_delta = cur_cum_return - prev_cum_return;
        let cum_volume_delta = cur_cum_volume - prev_cum_volume;

        result[[b, 0]] = bin_ts as f64;
        result[[b, 1]] = trade_count_f;
        result[[b, 2]] = trade_density;
        result[[b, 3]] = signed_volume_sum[b];
        result[[b, 4]] = buy_ratio;
        result[[b, 5]] = flip_rate;
        result[[b, 6]] = impact_1s_mean;
        result[[b, 7]] = impact_5s_mean;
        result[[b, 8]] = position_last[b];
        result[[b, 9]] = cum_return_delta;
        result[[b, 10]] = cum_volume_delta;

        prev_cum_return = cur_cum_return;
        prev_cum_volume = cur_cum_volume;
    }

    (result, names)
}

fn build_param_axis_feature_array(
    target_trades: &[AgentTrade],
    target_agent_idx: usize,
    all_agent_trades: &[Vec<AgentTrade>],
    param_window_ns: i64,
    all_agent_params: Option<&[f64]>,
) -> PyResult<(Array2<f64>, Vec<String>)> {
    let names = get_param_axis_feature_names_internal();
    let n_target = target_trades.len();
    if n_target == 0 {
        return Ok((Array2::<f64>::zeros((0, names.len())), names));
    }

    let n_agents = all_agent_trades.len();
    if n_agents == 0 {
        return Ok((Array2::<f64>::zeros((0, names.len())), names));
    }

    let params: Vec<f64> = if let Some(p) = all_agent_params {
        if p.len() != n_agents {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "all_agent_params 长度必须与Agent数量一致",
            ));
        }
        p.to_vec()
    } else {
        (0..n_agents).map(|i| i as f64).collect()
    };

    let mut param_order: Vec<usize> = (0..n_agents).collect();
    param_order.sort_by(|a, b| params[*a].partial_cmp(&params[*b]).unwrap_or(Ordering::Equal));
    let sorted_params: Vec<f64> = param_order.iter().map(|&i| params[i]).collect();

    let mut left_ptrs = vec![0usize; n_agents];
    let mut right_ptrs = vec![0usize; n_agents];

    let mut prefix_signed_flows: Vec<Vec<f64>> = Vec::with_capacity(n_agents);
    for trades in all_agent_trades {
        let mut prefix = Vec::with_capacity(trades.len() + 1);
        prefix.push(0.0);
        for t in trades {
            let next = prefix[prefix.len() - 1] + direction_sign(t.direction) * t.volume;
            prefix.push(next);
        }
        prefix_signed_flows.push(prefix);
    }

    let mut result = Array2::<f64>::zeros((n_target, names.len()));
    let half = (param_window_ns / 2).max(1);

    for i in 0..n_target {
        let t = target_trades[i].timestamp;
        let start_time = t.saturating_sub(half);
        let end_time = t.saturating_add(half);

        let mut trade_counts = vec![0.0; n_agents];
        let mut signed_flows = vec![0.0; n_agents];

        let mut active_agents = 0usize;
        let mut same_dir_active = 0usize;
        let mut active_others = 0usize;
        let mut lead_count = 0usize;

        let target_dir_sign = direction_sign(target_trades[i].direction);

        for a in 0..n_agents {
            let trades = &all_agent_trades[a];
            let n = trades.len();
            if n == 0 {
                continue;
            }

            while left_ptrs[a] < n && trades[left_ptrs[a]].timestamp < start_time {
                left_ptrs[a] += 1;
            }
            if right_ptrs[a] < left_ptrs[a] {
                right_ptrs[a] = left_ptrs[a];
            }
            while right_ptrs[a] < n && trades[right_ptrs[a]].timestamp <= end_time {
                right_ptrs[a] += 1;
            }

            let left = left_ptrs[a];
            let right = right_ptrs[a];
            let count = right.saturating_sub(left);

            if count > 0 {
                active_agents += 1;
                if a != target_agent_idx {
                    active_others += 1;
                    if trades[left].timestamp >= t {
                        lead_count += 1;
                    }
                }
            }

            trade_counts[a] = count as f64;
            let flow = prefix_signed_flows[a][right] - prefix_signed_flows[a][left];
            signed_flows[a] = flow;

            if flow * target_dir_sign > 0.0 {
                same_dir_active += 1;
            }
        }

        let crowding_index = same_dir_active as f64 / n_agents as f64;
        let active_agent_ratio = active_agents as f64 / n_agents as f64;
        let dispersion_trade_count = std_slice(&trade_counts);
        let dispersion_signed_flow = std_slice(&signed_flows);

        let aggregate_flow = signed_flows.iter().sum::<f64>();
        let majority_sign = if aggregate_flow > 1e-12 {
            1.0
        } else if aggregate_flow < -1e-12 {
            -1.0
        } else {
            0.0
        };
        let minority_score = if majority_sign == 0.0 || majority_sign == target_dir_sign {
            0.0
        } else {
            1.0
        };

        let lead_lag_score = if active_others == 0 {
            0.0
        } else {
            lead_count as f64 / active_others as f64
        };

        let mut sorted_flows = Vec::with_capacity(n_agents);
        for idx in &param_order {
            sorted_flows.push(signed_flows[*idx]);
        }

        let slope = linear_slope(&sorted_params, &sorted_flows);
        let curvature = curvature_mean_second_derivative(&sorted_params, &sorted_flows);
        let (monotonicity, turning_points) = monotonicity_and_turning_points(&sorted_flows);
        let response_nonlinearity = curvature.abs() / (slope.abs() + 1e-12);

        let target_flow = signed_flows[target_agent_idx];
        let mut rank_count = 0usize;
        for f in &signed_flows {
            if *f <= target_flow {
                rank_count += 1;
            }
        }
        let target_flow_rank = rank_count as f64 / n_agents as f64;

        result[[i, 0]] = t as f64;
        result[[i, 1]] = crowding_index;
        result[[i, 2]] = active_agent_ratio;
        result[[i, 3]] = dispersion_trade_count;
        result[[i, 4]] = dispersion_signed_flow;
        result[[i, 5]] = minority_score;
        result[[i, 6]] = lead_lag_score;
        result[[i, 7]] = slope;
        result[[i, 8]] = curvature;
        result[[i, 9]] = monotonicity;
        result[[i, 10]] = turning_points;
        result[[i, 11]] = response_nonlinearity;
        result[[i, 12]] = target_flow_rank;
    }

    Ok((result, names))
}

fn pack_outputs(
    py: Python<'_>,
    event_array: Array2<f64>,
    event_names: Vec<String>,
    grid_array: Array2<f64>,
    grid_names: Vec<String>,
    param_array: Array2<f64>,
    param_names: Vec<String>,
) -> FeatureOutputTuple {
    (
        event_array.into_pyarray(py).to_owned(),
        event_names,
        grid_array.into_pyarray(py).to_owned(),
        grid_names,
        param_array.into_pyarray(py).to_owned(),
        param_names,
    )
}

/// Python可调用的特征计算函数（单Agent版本）
///
/// 返回值：
/// 1) 事件级特征矩阵 + 列名
/// 2) 时间栅格特征矩阵 + 列名
/// 3) 参数轴特征矩阵 + 列名（单Agent版本为空矩阵）
#[pyfunction]
#[pyo3(signature = (
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
    window_ms=10000,
    grid_ms=1000
))]
fn compute_agent_trading_features(
    py: Python<'_>,
    market_timestamps: PyReadonlyArray1<i64>,
    market_prices: PyReadonlyArray1<f64>,
    market_volumes: PyReadonlyArray1<f64>,
    market_turnovers: PyReadonlyArray1<f64>,
    market_flags: PyReadonlyArray1<i32>,
    bid_order_ids: PyReadonlyArray1<i64>,
    ask_order_ids: PyReadonlyArray1<i64>,
    agent_market_indices: PyReadonlyArray1<usize>,
    agent_directions: PyReadonlyArray1<i32>,
    agent_volumes: PyReadonlyArray1<f64>,
    window_ms: i64,
    grid_ms: i64,
) -> PyResult<FeatureOutputTuple> {
    let market_ts = market_timestamps.as_slice()?;
    let market_pr = market_prices.as_slice()?;
    let market_vo = market_volumes.as_slice()?;
    let market_to = market_turnovers.as_slice()?;
    let market_fl = market_flags.as_slice()?;
    let bid_ids = bid_order_ids.as_slice()?;
    let ask_ids = ask_order_ids.as_slice()?;

    let n_market = market_ts.len();
    if n_market == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err("市场数据为空"));
    }

    validate_market_lengths(
        n_market, market_pr, market_vo, market_to, market_fl, bid_ids, ask_ids,
    )?;

    let window_ns = window_ms.saturating_mul(NS_PER_MS);
    if window_ns <= 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "window_ms 必须为正数",
        ));
    }

    let grid_ns = grid_ms.saturating_mul(NS_PER_MS);
    if grid_ns <= 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "grid_ms 必须为正数",
        ));
    }

    let market_ts_ns = normalize_timestamps_to_ns(market_ts);
    let market_trades = build_market_trades(
        &market_ts_ns,
        market_pr,
        market_vo,
        market_to,
        market_fl,
        bid_ids,
        ask_ids,
    );

    let agent_indices = agent_market_indices.as_slice()?;
    let agent_dirs = agent_directions.as_slice()?;
    let agent_vols = agent_volumes.as_slice()?;

    let agent_trades = build_agent_trades(
        agent_indices,
        agent_dirs,
        agent_vols,
        &market_trades,
        "Agent",
    )?;

    let base_features = compute_agent_features(&market_trades, &agent_trades, window_ns, None);

    let (event_array, event_names, signed_volume, direction_flip, impact_1s, impact_5s, _) =
        build_event_feature_array(&base_features, &agent_trades, &market_trades);

    let (grid_array, grid_names) = build_time_grid_feature_array(
        &agent_trades,
        &base_features,
        &signed_volume,
        &direction_flip,
        &impact_1s,
        &impact_5s,
        grid_ns,
    );

    let param_array = Array2::<f64>::zeros((base_features.len(), 0));
    let param_names: Vec<String> = Vec::new();

    Ok(pack_outputs(
        py,
        event_array,
        event_names,
        grid_array,
        grid_names,
        param_array,
        param_names,
    ))
}

/// Python可调用的多Agent特征计算（含同步率、参数轴特征）
#[pyfunction]
#[pyo3(signature = (
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
    target_agent_idx,
    window_ms=10000,
    grid_ms=1000,
    param_window_ms=2000,
    all_agent_params=None
))]
fn compute_agent_trading_features_multi(
    py: Python<'_>,
    market_timestamps: PyReadonlyArray1<i64>,
    market_prices: PyReadonlyArray1<f64>,
    market_volumes: PyReadonlyArray1<f64>,
    market_turnovers: PyReadonlyArray1<f64>,
    market_flags: PyReadonlyArray1<i32>,
    bid_order_ids: PyReadonlyArray1<i64>,
    ask_order_ids: PyReadonlyArray1<i64>,
    all_agent_market_indices: Vec<PyReadonlyArray1<usize>>,
    all_agent_directions: Vec<PyReadonlyArray1<i32>>,
    all_agent_volumes: Vec<PyReadonlyArray1<f64>>,
    target_agent_idx: usize,
    window_ms: i64,
    grid_ms: i64,
    param_window_ms: i64,
    all_agent_params: Option<Vec<f64>>,
) -> PyResult<FeatureOutputTuple> {
    let market_ts = market_timestamps.as_slice()?;
    let market_pr = market_prices.as_slice()?;
    let market_vo = market_volumes.as_slice()?;
    let market_to = market_turnovers.as_slice()?;
    let market_fl = market_flags.as_slice()?;
    let bid_ids = bid_order_ids.as_slice()?;
    let ask_ids = ask_order_ids.as_slice()?;

    let n_market = market_ts.len();
    if n_market == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err("市场数据为空"));
    }

    validate_market_lengths(
        n_market, market_pr, market_vo, market_to, market_fl, bid_ids, ask_ids,
    )?;

    let window_ns = window_ms.saturating_mul(NS_PER_MS);
    if window_ns <= 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "window_ms 必须为正数",
        ));
    }

    let grid_ns = grid_ms.saturating_mul(NS_PER_MS);
    if grid_ns <= 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "grid_ms 必须为正数",
        ));
    }

    let param_window_ns = param_window_ms.saturating_mul(NS_PER_MS);
    if param_window_ns <= 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "param_window_ms 必须为正数",
        ));
    }

    let market_ts_ns = normalize_timestamps_to_ns(market_ts);
    let market_trades = build_market_trades(
        &market_ts_ns,
        market_pr,
        market_vo,
        market_to,
        market_fl,
        bid_ids,
        ask_ids,
    );

    let n_agents = all_agent_market_indices.len();
    if n_agents == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err("没有Agent数据"));
    }

    if all_agent_directions.len() != n_agents || all_agent_volumes.len() != n_agents {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "all_agent_market_indices/all_agent_directions/all_agent_volumes 列表长度必须一致",
        ));
    }

    if target_agent_idx >= n_agents {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "目标Agent索引 {} 超出范围",
            target_agent_idx
        )));
    }

    let mut all_agent_trades: Vec<Vec<AgentTrade>> = Vec::with_capacity(n_agents);
    for agent_i in 0..n_agents {
        let indices = all_agent_market_indices[agent_i].as_slice()?;
        let dirs = all_agent_directions[agent_i].as_slice()?;
        let vols = all_agent_volumes[agent_i].as_slice()?;

        let trades = build_agent_trades(
            indices,
            dirs,
            vols,
            &market_trades,
            &format!("Agent {}", agent_i),
        )?;
        all_agent_trades.push(trades);
    }

    let target_trades = &all_agent_trades[target_agent_idx];

    let all_other_refs: Vec<&[AgentTrade]> = all_agent_trades
        .iter()
        .enumerate()
        .filter_map(|(i, v)| if i == target_agent_idx { None } else { Some(v.as_slice()) })
        .collect();

    let base_features = compute_agent_features(
        &market_trades,
        target_trades,
        window_ns,
        Some(&all_other_refs),
    );

    let (event_array, event_names, signed_volume, direction_flip, impact_1s, impact_5s, _) =
        build_event_feature_array(&base_features, target_trades, &market_trades);

    let (grid_array, grid_names) = build_time_grid_feature_array(
        target_trades,
        &base_features,
        &signed_volume,
        &direction_flip,
        &impact_1s,
        &impact_5s,
        grid_ns,
    );

    let (param_array, param_names) = build_param_axis_feature_array(
        target_trades,
        target_agent_idx,
        &all_agent_trades,
        param_window_ns,
        all_agent_params.as_deref(),
    )?;

    Ok(pack_outputs(
        py,
        event_array,
        event_names,
        grid_array,
        grid_names,
        param_array,
        param_names,
    ))
}

/// 获取事件级特征列名（Base + 快速扩展）
#[pyfunction]
fn get_agent_feature_names() -> Vec<String> {
    let mut names = get_base_feature_names_internal();
    names.extend(get_fast_event_feature_names_internal());
    names
}

/// 获取时间栅格特征列名
#[pyfunction]
fn get_agent_time_grid_feature_names() -> Vec<String> {
    get_time_grid_feature_names_internal()
}

/// 获取参数轴特征列名
#[pyfunction]
fn get_agent_param_axis_feature_names() -> Vec<String> {
    get_param_axis_feature_names_internal()
}

pub fn register_functions(m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_agent_trading_features, m)?)?;
    m.add_function(wrap_pyfunction!(compute_agent_trading_features_multi, m)?)?;
    m.add_function(wrap_pyfunction!(get_agent_feature_names, m)?)?;
    m.add_function(wrap_pyfunction!(get_agent_time_grid_feature_names, m)?)?;
    m.add_function(wrap_pyfunction!(get_agent_param_axis_feature_names, m)?)?;
    Ok(())
}
