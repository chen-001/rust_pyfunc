use numpy::PyArray2;
use pyo3::prelude::*;
use pyo3::types::PyList;
use std::collections::HashMap;

#[pyfunction]
#[pyo3(signature = (
    trade_times,
    trade_flags,
    trade_bid_orders,
    trade_ask_orders,
    trade_volumes,
    market_times,
    compute_direction_ratio = true,
    compute_flag_ratio = true
))]
pub fn calculate_passive_order_features(
    py: Python,
    trade_times: Vec<i64>,
    trade_flags: Vec<i32>,
    trade_bid_orders: Vec<i64>,
    trade_ask_orders: Vec<i64>,
    trade_volumes: Vec<i64>,
    market_times: Vec<i64>,
    compute_direction_ratio: bool,
    compute_flag_ratio: bool,
) -> PyResult<(Py<PyArray2<f64>>, Py<PyList>)> {
    let n_snapshots = market_times.len();
    let n_intervals = if n_snapshots == 0 { 0 } else { n_snapshots - 1 };

    // 计算特征数
    let base_features = 42; // 全部/买单/卖单各7个订单编号 + 全部/买单/卖单各7个体量
    let direction_features = if compute_direction_ratio { 21 } else { 0 }; // 全部/买单/卖单各7个方向比例
    let flag_features = if compute_flag_ratio { 21 } else { 0 }; // 全部/买单/卖单各7个flag比例
    let total_features = base_features + direction_features + flag_features;

    let column_names = get_column_names(compute_direction_ratio, compute_flag_ratio);
    let py_list = PyList::new(py, column_names.iter());

    if n_intervals == 0 {
        let empty_array = numpy::PyArray2::<f64>::zeros(py, (0, total_features), false).to_owned();
        return Ok((empty_array, py_list.into()));
    }

    let mut result = Vec::with_capacity(n_intervals * total_features);

    for interval_idx in 0..n_intervals {
        let start_time = market_times[interval_idx];
        let end_time = market_times[interval_idx + 1];

        let features = compute_interval_features(
            &trade_times,
            &trade_flags,
            &trade_bid_orders,
            &trade_ask_orders,
            &trade_volumes,
            start_time,
            end_time,
            compute_direction_ratio,
            compute_flag_ratio,
        );

        result.extend(features);
    }

    let result_2d: Vec<Vec<f64>> = result
        .chunks(total_features)
        .map(|chunk| chunk.to_vec())
        .collect();
    let array = PyArray2::from_vec2(py, &result_2d).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Failed to create array: {}", e))
    })?;

    Ok((array.to_owned(), py_list.into()))
}

fn get_column_names(compute_direction_ratio: bool, compute_flag_ratio: bool) -> Vec<String> {
    let mut names = Vec::new();

    // 全部被动订单编号特征 (1-7)
    names.extend(vec![
        "passive_all_order_mean",
        "passive_all_order_std",
        "passive_all_order_skew",
        "passive_all_order_kurtosis",
        "passive_all_order_autocorr",
        "passive_all_order_trend",
        "passive_all_order_lz_complexity",
    ]);

    // 被动买单编号特征 (8-14)
    names.extend(vec![
        "passive_bid_order_mean",
        "passive_bid_order_std",
        "passive_bid_order_skew",
        "passive_bid_order_kurtosis",
        "passive_bid_order_autocorr",
        "passive_bid_order_trend",
        "passive_bid_order_lz_complexity",
    ]);

    // 被动卖单编号特征 (15-21)
    names.extend(vec![
        "passive_ask_order_mean",
        "passive_ask_order_std",
        "passive_ask_order_skew",
        "passive_ask_order_kurtosis",
        "passive_ask_order_autocorr",
        "passive_ask_order_trend",
        "passive_ask_order_lz_complexity",
    ]);

    // 全部订单体量特征 (22-28)
    names.extend(vec![
        "order_volume_mean",
        "order_volume_std",
        "order_volume_skew",
        "order_volume_kurtosis",
        "order_volume_autocorr",
        "order_volume_trend",
        "order_volume_lz_complexity",
    ]);

    // 买单体量特征 (29-35)
    names.extend(vec![
        "bid_order_volume_mean",
        "bid_order_volume_std",
        "bid_order_volume_skew",
        "bid_order_volume_kurtosis",
        "bid_order_volume_autocorr",
        "bid_order_volume_trend",
        "bid_order_volume_lz_complexity",
    ]);

    // 卖单体量特征 (36-42)
    names.extend(vec![
        "ask_order_volume_mean",
        "ask_order_volume_std",
        "ask_order_volume_skew",
        "ask_order_volume_kurtosis",
        "ask_order_volume_autocorr",
        "ask_order_volume_trend",
        "ask_order_volume_lz_complexity",
    ]);

    if compute_direction_ratio {
        // 全部订单方向比例特征 (43-49)
        names.extend(vec![
            "direction_ratio_all_mean",
            "direction_ratio_all_std",
            "direction_ratio_all_skew",
            "direction_ratio_all_kurtosis",
            "direction_ratio_all_autocorr",
            "direction_ratio_all_trend",
            "direction_ratio_all_lz_complexity",
        ]);

        // 买单方向比例特征 (50-56)
        names.extend(vec![
            "direction_ratio_bid_mean",
            "direction_ratio_bid_std",
            "direction_ratio_bid_skew",
            "direction_ratio_bid_kurtosis",
            "direction_ratio_bid_autocorr",
            "direction_ratio_bid_trend",
            "direction_ratio_bid_lz_complexity",
        ]);

        // 卖单方向比例特征 (57-63)
        names.extend(vec![
            "direction_ratio_ask_mean",
            "direction_ratio_ask_std",
            "direction_ratio_ask_skew",
            "direction_ratio_ask_kurtosis",
            "direction_ratio_ask_autocorr",
            "direction_ratio_ask_trend",
            "direction_ratio_ask_lz_complexity",
        ]);
    }

    if compute_flag_ratio {
        // 全部订单flag比例特征 (64-70 or 43-49)
        names.extend(vec![
            "flag_ratio_all_mean",
            "flag_ratio_all_std",
            "flag_ratio_all_skew",
            "flag_ratio_all_kurtosis",
            "flag_ratio_all_autocorr",
            "flag_ratio_all_trend",
            "flag_ratio_all_lz_complexity",
        ]);

        // 买单flag比例特征
        names.extend(vec![
            "flag_ratio_bid_mean",
            "flag_ratio_bid_std",
            "flag_ratio_bid_skew",
            "flag_ratio_bid_kurtosis",
            "flag_ratio_bid_autocorr",
            "flag_ratio_bid_trend",
            "flag_ratio_bid_lz_complexity",
        ]);

        // 卖单flag比例特征
        names.extend(vec![
            "flag_ratio_ask_mean",
            "flag_ratio_ask_std",
            "flag_ratio_ask_skew",
            "flag_ratio_ask_kurtosis",
            "flag_ratio_ask_autocorr",
            "flag_ratio_ask_trend",
            "flag_ratio_ask_lz_complexity",
        ]);
    }

    names.into_iter().map(String::from).collect()
}

fn compute_interval_features(
    trade_times: &[i64],
    trade_flags: &[i32],
    trade_bid_orders: &[i64],
    trade_ask_orders: &[i64],
    trade_volumes: &[i64],
    start_time: i64,
    end_time: i64,
    compute_direction_ratio: bool,
    compute_flag_ratio: bool,
) -> Vec<f64> {
    let mut result = Vec::new();

    // 筛选区间内的交易
    let mut interval_indices: Vec<usize> = Vec::new();
    for i in 0..trade_times.len() {
        let ts = trade_times[i];
        if ts > start_time && ts <= end_time {
            interval_indices.push(i);
        }
    }

    if interval_indices.is_empty() {
        let n_features = 42
            + if compute_direction_ratio { 21 } else { 0 }
            + if compute_flag_ratio { 21 } else { 0 };
        return vec![f64::NAN; n_features];
    }

    // 识别被动订单
    let mut all_passive_orders: Vec<i64> = Vec::new();
    let mut passive_bid_orders: Vec<i64> = Vec::new();
    let mut passive_ask_orders: Vec<i64> = Vec::new();
    let mut order_volumes: HashMap<i64, i64> = HashMap::new();
    let mut bid_order_volumes: HashMap<i64, i64> = HashMap::new();
    let mut ask_order_volumes: HashMap<i64, i64> = HashMap::new();

    // 建立订单到索引和方向的映射
    let mut order_info: HashMap<i64, (usize, bool)> = HashMap::new();

    for &idx in &interval_indices {
        let flag = trade_flags[idx];
        let bid_order = trade_bid_orders[idx];
        let ask_order = trade_ask_orders[idx];
        let volume = trade_volumes[idx];

        if flag == 66 {
            let order = ask_order;
            all_passive_orders.push(order);
            passive_ask_orders.push(order);
            *order_volumes.entry(order).or_insert(0) += volume;
            *ask_order_volumes.entry(order).or_insert(0) += volume;
            order_info.insert(order, (idx, false));
        } else if flag == 83 {
            let order = bid_order;
            all_passive_orders.push(order);
            passive_bid_orders.push(order);
            *order_volumes.entry(order).or_insert(0) += volume;
            *bid_order_volumes.entry(order).or_insert(0) += volume;
            order_info.insert(order, (idx, true));
        }
    }

    // 特征1-7: 全部被动订单编号
    let all_orders_f64: Vec<f64> = all_passive_orders.iter().map(|&x| x as f64).collect();
    result.extend(compute_statistics(&all_orders_f64));

    // 特征8-14: 被动买单编号
    let bid_orders_f64: Vec<f64> = passive_bid_orders.iter().map(|&x| x as f64).collect();
    result.extend(compute_statistics(&bid_orders_f64));

    // 特征15-21: 被动卖单编号
    let ask_orders_f64: Vec<f64> = passive_ask_orders.iter().map(|&x| x as f64).collect();
    result.extend(compute_statistics(&ask_orders_f64));

    // 特征22-28: 全部订单体量
    let volumes: Vec<f64> = order_volumes.values().map(|&v| v as f64).collect();
    result.extend(compute_statistics(&volumes));

    // 特征29-35: 买单体量
    let bid_volumes: Vec<f64> = bid_order_volumes.values().map(|&v| v as f64).collect();
    result.extend(compute_statistics(&bid_volumes));

    // 特征36-42: 卖单体量
    let ask_volumes: Vec<f64> = ask_order_volumes.values().map(|&v| v as f64).collect();
    result.extend(compute_statistics(&ask_volumes));

    if compute_direction_ratio {
        // 特征43-49: 全部订单方向比例
        let direction_ratios_all = compute_direction_ratios(&all_passive_orders, &order_info, trade_flags);
        result.extend(compute_statistics(&direction_ratios_all));

        // 特征50-56: 买单方向比例
        let direction_ratios_bid = compute_direction_ratios(&passive_bid_orders, &order_info, trade_flags);
        result.extend(compute_statistics(&direction_ratios_bid));

        // 特征57-63: 卖单方向比例
        let direction_ratios_ask = compute_direction_ratios(&passive_ask_orders, &order_info, trade_flags);
        result.extend(compute_statistics(&direction_ratios_ask));
    }

    if compute_flag_ratio {
        // 特征64-70: 全部订单flag比例
        let flag_ratios_all = compute_flag_ratios(&all_passive_orders, &order_info, trade_flags);
        result.extend(compute_statistics(&flag_ratios_all));

        // 特征71-77: 买单flag比例
        let flag_ratios_bid = compute_flag_ratios(&passive_bid_orders, &order_info, trade_flags);
        result.extend(compute_statistics(&flag_ratios_bid));

        // 特征78-84: 卖单flag比例
        let flag_ratios_ask = compute_flag_ratios(&passive_ask_orders, &order_info, trade_flags);
        result.extend(compute_statistics(&flag_ratios_ask));
    }

    result
}

fn compute_direction_ratios(
    orders: &[i64],
    order_info: &HashMap<i64, (usize, bool)>,
    trade_flags: &[i32],
) -> Vec<f64> {
    let mut ratios = Vec::new();

    for &order in orders {
        if let Some(&(idx, is_passive_bid)) = order_info.get(&order) {
            let start_idx = if idx >= 50 { idx - 50 } else { 0 };
            let end_idx = (idx + 50).min(trade_flags.len());

            let mut same_direction_count = 0;
            let mut total_count = 0;

            for i in start_idx..end_idx {
                if i == idx {
                    continue;
                }
                total_count += 1;

                // 判断第i笔成交的被动方是被动买单还是被动卖单
                // flag == 66: 主动买，被动方是ask_order（被动卖单）
                // flag == 83: 主动卖，被动方是bid_order（被动买单）
                let other_flag = trade_flags[i];
                let other_is_passive_bid = other_flag == 83;

                if is_passive_bid == other_is_passive_bid {
                    same_direction_count += 1;
                }
            }

            if total_count > 0 {
                ratios.push(same_direction_count as f64 / total_count as f64);
            }
        }
    }

    ratios
}

fn compute_flag_ratios(
    orders: &[i64],
    order_info: &HashMap<i64, (usize, bool)>,
    trade_flags: &[i32],
) -> Vec<f64> {
    let mut ratios = Vec::new();

    for &order in orders {
        if let Some(&(idx, _)) = order_info.get(&order) {
            let start_idx = if idx >= 50 { idx - 50 } else { 0 };
            let end_idx = (idx + 50).min(trade_flags.len());

            let mut same_flag_count = 0;
            let mut total_count = 0;

            let my_flag = trade_flags[idx];

            for i in start_idx..end_idx {
                if i == idx {
                    continue;
                }
                total_count += 1;

                if trade_flags[i] == my_flag {
                    same_flag_count += 1;
                }
            }

            if total_count > 0 {
                ratios.push(same_flag_count as f64 / total_count as f64);
            }
        }
    }

    ratios
}

fn compute_statistics(values: &[f64]) -> [f64; 7] {
    if values.is_empty() {
        return [f64::NAN; 7];
    }

    let n = values.len();

    let mean: f64 = values.iter().sum::<f64>() / n as f64;

    let variance = values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n as f64;
    let std = variance.sqrt();

    let m3 = values.iter().map(|&x| (x - mean).powi(3)).sum::<f64>() / n as f64;
    let skewness = if std > 0.0 {
        m3 / (std.powi(3))
    } else {
        f64::NAN
    };

    let m4 = values.iter().map(|&x| (x - mean).powi(4)).sum::<f64>() / n as f64;
    let kurtosis = if std > 0.0 {
        m4 / (std.powi(4)) - 3.0
    } else {
        f64::NAN
    };

    let autocorr = if n > 1 {
        let sum_xy: f64 = values
            .iter()
            .zip(values.iter().skip(1))
            .map(|(x, y)| *x * *y)
            .sum();

        let mean_y = values.iter().skip(1).sum::<f64>() / (n - 1) as f64;

        let numerator = sum_xy - (n - 1) as f64 * mean * mean_y;
        let var_x: f64 = values.iter().take(n - 1).map(|x| (x - mean).powi(2)).sum();
        let var_y: f64 = values.iter().skip(1).map(|y| (y - mean_y).powi(2)).sum();

        if var_x > 0.0 && var_y > 0.0 {
            numerator / (var_x * var_y).sqrt()
        } else {
            f64::NAN
        }
    } else {
        f64::NAN
    };

    let trend = if n > 1 {
        let x_mean = (n - 1) as f64 / 2.0;
        let sum_xy: f64 = values.iter().enumerate().map(|(i, &x)| i as f64 * x).sum();

        let numerator = sum_xy - n as f64 * x_mean * mean;
        let var_x: f64 = (0..n).map(|i| (i as f64 - x_mean).powi(2)).sum();

        if var_x > 0.0 {
            numerator / var_x
        } else {
            0.0
        }
    } else {
        0.0
    };

    let lz_complexity = if !values.is_empty() {
        calc_lz_complexity(values)
    } else {
        0.0
    };

    [
        mean,
        std,
        skewness,
        kurtosis,
        autocorr,
        trend,
        lz_complexity,
    ]
}

fn calc_lz_complexity(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }

    let median = {
        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        sorted[sorted.len() / 2]
    };

    let s: String = values
        .iter()
        .map(|&x| if x >= median { '1' } else { '0' })
        .collect();

    let n = s.len();
    let mut i = 0;
    let mut complexity = 1;

    while i < n {
        let mut l = 1;
        while i + l <= n {
            let substr = &s[i..i + l];
            if s[..i + l - 1].contains(substr) {
                l += 1;
            } else {
                break;
            }
        }
        i += l;
        if i < n {
            complexity += 1;
        }
    }

    complexity as f64 / n as f64
}
