use pyo3::prelude::*;
use rayon::prelude::*;
use std::collections::HashMap;

#[derive(Debug, Clone, Copy)]
pub enum IntervalMode {
    FullDay,          // 全天最早到最晚
    HighLowRange,     // 最高价到最低价时间范围
    PerMinute,        // 每分钟
    VolumePercentile, // 按成交量百分比分割
    LocalHighs,       // 相邻局部高点
    LocalLows,        // 相邻局部低点
    HighToLow,        // 局部高点到下一个局部低点
    LowToHigh,        // 局部低点到下一个局部高点
    NewHighs,         // 相邻创新高点
    NewLows,          // 相邻创新低点
}

#[pyfunction]
#[pyo3(signature = (
    exchtime_trade,
    price_trade,
    volume_trade,
    exchtime_ask,
    price_ask,
    volume_ask,
    exchtime_bid,
    price_bid,
    volume_bid,
    mode = "full_day",
    percentile_count = 100
))]
pub fn price_volume_orderbook_correlation(
    exchtime_trade: Vec<f64>,
    price_trade: Vec<f64>,
    volume_trade: Vec<f64>,
    exchtime_ask: Vec<f64>,
    price_ask: Vec<f64>,
    volume_ask: Vec<f64>,
    exchtime_bid: Vec<f64>,
    price_bid: Vec<f64>,
    volume_bid: Vec<f64>,
    mode: &str,
    percentile_count: usize,
) -> PyResult<(Vec<Vec<f64>>, Vec<String>)> {
    // 将纳秒转换为秒
    let trade_times: Vec<f64> = exchtime_trade
        .iter()
        .map(|&t| t / 1_000_000_000.0)
        .collect();
    let ask_times: Vec<f64> = exchtime_ask.iter().map(|&t| t / 1_000_000_000.0).collect();
    let bid_times: Vec<f64> = exchtime_bid.iter().map(|&t| t / 1_000_000_000.0).collect();

    // 解析模式
    let interval_mode = match mode {
        "full_day" => IntervalMode::FullDay,
        "high_low_range" => IntervalMode::HighLowRange,
        "per_minute" => IntervalMode::PerMinute,
        "volume_percentile" => IntervalMode::VolumePercentile,
        "local_highs" => IntervalMode::LocalHighs,
        "local_lows" => IntervalMode::LocalLows,
        "high_to_low" => IntervalMode::HighToLow,
        "low_to_high" => IntervalMode::LowToHigh,
        "new_highs" => IntervalMode::NewHighs,
        "new_lows" => IntervalMode::NewLows,
        _ => {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Invalid mode",
            ))
        }
    };

    // 生成时间区间
    let intervals = generate_time_intervals(
        &trade_times,
        &price_trade,
        &volume_trade,
        interval_mode,
        percentile_count,
    )?;

    // 并行计算每个区间的相关性
    let results: Vec<Vec<f64>> = intervals
        .par_iter()
        .map(|(t1, t2)| {
            calculate_interval_correlation(
                &trade_times,
                &price_trade,
                &volume_trade,
                &ask_times,
                &price_ask,
                &volume_ask,
                &bid_times,
                &price_bid,
                &volume_bid,
                *t1,
                *t2,
            )
        })
        .collect();

    let column_names = vec![
        "成交量与卖出挂单量相关性".to_string(),
        "成交量与买入挂单量相关性".to_string(),
        "成交量与买卖挂单量差相关性".to_string(),
        "成交量与买卖挂单量差绝对值相关性".to_string(),
    ];

    Ok((results, column_names))
}

fn generate_time_intervals(
    trade_times: &[f64],
    price_trade: &[f64],
    volume_trade: &[f64],
    mode: IntervalMode,
    percentile_count: usize,
) -> PyResult<Vec<(f64, f64)>> {
    match mode {
        IntervalMode::FullDay => {
            if trade_times.is_empty() {
                return Ok(vec![]);
            }
            Ok(vec![(trade_times[0], trade_times[trade_times.len() - 1])])
        }

        IntervalMode::HighLowRange => {
            if price_trade.is_empty() {
                return Ok(vec![]);
            }
            let (min_idx, max_idx) = find_min_max_indices(price_trade);
            let start_time = trade_times[min_idx.min(max_idx)];
            let end_time = trade_times[min_idx.max(max_idx)];
            Ok(vec![(start_time, end_time)])
        }

        IntervalMode::PerMinute => {
            if trade_times.is_empty() {
                return Ok(vec![]);
            }
            let start_time = trade_times[0].floor();
            let end_time = trade_times[trade_times.len() - 1].ceil();
            let mut intervals = Vec::new();

            let mut current_minute = start_time;
            while current_minute < end_time {
                intervals.push((current_minute, current_minute + 60.0));
                current_minute += 60.0;
            }
            Ok(intervals)
        }

        IntervalMode::VolumePercentile => {
            if volume_trade.is_empty() {
                return Ok(vec![]);
            }
            let total_volume: f64 = volume_trade.iter().sum();
            let volume_per_percentile = total_volume / percentile_count as f64;

            let mut intervals = Vec::new();
            let mut cumulative_volume = 0.0;
            let mut start_idx = 0;

            for i in 0..volume_trade.len() {
                cumulative_volume += volume_trade[i];
                if cumulative_volume >= volume_per_percentile {
                    intervals.push((trade_times[start_idx], trade_times[i]));
                    start_idx = i + 1;
                    cumulative_volume = 0.0;
                }
            }

            // 处理剩余的数据
            if start_idx < trade_times.len() {
                intervals.push((trade_times[start_idx], trade_times[trade_times.len() - 1]));
            }

            Ok(intervals)
        }

        IntervalMode::LocalHighs => {
            let local_highs = find_local_extrema(price_trade, true);
            let mut intervals = Vec::new();

            for i in 0..local_highs.len().saturating_sub(1) {
                let start_time = trade_times[local_highs[i]];
                let end_time = trade_times[local_highs[i + 1]];
                intervals.push((start_time, end_time));
            }

            Ok(intervals)
        }

        IntervalMode::LocalLows => {
            let local_lows = find_local_extrema(price_trade, false);
            let mut intervals = Vec::new();

            for i in 0..local_lows.len().saturating_sub(1) {
                let start_time = trade_times[local_lows[i]];
                let end_time = trade_times[local_lows[i + 1]];
                intervals.push((start_time, end_time));
            }

            Ok(intervals)
        }

        IntervalMode::HighToLow => {
            let local_highs = find_local_extrema(price_trade, true);
            let local_lows = find_local_extrema(price_trade, false);
            let mut intervals = Vec::new();

            for &high_idx in &local_highs {
                if let Some(&low_idx) = local_lows.iter().find(|&&idx| idx > high_idx) {
                    intervals.push((trade_times[high_idx], trade_times[low_idx]));
                }
            }

            Ok(intervals)
        }

        IntervalMode::LowToHigh => {
            let local_highs = find_local_extrema(price_trade, true);
            let local_lows = find_local_extrema(price_trade, false);
            let mut intervals = Vec::new();

            for &low_idx in &local_lows {
                if let Some(&high_idx) = local_highs.iter().find(|&&idx| idx > low_idx) {
                    intervals.push((trade_times[low_idx], trade_times[high_idx]));
                }
            }

            Ok(intervals)
        }

        IntervalMode::NewHighs => {
            let new_highs = find_new_extrema(price_trade, true);
            let mut intervals = Vec::new();

            for i in 0..new_highs.len().saturating_sub(1) {
                let start_time = trade_times[new_highs[i]];
                let end_time = trade_times[new_highs[i + 1]];
                intervals.push((start_time, end_time));
            }

            Ok(intervals)
        }

        IntervalMode::NewLows => {
            let new_lows = find_new_extrema(price_trade, false);
            let mut intervals = Vec::new();

            for i in 0..new_lows.len().saturating_sub(1) {
                let start_time = trade_times[new_lows[i]];
                let end_time = trade_times[new_lows[i + 1]];
                intervals.push((start_time, end_time));
            }

            Ok(intervals)
        }
    }
}

fn find_min_max_indices(prices: &[f64]) -> (usize, usize) {
    let mut min_idx = 0;
    let mut max_idx = 0;

    for (i, &price) in prices.iter().enumerate() {
        if price < prices[min_idx] {
            min_idx = i;
        }
        if price > prices[max_idx] {
            max_idx = i;
        }
    }

    (min_idx, max_idx)
}

fn find_local_extrema(prices: &[f64], find_highs: bool) -> Vec<usize> {
    let mut extrema = Vec::new();

    if prices.len() < 3 {
        return extrema;
    }

    for i in 1..prices.len() - 1 {
        let curr = prices[i];
        let prev = prices[i - 1];

        // 找到后面第一个与当前价格不同的价格
        let mut next_diff_idx = i + 1;
        while next_diff_idx < prices.len() && prices[next_diff_idx] == curr {
            next_diff_idx += 1;
        }

        // 如果没有找到不同的价格，跳过
        if next_diff_idx >= prices.len() {
            continue;
        }

        let next_diff = prices[next_diff_idx];

        if find_highs {
            // 寻找局部高点：严格大于前一个价格，且严格大于后面第一个不同的价格
            if curr > prev && curr > next_diff {
                extrema.push(i);
            }
        } else {
            // 寻找局部低点：严格小于前一个价格，且严格小于后面第一个不同的价格
            if curr < prev && curr < next_diff {
                extrema.push(i);
            }
        }
    }

    extrema
}

fn find_new_extrema(prices: &[f64], find_highs: bool) -> Vec<usize> {
    let mut extrema = Vec::new();

    if prices.len() < 2 {
        return extrema;
    }

    for i in 0..prices.len() - 1 {
        let curr = prices[i];

        // 检查是否比之前所有价格都高/低
        let is_new_extreme = if i == 0 {
            // 第一个价格总是算作创新极值的候选
            true
        } else if find_highs {
            // 创新高：比之前所有价格都高
            prices[..i].iter().all(|&prev| curr > prev)
        } else {
            // 创新低：比之前所有价格都低
            prices[..i].iter().all(|&prev| curr < prev)
        };

        if !is_new_extreme {
            continue;
        }

        // 找到后面第一个与当前价格不同的价格
        let mut next_diff_idx = i + 1;
        while next_diff_idx < prices.len() && prices[next_diff_idx] == curr {
            next_diff_idx += 1;
        }

        // 如果没有找到不同的价格，但是是创新极值，也算作有效
        if next_diff_idx >= prices.len() {
            extrema.push(i);
            continue;
        }

        let next_diff = prices[next_diff_idx];

        // 检查是否也比后面第一个不同的价格高/低
        if find_highs {
            if curr > next_diff {
                extrema.push(i);
            }
        } else {
            if curr < next_diff {
                extrema.push(i);
            }
        }
    }

    extrema
}

fn calculate_interval_correlation(
    trade_times: &[f64],
    price_trade: &[f64],
    volume_trade: &[f64],
    ask_times: &[f64],
    price_ask: &[f64],
    volume_ask: &[f64],
    bid_times: &[f64],
    price_bid: &[f64],
    volume_bid: &[f64],
    t1: f64,
    t2: f64,
) -> Vec<f64> {
    // 获取时间范围内的交易数据
    let trade_data = extract_time_range_data(trade_times, price_trade, volume_trade, t1, t2);
    let ask_data = extract_time_range_data(ask_times, price_ask, volume_ask, t1, t2);
    let bid_data = extract_time_range_data(bid_times, price_bid, volume_bid, t1, t2);

    // 按价格聚合数据
    let trade_aggregated = aggregate_by_price(&trade_data);
    let ask_aggregated = aggregate_by_price_avg(&ask_data);
    let bid_aggregated = aggregate_by_price_avg(&bid_data);

    // 找到价格交集
    let common_prices = find_common_prices(&trade_aggregated, &ask_aggregated, &bid_aggregated);

    if common_prices.len() < 2 {
        return vec![0.0, 0.0, 0.0, 0.0];
    }

    // 构建对应的数量序列
    let series_t: Vec<f64> = common_prices
        .iter()
        .map(|&price| {
            let key = (price * 1000.0) as i64;
            *trade_aggregated.get(&key).unwrap_or(&0.0)
        })
        .collect();

    let series_a: Vec<f64> = common_prices
        .iter()
        .map(|&price| {
            let key = (price * 1000.0) as i64;
            *ask_aggregated.get(&key).unwrap_or(&0.0)
        })
        .collect();

    let series_b: Vec<f64> = common_prices
        .iter()
        .map(|&price| {
            let key = (price * 1000.0) as i64;
            *bid_aggregated.get(&key).unwrap_or(&0.0)
        })
        .collect();

    // 计算相关性
    let corr_ta = calculate_correlation(&series_t, &series_a);
    let corr_tb = calculate_correlation(&series_t, &series_b);

    let series_diff: Vec<f64> = series_a
        .iter()
        .zip(series_b.iter())
        .map(|(a, b)| a - b)
        .collect();
    let corr_t_diff = calculate_correlation(&series_t, &series_diff);

    let series_abs_diff: Vec<f64> = series_diff.iter().map(|x| x.abs()).collect();
    let corr_t_abs_diff = calculate_correlation(&series_t, &series_abs_diff);

    vec![corr_ta, corr_tb, corr_t_diff, corr_t_abs_diff]
}

fn extract_time_range_data(
    times: &[f64],
    prices: &[f64],
    volumes: &[f64],
    t1: f64,
    t2: f64,
) -> Vec<(f64, f64)> {
    let mut data = Vec::new();

    for i in 0..times.len() {
        if times[i] >= t1 && times[i] <= t2 {
            data.push((prices[i], volumes[i]));
        }
    }

    data
}

fn aggregate_by_price(data: &[(f64, f64)]) -> HashMap<i64, f64> {
    let mut aggregated = HashMap::new();

    for &(price, volume) in data {
        let price_key = (price * 1000.0) as i64; // 精确到0.001
        *aggregated.entry(price_key).or_insert(0.0) += volume;
    }

    aggregated
}

fn aggregate_by_price_avg(data: &[(f64, f64)]) -> HashMap<i64, f64> {
    let mut sums = HashMap::new();
    let mut counts = HashMap::new();

    for &(price, volume) in data {
        let price_key = (price * 1000.0) as i64;
        *sums.entry(price_key).or_insert(0.0) += volume;
        *counts.entry(price_key).or_insert(0) += 1;
    }

    let mut averages = HashMap::new();
    for (&price_key, &sum) in &sums {
        let count = counts[&price_key] as f64;
        averages.insert(price_key, sum / count);
    }

    averages
}

fn find_common_prices(
    trade_map: &HashMap<i64, f64>,
    ask_map: &HashMap<i64, f64>,
    bid_map: &HashMap<i64, f64>,
) -> Vec<f64> {
    let mut common_prices = Vec::new();

    for &price_key in trade_map.keys() {
        if ask_map.contains_key(&price_key) && bid_map.contains_key(&price_key) {
            common_prices.push(price_key as f64 / 1000.0);
        }
    }

    common_prices.sort_by(|a, b| a.partial_cmp(b).unwrap());
    common_prices
}

fn calculate_correlation(x: &[f64], y: &[f64]) -> f64 {
    if x.len() != y.len() || x.len() < 2 {
        return 0.0;
    }

    let n = x.len() as f64;
    let mean_x = x.iter().sum::<f64>() / n;
    let mean_y = y.iter().sum::<f64>() / n;

    let mut numerator = 0.0;
    let mut sum_sq_x = 0.0;
    let mut sum_sq_y = 0.0;

    for i in 0..x.len() {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        numerator += dx * dy;
        sum_sq_x += dx * dx;
        sum_sq_y += dy * dy;
    }

    let denominator = (sum_sq_x * sum_sq_y).sqrt();

    if denominator == 0.0 {
        0.0
    } else {
        numerator / denominator
    }
}
