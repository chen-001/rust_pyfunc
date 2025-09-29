use std::cmp::Ordering;
use std::cmp::min;
use ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2};
use pyo3::prelude::*;
use pyo3::types::PyDict;

const LEVEL_CAP: usize = 10;

#[derive(Clone, Debug)]
struct BookLevel {
    level: i32,
    price: f64,
    volume: f64,
}

#[derive(Clone, Debug)]
struct Snapshot {
    timestamp: i64,
    bids: Vec<BookLevel>,
    asks: Vec<BookLevel>,
}

impl Snapshot {
    fn new(timestamp: i64, mut bids: Vec<BookLevel>, mut asks: Vec<BookLevel>) -> Self {
        bids.sort_by_key(|lvl| lvl.level);
        asks.sort_by_key(|lvl| lvl.level);
        Self { timestamp, bids, asks }
    }

    fn find_bid_by_price(&self, price: f64, eps: f64) -> Option<&BookLevel> {
        self.bids.iter().find(|lvl| (lvl.price - price).abs() <= eps)
    }

    fn find_ask_by_price(&self, price: f64, eps: f64) -> Option<&BookLevel> {
        self.asks.iter().find(|lvl| (lvl.price - price).abs() <= eps)
    }

    fn best_bid(&self) -> Option<&BookLevel> {
        self.bids.iter().find(|lvl| lvl.level == 1)
    }

    fn best_ask(&self) -> Option<&BookLevel> {
        self.asks.iter().find(|lvl| lvl.level == 1)
    }

    fn mid_price(&self) -> Option<f64> {
        match (self.best_bid(), self.best_ask()) {
            (Some(b), Some(a)) => Some((b.price + a.price) * 0.5),
            _ => None,
        }
    }

    fn spread(&self) -> Option<f64> {
        match (self.best_bid(), self.best_ask()) {
            (Some(b), Some(a)) => Some(a.price - b.price),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
enum BType {
    BuyBreak,    // Price breaks below support level (buy side)
    SellBreak,   // Price breaks above resistance level (sell side)
}

#[derive(Clone, Debug)]
struct BTimingPoint {
    price: f64,
    trade_idx: usize,
    time_ns: i64,
    b_type: BType,
}

#[derive(Clone, Debug)]
struct BSegment {
    start_b: BTimingPoint,
    end_b: BTimingPoint,
    start_time: i64,
    end_time: i64,
    start_trade_idx: usize,
    end_trade_idx: usize,
}

#[derive(Clone, Debug)]
struct PriceLevelResult {
    price: f64,
    buy_segments: Vec<BSegment>,
    sell_segments: Vec<BSegment>,
    buy_segment_count: usize,
    sell_segment_count: usize,
    buy_metrics: Option<Vec<f64>>,  // Enhanced metrics for buy side (105 dimensions)
    sell_metrics: Option<Vec<f64>>, // Enhanced metrics for sell side (105 dimensions)
}

#[inline]
fn ns_to_ms(ns: i64) -> f64 {
    ns as f64 / 1_000_000.0
}

#[inline]
fn ns_to_s(ns: i64) -> f64 {
    ns as f64 / 1_000_000_000.0
}

fn compare_with_eps(value: f64, target: f64, eps: f64) -> Ordering {
    if value > target + eps {
        Ordering::Greater
    } else if value < target - eps {
        Ordering::Less
    } else {
        Ordering::Equal
    }
}

fn detect_b_points_by_side(
    price_level: f64,
    trades_time: &[i64],
    trades_price: &[f64],
    drop_threshold: f64,
    rise_threshold: f64,
    eps: f64,
) -> (Vec<BTimingPoint>, Vec<BTimingPoint>) {
    let mut buy_b_points = Vec::new();
    let mut sell_b_points = Vec::new();
    let mut last_relation = Ordering::Equal;

    for (idx, (&time, &price)) in trades_time.iter().zip(trades_price.iter()).enumerate() {
        let current_relation = compare_with_eps(price, price_level, eps);

        match (last_relation, current_relation) {
            // Buy side: price was at or above level, now breaks below
            (Ordering::Equal, Ordering::Less) | (Ordering::Greater, Ordering::Less) => {
                let threshold_price = price_level - drop_threshold;
                if price <= threshold_price + eps {
                    buy_b_points.push(BTimingPoint {
                        price: price_level,
                        trade_idx: idx,
                        time_ns: time,
                        b_type: BType::BuyBreak,
                    });
                }
            }
            // Sell side: price was at or below level, now breaks above
            (Ordering::Equal, Ordering::Greater) | (Ordering::Less, Ordering::Greater) => {
                let threshold_price = price_level + rise_threshold;
                if price >= threshold_price - eps {
                    sell_b_points.push(BTimingPoint {
                        price: price_level,
                        trade_idx: idx,
                        time_ns: time,
                        b_type: BType::SellBreak,
                    });
                }
            }
            _ => {}
        }

        last_relation = current_relation;
    }

    (buy_b_points, sell_b_points)
}

fn build_segments_for_side(
    b_points: &[BTimingPoint],
    side: BType,
) -> Vec<BSegment> {
    // Filter points for this specific side
    let side_points: Vec<&BTimingPoint> = b_points.iter()
        .filter(|point| point.b_type == side)
        .collect();

    if side_points.len() < 2 {
        return Vec::new();
    }

    let mut segments = Vec::new();

    for i in 0..side_points.len() - 1 {
        let start_b = side_points[i];
        let end_b = side_points[i + 1];

        segments.push(BSegment {
            start_b: start_b.clone(),
            end_b: end_b.clone(),
            start_time: start_b.time_ns,
            end_time: end_b.time_ns,
            start_trade_idx: start_b.trade_idx,
            end_trade_idx: end_b.trade_idx,
        });
    }

    segments
}

#[derive(Default, Clone, Debug)]
struct SegmentMetrics {
    values: Vec<f64>,
}

impl SegmentMetrics {
    fn new() -> Self {
        Self { values: Vec::new() }
    }

    fn add(&mut self, value: f64) {
        self.values.push(value);
    }
}

/// 计算21个基础特征指标
fn compute_segment_metrics(
    segments: &[BSegment],
    trades_time: &[i64],
    trades_price: &[f64],
    trades_volume: &[f64],
    trades_flag: &[i32],
    snapshots: &[Snapshot],
    price_level: f64,
    is_buy_side: bool,
    eps: f64,
) -> Vec<SegmentMetrics> {
    let mut all_metrics = Vec::new();

    for segment in segments {
        let mut metrics = SegmentMetrics::new();
        let start_time = segment.start_time;
        let end_time = segment.end_time;
        let start_idx = segment.start_trade_idx;
        let end_idx = segment.end_trade_idx;

        // 1. duration_ms: B段持续时间（毫秒）
        metrics.add(ns_to_ms(end_time - start_time));

        // Trade statistics in segment
        let mut total_volume = 0.0;
        let mut buy_volume = 0.0;
        let mut sell_volume = 0.0;
        let mut trade_count = 0;
        let mut min_price = f64::INFINITY;
        let mut max_price = f64::NEG_INFINITY;
        let mut sum_price_volume = 0.0;

        for idx in start_idx..=end_idx.min(trades_price.len() - 1) {
            if trades_time[idx] < start_time || trades_time[idx] > end_time {
                continue;
            }

            let price = trades_price[idx];
            let volume = trades_volume[idx];
            let flag = trades_flag[idx];

            total_volume += volume;
            trade_count += 1;
            min_price = min_price.min(price);
            max_price = max_price.max(price);
            sum_price_volume += price * volume;

            match flag {
                66 => buy_volume += volume,  // 'B'
                83 => sell_volume += volume, // 'S'
                _ => {}
            }
        }

        // 2. total_volume: 段内总成交量
        metrics.add(total_volume);
        // 3. trade_count: 段内交易笔数
        metrics.add(trade_count as f64);
        // 4. vwap: 成交量加权平均价格
        if total_volume > 0.0 {
            metrics.add(sum_price_volume / total_volume);
        } else {
            metrics.add(f64::NAN);
        }
        // 5. min_price: 段内最低价格
        metrics.add(min_price);
        // 6. max_price: 段内最高价格
        metrics.add(max_price);
        // 7. buy_ratio: 买成交量占比
        // 8. sell_ratio: 卖成交量占比
        if total_volume > 0.0 {
            metrics.add(buy_volume / total_volume);
            metrics.add(sell_volume / total_volume);
        } else {
            metrics.add(f64::NAN);
            metrics.add(f64::NAN);
        }

        // Spread and price analysis
        let start_spread = find_spread_at_time(snapshots, start_time);
        let end_spread = find_spread_at_time(snapshots, end_time);
        let start_mid = find_mid_at_time(snapshots, start_time);
        let end_mid = find_mid_at_time(snapshots, end_time);

        // 9. start_spread: 段开始时的买卖价差
        if let Some(spread) = start_spread {
            metrics.add(spread);
        } else {
            metrics.add(f64::NAN);
        }
        // 10. spread_change: 价差变化量
        if let (Some(start_s), Some(end_s)) = (start_spread, end_spread) {
            metrics.add(end_s - start_s);
        } else {
            metrics.add(f64::NAN);
        }
        // 11. mid_return_bp: 中间价收益率（基点）
        if let (Some(start_m), Some(end_m)) = (start_mid, end_mid) {
            let delta = (end_m - start_m) / price_level;
            metrics.add(10_000.0 * delta);
        } else {
            metrics.add(f64::NAN);
        }

        // 12. start_distance_to_level: 开始时中间价与目标水平的距离
        // 13. end_distance_to_level: 结束时中间价与目标水平的距离
        if let Some(start_m) = start_mid {
            metrics.add((start_m - price_level).abs());
        } else {
            metrics.add(f64::NAN);
        }
        if let Some(end_m) = end_mid {
            metrics.add((end_m - price_level).abs());
        } else {
            metrics.add(f64::NAN);
        }

        // Side-specific features
        if is_buy_side {
            // Buy side features: look at bid depth and behavior
            let (avg_depth, max_depth, min_depth) = compute_bid_depth_stats(
                snapshots, start_time, end_time, price_level, eps
            );
            // 14. avg_bid_depth: 平均买深度
            metrics.add(avg_depth);
            // 15. max_bid_depth: 最大买深度
            metrics.add(max_depth);
            // 16. min_bid_depth: 最小买深度
            metrics.add(min_depth);
        } else {
            // Sell side features: look at ask depth and behavior
            let (avg_depth, max_depth, min_depth) = compute_ask_depth_stats(
                snapshots, start_time, end_time, price_level, eps
            );
            // 14. avg_ask_depth: 平均卖深度
            metrics.add(avg_depth);
            // 15. max_ask_depth: 最大卖深度
            metrics.add(max_depth);
            // 16. min_ask_depth: 最小卖深度
            metrics.add(min_depth);
        }

        // Trade activity at the level
        let (vol_at_level, count_at_level) = compute_trade_at_level(
            start_idx, end_idx, trades_price, trades_volume, trades_time,
            start_time, end_time, price_level, eps
        );
        // 17. vol_at_level: 在目标价格水平的成交量
        metrics.add(vol_at_level);
        // 18. count_at_level: 在目标价格水平的交易笔数
        metrics.add(count_at_level as f64);

        // 19. time_between_ms: 相邻B点时间间隔（与duration_ms相同）
        metrics.add(ns_to_ms(end_time - start_time));
        // 20. trades_between: 相邻B点间交易笔数
        metrics.add((end_idx - start_idx) as f64);

        // Start and end price
        let start_price = trades_price[start_idx.min(trades_price.len() - 1)];
        let end_price = trades_price[end_idx.min(trades_price.len() - 1)];
        // 21. start_price: 段开始时的交易价格
        metrics.add(start_price);
        // 22. end_price: 段结束时的交易价格
        metrics.add(end_price);

        let price_change = (end_price - start_price) / start_price;
        // 23. total_return_bp: 段内总收益率（基点）
        metrics.add(10_000.0 * price_change);

        all_metrics.push(metrics);
    }

    all_metrics
}

fn find_spread_at_time(snapshots: &[Snapshot], target_time: i64) -> Option<f64> {
    let idx = asof_snapshot_index(snapshots, target_time)?;
    snapshots.get(idx)?.spread()
}

fn find_mid_at_time(snapshots: &[Snapshot], target_time: i64) -> Option<f64> {
    let idx = asof_snapshot_index(snapshots, target_time)?;
    snapshots.get(idx)?.mid_price()
}

fn compute_bid_depth_stats(
    snapshots: &[Snapshot],
    start_time: i64,
    end_time: i64,
    price_level: f64,
    eps: f64,
) -> (f64, f64, f64) {
    let start_idx = asof_snapshot_index(snapshots, start_time).unwrap_or(0);
    let end_idx = snapshots.len().min(asof_snapshot_index(snapshots, end_time).unwrap_or(snapshots.len()) + 1);

    let mut depths = Vec::new();

    for i in start_idx..end_idx {
        let snapshot = &snapshots[i];
        if snapshot.timestamp < start_time || snapshot.timestamp > end_time {
            continue;
        }

        // Look for bid depth at this price level
        if let Some(level) = snapshot.find_bid_by_price(price_level, eps) {
            depths.push(level.volume);
        }
    }

    if depths.is_empty() {
        return (0.0, 0.0, 0.0);
    }

    let avg: f64 = depths.iter().sum::<f64>() / depths.len() as f64;
    let max = *depths.iter().fold(&0.0, |a, b| if a > b { a } else { b });
    let min = *depths.iter().fold(&f64::INFINITY, |a, b| if a < b { a } else { b });

    (avg, max, min)
}

fn compute_ask_depth_stats(
    snapshots: &[Snapshot],
    start_time: i64,
    end_time: i64,
    price_level: f64,
    eps: f64,
) -> (f64, f64, f64) {
    let start_idx = asof_snapshot_index(snapshots, start_time).unwrap_or(0);
    let end_idx = snapshots.len().min(asof_snapshot_index(snapshots, end_time).unwrap_or(snapshots.len()) + 1);

    let mut depths = Vec::new();

    for i in start_idx..end_idx {
        let snapshot = &snapshots[i];
        if snapshot.timestamp < start_time || snapshot.timestamp > end_time {
            continue;
        }

        // Look for ask depth at this price level
        if let Some(level) = snapshot.find_ask_by_price(price_level, eps) {
            depths.push(level.volume);
        }
    }

    if depths.is_empty() {
        return (0.0, 0.0, 0.0);
    }

    let avg: f64 = depths.iter().sum::<f64>() / depths.len() as f64;
    let max = *depths.iter().fold(&0.0, |a, b| if a > b { a } else { b });
    let min = *depths.iter().fold(&f64::INFINITY, |a, b| if a < b { a } else { b });

    (avg, max, min)
}

fn compute_trade_at_level(
    start_idx: usize,
    end_idx: usize,
    trades_price: &[f64],
    trades_volume: &[f64],
    trades_time: &[i64],
    start_time: i64,
    end_time: i64,
    price_level: f64,
    eps: f64,
) -> (f64, usize) {
    let mut vol_at_level = 0.0;
    let mut count_at_level = 0;

    for i in start_idx..=end_idx.min(trades_price.len() - 1) {
        if trades_time[i] < start_time || trades_time[i] > end_time {
            continue;
        }

        if (trades_price[i] - price_level).abs() <= eps {
            vol_at_level += trades_volume[i];
            count_at_level += 1;
        }
    }

    (vol_at_level, count_at_level)
}

fn asof_snapshot_index(snapshots: &[Snapshot], ts: i64) -> Option<usize> {
    if snapshots.is_empty() {
        return None;
    }
    let mut left = 0;
    let mut right = snapshots.len();
    while left < right {
        let mid = (left + right) / 2;
        if snapshots[mid].timestamp <= ts {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    if left == 0 {
        None
    } else {
        Some(left - 1)
    }
}

fn build_snapshots(
    ask_exchtime: &[i64],
    bid_exchtime: &[i64],
    ask_price: &[f64],
    ask_volume: &[f64],
    ask_number: &[i32],
    bid_price: &[f64],
    bid_volume: &[f64],
    bid_number: &[i32],
) -> Vec<Snapshot> {
    let mut snapshots = Vec::new();
    let mut i = 0;
    let mut j = 0;
    let ask_len = ask_exchtime.len();
    let bid_len = bid_exchtime.len();

    while i < ask_len || j < bid_len {
        let next_time = match (ask_exchtime.get(i), bid_exchtime.get(j)) {
            (Some(&at), Some(&bt)) => at.min(bt),
            (Some(&at), None) => at,
            (None, Some(&bt)) => bt,
            (None, None) => break,
        };

        let mut bids = Vec::with_capacity(LEVEL_CAP);
        while j < bid_len && bid_exchtime[j] == next_time {
            bids.push(BookLevel {
                level: bid_number[j],
                price: bid_price[j],
                volume: bid_volume[j],
            });
            j += 1;
        }

        let mut asks = Vec::with_capacity(LEVEL_CAP);
        while i < ask_len && ask_exchtime[i] == next_time {
            asks.push(BookLevel {
                level: ask_number[i],
                price: ask_price[i],
                volume: ask_volume[i],
            });
            i += 1;
        }

        if !bids.is_empty() || !asks.is_empty() {
            snapshots.push(Snapshot::new(next_time, bids, asks));
        }
    }

    snapshots
}

fn build_price_grid(
    trades: &[f64],
    tick: f64,
) -> Vec<f64> {
    let mut prices: Vec<f64> = trades.iter().copied().collect();

    if tick > 0.0 {
        for value in &mut prices {
            *value = (*value / tick).round() * tick;
        }
    }

    prices.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    prices.dedup_by(|a, b| (*a - *b).abs() <= tick / 10.0);
    prices
}

/// 计算皮尔逊相关系数
fn pearson_correlation(x: &[f64], y: &[f64]) -> f64 {
    if x.len() != y.len() || x.len() < 2 {
        return f64::NAN;
    }

    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    let mut sum_xy = 0.0;
    let mut sum_x2 = 0.0;
    let mut sum_y2 = 0.0;
    let n = x.len() as f64;

    for i in 0..x.len() {
        let xi = x[i];
        let yi = y[i];
        sum_x += xi;
        sum_y += yi;
        sum_xy += xi * yi;
        sum_x2 += xi * xi;
        sum_y2 += yi * yi;
    }

    let numerator = n * sum_xy - sum_x * sum_y;
    let denominator = (n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y);

    if denominator <= 0.0 {
        return f64::NAN;
    }

    numerator / denominator.sqrt()
}

/// 计算自相关性
fn calculate_autocorrelation(data: &[f64]) -> f64 {
    if data.len() < 3 {
        return f64::NAN;
    }

    // 计算一阶自相关性 (lag=1)
    let mut sum_xy = 0.0;
    let mut sum_x2 = 0.0;
    let mut sum_y2 = 0.0;
    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    let n = (data.len() - 1) as f64;

    for i in 0..data.len() - 1 {
        let xi = data[i];
        let yi = data[i + 1];
        sum_x += xi;
        sum_y += yi;
        sum_xy += xi * yi;
        sum_x2 += xi * xi;
        sum_y2 += yi * yi;
    }

    let numerator = n * sum_xy - sum_x * sum_y;
    let denominator = (n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y);

    if denominator <= 0.0 {
        return f64::NAN;
    }

    numerator / denominator.sqrt()
}

/// 增强版聚合函数 - 计算多种统计指标（每个基础指标生成5个合成指标）
fn enhanced_aggregate_metrics(metrics_list: &[SegmentMetrics]) -> Option<Vec<f64>> {
    if metrics_list.is_empty() {
        return None;
    }

    let expected_len = metrics_list[0].values.len();
    let mut all_results = Vec::new();

    // 对每个原始指标计算多种统计指标
    for metric_idx in 0..expected_len {
        let mut values = Vec::new();

        // 收集当前指标的原始值
        for metric in metrics_list {
            if metric_idx < metric.values.len() {
                let val = metric.values[metric_idx];
                if !val.is_nan() {
                    values.push(val);
                }
            }
        }

        if values.is_empty() {
            // 如果没有有效值，所有统计指标都用NaN
            all_results.push(f64::NAN); // mean
            all_results.push(f64::NAN); // std/mean (CV)
            all_results.push(f64::NAN); // max
            all_results.push(f64::NAN); // corr with sequence
            all_results.push(f64::NAN); // abs corr with sequence
            all_results.push(f64::NAN); // autocorrelation
            continue;
        }

        if values.len() == 1 {
            // 只有一个值的情况下
            let val = values[0];
            all_results.push(val); // mean = 唯一值
            all_results.push(f64::NAN); // std/mean (无意义)
            all_results.push(val); // max = 唯一值
            all_results.push(f64::NAN); // corr with sequence (无意义)
            all_results.push(f64::NAN); // abs corr with sequence (无意义)
            all_results.push(f64::NAN); // autocorrelation (无意义)
            continue;
        }

        // 计算平均值
        let mean: f64 = values.iter().sum::<f64>() / values.len() as f64;
        all_results.push(mean);

        // 计算标准差和变异系数 (std/mean)
        let variance: f64 = values.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>() / (values.len() - 1) as f64;
        let std = variance.sqrt();
        let cv = if mean.abs() > 1e-10 { std / mean } else { f64::NAN };
        all_results.push(cv);

        // 计算最大值
        let max_val = *values.iter().fold(&values[0], |a, b| if a > b { a } else { b });
        all_results.push(max_val);

        // 计算与序列号 [1, 2, 3, ..., n] 的相关系数
        let sequence: Vec<f64> = (1..=values.len()).map(|i| i as f64).collect();
        let corr = pearson_correlation(&values, &sequence);
        all_results.push(corr);

        // 计算绝对相关系数
        let abs_corr = corr.abs();
        all_results.push(abs_corr);

        // 计算自相关性
        let autocorr = calculate_autocorrelation(&values);
        all_results.push(autocorr);
    }

    Some(all_results)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn compute_price_cycle_features_b_segments_enhanced(
    exchtime_trade: &PyArray1<i64>,
    price_trade: &PyArray1<f64>,
    volume_trade: &PyArray1<f64>,
    flag_trade: &PyArray1<i32>,
    ask_exchtime: &PyArray1<i64>,
    bid_exchtime: &PyArray1<i64>,
    bid_price: &PyArray1<f64>,
    bid_volume: &PyArray1<f64>,
    bid_number: &PyArray1<i32>,
    ask_price: &PyArray1<f64>,
    ask_volume: &PyArray1<f64>,
    ask_number: &PyArray1<i32>,
    tick: f64,
    drop_threshold: f64,
    rise_threshold: f64,
    use_trade_prices_as_grid: bool,
    price_grid_opt: Option<&PyArray1<f64>>,
    side: Option<String>, // Optional side parameter
    py: Python<'_>,
) -> PyResult<PyObject> {
    let trades_time = unsafe { exchtime_trade.as_slice()? };
    let trades_price = unsafe { price_trade.as_slice()? };
    let trades_volume = unsafe { volume_trade.as_slice()? };
    let trades_flag = unsafe { flag_trade.as_slice()? };

    let ask_time = unsafe { ask_exchtime.as_slice()? };
    let bid_time = unsafe { bid_exchtime.as_slice()? };
    let ask_price_slice = unsafe { ask_price.as_slice()? };
    let ask_volume_slice = unsafe { ask_volume.as_slice()? };
    let ask_number_slice = unsafe { ask_number.as_slice()? };
    let bid_price_slice = unsafe { bid_price.as_slice()? };
    let bid_volume_slice = unsafe { bid_volume.as_slice()? };
    let bid_number_slice = unsafe { bid_number.as_slice()? };

    if trades_time.len() != trades_price.len()
        || trades_time.len() != trades_volume.len()
        || trades_time.len() != trades_flag.len()
    {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "trade arrays length mismatch",
        ));
    }

    if ask_time.len() != ask_price_slice.len()
        || ask_time.len() != ask_volume_slice.len()
        || ask_time.len() != ask_number_slice.len()
        || bid_time.len() != bid_price_slice.len()
        || bid_time.len() != bid_volume_slice.len()
        || bid_time.len() != bid_number_slice.len()
    {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "snapshot arrays length mismatch",
        ));
    }

    let eps = tick.max(1e-6) / 10.0;

    // Parse side parameter
    let side_filter = match side.as_deref() {
        Some("buy") => Some(true),   // Only buy side
        Some("sell") => Some(false), // Only sell side
        Some("both") | None => None, // Both sides (default)
        _ => None,
    };

    let price_grid = if let Some(arr) = price_grid_opt {
        let slice = unsafe { arr.as_slice()? };
        slice.to_vec()
    } else if use_trade_prices_as_grid {
        build_price_grid(trades_price, tick)
    } else {
        // Use a reasonable range of prices around the trading range
        let min_price = trades_price.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_price = trades_price.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let mut prices = vec![];
        let mut current = (min_price / tick).floor() * tick;
        while current <= max_price {
            prices.push(current);
            current += tick;
        }
        prices
    };

    let snapshots = build_snapshots(
        ask_time,
        bid_time,
        ask_price_slice,
        ask_volume_slice,
        ask_number_slice,
        bid_price_slice,
        bid_volume_slice,
        bid_number_slice,
    );

    // Define feature names for 21 base metrics × 5 enhanced metrics = 105 dimensions
    let buy_feature_names = generate_enhanced_feature_names(true);
    let sell_feature_names = generate_enhanced_feature_names(false);

    // Collect results for each price level - ensure alignment
    let mut all_results: Vec<PriceLevelResult> = Vec::new();

    for (idx, &price) in price_grid.iter().enumerate() {
        let (buy_b_points, sell_b_points) = detect_b_points_by_side(
            price, trades_time, trades_price, drop_threshold, rise_threshold, eps
        );

        let buy_segments = build_segments_for_side(&buy_b_points, BType::BuyBreak);
        let sell_segments = build_segments_for_side(&sell_b_points, BType::SellBreak);

        let relevant_snapshots = &snapshots;

        // Compute metrics based on side filter
        let buy_metrics = if side_filter != Some(false) && !buy_segments.is_empty() {
            compute_segment_metrics(
                &buy_segments, trades_time, trades_price, trades_volume, trades_flag,
                relevant_snapshots, price, true, eps
            )
        } else {
            Vec::new()
        };

        let sell_metrics = if side_filter != Some(true) && !sell_segments.is_empty() {
            compute_segment_metrics(
                &sell_segments, trades_time, trades_price, trades_volume, trades_flag,
                relevant_snapshots, price, false, eps
            )
        } else {
            Vec::new()
        };

        // Aggregate metrics - 使用增强版聚合函数
        let buy_avg = if !buy_metrics.is_empty() {
            enhanced_aggregate_metrics(&buy_metrics)
        } else {
            None
        };

        let sell_avg = if !sell_metrics.is_empty() {
            enhanced_aggregate_metrics(&sell_metrics)
        } else {
            None
        };

        // Always include this price level - even if no segments, we'll use NaN features
        all_results.push(PriceLevelResult {
            price,
            buy_segments: buy_segments.clone(),
            sell_segments: sell_segments.clone(),
            buy_segment_count: buy_segments.len(),
            sell_segment_count: sell_segments.len(),
            buy_metrics: buy_avg,
            sell_metrics: sell_avg,
        });
    }

    // Build aligned result matrices - ensure all arrays have same length
    let num_prices = all_results.len();
    let mut prices_vec = Vec::with_capacity(num_prices);
    let mut buy_counts_vec = Vec::with_capacity(num_prices);
    let mut sell_counts_vec = Vec::with_capacity(num_prices);
    let mut buy_features_matrix: Vec<Vec<f64>> = Vec::with_capacity(num_prices);
    let mut sell_features_matrix: Vec<Vec<f64>> = Vec::with_capacity(num_prices);

    for (i, result) in all_results.iter().enumerate() {
        prices_vec.push(result.price);
        buy_counts_vec.push(result.buy_segment_count as f64);
        sell_counts_vec.push(result.sell_segment_count as f64);

        // Add buy features or NaN features if no data
        if let Some(metrics) = &result.buy_metrics {
            buy_features_matrix.push(metrics.clone());
        } else {
            // Add NaN features for this price level
            buy_features_matrix.push(vec![f64::NAN; buy_feature_names.len()]);
        }

        // Add sell features or NaN features if no data
        if let Some(metrics) = &result.sell_metrics {
            sell_features_matrix.push(metrics.clone());
        } else {
            // Add NaN features for this price level
            sell_features_matrix.push(vec![f64::NAN; sell_feature_names.len()]);
        }
    }

    // Convert to numpy arrays
    let prices_np = prices_vec.into_pyarray(py);
    let buy_counts_np = buy_counts_vec.into_pyarray(py);
    let sell_counts_np = sell_counts_vec.into_pyarray(py);

    // Build feature matrices - ensure they match prices length exactly
    let buy_features_array = if !buy_features_matrix.is_empty() {
        let rows = buy_features_matrix.len();
        let cols = buy_feature_names.len();
        let mut matrix = Array2::<f64>::from_elem((rows, cols), f64::NAN);

        for (i, row) in buy_features_matrix.iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                if j < cols {
                    matrix[[i, j]] = val;
                }
            }
        }
        matrix
    } else {
        Array2::<f64>::from_elem((num_prices, buy_feature_names.len()), f64::NAN)
    };

    let sell_features_array = if !sell_features_matrix.is_empty() {
        let rows = sell_features_matrix.len();
        let cols = sell_feature_names.len();
        let mut matrix = Array2::<f64>::from_elem((rows, cols), f64::NAN);

        for (i, row) in sell_features_matrix.iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                if j < cols {
                    matrix[[i, j]] = val;
                }
            }
        }
        matrix
    } else {
        Array2::<f64>::from_elem((num_prices, sell_feature_names.len()), f64::NAN)
    };

    let buy_features_np = PyArray2::from_array(py, &buy_features_array);
    let sell_features_np = PyArray2::from_array(py, &sell_features_array);

    let result = PyDict::new(py);
    result.set_item("prices", prices_np)?;
    result.set_item("buy_feature_names", buy_feature_names)?;
    result.set_item("sell_feature_names", sell_feature_names)?;
    result.set_item("buy_features", buy_features_np)?;
    result.set_item("sell_features", sell_features_np)?;
    result.set_item("buy_segment_counts", buy_counts_np)?;
    result.set_item("sell_segment_counts", sell_counts_np)?;

    Ok(result.into())
}

/// 生成增强特征名称列表
fn generate_enhanced_feature_names(is_buy_side: bool) -> Vec<String> {
    let base_metrics = vec![
        "duration_ms",
        "total_volume", 
        "trade_count",
        "vwap",
        "min_price",
        "max_price",
        "buy_ratio",
        "sell_ratio",
        "start_spread",
        "spread_change",
        "mid_return_bp",
        "start_distance_to_level",
        "end_distance_to_level",
        if is_buy_side { "avg_bid_depth" } else { "avg_ask_depth" },
        if is_buy_side { "max_bid_depth" } else { "max_ask_depth" },
        if is_buy_side { "min_bid_depth" } else { "min_ask_depth" },
        "vol_at_level",
        "count_at_level",
        "time_between_ms",
        "trades_between",
        "start_price",
        "end_price",
        "total_return_bp",
    ];

    let mut enhanced_names = Vec::new();
    
    // 为每个基础指标生成5个增强指标
    for base_metric in base_metrics {
        enhanced_names.push(format!("{}_mean", base_metric));
        enhanced_names.push(format!("{}_cv", base_metric));
        enhanced_names.push(format!("{}_max", base_metric));
        enhanced_names.push(format!("{}_corr_seq", base_metric));
        enhanced_names.push(format!("{}_abs_corr_seq", base_metric));
        enhanced_names.push(format!("{}_autocorr", base_metric));
    }

    enhanced_names
}