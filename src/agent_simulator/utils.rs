//! Agent交易模拟器 - 辅助函数

use super::types::*;

fn lower_bound_market(trades: &[MarketTrade], target_time: i64) -> usize {
    let mut left = 0usize;
    let mut right = trades.len();
    while left < right {
        let mid = left + (right - left) / 2;
        if trades[mid].timestamp < target_time {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    left
}

fn upper_bound_market(trades: &[MarketTrade], target_time: i64) -> usize {
    let mut left = 0usize;
    let mut right = trades.len();
    while left < right {
        let mid = left + (right - left) / 2;
        if trades[mid].timestamp <= target_time {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    left
}

/// 二分查找给定时间对应的价格
///
/// 找到小于等于target_time的最后一条成交价格
pub fn find_price_at_time(market_trades: &[MarketTrade], target_time: i64) -> Option<f64> {
    if market_trades.is_empty() {
        return None;
    }

    let ub = upper_bound_market(market_trades, target_time);
    if ub == 0 {
        Some(market_trades[0].price)
    } else {
        Some(market_trades[ub - 1].price)
    }
}

/// 计算时间窗口内的价格变化
pub fn calc_price_change(
    market_trades: &[MarketTrade],
    current_idx: usize,
    lookback_ns: i64,
    is_percentage: bool,
) -> Option<f64> {
    if market_trades.is_empty() || current_idx >= market_trades.len() {
        return None;
    }

    let current_time = market_trades[current_idx].timestamp;
    let window_start = current_time.saturating_sub(lookback_ns);

    let start_price = find_price_at_time(market_trades, window_start)?;
    let current_price = market_trades[current_idx].price;

    if is_percentage {
        if start_price.abs() < 1e-10 {
            return None;
        }
        Some((current_price - start_price) / start_price)
    } else {
        Some(current_price - start_price)
    }
}

/// 计算时间窗口内的主买占比
///
/// 仅统计到 current_idx 为止，避免把同时间戳但更后面的成交纳入（前视偏差）
pub fn calc_buy_ratio(
    market_trades: &[MarketTrade],
    current_idx: usize,
    lookback_ns: i64,
) -> Option<f64> {
    if market_trades.is_empty() || current_idx >= market_trades.len() {
        return None;
    }

    let current_time = market_trades[current_idx].timestamp;
    let window_start = current_time.saturating_sub(lookback_ns);

    let start_idx = lower_bound_market(market_trades, window_start);
    let end_idx = current_idx + 1;

    if start_idx >= end_idx {
        return Some(0.0);
    }

    let mut buy_volume = 0.0;
    let mut total_volume = 0.0;

    for trade in &market_trades[start_idx..end_idx] {
        total_volume += trade.volume;
        if trade.is_buy() {
            buy_volume += trade.volume;
        }
    }

    if total_volume < 1e-10 {
        return Some(0.0);
    }

    Some(buy_volume / total_volume)
}

/// 计算时间窗口内的成交量统计
pub fn calc_volume_stats(
    market_trades: &[MarketTrade],
    current_idx: usize,
    lookback_ns: i64,
) -> (f64, f64, f64) {
    if market_trades.is_empty() || current_idx >= market_trades.len() {
        return (0.0, 0.0, 0.0);
    }

    let current_time = market_trades[current_idx].timestamp;
    let window_start = current_time.saturating_sub(lookback_ns);

    let start_idx = lower_bound_market(market_trades, window_start);
    let end_idx = current_idx + 1;

    let mut buy_volume = 0.0;
    let mut sell_volume = 0.0;
    let mut total_count = 0usize;

    for trade in &market_trades[start_idx..end_idx] {
        total_count += 1;
        if trade.is_buy() {
            buy_volume += trade.volume;
        } else {
            sell_volume += trade.volume;
        }
    }

    (buy_volume, sell_volume, total_count as f64)
}
