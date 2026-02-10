//! Agent交易特征计算模块 - 各类指标计算函数

use super::types::*;
use std::f64;

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

fn lower_bound_agent(trades: &[AgentTrade], target_time: i64) -> usize {
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

fn upper_bound_agent(trades: &[AgentTrade], target_time: i64) -> usize {
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

/// 计算时间窗口统计（基于索引，规避同时间戳前视）
///
/// - 前窗口: [center_time - window_ns, center_idx)
/// - 后窗口: (center_idx, center_time + window_ns]
pub fn compute_window_stats_by_index(
    market_trades: &[MarketTrade],
    center_idx: usize,
    window_ns: i64,
    is_pre: bool,
) -> WindowStats {
    let mut stats = WindowStats::default();

    if market_trades.is_empty() || center_idx >= market_trades.len() {
        return stats;
    }

    let center_time = market_trades[center_idx].timestamp;
    let (start_idx, end_idx) = if is_pre {
        if center_idx == 0 {
            return stats;
        }
        let start_time = center_time.saturating_sub(window_ns);
        (lower_bound_market(market_trades, start_time), center_idx)
    } else {
        let start = center_idx.saturating_add(1);
        if start >= market_trades.len() {
            return stats;
        }
        let end_time = center_time.saturating_add(window_ns);
        (start, upper_bound_market(market_trades, end_time))
    };

    if start_idx >= end_idx {
        return stats;
    }

    for trade in &market_trades[start_idx..end_idx] {
        stats.price_series.push(trade.price);
    }

    stats.trade_count = stats.price_series.len();

    if stats.trade_count >= 2 {
        for i in 1..stats.price_series.len() {
            let prev = stats.price_series[i - 1];
            let curr = stats.price_series[i];
            if prev > 0.0 {
                stats.returns.push((curr / prev).ln());
            }
        }

        stats.volatility = compute_std(&stats.returns);

        let first = stats.price_series.first().copied().unwrap_or(0.0);
        let last = stats.price_series.last().copied().unwrap_or(0.0);
        if first.abs() > 1e-12 {
            stats.trend = (last - first) / first;
        }

        stats.mean_price = stats.price_series.iter().sum::<f64>() / stats.price_series.len() as f64;
    }

    stats
}

/// 计算标准差
fn compute_std(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;
    variance.sqrt()
}

/// 计算夏普比率
///
/// 简化公式: (平均收益率 - 无风险利率) / 收益率标准差
/// 对于日内交易，假设无风险利率为0
pub fn compute_sharpe_ratio(returns: &[f64]) -> f64 {
    if returns.len() < 2 {
        return 0.0;
    }

    let mean_ret = returns.iter().sum::<f64>() / returns.len() as f64;
    let std_ret = compute_std(returns);

    if std_ret < 1e-10 {
        return 0.0;
    }

    mean_ret / std_ret
}

/// 计算Sortino比率（只惩罚下行波动）
pub fn compute_sortino_ratio(returns: &[f64]) -> f64 {
    if returns.len() < 2 {
        return 0.0;
    }

    let mean_ret = returns.iter().sum::<f64>() / returns.len() as f64;

    let downside_returns: Vec<f64> = returns.iter().filter(|&&r| r < 0.0).copied().collect();
    if downside_returns.is_empty() {
        return if mean_ret > 0.0 { f64::INFINITY } else { 0.0 };
    }

    let downside_std = compute_std(&downside_returns);
    if downside_std < 1e-10 {
        return if mean_ret > 0.0 { f64::INFINITY } else { 0.0 };
    }

    mean_ret / downside_std
}

/// 计算最大回撤及持续时间
///
/// 返回: (最大回撤值, 最大回撤持续时间)
pub fn compute_max_drawdown(pnl_series: &[(i64, f64)]) -> (f64, i64) {
    if pnl_series.len() < 2 {
        return (0.0, 0);
    }

    let mut max_pnl = pnl_series[0].1;
    let mut max_dd = 0.0;
    let mut max_dd_duration = 0_i64;
    let mut dd_start_time = pnl_series[0].0;

    for (time, pnl) in pnl_series.iter().copied().skip(1) {
        if pnl > max_pnl {
            max_pnl = pnl;
            dd_start_time = time;
        } else {
            let dd = max_pnl - pnl;
            let dd_duration = time - dd_start_time;

            if dd > max_dd {
                max_dd = dd;
                max_dd_duration = dd_duration;
            } else if (dd - max_dd).abs() < 1e-10 && dd_duration > max_dd_duration {
                max_dd_duration = dd_duration;
            }
        }
    }

    (max_dd, max_dd_duration)
}

/// 计算盈利集中度（Top 10%交易贡献的利润占比）
pub fn compute_profit_concentration(trades_pnl: &[f64]) -> f64 {
    if trades_pnl.is_empty() {
        return 0.0;
    }

    let total_profit: f64 = trades_pnl.iter().filter(|&&p| p > 0.0).sum();
    if total_profit < 1e-10 {
        return 0.0;
    }

    let mut profits: Vec<f64> = trades_pnl.iter().filter(|&&p| p > 0.0).copied().collect();
    if profits.is_empty() {
        return 0.0;
    }

    profits.sort_by(|a, b| b.partial_cmp(a).unwrap());

    let top_count = (profits.len() as f64 * 0.1).ceil() as usize;
    let top_count = top_count.max(1);

    let top_profit: f64 = profits.iter().take(top_count).sum();

    top_profit / total_profit
}

/// 计算盈亏不对称性（平均盈利 / 平均亏损）
pub fn compute_gain_loss_ratio(winning_trades: &[f64], losing_trades: &[f64]) -> f64 {
    if winning_trades.is_empty() || losing_trades.is_empty() {
        return 1.0;
    }

    let avg_gain = winning_trades.iter().sum::<f64>() / winning_trades.len() as f64;
    let avg_loss = losing_trades.iter().sum::<f64>().abs() / losing_trades.len() as f64;

    if avg_loss < 1e-10 {
        return if avg_gain > 0.0 { f64::INFINITY } else { 1.0 };
    }

    avg_gain / avg_loss
}

/// 计算平均持仓时长（FIFO原则）
///
/// 返回毫秒数
pub fn compute_avg_holding_duration(holding_records: &[PositionRecord], current_time: i64) -> f64 {
    if holding_records.is_empty() {
        return 0.0;
    }

    let total_duration: i64 = holding_records
        .iter()
        .map(|r| current_time - r.entry_time)
        .sum();

    total_duration as f64 / holding_records.len() as f64
}

/// 计算与同向其他Agent的同步率
///
/// 同步率 = 在窗口内出现同向交易的其他Agent数 / 曾经出现过该方向交易的其他Agent总数
pub fn compute_sync_rate(
    agent_trade: &AgentTrade,
    all_agent_trades: &[&[AgentTrade]],
    window_ns: i64,
) -> f64 {
    if all_agent_trades.is_empty() {
        return 0.0;
    }

    let center_time = agent_trade.timestamp;
    let half = window_ns / 2;
    let start_time = center_time.saturating_sub(half);
    let end_time = center_time.saturating_add(half);

    let mut same_direction_agents = 0usize;
    let mut synced_agents = 0usize;

    for other_trades in all_agent_trades {
        if other_trades.is_empty() {
            continue;
        }

        if !other_trades
            .iter()
            .any(|t| t.direction == agent_trade.direction)
        {
            continue;
        }

        same_direction_agents += 1;

        let lb = lower_bound_agent(other_trades, start_time);
        let ub = upper_bound_agent(other_trades, end_time);

        if lb < ub
            && other_trades[lb..ub]
                .iter()
                .any(|t| t.direction == agent_trade.direction)
        {
            synced_agents += 1;
        }
    }

    if same_direction_agents == 0 {
        return 0.0;
    }

    synced_agents as f64 / same_direction_agents as f64
}

/// 计算日内均价
pub fn compute_daily_mean_price(market_trades: &[MarketTrade]) -> f64 {
    if market_trades.is_empty() {
        return 0.0;
    }

    let total_turnover: f64 = market_trades.iter().map(|t| t.turnover).sum();
    let total_volume: f64 = market_trades.iter().map(|t| t.volume).sum();

    if total_volume < 1e-10 {
        return market_trades.iter().map(|t| t.price).sum::<f64>() / market_trades.len() as f64;
    }

    total_turnover / total_volume
}
