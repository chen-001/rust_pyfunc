//! Agent交易模拟器 - 跟随型Agent

use super::trait_def::*;
use super::types::*;
use super::utils::*;

/// 跟随型Agent：
/// - 买入：过去 short_window_ns 主动买入成交额超过阈值，且 trend_window_ns 内价格上涨
/// - 卖出：过去 short_window_ns 主动卖出成交额超过阈值，且 trend_window_ns 内价格下跌
pub struct FollowFlowAgent {
    base: AgentBaseConfig,
    short_window_ns: i64,
    trend_window_ns: i64,
    amount_threshold: f64,
    last_trade_time: i64,
    trades: Vec<AgentTrade>,
}

impl FollowFlowAgent {
    pub fn new(
        name: &str,
        short_window_ns: i64,
        trend_window_ns: i64,
        amount_threshold: f64,
        cooldown_ns: i64,
    ) -> Self {
        Self {
            base: AgentBaseConfig {
                name: name.to_string(),
                lookback_ms: trend_window_ns,
                fixed_trade_size: 100,
                cooldown_ms: cooldown_ns,
                allow_short: true,
            },
            short_window_ns,
            trend_window_ns,
            amount_threshold,
            last_trade_time: -1,
            trades: Vec::new(),
        }
    }

    pub fn with_defaults(name: &str) -> Self {
        Self::new(name, 2_000_000_000, 15_000_000_000, 200_000.0, 0)
    }
}

impl TradingAgent for FollowFlowAgent {
    fn name(&self) -> &str {
        &self.base.name
    }

    fn check_trigger(
        &self,
        market_trades: &[MarketTrade],
        current_idx: usize,
        _current_time: i64,
    ) -> Option<i32> {
        let trend_change = calc_price_change(market_trades, current_idx, self.trend_window_ns, false)?;
        let (buy_turnover, sell_turnover) =
            calc_directional_turnover(market_trades, current_idx, self.short_window_ns);

        if sell_turnover > self.amount_threshold && trend_change < 0.0 {
            Some(83)
        } else if buy_turnover > self.amount_threshold && trend_change > 0.0 {
            Some(66)
        } else {
            None
        }
    }

    fn trade_size(&self) -> i64 {
        self.base.fixed_trade_size
    }

    fn can_trade(&self, current_time: i64) -> bool {
        self.default_can_trade(current_time, self.base.cooldown_ms)
    }

    fn record_trade(&mut self, time: i64, market_idx: usize, direction: i32, price: f64) {
        self.default_record_trade(
            time,
            market_idx,
            direction,
            self.base.fixed_trade_size as f64,
            price,
        );
    }

    fn trades(&self) -> &[AgentTrade] {
        &self.trades
    }

    fn trades_mut(&mut self) -> &mut Vec<AgentTrade> {
        &mut self.trades
    }

    fn last_trade_time(&self) -> i64 {
        self.last_trade_time
    }

    fn set_last_trade_time(&mut self, time: i64) {
        self.last_trade_time = time;
    }
}
