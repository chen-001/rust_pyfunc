//! Agent交易模拟器 - 加速跟随型Agent

use super::trait_def::*;
use super::types::*;
use super::utils::*;

/// 加速跟随型Agent：
/// - 买入：当前2秒主动买入成交额 > 阈值，且相对前2秒放大到 acceleration_factor 倍以上，同时15秒趋势上涨
/// - 卖出：当前2秒主动卖出成交额 > 阈值，且相对前2秒放大到 acceleration_factor 倍以上，同时15秒趋势下跌
pub struct AccelerationFollowAgent {
    base: AgentBaseConfig,
    short_window_ns: i64,
    trend_window_ns: i64,
    amount_threshold: f64,
    acceleration_factor: f64,
    last_trade_time: i64,
    trades: Vec<AgentTrade>,
}

impl AccelerationFollowAgent {
    pub fn new(
        name: &str,
        short_window_ns: i64,
        trend_window_ns: i64,
        amount_threshold: f64,
        acceleration_factor: f64,
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
            acceleration_factor,
            last_trade_time: -1,
            trades: Vec::new(),
        }
    }

    pub fn with_defaults(name: &str) -> Self {
        Self::new(name, 2_000_000_000, 15_000_000_000, 200_000.0, 1.5, 0)
    }
}

impl TradingAgent for AccelerationFollowAgent {
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
        let (buy_curr, sell_curr, buy_prev, sell_prev) =
            calc_directional_turnover_two_windows(market_trades, current_idx, self.short_window_ns);

        const EPS: f64 = 1e-9;
        if sell_curr > self.amount_threshold
            && sell_prev > EPS
            && sell_curr >= self.acceleration_factor * sell_prev
            && trend_change < 0.0
        {
            Some(83)
        } else if buy_curr > self.amount_threshold
            && buy_prev > EPS
            && buy_curr >= self.acceleration_factor * buy_prev
            && trend_change > 0.0
        {
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
