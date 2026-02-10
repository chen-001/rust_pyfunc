//! Agent交易模拟器 - 动量型Agent

use super::types::*;
use super::trait_def::*;
use super::utils::*;

/// 动量型Agent：根据价格变化触发交易
pub struct MomentumAgent {
    base: AgentBaseConfig,
    threshold: f64,
    is_percentage: bool,
    last_trade_time: i64,
    trades: Vec<AgentTrade>,
}

impl MomentumAgent {
    pub fn new(
        name: &str,
        lookback_ms: i64,
        threshold: f64,
        is_percentage: bool,
        fixed_trade_size: i64,
        cooldown_ms: i64,
        allow_short: bool,
    ) -> Self {
        Self {
            base: AgentBaseConfig {
                name: name.to_string(),
                lookback_ms,
                fixed_trade_size,
                cooldown_ms,
                allow_short,
            },
            threshold,
            is_percentage,
            last_trade_time: -1,
            trades: Vec::new(),
        }
    }
    
    /// 创建默认配置的新实例
    pub fn with_defaults(name: &str, lookback_ms: i64, threshold: f64) -> Self {
        Self::new(
            name,
            lookback_ms,
            threshold,
            false,  // 默认使用绝对价格
            100,    // 默认100股
            0,      // 默认无冷却
            true,   // 默认允许做空
        )
    }
}

impl TradingAgent for MomentumAgent {
    fn name(&self) -> &str {
        &self.base.name
    }
    
    fn check_trigger(
        &self,
        market_trades: &[MarketTrade],
        current_idx: usize,
        _current_time: i64,
    ) -> Option<i32> {
        let change = calc_price_change(
            market_trades,
            current_idx,
            self.base.lookback_ms,
            self.is_percentage,
        )?;
        
        if change > self.threshold {
            Some(66)  // 买入
        } else if self.base.allow_short && change < -self.threshold {
            Some(83)  // 卖出
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
