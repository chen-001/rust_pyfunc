//! Agent交易模拟器 - 主买占比型Agent

use super::types::*;
use super::trait_def::*;
use super::utils::*;

/// 主买占比型Agent：根据回看窗口内的主买占比触发交易
/// 
/// 当主买占比超过阈值时买入，否则卖出（或不做空）
pub struct BuyRatioAgent {
    base: AgentBaseConfig,
    threshold: f64,  // 主买占比阈值，如 0.67 表示大于2/3
    last_trade_time: i64,
    trades: Vec<AgentTrade>,
    last_ratio: f64, // 记录上次的主买占比，用于判断是否变化
}

impl BuyRatioAgent {
    pub fn new(
        name: &str,
        lookback_ms: i64,
        threshold: f64,
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
            last_trade_time: -1,
            trades: Vec::new(),
            last_ratio: 0.5, // 默认50%
        }
    }
    
    /// 创建默认配置的新实例
    pub fn with_defaults(name: &str, lookback_ms: i64, threshold: f64) -> Self {
        Self::new(
            name,
            lookback_ms,
            threshold,
            100,    // 默认100股
            0,      // 默认无冷却
            true,   // 默认允许做空
        )
    }
}

impl TradingAgent for BuyRatioAgent {
    fn name(&self) -> &str {
        &self.base.name
    }
    
    fn check_trigger(
        &self,
        market_trades: &[MarketTrade],
        current_idx: usize,
        _current_time: i64,
    ) -> Option<i32> {
        let ratio = calc_buy_ratio(
            market_trades,
            current_idx,
            self.base.lookback_ms,
        )?;
        
        // 主买占比超过阈值 -> 买入（跟随主力资金）
        if ratio > self.threshold {
            Some(66)  // 买入
        } else if self.base.allow_short && ratio < (1.0 - self.threshold) {
            // 主卖占比超过阈值 -> 卖出
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
        
        // 更新当前主买占比（用于记录，可选）
        // 注意：这里简化处理，实际可能需要重新计算
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
