//! Agent交易模拟器 - TradingAgent Trait定义

use super::types::*;

/// 所有Agent必须实现的Trait
pub trait TradingAgent {
    /// Agent名称
    fn name(&self) -> &str;
    
    /// 判断是否触发交易
    /// 
    /// 参数：
    /// - market_trades: 完整的市场成交数据
    /// - current_idx: 当前处理的市场成交索引
    /// - current_time: 当前时间戳
    /// 
    /// 返回：
    /// - Some(66) = 买入触发
    /// - Some(83) = 卖出触发  
    /// - None = 不触发
    fn check_trigger(
        &self,
        market_trades: &[MarketTrade],
        current_idx: usize,
        current_time: i64,
    ) -> Option<i32>;
    
    /// 获取固定交易数量
    fn trade_size(&self) -> i64;
    
    /// 检查冷却期是否已过
    fn can_trade(&self, current_time: i64) -> bool;
    
    /// 记录交易（更新最后交易时间等）
    fn record_trade(&mut self, time: i64, market_idx: usize, direction: i32, price: f64);
    
    /// 获取交易记录
    fn trades(&self) -> &[AgentTrade];
    
    /// 获取交易记录的可变引用（用于模拟器）
    fn trades_mut(&mut self) -> &mut Vec<AgentTrade>;
    
    /// 获取最后交易时间
    fn last_trade_time(&self) -> i64;
    
    /// 设置最后交易时间
    fn set_last_trade_time(&mut self, time: i64);
}

/// TradingAgent的通用实现（减少重复代码）
pub trait TradingAgentBase: TradingAgent {
    /// 默认的冷却期检查实现
    fn default_can_trade(&self, current_time: i64, cooldown_ms: i64) -> bool {
        if cooldown_ms <= 0 {
            return true;
        }
        current_time - self.last_trade_time() >= cooldown_ms
    }
    
    /// 默认的交易记录实现
    fn default_record_trade(
        &mut self,
        time: i64,
        market_idx: usize,
        direction: i32,
        volume: f64,
        price: f64,
    ) {
        self.trades_mut().push(AgentTrade {
            timestamp: time,
            market_trade_idx: market_idx,
            direction,
            volume,
            price,
        });
        self.set_last_trade_time(time);
    }
}

impl<T: TradingAgent + ?Sized> TradingAgentBase for T {}
