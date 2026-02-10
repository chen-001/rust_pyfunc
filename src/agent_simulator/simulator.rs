//! Agent交易模拟器 - 核心模拟逻辑

use super::types::*;
use super::trait_def::*;

/// 运行Agent模拟（支持任何实现TradingAgent的类型）
/// 
/// # 参数
/// - market_trades: 市场逐笔成交数据
/// - agents: 可变引用的Agent数组
/// 
/// # 返回
/// 每个Agent的模拟结果
pub fn run_simulation<T: TradingAgent>(
    market_trades: &[MarketTrade],
    agents: &mut [T],
) -> Vec<AgentSimulationResult> {
    let mut results: Vec<AgentSimulationResult> = agents
        .iter()
        .map(|a| AgentSimulationResult::new(a.name()))
        .collect();
    
    // 遍历市场数据
    for (idx, trade) in market_trades.iter().enumerate() {
        let current_time = trade.timestamp;
        
        // 检查每个Agent
        for (agent_idx, agent) in agents.iter_mut().enumerate() {
            // 检查冷却期
            if !agent.can_trade(current_time) {
                continue;
            }
            
            // 检查触发条件
            if let Some(direction) = agent.check_trigger(market_trades, idx, current_time) {
                // 记录交易
                agent.record_trade(current_time, idx, direction, trade.price);
                
                // 同步到结果
                let agent_trade = AgentTrade {
                    timestamp: current_time,
                    market_trade_idx: idx,
                    direction,
                    volume: agent.trade_size() as f64,
                    price: trade.price,
                };
                results[agent_idx].add_trade(agent_trade);
            }
        }
    }
    
    results
}

/// 批量运行多个Agent模拟（不同类型Agent用Box<dyn>）
/// 
/// 适用于混合不同类型Agent的场景
pub fn run_simulation_mixed(
    market_trades: &[MarketTrade],
    agents: &mut [Box<dyn TradingAgent>],
) -> Vec<AgentSimulationResult> {
    let mut results: Vec<AgentSimulationResult> = agents
        .iter()
        .map(|a| AgentSimulationResult::new(a.name()))
        .collect();
    
    for (idx, trade) in market_trades.iter().enumerate() {
        let current_time = trade.timestamp;
        
        for (agent_idx, agent) in agents.iter_mut().enumerate() {
            if !agent.can_trade(current_time) {
                continue;
            }
            
            if let Some(direction) = agent.check_trigger(market_trades, idx, current_time) {
                agent.record_trade(current_time, idx, direction, trade.price);
                
                let agent_trade = AgentTrade {
                    timestamp: current_time,
                    market_trade_idx: idx,
                    direction,
                    volume: agent.trade_size() as f64,
                    price: trade.price,
                };
                results[agent_idx].add_trade(agent_trade);
            }
        }
    }
    
    results
}

/// 便捷的批量模拟函数（专用于MomentumAgent）
pub fn simulate_momentum_agents(
    market_trades: &[MarketTrade],
    configs: &[(String, i64, f64, bool, i64, i64, bool)],
) -> Vec<AgentSimulationResult> {
    let mut agents: Vec<_> = configs
        .iter()
        .map(|(name, lookback, threshold, is_pct, size, cooldown, allow_short)| {
            crate::agent_simulator::momentum_agent::MomentumAgent::new(
                name,
                *lookback,
                *threshold,
                *is_pct,
                *size,
                *cooldown,
                *allow_short,
            )
        })
        .collect();
    
    run_simulation(market_trades, &mut agents)
}

/// 便捷的批量模拟函数（专用于BuyRatioAgent）
pub fn simulate_buy_ratio_agents(
    market_trades: &[MarketTrade],
    configs: &[(String, i64, f64, i64, i64, bool)],
) -> Vec<AgentSimulationResult> {
    let mut agents: Vec<_> = configs
        .iter()
        .map(|(name, lookback, threshold, size, cooldown, allow_short)| {
            crate::agent_simulator::buy_ratio_agent::BuyRatioAgent::new(
                name,
                *lookback,
                *threshold,
                *size,
                *cooldown,
                *allow_short,
            )
        })
        .collect();
    
    run_simulation(market_trades, &mut agents)
}
