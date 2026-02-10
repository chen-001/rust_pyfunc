//! Agent交易特征计算模块 - 核心计算逻辑

use super::metrics::*;
use super::types::*;

/// FIFO持仓管理器（同时支持多头与空头）
pub struct PositionManager {
    long_positions: Vec<PositionRecord>,
    short_positions: Vec<PositionRecord>,
    realized_pnl: f64,
}

impl PositionManager {
    pub fn new() -> Self {
        Self {
            long_positions: Vec::new(),
            short_positions: Vec::new(),
            realized_pnl: 0.0,
        }
    }

    /// 买入：先平空头，再开多头。返回本次买入产生的已实现盈亏。
    pub fn buy(&mut self, time: i64, price: f64, volume: f64) -> f64 {
        let mut remaining_volume = volume.max(0.0);
        let mut pnl = 0.0;

        while remaining_volume > 1e-10 && !self.short_positions.is_empty() {
            let mut pos = self.short_positions.remove(0);
            if pos.volume <= remaining_volume {
                pnl += (pos.entry_price - price) * pos.volume;
                remaining_volume -= pos.volume;
            } else {
                pnl += (pos.entry_price - price) * remaining_volume;
                pos.volume -= remaining_volume;
                remaining_volume = 0.0;
                self.short_positions.insert(0, pos);
            }
        }

        if remaining_volume > 1e-10 {
            self.long_positions.push(PositionRecord {
                entry_time: time,
                entry_price: price,
                volume: remaining_volume,
            });
        }

        self.realized_pnl += pnl;
        pnl
    }

    /// 卖出：先平多头，再开空头。返回本次卖出产生的已实现盈亏。
    pub fn sell(&mut self, time: i64, price: f64, volume: f64) -> f64 {
        let mut remaining_volume = volume.max(0.0);
        let mut pnl = 0.0;

        while remaining_volume > 1e-10 && !self.long_positions.is_empty() {
            let mut pos = self.long_positions.remove(0);
            if pos.volume <= remaining_volume {
                pnl += (price - pos.entry_price) * pos.volume;
                remaining_volume -= pos.volume;
            } else {
                pnl += (price - pos.entry_price) * remaining_volume;
                pos.volume -= remaining_volume;
                remaining_volume = 0.0;
                self.long_positions.insert(0, pos);
            }
        }

        if remaining_volume > 1e-10 {
            self.short_positions.push(PositionRecord {
                entry_time: time,
                entry_price: price,
                volume: remaining_volume,
            });
        }

        self.realized_pnl += pnl;
        pnl
    }

    /// 获取当前净持仓量（多头为正，空头为负）
    pub fn current_position(&self) -> f64 {
        let long_volume: f64 = self.long_positions.iter().map(|p| p.volume).sum();
        let short_volume: f64 = self.short_positions.iter().map(|p| p.volume).sum();
        long_volume - short_volume
    }

    /// 获取当前价格下的未实现盈亏（多空都支持）
    pub fn unrealized_pnl(&self, current_price: f64) -> f64 {
        let long_unrealized: f64 = self
            .long_positions
            .iter()
            .map(|p| (current_price - p.entry_price) * p.volume)
            .sum();

        let short_unrealized: f64 = self
            .short_positions
            .iter()
            .map(|p| (p.entry_price - current_price) * p.volume)
            .sum();

        long_unrealized + short_unrealized
    }

    /// 获取平均持仓时长（多空合并，按仓位量加权）
    pub fn avg_holding_duration(&self, current_time: i64) -> f64 {
        let mut weighted_duration_sum = 0.0;
        let mut total_volume = 0.0;

        for p in &self.long_positions {
            weighted_duration_sum += (current_time - p.entry_time).max(0) as f64 * p.volume;
            total_volume += p.volume;
        }

        for p in &self.short_positions {
            weighted_duration_sum += (current_time - p.entry_time).max(0) as f64 * p.volume;
            total_volume += p.volume;
        }

        if total_volume < 1e-10 {
            0.0
        } else {
            weighted_duration_sum / total_volume
        }
    }
}

/// 特征计算器
pub struct FeatureCalculator<'a> {
    market_trades: &'a [MarketTrade],
    daily_mean_price: f64,
    config: FeatureConfig,
}

impl<'a> FeatureCalculator<'a> {
    pub fn new(market_trades: &'a [MarketTrade], config: FeatureConfig) -> Self {
        let daily_mean_price = compute_daily_mean_price(market_trades);
        Self {
            market_trades,
            daily_mean_price,
            config,
        }
    }

    /// 计算单个Agent的所有特征
    pub fn compute_features(
        &self,
        agent_trades: &[AgentTrade],
        all_agent_trades: Option<&[&[AgentTrade]]>,
    ) -> Vec<AgentFeatures> {
        let mut features = Vec::with_capacity(agent_trades.len());
        if agent_trades.is_empty() {
            return features;
        }

        let mut pos_manager = PositionManager::new();
        let mut pnl_state = PnLState::default();

        let mut returns_history: Vec<f64> = Vec::new();
        let mut cumulative_volume = 0.0;
        let mut cumulative_turnover = 0.0;
        let mut prev_cumulative_return = 0.0;

        let mut first_trade_time: Option<i64> = None;
        let mut last_trade_time: Option<i64> = None;
        let mut prev_position: f64 = 0.0;
        let mut exposure_time_ns: i64 = 0;

        for (idx, agent_trade) in agent_trades.iter().enumerate() {
            if agent_trade.market_trade_idx >= self.market_trades.len() {
                continue;
            }

            let mut feature = AgentFeatures::default();
            feature.timestamp = agent_trade.timestamp;

            let market_trade = &self.market_trades[agent_trade.market_trade_idx];

            feature.order_id_diff = (market_trade.bid_order_id - market_trade.ask_order_id).abs();

            let pre_stats = compute_window_stats_by_index(
                self.market_trades,
                agent_trade.market_trade_idx,
                self.config.window_ns,
                true,
            );
            feature.pre_trade_count = pre_stats.trade_count;
            feature.pre_trade_volatility = pre_stats.volatility;

            let post_stats = compute_window_stats_by_index(
                self.market_trades,
                agent_trade.market_trade_idx,
                self.config.window_ns,
                false,
            );
            feature.post_trade_count = post_stats.trade_count;
            feature.post_trade_volatility = post_stats.volatility;
            feature.post_trade_price_trend = post_stats.trend;

            if self.daily_mean_price.abs() > 1e-12 {
                feature.price_deviation = (agent_trade.price - self.daily_mean_price) / self.daily_mean_price;
            }

            if first_trade_time.is_none() {
                first_trade_time = Some(agent_trade.timestamp);
            }
            if let Some(prev_t) = last_trade_time {
                let dt = agent_trade.timestamp - prev_t;
                if dt > 0 && prev_position.abs() > 1e-10 {
                    exposure_time_ns = exposure_time_ns.saturating_add(dt);
                }
            }

            let trade_pnl = match agent_trade.direction {
                66 => pos_manager.buy(agent_trade.timestamp, agent_trade.price, agent_trade.volume),
                83 => pos_manager.sell(agent_trade.timestamp, agent_trade.price, agent_trade.volume),
                _ => 0.0,
            };

            pnl_state.realized_pnl += trade_pnl;

            if trade_pnl.abs() > 1e-10 {
                pnl_state.trades_pnl.push(trade_pnl);
                if trade_pnl > 0.0 {
                    pnl_state.winning_trades.push(trade_pnl);
                } else {
                    pnl_state.losing_trades.push(trade_pnl);
                }
            }

            pnl_state.unrealized_pnl = pos_manager.unrealized_pnl(agent_trade.price);
            pnl_state.cumulative_pnl = pnl_state.realized_pnl + pnl_state.unrealized_pnl;
            pnl_state.current_position = pos_manager.current_position();

            if idx == 0 {
                pnl_state.max_cumulative_pnl = pnl_state.cumulative_pnl;
                pnl_state.drawdown_start_time = Some(agent_trade.timestamp);
            } else if pnl_state.cumulative_pnl >= pnl_state.max_cumulative_pnl {
                pnl_state.max_cumulative_pnl = pnl_state.cumulative_pnl;
                pnl_state.drawdown_start_time = Some(agent_trade.timestamp);
            } else {
                let dd = pnl_state.max_cumulative_pnl - pnl_state.cumulative_pnl;
                if dd > pnl_state.max_drawdown {
                    pnl_state.max_drawdown = dd;
                    if let Some(start_time) = pnl_state.drawdown_start_time {
                        pnl_state.max_drawdown_duration = agent_trade.timestamp - start_time;
                    }
                }
            }

            feature.max_drawdown = pnl_state.max_drawdown;
            feature.max_drawdown_duration = pnl_state.max_drawdown_duration;

            feature.profit_concentration = compute_profit_concentration(&pnl_state.trades_pnl);
            feature.gain_loss_ratio =
                compute_gain_loss_ratio(&pnl_state.winning_trades, &pnl_state.losing_trades);

            feature.avg_holding_duration = pos_manager.avg_holding_duration(agent_trade.timestamp);

            if let Some(first_time) = first_trade_time {
                let total_elapsed = agent_trade.timestamp - first_time;
                if total_elapsed > 0 {
                    feature.exposure_utilization = exposure_time_ns as f64 / total_elapsed as f64;

                    let total_minutes = total_elapsed as f64 / 60_000_000_000.0;
                    if total_minutes > 0.0 {
                        feature.trading_density = (idx + 1) as f64 / total_minutes;
                    }
                } else {
                    feature.exposure_utilization = if pnl_state.current_position.abs() > 1e-10 {
                        1.0
                    } else {
                        0.0
                    };
                }
            }

            cumulative_volume += agent_trade.volume;
            cumulative_turnover += agent_trade.volume.abs() * agent_trade.price.abs();
            feature.cumulative_volume = cumulative_volume;

            if cumulative_turnover > 1e-10 {
                feature.cumulative_return = pnl_state.cumulative_pnl / cumulative_turnover;
            }

            if idx > 0 {
                let step_ret = feature.cumulative_return - prev_cumulative_return;
                returns_history.push(step_ret);
            }
            prev_cumulative_return = feature.cumulative_return;

            feature.sharpe_ratio = compute_sharpe_ratio(&returns_history);
            feature.sortino_ratio = compute_sortino_ratio(&returns_history);

            if pnl_state.max_drawdown > 1e-10 {
                feature.calmar_ratio = pnl_state.cumulative_pnl / pnl_state.max_drawdown;
            }

            feature.current_position = pnl_state.current_position;

            if let Some(first_trade) = agent_trades.first() {
                if first_trade.price.abs() > 1e-12 {
                    let buy_hold_return = (agent_trade.price - first_trade.price) / first_trade.price;
                    feature.excess_return = feature.cumulative_return - buy_hold_return;
                }
            }

            if let Some(all_agents) = all_agent_trades {
                feature.sync_rate_same_direction =
                    compute_sync_rate(agent_trade, all_agents, self.config.window_ns);
            } else {
                feature.sync_rate_same_direction = 1.0;
            }

            last_trade_time = Some(agent_trade.timestamp);
            prev_position = pnl_state.current_position;

            features.push(feature);
        }

        features
    }
}

/// 便捷函数：计算单个Agent的特征
pub fn compute_agent_features(
    market_trades: &[MarketTrade],
    agent_trades: &[AgentTrade],
    window_ns: i64,
    all_agent_trades: Option<&[&[AgentTrade]]>,
) -> Vec<AgentFeatures> {
    let config = FeatureConfig {
        window_ns,
        ..Default::default()
    };
    let calculator = FeatureCalculator::new(market_trades, config);
    calculator.compute_features(agent_trades, all_agent_trades)
}
