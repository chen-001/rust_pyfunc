//! Agent交易特征计算模块 - 数据类型定义

/// 市场逐笔成交数据
#[derive(Clone, Debug)]
pub struct MarketTrade {
    pub timestamp: i64,        // 成交时间戳（统一为纳秒）
    pub price: f64,            // 成交价格
    pub volume: f64,           // 成交量
    pub turnover: f64,         // 成交额
    pub flag: i32,             // 66=主买, 83=主卖
    pub bid_order_id: i64,     // 买方订单编号
    pub ask_order_id: i64,     // 卖方订单编号
}

/// Agent的买卖记录（主动市价单）
#[derive(Clone, Debug)]
pub struct AgentTrade {
    pub timestamp: i64,        // 成交时间
    pub market_trade_idx: usize, // 对应MarketTrade的索引
    pub direction: i32,        // 66=Agent买入, 83=Agent卖出
    pub volume: f64,           // 成交量
    pub price: f64,            // 成交价格
}

/// Agent持仓记录（用于FIFO计算）
#[derive(Clone, Debug)]
pub struct PositionRecord {
    pub entry_time: i64,       // 买入时间
    pub entry_price: f64,      // 买入价格
    pub volume: f64,           // 持仓量
}

/// 时间窗口统计结果
#[derive(Clone, Debug, Default)]
pub struct WindowStats {
    pub trade_count: usize,    // 成交笔数
    pub price_series: Vec<f64>, // 价格序列
    pub returns: Vec<f64>,     // 收益率序列
    pub volatility: f64,       // 波动率（标准差）
    pub trend: f64,            // 趋势（收盘-开盘）/开盘
    pub mean_price: f64,       // 平均价格
}

/// 单个Agent的完整特征（22列）
#[derive(Clone, Debug, Default)]
pub struct AgentFeatures {
    pub timestamp: i64,                    // 0. 交易发生时间
    pub post_trade_price_trend: f64,       // 1. 成交后窗口内价格趋势
    pub order_id_diff: i64,                // 2. 两个订单编号差值绝对值
    pub pre_trade_count: usize,            // 3. 成交前窗口内市场成交笔数
    pub post_trade_count: usize,           // 4. 成交后窗口内市场成交笔数
    pub price_deviation: f64,              // 5. 相对日内均价的偏离
    pub pre_trade_volatility: f64,         // 6. 成交前窗口内波动率
    pub post_trade_volatility: f64,        // 7. 成交后窗口内波动率
    pub sync_rate_same_direction: f64,     // 8. 与同向其他Agent的同步率
    pub max_drawdown: f64,                 // 9. 截止此刻累计P&L的最大回撤
    pub max_drawdown_duration: i64,        // 10. 最大回撤持续时间
    pub profit_concentration: f64,         // 11. 盈利集中度(Top10%利润占比)
    pub gain_loss_ratio: f64,              // 12. 盈亏不对称性(平均盈利/平均亏损)
    pub avg_holding_duration: f64,         // 13. 平均持仓时长，纳秒
    pub exposure_utilization: f64,         // 14. 仓位利用率
    pub trading_density: f64,              // 15. 交易密度（笔/分钟）
    pub sharpe_ratio: f64,                 // 16. 夏普比率
    pub sortino_ratio: f64,                // 17. Sortino比率
    pub calmar_ratio: f64,                 // 18. Calmar比率
    pub cumulative_return: f64,            // 19. 累计收益率
    pub excess_return: f64,                // 20. 超额收益
    pub current_position: f64,             // 21. 当前持仓量
    pub cumulative_volume: f64,            // 22. 累计交易量
}

impl AgentFeatures {
    /// 转换为Vec<f64>用于输出
    pub fn to_vec(&self) -> Vec<f64> {
        vec![
            self.timestamp as f64,
            self.post_trade_price_trend,
            self.order_id_diff as f64,
            self.pre_trade_count as f64,
            self.post_trade_count as f64,
            self.price_deviation,
            self.pre_trade_volatility,
            self.post_trade_volatility,
            self.sync_rate_same_direction,
            self.max_drawdown,
            self.max_drawdown_duration as f64,
            self.profit_concentration,
            self.gain_loss_ratio,
            self.avg_holding_duration,
            self.exposure_utilization,
            self.trading_density,
            self.sharpe_ratio,
            self.sortino_ratio,
            self.calmar_ratio,
            self.cumulative_return,
            self.excess_return,
            self.current_position,
            self.cumulative_volume,
        ]
    }
}

/// P&L追踪状态
#[derive(Clone, Debug, Default)]
pub struct PnLState {
    pub realized_pnl: f64,                 // 已实现盈亏
    pub unrealized_pnl: f64,               // 未实现盈亏
    pub cumulative_pnl: f64,               // 累计盈亏
    pub max_cumulative_pnl: f64,           // 历史最大累计盈亏
    pub max_drawdown: f64,                 // 最大回撤
    pub max_drawdown_duration: i64,        // 最大回撤持续时间
    pub drawdown_start_time: Option<i64>,  // 当前回撤开始时间
    pub trades_pnl: Vec<f64>,              // 每笔交易的盈亏记录
    pub winning_trades: Vec<f64>,          // 盈利交易记录
    pub losing_trades: Vec<f64>,          // 亏损交易记录
    pub holding_records: Vec<PositionRecord>, // FIFO持仓队列
    pub current_position: f64,             // 当前净持仓
    pub position_entry_time: Option<i64>,  // 当前持仓建立时间
    pub total_exposure_time: i64,          // 总持仓时间
    pub first_trade_time: Option<i64>,     // 首笔交易时间
    pub trade_count: usize,                // 交易次数
}

/// 特征计算配置
#[derive(Clone, Debug)]
pub struct FeatureConfig {
    pub window_ns: i64,                    // 时间窗口（纳秒）
    pub risk_free_rate: f64,               // 无风险利率（年化）
    pub trading_hours_per_year: f64,       // 年交易小时数
}

impl Default for FeatureConfig {
    fn default() -> Self {
        Self {
            window_ns: 10_000_000_000,      // 默认10秒
            risk_free_rate: 0.0,            // 默认0（日内交易）
            trading_hours_per_year: 240.0 * 4.0, // 240天 * 4小时
        }
    }
}
