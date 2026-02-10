//! Agent交易模拟器 - 数据类型定义

/// 市场逐笔成交数据（与agent_trading_features模块共享）
#[derive(Clone, Debug)]
pub struct MarketTrade {
    pub timestamp: i64,
    pub price: f64,
    pub volume: f64,
    pub turnover: f64,
    pub flag: i32,  // 66=主买, 83=主卖
}

impl MarketTrade {
    /// 判断是否是主买
    pub fn is_buy(&self) -> bool {
        self.flag == 66
    }
    
    /// 判断是否是主卖
    pub fn is_sell(&self) -> bool {
        self.flag == 83
    }
}

/// Agent交易记录
#[derive(Clone, Debug)]
pub struct AgentTrade {
    pub timestamp: i64,        // 成交时间
    pub market_trade_idx: usize, // 对应MarketTrade的索引
    pub direction: i32,        // 66=买入, 83=卖出
    pub volume: f64,           // 成交量
    pub price: f64,            // 成交价格
}

/// Agent基础配置（所有Agent共有）
#[derive(Clone, Debug)]
pub struct AgentBaseConfig {
    pub name: String,
    pub lookback_ms: i64,      // 回看时间窗口（内部统一按纳秒存储）
    pub fixed_trade_size: i64, // 固定交易数量（股）
    pub cooldown_ms: i64,      // 冷却期（内部统一按纳秒存储）
    pub allow_short: bool,     // 是否允许做空
}

impl Default for AgentBaseConfig {
    fn default() -> Self {
        Self {
            name: "UnnamedAgent".to_string(),
            lookback_ms: 10_000_000_000, // 默认10秒（纳秒）
            fixed_trade_size: 100, // 默认100股
            cooldown_ms: 0,        // 默认无冷却
            allow_short: true,     // 默认允许做空
        }
    }
}

/// Agent模拟结果
#[derive(Clone, Debug)]
pub struct AgentSimulationResult {
    pub name: String,
    pub trades: Vec<AgentTrade>,
    pub n_trades: usize,
    pub total_buy_volume: f64,
    pub total_sell_volume: f64,
    pub final_position: i64,
}

impl AgentSimulationResult {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            trades: Vec::new(),
            n_trades: 0,
            total_buy_volume: 0.0,
            total_sell_volume: 0.0,
            final_position: 0,
        }
    }
    
    pub fn add_trade(&mut self, trade: AgentTrade) {
        if trade.direction == 66 {
            self.total_buy_volume += trade.volume;
            self.final_position += trade.volume as i64;
        } else {
            self.total_sell_volume += trade.volume;
            self.final_position -= trade.volume as i64;
        }
        self.trades.push(trade);
        self.n_trades += 1;
    }
}

/// 转换为numpy数组格式的输出
#[derive(Clone, Debug)]
pub struct AgentOutput {
    pub name: String,
    pub market_indices: Vec<usize>,
    pub directions: Vec<i32>,
    pub volumes: Vec<f64>,
    pub prices: Vec<f64>,
}

impl From<AgentSimulationResult> for AgentOutput {
    fn from(result: AgentSimulationResult) -> Self {
        let n = result.trades.len();
        let mut market_indices = Vec::with_capacity(n);
        let mut directions = Vec::with_capacity(n);
        let mut volumes = Vec::with_capacity(n);
        let mut prices = Vec::with_capacity(n);
        
        for trade in result.trades {
            market_indices.push(trade.market_trade_idx);
            directions.push(trade.direction);
            volumes.push(trade.volume);
            prices.push(trade.price);
        }
        
        Self {
            name: result.name,
            market_indices,
            directions,
            volumes,
            prices,
        }
    }
}
