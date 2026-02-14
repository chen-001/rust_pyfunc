---
name: agent-trading-builder
description: 构建Rust Agent交易模拟器中的自定义Agent类型。当用户需要创建新的Agent类型（如抄底/跟随/加速跟随/衰竭反转，或基于成交量、波动率、订单簿深度等触发条件）时使用。生成符合TradingAgent trait的完整Rust代码，与simulate_momentum_agents_py和simulate_buy_ratio_agents_py保持输出格式一致。
---

# Agent交易模拟器构建器

根据用户描述的Agent交易逻辑，生成完整的Rust Agent实现代码。

## 本Skill默认通用规则

1. 默认允许卖空：视为Agent有无限底仓，可直接卖出（`allow_short=true`）
2. 固定单笔交易量：每次交易固定`100`股
3. 交易方向定义：买入信号=主动买（`66`），卖出信号=主动卖（`83`）
4. 同主题多Agent：同一批次Agent应共享同一执行框架，仅在某个参数或某个触发逻辑点上有差异
5. 触发顺序统一：先冷却期，再计算指标，最后判断触发
6. 冲突处理：同一时刻买卖同时满足时，默认不交易

## 工作流程

1. **理解需求**：明确Agent的触发条件和交易规则
2. **确定触发类型**：Agent根据什么条件决定买入/卖出
3. **触发条件的参数**：不同agent的触发条件的核心差异
4. **生成代码**：输出完整的Rust文件

## Agent基础结构

所有Agent必须实现`TradingAgent` trait：

```rust
pub trait TradingAgent {
    fn name(&self) -> &str;
    fn check_trigger(&self, market_trades: &[MarketTrade], current_idx: usize, current_time: i64) -> Option<i32>;
    fn trade_size(&self) -> i64;
    fn can_trade(&self, current_time: i64) -> bool;
    fn record_trade(&mut self, time: i64, market_idx: usize, direction: i32, price: f64);
    fn trades(&self) -> &[AgentTrade];
    fn trades_mut(&mut self) -> &mut Vec<AgentTrade>;
    fn last_trade_time(&self) -> i64;
    fn set_last_trade_time(&mut self, time: i64);
}
```

## Agent结构体模板

```rust
pub struct YourAgentName {
    base: AgentBaseConfig,  // 共用配置：name, lookback_ms, fixed_trade_size, cooldown_ms, allow_short
    your_param: f64,        // 你的自定义参数
    last_trade_time: i64,
    trades: Vec<AgentTrade>,
}

impl YourAgentName {
    pub fn new(name: &str, lookback_ms: i64, your_param: f64, fixed_trade_size: i64, cooldown_ms: i64, allow_short: bool) -> Self {
        Self {
            base: AgentBaseConfig {
                name: name.to_string(),
                lookback_ms,
                fixed_trade_size,
                cooldown_ms,
                allow_short,
            },
            your_param,
            last_trade_time: -1,
            trades: Vec::new(),
        }
    }
}
```

## check_trigger实现模板

```rust
impl TradingAgent for YourAgentName {
    fn name(&self) -> &str { &self.base.name }
    
    fn check_trigger(&self, market_trades: &[MarketTrade], current_idx: usize, _current_time: i64) -> Option<i32> {
        // 1. 获取回看窗口数据
        let current_time = market_trades[current_idx].timestamp;
        let window_start = current_time - self.base.lookback_ms;
        
        // 2. 计算你的指标
        let indicator = calc_your_indicator(market_trades, window_start, current_time)?;
        
        // 3. 判断触发条件
        if indicator > self.your_param {
            Some(66)  // 买入
        } else if self.base.allow_short && indicator < -self.your_param {
            Some(83)  // 卖出
        } else {
            None
        }
    }
    
    fn trade_size(&self) -> i64 { self.base.fixed_trade_size }
    
    fn can_trade(&self, current_time: i64) -> bool {
        self.default_can_trade(current_time, self.base.cooldown_ms)
    }
    
    fn record_trade(&mut self, time: i64, market_idx: usize, direction: i32, price: f64) {
        self.default_record_trade(time, market_idx, direction, self.base.fixed_trade_size as f64, price);
    }
    
    fn trades(&self) -> &[AgentTrade] { &self.trades }
    fn trades_mut(&mut self) -> &mut Vec<AgentTrade> { &mut self.trades }
    fn last_trade_time(&self) -> i64 { self.last_trade_time }
    fn set_last_trade_time(&mut self, time: i64) { self.last_trade_time = time; }
}
```

## 输出要求

生成的Agent代码必须：
1. 放在 `src/agent_simulator/` 目录下
2. 文件名为 `{agent_type}_agent.rs`（如 `volume_spike_agent.rs`）
3. 在 `mod.rs` 中添加 `pub mod {agent_type}_agent;`
4. 在 `py_bindings.rs` 中添加对应的Python绑定函数

## 辅助函数参考

可用工具函数在 `src/agent_simulator/utils.rs`：

```rust
// 二分查找指定时间的价格
pub fn find_price_at_time(market_trades: &[MarketTrade], target_time: i64) -> Option<f64>

// 计算价格变化（支持百分比或绝对值）
pub fn calc_price_change(market_trades: &[MarketTrade], current_idx: usize, lookback_ms: i64, is_percentage: bool) -> Option<f64>

// 计算主买占比
pub fn calc_buy_ratio(market_trades: &[MarketTrade], current_time: i64, lookback_ms: i64) -> Option<f64>

// 计算窗口内成交量统计
pub fn calc_volume_stats(market_trades: &[MarketTrade], current_time: i64, lookback_ms: i64) -> (f64, f64, f64)  // (buy_vol, sell_vol, count)
```

## 示例Agent类型

### 动量型（MomentumAgent）
- 触发条件：价格变化超过阈值
- 买入：价格上涨 > 阈值
- 卖出：价格下跌 > 阈值（如果allow_short=true）

### 主买占比型（BuyRatioAgent）
- 触发条件：主买占比超过阈值
- 买入：主买占比 > 阈值
- 卖出：主买占比 < (1-阈值)

### 抄底型（BottomFishingAgent）
- 买入：过去2秒无主动卖出成交额，且最近15秒价格下跌
- 卖出：过去2秒无主动买入成交额，且最近15秒价格上涨
- 交易量：固定100股

### 跟随型（FollowFlowAgent）
- 买入：过去2秒主动买入成交额 > 阈值（如20万元），且最近15秒价格上涨
- 卖出：过去2秒主动卖出成交额 > 阈值（如20万元），且最近15秒价格下跌
- 交易量：固定100股

### 加速跟随型（AccelerationFollowAgent）
- 买入：过去2秒主动买入成交额 > 阈值，且相对前2秒放大（`curr >= k * prev`），且最近15秒价格上涨
- 卖出：过去2秒主动卖出成交额 > 阈值，且相对前2秒放大（`curr >= k * prev`），且最近15秒价格下跌
- 交易量：固定100股

### 衰竭抄底型（ExhaustionReversalAgent）
- 买入：前2秒主动卖出成交额很大（>`A`），最近2秒明显衰减（`curr <= d * prev`），且最近15秒价格下跌
- 卖出：前2秒主动买入成交额很大（>`A`），最近2秒明显衰减（`curr <= d * prev`），且最近15秒价格上涨
- 交易量：固定100股

### 可扩展类型（按需）
- 成交量突破型：窗口成交量突然放大
- 波动率突破型：回看窗口波动率突破
- 订单簿不平衡型：买卖盘力量严重失衡（需盘口快照）

## Python绑定函数模板

```rust
#[pyfunction]
#[pyo3(signature = (...))]
fn simulate_your_agents_py(
    py: Python<'_>,
    timestamps: PyReadonlyArray1<i64>,
    prices: PyReadonlyArray1<f64>,
    volumes: PyReadonlyArray1<f64>,
    flags: PyReadonlyArray1<i32>,
    agent_names: Vec<String>,
    lookback_ms_list: Vec<i64>,
    your_params: Vec<f64>,  // 你的自定义参数
    fixed_trade_sizes: Vec<i64>,
    cooldown_ms_list: Vec<i64>,
    allow_shorts: Vec<bool>,
) -> PyResult<Py<PyList>> {
    // 1. 转换市场数据
    // 2. 创建Agents
    // 3. 运行模拟
    // 4. 转换为Python结果（字典列表）
}
```

## 输出格式要求

Python函数返回 `List[Dict[str, Any]]`，每个字典包含：
- `'name'`: Agent名称 (str)
- `'n_trades'`: 交易次数 (int)
- `'total_buy_volume'`: 总买入量 (float)
- `'total_sell_volume'`: 总卖出量 (float)
- `'final_position'`: 最终持仓 (int)
- `'market_indices'`: NDArray[np.uint64]，对应市场数据索引
- `'directions'`: NDArray[np.int32]，66=买入，83=卖出
- `'volumes'`: NDArray[np.float64]，交易量
- `'prices'`: NDArray[np.float64]，交易价格

此格式与 `simulate_momentum_agents_py` 和 `simulate_buy_ratio_agents_py` 完全一致。

## 代码规范

1. **触发条件优先级**：先判断能否交易（冷却期），再计算指标，最后判断是否触发
2. **错误处理**：指标计算可能失败（如数据不足），使用 `Option` 返回
3. **性能优化**：使用二分查找快速定位回看窗口起点
4. **代码注释**：注明触发逻辑和参数含义
5. **默认参数**：提供合理的默认值便于测试

## 构建流程

1. 在 `src/agent_simulator/` 创建 `{agent_type}_agent.rs`
2. 在 `mod.rs` 添加 `pub mod {agent_type}_agent;`
3. 在 `py_bindings.rs` 添加绑定函数
4. 在 `mod.rs` 的 `py_bindings::register_functions` 注册新函数
5. 运行 `./alter.sh` 编译
6. 创建测试用例验证
