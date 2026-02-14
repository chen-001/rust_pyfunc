//! Agent交易模拟器 - Python绑定

use super::acceleration_follow_agent::AccelerationFollowAgent;
use super::bottom_fishing_agent::BottomFishingAgent;
use super::buy_ratio_agent::BuyRatioAgent;
use super::exhaustion_reversal_agent::ExhaustionReversalAgent;
use super::follow_flow_agent::FollowFlowAgent;
use super::momentum_agent::MomentumAgent;
use super::simulator::{run_simulation, run_simulation_mixed};
use super::trait_def::TradingAgent;
use super::types::*;
use numpy::{IntoPyArray, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::types::PyList;

fn detect_timestamp_factor_to_ns(timestamps: &[i64]) -> i64 {
    if timestamps.len() < 2 {
        return 1_000_000;
    }

    let min_ts = *timestamps.iter().min().unwrap_or(&0);
    let max_ts = *timestamps.iter().max().unwrap_or(&0);
    let range = max_ts.saturating_sub(min_ts);

    let mut min_positive_diff = i64::MAX;
    for w in timestamps.windows(2).take(20_000) {
        let diff = w[1].saturating_sub(w[0]);
        if diff > 0 && diff < min_positive_diff {
            min_positive_diff = diff;
        }
    }

    if range >= 100_000_000_000 || min_positive_diff >= 1_000_000 {
        1
    } else {
        1_000_000
    }
}

fn normalize_timestamps_to_ns(timestamps: &[i64]) -> Vec<i64> {
    let factor = detect_timestamp_factor_to_ns(timestamps);
    if factor == 1 {
        timestamps.to_vec()
    } else {
        timestamps
            .iter()
            .map(|t| t.saturating_mul(factor))
            .collect::<Vec<_>>()
    }
}

fn validate_market_lengths(
    n_market: usize,
    prices: &[f64],
    volumes: &[f64],
    flags: &[i32],
) -> PyResult<()> {
    if prices.len() != n_market || volumes.len() != n_market || flags.len() != n_market {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "市场数组长度不一致：timestamps/prices/volumes/flags 必须等长",
        ));
    }
    Ok(())
}

fn validate_non_negative_configs(values: &[i64], name: &str) -> PyResult<()> {
    if values.iter().any(|v| *v < 0) {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "{} 不能为负数",
            name
        )));
    }
    Ok(())
}

fn validate_non_negative_float_configs(values: &[f64], name: &str) -> PyResult<()> {
    if values.iter().any(|v| *v < 0.0) {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "{} 不能为负数",
            name
        )));
    }
    Ok(())
}

fn validate_min_float_configs(values: &[f64], min_value: f64, name: &str) -> PyResult<()> {
    if values.iter().any(|v| *v < min_value) {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "{} 的每个值都必须 >= {}",
            name, min_value
        )));
    }
    Ok(())
}

fn validate_decay_factors(values: &[f64], name: &str) -> PyResult<()> {
    if values.iter().any(|v| *v < 0.0 || *v > 1.0) {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "{} 的每个值都必须在 [0, 1] 区间内",
            name
        )));
    }
    Ok(())
}

fn build_market_trades(
    timestamps: &[i64],
    prices: &[f64],
    volumes: &[f64],
    flags: &[i32],
) -> PyResult<Vec<MarketTrade>> {
    let n_market = timestamps.len();
    if n_market == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err("市场数据为空"));
    }

    validate_market_lengths(n_market, prices, volumes, flags)?;
    let ts_ns = normalize_timestamps_to_ns(timestamps);

    let mut market_trades: Vec<MarketTrade> = Vec::with_capacity(n_market);
    for i in 0..n_market {
        market_trades.push(MarketTrade {
            timestamp: ts_ns[i],
            price: prices[i],
            volume: volumes[i],
            turnover: prices[i] * volumes[i],
            flag: flags[i],
        });
    }
    Ok(market_trades)
}

fn convert_results_to_py(py: Python<'_>, results: Vec<AgentSimulationResult>) -> PyResult<Py<PyList>> {
    let py_results = PyList::empty(py);
    for result in results {
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("name", result.name)?;
        dict.set_item("n_trades", result.n_trades)?;
        dict.set_item("total_buy_volume", result.total_buy_volume)?;
        dict.set_item("total_sell_volume", result.total_sell_volume)?;
        dict.set_item("final_position", result.final_position)?;

        let n = result.trades.len();
        let mut indices = Vec::with_capacity(n);
        let mut directions = Vec::with_capacity(n);
        let mut trade_volumes = Vec::with_capacity(n);
        let mut trade_prices = Vec::with_capacity(n);

        for trade in result.trades {
            indices.push(trade.market_trade_idx);
            directions.push(trade.direction);
            trade_volumes.push(trade.volume);
            trade_prices.push(trade.price);
        }

        dict.set_item("market_indices", indices.into_pyarray(py).to_owned())?;
        dict.set_item("directions", directions.into_pyarray(py).to_owned())?;
        dict.set_item("volumes", trade_volumes.into_pyarray(py).to_owned())?;
        dict.set_item("prices", trade_prices.into_pyarray(py).to_owned())?;

        py_results.append(dict)?;
    }

    Ok(py_results.into())
}

#[derive(Clone, Copy)]
enum ThematicAgentType {
    BottomFishing,
    FollowFlow,
    AccelerationFollow,
    ExhaustionReversal,
}

fn parse_thematic_agent_type(agent_type: &str) -> Option<ThematicAgentType> {
    let t = agent_type.trim().to_ascii_lowercase();
    match t.as_str() {
        "bottom_fishing" | "bottom" | "bottom_fishing_agent" | "抄底" | "chaodi" => {
            Some(ThematicAgentType::BottomFishing)
        }
        "follow_flow" | "follow" | "follow_flow_agent" | "跟随" | "gensui" => {
            Some(ThematicAgentType::FollowFlow)
        }
        "acceleration_follow"
        | "accel_follow"
        | "acceleration_follow_agent"
        | "加速跟随"
        | "jiasu_gensui" => Some(ThematicAgentType::AccelerationFollow),
        "exhaustion_reversal"
        | "exhaustion"
        | "exhaustion_reversal_agent"
        | "衰竭抄底"
        | "衰竭反转"
        | "shuaijie_fanzhuan" => Some(ThematicAgentType::ExhaustionReversal),
        _ => None,
    }
}

/// Python可调用的动量型Agent模拟函数
#[pyfunction]
#[pyo3(signature = (
    timestamps,
    prices,
    volumes,
    flags,
    agent_names,
    lookback_ms_list,
    thresholds,
    is_percentages,
    fixed_trade_sizes,
    cooldown_ms_list,
    allow_shorts
))]
fn simulate_momentum_agents_py(
    py: Python<'_>,
    timestamps: PyReadonlyArray1<i64>,
    prices: PyReadonlyArray1<f64>,
    volumes: PyReadonlyArray1<f64>,
    flags: PyReadonlyArray1<i32>,
    agent_names: Vec<String>,
    lookback_ms_list: Vec<i64>,
    thresholds: Vec<f64>,
    is_percentages: Vec<bool>,
    fixed_trade_sizes: Vec<i64>,
    cooldown_ms_list: Vec<i64>,
    allow_shorts: Vec<bool>,
) -> PyResult<Py<PyList>> {
    let ts = timestamps.as_slice()?;
    let pr = prices.as_slice()?;
    let vo = volumes.as_slice()?;
    let fl = flags.as_slice()?;

    let market_trades = build_market_trades(ts, pr, vo, fl)?;

    let n_agents = agent_names.len();
    if n_agents == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err("agent_names 不能为空"));
    }

    if lookback_ms_list.len() != n_agents
        || thresholds.len() != n_agents
        || is_percentages.len() != n_agents
        || fixed_trade_sizes.len() != n_agents
        || cooldown_ms_list.len() != n_agents
        || allow_shorts.len() != n_agents
    {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "所有配置参数列表的长度必须相同",
        ));
    }

    validate_non_negative_configs(&lookback_ms_list, "lookback_ms_list")?;
    validate_non_negative_configs(&cooldown_ms_list, "cooldown_ms_list")?;
    validate_non_negative_configs(&fixed_trade_sizes, "fixed_trade_sizes")?;

    let mut agents: Vec<MomentumAgent> = Vec::with_capacity(n_agents);
    for i in 0..n_agents {
        let lookback_ns = lookback_ms_list[i].saturating_mul(1_000_000);
        let cooldown_ns = cooldown_ms_list[i].saturating_mul(1_000_000);

        agents.push(MomentumAgent::new(
            &agent_names[i],
            lookback_ns,
            thresholds[i],
            is_percentages[i],
            fixed_trade_sizes[i],
            cooldown_ns,
            allow_shorts[i],
        ));
    }

    let results = run_simulation(&market_trades, &mut agents);
    convert_results_to_py(py, results)
}

/// Python可调用的主买占比型Agent模拟函数
#[pyfunction]
#[pyo3(signature = (
    timestamps,
    prices,
    volumes,
    flags,
    agent_names,
    lookback_ms_list,
    thresholds,
    fixed_trade_sizes,
    cooldown_ms_list,
    allow_shorts
))]
fn simulate_buy_ratio_agents_py(
    py: Python<'_>,
    timestamps: PyReadonlyArray1<i64>,
    prices: PyReadonlyArray1<f64>,
    volumes: PyReadonlyArray1<f64>,
    flags: PyReadonlyArray1<i32>,
    agent_names: Vec<String>,
    lookback_ms_list: Vec<i64>,
    thresholds: Vec<f64>,
    fixed_trade_sizes: Vec<i64>,
    cooldown_ms_list: Vec<i64>,
    allow_shorts: Vec<bool>,
) -> PyResult<Py<PyList>> {
    let ts = timestamps.as_slice()?;
    let pr = prices.as_slice()?;
    let vo = volumes.as_slice()?;
    let fl = flags.as_slice()?;

    let market_trades = build_market_trades(ts, pr, vo, fl)?;

    let n_agents = agent_names.len();
    if n_agents == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err("agent_names 不能为空"));
    }

    if lookback_ms_list.len() != n_agents
        || thresholds.len() != n_agents
        || fixed_trade_sizes.len() != n_agents
        || cooldown_ms_list.len() != n_agents
        || allow_shorts.len() != n_agents
    {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "所有配置参数列表的长度必须相同",
        ));
    }

    validate_non_negative_configs(&lookback_ms_list, "lookback_ms_list")?;
    validate_non_negative_configs(&cooldown_ms_list, "cooldown_ms_list")?;
    validate_non_negative_configs(&fixed_trade_sizes, "fixed_trade_sizes")?;

    let mut agents: Vec<BuyRatioAgent> = Vec::with_capacity(n_agents);
    for i in 0..n_agents {
        let lookback_ns = lookback_ms_list[i].saturating_mul(1_000_000);
        let cooldown_ns = cooldown_ms_list[i].saturating_mul(1_000_000);

        agents.push(BuyRatioAgent::new(
            &agent_names[i],
            lookback_ns,
            thresholds[i],
            fixed_trade_sizes[i],
            cooldown_ns,
            allow_shorts[i],
        ));
    }

    let results = run_simulation(&market_trades, &mut agents);
    convert_results_to_py(py, results)
}

/// Python可调用的统一主题Agent模拟函数（可在同一批次混合多种agent_type）
#[pyfunction]
#[pyo3(signature = (
    timestamps,
    prices,
    volumes,
    flags,
    agent_names,
    agent_types,
    short_window_ms_list,
    trend_window_ms_list,
    amount_thresholds,
    acceleration_factors,
    decay_factors,
    cooldown_ms_list
))]
fn simulate_thematic_agents_py(
    py: Python<'_>,
    timestamps: PyReadonlyArray1<i64>,
    prices: PyReadonlyArray1<f64>,
    volumes: PyReadonlyArray1<f64>,
    flags: PyReadonlyArray1<i32>,
    agent_names: Vec<String>,
    agent_types: Vec<String>,
    short_window_ms_list: Vec<i64>,
    trend_window_ms_list: Vec<i64>,
    amount_thresholds: Vec<f64>,
    acceleration_factors: Vec<f64>,
    decay_factors: Vec<f64>,
    cooldown_ms_list: Vec<i64>,
) -> PyResult<Py<PyList>> {
    let ts = timestamps.as_slice()?;
    let pr = prices.as_slice()?;
    let vo = volumes.as_slice()?;
    let fl = flags.as_slice()?;
    let market_trades = build_market_trades(ts, pr, vo, fl)?;

    let n_agents = agent_names.len();
    if n_agents == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err("agent_names 不能为空"));
    }
    if agent_types.len() != n_agents
        || short_window_ms_list.len() != n_agents
        || trend_window_ms_list.len() != n_agents
        || amount_thresholds.len() != n_agents
        || acceleration_factors.len() != n_agents
        || decay_factors.len() != n_agents
        || cooldown_ms_list.len() != n_agents
    {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "所有配置参数列表的长度必须相同",
        ));
    }

    validate_non_negative_configs(&short_window_ms_list, "short_window_ms_list")?;
    validate_non_negative_configs(&trend_window_ms_list, "trend_window_ms_list")?;
    validate_non_negative_configs(&cooldown_ms_list, "cooldown_ms_list")?;
    validate_non_negative_float_configs(&amount_thresholds, "amount_thresholds")?;

    let mut agents: Vec<Box<dyn TradingAgent>> = Vec::with_capacity(n_agents);
    for i in 0..n_agents {
        let kind = parse_thematic_agent_type(&agent_types[i]).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "agent_types[{}] 无效: {}，可选: bottom_fishing/follow_flow/acceleration_follow/exhaustion_reversal（含中英文别名）",
                i, agent_types[i]
            ))
        })?;

        let short_ns = short_window_ms_list[i].saturating_mul(1_000_000);
        let trend_ns = trend_window_ms_list[i].saturating_mul(1_000_000);
        let cooldown_ns = cooldown_ms_list[i].saturating_mul(1_000_000);

        match kind {
            ThematicAgentType::BottomFishing => {
                agents.push(Box::new(BottomFishingAgent::new(
                    &agent_names[i],
                    short_ns,
                    trend_ns,
                    cooldown_ns,
                )));
            }
            ThematicAgentType::FollowFlow => {
                agents.push(Box::new(FollowFlowAgent::new(
                    &agent_names[i],
                    short_ns,
                    trend_ns,
                    amount_thresholds[i],
                    cooldown_ns,
                )));
            }
            ThematicAgentType::AccelerationFollow => {
                if acceleration_factors[i] < 1.0 {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "acceleration_factors[{}] 必须 >= 1.0",
                        i
                    )));
                }
                agents.push(Box::new(AccelerationFollowAgent::new(
                    &agent_names[i],
                    short_ns,
                    trend_ns,
                    amount_thresholds[i],
                    acceleration_factors[i],
                    cooldown_ns,
                )));
            }
            ThematicAgentType::ExhaustionReversal => {
                if !(0.0..=1.0).contains(&decay_factors[i]) {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "decay_factors[{}] 必须在 [0, 1] 区间内",
                        i
                    )));
                }
                agents.push(Box::new(ExhaustionReversalAgent::new(
                    &agent_names[i],
                    short_ns,
                    trend_ns,
                    amount_thresholds[i],
                    decay_factors[i],
                    cooldown_ns,
                )));
            }
        }
    }

    let results = run_simulation_mixed(&market_trades, &mut agents);
    convert_results_to_py(py, results)
}

/// Python可调用的抄底型Agent模拟函数
#[pyfunction]
#[pyo3(signature = (
    timestamps,
    prices,
    volumes,
    flags,
    agent_names,
    short_window_ms_list,
    trend_window_ms_list,
    cooldown_ms_list
))]
fn simulate_bottom_fishing_agents_py(
    py: Python<'_>,
    timestamps: PyReadonlyArray1<i64>,
    prices: PyReadonlyArray1<f64>,
    volumes: PyReadonlyArray1<f64>,
    flags: PyReadonlyArray1<i32>,
    agent_names: Vec<String>,
    short_window_ms_list: Vec<i64>,
    trend_window_ms_list: Vec<i64>,
    cooldown_ms_list: Vec<i64>,
) -> PyResult<Py<PyList>> {
    let ts = timestamps.as_slice()?;
    let pr = prices.as_slice()?;
    let vo = volumes.as_slice()?;
    let fl = flags.as_slice()?;
    let market_trades = build_market_trades(ts, pr, vo, fl)?;

    let n_agents = agent_names.len();
    if n_agents == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err("agent_names 不能为空"));
    }
    if short_window_ms_list.len() != n_agents
        || trend_window_ms_list.len() != n_agents
        || cooldown_ms_list.len() != n_agents
    {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "所有配置参数列表的长度必须相同",
        ));
    }

    validate_non_negative_configs(&short_window_ms_list, "short_window_ms_list")?;
    validate_non_negative_configs(&trend_window_ms_list, "trend_window_ms_list")?;
    validate_non_negative_configs(&cooldown_ms_list, "cooldown_ms_list")?;

    let mut agents: Vec<BottomFishingAgent> = Vec::with_capacity(n_agents);
    for i in 0..n_agents {
        agents.push(BottomFishingAgent::new(
            &agent_names[i],
            short_window_ms_list[i].saturating_mul(1_000_000),
            trend_window_ms_list[i].saturating_mul(1_000_000),
            cooldown_ms_list[i].saturating_mul(1_000_000),
        ));
    }

    let results = run_simulation(&market_trades, &mut agents);
    convert_results_to_py(py, results)
}

/// Python可调用的跟随型Agent模拟函数
#[pyfunction]
#[pyo3(signature = (
    timestamps,
    prices,
    volumes,
    flags,
    agent_names,
    short_window_ms_list,
    trend_window_ms_list,
    amount_thresholds,
    cooldown_ms_list
))]
fn simulate_follow_flow_agents_py(
    py: Python<'_>,
    timestamps: PyReadonlyArray1<i64>,
    prices: PyReadonlyArray1<f64>,
    volumes: PyReadonlyArray1<f64>,
    flags: PyReadonlyArray1<i32>,
    agent_names: Vec<String>,
    short_window_ms_list: Vec<i64>,
    trend_window_ms_list: Vec<i64>,
    amount_thresholds: Vec<f64>,
    cooldown_ms_list: Vec<i64>,
) -> PyResult<Py<PyList>> {
    let ts = timestamps.as_slice()?;
    let pr = prices.as_slice()?;
    let vo = volumes.as_slice()?;
    let fl = flags.as_slice()?;
    let market_trades = build_market_trades(ts, pr, vo, fl)?;

    let n_agents = agent_names.len();
    if n_agents == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err("agent_names 不能为空"));
    }
    if short_window_ms_list.len() != n_agents
        || trend_window_ms_list.len() != n_agents
        || amount_thresholds.len() != n_agents
        || cooldown_ms_list.len() != n_agents
    {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "所有配置参数列表的长度必须相同",
        ));
    }

    validate_non_negative_configs(&short_window_ms_list, "short_window_ms_list")?;
    validate_non_negative_configs(&trend_window_ms_list, "trend_window_ms_list")?;
    validate_non_negative_configs(&cooldown_ms_list, "cooldown_ms_list")?;
    validate_non_negative_float_configs(&amount_thresholds, "amount_thresholds")?;

    let mut agents: Vec<FollowFlowAgent> = Vec::with_capacity(n_agents);
    for i in 0..n_agents {
        agents.push(FollowFlowAgent::new(
            &agent_names[i],
            short_window_ms_list[i].saturating_mul(1_000_000),
            trend_window_ms_list[i].saturating_mul(1_000_000),
            amount_thresholds[i],
            cooldown_ms_list[i].saturating_mul(1_000_000),
        ));
    }

    let results = run_simulation(&market_trades, &mut agents);
    convert_results_to_py(py, results)
}

/// Python可调用的加速跟随型Agent模拟函数
#[pyfunction]
#[pyo3(signature = (
    timestamps,
    prices,
    volumes,
    flags,
    agent_names,
    short_window_ms_list,
    trend_window_ms_list,
    amount_thresholds,
    acceleration_factors,
    cooldown_ms_list
))]
fn simulate_acceleration_follow_agents_py(
    py: Python<'_>,
    timestamps: PyReadonlyArray1<i64>,
    prices: PyReadonlyArray1<f64>,
    volumes: PyReadonlyArray1<f64>,
    flags: PyReadonlyArray1<i32>,
    agent_names: Vec<String>,
    short_window_ms_list: Vec<i64>,
    trend_window_ms_list: Vec<i64>,
    amount_thresholds: Vec<f64>,
    acceleration_factors: Vec<f64>,
    cooldown_ms_list: Vec<i64>,
) -> PyResult<Py<PyList>> {
    let ts = timestamps.as_slice()?;
    let pr = prices.as_slice()?;
    let vo = volumes.as_slice()?;
    let fl = flags.as_slice()?;
    let market_trades = build_market_trades(ts, pr, vo, fl)?;

    let n_agents = agent_names.len();
    if n_agents == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err("agent_names 不能为空"));
    }
    if short_window_ms_list.len() != n_agents
        || trend_window_ms_list.len() != n_agents
        || amount_thresholds.len() != n_agents
        || acceleration_factors.len() != n_agents
        || cooldown_ms_list.len() != n_agents
    {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "所有配置参数列表的长度必须相同",
        ));
    }

    validate_non_negative_configs(&short_window_ms_list, "short_window_ms_list")?;
    validate_non_negative_configs(&trend_window_ms_list, "trend_window_ms_list")?;
    validate_non_negative_configs(&cooldown_ms_list, "cooldown_ms_list")?;
    validate_non_negative_float_configs(&amount_thresholds, "amount_thresholds")?;
    validate_min_float_configs(&acceleration_factors, 1.0, "acceleration_factors")?;

    let mut agents: Vec<AccelerationFollowAgent> = Vec::with_capacity(n_agents);
    for i in 0..n_agents {
        agents.push(AccelerationFollowAgent::new(
            &agent_names[i],
            short_window_ms_list[i].saturating_mul(1_000_000),
            trend_window_ms_list[i].saturating_mul(1_000_000),
            amount_thresholds[i],
            acceleration_factors[i],
            cooldown_ms_list[i].saturating_mul(1_000_000),
        ));
    }

    let results = run_simulation(&market_trades, &mut agents);
    convert_results_to_py(py, results)
}

/// Python可调用的衰竭抄底型Agent模拟函数
#[pyfunction]
#[pyo3(signature = (
    timestamps,
    prices,
    volumes,
    flags,
    agent_names,
    short_window_ms_list,
    trend_window_ms_list,
    amount_thresholds,
    decay_factors,
    cooldown_ms_list
))]
fn simulate_exhaustion_reversal_agents_py(
    py: Python<'_>,
    timestamps: PyReadonlyArray1<i64>,
    prices: PyReadonlyArray1<f64>,
    volumes: PyReadonlyArray1<f64>,
    flags: PyReadonlyArray1<i32>,
    agent_names: Vec<String>,
    short_window_ms_list: Vec<i64>,
    trend_window_ms_list: Vec<i64>,
    amount_thresholds: Vec<f64>,
    decay_factors: Vec<f64>,
    cooldown_ms_list: Vec<i64>,
) -> PyResult<Py<PyList>> {
    let ts = timestamps.as_slice()?;
    let pr = prices.as_slice()?;
    let vo = volumes.as_slice()?;
    let fl = flags.as_slice()?;
    let market_trades = build_market_trades(ts, pr, vo, fl)?;

    let n_agents = agent_names.len();
    if n_agents == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err("agent_names 不能为空"));
    }
    if short_window_ms_list.len() != n_agents
        || trend_window_ms_list.len() != n_agents
        || amount_thresholds.len() != n_agents
        || decay_factors.len() != n_agents
        || cooldown_ms_list.len() != n_agents
    {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "所有配置参数列表的长度必须相同",
        ));
    }

    validate_non_negative_configs(&short_window_ms_list, "short_window_ms_list")?;
    validate_non_negative_configs(&trend_window_ms_list, "trend_window_ms_list")?;
    validate_non_negative_configs(&cooldown_ms_list, "cooldown_ms_list")?;
    validate_non_negative_float_configs(&amount_thresholds, "amount_thresholds")?;
    validate_decay_factors(&decay_factors, "decay_factors")?;

    let mut agents: Vec<ExhaustionReversalAgent> = Vec::with_capacity(n_agents);
    for i in 0..n_agents {
        agents.push(ExhaustionReversalAgent::new(
            &agent_names[i],
            short_window_ms_list[i].saturating_mul(1_000_000),
            trend_window_ms_list[i].saturating_mul(1_000_000),
            amount_thresholds[i],
            decay_factors[i],
            cooldown_ms_list[i].saturating_mul(1_000_000),
        ));
    }

    let results = run_simulation(&market_trades, &mut agents);
    convert_results_to_py(py, results)
}

pub fn register_functions(m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(simulate_momentum_agents_py, m)?)?;
    m.add_function(wrap_pyfunction!(simulate_buy_ratio_agents_py, m)?)?;
    m.add_function(wrap_pyfunction!(simulate_thematic_agents_py, m)?)?;
    m.add_function(wrap_pyfunction!(simulate_bottom_fishing_agents_py, m)?)?;
    m.add_function(wrap_pyfunction!(simulate_follow_flow_agents_py, m)?)?;
    m.add_function(wrap_pyfunction!(simulate_acceleration_follow_agents_py, m)?)?;
    m.add_function(wrap_pyfunction!(simulate_exhaustion_reversal_agents_py, m)?)?;
    Ok(())
}
