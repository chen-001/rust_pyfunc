//! Agent交易模拟器 - Python绑定

use super::buy_ratio_agent::BuyRatioAgent;
use super::momentum_agent::MomentumAgent;
use super::simulator::run_simulation;
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

    let n_market = ts.len();
    if n_market == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err("市场数据为空"));
    }

    validate_market_lengths(n_market, pr, vo, fl)?;

    let ts_ns = normalize_timestamps_to_ns(ts);

    let mut market_trades: Vec<MarketTrade> = Vec::with_capacity(n_market);
    for i in 0..n_market {
        market_trades.push(MarketTrade {
            timestamp: ts_ns[i],
            price: pr[i],
            volume: vo[i],
            turnover: pr[i] * vo[i],
            flag: fl[i],
        });
    }

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

    let n_market = ts.len();
    if n_market == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err("市场数据为空"));
    }

    validate_market_lengths(n_market, pr, vo, fl)?;

    let ts_ns = normalize_timestamps_to_ns(ts);

    let mut market_trades: Vec<MarketTrade> = Vec::with_capacity(n_market);
    for i in 0..n_market {
        market_trades.push(MarketTrade {
            timestamp: ts_ns[i],
            price: pr[i],
            volume: vo[i],
            turnover: pr[i] * vo[i],
            flag: fl[i],
        });
    }

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

pub fn register_functions(m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(simulate_momentum_agents_py, m)?)?;
    m.add_function(wrap_pyfunction!(simulate_buy_ratio_agents_py, m)?)?;
    Ok(())
}
