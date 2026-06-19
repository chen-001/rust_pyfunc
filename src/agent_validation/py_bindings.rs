//! Agent交互验证因子计算 - Python绑定

use super::compute_validation_factors;
use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

/// Python可调用的交互验证因子计算函数
///
/// 计算六类交互验证因子：前向验证(FV)、市场确认(MC)、盘口交互(OB)、
/// 交易后演化(PT)、跨Agent交互(CA)、Agent基础统计(AG)。
///
/// 返回值顺序与 Python _build_names() 严格一致。
#[pyfunction]
#[pyo3(signature = (
    market_timestamps,
    market_prices,
    market_volumes,
    market_flags,
    ob_timestamps,
    ob_bid_prc1,
    ob_ask_prc1,
    ob_bid_vol1,
    ob_ask_vol1,
    bid_order_ids,
    ask_order_ids,
    per_agent_market_indices,
    per_agent_directions,
    per_agent_volumes,
    fwd_horizons_sec,
    pt_horizons_sec,
))]
fn compute_agent_validation_factors_py(
    market_timestamps: PyReadonlyArray1<i64>,
    market_prices: PyReadonlyArray1<f64>,
    market_volumes: PyReadonlyArray1<f64>,
    market_flags: PyReadonlyArray1<i32>,
    ob_timestamps: PyReadonlyArray1<i64>,
    ob_bid_prc1: PyReadonlyArray1<f64>,
    ob_ask_prc1: PyReadonlyArray1<f64>,
    ob_bid_vol1: PyReadonlyArray1<f64>,
    ob_ask_vol1: PyReadonlyArray1<f64>,
    bid_order_ids: PyReadonlyArray1<i64>,
    ask_order_ids: PyReadonlyArray1<i64>,
    per_agent_market_indices: Vec<PyReadonlyArray1<i64>>,
    per_agent_directions: Vec<PyReadonlyArray1<i32>>,
    per_agent_volumes: Vec<PyReadonlyArray1<f64>>,
    fwd_horizons_sec: Vec<f64>,
    pt_horizons_sec: Vec<f64>,
) -> PyResult<Vec<f64>> {
    let mkt_ts = market_timestamps.as_slice()?.to_vec();
    let mkt_pr = market_prices.as_slice()?.to_vec();
    let mkt_vo = market_volumes.as_slice()?.to_vec();
    let mkt_fl = market_flags.as_slice()?.to_vec();

    let ob_ts = ob_timestamps.as_slice()?.to_vec();
    let ob_bid1 = ob_bid_prc1.as_slice()?.to_vec();
    let ob_ask1 = ob_ask_prc1.as_slice()?.to_vec();
    let ob_bid_vol1 = ob_bid_vol1.as_slice()?.to_vec();
    let ob_ask_vol1 = ob_ask_vol1.as_slice()?.to_vec();
    let bid_ids = bid_order_ids.as_slice()?.to_vec();
    let ask_ids = ask_order_ids.as_slice()?.to_vec();

    let per_agent_idx: Vec<Vec<i64>> = per_agent_market_indices
        .iter()
        .map(|a| a.as_slice().unwrap().to_vec())
        .collect();
    let per_agent_dir: Vec<Vec<i32>> = per_agent_directions
        .iter()
        .map(|a| a.as_slice().unwrap().to_vec())
        .collect();
    let per_agent_vol: Vec<Vec<f64>> = per_agent_volumes
        .iter()
        .map(|a| a.as_slice().unwrap().to_vec())
        .collect();

    Ok(compute_validation_factors(
        &mkt_ts,
        &mkt_pr,
        &mkt_vo,
        &mkt_fl,
        &ob_ts,
        &ob_bid1,
        &ob_ask1,
        &ob_bid_vol1,
        &ob_ask_vol1,
        &bid_ids,
        &ask_ids,
        &per_agent_idx,
        &per_agent_dir,
        &per_agent_vol,
        &fwd_horizons_sec,
        &pt_horizons_sec,
    ))
}

pub fn register_functions(m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_agent_validation_factors_py, m)?)?;
    Ok(())
}
