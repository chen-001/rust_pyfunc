//! 沙箱入口：复制主项目的读数据、降维、退火核心与 v1 模块，加上新版 anneal_cross_peer_v2。
#![allow(dead_code, unused_imports, unused_variables, unused_mut, clippy::all)]
pub mod anneal_cross_peer_metrics;
pub mod anneal_cross_peer_v2_metrics;
pub mod anneal_regime_diff_metrics;
pub mod data_paths;
pub mod fast_csv_reader;
pub mod features;

use pyo3::prelude::*;

/// 沙箱自检：随机序列上「升序起点的 from_guess 退火」应与 anneal_core16 逐位相同，返回不一致个数。
#[pyfunction]
fn check_core_equal(n_cases: usize) -> usize {
    anneal_cross_peer_v2_metrics::debug_check_core_equal(n_cases)
}

#[pymodule]
fn acp2_sandbox(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(anneal_cross_peer_v2_metrics::py_anneal_cross_peer_v2, m)?)?;
    m.add_function(wrap_pyfunction!(anneal_cross_peer_v2_metrics::py_anneal_cross_peer_v2_names, m)?)?;
    m.add_function(wrap_pyfunction!(anneal_cross_peer_v2_metrics::py_anneal_cross_peer_v2_from_data, m)?)?;
    m.add_function(wrap_pyfunction!(anneal_cross_peer_metrics::py_anneal_cross_peer_from_data, m)?)?;
    m.add_function(wrap_pyfunction!(check_core_equal, m)?)?;
    Ok(())
}
