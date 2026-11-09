//! 沙箱入口：复制主项目的读数据、降维、退火核心，加上 anneal_cross_peer_v3。
#![allow(dead_code, unused_imports, unused_variables, unused_mut, clippy::all)]
pub mod anneal_cross_peer_metrics;
pub mod anneal_cross_peer_v3_metrics;
pub mod anneal_regime_diff_metrics;
pub mod data_paths;
pub mod fast_csv_reader;
pub mod features;

use pyo3::prelude::*;

#[pymodule]
fn acp3_sandbox(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(anneal_cross_peer_v3_metrics::py_anneal_cross_peer_v3, m)?)?;
    m.add_function(wrap_pyfunction!(anneal_cross_peer_v3_metrics::py_anneal_cross_peer_v3_names, m)?)?;
    m.add_function(wrap_pyfunction!(anneal_cross_peer_v3_metrics::py_anneal_cross_peer_v3_build_profile, m)?)?;
    Ok(())
}
