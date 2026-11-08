// 沙箱入口：v3 函数 + 对拍用的 v1 原版退火
pub mod anneal_regime_raw_excess_metrics;
pub mod data_paths;
pub mod fast_csv_reader;
pub mod features;
pub mod old_ref;

use anneal_regime_raw_excess_metrics as v3;
use pyo3::prelude::*;

/// 对拍：同一序列，v1 原版 16 指标 vs v3 的 22 指标（前 16 应逐位相同）
#[pyfunction]
fn py_compare_old_v3(seq: Vec<f32>, cap: usize) -> (Vec<f32>, Vec<f32>) {
    let n = seq.len();
    let m = (n.saturating_mul(n).saturating_mul(6)).max(600).min(cap);
    let mut ob = old_ref::FastAnnealBuf::new();
    let old = old_ref::anneal_core16(&seq, m, &mut ob);
    let mut nb = v3::AnnealBuf::new();
    let new = v3::anneal_metrics22(&seq, cap, &mut nb);
    (old.to_vec(), new.to_vec())
}

#[pymodule]
fn anneal_regime_v3_sbx(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(v3::py_anneal_regime_raw_excess, m)?)?;
    m.add_function(wrap_pyfunction!(v3::py_anneal_regime_raw_excess_names, m)?)?;
    m.add_function(wrap_pyfunction!(v3::py_anneal_regime_raw_excess_from_data, m)?)?;
    m.add_function(wrap_pyfunction!(py_compare_old_v3, m)?)?;
    Ok(())
}
