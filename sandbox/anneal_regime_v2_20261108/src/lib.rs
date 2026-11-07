// 沙箱入口：只暴露新版函数与调试用的单序列退火接口。
pub mod anneal_regime_excess_metrics;
pub mod data_paths;
pub mod fast_csv_reader;
pub mod features;

use pyo3::prelude::*;

#[pymodule]
fn anneal_regime_v2_sbx(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(anneal_regime_excess_metrics::py_anneal_regime_excess, m)?)?;
    m.add_function(wrap_pyfunction!(anneal_regime_excess_metrics::py_anneal_regime_excess_names, m)?)?;
    m.add_function(wrap_pyfunction!(anneal_regime_excess_metrics::py_anneal_regime_excess_from_data, m)?)?;
    m.add_function(wrap_pyfunction!(anneal_regime_excess_metrics::py_anneal_seq_real_null_debug, m)?)?;
    Ok(())
}
