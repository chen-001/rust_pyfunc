//! sandbox_yhyb_indtop：一呼百应补充因子（行业头部配对池）沙箱验证。
pub mod fast_csv_reader;
pub mod yhyb_metrics;
pub mod yhyb_network;
pub mod yhyb_ext;

// 最小 stub（yhyb_metrics::compute_yhyb_full 的备份写路径；沙箱不写备份）
pub mod backup_reader {
    pub struct TaskResult {
        pub date: i64,
        pub code: String,
        pub timestamp: i64,
        pub facs: Vec<f32>,
    }
}
pub mod backup_writer {
    pub fn save_results_to_backup(
        _results: &[crate::backup_reader::TaskResult],
        _path: &str,
        _n: usize,
    ) -> Result<(), String> {
        Ok(())
    }
}

use pyo3::prelude::*;

#[pymodule]
fn sandbox_yhyb_indtop(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(yhyb_ext::py_yhyb_ext, m)?)?;
    m.add_function(wrap_pyfunction!(yhyb_ext::py_yhyb_ext_names, m)?)?;
    m.add_function(wrap_pyfunction!(yhyb_metrics::py_yhyb, m)?)?;
    m.add_function(wrap_pyfunction!(yhyb_metrics::py_yhyb_params, m)?)?;
    m.add_function(wrap_pyfunction!(yhyb_metrics::py_yhyb_names, m)?)?;
    Ok(())
}
