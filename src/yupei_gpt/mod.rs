//! 玉佩精炼版（GPT 改进版）横截面 pipeline。
//!
//! 与 sandbox_yupei_refined 同一套精确算法（对称复用优化版），拆成两段：
//! - `yupei_gpt`：单日 491 列（网络 + 控制/对比 + 价格响应 + 交互 + leadlag）
//! - `yupei_gpt_temporal`：29 个历史时序列，消费第一段写出的 .ygsnap.json
//!
//! 快照目录由环境变量 YUPEI_GPT_SNAP_DIR 指定；不设则只算单日列、不写快照。
pub mod compute;
pub mod extensions;
pub mod input;
pub mod kernel;
pub mod leadlag;
pub mod model;
pub mod names;
pub mod network;
pub mod network_simd;
pub mod network_stats;
pub mod temporal;

use pyo3::prelude::*;

#[pyfunction]
pub fn py_yupei_gpt(date: i64) -> PyResult<(Vec<String>, Vec<f32>)> {
    compute::compute_full(date)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("{e:?}")))
}

#[pyfunction]
pub fn py_yupei_gpt_names() -> Vec<String> {
    names::daily_names()
}

#[pyfunction]
pub fn py_yupei_gpt_temporal(date: i64) -> PyResult<(Vec<String>, Vec<f32>)> {
    compute::compute_temporal_full(date)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("{e:?}")))
}

#[pyfunction]
pub fn py_yupei_gpt_temporal_names() -> Vec<String> {
    names::temporal_names()
}
