//! hm0~hm92 旧框架因子的 Rust 复现注册表。
//!
//! 设计目标：让每个系列只新增一个文件，互不冲突。
//!
//! 约定（每个系列文件 hmXXX.rs 必须提供三个同名函数）：
//!
//! ```ignore
//! pub fn names() -> Vec<String>;                                  // 与 compute_* 输出严格对齐
//! pub fn compute_stock(code: &str, date: i64) -> Option<Vec<f32>>;// 逐股系列用；不适用返回 None
//! pub fn compute_cross(date: i64) -> Option<(Vec<String>, Vec<f32>)>; // 横截面/分钟系列用；不适用返回 None
//! ```
//!
//! 流水线名约定为 `okay_<系列名>`，例如 `okay_hm003`。调用方按 kind 选择入口：
//! - `kind='per_stock'` → 走 `TaskMessage::Task`，worker 调 [`dispatch_stock`]
//! - `kind='cross_section'` → 走 `TaskMessage::MinuteTask`，worker 调 [`dispatch_cross`]
//!
//! `registry.rs` 由 `lib/okay_database/gen_registry.py` 扫描本目录自动生成，
//! 新增系列后跑一次那个脚本即可，不要手写。

// 用 include! 而不是 mod registry;：生成文件里的 `pub mod hmXXX;` 要按本目录解析，
// 若放进子模块 registry 里就会被当成 src/okay_database/registry/hmXXX.rs。
include!("registry.rs");

use pyo3::prelude::*;

/// Python 侧拿某个系列的因子名清单（不要在 py 文件里重写名字逻辑，否则双端会漂移）。
#[pyfunction]
pub fn okay_series_names(series: &str) -> Vec<String> {
    series_names(series).unwrap_or_default()
}

/// 单股单日调试入口：抛异常可见完整栈，用于和老 Python 结果逐值对比。
#[pyfunction]
pub fn okay_compute_stock(series: &str, code: &str, date: i64) -> PyResult<Vec<f32>> {
    match dispatch_stock_raw(series, code, date) {
        Some(v) => Ok(v),
        None => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "系列 {series} 没有逐股实现或该股该日算不出来"
        ))),
    }
}

/// 单日全市场调试入口（横截面 / 分钟系列）。
#[pyfunction]
pub fn okay_compute_cross(series: &str, date: i64) -> PyResult<(Vec<String>, Vec<f32>)> {
    dispatch_cross(series, date).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(format!("系列 {series} 没有横截面实现或该日算不出来"))
    })
}

/// 把系列输出裁剪/补齐到流水线声明的长度，长度不符不报错只补 NaN。
pub fn fit(mut values: Vec<f32>, expected_len: usize) -> Vec<f32> {
    if values.len() < expected_len {
        values.resize(expected_len, f32::NAN);
    } else if values.len() > expected_len {
        values.truncate(expected_len);
    }
    values
}

/// 逐股系列的兜底：跑不出来一律返回整条 NaN，保证 worker 不崩、因子个数恒定。
pub fn nan_stock(expected_len: usize) -> Vec<f32> {
    vec![f32::NAN; expected_len]
}
