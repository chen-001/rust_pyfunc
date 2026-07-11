//! 分钟因子示例：计算全市场分钟数据的轻量统计因子。
//!
//! 作为分钟 pipeline 的端到端验证用例（非生产因子）。
//! 读取 close/volume/amount 三字段 → 对每只股票手算日内统计量 → fan-out 全市场。
//!
//! 优化原则：避免逐股票调 get_features_factors_rust_full（4767次 × O(n²) LZ复杂度）。
//! 直接手算 mean/std/skew/kurt/max/min，单日 < 1s。

use crate::minute_data_reader::read_minute_field;
use ndarray::Array2;
use pyo3::prelude::*;

/// 每只股票计算 3 个基础序列 × 8 个统计量 = 24 个因子。
/// 基础序列：ret（分钟收益率）、log_vol（对数成交量）、log_amt（对数成交额）
/// 统计量：mean, std, skew, kurt, max, min, range, autocorr1
const N_SERIES: usize = 3;
const N_STATS: usize = 8;
pub const N_FACTORS: usize = N_SERIES * N_STATS; // 24

/// 核心唯一真相源：读数据 → 计算 → 返回 (codes, vals)。
pub fn compute_minute_example_full(date: i64) -> std::io::Result<(Vec<String>, Vec<f32>)> {
    let (codes, close_data) = read_minute_field("close", date)?;
    let (_, volume_data) = read_minute_field("volume", date)?;
    let (_, amount_data) = read_minute_field("amount", date)?;

    let n_rows = close_data.nrows();
    let n_cols = close_data.ncols();

    let mut out_codes: Vec<String> = Vec::new();
    let mut out_vals: Vec<f32> = Vec::new();

    for j in 0..n_cols {
        // 跳过无效列（全 NaN）
        if !(0..n_rows).any(|i| close_data[(i, j)].is_finite()) {
            continue;
        }

        // 构建 3 个基础序列（跳过 NaN）
        let rets = minute_rets(&close_data, j, n_rows);
        let vols = finite_col(&volume_data, j, n_rows, |v| v > 0.0, |v| v.ln() as f32);
        let amts = finite_col(&amount_data, j, n_rows, |v| v > 0.0, |v| v.ln() as f32);

        // 逐序列计算 8 个统计量，追加到输出
        out_codes.push(codes[j].clone());
        for series in [&rets, &vols, &amts] {
            out_vals.extend(stats8(series));
        }
    }

    Ok((out_codes, out_vals))
}

/// 从 close 列计算分钟收益率序列。
fn minute_rets(data: &Array2<f64>, col: usize, n_rows: usize) -> Vec<f32> {
    let mut rets = Vec::with_capacity(n_rows);
    let mut prev = f64::NAN;
    for i in 0..n_rows {
        let c = data[(i, col)];
        if c.is_finite() && prev.is_finite() && prev > 0.0 {
            rets.push(((c / prev) - 1.0) as f32);
        }
        if c.is_finite() {
            prev = c;
        }
    }
    rets
}

/// 提取满足条件的有限值，可选变换。
fn finite_col(
    data: &Array2<f64>,
    col: usize,
    n_rows: usize,
    filter: impl Fn(f64) -> bool,
    transform: impl Fn(f64) -> f32,
) -> Vec<f32> {
    let mut out = Vec::with_capacity(n_rows);
    for i in 0..n_rows {
        let v = data[(i, col)];
        if v.is_finite() && filter(v) {
            out.push(transform(v));
        }
    }
    out
}

/// 对一个序列计算 8 个轻量统计量：mean, std, skew, kurt, max, min, range, autocorr1。
fn stats8(v: &[f32]) -> [f32; N_STATS] {
    if v.len() < 2 {
        return [f32::NAN; N_STATS];
    }
    let n = v.len() as f32;
    let mean = v.iter().sum::<f32>() / n;

    let mut var = 0.0f32;
    let mut m3 = 0.0f32;
    let mut m4 = 0.0f32;
    for &x in v {
        let d = x - mean;
        let d2 = d * d;
        var += d2;
        m3 += d2 * d;
        m4 += d2 * d2;
    }
    let std = (var / (n - 1.0)).sqrt();
    let skew = if std > 1e-12 {
        m3 / n / (std.powi(3))
    } else {
        f32::NAN
    };
    let kurt = if std > 1e-12 {
        m4 / n / (var / n).powi(2) - 3.0
    } else {
        f32::NAN
    };

    let mx = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mn = v.iter().copied().fold(f32::INFINITY, f32::min);

    // autocorr1 (lag=1)
    let autocorr1 = if v.len() >= 3 {
        let mean_adj: Vec<f32> = v.iter().map(|&x| x - mean).collect();
        let cov0: f32 = mean_adj.iter().map(|x| x * x).sum::<f32>() / n;
        let cov1: f32 = (0..v.len() - 1)
            .map(|i| mean_adj[i] * mean_adj[i + 1])
            .sum::<f32>()
            / n;
        if cov0.abs() > 1e-12 {
            cov1 / cov0
        } else {
            f32::NAN
        }
    } else {
        f32::NAN
    };

    [mean, std, skew, kurt, mx, mn, mx - mn, autocorr1]
}

/// 生成因子名列表（Rust 单源）。
pub fn minute_example_names() -> Vec<String> {
    let series = ["ret", "log_vol", "log_amt"];
    let stats = [
        "mean",
        "std",
        "skew",
        "kurt",
        "max",
        "min",
        "range",
        "autocorr1",
    ];
    series
        .iter()
        .flat_map(|s| stats.iter().map(move |st| format!("{s}_{st}")))
        .collect()
}

// ============================================================
// Python 包装
// ============================================================

#[pyfunction]
pub fn py_minute_example(date: i64) -> PyResult<(Vec<String>, Vec<f32>)> {
    compute_minute_example_full(date)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("{e:?}")))
}

#[pyfunction]
pub fn py_minute_example_names() -> Vec<String> {
    minute_example_names()
}
