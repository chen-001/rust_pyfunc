//! 横截面因子示例 + 模板：读一天全市场 Level2 逐笔 → per-stock 算特征 → 横截面标准化。
//!
//! 这是 `run_factor_pipeline_cross_section` 的端到端验证用例，也是**新增横截面因子的标准模板**。
//!
//! # 横截面因子的标准范式（A 类：可分解型）
//! 1. **rayon 并行读全市场** transaction（per-stock 文件，5000+ 个并行读）
//! 2. **per-stock 算中间特征**（复用现有单股 pipeline_xxx，或新写轻量统计）
//! 3. **横截面交互**（标准化 / 排名 / 占比 / 相关性等 map-reduce）
//! 4. **fan-out** → (codes, vals)，每只股票 N_FACTORS 个值
//!
//! # 与 minute_example / per-stock 因子的本质区别
//! - 数据源：Level2 CSV（per-stock 文件，需并行读），非 HDF5 列式
//! - 计算包含**跨股票横截面运算**（本例是 z-score 标准化），这是横截面因子的核心
//! - 输出是 per-stock 但**用了全市场信息**（标准化需要全市场均值/标准差）
//!
//! # 性能（实测 20251231）
//! - 读全市场 5000 股逐笔：~13s（20线程）/ ~10s（50线程），磁盘 ~2 GB/s
//! - 内存峰值：~18GB（逐笔+盘口全量），机器 1.5TB 富余
//! - 计算重因子（≥30s/天）配合 4 进程×50 线程异步，磁盘不争抢、CPU 满载

use crate::fast_csv_reader::{read_market_fast_inner, read_trade_fast_inner, TradeRecord};
use pyo3::prelude::*;
use rayon::prelude::*;
use std::collections::BTreeSet;
use std::fs;

/// 每只股票输出的因子数（示例：3 个横截面标准化特征）。
/// 新写因子时改成自己的值，并同步 `cross_section_example_names()`。
pub const N_FACTORS: usize = 3;

/// 列出某天某子目录下所有股票代码（从文件名 `{code}_{date}_{type}.csv` 提取）。
/// 供横截面因子枚举全市场股票用。
pub fn list_codes(date: i64, subdir: &str) -> Vec<String> {
    let dir = format!("/ssd_data/stock/{date}/{subdir}");
    let mut set = BTreeSet::new();
    if let Ok(entries) = fs::read_dir(&dir) {
        for e in entries.flatten() {
            let name = e.file_name().into_string().unwrap_or_default();
            if let Some(code) = name.split('_').next() {
                if code.chars().all(|c| c.is_ascii_digit()) {
                    set.insert(code.to_string());
                }
            }
        }
    }
    set.into_iter().collect()
}

/// per-stock 算 3 个原始特征：[log总成交量, log成交笔数, VWAP]。
///
/// 这是横截面因子的"步骤②"——per-stock 降维。生产因子可在这里复用任意单股逻辑
/// （如 `crate::order_pair_metrics`、`crate::features` 等），把全天逐笔压成固定长度特征。
fn per_stock_features(trades: &[TradeRecord]) -> Option<[f64; N_FACTORS]> {
    if trades.is_empty() {
        return None;
    }
    let total_vol: f64 = trades.iter().map(|t| t.volume as f64).sum();
    let total_amt: f64 = trades.iter().map(|t| t.turnover as f64).sum();
    if total_vol <= 0.0 || !total_amt.is_finite() {
        return None;
    }
    let vwap = total_amt / total_vol;
    let n = trades.len() as f64;
    Some([total_vol.ln(), n.ln(), vwap])
}

/// 横截面 z-score 标准化（就地修改）：减均值除标准差，仅用有限值统计。
///
/// 这是横截面因子的"步骤③"——跨股票运算。本例是简单标准化；
/// 生产因子可换成排名、行业中性化、占比、相关性矩阵等任意横截面逻辑。
fn zscore_column(vals: &mut [f64]) {
    let finite: Vec<f64> = vals.iter().filter(|v| v.is_finite()).copied().collect();
    if finite.len() < 2 {
        return;
    }
    let mean = finite.iter().sum::<f64>() / finite.len() as f64;
    let var = finite.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / finite.len() as f64;
    let std = var.sqrt();
    if std < 1e-12 {
        return;
    }
    for v in vals.iter_mut() {
        if v.is_finite() {
            *v = (*v - mean) / std;
        }
    }
}

/// 核心唯一真相源：读全市场 → per-stock 特征 → 横截面标准化 → (codes, vals)。
///
/// pipeline 和 Python 入口的共同调用点。改因子逻辑只改这里。
///
/// 返回：
/// - `codes`：有效股票代码列表（长度 n_stocks）
/// - `vals`：扁平化因子值，长度 = n_stocks × N_FACTORS，每 N_FACTORS 个对应一只股票
pub fn compute_cross_section_example_full(date: i64) -> std::io::Result<(Vec<String>, Vec<f32>)> {
    let codes = list_codes(date, "transaction");

    // ① rayon 并行读全市场逐笔 + per-stock 算特征（步骤①+②）
    //    parallel_threshold=usize::MAX：单文件串行解析，靠外层 par_iter 并行（与现有 pipeline 一致）
    let features: Vec<Option<[f64; N_FACTORS]>> = codes
        .par_iter()
        .map(|code| {
            let trades = read_trade_fast_inner(code, date, false, true, usize::MAX).ok()?;
            per_stock_features(&trades)
        })
        .collect();

    // ② 分离有效股票，组装特征矩阵（n_stocks × N_FACTORS）
    let mut valid_codes: Vec<String> = Vec::new();
    let mut feat_matrix: Vec<[f64; N_FACTORS]> = Vec::new();
    for (code, feat) in codes.iter().zip(features.iter()) {
        if let Some(f) = feat {
            valid_codes.push(code.clone());
            feat_matrix.push(*f);
        }
    }
    let n = valid_codes.len();
    if n == 0 {
        return Ok((Vec::new(), Vec::new()));
    }

    // ③ 横截面 z-score：每列（每个因子）独立标准化（步骤③）
    for j in 0..N_FACTORS {
        let mut col: Vec<f64> = feat_matrix.iter().map(|f| f[j]).collect();
        zscore_column(&mut col);
        for (i, &v) in col.iter().enumerate() {
            feat_matrix[i][j] = v;
        }
    }

    // ④ fan-out：扁平化输出（步骤④）
    let mut out_vals = Vec::with_capacity(n * N_FACTORS);
    for f in &feat_matrix {
        out_vals.extend(f.iter().map(|&v| v as f32));
    }

    Ok((valid_codes, out_vals))
}

/// 因子名（与 N_FACTORS 严格对齐，单一源）。
pub fn cross_section_example_names() -> Vec<String> {
    vec![
        "zscore_log_volume".to_string(),
        "zscore_log_ntrades".to_string(),
        "zscore_vwap".to_string(),
    ]
}

// ============================================================
// Python 单日调试入口（薄包装，调核心，错误抛异常）
// ============================================================

/// Python 单日调试：返回 (codes, vals)，可见完整错误栈。
#[pyfunction]
pub fn py_cross_section_example(py: Python<'_>, date: i64) -> PyResult<(Vec<String>, Vec<f32>)> {
    compute_cross_section_example_full(date)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("{e:?}")))
}

/// Python 拿因子名。
#[pyfunction]
pub fn py_cross_section_example_names() -> Vec<String> {
    cross_section_example_names()
}

// ============================================================
// 测试
// ============================================================
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_names_count() {
        assert_eq!(cross_section_example_names().len(), N_FACTORS);
    }

    #[test]
    fn test_zscore_basic() {
        let mut v = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        zscore_column(&mut v);
        // 标准化后均值≈0，标准差≈1
        let mean = v.iter().sum::<f64>() / v.len() as f64;
        assert!(mean.abs() < 1e-10);
    }

    #[test]
    fn test_per_stock_features_empty() {
        assert!(per_stock_features(&[]).is_none());
    }
}
