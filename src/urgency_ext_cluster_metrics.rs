//! COR-22「迫切交易联动」补充因子（正式版，仅入选因子）：迫切交易笔数。
//!
//! 判定结论（skill B 模式 1，supplement_ver=urgency_v2_supp_supplement_urgency_ext_cluster）：
//! 89 个候选因子中仅 `urgency_v1_ext_cluster_both_n_urg` 值得补充
//! （gap5 中性化 |IC|=0.0437 > 入选下限 0.0237，且未被初版高相关高 IC 因子阻挡）。
//! 本模块只实现这一个因子；评估目录保留全量 89 因子数据（不注册组合 store）。
//!
//! 计算口径与初版 urgency_metrics.rs 完全一致：
//!   ratio = (ask_order - bid_order) / (ask_order + bid_order)，sum<=0 → 0
//!   全市场当日 |ratio| 的 95% 分位 q95_abs；每股因子值 = 当日 |ratio| > q95_abs 的成交笔数。
//!   read_trade_fast_inner(code, date, false, true, usize::MAX) + retain(flag != 32)。
//!
//! 股票轴：默认输出当日有逐笔数据的所有股票；设置环境变量 RUST_PYFUNC_AXIS_ALLOWLIST
//! （每行一个 6 位代码的文本文件）时过滤到白名单（与初版 store 股票轴对齐用）。

use crate::fast_csv_reader::{read_trade_fast_inner, TradeRecord};
use pyo3::prelude::*;
use rayon::prelude::*;
use std::fs;
use std::sync::OnceLock;

pub const N_FACTORS: usize = 1;

/// 股票轴白名单（懒加载一次，进程内缓存）。
static AXIS_ALLOWLIST: OnceLock<Option<std::collections::HashSet<String>>> = OnceLock::new();

fn axis_allowlist() -> Option<&'static std::collections::HashSet<String>> {
    AXIS_ALLOWLIST
        .get_or_init(|| {
            std::env::var("RUST_PYFUNC_AXIS_ALLOWLIST").ok().and_then(|p| {
                fs::read_to_string(p).ok().map(|s| {
                    s.lines()
                        .map(|l| l.trim().to_string())
                        .filter(|x| !x.is_empty())
                        .collect()
                })
            })
        })
        .as_ref()
}

/// 列出某天全市场股票代码（横截面枚举，有序去重）。
pub fn list_codes(date: i64) -> Vec<String> {
    let dir = crate::data_paths::level2_dir(date, "transaction");
    let mut set = std::collections::BTreeSet::new();
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

/// 核心唯一真相源：读全市场 → 全市场 q95_abs → 每股迫切交易笔数 → (codes, vals)。
pub fn compute_urgency_ext_full(date: i64) -> std::io::Result<(Vec<String>, Vec<f32>)> {
    let mut codes = list_codes(date);
    if let Some(al) = axis_allowlist() {
        codes.retain(|c| al.contains(c.as_str()));
    }
    // ① rayon 并行读全市场（过滤撤单）
    let per_trades: Vec<(String, Vec<TradeRecord>)> = codes
        .par_iter()
        .filter_map(|code| {
            let mut trades = read_trade_fast_inner(code, date, false, true, usize::MAX).ok()?;
            trades.retain(|t| t.flag != 32);
            if trades.is_empty() {
                return None;
            }
            Some((code.clone(), trades))
        })
        .collect();
    if per_trades.is_empty() {
        return Ok((Vec::new(), Vec::new()));
    }
    // ② 全市场 |ratio| 95% 分位（与初版相同口径）
    let total: usize = per_trades.iter().map(|(_, t)| t.len()).sum();
    let mut ab: Vec<f32> = Vec::with_capacity(total);
    for (_, trades) in &per_trades {
        for t in trades {
            let sum = (t.ask_order + t.bid_order) as f64;
            let r = if sum > 0.0 {
                ((t.ask_order - t.bid_order) as f64 / sum) as f32
            } else {
                0.0
            };
            ab.push(r.abs());
        }
    }
    let n = ab.len();
    ab.par_sort_unstable_by(|a, b| a.total_cmp(b));
    let qidx = ((n as f64 * 0.95) as usize).min(n - 1);
    let q95_abs = ab[qidx];

    // ③ per-stock 计数（保序）
    let mut codes_out: Vec<String> = Vec::with_capacity(per_trades.len());
    let mut vals: Vec<f32> = Vec::with_capacity(per_trades.len());
    for (code, trades) in per_trades {
        let mut cnt = 0u32;
        for t in &trades {
            let sum = (t.ask_order + t.bid_order) as f64;
            let r = if sum > 0.0 {
                ((t.ask_order - t.bid_order) as f64 / sum) as f32
            } else {
                0.0
            };
            if r.abs() > q95_abs {
                cnt += 1;
            }
        }
        codes_out.push(code);
        vals.push(cnt as f32);
    }
    Ok((codes_out, vals))
}

/// 因子名（唯一真相源）。
pub fn ext_names() -> Vec<String> {
    vec!["urgency_v1_ext_cluster_both_n_urg".to_string()]
}

#[pyfunction]
pub fn py_urgency_ext(_py: Python<'_>, date: i64) -> PyResult<(Vec<String>, Vec<f32>)> {
    compute_urgency_ext_full(date).map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("{e:?}")))
}

#[pyfunction]
pub fn py_urgency_ext_names() -> Vec<String> {
    ext_names()
}
