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

/// 列出某天全市场**股票**代码（横截面枚举，有序去重；非股票标的已剔除）。
pub fn list_codes(date: i64) -> Vec<String> {
    crate::data_paths::list_stock_codes(date, "transaction")
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
    Ok(compute_urgency_ext_from_trades(&per_trades))
}

/// 由「每只股票一份逐笔记录」算迫切交易笔数。
///
/// 读磁盘（compute_urgency_ext_full）与 Python 传数组（py_urgency_ext_from_data）
/// 两条路径共用这一处；记录构造一侧走 build_trade_record（撤单在这里丢掉）。
/// 输入顺序就是输出 codes 的顺序；逐笔为空的股票在这里被跳过。
pub fn compute_urgency_ext_from_trades(
    per_trades: &[(String, Vec<TradeRecord>)],
) -> (Vec<String>, Vec<f32>) {
    if per_trades.is_empty() {
        return (Vec::new(), Vec::new());
    }
    // ② 全市场 |ratio| 95% 分位（与初版相同口径）
    let total: usize = per_trades.iter().map(|(_, t)| t.len()).sum();
    let mut ab: Vec<f32> = Vec::with_capacity(total);
    for (_, trades) in per_trades.iter() {
        for t in trades.iter() {
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
    for (code, trades) in per_trades.iter() {
        let mut cnt = 0u32;
        for t in trades.iter() {
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
        codes_out.push(code.clone());
        vals.push(cnt as f32);
    }
    (codes_out, vals)
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

/// Python 可调用（全市场数组版，供样例/分享用）：
/// `py_urgency_ext_from_data(codes, trade_arrays)`
///
/// 入参格式与 py_urgency_from_data 完全相同：每只股票一个 (n_i, 7) 数组，
/// 列序 [exchtime_us, price, volume, turnover, flag, bid_order, ask_order]。
/// 与 py_urgency_ext(date) 的输出逐位一致。
#[pyfunction]
pub fn py_urgency_ext_from_data(
    codes: Vec<String>,
    trade_arrays: Vec<numpy::PyReadonlyArray2<f64>>,
) -> PyResult<(Vec<String>, Vec<f32>)> {
    if codes.len() != trade_arrays.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "codes.len()={} != trade_arrays.len()={}",
            codes.len(),
            trade_arrays.len()
        )));
    }
    let mut per_trades: Vec<(String, Vec<TradeRecord>)> = Vec::with_capacity(codes.len());
    for (code, arr) in codes.iter().zip(trade_arrays.iter()) {
        let a = arr.as_array();
        if a.ncols() != 7 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "trade_arrays 每只股票必须是 (n, 7)：exchtime_us, price, volume, turnover, flag, bid_order, ask_order",
            ));
        }
        let mut trades: Vec<TradeRecord> = Vec::with_capacity(a.nrows());
        for i in 0..a.nrows() {
            let flag = a[[i, 4]] as i32;
            if flag == 32 {
                continue;
            }
            if let Some(r) = crate::fast_csv_reader::build_trade_record(
                a[[i, 0]] as i64,
                a[[i, 1]] as f32,
                a[[i, 2]] as f32,
                a[[i, 3]] as f32,
                flag,
                a[[i, 5]] as i64,
                a[[i, 6]] as i64,
                true,
            ) {
                trades.push(r);
            }
        }
        if trades.is_empty() {
            continue;
        }
        per_trades.push((code.clone(), trades));
    }
    Ok(compute_urgency_ext_from_trades(&per_trades))
}
