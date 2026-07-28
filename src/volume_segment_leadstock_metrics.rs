//! 成交量分段领衔股相关因子（横截面）。
//!
//! # 算法
//! 1. 全市场逐笔混合 → 按 exchtime(time) 排序
//! 2. 依 EP1 成交量累计 → 把考察时段(EP2)按累计量划分成 100 段
//! 3. 每股在每段的 EP1 成交量 → core 序列(长度100)；EP3 可选占比
//! 4. 每段选成交量最大的 20 只 → 该段"领衔股"
//! 5. 每股 core 序列与每段 20 只领衔股 core 序列的相关 → 100×20 矩阵
//! 6. 降维：方法1(行 mean/std/skew/top5 各降维) + 方法2(20列协方差上三角10统计)
//! 7. 另：core 序列直接降维 + 该股与全市场 core 序列相关序列降维
//!
//! # 笛卡尔全集
//! EP1(5)×EP2(3)×EP3(2)×EP4(5) = 150 case，每 case 148 因子，共 22200 因子。
//! - EP1: main_buy / main_sell / small(<40%分位) / small_buy / small_sell
//! - EP2: allday / afternoon / tail30(尾盘30分钟)
//! - EP3: raw(成交量) / ratio(占该股总量比)
//! - EP4: all100 / hi50 / hi30 / lo50 / lo30(按热点一致度选段)
//!
//! ddof 约定（与 Python 原型一致）：自算的 std/skew 用总体(ddof=0)；协方差用样本(ddof=1)。
//! 降维统一走 get_features_factors_rust_full(with_threshold_counts=true)，单列23统计。

use crate::fast_csv_reader::{read_trade_fast_inner, TradeRecord};
use crate::features::get_features_factors_rust_full;
use ndarray::Array2;
use numpy::PyReadonlyArray2;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::BTreeSet;
use std::fs;

// ============================ 常量 ============================
pub const EP1S: &[&str] = &["main_buy", "main_sell", "small", "small_buy", "small_sell"];
pub const EP2S: &[&str] = &["allday", "afternoon", "tail30"];
pub const EP3S: &[&str] = &["raw", "ratio"];
pub const EP4S: &[&str] = &["all100", "hi50", "hi30", "lo50", "lo30"];

/// 单列降维统计量名(23个, 顺序对齐 get_features_factors_rust_full with_threshold_counts=true)。
const STAT_SUFFIXES: &[&str] = &[
    "mean", "median", "std", "skew", "kurt", "p5", "p25", "p75", "p95", "iqr", "cv",
    "autocorr1", "autocorr1_abs", "trend", "curvature", "quad_coef", "period_diff",
    "period_ratio", "mean_above_p90", "mean_below_p10", "lz_complexity", "entropy_1d",
    "max_range_product",
];
const N_STATS: usize = 23;
/// 每 case 因子数：方法1(4×23) + 方法2(10) + 步骤7a(23) + 步骤7b(23) = 148
pub const FACS_PER_CASE: usize = 4 * N_STATS + 10 + N_STATS + N_STATS;
pub const N_FACTORS: usize = 5 * 3 * 2 * 5 * FACS_PER_CASE; // 22200

const M2_NAMES: &[&str] = &[
    "cov_abs_sum", "cov_raw_sum", "cov_pos_sum", "cov_pos_mean", "cov_std", "cov_abs_std",
    "cov_pos_std", "cov_skew", "cov_abs_skew", "cov_pos_skew",
];

// ============================ 通用工具 ============================
fn percentile_sorted(sorted: &[f64], p: f64) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return f64::NAN;
    }
    if n == 1 {
        return sorted[0];
    }
    let rank = p / 100.0 * (n - 1) as f64;
    let lo = rank.floor() as usize;
    let hi = (lo + 1).min(n - 1);
    let frac = rank - lo as f64;
    sorted[lo] + frac * (sorted[hi] - sorted[lo])
}

/// 总体偏度(ddof=0)，对齐 Python 原型 skew。std<eps 返回 0.0。
fn skew_pop(x: &[f64]) -> f64 {
    let valid: Vec<f64> = x.iter().filter(|v| v.is_finite()).copied().collect();
    if valid.len() < 2 {
        return f64::NAN;
    }
    let n = valid.len() as f64;
    let m = valid.iter().sum::<f64>() / n;
    let var = valid.iter().map(|v| (v - m).powi(2)).sum::<f64>() / n;
    let s = var.sqrt();
    if s < 1e-12 {
        return 0.0;
    }
    valid.iter().map(|v| ((v - m) / s).powi(3)).sum::<f64>() / n
}

/// 总体标准差(ddof=0)，对齐 numpy .std()。
fn std_pop(x: &[f64]) -> f64 {
    if x.is_empty() {
        return f64::NAN;
    }
    let n = x.len() as f64;
    let m = x.iter().sum::<f64>() / n;
    (x.iter().map(|v| (v - m).powi(2)).sum::<f64>() / n).sqrt()
}

/// core (n×k) → Pearson 相关矩阵 (n×n)，逐行 z-score 后点积/k（ddof 无关）。
fn corr_matrix_nxn(core: &[f64], n: usize, k: usize) -> Vec<f64> {
    let mut z = Array2::<f64>::zeros((n, k));
    for i in 0..n {
        let row = &core[i * k..(i + 1) * k];
        let m = row.iter().sum::<f64>() / k as f64;
        let var = row.iter().map(|v| (v - m).powi(2)).sum::<f64>() / k as f64;
        let s = var.sqrt();
        let denom = if s > 1e-12 { s } else { 1.0 };
        for j in 0..k {
            z[(i, j)] = (row[j] - m) / denom;
        }
    }
    let corr = z.dot(&z.t()) / (k as f64);
    let mut out = vec![0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            out[i * n + j] = corr[(i, j)];
        }
    }
    out
}

/// 单列序列 → get_features_factors_rust_full 降维(23个统计)。
fn reduce_single(seq: &[f64]) -> Vec<f32> {
    let data: Vec<f32> = seq.iter().map(|&v| v as f32).collect();
    let arr = Array2::from_shape_vec((seq.len(), 1), data).unwrap();
    let (vals, _) = get_features_factors_rust_full(&arr.view(), &[], true);
    vals
}

fn ep1_match(ep1: &str, vol: f64, flag: i32, q40: f64) -> bool {
    match ep1 {
        "main_buy" => flag == 66,
        "main_sell" => flag == 83,
        "small" => vol < q40,
        "small_buy" => vol < q40 && flag == 66,
        "small_sell" => vol < q40 && flag == 83,
        _ => false,
    }
}

fn ep2_range(ep2: &str) -> (f64, f64) {
    match ep2 {
        "afternoon" => (41401.0, 48420.0),
        "tail30" => (46620.0, 48420.0),
        _ => (0.0, f64::INFINITY), // allday
    }
}

// ============================ 步骤4-7：单 case 因子 ============================
/// 每段成交量前20的股票索引：lead[j*100+s] = 第s段第j大成交量的股票。
fn top20_per_col(core: &[f64], n: usize) -> Vec<usize> {
    let mut lead = vec![0usize; 20 * 100];
    for s in 0..100 {
        let mut col: Vec<(f64, usize)> = (0..n).map(|i| (core[i * 100 + s], i)).collect();
        col.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));
        for j in 0..20.min(n) {
            lead[j * 100 + s] = col[j].1;
        }
    }
    lead
}

/// 每段热点一致度：20只领衔股 core 序列相关矩阵上三角正元素之和。
fn consistency(core: &[f64], lead: &[usize]) -> Vec<f64> {
    (0..100)
        .map(|s| {
            let mut m = vec![0f64; 20 * 100];
            for j in 0..20 {
                let st = lead[j * 100 + s];
                for t in 0..100 {
                    m[j * 100 + t] = core[st * 100 + t];
                }
            }
            let c = corr_matrix_nxn(&m, 20, 100);
            let mut sum = 0f64;
            for a in 0..20 {
                for b in (a + 1)..20 {
                    let v = c[a * 20 + b];
                    if v > 0.0 {
                        sum += v;
                    }
                }
            }
            sum
        })
        .collect()
}

/// 按热点一致度选段（返回升序段索引）。
fn select_segments(consist: &[f64], ep4: &str) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..100).collect();
    idx.sort_by(|&a, &b| consist[b].partial_cmp(&consist[a]).unwrap_or(Ordering::Equal));
    let sel: Vec<usize> = match ep4 {
        "hi50" => idx[0..50].to_vec(),
        "hi30" => idx[0..30].to_vec(),
        "lo50" => idx[50..100].to_vec(),
        "lo30" => idx[70..100].to_vec(),
        _ => (0..100).collect(), // all100
    };
    let mut s = sel;
    s.sort();
    s
}

/// 单 case 全市场因子：core(n×100) + ep4 → n×148 因子向量。
fn case_factors(core: &[f64], n: usize, ep4: &str) -> Vec<f32> {
    let lead = top20_per_col(core, n);
    let consist = consistency(core, &lead);
    let sel = select_segments(&consist, ep4);
    let k = sel.len();
    // core_sel [n×k]
    let mut core_sel = vec![0f64; n * k];
    for i in 0..n {
        for (si, &s) in sel.iter().enumerate() {
            core_sel[i * k + si] = core[i * 100 + s];
        }
    }
    let corr = corr_matrix_nxn(&core_sel, n, k); // [n×n]

    let mut out = vec![0f32; n * FACS_PER_CASE];
    out.par_chunks_mut(FACS_PER_CASE)
        .enumerate()
        .for_each(|(i, chunk)| {
            // 步骤5: M_i [k×20], mi[s*20+j] = corr[i, lead[j,sel[s]]]
            let mut mi = vec![0f64; k * 20];
            for s in 0..k {
                for j in 0..20 {
                    mi[s * 20 + j] = corr[i * n + lead[j * 100 + sel[s]]];
                }
            }
            let mut pos = 0usize;
            // 方法1: 行 mean/std/skew/top5
            let mut rmean = vec![0f64; k];
            let mut rstd = vec![0f64; k];
            let mut rskew = vec![0f64; k];
            let mut rtop5 = vec![0f64; k];
            for s in 0..k {
                let row = &mi[s * 20..s * 20 + 20];
                rmean[s] = row.iter().sum::<f64>() / 20.0;
                rstd[s] = std_pop(row);
                rskew[s] = skew_pop(row);
                let mut rs = row.to_vec();
                rs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
                rtop5[s] = rs[15..20].iter().sum::<f64>() / 5.0;
            }
            for seq in [&rmean, &rstd, &rskew, &rtop5] {
                for x in reduce_single(seq) {
                    chunk[pos] = x;
                    pos += 1;
                }
            }
            // 方法2: 20列协方差(ddof=1)上三角10统计
            let mut colmean = [0f64; 20];
            for j in 0..20 {
                let mut acc = 0f64;
                for r in 0..k {
                    acc += mi[r * 20 + j];
                }
                colmean[j] = acc / k as f64;
            }
            let mut cov = [0f64; 400];
            for a in 0..20 {
                for b in 0..20 {
                    let mut acc = 0f64;
                    for r in 0..k {
                        acc += (mi[r * 20 + a] - colmean[a]) * (mi[r * 20 + b] - colmean[b]);
                    }
                    cov[a * 20 + b] = acc / (k - 1) as f64;
                }
            }
            let mut up = Vec::with_capacity(190);
            for a in 0..20 {
                for b in (a + 1)..20 {
                    up.push(cov[a * 20 + b]);
                }
            }
            let posv: Vec<f64> = up.iter().filter(|v| **v > 0.0).copied().collect();
            let absu: Vec<f64> = up.iter().map(|v| v.abs()).collect();
            let m2 = [
                absu.iter().sum::<f64>(),
                up.iter().sum::<f64>(),
                posv.iter().sum::<f64>(),
                if posv.is_empty() { 0.0 } else { posv.iter().sum::<f64>() / posv.len() as f64 },
                std_pop(&up),
                std_pop(&absu),
                if posv.is_empty() { 0.0 } else { std_pop(&posv) },
                skew_pop(&up),
                skew_pop(&absu),
                if posv.is_empty() { 0.0 } else { skew_pop(&posv) },
            ];
            for x in m2 {
                chunk[pos] = x as f32;
                pos += 1;
            }
            // 步骤7a: core_sel 行降维
            for x in reduce_single(&core_sel[i * k..(i + 1) * k]) {
                chunk[pos] = x;
                pos += 1;
            }
            // 步骤7b: corr 行(自相关置NaN)降维
            let mut cr = vec![0f64; n];
            for j in 0..n {
                cr[j] = corr[i * n + j];
            }
            cr[i] = f64::NAN;
            for x in reduce_single(&cr) {
                chunk[pos] = x;
                pos += 1;
            }
            debug_assert_eq!(pos, FACS_PER_CASE);
        });
    out
}

// ============================ 步骤1-3 + 笛卡尔全集 ============================
/// 纯计算核心：扁平化成交数据(times/cidx/vols/flags) → 全 150 case 因子 → (codes, vals)。
/// vals 按 [stock0:22200, stock1:22200, ...] 排布（pipeline fan-out 约定）。
fn compute_from_flat(
    times: &[f64],
    cidx: &[u32],
    vols: &[f64],
    flags: &[i32],
    n_stocks: usize,
    codes: &[String],
) -> (Vec<String>, Vec<f32>) {
    let n = times.len();
    // 步骤1: 主排序(按 time, 并列按 cidx 打破——确定次级键), EP2 子集为其子序列(仍有序)
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        times[a]
            .partial_cmp(&times[b])
            .unwrap_or(Ordering::Equal)
            .then(cidx[a].cmp(&cidx[b]))
    });

    // EP1 small 系列用的"全市场"单笔成交体量40%分位(全局, 不按时段切分)
    let mut all_vols = vols.to_vec();
    all_vols.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    let q40 = percentile_sorted(&all_vols, 40.0);

    let mut case_facs: Vec<Vec<f32>> = Vec::with_capacity(150);
    for &ep2 in EP2S {
        let (lo, hi) = ep2_range(ep2);
        let sub: Vec<usize> = order
            .iter()
            .copied()
            .filter(|&i| {
                let s = times[i].rem_euclid(86400.0);
                s >= lo && s <= hi
            })
            .collect();
        for &ep1 in EP1S {
            let esub: Vec<usize> = sub
                .iter()
                .copied()
                .filter(|&i| ep1_match(ep1, vols[i], flags[i], q40))
                .collect();
            // 步骤2-3: core [n×100]
            let (core, nonempty) = if esub.is_empty() {
                (vec![0f64; n_stocks * 100], false)
            } else {
                let mut cum = vec![0f64; esub.len()];
                let mut acc = 0f64;
                for (t, &i) in esub.iter().enumerate() {
                    acc += vols[i];
                    cum[t] = acc;
                }
                let total = acc;
                let target: Vec<f64> = (1..100).map(|p| total * p as f64 / 100.0).collect();
                let mut c = vec![0f64; n_stocks * 100];
                for (pos, &i) in esub.iter().enumerate() {
                    let seg = target.partition_point(|&t| t <= cum[pos]).min(99);
                    let ci = cidx[i] as usize;
                    c[ci * 100 + seg] += vols[i];
                }
                (c, true)
            };
            for &ep3 in EP3S {
                let core_t = if ep3 == "ratio" {
                    let mut r = core.clone();
                    for i in 0..n_stocks {
                        let rowsum: f64 = r[i * 100..(i + 1) * 100].iter().sum();
                        if rowsum > 1e-12 {
                            for j in 0..100 {
                                r[i * 100 + j] /= rowsum;
                            }
                        }
                    }
                    r
                } else {
                    core.clone()
                };
                for &ep4 in EP4S {
                    let facs = if nonempty {
                        case_factors(&core_t, n_stocks, ep4)
                    } else {
                        vec![f32::NAN; n_stocks * FACS_PER_CASE]
                    };
                    case_facs.push(facs);
                }
            }
        }
    }
    // 按 stock-major 输出
    let mut out = Vec::with_capacity(n_stocks * N_FACTORS);
    for i in 0..n_stocks {
        for cf in &case_facs {
            out.extend_from_slice(&cf[i * FACS_PER_CASE..(i + 1) * FACS_PER_CASE]);
        }
    }
    (codes.to_vec(), out)
}

// ============================ v1 / v2 入口 ============================
fn list_codes(date: i64) -> Vec<String> {
    let dir = format!("/ssd_data/stock/{date}/transaction");
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

/// v1 读盘入口：rayon 并行读全市场 → 扁平化 → 核心。
pub fn compute_vsld_full(date: i64) -> std::io::Result<(Vec<String>, Vec<f32>)> {
    let codes = list_codes(date);
    let trades: Vec<Option<Vec<TradeRecord>>> = codes
        .par_iter()
        .map(|c| read_trade_fast_inner(c, date, false, true, usize::MAX).ok())
        .collect();
    let mut times = Vec::new();
    let mut cidx = Vec::new();
    let mut vols = Vec::new();
    let mut flags = Vec::new();
    for (i, t) in trades.iter().enumerate() {
        if let Some(tr) = t {
            for r in tr {
                times.push(r.time_us as f64 / 1e6); // 精确秒
                cidx.push(i as u32);
                vols.push(r.volume as f64);
                flags.push(r.flag);
            }
        }
    }
    Ok(compute_from_flat(&times, &cidx, &vols, &flags, codes.len(), &codes))
}

/// v2 传数据入口：预加载 per-stock 逐笔 → 核心（用于样例验证/无磁盘环境）。
pub fn compute_vsld_from_trades(
    codes: &[String],
    trades_per_code: Vec<Option<Vec<TradeRecord>>>,
) -> (Vec<String>, Vec<f32>) {
    let n = codes.len();
    let mut times = Vec::new();
    let mut cidx = Vec::new();
    let mut vols = Vec::new();
    let mut flags = Vec::new();
    for (i, t) in trades_per_code.iter().enumerate() {
        if let Some(tr) = t {
            for r in tr {
                times.push(r.time_sec as f64);
                cidx.push(i as u32);
                vols.push(r.volume as f64);
                flags.push(r.flag);
            }
        }
    }
    compute_from_flat(&times, &cidx, &vols, &flags, n, codes)
}

// ============================ 因子名 ============================
pub fn vsld_names() -> Vec<String> {
    let mut names = Vec::with_capacity(N_FACTORS);
    for ep2 in EP2S {
        for ep1 in EP1S {
            for ep3 in EP3S {
                for ep4 in EP4S {
                    let prefix = format!("{ep1}_{ep2}_{ep3}_{ep4}");
                    for tag in ["rmean", "rstd", "rskew", "rtop5"] {
                        for s in STAT_SUFFIXES {
                            names.push(format!("{prefix}_{tag}_{s}"));
                        }
                    }
                    for m in M2_NAMES {
                        names.push(format!("{prefix}_{m}"));
                    }
                    for s in STAT_SUFFIXES {
                        names.push(format!("{prefix}_core_{s}"));
                    }
                    for s in STAT_SUFFIXES {
                        names.push(format!("{prefix}_corrall_{s}"));
                    }
                }
            }
        }
    }
    names
}

// ============================ PyO3 包装 ============================
#[pyfunction]
pub fn py_vsld(py: Python<'_>, date: i64) -> PyResult<(Vec<String>, Vec<f32>)> {
    compute_vsld_full(date).map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("{e:?}")))
}

#[pyfunction]
pub fn py_vsld_names() -> Vec<String> {
    vsld_names()
}

#[pyfunction]
pub fn py_vsld_from_data(
    _py: Python<'_>,
    codes: Vec<String>,
    trade_arrays: Vec<PyReadonlyArray2<f64>>,
) -> PyResult<(Vec<String>, Vec<f32>)> {
    if codes.len() != trade_arrays.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "codes.len()={} != trade_arrays.len()={}",
            codes.len(),
            trade_arrays.len()
        )));
    }
    let trades_per_code: Vec<Option<Vec<TradeRecord>>> = trade_arrays
        .iter()
        .map(|arr| {
            let a = arr.as_array();
            if a.nrows() == 0 {
                return None;
            }
            let mut recs = Vec::with_capacity(a.nrows());
            for i in 0..a.nrows() {
                recs.push(TradeRecord {
                    time_sec: a[[i, 0]] as f32,
                    time_us: 0,
                    price: a[[i, 1]] as f32,
                    volume: a[[i, 2]] as f32,
                    turnover: a[[i, 3]] as f32,
                    flag: a[[i, 4]] as i32,
                    bid_order: a[[i, 5]] as i64,
                    ask_order: a[[i, 6]] as i64,
                });
            }
            Some(recs)
        })
        .collect();
    Ok(compute_vsld_from_trades(&codes, trades_per_code))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_names_count() {
        assert_eq!(vsld_names().len(), N_FACTORS);
        assert_eq!(N_FACTORS, 22200);
    }

    #[test]
    fn test_skew_basic() {
        assert!(skew_pop(&[1.0, 2.0, 3.0, 4.0, 5.0]).abs() < 1e-10); // 对称→0
        assert!((skew_pop(&[1.0, 1.0, 1.0, 100.0]) - 1.0).abs() < 0.5); // 右偏
    }

    #[test]
    fn test_select_segments() {
        let consist: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let s = select_segments(&consist, "hi30");
        assert_eq!(s.len(), 30);
        assert_eq!(s, (70..100).collect::<Vec<_>>()); // top30 = 最大30个 = 70..100
    }
}
