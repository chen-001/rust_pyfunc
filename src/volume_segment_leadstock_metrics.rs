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

/// 从一阶/二阶/三阶原点矩计算总体偏度(ddof=0)，等价于 skew_pop 但避免重新遍历。
fn skew_from_moments(sum: f64, sq: f64, cb: f64, n: f64) -> f64 {
    let mean = sum / n;
    let var = (sq / n - mean * mean).max(0.0);
    let std = var.sqrt();
    if std < 1e-12 {
        return 0.0;
    }
    let m3 = cb / n - 3.0 * mean * (sq / n) + 2.0 * mean.powi(3);
    m3 / std.powi(3)
}

/// core (n×k) → Pearson 相关矩阵 (n×n)，逐行 z-score 后点积/k（ddof 无关）。
fn corr_matrix_nxn(core: &[f64], n: usize, k: usize) -> Vec<f64> {
    // 用 Vec 构建 z（避免 Array2::zeros 初始化开销），再用 ArrayView2 零拷贝做矩阵乘
    let mut z = vec![0f64; n * k];
    for i in 0..n {
        let row = &core[i * k..(i + 1) * k];
        let m = row.iter().sum::<f64>() / k as f64;
        let var = row.iter().map(|v| (v - m).powi(2)).sum::<f64>() / k as f64;
        let s = var.sqrt();
        let denom = if s > 1e-12 { s } else { 1.0 };
        let zi = &mut z[i * k..(i + 1) * k];
        for j in 0..k {
            zi[j] = (row[j] - m) / denom;
        }
    }
    let zv = ndarray::ArrayView2::from_shape((n, k), &z).unwrap();
    let corr = zv.dot(&zv.t()) / (k as f64);
    // corr 是 C-contiguous Array2，as_slice + to_vec = 一次 memcpy（替代 n² 逐元素循环）
    corr.as_slice().unwrap().to_vec()
}

/// 单列序列 → get_features_factors_rust_full 降维(23个统计)。
fn reduce_single(seq: &[f64]) -> Vec<f32> {
    let data: Vec<f32> = seq.iter().map(|&v| v as f32).collect();
    let arr = Array2::from_shape_vec((seq.len(), 1), data).unwrap();
    let (vals, _) = get_features_factors_rust_full(&arr.view(), &[], true);
    vals
}

/// reduce_single 的零分配版本：复用外部 f32 buffer，避免每股重建 Vec + Array2。
fn reduce_single_buf(seq: &[f64], f32buf: &mut Vec<f32>) -> Vec<f32> {
    if f32buf.len() < seq.len() {
        f32buf.resize(seq.len(), 0.0);
    }
    for (i, &v) in seq.iter().enumerate() {
        f32buf[i] = v as f32;
    }
    let view = ndarray::ArrayView2::from_shape((seq.len(), 1), &f32buf[..seq.len()]).unwrap();
    let (vals, _) = get_features_factors_rust_full(&view, &[], true);
    vals
}

/// reduce_single 的高性能版本：对长序列(步骤7b n≈6000)只排序一次复用分位数，
/// 调用 features 公开的统计量函数计算非排序项。比 reduce_single_buf 省 8/10 次 sort。
/// 输出顺序严格对齐 get_features_factors_rust_full(with_threshold_counts=true, 单列)。
fn reduce_single_long(seq: &[f64], f32buf: &mut Vec<f32>) -> Vec<f32> {
    use crate::features::{
        binned_entropy_1d, col_kurt, col_mean, col_skew, col_std, corr_pair,
        curvature_1d, lz_complexity_1d, max_range_product_strict, quad_coef_1d, trend_1d,
    };
    let n = seq.len();
    if f32buf.len() < n {
        f32buf.resize(n, 0.0);
    }
    let col = &mut f32buf[..n];
    for (i, &v) in seq.iter().enumerate() {
        col[i] = v as f32;
    }
    // 过滤 NaN → valid，排序一次 → sorted
    let mut valid: Vec<f32> = col.iter().filter(|v| v.is_finite()).copied().collect();
    let nv = valid.len();
    if nv < 4 {
        // 不足 4 个有效值时 fallback 到标准路径（确保边界行为一致）
        let view = ndarray::ArrayView2::from_shape((n, 1), col).unwrap();
        return get_features_factors_rust_full(&view, &[], true).0;
    }
    let mut sorted = valid.clone();
    sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // 从 sorted 快速算分位数（线性插值，对齐 col_quantile）
    let quant = |q: f32| -> f32 {
        if sorted.len() == 1 {
            return sorted[0];
        }
        let pos = q * (sorted.len() - 1) as f32;
        let lo = pos.floor() as usize;
        let hi = (lo + 1).min(sorted.len() - 1);
        let frac = pos - lo as f32;
        sorted[lo] * (1.0 - frac) + sorted[hi] * frac
    };
    let median = if nv % 2 == 0 {
        (sorted[nv / 2 - 1] + sorted[nv / 2]) / 2.0
    } else {
        sorted[nv / 2]
    };
    let p5 = quant(0.05);
    let p10 = quant(0.10);
    let p25 = quant(0.25);
    let p75 = quant(0.75);
    let p90 = quant(0.90);
    let p95 = quant(0.95);
    let iqr = p75 - p25;

    // 非排序统计量（调用 features 公开函数，公式与 get_features_factors_rust_full 一致）
    let mean = col_mean(col);
    let std = col_std(col);
    let skew = col_skew(col);
    let kurt = col_kurt(col);
    let cv = std / (mean.abs() + 1e-8);
    // autocorr1 = corr(col, col_shifted_1)
    let autocorr1 = if n >= 2 {
        let shifted: Vec<f32> = std::iter::once(f32::NAN)
            .chain(col[..n - 1].iter().copied())
            .collect();
        corr_pair(col, &shifted)
    } else {
        f32::NAN
    };
    let trend = trend_1d(col);
    let curvature = curvature_1d(col);
    let quad_coef = quad_coef_1d(col);
    // period_diff / period_ratio（三等分）
    let split = n / 3;
    let (period_diff, period_ratio) = if split > 0 {
        let first_mean = col_mean(&col[..split]);
        let last_mean = col_mean(&col[n - split..]);
        (last_mean - first_mean, last_mean / (first_mean.abs() + 1e-8))
    } else {
        (f32::NAN, f32::NAN)
    };
    // mean_above_p90 / mean_below_p10
    let mean_above_p90 = {
        let (s, cnt) = valid.iter().fold((0.0f32, 0usize), |(s, c), &v| {
            if v > p90 { (s + v, c + 1) } else { (s, c) }
        });
        if cnt == 0 { 0.0 } else { s / cnt as f32 }
    };
    let mean_below_p10 = {
        let (s, cnt) = valid.iter().fold((0.0f32, 0usize), |(s, c), &v| {
            if v < p10 { (s + v, c + 1) } else { (s, c) }
        });
        if cnt == 0 { 0.0 } else { s / cnt as f32 }
    };
    // lz / entropy / max_range（调 features 函数，lz 内部有 1 次排序，其余不排序）
    let lz = lz_complexity_1d(col);
    let n_bins = (n as f32).log2().ceil() as usize + 1;
    let entropy = binned_entropy_1d(col, n_bins);
    let max_range = max_range_product_strict(col);

    // 按序输出 23 个统计量（严格对齐 get_features_factors_rust_full 单列输出序）
    vec![
        mean, median, std, skew, kurt,
        p5, p25, p75, p95, iqr, cv,
        autocorr1, autocorr1.abs(), trend, curvature, quad_coef,
        period_diff, period_ratio, mean_above_p90, mean_below_p10,
        lz, entropy, max_range,
    ]
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

/// EP2 时段范围（微秒整数），用于重排后的顺序遍历比较。
fn ep2_range_us(ep2: &str) -> (i64, i64) {
    match ep2 {
        "afternoon" => (41_401_000_000, 48_420_000_000),
        "tail30" => (46_620_000_000, 48_420_000_000),
        _ => (0, i64::MAX), // allday
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
/// 优化：corr 矩阵用 ndarray dot(matrixmultiply GEMM)，结果直接 as_slice 引用（不 to_vec）。
fn case_factors(core: &[f64], n: usize, ep4: &str) -> Vec<f32> {
    let lead = top20_per_col(core, n);
    let consist = consistency(core, &lead);
    let sel = select_segments(&consist, ep4);
    let k = sel.len();
    // core_sel [n×k]：原始值（步骤7a）+ z [n×k] f64：行标准化（用于 Pearson 相关矩阵）
    let mut core_sel = vec![0f64; n * k];
    let mut z = vec![0f64; n * k];
    for i in 0..n {
        let ci = &mut core_sel[i * k..(i + 1) * k];
        for (si, &s) in sel.iter().enumerate() {
            ci[si] = core[i * 100 + s];
        }
        let m = ci.iter().sum::<f64>() / k as f64;
        let var = ci.iter().map(|v| (v - m).powi(2)).sum::<f64>() / k as f64;
        let s = var.sqrt();
        let denom = if s > 1e-12 { s } else { 1.0 };
        let zi = &mut z[i * k..(i + 1) * k];
        for si in 0..k {
            zi[si] = (ci[si] - m) / denom;
        }
    }
    // corr = z · zᵀ / k（f64 matrixmultiply GEMM，as_slice 零拷贝引用避免 to_vec）
    let zv = ndarray::ArrayView2::from_shape((n, k), &z).unwrap();
    let corr_arr = zv.dot(&zv.t()) / (k as f64);
    let corr: &[f64] = corr_arr.as_slice().unwrap();

    // 预分配复用 buffer
    let mut mi = vec![0f64; k * 20];
    let mut rmean = vec![0f64; k];
    let mut rstd = vec![0f64; k];
    let mut rskew = vec![0f64; k];
    let mut rtop5 = vec![0f64; k];
    let mut cr = vec![0f64; n];
    let mut f32buf = vec![0f32; n];

    let mut out = vec![0f32; n * FACS_PER_CASE];
    for i in 0..n {
        let chunk = &mut out[i * FACS_PER_CASE..(i + 1) * FACS_PER_CASE];
        // 步骤5: mi[s*20+j] = corr[i, lead[j*100+sel[s]]]（查表 O(1)）
        for s in 0..k {
            for j in 0..20 {
                mi[s * 20 + j] = corr[i * n + lead[j * 100 + sel[s]]];
            }
        }
        let mut pos = 0usize;
        // 方法1: 行 mean/std/skew/top5
        for s in 0..k {
            let row = &mi[s * 20..s * 20 + 20];
            rmean[s] = row.iter().sum::<f64>() / 20.0;
            rstd[s] = std_pop(row);
            rskew[s] = skew_pop(row);
            let mut rs = [0f64; 20];
            rs.copy_from_slice(row);
            rs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
            rtop5[s] = rs[15..20].iter().sum::<f64>() / 5.0;
        }
        for seq in [&rmean, &rstd, &rskew, &rtop5] {
            for x in reduce_single_buf(seq, &mut f32buf) {
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
        let mut up_sum = 0f64;
        let mut up_abs_sum = 0f64;
        let mut pos_sum = 0f64;
        let mut pos_cnt = 0usize;
        let mut up_sq = 0f64;
        let mut up_abs_sq = 0f64;
        let mut pos_sq = 0f64;
        let mut up_cb = 0f64;
        let mut up_abs_cb = 0f64;
        let mut pos_cb = 0f64;
        for a in 0..20 {
            for b in (a + 1)..20 {
                let v = cov[a * 20 + b];
                let av = v.abs();
                up_sum += v;
                up_abs_sum += av;
                up_sq += v * v;
                up_abs_sq += av * av;
                up_cb += v * v * v;
                up_abs_cb += av * av * av;
                if v > 0.0 {
                    pos_sum += v;
                    pos_sq += v * v;
                    pos_cb += v * v * v;
                    pos_cnt += 1;
                }
            }
        }
        let n_up = 190f64;
        let pos_mean = if pos_cnt > 0 { pos_sum / pos_cnt as f64 } else { 0.0 };
        let up_std = ((up_sq / n_up) - (up_sum / n_up).powi(2)).max(0.0).sqrt();
        let up_abs_std = ((up_abs_sq / n_up) - (up_abs_sum / n_up).powi(2)).max(0.0).sqrt();
        let pos_std = if pos_cnt > 0 {
            ((pos_sq / pos_cnt as f64) - pos_mean.powi(2)).max(0.0).sqrt()
        } else {
            0.0
        };
        let m2 = [
            up_abs_sum,
            up_sum,
            pos_sum,
            pos_mean,
            up_std,
            up_abs_std,
            pos_std,
            skew_from_moments(up_sum, up_sq, up_cb, n_up),
            skew_from_moments(up_abs_sum, up_abs_sq, up_abs_cb, n_up),
            if pos_cnt > 0 { skew_from_moments(pos_sum, pos_sq, pos_cb, pos_cnt as f64) } else { 0.0 },
        ];
        for x in m2 {
            chunk[pos] = x as f32;
            pos += 1;
        }
        // 步骤7a: core_sel 行降维
        for x in reduce_single_buf(&core_sel[i * k..(i + 1) * k], &mut f32buf) {
            chunk[pos] = x;
            pos += 1;
        }
        // 步骤7b: corr 行(自相关置NaN)降维（长序列优化降维）
        cr.copy_from_slice(&corr[i * n..(i + 1) * n]);
        cr[i] = f64::NAN;
        for x in reduce_single_long(&cr, &mut f32buf) {
            chunk[pos] = x;
            pos += 1;
        }
        debug_assert_eq!(pos, FACS_PER_CASE);
    }
    out
}

// ============================ 步骤1-3 + 笛卡尔全集 ============================
/// 步骤2-3：对给定 (ep1, ep2)，从**已按时间排序的扁平数据**顺序遍历构造 core [n_stocks×100]。
/// 合并 EP1+EP2 过滤为单趟遍历（两次：第一趟算 total 算 targets，第二趟 cumsum+bincount）。
fn build_core_sorted(
    s_sod: &[i64],
    s_cidx: &[u32],
    s_vol: &[f64],
    s_flag: &[i32],
    n_stocks: usize,
    ep1: &str,
    ep2: &str,
    q40: f64,
) -> (Vec<f64>, bool) {
    let (lo, hi) = ep2_range_us(ep2);
    // 第一趟：计算过滤后总量
    let mut total = 0f64;
    for i in 0..s_sod.len() {
        if s_sod[i] >= lo && s_sod[i] <= hi && ep1_match(ep1, s_vol[i], s_flag[i], q40) {
            total += s_vol[i];
        }
    }
    if total <= 0.0 {
        return (vec![0f64; n_stocks * 100], false);
    }
    // targets: 1%..99% 累计阈值
    let targets: [f64; 99] = std::array::from_fn(|p| total * (p + 1) as f64 / 100.0);
    // 第二趟：cumsum + 按段累加成交量
    let mut core = vec![0f64; n_stocks * 100];
    let mut cum = 0f64;
    for i in 0..s_sod.len() {
        if s_sod[i] >= lo && s_sod[i] <= hi && ep1_match(ep1, s_vol[i], s_flag[i], q40) {
            cum += s_vol[i];
            let seg = targets.partition_point(|&t| t <= cum).min(99);
            core[s_cidx[i] as usize * 100 + seg] += s_vol[i];
        }
    }
    (core, true)
}

/// 纯计算核心：扁平化成交数据 → 排序+重排 → 全 150 case 因子 → (codes, vals)。
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
    // 步骤1: 按 (time, cidx) 排序（stable 并行排序，rayon 多线程分治加速）
    let mut order: Vec<usize> = (0..n).collect();
    order.par_sort_by(|&a, &b| {
        times[a]
            .partial_cmp(&times[b])
            .unwrap_or(Ordering::Equal)
            .then(cidx[a].cmp(&cidx[b]))
    });
    // 步骤1b: 按 order 重排数据为连续 SoA（消除后续 15 次 build_core 的间接访问 cache miss）
    let mut s_sod = vec![0i64; n];
    let mut s_cidx = vec![0u32; n];
    let mut s_vol = vec![0f64; n];
    let mut s_flag = vec![0i32; n];
    for (pos, &i) in order.iter().enumerate() {
        s_sod[pos] = (times[i].rem_euclid(86400.0) * 1e6) as i64;
        s_cidx[pos] = cidx[i];
        s_vol[pos] = vols[i];
        s_flag[pos] = flags[i];
    }

    // q40（全局单笔成交体量40%分位）
    let mut all_vols = s_vol.clone();
    all_vols.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    let q40 = percentile_sorted(&all_vols, 40.0);

    // build_core × 15（rayon 并行：15 个 (ep1,ep2) 组合独立，每个顺序遍历重排后数据）
    let combos: Vec<(&str, &str)> = EP2S
        .iter()
        .flat_map(|&ep2| EP1S.iter().map(move |&ep1| (ep1, ep2)))
        .collect();
    let cores: Vec<(Vec<f64>, bool)> = combos
        .par_iter()
        .map(|&(ep1, ep2)| {
            build_core_sorted(&s_sod, &s_cidx, &s_vol, &s_flag, n_stocks, ep1, ep2, q40)
        })
        .collect();

    // 150 个 case 规格: (core_idx, ep3, ep4)，顺序与 vsld_names 一致
    let specs: Vec<(usize, &str, &str)> = (0..3)
        .flat_map(|e2i| {
            (0..5).flat_map(move |e1i| {
                let ci = e2i * 5 + e1i;
                EP3S.iter().flat_map(move |&ep3| {
                    EP4S.iter().map(move |&ep4| (ci, ep3, ep4))
                })
            })
        })
        .collect();
    // case 级 rayon 并行（case 间独立），case 内逐股串行（避免嵌套 rayon）
    let case_facs: Vec<Vec<f32>> = specs
        .par_iter()
        .map(|&(ci, ep3, ep4)| {
            let (raw, nonempty) = &cores[ci];
            if !nonempty {
                return vec![f32::NAN; n_stocks * FACS_PER_CASE];
            }
            let core_t = if ep3 == "ratio" {
                let mut r = raw.clone();
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
                raw.clone()
            };
            case_factors(&core_t, n_stocks, ep4)
        })
        .collect();

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
                times.push(r.time_us as f64 / 1e6);
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
