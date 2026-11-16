//! get_features_factors 的纯 Rust 实现（run_factor_pipeline 优化方案 Phase 2）。
//!
//! 完全脱离 Python/pandas/pyo3，所有统计在 Rust 内一次性按列并行计算（rayon）。
//! 关闭 lyapunov（按决策：花 69% 时间只产生 2.3% 特征，性价比极低）。
//!
//! 输出顺序与 python/rust_pyfunc/trading_data_utils.py 的 get_features_factors
//! （默认参数，with_lyapunov_exponent=False）严格对齐，保证 names 匹配。
//!
//! 消除的浪费：
//! - pandas mean/median/std/skew/kurt/quantile（单线程）→ Rust + rayon 列并行
//! - 132 次 df.apply 的 pyo3 往返 → 一次性批量 Rust
//! - lyapunov 69% 计算时间 → 关闭
use ndarray::ArrayView2;
use pyo3::prelude::*;

pub use crate::features_core::{
    batch_corr_upper_col_major_f32, batch_corr_upper_row_major_f32, batch_corr_upper_row_major_f64,
    binned_entropy_1d, col_kurt, col_mean, col_median, col_quantile, col_skew, col_std,
    col_summary_f64, corr_pair, curvature_1d, lz_complexity_1d, max_range_product_strict,
    quad_coef_1d, trend_1d, ColSummaryF64, FeaturesScratch,
};

#[inline]
fn solve3(mut m: [[f64; 4]; 3]) -> Option<[f64; 3]> {
    for k in 0..3 {
        let mut piv = k;
        for i in (k + 1)..3 {
            if m[i][k].abs() > m[piv][k].abs() {
                piv = i;
            }
        }
        if m[piv][k].abs() < 1e-30 {
            return None;
        }
        if piv != k {
            m.swap(piv, k);
        }
        for i in 0..3 {
            if i != k {
                let f = m[i][k] / m[k][k];
                for j in k..4 {
                    m[i][j] -= f * m[k][j];
                }
            }
        }
    }
    Some([m[0][3] / m[0][0], m[1][3] / m[1][1], m[2][3] / m[2][2]])
}

// ============================================================================
// 主函数：get_features_factors_rust
// ============================================================================

/// get_features_factors 的纯 Rust 实现。
pub fn get_features_factors_rust(
    data: &ArrayView2<f32>,
    col_names: &[String],
) -> (Vec<f32>, Vec<String>) {
    get_features_factors_rust_full(data, col_names, false)
}

/// 只返回数值的高性能入口。用于同一列结构被大量重复调用的场景，避免反复构造因子名。
pub fn get_features_factors_rust_values_only(
    data: &ArrayView2<f32>,
    with_threshold_counts: bool,
) -> Vec<f32> {
    get_features_factors_rust_full(data, &[], with_threshold_counts).0
}

/// 带参数版本：with_threshold_counts 控制 mean_above_p90/mean_below_p10 是否输出。
pub fn get_features_factors_rust_full(
    data: &ArrayView2<f32>,
    col_names: &[String],
    with_threshold_counts: bool,
) -> (Vec<f32>, Vec<String>) {
    get_features_factors_rust_full_opts(data, col_names, with_threshold_counts, true)
}

/// 与 `get_features_factors_rust_full` 相同，但可选择**不输出 curvature / quad_coef**
/// （`with_curvature=false` 时退回 2026-07-13 commit f3c0cc0 之前的 19 统计量口径）。
pub fn get_features_factors_rust_full_opts(
    data: &ArrayView2<f32>,
    col_names: &[String],
    with_threshold_counts: bool,
    with_curvature: bool,
) -> (Vec<f32>, Vec<String>) {
    let (n_rows, n_cols) = data.dim();
    if n_rows == 0 || n_cols == 0 {
        return (Vec::new(), Vec::new());
    }

    let mut res = Vec::new();
    crate::features_core::with_tls_scratch(|sc| {
        if let Some(slice) = data.as_slice() {
            crate::features_core::transpose_row_to_col_f32(slice, n_rows, n_cols, &mut sc.col_major);
        } else {
            sc.col_major.clear();
            sc.col_major.resize(n_rows * n_cols, 0.0f32);
            for j in 0..n_cols {
                let col = data.column(j);
                let dst = &mut sc.col_major[j * n_rows..(j + 1) * n_rows];
                if let Some(cs) = col.as_slice() {
                    dst.copy_from_slice(cs);
                } else {
                    for (r, &v) in col.iter().enumerate() {
                        dst[r] = v;
                    }
                }
            }
        }
        let col_major = std::mem::take(&mut sc.col_major);
        crate::features_core::extract_features_col_major_f32(
            &col_major,
            n_rows,
            n_cols,
            with_threshold_counts,
            with_curvature,
            sc,
            &mut res,
        );
        sc.col_major = col_major;
    });

    let names = if col_names.len() == n_cols {
        crate::features_core::generate_feature_names(
            col_names,
            with_threshold_counts,
            with_curvature,
        )
    } else {
        Vec::new()
    };

    (res, names)
}

// ============================================================================
// multi_factor_capm 专用高速入口：21 项单列统计（无 corr、零拷贝、每列单次排序）
// ============================================================================
//
// 与 get_features_factors_rust_values_only(_, false) 的前 21×n_cols 个值**逐位一致**：
// - 不计算 corr 上三角（调用方只取前 21×n_cols 个值，corr 属于纯浪费）
// - 不拷贝矩阵：直接按 (col_stride) 步长读取列
// - 每列只排序一次：median / 6 个分位 / lz 阈值共用同一份排序结果
// - 所有中间缓冲（valid/sorted/lz/SAM/熵计数）跨列复用，消除每列 ~20 次堆分配
//
// 统计项与输出顺序严格对齐 push_group 顺序（with_threshold_counts=false）：
// mean, median, std, skew, kurt, p5, p25, p75, p95, iqr, cv,
// autocorr1, autocorr1_abs, trend, curvature, quad_coef,
// period_diff, period_ratio, lz_complexity, entropy_1d, max_range_product

/// 线程级复用缓冲：跨列 / 跨矩阵复用，消除每列堆分配。
#[derive(Default)]
pub struct StatsScratch {
    valid: Vec<f32>,        // 非 NaN 值（按列序）
    sorted: Vec<f32>,       // valid 的排序副本（median/分位共用）
    ent_counts: Vec<usize>, // 熵分箱计数（保留：旧入口仍使用）
    tbuf: Vec<f32>,         // 行主序 → 列主序转置缓冲（multi_factor per-stock 用）
    radix: Vec<f32>,        // 基数排序临时缓冲
}

/// f32 → 可排序 u32 键（非 NaN）：正数（含 +inf）映射到 [0x80000000, 0xFFFFFFFF]，
/// 负数（含 -inf）按位取反映射到 [0, 0x80000000)，保证数值序 = 键序。
#[inline]
fn f32_sort_key(v: f32) -> u32 {
    let b = v.to_bits();
    if b >> 31 == 0 {
        b | 0x8000_0000
    } else {
        !b
    }
}

/// LSD 基数排序（4 遍 × 8 位，稳定）：4740 元素 ~20μs，比 pdqsort 快 5 倍以上。
/// 等值元素顺序与 pdqsort 不同，但分位数/中位数只读位置上的**值**，等值可互换 → 结果逐位一致。
fn radix_sort_f32(values: &mut [f32], tmp: &mut Vec<f32>) {
    let n = values.len();
    if n < 2 {
        return;
    }
    tmp.clear();
    tmp.resize(n, 0.0);
    let mut cnt = [0usize; 256];
    for shift in [0u32, 8, 16, 24] {
        cnt.fill(0);
        for &v in values.iter() {
            cnt[((f32_sort_key(v) >> shift) & 0xff) as usize] += 1;
        }
        let mut acc = 0usize;
        for c in cnt.iter_mut() {
            let t = *c;
            *c = acc;
            acc += t;
        }
        for &v in values.iter() {
            let k = f32_sort_key(v);
            let b = ((k >> shift) & 0xff) as usize;
            tmp[cnt[b]] = v;
            cnt[b] += 1;
        }
        values.swap_with_slice(tmp);
    }
}

impl StatsScratch {
    pub fn new() -> Self {
        Self::default()
    }

    /// 行主序矩阵 [n_rows × n_cols]（元素 (i,j) 位于 src[i*n_cols+j]）的 21 列统计：
    /// 先转置为列主序（列连续）再计算 —— 直接按列步长读列的多次扫描会命中不同的
    /// cache line（步长 4×n_cols 字节），慢 3-5 倍。
    pub fn col_stats_21_row_major(
        &mut self,
        src: &[f32],
        n_rows: usize,
        n_cols: usize,
        out: &mut [f32],
    ) {
        self.tbuf.clear();
        self.tbuf.resize(n_rows * n_cols, 0.0);
        for i in 0..n_rows {
            let row = &src[i * n_cols..(i + 1) * n_cols];
            for c in 0..n_cols {
                self.tbuf[c * n_rows + i] = row[c];
            }
        }
        // 把 tbuf 临时取出（Vec 移动仅指针交换），避免 data 与 scratch 自引用冲突
        let tbuf = std::mem::take(&mut self.tbuf);
        col_stats_21_strided(&tbuf, n_rows, n_cols, 1, n_rows, self, out);
        self.tbuf = tbuf;
    }
}

/// 两列 Pearson 相关系数（按行序配对、双方非 NaN），与 corr_pair 逐位一致。
/// 列 a 元素 i 位于 data[ia + i*ra]，列 b 元素 i 位于 data[ib + i*rb]。
#[inline]
fn corr_pair_strided(
    data: &[f32],
    n_rows: usize,
    ia: usize,
    ra: usize,
    ib: usize,
    rb: usize,
) -> f32 {
    let (mut sum_a, mut sum_b, mut n) = (0.0f32, 0.0f32, 0usize);
    let (mut pa, mut pb) = (ia, ib);
    for _ in 0..n_rows {
        let a = data[pa];
        let b = data[pb];
        if !a.is_nan() && !b.is_nan() {
            sum_a += a;
            sum_b += b;
            n += 1;
        }
        pa += ra;
        pb += rb;
    }
    if n < 2 {
        return f32::NAN;
    }
    let ma = sum_a / n as f32;
    let mb = sum_b / n as f32;
    let (mut cov, mut var_a, mut var_b) = (0.0f32, 0.0f32, 0.0f32);
    let (mut pa, mut pb) = (ia, ib);
    for _ in 0..n_rows {
        let a = data[pa];
        let b = data[pb];
        if !a.is_nan() && !b.is_nan() {
            let da = a - ma;
            let db = b - mb;
            cov += da * db;
            var_a += da * da;
            var_b += db * db;
        }
        pa += ra;
        pb += rb;
    }
    if var_a == 0.0 || var_b == 0.0 {
        0.0
    } else {
        cov / (var_a.sqrt() * var_b.sqrt())
    }
}

/// multi_factor_capm 专用高速入口：与旧流水线 `get_features_factors_rust_values_only`
/// 输出取 `vals[..21*n_cols]` 后的前 21×n_cols 个值**逐位一致**。
///
/// 旧函数输出顺序为 [18 个统计组 × n_cols][corr 上三角][lz/entropy/max_range 3 组]，
/// 而调用方只取前 21×n_cols 个值 → 位置 [18n, 21n) 实为 **corr 上三角的前 3n 个值**
/// （并非 lz/entropy/max_range，它们排在 corr 之后）。本函数复刻该布局：
/// 输出 [stat 组外层 × col 内层] 的 18 组统计 + corr_upper[0..3*n_cols]。
///
/// 数据布局：元素 (行 i, 列 j) 位于 `data[i * row_stride + j * col_stride]`
/// （行主序矩阵传 row_stride=n_cols, col_stride=1；列主序传 row_stride=1, col_stride=N_BINS）。
/// 所有中间缓冲复用，每列只排序一次。
pub fn col_stats_21_strided(
    data: &[f32],
    n_rows: usize,
    n_cols: usize,
    row_stride: usize,
    col_stride: usize,
    scratch: &mut StatsScratch,
    out: &mut [f32],
) {
    let cols = n_cols;
    // 单列内联统计：按原实现逐位复刻（同一输入 → 同一输出）
    for j in 0..cols {
        let col_off = j * col_stride;
        let o = &mut out[j..];
        let o_stride = cols;

        // ---------- pass 1：过滤 + 基础累积（顺序与原实现一致） ----------
        scratch.valid.clear();
        let mut mean_sum = 0.0f32;
        let mut mean_n = 0usize;
        let split = n_rows / 3;
        let mut f_sum = 0.0f32; // period_diff/ratio 前半段
        let mut f_n = 0usize;
        let mut l_sum = 0.0f32; // 后半段
        let mut l_n = 0usize;
        let mut ac_sum_i = 0.0f32; // autocorr：corr_pair(c, shift(c)) 的均值分子
        let mut ac_sum_j = 0.0f32;
        let mut ac_n = 0usize;
        let mut ptr = col_off;
        let mut prev = f32::NAN;
        for i in 0..n_rows {
            let v = data[ptr];
            ptr += row_stride;
            if !v.is_nan() {
                scratch.valid.push(v);
                mean_sum += v;
                mean_n += 1;
                if i < split {
                    f_sum += v;
                    f_n += 1;
                }
                if i >= n_rows - split {
                    l_sum += v;
                    l_n += 1;
                }
            }
            if i > 0 && !v.is_nan() && !prev.is_nan() {
                ac_sum_i += v;
                ac_sum_j += prev;
                ac_n += 1;
            }
            prev = v;
        }

        let mean = if mean_n == 0 {
            f32::NAN
        } else {
            mean_sum / mean_n as f32
        };
        let first_mean = if f_n == 0 {
            f32::NAN
        } else {
            f_sum / f_n as f32
        };
        let last_mean = if l_n == 0 {
            f32::NAN
        } else {
            l_sum / l_n as f32
        };
        o[0 * o_stride] = mean;
        o[16 * o_stride] = last_mean - first_mean;
        o[17 * o_stride] = last_mean / (first_mean.abs() + 1e-8);

        // ---------- std / skew / kurt（valid，两遍：均值 → 中心矩） ----------
        let n = scratch.valid.len();
        if n < 2 {
            o[2 * o_stride] = f32::NAN;
        } else {
            let m = scratch.valid.iter().sum::<f32>() / n as f32;
            let var = scratch.valid.iter().map(|&x| (x - m).powi(2)).sum::<f32>() / (n - 1) as f32;
            o[2 * o_stride] = var.sqrt();
        }
        if n < 3 {
            o[3 * o_stride] = f32::NAN;
        } else {
            let m = scratch.valid.iter().sum::<f32>() / n as f32;
            let nf = n as f32;
            let (mut s2, mut s3) = (0.0f32, 0.0f32);
            for &x in scratch.valid.iter() {
                let d = x - m;
                s2 += d * d;
                s3 += d * d * d;
            }
            let k2 = s2 / (nf - 1.0);
            let k3 = nf * s3 / ((nf - 1.0) * (nf - 2.0));
            o[3 * o_stride] = if k2.abs() < 1e-30 {
                0.0
            } else {
                k3 / k2.powf(1.5)
            };
        }
        if n < 4 {
            o[4 * o_stride] = f32::NAN;
        } else {
            let m = scratch.valid.iter().sum::<f32>() / n as f32;
            let nf = n as f32;
            let (mut s2, mut s4) = (0.0f32, 0.0f32);
            for &x in scratch.valid.iter() {
                let d = x - m;
                let d2 = d * d;
                s2 += d2;
                s4 += d2 * d2;
            }
            let k2 = s2 / (nf - 1.0);
            let k4 = nf * ((nf + 1.0) * s4 - 3.0 * (nf - 1.0) * s2 * s2 / nf)
                / ((nf - 1.0) * (nf - 2.0) * (nf - 3.0));
            o[4 * o_stride] = if k2.abs() < 1e-30 {
                0.0
            } else {
                k4 / (k2 * k2)
            };
        }

        // ---------- 单次排序 → median + 6 分位 ----------
        scratch.sorted.clear();
        scratch.sorted.extend_from_slice(&scratch.valid);
        radix_sort_f32(&mut scratch.sorted, &mut scratch.radix);
        let sorted = &scratch.sorted;
        if n == 0 {
            o[1 * o_stride] = f32::NAN;
            o[5 * o_stride] = f32::NAN;
            o[6 * o_stride] = f32::NAN;
            o[7 * o_stride] = f32::NAN;
            o[8 * o_stride] = f32::NAN;
        } else {
            o[1 * o_stride] = if n % 2 == 0 {
                (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
            } else {
                sorted[n / 2]
            };
            // 与 col_quantile 完全一致的 pandas 线性插值
            let quantile = |q: f32| -> f32 {
                if n == 1 {
                    return sorted[0];
                }
                let pos = q * (n - 1) as f32;
                let lower = pos.floor() as usize;
                let upper = (lower + 1).min(n - 1);
                let frac = pos - lower as f32;
                sorted[lower] * (1.0 - frac) + sorted[upper] * frac
            };
            o[5 * o_stride] = quantile(0.05);
            o[6 * o_stride] = quantile(0.25);
            o[7 * o_stride] = quantile(0.75);
            o[8 * o_stride] = quantile(0.95);
        }
        let p75 = o[7 * o_stride];
        let p25 = o[6 * o_stride];
        o[9 * o_stride] = p75 - p25;
        let std_v = o[2 * o_stride];
        o[10 * o_stride] = std_v / (mean.abs() + 1e-8);

        // ---------- autocorr1（corr_pair(c, shift(c))，两遍） ----------
        if ac_n < 2 {
            o[11 * o_stride] = f32::NAN;
        } else {
            let mi = ac_sum_i / ac_n as f32;
            let mj = ac_sum_j / ac_n as f32;
            let (mut cov, mut var_i, mut var_j) = (0.0f32, 0.0f32, 0.0f32);
            let mut ptr = col_off;
            let mut prev = f32::NAN;
            for _ in 0..n_rows {
                let v = data[ptr];
                ptr += row_stride;
                if !v.is_nan() && !prev.is_nan() {
                    let di = v - mi;
                    let dj = prev - mj;
                    cov += di * dj;
                    var_i += di * di;
                    var_j += dj * dj;
                }
                prev = v;
            }
            if var_i == 0.0 || var_j == 0.0 {
                o[11 * o_stride] = 0.0;
            } else {
                o[11 * o_stride] = cov / (var_i.sqrt() * var_j.sqrt());
            }
        }
        let ac1 = o[11 * o_stride];
        o[12 * o_stride] = ac1.abs();

        // ---------- trend（f32，两遍） ----------
        let mut t_n = 0usize;
        let mut t_sum_x = 0.0f32;
        let mut t_sum_y = 0.0f32;
        {
            let mut ptr = col_off;
            for i in 0..n_rows {
                let v = data[ptr];
                ptr += row_stride;
                if !v.is_nan() {
                    t_n += 1;
                    t_sum_x += (i + 1) as f32;
                    t_sum_y += v;
                }
            }
        }
        if t_n < 2 {
            o[13 * o_stride] = 0.0;
        } else {
            let mx = t_sum_x / t_n as f32;
            let my = t_sum_y / t_n as f32;
            let (mut cov, mut var_x, mut var_y) = (0.0f32, 0.0f32, 0.0f32);
            let mut ptr = col_off;
            for i in 0..n_rows {
                let v = data[ptr];
                ptr += row_stride;
                if !v.is_nan() {
                    let dx = (i + 1) as f32 - mx;
                    let dy = v - my;
                    cov += dx * dy;
                    var_x += dx * dx;
                    var_y += dy * dy;
                }
            }
            o[13 * o_stride] = if var_x == 0.0 || var_y == 0.0 {
                0.0
            } else {
                cov / (var_x.sqrt() * var_y.sqrt())
            };
        }

        // ---------- curvature（f64，两遍） ----------
        {
            let mut n64 = 0usize;
            let mut sum_t = 0.0f64;
            let mut sum_y = 0.0f64;
            {
                let mut ptr = col_off;
                for i in 0..n_rows {
                    let v = data[ptr];
                    ptr += row_stride;
                    if !v.is_nan() {
                        n64 += 1;
                        sum_t += (i + 1) as f64;
                        sum_y += v as f64;
                    }
                }
            }
            if n64 < 3 {
                o[14 * o_stride] = 0.0;
            } else {
                let nf = n64 as f64;
                let mean_t = sum_t / nf;
                let mean_y = sum_y / nf;
                let mut mean_q = 0.0f64;
                {
                    let mut ptr = col_off;
                    for i in 0..n_rows {
                        let v = data[ptr];
                        ptr += row_stride;
                        if !v.is_nan() {
                            let t = (i + 1) as f64;
                            mean_q += (t - mean_t).powi(2);
                        }
                    }
                }
                mean_q /= nf;
                let (mut cov, mut var_y, mut var_q) = (0.0f64, 0.0f64, 0.0f64);
                {
                    let mut ptr = col_off;
                    for i in 0..n_rows {
                        let v = data[ptr];
                        ptr += row_stride;
                        if !v.is_nan() {
                            let t = (i + 1) as f64;
                            let q = (t - mean_t).powi(2);
                            let dy = v as f64 - mean_y;
                            let dq = q - mean_q;
                            cov += dy * dq;
                            var_y += dy * dy;
                            var_q += dq * dq;
                        }
                    }
                }
                o[14 * o_stride] = if var_y == 0.0 || var_q == 0.0 {
                    0.0
                } else {
                    (cov / (var_y.sqrt() * var_q.sqrt())) as f32
                };
            }
        }

        // ---------- quad_coef（f64，一遍） ----------
        {
            let mut n64 = 0usize;
            let mut sum_t = 0.0f64;
            let mut sum_y = 0.0f64;
            {
                let mut ptr = col_off;
                for i in 0..n_rows {
                    let v = data[ptr];
                    ptr += row_stride;
                    if !v.is_nan() {
                        n64 += 1;
                        sum_t += (i + 1) as f64;
                        sum_y += v as f64;
                    }
                }
            }
            if n64 < 4 {
                o[15 * o_stride] = 0.0;
            } else {
                let nf = n64 as f64;
                let mean_t = sum_t / nf;
                let mean_y = sum_y / nf;
                let (mut s_uu, mut s_uuu, mut s_uuuu, mut s_uy, mut s_uu_y, mut ss_tot) =
                    (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
                {
                    let mut ptr = col_off;
                    for i in 0..n_rows {
                        let v = data[ptr];
                        ptr += row_stride;
                        if !v.is_nan() {
                            let t = (i + 1) as f64;
                            let u = t - mean_t;
                            let dy = v as f64 - mean_y;
                            let uu = u * u;
                            s_uu += uu;
                            s_uuu += uu * u;
                            s_uuuu += uu * uu;
                            s_uy += u * v as f64;
                            s_uu_y += uu * v as f64;
                            ss_tot += dy * dy;
                        }
                    }
                }
                if ss_tot <= 0.0 || s_uu <= 0.0 {
                    o[15 * o_stride] = 0.0;
                } else {
                    let b1 = s_uy / s_uu;
                    let ss_res_lin = ss_tot - b1 * s_uy;
                    let m = [
                        [nf, 0.0, s_uu, nf * mean_y],
                        [0.0, s_uu, s_uuu, s_uy],
                        [s_uu, s_uuu, s_uuuu, s_uu_y],
                    ];
                    match solve3(m) {
                        Some([c2, b2, a2]) => {
                            let ssr_quad =
                                c2 * (nf * mean_y) + b2 * s_uy + a2 * s_uu_y - nf * mean_y * mean_y;
                            let ss_res_quad = ss_tot - ssr_quad;
                            let r2_lin = 1.0 - ss_res_lin / ss_tot;
                            let r2_quad = 1.0 - ss_res_quad / ss_tot;
                            let delta = (r2_quad - r2_lin).clamp(0.0, 1.0);
                            let sign = if a2 >= 0.0 { 1.0 } else { -1.0 };
                            o[15 * o_stride] = (sign * delta) as f32;
                        }
                        None => {
                            o[15 * o_stride] = 0.0;
                        }
                    }
                }
            }
        }
    }
    // corr 上三角前 3×n_cols 个值（每矩阵只算一次，列循环外）
    corr_prefix(data, n_rows, n_cols, row_stride, col_stride, out);
}

/// corr 上三角前 3×n_cols 个值。
/// 旧流水线取 vals[..21*n_cols]，其布局为 [18 统计组][corr_upper][lz/entropy/max]，
/// 故位置 [18n, 21n) 实为 corr_upper[0..3n]（lz/entropy/max 排在 corr 之后，未被取用）。
/// 逐位复刻 corr_upper 顺序：for i in 0..n { for j in i+1..n { corr(col_i, col_j) } }，
/// 只需前 3n 个 → 只涉及前 ~4 列。
fn corr_prefix(
    data: &[f32],
    n_rows: usize,
    n_cols: usize,
    row_stride: usize,
    col_stride: usize,
    out: &mut [f32],
) {
    let need = 3 * n_cols;
    let mut got = 0usize;
    let mut ci = 0usize;
    while got < need && ci + 1 < n_cols {
        let mut cj = ci + 1;
        while got < need && cj < n_cols {
            let v = corr_pair_strided(
                data,
                n_rows,
                ci * col_stride,
                row_stride,
                cj * col_stride,
                row_stride,
            );
            out[18 * n_cols + got] = v;
            got += 1;
            cj += 1;
        }
        ci += 1;
    }
}

// ============================================================================
// PyO3 验证桥接（仅供 Python 端一致性验证）
// ============================================================================

/// 验证用：接收 numpy (n,m) 矩阵 + 列名，调纯 Rust get_features_factors_rust_full(with_threshold_counts=true)（对齐 Python 默认）。
#[pyfunction]
pub fn verify_get_features_factors_rust(
    data: numpy::PyReadonlyArray2<f64>,
    col_names: Vec<String>,
) -> PyResult<(Vec<f64>, Vec<String>)> {
    let view = data.as_array();
    let view_f32 = view.mapv(|x| x as f32);
    let (vals, names) = get_features_factors_rust_full(&view_f32.view(), &col_names, true);
    let vals_f64: Vec<f64> = vals.iter().map(|&v| v as f64).collect();
    Ok((vals_f64, names))
}

#[cfg(test)]
mod stats21_tests {
    use super::*;
    use ndarray::{s, Array2};

    /// 随机矩阵（含 NaN/inf/常量列）上对比新高速入口与旧入口的前 21×n_cols 值，要求逐位一致。
    #[test]
    fn col_stats_21_matches_reference() {
        let mut rng = 0x1234_5678u64;
        let mut next = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        for n_rows in [1usize, 2, 3, 63, 64, 65, 100, 2000, 4740] {
            for n_cols in [1usize, 2, 5, 18, 26] {
                for trial in 0..6 {
                    let mut data = vec![0.0f32; n_rows * n_cols];
                    for v in data.iter_mut() {
                        let r = (next() % 1000) as f32 / 1000.0;
                        *v = match r {
                            x if x < 0.03 => f32::NAN,
                            x if x < 0.05 => f32::INFINITY,
                            x if x < 0.07 => f32::NEG_INFINITY,
                            x if x < 0.10 => 0.0,
                            x if x < 0.13 => 1.0,
                            _ => ((next() % 2000) as f32 - 1000.0) / 37.0,
                        };
                    }
                    if trial == 5 {
                        for v in data.iter_mut() {
                            *v = f32::NAN;
                        }
                    }
                    let mat = Array2::from_shape_vec((n_rows, n_cols), data.clone()).unwrap();
                    let vals = get_features_factors_rust_values_only(&mat.view(), false);
                    let mut scratch = StatsScratch::new();
                    let mut out = vec![0.0f32; 21 * n_cols];
                    col_stats_21_strided(&data, n_rows, n_cols, n_cols, 1, &mut scratch, &mut out);
                    let expect = &vals[..21 * n_cols];
                    for i in 0..21 * n_cols {
                        assert!(
                            out[i].to_bits() == expect[i].to_bits(),
                            "mismatch rows={n_rows} cols={n_cols} trial={trial} idx={i}: new={:?} old={:?}",
                            out[i],
                            expect[i]
                        );
                    }
                    let mut cm = vec![0.0f32; n_rows * n_cols];
                    for c in 0..n_cols {
                        for r in 0..n_rows {
                            cm[c * n_rows + r] = data[r * n_cols + c];
                        }
                    }
                    let mut out2 = vec![0.0f32; 21 * n_cols];
                    let mut scratch2 = StatsScratch::new();
                    col_stats_21_strided(&cm, n_rows, n_cols, 1, n_rows, &mut scratch2, &mut out2);
                    for i in 0..21 * n_cols {
                        assert!(
                            out2[i].to_bits() == expect[i].to_bits(),
                            "col-major mismatch rows={n_rows} cols={n_cols} trial={trial} idx={i}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn batch_corr_and_quantiles_match_scalar_reference() {
        let mut rng = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        for n_rows in [1usize, 2, 3, 4, 30, 95, 96, 97, 237] {
            for n_cols in [1usize, 2, 3, 15, 16, 17, 33, 78] {
                let mut data = vec![0.0f32; n_rows * n_cols];
                for c in 0..n_cols {
                    let has_nan_col = c % 7 == 3;
                    let const_col = c % 11 == 5;
                    let neg_col = c % 13 == 2;
                    for r in 0..n_rows {
                        let v = if const_col {
                            2.5f32
                        } else if has_nan_col && (next() % 10 == 0) {
                            f32::NAN
                        } else {
                            let raw = ((next() % 4000) as f32 - 2000.0) / 53.0;
                            if neg_col { -raw.abs() - 1.0 } else { raw }
                        };
                        data[r * n_cols + c] = v;
                    }
                }
                let n_pairs = n_cols * (n_cols - 1) / 2;
                let mut got = vec![0.0f32; n_pairs];
                batch_corr_upper_row_major_f32(&data, n_rows, n_cols, &mut got);

                let mut idx = 0usize;
                for i in 0..n_cols {
                    let ci: Vec<f32> = (0..n_rows).map(|r| data[r * n_cols + i]).collect();
                    // 同时检验 select_nth 分位数与全排序分位数逐位相等
                    let mut valid: Vec<f32> = ci.iter().copied().filter(|v| !v.is_nan()).collect();
                    valid.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
                    for q in [0.05f32, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95] {
                        let q_sel = col_quantile(&ci, q);
                        let q_ref = if valid.is_empty() {
                            f32::NAN
                        } else if valid.len() == 1 {
                            valid[0]
                        } else {
                            let pos = q * (valid.len() - 1) as f32;
                            let lo = pos.floor() as usize;
                            let hi = (lo + 1).min(valid.len() - 1);
                            let fr = pos - lo as f32;
                            valid[lo] * (1.0 - fr) + valid[hi] * fr
                        };
                        assert_eq!(q_sel.to_bits(), q_ref.to_bits(), "quantile mismatch q={q}");
                    }
                    for j in (i + 1)..n_cols {
                        let cj: Vec<f32> = (0..n_rows).map(|r| data[r * n_cols + j]).collect();
                        let ref_c = corr_pair(&ci, &cj);
                        assert_eq!(
                            got[idx].to_bits(),
                            ref_c.to_bits(),
                            "corr mismatch rows={n_rows} cols={n_cols} ({i},{j}): got={} ref={}",
                            got[idx],
                            ref_c
                        );
                        idx += 1;
                    }
                }

                // 验证行切片视图（如 min_mat.slice(s![r0..r1, ..])）与独立拷贝矩阵逐位相等
                if n_rows >= 20 {
                    let mat = Array2::from_shape_vec((n_rows, n_cols), data.clone()).unwrap();
                    let sub_view = mat.slice(s![5..n_rows - 3, ..]);
                    let sub_owned = sub_view.to_owned();
                    let cols: Vec<String> = (0..n_cols).map(|k| format!("c{k}")).collect();
                    for tc in [false, true] {
                        for cv in [false, true] {
                            let (v1, n1) = get_features_factors_rust_full_opts(&sub_view, &cols, tc, cv);
                            let (v2, n2) = get_features_factors_rust_full_opts(&sub_owned.view(), &cols, tc, cv);
                            assert_eq!(n1, n2);
                            assert_eq!(v1.len(), v2.len());
                            for k in 0..v1.len() {
                                assert_eq!(v1[k].to_bits(), v2[k].to_bits());
                            }
                        }
                    }
                }
            }
        }
    }
}
