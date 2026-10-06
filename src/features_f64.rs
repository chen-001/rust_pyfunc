//! get_features_factors 的**纯 f64** 版（src/features.rs 的孪生实现，2026-10-05 加）。
//!
//! 由来：hm0~hm92 那批「go 函数调 order_pair_metrics + get_features_factors」的系列，
//! 老代码的 get_features_factors 是 pandas/f64，而 src/features.rs 全程 f32。
//! 把**完全相同**的 f64 指标矩阵分别喂给两边，44 列上 1958 项里 244 项相对误差 > 1e-4、
//! 最大 18%（mean_above_p90 / lz_complexity / kurt 这一族）；只看入选因子也有 9 项超 1e-4。
//! 所以照 src/features.rs 逐条复制一份、把标量类型统一换成 f64，算法与列顺序完全不动。
//!
//! 用法：
//!     use crate::features_f64;
//!     let (vals, names) = features_f64::get_features_factors_rust_full(&matrix.view(), &cols, true);
//! Python 侧验证入口：rp.verify_get_features_factors_rust_f64(matrix, cols)。
//!
//! 注意：src/features.rs 那条 f32 通路**不要动**（hm94~hm110 的新框架系列依赖它）。
//!
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
use rayon::prelude::*;
use std::collections::HashMap;

// ============================================================================
// 可选开关（2026-10-06 加）
// ============================================================================

/// `get_features_factors` 的可选开关。
///
/// 原来这条 f64 通路只实现了 Python `get_features_factors` 的**默认参数**那一档，
/// 于是 hm079 / hm081 / hm082 / hm083 / hm080 / hm091 这几个系列全都卡住——
/// 它们分别要用 with_max_min、with_lyapunov_exponent、append_for_corr。
/// 与其每个系列各补一遍，不如在这里一次补齐。
///
/// 注意：Python 的 `get_features_factors` 里**没有** curvature / quad_coef 这两组，
/// 那是新库 features 通路自己加的，所以对账老库（0_76_9 / 0_76_11 / 0_76_12）时
/// 要把 `with_curvature` 关掉。
#[derive(Clone, Copy, Debug)]
pub struct FeatureOpts {
    /// mean_above_p90 / mean_below_p10 两组（Python 默认 True）。
    pub with_threshold_counts: bool,
    /// curvature / quad_coef 两组（新库特有，Python 没有；对账老库时置 false）。
    pub with_curvature: bool,
    /// max / min / range 三组，位置在 kurt 之后、分位数之前（Python 默认 False）。
    pub with_max_min: bool,
    /// lyapunov 一组，位置在 corr 块之后、lz_complexity 之前（Python 默认 True）。
    pub with_lyapunov_exponent: bool,
    /// lz_complexity / entropy_1d / max_range_product 三组，排在最后（Python 默认 True）。
    /// 关掉时这三组一列都不输出，之前的所有组位置不变（2026-10-06 加，纯开关）。
    pub with_complexity: bool,
}

impl FeatureOpts {
    /// 与原来那条 f64 通路完全一致的默认值（关 lyapunov、开 curvature、开 complexity）。
    pub const LEGACY: FeatureOpts = FeatureOpts {
        with_threshold_counts: true,
        with_curvature: true,
        with_max_min: false,
        with_lyapunov_exponent: false,
        with_complexity: true,
    };
}

impl Default for FeatureOpts {
    fn default() -> Self {
        FeatureOpts::LEGACY
    }
}

// ============================================================================
// 基础统计量（单列计算，对齐 pandas 行为含 NaN 处理）
// ============================================================================

/// 单列均值（跳过 NaN，空列返回 NaN）。对齐 pandas df.mean()。
#[inline]
pub(crate) fn col_mean(col: &[f64]) -> f64 {
    let (sum, n) = col.iter().fold((0.0f64, 0usize), |(s, c), &v| {
        if v.is_nan() {
            (s, c)
        } else {
            (s + v, c + 1)
        }
    });
    if n == 0 {
        f64::NAN
    } else {
        sum / n as f64
    }
}

/// 单列标准差（样本标准差 ddof=1，对齐 pandas df.std()）。空或单元素返回 NaN。
#[inline]
pub(crate) fn col_std(col: &[f64]) -> f64 {
    let valid: Vec<f64> = col.iter().filter(|&&v| !v.is_nan()).copied().collect();
    let n = valid.len();
    if n < 2 {
        return f64::NAN;
    }
    let mean = valid.iter().sum::<f64>() / n as f64;
    let var = valid.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
    var.sqrt()
}

/// 单列偏度（对齐 pandas df.skew()，基于 G1 偏度估计量，用 k-statistic）。
/// n<3 返回 NaN。公式：g1 = k3/k2^1.5，其中 k2=S2/(n-1)，k3=n*S3/((n-1)(n-2))。
#[inline]
pub(crate) fn col_skew(col: &[f64]) -> f64 {
    let valid: Vec<f64> = col.iter().filter(|&&v| !v.is_nan()).copied().collect();
    let n = valid.len();
    if n < 3 {
        return f64::NAN;
    }
    let mean = valid.iter().sum::<f64>() / n as f64;
    let nf = n as f64;
    let mut s2 = 0.0;
    let mut s3 = 0.0;
    for &x in &valid {
        let d = x - mean;
        s2 += d * d;
        s3 += d * d * d;
    }
    let k2 = s2 / (nf - 1.0);
    let k3 = nf * s3 / ((nf - 1.0) * (nf - 2.0));
    // pandas 对零方差列（常量列）返回 0.0 而非 NaN
    if k2.abs() < 1e-30 {
        return 0.0;
    }
    k3 / k2.powf(1.5)
}

/// 单列峰度（对齐 pandas df.kurt()，基于 G2 超额峰度，用 k-statistic）。
/// n<4 返回 NaN。公式：g2 = k4/k2^2，其中
///   k2 = S2/(n-1)
///   k4 = n*[(n+1)*S4 - 3*(n-1)*S2^2/n] / [(n-1)*(n-2)*(n-3)]
#[inline]
pub(crate) fn col_kurt(col: &[f64]) -> f64 {
    let valid: Vec<f64> = col.iter().filter(|&&v| !v.is_nan()).copied().collect();
    let n = valid.len();
    if n < 4 {
        return f64::NAN;
    }
    let mean = valid.iter().sum::<f64>() / n as f64;
    let nf = n as f64;
    let mut s2 = 0.0;
    let mut s4 = 0.0;
    for &x in &valid {
        let d = x - mean;
        let d2 = d * d;
        s2 += d2;
        s4 += d2 * d2;
    }
    let k2 = s2 / (nf - 1.0);
    let k4 = nf * ((nf + 1.0) * s4 - 3.0 * (nf - 1.0) * s2 * s2 / nf)
        / ((nf - 1.0) * (nf - 2.0) * (nf - 3.0));
    // pandas 对零方差列（常量列）返回 0.0 而非 NaN
    if k2.abs() < 1e-30 {
        return 0.0;
    }
    k4 / (k2 * k2)
}

/// 单列最大值（跳过 NaN，全 NaN 给 NaN）。对齐 pandas df.max()。
#[inline]
pub(crate) fn col_max(col: &[f64]) -> f64 {
    let mut m = f64::NEG_INFINITY;
    let mut seen = false;
    for &v in col {
        if v.is_nan() {
            continue;
        }
        seen = true;
        if v > m {
            m = v;
        }
    }
    if seen {
        m
    } else {
        f64::NAN
    }
}

/// 单列最小值（跳过 NaN，全 NaN 给 NaN）。对齐 pandas df.min()。
#[inline]
pub(crate) fn col_min(col: &[f64]) -> f64 {
    let mut m = f64::INFINITY;
    let mut seen = false;
    for &v in col {
        if v.is_nan() {
            continue;
        }
        seen = true;
        if v < m {
            m = v;
        }
    }
    if seen {
        m
    } else {
        f64::NAN
    }
}

/// 单列中位数（跳过 NaN）。
#[inline]
fn col_median(col: &[f64]) -> f64 {
    let mut valid: Vec<f64> = col.iter().filter(|&&v| !v.is_nan()).copied().collect();
    if valid.is_empty() {
        return f64::NAN;
    }
    valid.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = valid.len();
    if n % 2 == 0 {
        (valid[n / 2 - 1] + valid[n / 2]) / 2.0
    } else {
        valid[n / 2]
    }
}

/// 单列分位数（线性插值，对齐 pandas df.quantile()）。
/// q in [0,1]。空列返回 NaN。
#[inline]
fn col_quantile(col: &[f64], q: f64) -> f64 {
    let mut valid: Vec<f64> = col.iter().filter(|&&v| !v.is_nan()).copied().collect();
    if valid.is_empty() {
        return f64::NAN;
    }
    valid.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = valid.len();
    if n == 1 {
        return valid[0];
    }
    // pandas 线性插值: pos = q*(n-1)
    let pos = q * (n - 1) as f64;
    let lower = pos.floor() as usize;
    let upper = (lower + 1).min(n - 1);
    let frac = pos - lower as f64;
    valid[lower] * (1.0 - frac) + valid[upper] * frac
}

// ============================================================================
// 复杂统计量（复制自现有 Rust 模块的纯算法）
// ============================================================================

/// 计算一维序列与 [1,2,...,n] 的 Pearson 相关系数（趋势）。
/// 对齐 time_series/trend_mod.rs 的 calculate_trend_1d（过滤 NaN）。
#[inline]
pub(crate) fn trend_1d(col: &[f64]) -> f64 {
    // 过滤 NaN，保留有效值及其原始索引（1-based）
    let pairs: Vec<(usize, f64)> = col
        .iter()
        .enumerate()
        .filter(|(_, &v)| !v.is_nan())
        .map(|(i, &v)| (i + 1, v))
        .collect();
    let n = pairs.len();
    if n < 2 {
        return 0.0;
    }
    let mean_x: f64 = pairs.iter().map(|(x, _)| *x as f64).sum::<f64>() / n as f64;
    let mean_y: f64 = pairs.iter().map(|(_, y)| *y).sum::<f64>() / n as f64;
    let (mut cov, mut var_x, mut var_y) = (0.0, 0.0, 0.0);
    for (x, y) in &pairs {
        let dx = *x as f64 - mean_x;
        let dy = *y - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }
    if var_x == 0.0 || var_y == 0.0 {
        return 0.0;
    }
    cov / (var_x.sqrt() * var_y.sqrt())
}

/// 计算一维序列与二次时间基 (t−t̄)² 的 Pearson 相关系数（二阶趋势/弯曲方向）。
/// 与 trend_1d 对称：trend 看"一阶线性方向"，curvature 看"二阶抛物线方向"。
/// 无量纲 [-1,1]：>0 凹(U型，中间低两头高)，<0 凸(倒U型，中间高两头低)，≈0 无二阶弯曲。
/// 对等距索引，t 与 (t−t̄)² 正交，故 curvature 与 trend 解耦、不冗余。
#[inline]
pub(crate) fn curvature_1d(col: &[f64]) -> f64 {
    let pairs: Vec<(f64, f64)> = col
        .iter()
        .enumerate()
        .filter(|(_, &v)| !v.is_nan())
        .map(|(i, &v)| ((i + 1) as f64, v as f64))
        .collect();
    let n = pairs.len();
    if n < 3 {
        return 0.0;
    }
    let nf = n as f64;
    let mean_t: f64 = pairs.iter().map(|(t, _)| t).sum::<f64>() / nf;
    let mean_y: f64 = pairs.iter().map(|(_, y)| y).sum::<f64>() / nf;
    // 二次基 q = (t−mean_t)²，先算其均值（Pearson 要求双侧去中心）
    let mean_q: f64 = pairs.iter().map(|(t, _)| (t - mean_t).powi(2)).sum::<f64>() / nf;
    let (mut cov, mut var_y, mut var_q) = (0.0f64, 0.0f64, 0.0f64);
    for (t, y) in &pairs {
        let q = (t - mean_t).powi(2);
        let dy = y - mean_y;
        let dq = q - mean_q;
        cov += dy * dq;
        var_y += dy * dy;
        var_q += dq * dq;
    }
    if var_y == 0.0 || var_q == 0.0 {
        return 0.0;
    }
    (cov / (var_y.sqrt() * var_q.sqrt())) as f64
}

/// 解 3×3 线性方程组（部分主元高斯消元）。增广矩阵 m=[[a0 a1 a2 b];...]，奇异返回 None。
#[inline]
fn solve3(mut m: [[f64; 4]; 3]) -> Option<[f64; 3]> {
    for k in 0..3 {
        // 部分主元：选第 k 列绝对值最大行
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
        // 全消元（消除其它行第 k 列）
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

/// 计算去趋势后的归一化二次拟合贡献（二阶弯曲强度）。
/// 对去中心化时间 u=t−t̄ 做线性拟合(R²_lin) 与 二次拟合(R²_quad)，返回
/// sign(a₂)·(R²_quad − R²_lin)，其中 a₂ 为二次项系数。
/// 与 curvature 互补：curvature 看"二阶方向一致度"，quad_coef 看"二次项相对线性项的边际解释方差"。
/// 无量纲 [-1,1]：0=纯线性无弯曲，|·|→1=方差几乎全由二次项解释；正=凹(U型)，负=凸(倒U型)。
#[inline]
pub(crate) fn quad_coef_1d(col: &[f64]) -> f64 {
    let pts: Vec<(f64, f64)> = col
        .iter()
        .enumerate()
        .filter(|(_, &v)| !v.is_nan())
        .map(|(i, &v)| ((i + 1) as f64, v as f64))
        .collect();
    let n = pts.len();
    if n < 4 {
        return 0.0;
    }
    let nf = n as f64;
    let mean_t: f64 = pts.iter().map(|(t, _)| t).sum::<f64>() / nf;
    let mean_y: f64 = pts.iter().map(|(_, y)| y).sum::<f64>() / nf;
    let (mut s_uu, mut s_uuu, mut s_uuuu, mut s_uy, mut s_uu_y, mut ss_tot) =
        (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for (t, y) in &pts {
        let u = t - mean_t;
        let dy = y - mean_y;
        let uu = u * u;
        s_uu += uu;
        s_uuu += uu * u;
        s_uuuu += uu * uu;
        s_uy += u * y;
        s_uu_y += uu * y;
        ss_tot += dy * dy;
    }
    if ss_tot <= 0.0 || s_uu <= 0.0 {
        return 0.0;
    }
    // 线性拟合（去中心化时间 Σu=0 → 截距=ȳ）：b1 = Σuy/Σu²
    let b1 = s_uy / s_uu;
    let ss_res_lin = ss_tot - b1 * s_uy; // SS_res = SS_tot − b·Σuy
                                         // 二次拟合正规方程（去中心化 u，Σu=0）：[[nf,0,s_uu],[0,s_uu,s_uuu],[s_uu,s_uuu,s_uuuu]]·[c,b,a]=[Σy,Σuy,Σu²y]
    let m = [
        [nf, 0.0, s_uu, nf * mean_y],
        [0.0, s_uu, s_uuu, s_uy],
        [s_uu, s_uuu, s_uuuu, s_uu_y],
    ];
    let sol = match solve3(m) {
        Some(s) => s,
        None => return 0.0,
    };
    let (c2, b2, a2) = (sol[0], sol[1], sol[2]);
    // 回归解释方差：SSR = c·Σy + b·Σuy + a·Σu²y − n·ȳ²（Σy=nf·ȳ）
    let ssr_quad = c2 * (nf * mean_y) + b2 * s_uy + a2 * s_uu_y - nf * mean_y * mean_y;
    let ss_res_quad = ss_tot - ssr_quad;
    let r2_lin = 1.0 - ss_res_lin / ss_tot;
    let r2_quad = 1.0 - ss_res_quad / ss_tot;
    let delta = (r2_quad - r2_lin).clamp(0.0, 1.0);
    let sign = if a2 >= 0.0 { 1.0 } else { -1.0 };
    (sign * delta) as f64
}

/// 两列的 Pearson 相关系数（共同有效位置）。对齐 pandas corr。
#[inline]
pub(crate) fn corr_pair(col_i: &[f64], col_j: &[f64]) -> f64 {
    let pairs: Vec<(f64, f64)> = col_i
        .iter()
        .zip(col_j.iter())
        .filter(|(&a, &b)| !a.is_nan() && !b.is_nan())
        .map(|(&a, &b)| (a, b))
        .collect();
    let n = pairs.len();
    if n < 2 {
        return f64::NAN;
    }
    let mean_i: f64 = pairs.iter().map(|(i, _)| i).sum::<f64>() / n as f64;
    let mean_j: f64 = pairs.iter().map(|(_, j)| j).sum::<f64>() / n as f64;
    let (mut cov, mut var_i, mut var_j) = (0.0, 0.0, 0.0);
    for (i, j) in &pairs {
        let di = i - mean_i;
        let dj = j - mean_j;
        cov += di * dj;
        var_i += di * di;
        var_j += dj * dj;
    }
    if var_i == 0.0 || var_j == 0.0 {
        return 0.0; // 常数列与其他列无协变，相关系数定义为 0（避免 NaN 污染降维）
    }
    cov / (var_i.sqrt() * var_j.sqrt())
}

/// LZ 复杂度（精确复制自 lz_complexity.rs）。分位数离散化 [0.33, 0.66] + 归一化。
pub(crate) fn lz_complexity_1d(col: &[f64]) -> f64 {
    // 过滤 NaN/inf，只用有效值（与 col_mean 等一致，避免单点 NaN 污染整列复杂度）
    let valid: Vec<f64> = col.iter().copied().filter(|v| v.is_finite()).collect();
    let n = valid.len();
    if n == 0 {
        return 0.0;
    }

    // 分位数离散化（quantiles=[0.33, 0.66]），精确复制 discretize_sequence
    let mut sorted = valid.clone();
    sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    let sn = sorted.len();
    let quantiles = [0.33f64, 0.66];
    let mut thresholds = Vec::with_capacity(quantiles.len());
    for &q in &quantiles {
        let idx = ((sn - 1) as f64 * q) as usize;
        thresholds.push(sorted[idx]);
    }

    let mut discrete: Vec<u8> = Vec::with_capacity(n);
    for &val in &valid {
        let symbol = thresholds
            .iter()
            .enumerate()
            .find(|(_, &threshold)| val <= threshold)
            .map(|(i, _)| i as u8)
            .unwrap_or(thresholds.len() as u8);
        discrete.push(symbol + 1); // 从1开始编号
    }

    // 计算复杂度
    let complexity = lz_calculate_complexity(&discrete);

    // 唯一符号数
    let k_eff = {
        let mut s = std::collections::HashSet::new();
        for &d in &discrete {
            s.insert(d);
        }
        s.len() as f64
    };
    if n <= 1 {
        return 0.0;
    }
    if k_eff < (quantiles.len() as f64 + 1.0) {
        return f64::NAN;
    }

    let log_n_base_k = (n as f64).ln() / k_eff.ln();
    complexity as f64 * log_n_base_k / n as f64
}

/// LZ 复杂度核心调度（复制自 lz_complexity.rs 的 calculate_lz_complexity）。
fn lz_calculate_complexity(seq: &[u8]) -> usize {
    let n = seq.len();
    if n == 0 {
        return 0;
    }
    if n <= 64 {
        return lz_complexity_simple(seq);
    }
    lz_complexity_suffix_automaton(seq)
}

/// LZ 复杂度核心调度（复用后缀自动机状态缓冲，消除每次 2n 状态堆分配）。
fn lz_calculate_complexity_reused(seq: &[u8], sam_buf: &mut Vec<SamState>) -> usize {
    let n = seq.len();
    if n == 0 {
        return 0;
    }
    if n <= 64 {
        return lz_complexity_simple(seq);
    }
    lz_complexity_suffix_automaton_in(seq, sam_buf)
}

/// LZ 复杂度暴力版（精确复制自 lz_complexity.rs:663）。
fn lz_complexity_simple(seq: &[u8]) -> usize {
    let n = seq.len();
    if n == 0 {
        return 0;
    }
    let mut complexity = 0;
    let mut i = 0;
    while i < n {
        let mut j = i + 1;
        while j <= n {
            let sub_len = j - i;
            let search_end = j - 1;
            if search_end < sub_len {
                break;
            }
            let mut found = false;
            for start_pos in 0..=(search_end - sub_len) {
                if seq[start_pos..start_pos + sub_len] == seq[i..j] {
                    found = true;
                    break;
                }
            }
            if found && j < n {
                j += 1;
            } else {
                break;
            }
        }
        complexity += 1;
        i = j;
    }
    complexity
}

// ============ 后缀自动机（精确复制自 lz_complexity.rs:165-304）============
#[derive(Clone)]
struct SamState {
    len: usize,
    link: Option<usize>,
    // LZ 复杂度只用 3 个符号(1,2,3)，直接索引数组替代 Vec 堆分配
    // transitions[sym] = target state，usize::MAX 表示无 transition
    transitions: [usize; 4],
}

impl SamState {
    fn new(len: usize) -> Self {
        Self {
            len,
            link: None,
            transitions: [usize::MAX; 4],
        }
    }
    #[inline]
    fn get(&self, c: u8) -> Option<usize> {
        let t = self.transitions[c as usize];
        if t == usize::MAX {
            None
        } else {
            Some(t)
        }
    }
    #[inline]
    fn set(&mut self, c: u8, state: usize) {
        self.transitions[c as usize] = state;
    }
}

struct SuffixAutomaton {
    states: Vec<SamState>,
    last: usize,
}

impl SuffixAutomaton {
    fn with_capacity(capacity: usize) -> Self {
        let mut states = Vec::with_capacity(capacity.max(2));
        states.push(SamState::new(0));
        Self { states, last: 0 }
    }
    #[inline]
    fn next_state(&self, state: usize, c: u8) -> Option<usize> {
        self.states[state].get(c)
    }
    fn extend(&mut self, c: u8) {
        let cur_index = self.states.len();
        let cur_len = self.states[self.last].len + 1;
        self.states.push(SamState::new(cur_len));
        let mut p_opt = Some(self.last);
        while let Some(p_idx) = p_opt {
            if self.states[p_idx].get(c).is_some() {
                break;
            }
            self.states[p_idx].set(c, cur_index);
            p_opt = self.states[p_idx].link;
        }
        if let Some(p_idx) = p_opt {
            let q_idx = self.states[p_idx].get(c).expect("transition must exist");
            if self.states[p_idx].len + 1 == self.states[q_idx].len {
                self.states[cur_index].link = Some(q_idx);
            } else {
                let clone_idx = self.states.len();
                let mut cloned_state = self.states[q_idx].clone();
                cloned_state.len = self.states[p_idx].len + 1;
                self.states.push(cloned_state);
                self.states[q_idx].link = Some(clone_idx);
                self.states[cur_index].link = Some(clone_idx);
                let mut current_opt = Some(p_idx);
                while let Some(current) = current_opt {
                    if self.states[current].get(c) == Some(q_idx) {
                        self.states[current].set(c, clone_idx);
                        current_opt = self.states[current].link;
                    } else {
                        break;
                    }
                }
            }
        } else {
            self.states[cur_index].link = Some(0);
        }
        self.last = cur_index;
    }
}

/// LZ 复杂度后缀自动机版（精确复制自 lz_complexity.rs:267）。
fn lz_complexity_suffix_automaton(seq: &[u8]) -> usize {
    lz_complexity_suffix_automaton_in(seq, &mut Vec::new())
}

/// 同上，但复用调用方提供的状态缓冲（clear 后原地重建，容量保持）。
fn lz_complexity_suffix_automaton_in(seq: &[u8], buf: &mut Vec<SamState>) -> usize {
    let n = seq.len();
    if n == 0 {
        return 0;
    }
    buf.clear();
    buf.reserve(2 * n);
    buf.push(SamState::new(0));
    let mut sam = SuffixAutomaton {
        states: std::mem::take(buf),
        last: 0,
    };
    let mut complexity = 0;
    let mut i = 0;
    while i < n {
        let mut state = 0;
        let mut j = i;
        while j < n {
            if let Some(next_state) = sam.next_state(state, seq[j]) {
                state = next_state;
                j += 1;
            } else {
                break;
            }
        }
        if j == n {
            complexity += 1;
            break;
        }
        complexity += 1;
        let phrase_end = j + 1;
        for &symbol in &seq[i..phrase_end] {
            sam.extend(symbol);
        }
        i = phrase_end;
    }
    *buf = sam.states;
    complexity
}

/// 分箱熵（复制自 entropy_analysis.rs）。等宽分箱 + Shannon 熵。
///
/// ！！2026-10-06 改：**NaN 要单独算一个箱**，分母用含 NaN 的整列长度。
/// 旧框架那张特征表里 entropy_1d 走的是 pyfunction
/// `entropy_analysis::calculate_binned_entropy_1d`，它内部的 `equal_width_binning`
/// 把每个 NaN 打成第 `n_bins` 号箱（在 [0, n_bins) 之外多出来的一个箱），
/// `total` 取的是 `bin_indices.len()`，也就是**原列长度**。
/// 原来是「先滤掉 NaN、分母用有效值个数」，只要某列含 NaN 就会差出百分比级。
/// 实测 hm081 的 direction_ratio_bid_mean（947 行里 10 个 NaN、n_bins=11）：
/// 滤 NaN 给 2.2811105875613937，NaN 单独成箱给 2.3155805442501585，
/// 后者与旧框架逐位相同；差 1.5%，直接越过 1e-4 验收线。
/// 注：src/features.rs 的 f32 同名函数**不动**（hm94~hm110 那批依赖它），
/// volume_segment_leadstock_metrics 用的是 f32 那份，不受这里影响。
pub(crate) fn binned_entropy_1d(col: &[f64], n_bins: usize) -> f64 {
    if col.is_empty() || n_bins == 0 {
        return 0.0;
    }
    let valid: Vec<f64> = col.iter().filter(|&&v| !v.is_nan()).copied().collect();
    if valid.len() < 2 {
        // 旧实现：没有有效值、或只剩 1 个有效值、或取不到宽度时，
        // equal_width_binning 给每个位置都打 0 号箱 → 熵 0。
        return 0.0;
    }

    let min_val = valid.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max_val = valid.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    if (max_val - min_val).abs() < f64::EPSILON {
        return 0.0;
    }

    let bin_width = (max_val - min_val) / n_bins as f64;
    // 用 Vec 替代 HashMap 保证遍历顺序确定（bin idx ∈ [0, n_bins] 可枚举），
    // 避免 HashMap 随机迭代序导致浮点求和非结合性 → 确定性 bug。
    // 多出来的最后一格（下标 n_bins）就是 NaN 箱。
    let mut counts: Vec<usize> = vec![0; n_bins + 1];
    for &v in col.iter() {
        if v.is_nan() {
            counts[n_bins] += 1;
            continue;
        }
        let mut idx = ((v - min_val) / bin_width).floor() as usize;
        if idx >= n_bins {
            idx = n_bins - 1;
        }
        counts[idx] += 1;
    }
    let total = col.len() as f64;
    counts
        .iter()
        .map(|&c| {
            let p = c as f64 / total;
            if p > 0.0 {
                -p * p.ln()
            } else {
                0.0
            }
        })
        .sum()
}

/// 最大范围积的严格对齐版（复制自 sequence/mod.rs 的双指针逻辑）。
/// 返回 abs(idx1 - idx2)/n，其中 idx1,idx2 是 find_max_range_product 返回的两个索引。
/// 注意：Python 的 _calc_max_range_product 取的是 abs(索引1-索引2)/n，不是值差。
pub(crate) fn max_range_product_strict(col: &[f64]) -> f64 {
    let valid: Vec<f64> = col.iter().filter(|&&v| !v.is_nan()).copied().collect();
    let n_total = col.len(); // Python 用 series.shape[0]（含NaN的原长度）
    let n = valid.len();
    if n < 2 || n_total == 0 {
        return 0.0;
    }

    let mut max_product = f64::NEG_INFINITY;
    let mut result = (0i64, 0i64);
    let mut left = 0;
    let mut right = n - 1;
    while left < right {
        let product = valid[left].min(valid[right]) * (right - left) as f64;
        if product > max_product {
            max_product = product;
            result = (left as i64, right as i64);
        }
        if valid[left] < valid[right] {
            left += 1;
        } else {
            right -= 1;
        }
    }
    for i in 0..n - 1 {
        let product = valid[i].min(valid[i + 1]) * 1.0;
        if product > max_product {
            max_product = product;
            result = (i as i64, (i + 1) as i64);
        }
    }
    let (i, j) = result;
    (i - j).abs() as f64 / n_total as f64
}

// ============================================================================
// 主函数：get_features_factors_rust
// ============================================================================

/// get_features_factors 的纯 Rust 实现。
///
/// 与 Python 版的默认参数对齐（with_corr=True, with_percentiles=True,
/// with_lag_autocorr=1, with_threshold_counts=False（便捷版默认；调 _full 可显式开启），
/// with_period_compare=True, with_complexity=True），但 **关闭 lyapunov**（with_lyapunov_exponent=False）。
///
/// 输入 data: (n_rows, n_cols) 的 f64 矩阵（order_pair_metrics 的输出）。
/// 输出 (vals, names)，vals 为展平的特征向量，names 与之等长。
///
/// 输出顺序与 Python get_features_factors(with_lyapunov_exponent=False) 严格一致。
pub fn get_features_factors_rust(
    data: &ArrayView2<f64>,
    col_names: &[String],
) -> (Vec<f64>, Vec<String>) {
    get_features_factors_rust_full(data, col_names, false)
}

/// 只返回数值的高性能入口。用于同一列结构被大量重复调用的场景，避免反复构造因子名。
pub fn get_features_factors_rust_values_only(
    data: &ArrayView2<f64>,
    with_threshold_counts: bool,
) -> Vec<f64> {
    get_features_factors_rust_full(data, &[], with_threshold_counts).0
}

/// 带参数版本：with_threshold_counts 控制 mean_above_p90/mean_below_p10 是否输出。
pub fn get_features_factors_rust_full(
    data: &ArrayView2<f64>,
    col_names: &[String],
    with_threshold_counts: bool,
) -> (Vec<f64>, Vec<String>) {
    get_features_factors_rust_full_opts(data, col_names, with_threshold_counts, true)
}

/// 与 `get_features_factors_rust_full` 相同，但可选择**不输出 curvature / quad_coef**
/// （`with_curvature=false` 时退回 2026-07-13 commit f3c0cc0 之前的 19 统计量口径）。
///
/// 用途：`observable_order` 系列因子在 0701c 版本（50400 列）之后才加入这两个二阶趋势指标；
/// 需要在**旧 store 上续算**（追加新日期）时，必须产出与旧列定义完全一致的值，
/// 否则列名/列序对不上，写入会被拒绝或整体错位。
pub fn get_features_factors_rust_full_opts(
    data: &ArrayView2<f64>,
    col_names: &[String],
    with_threshold_counts: bool,
    with_curvature: bool,
) -> (Vec<f64>, Vec<String>) {
    get_features_factors_rust_full_ex(
        data,
        col_names,
        FeatureOpts {
            with_threshold_counts,
            with_curvature,
            with_max_min: false,
            with_lyapunov_exponent: false,
            with_complexity: true,
        },
        None,
    )
}

/// 带全部开关的入口（2026-10-06 加）。
///
/// `append_for_corr` 对应 Python 的 `append_for_corr=<DataFrame>`：
/// **只有相关块**用 `concat([data, append])` 的列集，统计段仍然只算 data 的列，
/// 相关块的名字是 `{行名}_corr_{列名}`。传 None 就是原来那一档。
///
/// 对齐的是 python/rust_pyfunc/trading_data_utils.py::get_features_factors 的取值顺序：
///   mean/median/std/skew/kurt
///   [max/min/range]            with_max_min
///   p5/p25/p75/p95/iqr/cv
///   autocorr1/autocorr1_abs
///   trend
///   [curvature/quad_coef]      新库特有，Python 没有
///   period_diff/period_ratio
///   [mean_above_p90/mean_below_p10]  with_threshold_counts
///   corr 上三角（列集含 append_for_corr）
///   [lyapunov]                 with_lyapunov_exponent
///   lz_complexity/entropy_1d/max_range_product
pub fn get_features_factors_rust_full_ex(
    data: &ArrayView2<f64>,
    col_names: &[String],
    opts: FeatureOpts,
    append_for_corr: Option<(&ArrayView2<f64>, &[String])>,
) -> (Vec<f64>, Vec<String>) {
    let (n_rows, n_cols) = data.dim();

    // 空输入防护：0 行或 0 列时返回全 NaN（与 pandas 对空 DataFrame 的行为一致）
    if n_rows == 0 || n_cols == 0 {
        let mut res = Vec::new();
        let mut names = Vec::new();
        // 无法生成有意义的 names，返回空（调用方 pipeline 会用 nan_vec 兜底）
        return (res, names);
    }

    let cols: Vec<Vec<f64>> = (0..n_cols).map(|j| data.column(j).to_vec()).collect();

    // 按列计算所有单列统计量。
    // 注：用串行 iter 而非 par_iter——避免与外层 run_factor_pipeline 的自定义
    // rayon 池冲突（嵌套 rayon 池会导致线程互相等待）。外层已有 n_jobs=200 任务级并行。
    let col_stats: Vec<ColStats> = cols
        .iter()
        .map(|c| {
            let mean = col_mean(c);
            let median = col_median(c);
            let std = col_std(c);
            let skew = col_skew(c);
            let kurt = col_kurt(c);
            let p5 = col_quantile(c, 0.05);
            let p25 = col_quantile(c, 0.25);
            let p75 = col_quantile(c, 0.75);
            let p95 = col_quantile(c, 0.95);
            let iqr = p75 - p25;
            let cv = std / (mean.abs() + 1e-8);
            let p90 = col_quantile(c, 0.90);
            let p10 = col_quantile(c, 0.10);
            // mean_above_p90 / mean_below_p10
            let mean_above_p90 = {
                let (s, n) = c.iter().fold((0.0f64, 0usize), |(s, n), &v| {
                    if !v.is_nan() && v > p90 {
                        (s + v, n + 1)
                    } else {
                        (s, n)
                    }
                });
                if n == 0 {
                    0.0
                } else {
                    s / n as f64
                }
            };
            let mean_below_p10 = {
                let (s, n) = c.iter().fold((0.0f64, 0usize), |(s, n), &v| {
                    if !v.is_nan() && v < p10 {
                        (s + v, n + 1)
                    } else {
                        (s, n)
                    }
                });
                if n == 0 {
                    0.0
                } else {
                    s / n as f64
                }
            };
            // period_compare
            let split = n_rows / 3;
            let first_mean = if split > 0 {
                col_mean(&c[..split])
            } else {
                f64::NAN
            };
            let last_mean = if split > 0 {
                col_mean(&c[n_rows - split..])
            } else {
                f64::NAN
            };
            let period_diff = last_mean - first_mean;
            let period_ratio = last_mean / (first_mean.abs() + 1e-8);
            // trend
            let trend = trend_1d(c);
            // curvature / quad_coef（二阶趋势）
            let curvature = curvature_1d(c);
            let quad_coef = quad_coef_1d(c);
            // autocorr1（lag=1）
            let autocorr1 = if n_rows >= 2 {
                let shifted: Vec<f64> = std::iter::once(f64::NAN)
                    .chain(c[..n_rows - 1].iter().copied())
                    .collect();
                corr_pair(c, &shifted)
            } else {
                f64::NAN
            };
            // lz / entropy / max_range
            let lz = lz_complexity_1d(c);
            let n_bins = (n_rows as f64).log2().ceil() as usize + 1;
            let entropy = binned_entropy_1d(c, n_bins);
            let max_range = max_range_product_strict(c);
            // max / min / range（pandas df.max()/df.min()/相减，只在 with_max_min 时用到）
            let max = col_max(c);
            let min = col_min(c);
            let range = max - min;
            // lyapunov（Python 的 df.apply(_calc_lyapunov)：算不出来给 NaN）
            let lyapunov = if opts.with_lyapunov_exponent {
                crate::time_series::lyapunov::lyapunov_exponent_inner(c)
            } else {
                f64::NAN
            };

            ColStats {
                mean,
                median,
                std,
                skew,
                kurt,
                max,
                min,
                range,
                p5,
                p25,
                p75,
                p95,
                iqr,
                cv,
                autocorr1,
                trend,
                curvature,
                quad_coef,
                period_diff,
                period_ratio,
                mean_above_p90,
                mean_below_p10,
                lz,
                entropy,
                max_range,
                lyapunov,
            }
        })
        .collect();

    // corr 矩阵上三角（并行计算所有对）。
    // with_append：列集 = data 的列 + append_for_corr 的列（Python 里是 concat([df, append], axis=1)）。
    let mut corr_cols: Vec<Vec<f64>> = cols.clone();
    let mut corr_names_all: Vec<String> = col_names.to_vec();
    if let Some((ap, ap_names)) = append_for_corr {
        let (ap_rows, ap_cols) = ap.dim();
        if ap_rows == n_rows && ap_cols > 0 {
            for j in 0..ap_cols {
                corr_cols.push(ap.column(j).to_vec());
                let nm = ap_names
                    .get(j)
                    .cloned()
                    .unwrap_or_else(|| format!("append{}", j));
                corr_names_all.push(nm);
            }
        }
    }
    let n_corr_cols = corr_cols.len();
    let corr_upper: Vec<f64> = if n_corr_cols >= 2 {
        let pairs: Vec<(usize, usize)> = (0..n_corr_cols)
            .flat_map(|i| (i + 1..n_corr_cols).map(move |j| (i, j)))
            .collect();
        pairs
            .iter()
            .map(|&(i, j)| corr_pair(&corr_cols[i], &corr_cols[j]))
            .collect()
    } else {
        vec![]
    };

    // ============ 按顺序拼接结果（对齐 Python get_features_factors）============
    let mut res: Vec<f64> = Vec::new();
    let mut names: Vec<String> = Vec::new();

    // 1. mean/median/std/skew/kurt
    push_group(
        &mut res,
        &mut names,
        &col_stats.iter().map(|s| s.mean).collect::<Vec<_>>(),
        "mean",
        col_names,
    );
    push_group(
        &mut res,
        &mut names,
        &col_stats.iter().map(|s| s.median).collect::<Vec<_>>(),
        "median",
        col_names,
    );
    push_group(
        &mut res,
        &mut names,
        &col_stats.iter().map(|s| s.std).collect::<Vec<_>>(),
        "std",
        col_names,
    );
    push_group(
        &mut res,
        &mut names,
        &col_stats.iter().map(|s| s.skew).collect::<Vec<_>>(),
        "skew",
        col_names,
    );
    push_group(
        &mut res,
        &mut names,
        &col_stats.iter().map(|s| s.kurt).collect::<Vec<_>>(),
        "kurt",
        col_names,
    );
    // 2. max/min/range（Python 的 with_max_min，位置在 kurt 之后、分位数之前）
    if opts.with_max_min {
        push_group(
            &mut res,
            &mut names,
            &col_stats.iter().map(|s| s.max).collect::<Vec<_>>(),
            "max",
            col_names,
        );
        push_group(
            &mut res,
            &mut names,
            &col_stats.iter().map(|s| s.min).collect::<Vec<_>>(),
            "min",
            col_names,
        );
        push_group(
            &mut res,
            &mut names,
            &col_stats.iter().map(|s| s.range).collect::<Vec<_>>(),
            "range",
            col_names,
        );
    }
    // 3. p5/p25/p75/p95/iqr/cv
    push_group(
        &mut res,
        &mut names,
        &col_stats.iter().map(|s| s.p5).collect::<Vec<_>>(),
        "p5",
        col_names,
    );
    push_group(
        &mut res,
        &mut names,
        &col_stats.iter().map(|s| s.p25).collect::<Vec<_>>(),
        "p25",
        col_names,
    );
    push_group(
        &mut res,
        &mut names,
        &col_stats.iter().map(|s| s.p75).collect::<Vec<_>>(),
        "p75",
        col_names,
    );
    push_group(
        &mut res,
        &mut names,
        &col_stats.iter().map(|s| s.p95).collect::<Vec<_>>(),
        "p95",
        col_names,
    );
    push_group(
        &mut res,
        &mut names,
        &col_stats.iter().map(|s| s.iqr).collect::<Vec<_>>(),
        "iqr",
        col_names,
    );
    push_group(
        &mut res,
        &mut names,
        &col_stats.iter().map(|s| s.cv).collect::<Vec<_>>(),
        "cv",
        col_names,
    );
    // 3. autocorr1 / autocorr1_abs
    push_group(
        &mut res,
        &mut names,
        &col_stats.iter().map(|s| s.autocorr1).collect::<Vec<_>>(),
        "autocorr1",
        col_names,
    );
    push_group(
        &mut res,
        &mut names,
        &col_stats
            .iter()
            .map(|s| s.autocorr1.abs())
            .collect::<Vec<_>>(),
        "autocorr1_abs",
        col_names,
    );
    // 4. trend
    push_group(
        &mut res,
        &mut names,
        &col_stats.iter().map(|s| s.trend).collect::<Vec<_>>(),
        "trend",
        col_names,
    );
    // 4b. curvature / quad_coef（二阶趋势）——with_curvature=false 时跳过（旧 19 统计量口径）
    if opts.with_curvature {
        push_group(
            &mut res,
            &mut names,
            &col_stats.iter().map(|s| s.curvature).collect::<Vec<_>>(),
            "curvature",
            col_names,
        );
        push_group(
            &mut res,
            &mut names,
            &col_stats.iter().map(|s| s.quad_coef).collect::<Vec<_>>(),
            "quad_coef",
            col_names,
        );
    }
    // 5. period_diff / period_ratio
    push_group(
        &mut res,
        &mut names,
        &col_stats.iter().map(|s| s.period_diff).collect::<Vec<_>>(),
        "period_diff",
        col_names,
    );
    push_group(
        &mut res,
        &mut names,
        &col_stats.iter().map(|s| s.period_ratio).collect::<Vec<_>>(),
        "period_ratio",
        col_names,
    );
    // 6. mean_above_p90 / mean_below_p10（可选）
    if opts.with_threshold_counts {
        push_group(
            &mut res,
            &mut names,
            &col_stats
                .iter()
                .map(|s| s.mean_above_p90)
                .collect::<Vec<_>>(),
            "mean_above_p90",
            col_names,
        );
        push_group(
            &mut res,
            &mut names,
            &col_stats
                .iter()
                .map(|s| s.mean_below_p10)
                .collect::<Vec<_>>(),
            "mean_below_p10",
            col_names,
        );
    }

    // 7. corr 矩阵上三角（列集可能含 append_for_corr 追加的列）
    {
        let mut idx = 0;
        for i in 0..n_corr_cols {
            for j in (i + 1)..n_corr_cols {
                res.push(corr_upper[idx]);
                if corr_names_all.len() == n_corr_cols {
                    names.push(format!("{}_corr_{}", corr_names_all[i], corr_names_all[j]));
                }
                idx += 1;
            }
        }
    }

    // 7b. lyapunov（Python 的 with_lyapunov_exponent，位置在 corr 之后、lz_complexity 之前）
    if opts.with_lyapunov_exponent {
        push_group(
            &mut res,
            &mut names,
            &col_stats.iter().map(|s| s.lyapunov).collect::<Vec<_>>(),
            "lyapunov",
            col_names,
        );
    }

    // 8. lz_complexity / entropy_1d / max_range_product（opts.with_complexity，Python 默认 True）
    if opts.with_complexity {
        push_group(
            &mut res,
            &mut names,
            &col_stats.iter().map(|s| s.lz).collect::<Vec<_>>(),
            "lz_complexity",
            col_names,
        );
        push_group(
            &mut res,
            &mut names,
            &col_stats.iter().map(|s| s.entropy).collect::<Vec<_>>(),
            "entropy_1d",
            col_names,
        );
        push_group(
            &mut res,
            &mut names,
            &col_stats.iter().map(|s| s.max_range).collect::<Vec<_>>(),
            "max_range_product",
            col_names,
        );
    }

    (res, names)
}

/// 辅助：把一组列统计量追加到结果向量（vals 与 col_names 等长，逐列生成 name）。
fn push_group(
    res: &mut Vec<f64>,
    names: &mut Vec<String>,
    vals: &[f64],
    suffix: &str,
    col_names: &[String],
) {
    if col_names.len() != vals.len() {
        res.extend_from_slice(vals);
        return;
    }
    for (ci, &v) in vals.iter().enumerate() {
        res.push(v);
        let cn = col_names.get(ci).map(|s| s.as_str()).unwrap_or("");
        names.push(format!("{}_{}", cn, suffix));
    }
}

/// 单列所有统计量的预计算结果。
struct ColStats {
    mean: f64,
    median: f64,
    std: f64,
    skew: f64,
    kurt: f64,
    p5: f64,
    p25: f64,
    p75: f64,
    p95: f64,
    iqr: f64,
    cv: f64,
    autocorr1: f64,
    trend: f64,
    curvature: f64,
    quad_coef: f64,
    max: f64,
    min: f64,
    range: f64,
    period_diff: f64,
    period_ratio: f64,
    mean_above_p90: f64,
    mean_below_p10: f64,
    lz: f64,
    entropy: f64,
    max_range: f64,
    lyapunov: f64,
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
    valid: Vec<f64>,        // 非 NaN 值（按列序）
    sorted: Vec<f64>,       // valid 的排序副本（median/分位共用）
    ent_counts: Vec<usize>, // 熵分箱计数（保留：旧入口仍使用）
    tbuf: Vec<f64>,         // 行主序 → 列主序转置缓冲（multi_factor per-stock 用）
    radix: Vec<f64>,        // 基数排序临时缓冲
}

/// f64 → 可排序 u64 键（非 NaN）：正数（含 +inf）映射到高位段，
/// 负数（含 -inf）按位取反，保证数值序 = 键序。
#[inline]
fn f64_sort_key(v: f64) -> u64 {
    let b = v.to_bits();
    if b >> 63 == 0 {
        b | 0x8000_0000_0000_0000
    } else {
        !b
    }
}

/// LSD 基数排序（4 遍 × 8 位，稳定）：4740 元素 ~20μs，比 pdqsort 快 5 倍以上。
/// 等值元素顺序与 pdqsort 不同，但分位数/中位数只读位置上的**值**，等值可互换 → 结果逐位一致。
fn radix_sort_f64(values: &mut [f64], tmp: &mut Vec<f64>) {
    let n = values.len();
    if n < 2 {
        return;
    }
    tmp.clear();
    tmp.resize(n, 0.0);
    let mut cnt = [0usize; 256];
    for shift in [0u32, 8, 16, 24, 32, 40, 48, 56] {
        cnt.fill(0);
        for &v in values.iter() {
            cnt[((f64_sort_key(v) >> shift) & 0xff) as usize] += 1;
        }
        let mut acc = 0usize;
        for c in cnt.iter_mut() {
            let t = *c;
            *c = acc;
            acc += t;
        }
        for &v in values.iter() {
            let k = f64_sort_key(v);
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
        src: &[f64],
        n_rows: usize,
        n_cols: usize,
        out: &mut [f64],
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
    data: &[f64],
    n_rows: usize,
    ia: usize,
    ra: usize,
    ib: usize,
    rb: usize,
) -> f64 {
    let (mut sum_a, mut sum_b, mut n) = (0.0f64, 0.0f64, 0usize);
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
        return f64::NAN;
    }
    let ma = sum_a / n as f64;
    let mb = sum_b / n as f64;
    let (mut cov, mut var_a, mut var_b) = (0.0f64, 0.0f64, 0.0f64);
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
    data: &[f64],
    n_rows: usize,
    n_cols: usize,
    row_stride: usize,
    col_stride: usize,
    scratch: &mut StatsScratch,
    out: &mut [f64],
) {
    let cols = n_cols;
    // 单列内联统计：按原实现逐位复刻（同一输入 → 同一输出）
    for j in 0..cols {
        let col_off = j * col_stride;
        let o = &mut out[j..];
        let o_stride = cols;

        // ---------- pass 1：过滤 + 基础累积（顺序与原实现一致） ----------
        scratch.valid.clear();
        let mut mean_sum = 0.0f64;
        let mut mean_n = 0usize;
        let split = n_rows / 3;
        let mut f_sum = 0.0f64; // period_diff/ratio 前半段
        let mut f_n = 0usize;
        let mut l_sum = 0.0f64; // 后半段
        let mut l_n = 0usize;
        let mut ac_sum_i = 0.0f64; // autocorr：corr_pair(c, shift(c)) 的均值分子
        let mut ac_sum_j = 0.0f64;
        let mut ac_n = 0usize;
        let mut ptr = col_off;
        let mut prev = f64::NAN;
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
            f64::NAN
        } else {
            mean_sum / mean_n as f64
        };
        let first_mean = if f_n == 0 {
            f64::NAN
        } else {
            f_sum / f_n as f64
        };
        let last_mean = if l_n == 0 {
            f64::NAN
        } else {
            l_sum / l_n as f64
        };
        o[0 * o_stride] = mean;
        o[16 * o_stride] = last_mean - first_mean;
        o[17 * o_stride] = last_mean / (first_mean.abs() + 1e-8);

        // ---------- std / skew / kurt（valid，两遍：均值 → 中心矩） ----------
        let n = scratch.valid.len();
        if n < 2 {
            o[2 * o_stride] = f64::NAN;
        } else {
            let m = scratch.valid.iter().sum::<f64>() / n as f64;
            let var = scratch.valid.iter().map(|&x| (x - m).powi(2)).sum::<f64>() / (n - 1) as f64;
            o[2 * o_stride] = var.sqrt();
        }
        if n < 3 {
            o[3 * o_stride] = f64::NAN;
        } else {
            let m = scratch.valid.iter().sum::<f64>() / n as f64;
            let nf = n as f64;
            let (mut s2, mut s3) = (0.0f64, 0.0f64);
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
            o[4 * o_stride] = f64::NAN;
        } else {
            let m = scratch.valid.iter().sum::<f64>() / n as f64;
            let nf = n as f64;
            let (mut s2, mut s4) = (0.0f64, 0.0f64);
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
        radix_sort_f64(&mut scratch.sorted, &mut scratch.radix);
        let sorted = &scratch.sorted;
        if n == 0 {
            o[1 * o_stride] = f64::NAN;
            o[5 * o_stride] = f64::NAN;
            o[6 * o_stride] = f64::NAN;
            o[7 * o_stride] = f64::NAN;
            o[8 * o_stride] = f64::NAN;
        } else {
            o[1 * o_stride] = if n % 2 == 0 {
                (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
            } else {
                sorted[n / 2]
            };
            // 与 col_quantile 完全一致的 pandas 线性插值
            let quantile = |q: f64| -> f64 {
                if n == 1 {
                    return sorted[0];
                }
                let pos = q * (n - 1) as f64;
                let lower = pos.floor() as usize;
                let upper = (lower + 1).min(n - 1);
                let frac = pos - lower as f64;
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
            o[11 * o_stride] = f64::NAN;
        } else {
            let mi = ac_sum_i / ac_n as f64;
            let mj = ac_sum_j / ac_n as f64;
            let (mut cov, mut var_i, mut var_j) = (0.0f64, 0.0f64, 0.0f64);
            let mut ptr = col_off;
            let mut prev = f64::NAN;
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

        // ---------- trend（f64，两遍） ----------
        let mut t_n = 0usize;
        let mut t_sum_x = 0.0f64;
        let mut t_sum_y = 0.0f64;
        {
            let mut ptr = col_off;
            for i in 0..n_rows {
                let v = data[ptr];
                ptr += row_stride;
                if !v.is_nan() {
                    t_n += 1;
                    t_sum_x += (i + 1) as f64;
                    t_sum_y += v;
                }
            }
        }
        if t_n < 2 {
            o[13 * o_stride] = 0.0;
        } else {
            let mx = t_sum_x / t_n as f64;
            let my = t_sum_y / t_n as f64;
            let (mut cov, mut var_x, mut var_y) = (0.0f64, 0.0f64, 0.0f64);
            let mut ptr = col_off;
            for i in 0..n_rows {
                let v = data[ptr];
                ptr += row_stride;
                if !v.is_nan() {
                    let dx = (i + 1) as f64 - mx;
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
                    (cov / (var_y.sqrt() * var_q.sqrt())) as f64
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
                            o[15 * o_stride] = (sign * delta) as f64;
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
    data: &[f64],
    n_rows: usize,
    n_cols: usize,
    row_stride: usize,
    col_stride: usize,
    out: &mut [f64],
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

/// 验证用：接收 numpy (n,m) 的 f64 矩阵 + 列名，调纯 Rust
/// get_features_factors_rust_full(with_threshold_counts=true)，全程不做 f32 转换。
///
/// 名字与 src/features.rs 的同名 pyfunction 一致（只是数值换成 f64），
/// Python 侧用 rp.verify_get_features_factors_rust_f64(...) 调。
#[pyfunction]
#[pyo3(name = "verify_get_features_factors_rust_f64")]
pub fn verify_get_features_factors_rust(
    data: numpy::PyReadonlyArray2<f64>,
    col_names: Vec<String>,
) -> PyResult<(Vec<f64>, Vec<String>)> {
    let view = data.as_array();
    let (vals, names) = get_features_factors_rust_full(&view, &col_names, true);
    Ok((vals, names))
}

/// 带全部开关的验证桥接（2026-10-06 加），Python 侧名字 `verify_get_features_factors_rust_f64_ex`。
///
/// 用来和 Python 的 `get_features_factors(df, with_max_min=..., with_lyapunov_exponent=...,
/// append_for_corr=...)` 逐值比。`append_matrix` 传 None 表示不追加。
/// 注意 Python 那边没有 curvature / quad_coef，对账时 `with_curvature` 要传 False。
/// `with_complexity`（2026-10-06 加，默认 True）关掉时不再输出
/// lz_complexity / entropy_1d / max_range_product 三组，前面所有组的位置和取值不变。
#[pyfunction]
#[pyo3(name = "verify_get_features_factors_rust_f64_ex")]
#[pyo3(signature = (
    data,
    col_names,
    with_threshold_counts = true,
    with_curvature = false,
    with_max_min = false,
    with_lyapunov_exponent = false,
    append_matrix = None,
    append_col_names = None,
    with_complexity = true
))]
pub fn verify_get_features_factors_rust_ex(
    data: numpy::PyReadonlyArray2<f64>,
    col_names: Vec<String>,
    with_threshold_counts: bool,
    with_curvature: bool,
    with_max_min: bool,
    with_lyapunov_exponent: bool,
    append_matrix: Option<numpy::PyReadonlyArray2<f64>>,
    append_col_names: Option<Vec<String>>,
    with_complexity: bool,
) -> PyResult<(Vec<f64>, Vec<String>)> {
    let view = data.as_array();
    let opts = FeatureOpts {
        with_threshold_counts,
        with_curvature,
        with_max_min,
        with_lyapunov_exponent,
        with_complexity,
    };
    let ap = append_matrix.as_ref().map(|m| m.as_array());
    let ap_names: Vec<String> = append_col_names.unwrap_or_default();
    let append = ap.as_ref().map(|a| (a, ap_names.as_slice()));
    let (vals, names) = get_features_factors_rust_full_ex(&view, &col_names, opts, append);
    Ok((vals, names))
}

#[cfg(test)]
mod stats21_tests {
    use super::*;
    use ndarray::Array2;

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
                    let mut data = vec![0.0f64; n_rows * n_cols];
                    for v in data.iter_mut() {
                        let r = (next() % 1000) as f64 / 1000.0;
                        *v = match r {
                            x if x < 0.03 => f64::NAN,
                            x if x < 0.05 => f64::INFINITY,
                            x if x < 0.07 => f64::NEG_INFINITY,
                            x if x < 0.10 => 0.0, // 常量/零值
                            x if x < 0.13 => 1.0, // 常量
                            _ => ((next() % 2000) as f64 - 1000.0) / 37.0,
                        };
                    }
                    if trial == 5 {
                        // 全 NaN 列
                        for v in data.iter_mut() {
                            *v = f64::NAN;
                        }
                    }
                    // 旧入口
                    let mat = Array2::from_shape_vec((n_rows, n_cols), data.clone()).unwrap();
                    let vals = get_features_factors_rust_values_only(&mat.view(), false);
                    // 新入口（行主序 → col_stride = n_cols）
                    let mut scratch = StatsScratch::new();
                    let mut out = vec![0.0f64; 21 * n_cols];
                    col_stats_21_strided(&data, n_rows, n_cols, n_cols, &mut scratch, &mut out);
                    let expect = &vals[..21 * n_cols];
                    for i in 0..21 * n_cols {
                        assert!(
                            out[i].to_bits() == expect[i].to_bits(),
                            "mismatch rows={n_rows} cols={n_cols} trial={trial} idx={i}: new={:?} old={:?}",
                            out[i],
                            expect[i]
                        );
                    }
                    // 列主序（col_stride = n_rows）也应一致
                    let mut cm = vec![0.0f64; n_rows * n_cols];
                    for c in 0..n_cols {
                        for r in 0..n_rows {
                            cm[c * n_rows + r] = data[r * n_cols + c];
                        }
                    }
                    let mut out2 = vec![0.0f64; 21 * n_cols];
                    let mut scratch2 = StatsScratch::new();
                    col_stats_21_strided(&cm, n_rows, n_cols, n_rows, &mut scratch2, &mut out2);
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
}
