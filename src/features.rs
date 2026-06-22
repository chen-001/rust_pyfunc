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
use ndarray::{ArrayView2};
use pyo3::prelude::*;
use rayon::prelude::*;
use std::collections::HashMap;

// ============================================================================
// 基础统计量（单列计算，对齐 pandas 行为含 NaN 处理）
// ============================================================================

/// 单列均值（跳过 NaN，空列返回 NaN）。对齐 pandas df.mean()。
#[inline]
fn col_mean(col: &[f64]) -> f64 {
    let (sum, n) = col.iter().fold((0.0f64, 0usize), |(s, c), &v| {
        if v.is_nan() { (s, c) } else { (s + v, c + 1) }
    });
    if n == 0 { f64::NAN } else { sum / n as f64 }
}

/// 单列标准差（样本标准差 ddof=1，对齐 pandas df.std()）。空或单元素返回 NaN。
#[inline]
fn col_std(col: &[f64]) -> f64 {
    let valid: Vec<f64> = col.iter().filter(|&&v| !v.is_nan()).copied().collect();
    let n = valid.len();
    if n < 2 { return f64::NAN; }
    let mean = valid.iter().sum::<f64>() / n as f64;
    let var = valid.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
    var.sqrt()
}

/// 单列偏度（对齐 pandas df.skew()，基于 G1 偏度估计量，用 k-statistic）。
/// n<3 返回 NaN。公式：g1 = k3/k2^1.5，其中 k2=S2/(n-1)，k3=n*S3/((n-1)(n-2))。
#[inline]
fn col_skew(col: &[f64]) -> f64 {
    let valid: Vec<f64> = col.iter().filter(|&&v| !v.is_nan()).copied().collect();
    let n = valid.len();
    if n < 3 { return f64::NAN; }
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
    if k2.abs() < 1e-30 { return 0.0; }
    k3 / k2.powf(1.5)
}

/// 单列峰度（对齐 pandas df.kurt()，基于 G2 超额峰度，用 k-statistic）。
/// n<4 返回 NaN。公式：g2 = k4/k2^2，其中
///   k2 = S2/(n-1)
///   k4 = n*[(n+1)*S4 - 3*(n-1)*S2^2/n] / [(n-1)*(n-2)*(n-3)]
#[inline]
fn col_kurt(col: &[f64]) -> f64 {
    let valid: Vec<f64> = col.iter().filter(|&&v| !v.is_nan()).copied().collect();
    let n = valid.len();
    if n < 4 { return f64::NAN; }
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
    if k2.abs() < 1e-30 { return 0.0; }
    k4 / (k2 * k2)
}

/// 单列中位数（跳过 NaN）。
#[inline]
fn col_median(col: &[f64]) -> f64 {
    let mut valid: Vec<f64> = col.iter().filter(|&&v| !v.is_nan()).copied().collect();
    if valid.is_empty() { return f64::NAN; }
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
    if valid.is_empty() { return f64::NAN; }
    valid.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = valid.len();
    if n == 1 { return valid[0]; }
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
fn trend_1d(col: &[f64]) -> f64 {
    // 过滤 NaN，保留有效值及其原始索引（1-based）
    let pairs: Vec<(usize, f64)> = col
        .iter()
        .enumerate()
        .filter(|(_, &v)| !v.is_nan())
        .map(|(i, &v)| (i + 1, v))
        .collect();
    let n = pairs.len();
    if n < 2 { return 0.0; }
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
    if var_x == 0.0 || var_y == 0.0 { return 0.0; }
    cov / (var_x.sqrt() * var_y.sqrt())
}

/// 两列的 Pearson 相关系数（共同有效位置）。对齐 pandas corr。
#[inline]
fn corr_pair(col_i: &[f64], col_j: &[f64]) -> f64 {
    let pairs: Vec<(f64, f64)> = col_i
        .iter()
        .zip(col_j.iter())
        .filter(|(&a, &b)| !a.is_nan() && !b.is_nan())
        .map(|(&a, &b)| (a, b))
        .collect();
    let n = pairs.len();
    if n < 2 { return f64::NAN; }
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
    if var_i == 0.0 || var_j == 0.0 { return f64::NAN; }
    cov / (var_i.sqrt() * var_j.sqrt())
}

/// LZ 复杂度（精确复制自 lz_complexity.rs）。分位数离散化 [0.33, 0.66] + 归一化。
fn lz_complexity_1d(col: &[f64]) -> f64 {
    let n = col.len();
    if n == 0 { return 0.0; }

    // 分位数离散化（quantiles=[0.33, 0.66]），精确复制 discretize_sequence
    let mut sorted: Vec<f64> = col.iter().copied().collect();
    sorted.sort_unstable_by(|a, b| match (a.is_nan(), b.is_nan()) {
        (true, true) => std::cmp::Ordering::Equal,
        (true, false) => std::cmp::Ordering::Greater,
        (false, true) => std::cmp::Ordering::Less,
        (false, false) => a.partial_cmp(b).unwrap(),
    });
    let sn = sorted.len();
    let quantiles = [0.33f64, 0.66];
    let mut thresholds = Vec::with_capacity(quantiles.len());
    for &q in &quantiles {
        let idx = ((sn - 1) as f64 * q) as usize;
        thresholds.push(sorted[idx]);
    }

    let mut discrete: Vec<u8> = Vec::with_capacity(n);
    for &val in col.iter() {
        if val.is_nan() { return f64::NAN; }
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
        for &d in &discrete { s.insert(d); }
        s.len() as f64
    };
    if n <= 1 { return 0.0; }
    if k_eff < (quantiles.len() as f64 + 1.0) { return f64::NAN; }

    let log_n_base_k = (n as f64).ln() / k_eff.ln();
    complexity as f64 * log_n_base_k / n as f64
}

/// LZ 复杂度核心调度（复制自 lz_complexity.rs 的 calculate_lz_complexity）。
fn lz_calculate_complexity(seq: &[u8]) -> usize {
    let n = seq.len();
    if n == 0 { return 0; }
    if n <= 64 { return lz_complexity_simple(seq); }
    lz_complexity_suffix_automaton(seq)
}

/// LZ 复杂度暴力版（精确复制自 lz_complexity.rs:663）。
fn lz_complexity_simple(seq: &[u8]) -> usize {
    let n = seq.len();
    if n == 0 { return 0; }
    let mut complexity = 0;
    let mut i = 0;
    while i < n {
        let mut j = i + 1;
        while j <= n {
            let sub_len = j - i;
            let search_end = j - 1;
            if search_end < sub_len { break; }
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
    transitions: Vec<(u8, usize)>,
}

impl SamState {
    fn new(len: usize) -> Self {
        Self { len, link: None, transitions: Vec::with_capacity(2) }
    }
    #[inline]
    fn get(&self, c: u8) -> Option<usize> {
        self.transitions.iter().find_map(|&(ch, state)| if ch == c { Some(state) } else { None })
    }
    #[inline]
    fn set(&mut self, c: u8, state: usize) {
        for (ch, target) in &mut self.transitions {
            if *ch == c { *target = state; return; }
        }
        self.transitions.push((c, state));
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
            if self.states[p_idx].get(c).is_some() { break; }
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
    let n = seq.len();
    if n == 0 { return 0; }
    let mut sam = SuffixAutomaton::with_capacity(2 * n);
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
    complexity
}

/// 分箱熵（复制自 entropy_analysis.rs）。等宽分箱 + Shannon 熵。
fn binned_entropy_1d(col: &[f64], n_bins: usize) -> f64 {
    let valid: Vec<f64> = col.iter().filter(|&&v| !v.is_nan()).copied().collect();
    if valid.is_empty() { return 0.0; }
    if n_bins == 0 { return 0.0; }

    let min_val = valid.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max_val = valid.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    if (max_val - min_val).abs() < f64::EPSILON { return 0.0; }

    let bin_width = (max_val - min_val) / n_bins as f64;
    let mut counts: HashMap<usize, usize> = HashMap::new();
    for &v in &valid {
        let mut idx = ((v - min_val) / bin_width).floor() as usize;
        if idx >= n_bins { idx = n_bins - 1; }
        *counts.entry(idx).or_insert(0) += 1;
    }
    let total = valid.len() as f64;
    counts
        .values()
        .map(|&c| {
            let p = c as f64 / total;
            if p > 0.0 { -p * p.ln() } else { 0.0 }
        })
        .sum()
}

/// 最大范围积的严格对齐版（复制自 sequence/mod.rs 的双指针逻辑）。
/// 返回 abs(idx1 - idx2)/n，其中 idx1,idx2 是 find_max_range_product 返回的两个索引。
/// 注意：Python 的 _calc_max_range_product 取的是 abs(索引1-索引2)/n，不是值差。
fn max_range_product_strict(col: &[f64]) -> f64 {
    let valid: Vec<f64> = col.iter().filter(|&&v| !v.is_nan()).copied().collect();
    let n_total = col.len(); // Python 用 series.shape[0]（含NaN的原长度）
    let n = valid.len();
    if n < 2 || n_total == 0 { return 0.0; }

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
        if valid[left] < valid[right] { left += 1; } else { right -= 1; }
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
/// with_lag_autocorr=1, with_threshold_counts=True, with_period_compare=True,
/// with_complexity=True），但 **关闭 lyapunov**（with_lyapunov_exponent=False）。
///
/// 输入 data: (n_rows, n_cols) 的 f64 矩阵（order_pair_metrics 的输出）。
/// 输出 (vals, names)，vals 为展平的特征向量，names 与之等长。
///
/// 输出顺序与 Python get_features_factors(with_lyapunov_exponent=False) 严格一致。
pub fn get_features_factors_rust(
    data: &ArrayView2<f64>,
    col_names: &[String],
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
                    if !v.is_nan() && v > p90 { (s + v, n + 1) } else { (s, n) }
                });
                if n == 0 { 0.0 } else { s / n as f64 }
            };
            let mean_below_p10 = {
                let (s, n) = c.iter().fold((0.0f64, 0usize), |(s, n), &v| {
                    if !v.is_nan() && v < p10 { (s + v, n + 1) } else { (s, n) }
                });
                if n == 0 { 0.0 } else { s / n as f64 }
            };
            // period_compare
            let split = n_rows / 3;
            let first_mean = if split > 0 { col_mean(&c[..split]) } else { f64::NAN };
            let last_mean = if split > 0 { col_mean(&c[n_rows - split..]) } else { f64::NAN };
            let period_diff = last_mean - first_mean;
            let period_ratio = last_mean / (first_mean.abs() + 1e-8);
            // trend
            let trend = trend_1d(c);
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

            ColStats {
                mean, median, std, skew, kurt,
                p5, p25, p75, p95, iqr, cv,
                autocorr1, trend,
                period_diff, period_ratio,
                mean_above_p90, mean_below_p10,
                lz, entropy, max_range,
            }
        })
        .collect();

    // corr 矩阵上三角（并行计算所有对）
    let corr_upper: Vec<f64> = if n_cols >= 2 {
        let pairs: Vec<(usize, usize)> = (0..n_cols)
            .flat_map(|i| (i + 1..n_cols).map(move |j| (i, j)))
            .collect();
        pairs
            .iter()
            .map(|&(i, j)| corr_pair(&cols[i], &cols[j]))
            .collect()
    } else {
        vec![]
    };

    // ============ 按顺序拼接结果（对齐 Python get_features_factors）============
    let mut res: Vec<f64> = Vec::new();
    let mut names: Vec<String> = Vec::new();

    // 1. mean/median/std/skew/kurt
    push_group(&mut res, &mut names, &col_stats.iter().map(|s| s.mean).collect::<Vec<_>>(), "mean", col_names);
    push_group(&mut res, &mut names, &col_stats.iter().map(|s| s.median).collect::<Vec<_>>(), "median", col_names);
    push_group(&mut res, &mut names, &col_stats.iter().map(|s| s.std).collect::<Vec<_>>(), "std", col_names);
    push_group(&mut res, &mut names, &col_stats.iter().map(|s| s.skew).collect::<Vec<_>>(), "skew", col_names);
    push_group(&mut res, &mut names, &col_stats.iter().map(|s| s.kurt).collect::<Vec<_>>(), "kurt", col_names);
    // 2. p5/p25/p75/p95/iqr/cv
    push_group(&mut res, &mut names, &col_stats.iter().map(|s| s.p5).collect::<Vec<_>>(), "p5", col_names);
    push_group(&mut res, &mut names, &col_stats.iter().map(|s| s.p25).collect::<Vec<_>>(), "p25", col_names);
    push_group(&mut res, &mut names, &col_stats.iter().map(|s| s.p75).collect::<Vec<_>>(), "p75", col_names);
    push_group(&mut res, &mut names, &col_stats.iter().map(|s| s.p95).collect::<Vec<_>>(), "p95", col_names);
    push_group(&mut res, &mut names, &col_stats.iter().map(|s| s.iqr).collect::<Vec<_>>(), "iqr", col_names);
    push_group(&mut res, &mut names, &col_stats.iter().map(|s| s.cv).collect::<Vec<_>>(), "cv", col_names);
    // 3. autocorr1 / autocorr1_abs
    push_group(&mut res, &mut names, &col_stats.iter().map(|s| s.autocorr1).collect::<Vec<_>>(), "autocorr1", col_names);
    push_group(&mut res, &mut names, &col_stats.iter().map(|s| s.autocorr1.abs()).collect::<Vec<_>>(), "autocorr1_abs", col_names);
    // 4. trend
    push_group(&mut res, &mut names, &col_stats.iter().map(|s| s.trend).collect::<Vec<_>>(), "trend", col_names);
    // 5. period_diff / period_ratio
    push_group(&mut res, &mut names, &col_stats.iter().map(|s| s.period_diff).collect::<Vec<_>>(), "period_diff", col_names);
    push_group(&mut res, &mut names, &col_stats.iter().map(|s| s.period_ratio).collect::<Vec<_>>(), "period_ratio", col_names);
    // 6. mean_above_p90 / mean_below_p10
    push_group(&mut res, &mut names, &col_stats.iter().map(|s| s.mean_above_p90).collect::<Vec<_>>(), "mean_above_p90", col_names);
    push_group(&mut res, &mut names, &col_stats.iter().map(|s| s.mean_below_p10).collect::<Vec<_>>(), "mean_below_p10", col_names);

    // 7. corr 矩阵上三角
    if n_cols >= 2 {
        let mut idx = 0;
        for i in 0..n_cols {
            for j in (i + 1)..n_cols {
                res.push(corr_upper[idx]);
                names.push(format!("{}_corr_{}", col_names[i], col_names[j]));
                idx += 1;
            }
        }
    }

    // 8. lz_complexity / entropy_1d / max_range_product（关闭 lyapunov）
    push_group(&mut res, &mut names, &col_stats.iter().map(|s| s.lz).collect::<Vec<_>>(), "lz_complexity", col_names);
    push_group(&mut res, &mut names, &col_stats.iter().map(|s| s.entropy).collect::<Vec<_>>(), "entropy_1d", col_names);
    push_group(&mut res, &mut names, &col_stats.iter().map(|s| s.max_range).collect::<Vec<_>>(), "max_range_product", col_names);

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
    period_diff: f64,
    period_ratio: f64,
    mean_above_p90: f64,
    mean_below_p10: f64,
    lz: f64,
    entropy: f64,
    max_range: f64,
}

// ============================================================================
// PyO3 验证桥接（仅供 Python 端一致性验证）
// ============================================================================

/// 验证用：接收 numpy (n,m) 矩阵 + 列名，调纯 Rust get_features_factors_rust。
#[pyfunction]
pub fn verify_get_features_factors_rust(
    data: numpy::PyReadonlyArray2<f64>,
    col_names: Vec<String>,
) -> PyResult<(Vec<f64>, Vec<String>)> {
    let view = data.as_array();
    Ok(get_features_factors_rust(&view, &col_names))
}
