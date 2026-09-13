//! 21 统计量降维（stat-major 顺序），与正式库 features.rs 的
//! get_features_factors_rust_full(with_threshold_counts=false) 的前 21×n_cols
//! 个值逐位一致（不含 corr 上三角、不含 mean_above_p90/mean_below_p10）。
//!
//! 输出顺序：mean, median, std, skew, kurt, p5, p25, p75, p95, iqr, cv,
//! autocorr1, autocorr1_abs, trend, curvature, quad_coef,
//! period_diff, period_ratio, lz_complexity, entropy_1d, max_range_product
//! 每组统计量按列顺序展开（stat-major）。

/// 单列均值（跳过 NaN，空列返回 NaN）。对齐 pandas df.mean()。
#[inline]
fn col_mean(col: &[f32]) -> f32 {
    let (sum, n) = col.iter().fold((0.0f32, 0usize), |(s, c), &v| {
        if v.is_nan() {
            (s, c)
        } else {
            (s + v, c + 1)
        }
    });
    if n == 0 {
        f32::NAN
    } else {
        sum / n as f32
    }
}

/// 单列标准差（样本标准差 ddof=1，对齐 pandas df.std()）。
#[inline]
fn col_std(col: &[f32]) -> f32 {
    let valid: Vec<f32> = col.iter().filter(|&&v| !v.is_nan()).copied().collect();
    let n = valid.len();
    if n < 2 {
        return f32::NAN;
    }
    let mean = valid.iter().sum::<f32>() / n as f32;
    let var = valid.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / (n - 1) as f32;
    var.sqrt()
}

/// 单列偏度（对齐 pandas df.skew()，G1 估计量）。
#[inline]
fn col_skew(col: &[f32]) -> f32 {
    let valid: Vec<f32> = col.iter().filter(|&&v| !v.is_nan()).copied().collect();
    let n = valid.len();
    if n < 3 {
        return f32::NAN;
    }
    let mean = valid.iter().sum::<f32>() / n as f32;
    let nf = n as f32;
    let mut s2 = 0.0;
    let mut s3 = 0.0;
    for &x in &valid {
        let d = x - mean;
        s2 += d * d;
        s3 += d * d * d;
    }
    let k2 = s2 / (nf - 1.0);
    let k3 = nf * s3 / ((nf - 1.0) * (nf - 2.0));
    if k2.abs() < 1e-30 {
        return 0.0;
    }
    k3 / k2.powf(1.5)
}

/// 单列峰度（对齐 pandas df.kurt()，G2 超额峰度）。
#[inline]
fn col_kurt(col: &[f32]) -> f32 {
    let valid: Vec<f32> = col.iter().filter(|&&v| !v.is_nan()).copied().collect();
    let n = valid.len();
    if n < 4 {
        return f32::NAN;
    }
    let mean = valid.iter().sum::<f32>() / n as f32;
    let nf = n as f32;
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
    if k2.abs() < 1e-30 {
        return 0.0;
    }
    k4 / (k2 * k2)
}

/// 单列中位数（跳过 NaN）。
#[inline]
fn col_median(col: &[f32]) -> f32 {
    let mut valid: Vec<f32> = col.iter().filter(|&&v| !v.is_nan()).copied().collect();
    if valid.is_empty() {
        return f32::NAN;
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
#[inline]
fn col_quantile(col: &[f32], q: f32) -> f32 {
    let mut valid: Vec<f32> = col.iter().filter(|&&v| !v.is_nan()).copied().collect();
    if valid.is_empty() {
        return f32::NAN;
    }
    valid.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = valid.len();
    if n == 1 {
        return valid[0];
    }
    let pos = q * (n - 1) as f32;
    let lower = pos.floor() as usize;
    let upper = (lower + 1).min(n - 1);
    let frac = pos - lower as f32;
    valid[lower] * (1.0 - frac) + valid[upper] * frac
}

/// 一阶趋势：序列与 [1..n] 的 Pearson 相关（过滤 NaN）。
#[inline]
fn trend_1d(col: &[f32]) -> f32 {
    let pairs: Vec<(usize, f32)> = col
        .iter()
        .enumerate()
        .filter(|(_, &v)| !v.is_nan())
        .map(|(i, &v)| (i + 1, v))
        .collect();
    let n = pairs.len();
    if n < 2 {
        return 0.0;
    }
    let mean_x: f32 = pairs.iter().map(|(x, _)| *x as f32).sum::<f32>() / n as f32;
    let mean_y: f32 = pairs.iter().map(|(_, y)| *y).sum::<f32>() / n as f32;
    let (mut cov, mut var_x, mut var_y) = (0.0, 0.0, 0.0);
    for (x, y) in &pairs {
        let dx = *x as f32 - mean_x;
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

/// 二阶弯曲方向：序列与 (t−t̄)² 的 Pearson 相关。
#[inline]
fn curvature_1d(col: &[f32]) -> f32 {
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
    (cov / (var_y.sqrt() * var_q.sqrt())) as f32
}

/// 3×3 线性方程组求解（部分主元高斯消元）。
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

/// 二次项相对线性项的边际解释方差（带符号）。
#[inline]
fn quad_coef_1d(col: &[f32]) -> f32 {
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
    let b1 = s_uy / s_uu;
    let ss_res_lin = ss_tot - b1 * s_uy;
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
    let ssr_quad = c2 * (nf * mean_y) + b2 * s_uy + a2 * s_uu_y - nf * mean_y * mean_y;
    let ss_res_quad = ss_tot - ssr_quad;
    let r2_lin = 1.0 - ss_res_lin / ss_tot;
    let r2_quad = 1.0 - ss_res_quad / ss_tot;
    let delta = (r2_quad - r2_lin).clamp(0.0, 1.0);
    let sign = if a2 >= 0.0 { 1.0 } else { -1.0 };
    (sign * delta) as f32
}

/// 两列 Pearson 相关（共同有效位置）。对齐 pandas corr。
#[inline]
fn corr_pair(col_i: &[f32], col_j: &[f32]) -> f32 {
    let pairs: Vec<(f32, f32)> = col_i
        .iter()
        .zip(col_j.iter())
        .filter(|(&a, &b)| !a.is_nan() && !b.is_nan())
        .map(|(&a, &b)| (a, b))
        .collect();
    let n = pairs.len();
    if n < 2 {
        return f32::NAN;
    }
    let mean_i: f32 = pairs.iter().map(|(i, _)| i).sum::<f32>() / n as f32;
    let mean_j: f32 = pairs.iter().map(|(_, j)| j).sum::<f32>() / n as f32;
    let (mut cov, mut var_i, mut var_j) = (0.0, 0.0, 0.0);
    for (i, j) in &pairs {
        let di = i - mean_i;
        let dj = j - mean_j;
        cov += di * dj;
        var_i += di * di;
        var_j += dj * dj;
    }
    if var_i == 0.0 || var_j == 0.0 {
        return 0.0;
    }
    cov / (var_i.sqrt() * var_j.sqrt())
}

// ---------------- LZ 复杂度（复制自正式库 lz_complexity.rs / features.rs） ----------------

#[derive(Clone)]
struct SamState {
    len: usize,
    link: Option<usize>,
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

fn lz_calculate_complexity(seq: &[u8]) -> usize {
    let n = seq.len();
    if n == 0 {
        return 0;
    }
    if n <= 64 {
        return lz_complexity_simple(seq);
    }
    lz_complexity_suffix_automaton_in(seq, &mut Vec::new())
}

fn lz_complexity_1d(col: &[f32]) -> f32 {
    let valid: Vec<f32> = col.iter().copied().filter(|v| v.is_finite()).collect();
    let n = valid.len();
    if n == 0 {
        return 0.0;
    }
    let mut sorted = valid.clone();
    sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    let sn = sorted.len();
    let quantiles = [0.33f32, 0.66];
    let mut thresholds = Vec::with_capacity(quantiles.len());
    for &q in &quantiles {
        let idx = ((sn - 1) as f32 * q) as usize;
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
        discrete.push(symbol + 1);
    }
    let complexity = lz_calculate_complexity(&discrete);
    let k_eff = {
        let mut s = std::collections::HashSet::new();
        for &d in &discrete {
            s.insert(d);
        }
        s.len() as f32
    };
    if n <= 1 {
        return 0.0;
    }
    if k_eff < (quantiles.len() as f32 + 1.0) {
        return f32::NAN;
    }
    let log_n_base_k = (n as f32).ln() / k_eff.ln();
    complexity as f32 * log_n_base_k / n as f32
}

/// 分箱熵（等宽分箱 + Shannon 熵，n_bins = log2(n).ceil()+1）。
fn binned_entropy_1d(col: &[f32], n_bins: usize) -> f32 {
    let valid: Vec<f32> = col.iter().filter(|&&v| !v.is_nan()).copied().collect();
    if valid.is_empty() {
        return 0.0;
    }
    if n_bins == 0 {
        return 0.0;
    }
    let min_val = valid.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    let max_val = valid.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    if (max_val - min_val).abs() < f32::EPSILON {
        return 0.0;
    }
    let bin_width = (max_val - min_val) / n_bins as f32;
    let mut counts: Vec<usize> = vec![0; n_bins];
    for &v in &valid {
        let mut idx = ((v - min_val) / bin_width).floor() as usize;
        if idx >= n_bins {
            idx = n_bins - 1;
        }
        counts[idx] += 1;
    }
    let total = valid.len() as f32;
    counts
        .iter()
        .map(|&c| {
            let p = c as f32 / total;
            if p > 0.0 {
                -p * p.ln()
            } else {
                0.0
            }
        })
        .sum()
}

/// 最大范围积的严格对齐版：abs(idx1-idx2)/n。
fn max_range_product_strict(col: &[f32]) -> f32 {
    let valid: Vec<f32> = col.iter().filter(|&&v| !v.is_nan()).copied().collect();
    let n_total = col.len();
    let n = valid.len();
    if n < 2 || n_total == 0 {
        return 0.0;
    }
    let mut max_product = f32::NEG_INFINITY;
    let mut result = (0i64, 0i64);
    let mut left = 0;
    let mut right = n - 1;
    while left < right {
        let product = valid[left].min(valid[right]) * (right - left) as f32;
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
    (i - j).abs() as f32 / n_total as f32
}

/// 21 统计量降维入口：输入 (n_rows, n_cols) 行主序矩阵，输出 stat-major 的 21×n_cols 个值。
pub fn reduce_21_flat(flat: &[f32], n_rows: usize, n_cols: usize) -> Vec<f32> {
    if n_rows == 0 || n_cols == 0 {
        return vec![f32::NAN; 21 * n_cols];
    }
    // 转置为列主序
    let mut cols: Vec<Vec<f32>> = vec![Vec::with_capacity(n_rows); n_cols];
    for r in 0..n_rows {
        for (c, colv) in cols.iter_mut().enumerate() {
            colv.push(flat[r * n_cols + c]);
        }
    }
    let mut res: Vec<f32> = Vec::with_capacity(21 * n_cols);
    // 1. mean / median / std / skew / kurt
    let means: Vec<f32> = cols.iter().map(|c| col_mean(c)).collect();
    res.extend_from_slice(&means);
    res.extend(cols.iter().map(|c| col_median(c)));
    res.extend(cols.iter().map(|c| col_std(c)));
    res.extend(cols.iter().map(|c| col_skew(c)));
    res.extend(cols.iter().map(|c| col_kurt(c)));
    // 2. p5/p25/p75/p95/iqr/cv
    let p5: Vec<f32> = cols.iter().map(|c| col_quantile(c, 0.05)).collect();
    let p25: Vec<f32> = cols.iter().map(|c| col_quantile(c, 0.25)).collect();
    let p75: Vec<f32> = cols.iter().map(|c| col_quantile(c, 0.75)).collect();
    let p95: Vec<f32> = cols.iter().map(|c| col_quantile(c, 0.95)).collect();
    let iqr: Vec<f32> = p75.iter().zip(p25.iter()).map(|(a, b)| a - b).collect();
    let cv: Vec<f32> = cols
        .iter()
        .zip(means.iter())
        .map(|(c, &m)| col_std(c) / (m.abs() + 1e-8))
        .collect();
    res.extend_from_slice(&p5);
    res.extend_from_slice(&p25);
    res.extend_from_slice(&p75);
    res.extend_from_slice(&p95);
    res.extend_from_slice(&iqr);
    res.extend_from_slice(&cv);
    // 3. autocorr1 / autocorr1_abs
    let autocorr: Vec<f32> = cols
        .iter()
        .map(|c| {
            if c.len() < 2 {
                return f32::NAN;
            }
            let shifted: Vec<f32> = std::iter::once(f32::NAN)
                .chain(c[..c.len() - 1].iter().copied())
                .collect();
            corr_pair(c, &shifted)
        })
        .collect();
    let autocorr_abs: Vec<f32> = autocorr.iter().map(|v| v.abs()).collect();
    res.extend_from_slice(&autocorr);
    res.extend_from_slice(&autocorr_abs);
    // 4. trend / curvature / quad_coef
    res.extend(cols.iter().map(|c| trend_1d(c)));
    res.extend(cols.iter().map(|c| curvature_1d(c)));
    res.extend(cols.iter().map(|c| quad_coef_1d(c)));
    // 5. period_diff / period_ratio
    for c in &cols {
        let n = c.len();
        let split = n / 3;
        let first_mean = if split > 0 { col_mean(&c[..split]) } else { f32::NAN };
        let last_mean = if split > 0 { col_mean(&c[n - split..]) } else { f32::NAN };
        let pd = last_mean - first_mean;
        res.push(pd);
    }
    for c in &cols {
        let n = c.len();
        let split = n / 3;
        let first_mean = if split > 0 { col_mean(&c[..split]) } else { f32::NAN };
        let last_mean = if split > 0 { col_mean(&c[n - split..]) } else { f32::NAN };
        res.push(last_mean / (first_mean.abs() + 1e-8));
    }
    // 6. lz / entropy / max_range
    res.extend(cols.iter().map(|c| lz_complexity_1d(c)));
    for c in &cols {
        let n = c.len();
        let n_bins = (n as f32).log2().ceil() as usize + 1;
        res.push(binned_entropy_1d(c, n_bins));
    }
    res.extend(cols.iter().map(|c| max_range_product_strict(c)));
    res
}

/// 21 统计量名（stat-major 展开给定列名）。
pub fn reduce_21_names(col_names: &[&str]) -> Vec<String> {
    let stats = [
        "mean", "median", "std", "skew", "kurt", "p5", "p25", "p75", "p95", "iqr", "cv",
        "autocorr1", "autocorr1_abs", "trend", "curvature", "quad_coef", "period_diff",
        "period_ratio", "lz_complexity", "entropy_1d", "max_range_product",
    ];
    let mut out = Vec::with_capacity(stats.len() * col_names.len());
    for s in stats.iter() {
        for cn in col_names.iter() {
            out.push(format!("{}_{}", cn, s));
        }
    }
    out
}
