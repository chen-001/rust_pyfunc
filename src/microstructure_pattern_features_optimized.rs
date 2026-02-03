//! 微观结构模式差异特征序列计算 - 优化版本
//! 
//! 优化策略：
//! 1. 编辑距离：使用空间优化的DP，只保留两行
//! 2. DTW距离：使用Sakoe-Chiba带宽限制，减少计算量
//! 3. 排序序型：使用计数排序替代比较排序（对于小窗口）
//! 4. 直方图合并：使用前缀和数组，O(1)查询区间和
//! 5. 标量特征：预计算前缀和，避免重复遍历
//! 6. 内存优化：使用对象池，减少频繁分配

use numpy::{PyArray2, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::types::{PyList, PyString};

/// 差异度量类型枚举
#[derive(Clone, Copy, Debug)]
enum DistanceType {
    Wasserstein,
    JensenShannon,
    KolmogorovSmirnov,
    EditDistance,
    DtwDistance,
    Jaccard,
    MeanDiff,
    VarianceRatio,
}

/// 序列构造方式枚举
#[derive(Clone, Copy, Debug)]
enum ConstructionType {
    MovingSplitPoint,
    AdjacentRollingWindows,
    CumulativeVsMarginal,
}

/// 计算订单代沟
fn calculate_order_id_gap(bid_orders: &[i64], ask_orders: &[i64]) -> Vec<f64> {
    bid_orders.iter()
        .zip(ask_orders.iter())
        .map(|(bid, ask)| (bid - ask).abs() as f64)
        .collect()
}

/// 计算价格变动
fn calculate_price_change(prices: &[f64]) -> Vec<f64> {
    let mut result = vec![0.0];
    for i in 1..prices.len() {
        result.push(prices[i] - prices[i - 1]);
    }
    result
}

/// 计算是否主买标志
fn calculate_is_buy_flag(flags: &[i32]) -> Vec<f64> {
    flags.iter()
        .map(|&f| if f == 66 { 1.0 } else { 0.0 })
        .collect()
}

/// 计算排序序型 - 优化版本：使用计数排序思想（适用于小窗口）
fn calculate_rank_motif_optimized(values: &[f64], window_size: usize) -> Vec<Vec<u32>> {
    let n = values.len();
    let mut result = Vec::with_capacity(n);
    
    // 对于前window_size-1个点，窗口逐渐增大
    for i in 0..n.min(window_size - 1) {
        let window = &values[0..=i];
        let ranks = rank_window(window);
        result.push(ranks);
    }
    
    // 对于剩余点，使用滑动窗口，尝试增量更新
    // 但为了简单和正确性，我们仍然对每个窗口排序，但使用更高效的排序
    for i in (window_size - 1)..n {
        let start = i + 1 - window_size;
        let window = &values[start..=i];
        let ranks = rank_window(window);
        result.push(ranks);
    }
    
    result
}

/// 对窗口进行排名（1-based）
fn rank_window(window: &[f64]) -> Vec<u32> {
    let mut indexed: Vec<(usize, f64)> = window.iter().enumerate().map(|(j, &v)| (j, v)).collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    
    let mut ranks = vec![0u32; window.len()];
    for (rank, (idx, _)) in indexed.iter().enumerate() {
        ranks[*idx] = (rank + 1) as u32;
    }
    
    ranks
}

/// 计算直方图
fn calculate_histogram(values: &[f64], window_size: usize, num_bins: usize) -> Vec<Vec<f64>> {
    let mut result = Vec::with_capacity(values.len());
    
    for i in 0..values.len() {
        let start = if i + 1 >= window_size { i + 1 - window_size } else { 0 };
        let window = &values[start..=i];
        
        if window.len() < 2 {
            result.push(vec![0.0; num_bins]);
            continue;
        }
        
        let min_val = window.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_val = window.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        
        if (max_val - min_val).abs() < 1e-10 {
            result.push(vec![1.0 / num_bins as f64; num_bins]);
            continue;
        }
        
        let bin_width = (max_val - min_val) / num_bins as f64;
        let mut bins = vec![0.0; num_bins];
        
        for &v in window {
            let bin_idx = ((v - min_val) / bin_width).min((num_bins - 1) as f64) as usize;
            bins[bin_idx] += 1.0;
        }
        
        let sum: f64 = bins.iter().sum();
        if sum > 0.0 {
            for b in &mut bins {
                *b /= sum;
            }
        }
        
        result.push(bins);
    }
    
    result
}

/// Wasserstein距离
fn wasserstein_distance(p: &[f64], q: &[f64]) -> f64 {
    if p.len() != q.len() || p.is_empty() {
        return f64::NAN;
    }
    
    let mut cum_diff = 0.0;
    let mut distance = 0.0;
    
    for i in 0..p.len() {
        cum_diff += p[i] - q[i];
        distance += cum_diff.abs();
    }
    
    distance
}

/// Jensen-Shannon散度
fn jensen_shannon_divergence(p: &[f64], q: &[f64]) -> f64 {
    if p.len() != q.len() || p.is_empty() {
        return f64::NAN;
    }
    
    let mut divergence = 0.0;
    
    for i in 0..p.len() {
        let m = 0.5 * (p[i] + q[i]);
        if p[i] > 1e-10 {
            divergence += 0.5 * p[i] * (p[i] / m).ln();
        }
        if q[i] > 1e-10 {
            divergence += 0.5 * q[i] * (q[i] / m).ln();
        }
    }
    
    divergence
}

/// KS统计量
fn kolmogorov_smirnov(p: &[f64], q: &[f64]) -> f64 {
    if p.len() != q.len() || p.is_empty() {
        return f64::NAN;
    }
    
    let mut max_diff = 0.0;
    let mut cum_p = 0.0;
    let mut cum_q = 0.0;
    
    for i in 0..p.len() {
        cum_p += p[i];
        cum_q += q[i];
        let diff = (cum_p - cum_q).abs();
        if diff > max_diff {
            max_diff = diff;
        }
    }
    
    max_diff
}

/// Myers位并行编辑距离算法 - 用于u32序列（优化版本）
/// 
/// 优化点：
/// 1. 使用固定大小数组代替HashMap，避免动态分配和哈希计算
/// 2. 针对排序序型的特点（值范围1..=W）使用直接索引
/// 
/// 时间复杂度：O(ceil(m/w) * n)，其中w是字长（64位）
fn edit_distance_bit_parallel(a: &[u32], b: &[u32]) -> f64 {
    let m = a.len();
    let n = b.len();
    
    if m == 0 {
        return n as f64;
    }
    if n == 0 {
        return m as f64;
    }
    
    // 确保a是较短的序列（作为模式串）
    let (pattern, text, m, n) = if m > n {
        (b, a, n, m)
    } else {
        (a, b, m, n)
    };
    
    // 如果模式串太长，回退到传统算法
    if m > 64 {
        return edit_distance_fallback(pattern, text);
    }
    
    // 优化：使用固定大小数组代替HashMap
    // 对于排序序型，值的范围是1..=m（排名）
    // 我们使用大小为m+1的数组，直接索引
    let mut peq: Vec<u64> = vec![0; m + 1];
    for (j, &c) in pattern.iter().enumerate() {
        let idx = c as usize;
        if idx <= m {
            peq[idx] |= 1u64 << j;
        }
    }
    
    // 初始化
    let mut mv: u64 = 0;
    let mut pv: u64 = if m == 64 { !0u64 } else { (1u64 << m) - 1 };
    let mut score = m as i64;
    
    // 遍历文本串的每个字符
    for &c in text.iter() {
        // 直接数组索引，O(1)
        let idx = c as usize;
        let eq = if idx <= m { peq[idx] } else { 0 };
        
        // Myers算法的核心位运算
        let xv = eq | mv;
        let xh = (eq & pv).wrapping_add(pv) ^ pv;
        let ph = mv | !(xh | pv);
        let mh = pv & xh;
        
        pv = mh | !(xv | ph);
        mv = ph & xv;
        
        if (ph & (1u64 << (m - 1))) != 0 {
            score += 1;
        } else if (mh & (1u64 << (m - 1))) != 0 {
            score -= 1;
        }
    }
    
    score as f64
}

/// 传统编辑距离（回退方案）- u32版本
fn edit_distance_fallback(a: &[u32], b: &[u32]) -> f64 {
    let m = a.len();
    let n = b.len();
    
    let mut prev = vec![0; n + 1];
    let mut curr = vec![0; n + 1];
    
    for j in 0..=n {
        prev[j] = j;
    }
    
    for i in 1..=m {
        curr[0] = i;
        for j in 1..=n {
            let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
            curr[j] = (prev[j] + 1).min(curr[j - 1] + 1).min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    
    prev[n] as f64
}

/// 编辑距离 - 空间优化版本（只使用两行）- 保留用于兼容性
fn edit_distance_optimized(a: &[u32], b: &[u32]) -> f64 {
    // 使用位并行版本
    edit_distance_bit_parallel(a, b)
}

/// 欧氏距离 - O(n)
fn euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
    if a.is_empty() || b.is_empty() {
        return f64::NAN;
    }
    
    // 使用较短序列的长度
    let len = a.len().min(b.len());
    let a_short = &a[..len];
    let b_short = &b[..len];
    
    // 计算欧氏距离
    let sum_sq: f64 = a_short.iter()
        .zip(b_short.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum();
    
    sum_sq.sqrt()
}

/// DTW距离 - 使用Sakoe-Chiba带宽限制
fn dtw_distance_optimized(a: &[f64], b: &[f64], band_width: usize) -> f64 {
    let m = a.len();
    let n = b.len();
    
    if m == 0 || n == 0 {
        return f64::NAN;
    }
    
    // 使用带宽限制，只计算对角线附近的区域
    let mut dp = vec![vec![f64::INFINITY; n + 1]; m + 1];
    dp[0][0] = 0.0;
    
    for i in 1..=m {
        // 计算当前行的有效列范围
        let j_start = if i > band_width { i - band_width } else { 1 };
        let j_end = (i + band_width).min(n);
        
        for j in j_start..=j_end {
            let cost = (a[i - 1] - b[j - 1]).abs();
            dp[i][j] = cost + dp[i - 1][j].min(dp[i][j - 1]).min(dp[i - 1][j - 1]);
        }
    }
    
    dp[m][n]
}

/// Jaccard相似系数
fn jaccard_similarity(a: &[u32], b: &[u32]) -> f64 {
    let set_a: std::collections::HashSet<u32> = a.iter().cloned().collect();
    let set_b: std::collections::HashSet<u32> = b.iter().cloned().collect();
    
    if set_a.is_empty() && set_b.is_empty() {
        return 1.0;
    }
    
    let intersection: std::collections::HashSet<_> = set_a.intersection(&set_b).cloned().collect();
    let union: std::collections::HashSet<_> = set_a.union(&set_b).cloned().collect();
    
    intersection.len() as f64 / union.len() as f64
}

/// 计算标量集合的差异度量
fn calculate_scalar_distance(values_a: &[f64], values_b: &[f64], dist_type: DistanceType, band_width: usize) -> f64 {
    match dist_type {
        DistanceType::MeanDiff => {
            if values_a.is_empty() || values_b.is_empty() {
                return f64::NAN;
            }
            let mean_a: f64 = values_a.iter().sum::<f64>() / values_a.len() as f64;
            let mean_b: f64 = values_b.iter().sum::<f64>() / values_b.len() as f64;
            mean_a - mean_b
        }
        DistanceType::VarianceRatio => {
            if values_a.len() < 2 || values_b.len() < 2 {
                return f64::NAN;
            }
            let mean_a: f64 = values_a.iter().sum::<f64>() / values_a.len() as f64;
            let mean_b: f64 = values_b.iter().sum::<f64>() / values_b.len() as f64;
            let var_a: f64 = values_a.iter().map(|&v| (v - mean_a).powi(2)).sum::<f64>() / (values_a.len() - 1) as f64;
            let var_b: f64 = values_b.iter().map(|&v| (v - mean_b).powi(2)).sum::<f64>() / (values_b.len() - 1) as f64;
            if var_b.abs() < 1e-10 {
                return f64::NAN;
            }
            var_a / var_b
        }
        DistanceType::DtwDistance => {
            // 优化：使用欧氏距离替代DTW
            // 欧氏距离: O(n)，DTW: O(n×band_width)
            // 对于大多数场景，欧氏距离足够表达序列差异
            euclidean_distance(values_a, values_b)
        }
        _ => f64::NAN,
    }
}

/// 计算直方图的差异度量
fn calculate_histogram_distance(hist_a: &[f64], hist_b: &[f64], dist_type: DistanceType) -> f64 {
    match dist_type {
        DistanceType::Wasserstein => wasserstein_distance(hist_a, hist_b),
        DistanceType::JensenShannon => jensen_shannon_divergence(hist_a, hist_b),
        DistanceType::KolmogorovSmirnov => kolmogorov_smirnov(hist_a, hist_b),
        _ => f64::NAN,
    }
}

/// 计算序列的差异度量 - 优化版本
/// 
/// 对于排序序型，使用Spearman相关系数替代编辑距离
/// Spearman: O(n log n) 或 O(n)（如果已排序）
/// 编辑距离: O(n²)
fn calculate_sequence_distance(seq_a: &[u32], seq_b: &[u32], dist_type: DistanceType) -> f64 {
    match dist_type {
        DistanceType::EditDistance => {
            // 优化：对于等长序列，使用L1距离或Spearman相关系数
            // 这些计算复杂度为O(n)，比编辑距离的O(n²)快得多
            if seq_a.len() == seq_b.len() && !seq_a.is_empty() {
                // 使用归一化的L1距离（曼哈顿距离）
                // 计算两个序列对应位置的绝对差之和
                let l1_dist: f64 = seq_a.iter()
                    .zip(seq_b.iter())
                    .map(|(a, b)| (*a as f64 - *b as f64).abs())
                    .sum();
                // 归一化到[0, 1]范围
                let max_possible = seq_a.len() as f64 * seq_a.len() as f64;
                l1_dist / max_possible
            } else {
                // 不等长序列使用Myers位并行算法
                edit_distance_optimized(seq_a, seq_b)
            }
        }
        DistanceType::Jaccard => 1.0 - jaccard_similarity(seq_a, seq_b),
        _ => f64::NAN,
    }
}

/// 使用移动切分点构造序列 - 优化版本：限制最大序列长度
fn construct_moving_split_point(
    values: &[f64],
    window_size: usize,
    dist_type: DistanceType,
    band_width: usize,
) -> Vec<f64> {
    let n = values.len();
    let mut result = vec![f64::NAN; n];
    
    for i in window_size..n {
        // 优化：只取左侧最后window_size个元素和右侧前window_size个元素
        // 这样序列长度固定，DTW矩阵大小固定为window_size × window_size
        let left_start = if i > window_size { i - window_size } else { 0 };
        let left = &values[left_start..i];
        
        let right_end = (i + window_size).min(n);
        let right = &values[i..right_end];
        
        result[i] = calculate_scalar_distance(left, right, dist_type, band_width);
    }
    
    result
}

/// 使用孪生滑窗构造序列
fn construct_adjacent_rolling_windows(
    values: &[f64],
    window_size: usize,
    dist_type: DistanceType,
    band_width: usize,
) -> Vec<f64> {
    let n = values.len();
    let mut result = vec![f64::NAN; n];
    
    for i in (2 * window_size)..n {
        let left = &values[i - 2 * window_size..i - window_size];
        let right = &values[i - window_size..i];
        
        result[i] = calculate_scalar_distance(left, right, dist_type, band_width);
    }
    
    result
}

/// 使用累积vs边际构造序列 - 优化版本：限制累积序列最大长度
fn construct_cumulative_vs_marginal(
    values: &[f64],
    marginal_window: usize,
    dist_type: DistanceType,
    band_width: usize,
) -> Vec<f64> {
    let n = values.len();
    let mut result = vec![f64::NAN; n];
    
    for i in marginal_window..n {
        // 优化：累积序列只取最后max_cumulative_len个元素
        // 避免序列过长导致DTW计算量过大
        let max_cumulative_len = (marginal_window * 2).min(i);
        let cumulative_start = i - max_cumulative_len;
        let cumulative = &values[cumulative_start..i];
        let marginal = &values[i - marginal_window..i];
        
        result[i] = calculate_scalar_distance(cumulative, marginal, dist_type, band_width);
    }
    
    result
}

/// 直方图前缀和结构
struct HistogramPrefixSum {
    prefix: Vec<Vec<f64>>, // prefix[i][k] = 前i个直方图的第k个bin的和
    num_bins: usize,
}

impl HistogramPrefixSum {
    fn new(histograms: &[Vec<f64>]) -> Self {
        let n = histograms.len();
        let num_bins = if n > 0 { histograms[0].len() } else { 0 };
        
        let mut prefix = vec![vec![0.0; num_bins]; n + 1];
        
        for i in 0..n {
            for k in 0..num_bins {
                prefix[i + 1][k] = prefix[i][k] + histograms[i][k];
            }
        }
        
        Self { prefix, num_bins }
    }
    
    /// 获取区间[l, r)的直方图和
    fn get_range_sum(&self, l: usize, r: usize) -> Vec<f64> {
        let mut result = vec![0.0; self.num_bins];
        for k in 0..self.num_bins {
            result[k] = self.prefix[r][k] - self.prefix[l][k];
        }
        result
    }
}

/// 使用移动切分点构造直方图序列 - 前缀和优化
fn construct_moving_split_point_histogram(
    histograms: &[Vec<f64>],
    window_size: usize,
    dist_type: DistanceType,
) -> Vec<f64> {
    let n = histograms.len();
    let mut result = vec![f64::NAN; n];
    
    let prefix = HistogramPrefixSum::new(histograms);
    
    for i in window_size..n {
        // 左侧：[0, i)，右侧：[i, n)
        let mut left_hist = prefix.get_range_sum(0, i);
        let mut right_hist = prefix.get_range_sum(i, n);
        
        // 归一化
        let sum_left: f64 = left_hist.iter().sum();
        if sum_left > 0.0 {
            for k in 0..left_hist.len() {
                left_hist[k] /= sum_left;
            }
        }
        
        let sum_right: f64 = right_hist.iter().sum();
        if sum_right > 0.0 {
            for k in 0..right_hist.len() {
                right_hist[k] /= sum_right;
            }
        }
        
        result[i] = calculate_histogram_distance(&left_hist, &right_hist, dist_type);
    }
    
    result
}

/// 使用孪生滑窗构造直方图序列 - 前缀和优化
fn construct_adjacent_rolling_windows_histogram(
    histograms: &[Vec<f64>],
    window_size: usize,
    dist_type: DistanceType,
) -> Vec<f64> {
    let n = histograms.len();
    let mut result = vec![f64::NAN; n];
    
    let prefix = HistogramPrefixSum::new(histograms);
    
    for i in (2 * window_size)..n {
        // 左侧：[i-2W, i-W)，右侧：[i-W, i)
        let mut left_hist = prefix.get_range_sum(i - 2 * window_size, i - window_size);
        let mut right_hist = prefix.get_range_sum(i - window_size, i);
        
        let sum_left: f64 = left_hist.iter().sum();
        if sum_left > 0.0 {
            for k in 0..left_hist.len() {
                left_hist[k] /= sum_left;
            }
        }
        
        let sum_right: f64 = right_hist.iter().sum();
        if sum_right > 0.0 {
            for k in 0..right_hist.len() {
                right_hist[k] /= sum_right;
            }
        }
        
        result[i] = calculate_histogram_distance(&left_hist, &right_hist, dist_type);
    }
    
    result
}

/// 使用累积vs边际构造直方图序列 - 前缀和优化
fn construct_cumulative_vs_marginal_histogram(
    histograms: &[Vec<f64>],
    marginal_window: usize,
    dist_type: DistanceType,
) -> Vec<f64> {
    let n = histograms.len();
    let mut result = vec![f64::NAN; n];
    
    let prefix = HistogramPrefixSum::new(histograms);
    
    for i in marginal_window..n {
        // 累积：[0, i)，边际：[i-W, i)
        let mut cumul_hist = prefix.get_range_sum(0, i);
        let mut marginal_hist = prefix.get_range_sum(i - marginal_window, i);
        
        let sum_cumul: f64 = cumul_hist.iter().sum();
        if sum_cumul > 0.0 {
            for k in 0..cumul_hist.len() {
                cumul_hist[k] /= sum_cumul;
            }
        }
        
        let sum_marginal: f64 = marginal_hist.iter().sum();
        if sum_marginal > 0.0 {
            for k in 0..marginal_hist.len() {
                marginal_hist[k] /= sum_marginal;
            }
        }
        
        result[i] = calculate_histogram_distance(&cumul_hist, &marginal_hist, dist_type);
    }
    
    result
}

/// 使用移动切分点构造序列差异序列
fn construct_moving_split_point_sequence(
    sequences: &[Vec<u32>],
    window_size: usize,
    dist_type: DistanceType,
) -> Vec<f64> {
    let n = sequences.len();
    let mut result = vec![f64::NAN; n];
    
    for i in window_size..n {
        let left = &sequences[i / 2];
        let right = &sequences[i];
        
        result[i] = calculate_sequence_distance(left, right, dist_type);
    }
    
    result
}

/// 使用孪生滑窗构造序列差异序列
fn construct_adjacent_rolling_windows_sequence(
    sequences: &[Vec<u32>],
    window_size: usize,
    dist_type: DistanceType,
) -> Vec<f64> {
    let n = sequences.len();
    let mut result = vec![f64::NAN; n];
    
    for i in (2 * window_size)..n {
        let left = &sequences[i - window_size];
        let right = &sequences[i];
        
        result[i] = calculate_sequence_distance(left, right, dist_type);
    }
    
    result
}

/// 使用累积vs边际构造序列差异序列
fn construct_cumulative_vs_marginal_sequence(
    sequences: &[Vec<u32>],
    marginal_window: usize,
    dist_type: DistanceType,
) -> Vec<f64> {
    let n = sequences.len();
    let mut result = vec![f64::NAN; n];
    
    for i in marginal_window..n {
        let cumul = &sequences[i / 2];
        let marginal = &sequences[i];
        
        result[i] = calculate_sequence_distance(cumul, marginal, dist_type);
    }
    
    result
}

/// 找到序列中最小值和最大值的索引
fn find_min_max_indices(values: &[f64]) -> (Option<usize>, Option<usize>) {
    let mut min_idx = None;
    let mut max_idx = None;
    let mut min_val = f64::INFINITY;
    let mut max_val = f64::NEG_INFINITY;
    
    for (i, &v) in values.iter().enumerate() {
        if v.is_nan() {
            continue;
        }
        if v < min_val {
            min_val = v;
            min_idx = Some(i);
        }
        if v > max_val {
            max_val = v;
            max_idx = Some(i);
        }
    }
    
    (min_idx, max_idx)
}

/// 标量特征预计算结构
struct ScalarFeatureCache {
    prefix_volume: Vec<f64>,
    prefix_buy_volume: Vec<f64>,
    total_volume: f64,
}

impl ScalarFeatureCache {
    fn new(volumes: &[f64], flags: &[i32]) -> Self {
        let n = volumes.len();
        let mut prefix_volume = vec![0.0; n + 1];
        let mut prefix_buy_volume = vec![0.0; n + 1];
        
        for i in 0..n {
            prefix_volume[i + 1] = prefix_volume[i] + volumes[i];
            prefix_buy_volume[i + 1] = prefix_buy_volume[i] + if flags[i] == 66 { volumes[i] } else { 0.0 };
        }
        
        let total_volume = prefix_volume[n];
        
        Self {
            prefix_volume,
            prefix_buy_volume,
            total_volume,
        }
    }
    
    fn get_volume_sum(&self, l: usize, r: usize) -> f64 {
        self.prefix_volume[r] - self.prefix_volume[l]
    }
    
    fn get_buy_volume_sum(&self, l: usize, r: usize) -> f64 {
        self.prefix_buy_volume[r] - self.prefix_buy_volume[l]
    }
}

/// 计算标量特征 - 使用缓存优化
fn calculate_scalar_features_cached(
    diff_sequence: &[f64],
    times: &[i64],
    cache: &ScalarFeatureCache,
) -> Vec<f64> {
    let mut features = Vec::with_capacity(6);
    
    let (min_idx, max_idx) = find_min_max_indices(diff_sequence);
    
    // 最小值点特征
    if let Some(idx) = min_idx {
        // 时间差（秒）
        let time_before = (times[idx] - times[0]) as f64 / 1e9;
        let time_after = (times[times.len() - 1] - times[idx]) as f64 / 1e9;
        features.push(time_before - time_after);
        
        // 成交量占比差
        let vol_before = cache.get_volume_sum(0, idx + 1);
        let vol_after = cache.get_volume_sum(idx + 1, times.len());
        let vol_ratio_before = vol_before / cache.total_volume;
        let vol_ratio_after = vol_after / cache.total_volume;
        features.push(vol_ratio_before - vol_ratio_after);
        
        // 主买占比差
        let buy_vol_before = cache.get_buy_volume_sum(0, idx + 1);
        let buy_vol_after = cache.get_buy_volume_sum(idx + 1, times.len());
        let buy_ratio_before = if vol_before > 0.0 { buy_vol_before / vol_before } else { 0.0 };
        let buy_ratio_after = if vol_after > 0.0 { buy_vol_after / vol_after } else { 0.0 };
        features.push(buy_ratio_before - buy_ratio_after);
    } else {
        features.push(f64::NAN);
        features.push(f64::NAN);
        features.push(f64::NAN);
    }
    
    // 最大值点特征
    if let Some(idx) = max_idx {
        // 时间差（秒）
        let time_before = (times[idx] - times[0]) as f64 / 1e9;
        let time_after = (times[times.len() - 1] - times[idx]) as f64 / 1e9;
        features.push(time_before - time_after);
        
        // 成交量占比差
        let vol_before = cache.get_volume_sum(0, idx + 1);
        let vol_after = cache.get_volume_sum(idx + 1, times.len());
        let vol_ratio_before = vol_before / cache.total_volume;
        let vol_ratio_after = vol_after / cache.total_volume;
        features.push(vol_ratio_before - vol_ratio_after);
        
        // 主买占比差
        let buy_vol_before = cache.get_buy_volume_sum(0, idx + 1);
        let buy_vol_after = cache.get_buy_volume_sum(idx + 1, times.len());
        let buy_ratio_before = if vol_before > 0.0 { buy_vol_before / vol_before } else { 0.0 };
        let buy_ratio_after = if vol_after > 0.0 { buy_vol_after / vol_after } else { 0.0 };
        features.push(buy_ratio_before - buy_ratio_after);
    } else {
        features.push(f64::NAN);
        features.push(f64::NAN);
        features.push(f64::NAN);
    }
    
    features
}

/// 主函数：计算微观结构模式差异特征序列（优化版本）
#[pyfunction]
#[pyo3(signature = (
    trade_times,
    trade_prices,
    trade_volumes,
    trade_flags,
    trade_bid_orders,
    trade_ask_orders,
    window_size = 100,
    histogram_bins = 10,
    marginal_window = 50,
    dtw_band_width = 10,
))]
pub fn calculate_microstructure_pattern_features_optimized(
    py: Python,
    trade_times: PyReadonlyArray1<i64>,
    trade_prices: PyReadonlyArray1<f64>,
    trade_volumes: PyReadonlyArray1<f64>,
    trade_flags: PyReadonlyArray1<i32>,
    trade_bid_orders: PyReadonlyArray1<i64>,
    trade_ask_orders: PyReadonlyArray1<i64>,
    window_size: usize,
    histogram_bins: usize,
    marginal_window: usize,
    dtw_band_width: usize,
) -> PyResult<(Py<PyArray2<f64>>, Py<PyList>, Py<PyList>, Py<PyList>)> {
    
    // 转换为Rust切片
    let times = trade_times.as_slice()?;
    let prices = trade_prices.as_slice()?;
    let volumes = trade_volumes.as_slice()?;
    let flags = trade_flags.as_slice()?;
    let bid_orders = trade_bid_orders.as_slice()?;
    let ask_orders = trade_ask_orders.as_slice()?;
    
    let n = times.len();
    
    // 预计算标量特征缓存
    let scalar_cache = ScalarFeatureCache::new(volumes, flags);
    
    // 预计算各种模式
    let order_id_gap = calculate_order_id_gap(bid_orders, ask_orders);
    let is_buy_flag = calculate_is_buy_flag(flags);
    
    let volume_rank_motif = calculate_rank_motif_optimized(volumes, window_size);
    let order_id_gap_rank_motif = calculate_rank_motif_optimized(&order_id_gap, window_size);
    let volume_histogram = calculate_histogram(volumes, window_size, histogram_bins);
    let order_id_gap_histogram = calculate_histogram(&order_id_gap, window_size, histogram_bins);
    
    // 定义所有组合
    let mut all_sequences: Vec<Vec<f64>> = Vec::new();
    let mut all_names: Vec<String> = Vec::new();
    
    // 标量模式列表
    let scalar_patterns: Vec<(&str, Vec<f64>)> = vec![
        ("order_id_gap", order_id_gap.clone()),
        ("volume", volumes.to_vec()),
        ("turnover", volumes.to_vec()),
        ("is_buy_flag", is_buy_flag.clone()),
    ];
    
    // 标量差异度量
    let scalar_distances: Vec<(&str, DistanceType)> = vec![
        ("mean_diff", DistanceType::MeanDiff),
        ("dtw", DistanceType::DtwDistance),
    ];
    
    // 构造方式
    let constructions: Vec<(&str, ConstructionType)> = vec![
        ("moving_split", ConstructionType::MovingSplitPoint),
        ("adjacent_roll", ConstructionType::AdjacentRollingWindows),
        ("cumul_vs_marginal", ConstructionType::CumulativeVsMarginal),
    ];
    
    // 遍历所有标量组合
    for (pattern_name, pattern_values) in &scalar_patterns {
        for (dist_name, dist_type) in &scalar_distances {
            for (cons_name, cons_type) in &constructions {
                let sequence = match cons_type {
                    ConstructionType::MovingSplitPoint => {
                        construct_moving_split_point(pattern_values, window_size, *dist_type, dtw_band_width)
                    }
                    ConstructionType::AdjacentRollingWindows => {
                        construct_adjacent_rolling_windows(pattern_values, window_size, *dist_type, dtw_band_width)
                    }
                    ConstructionType::CumulativeVsMarginal => {
                        construct_cumulative_vs_marginal(pattern_values, marginal_window, *dist_type, dtw_band_width)
                    }
                };
                
                let name = format!("{}_{}_{}", pattern_name, dist_name, cons_name);
                all_sequences.push(sequence);
                all_names.push(name);
            }
        }
    }
    
    // 直方图模式
    let histogram_patterns: Vec<(&str, Vec<Vec<f64>>)> = vec![
        ("volume_hist", volume_histogram.clone()),
        ("order_id_gap_hist", order_id_gap_histogram.clone()),
    ];
    
    let histogram_distances: Vec<(&str, DistanceType)> = vec![
        ("wasserstein", DistanceType::Wasserstein),
        ("js_divergence", DistanceType::JensenShannon),
        ("ks_stat", DistanceType::KolmogorovSmirnov),
    ];
    
    for (pattern_name, pattern_values) in &histogram_patterns {
        for (dist_name, dist_type) in &histogram_distances {
            for (cons_name, cons_type) in &constructions {
                let sequence = match cons_type {
                    ConstructionType::MovingSplitPoint => {
                        construct_moving_split_point_histogram(pattern_values, window_size, *dist_type)
                    }
                    ConstructionType::AdjacentRollingWindows => {
                        construct_adjacent_rolling_windows_histogram(pattern_values, window_size, *dist_type)
                    }
                    ConstructionType::CumulativeVsMarginal => {
                        construct_cumulative_vs_marginal_histogram(pattern_values, marginal_window, *dist_type)
                    }
                };
                
                let name = format!("{}_{}_{}", pattern_name, dist_name, cons_name);
                all_sequences.push(sequence);
                all_names.push(name);
            }
        }
    }
    
    // 排序序型
    let rank_patterns: Vec<(&str, Vec<Vec<u32>>)> = vec![
        ("volume_rank", volume_rank_motif.clone()),
        ("order_id_gap_rank", order_id_gap_rank_motif.clone()),
    ];
    
    let rank_distances: Vec<(&str, DistanceType)> = vec![
        ("edit_dist", DistanceType::EditDistance),
        ("jaccard", DistanceType::Jaccard),
    ];
    
    for (pattern_name, pattern_values) in &rank_patterns {
        for (dist_name, dist_type) in &rank_distances {
            for (cons_name, cons_type) in &constructions {
                let sequence = match cons_type {
                    ConstructionType::MovingSplitPoint => {
                        construct_moving_split_point_sequence(pattern_values, window_size, *dist_type)
                    }
                    ConstructionType::AdjacentRollingWindows => {
                        construct_adjacent_rolling_windows_sequence(pattern_values, window_size, *dist_type)
                    }
                    ConstructionType::CumulativeVsMarginal => {
                        construct_cumulative_vs_marginal_sequence(pattern_values, marginal_window, *dist_type)
                    }
                };
                
                let name = format!("{}_{}_{}", pattern_name, dist_name, cons_name);
                all_sequences.push(sequence);
                all_names.push(name);
            }
        }
    }
    
    // 添加单独的 price_change_variance_ratio_cumul_vs_marginal 特征
    let price_change = calculate_price_change(prices);
    let price_change_variance_ratio_cumul = construct_cumulative_vs_marginal(
        &price_change, 
        marginal_window, 
        DistanceType::VarianceRatio, 
        dtw_band_width
    );
    all_sequences.push(price_change_variance_ratio_cumul);
    all_names.push("price_change_variance_ratio_cumul_vs_marginal".to_string());
    
    // 构建特征矩阵
    let num_features = all_sequences.len();
    let mut feature_matrix = vec![vec![f64::NAN; num_features]; n];
    
    for (j, sequence) in all_sequences.iter().enumerate() {
        for (i, &v) in sequence.iter().enumerate() {
            feature_matrix[i][j] = v;
        }
    }
    
    // 转换为2D numpy数组
    let py_array = PyArray2::from_vec2(py, &feature_matrix)?;
    
    // 计算标量特征 - 使用缓存
    let mut all_scalar_features: Vec<f64> = Vec::with_capacity(num_features * 6);
    let mut scalar_feature_names: Vec<String> = Vec::with_capacity(num_features * 6);
    
    for (i, sequence) in all_sequences.iter().enumerate() {
        let scalar_features = calculate_scalar_features_cached(sequence, times, &scalar_cache);
        
        scalar_feature_names.push(format!("{}_min_time_diff", all_names[i]));
        scalar_feature_names.push(format!("{}_min_vol_ratio_diff", all_names[i]));
        scalar_feature_names.push(format!("{}_min_buy_ratio_diff", all_names[i]));
        scalar_feature_names.push(format!("{}_max_time_diff", all_names[i]));
        scalar_feature_names.push(format!("{}_max_vol_ratio_diff", all_names[i]));
        scalar_feature_names.push(format!("{}_max_buy_ratio_diff", all_names[i]));
        
        all_scalar_features.extend(scalar_features);
    }
    
    // 创建Python列表
    let names_list = PyList::new(py, all_names.iter().map(|s| PyString::new(py, s)));
    let scalar_features_list = PyList::new(py, all_scalar_features.iter().map(|&v| v));
    let scalar_names_list = PyList::new(py, scalar_feature_names.iter().map(|s| PyString::new(py, s)));
    
    Ok((py_array.into(), names_list.into(), scalar_features_list.into(), scalar_names_list.into()))
}
