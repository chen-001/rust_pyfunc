//! 微观结构模式差异特征序列计算
//! 
//! 实现三步法构建日内特征序列：
//! 1. 模式定义：从逐笔成交和盘口快照中提取微观结构模式
//! 2. 差异度量：计算两个时间片段之间的模式差异
//! 3. 序列构造：通过移动时间窗生成日内特征序列

use numpy::{PyArray2, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::types::{PyList, PyString};

/// 差异度量类型枚举
#[derive(Clone, Copy, Debug)]
enum DistanceType {
    Wasserstein,      // Wasserstein距离
    JensenShannon,    // Jensen-Shannon散度
    KolmogorovSmirnov,// KS统计量
    EditDistance,     // 编辑距离
    DtwDistance,      // 动态时间规整距离
    Jaccard,          // Jaccard相似系数
    MeanDiff,         // 均值差异
    VarianceRatio,    // 方差比
    SpearmanCorr,     // Spearman秩相关
}

/// 序列构造方式枚举
#[derive(Clone, Copy, Debug)]
enum ConstructionType {
    MovingSplitPoint,      // 移动切分点
    AdjacentRollingWindows,// 孪生滑窗
    CumulativeVsMarginal,  // 累积vs边际
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

/// 计算是否主买标志 (66=主买->1, 83=主卖->0, 其他->0)
fn calculate_is_buy_flag(flags: &[i32]) -> Vec<f64> {
    flags.iter()
        .map(|&f| if f == 66 { 1.0 } else { 0.0 })
        .collect()
}

/// 计算排序序型（返回每个元素在窗口内的排名）
fn calculate_rank_motif(values: &[f64], window_size: usize) -> Vec<Vec<u32>> {
    let mut result = Vec::new();
    
    for i in 0..values.len() {
        let start = if i + 1 >= window_size { i + 1 - window_size } else { 0 };
        let window = &values[start..=i];
        
        // 计算排名（1-based）
        let mut indexed: Vec<(usize, f64)> = window.iter().enumerate().map(|(j, &v)| (j, v)).collect();
        indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        
        let mut ranks = vec![0u32; window.len()];
        for (rank, (idx, _)) in indexed.iter().enumerate() {
            ranks[*idx] = (rank + 1) as u32;
        }
        
        result.push(ranks);
    }
    
    result
}

/// 计算价格变动方向序列 (-1, 0, 1)
fn calculate_price_direction_seq(prices: &[f64], window_size: usize) -> Vec<Vec<i32>> {
    let price_changes = calculate_price_change(prices);
    let mut result = Vec::new();
    
    for i in 0..price_changes.len() {
        let start = if i + 1 >= window_size { i + 1 - window_size } else { 0 };
        let window: Vec<i32> = price_changes[start..=i].iter()
            .map(|&v| {
                if v > 1e-10 { 1 }
                else if v < -1e-10 { -1 }
                else { 0 }
            })
            .collect();
        result.push(window);
    }
    
    result
}

/// 计算直方图（等宽分箱）
fn calculate_histogram(values: &[f64], window_size: usize, num_bins: usize) -> Vec<Vec<f64>> {
    let mut result = Vec::new();
    
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
        
        // 归一化
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

/// Wasserstein距离（一维）
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

/// 编辑距离（用于整数序列）
fn edit_distance(a: &[u32], b: &[u32]) -> f64 {
    let m = a.len();
    let n = b.len();
    
    if m == 0 {
        return n as f64;
    }
    if n == 0 {
        return m as f64;
    }
    
    let mut dp = vec![vec![0; n + 1]; m + 1];
    
    for i in 0..=m {
        dp[i][0] = i;
    }
    for j in 0..=n {
        dp[0][j] = j;
    }
    
    for i in 1..=m {
        for j in 1..=n {
            let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
            dp[i][j] = (dp[i - 1][j] + 1)
                .min(dp[i][j - 1] + 1)
                .min(dp[i - 1][j - 1] + cost);
        }
    }
    
    dp[m][n] as f64
}

/// 编辑距离（用于方向序列）
fn edit_distance_i32(a: &[i32], b: &[i32]) -> f64 {
    let m = a.len();
    let n = b.len();
    
    if m == 0 {
        return n as f64;
    }
    if n == 0 {
        return m as f64;
    }
    
    let mut dp = vec![vec![0; n + 1]; m + 1];
    
    for i in 0..=m {
        dp[i][0] = i;
    }
    for j in 0..=n {
        dp[0][j] = j;
    }
    
    for i in 1..=m {
        for j in 1..=n {
            let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
            dp[i][j] = (dp[i - 1][j] + 1)
                .min(dp[i][j - 1] + 1)
                .min(dp[i - 1][j - 1] + cost);
        }
    }
    
    dp[m][n] as f64
}

/// 动态时间规整距离（简化版）
fn dtw_distance(a: &[f64], b: &[f64]) -> f64 {
    let m = a.len();
    let n = b.len();
    
    if m == 0 || n == 0 {
        return f64::NAN;
    }
    
    let mut dp = vec![vec![f64::INFINITY; n + 1]; m + 1];
    dp[0][0] = 0.0;
    
    for i in 1..=m {
        for j in 1..=n {
            let cost = (a[i - 1] - b[j - 1]).abs();
            dp[i][j] = cost + dp[i - 1][j].min(dp[i][j - 1]).min(dp[i - 1][j - 1]);
        }
    }
    
    dp[m][n]
}

/// Jaccard相似系数（用于集合）
fn jaccard_similarity(a: &[u32], b: &[u32]) -> f64 {
    let set_a: std::collections::HashSet<u32> = a.iter().cloned().collect();
    let set_b: std::collections::HashSet<u32> = b.iter().cloned().collect();
    
    let intersection: std::collections::HashSet<_> = set_a.intersection(&set_b).cloned().collect();
    let union: std::collections::HashSet<_> = set_a.union(&set_b).cloned().collect();
    
    if union.is_empty() {
        return 0.0;
    }
    
    intersection.len() as f64 / union.len() as f64
}

/// Jaccard相似系数（用于方向序列）
fn jaccard_similarity_i32(a: &[i32], b: &[i32]) -> f64 {
    let set_a: std::collections::HashSet<i32> = a.iter().cloned().collect();
    let set_b: std::collections::HashSet<i32> = b.iter().cloned().collect();
    
    let intersection: std::collections::HashSet<_> = set_a.intersection(&set_b).cloned().collect();
    let union: std::collections::HashSet<_> = set_a.union(&set_b).cloned().collect();
    
    if union.is_empty() {
        return 0.0;
    }
    
    intersection.len() as f64 / union.len() as f64
}

/// 计算Spearman秩相关系数
fn spearman_correlation(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() || a.len() < 2 {
        return f64::NAN;
    }
    
    let n = a.len() as f64;
    
    // 计算排名
    let rank_a = get_ranks(a);
    let rank_b = get_ranks(b);
    
    // 计算相关系数
    let mean_rank = (n + 1.0) / 2.0;
    let mut numerator = 0.0;
    let mut denom_a = 0.0;
    let mut denom_b = 0.0;
    
    for i in 0..a.len() {
        let diff_a = rank_a[i] - mean_rank;
        let diff_b = rank_b[i] - mean_rank;
        numerator += diff_a * diff_b;
        denom_a += diff_a * diff_a;
        denom_b += diff_b * diff_b;
    }
    
    let denominator = denom_a.sqrt() * denom_b.sqrt();
    if denominator < 1e-10 {
        return 0.0;
    }
    
    numerator / denominator
}

/// 获取排名（1-based）
fn get_ranks(values: &[f64]) -> Vec<f64> {
    let mut indexed: Vec<(usize, f64)> = values.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    
    let mut ranks = vec![0.0; values.len()];
    for (rank, (idx, _)) in indexed.iter().enumerate() {
        ranks[*idx] = (rank + 1) as f64;
    }
    
    ranks
}

/// 计算标量集合的差异度量
fn calculate_scalar_distance(values_a: &[f64], values_b: &[f64], dist_type: DistanceType) -> f64 {
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
        DistanceType::SpearmanCorr => {
            if values_a.len() != values_b.len() {
                return f64::NAN;
            }
            spearman_correlation(values_a, values_b)
        }
        DistanceType::DtwDistance => {
            dtw_distance(values_a, values_b)
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

/// 计算序列的差异度量
fn calculate_sequence_distance(seq_a: &[u32], seq_b: &[u32], dist_type: DistanceType) -> f64 {
    match dist_type {
        DistanceType::EditDistance => edit_distance(seq_a, seq_b),
        DistanceType::Jaccard => 1.0 - jaccard_similarity(seq_a, seq_b), // 转换为距离
        _ => f64::NAN,
    }
}

/// 计算方向序列的差异度量
fn calculate_direction_distance(seq_a: &[i32], seq_b: &[i32], dist_type: DistanceType) -> f64 {
    match dist_type {
        DistanceType::EditDistance => edit_distance_i32(seq_a, seq_b),
        DistanceType::Jaccard => 1.0 - jaccard_similarity_i32(seq_a, seq_b),
        _ => f64::NAN,
    }
}

/// 使用移动切分点构造序列
fn construct_moving_split_point(
    values: &[f64],
    window_size: usize,
    dist_type: DistanceType,
) -> Vec<f64> {
    let n = values.len();
    let mut result = vec![f64::NAN; n];
    
    for i in window_size..n {
        let left = &values[0..i];
        let right = &values[i..n];
        
        result[i] = calculate_scalar_distance(left, right, dist_type);
    }
    
    result
}

/// 使用孪生滑窗构造序列
fn construct_adjacent_rolling_windows(
    values: &[f64],
    window_size: usize,
    dist_type: DistanceType,
) -> Vec<f64> {
    let n = values.len();
    let mut result = vec![f64::NAN; n];
    
    for i in (2 * window_size)..n {
        let left = &values[i - 2 * window_size..i - window_size];
        let right = &values[i - window_size..i];
        
        result[i] = calculate_scalar_distance(left, right, dist_type);
    }
    
    result
}

/// 使用累积vs边际构造序列
fn construct_cumulative_vs_marginal(
    values: &[f64],
    marginal_window: usize,
    dist_type: DistanceType,
) -> Vec<f64> {
    let n = values.len();
    let mut result = vec![f64::NAN; n];
    
    for i in marginal_window..n {
        let cumulative = &values[0..i];
        let marginal = &values[i - marginal_window..i];
        
        result[i] = calculate_scalar_distance(cumulative, marginal, dist_type);
    }
    
    result
}

/// 使用移动切分点构造直方图序列
fn construct_moving_split_point_histogram(
    histograms: &[Vec<f64>],
    window_size: usize,
    dist_type: DistanceType,
) -> Vec<f64> {
    let n = histograms.len();
    let mut result = vec![f64::NAN; n];
    
    for i in window_size..n {
        // 合并左侧所有直方图
        let num_bins = histograms[0].len();
        let mut left_hist = vec![0.0; num_bins];
        for j in 0..i {
            for k in 0..num_bins {
                left_hist[k] += histograms[j][k];
            }
        }
        let sum_left: f64 = left_hist.iter().sum();
        if sum_left > 0.0 {
            for k in 0..num_bins {
                left_hist[k] /= sum_left;
            }
        }
        
        // 合并右侧所有直方图
        let mut right_hist = vec![0.0; num_bins];
        for j in i..n {
            for k in 0..num_bins {
                right_hist[k] += histograms[j][k];
            }
        }
        let sum_right: f64 = right_hist.iter().sum();
        if sum_right > 0.0 {
            for k in 0..num_bins {
                right_hist[k] /= sum_right;
            }
        }
        
        result[i] = calculate_histogram_distance(&left_hist, &right_hist, dist_type);
    }
    
    result
}

/// 使用孪生滑窗构造直方图序列
fn construct_adjacent_rolling_windows_histogram(
    histograms: &[Vec<f64>],
    window_size: usize,
    dist_type: DistanceType,
) -> Vec<f64> {
    let n = histograms.len();
    let mut result = vec![f64::NAN; n];
    let num_bins = histograms[0].len();
    
    for i in (2 * window_size)..n {
        // 合并左侧窗口直方图
        let mut left_hist = vec![0.0; num_bins];
        for j in (i - 2 * window_size)..(i - window_size) {
            for k in 0..num_bins {
                left_hist[k] += histograms[j][k];
            }
        }
        let sum_left: f64 = left_hist.iter().sum();
        if sum_left > 0.0 {
            for k in 0..num_bins {
                left_hist[k] /= sum_left;
            }
        }
        
        // 合并右侧窗口直方图
        let mut right_hist = vec![0.0; num_bins];
        for j in (i - window_size)..i {
            for k in 0..num_bins {
                right_hist[k] += histograms[j][k];
            }
        }
        let sum_right: f64 = right_hist.iter().sum();
        if sum_right > 0.0 {
            for k in 0..num_bins {
                right_hist[k] /= sum_right;
            }
        }
        
        result[i] = calculate_histogram_distance(&left_hist, &right_hist, dist_type);
    }
    
    result
}

/// 使用累积vs边际构造直方图序列
fn construct_cumulative_vs_marginal_histogram(
    histograms: &[Vec<f64>],
    marginal_window: usize,
    dist_type: DistanceType,
) -> Vec<f64> {
    let n = histograms.len();
    let mut result = vec![f64::NAN; n];
    let num_bins = histograms[0].len();
    
    for i in marginal_window..n {
        // 合并累积直方图
        let mut cumul_hist = vec![0.0; num_bins];
        for j in 0..i {
            for k in 0..num_bins {
                cumul_hist[k] += histograms[j][k];
            }
        }
        let sum_cumul: f64 = cumul_hist.iter().sum();
        if sum_cumul > 0.0 {
            for k in 0..num_bins {
                cumul_hist[k] /= sum_cumul;
            }
        }
        
        // 合并边际窗口直方图
        let mut marginal_hist = vec![0.0; num_bins];
        for j in (i - marginal_window)..i {
            for k in 0..num_bins {
                marginal_hist[k] += histograms[j][k];
            }
        }
        let sum_marginal: f64 = marginal_hist.iter().sum();
        if sum_marginal > 0.0 {
            for k in 0..num_bins {
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
        // 使用中间序列作为代表
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

/// 使用移动切分点构造方向序列差异序列
fn construct_moving_split_point_direction(
    sequences: &[Vec<i32>],
    window_size: usize,
    dist_type: DistanceType,
) -> Vec<f64> {
    let n = sequences.len();
    let mut result = vec![f64::NAN; n];
    
    for i in window_size..n {
        let left = &sequences[i / 2];
        let right = &sequences[i];
        
        result[i] = calculate_direction_distance(left, right, dist_type);
    }
    
    result
}

/// 使用孪生滑窗构造方向序列差异序列
fn construct_adjacent_rolling_windows_direction(
    sequences: &[Vec<i32>],
    window_size: usize,
    dist_type: DistanceType,
) -> Vec<f64> {
    let n = sequences.len();
    let mut result = vec![f64::NAN; n];
    
    for i in (2 * window_size)..n {
        let left = &sequences[i - window_size];
        let right = &sequences[i];
        
        result[i] = calculate_direction_distance(left, right, dist_type);
    }
    
    result
}

/// 使用累积vs边际构造方向序列差异序列
fn construct_cumulative_vs_marginal_direction(
    sequences: &[Vec<i32>],
    marginal_window: usize,
    dist_type: DistanceType,
) -> Vec<f64> {
    let n = sequences.len();
    let mut result = vec![f64::NAN; n];
    
    for i in marginal_window..n {
        let cumul = &sequences[i / 2];
        let marginal = &sequences[i];
        
        result[i] = calculate_direction_distance(cumul, marginal, dist_type);
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

/// 计算标量特征
fn calculate_scalar_features(
    diff_sequence: &[f64],
    times: &[i64],
    volumes: &[f64],
    flags: &[i32],
) -> Vec<f64> {
    let mut features = Vec::new();
    
    let (min_idx, max_idx) = find_min_max_indices(diff_sequence);
    
    // 计算总成交量
    let total_volume: f64 = volumes.iter().sum();
    
    // 最小值点特征
    if let Some(idx) = min_idx {
        // 时间差（秒）
        let time_before = (times[idx] - times[0]) as f64 / 1e9;
        let time_after = (times[times.len() - 1] - times[idx]) as f64 / 1e9;
        features.push(time_before - time_after);
        
        // 成交量占比差
        let vol_before: f64 = volumes[0..=idx].iter().sum();
        let vol_after: f64 = volumes[idx + 1..].iter().sum();
        let vol_ratio_before = vol_before / total_volume;
        let vol_ratio_after = vol_after / total_volume;
        features.push(vol_ratio_before - vol_ratio_after);
        
        // 主买占比差
        let buy_vol_before: f64 = volumes[0..=idx].iter()
            .zip(flags[0..=idx].iter())
            .filter(|(_, &f)| f == 66)
            .map(|(v, _)| v)
            .sum();
        let buy_vol_after: f64 = volumes[idx + 1..].iter()
            .zip(flags[idx + 1..].iter())
            .filter(|(_, &f)| f == 66)
            .map(|(v, _)| v)
            .sum();
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
        let vol_before: f64 = volumes[0..=idx].iter().sum();
        let vol_after: f64 = volumes[idx + 1..].iter().sum();
        let vol_ratio_before = vol_before / total_volume;
        let vol_ratio_after = vol_after / total_volume;
        features.push(vol_ratio_before - vol_ratio_after);
        
        // 主买占比差
        let buy_vol_before: f64 = volumes[0..=idx].iter()
            .zip(flags[0..=idx].iter())
            .filter(|(_, &f)| f == 66)
            .map(|(v, _)| v)
            .sum();
        let buy_vol_after: f64 = volumes[idx + 1..].iter()
            .zip(flags[idx + 1..].iter())
            .filter(|(_, &f)| f == 66)
            .map(|(v, _)| v)
            .sum();
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

/// 主函数：计算微观结构模式差异特征序列
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
))]
pub fn calculate_microstructure_pattern_features(
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
) -> PyResult<(Py<PyArray2<f64>>, Py<PyList>, Py<PyList>, Py<PyList>)> {
    
    // 转换为Rust切片
    let times = trade_times.as_slice()?;
    let prices = trade_prices.as_slice()?;
    let volumes = trade_volumes.as_slice()?;
    let flags = trade_flags.as_slice()?;
    let bid_orders = trade_bid_orders.as_slice()?;
    let ask_orders = trade_ask_orders.as_slice()?;
    
    let n = times.len();
    
    // 预计算各种模式
    let order_id_gap = calculate_order_id_gap(bid_orders, ask_orders);
    let price_change = calculate_price_change(prices);
    let is_buy_flag = calculate_is_buy_flag(flags);
    
    let volume_rank_motif = calculate_rank_motif(volumes, window_size);
    let order_id_gap_rank_motif = calculate_rank_motif(&order_id_gap, window_size);
    let price_direction_seq = calculate_price_direction_seq(prices, window_size);
    let volume_histogram = calculate_histogram(volumes, window_size, histogram_bins);
    let order_id_gap_histogram = calculate_histogram(&order_id_gap, window_size, histogram_bins);
    
    // 定义所有组合
    let mut all_sequences: Vec<Vec<f64>> = Vec::new();
    let mut all_names: Vec<String> = Vec::new();
    
    // ===== 标量模式 × 差异度量 × 构造方式 =====
    
    // 标量模式列表
    let scalar_patterns: Vec<(&str, Vec<f64>)> = vec![
        ("order_id_gap", order_id_gap.clone()),
        ("volume", volumes.to_vec()),
        ("turnover", volumes.to_vec()), // 使用volume代替，实际应该用turnover
        ("price_change", price_change.clone()),
        ("is_buy_flag", is_buy_flag.clone()),
    ];
    
    // 标量差异度量
    let scalar_distances: Vec<(&str, DistanceType)> = vec![
        ("mean_diff", DistanceType::MeanDiff),
        ("variance_ratio", DistanceType::VarianceRatio),
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
                        construct_moving_split_point(pattern_values, window_size, *dist_type)
                    }
                    ConstructionType::AdjacentRollingWindows => {
                        construct_adjacent_rolling_windows(pattern_values, window_size, *dist_type)
                    }
                    ConstructionType::CumulativeVsMarginal => {
                        construct_cumulative_vs_marginal(pattern_values, marginal_window, *dist_type)
                    }
                };
                
                let name = format!("{}_{}_{}", pattern_name, dist_name, cons_name);
                all_sequences.push(sequence);
                all_names.push(name);
            }
        }
    }
    
    // ===== 直方图模式 × 差异度量 × 构造方式 =====
    
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
    
    // ===== 排序序型 × 差异度量 × 构造方式 =====
    
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
    
    // ===== 方向序列 × 差异度量 × 构造方式 =====
    
    let direction_patterns: Vec<(&str, Vec<Vec<i32>>)> = vec![
        ("price_direction", price_direction_seq.clone()),
    ];
    
    for (pattern_name, pattern_values) in &direction_patterns {
        for (dist_name, dist_type) in &rank_distances {
            for (cons_name, cons_type) in &constructions {
                let sequence = match cons_type {
                    ConstructionType::MovingSplitPoint => {
                        construct_moving_split_point_direction(pattern_values, window_size, *dist_type)
                    }
                    ConstructionType::AdjacentRollingWindows => {
                        construct_adjacent_rolling_windows_direction(pattern_values, window_size, *dist_type)
                    }
                    ConstructionType::CumulativeVsMarginal => {
                        construct_cumulative_vs_marginal_direction(pattern_values, marginal_window, *dist_type)
                    }
                };
                
                let name = format!("{}_{}_{}", pattern_name, dist_name, cons_name);
                all_sequences.push(sequence);
                all_names.push(name);
            }
        }
    }
    
    // 构建特征矩阵 (n × num_features)
    let num_features = all_sequences.len();
    let mut feature_matrix = vec![vec![f64::NAN; num_features]; n];
    
    for (j, sequence) in all_sequences.iter().enumerate() {
        for (i, &v) in sequence.iter().enumerate() {
            feature_matrix[i][j] = v;
        }
    }
    
    // 转换为2D numpy数组
    let py_array = PyArray2::from_vec2(py, &feature_matrix)?;
    
    // 计算标量特征
    let mut all_scalar_features: Vec<f64> = Vec::new();
    let mut scalar_feature_names: Vec<String> = Vec::new();
    
    for (i, sequence) in all_sequences.iter().enumerate() {
        let scalar_features = calculate_scalar_features(sequence, times, volumes, flags);
        
        // 添加特征名
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
