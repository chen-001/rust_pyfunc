/// GP 相关维数 (D₂) 计算模块
///
/// 实现完全确定性的 Grassberger-Procaccia 算法
/// 所有中间参数（τ、m、Theiler窗口、半径网格与拟合区间）均由库内部自动确定
/// 无随机性、无采样、结果可复现
///
/// 最终输出指标是 **相关维数 (D₂)**
/// 在 log C(r) = D₂ log r + const 中，线性段斜率即为 D₂

use pyo3::prelude::*;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use std::f64::consts::E;
use std::collections::HashMap;

/// 全局欧几里得距离函数（带优化）
fn euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() {
        return f64::INFINITY;
    }

    // 对于长向量，使用分块计算提高缓存效率
    const CHUNK_SIZE: usize = 64;
    let mut sum_squares = 0.0;

    if a.len() > CHUNK_SIZE {
        // 分块计算大向量
        for (chunk_a, chunk_b) in a.chunks(CHUNK_SIZE).zip(b.chunks(CHUNK_SIZE)) {
            let chunk_sum: f64 = chunk_a.iter()
                .zip(chunk_b.iter())
                .map(|(x, y)| {
                    let diff = x - y;
                    diff * diff
                })
                .sum();
            sum_squares += chunk_sum;
        }
    } else {
        // 小向量直接计算
        sum_squares = a.iter()
            .zip(b.iter())
            .map(|(x, y)| {
                let diff = x - y;
                diff * diff
            })
            .sum();
    }

    sum_squares.sqrt()
}

/// GP 相关维数计算选项（所有参数均有确定默认值）
#[derive(Debug, Clone)]
#[pyclass]
pub struct GpOptions {
    // AMI（平均互信息）与 τ 选择
    #[pyo3(get)]
    pub ami_max_lag: usize,       // 默认：min(200, len(x)/10)
    #[pyo3(get)]
    pub ami_n_bins: usize,        // 默认：32
    #[pyo3(get)]
    pub ami_quantile_bins: bool,  // 默认：true（分位数分箱，确定性）
    #[pyo3(get)]
    pub tau_override: Option<usize>, // 默认 None：走自动选择

    // FNN（假近邻）与 m 选择
    #[pyo3(get)]
    pub fnn_m_max: usize,         // 默认：12
    #[pyo3(get)]
    pub fnn_rtol: f64,            // 默认：10.0
    #[pyo3(get)]
    pub fnn_atol: f64,            // 默认：2.0
    #[pyo3(get)]
    pub fnn_threshold: f64,       // 默认：0.02
    #[pyo3(get)]
    pub m_override: Option<usize>,// 默认 None：走自动选择

    // Theiler 窗口
    #[pyo3(get)]
    pub theiler_override: Option<usize>, // 默认 None：走自动选择

    // 半径网格 r
    #[pyo3(get)]
    pub n_r: usize,               // 默认：48（对数均匀取点）
    #[pyo3(get)]
    pub r_percentile_lo: f64,     // 默认：5.0
    #[pyo3(get)]
    pub r_percentile_hi: f64,     // 默认：90.0

    // 线性段选择（在 log C – log r 上）
    #[pyo3(get)]
    pub fit_min_len: usize,       // 默认：6
    #[pyo3(get)]
    pub fit_max_len: usize,       // 默认：14
    #[pyo3(get)]
    pub c_lo: f64,                // 默认：1e-4（排除极小）
    #[pyo3(get)]
    pub c_hi: f64,                // 默认：1.0 - 1e-3（排除饱和）
    #[pyo3(get)]
    pub stability_alpha: f64,     // 默认：0.05（评分：R² - alpha*局部斜率std）
}

impl Default for GpOptions {
    fn default() -> Self {
        Self {
            ami_max_lag: 200,
            ami_n_bins: 32,
            ami_quantile_bins: true,
            tau_override: None,
            fnn_m_max: 12,
            fnn_rtol: 10.0,
            fnn_atol: 2.0,
            fnn_threshold: 0.02,
            m_override: None,
            theiler_override: None,
            n_r: 48,
            r_percentile_lo: 5.0,
            r_percentile_hi: 90.0,
            fit_min_len: 6,
            fit_max_len: 14,
            c_lo: 1e-4,
            c_hi: 1.0 - 1e-3,
            stability_alpha: 0.05,
        }
    }
}

/// GP 相关维数计算结果
#[derive(Debug, Clone)]
#[pyclass]
pub struct GpResult {
    #[pyo3(get)]
    pub tau: usize,               // 自动选出的 τ
    #[pyo3(get)]
    pub m: usize,                 // 自动选出的 m
    #[pyo3(get)]
    pub theiler: usize,           // 自动选出的 Theiler 窗口

    #[pyo3(get)]
    pub rs: Vec<f64>,             // 半径序列
    #[pyo3(get)]
    pub cs: Vec<f64>,             // 对应 C(r)
    #[pyo3(get)]
    pub log_r: Vec<f64>,
    #[pyo3(get)]
    pub log_c: Vec<f64>,

    #[pyo3(get)]
    pub local_slopes: Vec<Option<f64>>, // 局部滑窗斜率

    // 自动选择的线性段（闭开区间）与拟合结果
    #[pyo3(get)]
    pub fit_start: usize,
    #[pyo3(get)]
    pub fit_end: usize,
    #[pyo3(get)]
    pub d2_est: f64,              // 相关维数估计（斜率）
    #[pyo3(get)]
    pub fit_intercept: f64,
    #[pyo3(get)]
    pub fit_r2: f64,

    // 诊断输出（便于可视化/审核）
    #[pyo3(get)]
    pub ami_lags: Vec<usize>,
    #[pyo3(get)]
    pub ami_values: Vec<f64>,
    #[pyo3(get)]
    pub fnn_ms: Vec<usize>,
    #[pyo3(get)]
    pub fnn_ratios: Vec<f64>,
}

/// GP 相关维数计算错误类型
#[derive(Debug, thiserror::Error)]
pub enum GpError {
    #[error("输入序列过短，需要至少 {min_len} 个点，得到 {actual_len}")]
    InputTooShort { min_len: usize, actual_len: usize },

    #[error("输入序列过于恒定，标准差为 0")]
    InputTooConstant,

    #[error("无效的 τ 或 m 组合：(m-1)*τ ≥ 序列长度")]
    InvalidTauOrM,

    #[error("KD-Tree 构建失败")]
    TreeBuildFailed,

    #[error("未找到有效的线性段用于拟合")]
    NoScalingWindow,

    #[error("数值计算问题：{msg}")]
    NumericalIssue { msg: String },
}

/// 确定性数值常量
mod constants {
    pub const EPS: f64 = 1e-12;
    pub const TINY: f64 = 1e-300;
    pub const LOG_TINY: f64 = -690.7755; // ln(1e-300)
}

/// AMI（平均互信息）计算和 τ 选择模块
mod ami {
    use super::*;

    /// 计算单个 lag 的 AMI 值
    fn calculate_single_ami(x: &[f64], y: &[f64], n_bins: usize, quantile_bins: bool) -> f64 {
        if x.len() != y.len() || x.is_empty() {
            return 0.0;
        }

        let (x_bins, y_bins) = if quantile_bins {
            (quantile_binning(x, n_bins), quantile_binning(y, n_bins))
        } else {
            (linear_binning(x, n_bins), linear_binning(y, n_bins))
        };

        // 计算联合分布
        let mut joint_count = HashMap::new();
        let mut x_count = HashMap::new();
        let mut y_count = HashMap::new();
        let total_pairs = x.len();

        for (i, &xb) in x_bins.iter().enumerate() {
            let yb = y_bins[i];
            *joint_count.entry((xb, yb)).or_insert(0) += 1;
            *x_count.entry(xb).or_insert(0) += 1;
            *y_count.entry(yb).or_insert(0) += 1;
        }

        // 计算 AMI
        let mut ami = 0.0;
        for (&(xb, yb), &count_xy) in &joint_count {
            let count_x = x_count[&xb];
            let count_y = y_count[&yb];

            if count_xy > 0 && count_x > 0 && count_y > 0 {
                let p_xy = count_xy as f64 / total_pairs as f64;
                let p_x = count_x as f64 / total_pairs as f64;
                let p_y = count_y as f64 / total_pairs as f64;

                ami += p_xy * (p_xy / (p_x * p_y)).ln();
            }
        }

        ami
    }

    /// 分位数分箱（确定性）
    fn quantile_binning(data: &[f64], n_bins: usize) -> Vec<usize> {
        if data.is_empty() {
            return Vec::new();
        }

        let mut sorted_data = data.to_vec();
        sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap());

        data.iter().map(|&val| {
            let pos = sorted_data.binary_search_by(|&x| x.partial_cmp(&val).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or_else(|idx| idx);
            let bin = (pos as f64 / data.len() as f64 * n_bins as f64).floor() as usize;
            bin.min(n_bins - 1)
        }).collect()
    }

    /// 线性等距分箱
    fn linear_binning(data: &[f64], n_bins: usize) -> Vec<usize> {
        if data.is_empty() {
            return Vec::new();
        }

        let min_val = data.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_val = data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let range = max_val - min_val;

        if range < constants::EPS {
            return vec![0; data.len()];
        }

        data.iter().map(|&val| {
            let bin = ((val - min_val) / range * n_bins as f64).floor() as usize;
            bin.min(n_bins - 1)
        }).collect()
    }

    /// 选择 τ 的确定性规则
    fn select_tau(ami_values: &[f64]) -> usize {
        // 规则1：找第一局部极小值
        for i in 1..ami_values.len() - 1 {
            if utils::is_strict_local_minimum(ami_values, i) {
                return i + 1; // τ 从 1 开始，所以 +1
            }
        }

        // 规则2：找首次 AMI ≤ AMI(1)/e 的 lag
        let ami_1 = ami_values[0];
        let threshold = ami_1 / E;

        for (i, &ami_val) in ami_values.iter().enumerate() {
            if ami_val <= threshold {
                return i + 1; // τ 从 1 开始
            }
        }

        // 规则3：默认值
        (ami_values.len() / 10).max(1)
    }

    /// 公共接口：计算 AMI 曲线并选择 τ
    pub fn calculate_ami_and_select_tau(
        x: &[f64],
        max_lag: usize,
        n_bins: usize,
        quantile_bins: bool
    ) -> (usize, Vec<usize>, Vec<f64>) {
        let mut ami_values = Vec::with_capacity(max_lag);
        let mut lags = Vec::new();

        for lag in 1..=max_lag {
            if lag >= x.len() {
                break;
            }

            let x_lag = &x[..x.len() - lag];
            let x_shifted = &x[lag..];

            let ami_val = calculate_single_ami(x_lag, x_shifted, n_bins, quantile_bins);
            ami_values.push(ami_val);
            lags.push(lag);
        }

        let selected_tau = select_tau(&ami_values);
        (selected_tau, lags, ami_values)
    }
}

/// FNN（假近邻）计算和 m 选择模块
mod fnn {
    use super::*;

    /// 构建延迟嵌入向量
    fn build_delay_embedding(x: &[f64], m: usize, tau: usize) -> Result<Vec<Vec<f64>>, GpError> {
        let n = x.len();
        if n < (m - 1) * tau + 1 {
            return Err(GpError::InvalidTauOrM);
        }

        let n_vectors = n - (m - 1) * tau;
        let mut embedding = Vec::with_capacity(n_vectors);

        for i in 0..n_vectors {
            let mut vector = Vec::with_capacity(m);
            for j in 0..m {
                vector.push(x[i + j * tau]);
            }
            embedding.push(vector);
        }

        Ok(embedding)
    }

    /// 使用简单搜索找到最近邻（确定性）
    fn find_nearest_neighbor_simple(
        query_idx: usize,
        embedding: &[Vec<f64>],
        theiler: usize
    ) -> Option<(usize, f64)> {
        let query_point = &embedding[query_idx];
        let mut min_distance = f64::INFINITY;
        let mut nearest_idx = None;

        for (i, point) in embedding.iter().enumerate() {
            if i == query_idx {
                continue;
            }

            // Theiler 窗口过滤
            let time_diff = (i as isize - query_idx as isize).abs() as usize;
            if time_diff <= theiler {
                continue;
            }

            // 计算欧几里得距离
            let distance = euclidean_distance(query_point, point);
            if distance < min_distance {
                min_distance = distance;
                nearest_idx = Some(i);
            }
        }

        nearest_idx.map(|idx| (idx, min_distance))
    }

    /// 计算单个 m 的假近邻比例
    fn calculate_fnn_ratio_for_m(
        x: &[f64],
        m: usize,
        tau: usize,
        rtol: f64,
        atol: f64,
        theiler: usize,
        x_std: f64
    ) -> Result<f64, GpError> {
        let embedding_m = build_delay_embedding(x, m, tau)?;
        let embedding_m1 = build_delay_embedding(x, m + 1, tau)?;

        if embedding_m.len() < 2 {
            return Ok(1.0); // 所有点都是假近邻
        }

        let mut false_neighbors = 0;
        let mut valid_samples = 0;

        // 确保两个嵌入有相同的长度进行比较
        let min_len = embedding_m.len().min(embedding_m1.len());

        for i in 0..min_len {
            if let Some((nearest_idx, distance_m)) = find_nearest_neighbor_simple(i, &embedding_m, theiler) {
                // 确保 nearest_idx 也在 embedding_m1 的范围内
                if nearest_idx < embedding_m1.len() {
                    // 计算 m+1 维距离
                    let distance_m1 = euclidean_distance(&embedding_m1[i], &embedding_m1[nearest_idx]);

                // Kennedy et al. 假近邻判据
                    let is_false = if distance_m > constants::EPS {
                        let ratio_increase = (distance_m1 - distance_m) / distance_m;
                        ratio_increase > rtol
                    } else {
                        false
                    } || {
                        // 第二个判据：新增分量的相对偏差
                        // 确保索引有效
                        let idx_i = i + m * tau;
                        let idx_nearest = nearest_idx + m * tau;
                        if idx_i < x.len() && idx_nearest < x.len() {
                            let delta_new = (x[idx_i] - x[idx_nearest]).abs();
                            delta_new / x_std > atol
                        } else {
                            false
                        }
                    };

                    if is_false {
                        false_neighbors += 1;
                    }
                    valid_samples += 1;
                }
            }
        }

        if valid_samples == 0 {
            return Ok(1.0);
        }

        Ok(false_neighbors as f64 / valid_samples as f64)
    }

    /// 选择最优 m 的确定性规则
    fn select_optimal_m(fnn_ratios: &[f64], threshold: f64) -> usize {
        // 规则1：首次比例≤阈值的 m（m 从 2 开始）
        for (i, &ratio) in fnn_ratios.iter().enumerate() {
            if ratio <= threshold {
                return i + 2; // m 从 2 开始
            }
        }

        // 规则2：没有低于阈值的，返回最大值
        fnn_ratios.len() + 1
    }

    /// 公共接口：计算 FNN 曲线并选择 m
    pub fn calculate_fnn_and_select_m(
        x: &[f64],
        tau: usize,
        m_max: usize,
        rtol: f64,
        atol: f64,
        threshold: f64
    ) -> Result<(usize, Vec<usize>, Vec<f64>), GpError> {
        let x_std = {
            let mean = x.iter().sum::<f64>() / x.len() as f64;
            let variance = x.iter().map(|&val| (val - mean).powi(2)).sum::<f64>() / x.len() as f64;
            variance.sqrt()
        };

        let mut fnn_ratios = Vec::with_capacity(m_max - 1);
        let mut ms = Vec::new();

        for m in 2..=m_max {
            let ratio = calculate_fnn_ratio_for_m(x, m, tau, rtol, atol, 0, x_std)?;
            fnn_ratios.push(ratio);
            ms.push(m);
        }

        let selected_m = select_optimal_m(&fnn_ratios, threshold);
        Ok((selected_m, ms, fnn_ratios))
    }
}

/// Theiler 窗口选择模块
mod theiler {
    use super::*;

    /// 确定性 Theiler 窗口选择
    pub fn select_theiler_window(
        ami_lags: &[usize],
        ami_values: &[f64],
        tau: usize
    ) -> usize {
        // 规则：从 AMI 曲线找首次 AMI ≤ AMI(1)/e 的 lag
        if ami_values.is_empty() {
            return (10 * tau).max(1);
        }

        let ami_1 = ami_values[0];
        let threshold = ami_1 / E;

        for (&lag, &ami_val) in ami_lags.iter().zip(ami_values.iter()) {
            if ami_val <= threshold {
                return (lag.min(10 * tau)).max(1);
            }
        }

        // 默认值
        (10 * tau).max(1)
    }
}

/// 相关和 C(r) 计算模块
mod correlation_sum {
    use super::*;

    /// 生成对数均匀半径网格
    fn generate_radius_grid(
        distances: &[f64],
        n_r: usize,
        percentile_lo: f64,
        percentile_hi: f64
    ) -> Result<Vec<f64>, GpError> {
        if distances.is_empty() {
            return Err(GpError::NumericalIssue {
                msg: "距离列表为空".to_string()
            });
        }

        let mut sorted_distances = distances.to_vec();
        sorted_distances.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let r_lo = utils::percentile_sorted(&sorted_distances, percentile_lo) * 0.5;
        let r_lo = r_lo.max(constants::EPS);
        let r_hi = utils::percentile_sorted(&sorted_distances, percentile_hi);

        if r_hi <= r_lo {
            return Err(GpError::NumericalIssue {
                msg: "半径范围无效".to_string()
            });
        }

        // 对数均匀分布
        let log_lo = r_lo.ln();
        let log_hi = r_hi.ln();
        let log_step = (log_hi - log_lo) / (n_r - 1) as f64;

        let mut radii = Vec::with_capacity(n_r);
        for i in 0..n_r {
            let log_r = log_lo + i as f64 * log_step;
            radii.push(log_r.exp());
        }

        Ok(radii)
    }

    /// 计算全量点对距离（分块处理以避免内存峰值）
    fn calculate_all_pair_distances(embedding: &[Vec<f64>]) -> Vec<f64> {
        let n = embedding.len();
        let mut distances = Vec::with_capacity(n * (n - 1) / 2);

        // 分块计算，避免同时存储太多距离
        let chunk_size = 1000.min(n);

        for i in 0..n {
            for j in (i + 1)..n {
                let dist = euclidean_distance(&embedding[i], &embedding[j]);
                distances.push(dist);
            }
        }

        distances
    }

    /// 计算欧几里得距离（带SIMD优化）
    fn euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
        if a.len() != b.len() {
            return f64::INFINITY;
        }

        // 对于长向量，使用分块计算提高缓存效率
        const CHUNK_SIZE: usize = 64;
        let mut sum_squares = 0.0;

        if a.len() > CHUNK_SIZE {
            // 分块计算大向量
            for (chunk_a, chunk_b) in a.chunks(CHUNK_SIZE).zip(b.chunks(CHUNK_SIZE)) {
                let chunk_sum: f64 = chunk_a.iter()
                    .zip(chunk_b.iter())
                    .map(|(x, y)| {
                        let diff = x - y;
                        diff * diff
                    })
                    .sum();
                sum_squares += chunk_sum;
            }
        } else {
            // 小向量直接计算
            sum_squares = a.iter()
                .zip(b.iter())
                .map(|(x, y)| {
                    let diff = x - y;
                    diff * diff
                })
                .sum();
        }

        sum_squares.sqrt()
    }

    /// 优化版本：使用单次遍历+排序方法计算相关和 C(r)
    fn calculate_correlation_sum_simple(
        embedding: &[Vec<f64>],
        radii: &[f64],
        theiler: usize
    ) -> Result<Vec<f64>, GpError> {
        let n = embedding.len();
        if n < 2 {
            return Err(GpError::NumericalIssue {
                msg: "嵌入向量数量不足".to_string()
            });
        }

        // 步骤1：计算所有有效的点对距离（单次遍历）
        let mut distances = Vec::new();
        let mut valid_pairs = 0;

        for i in 0..n {
            for j in (i + 1)..n {
                // Theiler窗口过滤
                if j - i <= theiler {
                    continue;
                }

                let distance = euclidean_distance(&embedding[i], &embedding[j]);
                distances.push(distance);
                valid_pairs += 1;
            }
        }

        if valid_pairs == 0 {
            return Ok(vec![0.0; radii.len()]);
        }

        // 步骤2：距离排序（用于快速计数）
        distances.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // 步骤3：对每个半径使用二分查找+累积计数
        let total_pairs = valid_pairs as f64;
        let mut correlation_sums = Vec::with_capacity(radii.len());

        for &radius in radii {
            let count = binary_search_count_leq(&distances, radius);
            let c_r = (2.0 * count as f64) / total_pairs;
            correlation_sums.push(c_r);
        }

        Ok(correlation_sums)
    }

    /// 二分查找：返回小于等于target的元素数量
    fn binary_search_count_leq(sorted_arr: &[f64], target: f64) -> usize {
        match sorted_arr.binary_search_by(|&x| x.partial_cmp(&target).unwrap_or(std::cmp::Ordering::Greater)) {
            Ok(idx) | Err(idx) => idx + 1,  // 插入位置+1就是<=target的元素数量
        }
    }

    /// 公共接口：计算相关和
    pub fn calculate_correlation_sum(
        embedding: &[Vec<f64>],
        theiler: usize,
        n_r: usize,
        percentile_lo: f64,
        percentile_hi: f64
    ) -> Result<(Vec<f64>, Vec<f64>), GpError> {
        // 计算距离分布
        let distances = calculate_all_pair_distances(embedding);

        // 生成半径网格
        let radii = generate_radius_grid(&distances, n_r, percentile_lo, percentile_hi)?;

        // 计算相关和
        let correlation_sums = calculate_correlation_sum_simple(embedding, &radii, theiler)?;

        Ok((radii, correlation_sums))
    }
}

/// 线性段检测和 D2 估计模块
mod linear_segment {
    use super::*;

    /// 计算局部斜率（固定窗口）
    fn calculate_local_slopes(
        log_r: &[f64],
        log_c: &[f64],
        window_size: usize
    ) -> Vec<Option<f64>> {
        let n = log_r.len();
        if n < window_size {
            return vec![None; n];
        }

        let mut slopes = vec![None; n];
        let half_window = window_size / 2;

        for i in half_window..(n - half_window) {
            let start = i - half_window;
            let end = i + half_window + 1;

            if let Ok((slope, _, _)) = utils::linear_regression(&log_r[start..end], &log_c[start..end]) {
                slopes[i] = Some(slope);
            }
        }

        slopes
    }

    /// 评估候选段的稳定性
    fn evaluate_segment_stability(
        log_r: &[f64],
        log_c: &[f64],
        start: usize,
        end: usize,
        short_window: usize
    ) -> Result<(f64, f64, f64), GpError> {
        if end <= start || start >= log_r.len() || end > log_r.len() {
            return Err(GpError::NumericalIssue {
                msg: "无效的拟合段范围".to_string()
            });
        }

        // 计算整体线性拟合
        let (slope, intercept, r2) = utils::linear_regression(&log_r[start..end], &log_c[start..end])?;

        // 计算局部斜率的标准差
        let local_slopes = calculate_local_slopes(log_r, log_c, short_window);
        let segment_slopes: Vec<f64> = local_slopes[start..end]
            .iter()
            .filter_map(|&opt| opt)
            .collect();

        let slope_std = if segment_slopes.len() < 2 {
            0.0
        } else {
            let mean = segment_slopes.iter().sum::<f64>() / segment_slopes.len() as f64;
            let variance = segment_slopes.iter()
                .map(|&s| (s - mean).powi(2))
                .sum::<f64>() / (segment_slopes.len() - 1) as f64;
            variance.sqrt()
        };

        Ok((slope, r2, slope_std))
    }

    /// 枚举所有可能的拟合段并选择最佳段（使用相对标准和回退策略）
    fn find_best_scaling_region(
        log_r: &[f64],
        log_c: &[f64],
        correlation_sums: &[f64],
        fit_min_len: usize,
        fit_max_len: usize,
        c_lo: f64,
        c_hi: f64,
        stability_alpha: f64
    ) -> Result<(usize, usize, f64, f64, f64), GpError> {
        let n = log_r.len();

        // 计算相关和的统计信息用于动态约束
        let c_stats = calculate_correlation_sum_stats(correlation_sums);

        // 尝试多个约束级别，从严格到宽松
        let constraint_levels = vec![
            (c_lo, c_hi),                              // 原始约束
            (c_stats.min * 1.1, c_stats.max * 0.9),   // 相对约束（宽松）
            (c_stats.min * 1.01, c_stats.max * 0.99), // 相对约束（更宽松）
            (0.0, 1.0),                                // 几乎无约束（最后回退）
        ];

        for (current_c_lo, current_c_hi) in constraint_levels {
            if let Ok(result) = try_find_with_constraints(
                log_r, log_c, correlation_sums,
                fit_min_len, fit_max_len,
                current_c_lo, current_c_hi,
                stability_alpha
            ) {
                return Ok(result);
            }
        }

        // 如果所有约束都失败，使用最简单的线性拟合
        find_simple_linear_fit(log_r, log_c, fit_min_len)
    }

    /// 计算相关和的统计信息
    fn calculate_correlation_sum_stats(correlation_sums: &[f64]) -> CorrelationSumStats {
        let min_val = correlation_sums.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_val = correlation_sums.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let mean_val = correlation_sums.iter().sum::<f64>() / correlation_sums.len() as f64;

        CorrelationSumStats {
            min: min_val,
            max: max_val,
            mean: mean_val,
        }
    }

    /// 相关和统计信息结构
    #[derive(Debug, Clone)]
    struct CorrelationSumStats {
        min: f64,
        max: f64,
        mean: f64,
    }

    /// 使用给定约束尝试寻找最佳段
    fn try_find_with_constraints(
        log_r: &[f64],
        log_c: &[f64],
        correlation_sums: &[f64],
        fit_min_len: usize,
        fit_max_len: usize,
        c_lo: f64,
        c_hi: f64,
        stability_alpha: f64
    ) -> Result<(usize, usize, f64, f64, f64), GpError> {
        let n = log_r.len();
        let mut best_score = f64::NEG_INFINITY;
        let mut best_start = 0;
        let mut best_end = 0;
        let mut best_slope = 0.0;
        let mut best_intercept = 0.0;
        let mut best_r2 = 0.0;

        // 找到有效的索引范围（满足 C(r) 约束）
        let valid_indices: Vec<usize> = (0..n)
            .filter(|&i| correlation_sums[i] > c_lo && correlation_sums[i] < c_hi)
            .collect();

        if valid_indices.len() < fit_min_len {
            return Err(GpError::NoScalingWindow);
        }

        // 枚举所有可能的窗口长度和起点
        for window_len in (fit_min_len..=fit_max_len).rev() { // 从长到短，优先选择长窗口
            for &start_idx in &valid_indices {
                let end_idx = start_idx + window_len;

                if end_idx > n || end_idx > valid_indices[valid_indices.len() - 1] + 1 {
                    continue;
                }

                // 检查窗口内所有点是否都有效
                let window_valid = (start_idx..end_idx).all(|i| {
                    correlation_sums[i] > c_lo && correlation_sums[i] < c_hi
                });

                if !window_valid {
                    continue;
                }

                // 评估该段
                if let Ok((slope, r2, slope_std)) = evaluate_segment_stability(
                    log_r, log_c, start_idx, end_idx, 5
                ) {
                    // 计算评分：R² - α * 局部斜率标准差
                    let score = r2 - stability_alpha * slope_std;

                    // 确定性选择规则
                    if score > best_score + constants::EPS {
                        best_score = score;
                        best_start = start_idx;
                        best_end = end_idx;
                        best_slope = slope;
                        best_intercept = {
                            let mean_x = log_r[start_idx..end_idx].iter().sum::<f64>() / window_len as f64;
                            let mean_y = log_c[start_idx..end_idx].iter().sum::<f64>() / window_len as f64;
                            mean_y - slope * mean_x
                        };
                        best_r2 = r2;
                    } else if (score - best_score).abs() <= constants::EPS {
                        // 并列时选择更长的窗口
                        if window_len > (best_end - best_start) {
                            best_start = start_idx;
                            best_end = end_idx;
                            best_slope = slope;
                            best_intercept = {
                                let mean_x = log_r[start_idx..end_idx].iter().sum::<f64>() / window_len as f64;
                                let mean_y = log_c[start_idx..end_idx].iter().sum::<f64>() / window_len as f64;
                                mean_y - slope * mean_x
                            };
                            best_r2 = r2;
                        } else if window_len == (best_end - best_start) && start_idx < best_start {
                            // 长度相同时选择起点更小的
                            best_start = start_idx;
                            best_end = end_idx;
                            best_slope = slope;
                            best_intercept = {
                                let mean_x = log_r[start_idx..end_idx].iter().sum::<f64>() / window_len as f64;
                                let mean_y = log_c[start_idx..end_idx].iter().sum::<f64>() / window_len as f64;
                                mean_y - slope * mean_x
                            };
                            best_r2 = r2;
                        }
                    }
                }
            }
        }

        if best_end > best_start {
            Ok((best_start, best_end, best_slope, best_intercept, best_r2))
        } else {
            Err(GpError::NoScalingWindow)
        }
    }

    /// 简单线性拟合回退策略
    fn find_simple_linear_fit(
        log_r: &[f64],
        log_c: &[f64],
        min_len: usize
    ) -> Result<(usize, usize, f64, f64, f64), GpError> {
        let n = log_r.len();

        if n < min_len {
            return Err(GpError::NoScalingWindow);
        }

        // 使用中间50%的数据进行简单拟合
        let start = n / 4;
        let end = 3 * n / 4;
        let window_len = end - start;

        if window_len < min_len {
            return Err(GpError::NoScalingWindow);
        }

        // 计算简单线性回归
        let sum_x: f64 = log_r[start..end].iter().sum();
        let sum_y: f64 = log_c[start..end].iter().sum();
        let sum_xy: f64 = log_r[start..end].iter().zip(log_c[start..end].iter())
            .map(|(x, y)| x * y).sum();
        let sum_x2: f64 = log_r[start..end].iter().map(|x| x * x).sum();

        let n_f64 = window_len as f64;
        let slope = (n_f64 * sum_xy - sum_x * sum_y) / (n_f64 * sum_x2 - sum_x * sum_x);
        let intercept = (sum_y - slope * sum_x) / n_f64;

        // 计算R²
        let mean_y = sum_y / n_f64;
        let total_sum_squares: f64 = log_c[start..end].iter()
            .map(|y| (y - mean_y).powi(2)).sum();
        let residual_sum_squares: f64 = log_r[start..end].iter().zip(log_c[start..end].iter())
            .map(|(x, y)| {
                let predicted = slope * x + intercept;
                (y - predicted).powi(2)
            }).sum();

        let r2 = if total_sum_squares > 0.0 {
            1.0 - residual_sum_squares / total_sum_squares
        } else {
            0.0
        };

        Ok((start, end, slope, intercept, r2))
    }

    /// 公共接口：检测线性段并估计 D2
    pub fn detect_scaling_region_and_estimate_d2(
        radii: &[f64],
        correlation_sums: &[f64],
        fit_min_len: usize,
        fit_max_len: usize,
        c_lo: f64,
        c_hi: f64,
        stability_alpha: f64
    ) -> Result<(usize, usize, f64, f64, f64, Vec<f64>, Vec<f64>, Vec<Option<f64>>), GpError> {
        if radii.len() != correlation_sums.len() || radii.is_empty() {
            return Err(GpError::NumericalIssue {
                msg: "半径和相关和数据不匹配或为空".to_string()
            });
        }

        // 计算 log 值
        let log_r: Vec<f64> = radii.iter().map(|&r| r.ln()).collect();
        let log_c: Vec<f64> = correlation_sums.iter()
            .map(|&c| if c > 0.0 { c.ln() } else { constants::LOG_TINY })
            .collect();

        // 计算局部斜率
        let local_slopes = calculate_local_slopes(&log_r, &log_c, 7);

        // 找到最佳线性段
        let (fit_start, fit_end, d2_est, intercept, r2) = find_best_scaling_region(
            &log_r, &log_c, correlation_sums,
            fit_min_len, fit_max_len, c_lo, c_hi, stability_alpha
        )?;

        Ok((fit_start, fit_end, d2_est, intercept, r2, log_r, log_c, local_slopes))
    }
}

/// 延迟嵌入构建模块
mod embedding {
    use super::*;

    /// 构建 m 维延迟嵌入
    pub fn build_delay_embedding(x: &[f64], m: usize, tau: usize) -> Result<Vec<Vec<f64>>, GpError> {
        let n = x.len();
        if n < (m - 1) * tau + 1 {
            return Err(GpError::InvalidTauOrM);
        }

        let n_vectors = n - (m - 1) * tau;
        let mut embedding = Vec::with_capacity(n_vectors);

        for i in 0..n_vectors {
            let mut vector = Vec::with_capacity(m);
            for j in 0..m {
                vector.push(x[i + j * tau]);
            }
            embedding.push(vector);
        }

        Ok(embedding)
    }
}

// 工具函数
mod utils {
    use super::*;

    /// 序列标准化：减均值/除标准差（完全确定性）
    pub fn standardize_sequence(x: &[f64]) -> Result<Vec<f64>, GpError> {
        if x.is_empty() {
            return Err(GpError::InputTooShort { min_len: 1, actual_len: 0 });
        }

        let n = x.len();
        let mean: f64 = x.iter().sum::<f64>() / n as f64;

        let variance: f64 = x.iter()
            .map(|&val| {
                let diff = val - mean;
                diff * diff
            })
            .sum::<f64>() / n as f64;

        let std = variance.sqrt();

        if std < constants::EPS {
            return Err(GpError::InputTooConstant);
        }

        Ok(x.iter().map(|&val| (val - mean) / std).collect())
    }

    /// 确定性分位数计算（向下取整索引）
    pub fn percentile_sorted(sorted_data: &[f64], percentile: f64) -> f64 {
        if sorted_data.is_empty() {
            return 0.0;
        }

        let n = sorted_data.len();
        let index = ((percentile / 100.0) * (n as f64 - 1.0)).floor() as usize;
        let index = index.min(n - 1);
        sorted_data[index]
    }

    /// 检查是否为局部极小值（严格小于相邻值）
    pub fn is_strict_local_minimum(values: &[f64], i: usize) -> bool {
        if i == 0 || i >= values.len() - 1 {
            return false;
        }
        values[i] < values[i - 1] && values[i] < values[i + 1]
    }

    /// 线性回归（最小二乘）
    pub fn linear_regression(x: &[f64], y: &[f64]) -> Result<(f64, f64, f64), GpError> {
        if x.len() != y.len() || x.len() < 2 {
            return Err(GpError::NumericalIssue {
                msg: "线性回归需要至少2个等长数据点".to_string()
            });
        }

        let n = x.len() as f64;
        let sum_x: f64 = x.iter().sum();
        let sum_y: f64 = y.iter().sum();
        let sum_xx: f64 = x.iter().map(|&xi| xi * xi).sum();
        let sum_xy: f64 = x.iter().zip(y.iter()).map(|(xi, yi)| xi * yi).sum();

        let denominator = n * sum_xx - sum_x * sum_x;
        if denominator.abs() < constants::EPS {
            return Err(GpError::NumericalIssue {
                msg: "线性回归分母过小".to_string()
            });
        }

        let slope = (n * sum_xy - sum_x * sum_y) / denominator;
        let intercept = (sum_y - slope * sum_x) / n;

        // 计算 R²
        let mean_y = sum_y / n;
        let ss_tot: f64 = y.iter().map(|&yi| (yi - mean_y).powi(2)).sum();
        let ss_res: f64 = x.iter().zip(y.iter())
            .map(|(&xi, &yi)| (yi - (slope * xi + intercept)).powi(2))
            .sum();

        let r2 = if ss_tot.abs() < constants::EPS {
            1.0 // 完美拟合
        } else {
            1.0 - ss_res / ss_tot
        };

        Ok((slope, intercept, r2.max(0.0).min(1.0)))
    }
}

/// 智能调整 τ 和 m 的组合以适应数据长度
fn adjust_tau_m_combination(data_len: usize, mut tau: usize, mut m: usize) -> (usize, usize) {
    // 确保 (m-1)*τ + 1 ≤ data_len
    while data_len < (m - 1) * tau + 1 && (m > 2 || tau > 1) {
        if m > 2 {
            m -= 1;
        } else if tau > 1 {
            tau -= 1;
        } else {
            break;  // 无法再调整
        }
    }

    // 如果还是不满足，使用最小可行参数
    if data_len < (m - 1) * tau + 1 {
        tau = 1;
        m = (data_len / 2).max(2).min(12);  // 限制在合理范围内
    }

    (tau, m)
}

/// GP 相关维数计算的主实现
impl GpOptions {
    /// 根据序列长度调整默认参数
    pub fn adjust_for_sequence_length(mut self, n: usize) -> Self {
        // AMI参数调整
        self.ami_max_lag = self.ami_max_lag.min(200.max(n / 10));
        self.ami_n_bins = self.ami_n_bins.max(8).min(64);

        // FNN参数调整
        self.fnn_m_max = self.fnn_m_max.min(20.max(n / 50));

        // 半径网格调整 - 对小数据集减少半径数量
        if n < 200 {
            self.n_r = self.n_r.max(12).min(n / 5);
        } else if n < 500 {
            self.n_r = self.n_r.max(24).min(n / 8);
        }

        // 拟合窗口自适应调整
        self.fit_min_len = self.fit_min_len.max(3).min(self.n_r / 10);
        self.fit_max_len = self.fit_max_len.min(self.n_r / 2).max(self.fit_min_len + 2);

        self
    }

    /// 根据嵌入维数进一步调整参数
    pub fn adjust_for_embedding_dimension(mut self, m: usize) -> Self {
        // 高维嵌入需要更宽松的参数
        if m > 10 {
            self.fnn_threshold = self.fnn_threshold * 1.5;
            self.stability_alpha = self.stability_alpha * 0.8;  // 更宽松的稳定性要求
        }

        self
    }

    /// 根据数据特征调整参数
    pub fn adapt_to_data_characteristics(
        mut self,
        data_length: usize,
        embedding_dim: usize,
        correlation_sum_range: (f64, f64)
    ) -> Self {
        self = self.adjust_for_sequence_length(data_length);
        self = self.adjust_for_embedding_dimension(embedding_dim);

        // 根据相关和分布调整约束条件
        let (c_min, c_max) = correlation_sum_range;
        let c_range = c_max - c_min;

        if c_range < 0.1 {
            // 相关和分布过于集中，放宽约束
            self.c_lo = c_min * 0.5;
            self.c_hi = c_max + (1.0 - c_max) * 0.5;
            self.stability_alpha *= 0.5;
        }

        self
    }
}

/// 零参数入口：只需传入序列，所有参数自动确定
#[pyfunction]
#[pyo3(signature = (x))]
pub fn gp_correlation_dimension_auto(
    x: PyReadonlyArray1<f64>
) -> PyResult<GpResult> {
    let x_slice = x.as_slice()?;
    let result = internal_gp_correlation_dimension(x_slice, None).map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!("GP 相关维数计算失败: {}", e))
    })?;
    Ok(result)
}

/// 可选参数入口：可提供自定义选项
#[pyfunction]
#[pyo3(signature = (x, opts=None))]
pub fn gp_correlation_dimension(
    x: PyReadonlyArray1<f64>,
    opts: Option<GpOptions>
) -> PyResult<GpResult> {
    let x_slice = x.as_slice()?;
    let result = internal_gp_correlation_dimension(x_slice, opts).map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!("GP 相关维数计算失败: {}", e))
    })?;
    Ok(result)
}

/// 内部实现函数
fn internal_gp_correlation_dimension(
    x: &[f64],
    opts: Option<GpOptions>
) -> Result<GpResult, GpError> {
    // 输入验证 - 降低最小长度要求
    if x.len() < 30 {
        return Err(GpError::InputTooShort {
            min_len: 30,
            actual_len: x.len()
        });
    }

    // 获取基础选项
    let base_options = opts.unwrap_or_default().adjust_for_sequence_length(x.len());

    // 步骤0：标准化序列
    let x_std = utils::standardize_sequence(x)?;

    // 步骤1：AMI 计算与 τ 选择
    let (tau, ami_lags, ami_values) = if let Some(tau_override) = base_options.tau_override {
        (tau_override, Vec::new(), Vec::new())
    } else {
        ami::calculate_ami_and_select_tau(
            &x_std,
            base_options.ami_max_lag,
            base_options.ami_n_bins,
            base_options.ami_quantile_bins
        )
    };

    // 步骤2：FNN 计算与 m 选择
    let (m, fnn_ms, fnn_ratios) = if let Some(m_override) = base_options.m_override {
        (m_override, Vec::new(), Vec::new())
    } else {
        fnn::calculate_fnn_and_select_m(
            &x_std,
            tau,
            base_options.fnn_m_max,
            base_options.fnn_rtol,
            base_options.fnn_atol,
            base_options.fnn_threshold
        )?
    };

    // 智能调整 τ 和 m 的组合
    let (tau, m) = adjust_tau_m_combination(x.len(), tau, m);

    // 步骤3：Theiler 窗口选择
    let theiler = if let Some(theiler_override) = base_options.theiler_override {
        theiler_override
    } else {
        theiler::select_theiler_window(&ami_lags, &ami_values, tau)
    };

    // 步骤4：构建延迟嵌入
    let embedding = embedding::build_delay_embedding(&x_std, m, tau)?;

    // 步骤5：相关和 C(r) 计算
    let (rs, cs) = correlation_sum::calculate_correlation_sum(
        &embedding,
        theiler,
        base_options.n_r,
        base_options.r_percentile_lo,
        base_options.r_percentile_hi
    )?;

    // 获取相关和统计信息用于参数自适应
    let c_min = cs.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let c_max = cs.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let correlation_range = (c_min, c_max);

    // 根据数据特征自适应调整参数
    let options = base_options.adapt_to_data_characteristics(
        x.len(),
        m,
        correlation_range
    );

    // 步骤6：线性段检测和 D2 估计（使用自适应后的参数）
    let (fit_start, fit_end, d2_est, fit_intercept, fit_r2, log_r, log_c, local_slopes) =
        linear_segment::detect_scaling_region_and_estimate_d2(
            &rs,
            &cs,
            options.fit_min_len,
            options.fit_max_len,
            options.c_lo,
            options.c_hi,
            options.stability_alpha
        )?;

    Ok(GpResult {
        tau,
        m,
        theiler,
        rs,
        cs,
        log_r,
        log_c,
        local_slopes,
        fit_start,
        fit_end,
        d2_est,
        fit_intercept,
        fit_r2,
        ami_lags,
        ami_values,
        fnn_ms,
        fnn_ratios,
    })
}

/// 为 Python 导出的辅助函数
#[pyfunction]
pub fn gp_create_default_options() -> GpOptions {
    GpOptions::default()
}

#[pyfunction]
#[pyo3(signature = (
    ami_max_lag = 200,
    ami_n_bins = 32,
    ami_quantile_bins = true,
    tau_override = None,
    fnn_m_max = 12,
    fnn_rtol = 10.0,
    fnn_atol = 2.0,
    fnn_threshold = 0.02,
    m_override = None,
    theiler_override = None,
    n_r = 48,
    r_percentile_lo = 5.0,
    r_percentile_hi = 90.0,
    fit_min_len = 6,
    fit_max_len = 14,
    c_lo = 1e-4,
    c_hi = 1.0 - 1e-3,
    stability_alpha = 0.05
))]
pub fn gp_create_options(
    ami_max_lag: usize,
    ami_n_bins: usize,
    ami_quantile_bins: bool,
    tau_override: Option<usize>,
    fnn_m_max: usize,
    fnn_rtol: f64,
    fnn_atol: f64,
    fnn_threshold: f64,
    m_override: Option<usize>,
    theiler_override: Option<usize>,
    n_r: usize,
    r_percentile_lo: f64,
    r_percentile_hi: f64,
    fit_min_len: usize,
    fit_max_len: usize,
    c_lo: f64,
    c_hi: f64,
    stability_alpha: f64,
) -> GpOptions {
    GpOptions {
        ami_max_lag,
        ami_n_bins,
        ami_quantile_bins,
        tau_override,
        fnn_m_max,
        fnn_rtol,
        fnn_atol,
        fnn_threshold,
        m_override,
        theiler_override,
        n_r,
        r_percentile_lo,
        r_percentile_hi,
        fit_min_len,
        fit_max_len,
        c_lo,
        c_hi,
        stability_alpha,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 生成 Logistic 映射序列（r=3.9）
    fn generate_logistic_map(n: usize, r: f64, x0: f64) -> Vec<f64> {
        let mut x = vec![x0];
        for i in 1..n {
            let x_prev = x[i - 1];
            let x_next = r * x_prev * (1.0 - x_prev);
            x.push(x_next);
        }
        x
    }

    /// 生成白噪声序列（确定性，使用简单的伪随机生成）
    fn generate_white_noise(n: usize, seed: u32) -> Vec<f64> {
        let mut x = Vec::with_capacity(n);
        let mut state = seed as u64;

        for _ in 0..n {
            // 简单的线性同余生成器
            state = state.wrapping_mul(1103515245).wrapping_add(12345);
            let value = (state as f64) / (u64::MAX as f64) * 2.0 - 1.0;
            x.push(value);
        }
        x
    }

    #[test]
    fn test_logistic_map_gp_correlation_dimension() {
        // 生成 Logistic 映射数据
        let x = generate_logistic_map(6000, 3.9, 0.1);

        // 计算 GP 相关维数
        let result = internal_gp_correlation_dimension(&x, None).unwrap();

        // 验证结果
        assert!(result.tau >= 1, "τ 应该至少为 1");
        assert!(result.m >= 2, "m 应该至少为 2");
        assert!(result.fit_r2 >= 0.9, "拟合 R² 应该较高");
        assert!(result.d2_est >= 0.5 && result.d2_est <= 3.0, "D₂ 估计应该在合理范围内");
        assert!(result.fit_end > result.fit_start, "拟合段应该有效");

        // 验证数据长度一致性
        assert_eq!(result.rs.len(), result.cs.len());
        assert_eq!(result.log_r.len(), result.log_c.len());
        assert_eq!(result.rs.len(), result.local_slopes.len());
    }

    #[test]
    fn test_white_noise_gp_correlation_dimension() {
        // 生成白噪声数据
        let x = generate_white_noise(2000, 42);

        let result = internal_gp_correlation_dimension(&x, None).unwrap();

        // 白噪声的 D₂ 应该接近嵌入维数
        assert!(result.d2_est >= result.m as f64 * 0.7, "白噪声的 D₂ 应该较高");
        assert!(result.fit_r2 >= 0.8, "拟合质量应该较好");
    }

    #[test]
    fn test_custom_parameters() {
        let x = generate_logistic_map(3000, 3.9, 0.1);

        // 使用自定义参数
        let options = GpOptions {
            tau_override: Some(3),
            m_override: Some(5),
            theiler_override: Some(10),
            ..Default::default()
        };

        let result = internal_gp_correlation_dimension(&x, Some(options)).unwrap();

        assert_eq!(result.tau, 3);
        assert_eq!(result.m, 5);
        assert_eq!(result.theiler, 10);
    }

    #[test]
    fn test_error_cases() {
        // 测试过短序列
        let short_x = vec![1.0, 2.0, 3.0];
        assert!(matches!(
            internal_gp_correlation_dimension(&short_x, None),
            Err(GpError::InputTooShort { .. })
        ));

        // 测试常数序列
        let constant_x = vec![1.0; 1000];
        assert!(matches!(
            internal_gp_correlation_dimension(&constant_x, None),
            Err(GpError::InputTooConstant)
        ));
    }

    #[test]
    fn test_ami_calculation() {
        let x = generate_logistic_map(1000, 3.9, 0.1);
        let (tau, lags, ami_values) = ami::calculate_ami_and_select_tau(&x, 50, 16, true).unwrap();

        assert!(tau >= 1, "选择的 τ 应该有效");
        assert_eq!(lags.len(), ami_values.len(), "lags 和 ami_values 长度应该一致");
        assert!(lags.len() > 0, "应该计算了 AMI 值");
    }

    #[test]
    fn test_fnn_calculation() {
        let x = generate_logistic_map(1000, 3.9, 0.1);
        let (m, ms, fnn_ratios) = fnn::calculate_fnn_and_select_m(&x, 2, 8, 10.0, 2.0, 0.02).unwrap();

        assert!(m >= 2, "选择的 m 应该有效");
        assert_eq!(ms.len(), fnn_ratios.len(), "ms 和 fnn_ratios 长度应该一致");
        assert!(ms.len() > 0, "应该计算了 FNN 值");
    }

    #[test]
    fn test_embedding_construction() {
        let x = generate_logistic_map(100, 3.9, 0.1);
        let embedding = embedding::build_delay_embedding(&x, 3, 2).unwrap();

        assert_eq!(embedding.len(), 100 - (3 - 1) * 2, "嵌入向量数量应该正确");
        assert_eq!(embedding[0].len(), 3, "每个嵌入向量的维数应该正确");
    }

    #[test]
    fn test_deterministic_behavior() {
        // 多次运行相同输入应该得到相同结果
        let x = generate_logistic_map(1000, 3.9, 0.1);

        let result1 = internal_gp_correlation_dimension(&x, None).unwrap();
        let result2 = internal_gp_correlation_dimension(&x, None).unwrap();

        assert_eq!(result1.tau, result2.tau, "τ 选择应该是确定性的");
        assert_eq!(result1.m, result2.m, "m 选择应该是确定性的");
        assert_eq!(result1.theiler, result2.theiler, "Theiler 窗口选择应该是确定性的");
        assert!((result1.d2_est - result2.d2_est).abs() < 1e-10, "D₂ 估计应该是确定性的");
    }
}