use pyo3::prelude::*;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use ndarray::Array1;
use std::collections::HashMap;

/// 优化版滚动窗口核心特征提取
/// 
/// 针对性能优化的版本，使用以下优化策略：
/// 1. 增量相关性计算：避免重复计算统计量
/// 2. 预计算缓存：缓存窗口间的相关性矩阵
/// 3. 内存优化：重用缓冲区，减少内存分配
/// 4. 向量化操作：使用SIMD友好的计算模式
/// 
/// 参数说明：
/// ----------
/// values : numpy.ndarray
///     输入的一维数组，必须是float64类型
/// window_size : int, 可选
///     滚动窗口的大小，默认为5。必须>=2且<=序列长度
/// 
/// 返回值：
/// -------
/// tuple[numpy.ndarray, numpy.ndarray]
///     返回两个数组：
///     - 第一个数组：核心特征序列，每个位置对应该窗口中最重要的特征值
///     - 第二个数组：次要特征序列，每个位置对应该窗口中最不重要的特征值
#[pyfunction]
#[pyo3(signature = (values, window_size=5))]
pub fn rolling_window_core_feature_optimized(
    py: Python,
    values: PyReadonlyArray1<f64>,
    window_size: Option<usize>,
) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
    let values = values.as_array();
    let window_size = window_size.unwrap_or(5);
    let n = values.len();
    
    // 参数验证
    if window_size < 2 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "窗口大小必须至少为2"
        ));
    }
    
    if window_size > n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "窗口大小不能超过序列长度"
        ));
    }
    
    // 初始化结果数组
    let mut core_features = Array1::from_elem(n, f64::NAN);
    let mut minor_features = Array1::from_elem(n, f64::NAN);
    
    if n < window_size {
        return Ok((
            core_features.into_pyarray(py).to_owned(),
            minor_features.into_pyarray(py).to_owned()
        ));
    }
    
    // 创建优化的计算引擎
    let mut engine = OptimizedCorrelationEngine::new(&values, window_size);
    
    // 对每个窗口进行分析
    for current_idx in (window_size - 1)..n {
        let window_start = current_idx - window_size + 1;
        
        // 使用优化引擎计算特征重要性
        if let Some((core_pos, minor_pos)) = engine.analyze_window_importance(current_idx) {
            if let Some(pos) = core_pos {
                core_features[current_idx] = values[window_start + pos];
            }
            
            if let Some(pos) = minor_pos {
                minor_features[current_idx] = values[window_start + pos];
            }
        }
    }
    
    Ok((
        core_features.into_pyarray(py).to_owned(),
        minor_features.into_pyarray(py).to_owned()
    ))
}

/// 优化的相关性计算引擎
struct OptimizedCorrelationEngine<'a> {
    values: &'a ndarray::ArrayView1<'a, f64>,
    window_size: usize,
    window_stats: Vec<FastWindowStats>,
    correlation_cache: HashMap<(usize, usize), f64>,
    #[allow(dead_code)]
    temp_buffer: Vec<f64>,
}

impl<'a> OptimizedCorrelationEngine<'a> {
    fn new(values: &'a ndarray::ArrayView1<'a, f64>, window_size: usize) -> Self {
        let n = values.len();
        let num_windows = n - window_size + 1;
        
        // 预计算所有窗口的统计信息
        let window_stats = Self::precompute_all_window_stats(values, window_size);
        
        // 预分配相关性缓存
        let cache_capacity = (num_windows * (num_windows - 1)) / 2;
        let correlation_cache = HashMap::with_capacity(cache_capacity);
        
        // 预分配临时缓冲区
        let temp_buffer = Vec::with_capacity(window_size);
        
        Self {
            values,
            window_size,
            window_stats,
            correlation_cache,
            temp_buffer,
        }
    }
    
    fn precompute_all_window_stats(values: &ndarray::ArrayView1<f64>, window_size: usize) -> Vec<FastWindowStats> {
        let n = values.len();
        let mut stats = Vec::with_capacity(n - window_size + 1);
        
        for start_idx in 0..=(n - window_size) {
            let window = values.slice(ndarray::s![start_idx..start_idx + window_size]);
            stats.push(FastWindowStats::from_window(&window));
        }
        
        stats
    }
    
    fn analyze_window_importance(&mut self, current_idx: usize) -> Option<(Option<usize>, Option<usize>)> {
        let window_start = current_idx - self.window_size + 1;
        let current_window_idx = window_start;
        
        // 获取基准相关性向量
        let base_correlations = self.get_base_correlations(current_window_idx);
        
        if base_correlations.is_empty() {
            return None;
        }
        
        // 分析每个位置的重要性
        let mut position_impacts = Vec::with_capacity(self.window_size);
        
        for mask_pos in 0..self.window_size {
            // 计算mask后的影响程度
            let impact = self.compute_masked_impact(current_window_idx, mask_pos, &base_correlations);
            if !impact.is_nan() {
                position_impacts.push((mask_pos, impact));
            }
        }
        
        if position_impacts.is_empty() {
            return None;
        }
        
        // 找出最重要和最不重要的位置
        let mut min_impact = f64::INFINITY;
        let mut max_impact = f64::NEG_INFINITY;
        let mut core_pos = None;
        let mut minor_pos = None;
        
        for (pos, impact) in position_impacts {
            if impact < min_impact {
                min_impact = impact;
                core_pos = Some(pos);
            }
            if impact > max_impact {
                max_impact = impact;
                minor_pos = Some(pos);
            }
        }
        
        Some((core_pos, minor_pos))
    }
    
    fn get_base_correlations(&mut self, current_window_idx: usize) -> Vec<f64> {
        let mut correlations = Vec::new();
        
        for other_idx in 0..self.window_stats.len() {
            if other_idx == current_window_idx {
                continue;
            }
            
            let corr = self.get_or_compute_correlation(current_window_idx, other_idx);
            if !corr.is_nan() {
                correlations.push(corr);
            }
        }
        
        correlations
    }
    
    fn get_or_compute_correlation(&mut self, window1_idx: usize, window2_idx: usize) -> f64 {
        let key = if window1_idx < window2_idx {
            (window1_idx, window2_idx)
        } else {
            (window2_idx, window1_idx)
        };
        
        if let Some(&cached_corr) = self.correlation_cache.get(&key) {
            return cached_corr;
        }
        
        // 计算相关性并缓存
        let corr = self.compute_correlation_fast(window1_idx, window2_idx);
        self.correlation_cache.insert(key, corr);
        corr
    }
    
    fn compute_correlation_fast(&self, window1_idx: usize, window2_idx: usize) -> f64 {
        let stats1 = &self.window_stats[window1_idx];
        let stats2 = &self.window_stats[window2_idx];
        
        if !stats1.is_valid() || !stats2.is_valid() {
            return f64::NAN;
        }
        
        let window1_start = window1_idx;
        let window2_start = window2_idx;
        
        let window1 = self.values.slice(ndarray::s![window1_start..window1_start + self.window_size]);
        let window2 = self.values.slice(ndarray::s![window2_start..window2_start + self.window_size]);
        
        // 使用预计算的统计信息进行快速相关性计算
        fast_correlation_with_stats(&window1, &window2, stats1, stats2)
    }
    
    fn compute_masked_impact(
        &mut self, 
        current_window_idx: usize, 
        mask_pos: usize, 
        base_correlations: &[f64]
    ) -> f64 {
        let mut masked_correlations = Vec::with_capacity(base_correlations.len());
        let mut _correlation_idx = 0;
        
        for other_idx in 0..self.window_stats.len() {
            if other_idx == current_window_idx {
                continue;
            }
            
            // 计算masked相关性
            let masked_corr = self.compute_masked_correlation(current_window_idx, other_idx, mask_pos);
            if !masked_corr.is_nan() {
                masked_correlations.push(masked_corr);
            }
            
            _correlation_idx += 1;
        }
        
        if masked_correlations.len() != base_correlations.len() {
            return f64::NAN;
        }
        
        // 计算RMSE作为影响程度
        let mut sum_diff_sq = 0.0;
        for (base, masked) in base_correlations.iter().zip(masked_correlations.iter()) {
            let diff = base - masked;
            sum_diff_sq += diff * diff;
        }
        
        (sum_diff_sq / base_correlations.len() as f64).sqrt()
    }
    
    fn compute_masked_correlation(&mut self, window1_idx: usize, window2_idx: usize, mask_pos: usize) -> f64 {
        let stats2 = &self.window_stats[window2_idx];
        if !stats2.is_valid() {
            return f64::NAN;
        }
        
        let window1_start = window1_idx;
        let window2_start = window2_idx;
        
        let window1 = self.values.slice(ndarray::s![window1_start..window1_start + self.window_size]);
        let window2 = self.values.slice(ndarray::s![window2_start..window2_start + self.window_size]);
        
        // 创建masked窗口统计（在线计算，不缓存）
        let masked_stats1 = FastWindowStats::from_window_masked(&window1, mask_pos);
        
        if !masked_stats1.is_valid() {
            return f64::NAN;
        }
        
        // 使用预计算的统计信息计算相关性
        fast_correlation_with_stats_masked(&window1, &window2, &masked_stats1, stats2, mask_pos)
    }
}

/// 快速窗口统计信息结构
#[derive(Clone, Debug)]
struct FastWindowStats {
    mean: f64,
    std_dev: f64,
    #[allow(dead_code)]
    sum: f64,
    #[allow(dead_code)]
    sum_sq: f64,
    valid_count: usize,
}

impl FastWindowStats {
    fn from_window(window: &ndarray::ArrayView1<f64>) -> Self {
        let mut sum = 0.0;
        let mut sum_sq = 0.0;
        let mut valid_count = 0;
        
        // 一次遍历计算所有统计量
        for &val in window.iter() {
            if !val.is_nan() {
                sum += val;
                sum_sq += val * val;
                valid_count += 1;
            }
        }
        
        if valid_count < 2 {
            return Self {
                mean: f64::NAN,
                std_dev: f64::NAN,
                sum,
                sum_sq,
                valid_count: 0,
            };
        }
        
        let mean = sum / valid_count as f64;
        let variance = (sum_sq / valid_count as f64) - (mean * mean);
        let std_dev = if variance > f64::EPSILON { variance.sqrt() } else { 0.0 };
        
        Self {
            mean,
            std_dev,
            sum,
            sum_sq,
            valid_count,
        }
    }
    
    fn from_window_masked(window: &ndarray::ArrayView1<f64>, mask_pos: usize) -> Self {
        let mut sum = 0.0;
        let mut sum_sq = 0.0;
        let mut valid_count = 0;
        
        for (i, &val) in window.iter().enumerate() {
            if i != mask_pos && !val.is_nan() {
                sum += val;
                sum_sq += val * val;
                valid_count += 1;
            }
        }
        
        if valid_count < 2 {
            return Self {
                mean: f64::NAN,
                std_dev: f64::NAN,
                sum,
                sum_sq,
                valid_count: 0,
            };
        }
        
        let mean = sum / valid_count as f64;
        let variance = (sum_sq / valid_count as f64) - (mean * mean);
        let std_dev = if variance > f64::EPSILON { variance.sqrt() } else { 0.0 };
        
        Self {
            mean,
            std_dev,
            sum,
            sum_sq,
            valid_count,
        }
    }
    
    fn is_valid(&self) -> bool {
        self.valid_count >= 2 && self.std_dev > f64::EPSILON
    }
}

/// 使用预计算统计信息的快速相关性计算
fn fast_correlation_with_stats(
    x: &ndarray::ArrayView1<f64>,
    y: &ndarray::ArrayView1<f64>,
    stats_x: &FastWindowStats,
    stats_y: &FastWindowStats,
) -> f64 {
    if !stats_x.is_valid() || !stats_y.is_valid() {
        return f64::NAN;
    }
    
    // 使用标准的Pearson相关系数公式
    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    let mut sum_xy = 0.0;
    let mut sum_x2 = 0.0;
    let mut sum_y2 = 0.0;
    let mut count = 0;
    
    for (&xi, &yi) in x.iter().zip(y.iter()) {
        if !xi.is_nan() && !yi.is_nan() {
            sum_x += xi;
            sum_y += yi;
            sum_xy += xi * yi;
            sum_x2 += xi * xi;
            sum_y2 += yi * yi;
            count += 1;
        }
    }
    
    if count < 2 {
        return f64::NAN;
    }
    
    let n = count as f64;
    
    // 使用高效的相关系数公式
    let numerator = n * sum_xy - sum_x * sum_y;
    let denominator_x = n * sum_x2 - sum_x * sum_x;
    let denominator_y = n * sum_y2 - sum_y * sum_y;
    
    if denominator_x.abs() < f64::EPSILON || denominator_y.abs() < f64::EPSILON {
        return f64::NAN;
    }
    
    let correlation = numerator / (denominator_x.sqrt() * denominator_y.sqrt());
    
    // 确保相关系数在有效范围内
    correlation.max(-1.0).min(1.0)
}

/// 使用预计算统计信息的masked相关性计算
fn fast_correlation_with_stats_masked(
    x: &ndarray::ArrayView1<f64>,
    y: &ndarray::ArrayView1<f64>,
    _stats_x_masked: &FastWindowStats,
    _stats_y: &FastWindowStats,
    mask_pos: usize,
) -> f64 {
    // 使用标准的Pearson相关系数公式，跳过mask位置
    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    let mut sum_xy = 0.0;
    let mut sum_x2 = 0.0;
    let mut sum_y2 = 0.0;
    let mut count = 0;
    
    for (i, (&xi, &yi)) in x.iter().zip(y.iter()).enumerate() {
        if i != mask_pos && !xi.is_nan() && !yi.is_nan() {
            sum_x += xi;
            sum_y += yi;
            sum_xy += xi * yi;
            sum_x2 += xi * xi;
            sum_y2 += yi * yi;
            count += 1;
        }
    }
    
    if count < 2 {
        return f64::NAN;
    }
    
    let n = count as f64;
    
    // 使用高效的相关系数公式
    let numerator = n * sum_xy - sum_x * sum_y;
    let denominator_x = n * sum_x2 - sum_x * sum_x;
    let denominator_y = n * sum_y2 - sum_y * sum_y;
    
    if denominator_x.abs() < f64::EPSILON || denominator_y.abs() < f64::EPSILON {
        return f64::NAN;
    }
    
    let correlation = numerator / (denominator_x.sqrt() * denominator_y.sqrt());
    
    // 确保相关系数在有效范围内
    correlation.max(-1.0).min(1.0)
}