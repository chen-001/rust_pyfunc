use pyo3::prelude::*;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use ndarray::Array1;

/// 滚动窗口核心特征提取
/// 
/// 对输入序列进行滚动窗口分析，识别每个窗口中最重要的特征位置（核心特征）
/// 和最不重要的特征位置。通过计算窗口间的相关性并分析mask效应来确定特征重要性。
/// 
/// 算法原理：
/// 1. 对每个滚动窗口，计算其与所有其他窗口的相关系数（基准相关性）
/// 2. 依次将窗口内每个位置设为NaN，重新计算相关系数
/// 3. 相关性变化最小的位置为最重要特征（核心代表性）
/// 4. 相关性变化最大的位置为最不重要特征
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
///     两个数组的前(window_size-1)个位置均为NaN
/// 
/// Python调用示例：
/// ```python
/// import numpy as np
/// from rust_pyfunc import rolling_window_core_feature
/// 
/// # 创建测试数据
/// data = np.random.randn(1000).astype(np.float64)
/// 
/// # 使用默认窗口大小5
/// core_features, minor_features = rolling_window_core_feature(data)
/// 
/// # 使用自定义窗口大小10
/// core_features, minor_features = rolling_window_core_feature(data, window_size=10)
/// 
/// # 验证结果
/// print(f"核心特征前10个值: {core_features[:10]}")
/// print(f"次要特征前10个值: {minor_features[:10]}")
/// ```
#[pyfunction]
#[pyo3(signature = (values, window_size=5))]
pub fn rolling_window_core_feature(
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
    
    // 初始化结果数组，前(window_size-1)个位置为NaN
    let mut core_features = Array1::from_elem(n, f64::NAN);
    let mut minor_features = Array1::from_elem(n, f64::NAN);
    
    if n < window_size {
        return Ok((
            core_features.into_pyarray(py).to_owned(),
            minor_features.into_pyarray(py).to_owned()
        ));
    }
    
    // 计算所有可能的滚动窗口数量（预留用于优化）
    let _num_windows = n - window_size + 1;
    
    // 预计算所有窗口的基本统计信息
    let windows_stats = precompute_windows_stats(&values, window_size);
    
    // 对每个窗口进行分析（从第window_size-1个位置开始）
    for current_idx in (window_size - 1)..n {
        let window_start = current_idx - window_size + 1;
        
        // 获取当前窗口
        let current_window = values.slice(ndarray::s![window_start..=current_idx]);
        
        // 计算基准相关性：当前窗口与所有其他窗口的相关系数
        let base_correlations = compute_base_correlations(
            &current_window, 
            &values, 
            &windows_stats, 
            window_size,
            current_idx
        );
        
        // 分析每个位置的重要性
        let (core_pos, minor_pos) = analyze_position_importance(
            &current_window,
            &values,
            &windows_stats,
            &base_correlations,
            window_size,
            current_idx
        );
        
        // 存储结果
        if let Some(pos) = core_pos {
            core_features[current_idx] = current_window[pos];
        }
        
        if let Some(pos) = minor_pos {
            minor_features[current_idx] = current_window[pos];
        }
    }
    
    Ok((
        core_features.into_pyarray(py).to_owned(),
        minor_features.into_pyarray(py).to_owned()
    ))
}

/// 窗口统计信息结构
#[derive(Clone, Debug)]
struct WindowStats {
    #[allow(dead_code)]
    mean: f64,
    std_dev: f64,
    valid_count: usize,
    #[allow(dead_code)]
    start_idx: usize,
}

impl WindowStats {
    fn is_valid(&self) -> bool {
        self.valid_count >= 2 && self.std_dev > f64::EPSILON
    }
}

/// 预计算所有窗口的统计信息
fn precompute_windows_stats(values: &ndarray::ArrayView1<f64>, window_size: usize) -> Vec<WindowStats> {
    let n = values.len();
    let mut stats = Vec::new();
    
    for start_idx in 0..=(n - window_size) {
        let window = values.slice(ndarray::s![start_idx..start_idx + window_size]);
        
        // 计算有效值的统计信息
        let mut sum = 0.0;
        let mut valid_count = 0;
        
        for &val in window.iter() {
            if !val.is_nan() {
                sum += val;
                valid_count += 1;
            }
        }
        
        if valid_count == 0 {
            stats.push(WindowStats {
                mean: f64::NAN,
                std_dev: f64::NAN,
                valid_count: 0,
                start_idx,
            });
            continue;
        }
        
        let mean = sum / valid_count as f64;
        
        // 计算标准差
        let mut sum_sq_dev = 0.0;
        for &val in window.iter() {
            if !val.is_nan() {
                let dev = val - mean;
                sum_sq_dev += dev * dev;
            }
        }
        
        let std_dev = if valid_count > 1 {
            (sum_sq_dev / valid_count as f64).sqrt()
        } else {
            0.0
        };
        
        stats.push(WindowStats {
            mean,
            std_dev,
            valid_count,
            start_idx,
        });
    }
    
    stats
}

/// 计算基准相关性：当前窗口与所有其他窗口的相关系数
fn compute_base_correlations(
    current_window: &ndarray::ArrayView1<f64>,
    all_values: &ndarray::ArrayView1<f64>,
    windows_stats: &[WindowStats],
    window_size: usize,
    current_idx: usize,
) -> Vec<f64> {
    let mut correlations = Vec::new();
    let current_start = current_idx - window_size + 1;
    
    for (i, window_stat) in windows_stats.iter().enumerate() {
        // 跳过当前窗口本身
        if i == current_start {
            continue;
        }
        
        if !window_stat.is_valid() {
            continue;
        }
        
        let other_window = all_values.slice(ndarray::s![i..i + window_size]);
        
        if let Some(corr) = compute_correlation_fast(current_window, &other_window) {
            correlations.push(corr);
        }
    }
    
    correlations
}

/// 分析位置重要性：通过mask分析确定最重要和最不重要的位置
fn analyze_position_importance(
    current_window: &ndarray::ArrayView1<f64>,
    all_values: &ndarray::ArrayView1<f64>,
    windows_stats: &[WindowStats],
    base_correlations: &[f64],
    window_size: usize,
    current_idx: usize,
) -> (Option<usize>, Option<usize>) {
    let mut position_impacts = Vec::new();
    
    // 对窗口内每个位置进行分析
    for mask_pos in 0..window_size {
        // 直接计算masked相关性，避免创建新数组
        let masked_correlations = compute_masked_correlations_direct(
            current_window,
            all_values,
            windows_stats,
            window_size,
            current_idx,
            mask_pos,
        );
        
        // 计算相关性变化程度
        let impact = calculate_correlation_impact(base_correlations, &masked_correlations);
        position_impacts.push((mask_pos, impact));
    }
    
    // 根据相关性变化找出最重要和最不重要的位置
    let mut min_impact = f64::INFINITY;
    let mut max_impact = f64::NEG_INFINITY;
    let mut core_pos = None;
    let mut minor_pos = None;
    
    for (pos, impact) in position_impacts {
        if !impact.is_nan() {
            if impact < min_impact {
                min_impact = impact;
                core_pos = Some(pos); // 影响最小的是核心特征
            }
            if impact > max_impact {
                max_impact = impact;
                minor_pos = Some(pos); // 影响最大的是次要特征
            }
        }
    }
    
    (core_pos, minor_pos)
}

/// 直接计算masked窗口的相关性（避免内存分配）
fn compute_masked_correlations_direct(
    current_window: &ndarray::ArrayView1<f64>,
    all_values: &ndarray::ArrayView1<f64>,
    windows_stats: &[WindowStats],
    window_size: usize,
    current_idx: usize,
    mask_pos: usize,
) -> Vec<f64> {
    let mut correlations = Vec::new();
    let current_start = current_idx - window_size + 1;
    
    for (i, window_stat) in windows_stats.iter().enumerate() {
        // 跳过当前窗口本身
        if i == current_start {
            continue;
        }
        
        if !window_stat.is_valid() {
            continue;
        }
        
        let other_window = all_values.slice(ndarray::s![i..i + window_size]);
        
        if let Some(corr) = compute_correlation_masked(current_window, &other_window, mask_pos) {
            correlations.push(corr);
        }
    }
    
    correlations
}

/// 计算masked窗口的相关性（保留用于兼容）
fn compute_masked_correlations(
    masked_window: &ndarray::ArrayView1<f64>,
    all_values: &ndarray::ArrayView1<f64>,
    windows_stats: &[WindowStats],
    window_size: usize,
    current_idx: usize,
) -> Vec<f64> {
    let mut correlations = Vec::new();
    let current_start = current_idx - window_size + 1;
    
    for (i, window_stat) in windows_stats.iter().enumerate() {
        // 跳过当前窗口本身
        if i == current_start {
            continue;
        }
        
        if !window_stat.is_valid() {
            continue;
        }
        
        let other_window = all_values.slice(ndarray::s![i..i + window_size]);
        
        if let Some(corr) = compute_correlation_fast(masked_window, &other_window) {
            correlations.push(corr);
        }
    }
    
    correlations
}

/// 计算相关性变化的影响程度
fn calculate_correlation_impact(base_correlations: &[f64], masked_correlations: &[f64]) -> f64 {
    if base_correlations.is_empty() || masked_correlations.is_empty() {
        return f64::NAN;
    }
    
    // 计算两组相关性的相关系数，作为影响程度的度量
    if base_correlations.len() != masked_correlations.len() {
        return f64::NAN;
    }
    
    // 计算变化程度：相关系数的差异
    let mut sum_diff_sq = 0.0;
    let mut count = 0;
    
    for (base, masked) in base_correlations.iter().zip(masked_correlations.iter()) {
        if !base.is_nan() && !masked.is_nan() {
            let diff = base - masked;
            sum_diff_sq += diff * diff;
            count += 1;
        }
    }
    
    if count == 0 {
        return f64::NAN;
    }
    
    // 返回均方根误差作为影响程度
    (sum_diff_sq / count as f64).sqrt()
}

/// 快速计算两个序列的Pearson相关系数（内联优化版）
#[inline(always)]
fn compute_correlation_fast(
    x: &ndarray::ArrayView1<f64>, 
    y: &ndarray::ArrayView1<f64>
) -> Option<f64> {
    if x.len() != y.len() {
        return None;
    }
    
    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    let mut sum_xy = 0.0;
    let mut sum_x2 = 0.0;
    let mut sum_y2 = 0.0;
    let mut count = 0;
    
    // 一次遍历计算所有需要的统计量，展开循环以提高性能
    let mut i = 0;
    let len = x.len();
    
    // 处理4个元素为一组（向量化友好）
    while i + 3 < len {
        let x0 = x[i];
        let y0 = y[i];
        let x1 = x[i + 1];
        let y1 = y[i + 1];
        let x2 = x[i + 2];
        let y2 = y[i + 2];
        let x3 = x[i + 3];
        let y3 = y[i + 3];
        
        // 批量处理有效值
        if !x0.is_nan() && !y0.is_nan() {
            sum_x += x0; sum_y += y0; sum_xy += x0 * y0;
            sum_x2 += x0 * x0; sum_y2 += y0 * y0; count += 1;
        }
        if !x1.is_nan() && !y1.is_nan() {
            sum_x += x1; sum_y += y1; sum_xy += x1 * y1;
            sum_x2 += x1 * x1; sum_y2 += y1 * y1; count += 1;
        }
        if !x2.is_nan() && !y2.is_nan() {
            sum_x += x2; sum_y += y2; sum_xy += x2 * y2;
            sum_x2 += x2 * x2; sum_y2 += y2 * y2; count += 1;
        }
        if !x3.is_nan() && !y3.is_nan() {
            sum_x += x3; sum_y += y3; sum_xy += x3 * y3;
            sum_x2 += x3 * x3; sum_y2 += y3 * y3; count += 1;
        }
        
        i += 4;
    }
    
    // 处理剩余元素
    while i < len {
        let xi = x[i];
        let yi = y[i];
        if !xi.is_nan() && !yi.is_nan() {
            sum_x += xi;
            sum_y += yi;
            sum_xy += xi * yi;
            sum_x2 += xi * xi;
            sum_y2 += yi * yi;
            count += 1;
        }
        i += 1;
    }
    
    if count < 2 {
        return None;
    }
    
    let n = count as f64;
    
    // 使用高效的相关系数公式
    let numerator = n * sum_xy - sum_x * sum_y;
    let denominator_x = n * sum_x2 - sum_x * sum_x;
    let denominator_y = n * sum_y2 - sum_y * sum_y;
    
    if denominator_x.abs() < f64::EPSILON || denominator_y.abs() < f64::EPSILON {
        return None;
    }
    
    let correlation = numerator / (denominator_x.sqrt() * denominator_y.sqrt());
    
    // 处理数值误差，确保相关系数在[-1, 1]范围内
    Some(correlation.max(-1.0).min(1.0))
}

/// 计算masked相关系数（专门用于重要性分析）
#[inline(always)]
fn compute_correlation_masked(
    x: &ndarray::ArrayView1<f64>, 
    y: &ndarray::ArrayView1<f64>,
    mask_pos: usize
) -> Option<f64> {
    if x.len() != y.len() {
        return None;
    }
    
    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    let mut sum_xy = 0.0;
    let mut sum_x2 = 0.0;
    let mut sum_y2 = 0.0;
    let mut count = 0;
    
    // 一次遍历计算所有需要的统计量，跳过mask位置
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
        return None;
    }
    
    let n = count as f64;
    
    // 使用高效的相关系数公式
    let numerator = n * sum_xy - sum_x * sum_y;
    let denominator_x = n * sum_x2 - sum_x * sum_x;
    let denominator_y = n * sum_y2 - sum_y * sum_y;
    
    if denominator_x.abs() < f64::EPSILON || denominator_y.abs() < f64::EPSILON {
        return None;
    }
    
    let correlation = numerator / (denominator_x.sqrt() * denominator_y.sqrt());
    
    // 处理数值误差，确保相关系数在[-1, 1]范围内
    Some(correlation.max(-1.0).min(1.0))
}