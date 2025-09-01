use pyo3::prelude::*;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray2};
use ndarray::{Array1, Array2};
use rayon::prelude::*;
use std::sync::Arc;

/// 滚动窗口计算相关性矩阵均值
/// 
/// 对于输入数据的每一行，计算其过去window_size行的相关性矩阵，
/// 然后计算该相关性矩阵中所有值的均值。
/// 
/// 参数说明：
/// ----------
/// data : numpy.ndarray
///     输入数据矩阵，形状为(n_samples, n_features)，每列代表一个变量
/// window_size : int
///     滚动窗口的大小，表示向前取多少行数据计算相关性矩阵
/// min_periods : int, 可选
///     计算相关性所需的最小样本数，默认为window_size
/// max_workers : int, 可选
///     最大并行工作线程数，默认为10，设置为0则使用所有可用核心
/// 
/// 返回值：
/// -------
/// numpy.ndarray
///     一维数组，长度等于输入数据的行数，每个值为对应行的相关性矩阵均值
///     
/// Python调用示例：
/// ```python
/// import numpy as np
/// import pandas as pd
/// from rust_pyfunc import rolling_correlation_mean
/// 
/// # 创建测试数据
/// data = np.random.randn(1000, 20).astype(np.float64)
/// window_size = 50
/// 
/// # 使用Rust版本计算
/// rust_result = rolling_correlation_mean(data, window_size)
/// 
/// # 与pandas版本对比（纯Python实现）
/// df = pd.DataFrame(data)
/// python_result = []
/// for i in range(len(df)):
///     start_idx = max(0, i + 1 - window_size)
///     window_df = df.iloc[start_idx:i+1]
///     if len(window_df) >= window_size:
///         corr_matrix = window_df.corr().values
///         mean_corr = np.nanmean(corr_matrix)
///         python_result.append(mean_corr)
///     else:
///         python_result.append(np.nan)
/// 
/// # 验证结果一致性
/// print(f"结果差异最大值: {np.nanmax(np.abs(rust_result - python_result))}")
/// ```
#[pyfunction]
#[pyo3(signature = (data, window_size, min_periods=None, max_workers=10))]
pub fn rolling_correlation_mean(
    py: Python,
    data: PyReadonlyArray2<f64>,
    window_size: usize,
    min_periods: Option<usize>,
    max_workers: Option<usize>,
) -> PyResult<Py<PyArray1<f64>>> {
    let data = data.as_array();
    let min_periods = min_periods.unwrap_or(window_size);
    let max_workers = max_workers.unwrap_or(10);
    
    let n_samples = data.nrows();
    let n_features = data.ncols();
    
    if window_size == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "窗口大小必须大于0",
        ));
    }
    
    if min_periods == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "最小周期数必须大于0",
        ));
    }
    
    // 创建线程池
    let pool = if max_workers > 0 && max_workers < rayon::current_num_threads() {
        Some(rayon::ThreadPoolBuilder::new()
            .num_threads(max_workers)
            .build()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                format!("创建线程池失败: {}", e)
            ))?)
    } else {
        None
    };
    
    // 转换数据为Arc以便在线程间共享
    let data_arc = Arc::new(data.to_owned());
    
    // 定义计算函数
    let compute_with_pool = |f: Box<dyn FnOnce() -> Vec<f64> + Send>| {
        if let Some(ref pool) = pool {
            pool.install(|| f())
        } else {
            f()
        }
    };
    
    // 并行计算每一行的滚动窗口相关性矩阵均值
    let result = compute_with_pool(Box::new(move || {
        (0..n_samples)
            .into_par_iter()
            .map(|i| {
                compute_rolling_mean_for_row(&data_arc, i, window_size, min_periods, n_features)
            })
            .collect()
    }));
    
    let result_array = Array1::from_vec(result);
    Ok(result_array.into_pyarray(py).to_owned())
}

/// 计算单行的滚动窗口相关性矩阵均值
fn compute_rolling_mean_for_row(
    data: &Array2<f64>,
    current_row: usize,
    window_size: usize,
    min_periods: usize,
    n_features: usize,
) -> f64 {
    // 确定窗口的起始和结束位置
    let window_start = if current_row + 1 >= window_size {
        current_row + 1 - window_size
    } else {
        0
    };
    let window_end = current_row + 1;
    let actual_window_size = window_end - window_start;
    
    // 检查是否有足够的数据点
    if actual_window_size < min_periods {
        return f64::NAN;
    }
    
    // 提取窗口数据
    let window_data = data.slice(ndarray::s![window_start..window_end, ..]);
    
    // 计算相关性矩阵
    let corr_matrix = match compute_correlation_matrix_fast(&window_data, n_features) {
        Some(matrix) => matrix,
        None => return f64::NAN,
    };
    
    // 计算相关性矩阵所有值的均值
    compute_matrix_mean(&corr_matrix)
}

/// 快速计算相关性矩阵
fn compute_correlation_matrix_fast(
    data: &ndarray::ArrayView2<f64>,
    n_features: usize,
) -> Option<Array2<f64>> {
    let n_samples = data.nrows();
    
    if n_samples < 2 {
        return None;
    }
    
    // 预计算每列的统计信息
    let col_stats: Vec<ColumnStats> = (0..n_features)
        .map(|i| {
            let col = data.column(i);
            compute_column_stats(&col)
        })
        .collect();
    
    // 检查是否有足够的有效列
    let valid_cols: Vec<usize> = col_stats
        .iter()
        .enumerate()
        .filter_map(|(i, stats)| if stats.is_valid() { Some(i) } else { None })
        .collect();
    
    if valid_cols.len() < 2 {
        return None;
    }
    
    // 创建相关性矩阵
    let mut corr_matrix = Array2::from_elem((n_features, n_features), f64::NAN);
    
    // 对角线设为1.0
    for i in 0..n_features {
        corr_matrix[[i, i]] = 1.0;
    }
    
    // 计算上三角矩阵（相关性矩阵是对称的）
    for i in 0..n_features {
        for j in i + 1..n_features {
            if col_stats[i].is_valid() && col_stats[j].is_valid() {
                let col_i = data.column(i);
                let col_j = data.column(j);
                
                if let Some(corr) = compute_correlation_between_columns(
                    &col_i, &col_j, &col_stats[i], &col_stats[j]
                ) {
                    corr_matrix[[i, j]] = corr;
                    corr_matrix[[j, i]] = corr; // 利用对称性
                }
            }
        }
    }
    
    Some(corr_matrix)
}

/// 列统计信息结构
#[derive(Clone)]
struct ColumnStats {
    #[allow(dead_code)]
    mean: f64,
    std_dev: f64,
    #[allow(dead_code)]
    sum_x: f64,
    valid_count: usize,
    valid_indices: Vec<usize>,
}

impl ColumnStats {
    fn is_valid(&self) -> bool {
        self.valid_count >= 2 && self.std_dev > f64::EPSILON
    }
}

/// 计算列统计信息
fn compute_column_stats(col: &ndarray::ArrayView1<f64>) -> ColumnStats {
    let mut sum = 0.0;
    let mut valid_indices = Vec::new();
    
    // 收集有效值和索引
    for (idx, &val) in col.iter().enumerate() {
        if !val.is_nan() {
            sum += val;
            valid_indices.push(idx);
        }
    }
    
    let valid_count = valid_indices.len();
    
    if valid_count < 2 {
        return ColumnStats {
            mean: f64::NAN,
            std_dev: f64::NAN,
            sum_x: f64::NAN,
            valid_count: 0,
            valid_indices: Vec::new(),
        };
    }
    
    let mean = sum / valid_count as f64;
    
    // 计算标准差
    let sum_sq_dev: f64 = valid_indices
        .iter()
        .map(|&idx| {
            let dev = col[idx] - mean;
            dev * dev
        })
        .sum();
    
    let std_dev = (sum_sq_dev / valid_count as f64).sqrt();
    
    ColumnStats {
        mean,
        std_dev,
        sum_x: sum,
        valid_count,
        valid_indices,
    }
}

/// 计算两列之间的相关性
fn compute_correlation_between_columns(
    col_i: &ndarray::ArrayView1<f64>,
    col_j: &ndarray::ArrayView1<f64>,
    stats_i: &ColumnStats,
    stats_j: &ColumnStats,
) -> Option<f64> {
    if !stats_i.is_valid() || !stats_j.is_valid() {
        return None;
    }
    
    // 找到两列都有效的索引
    let mut common_indices = Vec::new();
    let mut i_ptr = 0;
    let mut j_ptr = 0;
    
    while i_ptr < stats_i.valid_indices.len() && j_ptr < stats_j.valid_indices.len() {
        let i_idx = stats_i.valid_indices[i_ptr];
        let j_idx = stats_j.valid_indices[j_ptr];
        
        if i_idx == j_idx {
            common_indices.push(i_idx);
            i_ptr += 1;
            j_ptr += 1;
        } else if i_idx < j_idx {
            i_ptr += 1;
        } else {
            j_ptr += 1;
        }
    }
    
    let common_count = common_indices.len();
    if common_count < 2 {
        return None;
    }
    
    // 重新计算共同有效值的均值
    let mut sum_i = 0.0;
    let mut sum_j = 0.0;
    
    for &idx in &common_indices {
        sum_i += col_i[idx];
        sum_j += col_j[idx];
    }
    
    let mean_i = sum_i / common_count as f64;
    let mean_j = sum_j / common_count as f64;
    
    // 计算协方差和方差
    let mut cov_ij = 0.0;
    let mut var_i = 0.0;
    let mut var_j = 0.0;
    
    for &idx in &common_indices {
        let dev_i = col_i[idx] - mean_i;
        let dev_j = col_j[idx] - mean_j;
        
        cov_ij += dev_i * dev_j;
        var_i += dev_i * dev_i;
        var_j += dev_j * dev_j;
    }
    
    // 计算相关系数
    if var_i.abs() < f64::EPSILON || var_j.abs() < f64::EPSILON {
        return None;
    }
    
    Some(cov_ij / (var_i.sqrt() * var_j.sqrt()))
}

/// 滚动窗口计算相关性矩阵偏度
/// 
/// 对于输入数据的每一行，计算其过去window_size行的相关性矩阵，
/// 然后计算该相关性矩阵中所有值的偏度（skewness）。
/// 偏度衡量相关性分布的不对称性：
/// - 偏度 > 0：右偏（正偏），大部分相关性值较小，少数值较大
/// - 偏度 < 0：左偏（负偏），大部分相关性值较大，少数值较小
/// - 偏度 ≈ 0：接近对称分布
/// 
/// 参数说明：
/// ----------
/// data : numpy.ndarray
///     输入数据矩阵，形状为(n_samples, n_features)，每列代表一个变量
/// window_size : int
///     滚动窗口的大小，表示向前取多少行数据计算相关性矩阵
/// min_periods : int, 可选
///     计算相关性所需的最小样本数，默认为window_size
/// max_workers : int, 可选
///     最大并行工作线程数，默认为10，设置为0则使用所有可用核心
/// 
/// 返回值：
/// -------
/// numpy.ndarray
///     一维数组，长度等于输入数据的行数，每个值为对应行的相关性矩阵偏度
///     
/// Python调用示例：
/// ```python
/// import numpy as np
/// from rust_pyfunc import rolling_correlation_skew
/// 
/// # 创建测试数据
/// data = np.random.randn(1000, 20).astype(np.float64)
/// window_size = 50
/// 
/// # 计算滚动相关性矩阵偏度
/// skew_result = rolling_correlation_skew(data, window_size)
/// 
/// # 分析结果
/// valid_skews = skew_result[~np.isnan(skew_result)]
/// print(f"偏度范围: {np.min(valid_skews):.4f} ~ {np.max(valid_skews):.4f}")
/// print(f"平均偏度: {np.mean(valid_skews):.4f}")
/// 
/// # 正偏度表示大多数相关性较低，少数相关性较高
/// # 负偏度表示大多数相关性较高，少数相关性较低
/// ```
#[pyfunction]
#[pyo3(signature = (data, window_size, min_periods=None, max_workers=10))]
pub fn rolling_correlation_skew(
    py: Python,
    data: PyReadonlyArray2<f64>,
    window_size: usize,
    min_periods: Option<usize>,
    max_workers: Option<usize>,
) -> PyResult<Py<PyArray1<f64>>> {
    let data = data.as_array();
    let min_periods = min_periods.unwrap_or(window_size);
    let max_workers = max_workers.unwrap_or(10);
    
    let n_samples = data.nrows();
    let n_features = data.ncols();
    
    if window_size == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "窗口大小必须大于0",
        ));
    }
    
    if min_periods == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "最小周期数必须大于0",
        ));
    }
    
    // 创建线程池
    let pool = if max_workers > 0 && max_workers < rayon::current_num_threads() {
        Some(rayon::ThreadPoolBuilder::new()
            .num_threads(max_workers)
            .build()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                format!("创建线程池失败: {}", e)
            ))?)
    } else {
        None
    };
    
    // 转换数据为Arc以便在线程间共享
    let data_arc = Arc::new(data.to_owned());
    
    // 定义计算函数
    let compute_with_pool = |f: Box<dyn FnOnce() -> Vec<f64> + Send>| {
        if let Some(ref pool) = pool {
            pool.install(|| f())
        } else {
            f()
        }
    };
    
    // 并行计算每一行的滚动窗口相关性矩阵偏度
    let result = compute_with_pool(Box::new(move || {
        (0..n_samples)
            .into_par_iter()
            .map(|i| {
                compute_rolling_skew_for_row(&data_arc, i, window_size, min_periods, n_features)
            })
            .collect()
    }));
    
    let result_array = Array1::from_vec(result);
    Ok(result_array.into_pyarray(py).to_owned())
}

/// 计算单行的滚动窗口相关性矩阵偏度
fn compute_rolling_skew_for_row(
    data: &Array2<f64>,
    current_row: usize,
    window_size: usize,
    min_periods: usize,
    n_features: usize,
) -> f64 {
    // 确定窗口的起始和结束位置
    let window_start = if current_row + 1 >= window_size {
        current_row + 1 - window_size
    } else {
        0
    };
    let window_end = current_row + 1;
    let actual_window_size = window_end - window_start;
    
    // 检查是否有足够的数据点
    if actual_window_size < min_periods {
        return f64::NAN;
    }
    
    // 提取窗口数据
    let window_data = data.slice(ndarray::s![window_start..window_end, ..]);
    
    // 计算相关性矩阵
    let corr_matrix = match compute_correlation_matrix_fast(&window_data, n_features) {
        Some(matrix) => matrix,
        None => return f64::NAN,
    };
    
    // 计算相关性矩阵所有值的偏度
    compute_matrix_skew(&corr_matrix)
}

/// 计算矩阵所有元素的均值（使用Kahan求和算法提高精度）
fn compute_matrix_mean(matrix: &Array2<f64>) -> f64 {
    let mut sum = 0.0;
    let mut compensation = 0.0;
    let mut count = 0;
    
    for &value in matrix.iter() {
        if !value.is_nan() {
            // Kahan求和算法
            let y = value - compensation;
            let temp = sum + y;
            compensation = (temp - sum) - y;
            sum = temp;
            count += 1;
        }
    }
    
    if count > 0 {
        sum / count as f64
    } else {
        f64::NAN
    }
}

/// 计算矩阵所有元素的偏度（skewness）
fn compute_matrix_skew(matrix: &Array2<f64>) -> f64 {
    // 收集所有有效值
    let valid_values: Vec<f64> = matrix
        .iter()
        .filter(|&&value| !value.is_nan())
        .cloned()
        .collect();
    
    let n = valid_values.len();
    
    // 需要至少3个值来计算偏度
    if n < 3 {
        return f64::NAN;
    }
    
    // 计算均值（使用Kahan求和算法）
    let mut sum = 0.0;
    let mut compensation = 0.0;
    
    for &value in &valid_values {
        let y = value - compensation;
        let temp = sum + y;
        compensation = (temp - sum) - y;
        sum = temp;
    }
    
    let mean = sum / n as f64;
    
    // 计算二阶和三阶中心矩
    let mut m2 = 0.0;  // 方差
    let mut m3 = 0.0;  // 三阶中心矩
    
    for &value in &valid_values {
        let dev = value - mean;
        let dev2 = dev * dev;
        let dev3 = dev2 * dev;
        
        m2 += dev2;
        m3 += dev3;
    }
    
    m2 /= n as f64;
    m3 /= n as f64;
    
    // 计算偏度：skewness = m3 / (m2^1.5)
    if m2.abs() < f64::EPSILON {
        // 如果方差为0，偏度未定义
        return f64::NAN;
    }
    
    let skewness = m3 / m2.powf(1.5);
    
    // 应用样本校正（使用Fisher-Pearson标准化的偏度系数）
    if n > 2 {
        let correction = ((n * (n - 1)) as f64).sqrt() / (n - 2) as f64;
        skewness * correction
    } else {
        skewness
    }
}