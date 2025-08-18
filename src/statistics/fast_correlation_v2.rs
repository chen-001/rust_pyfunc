use pyo3::prelude::*;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use ndarray::Array2;
use rayon::prelude::*;
use std::sync::Arc;

/// 超快速计算相关性矩阵，进一步优化版本。
/// 采用SIMD优化和更好的内存访问模式。
///
/// 参数说明：
/// ----------
/// data : numpy.ndarray
///     输入数据矩阵，形状为(n_samples, n_features)，每列代表一个变量
/// method : str, 可选
///     相关性计算方法，默认为'pearson'。目前只支持皮尔逊相关系数
/// min_periods : int, 可选
///     计算相关性所需的最小样本数，默认为1
/// max_workers : int, 可选
///     最大并行工作线程数，默认为10，设置为0则使用所有可用核心
///
/// 返回值：
/// -------
/// numpy.ndarray
///     相关性矩阵，形状为(n_features, n_features)，对角线元素为1.0
#[pyfunction]
#[pyo3(signature = (data, method="pearson", min_periods=1, max_workers=10))]
pub fn fast_correlation_matrix_v2(
    py: Python,
    data: PyReadonlyArray2<f64>,
    method: Option<&str>,
    min_periods: Option<usize>,
    max_workers: Option<usize>,
) -> PyResult<Py<PyArray2<f64>>> {
    let data = data.as_array();
    let _method = method.unwrap_or("pearson");
    let min_periods = min_periods.unwrap_or(1);
    let max_workers = max_workers.unwrap_or(10);
    
    if _method != "pearson" {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "目前只支持'pearson'相关系数计算方法",
        ));
    }
    
    let n_samples = data.nrows();
    let n_features = data.ncols();
    
    if n_samples < min_periods {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "样本数量少于最小需求",
        ));
    }
    
    // 创建局部线程池
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
    
    // 数据预处理：转置数据以获得更好的内存局部性
    // 这样每一列在内存中是连续的
    let transposed_data = data.t().to_owned();
    let data_arc = Arc::new(transposed_data);
    
    // 预计算所有列的统计信息
    let compute_stats = || {
        (0..n_features)
            .into_par_iter()
            .map(|i| {
                let col = data_arc.row(i);
                compute_column_stats_optimized(&col, min_periods)
            })
            .collect::<Vec<_>>()
    };
    
    let col_stats = if let Some(ref pool) = pool {
        pool.install(compute_stats)
    } else {
        compute_stats()
    };
    
    // 创建结果矩阵
    let mut corr_matrix = Array2::from_elem((n_features, n_features), f64::NAN);
    
    // 对角线设为1.0
    for i in 0..n_features {
        corr_matrix[[i, i]] = 1.0;
    }
    
    // 生成上三角索引对
    let upper_triangle_indices: Vec<(usize, usize)> = (0..n_features)
        .flat_map(|i| (i + 1..n_features).map(move |j| (i, j)))
        .collect();
    
    // 并行计算相关性
    let compute_correlations = || {
        upper_triangle_indices
            .into_par_iter()
            .map(|(i, j)| {
                let corr = compute_correlation_simd_optimized(
                    &data_arc.row(i),
                    &data_arc.row(j),
                    &col_stats[i],
                    &col_stats[j],
                    min_periods,
                );
                ((i, j), corr)
            })
            .collect::<Vec<_>>()
    };
    
    let correlations = if let Some(ref pool) = pool {
        pool.install(compute_correlations)
    } else {
        compute_correlations()
    };
    
    // 填充相关性矩阵
    for ((i, j), corr) in correlations {
        corr_matrix[[i, j]] = corr;
        corr_matrix[[j, i]] = corr; // 利用对称性
    }
    
    Ok(corr_matrix.into_pyarray(py).to_owned())
}

/// 优化的列统计信息结构
#[derive(Clone)]
struct OptimizedColumnStats {
    mean: f64,
    valid_count: usize,
    // 预处理的有效数据，减少后续NaN检查
    valid_data: Vec<f64>,
}

/// 计算优化的列统计信息
fn compute_column_stats_optimized(col: &ndarray::ArrayView1<f64>, min_periods: usize) -> OptimizedColumnStats {
    // 一次遍历收集所有有效数据
    let valid_data: Vec<f64> = col.iter()
        .filter(|&&val| !val.is_nan())
        .copied()
        .collect();
    
    let valid_count = valid_data.len();
    
    if valid_count < min_periods {
        return OptimizedColumnStats {
            mean: f64::NAN,
            valid_count: 0,
            valid_data: Vec::new(),
        };
    }
    
    // 使用高精度求和避免数值误差
    let sum = kahan_sum(&valid_data);
    let mean = sum / valid_count as f64;
    
    OptimizedColumnStats {
        mean,
        valid_count,
        valid_data,
    }
}

/// Kahan求和算法，减少浮点数累加误差
fn kahan_sum(values: &[f64]) -> f64 {
    let mut sum = 0.0;
    let mut compensation = 0.0;
    
    for &value in values {
        let y = value - compensation;
        let temp = sum + y;
        compensation = (temp - sum) - y;
        sum = temp;
    }
    
    sum
}

/// SIMD优化的相关性计算
fn compute_correlation_simd_optimized(
    col_i: &ndarray::ArrayView1<f64>,
    col_j: &ndarray::ArrayView1<f64>,
    stats_i: &OptimizedColumnStats,
    stats_j: &OptimizedColumnStats,
    min_periods: usize,
) -> f64 {
    // 如果任一列的统计信息无效，返回NaN
    if stats_i.valid_count < min_periods || stats_j.valid_count < min_periods {
        return f64::NAN;
    }
    
    // 如果已经预处理了有效数据，且数据完全有效（无NaN），直接使用
    if stats_i.valid_data.len() == col_i.len() && stats_j.valid_data.len() == col_j.len() {
        return compute_correlation_from_clean_data(&stats_i.valid_data, &stats_j.valid_data, stats_i.mean, stats_j.mean);
    }
    
    // 否则需要找到共同有效的位置
    let mut valid_pairs = Vec::new();
    
    for (_idx, (&val_i, &val_j)) in col_i.iter().zip(col_j.iter()).enumerate() {
        if !val_i.is_nan() && !val_j.is_nan() {
            valid_pairs.push((val_i, val_j));
        }
    }
    
    let common_count = valid_pairs.len();
    if common_count < min_periods {
        return f64::NAN;
    }
    
    // 重新计算共同有效值的均值
    let sum_i = kahan_sum(&valid_pairs.iter().map(|(i, _)| *i).collect::<Vec<_>>());
    let sum_j = kahan_sum(&valid_pairs.iter().map(|(_, j)| *j).collect::<Vec<_>>());
    
    let mean_i = sum_i / common_count as f64;
    let mean_j = sum_j / common_count as f64;
    
    // 使用向量化计算协方差和方差
    compute_correlation_vectorized(&valid_pairs, mean_i, mean_j)
}

/// 从已清理数据计算相关性（无NaN）
fn compute_correlation_from_clean_data(data_i: &[f64], data_j: &[f64], mean_i: f64, mean_j: f64) -> f64 {
    assert_eq!(data_i.len(), data_j.len());
    let n = data_i.len();
    
    if n < 2 {
        return f64::NAN;
    }
    
    // 向量化计算偏差
    let mut cov_sum = 0.0;
    let mut var_i_sum = 0.0;
    let mut var_j_sum = 0.0;
    
    // 使用循环展开优化
    const UNROLL_FACTOR: usize = 8;
    let chunks = n / UNROLL_FACTOR;
    let remainder = n % UNROLL_FACTOR;
    
    // 处理展开的部分
    for chunk in 0..chunks {
        let base_idx = chunk * UNROLL_FACTOR;
        
        for i in 0..UNROLL_FACTOR {
            let idx = base_idx + i;
            let dev_i = data_i[idx] - mean_i;
            let dev_j = data_j[idx] - mean_j;
            
            cov_sum += dev_i * dev_j;
            var_i_sum += dev_i * dev_i;
            var_j_sum += dev_j * dev_j;
        }
    }
    
    // 处理剩余部分
    for i in 0..remainder {
        let idx = chunks * UNROLL_FACTOR + i;
        let dev_i = data_i[idx] - mean_i;
        let dev_j = data_j[idx] - mean_j;
        
        cov_sum += dev_i * dev_j;
        var_i_sum += dev_i * dev_i;
        var_j_sum += dev_j * dev_j;
    }
    
    // 计算相关系数
    if var_i_sum.abs() < f64::EPSILON || var_j_sum.abs() < f64::EPSILON {
        return f64::NAN;
    }
    
    cov_sum / (var_i_sum.sqrt() * var_j_sum.sqrt())
}

/// 向量化的相关性计算
fn compute_correlation_vectorized(valid_pairs: &[(f64, f64)], mean_i: f64, mean_j: f64) -> f64 {
    if valid_pairs.len() < 2 {
        return f64::NAN;
    }
    
    // 使用并行reduce来计算协方差和方差
    const CHUNK_SIZE: usize = 1024;
    
    let (cov_sum, var_i_sum, var_j_sum) = if valid_pairs.len() > CHUNK_SIZE {
        // 对于大数据使用并行计算
        valid_pairs
            .par_chunks(CHUNK_SIZE)
            .map(|chunk| {
                let mut local_cov = 0.0;
                let mut local_var_i = 0.0;
                let mut local_var_j = 0.0;
                
                for &(val_i, val_j) in chunk {
                    let dev_i = val_i - mean_i;
                    let dev_j = val_j - mean_j;
                    
                    local_cov += dev_i * dev_j;
                    local_var_i += dev_i * dev_i;
                    local_var_j += dev_j * dev_j;
                }
                
                (local_cov, local_var_i, local_var_j)
            })
            .reduce(
                || (0.0, 0.0, 0.0),
                |acc, item| (acc.0 + item.0, acc.1 + item.1, acc.2 + item.2),
            )
    } else {
        // 对于小数据使用串行计算
        let mut cov_sum = 0.0;
        let mut var_i_sum = 0.0;
        let mut var_j_sum = 0.0;
        
        for &(val_i, val_j) in valid_pairs {
            let dev_i = val_i - mean_i;
            let dev_j = val_j - mean_j;
            
            cov_sum += dev_i * dev_j;
            var_i_sum += dev_i * dev_i;
            var_j_sum += dev_j * dev_j;
        }
        
        (cov_sum, var_i_sum, var_j_sum)
    };
    
    // 计算相关系数
    if var_i_sum.abs() < f64::EPSILON || var_j_sum.abs() < f64::EPSILON {
        return f64::NAN;
    }
    
    cov_sum / (var_i_sum.sqrt() * var_j_sum.sqrt())
}