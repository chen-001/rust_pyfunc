use ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;
use std::sync::Arc;

/// 快速计算相关性矩阵，类似于pandas的df.corr()功能。
/// 使用并行计算和优化算法大幅提升计算性能。
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
///
/// Python调用示例：
/// ```python
/// import numpy as np
/// import pandas as pd
/// from rust_pyfunc import fast_correlation_matrix
///
/// # 创建测试数据
/// data = np.random.randn(1000, 50).astype(np.float64)
///
/// # 使用快速相关性矩阵计算
/// rust_corr = fast_correlation_matrix(data)
///
/// # 与pandas结果对比
/// df = pd.DataFrame(data)
/// pandas_corr = df.corr().values
///
/// # 验证结果一致性
/// print(f"结果差异最大值: {np.max(np.abs(rust_corr - pandas_corr))}")
/// ```
#[pyfunction]
#[pyo3(signature = (data, method="pearson", min_periods=1, max_workers=10))]
pub fn fast_correlation_matrix(
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

    // 目前只支持皮尔逊相关系数
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

    // 创建局部线程池来控制并行度
    let pool = if max_workers > 0 && max_workers < rayon::current_num_threads() {
        Some(
            rayon::ThreadPoolBuilder::new()
                .num_threads(max_workers)
                .build()
                .map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                        "创建线程池失败: {}",
                        e
                    ))
                })?,
        )
    } else {
        None
    };

    // 定义计算函数，使用适当的线程池
    let compute_with_pool = |f: Box<dyn FnOnce() -> Vec<ColumnStats> + Send>| {
        if let Some(ref pool) = pool {
            pool.install(|| f())
        } else {
            f()
        }
    };

    let compute_correlations_with_pool =
        |f: Box<dyn FnOnce() -> Vec<((usize, usize), f64)> + Send>| {
            if let Some(ref pool) = pool {
                pool.install(|| f())
            } else {
                f()
            }
        };

    // 预计算列的统计信息，避免重复计算
    let data_arc = Arc::new(data.to_owned());
    let col_stats = compute_with_pool(Box::new(move || {
        (0..n_features)
            .into_par_iter()
            .map(|i| {
                let col = data_arc.column(i);
                compute_column_stats(&col, min_periods)
            })
            .collect()
    }));

    // 创建结果矩阵
    let mut corr_matrix = Array2::from_elem((n_features, n_features), f64::NAN);

    // 对角线设为1.0
    for i in 0..n_features {
        corr_matrix[[i, i]] = 1.0;
    }

    // 并行计算上三角矩阵（相关矩阵是对称的）
    let upper_triangle_indices: Vec<(usize, usize)> = (0..n_features)
        .flat_map(|i| (i + 1..n_features).map(move |j| (i, j)))
        .collect();

    let data_arc2 = Arc::new(data.to_owned());
    let col_stats_arc = Arc::new(col_stats);
    let correlations = compute_correlations_with_pool(Box::new(move || {
        upper_triangle_indices
            .into_par_iter()
            .map(|(i, j)| {
                let col_i = data_arc2.column(i);
                let col_j = data_arc2.column(j);
                let corr = compute_correlation_fast(
                    &col_i,
                    &col_j,
                    &col_stats_arc[i],
                    &col_stats_arc[j],
                    min_periods,
                );
                ((i, j), corr)
            })
            .collect()
    }));

    // 填充相关性矩阵
    for ((i, j), corr) in correlations {
        corr_matrix[[i, j]] = corr;
        corr_matrix[[j, i]] = corr; // 利用对称性
    }

    Ok(corr_matrix.into_pyarray(py).to_owned())
}

/// 列统计信息结构体，用于缓存计算结果
#[derive(Clone)]
#[allow(dead_code)]
struct ColumnStats {
    mean: f64,
    sum_sq_dev: f64,
    valid_count: usize,
    valid_indices: Vec<usize>,
}

/// 计算列的统计信息
fn compute_column_stats(col: &ndarray::ArrayView1<f64>, min_periods: usize) -> ColumnStats {
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

    if valid_count < min_periods {
        return ColumnStats {
            mean: f64::NAN,
            sum_sq_dev: f64::NAN,
            valid_count: 0,
            valid_indices: Vec::new(),
        };
    }

    let mean = sum / valid_count as f64;

    // 计算平方偏差的和
    let sum_sq_dev: f64 = valid_indices
        .iter()
        .map(|&idx| {
            let dev = col[idx] - mean;
            dev * dev
        })
        .sum();

    ColumnStats {
        mean,
        sum_sq_dev,
        valid_count,
        valid_indices,
    }
}

/// 快速计算两列之间的相关性
fn compute_correlation_fast(
    col_i: &ndarray::ArrayView1<f64>,
    col_j: &ndarray::ArrayView1<f64>,
    stats_i: &ColumnStats,
    stats_j: &ColumnStats,
    min_periods: usize,
) -> f64 {
    // 如果任一列的统计信息无效，返回NaN
    if stats_i.valid_count < min_periods || stats_j.valid_count < min_periods {
        return f64::NAN;
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
    if common_count < min_periods {
        return f64::NAN;
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

    // 使用批处理优化内存访问模式
    const BATCH_SIZE: usize = 64;
    for chunk in common_indices.chunks(BATCH_SIZE) {
        let mut local_cov = 0.0;
        let mut local_var_i = 0.0;
        let mut local_var_j = 0.0;

        for &idx in chunk {
            let dev_i = col_i[idx] - mean_i;
            let dev_j = col_j[idx] - mean_j;

            local_cov += dev_i * dev_j;
            local_var_i += dev_i * dev_i;
            local_var_j += dev_j * dev_j;
        }

        cov_ij += local_cov;
        var_i += local_var_i;
        var_j += local_var_j;
    }

    // 计算相关系数
    if var_i.abs() < f64::EPSILON || var_j.abs() < f64::EPSILON {
        return f64::NAN;
    }

    cov_ij / (var_i.sqrt() * var_j.sqrt())
}
