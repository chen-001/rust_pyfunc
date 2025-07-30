use pyo3::prelude::*;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use ndarray::{Array2, ArrayView1};
use rayon::prelude::*;
use nalgebra::DMatrix;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// 安全版本的矩阵特征值分析函数，包含超时和数值稳定性检查
#[pyfunction]
#[pyo3(signature = (matrix, timeout_seconds = 30.0))]
pub fn matrix_eigenvalue_analysis_safe(
    py: Python,
    matrix: PyReadonlyArray2<f64>,
    timeout_seconds: f64,
) -> PyResult<Py<PyArray2<f64>>> {
    let input_matrix = matrix.as_array();
    let (n_rows, n_cols) = input_matrix.dim();
    
    if n_rows == 0 || n_cols == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            format!("输入矩阵不能为空，得到形状({}, {})", n_rows, n_cols)
        ));
    }
    
    // 检查矩阵尺寸限制
    if n_rows > 1000 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            format!("矩阵行数过大({})，为避免内存溢出，限制为1000行以内", n_rows)
        ));
    }
    
    let mut result = Array2::<f64>::zeros((n_rows, n_cols));
    let timeout_duration = Duration::from_secs_f64(timeout_seconds);
    let start_time = Instant::now();
    
    let thread_pool = rayon::ThreadPoolBuilder::new()
        .num_threads(std::cmp::min(4, num_cpus::get())) // 减少线程数避免资源竞争
        .build()
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("线程池创建失败: {}", e)))?;
    
    let input_arc = Arc::new(input_matrix.to_owned());
    
    thread_pool.install(|| {
        let eigenvalue_results: Vec<_> = (0..n_cols)
            .into_par_iter()
            .map(|col_idx| {
                // 检查全局超时
                if start_time.elapsed() > timeout_duration {
                    return (col_idx, vec![f64::NAN; n_rows]);
                }
                
                let column = input_arc.column(col_idx);
                
                match compute_safe_eigenvalues(&column, timeout_duration - start_time.elapsed()) {
                    Ok(eigenvalues) => (col_idx, eigenvalues),
                    Err(_) => (col_idx, vec![f64::NAN; n_rows]),
                }
            })
            .collect();
        
        for (col_idx, eigenvalues) in eigenvalue_results {
            for (i, &val) in eigenvalues.iter().enumerate() {
                if i < n_rows {
                    result[[i, col_idx]] = val;
                }
            }
        }
    });
    
    Ok(result.into_pyarray(py).to_owned())
}

/// 安全的单列特征值计算，包含数值稳定性检查和超时机制
fn compute_safe_eigenvalues(
    column: &ArrayView1<f64>, 
    remaining_timeout: Duration
) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    let start_time = Instant::now();
    let n = column.len();
    
    // 1. 数据有效性检查
    let mut valid_count = 0;
    let mut min_val = f64::INFINITY;
    let mut max_val = f64::NEG_INFINITY;
    let mut has_nan = false;
    let mut has_inf = false;
    
    for &val in column.iter() {
        if val.is_nan() {
            has_nan = true;
        } else if val.is_infinite() {
            has_inf = true;
        } else {
            valid_count += 1;
            min_val = min_val.min(val);
            max_val = max_val.max(val);
        }
    }
    
    // 检查数据质量
    if has_nan {
        return Err("数据包含NaN值".into());
    }
    
    if has_inf {
        return Err("数据包含无穷大值".into());
    }
    
    if valid_count < 2 {
        return Err("有效数据点少于2个".into());
    }
    
    // 检查数值范围
    let range = max_val - min_val;
    if range == 0.0 {
        // 所有值相同，返回全零特征值
        return Ok(vec![0.0; n]);
    }
    
    if range > 1e12 || range < 1e-12 {
        return Err("数据范围过大或过小，可能导致数值不稳定".into());
    }
    
    // 检查超时
    if start_time.elapsed() > remaining_timeout {
        return Err("数据预检查超时".into());
    }
    
    // 2. 构建差值矩阵（使用更安全的方法）
    let mut matrix_data = vec![0.0; n * n];
    
    for i in 0..n {
        if start_time.elapsed() > remaining_timeout {
            return Err("矩阵构建超时".into());
        }
        
        let val_i = column[i];
        let row_offset = i * n;
        
        for j in 0..n {
            if i != j {
                let val_j = column[j];
                let diff = val_i - val_j;
                
                // 检查差值是否在安全范围内
                if diff.abs() > 1e10 {
                    return Err("差值过大，可能导致数值溢出".into());
                }
                
                matrix_data[row_offset + j] = if i < j {
                    diff
                } else {
                    diff.abs()
                };
            }
        }
    }
    
    // 3. 创建矩阵并检查条件数
    let diff_matrix = DMatrix::from_vec(n, n, matrix_data);
    
    // 简单的条件数估计（使用矩阵的最大和最小非零元素）
    let mut max_elem = 0.0;
    let mut min_elem = f64::INFINITY;
    
    for &val in diff_matrix.iter() {
        if val != 0.0 {
            max_elem = max_elem.max(val.abs());
            min_elem = min_elem.min(val.abs());
        }
    }
    
    if max_elem / min_elem > 1e12 {
        return Err("矩阵条件数过大，可能导致特征值计算不稳定".into());
    }
    
    // 检查超时
    if start_time.elapsed() > remaining_timeout {
        return Err("特征值计算前超时".into());
    }
    
    // 4. 计算特征值（这是最耗时的步骤）
    let eigenvalue_start = Instant::now();
    let eigenvalues = diff_matrix.complex_eigenvalues();
    
    // 检查特征值计算是否超时（这个检查可能来不及，但至少记录）
    if eigenvalue_start.elapsed() > Duration::from_secs(10) {
        eprintln!("警告：特征值计算耗时过长");
    }
    
    // 5. 处理结果
    let mut real_eigenvalues: Vec<f64> = eigenvalues
        .iter()
        .map(|complex_val| {
            let norm = complex_val.norm();
            // 过滤掉异常值
            if norm.is_finite() && norm < 1e15 {
                norm
            } else {
                0.0
            }
        })
        .collect();
    
    real_eigenvalues.sort_unstable_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    
    Ok(real_eigenvalues)
}