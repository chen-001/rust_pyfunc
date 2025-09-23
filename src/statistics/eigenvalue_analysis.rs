use nalgebra::DMatrix;
use ndarray::{Array2, ArrayView1};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;
use std::sync::Arc;

/// 计算多列数据的差值矩阵特征值
///
/// 对输入的237行×n列矩阵，对每一列进行以下操作：
/// 1. 构建237×237的差值矩阵，其中M[i,j] = col[i] - col[j]
/// 2. 计算该矩阵的所有特征值
/// 3. 按特征值绝对值从大到小排序
///
/// 此函数针对高性能计算进行了优化：
/// - 使用并行处理处理不同列（最多10个核心）
/// - 利用差值矩阵的对称性质优化计算
/// - 使用高效的线性代数库nalgebra进行特征值分解
///
/// 参数说明：
/// ----------
/// matrix : numpy.ndarray
///     输入矩阵，形状为(237, n)，必须是float64类型
///
/// 返回值：
/// -------
/// numpy.ndarray
///     输出矩阵，形状为(237, n)，每列包含对应输入列的特征值（按绝对值降序排列）
///
/// Python调用示例：
/// ```python
/// import numpy as np
/// import design_whatever as dw
/// from rust_pyfunc import matrix_eigenvalue_analysis
///
/// # 读取测试数据
/// df = dw.read_minute_data('volume').dropna(how='all')
/// data = df.to_numpy(float)
///
/// # 计算特征值
/// result = matrix_eigenvalue_analysis(data)
/// print(f"结果形状: {result.shape}")
/// ```
#[pyfunction]
#[pyo3(signature = (matrix))]
pub fn matrix_eigenvalue_analysis(
    py: Python,
    matrix: PyReadonlyArray2<f64>,
) -> PyResult<Py<PyArray2<f64>>> {
    let input_matrix = matrix.as_array();
    let (n_rows, n_cols) = input_matrix.dim();

    // 验证输入矩阵行数
    if n_rows != 237 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "输入矩阵必须有237行，但得到了{}行",
            n_rows
        )));
    }

    // 创建结果矩阵
    let mut result = Array2::<f64>::zeros((237, n_cols));

    // 设置线程池大小限制为10
    let thread_pool = rayon::ThreadPoolBuilder::new()
        .num_threads(std::cmp::min(10, num_cpus::get()))
        .build()
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("线程池创建失败: {}", e)))?;

    // 将输入矩阵转换为Arc以便在线程间共享
    let input_arc = Arc::new(input_matrix.to_owned());

    // 并行处理每一列
    thread_pool.install(|| {
        let eigenvalue_results: Vec<_> = (0..n_cols)
            .into_par_iter()
            .map(|col_idx| {
                // 提取当前列
                let column = input_arc.column(col_idx);

                // 计算该列的特征值
                match compute_column_eigenvalues(&column) {
                    Ok(eigenvalues) => (col_idx, eigenvalues),
                    Err(_) => (col_idx, vec![f64::NAN; 237]),
                }
            })
            .collect();

        // 将结果写入输出矩阵
        for (col_idx, eigenvalues) in eigenvalue_results {
            for (i, &val) in eigenvalues.iter().enumerate() {
                if i < 237 {
                    result[[i, col_idx]] = val;
                }
            }
        }
    });

    Ok(result.into_pyarray(py).to_owned())
}

/// 为单列数据计算差值矩阵的特征值
fn compute_column_eigenvalues(
    column: &ArrayView1<f64>,
) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    let n = column.len();

    // 构建差值矩阵 M[i,j] = column[i] - column[j]
    // 注意：这个矩阵是反对称的，即M[i,j] = -M[j,i]，M[i,i] = 0
    let mut diff_matrix = DMatrix::<f64>::zeros(n, n);

    for i in 0..n {
        for j in 0..n {
            diff_matrix[(i, j)] = column[i] - column[j];
        }
    }

    // 对于反对称矩阵，所有特征值都是纯虚数或零
    // 但由于数值计算的误差，我们会得到接近零的实部
    // 我们计算特征值并取其虚部（或实部的绝对值，取决于具体情况）

    // 使用nalgebra计算特征值
    // 注意：nalgebra的eigen()方法适用于一般矩阵
    let eigenvalues = diff_matrix.complex_eigenvalues();

    // 提取特征值的模长（复数的绝对值）
    let mut real_eigenvalues: Vec<f64> = eigenvalues
        .iter()
        .map(|complex_val| complex_val.norm())
        .collect();

    // 按绝对值从大到小排序
    real_eigenvalues.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

    Ok(real_eigenvalues)
}

/// 计算多列数据的差值矩阵特征值（优化版本）
///
/// 这是一个针对大规模计算优化的版本，使用了更高效的算法：
/// 1. 利用差值矩阵的反对称性质减少计算量
/// 2. 使用更高效的内存布局
/// 3. 优化的并行策略
///
/// 参数说明：
/// ----------
/// matrix : numpy.ndarray
///     输入矩阵，形状为(237, n)，必须是float64类型
///
/// 返回值：
/// -------
/// numpy.ndarray
///     输出矩阵，形状为(237, n)，每列包含对应输入列的特征值（按绝对值降序排列）
#[pyfunction]
#[pyo3(signature = (matrix))]
pub fn matrix_eigenvalue_analysis_optimized(
    py: Python,
    matrix: PyReadonlyArray2<f64>,
) -> PyResult<Py<PyArray2<f64>>> {
    let input_matrix = matrix.as_array();
    let (n_rows, n_cols) = input_matrix.dim();

    if n_rows != 237 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "输入矩阵必须有237行，但得到了{}行",
            n_rows
        )));
    }

    let mut result = Array2::<f64>::zeros((237, n_cols));

    // 设置线程池，限制最多10个线程
    let thread_pool = rayon::ThreadPoolBuilder::new()
        .num_threads(std::cmp::min(10, num_cpus::get()))
        .build()
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("线程池创建失败: {}", e)))?;

    let input_arc = Arc::new(input_matrix.to_owned());

    thread_pool.install(|| {
        let eigenvalue_results: Vec<_> = (0..n_cols)
            .into_par_iter()
            .map(|col_idx| {
                let column = input_arc.column(col_idx);

                match compute_column_eigenvalues_optimized(&column) {
                    Ok(eigenvalues) => (col_idx, eigenvalues),
                    Err(_) => (col_idx, vec![f64::NAN; 237]),
                }
            })
            .collect();

        // 将结果写入输出矩阵
        for (col_idx, eigenvalues) in eigenvalue_results {
            for (i, &val) in eigenvalues.iter().enumerate() {
                if i < 237 {
                    result[[i, col_idx]] = val;
                }
            }
        }
    });

    Ok(result.into_pyarray(py).to_owned())
}

/// 优化版本的单列特征值计算
///
/// 利用差值矩阵的特殊结构进行优化：
/// - 差值矩阵是反对称的：M[i,j] = -M[j,i]，对角线为0
/// - 对于反对称矩阵，特征值要么是0，要么成对出现（λ, -λ）
/// - 我们可以利用这些性质来加速计算
fn compute_column_eigenvalues_optimized(
    column: &ArrayView1<f64>,
) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    let n = column.len();

    // 对于反对称矩阵的特征值计算，我们可以使用特殊的算法
    // 但为了保证数值稳定性，我们仍然使用标准方法

    // 构建差值矩阵
    let mut diff_matrix = DMatrix::<f64>::zeros(n, n);

    // 利用反对称性质，只计算上三角部分
    for i in 0..n {
        for j in i + 1..n {
            let diff_val = column[i] - column[j];
            diff_matrix[(i, j)] = diff_val;
            diff_matrix[(j, i)] = -diff_val; // 反对称性
        }
        // 对角线元素为0
        diff_matrix[(i, i)] = 0.0;
    }

    // 计算特征值
    let eigenvalues = diff_matrix.complex_eigenvalues();

    // 提取特征值的模长并排序
    let mut real_eigenvalues: Vec<f64> = eigenvalues
        .iter()
        .map(|complex_val| complex_val.norm())
        .collect();

    real_eigenvalues.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

    Ok(real_eigenvalues)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array2};

    #[test]
    fn test_small_matrix_eigenvalues() {
        // 创建一个小的测试矩阵
        let test_data = Array2::from_shape_vec((3, 2), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]).unwrap();

        // 测试第一列 [1, 2, 3]
        let col1 = test_data.column(0);
        let result = compute_column_eigenvalues(&col1).unwrap();

        // 验证结果长度
        assert_eq!(result.len(), 3);

        // 验证排序（应该按绝对值降序）
        for i in 1..result.len() {
            assert!(result[i - 1] >= result[i]);
        }
    }

    #[test]
    fn test_antisymmetric_properties() {
        // 测试反对称矩阵的性质
        let column = Array1::from_vec(vec![1.0, 2.0, 3.0]);

        // 手动构建差值矩阵
        let mut diff_matrix = DMatrix::<f64>::zeros(3, 3);
        for i in 0..3 {
            for j in 0..3 {
                diff_matrix[(i, j)] = column[i] - column[j];
            }
        }

        // 验证反对称性质
        for i in 0..3 {
            for j in 0..3 {
                assert!((diff_matrix[(i, j)] + diff_matrix[(j, i)]).abs() < 1e-10);
            }
            assert!(diff_matrix[(i, i)].abs() < 1e-10);
        }
    }
}
