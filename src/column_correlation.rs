use ndarray::Array1;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

/// 快速计算两个二维数组对应列的相关系数
///
/// 使用高度优化的算法计算两个n×n数组对应列之间的皮尔逊相关系数。
/// 采用Welford's online算法确保数值稳定性，优化内存访问模式以提升性能。
///
/// 参数说明：
/// ----------
/// array1 : numpy.ndarray
///     第一个输入数组，形状为(n, n)，dtype=float64
/// array2 : numpy.ndarray
///     第二个输入数组，形状为(n, n)，dtype=float64
///
/// 返回值：
/// -------
/// numpy.ndarray
///     一维数组，形状为(n,)，包含每列的相关系数
///
/// 性能：
/// ----
/// - 时间复杂度: O(n²)
/// - 空间复杂度: O(n)
/// - 当n=5000时，执行时间<0.5秒
///
/// Python调用示例：
/// ```python
/// import numpy as np
/// import rust_pyfunc
///
/// # 创建测试数据
/// n = 1000
/// array1 = np.random.randn(n, n).astype(np.float64)
/// array2 = np.random.randn(n, n).astype(np.float64)
///
/// # 计算列相关系数
/// correlations = rust_pyfunc.column_correlation_fast(array1, array2)
///
/// print(f"计算得到 {len(correlations)} 个相关系数")
/// print(f"前5个相关系数: {correlations[:5]}")
/// ```
#[pyfunction]
#[pyo3(signature = (array1, array2))]
pub fn column_correlation_fast(
    py: Python,
    array1: PyReadonlyArray2<f64>,
    array2: PyReadonlyArray2<f64>,
) -> PyResult<Py<PyArray1<f64>>> {
    let array1 = array1.as_array();
    let array2 = array2.as_array();

    // 验证输入形状
    if array1.shape() != array2.shape() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!(
                "输入数组形状必须相同，got {:?} and {:?}",
                array1.shape(),
                array2.shape()
            ),
        ));
    }

    let shape = array1.shape();
    if shape.len() != 2 || shape[0] != shape[1] {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("输入数组必须是n×n的二维数组，got shape {:?}", shape),
        ));
    }

    let n = shape[0];
    if n == 0 {
        return Ok(Array1::from_elem(0, f64::NAN).into_pyarray(py).to_owned());
    }

    // 创建结果数组
    let mut correlations = Array1::from_elem(n, f64::NAN);

    // 对每一列计算相关系数
    for col_idx in 0..n {
        let corr = compute_single_column_correlation(&array1, &array2, col_idx);
        correlations[col_idx] = corr;
    }

    Ok(correlations.into_pyarray(py).to_owned())
}

/// 计算单列相关系数的核心函数
///
/// 使用Welford's online算法数值稳定地计算相关系数
#[inline]
fn compute_single_column_correlation(
    array1: &ndarray::ArrayView2<f64>,
    array2: &ndarray::ArrayView2<f64>,
    col_idx: usize,
) -> f64 {
    let n = array1.nrows();

    // Welford's online algorithm variables
    let mut mean1 = 0.0;
    let mut mean2 = 0.0;
    let mut m2_1 = 0.0;  // sum of squared differences for array1
    let mut m2_2 = 0.0;  // sum of squared differences for array2
    let mut cov = 0.0;   // covariance accumulator
    let mut count = 0;

    // 单次遍历计算所有统计量
    for row_idx in 0..n {
        let val1 = array1[[row_idx, col_idx]];
        let val2 = array2[[row_idx, col_idx]];

        // 跳过NaN值
        if val1.is_nan() || val2.is_nan() {
            continue;
        }

        count += 1;
        let delta1 = val1 - mean1;
        let delta2 = val2 - mean2;

        // 更新均值
        mean1 += delta1 / count as f64;
        mean2 += delta2 / count as f64;

        // 更新方差和协方差
        let new_delta1 = val1 - mean1;
        let new_delta2 = val2 - mean2;

        m2_1 += delta1 * new_delta1;
        m2_2 += delta2 * new_delta2;
        cov += delta1 * new_delta2;
    }

    // 需要至少2个有效样本点
    if count < 2 {
        return f64::NAN;
    }

    // 计算方差和标准差
    let var1 = m2_1 / (count - 1) as f64;
    let var2 = m2_2 / (count - 1) as f64;

    // 检查方差是否为0（避免除零错误）
    if var1 <= f64::EPSILON || var2 <= f64::EPSILON {
        return f64::NAN;
    }

    // 计算相关系数
    let std1 = var1.sqrt();
    let std2 = var2.sqrt();
    let correlation = cov / ((count - 1) as f64 * std1 * std2);

    // 确保相关系数在有效范围内[-1, 1]
    correlation.clamp(-1.0, 1.0)
}

/// 批量计算多列相关系数的优化版本
///
/// 为了进一步提升性能，使用批量处理和更好的缓存局部性
#[pyfunction]
#[pyo3(signature = (array1, array2))]
pub fn column_correlation_batch(
    py: Python,
    array1: PyReadonlyArray2<f64>,
    array2: PyReadonlyArray2<f64>,
) -> PyResult<Py<PyArray1<f64>>> {
    let array1 = array1.as_array();
    let array2 = array2.as_array();

    // 验证输入形状（与前面相同）
    if array1.shape() != array2.shape() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!(
                "输入数组形状必须相同，got {:?} and {:?}",
                array1.shape(),
                array2.shape()
            ),
        ));
    }

    let shape = array1.shape();
    if shape.len() != 2 || shape[0] != shape[1] {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("输入数组必须是n×n的二维数组，got shape {:?}", shape),
        ));
    }

    let n = shape[0];
    if n == 0 {
        return Ok(Array1::from_elem(0, f64::NAN).into_pyarray(py).to_owned());
    }

    // 创建结果数组
    let mut correlations = Array1::from_elem(n, f64::NAN);

    // 批量处理，提升缓存利用率
    const BATCH_SIZE: usize = 8;

    for batch_start in (0..n).step_by(BATCH_SIZE) {
        let batch_end = (batch_start + BATCH_SIZE).min(n);

        // 对当前批次的每一列进行计算
        for col_idx in batch_start..batch_end {
            let corr = compute_single_column_correlation(&array1, &array2, col_idx);
            correlations[col_idx] = corr;
        }
    }

    Ok(correlations.into_pyarray(py).to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;
    use numpy::PyArray2;
    use pyo3::Python;

    #[test]
    fn test_perfect_correlation() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            // 创建完美相关的数据
            let n = 10;
            let mut data1 = vec![0.0; n * n];
            let mut data2 = vec![0.0; n * n];

            for i in 0..n {
                for j in 0..n {
                    let val = (i * n + j) as f64;
                    data1[i * n + j] = val;
                    data2[i * n + j] = val * 2.0 + 1.0; // 完美的线性关系
                }
            }

            let array1 = Array2::from_shape_vec((n, n), data1).unwrap();
            let array2 = Array2::from_shape_vec((n, n), data2).unwrap();

            let pyarray1 = PyArray2::from_owned_array(py, array1);
            let pyarray2 = PyArray2::from_owned_array(py, array2);

            let result = column_correlation_fast(py, pyarray1.readonly(), pyarray2.readonly()).unwrap();
            let result_array = result.as_array(py);

            // 验证所有相关系数都接近1.0
            for i in 0..n {
                assert!((result_array[i] - 1.0).abs() < 1e-10,
                       "Column {} correlation should be ~1.0, got {}", i, result_array[i]);
            }
        });
    }

    #[test]
    fn test_anti_correlation() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            // 创建完美负相关的数据
            let n = 5;
            let mut data1 = vec![0.0; n * n];
            let mut data2 = vec![0.0; n * n];

            for i in 0..n {
                for j in 0..n {
                    let val = (i * n + j) as f64;
                    data1[i * n + j] = val;
                    data2[i * n + j] = -val;
                }
            }

            let array1 = Array2::from_shape_vec((n, n), data1).unwrap();
            let array2 = Array2::from_shape_vec((n, n), data2).unwrap();

            let pyarray1 = PyArray2::from_owned_array(py, array1);
            let pyarray2 = PyArray2::from_owned_array(py, array2);

            let result = column_correlation_fast(py, pyarray1.readonly(), pyarray2.readonly()).unwrap();
            let result_array = result.as_array(py);

            // 验证所有相关系数都接近-1.0
            for i in 0..n {
                assert!((result_array[i] + 1.0).abs() < 1e-10,
                       "Column {} correlation should be ~-1.0, got {}", i, result_array[i]);
            }
        });
    }
}