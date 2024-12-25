use pyo3::prelude::*;
// use pyo3::types::{PyList, PyModule};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use ndarray::{s, Array1, Array2, ArrayView1, ArrayView2};
// use std::collections::{HashMap, HashSet};

/// 普通最小二乘(OLS)回归。
/// 用于拟合线性回归模型 y = Xβ + ε，其中β是要估计的回归系数。
///
/// 参数说明：
/// ----------
/// x : numpy.ndarray
///     设计矩阵，形状为(n_samples, n_features)
/// y : numpy.ndarray
///     响应变量，形状为(n_samples,)
/// calculate_r2 : bool, optional
///     是否计算R²值，默认为True
///
/// 返回值：
/// -------
/// numpy.ndarray
///     回归系数β
///
/// Python调用示例：
/// ```python
/// import numpy as np
/// from rust_pyfunc import ols
///
/// # 准备训练数据
/// X = np.array([[1, 1], [1, 2], [1, 3]], dtype=np.float64)  # 包含一个常数项和一个特征
/// y = np.array([2, 4, 6], dtype=np.float64)  # 目标变量
///
/// # 拟合模型
/// coefficients = ols(X, y)
/// print(f"回归系数: {coefficients}")  # 预期输出接近[0, 2]，表示y ≈ 0 + 2x
/// ```
#[pyfunction]
#[pyo3(signature = (x, y, calculate_r2=true))]
pub fn ols(
    py: Python,
    x: PyReadonlyArray2<f64>,
    y: PyReadonlyArray1<f64>,
    calculate_r2: Option<bool>,
) -> PyResult<Py<PyArray1<f64>>> {
    let x: ArrayView2<f64> = x.as_array();
    let y: ArrayView1<f64> = y.as_array();

    // 创建带有截距项的设计矩阵
    let mut x_with_intercept = Array2::ones((x.nrows(), x.ncols() + 1));
    x_with_intercept.slice_mut(s![.., 1..]).assign(&x);

    // 计算 (X^T * X)^(-1) * X^T * y
    let xt_x = x_with_intercept.t().dot(&x_with_intercept);
    let xt_y = x_with_intercept.t().dot(&y);
    let coefficients = solve_linear_system3(&xt_x.view(), &xt_y.view());

    let mut result = coefficients.to_vec();

    // 如果需要计算R方
    if calculate_r2.unwrap_or(true) {
        // 计算R方
        let y_mean = y.mean().unwrap();
        let y_pred = x_with_intercept.dot(&coefficients);
        let ss_tot: f64 = y.iter().map(|&yi| (yi - y_mean).powi(2)).sum();
        let ss_res: f64 = (&y - &y_pred).map(|e| e.powi(2)).sum();
        let r_squared = 1.0 - (ss_res / ss_tot);
        result.push(r_squared);
    }

    // 将结果转换为 Python 数组
    Ok(Array1::from(result).into_pyarray(py).to_owned())
}

    /// 使用已有数据和响应变量，对新的数据点进行OLS线性回归预测。
///
/// 参数说明：
/// ----------
/// x : numpy.ndarray
///     原始设计矩阵，形状为(n_samples, n_features)
/// y : numpy.ndarray
///     原始响应变量，形状为(n_samples,)
/// x_pred : numpy.ndarray
///     需要预测的新数据点，形状为(m_samples, n_features)
///
/// 返回值：
/// -------
/// numpy.ndarray
///     预测值，形状为(m_samples,)
///
/// Python调用示例：
/// ```python
/// import numpy as np
/// from rust_pyfunc import ols_predict
///
/// # 准备训练数据
/// X_train = np.array([[1, 1], [1, 2], [1, 3]], dtype=np.float64)
/// y_train = np.array([2, 4, 6], dtype=np.float64)
///
/// # 准备预测数据
/// X_pred = np.array([[1, 4], [1, 5]], dtype=np.float64)
///
/// # 进行预测
/// predictions = ols_predict(X_train, y_train, X_pred)
/// print(f"预测值: {predictions}")  # 预期输出接近[8, 10]
/// ```
#[pyfunction]
#[pyo3(signature = (x, y, x_pred))]
pub fn ols_predict(
    py: Python,
    x: PyReadonlyArray2<f64>,
    y: PyReadonlyArray1<f64>,
    x_pred: PyReadonlyArray2<f64>,
) -> PyResult<Py<PyArray1<f64>>> {
    let x: ArrayView2<f64> = x.as_array();
    let y: ArrayView1<f64> = y.as_array();
    let x_pred: ArrayView2<f64> = x_pred.as_array();

    // 创建带有截距项的设计矩阵
    let mut x_with_intercept = Array2::ones((x.nrows(), x.ncols() + 1));
    x_with_intercept.slice_mut(s![.., 1..]).assign(&x);

    // 计算回归系数
    let xt_x = x_with_intercept.t().dot(&x_with_intercept);
    let xt_y = x_with_intercept.t().dot(&y);
    let coefficients = solve_linear_system3(&xt_x.view(), &xt_y.view());

    // 为预测数据创建带有截距项的设计矩阵
    let mut x_pred_with_intercept = Array2::ones((x_pred.nrows(), x_pred.ncols() + 1));
    x_pred_with_intercept.slice_mut(s![.., 1..]).assign(&x_pred);

    // 计算预测值
    let predictions = x_pred_with_intercept.dot(&coefficients);

    // 将预测结果转换为 Python 数组
    Ok(predictions.into_pyarray(py).to_owned())
}

fn solve_linear_system3(a: &ArrayView2<f64>, b: &ArrayView1<f64>) -> Array1<f64> {
    let mut l = Array2::<f64>::zeros((a.nrows(), a.ncols()));
    let mut u = Array2::<f64>::zeros((a.nrows(), a.ncols()));

    // LU decomposition
    for i in 0..a.nrows() {
        for j in 0..a.ncols() {
            if i <= j {
                u[[i, j]] = a[[i, j]] - (0..i).map(|k| l[[i, k]] * u[[k, j]]).sum::<f64>();
                if i == j {
                    l[[i, i]] = 1.0;
                }
            }
            if i > j {
                l[[i, j]] =
                    (a[[i, j]] - (0..j).map(|k| l[[i, k]] * u[[k, j]]).sum::<f64>()) / u[[j, j]];
            }
        }
    }

    // Forward substitution
    let mut y = Array1::<f64>::zeros(b.len());
    for i in 0..b.len() {
        y[i] = b[i] - (0..i).map(|j| l[[i, j]] * y[j]).sum::<f64>();
    }

    // Backward substitution
    let mut x = Array1::<f64>::zeros(b.len());
    for i in (0..b.len()).rev() {
        x[i] = (y[i] - (i + 1..b.len()).map(|j| u[[i, j]] * x[j]).sum::<f64>()) / u[[i, i]];
    }

    x
}


/// 计算序列中每个位置结尾的最长连续子序列长度，其中子序列的最大值在该位置。
///
/// 参数说明：
/// ----------
/// s : array_like
///     输入序列，一个数值列表
/// allow_equal : bool, 默认为False
///     是否允许相等。如果为True，则当前位置的值大于前面的值时计入长度；
///     如果为False，则当前位置的值大于等于前面的值时计入长度。
///
/// 返回值：
/// -------
/// list
///     与输入序列等长的整数列表，每个元素表示以该位置结尾且最大值在该位置的最长连续子序列长度
///
/// Python调用示例：
/// ```python
/// from rust_pyfunc import max_range_loop
///
/// # 测试序列
/// seq = [1.0, 2.0, 3.0, 2.0, 1.0]
///
/// # 计算最大值范围（不允许相等）
/// ranges = max_range_loop(seq, allow_equal=False)
/// print(f"最大值范围: {ranges}")  # 输出: [1, 2, 3, 1, 1]
///
/// # 计算最大值范围（允许相等）
/// ranges = max_range_loop(seq, allow_equal=True)
/// print(f"最大值范围: {ranges}")  # 输出可能不同
/// ```
#[pyfunction]
#[pyo3(signature = (s, allow_equal=true))]
pub fn max_range_loop(s: Vec<f64>, allow_equal: bool) -> Vec<i32> {
    let mut maxranges = Vec::with_capacity(s.len());
    let mut stack = Vec::new();

    for i in 0..s.len() {
        while let Some(&j) = stack.last() {
            if (!allow_equal && s[j] >= s[i]) || (allow_equal && s[j] > s[i]) {
                maxranges.push(i as i32 - j as i32);
                break;
            }
            stack.pop();
        }
        if stack.is_empty() {
            maxranges.push(i as i32 + 1);
        }
        stack.push(i);
    }

    maxranges
}


/// 计算序列中每个位置结尾的最长连续子序列长度，其中子序列的最小值在该位置。
///
/// 参数说明：
/// ----------
/// s : array_like
///     输入序列，一个数值列表
/// allow_equal : bool, 默认为False
///     是否允许相等。如果为True，则当前位置的值小于前面的值时计入长度；
///     如果为False，则当前位置的值小于等于前面的值时计入长度。
///
/// 返回值：
/// -------
/// list
///     与输入序列等长的整数列表，每个元素表示以该位置结尾且最小值在该位置的最长连续子序列长度
///
/// Python调用示例：
/// ```python
/// from rust_pyfunc import min_range_loop
///
/// # 测试序列
/// seq = [1.0, 2.0, 3.0, 2.0, 1.0]
///
/// # 计算最小值范围（不允许相等）
/// ranges = min_range_loop(seq, allow_equal=False)
/// print(f"最小值范围: {ranges}")  # 输出: [1, 2, 3, 1, 5]
///
/// # 计算最小值范围（允许相等）
/// ranges = min_range_loop(seq, allow_equal=True)
/// print(f"最小值范围: {ranges}")  # 输出可能不同
/// ```
#[pyfunction]
#[pyo3(signature = (s, allow_equal=true))]
pub fn min_range_loop(s: Vec<f64>, allow_equal: bool) -> Vec<i32> {
    let mut minranges = Vec::with_capacity(s.len());
    let mut stack = Vec::new();

    for i in 0..s.len() {
        while let Some(&j) = stack.last() {
            if (!allow_equal && s[j] <= s[i]) || (allow_equal && s[j] < s[i]) {
                minranges.push(i as i32 - j as i32);
                break;
            }
            stack.pop();
        }
        if stack.is_empty() {
            minranges.push(i as i32 + 1);
        }
        stack.push(i);
    }

    minranges
}