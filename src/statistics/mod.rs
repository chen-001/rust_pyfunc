use pyo3::prelude::*;
// use pyo3::types::{PyList, PyModule};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use ndarray::{s, Array1, Array2, ArrayView1, ArrayView2};
use rayon::prelude::*;
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

/// 计算普通最小二乘(OLS)回归的残差序列。
/// 残差表示实际观测值与模型预测值之间的差异: ε = y - Xβ。
///
/// 参数说明：
/// ----------
/// x : numpy.ndarray
///     设计矩阵，形状为(n_samples, n_features)
/// y : numpy.ndarray
///     响应变量，形状为(n_samples,)
///
/// 返回值：
/// -------
/// numpy.ndarray
///     残差序列，形状为(n_samples,)
///
/// Python调用示例：
/// ```python
/// import numpy as np
/// from rust_pyfunc import ols_residuals
///
/// # 准备训练数据
/// X = np.array([[1, 1], [1, 2], [1, 3]], dtype=np.float64)  # 包含一个常数项和一个特征
/// y = np.array([2, 4, 6], dtype=np.float64)  # 目标变量
///
/// # 计算残差
/// residuals = ols_residuals(X, y)
/// print(f"残差: {residuals}")  # 如果模型拟合良好，残差应该接近于零
/// ```
#[pyfunction]
#[pyo3(signature = (x, y))]
pub fn ols_residuals(
    py: Python,
    x: PyReadonlyArray2<f64>,
    y: PyReadonlyArray1<f64>,
) -> PyResult<Py<PyArray1<f64>>> {
    let x: ArrayView2<f64> = x.as_array();
    let y: ArrayView1<f64> = y.as_array();

    // 新增：过滤含 NaN 的样本
    let n_samples = x.nrows();
    let n_features = x.ncols();
    let mut valid_indices = Vec::new();
    for i in 0..n_samples {
        if y[i].is_nan() {
            continue;
        }
        let mut row_valid = true;
        for j in 0..n_features {
            if x[[i, j]].is_nan() {
                row_valid = false;
                break;
            }
        }
        if row_valid {
            valid_indices.push(i);
        }
    }
    if valid_indices.is_empty() {
        let full_residuals = Array1::<f64>::from_elem(n_samples, f64::NAN);
        return Ok(full_residuals.into_pyarray(py).to_owned());
    }
    let m = valid_indices.len();
    let mut x_sub = Array2::<f64>::zeros((m, n_features));
    let mut y_sub = Array1::<f64>::zeros(m);
    for (row, &orig_idx) in valid_indices.iter().enumerate() {
        x_sub.row_mut(row).assign(&x.row(orig_idx));
        y_sub[row] = y[orig_idx];
    }
    // 创建带截距项的设计矩阵
    let mut x_sub_with_intercept = Array2::ones((m, n_features + 1));
    x_sub_with_intercept.slice_mut(s![.., 1..]).assign(&x_sub);
    // 计算回归系数
    let xt_x = x_sub_with_intercept.t().dot(&x_sub_with_intercept);
    let xt_y = x_sub_with_intercept.t().dot(&y_sub);
    let coefficients = solve_linear_system3(&xt_x.view(), &xt_y.view());
    // 计算预测值和残差
    let y_pred_sub = x_sub_with_intercept.dot(&coefficients);
    let residuals_sub = &y_sub - &y_pred_sub;
    // 填充到完整残差向量
    let mut full_residuals = Array1::<f64>::from_elem(n_samples, f64::NAN);
    for (row, &orig_idx) in valid_indices.iter().enumerate() {
        full_residuals[orig_idx] = residuals_sub[row];
    }
    Ok(full_residuals.into_pyarray(py).to_owned())
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

/// 计算价格序列的滚动波动率。
///
/// 对于位置i，从数据范围[i-lookback+1, i]中每隔interval个点取样，
/// 然后计算相邻样本之间的对数收益率（后面的价格除以前面的价格的对数），
/// 最后计算这些收益率的标准差作为波动率。
///
/// 参数说明：
/// ----------
/// prices : array_like
///     价格序列
/// lookback : usize
///     表示回溯的数据范围长度，对于位置i，考虑[i-lookback+1, i]范围内的数据
/// interval : usize
///     取样间隔，每隔interval个点取一个样本
/// min_periods : usize, 可选
///     计算波动率所需的最小样本数，默认为2
///
/// 返回值：
/// -------
/// array_like
///     与输入序列等长的波动率序列
///
/// Python调用示例：
/// ```python
/// import numpy as np
/// from rust_pyfunc import rolling_volatility
///
/// # 创建价格序列
/// prices = np.array([a1, a2, a3, a4, a5, a6, a7, a8, a9], dtype=np.float64)
///
/// # 计算滚动波动率，lookback=5, interval=1
/// # 结果应该是[nan, nan, nan, nan, std(log(a3/a1), log(a5/a3)), ...]
/// vol = rolling_volatility(prices, 5, 1)
///
/// # 计算滚动波动率，lookback=7, interval=2
/// # 结果应该是[nan, nan, nan, nan, nan, nan, std(log(a4/a1), log(a7/a4)), ...]
/// vol = rolling_volatility(prices, 7, 2)
/// ```
#[pyfunction]
#[pyo3(signature = (prices, lookback, interval, min_periods=2))]
pub fn rolling_volatility(
    py: Python,
    prices: PyReadonlyArray1<f64>,
    lookback: usize,
    interval: usize,
    min_periods: Option<usize>,
) -> PyResult<Py<PyArray1<f64>>> {
    let prices = prices.as_array();
    let n = prices.len();
    let min_periods = min_periods.unwrap_or(2);
    
    // 创建结果数组，初始化为NaN
    let mut result = Array1::from_elem(n, f64::NAN);
    
    // 对每个位置计算波动率
    for i in 0..n {
        // 如果历史数据不足，直接跳过
        if i < lookback - 1 {
            continue;
        }
        
        // 确定数据范围的起始位置
        let start_idx = i - (lookback - 1);
        
        // 在范围[start_idx, i]内按interval间隔收集样本
        let mut samples = Vec::new();
        let mut pos = start_idx;
        while pos <= i {
            samples.push(prices[pos]);
            pos += interval;
            if pos > i {
                break;
            }
        }
        
        // 确保有足够的样本点
        if samples.len() < min_periods {
            continue;
        }
        
        // 计算相邻样本之间的对数收益率
        let mut returns = Vec::new();
        for k in 0..samples.len() - 1 {
            // 后面的价格除以前面的价格的对数
            let ret = (samples[k+1] / samples[k]).ln();
            returns.push(ret);
        }
        
        // 有足够的收益率样本才计算标准差
        if returns.len() >= min_periods - 1 {
            // 计算均值
            let mean = returns.iter().sum::<f64>() / returns.len() as f64;
            
            // 计算方差
            let variance = returns.iter()
                .map(|&r| (r - mean).powi(2))
                .sum::<f64>() / returns.len() as f64;
            
            // 计算标准差（波动率）
            if variance > 0.0 {
                result[i] = variance.sqrt();
            } else {
                // 如果方差为0，波动率也为0
                result[i] = 0.0;
            }
        }
    }
    
    Ok(result.into_pyarray(py).to_owned())
}

/// 计算价格序列的滚动变异系数(CV)。
///
/// 对于位置i，从数据范围[i-lookback+1, i]中每隔interval个点取样，
/// 然后计算相邻样本之间的对数收益率（后面的价格除以前面的价格的对数），
/// 最后计算这些收益率的变异系数（标准差除以均值）。
///
/// 参数说明：
/// ----------
/// values : array_like
///     数值序列
/// lookback : usize
///     表示回溯的数据范围长度，对于位置i，考虑[i-lookback+1, i]范围内的数据
/// interval : usize
///     取样间隔，每隔interval个点取一个样本
/// min_periods : usize, 可选
///     计算变异系数所需的最小样本数，默认为2
///
/// 返回值：
/// -------
/// array_like
///     与输入序列等长的变异系数序列
///
/// Python调用示例：
/// ```python
/// import numpy as np
/// from rust_pyfunc import rolling_cv
///
/// # 创建数值序列
/// values = np.array([a1, a2, a3, a4, a5, a6, a7, a8, a9], dtype=np.float64)
///
/// # 计算滚动变异系数，lookback=5, interval=1
/// cv = rolling_cv(values, 5, 1)
///
/// # 计算滚动变异系数，lookback=7, interval=2
/// cv = rolling_cv(values, 7, 2)
/// ```
#[pyfunction]
#[pyo3(signature = (values, lookback, interval, min_periods=2))]
pub fn rolling_cv(
    py: Python,
    values: PyReadonlyArray1<f64>,
    lookback: usize,
    interval: usize,
    min_periods: Option<usize>,
) -> PyResult<Py<PyArray1<f64>>> {
    let values = values.as_array();
    let n = values.len();
    let min_periods = min_periods.unwrap_or(2);
    
    // 创建结果数组，初始化为NaN
    let mut result = Array1::from_elem(n, f64::NAN);
    
    // 对每个位置计算变异系数
    for i in 0..n {
        // 如果历史数据不足，直接跳过
        if i < lookback - 1 {
            continue;
        }
        
        // 确定数据范围的起始位置
        let start_idx = i - (lookback - 1);
        
        // 在范围[start_idx, i]内按interval间隔收集样本
        let mut samples = Vec::new();
        let mut pos = start_idx;
        while pos <= i {
            samples.push(values[pos]);
            pos += interval;
            if pos > i {
                break;
            }
        }
        
        // 确保有足够的样本点
        if samples.len() < min_periods {
            continue;
        }
        
        // 计算相邻样本之间的对数收益率
        let mut returns = Vec::new();
        for k in 0..samples.len() - 1 {
            // 后面的价格除以前面的价格的对数
            let ret = (samples[k+1] / samples[k]).ln();
            returns.push(ret);
        }
        
        // 有足够的收益率样本才计算变异系数
        if returns.len() >= min_periods - 1 {
            // 计算收益率均值
            let mean = returns.iter().sum::<f64>() / returns.len() as f64;
            
            // 避免除以零的情况
            if mean.abs() < f64::EPSILON {
                continue;
            }
            
            // 计算收益率的方差
            let variance = returns.iter()
                .map(|&r| (r - mean).powi(2))
                .sum::<f64>() / returns.len() as f64;
            
            if variance > 0.0 {
                let std_dev = variance.sqrt();
                // 计算变异系数 (std/mean)
                result[i] = std_dev / mean.abs();  // 使用绝对值避免负均值导致的符号问题
            } else {
                // 如果方差为0，变异系数也为0
                result[i] = 0.0;
            }
        }
    }
    
    Ok(result.into_pyarray(py).to_owned())
}

/// 计算价格序列的滚动四分位变异系数(QCV)。
///
/// 对于位置i，从数据范围[i-lookback+1, i]中每隔interval个点取样，
/// 然后计算相邻样本之间的对数收益率（后面的价格除以前面的价格的对数），
/// 最后计算这些收益率的四分位变异系数（四分位间距除以中位数的绝对值）。
/// 这种方法对异常值和均值接近零的情况更加稳健。
///
/// 参数说明：
/// ----------
/// values : array_like
///     数值序列
/// lookback : usize
///     表示回溯的数据范围长度，对于位置i，考虑[i-lookback+1, i]范围内的数据
/// interval : usize
///     取样间隔，每隔interval个点取一个样本
/// min_periods : usize, 可选
///     计算变异系数所需的最小样本数，默认为3（需要至少3个点才能计算有意义的IQR）
///
/// 返回值：
/// -------
/// array_like
///     与输入序列等长的四分位变异系数序列
///
/// Python调用示例：
/// ```python
/// import numpy as np
/// from rust_pyfunc import rolling_qcv
///
/// # 创建数值序列
/// values = np.array([a1, a2, a3, a4, a5, a6, a7, a8, a9], dtype=np.float64)
///
/// # 计算滚动四分位变异系数，lookback=5, interval=1
/// qcv = rolling_qcv(values, 5, 1)
/// ```
#[pyfunction]
#[pyo3(signature = (values, lookback, interval, min_periods=3))]
pub fn rolling_qcv(
    py: Python,
    values: PyReadonlyArray1<f64>,
    lookback: usize,
    interval: usize,
    min_periods: Option<usize>,
) -> PyResult<Py<PyArray1<f64>>> {
    let values = values.as_array();
    let n = values.len();
    let min_periods = min_periods.unwrap_or(3);
    
    // 创建结果数组，初始化为NaN
    let mut result = Array1::from_elem(n, f64::NAN);
    
    // 对每个位置计算四分位变异系数
    for i in 0..n {
        // 如果历史数据不足，直接跳过
        if i < lookback - 1 {
            continue;
        }
        
        // 确定数据范围的起始位置
        let start_idx = i - (lookback - 1);
        
        // 在范围[start_idx, i]内按interval间隔收集样本
        let mut samples = Vec::new();
        let mut pos = start_idx;
        while pos <= i {
            samples.push(values[pos]);
            pos += interval;
            if pos > i {
                break;
            }
        }
        
        // 确保有足够的样本点
        if samples.len() < min_periods {
            continue;
        }
        
        // 计算相邻样本之间的对数收益率
        let mut returns = Vec::new();
        for k in 0..samples.len() - 1 {
            // 后面的价格除以前面的价格的对数
            let ret = (samples[k+1] / samples[k]).ln();
            returns.push(ret);
        }
        
        // 有足够的收益率样本才计算四分位变异系数
        if returns.len() >= min_periods - 1 {
            // 对收益率进行排序（用于计算中位数和四分位数）
            returns.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            
            // 计算中位数
            let median = if returns.len() % 2 == 0 {
                (returns[returns.len() / 2 - 1] + returns[returns.len() / 2]) / 2.0
            } else {
                returns[returns.len() / 2]
            };
            
            // 如果中位数接近0，跳过计算
            if median.abs() < f64::EPSILON {
                continue;
            }
            
            // 计算第一四分位数 (Q1)
            let q1_pos = returns.len() / 4;
            let q1 = if returns.len() % 4 == 0 {
                (returns[q1_pos - 1] + returns[q1_pos]) / 2.0
            } else {
                returns[q1_pos]
            };
            
            // 计算第三四分位数 (Q3)
            let q3_pos = returns.len() * 3 / 4;
            let q3 = if returns.len() % 4 == 0 {
                (returns[q3_pos - 1] + returns[q3_pos]) / 2.0
            } else {
                returns[q3_pos]
            };
            
            // 计算四分位间距 (IQR)
            let iqr = q3 - q1;
            
            // 计算四分位变异系数 (IQR/|中位数|)
            if iqr >= 0.0 {
                result[i] = iqr / median.abs();
            } else {
                // 如果IQR为负（不应该发生，但以防万一），跳过计算
                continue;
            }
        }
    }
    
    Ok(result.into_pyarray(py).to_owned())
}

/// 计算两个数据框对应列的相关系数。
///
/// 这个函数类似于pandas中的df.corrwith(df1)，计算两个数据框中对应列之间的皮尔逊相关系数。
/// 相关系数范围为[-1, 1]，其中：
/// - 1表示完全正相关
/// - -1表示完全负相关
/// - 0表示无相关性
///
/// 参数说明：
/// ----------
/// df1 : numpy.ndarray
///     第一个数据框，形状为(n_rows, n_cols)，必须是float64类型
/// df2 : numpy.ndarray
///     第二个数据框，形状为(n_rows, m_cols)，必须是float64类型
/// axis : int, 可选
///     计算相关性的轴，默认为0（按列计算）。目前只支持按列计算。
///
/// 返回值：
/// -------
/// numpy.ndarray
///     一维数组，长度为min(n_cols, m_cols)，包含对应列的相关系数
///
/// Python调用示例：
/// ```python
/// import numpy as np
/// import pandas as pd
/// from rust_pyfunc import dataframe_corrwith
///
/// # 创建两个数据框
/// df1 = pd.DataFrame({
///     'A': [1.0, 2.0, 3.0, 4.0, 5.0],
///     'B': [5.0, 4.0, 3.0, 2.0, 1.0],
///     'C': [2.0, 4.0, 6.0, 8.0, 10.0]
/// })
/// df2 = pd.DataFrame({
///     'A': [1.1, 2.2, 2.9, 4.1, 5.2],
///     'B': [5.2, 4.1, 2.9, 2.1, 0.9],
///     'D': [1.0, 2.0, 3.0, 4.0, 5.0]
/// })
///
/// # 计算相关系数
/// corr = dataframe_corrwith(df1.values, df2.values)
/// # 转换为Series以获得与pandas相同的输出格式
/// result = pd.Series(corr, index=['A', 'B', 'C'])
/// # 只保留有效对应列（A和B）
/// result = result.iloc[:2]
/// ```
#[pyfunction]
#[pyo3(signature = (df1, df2, axis=0, drop_na=true))]
pub fn dataframe_corrwith(
    py: Python,
    df1: PyReadonlyArray2<f64>,
    df2: PyReadonlyArray2<f64>,
    axis: Option<i32>,
    drop_na: Option<bool>,
) -> PyResult<Py<PyArray1<f64>>> {
    let df1 = df1.as_array();
    let df2 = df2.as_array();
    
    // 目前只支持按列计算相关系数（axis=0）
    let _axis = axis.unwrap_or(0);
    if _axis != 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "当前仅支持按列计算相关系数(axis=0)",
        ));
    }
    
    let drop_na = drop_na.unwrap_or(true);
    
    // 判断行数是否相同
    if df1.nrows() != df2.nrows() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "两个数据框的行数必须相同",
        ));
    }
    
    // 获取列数并确定结果数组大小
    let n_cols = std::cmp::min(df1.ncols(), df2.ncols());
    
    // 创建结果数组
    let result = if n_cols > 0 {
        // 使用并行迭代计算相关系数
        let mut corrs: Vec<f64> = vec![f64::NAN; n_cols];
        
        // 将计算分块并行处理
        corrs.par_iter_mut().enumerate().for_each(|(i, corr)| {
            if i < n_cols {
                // 提取对应列的数据
                let col1 = df1.column(i);
                let col2 = df2.column(i);
                
                // 计算相关系数
                *corr = calculate_correlation_optimized(&col1, &col2, drop_na);
            }
        });
        
        Array1::from(corrs)
    } else {
        Array1::from_elem(n_cols, f64::NAN)
    };
    
    Ok(result.into_pyarray(py).to_owned())
}


// 已优化的辅助函数：计算两个向量的皮尔逊相关系数
fn calculate_correlation_optimized(x: &ArrayView1<f64>, y: &ArrayView1<f64>, drop_na: bool) -> f64 {
    assert_eq!(x.len(), y.len());
    let n = x.len();
    
    // 如果长度为0，返回NaN
    if n == 0 {
        return f64::NAN;
    }
    
    // 优化的NaN处理：一次遍历计算均值和有效数据点
    // 这种方法减少了多次遍历向量的开销
    let mut count = 0;
    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    let mut valid_pairs = Vec::with_capacity(n);
    
    // 一次遍历找出有效值对并计算和
    for i in 0..n {
        let xi = x[i];
        let yi = y[i];
        
        if !xi.is_nan() && !yi.is_nan() {
            valid_pairs.push((xi, yi));
            sum_x += xi;
            sum_y += yi;
            count += 1;
        }
    }
    
    // 如果有NaN值且不允许忽略，或有效点太少
    if (valid_pairs.len() < n && !drop_na) || count < 2 {
        return f64::NAN;
    }
    
    let mean_x = sum_x / count as f64;
    let mean_y = sum_y / count as f64;
    
    // 计算协方差和方差 - 向量化计算以提高缓存局部性
    // 使用高效的向量累加，最大化CPU的SIMD指令使用
    
    // 使用预先分配和批处理来提高性能
    const BATCH_SIZE: usize = 32; // 根据CPU缓存行大小选择合适的批次大小
    
    let mut cov_xy = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;
    
    // 批量处理向量元素以提高缓存命中率
    for chunk in valid_pairs.chunks(BATCH_SIZE) {
        let mut local_cov_xy = 0.0;
        let mut local_var_x = 0.0;
        let mut local_var_y = 0.0;
        
        for &(xi, yi) in chunk {
            let dx = xi - mean_x;
            let dy = yi - mean_y;
            
            local_cov_xy += dx * dy;
            local_var_x += dx * dx;
            local_var_y += dy * dy;
        }
        
        cov_xy += local_cov_xy;
        var_x += local_var_x;
        var_y += local_var_y;
    }
    
    // 计算相关系数
    if var_x.abs() < f64::EPSILON || var_y.abs() < f64::EPSILON {
        return f64::NAN; // 如果任一变量的方差为0，则相关系数未定义
    }
    
    cov_xy / (var_x.sqrt() * var_y.sqrt())
}


