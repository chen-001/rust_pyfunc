use nalgebra::{DMatrix, DVector};
use numpy::{PyArray1, PyArray2};
use pyo3::prelude::*;

/// 滞后自回归分析函数
///
/// 参数:
/// - series: 一维时间序列数据
/// - past_periods: 过去观察期数
/// - future_periods: 预测期数，必须 <= past_periods
///
/// 返回:
/// - 二维数组 (n, 2*x)，其中 x = past_periods - future_periods + 1
/// - 每行包含 [r_lag1_past, r_lag2_past, ..., r_lagx_past, r_lag1_future, r_lag2_future, ..., r_lagx_future]
#[pyfunction]
pub fn rolling_lagged_regression(
    py: Python,
    series: &PyArray1<f64>,
    past_periods: usize,
    future_periods: usize,
) -> PyResult<Py<PyArray2<f64>>> {
    let data = unsafe { series.as_slice()? };
    let n = data.len();

    // 验证参数
    if future_periods > past_periods {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "future_periods must be <= past_periods",
        ));
    }

    // 计算最大滞后阶数
    let max_lag = past_periods - future_periods + 1;
    let total_periods = past_periods + future_periods;

    // 初始化结果矩阵，使用 NaN 填充
    let mut result = vec![f64::NAN; n * 2 * max_lag];

    // 对每一行进行滚动计算
    for i in total_periods..=n {
        // 提取当前窗口数据
        let window_start = i - total_periods;
        let window_data = &data[window_start..i];

        // 分离过去期和未来期数据
        let past_data = &window_data[0..past_periods];
        let future_data = &window_data[past_periods..];

        // 对每个滞后阶数进行计算
        for lag in 1..=max_lag {
            // 计算过去期拟合优度
            let r_past = calculate_ar_r_squared(past_data, lag);

            // 计算未来期预测准确度
            let r_future = calculate_prediction_r_squared(past_data, future_data, lag);

            // 存储结果
            let row_idx = i - 1;
            let past_col_idx = lag - 1;
            let future_col_idx = max_lag + lag - 1;

            result[row_idx * 2 * max_lag + past_col_idx] = r_past;
            result[row_idx * 2 * max_lag + future_col_idx] = r_future;
        }
    }

    // 转换为 numpy 数组
    let result_array = PyArray2::from_vec2(py, &vec_to_matrix(result, n, 2 * max_lag))?;
    Ok(result_array.to_owned())
}

/// 计算AR模型的R²值
fn calculate_ar_r_squared(data: &[f64], lag: usize) -> f64 {
    if data.len() <= lag {
        return f64::NAN;
    }

    // 构建设计矩阵和目标向量
    let n_obs = data.len() - lag;
    let mut x_matrix = vec![1.0; n_obs * (lag + 1)]; // 包含常数项
    let mut y_vector = vec![0.0; n_obs];

    for i in 0..n_obs {
        y_vector[i] = data[i + lag];
        // 常数项已经初始化为1.0
        for j in 0..lag {
            x_matrix[i * (lag + 1) + j + 1] = data[i + lag - j - 1];
        }
    }

    // 使用最小二乘法计算R²
    calculate_r_squared(&x_matrix, &y_vector, n_obs, lag + 1)
}

/// 计算预测的R²值 - 使用滚动预测而非递归预测
fn calculate_prediction_r_squared(past_data: &[f64], future_data: &[f64], lag: usize) -> f64 {
    if past_data.len() <= lag || future_data.is_empty() {
        return f64::NAN;
    }

    // 使用过去数据拟合AR模型
    let coeffs = fit_ar_model(past_data, lag);
    if coeffs.is_empty() {
        return f64::NAN;
    }

    // 准备完整的数据序列（过去数据 + 未来数据）
    let mut full_data = past_data.to_vec();
    full_data.extend_from_slice(future_data);

    // 使用滚动预测：每次预测都基于真实的历史数据
    let mut predictions = Vec::new();
    let past_len = past_data.len();

    for i in 0..future_data.len() {
        let mut pred = coeffs[0]; // 常数项

        // 使用真实的历史数据作为滞后项（包括已观测的未来数据）
        for j in 0..lag {
            let data_idx = past_len + i - j - 1;
            if data_idx < full_data.len() {
                pred += coeffs[j + 1] * full_data[data_idx];
            }
        }
        predictions.push(pred);
    }

    // 计算预测值与实际值的R²
    calculate_prediction_accuracy(&predictions, future_data)
}

/// 拟合AR模型并返回系数
fn fit_ar_model(data: &[f64], lag: usize) -> Vec<f64> {
    if data.len() <= lag {
        return Vec::new();
    }

    let n_obs = data.len() - lag;
    let mut x_matrix = vec![1.0; n_obs * (lag + 1)];
    let mut y_vector = vec![0.0; n_obs];

    for i in 0..n_obs {
        y_vector[i] = data[i + lag];
        for j in 0..lag {
            x_matrix[i * (lag + 1) + j + 1] = data[i + lag - j - 1];
        }
    }

    // 使用矩阵运算求解最小二乘
    solve_least_squares(&x_matrix, &y_vector, n_obs, lag + 1)
}

/// 最小二乘法求解
fn solve_least_squares(x_data: &[f64], y_data: &[f64], n_rows: usize, n_cols: usize) -> Vec<f64> {
    // 使用 nalgebra 进行矩阵运算
    let x_matrix = DMatrix::from_row_slice(n_rows, n_cols, x_data);
    let y_vector = DVector::from_column_slice(y_data);

    // 添加数值稳定性检查
    if n_rows < n_cols {
        return Vec::new();
    }

    // 计算 (X'X)^-1 X'y
    let xtx = x_matrix.transpose() * &x_matrix;

    // 检查是否为奇异矩阵
    match xtx.try_inverse() {
        Some(inv) => {
            let coeffs = inv * x_matrix.transpose() * y_vector;
            let result = coeffs.as_slice().to_vec();

            // 检查结果是否有效（无 NaN 或无穷大）
            if result.iter().all(|&x| x.is_finite()) {
                result
            } else {
                Vec::new()
            }
        }
        None => Vec::new(),
    }
}

/// 计算R²值
fn calculate_r_squared(x_data: &[f64], y_data: &[f64], n_rows: usize, n_cols: usize) -> f64 {
    let coeffs = solve_least_squares(x_data, y_data, n_rows, n_cols);
    if coeffs.is_empty() {
        return f64::NAN;
    }

    // 计算预测值
    let mut y_pred = vec![0.0; n_rows];
    for i in 0..n_rows {
        for j in 0..n_cols {
            y_pred[i] += coeffs[j] * x_data[i * n_cols + j];
        }
    }

    // 计算R²
    calculate_prediction_accuracy(&y_pred, y_data)
}

/// 计算预测准确度（R²）
fn calculate_prediction_accuracy(y_pred: &[f64], y_actual: &[f64]) -> f64 {
    if y_pred.len() != y_actual.len() || y_actual.is_empty() {
        return f64::NAN;
    }

    // 检查输入数据是否有效
    if !y_pred.iter().all(|&x| x.is_finite()) || !y_actual.iter().all(|&x| x.is_finite()) {
        return f64::NAN;
    }

    // 计算均值
    let y_mean = y_actual.iter().sum::<f64>() / y_actual.len() as f64;

    // 计算总平方和(TSS)和残差平方和(RSS)
    let mut tss = 0.0;
    let mut rss = 0.0;

    for i in 0..y_actual.len() {
        let residual = y_actual[i] - y_pred[i];
        let deviation = y_actual[i] - y_mean;

        rss += residual * residual;
        tss += deviation * deviation;
    }

    // 检查TSS是否为0（所有实际值都相同）
    if tss < 1e-15 {
        // 如果所有实际值都相同，且预测完全准确，R²=1
        if rss < 1e-15 {
            return 1.0;
        } else {
            return f64::NAN;
        }
    }

    // R² = 1 - RSS/TSS
    let r_squared = 1.0 - rss / tss;

    // 返回合理的 R² 值，防止异常情况
    if r_squared.is_finite() {
        // 限制R²在合理范围内，防止极端异常值
        // 对于时间序列预测，R²可能为负（预测效果差于简单均值）
        // 但极大的负值通常表明数值问题
        if r_squared < -100.0 {
            f64::NAN
        } else if r_squared > 1.0 {
            // R²理论上不应超过1，但由于数值误差可能略超过
            if r_squared > 1.1 {
                f64::NAN
            } else {
                1.0
            }
        } else {
            r_squared
        }
    } else {
        f64::NAN
    }
}

/// 将向量转换为矩阵格式
fn vec_to_matrix(data: Vec<f64>, rows: usize, cols: usize) -> Vec<Vec<f64>> {
    let mut matrix = Vec::with_capacity(rows);
    for i in 0..rows {
        let mut row = Vec::with_capacity(cols);
        for j in 0..cols {
            row.push(data[i * cols + j]);
        }
        matrix.push(row);
    }
    matrix
}

/// Ridge回归版本的滞后自回归分析函数
///
/// 参数:
/// - series: 一维时间序列数据
/// - past_periods: 过去观察期数
/// - future_periods: 预测期数，必须 <= past_periods
/// - alpha: Ridge正则化参数，默认使用自适应选择
///
/// 返回:
/// - 二维数组 (n, 2*x)，其中 x = past_periods - future_periods + 1
/// - 每行包含 [r_lag1_past, r_lag2_past, ..., r_lagx_past, r_lag1_future, r_lag2_future, ..., r_lagx_future]
#[pyfunction]
pub fn rolling_lagged_regression_ridge(
    py: Python,
    series: &PyArray1<f64>,
    past_periods: usize,
    future_periods: usize,
    alpha: Option<f64>,
) -> PyResult<Py<PyArray2<f64>>> {
    let data = unsafe { series.as_slice()? };
    let n = data.len();

    // 验证参数
    if future_periods > past_periods {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "future_periods must be <= past_periods",
        ));
    }

    // 计算最大滞后阶数
    let max_lag = past_periods - future_periods + 1;
    let total_periods = past_periods + future_periods;

    // 初始化结果矩阵，使用 NaN 填充
    let mut result = vec![f64::NAN; n * 2 * max_lag];

    // 对每一行进行滚动计算
    for i in total_periods..=n {
        // 提取当前窗口数据
        let window_start = i - total_periods;
        let window_data = &data[window_start..i];

        // 分离过去期和未来期数据
        let past_data = &window_data[0..past_periods];
        let future_data = &window_data[past_periods..];

        // 对每个滞后阶数进行计算
        for lag in 1..=max_lag {
            // 计算过去期拟合优度（使用Ridge回归）
            let r_past = calculate_ridge_ar_r_squared(past_data, lag, alpha);

            // 计算未来期预测准确度（使用Ridge回归）
            let r_future = calculate_ridge_prediction_r_squared(past_data, future_data, lag, alpha);

            // 存储结果
            let row_idx = i - 1;
            let past_col_idx = lag - 1;
            let future_col_idx = max_lag + lag - 1;

            result[row_idx * 2 * max_lag + past_col_idx] = r_past;
            result[row_idx * 2 * max_lag + future_col_idx] = r_future;
        }
    }

    // 转换为 numpy 数组
    let result_array = PyArray2::from_vec2(py, &vec_to_matrix(result, n, 2 * max_lag))?;
    Ok(result_array.to_owned())
}

/// 计算Ridge AR模型的R²值
fn calculate_ridge_ar_r_squared(data: &[f64], lag: usize, alpha: Option<f64>) -> f64 {
    if data.len() <= lag {
        return f64::NAN;
    }

    // 构建设计矩阵和目标向量
    let n_obs = data.len() - lag;
    let mut x_matrix = vec![1.0; n_obs * (lag + 1)]; // 包含常数项
    let mut y_vector = vec![0.0; n_obs];

    for i in 0..n_obs {
        y_vector[i] = data[i + lag];
        // 常数项已经初始化为1.0
        for j in 0..lag {
            x_matrix[i * (lag + 1) + j + 1] = data[i + lag - j - 1];
        }
    }

    // 使用Ridge回归计算R²
    calculate_ridge_r_squared(&x_matrix, &y_vector, n_obs, lag + 1, alpha)
}

/// 计算Ridge预测的R²值
fn calculate_ridge_prediction_r_squared(
    past_data: &[f64],
    future_data: &[f64],
    lag: usize,
    alpha: Option<f64>,
) -> f64 {
    if past_data.len() <= lag || future_data.is_empty() {
        return f64::NAN;
    }

    // 使用过去数据拟合Ridge AR模型
    let coeffs = fit_ridge_ar_model(past_data, lag, alpha);
    if coeffs.is_empty() {
        return f64::NAN;
    }

    // 准备完整的数据序列（过去数据 + 未来数据）
    let mut full_data = past_data.to_vec();
    full_data.extend_from_slice(future_data);

    // 使用滚动预测：每次预测都基于真实的历史数据
    let mut predictions = Vec::new();
    let past_len = past_data.len();

    for i in 0..future_data.len() {
        let mut pred = coeffs[0]; // 常数项

        // 使用真实的历史数据作为滞后项（包括已观测的未来数据）
        for j in 0..lag {
            let data_idx = past_len + i - j - 1;
            if data_idx < full_data.len() {
                pred += coeffs[j + 1] * full_data[data_idx];
            }
        }
        predictions.push(pred);
    }

    // 计算预测值与实际值的R²
    calculate_prediction_accuracy(&predictions, future_data)
}

/// 拟合Ridge AR模型并返回系数
fn fit_ridge_ar_model(data: &[f64], lag: usize, alpha: Option<f64>) -> Vec<f64> {
    if data.len() <= lag {
        return Vec::new();
    }

    let n_obs = data.len() - lag;
    let mut x_matrix = vec![1.0; n_obs * (lag + 1)];
    let mut y_vector = vec![0.0; n_obs];

    for i in 0..n_obs {
        y_vector[i] = data[i + lag];
        for j in 0..lag {
            x_matrix[i * (lag + 1) + j + 1] = data[i + lag - j - 1];
        }
    }

    // 使用Ridge回归求解
    solve_ridge_least_squares(&x_matrix, &y_vector, n_obs, lag + 1, alpha)
}

/// Ridge最小二乘法求解
fn solve_ridge_least_squares(
    x_data: &[f64],
    y_data: &[f64],
    n_rows: usize,
    n_cols: usize,
    alpha: Option<f64>,
) -> Vec<f64> {
    // 使用 nalgebra 进行矩阵运算
    let x_matrix = DMatrix::from_row_slice(n_rows, n_cols, x_data);
    let y_vector = DVector::from_column_slice(y_data);

    // 添加数值稳定性检查
    if n_rows < n_cols {
        return Vec::new();
    }

    // 自适应选择正则化参数
    let ridge_alpha =
        alpha.unwrap_or_else(|| adaptive_ridge_alpha(n_rows, n_cols, &x_matrix, &y_vector));

    // 计算 X'X
    let xtx = x_matrix.transpose() * &x_matrix;

    // 添加Ridge正则化项: X'X + αI
    let mut xtx_ridge = xtx.clone();
    for i in 0..n_cols {
        xtx_ridge[(i, i)] += ridge_alpha;
    }

    // 计算Ridge解: (X'X + αI)^(-1) X'y
    match xtx_ridge.try_inverse() {
        Some(inv) => {
            let coeffs = inv * x_matrix.transpose() * y_vector;
            let result = coeffs.as_slice().to_vec();

            // 检查结果是否有效（无 NaN 或无穷大）
            if result.iter().all(|&x| x.is_finite()) {
                result
            } else {
                Vec::new()
            }
        }
        None => Vec::new(),
    }
}

/// 自适应选择Ridge正则化参数
fn adaptive_ridge_alpha(
    n_rows: usize,
    n_cols: usize,
    x_matrix: &DMatrix<f64>,
    y_vector: &DVector<f64>,
) -> f64 {
    // 基于数据特征自适应选择alpha
    let base_alpha = 1.0;

    // 考虑样本大小：样本越小，正则化越强
    let sample_factor = if n_rows < 50 {
        2.0
    } else if n_rows < 100 {
        1.0
    } else {
        0.5
    };

    // 考虑特征数量：特征越多，正则化越强
    let feature_factor = (n_cols as f64).sqrt();

    // 估计噪声水平
    let noise_factor = estimate_noise_level(x_matrix, y_vector);

    base_alpha * sample_factor * feature_factor * noise_factor
}

/// 估计数据的噪声水平
fn estimate_noise_level(_x_matrix: &DMatrix<f64>, y_vector: &DVector<f64>) -> f64 {
    // 简单估计：基于y的方差
    let y_mean = y_vector.mean();
    let y_var =
        y_vector.iter().map(|&y| (y - y_mean).powi(2)).sum::<f64>() / (y_vector.len() as f64 - 1.0);

    // 基于方差调整正则化强度
    if y_var > 1e6 {
        2.0 // 高方差数据需要更强正则化
    } else if y_var > 1e3 {
        1.0
    } else {
        0.5 // 低方差数据需要较弱正则化
    }
}

/// 计算Ridge R²值
fn calculate_ridge_r_squared(
    x_data: &[f64],
    y_data: &[f64],
    n_rows: usize,
    n_cols: usize,
    alpha: Option<f64>,
) -> f64 {
    let coeffs = solve_ridge_least_squares(x_data, y_data, n_rows, n_cols, alpha);
    if coeffs.is_empty() {
        return f64::NAN;
    }

    // 计算预测值
    let mut y_pred = vec![0.0; n_rows];
    for i in 0..n_rows {
        for j in 0..n_cols {
            y_pred[i] += coeffs[j] * x_data[i * n_cols + j];
        }
    }

    // 计算R²
    calculate_prediction_accuracy(&y_pred, y_data)
}
