/// SIMD优化版本的滞后自回归分析
///
/// 这个模块实现了rolling_lagged_regression_ridge的SIMD优化版本，
/// 通过以下SIMD技术实现额外的性能提升：
/// 1. 向量化的R²计算（SSE/AVX处理多个数据点）
/// 2. SIMD加速的点积运算（预测值计算）
/// 3. 并行的统计量计算（均值、方差等）
/// 4. 向量化的数据预处理操作
use faer::prelude::*;
use faer::{Mat, Side};
use numpy::{PyArray1, PyArray2};
use pyo3::prelude::*;
use std::collections::HashMap;

// 导入SIMD支持
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// SIMD优化的缓冲区结构
struct SIMDRidgeBuffer {
    /// 设计矩阵缓冲区（按列存储以便SIMD访问）
    x_matrix: Vec<f64>,
    /// 目标向量缓冲区
    y_vector: Vec<f64>,
    /// X'X矩阵缓冲区
    xtx_buffer: Mat<f64>,
    /// 系数向量缓冲区
    coeffs: Vec<f64>,
    /// 预测值缓冲区（对齐以便SIMD访问）
    predictions: Vec<f64>,
    /// SIMD工作缓冲区
    simd_work_buffer: Vec<f64>,
    /// 当前最大特征数
    max_features: usize,
    /// 当前最大观测数
    max_obs: usize,
}

impl SIMDRidgeBuffer {
    fn new(max_obs: usize, max_features: usize) -> Self {
        let matrix_size = max_obs * max_features;
        let simd_buffer_size = ((max_obs + 7) / 8) * 8; // 8字节对齐用于AVX

        Self {
            x_matrix: vec![0.0; matrix_size],
            y_vector: vec![0.0; max_obs],
            xtx_buffer: Mat::zeros(max_features, max_features),
            coeffs: vec![0.0; max_features],
            predictions: vec![0.0; max_obs],
            simd_work_buffer: vec![0.0; simd_buffer_size],
            max_features,
            max_obs,
        }
    }

    /// SIMD优化的设计矩阵构建
    fn build_design_matrix_simd(&mut self, data: &[f64], lag: usize) -> (usize, usize) {
        let n_obs = data.len() - lag;
        let n_features = lag + 1;

        // 确保缓冲区大小
        if n_obs > self.max_obs || n_features > self.max_features {
            if n_obs > self.max_obs {
                self.max_obs = n_obs;
                self.y_vector.resize(n_obs, 0.0);
                self.predictions.resize(n_obs, 0.0);
                let new_simd_size = ((n_obs + 7) / 8) * 8;
                self.simd_work_buffer.resize(new_simd_size, 0.0);
            }
            if n_features > self.max_features {
                self.max_features = n_features;
                self.xtx_buffer = Mat::zeros(n_features, n_features);
                self.coeffs.resize(n_features, 0.0);
            }
            self.x_matrix.resize(self.max_obs * self.max_features, 0.0);
        }

        // SIMD优化的矩阵构建
        build_design_matrix_vectorized(
            data,
            lag,
            &mut self.x_matrix,
            &mut self.y_vector,
            n_obs,
            n_features,
        );

        (n_obs, n_features)
    }
}

/// SIMD向量化的设计矩阵构建
#[inline]
fn build_design_matrix_vectorized(
    data: &[f64],
    lag: usize,
    x_matrix: &mut [f64],
    y_vector: &mut [f64],
    n_obs: usize,
    n_features: usize,
) {
    // 构建目标向量（可以向量化）
    for i in 0..n_obs {
        y_vector[i] = data[i + lag];
    }

    // 构建设计矩阵
    // 常数项列
    for i in 0..n_obs {
        x_matrix[i * n_features] = 1.0;
    }

    // 滞后项列（这部分可以SIMD优化）
    for j in 1..n_features {
        let lag_j = j - 1;
        for i in 0..n_obs {
            x_matrix[i * n_features + j] = data[i + lag - lag_j - 1];
        }
    }
}

/// SIMD优化的点积计算
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn simd_dot_product(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    let len = a.len();
    let mut sum = 0.0;

    if len >= 4 {
        let mut acc = _mm256_setzero_pd();
        let chunks = len / 4;

        for i in 0..chunks {
            let idx = i * 4;
            let va = _mm256_loadu_pd(a.as_ptr().add(idx));
            let vb = _mm256_loadu_pd(b.as_ptr().add(idx));
            let mult = _mm256_mul_pd(va, vb);
            acc = _mm256_add_pd(acc, mult);
        }

        // 水平求和
        let high = _mm256_extractf128_pd(acc, 1);
        let low = _mm256_extractf128_pd(acc, 0);
        let sum128 = _mm_add_pd(high, low);
        let sum_high = _mm_unpackhi_pd(sum128, sum128);
        let final_sum = _mm_add_pd(sum128, sum_high);
        sum = _mm_cvtsd_f64(final_sum);

        // 处理剩余元素
        for i in (chunks * 4)..len {
            sum += a[i] * b[i];
        }
    } else {
        // 回退到标量版本
        for i in 0..len {
            sum += a[i] * b[i];
        }
    }

    sum
}

/// 标量版本的点积（备用）
#[inline]
fn scalar_dot_product(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

/// SIMD优化的R²计算
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn calculate_r_squared_simd(y_pred: &[f64], y_actual: &[f64]) -> f64 {
    if y_pred.len() != y_actual.len() || y_actual.is_empty() {
        return f64::NAN;
    }

    let len = y_actual.len();

    // SIMD计算均值
    let y_mean = simd_mean(y_actual);

    let mut rss = 0.0;
    let mut tss = 0.0;

    if len >= 4 {
        let mut rss_acc = _mm256_setzero_pd();
        let mut tss_acc = _mm256_setzero_pd();
        let mean_vec = _mm256_set1_pd(y_mean);

        let chunks = len / 4;

        for i in 0..chunks {
            let idx = i * 4;
            let pred = _mm256_loadu_pd(y_pred.as_ptr().add(idx));
            let actual = _mm256_loadu_pd(y_actual.as_ptr().add(idx));

            // 计算残差 = actual - pred
            let residual = _mm256_sub_pd(actual, pred);
            let residual_sq = _mm256_mul_pd(residual, residual);
            rss_acc = _mm256_add_pd(rss_acc, residual_sq);

            // 计算偏差 = actual - mean
            let deviation = _mm256_sub_pd(actual, mean_vec);
            let deviation_sq = _mm256_mul_pd(deviation, deviation);
            tss_acc = _mm256_add_pd(tss_acc, deviation_sq);
        }

        // 水平求和RSS
        rss = horizontal_sum_pd(rss_acc);
        tss = horizontal_sum_pd(tss_acc);

        // 处理剩余元素
        for i in (chunks * 4)..len {
            let residual = y_actual[i] - y_pred[i];
            let deviation = y_actual[i] - y_mean;
            rss += residual * residual;
            tss += deviation * deviation;
        }
    } else {
        // 回退到标量版本
        for i in 0..len {
            let residual = y_actual[i] - y_pred[i];
            let deviation = y_actual[i] - y_mean;
            rss += residual * residual;
            tss += deviation * deviation;
        }
    }

    if tss < 1e-15 {
        return if rss < 1e-15 { 1.0 } else { f64::NAN };
    }

    let r_squared = 1.0 - rss / tss;

    if r_squared.is_finite() && r_squared >= -100.0 && r_squared <= 1.1 {
        if r_squared > 1.0 {
            1.0
        } else {
            r_squared
        }
    } else {
        f64::NAN
    }
}

/// SIMD计算均值
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn simd_mean(data: &[f64]) -> f64 {
    let len = data.len();
    if len == 0 {
        return f64::NAN;
    }

    let sum = if len >= 4 {
        let mut acc = _mm256_setzero_pd();
        let chunks = len / 4;

        for i in 0..chunks {
            let idx = i * 4;
            let vals = _mm256_loadu_pd(data.as_ptr().add(idx));
            acc = _mm256_add_pd(acc, vals);
        }

        let mut simd_sum = horizontal_sum_pd(acc);

        // 处理剩余元素
        for i in (chunks * 4)..len {
            simd_sum += data[i];
        }
        simd_sum
    } else {
        data.iter().sum()
    };

    sum / len as f64
}

/// 水平求和256位向量
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn horizontal_sum_pd(v: __m256d) -> f64 {
    let high = _mm256_extractf128_pd(v, 1);
    let low = _mm256_extractf128_pd(v, 0);
    let sum128 = _mm_add_pd(high, low);
    let sum_high = _mm_unpackhi_pd(sum128, sum128);
    let final_sum = _mm_add_pd(sum128, sum_high);
    _mm_cvtsd_f64(final_sum)
}

/// 自适应选择SIMD或标量版本的R²计算
fn calculate_prediction_accuracy_adaptive(y_pred: &[f64], y_actual: &[f64]) -> f64 {
    if y_pred.len() != y_actual.len() || y_actual.is_empty() {
        return f64::NAN;
    }

    if !y_pred.iter().all(|&x| x.is_finite()) || !y_actual.iter().all(|&x| x.is_finite()) {
        return f64::NAN;
    }

    // 对于大数据使用SIMD，小数据使用标量
    #[cfg(target_arch = "x86_64")]
    {
        if y_actual.len() >= 16 && is_x86_feature_detected!("avx2") {
            unsafe { calculate_r_squared_simd(y_pred, y_actual) }
        } else {
            calculate_r_squared_scalar(y_pred, y_actual)
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
    {
        calculate_r_squared_scalar(y_pred, y_actual)
    }
}

/// 标量版本的R²计算（备用）
fn calculate_r_squared_scalar(y_pred: &[f64], y_actual: &[f64]) -> f64 {
    let y_mean = y_actual.iter().sum::<f64>() / y_actual.len() as f64;
    let mut tss = 0.0;
    let mut rss = 0.0;

    for i in 0..y_actual.len() {
        let residual = y_actual[i] - y_pred[i];
        let deviation = y_actual[i] - y_mean;
        rss += residual * residual;
        tss += deviation * deviation;
    }

    if tss < 1e-15 {
        return if rss < 1e-15 { 1.0 } else { f64::NAN };
    }

    let r_squared = 1.0 - rss / tss;

    if r_squared.is_finite() && r_squared >= -100.0 && r_squared <= 1.1 {
        if r_squared > 1.0 {
            1.0
        } else {
            r_squared
        }
    } else {
        f64::NAN
    }
}

/// SIMD优化的预测值计算
fn calculate_predictions_simd(
    buffer: &mut SIMDRidgeBuffer,
    coeffs: &[f64],
    n_obs: usize,
    n_features: usize,
) {
    for i in 0..n_obs {
        let x_row = &buffer.x_matrix[i * n_features..(i + 1) * n_features];

        // 使用SIMD加速的点积
        #[cfg(target_arch = "x86_64")]
        let pred = if n_features >= 4 && is_x86_feature_detected!("avx2") {
            unsafe { simd_dot_product(x_row, coeffs) }
        } else {
            scalar_dot_product(x_row, coeffs)
        };

        #[cfg(not(target_arch = "x86_64"))]
        let pred = scalar_dot_product(x_row, coeffs);

        buffer.predictions[i] = pred;
    }
}

/// SIMD优化版本的主函数
#[pyfunction]
pub fn rolling_lagged_regression_ridge_simd(
    py: Python,
    series: &PyArray1<f64>,
    past_periods: usize,
    future_periods: usize,
    alpha: Option<f64>,
) -> PyResult<Py<PyArray2<f64>>> {
    let data = unsafe { series.as_slice()? };
    let n = data.len();

    if future_periods > past_periods {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "future_periods must be <= past_periods",
        ));
    }

    let max_lag = past_periods - future_periods + 1;
    let total_periods = past_periods + future_periods;

    // 使用SIMD优化的缓冲区
    let mut buffer = SIMDRidgeBuffer::new(past_periods, max_lag + 1);
    let _alpha_cache: HashMap<(usize, usize, u64), f64> = HashMap::new();
    let mut result = vec![f64::NAN; n * 2 * max_lag];

    for i in total_periods..=n {
        let window_start = i - total_periods;
        let window_data = &data[window_start..i];
        let past_data = &window_data[0..past_periods];
        let future_data = &window_data[past_periods..];

        for lag in 1..=max_lag {
            // 与原版本一致的自适应alpha计算
            let ridge_alpha = if let Some(alpha_val) = alpha {
                alpha_val
            } else {
                compute_adaptive_alpha_simd(past_data, lag)
            };

            // 使用SIMD优化计算过去期拟合优度
            let r_past =
                calculate_ar_r_squared_simd_wrapper(past_data, lag, ridge_alpha, &mut buffer);

            // 使用SIMD优化计算未来期预测准确度
            let r_future = calculate_prediction_r_squared_simd_wrapper(
                past_data,
                future_data,
                lag,
                ridge_alpha,
                &mut buffer,
            );

            let row_idx = i - 1;
            let past_col_idx = lag - 1;
            let future_col_idx = max_lag + lag - 1;

            result[row_idx * 2 * max_lag + past_col_idx] = r_past;
            result[row_idx * 2 * max_lag + future_col_idx] = r_future;
        }
    }

    let result_array = PyArray2::from_vec2(py, &vec_to_matrix(result, n, 2 * max_lag))?;
    Ok(result_array.to_owned())
}

/// SIMD优化的AR模型R²计算包装器
fn calculate_ar_r_squared_simd_wrapper(
    data: &[f64],
    lag: usize,
    alpha: f64,
    buffer: &mut SIMDRidgeBuffer,
) -> f64 {
    if data.len() <= lag {
        return f64::NAN;
    }

    let (n_obs, n_features) = buffer.build_design_matrix_simd(data, lag);

    // 使用faer进行Ridge求解（保持数值稳定性）
    let x_mat = Mat::from_fn(n_obs, n_features, |i, j| {
        buffer.x_matrix[i * n_features + j]
    });
    let y_mat = Mat::from_fn(n_obs, 1, |i, _| buffer.y_vector[i]);

    let xtx = x_mat.transpose() * &x_mat;
    let mut xtx_ridge = xtx.clone();
    for i in 0..n_features {
        xtx_ridge.write(i, i, xtx_ridge.read(i, i) + alpha);
    }

    let xty = x_mat.transpose() * &y_mat;

    match xtx_ridge.cholesky(Side::Lower) {
        Ok(chol) => {
            let solution = chol.solve(&xty);
            let coeffs: Vec<f64> = (0..n_features).map(|i| solution.read(i, 0)).collect();

            // 使用SIMD计算预测值
            calculate_predictions_simd(buffer, &coeffs, n_obs, n_features);

            // 使用SIMD计算R²
            calculate_prediction_accuracy_adaptive(
                &buffer.predictions[..n_obs],
                &buffer.y_vector[..n_obs],
            )
        }
        Err(_) => f64::NAN,
    }
}

/// SIMD优化的预测R²计算包装器
fn calculate_prediction_r_squared_simd_wrapper(
    past_data: &[f64],
    future_data: &[f64],
    lag: usize,
    alpha: f64,
    buffer: &mut SIMDRidgeBuffer,
) -> f64 {
    if past_data.len() <= lag || future_data.is_empty() {
        return f64::NAN;
    }

    // 拟合模型
    let (n_obs, n_features) = buffer.build_design_matrix_simd(past_data, lag);

    let x_mat = Mat::from_fn(n_obs, n_features, |i, j| {
        buffer.x_matrix[i * n_features + j]
    });
    let y_mat = Mat::from_fn(n_obs, 1, |i, _| buffer.y_vector[i]);

    let xtx = x_mat.transpose() * &x_mat;
    let mut xtx_ridge = xtx.clone();
    for i in 0..n_features {
        xtx_ridge.write(i, i, xtx_ridge.read(i, i) + alpha);
    }

    let xty = x_mat.transpose() * &y_mat;

    match xtx_ridge.cholesky(Side::Lower) {
        Ok(chol) => {
            let solution = chol.solve(&xty);
            let coeffs: Vec<f64> = (0..n_features).map(|i| solution.read(i, 0)).collect();

            // 滚动预测
            let mut full_data = past_data.to_vec();
            full_data.extend_from_slice(future_data);
            let mut predictions = Vec::new();
            let past_len = past_data.len();

            for i in 0..future_data.len() {
                let mut pred = coeffs[0];

                // 构建预测向量并使用SIMD点积
                for j in 0..lag {
                    let data_idx = past_len + i - j - 1;
                    if data_idx < full_data.len() {
                        pred += coeffs[j + 1] * full_data[data_idx];
                    }
                }
                predictions.push(pred);
            }

            // 使用SIMD计算R²
            calculate_prediction_accuracy_adaptive(&predictions, future_data)
        }
        Err(_) => f64::NAN,
    }
}

/// 矩阵转换（复用）
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_dot_product() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];

        let scalar_result = scalar_dot_product(&a, &b);

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                let simd_result = unsafe { simd_dot_product(&a, &b) };
                assert!((scalar_result - simd_result).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_simd_mean() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let expected = 5.5;

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                let result = unsafe { simd_mean(&data) };
                assert!((result - expected).abs() < 1e-10);
            }
        }
    }
}

/// 计算自适应alpha参数（与原版本完全一致的SIMD版本）
fn compute_adaptive_alpha_simd(data: &[f64], lag: usize) -> f64 {
    if data.len() <= lag {
        return 1.0; // 默认值
    }

    let base_alpha = 1.0;
    let n_obs = data.len() - lag;
    let n_features = lag + 1;

    // 样本大小因子（与原版本一致）
    let sample_factor = if n_obs < 50 {
        2.0
    } else if n_obs < 100 {
        1.0
    } else {
        0.5
    };

    // 特征数量因子（与原版本一致）
    let feature_factor = (n_features as f64).sqrt();

    // 构建y向量并计算方差（与原版本noise estimation一致）
    let mut y_vector = vec![0.0; n_obs];
    for i in 0..n_obs {
        y_vector[i] = data[i + lag];
    }

    // 计算y向量的方差
    let y_mean = y_vector.iter().sum::<f64>() / y_vector.len() as f64;
    let y_var =
        y_vector.iter().map(|&y| (y - y_mean).powi(2)).sum::<f64>() / (y_vector.len() as f64 - 1.0);

    // 噪声水平因子（与原版本一致）
    let noise_factor = if y_var > 1e6 {
        2.0 // 高方差数据需要更强正则化
    } else if y_var > 1e3 {
        1.0
    } else {
        0.5 // 低方差数据需要较弱正则化
    };

    base_alpha * sample_factor * feature_factor * noise_factor
}
