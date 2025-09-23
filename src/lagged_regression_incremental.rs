/// 增量更新优化版本的滞后自回归分析
///
/// 核心优化：利用滑动窗口的增量特性，避免重复计算X'X和X'y矩阵
/// 预期性能提升：30-40%
use faer::prelude::*;
use faer::{Mat, Side};
use numpy::{PyArray1, PyArray2};
use pyo3::prelude::*;
use std::collections::HashMap;

/// 增量更新的缓冲区结构
struct IncrementalRidgeBuffer {
    /// 当前X'X矩阵（对称矩阵）
    xtx_matrix: Mat<f64>,
    /// 当前X'y向量
    xty_vector: Mat<f64>,
    /// 滑动窗口的历史数据缓存
    window_cache: Vec<f64>,
    /// 当前窗口在原始数据中的起始位置
    current_window_start: usize,
    /// 当前窗口大小
    window_size: usize,
    /// 当前滞后阶数
    current_lag: usize,
    /// 系数求解缓冲区
    coeffs_buffer: Mat<f64>,
    /// 是否已初始化
    initialized: bool,
    /// 最大特征数
    max_features: usize,
}

impl IncrementalRidgeBuffer {
    fn new(max_features: usize) -> Self {
        Self {
            xtx_matrix: Mat::zeros(max_features, max_features),
            xty_vector: Mat::zeros(max_features, 1),
            window_cache: Vec::new(),
            current_window_start: 0,
            window_size: 0,
            current_lag: 0,
            coeffs_buffer: Mat::zeros(max_features, 1),
            initialized: false,
            max_features,
        }
    }

    /// 初始化第一个窗口的X'X和X'y矩阵
    fn initialize_window(&mut self, data: &[f64], lag: usize) {
        let n_obs = data.len() - lag;
        let n_features = lag + 1;

        // 确保矩阵大小足够
        if n_features > self.max_features {
            self.xtx_matrix = Mat::zeros(n_features, n_features);
            self.xty_vector = Mat::zeros(n_features, 1);
            self.coeffs_buffer = Mat::zeros(n_features, 1);
        }

        // 清零矩阵
        self.xtx_matrix.fill_zero();
        self.xty_vector.fill_zero();

        // 构建初始X'X和X'y
        for i in 0..n_obs {
            let y_val = data[i + lag];

            // 构建X向量 [1, x_{i+lag-1}, x_{i+lag-2}, ..., x_i]
            let mut x_vec = vec![1.0]; // 常数项
            for j in 0..lag {
                x_vec.push(data[i + lag - j - 1]);
            }

            // 更新X'X：X'X += x_i * x_i'
            for row in 0..n_features {
                for col in 0..n_features {
                    let old_val = self.xtx_matrix.read(row, col);
                    self.xtx_matrix
                        .write(row, col, old_val + x_vec[row] * x_vec[col]);
                }
            }

            // 更新X'y：X'y += x_i * y_i
            for row in 0..n_features {
                let old_val = self.xty_vector.read(row, 0);
                self.xty_vector.write(row, 0, old_val + x_vec[row] * y_val);
            }
        }

        // 缓存当前窗口数据
        self.window_cache = data.to_vec();
        self.window_size = data.len();
        self.current_lag = lag;
        self.initialized = true;
    }

    /// 增量更新到下一个窗口
    fn update_to_next_window(&mut self, new_data_point: f64) -> Result<(), String> {
        if !self.initialized {
            return Err("Buffer not initialized".to_string());
        }

        let lag = self.current_lag;
        let n_features = lag + 1;
        let n_obs = self.window_cache.len() - lag;

        // 移除最旧的观测（第一个有效观测）
        let y_old = self.window_cache[lag]; // 第一个y值

        // 构建要移除的X向量 [1, x_{lag-1}, x_{lag-2}, ..., x_0]
        let mut x_old = vec![1.0];
        for j in 0..lag {
            x_old.push(self.window_cache[lag - j - 1]);
        }

        // 从X'X中减去最旧观测的贡献
        for row in 0..n_features {
            for col in 0..n_features {
                let old_val = self.xtx_matrix.read(row, col);
                self.xtx_matrix
                    .write(row, col, old_val - x_old[row] * x_old[col]);
            }
        }

        // 从X'y中减去最旧观测的贡献
        for row in 0..n_features {
            let old_val = self.xty_vector.read(row, 0);
            self.xty_vector.write(row, 0, old_val - x_old[row] * y_old);
        }

        // 更新窗口缓存：移除第一个点，添加新的点
        self.window_cache.remove(0);
        self.window_cache.push(new_data_point);

        // 新观测的索引（最后一个有效观测）
        let cache_len = self.window_cache.len();
        let new_y_idx = cache_len - 1; // 新的y值索引
        let y_new = self.window_cache[new_y_idx];

        // 构建新的X向量 [1, x_{new-1}, x_{new-2}, ..., x_{new-lag}]
        let mut x_new = vec![1.0];
        for j in 0..lag {
            let x_idx = new_y_idx - j - 1;
            if x_idx < cache_len {
                x_new.push(self.window_cache[x_idx]);
            } else {
                return Err("Invalid window state".to_string());
            }
        }

        // 向X'X中添加新观测的贡献
        for row in 0..n_features {
            for col in 0..n_features {
                let old_val = self.xtx_matrix.read(row, col);
                self.xtx_matrix
                    .write(row, col, old_val + x_new[row] * x_new[col]);
            }
        }

        // 向X'y中添加新观测的贡献
        for row in 0..n_features {
            let old_val = self.xty_vector.read(row, 0);
            self.xty_vector.write(row, 0, old_val + x_new[row] * y_new);
        }

        Ok(())
    }

    /// 求解当前的Ridge回归系数
    fn solve_ridge_regression(&mut self, alpha: f64) -> Result<Vec<f64>, String> {
        let n_features = self.current_lag + 1;

        // 添加Ridge正则化项：X'X + αI
        let mut xtx_ridge = self
            .xtx_matrix
            .submatrix(0, 0, n_features, n_features)
            .to_owned();
        for i in 0..n_features {
            let old_val = xtx_ridge.read(i, i);
            xtx_ridge.write(i, i, old_val + alpha);
        }

        // 获取对应的X'y子向量
        let xty_sub = self.xty_vector.submatrix(0, 0, n_features, 1);

        // 使用Cholesky分解求解
        match xtx_ridge.cholesky(Side::Lower) {
            Ok(chol) => {
                let solution = chol.solve(&xty_sub);
                let coeffs: Vec<f64> = (0..n_features).map(|i| solution.read(i, 0)).collect();
                Ok(coeffs)
            }
            Err(_) => Err("Matrix is not positive definite".to_string()),
        }
    }
}

/// Alpha参数缓存（复用之前的实现）
struct AlphaCache {
    cache: HashMap<(usize, usize, u64), f64>,
}

impl AlphaCache {
    fn new() -> Self {
        Self {
            cache: HashMap::new(),
        }
    }

    fn get_or_compute(
        &mut self,
        n_obs: usize,
        n_features: usize,
        variance_hash: u64,
        alpha_override: Option<f64>,
    ) -> f64 {
        if let Some(alpha) = alpha_override {
            return alpha;
        }

        let key = (n_obs, n_features, variance_hash);
        if let Some(&cached_alpha) = self.cache.get(&key) {
            return cached_alpha;
        }

        let alpha = compute_adaptive_alpha(n_obs, n_features, variance_hash);
        self.cache.insert(key, alpha);
        alpha
    }
}

/// 计算自适应alpha参数（与原版本保持一致）
fn compute_adaptive_alpha(n_obs: usize, n_features: usize, variance_hash: u64) -> f64 {
    let base_alpha = 1.0;

    let sample_factor = if n_obs < 50 {
        2.0
    } else if n_obs < 100 {
        1.0
    } else {
        0.5
    };

    let feature_factor = (n_features as f64).sqrt();

    let variance = variance_hash as f64 / 1000.0;
    let noise_factor = if variance > 1e6 {
        2.0
    } else if variance > 1e3 {
        1.0
    } else {
        0.5
    };

    base_alpha * sample_factor * feature_factor * noise_factor
}

/// 计算y向量方差的哈希值
fn compute_y_variance_hash(data: &[f64], lag: usize) -> u64 {
    if data.len() <= lag {
        return 1000;
    }

    let n_obs = data.len() - lag;
    let mut y_vector = vec![0.0; n_obs];
    for i in 0..n_obs {
        y_vector[i] = data[i + lag];
    }

    let y_mean = y_vector.iter().sum::<f64>() / y_vector.len() as f64;
    let y_var =
        y_vector.iter().map(|&y| (y - y_mean).powi(2)).sum::<f64>() / (y_vector.len() as f64 - 1.0);

    (y_var * 1000.0).round() as u64
}

/// 计算AR模型的R²值（使用增量缓冲区）
fn calculate_ar_r_squared_incremental(buffer: &mut IncrementalRidgeBuffer, alpha: f64) -> f64 {
    let coeffs = match buffer.solve_ridge_regression(alpha) {
        Ok(c) => c,
        Err(_) => return f64::NAN,
    };

    let lag = buffer.current_lag;
    let n_obs = buffer.window_cache.len() - lag;

    // 计算预测值
    let mut predictions = Vec::with_capacity(n_obs);
    for i in 0..n_obs {
        let mut pred = coeffs[0]; // 常数项
        for j in 0..lag {
            pred += coeffs[j + 1] * buffer.window_cache[i + lag - j - 1];
        }
        predictions.push(pred);
    }

    // 计算R²
    let actual_values: Vec<f64> = (0..n_obs).map(|i| buffer.window_cache[i + lag]).collect();

    calculate_prediction_accuracy(&predictions, &actual_values)
}

/// 计算预测的R²值（滚动预测）
fn calculate_prediction_r_squared_incremental(
    past_buffer: &mut IncrementalRidgeBuffer,
    future_data: &[f64],
    alpha: f64,
) -> f64 {
    let coeffs = match past_buffer.solve_ridge_regression(alpha) {
        Ok(c) => c,
        Err(_) => return f64::NAN,
    };

    let lag = past_buffer.current_lag;
    let past_data = &past_buffer.window_cache;

    // 准备完整数据序列
    let mut full_data = past_data.clone();
    full_data.extend_from_slice(future_data);

    // 滚动预测
    let mut predictions = Vec::with_capacity(future_data.len());
    let past_len = past_data.len();

    for i in 0..future_data.len() {
        let mut pred = coeffs[0]; // 常数项

        for j in 0..lag {
            let data_idx = past_len + i - j - 1;
            if data_idx < full_data.len() {
                pred += coeffs[j + 1] * full_data[data_idx];
            }
        }
        predictions.push(pred);
    }

    calculate_prediction_accuracy(&predictions, future_data)
}

/// 计算预测准确度（R²）
fn calculate_prediction_accuracy(y_pred: &[f64], y_actual: &[f64]) -> f64 {
    if y_pred.len() != y_actual.len() || y_actual.is_empty() {
        return f64::NAN;
    }

    if !y_pred.iter().all(|&x| x.is_finite()) || !y_actual.iter().all(|&x| x.is_finite()) {
        return f64::NAN;
    }

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

/// 将一维向量转换为二维矩阵
fn vec_to_matrix(vec: Vec<f64>, rows: usize, cols: usize) -> Vec<Vec<f64>> {
    vec.chunks(cols).map(|chunk| chunk.to_vec()).collect()
}

/// 增量更新版本的滞后自回归分析
///
/// 核心优化：
/// 1. 维护X'X和X'y矩阵的增量更新，避免重复计算
/// 2. 利用滑动窗口的重叠特性，每次只计算差量
/// 3. 预期性能提升30-40%
#[pyfunction]
#[pyo3(signature = (series, past_periods, future_periods, alpha = None))]
pub fn rolling_lagged_regression_ridge_incremental(
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

    // 初始化增量缓冲区和缓存
    let mut alpha_cache = AlphaCache::new();
    let mut result = vec![f64::NAN; n * 2 * max_lag];

    // 为每个滞后阶数创建独立的增量缓冲区
    let mut buffers: Vec<IncrementalRidgeBuffer> = (1..=max_lag)
        .map(|_| IncrementalRidgeBuffer::new(max_lag + 1))
        .collect();

    // 主循环 - 滚动计算
    for (window_idx, i) in (total_periods..=n).enumerate() {
        let window_start = i - total_periods;
        let window_data = &data[window_start..i];

        let past_data = &window_data[0..past_periods];
        let future_data = &window_data[past_periods..];

        // 对每个滞后阶数进行计算
        for lag in 1..=max_lag {
            let buffer_idx = lag - 1;
            let buffer = &mut buffers[buffer_idx];

            // 计算alpha参数
            let variance_hash = compute_y_variance_hash(past_data, lag);
            let ridge_alpha =
                alpha_cache.get_or_compute(past_periods - lag, lag + 1, variance_hash, alpha);

            // 第一个窗口：完整初始化
            if window_idx == 0 {
                buffer.initialize_window(past_data, lag);
            } else {
                // 后续窗口：增量更新
                let new_point = data[i - 1];
                if let Err(_) = buffer.update_to_next_window(new_point) {
                    // 如果增量更新失败，回退到完整计算
                    buffer.initialize_window(past_data, lag);
                }
            }

            // 计算过去期拟合优度
            let r_past = calculate_ar_r_squared_incremental(buffer, ridge_alpha);

            // 计算未来期预测准确度
            let r_future =
                calculate_prediction_r_squared_incremental(buffer, future_data, ridge_alpha);

            // 存储结果
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
