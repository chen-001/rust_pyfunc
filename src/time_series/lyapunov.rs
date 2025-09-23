use ndarray::{s, Array1, Array2};
use numpy::{IntoPyArray, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// 自动调整参数以确保相空间重构的安全性（增强版）
fn auto_adjust_parameters(data_len: usize, m: usize, tau: usize) -> (usize, usize) {
    // 设置更严格的限制
    let min_rows_required = 5; // 最少需要5个相空间点
    let max_safe_elements = 10_000; // 相空间数组最大元素数

    // 对于极短数据，直接返回最小参数
    if data_len < 10 {
        return (2, 1);
    }

    // 安全的初始值
    let mut safe_m = m.min(10).max(2); // 限制m的范围
    let mut safe_tau = tau.min(data_len / 5).max(1); // 限制tau的范围

    // 迭代调整直到找到安全参数
    for _attempt in 0..20 {
        // 最多尝试20次
        let required_length = (safe_m.saturating_sub(1)).saturating_mul(safe_tau);

        // 检查数据长度是否足够
        if data_len <= required_length {
            // 减小参数
            if safe_tau > 1 {
                safe_tau = safe_tau.saturating_sub(1);
            } else if safe_m > 2 {
                safe_m = safe_m.saturating_sub(1);
                safe_tau = tau.min(data_len / 5).max(1);
            } else {
                // 已经是最小参数，但仍不够
                return (2, 1);
            }
            continue;
        }

        let rows = data_len - required_length;

        // 检查是否满足最小行数要求
        if rows < min_rows_required {
            if safe_tau > 1 {
                safe_tau = safe_tau.saturating_sub(1);
            } else if safe_m > 2 {
                safe_m = safe_m.saturating_sub(1);
                safe_tau = tau.min(data_len / 5).max(1);
            } else {
                return (2, 1);
            }
            continue;
        }

        // 检查数组大小是否安全
        let array_elements = rows.saturating_mul(safe_m);
        if array_elements > max_safe_elements {
            // 数组太大，减小参数
            if safe_m > 2 {
                safe_m = safe_m.saturating_sub(1);
            } else if safe_tau > 1 {
                safe_tau = safe_tau.saturating_sub(1);
            } else {
                return (2, 1);
            }
            continue;
        }

        // 检查isize溢出
        if rows > (isize::MAX as usize) || safe_m > (isize::MAX as usize) {
            if safe_m > 2 {
                safe_m = safe_m.saturating_sub(1);
            } else if safe_tau > 1 {
                safe_tau = safe_tau.saturating_sub(1);
            } else {
                return (2, 1);
            }
            continue;
        }

        // 所有检查都通过，返回安全参数
        return (safe_m, safe_tau);
    }

    // 如果所有尝试都失败，返回最保守的参数
    (2, 1)
}

/// 计算互信息以确定最优延迟时间τ
fn calculate_mutual_information(data: &Array1<f64>, tau: usize, bins: usize) -> f64 {
    let n = data.len();
    if tau >= n {
        return 0.0;
    }

    let x = data.slice(s![..n - tau]);
    let y = data.slice(s![tau..]);

    // 数据范围
    let x_min = x.fold(f64::INFINITY, |acc, &val| acc.min(val));
    let x_max = x.fold(f64::NEG_INFINITY, |acc, &val| acc.max(val));
    let y_min = y.fold(f64::INFINITY, |acc, &val| acc.min(val));
    let y_max = y.fold(f64::NEG_INFINITY, |acc, &val| acc.max(val));

    let x_range = x_max - x_min;
    let y_range = y_max - y_min;

    if x_range == 0.0 || y_range == 0.0 {
        return 0.0;
    }

    // 离散化
    let mut joint_hist = vec![vec![0; bins]; bins];
    let mut x_hist = vec![0; bins];
    let mut y_hist = vec![0; bins];

    for i in 0..x.len() {
        let x_bin = ((x[i] - x_min) / x_range * (bins as f64 - 1.0)).floor() as usize;
        let y_bin = ((y[i] - y_min) / y_range * (bins as f64 - 1.0)).floor() as usize;

        let x_bin = x_bin.min(bins - 1);
        let y_bin = y_bin.min(bins - 1);

        joint_hist[x_bin][y_bin] += 1;
        x_hist[x_bin] += 1;
        y_hist[y_bin] += 1;
    }

    // 计算互信息
    let total = x.len() as f64;
    let mut mi = 0.0;

    for i in 0..bins {
        for j in 0..bins {
            if joint_hist[i][j] > 0 && x_hist[i] > 0 && y_hist[j] > 0 {
                let joint_prob = joint_hist[i][j] as f64 / total;
                let x_prob = x_hist[i] as f64 / total;
                let y_prob = y_hist[j] as f64 / total;

                mi += joint_prob * (joint_prob / (x_prob * y_prob)).ln();
            }
        }
    }

    mi
}

/// 计算自相关函数
fn calculate_autocorrelation(data: &Array1<f64>, tau: usize) -> f64 {
    let n = data.len();
    if tau >= n {
        return 0.0;
    }

    let mean = data.mean().unwrap_or(0.0);
    let variance = data.var(0.0);

    if variance == 0.0 {
        return 1.0;
    }

    let x = data.slice(s![..n - tau]);
    let y = data.slice(s![tau..]);

    let covariance: f64 = x
        .iter()
        .zip(y.iter())
        .map(|(&xi, &yi)| (xi - mean) * (yi - mean))
        .sum::<f64>()
        / (n - tau) as f64;

    covariance / variance
}

/// 使用互信息法确定最优延迟时间τ（安全版）
fn find_optimal_tau_mutual_info(data: &Array1<f64>, max_tau: usize, bins: usize) -> usize {
    let data_len = data.len();

    // 为短序列设置极保守的tau上限
    let conservative_max_tau = if data_len < 20 {
        1 // 极短序列只用tau=1
    } else if data_len < 50 {
        ((data_len - 10) / 10).min(max_tau).max(1)
    } else if data_len < 100 {
        ((data_len - 10) / 5).min(max_tau).max(1)
    } else {
        max_tau
    };

    let mut mi_values = Vec::with_capacity(conservative_max_tau);

    for tau in 1..=conservative_max_tau {
        // 检查这个tau是否安全
        if tau >= data_len / 2 {
            break; // tau太大，停止搜索
        }

        let mi = calculate_mutual_information(data, tau, bins);
        mi_values.push(mi);
    }

    if mi_values.is_empty() {
        return 1;
    }

    // 寻找第一个局部最小值
    for i in 1..mi_values.len().saturating_sub(1) {
        if mi_values[i] < mi_values[i - 1] && mi_values[i] < mi_values[i + 1] {
            return i + 1;
        }
    }

    // 如果没有找到局部最小值，返回MI下降最快的点
    let mut best_tau = 1;
    let mut max_decrease = 0.0;

    for i in 1..mi_values.len() {
        let decrease = mi_values[i - 1] - mi_values[i];
        if decrease > max_decrease {
            max_decrease = decrease;
            best_tau = i + 1;
        }
    }

    best_tau.min(conservative_max_tau)
}

/// 使用自相关函数确定最优延迟时间τ（安全版）
fn find_optimal_tau_autocorr(data: &Array1<f64>, max_tau: usize) -> usize {
    let data_len = data.len();
    let target = 1.0 / std::f64::consts::E;

    // 为短序列设置极保守的tau上限
    let conservative_max_tau = if data_len < 20 {
        1 // 极短序列只用tau=1
    } else if data_len < 50 {
        ((data_len - 10) / 10).min(max_tau).max(1)
    } else if data_len < 100 {
        ((data_len - 10) / 5).min(max_tau).max(1)
    } else {
        max_tau
    };

    for tau in 1..=conservative_max_tau {
        // 安全检查
        if tau >= data_len / 2 {
            break;
        }

        let autocorr = calculate_autocorrelation(data, tau);
        if autocorr <= target {
            return tau;
        }
    }

    conservative_max_tau.min(3).max(1) // 确保至少返回1
}

/// 重构相空间（带安全检查）
fn reconstruct_phase_space(data: &Array1<f64>, m: usize, tau: usize) -> Array2<f64> {
    let n = data.len();

    // 安全计算，防止下溢
    let required_length = (m.saturating_sub(1)).saturating_mul(tau);

    if n <= required_length {
        // 数据不足，返回空数组
        return Array2::zeros((0, m.max(1)));
    }

    let rows = n - required_length;

    // 额外的安全检查：防止创建过大的数组
    const MAX_ELEMENTS: usize = 1_000_000; // 限制数组最大元素数
    if rows.saturating_mul(m) > MAX_ELEMENTS {
        // 数组太大，返回空数组
        return Array2::zeros((0, m.max(1)));
    }

    // 检查isize溢出
    if rows > (isize::MAX as usize) || m > (isize::MAX as usize) {
        return Array2::zeros((0, m.max(1)));
    }

    let mut phase_space = Array2::zeros((rows, m));

    for i in 0..rows {
        for j in 0..m {
            let data_idx = i + j * tau;
            if data_idx < n {
                phase_space[[i, j]] = data[data_idx];
            }
        }
    }

    phase_space
}

/// 使用简单规则确定最优嵌入维度m（快速版）
fn find_optimal_m_simple(data_len: usize, tau: usize, max_m: usize) -> usize {
    // 基于数据长度的简单规则，避免复杂计算
    let conservative_m = if data_len < 20 {
        2 // 极短序列
    } else if data_len < 50 {
        3 // 短序列
    } else if data_len < 100 {
        4 // 中等序列
    } else if data_len < 200 {
        5 // 较长序列
    } else {
        6 // 长序列
    };

    // 确保参数安全性
    let min_rows_required = 5;
    let mut safe_m = conservative_m.min(max_m).max(2);

    // 检查这个m值是否安全
    loop {
        let required_length = (safe_m.saturating_sub(1)).saturating_mul(tau);
        if data_len > required_length {
            let rows = data_len - required_length;
            if rows >= min_rows_required && rows.saturating_mul(safe_m) <= 10_000 {
                break; // 找到安全的m值
            }
        }

        if safe_m <= 2 {
            break; // 已经是最小值
        }
        safe_m -= 1;
    }

    safe_m
}

/// 超快速最近邻搜索（采样策略）
fn find_nearest_neighbors_fast(phase_space: &Array2<f64>) -> Vec<usize> {
    let n = phase_space.nrows();
    let m = phase_space.ncols();

    // 对于大的相空间，使用采样策略
    if n > 200 {
        return find_nearest_neighbors_sampled(phase_space);
    }

    let mut neighbors = vec![0; n];

    // 直接访问原始数据指针，避免边界检查
    for i in 0..n {
        let mut min_distance_sq = f64::INFINITY;
        let mut min_idx = 0;

        for j in 0..n {
            if i != j {
                let mut distance_sq = 0.0;

                // 展开距离计算，避免迭代器开销
                for k in 0..m {
                    let diff = phase_space[[i, k]] - phase_space[[j, k]];
                    distance_sq += diff * diff;
                }

                if distance_sq < min_distance_sq {
                    min_distance_sq = distance_sq;
                    min_idx = j;
                }
            }
        }

        neighbors[i] = min_idx;
    }

    neighbors
}

/// 采样策略的最近邻搜索（用于大数据）
fn find_nearest_neighbors_sampled(phase_space: &Array2<f64>) -> Vec<usize> {
    let n = phase_space.nrows();
    let m = phase_space.ncols();
    let mut neighbors = vec![0; n];

    // 对于大数据集，只搜索附近的点而不是全部
    let search_radius = (n / 10).max(20).min(100); // 搜索半径

    for i in 0..n {
        let mut min_distance_sq = f64::INFINITY;
        let mut min_idx = if i > 0 { i - 1 } else { 1 };

        // 在i附近搜索，而不是全部搜索
        let start = i.saturating_sub(search_radius);
        let end = (i + search_radius).min(n);

        for j in start..end {
            if i != j {
                let mut distance_sq = 0.0;

                for k in 0..m {
                    let diff = phase_space[[i, k]] - phase_space[[j, k]];
                    distance_sq += diff * diff;
                }

                if distance_sq < min_distance_sq {
                    min_distance_sq = distance_sq;
                    min_idx = j;
                }
            }
        }

        neighbors[i] = min_idx;
    }

    neighbors
}

/// 优化的发散率计算（减少内存分配和重复计算）
fn calculate_lyapunov_divergence_optimized(
    phase_space: &Array2<f64>,
    neighbors: &[usize],
    max_t: usize,
) -> Vec<f64> {
    let n = phase_space.nrows();
    let m = phase_space.ncols();
    let mut divergence = Vec::with_capacity(max_t);

    // 预分配缓冲区
    let mut log_distances = Vec::with_capacity(n);

    for t in 1..max_t.min(n / 2) {
        log_distances.clear(); // 重用向量，避免重新分配

        for i in 0..(n - t) {
            if i >= neighbors.len() {
                break;
            }

            let j = neighbors[i];

            if i + t < n && j + t < n {
                let mut distance_sq = 0.0;

                // 内联距离计算，避免函数调用开销
                for k in 0..m {
                    let diff = phase_space[[i + t, k]] - phase_space[[j + t, k]];
                    distance_sq += diff * diff;
                }

                if distance_sq > 1e-24 {
                    // 避免log(0)，使用平方距离阈值
                    log_distances.push(0.5 * distance_sq.ln()); // log(sqrt(x)) = 0.5*log(x)
                }
            }
        }

        if !log_distances.is_empty() {
            let sum: f64 = log_distances.iter().sum();
            let avg_log_distance = sum / log_distances.len() as f64;
            divergence.push(avg_log_distance);
        } else {
            break; // 如果没有有效距离，提前终止
        }
    }

    divergence
}

/// 线性拟合计算Lyapunov指数
fn linear_fit(x: &[f64], y: &[f64]) -> (f64, f64, f64) {
    let n = x.len() as f64;
    if n < 2.0 {
        return (0.0, 0.0, 0.0);
    }

    let sum_x: f64 = x.iter().sum();
    let sum_y: f64 = y.iter().sum();
    let sum_xy: f64 = x.iter().zip(y.iter()).map(|(&xi, &yi)| xi * yi).sum();
    let sum_x2: f64 = x.iter().map(|&xi| xi * xi).sum();

    let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);
    let intercept = (sum_y - slope * sum_x) / n;

    // 计算R²
    let y_mean = sum_y / n;
    let ss_tot: f64 = y.iter().map(|&yi| (yi - y_mean).powi(2)).sum();
    let ss_res: f64 = x
        .iter()
        .zip(y.iter())
        .map(|(&xi, &yi)| (yi - (intercept + slope * xi)).powi(2))
        .sum();

    let r_squared = if ss_tot > 0.0 {
        1.0 - ss_res / ss_tot
    } else {
        0.0
    };

    (slope, intercept, r_squared)
}

/// 辅助函数：使用调整后的参数计算结果（优化版）
fn calculate_with_adjusted_params_optimized(
    py: Python,
    normalized_data: &Array1<f64>,
    m: usize,
    tau: usize,
    max_t: usize,
    method: &str,
    original_data: &Array1<f64>,
) -> PyResult<PyObject> {
    // 相空间重构
    let phase_space = reconstruct_phase_space(normalized_data, m, tau);

    if phase_space.nrows() < 5 {
        return create_nan_result_dict(py, m, tau, method, phase_space.nrows(), original_data);
    }

    // 使用优化的最近邻搜索
    let neighbors = find_nearest_neighbors_fast(&phase_space);

    // 使用优化的发散率计算
    let divergence = calculate_lyapunov_divergence_optimized(&phase_space, &neighbors, max_t);

    if divergence.len() < 2 {
        // 如果发散序列太短，返回NaN结果
        return create_nan_result_dict(py, m, tau, method, phase_space.nrows(), original_data);
    }

    create_result_dict(py, &divergence, m, tau, method, &phase_space, original_data)
}

/// 辅助函数：创建包含NaN值的结果字典（用于数据不足的情况）
fn create_nan_result_dict(
    py: Python,
    m: usize,
    tau: usize,
    method: &str,
    phase_space_size: usize,
    original_data: &Array1<f64>,
) -> PyResult<PyObject> {
    let nan = f64::NAN;
    let empty_array = Vec::<f64>::new();

    let result = PyDict::new(py);
    result.set_item("lyapunov_exponent", nan)?;
    result.set_item("divergence_sequence", empty_array.into_pyarray(py))?;
    result.set_item("optimal_m", m)?;
    result.set_item("optimal_tau", tau)?;
    result.set_item("method_used", method)?;
    result.set_item("intercept", nan)?;
    result.set_item("r_squared", nan)?;
    result.set_item("phase_space_size", phase_space_size)?;
    result.set_item("data_length", original_data.len())?;

    Ok(result.into())
}

/// 辅助函数：创建返回结果字典
fn create_result_dict(
    py: Python,
    divergence: &[f64],
    m: usize,
    tau: usize,
    method: &str,
    phase_space: &Array2<f64>,
    original_data: &Array1<f64>,
) -> PyResult<PyObject> {
    // 线性拟合计算Lyapunov指数
    let time_steps: Vec<f64> = (1..=divergence.len()).map(|i| i as f64).collect();
    let (lyapunov_exponent, intercept, r_squared) = linear_fit(&time_steps, divergence);

    // 构造返回结果
    let result = PyDict::new(py);
    result.set_item("lyapunov_exponent", lyapunov_exponent)?;
    result.set_item("divergence_sequence", divergence.to_vec().into_pyarray(py))?;
    result.set_item("optimal_m", m)?;
    result.set_item("optimal_tau", tau)?;
    result.set_item("method_used", method)?;
    result.set_item("intercept", intercept)?;
    result.set_item("r_squared", r_squared)?;
    result.set_item("phase_space_size", phase_space.nrows())?;
    result.set_item("data_length", original_data.len())?;

    Ok(result.into())
}

/// 统一的Lyapunov指数计算函数
#[pyfunction]
#[pyo3(signature = (
    data,
    method = "auto",
    m = None,
    tau = None,
    max_t = 30,
    max_tau = 20,
    max_m = 10,
    mi_bins = 20,
    _fnn_rtol = 15.0,
    _fnn_atol = 2.0
))]
pub fn calculate_lyapunov_exponent(
    py: Python,
    data: PyReadonlyArray1<f64>,
    method: &str,
    m: Option<usize>,
    tau: Option<usize>,
    max_t: usize,
    max_tau: usize,
    max_m: usize,
    mi_bins: usize,
    _fnn_rtol: f64,
    _fnn_atol: f64,
) -> PyResult<PyObject> {
    let data_array = data.as_array().to_owned();

    // 数据标准化
    let data_min = data_array.fold(f64::INFINITY, |acc, &val| acc.min(val));
    let data_max = data_array.fold(f64::NEG_INFINITY, |acc, &val| acc.max(val));
    let data_range = data_max - data_min;

    let normalized_data = if data_range > 0.0 {
        data_array.mapv(|x| (x - data_min) / data_range)
    } else {
        data_array.clone()
    };

    // 根据method参数确定参数选择策略
    let (optimal_tau, optimal_m) = match method {
        "manual" => {
            // 手动指定参数，必须提供m和tau
            if m.is_none() || tau.is_none() {
                return Err(PyValueError::new_err("手动模式下必须指定m和tau参数"));
            }
            (tau.unwrap(), m.unwrap())
        }
        "mutual_info" => {
            // 仅使用互信息法确定tau，使用简单规则确定m
            let tau_mi = find_optimal_tau_mutual_info(&normalized_data, max_tau, mi_bins);
            let m_simple = find_optimal_m_simple(normalized_data.len(), tau_mi, max_m);
            (tau_mi, m_simple)
        }
        "autocorrelation" => {
            // 仅使用自相关法确定tau，使用简单规则确定m
            let tau_ac = find_optimal_tau_autocorr(&normalized_data, max_tau);
            let m_simple = find_optimal_m_simple(normalized_data.len(), tau_ac, max_m);
            (tau_ac, m_simple)
        }
        "auto" | _ => {
            // 自动模式：综合多种方法
            let tau_mi = find_optimal_tau_mutual_info(&normalized_data, max_tau, mi_bins);
            let tau_ac = find_optimal_tau_autocorr(&normalized_data, max_tau);
            // 取两种方法的中位数
            let optimal_tau = if tau.is_some() {
                tau.unwrap()
            } else {
                (tau_mi + tau_ac) / 2
            };

            let optimal_m = if m.is_some() {
                m.unwrap()
            } else {
                find_optimal_m_simple(normalized_data.len(), optimal_tau, max_m)
            };

            (optimal_tau, optimal_m)
        }
    };

    // 应用参数安全调整，确保不会出现数组溢出
    let (final_m, final_tau) =
        auto_adjust_parameters(normalized_data.len(), optimal_m, optimal_tau);

    // 相空间重构
    let phase_space = reconstruct_phase_space(&normalized_data, final_m, final_tau);

    if phase_space.nrows() < 5 {
        // 对于极短序列，使用最基本的参数重试一次
        let (retry_m, retry_tau) = (2, 1);
        let retry_phase_space = reconstruct_phase_space(&normalized_data, retry_m, retry_tau);

        if retry_phase_space.nrows() < 5 {
            // 数据长度确实不足，返回NaN结果而不是抛出异常
            return create_nan_result_dict(py, final_m, final_tau, method, 0, &data_array);
        }

        // 使用重试的参数继续计算
        return calculate_with_adjusted_params_optimized(
            py,
            &normalized_data,
            retry_m,
            retry_tau,
            max_t,
            method,
            &data_array,
        );
    }

    // 使用调整后的参数继续正常计算
    calculate_with_adjusted_params_optimized(
        py,
        &normalized_data,
        final_m,
        final_tau,
        max_t,
        method,
        &data_array,
    )
}
