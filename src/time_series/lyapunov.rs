use pyo3::prelude::*;
use pyo3::types::PyDict;
use numpy::{PyArray1, PyReadonlyArray1, IntoPyArray, PyArray2};
use ndarray::{Array1, Array2, s, Axis};
use pyo3::exceptions::PyValueError;
use rayon::prelude::*;
use std::collections::HashMap;

/// 计算互信息以确定最优延迟时间τ
fn calculate_mutual_information(data: &Array1<f64>, tau: usize, bins: usize) -> f64 {
    let n = data.len();
    if tau >= n {
        return 0.0;
    }
    
    let x = data.slice(s![..n-tau]);
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
    
    let x = data.slice(s![..n-tau]);
    let y = data.slice(s![tau..]);
    
    let covariance: f64 = x.iter().zip(y.iter())
        .map(|(&xi, &yi)| (xi - mean) * (yi - mean))
        .sum::<f64>() / (n - tau) as f64;
    
    covariance / variance
}

/// 使用互信息法确定最优延迟时间τ
fn find_optimal_tau_mutual_info(data: &Array1<f64>, max_tau: usize, bins: usize) -> usize {
    let mut mi_values = Vec::with_capacity(max_tau);
    
    for tau in 1..=max_tau {
        let mi = calculate_mutual_information(data, tau, bins);
        mi_values.push(mi);
    }
    
    // 寻找第一个局部最小值
    for i in 1..mi_values.len()-1 {
        if mi_values[i] < mi_values[i-1] && mi_values[i] < mi_values[i+1] {
            return i + 1;
        }
    }
    
    // 如果没有找到局部最小值，返回MI下降最快的点
    let mut best_tau = 1;
    let mut max_decrease = 0.0;
    
    for i in 1..mi_values.len() {
        let decrease = mi_values[i-1] - mi_values[i];
        if decrease > max_decrease {
            max_decrease = decrease;
            best_tau = i + 1;
        }
    }
    
    best_tau
}

/// 使用自相关函数确定最优延迟时间τ
fn find_optimal_tau_autocorr(data: &Array1<f64>, max_tau: usize) -> usize {
    let target = 1.0 / std::f64::consts::E;
    
    for tau in 1..=max_tau {
        let autocorr = calculate_autocorrelation(data, tau);
        if autocorr <= target {
            return tau;
        }
    }
    
    max_tau.min(3) // 默认返回3
}

/// 重构相空间
fn reconstruct_phase_space(data: &Array1<f64>, m: usize, tau: usize) -> Array2<f64> {
    let n = data.len();
    let rows = n - (m - 1) * tau;
    
    if rows <= 0 {
        return Array2::zeros((0, m));
    }
    
    let mut phase_space = Array2::zeros((rows, m));
    
    for i in 0..rows {
        for j in 0..m {
            phase_space[[i, j]] = data[i + j * tau];
        }
    }
    
    phase_space
}

/// 计算欧几里得距离
fn euclidean_distance(x: &Array1<f64>, y: &Array1<f64>) -> f64 {
    x.iter().zip(y.iter())
        .map(|(&xi, &yi)| (xi - yi).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// 计算距离矩阵并找到最近邻
fn find_nearest_neighbors(phase_space: &Array2<f64>) -> Vec<usize> {
    let n = phase_space.nrows();
    let mut neighbors = vec![0; n];
    
    // 使用并行计算加速
    neighbors.par_iter_mut().enumerate().for_each(|(i, neighbor)| {
        let mut min_distance = f64::INFINITY;
        let mut min_idx = 0;
        
        for j in 0..n {
            if i != j {
                let row_i = phase_space.row(i).to_owned();
                let row_j = phase_space.row(j).to_owned();
                let distance = euclidean_distance(&row_i, &row_j);
                
                if distance < min_distance {
                    min_distance = distance;
                    min_idx = j;
                }
            }
        }
        
        *neighbor = min_idx;
    });
    
    neighbors
}

/// 计算假最近邻比例以确定最优嵌入维度m
fn calculate_false_nearest_neighbors(
    data: &Array1<f64>, 
    m: usize, 
    tau: usize, 
    rtol: f64, 
    atol: f64
) -> f64 {
    let phase_space_m = reconstruct_phase_space(data, m, tau);
    let n = phase_space_m.nrows();
    
    if n < 10 {
        return 100.0; // 数据点太少
    }
    
    // 构造m+1维相空间
    let phase_space_m_plus = reconstruct_phase_space(data, m + 1, tau);
    let n_plus = phase_space_m_plus.nrows();
    
    if n_plus != n {
        return 100.0;
    }
    
    let neighbors = find_nearest_neighbors(&phase_space_m);
    
    let mut false_neighbors = 0;
    let mut total_neighbors = 0;
    
    for i in 0..n {
        let neighbor_idx = neighbors[i];
        
        if neighbor_idx >= n_plus {
            continue;
        }
        
        // 计算在m维空间中的距离
        let row_i_m = phase_space_m.row(i).to_owned();
        let row_neighbor_m = phase_space_m.row(neighbor_idx).to_owned();
        let dist_m = euclidean_distance(&row_i_m, &row_neighbor_m);
        
        if dist_m == 0.0 {
            continue;
        }
        
        // 计算在m+1维空间中的距离
        let row_i_m_plus = phase_space_m_plus.row(i).to_owned();
        let row_neighbor_m_plus = phase_space_m_plus.row(neighbor_idx).to_owned();
        let dist_m_plus = euclidean_distance(&row_i_m_plus, &row_neighbor_m_plus);
        
        // 判断是否为假最近邻
        let ratio = (dist_m_plus - dist_m).abs() / dist_m;
        
        if ratio > rtol / 100.0 || dist_m_plus > atol {
            false_neighbors += 1;
        }
        
        total_neighbors += 1;
    }
    
    if total_neighbors == 0 {
        100.0
    } else {
        (false_neighbors as f64 / total_neighbors as f64) * 100.0
    }
}

/// 使用假最近邻法确定最优嵌入维度m
fn find_optimal_m_fnn(
    data: &Array1<f64>, 
    tau: usize, 
    max_m: usize, 
    rtol: f64, 
    atol: f64
) -> usize {
    let threshold = 1.0; // 1%阈值
    
    for m in 2..=max_m {
        let fnn_percentage = calculate_false_nearest_neighbors(data, m, tau, rtol, atol);
        if fnn_percentage < threshold {
            return m;
        }
    }
    
    max_m
}

/// 计算Lyapunov指数的发散率序列
fn calculate_lyapunov_divergence(
    phase_space: &Array2<f64>, 
    neighbors: &[usize], 
    max_t: usize
) -> Vec<f64> {
    let n = phase_space.nrows();
    let mut divergence = Vec::new();
    
    for t in 1..max_t {
        let mut log_distances = Vec::new();
        
        for i in 0..(n - t) {
            let j = neighbors[i];
            
            if i + t < n && j + t < n {
                let row_i = phase_space.row(i + t).to_owned();
                let row_j = phase_space.row(j + t).to_owned();
                let distance = euclidean_distance(&row_i, &row_j);
                
                if distance > 0.0 {
                    log_distances.push(distance.ln());
                }
            }
        }
        
        if !log_distances.is_empty() {
            let avg_log_distance: f64 = log_distances.iter().sum::<f64>() / log_distances.len() as f64;
            divergence.push(avg_log_distance);
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
    let ss_res: f64 = x.iter().zip(y.iter())
        .map(|(&xi, &yi)| (yi - (intercept + slope * xi)).powi(2))
        .sum();
    
    let r_squared = if ss_tot > 0.0 { 1.0 - ss_res / ss_tot } else { 0.0 };
    
    (slope, intercept, r_squared)
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
    fnn_rtol = 15.0,
    fnn_atol = 2.0
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
    fnn_rtol: f64,
    fnn_atol: f64,
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
        },
        "mutual_info" => {
            // 仅使用互信息法确定tau，使用假最近邻法确定m
            let tau_mi = find_optimal_tau_mutual_info(&normalized_data, max_tau, mi_bins);
            let m_fnn = find_optimal_m_fnn(&normalized_data, tau_mi, max_m, fnn_rtol, fnn_atol);
            (tau_mi, m_fnn)
        },
        "autocorrelation" => {
            // 仅使用自相关法确定tau，使用假最近邻法确定m
            let tau_ac = find_optimal_tau_autocorr(&normalized_data, max_tau);
            let m_fnn = find_optimal_m_fnn(&normalized_data, tau_ac, max_m, fnn_rtol, fnn_atol);
            (tau_ac, m_fnn)
        },
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
                find_optimal_m_fnn(&normalized_data, optimal_tau, max_m, fnn_rtol, fnn_atol)
            };
            
            (optimal_tau, optimal_m)
        }
    };
    
    // 相空间重构
    let phase_space = reconstruct_phase_space(&normalized_data, optimal_m, optimal_tau);
    
    if phase_space.nrows() < 10 {
        return Err(PyValueError::new_err("数据长度不足以进行相空间重构"));
    }
    
    // 找到最近邻
    let neighbors = find_nearest_neighbors(&phase_space);
    
    // 计算发散率序列
    let divergence = calculate_lyapunov_divergence(&phase_space, &neighbors, max_t);
    
    if divergence.len() < 3 {
        return Err(PyValueError::new_err("发散率序列长度不足以进行线性拟合"));
    }
    
    // 线性拟合计算Lyapunov指数
    let time_steps: Vec<f64> = (1..=divergence.len()).map(|i| i as f64).collect();
    let (lyapunov_exponent, intercept, r_squared) = linear_fit(&time_steps, &divergence);
    
    // 构造返回结果
    let result = PyDict::new(py);
    result.set_item("lyapunov_exponent", lyapunov_exponent)?;
    result.set_item("divergence_sequence", divergence.into_pyarray(py))?;
    result.set_item("optimal_m", optimal_m)?;
    result.set_item("optimal_tau", optimal_tau)?;
    result.set_item("method_used", method)?;
    result.set_item("intercept", intercept)?;
    result.set_item("r_squared", r_squared)?;
    result.set_item("phase_space_size", phase_space.nrows())?;
    result.set_item("data_length", data_array.len())?;
    
    Ok(result.into())
}

