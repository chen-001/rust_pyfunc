use ndarray::{Array2, ArrayView1};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::types::PyList;

/// 交易高峰模式分析函数
///
/// 该函数用于分析交易数据中的高峰模式，包括：
/// 1. 识别成交量的局部高峰(根据top_tier1百分比)
/// 2. 在每个高峰的时间窗口内识别小峰(根据top_tier2百分比)
/// 3. 计算17个统计指标来描述高峰-小峰的模式特征
///
/// 参数说明：
/// ----------
/// exchtime : array_like
///     交易时间数组(纳秒时间戳)
/// volume : array_like
///     成交量数组
/// flag : array_like
///     交易标志数组(主动买入/卖出标志)
/// top_tier1 : float
///     高峰识别的百分比阈值(例如0.01表示前1%的大成交量)
/// top_tier2 : float
///     小峰识别的百分比阈值(例如0.10表示前10%的大成交量)
/// time_window : float
///     时间窗口大小(秒)
/// flag_different : bool
///     是否只考虑与高峰flag不同的小峰
/// with_forth : bool
///     是否同时考虑高峰前后的时间窗口
///
/// 返回值：
/// -------
/// tuple[numpy.ndarray, list[str]]
///     第一个元素：N行17列的数组，每行对应一个局部高峰的17个统计指标
///     第二个元素：包含17个特征名称的字符串列表
///
/// Python调用示例：
/// ```python
/// import numpy as np
/// from rust_pyfunc import trade_peak_analysis
///
/// # 模拟交易数据
/// exchtime = np.array([1000000000, 2000000000, 3000000000, ...], dtype=np.int64)
/// volume = np.array([100, 200, 500, 150, 300, ...], dtype=np.float64)
/// flag = np.array([66, 83, 66, 83, 66, ...], dtype=np.int32)
///
/// # 分析高峰模式
/// result_matrix, feature_names = trade_peak_analysis(exchtime, volume, flag, 0.01, 0.10, 30.0, False, True)
///
/// # 构建DataFrame
/// import pandas as pd
/// df = pd.DataFrame(result_matrix, columns=feature_names)
/// ```
#[pyfunction]
#[pyo3(signature = (exchtime, volume, flag, top_tier1, top_tier2, time_window, flag_different, with_forth))]
pub fn trade_peak_analysis(
    py: Python,
    exchtime: PyReadonlyArray1<i64>,
    volume: PyReadonlyArray1<f64>,
    flag: PyReadonlyArray1<i32>,
    top_tier1: f64,
    top_tier2: f64,
    time_window: f64,
    flag_different: bool,
    with_forth: bool,
) -> PyResult<(Py<PyArray2<f64>>, Py<PyList>)> {
    let exchtime = exchtime.as_array();
    let volume = volume.as_array();
    let flag = flag.as_array();

    let n = exchtime.len();
    if n == 0 {
        let empty_matrix = Array2::zeros((0, 17));
        let feature_names = get_feature_names(py)?;
        return Ok((empty_matrix.into_pyarray(py).to_owned(), feature_names));
    }

    // 检查数组长度一致性
    if volume.len() != n || flag.len() != n {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "输入数组长度不一致",
        ));
    }

    // 将时间从纳秒转换为秒
    let time_window_ns = (time_window * 1_000_000_000.0) as i64;

    // 第一步：找到前top_tier1%的大成交量阈值
    let mut volume_sorted = volume.to_vec();
    volume_sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
    let tier1_threshold = volume_sorted
        [((volume_sorted.len() as f64 * top_tier1) as usize).min(volume_sorted.len() - 1)];

    // 第二步：找到前top_tier2%的大成交量阈值
    let tier2_threshold = volume_sorted
        [((volume_sorted.len() as f64 * top_tier2) as usize).min(volume_sorted.len() - 1)];

    // 第三步：识别局部高峰
    let mut peaks = Vec::new();
    for i in 0..n {
        if volume[i] >= tier1_threshold {
            // 检查是否是局部高峰(前后time_window秒内没有更大的成交量)
            let is_local_peak = is_local_peak(i, &exchtime, &volume, time_window_ns);
            if is_local_peak {
                peaks.push(i);
            }
        }
    }

    if peaks.is_empty() {
        let empty_matrix = Array2::zeros((0, 17));
        let feature_names = get_feature_names(py)?;
        return Ok((empty_matrix.into_pyarray(py).to_owned(), feature_names));
    }

    // 第四步：为每个高峰计算17个统计指标
    let mut results = Vec::with_capacity(peaks.len());

    for &peak_idx in &peaks {
        let peak_time = exchtime[peak_idx];
        let peak_volume = volume[peak_idx];
        let peak_flag = flag[peak_idx];

        // 找到时间窗口内的小峰
        let mut minor_peaks = Vec::new();
        let mut time_diffs = Vec::new();

        for i in 0..n {
            if i == peak_idx {
                continue; // 跳过高峰本身
            }

            let time_diff = (exchtime[i] - peak_time).abs();
            if time_diff <= time_window_ns {
                // 检查是否应该考虑这个点
                let should_consider = if with_forth {
                    true // 考虑前后时间窗口
                } else {
                    exchtime[i] > peak_time // 只考虑后面的时间窗口
                };

                if should_consider && volume[i] >= tier2_threshold {
                    // 检查flag_different条件
                    let flag_matches = if flag_different {
                        flag[i] != peak_flag
                    } else {
                        true
                    };

                    if flag_matches {
                        minor_peaks.push(volume[i]);
                        time_diffs.push(time_diff as f64 / 1_000_000_000.0); // 转换为秒
                    }
                }
            }
        }

        // 计算17个统计指标
        let features = calculate_features(peak_volume, &minor_peaks, &time_diffs);
        results.push(features);
    }

    // 构建结果矩阵 - N行17列（每行是一个高峰的17个特征）
    let n_peaks = results.len();
    let mut result_matrix = Array2::zeros((n_peaks, 17));

    for (row, features) in results.iter().enumerate() {
        for (col, &value) in features.iter().enumerate() {
            result_matrix[[row, col]] = value;
        }
    }

    // 获取特征名称
    let feature_names = get_feature_names(py)?;

    Ok((result_matrix.into_pyarray(py).to_owned(), feature_names))
}

/// 获取17个特征的名称
fn get_feature_names(py: Python) -> PyResult<Py<PyList>> {
    let names = vec![
        "小峰成交量总和比值",
        "小峰平均成交量比值",
        "小峰个数",
        "时间间隔均值秒",
        "成交量时间相关系数",
        "DTW距离",
        "成交量变异系数",
        "成交量偏度",
        "成交量峰度",
        "成交量趋势",
        "成交量自相关",
        "时间变异系数",
        "时间偏度",
        "时间峰度",
        "时间趋势",
        "时间自相关",
        "成交量加权时间距离",
    ];

    let py_list = PyList::new(py, names);
    Ok(py_list.into())
}

/// 检查是否是局部高峰
fn is_local_peak(
    idx: usize,
    exchtime: &ArrayView1<i64>,
    volume: &ArrayView1<f64>,
    time_window_ns: i64,
) -> bool {
    let target_time = exchtime[idx];
    let target_volume = volume[idx];
    let n = exchtime.len();

    for i in 0..n {
        if i == idx {
            continue;
        }

        let time_diff = (exchtime[i] - target_time).abs();
        if time_diff <= time_window_ns && volume[i] > target_volume {
            return false; // 找到了更大的成交量
        }
    }

    true
}

/// 计算17个统计特征
fn calculate_features(peak_volume: f64, minor_peaks: &[f64], time_diffs: &[f64]) -> Vec<f64> {
    let mut features = vec![0.0; 17];

    if minor_peaks.is_empty() {
        return features; // 如果没有小峰，返回全0
    }

    let n = minor_peaks.len();
    let minor_sum: f64 = minor_peaks.iter().sum();
    let minor_mean = minor_sum / n as f64;

    // 1. 局部小峰的成交量总和与高峰成交量的比值
    features[0] = minor_sum / peak_volume;

    // 2. 局部小峰的平均成交量与高峰成交量的比值
    features[1] = minor_mean / peak_volume;

    // 3. 局部小峰的个数
    features[2] = n as f64;

    // 4. 小峰与高峰间隔时间的均值
    let time_mean: f64 = time_diffs.iter().sum::<f64>() / n as f64;
    features[3] = time_mean;

    // 5. 小峰成交量与时间间隔的相关系数
    features[4] = correlation(minor_peaks, time_diffs);

    // 6. 小峰成交量与时间间隔的DTW距离
    features[5] = dtw_distance_simple(minor_peaks, time_diffs);

    // 7. 小峰成交量的变异系数 (std/mean)
    let minor_std = calculate_std(minor_peaks, minor_mean);
    features[6] = if minor_mean.abs() > f64::EPSILON {
        minor_std / minor_mean
    } else {
        0.0
    };

    // 8. 小峰成交量的偏度
    features[7] = calculate_skewness(minor_peaks, minor_mean, minor_std);

    // 9. 小峰成交量的峰度
    features[8] = calculate_kurtosis(minor_peaks, minor_mean, minor_std);

    // 10. 小峰成交量的趋势
    features[9] = calculate_trend(minor_peaks);

    // 11. 小峰成交量的自相关
    features[10] = calculate_autocorr(minor_peaks);

    // 12. 时间间隔的变异系数
    let time_std = calculate_std(time_diffs, time_mean);
    features[11] = if time_mean.abs() > f64::EPSILON {
        time_std / time_mean
    } else {
        0.0
    };

    // 13. 时间间隔的偏度
    features[12] = calculate_skewness(time_diffs, time_mean, time_std);

    // 14. 时间间隔的峰度
    features[13] = calculate_kurtosis(time_diffs, time_mean, time_std);

    // 15. 时间间隔的趋势
    features[14] = calculate_trend(time_diffs);

    // 16. 时间间隔的自相关
    features[15] = calculate_autocorr(time_diffs);

    // 17. 成交量加权时间距离
    features[16] = calculate_volume_weighted_time_distance(minor_peaks, time_diffs);

    features
}

/// 计算皮尔逊相关系数
fn correlation(x: &[f64], y: &[f64]) -> f64 {
    if x.len() != y.len() || x.len() < 2 {
        return 0.0;
    }

    let n = x.len() as f64;
    let x_mean = x.iter().sum::<f64>() / n;
    let y_mean = y.iter().sum::<f64>() / n;

    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;

    for i in 0..x.len() {
        let dx = x[i] - x_mean;
        let dy = y[i] - y_mean;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    if var_x.abs() < f64::EPSILON || var_y.abs() < f64::EPSILON {
        return 0.0;
    }

    cov / (var_x.sqrt() * var_y.sqrt())
}

/// 简化的DTW距离计算
fn dtw_distance_simple(s1: &[f64], s2: &[f64]) -> f64 {
    if s1.is_empty() || s2.is_empty() {
        return 0.0;
    }

    let len1 = s1.len();
    let len2 = s2.len();

    // 使用简化版本以避免内存开销过大
    if len1 > 100 || len2 > 100 {
        // 对于大数组，使用欧氏距离作为近似
        let min_len = len1.min(len2);
        let mut sum = 0.0;
        for i in 0..min_len {
            sum += (s1[i] - s2[i]).powi(2);
        }
        return sum.sqrt();
    }

    let mut dp = vec![vec![f64::INFINITY; len2]; len1];
    dp[0][0] = (s1[0] - s2[0]).abs();

    for i in 1..len1 {
        dp[i][0] = dp[i - 1][0] + (s1[i] - s2[0]).abs();
    }

    for j in 1..len2 {
        dp[0][j] = dp[0][j - 1] + (s1[0] - s2[j]).abs();
    }

    for i in 1..len1 {
        for j in 1..len2 {
            let cost = (s1[i] - s2[j]).abs();
            dp[i][j] = cost + dp[i - 1][j].min(dp[i][j - 1]).min(dp[i - 1][j - 1]);
        }
    }

    dp[len1 - 1][len2 - 1]
}

/// 计算标准差
fn calculate_std(data: &[f64], mean: f64) -> f64 {
    if data.len() < 2 {
        return 0.0;
    }

    let variance: f64 =
        data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (data.len() - 1) as f64;

    variance.sqrt()
}

/// 计算偏度
fn calculate_skewness(data: &[f64], mean: f64, std: f64) -> f64 {
    if data.len() < 3 || std.abs() < f64::EPSILON {
        return 0.0;
    }

    let n = data.len() as f64;
    let skew: f64 = data
        .iter()
        .map(|&x| ((x - mean) / std).powi(3))
        .sum::<f64>()
        / n;

    // 使用Fisher的修正公式，与scipy.stats.skew一致
    if data.len() > 2 {
        let adj_factor = (n * (n - 1.0)).sqrt() / (n - 2.0);
        skew * adj_factor
    } else {
        skew
    }
}

/// 计算峰度
fn calculate_kurtosis(data: &[f64], mean: f64, std: f64) -> f64 {
    if data.len() < 4 || std.abs() < f64::EPSILON {
        return 0.0;
    }

    let n = data.len() as f64;
    let kurt: f64 = data
        .iter()
        .map(|&x| ((x - mean) / std).powi(4))
        .sum::<f64>()
        / n;

    // 使用Fisher的修正公式，与scipy.stats.kurtosis一致
    if data.len() > 3 {
        let adj_factor =
            ((n - 1.0) / ((n - 2.0) * (n - 3.0))) * ((n + 1.0) * kurt - 3.0 * (n - 1.0));
        adj_factor
    } else {
        kurt - 3.0
    }
}

/// 计算趋势(与序列索引的相关性)
fn calculate_trend(data: &[f64]) -> f64 {
    if data.len() < 2 {
        return 0.0;
    }

    let indices: Vec<f64> = (0..data.len()).map(|i| i as f64).collect();
    correlation(data, &indices)
}

/// 计算一阶自相关
fn calculate_autocorr(data: &[f64]) -> f64 {
    if data.len() < 2 {
        return 0.0;
    }

    let x1 = &data[0..data.len() - 1];
    let x2 = &data[1..data.len()];

    correlation(x1, x2)
}

/// 计算成交量加权的时间距离
/// 用各个小峰的成交量作为权重，对时间距离进行加权平均
fn calculate_volume_weighted_time_distance(minor_peaks: &[f64], time_diffs: &[f64]) -> f64 {
    if minor_peaks.is_empty() || time_diffs.is_empty() || minor_peaks.len() != time_diffs.len() {
        return 0.0;
    }

    let total_volume: f64 = minor_peaks.iter().sum();
    if total_volume.abs() < f64::EPSILON {
        return 0.0; // 避免除零
    }

    // 计算加权平均：Σ(volume_i * time_diff_i) / Σ(volume_i)
    let weighted_sum: f64 = minor_peaks
        .iter()
        .zip(time_diffs.iter())
        .map(|(&volume, &time_diff)| volume * time_diff)
        .sum();

    weighted_sum / total_volume
}
