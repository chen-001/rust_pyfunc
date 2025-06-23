use pyo3::prelude::*;
// use pyo3::types::PyList;
use numpy::{PyReadonlyArray2, PyArray2};
use ndarray::{Array2, Axis};
use std::collections::{HashMap, HashSet, VecDeque};
use pyo3::types::{PyList, PyString};
// use std::collections::BTreeMap;

/// 计算时间序列在指定时间窗口内向后滚动的统计量。
/// 对于每个时间点，计算该点之后指定时间窗口内所有数据的指定统计量。
///
/// 参数说明：
/// ----------
/// times : array_like
///     时间戳数组（单位：秒）
/// values : array_like
///     数值数组
/// window : float
///     时间窗口大小（单位：秒）
/// stat_type : str
///     统计量类型，可选值：
///     - "mean": 均值
///     - "sum": 总和
///     - "max": 最大值
///     - "min": 最小值
///     - "last": 时间窗口内最后一个值
///     - "std": 标准差
///     - "median": 中位数
///     - "count": 数据点数量
///     - "rank": 分位数（0到1之间）
///     - "skew": 偏度
///     - "trend_time": 与时间序列的相关系数
///     - "last": 时间窗口内最后一个值
///     - "trend_oneton": 与1到n序列的相关系数（时间间隔）
/// * `include_current` - 是否包含当前时间点的值
///
/// 返回值：
/// -------
/// numpy.ndarray
///     计算得到的向后滚动统计量数组
///
/// Python调用示例：
/// ```python
/// import numpy as np
/// from rust_pyfunc import rolling_window_stat
///
/// # 创建示例数据
/// times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
/// values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
/// window = 2.0  # 2秒的时间窗口
///
/// # 计算向后滚动均值
/// mean_result = rolling_window_stat(times, values, window, "mean")
/// ```
#[pyfunction]
pub fn rolling_window_stat(times: Vec<f64>, values: Vec<f64>, window: f64, stat_type: &str, include_current: bool) -> Vec<f64> {
    let n = times.len();
    if n == 0 {
        return vec![];
    }
    
    let window_ns = window;
    let mut result = vec![f64::NAN; n];
    
    match stat_type {
        "mean" | "sum" => {
            let mut window_sum = 0.0;
            let mut window_count = 0;
            let mut window_start = 0;
            let mut window_end = 0;
            
            for i in 0..n {
                // 移除窗口前面的值
                while window_start < i {
                    window_sum -= values[window_start];
                    window_count -= 1;
                    window_start += 1;
                }
                
                // 添加新的值到窗口
                while window_end < n && times[window_end] - times[i] <= window_ns {
                    window_sum += values[window_end];
                    window_count += 1;
                    window_end += 1;
                }
                
                // 计算结果
                if window_count > 0 {
                    if !include_current {
                        // 如果不包含当前值，减去当前值的影响
                        let current_value = values[i];
                        if stat_type == "mean" {
                            result[i] = if window_count == 1 {
                                f64::NAN
                            } else {
                                (window_sum - current_value) / (window_count - 1) as f64
                            };
                        } else {
                            result[i] = window_sum - current_value;
                        }
                    } else {
                        // 包含当前值的情况
                        if stat_type == "mean" {
                            result[i] = window_sum / window_count as f64;
                        } else {
                            result[i] = window_sum;
                        }
                    }
                }
            }
        },
        "max" | "min" => {
            // 使用单调队列优化
            let mut deque: VecDeque<usize> = VecDeque::with_capacity(n);
            let mut window_end = 0;
            
            for i in 0..n {
                // 复用上一个窗口的结束位置
                if i > 0 && window_end > i {
                    // 移除不在当前窗口的值
                    while !deque.is_empty() && deque[0] < (if include_current { i } else { i + 1 }) {
                        deque.pop_front();
                    }
                } else {
                    window_end = if include_current { i } else { i + 1 };
                    deque.clear();
                }
                
                // 扩展窗口直到超出时间范围
                while window_end < n && times[window_end] - times[i] <= window_ns {
                    // 维护单调队列
                    if window_end >= (if include_current { i } else { i + 1 }) {
                        while !deque.is_empty() && {
                            let last = *deque.back().unwrap();
                            if stat_type == "max" {
                                values[last] <= values[window_end]
                            } else {
                                values[last] >= values[window_end]
                            }
                        } {
                            deque.pop_back();
                        }
                        deque.push_back(window_end);
                    }
                    window_end += 1;
                }
                
                // 计算结果
                if !deque.is_empty() {
                    result[i] = values[deque[0]];
                }
            }
        },
        "std" => {
            if include_current {
                // 保持原有逻辑不变
                let mut window_sum = 0.0;
                let mut window_sum_sq = 0.0;
                let mut count = 0;
                let mut window_end = 0;
                let mut window_start = 0;
                
                for i in 0..n {
                    while window_start < i {
                        if window_start < window_end {
                            window_sum -= values[window_start];
                            window_sum_sq -= values[window_start] * values[window_start];
                            count -= 1;
                        }
                        window_start += 1;
                    }
                    
                    while window_end < n && times[window_end] - times[i] <= window_ns {
                        window_sum += values[window_end];
                        window_sum_sq += values[window_end] * values[window_end];
                        count += 1;
                        window_end += 1;
                    }
                    
                    if count > 1 {
                        let mean = window_sum / count as f64;
                        let variance = (window_sum_sq - window_sum * mean) / (count - 1) as f64;
                        if variance > 0.0 {
                            result[i] = variance.sqrt();
                        }
                    }
                }
            } else {
                let mut window_sum = 0.0;
                let mut window_sum_sq = 0.0;
                let mut count = 0;
                let mut window_end = 1;  // 从1开始，因为不包含当前值
                
                for i in 0..n {
                    // 移除过期的值（如果window_end落后，则会在下一步重新计算）
                    if i > 0 && window_end > i {
                        window_sum -= values[i];
                        window_sum_sq -= values[i] * values[i];
                        count -= 1;
                    }
                    
                    // 添加新的值到窗口
                    while window_end < n && times[window_end] - times[i] <= window_ns {
                        window_sum += values[window_end];
                        window_sum_sq += values[window_end] * values[window_end];
                        count += 1;
                        window_end += 1;
                    }
                    
                    if count > 1 {
                        let mean = window_sum / count as f64;
                        let variance = (window_sum_sq - window_sum * mean) / (count - 1) as f64;
                        if variance > 0.0 {
                            result[i] = variance.sqrt();
                        }
                    }
                }
            }
        },
        "median" => {
            if include_current {
                // 保持原有逻辑不变
                let mut window_values: Vec<f64> = Vec::with_capacity(n);
                let mut window_end = 0;
                let mut window_start = 0;
                
                for i in 0..n {
                    // 移除窗口前面的值
                    while window_start < i {
                        if window_start < window_end {
                            if let Ok(pos) = window_values.binary_search_by(|x| x.partial_cmp(&values[window_start]).unwrap_or(std::cmp::Ordering::Equal)) {
                                window_values.remove(pos);
                            }
                        }
                        window_start += 1;
                    }
                    
                    // 添加新的值到窗口
                    while window_end < n && times[window_end] - times[i] <= window_ns {
                        let val = values[window_end];
                        match window_values.binary_search_by(|x| x.partial_cmp(&val).unwrap_or(std::cmp::Ordering::Equal)) {
                            Ok(pos) | Err(pos) => window_values.insert(pos, val),
                        }
                        window_end += 1;
                    }
                    
                    // 计算中位数
                    if !window_values.is_empty() {
                        let len = window_values.len();
                        if len % 2 == 0 {
                            result[i] = (window_values[len/2 - 1] + window_values[len/2]) / 2.0;
                        } else {
                            result[i] = window_values[len/2];
                        }
                    }
                }
            } else {
                let mut window_values: Vec<f64> = Vec::with_capacity(n);
                let mut window_end = 1;  // 从1开始，因为不包含当前值
                let mut window_start = 0;
                
                for i in 0..n {
                    // 如果window_end落后了，重置它
                    if window_end <= i + 1 {
                        window_end = i + 1;
                        window_values.clear();  // 重置窗口
                        
                        // 重新填充窗口
                        while window_end < n && times[window_end] - times[i] <= window_ns {
                            let val = values[window_end];
                            match window_values.binary_search_by(|x| x.partial_cmp(&val).unwrap_or(std::cmp::Ordering::Equal)) {
                                Ok(pos) | Err(pos) => window_values.insert(pos, val),
                            }
                            window_end += 1;
                        }
                    } else {
                        // 移除超出时间窗口的值
                        while window_end < n && times[window_end] - times[i] <= window_ns {
                            let val = values[window_end];
                            match window_values.binary_search_by(|x| x.partial_cmp(&val).unwrap_or(std::cmp::Ordering::Equal)) {
                                Ok(pos) | Err(pos) => window_values.insert(pos, val),
                            }
                            window_end += 1;
                        }
                        
                        // 移除窗口前面的值（包括当前值i）
                        while window_start <= i {
                            if let Ok(pos) = window_values.binary_search_by(|x| x.partial_cmp(&values[window_start]).unwrap_or(std::cmp::Ordering::Equal)) {
                                window_values.remove(pos);
                            }
                            window_start += 1;
                        }
                    }
                    
                    // 计算中位数
                    if !window_values.is_empty() {
                        let len = window_values.len();
                        if len % 2 == 0 {
                            result[i] = (window_values[len/2 - 1] + window_values[len/2]) / 2.0;
                        } else {
                            result[i] = window_values[len/2];
                        }
                    }
                }
            }
        },
        "count" => {
            let mut window_end = 0;
            let mut count;
            
            for i in 0..n {
                // 复用上一个窗口的结束位置，如果可能的话
                if i > 0 && window_end > i {
                    // 调整count以反映新窗口的起始位置
                    count = window_end - i;  // 更新count为当前窗口内���元素数量
                } else {
                    // 重新寻找窗口结束位置和计数
                    window_end = i;
                    count = 0;
                }
                
                // 扩展窗口直到超出时间范围
                while window_end < n && times[window_end] - times[i] <= window_ns {
                    count += 1;
                    window_end += 1;
                }
                
                if count > 0 {
                    if !include_current {
                        result[i] = (count - 1) as f64;  // 不包含当前值时减1
                    } else {
                        result[i] = count as f64;  // 包含当前值时用完整计数
                    }
                }
            }
        },
        "rank" => {
            // 对于 rank 统计，忽略 include_current 参数，始终包含当前值
            let mut window_values: Vec<(f64, usize)> = Vec::with_capacity(n);
            let mut window_end = 0;
            let mut window_start = 0;
            
            for i in 0..n {
                // 移除窗口前面的值
                while window_start < i {
                    if window_start < window_end {
                        if let Ok(pos) = window_values.binary_search_by(|(x, _)| x.partial_cmp(&values[window_start]).unwrap_or(std::cmp::Ordering::Equal)) {
                            window_values.remove(pos);
                        }
                    }
                    window_start += 1;
                }
                
                // 添加新的值到窗口
                while window_end < n && times[window_end] - times[i] <= window_ns {
                    let val = values[window_end];
                    match window_values.binary_search_by(|(x, _)| x.partial_cmp(&val).unwrap_or(std::cmp::Ordering::Equal)) {
                        Ok(pos) | Err(pos) => window_values.insert(pos, (val, window_end)),
                    }
                    window_end += 1;
                }
                
                // 计算排名
                if window_values.len() > 1 {
                    let current_value = values[i];
                    let window_len = window_values.len();
                    
                    // 使用二分查找找到当前值的位置
                    match window_values.binary_search_by(|(x, _)| x.partial_cmp(&current_value).unwrap_or(std::cmp::Ordering::Equal)) {
                        Ok(pos) => {
                            // 处理相等值
                            let mut equal_start = pos;
                            while equal_start > 0 && (window_values[equal_start - 1].0 - current_value).abs() < 1e-10 {
                                equal_start -= 1;
                            }
                            let mut equal_end = pos;
                            while equal_end < window_len - 1 && (window_values[equal_end + 1].0 - current_value).abs() < 1e-10 {
                                equal_end += 1;
                            }
                            let rank = (equal_start + equal_end) as f64 / 2.0;
                            result[i] = rank / (window_len - 1) as f64;
                        },
                        Err(pos) => {
                            result[i] = pos as f64 / (window_len - 1) as f64;
                        }
                    }
                }
            }
        },
        "skew" => {
            if include_current {
                let mut window_end = 0;
                let mut sum = 0.0;
                let mut sum_sq = 0.0;
                let mut sum_cube = 0.0;
                let mut count = 0;
                let mut last_start = 0;
                
                for i in 0..n {
                    // 快速移除窗口前面的值
                    if i > last_start {
                        let remove_count = i - last_start;
                        for j in last_start..i {
                            sum -= values[j];
                            sum_sq -= values[j] * values[j];
                            sum_cube -= values[j] * values[j] * values[j];
                        }
                        count -= remove_count;
                        last_start = i;
                    }
                    
                    // 快速添加新值
                    let target_time = times[i] + window_ns;
                    while window_end < n && times[window_end] <= target_time {
                        let val = values[window_end];
                        let val_sq = val * val;
                        sum += val;
                        sum_sq += val_sq;
                        sum_cube += val_sq * val;
                        count += 1;
                        window_end += 1;
                    }
                    
                    // 计算偏度
                    if count > 2 {
                        let n = count as f64;
                        let mean = sum / n;
                        let mean_sq = mean * mean;
                        
                        let m2 = sum_sq - 2.0 * mean * sum + n * mean_sq;
                        let m3 = sum_cube - 3.0 * mean * sum_sq + 3.0 * mean_sq * sum - n * mean_sq * mean;
                        
                        let variance = m2 / n;
                        if variance > 0.0 {
                            let std_dev = variance.sqrt();
                            result[i] = (m3 / n) / (std_dev * std_dev * std_dev);
                        }
                    }
                }
            } else {
                let mut window_end = 1;
                let mut sum = 0.0;
                let mut sum_sq = 0.0;
                let mut sum_cube = 0.0;
                let mut count = 0;
                
                for i in 0..n {
                    // 重置窗口，如果需要
                    if window_end <= i + 1 {
                        window_end = i + 1;
                        sum = 0.0;
                        sum_sq = 0.0;
                        sum_cube = 0.0;
                        count = 0;
                    } else {
                        // 移除当前值
                        let val = values[i];
                        let val_sq = val * val;
                        sum -= val;
                        sum_sq -= val_sq;
                        sum_cube -= val_sq * val;
                        count -= 1;
                    }
                    
                    // 快速添加新值
                    let target_time = times[i] + window_ns;
                    while window_end < n && times[window_end] <= target_time {
                        let val = values[window_end];
                        let val_sq = val * val;
                        sum += val;
                        sum_sq += val_sq;
                        sum_cube += val_sq * val;
                        count += 1;
                        window_end += 1;
                    }
                    
                    // 计算偏度
                    if count > 2 {
                        let n = count as f64;
                        let mean = sum / n;
                        let mean_sq = mean * mean;
                        
                        let m2 = sum_sq - 2.0 * mean * sum + n * mean_sq;
                        let m3 = sum_cube - 3.0 * mean * sum_sq + 3.0 * mean_sq * sum - n * mean_sq * mean;
                        
                        let variance = m2 / n;
                        if variance > 0.0 {
                            let std_dev = variance.sqrt();
                            result[i] = (m3 / n) / (std_dev * std_dev * std_dev);
                        }
                    }
                }
            }
        },
        "trend_time" => {
            if include_current {
                let mut window_sum_y = 0.0;
                let mut window_sum_x = 0.0;
                let mut window_sum_xy = 0.0;
                let mut window_sum_xx = 0.0;
                let mut window_sum_yy = 0.0;
                let mut count = 0;
                let mut window_end = 0;
                let mut window_start = 0;
                
                for i in 0..n {
                    // 移除窗口前面的值
                    while window_start < i {
                        if window_start < window_end {
                            let y = values[window_start];
                            let x = times[window_start];
                            window_sum_y -= y;
                            window_sum_x -= x;
                            window_sum_xy -= x * y;
                            window_sum_xx -= x * x;
                            window_sum_yy -= y * y;
                            count -= 1;
                        }
                        window_start += 1;
                    }
                    
                    // 扩展窗口直到超出时间范围
                    while window_end < n && times[window_end] - times[i] <= window_ns {
                        let y = values[window_end];
                        let x = times[window_end];
                        window_sum_y += y;
                        window_sum_x += x;
                        window_sum_xy += x * y;
                        window_sum_xx += x * x;
                        window_sum_yy += y * y;
                        count += 1;
                        window_end += 1;
                    }
                    
                    // 计算相关系数
                    if count > 1 {
                        let n = count as f64;
                        let cov = window_sum_xy - window_sum_x * window_sum_y / n;
                        let var_x = window_sum_xx - window_sum_x * window_sum_x / n;
                        let var_y = window_sum_yy - window_sum_y * window_sum_y / n;
                        
                        if var_x > 0.0 && var_y > 0.0 {
                            result[i] = cov / (var_x.sqrt() * var_y.sqrt());
                        }
                    }
                }
            } else {
                let mut window_sum_y = 0.0;
                let mut window_sum_x = 0.0;
                let mut window_sum_xy = 0.0;
                let mut window_sum_xx = 0.0;
                let mut window_sum_yy = 0.0;
                let mut count = 0;
                let mut window_end = 1;  // 从1开始，因为不包含当前值
                
                for i in 0..n {
                    // 重置窗口统计值，如果window_end落后了
                    if window_end <= i + 1 {
                        window_end = i + 1;
                        window_sum_y = 0.0;
                        window_sum_x = 0.0;
                        window_sum_xy = 0.0;
                        window_sum_xx = 0.0;
                        window_sum_yy = 0.0;
                        count = 0;
                    } else {
                        // 移除当前值i
                        let y = values[i];
                        let x = times[i];
                        window_sum_y -= y;
                        window_sum_x -= x;
                        window_sum_xy -= x * y;
                        window_sum_xx -= x * x;
                        window_sum_yy -= y * y;
                        count -= 1;
                    }
                    
                    // 扩展窗口直到超出时间范围
                    while window_end < n && times[window_end] - times[i] <= window_ns {
                        let y = values[window_end];
                        let x = times[window_end];
                        window_sum_y += y;
                        window_sum_x += x;
                        window_sum_xy += x * y;
                        window_sum_xx += x * x;
                        window_sum_yy += y * y;
                        count += 1;
                        window_end += 1;
                    }
                    
                    // 计算相关系数
                    if count > 1 {
                        let n = count as f64;
                        let cov = window_sum_xy - window_sum_x * window_sum_y / n;
                        let var_x = window_sum_xx - window_sum_x * window_sum_x / n;
                        let var_y = window_sum_yy - window_sum_y * window_sum_y / n;
                        
                        if var_x > 0.0 && var_y > 0.0 {
                            result[i] = cov / (var_x.sqrt() * var_y.sqrt());
                        }
                    }
                }
            }
        },
        "trend_oneton" => {
            let mut result = vec![f64::NAN; n];
            let mut window_end = 0;

            for i in 0..n {
                let mut window_sum_y = 0.0;
                let mut window_sum_yy = 0.0;
                let mut window_sum_xy = 0.0;
                let mut count = 0;

                // 扩展窗口到超出时间范围
                while window_end < n && times[window_end] - times[i] <= window_ns {
                    let y = values[window_end];
                    let x = (count + 1) as f64; // 正整数序列 1, 2, 3, ...

                    window_sum_y += y;
                    window_sum_yy += y * y;
                    window_sum_xy += x * y;
                    count += 1;
                    window_end += 1;
                }

                // 计算相关系数
                if count > 1 {
                    let n = count as f64;
                    let mean_y = window_sum_y / n;
                    let mean_x = (n + 1.0) / 2.0; // 正整数序列的均值

                    let cov = window_sum_xy - n * mean_x * mean_y;
                    let var_y = window_sum_yy - n * mean_y * mean_y;
                    let var_x = (n * n - 1.0) / 12.0; // 正整数序列的方差

                    if var_x > 0.0 && var_y > 0.0 {
                        result[i] = cov / (var_x.sqrt() * var_y.sqrt());
                    }
                }
            }
        },
        "last" => {
            let mut window_end = 0;
            
            for i in 0..n {
                // 复用上一个窗口的结束位置，如果可能的话
                if i > 0 && window_end > i {
                    // 继续使用之前的window_end
                } else {
                    // 重新寻找窗口结束位置
                    window_end = if include_current { i } else { i + 1 };
                }
                
                // 扩展窗口直到超出时间范围
                while window_end < n && times[window_end] - times[i] <= window_ns {
                    window_end += 1;
                }
                
                // 如果找到了有效值，取最后一个
                if window_end > (if include_current { i } else { i + 1 }) {
                    result[i] = values[window_end - 1];
                }
            }
        },
        _ => panic!("不支持的统计类型: {}", stat_type),
    }
    
    result
}

/// 高性能的DataFrame rank函数，支持axis=1（沿行方向排名）
/// 相比pandas的rank函数能显著提升性能
/// 
/// 参数说明：
/// ----------
/// data : numpy.ndarray (2D, f64)
///     输入的2D数组
/// method : str
///     排名方法，支持：
///     - "average": 并列值取平均排名（默认）
///     - "min": 并列值取最小排名
///     - "max": 并列值取最大排名
///     - "first": 按出现顺序排名
///     - "dense": 密集排名（不跳号）
/// ascending : bool
///     是否升序排名，True为升序（默认），False为降序
/// na_option : str
///     NaN处理方式：
///     - "keep": 保持NaN为NaN（默认）
///     - "top": NaN值排在最前
///     - "bottom": NaN值排在最后
/// 
/// 返回值：
/// -------
/// numpy.ndarray (2D, f64)
///     排名结果数组，形状与输入相同
/// 
/// Python调用示例：
/// ```python
/// import numpy as np
/// from rust_pyfunc import rank_axis1
/// 
/// # 创建测试数据
/// data = np.array([[3.0, 1.0, 4.0, 2.0],
///                  [2.0, 4.0, 1.0, 3.0]], dtype=np.float64)
/// 
/// # 沿行方向排名
/// ranks = rank_axis1(data, method="average", ascending=True, na_option="keep")
/// print(ranks)
/// # 输出: [[3.0, 1.0, 4.0, 2.0],
/// #        [2.0, 4.0, 1.0, 3.0]]
/// ```
#[pyfunction]
#[pyo3(signature = (data, method="average", ascending=true, na_option="keep"))]
pub fn rank_axis1(
    py: Python,
    data: PyReadonlyArray2<f64>,
    method: &str,
    ascending: bool,
    na_option: &str,
) -> PyResult<PyObject> {
    let data_array = data.as_array();
    let (nrows, ncols) = data_array.dim();
    
    // 创建结果数组
    let mut result = Array2::<f64>::zeros((nrows, ncols));
    
    // 对每一行进行排名
    for row_idx in 0..nrows {
        let row_data = data_array.row(row_idx);
        let row_ranks = rank_1d_array(row_data.to_vec(), method, ascending, na_option);
        
        // 将结果写入对应行
        for (col_idx, &rank_val) in row_ranks.iter().enumerate() {
            result[[row_idx, col_idx]] = rank_val;
        }
    }
    
    // 转换为Python对象返回
    let result_array = PyArray2::from_owned_array(py, result);
    Ok(result_array.to_object(py))
}

/// 对1D数组进行排名的核心函数
fn rank_1d_array(data: Vec<f64>, method: &str, ascending: bool, na_option: &str) -> Vec<f64> {
    let n = data.len();
    if n == 0 {
        return vec![];
    }
    
    // 创建索引-值对，用于排序
    let indexed_data: Vec<(usize, f64)> = data.iter()
        .enumerate()
        .map(|(i, &val)| (i, val))
        .collect();
    
    // 分离NaN和非NaN值
    let mut nan_indices = vec![];
    let mut valid_data = vec![];
    
    for (i, val) in indexed_data.iter() {
        if val.is_nan() {
            nan_indices.push(*i);
        } else {
            valid_data.push((*i, *val));
        }
    }
    
    // 对非NaN值进行排序
    if ascending {
        valid_data.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    } else {
        valid_data.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    }
    
    // 初始化结果数组
    let mut result = vec![f64::NAN; n];
    
    // 处理NaN值
    match na_option {
        "keep" => {
            // NaN保持为NaN，已经在初始化时设置
        },
        "top" => {
            if !nan_indices.is_empty() {
                match method {
                    "average" => {
                        let avg_rank = (1 + nan_indices.len()) as f64 / 2.0;
                        for &idx in nan_indices.iter() {
                            result[idx] = avg_rank;
                        }
                    },
                    "min" => {
                        for &idx in nan_indices.iter() {
                            result[idx] = 1.0;
                        }
                    },
                    "max" => {
                        let max_rank = nan_indices.len() as f64;
                        for &idx in nan_indices.iter() {
                            result[idx] = max_rank;
                        }
                    },
                    "first" => {
                        for (rank, &idx) in nan_indices.iter().enumerate() {
                            result[idx] = (rank + 1) as f64;
                        }
                    },
                    "dense" => {
                        for &idx in nan_indices.iter() {
                            result[idx] = 1.0;
                        }
                    },
                    _ => panic!("不支持的method: {}", method),
                }
            }
        },
        "bottom" => {
            if !nan_indices.is_empty() {
                let nan_start_rank = valid_data.len() + 1;
                match method {
                    "average" => {
                        let avg_rank = (nan_start_rank as f64 + (nan_start_rank + nan_indices.len() - 1) as f64) / 2.0;
                        for &idx in nan_indices.iter() {
                            result[idx] = avg_rank;
                        }
                    },
                    "min" => {
                        for &idx in nan_indices.iter() {
                            result[idx] = nan_start_rank as f64;
                        }
                    },
                    "max" => {
                        let max_rank = (nan_start_rank + nan_indices.len() - 1) as f64;
                        for &idx in nan_indices.iter() {
                            result[idx] = max_rank;
                        }
                    },
                    "first" => {
                        for (rank, &idx) in nan_indices.iter().enumerate() {
                            result[idx] = (nan_start_rank + rank) as f64;
                        }
                    },
                    "dense" => {
                        let dense_rank = if valid_data.is_empty() { 1.0 } else {
                            // 需要计算非NaN值中有多少个不同的值
                            let mut unique_values = valid_data.iter().map(|(_, val)| *val).collect::<Vec<_>>();
                            unique_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                            unique_values.dedup_by(|a, b| (*a - *b).abs() < 1e-15);
                            (unique_values.len() + 1) as f64
                        };
                        for &idx in nan_indices.iter() {
                            result[idx] = dense_rank;
                        }
                    },
                    _ => panic!("不支持的method: {}", method),
                }
            }
        },
        _ => panic!("不支持的na_option: {}", na_option),
    }
    
    // 处理非NaN值的排名
    if !valid_data.is_empty() {
        let rank_offset = match na_option {
            "top" => nan_indices.len() as f64,
            _ => 0.0,
        };
        
        match method {
            "average" => {
                assign_average_ranks(&mut result, &valid_data, rank_offset);
            },
            "min" => {
                assign_min_ranks(&mut result, &valid_data, rank_offset);
            },
            "max" => {
                assign_max_ranks(&mut result, &valid_data, rank_offset);
            },
            "first" => {
                assign_first_ranks(&mut result, &valid_data, rank_offset);
            },
            "dense" => {
                assign_dense_ranks(&mut result, &valid_data, rank_offset);
            },
            _ => panic!("不支持的method: {}", method),
        }
    }
    
    result
}

/// 分配平均排名
fn assign_average_ranks(result: &mut Vec<f64>, valid_data: &[(usize, f64)], rank_offset: f64) {
    let mut i = 0;
    while i < valid_data.len() {
        let current_val = valid_data[i].1;
        let mut j = i;
        
        // 找到所有相等的值
        while j < valid_data.len() && (valid_data[j].1 - current_val).abs() < 1e-15 {
            j += 1;
        }
        
        // 计算平均排名
        let avg_rank = rank_offset + ((i + 1) + j) as f64 / 2.0;
        
        // 分配排名
        for k in i..j {
            result[valid_data[k].0] = avg_rank;
        }
        
        i = j;
    }
}

/// 分配最小排名
fn assign_min_ranks(result: &mut Vec<f64>, valid_data: &[(usize, f64)], rank_offset: f64) {
    let mut i = 0;
    while i < valid_data.len() {
        let current_val = valid_data[i].1;
        let min_rank = rank_offset + (i + 1) as f64;
        let mut j = i;
        
        // 找到所有相等的值并分配最小排名
        while j < valid_data.len() && (valid_data[j].1 - current_val).abs() < 1e-15 {
            result[valid_data[j].0] = min_rank;
            j += 1;
        }
        
        i = j;
    }
}

/// 分配最大排名
fn assign_max_ranks(result: &mut Vec<f64>, valid_data: &[(usize, f64)], rank_offset: f64) {
    let mut i = 0;
    while i < valid_data.len() {
        let current_val = valid_data[i].1;
        let mut j = i;
        
        // 找到所有相等的值
        while j < valid_data.len() && (valid_data[j].1 - current_val).abs() < 1e-15 {
            j += 1;
        }
        
        let max_rank = rank_offset + j as f64;
        
        // 分配最大排名
        for k in i..j {
            result[valid_data[k].0] = max_rank;
        }
        
        i = j;
    }
}

/// 分配先出现排名
fn assign_first_ranks(result: &mut Vec<f64>, valid_data: &[(usize, f64)], rank_offset: f64) {
    for (rank, &(idx, _)) in valid_data.iter().enumerate() {
        result[idx] = rank_offset + (rank + 1) as f64;
    }
}

/// 分配密集排名
fn assign_dense_ranks(result: &mut Vec<f64>, valid_data: &[(usize, f64)], rank_offset: f64) {
    let mut i = 0;
    let mut dense_rank = 1.0 + rank_offset;
    
    while i < valid_data.len() {
        let current_val = valid_data[i].1;
        let mut j = i;
        
        // 找到所有相等的值并分配相同的密集排名
        while j < valid_data.len() && (valid_data[j].1 - current_val).abs() < 1e-15 {
            result[valid_data[j].0] = dense_rank;
            j += 1;
        }
        
        dense_rank += 1.0;
        i = j;
    }
}

/// 高性能merge函数，支持数据表连接操作
/// 
/// 参数说明：
/// ----------
/// left_data : PyReadonlyArray2<f64>
///     左表数据（numpy数组）
/// right_data : PyReadonlyArray2<f64>  
///     右表数据（numpy数组）
/// left_keys : Vec<usize>
///     左表连接键的列索引
/// right_keys : Vec<usize>
///     右表连接键的列索引
/// how : &str
///     连接类型，支持："inner", "left", "right", "outer"
/// 
/// 返回值：
/// -------
/// PyResult<(Vec<Vec<usize>>, Vec<Vec<f64>>)>
///     返回 (行索引对, 合并后的数据)
///     - 第一个Vec是行索引对：[(left_idx, right_idx), ...]
///     - 第二个Vec是合并后的数据行
#[pyfunction]
#[pyo3(signature = (left_data, right_data, left_keys, right_keys, how="inner"))]
pub fn fast_merge(
    py: Python,
    left_data: PyReadonlyArray2<f64>,
    right_data: PyReadonlyArray2<f64>, 
    left_keys: Vec<usize>,
    right_keys: Vec<usize>,
    how: &str,
) -> PyResult<PyObject> {
    let left_array = left_data.as_array();
    let right_array = right_data.as_array();
    
    // 验证参数
    if left_keys.len() != right_keys.len() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "左表和右表的连接键数量必须相同"
        ));
    }
    
    if left_keys.iter().any(|&k| k >= left_array.ncols()) {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "左表连接键索引超出范围"
        ));
    }
    
    if right_keys.iter().any(|&k| k >= right_array.ncols()) {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "右表连接键索引超出范围"
        ));
    }
    
    match how {
        "inner" => fast_inner_join(py, &left_array, &right_array, &left_keys, &right_keys),
        "left" => fast_left_join(py, &left_array, &right_array, &left_keys, &right_keys),
        "right" => fast_right_join(py, &left_array, &right_array, &left_keys, &right_keys),
        "outer" => fast_outer_join(py, &left_array, &right_array, &left_keys, &right_keys),
        _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("不支持的连接类型: {}", how)
        )),
    }
}


/// 内连接实现
fn fast_inner_join(
    py: Python,
    left: &ndarray::ArrayView2<f64>,
    right: &ndarray::ArrayView2<f64>,
    left_keys: &[usize],
    right_keys: &[usize],
) -> PyResult<PyObject> {
    // 构建右表哈希索引
    let mut right_index: HashMap<Vec<u64>, Vec<usize>> = HashMap::new();
    
    for (row_idx, row) in right.axis_iter(Axis(0)).enumerate() {
        let key: Vec<u64> = right_keys.iter()
            .map(|&col_idx| row[col_idx].to_bits())
            .collect();
        
        right_index.entry(key).or_insert_with(Vec::new).push(row_idx);
    }
    
    // 执行内连接
    let mut result_rows: Vec<Vec<f64>> = Vec::new();
    let mut left_indices: Vec<usize> = Vec::new();
    let mut right_indices: Vec<usize> = Vec::new();
    
    for (left_row_idx, left_row) in left.axis_iter(Axis(0)).enumerate() {
        let left_key: Vec<u64> = left_keys.iter()
            .map(|&col_idx| left_row[col_idx].to_bits())
            .collect();
        
        if let Some(right_row_indices) = right_index.get(&left_key) {
            for &right_row_idx in right_row_indices {
                // 合并行数据
                let mut merged_row = Vec::with_capacity(left.ncols() + right.ncols());
                
                // 添加左表数据
                for &val in left_row.iter() {
                    merged_row.push(val);
                }
                
                // 添加右表数据
                let right_row = right.row(right_row_idx);
                for &val in right_row.iter() {
                    merged_row.push(val);
                }
                
                result_rows.push(merged_row);
                left_indices.push(left_row_idx);
                right_indices.push(right_row_idx);
            }
        }
    }
    
    // 转换为Python对象
    let indices = (left_indices, right_indices);
    let result_tuple = (indices, result_rows);
    
    Ok(result_tuple.to_object(py))
}

/// 左连接实现
fn fast_left_join(
    py: Python,
    left: &ndarray::ArrayView2<f64>,
    right: &ndarray::ArrayView2<f64>,
    left_keys: &[usize],
    right_keys: &[usize],
) -> PyResult<PyObject> {
    // 构建右表哈希索引
    let mut right_index: HashMap<Vec<u64>, Vec<usize>> = HashMap::new();
    
    for (row_idx, row) in right.axis_iter(Axis(0)).enumerate() {
        let key: Vec<u64> = right_keys.iter()
            .map(|&col_idx| row[col_idx].to_bits())
            .collect();
        
        right_index.entry(key).or_insert_with(Vec::new).push(row_idx);
    }
    
    // 执行左连接
    let mut result_rows: Vec<Vec<f64>> = Vec::new();
    let mut left_indices: Vec<usize> = Vec::new();
    let mut right_indices: Vec<Option<usize>> = Vec::new();
    
    for (left_row_idx, left_row) in left.axis_iter(Axis(0)).enumerate() {
        let left_key: Vec<u64> = left_keys.iter()
            .map(|&col_idx| left_row[col_idx].to_bits())
            .collect();
        
        if let Some(right_row_indices) = right_index.get(&left_key) {
            // 有匹配的右表记录
            for &right_row_idx in right_row_indices {
                let mut merged_row = Vec::with_capacity(left.ncols() + right.ncols());
                
                // 添加左表数据
                for &val in left_row.iter() {
                    merged_row.push(val);
                }
                
                // 添加右表数据
                let right_row = right.row(right_row_idx);
                for &val in right_row.iter() {
                    merged_row.push(val);
                }
                
                result_rows.push(merged_row);
                left_indices.push(left_row_idx);
                right_indices.push(Some(right_row_idx));
            }
        } else {
            // 没有匹配的右表记录，填充NaN
            let mut merged_row = Vec::with_capacity(left.ncols() + right.ncols());
            
            // 添加左表数据
            for &val in left_row.iter() {
                merged_row.push(val);
            }
            
            // 右表部分填充NaN
            for _ in 0..right.ncols() {
                merged_row.push(f64::NAN);
            }
            
            result_rows.push(merged_row);
            left_indices.push(left_row_idx);
            right_indices.push(None);
        }
    }
    
    // 转换为Python对象
    let indices = (left_indices, right_indices);
    let result_tuple = (indices, result_rows);
    
    Ok(result_tuple.to_object(py))
}

/// 右连接实现
fn fast_right_join(
    py: Python,
    left: &ndarray::ArrayView2<f64>,
    right: &ndarray::ArrayView2<f64>,
    left_keys: &[usize],
    right_keys: &[usize],
) -> PyResult<PyObject> {
    // 通过交换左右表实现右连接
    fast_left_join(py, right, left, right_keys, left_keys)
}

/// 外连接实现
fn fast_outer_join(
    py: Python,
    left: &ndarray::ArrayView2<f64>,
    right: &ndarray::ArrayView2<f64>,
    left_keys: &[usize],
    right_keys: &[usize],
) -> PyResult<PyObject> {
    use std::collections::HashSet;
    
    // 构建右表哈希索引
    let mut right_index: HashMap<Vec<u64>, Vec<usize>> = HashMap::new();
    let mut all_right_keys: HashSet<Vec<u64>> = HashSet::new();
    
    for (row_idx, row) in right.axis_iter(Axis(0)).enumerate() {
        let key: Vec<u64> = right_keys.iter()
            .map(|&col_idx| row[col_idx].to_bits())
            .collect();
        
        right_index.entry(key.clone()).or_insert_with(Vec::new).push(row_idx);
        all_right_keys.insert(key);
    }
    
    let mut result_rows: Vec<Vec<f64>> = Vec::new();
    let mut left_indices: Vec<Option<usize>> = Vec::new();
    let mut right_indices: Vec<Option<usize>> = Vec::new();
    let mut processed_right_keys: HashSet<Vec<u64>> = HashSet::new();
    
    // 处理左表行
    for (left_row_idx, left_row) in left.axis_iter(Axis(0)).enumerate() {
        let left_key: Vec<u64> = left_keys.iter()
            .map(|&col_idx| left_row[col_idx].to_bits())
            .collect();
        
        if let Some(right_row_indices) = right_index.get(&left_key) {
            processed_right_keys.insert(left_key);
            
            for &right_row_idx in right_row_indices {
                let mut merged_row = Vec::with_capacity(left.ncols() + right.ncols());
                
                // 添加左表数据
                for &val in left_row.iter() {
                    merged_row.push(val);
                }
                
                // 添加右表数据
                let right_row = right.row(right_row_idx);
                for &val in right_row.iter() {
                    merged_row.push(val);
                }
                
                result_rows.push(merged_row);
                left_indices.push(Some(left_row_idx));
                right_indices.push(Some(right_row_idx));
            }
        } else {
            // 左表独有记录
            let mut merged_row = Vec::with_capacity(left.ncols() + right.ncols());
            
            for &val in left_row.iter() {
                merged_row.push(val);
            }
            
            for _ in 0..right.ncols() {
                merged_row.push(f64::NAN);
            }
            
            result_rows.push(merged_row);
            left_indices.push(Some(left_row_idx));
            right_indices.push(None);
        }
    }
    
    // 处理右表独有记录
    for right_key in all_right_keys.difference(&processed_right_keys) {
        if let Some(right_row_indices) = right_index.get(right_key) {
            for &right_row_idx in right_row_indices {
                let mut merged_row = Vec::with_capacity(left.ncols() + right.ncols());
                
                // 左表部分填充NaN
                for _ in 0..left.ncols() {
                    merged_row.push(f64::NAN);
                }
                
                // 添加右表数据
                let right_row = right.row(right_row_idx);
                for &val in right_row.iter() {
                    merged_row.push(val);
                }
                
                result_rows.push(merged_row);
                left_indices.push(None);
                right_indices.push(Some(right_row_idx));
            }
        }
    }
    
    let indices = (left_indices, right_indices);
    let result_tuple = (indices, result_rows);
    
    Ok(result_tuple.to_object(py))
}


/// 通用的哈希键类型，支持字符串、数值和日期时间
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum MergeKey {
    String(String),
    Float(u64),  // 使用f64的位表示来保证Hash和Eq
    Int(i64),
    Timestamp(i64),  // 使用纳秒时间戳
}

impl MergeKey {
    /// 从Python对象创建MergeKey
    fn from_py_any(obj: &PyAny) -> PyResult<Self> {
        if let Ok(s) = obj.downcast::<PyString>() {
            Ok(MergeKey::String(s.to_str()?.to_string()))
        } else if let Ok(f) = obj.extract::<f64>() {
            Ok(MergeKey::Float(f.to_bits()))
        } else if let Ok(i) = obj.extract::<i64>() {
            Ok(MergeKey::Int(i))
        } else {
            // 尝试处理pandas Timestamp类型
            let type_name = obj.get_type().name()?;
            if type_name == "Timestamp" {
                // 尝试获取timestamp的value属性（纳秒时间戳）
                if let Ok(value) = obj.getattr("value") {
                    if let Ok(timestamp_ns) = value.extract::<i64>() {
                        return Ok(MergeKey::Timestamp(timestamp_ns));
                    }
                }
                // 如果上面失败，尝试转换为字符串
                if let Ok(timestamp_str) = obj.str() {
                    if let Ok(s) = timestamp_str.to_str() {
                        return Ok(MergeKey::String(s.to_string()));
                    }
                }
            }
            
            // 对于其他日期时间类型，尝试转换为字符串
            if type_name.contains("datetime") || type_name.contains("date") || type_name.contains("time") {
                if let Ok(datetime_str) = obj.str() {
                    if let Ok(s) = datetime_str.to_str() {
                        return Ok(MergeKey::String(s.to_string()));
                    }
                }
            }
            
            Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                format!("不支持的键类型: {} (尝试转换为字符串或时间戳失败)", type_name)
            ))
        }
    }
}

/// 高性能DataFrame内连接，专门优化Python DataFrame处理
#[pyfunction]
#[pyo3(signature = (left_df, right_df, left_keys, right_keys))]
pub fn fast_inner_join_dataframes(
    py: Python,
    left_df: &PyAny,
    right_df: &PyAny,
    left_keys: Vec<usize>,
    right_keys: Vec<usize>,
) -> PyResult<PyObject> {
    // 使用PyO3直接访问DataFrame的values属性，避免Python转换
    let left_values = left_df.getattr("values")?;
    let right_values = right_df.getattr("values")?;
    
    // 获取numpy数组
    let left_array = left_values.downcast::<PyArray2<f64>>()?;
    let right_array = right_values.downcast::<PyArray2<f64>>()?;
    
    // 转换为Rust数组
    let left_readonly = left_array.readonly();
    let right_readonly = right_array.readonly();
    
    let left_data = left_readonly.as_array();
    let right_data = right_readonly.as_array();
    
    // 使用优化的哈希表进行内连接
    let result = fast_inner_join_arrays(&left_data.view(), &right_data.view(), &left_keys, &right_keys);
    
    // 构造结果
    let result_rows: Vec<Vec<f64>> = result.into_iter().collect();
    
    // 返回Python对象
    Ok(result_rows.to_object(py))
}

/// 优化的数组内连接实现
fn fast_inner_join_arrays(
    left: &ndarray::ArrayView2<f64>,
    right: &ndarray::ArrayView2<f64>,
    left_keys: &[usize],
    right_keys: &[usize],
) -> Vec<Vec<f64>> {
    use std::collections::HashMap;
    
    // 构建右表的哈希映射
    let mut right_map: HashMap<Vec<u64>, Vec<usize>> = HashMap::new();
    
    for (row_idx, row) in right.axis_iter(Axis(0)).enumerate() {
        let key: Vec<u64> = right_keys.iter()
            .map(|&col_idx| row[col_idx].to_bits())
            .collect();
        
        right_map.entry(key).or_insert_with(Vec::new).push(row_idx);
    }
    
    // 执行连接
    let mut result = Vec::new();
    
    for (_left_row_idx, left_row) in left.axis_iter(Axis(0)).enumerate() {
        let left_key: Vec<u64> = left_keys.iter()
            .map(|&col_idx| left_row[col_idx].to_bits())
            .collect();
        
        if let Some(right_indices) = right_map.get(&left_key) {
            for &right_row_idx in right_indices {
                let right_row = right.row(right_row_idx);
                
                // 构造结果行：左表所有列 + 右表非连接键列
                let mut result_row = Vec::new();
                
                // 添加左表列
                for &val in left_row.iter() {
                    result_row.push(val);
                }
                
                // 添加右表非连接键列
                for (col_idx, &val) in right_row.iter().enumerate() {
                    if !right_keys.contains(&col_idx) {
                        result_row.push(val);
                    }
                }
                
                result.push(result_row);
            }
        }
    }
    
    result
}

/// 高性能merge函数，支持字符串和数值类型的连接键
#[pyfunction]
#[pyo3(signature = (left_data, right_data, left_keys, right_keys, how="inner"))]
pub fn fast_merge_mixed(
    py: Python,
    left_data: &PyAny,
    right_data: &PyAny,
    left_keys: Vec<usize>,
    right_keys: Vec<usize>,
    how: &str,
) -> PyResult<PyObject> {
    // 验证参数
    if left_keys.len() != right_keys.len() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "左表和右表的连接键数量必须相同"
        ));
    }
    
    // 提取数据为Python对象的二维数组
    let left_rows = extract_data_as_py_objects(py, left_data)?;
    let right_rows = extract_data_as_py_objects(py, right_data)?;
    
    // 验证列索引
    if !left_rows.is_empty() && left_keys.iter().any(|&k| k >= left_rows[0].len()) {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "左表连接键索引超出范围"
        ));
    }
    
    if !right_rows.is_empty() && right_keys.iter().any(|&k| k >= right_rows[0].len()) {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "右表连接键索引超出范围"
        ));
    }
    
    match how {
        "inner" => fast_inner_join_mixed(py, &left_rows, &right_rows, &left_keys, &right_keys),
        "left" => fast_left_join_mixed(py, &left_rows, &right_rows, &left_keys, &right_keys),
        "right" => fast_right_join_mixed(py, &left_rows, &right_rows, &left_keys, &right_keys),
        "outer" => fast_outer_join_mixed(py, &left_rows, &right_rows, &left_keys, &right_keys),
        _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("不支持的连接类型: {}", how)
        )),
    }
}

/// 从Python数据中提取为PyObject的二维数组
fn extract_data_as_py_objects(py: Python, data: &PyAny) -> PyResult<Vec<Vec<PyObject>>> {
    let mut rows = Vec::new();
    
    // 尝试作为pandas DataFrame处理
    if data.hasattr("values")? && data.hasattr("index")? {
        // 这是一个pandas DataFrame
        let values = data.getattr("values")?;
        return extract_from_numpy_like(py, values);
    }
    
    // 尝试作为列表处理
    if let Ok(list) = data.downcast::<PyList>() {
        for item in list.iter() {
            if let Ok(row_list) = item.downcast::<PyList>() {
                let mut row = Vec::new();
                for cell in row_list.iter() {
                    row.push(cell.to_object(py));
                }
                rows.push(row);
            } else {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "数据必须是二维列表"
                ));
            }
        }
        return Ok(rows);
    }
    
    // 尝试作为numpy数组处理
    extract_from_numpy_like(py, data)
}

/// 从numpy-like数组中提取数据
fn extract_from_numpy_like(py: Python, array: &PyAny) -> PyResult<Vec<Vec<PyObject>>> {
    let mut rows = Vec::new();
    
    // 获取数组的形状
    let shape: (usize, usize) = if let Ok(shape_tuple) = array.getattr("shape") {
        shape_tuple.extract()?
    } else {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "无法获取数组形状"
        ));
    };
    
    let (n_rows, n_cols) = shape;
    
    for i in 0..n_rows {
        let mut row = Vec::new();
        for j in 0..n_cols {
            let item = array.get_item((i, j))?;
            row.push(item.to_object(py));
        }
        rows.push(row);
    }
    
    Ok(rows)
}

/// 提取多个列作为组合键
fn extract_multi_key(row: &[PyObject], key_indices: &[usize]) -> PyResult<Vec<MergeKey>> {
    Python::with_gil(|py| {
        let mut keys = Vec::new();
        for &idx in key_indices {
            let py_obj = &row[idx];
            let key = MergeKey::from_py_any(py_obj.as_ref(py))?;
            keys.push(key);
        }
        Ok(keys)
    })
}

/// 内连接实现（混合类型）
fn fast_inner_join_mixed(
    py: Python,
    left_rows: &[Vec<PyObject>],
    right_rows: &[Vec<PyObject>],
    left_keys: &[usize],
    right_keys: &[usize],
) -> PyResult<PyObject> {
    // 构建右表哈希索引
    let mut right_index: HashMap<Vec<MergeKey>, Vec<usize>> = HashMap::new();
    
    for (row_idx, row) in right_rows.iter().enumerate() {
        let key = extract_multi_key(row, right_keys)?;
        right_index.entry(key).or_insert_with(Vec::new).push(row_idx);
    }
    
    // 执行内连接
    let mut result_rows: Vec<Vec<PyObject>> = Vec::new();
    let mut left_indices: Vec<usize> = Vec::new();
    let mut right_indices: Vec<usize> = Vec::new();
    
    for (left_row_idx, left_row) in left_rows.iter().enumerate() {
        let left_key = extract_multi_key(left_row, left_keys)?;
        
        if let Some(right_row_indices) = right_index.get(&left_key) {
            for &right_row_idx in right_row_indices {
                // 合并行数据
                let mut merged_row = Vec::with_capacity(left_row.len() + right_rows[right_row_idx].len());
                
                // 添加左表数据
                for obj in left_row.iter() {
                    merged_row.push(obj.clone_ref(py));
                }
                
                // 添加右表数据
                for obj in right_rows[right_row_idx].iter() {
                    merged_row.push(obj.clone_ref(py));
                }
                
                result_rows.push(merged_row);
                left_indices.push(left_row_idx);
                right_indices.push(right_row_idx);
            }
        }
    }
    
    // 转换为Python对象
    let indices = (left_indices, right_indices);
    let result_tuple = (indices, result_rows);
    
    Ok(result_tuple.to_object(py))
}

/// 左连接实现（混合类型）
fn fast_left_join_mixed(
    py: Python,
    left_rows: &[Vec<PyObject>],
    right_rows: &[Vec<PyObject>],
    left_keys: &[usize],
    right_keys: &[usize],
) -> PyResult<PyObject> {
    // 构建右表哈希索引
    let mut right_index: HashMap<Vec<MergeKey>, Vec<usize>> = HashMap::new();
    
    for (row_idx, row) in right_rows.iter().enumerate() {
        let key = extract_multi_key(row, right_keys)?;
        right_index.entry(key).or_insert_with(Vec::new).push(row_idx);
    }
    
    // 执行左连接
    let mut result_rows: Vec<Vec<PyObject>> = Vec::new();
    let mut left_indices: Vec<usize> = Vec::new();
    let mut right_indices: Vec<Option<usize>> = Vec::new();
    
    let right_cols = if right_rows.is_empty() { 0 } else { right_rows[0].len() };
    
    for (left_row_idx, left_row) in left_rows.iter().enumerate() {
        let left_key = extract_multi_key(left_row, left_keys)?;
        
        if let Some(right_row_indices) = right_index.get(&left_key) {
            // 有匹配的右表记录
            for &right_row_idx in right_row_indices {
                let mut merged_row = Vec::with_capacity(left_row.len() + right_rows[right_row_idx].len());
                
                // 添加左表数据
                for obj in left_row.iter() {
                    merged_row.push(obj.clone_ref(py));
                }
                
                // 添加右表数据
                for obj in right_rows[right_row_idx].iter() {
                    merged_row.push(obj.clone_ref(py));
                }
                
                result_rows.push(merged_row);
                left_indices.push(left_row_idx);
                right_indices.push(Some(right_row_idx));
            }
        } else {
            // 没有匹配的右表记录，填充None
            let mut merged_row = Vec::with_capacity(left_row.len() + right_cols);
            
            // 添加左表数据
            for obj in left_row.iter() {
                merged_row.push(obj.clone_ref(py));
            }
            
            // 右表部分填充None
            for _ in 0..right_cols {
                merged_row.push(py.None());
            }
            
            result_rows.push(merged_row);
            left_indices.push(left_row_idx);
            right_indices.push(None);
        }
    }
    
    // 转换为Python对象
    let indices = (left_indices, right_indices);
    let result_tuple = (indices, result_rows);
    
    Ok(result_tuple.to_object(py))
}

/// 右连接实现（混合类型）
fn fast_right_join_mixed(
    py: Python,
    left_rows: &[Vec<PyObject>],
    right_rows: &[Vec<PyObject>],
    left_keys: &[usize],
    right_keys: &[usize],
) -> PyResult<PyObject> {
    // 通过交换左右表实现右连接
    fast_left_join_mixed(py, right_rows, left_rows, right_keys, left_keys)
}

/// 外连接实现（混合类型）
fn fast_outer_join_mixed(
    py: Python,
    left_rows: &[Vec<PyObject>],
    right_rows: &[Vec<PyObject>],
    left_keys: &[usize],
    right_keys: &[usize],
) -> PyResult<PyObject> {
    // 构建右表哈希索引
    let mut right_index: HashMap<Vec<MergeKey>, Vec<usize>> = HashMap::new();
    let mut all_right_keys: HashSet<Vec<MergeKey>> = HashSet::new();
    
    for (row_idx, row) in right_rows.iter().enumerate() {
        let key = extract_multi_key(row, right_keys)?;
        right_index.entry(key.clone()).or_insert_with(Vec::new).push(row_idx);
        all_right_keys.insert(key);
    }
    
    let mut result_rows: Vec<Vec<PyObject>> = Vec::new();
    let mut left_indices: Vec<Option<usize>> = Vec::new();
    let mut right_indices: Vec<Option<usize>> = Vec::new();
    let mut processed_right_keys: HashSet<Vec<MergeKey>> = HashSet::new();
    
    let left_cols = if left_rows.is_empty() { 0 } else { left_rows[0].len() };
    let right_cols = if right_rows.is_empty() { 0 } else { right_rows[0].len() };
    
    // 处理左表行
    for (left_row_idx, left_row) in left_rows.iter().enumerate() {
        let left_key = extract_multi_key(left_row, left_keys)?;
        
        if let Some(right_row_indices) = right_index.get(&left_key) {
            processed_right_keys.insert(left_key);
            
            for &right_row_idx in right_row_indices {
                let mut merged_row = Vec::with_capacity(left_row.len() + right_rows[right_row_idx].len());
                
                // 添加左表数据
                for obj in left_row.iter() {
                    merged_row.push(obj.clone_ref(py));
                }
                
                // 添加右表数据
                for obj in right_rows[right_row_idx].iter() {
                    merged_row.push(obj.clone_ref(py));
                }
                
                result_rows.push(merged_row);
                left_indices.push(Some(left_row_idx));
                right_indices.push(Some(right_row_idx));
            }
        } else {
            // 左表独有记录
            let mut merged_row = Vec::with_capacity(left_cols + right_cols);
            
            for obj in left_row.iter() {
                merged_row.push(obj.clone_ref(py));
            }
            
            for _ in 0..right_cols {
                merged_row.push(py.None());
            }
            
            result_rows.push(merged_row);
            left_indices.push(Some(left_row_idx));
            right_indices.push(None);
        }
    }
    
    // 处理右表独有记录
    for right_key in all_right_keys.difference(&processed_right_keys) {
        if let Some(right_row_indices) = right_index.get(right_key) {
            for &right_row_idx in right_row_indices {
                let mut merged_row = Vec::with_capacity(left_cols + right_cols);
                
                // 左表部分填充None
                for _ in 0..left_cols {
                    merged_row.push(py.None());
                }
                
                // 添加右表数据
                for obj in right_rows[right_row_idx].iter() {
                    merged_row.push(obj.clone_ref(py));
                }
                
                result_rows.push(merged_row);
                left_indices.push(None);
                right_indices.push(Some(right_row_idx));
            }
        }
    }
    
    let indices = (left_indices, right_indices);
    let result_tuple = (indices, result_rows);
    
    Ok(result_tuple.to_object(py))
}
