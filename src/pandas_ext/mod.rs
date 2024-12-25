use pyo3::prelude::*;
// use pyo3::types::PyList;
// use numpy::PyReadonlyArray1;
// use ndarray::Array1;
// use std::collections::HashMap;
use std::collections::VecDeque;
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
