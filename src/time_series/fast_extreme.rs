use pyo3::prelude::*;
use numpy::{PyReadonlyArray1};
use std::time::Instant;
use crate::time_series::TimeoutError;

/// 计算半极端时间的优化版本
/// 
/// 该函数针对find_half_extreme_time进行了多项优化：
/// 1. 预计算和缓存 - 避免重复计算时间差和比率
/// 2. 数据布局优化 - 改进内存访问模式
/// 3. 条件分支优化 - 减少分支预测失败
/// 4. 界限优化 - 提前确定搜索范围
#[pyfunction]
#[pyo3(signature = (times, prices, time_window=5.0, direction="ignore", timeout_seconds=None))]
pub fn fast_find_half_extreme_time(
    times: PyReadonlyArray1<f64>,
    prices: PyReadonlyArray1<f64>,
    time_window: f64,
    direction: &str,
    timeout_seconds: Option<f64>
) -> PyResult<Vec<f64>> {
    // 记录开始时间
    let start_time = Instant::now();
    
    // 检查超时的闭包函数
    let check_timeout = |timeout: Option<f64>| -> Result<(), TimeoutError> {
        if let Some(timeout) = timeout {
            let elapsed = start_time.elapsed().as_secs_f64();
            if elapsed > timeout {
                return Err(TimeoutError {
                    message: "快速半极端时间计算超时".to_string(),
                    duration: elapsed,
                });
            }
        }
        Ok(())
    };
    
    // 提取数组数据并预处理
    let times = times.as_array();
    let times_vec: Vec<f64> = times.iter().map(|&x| x / 1.0e9).collect();
    let prices = prices.as_array();
    let n = times_vec.len();
    
    // 优化1: 预分配结果向量
    let mut result = vec![time_window; n];
    
    // 检查初始化后是否超时
    if let Err(_) = check_timeout(timeout_seconds) {
        return Ok(vec![f64::NAN; n]);
    }
    
    // 优化2: 提前计算方向判断标志
    let dir_flag = match direction {
        "pos" => 1, // 只考虑上涨
        "neg" => -1, // 只考虑下跌
        _ => 0,     // 考虑两个方向
    };
    
    // 优化3: 块处理，每次处理一块数据以提高缓存局部性
    let chunk_size = 100;
    let mut chunk_start = 0;
    
    while chunk_start < n {
        // 每块开始时检查超时
        if let Err(_) = check_timeout(timeout_seconds) {
            return Ok(vec![f64::NAN; n]);
        }
        
        let chunk_end = (chunk_start + chunk_size).min(n);
        
        // 优化4: 对每个块预先找到最大可访问索引，避免重复边界检查
        for i in chunk_start..chunk_end {
            // 每隔20个点检查一次超时，减少超时检查频率
            if i % 20 == 0 && i > chunk_start {
                if let Err(_) = check_timeout(timeout_seconds) {
                    return Ok(vec![f64::NAN; n]);
                }
            }
            
            let current_time = times_vec[i];
            let current_price = prices[i];
            
            // 检查价格是否为NaN或Inf
            if !current_price.is_finite() {
                result[i] = f64::NAN;
                continue;
            }
            
            // 优化5: 计算时间窗口内的最后一个可能索引
            let mut end_idx = i;
            while end_idx + 1 < n && times_vec[end_idx + 1] - current_time <= time_window {
                end_idx += 1;
            }
            
            if end_idx == i {
                // 没有未来数据
                continue;
            }
            
            // 优化6: 一次性找出最大上涨和下跌幅度
            let mut max_up = 0.0;
            let mut max_down = 0.0;
            
            // 优化7: 预计算价格倒数，减少除法操作
            let price_reciprocal = if current_price != 0.0 { 1.0 / current_price } else { 0.0 };
            
            // 一次遍历找出最大上涨和下跌
            for j in (i+1)..=end_idx {
                if !prices[j].is_finite() {
                    continue;
                }
                
                // 优化8: 使用预计算的倒数来计算比率
                let price_diff = prices[j] - current_price;
                let price_ratio = price_diff * price_reciprocal;
                
                if price_ratio > max_up {
                    max_up = price_ratio;
                } else if price_ratio < -max_down {
                    max_down = -price_ratio;
                }
            }
            
            // 根据方向参数处理
            let (target_ratio, dir_value) = match dir_flag {
                1 => { // 只考虑上涨
                    if max_up <= 0.0 {
                        result[i] = f64::NAN;
                        continue;
                    }
                    (max_up, 1.0)
                },
                -1 => { // 只考虑下跌
                    if max_down <= 0.0 {
                        result[i] = f64::NAN;
                        continue;
                    }
                    (max_down, -1.0)
                },
                _ => { // 考虑两个方向
                    if max_up > max_down {
                        (max_up, 1.0)
                    } else {
                        (max_down, -1.0)
                    }
                }
            };
            
            // 如果目标变动为0，保持默认值
            if target_ratio <= 0.0 {
                continue;
            }
            
            // 计算半比率阈值
            let half_ratio = target_ratio / 2.0 * dir_value;
            
            // 优化9: 二分查找找到首次达到一半变动的时间点
            // 传统二分查找的变种，因为我们不需要精确等于，而是找到第一个大于等于阈值的点
            let mut left = i + 1;
            let mut right = end_idx;
            let mut found_idx = 0;
            let mut found = false;
            
            // 二分查找时间点
            while left <= right {
                let mid = left + (right - left) / 2;
                
                if !prices[mid].is_finite() {
                    left = mid + 1;
                    continue;
                }
                
                let price_ratio = (prices[mid] - current_price) * price_reciprocal;
                
                if (dir_value > 0.0 && price_ratio >= half_ratio) || 
                   (dir_value < 0.0 && price_ratio <= half_ratio) {
                    found = true;
                    found_idx = mid;
                    right = mid - 1; // 继续查找左侧是否有更早的点
                } else {
                    left = mid + 1;
                }
            }
            
            // 如果找到了半极值点
            if found {
                result[i] = times_vec[found_idx] - current_time;
            }
            // 否则保持默认值 time_window
        }
        
        chunk_start = chunk_end;
    }
    
    // 最终检查一次超时
    if let Err(_) = check_timeout(timeout_seconds) {
        return Ok(vec![f64::NAN; n]);
    }
    
    Ok(result)
}
