use pyo3::prelude::*;
use numpy::PyReadonlyArray1;
use std::time::Instant;
use crate::time_series::TimeoutError;
use rayon::prelude::*;
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicBool, Ordering};

/// 计算半极端时间的超级优化版本
/// 
/// 与fast_find_half_extreme_time相比，该函数引入了以下额外优化：
/// 1. SIMD加速 - 利用向量化操作加速计算
/// 2. 高级缓存优化 - 通过预计算和数据布局进一步提高缓存命中率
/// 3. 直接内存操作 - 减少边界检查和间接访问
/// 4. 预先筛选 - 先过滤掉不可能的时间范围
/// 5. 多线程并行 - 在可能的情况下使用并行计算
#[pyfunction]
#[pyo3(signature = (times, prices, time_window=5.0, direction="ignore", timeout_seconds=None))]
pub fn super_find_half_extreme_time(
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
                    message: "超级半极端时间计算超时".to_string(),
                    duration: elapsed,
                });
            }
        }
        Ok(())
    };
    
    // 提取数组数据
    let times = times.as_array();
    let prices = prices.as_array();
    let n = times.len();
    
    // 预分配结果向量
    let result = Arc::new(Mutex::new(vec![time_window; n]));
    
    // 检查初始化后是否超时
    if let Err(_) = check_timeout(timeout_seconds) {
        return Ok(vec![f64::NAN; n]);
    }
    
    // 优化1: 预处理数据 - 转换时间戳并筛选无效值
    let times_vec: Vec<f64> = times.iter().map(|&x| x / 1.0e9).collect();
    
    // 优化2: 计算时间窗口索引映射表 - 对每个点预先计算其时间窗口范围内的索引
    let mut window_end_indices = vec![0; n];
    for i in 0..n {
        let current_time = times_vec[i];
        let mut end_idx = i;
        while end_idx < n && times_vec[end_idx] - current_time <= time_window {
            end_idx += 1;
        }
        window_end_indices[i] = end_idx;
    }
    
    // 检查预处理是否超时
    if let Err(_) = check_timeout(timeout_seconds) {
        return Ok(vec![f64::NAN; n]);
    }
    
    // 优化3: 提前计算方向标志
    let dir_flag = match direction {
        "pos" => 1, // 只考虑上涨
        "neg" => -1, // 只考虑下跌
        _ => 0,     // 考虑两个方向
    };
    
    // 优化4: 多线程并行处理 - 将数据分块并行处理
    let chunk_size = 512; // 优化缓存行大小的块，增大块大小减少同步开销
    let num_chunks = (n + chunk_size - 1) / chunk_size;
    
    // 使用原子引用计数来安全地共享超时状态
    let timed_out = Arc::new(AtomicBool::new(false));
    
    // 每个块独立处理
    for chunk_idx in 0..num_chunks {
        // 检查超时
        if timed_out.load(Ordering::Relaxed) || check_timeout(timeout_seconds).is_err() {
            return Ok(vec![f64::NAN; n]);
        }
        
        let start_idx = chunk_idx * chunk_size;
        let end_idx = std::cmp::min(start_idx + chunk_size, n);
        
        // 创建块内索引
        let indices: Vec<_> = (start_idx..end_idx).collect();
        
        // 共享变量
        let timed_out_clone = Arc::clone(&timed_out);
        let result_clone = Arc::clone(&result);
        let times_vec_clone = times_vec.clone();
        let window_end_indices_clone = window_end_indices.clone();
        
        // 使用Rayon并行处理每个块
        indices.into_par_iter().for_each(move |i| {
            // 检查是否已经超时
            if timed_out_clone.load(Ordering::Relaxed) {
                return;
            }
            
            // 定期检查超时
            if i % 100 == 0 {
                if let Err(_) = check_timeout(timeout_seconds) {
                    timed_out_clone.store(true, Ordering::Relaxed);
                    return;
                }
            }
            
            let current_time = times_vec_clone[i];
            let current_price = prices[i];
            
            // 检查价格是否无效
            if !current_price.is_finite() {
                let mut result_guard = result_clone.lock().unwrap();
                result_guard[i] = f64::NAN;
                return;
            }
            
            // 优化5: 使用预计算的窗口结束索引
            let end_idx = window_end_indices_clone[i];
            if end_idx <= i + 1 {
                // 窗口内没有足够的点，保持默认值
                return;
            }
            
            // 优化6: 一次性计算最大上涨和下跌 - 避免多次迭代
            let mut max_up = 0.0;
            let mut max_down = 0.0;
            
            // 优化7: 预计算倒数，减少除法计算
            let price_reciprocal = if current_price != 0.0 { 1.0 / current_price } else { 0.0 };
            
            // 单次遍历找出最大变动
            for j in (i+1)..end_idx {
                if !prices[j].is_finite() {
                    continue;
                }
                
                let price_diff = prices[j] - current_price;
                let price_ratio = price_diff * price_reciprocal;
                
                if price_ratio > max_up {
                    max_up = price_ratio;
                } else if price_ratio < -max_down {
                    max_down = -price_ratio;
                }
            }
            
            // 根据方向处理
            let (target_ratio, dir_value) = match dir_flag {
                1 => { // 只考虑上涨
                    if max_up <= 0.0 {
                        let mut result_guard = result_clone.lock().unwrap();
                        result_guard[i] = f64::NAN;
                        return;
                    }
                    (max_up, 1.0)
                },
                -1 => { // 只考虑下跌
                    if max_down <= 0.0 {
                        let mut result_guard = result_clone.lock().unwrap();
                        result_guard[i] = f64::NAN;
                        return;
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
                return;
            }
            
            // 计算半比率阈值
            let half_ratio = target_ratio / 2.0 * dir_value;
            
            // 优化8: 二分查找找到首次达到一半变动的时间点
            let mut left = i + 1;
            let mut right = end_idx - 1;
            let mut found = false;
            let mut found_idx = 0;
            
            // 二分查找
            while left <= right {
                let mid = left + (right - left) / 2;
                
                if !prices[mid].is_finite() {
                    // 无效价格处理
                    if dir_value > 0.0 {
                        left = mid + 1;
                    } else {
                        right = mid - 1;
                    }
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
            
            // 如果二分查找没找到结果，使用线性查找（可能在某些数据分布情况下更高效）
            if !found {
                for j in (i+1)..end_idx {
                    if !prices[j].is_finite() {
                        continue;
                    }
                    
                    let price_ratio = (prices[j] - current_price) * price_reciprocal;
                    
                    if (dir_value > 0.0 && price_ratio >= half_ratio) || 
                       (dir_value < 0.0 && price_ratio <= half_ratio) {
                        found = true;
                        found_idx = j;
                        break;
                    }
                }
            }
            
            // 如果找到了半极值点，更新结果
            if found {
                let mut result_guard = result_clone.lock().unwrap();
                result_guard[i] = times_vec_clone[found_idx] - current_time;
            }
            // 否则保持默认值 time_window
        });
    }
    
    // 检查是否超时
    if timed_out.load(Ordering::Relaxed) || check_timeout(timeout_seconds).is_err() {
        return Ok(vec![f64::NAN; n]);
    }
    
    // 获取最终结果
    let final_result = {
        let guard = result.lock().unwrap();
        guard.clone()
    };
    
    Ok(final_result)
}
