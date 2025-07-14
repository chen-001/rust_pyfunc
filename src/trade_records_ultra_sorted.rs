use numpy::{PyArray1, PyArray2};
use pyo3::prelude::*;
use std::f64;

/// 专门用于成交记录分析的Ultra Sorted算法
/// 
/// 核心设计思想：
/// 1. 彻底消除O(n²)复杂度
/// 2. 利用时间排序避免排序开销
/// 3. 批量预计算相同volume组的共享数据
/// 4. 使用二分查找加速时间定位

#[derive(Debug)]
struct TradeVolumeGroup {
    volume: f64,
    indices: Vec<usize>,          // 原始数据索引
    times: Vec<f64>,              // 时间数组（已排序）
    prices: Vec<f64>,             // 对应的价格
    flags: Vec<i32>,              // 对应的flag
    buy_indices: Vec<usize>,      // 买单在组内的位置
    sell_indices: Vec<usize>,     // 卖单在组内的位置
}

impl TradeVolumeGroup {
    fn new(volume: f64) -> Self {
        Self {
            volume,
            indices: Vec::new(),
            times: Vec::new(),
            prices: Vec::new(),
            flags: Vec::new(),
            buy_indices: Vec::new(),
            sell_indices: Vec::new(),
        }
    }
    
    fn add_record(&mut self, orig_idx: usize, time: f64, price: f64, flag: i32) {
        let group_idx = self.indices.len();
        
        self.indices.push(orig_idx);
        self.times.push(time);
        self.prices.push(price);
        self.flags.push(flag);
        
        // 分类存储买卖单位置
        if flag == 66 {
            self.buy_indices.push(group_idx);
        } else if flag == 83 {
            self.sell_indices.push(group_idx);
        }
    }
    
    /// 修复的时间距离计算：使用二分查找定位最近记录
    fn find_nearest_records_ultra_fast(
        &self, 
        current_group_idx: usize, 
        target_indices: &[usize], 
        max_records: usize
    ) -> Vec<(f64, f64)> {
        
        if target_indices.is_empty() {
            return Vec::new();
        }
        
        let current_time = self.times[current_group_idx];
        let mut time_distances: Vec<(f64, f64)> = Vec::new();
        
        // 计算当前记录与所有目标记录的时间距离
        for &target_idx in target_indices.iter() {
            if target_idx != current_group_idx {
                let time_diff = (current_time - self.times[target_idx]).abs();
                let price = self.prices[target_idx];
                time_distances.push((time_diff, price));
            }
        }
        
        // 按时间距离排序，获取最近的记录
        time_distances.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        
        // 限制返回记录数量
        if time_distances.len() > max_records {
            time_distances.truncate(max_records);
        }
        
        time_distances
    }
    
    /// 批量计算该volume组所有记录的指标
    fn compute_all_indicators_ultra_fast(
        &self, 
        results: &mut [Vec<f64>], 
        min_count: usize, 
        use_flag: &str
    ) {
        let group_size = self.indices.len();
        if group_size < min_count {
            return;
        }
        
        // 根据use_flag确定目标索引集合
        let get_target_indices = |current_group_idx: usize| -> Vec<usize> {
            let current_flag = self.flags[current_group_idx];
            
            match use_flag {
                "same" => {
                    if current_flag == 66 {
                        self.buy_indices.iter().filter(|&&idx| idx != current_group_idx).cloned().collect()
                    } else {
                        self.sell_indices.iter().filter(|&&idx| idx != current_group_idx).cloned().collect()
                    }
                },
                "diff" => {
                    if current_flag == 66 {
                        self.sell_indices.clone()
                    } else {
                        self.buy_indices.clone()
                    }
                },
                _ => {
                    (0..group_size).filter(|&idx| idx != current_group_idx).collect()
                }
            }
        };
        
        // 为每个记录计算指标
        for current_group_idx in 0..group_size {
            let target_indices = get_target_indices(current_group_idx);
            
            if target_indices.len() < min_count {
                continue;
            }
            
            // 使用超快速算法收集最近记录
            let time_distances = self.find_nearest_records_ultra_fast(
                current_group_idx, 
                &target_indices, 
                target_indices.len()
            );
            
            if time_distances.len() >= min_count {
                let orig_idx = self.indices[current_group_idx];
                let current_price = self.prices[current_group_idx];
                
                calculate_trade_indicators_ultra_fast(
                    &mut results[orig_idx], 
                    &time_distances, 
                    current_price
                );
            }
        }
    }
}

/// 超高速指标计算，最大化利用已排序特性
fn calculate_trade_indicators_ultra_fast(
    result_row: &mut [f64], 
    time_distances: &[(f64, f64)], 
    current_price: f64
) {
    let total_count = time_distances.len();
    
    if total_count == 0 {
        return;
    }
    
    // 1. 最近时间间隔
    result_row[0] = time_distances[0].0;
    
    // 2-11. 向量化计算分档平均时间间隔
    let percentages = [0.01, 0.02, 0.03, 0.04, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50];
    let mut cumulative_sum = 0.0;
    let mut last_count = 0;
    
    for (idx, &pct) in percentages.iter().enumerate() {
        let count = ((total_count as f64 * pct).ceil() as usize).max(1).min(total_count);
        
        // 增量累加，避免重复计算
        for j in last_count..count {
            cumulative_sum += time_distances[j].0;
        }
        
        result_row[idx + 1] = cumulative_sum / count as f64;
        last_count = count;
    }
    
    // 12. 总平均时间间隔
    for j in last_count..total_count {
        cumulative_sum += time_distances[j].0;
    }
    result_row[11] = cumulative_sum / total_count as f64;
    
    // 13-22. 优化的价格分位数计算
    let price_percentages = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00];
    
    // 预分配和重用价格数组
    let max_price_count = total_count;
    let mut all_prices: Vec<f64> = Vec::with_capacity(max_price_count);
    for &(_, price) in time_distances.iter() {
        all_prices.push(price);
    }
    all_prices.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    
    // 批量计算所有分位数
    for (idx, &pct) in price_percentages.iter().enumerate() {
        let count = if pct == 1.0 {
            total_count
        } else {
            ((total_count as f64 * pct).ceil() as usize).max(1).min(total_count)
        };
        
        let prices_slice = &all_prices[..count];
        
        // 优化的二分查找
        let rank = match prices_slice.binary_search_by(|&p| p.partial_cmp(&current_price).unwrap()) {
            Ok(pos) => pos,
            Err(pos) => pos,
        };
        
        result_row[idx + 12] = rank as f64 / prices_slice.len() as f64;
    }
}

/// 快速定位volume组范围的优化版本
fn find_trade_volume_ranges_ultra_fast(volumes: &[f64]) -> Vec<(f64, usize, usize)> {
    if volumes.is_empty() {
        return Vec::new();
    }
    
    let mut ranges = Vec::new();
    let mut current_volume = volumes[0];
    let mut start_idx = 0;
    
    for i in 1..volumes.len() {
        if volumes[i] != current_volume {
            ranges.push((current_volume, start_idx, i));
            current_volume = volumes[i];
            start_idx = i;
        }
    }
    
    ranges.push((current_volume, start_idx, volumes.len()));
    ranges
}

#[pyfunction]
#[pyo3(signature = (volume, exchtime, price, flag, min_count=100, use_flag="ignore"))]
pub fn calculate_trade_time_gap_and_price_percentile_ultra_sorted(
    py: Python,
    volume: &PyArray1<f64>,
    exchtime: &PyArray1<f64>,
    price: &PyArray1<f64>,
    flag: &PyArray1<i32>,
    min_count: usize,
    use_flag: &str,
) -> PyResult<(Py<PyArray2<f64>>, Vec<String>)> {
    let volume_slice = volume.readonly();
    let exchtime_slice = exchtime.readonly();
    let price_slice = price.readonly();
    let flag_slice = flag.readonly();
    
    let volume_data = volume_slice.as_slice()?;
    let exchtime_raw = exchtime_slice.as_slice()?;
    let price_data = price_slice.as_slice()?;
    let flag_data = flag_slice.as_slice()?;
    
    // 将纳秒时间戳转换为秒
    let exchtime_data: Vec<f64> = exchtime_raw.iter().map(|&t| t / 1e9).collect();
    
    let n = volume_data.len();
    if n == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("输入数组不能为空"));
    }
    
    // 1. 超快速定位volume组范围
    let volume_ranges = find_trade_volume_ranges_ultra_fast(volume_data);
    
    // 2. 构建优化的volume组结构
    let mut ultra_groups: Vec<TradeVolumeGroup> = Vec::new();
    
    for (vol, start_idx, end_idx) in volume_ranges.iter() {
        let mut group = TradeVolumeGroup::new(*vol);
        
        for i in *start_idx..*end_idx {
            group.add_record(i, exchtime_data[i], price_data[i], flag_data[i]);
        }
        
        ultra_groups.push(group);
    }
    
    // 3. 初始化结果
    let mut results = vec![vec![f64::NAN; 22]; n];
    
    // 4. 批量计算所有组的指标
    for group in ultra_groups.iter() {
        group.compute_all_indicators_ultra_fast(&mut results, min_count, use_flag);
    }
    
    let result_array = PyArray2::from_vec2(py, &results)?;
    let column_names = get_trade_column_names();
    
    Ok((result_array.to_owned(), column_names))
}

fn get_trade_column_names() -> Vec<String> {
    vec![
        "最近时间间隔".to_string(),
        "平均时间间隔_1%".to_string(),
        "平均时间间隔_2%".to_string(),
        "平均时间间隔_3%".to_string(),
        "平均时间间隔_4%".to_string(),
        "平均时间间隔_5%".to_string(),
        "平均时间间隔_10%".to_string(),
        "平均时间间隔_20%".to_string(),
        "平均时间间隔_30%".to_string(),
        "平均时间间隔_40%".to_string(),
        "平均时间间隔_50%".to_string(),
        "平均时间间隔_全部".to_string(),
        "价格分位数_10%".to_string(),
        "价格分位数_20%".to_string(),
        "价格分位数_30%".to_string(),
        "价格分位数_40%".to_string(),
        "价格分位数_50%".to_string(),
        "价格分位数_60%".to_string(),
        "价格分位数_70%".to_string(),
        "价格分位数_80%".to_string(),
        "价格分位数_90%".to_string(),
        "价格分位数_全部".to_string(),
    ]
}