use numpy::{PyArray1, PyArray2};
use pyo3::prelude::*;
use std::collections::HashMap;
use std::f64;

/// 超高性能批量计算结构
#[derive(Debug)]
struct VolumeGroup {
    indices: Vec<usize>,          // 该volume组的所有索引
    times: Vec<f64>,             // 对应的时间
    prices: Vec<f64>,            // 对应的价格
    sorted_time_indices: Vec<usize>, // 按时间排序的索引位置
}

impl VolumeGroup {
    fn new() -> Self {
        Self {
            indices: Vec::new(),
            times: Vec::new(),
            prices: Vec::new(),
            sorted_time_indices: Vec::new(),
        }
    }
    
    fn add_record(&mut self, idx: usize, time: f64, price: f64) {
        self.indices.push(idx);
        self.times.push(time);
        self.prices.push(price);
    }
    
    fn prepare_for_computation(&mut self) {
        // 创建按时间排序的索引
        let mut time_index_pairs: Vec<(f64, usize)> = self.times.iter()
            .enumerate()
            .map(|(i, &time)| (time, i))
            .collect();
        
        time_index_pairs.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        
        self.sorted_time_indices = time_index_pairs.into_iter()
            .map(|(_, i)| i)
            .collect();
    }
    
    /// 批量计算所有记录的指标，一次性完成
    fn compute_all_indicators(&self, results: &mut [Vec<f64>], min_count: usize) {
        let n = self.indices.len();
        if n < min_count {
            return;
        }
        
        // 预计算所有时间距离对 - 仍然是O(n²)，但是批量进行
        let mut all_time_distances = Vec::with_capacity(n);
        let mut all_price_lists = Vec::with_capacity(n);
        
        for i in 0..n {
            let current_time = self.times[i];
            let current_price = self.prices[i];
            
            // 计算与其他所有记录的时间距离
            let mut time_distances: Vec<(f64, f64)> = Vec::with_capacity(n - 1);
            
            for j in 0..n {
                if i != j {
                    let time_diff = (current_time - self.times[j]).abs();
                    time_distances.push((time_diff, self.prices[j]));
                }
            }
            
            if time_distances.len() < min_count {
                all_time_distances.push(Vec::new());
                all_price_lists.push(Vec::new());
                continue;
            }
            
            // 排序
            time_distances.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            
            all_time_distances.push(time_distances);
            
            // 提取价格用于分位数计算
            let prices: Vec<f64> = all_time_distances[i].iter().map(|(_, p)| *p).collect();
            all_price_lists.push(prices);
        }
        
        // 批量计算指标
        for i in 0..n {
            if all_time_distances[i].is_empty() {
                continue;
            }
            
            let orig_idx = self.indices[i];
            let current_price = self.prices[i];
            let time_distances = &all_time_distances[i];
            
            calculate_indicators(&mut results[orig_idx], time_distances, current_price);
        }
    }
}

#[pyfunction]
#[pyo3(signature = (volume, exchtime, price, flag, min_count=100, use_flag="ignore"))]
pub fn analyze_trade_records_ultra_turbo(
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
    let exchtime_data = exchtime_slice.as_slice()?;
    let price_data = price_slice.as_slice()?;
    let flag_data = flag_slice.as_slice()?;
    
    let n = volume_data.len();
    if n == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("输入数组不能为空"));
    }
    
    // 1. 按volume分组
    let mut volume_groups: HashMap<String, VolumeGroup> = HashMap::new();
    
    for i in 0..n {
        let vol = volume_data[i];
        let current_flag = flag_data[i];
        
        let key = match use_flag {
            "same" => format!("{}_{}", vol, current_flag),
            "diff" => format!("{}_{}", vol, if current_flag == 66 { 83 } else { 66 }),
            _ => vol.to_string(),
        };
        
        if use_flag == "diff" {
            // 对于diff模式，需要特殊处理
            for j in 0..n {
                let other_flag = flag_data[j];
                if current_flag != other_flag {
                    let group = volume_groups.entry(key.clone()).or_insert_with(VolumeGroup::new);
                    group.add_record(i, exchtime_data[j], price_data[j]);
                }
            }
        } else {
            let group = volume_groups.entry(key).or_insert_with(VolumeGroup::new);
            group.add_record(i, exchtime_data[i], price_data[i]);
        }
    }
    
    // 2. 准备计算
    for (_, group) in volume_groups.iter_mut() {
        group.prepare_for_computation();
    }
    
    // 3. 初始化结果
    let mut results = vec![vec![f64::NAN; 22]; n];
    
    // 4. 批量计算每个组的指标
    for (_, group) in volume_groups.iter() {
        group.compute_all_indicators(&mut results, min_count);
    }
    
    let result_array = PyArray2::from_vec2(py, &results)?;
    let column_names = get_column_names();
    
    Ok((result_array.to_owned(), column_names))
}

#[pyfunction]
#[pyo3(signature = (volume, exchtime, price, flag, ask_order, bid_order, min_count=100, use_flag="ignore"))]
pub fn analyze_order_records_ultra_turbo(
    py: Python,
    volume: &PyArray1<f64>,
    exchtime: &PyArray1<f64>,
    price: &PyArray1<f64>,
    flag: &PyArray1<i32>,
    ask_order: &PyArray1<i64>,
    bid_order: &PyArray1<i64>,
    min_count: usize,
    use_flag: &str,
) -> PyResult<(Py<PyArray2<f64>>, Vec<String>)> {
    let volume_slice = volume.readonly();
    let exchtime_slice = exchtime.readonly();
    let price_slice = price.readonly();
    let flag_slice = flag.readonly();
    let ask_order_slice = ask_order.readonly();
    let bid_order_slice = bid_order.readonly();
    
    let volume_data = volume_slice.as_slice()?;
    let exchtime_data = exchtime_slice.as_slice()?;
    let price_data = price_slice.as_slice()?;
    let flag_data = flag_slice.as_slice()?;
    let ask_order_data = ask_order_slice.as_slice()?;
    let bid_order_data = bid_order_slice.as_slice()?;
    
    let n = volume_data.len();
    if n == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("输入数组不能为空"));
    }
    
    // 1. 高效订单聚合
    let mut orders: HashMap<(i64, i32), (f64, f64, f64, f64)> = HashMap::new();
    
    for i in 0..n {
        let current_flag = flag_data[i];
        let order_id = if current_flag == 66 { bid_order_data[i] } else { ask_order_data[i] };
        
        if order_id == 0 {
            continue;
        }
        
        let key = (order_id, current_flag);
        let entry = orders.entry(key).or_insert((0.0, 0.0, 0.0, 0.0));
        entry.0 += volume_data[i]; // total_volume
        entry.3 += volume_data[i] * price_data[i]; // total_amount
        entry.2 = entry.2.max(exchtime_data[i]); // last_time
    }
    
    // 计算加权平均价格
    for (_, entry) in orders.iter_mut() {
        if entry.0 > 0.0 {
            entry.1 = entry.3 / entry.0; // weighted_price = total_amount / total_volume
        }
    }
    
    // 2. 按订单volume分组进行批量计算
    let mut order_volume_groups: HashMap<String, VolumeGroup> = HashMap::new();
    let mut order_lookup: HashMap<usize, (i64, i32)> = HashMap::new();
    
    for ((order_id, order_flag), (order_volume, order_price, order_time, _)) in orders.iter() {
        let key = match use_flag {
            "same" => format!("{}_{}", order_volume, order_flag),
            "diff" => format!("{}_{}", order_volume, if *order_flag == 66 { 83 } else { 66 }),
            _ => order_volume.to_string(),
        };
        
        let group = order_volume_groups.entry(key).or_insert_with(VolumeGroup::new);
        let group_idx = group.indices.len();
        group.add_record(group_idx, *order_time, *order_price);
        order_lookup.insert(group_idx, (*order_id, *order_flag));
    }
    
    // 3. 准备计算
    for (_, group) in order_volume_groups.iter_mut() {
        group.prepare_for_computation();
    }
    
    // 4. 初始化结果
    let mut results = vec![vec![f64::NAN; 22]; n];
    let mut order_results: HashMap<(i64, i32), Vec<f64>> = HashMap::new();
    
    // 5. 批量计算订单指标
    for (_, group) in order_volume_groups.iter() {
        if group.indices.len() < min_count {
            continue;
        }
        
        // 临时结果存储
        let mut temp_results = vec![vec![f64::NAN; 22]; group.indices.len()];
        group.compute_all_indicators(&mut temp_results, min_count);
        
        // 将结果映射回订单
        for (i, result) in temp_results.iter().enumerate() {
            if let Some(&order_key) = order_lookup.get(&group.indices[i]) {
                order_results.insert(order_key, result.clone());
            }
        }
    }
    
    // 6. 将订单结果分配给交易记录
    for i in 0..n {
        let current_flag = flag_data[i];
        let order_id = if current_flag == 66 { bid_order_data[i] } else { ask_order_data[i] };
        
        if order_id != 0 {
            let order_key = (order_id, current_flag);
            if let Some(order_result) = order_results.get(&order_key) {
                results[i] = order_result.clone();
            }
        }
    }
    
    let result_array = PyArray2::from_vec2(py, &results)?;
    let column_names = get_column_names();
    
    Ok((result_array.to_owned(), column_names))
}

/// 高效计算所有22个指标
fn calculate_indicators(result_row: &mut [f64], time_distances: &[(f64, f64)], current_price: f64) {
    let total_count = time_distances.len();
    
    if total_count == 0 {
        return;
    }
    
    // 1. 最近一条记录的时间间隔
    result_row[0] = time_distances[0].0;
    
    // 2-11. 不同百分比的平均时间间隔
    let percentages = [0.01, 0.02, 0.03, 0.04, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50];
    for (idx, &pct) in percentages.iter().enumerate() {
        let count = ((total_count as f64 * pct).ceil() as usize).max(1).min(total_count);
        
        let avg_time = time_distances[..count]
            .iter()
            .map(|(time, _)| time)
            .sum::<f64>() / count as f64;
        
        result_row[idx + 1] = avg_time;
    }
    
    // 12. 所有记录的平均时间间隔
    let total_avg = time_distances.iter().map(|(time, _)| time).sum::<f64>() / total_count as f64;
    result_row[11] = total_avg;
    
    // 13-22. 价格分位数 - 优化版本
    let price_percentages = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00];
    for (idx, &pct) in price_percentages.iter().enumerate() {
        let count = if pct == 1.0 {
            total_count
        } else {
            ((total_count as f64 * pct).ceil() as usize).max(1).min(total_count)
        };
        
        // 收集价格并排序
        let mut prices: Vec<f64> = time_distances[..count]
            .iter()
            .map(|(_, price)| *price)
            .collect();
        prices.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        
        // 使用二分查找计算分位数
        let rank = prices.binary_search_by(|&p| {
            p.partial_cmp(&current_price).unwrap()
        }).unwrap_or_else(|e| e);
        
        let percentile = rank as f64 / prices.len() as f64;
        result_row[idx + 12] = percentile;
    }
}

fn get_column_names() -> Vec<String> {
    vec![
        "nearest_time_gap".to_string(),
        "avg_time_gap_1pct".to_string(),
        "avg_time_gap_2pct".to_string(),
        "avg_time_gap_3pct".to_string(),
        "avg_time_gap_4pct".to_string(),
        "avg_time_gap_5pct".to_string(),
        "avg_time_gap_10pct".to_string(),
        "avg_time_gap_20pct".to_string(),
        "avg_time_gap_30pct".to_string(),
        "avg_time_gap_40pct".to_string(),
        "avg_time_gap_50pct".to_string(),
        "avg_time_gap_all".to_string(),
        "price_percentile_10pct".to_string(),
        "price_percentile_20pct".to_string(),
        "price_percentile_30pct".to_string(),
        "price_percentile_40pct".to_string(),
        "price_percentile_50pct".to_string(),
        "price_percentile_60pct".to_string(),
        "price_percentile_70pct".to_string(),
        "price_percentile_80pct".to_string(),
        "price_percentile_90pct".to_string(),
        "price_percentile_all".to_string(),
    ]
}