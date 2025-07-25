use pyo3::prelude::*;
use numpy::{PyArray2, PyReadonlyArray1};
use std::collections::HashMap;
use rayon::prelude::*;

/// 超级高速订单邻域分析函数 - 为13万数据优化
/// 
/// 激进优化策略：
/// 1. 预计算邻域索引表
/// 2. 批量SIMD计算
/// 3. 内存预分配避免动态分配
/// 4. 分层并行处理
/// 5. 零拷贝数据传递
#[pyfunction]
#[pyo3(signature = (ask_order, bid_order, volume, exchtime, neighborhood_type = "fixed", fixed_range = 1000, percentage_range = 10.0))]
pub fn order_neighborhood_analysis(
    ask_order: PyReadonlyArray1<i64>,
    bid_order: PyReadonlyArray1<i64>,
    volume: PyReadonlyArray1<f64>,
    exchtime: PyReadonlyArray1<i64>,
    neighborhood_type: &str,
    fixed_range: i64,
    percentage_range: f64,
) -> PyResult<(Py<PyArray2<f64>>, Vec<String>)> {
    let py = unsafe { Python::assume_gil_acquired() };
    
    let ask_slice = ask_order.as_array();
    let bid_slice = bid_order.as_array();
    let volume_slice = volume.as_array();
    let exchtime_slice = exchtime.as_array();
    
    let n = ask_slice.len();
    if n != bid_slice.len() || n != volume_slice.len() || n != exchtime_slice.len() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "输入数组长度不一致"
        ));
    }
    
    if n == 0 {
        let empty_result = PyArray2::zeros(py, (0, 18), false);
        return Ok((empty_result.to_owned(), get_column_names()));
    }
    
    // 调用超级高速算法
    let results = hyper_optimized_analysis(
        ask_slice.as_slice().unwrap(),
        bid_slice.as_slice().unwrap(),
        volume_slice.as_slice().unwrap(),
        exchtime_slice.as_slice().unwrap(),
        neighborhood_type,
        fixed_range,
        percentage_range,
    );
    
    // 超高效结果转换
    let num_orders = results.len();
    let result_array = PyArray2::zeros(py, (num_orders, 18), false);
    
    // 直接填充结果 - 避免并行闭包中的可变借用问题
    {
        let mut result_slice = unsafe { result_array.as_array_mut() };
        for (i, &row) in results.iter().enumerate() {
            for (j, &value) in row.iter().enumerate() {
                unsafe {
                    *result_slice.uget_mut((i, j)) = value;
                }
            }
        }
    }
    
    Ok((result_array.to_owned(), get_column_names()))
}

/// 超紧凑订单结构 - 32字节对齐
#[derive(Clone, Copy)]
#[repr(C, align(32))]
struct UltraCompactOrder {
    id: i64,
    volume: f32,
    time: i64,
    order_type: u8,
    _padding: [u8; 7], // 32字节对齐
}

/// 邻域索引表 - 预计算所有邻域关系
struct NeighborhoodIndex {
    same_neighbors: Vec<Vec<u32>>, // 使用u32节省内存
    diff_neighbors: Vec<Vec<u32>>,
}

/// 超高速分析算法 - 针对大数据集优化
fn hyper_optimized_analysis(
    ask_order: &[i64],
    bid_order: &[i64],
    volume: &[f64],
    exchtime: &[i64],
    neighborhood_type: &str,
    fixed_range: i64,
    percentage_range: f64,
) -> Vec<[f64; 18]> {
    
    // 8核并行池
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(8)
        .build()
        .unwrap_or_else(|_| rayon::ThreadPoolBuilder::new().build().unwrap());
    
    pool.install(|| {
        // Step 1: 超快速聚合 - 并行分块
        let orders = ultra_fast_aggregate(ask_order, bid_order, volume, exchtime);
        
        if orders.is_empty() {
            return Vec::new();
        }
        
        // Step 2: 预计算邻域索引表
        let neighborhood_index = precompute_neighborhoods(&orders, neighborhood_type, fixed_range, percentage_range);
        
        // Step 3: 批量向量化计算
        vectorized_batch_compute(&orders, &neighborhood_index)
    })
}

/// 超快速聚合 - 最小化内存分配和Hash计算
fn ultra_fast_aggregate(
    ask_order: &[i64],
    bid_order: &[i64],
    volume: &[f64],
    exchtime: &[i64],
) -> Vec<UltraCompactOrder> {
    
    let n = ask_order.len();
    
    // 预估容量避免重新分配
    let estimated_orders = n / 10; // 假设平均每10个原始记录对应1个唯一订单
    let mut order_map: HashMap<i64, (f32, i64, u8)> = HashMap::with_capacity(estimated_orders);
    
    // 单线程聚合 - 减少内存开销
    for i in 0..n {
        // 处理卖单
        let ask_id = ask_order[i];
        if ask_id != 0 {
            let vol = volume[i] as f32;
            let time = exchtime[i];
            
            match order_map.get_mut(&ask_id) {
                Some(entry) => {
                    entry.0 += vol;
                    if time > entry.1 { entry.1 = time; }
                },
                None => {
                    order_map.insert(ask_id, (vol, time, 1));
                }
            }
        }
        
        // 处理买单
        let bid_id = bid_order[i];
        if bid_id != 0 {
            let vol = volume[i] as f32;
            let time = exchtime[i];
            
            match order_map.get_mut(&bid_id) {
                Some(entry) => {
                    entry.0 += vol;
                    if time > entry.1 { entry.1 = time; }
                },
                None => {
                    order_map.insert(bid_id, (vol, time, 0));
                }
            }
        }
    }
    
    // 预分配并直接填充
    let mut orders = Vec::with_capacity(order_map.len());
    for (id, (vol, time, typ)) in order_map {
        orders.push(UltraCompactOrder {
            id,
            volume: vol,
            time,
            order_type: typ,
            _padding: [0; 7],
        });
    }
    
    // 使用更快的排序
    orders.sort_unstable_by_key(|o| o.id);
    orders
}

/// 智能邻域索引表 - 动态范围限制
fn precompute_neighborhoods(
    orders: &[UltraCompactOrder],
    neighborhood_type: &str,
    fixed_range: i64,
    percentage_range: f64,
) -> NeighborhoodIndex {
    
    let orders_len = orders.len();
    let mut same_neighbors = vec![Vec::new(); orders_len];
    let mut diff_neighbors = vec![Vec::new(); orders_len];
    
    // 移除邻居数量限制，完全按照邻域范围计算所有邻居
    
    // 并行预计算邻域关系 - 增加智能限制
    same_neighbors.par_iter_mut()
        .zip(diff_neighbors.par_iter_mut())
        .enumerate()
        .for_each(|(target_idx, (same_vec, diff_vec))| {
            let target_order = &orders[target_idx];
            
            // 动态调整邻域范围 - 如果订单密度太高则缩小范围
            let base_range = match neighborhood_type {
                "fixed" => fixed_range,
                "percentage" => {
                    ((target_order.id as f64 * percentage_range / 100.0).abs() as i64).max(1)
                },
                _ => return,
            };
            
            // 移除有问题的动态范围调整，使用原始范围
            // 保持用户指定的邻域范围不变，只在邻居收集时限制数量
            let actual_range = base_range;
            
            let (min_id, max_id) = (target_order.id - actual_range, target_order.id + actual_range);
            
            // 二分查找边界
            let start_idx = binary_search_ge_ultra(orders, min_id);
            let end_idx = binary_search_le_ultra(orders, max_id);
            
            if start_idx <= end_idx {
                // 预分配空间，提高性能
                let range_size = (end_idx - start_idx + 1).min(orders_len);
                same_vec.reserve(range_size / 4);
                diff_vec.reserve(range_size / 4);
                
                // 收集所有在邻域范围内的邻居，不做数量限制
                for i in start_idx..=end_idx.min(orders_len - 1) {
                    let neighbor = &orders[i];
                    if neighbor.id == target_order.id { continue; }
                    
                    if neighbor.order_type == target_order.order_type {
                        same_vec.push(i as u32);
                    } else {
                        diff_vec.push(i as u32);
                    }
                }
            }
        });
    
    NeighborhoodIndex { same_neighbors, diff_neighbors }
}

/// 向量化批量计算 - SIMD友好
fn vectorized_batch_compute(
    orders: &[UltraCompactOrder],
    neighborhood_index: &NeighborhoodIndex,
) -> Vec<[f64; 18]> {
    
    let orders_len = orders.len();
    
    // 并行处理大块
    (0..orders_len)
        .into_par_iter()
        .map(|target_idx| {
            let target_order = &orders[target_idx];
            let same_neighbors = &neighborhood_index.same_neighbors[target_idx];
            let diff_neighbors = &neighborhood_index.diff_neighbors[target_idx];
            
            ultra_fast_compute_metrics(target_order, orders, same_neighbors, diff_neighbors)
        })
        .collect()
}

/// 超快速指标计算 - 内联所有操作，极致性能优化
#[inline(always)]
fn ultra_fast_compute_metrics(
    target_order: &UltraCompactOrder,
    all_orders: &[UltraCompactOrder],
    same_neighbor_indices: &[u32],
    diff_neighbor_indices: &[u32],
) -> [f64; 18] {
    
    let mut metrics = [0.0; 18];
    metrics[0] = target_order.id as f64;
    metrics[1] = target_order.order_type as f64;
    
    let same_count = same_neighbor_indices.len();
    let diff_count = diff_neighbor_indices.len();
    
    if same_count == 0 && diff_count == 0 {
        return metrics;
    }
    
    // 处理所有邻居，不做数量限制
    let same_limit = same_count;
    let diff_limit = diff_count;
    
    // 超快速SIMD友好计算
    let target_vol = target_order.volume as f64;
    let target_time = target_order.time;
    let inv_target_vol = if target_vol > 0.0 { 1.0 / target_vol } else { 0.0 };
    
    // 同方向计算 - 使用迭代器链优化
    let (same_vol_sum, same_time_sum) = if same_limit > 0 {
        same_neighbor_indices[..same_limit].iter()
            .map(|&idx| {
                let neighbor = unsafe { all_orders.get_unchecked(idx as usize) };
                let vol = neighbor.volume as f64;
                let time_diff = (neighbor.time - target_time).unsigned_abs() as f64;
                (vol, time_diff)
            })
            .fold((0.0, 0.0), |(vol_acc, time_acc), (vol, time_diff)| {
                (vol_acc + vol, time_acc + time_diff)
            })
    } else {
        (0.0, 0.0)
    };
    
    // 不同方向计算
    let (diff_vol_sum, diff_time_sum) = if diff_limit > 0 {
        diff_neighbor_indices[..diff_limit].iter()
            .map(|&idx| {
                let neighbor = unsafe { all_orders.get_unchecked(idx as usize) };
                let vol = neighbor.volume as f64;
                let time_diff = (neighbor.time - target_time).unsigned_abs() as f64;
                (vol, time_diff)
            })
            .fold((0.0, 0.0), |(vol_acc, time_acc), (vol, time_diff)| {
                (vol_acc + vol, time_acc + time_diff)
            })
    } else {
        (0.0, 0.0)
    };
    
    // 批量计算所有指标
    let same_count_f = same_count as f64;
    let diff_count_f = diff_count as f64;
    let total_vol = same_vol_sum + diff_vol_sum;
    let total_count = same_count_f + diff_count_f;
    
    if target_vol > 0.0 {
        metrics[2] = same_vol_sum * inv_target_vol;
        metrics[3] = diff_vol_sum * inv_target_vol;
        metrics[4] = if same_count > 0 { same_vol_sum / same_count_f * inv_target_vol } else { 0.0 };
        metrics[5] = if diff_count > 0 { diff_vol_sum / diff_count_f * inv_target_vol } else { 0.0 };
        metrics[6] = total_vol * inv_target_vol;
        metrics[7] = if total_count > 0.0 { total_vol / total_count * inv_target_vol } else { 0.0 };
    }
    
    metrics[8] = same_count_f;
    metrics[9] = diff_count_f;
    
    // 时间指标 - 预计算常数
    const NS_TO_SEC: f64 = 1e-9;
    metrics[10] = same_time_sum * NS_TO_SEC;
    metrics[11] = diff_time_sum * NS_TO_SEC;
    metrics[12] = if same_count > 0 { same_time_sum / same_count_f * NS_TO_SEC } else { 0.0 };
    metrics[13] = if diff_count > 0 { diff_time_sum / diff_count_f * NS_TO_SEC } else { 0.0 };
    
    // 恢复相关系数计算 - 仅在有足够邻居时计算
    if same_limit >= 2 {
        let same_indices = &same_neighbor_indices[..same_limit];
        metrics[14] = fast_correlation(target_order, all_orders, same_indices, true);
        metrics[16] = fast_correlation(target_order, all_orders, same_indices, false);
    }
    
    if diff_limit >= 2 {
        let diff_indices = &diff_neighbor_indices[..diff_limit];
        metrics[15] = fast_correlation(target_order, all_orders, diff_indices, true);
        metrics[17] = fast_correlation(target_order, all_orders, diff_indices, false);
    }
    
    metrics
}

/// 快速相关系数计算 - 修复数值溢出问题
#[inline(always)]
fn fast_correlation(
    target_order: &UltraCompactOrder,
    all_orders: &[UltraCompactOrder],
    neighbor_indices: &[u32],
    use_time_diff: bool,
) -> f64 {
    let n = neighbor_indices.len();
    if n < 2 { return 0.0; }
    
    let target_time = target_order.time;
    
    // 收集x和y值，避免大数乘法
    let mut x_values = Vec::with_capacity(n);
    let mut y_values = Vec::with_capacity(n);
    
    for &idx in neighbor_indices {
        let neighbor = unsafe { all_orders.get_unchecked(idx as usize) };
        let x = if use_time_diff {
            (neighbor.time - target_time).unsigned_abs() as f64
        } else {
            // 对绝对时间进行归一化，避免大数计算
            (neighbor.time as f64) / 1_000_000_000.0  // 转换为秒
        };
        let y = neighbor.volume as f64;
        x_values.push(x);
        y_values.push(y);
    }
    
    // 计算均值
    let n_f = n as f64;
    let mean_x = x_values.iter().sum::<f64>() / n_f;
    let mean_y = y_values.iter().sum::<f64>() / n_f;
    
    // 计算协方差和方差
    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;
    
    for i in 0..n {
        let dx = x_values[i] - mean_x;
        let dy = y_values[i] - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }
    
    let std_x = var_x.sqrt();
    let std_y = var_y.sqrt();
    
    if std_x < 1e-15 || std_y < 1e-15 { 
        0.0 
    } else { 
        cov / (std_x * std_y) 
    }
}

/// 超快速二分查找 - 第一个 >= target 的位置
#[inline(always)]
fn binary_search_ge_ultra(orders: &[UltraCompactOrder], target: i64) -> usize {
    let mut left = 0;
    let mut right = orders.len();
    
    while left < right {
        let mid = (left + right) >> 1; // 位移比除法更快
        if orders[mid].id < target {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    
    left
}

/// 超快速二分查找 - 最后一个 <= target 的位置
#[inline(always)]
fn binary_search_le_ultra(orders: &[UltraCompactOrder], target: i64) -> usize {
    let mut left = 0;
    let mut right = orders.len();
    
    while left < right {
        let mid = (left + right) >> 1;
        if orders[mid].id <= target {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    
    if left > 0 { left - 1 } else { 0 }
}

/// 获取列名 - 简洁的中文名称
fn get_column_names() -> Vec<String> {
    vec![
        "订单编号".to_string(),
        "订单类型".to_string(),
        "同向成交量比".to_string(),
        "异向成交量比".to_string(),
        "同向均量比".to_string(),
        "异向均量比".to_string(),
        "总成交量比".to_string(),
        "总均量比".to_string(),
        "同向邻居数".to_string(),
        "异向邻居数".to_string(),
        "同向时差总和".to_string(),
        "异向时差总和".to_string(),
        "同向时差均值".to_string(),
        "异向时差均值".to_string(),
        "同向时差量相关".to_string(),
        "异向时差量相关".to_string(),
        "同向时间量相关".to_string(),
        "异向时间量相关".to_string(),
    ]
}