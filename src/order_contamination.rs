use pyo3::prelude::*;
use numpy::{PyArray1, PyReadonlyArray1};
use std::collections::HashMap;
use rayon::prelude::*;

/// 优化版订单浸染函数 - 高性能单线程版本
/// 
/// # 参数
/// - exchtime: 成交时间数组（纳秒）
/// - order: 订单编号数组
/// - volume: 成交量数组  
/// - top_percentile: 大单百分比阈值 (0.0-1.0)，默认0.1表示前10%
/// - time_window_ns: 时间窗口（纳秒），默认1秒=1_000_000_000ns
/// 
/// # 返回
/// 浸染后的订单编号数组
#[pyfunction]
#[pyo3(signature = (exchtime, order, volume, top_percentile = 0.1, time_window_ns = 1_000_000_000_i64))]
pub fn order_contamination(
    exchtime: PyReadonlyArray1<i64>,
    order: PyReadonlyArray1<i64>, 
    volume: PyReadonlyArray1<i64>,
    top_percentile: f64,
    time_window_ns: i64,
) -> PyResult<Py<PyArray1<i64>>> {
    let py = unsafe { Python::assume_gil_acquired() };
    
    let exchtime = exchtime.as_array();
    let order = order.as_array();
    let volume = volume.as_array();
    
    let n = exchtime.len();
    if n != order.len() || n != volume.len() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "输入数组长度不一致"
        ));
    }
    
    if n == 0 {
        return Ok(PyArray1::from_vec(py, vec![]).to_owned());
    }
    
    // 1. 快速聚合订单成交量（使用容量预分配）
    let mut order_volumes: HashMap<i64, i64> = HashMap::with_capacity(n / 4);
    for i in 0..n {
        *order_volumes.entry(order[i]).or_insert(0) += volume[i];
    }
    
    // 2. 快速找到大单阈值（避免完整排序）
    let mut volumes: Vec<i64> = order_volumes.values().cloned().collect();
    let top_count = ((volumes.len() as f64 * top_percentile).ceil() as usize)
        .max(1)
        .min(volumes.len()); // 确保不超过数组长度
    
    // 使用nth_element获取阈值，比完整排序快
    let threshold = if top_count > 0 && top_count <= volumes.len() {
        volumes.select_nth_unstable_by(top_count - 1, |a, b| b.cmp(a));
        volumes[top_count - 1]
    } else {
        // 如果计算出错，回退到完整排序
        volumes.sort_unstable_by(|a, b| b.cmp(a));
        volumes.get(top_count.saturating_sub(1)).cloned().unwrap_or(0)
    };
    
    // 3. 预标记大单位置（避免重复HashSet查找）
    let mut is_large_order = vec![false; n];
    let mut large_order_positions = Vec::with_capacity(n / 10);
    
    for i in 0..n {
        if let Some(&total_vol) = order_volumes.get(&order[i]) {
            if total_vol >= threshold {
                is_large_order[i] = true;
                large_order_positions.push(i);
            }
        }
    }
    
    // 4. 创建时间索引用于快速查找
    let mut time_indices: Vec<usize> = (0..n).collect();
    time_indices.sort_unstable_by(|&a, &b| exchtime[a].cmp(&exchtime[b]));
    
    let mut result = order.to_vec();
    
    // 5. 优化的浸染过程
    for &pos in &large_order_positions {
        let center_time = exchtime[pos];
        let large_order_id = order[pos];
        let start_time = center_time - time_window_ns;
        let end_time = center_time + time_window_ns;
        
        // 使用二分查找找到时间窗口边界
        let start_idx = time_indices.binary_search_by(|&i| {
            if exchtime[i] < start_time { std::cmp::Ordering::Less }
            else { std::cmp::Ordering::Greater }
        }).unwrap_or_else(|i| i);
        
        let end_idx = time_indices.binary_search_by(|&i| {
            if exchtime[i] <= end_time { std::cmp::Ordering::Less }
            else { std::cmp::Ordering::Greater }
        }).unwrap_or_else(|i| i);
        
        // 只处理时间窗口内的交易
        for &i in &time_indices[start_idx..end_idx] {
            result[i] = large_order_id;
        }
    }
    
    Ok(PyArray1::from_vec(py, result).to_owned())
}

/// 并行版本的订单浸染函数（使用5核心，适用于大数据量）
#[pyfunction]
#[pyo3(signature = (exchtime, order, volume, top_percentile = 0.1, time_window_ns = 1_000_000_000_i64))]
pub fn order_contamination_parallel(
    exchtime: PyReadonlyArray1<i64>,
    order: PyReadonlyArray1<i64>,
    volume: PyReadonlyArray1<i64>, 
    top_percentile: f64,
    time_window_ns: i64,
) -> PyResult<Py<PyArray1<i64>>> {
    let py = unsafe { Python::assume_gil_acquired() };
    
    let exchtime = exchtime.as_array();
    let order = order.as_array();
    let volume = volume.as_array();
    
    let n = exchtime.len();
    if n != order.len() || n != volume.len() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "输入数组长度不一致"
        ));
    }
    
    // 创建5核心的线程池
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(5)
        .build()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
            format!("创建线程池失败: {}", e)
        ))?;
    
    // 1. 并行聚合订单成交量（使用5核心）
    let order_volumes: HashMap<i64, i64> = pool.install(|| {
        (0..n)
            .into_par_iter()
            .map(|i| (order[i], volume[i]))
            .fold(
                HashMap::new,
                |mut acc, (ord, vol)| {
                    *acc.entry(ord).or_insert(0) += vol;
                    acc
                }
            )
            .reduce(
                HashMap::new,
                |mut acc1, acc2| {
                    for (ord, vol) in acc2 {
                        *acc1.entry(ord).or_insert(0) += vol;
                    }
                    acc1
                }
            )
    });
    
    // 2. 找到前top_percentile%的大单（使用5核心）
    let mut volumes: Vec<i64> = order_volumes.values().cloned().collect();
    pool.install(|| {
        volumes.par_sort_unstable_by(|a, b| b.cmp(a));
    });
    
    let top_count = ((volumes.len() as f64 * top_percentile).ceil() as usize).max(1);
    let threshold = volumes.get(top_count - 1).cloned().unwrap_or(0);
    
    let large_orders: std::collections::HashSet<i64> = pool.install(|| {
        order_volumes
            .par_iter()
            .filter(|(_, &vol)| vol >= threshold)
            .map(|(&ord, _)| ord)
            .collect()
    });
    
    // 3. 创建结果数组并进行浸染
    let mut result = order.to_vec();
    
    // 找到所有大单的位置
    let large_order_positions: Vec<usize> = (0..n)
        .filter(|&i| large_orders.contains(&order[i]))
        .collect();
    
    // 对每个大单位置进行浸染（使用5核心）
    for &pos in &large_order_positions {
        let center_time = exchtime[pos];
        let large_order_id = order[pos];
        
        // 并行处理时间窗口内的所有交易
        let updates: Vec<(usize, i64)> = pool.install(|| {
            (0..n)
                .into_par_iter()
                .filter_map(|i| {
                    let time_diff = (exchtime[i] - center_time).abs();
                    if time_diff <= time_window_ns {
                        Some((i, large_order_id))
                    } else {
                        None
                    }
                })
                .collect()
        });
        
        // 应用更新
        for (i, new_order) in updates {
            result[i] = new_order;
        }
    }
    
    Ok(PyArray1::from_vec(py, result).to_owned())
}