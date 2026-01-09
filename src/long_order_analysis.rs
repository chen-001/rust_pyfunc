use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use std::collections::HashMap;

#[derive(Debug, Clone)]
struct OrderInfo {
    first_idx: usize,
    last_idx: usize,
    total_volume: f64,
    exchtime_values: Vec<i64>,
}

#[derive(Debug, Clone)]
struct LongOrderInfo {
    order_id: i64,
    first_idx: usize,
    last_idx: usize,
    total_volume: f64,
    time_span: i64,          // 时间跨度（纳秒）
    occurrence_count: usize, // 出现次数
    ratio: f64,              // 其他订单成交量与该订单成交量的比值
}

#[derive(Debug, Clone, Copy)]
enum OrderType {
    TimeLong,
    CountLong,
    BothLong,
}

/// 根据top_ratio筛选最漫长的一部分订单
fn filter_long_orders(
    orders: &[LongOrderInfo],
    top_ratio: f64,
    order_type: OrderType,
) -> Vec<LongOrderInfo> {
    if orders.is_empty() || top_ratio >= 1.0 {
        return orders.to_vec();
    }

    if top_ratio <= 0.0 {
        return Vec::new();
    }

    // 复制一份用于排序
    let mut sorted_orders = orders.to_vec();

    // 根据订单类型选择排序键
    match order_type {
        OrderType::TimeLong => {
            // 时间漫长订单按时间跨度降序排序
            sorted_orders.sort_by(|a, b| b.time_span.cmp(&a.time_span));
        }
        OrderType::CountLong => {
            // 次数漫长订单按出现次数降序排序
            sorted_orders.sort_by(|a, b| b.occurrence_count.cmp(&a.occurrence_count));
        }
        OrderType::BothLong => {
            // 两者都漫长订单按时间跨度降序排序
            sorted_orders.sort_by(|a, b| b.time_span.cmp(&a.time_span));
        }
    }

    // 计算要保留的订单数量
    let keep_count = (orders.len() as f64 * top_ratio).ceil() as usize;

    // 截取前 keep_count 个订单
    sorted_orders.into_iter().take(keep_count).collect()
}

/// 分析漫长订单并计算比值
///
/// 参数:
/// - exchtime: 交易时间数组(纳秒)
/// - order: 订单编号数组
/// - volume: 成交量数组
/// - top_ratio: 可选参数，表示只计算最漫长的一部分订单的比例(0.0-1.0)
///              默认为1.0表示计算所有订单，0.5表示只计算最漫长的一半订单
///
/// 返回: (时间漫长比值序列, 次数漫长比值序列, 两者都漫长比值序列,
///       时间漫长总比值, 次数漫长总比值, 两者都漫长总比值)
#[pyfunction]
pub fn analyze_long_orders<'py>(
    py: Python<'py>,
    exchtime: PyReadonlyArray1<i64>,
    order: PyReadonlyArray1<i64>,
    volume: PyReadonlyArray1<f64>,
    top_ratio: Option<f64>,
) -> PyResult<(
    Py<PyArray1<f64>>,
    Py<PyArray1<f64>>,
    Py<PyArray1<f64>>,
    Py<PyArray1<f64>>,
    Py<PyArray1<f64>>,
    Py<PyArray1<f64>>,
)> {
    let exchtime_slice = exchtime.as_slice()?;
    let order_slice = order.as_slice()?;
    let volume_slice = volume.as_slice()?;

    if exchtime_slice.len() != order_slice.len() || order_slice.len() != volume_slice.len() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "输入数组长度必须相等",
        ));
    }

    let n = exchtime_slice.len();
    if n == 0 {
        let empty_array = vec![].into_pyarray(py);
        return Ok((
            empty_array.into(),
            empty_array.into(),
            empty_array.into(),
            empty_array.into(),
            empty_array.into(),
            empty_array.into(),
        ));
    }

    // 处理 top_ratio 参数
    let top_ratio = top_ratio.unwrap_or(1.0);
    if top_ratio < 0.0 || top_ratio > 1.0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "top_ratio 必须在 0.0 到 1.0 之间",
        ));
    }

    // 第一步：收集每个订单的信息
    let mut order_map: HashMap<i64, OrderInfo> = HashMap::new();

    for (idx, ((&order_id, &exchtime_val), &volume_val)) in order_slice
        .iter()
        .zip(exchtime_slice.iter())
        .zip(volume_slice.iter())
        .enumerate()
    {
        let info = order_map.entry(order_id).or_insert(OrderInfo {
            first_idx: idx,
            last_idx: idx,
            total_volume: 0.0,
            exchtime_values: Vec::new(),
        });

        info.last_idx = idx;
        info.total_volume += volume_val;
        info.exchtime_values.push(exchtime_val);
    }

    // 第二步：收集所有候选的漫长订单信息
    let mut time_long_orders = Vec::new();
    let mut count_long_orders = Vec::new();
    let mut both_long_orders = Vec::new();

    for (&order_id, order_info) in &order_map {
        // 跳过只有一个出现记录的订单
        if order_info.first_idx == order_info.last_idx {
            continue;
        }

        let is_time_long = {
            let mut unique_times = std::collections::HashSet::new();
            for &time in &order_info.exchtime_values {
                unique_times.insert(time);
            }
            unique_times.len() > 1
        };

        let is_count_long =
            order_info.last_idx - order_info.first_idx > order_info.exchtime_values.len() - 1;

        // 如果是漫长订单，收集信息
        if is_time_long || is_count_long {
            let other_orders_volume = calculate_other_orders_volume(
                order_info.first_idx,
                order_info.last_idx,
                order_id,
                order_slice,
                volume_slice,
            );

            let ratio = if order_info.total_volume > 0.0 {
                other_orders_volume / order_info.total_volume
            } else {
                0.0
            };

            // 计算时间跨度和出现次数
            let time_span = *order_info.exchtime_values.iter().max().unwrap()
                - *order_info.exchtime_values.iter().min().unwrap();
            let occurrence_count = order_info.exchtime_values.len();

            let long_order_info = LongOrderInfo {
                order_id,
                first_idx: order_info.first_idx,
                last_idx: order_info.last_idx,
                total_volume: order_info.total_volume,
                time_span,
                occurrence_count,
                ratio,
            };

            if is_time_long {
                time_long_orders.push(long_order_info.clone());
            }
            if is_count_long {
                count_long_orders.push(long_order_info.clone());
            }
            if is_time_long && is_count_long {
                both_long_orders.push(long_order_info.clone());
            }
        }
    }

    // 第三步：根据 top_ratio 进行筛选
    let final_time_orders = filter_long_orders(&time_long_orders, top_ratio, OrderType::TimeLong);
    let final_count_orders =
        filter_long_orders(&count_long_orders, top_ratio, OrderType::CountLong);
    let final_both_orders = filter_long_orders(&both_long_orders, top_ratio, OrderType::BothLong);

    // 第四步：计算最终的比值序列
    let mut time_long_ratios = Vec::new();
    let mut count_long_ratios = Vec::new();
    let mut both_long_ratios = Vec::new();

    // 用于计算总比值
    let mut total_time_long_other_volume = 0.0;
    let mut total_time_long_order_volume = 0.0;
    let mut total_count_long_other_volume = 0.0;
    let mut total_count_long_order_volume = 0.0;
    let mut total_both_long_other_volume = 0.0;
    let mut total_both_long_order_volume = 0.0;

    for order_info in final_time_orders {
        time_long_ratios.push(order_info.ratio);
        let other_volume = calculate_other_orders_volume(
            order_info.first_idx,
            order_info.last_idx,
            order_info.order_id,
            order_slice,
            volume_slice,
        );
        total_time_long_other_volume += other_volume;
        total_time_long_order_volume += order_info.total_volume;
    }

    for order_info in final_count_orders {
        count_long_ratios.push(order_info.ratio);
        let other_volume = calculate_other_orders_volume(
            order_info.first_idx,
            order_info.last_idx,
            order_info.order_id,
            order_slice,
            volume_slice,
        );
        total_count_long_other_volume += other_volume;
        total_count_long_order_volume += order_info.total_volume;
    }

    for order_info in final_both_orders {
        both_long_ratios.push(order_info.ratio);
        let other_volume = calculate_other_orders_volume(
            order_info.first_idx,
            order_info.last_idx,
            order_info.order_id,
            order_slice,
            volume_slice,
        );
        total_both_long_other_volume += other_volume;
        total_both_long_order_volume += order_info.total_volume;
    }

    // 计算总比值
    let time_long_total_ratio = if total_time_long_order_volume > 0.0 {
        total_time_long_other_volume / total_time_long_order_volume
    } else {
        0.0
    };

    let count_long_total_ratio = if total_count_long_order_volume > 0.0 {
        total_count_long_other_volume / total_count_long_order_volume
    } else {
        0.0
    };

    let both_long_total_ratio = if total_both_long_order_volume > 0.0 {
        total_both_long_other_volume / total_both_long_order_volume
    } else {
        0.0
    };

    let time_array = time_long_ratios.into_pyarray(py);
    let count_array = count_long_ratios.into_pyarray(py);
    let both_array = both_long_ratios.into_pyarray(py);

    let time_total_array = vec![time_long_total_ratio].into_pyarray(py);
    let count_total_array = vec![count_long_total_ratio].into_pyarray(py);
    let both_total_array = vec![both_long_total_ratio].into_pyarray(py);

    Ok((
        time_array.into(),
        count_array.into(),
        both_array.into(),
        time_total_array.into(),
        count_total_array.into(),
        both_total_array.into(),
    ))
}

/// 计算指定范围内其他订单的成交量总和
fn calculate_other_orders_volume(
    start_idx: usize,
    end_idx: usize,
    target_order_id: i64,
    order_slice: &[i64],
    volume_slice: &[f64],
) -> f64 {
    let mut other_volume = 0.0;

    for idx in start_idx..=end_idx {
        if order_slice[idx] != target_order_id {
            other_volume += volume_slice[idx];
        }
    }

    other_volume
}

/// Python版本的漫长订单分析(用于测试对比)
#[pyfunction]
pub fn analyze_long_orders_python<'py>(
    py: Python<'py>,
    exchtime: PyReadonlyArray1<i64>,
    order: PyReadonlyArray1<i64>,
    volume: PyReadonlyArray1<f64>,
    top_ratio: Option<f64>,
) -> PyResult<(
    Py<PyArray1<f64>>,
    Py<PyArray1<f64>>,
    Py<PyArray1<f64>>,
    Py<PyArray1<f64>>,
    Py<PyArray1<f64>>,
    Py<PyArray1<f64>>,
)> {
    let exchtime_slice = exchtime.as_slice()?;
    let order_slice = order.as_slice()?;
    let volume_slice = volume.as_slice()?;

    let n = exchtime_slice.len();
    if n == 0 {
        let empty_array = vec![].into_pyarray(py);
        return Ok((
            empty_array.into(),
            empty_array.into(),
            empty_array.into(),
            empty_array.into(),
            empty_array.into(),
            empty_array.into(),
        ));
    }

    // 处理 top_ratio 参数
    let top_ratio = top_ratio.unwrap_or(1.0);
    if top_ratio < 0.0 || top_ratio > 1.0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "top_ratio 必须在 0.0 到 1.0 之间",
        ));
    }

    // 构建订单到索引的映射
    let mut order_indices: HashMap<i64, Vec<usize>> = HashMap::new();
    for (i, &order_id) in order_slice.iter().enumerate() {
        order_indices.entry(order_id).or_insert(Vec::new()).push(i);
    }

    let mut time_long_ratios = Vec::new();
    let mut count_long_ratios = Vec::new();
    let mut both_long_ratios = Vec::new();

    // 用于计算总比值
    let mut total_time_long_other_volume = 0.0;
    let mut total_time_long_order_volume = 0.0;
    let mut total_count_long_other_volume = 0.0;
    let mut total_count_long_order_volume = 0.0;
    let mut total_both_long_other_volume = 0.0;
    let mut total_both_long_order_volume = 0.0;

    for (&order_id, indices) in &order_indices {
        if indices.len() <= 1 {
            continue;
        }

        // 检查时间漫长
        let mut unique_times = std::collections::HashSet::new();
        for &idx in indices {
            unique_times.insert(exchtime_slice[idx]);
        }
        let is_time_long = unique_times.len() > 1;

        // 检查次数漫长
        let first_idx = indices[0];
        let last_idx = indices[indices.len() - 1];
        let is_count_long = (last_idx - first_idx + 1) > indices.len();

        if is_time_long || is_count_long {
            // 计算比值
            let order_volume: f64 = indices.iter().map(|&idx| volume_slice[idx]).sum();

            let other_volume: f64 = (first_idx..=last_idx)
                .filter(|&idx| order_slice[idx] != order_id)
                .map(|idx| volume_slice[idx])
                .sum();

            let ratio = if order_volume > 0.0 {
                other_volume / order_volume
            } else {
                0.0
            };

            if is_time_long {
                time_long_ratios.push(ratio);
                total_time_long_other_volume += other_volume;
                total_time_long_order_volume += order_volume;
            }

            if is_count_long {
                count_long_ratios.push(ratio);
                total_count_long_other_volume += other_volume;
                total_count_long_order_volume += order_volume;
            }

            if is_time_long && is_count_long {
                both_long_ratios.push(ratio);
                total_both_long_other_volume += other_volume;
                total_both_long_order_volume += order_volume;
            }
        }
    }

    // 计算总比值
    let time_long_total_ratio = if total_time_long_order_volume > 0.0 {
        total_time_long_other_volume / total_time_long_order_volume
    } else {
        0.0
    };

    let count_long_total_ratio = if total_count_long_order_volume > 0.0 {
        total_count_long_other_volume / total_count_long_order_volume
    } else {
        0.0
    };

    let both_long_total_ratio = if total_both_long_order_volume > 0.0 {
        total_both_long_other_volume / total_both_long_order_volume
    } else {
        0.0
    };

    let time_array = time_long_ratios.into_pyarray(py);
    let count_array = count_long_ratios.into_pyarray(py);
    let both_array = both_long_ratios.into_pyarray(py);

    let time_total_array = vec![time_long_total_ratio].into_pyarray(py);
    let count_total_array = vec![count_long_total_ratio].into_pyarray(py);
    let both_total_array = vec![both_long_total_ratio].into_pyarray(py);

    Ok((
        time_array.into(),
        count_array.into(),
        both_array.into(),
        time_total_array.into(),
        count_total_array.into(),
        both_total_array.into(),
    ))
}
