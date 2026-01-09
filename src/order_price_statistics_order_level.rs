use numpy::{PyArray1, PyArray2};
use pyo3::prelude::*;

// 复用必要的结构体和函数
use crate::order_price_statistics::{
    find_trade_volume_ranges, get_price_statistics_column_names, OrderVolumeGroupV2,
};

/// 计算订单聚合后的VWAP价格统计指标（订单级别输出版本）
///
/// 该函数与 calculate_trade_price_statistics_by_volume_v2 的逻辑相同，
/// 但返回订单级别的结果，避免同一订单的多笔成交产生重复数据。
///
/// 🎯 核心特点：
/// ============
/// - 订单级别输出：每个订单返回一行统计指标
/// - 避免数据重复：同一订单的多笔成交不会产生重复结果
/// - VWAP计算：使用订单的成交量加权平均价格
/// - 基于订单类型：通过ask_order/bid_order区分买卖方向
///
/// 📊 输出结构：
/// ============
/// - means数组：num_orders行10列，每行对应一个订单的VWAP价格均值
/// - stds数组：num_orders行10列，每行对应一个订单的VWAP价格标准差
/// - column_names：20个列名（10个均值+10个标准差）
///
/// 🔄 订单聚合逻辑：
/// ==================
/// - 卖单（ask_order != 0）：基于ask_order聚合成交记录
/// - 买单（bid_order != 0）：基于bid_order聚合成交记录
/// - 每个订单计算：
///   - 总volume：累加所有成交volume
///   - VWAP价格：Σ(volume × price) / Σ(volume)
///   - 最后时间：所有成交时间的最大值
///
/// 参数：
/// =====
/// volume : NDArray[np.float64]
///     成交量数组
/// exchtime : NDArray[np.int64]
///     成交时间数组（纳秒时间戳，函数内部自动转换为秒）
/// price : NDArray[np.float64]
///     成交价格数组
/// flag : NDArray[np.int32]
///     主买卖标志数组（在订单级别版本中被忽略）
/// ask_order : NDArray[np.int64]
///     卖单订单号数组
/// bid_order : NDArray[np.int64]
///     买单订单号数组
/// min_count : int, default=10
///     计算统计指标所需的最少同类型订单数
/// use_flag : str, default="same"
///     类型筛选参数："same"=同类型，"diff"=反类型，"ignore"=忽略类型
///
/// 返回值：
/// =======
/// Tuple[NDArray[np.float64], NDArray[np.float64], List[str]]
///     - VWAP价格均值数组：num_orders行10列，每行对应一个订单的10个档位VWAP价格均值
///     - VWAP价格标准差数组：num_orders行10列，每行对应一个订单的10个档位VWAP价格标准差
///     - 列名列表：包含20个列名（10个均值+10个标准差）
///
/// 示例：
/// =====
/// >>> import rust_pyfunc as rp
/// >>> import numpy as np
/// >>>
/// >>> # 准备测试数据
/// >>> volume = np.array([100.0, 100.0, 200.0, 200.0, 100.0])
/// >>> exchtime = np.array([1609459200000000000, 1609459201000000000, 1609459202000000000,
/// ...                     1609459203000000000, 1609459204000000000])
/// >>> price = np.array([10.1, 10.2, 20.1, 20.2, 10.3])
/// >>> flag = np.array([66, 66, 83, 83, 66])  # 66=买，83=卖
/// >>> ask_order = np.array([0, 0, 1001, 1001, 0])  # 卖单订单号
/// >>> bid_order = np.array([2001, 2001, 0, 0, 2002])  # 买单订单号
/// >>>
/// >>> # 计算订单级别的VWAP价格统计指标
/// >>> means, stds, columns = rp.calculate_trade_price_statistics_by_volume_order_level(
/// ...     volume, exchtime, price, flag, ask_order, bid_order, min_count=2, use_flag="same"
/// ... )
/// >>> print(f"订单数量: {means.shape[0]}")  # 3个订单
/// >>> print(f"VWAP均值数组形状: {means.shape}")  # (3, 10)
/// >>> print(f"VWAP标准差数组形状: {stds.shape}")  # (3, 10)
#[pyfunction]
#[pyo3(signature = (volume, exchtime, price, flag, ask_order, bid_order, min_count=10, use_flag="same"))]
pub fn calculate_trade_price_statistics_by_volume_order_level(
    py: Python,
    volume: &PyArray1<f64>,
    exchtime: &PyArray1<i64>,
    price: &PyArray1<f64>,
    flag: &PyArray1<i32>, // 该参数在订单级别版本中被忽略
    ask_order: &PyArray1<i64>,
    bid_order: &PyArray1<i64>,
    min_count: usize,
    use_flag: &str,
) -> PyResult<(Py<PyArray2<f64>>, Py<PyArray2<f64>>, Vec<String>)> {
    let volume_slice = volume.readonly();
    let exchtime_slice = exchtime.readonly();
    let price_slice = price.readonly();
    let _flag_slice = flag.readonly(); // 不再使用flag参数
    let ask_order_slice = ask_order.readonly();
    let bid_order_slice = bid_order.readonly();

    let volume_data = volume_slice.as_slice()?;
    let exchtime_raw = exchtime_slice.as_slice()?;
    let price_data = price_slice.as_slice()?;
    let _flag_data = _flag_slice.as_slice()?; // 不再使用flag数据
    let ask_order_data = ask_order_slice.as_slice()?;
    let bid_order_data = bid_order_slice.as_slice()?;

    // 将纳秒时间戳转换为秒
    let exchtime_data: Vec<f64> = exchtime_raw.iter().map(|&t| t as f64 / 1e9).collect();

    let n = volume_data.len();
    if n == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "输入数组不能为空",
        ));
    }

    // 1. 基于订单类型的订单聚合（不再使用flag）
    let mut orders: Vec<(i64, bool, f64, f64, f64)> = Vec::new(); // (order_id, is_bid, volume, vwap_price, time)
    let mut order_map: std::collections::HashMap<i64, usize> = std::collections::HashMap::new();

    for i in 0..n {
        // 分别处理买单和卖单
        if ask_order_data[i] != 0 {
            // 卖单
            let order_id = ask_order_data[i];
            let is_bid = false;

            if let Some(&order_idx) = order_map.get(&order_id) {
                // 更新现有订单
                let old_vol = orders[order_idx].2;
                let old_vwap = orders[order_idx].3;
                let new_vol = old_vol + volume_data[i];
                // 计算新的VWAP: (old_vol * old_vwap + volume * price) / new_vol
                let new_vwap = (old_vol * old_vwap + volume_data[i] * price_data[i]) / new_vol;

                orders[order_idx].2 = new_vol;
                orders[order_idx].3 = new_vwap;
                orders[order_idx].4 = orders[order_idx].4.max(exchtime_data[i]);
            } else {
                // 新订单
                let order_idx = orders.len();
                orders.push((
                    order_id,
                    is_bid,
                    volume_data[i],
                    price_data[i], // 初始VWAP就是第一笔成交价格
                    exchtime_data[i],
                ));
                order_map.insert(order_id, order_idx);
            }
        }

        if bid_order_data[i] != 0 {
            // 买单
            let order_id = bid_order_data[i];
            let is_bid = true;

            if let Some(&order_idx) = order_map.get(&order_id) {
                // 更新现有订单
                let old_vol = orders[order_idx].2;
                let old_vwap = orders[order_idx].3;
                let new_vol = old_vol + volume_data[i];
                // 计算新的VWAP: (old_vol * old_vwap + volume * price) / new_vol
                let new_vwap = (old_vol * old_vwap + volume_data[i] * price_data[i]) / new_vol;

                orders[order_idx].2 = new_vol;
                orders[order_idx].3 = new_vwap;
                orders[order_idx].4 = orders[order_idx].4.max(exchtime_data[i]);
            } else {
                // 新订单
                let order_idx = orders.len();
                orders.push((
                    order_id,
                    is_bid,
                    volume_data[i],
                    price_data[i], // 初始VWAP就是第一笔成交价格
                    exchtime_data[i],
                ));
                order_map.insert(order_id, order_idx);
            }
        }
    }

    // 2. 按订单volume和时间排序
    orders.sort_unstable_by(|a, b| {
        a.2.partial_cmp(&b.2)
            .unwrap()
            .then(a.4.partial_cmp(&b.4).unwrap())
    });

    // 3. 构建订单volume组
    let order_volumes: Vec<f64> = orders.iter().map(|(_, _, vol, _, _)| *vol).collect();
    let order_ranges = find_trade_volume_ranges(&order_volumes);

    let mut order_groups: Vec<OrderVolumeGroupV2> = Vec::new();

    for (vol, start_idx, end_idx) in order_ranges.iter() {
        let mut group = OrderVolumeGroupV2::new(*vol);

        for i in *start_idx..*end_idx {
            let (_, is_bid, _, vwap_price, time) = orders[i];
            group.add_order(i, time, vwap_price, is_bid);
        }

        order_groups.push(group);
    }

    // 4. 计算订单VWAP统计指标（应用V3版本优化）
    let mut order_means = vec![vec![f64::NAN; 10]; orders.len()];
    let mut order_stds = vec![vec![f64::NAN; 10]; orders.len()];
    let percentages = [0.01, 0.02, 0.03, 0.04, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50];

    for group in order_groups.iter() {
        let group_size = group.indices.len();
        if group_size < min_count {
            continue;
        }

        // 构建时间排序的买/卖订单记录（关键优化：一次排序，多次使用）
        let mut buy_records: Vec<(f64, usize, f64)> = Vec::new(); // (time, group_idx, vwap_price)
        let mut sell_records: Vec<(f64, usize, f64)> = Vec::new();

        for i in 0..group_size {
            let time = group.times[i];
            let vwap_price = group.vwap_prices[i];
            let is_bid = group.order_types[i];

            if is_bid {
                buy_records.push((time, i, vwap_price));
            } else {
                sell_records.push((time, i, vwap_price));
            }
        }

        // 按时间排序（只排序一次！）
        buy_records.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        sell_records.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        // 预分配工作缓冲区
        let max_records = buy_records.len().max(sell_records.len());
        let mut distances_buffer: Vec<(f64, f64)> = Vec::with_capacity(max_records);

        // 批量处理每个订单
        for i in 0..group_size {
            let orig_idx = group.indices[i];
            let current_time = group.times[i];
            let current_is_bid = group.order_types[i];

            // 选择目标订单集合
            let target_records = match use_flag {
                "same" => {
                    if current_is_bid {
                        &buy_records
                    } else {
                        &sell_records
                    }
                }
                "diff" => {
                    if current_is_bid {
                        &sell_records
                    } else {
                        &buy_records
                    }
                }
                _ => continue,
            };

            if target_records.len() < min_count + 1 {
                continue;
            }

            // 清空缓冲区并计算时间距离（遍历已排序的记录）
            distances_buffer.clear();

            for &(time, group_idx, vwap_price) in target_records.iter() {
                if use_flag == "same" && group_idx == i {
                    continue; // 跳过自己
                }
                let time_diff = (current_time - time).abs();
                distances_buffer.push((time_diff, vwap_price));
            }

            let available = distances_buffer.len();
            if available < min_count {
                continue;
            }

            // 部分排序优化：只排序需要的部分（最大到50%档位）
            let max_needed = ((available as f64 * 0.50).ceil() as usize).min(available);

            if max_needed < available {
                distances_buffer
                    .select_nth_unstable_by(max_needed, |a, b| a.0.partial_cmp(&b.0).unwrap());
                distances_buffer[..=max_needed]
                    .sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            } else {
                distances_buffer.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            }

            // 批量计算所有百分比档位（增量算法，避免重复计算）
            let mut sum = 0.0;
            let mut sum_sq = 0.0;
            let mut count = 0;

            for (pct_idx, &pct) in percentages.iter().enumerate() {
                let target_count = ((available as f64 * pct).ceil() as usize)
                    .max(1)
                    .min(available);

                if target_count < min_count {
                    continue;
                }

                // 增量添加元素
                while count < target_count {
                    let price = distances_buffer[count].1;
                    sum += price;
                    sum_sq += price * price;
                    count += 1;
                }

                // 计算统计量
                let mean = sum / count as f64;
                let variance = (sum_sq / count as f64) - (mean * mean);
                let std = variance.max(0.0).sqrt();

                order_means[orig_idx][pct_idx] = mean;
                order_stds[orig_idx][pct_idx] = std;
            }
        }
    }

    // 5. 构建订单级别的输出
    let num_orders = orders.len();
    let mut means = vec![vec![f64::NAN; 10]; num_orders];
    let mut stds = vec![vec![f64::NAN; 10]; num_orders];

    // 填充订单级别的统计指标
    for i in 0..num_orders {
        // 复制统计指标
        for j in 0..10 {
            means[i][j] = order_means[i][j];
            stds[i][j] = order_stds[i][j];
        }
    }

    let means_array = PyArray2::from_vec2(py, &means)?;
    let stds_array = PyArray2::from_vec2(py, &stds)?;
    let column_names = get_price_statistics_column_names();

    // 返回订单级别的结果
    Ok((means_array.to_owned(), stds_array.to_owned(), column_names))
}
