/// 订单记录分析的超高性能版本V3
/// 针对13万+大数据量的专门优化，避免重复排序和不必要的数据转换
use numpy::{PyArray1, PyArray2};
use pyo3::prelude::*;
use std::collections::HashMap;

/// 优化的订单volume组结构V3 - 专注于内存局部性和减少分配
#[derive(Debug)]
struct OrderVolumeGroupV3 {
    _volume: f64,
    indices: Vec<usize>,
    times_nanoseconds: Vec<i64>, // 直接存储纳秒时间戳，避免转换
    prices: Vec<f64>,
    order_types: Vec<bool>,
    ask_indices: Vec<usize>,
    bid_indices: Vec<usize>,

    // V3优化：预计算的时间排序索引，避免重复排序
    time_sorted_indices: Vec<usize>,
}

impl OrderVolumeGroupV3 {
    fn new(volume: f64, capacity: usize) -> Self {
        let mut group = Self {
            _volume: volume,
            indices: Vec::with_capacity(capacity),
            times_nanoseconds: Vec::with_capacity(capacity),
            prices: Vec::with_capacity(capacity),
            order_types: Vec::with_capacity(capacity),
            ask_indices: Vec::new(),
            bid_indices: Vec::new(),
            time_sorted_indices: Vec::with_capacity(capacity),
        };

        // 预分配索引数组
        group.time_sorted_indices.reserve(capacity);
        group
    }

    fn add_record(&mut self, orig_idx: usize, time_ns: i64, price: f64, is_bid: bool) {
        let group_idx = self.indices.len();

        self.indices.push(orig_idx);
        self.times_nanoseconds.push(time_ns);
        self.prices.push(price);
        self.order_types.push(is_bid);
        self.time_sorted_indices.push(group_idx);

        // 根据订单类型分类存储位置
        if is_bid {
            self.bid_indices.push(group_idx);
        } else {
            self.ask_indices.push(group_idx);
        }
    }

    /// V3优化：一次性时间排序，避免重复排序
    fn finalize(&mut self) {
        if self.indices.len() > 1 {
            // 根据时间进行排序
            self.time_sorted_indices.sort_unstable_by(|&a, &b| {
                self.times_nanoseconds[a].cmp(&self.times_nanoseconds[b])
            });
        }
    }

    /// V3优化的最近记录查找 - 利用预排序的时间索引
    fn find_nearest_records_optimized(
        &self,
        current_group_idx: usize,
        target_indices: &[usize],
        max_records: usize,
    ) -> Vec<(i64, f64)> {
        if target_indices.is_empty() {
            return Vec::new();
        }

        let current_time_ns = self.times_nanoseconds[current_group_idx];
        let mut result = Vec::with_capacity(target_indices.len().min(max_records));

        // V3优化：利用时间已排序的特性，用二分查找快速定位时间位置
        let current_sorted_pos = self
            .time_sorted_indices
            .binary_search_by(|&idx| self.times_nanoseconds[idx].cmp(&current_time_ns))
            .unwrap_or_else(|pos| pos);

        // 从当前时间位置向两边搜索最近的记录
        let mut left_pos = current_sorted_pos.saturating_sub(1);
        let mut right_pos = current_sorted_pos;

        while result.len() < max_records
            && (left_pos > 0 || right_pos < self.time_sorted_indices.len())
        {
            // 检查左边
            if left_pos > 0 {
                let left_idx = self.time_sorted_indices[left_pos];
                if target_indices.contains(&left_idx) && left_idx != current_group_idx {
                    let time_diff = (current_time_ns - self.times_nanoseconds[left_idx]).abs();
                    result.push((time_diff, self.prices[left_idx]));
                }
                left_pos -= 1;
            }

            // 检查右边
            if right_pos < self.time_sorted_indices.len() && result.len() < max_records {
                let right_idx = self.time_sorted_indices[right_pos];
                if target_indices.contains(&right_idx) && right_idx != current_group_idx {
                    let time_diff = (self.times_nanoseconds[right_idx] - current_time_ns).abs();
                    result.push((time_diff, self.prices[right_idx]));
                }
                right_pos += 1;
            }
        }

        // 如果还不够，继续搜索更远的记录
        if result.len() < max_records {
            for &target_idx in target_indices.iter() {
                if target_idx != current_group_idx
                    && !result.iter().any(|(_, _p)| {
                        // 检查是否已经在结果中（通过价格判断，因为时间差可能重复）
                        let target_price = self.prices[target_idx];
                        result
                            .iter()
                            .any(|(_, price)| (price - target_price).abs() < f64::EPSILON)
                    })
                {
                    let time_diff = (current_time_ns - self.times_nanoseconds[target_idx]).abs();
                    result.push((time_diff, self.prices[target_idx]));
                    if result.len() >= max_records {
                        break;
                    }
                }
            }
        }

        // 最后按时间差排序（这个排序的向量会更小）
        result.sort_unstable_by_key(|(time_diff, _)| *time_diff);
        result.truncate(max_records);

        result
    }

    /// V3优化的批量指标计算
    fn compute_all_indicators_optimized(
        &self,
        results: &mut [Vec<f64>],
        min_count: usize,
        use_flag: &str,
    ) {
        let group_size = self.indices.len();
        if group_size < min_count {
            return;
        }

        // 根据use_flag确定目标索引集合（基于订单类型）
        let get_target_indices = |current_group_idx: usize| -> &[usize] {
            let current_is_bid = self.order_types[current_group_idx];

            match use_flag {
                "same" => {
                    if current_is_bid {
                        &self.bid_indices
                    } else {
                        &self.ask_indices
                    }
                }
                "diff" => {
                    if current_is_bid {
                        &self.ask_indices
                    } else {
                        &self.bid_indices
                    }
                }
                "ignore" => &self.time_sorted_indices, // 使用预排序的索引
                _ => &self.time_sorted_indices,
            }
        };

        // 批量计算所有订单的指标
        for (group_idx, &orig_idx) in self.indices.iter().enumerate() {
            let target_indices = get_target_indices(group_idx);

            if target_indices.len() < 2 {
                continue;
            }

            // V3优化：使用优化的查找算法
            let time_distances = self.find_nearest_records_optimized(
                group_idx,
                target_indices,
                50, // 限制记录数量以提高性能
            );

            if time_distances.is_empty() {
                continue;
            }

            let current_price = self.prices[group_idx];
            let _current_time_ns = self.times_nanoseconds[group_idx];

            // V3优化：重用结果数组，避免重复分配
            let result_row = &mut results[orig_idx];
            if result_row.len() < 22 {
                result_row.resize(22, f64::NAN);
            }

            // 1. 最近时间间隔
            if let Some((nearest_time, _)) = time_distances.first() {
                result_row[0] = *nearest_time as f64 / 1e9; // 转换为秒
            }

            let total_count = time_distances.len();

            // 2-11. 分档计算平均时间间隔（1%到50%）
            let thresholds = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50];
            for (i, &threshold) in thresholds.iter().enumerate() {
                let count = ((total_count as f64 * threshold).ceil() as usize)
                    .max(1)
                    .min(total_count);

                let sum: f64 = time_distances
                    .iter()
                    .take(count)
                    .map(|(time_diff, _)| *time_diff as f64 / 1e9) // 转换为秒
                    .sum();

                result_row[i + 1] = sum / count as f64;
            }

            // 12. 全部平均时间间隔
            let total_sum: f64 = time_distances
                .iter()
                .map(|(time_diff, _)| *time_diff as f64 / 1e9)
                .sum();
            result_row[11] = total_sum / total_count as f64;

            // V3优化：更高效的价格分位数计算
            let price_percentages = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00];

            // 提取价格并排序（只对需要的记录）
            let mut prices: Vec<f64> = time_distances.iter().map(|(_, price)| *price).collect();
            prices.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

            // 批量计算所有分位数
            for (idx, &pct) in price_percentages.iter().enumerate() {
                let count = if pct == 1.0 {
                    total_count
                } else {
                    ((total_count as f64 * pct).ceil() as usize)
                        .max(1)
                        .min(total_count)
                };

                let prices_slice = &prices[..count];

                // 优化的二分查找
                let rank = match prices_slice
                    .binary_search_by(|&p| p.partial_cmp(&current_price).unwrap())
                {
                    Ok(pos) => pos,
                    Err(pos) => pos,
                };

                result_row[idx + 12] = rank as f64 / prices_slice.len() as f64;
            }
        }
    }
}

/// 计算订单时间间隔和价格分位数的超高性能版本V3
///
/// # 参数
/// - volume: 交易volume数组
/// - exchtime: 交易所时间戳数组（纳秒）
/// - price: 交易价格数组
/// - flag: 交易标志数组（V3版本中忽略）
/// - ask_order: 卖单订单号数组
/// - bid_order: 买单订单号数组
/// - min_count: 最小计算记录数
/// - use_flag: 计算策略 ("same"/"diff"/"ignore")
///
/// # 返回值
/// - (结果数组, 列名列表)：结果数组形状为(n, 27)
#[pyfunction]
#[pyo3(signature = (volume, exchtime, price, flag, ask_order, bid_order, min_count=100, use_flag="ignore"))]
pub fn calculate_order_time_gap_and_price_percentile_ultra_sorted_v3(
    py: Python,
    volume: &PyArray1<f64>,
    exchtime: &PyArray1<f64>,
    price: &PyArray1<f64>,
    flag: &PyArray1<i32>, // 该参数在V3版本中被忽略
    ask_order: &PyArray1<i64>,
    bid_order: &PyArray1<i64>,
    min_count: usize,
    use_flag: &str,
) -> PyResult<(Py<PyArray2<f64>>, Vec<String>)> {
    let volume_slice = volume.readonly();
    let exchtime_slice = exchtime.readonly();
    let price_slice = price.readonly();
    let _flag_slice = flag.readonly(); // 不再使用flag参数
    let ask_order_slice = ask_order.readonly();
    let bid_order_slice = bid_order.readonly();

    let volume_data = volume_slice.as_slice()?;
    let exchtime_data = exchtime_slice.as_slice()?; // 保持为f64纳秒
    let price_data = price_slice.as_slice()?;
    let _flag_data = _flag_slice.as_slice()?; // 不再使用flag数据
    let ask_order_data = ask_order_slice.as_slice()?;
    let bid_order_data = bid_order_slice.as_slice()?;

    let n = volume_data.len();
    if n == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "输入数组不能为空",
        ));
    }

    // V3优化：预估订单数量，减少HashMap扩容
    let estimated_orders = n / 10; // 假设平均每个订单有10条记录

    // 1. 基于订单类型的订单聚合（V3优化：使用纳秒时间戳）
    let mut orders: Vec<(i64, bool, f64, f64, i64)> = Vec::with_capacity(estimated_orders); // (order_id, is_bid, volume, price, time_ns)
    let mut order_map: HashMap<i64, usize> = HashMap::with_capacity(estimated_orders);

    // V3优化：批量处理订单数据，减少HashMap操作
    for i in 0..n {
        let current_time_ns = exchtime_data[i] as i64;
        let current_volume = volume_data[i];
        let current_price = price_data[i];

        // 处理卖单
        if ask_order_data[i] != 0 {
            let order_id = ask_order_data[i];

            if let Some(&order_idx) = order_map.get(&order_id) {
                // 更新现有订单（V3优化：直接操作，避免重复查找）
                let (existing_order_id, is_bid, old_vol, old_price, old_time_ns) =
                    orders[order_idx];
                let new_vol = old_vol + current_volume;
                let new_price = (old_vol * old_price + current_volume * current_price) / new_vol;
                let new_time_ns = old_time_ns.max(current_time_ns);

                orders[order_idx] = (existing_order_id, is_bid, new_vol, new_price, new_time_ns);
            } else {
                // 新订单
                let order_idx = orders.len();
                orders.push((
                    order_id,
                    false,
                    current_volume,
                    current_price,
                    current_time_ns,
                ));
                order_map.insert(order_id, order_idx);
            }
        }

        // 处理买单
        if bid_order_data[i] != 0 {
            let order_id = bid_order_data[i];

            if let Some(&order_idx) = order_map.get(&order_id) {
                // 更新现有订单
                let (existing_order_id, is_bid, old_vol, old_price, old_time_ns) =
                    orders[order_idx];
                let new_vol = old_vol + current_volume;
                let new_price = (old_vol * old_price + current_volume * current_price) / new_vol;
                let new_time_ns = old_time_ns.max(current_time_ns);

                orders[order_idx] = (existing_order_id, is_bid, new_vol, new_price, new_time_ns);
            } else {
                // 新订单
                let order_idx = orders.len();
                orders.push((
                    order_id,
                    true,
                    current_volume,
                    current_price,
                    current_time_ns,
                ));
                order_map.insert(order_id, order_idx);
            }
        }
    }

    // 2. 按订单volume和时间排序（保持纳秒精度）
    orders.sort_unstable_by(|a, b| {
        a.2.partial_cmp(&b.2).unwrap().then(a.4.cmp(&b.4)) // 直接比较纳秒时间戳
    });

    // 3. 构建订单volume组（V3优化：预分组）
    let order_volumes: Vec<f64> = orders.iter().map(|(_, _, vol, _, _)| *vol).collect();
    let order_ranges = find_order_volume_ranges_optimized(&order_volumes);

    // V3优化：预分配结果数组
    let mut order_results: Vec<Vec<f64>> = vec![vec![f64::NAN; 22]; orders.len()];

    // 4. 批量计算订单指标
    for (vol, start_idx, end_idx) in order_ranges.iter() {
        let group_size = *end_idx - *start_idx;
        let mut group = OrderVolumeGroupV3::new(*vol, group_size);

        // 批量添加记录到组
        for i in *start_idx..*end_idx {
            let (_order_id, is_bid, _, price, time_ns) = orders[i];
            group.add_record(i, time_ns, price, is_bid);
        }

        // V3优化：一次性时间排序
        group.finalize();

        // 批量计算指标
        group.compute_all_indicators_optimized(&mut order_results, min_count, use_flag);
    }

    // 5. 映射回交易记录，包含订单信息
    let mut results = vec![vec![f64::NAN; 27]; n];

    for i in 0..n {
        // 分别处理买单和卖单
        if ask_order_data[i] != 0 {
            let order_id = ask_order_data[i];
            if let Some(&order_idx) = order_map.get(&order_id) {
                // 复制22个计算指标
                results[i][..22].copy_from_slice(&order_results[order_idx]);

                // 添加订单信息
                results[i][22] = order_id as f64;
                results[i][23] = 0.0; // 卖单

                let (_, _, order_volume, order_price, order_time_ns) = orders[order_idx];
                results[i][24] = order_volume;
                results[i][25] = order_time_ns as f64 / 1e9; // 转换为秒
                results[i][26] = order_price;
            }
        }

        if bid_order_data[i] != 0 {
            let order_id = bid_order_data[i];
            if let Some(&order_idx) = order_map.get(&order_id) {
                // 复制22个计算指标
                results[i][..22].copy_from_slice(&order_results[order_idx]);

                // 添加订单信息
                results[i][22] = order_id as f64;
                results[i][23] = 1.0; // 买单

                let (_, _, order_volume, order_price, order_time_ns) = orders[order_idx];
                results[i][24] = order_volume;
                results[i][25] = order_time_ns as f64 / 1e9; // 转换为秒
                results[i][26] = order_price;
            }
        }
    }

    let result_array = PyArray2::from_vec2(py, &results)?;
    let column_names = get_order_column_names_v3();

    Ok((result_array.to_owned(), column_names))
}

/// V3优化的volume组范围查找
fn find_order_volume_ranges_optimized(volumes: &[f64]) -> Vec<(f64, usize, usize)> {
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

/// V3版本的列名定义
fn get_order_column_names_v3() -> Vec<String> {
    vec![
        // 时间间隔指标 (1-12)
        "最近时间间隔".to_string(),
        "平均时间间隔_1%".to_string(),
        "平均时间间隔_2%".to_string(),
        "平均时间间隔_5%".to_string(),
        "平均时间间隔_10%".to_string(),
        "平均时间间隔_15%".to_string(),
        "平均时间间隔_20%".to_string(),
        "平均时间间隔_25%".to_string(),
        "平均时间间隔_30%".to_string(),
        "平均时间间隔_40%".to_string(),
        "平均时间间隔_50%".to_string(),
        "平均时间间隔_全部".to_string(),
        // 价格分位数指标 (13-22)
        "价格分位数_10%".to_string(),
        "价格分位数_20%".to_string(),
        "价格分位数_30%".to_string(),
        "价格分位数_40%".to_string(),
        "价格分位数_50%".to_string(),
        "价格分位数_60%".to_string(),
        "价格分位数_70%".to_string(),
        "价格分位数_80%".to_string(),
        "价格分位数_90%".to_string(),
        "价格分位数_100%".to_string(),
        // 订单信息 (23-27)
        "订单编号".to_string(),
        "买单标识".to_string(),
        "订单总量".to_string(),
        "订单时间".to_string(),
        "订单价格".to_string(),
    ]
}
