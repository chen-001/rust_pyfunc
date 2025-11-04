use numpy::{PyArray1, PyArray2};
use pyo3::prelude::*;
use std::f64;

/// 体量分桶算法：将体量分成20个区间或者保持原始体量（如果种类≤20）
fn create_volume_buckets(volumes: &[f64], num_buckets: usize) -> Vec<f64> {
    if volumes.is_empty() {
        return Vec::new();
    }

    // 收集所有唯一的体量值
    let mut unique_volumes: Vec<f64> = volumes.iter().cloned().collect();
    unique_volumes.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    unique_volumes.dedup();

    // 如果唯一体量种类不超过num_buckets，使用原始体量
    if unique_volumes.len() <= num_buckets {
        return volumes.to_vec();
    }

    // 否则创建分桶
    let min_vol = unique_volumes[0];
    let max_vol = unique_volumes[unique_volumes.len() - 1];

    // 避免除零错误
    if max_vol == min_vol {
        return volumes.to_vec();
    }

    let bucket_size = (max_vol - min_vol) / num_buckets as f64;
    let mut bucketed_volumes = Vec::with_capacity(volumes.len());

    for &volume in volumes.iter() {
        let bucket_idx = ((volume - min_vol) / bucket_size).floor() as usize;
        let bucket_idx = bucket_idx.min(num_buckets - 1); // 确保不超出范围
        let bucket_center = min_vol + (bucket_idx as f64 + 0.5) * bucket_size;
        bucketed_volumes.push(bucket_center);
    }

    bucketed_volumes
}

/// 专门用于订单记录分析的Ultra Sorted算法（分桶版本）
///
/// 核心设计思想：
/// 1. 彻底消除O(n²)复杂度
/// 2. 利用时间排序避免排序开销
/// 3. 批量预计算相同volume分桶组的共享数据
/// 4. 使用二分查找加速时间定位

#[derive(Debug)]
struct BucketedOrderVolumeGroup {
    #[allow(dead_code)]
    volume: f64,
    indices: Vec<usize>,      // 原始数据索引
    times: Vec<f64>,          // 时间数组（已排序）
    prices: Vec<f64>,         // 对应的价格
    flags: Vec<i32>,          // 对应的flag
    buy_indices: Vec<usize>,  // 买单在组内的位置
    sell_indices: Vec<usize>, // 卖单在组内的位置
}

impl BucketedOrderVolumeGroup {
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

    /// 修复的时间距离计算：使用二分查找定位最近记录（优化版本：预分配内存）
    fn find_nearest_records_ultra_fast(
        &self,
        current_group_idx: usize,
        target_indices: &[usize],
        max_records: usize,
    ) -> Vec<(f64, f64)> {
        if target_indices.is_empty() {
            return Vec::new();
        }

        let current_time = self.times[current_group_idx];
        let mut time_distances: Vec<(f64, f64)> = Vec::with_capacity(target_indices.len());

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

    /// 批量计算该volume组所有记录的指标（优化版本：简化为单次排序）
    fn compute_all_indicators_ultra_fast(
        &self,
        results: &mut [Vec<f64>],
        min_count: usize,
        use_flag: &str,
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
                        self.buy_indices
                            .iter()
                            .filter(|&&idx| idx != current_group_idx)
                            .cloned()
                            .collect()
                    } else {
                        self.sell_indices
                            .iter()
                            .filter(|&&idx| idx != current_group_idx)
                            .cloned()
                            .collect()
                    }
                }
                "diff" => {
                    if current_flag == 66 {
                        self.sell_indices.clone()
                    } else {
                        self.buy_indices.clone()
                    }
                }
                _ => (0..group_size)
                    .filter(|&idx| idx != current_group_idx)
                    .collect(),
            }
        };

        // 为每个记录计算指标
        for current_group_idx in 0..group_size {
            let target_indices = get_target_indices(current_group_idx);

            if target_indices.len() < min_count {
                continue;
            }

            // 直接使用超快速算法收集所有最近记录（一次性排序）
            let time_distances = self.find_nearest_records_ultra_fast(
                current_group_idx,
                &target_indices,
                target_indices.len(),
            );

            if time_distances.len() >= min_count {
                let orig_idx = self.indices[current_group_idx];
                let current_price = self.prices[current_group_idx];

                calculate_order_indicators_ultra_fast(
                    &mut results[orig_idx],
                    &time_distances,
                    current_price,
                );
            }
        }
    }
}

/// 超高速指标计算，最大化利用已排序特性
fn calculate_order_indicators_ultra_fast(
    result_row: &mut [f64],
    time_distances: &[(f64, f64)],
    current_price: f64,
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
        let count = ((total_count as f64 * pct).ceil() as usize)
            .max(1)
            .min(total_count);

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
    let price_percentiles = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00];

    // 预分配和重用价格数组
    let max_price_count = total_count;
    let mut all_prices: Vec<f64> = Vec::with_capacity(max_price_count);
    for &(_, price) in time_distances.iter() {
        all_prices.push(price);
    }
    all_prices.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

    // 批量计算所有分位数
    for (idx, &pct) in price_percentiles.iter().enumerate() {
        let count = if pct == 1.0 {
            total_count
        } else {
            ((total_count as f64 * pct).ceil() as usize)
                .max(1)
                .min(total_count)
        };

        let prices_slice = &all_prices[..count];

        // 优化的二分查找
        let rank = match prices_slice.binary_search_by(|&p| p.partial_cmp(&current_price).unwrap())
        {
            Ok(pos) => pos,
            Err(pos) => pos,
        };

        result_row[idx + 12] = rank as f64 / prices_slice.len() as f64;
    }
}

/// 快速定位volume组范围的优化版本（分桶）
fn find_bucketed_order_volume_ranges(volumes: &[f64]) -> Vec<(f64, usize, usize)> {
    if volumes.is_empty() {
        return Vec::new();
    }

    let mut ranges = Vec::new();
    let mut current_volume = volumes[0];
    let mut start_idx = 0;

    for i in 1..volumes.len() {
        if (volumes[i] - current_volume).abs() > f64::EPSILON {
            ranges.push((current_volume, start_idx, i));
            current_volume = volumes[i];
            start_idx = i;
        }
    }

    ranges.push((current_volume, start_idx, volumes.len()));
    ranges
}

#[pyfunction]
#[pyo3(signature = (volume, exchtime, price, flag, ask_order, bid_order, min_count=100, use_flag="ignore", num_buckets=20))]
pub fn calculate_order_time_gap_and_price_percentile_ultra_sorted_bucketed(
    py: Python,
    volume: &PyArray1<f64>,
    exchtime: &PyArray1<f64>,
    price: &PyArray1<f64>,
    flag: &PyArray1<i32>,
    ask_order: &PyArray1<i64>,
    bid_order: &PyArray1<i64>,
    min_count: usize,
    use_flag: &str,
    num_buckets: usize,
) -> PyResult<(Py<PyArray2<f64>>, Vec<String>)> {
    let volume_slice = volume.readonly();
    let exchtime_slice = exchtime.readonly();
    let price_slice = price.readonly();
    let flag_slice = flag.readonly();
    let ask_order_slice = ask_order.readonly();
    let bid_order_slice = bid_order.readonly();

    let volume_data = volume_slice.as_slice()?;
    let exchtime_raw = exchtime_slice.as_slice()?;
    let price_data = price_slice.as_slice()?;
    let flag_data = flag_slice.as_slice()?;
    let ask_order_data = ask_order_slice.as_slice()?;
    let bid_order_data = bid_order_slice.as_slice()?;

    // 将纳秒时间戳转换为秒
    let exchtime_data: Vec<f64> = exchtime_raw.iter().map(|&t| t / 1e9).collect();

    let n = volume_data.len();
    if n == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "输入数组不能为空",
        ));
    }

    // 1. 超快速订单聚合
    let mut orders: Vec<(i64, i32, f64, f64, f64)> = Vec::new(); // (order_id, flag, volume, price, time)
    let mut order_map: std::collections::HashMap<(i64, i32), usize> =
        std::collections::HashMap::new();

    for i in 0..n {
        let current_flag = flag_data[i];
        let order_id = if current_flag == 66 {
            bid_order_data[i]
        } else {
            ask_order_data[i]
        };

        if order_id == 0 {
            continue;
        }

        let key = (order_id, current_flag);

        if let Some(&order_idx) = order_map.get(&key) {
            // 更新现有订单
            let old_vol = orders[order_idx].2;
            let old_price = orders[order_idx].3;
            let new_vol = old_vol + volume_data[i];
            let new_price = (old_vol * old_price + volume_data[i] * price_data[i]) / new_vol;

            orders[order_idx].2 = new_vol;
            orders[order_idx].3 = new_price;
            orders[order_idx].4 = orders[order_idx].4.max(exchtime_data[i]);
        } else {
            // 新订单
            let order_idx = orders.len();
            orders.push((
                order_id,
                current_flag,
                volume_data[i],
                price_data[i],
                exchtime_data[i],
            ));
            order_map.insert(key, order_idx);
        }
    }

    // 2. 对订单volume进行分桶
    let order_volumes: Vec<f64> = orders.iter().map(|(_, _, vol, _, _)| *vol).collect();
    let bucketed_order_volumes = create_volume_buckets(&order_volumes, num_buckets);

    // 3. 直接按分桶后的volume和时间排序orders（避免额外的索引数组）
    let mut sorted_orders_with_bucket: Vec<(f64, (i64, i32, f64, f64, f64))> = orders
        .into_iter()
        .enumerate()
        .map(|(idx, order)| (bucketed_order_volumes[idx], order))
        .collect();
    
    sorted_orders_with_bucket.sort_unstable_by(|a, b| {
        a.0.partial_cmp(&b.0)
            .unwrap()
            .then(a.1.4.partial_cmp(&b.1.4).unwrap())
    });

    // 4. 构建分桶后的订单volume组（直接使用排序后的数据）
    let mut order_groups: Vec<BucketedOrderVolumeGroup> = Vec::new();
    let mut order_results = vec![vec![f64::NAN; 22]; sorted_orders_with_bucket.len()];
    
    if !sorted_orders_with_bucket.is_empty() {
        let mut current_volume = sorted_orders_with_bucket[0].0;
        let mut group = BucketedOrderVolumeGroup::new(current_volume);
        
        for (idx, (bucket_vol, order)) in sorted_orders_with_bucket.iter().enumerate() {
            if (*bucket_vol - current_volume).abs() > f64::EPSILON {
                // 完成当前组
                if !group.indices.is_empty() {
                    order_groups.push(group);
                }
                // 开始新组
                current_volume = *bucket_vol;
                group = BucketedOrderVolumeGroup::new(current_volume);
            }
            
            group.add_record(idx, order.4, order.3, order.1);
        }
        
        // 添加最后一组
        if !group.indices.is_empty() {
            order_groups.push(group);
        }
    }

    // 5. 计算订单指标
    for group in order_groups.iter() {
        group.compute_all_indicators_ultra_fast(&mut order_results, min_count, use_flag);
    }

    // 6. 映射回交易记录，包含订单信息
    let mut results = vec![vec![f64::NAN; 27]; n]; // 增加到27列

    for i in 0..n {
        let current_flag = flag_data[i];
        let order_id = if current_flag == 66 {
            bid_order_data[i]
        } else {
            ask_order_data[i]
        };

        if order_id != 0 {
            let key = (order_id, current_flag);
            if let Some(&order_idx) = order_map.get(&key) {
                // 复制22个计算指标
                for j in 0..22 {
                    results[i][j] = order_results[order_idx][j];
                }
                // 添加订单编号（第23列）
                results[i][22] = order_id as f64;
                // 添加买卖标识（第24列）：66=买单(1.0)，83=卖单(0.0)
                results[i][23] = if current_flag == 66 { 1.0 } else { 0.0 };

                // 添加订单聚合信息
                let (_, _, order_volume, order_price, order_time) = &sorted_orders_with_bucket[order_idx].1;
                // 添加订单volume总量（第25列）
                results[i][24] = *order_volume;
                // 添加订单最后时间（第26列）
                results[i][25] = *order_time;
                // 添加订单加权平均价格（第27列）
                results[i][26] = *order_price;
            }
        }
    }

    let result_array = PyArray2::from_vec2(py, &results)?;
    let column_names = get_order_column_names();

    Ok((result_array.to_owned(), column_names))
}

fn get_order_column_names() -> Vec<String> {
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
        "订单编号".to_string(), // 第23列：订单编号
        "买单标识".to_string(), // 第24列：买单标识(1.0=买单，0.0=卖单)
        "订单总量".to_string(), // 第25列：订单volume总量
        "订单时间".to_string(), // 第26列：订单最后成交时间
        "订单价格".to_string(), // 第27列：订单加权平均价格
    ]
}

// V2版本的订单volume组（分桶版本），基于订单类型（ask/bid）而非交易标志进行分类
#[derive(Debug)]
struct BucketedOrderVolumeGroupV2 {
    #[allow(dead_code)]
    volume: f64,
    indices: Vec<usize>,     // 原始数据索引
    times: Vec<f64>,         // 时间数组（已排序）
    prices: Vec<f64>,        // 对应的价格
    order_types: Vec<bool>,  // 对应的订单类型：true=买单，false=卖单
    ask_indices: Vec<usize>, // 卖单在组内的位置
    bid_indices: Vec<usize>, // 买单在组内的位置
}

impl BucketedOrderVolumeGroupV2 {
    fn new(volume: f64) -> Self {
        Self {
            volume,
            indices: Vec::new(),
            times: Vec::new(),
            prices: Vec::new(),
            order_types: Vec::new(),
            ask_indices: Vec::new(),
            bid_indices: Vec::new(),
        }
    }

    fn add_record(&mut self, orig_idx: usize, time: f64, price: f64, is_bid: bool) {
        let group_idx = self.indices.len();

        self.indices.push(orig_idx);
        self.times.push(time);
        self.prices.push(price);
        self.order_types.push(is_bid);

        // 根据订单类型分类存储位置
        if is_bid {
            self.bid_indices.push(group_idx);
        } else {
            self.ask_indices.push(group_idx);
        }
    }

    /// 基于订单类型的时间距离计算（优化版本：预分配内存）
    fn find_nearest_records_ultra_fast_v2(
        &self,
        current_group_idx: usize,
        target_indices: &[usize],
        max_records: usize,
    ) -> Vec<(f64, f64)> {
        if target_indices.is_empty() {
            return Vec::new();
        }

        let current_time = self.times[current_group_idx];
        let mut time_distances: Vec<(f64, f64)> = Vec::with_capacity(target_indices.len());

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

    /// 基于订单类型的批量计算指标（优化版本：简化为单次排序）
    fn compute_all_indicators_ultra_fast_v2(
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
        let get_target_indices = |current_group_idx: usize| -> Vec<usize> {
            let current_is_bid = self.order_types[current_group_idx];

            match use_flag {
                "same" => {
                    if current_is_bid {
                        self.bid_indices
                            .iter()
                            .filter(|&&idx| idx != current_group_idx)
                            .cloned()
                            .collect()
                    } else {
                        self.ask_indices
                            .iter()
                            .filter(|&&idx| idx != current_group_idx)
                            .cloned()
                            .collect()
                    }
                }
                "diff" => {
                    if current_is_bid {
                        self.ask_indices.clone()
                    } else {
                        self.bid_indices.clone()
                    }
                }
                _ => (0..group_size)
                    .filter(|&idx| idx != current_group_idx)
                    .collect(),
            }
        };

        // 为每个记录计算指标
        for current_group_idx in 0..group_size {
            let target_indices = get_target_indices(current_group_idx);

            if target_indices.len() < min_count {
                continue;
            }

            // 直接使用超快速算法收集所有最近记录（一次性排序）
            let time_distances = self.find_nearest_records_ultra_fast_v2(
                current_group_idx,
                &target_indices,
                target_indices.len(),
            );

            if time_distances.len() >= min_count {
                let orig_idx = self.indices[current_group_idx];
                let current_price = self.prices[current_group_idx];

                calculate_order_indicators_ultra_fast(
                    &mut results[orig_idx],
                    &time_distances,
                    current_price,
                );
            }
        }
    }
}

#[pyfunction]
#[pyo3(signature = (volume, exchtime, price, flag, ask_order, bid_order, min_count=100, use_flag="ignore", num_buckets=20))]
pub fn calculate_order_time_gap_and_price_percentile_ultra_sorted_v2_bucketed(
    py: Python,
    volume: &PyArray1<f64>,
    exchtime: &PyArray1<f64>,
    price: &PyArray1<f64>,
    flag: &PyArray1<i32>, // 该参数在V2版本中被忽略
    ask_order: &PyArray1<i64>,
    bid_order: &PyArray1<i64>,
    min_count: usize,
    use_flag: &str,
    num_buckets: usize,
) -> PyResult<(Py<PyArray2<f64>>, Vec<String>)> {
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
    let exchtime_data: Vec<f64> = exchtime_raw.iter().map(|&t| t / 1e9).collect();

    let n = volume_data.len();
    if n == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "输入数组不能为空",
        ));
    }

    // 1. 超快速订单聚合（V2版本：基于订单类型而非flag）
    let mut orders: Vec<(i64, bool, f64, f64, f64)> = Vec::new(); // (order_id, is_bid, volume, price, time)
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
                let old_price = orders[order_idx].3;
                let new_vol = old_vol + volume_data[i];
                let new_price = (old_vol * old_price + volume_data[i] * price_data[i]) / new_vol;

                orders[order_idx].2 = new_vol;
                orders[order_idx].3 = new_price;
                orders[order_idx].4 = orders[order_idx].4.max(exchtime_data[i]);
            } else {
                // 新订单
                let order_idx = orders.len();
                orders.push((
                    order_id,
                    is_bid,
                    volume_data[i],
                    price_data[i],
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
                let old_price = orders[order_idx].3;
                let new_vol = old_vol + volume_data[i];
                let new_price = (old_vol * old_price + volume_data[i] * price_data[i]) / new_vol;

                orders[order_idx].2 = new_vol;
                orders[order_idx].3 = new_price;
                orders[order_idx].4 = orders[order_idx].4.max(exchtime_data[i]);
            } else {
                // 新订单
                let order_idx = orders.len();
                orders.push((
                    order_id,
                    is_bid,
                    volume_data[i],
                    price_data[i],
                    exchtime_data[i],
                ));
                order_map.insert(order_id, order_idx);
            }
        }
    }

    // 2. 对订单volume进行分桶
    let order_volumes: Vec<f64> = orders.iter().map(|(_, _, vol, _, _)| *vol).collect();
    let bucketed_order_volumes = create_volume_buckets(&order_volumes, num_buckets);

    // 3. 直接按分桶后的volume和时间排序orders（避免额外的索引数组）
    let mut sorted_orders_with_bucket: Vec<(f64, (i64, bool, f64, f64, f64))> = orders
        .into_iter()
        .enumerate()
        .map(|(idx, order)| (bucketed_order_volumes[idx], order))
        .collect();
    
    sorted_orders_with_bucket.sort_unstable_by(|a, b| {
        a.0.partial_cmp(&b.0)
            .unwrap()
            .then(a.1.4.partial_cmp(&b.1.4).unwrap())
    });

    // 4. 构建分桶后的订单volume组（直接使用排序后的数据）
    let mut order_groups: Vec<BucketedOrderVolumeGroupV2> = Vec::new();
    let mut order_results = vec![vec![f64::NAN; 22]; sorted_orders_with_bucket.len()];
    
    if !sorted_orders_with_bucket.is_empty() {
        let mut current_volume = sorted_orders_with_bucket[0].0;
        let mut group = BucketedOrderVolumeGroupV2::new(current_volume);
        
        for (idx, (bucket_vol, order)) in sorted_orders_with_bucket.iter().enumerate() {
            if (*bucket_vol - current_volume).abs() > f64::EPSILON {
                // 完成当前组
                if !group.indices.is_empty() {
                    order_groups.push(group);
                }
                // 开始新组
                current_volume = *bucket_vol;
                group = BucketedOrderVolumeGroupV2::new(current_volume);
            }
            
            group.add_record(idx, order.4, order.3, order.1);
        }
        
        // 添加最后一组
        if !group.indices.is_empty() {
            order_groups.push(group);
        }
    }

    // 5. 计算订单指标
    for group in order_groups.iter() {
        group.compute_all_indicators_ultra_fast_v2(&mut order_results, min_count, use_flag);
    }

    // 6. 映射回交易记录，包含订单信息
    let mut results = vec![vec![f64::NAN; 27]; n]; // 增加到27列

    for i in 0..n {
        // 分别处理买单和卖单
        if ask_order_data[i] != 0 {
            let order_id = ask_order_data[i];
            if let Some(&order_idx) = order_map.get(&order_id) {
                // 复制22个计算指标
                for j in 0..22 {
                    results[i][j] = order_results[order_idx][j];
                }
                // 添加订单编号（第23列）
                results[i][22] = order_id as f64;
                // 添加买卖标识（第24列）：买单=1.0，卖单=0.0
                results[i][23] = 0.0; // V2版本中ask订单都是卖单

                // 添加订单聚合信息
                let (_, _is_bid, order_volume, order_price, order_time) = &sorted_orders_with_bucket[order_idx].1;
                // 添加订单volume总量（第25列）
                results[i][24] = *order_volume;
                // 添加订单最后时间（第26列）
                results[i][25] = *order_time;
                // 添加订单加权平均价格（第27列）
                results[i][26] = *order_price;
            }
        }

        if bid_order_data[i] != 0 {
            let order_id = bid_order_data[i];
            if let Some(&order_idx) = order_map.get(&order_id) {
                // 复制22个计算指标
                for j in 0..22 {
                    results[i][j] = order_results[order_idx][j];
                }
                // 添加订单编号（第23列）
                results[i][22] = order_id as f64;
                // 添加买卖标识（第24列）：买单=1.0，卖单=0.0
                results[i][23] = 1.0; // V2版本中bid订单都是买单

                // 添加订单聚合信息
                let (_, _is_bid, order_volume, order_price, order_time) = &sorted_orders_with_bucket[order_idx].1;
                // 添加订单volume总量（第25列）
                results[i][24] = *order_volume;
                // 添加订单最后时间（第26列）
                results[i][25] = *order_time;
                // 添加订单加权平均价格（第27列）
                results[i][26] = *order_price;
            }
        }
    }

    let result_array = PyArray2::from_vec2(py, &results)?;
    let column_names = get_order_column_names();

    Ok((result_array.to_owned(), column_names))
}