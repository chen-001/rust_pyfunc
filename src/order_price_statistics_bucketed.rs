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

/// 快速定位volume组范围（分桶版本）
fn find_bucketed_volume_ranges(volumes: &[f64]) -> Vec<(f64, usize, usize)> {
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

#[derive(Debug)]
struct BucketedTradeVolumeGroup {
    indices: Vec<usize>,      // 原始数据索引
    times: Vec<f64>,          // 时间数组
    prices: Vec<f64>,         // 对应的价格
    flags: Vec<i32>,          // 对应的flag
    buy_indices: Vec<usize>,  // 买单在组内的位置
    sell_indices: Vec<usize>, // 卖单在组内的位置
}

impl BucketedTradeVolumeGroup {
    fn new() -> Self {
        Self {
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

    /// 批量计算所有百分比档位的价格统计（核心优化：一次排序，多次使用）
    fn find_nearest_same_direction_trades_batch(
        &self,
        current_group_idx: usize,
        target_indices: &[usize],
        max_counts: &[usize],
    ) -> Vec<Vec<f64>> {
        if target_indices.is_empty() {
            return vec![Vec::new(); max_counts.len()];
        }

        let current_time = self.times[current_group_idx];
        let mut time_distances: Vec<(f64, f64)> = Vec::with_capacity(target_indices.len());

        // 计算时间距离
        for &target_idx in target_indices.iter() {
            if target_idx != current_group_idx {
                let time_diff = (current_time - self.times[target_idx]).abs();
                let price = self.prices[target_idx];
                time_distances.push((time_diff, price));
            }
        }

        // 按时间距离排序（只排序一次！）
        time_distances.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        // 批量提取不同数量的价格
        let mut results = Vec::with_capacity(max_counts.len());
        for &max_count in max_counts.iter() {
            let count = time_distances.len().min(max_count);
            let mut prices: Vec<f64> = Vec::with_capacity(count);
            for i in 0..count {
                prices.push(time_distances[i].1);
            }
            results.push(prices);
        }

        results
    }

    /// 批量计算该volume组所有记录的价格统计指标（优化版本：批量计算，避免重复排序）
    fn compute_price_statistics(
        &self,
        means: &mut [Vec<f64>],
        stds: &mut [Vec<f64>],
        min_count: usize,
        use_flag: &str,
    ) {
        let group_size = self.indices.len();
        if group_size < min_count {
            return;
        }

        // 百分比档位
        let percentages = [0.01, 0.02, 0.03, 0.04, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50];

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

            let orig_idx = self.indices[current_group_idx];
            let max_available = target_indices.len();

            // 预计算所有百分比档位需要的数量
            let mut max_counts = Vec::with_capacity(10);
            let mut valid_pct_indices = Vec::with_capacity(10);

            for (pct_idx, &pct) in percentages.iter().enumerate() {
                let count = ((max_available as f64 * pct).ceil() as usize)
                    .max(1)
                    .min(max_available);
                if count >= min_count {
                    max_counts.push(count);
                    valid_pct_indices.push(pct_idx);
                }
            }

            if max_counts.is_empty() {
                continue;
            }

            // 批量计算所有百分比档位的价格（核心优化：一次排序，获取所有结果！）
            let price_batches = self.find_nearest_same_direction_trades_batch(
                current_group_idx,
                &target_indices,
                &max_counts,
            );

            // 分别计算每个有效百分比档位的统计指标
            for (batch_idx, prices) in price_batches.into_iter().enumerate() {
                if prices.len() >= min_count {
                    let pct_idx = valid_pct_indices[batch_idx];

                    // 计算平均值
                    let sum: f64 = prices.iter().sum();
                    let mean = sum / prices.len() as f64;
                    means[orig_idx][pct_idx] = mean;

                    // 计算标准差
                    let variance_sum: f64 = prices
                        .iter()
                        .map(|&price| {
                            let diff = price - mean;
                            diff * diff
                        })
                        .sum();
                    let std = (variance_sum / prices.len() as f64).sqrt();
                    stds[orig_idx][pct_idx] = std;
                }
            }
        }
    }
}

#[pyfunction]
#[pyo3(signature = (volume, exchtime, price, flag, min_count=10, use_flag="same", num_buckets=20))]
pub fn calculate_trade_price_statistics_by_volume_bucketed(
    py: Python,
    volume: &PyArray1<f64>,
    exchtime: &PyArray1<i64>,
    price: &PyArray1<f64>,
    flag: &PyArray1<i32>,
    min_count: usize,
    use_flag: &str,
    num_buckets: usize,
) -> PyResult<(Py<PyArray2<f64>>, Py<PyArray2<f64>>, Vec<String>)> {
    let volume_slice = volume.readonly();
    let exchtime_slice = exchtime.readonly();
    let price_slice = price.readonly();
    let flag_slice = flag.readonly();

    let volume_data = volume_slice.as_slice()?;
    let exchtime_raw = exchtime_slice.as_slice()?;
    let price_data = price_slice.as_slice()?;
    let flag_data = flag_slice.as_slice()?;

    // 将纳秒时间戳转换为秒
    let exchtime_data: Vec<f64> = exchtime_raw.iter().map(|&t| t as f64 / 1e9).collect();

    let n = volume_data.len();
    if n == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "输入数组不能为空",
        ));
    }

    // 1. 创建体量分桶
    let bucketed_volumes = create_volume_buckets(volume_data, num_buckets);

    // 2. 按分桶后的volume和时间排序
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_unstable_by(|&a, &b| {
        bucketed_volumes[a]
            .partial_cmp(&bucketed_volumes[b])
            .unwrap()
            .then(exchtime_data[a].partial_cmp(&exchtime_data[b]).unwrap())
    });

    // 3. 重新组织数据
    let mut sorted_bucketed_volumes = Vec::with_capacity(n);
    let mut sorted_exchtime = Vec::with_capacity(n);
    let mut sorted_price = Vec::with_capacity(n);
    let mut sorted_flag = Vec::with_capacity(n);
    let mut orig_indices = Vec::with_capacity(n);

    for &idx in indices.iter() {
        sorted_bucketed_volumes.push(bucketed_volumes[idx]);
        sorted_exchtime.push(exchtime_data[idx]);
        sorted_price.push(price_data[idx]);
        sorted_flag.push(flag_data[idx]);
        orig_indices.push(idx);
    }

    // 4. 构建分桶后的volume组
    let bucketed_ranges = find_bucketed_volume_ranges(&sorted_bucketed_volumes);

    let mut volume_groups: Vec<BucketedTradeVolumeGroup> = Vec::new();

    for (_vol, start_idx, end_idx) in bucketed_ranges.iter() {
        let mut group = BucketedTradeVolumeGroup::new();

        for i in *start_idx..*end_idx {
            group.add_record(i, sorted_exchtime[i], sorted_price[i], sorted_flag[i]);
        }

        volume_groups.push(group);
    }

    // 5. 计算价格统计指标
    let mut sorted_means = vec![vec![f64::NAN; 10]; n];
    let mut sorted_stds = vec![vec![f64::NAN; 10]; n];

    for group in volume_groups.iter() {
        group.compute_price_statistics(&mut sorted_means, &mut sorted_stds, min_count, use_flag);
    }

    // 6. 将结果映射回原始顺序
    let mut means = vec![vec![f64::NAN; 10]; n];
    let mut stds = vec![vec![f64::NAN; 10]; n];

    for (sorted_idx, &orig_idx) in orig_indices.iter().enumerate() {
        for j in 0..10 {
            means[orig_idx][j] = sorted_means[sorted_idx][j];
            stds[orig_idx][j] = sorted_stds[sorted_idx][j];
        }
    }

    // 7. 创建结果数组
    let means_array = PyArray2::from_vec2(py, &means)?;
    let stds_array = PyArray2::from_vec2(py, &stds)?;
    let column_names = get_price_statistics_column_names();

    Ok((means_array.to_owned(), stds_array.to_owned(), column_names))
}

fn get_price_statistics_column_names() -> Vec<String> {
    let percentages = [
        "1%", "2%", "3%", "4%", "5%", "10%", "20%", "30%", "40%", "50%",
    ];
    let mut names = Vec::new();

    for &pct in percentages.iter() {
        names.push(format!("价格均值_{}", pct));
    }
    for &pct in percentages.iter() {
        names.push(format!("价格标准差_{}", pct));
    }

    names
}

// V2版本的订单volume组（分桶版本），基于订单类型而非交易标志
#[derive(Debug)]
struct BucketedOrderVolumeGroupV2 {
    indices: Vec<usize>,     // 原始数据索引
    times: Vec<f64>,         // 时间数组（已排序）
    vwap_prices: Vec<f64>,   // 订单的VWAP价格
    order_types: Vec<bool>,  // 对应的订单类型：true=买单，false=卖单
    ask_indices: Vec<usize>, // 卖单在组内的位置
    bid_indices: Vec<usize>, // 买单在组内的位置
}

impl BucketedOrderVolumeGroupV2 {
    fn new() -> Self {
        Self {
            indices: Vec::new(),
            times: Vec::new(),
            vwap_prices: Vec::new(),
            order_types: Vec::new(),
            ask_indices: Vec::new(),
            bid_indices: Vec::new(),
        }
    }

    fn add_order(&mut self, orig_idx: usize, time: f64, vwap_price: f64, is_bid: bool) {
        let group_idx = self.indices.len();

        self.indices.push(orig_idx);
        self.times.push(time);
        self.vwap_prices.push(vwap_price);
        self.order_types.push(is_bid);

        // 根据订单类型分类存储位置
        if is_bid {
            self.bid_indices.push(group_idx);
        } else {
            self.ask_indices.push(group_idx);
        }
    }

    /// 基于订单类型的时间距离计算
    fn find_nearest_same_type_orders(
        &self,
        current_group_idx: usize,
        target_indices: &[usize],
        max_count: usize,
    ) -> Vec<f64> {
        if target_indices.is_empty() {
            return Vec::new();
        }

        let current_time = self.times[current_group_idx];
        let mut time_distances: Vec<(f64, f64)> = Vec::new();

        // 计算时间距离
        for &target_idx in target_indices.iter() {
            if target_idx != current_group_idx {
                let time_diff = (current_time - self.times[target_idx]).abs();
                let vwap_price = self.vwap_prices[target_idx];
                time_distances.push((time_diff, vwap_price));
            }
        }

        // 按时间距离排序
        time_distances.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        // 限制返回数量并提取VWAP价格
        let count = time_distances.len().min(max_count);
        let mut vwap_prices: Vec<f64> = Vec::with_capacity(count);
        for i in 0..count {
            vwap_prices.push(time_distances[i].1);
        }

        vwap_prices
    }

    /// 基于订单类型的批量计算指标
    fn compute_vwap_statistics(
        &self,
        means: &mut [Vec<f64>],
        stds: &mut [Vec<f64>],
        min_count: usize,
        use_flag: &str,
    ) {
        let group_size = self.indices.len();
        if group_size < min_count {
            return;
        }

        // 百分比档位
        let percentages = [0.01, 0.02, 0.03, 0.04, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50];

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

        // 为每个订单计算指标
        for current_group_idx in 0..group_size {
            let target_indices = get_target_indices(current_group_idx);

            if target_indices.len() < min_count {
                continue;
            }

            let orig_idx = self.indices[current_group_idx];
            let max_available = target_indices.len();

            // 对每个百分比档位计算统计指标
            for (pct_idx, &pct) in percentages.iter().enumerate() {
                let count = ((max_available as f64 * pct).ceil() as usize)
                    .max(1)
                    .min(max_available);

                if count >= min_count {
                    let vwap_prices = self.find_nearest_same_type_orders(
                        current_group_idx,
                        &target_indices,
                        count,
                    );

                    if vwap_prices.len() >= min_count {
                        // 计算VWAP价格的平均值
                        let sum: f64 = vwap_prices.iter().sum();
                        let mean = sum / vwap_prices.len() as f64;
                        means[orig_idx][pct_idx] = mean;

                        // 计算VWAP价格的标准差
                        let variance_sum: f64 = vwap_prices
                            .iter()
                            .map(|&price| {
                                let diff = price - mean;
                                diff * diff
                            })
                            .sum();
                        let std = (variance_sum / vwap_prices.len() as f64).sqrt();
                        stds[orig_idx][pct_idx] = std;
                    }
                }
            }
        }
    }
}

#[pyfunction]
#[pyo3(signature = (volume, exchtime, price, flag, ask_order, bid_order, min_count=10, use_flag="same", num_buckets=20))]
pub fn calculate_trade_price_statistics_by_volume_v2_bucketed(
    py: Python,
    volume: &PyArray1<f64>,
    exchtime: &PyArray1<i64>,
    price: &PyArray1<f64>,
    flag: &PyArray1<i32>, // 该参数在V2版本中被忽略
    ask_order: &PyArray1<i64>,
    bid_order: &PyArray1<i64>,
    min_count: usize,
    use_flag: &str,
    num_buckets: usize,
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

    // 2. 对订单volume进行分桶
    let order_volumes: Vec<f64> = orders.iter().map(|(_, _, vol, _, _)| *vol).collect();
    let bucketed_order_volumes = create_volume_buckets(&order_volumes, num_buckets);

    // 3. 按分桶后的订单volume和时间排序
    let mut order_indices: Vec<usize> = (0..orders.len()).collect();
    order_indices.sort_unstable_by(|&a, &b| {
        bucketed_order_volumes[a]
            .partial_cmp(&bucketed_order_volumes[b])
            .unwrap()
            .then(orders[a].4.partial_cmp(&orders[b].4).unwrap())
    });

    // 4. 重新组织订单数据
    let mut sorted_bucketed_volumes = Vec::with_capacity(orders.len());
    let mut sorted_orders = Vec::with_capacity(orders.len());
    let mut orig_order_indices = Vec::with_capacity(orders.len());

    for &idx in order_indices.iter() {
        sorted_bucketed_volumes.push(bucketed_order_volumes[idx]);
        sorted_orders.push(orders[idx].clone());
        orig_order_indices.push(idx);
    }

    // 5. 构建分桶后的订单volume组
    let bucketed_order_ranges = find_bucketed_volume_ranges(&sorted_bucketed_volumes);

    let mut order_groups: Vec<BucketedOrderVolumeGroupV2> = Vec::new();

    for (_vol, start_idx, end_idx) in bucketed_order_ranges.iter() {
        let mut group = BucketedOrderVolumeGroupV2::new();

        for i in *start_idx..*end_idx {
            let (_, is_bid, _, vwap_price, time) = sorted_orders[i];
            group.add_order(i, time, vwap_price, is_bid);
        }

        order_groups.push(group);
    }

    // 6. 计算订单VWAP统计指标
    let mut sorted_order_means = vec![vec![f64::NAN; 10]; orders.len()];
    let mut sorted_order_stds = vec![vec![f64::NAN; 10]; orders.len()];

    for group in order_groups.iter() {
        group.compute_vwap_statistics(
            &mut sorted_order_means,
            &mut sorted_order_stds,
            min_count,
            use_flag,
        );
    }

    // 7. 将结果映射回原始订单顺序
    let mut order_means = vec![vec![f64::NAN; 10]; orders.len()];
    let mut order_stds = vec![vec![f64::NAN; 10]; orders.len()];

    for (sorted_idx, &orig_idx) in orig_order_indices.iter().enumerate() {
        for j in 0..10 {
            order_means[orig_idx][j] = sorted_order_means[sorted_idx][j];
            order_stds[orig_idx][j] = sorted_order_stds[sorted_idx][j];
        }
    }

    // 8. 映射回交易记录
    let mut means = vec![vec![f64::NAN; 10]; n];
    let mut stds = vec![vec![f64::NAN; 10]; n];

    for i in 0..n {
        // 分别处理买单和卖单
        if ask_order_data[i] != 0 {
            let order_id = ask_order_data[i];
            if let Some(&order_idx) = order_map.get(&order_id) {
                // 复制订单的统计指标
                for j in 0..10 {
                    means[i][j] = order_means[order_idx][j];
                    stds[i][j] = order_stds[order_idx][j];
                }
            }
        }

        if bid_order_data[i] != 0 {
            let order_id = bid_order_data[i];
            if let Some(&order_idx) = order_map.get(&order_id) {
                // 复制订单的统计指标
                for j in 0..10 {
                    means[i][j] = order_means[order_idx][j];
                    stds[i][j] = order_stds[order_idx][j];
                }
            }
        }
    }

    let means_array = PyArray2::from_vec2(py, &means)?;
    let stds_array = PyArray2::from_vec2(py, &stds)?;
    let column_names = get_price_statistics_column_names();

    Ok((means_array.to_owned(), stds_array.to_owned(), column_names))
}
/// 优化版本：极致性能的分桶统计计算函数（V3）
///
/// 针对13万数据量快速完成的极致优化版本
/// 核心思路：预排序 + 批量处理，避免对每个记录单独排序
///
/// 🚀 核心优化：
/// ==================
/// - 在volume组级别预排序时间索引
/// - 使用二分查找定位邻近记录
/// - 批量计算所有百分比档位
/// - 部分排序（只排序需要的元素）
#[pyfunction]
#[pyo3(signature = (volume, exchtime, price, flag, min_count=10, use_flag="same", num_buckets=20))]
pub fn calculate_trade_price_statistics_by_volume_bucketed_v3(
    py: Python,
    volume: &PyArray1<f64>,
    exchtime: &PyArray1<i64>,
    price: &PyArray1<f64>,
    flag: &PyArray1<i32>,
    min_count: usize,
    use_flag: &str,
    num_buckets: usize,
) -> PyResult<(Py<PyArray2<f64>>, Py<PyArray2<f64>>, Vec<String>)> {
    let volume_slice = volume.readonly();
    let exchtime_slice = exchtime.readonly();
    let price_slice = price.readonly();
    let flag_slice = flag.readonly();

    let volume_data = volume_slice.as_slice()?;
    let exchtime_raw = exchtime_slice.as_slice()?;
    let price_data = price_slice.as_slice()?;
    let flag_data = flag_slice.as_slice()?;

    // 将纳秒时间戳转换为秒
    let exchtime_data: Vec<f64> = exchtime_raw.iter().map(|&t| t as f64 * 1e-9).collect();

    let n = volume_data.len();
    if n == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "输入数组不能为空",
        ));
    }

    // 1. 创建体量分桶
    let bucketed_volumes = create_volume_buckets(volume_data, num_buckets);

    // 2. 按分桶后的volume和时间排序
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_unstable_by(|&a, &b| {
        bucketed_volumes[a]
            .partial_cmp(&bucketed_volumes[b])
            .unwrap()
            .then(exchtime_data[a].partial_cmp(&exchtime_data[b]).unwrap())
    });

    // 3. 重新组织数据
    let mut sorted_bucketed_volumes = Vec::with_capacity(n);
    let mut sorted_exchtime = Vec::with_capacity(n);
    let mut sorted_price = Vec::with_capacity(n);
    let mut sorted_flag = Vec::with_capacity(n);
    let mut orig_indices = Vec::with_capacity(n);

    for &idx in indices.iter() {
        sorted_bucketed_volumes.push(bucketed_volumes[idx]);
        sorted_exchtime.push(exchtime_data[idx]);
        sorted_price.push(price_data[idx]);
        sorted_flag.push(flag_data[idx]);
        orig_indices.push(idx);
    }

    // 4. 构建分桶后的volume组
    let bucketed_ranges = find_bucketed_volume_ranges(&sorted_bucketed_volumes);

    // 预分配结果数组
    let mut sorted_means = vec![vec![f64::NAN; 10]; n];
    let mut sorted_stds = vec![vec![f64::NAN; 10]; n];

    // 百分比档位
    let percentages = [0.01, 0.02, 0.03, 0.04, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50];

    // 5. 处理每个volume组（应用V3优化策略）
    for (_, start_idx, end_idx) in bucketed_ranges.iter() {
        let group_size = end_idx - start_idx;
        if group_size < min_count {
            continue;
        }

        // 构建买卖单的时间排序索引（关键优化：一次排序，多次使用）
        let mut buy_records: Vec<(f64, usize, f64)> = Vec::new(); // (time, group_idx, price)
        let mut sell_records: Vec<(f64, usize, f64)> = Vec::new();

        for i in 0..group_size {
            let sorted_idx = start_idx + i;
            let time = sorted_exchtime[sorted_idx];
            let price = sorted_price[sorted_idx];

            if sorted_flag[sorted_idx] == 66 {
                buy_records.push((time, i, price));
            } else if sorted_flag[sorted_idx] == 83 {
                sell_records.push((time, i, price));
            }
        }

        // 按时间排序（只排序一次！）
        buy_records.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        sell_records.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        // 预分配工作缓冲区
        let max_records = buy_records.len().max(sell_records.len());
        let mut distances_buffer: Vec<(f64, f64)> = Vec::with_capacity(max_records);

        // 批量处理每个记录
        for i in 0..group_size {
            let sorted_idx = start_idx + i;
            let current_flag = sorted_flag[sorted_idx];
            let current_time = sorted_exchtime[sorted_idx];

            // 选择目标记录集合
            let target_records = match use_flag {
                "same" => {
                    if current_flag == 66 {
                        &buy_records
                    } else {
                        &sell_records
                    }
                }
                "diff" => {
                    if current_flag == 66 {
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

            // 清空缓冲区
            distances_buffer.clear();

            // 计算时间距离（遍历已排序的记录）
            for &(time, group_idx, price) in target_records.iter() {
                if use_flag == "same" && group_idx == i {
                    continue; // 跳过自己
                }
                let time_diff = (current_time - time).abs();
                distances_buffer.push((time_diff, price));
            }

            let available = distances_buffer.len();
            if available < min_count {
                continue;
            }

            // 部分排序优化：只排序需要的部分
            let max_needed = ((available as f64 * 0.50).ceil() as usize).min(available);

            if max_needed < available {
                distances_buffer
                    .select_nth_unstable_by(max_needed, |a, b| a.0.partial_cmp(&b.0).unwrap());
                distances_buffer[..=max_needed]
                    .sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            } else {
                distances_buffer.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            }

            // 批量计算所有百分比档位（增量算法）
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

                sorted_means[sorted_idx][pct_idx] = mean;
                sorted_stds[sorted_idx][pct_idx] = std;
            }
        }
    }

    // 6. 将结果映射回原始顺序
    let mut means = vec![vec![f64::NAN; 10]; n];
    let mut stds = vec![vec![f64::NAN; 10]; n];

    for (sorted_idx, &orig_idx) in orig_indices.iter().enumerate() {
        for j in 0..10 {
            means[orig_idx][j] = sorted_means[sorted_idx][j];
            stds[orig_idx][j] = sorted_stds[sorted_idx][j];
        }
    }

    // 7. 创建结果数组
    let means_array = PyArray2::from_vec2(py, &means)?;
    let stds_array = PyArray2::from_vec2(py, &stds)?;
    let column_names = get_price_statistics_column_names();

    Ok((means_array.to_owned(), stds_array.to_owned(), column_names))
}
