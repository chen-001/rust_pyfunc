use pyo3::prelude::*;
use numpy::{PyArray2, PyReadonlyArray1};
use std::collections::HashMap;

/// 异常挂单区间特征提取器
/// 
/// 输入参数：
/// - exchtime: 交易时间戳数组(纳秒)
/// - number: 档位编号数组(1-10)
/// - price: 挂单价格数组 
/// - volume: 挂单量数组
/// - volume_percentile: 异常阈值分位数，默认0.9
/// - min_duration: 最小持续行数，默认1
/// 
/// 返回：
/// - 特征矩阵(N×24)和特征名称列表
#[pyfunction]
#[pyo3(signature = (exchtime, number, price, volume, volume_percentile = 0.9, min_duration = 1))]
pub fn analyze_asks(
    exchtime: PyReadonlyArray1<f64>,
    number: PyReadonlyArray1<i32>,
    price: PyReadonlyArray1<f64>,
    volume: PyReadonlyArray1<i64>,
    volume_percentile: f64,
    min_duration: usize,
) -> PyResult<(Py<PyArray2<f64>>, Vec<String>)> {
    let py = unsafe { Python::assume_gil_acquired() };
    
    let exchtime_slice = exchtime.as_array();
    let number_slice = number.as_array();
    let price_slice = price.as_array();
    let volume_slice = volume.as_array();
    
    let n = exchtime_slice.len();
    if n != number_slice.len() || n != price_slice.len() || n != volume_slice.len() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "输入数组长度不一致"
        ));
    }
    
    if n == 0 {
        let empty_result = PyArray2::zeros(py, (0, 24), false);
        return Ok((empty_result.to_owned(), get_feature_names()));
    }
    
    // 转换为Rust切片
    let exchtime_data = exchtime_slice.as_slice().unwrap();
    let number_data = number_slice.as_slice().unwrap();
    let price_data = price_slice.as_slice().unwrap();
    let volume_data = volume_slice.as_slice().unwrap();
    
    // 调用核心分析函数
    let abnormal_segments = detect_abnormal_segments(
        exchtime_data, 
        number_data, 
        price_data, 
        volume_data, 
        volume_percentile, 
        min_duration
    );
    
    // 计算特征
    let features = calculate_features(&abnormal_segments, exchtime_data, number_data, price_data, volume_data);
    
    // 转换为NumPy数组
    let num_segments = features.len();
    let result_array = PyArray2::zeros(py, (num_segments, 24), false);
    
    {
        let mut result_slice = unsafe { result_array.as_array_mut() };
        for (i, row) in features.iter().enumerate() {
            for (j, &value) in row.iter().enumerate() {
                unsafe {
                    *result_slice.uget_mut((i, j)) = value;
                }
            }
        }
    }
    
    Ok((result_array.to_owned(), get_feature_names()))
}

/// 异常挂单区间数据结构
#[derive(Debug, Clone)]
struct AbnormalSegment {
    start_row: usize,
    end_row: usize,
    level: i32,              // 异常档位(3-9)
    price: f64,              // 异常价格
    start_time: f64,         // 起始时间(纳秒)
    end_time: f64,          // 结束时间(纳秒)
    duration_rows: usize,    // 持续行数
}

/// 检测异常挂单区间
fn detect_abnormal_segments(
    exchtime: &[f64],
    number: &[i32], 
    price: &[f64],
    volume: &[i64],
    volume_percentile: f64,
    min_duration: usize,
) -> Vec<AbnormalSegment> {
    let n = exchtime.len();
    let mut segments = Vec::new();
    
    if n == 0 {
        return segments;
    }
    
    // 计算全局volume阈值
    let mut sorted_volumes: Vec<i64> = volume.to_vec();
    sorted_volumes.sort_unstable();
    let threshold_idx = ((sorted_volumes.len() as f64) * volume_percentile) as usize;
    let volume_threshold = if threshold_idx < sorted_volumes.len() {
        sorted_volumes[threshold_idx] as f64
    } else {
        sorted_volumes[sorted_volumes.len() - 1] as f64
    };
    
    // 按时间戳分组数据 - 每个时间戳应该有10行数据(1-10档)
    let mut time_groups = HashMap::new();
    for i in 0..n {
        let time = exchtime[i];
        let entry = time_groups.entry(time as i64).or_insert(Vec::new());
        entry.push(i);
    }
    
    let mut sorted_times: Vec<i64> = time_groups.keys().cloned().collect();
    sorted_times.sort_unstable();
    
    // 检测每个时刻的异常挂单
    let mut tracked_abnormals: HashMap<String, AbnormalSegment> = HashMap::new(); // price -> segment
    
    for &time in &sorted_times {
        let rows = time_groups.get(&time).unwrap();
        
        // 检查当前时刻是否有异常挂单
        let mut current_abnormal: Option<(usize, i32, f64, i64)> = None; // (row_idx, level, price, volume)
        
        // 找到当前时刻volume最大的档位（3-9档）
        let mut max_volume = 0i64;
        let mut max_row = None;
        
        for &row_idx in rows {
            let level = number[row_idx];
            let vol = volume[row_idx];
            
            if level >= 3 && level <= 9 && vol as f64 >= volume_threshold && vol > max_volume {
                max_volume = vol;
                max_row = Some((row_idx, level, price[row_idx], vol));
            }
        }
        
        if let Some((row_idx, level, px, vol)) = max_row {
            current_abnormal = Some((row_idx, level, px, vol));
        }
        
        // 更新追踪的异常挂单
        let price_key = format!("{:.6}", current_abnormal.as_ref().map_or(0.0, |(_, _, p, _)| *p));
        
        match current_abnormal {
            Some((row_idx, level, px, _)) => {
                // 检查是否是已存在的异常挂单的延续
                if let Some(segment) = tracked_abnormals.get_mut(&price_key) {
                    // 检查档位是否在允许范围内 (±1)
                    if (level - segment.level).abs() <= 1 && (px - segment.price).abs() < f64::EPSILON {
                        // 延续现有区间
                        segment.end_row = row_idx;
                        segment.end_time = exchtime[row_idx];
                        segment.duration_rows += 1;
                    } else {
                        // 结束旧区间，开始新区间
                        if segment.duration_rows >= min_duration {
                            segments.push(segment.clone());
                        }
                        tracked_abnormals.insert(price_key.clone(), AbnormalSegment {
                            start_row: row_idx,
                            end_row: row_idx,
                            level,
                            price: px,
                            start_time: exchtime[row_idx],
                            end_time: exchtime[row_idx],
                            duration_rows: 1,
                        });
                    }
                } else {
                    // 开始新的异常区间
                    tracked_abnormals.insert(price_key, AbnormalSegment {
                        start_row: row_idx,
                        end_row: row_idx,
                        level,
                        price: px,
                        start_time: exchtime[row_idx],
                        end_time: exchtime[row_idx],
                        duration_rows: 1,
                    });
                }
            }
            None => {
                // 当前时刻没有异常挂单，结束所有追踪的区间
                for (_, segment) in tracked_abnormals.drain() {
                    if segment.duration_rows >= min_duration {
                        segments.push(segment);
                    }
                }
            }
        }
    }
    
    // 处理最后剩余的追踪区间
    for (_, segment) in tracked_abnormals {
        if segment.duration_rows >= min_duration {
            segments.push(segment);
        }
    }
    
    segments
}

/// 计算24个特征
fn calculate_features(
    segments: &[AbnormalSegment],
    exchtime: &[f64],
    number: &[i32],
    price: &[f64],
    volume: &[i64],
) -> Vec<[f64; 24]> {
    let mut features = Vec::with_capacity(segments.len());
    
    for segment in segments {
        let mut feat = [0.0f64; 24];
        
        // 收集区间内的数据
        let segment_data = collect_segment_data(segment, exchtime, number, price, volume);
        
        // 计算各种特征
        calculate_volume_change_features(segment, &segment_data, &mut feat);
        calculate_time_features(segment, &mut feat);
        calculate_ratio_features(&segment_data, segment.level, &mut feat);
        calculate_correlation_features(&segment_data, segment.level, &mut feat);
        calculate_statistical_features(&segment_data, &mut feat);
        calculate_cv_features(&segment_data, segment.level, &mut feat);
        calculate_price_change_features(segment, &segment_data, &mut feat);
        
        features.push(feat);
    }
    
    features
}

/// 区间数据结构
struct SegmentData {
    time_groups: Vec<TimeGroup>,  // 按时间分组的数据
}

struct TimeGroup {
    time: f64,
    levels: [i64; 10],    // 1-10档的volume，索引0对应1档
    prices: [f64; 10],    // 1-10档的price
}

/// 收集区间数据
fn collect_segment_data(
    segment: &AbnormalSegment,
    exchtime: &[f64],
    number: &[i32],
    price: &[f64],
    volume: &[i64],
) -> SegmentData {
    let mut time_groups_map: HashMap<i64, TimeGroup> = HashMap::new();
    
    // 找到区间内的所有行
    for i in 0..exchtime.len() {
        if exchtime[i] >= segment.start_time && exchtime[i] <= segment.end_time {
            let time_key = exchtime[i] as i64;
            let entry = time_groups_map.entry(time_key).or_insert(TimeGroup {
                time: exchtime[i],
                levels: [0; 10],
                prices: [0.0; 10],
            });
            
            let level_idx = (number[i] - 1) as usize;
            if level_idx < 10 {
                entry.levels[level_idx] = volume[i];
                entry.prices[level_idx] = price[i];
            }
        }
    }
    
    let mut time_groups: Vec<TimeGroup> = time_groups_map.into_values().collect();
    time_groups.sort_by(|a, b| a.time.partial_cmp(&b.time).unwrap());
    
    SegmentData { time_groups }
}

/// 计算volume变化量特征 (特征0-6)
fn calculate_volume_change_features(
    segment: &AbnormalSegment,
    data: &SegmentData,
    feat: &mut [f64; 24],
) {
    if data.time_groups.len() < 2 {
        return;
    }
    
    let x_idx = (segment.level - 1) as usize;
    let mut x_changes = Vec::new();
    let mut x_minus_1_changes = Vec::new();
    let mut x_plus_1_changes = Vec::new();
    let mut left_changes = Vec::new();  // 1 to x-1档的变化
    let mut right_changes = Vec::new(); // x+1 to 10档的变化
    
    for i in 1..data.time_groups.len() {
        let prev = &data.time_groups[i-1];
        let curr = &data.time_groups[i];
        
        // x档变化量
        let x_change = curr.levels[x_idx] - prev.levels[x_idx];
        x_changes.push(x_change as f64);
        
        // x-1档变化量
        if x_idx > 0 {
            let x_minus_1_change = curr.levels[x_idx-1] - prev.levels[x_idx-1];
            x_minus_1_changes.push(x_minus_1_change as f64);
        }
        
        // x+1档变化量  
        if x_idx < 9 {
            let x_plus_1_change = curr.levels[x_idx+1] - prev.levels[x_idx+1];
            x_plus_1_changes.push(x_plus_1_change as f64);
        }
        
        // 1到x-1档变化量总和
        let left_sum_change: i64 = (0..x_idx).map(|j| curr.levels[j] - prev.levels[j]).sum();
        left_changes.push(left_sum_change as f64);
        
        // x+1到10档变化量总和
        let right_sum_change: i64 = ((x_idx+1)..10).map(|j| curr.levels[j] - prev.levels[j]).sum();
        right_changes.push(right_sum_change as f64);
    }
    
    // 0. x档volume变化量的均值
    feat[0] = calculate_mean(&x_changes);
    
    // 1. x-1档volume变化量的均值
    feat[1] = calculate_mean(&x_minus_1_changes);
    
    // 2. x+1档volume变化量的均值
    feat[2] = calculate_mean(&x_plus_1_changes);
    
    // 3. 1~x-1档各自Δvolume的均值的均值
    if x_idx > 0 {
        let mut individual_means = Vec::new();
        for j in 0..x_idx {
            let mut changes_for_level = Vec::new();
            for i in 1..data.time_groups.len() {
                let change = data.time_groups[i].levels[j] - data.time_groups[i-1].levels[j];
                changes_for_level.push(change as f64);
            }
            individual_means.push(calculate_mean(&changes_for_level));
        }
        feat[3] = calculate_mean(&individual_means);
    }
    
    // 4. x+1~10档各自Δvolume的均值的均值
    if x_idx < 9 {
        let mut individual_means = Vec::new();
        for j in (x_idx+1)..10 {
            let mut changes_for_level = Vec::new();
            for i in 1..data.time_groups.len() {
                let change = data.time_groups[i].levels[j] - data.time_groups[i-1].levels[j];
                changes_for_level.push(change as f64);
            }
            individual_means.push(calculate_mean(&changes_for_level));
        }
        feat[4] = calculate_mean(&individual_means);
    }
    
    // 5. 1~x-1档Δvolume总和的均值
    feat[5] = calculate_mean(&left_changes);
    
    // 6. x+1~10档Δvolume总和的均值
    feat[6] = calculate_mean(&right_changes);
}

/// 计算时间特征 (特征7-8)
fn calculate_time_features(segment: &AbnormalSegment, feat: &mut [f64; 24]) {
    // 7. 异常持续时间(秒)
    feat[7] = (segment.end_time - segment.start_time) / 1e9;
    
    // 8. 异常持续行数
    feat[8] = segment.duration_rows as f64;
}

/// 计算比例特征 (特征9-10)
fn calculate_ratio_features(data: &SegmentData, x_level: i32, feat: &mut [f64; 24]) {
    if data.time_groups.is_empty() {
        return;
    }
    
    let x_idx = (x_level - 1) as usize;
    let mut x_ratios = Vec::new();
    let mut total_volumes = Vec::new();
    
    for group in &data.time_groups {
        let total_volume: i64 = group.levels.iter().sum();
        total_volumes.push(total_volume as f64);
        
        if total_volume > 0 && x_idx < 10 {
            // 正确计算异常档位的volume占比
            let x_ratio = group.levels[x_idx] as f64 / total_volume as f64;
            x_ratios.push(x_ratio);
        }
    }
    
    // 9. 每期x档volume/当期1~10档volume总和的均值
    feat[9] = calculate_mean(&x_ratios);
    
    // 10. 每期1~10档volume总和的均值
    feat[10] = calculate_mean(&total_volumes);
}

/// 计算相关性特征 (特征11-13)
fn calculate_correlation_features(data: &SegmentData, x_level: i32, feat: &mut [f64; 24]) {
    if data.time_groups.len() < 2 {
        return;
    }
    
    let x_idx = (x_level - 1) as usize;
    let mut correlations_1_10 = Vec::new();
    let mut correlations_1_x = Vec::new();
    let mut correlations_x_10 = Vec::new();
    
    for group in &data.time_groups {
        // 1~10档volume与[1..10]的相关系数
        let volumes_1_10: Vec<f64> = group.levels.iter().map(|&v| v as f64).collect();
        let indices_1_10: Vec<f64> = (1..=10).map(|i| i as f64).collect();
        correlations_1_10.push(calculate_correlation(&volumes_1_10, &indices_1_10));
        
        // 1~x档volume与[1..x]的相关系数
        let volumes_1_x: Vec<f64> = group.levels[0..=x_idx].iter().map(|&v| v as f64).collect();
        let indices_1_x: Vec<f64> = (1..=(x_idx+1)).map(|i| i as f64).collect();
        correlations_1_x.push(calculate_correlation(&volumes_1_x, &indices_1_x));
        
        // x~10档volume与[x..10]的相关系数
        let volumes_x_10: Vec<f64> = group.levels[x_idx..].iter().map(|&v| v as f64).collect();
        let indices_x_10: Vec<f64> = ((x_idx+1)..=10).map(|i| i as f64).collect();
        correlations_x_10.push(calculate_correlation(&volumes_x_10, &indices_x_10));
    }
    
    // 11. 每期1~10档volume与[1..10]的Pearson相关系数的均值
    feat[11] = calculate_mean(&correlations_1_10);
    
    // 12. 每期1~x档volume与[1..x]的Pearson相关系数的均值
    feat[12] = calculate_mean(&correlations_1_x);
    
    // 13. 每期x~10档volume与[x..10]的Pearson相关系数的均值
    feat[13] = calculate_mean(&correlations_x_10);
}

/// 计算统计特征 (特征14-17)
fn calculate_statistical_features(data: &SegmentData, feat: &mut [f64; 24]) {
    if data.time_groups.is_empty() {
        return;
    }
    
    let mut stds = Vec::new();
    let mut skewnesses = Vec::new();
    let mut kurtoses = Vec::new();
    let mut autocorrs = Vec::new();
    
    for group in &data.time_groups {
        let volumes: Vec<f64> = group.levels.iter().map(|&v| v as f64).collect();
        
        // 标准差
        let mean = calculate_mean(&volumes);
        let std = calculate_std(&volumes, mean);
        stds.push(std);
        
        // 偏度
        skewnesses.push(calculate_skewness(&volumes, mean, std));
        
        // 峰度
        kurtoses.push(calculate_kurtosis(&volumes, mean, std));
    }
    
    // 自相关系数需要用时间序列数据
    if data.time_groups.len() > 1 {
        for i in 0..10 {
            let time_series: Vec<f64> = data.time_groups.iter().map(|g| g.levels[i] as f64).collect();
            autocorrs.push(calculate_autocorrelation(&time_series));
        }
    }
    
    // 14. 每期1~10档volume的标准差的均值
    feat[14] = calculate_mean(&stds);
    
    // 15. 每期1~10档volume的偏度的均值
    feat[15] = calculate_mean(&skewnesses);
    
    // 16. 每期1~10档volume的峰度的均值
    feat[16] = calculate_mean(&kurtoses);
    
    // 17. 每期1~10档volume的一阶自相关系数的均值
    feat[17] = calculate_mean(&autocorrs);
}

/// 计算变异系数特征 (特征18-19, 21-23)
fn calculate_cv_features(data: &SegmentData, x_level: i32, feat: &mut [f64; 24]) {
    if data.time_groups.is_empty() {
        return;
    }
    
    let x_idx = (x_level - 1) as usize;
    let mut cvs_1_x = Vec::new();
    let mut cvs_x_10 = Vec::new();
    
    for group in &data.time_groups {
        // 1~x档volume的cv
        let volumes_1_x: Vec<f64> = group.levels[0..=x_idx].iter().map(|&v| v as f64).collect();
        let mean_1_x = calculate_mean(&volumes_1_x);
        let std_1_x = calculate_std(&volumes_1_x, mean_1_x);
        let cv_1_x = if mean_1_x.abs() > f64::EPSILON { std_1_x / mean_1_x } else { 0.0 };
        cvs_1_x.push(cv_1_x);
        
        // x~10档volume的cv
        let volumes_x_10: Vec<f64> = group.levels[x_idx..].iter().map(|&v| v as f64).collect();
        let mean_x_10 = calculate_mean(&volumes_x_10);
        let std_x_10 = calculate_std(&volumes_x_10, mean_x_10);
        let cv_x_10 = if mean_x_10.abs() > f64::EPSILON { std_x_10 / mean_x_10 } else { 0.0 };
        cvs_x_10.push(cv_x_10);
    }
    
    // 18. 每期1~x档volume的cv的均值
    feat[18] = calculate_mean(&cvs_1_x);
    
    // 19. 每期x~10档volume的cv的均值
    feat[19] = calculate_mean(&cvs_x_10);
    
    // 21-23: x档、x-1档、x+1档volume的cv
    if data.time_groups.len() > 1 {
        // x档volume的时间序列cv
        let x_time_series: Vec<f64> = data.time_groups.iter().map(|g| g.levels[x_idx] as f64).collect();
        let x_mean = calculate_mean(&x_time_series);
        let x_std = calculate_std(&x_time_series, x_mean);
        feat[21] = if x_mean.abs() > f64::EPSILON { x_std / x_mean } else { 0.0 };
        
        // x-1档volume的cv
        if x_idx > 0 {
            let x_minus_1_series: Vec<f64> = data.time_groups.iter().map(|g| g.levels[x_idx-1] as f64).collect();
            let x_minus_1_mean = calculate_mean(&x_minus_1_series);
            let x_minus_1_std = calculate_std(&x_minus_1_series, x_minus_1_mean);
            feat[22] = if x_minus_1_mean.abs() > f64::EPSILON { x_minus_1_std / x_minus_1_mean } else { 0.0 };
        }
        
        // x+1档volume的cv
        if x_idx < 9 {
            let x_plus_1_series: Vec<f64> = data.time_groups.iter().map(|g| g.levels[x_idx+1] as f64).collect();
            let x_plus_1_mean = calculate_mean(&x_plus_1_series);
            let x_plus_1_std = calculate_std(&x_plus_1_series, x_plus_1_mean);
            feat[23] = if x_plus_1_mean.abs() > f64::EPSILON { x_plus_1_std / x_plus_1_mean } else { 0.0 };
        }
    }
}

/// 计算价格变化特征 (特征20)
fn calculate_price_change_features(segment: &AbnormalSegment, data: &SegmentData, feat: &mut [f64; 24]) {
    // 20. 异常区间内price变化率
    if data.time_groups.len() < 2 {
        feat[20] = 0.0;
        return;
    }
    
    let x_idx = (segment.level - 1) as usize;
    if x_idx >= 10 {
        feat[20] = 0.0;
        return;
    }
    
    // 获取异常档位在区间内的首末价格
    let first_price = data.time_groups.first()
        .map(|g| g.prices[x_idx])
        .unwrap_or(segment.price);
    
    let last_price = data.time_groups.last()
        .map(|g| g.prices[x_idx])
        .unwrap_or(segment.price);
    
    // 计算价格变化率
    if first_price.abs() > f64::EPSILON {
        feat[20] = (last_price - first_price) / first_price;
    } else {
        feat[20] = 0.0;
    }
}

/// 辅助函数：计算均值
fn calculate_mean(data: &[f64]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    data.iter().sum::<f64>() / data.len() as f64
}

/// 辅助函数：计算标准差
fn calculate_std(data: &[f64], mean: f64) -> f64 {
    if data.len() < 2 {
        return 0.0;
    }
    
    let variance: f64 = data.iter()
        .map(|&x| (x - mean).powi(2))
        .sum::<f64>() / (data.len() - 1) as f64;
    
    variance.sqrt()
}

/// 辅助函数：计算偏度
fn calculate_skewness(data: &[f64], mean: f64, std: f64) -> f64 {
    if data.len() < 3 || std.abs() < f64::EPSILON {
        return 0.0;
    }
    
    let n = data.len() as f64;
    let skew: f64 = data.iter()
        .map(|&x| ((x - mean) / std).powi(3))
        .sum::<f64>() / n;
    
    if data.len() > 2 {
        let adj_factor = (n * (n - 1.0)).sqrt() / (n - 2.0);
        skew * adj_factor
    } else {
        skew
    }
}

/// 辅助函数：计算峰度
fn calculate_kurtosis(data: &[f64], mean: f64, std: f64) -> f64 {
    if data.len() < 4 || std.abs() < f64::EPSILON {
        return 0.0;
    }
    
    let n = data.len() as f64;
    let kurt: f64 = data.iter()
        .map(|&x| ((x - mean) / std).powi(4))
        .sum::<f64>() / n;
    
    if data.len() > 3 {
        let adj_factor = ((n - 1.0) / ((n - 2.0) * (n - 3.0))) * 
                        ((n + 1.0) * kurt - 3.0 * (n - 1.0));
        adj_factor
    } else {
        kurt - 3.0
    }
}

/// 辅助函数：计算Pearson相关系数
fn calculate_correlation(x: &[f64], y: &[f64]) -> f64 {
    if x.len() != y.len() || x.len() < 2 {
        return 0.0;
    }
    
    let n = x.len() as f64;
    let x_mean = calculate_mean(x);
    let y_mean = calculate_mean(y);
    
    let mut numerator = 0.0;
    let mut x_variance = 0.0;
    let mut y_variance = 0.0;
    
    for i in 0..x.len() {
        let x_dev = x[i] - x_mean;
        let y_dev = y[i] - y_mean;
        numerator += x_dev * y_dev;
        x_variance += x_dev * x_dev;
        y_variance += y_dev * y_dev;
    }
    
    let denominator = (x_variance * y_variance).sqrt();
    if denominator.abs() < f64::EPSILON {
        return 0.0;
    }
    
    numerator / denominator
}

/// 辅助函数：计算一阶自相关系数
fn calculate_autocorrelation(data: &[f64]) -> f64 {
    if data.len() < 2 {
        return 0.0;
    }
    
    let x1 = &data[0..data.len()-1];
    let x2 = &data[1..data.len()];
    
    calculate_correlation(x1, x2)
}

/// 获取特征名称列表
fn get_feature_names() -> Vec<String> {
    vec![
        "x档volume变化量均值".to_string(),              // 0
        "x-1档volume变化量均值".to_string(),            // 1  
        "x+1档volume变化量均值".to_string(),            // 2
        "1到x-1档各自变化量均值的均值".to_string(),       // 3
        "x+1到10档各自变化量均值的均值".to_string(),      // 4
        "1到x-1档变化量总和均值".to_string(),            // 5
        "x+1到10档变化量总和均值".to_string(),           // 6
        "异常持续时间_秒".to_string(),                  // 7
        "异常持续行数".to_string(),                     // 8
        "x档volume占比均值".to_string(),                // 9
        "1到10档volume总和均值".to_string(),            // 10
        "1到10档volume与档位相关系数均值".to_string(),   // 11
        "1到x档volume与档位相关系数均值".to_string(),    // 12
        "x到10档volume与档位相关系数均值".to_string(),   // 13
        "1到10档volume标准差均值".to_string(),          // 14
        "1到10档volume偏度均值".to_string(),            // 15
        "1到10档volume峰度均值".to_string(),            // 16
        "1到10档volume一阶自相关均值".to_string(),      // 17
        "1到x档volume变异系数均值".to_string(),         // 18
        "x到10档volume变异系数均值".to_string(),        // 19
        "异常区间price变化率".to_string(),              // 20
        "x档volume变异系数".to_string(),                // 21
        "x-1档volume变异系数".to_string(),              // 22
        "x+1档volume变异系数".to_string(),              // 23
    ]
}