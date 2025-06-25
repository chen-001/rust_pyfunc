use pyo3::prelude::*;
use ndarray::Array1;


/// 分析股票交易中的"以退为进"现象（纳秒版本）
/// 
/// 该函数分析当价格触及某个局部高点后回落，然后在该价格的异常大挂单量消失后
/// 成功突破该价格的现象。该版本包含局部高点去重功能，避免在同一价格水平的
/// 连续成交中重复识别相同的"以退为进"过程。
/// 
/// 参数说明：
/// ----------
/// trade_times : Vec<f64>
///     逐笔成交数据的时间戳序列（纳秒时间戳）
/// trade_prices : Vec<f64>
///     逐笔成交数据的价格序列
/// trade_volumes : Vec<f64>
///     逐笔成交数据的成交量序列
/// trade_flags : Vec<f64>
///     逐笔成交数据的标志序列（买卖方向），66表示主动买入，83表示主动卖出
/// orderbook_times : Vec<f64>
///     盘口快照数据的时间戳序列（纳秒时间戳）
/// orderbook_prices : Vec<f64>
///     盘口快照数据的价格序列
/// orderbook_volumes : Vec<f64>
///     盘口快照数据的挂单量序列
/// volume_percentile : float, optional
///     异常大挂单量的百分位数阈值，默认为99.0（即前1%）
/// time_window_minutes : float, optional
///     检查异常大挂单量的时间窗口（分钟），默认为1.0分钟
/// breakthrough_threshold : float, optional
///     突破阈值（百分比），默认为0.0（即只要高于局部高点任何幅度都算突破）
///     例如：0.1表示需要高出局部高点0.1%才算突破
/// dedup_time_seconds : float, optional
///     去重时间阈值（秒），默认为30.0。相同价格且时间间隔小于此值的局部高点将被视为重复
/// find_local_lows : bool, optional
///     是否查找局部低点，默认为False（查找局部高点）。
///     当为True时，分析"以进为退"现象：价格跌至局部低点后反弹，在该价格的异常大买单量消失后成功跌破该价格
/// 
/// 返回值：
/// -------
/// tuple
///     包含9个列表的元组：
///     - 过程期间的成交量
///     - 局部极值价格在盘口上时间最近的挂单量
///     - 过程开始后指定时间窗口内的成交量
///     - 过程期间的主动买入成交量占比
///     - 过程期间的价格种类数
///     - 过程期间价格相对局部极值的最大变化比例（高点为下降，低点为上升）
///     - 过程持续时间（秒）
///     - 过程开始时间（纳秒时间戳）
///     - 局部极值的价格
/// 
/// Python调用示例：
/// ```python
/// from rust_pyfunc import analyze_retreat_advance_v2
/// 
/// # 准备数据（纳秒时间戳）
/// trade_times = [1661743800000000000.0, 1661743860000000000.0, 1661743920000000000.0]
/// trade_prices = [10.0, 10.1, 10.2]
/// trade_volumes = [100.0, 200.0, 150.0]
/// trade_flags = [66.0, 66.0, 83.0]
/// 
/// orderbook_times = [1661743800000000000.0, 1661743860000000000.0]
/// orderbook_prices = [10.0, 10.1]
/// orderbook_volumes = [1000.0, 5000.0]
/// 
/// # 分析"以退为进"现象，使用2分钟时间窗口，0.1%突破阈值，60秒去重时间
/// results = analyze_retreat_advance_v2(
///     trade_times, trade_prices, trade_volumes, trade_flags,
///     orderbook_times, orderbook_prices, orderbook_volumes,
///     volume_percentile=95.0, time_window_minutes=2.0, breakthrough_threshold=0.1, dedup_time_seconds=60.0, find_local_lows=False
/// )
/// 
/// # 分析"以进为退"现象（局部低点版本）
/// results_lows = analyze_retreat_advance_v2(
///     trade_times, trade_prices, trade_volumes, trade_flags,
///     orderbook_times, orderbook_prices, orderbook_volumes,
///     volume_percentile=95.0, time_window_minutes=2.0, breakthrough_threshold=0.1, dedup_time_seconds=60.0, find_local_lows=True
/// )
/// 
/// process_volumes, large_volumes, time_window_volumes, buy_ratios, price_counts, max_changes, process_durations, process_start_times, extreme_prices = results
/// ```
#[pyfunction]
#[pyo3(signature = (
    trade_times, trade_prices, trade_volumes, trade_flags,
    orderbook_times, orderbook_prices, orderbook_volumes,
    volume_percentile=99.0,
    time_window_minutes=1.0,
    breakthrough_threshold=0.0,
    dedup_time_seconds=30.0,
    find_local_lows=false
))]
pub fn analyze_retreat_advance_v2(
    trade_times: Vec<f64>,
    trade_prices: Vec<f64>,
    trade_volumes: Vec<f64>,
    trade_flags: Vec<f64>,
    orderbook_times: Vec<f64>,
    orderbook_prices: Vec<f64>,
    orderbook_volumes: Vec<f64>,
    volume_percentile: Option<f64>,
    time_window_minutes: Option<f64>,
    breakthrough_threshold: Option<f64>,
    dedup_time_seconds: Option<f64>,
    find_local_lows: Option<bool>,
) -> PyResult<(
    Vec<f64>, // 过程期间的成交量
    Vec<f64>, // 过程期间首次观察到的异常大挂单量
    Vec<f64>, // 过程开始后指定时间窗口内的成交量
    Vec<f64>, // 过程期间的主动买入成交量占比
    Vec<f64>, // 过程期间的价格种类数
    Vec<f64>, // 过程期间价格相对局部高点的最大下降比例
    Vec<f64>, // 过程持续时间（秒）
    Vec<f64>, // 过程开始时间（纳秒时间戳）
    Vec<f64>, // 局部高点的价格
)> {
    // 直接使用Vec<f64>创建Array1
    let trade_times = Array1::from_vec(trade_times);
    let trade_prices = Array1::from_vec(trade_prices);
    let trade_volumes = Array1::from_vec(trade_volumes);
    let trade_flags = Array1::from_vec(trade_flags);
    let orderbook_times = Array1::from_vec(orderbook_times);
    let orderbook_prices = Array1::from_vec(orderbook_prices);
    let orderbook_volumes = Array1::from_vec(orderbook_volumes);
    
    let volume_percentile = volume_percentile.unwrap_or(99.0);
    let time_window_minutes = time_window_minutes.unwrap_or(1.0);
    let breakthrough_threshold = breakthrough_threshold.unwrap_or(0.0);
    let dedup_time_seconds = dedup_time_seconds.unwrap_or(30.0);
    let find_local_lows = find_local_lows.unwrap_or(false);
    
    // 验证输入数据长度一致性
    if trade_times.len() != trade_prices.len() || 
       trade_times.len() != trade_volumes.len() ||
       trade_times.len() != trade_flags.len() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "逐笔成交数据各列长度不一致"
        ));
    }
    
    if orderbook_times.len() != orderbook_prices.len() ||
       orderbook_times.len() != orderbook_volumes.len() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "盘口快照数据各列长度不一致"
        ));
    }
    
    // 步骤1：找到所有局部极值点（高点或低点）
    let local_peaks = find_local_extremes_v2(&trade_prices.view(), find_local_lows);
    
    // 步骤1.5：去除重复的局部极值点（相同价格且时间接近的极值点）
    let deduplicated_peaks = deduplicate_local_peaks_v2(&trade_times.view(), &trade_prices.view(), &local_peaks, dedup_time_seconds);
    
    // 步骤2：计算挂单量的百分位数阈值
    let volume_threshold = calculate_percentile_v2(&orderbook_volumes.view(), volume_percentile);
    
    // 步骤3：识别"以退为进"或"以进为退"过程
    let processes = identify_retreat_advance_processes_v2(
        &trade_times.view(), &trade_prices.view(), &trade_volumes.view(), &trade_flags.view(),
        &orderbook_times.view(), &orderbook_prices.view(), &orderbook_volumes.view(),
        &deduplicated_peaks, volume_threshold, time_window_minutes, breakthrough_threshold, find_local_lows
    );
    
    // 步骤4：计算每个过程的9个指标
    let (process_volumes, large_volumes, time_window_volumes, buy_ratios, price_counts, max_declines, process_durations, process_start_times, peak_prices) = 
        calculate_process_metrics_v2(
            &trade_times.view(), &trade_prices.view(), &trade_volumes.view(), &trade_flags.view(),
            &orderbook_times.view(), &orderbook_prices.view(), &orderbook_volumes.view(),
            &processes, time_window_minutes, find_local_lows
        );
    
    Ok((
        process_volumes,
        large_volumes,
        time_window_volumes,
        buy_ratios,
        price_counts,
        max_declines,
        process_durations,
        process_start_times,
        peak_prices,
    ))
}

/// 找到价格序列中的局部极值点（高点或低点）
fn find_local_extremes_v2(prices: &ndarray::ArrayView1<f64>, find_lows: bool) -> Vec<usize> {
    let mut peaks = Vec::new();
    let n = prices.len();
    
    if n < 3 {
        return peaks;
    }
    
    for i in 1..n-1 {
        let current_price = prices[i];
        
        // 向左查找第一个不同的价格
        let mut left_different = false;
        let mut left_lower = false;
        for j in (0..i).rev() {
            if (prices[j] - current_price).abs() > f64::EPSILON {
                left_different = true;
                left_lower = prices[j] < current_price;
                break;
            }
        }
        
        // 向右查找第一个不同的价格
        let mut right_different = false;
        let mut right_lower = false;
        for j in i+1..n {
            if (prices[j] - current_price).abs() > f64::EPSILON {
                right_different = true;
                right_lower = prices[j] < current_price;
                break;
            }
        }
        
        // 根据参数决定是找局部高点还是局部低点
        if find_lows {
            // 如果左右两边的第一个不同价格都比当前价格高，则为局部低点
            if left_different && right_different && !left_lower && !right_lower {
                peaks.push(i);
            }
        } else {
            // 如果左右两边的第一个不同价格都比当前价格低，则为局部高点
            if left_different && right_different && left_lower && right_lower {
                peaks.push(i);
            }
        }
    }
    
    peaks
}

/// 计算数组的百分位数
fn calculate_percentile_v2(values: &ndarray::ArrayView1<f64>, percentile: f64) -> f64 {
    let mut sorted_values: Vec<f64> = values.iter().copied().collect();
    sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    
    let index = (sorted_values.len() as f64 * percentile / 100.0) as usize;
    let index = index.min(sorted_values.len() - 1);
    
    sorted_values[index]
}

/// 表示一个"以退为进"过程
#[derive(Debug, Clone)]
struct RetreatAdvanceProcessV2 {
    peak_price: f64,        // 局部高点的价格
    start_time: f64,        // 过程开始时间（纳秒）
    end_time: f64,          // 过程结束时间（纳秒）
    start_index: usize,     // 过程开始的成交索引
    end_index: usize,       // 过程结束的成交索引
}

/// 识别"以退为进"或"以进为退"过程
fn identify_retreat_advance_processes_v2(
    trade_times: &ndarray::ArrayView1<f64>,
    trade_prices: &ndarray::ArrayView1<f64>,
    _trade_volumes: &ndarray::ArrayView1<f64>,
    _trade_flags: &ndarray::ArrayView1<f64>,
    orderbook_times: &ndarray::ArrayView1<f64>,
    orderbook_prices: &ndarray::ArrayView1<f64>,
    orderbook_volumes: &ndarray::ArrayView1<f64>,
    local_peaks: &[usize],
    volume_threshold: f64,
    time_window_minutes: f64,
    breakthrough_threshold: f64,
    find_local_lows: bool,
) -> Vec<RetreatAdvanceProcessV2> {
    let mut processes = Vec::new();
    
    for &peak_idx in local_peaks {
        let peak_price = trade_prices[peak_idx];
        let peak_time = trade_times[peak_idx];
        
        // 检查在局部极值点附近指定时间窗口内是否有异常大的挂单量
        let has_large_volume = check_large_volume_near_peak_v2(
            orderbook_times, orderbook_prices, orderbook_volumes,
            peak_price, peak_time, volume_threshold, time_window_minutes, find_local_lows
        );
        
        if !has_large_volume {
            continue;
        }
        
        // 寻找过程的结束点：价格成功突破局部极值点
        if let Some(end_idx) = find_breakthrough_point_v2(
            trade_times, trade_prices, peak_idx, peak_price, breakthrough_threshold, find_local_lows
        ) {
            let process = RetreatAdvanceProcessV2 {
                peak_price,
                start_time: peak_time,
                end_time: trade_times[end_idx],
                start_index: peak_idx,
                end_index: end_idx,
            };
            processes.push(process);
        }
    }
    
    processes
}

/// 检查局部极值点附近是否有异常大的挂单量（纳秒版本）
fn check_large_volume_near_peak_v2(
    orderbook_times: &ndarray::ArrayView1<f64>,
    orderbook_prices: &ndarray::ArrayView1<f64>,
    orderbook_volumes: &ndarray::ArrayView1<f64>,
    peak_price: f64,
    peak_time: f64,
    volume_threshold: f64,
    time_window_minutes: f64,
    find_local_lows: bool,
) -> bool {
    let time_window = time_window_minutes * 60.0 * 1_000_000_000.0; // 转换为纳秒
    
    for i in 0..orderbook_times.len() {
        let time_diff = (orderbook_times[i] - peak_time).abs();
        let price_diff = (orderbook_prices[i] - peak_price).abs();
        
        // 在时间窗口内且价格相近的挂单
        // 对于局部低点模式，我们可以设置稍微更宽松的价格容忍度
        let price_tolerance = if find_local_lows { 0.0015 } else { 0.001 }; // 0.15% vs 0.1%
        
        if time_diff <= time_window && price_diff < price_tolerance {
            if orderbook_volumes[i] >= volume_threshold {
                return true;
            }
        }
    }
    
    false
}

/// 寻找突破点：价格成功越过局部极值点（纳秒版本）
fn find_breakthrough_point_v2(
    trade_times: &ndarray::ArrayView1<f64>,
    trade_prices: &ndarray::ArrayView1<f64>,
    peak_idx: usize,
    peak_price: f64,
    breakthrough_threshold: f64,
    find_local_lows: bool,
) -> Option<usize> {
    let n = trade_prices.len();
    
    // 计算突破价格门槛
    let breakthrough_price = if find_local_lows {
        // 对于局部低点，突破意味着价格跌破低点
        peak_price * (1.0 - breakthrough_threshold / 100.0)
    } else {
        // 对于局部高点，突破意味着价格超过高点
        peak_price * (1.0 + breakthrough_threshold / 100.0)
    };
    
    // 从局部极值点之后开始查找
    for i in peak_idx + 1..n {
        let price_breakthrough = if find_local_lows {
            trade_prices[i] < breakthrough_price
        } else {
            trade_prices[i] > breakthrough_price
        };
        
        if price_breakthrough {
            return Some(i);
        }
        
        // 设置最大搜索时间窗口，避免无限搜索
        let time_diff = trade_times[i] - trade_times[peak_idx];
        if time_diff > 4.0 * 60.0 * 60.0 * 1_000_000_000.0 { // 4小时后仍未突破则放弃（纳秒）
            break;
        }
    }
    
    None
}

/// 计算每个过程的9个指标
fn calculate_process_metrics_v2(
    trade_times: &ndarray::ArrayView1<f64>,
    trade_prices: &ndarray::ArrayView1<f64>,
    trade_volumes: &ndarray::ArrayView1<f64>,
    trade_flags: &ndarray::ArrayView1<f64>,
    orderbook_times: &ndarray::ArrayView1<f64>,
    orderbook_prices: &ndarray::ArrayView1<f64>,
    orderbook_volumes: &ndarray::ArrayView1<f64>,
    processes: &[RetreatAdvanceProcessV2],
    time_window_minutes: f64,
    find_local_lows: bool,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut process_volumes = Vec::new();
    let mut large_volumes = Vec::new();
    let mut time_window_volumes = Vec::new();
    let mut buy_ratios = Vec::new();
    let mut price_counts = Vec::new();
    let mut max_declines = Vec::new();
    let mut process_durations = Vec::new();
    let mut process_start_times = Vec::new();
    let mut peak_prices = Vec::new();
    
    for process in processes {
        // 指标1：过程期间的成交量
        let total_volume = calculate_total_volume_v2(
            trade_volumes, process.start_index, process.end_index
        );
        process_volumes.push(total_volume);
        
        // 指标2：局部高点价格在盘口上时间最近的挂单量
        let first_large_volume = find_first_large_volume_v2(
            orderbook_times, orderbook_prices, orderbook_volumes,
            process.peak_price, process.start_time, process.end_time
        );
        large_volumes.push(first_large_volume);
        
        // 指标3：过程开始后指定时间窗口内的成交量
        let window_volume = calculate_time_window_volume_v2(
            trade_times, trade_volumes, process.start_index, process.start_time, time_window_minutes
        );
        time_window_volumes.push(window_volume);
        
        // 指标4：过程期间的主动买入成交量占比
        let buy_ratio = calculate_buy_ratio_v2(
            trade_flags, trade_volumes, process.start_index, process.end_index
        );
        buy_ratios.push(buy_ratio);
        
        // 指标5：过程期间的价格种类数
        let price_count = calculate_unique_prices_v2(
            trade_prices, process.start_index, process.end_index
        );
        price_counts.push(price_count as f64);
        
        // 指标6：过程期间价格相对局部极值的最大变化比例
        let max_change = if find_local_lows {
            calculate_max_incline_v2(trade_prices, process.start_index, process.end_index, process.peak_price)
        } else {
            calculate_max_decline_v2(trade_prices, process.start_index, process.end_index, process.peak_price)
        };
        max_declines.push(max_change);
        
        // 指标7：过程持续时间（秒）
        let duration_seconds = (process.end_time - process.start_time) / 1_000_000_000.0;
        process_durations.push(duration_seconds);
        
        // 指标8：过程开始时间（纳秒时间戳）
        process_start_times.push(process.start_time);
        
        // 指标9：局部高点的价格
        peak_prices.push(process.peak_price);
    }
    
    (process_volumes, large_volumes, time_window_volumes, buy_ratios, price_counts, max_declines, process_durations, process_start_times, peak_prices)
}

/// 计算指定范围内的总成交量
fn calculate_total_volume_v2(
    trade_volumes: &ndarray::ArrayView1<f64>,
    start_idx: usize,
    end_idx: usize,
) -> f64 {
    trade_volumes.slice(ndarray::s![start_idx..=end_idx]).sum()
}

/// 找到局部高点价格在盘口上时间最近的挂单量
fn find_first_large_volume_v2(
    orderbook_times: &ndarray::ArrayView1<f64>,
    orderbook_prices: &ndarray::ArrayView1<f64>,
    orderbook_volumes: &ndarray::ArrayView1<f64>,
    peak_price: f64,
    peak_time: f64,
    _end_time: f64, // 不再使用end_time限制
) -> f64 {
    let mut best_volume = 0.0;
    let mut min_time_diff = f64::INFINITY;
    
    for i in 0..orderbook_times.len() {
        let time = orderbook_times[i];
        let price = orderbook_prices[i];
        let volume = orderbook_volumes[i];
        
        // 检查价格是否相近（0.1%容忍度）
        let price_diff = (price - peak_price).abs();
        if price_diff < peak_price * 0.001 {
            // 计算与局部高点时间的距离
            let time_diff = (time - peak_time).abs();
            
            // 如果这个快照在时间上更接近局部高点，则更新
            if time_diff < min_time_diff {
                min_time_diff = time_diff;
                best_volume = volume;
            }
        }
    }
    
    best_volume
}

/// 计算过程开始后指定时间窗口内的成交量（纳秒版本）
fn calculate_time_window_volume_v2(
    trade_times: &ndarray::ArrayView1<f64>,
    trade_volumes: &ndarray::ArrayView1<f64>,
    start_idx: usize,
    start_time: f64,
    time_window_minutes: f64,
) -> f64 {
    let mut volume = 0.0;
    let time_window = time_window_minutes * 60.0 * 1_000_000_000.0; // 转换为纳秒
    
    for i in start_idx..trade_times.len() {
        let time_diff = trade_times[i] - start_time;
        if time_diff <= time_window {
            volume += trade_volumes[i];
        } else {
            break;
        }
    }
    
    volume
}

/// 计算主动买入成交量占比
fn calculate_buy_ratio_v2(
    trade_flags: &ndarray::ArrayView1<f64>,
    trade_volumes: &ndarray::ArrayView1<f64>,
    start_idx: usize,
    end_idx: usize,
) -> f64 {
    let mut total_volume = 0.0;
    let mut buy_volume = 0.0;
    
    for i in start_idx..=end_idx {
        let volume = trade_volumes[i];
        let flag = trade_flags[i] as i32;
        
        total_volume += volume;
        
        if flag == 66 { // 主动买入
            buy_volume += volume;
        }
        // flag == 83 表示主动卖出，不计入buy_volume
    }
    
    if total_volume > 0.0 {
        buy_volume / total_volume
    } else {
        0.0
    }
}

/// 计算过程期间的唯一价格数量
fn calculate_unique_prices_v2(
    trade_prices: &ndarray::ArrayView1<f64>,
    start_idx: usize,
    end_idx: usize,
) -> usize {
    let mut unique_prices = std::collections::HashSet::new();
    
    for i in start_idx..=end_idx {
        // 使用价格的整数表示来避免浮点数精度问题
        let price_key = (trade_prices[i] * 1000.0).round() as i64;
        unique_prices.insert(price_key);
    }
    
    unique_prices.len()
}

/// 计算过程期间价格相对局部高点的最大下降比例
fn calculate_max_decline_v2(
    trade_prices: &ndarray::ArrayView1<f64>,
    start_idx: usize,
    end_idx: usize,
    peak_price: f64,
) -> f64 {
    let mut max_decline = 0.0;
    
    for i in start_idx..=end_idx {
        let decline = (peak_price - trade_prices[i]) / peak_price;
        if decline > max_decline {
            max_decline = decline;
        }
    }
    
    max_decline
}

/// 计算过程期间价格相对局部低点的最大上升比例
fn calculate_max_incline_v2(
    trade_prices: &ndarray::ArrayView1<f64>,
    start_idx: usize,
    end_idx: usize,
    low_price: f64,
) -> f64 {
    let mut max_incline = 0.0;
    
    for i in start_idx..=end_idx {
        let incline = (trade_prices[i] - low_price) / low_price;
        if incline > max_incline {
            max_incline = incline;
        }
    }
    
    max_incline
}

/// 去除重复的局部高点
/// 
/// 对于价格相同且时间接近的多个局部高点，只保留时间最早的那个。
/// 这可以避免在同一价格水平的连续成交中重复识别相同的"以退为进"过程。
fn deduplicate_local_peaks_v2(
    trade_times: &ndarray::ArrayView1<f64>,
    trade_prices: &ndarray::ArrayView1<f64>,
    local_peaks: &[usize],
    dedup_time_seconds: f64,
) -> Vec<usize> {
    let mut deduplicated_peaks = Vec::new();
    
    if local_peaks.is_empty() {
        return deduplicated_peaks;
    }
    
    // 按价格分组局部高点
    let mut price_groups: std::collections::HashMap<i64, Vec<usize>> = std::collections::HashMap::new();
    
    for &peak_idx in local_peaks {
        // 使用价格的整数表示来避免浮点数精度问题（保留3位小数）
        let price_key = (trade_prices[peak_idx] * 1000.0).round() as i64;
        price_groups.entry(price_key).or_insert_with(Vec::new).push(peak_idx);
    }
    
    // 对每个价格组，只保留时间最早的局部高点
    for (_price_key, mut indices) in price_groups {
        if indices.len() == 1 {
            // 如果该价格只有一个局部高点，直接保留
            deduplicated_peaks.push(indices[0]);
        } else {
            // 如果该价格有多个局部高点，进行时间聚类
            // 按时间排序
            indices.sort_by(|&a, &b| trade_times[a].partial_cmp(&trade_times[b]).unwrap());
            
            let mut clusters = Vec::new();
            let mut current_cluster = vec![indices[0]];
            
            // 设置时间聚类阈值（纳秒）
            let time_cluster_threshold = dedup_time_seconds * 1_000_000_000.0;
            
            for i in 1..indices.len() {
                let time_diff = trade_times[indices[i]] - trade_times[indices[i-1]];
                
                if time_diff <= time_cluster_threshold {
                    // 时间接近，加入当前聚类
                    current_cluster.push(indices[i]);
                } else {
                    // 时间差距较大，开始新的聚类
                    clusters.push(current_cluster);
                    current_cluster = vec![indices[i]];
                }
            }
            clusters.push(current_cluster);
            
            // 每个聚类只保留时间最早的那个
            for cluster in clusters {
                deduplicated_peaks.push(cluster[0]); // cluster已经按时间排序，第一个是最早的
            }
        }
    }
    
    // 按索引排序结果
    deduplicated_peaks.sort();
    deduplicated_peaks
}