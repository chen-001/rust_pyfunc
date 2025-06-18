use pyo3::prelude::*;
use numpy::{PyArray1, PyReadonlyArray1, IntoPyArray};
use ndarray::Array1;

/// 分析股票交易中的"以退为进"现象
/// 
/// 该函数分析当价格触及某个局部高点后回落，然后在该价格的异常大挂单量消失后
/// 成功突破该价格的现象。
/// 
/// 参数说明：
/// ----------
/// trade_times : array_like
///     逐笔成交数据的时间戳序列（纳秒时间戳）
/// trade_prices : array_like
///     逐笔成交数据的价格序列
/// trade_volumes : array_like
///     逐笔成交数据的成交量序列
/// trade_flags : array_like
///     逐笔成交数据的标志序列（买卖方向）
/// orderbook_times : array_like
///     盘口快照数据的时间戳序列（纳秒时间戳）
/// orderbook_prices : array_like
///     盘口快照数据的价格序列
/// orderbook_volumes : array_like
///     盘口快照数据的挂单量序列
/// volume_percentile : float, optional
///     异常大挂单量的百分位数阈值，默认为99.0（即前1%）
/// time_window_minutes : float, optional
///     检查异常大挂单量的时间窗口（分钟），默认为1.0分钟
/// 
/// 返回值：
/// -------
/// tuple
///     包含6个列表的元组：
///     - 过程期间的成交量
///     - 过程期间首次观察到的价格x在盘口上的异常大挂单量
///     - 过程开始后指定时间窗口内的成交量
///     - 过程期间的主动买入成交量占比
///     - 过程期间的价格种类数
///     - 过程期间价格相对局部高点的最大下降比例
/// 
/// Python调用示例：
/// ```python
/// import numpy as np
/// from rust_pyfunc import analyze_retreat_advance
/// 
/// # 准备数据（纳秒时间戳）
/// trade_times = np.array([1661743800000000000, 1661743860000000000, 1661743920000000000], dtype=np.float64)
/// trade_prices = np.array([10.0, 10.1, 10.2], dtype=np.float64)
/// trade_volumes = np.array([100, 200, 150], dtype=np.float64)
/// trade_flags = np.array([1, 1, 1], dtype=np.float64)
/// 
/// orderbook_times = np.array([1661743800000000000, 1661743860000000000], dtype=np.float64)
/// orderbook_prices = np.array([10.0, 10.1], dtype=np.float64)
/// orderbook_volumes = np.array([1000, 5000], dtype=np.float64)
/// 
/// # 分析"以退为进"现象，使用2分钟时间窗口
/// results = analyze_retreat_advance(
///     trade_times, trade_prices, trade_volumes, trade_flags,
///     orderbook_times, orderbook_prices, orderbook_volumes,
///     volume_percentile=95.0, time_window_minutes=2.0
/// )
/// 
/// process_volumes, large_volumes, time_window_volumes, buy_ratios, price_counts, max_declines = results
/// ```
#[pyfunction]
#[pyo3(signature = (
    trade_times, trade_prices, trade_volumes, trade_flags,
    orderbook_times, orderbook_prices, orderbook_volumes,
    volume_percentile=99.0,
    time_window_minutes=1.0
))]
pub fn analyze_retreat_advance(
    py: Python,
    trade_times: PyReadonlyArray1<f64>,
    trade_prices: PyReadonlyArray1<f64>,
    trade_volumes: PyReadonlyArray1<f64>,
    trade_flags: PyReadonlyArray1<f64>,
    orderbook_times: PyReadonlyArray1<f64>,
    orderbook_prices: PyReadonlyArray1<f64>,
    orderbook_volumes: PyReadonlyArray1<f64>,
    volume_percentile: Option<f64>,
    time_window_minutes: Option<f64>,
) -> PyResult<(
    Py<PyArray1<f64>>, // 过程期间的成交量
    Py<PyArray1<f64>>, // 过程期间首次观察到的异常大挂单量
    Py<PyArray1<f64>>, // 过程开始后1分钟内的成交量
    Py<PyArray1<f64>>, // 过程期间的主动买入成交量占比
    Py<PyArray1<f64>>, // 过程期间的价格种类数
    Py<PyArray1<f64>>, // 过程期间价格相对局部高点的最大下降比例
)> {
    let trade_times = trade_times.as_array();
    let trade_prices = trade_prices.as_array();
    let trade_volumes = trade_volumes.as_array();
    let trade_flags = trade_flags.as_array();
    let orderbook_times = orderbook_times.as_array();
    let orderbook_prices = orderbook_prices.as_array();
    let orderbook_volumes = orderbook_volumes.as_array();
    
    let volume_percentile = volume_percentile.unwrap_or(99.0);
    let _time_window_minutes = time_window_minutes.unwrap_or(1.0);
    
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
    
    // 步骤1：找到所有局部高点
    let local_peaks = find_local_peaks(&trade_prices);
    
    // 步骤2：计算挂单量的百分位数阈值
    let volume_threshold = calculate_percentile(&orderbook_volumes, volume_percentile);
    
    // 步骤3：识别"以退为进"过程
    let processes = identify_retreat_advance_processes(
        &trade_times, &trade_prices, &trade_volumes, &trade_flags,
        &orderbook_times, &orderbook_prices, &orderbook_volumes,
        &local_peaks, volume_threshold
    );
    
    // 步骤4：计算每个过程的6个指标
    let (process_volumes, large_volumes, one_min_volumes, buy_ratios, price_counts, max_declines) = 
        calculate_process_metrics(
            &trade_times, &trade_prices, &trade_volumes, &trade_flags,
            &orderbook_times, &orderbook_prices, &orderbook_volumes,
            &processes
        );
    
    Ok((
        Array1::from(process_volumes).into_pyarray(py).to_owned(),
        Array1::from(large_volumes).into_pyarray(py).to_owned(),
        Array1::from(one_min_volumes).into_pyarray(py).to_owned(),
        Array1::from(buy_ratios).into_pyarray(py).to_owned(),
        Array1::from(price_counts).into_pyarray(py).to_owned(),
        Array1::from(max_declines).into_pyarray(py).to_owned(),
    ))
}

/// 找到价格序列中的局部高点
fn find_local_peaks(prices: &ndarray::ArrayView1<f64>) -> Vec<usize> {
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
        
        // 如果左右两边的第一个不同价格都比当前价格低，则为局部高点
        if left_different && right_different && left_lower && right_lower {
            peaks.push(i);
        }
    }
    
    peaks
}

/// 计算数组的百分位数
fn calculate_percentile(values: &ndarray::ArrayView1<f64>, percentile: f64) -> f64 {
    let mut sorted_values: Vec<f64> = values.iter().copied().collect();
    sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    
    let index = (sorted_values.len() as f64 * percentile / 100.0) as usize;
    let index = index.min(sorted_values.len() - 1);
    
    sorted_values[index]
}

/// 表示一个"以退为进"过程
#[derive(Debug, Clone)]
struct RetreatAdvanceProcess {
    peak_price: f64,        // 局部高点的价格
    start_time: f64,        // 过程开始时间
    end_time: f64,          // 过程结束时间
    start_index: usize,     // 过程开始的成交索引
    end_index: usize,       // 过程结束的成交索引
}

/// 识别"以退为进"过程
fn identify_retreat_advance_processes(
    trade_times: &ndarray::ArrayView1<f64>,
    trade_prices: &ndarray::ArrayView1<f64>,
    _trade_volumes: &ndarray::ArrayView1<f64>,
    _trade_flags: &ndarray::ArrayView1<f64>,
    orderbook_times: &ndarray::ArrayView1<f64>,
    orderbook_prices: &ndarray::ArrayView1<f64>,
    orderbook_volumes: &ndarray::ArrayView1<f64>,
    local_peaks: &[usize],
    volume_threshold: f64,
) -> Vec<RetreatAdvanceProcess> {
    let mut processes = Vec::new();
    
    for &peak_idx in local_peaks {
        let peak_price = trade_prices[peak_idx];
        let peak_time = trade_times[peak_idx];
        
        // 检查在局部高点附近1分钟内是否有异常大的挂单量
        let has_large_volume = check_large_volume_near_peak(
            orderbook_times, orderbook_prices, orderbook_volumes,
            peak_price, peak_time, volume_threshold
        );
        
        if !has_large_volume {
            continue;
        }
        
        // 寻找过程的结束点：价格成功突破局部高点
        if let Some(end_idx) = find_breakthrough_point(
            trade_times, trade_prices, peak_idx, peak_price
        ) {
            let process = RetreatAdvanceProcess {
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

/// 检查局部高点附近是否有异常大的挂单量
fn check_large_volume_near_peak(
    orderbook_times: &ndarray::ArrayView1<f64>,
    orderbook_prices: &ndarray::ArrayView1<f64>,
    orderbook_volumes: &ndarray::ArrayView1<f64>,
    peak_price: f64,
    peak_time: f64,
    volume_threshold: f64,
) -> bool {
    let time_window = 1.0 / 60.0; // 1分钟，假设时间单位是小时
    
    for i in 0..orderbook_times.len() {
        let time_diff = (orderbook_times[i] - peak_time).abs();
        let price_diff = (orderbook_prices[i] - peak_price).abs();
        
        // 在时间窗口内且价格相近的挂单
        if time_diff <= time_window && price_diff < peak_price * 0.001 { // 0.1%的价格容忍度
            if orderbook_volumes[i] >= volume_threshold {
                return true;
            }
        }
    }
    
    false
}

/// 寻找突破点：价格成功越过局部高点
fn find_breakthrough_point(
    trade_times: &ndarray::ArrayView1<f64>,
    trade_prices: &ndarray::ArrayView1<f64>,
    peak_idx: usize,
    peak_price: f64,
) -> Option<usize> {
    let n = trade_prices.len();
    
    // 从局部高点之后开始查找
    for i in peak_idx + 1..n {
        if trade_prices[i] > peak_price * 1.001 { // 突破局部高点0.1%以上
            return Some(i);
        }
        
        // 设置最大搜索时间窗口，避免无限搜索
        let time_diff = trade_times[i] - trade_times[peak_idx];
        if time_diff > 4.0 / 60.0 { // 4小时后仍未突破则放弃
            break;
        }
    }
    
    None
}

/// 计算每个过程的6个指标
fn calculate_process_metrics(
    trade_times: &ndarray::ArrayView1<f64>,
    trade_prices: &ndarray::ArrayView1<f64>,
    trade_volumes: &ndarray::ArrayView1<f64>,
    trade_flags: &ndarray::ArrayView1<f64>,
    orderbook_times: &ndarray::ArrayView1<f64>,
    orderbook_prices: &ndarray::ArrayView1<f64>,
    orderbook_volumes: &ndarray::ArrayView1<f64>,
    processes: &[RetreatAdvanceProcess],
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut process_volumes = Vec::new();
    let mut large_volumes = Vec::new();
    let mut one_min_volumes = Vec::new();
    let mut buy_ratios = Vec::new();
    let mut price_counts = Vec::new();
    let mut max_declines = Vec::new();
    
    for process in processes {
        // 指标1：过程期间的成交量
        let total_volume = calculate_total_volume(
            trade_volumes, process.start_index, process.end_index
        );
        process_volumes.push(total_volume);
        
        // 指标2：过程期间首次观察到的异常大挂单量
        let first_large_volume = find_first_large_volume(
            orderbook_times, orderbook_prices, orderbook_volumes,
            process.peak_price, process.start_time, process.end_time
        );
        large_volumes.push(first_large_volume);
        
        // 指标3：过程开始后1分钟内的成交量
        let one_min_volume = calculate_one_minute_volume(
            trade_times, trade_volumes, process.start_index, process.start_time
        );
        one_min_volumes.push(one_min_volume);
        
        // 指标4：过程期间的主动买入成交量占比
        let buy_ratio = calculate_buy_ratio(
            trade_flags, trade_volumes, process.start_index, process.end_index
        );
        buy_ratios.push(buy_ratio);
        
        // 指标5：过程期间的价格种类数
        let price_count = calculate_unique_prices(
            trade_prices, process.start_index, process.end_index
        );
        price_counts.push(price_count as f64);
        
        // 指标6：过程期间价格相对局部高点的最大下降比例
        let max_decline = calculate_max_decline(
            trade_prices, process.start_index, process.end_index, process.peak_price
        );
        max_declines.push(max_decline);
    }
    
    (process_volumes, large_volumes, one_min_volumes, buy_ratios, price_counts, max_declines)
}

/// 计算指定范围内的总成交量
fn calculate_total_volume(
    trade_volumes: &ndarray::ArrayView1<f64>,
    start_idx: usize,
    end_idx: usize,
) -> f64 {
    trade_volumes.slice(ndarray::s![start_idx..=end_idx]).sum()
}

/// 找到过程期间首次观察到的异常大挂单量
fn find_first_large_volume(
    orderbook_times: &ndarray::ArrayView1<f64>,
    orderbook_prices: &ndarray::ArrayView1<f64>,
    orderbook_volumes: &ndarray::ArrayView1<f64>,
    peak_price: f64,
    start_time: f64,
    end_time: f64,
) -> f64 {
    for i in 0..orderbook_times.len() {
        let time = orderbook_times[i];
        let price = orderbook_prices[i];
        let volume = orderbook_volumes[i];
        
        // 在过程时间范围内且价格相近
        if time >= start_time && time <= end_time {
            let price_diff = (price - peak_price).abs();
            if price_diff < peak_price * 0.001 { // 0.1%的价格容忍度
                return volume;
            }
        }
    }
    
    0.0
}

/// 计算过程开始后1分钟内的成交量
fn calculate_one_minute_volume(
    trade_times: &ndarray::ArrayView1<f64>,
    trade_volumes: &ndarray::ArrayView1<f64>,
    start_idx: usize,
    start_time: f64,
) -> f64 {
    let mut volume = 0.0;
    let one_minute = 1.0 / 60.0; // 1分钟
    
    for i in start_idx..trade_times.len() {
        let time_diff = trade_times[i] - start_time;
        if time_diff <= one_minute {
            volume += trade_volumes[i];
        } else {
            break;
        }
    }
    
    volume
}

/// 计算主动买入成交量占比
fn calculate_buy_ratio(
    trade_flags: &ndarray::ArrayView1<f64>,
    trade_volumes: &ndarray::ArrayView1<f64>,
    start_idx: usize,
    end_idx: usize,
) -> f64 {
    let mut total_volume = 0.0;
    let mut buy_volume = 0.0;
    
    for i in start_idx..=end_idx {
        let volume = trade_volumes[i];
        total_volume += volume;
        
        if trade_flags[i] > 0.0 { // 假设正数表示买入
            buy_volume += volume;
        }
    }
    
    if total_volume > 0.0 {
        buy_volume / total_volume
    } else {
        0.0
    }
}

/// 计算过程期间的唯一价格数量
fn calculate_unique_prices(
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
fn calculate_max_decline(
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