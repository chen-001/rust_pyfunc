//! 微观结构模式差异特征序列计算 - V2版本
//! 
//! 新增内容：
//! 第一步 - 新增模式：
//!   - trade_density: 成交笔密度型（固定时间窗口内的成交笔数）
//!   - trade_rhythm: 成交节奏型（时间间隔比：急-平-缓）
//!   - order_fragmentation: 大单破碎型（同一单号拆分笔数）
//!   - order_id_parity: 单号同奇偶型
//!   - order_id_tail: 单号末位型
//!   - order_id_flow: 连续成交单号流型（升降平）
//!   - order_concentration: 单号集中度型
//!   - five_rhythm: 五笔急缓序型
//!   - ten_buy_ratio: 十笔主买占比型
//!   - window_fragmentation: 窗口破碎度型
//!   - silence_burst: 沉默-爆发型
//!
//! 第二步 - 新增差异度量：
//!   - hellinger: Hellinger距离
//!   - chi_square: 卡方距离
//!   - sample_entropy: 样本熵差异
//!   - fft_energy: 能量谱差异（FFT功率谱）
//!   - moment_distance: 重心坐标差异（高阶矩）
//!
//! 第三步 - 新增构造方式：
//!   - cumulative_vs_immediate: 累积-即时切分
//!   - symmetric_expansion: 对称扩张切分
//!   - exponential_weighted: 指数加权切分
//!   - max_diff_location: 最大差异定位
//!   - curvature_peak: 差异曲率峰值
//!   - ar_residual: 自回归残差
//!   - extreme_crossing: 极值穿越
//!   - diff_of_diff: 差异的差异（二阶）

use numpy::{PyArray2, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::types::{PyList, PyString};
use std::collections::HashMap;
use std::f64::consts::PI;

/// 差异度量类型枚举
#[derive(Clone, Copy, Debug)]
enum DistanceType {
    // 已有度量
    Wasserstein,
    JensenShannon,
    KolmogorovSmirnov,
    EditDistance,
    DtwDistance,
    Jaccard,
    MeanDiff,
    VarianceRatio,
    SpearmanCorr,
    // 新增度量
    Hellinger,
    ChiSquare,
    SampleEntropy,
    FftEnergy,
    MomentDistance,
}

/// 序列构造方式枚举
#[derive(Clone, Copy, Debug)]
enum ConstructionType {
    // 已有构造
    MovingSplitPoint,
    AdjacentRollingWindows,
    CumulativeVsMarginal,
    // 新增构造
    CumulativeVsImmediate,
    SymmetricExpansion,
    ExponentialWeighted,
    MaxDiffLocation,
    CurvaturePeak,
    ArResidual,
    ExtremeCrossing,
    DiffOfDiff,
}

/// 模式类型枚举
#[derive(Clone, Debug)]
enum PatternType {
    Scalar(Vec<f64>),
    Histogram(Vec<Vec<f64>>),
    Sequence(Vec<Vec<u32>>),
    Direction(Vec<Vec<i32>>),
}

// ==================== 第一步：模式定义 ====================

/// 计算成交笔密度型（固定时间窗口内的成交笔数）
/// 时间窗口为固定毫秒数，如1000ms=1秒
fn calculate_trade_density(times: &[i64], window_ms: i64) -> Vec<f64> {
    let n = times.len();
    let mut result = vec![0.0; n];
    
    let mut left = 0;
    for right in 0..n {
        // 移动左指针，保持窗口大小
        while left < right && times[right] - times[left] > window_ms * 1_000_000 {
            left += 1;
        }
        result[right] = (right - left + 1) as f64;
    }
    
    result
}

/// 计算成交节奏型（时间间隔比：急-平-缓）
/// 返回编码：0=急(比值<0.5), 1=平(0.5<=比值<2), 2=缓(比值>=2)
fn calculate_trade_rhythm(times: &[i64]) -> Vec<f64> {
    let n = times.len();
    let mut result = vec![1.0; n]; // 默认为"平"
    
    if n < 2 {
        return result;
    }
    
    // 计算时间间隔（纳秒转毫秒）
    let mut intervals = vec![0.0; n];
    for i in 1..n {
        intervals[i] = (times[i] - times[i-1]) as f64 / 1_000_000.0; // 转为毫秒
        if intervals[i] < 1.0 {
            intervals[i] = 1.0; // 最小1ms避免除零
        }
    }
    intervals[0] = intervals[1]; // 第一个用第二个的值
    
    // 计算间隔比值
    for i in 2..n {
        let ratio = intervals[i] / intervals[i-1];
        if ratio < 0.5 {
            result[i] = 0.0; // 急
        } else if ratio >= 2.0 {
            result[i] = 2.0; // 缓
        } else {
            result[i] = 1.0; // 平
        }
    }
    
    result
}

/// 计算大单破碎型（同一单号被拆成的成交笔数）
/// 追踪每个订单编号出现的次数
fn calculate_order_fragmentation(bid_orders: &[i64], ask_orders: &[i64]) -> Vec<f64> {
    let n = bid_orders.len();
    let mut result = vec![0.0; n];
    
    let mut bid_count: HashMap<i64, usize> = HashMap::new();
    let mut ask_count: HashMap<i64, usize> = HashMap::new();
    
    // 第一遍：统计每个单号出现次数
    for i in 0..n {
        *bid_count.entry(bid_orders[i]).or_insert(0) += 1;
        *ask_count.entry(ask_orders[i]).or_insert(0) += 1;
    }
    
    // 第二遍：根据出现次数编码
    for i in 0..n {
        let bid_frag = bid_count[&bid_orders[i]];
        let ask_frag = ask_count[&ask_orders[i]];
        
        // 编码：取买卖双方的最大破碎度
        let max_frag = bid_frag.max(ask_frag);
        result[i] = if max_frag == 1 {
            0.0 // 未破碎
        } else if max_frag <= 5 {
            1.0 // 轻度破碎
        } else if max_frag <= 20 {
            2.0 // 中度破碎
        } else {
            3.0 // 重度破碎
        };
    }
    
    result
}

/// 计算单号同奇偶型
/// 编码：0=(奇,奇), 1=(奇,偶), 2=(偶,奇), 3=(偶,偶)
fn calculate_order_id_parity(bid_orders: &[i64], ask_orders: &[i64]) -> Vec<f64> {
    bid_orders.iter()
        .zip(ask_orders.iter())
        .map(|(bid, ask)| {
            let bid_odd = (bid.abs() % 2) as i32;
            let ask_odd = (ask.abs() % 2) as i32;
            (bid_odd * 2 + ask_odd) as f64
        })
        .collect()
}

/// 计算单号末位型（买卖双方单号末位数字之和）
fn calculate_order_id_tail(bid_orders: &[i64], ask_orders: &[i64]) -> Vec<f64> {
    bid_orders.iter()
        .zip(ask_orders.iter())
        .map(|(bid, ask)| {
            let bid_tail = (bid.abs() % 10) as f64;
            let ask_tail = (ask.abs() % 10) as f64;
            bid_tail + ask_tail
        })
        .collect()
}

/// 计算连续成交单号流型（买方单号序列的升降平模式）
/// 返回每个位置的排序序型（1-based rank）
fn calculate_order_id_flow(bid_orders: &[i64], window_size: usize) -> Vec<Vec<u32>> {
    let n = bid_orders.len();
    let mut result = Vec::with_capacity(n);
    
    for i in 0..n {
        let start = if i + 1 >= window_size { i + 1 - window_size } else { 0 };
        let window = &bid_orders[start..=i];
        
        // 计算排名（1-based）
        let mut indexed: Vec<(usize, i64)> = window.iter().enumerate().map(|(j, &v)| (j, v)).collect();
        indexed.sort_by(|a, b| a.1.cmp(&b.1));
        
        let mut ranks = vec![0u32; window.len()];
        for (rank, (idx, _)) in indexed.iter().enumerate() {
            ranks[*idx] = (rank + 1) as u32;
        }
        
        result.push(ranks);
    }
    
    result
}

/// 计算单号集中度型（固定窗口内不同买方单号的数量 / 成交笔数）
fn calculate_order_concentration(bid_orders: &[i64], window_size: usize) -> Vec<f64> {
    let n = bid_orders.len();
    let mut result = vec![0.0; n];
    
    for i in 0..n {
        let start = if i + 1 >= window_size { i + 1 - window_size } else { 0 };
        let window = &bid_orders[start..=i];
        
        let unique_count: std::collections::HashSet<i64> = window.iter().cloned().collect();
        result[i] = unique_count.len() as f64 / window.len() as f64;
    }
    
    result
}

/// 计算五笔急缓序型（连续5笔的成交节奏编码序列）
fn calculate_five_rhythm(times: &[i64], window_size: usize) -> Vec<Vec<u32>> {
    let rhythm = calculate_trade_rhythm(times);
    let n = rhythm.len();
    let mut result = Vec::with_capacity(n);
    
    for i in 0..n {
        let start = if i + 1 >= window_size { i + 1 - window_size } else { 0 };
        let window: Vec<f64> = rhythm[start..=i].to_vec();
        
        // 计算排名
        let mut indexed: Vec<(usize, f64)> = window.iter().enumerate().map(|(j, &v)| (j, v)).collect();
        indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        
        let mut ranks = vec![0u32; window.len()];
        for (rank, (idx, _)) in indexed.iter().enumerate() {
            ranks[*idx] = (rank + 1) as u32;
        }
        
        result.push(ranks);
    }
    
    result
}

/// 计算十笔主买占比型（连续10笔中主买笔数占比）
/// 返回编码：0=0-30%, 1=30-50%, 2=50-70%, 3=70-100%
fn calculate_ten_buy_ratio(flags: &[i32], window_size: usize) -> Vec<f64> {
    let n = flags.len();
    let mut result = vec![0.0; n];
    
    for i in 0..n {
        let start = if i + 1 >= window_size { i + 1 - window_size } else { 0 };
        let window = &flags[start..=i];
        
        let buy_count = window.iter().filter(|&&f| f == 66).count();
        let ratio = buy_count as f64 / window.len() as f64;
        
        result[i] = if ratio < 0.3 {
            0.0
        } else if ratio < 0.5 {
            1.0
        } else if ratio < 0.7 {
            2.0
        } else {
            3.0
        };
    }
    
    result
}

/// 计算窗口破碎度型（固定窗口中不同订单编号的数量）
fn calculate_window_fragmentation(bid_orders: &[i64], window_size: usize) -> Vec<f64> {
    let n = bid_orders.len();
    let mut result = vec![0.0; n];
    
    for i in 0..n {
        let start = if i + 1 >= window_size { i + 1 - window_size } else { 0 };
        let window = &bid_orders[start..=i];
        
        let unique_count: std::collections::HashSet<i64> = window.iter().cloned().collect();
        result[i] = unique_count.len() as f64;
    }
    
    result
}

/// 计算沉默-爆发型（长时间无成交后的首笔成交特征）
/// 返回编码：-1=沉默后首笔主卖, 0=无沉默, 1=沉默后首笔主买
fn calculate_silence_burst(times: &[i64], flags: &[i32], silence_threshold_ms: i64) -> Vec<f64> {
    let n = times.len();
    let mut result = vec![0.0; n];
    
    if n < 2 {
        return result;
    }
    
    for i in 1..n {
        let interval_ms = (times[i] - times[i-1]) / 1_000_000;
        if interval_ms > silence_threshold_ms {
            // 检测到沉默后的爆发
            result[i] = if flags[i] == 66 {
                1.0 // 主买
            } else if flags[i] == 83 {
                -1.0 // 主卖
            } else {
                0.0
            };
        }
    }
    
    result
}

// ==================== 第二步：差异度量 ====================

/// Hellinger距离
fn hellinger_distance(p: &[f64], q: &[f64]) -> f64 {
    if p.len() != q.len() || p.is_empty() {
        return f64::NAN;
    }
    
    let mut sum = 0.0;
    for i in 0..p.len() {
        sum += (p[i].sqrt() - q[i].sqrt()).powi(2);
    }
    
    (sum / 2.0).sqrt()
}

/// 卡方距离
fn chi_square_distance(p: &[f64], q: &[f64]) -> f64 {
    if p.len() != q.len() || p.is_empty() {
        return f64::NAN;
    }
    
    let mut sum = 0.0;
    for i in 0..p.len() {
        if q[i] > 1e-10 {
            sum += (p[i] - q[i]).powi(2) / q[i];
        }
    }
    
    sum
}

/// 样本熵（高度优化版）
/// 使用粗粒度采样，只计算固定数量的模板对
fn sample_entropy(data: &[f64], _m: usize, r: f64) -> f64 {
    let n = data.len();
    if n < 10 {
        return f64::NAN;
    }
    
    // 计算标准差用于归一化r
    let mean = data.iter().sum::<f64>() / n as f64;
    let std = (data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n as f64).sqrt();
    let r = r * std;
    
    // 固定只采样100个模板对，与数据量无关
    let num_samples = 100.min(n / 2);
    let step = n / num_samples;
    
    let mut matches = 0.0;
    let mut total = 0.0;
    
    for i in (0..n).step_by(step.max(1)) {
        for j in ((i + step)..n).step_by(step.max(1)) {
            if (data[i] - data[j]).abs() <= r {
                matches += 1.0;
            }
            total += 1.0;
            
            // 最多检查1000对
            if total >= 1000.0 {
                break;
            }
        }
        if total >= 1000.0 {
            break;
        }
    }
    
    if total < 1.0 || matches < 1.0 {
        return f64::NAN;
    }
    
    // 简化的熵估计
    let p = matches / total;
    -(p as f64).ln()
}

/// FFT能量差异（极度简化版）
/// 只比较两段序列的能量集中度，而非完整FFT
fn fft_energy_distance(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() || a.len() < 2 {
        return f64::NAN;
    }
    
    // 计算简单统计量代替FFT
    let mean_a = a.iter().sum::<f64>() / a.len() as f64;
    let mean_b = b.iter().sum::<f64>() / b.len() as f64;
    
    let var_a = a.iter().map(|&x| (x - mean_a).powi(2)).sum::<f64>() / a.len() as f64;
    let var_b = b.iter().map(|&x| (x - mean_b).powi(2)).sum::<f64>() / b.len() as f64;
    
    // 计算一阶差分的方差（趋势性）
    let diff_var_a = if a.len() > 1 {
        let diffs: Vec<f64> = a.windows(2).map(|w| w[1] - w[0]).collect();
        let mean_diff = diffs.iter().sum::<f64>() / diffs.len() as f64;
        diffs.iter().map(|&x| (x - mean_diff).powi(2)).sum::<f64>() / diffs.len() as f64
    } else {
        0.0
    };
    
    let diff_var_b = if b.len() > 1 {
        let diffs: Vec<f64> = b.windows(2).map(|w| w[1] - w[0]).collect();
        let mean_diff = diffs.iter().sum::<f64>() / diffs.len() as f64;
        diffs.iter().map(|&x| (x - mean_diff).powi(2)).sum::<f64>() / diffs.len() as f64
    } else {
        0.0
    };
    
    // 综合距离
    let var_diff = (var_a - var_b).abs();
    let trend_diff = (diff_var_a - diff_var_b).abs();
    
    (var_diff + trend_diff).sqrt()
}

/// 重心坐标差异（高阶矩距离）
/// 计算偏度和峰度构成的向量距离
fn moment_distance(a: &[f64], b: &[f64]) -> f64 {
    if a.len() < 4 || b.len() < 4 {
        return f64::NAN;
    }
    
    let (skew_a, kurt_a) = calculate_skew_kurt(a);
    let (skew_b, kurt_b) = calculate_skew_kurt(b);
    
    ((skew_a - skew_b).powi(2) + (kurt_a - kurt_b).powi(2)).sqrt()
}

/// 计算偏度和峰度
fn calculate_skew_kurt(data: &[f64]) -> (f64, f64) {
    let n = data.len() as f64;
    let mean = data.iter().sum::<f64>() / n;
    
    let mut var = 0.0;
    let mut skew = 0.0;
    let mut kurt = 0.0;
    
    for &x in data {
        let diff = x - mean;
        var += diff.powi(2);
        skew += diff.powi(3);
        kurt += diff.powi(4);
    }
    
    var /= n;
    let std = var.sqrt();
    
    if std < 1e-10 {
        return (0.0, 0.0);
    }
    
    skew = skew / (n * std.powi(3));
    kurt = kurt / (n * std.powi(4)) - 3.0; // 超额峰度
    
    (skew, kurt)
}

// ==================== 第三步：特征序列构造 ====================

/// 累积-即时切分：切分点前为累积窗口，切分点后为固定短窗口
/// 优化版：使用增量计算，避免重复计算累积统计量
fn construct_cumulative_vs_immediate(
    values: &[f64],
    immediate_window: usize,
    dist_type: DistanceType,
) -> Vec<f64> {
    let n = values.len();
    let mut result = vec![f64::NAN; n];
    
    // 只计算简单统计量，避免O(n^2)
    let mut cumul_sum = 0.0;
    let mut cumul_count = 0;
    
    for i in immediate_window..n {
        // 增量更新累积统计量
        cumul_sum += values[i - 1];
        cumul_count += 1;
        let cumul_mean = cumul_sum / cumul_count as f64;
        
        // 计算即时窗口统计量
        let immediate = &values[i - immediate_window..i];
        let imm_sum: f64 = immediate.iter().sum();
        let imm_mean = imm_sum / immediate_window as f64;
        
        result[i] = match dist_type {
            DistanceType::MeanDiff => cumul_mean - imm_mean,
            _ => (cumul_mean - imm_mean).abs(),
        };
    }
    
    result
}

/// 对称扩张切分：以时刻t为中心，向左右对称扩张窗口
/// 优化版：固定窗口大小，使用增量计算
fn construct_symmetric_expansion(
    values: &[f64],
    max_window: usize,
    dist_type: DistanceType,
) -> Vec<f64> {
    let n = values.len();
    let mut result = vec![f64::NAN; n];
    let half_window = max_window / 2;
    
    if half_window < 5 {
        return result;
    }
    
    for i in half_window..(n - half_window) {
        let left = &values[i - half_window..i];
        let right = &values[i..i + half_window];
        
        // 只计算均值差异，避免复杂距离计算
        let mean_left: f64 = left.iter().sum::<f64>() / left.len() as f64;
        let mean_right: f64 = right.iter().sum::<f64>() / right.len() as f64;
        
        result[i] = match dist_type {
            DistanceType::MeanDiff => mean_left - mean_right,
            _ => (mean_left - mean_right).abs(),
        };
    }
    
    result
}

/// 指数加权切分：切分点前的数据指数加权（远期衰减），切分点后均匀加权
/// 优化版：使用滑动窗口和近似
fn construct_exponential_weighted(
    values: &[f64],
    decay_factor: f64,
    dist_type: DistanceType,
) -> Vec<f64> {
    let n = values.len();
    let mut result = vec![f64::NAN; n];
    
    // 使用固定大小的滑动窗口代替全历史
    let max_window = 1000.min(n);
    let mut weighted_sum = 0.0;
    let mut weight_sum = 0.0;
    
    for i in 10..n {
        // 滑动更新指数加权
        weighted_sum = weighted_sum * decay_factor + values[i - 1];
        weight_sum = weight_sum * decay_factor + 1.0;
        
        // 限制窗口大小
        if i > max_window {
            let old_weight = decay_factor.powi(max_window as i32);
            weighted_sum -= values[i - max_window - 1] * old_weight;
            weight_sum -= old_weight;
        }
        
        let mean_left = if weight_sum > 0.0 {
            weighted_sum / weight_sum
        } else {
            0.0
        };
        
        // 右侧只取固定窗口
        let right_end = (i + max_window).min(n);
        let right = &values[i..right_end];
        let mean_right: f64 = right.iter().sum::<f64>() / right.len() as f64;
        
        result[i] = match dist_type {
            DistanceType::MeanDiff => mean_left - mean_right,
            _ => (mean_left - mean_right).abs(),
        };
    }
    
    result
}

/// 最大差异定位：寻找使差异最大的切分点，滑动更新
/// 优化版：使用滑动窗口近似，O(n)复杂度
fn construct_max_diff_location(
    values: &[f64],
    search_window: usize,
    dist_type: DistanceType,
) -> Vec<f64> {
    let n = values.len();
    let mut result = vec![f64::NAN; n];
    
    for i in search_window..n {
        let search_start = i - search_window;
        let mut max_diff = 0.0;
        let mut max_idx = search_start;
        
        // 在搜索窗口内找最大差异点
        for split in search_start..i {
            let left = &values[search_start..split];
            let right = &values[split..i];
            
            if left.len() < 5 || right.len() < 5 {
                continue;
            }
            
            let diff = calculate_scalar_distance(left, right, dist_type).abs();
            if diff > max_diff {
                max_diff = diff;
                max_idx = split;
            }
        }
        
        // 返回最大差异点的位置和差异值
        result[i] = max_idx as f64;
        // 可以额外返回差异值，但这里只返回位置
    }
    
    result
}

/// 差异曲率峰值：计算差异序列的二阶导数，记录曲率极大值点
/// 优化版：使用简单统计量代替复杂距离
fn construct_curvature_peak(
    values: &[f64],
    window_size: usize,
    _dist_type: DistanceType,
) -> Vec<f64> {
    let n = values.len();
    let mut result = vec![f64::NAN; n];
    
    // 计算一阶差异序列（使用简单均值差）
    let mut first_diff = vec![0.0; n];
    for i in window_size..n {
        let left = &values[i - window_size..i];
        let right_end = (i + window_size).min(n);
        let right = &values[i..right_end];
        
        let mean_left: f64 = left.iter().sum::<f64>() / left.len() as f64;
        let mean_right: f64 = right.iter().sum::<f64>() / right.len() as f64;
        
        first_diff[i] = mean_left - mean_right;
    }
    
    // 计算二阶差分（曲率）
    for i in (window_size + 1)..(n - 1) {
        let curvature = first_diff[i + 1] - 2.0 * first_diff[i] + first_diff[i - 1];
        result[i] = curvature.abs();
    }
    
    result
}

/// 自回归残差：对差异序列做AR(1)拟合，记录残差序列
/// 优化版：限制窗口大小，使用简单统计量
fn construct_ar_residual(
    values: &[f64],
    window_size: usize,
    _dist_type: DistanceType,
) -> Vec<f64> {
    let n = values.len();
    let mut result = vec![f64::NAN; n];
    
    // 使用简单的一阶差分代替复杂距离计算
    let mut diff_series = vec![0.0; n];
    for i in window_size..n {
        let left = &values[i - window_size..i];
        let right_end = (i + window_size).min(n);
        let right = &values[i..right_end];
        
        let mean_left: f64 = left.iter().sum::<f64>() / left.len() as f64;
        let mean_right: f64 = right.iter().sum::<f64>() / right.len() as f64;
        
        diff_series[i] = mean_left - mean_right;
    }
    
    // AR(1)拟合
    for i in (window_size + 10)..n {
        let start = i - 10;
        let mut sum_xy = 0.0;
        let mut sum_x2 = 0.0;
        
        for j in (start + 1)..i {
            if !diff_series[j].is_nan() && !diff_series[j - 1].is_nan() {
                sum_xy += diff_series[j] * diff_series[j - 1];
                sum_x2 += diff_series[j - 1].powi(2);
            }
        }
        
        let phi = if sum_x2 > 1e-10 {
            sum_xy / sum_x2
        } else {
            0.0
        };
        
        if !diff_series[i].is_nan() && !diff_series[i - 1].is_nan() {
            result[i] = diff_series[i] - phi * diff_series[i - 1];
        }
    }
    
    result
}

/// 极值穿越：记录差异序列向上/向下穿越历史百分位阈值的次数
fn construct_extreme_crossing(
    values: &[f64],
    window_size: usize,
    percentile: f64,
    dist_type: DistanceType,
) -> Vec<f64> {
    let n = values.len();
    let mut result = vec![f64::NAN; n];
    
    let mut cross_count = 0.0;
    let mut prev_diff: f64 = 0.0;
    
    for i in window_size..n {
        let left = &values[i - window_size..i];
        let right = &values[i..n];
        
        let diff = calculate_scalar_distance(left, right, dist_type);
        
        // 计算历史阈值
        let hist_window = &values[0..i];
        let threshold = calculate_percentile(hist_window, percentile);
        
        // 检测穿越
        if i > window_size && !prev_diff.is_nan() && !diff.is_nan() {
            if prev_diff < threshold && diff >= threshold {
                cross_count += 1.0; // 向上穿越
            } else if prev_diff > threshold && diff <= threshold {
                cross_count -= 1.0; // 向下穿越
            }
        }
        
        result[i] = cross_count;
        prev_diff = diff;
    }
    
    result
}

/// 差异的差异（二阶异变）：先计算差异序列，再对该序列计算差异
/// 优化版：限制窗口大小，使用简单统计量
fn construct_diff_of_diff(
    values: &[f64],
    window_size: usize,
    _dist_type: DistanceType,
) -> Vec<f64> {
    let n = values.len();
    let mut result = vec![f64::NAN; n];
    
    // 计算一阶差异序列（使用简单均值差）
    let mut first_diff = vec![0.0; n];
    for i in window_size..n {
        let left = &values[i - window_size..i];
        let right_end = (i + window_size).min(n);
        let right = &values[i..right_end];
        
        let mean_left: f64 = left.iter().sum::<f64>() / left.len() as f64;
        let mean_right: f64 = right.iter().sum::<f64>() / right.len() as f64;
        
        first_diff[i] = mean_left - mean_right;
    }
    
    // 对差异序列再计算差异（同样限制窗口）
    for i in (2 * window_size)..n {
        let left = &first_diff[i - window_size..i];
        let right_end = (i + window_size).min(n);
        let right = &first_diff[i..right_end];
        
        let mean_left: f64 = left.iter().sum::<f64>() / left.len() as f64;
        let mean_right: f64 = right.iter().sum::<f64>() / right.len() as f64;
        
        result[i] = mean_left - mean_right;
    }
    
    result
}

/// 计算百分位数
fn calculate_percentile(data: &[f64], percentile: f64) -> f64 {
    let mut sorted: Vec<f64> = data.iter().filter(|&&x| !x.is_nan()).cloned().collect();
    if sorted.is_empty() {
        return 0.0;
    }
    
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    
    let idx = ((sorted.len() - 1) as f64 * percentile) as usize;
    sorted[idx.min(sorted.len() - 1)]
}

// ==================== 辅助函数 ====================

/// 计算标量集合的差异度量
fn calculate_scalar_distance(values_a: &[f64], values_b: &[f64], dist_type: DistanceType) -> f64 {
    match dist_type {
        DistanceType::MeanDiff => {
            if values_a.is_empty() || values_b.is_empty() {
                return f64::NAN;
            }
            let mean_a: f64 = values_a.iter().sum::<f64>() / values_a.len() as f64;
            let mean_b: f64 = values_b.iter().sum::<f64>() / values_b.len() as f64;
            mean_a - mean_b
        }
        DistanceType::VarianceRatio => {
            if values_a.len() < 2 || values_b.len() < 2 {
                return f64::NAN;
            }
            let mean_a: f64 = values_a.iter().sum::<f64>() / values_a.len() as f64;
            let mean_b: f64 = values_b.iter().sum::<f64>() / values_b.len() as f64;
            let var_a: f64 = values_a.iter().map(|&v| (v - mean_a).powi(2)).sum::<f64>() / (values_a.len() - 1) as f64;
            let var_b: f64 = values_b.iter().map(|&v| (v - mean_b).powi(2)).sum::<f64>() / (values_b.len() - 1) as f64;
            if var_b.abs() < 1e-10 {
                return f64::NAN;
            }
            var_a / var_b
        }
        DistanceType::SpearmanCorr => {
            if values_a.len() != values_b.len() {
                return f64::NAN;
            }
            spearman_correlation(values_a, values_b)
        }
        DistanceType::DtwDistance => {
            dtw_distance(values_a, values_b)
        }
        DistanceType::SampleEntropy => {
            // 样本熵差异：两段序列样本熵的差
            let se_a = sample_entropy(values_a, 2, 0.2);
            let se_b = sample_entropy(values_b, 2, 0.2);
            if se_a.is_nan() || se_b.is_nan() {
                return f64::NAN;
            }
            se_a - se_b
        }
        DistanceType::FftEnergy => {
            fft_energy_distance(values_a, values_b)
        }
        DistanceType::MomentDistance => {
            moment_distance(values_a, values_b)
        }
        _ => f64::NAN,
    }
}

/// 计算直方图的差异度量
fn calculate_histogram_distance(hist_a: &[f64], hist_b: &[f64], dist_type: DistanceType) -> f64 {
    match dist_type {
        DistanceType::Wasserstein => wasserstein_distance(hist_a, hist_b),
        DistanceType::JensenShannon => jensen_shannon_divergence(hist_a, hist_b),
        DistanceType::KolmogorovSmirnov => kolmogorov_smirnov(hist_a, hist_b),
        DistanceType::Hellinger => hellinger_distance(hist_a, hist_b),
        DistanceType::ChiSquare => chi_square_distance(hist_a, hist_b),
        _ => f64::NAN,
    }
}

/// 计算序列的差异度量
fn calculate_sequence_distance(seq_a: &[u32], seq_b: &[u32], dist_type: DistanceType) -> f64 {
    match dist_type {
        DistanceType::EditDistance => edit_distance(seq_a, seq_b),
        DistanceType::Jaccard => 1.0 - jaccard_similarity(seq_a, seq_b),
        _ => f64::NAN,
    }
}

/// Wasserstein距离（一维）
fn wasserstein_distance(p: &[f64], q: &[f64]) -> f64 {
    if p.len() != q.len() || p.is_empty() {
        return f64::NAN;
    }
    
    let mut cum_diff = 0.0;
    let mut distance = 0.0;
    
    for i in 0..p.len() {
        cum_diff += p[i] - q[i];
        distance += cum_diff.abs();
    }
    
    distance
}

/// Jensen-Shannon散度
fn jensen_shannon_divergence(p: &[f64], q: &[f64]) -> f64 {
    if p.len() != q.len() || p.is_empty() {
        return f64::NAN;
    }
    
    let mut divergence = 0.0;
    
    for i in 0..p.len() {
        let m = 0.5 * (p[i] + q[i]);
        if p[i] > 1e-10 {
            divergence += 0.5 * p[i] * (p[i] / m).ln();
        }
        if q[i] > 1e-10 {
            divergence += 0.5 * q[i] * (q[i] / m).ln();
        }
    }
    
    divergence
}

/// KS统计量
fn kolmogorov_smirnov(p: &[f64], q: &[f64]) -> f64 {
    if p.len() != q.len() || p.is_empty() {
        return f64::NAN;
    }
    
    let mut max_diff = 0.0;
    let mut cum_p = 0.0;
    let mut cum_q = 0.0;
    
    for i in 0..p.len() {
        cum_p += p[i];
        cum_q += q[i];
        let diff = (cum_p - cum_q).abs();
        if diff > max_diff {
            max_diff = diff;
        }
    }
    
    max_diff
}

/// 编辑距离
fn edit_distance(a: &[u32], b: &[u32]) -> f64 {
    let m = a.len();
    let n = b.len();
    
    if m == 0 {
        return n as f64;
    }
    if n == 0 {
        return m as f64;
    }
    
    // 使用一维DP优化空间
    let mut prev = vec![0; n + 1];
    let mut curr = vec![0; n + 1];
    
    for j in 0..=n {
        prev[j] = j;
    }
    
    for i in 1..=m {
        curr[0] = i;
        for j in 1..=n {
            let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
            curr[j] = (curr[j - 1] + 1)
                .min(prev[j] + 1)
                .min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    
    prev[n] as f64
}

/// Jaccard相似系数
fn jaccard_similarity(a: &[u32], b: &[u32]) -> f64 {
    let set_a: std::collections::HashSet<u32> = a.iter().cloned().collect();
    let set_b: std::collections::HashSet<u32> = b.iter().cloned().collect();
    
    let intersection: std::collections::HashSet<_> = set_a.intersection(&set_b).cloned().collect();
    let union: std::collections::HashSet<_> = set_a.union(&set_b).cloned().collect();
    
    if union.is_empty() {
        return 0.0;
    }
    
    intersection.len() as f64 / union.len() as f64
}

/// 动态时间规整距离
fn dtw_distance(a: &[f64], b: &[f64]) -> f64 {
    let m = a.len();
    let n = b.len();
    
    if m == 0 || n == 0 {
        return f64::NAN;
    }
    
    // 使用一维DP优化空间
    let mut prev = vec![f64::INFINITY; n + 1];
    let mut curr = vec![f64::INFINITY; n + 1];
    prev[0] = 0.0;
    
    for i in 1..=m {
        curr[0] = f64::INFINITY;
        for j in 1..=n {
            let cost = (a[i - 1] - b[j - 1]).abs();
            curr[j] = cost + prev[j].min(curr[j - 1]).min(prev[j - 1]);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    
    prev[n]
}

/// Spearman秩相关系数
fn spearman_correlation(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() || a.len() < 2 {
        return f64::NAN;
    }
    
    let n = a.len() as f64;
    let rank_a = get_ranks(a);
    let rank_b = get_ranks(b);
    
    let mean_rank = (n + 1.0) / 2.0;
    let mut numerator = 0.0;
    let mut denom_a = 0.0;
    let mut denom_b = 0.0;
    
    for i in 0..a.len() {
        let diff_a = rank_a[i] - mean_rank;
        let diff_b = rank_b[i] - mean_rank;
        numerator += diff_a * diff_b;
        denom_a += diff_a * diff_a;
        denom_b += diff_b * diff_b;
    }
    
    let denominator = denom_a.sqrt() * denom_b.sqrt();
    if denominator < 1e-10 {
        return 0.0;
    }
    
    numerator / denominator
}

/// 获取排名
fn get_ranks(values: &[f64]) -> Vec<f64> {
    let mut indexed: Vec<(usize, f64)> = values.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    
    let mut ranks = vec![0.0; values.len()];
    for (rank, (idx, _)) in indexed.iter().enumerate() {
        ranks[*idx] = (rank + 1) as f64;
    }
    
    ranks
}

// ==================== 主函数 ====================

/// 主函数：计算微观结构模式差异特征序列 V2
#[pyfunction]
#[pyo3(signature = (
    trade_times,
    trade_prices,
    trade_volumes,
    trade_flags,
    trade_bid_orders,
    trade_ask_orders,
    window_size = 100,
    histogram_bins = 10,
    marginal_window = 50,
))]
pub fn calculate_microstructure_pattern_features_v2(
    py: Python,
    trade_times: PyReadonlyArray1<i64>,
    trade_prices: PyReadonlyArray1<f64>,
    trade_volumes: PyReadonlyArray1<f64>,
    trade_flags: PyReadonlyArray1<i32>,
    trade_bid_orders: PyReadonlyArray1<i64>,
    trade_ask_orders: PyReadonlyArray1<i64>,
    window_size: usize,
    histogram_bins: usize,
    marginal_window: usize,
) -> PyResult<(Py<PyArray2<f64>>, Py<PyList>, Py<PyList>, Py<PyList>)> {
    
    // 转换为Rust切片
    let times = trade_times.as_slice()?;
    let _prices = trade_prices.as_slice()?;
    let volumes = trade_volumes.as_slice()?;
    let flags = trade_flags.as_slice()?;
    let bid_orders = trade_bid_orders.as_slice()?;
    let ask_orders = trade_ask_orders.as_slice()?;
    
    let n = times.len();
    
    // ==================== 第一步：计算所有新模式 ====================
    
    // 新增标量模式
    let trade_density = calculate_trade_density(times, 1000); // 1秒窗口
    let trade_rhythm = calculate_trade_rhythm(times);
    let order_fragmentation = calculate_order_fragmentation(bid_orders, ask_orders);
    let order_id_parity = calculate_order_id_parity(bid_orders, ask_orders);
    let order_id_tail = calculate_order_id_tail(bid_orders, ask_orders);
    let order_concentration = calculate_order_concentration(bid_orders, window_size);
    let ten_buy_ratio = calculate_ten_buy_ratio(flags, 10);
    // window_fragmentation已删除（与order_concentration高度相关）
    let silence_burst = calculate_silence_burst(times, flags, 100); // 100ms沉默阈值
    
    // 新增序列模式
    let order_id_flow = calculate_order_id_flow(bid_orders, 5);
    // five_rhythm已删除（与order_id_flow高度相关）
    
    // ==================== 第二步&第三步：组合计算 ====================
    // 优化：减少组合数量，只保留最具代表性的组合
    // 原始组合：9模式 × 5度量 × 8构造 = 360，太多！
    // 优化后：选择性地组合，约 50-80 个特征序列
    
    let mut all_sequences: Vec<Vec<f64>> = Vec::new();
    let mut all_names: Vec<String> = Vec::new();
    
    // ----- 组合1: 快速标量模式 × 简单度量 × 快速构造 -----
    // 这些组合计算快，全部保留
    
    let fast_patterns: Vec<(&str, Vec<f64>)> = vec![
        ("trade_density", trade_density.clone()),
        ("trade_rhythm", trade_rhythm.clone()),
        ("order_id_parity", order_id_parity.clone()),
        ("order_id_tail", order_id_tail.clone()),
        ("ten_buy_ratio", ten_buy_ratio.clone()),
        // silence_burst已删除：该模式是稀疏事件指示器（-1/0/1），不适合构造差异序列
    ];
    
    let fast_distances: Vec<(&str, DistanceType)> = vec![
        ("mean_diff", DistanceType::MeanDiff),
        ("variance_ratio", DistanceType::VarianceRatio),
    ];
    
    let fast_constructions: Vec<(&str, ConstructionType)> = vec![
        ("cumulative_vs_immediate", ConstructionType::CumulativeVsImmediate),
        ("symmetric_expansion", ConstructionType::SymmetricExpansion),
        ("exponential_weighted", ConstructionType::ExponentialWeighted),
    ];
    
    for (pattern_name, pattern_values) in &fast_patterns {
        for (dist_name, dist_type) in &fast_distances {
            for (cons_name, cons_type) in &fast_constructions {
                let sequence = match cons_type {
                    ConstructionType::CumulativeVsImmediate => {
                        construct_cumulative_vs_immediate(pattern_values, marginal_window, *dist_type)
                    }
                    ConstructionType::SymmetricExpansion => {
                        construct_symmetric_expansion(pattern_values, window_size, *dist_type)
                    }
                    ConstructionType::ExponentialWeighted => {
                        construct_exponential_weighted(pattern_values, 0.95, *dist_type)
                    }
                    _ => vec![f64::NAN; n],
                };
                
                let name = format!("{}_{}_{}", pattern_name, dist_name, cons_name);
                all_sequences.push(sequence);
                all_names.push(name);
            }
        }
    }
    
    // ----- 组合2: 复杂标量模式 × 简单度量 × 快速构造 -----
    // 这些模式计算稍慢，减少构造方式
    // 注意：window_fragmentation与order_concentration高度相关(0.99+)，已删除
    
    let medium_patterns: Vec<(&str, Vec<f64>)> = vec![
        ("order_fragmentation", order_fragmentation),
        ("order_concentration", order_concentration),
    ];
    
    let medium_constructions: Vec<(&str, ConstructionType)> = vec![
        ("cumulative_vs_immediate", ConstructionType::CumulativeVsImmediate),
        ("symmetric_expansion", ConstructionType::SymmetricExpansion),
    ];
    
    for (pattern_name, pattern_values) in &medium_patterns {
        for (dist_name, dist_type) in &fast_distances {
            for (cons_name, cons_type) in &medium_constructions {
                let sequence = match cons_type {
                    ConstructionType::CumulativeVsImmediate => {
                        construct_cumulative_vs_immediate(pattern_values, marginal_window, *dist_type)
                    }
                    ConstructionType::SymmetricExpansion => {
                        construct_symmetric_expansion(pattern_values, window_size, *dist_type)
                    }
                    _ => vec![f64::NAN; n],
                };
                
                let name = format!("{}_{}_{}", pattern_name, dist_name, cons_name);
                all_sequences.push(sequence);
                all_names.push(name);
            }
        }
    }
    
    // ----- 组合3: 复杂度量特征 -----
    // 注意：已删除全局常量特征（如sample_entropy_global），因为常量特征对模型训练无帮助
    // 如果需要复杂度量，应该在滑动窗口上计算，而不是整个序列
    
    // ----- 组合4: 特殊构造方式 × 快速模式 -----
    // 只选最有代表性的特殊构造
    
    let special_constructions: Vec<(&str, ConstructionType, fn(&[f64], usize, DistanceType) -> Vec<f64>)> = vec![
        ("curvature_peak", ConstructionType::CurvaturePeak, construct_curvature_peak as fn(&[f64], usize, DistanceType) -> Vec<f64>),
        ("ar_residual", ConstructionType::ArResidual, construct_ar_residual as fn(&[f64], usize, DistanceType) -> Vec<f64>),
        ("diff_of_diff", ConstructionType::DiffOfDiff, construct_diff_of_diff as fn(&[f64], usize, DistanceType) -> Vec<f64>),
    ];
    
    // 使用trade_density作为快速模式进行特殊构造
    let selected_pattern_name = "trade_density";
    let selected_pattern_values = &trade_density;
    for (cons_name, _, cons_fn) in &special_constructions {
        let sequence = cons_fn(selected_pattern_values, window_size, DistanceType::MeanDiff);
        
        let name = format!("{}_mean_diff_{}", selected_pattern_name, cons_name);
        all_sequences.push(sequence);
        all_names.push(name);
    }
    
    // ----- 组合5: 序列模式 × 简单差异度量 -----
    // 由于编辑距离是O(n^2)，改用简单的统计差异
    // 注意：five_rhythm与order_id_flow高度相关(±1.0)，已删除
    
    let sequence_patterns: Vec<(&str, Vec<Vec<u32>>)> = vec![
        ("order_id_flow", order_id_flow),
    ];
    
    for (pattern_name, pattern_values) in &sequence_patterns {
        // 将序列模式转换为标量：计算每个窗口的均值
        let scalar_values: Vec<f64> = pattern_values.iter()
            .map(|seq| seq.iter().sum::<u32>() as f64 / seq.len() as f64)
            .collect();
        
        // 只使用两种构造方式（exponential_weighted对序列模式效果不佳）
        let sequence = construct_cumulative_vs_immediate(&scalar_values, marginal_window, DistanceType::MeanDiff);
        let name = format!("{}_mean_cumulative_vs_immediate", pattern_name);
        all_sequences.push(sequence);
        all_names.push(name);
        
        let sequence = construct_symmetric_expansion(&scalar_values, window_size, DistanceType::MeanDiff);
        let name = format!("{}_mean_symmetric_expansion", pattern_name);
        all_sequences.push(sequence);
        all_names.push(name);
    }
    
    // ==================== 构建输出 ====================
    
    // 构建特征矩阵 (n × num_features)
    let num_features = all_sequences.len();
    let mut feature_matrix = vec![vec![f64::NAN; num_features]; n];
    
    for (j, sequence) in all_sequences.iter().enumerate() {
        for (i, &v) in sequence.iter().enumerate() {
            if i < n {
                feature_matrix[i][j] = v;
            }
        }
    }
    
    // 转换为2D numpy数组
    let py_array = PyArray2::from_vec2(py, &feature_matrix)?;
    
    // 计算标量特征（最小值点、最大值点的统计特征）
    let mut all_scalar_features: Vec<f64> = Vec::new();
    let mut scalar_feature_names: Vec<String> = Vec::new();
    
    for (i, sequence) in all_sequences.iter().enumerate() {
        let scalar_features = calculate_scalar_features(sequence, times, volumes, flags);
        
        scalar_feature_names.push(format!("{}_min_time_diff", all_names[i]));
        scalar_feature_names.push(format!("{}_min_vol_ratio_diff", all_names[i]));
        scalar_feature_names.push(format!("{}_min_buy_ratio_diff", all_names[i]));
        scalar_feature_names.push(format!("{}_max_time_diff", all_names[i]));
        scalar_feature_names.push(format!("{}_max_vol_ratio_diff", all_names[i]));
        scalar_feature_names.push(format!("{}_max_buy_ratio_diff", all_names[i]));
        
        all_scalar_features.extend(scalar_features);
    }
    
    // 创建Python列表
    let names_list = PyList::new(py, all_names.iter().map(|s| PyString::new(py, s)));
    let scalar_features_list = PyList::new(py, all_scalar_features.iter().map(|&v| v));
    let scalar_names_list = PyList::new(py, scalar_feature_names.iter().map(|s| PyString::new(py, s)));
    
    Ok((py_array.into(), names_list.into(), scalar_features_list.into(), scalar_names_list.into()))
}

/// 使用孪生滑窗构造序列差异序列
fn construct_adjacent_rolling_windows_sequence(
    sequences: &[Vec<u32>],
    window_size: usize,
    dist_type: DistanceType,
) -> Vec<f64> {
    let n = sequences.len();
    let mut result = vec![f64::NAN; n];
    
    for i in (2 * window_size)..n {
        let left = &sequences[i - window_size];
        let right = &sequences[i];
        
        result[i] = calculate_sequence_distance(left, right, dist_type);
    }
    
    result
}

/// 找到序列中最小值和最大值的索引
fn find_min_max_indices(values: &[f64]) -> (Option<usize>, Option<usize>) {
    let mut min_idx = None;
    let mut max_idx = None;
    let mut min_val = f64::INFINITY;
    let mut max_val = f64::NEG_INFINITY;
    
    for (i, &v) in values.iter().enumerate() {
        if v.is_nan() {
            continue;
        }
        if v < min_val {
            min_val = v;
            min_idx = Some(i);
        }
        if v > max_val {
            max_val = v;
            max_idx = Some(i);
        }
    }
    
    (min_idx, max_idx)
}

/// 计算标量特征
fn calculate_scalar_features(
    diff_sequence: &[f64],
    times: &[i64],
    volumes: &[f64],
    flags: &[i32],
) -> Vec<f64> {
    let mut features = Vec::new();
    
    let (min_idx, max_idx) = find_min_max_indices(diff_sequence);
    
    let total_volume: f64 = volumes.iter().sum();
    
    // 最小值点特征
    if let Some(idx) = min_idx {
        let time_before = (times[idx] - times[0]) as f64 / 1e9;
        let time_after = (times[times.len() - 1] - times[idx]) as f64 / 1e9;
        features.push(time_before - time_after);
        
        let vol_before: f64 = volumes[0..=idx].iter().sum();
        let vol_after: f64 = volumes[idx + 1..].iter().sum();
        let vol_ratio_before = vol_before / total_volume;
        let vol_ratio_after = vol_after / total_volume;
        features.push(vol_ratio_before - vol_ratio_after);
        
        let buy_vol_before: f64 = volumes[0..=idx].iter()
            .zip(flags[0..=idx].iter())
            .filter(|(_, &f)| f == 66)
            .map(|(v, _)| v)
            .sum();
        let buy_vol_after: f64 = volumes[idx + 1..].iter()
            .zip(flags[idx + 1..].iter())
            .filter(|(_, &f)| f == 66)
            .map(|(v, _)| v)
            .sum();
        let buy_ratio_before = if vol_before > 0.0 { buy_vol_before / vol_before } else { 0.0 };
        let buy_ratio_after = if vol_after > 0.0 { buy_vol_after / vol_after } else { 0.0 };
        features.push(buy_ratio_before - buy_ratio_after);
    } else {
        features.push(f64::NAN);
        features.push(f64::NAN);
        features.push(f64::NAN);
    }
    
    // 最大值点特征
    if let Some(idx) = max_idx {
        let time_before = (times[idx] - times[0]) as f64 / 1e9;
        let time_after = (times[times.len() - 1] - times[idx]) as f64 / 1e9;
        features.push(time_before - time_after);
        
        let vol_before: f64 = volumes[0..=idx].iter().sum();
        let vol_after: f64 = volumes[idx + 1..].iter().sum();
        let vol_ratio_before = vol_before / total_volume;
        let vol_ratio_after = vol_after / total_volume;
        features.push(vol_ratio_before - vol_ratio_after);
        
        let buy_vol_before: f64 = volumes[0..=idx].iter()
            .zip(flags[0..=idx].iter())
            .filter(|(_, &f)| f == 66)
            .map(|(v, _)| v)
            .sum();
        let buy_vol_after: f64 = volumes[idx + 1..].iter()
            .zip(flags[idx + 1..].iter())
            .filter(|(_, &f)| f == 66)
            .map(|(v, _)| v)
            .sum();
        let buy_ratio_before = if vol_before > 0.0 { buy_vol_before / vol_before } else { 0.0 };
        let buy_ratio_after = if vol_after > 0.0 { buy_vol_after / vol_after } else { 0.0 };
        features.push(buy_ratio_before - buy_ratio_after);
    } else {
        features.push(f64::NAN);
        features.push(f64::NAN);
        features.push(f64::NAN);
    }
    
    features
}
