//! 非对称大挂单（ALLO）微观结构特征计算模块 v3
//!
//! v3版本：性能优化版本
//! - 预计算滑动窗口平均值（O(1)查询）
//! - 二分查找范围查询
//! - 减少HashMap操作开销
//! - 减少内存分配

use ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1};
use pyo3::prelude::*;
use std::collections::HashMap;

const LEVEL_COUNT: usize = 10;
const EPS: f64 = 1e-9;

/// 盘口单档数据
#[derive(Copy, Clone, Debug)]
struct BookLevel {
    price: f64,
    volume: f64,
}

/// 盘口快照
#[derive(Copy, Clone, Debug)]
struct Snapshot {
    timestamp: i64,
    bids: [BookLevel; LEVEL_COUNT],
    asks: [BookLevel; LEVEL_COUNT],
}

impl Snapshot {
    fn mid_price(&self) -> f64 {
        (self.bids[0].price + self.asks[0].price) * 0.5
    }

    fn total_bid_volume(&self) -> f64 {
        self.bids.iter().map(|b| b.volume).sum()
    }

    fn total_ask_volume(&self) -> f64 {
        self.asks.iter().map(|a| a.volume).sum()
    }

    /// 按价格搜索买侧挂单量，返回(挂单量, 所在档位)
    fn find_bid_volume_by_price(&self, target_price: f64) -> Option<(f64, usize)> {
        for (level, bid) in self.bids.iter().enumerate() {
            if (bid.price - target_price).abs() < EPS {
                return Some((bid.volume, level));
            }
        }
        None
    }

    /// 按价格搜索卖侧挂单量，返回(挂单量, 所在档位)
    fn find_ask_volume_by_price(&self, target_price: f64) -> Option<(f64, usize)> {
        for (level, ask) in self.asks.iter().enumerate() {
            if (ask.price - target_price).abs() < EPS {
                return Some((ask.volume, level));
            }
        }
        None
    }

    /// 获取某价格前方的档位挂单量（对于买侧，前方是价格更高的档位）
    fn bid_volume_before_price(&self, target_price: f64) -> f64 {
        self.bids.iter()
            .filter(|b| b.price > target_price + EPS)
            .map(|b| b.volume)
            .sum()
    }

    /// 获取某价格前方的档位挂单量（对于卖侧，前方是价格更低的档位）
    fn ask_volume_before_price(&self, target_price: f64) -> f64 {
        self.asks.iter()
            .filter(|a| a.price < target_price - EPS)
            .map(|a| a.volume)
            .sum()
    }
}

/// 逐笔成交
#[derive(Copy, Clone, Debug)]
struct Trade {
    timestamp: i64,
    price: f64,
    volume: f64,
    turnover: f64,
    flag: i32,
}

/// 异常流动性聚集事件 v2 - 价格锚定
#[derive(Copy, Clone, Debug)]
struct ALAEventV2 {
    start_idx: usize,
    end_idx: usize,
    start_time: i64,
    end_time: i64,
    is_bid: bool,
    price: f64,
    peak_volume: f64,
    initial_volume: f64,
    initial_level: usize,
}

/// ALA事件特征
#[derive(Clone, Debug, Default)]
struct ALAFeatures {
    m1_relative_prominence: f64,
    m3_flicker_frequency: f64,
    m7_queue_loitering_duration: f64,
    m8_frontrun_passive: f64,
    m9_frontrun_active: f64,
    m10_ally_retreat_rate: f64,
    m11a_attack_skewness_opponent: f64,
    m12a_peak_latency_ratio_opponent: f64,
    m13a_courage_acceleration_opponent: f64,
    m14a_rhythm_entropy_opponent: f64,
    m11b_attack_skewness_ally: f64,
    m12b_peak_latency_ratio_ally: f64,
    m13b_courage_acceleration_ally: f64,
    m14b_rhythm_entropy_ally: f64,
    m15_fox_tiger_index: f64,
    m16_shadow_projection_ratio: f64,
    m17_gravitational_redshift: f64,
    m19_shielding_thickness_ratio: f64,
    m20_oxygen_saturation: f64,
    m21_suffocation_integral: f64,
    m22_local_survivor_bias: f64,
}

impl ALAFeatures {
    fn to_vec(&self) -> Vec<f64> {
        vec![
            self.m1_relative_prominence,
            self.m3_flicker_frequency,
            self.m7_queue_loitering_duration,
            self.m8_frontrun_passive,
            self.m9_frontrun_active,
            self.m10_ally_retreat_rate,
            self.m11a_attack_skewness_opponent,
            self.m12a_peak_latency_ratio_opponent,
            self.m13a_courage_acceleration_opponent,
            self.m14a_rhythm_entropy_opponent,
            self.m11b_attack_skewness_ally,
            self.m12b_peak_latency_ratio_ally,
            self.m13b_courage_acceleration_ally,
            self.m14b_rhythm_entropy_ally,
            self.m15_fox_tiger_index,
            self.m16_shadow_projection_ratio,
            self.m17_gravitational_redshift,
            self.m19_shielding_thickness_ratio,
            self.m20_oxygen_saturation,
            self.m21_suffocation_integral,
            self.m22_local_survivor_bias,
        ]
    }
}

fn get_feature_names() -> Vec<String> {
    vec![
        "M1_relative_prominence".to_string(),
        "M3_flicker_frequency".to_string(),
        "M7_queue_loitering_duration".to_string(),
        "M8_frontrun_passive".to_string(),
        "M9_frontrun_active".to_string(),
        "M10_ally_retreat_rate".to_string(),
        "M11a_attack_skewness_opponent".to_string(),
        "M12a_peak_latency_ratio_opponent".to_string(),
        "M13a_courage_acceleration_opponent".to_string(),
        "M14a_rhythm_entropy_opponent".to_string(),
        "M11b_attack_skewness_ally".to_string(),
        "M12b_peak_latency_ratio_ally".to_string(),
        "M13b_courage_acceleration_ally".to_string(),
        "M14b_rhythm_entropy_ally".to_string(),
        "M15_fox_tiger_index".to_string(),
        "M16_shadow_projection_ratio".to_string(),
        "M17_gravitational_redshift".to_string(),
        "M19_shielding_thickness_ratio".to_string(),
        "M20_oxygen_saturation".to_string(),
        "M21_suffocation_integral".to_string(),
        "M22_local_survivor_bias".to_string(),
    ]
}

/// 价格滑动窗口平均值缓存 - 核心优化
/// 使用滑动窗口累积和，实现O(1)的平均值查询
struct PriceMovingAvgCache {
    // 每个价格: (快照索引数组, 累积和数组)
    data: HashMap<i64, (Vec<usize>, Vec<f64>)>,
    window_size: usize,
}

impl PriceMovingAvgCache {
    fn new(snapshots: &[Snapshot], is_bid: bool, window_size: usize) -> Self {
        let mut data: HashMap<i64, (Vec<usize>, Vec<f64>)> = HashMap::new();
        let price_to_key = |p: f64| -> i64 { (p * 10000.0).round() as i64 };
        
        // 收集每个价格出现的(索引, 挂单量)
        for (idx, snap) in snapshots.iter().enumerate() {
            let levels = if is_bid { &snap.bids } else { &snap.asks };
            for level in levels.iter() {
                if level.volume > EPS {
                    let key = price_to_key(level.price);
                    let entry = data.entry(key).or_insert((Vec::new(), Vec::new()));
                    entry.0.push(idx);
                    entry.1.push(level.volume);
                }
            }
        }
        
        // 构建累积和（用于快速计算窗口内总和）
        for (_, (indices, cumsum)) in data.iter_mut() {
            let n = cumsum.len();
            if n == 0 {
                continue;
            }
            // 原地转换为累积和
            for i in 1..n {
                cumsum[i] += cumsum[i - 1];
            }
        }
        
        PriceMovingAvgCache { data, window_size }
    }
    
    /// 获取某价格在某快照索引之前的window_size个快照中的平均挂单量
    /// 使用二分查找 + 累积和，复杂度O(log N)
    fn get_avg(&self, price_key: i64, current_idx: usize) -> f64 {
        if let Some((indices, cumsum)) = self.data.get(&price_key) {
            if indices.is_empty() {
                return 0.0;
            }
            
            // 二分查找: 找到最后一个 < current_idx 的位置
            let end_pos = match indices.binary_search(&current_idx) {
                Ok(pos) => pos,  // 找到current_idx，之前的位置
                Err(pos) => pos, // 插入位置就是第一个>=current_idx的位置
            };
            
            if end_pos == 0 {
                return 0.0;
            }
            
            // 计算窗口起始索引
            let window_start = current_idx.saturating_sub(self.window_size);
            
            // 二分查找: 找到第一个 >= window_start 的位置
            let start_pos = match indices.binary_search(&window_start) {
                Ok(pos) => pos,
                Err(pos) => pos,
            };
            
            // 计算 (start_pos, end_pos) 范围内的平均值
            let count = end_pos - start_pos;
            if count == 0 {
                return 0.0;
            }
            
            let sum = if start_pos == 0 {
                cumsum[end_pos - 1]
            } else {
                cumsum[end_pos - 1] - cumsum[start_pos - 1]
            };
            
            sum / count as f64
        } else {
            0.0
        }
    }
}

/// 带标签的ALA事件
#[derive(Copy, Clone, Debug)]
struct LabeledALAEventV2 {
    event: ALAEventV2,
    detection_mode_idx: usize,
    side_filter_idx: usize,
}

/// 检测ALA事件 v3（优化版本）
fn detect_ala_events_v3(
    snapshots: &[Snapshot],
    detection_modes: &[&str],
    side_filters: &[&str],
    k1_horizontal: f64,
    k2_vertical: f64,
    window_size: usize,
    decay_threshold: f64,
) -> Vec<LabeledALAEventV2> {
    let mut labeled_events = Vec::new();
    let n = snapshots.len();
    if n < window_size + 1 {
        return labeled_events;
    }

    let price_to_key = |p: f64| -> i64 { (p * 10000.0).round() as i64 };

    // 预构建价格滑动窗口平均值缓存
    let bid_avg_cache = PriceMovingAvgCache::new(snapshots, true, window_size);
    let ask_avg_cache = PriceMovingAvgCache::new(snapshots, false, window_size);

    // 对每种组合进行检测
    for (dm_idx, &detection_mode) in detection_modes.iter().enumerate() {
        for (sf_idx, &side_filter) in side_filters.iter().enumerate() {
            let check_bid = side_filter == "bid" || side_filter == "both";
            let check_ask = side_filter == "ask" || side_filter == "both";

            // 事件追踪状态
            let mut in_event_bid: HashMap<i64, (usize, f64, f64)> = HashMap::new();
            let mut in_event_ask: HashMap<i64, (usize, f64, f64)> = HashMap::new();

            for i in window_size..n {
                let snap = &snapshots[i];

                // 检查买侧
                if check_bid {
                    for (level, bid) in snap.bids.iter().enumerate() {
                        let price_key = price_to_key(bid.price);
                        
                        if in_event_bid.contains_key(&price_key) {
                            let (start_idx, initial_vol, peak_vol) = *in_event_bid.get(&price_key).unwrap();
                            let new_peak = peak_vol.max(bid.volume);
                            in_event_bid.insert(price_key, (start_idx, initial_vol, new_peak));
                            
                            if bid.volume < decay_threshold * initial_vol {
                                let (start_idx, initial_vol, peak_vol) = in_event_bid.remove(&price_key).unwrap();
                                labeled_events.push(LabeledALAEventV2 {
                                    event: ALAEventV2 {
                                        start_idx,
                                        end_idx: i,
                                        start_time: snapshots[start_idx].timestamp,
                                        end_time: snap.timestamp,
                                        is_bid: true,
                                        price: bid.price,
                                        peak_volume: peak_vol,
                                        initial_volume: initial_vol,
                                        initial_level: level,
                                    },
                                    detection_mode_idx: dm_idx,
                                    side_filter_idx: sf_idx,
                                });
                            }
                            continue;
                        }

                        let h_trigger = {
                            // 优化：预计算总挂单量，避免两次遍历
                            let total_vol: f64 = snap.bids.iter().map(|b| b.volume).sum();
                            let other_vol = total_vol - bid.volume;
                            other_vol > EPS && bid.volume > k1_horizontal * other_vol
                        };

                        let v_trigger = {
                            let avg = bid_avg_cache.get_avg(price_key, i);
                            avg > EPS && bid.volume > k2_vertical * avg
                        };

                        let triggered = match detection_mode {
                            "horizontal" => h_trigger,
                            "vertical" => v_trigger,
                            _ => h_trigger || v_trigger,
                        };

                        if triggered && !in_event_bid.contains_key(&price_key) {
                            in_event_bid.insert(price_key, (i, bid.volume, bid.volume));
                        }
                    }
                    
                    // 检查买侧事件是否因价格消失而结束
                    let prices_to_check: Vec<i64> = in_event_bid.keys().cloned().collect();
                    for price_key in prices_to_check {
                        let target_price = price_key as f64 / 10000.0;
                        if snap.find_bid_volume_by_price(target_price).is_none() {
                            let (start_idx, initial_vol, peak_vol) = in_event_bid.remove(&price_key).unwrap();
                            labeled_events.push(LabeledALAEventV2 {
                                event: ALAEventV2 {
                                    start_idx,
                                    end_idx: i,
                                    start_time: snapshots[start_idx].timestamp,
                                    end_time: snap.timestamp,
                                    is_bid: true,
                                    price: target_price,
                                    peak_volume: peak_vol,
                                    initial_volume: initial_vol,
                                    initial_level: 0,
                                },
                                detection_mode_idx: dm_idx,
                                side_filter_idx: sf_idx,
                            });
                        }
                    }
                }

                // 检查卖侧
                if check_ask {
                    for (level, ask) in snap.asks.iter().enumerate() {
                        let price_key = price_to_key(ask.price);
                        
                        if in_event_ask.contains_key(&price_key) {
                            let (start_idx, initial_vol, peak_vol) = *in_event_ask.get(&price_key).unwrap();
                            let new_peak = peak_vol.max(ask.volume);
                            in_event_ask.insert(price_key, (start_idx, initial_vol, new_peak));
                            
                            if ask.volume < decay_threshold * initial_vol {
                                let (start_idx, initial_vol, peak_vol) = in_event_ask.remove(&price_key).unwrap();
                                labeled_events.push(LabeledALAEventV2 {
                                    event: ALAEventV2 {
                                        start_idx,
                                        end_idx: i,
                                        start_time: snapshots[start_idx].timestamp,
                                        end_time: snap.timestamp,
                                        is_bid: false,
                                        price: ask.price,
                                        peak_volume: peak_vol,
                                        initial_volume: initial_vol,
                                        initial_level: level,
                                    },
                                    detection_mode_idx: dm_idx,
                                    side_filter_idx: sf_idx,
                                });
                            }
                            continue;
                        }

                        let h_trigger = {
                            // 优化：预计算总挂单量，避免两次遍历
                            let total_vol: f64 = snap.asks.iter().map(|a| a.volume).sum();
                            let other_vol = total_vol - ask.volume;
                            other_vol > EPS && ask.volume > k1_horizontal * other_vol
                        };

                        let v_trigger = {
                            let avg = ask_avg_cache.get_avg(price_key, i);
                            avg > EPS && ask.volume > k2_vertical * avg
                        };

                        let triggered = match detection_mode {
                            "horizontal" => h_trigger,
                            "vertical" => v_trigger,
                            _ => h_trigger || v_trigger,
                        };

                        if triggered && !in_event_ask.contains_key(&price_key) {
                            in_event_ask.insert(price_key, (i, ask.volume, ask.volume));
                        }
                    }
                    
                    // 检查卖侧事件是否因价格消失而结束
                    let prices_to_check: Vec<i64> = in_event_ask.keys().cloned().collect();
                    for price_key in prices_to_check {
                        let target_price = price_key as f64 / 10000.0;
                        if snap.find_ask_volume_by_price(target_price).is_none() {
                            let (start_idx, initial_vol, peak_vol) = in_event_ask.remove(&price_key).unwrap();
                            labeled_events.push(LabeledALAEventV2 {
                                event: ALAEventV2 {
                                    start_idx,
                                    end_idx: i,
                                    start_time: snapshots[start_idx].timestamp,
                                    end_time: snap.timestamp,
                                    is_bid: false,
                                    price: target_price,
                                    peak_volume: peak_vol,
                                    initial_volume: initial_vol,
                                    initial_level: 0,
                                },
                                detection_mode_idx: dm_idx,
                                side_filter_idx: sf_idx,
                            });
                        }
                    }
                }
            }

            // 处理未结束的事件
            let last_idx = n - 1;
            for (&price_key, &(start_idx, initial_vol, peak_vol)) in in_event_bid.iter() {
                let target_price = price_key as f64 / 10000.0;
                labeled_events.push(LabeledALAEventV2 {
                    event: ALAEventV2 {
                        start_idx,
                        end_idx: last_idx,
                        start_time: snapshots[start_idx].timestamp,
                        end_time: snapshots[last_idx].timestamp,
                        is_bid: true,
                        price: target_price,
                        peak_volume: peak_vol,
                        initial_volume: initial_vol,
                        initial_level: 0,
                    },
                    detection_mode_idx: dm_idx,
                    side_filter_idx: sf_idx,
                });
            }
            for (&price_key, &(start_idx, initial_vol, peak_vol)) in in_event_ask.iter() {
                let target_price = price_key as f64 / 10000.0;
                labeled_events.push(LabeledALAEventV2 {
                    event: ALAEventV2 {
                        start_idx,
                        end_idx: last_idx,
                        start_time: snapshots[start_idx].timestamp,
                        end_time: snapshots[last_idx].timestamp,
                        is_bid: false,
                        price: target_price,
                        peak_volume: peak_vol,
                        initial_volume: initial_vol,
                        initial_level: 0,
                    },
                    detection_mode_idx: dm_idx,
                    side_filter_idx: sf_idx,
                });
            }
        }
    }

    // 对事件进行排序，确保一致的顺序
    // 排序规则：按 (detection_mode_idx, side_filter_idx, start_idx, price_key) 排序
    labeled_events.sort_by(|a, b| {
        a.detection_mode_idx
            .cmp(&b.detection_mode_idx)
            .then(a.side_filter_idx.cmp(&b.side_filter_idx))
            .then(a.event.start_idx.cmp(&b.event.start_idx))
            .then(
                (a.event.price * 10000.0).round().partial_cmp(&(b.event.price * 10000.0).round())
                    .unwrap_or(std::cmp::Ordering::Equal)
            )
    });

    labeled_events
}

/// 二分查找获取时间范围内的索引（包含边界）
fn find_range_indices(
    timestamps: &[i64],
    start_time: i64,
    end_time: i64,
) -> (usize, usize) {
    let n = timestamps.len();
    if n == 0 {
        return (0, 0);
    }
    
    // 找到第一个 >= start_time 的位置
    let start_idx = match timestamps.binary_search(&start_time) {
        Ok(pos) => pos,
        Err(pos) => pos.min(n),
    };
    
    // 找到最后一个 <= end_time 的位置，然后 end_idx = 该位置 + 1
    // 使用 binary_search_by 来找第一个 > end_time 的位置
    let end_idx = timestamps.partition_point(|&t| t <= end_time);
    
    (start_idx, end_idx)
}

/// 计算标准差
fn std_dev(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (values.len() - 1) as f64;
    variance.sqrt()
}

/// 计算偏度
fn skewness(values: &[f64]) -> f64 {
    if values.len() < 3 {
        return 0.0;
    }
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let m2 = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
    let m3 = values.iter().map(|v| (v - mean).powi(3)).sum::<f64>() / n;
    if m2 < EPS {
        return 0.0;
    }
    m3 / m2.powf(1.5)
}

/// 计算相关系数
fn correlation(x: &[f64], y: &[f64]) -> f64 {
    if x.len() != y.len() || x.len() < 2 {
        return 0.0;
    }
    let n = x.len() as f64;
    let mean_x = x.iter().sum::<f64>() / n;
    let mean_y = y.iter().sum::<f64>() / n;
    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;
    for i in 0..x.len() {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }
    if var_x < EPS || var_y < EPS {
        return 0.0;
    }
    cov / (var_x.sqrt() * var_y.sqrt())
}

/// 计算熵
fn entropy(intervals: &[f64]) -> f64 {
    if intervals.is_empty() {
        return 0.0;
    }
    let min_val = intervals.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_val = intervals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if (max_val - min_val).abs() < EPS {
        return 0.0;
    }
    let num_bins = 10;
    let bin_width = (max_val - min_val) / num_bins as f64;
    let mut counts = [0usize; 10];
    for &v in intervals {
        let bin = ((v - min_val) / bin_width).floor() as usize;
        let bin = bin.min(num_bins - 1);
        counts[bin] += 1;
    }
    let total = intervals.len() as f64;
    let mut h = 0.0;
    for &c in &counts {
        if c > 0 {
            let p = c as f64 / total;
            h -= p * p.ln();
        }
    }
    h
}

/// 计算时间形态学特征
fn compute_temporal_morphology(
    trades: &[Trade],
    trade_indices: &[usize],
    event_start_time: i64,
    event_end_time: i64,
) -> (f64, f64, f64, f64) {
    if trade_indices.is_empty() {
        return (0.0, 0.0, 0.0, 0.0);
    }
    
    let duration = (event_end_time - event_start_time) as f64;
    if duration < EPS {
        return (0.0, 0.0, 0.0, 0.0);
    }

    let mut trade_times = Vec::with_capacity(trade_indices.len());
    let mut trade_sizes = Vec::with_capacity(trade_indices.len());
    let mut max_vol = 0.0;
    let mut max_time = event_start_time;
    
    for &idx in trade_indices {
        let t = &trades[idx];
        let rel_time = (t.timestamp - event_start_time) as f64;
        trade_times.push(rel_time);
        trade_sizes.push(t.volume);
        if t.volume > max_vol {
            max_vol = t.volume;
            max_time = t.timestamp;
        }
    }
    
    let attack_skewness = skewness(&trade_times);
    let peak_latency_ratio = (max_time - event_start_time) as f64 / duration;
    
    let trade_seq: Vec<f64> = (0..trade_sizes.len()).map(|i| i as f64).collect();
    let courage_acceleration = correlation(&trade_seq, &trade_sizes);

    let mut intervals = Vec::with_capacity(trade_indices.len().saturating_sub(1));
    for i in 1..trade_indices.len() {
        let dt = (trades[trade_indices[i]].timestamp - trades[trade_indices[i - 1]].timestamp) as f64;
        intervals.push(dt);
    }
    let rhythm_entropy = entropy(&intervals);

    (attack_skewness, peak_latency_ratio, courage_acceleration, rhythm_entropy)
}

/// 计算单个ALA事件的特征 - 优化版本
fn compute_ala_features_v3(
    event: &ALAEventV2,
    snapshots: &[Snapshot],
    snap_timestamps: &[i64],
    trades: &[Trade],
    trade_timestamps: &[i64],
) -> ALAFeatures {
    let mut features = ALAFeatures::default();

    // 使用二分查找获取索引范围
    let (snap_start, snap_end) = find_range_indices(snap_timestamps, event.start_time, event.end_time);
    let (trade_start, trade_end) = find_range_indices(trade_timestamps, event.start_time, event.end_time);
    
    if snap_start >= snap_end {
        return features;
    }

    let event_snap_count = snap_end - snap_start;
    let duration_seconds = (event.end_time - event.start_time) as f64 / 1e9;

    // ============ 合并遍历：M1, M3, M7, M8 ============
    let mut sum_opposite = 0.0;
    let mut sum_better_levels = 0.0;
    let mut volumes = Vec::with_capacity(event_snap_count);
    let mut became_level1_time: Option<i64> = None;
    
    for i in snap_start..snap_end {
        let snap = &snapshots[i];
        
        // M1: 相对凸度 - 累加对侧前5档平均挂单量
        if event.is_bid {
            sum_opposite += snap.asks[0..5].iter().map(|a| a.volume).sum::<f64>() / 5.0;
        } else {
            sum_opposite += snap.bids[0..5].iter().map(|b| b.volume).sum::<f64>() / 5.0;
        }
        
        // M3: 闪烁频率 - 收集该价格的挂单量
        if event.is_bid {
            if let Some((vol, _)) = snap.find_bid_volume_by_price(event.price) {
                volumes.push(vol);
            }
        } else {
            if let Some((vol, _)) = snap.find_ask_volume_by_price(event.price) {
                volumes.push(vol);
            }
        }
        
        // M7: 队列滞留时长 - 检查是否成为一档
        if became_level1_time.is_none() {
            let is_level1 = if event.is_bid {
                (snap.bids[0].price - event.price).abs() < EPS
            } else {
                (snap.asks[0].price - event.price).abs() < EPS
            };
            if is_level1 {
                became_level1_time = Some(snap.timestamp);
            }
        }
        
        // M8: 抢跑强度-挂单版 - 累加前方档位挂单量
        sum_better_levels += if event.is_bid {
            snap.bid_volume_before_price(event.price)
        } else {
            snap.ask_volume_before_price(event.price)
        };
    }
    
    // M1 计算
    let avg_opposite = sum_opposite / event_snap_count as f64;
    features.m1_relative_prominence = if avg_opposite > EPS {
        event.peak_volume / avg_opposite
    } else {
        0.0
    };

    // M3 计算
    let vol_std = std_dev(&volumes);
    let vol_mean = if !volumes.is_empty() {
        volumes.iter().sum::<f64>() / volumes.len() as f64
    } else {
        0.0
    };
    let mut change_count = 0;
    for i in 1..volumes.len() {
        if (volumes[i] - volumes[i - 1]).abs() > vol_mean * 0.1 {
            change_count += 1;
        }
    }
    let flicker_freq = if duration_seconds > EPS {
        change_count as f64 / duration_seconds
    } else {
        0.0
    };
    features.m3_flicker_frequency = if vol_std < vol_mean * 0.2 {
        flicker_freq
    } else {
        flicker_freq * 0.5
    };

    // M7 计算
    features.m7_queue_loitering_duration = match became_level1_time {
        Some(t) => (t - event.start_time) as f64 / 1e9,
        None => duration_seconds,
    };

    // M8 计算
    let avg_better_levels = sum_better_levels / event_snap_count as f64;
    features.m8_frontrun_passive = if event.peak_volume > EPS {
        avg_better_levels / event.peak_volume
    } else {
        0.0
    };

    // M9: 抢跑强度-主买版
    let mut m9_sum = 0.0;
    for i in trade_start..trade_end {
        let t = &trades[i];
        if event.is_bid && t.flag == 66 || !event.is_bid && t.flag == 83 {
            m9_sum += t.volume;
        }
    }
    features.m9_frontrun_active = m9_sum;

    // M10: 同侧撤单率
    let same_side_start: f64 = if event.is_bid {
        snapshots[event.start_idx].total_bid_volume()
    } else {
        snapshots[event.start_idx].total_ask_volume()
    };
    let same_side_end: f64 = if event.is_bid {
        snapshots[event.end_idx].total_bid_volume()
    } else {
        snapshots[event.end_idx].total_ask_volume()
    };
    features.m10_ally_retreat_rate = if same_side_start > EPS {
        (same_side_start - same_side_end).max(0.0) / same_side_start
    } else {
        0.0
    };

    // ============ 第四部分：时间形态学特征 ============

    // 分离对手和己方成交
    let mut opponent_indices = Vec::new();
    let mut ally_indices = Vec::new();
    for i in trade_start..trade_end {
        let t = &trades[i];
        if event.is_bid {
            if t.flag == 83 {
                opponent_indices.push(i);
            } else if t.flag == 66 {
                ally_indices.push(i);
            }
        } else {
            if t.flag == 66 {
                opponent_indices.push(i);
            } else if t.flag == 83 {
                ally_indices.push(i);
            }
        }
    }

    let (skew_opp, peak_opp, courage_opp, entropy_opp) = 
        compute_temporal_morphology(trades, &opponent_indices, event.start_time, event.end_time);
    features.m11a_attack_skewness_opponent = skew_opp;
    features.m12a_peak_latency_ratio_opponent = peak_opp;
    features.m13a_courage_acceleration_opponent = courage_opp;
    features.m14a_rhythm_entropy_opponent = entropy_opp;

    let (skew_ally, peak_ally, courage_ally, entropy_ally) = 
        compute_temporal_morphology(trades, &ally_indices, event.start_time, event.end_time);
    features.m11b_attack_skewness_ally = skew_ally;
    features.m12b_peak_latency_ratio_ally = peak_ally;
    features.m13b_courage_acceleration_ally = courage_ally;
    features.m14b_rhythm_entropy_ally = entropy_ally;

    // ============ 第五部分：空间场论特征 ============

    // M15: 狐假虎威指数
    let during_avg = avg_better_levels;
    
    let history_start = event.start_idx.saturating_sub(100);
    let history_count = event.start_idx - history_start;
    let history_avg: f64 = if history_count > 0 {
        let mut sum = 0.0;
        for i in history_start..event.start_idx {
            let snap = &snapshots[i];
            sum += if event.is_bid {
                snap.bid_volume_before_price(event.price)
            } else {
                snap.ask_volume_before_price(event.price)
            };
        }
        sum / history_count as f64
    } else {
        during_avg
    };
    features.m15_fox_tiger_index = if history_avg > EPS {
        during_avg / history_avg
    } else {
        1.0
    };

    // M16: 阴影投射比
    let mut front_trades_during = 0usize;
    for i in trade_start..trade_end {
        let t = &trades[i];
        if event.is_bid && t.price > event.price || !event.is_bid && t.price < event.price {
            front_trades_during += 1;
        }
    }
    let freq_during = if duration_seconds > EPS {
        front_trades_during as f64 / duration_seconds
    } else {
        0.0
    };

    let history_start_time = event.start_time - 300_000_000_000i64;
    let (hist_trade_start, hist_trade_end) = find_range_indices(trade_timestamps, history_start_time, event.start_time);
    let mut front_trades_history = 0usize;
    for i in hist_trade_start..hist_trade_end {
        let t = &trades[i];
        if event.is_bid && t.price > event.price || !event.is_bid && t.price < event.price {
            front_trades_history += 1;
        }
    }
    let history_duration = 300.0;
    let freq_history = front_trades_history as f64 / history_duration;

    features.m16_shadow_projection_ratio = if freq_history > EPS {
        freq_during / freq_history
    } else {
        1.0
    };

    // M17: 引力红移速率
    let mut approach_speeds = Vec::with_capacity(event_snap_count.saturating_sub(1));
    for i in snap_start..snap_end.saturating_sub(1) {
        let snap_before = &snapshots[i];
        let snap_after = &snapshots[i + 1];
        
        let gap_before = if event.is_bid {
            snap_before.asks[0].price - event.price
        } else {
            event.price - snap_before.bids[0].price
        };
        let gap_after = if event.is_bid {
            snap_after.asks[0].price - event.price
        } else {
            event.price - snap_after.bids[0].price
        };
        let dt = (snap_after.timestamp - snap_before.timestamp) as f64 / 1e9;
        if dt > EPS {
            approach_speeds.push((gap_before - gap_after) / dt);
        }
    }
    features.m17_gravitational_redshift = if !approach_speeds.is_empty() {
        approach_speeds.iter().sum::<f64>() / approach_speeds.len() as f64
    } else {
        0.0
    };

    // M19: 垫单厚度比
    features.m19_shielding_thickness_ratio = if event.peak_volume > EPS {
        during_avg / event.peak_volume
    } else {
        0.0
    };

    // ============ 第六部分：盈亏结局特征 ============

    // M20, M21: 氧气饱和度和窒息积分
    // 收集active trades
    let mut active_trades: Vec<(usize, f64)> = Vec::new();  // (index, price)
    for i in trade_start..trade_end {
        let t = &trades[i];
        if (event.is_bid && t.flag == 66) || (!event.is_bid && t.flag == 83) {
            active_trades.push((i, t.price));
        }
    }
    
    if !active_trades.is_empty() {
        let mut profit_time = 0.0;
        let mut total_time = 0.0;
        let mut suffocation = 0.0;
        
        for &(trade_idx, trade_price) in &active_trades {
            let (rem_start, rem_end) = find_range_indices(
                snap_timestamps,
                trades[trade_idx].timestamp + 1,
                i64::MAX,
            );
            
            let rem_count = rem_end - rem_start;
            for (offset, j) in (rem_start..rem_end).enumerate() {
                let snap = &snapshots[j];
                let (dt_nano, dt_sec) = if offset + 1 < rem_count {
                    let dt = (snapshots[j + 1].timestamp - snap.timestamp) as f64;
                    (dt, dt / 1e9)
                } else {
                    (1e9, 0.0)
                };
                total_time += dt_nano;
                let mid = snap.mid_price();
                let in_profit = if event.is_bid {
                    mid > trade_price
                } else {
                    mid < trade_price
                };
                if in_profit {
                    profit_time += dt_nano;
                }
                
                let loss = if event.is_bid {
                    (trade_price - mid).max(0.0) / trade_price
                } else {
                    (mid - trade_price).max(0.0) / trade_price
                };
                suffocation += loss * dt_sec;
            }
        }
        features.m20_oxygen_saturation = if total_time > EPS {
            profit_time / total_time
        } else {
            0.5
        };
        features.m21_suffocation_integral = suffocation;
    } else {
        features.m20_oxygen_saturation = 0.5;
    }

    // M22: 本地幸存者偏差
    let vwap_during = if trade_start < trade_end {
        let total_turnover: f64 = (trade_start..trade_end).map(|i| trades[i].turnover).sum();
        let total_vol: f64 = (trade_start..trade_end).map(|i| trades[i].volume).sum();
        if total_vol > EPS {
            total_turnover / total_vol
        } else {
            0.0
        }
    } else {
        0.0
    };

    let neighbor_start = event.start_time - 300_000_000_000i64;
    let neighbor_end = event.end_time + 300_000_000_000i64;
    
    // 获取前后5分钟的成交（不包含事件期间的成交）
    // v2逻辑: (t.timestamp >= neighbor_start && t.timestamp < event.start_time)
    //         || (t.timestamp > event.end_time && t.timestamp <= neighbor_end)
    let (before_start, before_end) = find_range_indices(trade_timestamps, neighbor_start, event.start_time - 1);
    let (after_start, after_end) = find_range_indices(trade_timestamps, event.end_time + 1, neighbor_end);
    
    let vwap_neighbor = {
        let mut total_turnover = 0.0;
        let mut total_vol = 0.0;
        for i in before_start..before_end {
            total_turnover += trades[i].turnover;
            total_vol += trades[i].volume;
        }
        for i in after_start..after_end {
            total_turnover += trades[i].turnover;
            total_vol += trades[i].volume;
        }
        if total_vol > EPS {
            total_turnover / total_vol
        } else {
            vwap_during
        }
    };
    features.m22_local_survivor_bias = vwap_during - vwap_neighbor;

    features
}


/// 计算非对称大挂单微观结构特征 v3 - 性能优化版本
#[pyfunction]
#[pyo3(signature = (
    trade_exchtime,
    trade_price,
    trade_volume,
    trade_turnover,
    trade_flag,
    snap_exchtime,
    bid_prc1, bid_prc2, bid_prc3, bid_prc4, bid_prc5,
    bid_prc6, bid_prc7, bid_prc8, bid_prc9, bid_prc10,
    bid_vol1, bid_vol2, bid_vol3, bid_vol4, bid_vol5,
    bid_vol6, bid_vol7, bid_vol8, bid_vol9, bid_vol10,
    ask_prc1, ask_prc2, ask_prc3, ask_prc4, ask_prc5,
    ask_prc6, ask_prc7, ask_prc8, ask_prc9, ask_prc10,
    ask_vol1, ask_vol2, ask_vol3, ask_vol4, ask_vol5,
    ask_vol6, ask_vol7, ask_vol8, ask_vol9, ask_vol10,
    k1_horizontal = 2.0,
    k2_vertical = 5.0,
    window_size = 100,
    decay_threshold = 0.5
))]
pub fn compute_allo_microstructure_features_tris_expanded_v3(
    py: Python,
    trade_exchtime: PyReadonlyArray1<i64>,
    trade_price: PyReadonlyArray1<f64>,
    trade_volume: PyReadonlyArray1<f64>,
    trade_turnover: PyReadonlyArray1<f64>,
    trade_flag: PyReadonlyArray1<i32>,
    snap_exchtime: PyReadonlyArray1<i64>,
    bid_prc1: PyReadonlyArray1<f64>,
    bid_prc2: PyReadonlyArray1<f64>,
    bid_prc3: PyReadonlyArray1<f64>,
    bid_prc4: PyReadonlyArray1<f64>,
    bid_prc5: PyReadonlyArray1<f64>,
    bid_prc6: PyReadonlyArray1<f64>,
    bid_prc7: PyReadonlyArray1<f64>,
    bid_prc8: PyReadonlyArray1<f64>,
    bid_prc9: PyReadonlyArray1<f64>,
    bid_prc10: PyReadonlyArray1<f64>,
    bid_vol1: PyReadonlyArray1<f64>,
    bid_vol2: PyReadonlyArray1<f64>,
    bid_vol3: PyReadonlyArray1<f64>,
    bid_vol4: PyReadonlyArray1<f64>,
    bid_vol5: PyReadonlyArray1<f64>,
    bid_vol6: PyReadonlyArray1<f64>,
    bid_vol7: PyReadonlyArray1<f64>,
    bid_vol8: PyReadonlyArray1<f64>,
    bid_vol9: PyReadonlyArray1<f64>,
    bid_vol10: PyReadonlyArray1<f64>,
    ask_prc1: PyReadonlyArray1<f64>,
    ask_prc2: PyReadonlyArray1<f64>,
    ask_prc3: PyReadonlyArray1<f64>,
    ask_prc4: PyReadonlyArray1<f64>,
    ask_prc5: PyReadonlyArray1<f64>,
    ask_prc6: PyReadonlyArray1<f64>,
    ask_prc7: PyReadonlyArray1<f64>,
    ask_prc8: PyReadonlyArray1<f64>,
    ask_prc9: PyReadonlyArray1<f64>,
    ask_prc10: PyReadonlyArray1<f64>,
    ask_vol1: PyReadonlyArray1<f64>,
    ask_vol2: PyReadonlyArray1<f64>,
    ask_vol3: PyReadonlyArray1<f64>,
    ask_vol4: PyReadonlyArray1<f64>,
    ask_vol5: PyReadonlyArray1<f64>,
    ask_vol6: PyReadonlyArray1<f64>,
    ask_vol7: PyReadonlyArray1<f64>,
    ask_vol8: PyReadonlyArray1<f64>,
    ask_vol9: PyReadonlyArray1<f64>,
    ask_vol10: PyReadonlyArray1<f64>,
    k1_horizontal: f64,
    k2_vertical: f64,
    window_size: usize,
    decay_threshold: f64,
) -> PyResult<(Vec<Py<PyArray2<f64>>>, Vec<Vec<String>>)> {
    let trade_exchtime = trade_exchtime.as_slice()?;
    let trade_price = trade_price.as_slice()?;
    let trade_volume = trade_volume.as_slice()?;
    let trade_turnover = trade_turnover.as_slice()?;
    let trade_flag = trade_flag.as_slice()?;
    let snap_exchtime = snap_exchtime.as_slice()?;

    let bid_prc: [&[f64]; LEVEL_COUNT] = [
        bid_prc1.as_slice()?, bid_prc2.as_slice()?, bid_prc3.as_slice()?,
        bid_prc4.as_slice()?, bid_prc5.as_slice()?, bid_prc6.as_slice()?,
        bid_prc7.as_slice()?, bid_prc8.as_slice()?, bid_prc9.as_slice()?,
        bid_prc10.as_slice()?,
    ];
    let bid_vol: [&[f64]; LEVEL_COUNT] = [
        bid_vol1.as_slice()?, bid_vol2.as_slice()?, bid_vol3.as_slice()?,
        bid_vol4.as_slice()?, bid_vol5.as_slice()?, bid_vol6.as_slice()?,
        bid_vol7.as_slice()?, bid_vol8.as_slice()?, bid_vol9.as_slice()?,
        bid_vol10.as_slice()?,
    ];
    let ask_prc: [&[f64]; LEVEL_COUNT] = [
        ask_prc1.as_slice()?, ask_prc2.as_slice()?, ask_prc3.as_slice()?,
        ask_prc4.as_slice()?, ask_prc5.as_slice()?, ask_prc6.as_slice()?,
        ask_prc7.as_slice()?, ask_prc8.as_slice()?, ask_prc9.as_slice()?,
        ask_prc10.as_slice()?,
    ];
    let ask_vol: [&[f64]; LEVEL_COUNT] = [
        ask_vol1.as_slice()?, ask_vol2.as_slice()?, ask_vol3.as_slice()?,
        ask_vol4.as_slice()?, ask_vol5.as_slice()?, ask_vol6.as_slice()?,
        ask_vol7.as_slice()?, ask_vol8.as_slice()?, ask_vol9.as_slice()?,
        ask_vol10.as_slice()?,
    ];

    let n_trades = trade_exchtime.len();
    let n_snaps = snap_exchtime.len();

    if trade_price.len() != n_trades || trade_volume.len() != n_trades
        || trade_turnover.len() != n_trades || trade_flag.len() != n_trades
    {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "逐笔成交数据各列长度不一致",
        ));
    }

    for i in 0..LEVEL_COUNT {
        if bid_prc[i].len() != n_snaps || bid_vol[i].len() != n_snaps
            || ask_prc[i].len() != n_snaps || ask_vol[i].len() != n_snaps
        {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "盘口快照数据各列长度不一致",
            ));
        }
    }

    // 解析快照数据
    let mut snapshots = Vec::with_capacity(n_snaps);
    for i in 0..n_snaps {
        let mut bids = [BookLevel { price: 0.0, volume: 0.0 }; LEVEL_COUNT];
        let mut asks = [BookLevel { price: 0.0, volume: 0.0 }; LEVEL_COUNT];

        for j in 0..LEVEL_COUNT {
            bids[j] = BookLevel {
                price: bid_prc[j][i],
                volume: bid_vol[j][i],
            };
            asks[j] = BookLevel {
                price: ask_prc[j][i],
                volume: ask_vol[j][i],
            };
        }

        snapshots.push(Snapshot {
            timestamp: snap_exchtime[i],
            bids,
            asks,
        });
    }

    // 解析成交数据
    let mut trades = Vec::with_capacity(n_trades);
    for i in 0..n_trades {
        trades.push(Trade {
            timestamp: trade_exchtime[i],
            price: trade_price[i],
            volume: trade_volume[i],
            turnover: trade_turnover[i],
            flag: trade_flag[i],
        });
    }

    let detection_modes = vec!["horizontal", "vertical", "both"];
    let side_filters = vec!["bid", "ask", "both"];

    let dm_names = ["horizontal", "vertical", "both"];
    let sf_names = ["bid", "ask", "both"];

    let labeled_events = detect_ala_events_v3(
        &snapshots,
        &detection_modes,
        &side_filters,
        k1_horizontal,
        k2_vertical,
        window_size,
        decay_threshold,
    );

    let n_dm = detection_modes.len();
    let n_sf = side_filters.len();
    let n_combinations = n_dm * n_sf;

    // 提取时间戳数组用于二分查找
    let snap_timestamps: Vec<i64> = snapshots.iter().map(|s| s.timestamp).collect();
    let trade_timestamps: Vec<i64> = trades.iter().map(|t| t.timestamp).collect();

    let mut grouped_features: Vec<Vec<Vec<f64>>> = vec![Vec::new(); n_combinations];
    for le in &labeled_events {
        let group_idx = le.detection_mode_idx * n_sf + le.side_filter_idx;
        let features = compute_ala_features_v3(
            &le.event,
            &snapshots,
            &snap_timestamps,
            &trades,
            &trade_timestamps,
        );
        grouped_features[group_idx].push(features.to_vec());
    }

    let base_names = get_feature_names();
    let n_base_features = base_names.len();

    let mut result_arrays: Vec<Py<PyArray2<f64>>> = Vec::with_capacity(n_combinations);
    let mut result_names: Vec<Vec<String>> = Vec::with_capacity(n_combinations);

    for group_idx in 0..n_combinations {
        let dm_idx = group_idx / n_sf;
        let sf_idx = group_idx % n_sf;
        let dm_name = dm_names[dm_idx];
        let sf_name = sf_names[sf_idx];

        let mut feature_names = Vec::with_capacity(n_base_features);
        for base_name in &base_names {
            feature_names.push(format!("{}_{}_{}", dm_name, sf_name, base_name));
        }
        result_names.push(feature_names);

        let group_features = &grouped_features[group_idx];
        let n_events = group_features.len();
        let feature_array = if n_events > 0 {
            let mut result = Array2::<f64>::zeros((n_events, n_base_features));
            for (i, features) in group_features.iter().enumerate() {
                for (j, &val) in features.iter().enumerate() {
                    result[[i, j]] = if val.is_nan() || val.is_infinite() {
                        0.0
                    } else {
                        val
                    };
                }
            }
            result
        } else {
            Array2::<f64>::zeros((0, n_base_features))
        };
        result_arrays.push(feature_array.into_pyarray(py).to_owned());
    }

    Ok((result_arrays, result_names))
}
