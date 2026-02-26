//! 非对称大挂单（ALLO）微观结构特征计算模块 v2
//!
//! v2版本：全程价格锚定
//! - 纵向触发：计算某价格的近期平均挂单量（而非某档位）
//! - 事件追踪：按价格搜索，而非按档位
//! - 结束条件：价格消失或挂单量衰减

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
    initial_level: usize,  // 触发时的档位（仅用于记录）
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

/// 解析盘口快照数据
fn parse_snapshots(
    exchtime: &[i64],
    bid_prc: &[&[f64]; LEVEL_COUNT],
    bid_vol: &[&[f64]; LEVEL_COUNT],
    ask_prc: &[&[f64]; LEVEL_COUNT],
    ask_vol: &[&[f64]; LEVEL_COUNT],
) -> Vec<Snapshot> {
    let n = exchtime.len();
    let mut snapshots = Vec::with_capacity(n);

    for i in 0..n {
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
            timestamp: exchtime[i],
            bids,
            asks,
        });
    }

    snapshots
}

/// 解析逐笔成交数据
fn parse_trades(
    exchtime: &[i64],
    price: &[f64],
    volume: &[f64],
    turnover: &[f64],
    flag: &[i32],
) -> Vec<Trade> {
    let n = exchtime.len();
    let mut trades = Vec::with_capacity(n);

    for i in 0..n {
        trades.push(Trade {
            timestamp: exchtime[i],
            price: price[i],
            volume: volume[i],
            turnover: turnover[i],
            flag: flag[i],
        });
    }

    trades
}

/// 构建价格历史挂单量映射（用于纵向触发计算）
/// 返回: HashMap<价格, Vec<(快照索引, 挂单量)>>
fn build_price_history_map(
    snapshots: &[Snapshot],
    is_bid: bool,
) -> HashMap<i64, Vec<(usize, f64)>> {
    let mut price_history: HashMap<i64, Vec<(usize, f64)>> = HashMap::new();
    
    // 将价格乘以10000转换为整数作为key，避免浮点精度问题
    let price_to_key = |p: f64| -> i64 { (p * 10000.0).round() as i64 };
    
    for (idx, snap) in snapshots.iter().enumerate() {
        let levels = if is_bid { &snap.bids } else { &snap.asks };
        for level in levels.iter() {
            if level.volume > EPS {
                let key = price_to_key(level.price);
                price_history.entry(key)
                    .or_insert_with(Vec::new)
                    .push((idx, level.volume));
            }
        }
    }
    
    price_history
}

/// 计算某价格在某快照索引之前的window_size个快照中的平均挂单量
fn get_price_moving_avg(
    price_key: i64,
    current_idx: usize,
    window_size: usize,
    price_history: &HashMap<i64, Vec<(usize, f64)>>,
) -> f64 {
    if let Some(history) = price_history.get(&price_key) {
        let mut sum = 0.0;
        let mut count = 0;
        
        // 从历史记录中找出在 (current_idx - window_size, current_idx) 范围内的记录
        for &(idx, vol) in history.iter() {
            if idx >= current_idx {
                break;
            }
            if idx >= current_idx.saturating_sub(window_size) {
                sum += vol;
                count += 1;
            }
        }
        
        if count > 0 {
            sum / count as f64
        } else {
            0.0
        }
    } else {
        0.0
    }
}

/// 带标签的ALA事件
#[derive(Copy, Clone, Debug)]
struct LabeledALAEventV2 {
    event: ALAEventV2,
    detection_mode_idx: usize,
    side_filter_idx: usize,
}

/// 检测ALA事件 v2（价格锚定版本）
fn detect_ala_events_v2(
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

    // 预构建价格历史映射（买侧和卖侧分别构建）
    let bid_price_history = build_price_history_map(snapshots, true);
    let ask_price_history = build_price_history_map(snapshots, false);

    // 对每种组合进行检测
    for (dm_idx, &detection_mode) in detection_modes.iter().enumerate() {
        for (sf_idx, &side_filter) in side_filters.iter().enumerate() {
            let check_bid = side_filter == "bid" || side_filter == "both";
            let check_ask = side_filter == "ask" || side_filter == "both";

            // 事件追踪状态：按价格追踪
            // key = 价格的整数表示，value = (start_idx, initial_vol, peak_vol)
            let mut in_event_bid: HashMap<i64, (usize, f64, f64)> = HashMap::new();
            let mut in_event_ask: HashMap<i64, (usize, f64, f64)> = HashMap::new();

            for i in window_size..n {
                let snap = &snapshots[i];

                // 检查买侧
                if check_bid {
                    for (level, bid) in snap.bids.iter().enumerate() {
                        let price_key = price_to_key(bid.price);
                        
                        // 如果已经在事件中，更新追踪
                        if in_event_bid.contains_key(&price_key) {
                            let (start_idx, initial_vol, peak_vol) = *in_event_bid.get(&price_key).unwrap();
                            let new_peak = peak_vol.max(bid.volume);
                            in_event_bid.insert(price_key, (start_idx, initial_vol, new_peak));
                            
                            // 检查结束条件
                            if bid.volume < decay_threshold * initial_vol {
                                // 事件结束 - 衰减
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

                        // 检查是否触发新事件
                        let h_trigger = {
                            // 横向：当前挂单量 > k1 × 同侧其他档位挂单量之和
                            let other_vol: f64 = snap.bids.iter()
                                .enumerate()
                                .filter(|(l, _)| *l != level)
                                .map(|(_, b)| b.volume)
                                .sum();
                            other_vol > EPS && bid.volume > k1_horizontal * other_vol
                        };

                        let v_trigger = {
                            // 纵向：当前挂单量 > k2 × 该价格近期平均挂单量
                            let avg = get_price_moving_avg(price_key, i, window_size, &bid_price_history);
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
                    let mut ended_prices = Vec::new();
                    for (&price_key, &(start_idx, initial_vol, peak_vol)) in in_event_bid.iter() {
                        let target_price = price_key as f64 / 10000.0;
                        if snap.find_bid_volume_by_price(target_price).is_none() {
                            ended_prices.push(price_key);
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
                    for price_key in ended_prices {
                        in_event_bid.remove(&price_key);
                    }
                }

                // 检查卖侧（类似逻辑）
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
                            let other_vol: f64 = snap.asks.iter()
                                .enumerate()
                                .filter(|(l, _)| *l != level)
                                .map(|(_, a)| a.volume)
                                .sum();
                            other_vol > EPS && ask.volume > k1_horizontal * other_vol
                        };

                        let v_trigger = {
                            let avg = get_price_moving_avg(price_key, i, window_size, &ask_price_history);
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
                    let mut ended_prices = Vec::new();
                    for (&price_key, &(start_idx, initial_vol, peak_vol)) in in_event_ask.iter() {
                        let target_price = price_key as f64 / 10000.0;
                        if snap.find_ask_volume_by_price(target_price).is_none() {
                            ended_prices.push(price_key);
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
                    for price_key in ended_prices {
                        in_event_ask.remove(&price_key);
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

/// 获取时间范围内的成交记录
fn get_trades_in_range(trades: &[Trade], start_time: i64, end_time: i64) -> Vec<&Trade> {
    trades
        .iter()
        .filter(|t| t.timestamp >= start_time && t.timestamp <= end_time)
        .collect()
}

/// 获取时间范围内的快照
fn get_snapshots_in_range(snapshots: &[Snapshot], start_time: i64, end_time: i64) -> Vec<&Snapshot> {
    snapshots
        .iter()
        .filter(|s| s.timestamp >= start_time && s.timestamp <= end_time)
        .collect()
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
    let mut counts = vec![0usize; num_bins];
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
fn compute_temporal_morphology(trades: &[&Trade], event_start_time: i64, event_end_time: i64) -> (f64, f64, f64, f64) {
    if trades.is_empty() {
        return (0.0, 0.0, 0.0, 0.0);
    }
    
    let duration = (event_end_time - event_start_time) as f64;
    if duration < EPS {
        return (0.0, 0.0, 0.0, 0.0);
    }

    let trade_times: Vec<f64> = trades
        .iter()
        .map(|t| (t.timestamp - event_start_time) as f64)
        .collect();
    let attack_skewness = skewness(&trade_times);

    let mut max_vol = 0.0;
    let mut max_time = event_start_time;
    for t in trades {
        if t.volume > max_vol {
            max_vol = t.volume;
            max_time = t.timestamp;
        }
    }
    let peak_latency_ratio = (max_time - event_start_time) as f64 / duration;

    let trade_sizes: Vec<f64> = trades.iter().map(|t| t.volume).collect();
    let trade_seq: Vec<f64> = (0..trade_sizes.len()).map(|i| i as f64).collect();
    let courage_acceleration = correlation(&trade_seq, &trade_sizes);

    let mut intervals = Vec::new();
    for i in 1..trades.len() {
        let dt = (trades[i].timestamp - trades[i - 1].timestamp) as f64;
        intervals.push(dt);
    }
    let rhythm_entropy = entropy(&intervals);

    (attack_skewness, peak_latency_ratio, courage_acceleration, rhythm_entropy)
}

/// 计算单个ALA事件的特征
fn compute_ala_features_v2(
    event: &ALAEventV2,
    snapshots: &[Snapshot],
    trades: &[Trade],
) -> ALAFeatures {
    let mut features = ALAFeatures::default();

    let event_snaps = get_snapshots_in_range(snapshots, event.start_time, event.end_time);
    let event_trades = get_trades_in_range(trades, event.start_time, event.end_time);

    if event_snaps.is_empty() {
        return features;
    }

    let duration_seconds = (event.end_time - event.start_time) as f64 / 1e9;

    // ============ 第一部分：挂单量特征 ============

    // M1: 相对凸度
    let avg_opposite: f64 = if event.is_bid {
        event_snaps.iter()
            .map(|s| s.asks[0..5].iter().map(|a| a.volume).sum::<f64>() / 5.0)
            .sum::<f64>() / event_snaps.len() as f64
    } else {
        event_snaps.iter()
            .map(|s| s.bids[0..5].iter().map(|b| b.volume).sum::<f64>() / 5.0)
            .sum::<f64>() / event_snaps.len() as f64
    };
    features.m1_relative_prominence = if avg_opposite > EPS {
        event.peak_volume / avg_opposite
    } else {
        0.0
    };

    // M3: 闪烁频率
    let mut volumes = Vec::new();
    for snap in &event_snaps {
        if event.is_bid {
            if let Some((vol, _)) = snap.find_bid_volume_by_price(event.price) {
                volumes.push(vol);
            }
        } else {
            if let Some((vol, _)) = snap.find_ask_volume_by_price(event.price) {
                volumes.push(vol);
            }
        }
    }
    
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

    // ============ 第二部分：队列位置特征 ============

    // M7: 队列滞留时长 - 从首次出现到变为一档
    let became_level1_time = event_snaps
        .iter()
        .find(|s| {
            if event.is_bid {
                (s.bids[0].price - event.price).abs() < EPS
            } else {
                (s.asks[0].price - event.price).abs() < EPS
            }
        })
        .map(|s| s.timestamp);
    features.m7_queue_loitering_duration = match became_level1_time {
        Some(t) => (t - event.start_time) as f64 / 1e9,
        None => duration_seconds,
    };

    // ============ 第三部分：同侧市场结构特征 ============

    // M8: 抢跑强度-挂单版 - 前方档位挂单量与大单的比值
    let avg_better_levels: f64 = event_snaps
        .iter()
        .map(|s| {
            if event.is_bid {
                s.bid_volume_before_price(event.price)
            } else {
                s.ask_volume_before_price(event.price)
            }
        })
        .sum::<f64>() / event_snaps.len().max(1) as f64;
    features.m8_frontrun_passive = if event.peak_volume > EPS {
        avg_better_levels / event.peak_volume
    } else {
        0.0
    };

    // M9: 抢跑强度-主买版
    features.m9_frontrun_active = event_trades
        .iter()
        .filter(|t| {
            if event.is_bid {
                t.flag == 66
            } else {
                t.flag == 83
            }
        })
        .map(|t| t.volume)
        .sum();

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

    let opponent_trades: Vec<&Trade> = event_trades
        .iter()
        .filter(|t| {
            if event.is_bid {
                t.flag == 83
            } else {
                t.flag == 66
            }
        })
        .cloned()
        .collect();

    let ally_trades: Vec<&Trade> = event_trades
        .iter()
        .filter(|t| {
            if event.is_bid {
                t.flag == 66
            } else {
                t.flag == 83
            }
        })
        .cloned()
        .collect();

    let (skew_opp, peak_opp, courage_opp, entropy_opp) = 
        compute_temporal_morphology(&opponent_trades, event.start_time, event.end_time);
    features.m11a_attack_skewness_opponent = skew_opp;
    features.m12a_peak_latency_ratio_opponent = peak_opp;
    features.m13a_courage_acceleration_opponent = courage_opp;
    features.m14a_rhythm_entropy_opponent = entropy_opp;

    let (skew_ally, peak_ally, courage_ally, entropy_ally) = 
        compute_temporal_morphology(&ally_trades, event.start_time, event.end_time);
    features.m11b_attack_skewness_ally = skew_ally;
    features.m12b_peak_latency_ratio_ally = peak_ally;
    features.m13b_courage_acceleration_ally = courage_ally;
    features.m14b_rhythm_entropy_ally = entropy_ally;

    // ============ 第五部分：空间场论特征 ============

    // M15: 狐假虎威指数
    let during_avg: f64 = event_snaps
        .iter()
        .map(|s| {
            if event.is_bid {
                s.bid_volume_before_price(event.price)
            } else {
                s.ask_volume_before_price(event.price)
            }
        })
        .sum::<f64>() / event_snaps.len().max(1) as f64;

    let history_start = event.start_idx.saturating_sub(100);
    let history_snaps = &snapshots[history_start..event.start_idx];
    let history_avg: f64 = if !history_snaps.is_empty() {
        history_snaps
            .iter()
            .map(|s| {
                if event.is_bid {
                    s.bid_volume_before_price(event.price)
                } else {
                    s.ask_volume_before_price(event.price)
                }
            })
            .sum::<f64>() / history_snaps.len() as f64
    } else {
        during_avg
    };
    features.m15_fox_tiger_index = if history_avg > EPS {
        during_avg / history_avg
    } else {
        1.0
    };

    // M16: 阴影投射比
    let front_trades_during: usize = event_trades
        .iter()
        .filter(|t| {
            if event.is_bid {
                t.price > event.price
            } else {
                t.price < event.price
            }
        })
        .count();
    let freq_during = if duration_seconds > EPS {
        front_trades_during as f64 / duration_seconds
    } else {
        0.0
    };

    let history_start_time = event.start_time - 300_000_000_000i64;
    let history_trades = get_trades_in_range(trades, history_start_time, event.start_time);
    let history_duration = 300.0;
    let front_trades_history: usize = history_trades
        .iter()
        .filter(|t| {
            if event.is_bid {
                t.price > event.price
            } else {
                t.price < event.price
            }
        })
        .count();
    let freq_history = front_trades_history as f64 / history_duration;

    features.m16_shadow_projection_ratio = if freq_history > EPS {
        freq_during / freq_history
    } else {
        1.0
    };

    // M17: 引力红移速率
    let approach_speeds: Vec<f64> = event_snaps
        .windows(2)
        .map(|w| {
            let gap_before = if event.is_bid {
                w[0].asks[0].price - event.price
            } else {
                event.price - w[0].bids[0].price
            };
            let gap_after = if event.is_bid {
                w[1].asks[0].price - event.price
            } else {
                event.price - w[1].bids[0].price
            };
            let dt = (w[1].timestamp - w[0].timestamp) as f64 / 1e9;
            if dt > EPS {
                (gap_before - gap_after) / dt
            } else {
                0.0
            }
        })
        .collect();
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

    let active_trades: Vec<&Trade> = event_trades
        .iter()
        .filter(|t| {
            if event.is_bid {
                t.flag == 66
            } else {
                t.flag == 83
            }
        })
        .cloned()
        .collect();

    if !active_trades.is_empty() {
        let mut profit_time = 0.0;
        let mut total_time = 0.0;
        for trade in &active_trades {
            let remaining_snaps = snapshots
                .iter()
                .filter(|s| s.timestamp > trade.timestamp)
                .collect::<Vec<_>>();
            for i in 0..remaining_snaps.len() {
                let dt = if i + 1 < remaining_snaps.len() {
                    (remaining_snaps[i + 1].timestamp - remaining_snaps[i].timestamp) as f64
                } else {
                    1e9
                };
                total_time += dt;
                let mid = remaining_snaps[i].mid_price();
                let in_profit = if event.is_bid {
                    mid > trade.price
                } else {
                    mid < trade.price
                };
                if in_profit {
                    profit_time += dt;
                }
            }
        }
        features.m20_oxygen_saturation = if total_time > EPS {
            profit_time / total_time
        } else {
            0.5
        };
    } else {
        features.m20_oxygen_saturation = 0.5;
    }

    if !active_trades.is_empty() {
        let mut suffocation = 0.0;
        for trade in &active_trades {
            let remaining_snaps = snapshots
                .iter()
                .filter(|s| s.timestamp > trade.timestamp)
                .collect::<Vec<_>>();
            for i in 0..remaining_snaps.len() {
                let dt = if i + 1 < remaining_snaps.len() {
                    (remaining_snaps[i + 1].timestamp - remaining_snaps[i].timestamp) as f64 / 1e9
                } else {
                    0.0
                };
                let mid = remaining_snaps[i].mid_price();
                let loss = if event.is_bid {
                    (trade.price - mid).max(0.0) / trade.price
                } else {
                    (mid - trade.price).max(0.0) / trade.price
                };
                suffocation += loss * dt;
            }
        }
        features.m21_suffocation_integral = suffocation;
    }

    let vwap_during = if !event_trades.is_empty() {
        let total_turnover: f64 = event_trades.iter().map(|t| t.turnover).sum();
        let total_vol: f64 = event_trades.iter().map(|t| t.volume).sum();
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
    let neighbor_trades: Vec<&Trade> = trades
        .iter()
        .filter(|t| {
            (t.timestamp >= neighbor_start && t.timestamp < event.start_time)
                || (t.timestamp > event.end_time && t.timestamp <= neighbor_end)
        })
        .collect();
    let vwap_neighbor = if !neighbor_trades.is_empty() {
        let total_turnover: f64 = neighbor_trades.iter().map(|t| t.turnover).sum();
        let total_vol: f64 = neighbor_trades.iter().map(|t| t.volume).sum();
        if total_vol > EPS {
            total_turnover / total_vol
        } else {
            0.0
        }
    } else {
        vwap_during
    };
    features.m22_local_survivor_bias = vwap_during - vwap_neighbor;

    features
}


/// 计算非对称大挂单微观结构特征 v2 - 价格锚定版本
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
pub fn compute_allo_microstructure_features_tris_expanded_v2(
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

    let snapshots = parse_snapshots(snap_exchtime, &bid_prc, &bid_vol, &ask_prc, &ask_vol);
    let trades = parse_trades(trade_exchtime, trade_price, trade_volume, trade_turnover, trade_flag);

    let detection_modes = vec!["horizontal", "vertical", "both"];
    let side_filters = vec!["bid", "ask", "both"];

    let dm_names = ["horizontal", "vertical", "both"];
    let sf_names = ["bid", "ask", "both"];

    let labeled_events = detect_ala_events_v2(
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

    let mut grouped_features: Vec<Vec<Vec<f64>>> = vec![Vec::new(); n_combinations];
    for le in &labeled_events {
        let group_idx = le.detection_mode_idx * n_sf + le.side_filter_idx;
        let features = compute_ala_features_v2(&le.event, &snapshots, &trades);
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
