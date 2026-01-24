//! 非对称大挂单（ALLO）微观结构特征计算模块
//!
//! 计算异常流动性聚集事件(ALA)的特征指标

use ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1};
use pyo3::prelude::*;

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
    bids: [BookLevel; LEVEL_COUNT], // bid1-bid10
    asks: [BookLevel; LEVEL_COUNT], // ask1-ask10
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

    fn bid_volume_excluding(&self, level: usize) -> f64 {
        self.bids
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != level)
            .map(|(_, b)| b.volume)
            .sum()
    }

    fn ask_volume_excluding(&self, level: usize) -> f64 {
        self.asks
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != level)
            .map(|(_, a)| a.volume)
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
    flag: i32, // 66=主买, 83=主卖
}

/// 异常流动性聚集事件
#[derive(Copy, Clone, Debug)]
struct ALAEvent {
    start_idx: usize,
    end_idx: usize,
    start_time: i64,
    end_time: i64,
    level: usize,         // 0-9 对应 1-10档
    is_bid: bool,         // true=买单, false=卖单
    price: f64,           // 大单价格
    initial_volume: f64,  // 初始挂单量
    peak_volume: f64,     // 峰值挂单量
}

/// ALA事件特征 - 17个基础特征 + 8个时间形态学特征
#[derive(Clone, Debug, Default)]
struct ALAFeatures {
    // 第一部分：巨石的物理属性
    m1_relative_prominence: f64,     // 相对凸度
    m3_flicker_frequency: f64,       // 闪烁频率

    // 第二部分：攻城战的流体力学
    m7_queue_loitering_duration: f64, // 队列滞留时长

    // 第三部分：友军的生态结构
    m8_frontrun_passive: f64,        // 抢跑强度-挂单版
    m9_frontrun_active: f64,         // 抢跑强度-主买版
    m10_ally_retreat_rate: f64,      // 同侧撤单率

    // 第四部分：群体行为的时间形态学 - 对手攻击单
    m11a_attack_skewness_opponent: f64,      // 攻击偏度-对手盘
    m12a_peak_latency_ratio_opponent: f64,   // 峰值延迟率-对手盘
    m13a_courage_acceleration_opponent: f64, // 勇气加速度-对手盘
    m14a_rhythm_entropy_opponent: f64,       // 节奏熵-对手盘
    // 第四部分：群体行为的时间形态学 - 同侧抢跑单
    m11b_attack_skewness_ally: f64,          // 攻击偏度-同侧
    m12b_peak_latency_ratio_ally: f64,       // 峰值延迟率-同侧
    m13b_courage_acceleration_ally: f64,     // 勇气加速度-同侧
    m14b_rhythm_entropy_ally: f64,           // 节奏熵-同侧

    // 第五部分：空间场论与距离效应
    m15_fox_tiger_index: f64,        // 狐假虎威指数
    m16_shadow_projection_ratio: f64, // 阴影投射比
    m17_gravitational_redshift: f64,  // 引力红移速率
    m19_shielding_thickness_ratio: f64,   // 垫单厚度比

    // 第六部分：命运与结局
    m20_oxygen_saturation: f64,      // 氧气饱和度
    m21_suffocation_integral: f64,   // 窒息深度积分
    m22_local_survivor_bias: f64,    // 幸存者偏差-邻域版
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

/// 带标签的ALA事件，用于tris模式区分不同参数组合
#[derive(Copy, Clone, Debug)]
struct LabeledALAEvent {
    event: ALAEvent,
    detection_mode_idx: usize, // 0=horizontal, 1=vertical, 2=both
    side_filter_idx: usize,    // 0=bid, 1=ask, 2=both
}

/// 检测ALA事件
/// detection_mode: "horizontal", "vertical", "both", 或 "tris"
/// side_filter: "bid", "ask", "both", 或 "tris"
fn detect_ala_events(
    snapshots: &[Snapshot],
    detection_mode: &str,
    side_filter: &str,
    k1_horizontal: f64, // 横向阈值
    k2_vertical: f64,   // 纵向阈值
    window_size: usize, // 纵向移动窗口大小
    decay_threshold: f64, // 衰减阈值（结束条件）
) -> Vec<ALAEvent> {
    let mut events = Vec::new();
    let n = snapshots.len();
    if n < window_size + 1 {
        return events;
    }

    let check_bid = side_filter == "bid" || side_filter == "both";
    let check_ask = side_filter == "ask" || side_filter == "both";

    // 预计算移动平均（纵向模式）
    let mut bid_moving_avg: Vec<[f64; LEVEL_COUNT]> = Vec::with_capacity(n);
    let mut ask_moving_avg: Vec<[f64; LEVEL_COUNT]> = Vec::with_capacity(n);

    for i in 0..n {
        let mut bid_avg = [0.0; LEVEL_COUNT];
        let mut ask_avg = [0.0; LEVEL_COUNT];

        if i >= window_size {
            for level in 0..LEVEL_COUNT {
                let sum_bid: f64 = (i - window_size..i)
                    .map(|j| snapshots[j].bids[level].volume)
                    .sum();
                let sum_ask: f64 = (i - window_size..i)
                    .map(|j| snapshots[j].asks[level].volume)
                    .sum();
                bid_avg[level] = sum_bid / window_size as f64;
                ask_avg[level] = sum_ask / window_size as f64;
            }
        }
        bid_moving_avg.push(bid_avg);
        ask_moving_avg.push(ask_avg);
    }

    // 事件追踪状态
    let mut in_event = [[false; LEVEL_COUNT]; 2]; // [bid/ask][level]
    let mut event_start = [[0usize; LEVEL_COUNT]; 2];
    let mut event_initial_vol = [[0.0f64; LEVEL_COUNT]; 2];
    let mut event_peak_vol = [[0.0f64; LEVEL_COUNT]; 2];
    let mut event_price = [[0.0f64; LEVEL_COUNT]; 2];

    for i in window_size..n {
        let snap = &snapshots[i];

        // 检查每个档位
        for level in 0..LEVEL_COUNT {
            // 检查买单侧
            if check_bid {
                let bid_vol = snap.bids[level].volume;
                let bid_triggered = match detection_mode {
                    "horizontal" => {
                        let other_vol = snap.bid_volume_excluding(level);
                        bid_vol > k1_horizontal * other_vol && other_vol > EPS
                    }
                    "vertical" => {
                        let avg = bid_moving_avg[i][level];
                        bid_vol > k2_vertical * avg && avg > EPS
                    }
                    _ => {
                        // both: 同时满足两个条件之一
                        let other_vol = snap.bid_volume_excluding(level);
                        let avg = bid_moving_avg[i][level];
                        (other_vol > EPS && bid_vol > k1_horizontal * other_vol)
                            || (avg > EPS && bid_vol > k2_vertical * avg)
                    }
                };

                if !in_event[0][level] && bid_triggered {
                    // 事件开始
                    in_event[0][level] = true;
                    event_start[0][level] = i;
                    event_initial_vol[0][level] = bid_vol;
                    event_peak_vol[0][level] = bid_vol;
                    event_price[0][level] = snap.bids[level].price;
                } else if in_event[0][level] {
                    // 更新峰值
                    if bid_vol > event_peak_vol[0][level] {
                        event_peak_vol[0][level] = bid_vol;
                    }
                    // 检查结束条件
                    let decayed = bid_vol < decay_threshold * event_initial_vol[0][level];
                    let price_out = (snap.bids[level].price - event_price[0][level]).abs() > EPS
                        && snap.bids.iter().all(|b| (b.price - event_price[0][level]).abs() > EPS);

                    if decayed || price_out {
                        // 事件结束
                        events.push(ALAEvent {
                            start_idx: event_start[0][level],
                            end_idx: i,
                            start_time: snapshots[event_start[0][level]].timestamp,
                            end_time: snap.timestamp,
                            level,
                            is_bid: true,
                            price: event_price[0][level],
                            initial_volume: event_initial_vol[0][level],
                            peak_volume: event_peak_vol[0][level],
                        });
                        in_event[0][level] = false;
                    }
                }
            }

            // 检查卖单侧
            if check_ask {
                let ask_vol = snap.asks[level].volume;
                let ask_triggered = match detection_mode {
                    "horizontal" => {
                        let other_vol = snap.ask_volume_excluding(level);
                        ask_vol > k1_horizontal * other_vol && other_vol > EPS
                    }
                    "vertical" => {
                        let avg = ask_moving_avg[i][level];
                        ask_vol > k2_vertical * avg && avg > EPS
                    }
                    _ => {
                        let other_vol = snap.ask_volume_excluding(level);
                        let avg = ask_moving_avg[i][level];
                        (other_vol > EPS && ask_vol > k1_horizontal * other_vol)
                            || (avg > EPS && ask_vol > k2_vertical * avg)
                    }
                };

                if !in_event[1][level] && ask_triggered {
                    in_event[1][level] = true;
                    event_start[1][level] = i;
                    event_initial_vol[1][level] = ask_vol;
                    event_peak_vol[1][level] = ask_vol;
                    event_price[1][level] = snap.asks[level].price;
                } else if in_event[1][level] {
                    if ask_vol > event_peak_vol[1][level] {
                        event_peak_vol[1][level] = ask_vol;
                    }
                    let decayed = ask_vol < decay_threshold * event_initial_vol[1][level];
                    let price_out = (snap.asks[level].price - event_price[1][level]).abs() > EPS
                        && snap.asks.iter().all(|a| (a.price - event_price[1][level]).abs() > EPS);

                    if decayed || price_out {
                        events.push(ALAEvent {
                            start_idx: event_start[1][level],
                            end_idx: i,
                            start_time: snapshots[event_start[1][level]].timestamp,
                            end_time: snap.timestamp,
                            level,
                            is_bid: false,
                            price: event_price[1][level],
                            initial_volume: event_initial_vol[1][level],
                            peak_volume: event_peak_vol[1][level],
                        });
                        in_event[1][level] = false;
                    }
                }
            }
        }
    }

    // 处理未结束的事件
    let last_idx = n - 1;
    for level in 0..LEVEL_COUNT {
        if check_bid && in_event[0][level] {
            events.push(ALAEvent {
                start_idx: event_start[0][level],
                end_idx: last_idx,
                start_time: snapshots[event_start[0][level]].timestamp,
                end_time: snapshots[last_idx].timestamp,
                level,
                is_bid: true,
                price: event_price[0][level],
                initial_volume: event_initial_vol[0][level],
                peak_volume: event_peak_vol[0][level],
            });
        }
        if check_ask && in_event[1][level] {
            events.push(ALAEvent {
                start_idx: event_start[1][level],
                end_idx: last_idx,
                start_time: snapshots[event_start[1][level]].timestamp,
                end_time: snapshots[last_idx].timestamp,
                level,
                is_bid: false,
                price: event_price[1][level],
                initial_volume: event_initial_vol[1][level],
                peak_volume: event_peak_vol[1][level],
            });
        }
    }

    events
}

/// 检测带标签的ALA事件（用于tris模式）
/// 一次性计算所有detection_mode和side_filter组合，复用移动平均等中间计算结果
fn detect_labeled_ala_events(
    snapshots: &[Snapshot],
    detection_modes: &[&str],  // 要计算的detection_mode列表
    side_filters: &[&str],     // 要计算的side_filter列表
    k1_horizontal: f64,
    k2_vertical: f64,
    window_size: usize,
    decay_threshold: f64,
) -> Vec<LabeledALAEvent> {
    let mut labeled_events = Vec::new();
    let n = snapshots.len();
    if n < window_size + 1 {
        return labeled_events;
    }

    // 预计算移动平均（纵向模式需要，只计算一次）
    let mut bid_moving_avg: Vec<[f64; LEVEL_COUNT]> = Vec::with_capacity(n);
    let mut ask_moving_avg: Vec<[f64; LEVEL_COUNT]> = Vec::with_capacity(n);

    for i in 0..n {
        let mut bid_avg = [0.0; LEVEL_COUNT];
        let mut ask_avg = [0.0; LEVEL_COUNT];

        if i >= window_size {
            for level in 0..LEVEL_COUNT {
                let sum_bid: f64 = (i - window_size..i)
                    .map(|j| snapshots[j].bids[level].volume)
                    .sum();
                let sum_ask: f64 = (i - window_size..i)
                    .map(|j| snapshots[j].asks[level].volume)
                    .sum();
                bid_avg[level] = sum_bid / window_size as f64;
                ask_avg[level] = sum_ask / window_size as f64;
            }
        }
        bid_moving_avg.push(bid_avg);
        ask_moving_avg.push(ask_avg);
    }

    // 预计算触发条件，避免tris组合下重复计算
    let mut bid_trigger_h = vec![[false; LEVEL_COUNT]; n];
    let mut bid_trigger_v = vec![[false; LEVEL_COUNT]; n];
    let mut ask_trigger_h = vec![[false; LEVEL_COUNT]; n];
    let mut ask_trigger_v = vec![[false; LEVEL_COUNT]; n];

    for i in window_size..n {
        let snap = &snapshots[i];
        for level in 0..LEVEL_COUNT {
            let bid_vol = snap.bids[level].volume;
            let bid_other_vol = snap.bid_volume_excluding(level);
            bid_trigger_h[i][level] = bid_other_vol > EPS && bid_vol > k1_horizontal * bid_other_vol;
            let bid_avg = bid_moving_avg[i][level];
            bid_trigger_v[i][level] = bid_avg > EPS && bid_vol > k2_vertical * bid_avg;

            let ask_vol = snap.asks[level].volume;
            let ask_other_vol = snap.ask_volume_excluding(level);
            ask_trigger_h[i][level] = ask_other_vol > EPS && ask_vol > k1_horizontal * ask_other_vol;
            let ask_avg = ask_moving_avg[i][level];
            ask_trigger_v[i][level] = ask_avg > EPS && ask_vol > k2_vertical * ask_avg;
        }
    }

    // 对每种detection_mode和side_filter组合进行检测
    for (dm_idx, &detection_mode) in detection_modes.iter().enumerate() {
        for (sf_idx, &side_filter) in side_filters.iter().enumerate() {
            let check_bid = side_filter == "bid" || side_filter == "both";
            let check_ask = side_filter == "ask" || side_filter == "both";

            // 事件追踪状态
            let mut in_event = [[false; LEVEL_COUNT]; 2];
            let mut event_start = [[0usize; LEVEL_COUNT]; 2];
            let mut event_initial_vol = [[0.0f64; LEVEL_COUNT]; 2];
            let mut event_peak_vol = [[0.0f64; LEVEL_COUNT]; 2];
            let mut event_price = [[0.0f64; LEVEL_COUNT]; 2];

            for i in window_size..n {
                let snap = &snapshots[i];

                for level in 0..LEVEL_COUNT {
                    // 检查买单侧
                    if check_bid {
                        let bid_vol = snap.bids[level].volume;
                        let bid_triggered = match detection_mode {
                            "horizontal" => bid_trigger_h[i][level],
                            "vertical" => bid_trigger_v[i][level],
                            _ => bid_trigger_h[i][level] || bid_trigger_v[i][level],
                        };

                        if !in_event[0][level] && bid_triggered {
                            in_event[0][level] = true;
                            event_start[0][level] = i;
                            event_initial_vol[0][level] = bid_vol;
                            event_peak_vol[0][level] = bid_vol;
                            event_price[0][level] = snap.bids[level].price;
                        } else if in_event[0][level] {
                            if bid_vol > event_peak_vol[0][level] {
                                event_peak_vol[0][level] = bid_vol;
                            }
                            let decayed = bid_vol < decay_threshold * event_initial_vol[0][level];
                            let price_out = (snap.bids[level].price - event_price[0][level]).abs() > EPS
                                && snap.bids.iter().all(|b| (b.price - event_price[0][level]).abs() > EPS);

                            if decayed || price_out {
                                labeled_events.push(LabeledALAEvent {
                                    event: ALAEvent {
                                        start_idx: event_start[0][level],
                                        end_idx: i,
                                        start_time: snapshots[event_start[0][level]].timestamp,
                                        end_time: snap.timestamp,
                                        level,
                                        is_bid: true,
                                        price: event_price[0][level],
                                        initial_volume: event_initial_vol[0][level],
                                        peak_volume: event_peak_vol[0][level],
                                    },
                                    detection_mode_idx: dm_idx,
                                    side_filter_idx: sf_idx,
                                });
                                in_event[0][level] = false;
                            }
                        }
                    }

                    // 检查卖单侧
                    if check_ask {
                        let ask_vol = snap.asks[level].volume;
                        let ask_triggered = match detection_mode {
                            "horizontal" => ask_trigger_h[i][level],
                            "vertical" => ask_trigger_v[i][level],
                            _ => ask_trigger_h[i][level] || ask_trigger_v[i][level],
                        };

                        if !in_event[1][level] && ask_triggered {
                            in_event[1][level] = true;
                            event_start[1][level] = i;
                            event_initial_vol[1][level] = ask_vol;
                            event_peak_vol[1][level] = ask_vol;
                            event_price[1][level] = snap.asks[level].price;
                        } else if in_event[1][level] {
                            if ask_vol > event_peak_vol[1][level] {
                                event_peak_vol[1][level] = ask_vol;
                            }
                            let decayed = ask_vol < decay_threshold * event_initial_vol[1][level];
                            let price_out = (snap.asks[level].price - event_price[1][level]).abs() > EPS
                                && snap.asks.iter().all(|a| (a.price - event_price[1][level]).abs() > EPS);

                            if decayed || price_out {
                                labeled_events.push(LabeledALAEvent {
                                    event: ALAEvent {
                                        start_idx: event_start[1][level],
                                        end_idx: i,
                                        start_time: snapshots[event_start[1][level]].timestamp,
                                        end_time: snap.timestamp,
                                        level,
                                        is_bid: false,
                                        price: event_price[1][level],
                                        initial_volume: event_initial_vol[1][level],
                                        peak_volume: event_peak_vol[1][level],
                                    },
                                    detection_mode_idx: dm_idx,
                                    side_filter_idx: sf_idx,
                                });
                                in_event[1][level] = false;
                            }
                        }
                    }
                }
            }

            // 处理未结束的事件
            let last_idx = n - 1;
            for level in 0..LEVEL_COUNT {
                if check_bid && in_event[0][level] {
                    labeled_events.push(LabeledALAEvent {
                        event: ALAEvent {
                            start_idx: event_start[0][level],
                            end_idx: last_idx,
                            start_time: snapshots[event_start[0][level]].timestamp,
                            end_time: snapshots[last_idx].timestamp,
                            level,
                            is_bid: true,
                            price: event_price[0][level],
                            initial_volume: event_initial_vol[0][level],
                            peak_volume: event_peak_vol[0][level],
                        },
                        detection_mode_idx: dm_idx,
                        side_filter_idx: sf_idx,
                    });
                }
                if check_ask && in_event[1][level] {
                    labeled_events.push(LabeledALAEvent {
                        event: ALAEvent {
                            start_idx: event_start[1][level],
                            end_idx: last_idx,
                            start_time: snapshots[event_start[1][level]].timestamp,
                            end_time: snapshots[last_idx].timestamp,
                            level,
                            is_bid: false,
                            price: event_price[1][level],
                            initial_volume: event_initial_vol[1][level],
                            peak_volume: event_peak_vol[1][level],
                        },
                        detection_mode_idx: dm_idx,
                        side_filter_idx: sf_idx,
                    });
                }
            }
        }
    }

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
    // 离散化：分成10个桶
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

/// 计算时间形态学特征（偏度、峰值延迟率、勇气加速度、节奏熵）
/// 返回 (skewness, peak_latency_ratio, courage_acceleration, rhythm_entropy)
fn compute_temporal_morphology(trades: &[&Trade], event_start_time: i64, event_end_time: i64) -> (f64, f64, f64, f64) {
    if trades.is_empty() {
        return (0.0, 0.0, 0.0, 0.0);
    }
    
    let duration = (event_end_time - event_start_time) as f64;
    if duration < EPS {
        return (0.0, 0.0, 0.0, 0.0);
    }

    // 1. 攻击偏度：成交量在时间轴上的分布偏度
    let trade_times: Vec<f64> = trades
        .iter()
        .map(|t| (t.timestamp - event_start_time) as f64)
        .collect();
    let attack_skewness = skewness(&trade_times);

    // 2. 峰值延迟率：达到成交最高潮所需的时间占比
    let mut max_vol = 0.0;
    let mut max_time = event_start_time;
    for t in trades {
        if t.volume > max_vol {
            max_vol = t.volume;
            max_time = t.timestamp;
        }
    }
    let peak_latency_ratio = (max_time - event_start_time) as f64 / duration;

    // 3. 勇气加速度：单笔成交体量与时间的相关性
    let trade_sizes: Vec<f64> = trades.iter().map(|t| t.volume).collect();
    let trade_seq: Vec<f64> = (0..trade_sizes.len()).map(|i| i as f64).collect();
    let courage_acceleration = correlation(&trade_seq, &trade_sizes);

    // 4. 节奏熵：订单到达时间间隔的熵值
    let mut intervals = Vec::new();
    for i in 1..trades.len() {
        let dt = (trades[i].timestamp - trades[i - 1].timestamp) as f64;
        intervals.push(dt);
    }
    let rhythm_entropy = entropy(&intervals);

    (attack_skewness, peak_latency_ratio, courage_acceleration, rhythm_entropy)
}

/// 计算单个ALA事件的特征
fn compute_ala_features(
    event: &ALAEvent,
    snapshots: &[Snapshot],
    trades: &[Trade],
    _close_price: f64,
) -> ALAFeatures {
    let mut features = ALAFeatures::default();

    let event_snaps = get_snapshots_in_range(snapshots, event.start_time, event.end_time);
    let event_trades = get_trades_in_range(trades, event.start_time, event.end_time);

    if event_snaps.is_empty() {
        return features;
    }

    let duration_seconds = (event.end_time - event.start_time) as f64 / 1e9;

    // ============ 第一部分：巨石的物理属性 =============

    // M1: 相对凸度 - 大单量与对侧前5档均值的比值
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

    // M3: 闪烁频率 - 检测频繁变化但总量稳定
    let volumes: Vec<f64> = event_snaps
        .iter()
        .map(|s| {
            if event.is_bid {
                s.bids[event.level].volume
            } else {
                s.asks[event.level].volume
            }
        })
        .collect();
    let vol_std = std_dev(&volumes);
    let vol_mean = volumes.iter().sum::<f64>() / volumes.len().max(1) as f64;

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
    // 低波动但高频率变化 = 幌骗
    features.m3_flicker_frequency = if vol_std < vol_mean * 0.2 {
        flicker_freq
    } else {
        flicker_freq * 0.5
    };

    // ============ 第二部分：攻城战的流体力学 ============

    // M7: 队列滞留时长 - 从首次出现到变为一档的时间
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
        None => duration_seconds, // 从未变为一档
    };

    // ============ 第三部分：友军的生态结构 ============

    // M8: 抢跑强度-挂单版
    let avg_better_levels: f64 = event_snaps
        .iter()
        .map(|s| {
            if event.is_bid {
                // 买单：更优=更高价格=前面的档位
                s.bids[0..event.level].iter().map(|b| b.volume).sum::<f64>()
            } else {
                // 卖单：更优=更低价格=前面的档位
                s.asks[0..event.level].iter().map(|a| a.volume).sum::<f64>()
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
                t.flag == 66 // 主买
            } else {
                t.flag == 83 // 主卖
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

    // ============ 第四部分：群体行为的时间形态学 ============
    // 分类主动成交订单：
    // - 对手攻击单（opponent）：与大单方向相反的主动单
    //   若大单是买单(is_bid=true)，对手攻击单是主卖(flag=83)
    //   若大单是卖单(is_bid=false)，对手攻击单是主买(flag=66)
    // - 同侧抢跑单（ally）：与大单方向相同的主动单
    //   若大单是买单(is_bid=true)，同侧抢跑单是主买(flag=66)
    //   若大单是卖单(is_bid=false)，同侧抢跑单是主卖(flag=83)

    let opponent_trades: Vec<&Trade> = event_trades
        .iter()
        .filter(|t| {
            if event.is_bid {
                t.flag == 83 // 大单买入，对手是主卖
            } else {
                t.flag == 66 // 大单卖出，对手是主买
            }
        })
        .cloned()
        .collect();

    let ally_trades: Vec<&Trade> = event_trades
        .iter()
        .filter(|t| {
            if event.is_bid {
                t.flag == 66 // 大单买入，同侧是主买
            } else {
                t.flag == 83 // 大单卖出，同侧是主卖
            }
        })
        .cloned()
        .collect();

    // 计算对手攻击单的时间形态学特征
    let (skew_opp, peak_opp, courage_opp, entropy_opp) = 
        compute_temporal_morphology(&opponent_trades, event.start_time, event.end_time);
    features.m11a_attack_skewness_opponent = skew_opp;
    features.m12a_peak_latency_ratio_opponent = peak_opp;
    features.m13a_courage_acceleration_opponent = courage_opp;
    features.m14a_rhythm_entropy_opponent = entropy_opp;

    // 计算同侧抢跑单的时间形态学特征
    let (skew_ally, peak_ally, courage_ally, entropy_ally) = 
        compute_temporal_morphology(&ally_trades, event.start_time, event.end_time);
    features.m11b_attack_skewness_ally = skew_ally;
    features.m12b_peak_latency_ratio_ally = peak_ally;
    features.m13b_courage_acceleration_ally = courage_ally;
    features.m14b_rhythm_entropy_ally = entropy_ally;

    // ============ 第五部分：空间场论与距离效应 ============

    // M15: 狐假虎威指数 - 前方档位挂单量的增加
    if event.level > 0 {
        // 计算事件期间前方档位的平均挂单量
        let during_avg: f64 = event_snaps
            .iter()
            .map(|s| {
                if event.is_bid {
                    s.bids[0..event.level].iter().map(|b| b.volume).sum::<f64>()
                        / event.level as f64
                } else {
                    s.asks[0..event.level].iter().map(|a| a.volume).sum::<f64>()
                        / event.level as f64
                }
            })
            .sum::<f64>() / event_snaps.len().max(1) as f64;

        // 计算历史平均（事件开始前的快照）
        let history_start = event.start_idx.saturating_sub(100);
        let history_snaps = &snapshots[history_start..event.start_idx];
        let history_avg: f64 = if !history_snaps.is_empty() {
            history_snaps
                .iter()
                .map(|s| {
                    if event.is_bid {
                        s.bids[0..event.level].iter().map(|b| b.volume).sum::<f64>()
                            / event.level as f64
                    } else {
                        s.asks[0..event.level].iter().map(|a| a.volume).sum::<f64>()
                            / event.level as f64
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
    } else {
        features.m15_fox_tiger_index = 1.0;
    }

    // M16: 阴影投射比 - 前方档位的成交频率变化
    if event.level > 0 {
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

        // 历史频率
        let history_start_time = event.start_time - 300_000_000_000i64; // 5分钟前
        let history_trades = get_trades_in_range(trades, history_start_time, event.start_time);
        let history_duration = 300.0; // 5分钟
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
    } else {
        features.m16_shadow_projection_ratio = 1.0;
    }

    // M17: 引力红移速率 - 对向一档价格向大单逼近的速度
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

    // M19: 垫单厚度比 - 前方挂单量与大单本身的比值
    let avg_shield: f64 = event_snaps
        .iter()
        .map(|s| {
            if event.is_bid {
                s.bids[0..event.level].iter().map(|b| b.volume).sum::<f64>()
            } else {
                s.asks[0..event.level].iter().map(|a| a.volume).sum::<f64>()
            }
        })
        .sum::<f64>() / event_snaps.len().max(1) as f64;
    features.m19_shielding_thickness_ratio = if event.peak_volume > EPS {
        avg_shield / event.peak_volume
    } else {
        0.0
    };

    // ============ 第六部分：命运与结局 ============

    // M20: 氧气饱和度 - 主动单进场后处于浮盈状态的时长比例
    let active_trades: Vec<&Trade> = event_trades
        .iter()
        .filter(|t| {
            if event.is_bid {
                t.flag == 66 // 主买
            } else {
                t.flag == 83 // 主卖
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
                    mid > trade.price // 买入后涨了
                } else {
                    mid < trade.price // 卖出后跌了
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

    // M21: 窒息深度积分 - 亏损深度×时间的积分
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

    // M22: 幸存者偏差-邻域版 - 期间VWAP与邻近时段VWAP的差
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

    let neighbor_start = event.start_time - 300_000_000_000i64; // 前5分钟
    let neighbor_end = event.end_time + 300_000_000_000i64; // 后5分钟
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



/// 计算非对称大挂单微观结构特征 - tris扩展版本
///
/// 此函数专门用于detection_mode="tris"且side_filter="tris"的场景，
/// 返回9种参数组合的原始事件特征，而非均值
///
/// 返回：
/// - (feature_arrays, feature_names_arrays)
/// - feature_arrays: 9个二维数组的列表，每个数组对应一种(detection_mode, side_filter)组合
/// - feature_names_arrays: 9个特征名列表的列表，每个列表对应一种组合的特征名
///
/// 9种组合的顺序：
/// 0. (horizontal, bid)
/// 1. (horizontal, ask)
/// 2. (horizontal, both)
/// 3. (vertical, bid)
/// 4. (vertical, ask)
/// 5. (vertical, both)
/// 6. (both, bid)
/// 7. (both, ask)
/// 8. (both, both)
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
pub fn compute_allo_microstructure_features_tris_expanded(
    py: Python,
    // 逐笔成交数据
    trade_exchtime: PyReadonlyArray1<i64>,
    trade_price: PyReadonlyArray1<f64>,
    trade_volume: PyReadonlyArray1<f64>,
    trade_turnover: PyReadonlyArray1<f64>,
    trade_flag: PyReadonlyArray1<i32>,
    // 快照数据
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
    // 参数
    k1_horizontal: f64,
    k2_vertical: f64,
    window_size: usize,
    decay_threshold: f64,
) -> PyResult<(Vec<Py<PyArray2<f64>>>, Vec<Vec<String>>)> {
    // 转换输入数据
    let trade_exchtime = trade_exchtime.as_slice()?;
    let trade_price = trade_price.as_slice()?;
    let trade_volume = trade_volume.as_slice()?;
    let trade_turnover = trade_turnover.as_slice()?;
    let trade_flag = trade_flag.as_slice()?;

    let snap_exchtime = snap_exchtime.as_slice()?;

    // 收集所有买卖档位数据
    let bid_prc: [&[f64]; LEVEL_COUNT] = [
        bid_prc1.as_slice()?,
        bid_prc2.as_slice()?,
        bid_prc3.as_slice()?,
        bid_prc4.as_slice()?,
        bid_prc5.as_slice()?,
        bid_prc6.as_slice()?,
        bid_prc7.as_slice()?,
        bid_prc8.as_slice()?,
        bid_prc9.as_slice()?,
        bid_prc10.as_slice()?,
    ];
    let bid_vol: [&[f64]; LEVEL_COUNT] = [
        bid_vol1.as_slice()?,
        bid_vol2.as_slice()?,
        bid_vol3.as_slice()?,
        bid_vol4.as_slice()?,
        bid_vol5.as_slice()?,
        bid_vol6.as_slice()?,
        bid_vol7.as_slice()?,
        bid_vol8.as_slice()?,
        bid_vol9.as_slice()?,
        bid_vol10.as_slice()?,
    ];
    let ask_prc: [&[f64]; LEVEL_COUNT] = [
        ask_prc1.as_slice()?,
        ask_prc2.as_slice()?,
        ask_prc3.as_slice()?,
        ask_prc4.as_slice()?,
        ask_prc5.as_slice()?,
        ask_prc6.as_slice()?,
        ask_prc7.as_slice()?,
        ask_prc8.as_slice()?,
        ask_prc9.as_slice()?,
        ask_prc10.as_slice()?,
    ];
    let ask_vol: [&[f64]; LEVEL_COUNT] = [
        ask_vol1.as_slice()?,
        ask_vol2.as_slice()?,
        ask_vol3.as_slice()?,
        ask_vol4.as_slice()?,
        ask_vol5.as_slice()?,
        ask_vol6.as_slice()?,
        ask_vol7.as_slice()?,
        ask_vol8.as_slice()?,
        ask_vol9.as_slice()?,
        ask_vol10.as_slice()?,
    ];

    // 验证数据长度
    let n_trades = trade_exchtime.len();
    let n_snaps = snap_exchtime.len();

    if trade_price.len() != n_trades
        || trade_volume.len() != n_trades
        || trade_turnover.len() != n_trades
        || trade_flag.len() != n_trades
    {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "逐笔成交数据各列长度不一致",
        ));
    }

    for i in 0..LEVEL_COUNT {
        if bid_prc[i].len() != n_snaps
            || bid_vol[i].len() != n_snaps
            || ask_prc[i].len() != n_snaps
            || ask_vol[i].len() != n_snaps
        {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "盘口快照数据各列长度不一致",
            ));
        }
    }

    // 解析数据
    let snapshots = parse_snapshots(snap_exchtime, &bid_prc, &bid_vol, &ask_prc, &ask_vol);
    let trades = parse_trades(trade_exchtime, trade_price, trade_volume, trade_turnover, trade_flag);

    // 计算收盘价（最后一笔成交价格）
    let close_price = trades.last().map(|t| t.price).unwrap_or(0.0);

    // 固定使用tris模式：9种组合
    let detection_modes = vec!["horizontal", "vertical", "both"];
    let side_filters = vec!["bid", "ask", "both"];

    let dm_names = ["horizontal", "vertical", "both"];
    let sf_names = ["bid", "ask", "both"];

    // 使用带标签的事件检测
    let labeled_events = detect_labeled_ala_events(
        &snapshots,
        &detection_modes,
        &side_filters,
        k1_horizontal,
        k2_vertical,
        window_size,
        decay_threshold,
    );

    // 按(detection_mode_idx, side_filter_idx)分组事件
    let n_dm = detection_modes.len();
    let n_sf = side_filters.len();
    let n_combinations = n_dm * n_sf;

    // 计算每个事件的特征并按组存储
    let mut grouped_features: Vec<Vec<Vec<f64>>> = vec![Vec::new(); n_combinations];
    for le in &labeled_events {
        let group_idx = le.detection_mode_idx * n_sf + le.side_filter_idx;
        let features = compute_ala_features(&le.event, &snapshots, &trades, close_price);
        grouped_features[group_idx].push(features.to_vec());
    }

    // 获取基础特征名
    let base_names = get_feature_names();
    let n_base_features = base_names.len();

    // 构建输出：9个数组和9个特征名列表
    let mut result_arrays: Vec<Py<PyArray2<f64>>> = Vec::with_capacity(n_combinations);
    let mut result_names: Vec<Vec<String>> = Vec::with_capacity(n_combinations);

    for group_idx in 0..n_combinations {
        let dm_idx = group_idx / n_sf;
        let sf_idx = group_idx % n_sf;
        let dm_name = dm_names[dm_idx];
        let sf_name = sf_names[sf_idx];

        // 构建特征名（带前缀）
        let mut feature_names = Vec::with_capacity(n_base_features);
        for base_name in &base_names {
            feature_names.push(format!("{}_{}_{}", dm_name, sf_name, base_name));
        }
        result_names.push(feature_names);

        // 构建特征数组
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

/// 计算非对称大挂单微观结构特征
///
/// 参数：
/// - trade_exchtime: 逐笔成交时间戳（纳秒）
/// - trade_price: 逐笔成交价格
/// - trade_volume: 逐笔成交量
/// - trade_turnover: 逐笔成交金额
/// - trade_flag: 逐笔成交标志（66=主买, 83=主卖）
/// - snap_exchtime: 快照时间戳（纳秒）
/// - bid_prc1-10: 买一到买十价格
/// - bid_vol1-10: 买一到买十挂单量
/// - ask_prc1-10: 卖一到卖十价格
/// - ask_vol1-10: 卖一到卖十挂单量
/// - detection_mode: "horizontal" 或 "vertical" 或 "both"（横向/纵向/两者兼顾）
/// - side_filter: "bid" 或 "ask" 或 "both"（买入侧/卖出侧/两者兼顾）
/// - k1_horizontal: 横向阈值（默认2.0）
/// - k2_vertical: 纵向阈值（默认5.0）
/// - window_size: 纵向移动窗口大小（默认100）
/// - decay_threshold: 衰减阈值（默认0.5）
///
/// 返回：
/// - (features_array, feature_names)
/// - features_array: 21列特征矩阵，每行对应一个ALA事件
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
    detection_mode = "both",
    side_filter = "both",
    k1_horizontal = 2.0,
    k2_vertical = 5.0,
    window_size = 100,
    decay_threshold = 0.5
))]
pub fn compute_allo_microstructure_features(
    py: Python,
    // 逐笔成交数据
    trade_exchtime: PyReadonlyArray1<i64>,
    trade_price: PyReadonlyArray1<f64>,
    trade_volume: PyReadonlyArray1<f64>,
    trade_turnover: PyReadonlyArray1<f64>,
    trade_flag: PyReadonlyArray1<i32>,
    // 快照数据
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
    // 参数
    detection_mode: &str,
    side_filter: &str,
    k1_horizontal: f64,
    k2_vertical: f64,
    window_size: usize,
    decay_threshold: f64,
) -> PyResult<(Py<PyArray2<f64>>, Vec<String>)> {
    // 转换输入数据
    let trade_exchtime = trade_exchtime.as_slice()?;
    let trade_price = trade_price.as_slice()?;
    let trade_volume = trade_volume.as_slice()?;
    let trade_turnover = trade_turnover.as_slice()?;
    let trade_flag = trade_flag.as_slice()?;

    let snap_exchtime = snap_exchtime.as_slice()?;

    // 收集所有买卖档位数据
    let bid_prc: [&[f64]; LEVEL_COUNT] = [
        bid_prc1.as_slice()?,
        bid_prc2.as_slice()?,
        bid_prc3.as_slice()?,
        bid_prc4.as_slice()?,
        bid_prc5.as_slice()?,
        bid_prc6.as_slice()?,
        bid_prc7.as_slice()?,
        bid_prc8.as_slice()?,
        bid_prc9.as_slice()?,
        bid_prc10.as_slice()?,
    ];
    let bid_vol: [&[f64]; LEVEL_COUNT] = [
        bid_vol1.as_slice()?,
        bid_vol2.as_slice()?,
        bid_vol3.as_slice()?,
        bid_vol4.as_slice()?,
        bid_vol5.as_slice()?,
        bid_vol6.as_slice()?,
        bid_vol7.as_slice()?,
        bid_vol8.as_slice()?,
        bid_vol9.as_slice()?,
        bid_vol10.as_slice()?,
    ];
    let ask_prc: [&[f64]; LEVEL_COUNT] = [
        ask_prc1.as_slice()?,
        ask_prc2.as_slice()?,
        ask_prc3.as_slice()?,
        ask_prc4.as_slice()?,
        ask_prc5.as_slice()?,
        ask_prc6.as_slice()?,
        ask_prc7.as_slice()?,
        ask_prc8.as_slice()?,
        ask_prc9.as_slice()?,
        ask_prc10.as_slice()?,
    ];
    let ask_vol: [&[f64]; LEVEL_COUNT] = [
        ask_vol1.as_slice()?,
        ask_vol2.as_slice()?,
        ask_vol3.as_slice()?,
        ask_vol4.as_slice()?,
        ask_vol5.as_slice()?,
        ask_vol6.as_slice()?,
        ask_vol7.as_slice()?,
        ask_vol8.as_slice()?,
        ask_vol9.as_slice()?,
        ask_vol10.as_slice()?,
    ];

    // 验证数据长度
    let n_trades = trade_exchtime.len();
    let n_snaps = snap_exchtime.len();

    if trade_price.len() != n_trades
        || trade_volume.len() != n_trades
        || trade_turnover.len() != n_trades
        || trade_flag.len() != n_trades
    {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "逐笔成交数据各列长度不一致",
        ));
    }

    for i in 0..LEVEL_COUNT {
        if bid_prc[i].len() != n_snaps
            || bid_vol[i].len() != n_snaps
            || ask_prc[i].len() != n_snaps
            || ask_vol[i].len() != n_snaps
        {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "盘口快照数据各列长度不一致",
            ));
        }
    }

    // 解析数据
    let snapshots = parse_snapshots(snap_exchtime, &bid_prc, &bid_vol, &ask_prc, &ask_vol);
    let trades = parse_trades(trade_exchtime, trade_price, trade_volume, trade_turnover, trade_flag);

    // 计算收盘价（最后一笔成交价格）
    let close_price = trades.last().map(|t| t.price).unwrap_or(0.0);

    // 确定需要计算的detection_mode和side_filter组合
    let detection_modes: Vec<&str> = if detection_mode == "tris" {
        vec!["horizontal", "vertical", "both"]
    } else {
        vec![detection_mode]
    };
    let side_filters: Vec<&str> = if side_filter == "tris" {
        vec!["bid", "ask", "both"]
    } else {
        vec![side_filter]
    };

    let dm_names = ["horizontal", "vertical", "both"];
    let sf_names = ["bid", "ask", "both"];

    // 判断是否使用tris模式
    let use_tris_mode = detection_mode == "tris" || side_filter == "tris";

    if use_tris_mode {
        // tris模式：使用带标签的事件检测，复用移动平均计算
        let labeled_events = detect_labeled_ala_events(
            &snapshots,
            &detection_modes,
            &side_filters,
            k1_horizontal,
            k2_vertical,
            window_size,
            decay_threshold,
        );

        // 按(detection_mode_idx, side_filter_idx)分组事件
        let n_dm = detection_modes.len();
        let n_sf = side_filters.len();
        let n_combinations = n_dm * n_sf;

        // 计算每个事件的特征并按组存储
        let mut grouped_features: Vec<Vec<Vec<f64>>> = vec![Vec::new(); n_combinations];
        for le in &labeled_events {
            let group_idx = le.detection_mode_idx * n_sf + le.side_filter_idx;
            let features = compute_ala_features(&le.event, &snapshots, &trades, close_price);
            grouped_features[group_idx].push(features.to_vec());
        }

        // 计算每组的均值特征
        let base_names = get_feature_names();
        let n_base_features = base_names.len();
        let total_features = n_base_features * n_combinations;

        // 构建特征名（带前缀）
        let mut feature_names = Vec::with_capacity(total_features);
        for dm_idx in 0..n_dm {
            for sf_idx in 0..n_sf {
                let dm_name = dm_names[dm_idx];
                let sf_name = sf_names[sf_idx];
                for base_name in &base_names {
                    feature_names.push(format!("{}_{}_{}",dm_name, sf_name, base_name));
                }
            }
        }

        // 计算每组的均值特征（输出1行）
        let mut result_row = vec![0.0f64; total_features];
        for (group_idx, group_features) in grouped_features.iter().enumerate() {
            if !group_features.is_empty() {
                let n_events = group_features.len() as f64;
                for (feat_idx, _) in base_names.iter().enumerate() {
                    let sum: f64 = group_features.iter().map(|f| f[feat_idx]).sum();
                    let col_idx = group_idx * n_base_features + feat_idx;
                    let val = sum / n_events;
                    result_row[col_idx] = if val.is_nan() || val.is_infinite() { 0.0 } else { val };
                }
            }
        }

        let result_array = Array2::from_shape_vec((1, total_features), result_row)
            .unwrap_or_else(|_| Array2::<f64>::zeros((1, total_features)));

        Ok((
            result_array.into_pyarray(py).to_owned(),
            feature_names,
        ))
    } else {
        // 非tris模式：使用原有逻辑
        let events = detect_ala_events(
            &snapshots,
            detection_mode,
            side_filter,
            k1_horizontal,
            k2_vertical,
            window_size,
            decay_threshold,
        );

        // 计算每个事件的特征
        let mut all_features: Vec<Vec<f64>> = Vec::with_capacity(events.len());
        for event in &events {
            let features = compute_ala_features(event, &snapshots, &trades, close_price);
            all_features.push(features.to_vec());
        }

        // 构建输出
        let n_events = all_features.len();
        let n_features = 21; // 21个特征

        let result_array = if n_events > 0 {
            let mut result = Array2::<f64>::zeros((n_events, n_features));
            for (i, features) in all_features.iter().enumerate() {
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
            Array2::<f64>::zeros((0, n_features))
        };

        let feature_names = get_feature_names();

        Ok((
            result_array.into_pyarray(py).to_owned(),
            feature_names,
        ))
    }
}
