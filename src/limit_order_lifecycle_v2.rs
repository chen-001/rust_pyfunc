use ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use std::sync::Arc;

const OUTPUT_COLS: usize = 15;
const LEVELS: usize = 10;

/// 计算均值
fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

/// 计算标准差
fn std_dev(values: &[f64], mean_val: f64) -> f64 {
    if values.len() < 2 {
        return f64::NAN;
    }
    let var = values
        .iter()
        .map(|v| {
            let diff = v - mean_val;
            diff * diff
        })
        .sum::<f64>()
        / values.len() as f64;
    var.sqrt()
}

/// 计算偏度
fn skewness(values: &[f64], mean_val: f64, std_val: f64) -> f64 {
    if values.len() < 3 || !std_val.is_finite() || std_val == 0.0 {
        return f64::NAN;
    }
    let m3 = values
        .iter()
        .map(|v| {
            let diff = v - mean_val;
            diff * diff * diff
        })
        .sum::<f64>()
        / values.len() as f64;
    m3 / (std_val * std_val * std_val)
}

/// 计算相关系数
fn correlation(x: &[f64], y: &[f64]) -> f64 {
    if x.len() < 2 || x.len() != y.len() {
        return f64::NAN;
    }
    let mean_x = mean(x);
    let mean_y = mean(y);
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
    if var_x == 0.0 || var_y == 0.0 {
        return f64::NAN;
    }
    cov / (var_x.sqrt() * var_y.sqrt())
}

/// 计算滞后1阶自相关
fn autocorr_lag1(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return f64::NAN;
    }
    let x = &values[..values.len() - 1];
    let y = &values[1..];
    correlation(x, y)
}

/// 计算趋势相关性（与序列[1,2,...,N]的相关性）
fn trend_corr(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return f64::NAN;
    }
    let idx: Vec<f64> = (1..=values.len()).map(|i| i as f64).collect();
    correlation(values, &idx)
}

/// 限价单生命周期重建 v2
/// 
/// 功能：通过匹配高频逐笔成交数据与盘口快照数据，使用订单ID作为追踪器，
/// 重建限价单生命周期指标
/// 
/// 输入：
/// - ticks_array: 逐笔成交数据 (N_ticks x 7)
///   - 列0: exchtime (时间戳)
///   - 列1: price (成交价格)
///   - 列2: volume (成交量)
///   - 列3: turnover (成交金额)
///   - 列4: flag (交易标志: 66=主买, 83=主卖, 32=撤单)
///   - 列5: ask_order (卖单订单ID)
///   - 列6: bid_order (买单订单ID)
/// 
/// - snaps_array: 盘口快照数据 (N_snaps x 41+)
///   - 列0: exchtime (时间戳)
///   - 列1-10: bid_prc1-10 (买价1-10档)
///   - 列11-20: bid_vol1-10 (买量1-10档)
///   - 列21-30: ask_prc1-10 (卖价1-10档)
///   - 列31-40: ask_vol1-10 (卖量1-10档)
/// 
/// 输出：
/// - features_array: 特征数组 (N_snaps * 20 x 15)
///   - 每行代表一个(快照, 档位, 方向)组合
///   - 列0: timestamp (时间戳)
///   - 列1: side_flag (0=Bid, 1=Ask)
///   - 列2: level_index (1-10档)
///   - 列3: vol_sum (成交量总和)
///   - 列4: vol_mean (成交量均值)
///   - 列5: vol_std (成交量标准差)
///   - 列6: vol_skew (成交量偏度)
///   - 列7: vol_autocorr (成交量滞后1阶自相关)
///   - 列8: vol_trend (成交量趋势相关性)
///   - 列9: id_count (匹配订单数)
///   - 列10: id_span (订单ID跨度)
///   - 列11: id_mean_diff (订单ID差值均值)
///   - 列12: id_std_diff (订单ID差值标准差)
///   - 列13: id_skew_diff (订单ID差值偏度)
///   - 列14: id_trend (订单ID趋势相关性)
#[pyfunction]
#[pyo3(signature = (ticks_array, snaps_array))]
pub fn reconstruct_limit_order_lifecycle_v2(
    py: Python,
    ticks_array: PyReadonlyArray2<f64>,
    snaps_array: PyReadonlyArray2<f64>,
) -> PyResult<Py<PyArray2<f64>>> {
    let ticks = ticks_array.as_array();
    let snaps = snaps_array.as_array();
    let (n_ticks, tick_cols) = ticks.dim();
    let (n_snaps, snap_cols) = snaps.dim();

    // 输入验证
    if tick_cols < 7 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "ticks_array需要至少7列 (exchtime, price, volume, turnover, flag, ask_order, bid_order)",
        ));
    }
    if snap_cols < 41 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "snaps_array需要至少41列 (exchtime + bid_prc1-10 + bid_vol1-10 + ask_prc1-10 + ask_vol1-10)",
        ));
    }

    // ============================================
    // Phase A: Anchoring (预处理阶段)
    // ============================================
    // 对于每个快照，计算：
    // 1. 在快照时间点之前的最大订单ID (MAX_BID_ID_t, MAX_ASK_ID_t)
    // 2. 该快照在tick数组中的起始索引
    
    let mut start_indices = vec![0usize; n_snaps];
    let mut max_bid_ids = vec![0i64; n_snaps];
    let mut max_ask_ids = vec![0i64; n_snaps];
    
    let mut tick_idx = 0usize;
    let mut current_max_bid = 0i64;
    let mut current_max_ask = 0i64;

    for s_idx in 0..n_snaps {
        let snap_ts = snaps[[s_idx, 0]];
        
        // 处理时间戳小于当前快照的所有tick
        while tick_idx < n_ticks && ticks[[tick_idx, 0]] < snap_ts {
            let ask_id = ticks[[tick_idx, 5]] as i64;
            let bid_id = ticks[[tick_idx, 6]] as i64;
            if ask_id > current_max_ask {
                current_max_ask = ask_id;
            }
            if bid_id > current_max_bid {
                current_max_bid = bid_id;
            }
            tick_idx += 1;
        }

        // 处理时间戳等于当前快照的所有tick
        let mut temp_idx = tick_idx;
        while temp_idx < n_ticks && ticks[[temp_idx, 0]] == snap_ts {
            let ask_id = ticks[[temp_idx, 5]] as i64;
            let bid_id = ticks[[temp_idx, 6]] as i64;
            if ask_id > current_max_ask {
                current_max_ask = ask_id;
            }
            if bid_id > current_max_bid {
                current_max_bid = bid_id;
            }
            temp_idx += 1;
        }

        start_indices[s_idx] = tick_idx;
        max_bid_ids[s_idx] = current_max_bid;
        max_ask_ids[s_idx] = current_max_ask;
        tick_idx = temp_idx;
    }

    // 使用Arc共享数据，避免拷贝
    let ticks_arc = Arc::new(ticks.to_owned());
    let snaps_arc = Arc::new(snaps.to_owned());
    let start_indices = Arc::new(start_indices);
    let max_bid_ids = Arc::new(max_bid_ids);
    let max_ask_ids = Arc::new(max_ask_ids);

    // 计算输出数组大小
    let total_rows = n_snaps * LEVELS * 2; // 每个快照 x 10档 x 2个方向
    let mut output = vec![f64::NAN; total_rows * OUTPUT_COLS];

    // ============================================
    // Phase B-D: Window Scanning & Trade Matching & Feature Engineering
    // ============================================
    // 非并行处理每个快照
    py.allow_threads(|| {
        output
            .chunks_mut(LEVELS * 2 * OUTPUT_COLS)
            .enumerate()
            .for_each(|(s_idx, chunk)| {
                let snap_ts = snaps_arc[[s_idx, 0]];
                let start_idx = start_indices[s_idx];
                let max_bid_id = max_bid_ids[s_idx];
                let max_ask_id = max_ask_ids[s_idx];

                // 遍历两个方向：0=Bid, 1=Ask
                for side in 0..2 {
                    // 遍历10个档位
                    for level in 0..LEVELS {
                        let row = side * LEVELS + level;
                        let base = row * OUTPUT_COLS;
                        
                        // 输出基本信息
                        chunk[base] = snap_ts;                    // timestamp
                        chunk[base + 1] = side as f64;            // side_flag (0=Bid, 1=Ask)
                        chunk[base + 2] = (level + 1) as f64;     // level_index (1-10)

                        // 确定目标价格和ID限制
                        let price_col = if side == 0 {
                            1 + level  // Bid: bid_prc1-10 在列1-10
                        } else {
                            21 + level // Ask: ask_prc1-10 在列21-30
                        };
                        let target_price = snaps_arc[[s_idx, price_col]];
                        let id_limit = if side == 0 { max_bid_id } else { max_ask_id };

                        // 收集匹配的成交量和被动订单ID
                        let mut volumes: Vec<f64> = Vec::new();
                        let mut passive_ids: Vec<f64> = Vec::new();

                        // ============================================
                        // Phase B: Window Scanning (前向扫描)
                        // ============================================
                        let mut idx = start_idx;
                        while idx < n_ticks {
                            let tick_price = ticks_arc[[idx, 1]];
                            
                            // 检查是否突破档位（Breakdown Condition）
                            if side == 0 {
                                // Bid侧：价格低于目标价表示档位被向下突破
                                if tick_price < target_price {
                                    break;
                                }
                            } else {
                                // Ask侧：价格高于目标价表示档位被向上突破
                                if tick_price > target_price {
                                    break;
                                }
                            }

                            let flag = ticks_arc[[idx, 4]] as i32;
                            let ask_id = ticks_arc[[idx, 5]] as i64;
                            let bid_id = ticks_arc[[idx, 6]] as i64;

                            // ============================================
                            // Phase C: Trade Matching (交易匹配)
                            // ============================================
                            if tick_price == target_price {
                                if side == 0 {
                                    // Bid侧分析：寻找主动卖单(flag=83)击中被动买单
                                    // 且买单ID不超过当前最大ID（确保是已存在的挂单）
                                    if flag == 83 && bid_id <= id_limit {
                                        volumes.push(ticks_arc[[idx, 2]]);
                                        passive_ids.push(bid_id as f64);
                                    }
                                } else {
                                    // Ask侧分析：寻找主动买单(flag=66)击中被动卖单
                                    // 且卖单ID不超过当前最大ID（确保是已存在的挂单）
                                    if flag == 66 && ask_id <= id_limit {
                                        volumes.push(ticks_arc[[idx, 2]]);
                                        passive_ids.push(ask_id as f64);
                                    }
                                }
                            }

                            idx += 1;
                        }

                        // ============================================
                        // Phase D: Feature Engineering (特征工程)
                        // ============================================
                        
                        // 如果没有匹配到任何交易
                        if volumes.is_empty() {
                            chunk[base + 3] = 0.0;  // vol_sum = 0
                            chunk[base + 9] = 0.0;  // id_count = 0
                            // 其他列保持NaN
                            continue;
                        }

                        // Group 1: Volume Statistics (成交量统计)
                        let vol_sum = volumes.iter().sum::<f64>();
                        let vol_mean = mean(&volumes);
                        let vol_std = std_dev(&volumes, vol_mean);
                        let vol_skew = skewness(&volumes, vol_mean, vol_std);
                        let vol_autocorr = autocorr_lag1(&volumes);
                        let vol_trend = trend_corr(&volumes);

                        chunk[base + 3] = vol_sum;
                        chunk[base + 4] = vol_mean;
                        chunk[base + 5] = vol_std;
                        chunk[base + 6] = vol_skew;
                        chunk[base + 7] = vol_autocorr;
                        chunk[base + 8] = vol_trend;

                        // Group 2: Order ID Statistics (订单ID统计)
                        // 预处理：将ID转换为相对值 ΔID = ID_limit - tick.passive_ID
                        let mut deltas: Vec<f64> = passive_ids
                            .iter()
                            .map(|id| (id_limit as f64) - id)
                            .collect();
                        
                        let id_count = deltas.len() as f64;
                        let id_mean = mean(&deltas);
                        let id_std = std_dev(&deltas, id_mean);
                        let id_skew = skewness(&deltas, id_mean, id_std);
                        
                        // ID趋势：使用绝对ID计算与交易序列索引的相关性
                        // 正值表示先吃老订单，负值表示先吃新订单
                        let id_trend = trend_corr(&passive_ids);

                        // 计算ID跨度
                        deltas.sort_by(|a, b| a.partial_cmp(b).unwrap());
                        let id_span = if deltas.len() >= 2 {
                            deltas[deltas.len() - 1] - deltas[0]
                        } else {
                            0.0
                        };

                        chunk[base + 9] = id_count;
                        chunk[base + 10] = id_span;
                        chunk[base + 11] = id_mean;
                        chunk[base + 12] = id_std;
                        chunk[base + 13] = id_skew;
                        chunk[base + 14] = id_trend;
                    }
                }
            });
    });

    // 构建输出数组
    let result = Array2::from_shape_vec((total_rows, OUTPUT_COLS), output).map_err(|_| {
        pyo3::exceptions::PyValueError::new_err("输出数组形状构建失败")
    })?;
    
    Ok(result.into_pyarray(py).to_owned())
}
