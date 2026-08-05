//! 多因子 CAPM 数据探索：14 个基础指标的全市场等权均值序列 + 两两相关矩阵。
//!
//! 目的：为 multi_factor_capm 的 y 精简方案提供真实数据依据。
//! 12 个指标沿用 microstructure_capm_metrics 的 3 秒网格定义；
//! 新增 log_volume_3s（桶内成交量 log1p）与 trade_arrival_clustering
//! （桶内逐笔时间间隔 CV，度量成交聚集度）。
//!
//! 用法: mf_sandbox <date> [date ...]

mod fast_csv_reader;
mod multi_factor;

use fast_csv_reader::{read_market_fast_inner, read_trade_fast_inner, MarketRecord, TradeRecord};
use multi_factor::MODELS;
use rayon::prelude::*;
use std::fs;
use std::time::Instant;

const BIN_US: i64 = 3_000_000;
const MARKET_OPEN_US: i64 = (9 * 3600 + 30 * 60) * 1_000_000;
const MORNING_END_US: i64 = (11 * 3600 + 30 * 60) * 1_000_000;
const AFTERNOON_OPEN_US: i64 = 13 * 3600 * 1_000_000;
const MARKET_CLOSE_US: i64 = (14 * 3600 + 57 * 60) * 1_000_000;
const N_BINS: usize = 4_740;
const MIDDAY_BIN: usize = 2_400;
const MAX_FFILL_BINS: usize = 5;
const N_BASE: usize = 14;

const BASE_NAMES: [&str; N_BASE] = [
    "active_buy_volume_ratio",   // 0  主买占比（方向）
    "order_gap_signed_vw",       // 1  带符号订单编号差（方向）
    "observable_ratio_level",    // 2  可观测挂单占比水平（结构）
    "book_imbalance10_level",    // 3  10档不平衡水平（方向）
    "observable_ratio_innovation", // 4 可观测占比差分（结构变化）
    "book_imbalance10_innovation", // 5 不平衡差分（结构变化）
    "spread_bps",                // 6  价差（成本）
    "near3_depth_share",         // 7  近3档深度占比（结构）
    "microprice_pressure_bps",   // 8  微价压力（压力）
    "order_gap_magnitude",       // 9  订单编号差幅度（强度）
    "large_trade_direction_v2",  // 10 大单方向（方向）
    "price_log_return_3s",       // 11 3秒对数收益（价格）
    "log_volume_3s",             // 12 桶内成交量 log1p（量，新增）
    "trade_arrival_clustering",  // 13 桶内逐笔间隔CV（聚集度，新增）
];

fn grid_start_us_for_date(date: i64) -> i64 {
    let year = (date / 10_000) as i32;
    let month = ((date / 100) % 100) as u32;
    let day = (date % 100) as u32;
    let midnight_us = chrono_like_midnight_us(year, month, day);
    midnight_us + MARKET_OPEN_US
}

// 不引入 chrono，手写天数累加（2020-2030 够用）
fn chrono_like_midnight_us(year: i32, month: u32, day: u32) -> i64 {
    let days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
    let leap = |y: i32| (y % 4 == 0 && y % 100 != 0) || y % 400 == 0;
    let mut days: i64 = 0;
    for y in 1970..year {
        days += if leap(y) { 366 } else { 365 };
    }
    for m in 1..month {
        days += days_in_month[(m - 1) as usize] as i64;
        if m == 2 && leap(year) {
            days += 1;
        }
    }
    days += (day - 1) as i64;
    days * 86_400 * 1_000_000
}

#[inline]
fn bin_index(timestamp_us: i64, start_us: i64) -> Option<usize> {
    let day_start_us = start_us - MARKET_OPEN_US;
    let offset = timestamp_us - day_start_us;
    if (MARKET_OPEN_US..MORNING_END_US).contains(&offset) {
        Some(((offset - MARKET_OPEN_US) / BIN_US) as usize)
    } else if (AFTERNOON_OPEN_US..MARKET_CLOSE_US).contains(&offset) {
        Some(MIDDAY_BIN + ((offset - AFTERNOON_OPEN_US) / BIN_US) as usize)
    } else {
        None
    }
}

#[inline]
fn ratio(num: f64, den: f64) -> f32 {
    if den > 0.0 && num.is_finite() && den.is_finite() {
        (num / den) as f32
    } else {
        f32::NAN
    }
}

/// 单只股票 → 4740×14 的 3 秒网格特征。
/// 12 列逻辑与 microstructure_capm_metrics::extract_3s_features 完全一致；
/// 12/13 列为新增（log_volume_3s, trade_arrival_clustering）。
fn extract_14_features(trades: &[TradeRecord], market: &[MarketRecord], start_us: i64) -> Vec<f32> {
    let mut out = vec![f32::NAN; N_BINS * N_BASE];

    let mut buy_volume = vec![0.0f64; N_BINS];
    let mut sell_volume = vec![0.0f64; N_BINS];
    let mut gap_signed_num = vec![0.0f64; N_BINS];
    let mut gap_abs_num = vec![0.0f64; N_BINS];
    let mut gap_weight = vec![0.0f64; N_BINS];
    let mut large_signed = vec![0.0f64; N_BINS];
    let mut large_total = vec![0.0f64; N_BINS];
    // 新增：成交时间聚集度（桶内逐笔间隔 CV）
    let mut bin_ticks = vec![0u32; N_BINS];
    let mut bin_prev = vec![0i64; N_BINS];
    let mut bin_sum_dt = vec![0.0f64; N_BINS];
    let mut bin_sum_dt2 = vec![0.0f64; N_BINS];

    for trade in trades {
        let Some(bin) = bin_index(trade.time_us, start_us) else {
            continue;
        };
        let volume = trade.volume as f64;
        if !volume.is_finite() || volume <= 0.0 {
            continue;
        }
        let direction = match trade.flag {
            66 => {
                buy_volume[bin] += volume;
                1.0
            }
            83 => {
                sell_volume[bin] += volume;
                -1.0
            }
            _ => continue,
        };
        // 有效成交（66/83）的时间间隔累计
        if bin_ticks[bin] > 0 {
            let dt = (trade.time_us - bin_prev[bin]) as f64;
            bin_sum_dt[bin] += dt;
            bin_sum_dt2[bin] += dt * dt;
        }
        bin_ticks[bin] += 1;
        bin_prev[bin] = trade.time_us;

        let volume2 = volume * volume;
        large_signed[bin] += direction * volume2;
        large_total[bin] += volume2;

        if trade.bid_order > 0 && trade.ask_order > 0 {
            let bid = trade.bid_order as f64;
            let ask = trade.ask_order as f64;
            let den = bid.abs() + ask.abs();
            if den > 0.0 {
                let relative_gap = (bid - ask) / den;
                gap_signed_num[bin] += volume * relative_gap;
                gap_abs_num[bin] += volume * relative_gap.abs();
                gap_weight[bin] += volume;
            }
        }
    }

    for bin in 0..N_BINS {
        let total_volume = buy_volume[bin] + sell_volume[bin];
        out[bin * N_BASE + 0] = ratio(buy_volume[bin], total_volume);
        out[bin * N_BASE + 1] = ratio(gap_signed_num[bin], gap_weight[bin]);
        out[bin * N_BASE + 9] = ratio(gap_abs_num[bin], gap_weight[bin]);
        out[bin * N_BASE + 10] = ratio(large_signed[bin], large_total[bin]);
        // 新增：对数成交量
        out[bin * N_BASE + 12] = if total_volume > 0.0 {
            (total_volume).ln_1p() as f32
        } else {
            f32::NAN
        };
        // 新增：成交时间聚集度（间隔 CV）
        let n = bin_ticks[bin];
        if n >= 2 {
            let m = (n - 1) as f64;
            let mean = bin_sum_dt[bin] / m;
            if mean > 0.0 {
                let var = (bin_sum_dt2[bin] / m - mean * mean).max(0.0);
                out[bin * N_BASE + 13] = (var.sqrt() / mean) as f32;
            }
        }
    }

    // 盘口是状态量：同桶存在多个快照时取最后一个；空桶随后最多向前填充 15 秒。
    for snapshot in market {
        let Some(bin) = bin_index(snapshot.time_us, start_us) else {
            continue;
        };
        let ask10: f64 = snapshot.ask_vols.iter().map(|&x| x as f64).sum();
        let bid10: f64 = snapshot.bid_vols.iter().map(|&x| x as f64).sum();
        let ask3: f64 = snapshot.ask_vols[..3].iter().map(|&x| x as f64).sum();
        let bid3: f64 = snapshot.bid_vols[..3].iter().map(|&x| x as f64).sum();
        let obs_ask = ratio(ask10, snapshot.total_ask_vol as f64);
        let obs_bid = ratio(bid10, snapshot.total_bid_vol as f64);
        let observable = if obs_ask.is_finite() && obs_bid.is_finite() {
            0.5 * (obs_ask + obs_bid)
        } else {
            f32::NAN
        };
        let imbalance = ratio(bid10 - ask10, bid10 + ask10);
        let spread = {
            let ask1 = snapshot.ask_prcs[0] as f64;
            let bid1 = snapshot.bid_prcs[0] as f64;
            let mid = 0.5 * (ask1 + bid1);
            if ask1 >= bid1 && bid1 > 0.0 && mid > 0.0 {
                let value = (ask1 - bid1) / mid * 10_000.0;
                if value <= 1_000.0 {
                    value as f32
                } else {
                    f32::NAN
                }
            } else {
                f32::NAN
            }
        };
        let near3 = ratio(ask3 + bid3, ask10 + bid10);
        let micropressure = {
            let ask1 = snapshot.ask_prcs[0] as f64;
            let bid1 = snapshot.bid_prcs[0] as f64;
            let ask_vol1 = snapshot.ask_vols[0] as f64;
            let bid_vol1 = snapshot.bid_vols[0] as f64;
            let depth = ask_vol1 + bid_vol1;
            let mid = 0.5 * (ask1 + bid1);
            if depth > 0.0 && mid > 0.0 && ask1 >= bid1 && bid1 > 0.0 {
                let microprice = (ask1 * bid_vol1 + bid1 * ask_vol1) / depth;
                let value = (microprice - mid) / mid * 10_000.0;
                if value.abs() <= 1_000.0 {
                    value as f32
                } else {
                    f32::NAN
                }
            } else {
                f32::NAN
            }
        };
        let base = bin * N_BASE;
        out[base + 2] = observable;
        out[base + 3] = imbalance;
        out[base + 6] = spread;
        out[base + 7] = near3;
        out[base + 8] = micropressure;
        out[base + 11] = if snapshot.last_prc > 0.0 {
            snapshot.last_prc
        } else {
            f32::NAN
        };
    }

    for feature in [2usize, 3, 6, 7, 8] {
        let mut last = f32::NAN;
        let mut age = MAX_FFILL_BINS + 1;
        for bin in 0..N_BINS {
            if bin == MIDDAY_BIN {
                last = f32::NAN;
                age = MAX_FFILL_BINS + 1;
            }
            let idx = bin * N_BASE + feature;
            if out[idx].is_finite() {
                last = out[idx];
                age = 0;
            } else {
                age += 1;
                if age <= MAX_FFILL_BINS && last.is_finite() {
                    out[idx] = last;
                }
            }
        }
    }

    // 价格单独记录填充年龄，恢复报价时可阻止跨越超过 15 秒的收益。
    let mut price_age = vec![u8::MAX; N_BINS];
    for (session_start, session_end) in [(0usize, MIDDAY_BIN), (MIDDAY_BIN, N_BINS)] {
        let mut last = f32::NAN;
        let mut age = MAX_FFILL_BINS + 1;
        for (bin, age_out) in price_age
            .iter_mut()
            .enumerate()
            .take(session_end)
            .skip(session_start)
        {
            let idx = bin * N_BASE + 11;
            if out[idx].is_finite() {
                last = out[idx];
                age = 0;
            } else {
                age += 1;
                if age <= MAX_FFILL_BINS && last.is_finite() {
                    out[idx] = last;
                }
            }
            if out[idx].is_finite() {
                *age_out = age as u8;
            }
        }
    }

    for bin in 1..N_BINS {
        if bin == MIDDAY_BIN {
            continue;
        }
        let base = bin * N_BASE;
        let prev = (bin - 1) * N_BASE;
        if out[base + 2].is_finite() && out[prev + 2].is_finite() {
            out[base + 4] = out[base + 2] - out[prev + 2];
        }
        if out[base + 3].is_finite() && out[prev + 3].is_finite() {
            out[base + 5] = out[base + 3] - out[prev + 3];
        }
    }

    // 价格 CAPM 使用相邻 3 秒桶的对数收益；午休和超过 15 秒的空档都不跨越。
    for (session_start, session_end) in [(0usize, MIDDAY_BIN), (MIDDAY_BIN, N_BINS)] {
        let mut previous = out[session_start * N_BASE + 11];
        out[session_start * N_BASE + 11] = f32::NAN;
        for bin in session_start + 1..session_end {
            let idx = bin * N_BASE + 11;
            let current = out[idx];
            let resumed_after_too_long =
                price_age[bin] == 0 && price_age[bin - 1] >= MAX_FFILL_BINS as u8;
            out[idx] = if current > 0.0 && previous > 0.0 && !resumed_after_too_long {
                (current / previous).ln()
            } else {
                f32::NAN
            };
            previous = current;
        }
    }
    out
}

fn list_codes(date: i64) -> Vec<String> {
    let dir = format!("/ssd_data/stock/{date}/transaction");
    let mut codes = std::collections::BTreeSet::new();
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let name = entry.file_name().into_string().unwrap_or_default();
            if let Some(code) = name.split('_').next() {
                if code.len() == 6 && code.bytes().all(|x| x.is_ascii_digit()) {
                    codes.insert(code.to_string());
                }
            }
        }
    }
    codes.into_iter().collect()
}

/// 全市场等权均值序列 [N_BINS*N_BASE] + 每桶有效股票数 + 处理股票数。
fn compute_market(date: i64) -> (Vec<f32>, Vec<u32>, usize) {
    let codes = list_codes(date);
    let n_stocks = codes.len();
    let total = N_BINS * N_BASE;
    let zero = || (vec![0.0f64; total], vec![0u32; total]);
    let start_us = grid_start_us_for_date(date);

    let (sums, counts) = codes
        .par_iter()
        .fold(zero, |(mut s, mut c), code| {
            if let Ok(trades) = read_trade_fast_inner(code, date, false, false, usize::MAX) {
                if let Ok(market) = read_market_fast_inner(code, date, false, false, usize::MAX) {
                    let feats = extract_14_features(&trades, &market, start_us);
                    for b in 0..N_BINS {
                        let base = b * N_BASE;
                        for f in 0..N_BASE {
                            let v = feats[base + f];
                            if v.is_finite() {
                                s[base + f] += v as f64;
                                c[base + f] += 1;
                            }
                        }
                    }
                }
            }
            (s, c)
        })
        .reduce(zero, |(mut s1, mut c1), (s2, c2)| {
            for i in 0..total {
                s1[i] += s2[i];
                c1[i] += c2[i];
            }
            (s1, c1)
        });

    let mut means = vec![f32::NAN; total];
    for i in 0..total {
        if counts[i] > 0 {
            means[i] = (sums[i] / counts[i] as f64) as f32;
        }
    }
    (means, counts, n_stocks)
}

fn pearson_corr(a: &[f32], b: &[f32]) -> Option<(f64, usize)> {
    let mut n = 0usize;
    let (mut sx, mut sy, mut sxx, mut syy, mut sxy) = (0.0f64, 0.0, 0.0, 0.0, 0.0);
    for i in 0..a.len() {
        let x = a[i] as f64;
        let y = b[i] as f64;
        if x.is_finite() && y.is_finite() {
            n += 1;
            sx += x;
            sy += y;
            sxx += x * x;
            syy += y * y;
            sxy += x * y;
        }
    }
    if n < 30 {
        return None;
    }
    let nf = n as f64;
    let vx = sxx - sx * sx / nf;
    let vy = syy - sy * sy / nf;
    let vxy = sxy - sx * sy / nf;
    if vx <= 1e-18 || vy <= 1e-18 {
        return None;
    }
    Some(((vxy / (vx * vy).sqrt()).clamp(-1.0, 1.0), n))
}

fn describe(seq: &[f32]) -> (f64, f64, f64, f64, usize) {
    // (mean, std, min, max, valid_bins)
    let mut vals = Vec::with_capacity(seq.len());
    for &v in seq {
        if v.is_finite() {
            vals.push(v as f64);
        }
    }
    if vals.is_empty() {
        return (f64::NAN, f64::NAN, f64::NAN, f64::NAN, 0);
    }
    let n = vals.len() as f64;
    let mean = vals.iter().sum::<f64>() / n;
    let var = vals.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / n;
    let min = vals.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    (mean, var.sqrt(), min, max, vals.len())
}

/// 阶段 A：读全市场 → 网格 → 转置为 [feature][bin][stock] + 市场均值 [feature][bin]。
fn load_market_grid(date: i64) -> (Vec<String>, Vec<f32>, Vec<f64>, usize) {
    let codes = list_codes(date);
    let n_stocks = codes.len();
    let start_us = grid_start_us_for_date(date);
    let t0 = Instant::now();
    let feature_rows: Vec<Option<Vec<f32>>> = codes
        .par_iter()
        .map(|code| {
            let trades = read_trade_fast_inner(code, date, false, false, usize::MAX).ok()?;
            let market = read_market_fast_inner(code, date, false, false, usize::MAX).ok()?;
            Some(extract_14_features(&trades, &market, start_us))
        })
        .collect();
    eprintln!("  读盘+网格: {:.1}s", t0.elapsed().as_secs_f64());

    // 转置 [stock][bin][14] → [feature][bin][stock]
    let total = multi_factor::N_FEATURES * multi_factor::N_BINS * n_stocks;
    let mut signals = vec![f32::NAN; total];
    let t1 = Instant::now();
    signals
        .par_chunks_mut(n_stocks)
        .enumerate()
        .for_each(|(row, destination)| {
            let feature = row / multi_factor::N_BINS;
            let bin = row % multi_factor::N_BINS;
            let source = bin * N_BASE + feature;
            for stock in 0..n_stocks {
                if let Some(values) = &feature_rows[stock] {
                    destination[stock] = values[source];
                }
            }
        });
    eprintln!("  转置: {:.1}s", t1.elapsed().as_secs_f64());
    drop(feature_rows);

    // 市场均值 [feature][bin]
    let mut market = vec![f64::NAN; multi_factor::N_FEATURES * multi_factor::N_BINS];
    market
        .par_iter_mut()
        .enumerate()
        .for_each(|(row, output)| {
            let base = row * n_stocks;
            let mut sum = 0.0;
            let mut count = 0usize;
            for stock in 0..n_stocks {
                let value = signals[base + stock] as f64;
                if value.is_finite() {
                    sum += value;
                    count += 1;
                }
            }
            if count > 0 {
                *output = sum / count as f64;
            }
        });
    eprintln!("  市场均值: {:.1}s", t0.elapsed().as_secs_f64() - t1.elapsed().as_secs_f64());
    (codes, signals, market, n_stocks)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    // export_grid 模式: mf_sandbox export_grid <date> <dir> → 导出 signals/market 二进制
    if args.first().map(|s| s.as_str()) == Some("export_grid") && args.len() >= 3 {
        let date: i64 = args[1].parse().unwrap();
        let dir = &args[2];
        let (codes, signals, market, n_stocks) = load_market_grid(date);
        std::fs::create_dir_all(dir).unwrap();
        // signals: [feature][bin][stock] f32
        let mut sig = signals.clone();
        let sbuf: &[u8] = unsafe {
            std::slice::from_raw_parts(sig.as_mut_ptr() as *const u8, sig.len() * 4)
        };
        std::fs::write(format!("{dir}/signals_{date}.bin"), sbuf).unwrap();
        // market: [feature][bin] f64
        let m: Vec<f64> = market.to_vec();
        let mbuf: &[u8] = unsafe {
            std::slice::from_raw_parts(m.as_ptr() as *const u8, m.len() * 8)
        };
        std::fs::write(format!("{dir}/market_{date}.bin"), mbuf).unwrap();
        std::fs::write(format!("{dir}/codes_{date}.txt"), codes.join("\n")).unwrap();
        eprintln!("导出完成: {n_stocks} 股, signals {} MB, market {} KB",
            sig.len() * 4 / 1_000_000, m.len() * 8 / 1000);
        return;
    }
    // export_col 模式: mf_sandbox export_col <date> <model> <y> <col> <path> → 导出单列时序二进制 [4740, n]
    if args.first().map(|s| s.as_str()) == Some("export_col") && args.len() >= 6 {
        let date: i64 = args[1].parse().unwrap();
        let model_name = &args[2];
        let y: usize = args[3].parse().unwrap();
        let col: usize = args[4].parse().unwrap();
        let out_path = &args[5];
        let n_stocks = list_codes(date).len();
        let (prealloc, grid) = rayon::join(
            || multi_factor::prealloc_combos(n_stocks),
            || load_market_grid(date),
        );
        let (codes, signals, market, _) = grid;
        let routes =
            multi_factor::compute_route_timeseries(&signals, &market, n_stocks, prealloc);
        let route = routes.iter().find(|r| r.y == y).unwrap();
        let mi = MODELS.iter().position(|m| m.name == model_name).unwrap();
        let ci = route.model_indices.iter().position(|&v| v == mi).unwrap();
        let buf = &route.combos[ci];
        // 提取列时序 [4740, n]
        let mut col_data = Vec::with_capacity(multi_factor::N_BINS * n_stocks);
        for b in 0..multi_factor::N_BINS {
            for s in 0..n_stocks {
                col_data.push(buf.read(col, b, s));
            }
        }
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(col_data.as_ptr() as *const u8, col_data.len() * 4)
        };
        std::fs::write(out_path, bytes).unwrap();
        eprintln!("导出 {model_name} y={y} col={col} -> {out_path}");
        return;
    }
    // compute 模式: mf_sandbox compute <date> [model y]  → 核心计算 + 输出指定组合每股列均值
    if args.first().map(|s| s.as_str()) == Some("compute") && args.len() >= 2 {
        let date: i64 = args[1].parse().unwrap();
        // 预分配缓冲与读盘并行（页错误藏进 IO 等待）
        let n_stocks = list_codes(date).len();
        let (prealloc, grid) = rayon::join(
            || multi_factor::prealloc_combos(n_stocks),
            || load_market_grid(date),
        );
        let (codes, signals, market, _) = grid;
        let t0 = Instant::now();
        let routes =
            multi_factor::compute_route_timeseries(&signals, &market, n_stocks, prealloc);
        eprintln!("  核心计算: {:.1}s", t0.elapsed().as_secs_f64());
        // 输出模式：compute <date> <model> <y> → 该组合每股每列均值
        if args.len() >= 4 {
            let model_name = &args[2];
            let y: usize = args[3].parse().unwrap();
            let mi = MODELS.iter().position(|m| m.name == model_name).unwrap();
            let route = routes.iter().find(|r| r.y == y).unwrap();
            let ci = route
                .model_indices
                .iter()
                .position(|&v| v == mi)
                .unwrap();
            let buf = &route.combos[ci];
            let k = MODELS[mi].k;
            let cols = multi_factor::n_cols(k);
            let names = multi_factor::col_names(k, &MODELS[mi].factors);
            println!("code,{}", names.join(","));
            for s in 0..n_stocks {
                let mut row = Vec::with_capacity(cols);
                for c in 0..cols {
                    // 列均值（对 4740 桶）
                    let mut sum = 0.0f64;
                    let mut cnt = 0u64;
                    for b in 0..multi_factor::N_BINS {
                        let v = buf.read(c, b, s);
                        if v.is_finite() {
                            sum += v as f64;
                            cnt += 1;
                        }
                    }
                    row.push(if cnt > 0 {
                        format!("{:.6}", sum / cnt as f64)
                    } else {
                        "NA".to_string()
                    });
                }
                println!("{},{}", codes[s], row.join(","));
            }
            return;
        }
        // 无输出模式：只报内存与耗时概览
        let mut total_bytes = 0usize;
        for r in &routes {
            for cb in &r.combos {
                total_bytes += cb.per_stock.len() * 4 + cb.shared.len() * 4;
            }
        }
        println!(
            "compute {date}: stocks={n_stocks}, 时序缓冲 {:.1} GB",
            total_bytes as f64 / 1e9
        );
        for r in &routes {
            eprintln!("  y={} combos={}", r.y, r.combos.len());
        }
        return;
    }
    // 调试模式: mf_sandbox dump <date> <code>  → 输出单股 4740×14 网格(CSV)
    if args.first().map(|s| s.as_str()) == Some("dump") && args.len() >= 3 {
        let date: i64 = args[1].parse().unwrap();
        let code = &args[2];
        let start_us = grid_start_us_for_date(date);
        let trades = read_trade_fast_inner(code, date, false, false, usize::MAX).unwrap();
        let market = read_market_fast_inner(code, date, false, false, usize::MAX).unwrap();
        let feats = extract_14_features(&trades, &market, start_us);
        println!("{}", BASE_NAMES.join(","));
        for b in 0..N_BINS {
            let row: Vec<String> = (0..N_BASE)
                .map(|f| {
                    let v = feats[b * N_BASE + f];
                    if v.is_finite() {
                        format!("{:.8}", v)
                    } else {
                        "NA".to_string()
                    }
                })
                .collect();
            println!("{}", row.join(","));
        }
        return;
    }
    if args.is_empty() {
        eprintln!("用法: mf_sandbox <date> [date ...]  |  mf_sandbox dump <date> <code>");
        std::process::exit(1);
    }
    for date_str in &args {
        let date: i64 = match date_str.parse() {
            Ok(d) => d,
            Err(_) => {
                eprintln!("无效日期: {date_str}");
                continue;
            }
        };
        let t0 = Instant::now();
        let (means, counts, n_stocks) = compute_market(date);
        println!("=== date {date} ===  (耗时 {:.1}s)", t0.elapsed().as_secs_f64());
        println!("stocks: {n_stocks}, bins: {N_BINS}, features: {N_BASE}");

        println!("\n[每指标市场均值序列描述统计]");
        println!("{:<28} {:>10} {:>10} {:>12} {:>12} {:>10} {:>10}", "feature", "mean", "std", "min", "max", "valid_bins", "avg_stocks");
        let mut per_feature: Vec<Vec<f32>> = (0..N_BASE).map(|f| {
            (0..N_BINS).map(|b| means[b * N_BASE + f]).collect()
        }).collect();
        let mut avg_stocks = vec![0.0f64; N_BASE];
        for f in 0..N_BASE {
            let mut s = 0u64;
            for b in 0..N_BINS {
                s += counts[b * N_BASE + f] as u64;
            }
            avg_stocks[f] = s as f64 / N_BINS as f64;
        }
        for f in 0..N_BASE {
            let (mean, std, min, max, valid) = describe(&per_feature[f]);
            println!("{:<28} {:>10.4} {:>10.4} {:>12.4} {:>12.4} {:>10} {:>10.1}",
                BASE_NAMES[f], mean, std, min, max, valid, avg_stocks[f]);
        }

        println!("\n[14×14 相关矩阵] (Pearson, 双方非NaN桶)");
        print!("{:<30}", "");
        for j in 0..N_BASE {
            print!(" {:>6}", j);
        }
        println!();
        let mut corr = vec![vec![f64::NAN; N_BASE]; N_BASE];
        for i in 0..N_BASE {
            print!("{:<28} ", &BASE_NAMES[i][..BASE_NAMES[i].len().min(28)]);
            for j in 0..N_BASE {
                if let Some((r, _n)) = pearson_corr(&per_feature[i], &per_feature[j]) {
                    corr[i][j] = r;
                    print!(" {:>7.3}", r);
                } else {
                    print!(" {:>6}", "NA");
                }
            }
            println!();
        }

        println!("\n[|r| >= 0.55 的高相关对]");
        for i in 0..N_BASE {
            for j in i + 1..N_BASE {
                let r = corr[i][j];
                if r.is_finite() && r.abs() >= 0.55 {
                    println!("  ({:>2},{:>2}) {:<26} vs {:<26}  r = {:+.3}", i, j, BASE_NAMES[i], BASE_NAMES[j], r);
                }
            }
        }
        println!();
    }
}
