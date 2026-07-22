//! 3 秒微观结构 CAPM 横截面因子。
//!
//! 每只股票先从逐笔成交和盘口快照构造固定 3 秒网格；对每个基础指标构造
//! 全市场等权基准，并用过去 10 分钟、严格截止当前桶之前的数据估计滚动暴露；
//! 最后在同一个 3 秒桶上做横截面回归，输出逐股日内均值。

use crate::fast_csv_reader::{
    read_market_fast_inner, read_trade_fast_inner, MarketRecord, TradeRecord,
};
use crate::features;
use ndarray::Array2;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::fs;
use std::io;

const BIN_US: i64 = 3_000_000;
const DAY_US: i64 = 86_400_000_000;
const MARKET_OPEN_US: i64 = (9 * 3600 + 30 * 60) * 1_000_000;
const MORNING_END_US: i64 = (11 * 3600 + 30 * 60) * 1_000_000;
const AFTERNOON_OPEN_US: i64 = 13 * 3600 * 1_000_000;
const MARKET_CLOSE_US: i64 = (14 * 3600 + 57 * 60) * 1_000_000;
const N_BINS: usize = 4_740;
const MIDDAY_BIN: usize = 2_400;
const MAX_FFILL_BINS: usize = 5;
const ROLLING_WINDOW: usize = 200;
const MIN_HISTORY_OBS: u32 = 60;
const N_BASE: usize = 12;
const N_STATS: usize = 10;

const BASE_NAMES: [&str; N_BASE] = [
    "active_buy_volume_ratio",
    "order_gap_signed_vw",
    "observable_ratio_level",
    "book_imbalance10_level",
    "observable_ratio_innovation",
    "book_imbalance10_innovation",
    "spread_bps",
    "near3_depth_share",
    "microprice_pressure_bps",
    "order_gap_magnitude",
    "large_trade_direction_v2",
    "price_log_return_3s",
];

const STAT_NAMES: [&str; N_STATS] = [
    "rolling_beta_mean",
    "rolling_alpha_mean",
    "rolling_corr_market_mean",
    "rolling_residual_std_mean",
    "capm_residual_mean",
    "capm_residual_zscore_mean",
    "capm_residual_rank_mean",
    "capm_abs_residual_mean",
    "capm_leverage_mean",
    "capm_cooks_distance_mean",
];

pub const N_CAPM_COLUMNS: usize = N_BASE * N_STATS;
/// get_features_factors_rust_full(false): 21 个单列统计 + C(n, 2) 个列间相关。
pub const N_FACTORS: usize = 21 * N_CAPM_COLUMNS + N_CAPM_COLUMNS * (N_CAPM_COLUMNS - 1) / 2;

#[derive(Clone, Copy, Default)]
struct RollingSums {
    n: u32,
    sx: f64,
    sy: f64,
    sxx: f64,
    syy: f64,
    sxy: f64,
}

#[derive(Clone, Copy)]
struct Exposure {
    beta: f64,
    alpha: f64,
    corr: f64,
    residual_std: f64,
}

#[inline]
fn grid_start_us(timestamp_us: i64) -> i64 {
    timestamp_us.div_euclid(DAY_US) * DAY_US + MARKET_OPEN_US
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

fn extract_3s_features(trades: &[TradeRecord], market: &[MarketRecord]) -> Vec<f32> {
    let mut out = vec![f32::NAN; N_BINS * N_BASE];
    let timestamp = trades
        .first()
        .map(|x| x.time_us)
        .or_else(|| market.first().map(|x| x.time_us));
    let Some(timestamp) = timestamp else {
        return out;
    };
    let start_us = grid_start_us(timestamp);

    let mut buy_volume = vec![0.0f64; N_BINS];
    let mut sell_volume = vec![0.0f64; N_BINS];
    let mut gap_signed_num = vec![0.0f64; N_BINS];
    let mut gap_abs_num = vec![0.0f64; N_BINS];
    let mut gap_weight = vec![0.0f64; N_BINS];
    let mut large_signed = vec![0.0f64; N_BINS];
    let mut large_total = vec![0.0f64; N_BINS];

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
        out[bin * N_BASE] = ratio(buy_volume[bin], total_volume);
        out[bin * N_BASE + 1] = ratio(gap_signed_num[bin], gap_weight[bin]);
        out[bin * N_BASE + 9] = ratio(gap_abs_num[bin], gap_weight[bin]);
        out[bin * N_BASE + 10] = ratio(large_signed[bin], large_total[bin]);
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

pub fn microstructure_3s_feature_names() -> Vec<String> {
    BASE_NAMES.iter().map(|x| (*x).to_string()).collect()
}

pub fn compute_microstructure_3s_features(code: &str, date: i64) -> io::Result<Vec<f32>> {
    // 保留原始上午/下午时间后自行映射，避免 11:30 与平移后的 13:00 碰撞到同一桶。
    let trades = read_trade_fast_inner(code, date, false, false, usize::MAX)?;
    let market = read_market_fast_inner(code, date, false, false, usize::MAX)?;
    Ok(extract_3s_features(&trades, &market))
}

pub fn list_codes(date: i64) -> Vec<String> {
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

#[inline]
fn add_pair(sums: &mut RollingSums, x: f64, y: f64, sign: f64) {
    if !x.is_finite() || !y.is_finite() {
        return;
    }
    if sign > 0.0 {
        sums.n += 1;
    } else {
        sums.n = sums.n.saturating_sub(1);
    }
    sums.sx += sign * x;
    sums.sy += sign * y;
    sums.sxx += sign * x * x;
    sums.syy += sign * y * y;
    sums.sxy += sign * x * y;
}

fn exposure(sums: RollingSums) -> Option<Exposure> {
    if sums.n < MIN_HISTORY_OBS {
        return None;
    }
    let n = sums.n as f64;
    let sxx = sums.sxx - sums.sx * sums.sx / n;
    let syy = sums.syy - sums.sy * sums.sy / n;
    let sxy = sums.sxy - sums.sx * sums.sy / n;
    if sxx <= 1e-18 || syy <= 1e-18 {
        return None;
    }
    let beta = sxy / sxx;
    let alpha = sums.sy / n - beta * sums.sx / n;
    let corr = (sxy / (sxx * syy).sqrt()).clamp(-1.0, 1.0);
    let sse = (syy - beta * sxy).max(0.0);
    Some(Exposure {
        beta,
        alpha,
        corr,
        residual_std: (sse / (n - 2.0)).sqrt(),
    })
}

fn percentile_ranks_in_place(values: &mut [(usize, f64)], ranks: &mut [f64]) {
    values.sort_unstable_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
    ranks.fill(f64::NAN);
    if values.is_empty() {
        return;
    }
    let denominator = values.len().saturating_sub(1).max(1) as f64;
    let mut start = 0;
    while start < values.len() {
        let mut end = start + 1;
        while end < values.len() && values[end].1 == values[start].1 {
            end += 1;
        }
        let rank = (start + end - 1) as f64 * 0.5 / denominator;
        for &(stock, _) in &values[start..end] {
            ranks[stock] = rank;
        }
        start = end;
    }
}

#[allow(clippy::too_many_arguments)]
fn fit_cross_section_in_place(
    current: &[f32],
    exposures: &[Option<Exposure>],
    valid: &mut Vec<usize>,
    residual: &mut [f64],
    residual_rank: &mut [f64],
    leverage: &mut [f64],
    rank_pairs: &mut Vec<(usize, f64)>,
) -> Option<(f64, f64)> {
    let n_stocks = current.len();
    valid.clear();
    let mut sx = 0.0;
    let mut sy = 0.0;
    for stock in 0..n_stocks {
        let y = current[stock] as f64;
        if y.is_finite() {
            if let Some(e) = exposures[stock] {
                valid.push(stock);
                sx += e.beta;
                sy += y;
            }
        }
    }
    if valid.len() < 30 {
        return None;
    }
    let n = valid.len() as f64;
    let mean_x = sx / n;
    let mean_y = sy / n;
    let mut sxx = 0.0;
    let mut sxy = 0.0;
    let mut syy = 0.0;
    for &stock in valid.iter() {
        let x = exposures[stock].unwrap().beta;
        let y = current[stock] as f64;
        sxx += (x - mean_x).powi(2);
        sxy += (x - mean_x) * (y - mean_y);
        syy += (y - mean_y).powi(2);
    }
    if sxx <= 1e-18 || syy <= 1e-18 {
        return None;
    }
    let slope = sxy / sxx;
    let intercept = mean_y - slope * mean_x;
    residual.fill(f64::NAN);
    leverage.fill(f64::NAN);
    rank_pairs.clear();
    let mut sse = 0.0;
    for &stock in valid.iter() {
        let x = exposures[stock].unwrap().beta;
        let value = current[stock] as f64 - (intercept + slope * x);
        residual[stock] = value;
        leverage[stock] = 1.0 / n + (x - mean_x).powi(2) / sxx;
        rank_pairs.push((stock, value));
        sse += value * value;
    }
    percentile_ranks_in_place(rank_pairs, residual_rank);
    Some(((sse / (n - 2.0)).sqrt(), sse))
}

fn compute_one_feature(signal: &[f32], market: &[f64], n_stocks: usize) -> Vec<f32> {
    // 布局为 [time][stock][stat]，横截面热循环与后续按股票抽取都保持顺序访问。
    let mut output = vec![f32::NAN; N_BINS * n_stocks * N_STATS];
    let mut rolling = vec![RollingSums::default(); n_stocks];
    let mut exposures = vec![None; n_stocks];
    let mut valid = Vec::with_capacity(n_stocks);
    let mut residual = vec![f64::NAN; n_stocks];
    let mut residual_rank = vec![f64::NAN; n_stocks];
    let mut leverage = vec![f64::NAN; n_stocks];
    let mut rank_pairs = Vec::with_capacity(n_stocks);

    for bin in 0..N_BINS {
        if bin == MIDDAY_BIN {
            rolling.fill(RollingSums::default());
            exposures.fill(None);
        }
        if bin >= ROLLING_WINDOW {
            let old_bin = bin - ROLLING_WINDOW;
            if !(old_bin < MIDDAY_BIN && bin >= MIDDAY_BIN) {
                let old_market = market[old_bin];
                let old_base = old_bin * n_stocks;
                for stock in 0..n_stocks {
                    add_pair(
                        &mut rolling[stock],
                        old_market,
                        signal[old_base + stock] as f64,
                        -1.0,
                    );
                }
            }
        }
        for stock in 0..n_stocks {
            exposures[stock] = exposure(rolling[stock]);
        }
        let current_base = bin * n_stocks;
        let current = &signal[current_base..current_base + n_stocks];
        if let Some((residual_std, sse)) = fit_cross_section_in_place(
            current,
            &exposures,
            &mut valid,
            &mut residual,
            &mut residual_rank,
            &mut leverage,
            &mut rank_pairs,
        ) {
            for &stock in &valid {
                let e = exposures[stock].unwrap();
                let stock_residual = residual[stock];
                let h = leverage[stock];
                let one_minus_h = (1.0 - h).max(1e-12);
                let z = if residual_std > 0.0 {
                    stock_residual / residual_std
                } else {
                    f64::NAN
                };
                let studentized = z / one_minus_h.sqrt();
                let values = [
                    e.beta,
                    e.alpha,
                    e.corr,
                    e.residual_std,
                    stock_residual,
                    z,
                    residual_rank[stock],
                    stock_residual.abs(),
                    h,
                    if sse > 0.0 {
                        studentized * studentized * h / (2.0 * one_minus_h)
                    } else {
                        f64::NAN
                    },
                ];
                let output_base = (bin * n_stocks + stock) * N_STATS;
                for (stat, &value) in values.iter().enumerate() {
                    output[output_base + stat] = value as f32;
                }
            }
        }
        let current_market = market[bin];
        for stock in 0..n_stocks {
            add_pair(
                &mut rolling[stock],
                current_market,
                current[stock] as f64,
                1.0,
            );
        }
    }
    output
}

fn compute_for_codes(
    date: i64,
    requested_codes: Vec<String>,
) -> io::Result<(Vec<String>, Vec<f32>)> {
    let n_stocks = requested_codes.len();
    if n_stocks == 0 {
        return Ok((Vec::new(), Vec::new()));
    }
    // 第一阶段：股票间独立，按股票并行读取并降维到 3 秒网格。
    let feature_rows: Vec<Option<Vec<f32>>> = requested_codes
        .par_iter()
        .map(|code| compute_microstructure_3s_features(code, date).ok())
        .collect();
    let loaded: Vec<bool> = feature_rows.iter().map(Option::is_some).collect();

    // 第二阶段：转成 [feature][time][stock]，CAPM 热循环按 stock 连续访问。
    let mut signals = vec![f32::NAN; N_BASE * N_BINS * n_stocks];
    signals
        .par_chunks_mut(n_stocks)
        .enumerate()
        .for_each(|(row, destination)| {
            let feature = row / N_BINS;
            let bin = row % N_BINS;
            let source = bin * N_BASE + feature;
            for stock in 0..n_stocks {
                if let Some(values) = &feature_rows[stock] {
                    destination[stock] = values[source];
                }
            }
        });
    drop(feature_rows);

    let mut market = vec![f64::NAN; N_BASE * N_BINS];
    market.par_iter_mut().enumerate().for_each(|(row, output)| {
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

    // 第三阶段：12 个基础指标彼此独立，按指标并行做滚动暴露和截面回归。
    let feature_results: Vec<Vec<f32>> = (0..N_BASE)
        .into_par_iter()
        .map(|feature| {
            let start = feature * N_BINS * n_stocks;
            let end = start + N_BINS * n_stocks;
            compute_one_feature(
                &signals[start..end],
                &market[feature * N_BINS..(feature + 1) * N_BINS],
                n_stocks,
            )
        })
        .collect();

    // 第四阶段：每只股票组装 4740×120 CAPM 序列矩阵，再并行降维到 9660 维。
    // get_features_factors_rust_full 内部是串行列计算，因此外层按股票并行，不产生嵌套池。
    let reduced: Vec<Option<Vec<f32>>> = (0..n_stocks)
        .into_par_iter()
        .map(|stock| {
            if !loaded[stock] {
                return None;
            }
            let mut matrix_data = vec![f32::NAN; N_BINS * N_CAPM_COLUMNS];
            let mut has_value = false;
            for bin in 0..N_BINS {
                let destination = bin * N_CAPM_COLUMNS;
                for feature in 0..N_BASE {
                    let source = (bin * n_stocks + stock) * N_STATS;
                    let target = destination + feature * N_STATS;
                    let slice = &feature_results[feature][source..source + N_STATS];
                    has_value |= slice.iter().any(|x| x.is_finite());
                    matrix_data[target..target + N_STATS].copy_from_slice(slice);
                }
            }
            if !has_value {
                return None;
            }
            let matrix = Array2::from_shape_vec((N_BINS, N_CAPM_COLUMNS), matrix_data).ok()?;
            let values = features::get_features_factors_rust_values_only(&matrix.view(), false);
            (values.len() == N_FACTORS).then_some(values)
        })
        .collect();

    let valid_count = reduced.iter().filter(|x| x.is_some()).count();
    let mut codes = Vec::with_capacity(valid_count);
    let mut values = Vec::with_capacity(valid_count * N_FACTORS);
    for (stock, result) in reduced.into_iter().enumerate() {
        if let Some(result) = result {
            codes.push(requested_codes[stock].clone());
            values.extend(result);
        }
    }
    Ok((codes, values))
}

pub fn compute_microstructure_capm_full(date: i64) -> io::Result<(Vec<String>, Vec<f32>)> {
    compute_for_codes(date, list_codes(date))
}

pub fn microstructure_capm_sequence_names() -> Vec<String> {
    BASE_NAMES
        .iter()
        .flat_map(|base| {
            STAT_NAMES
                .iter()
                .map(move |stat| format!("microcapm_{base}_{stat}"))
        })
        .collect()
}

pub fn microstructure_capm_names() -> Vec<String> {
    let columns = microstructure_capm_sequence_names();
    let dummy = Array2::from_elem((1, N_CAPM_COLUMNS), f32::NAN);
    let (_, names) = features::get_features_factors_rust_full(&dummy.view(), &columns, false);
    names
}

#[pyfunction]
pub fn py_microstructure_3s_features(py: Python<'_>, code: &str, date: i64) -> PyResult<Vec<f32>> {
    py.allow_threads(|| compute_microstructure_3s_features(code, date))
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
}

#[pyfunction]
pub fn py_microstructure_3s_feature_names() -> Vec<String> {
    microstructure_3s_feature_names()
}

#[pyfunction]
pub fn py_microstructure_capm(py: Python<'_>, date: i64) -> PyResult<(Vec<String>, Vec<f32>)> {
    py.allow_threads(|| compute_microstructure_capm_full(date))
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
}

#[pyfunction]
pub fn py_microstructure_capm_codes(
    py: Python<'_>,
    date: i64,
    codes: Vec<String>,
) -> PyResult<(Vec<String>, Vec<f32>)> {
    py.allow_threads(|| compute_for_codes(date, codes))
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
}

#[pyfunction]
pub fn py_microstructure_capm_names() -> Vec<String> {
    microstructure_capm_names()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn factor_names_have_fixed_size() {
        assert_eq!(microstructure_3s_feature_names().len(), N_BASE);
        assert_eq!(microstructure_capm_sequence_names().len(), N_CAPM_COLUMNS);
        assert_eq!(microstructure_capm_names().len(), N_FACTORS);
        assert_eq!(N_CAPM_COLUMNS, 120);
        assert_eq!(N_FACTORS, 9_660);
    }

    #[test]
    fn rolling_exposure_recovers_linear_signal() {
        let mut sums = RollingSums::default();
        for i in 0..100 {
            let x = i as f64 / 100.0;
            add_pair(&mut sums, x, 0.2 + 1.7 * x, 1.0);
        }
        let got = exposure(sums).unwrap();
        assert!((got.beta - 1.7).abs() < 1e-10);
        assert!((got.alpha - 0.2).abs() < 1e-10);
        assert!((got.corr - 1.0).abs() < 1e-10);
    }

    #[test]
    fn grid_excludes_end_point() {
        let start = 10 * DAY_US + MARKET_OPEN_US;
        assert_eq!(bin_index(start, start), Some(0));
        assert_eq!(
            bin_index(start + (MIDDAY_BIN as i64 - 1) * BIN_US, start),
            Some(MIDDAY_BIN - 1)
        );
        assert_eq!(bin_index(start + MIDDAY_BIN as i64 * BIN_US, start), None);
        let day_start = start - MARKET_OPEN_US;
        assert_eq!(
            bin_index(day_start + AFTERNOON_OPEN_US, start),
            Some(MIDDAY_BIN)
        );
        assert_eq!(bin_index(day_start + MARKET_CLOSE_US, start), None);
    }
}
