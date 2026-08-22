//! 多因子 CAPM 横截面因子：5 套市场因子模型 × 53 个 (模型, y) 组合。
//!
//! 设计（与用户确认的方案）：
//! - 14 个基础指标（12 个沿用 microstructure_capm + log_volume_3s + trade_arrival_clustering）
//! - 5 套市场因子模型：T1 核心三因子 / T2 订单流三因子 / T3 盘口节奏三因子 /
//!   F1 点名五因子 / F2 均衡五因子（市场因子 = 全市场等权均值，延续 microstructure_capm）
//! - y 清单经真实数据相关矩阵探索校准（53 组合，剔除同源变体与稳定高相关）
//! - 每组合 39/55 列统计量时序（4740 桶），每列 21 个单列时序统计（无列间相关）
//!
//! 计算流程（组合级并行，每组合独立滚动矩）：
//!   1. 滚动窗口（200 桶）增量维护 (y, F) 联合矩与 F-F 交叉矩
//!   2. 每桶解时间序列多元回归 → 暴露 β_k；单因子回归 → β(y, F_k)
//!   3. 每桶横截面回归（y ~ β 向量）→ λ/R²/残差/VIF/条件数等
//!   4. 主对比（vs y~market(y) 单因子）与因子级对比（vs 每个单因子）
//!   5. 每列 21 个时序统计（mean/median/std/skew/kurt/分位数/趋势/复杂度等）
//!
//! 输出：每股票 49,791 维（2371 列 × 21 统计）。

use crate::fast_csv_reader::{
    read_market_fast_inner, read_trade_fast_inner, MarketRecord, TradeRecord,
};
use crate::features;
use chrono::NaiveDate;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::fs;
use std::io;

pub const N_BINS: usize = 4_740;
pub const MIDDAY_BIN: usize = 2_400;
pub const N_FEATURES: usize = 14;
pub const ROLLING_WINDOW: usize = 200;
pub const MIN_HISTORY_OBS: u32 = 60;
pub const MIN_CS_STOCKS: usize = 30;
pub const MAX_FACTORS: usize = 5;
const BIN_US: i64 = 3_000_000;
const MARKET_OPEN_US: i64 = (9 * 3600 + 30 * 60) * 1_000_000;
const MORNING_END_US: i64 = (11 * 3600 + 30 * 60) * 1_000_000;
const AFTERNOON_OPEN_US: i64 = 13 * 3600 * 1_000_000;
const MARKET_CLOSE_US: i64 = (14 * 3600 + 57 * 60) * 1_000_000;
const MAX_FFILL_BINS: usize = 5;

// ---------------------------------------------------------------------------
// 基础指标（14 个）
// ---------------------------------------------------------------------------

pub const BASE_NAMES: [&str; N_FEATURES] = [
    "active_buy_volume_ratio",     // 0  主买占比（方向）
    "order_gap_signed_vw",         // 1  带符号订单编号差（方向）
    "observable_ratio_level",      // 2  可观测挂单占比水平（结构）
    "book_imbalance10_level",      // 3  10档不平衡水平（方向）
    "observable_ratio_innovation", // 4 可观测占比差分（结构变化）
    "book_imbalance10_innovation", // 5 不平衡差分（结构变化）
    "spread_bps",                  // 6  价差（成本）
    "near3_depth_share",           // 7  近3档深度占比（结构）
    "microprice_pressure_bps",     // 8  微价压力（压力）
    "order_gap_magnitude",         // 9  订单编号差幅度（强度）
    "large_trade_direction_v2",    // 10 大单方向（方向）
    "price_log_return_3s",         // 11 3秒对数收益（价格）
    "log_volume_3s",               // 12 桶内成交量 log1p（量，新增）
    "trade_arrival_clustering",    // 13 桶内逐笔间隔CV（聚集度，新增）
];

// ---------------------------------------------------------------------------
// 模型定义
// ---------------------------------------------------------------------------

pub struct ModelDef {
    pub name: &'static str,
    pub factors: [usize; MAX_FACTORS],
    pub k: usize,
    pub ys: &'static [usize],
}

/// 5 套模型。y 清单 = 数据探索校准后的 53 组合（同源变体 + 稳定高相关剔除，
/// 核心收益 y=11 全保留）。
pub const MODELS: [ModelDef; 5] = [
    ModelDef {
        name: "T1",
        factors: [0, 12, 2, 0, 0],
        k: 3,
        ys: &[0, 1, 2, 3, 5, 6, 8, 9, 11, 12],
    },
    ModelDef {
        name: "T2",
        factors: [1, 9, 6, 0, 0],
        k: 3,
        ys: &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    },
    ModelDef {
        name: "T3",
        factors: [3, 7, 13, 0, 0],
        k: 3,
        ys: &[0, 1, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13],
    },
    ModelDef {
        name: "F1",
        factors: [0, 12, 13, 2, 1],
        k: 5,
        ys: &[0, 1, 2, 3, 5, 6, 8, 11, 12, 13],
    },
    ModelDef {
        name: "F2",
        factors: [0, 12, 2, 6, 3],
        k: 5,
        ys: &[0, 1, 2, 3, 6, 8, 9, 11, 12],
    },
];

/// 组合总数 = 53。
pub const N_COMBOS: usize = 10 + 12 + 12 + 10 + 9;

/// 总列数（39/55 列 × 21 统计）→ N_FACTORS。
pub fn total_factor_count() -> usize {
    let mut n_cols_total = 0usize;
    for m in MODELS.iter() {
        n_cols_total += m.ys.len() * n_cols(m.k);
    }
    n_cols_total * 21
}

pub const N_FACTORS: usize = 49_791;

// ---------------------------------------------------------------------------
// 列布局（每组合，K = 因子数）：共 15 + 8K 列
// ---------------------------------------------------------------------------

// 列布局：per-stock 列（4K+6）靠前，共享列（4K+9）靠后。
// per-stock: beta(K) | beta_t(K) | resid 6 | beta_shift(K) | lambda_shift(K)
// shared: r2 | adj_r2 | alpha | lambda(K) | f_stat | vif(K) | cond |
//         delta_r2 | resid_improve | alpha_shift | resid_corr | nested_f(K) | nested_p(K)
#[inline]
pub fn n_cols(k: usize) -> usize {
    15 + 8 * k
}
#[inline]
pub fn col_beta(k: usize) -> usize {
    0
}
#[inline]
pub fn col_beta_t(k: usize) -> usize {
    k
}
#[inline]
pub fn col_resid(k: usize) -> usize {
    2 * k
}
#[inline]
pub fn col_resid_z(k: usize) -> usize {
    2 * k + 1
}
#[inline]
pub fn col_resid_rank(k: usize) -> usize {
    2 * k + 2
}
#[inline]
pub fn col_resid_abs(k: usize) -> usize {
    2 * k + 3
}
#[inline]
pub fn col_leverage(k: usize) -> usize {
    2 * k + 4
}
#[inline]
pub fn col_cooks(k: usize) -> usize {
    2 * k + 5
}
#[inline]
pub fn col_beta_shift(k: usize) -> usize {
    2 * k + 6
}
#[inline]
pub fn col_lambda_shift(k: usize) -> usize {
    3 * k + 6
}
#[inline]
pub fn col_r2(k: usize) -> usize {
    4 * k + 6
}
#[inline]
pub fn col_adj_r2(k: usize) -> usize {
    4 * k + 7
}
#[inline]
pub fn col_alpha(k: usize) -> usize {
    4 * k + 8
}
#[inline]
pub fn col_lambda(k: usize) -> usize {
    4 * k + 9
}
#[inline]
pub fn col_f_stat(k: usize) -> usize {
    5 * k + 9
}
#[inline]
pub fn col_vif(k: usize) -> usize {
    5 * k + 10
}
#[inline]
pub fn col_cond(k: usize) -> usize {
    6 * k + 10
}
#[inline]
pub fn col_delta_r2(k: usize) -> usize {
    6 * k + 11
}
#[inline]
pub fn col_resid_improve(k: usize) -> usize {
    6 * k + 12
}
#[inline]
pub fn col_alpha_shift(k: usize) -> usize {
    6 * k + 13
}
#[inline]
pub fn col_resid_corr(k: usize) -> usize {
    6 * k + 14
}
#[inline]
pub fn col_nested_f(k: usize) -> usize {
    6 * k + 15
}
#[inline]
pub fn col_nested_p(k: usize) -> usize {
    7 * k + 15
}

/// 组合的列名（stat 部分，与 col_* 布局一致：per-stock 在前，shared 在后）。
pub fn col_names(k: usize, factors: &[usize; MAX_FACTORS]) -> Vec<String> {
    let mut out = Vec::with_capacity(n_cols(k));
    for i in 0..k {
        out.push(format!("beta_f{}", factors[i]));
    }
    for i in 0..k {
        out.push(format!("beta_t_f{}", factors[i]));
    }
    out.push("resid".into());
    out.push("resid_zscore".into());
    out.push("resid_rank".into());
    out.push("resid_abs".into());
    out.push("leverage".into());
    out.push("cooks".into());
    for i in 0..k {
        out.push(format!("beta_shift_f{}", factors[i]));
    }
    for i in 0..k {
        out.push(format!("lambda_shift_f{}", factors[i]));
    }
    out.push("r2".into());
    out.push("adj_r2".into());
    out.push("alpha".into());
    for i in 0..k {
        out.push(format!("lambda_f{}", factors[i]));
    }
    out.push("f_stat".into());
    for i in 0..k {
        out.push(format!("vif_f{}", factors[i]));
    }
    out.push("cond".into());
    out.push("delta_r2".into());
    out.push("resid_improve".into());
    out.push("alpha_shift".into());
    out.push("resid_corr".into());
    for i in 0..k {
        out.push(format!("nested_f_f{}", factors[i]));
    }
    for i in 0..k {
        out.push(format!("nested_p_f{}", factors[i]));
    }
    out
}

// ---------------------------------------------------------------------------
// 3 秒网格（14 指标）
// ---------------------------------------------------------------------------

fn grid_start_us_for_date(date: i64) -> io::Result<i64> {
    let year = (date / 10_000) as i32;
    let month = ((date / 100) % 100) as u32;
    let day = (date % 100) as u32;
    let midnight_us = NaiveDate::from_ymd_opt(year, month, day)
        .and_then(|value| value.and_hms_opt(0, 0, 0))
        .ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidInput, format!("无效交易日期: {date}"))
        })?
        .and_utc()
        .timestamp_micros();
    Ok(midnight_us + MARKET_OPEN_US)
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
    let mut out = vec![f32::NAN; N_BINS * N_FEATURES];

    let mut buy_volume = vec![0.0f64; N_BINS];
    let mut sell_volume = vec![0.0f64; N_BINS];
    let mut gap_signed_num = vec![0.0f64; N_BINS];
    let mut gap_abs_num = vec![0.0f64; N_BINS];
    let mut gap_weight = vec![0.0f64; N_BINS];
    let mut large_signed = vec![0.0f64; N_BINS];
    let mut large_total = vec![0.0f64; N_BINS];
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
        out[bin * N_FEATURES + 0] = ratio(buy_volume[bin], total_volume);
        out[bin * N_FEATURES + 1] = ratio(gap_signed_num[bin], gap_weight[bin]);
        out[bin * N_FEATURES + 9] = ratio(gap_abs_num[bin], gap_weight[bin]);
        out[bin * N_FEATURES + 10] = ratio(large_signed[bin], large_total[bin]);
        out[bin * N_FEATURES + 12] = if total_volume > 0.0 {
            (total_volume).ln_1p() as f32
        } else {
            f32::NAN
        };
        let n = bin_ticks[bin];
        if n >= 2 {
            let m = (n - 1) as f64;
            let mean = bin_sum_dt[bin] / m;
            if mean > 0.0 {
                let var = (bin_sum_dt2[bin] / m - mean * mean).max(0.0);
                out[bin * N_FEATURES + 13] = (var.sqrt() / mean) as f32;
            }
        }
    }

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
        let base = bin * N_FEATURES;
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
            let idx = bin * N_FEATURES + feature;
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
            let idx = bin * N_FEATURES + 11;
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
        let base = bin * N_FEATURES;
        let prev = (bin - 1) * N_FEATURES;
        if out[base + 2].is_finite() && out[prev + 2].is_finite() {
            out[base + 4] = out[base + 2] - out[prev + 2];
        }
        if out[base + 3].is_finite() && out[prev + 3].is_finite() {
            out[base + 5] = out[base + 3] - out[prev + 3];
        }
    }

    for (session_start, session_end) in [(0usize, MIDDAY_BIN), (MIDDAY_BIN, N_BINS)] {
        let mut previous = out[session_start * N_FEATURES + 11];
        out[session_start * N_FEATURES + 11] = f32::NAN;
        for bin in session_start + 1..session_end {
            let idx = bin * N_FEATURES + 11;
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

// ===========================================================================
// 以下核心计算与 sandbox_multi_factor/src/multi_factor.rs 一致
// ===========================================================================

// 滚动矩
#[derive(Clone, Copy, Default)]
struct Rolling6 {
    n: u32,
    sx: f64,
    sy: f64,
    sxx: f64,
    syy: f64,
    sxy: f64,
}

#[derive(Clone, Copy, Default)]
struct Cross2 {
    n: u32,
    sx: f64,
    sy: f64,
    sxy: f64,
}

#[derive(Clone, Copy, Default)]
struct StockState {
    pairs: [Rolling6; N_FEATURES],
    cross: [Cross2; 23],
}

fn build_cross_pairs() -> (Vec<(usize, usize)>, [[i8; N_FEATURES]; N_FEATURES]) {
    let mut set = std::collections::BTreeSet::new();
    for m in MODELS.iter() {
        for a in 0..m.k {
            for b in (a + 1)..m.k {
                let (i, j) = if m.factors[a] < m.factors[b] {
                    (m.factors[a], m.factors[b])
                } else {
                    (m.factors[b], m.factors[a])
                };
                set.insert((i, j));
            }
        }
    }
    let pairs: Vec<(usize, usize)> = set.into_iter().collect();
    let mut table = [[-1i8; N_FEATURES]; N_FEATURES];
    for (idx, &(i, j)) in pairs.iter().enumerate() {
        table[i][j] = idx as i8;
        table[j][i] = idx as i8;
    }
    (pairs, table)
}

#[inline]
fn add_pair(s: &mut Rolling6, x: f64, y: f64, sign: f64) {
    if !x.is_finite() || !y.is_finite() {
        return;
    }
    if sign > 0.0 {
        s.n += 1;
    } else {
        s.n = s.n.saturating_sub(1);
    }
    s.sx += sign * x;
    s.sy += sign * y;
    s.sxx += sign * x * x;
    s.syy += sign * y * y;
    s.sxy += sign * x * y;
}

#[inline]
fn add_cross(s: &mut Cross2, x: f64, y: f64, y_valid: bool, sign: f64) {
    if !y_valid || !x.is_finite() || !y.is_finite() {
        return;
    }
    if sign > 0.0 {
        s.n += 1;
    } else {
        s.n = s.n.saturating_sub(1);
    }
    s.sx += sign * x;
    s.sy += sign * y;
    s.sxy += sign * x * y;
}

fn exposure(p: &Rolling6) -> Option<(f64, f64, f64, f64, u32)> {
    if p.n < MIN_HISTORY_OBS {
        return None;
    }
    let n = p.n as f64;
    let sxx = p.sxx - p.sx * p.sx / n;
    let syy = p.syy - p.sy * p.sy / n;
    let sxy = p.sxy - p.sx * p.sy / n;
    if sxx <= 1e-18 || syy <= 1e-18 {
        return None;
    }
    let beta = sxy / sxx;
    let alpha = p.sy / n - beta * p.sx / n;
    let corr = (sxy / (sxx * syy).sqrt()).clamp(-1.0, 1.0);
    let sse = (syy - beta * sxy).max(0.0);
    Some((beta, alpha, corr, (sse / (n - 2.0)).sqrt(), p.n))
}

// 小矩阵工具（K ≤ 5，统一 MAX_FACTORS 步长）

fn cholesky(a: &[f64], n: usize, out: &mut [f64]) -> bool {
    out[..n * n].fill(0.0);
    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[i * MAX_FACTORS + j];
            for k in 0..j {
                sum -= out[i * MAX_FACTORS + k] * out[j * MAX_FACTORS + k];
            }
            if i == j {
                if sum <= 1e-14 {
                    return false;
                }
                out[i * MAX_FACTORS + i] = sum.sqrt();
            } else {
                out[i * MAX_FACTORS + j] = sum / out[j * MAX_FACTORS + j];
            }
        }
    }
    true
}

fn cholesky_solve(l: &[f64], n: usize, b: &[f64], x: &mut [f64]) {
    let mut z = [0.0f64; MAX_FACTORS];
    for i in 0..n {
        let mut sum = b[i];
        for k in 0..i {
            sum -= l[i * MAX_FACTORS + k] * z[k];
        }
        z[i] = sum / l[i * MAX_FACTORS + i];
    }
    for i in (0..n).rev() {
        let mut sum = z[i];
        for k in (i + 1)..n {
            sum -= l[k * MAX_FACTORS + i] * x[k];
        }
        x[i] = sum / l[i * MAX_FACTORS + i];
    }
}

fn invert_sym(a: &[f64], n: usize, out: &mut [f64]) -> bool {
    let w = 2 * n;
    let mut m = vec![0.0f64; n * w];
    for i in 0..n {
        for j in 0..n {
            m[i * w + j] = a[i * MAX_FACTORS + j];
        }
        m[i * w + n + i] = 1.0;
    }
    for col in 0..n {
        let mut piv = col;
        let mut best = m[col * w + col].abs();
        for r in col + 1..n {
            let v = m[r * w + col].abs();
            if v > best {
                best = v;
                piv = r;
            }
        }
        if best <= 1e-14 {
            return false;
        }
        if piv != col {
            for c in 0..w {
                m.swap(col * w + c, piv * w + c);
            }
        }
        let d = m[col * w + col];
        for c in 0..w {
            m[col * w + c] /= d;
        }
        for r in 0..n {
            if r != col {
                let f = m[r * w + col];
                if f != 0.0 {
                    for c in 0..w {
                        m[r * w + c] -= f * m[col * w + c];
                    }
                }
            }
        }
    }
    for i in 0..n {
        for j in 0..n {
            out[i * MAX_FACTORS + j] = m[i * w + n + j];
        }
    }
    true
}

fn eigen_sym(a: &[f64], n: usize, vals: &mut [f64]) {
    let mut v = [0.0f64; MAX_FACTORS * MAX_FACTORS];
    for i in 0..n {
        v[i * MAX_FACTORS + i] = 1.0;
    }
    let mut a = a.to_vec();
    for _iter in 0..64 {
        let mut p = 0;
        let mut q = 1;
        let mut max_off = 0.0f64;
        for i in 0..n {
            for j in (i + 1)..n {
                let x = a[i * MAX_FACTORS + j].abs();
                if x > max_off {
                    max_off = x;
                    p = i;
                    q = j;
                }
            }
        }
        if max_off < 1e-14 {
            break;
        }
        let app = a[p * MAX_FACTORS + p];
        let aqq = a[q * MAX_FACTORS + q];
        let apq = a[p * MAX_FACTORS + q];
        // Numerical Recipes 标准参数化：tau = (aqq-app)/(2apq)，t=tan(φ)
        let tau = (aqq - app) / (2.0 * apq);
        let mut t = 1.0 / (tau.abs() + (1.0 + tau * tau).sqrt());
        if tau < 0.0 {
            t = -t;
        }
        let c = 1.0 / (t * t + 1.0).sqrt();
        let s = t * c;
        for k in 0..n {
            if k == p || k == q {
                continue;
            }
            let akp = a[k * MAX_FACTORS + p];
            let akq = a[k * MAX_FACTORS + q];
            a[k * MAX_FACTORS + p] = c * akp - s * akq;
            a[p * MAX_FACTORS + k] = a[k * MAX_FACTORS + p];
            a[k * MAX_FACTORS + q] = s * akp + c * akq;
            a[q * MAX_FACTORS + k] = a[k * MAX_FACTORS + q];
            let vkp = v[k * MAX_FACTORS + p];
            let vkq = v[k * MAX_FACTORS + q];
            v[k * MAX_FACTORS + p] = c * vkp - s * vkq;
            v[k * MAX_FACTORS + q] = s * vkp + c * vkq;
        }
        a[p * MAX_FACTORS + p] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
        a[q * MAX_FACTORS + q] = s * s * app + 2.0 * s * c * apq + c * c * aqq;
        a[p * MAX_FACTORS + q] = 0.0;
        a[q * MAX_FACTORS + p] = 0.0;
    }
    for i in 0..n {
        vals[i] = a[i * MAX_FACTORS + i];
    }
    vals[..n].sort_by(|x, y| x.total_cmp(y));
}

fn multi_exposure(
    pairs: &[Rolling6; N_FEATURES],
    cross: &[Cross2; 23],
    cross_idx: &[[i8; N_FEATURES]; N_FEATURES],
    factors: &[usize; MAX_FACTORS],
    k: usize,
    out_beta: &mut [f64],
) -> Option<(f64, u32)> {
    let p0 = &pairs[factors[0]];
    if p0.n < MIN_HISTORY_OBS {
        return None;
    }
    let n = p0.n as f64;
    let mut xtx = [0.0f64; MAX_FACTORS * MAX_FACTORS];
    let mut xty = [0.0f64; MAX_FACTORS];
    for i in 0..k {
        let fi = factors[i];
        let pi = &pairs[fi];
        xty[i] = pi.sxy - pi.sx * p0.sy / n;
        for j in 0..k {
            let fj = factors[j];
            let pj = &pairs[fj];
            let sxy = if i == j {
                pi.sxx
            } else {
                let idx = cross_idx[fi][fj];
                if idx < 0 {
                    return None;
                }
                cross[idx as usize].sxy
            };
            xtx[i * MAX_FACTORS + j] = sxy - pi.sx * pj.sx / n;
        }
    }
    let syy_c = p0.syy - p0.sy * p0.sy / n;
    if syy_c <= 1e-18 {
        return None;
    }
    let mut l = [0.0f64; MAX_FACTORS * MAX_FACTORS];
    if !cholesky(&xtx, k, &mut l) {
        return None;
    }
    cholesky_solve(&l, k, &xty, out_beta);
    let mut sse = syy_c;
    for i in 0..k {
        sse -= out_beta[i] * xty[i];
    }
    if sse < 0.0 {
        sse = 0.0;
    }
    let dof = n - k as f64 - 1.0;
    if dof <= 0.0 {
        return None;
    }
    Some((sse / dof, p0.n))
}

fn cs_single(
    y: &[f32],
    beta: &[f64],
    n_stocks: usize,
    force_valid: Option<&[usize]>,
) -> Option<(f64, f64, f64, f64, usize)> {
    let (mut n, mut sx, mut sy, mut sxx, mut syy, mut sxy) = (0usize, 0.0f64, 0.0, 0.0, 0.0, 0.0);
    match force_valid {
        Some(valid) => {
            for &s in valid {
                let b = beta[s];
                let v = y[s] as f64;
                if b.is_finite() && v.is_finite() {
                    n += 1;
                    sx += b;
                    sy += v;
                    sxx += b * b;
                    syy += v * v;
                    sxy += b * v;
                }
            }
        }
        None => {
            for s in 0..n_stocks {
                let b = beta[s];
                let v = y[s] as f64;
                if b.is_finite() && v.is_finite() {
                    n += 1;
                    sx += b;
                    sy += v;
                    sxx += b * b;
                    syy += v * v;
                    sxy += b * v;
                }
            }
        }
    }
    if n < MIN_CS_STOCKS {
        return None;
    }
    let nf = n as f64;
    let sxx_c = sxx - sx * sx / nf;
    let syy_c = syy - sy * sy / nf;
    let sxy_c = sxy - sx * sy / nf;
    if sxx_c <= 1e-18 || syy_c <= 1e-18 {
        return None;
    }
    let lambda = sxy_c / sxx_c;
    let alpha = sy / nf - lambda * sx / nf;
    let sse = (syy_c - lambda * sxy_c).max(0.0);
    let r2 = 1.0 - sse / syy_c;
    Some((alpha, lambda, r2, sse, n))
}

pub struct CsResult {
    pub n: usize,
    pub alpha: f64,
    pub lambda: [f64; MAX_FACTORS],
    pub r2: f64,
    pub adj_r2: f64,
    pub f_stat: f64,
    pub sse: f64,
    pub residual_std: f64,
    pub cond: f64,
    pub vif: [f64; MAX_FACTORS],
    pub valid: Vec<usize>,
    pub inv: [f64; MAX_FACTORS * MAX_FACTORS],
    pub syy_c: f64,
    pub beta: [f64; MAX_FACTORS],
    pub se: [f64; MAX_FACTORS],
}

impl Default for CsResult {
    fn default() -> Self {
        CsResult {
            n: 0,
            alpha: f64::NAN,
            lambda: [f64::NAN; MAX_FACTORS],
            r2: f64::NAN,
            adj_r2: f64::NAN,
            f_stat: f64::NAN,
            sse: f64::NAN,
            residual_std: f64::NAN,
            cond: f64::NAN,
            vif: [f64::NAN; MAX_FACTORS],
            valid: Vec::new(),
            inv: [0.0; MAX_FACTORS * MAX_FACTORS],
            syy_c: f64::NAN,
            beta: [f64::NAN; MAX_FACTORS],
            se: [f64::NAN; MAX_FACTORS],
        }
    }
}

fn cs_multi(y: &[f32], betas: &[f64], k: usize, n_stocks: usize, out: &mut CsResult) -> bool {
    let (mut n, mut sy, mut syy) = (0usize, 0.0f64, 0.0);
    let mut sbeta = [0.0f64; MAX_FACTORS];
    let mut sbb = [0.0f64; MAX_FACTORS * MAX_FACTORS];
    let mut sby = [0.0f64; MAX_FACTORS];
    // 复用 out.valid 缓冲（跨桶循环复用，避免每桶一次堆分配）
    out.valid.clear();
    for s in 0..n_stocks {
        let v = y[s] as f64;
        if !v.is_finite() {
            continue;
        }
        let mut ok = true;
        for i in 0..k {
            if !betas[i * n_stocks + s].is_finite() {
                ok = false;
                break;
            }
        }
        if !ok {
            continue;
        }
        out.valid.push(s);
        n += 1;
        sy += v;
        syy += v * v;
        for i in 0..k {
            let b = betas[i * n_stocks + s];
            sbeta[i] += b;
            sby[i] += b * v;
            for j in 0..k {
                sbb[i * MAX_FACTORS + j] += b * betas[j * n_stocks + s];
            }
        }
    }
    if n < MIN_CS_STOCKS {
        return false;
    }
    let nf = n as f64;
    let syy_c = syy - sy * sy / nf;
    if syy_c <= 1e-18 {
        return false;
    }
    let mut xtx = [0.0f64; MAX_FACTORS * MAX_FACTORS];
    let mut xty = [0.0f64; MAX_FACTORS];
    for i in 0..k {
        xty[i] = sby[i] - sbeta[i] * sy / nf;
        for j in 0..k {
            xtx[i * MAX_FACTORS + j] = sbb[i * MAX_FACTORS + j] - sbeta[i] * sbeta[j] / nf;
        }
    }
    let mut l = [0.0f64; MAX_FACTORS * MAX_FACTORS];
    if !cholesky(&xtx, k, &mut l) {
        return false;
    }
    let mut beta = [0.0f64; MAX_FACTORS];
    cholesky_solve(&l, k, &xty, &mut beta);
    // 截距：alpha = ȳ - Σ λ_k · β̄_k（中心化回归的截距还原）
    let mut alpha = sy / nf;
    for i in 0..k {
        alpha -= beta[i] * sbeta[i] / nf;
    }
    let mut sse = syy_c;
    for i in 0..k {
        sse -= beta[i] * xty[i];
    }
    if sse < 0.0 {
        sse = 0.0;
    }
    let dof = nf - k as f64 - 1.0;
    let r2 = 1.0 - sse / syy_c;
    let adj_r2 = 1.0 - (1.0 - r2) * (nf - 1.0) / dof;
    let f_stat = if 1.0 - r2 > 1e-18 {
        r2 / (1.0 - r2) * dof / k as f64
    } else {
        f64::INFINITY
    };
    let residual_std = (sse / dof).sqrt();
    let mut inv = [0.0f64; MAX_FACTORS * MAX_FACTORS];
    if !invert_sym(&xtx, k, &mut inv) {
        return false;
    }
    let mut corr = [0.0f64; MAX_FACTORS * MAX_FACTORS];
    let mut sd = [0.0f64; MAX_FACTORS];
    for i in 0..k {
        let v = xtx[i * MAX_FACTORS + i].max(0.0).sqrt();
        sd[i] = if v > 1e-18 { v } else { f64::NAN };
    }
    for i in 0..k {
        for j in 0..k {
            let d = sd[i] * sd[j];
            corr[i * MAX_FACTORS + j] = if d > 0.0 {
                xtx[i * MAX_FACTORS + j] / d
            } else {
                0.0
            };
        }
    }
    let mut corr_inv = [0.0f64; MAX_FACTORS * MAX_FACTORS];
    let mut vif = [f64::NAN; MAX_FACTORS];
    if invert_sym(&corr, k, &mut corr_inv) {
        for i in 0..k {
            vif[i] = corr_inv[i * MAX_FACTORS + i].max(0.0);
        }
    }
    let mut ev = [0.0f64; MAX_FACTORS];
    eigen_sym(&corr, k, &mut ev);
    let cond = if ev[0] > 1e-18 {
        ev[k - 1] / ev[0]
    } else {
        f64::INFINITY
    };
    out.n = n;
    out.alpha = alpha;
    for i in 0..k {
        out.lambda[i] = beta[i];
        out.vif[i] = vif[i];
    }
    out.r2 = r2;
    out.adj_r2 = adj_r2;
    out.f_stat = f_stat;
    out.sse = sse;
    out.residual_std = residual_std;
    out.cond = cond;
    out.inv = inv;
    out.syy_c = syy_c;
    out.beta = beta;
    for i in 0..k {
        let se = inv[i * MAX_FACTORS + i].max(0.0).sqrt() * residual_std;
        out.se[i] = se;
    }
    true
}

fn f_pvalue(f: f64, d1: f64, _d2: f64) -> f64 {
    if !f.is_finite() || f <= 0.0 {
        return f64::NAN;
    }
    let x = f * d1;
    let nu = d1;
    let z = ((x / nu).cbrt() - (1.0 - 2.0 / (9.0 * nu))) / (2.0 / (9.0 * nu)).sqrt();
    let zz = z.abs();
    let p = 0.231_641_9;
    let b1 = 0.319_381_530;
    let b2 = -0.356_563_782;
    let b3 = 1.781_477_937;
    let b4 = -1.821_255_978;
    let b5 = 1.330_274_429;
    let u = 1.0 / (1.0 + p * zz);
    let phi = (-0.5 * zz * zz).exp() / (2.0 * std::f64::consts::PI).sqrt();
    let upper = phi * (b1 * u + b2 * u.powi(2) + b3 * u.powi(3) + b4 * u.powi(4) + b5 * u.powi(5));
    upper.clamp(0.0, 1.0)
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

// ---------------------------------------------------------------------------
// 组合缓冲
// ---------------------------------------------------------------------------

/// 一个 (模型, y) 组合的时序缓冲。
/// per-stock 列按 [股票] 展开；共享列（横截面全局量）只存一份。
pub struct ComboBuf {
    pub n_per_stock: usize,
    pub n_shared: usize,
    pub n_stocks: usize,
    pub per_stock: Vec<f32>,
    pub shared: Vec<f32>,
}

impl Default for ComboBuf {
    fn default() -> Self {
        ComboBuf {
            n_per_stock: 0,
            n_shared: 0,
            n_stocks: 0,
            per_stock: Vec::new(),
            shared: Vec::new(),
        }
    }
}

fn madvise_huge(buf: &mut Vec<f32>) {
    // MADV_HUGEPAGE 仅 Linux 提供（macOS/BSD 无此常量）；非 Linux 直接 no-op。
    // 纯性能提示（大页透明化），不影响正确性。
    if buf.is_empty() {
        return;
    }
    #[cfg(target_os = "linux")]
    unsafe {
        let raw = buf.as_mut_ptr() as usize;
        let page = 4096usize;
        let aligned = (raw + page - 1) & !(page - 1);
        let adj = aligned - raw;
        if buf.len() * 4 > adj {
            libc::madvise(
                aligned as *mut libc::c_void,
                buf.len() * 4 - adj,
                libc::MADV_HUGEPAGE,
            );
        }
    }
}

impl ComboBuf {
    pub(crate) fn new(n_per_stock: usize, n_shared: usize, n_stocks: usize) -> Self {
        let mut per_stock = Vec::with_capacity(n_per_stock * N_BINS * n_stocks);
        madvise_huge(&mut per_stock);
        per_stock.resize(n_per_stock * N_BINS * n_stocks, f32::NAN);
        let mut shared = Vec::with_capacity(n_shared * N_BINS);
        madvise_huge(&mut shared);
        shared.resize(n_shared * N_BINS, f32::NAN);
        ComboBuf {
            n_per_stock,
            n_shared,
            n_stocks,
            per_stock,
            shared,
        }
    }
    #[inline]
    fn cols(&self) -> usize {
        self.n_per_stock + self.n_shared
    }
    #[inline]
    fn write_ps(&mut self, col: usize, bin: usize, stock: usize, v: f32) {
        // 布局 [stock][bin][col_ps]：col 最内层，扫桶写入连续、21 统计拷贝连续
        let n_ps = self.n_per_stock;
        self.per_stock[((stock * N_BINS + bin) * n_ps) + col] = v;
    }
    #[inline]
    fn write_sh(&mut self, col: usize, bin: usize, v: f32) {
        self.shared[((col - self.n_per_stock) * N_BINS) + bin] = v;
    }
    #[inline]
    fn read(&self, col: usize, bin: usize, stock: usize) -> f32 {
        if col < self.n_per_stock {
            let n_ps = self.n_per_stock;
            self.per_stock[((stock * N_BINS + bin) * n_ps) + col]
        } else {
            self.shared[((col - self.n_per_stock) * N_BINS) + bin]
        }
    }
}

/// 预分配全部 53 个组合缓冲（页错误与读盘/其他计算重叠）。
pub fn prealloc_combos(n_stocks: usize) -> Vec<ComboBuf> {
    // 并行分配：大块（2-3GB/个）走 glibc mmap，无 malloc arena 争抢；
    // touch（resize 填 NaN）也并行，50 线程下远快于单线程顺序 touch。
    // 每组合大小 = (4k+6) × 4740 × n_stocks × 4B：k=3 约 2.05GB，k=5 约 2.97GB。
    let mut layout: Vec<(usize, usize)> = Vec::with_capacity(N_COMBOS);
    for m in MODELS.iter() {
        for _ in m.ys.iter() {
            layout.push((4 * m.k + 6, 4 * m.k + 9));
        }
    }
    layout
        .into_par_iter()
        .map(|(n_ps, n_sh)| ComboBuf::new(n_ps, n_sh, n_stocks))
        .collect()
}

/// 滚动状态按 y 共享：同一 y 的多个 (模型) 组合共用一份滚动矩与单因子暴露。
/// 滚动更新（14+23 对矩）与 exposure 从 53 份重复计算降为 14 份，
/// 且 14 路工作集约 119MB 可驻留 L3（旧版 53 路约 450MB 导致 L3 抖动）。
/// 每组合看到的状态更新序列与旧版逐位一致 → 输出逐位一致。

/// 每 (路由, 组合) 的桶级复用缓冲。
struct ComboScratch {
    mbeta: Vec<f64>, // [MAX_FACTORS * n_stocks]
    cs: CsResult,
    rank_pairs: Vec<(usize, f64)>,
    ranks: Vec<f64>,
    tmp_resid: Vec<f64>,
    tmp_z: Vec<f64>,
    tmp_h: Vec<f64>,
    tmp_cooks: Vec<f64>,
    bmean: [f64; MAX_FACTORS],
    sse_sk: [f64; MAX_FACTORS],
    lam_sk: [f64; MAX_FACTORS],
}

impl ComboScratch {
    fn new(n_stocks: usize) -> Self {
        ComboScratch {
            mbeta: vec![f64::NAN; MAX_FACTORS * n_stocks],
            cs: CsResult::default(),
            rank_pairs: Vec::with_capacity(n_stocks),
            ranks: vec![f64::NAN; n_stocks],
            tmp_resid: vec![f64::NAN; n_stocks],
            tmp_z: vec![f64::NAN; n_stocks],
            tmp_h: vec![f64::NAN; n_stocks],
            tmp_cooks: vec![f64::NAN; n_stocks],
            bmean: [f64::NAN; MAX_FACTORS],
            sse_sk: [f64::NAN; MAX_FACTORS],
            lam_sk: [f64::NAN; MAX_FACTORS],
        }
    }
}

impl Default for ComboScratch {
    fn default() -> Self {
        ComboScratch::new(0)
    }
}

/// 处理单个 (模型, y) 组合：独立滚动矩 + 桶循环。
pub(crate) fn compute_one_combo(
    signals: &[f32],
    market: &[f64],
    n_stocks: usize,
    y: usize,
    mi: usize,
    mut buf: ComboBuf,
) -> ComboBuf {
    let m = &MODELS[mi];
    let k = m.k;
    // 只更新本模型实际用到的交叉对（T1/T2/T3 各 3 对、F1/F2 各 10 对，
    // 平均 5.5 对 vs 全局 23 对）——未更新的交叉槽本组合从不读取，输出逐位一致。
    let (_, cross_idx) = build_cross_pairs();
    let mut cross_set = std::collections::BTreeSet::new();
    for a in 0..k {
        for b in (a + 1)..k {
            let (i, j) = if m.factors[a] < m.factors[b] {
                (m.factors[a], m.factors[b])
            } else {
                (m.factors[b], m.factors[a])
            };
            cross_set.insert((i, j));
        }
    }
    let cross_pairs: Vec<(usize, usize)> = cross_set.into_iter().collect();
    let mut states = vec![StockState::default(); n_stocks];
    let mut beta1 = vec![f64::NAN; N_FEATURES * n_stocks];
    let mut mbeta = vec![f64::NAN; MAX_FACTORS * n_stocks];
    let mut cs = CsResult::default();
    let mut rank_pairs: Vec<(usize, f64)> = Vec::with_capacity(n_stocks);
    let mut ranks = vec![f64::NAN; n_stocks];
    let mut tmp_resid = vec![f64::NAN; n_stocks];
    let mut tmp_z = vec![f64::NAN; n_stocks];
    let mut tmp_h = vec![f64::NAN; n_stocks];
    let mut tmp_cooks = vec![f64::NAN; n_stocks];

    let ysig_base = y * N_BINS * n_stocks;

    for bin in 0..N_BINS {
        if bin == MIDDAY_BIN {
            states.fill(StockState::default());
        }
        if bin >= ROLLING_WINDOW {
            let old = bin - ROLLING_WINDOW;
            if !(old < MIDDAY_BIN && bin >= MIDDAY_BIN) {
                let sign = -1.0;
                for s in 0..n_stocks {
                    let yv = signals[ysig_base + old * n_stocks + s] as f64;
                    let st = &mut states[s];
                    for f in 0..N_FEATURES {
                        add_pair(&mut st.pairs[f], market[f * N_BINS + old], yv, sign);
                    }
                    if yv.is_finite() {
                        for &(i, j) in cross_pairs.iter() {
                            add_cross(
                                &mut st.cross[cross_idx[i][j] as usize],
                                market[i * N_BINS + old],
                                market[j * N_BINS + old],
                                true,
                                sign,
                            );
                        }
                    }
                }
            }
        }
        for s in 0..n_stocks {
            let st = &states[s];
            for f in 0..N_FEATURES {
                beta1[f * n_stocks + s] = match exposure(&st.pairs[f]) {
                    Some((b, _, _, _, _)) => b,
                    None => f64::NAN,
                };
            }
        }
        let ycur_base = ysig_base + bin * n_stocks;
        for s in 0..n_stocks {
            let mut bvec = [0.0f64; MAX_FACTORS];
            if let Some((_mse, _n)) = multi_exposure(
                &states[s].pairs,
                &states[s].cross,
                &cross_idx,
                &m.factors,
                k,
                &mut bvec,
            ) {
                for i in 0..k {
                    mbeta[i * n_stocks + s] = bvec[i];
                }
            } else {
                for i in 0..k {
                    mbeta[i * n_stocks + s] = f64::NAN;
                }
            }
        }
        let ycur = &signals[ycur_base..ycur_base + n_stocks];
        let beta_yy = &beta1[y * n_stocks..(y + 1) * n_stocks];
        let s_fit = cs_single(ycur, beta_yy, n_stocks, None);
        let ok = cs_multi(ycur, &mbeta, k, n_stocks, &mut cs);
        if !ok {
            let sign = 1.0;
            for s in 0..n_stocks {
                let yv = signals[ysig_base + bin * n_stocks + s] as f64;
                let st = &mut states[s];
                for f in 0..N_FEATURES {
                    add_pair(&mut st.pairs[f], market[f * N_BINS + bin], yv, sign);
                }
                if yv.is_finite() {
                    for &(i, j) in cross_pairs.iter() {
                        add_cross(
                            &mut st.cross[cross_idx[i][j] as usize],
                            market[i * N_BINS + bin],
                            market[j * N_BINS + bin],
                            true,
                            sign,
                        );
                    }
                }
            }
            continue;
        }
        let mut sse_sk = [f64::NAN; MAX_FACTORS];
        let mut lam_sk = [f64::NAN; MAX_FACTORS];
        for i in 0..k {
            let fi = m.factors[i];
            let b = &beta1[fi * n_stocks..(fi + 1) * n_stocks];
            if let Some((_a, l, _r2, sse, _n)) = cs_single(ycur, b, n_stocks, Some(&cs.valid)) {
                sse_sk[i] = sse;
                lam_sk[i] = l;
            }
        }
        let n = cs.n;
        let nf = n as f64;
        let dof = nf - k as f64 - 1.0;
        let (r2_s, sse_s, alpha_s) = match s_fit {
            Some((a, _l, r2, sse, _n)) => (r2, sse, a),
            None => (f64::NAN, f64::NAN, f64::NAN),
        };
        let residual_std = cs.residual_std;
        let mut bmean = [0.0f64; MAX_FACTORS];
        for i in 0..k {
            let mut sm = 0.0;
            for &s in &cs.valid {
                sm += mbeta[i * n_stocks + s];
            }
            bmean[i] = sm / nf;
        }
        let mut sum_m = 0.0f64;
        let mut sum_s = 0.0f64;
        let mut sum_mm = 0.0f64;
        let mut sum_ss = 0.0f64;
        let mut sum_ms = 0.0f64;
        let mut cnt_r = 0usize;
        rank_pairs.clear();
        for &s in &cs.valid {
            let yv = ycur[s] as f64;
            let mut fitted = cs.alpha;
            for i in 0..k {
                fitted += cs.lambda[i] * mbeta[i * n_stocks + s];
            }
            let resid = yv - fitted;
            rank_pairs.push((s, resid));
            if let Some((a, l, _, _, _)) = s_fit {
                let rs = yv - (a + l * beta1[y * n_stocks + s]);
                sum_m += resid;
                sum_s += rs;
                sum_mm += resid * resid;
                sum_ss += rs * rs;
                sum_ms += resid * rs;
                cnt_r += 1;
            }
        }
        percentile_ranks_in_place(&mut rank_pairs, &mut ranks);
        let resid_corr = if cnt_r > 30 {
            let nf2 = cnt_r as f64;
            let vm = sum_mm - sum_m * sum_m / nf2;
            let vs = sum_ss - sum_s * sum_s / nf2;
            let vms = sum_ms - sum_m * sum_s / nf2;
            if vm > 1e-18 && vs > 1e-18 {
                (vms / (vm * vs).sqrt()).clamp(-1.0, 1.0)
            } else {
                f64::NAN
            }
        } else {
            f64::NAN
        };
        let b0 = col_beta(k);
        let bt0 = col_beta_t(k);
        let r0 = col_resid(k);
        let z0 = col_resid_z(k);
        let rk0 = col_resid_rank(k);
        let ab0 = col_resid_abs(k);
        let lv0 = col_leverage(k);
        let ck0 = col_cooks(k);
        let bs0 = col_beta_shift(k);
        let ls0 = col_lambda_shift(k);
        tmp_resid.fill(f64::NAN);
        tmp_z.fill(f64::NAN);
        tmp_h.fill(f64::NAN);
        tmp_cooks.fill(f64::NAN);
        for &s in &cs.valid {
            let yv = ycur[s] as f64;
            let mut fitted = cs.alpha;
            let mut h = 1.0 / nf;
            for i in 0..k {
                let b = mbeta[i * n_stocks + s];
                fitted += cs.lambda[i] * b;
            }
            let mut d = [0.0f64; MAX_FACTORS];
            for i in 0..k {
                d[i] = mbeta[i * n_stocks + s] - bmean[i];
            }
            for i in 0..k {
                for j in 0..k {
                    h += d[i] * cs.inv[i * MAX_FACTORS + j] * d[j];
                }
            }
            let resid = yv - fitted;
            let z = if residual_std > 0.0 {
                resid / residual_std
            } else {
                f64::NAN
            };
            let one_minus_h = (1.0 - h).max(1e-12);
            let studentized = if z.is_finite() {
                z / one_minus_h.sqrt()
            } else {
                f64::NAN
            };
            let cooks = if cs.sse > 0.0 && studentized.is_finite() {
                studentized * studentized * h / (2.0 * one_minus_h)
            } else {
                f64::NAN
            };
            tmp_resid[s] = resid;
            tmp_z[s] = z;
            tmp_h[s] = h;
            tmp_cooks[s] = cooks;
        }
        for i in 0..k {
            let fi = m.factors[i];
            let se = cs.se[i];
            for &s in &cs.valid {
                let bi = mbeta[i * n_stocks + s];
                buf.write_ps(b0 + i, bin, s, bi as f32);
                let t = if se > 0.0 && bi.is_finite() {
                    bi / se
                } else {
                    f64::NAN
                };
                buf.write_ps(bt0 + i, bin, s, t as f32);
                let bs = beta1[fi * n_stocks + s];
                buf.write_ps(bs0 + i, bin, s, (bi - bs) as f32);
                let ls = lam_sk[i];
                buf.write_ps(ls0 + i, bin, s, (cs.lambda[i] - ls) as f32);
            }
        }
        for &s in &cs.valid {
            buf.write_ps(r0, bin, s, tmp_resid[s] as f32);
            buf.write_ps(z0, bin, s, tmp_z[s] as f32);
            buf.write_ps(rk0, bin, s, ranks[s] as f32);
            buf.write_ps(ab0, bin, s, tmp_resid[s].abs() as f32);
            buf.write_ps(lv0, bin, s, tmp_h[s] as f32);
            buf.write_ps(ck0, bin, s, tmp_cooks[s] as f32);
        }
        buf.write_sh(col_r2(k), bin, cs.r2 as f32);
        buf.write_sh(col_adj_r2(k), bin, cs.adj_r2 as f32);
        buf.write_sh(col_alpha(k), bin, cs.alpha as f32);
        buf.write_sh(col_f_stat(k), bin, cs.f_stat as f32);
        buf.write_sh(col_cond(k), bin, cs.cond as f32);
        for i in 0..k {
            buf.write_sh(col_lambda(k) + i, bin, cs.lambda[i] as f32);
            buf.write_sh(col_vif(k) + i, bin, cs.vif[i] as f32);
        }
        let delta_r2 = cs.r2 - r2_s;
        let resid_improve = if sse_s > 0.0 && sse_s.is_finite() {
            (sse_s - cs.sse) / sse_s
        } else {
            f64::NAN
        };
        let alpha_shift = cs.alpha - alpha_s;
        buf.write_sh(col_delta_r2(k), bin, delta_r2 as f32);
        buf.write_sh(col_resid_improve(k), bin, resid_improve as f32);
        buf.write_sh(col_alpha_shift(k), bin, alpha_shift as f32);
        buf.write_sh(col_resid_corr(k), bin, resid_corr as f32);
        for i in 0..k {
            let sse_sk_i = sse_sk[i];
            if sse_sk_i.is_finite() && dof > 0.0 {
                let num = (sse_sk_i - cs.sse).max(0.0) / (k as f64 - 1.0);
                let den = cs.sse / dof;
                let f = if den > 0.0 { num / den } else { f64::INFINITY };
                buf.write_sh(col_nested_f(k) + i, bin, f as f32);
                buf.write_sh(
                    col_nested_p(k) + i,
                    bin,
                    f_pvalue(f, k as f64 - 1.0, dof) as f32,
                );
            } else {
                buf.write_sh(col_nested_f(k) + i, bin, f32::NAN);
                buf.write_sh(col_nested_p(k) + i, bin, f32::NAN);
            }
        }
        let sign = 1.0;
        for s in 0..n_stocks {
            let yv = signals[ysig_base + bin * n_stocks + s] as f64;
            let st = &mut states[s];
            for f in 0..N_FEATURES {
                add_pair(&mut st.pairs[f], market[f * N_BINS + bin], yv, sign);
            }
            if yv.is_finite() {
                for &(i, j) in cross_pairs.iter() {
                    add_cross(
                        &mut st.cross[cross_idx[i][j] as usize],
                        market[i * N_BINS + bin],
                        market[j * N_BINS + bin],
                        true,
                        sign,
                    );
                }
            }
        }
    }
    buf
}

// 主入口：读全市场 → 网格 → 组合并行计算 → 21 统计降维 → (codes, vals)
// ---------------------------------------------------------------------------

/// 阶段 A：读全市场 → 14 指标网格 → 转置 [feature][bin][stock] + 市场均值。
fn load_market_grid(date: i64) -> io::Result<(Vec<String>, Vec<f32>, Vec<f64>, usize)> {
    let codes = list_codes(date);
    let n_stocks = codes.len();
    if n_stocks == 0 {
        return Ok((codes, Vec::new(), Vec::new(), 0));
    }
    let start_us = grid_start_us_for_date(date)?;

    let feature_rows: Vec<Option<Vec<f32>>> = codes
        .par_iter()
        .map(|code| {
            let trades = read_trade_fast_inner(code, date, false, false, usize::MAX).ok()?;
            let market = read_market_fast_inner(code, date, false, false, usize::MAX).ok()?;
            Some(extract_14_features(&trades, &market, start_us))
        })
        .collect();

    let total = N_FEATURES * N_BINS * n_stocks;
    let mut signals = vec![f32::NAN; total];
    signals
        .par_chunks_mut(n_stocks)
        .enumerate()
        .for_each(|(row, destination)| {
            let feature = row / N_BINS;
            let bin = row % N_BINS;
            let source = bin * N_FEATURES + feature;
            for stock in 0..n_stocks {
                if let Some(values) = &feature_rows[stock] {
                    destination[stock] = values[source];
                }
            }
        });
    drop(feature_rows);

    let mut market = vec![f64::NAN; N_FEATURES * N_BINS];
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

    Ok((codes, signals, market, n_stocks))
}

/// 21 个单列时序统计（with_threshold_counts=false 时的输出，见 features.rs）。
const STAT_SUFFIXES: [&str; 21] = [
    "mean",
    "median",
    "std",
    "skew",
    "kurt",
    "p5",
    "p25",
    "p75",
    "p95",
    "iqr",
    "cv",
    "autocorr1",
    "autocorr1_abs",
    "trend",
    "curvature",
    "quad_coef",
    "period_diff",
    "period_ratio",
    "lz_complexity",
    "entropy_1d",
    "max_range_product",
];

/// 每组合的 (模型名, y, 组合列名) 元信息，用于命名。
struct ComboMeta {
    model: &'static str,
    y: usize,
    col_names: Vec<String>,
    n_per_stock: usize,
    n_shared: usize,
}

fn combo_metas() -> Vec<ComboMeta> {
    let mut out = Vec::with_capacity(N_COMBOS);
    for m in MODELS.iter() {
        for &y in m.ys {
            out.push(ComboMeta {
                model: m.name,
                y,
                col_names: col_names(m.k, &m.factors),
                n_per_stock: 4 * m.k + 6,
                n_shared: 4 * m.k + 9,
            });
        }
    }
    out
}

/// 全部因子名（53 组合 × 39/55 列 × 21 统计）。
pub fn multi_factor_capm_names() -> Vec<String> {
    let mut names = Vec::with_capacity(N_FACTORS);
    for meta in combo_metas() {
        let prefix = format!("mfcapm_{}_{}", meta.model, BASE_NAMES[meta.y]);
        // 顺序与 features.rs 输出一致：[统计] 组外层、[列] 内层
        for stat in STAT_SUFFIXES.iter() {
            for col in &meta.col_names {
                names.push(format!("{prefix}_{col}_{stat}"));
            }
        }
    }
    debug_assert_eq!(names.len(), N_FACTORS);
    names
}

/// 50 核线程池（进程级单例）：multi_factor_capm 全流程（读盘/组合/统计）限制 50 线程并行。
fn mf_pool() -> &'static rayon::ThreadPool {
    static POOL: std::sync::OnceLock<rayon::ThreadPool> = std::sync::OnceLock::new();
    POOL.get_or_init(|| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(50)
            .thread_name(|i| format!("mfcapm-{i}"))
            .build()
            .expect("构建 mfcapm 50 线程池失败")
    })
}

/// 主入口：单日全市场 49,791 维横截面因子。
pub fn compute_multi_factor_capm_full(date: i64) -> io::Result<(Vec<String>, Vec<f32>)> {
    // 整个流程（读盘/组合/21 统计/组装）固定运行在 50 线程池内，保证核数上限
    mf_pool().install(|| compute_multi_factor_capm_full_inner(date))
}

fn compute_multi_factor_capm_full_inner(date: i64) -> io::Result<(Vec<String>, Vec<f32>)> {
    let _t_all = std::time::Instant::now();
    // 预分配（页错误）与读盘并行
    let n_est = list_codes(date).len();
    let (prealloc, grid) = rayon::join(|| prealloc_combos(n_est), || load_market_grid(date));
    let (codes, signals, market, n_stocks) = grid?;
    eprintln!(
        "  [mfcapm] 读盘+网格: {:.1}s",
        _t_all.elapsed().as_secs_f64()
    );
    if n_stocks == 0 {
        return Ok((codes, Vec::new()));
    }

    // 53 个组合并行（扁平任务，每组合独立滚动矩；与旧版逻辑逐位一致）
    let combos: Vec<ComboBuf> = prealloc
        .into_par_iter()
        .zip(0..N_COMBOS)
        .map(|(buf, idx)| {
            // idx → (模型, y)
            let mut acc = 0usize;
            for (mi, m) in MODELS.iter().enumerate() {
                if idx < acc + m.ys.len() {
                    let y = m.ys[idx - acc];
                    return compute_one_combo(&signals, &market, n_stocks, y, mi, buf);
                }
                acc += m.ys.len();
            }
            unreachable!()
        })
        .collect();
    drop(signals);
    drop(market);
    eprintln!(
        "  [mfcapm] 组合计算: {:.1}s",
        _t_all.elapsed().as_secs_f64()
    );

    // 组合视图：模型主序（与 combo_metas()/因子名顺序一致）
    let combos_ref: Vec<&ComboBuf> = combos.iter().collect();
    let combos = combos_ref;
    debug_assert_eq!(combos.len(), N_COMBOS);

    // shared 列 21 统计（每股相同，算一次；53 组合并行；零拷贝 + 复用缓冲）
    let shared_stats: Vec<Vec<f32>> = combos
        .par_iter()
        .map(|cb| {
            let n_shared = cb.n_shared;
            let mut out = vec![0.0f32; 21 * n_shared];
            let mut scratch = features::StatsScratch::new();
            // shared 布局 [col][bin]：行步长 1，列步长 N_BINS
            features::col_stats_21_strided(
                &cb.shared,
                N_BINS,
                n_shared,
                1,
                N_BINS,
                &mut scratch,
                &mut out,
            );
            out
        })
        .collect();

    // per-stock 列 21 统计 + 组装（按股票并行，直接写入最终 vals 布局）
    // 每股段内布局：[组合0: stat0(ps|sh) stat1...][组合1...]，与 names 顺序一致
    let mut combo_bases = Vec::with_capacity(N_COMBOS);
    {
        let mut acc = 0usize;
        for cb in &combos {
            combo_bases.push(acc);
            acc += 21 * (cb.n_per_stock + cb.n_shared);
        }
        debug_assert_eq!(acc, N_FACTORS);
    }
    let max_ps = combos.iter().map(|cb| cb.n_per_stock).max().unwrap_or(0);
    let mut vals = vec![0.0f32; n_stocks * N_FACTORS];
    let _t_shared = std::time::Instant::now();
    vals.par_chunks_mut(N_FACTORS)
        .enumerate()
        .for_each(|(s, seg)| {
            let mut scratch = features::StatsScratch::new();
            let mut tmp = vec![0.0f32; 21 * max_ps];
            for (ci, cb) in combos.iter().enumerate() {
                let n_ps = cb.n_per_stock;
                if n_ps == 0 {
                    continue;
                }
                // 布局 [stock][bin][col]：行主序矩阵 [N_BINS, n_ps]。
                // 先转置为列主序（列连续）再统计 —— 直接按 n_ps 步长读列的 10+ 次
                // 扫描会命中不同的 cache line（stride=4n_ps 字节），慢 3-5 倍。
                let start = (s * N_BINS) * n_ps;
                let end = ((s + 1) * N_BINS) * n_ps;
                scratch.col_stats_21_row_major(&cb.per_stock[start..end], N_BINS, n_ps, &mut tmp);
                let base = combo_bases[ci];
                let n_sh = cb.n_shared;
                let sh = &shared_stats[ci];
                for stat in 0..21 {
                    let row = base + stat * (n_ps + n_sh);
                    for col in 0..n_ps {
                        seg[row + col] = tmp[stat * n_ps + col];
                    }
                    for col in 0..n_sh {
                        seg[row + n_ps + col] = sh[stat * n_sh + col];
                    }
                }
            }
        });

    eprintln!(
        "  [mfcapm] shared统计: {:.1}s",
        _t_shared.elapsed().as_secs_f64()
    );
    eprintln!("  [mfcapm] 总耗时: {:.1}s", _t_all.elapsed().as_secs_f64());
    Ok((codes, vals))
}

// ---------------------------------------------------------------------------
// PyO3 接口
// ---------------------------------------------------------------------------

#[pyfunction]
pub fn py_multi_factor_capm(py: Python<'_>, date: i64) -> PyResult<(Vec<String>, Vec<f32>)> {
    py.allow_threads(|| compute_multi_factor_capm_full(date))
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
}

#[pyfunction]
pub fn py_multi_factor_capm_names() -> Vec<String> {
    multi_factor_capm_names()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn factor_count_matches() {
        assert_eq!(N_FACTORS, 49_791);
        assert_eq!(multi_factor_capm_names().len(), N_FACTORS);
        let mut n_cols_total = 0usize;
        for m in MODELS.iter() {
            n_cols_total += m.ys.len() * n_cols(m.k);
        }
        assert_eq!(n_cols_total * 21, N_FACTORS);
        // 列索引不越界
        for m in MODELS.iter() {
            let k = m.k;
            assert!(col_lambda_shift(k) + k <= n_cols(k));
            assert_eq!(col_beta(k) + k, col_beta_t(k));
            assert!(col_nested_p(k) + k <= n_cols(k));
        }
        // 列名数量
        for m in MODELS.iter() {
            assert_eq!(col_names(m.k, &m.factors).len(), n_cols(m.k));
        }
    }

    #[test]
    fn cholesky_roundtrip() {
        let a = [
            4.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let _ = a;
        // 2×2 正定矩阵（MAX_FACTORS 步长布局）
        let mut m = [0.0f64; MAX_FACTORS * MAX_FACTORS];
        m[0] = 4.0;
        m[1] = 2.0;
        m[MAX_FACTORS] = 2.0;
        m[MAX_FACTORS + 1] = 3.0;
        let mut l = [0.0f64; MAX_FACTORS * MAX_FACTORS];
        assert!(cholesky(&m, 2, &mut l));
        let mut x = [0.0f64; MAX_FACTORS];
        cholesky_solve(&l, 2, &[1.0, 2.0], &mut x);
        // A x = [1,2] → x = [ -0.125, 0.75 ]
        assert!((x[0] + 0.125).abs() < 1e-12);
        assert!((x[1] - 0.75).abs() < 1e-12);
    }
}
