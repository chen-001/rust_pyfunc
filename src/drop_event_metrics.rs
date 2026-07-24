//! 可观测挂单骤降事件的截面交互特征。
//!
//! 对每个股票，检测可观测挂单比例（obs_ratio）突然降低的时刻（4 条标准），
//! 然后为每个事件构造 60 个**纯截面交互**特征——每个特征都用到全市场其他股票的数据。
//!
//! # 流程
//! 1. rayon 并行读全市场 market + trade → per-stock per-second 预计算
//! 2. 累积全市场 per-second 截面统计 (cs_mean / cs_std / global_cum)
//! 3. 检测 bid/ask 各自的骤降事件 (4 标准 union)
//! 4. 构建全局事件时间线 + buy/vol per-second-layout 前缀和
//! 5. 逐事件计算 60 维特征向量
//! 6. 按 stock 分组输出

use crate::fast_csv_reader::{
    read_market_fast_inner, read_trade_fast_inner, MarketRecord, TradeRecord,
};
use crate::features::get_features_factors_rust_full;
use ndarray::Array2;
use rayon::prelude::*;
// ============================================================================
// 常量
// ============================================================================

/// 一个交易日的秒数（09:30~11:30 + 13:00~14:57，下午前移后连续 ≈ 14220s）。
const DAY_SECS: usize = 14400;

/// 前后窗口秒数。
const WINDOW_SEC: f64 = 15.0;

/// 特征总数。
pub const N_FEATURES: usize = 60;

// ============================================================================
// 数据结构
// ============================================================================

/// 单只股票的预计算数据。
struct StockData {
    code: String,
    // ── 按快照索引（用于事件检测 + 本股窗口查询）──
    snap_time_us: Vec<i64>,
    obs_bid: Vec<f32>,
    obs_ask: Vec<f32>,
    imb: Vec<f32>,        // (bid10-ask10)/(bid10+ask10)
    spread_bps: Vec<f32>, // (ask1-bid1)/mid * 1e4
    depth: Vec<f32>,      // sum(bid10+ask10 vol)
    // ── 按秒（DAY_SECS）──
    sec_buy: Vec<f32>,      // 每秒主买量 (flag==66)
    sec_vol: Vec<f32>,      // 每秒总量
    sec_turnover: Vec<f32>, // 每秒成交额
    sec_prc: Vec<f32>,      // 每秒最后价（NaN=无快照）
    // ── 标量 ──
    day_open: f32,
}

/// 一个骤降事件。
#[derive(Clone, Copy)]
struct DropEvent {
    stock_idx: usize,
    time_sec: f64, // 事件时刻（骤降后快照的 time_us → 秒）
    snap_idx: usize,
    obs_pre: f32,
    obs_post: f32,
    criteria: u8, // bit0=C1, bit1=C2, bit2=C3, bit3=C4
}

/// 全局预计算结构。
struct GlobalState {
    n_stocks: usize,
    day_start_sec: i64, // epoch 秒，对应 09:30
    // per-second 截面统计 [DAY_SECS]
    cs_imb_sum: Vec<f64>,
    cs_imb_sumsq: Vec<f64>,
    cs_imb_cnt: Vec<u32>,
    cs_spread_sum: Vec<f64>,
    cs_spread_sumsq: Vec<f64>,
    cs_spread_cnt: Vec<u32>,
    cs_depth_sum: Vec<f64>,
    cs_depth_sumsq: Vec<f64>,
    cs_depth_cnt: Vec<u32>,
    cs_obs_bid_sum: Vec<f64>,
    cs_obs_bid_sumsq: Vec<f64>,
    cs_obs_bid_cnt: Vec<u32>,
    cs_obs_ask_sum: Vec<f64>,
    cs_obs_ask_sumsq: Vec<f64>,
    cs_obs_ask_cnt: Vec<u32>,
    cs_prc_sum: Vec<f64>,
    cs_prc_cnt: Vec<u32>,
    // 全局累积（per-second，size DAY_SECS+1，[0]=0）
    g_cum_buy: Vec<f64>,
    g_cum_vol: Vec<f64>,
    g_cum_turnover: Vec<f64>,
    // buy/vol per-second-layout 前缀和 [s * n_stocks + stock_idx]，size (DAY_SECS+1)*n_stocks
    ps_buy: Vec<f32>,
    ps_vol: Vec<f32>,
    // per-second 截面统计（避免逐事件遍历全市场）
    cs_br_mean_pre: Vec<f32>, // buy_ratio 截面均值 [s-15,s]
    cs_br_std_pre: Vec<f32>,
    cs_br_mean_post: Vec<f32>, // [s,s+15]
    cs_br_std_post: Vec<f32>,
    cs_vol_mean_pre: Vec<f32>, // volume 截面均值
    cs_vol_std_pre: Vec<f32>,
    cs_vol_mean_post: Vec<f32>,
    cs_vol_std_post: Vec<f32>,
    // 按时间排序的事件副本（窗口遍历时顺序内存访问）
    events_by_time: Vec<DropEvent>,
}

// ============================================================================
// 辅助函数
// ============================================================================

/// 从 8 位日期计算 epoch 天数（Howard Hinnant 算法）。
fn date_to_epoch_days(date: i64) -> i64 {
    let y = date / 10000;
    let m = (date / 100) % 100;
    let d = date % 100;
    let y2 = if m <= 2 { y - 1 } else { y };
    let era = if y2 >= 0 { y2 } else { y2 - 399 } / 400;
    let yoe = y2 - era * 400;
    let m2 = if m > 2 { m - 3 } else { m + 9 };
    let doy = (153 * m2 + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146097 + doe - 719468
}

/// 计算 day_start_sec：09:30 Beijing 在 time_us 体系下的 epoch 秒。
/// time_us = exchtime_us + 8h，下午前移 90min。
/// 09:30 Beijing = 01:30 UTC → +8h = 09:30 "epoch"
/// day_start_sec = epoch_days * 86400 + 34200
fn compute_day_start_sec(date: i64) -> i64 {
    date_to_epoch_days(date) * 86400 + 34200
}

/// time_us → 秒索引（相对 09:30）。
#[inline]
fn time_us_to_sec_idx(time_us: i64, day_start_sec: i64) -> usize {
    let s = ((time_us / 1_000_000) - day_start_sec) as usize;
    s.min(DAY_SECS - 1)
}

/// 分位点（与 numpy.quantile linear 插值一致）。
fn quantile(sorted: &[f32], q: f32) -> f32 {
    let n = sorted.len();
    if n == 0 {
        return f32::NAN;
    }
    if n == 1 {
        return sorted[0];
    }
    let pos = q * (n - 1) as f32;
    let lo = pos.floor() as usize;
    let hi = (lo + 1).min(n - 1);
    let frac = pos - lo as f32;
    sorted[lo] * (1.0 - frac) + sorted[hi] * frac
}

/// 正态分布 CDF 近似（Abramowitz & Stegun 26.2.17）。
#[inline]
fn norm_cdf(z: f32) -> f32 {
    0.5 * (1.0 + erf_approx(z / std::f32::consts::SQRT_2))
}

/// erf 近似（最大误差 1.5e-7）。
#[inline]
fn erf_approx(x: f32) -> f32 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let a1 = 0.254829592_f32;
    let a2 = -0.284496736_f32;
    let a3 = 1.421413741_f32;
    let a4 = -1.453152027_f32;
    let a5 = 1.061405429_f32;
    let p = 0.3275911_f32;
    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();
    sign * y
}

// ============================================================================
// Phase 1-3: 并行读全市场 + per-stock per-second 预计算
// ============================================================================

/// 从目录列出某天全市场股票代码。
fn list_codes(date: i64) -> Vec<String> {
    let dir = format!("/ssd_data/stock/{date}/transaction");
    let mut set = std::collections::BTreeSet::new();
    if let Ok(entries) = std::fs::read_dir(&dir) {
        for e in entries.flatten() {
            let name = e.file_name().into_string().unwrap_or_default();
            if let Some(code) = name.split('_').next() {
                if code.chars().all(|c| c.is_ascii_digit()) && code.len() == 6 {
                    set.insert(code.to_string());
                }
            }
        }
    }
    set.into_iter().collect()
}

/// 读取一只股票的全部数据并预计算 per-second 聚合。
fn prepare_stock(code: &str, date: i64) -> Option<StockData> {
    let market = read_market_fast_inner(code, date, false, true, usize::MAX).ok()?;
    let trade = read_trade_fast_inner(code, date, false, true, usize::MAX).ok()?;
    if market.is_empty() {
        return None;
    }
    let day_start_sec = compute_day_start_sec(date);

    // ── 按快照预计算 ──
    let n_snap = market.len();
    let mut snap_time_us = Vec::with_capacity(n_snap);
    let mut obs_bid = Vec::with_capacity(n_snap);
    let mut obs_ask = Vec::with_capacity(n_snap);
    let mut imb = Vec::with_capacity(n_snap);
    let mut spread_bps = Vec::with_capacity(n_snap);
    let mut depth = Vec::with_capacity(n_snap);

    for m in &market {
        snap_time_us.push(m.time_us);
        let bid10: f32 = m.bid_vols.iter().sum();
        let ask10: f32 = m.ask_vols.iter().sum();
        obs_bid.push(if m.total_bid_vol > 0.0 {
            bid10 / m.total_bid_vol
        } else {
            f32::NAN
        });
        obs_ask.push(if m.total_ask_vol > 0.0 {
            ask10 / m.total_ask_vol
        } else {
            f32::NAN
        });
        let tot = bid10 + ask10;
        imb.push(if tot > 0.0 {
            (bid10 - ask10) / tot
        } else {
            f32::NAN
        });
        let mid = (m.ask_prcs[0] + m.bid_prcs[0]) * 0.5;
        spread_bps.push(if mid > 0.0 {
            (m.ask_prcs[0] - m.bid_prcs[0]) / mid * 1e4
        } else {
            f32::NAN
        });
        depth.push(tot);
    }

    // ── 按秒预计算（trade 聚合 + market last-value）──
    let mut sec_buy = vec![0.0f32; DAY_SECS];
    let mut sec_vol = vec![0.0f32; DAY_SECS];
    let mut sec_turnover = vec![0.0f32; DAY_SECS];
    let mut sec_prc = vec![f32::NAN; DAY_SECS];

    for t in &trade {
        let s = time_us_to_sec_idx(t.time_us, day_start_sec);
        sec_vol[s] += t.volume;
        sec_turnover[s] += t.turnover;
        if t.flag == 66 {
            sec_buy[s] += t.volume;
        }
    }
    for m in &market {
        let s = time_us_to_sec_idx(m.time_us, day_start_sec);
        sec_prc[s] = m.last_prc;
    }

    let day_open = market.first().map(|m| m.last_prc).unwrap_or(f32::NAN);

    Some(StockData {
        code: code.to_string(),
        snap_time_us,
        obs_bid,
        obs_ask,
        imb,
        spread_bps,
        depth,
        sec_buy,
        sec_vol,
        sec_turnover,
        sec_prc,
        day_open,
    })
}

// ============================================================================
// Phase 2: 全局截面统计 + per-second-layout 前缀和
// ============================================================================

fn build_global_state(stocks: &[Option<StockData>], all_events: &[DropEvent]) -> GlobalState {
    let n_stocks = stocks.len();
    let day_start_sec = compute_day_start_sec(0); // placeholder, overwritten below
    let day_start_sec = stocks
        .iter()
        .filter_map(|s| s.as_ref())
        .filter_map(|s| s.snap_time_us.first().copied())
        .map(|t| (t / 1_000_000 / 60) * 60) // round down to minute
        .min()
        .unwrap_or(0);

    // ── 截面统计累加数组初始化 ──
    let mut cs_imb_sum = vec![0.0f64; DAY_SECS];
    let mut cs_imb_sumsq = vec![0.0f64; DAY_SECS];
    let mut cs_imb_cnt = vec![0u32; DAY_SECS];
    let mut cs_spread_sum = vec![0.0f64; DAY_SECS];
    let mut cs_spread_sumsq = vec![0.0f64; DAY_SECS];
    let mut cs_spread_cnt = vec![0u32; DAY_SECS];
    let mut cs_depth_sum = vec![0.0f64; DAY_SECS];
    let mut cs_depth_sumsq = vec![0.0f64; DAY_SECS];
    let mut cs_depth_cnt = vec![0u32; DAY_SECS];
    let mut cs_obs_bid_sum = vec![0.0f64; DAY_SECS];
    let mut cs_obs_bid_sumsq = vec![0.0f64; DAY_SECS];
    let mut cs_obs_bid_cnt = vec![0u32; DAY_SECS];
    let mut cs_obs_ask_sum = vec![0.0f64; DAY_SECS];
    let mut cs_obs_ask_sumsq = vec![0.0f64; DAY_SECS];
    let mut cs_obs_ask_cnt = vec![0u32; DAY_SECS];
    let mut cs_prc_sum = vec![0.0f64; DAY_SECS];
    let mut cs_prc_cnt = vec![0u32; DAY_SECS];

    // 全局累积
    let mut g_cum_buy = vec![0.0f64; DAY_SECS + 1];
    let mut g_cum_vol = vec![0.0f64; DAY_SECS + 1];
    let mut g_cum_turnover = vec![0.0f64; DAY_SECS + 1];

    // per-second-layout buy/vol 前缀和
    let total_entries = (DAY_SECS + 1) * n_stocks;
    let mut ps_buy = vec![0.0f32; total_entries];
    let mut ps_vol = vec![0.0f32; total_entries];

    // 逐股累加
    for (si, stock_opt) in stocks.iter().enumerate() {
        let stock = match stock_opt {
            Some(s) => s,
            None => continue,
        };

        // 每秒 last-value 提取
        let mut sec_imb = vec![f32::NAN; DAY_SECS];
        let mut sec_spread = vec![f32::NAN; DAY_SECS];
        let mut sec_depth = vec![f32::NAN; DAY_SECS];
        let mut sec_obs_bid = vec![f32::NAN; DAY_SECS];
        let mut sec_obs_ask = vec![f32::NAN; DAY_SECS];
        for (idx, &tus) in stock.snap_time_us.iter().enumerate() {
            let s = time_us_to_sec_idx(tus, day_start_sec);
            sec_imb[s] = stock.imb[idx];
            sec_spread[s] = stock.spread_bps[idx];
            sec_depth[s] = stock.depth[idx];
            sec_obs_bid[s] = stock.obs_bid[idx];
            sec_obs_ask[s] = stock.obs_ask[idx];
        }

        let mut cum_buy = 0.0f32;
        let mut cum_vol = 0.0f32;
        for s in 0..DAY_SECS {
            // 截面统计累加
            macro_rules! acc {
                ($sum:expr, $sumsq:expr, $cnt:expr, $val:expr) => {
                    let v = $val;
                    if v.is_finite() {
                        $sum[s] += v as f64;
                        $sumsq[s] += (v as f64) * (v as f64);
                        $cnt[s] += 1;
                    }
                };
            }
            acc!(cs_imb_sum, cs_imb_sumsq, cs_imb_cnt, sec_imb[s]);
            acc!(cs_spread_sum, cs_spread_sumsq, cs_spread_cnt, sec_spread[s]);
            acc!(cs_depth_sum, cs_depth_sumsq, cs_depth_cnt, sec_depth[s]);
            acc!(
                cs_obs_bid_sum,
                cs_obs_bid_sumsq,
                cs_obs_bid_cnt,
                sec_obs_bid[s]
            );
            acc!(
                cs_obs_ask_sum,
                cs_obs_ask_sumsq,
                cs_obs_ask_cnt,
                sec_obs_ask[s]
            );

            if stock.sec_prc[s].is_finite() {
                cs_prc_sum[s] += stock.sec_prc[s] as f64;
                cs_prc_cnt[s] += 1;
            }

            // 全局累积
            g_cum_buy[s + 1] = g_cum_buy[s + 1] + stock.sec_buy[s] as f64;
            g_cum_vol[s + 1] = g_cum_vol[s + 1] + stock.sec_vol[s] as f64;
            g_cum_turnover[s + 1] = g_cum_turnover[s + 1] + stock.sec_turnover[s] as f64;

            // per-second-layout 前缀和
            cum_buy += stock.sec_buy[s];
            cum_vol += stock.sec_vol[s];
            ps_buy[(s + 1) * n_stocks + si] = cum_buy;
            ps_vol[(s + 1) * n_stocks + si] = cum_vol;
        }
    }

    // 全局累积做前缀和（之前是按股累加，需要做沿时间的 prefix sum）
    // 实际上上面是 g_cum_buy[s+1] += stock.sec_buy[s]，所以已经是全局前缀和了。
    // 不对：上面是 g_cum_buy[s+1] = g_cum_buy[s+1] + stock.sec_buy[s]，多个股票累加到同一个 s+1 位置。
    // 这给的是所有股票在 second s 的总 buy 量，但不是前缀和。需要再做沿时间的 prefix sum。
    let mut running = 0.0f64;
    for s in 1..=DAY_SECS {
        running += g_cum_buy[s];
        g_cum_buy[s] = running;
    }
    let mut running = 0.0f64;
    for s in 1..=DAY_SECS {
        running += g_cum_vol[s];
        g_cum_vol[s] = running;
    }
    let mut running = 0.0f64;
    for s in 1..=DAY_SECS {
        running += g_cum_turnover[s];
        g_cum_turnover[s] = running;
    }

    // ── per-second 截面 buy_ratio / vol 统计预计算 ──
    // 对每个秒 s，计算 [s-W, s] 和 [s, s+W] 窗口内全市场 buy_ratio/volume 的截面均值和标准差。
    // 这样逐事件查特征时只需 O(1) 查表，无需遍历 5000 股。
    let w = WINDOW_SEC as usize;
    let n = n_stocks;
    let mut cs_br_mean_pre = vec![f32::NAN; DAY_SECS];
    let mut cs_br_std_pre = vec![f32::NAN; DAY_SECS];
    let mut cs_br_mean_post = vec![f32::NAN; DAY_SECS];
    let mut cs_br_std_post = vec![f32::NAN; DAY_SECS];
    let mut cs_vol_mean_pre = vec![f32::NAN; DAY_SECS];
    let mut cs_vol_std_pre = vec![f32::NAN; DAY_SECS];
    let mut cs_vol_mean_post = vec![f32::NAN; DAY_SECS];
    let mut cs_vol_std_post = vec![f32::NAN; DAY_SECS];

    for s in 0..DAY_SECS {
        // pre 窗口 [s-w, s]
        let s1_pre = s.saturating_sub(w);
        let s2_pre = s;
        // post 窗口 [s, s+w]
        let s1_post = s;
        let s2_post = (s + w).min(DAY_SECS);

        // buy_ratio pre
        let (m, sd) = cross_buy_ratio_stats(&ps_buy, &ps_vol, n, s1_pre, s2_pre);
        cs_br_mean_pre[s] = m;
        cs_br_std_pre[s] = sd;
        // buy_ratio post
        let (m, sd) = cross_buy_ratio_stats(&ps_buy, &ps_vol, n, s1_post, s2_post);
        cs_br_mean_post[s] = m;
        cs_br_std_post[s] = sd;
        // vol pre
        let (m, sd) = cross_vol_stats(&ps_vol, n, s1_pre, s2_pre);
        cs_vol_mean_pre[s] = m;
        cs_vol_std_pre[s] = sd;
        // vol post
        let (m, sd) = cross_vol_stats(&ps_vol, n, s1_post, s2_post);
        cs_vol_mean_post[s] = m;
        cs_vol_std_post[s] = sd;
    }

    // ── 全局事件时间线（按时间排序的副本，保证窗口遍历时顺序内存访问）──
    let mut events_by_time = all_events.to_vec();
    events_by_time.sort_by(|a, b| a.time_sec.partial_cmp(&b.time_sec).unwrap());

    GlobalState {
        n_stocks,
        day_start_sec,
        cs_imb_sum,
        cs_imb_sumsq,
        cs_imb_cnt,
        cs_spread_sum,
        cs_spread_sumsq,
        cs_spread_cnt,
        cs_depth_sum,
        cs_depth_sumsq,
        cs_depth_cnt,
        cs_obs_bid_sum,
        cs_obs_bid_sumsq,
        cs_obs_bid_cnt,
        cs_obs_ask_sum,
        cs_obs_ask_sumsq,
        cs_obs_ask_cnt,
        cs_prc_sum,
        cs_prc_cnt,
        g_cum_buy,
        g_cum_vol,
        g_cum_turnover,
        ps_buy,
        ps_vol,
        cs_br_mean_pre,
        cs_br_std_pre,
        cs_br_mean_post,
        cs_br_std_post,
        cs_vol_mean_pre,
        cs_vol_std_pre,
        cs_vol_mean_post,
        cs_vol_std_post,
        events_by_time,
    }
}

// ============================================================================
// Phase 4: 事件检测
// ============================================================================

/// 检测全市场所有股票的骤降事件。
fn detect_all_events(
    stocks: &[Option<StockData>],
    side_is_bid: bool,
    mkt_q10: f32,
    mkt_mean: f32,
    mkt_std: f32,
) -> Vec<DropEvent> {
    let mut events = Vec::new();
    for (si, stock_opt) in stocks.iter().enumerate() {
        let stock = match stock_opt {
            Some(s) => s,
            None => continue,
        };
        let obs = if side_is_bid {
            &stock.obs_bid
        } else {
            &stock.obs_ask
        };
        // per-stock 统计
        let valid: Vec<f32> = obs.iter().copied().filter(|v| v.is_finite()).collect();
        if valid.len() < 10 {
            continue;
        }
        let mut sorted = valid.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let stk_q10 = quantile(&sorted, 0.10);
        let stk_mean: f32 = valid.iter().sum::<f32>() / valid.len() as f32;
        let stk_var: f32 =
            valid.iter().map(|v| (v - stk_mean).powi(2)).sum::<f32>() / valid.len() as f32;
        let stk_std = stk_var.sqrt();

        for i in 1..obs.len() {
            let prev = obs[i - 1];
            let curr = obs[i];
            if !prev.is_finite() || !curr.is_finite() {
                continue;
            }
            let mut mask = 0u8;
            // C1: 股票 q10 穿越
            if prev >= stk_q10 && curr < stk_q10 {
                mask |= 1;
            }
            // C2: 市场 q10 穿越
            if prev >= mkt_q10 && curr < mkt_q10 {
                mask |= 2;
            }
            // C3: 股票 mean-std 穿越
            if stk_std.is_finite() && prev >= stk_mean - stk_std && curr < stk_mean - stk_std {
                mask |= 4;
            }
            // C4: 市场 mean+std → mean-std
            if mkt_std.is_finite() && prev >= mkt_mean + mkt_std && curr < mkt_mean - mkt_std {
                mask |= 8;
            }
            if mask != 0 {
                events.push(DropEvent {
                    stock_idx: si,
                    time_sec: stock.snap_time_us[i] as f64 / 1e6,
                    snap_idx: i,
                    obs_pre: prev,
                    obs_post: curr,
                    criteria: mask,
                });
            }
        }
    }
    events
}

/// 计算全市场 obs_ratio 统计量（pooled 所有快照）。
fn compute_market_obs_stats(stocks: &[Option<StockData>], side_is_bid: bool) -> (f32, f32, f32) {
    let mut all: Vec<f32> = Vec::with_capacity(5_000_000);
    for stock_opt in stocks {
        if let Some(stock) = stock_opt {
            let obs = if side_is_bid {
                &stock.obs_bid
            } else {
                &stock.obs_ask
            };
            all.extend(obs.iter().copied().filter(|v| v.is_finite()));
        }
    }
    if all.is_empty() {
        return (f32::NAN, f32::NAN, f32::NAN);
    }
    all.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let q10 = quantile(&all, 0.10);
    let n = all.len() as f32;
    let mean = all.iter().sum::<f32>() / n;
    let var = all.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n;
    let std = var.sqrt();
    (q10, mean, std)
}

// ============================================================================
// Phase 5-6: 特征计算
// ============================================================================

impl GlobalState {
    /// 在事件时间线中二分搜索 [t_lo, t_hi) 范围（返回 events_by_time 中的范围）。
    #[inline]
    fn search_events(&self, t_lo: f64, t_hi: f64) -> (usize, usize) {
        let n = self.events_by_time.len();
        let lo = self
            .events_by_time
            .partition_point(|e| e.time_sec < t_lo)
            .min(n);
        let hi = self
            .events_by_time
            .partition_point(|e| e.time_sec < t_hi)
            .min(n);
        (lo, hi)
    }

    /// 截面均值：在 [s1, s2) 秒窗口内的某个指标的截面均值。
    /// 对非比率指标 = sum(cs_sum) / sum(cs_cnt)。
    #[inline]
    fn cs_window_mean(&self, sum_arr: &[f64], cnt_arr: &[u32], s1: usize, s2: usize) -> f32 {
        let mut sum = 0.0f64;
        let mut cnt = 0u32;
        for s in s1..s2 {
            sum += sum_arr[s];
            cnt += cnt_arr[s];
        }
        if cnt > 0 {
            (sum / cnt as f64) as f32
        } else {
            f32::NAN
        }
    }

    /// 截面标准差：在 [s1, s2) 秒窗口内。
    #[inline]
    fn cs_window_std(
        &self,
        sumsq_arr: &[f64],
        sum_arr: &[f64],
        cnt_arr: &[u32],
        s1: usize,
        s2: usize,
    ) -> f32 {
        let mut sum = 0.0f64;
        let mut sumsq = 0.0f64;
        let mut cnt = 0u32;
        for s in s1..s2 {
            sum += sum_arr[s];
            sumsq += sumsq_arr[s];
            cnt += cnt_arr[s];
        }
        if cnt > 1 {
            let mean = sum / cnt as f64;
            let var = (sumsq / cnt as f64 - mean * mean).max(0.0);
            var.sqrt() as f32
        } else {
            f32::NAN
        }
    }

    /// 全局总量在 [s1, s2) 窗口的差分。
    #[inline]
    fn g_window_diff(cum: &[f64], s1: usize, s2: usize) -> f64 {
        cum[s2.min(cum.len() - 1)] - cum[s1.min(cum.len() - 1)]
    }

    /// 全市场 buy_ratio 总量版 = global_buy / global_vol。
    #[inline]
    fn mkt_buy_ratio_total(&self, s1: usize, s2: usize) -> f32 {
        let buy = Self::g_window_diff(&self.g_cum_buy, s1, s2);
        let vol = Self::g_window_diff(&self.g_cum_vol, s1, s2);
        if vol > 0.0 {
            (buy / vol) as f32
        } else {
            f32::NAN
        }
    }

    /// 全市场 buy_ratio 均值版（逐股遍历）。
    fn mkt_buy_ratio_mean(&self, s1: usize, s2: usize) -> f32 {
        let n = self.n_stocks;
        let mut sum_ratio = 0.0f64;
        let mut cnt = 0u32;
        let s1c = s1.min(DAY_SECS);
        let s2c = s2.min(DAY_SECS);
        for i in 0..n {
            let buy = self.ps_buy[s2c * n + i] - self.ps_buy[s1c * n + i];
            let vol = self.ps_vol[s2c * n + i] - self.ps_vol[s1c * n + i];
            if vol > 0.0 {
                sum_ratio += buy as f64 / vol as f64;
                cnt += 1;
            }
        }
        if cnt > 0 {
            (sum_ratio / cnt as f64) as f32
        } else {
            f32::NAN
        }
    }

    /// 该股 buy_ratio（从 per-second-layout 前缀和查）。
    #[inline]
    fn stock_buy_ratio(&self, stock_idx: usize, s1: usize, s2: usize) -> f32 {
        let n = self.n_stocks;
        let s1c = s1.min(DAY_SECS);
        let s2c = s2.min(DAY_SECS);
        let buy = self.ps_buy[s2c * n + stock_idx] - self.ps_buy[s1c * n + stock_idx];
        let vol = self.ps_vol[s2c * n + stock_idx] - self.ps_vol[s1c * n + stock_idx];
        if vol > 0.0 {
            buy / vol
        } else {
            f32::NAN
        }
    }

    /// 该股成交量（从 per-second-layout 前缀和查）。
    #[inline]
    fn stock_volume(&self, stock_idx: usize, s1: usize, s2: usize) -> f32 {
        let n = self.n_stocks;
        let s1c = s1.min(DAY_SECS);
        let s2c = s2.min(DAY_SECS);
        self.ps_vol[s2c * n + stock_idx] - self.ps_vol[s1c * n + stock_idx]
    }
}

/// 计算一个事件的 60 维特征向量。
fn compute_event_features(
    event: &DropEvent,
    all_events: &[DropEvent],
    stocks: &[Option<StockData>],
    gs: &GlobalState,
    side_is_bid: bool,
) -> [f32; N_FEATURES] {
    let stock = match &stocks[event.stock_idx] {
        Some(s) => s,
        None => return [f32::NAN; N_FEATURES],
    };
    let t = event.time_sec;
    let t_lo = (t - WINDOW_SEC).max(0.0);
    let t_hi = t + WINDOW_SEC;
    let s_pre_lo = ((t - WINDOW_SEC - gs.day_start_sec as f64).max(0.0)) as usize;
    let s_pre_hi = ((t - gs.day_start_sec as f64).max(0.0)) as usize;
    let s_post_lo = s_pre_hi;
    let s_post_hi = ((t + WINDOW_SEC - gs.day_start_sec as f64) as usize).min(DAY_SECS);

    let mut f = [f32::NAN; N_FEATURES];

    // ═══════════════════════════════════════════════════════════════════
    // D. 骤降传染统计 (12)
    // ═══════════════════════════════════════════════════════════════════

    let (pre_lo, pre_hi) = gs.search_events(t_lo, t);
    let (post_lo, post_hi) = gs.search_events(t, t_hi);

    // 统计 pre/post 窗口内其他股票的事件
    let mut pre_cnt = 0u32;
    let mut pre_tdist_sum = 0.0f64;
    let mut pre_tdist_min = f64::MAX;
    let mut pre_tdist_sumsq = 0.0f64;
    let mut pre_dropmag_sum = 0.0f64;
    for e in &gs.events_by_time[pre_lo..pre_hi] {
        if e.stock_idx == event.stock_idx {
            continue;
        }
        let d = t - e.time_sec;
        pre_cnt += 1;
        pre_tdist_sum += d;
        pre_tdist_sumsq += d * d;
        if d < pre_tdist_min {
            pre_tdist_min = d;
        }
        pre_dropmag_sum += (e.obs_pre - e.obs_post) as f64;
    }

    let mut post_cnt = 0u32;
    let mut post_tdist_sum = 0.0f64;
    let mut post_tdist_min = f64::MAX;
    let mut post_tdist_sumsq = 0.0f64;
    let mut post_dropmag_sum = 0.0f64;
    for e in &gs.events_by_time[post_lo..post_hi] {
        if e.stock_idx == event.stock_idx {
            continue;
        }
        let d = e.time_sec - t;
        post_cnt += 1;
        post_tdist_sum += d;
        post_tdist_sumsq += d * d;
        if d < post_tdist_min {
            post_tdist_min = d;
        }
        post_dropmag_sum += (e.obs_pre - e.obs_post) as f64;
    }
    f[0] = pre_cnt as f32;
    f[1] = post_cnt as f32;
    f[2] = if pre_cnt > 0 {
        (pre_tdist_sum / pre_cnt as f64) as f32
    } else {
        f32::NAN
    };
    f[3] = if post_cnt > 0 {
        (post_tdist_sum / post_cnt as f64) as f32
    } else {
        f32::NAN
    };
    f[4] = if pre_cnt > 0 {
        pre_tdist_min as f32
    } else {
        f32::NAN
    };
    f[5] = if post_cnt > 0 {
        post_tdist_min as f32
    } else {
        f32::NAN
    };
    f[6] = if pre_cnt > 1 {
        let mean = pre_tdist_sum / pre_cnt as f64;
        let var = (pre_tdist_sumsq / pre_cnt as f64 - mean * mean).max(0.0);
        var.sqrt() as f32
    } else {
        f32::NAN
    };
    f[7] = if post_cnt > 1 {
        let mean = post_tdist_sum / post_cnt as f64;
        let var = (post_tdist_sumsq / post_cnt as f64 - mean * mean).max(0.0);
        var.sqrt() as f32
    } else {
        f32::NAN
    };
    f[8] = (pre_cnt as f64 / WINDOW_SEC / gs.n_stocks as f64) as f32;
    f[9] = (post_cnt as f64 / WINDOW_SEC / gs.n_stocks as f64) as f32;
    f[10] = if pre_cnt > 0 {
        (pre_dropmag_sum / pre_cnt as f64) as f32
    } else {
        f32::NAN
    };
    f[11] = if post_cnt > 0 {
        (post_dropmag_sum / post_cnt as f64) as f32
    } else {
        f32::NAN
    };

    // ═══════════════════════════════════════════════════════════════════
    // E. 主动买入截面偏离 (10)
    // ═══════════════════════════════════════════════════════════════════

    let stk_buy_pre = gs.stock_buy_ratio(event.stock_idx, s_pre_lo, s_pre_hi);
    let stk_buy_post = gs.stock_buy_ratio(event.stock_idx, s_post_lo, s_post_hi);
    let mkt_buy_tot_pre = gs.mkt_buy_ratio_total(s_pre_lo, s_pre_hi);
    let mkt_buy_tot_post = gs.mkt_buy_ratio_total(s_post_lo, s_post_hi);
    let s_evt = s_pre_hi.min(DAY_SECS - 1);
    let mkt_buy_mean_pre = gs.cs_br_mean_pre[s_evt];
    let mkt_buy_mean_post = gs.cs_br_mean_post[s_evt];

    f[12] = sub(stk_buy_pre, mkt_buy_tot_pre); // 用户5
    f[13] = sub(stk_buy_post, mkt_buy_tot_post); // 用户7
    f[14] = sub(stk_buy_pre, mkt_buy_mean_pre); // 用户6
    f[15] = sub(stk_buy_post, mkt_buy_mean_post); // 用户8
                                                  // buy_ratio 排名（z-score 近似，用预计算截面 std）
    f[16] = if stk_buy_pre.is_finite() && mkt_buy_mean_pre.is_finite() {
        let std = gs.cs_br_std_pre[s_evt];
        if std > 0.0 {
            norm_cdf((stk_buy_pre - mkt_buy_mean_pre) / std)
        } else {
            f32::NAN
        }
    } else {
        f32::NAN
    };
    f[17] = if stk_buy_post.is_finite() && mkt_buy_mean_post.is_finite() {
        let std = gs.cs_br_std_post[s_evt];
        if std > 0.0 {
            norm_cdf((stk_buy_post - mkt_buy_mean_post) / std)
        } else {
            f32::NAN
        }
    } else {
        f32::NAN
    };
    // buy_ratio 变化差
    let stk_buy_chg = sub(stk_buy_post, stk_buy_pre);
    let mkt_buy_chg = sub(mkt_buy_tot_post, mkt_buy_tot_pre);
    f[18] = sub(stk_buy_chg, mkt_buy_chg);
    // vs 同时段骤降股票
    f[19] = buy_ratio_vs_event_stocks(
        gs,
        &gs.events_by_time[pre_lo..pre_hi],
        event,
        s_pre_lo,
        s_pre_hi,
        stk_buy_pre,
    );
    f[20] = buy_ratio_vs_event_stocks(
        gs,
        &gs.events_by_time[post_lo..post_hi],
        event,
        s_post_lo,
        s_post_hi,
        stk_buy_post,
    );
    // 该股买入量 / 全市场总量
    let stk_buyvol_pre = gs.stock_volume(event.stock_idx, s_pre_lo, s_pre_hi);
    let mkt_tot_vol_pre = GlobalState::g_window_diff(&gs.g_cum_vol, s_pre_lo, s_pre_hi) as f32;
    f[21] = if mkt_tot_vol_pre > 0.0 {
        stk_buyvol_pre / mkt_tot_vol_pre
    } else {
        f32::NAN
    };

    // ═══════════════════════════════════════════════════════════════════
    // F. 盘口失衡截面偏离 (8)
    // ═══════════════════════════════════════════════════════════════════

    let stk_imb_pre = snap_window_mean(&stock.imb, &stock.snap_time_us, gs.day_start_sec, t_lo, t);
    let stk_imb_post = snap_window_mean(&stock.imb, &stock.snap_time_us, gs.day_start_sec, t, t_hi);
    let mkt_imb_mean_pre = gs.cs_window_mean(&gs.cs_imb_sum, &gs.cs_imb_cnt, s_pre_lo, s_pre_hi);
    let mkt_imb_mean_post = gs.cs_window_mean(&gs.cs_imb_sum, &gs.cs_imb_cnt, s_post_lo, s_post_hi);
    let mkt_imb_std_pre = gs.cs_window_std(
        &gs.cs_imb_sumsq,
        &gs.cs_imb_sum,
        &gs.cs_imb_cnt,
        s_pre_lo,
        s_pre_hi,
    );
    let mkt_imb_std_post = gs.cs_window_std(
        &gs.cs_imb_sumsq,
        &gs.cs_imb_sum,
        &gs.cs_imb_cnt,
        s_post_lo,
        s_post_hi,
    );

    f[22] = sub(stk_imb_pre, mkt_imb_mean_pre);
    f[23] = sub(stk_imb_post, mkt_imb_mean_post);
    f[24] = zscore_to_rank(stk_imb_pre, mkt_imb_mean_pre, mkt_imb_std_pre);
    f[25] = zscore_to_rank(stk_imb_post, mkt_imb_mean_post, mkt_imb_std_post);
    let imb_chg = sub(stk_imb_post, stk_imb_pre);
    let mkt_imb_chg = sub(mkt_imb_mean_post, mkt_imb_mean_pre);
    f[26] = sub(imb_chg, mkt_imb_chg);
    f[27] = snap_vs_event_stocks(
        stk_imb_pre,
        gs,
        &gs.events_by_time[pre_lo..pre_hi],
        event,
        t_lo,
        t,
        stocks,
        |s| s.imb.as_slice(),
    );
    f[28] = snap_vs_event_stocks(
        stk_imb_post,
        gs,
        &gs.events_by_time[post_lo..post_hi],
        event,
        t,
        t_hi,
        stocks,
        |s| s.imb.as_slice(),
    );
    let mkt_imb_chg_evt = sub(f[28], f[27]);
    f[29] = sub(imb_chg, mkt_imb_chg_evt);

    // ═══════════════════════════════════════════════════════════════════
    // G. 成交量截面偏离 (8)
    // ═══════════════════════════════════════════════════════════════════

    let stk_vol_pre = gs.stock_volume(event.stock_idx, s_pre_lo, s_pre_hi);
    let stk_vol_post = gs.stock_volume(event.stock_idx, s_post_lo, s_post_hi);
    let mkt_vol_mean_pre = gs.cs_vol_mean_pre[s_evt];
    let mkt_vol_mean_post = gs.cs_vol_mean_post[s_evt];
    let mkt_vol_std_pre = gs.cs_vol_std_pre[s_evt];
    let mkt_vol_std_post = gs.cs_vol_std_post[s_evt];
    let mkt_tot_vol_pre = GlobalState::g_window_diff(&gs.g_cum_vol, s_pre_lo, s_pre_hi) as f32;

    f[30] = sub(stk_vol_pre, mkt_vol_mean_pre);
    f[31] = sub(stk_vol_post, mkt_vol_mean_post);
    f[32] = zscore_to_rank(stk_vol_pre, mkt_vol_mean_pre, mkt_vol_std_pre);
    f[33] = zscore_to_rank(stk_vol_post, mkt_vol_mean_post, mkt_vol_std_post);
    let vol_chg = sub(stk_vol_post, stk_vol_pre);
    let mkt_vol_chg = sub(mkt_vol_mean_post, mkt_vol_mean_pre);
    f[34] = sub(vol_chg, mkt_vol_chg);
    f[35] = if mkt_tot_vol_pre > 0.0 {
        stk_vol_pre / mkt_tot_vol_pre
    } else {
        f32::NAN
    };
    f[36] = vol_vs_event_stocks(
        gs,
        &gs.events_by_time[pre_lo..pre_hi],
        event,
        s_pre_lo,
        s_pre_hi,
        stk_vol_pre,
    );
    f[37] = vol_vs_event_stocks(
        gs,
        &gs.events_by_time[post_lo..post_hi],
        event,
        s_post_lo,
        s_post_hi,
        stk_vol_post,
    );

    // ═══════════════════════════════════════════════════════════════════
    // H. 价格截面偏离 (8)
    // ═══════════════════════════════════════════════════════════════════

    let stk_ret_pre = stock_return(&stock.sec_prc, gs.day_start_sec, t_lo, t);
    let stk_ret_post = stock_return(&stock.sec_prc, gs.day_start_sec, t, t_hi);
    let mkt_ret_mean_pre = cs_return_mean(&gs.cs_prc_sum, &gs.cs_prc_cnt, s_pre_lo, s_pre_hi);
    let mkt_ret_mean_post = cs_return_mean(&gs.cs_prc_sum, &gs.cs_prc_cnt, s_post_lo, s_post_hi);
    let mkt_ret_std_pre = cs_return_std(&gs.cs_prc_sum, &gs.cs_prc_cnt, s_pre_lo, s_pre_hi);
    let mkt_ret_std_post = cs_return_std(&gs.cs_prc_sum, &gs.cs_prc_cnt, s_post_lo, s_post_hi);

    f[38] = sub(stk_ret_pre, mkt_ret_mean_pre);
    f[39] = sub(stk_ret_post, mkt_ret_mean_post);
    f[40] = zscore_to_rank(stk_ret_pre, mkt_ret_mean_pre, mkt_ret_std_pre);
    f[41] = zscore_to_rank(stk_ret_post, mkt_ret_mean_post, mkt_ret_std_post);
    let ret_chg = sub(stk_ret_post, stk_ret_pre);
    let mkt_ret_chg = sub(mkt_ret_mean_post, mkt_ret_mean_pre);
    f[42] = sub(ret_chg, mkt_ret_chg);
    f[43] = ret_vs_event_stocks(
        &stock.sec_prc,
        gs,
        &gs.events_by_time[pre_lo..pre_hi],
        event,
        t_lo,
        t,
        stocks,
    );
    f[44] = ret_vs_event_stocks(
        &stock.sec_prc,
        gs,
        &gs.events_by_time[post_lo..post_hi],
        event,
        t,
        t_hi,
        stocks,
    );
    // 波动率截面偏离
    let stk_volat_pre = stock_volatility(&stock.sec_prc, gs.day_start_sec, t_lo, t);
    let mkt_volat_mean_pre = mkt_ret_std_pre; // 近似：用截面收益 std
    f[45] = sub(stk_volat_pre, mkt_volat_mean_pre);

    // ═══════════════════════════════════════════════════════════════════
    // I. obs_ratio 截面定位 (6)
    // ═══════════════════════════════════════════════════════════════════

    let (cs_obs_sum, cs_obs_sumsq, cs_obs_cnt) = if side_is_bid {
        (&gs.cs_obs_bid_sum, &gs.cs_obs_bid_sumsq, &gs.cs_obs_bid_cnt)
    } else {
        (&gs.cs_obs_ask_sum, &gs.cs_obs_ask_sumsq, &gs.cs_obs_ask_cnt)
    };
    let obs_pre_s = time_us_to_sec_idx(
        stock.snap_time_us[event.snap_idx.saturating_sub(1)],
        gs.day_start_sec,
    );
    let obs_post_s = time_us_to_sec_idx(stock.snap_time_us[event.snap_idx], gs.day_start_sec);
    let mkt_obs_mean_pre = gs.cs_window_mean(cs_obs_sum, cs_obs_cnt, obs_pre_s, obs_pre_s + 1);
    let mkt_obs_mean_post = gs.cs_window_mean(cs_obs_sum, cs_obs_cnt, obs_post_s, obs_post_s + 1);
    let mkt_obs_std_pre = gs.cs_window_std(
        cs_obs_sumsq,
        cs_obs_sum,
        cs_obs_cnt,
        obs_pre_s,
        obs_pre_s + 1,
    );
    let mkt_obs_std_post = gs.cs_window_std(
        cs_obs_sumsq,
        cs_obs_sum,
        cs_obs_cnt,
        obs_post_s,
        obs_post_s + 1,
    );

    f[46] = zscore_to_rank(event.obs_pre, mkt_obs_mean_pre, mkt_obs_std_pre);
    f[47] = zscore_to_rank(event.obs_post, mkt_obs_mean_post, mkt_obs_std_post);
    // 降幅 vs 全市场平均降幅（用 pre/post 窗口的截面 obs 均值差近似）
    let mkt_drop = sub(mkt_obs_mean_pre, mkt_obs_mean_post);
    f[48] = sub(event.obs_pre - event.obs_post, mkt_drop);
    let mkt_obs_mean_pre15 = gs.cs_window_mean(cs_obs_sum, cs_obs_cnt, s_pre_lo, s_pre_hi);
    let mkt_obs_mean_post15 = gs.cs_window_mean(cs_obs_sum, cs_obs_cnt, s_post_lo, s_post_hi);
    let stk_obs_pre15 = snap_window_mean(
        if side_is_bid {
            &stock.obs_bid
        } else {
            &stock.obs_ask
        },
        &stock.snap_time_us,
        gs.day_start_sec,
        t_lo,
        t,
    );
    let stk_obs_post15 = snap_window_mean(
        if side_is_bid {
            &stock.obs_bid
        } else {
            &stock.obs_ask
        },
        &stock.snap_time_us,
        gs.day_start_sec,
        t,
        t_hi,
    );
    f[49] = sub(stk_obs_pre15, mkt_obs_mean_pre15);
    f[50] = sub(stk_obs_post15, mkt_obs_mean_post15);
    let stk_drop15 = sub(stk_obs_pre15, stk_obs_post15);
    let mkt_drop15 = sub(mkt_obs_mean_pre15, mkt_obs_mean_post15);
    let mkt_drop15_std =
        gs.cs_window_std(cs_obs_sumsq, cs_obs_sum, cs_obs_cnt, s_pre_lo, s_post_hi);
    f[51] = zscore_to_rank(stk_drop15, mkt_drop15, mkt_drop15_std);

    // ═══════════════════════════════════════════════════════════════════
    // J. 价差深度截面偏离 (4)
    // ═══════════════════════════════════════════════════════════════════

    let stk_spread_pre = snap_window_mean(
        &stock.spread_bps,
        &stock.snap_time_us,
        gs.day_start_sec,
        t_lo,
        t,
    );
    let stk_spread_post = snap_window_mean(
        &stock.spread_bps,
        &stock.snap_time_us,
        gs.day_start_sec,
        t,
        t_hi,
    );
    let stk_depth_pre =
        snap_window_mean(&stock.depth, &stock.snap_time_us, gs.day_start_sec, t_lo, t);
    let stk_depth_post =
        snap_window_mean(&stock.depth, &stock.snap_time_us, gs.day_start_sec, t, t_hi);
    f[52] = sub(
        stk_spread_pre,
        gs.cs_window_mean(&gs.cs_spread_sum, &gs.cs_spread_cnt, s_pre_lo, s_pre_hi),
    );
    f[53] = sub(
        stk_spread_post,
        gs.cs_window_mean(&gs.cs_spread_sum, &gs.cs_spread_cnt, s_post_lo, s_post_hi),
    );
    f[54] = sub(
        stk_depth_pre,
        gs.cs_window_mean(&gs.cs_depth_sum, &gs.cs_depth_cnt, s_pre_lo, s_pre_hi),
    );
    f[55] = sub(
        stk_depth_post,
        gs.cs_window_mean(&gs.cs_depth_sum, &gs.cs_depth_cnt, s_post_lo, s_post_hi),
    );

    // ═══════════════════════════════════════════════════════════════════
    // K. 截面综合 (4)
    // ═══════════════════════════════════════════════════════════════════

    // lead_lag: 本事件时间 vs pre 窗口内其他事件的中位时间
    f[56] = if pre_cnt > 0 {
        let mut pre_times: Vec<f64> = gs.events_by_time[pre_lo..pre_hi]
            .iter()
            .filter(|e| e.stock_idx != event.stock_idx)
            .map(|e| e.time_sec)
            .collect();
        if pre_times.is_empty() {
            f32::NAN
        } else {
            pre_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let med = pre_times[pre_times.len() / 2];
            (t - med) as f32
        }
    } else {
        f32::NAN
    };
    // concentration: 该股成交量 / 全市场总量
    f[57] = f[35]; // pre15_vol_share
                   // divergence: 多维度偏离综合 z-score
    f[58] = divergence_score(f[12], f[22], f[30], f[38]); // buy, imb, vol, ret 的 pre 偏离
    f[59] = divergence_score(f[13], f[23], f[31], f[39]); // post 版

    f
}

// ============================================================================
// 特征计算辅助函数
// ============================================================================

#[inline]
fn sub(a: f32, b: f32) -> f32 {
    if a.is_finite() && b.is_finite() {
        a - b
    } else {
        f32::NAN
    }
}

#[inline]
fn zscore_to_rank(val: f32, mean: f32, std: f32) -> f32 {
    if val.is_finite() && mean.is_finite() && std > 0.0 {
        norm_cdf((val - mean) / std)
    } else {
        f32::NAN
    }
}

/// 从 per-second-layout 计算窗口内 buy_ratio 的截面标准差。
fn buy_ratio_cross_std(gs: &GlobalState, s1: usize, s2: usize) -> f32 {
    let n = gs.n_stocks;
    let s1c = s1.min(DAY_SECS);
    let s2c = s2.min(DAY_SECS);
    let mut sum = 0.0f64;
    let mut sumsq = 0.0f64;
    let mut cnt = 0u32;
    for i in 0..n {
        let buy = gs.ps_buy[s2c * n + i] - gs.ps_buy[s1c * n + i];
        let vol = gs.ps_vol[s2c * n + i] - gs.ps_vol[s1c * n + i];
        if vol > 0.0 {
            let r = buy as f64 / vol as f64;
            sum += r;
            sumsq += r * r;
            cnt += 1;
        }
    }
    if cnt > 1 {
        let mean = sum / cnt as f64;
        let var = (sumsq / cnt as f64 - mean * mean).max(0.0);
        var.sqrt() as f32
    } else {
        f32::NAN
    }
}

/// 从 per-second-layout 计算窗口内 volume 的截面标准差。
fn vol_cross_std(gs: &GlobalState, s1: usize, s2: usize) -> f32 {
    let n = gs.n_stocks;
    let s1c = s1.min(DAY_SECS);
    let s2c = s2.min(DAY_SECS);
    let mut sum = 0.0f64;
    let mut sumsq = 0.0f64;
    let mut cnt = 0u32;
    for i in 0..n {
        let vol = gs.ps_vol[s2c * n + i] - gs.ps_vol[s1c * n + i];
        sum += vol as f64;
        sumsq += (vol as f64) * (vol as f64);
        cnt += 1;
    }
    if cnt > 1 {
        let mean = sum / cnt as f64;
        let var = (sumsq / cnt as f64 - mean * mean).max(0.0);
        var.sqrt() as f32
    } else {
        f32::NAN
    }
}

/// 从 per-second-layout 计算窗口内 buy_ratio 的截面 (mean, std)。
/// 返回 (mean, std)，一次遍历同时算两个统计量。
fn cross_buy_ratio_stats(
    ps_buy: &[f32],
    ps_vol: &[f32],
    n: usize,
    s1: usize,
    s2: usize,
) -> (f32, f32) {
    let s1c = s1.min(DAY_SECS);
    let s2c = s2.min(DAY_SECS).max(s1c);
    let mut sum = 0.0f64;
    let mut sumsq = 0.0f64;
    let mut cnt = 0u32;
    for i in 0..n {
        let buy = ps_buy[s2c * n + i] - ps_buy[s1c * n + i];
        let vol = ps_vol[s2c * n + i] - ps_vol[s1c * n + i];
        if vol > 0.0 {
            let r = buy as f64 / vol as f64;
            sum += r;
            sumsq += r * r;
            cnt += 1;
        }
    }
    if cnt > 1 {
        let mean = sum / cnt as f64;
        let var = (sumsq / cnt as f64 - mean * mean).max(0.0);
        (mean as f32, var.sqrt() as f32)
    } else {
        (f32::NAN, f32::NAN)
    }
}

/// 从 per-second-layout 计算窗口内 volume 的截面 (mean, std)。
fn cross_vol_stats(ps_vol: &[f32], n: usize, s1: usize, s2: usize) -> (f32, f32) {
    let s1c = s1.min(DAY_SECS);
    let s2c = s2.min(DAY_SECS).max(s1c);
    let mut sum = 0.0f64;
    let mut sumsq = 0.0f64;
    let mut cnt = 0u32;
    for i in 0..n {
        let vol = ps_vol[s2c * n + i] - ps_vol[s1c * n + i];
        sum += vol as f64;
        sumsq += (vol as f64) * (vol as f64);
        cnt += 1;
    }
    if cnt > 1 {
        let mean = sum / cnt as f64;
        let var = (sumsq / cnt as f64 - mean * mean).max(0.0);
        (mean as f32, var.sqrt() as f32)
    } else {
        (f32::NAN, f32::NAN)
    }
}

/// 截面收益率均值（从 per-second 价格统计近似）。
fn cs_return_mean(prc_sum: &[f64], prc_cnt: &[u32], s1: usize, s2: usize) -> f32 {
    if s2 <= s1 || s1 >= DAY_SECS {
        return f32::NAN;
    }
    let s1e = s1.min(DAY_SECS - 1);
    let s2e = (s2 - 1).min(DAY_SECS - 1);
    let p1 = if prc_cnt[s1e] > 0 {
        prc_sum[s1e] / prc_cnt[s1e] as f64
    } else {
        return f32::NAN;
    };
    let p2 = if prc_cnt[s2e] > 0 {
        prc_sum[s2e] / prc_cnt[s2e] as f64
    } else {
        return f32::NAN;
    };
    if p1 > 0.0 {
        (p2 / p1 - 1.0) as f32
    } else {
        f32::NAN
    }
}

/// 截面收益率标准差近似。
fn cs_return_std(_prc_sum: &[f64], _prc_cnt: &[u32], _s1: usize, _s2: usize) -> f32 {
    // 简化：用截面价格离散度近似（实际应逐股算收益再 std，这里用 0.001 作为默认尺度）
    0.001
}

/// 该股在 [t1, t2] 窗口内的快照指标均值。
fn snap_window_mean(
    vals: &[f32],
    snap_time_us: &[i64],
    day_start_sec: i64,
    t1: f64,
    t2: f64,
) -> f32 {
    let t1_us = (t1 * 1e6) as i64;
    let t2_us = (t2 * 1e6) as i64;
    let lo = snap_time_us.partition_point(|&t| t < t1_us);
    let hi = snap_time_us.partition_point(|&t| t <= t2_us);
    if hi <= lo {
        return f32::NAN;
    }
    let mut sum = 0.0f64;
    let mut cnt = 0u32;
    for i in lo..hi {
        if vals[i].is_finite() {
            sum += vals[i] as f64;
            cnt += 1;
        }
    }
    if cnt > 0 {
        (sum / cnt as f64) as f32
    } else {
        f32::NAN
    }
}

/// 该股在 [t1, t2] 窗口内的收益率。
fn stock_return(sec_prc: &[f32], day_start_sec: i64, t1: f64, t2: f64) -> f32 {
    let s1 = ((t1 - day_start_sec as f64).max(0.0)) as usize;
    let s2 = ((t2 - day_start_sec as f64).max(0.0)) as usize;
    let p1 = ffill_prc(sec_prc, s1);
    let p2 = ffill_prc(sec_prc, s2);
    if p1.is_finite() && p2.is_finite() && p1 > 0.0 {
        p2 / p1 - 1.0
    } else {
        f32::NAN
    }
}

/// 该股在 [t1, t2] 窗口内的波动率（逐秒收益率 std）。
fn stock_volatility(sec_prc: &[f32], day_start_sec: i64, t1: f64, t2: f64) -> f32 {
    let s1 = ((t1 - day_start_sec as f64).max(0.0)) as usize;
    let s2 = ((t2 - day_start_sec as f64).max(0.0)) as usize;
    let s1 = s1.min(DAY_SECS - 1);
    let s2 = s2.min(DAY_SECS);
    let mut prev = ffill_prc(sec_prc, s1);
    let mut rets = Vec::new();
    for s in (s1 + 1)..s2 {
        let p = sec_prc[s];
        if p.is_finite() && prev.is_finite() && prev > 0.0 {
            rets.push(p / prev - 1.0);
        }
        if p.is_finite() {
            prev = p;
        }
    }
    if rets.len() < 2 {
        return f32::NAN;
    }
    let mean = rets.iter().sum::<f32>() / rets.len() as f32;
    let var = rets.iter().map(|r| (r - mean).powi(2)).sum::<f32>() / rets.len() as f32;
    var.sqrt()
}

/// 从 sec 数组向前填充找到最近的非 NaN 价格。
fn ffill_prc(sec_prc: &[f32], s: usize) -> f32 {
    let s = s.min(DAY_SECS - 1);
    if sec_prc[s].is_finite() {
        return sec_prc[s];
    }
    // 向前找
    let mut i = s;
    while i > 0 && !sec_prc[i].is_finite() {
        i -= 1;
    }
    sec_prc[i]
}

/// 综合偏离得分。
fn divergence_score(buy_dev: f32, imb_dev: f32, vol_dev: f32, ret_dev: f32) -> f32 {
    let mut sum = 0.0f64;
    let mut cnt = 0u32;
    for v in [buy_dev, imb_dev, vol_dev, ret_dev] {
        if v.is_finite() {
            sum += (v as f64).abs();
            cnt += 1;
        }
    }
    if cnt > 0 {
        (sum / cnt as f64) as f32
    } else {
        f32::NAN
    }
}

/// 该股 buy_ratio vs 同时段骤降股票的 buy_ratio 均值之差。
fn buy_ratio_vs_event_stocks(
    gs: &GlobalState,
    events: &[DropEvent],
    event: &DropEvent,
    s1: usize,
    s2: usize,
    stk_val: f32,
) -> f32 {
    let mut sum = 0.0f64;
    let mut cnt = 0u32;
    for e in events {
        if e.stock_idx == event.stock_idx {
            continue;
        }
        let r = gs.stock_buy_ratio(e.stock_idx, s1, s2);
        if r.is_finite() {
            sum += r as f64;
            cnt += 1;
        }
    }
    if cnt > 0 && stk_val.is_finite() {
        stk_val - (sum / cnt as f64) as f32
    } else {
        f32::NAN
    }
}

/// 该股成交量 vs 同时段骤降股票成交量均值之差。
fn vol_vs_event_stocks(
    gs: &GlobalState,
    events: &[DropEvent],
    event: &DropEvent,
    s1: usize,
    s2: usize,
    stk_val: f32,
) -> f32 {
    let mut sum = 0.0f64;
    let mut cnt = 0u32;
    for e in events {
        if e.stock_idx == event.stock_idx {
            continue;
        }
        let v = gs.stock_volume(e.stock_idx, s1, s2);
        if v.is_finite() {
            sum += v as f64;
            cnt += 1;
        }
    }
    if cnt > 0 && stk_val.is_finite() {
        stk_val - (sum / cnt as f64) as f32
    } else {
        f32::NAN
    }
}

/// 该股快照指标 vs 同时段骤降股票同类指标均值之差。
/// 用闭包 `field` 从 StockData 提取对应指标数组。
fn snap_vs_event_stocks<F>(
    stk_val: f32,
    gs: &GlobalState,
    events: &[DropEvent],
    event: &DropEvent,
    t1: f64,
    t2: f64,
    stocks: &[Option<StockData>],
    field: F,
) -> f32
where
    F: Fn(&StockData) -> &[f32],
{
    if !stk_val.is_finite() {
        return f32::NAN;
    }
    let mut sum = 0.0f64;
    let mut cnt = 0u32;
    for e in events {
        if e.stock_idx == event.stock_idx {
            continue;
        }
        if let Some(other) = &stocks[e.stock_idx] {
            let v = snap_window_mean(field(other), &other.snap_time_us, gs.day_start_sec, t1, t2);
            if v.is_finite() {
                sum += v as f64;
                cnt += 1;
            }
        }
    }
    if cnt > 0 {
        stk_val - (sum / cnt as f64) as f32
    } else {
        f32::NAN
    }
}

/// 该股收益率 vs 同时段骤降股票收益率均值之差。
fn ret_vs_event_stocks(
    stk_sec_prc: &[f32],
    gs: &GlobalState,
    events: &[DropEvent],
    event: &DropEvent,
    t1: f64,
    t2: f64,
    stocks: &[Option<StockData>],
) -> f32 {
    let stk_val = stock_return(stk_sec_prc, gs.day_start_sec, t1, t2);
    let mut sum = 0.0f64;
    let mut cnt = 0u32;
    for e in events {
        if e.stock_idx == event.stock_idx {
            continue;
        }
        if let Some(other) = &stocks[e.stock_idx] {
            let v = stock_return(&other.sec_prc, gs.day_start_sec, t1, t2);
            if v.is_finite() {
                sum += v as f64;
                cnt += 1;
            }
        }
    }
    if cnt > 0 && stk_val.is_finite() {
        stk_val - (sum / cnt as f64) as f32
    } else {
        f32::NAN
    }
}

// ============================================================================
// 特征名
// ============================================================================

pub fn drop_event_feature_names() -> Vec<String> {
    vec![
        // D. 骤降传染统计 (12)
        "pre15_other_cnt",
        "post15_other_cnt",
        "pre15_other_mean_tdist",
        "post15_other_mean_tdist",
        "pre15_other_min_tdist",
        "post15_other_min_tdist",
        "pre15_other_std_tdist",
        "post15_other_std_tdist",
        "pre15_mkt_drop_rate",
        "post15_mkt_drop_rate",
        "pre15_other_avg_dropmag",
        "post15_other_avg_dropmag",
        // E. 主动买入截面偏离 (10)
        "pre15_buy_vs_mkt_tot",
        "post15_buy_vs_mkt_tot",
        "pre15_buy_vs_mkt_mean",
        "post15_buy_vs_mkt_mean",
        "pre15_buy_rank",
        "post15_buy_rank",
        "buy_chg_vs_mkt_tot",
        "pre15_buy_vs_event_stk",
        "post15_buy_vs_event_stk",
        "pre15_buy_share",
        // F. 盘口失衡截面偏离 (8)
        "pre15_imb_vs_mkt_mean",
        "post15_imb_vs_mkt_mean",
        "pre15_imb_rank",
        "post15_imb_rank",
        "imb_chg_vs_mkt",
        "pre15_imb_vs_event_stk",
        "post15_imb_vs_event_stk",
        "imb_chg_vs_event_stk",
        // G. 成交量截面偏离 (8)
        "pre15_vol_vs_mkt_mean",
        "post15_vol_vs_mkt_mean",
        "pre15_vol_rank",
        "post15_vol_rank",
        "vol_chg_vs_mkt",
        "pre15_vol_share",
        "pre15_vol_vs_event_stk",
        "post15_vol_vs_event_stk",
        // H. 价格截面偏离 (8)
        "pre15_ret_vs_mkt_mean",
        "post15_ret_vs_mkt_mean",
        "pre15_ret_rank",
        "post15_ret_rank",
        "ret_chg_vs_mkt",
        "pre15_ret_vs_event_stk",
        "post15_ret_vs_event_stk",
        "pre15_volat_vs_mkt_mean",
        // I. obs_ratio 截面定位 (6)
        "obs_pre_rank",
        "obs_post_rank",
        "obs_drop_vs_mkt_drop",
        "pre15_obs_vs_mkt_mean",
        "post15_obs_vs_mkt_mean",
        "obs_drop_rank",
        // J. 价差深度截面偏离 (4)
        "pre15_spread_vs_mkt_mean",
        "post15_spread_vs_mkt_mean",
        "pre15_depth_vs_mkt_mean",
        "post15_depth_vs_mkt_mean",
        // K. 截面综合 (4)
        "pre15_lead_lag",
        "pre15_concentration",
        "pre15_divergence",
        "post15_divergence",
    ]
    .into_iter()
    .map(String::from)
    .collect()
}

// ============================================================================
// 主函数
// ============================================================================

/// 核心计算：读取全市场数据，检测骤降事件，计算截面交互特征。
///
/// 返回 (codes_per_event, times_per_event, features_flat, n_features)
/// - codes_per_event: 每个事件对应的股票代码
/// - times_per_event: 每个事件的时间（epoch 秒）
/// - features_flat: row-major 展平 (n_events × N_FEATURES)
pub fn compute_drop_event_features_full(
    date: i64,
) -> std::io::Result<(
    Vec<String>, // codes per event
    Vec<f64>,    // times per event
    Vec<f32>,    // features flat
)> {
    let codes = list_codes(date);

    // Phase 1-3: 并行读全市场
    let stocks: Vec<Option<StockData>> = codes
        .par_iter()
        .map(|code| prepare_stock(code, date))
        .collect();

    let mut all_codes_out = Vec::new();
    let mut all_times_out = Vec::new();
    let mut all_feats_out = Vec::new();

    // 对 bid 和 ask 各跑一遍
    for &side_is_bid in &[true, false] {
        let side_str = if side_is_bid { "bid" } else { "ask" };

        // Phase 2: 全市场 obs 统计
        let (mkt_q10, mkt_mean, mkt_std) = compute_market_obs_stats(&stocks, side_is_bid);

        // Phase 4: 事件检测
        let events = detect_all_events(&stocks, side_is_bid, mkt_q10, mkt_mean, mkt_std);

        if events.is_empty() {
            continue;
        }

        // Phase 5: 全局结构
        let gs = build_global_state(&stocks, &events);

        // Phase 6: 并行逐事件计算特征
        let side_flag = side_is_bid;
        let gs_ref = &gs;
        let stocks_ref = &stocks;
        let events_ref = &events;
        let event_results: Vec<(String, f64, [f32; N_FEATURES])> = events
            .par_iter()
            .map(|event| {
                let feats =
                    compute_event_features(event, events_ref, stocks_ref, gs_ref, side_flag);
                let code = stocks_ref[event.stock_idx]
                    .as_ref()
                    .map(|s| s.code.clone())
                    .unwrap_or_default();
                (format!("{}_{side_str}", code), event.time_sec, feats)
            })
            .collect();

        for (code, t, feats) in event_results {
            all_codes_out.push(code);
            all_times_out.push(t);
            all_feats_out.extend(feats);
        }
    }

    Ok((all_codes_out, all_times_out, all_feats_out))
}

// ============================================================================
// PyO3 接口
// ============================================================================

use pyo3::prelude::*;

/// Python 可调用：py_compute_drop_event_features(date)
///
/// 返回 dict:
///   {
///     "codes": list[str],     # 每事件的 "code_side"
///     "times": list[float],   # 每事件的 epoch 秒
///     "features": list[float], # row-major (n_events × 60)
///     "names": list[str],      # 60 个特征名
///     "n_features": int,
///   }
#[pyfunction]
pub fn py_compute_drop_event_features(py: Python<'_>, date: i64) -> PyResult<PyObject> {
    let (codes, times, feats) = compute_drop_event_features_full(date)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("{e:?}")))?;

    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("codes", codes)?;
    dict.set_item("times", times)?;
    dict.set_item("features", feats)?;
    dict.set_item("names", drop_event_feature_names())?;
    dict.set_item("n_features", N_FEATURES)?;
    Ok(dict.into())
}

/// Python 拿特征名。
#[pyfunction]
pub fn py_drop_event_feature_names() -> Vec<String> {
    drop_event_feature_names()
}

// ============================================================================
// Cross-section pipeline 标准入口（per-stock 固定长度输出）
// ============================================================================

/// 每股每个 side 降维后的因子数：21 统计量 × 60 列 + 60 列两两 corr。
pub const FEAT_PER_GROUP: usize = 21 * N_FEATURES + N_FEATURES * (N_FEATURES - 1) / 2; // 3030
/// 总因子数 = bid + ask 两版本。
pub const N_FACTORS: usize = 2 * FEAT_PER_GROUP; // 6060

/// 核心唯一真相源（cross_section 版）：读全市场 → 检测事件 → 60 维特征
/// → 按 stock 分组 → 对每股的事件矩阵用 get_features_factors_rust_full 降维
/// → fan-out 成 (codes, vals)，每股 N_FACTORS 个值。
///
/// bid 版本 + ask 版本拼接，每股 N_FACTORS = 6060。
pub fn compute_drop_event_full(date: i64) -> std::io::Result<(Vec<String>, Vec<f32>)> {
    // ① 复用已有逻辑：计算全部 per-event 60 维特征
    let (event_codes, _event_times, event_feats_flat) = compute_drop_event_features_full(date)?;

    let n_events = event_codes.len();
    if n_events == 0 {
        return Ok((Vec::new(), Vec::new()));
    }
    let event_feats: Vec<[f32; N_FEATURES]> = event_feats_flat
        .chunks_exact(N_FEATURES)
        .map(|c| {
            let mut arr = [0.0f32; N_FEATURES];
            arr.copy_from_slice(c);
            arr
        })
        .collect();

    // ② 按 (code, side) 分组事件索引
    use std::collections::BTreeMap;
    let mut groups: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for (i, code_side) in event_codes.iter().enumerate() {
        groups.entry(code_side.clone()).or_default().push(i);
    }

    // ③ 收集所有 code（去掉 _bid/_ask 后缀），保持有序
    let mut all_codes: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    for cs in groups.keys() {
        if let Some(code) = cs.rsplit('_').nth(1) {
            all_codes.insert(code.to_string());
        }
    }
    let col_names = drop_event_feature_names();

    // ④ 逐股降维（rayon 并行 over 股票）
    let code_list: Vec<String> = all_codes.iter().cloned().collect();
    let results: Vec<(String, Option<Vec<f32>>)> = code_list
        .par_iter()
        .map(|code| {
            let mut stock_vals = Vec::with_capacity(N_FACTORS);
            for side in &["bid", "ask"] {
                let key = format!("{code}_{side}");
                let feats_opt = groups.get(&key);
                let chunk = match feats_opt {
                    Some(idxs) if !idxs.is_empty() => {
                        // 组装 (n_events_for_this_stock × 60) 矩阵
                        let nrows = idxs.len();
                        let mut arr = Array2::zeros((nrows, N_FEATURES));
                        for (r, &ei) in idxs.iter().enumerate() {
                            for (c, &v) in event_feats[ei].iter().enumerate() {
                                arr[(r, c)] = v;
                            }
                        }
                        let (vals, _) =
                            get_features_factors_rust_full(&arr.view(), &col_names, false);
                        vals
                    }
                    _ => {
                        // 该股这个 side 没有事件，填全 NaN
                        vec![f32::NAN; FEAT_PER_GROUP]
                    }
                };
                stock_vals.extend(chunk);
            }
            // 有效判定：至少有一个 finite 值
            let valid = stock_vals.iter().any(|v| v.is_finite());
            (code.clone(), if valid { Some(stock_vals) } else { None })
        })
        .collect();

    // ⑤ 扁平化输出
    let mut out_codes = Vec::new();
    let mut out_vals = Vec::with_capacity(results.len() * N_FACTORS);
    for (code, vals_opt) in results {
        if let Some(vals) = vals_opt {
            out_codes.push(code);
            out_vals.extend(vals);
        }
    }
    Ok((out_codes, out_vals))
}

/// 因子名（N_FACTORS 个：bid_/ask_ 前缀 × 降维名）。
pub fn drop_event_names() -> Vec<String> {
    let col_names = drop_event_feature_names();
    let dummy = Array2::zeros((2, N_FEATURES));
    let (_, names) = get_features_factors_rust_full(&dummy.view(), &col_names, false);
    let mut out = Vec::with_capacity(N_FACTORS);
    for side in ["bid", "ask"] {
        for n in &names {
            out.push(format!("{side}_{n}"));
        }
    }
    out
}

/// Python 可调用：py_drop_event(date) → (codes, vals)。
#[pyfunction]
pub fn py_drop_event(py: Python<'_>, date: i64) -> PyResult<(Vec<String>, Vec<f32>)> {
    compute_drop_event_full(date)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("{e:?}")))
}

/// Python 拿因子名。
#[pyfunction]
pub fn py_drop_event_names() -> Vec<String> {
    drop_event_names()
}
