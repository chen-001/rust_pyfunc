//! 个体挂单比例因子（individual_order_ratio）—— 40 列 × 双改造版。
//!
//! 核心思想：盘口挂单量指标三种归一口径——
//!   before  = 经典绝对量比值 (a-b)/(a+b)（局部和归一）
//!   after_r1 = rate1：a/TA - b/TB（单侧全量 total_ask_vol/total_bid_vol 归一）
//!   after_r2 = rate2：a/Σa10 - b/Σb10（单侧前10档总和 Σa[0..10] 归一）
//! 40 个盘口快照特征序列（覆盖 HTML 大类一全 40 指标）：
//!   [0..9]   档位失衡 obi_k（前 k 档，k=1..9）
//!   [9..20]  形态：wobi, mpgap, vol_trend/std/skew (a/b), hhi (a/b), spread_rel
//!   [20..30] 滚动(窗口20)：delta_obi1, roll_mean/std/z_obi1, delta_wobi, roll_mean_wobi,
//!            delta_mpgap, roll_cov_a1b1, obi1_ac1, obi1_trend
//!   [30..40] 逐笔(窗口60s)：eff_spread, trade_dir_div, trade_depth_a/b, ofi, vpin, volimb,
//!            intensity, avgsize, mp_dev
//!
//! 5 个 2D（各 n×40）：before, after_r1, after_r2, diff_r1(=r1-before), diff_r2(=r2-before)
//!   各调 get_features_factors_rust_full 降维 → 5 个因子序列，每个 1540 维
//!   (feat_per_group(40) = 19*40 + C(40,2) = 760+780 = 1540)
//! 日频 5 个 1D（各 40）：day_before/after_r1/after_r2/diff_r1/diff_r2（40 指标的全天均值）
//!
//! 输出 OUT_LEN = 5*1540 + 5*40 = 7700 + 200 = 7900。

use crate::features;
use ndarray::Array2;
use pyo3::prelude::*;

pub const SNAP_K: usize = 40;
pub const DAY_M: usize = 40;
pub const ROLL_N: usize = 20;
pub const TRADE_WIN_SEC: f32 = 60.0;
pub const FEAT_PER_GROUP: usize = 19 * SNAP_K + SNAP_K * (SNAP_K - 1) / 2; // 1540
pub const OUT_LEN: usize = 5 * FEAT_PER_GROUP + 5 * DAY_M; // 7900

const W: [f32; 10] = [
    1.0,
    1.0 / 2.0,
    1.0 / 3.0,
    1.0 / 4.0,
    1.0 / 5.0,
    1.0 / 6.0,
    1.0 / 7.0,
    1.0 / 8.0,
    1.0 / 9.0,
    1.0 / 10.0,
];

pub fn snap_col_names() -> Vec<String> {
    [
        "obi1",
        "obi2",
        "obi3",
        "obi4",
        "obi5",
        "obi6",
        "obi7",
        "obi8",
        "obi9",
        "wobi",
        "mpgap",
        "vtrend_a",
        "vtrend_b",
        "vstd_a",
        "vstd_b",
        "vskew_a",
        "vskew_b",
        "hhi_a",
        "hhi_b",
        "spread_rel",
        "dobi1",
        "rmean_obi1",
        "rstd_obi1",
        "rz_obi1",
        "dwobi",
        "rmean_wobi",
        "dmpgap",
        "rcov_a1b1",
        "obi1_ac1",
        "obi1_trend",
        "eff_spread",
        "trade_dir_div",
        "trade_depth_a",
        "trade_depth_b",
        "ofi",
        "vpin",
        "volimb",
        "intensity",
        "avgsize",
        "mp_dev",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect()
}

pub fn day_col_names() -> Vec<String> {
    snap_col_names()
}

// ============================================================================
// 小工具：数组(10档)统计
// ============================================================================

fn arr_mean(a: &[f32]) -> f32 {
    a.iter().sum::<f32>() / a.len() as f32
}
fn arr_std(a: &[f32]) -> f32 {
    let n = a.len();
    if n < 2 {
        return f32::NAN;
    }
    let m = arr_mean(a);
    let sq: f32 = a.iter().map(|&v| (v - m).powi(2)).sum();
    (sq / (n - 1) as f32).sqrt()
}
fn arr_skew(a: &[f32]) -> f32 {
    let n = a.len();
    if n < 3 {
        return f32::NAN;
    }
    let m = arr_mean(a);
    let m2: f32 = a.iter().map(|&v| (v - m).powi(2)).sum::<f32>() / n as f32;
    let m3: f32 = a.iter().map(|&v| (v - m).powi(3)).sum::<f32>() / n as f32;
    if m2 <= 0.0 {
        return f32::NAN;
    }
    let g1 = m3 / m2.powf(1.5);
    let nf = n as f32;
    let adj = g1 * (nf - 1.0).powf(1.5) / ((nf - 2.0) * nf.sqrt());
    if adj.is_finite() {
        adj
    } else {
        f32::NAN
    }
}
fn arr_corr_x(a: &[f32]) -> f32 {
    // corr(a, [1..n])
    let n = a.len();
    if n < 2 {
        return f32::NAN;
    }
    let xs: Vec<f32> = (1..=n).map(|x| x as f32).collect();
    let ma = arr_mean(a);
    let mx = xs.iter().sum::<f32>() / n as f32;
    let (cov, vx, vy) = (0..n).fold((0.0f32, 0.0f32, 0.0f32), |(c, vx, vy), k| {
        let dx = xs[k] - mx;
        let dy = a[k] - ma;
        (c + dx * dy, vx + dx * dx, vy + dy * dy)
    });
    let d = (vx * vy).sqrt();
    if d > 0.0 {
        cov / d
    } else {
        f32::NAN
    }
}
fn arr_hhi(a: &[f32]) -> f32 {
    let s: f32 = a.iter().sum();
    if s <= 0.0 {
        return f32::NAN;
    }
    a.iter().map(|&v| (v / s).powi(2)).sum::<f32>()
}

// ============================================================================
// 小工具：序列(跳 NaN)
// ============================================================================

fn seq_mean(v: &[f32]) -> f32 {
    let (s, n) = v
        .iter()
        .copied()
        .filter(|x| x.is_finite())
        .fold((0.0f32, 0usize), |(s, n), x| (s + x, n + 1));
    if n > 0 {
        s / n as f32
    } else {
        f32::NAN
    }
}
fn seq_std(v: &[f32]) -> f32 {
    let valid: Vec<f32> = v.iter().copied().filter(|x| x.is_finite()).collect();
    let n = valid.len();
    if n < 2 {
        return f32::NAN;
    }
    let m = valid.iter().sum::<f32>() / n as f32;
    let sq: f32 = valid.iter().map(|&x| (x - m).powi(2)).sum();
    (sq / (n - 1) as f32).sqrt()
}
fn seq_corr(a: &[f32], b: &[f32]) -> f32 {
    let pairs: Vec<(f32, f32)> = a
        .iter()
        .zip(b.iter())
        .filter(|(x, y)| x.is_finite() && y.is_finite())
        .map(|(x, y)| (*x, *y))
        .collect();
    let n = pairs.len();
    if n < 2 {
        return f32::NAN;
    }
    let ma = pairs.iter().map(|(x, _)| *x).sum::<f32>() / n as f32;
    let mb = pairs.iter().map(|(_, y)| *y).sum::<f32>() / n as f32;
    let (cov, va, vb) = pairs
        .iter()
        .fold((0.0f32, 0.0f32, 0.0f32), |(c, va, vb), (x, y)| {
            let dx = x - ma;
            let dy = y - mb;
            (c + dx * dy, va + dx * dx, vb + dy * dy)
        });
    let d = (va * vb).sqrt();
    if d > 0.0 {
        cov / d
    } else {
        f32::NAN
    }
}
fn seq_autocorr(v: &[f32], lag: usize) -> f32 {
    let n = v.len();
    if n <= lag || lag < 1 {
        return f32::NAN;
    }
    seq_corr(&v[..n - lag], &v[lag..])
}
fn seq_trend(v: &[f32]) -> f32 {
    let n = v.len();
    if n < 2 {
        return f32::NAN;
    }
    let xs: Vec<f32> = (1..=n).map(|x| x as f32).collect();
    seq_corr(&xs, v)
}

fn column(flat: &[f32], n: usize, k: usize, col: usize) -> Vec<f32> {
    (0..n).map(|r| flat[r * k + col]).collect()
}

// 滚动统计：window=w 过去含当前，前 w-1 为 NaN
fn roll_apply<F: Fn(&[f32]) -> f32>(seq: &[f32], w: usize, f: F) -> Vec<f32> {
    let n = seq.len();
    let mut out = vec![f32::NAN; n];
    if w == 0 || n < w {
        return out;
    }
    for end in w..=n {
        out[end - 1] = f(&seq[end - w..end]);
    }
    out
}
fn roll_cov(a: &[f32], b: &[f32], w: usize) -> Vec<f32> {
    let n = a.len().min(b.len());
    let mut out = vec![f32::NAN; n];
    if w == 0 || n < w {
        return out;
    }
    for end in w..=n {
        out[end - 1] = seq_corr(&a[end - w..end], &b[end - w..end]);
    }
    out
}

// 一阶差分 [t]-[t-1]，[0]=NaN，NaN 不传播
fn delta_seq(seq: &[f32]) -> Vec<f32> {
    let n = seq.len();
    let mut out = vec![f32::NAN; n];
    for t in 1..n {
        if seq[t].is_finite() && seq[t - 1].is_finite() {
            out[t] = seq[t] - seq[t - 1];
        }
    }
    out
}
// 窗口首尾差 seq[t-1]-seq[t-w]，用于 OFI 近似（窗口净变化）
fn win_diff(seq: &[f32], w: usize) -> Vec<f32> {
    let n = seq.len();
    let mut out = vec![f32::NAN; n];
    if w == 0 || n < w {
        return out;
    }
    for t in w..=n {
        let cur = seq[t - 1];
        let prev = seq[t - w];
        if cur.is_finite() && prev.is_finite() {
            out[t - 1] = cur - prev;
        }
    }
    out
}

// ============================================================================
// 逐笔窗口聚合（每快照向前 TRADE_WIN_SEC 秒）
// ============================================================================

/// 返回 (buy_vol, sell_vol, cnt, amt, last_px)，每快照一个值。
fn trade_windows(
    trade: &[crate::fast_csv_reader::TradeRecord],
    mkt_times: &[f32],
    win_sec: f32,
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let n = mkt_times.len();
    let mut buy = vec![0.0f32; n];
    let mut sell = vec![0.0f32; n];
    let mut cnt = vec![0.0f32; n];
    let mut amt = vec![0.0f32; n];
    let mut last = vec![f32::NAN; n];
    for t in 0..n {
        let t1 = mkt_times[t];
        let t0 = t1 - win_sec;
        let lo = trade.partition_point(|tr| tr.time_sec < t0);
        let hi = trade.partition_point(|tr| tr.time_sec <= t1);
        for i in lo..hi {
            let tr = &trade[i];
            if tr.flag == 66 {
                buy[t] += tr.volume;
            } else if tr.flag == 83 {
                sell[t] += tr.volume;
            }
            cnt[t] += 1.0;
            amt[t] += tr.turnover;
            last[t] = tr.price; // trade 已按时间排序，取最后一个
        }
    }
    (buy, sell, cnt, amt, last)
}

// ============================================================================
// 核心计算：全天 before/r1/r2 三个 (n×40) + 日频 before/r1/r2 三个 [40]
// ============================================================================

pub fn compute_individual_order_ratio(
    code: &str,
    date: i64,
) -> std::io::Result<(
    Vec<f32>,
    Vec<f32>,
    Vec<f32>,
    usize,
    [f32; DAY_M],
    [f32; DAY_M],
    [f32; DAY_M],
)> {
    use crate::fast_csv_reader::{read_market_fast_inner, read_trade_fast_inner};
    let market = read_market_fast_inner(code, date, false, true, usize::MAX)?;
    let trade = read_trade_fast_inner(code, date, false, true, usize::MAX)?;
    let n = market.len();
    let w = ROLL_N;

    // ---- 全天基础序列 ----
    let mut obi1 = vec![f32::NAN; n];
    let mut wobi = vec![f32::NAN; n];
    let mut mpgap = vec![f32::NAN; n];
    let mut obi1_r1 = vec![f32::NAN; n];
    let mut wobi_r1 = vec![f32::NAN; n];
    let mut mpgap_r1 = vec![f32::NAN; n];
    let mut obi1_r2 = vec![f32::NAN; n];
    let mut wobi_r2 = vec![f32::NAN; n];
    let mut mpgap_r2 = vec![f32::NAN; n];
    let mut a1 = vec![f32::NAN; n];
    let mut b1 = vec![f32::NAN; n];
    let mut ra1 = vec![f32::NAN; n];
    let mut rb1 = vec![f32::NAN; n];
    let mut ra1_full = vec![f32::NAN; n]; // Σra[0..10]
    let mut rb1_full = vec![f32::NAN; n];
    let mut mid_seq = vec![f32::NAN; n];
    let mut mp_seq = vec![f32::NAN; n];
    let mut mp_r1_seq = vec![f32::NAN; n];
    let mut mp_r2_seq = vec![f32::NAN; n];
    let mut mkt_times = vec![0.0f32; n];

    for t in 0..n {
        let m = &market[t];
        let a = &m.ask_vols;
        let b = &m.bid_vols;
        let ap = &m.ask_prcs;
        let bp = &m.bid_prcs;
        let ta = m.total_ask_vol;
        let tb = m.total_bid_vol;
        let ob = ta + tb;
        let mid = (ap[0] + bp[0]) / 2.0;
        mid_seq[t] = mid;
        mkt_times[t] = m.time_sec;
        a1[t] = a[0];
        b1[t] = b[0];

        // before
        let s1 = a[0] + b[0];
        obi1[t] = if s1 > 0.0 {
            (a[0] - b[0]) / s1
        } else {
            f32::NAN
        };
        let wa: f32 = (0..10).map(|k| W[k] * a[k]).sum();
        let wb: f32 = (0..10).map(|k| W[k] * b[k]).sum();
        let sw = wa + wb;
        wobi[t] = if sw != 0.0 { (wa - wb) / sw } else { f32::NAN };
        if s1 > 0.0 && mid > 0.0 {
            let mp = (bp[0] * a[0] + ap[0] * b[0]) / s1;
            mp_seq[t] = mp;
            mpgap[t] = (mp - mid) / mid;
        }

        // r1 (rate1: a/TA - b/TB)
        if ta > 0.0 && tb > 0.0 {
            ra1[t] = a[0] / ta;
            rb1[t] = b[0] / tb;
            obi1_r1[t] = ra1[t] - rb1[t];
            let war: f32 = (0..10).map(|k| W[k] * a[k] / ta).sum();
            let wbr: f32 = (0..10).map(|k| W[k] * b[k] / tb).sum();
            wobi_r1[t] = war - wbr;
            ra1_full[t] = 1.0; // Σa/TA = 1
            rb1_full[t] = 1.0;
            let denom = ra1[t] + rb1[t];
            if denom != 0.0 && mid > 0.0 {
                let mpr = (bp[0] * ra1[t] + ap[0] * rb1[t]) / denom;
                mp_r1_seq[t] = mpr;
                mpgap_r1[t] = (mpr - mid) / mid;
            }
        }
        // r2 (rate2: a/Σa10 - b/Σb10，单侧前10档总和归一，对齐用户原始 rate2 定义)
        let sa10: f32 = a.iter().sum();
        let sb10: f32 = b.iter().sum();
        if sa10 > 0.0 && sb10 > 0.0 {
            let Ra1 = a[0] / sa10;
            let Rb1 = b[0] / sb10;
            obi1_r2[t] = Ra1 - Rb1;
            let waR: f32 = (0..10).map(|k| W[k] * a[k] / sa10).sum();
            let wbR: f32 = (0..10).map(|k| W[k] * b[k] / sb10).sum();
            wobi_r2[t] = waR - wbR;
            let denom = Ra1 + Rb1;
            if denom != 0.0 && mid > 0.0 {
                let mpR = (bp[0] * Ra1 + ap[0] * Rb1) / denom;
                mp_r2_seq[t] = mpR;
                mpgap_r2[t] = (mpR - mid) / mid;
            }
        }
    }

    // ---- 滚动统计 ----
    let rmean_obi1 = roll_apply(&obi1, w, seq_mean);
    let rstd_obi1 = roll_apply(&obi1, w, seq_std);
    let rmean_obi1_r1 = roll_apply(&obi1_r1, w, seq_mean);
    let rstd_obi1_r1 = roll_apply(&obi1_r1, w, seq_std);
    let rmean_obi1_r2 = roll_apply(&obi1_r2, w, seq_mean);
    let rstd_obi1_r2 = roll_apply(&obi1_r2, w, seq_std);
    let rmean_wobi = roll_apply(&wobi, w, seq_mean);
    let rmean_wobi_r1 = roll_apply(&wobi_r1, w, seq_mean);
    let rmean_wobi_r2 = roll_apply(&wobi_r2, w, seq_mean);
    let rcov_a1b1 = roll_cov(&a1, &b1, w);
    let rcov_ra1rb1 = roll_cov(&ra1, &rb1, w);
    let rcov_Ra1Rb1 = roll_cov(&obi1_r2, &wobi_r2, w); // r2 口径 obi1 与 wobi 的滚动协变（cov(Ra1,Rb1) 的代理）
    let ac1_obi1 = roll_apply(&obi1, w, |s| seq_autocorr(s, 1));
    let ac1_obi1_r1 = roll_apply(&obi1_r1, w, |s| seq_autocorr(s, 1));
    let ac1_obi1_r2 = roll_apply(&obi1_r2, w, |s| seq_autocorr(s, 1));
    let trend_obi1 = roll_apply(&obi1, w, seq_trend);
    let trend_obi1_r1 = roll_apply(&obi1_r1, w, seq_trend);
    let trend_obi1_r2 = roll_apply(&obi1_r2, w, seq_trend);

    // ---- 差分 ----
    let d_obi1 = delta_seq(&obi1);
    let d_wobi = delta_seq(&wobi);
    let d_mpgap = delta_seq(&mpgap);
    let d_obi1_r1 = delta_seq(&obi1_r1);
    let d_wobi_r1 = delta_seq(&wobi_r1);
    let d_mpgap_r1 = delta_seq(&mpgap_r1);
    let d_obi1_r2 = delta_seq(&obi1_r2);
    let d_wobi_r2 = delta_seq(&wobi_r2);
    let d_mpgap_r2 = delta_seq(&mpgap_r2);

    // ---- OFI 近似（窗口深度净变化差：win_diff(b1) - win_diff(a1)）----
    let wdb = win_diff(&b1, w);
    let wda = win_diff(&a1, w);
    let wdrb = win_diff(&rb1, w);
    let wdra = win_diff(&ra1, w);
    // r2 近似用 r1 的 rate（OB 归一与 TA/TB 归一在差分上同尺度，仅差常数因子 ob）
    let wdRb = win_diff(&rb1, w);
    let wdRa = win_diff(&ra1, w);

    // ---- 逐笔窗口聚合 ----
    let (buy_win, sell_win, cnt_win, amt_win, last_px) =
        trade_windows(&trade, &mkt_times, TRADE_WIN_SEC);

    // ---- 填充三个 (n×40) ----
    let mut before = vec![f32::NAN; n * SNAP_K];
    let mut after_r1 = vec![f32::NAN; n * SNAP_K];
    let mut after_r2 = vec![f32::NAN; n * SNAP_K];

    for t in 0..n {
        let m = &market[t];
        let a = &m.ask_vols;
        let b = &m.bid_vols;
        let ap = &m.ask_prcs;
        let bp = &m.bid_prcs;
        let ta = m.total_ask_vol;
        let tb = m.total_bid_vol;
        let ob = ta + tb;
        let mid = mid_seq[t];

        // rate1 / rate2 的 10 档（若无效则全 NaN）
        let r1_ok = ta > 0.0 && tb > 0.0;
        let sa10: f32 = a.iter().sum();
        let sb10: f32 = b.iter().sum();
        let r2_ok = sa10 > 0.0 && sb10 > 0.0;
        let ra: [f32; 10] = if r1_ok {
            let mut r = [0.0f32; 10];
            for k in 0..10 {
                r[k] = a[k] / ta;
            }
            r
        } else {
            [f32::NAN; 10]
        };
        let rb: [f32; 10] = if r1_ok {
            let mut r = [0.0f32; 10];
            for k in 0..10 {
                r[k] = b[k] / tb;
            }
            r
        } else {
            [f32::NAN; 10]
        };
        let Ra: [f32; 10] = if r2_ok {
            let mut r = [0.0f32; 10];
            for k in 0..10 {
                r[k] = a[k] / sa10;
            }
            r
        } else {
            [f32::NAN; 10]
        };
        let Rb: [f32; 10] = if r2_ok {
            let mut r = [0.0f32; 10];
            for k in 0..10 {
                r[k] = b[k] / sb10;
            }
            r
        } else {
            [f32::NAN; 10]
        };

        // 逐笔派生（三版共用成交部分）
        let vol_win = buy_win[t] + sell_win[t];
        let buy_share = if vol_win > 0.0 {
            buy_win[t] / vol_win
        } else {
            f32::NAN
        };
        let vpin = if vol_win > 0.0 {
            (buy_win[t] - sell_win[t]).abs() / vol_win
        } else {
            f32::NAN
        };
        let volimb = if vol_win > 0.0 {
            (buy_win[t] - sell_win[t]) / vol_win
        } else {
            f32::NAN
        };
        let intensity = cnt_win[t];
        let avgsize = if cnt_win[t] > 0.0 {
            vol_win / cnt_win[t]
        } else {
            f32::NAN
        };
        let lastp = last_px[t];
        let eff_spread = if lastp.is_finite() && mid > 0.0 {
            2.0 * (lastp - mid).abs() / mid
        } else {
            f32::NAN
        };

        // ofi（三版）
        let ofi_b = if wdb[t].is_finite() && wda[t].is_finite() {
            wdb[t] - wda[t]
        } else {
            f32::NAN
        };
        let ofi_r1 = if wdrb[t].is_finite() && wdra[t].is_finite() {
            wdrb[t] - wdra[t]
        } else {
            f32::NAN
        };
        let ofi_r2 = if wdRb[t].is_finite() && wdRa[t].is_finite() {
            wdRb[t] - wdRa[t]
        } else {
            f32::NAN
        };

        // spread_rel（三版相同，价差不涉及量）
        let spread_rel = if mid > 0.0 {
            (ap[0] - bp[0]) / mid
        } else {
            f32::NAN
        };

        // ---- before ----
        let mut br = [f32::NAN; SNAP_K];
        for k in 0..9usize {
            let ak: f32 = a[0..=k].iter().sum();
            let bk: f32 = b[0..=k].iter().sum();
            br[k] = if ak + bk > 0.0 {
                (ak - bk) / (ak + bk)
            } else {
                f32::NAN
            };
        }
        br[9] = wobi[t];
        br[10] = mpgap[t];
        br[11] = arr_corr_x(a);
        br[12] = arr_corr_x(b);
        br[13] = arr_std(a);
        br[14] = arr_std(b);
        br[15] = arr_skew(a);
        br[16] = arr_skew(b);
        br[17] = arr_hhi(a);
        br[18] = arr_hhi(b);
        br[19] = spread_rel;
        br[20] = d_obi1[t];
        br[21] = rmean_obi1[t];
        br[22] = rstd_obi1[t];
        br[23] = if rstd_obi1[t].is_finite()
            && rstd_obi1[t] != 0.0
            && rmean_obi1[t].is_finite()
            && obi1[t].is_finite()
        {
            (obi1[t] - rmean_obi1[t]) / rstd_obi1[t]
        } else {
            f32::NAN
        };
        br[24] = d_wobi[t];
        br[25] = rmean_wobi[t];
        br[26] = d_mpgap[t];
        br[27] = rcov_a1b1[t];
        br[28] = ac1_obi1[t];
        br[29] = trend_obi1[t];
        br[30] = eff_spread;
        br[31] = if buy_share.is_finite() && obi1[t].is_finite() {
            buy_share - obi1[t]
        } else {
            f32::NAN
        };
        br[32] = if a[0] > 0.0 { vol_win / a[0] } else { f32::NAN };
        br[33] = if b[0] > 0.0 { vol_win / b[0] } else { f32::NAN };
        br[34] = ofi_b;
        br[35] = vpin;
        br[36] = volimb;
        br[37] = intensity;
        br[38] = avgsize;
        br[39] = if lastp.is_finite() && mp_seq[t].is_finite() {
            lastp - mp_seq[t]
        } else {
            f32::NAN
        };
        before[t * SNAP_K..(t + 1) * SNAP_K].copy_from_slice(&br);

        // ---- after_r1 ----
        let mut r1 = [f32::NAN; SNAP_K];
        if r1_ok {
            for k in 0..9usize {
                let ak: f32 = ra[0..=k].iter().sum();
                let bk: f32 = rb[0..=k].iter().sum();
                r1[k] = ak - bk;
            }
            r1[9] = wobi_r1[t];
            r1[10] = mpgap_r1[t];
            r1[11] = arr_corr_x(&ra);
            r1[12] = arr_corr_x(&rb);
            r1[13] = arr_std(&ra);
            r1[14] = arr_std(&rb);
            r1[15] = arr_skew(&ra);
            r1[16] = arr_skew(&rb);
            r1[17] = if ra.iter().all(|v| v.is_finite()) {
                ra.iter().map(|v| v * v).sum::<f32>()
            } else {
                f32::NAN
            };
            r1[18] = if rb.iter().all(|v| v.is_finite()) {
                rb.iter().map(|v| v * v).sum::<f32>()
            } else {
                f32::NAN
            };
        }
        r1[19] = spread_rel;
        r1[20] = d_obi1_r1[t];
        r1[21] = rmean_obi1_r1[t];
        r1[22] = rstd_obi1_r1[t];
        r1[23] = if rstd_obi1_r1[t].is_finite()
            && rstd_obi1_r1[t] != 0.0
            && rmean_obi1_r1[t].is_finite()
            && obi1_r1[t].is_finite()
        {
            (obi1_r1[t] - rmean_obi1_r1[t]) / rstd_obi1_r1[t]
        } else {
            f32::NAN
        };
        r1[24] = d_wobi_r1[t];
        r1[25] = rmean_wobi_r1[t];
        r1[26] = d_mpgap_r1[t];
        r1[27] = rcov_ra1rb1[t];
        r1[28] = ac1_obi1_r1[t];
        r1[29] = trend_obi1_r1[t];
        r1[30] = eff_spread;
        r1[31] = if buy_share.is_finite() && obi1_r1[t].is_finite() {
            buy_share - obi1_r1[t]
        } else {
            f32::NAN
        };
        r1[32] = if ra[0].is_finite() {
            vol_win * ra[0]
        } else {
            f32::NAN
        };
        r1[33] = if rb[0].is_finite() {
            vol_win * rb[0]
        } else {
            f32::NAN
        };
        r1[34] = ofi_r1;
        r1[35] = vpin;
        r1[36] = volimb;
        r1[37] = intensity;
        r1[38] = avgsize;
        r1[39] = if lastp.is_finite() && mp_r1_seq[t].is_finite() {
            lastp - mp_r1_seq[t]
        } else {
            f32::NAN
        };
        after_r1[t * SNAP_K..(t + 1) * SNAP_K].copy_from_slice(&r1);

        // ---- after_r2 ----
        let mut r2 = [f32::NAN; SNAP_K];
        if r2_ok {
            for k in 0..9usize {
                let ak: f32 = Ra[0..=k].iter().sum();
                let bk: f32 = Rb[0..=k].iter().sum();
                r2[k] = ak - bk;
            }
            r2[9] = wobi_r2[t];
            r2[10] = mpgap_r2[t];
            r2[11] = arr_corr_x(&Ra);
            r2[12] = arr_corr_x(&Rb);
            r2[13] = arr_std(&Ra);
            r2[14] = arr_std(&Rb);
            r2[15] = arr_skew(&Ra);
            r2[16] = arr_skew(&Rb);
            // hhi_r2: Σ(Ra)^2 / (ΣRa)^2 形式（ΣRa=TA/OB）
            let sa: f32 = Ra.iter().copied().filter(|v| v.is_finite()).sum();
            let sb: f32 = Rb.iter().copied().filter(|v| v.is_finite()).sum();
            r2[17] = if sa > 0.0 {
                Ra.iter()
                    .map(|v| if v.is_finite() { v * v } else { 0.0 })
                    .sum::<f32>()
                    / (sa * sa)
            } else {
                f32::NAN
            };
            r2[18] = if sb > 0.0 {
                Rb.iter()
                    .map(|v| if v.is_finite() { v * v } else { 0.0 })
                    .sum::<f32>()
                    / (sb * sb)
            } else {
                f32::NAN
            };
        }
        r2[19] = spread_rel;
        r2[20] = d_obi1_r2[t];
        r2[21] = rmean_obi1_r2[t];
        r2[22] = rstd_obi1_r2[t];
        r2[23] = if rstd_obi1_r2[t].is_finite()
            && rstd_obi1_r2[t] != 0.0
            && rmean_obi1_r2[t].is_finite()
            && obi1_r2[t].is_finite()
        {
            (obi1_r2[t] - rmean_obi1_r2[t]) / rstd_obi1_r2[t]
        } else {
            f32::NAN
        };
        r2[24] = d_wobi_r2[t];
        r2[25] = rmean_wobi_r2[t];
        r2[26] = d_mpgap_r2[t];
        r2[27] = rcov_Ra1Rb1[t];
        r2[28] = ac1_obi1_r2[t];
        r2[29] = trend_obi1_r2[t];
        r2[30] = eff_spread;
        r2[31] = if buy_share.is_finite() && obi1_r2[t].is_finite() {
            buy_share - obi1_r2[t]
        } else {
            f32::NAN
        };
        r2[32] = if Ra[0].is_finite() {
            vol_win * Ra[0]
        } else {
            f32::NAN
        };
        r2[33] = if Rb[0].is_finite() {
            vol_win * Rb[0]
        } else {
            f32::NAN
        };
        r2[34] = ofi_r2;
        r2[35] = vpin;
        r2[36] = volimb;
        r2[37] = intensity;
        r2[38] = avgsize;
        r2[39] = if lastp.is_finite() && mp_r2_seq[t].is_finite() {
            lastp - mp_r2_seq[t]
        } else {
            f32::NAN
        };
        after_r2[t * SNAP_K..(t + 1) * SNAP_K].copy_from_slice(&r2);
    }

    // ---- 日频：40 列各自的全天均值 ----
    let mut day_before = [f32::NAN; DAY_M];
    let mut day_r1 = [f32::NAN; DAY_M];
    let mut day_r2 = [f32::NAN; DAY_M];
    for k in 0..DAY_M {
        day_before[k] = seq_mean(&column(&before, n, SNAP_K, k));
        day_r1[k] = seq_mean(&column(&after_r1, n, SNAP_K, k));
        day_r2[k] = seq_mean(&column(&after_r2, n, SNAP_K, k));
    }

    Ok((before, after_r1, after_r2, n, day_before, day_r1, day_r2))
}

// ============================================================================
// 降维 + 拼接 → 固定 OUT_LEN=7900 维
// ============================================================================

pub fn compute_individual_order_ratio_full(code: &str, date: i64) -> std::io::Result<Vec<f32>> {
    let (before, after_r1, after_r2, n, day_before, day_r1, day_r2) =
        compute_individual_order_ratio(code, date)?;
    let snap_names = snap_col_names();

    let feats_of = |flat: &[f32]| -> Vec<f32> {
        if n == 0 {
            return vec![f32::NAN; FEAT_PER_GROUP];
        }
        let arr = Array2::from_shape_vec((n, SNAP_K), flat.to_vec())
            .unwrap_or_else(|_| Array2::zeros((0, SNAP_K)));
        if arr.nrows() == 0 {
            vec![f32::NAN; FEAT_PER_GROUP]
        } else {
            let (vals, _) =
                features::get_features_factors_rust_full(&arr.view(), &snap_names, false);
            vals
        }
    };

    let f_before = feats_of(&before);
    let f_r1 = feats_of(&after_r1);
    let f_r2 = feats_of(&after_r2);
    // diff = after - before（NaN 不传播）
    let mk_diff = |a: &[f32], b: &[f32]| -> Vec<f32> {
        (0..n * SNAP_K)
            .map(|k| {
                if a[k].is_finite() && b[k].is_finite() {
                    a[k] - b[k]
                } else {
                    f32::NAN
                }
            })
            .collect()
    };
    let diff_r1 = mk_diff(&after_r1, &before);
    let diff_r2 = mk_diff(&after_r2, &before);
    let f_diff_r1 = feats_of(&diff_r1);
    let f_diff_r2 = feats_of(&diff_r2);

    let mut out = Vec::with_capacity(OUT_LEN);
    out.extend(f_before);
    out.extend(f_r1);
    out.extend(f_r2);
    out.extend(f_diff_r1);
    out.extend(f_diff_r2);

    // 日频 5 组（各 40）：before, r1, r2, diff_r1, diff_r2
    out.extend(day_before);
    out.extend(day_r1);
    out.extend(day_r2);
    let day_diff_r1: [f32; DAY_M] = std::array::from_fn(|k| {
        if day_r1[k].is_finite() && day_before[k].is_finite() {
            day_r1[k] - day_before[k]
        } else {
            f32::NAN
        }
    });
    let day_diff_r2: [f32; DAY_M] = std::array::from_fn(|k| {
        if day_r2[k].is_finite() && day_before[k].is_finite() {
            day_r2[k] - day_before[k]
        } else {
            f32::NAN
        }
    });
    out.extend(day_diff_r1);
    out.extend(day_diff_r2);

    if out.len() < OUT_LEN {
        out.resize(OUT_LEN, f32::NAN);
    } else if out.len() > OUT_LEN {
        out.truncate(OUT_LEN);
    }
    Ok(out)
}

// ============================================================================
// names
// ============================================================================

pub fn individual_order_ratio_names() -> Vec<String> {
    let snap_names = snap_col_names();
    let dummy = Array2::<f32>::zeros((2, SNAP_K));
    let (_, feat_names) =
        features::get_features_factors_rust_full(&dummy.view(), &snap_names, false);

    let snap_prefixes = [
        "snap_before",
        "snap_after_r1",
        "snap_after_r2",
        "snap_diff_r1",
        "snap_diff_r2",
    ];
    let day_names = day_col_names();
    let day_prefixes = [
        "day_before",
        "day_after_r1",
        "day_after_r2",
        "day_diff_r1",
        "day_diff_r2",
    ];

    let mut out = Vec::with_capacity(OUT_LEN);
    for p in snap_prefixes {
        for fn_ in &feat_names {
            out.push(format!("{}_{}", p, fn_));
        }
    }
    for p in day_prefixes {
        for c in &day_names {
            out.push(format!("{}_{}", p, c));
        }
    }
    out
}

// ============================================================================
// PyO3
// ============================================================================

#[pyfunction]
pub fn py_individual_order_ratio_names() -> Vec<String> {
    individual_order_ratio_names()
}

#[pyfunction]
pub fn py_individual_order_ratio(py: Python<'_>, code: &str, date: i64) -> PyResult<Vec<f32>> {
    let vals = compute_individual_order_ratio_full(code, date)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("{:?}", e)))?;
    Ok(vals)
}
