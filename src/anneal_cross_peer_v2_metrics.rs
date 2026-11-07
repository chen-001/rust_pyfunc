//! 柜员游戏扩展系列3 修正版（anneal_cross_peer_v2）：全市场横截面联动与同伴可还原性。
//!
//! 横截面因子（接入 `run_factor_pipeline_cross_section`，pipeline 名 "anneal_cross_peer_v2"）。
//! v1（anneal_cross_peer_metrics.rs）保持不动，本文件是独立的新版本。与 v1 的区别：
//!   1. 时间切分改用精确到微秒的 time_us。v1 用 time_sec（f32 的 epoch 秒，2024 年附近只能精确到
//!      128 秒），237 个分钟里一半以上是空的，领先滞后相关因此恒为 0。
//!   2. 算不出来的值一律输出 NaN：相关系数样本不足或方差为 0、贝塔、同伴群排名、日内波动率、
//!      早盘/尾盘收益、有效时间格太少的股票的近邻，都不再用 0 / 0.5 / 1.0 / 代码顺序占位。
//!   3. 模块 7 真正把模板排列当作退火起点（v1 实际仍从升序起步，市场模板与同伴模板结果完全相同）。
//!   4. 时间格残差改为横截面标准分（v1 只减均值）；龙头群/冷门群均值剔除本股自身。
//!   5. 时间格收益改为相邻时间格「收盘价对收盘价」（v1 是格内首笔到末笔）。
//!   6. 模块 5b 的 18 个通道改为 6 个核心指标 × 3 方向（v1 是偶数编号通道）。
//!   7. 新增 30 秒时间格尺度，与 60 秒尺度并列。
//!   8. 时段拓展：只用全天数据的模块 1/2/6 在每个尺度上各算 全天 / 下午 / 尾盘30分钟 三段；
//!      模块 3/4/5b 由「全天 + 尾盘」补成「全天 + 下午 + 尾盘」；模块 7 增加下午段。
//!   9. 尾盘 30 分钟取 14:27–14:57（v1 是 14:30–14:57，只有 27 分钟）。
//!
//! 每只股票 55,680 维，顺序：
//!   [m60 全天 | m60 下午 | m60 尾盘30 | s30 全天 | s30 下午 | s30 尾盘30]，每块 8,496 =
//!       模块1 大盘联动 1,152 + 模块2 时间格矩阵降维 6,576 + 模块6 大盘状态响应 768
//!   模块3 同伴群 2,160 + 模块4 价格双胞胎 1,440 + 模块5 复杂度近邻 720 + 模块7 模板退火 384
//!
//! 时间口径：读数据时已做下午前移（13:00 → 11:30），交易时段为相对 09:30 的 [0, 14220] 秒。
//! 股票白名单：沿用 v1 的 symbol_map.csv 过滤。

use crate::anneal_cross_peer_metrics::{list_codes, load_valid_stock_set};
use crate::anneal_regime_diff_metrics::{
    adaptive_steps, anneal_core16, FastAnnealBuf, CORE_NAMES, N_CORE,
};
use crate::fast_csv_reader::{read_trade_fast_inner, TradeRecord};
use crate::features;
use ndarray::Array2;
use numpy::PyReadonlyArray2;
use pyo3::prelude::*;
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use std::collections::hash_map::Entry;
use std::io::{Error, ErrorKind};

// ============================================================================
// 常量
// ============================================================================

/// 每个时间格的通道：3 方向(bid/ask/bid_minus_ask) × 16 退火指标 = 48
pub const N_CH: usize = 3 * N_CORE;
pub const DIR3_NAMES: [&str; 3] = ["bid", "ask", "bid_minus_ask"];

pub const N_SUMMARY: usize = 9;
pub const SUMMARY_NAMES: [&str; N_SUMMARY] = [
    "day_ret",
    "late30_ret",
    "early30_ret",
    "log_turnover",
    "intraday_vol",
    "act_buy_ratio",
    "net_inflow_ratio",
    "mean_ord_vol",
    "trade_cnt_log",
];

/// 交易时段总秒数：09:30–11:30（7200 秒）+ 13:00–14:57（7020 秒）
const SESSION_SEC: i64 = 14_220;
const US: i64 = 1_000_000;

/// 时间格尺度：60 秒（237 格）与 30 秒（474 格）
const N_SCALE: usize = 2;
const SCALE_SEC: [i64; N_SCALE] = [60, 30];
const SCALE_TAGS: [&str; N_SCALE] = ["m60", "s30"];

/// 模块 1/2/6 的时段：全天、下午（13:00 起，即前移后 7200 秒）、尾盘 30 分钟（14:27 起）
const N_SEG: usize = 3;
const SEG_START_SEC: [i64; N_SEG] = [0, 7_200, 12_420];
const SEG_TAGS: [&str; N_SEG] = ["full", "pm", "late30"];
/// 模块 1 中「段首 / 段尾」子窗口长度：全天与下午取 30 分钟，尾盘 30 分钟段取 10 分钟
const SEG_EDGE_SEC: [i64; N_SEG] = [1_800, 1_800, 600];

/// 单股整段窗口（模块 3/4/5b）：全天、下午、尾盘 30 分钟
const N_WIN: usize = 3;
const WIN_START_SEC: [i64; N_WIN] = [0, 7_200, 12_420];
const WIN_TAGS: [&str; N_WIN] = ["full", "pm", "late30"];

/// 订单量形态窗口（模块 7）：全天、早盘 30 分钟、下午、尾盘 30 分钟（左闭右开，单位秒）
const N_OWIN: usize = 4;
const OWIN_BOUNDS: [(i64, i64); N_OWIN] = [
    (0, SESSION_SEC + 1),
    (0, 1_800),
    (7_200, SESSION_SEC + 1),
    (12_420, SESSION_SEC + 1),
];
const OWIN_TAGS: [&str; N_OWIN] = ["full", "early30", "pm", "late30"];

/// 模块 5b 跨通道溢出所用的 6 个核心指标（final_r / rel_r_eff / inertia_norm / max_drawdown /
/// posD_ratio / sign_flip_prob），× 3 方向 = 18 个通道
const XCH_CORE: [usize; 6] = [1, 3, 7, 9, 14, 15];
const N_XCH: usize = 3 * XCH_CORE.len();

/// 复杂度近邻网络所用的 60 秒时间格通道：bid_final_r / ask_final_r / bid_mid_deter / bma_final_r
const COMP_NET_CHS: [usize; 4] = [1, 17, 11, 33];
const COMP_NET_NAMES: [&str; 4] = ["cnet_bid_r", "cnet_ask_r", "cnet_bid_deter", "cnet_bma_r"];

pub const N_P1: usize = N_CH * 24; // 1,152
pub const N_RED_COLS: usize = 2 * N_CH; // 96
pub const N_P2: usize = 21 * N_RED_COLS + N_RED_COLS * (N_RED_COLS - 1) / 2; // 6,576
pub const N_P6: usize = 8 * N_CH * 2; // 768
pub const N_BLOCK: usize = N_P1 + N_P2 + N_P6; // 8,496
pub const N_SCALE_SEG: usize = N_SCALE * N_SEG * N_BLOCK; // 50,976
pub const N_P3: usize = 3 * N_WIN * N_CH * 5; // 2,160
pub const N_P4: usize = N_WIN * N_CH * 10; // 1,440
pub const N_P5: usize = 4 * (N_SUMMARY * 8 + N_XCH * 2 * N_WIN); // 720
pub const N_P7: usize = N_OWIN * 2 * 3 * N_CORE; // 384

/// 总维度 55,680
pub const N_FACTORS: usize = N_SCALE_SEG + N_P3 + N_P4 + N_P5 + N_P7;

const EPS: f32 = 1e-6;
/// 相关系数 / 回归最少需要的共同有效点数，不足输出 NaN
const MIN_PAIRS: usize = 10;
/// 近邻数
const K_NN: usize = 15;
/// 模块 7 模板退火步数（64 点序列）
const TMPL_STEPS: usize = 2_400;
const SEED: u64 = 0x9E37_79B9_7F4A_7C15;

#[inline]
fn n_bins(s: usize) -> usize {
    (SESSION_SEC / SCALE_SEC[s]) as usize
}

// ============================================================================
// 退火核心：从任意初始排列出发（与 anneal_regime_diff_metrics 的核心逐行一致，
// 只是初始猜测由调用方放进 buf.guess）。升序起点时结果与 anneal_core16 完全相同。
// ============================================================================

struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    #[inline]
    fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 {
                0xDEAD_BEEF_DEAD_BEEF
            } else {
                seed
            },
        }
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    #[inline]
    fn next_index(&mut self, n: usize) -> usize {
        if n == 0 {
            return 0;
        }
        ((self.next_u64() >> 32).wrapping_mul(n as u64) >> 32) as usize
    }
}

/// 从 buf.guess（必须是 seq 的一个重排）出发做确定性模拟退火，返回 16 个可还原性指标。
fn anneal_core16_from_guess(seq: &[f32], m_max: usize, buf: &mut FastAnnealBuf) -> [f32; N_CORE] {
    let n = seq.len();
    if n < 3 || m_max < 10 || buf.guess.len() != n {
        return [f32::NAN; N_CORE];
    }
    let mut sum = 0.0f64;
    for &v in seq {
        if !v.is_finite() {
            return [f32::NAN; N_CORE];
        }
        sum += v as f64;
    }
    let mean = sum / n as f64;
    let mut sq_sum = 0.0f64;
    for &v in seq {
        let d = v as f64 - mean;
        sq_sum += d * d;
    }
    let sigma2 = sq_sum / n as f64;
    if sigma2 <= 1e-20 {
        return [f32::NAN; N_CORE];
    }
    let mode: u8 = 0;

    let mut s = 0.0f64;
    for k in 0..n {
        let d = buf.guess[k] as f64 - seq[k] as f64;
        s += d * d;
    }
    let denom = 2.0 * sigma2 * n as f64;
    let inv_denom = 1.0 / denom;
    let r0 = (1.0 - s * inv_denom).clamp(-1.0, 1.0);
    let hl_target = (1.0 + r0) * 0.5;
    let s_tol = (sigma2 * n as f64 * 1e-10).max(1e-12);
    let inv_m = 1.0 / (m_max as f64 - 1.0).max(1.0);

    let mut rng = XorShift64::new(SEED ^ ((mode as u64) << 32));
    let mut prev_r = r0;
    let mut final_r = r0;
    let mut peak_r = r0;
    let mut max_dd = 0.0f64;

    let mut a1_step = m_max;
    let mut a2_step = m_max;
    let mut a3_step = m_max;

    let mut inertia_sum = 0.0f64;
    let mut declines = 0u32;
    let mut dr_cnt = 0u32;
    let mut dr_s1 = 0.0f64;
    let mut dr_s2 = 0.0f64;

    let s1_end = m_max / 3;
    let s2_end = 2 * (m_max / 3);
    let mut s1_cnt = 0u32;
    let mut s1_s1 = 0.0f64;
    let mut s1_s2 = 0.0f64;
    let mut s2_cnt = 0u32;
    let mut s2_decl = 0u32;
    let mut s3_cnt = 0u32;
    let mut s3_s1 = 0.0f64;
    let mut s3_s2 = 0.0f64;
    let mut r_at_s2 = r0;

    let mut d_cnt = 0u32;
    let mut d_pos = 0u32;
    let mut d_flips = 0u32;
    let mut prev_d_sign: i8 = 0;

    let mut actual_steps = 0usize;

    for t in 0..m_max {
        let i = rng.next_index(n);
        let mut j = rng.next_index(n);
        if j == i {
            j = (i + 1) % n;
        }

        if d_cnt < 2000 && buf.guess[i] != buf.guess[j] {
            let dv = seq[j] - seq[i];
            d_cnt += 1;
            if dv > 0.0 {
                d_pos += 1;
            }
            let cur_sign: i8 = if dv > 0.0 {
                1
            } else if dv < 0.0 {
                -1
            } else {
                0
            };
            if prev_d_sign != 0 && cur_sign != 0 && cur_sign != prev_d_sign {
                d_flips += 1;
            }
            if cur_sign != 0 {
                prev_d_sign = cur_sign;
            }
        }

        let gi = buf.guess[i];
        let gj = buf.guess[j];
        let ti = seq[i];
        let tj = seq[j];
        let ds = 2.0f32 * (gi - gj) * (ti - tj);
        let ct = sigma2 * (1.0 - t as f64 * inv_m).max(0.0);
        if (ds as f64) < 0.0 || (ds as f64) < ct {
            s += ds as f64;
            buf.guess.swap(i, j);
        }

        let cr = (1.0 - s * inv_denom).clamp(-1.0, 1.0);
        final_r = cr;
        actual_steps = t + 1;

        if t > 0 {
            let dv = cr - prev_r;
            dr_cnt += 1;
            dr_s1 += dv;
            dr_s2 += dv * dv;
            if cr < prev_r {
                declines += 1;
                if t >= s1_end && t < s2_end {
                    s2_decl += 1;
                }
            }
            if t < s1_end {
                s1_cnt += 1;
                s1_s1 += dv;
                s1_s2 += dv * dv;
            } else if t < s2_end {
                s2_cnt += 1;
            } else {
                s3_cnt += 1;
                s3_s1 += dv;
                s3_s2 += dv * dv;
            }
        }

        if a1_step == m_max && cr >= hl_target {
            a1_step = t;
        }
        if a2_step == m_max && cr >= 0.80 {
            a2_step = t;
        }
        if a3_step == m_max && cr >= 0.90 {
            a3_step = t;
        }
        inertia_sum += 1.0 - cr;
        if cr > peak_r {
            peak_r = cr;
        } else {
            let dd = peak_r - cr;
            if dd > max_dd {
                max_dd = dd;
            }
        }
        if t + 1 == s2_end {
            r_at_s2 = cr;
        }
        prev_r = cr;
        if s <= s_tol {
            if t + 1 < s2_end {
                r_at_s2 = cr;
            }
            break;
        }
    }

    let m_f = m_max as f32;
    let act_f = actual_steps.max(1) as f32;
    let r0_f = r0 as f32;
    let fin_f = final_r as f32;

    let dr_std = if dr_cnt >= 2 {
        let nd = dr_cnt as f64;
        ((dr_s2 - dr_s1 * dr_s1 / nd) / (nd - 1.0)).max(0.0).sqrt() as f32
    } else {
        0.0
    };
    let hot_std = if s1_cnt >= 2 {
        let nd = s1_cnt as f64;
        ((s1_s2 - s1_s1 * s1_s1 / nd) / (nd - 1.0)).max(0.0).sqrt()
    } else {
        0.0
    };
    let cold_std = if s3_cnt >= 2 {
        let nd = s3_cnt as f64;
        ((s3_s2 - s3_s1 * s3_s1 / nd) / (nd - 1.0)).max(0.0).sqrt()
    } else {
        0.0
    };

    let mut out = [f32::NAN; N_CORE];
    out[0] = r0_f;
    out[1] = fin_f;
    out[2] = fin_f - r0_f.abs();
    out[3] = (fin_f - r0_f) / (1.0 - r0_f + 1e-4);
    out[4] = a1_step as f32 / m_f;
    out[5] = a2_step as f32 / m_f;
    out[6] = a3_step as f32 / m_f;
    out[7] = (inertia_sum / act_f as f64) as f32;
    out[8] = declines as f32 / act_f;
    out[9] = max_dd as f32;
    out[10] = dr_std;
    out[11] = if s2_cnt > 0 {
        s2_decl as f32 / s2_cnt as f32
    } else {
        0.0
    };
    out[12] = (final_r - r_at_s2) as f32;
    out[13] = (hot_std / (cold_std + 1e-6)) as f32;
    out[14] = if d_cnt > 0 {
        d_pos as f32 / d_cnt as f32
    } else {
        0.5
    };
    out[15] = if d_cnt > 1 {
        d_flips as f32 / (d_cnt - 1) as f32
    } else {
        0.5
    };
    out
}

/// 以模板的高低次序作为初始排列：模板最小的位置放序列最小值，依此类推，再退火。
fn anneal_from_template64(
    seq: &[f32; 64],
    tmpl: &[f32; 64],
    buf: &mut FastAnnealBuf,
) -> [f32; N_CORE] {
    if seq.iter().any(|v| !v.is_finite()) || tmpl.iter().any(|v| !v.is_finite()) {
        return [f32::NAN; N_CORE];
    }
    let mut sorted_vals: [f32; 64] = *seq;
    sorted_vals.sort_by(|a, b| a.total_cmp(b));
    let mut pos: [usize; 64] = std::array::from_fn(|k| k);
    pos.sort_by(|&a, &b| tmpl[a].total_cmp(&tmpl[b]).then(a.cmp(&b)));
    buf.guess.clear();
    buf.guess.resize(64, 0.0);
    for (r, &p) in pos.iter().enumerate() {
        buf.guess[p] = sorted_vals[r];
    }
    anneal_core16_from_guess(seq, TMPL_STEPS, buf)
}

// ============================================================================
// 小工具：均值、相关、回归（样本不足一律 NaN）
// ============================================================================

#[inline]
fn nanmean(v: &[f32]) -> f32 {
    let mut s = 0.0f64;
    let mut c = 0usize;
    for &x in v {
        if x.is_finite() {
            s += x as f64;
            c += 1;
        }
    }
    if c == 0 {
        f32::NAN
    } else {
        (s / c as f64) as f32
    }
}

/// x 对 y 的共同有效点上的 (相关系数, x 对 y 的回归斜率 beta, 截距 alpha)。
fn pair_reg(x: &[f32], y: &[f32]) -> (f32, f32, f32) {
    let (mut n, mut sx, mut sy) = (0usize, 0.0f64, 0.0f64);
    for (&a, &b) in x.iter().zip(y) {
        if a.is_finite() && b.is_finite() {
            n += 1;
            sx += a as f64;
            sy += b as f64;
        }
    }
    if n < MIN_PAIRS {
        return (f32::NAN, f32::NAN, f32::NAN);
    }
    let (mx, my) = (sx / n as f64, sy / n as f64);
    let (mut sxy, mut sxx, mut syy) = (0.0f64, 0.0f64, 0.0f64);
    for (&a, &b) in x.iter().zip(y) {
        if a.is_finite() && b.is_finite() {
            let (da, db) = (a as f64 - mx, b as f64 - my);
            sxy += da * db;
            sxx += da * da;
            syy += db * db;
        }
    }
    let tiny = 1e-18 * n as f64;
    let corr = if sxx > tiny && syy > tiny {
        (sxy / (sxx.sqrt() * syy.sqrt())) as f32
    } else {
        f32::NAN
    };
    let (beta, alpha) = if syy > tiny {
        let b = sxy / syy;
        (b as f32, (mx - b * my) as f32)
    } else {
        (f32::NAN, f32::NAN)
    };
    (corr, beta, alpha)
}

#[inline]
fn pair_corr(x: &[f32], y: &[f32]) -> f32 {
    pair_reg(x, y).0
}

/// 错位相关：corr(a[t], b[t+k])
#[inline]
fn lag_corr(a: &[f32], b: &[f32], k: usize) -> f32 {
    if a.len() <= k {
        return f32::NAN;
    }
    pair_corr(&a[..a.len() - k], &b[k..])
}

/// 横截面百分位排名 [0,1]；有效值少于 2 个时全部 NaN。order_buf 为复用缓冲。
fn rank_pct_into(vals: &[f32], out: &mut [f32], order_buf: &mut Vec<(f32, u32)>) {
    order_buf.clear();
    for (i, &v) in vals.iter().enumerate() {
        if v.is_finite() {
            order_buf.push((v, i as u32));
        }
    }
    out.iter_mut().for_each(|o| *o = f32::NAN);
    let m = order_buf.len();
    if m < 2 {
        return;
    }
    order_buf.sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    let inv = 1.0 / (m - 1) as f32;
    for (r, &(_, i)) in order_buf.iter().enumerate() {
        out[i as usize] = r as f32 * inv;
    }
}

/// 只在有效值上分 10 档；无效值的股票不属于任何档（None）。
fn assign_decile_groups(metric: &[f32]) -> Vec<Option<u8>> {
    let mut idx: Vec<(f32, usize)> = metric
        .iter()
        .enumerate()
        .filter(|(_, v)| v.is_finite())
        .map(|(i, &v)| (v, i))
        .collect();
    idx.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    let m = idx.len();
    let mut grp = vec![None; metric.len()];
    for (pos, &(_, i)) in idx.iter().enumerate() {
        grp[i] = Some(((pos * 10 / m).min(9)) as u8);
    }
    grp
}

/// 在有效值中取最高 / 最低 20% 的位置；有效值少于 5 个时两组都为空。
fn extreme_masks(vals: &[f32]) -> (Vec<bool>, Vec<bool>) {
    let n = vals.len();
    let mut idx: Vec<(f32, usize)> = vals
        .iter()
        .enumerate()
        .filter(|(_, v)| v.is_finite())
        .map(|(i, &v)| (v, i))
        .collect();
    let mut hi = vec![false; n];
    let mut lo = vec![false; n];
    let m = idx.len();
    if m < 5 {
        return (hi, lo);
    }
    idx.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    let k = m / 5;
    for &(_, i) in &idx[..k] {
        lo[i] = true;
    }
    for &(_, i) in &idx[m - k..] {
        hi[i] = true;
    }
    (hi, lo)
}

/// 8 路独立累加的点积，便于编译器向量化
#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0.0f32; 8];
    let ca = a.chunks_exact(8);
    let cb = b.chunks_exact(8);
    let (ra, rb) = (ca.remainder(), cb.remainder());
    for (x, y) in ca.zip(cb) {
        for k in 0..8 {
            acc[k] += x[k] * y[k];
        }
    }
    let mut s = acc.iter().sum::<f32>();
    for (x, y) in ra.iter().zip(rb) {
        s += x * y;
    }
    s
}

// ============================================================================
// 单股中间特征（第 ② 步，per-stock）
// ============================================================================

pub struct StockMidV2 {
    /// 每个尺度：nb × 48 通道（时间格内买单 / 卖单订单量序列的退火指标）
    pub bin_series: [Vec<f32>; N_SCALE],
    /// 每个尺度：nb 个时间格的收盘价对收盘价收益（空格为 NaN）
    pub bin_ret: [Vec<f32>; N_SCALE],
    /// 每个尺度：nb 个时间格的成交额
    pub bin_amt: [Vec<f32>; N_SCALE],
    /// 3 个整段窗口（全天 / 下午 / 尾盘30）× 48 通道
    pub win_ch: [[f32; N_CH]; N_WIN],
    pub summary: [f32; N_SUMMARY],
    /// 4 个窗口 × 2 方向 的订单量形态（按时间顺序重采样到 64 点并除以均值）
    pub ord_seq64: [[[f32; 64]; 2]; N_OWIN],
}

/// 按首次出现顺序累加某一方向（买单号或卖单号）的订单成交量；哈希表与输出缓冲复用。
struct OrdAgg {
    pos: FxHashMap<i64, u32>,
    vols: Vec<f32>,
}

impl OrdAgg {
    fn new() -> Self {
        Self {
            pos: FxHashMap::default(),
            vols: Vec::new(),
        }
    }
    fn run(&mut self, trades: &[TradeRecord], idx: &[u32], bid_side: bool) {
        self.pos.clear();
        self.vols.clear();
        for &k in idx {
            let t = &trades[k as usize];
            let id = if bid_side { t.bid_order } else { t.ask_order };
            match self.pos.entry(id) {
                Entry::Occupied(e) => self.vols[*e.get() as usize] += t.volume,
                Entry::Vacant(e) => {
                    e.insert(self.vols.len() as u32);
                    self.vols.push(t.volume);
                }
            }
        }
    }
}

/// 订单量序列按时间顺序重采样到 64 点并除以均值（少于 4 笔订单为 NaN）
fn resample_time_seq64(seq: &[f32]) -> [f32; 64] {
    let n = seq.len();
    if n < 4 {
        return [f32::NAN; 64];
    }
    let mean = (seq.iter().map(|&v| v as f64).sum::<f64>() / n as f64) as f32;
    if !(mean > 0.0) {
        return [f32::NAN; 64];
    }
    let mut out = [0.0f32; 64];
    for (i, o) in out.iter_mut().enumerate() {
        *o = seq[i * (n - 1) / 63] / mean;
    }
    out
}

#[inline]
fn write_ch(dst: &mut [f32], b: &[f32; N_CORE], a: &[f32; N_CORE]) {
    for k in 0..N_CORE {
        dst[k] = b[k];
        dst[N_CORE + k] = a[k];
        dst[2 * N_CORE + k] = if b[k].is_finite() && a[k].is_finite() {
            b[k] - a[k]
        } else {
            f32::NAN
        };
    }
}

pub fn extract_stock_mid_v2(trades: &[TradeRecord]) -> Option<StockMidV2> {
    let n = trades.len();
    if n < 15 {
        return None;
    }
    let (mut total_vol, mut total_amt, mut buy_amt) = (0.0f64, 0.0f64, 0.0f64);
    for t in trades {
        total_vol += t.volume as f64;
        total_amt += t.turnover as f64;
        if t.flag == 66 {
            buy_amt += t.turnover as f64;
        }
    }
    if total_vol <= 0.0 || total_amt <= 0.0 {
        return None;
    }

    // 按精确时间排序的下标（文件本身基本有序，稳定排序近似线性）
    let mut order: Vec<u32> = (0..n as u32).collect();
    order.sort_by_key(|&k| trades[k as usize].time_us);
    let t0 = trades[order[0] as usize].time_us;
    let open_us = t0 - t0.rem_euclid(86_400 * US) + (9 * 3600 + 30 * 60) * US;
    let offs: Vec<i64> = order
        .iter()
        .map(|&k| trades[k as usize].time_us - open_us)
        .collect();
    let range = |lo_us: i64, hi_us: i64| -> (usize, usize) {
        (
            offs.partition_point(|&o| o < lo_us),
            offs.partition_point(|&o| o < hi_us),
        )
    };
    let price_at = |pos: usize| trades[order[pos] as usize].price;
    let end_us = (SESSION_SEC + 1) * US;

    let mut buf = FastAnnealBuf::new();
    let mut agg_b = OrdAgg::new();
    let mut agg_a = OrdAgg::new();

    // ---- 两个尺度的时间格序列 ----
    let mut bin_series: [Vec<f32>; N_SCALE] = Default::default();
    let mut bin_ret: [Vec<f32>; N_SCALE] = Default::default();
    let mut bin_amt: [Vec<f32>; N_SCALE] = Default::default();
    for s in 0..N_SCALE {
        let nb = n_bins(s);
        let w_us = SCALE_SEC[s] * US;
        let mut ser = vec![f32::NAN; nb * N_CH];
        let mut ret = vec![f32::NAN; nb];
        let mut amt = vec![0.0f32; nb];
        let mut prev_close: Option<f32> = None;
        for b in 0..nb {
            let lo_us = b as i64 * w_us;
            let hi_us = if b + 1 == nb { end_us } else { lo_us + w_us };
            let (lo, hi) = range(lo_us, hi_us);
            if hi == lo {
                continue;
            }
            let close = price_at(hi - 1);
            let refp = prev_close.unwrap_or_else(|| price_at(lo));
            if refp > 0.0 {
                ret[b] = (close - refp) / refp;
            }
            prev_close = Some(close);
            let idx = &order[lo..hi];
            amt[b] = idx.iter().map(|&k| trades[k as usize].turnover).sum();
            if hi - lo < 3 {
                continue;
            }
            agg_b.run(trades, idx, true);
            agg_a.run(trades, idx, false);
            let bc = anneal_core16(&agg_b.vols, adaptive_steps(agg_b.vols.len(), 4_000), &mut buf);
            let ac = anneal_core16(&agg_a.vols, adaptive_steps(agg_a.vols.len(), 4_000), &mut buf);
            write_ch(&mut ser[b * N_CH..(b + 1) * N_CH], &bc, &ac);
        }
        bin_series[s] = ser;
        bin_ret[s] = ret;
        bin_amt[s] = amt;
    }

    // ---- 整段窗口的退火指标 ----
    let mut win_ch = [[f32::NAN; N_CH]; N_WIN];
    for w in 0..N_WIN {
        let (lo, hi) = range(WIN_START_SEC[w] * US, end_us);
        if hi - lo < 3 {
            continue;
        }
        let idx = &order[lo..hi];
        agg_b.run(trades, idx, true);
        agg_a.run(trades, idx, false);
        let bc = anneal_core16(&agg_b.vols, adaptive_steps(agg_b.vols.len(), 15_000), &mut buf);
        let ac = anneal_core16(&agg_a.vols, adaptive_steps(agg_a.vols.len(), 15_000), &mut buf);
        write_ch(&mut win_ch[w], &bc, &ac);
    }

    // ---- 订单量形态 64 点 + 窗口收益 + 全天订单数 ----
    let mut ord_seq64 = [[[f32::NAN; 64]; 2]; N_OWIN];
    let mut full_ord_cnt = 0usize;
    for (w, &(a, b)) in OWIN_BOUNDS.iter().enumerate() {
        let (lo, hi) = range(a * US, b * US);
        if hi == lo {
            continue;
        }
        let idx = &order[lo..hi];
        agg_b.run(trades, idx, true);
        agg_a.run(trades, idx, false);
        if w == 0 {
            full_ord_cnt = agg_b.vols.len() + agg_a.vols.len();
        }
        ord_seq64[w][0] = resample_time_seq64(&agg_b.vols);
        ord_seq64[w][1] = resample_time_seq64(&agg_a.vols);
    }
    let win_ret = |a: i64, b: i64| -> f32 {
        let (lo, hi) = range(a * US, b * US);
        if hi < lo + 2 {
            return f32::NAN;
        }
        let (p0, p1) = (price_at(lo), price_at(hi - 1));
        if p0 > 0.0 {
            (p1 - p0) / p0
        } else {
            f32::NAN
        }
    };
    let day_ret = win_ret(0, SESSION_SEC + 1);
    let early_ret = win_ret(0, 1_800);
    let late_ret = win_ret(12_420, SESSION_SEC + 1);

    // 日内波动率：60 秒时间格收益的标准差（有效格少于 10 个为 NaN）
    let valid_ret = bin_ret[0].iter().filter(|v| v.is_finite()).count();
    let intraday_vol = if valid_ret >= 10 {
        features::col_std(&bin_ret[0])
    } else {
        f32::NAN
    };

    let summary = [
        day_ret,
        late_ret,
        early_ret,
        total_amt.ln() as f32,
        intraday_vol,
        (buy_amt / total_amt) as f32,
        ((2.0 * buy_amt - total_amt) / total_amt) as f32,
        if full_ord_cnt > 0 {
            (total_vol / full_ord_cnt as f64) as f32
        } else {
            f32::NAN
        },
        (n as f32).ln(),
    ];

    Some(StockMidV2 {
        bin_series,
        bin_ret,
        bin_amt,
        win_ch,
        summary,
        ord_seq64,
    })
}

// ============================================================================
// 第 ③ 步：横截面
// ============================================================================

/// 某一尺度上全市场每个时间格的横截面结果（时间格优先存放：[b][i][ch]）
struct ScaleCross {
    nb: usize,
    z: Vec<f32>,
    rk: Vec<f32>,
    mkt_mean: Vec<f32>, // nb × N_CH
    lead_sum: Vec<f64>,
    lead_cnt: Vec<u32>,
    lag_sum: Vec<f64>,
    lag_cnt: Vec<u32>,
    mret: Vec<f32>,
    mdisp: Vec<f32>,
    mamt: Vec<f32>,
}

struct BinAgg {
    mean: [f32; N_CH],
    lead_sum: [f64; N_CH],
    lead_cnt: [u32; N_CH],
    lag_sum: [f64; N_CH],
    lag_cnt: [u32; N_CH],
    mret: f32,
    mdisp: f32,
    mamt: f32,
}

fn build_scale_cross(
    s: usize,
    stocks: &[StockMidV2],
    is_leader: &[bool],
    is_laggard: &[bool],
) -> ScaleCross {
    let n = stocks.len();
    let nb = n_bins(s);
    let mut z = vec![f32::NAN; nb * n * N_CH];
    let mut rk = vec![f32::NAN; nb * n * N_CH];
    let aggs: Vec<BinAgg> = z
        .par_chunks_mut(n * N_CH)
        .zip(rk.par_chunks_mut(n * N_CH))
        .enumerate()
        .map(|(b, (zc, rc))| {
            let mut col = vec![0.0f32; n];
            let mut rcol = vec![0.0f32; n];
            let mut obuf: Vec<(f32, u32)> = Vec::with_capacity(n);
            let mut agg = BinAgg {
                mean: [f32::NAN; N_CH],
                lead_sum: [0.0; N_CH],
                lead_cnt: [0; N_CH],
                lag_sum: [0.0; N_CH],
                lag_cnt: [0; N_CH],
                mret: f32::NAN,
                mdisp: f32::NAN,
                mamt: 0.0,
            };
            for ch in 0..N_CH {
                let (mut sum, mut cnt) = (0.0f64, 0usize);
                for i in 0..n {
                    let v = stocks[i].bin_series[s][b * N_CH + ch];
                    col[i] = v;
                    if v.is_finite() {
                        sum += v as f64;
                        cnt += 1;
                        if is_leader[i] {
                            agg.lead_sum[ch] += v as f64;
                            agg.lead_cnt[ch] += 1;
                        }
                        if is_laggard[i] {
                            agg.lag_sum[ch] += v as f64;
                            agg.lag_cnt[ch] += 1;
                        }
                    }
                }
                if cnt == 0 {
                    continue;
                }
                let mean = sum / cnt as f64;
                agg.mean[ch] = mean as f32;
                let mut ss = 0.0f64;
                for &v in &col {
                    if v.is_finite() {
                        let d = v as f64 - mean;
                        ss += d * d;
                    }
                }
                let sd = if cnt >= 2 {
                    (ss / (cnt - 1) as f64).sqrt()
                } else {
                    f64::NAN
                };
                if sd.is_finite() && sd > 1e-6 {
                    for i in 0..n {
                        let v = col[i];
                        if v.is_finite() {
                            zc[i * N_CH + ch] = ((v as f64 - mean) / sd) as f32;
                        }
                    }
                }
                rank_pct_into(&col, &mut rcol, &mut obuf);
                for i in 0..n {
                    rc[i * N_CH + ch] = rcol[i];
                }
            }
            let rets: Vec<f32> = stocks.iter().map(|st| st.bin_ret[s][b]).collect();
            let valid = rets.iter().filter(|v| v.is_finite()).count();
            if valid > 0 {
                agg.mret = nanmean(&rets);
            }
            if valid >= 2 {
                agg.mdisp = features::col_std(&rets);
            }
            agg.mamt = stocks.iter().map(|st| st.bin_amt[s][b]).sum();
            agg
        })
        .collect();

    let mut sc = ScaleCross {
        nb,
        z,
        rk,
        mkt_mean: vec![f32::NAN; nb * N_CH],
        lead_sum: vec![0.0; nb * N_CH],
        lead_cnt: vec![0; nb * N_CH],
        lag_sum: vec![0.0; nb * N_CH],
        lag_cnt: vec![0; nb * N_CH],
        mret: vec![f32::NAN; nb],
        mdisp: vec![f32::NAN; nb],
        mamt: vec![0.0; nb],
    };
    for (b, a) in aggs.iter().enumerate() {
        let o = b * N_CH;
        sc.mkt_mean[o..o + N_CH].copy_from_slice(&a.mean);
        sc.lead_sum[o..o + N_CH].copy_from_slice(&a.lead_sum);
        sc.lead_cnt[o..o + N_CH].copy_from_slice(&a.lead_cnt);
        sc.lag_sum[o..o + N_CH].copy_from_slice(&a.lag_sum);
        sc.lag_cnt[o..o + N_CH].copy_from_slice(&a.lag_cnt);
        sc.mret[b] = a.mret;
        sc.mdisp[b] = a.mdisp;
        sc.mamt[b] = a.mamt;
    }
    sc
}

/// 剔除本股后的群均值
#[inline]
fn group_mean_excl(sum: f64, cnt: u32, self_in: bool, x: f32) -> f32 {
    if self_in && x.is_finite() {
        if cnt > 1 {
            ((sum - x as f64) / (cnt - 1) as f64) as f32
        } else {
            f32::NAN
        }
    } else if cnt > 0 {
        (sum / cnt as f64) as f32
    } else {
        f32::NAN
    }
}

/// 每个 (尺度, 时段) 的 8 个大盘状态掩码（相对时段起点的下标）
fn seg_masks(sc: &ScaleCross, b0: usize) -> [Vec<bool>; 8] {
    let ease: Vec<f32> = (b0..sc.nb).map(|b| sc.mkt_mean[b * N_CH + 1]).collect();
    let (up, dn) = extreme_masks(&sc.mret[b0..]);
    let (dh, dl) = extreme_masks(&sc.mdisp[b0..]);
    let amt: Vec<f32> = sc.mamt[b0..].to_vec();
    let (ah, al) = extreme_masks(&amt);
    let (eh, el) = extreme_masks(&ease);
    [up, dn, dh, dl, ah, al, eh, el]
}

/// 每个线程复用的缓冲
struct BlockBufs {
    z_loc: Vec<f32>,
    rk_loc: Vec<f32>,
    x: Vec<f32>,
    m: Vec<f32>,
    l: Vec<f32>,
    g: Vec<f32>,
    zs: Vec<f32>,
    rs: Vec<f32>,
}

impl BlockBufs {
    fn new() -> Self {
        let cap = n_bins(N_SCALE - 1);
        Self {
            z_loc: Vec::with_capacity(cap * N_CH),
            rk_loc: Vec::with_capacity(cap * N_CH),
            x: Vec::with_capacity(cap),
            m: Vec::with_capacity(cap),
            l: Vec::with_capacity(cap),
            g: Vec::with_capacity(cap),
            zs: Vec::with_capacity(cap),
            rs: Vec::with_capacity(cap),
        }
    }
}

/// 某只股票在某个尺度的三个时段块（模块 1 / 2 / 6），写入 out（长 N_SEG × N_BLOCK）
#[allow(clippy::too_many_arguments)]
fn fill_scale_blocks(
    out: &mut [f32],
    i: usize,
    s: usize,
    sc: &ScaleCross,
    masks: &[[Vec<bool>; 8]; N_SEG],
    st: &StockMidV2,
    is_leader: bool,
    is_laggard: bool,
    n: usize,
    bb: &mut BlockBufs,
) {
    let nb = sc.nb;
    let w = SCALE_SEC[s];
    // 收集本股的标准分 / 排名矩阵（nb × 48，时间格优先）
    bb.z_loc.clear();
    bb.rk_loc.clear();
    for b in 0..nb {
        let o = (b * n + i) * N_CH;
        bb.z_loc.extend_from_slice(&sc.z[o..o + N_CH]);
        bb.rk_loc.extend_from_slice(&sc.rk[o..o + N_CH]);
    }
    let ser = &st.bin_series[s];

    for seg in 0..N_SEG {
        let b0 = (SEG_START_SEC[seg] / w) as usize;
        let len = nb - b0;
        let edge = ((SEG_EDGE_SEC[seg] / w) as usize).min(len / 2);
        let blk = &mut out[seg * N_BLOCK..(seg + 1) * N_BLOCK];
        let mut p = 0usize;

        // ---------------- 模块 1：大盘共同节奏、领先滞后、龙头 / 冷门群、排名霸榜 ----------------
        for ch in 0..N_CH {
            bb.x.clear();
            bb.m.clear();
            bb.l.clear();
            bb.g.clear();
            bb.zs.clear();
            bb.rs.clear();
            for b in b0..nb {
                let o = b * N_CH + ch;
                let x = ser[o];
                bb.x.push(x);
                bb.m.push(sc.mkt_mean[o]);
                bb.l.push(group_mean_excl(sc.lead_sum[o], sc.lead_cnt[o], is_leader, x));
                bb.g.push(group_mean_excl(sc.lag_sum[o], sc.lag_cnt[o], is_laggard, x));
                bb.zs.push(bb.z_loc[o]);
                bb.rs.push(bb.rk_loc[o]);
            }
            let (corr_m, beta_m, alpha_m) = pair_reg(&bb.x, &bb.m);
            let r2 = corr_m * corr_m;
            let res_e = nanmean(&bb.zs[..edge]);
            let res_m = nanmean(&bb.zs[edge..len - edge]);
            let res_l = nanmean(&bb.zs[len - edge..]);
            let lead1 = lag_corr(&bb.x, &bb.m, 1);
            let lag1 = lag_corr(&bb.m, &bb.x, 1);
            let lead3 = lag_corr(&bb.x, &bb.m, 3);
            let lag3 = lag_corr(&bb.m, &bb.x, 3);
            let lead5 = lag_corr(&bb.x, &bb.m, 5);
            let lag5 = lag_corr(&bb.m, &bb.x, 5);
            let c_lead = pair_corr(&bb.x, &bb.l);
            let c_lag = pair_corr(&bb.x, &bb.g);

            let (mut valid, mut top5, mut top10, mut bot10, mut streak, mut max_streak) =
                (0usize, 0usize, 0usize, 0usize, 0usize, 0usize);
            for &r in &bb.rs {
                if r.is_finite() {
                    valid += 1;
                    if r >= 0.95 {
                        top5 += 1;
                    }
                    if r >= 0.90 {
                        top10 += 1;
                        streak += 1;
                        max_streak = max_streak.max(streak);
                    } else {
                        streak = 0;
                    }
                    if r <= 0.10 {
                        bot10 += 1;
                    }
                }
            }
            let share = |c: usize| {
                if valid > 0 {
                    c as f32 / valid as f32
                } else {
                    f32::NAN
                }
            };
            let rk_e = nanmean(&bb.rs[..edge]);
            let rk_l = nanmean(&bb.rs[len - edge..]);
            let vals = [
                corr_m,
                beta_m,
                alpha_m,
                r2,
                res_e,
                res_m,
                res_l,
                res_l - res_e,
                res_l - res_m,
                lead1,
                lag1,
                lead1 - lag1,
                lead3,
                lag3,
                lead3 - lag3,
                lead5 - lag5,
                c_lead,
                c_lag,
                c_lead - c_lag,
                share(top5),
                share(top10),
                share(bot10),
                share(max_streak),
                rk_l - rk_e,
            ];
            blk[p..p + 24].copy_from_slice(&vals);
            p += 24;
        }

        // ---------------- 模块 2：标准分(48) + 排名(48) 矩阵降维 ----------------
        let mut mat = Array2::<f32>::from_elem((len, N_RED_COLS), f32::NAN);
        for (r, b) in (b0..nb).enumerate() {
            let o = b * N_CH;
            for ch in 0..N_CH {
                mat[[r, ch]] = bb.z_loc[o + ch];
                mat[[r, N_CH + ch]] = bb.rk_loc[o + ch];
            }
        }
        let red = features::get_features_factors_rust_values_only(&mat.view(), false);
        debug_assert_eq!(red.len(), N_P2);
        blk[p..p + N_P2].copy_from_slice(&red);
        p += N_P2;

        // ---------------- 模块 6：大盘状态下的超额响应 ----------------
        for mask in masks[seg].iter() {
            for ch in 0..N_CH {
                let (mut sz, mut cz, mut sr, mut cr) = (0.0f64, 0usize, 0.0f64, 0usize);
                for (r, &on) in mask.iter().enumerate() {
                    if !on {
                        continue;
                    }
                    let o = (b0 + r) * N_CH + ch;
                    let zv = bb.z_loc[o];
                    let rv = bb.rk_loc[o];
                    if zv.is_finite() {
                        sz += zv as f64;
                        cz += 1;
                    }
                    if rv.is_finite() {
                        sr += rv as f64;
                        cr += 1;
                    }
                }
                blk[p] = if cz > 0 { (sz / cz as f64) as f32 } else { f32::NAN };
                blk[p + 1] = if cr > 0 { (sr / cr as f64) as f32 } else { f32::NAN };
                p += 2;
            }
        }
        debug_assert_eq!(p, N_BLOCK);
    }
}

type NbrList = Vec<(u32, f32)>;

/// 对一组归一化向量（n × dim，连续存放；不合格股票为 None）求每只股票的 Top-K / Bottom-K 近邻
fn knn_top_bot(vecs: &[f32], dim: usize, ok: &[bool]) -> Vec<Option<(NbrList, NbrList)>> {
    let n = ok.len();
    (0..n)
        .into_par_iter()
        .map_init(
            || Vec::<(f32, u32)>::with_capacity(n),
            |sims, i| {
                if !ok[i] {
                    return None;
                }
                let vi = &vecs[i * dim..(i + 1) * dim];
                sims.clear();
                for j in 0..n {
                    if j != i && ok[j] {
                        sims.push((dot(vi, &vecs[j * dim..(j + 1) * dim]), j as u32));
                    }
                }
                let m = sims.len();
                if m == 0 {
                    return None;
                }
                let k = K_NN.min(m);
                let desc = |a: &(f32, u32), b: &(f32, u32)| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1));
                let asc = |a: &(f32, u32), b: &(f32, u32)| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1));
                sims.select_nth_unstable_by(k - 1, desc);
                let mut top: Vec<(f32, u32)> = sims[..k].to_vec();
                top.sort_by(desc);
                sims.select_nth_unstable_by(k - 1, asc);
                let mut bot: Vec<(f32, u32)> = sims[..k].to_vec();
                bot.sort_by(asc);
                Some((
                    top.into_iter().map(|(s, j)| (j, s)).collect(),
                    bot.into_iter().map(|(s, j)| (j, s)).collect(),
                ))
            },
        )
        .collect()
}

/// 去均值 + 单位长度；有效点少于一半或方差为 0 时不合格
fn normalize_into(v: &[f32], dst: &mut [f32]) -> bool {
    let (mut s, mut c) = (0.0f64, 0usize);
    for &x in v {
        if x.is_finite() {
            s += x as f64;
            c += 1;
        }
    }
    dst.iter_mut().for_each(|d| *d = 0.0);
    if c * 2 < v.len() || c < 5 {
        return false;
    }
    let mean = s / c as f64;
    let mut ss = 0.0f64;
    for &x in v {
        if x.is_finite() {
            ss += (x as f64 - mean).powi(2);
        }
    }
    let norm = ss.sqrt();
    if norm < 1e-9 {
        return false;
    }
    for (d, &x) in dst.iter_mut().zip(v) {
        if x.is_finite() {
            *d = ((x as f64 - mean) / norm) as f32;
        }
    }
    true
}

/// 近邻上的均值 / 加权均值 / 标准差 / 计数 / 最大 / 最小 / 被本股超过的个数
struct NbrStat {
    mean: f32,
    wmean: f32,
    std: f32,
    max: f32,
    min: f32,
    cnt: usize,
    beat: usize,
}

fn nbr_stat(nbrs: &[(u32, f32)], val: impl Fn(usize) -> f32, x: f32) -> NbrStat {
    let (mut s, mut sq, mut ws, mut wt, mut c, mut beat) = (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0usize, 0usize);
    let (mut mx, mut mn) = (f32::NEG_INFINITY, f32::INFINITY);
    for &(j, sim) in nbrs {
        let v = val(j as usize);
        if v.is_finite() {
            s += v as f64;
            sq += (v as f64) * (v as f64);
            let w = sim.max(0.05) as f64;
            ws += v as f64 * w;
            wt += w;
            c += 1;
            mx = mx.max(v);
            mn = mn.min(v);
            if x.is_finite() && x > v {
                beat += 1;
            }
        }
    }
    if c == 0 {
        return NbrStat {
            mean: f32::NAN,
            wmean: f32::NAN,
            std: f32::NAN,
            max: f32::NAN,
            min: f32::NAN,
            cnt: 0,
            beat: 0,
        };
    }
    let nf = c as f64;
    NbrStat {
        mean: (s / nf) as f32,
        wmean: (ws / wt) as f32,
        std: if c >= 2 {
            ((sq - s * s / nf) / (nf - 1.0)).max(0.0).sqrt() as f32
        } else {
            f32::NAN
        },
        max: mx,
        min: mn,
        cnt: c,
        beat,
    }
}

// ============================================================================
// 横截面主合成核心（无 IO，唯一真相源）
// ============================================================================

pub fn compute_cross_peer_v2_from_mids(
    codes: &[String],
    mids: Vec<Option<StockMidV2>>,
) -> (Vec<String>, Vec<f32>) {
    let mut valid_codes = Vec::new();
    let mut stocks: Vec<StockMidV2> = Vec::new();
    for (c, m) in codes.iter().zip(mids) {
        if let Some(st) = m {
            valid_codes.push(c.clone());
            stocks.push(st);
        }
    }
    let n = stocks.len();
    if n == 0 {
        return (Vec::new(), Vec::new());
    }
    let mut out = vec![f32::NAN; n * N_FACTORS];

    // ---- 龙头群（成交额前 10%，最多 50 只）与冷门群（成交额后 20%） ----
    let mut amt_order: Vec<(f32, usize)> = stocks
        .iter()
        .enumerate()
        .map(|(i, s)| (s.summary[3], i))
        .collect();
    amt_order.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    let k_lead = (n / 10).clamp(1, 50.min(n));
    let k_lag = (n / 5).max(1);
    let mut is_leader = vec![false; n];
    let mut is_laggard = vec![false; n];
    for &(_, i) in &amt_order[n - k_lead..] {
        is_leader[i] = true;
    }
    for &(_, i) in &amt_order[..k_lag] {
        is_laggard[i] = true;
    }

    // ---- 模块 1/2/6：逐个尺度计算，算完即释放该尺度的横截面矩阵 ----
    for s in 0..N_SCALE {
        let sc = build_scale_cross(s, &stocks, &is_leader, &is_laggard);
        let masks: [[Vec<bool>; 8]; N_SEG] =
            std::array::from_fn(|seg| seg_masks(&sc, (SEG_START_SEC[seg] / SCALE_SEC[s]) as usize));
        let off = s * N_SEG * N_BLOCK;
        out.par_chunks_mut(N_FACTORS).enumerate().for_each_init(
            BlockBufs::new,
            |bb, (i, row)| {
                fill_scale_blocks(
                    &mut row[off..off + N_SEG * N_BLOCK],
                    i,
                    s,
                    &sc,
                    &masks,
                    &stocks[i],
                    is_leader[i],
                    is_laggard[i],
                    n,
                    bb,
                );
            },
        );
    }

    // ---- 同伴群（成交额 / 日内波动率 / 主买占比 各 10 档） ----
    let metric = |k: usize| -> Vec<f32> { stocks.iter().map(|s| s.summary[k]).collect() };
    let peer_groups: [Vec<Option<u8>>; 3] = [
        assign_decile_groups(&metric(3)),
        assign_decile_groups(&metric(4)),
        assign_decile_groups(&metric(5)),
    ];
    let mut grp_mean = vec![f32::NAN; 3 * 10 * N_WIN * N_CH];
    let mut grp_std = vec![f32::NAN; 3 * 10 * N_WIN * N_CH];
    let gidx = |g: usize, d: usize, w: usize, ch: usize| ((g * 10 + d) * N_WIN + w) * N_CH + ch;
    let mut grp_rank = vec![f32::NAN; n * 3 * N_WIN * N_CH];
    let ridx = |i: usize, g: usize, w: usize, ch: usize| ((i * 3 + g) * N_WIN + w) * N_CH + ch;
    let mut glob_rank = vec![f32::NAN; n * N_WIN * N_CH];
    {
        let mut col = vec![0.0f32; n];
        let mut rcol = vec![0.0f32; n];
        let mut obuf = Vec::with_capacity(n);
        let members: [Vec<Vec<usize>>; 3] = std::array::from_fn(|g| {
            let mut m = vec![Vec::new(); 10];
            for (i, d) in peer_groups[g].iter().enumerate() {
                if let Some(d) = d {
                    m[*d as usize].push(i);
                }
            }
            m
        });
        for w in 0..N_WIN {
            for ch in 0..N_CH {
                for i in 0..n {
                    col[i] = stocks[i].win_ch[w][ch];
                }
                rank_pct_into(&col, &mut rcol, &mut obuf);
                for i in 0..n {
                    glob_rank[(i * N_WIN + w) * N_CH + ch] = rcol[i];
                }
                for g in 0..3 {
                    for d in 0..10 {
                        let sub: Vec<f32> = members[g][d].iter().map(|&i| col[i]).collect();
                        let valid = sub.iter().filter(|v| v.is_finite()).count();
                        if valid > 0 {
                            grp_mean[gidx(g, d, w, ch)] = nanmean(&sub);
                        }
                        if valid >= 2 {
                            let sd = features::col_std(&sub);
                            if sd.is_finite() && sd > 1e-6 {
                                grp_std[gidx(g, d, w, ch)] = sd;
                            }
                        }
                        let mut sr = vec![0.0f32; sub.len()];
                        rank_pct_into(&sub, &mut sr, &mut obuf);
                        for (k, &i) in members[g][d].iter().enumerate() {
                            grp_rank[ridx(i, g, w, ch)] = sr[k];
                        }
                    }
                }
            }
        }
    }

    // ---- 近邻网络：价格双胞胎（60 秒收益）+ 4 个复杂度近邻 ----
    let nb60 = n_bins(0);
    let build_vecs = |get: &(dyn Fn(&StockMidV2, usize) -> f32 + Sync)| -> (Vec<f32>, Vec<bool>) {
        let mut vecs = vec![0.0f32; n * nb60];
        let ok: Vec<bool> = vecs
            .par_chunks_mut(nb60)
            .enumerate()
            .map(|(i, dst)| {
                let raw: Vec<f32> = (0..nb60).map(|b| get(&stocks[i], b)).collect();
                normalize_into(&raw, dst)
            })
            .collect();
        (vecs, ok)
    };
    let (rv, rok) = build_vecs(&|st, b| st.bin_ret[0][b]);
    let price_nbrs = knn_top_bot(&rv, nb60, &rok);
    drop(rv);
    let comp_nbrs: Vec<Vec<Option<(NbrList, NbrList)>>> = COMP_NET_CHS
        .iter()
        .map(|&ch| {
            let (cv, cok) = build_vecs(&|st, b| st.bin_series[0][b * N_CH + ch]);
            knn_top_bot(&cv, nb60, &cok)
        })
        .collect();

    // ---- 模块 7 模板：全市场与同体量群的订单量形态（之后对每只股票剔除自身） ----
    let size_grp = &peer_groups[0];
    let mut t_sum = vec![[0.0f64; 64]; N_OWIN * 2];
    let mut t_cnt = vec![0u32; N_OWIN * 2];
    let mut g_sum = vec![[0.0f64; 64]; 10 * N_OWIN * 2];
    let mut g_cnt = vec![0u32; 10 * N_OWIN * 2];
    for (i, st) in stocks.iter().enumerate() {
        for w in 0..N_OWIN {
            for d in 0..2 {
                let sq = &st.ord_seq64[w][d];
                if !sq[0].is_finite() {
                    continue;
                }
                let k = w * 2 + d;
                t_cnt[k] += 1;
                for p in 0..64 {
                    t_sum[k][p] += sq[p] as f64;
                }
                if let Some(gd) = size_grp[i] {
                    let gk = gd as usize * N_OWIN * 2 + k;
                    g_cnt[gk] += 1;
                    for p in 0..64 {
                        g_sum[gk][p] += sq[p] as f64;
                    }
                }
            }
        }
    }
    let tmpl_excl = |sum: &[f64; 64], cnt: u32, own: &[f32; 64]| -> [f32; 64] {
        let own_in = own[0].is_finite();
        let c = if own_in { cnt.saturating_sub(1) } else { cnt };
        if c == 0 {
            return [f32::NAN; 64];
        }
        std::array::from_fn(|p| {
            let s = if own_in { sum[p] - own[p] as f64 } else { sum[p] };
            (s / c as f64) as f32
        })
    };

    // ---- 模块 3 / 4 / 5 / 7：逐股票并行写出 ----
    out.par_chunks_mut(N_FACTORS).enumerate().for_each_init(
        FastAnnealBuf::new,
        |buf, (i, row)| {
            let st = &stocks[i];
            let tail = &mut row[N_SCALE_SEG..];
            let mut p = 0usize;

            // 模块 3：同伴群相对偏离
            for g in 0..3 {
                for w in 0..N_WIN {
                    for ch in 0..N_CH {
                        let x = st.win_ch[w][ch];
                        let (gm, gs) = match peer_groups[g][i] {
                            Some(d) => (
                                grp_mean[gidx(g, d as usize, w, ch)],
                                grp_std[gidx(g, d as usize, w, ch)],
                            ),
                            None => (f32::NAN, f32::NAN),
                        };
                        let grk = grp_rank[ridx(i, g, w, ch)];
                        let glr = glob_rank[(i * N_WIN + w) * N_CH + ch];
                        tail[p] = x - gm;
                        tail[p + 1] = (x - gm) / gs;
                        tail[p + 2] = x / (gm.abs() + EPS);
                        tail[p + 3] = grk;
                        tail[p + 4] = grk - glr;
                        p += 5;
                    }
                }
            }
            debug_assert_eq!(p, N_P3);

            // 模块 4：价格双胞胎
            for w in 0..N_WIN {
                for ch in 0..N_CH {
                    let x = st.win_ch[w][ch];
                    let vals: [f32; 10] = match &price_nbrs[i] {
                        Some((top, bot)) if x.is_finite() => {
                            let t = nbr_stat(top, |j| stocks[j].win_ch[w][ch], x);
                            let b = nbr_stat(bot, |j| stocks[j].win_ch[w][ch], x);
                            if t.cnt == 0 {
                                [f32::NAN; 10]
                            } else {
                                [
                                    t.mean,
                                    x - t.mean,
                                    (x - t.mean) / (t.std + 1e-4),
                                    x / (t.mean.abs() + EPS),
                                    x - t.wmean,
                                    t.beat as f32 / t.cnt as f32,
                                    t.std,
                                    x - t.max,
                                    x - t.min,
                                    t.mean - b.mean,
                                ]
                            }
                        }
                        _ => [f32::NAN; 10],
                    };
                    tail[p..p + 10].copy_from_slice(&vals);
                    p += 10;
                }
            }
            debug_assert_eq!(p, N_P3 + N_P4);

            // 模块 5：复杂度近邻网络的溢出
            for net in comp_nbrs.iter() {
                let nb = &net[i];
                for k in 0..N_SUMMARY {
                    let x = st.summary[k];
                    let vals: [f32; 8] = match nb {
                        Some((top, bot)) if x.is_finite() => {
                            let t = nbr_stat(top, |j| stocks[j].summary[k], x);
                            let b = nbr_stat(bot, |j| stocks[j].summary[k], x);
                            if t.cnt == 0 {
                                [f32::NAN; 8]
                            } else {
                                [
                                    t.mean,
                                    t.wmean,
                                    x - t.mean,
                                    (x - t.mean) / (t.std + 1e-4),
                                    x - t.wmean,
                                    t.std,
                                    b.mean,
                                    t.mean - b.mean,
                                ]
                            }
                        }
                        _ => [f32::NAN; 8],
                    };
                    tail[p..p + 8].copy_from_slice(&vals);
                    p += 8;
                }
                for dir in 0..3 {
                    for &core in XCH_CORE.iter() {
                        let ch = dir * N_CORE + core;
                        for w in 0..N_WIN {
                            let x = st.win_ch[w][ch];
                            let m = match nb {
                                Some((top, _)) => nbr_stat(top, |j| stocks[j].win_ch[w][ch], x).mean,
                                None => f32::NAN,
                            };
                            tail[p] = m;
                            tail[p + 1] = x - m;
                            p += 2;
                        }
                    }
                }
            }
            debug_assert_eq!(p, N_P3 + N_P4 + N_P5);

            // 模块 7：以全市场模板 / 同体量群模板为起点的退火
            for w in 0..N_OWIN {
                for d in 0..2 {
                    let k = w * 2 + d;
                    let own = &st.ord_seq64[w][d];
                    let mt = tmpl_excl(&t_sum[k], t_cnt[k], own);
                    let gt = match size_grp[i] {
                        Some(gd) => {
                            let gk = gd as usize * N_OWIN * 2 + k;
                            tmpl_excl(&g_sum[gk], g_cnt[gk], own)
                        }
                        None => [f32::NAN; 64],
                    };
                    let vm = anneal_from_template64(own, &mt, buf);
                    let vg = anneal_from_template64(own, &gt, buf);
                    tail[p..p + N_CORE].copy_from_slice(&vm);
                    tail[p + N_CORE..p + 2 * N_CORE].copy_from_slice(&vg);
                    for c in 0..N_CORE {
                        tail[p + 2 * N_CORE + c] = vm[c] - vg[c];
                    }
                    p += 3 * N_CORE;
                }
            }
            debug_assert_eq!(p, N_P3 + N_P4 + N_P5 + N_P7);
        },
    );

    (valid_codes, out)
}

// ============================================================================
// 因子名（与上面的输出顺序逐一对齐）
// ============================================================================

fn ch48_names() -> Vec<String> {
    let mut v = Vec::with_capacity(N_CH);
    for d in DIR3_NAMES {
        for m in CORE_NAMES {
            v.push(format!("{d}_{m}"));
        }
    }
    v
}

pub fn anneal_cross_peer_v2_names() -> Vec<String> {
    let mut names = Vec::with_capacity(N_FACTORS);
    let ch48 = ch48_names();
    let p1 = [
        "corr_mkt", "beta_mkt", "alpha_mkt", "r2_mkt", "res_early", "res_mid", "res_late",
        "res_late_minus_early", "res_late_minus_mid", "lead1_mkt", "lag1_mkt", "lead_minus_lag1",
        "lead3_mkt", "lag3_mkt", "lead_minus_lag3", "lead_minus_lag5", "corr_leaders",
        "corr_laggards", "corr_lead_minus_lag_grp", "rk_top5_share", "rk_top10_share",
        "rk_bot10_share", "rk_top10_max_streak", "rk_late_minus_early",
    ];
    let regimes = [
        "mkt_surge", "mkt_crash", "mkt_high_disp", "mkt_low_disp", "mkt_high_amt", "mkt_low_amt",
        "mkt_easy_ord", "mkt_hard_ord",
    ];
    for s in 0..N_SCALE {
        for seg in 0..N_SEG {
            let tag = format!("acp2_{}_{}", SCALE_TAGS[s], SEG_TAGS[seg]);
            for c in &ch48 {
                for st in p1 {
                    names.push(format!("{tag}_mkt_{c}_{st}"));
                }
            }
            let mut cols = Vec::with_capacity(N_RED_COLS);
            for c in &ch48 {
                cols.push(format!("{tag}_z_{c}"));
            }
            for c in &ch48 {
                cols.push(format!("{tag}_rk_{c}"));
            }
            let dummy = Array2::<f32>::ones((2, N_RED_COLS));
            let (_, red) = features::get_features_factors_rust_full(&dummy.view(), &cols, false);
            names.extend(red);
            for r in regimes {
                for c in &ch48 {
                    names.push(format!("{tag}_mreg_{r}_{c}_mean_z"));
                    names.push(format!("{tag}_mreg_{r}_{c}_mean_rank"));
                }
            }
        }
    }
    for g in ["size_peer", "vol_peer", "abr_peer"] {
        for w in WIN_TAGS {
            for c in &ch48 {
                for op in ["diff", "zscore", "ratio", "grp_rk", "grp_minus_gl_rk"] {
                    names.push(format!("acp2_{g}_{w}_{c}_{op}"));
                }
            }
        }
    }
    let p4 = [
        "twin_mean", "self_minus_twin", "twin_zscore", "self_over_twin", "self_minus_wtwin",
        "twin_beat_ratio", "twin_std", "self_minus_twin_max", "self_minus_twin_min",
        "top_minus_bot_twin",
    ];
    for w in WIN_TAGS {
        for c in &ch48 {
            for op in p4 {
                names.push(format!("acp2_prctwin_{w}_{c}_{op}"));
            }
        }
    }
    let p5a = [
        "peer_mean", "peer_wmean", "self_minus_peer", "peer_zscore", "self_minus_wpeer",
        "peer_std", "antipeer_mean", "peer_minus_antipeer",
    ];
    for net in COMP_NET_NAMES {
        for sname in SUMMARY_NAMES {
            for op in p5a {
                names.push(format!("acp2_{net}_spill_{sname}_{op}"));
            }
        }
        for dir in 0..3 {
            for &core in XCH_CORE.iter() {
                let c = &ch48[dir * N_CORE + core];
                for w in WIN_TAGS {
                    names.push(format!("acp2_{net}_xch_{c}_{w}_peer"));
                    names.push(format!("acp2_{net}_xch_{c}_{w}_diff"));
                }
            }
        }
    }
    for w in OWIN_TAGS {
        for d in ["bid", "ask"] {
            for t in ["mkt_tmpl", "grp_tmpl", "mkt_minus_grp_tmpl"] {
                for m in CORE_NAMES {
                    names.push(format!("acp2_joint_{w}_{d}_{t}_{m}"));
                }
            }
        }
    }
    assert_eq!(names.len(), N_FACTORS);
    names
}

// ============================================================================
// I/O 入口与 PyO3 绑定
// ============================================================================

/// v1 入口：读当天全市场股票（symbol_map.csv 白名单）并计算
pub fn compute_anneal_cross_peer_v2_full(date: i64) -> std::io::Result<(Vec<String>, Vec<f32>)> {
    let codes = list_codes(date, "transaction")?;
    let mids: Vec<Option<StockMidV2>> = codes
        .par_iter()
        .map(|code| {
            let trades = read_trade_fast_inner(code, date, false, true, usize::MAX).ok()?;
            extract_stock_mid_v2(&trades)
        })
        .collect();
    Ok(compute_cross_peer_v2_from_mids(&codes, mids))
}

/// v2 入口：传入已读好的逐笔（白名单校验，混入非股票立即报错）
pub fn compute_anneal_cross_peer_v2_from_trades(
    codes: &[String],
    trades_per_code: Vec<Option<Vec<TradeRecord>>>,
) -> std::io::Result<(Vec<String>, Vec<f32>)> {
    let symbol_map = crate::data_paths::basic_info_path("symbol_map.csv");
    let valid = load_valid_stock_set(&symbol_map)?;
    for code in codes {
        if !valid.contains(code) {
            return Err(Error::new(
                ErrorKind::InvalidData,
                format!("输入代码 {code} 不在 symbol_map.csv 股票白名单中，拒绝横截面计算"),
            ));
        }
    }
    let mids: Vec<Option<StockMidV2>> = trades_per_code
        .into_par_iter()
        .map(|t| t.and_then(|tr| extract_stock_mid_v2(&tr)))
        .collect();
    Ok(compute_cross_peer_v2_from_mids(codes, mids))
}

/// 柜员游戏扩展系列3 修正版：读当天全市场逐笔，返回 (股票代码, 展平因子值)，每股 55,680 维
#[pyfunction]
pub fn py_anneal_cross_peer_v2(date: i64) -> PyResult<(Vec<String>, Vec<f32>)> {
    compute_anneal_cross_peer_v2_full(date)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("{e:?}")))
}

#[pyfunction]
pub fn py_anneal_cross_peer_v2_names() -> Vec<String> {
    anneal_cross_peer_v2_names()
}

/// 传数据版：trades_list 每项为 read_trade_fast_us(code, date, 0, True) 返回的 (n, 8) 数组
/// [time_us, time_sec, price, volume, turnover, flag, bid_order, ask_order]，读不到的股票传 None。
#[pyfunction]
pub fn py_anneal_cross_peer_v2_from_data(
    codes: Vec<String>,
    trades_list: Vec<Option<PyReadonlyArray2<f64>>>,
) -> PyResult<(Vec<String>, Vec<f32>)> {
    if codes.len() != trades_list.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "codes 与 trades_list 长度必须一致",
        ));
    }
    let mut per_code = Vec::with_capacity(codes.len());
    for arr in trades_list {
        per_code.push(match arr {
            None => None,
            Some(a) => {
                let v = a.as_array();
                if v.ncols() != 8 {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "逐笔数组必须是 read_trade_fast_us 的 8 列格式",
                    ));
                }
                Some(
                    (0..v.nrows())
                        .map(|r| TradeRecord {
                            time_us: v[[r, 0]] as i64,
                            time_sec: v[[r, 1]] as f32,
                            price: v[[r, 2]] as f32,
                            volume: v[[r, 3]] as f32,
                            turnover: v[[r, 4]] as f32,
                            flag: v[[r, 5]] as i32,
                            bid_order: v[[r, 6]] as i64,
                            ask_order: v[[r, 7]] as i64,
                        })
                        .collect::<Vec<_>>(),
                )
            }
        });
    }
    compute_anneal_cross_peer_v2_from_trades(&codes, per_code)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("{e:?}")))
}

