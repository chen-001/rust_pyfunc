//! Tick 量化器因子（distill_tick）—— 480 维日频横截面因子。
//!
//! 核心思想：A 股 tick 网格（0.01 元）是强制量化器。
//! 投资者连续意图被量化到离散 tick，产生的量化误差本身就是蒸馏损失。
//! 用 Laplace 连续分布做 teacher，挂单离散分布做 student，
//! Wasserstein-1 距离度量量化损失。
//!
//! 3 个分支：
//! - X(78)：瞬时盘口多尺度量化器，σ=spread
//! - Y(150)：5min 滚动窗口多尺度量化器，σ=short
//! - Z(252)：双窗口 + 3 种 σ 估计
//!
//! 三层结构：compute_distill_tick_full（核心）+ pipeline_distill_tick（批量）+ py_distill_tick（调试）。
//! 详见 docs/superpowers/specs/2026-07-11-tick-quantizer-factor-design.md

use crate::fast_csv_reader::{
    read_market_fast_inner, read_trade_fast_inner, MarketRecord, TradeRecord,
};
use pyo3::prelude::*;

// ============================================================================
// 常量
// ============================================================================

/// X(78) + Y(150) + Z(252) = 480。
pub const OUT_LEN: usize = 480;

const ROLL_WINDOW_SEC: f32 = 300.0; // 5 分钟
const SESSION_START: f32 = 34200.0; // 9:30
const SESSION_END: f32 = 48420.0; // 13:27（平移后）
const EARLY_END: f32 = 36000.0; // 10:00
const SIGMA_MAX: f32 = 0.1; // σ 上限（return space，10%）
const MIN_TRADES_EARLY: usize = 30;
const MIN_TRADES_SHORT: usize = 5;
const MIN_WINDOW_SNAPSHOTS: usize = 10;
const MIN_HURST_LEN: usize = 20;
const EPS: f32 = 1e-12;

// 尺度定义
// S1: 10 档原始 → 10 桶
// S2: 每 2 档合并 → 5 桶
// S3: 每 5 档合并 → 2 桶

// 144 = 3 scales × 2 sides × 3 sigmas × 8 metrics
const STRIDE_SNAPSHOT: usize = 144;
const STRIDE_SCALE: usize = 48; // 2 sides × 3 sigmas × 8
const STRIDE_SIDE: usize = 24; // 3 sigmas × 8
const STRIDE_SIGMA: usize = 8;

// Metric indices within Metrics8
const M_W1: usize = 0;
const M_GINI: usize = 1;
const M_ENTROPY: usize = 2;
const M_HHI: usize = 3;
const M_CONC: usize = 4;
const M_PEAK: usize = 5;
const M_SKEW: usize = 6;
const M_KURT: usize = 7;

// Sigma indices
const SIG_SPREAD: usize = 0;
const SIG_SHORT: usize = 1;
const SIG_EARLY: usize = 2;

// Side indices
const SIDE_ASK: usize = 0;
const SIDE_BID: usize = 1;

// ============================================================================
// 统计辅助函数 —— NaN 感知
// ============================================================================

/// NaN 感知均值，全 NaN 返回 NaN。
fn nanmean(v: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    let mut n = 0usize;
    for &x in v {
        if x.is_finite() {
            sum += x;
            n += 1;
        }
    }
    if n == 0 {
        f32::NAN
    } else {
        sum / n as f32
    }
}

/// NaN 感知标准差（ddof=1），有效值 < 2 返回 NaN。
fn nanstd(v: &[f32]) -> f32 {
    let valid: Vec<f32> = v.iter().copied().filter(|x| x.is_finite()).collect();
    let n = valid.len();
    if n < 2 {
        return f32::NAN;
    }
    let m = valid.iter().sum::<f32>() / n as f32;
    let var = valid.iter().map(|x| (x - m).powi(2)).sum::<f32>() / (n - 1) as f32;
    var.max(0.0).sqrt()
}

/// NaN 感知偏度（G1 无偏校正），有效值 < 3 返回 NaN。
fn nanskew(v: &[f32]) -> f32 {
    let valid: Vec<f32> = v.iter().copied().filter(|x| x.is_finite()).collect();
    let n = valid.len();
    if n < 3 {
        return f32::NAN;
    }
    let m = valid.iter().sum::<f32>() / n as f32;
    let mut m2 = 0.0f32;
    let mut m3 = 0.0f32;
    for &x in &valid {
        let d = x - m;
        m2 += d * d;
        m3 += d * d * d;
    }
    m2 /= n as f32;
    m3 /= n as f32;
    if m2 <= 0.0 {
        return f32::NAN;
    }
    let g1 = m3 / m2.powf(1.5);
    let nf = n as f32;
    g1 * ((nf - 1.0).powf(1.5)) / ((nf - 2.0) * nf.sqrt())
}

/// NaN 感知峰度（超额峰度 G2 无偏校正），有效值 < 4 返回 NaN。
fn nankurt(v: &[f32]) -> f32 {
    let valid: Vec<f32> = v.iter().copied().filter(|x| x.is_finite()).collect();
    let n = valid.len();
    if n < 4 {
        return f32::NAN;
    }
    let m = valid.iter().sum::<f32>() / n as f32;
    let mut m2 = 0.0f32;
    let mut m4 = 0.0f32;
    for &x in &valid {
        let d = x - m;
        let d2 = d * d;
        m2 += d2;
        m4 += d2 * d2;
    }
    m2 /= n as f32;
    m4 /= n as f32;
    if m2 <= 0.0 {
        return f32::NAN;
    }
    let raw_k = m4 / (m2 * m2) - 3.0;
    let nf = n as f32;
    raw_k * ((nf + 1.0) * (nf - 1.0)) / ((nf - 2.0) * (nf - 3.0))
}

/// NaN 感知 Pearson 相关系数。
fn nancorr(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let pairs: Vec<(f32, f32)> = (0..n)
        .filter(|&i| a[i].is_finite() && b[i].is_finite())
        .map(|i| (a[i], b[i]))
        .collect();
    let m = pairs.len();
    if m < 2 {
        return f32::NAN;
    }
    let ma = pairs.iter().map(|p| p.0).sum::<f32>() / m as f32;
    let mb = pairs.iter().map(|p| p.1).sum::<f32>() / m as f32;
    let mut cov = 0.0f32;
    let mut va = 0.0f32;
    let mut vb = 0.0f32;
    for &(x, y) in &pairs {
        let dx = x - ma;
        let dy = y - mb;
        cov += dx * dy;
        va += dx * dx;
        vb += dy * dy;
    }
    let denom = (va * vb).sqrt();
    if denom > 1e-30 {
        (cov / denom).clamp(-1.0, 1.0)
    } else {
        f32::NAN
    }
}

/// NaN 感知 lag-1 自相关。
fn nan_autocorr1(v: &[f32]) -> f32 {
    let valid: Vec<f32> = v.iter().copied().filter(|x| x.is_finite()).collect();
    let n = valid.len();
    if n < 3 {
        return f32::NAN;
    }
    nancorr(&valid[..n - 1], &valid[1..n])
}

/// NaN 感知线性趋势（与 [1..n] 的 Pearson 相关）。
fn nan_trend(v: &[f32]) -> f32 {
    let pairs: Vec<(f32, f32)> = v
        .iter()
        .enumerate()
        .filter(|(_, &x)| x.is_finite())
        .map(|(i, &x)| ((i + 1) as f32, x))
        .collect();
    let n = pairs.len();
    if n < 2 {
        return f32::NAN;
    }
    let xs: Vec<f32> = pairs.iter().map(|p| p.0).collect();
    let ys: Vec<f32> = pairs.iter().map(|p| p.1).collect();
    nancorr(&xs, &ys)
}

/// Hurst 指数（R/S 法）。序列长度 < 20 返回 0.5（随机游走默认值）。
fn hurst_rs(v: &[f32]) -> f32 {
    let valid: Vec<f32> = v.iter().copied().filter(|x| x.is_finite()).collect();
    let n = valid.len();
    if n < MIN_HURST_LEN {
        return 0.5;
    }
    // 对多个子序列长度计算 R/S，做 log-log 回归
    let mut log_rs = Vec::new();
    let mut log_n = Vec::new();
    let min_k = 10usize;
    let mut k = min_k;
    while k <= n / 2 {
        let n_sub = n / k; // 子序列个数
        if n_sub < 1 {
            break;
        }
        let mut rs_sum = 0.0f32;
        for s in 0..n_sub {
            let start = s * k;
            let end = start + k;
            let slice = &valid[start..end];
            let mean = slice.iter().sum::<f32>() / k as f32;
            let mut cumdev = 0.0f32;
            let mut max_dev = f32::MIN;
            let mut min_dev = f32::MAX;
            let mut sum_sq = 0.0f32;
            for &x in slice {
                cumdev += x - mean;
                if cumdev > max_dev {
                    max_dev = cumdev;
                }
                if cumdev < min_dev {
                    min_dev = cumdev;
                }
                sum_sq += (x - mean).powi(2);
            }
            let r = max_dev - min_dev;
            let s_val = (sum_sq / k as f32).sqrt();
            if s_val > 1e-30 && r.is_finite() {
                rs_sum += r / s_val;
            }
        }
        let rs_avg = rs_sum / n_sub as f32;
        if rs_avg > 0.0 && rs_avg.is_finite() {
            log_rs.push(rs_avg.ln());
            log_n.push((k as f32).ln());
        }
        k = (k as f32 * 1.5) as usize;
    }
    if log_rs.len() < 3 {
        return 0.5;
    }
    // OLS slope = Hurst
    let nn = log_n.len();
    let mn = log_n.iter().sum::<f32>() / nn as f32;
    let mr = log_rs.iter().sum::<f32>() / nn as f32;
    let mut num = 0.0f32;
    let mut den = 0.0f32;
    for i in 0..nn {
        num += (log_n[i] - mn) * (log_rs[i] - mr);
        den += (log_n[i] - mn).powi(2);
    }
    if den > 1e-30 {
        let h = num / den;
        h.clamp(0.0, 1.0)
    } else {
        0.5
    }
}

// ============================================================================
// 分布度量（操作在 vol 分布上）
// ============================================================================

/// 基尼系数：G = Σ|v_i - v_j| / (2n²·mean)，范围 [0, (n-1)/n]。
fn dist_gini(vols: &[f32]) -> f32 {
    let mut sorted: Vec<f32> = vols
        .iter()
        .copied()
        .filter(|v| v.is_finite() && *v >= 0.0)
        .collect();
    let n = sorted.len();
    if n == 0 {
        return f32::NAN;
    }
    sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    let total: f32 = sorted.iter().sum();
    if total <= 0.0 {
        return f32::NAN;
    }
    // Σ_{i<j}(v_j - v_i) = Σ_k v_k · (2k - n + 1)
    let mut numerator = 0.0f32;
    for (k, &v) in sorted.iter().enumerate() {
        numerator += v * (2.0 * k as f32 - n as f32 + 1.0);
    }
    let mean = total / n as f32;
    numerator / (n as f32 * n as f32 * mean)
}

/// Shannon 熵（归一化到 [0, 1]）。
fn dist_entropy(vols: &[f32]) -> f32 {
    let total: f32 = vols
        .iter()
        .copied()
        .filter(|v| v.is_finite() && *v > 0.0)
        .sum();
    if total <= 0.0 {
        return f32::NAN;
    }
    let n = vols.len();
    let max_entropy = (n as f32).ln();
    if max_entropy <= 0.0 {
        return f32::NAN;
    }
    let mut entropy = 0.0f32;
    for &v in vols {
        if v > 0.0 {
            let p = v / total;
            entropy -= p * (p + EPS).ln();
        }
    }
    entropy / max_entropy
}

/// HHI（赫芬达尔指数）：Σ p²，范围 [1/n, 1]。
fn dist_hhi(vols: &[f32]) -> f32 {
    let total: f32 = vols
        .iter()
        .copied()
        .filter(|v| v.is_finite() && *v > 0.0)
        .sum();
    if total <= 0.0 {
        return f32::NAN;
    }
    vols.iter()
        .filter(|v| **v > 0.0)
        .map(|v| {
            let p = *v / total;
            p * p
        })
        .sum()
}

/// 前 2 档集中度：top-2 vol 占总 vol 比例。
fn dist_concentration_top2(vols: &[f32]) -> f32 {
    let total: f32 = vols
        .iter()
        .copied()
        .filter(|v| v.is_finite() && *v > 0.0)
        .sum();
    if total <= 0.0 {
        return f32::NAN;
    }
    let mut sorted = vols.to_vec();
    sorted.sort_unstable_by(|a, b| b.partial_cmp(a).unwrap());
    let top2: f32 = sorted.iter().take(2).sum();
    top2 / total
}

/// 峰值档位（1-based），最大 vol 所在桶位置。
fn dist_peak_pos(vols: &[f32]) -> f32 {
    let total: f32 = vols
        .iter()
        .copied()
        .filter(|v| v.is_finite() && *v > 0.0)
        .sum();
    if total <= 0.0 {
        return f32::NAN;
    }
    let mut max_vol = f32::MIN;
    let mut peak = 0usize;
    for (i, &v) in vols.iter().enumerate() {
        if v > max_vol {
            max_vol = v;
            peak = i;
        }
    }
    (peak + 1) as f32
}

/// 分布偏度（volume 值的样本偏度）。
fn dist_skew(vols: &[f32]) -> f32 {
    nanskew(vols)
}

/// 分布峰度（volume 值的样本峰度）。
fn dist_kurt(vols: &[f32]) -> f32 {
    nankurt(vols)
}

/// Wasserstein-1 距离（1D 闭式解）。
/// 用 one-sided exponential CDF 作 teacher：F(d) = 1 - exp(-d/sigma_price)。
/// W1 = Σ_i |F_cont(d_i) - F_disc(d_i)| · dx
fn wasserstein_1d(vols: &[f32], distances: &[f32], sigma_price: f32, dx: f32) -> f32 {
    if sigma_price <= 1e-8 || !sigma_price.is_finite() {
        return f32::NAN;
    }
    let total: f32 = vols
        .iter()
        .copied()
        .filter(|v| v.is_finite() && *v > 0.0)
        .sum();
    if total <= 0.0 {
        return f32::NAN;
    }
    let n = vols.len();
    let inv_total = 1.0 / total;
    let inv_sigma = 1.0 / sigma_price;
    let mut cum_disc = 0.0f32;
    let mut w1 = 0.0f32;
    for i in 0..n {
        cum_disc += vols[i].max(0.0) * inv_total;
        let f_cont = 1.0 - (-distances[i] * inv_sigma).exp();
        w1 += (f_cont - cum_disc).abs() * dx;
    }
    w1
}

/// KL 散度：Σ p·log(p/q)，加 ε 平滑。
fn kl_divergence(p: &[f32], q: &[f32]) -> f32 {
    let total_p: f32 = p.iter().copied().filter(|v| *v > 0.0).sum();
    let total_q: f32 = q.iter().copied().filter(|v| *v > 0.0).sum();
    if total_p <= 0.0 || total_q <= 0.0 {
        return f32::NAN;
    }
    let n = p.len().min(q.len());
    let mut kl = 0.0f32;
    for i in 0..n {
        let pi = p[i].max(0.0) / total_p + EPS;
        let qi = q[i].max(0.0) / total_q + EPS;
        kl += pi * (pi / qi).ln();
    }
    kl
}

/// JS 散度：0.5·KL(p||m) + 0.5·KL(q||m)，m = 0.5(p+q)。
fn js_divergence(p: &[f32], q: &[f32]) -> f32 {
    let total_p: f32 = p.iter().copied().filter(|v| *v > 0.0).sum();
    let total_q: f32 = q.iter().copied().filter(|v| *v > 0.0).sum();
    if total_p <= 0.0 || total_q <= 0.0 {
        return f32::NAN;
    }
    let n = p.len().min(q.len());
    let mut js = 0.0f32;
    for i in 0..n {
        let pi = p[i].max(0.0) / total_p + EPS;
        let qi = q[i].max(0.0) / total_q + EPS;
        let mi = 0.5 * (pi + qi);
        js += 0.5 * pi * (pi / mi).ln() + 0.5 * qi * (qi / mi).ln();
    }
    js
}

// ============================================================================
// 数据结构 + 提取
// ============================================================================

#[derive(Clone)]
struct OrderBookSnapshot {
    mid: f32,
    spread: f32, // (ask1 - bid1) / mid
    ask_vols: [f32; 10],
    bid_vols: [f32; 10],
    time_sec: f32,
}

/// 从 MarketRecord 提取 OrderBookSnapshot。
fn extract_orderbook(market: &[MarketRecord]) -> Vec<OrderBookSnapshot> {
    market
        .iter()
        .map(|m| {
            let ask1 = m.ask_prcs[0];
            let bid1 = m.bid_prcs[0];
            let mid = if ask1 > 0.0 && bid1 > 0.0 {
                (ask1 + bid1) * 0.5
            } else {
                f32::NAN
            };
            let spread = if mid > 0.0 && ask1 > 0.0 && bid1 > 0.0 {
                (ask1 - bid1) / mid
            } else {
                f32::NAN
            };
            OrderBookSnapshot {
                mid,
                spread,
                ask_vols: m.ask_vols,
                bid_vols: m.bid_vols,
                time_sec: m.time_sec.rem_euclid(86400.0),
            }
        })
        .collect()
}

/// 计算中位数 tick（从 ask 价差）。
fn compute_tick(market: &[MarketRecord]) -> f32 {
    let mut ticks: Vec<f32> = market
        .iter()
        .filter(|m| m.ask_prcs[0] > 0.0 && m.ask_prcs[1] > 0.0)
        .map(|m| m.ask_prcs[1] - m.ask_prcs[0])
        .filter(|t| *t > 0.0 && t.is_finite())
        .collect();
    if ticks.is_empty() {
        return 0.01;
    }
    ticks.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    ticks[ticks.len() / 2]
}

// ============================================================================
// 尺度重采样
// ============================================================================

/// S2 合并：每 2 档 → 5 桶。
fn resample_s2(vols: &[f32; 10]) -> [f32; 5] {
    [
        vols[0] + vols[1],
        vols[2] + vols[3],
        vols[4] + vols[5],
        vols[6] + vols[7],
        vols[8] + vols[9],
    ]
}

/// S3 合并：每 5 档 → 2 桶。
fn resample_s3(vols: &[f32; 10]) -> [f32; 2] {
    [
        vols[0] + vols[1] + vols[2] + vols[3] + vols[4],
        vols[5] + vols[6] + vols[7] + vols[8] + vols[9],
    ]
}

// ============================================================================
// Sigma 计算
// ============================================================================

/// σ_spread：每 snapshot 的 (ask1-bid1)/mid。
fn compute_sigma_spread(ob: &[OrderBookSnapshot]) -> Vec<f32> {
    ob.iter().map(|s| s.spread).collect()
}

/// σ_short：5min 滚动 trade logret std，对齐到 snapshot 时间戳。
fn compute_sigma_short(trade: &[TradeRecord], ob_times: &[f32]) -> Vec<f32> {
    let n = ob_times.len();
    let mut sigmas = vec![f32::NAN; n];
    if trade.len() < 2 {
        return sigmas;
    }

    // 预计算 trade log-returns
    let mut logrets: Vec<(f32, f32)> = Vec::with_capacity(trade.len());
    for i in 1..trade.len() {
        let prev = trade[i - 1].price;
        let curr = trade[i].price;
        if prev > 0.0 && curr > 0.0 {
            let lr = (curr / prev).ln();
            if lr.is_finite() {
                logrets.push((trade[i].time_sec.rem_euclid(86400.0), lr));
            }
        }
    }
    if logrets.len() < MIN_TRADES_SHORT {
        return sigmas;
    }

    // 双指针滑动窗口
    let mut lo = 0usize;
    for (idx, &t) in ob_times.iter().enumerate() {
        if !t.is_finite() {
            continue;
        }
        let win_start = t - ROLL_WINDOW_SEC;
        // 推进 lo 到窗口内
        while lo < logrets.len() && logrets[lo].0 < win_start {
            lo += 1;
        }
        // 找 hi：第一个 > t 的
        let mut hi = lo;
        while hi < logrets.len() && logrets[hi].0 <= t {
            hi += 1;
        }
        let count = hi - lo;
        if count < MIN_TRADES_SHORT {
            continue;
        }
        // 计算 std
        let sum: f32 = logrets[lo..hi].iter().map(|p| p.1).sum();
        let mean = sum / count as f32;
        let var: f32 = logrets[lo..hi]
            .iter()
            .map(|p| (p.1 - mean).powi(2))
            .sum::<f32>()
            / (count - 1) as f32;
        let sigma = var.max(0.0).sqrt();
        // 限制上限
        sigmas[idx] = sigma.min(SIGMA_MAX);
    }
    sigmas
}

/// σ_early：当天前 30min（9:30-10:00）trade logret std，标量。
fn compute_sigma_early(trade: &[TradeRecord]) -> f32 {
    let mut logrets = Vec::new();
    for i in 1..trade.len() {
        let t = trade[i].time_sec.rem_euclid(86400.0);
        if t < SESSION_START || t >= EARLY_END {
            continue;
        }
        let prev = trade[i - 1].price;
        let curr = trade[i].price;
        if prev > 0.0 && curr > 0.0 {
            let lr = (curr / prev).ln();
            if lr.is_finite() {
                logrets.push(lr);
            }
        }
    }
    if logrets.len() < MIN_TRADES_EARLY {
        return f32::NAN;
    }
    let n = logrets.len();
    let mean = logrets.iter().sum::<f32>() / n as f32;
    let var = logrets.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / (n - 1) as f32;
    var.max(0.0).sqrt().min(SIGMA_MAX)
}

// ============================================================================
// 预计算：per-snapshot 全量 144 指标
// ============================================================================

/// 计算 8 个分布度量。
fn compute_metrics_8(vols: &[f32], distances: &[f32], sigma_price: f32, dx: f32) -> [f32; 8] {
    [
        wasserstein_1d(vols, distances, sigma_price, dx),
        dist_gini(vols),
        dist_entropy(vols),
        dist_hhi(vols),
        dist_concentration_top2(vols),
        dist_peak_pos(vols),
        dist_skew(vols),
        dist_kurt(vols),
    ]
}

/// 对全部 snapshot 预计算 144 指标，展平为 Vec<f32>。
fn precompute_snapshot_metrics(
    ob: &[OrderBookSnapshot],
    sigma_spread: &[f32],
    sigma_short: &[f32],
    sigma_early: f32,
    tick: f32,
) -> Vec<f32> {
    let n = ob.len();
    let mut flat = vec![f32::NAN; n * STRIDE_SNAPSHOT];

    // 预计算 distances 和 dx（不随 snapshot 变化）
    let dist_s1: Vec<f32> = (1..=10).map(|k| k as f32 * tick).collect();
    let dist_s2: Vec<f32> = (1..=5).map(|k| (2 * k) as f32 * tick).collect();
    let dist_s3: Vec<f32> = [5.0 * tick, 10.0 * tick].to_vec();
    let dx_s1 = tick;
    let dx_s2 = 2.0 * tick;
    let dx_s3 = 5.0 * tick;

    for t in 0..n {
        let mid = ob[t].mid;
        if !mid.is_finite() || mid <= 0.0 {
            continue;
        }
        let sigmas = [sigma_spread[t], sigma_short[t], sigma_early];
        let ask_s2 = resample_s2(&ob[t].ask_vols);
        let ask_s3 = resample_s3(&ob[t].ask_vols);
        let bid_s2 = resample_s2(&ob[t].bid_vols);
        let bid_s3 = resample_s3(&ob[t].bid_vols);

        for (scale, (ask_v, bid_v, dist, dx)) in [
            (
                &ob[t].ask_vols[..],
                &ob[t].bid_vols[..],
                &dist_s1[..],
                dx_s1,
            ),
            (&ask_s2[..], &bid_s2[..], &dist_s2[..], dx_s2),
            (&ask_s3[..], &bid_s3[..], &dist_s3[..], dx_s3),
        ]
        .iter()
        .enumerate()
        {
            for side in 0..2usize {
                let vols = if side == 0 { *ask_v } else { *bid_v };
                for (sig_idx, &sigma) in sigmas.iter().enumerate() {
                    let sigma_price = if sigma.is_finite() && sigma > 0.0 {
                        sigma.min(SIGMA_MAX) * mid
                    } else {
                        f32::NAN
                    };
                    let m8 = compute_metrics_8(vols, dist, sigma_price, *dx);
                    let base = t * STRIDE_SNAPSHOT
                        + scale * STRIDE_SCALE
                        + side * STRIDE_SIDE
                        + sig_idx * STRIDE_SIGMA;
                    flat[base..base + 8].copy_from_slice(&m8);
                }
            }
        }
    }
    flat
}

// ============================================================================
// 窗口管理
// ============================================================================

/// 将 snapshot 按非重叠 5min 窗口分组，返回 (start_idx, end_idx) 列表。
fn build_windows(times: &[f32]) -> Vec<(usize, usize)> {
    let n_windows = ((SESSION_END - SESSION_START) / ROLL_WINDOW_SEC).floor() as usize;
    let mut starts = vec![usize::MAX; n_windows];
    let mut ends = vec![0usize; n_windows];

    for (i, &t) in times.iter().enumerate() {
        if !t.is_finite() || t < SESSION_START || t >= SESSION_END {
            continue;
        }
        let w = ((t - SESSION_START) / ROLL_WINDOW_SEC) as usize;
        if w >= n_windows {
            continue;
        }
        if starts[w] == usize::MAX {
            starts[w] = i;
        }
        ends[w] = i + 1;
    }

    (0..n_windows)
        .filter(|&w| starts[w] != usize::MAX && ends[w] > starts[w])
        .map(|w| (starts[w], ends[w]))
        .collect()
}

/// 预计算 per-window 累积分布的 144 指标。
fn precompute_window_metrics(
    ob: &[OrderBookSnapshot],
    sigma_spread: &[f32],
    sigma_short: &[f32],
    sigma_early: f32,
    tick: f32,
    windows: &[(usize, usize)],
) -> Vec<f32> {
    let n_win = windows.len();
    let mut flat = vec![f32::NAN; n_win * STRIDE_SNAPSHOT];

    let dist_s1: Vec<f32> = (1..=10).map(|k| k as f32 * tick).collect();
    let dist_s2: Vec<f32> = (1..=5).map(|k| (2 * k) as f32 * tick).collect();
    let dist_s3: Vec<f32> = [5.0 * tick, 10.0 * tick].to_vec();
    let dx_s1 = tick;
    let dx_s2 = 2.0 * tick;
    let dx_s3 = 5.0 * tick;

    for (wi, &(lo, hi)) in windows.iter().enumerate() {
        if hi - lo < MIN_WINDOW_SNAPSHOTS {
            continue;
        }
        // 累积 vol 分布
        let mut cum_ask = [0.0f32; 10];
        let mut cum_bid = [0.0f32; 10];
        let mut sum_mid = 0.0f32;
        let mut cnt_mid = 0usize;
        let mut sum_spread = 0.0f32;
        let mut cnt_spread = 0usize;
        let mut sum_short = 0.0f32;
        let mut cnt_short = 0usize;

        for idx in lo..hi {
            for k in 0..10 {
                cum_ask[k] += ob[idx].ask_vols[k];
                cum_bid[k] += ob[idx].bid_vols[k];
            }
            if ob[idx].mid.is_finite() && ob[idx].mid > 0.0 {
                sum_mid += ob[idx].mid;
                cnt_mid += 1;
            }
            if sigma_spread[idx].is_finite() {
                sum_spread += sigma_spread[idx];
                cnt_spread += 1;
            }
            if sigma_short[idx].is_finite() {
                sum_short += sigma_short[idx];
                cnt_short += 1;
            }
        }

        if cnt_mid == 0 {
            continue;
        }
        let win_mid = sum_mid / cnt_mid as f32;
        let win_spread = if cnt_spread > 0 {
            sum_spread / cnt_spread as f32
        } else {
            f32::NAN
        };
        let win_short = if cnt_short > 0 {
            sum_short / cnt_short as f32
        } else {
            f32::NAN
        };
        let sigmas = [win_spread, win_short, sigma_early];

        let ask_s2 = resample_s2(&cum_ask);
        let ask_s3 = resample_s3(&cum_ask);
        let bid_s2 = resample_s2(&cum_bid);
        let bid_s3 = resample_s3(&cum_bid);

        for (scale, (ask_v, bid_v, dist, dx)) in [
            (&cum_ask[..], &cum_bid[..], &dist_s1[..], dx_s1),
            (&ask_s2[..], &bid_s2[..], &dist_s2[..], dx_s2),
            (&ask_s3[..], &bid_s3[..], &dist_s3[..], dx_s3),
        ]
        .iter()
        .enumerate()
        {
            for side in 0..2usize {
                let vols = if side == 0 { *ask_v } else { *bid_v };
                for (sig_idx, &sigma) in sigmas.iter().enumerate() {
                    let sigma_price = if sigma.is_finite() && sigma > 0.0 {
                        sigma.min(SIGMA_MAX) * win_mid
                    } else {
                        f32::NAN
                    };
                    let m8 = compute_metrics_8(vols, dist, sigma_price, *dx);
                    let base = wi * STRIDE_SNAPSHOT
                        + scale * STRIDE_SCALE
                        + side * STRIDE_SIDE
                        + sig_idx * STRIDE_SIGMA;
                    flat[base..base + 8].copy_from_slice(&m8);
                }
            }
        }
    }
    flat
}

/// 从 flat 数组取某个 snapshot/window 的某个指标序列。
fn extract_metric_series(
    flat: &[f32],
    n_rows: usize,
    scale: usize,
    side: usize,
    sigma: usize,
    metric: usize,
) -> Vec<f32> {
    let base_offset = scale * STRIDE_SCALE + side * STRIDE_SIDE + sigma * STRIDE_SIGMA + metric;
    (0..n_rows)
        .map(|t| flat[t * STRIDE_SNAPSHOT + base_offset])
        .collect()
}

// ============================================================================
// Branch X：瞬时盘口多尺度量化器（78 因子，σ=spread）
// ============================================================================

mod branch_x {
    use super::*;

    pub const COUNT: usize = 78;

    pub fn compute(flat: &[f32], n: usize) -> Vec<f32> {
        let mut out = Vec::with_capacity(COUNT);

        let sides = [SIDE_ASK, SIDE_BID];
        let scales = [0usize, 1, 2]; // S1, S2, S3
        let dist_names = ["ask_S1", "bid_S1", "ask_S2", "bid_S2", "ask_S3", "bid_S3"];

        // (a) 单分布量化损失：6 分布 × 8 度量 = 48
        // 顺序：ask_S1, bid_S1, ask_S2, bid_S2, ask_S3, bid_S3
        // 每个分布 8 度量：w1, gini, entropy, hhi, concentration, peak_pos, skew, kurt
        let mut di = 0;
        for &scale in &scales {
            for &side in &sides {
                for m in 0..8 {
                    let series = extract_metric_series(flat, n, scale, side, SIG_SPREAD, m);
                    out.push(nanmean(&series));
                }
                di += 1;
            }
        }
        debug_assert_eq!(di, 6);
        debug_assert_eq!(out.len(), 48);

        // (b) 跨尺度曲线特征：2 sides × 12 features = 24
        for &side in &sides {
            // 日均 W1 / Gini / Entropy at S1, S2, S3
            let w1: Vec<f32> = (0..3)
                .map(|s| nanmean(&extract_metric_series(flat, n, s, side, SIG_SPREAD, M_W1)))
                .collect();
            let gini: Vec<f32> = (0..3)
                .map(|s| nanmean(&extract_metric_series(flat, n, s, side, SIG_SPREAD, M_GINI)))
                .collect();
            let entropy: Vec<f32> = (0..3)
                .map(|s| {
                    nanmean(&extract_metric_series(
                        flat, n, s, side, SIG_SPREAD, M_ENTROPY,
                    ))
                })
                .collect();

            // 12 features
            // W1 slope (S1→S3 线性回归)
            out.push(nancorr(&[1.0, 2.0, 3.0], &w1));
            // W1 ratio S1/S2
            out.push(if w1[1].abs() > 1e-30 {
                w1[0] / w1[1]
            } else {
                f32::NAN
            });
            // W1 ratio S2/S3
            out.push(if w1[2].abs() > 1e-30 {
                w1[1] / w1[2]
            } else {
                f32::NAN
            });
            // W1 area (sum)
            out.push(w1.iter().sum::<f32>());
            // Gini slope
            out.push(nancorr(&[1.0, 2.0, 3.0], &gini));
            // Gini ratio S1/S2
            out.push(if gini[1].abs() > 1e-30 {
                gini[0] / gini[1]
            } else {
                f32::NAN
            });
            // Gini ratio S2/S3
            out.push(if gini[2].abs() > 1e-30 {
                gini[1] / gini[2]
            } else {
                f32::NAN
            });
            // Entropy slope
            out.push(nancorr(&[1.0, 2.0, 3.0], &entropy));
            // Entropy ratio S1/S2
            out.push(if entropy[1].abs() > 1e-30 {
                entropy[0] / entropy[1]
            } else {
                f32::NAN
            });
            // Entropy ratio S2/S3
            out.push(if entropy[2].abs() > 1e-30 {
                entropy[1] / entropy[2]
            } else {
                f32::NAN
            });
            // W1 curvature (2nd difference)
            out.push(w1[0] - 2.0 * w1[1] + w1[2]);
            // W1 monotonicity: sign change count of consecutive differences
            let d1 = w1[1] - w1[0];
            let d2 = w1[2] - w1[1];
            let sign_changes = if d1 * d2 < 0.0 { 1.0 } else { 0.0 };
            out.push(sign_changes);
        }
        debug_assert_eq!(out.len(), 48 + 24);

        // (c) 跨边不对称特征：6
        {
            let w1_ask_s1 = nanmean(&extract_metric_series(
                flat, n, 0, SIDE_ASK, SIG_SPREAD, M_W1,
            ));
            let w1_bid_s1 = nanmean(&extract_metric_series(
                flat, n, 0, SIDE_BID, SIG_SPREAD, M_W1,
            ));
            let w1_ask_s2 = nanmean(&extract_metric_series(
                flat, n, 1, SIDE_ASK, SIG_SPREAD, M_W1,
            ));
            let w1_bid_s2 = nanmean(&extract_metric_series(
                flat, n, 1, SIDE_BID, SIG_SPREAD, M_W1,
            ));
            let gini_ask_s1 = nanmean(&extract_metric_series(
                flat, n, 0, SIDE_ASK, SIG_SPREAD, M_GINI,
            ));
            let gini_bid_s1 = nanmean(&extract_metric_series(
                flat, n, 0, SIDE_BID, SIG_SPREAD, M_GINI,
            ));
            let gini_ask_s2 = nanmean(&extract_metric_series(
                flat, n, 1, SIDE_ASK, SIG_SPREAD, M_GINI,
            ));
            let gini_bid_s2 = nanmean(&extract_metric_series(
                flat, n, 1, SIDE_BID, SIG_SPREAD, M_GINI,
            ));
            let conc_ask_s1 = nanmean(&extract_metric_series(
                flat, n, 0, SIDE_ASK, SIG_SPREAD, M_CONC,
            ));
            let conc_bid_s1 = nanmean(&extract_metric_series(
                flat, n, 0, SIDE_BID, SIG_SPREAD, M_CONC,
            ));
            let peak_ask_s1 = nanmean(&extract_metric_series(
                flat, n, 0, SIDE_ASK, SIG_SPREAD, M_PEAK,
            ));
            let peak_bid_s1 = nanmean(&extract_metric_series(
                flat, n, 0, SIDE_BID, SIG_SPREAD, M_PEAK,
            ));

            // ask-bid W1 diff S1
            out.push(w1_ask_s1 - w1_bid_s1);
            // ask-bid W1 diff S2
            out.push(w1_ask_s2 - w1_bid_s2);
            // ask-bid Gini diff S1
            out.push(gini_ask_s1 - gini_bid_s1);
            // ask-bid Gini diff S2
            out.push(gini_ask_s2 - gini_bid_s2);
            // ask-bid concentration ratio
            out.push(if conc_bid_s1.abs() > 1e-30 {
                conc_ask_s1 / conc_bid_s1
            } else {
                f32::NAN
            });
            // ask-bid peak shift (S1)
            out.push(peak_ask_s1 - peak_bid_s1);
        }
        debug_assert_eq!(out.len(), COUNT);
        out
    }

    pub fn names() -> Vec<String> {
        let mut out = Vec::with_capacity(COUNT);
        let dist_names = ["ask_S1", "bid_S1", "ask_S2", "bid_S2", "ask_S3", "bid_S3"];
        let metrics = [
            "wasserstein",
            "gini",
            "entropy",
            "hhi",
            "concentration",
            "peak_pos",
            "skew",
            "kurt",
        ];

        // (a) 48
        for d in &dist_names {
            for m in &metrics {
                out.push(format!("qX_{}_{}", m, d));
            }
        }
        // (b) 24: 2 sides × 12
        let side_names = ["ask", "bid"];
        let b_features = [
            "wasserstein_slope",
            "wasserstein_ratio_S1S2",
            "wasserstein_ratio_S2S3",
            "curve_area",
            "gini_slope",
            "gini_ratio_S1S2",
            "gini_ratio_S2S3",
            "entropy_slope",
            "entropy_ratio_S1S2",
            "entropy_ratio_S2S3",
            "wasserstein_curvature",
            "wasserstein_monotonicity",
        ];
        for s in &side_names {
            for f in &b_features {
                out.push(format!("qX_{}_{}", f, s));
            }
        }
        // (c) 6
        out.push("qX_ask_bid_wasserstein_diff_S1".to_string());
        out.push("qX_ask_bid_wasserstein_diff_S2".to_string());
        out.push("qX_ask_bid_gini_diff_S1".to_string());
        out.push("qX_ask_bid_gini_diff_S2".to_string());
        out.push("qX_ask_bid_concentration_ratio".to_string());
        out.push("qX_ask_bid_peak_shift".to_string());

        debug_assert_eq!(out.len(), COUNT);
        out
    }
}

// ============================================================================
// Branch Y：5min 滚动窗口多尺度量化器（150 因子，σ=short）
// ============================================================================

mod branch_y {
    use super::*;

    pub const COUNT: usize = 150;

    pub fn compute(flat: &[f32], n: usize, windows: &[(usize, usize)]) -> Vec<f32> {
        let mut out = Vec::with_capacity(COUNT);
        let sides = [SIDE_ASK, SIDE_BID];
        let scales = [0usize, 1, 2];

        // (a) 单分布量化损失 + 时序统计：6 分布 × 15 = 90
        for &scale in &scales {
            for &side in &sides {
                // 8 base metrics: daily mean (σ=short)
                for m in 0..8 {
                    let series = extract_metric_series(flat, n, scale, side, SIG_SHORT, m);
                    out.push(nanmean(&series));
                }
                // 7 temporal stats: within each 5min window, compute stats of W1 and Gini series
                // Then average across windows
                let w1_series = extract_metric_series(flat, n, scale, side, SIG_SHORT, M_W1);
                let gini_series = extract_metric_series(flat, n, scale, side, SIG_SHORT, M_GINI);
                let entropy_series =
                    extract_metric_series(flat, n, scale, side, SIG_SHORT, M_ENTROPY);

                let mut win_w1_std = Vec::new();
                let mut win_w1_trend = Vec::new();
                let mut win_w1_autocorr = Vec::new();
                let mut win_w1_hurst = Vec::new();
                let mut win_gini_std = Vec::new();
                let mut win_gini_trend = Vec::new();
                let mut win_entropy_std = Vec::new();

                for &(lo, hi) in windows {
                    if hi - lo < MIN_WINDOW_SNAPSHOTS {
                        continue;
                    }
                    let w1_win = &w1_series[lo..hi];
                    let gini_win = &gini_series[lo..hi];
                    let entropy_win = &entropy_series[lo..hi];

                    win_w1_std.push(nanstd(w1_win));
                    win_w1_trend.push(nan_trend(w1_win));
                    win_w1_autocorr.push(nan_autocorr1(w1_win));
                    win_w1_hurst.push(hurst_rs(w1_win));
                    win_gini_std.push(nanstd(gini_win));
                    win_gini_trend.push(nan_trend(gini_win));
                    win_entropy_std.push(nanstd(entropy_win));
                }

                // 7 temporal stats
                out.push(nanmean(&win_w1_std));
                out.push(nanmean(&win_w1_trend));
                out.push(nanmean(&win_w1_autocorr));
                out.push(nanmean(&win_w1_hurst));
                out.push(nanmean(&win_gini_std));
                out.push(nanmean(&win_gini_trend));
                out.push(nanmean(&win_entropy_std));
            }
        }
        debug_assert_eq!(out.len(), 90);

        // (b) 跨尺度曲线特征 + 时序：2 sides × 3 scale-pairs × 10 = 60
        // Scale pairs: (S1,S2)=P12, (S2,S3)=P23, (S1,S3)=P13
        let pairs = [(0usize, 1usize), (1, 2), (0, 2)];
        for &side in &sides {
            for &(s_lo, s_hi) in &pairs {
                let w1_lo = extract_metric_series(flat, n, s_lo, side, SIG_SHORT, M_W1);
                let w1_hi = extract_metric_series(flat, n, s_hi, side, SIG_SHORT, M_W1);
                let gini_lo = extract_metric_series(flat, n, s_lo, side, SIG_SHORT, M_GINI);
                let gini_hi = extract_metric_series(flat, n, s_hi, side, SIG_SHORT, M_GINI);
                let entropy_lo = extract_metric_series(flat, n, s_lo, side, SIG_SHORT, M_ENTROPY);
                let entropy_hi = extract_metric_series(flat, n, s_hi, side, SIG_SHORT, M_ENTROPY);

                // ratio = lo / hi (finer / coarser)
                let w1_ratio: Vec<f32> = (0..n)
                    .map(|t| {
                        if w1_hi[t].abs() > 1e-30 && w1_lo[t].is_finite() && w1_hi[t].is_finite() {
                            w1_lo[t] / w1_hi[t]
                        } else {
                            f32::NAN
                        }
                    })
                    .collect();
                let w1_diff: Vec<f32> = (0..n).map(|t| w1_lo[t] - w1_hi[t]).collect();
                let gini_ratio: Vec<f32> = (0..n)
                    .map(|t| {
                        if gini_hi[t].abs() > 1e-30
                            && gini_lo[t].is_finite()
                            && gini_hi[t].is_finite()
                        {
                            gini_lo[t] / gini_hi[t]
                        } else {
                            f32::NAN
                        }
                    })
                    .collect();
                let gini_diff: Vec<f32> = (0..n).map(|t| gini_lo[t] - gini_hi[t]).collect();
                let entropy_diff: Vec<f32> =
                    (0..n).map(|t| entropy_lo[t] - entropy_hi[t]).collect();

                // 5 static
                out.push(nanmean(&w1_ratio));
                out.push(nanmean(&w1_diff));
                out.push(nanmean(&gini_ratio));
                out.push(nanmean(&gini_diff));
                out.push(nanmean(&entropy_diff));

                // 5 temporal
                out.push(nanstd(&w1_ratio));
                out.push(nanstd(&w1_diff));
                out.push(nan_trend(&w1_diff));
                out.push(nanstd(&gini_diff));
                out.push(nanstd(&entropy_diff));
            }
        }
        debug_assert_eq!(out.len(), COUNT);
        out
    }

    pub fn names() -> Vec<String> {
        let mut out = Vec::with_capacity(COUNT);
        let dist_names = ["ask_S1", "bid_S1", "ask_S2", "bid_S2", "ask_S3", "bid_S3"];
        let base_metrics = [
            "wasserstein",
            "gini",
            "entropy",
            "hhi",
            "concentration",
            "peak_pos",
            "skew",
            "kurt",
        ];
        let temporal_metrics = [
            "wasserstein_std",
            "wasserstein_trend",
            "wasserstein_autocorr",
            "wasserstein_hurst",
            "gini_std",
            "gini_trend",
            "entropy_std",
        ];

        // (a) 90
        for d in &dist_names {
            for m in &base_metrics {
                out.push(format!("qY_{}_{}", m, d));
            }
            for m in &temporal_metrics {
                out.push(format!("qY_{}_{}", m, d));
            }
        }
        // (b) 60: 2 sides × 3 pairs × 10
        let side_names = ["ask", "bid"];
        let pair_names = ["S1S2", "S2S3", "S1S3"];
        let b_static = [
            "w1_ratio",
            "w1_diff",
            "gini_ratio",
            "gini_diff",
            "entropy_diff",
        ];
        let b_temporal = [
            "w1_ratio_std",
            "w1_diff_std",
            "w1_diff_trend",
            "gini_diff_std",
            "entropy_diff_std",
        ];
        for s in &side_names {
            for p in &pair_names {
                for f in &b_static {
                    out.push(format!("qY_{}_{}_{}", f, p, s));
                }
                for f in &b_temporal {
                    out.push(format!("qY_{}_{}_{}", f, p, s));
                }
            }
        }
        debug_assert_eq!(out.len(), COUNT);
        out
    }
}

// ============================================================================
// Branch Z：双窗口 + 多 σ 估计（252 因子）
// ============================================================================

mod branch_z {
    use super::*;

    pub const COUNT: usize = 252;

    pub fn compute(
        flat_snap: &[f32],
        n_snap: usize,
        flat_win: &[f32],
        n_win: usize,
        ob: &[OrderBookSnapshot],
        sigma_spread: &[f32],
        sigma_short: &[f32],
        sigma_early: f32,
        tick: f32,
        windows: &[(usize, usize)],
    ) -> Vec<f32> {
        let mut out = Vec::with_capacity(COUNT);
        let sigmas = [SIG_SPREAD, SIG_SHORT, SIG_EARLY];
        let sigma_names = ["spread", "short", "early"];
        let scales = [0usize, 1, 2];
        let scale_names = ["S1", "S2", "S3"];

        // (a) 单尺度量化损失：3σ × 3 scales × 2 windows × 8 metrics = 144
        // ask + bid averaged
        for &sig in &sigmas {
            for &scale in &scales {
                // inst window
                for m in 0..8 {
                    let ask_series =
                        extract_metric_series(flat_snap, n_snap, scale, SIDE_ASK, sig, m);
                    let bid_series =
                        extract_metric_series(flat_snap, n_snap, scale, SIDE_BID, sig, m);
                    let avg: Vec<f32> = (0..n_snap)
                        .map(|t| {
                            if ask_series[t].is_finite() && bid_series[t].is_finite() {
                                (ask_series[t] + bid_series[t]) * 0.5
                            } else {
                                f32::NAN
                            }
                        })
                        .collect();
                    out.push(nanmean(&avg));
                }
                // roll5m window
                for m in 0..8 {
                    let ask_series =
                        extract_metric_series(flat_win, n_win, scale, SIDE_ASK, sig, m);
                    let bid_series =
                        extract_metric_series(flat_win, n_win, scale, SIDE_BID, sig, m);
                    let avg: Vec<f32> = (0..n_win)
                        .map(|t| {
                            if ask_series[t].is_finite() && bid_series[t].is_finite() {
                                (ask_series[t] + bid_series[t]) * 0.5
                            } else {
                                f32::NAN
                            }
                        })
                        .collect();
                    out.push(nanmean(&avg));
                }
            }
        }
        debug_assert_eq!(out.len(), 144);

        // (b) 多尺度曲线特征：3σ × 2 windows × 10 = 60
        for &sig in &sigmas {
            for flat_ref in [flat_snap, flat_win] {
                let n_rows = if std::ptr::eq(flat_ref, flat_snap) {
                    n_snap
                } else {
                    n_win
                };
                // W1 at S1, S2, S3 (ask+bid averaged)
                let w1_vals: Vec<f32> = (0..3)
                    .map(|scale| {
                        let ask =
                            extract_metric_series(flat_ref, n_rows, scale, SIDE_ASK, sig, M_W1);
                        let bid =
                            extract_metric_series(flat_ref, n_rows, scale, SIDE_BID, sig, M_W1);
                        let avg: Vec<f32> = (0..n_rows)
                            .map(|t| {
                                if ask[t].is_finite() && bid[t].is_finite() {
                                    (ask[t] + bid[t]) * 0.5
                                } else {
                                    f32::NAN
                                }
                            })
                            .collect();
                        nanmean(&avg)
                    })
                    .collect();

                // 10 multi-scale curve features
                // 1. W1 slope
                out.push(nancorr(&[1.0, 2.0, 3.0], &w1_vals));
                // 2. W1 curvature (2nd diff)
                out.push(w1_vals[0] - 2.0 * w1_vals[1] + w1_vals[2]);
                // 3. Inflection scale (index of max |diff|)
                let d1 = (w1_vals[1] - w1_vals[0]).abs();
                let d2 = (w1_vals[2] - w1_vals[1]).abs();
                out.push(if d1 > d2 { 1.0 } else { 2.0 });
                // 4. Monotonicity (sign changes)
                let s1 = w1_vals[1] - w1_vals[0];
                let s2 = w1_vals[2] - w1_vals[1];
                out.push(if s1 * s2 < 0.0 { 1.0 } else { 0.0 });
                // 5. Area (sum)
                out.push(w1_vals.iter().sum::<f32>());
                // 6. Fractal dimension: -log(w1_S1/w1_S3) / log(5)
                out.push(
                    if w1_vals[2].abs() > 1e-30 && w1_vals[0] > 0.0 && w1_vals[2] > 0.0 {
                        -(w1_vals[0] / w1_vals[2]).ln() / 5f32.ln()
                    } else {
                        f32::NAN
                    },
                );
                // 7. Extreme scale (index of max w1, 1-based)
                let max_idx = (0..3)
                    .map(|i| (w1_vals[i], i))
                    .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(_, i)| i)
                    .unwrap_or(0);
                out.push((max_idx + 1) as f32);
                // 8. Cross-scale variance
                out.push(nanstd(&w1_vals));
                // 9. Convexity (curvature sign)
                let curvature = w1_vals[0] - 2.0 * w1_vals[1] + w1_vals[2];
                out.push(if curvature > 0.0 {
                    1.0
                } else if curvature < 0.0 {
                    -1.0
                } else {
                    0.0
                });
                // 10. Inflection W1 value (W1 at the inflection scale)
                let infl_idx = if d1 > d2 { 1 } else { 2 };
                out.push(w1_vals[infl_idx]);
            }
        }
        debug_assert_eq!(out.len(), 144 + 60);

        // (c) σ 敏感度：3 scales × 2 windows × 3 comparisons = 18
        for &scale in &scales {
            for flat_ref in [flat_snap, flat_win] {
                let n_rows = if std::ptr::eq(flat_ref, flat_snap) {
                    n_snap
                } else {
                    n_win
                };
                // W1 for each sigma (ask+bid averaged)
                let w1_per_sigma: Vec<f32> = sigmas
                    .iter()
                    .map(|&sig| {
                        let ask =
                            extract_metric_series(flat_ref, n_rows, scale, SIDE_ASK, sig, M_W1);
                        let bid =
                            extract_metric_series(flat_ref, n_rows, scale, SIDE_BID, sig, M_W1);
                        let avg: Vec<f32> = (0..n_rows)
                            .map(|t| {
                                if ask[t].is_finite() && bid[t].is_finite() {
                                    (ask[t] + bid[t]) * 0.5
                                } else {
                                    f32::NAN
                                }
                            })
                            .collect();
                        nanmean(&avg)
                    })
                    .collect();

                // 3 comparisons
                out.push(w1_per_sigma[SIG_SPREAD] - w1_per_sigma[SIG_SHORT]); // spread - short
                out.push(w1_per_sigma[SIG_SHORT] - w1_per_sigma[SIG_EARLY]); // short - early
                out.push(w1_per_sigma[SIG_SPREAD] - w1_per_sigma[SIG_EARLY]); // spread - early
            }
        }
        debug_assert_eq!(out.len(), 144 + 60 + 18);

        // (d) 跨尺度互信息（JS/KL）：2 windows × 3 scale-pairs × 2 measures = 12
        for flat_ref in [flat_snap, flat_win] {
            let n_rows = if std::ptr::eq(flat_ref, flat_snap) {
                n_snap
            } else {
                n_win
            };
            let pairs = [(0usize, 1usize), (1, 2), (0, 2)];
            for &(s_lo, s_hi) in &pairs {
                // JS/KL between ask and bid at s_lo and s_hi, averaged
                let mut js_vals = Vec::new();
                let mut kl_vals = Vec::new();
                for &scale in &[s_lo, s_hi] {
                    // Per-row JS/KL between ask and bid distributions
                    let distances = match scale {
                        0 => (1..=10).map(|k| k as f32 * tick).collect::<Vec<f32>>(),
                        1 => (1..=5).map(|k| (2 * k) as f32 * tick).collect::<Vec<f32>>(),
                        _ => vec![5.0 * tick, 10.0 * tick],
                    };
                    for t in 0..n_rows {
                        let mid = if std::ptr::eq(flat_ref, flat_snap) {
                            ob[t].mid
                        } else {
                            // window: use average mid (approximate with ob mid at window start)
                            ob[t].mid
                        };
                        if !mid.is_finite() || mid <= 0.0 {
                            continue;
                        }
                        let ask_vols: Vec<f32> = match scale {
                            0 => ob[t].ask_vols.to_vec(),
                            1 => resample_s2(&ob[t].ask_vols).to_vec(),
                            _ => resample_s3(&ob[t].ask_vols).to_vec(),
                        };
                        let bid_vols: Vec<f32> = match scale {
                            0 => ob[t].bid_vols.to_vec(),
                            1 => resample_s2(&ob[t].bid_vols).to_vec(),
                            _ => resample_s3(&ob[t].bid_vols).to_vec(),
                        };
                        let js = js_divergence(&ask_vols, &bid_vols);
                        let kl = kl_divergence(&ask_vols, &bid_vols);
                        if js.is_finite() {
                            js_vals.push(js);
                        }
                        if kl.is_finite() {
                            kl_vals.push(kl);
                        }
                    }
                }
                out.push(nanmean(&js_vals));
                out.push(nanmean(&kl_vals));
            }
        }
        debug_assert_eq!(out.len(), 144 + 60 + 18 + 12);

        // (e) 跨窗口对比：3σ × 3 scales × 2 measures = 18
        for &sig in &sigmas {
            for &scale in &scales {
                let ask_inst = extract_metric_series(flat_snap, n_snap, scale, SIDE_ASK, sig, M_W1);
                let bid_inst = extract_metric_series(flat_snap, n_snap, scale, SIDE_BID, sig, M_W1);
                let ask_roll = extract_metric_series(flat_win, n_win, scale, SIDE_ASK, sig, M_W1);
                let bid_roll = extract_metric_series(flat_win, n_win, scale, SIDE_BID, sig, M_W1);
                let inst_avg: Vec<f32> = (0..n_snap)
                    .map(|t| {
                        if ask_inst[t].is_finite() && bid_inst[t].is_finite() {
                            (ask_inst[t] + bid_inst[t]) * 0.5
                        } else {
                            f32::NAN
                        }
                    })
                    .collect();
                let roll_avg: Vec<f32> = (0..n_win)
                    .map(|t| {
                        if ask_roll[t].is_finite() && bid_roll[t].is_finite() {
                            (ask_roll[t] + bid_roll[t]) * 0.5
                        } else {
                            f32::NAN
                        }
                    })
                    .collect();
                let w1_inst = nanmean(&inst_avg);
                let w1_roll = nanmean(&roll_avg);
                // diff
                out.push(w1_inst - w1_roll);
                // ratio
                out.push(if w1_roll.abs() > 1e-30 {
                    w1_inst / w1_roll
                } else {
                    f32::NAN
                });
            }
        }
        debug_assert_eq!(out.len(), COUNT);
        out
    }

    pub fn names() -> Vec<String> {
        let mut out = Vec::with_capacity(COUNT);
        let sigma_names = ["spread", "short", "early"];
        let scale_names = ["S1", "S2", "S3"];
        let window_names = ["inst", "roll5m"];
        let metrics = [
            "wasserstein",
            "gini",
            "entropy",
            "hhi",
            "concentration",
            "peak_pos",
            "skew",
            "kurt",
        ];

        // (a) 144: 3σ × 3 scales × 2 windows × 8 metrics
        for sig in &sigma_names {
            for sc in &scale_names {
                for w in &window_names {
                    for m in &metrics {
                        out.push(format!("qZ_{}_{}_{}_{}", m, sig, sc, w));
                    }
                }
            }
        }
        // (b) 60: 3σ × 2 windows × 10
        let b_features = [
            "w1_slope",
            "w1_curvature",
            "w1_inflection_scale",
            "w1_monotonicity",
            "w1_area",
            "fractal_dim",
            "w1_extreme_scale",
            "w1_cross_scale_var",
            "w1_convexity",
            "w1_inflection_value",
        ];
        for sig in &sigma_names {
            for w in &window_names {
                for f in &b_features {
                    out.push(format!("qZ_{}_{}_{}", f, sig, w));
                }
            }
        }
        // (c) 18: 3 scales × 2 windows × 3 comparisons
        let comparisons = ["spread_short", "short_early", "spread_early"];
        for sc in &scale_names {
            for w in &window_names {
                for c in &comparisons {
                    out.push(format!("qZ_sigma_diff_{}_{}_{}", c, sc, w));
                }
            }
        }
        // (d) 12: 2 windows × 3 pairs × 2 measures
        let pair_names = ["S1S2", "S2S3", "S1S3"];
        let measures = ["js_divergence", "kl_divergence"];
        for w in &window_names {
            for p in &pair_names {
                for m in &measures {
                    out.push(format!("qZ_{}_{}_{}", m, p, w));
                }
            }
        }
        // (e) 18: 3σ × 3 scales × 2 measures
        let e_measures = ["window_diff", "window_ratio"];
        for sig in &sigma_names {
            for sc in &scale_names {
                for m in &e_measures {
                    out.push(format!("qZ_{}_wasserstein_{}_{}", m, sig, sc));
                }
            }
        }
        debug_assert_eq!(out.len(), COUNT);
        out
    }
}

// ============================================================================
// 核心计算 + 名字 + PyO3
// ============================================================================

/// 核心唯一真相源：pipeline 和 Python 入口的共同调用点。
pub fn compute_distill_tick_full(code: &str, date: i64) -> std::io::Result<Vec<f32>> {
    // 步骤 1：数据读取
    let market = read_market_fast_inner(code, date, false, true, usize::MAX)?;
    let trade = read_trade_fast_inner(code, date, false, true, usize::MAX)?;

    // 数据缺口处理
    if market.is_empty() || market.len() < 100 {
        return Ok(vec![f32::NAN; OUT_LEN]);
    }

    // 步骤 2：共享底座计算
    let ob = extract_orderbook(&market);
    let n = ob.len();
    let tick = compute_tick(&market);
    let ob_times: Vec<f32> = ob.iter().map(|s| s.time_sec).collect();

    let sigma_spread = compute_sigma_spread(&ob);
    let sigma_short = compute_sigma_short(&trade, &ob_times);
    let sigma_early = compute_sigma_early(&trade);

    // 预计算 per-snapshot 指标
    let flat_snap =
        precompute_snapshot_metrics(&ob, &sigma_spread, &sigma_short, sigma_early, tick);

    // 5min 窗口
    let windows = build_windows(&ob_times);
    let n_win = windows.len();

    // 预计算 per-window 累积指标
    let flat_win = precompute_window_metrics(
        &ob,
        &sigma_spread,
        &sigma_short,
        sigma_early,
        tick,
        &windows,
    );

    // 步骤 3：分发到 3 个分支
    let mut out = Vec::with_capacity(OUT_LEN);
    out.extend(branch_x::compute(&flat_snap, n));
    out.extend(branch_y::compute(&flat_snap, n, &windows));
    out.extend(branch_z::compute(
        &flat_snap,
        n,
        &flat_win,
        n_win,
        &ob,
        &sigma_spread,
        &sigma_short,
        sigma_early,
        tick,
        &windows,
    ));

    assert_eq!(out.len(), OUT_LEN);
    Ok(out)
}

/// 因子名（与 compute_distill_tick_full 输出严格对齐，单一源）。
pub fn distill_tick_names() -> Vec<String> {
    let mut names = Vec::with_capacity(OUT_LEN);
    names.extend(branch_x::names());
    names.extend(branch_y::names());
    names.extend(branch_z::names());
    assert_eq!(names.len(), OUT_LEN);
    names
}

/// 单股单日调试用，错误抛异常（可见完整栈）。
#[pyfunction]
pub fn py_distill_tick(py: Python<'_>, code: &str, date: i64) -> PyResult<Vec<f32>> {
    compute_distill_tick_full(code, date)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("{:?}", e)))
}

/// Python 拿因子名。
#[pyfunction]
pub fn py_distill_tick_names() -> Vec<String> {
    distill_tick_names()
}
