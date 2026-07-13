//! 成交量序列模拟退火恢复因子。
//!
//! 读逐笔成交 → 按 bid_order / ask_order 聚合为订单级成交量 → 按时段/方向/分位数
//! 切出 65 个 universe → 每个 universe 跑确定性模拟退火 → 提取 25 因子。
//! 另有逐分钟（237 分钟）× 3 版本 × 25 因子 = 237×75 矩阵，经 get_features_factors_rust_full 降维。
//!
//! 全程确定性：固定种子 xorshift64 + first_idx 时间序排序 + 单线程退火。
//! 同股同日反复运算结果逐比特相同。

use crate::fast_csv_reader::{read_trade_fast_inner, TradeRecord};
use crate::features;
use ndarray::Array2;
use pyo3::prelude::*;
use std::collections::HashMap;

// ============================================================================
// 常量
// ============================================================================

pub const N_FACTORS: usize = 25;
pub const M_MAX_SCALAR: usize = 500_000;
pub const M_MAX_MINUTE: usize = 50_000;
pub const C1_FRAC: f64 = 1.0;
pub const SEED: u64 = 0x9E37_79B9_7F4A_7C15;
pub const N_SCALAR_SEGMENTS: usize = 65;
pub const N_MINUTES: usize = 237;
pub const N_MINUTE_VERSIONS: usize = 3;
pub const N_MINUTE_COLS: usize = N_MINUTE_VERSIONS * N_FACTORS; // 75
/// 降维后列数 = 21 * 75 + C(75,2) = 1575 + 2775
pub const N_REDUCED: usize = 21 * N_MINUTE_COLS + N_MINUTE_COLS * (N_MINUTE_COLS - 1) / 2;
pub const EXPECTED_LEN: usize = N_SCALAR_SEGMENTS * N_FACTORS + N_REDUCED; // 1625 + 4350 = 5975

const FACTOR_NAMES: &[&str] = &[
    "A1_half_life",
    "A2_steps_r80",
    "A3_steps_r90",
    "A4_final_r",
    "A5_inertia_area",
    "B1_decline_count",
    "B2_max_drawdown",
    "B3_max_recovery",
    "B4_dr_std",
    "C1_hot_dr_std",
    "C2_mid_deter_ratio",
    "C3_cold_gain",
    "C4_hot_cold_std_ratio",
    "D1_jump_runs_z",
    "D2_dr_skew",
    "D3_dr_kurt_excess",
    "E1_hurst_rs",
    "E2_dfa_alpha",
    "F1_absD_mean",
    "F2_posD_ratio",
    "F3_D_std",
    "F4_D_ac1",
    "F5_sign_flip_prob",
    "F6_hidden_big_freq",
    "F7_absD_tau_slope",
];

const WINDOW_NAMES: &[&str] = &["fullday", "early30", "late30", "mid3h", "morn2h", "aft2h"];

/// 6 个宏观时间窗口的 [sec_lo, sec_hi)（秒，相对于 t_open）。
const WINDOW_BOUNDS: [(f32, f32); 6] = [
    (0.0, 14_220.0),      // fullday: 09:30-14:57
    (0.0, 1_800.0),       // early30: 09:30-10:00
    (12_600.0, 14_220.0), // late30: 14:30-14:57
    (1_800.0, 12_600.0),  // mid3h: 10:00-14:30
    (0.0, 7_200.0),       // morn2h: 09:30-11:30
    (7_200.0, 14_220.0),  // aft2h: 13:00-14:57
];

// ============================================================================
// xorshift64 — 确定性 PRNG
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

    /// [0, n) 均匀分布。
    #[inline]
    fn next_index(&mut self, n: usize) -> usize {
        if n == 0 {
            return 0;
        }
        (self.next_u64() % n as u64) as usize
    }
}

// ============================================================================
// 统计辅助函数
// ============================================================================

/// 样本标准差（ddof=1），不足 2 点返回 NaN。
#[inline]
fn std_ddof1(data: &[f32]) -> f32 {
    let n = data.len();
    if n < 2 {
        return f32::NAN;
    }
    let mean = data.iter().sum::<f32>() / n as f32;
    let var = data
        .iter()
        .map(|&v| {
            let d = v - mean;
            d * d
        })
        .sum::<f32>()
        / (n - 1) as f32;
    var.sqrt()
}

/// Fisher-Pearson 偏度系数 G1，不足 3 点返回 NaN。
#[inline]
fn skewness(data: &[f32]) -> f32 {
    let n = data.len();
    if n < 3 {
        return f32::NAN;
    }
    let mean = data.iter().sum::<f32>() / n as f32;
    let mut m2 = 0.0f64;
    let mut m3 = 0.0f64;
    for &v in data {
        let d = (v - mean) as f64;
        m2 += d * d;
        m3 += d * d * d;
    }
    m2 /= n as f64;
    m3 /= n as f64;
    if m2 < 1e-20 {
        return f32::NAN;
    }
    let g1 = m3 / m2.powf(1.5);
    // 无偏校正
    let nf = n as f64;
    let correction = ((nf - 1.0) * nf).sqrt() / (nf - 2.0);
    (g1 * correction) as f32
}

/// 超额峰度 G2，不足 4 点返回 NaN。
#[inline]
fn excess_kurtosis(data: &[f32]) -> f32 {
    let n = data.len();
    if n < 4 {
        return f32::NAN;
    }
    let mean = data.iter().sum::<f32>() / n as f32;
    let mut m2 = 0.0f64;
    let mut m4 = 0.0f64;
    for &v in data {
        let d = (v - mean) as f64;
        let d2 = d * d;
        m2 += d2;
        m4 += d2 * d2;
    }
    m2 /= n as f64;
    m4 /= n as f64;
    if m2 < 1e-20 {
        return f32::NAN;
    }
    let g2 = m4 / (m2 * m2);
    let nf = n as f64;
    let correction =
        ((nf - 1.0) / ((nf - 2.0) * (nf - 3.0))) * ((nf + 1.0) * g2 - 3.0 * (nf - 1.0));
    correction as f32
}

/// Pearson 相关系数，长度 < 2 或方差为 0 返回 NaN。
#[inline]
fn corr(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    if n < 2 {
        return f32::NAN;
    }
    let ma = a.iter().sum::<f32>() / n as f32;
    let mb = b.iter().sum::<f32>() / n as f32;
    let mut sab = 0.0f64;
    let mut saa = 0.0f64;
    let mut sbb = 0.0f64;
    for i in 0..n {
        let da = (a[i] - ma) as f64;
        let db = (b[i] - mb) as f64;
        sab += da * db;
        saa += da * da;
        sbb += db * db;
    }
    let denom = (saa * sbb).sqrt();
    if denom < 1e-20 {
        return f32::NAN;
    }
    (sab / denom) as f32
}

/// 绝对值的分位数（与 numpy.quantile linear 插值一致）。
fn percentile_abs(data: &[f32], q: f64) -> f32 {
    let mut sorted: Vec<f32> = data.iter().map(|v| v.abs()).collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    if n == 0 {
        return f32::NAN;
    }
    if n == 1 {
        return sorted[0];
    }
    let pos = q * (n as f64 - 1.0);
    let lo = pos.floor() as usize;
    let hi = (lo + 1).min(n - 1);
    let frac = pos - lo as f64;
    sorted[lo] * (1.0 - frac as f32) + sorted[hi] * frac as f32
}

/// 线性回归斜率（y 对 x），不足 3 点或 x 方差为 0 返回 NaN。
fn linear_slope(x: &[f32], y: &[f32]) -> f32 {
    let n = x.len().min(y.len());
    if n < 3 {
        return f32::NAN;
    }
    let mx = x.iter().sum::<f32>() / n as f32;
    let my = y.iter().sum::<f32>() / n as f32;
    let mut sxy = 0.0f64;
    let mut sxx = 0.0f64;
    for i in 0..n {
        let dx = (x[i] - mx) as f64;
        sxy += dx * (y[i] - my) as f64;
        sxx += dx * dx;
    }
    if sxx < 1e-20 {
        return f32::NAN;
    }
    (sxy / sxx) as f32
}

/// 游程检验 z-score。data 为 0/1 序列（f32 形式）。
fn runs_test_z(data: &[f32]) -> f32 {
    let n = data.len();
    if n < 2 {
        return f32::NAN;
    }
    let n1 = data.iter().filter(|&&v| v >= 0.5).count();
    let n0 = n - n1;
    if n0 == 0 || n1 == 0 {
        return f32::NAN;
    }
    // 计算游程数
    let mut runs = 1usize;
    for i in 1..n {
        let prev_is_one = data[i - 1] >= 0.5;
        let curr_is_one = data[i] >= 0.5;
        if prev_is_one != curr_is_one {
            runs += 1;
        }
    }
    let n_total = n as f64;
    let er = 2.0 * n0 as f64 * n1 as f64 / n_total + 1.0;
    let var_num = 2.0 * n0 as f64 * n1 as f64 * (2.0 * n0 as f64 * n1 as f64 - n_total);
    let var_den = n_total * n_total * (n_total - 1.0);
    if var_den < 1e-20 {
        return f32::NAN;
    }
    let var_r = var_num / var_den;
    if var_r < 1e-20 {
        return f32::NAN;
    }
    ((runs as f64 - er) / var_r.sqrt()) as f32
}

/// R/S 分析 Hurst 指数。
fn hurst_rs(data: &[f32]) -> f32 {
    let n = data.len();
    if n < 16 {
        return f32::NAN;
    }

    // 窗口大小：2 的幂
    let mut sizes: Vec<usize> = Vec::new();
    let mut w = 4usize;
    while w <= n / 2 {
        sizes.push(w);
        w *= 2;
    }
    if sizes.len() < 3 {
        return f32::NAN;
    }

    let mut log_sizes: Vec<f64> = Vec::new();
    let mut log_rs: Vec<f64> = Vec::new();

    for &w in &sizes {
        let n_chunks = n / w;
        let mut rs_sum = 0.0f64;
        let mut rs_count = 0usize;

        for c in 0..n_chunks {
            let chunk = &data[c * w..(c + 1) * w];
            let mean: f64 = chunk.iter().map(|&v| v as f64).sum::<f64>() / w as f64;

            // 累积离差
            let mut cum_dev = 0.0f64;
            let mut min_cd = f64::INFINITY;
            let mut max_cd = f64::NEG_INFINITY;
            let mut ss = 0.0f64;
            for &v in chunk {
                let d = v as f64 - mean;
                cum_dev += d;
                if cum_dev < min_cd {
                    min_cd = cum_dev;
                }
                if cum_dev > max_cd {
                    max_cd = cum_dev;
                }
                ss += d * d;
            }
            let r = max_cd - min_cd;
            let s = (ss / w as f64).sqrt();
            if s > 1e-10 {
                rs_sum += r / s;
                rs_count += 1;
            }
        }

        if rs_count > 0 {
            let avg_rs = rs_sum / rs_count as f64;
            if avg_rs > 0.0 {
                log_sizes.push((w as f64).ln());
                log_rs.push(avg_rs.ln());
            }
        }
    }

    if log_sizes.len() < 3 {
        return f32::NAN;
    }
    // 最小二乘拟合斜率
    let ns = log_sizes.len();
    let mx = log_sizes.iter().sum::<f64>() / ns as f64;
    let my = log_rs.iter().sum::<f64>() / ns as f64;
    let mut sxy = 0.0f64;
    let mut sxx = 0.0f64;
    for i in 0..ns {
        let dx = log_sizes[i] - mx;
        sxy += dx * (log_rs[i] - my);
        sxx += dx * dx;
    }
    if sxx < 1e-20 {
        return f32::NAN;
    }
    (sxy / sxx) as f32
}

/// DFA 标度指数 α。
fn dfa_alpha(data: &[f32]) -> f32 {
    let n = data.len();
    if n < 16 {
        return f32::NAN;
    }

    // 累积离差序列
    let mean: f64 = data.iter().map(|&v| v as f64).sum::<f64>() / n as f64;
    let mut y = vec![0.0f64; n];
    y[0] = data[0] as f64 - mean;
    for i in 1..n {
        y[i] = y[i - 1] + data[i] as f64 - mean;
    }

    let mut sizes: Vec<usize> = Vec::new();
    let mut w = 4usize;
    while w <= n / 4 {
        sizes.push(w);
        w *= 2;
    }
    if sizes.len() < 3 {
        return f32::NAN;
    }

    let mut log_sizes: Vec<f64> = Vec::new();
    let mut log_f: Vec<f64> = Vec::new();

    for &w in &sizes {
        let n_seg = n / w;
        let mut var_sum = 0.0f64;
        let mut seg_count = 0usize;

        for seg in 0..n_seg {
            let start = seg * w;
            // 线性回归 y[start..start+w] vs [0..w]
            let mx = (0..w).map(|i| i as f64).sum::<f64>() / w as f64;
            let my = (0..w).map(|i| y[start + i]).sum::<f64>() / w as f64;
            let mut sxy = 0.0f64;
            let mut sxx = 0.0f64;
            for i in 0..w {
                let dx = i as f64 - mx;
                sxy += dx * (y[start + i] - my);
                sxx += dx * dx;
            }
            if sxx < 1e-20 {
                continue;
            }
            let slope = sxy / sxx;
            let intercept = my - slope * mx;

            let mut sq_resid = 0.0f64;
            for i in 0..w {
                let predicted = slope * i as f64 + intercept;
                let resid = y[start + i] - predicted;
                sq_resid += resid * resid;
            }
            var_sum += sq_resid / w as f64;
            seg_count += 1;
        }

        if seg_count > 0 {
            let f = (var_sum / seg_count as f64).sqrt();
            if f > 1e-10 {
                log_sizes.push((w as f64).ln());
                log_f.push(f.ln());
            }
        }
    }

    if log_sizes.len() < 3 {
        return f32::NAN;
    }
    let ns = log_sizes.len();
    let mx = log_sizes.iter().sum::<f64>() / ns as f64;
    let my = log_f.iter().sum::<f64>() / ns as f64;
    let mut sxy = 0.0f64;
    let mut sxx = 0.0f64;
    for i in 0..ns {
        let dx = log_sizes[i] - mx;
        sxy += dx * (log_f[i] - my);
        sxx += dx * dx;
    }
    if sxx < 1e-20 {
        return f32::NAN;
    }
    (sxy / sxx) as f32
}

// ============================================================================
// 订单聚合
// ============================================================================

#[derive(Clone)]
struct AggOrder {
    volume: f32,
    n_active: u32,
    n_passive: u32,
    first_idx: usize,
}

/// 在 trade 切片内按 bid_order / ask_order 聚合。
/// first_idx = 该订单在切片中的首次出现位置（时间序，确定性关键）。
fn aggregate_orders(trades: &[TradeRecord]) -> (HashMap<i64, AggOrder>, HashMap<i64, AggOrder>) {
    let mut bid_map: HashMap<i64, AggOrder> = HashMap::with_capacity(trades.len());
    let mut ask_map: HashMap<i64, AggOrder> = HashMap::with_capacity(trades.len());

    for (i, t) in trades.iter().enumerate() {
        // bid_order: flag=66 → 主动, flag=83 → 被动
        let bid_e = bid_map.entry(t.bid_order).or_insert(AggOrder {
            volume: 0.0,
            n_active: 0,
            n_passive: 0,
            first_idx: i,
        });
        bid_e.volume += t.volume;
        match t.flag {
            66 => bid_e.n_active += 1,
            83 => bid_e.n_passive += 1,
            _ => {}
        }

        // ask_order: flag=83 → 主动, flag=66 → 被动
        let ask_e = ask_map.entry(t.ask_order).or_insert(AggOrder {
            volume: 0.0,
            n_active: 0,
            n_passive: 0,
            first_idx: i,
        });
        ask_e.volume += t.volume;
        match t.flag {
            83 => ask_e.n_active += 1,
            66 => ask_e.n_passive += 1,
            _ => {}
        }
    }

    (bid_map, ask_map)
}

// ============================================================================
// 方向 (Side) 与提取
// ============================================================================

#[derive(Clone, Copy, PartialEq)]
enum Side {
    Bid,
    Ask,
    Mixed,
    Active,
    Passive,
    ActBid,
    ActAsk,
    PasBid,
    PasAsk,
}

impl Side {
    fn as_str(&self) -> &'static str {
        match self {
            Side::Bid => "bid",
            Side::Ask => "ask",
            Side::Mixed => "mixed",
            Side::Active => "active",
            Side::Passive => "passive",
            Side::ActBid => "actbid",
            Side::ActAsk => "actask",
            Side::PasBid => "pasbid",
            Side::PasAsk => "pasask",
        }
    }
}

/// 从 bid_map 提取 volume（按 first_idx 时间序），可选过滤。
fn extract_from_map(
    map: &HashMap<i64, AggOrder>,
    side_tag: u8,
    filter: impl Fn(&AggOrder) -> bool,
) -> Vec<(usize, u8, f32)> {
    let mut v: Vec<(usize, u8, f32)> = map
        .values()
        .filter(|e| filter(e))
        .map(|e| (e.first_idx, side_tag, e.volume))
        .collect();
    v.sort_by(|a, b| (a.0, a.1).cmp(&(b.0, b.1)));
    v
}

/// 按 Side 提取 volume 列表（时间序确定）。
fn extract_side(
    bid_map: &HashMap<i64, AggOrder>,
    ask_map: &HashMap<i64, AggOrder>,
    side: Side,
) -> Vec<f32> {
    let triples = match side {
        Side::Bid => extract_from_map(bid_map, 0, |_| true),
        Side::Ask => extract_from_map(ask_map, 0, |_| true),
        Side::Mixed => {
            let mut v = extract_from_map(bid_map, 0, |_| true);
            v.extend(extract_from_map(ask_map, 1, |_| true));
            v.sort_by(|a, b| (a.0, a.1).cmp(&(b.0, b.1)));
            v
        }
        Side::Active => {
            let f = |e: &AggOrder| e.n_active > 0 && e.n_passive == 0;
            let mut v = extract_from_map(bid_map, 0, f);
            v.extend(extract_from_map(ask_map, 1, f));
            v.sort_by(|a, b| (a.0, a.1).cmp(&(b.0, b.1)));
            v
        }
        Side::Passive => {
            let f = |e: &AggOrder| e.n_passive > 0 && e.n_active == 0;
            let mut v = extract_from_map(bid_map, 0, f);
            v.extend(extract_from_map(ask_map, 1, f));
            v.sort_by(|a, b| (a.0, a.1).cmp(&(b.0, b.1)));
            v
        }
        Side::ActBid => extract_from_map(bid_map, 0, |e| e.n_active > 0 && e.n_passive == 0),
        Side::ActAsk => extract_from_map(ask_map, 0, |e| e.n_active > 0 && e.n_passive == 0),
        Side::PasBid => extract_from_map(bid_map, 0, |e| e.n_passive > 0 && e.n_active == 0),
        Side::PasAsk => extract_from_map(ask_map, 0, |e| e.n_passive > 0 && e.n_active == 0),
    };
    triples.into_iter().map(|(_, _, vol)| vol).collect()
}

// ============================================================================
// 分位数过滤
// ============================================================================

#[derive(Clone, Copy, PartialEq)]
enum Quantile {
    All,
    Top10,
    Mid50,
    Bot40,
}

impl Quantile {
    fn as_str(&self) -> &'static str {
        match self {
            Quantile::All => "all",
            Quantile::Top10 => "top10",
            Quantile::Mid50 => "mid50",
            Quantile::Bot40 => "bot40",
        }
    }
}

/// 按个数分位数过滤 volume 列表，**保留时间序**。
fn quantile_filter(volumes: &[f32], q: Quantile) -> Vec<f32> {
    let n = volumes.len();
    if n == 0 || q == Quantile::All {
        return volumes.to_vec();
    }
    // 按值降序排列，确定保留哪些原始索引
    let mut indexed: Vec<(f32, usize)> = volumes.iter().enumerate().map(|(i, &v)| (v, i)).collect();
    indexed.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let (start, end) = match q {
        Quantile::Top10 => {
            let k = ((n as f64) * 0.1).ceil() as usize;
            (0, k.max(1).min(n))
        }
        Quantile::Mid50 => {
            let s = ((n as f64) * 0.1) as usize;
            let e = ((n as f64) * 0.6) as usize;
            (s, e.min(n).max(s))
        }
        Quantile::Bot40 => {
            let k = ((n as f64) * 0.4) as usize;
            (n - k, n)
        }
        _ => (0, n),
    };

    let mut keep = vec![false; n];
    for &(_, idx) in &indexed[start..end] {
        keep[idx] = true;
    }
    volumes
        .iter()
        .enumerate()
        .filter(|(i, _)| keep[*i])
        .map(|(_, &v)| v)
        .collect()
}

// ============================================================================
// 65 个标量片段定义
// ============================================================================

/// 返回 65 个 (window_idx, side, quantile)，顺序严格对齐输出和命名。
fn segment_defs() -> Vec<(usize, Side, Quantile)> {
    let mut segs = Vec::with_capacity(65);

    // Group 1: 6 窗口 × (bid, ask, mixed), all (18)
    for &w in &[0, 1, 2, 3, 4, 5] {
        for &s in &[Side::Bid, Side::Ask, Side::Mixed] {
            segs.push((w, s, Quantile::All));
        }
    }
    // Group 2: fullday × (bid, ask, mixed) × (top10, mid50, bot40) (9)
    for &s in &[Side::Bid, Side::Ask, Side::Mixed] {
        for &q in &[Quantile::Top10, Quantile::Mid50, Quantile::Bot40] {
            segs.push((0, s, q));
        }
    }
    // Group 3: fullday × (active, passive), all (2)
    for &s in &[Side::Active, Side::Passive] {
        segs.push((0, s, Quantile::All));
    }
    // Group 4: fullday × (active, passive) × (top10, mid50, bot40) (6)
    for &s in &[Side::Active, Side::Passive] {
        for &q in &[Quantile::Top10, Quantile::Mid50, Quantile::Bot40] {
            segs.push((0, s, q));
        }
    }
    // Group 5: fullday × (actbid, pasbid, actask, pasask) × (top10, mid50, bot40) (12)
    for &s in &[Side::ActBid, Side::PasBid, Side::ActAsk, Side::PasAsk] {
        for &q in &[Quantile::Top10, Quantile::Mid50, Quantile::Bot40] {
            segs.push((0, s, q));
        }
    }
    // Group 6a: late30(2) × (actbid, pasbid, actask, pasask, active, passive), all (6)
    for &s in &[
        Side::ActBid,
        Side::PasBid,
        Side::ActAsk,
        Side::PasAsk,
        Side::Active,
        Side::Passive,
    ] {
        segs.push((2, s, Quantile::All));
    }
    // Group 6b: late30(2) × (actbid, actask, pasbid, pasask) × (top10, bot40, mid50) (12)
    for &s in &[Side::ActBid, Side::ActAsk, Side::PasBid, Side::PasAsk] {
        for &q in &[Quantile::Top10, Quantile::Bot40, Quantile::Mid50] {
            segs.push((2, s, q));
        }
    }

    assert_eq!(segs.len(), N_SCALAR_SEGMENTS);
    segs
}

// ============================================================================
// 退火引擎
// ============================================================================

/// 对真实成交量序列 true_vol 跑确定性模拟退火，返回 25 个因子。
///
/// - true_vol: 时间序排列的真实成交量（ground truth T）
/// - m_max: 步数预算上限
///
/// 全程确定性：固定种子、单线程、增量 ΔS 更新。
fn anneal(true_vol: &[f32], m_max: usize) -> [f32; N_FACTORS] {
    let n = true_vol.len();
    if n < 2 {
        return [f32::NAN; N_FACTORS];
    }

    // 总体方差 σ²（除以 N）
    let mean: f64 = true_vol.iter().map(|&v| v as f64).sum::<f64>() / n as f64;
    let sigma2: f64 = true_vol
        .iter()
        .map(|&v| {
            let d = v as f64 - mean;
            d * d
        })
        .sum::<f64>()
        / n as f64;
    if sigma2 <= 0.0 {
        return [f32::NAN; N_FACTORS];
    }

    // G0 = sorted(T) ascending
    let mut guess: Vec<f32> = {
        let mut g = true_vol.to_vec();
        g.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        g
    };

    // S0 = Σ (g_k - t_k)²
    let mut s: f64 = (0..n)
        .map(|k| {
            let d = guess[k] as f64 - true_vol[k] as f64;
            d * d
        })
        .sum();

    let denom = 2.0 * sigma2 * n as f64;
    let r0 = 1.0 - s / denom;

    // median（用于 F6）
    let median = {
        let mut sorted_t = true_vol.to_vec();
        sorted_t.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        if n % 2 == 0 {
            (sorted_t[n / 2 - 1] + sorted_t[n / 2]) * 0.5
        } else {
            sorted_t[n / 2]
        }
    };

    let mut rng = XorShift64::new(SEED);

    let m = m_max;
    let mut r_seq = vec![0.0f32; m];
    let mut d_vals: Vec<f32> = Vec::new();
    let mut d_tau: Vec<usize> = Vec::new();
    let mut g_below: Vec<bool> = Vec::new();

    let s_tol = (sigma2 * n as f64 * 1e-10_f64).max(1e-12);

    for t in 0..m {
        // 取对 (i, j)
        let i = rng.next_index(n);
        let mut j = rng.next_index(n);
        if j == i {
            j = rng.next_index(n);
        }

        let current_r = (1.0 - s / denom) as f32;
        r_seq[t] = current_r;

        if i == j {
            continue;
        }

        // ΔS = 2*(g_i - g_j)*(t_i - t_j)
        let delta_s =
            2.0 * (guess[i] as f64 - guess[j] as f64) * (true_vol[i] as f64 - true_vol[j] as f64);

        // 温度
        let c_t = if m > 1 {
            sigma2 * C1_FRAC * (1.0 - t as f64 / (m as f64 - 1.0)).max(0.0)
        } else {
            0.0
        };

        // 探测记录（每步，无论是否接受，若 g_i≠g_j）
        if guess[i] != guess[j] {
            d_vals.push(true_vol[j] - true_vol[i]);
            d_tau.push(if j > i { j - i } else { i - j });
            g_below.push(guess[i] < median);
        }

        // 接受/拒绝
        if delta_s < 0.0 || delta_s < c_t {
            s += delta_s;
            guess.swap(i, j);
            r_seq[t] = (1.0 - s / denom) as f32;
        }

        // 提前终止
        if s <= s_tol {
            let final_r = r_seq[t];
            for k in (t + 1)..m {
                r_seq[k] = final_r;
            }
            break;
        }
    }

    compute_factors(&r_seq, &d_vals, &d_tau, &g_below, r0 as f32)
}

/// 从退火轨迹计算 25 个因子。
fn compute_factors(
    r_seq: &[f32],
    d_vals: &[f32],
    d_tau: &[usize],
    g_below: &[bool],
    r0: f32,
) -> [f32; N_FACTORS] {
    let m = r_seq.len();
    let mut f = [f32::NAN; N_FACTORS];

    // A1: 半衰步数 — 最小 t 使 r_seq[t] ≥ (1+r0)/2
    let hl_thresh = (1.0 + r0) * 0.5;
    f[0] = r_seq
        .iter()
        .position(|&r| r >= hl_thresh)
        .map(|t| t as f32)
        .unwrap_or(m as f32);

    // A2: 达 80% 步数
    f[1] = r_seq
        .iter()
        .position(|&r| r >= 0.80)
        .map(|t| t as f32)
        .unwrap_or(m as f32);

    // A3: 达 90% 步数
    f[2] = r_seq
        .iter()
        .position(|&r| r >= 0.90)
        .map(|t| t as f32)
        .unwrap_or(m as f32);

    // A4: 最终 r
    f[3] = r_seq[m - 1];

    // A5: 总惰性面积
    f[4] = r_seq.iter().map(|&r| 1.0 - r).sum();

    // B1: 下降总次数
    f[5] = (1..m).filter(|&t| r_seq[t] < r_seq[t - 1]).count() as f32;

    // B2: 最大回撤深度
    let mut running_max = r_seq[0];
    let mut max_dd = 0.0f32;
    for &r in r_seq {
        if r > running_max {
            running_max = r;
        }
        let dd = running_max - r;
        if dd > max_dd {
            max_dd = dd;
        }
    }
    f[6] = max_dd;

    // B3: 最长回撤恢复时间（最长 underwater 连续步数）
    let mut max_recovery = 0usize;
    let mut peak_step = 0usize;
    running_max = r_seq[0];
    for t in 1..m {
        if r_seq[t] >= running_max {
            running_max = r_seq[t];
            peak_step = t;
        } else {
            let underwater = t - peak_step;
            if underwater > max_recovery {
                max_recovery = underwater;
            }
        }
    }
    f[7] = max_recovery as f32;

    // Δr 序列
    let dr: Vec<f32> = (1..m).map(|t| r_seq[t] - r_seq[t - 1]).collect();

    // B4: Δr 波动率
    f[8] = std_ddof1(&dr);

    // 三等分
    let seg_size = m / 3;
    let seg1_end = seg_size;
    let seg2_end = 2 * seg_size;

    // C1: 高温段 Δr std
    let dr_seg1: Vec<f32> = (1..seg1_end.max(1))
        .map(|t| r_seq[t] - r_seq[t - 1])
        .collect();
    f[9] = std_ddof1(&dr_seg1);

    // C2: 中温段劣化接受比
    if seg2_end > seg1_end && seg1_end > 0 {
        let mid_count = (seg1_end..seg2_end)
            .filter(|&t| r_seq[t] < r_seq[t - 1])
            .count();
        f[10] = mid_count as f32 / (seg2_end - seg1_end) as f32;
    }

    // C3: 低温段改善量
    if seg2_end > 0 && m > seg2_end {
        f[11] = r_seq[m - 1] - r_seq[seg2_end - 1];
    }

    // C4: 高/低温波动比
    let dr_seg3: Vec<f32> = if seg2_end > 0 {
        (seg2_end..m).map(|t| r_seq[t] - r_seq[t - 1]).collect()
    } else {
        vec![]
    };
    let std_seg3 = std_ddof1(&dr_seg3);
    if std_seg3.abs() > 1e-10 {
        f[12] = f[9] / std_seg3;
    }

    // D1: 大跳变游程 z
    if !dr.is_empty() {
        let p90 = percentile_abs(&dr, 0.90);
        let binary: Vec<f32> = dr
            .iter()
            .map(|&v| if v.abs() >= p90 { 1.0 } else { 0.0 })
            .collect();
        f[13] = runs_test_z(&binary);
    }

    // D2: Δr 偏度
    f[14] = skewness(&dr);

    // D3: Δr 超额峰度
    f[15] = excess_kurtosis(&dr);

    // E1: Hurst (R/S)
    f[16] = hurst_rs(r_seq);

    // E2: DFA alpha
    f[17] = dfa_alpha(r_seq);

    // F1-F7: 探测差值
    let k = d_vals.len();
    if k > 0 {
        // F1: |D| 均值
        f[18] = d_vals.iter().map(|d| d.abs()).sum::<f32>() / k as f32;

        // F2: D 正数比
        f[19] = d_vals.iter().filter(|d| **d > 0.0).count() as f32 / k as f32;

        // F3: D 标准差
        f[20] = std_ddof1(d_vals);

        // F4: D 一阶自相关
        if k >= 2 {
            f[21] = corr(&d_vals[..k - 1], &d_vals[1..]);
        }

        // F5: 符号反转概率
        if k >= 2 {
            let flips = (0..k - 1)
                .filter(|&i| d_vals[i].signum() != d_vals[i + 1].signum())
                .count();
            f[22] = flips as f32 / (k - 1) as f32;
        }

        // F6: 潜伏大单探测频率
        let p95 = percentile_abs(d_vals, 0.95);
        let hidden = (0..k)
            .filter(|&i| g_below[i] && d_vals[i].abs() > p95)
            .count();
        f[23] = hidden as f32 / k as f32;

        // F7: |D| 对 τ 的线性回归斜率
        if k >= 3 {
            let abs_d: Vec<f32> = d_vals.iter().map(|d| d.abs()).collect();
            let tau_f: Vec<f32> = d_tau.iter().map(|&t| t as f32).collect();
            f[24] = linear_slope(&tau_f, &abs_d);
        }
    }

    f
}

// ============================================================================
// 主计算入口
// ============================================================================

/// 核心：pipeline 和 Python 的唯一共同调用点。
pub fn compute_anneal_volume_full(code: &str, date: i64) -> std::io::Result<Vec<f32>> {
    let trades = read_trade_fast_inner(code, date, false, true, usize::MAX)?;
    if trades.is_empty() {
        return Ok(vec![f32::NAN; EXPECTED_LEN]);
    }

    let t_open = trades.first().unwrap().time_sec;

    // 6 个宏观窗口聚合（缓存）
    let win_aggs: Vec<(HashMap<i64, AggOrder>, HashMap<i64, AggOrder>)> = WINDOW_BOUNDS
        .iter()
        .map(|&(sec_lo, sec_hi)| {
            let lo_time = t_open + sec_lo;
            let hi_time = t_open + sec_hi;
            let lo = trades.partition_point(|t| t.time_sec < lo_time);
            let hi = trades.partition_point(|t| t.time_sec < hi_time);
            aggregate_orders(&trades[lo..hi])
        })
        .collect();

    // 65 个标量片段
    let segs = segment_defs();
    let mut out: Vec<f32> = Vec::with_capacity(EXPECTED_LEN);

    for &(win_idx, side, quantile) in &segs {
        let (bid_map, ask_map) = &win_aggs[win_idx];
        let vols = extract_side(bid_map, ask_map, side);
        let filtered = quantile_filter(&vols, quantile);
        let factors = anneal(&filtered, M_MAX_SCALAR);
        out.extend_from_slice(&factors);
    }

    // 逐分钟矩阵 237 × 75
    let minute_col_names = build_minute_col_names();
    let mut matrix = Array2::zeros((N_MINUTES, N_MINUTE_COLS));

    for m_idx in 0..N_MINUTES {
        let lo_time = t_open + (m_idx as f32) * 60.0;
        let hi_time = t_open + ((m_idx + 1) as f32) * 60.0;
        let lo = trades.partition_point(|t| t.time_sec < lo_time);
        let hi = trades.partition_point(|t| t.time_sec < hi_time);

        let (bid_map, ask_map) = aggregate_orders(&trades[lo..hi]);

        // 3 个版本
        let versions = [Side::Bid, Side::Ask, Side::Mixed];
        for (vi, &ver) in versions.iter().enumerate() {
            let vols = extract_side(&bid_map, &ask_map, ver);
            let factors = anneal(&vols, M_MAX_MINUTE);
            for (fi, &val) in factors.iter().enumerate() {
                matrix[[m_idx, vi * N_FACTORS + fi]] = val;
            }
        }
    }

    // 降维
    let (reduced_vals, _) =
        features::get_features_factors_rust_full(&matrix.view(), &minute_col_names, false);
    out.extend_from_slice(&reduced_vals);

    // 长度校准
    if out.len() < EXPECTED_LEN {
        out.resize(EXPECTED_LEN, f32::NAN);
    } else if out.len() > EXPECTED_LEN {
        out.truncate(EXPECTED_LEN);
    }

    Ok(out)
}

// ============================================================================
// 因子名
// ============================================================================

fn build_minute_col_names() -> Vec<String> {
    let versions = ["bid", "ask", "mixed"];
    let mut names = Vec::with_capacity(N_MINUTE_COLS);
    for ver in &versions {
        for factor in FACTOR_NAMES {
            names.push(format!("min_{}_{}", ver, factor));
        }
    }
    names
}

/// 生成全部 5975 个因子名（与 compute_anneal_volume_full 输出严格对齐）。
pub fn anneal_volume_names() -> Vec<String> {
    let mut names: Vec<String> = Vec::with_capacity(EXPECTED_LEN);

    // 65 标量片段名
    let segs = segment_defs();
    for &(win_idx, side, quantile) in &segs {
        let seg_prefix = format!(
            "{}_{}_{}",
            WINDOW_NAMES[win_idx],
            side.as_str(),
            quantile.as_str()
        );
        for factor in FACTOR_NAMES {
            names.push(format!("{}_{}", seg_prefix, factor));
        }
    }

    // 逐分钟降维名：用 1×75 dummy 矩阵调 get_features_factors_rust_full 拿名字
    let minute_col_names = build_minute_col_names();
    let dummy = Array2::ones((1, N_MINUTE_COLS));
    let (_, reduced_names) =
        features::get_features_factors_rust_full(&dummy.view(), &minute_col_names, false);
    names.extend(reduced_names);

    // 兜底：如果降维名字不足（理论上不会），补齐
    while names.len() < EXPECTED_LEN {
        names.push(format!("extra_{}", names.len()));
    }
    names.truncate(EXPECTED_LEN);

    names
}

// ============================================================================
// PyO3 接口
// ============================================================================

/// Python 可调用：py_anneal_volume(code, date)
/// 单股单日计算，返回 5975 个因子值。错误抛 PyIOError。
#[pyfunction]
pub fn py_anneal_volume(code: &str, date: i64) -> PyResult<Vec<f32>> {
    compute_anneal_volume_full(code, date)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("{:?}", e)))
}

/// Python 拿因子名。
#[pyfunction]
pub fn py_anneal_volume_names() -> Vec<String> {
    anneal_volume_names()
}
