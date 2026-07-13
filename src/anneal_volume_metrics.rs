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
use rustc_hash::FxHashMap;

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

/// 自适应步数：M = min(m_max_cap, max(2000, N²×10))。
/// 小 N 时大幅减少步数（N=10→M=2000），大 N 时跑满上限。
/// 配合 S=0 提前终止，多数小 N 片段在几百步内收敛。
fn adaptive_m_max(n: usize, m_max_cap: usize) -> usize {
    (n * n * 10).max(2000).min(m_max_cap)
}
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

    /// [0, n) 均匀分布。用高 32 位乘 n 再右移 32 位，替代昂贵的 % n 整数除法。
    #[inline]
    fn next_index(&mut self, n: usize) -> usize {
        if n == 0 {
            return 0;
        }
        ((self.next_u64() >> 32) as u64).wrapping_mul(n as u64) as usize >> 32
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
fn aggregate_orders(trades: &[TradeRecord]) -> (FxHashMap<i64, AggOrder>, FxHashMap<i64, AggOrder>) {
    let mut bid_map: FxHashMap<i64, AggOrder> = FxHashMap::with_capacity_and_hasher(
        trades.len(),
        Default::default(),
    );
    let mut ask_map: FxHashMap<i64, AggOrder> = FxHashMap::with_capacity_and_hasher(
        trades.len(),
        Default::default(),
    );

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
    #[inline]
    fn index(self) -> usize {
        self as usize
    }

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
    map: &FxHashMap<i64, AggOrder>,
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
    bid_map: &FxHashMap<i64, AggOrder>,
    ask_map: &FxHashMap<i64, AggOrder>,
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

/// 使用已按成交量降序排列的索引做分位数过滤，保留原始时间序。
/// 排序顺序与 `quantile_filter` 完全一致，用于同一 universe 的多个分位数。
fn quantile_filter_from_sorted(
    volumes: &[f32],
    sorted: &[(f32, usize)],
    q: Quantile,
) -> Vec<f32> {
    let n = volumes.len();
    if n == 0 || q == Quantile::All {
        return volumes.to_vec();
    }
    let (start, end) = match q {
        Quantile::Top10 => {
            let k = ((n as f64) * 0.1).ceil() as usize;
            (0, k.max(1).min(n))
        }
        Quantile::Mid50 => {
            let start = ((n as f64) * 0.1) as usize;
            let end = ((n as f64) * 0.6) as usize;
            (start, end.min(n).max(start))
        }
        Quantile::Bot40 => {
            let k = ((n as f64) * 0.4) as usize;
            (n - k, n)
        }
        Quantile::All => unreachable!(),
    };
    let mut keep = vec![false; n];
    for &(_, idx) in &sorted[start..end] {
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
// 退火引擎（优化版：在线统计 + 采样 r_seq + 缓冲区复用）
// ============================================================================

const R_SAMPLE_MAX: usize = 2000;
const D_MAX: usize = 5000;

struct AnnealBuf {
    guess: Vec<f32>,
    r_sample: Vec<f32>,
    d_vals: Vec<f32>,
    d_tau: Vec<f32>,
    g_below: Vec<u8>,
}

impl AnnealBuf {
    fn new() -> Self {
        Self {
            guess: Vec::new(),
            r_sample: Vec::new(),
            d_vals: Vec::new(),
            d_tau: Vec::new(),
            g_below: Vec::new(),
        }
    }
}

fn anneal(true_vol: &[f32], m_max: usize, buf: &mut AnnealBuf) -> [f32; N_FACTORS] {
    let n = true_vol.len();
    if n < 2 {
        return [f32::NAN; N_FACTORS];
    }
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
    buf.guess.clear();
    buf.guess.extend_from_slice(true_vol);
    buf.guess
        .sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mut s: f64 = (0..n)
        .map(|k| {
            let d = buf.guess[k] as f64 - true_vol[k] as f64;
            d * d
        })
        .sum();
    let denom = 2.0 * sigma2 * n as f64;
    let inv_denom = 1.0 / denom;
    let r0 = 1.0_f64 - s * inv_denom;
    let median = (if n % 2 == 0 {
        (buf.guess[n / 2 - 1] + buf.guess[n / 2]) * 0.5
    } else {
        buf.guess[n / 2]
    }) as f32;
    buf.r_sample.clear();
    buf.d_vals.clear();
    buf.d_tau.clear();
    let s_tol = (sigma2 * n as f64 * 1e-10_f64).max(1e-12);
    let ct_base = sigma2 * C1_FRAC;
    let inv_m = if m_max > 1 {
        1.0 / (m_max as f64 - 1.0)
    } else {
        0.0
    };
    let stride = if m_max <= R_SAMPLE_MAX {
        1
    } else {
        (m_max + R_SAMPLE_MAX - 1) / R_SAMPLE_MAX
    };
    let mut rng = XorShift64::new(SEED);
    let mut prev_r = r0;
    let mut final_r = r0;
    let hl = (1.0 + r0) * 0.5;
    let mut a1 = usize::MAX;
    let mut a2 = usize::MAX;
    let mut a3 = usize::MAX;
    let mut inertia = 0.0f64;
    let mut decl = 0u32;
    let mut rmr = r0;
    let mut mdd = 0.0f64;
    let mut ps = 0usize;
    let mut mr = 0usize;
    let mut drn = 0u32;
    let mut dr_s1 = 0.0f64;
    let mut dr_s2 = 0.0f64;
    let mut dr_s3 = 0.0f64;
    let mut dr_s4 = 0.0f64;
    let ss = m_max / 3;
    let s1e = ss;
    let s2e = 2 * ss;
    let mut s1n = 0u32;
    let mut s1s = 0.0f64;
    let mut s1sq = 0.0f64;
    let mut s2n = 0u32;
    let mut s2d = 0u32;
    let mut s3n = 0u32;
    let mut s3s = 0.0f64;
    let mut s3sq = 0.0f64;
    let mut r_s2 = f64::NAN;
    let mut d_count = 0usize;
    let mut t = 0usize;
    while t < m_max {
        let i = rng.next_index(n);
        let mut j = rng.next_index(n);
        if j == i {
            j = rng.next_index(n);
        }
        // 1. 探测记录（交换前 guess）— 限制总量到 D_MAX，避免大 N 的百万级排序
        if d_count < D_MAX && i != j && buf.guess[i] != buf.guess[j] {
            buf.d_vals.push(true_vol[j] - true_vol[i]);
            buf.d_tau.push(if j > i {
                (j - i) as f32
            } else {
                (i - j) as f32
            });
            buf.g_below.push(if buf.guess[i] < median { 1 } else { 0 });
            d_count += 1;
        }
        // 2. 接受/拒绝交换（ΔS 用 f32 计算，省 4 次 f32→f64 转换）
        if i != j {
            let gi = buf.guess[i];
            let gj = buf.guess[j];
            let ti = true_vol[i];
            let tj = true_vol[j];
            let ds_f32 = 2.0f32 * (gi - gj) * (ti - tj);
            let ct = ct_base * (1.0 - t as f64 * inv_m).max(0.0);
            if (ds_f32 as f64) < 0.0 || (ds_f32 as f64) < ct {
                s += ds_f32 as f64;
                buf.guess.swap(i, j);
            }
        }
        let cr = 1.0_f64 - s * inv_denom;
        final_r = cr;
        // 3. 采样 + 在线统计
        if t % stride == 0 {
            buf.r_sample.push(cr as f32);
        }
        if t > 0 {
            let dv = cr - prev_r;
            drn += 1;
            dr_s1 += dv;
            dr_s2 += dv * dv;
            dr_s3 += dv * dv * dv;
            dr_s4 += dv * dv * dv * dv;
            if (cr as f32) < (prev_r as f32) {
                decl += 1;
                if t >= s1e && t < s2e {
                    s2d += 1;
                }
            }
            if t < s1e {
                s1n += 1;
                s1s += dv;
                s1sq += dv * dv;
            } else if t < s2e {
                s2n += 1;
            } else {
                s3n += 1;
                s3s += dv;
                s3sq += dv * dv;
            }
        }
        if a1 == usize::MAX && cr >= hl {
            a1 = t;
        }
        if a2 == usize::MAX && cr >= 0.80 {
            a2 = t;
        }
        if a3 == usize::MAX && cr >= 0.90 {
            a3 = t;
        }
        inertia += (1.0_f32 - cr as f32) as f64;
        if cr >= rmr {
            rmr = cr;
            ps = t;
        } else {
            let uw = t - ps;
            if uw > mr {
                mr = uw;
            }
        }
        let dd = rmr - cr;
        if dd > mdd {
            mdd = dd;
        }
        if t + 1 == s2e || (t + 1 == m_max && r_s2.is_nan()) {
            r_s2 = cr;
        }
        prev_r = final_r;
        if s <= s_tol {
            break;
        }
        t += 1;
    }
    while buf.r_sample.len() < 4 {
        buf.r_sample.push(final_r as f32);
    }
    // 组装因子
    let mut f = [f32::NAN; N_FACTORS];
    f[0] = if a1 != usize::MAX {
        a1 as f32
    } else {
        m_max as f32
    };
    f[1] = if a2 != usize::MAX {
        a2 as f32
    } else {
        m_max as f32
    };
    f[2] = if a3 != usize::MAX {
        a3 as f32
    } else {
        m_max as f32
    };
    f[3] = final_r as f32;
    f[4] = inertia as f32;
    f[5] = decl as f32;
    f[6] = mdd as f32;
    f[7] = mr as f32;
    if drn >= 2 {
        let nd = drn as f64;
        f[8] = ((dr_s2 - dr_s1 * dr_s1 / nd) / (nd - 1.0)).max(0.0).sqrt() as f32;
    }
    if s1n >= 2 {
        let nd = s1n as f64;
        f[9] = ((s1sq - s1s * s1s / nd) / (nd - 1.0)).max(0.0).sqrt() as f32;
    }
    if s2n > 0 {
        f[10] = s2d as f32 / s2n as f32;
    }
    if !r_s2.is_nan() {
        f[11] = (final_r - r_s2) as f32;
    }
    if s3n >= 2 {
        let nd = s3n as f64;
        let st3 = ((s3sq - s3s * s3s / nd) / (nd - 1.0)).max(0.0).sqrt();
        if st3 > 1e-10 && s1n >= 2 {
            let nd1 = s1n as f64;
            f[12] = (((s1sq - s1s * s1s / nd1) / (nd1 - 1.0)).max(0.0).sqrt() / st3) as f32;
        }
    }
    if buf.r_sample.len() >= 4 {
        let drs: Vec<f32> = (1..buf.r_sample.len())
            .map(|t| buf.r_sample[t] - buf.r_sample[t - 1])
            .collect();
        if !drs.is_empty() {
            let p90 = percentile_abs(&drs, 0.90);
            let bin: Vec<f32> = drs
                .iter()
                .map(|&v| if v.abs() >= p90 { 1.0 } else { 0.0 })
                .collect();
            f[13] = runs_test_z(&bin);
        }
    }
    if drn >= 4 {
        let nd = drn as f64;
        let mean = dr_s1 / nd;
        let m2 = dr_s2 / nd - mean * mean;
        let m3 = dr_s3 / nd - 3.0 * mean * (dr_s2 / nd) + 2.0 * mean * mean * mean;
        let m4 = dr_s4 / nd - 4.0 * mean * (dr_s3 / nd) + 6.0 * mean * mean * (dr_s2 / nd)
            - 3.0 * mean.powi(4);
        if m2 > 1e-20 {
            let g1 = m3 / m2.powf(1.5);
            f[14] = (g1 * ((nd - 1.0) * nd).sqrt() / (nd - 2.0)) as f32;
            let g2 = m4 / (m2 * m2);
            f[15] = (((nd - 1.0) / ((nd - 2.0) * (nd - 3.0)))
                * ((nd + 1.0) * g2 - 3.0 * (nd - 1.0))) as f32;
        }
    }
    f[16] = hurst_rs(&buf.r_sample);
    f[17] = dfa_alpha(&buf.r_sample);
    let k = buf.d_vals.len();
    if k > 0 {
        f[18] = buf.d_vals.iter().map(|d| d.abs()).sum::<f32>() / k as f32;
        f[19] = buf.d_vals.iter().filter(|d| **d > 0.0).count() as f32 / k as f32;
        f[20] = std_ddof1(&buf.d_vals);
        if k >= 2 {
            f[21] = corr(&buf.d_vals[..k - 1], &buf.d_vals[1..]);
        }
        if k >= 2 {
            let flips = (0..k - 1)
                .filter(|&i| buf.d_vals[i].signum() != buf.d_vals[i + 1].signum())
                .count();
            f[22] = flips as f32 / (k - 1) as f32;
        }
        let p95 = percentile_abs(&buf.d_vals, 0.95);
        let hidden = (0..k)
            .filter(|&i| buf.g_below[i] == 1 && buf.d_vals[i].abs() > p95)
            .count();
        f[23] = hidden as f32 / k as f32;
        if k >= 3 {
            let ad: Vec<f32> = buf.d_vals.iter().map(|d| d.abs()).collect();
            f[24] = linear_slope(&buf.d_tau, &ad);
        }
    }
    f
}

// ============================================================================
// 主计算入口
// ============================================================================

/// 核心：pipeline 和 Python 的唯一共同调用点。
pub fn compute_anneal_volume_full(code: &str, date: i64) -> std::io::Result<Vec<f32>> {
    use std::time::Instant;
    let t_total = Instant::now();

    let t_read = Instant::now();
    let trades = read_trade_fast_inner(code, date, false, true, usize::MAX)?;
    let t_read_elapsed = t_read.elapsed();
    eprintln!("[prof] {} {} read_data: {:?}  n_trades={}", code, date, t_read_elapsed, trades.len());

    if trades.is_empty() {
        return Ok(vec![f32::NAN; EXPECTED_LEN]);
    }

    let t_open = trades.first().unwrap().time_sec;

    // 6 个宏观窗口聚合（缓存）
    let t_win = Instant::now();
    let win_aggs: Vec<(FxHashMap<i64, AggOrder>, FxHashMap<i64, AggOrder>)> = WINDOW_BOUNDS
        .iter()
        .map(|&(sec_lo, sec_hi)| {
            let lo_time = t_open + sec_lo;
            let hi_time = t_open + sec_hi;
            let lo = trades.partition_point(|t| t.time_sec < lo_time);
            let hi = trades.partition_point(|t| t.time_sec < hi_time);
            aggregate_orders(&trades[lo..hi])
        })
        .collect();
    eprintln!("[prof] {} {} 6_window_agg: {:?}", code, date, t_win.elapsed());

    // 65 个标量片段
    let t_scalar = Instant::now();
    let segs = segment_defs();
    let mut buf = AnnealBuf::new();
    let mut out: Vec<f32> = Vec::with_capacity(EXPECTED_LEN);
    let cache_len = WINDOW_BOUNDS.len() * 9;
    let mut extracted: Vec<Option<Vec<f32>>> = (0..cache_len).map(|_| None).collect();
    let mut quantile_orders: Vec<Option<Vec<(f32, usize)>>> =
        (0..cache_len).map(|_| None).collect();

    for &(win_idx, side, quantile) in &segs {
        let cache_idx = win_idx * 9 + side.index();
        if extracted[cache_idx].is_none() {
            let (bid_map, ask_map) = &win_aggs[win_idx];
            extracted[cache_idx] = Some(extract_side(bid_map, ask_map, side));
        }
        let vols = extracted[cache_idx].as_ref().unwrap();
        let filtered = if quantile == Quantile::All {
            vols.clone()
        } else {
            if quantile_orders[cache_idx].is_none() {
                let mut order: Vec<(f32, usize)> =
                    vols.iter().enumerate().map(|(i, &v)| (v, i)).collect();
                order.sort_by(|a, b| {
                    b.0.partial_cmp(&a.0)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                quantile_orders[cache_idx] = Some(order);
            }
            quantile_filter_from_sorted(
                vols,
                quantile_orders[cache_idx].as_ref().unwrap(),
                quantile,
            )
        };
        let m_adapt = adaptive_m_max(filtered.len(), M_MAX_SCALAR);
        let factors = anneal(&filtered, m_adapt, &mut buf);
        out.extend_from_slice(&factors);
    }
    eprintln!("[prof] {} {} 65_scalar: {:?}", code, date, t_scalar.elapsed());

    // 逐分钟矩阵 237 × 75
    let t_minute = Instant::now();
    let minute_col_names = build_minute_col_names();
    let mut matrix = Array2::zeros((N_MINUTES, N_MINUTE_COLS));

    let mut n_nonempty = 0usize;
    for m_idx in 0..N_MINUTES {
        let lo_time = t_open + (m_idx as f32) * 60.0;
        let hi_time = t_open + ((m_idx + 1) as f32) * 60.0;
        let lo = trades.partition_point(|t| t.time_sec < lo_time);
        let hi = trades.partition_point(|t| t.time_sec < hi_time);

        if lo >= hi {
            continue;
        }
        n_nonempty += 1;

        let (bid_map, ask_map) = aggregate_orders(&trades[lo..hi]);

        // 3 个版本
        let versions = [Side::Bid, Side::Ask, Side::Mixed];
        for (vi, &ver) in versions.iter().enumerate() {
            let vols = extract_side(&bid_map, &ask_map, ver);
            let m_adapt = adaptive_m_max(vols.len(), M_MAX_MINUTE);
            let factors = anneal(&vols, m_adapt, &mut buf);
            for (fi, &val) in factors.iter().enumerate() {
                matrix[[m_idx, vi * N_FACTORS + fi]] = val;
            }
        }
    }
    eprintln!("[prof] {} {} 237_minute: {:?}  n_nonempty={}", code, date, t_minute.elapsed(), n_nonempty);

    // 降维
    let t_reduce = Instant::now();
    let (reduced_vals, _) =
        features::get_features_factors_rust_full(&matrix.view(), &minute_col_names, false);
    eprintln!("[prof] {} {} dim_reduce: {:?}", code, date, t_reduce.elapsed());
    out.extend_from_slice(&reduced_vals);

    // 长度校准
    if out.len() < EXPECTED_LEN {
        out.resize(EXPECTED_LEN, f32::NAN);
    } else if out.len() > EXPECTED_LEN {
        out.truncate(EXPECTED_LEN);
    }

    eprintln!("[prof] {} {} TOTAL: {:?}", code, date, t_total.elapsed());

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
