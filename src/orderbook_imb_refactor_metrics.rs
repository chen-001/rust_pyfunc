//! 盘口挂单量 imb 改造因子（逐系列纯 Rust 复刻 + 8 段组合 + 串联汇总）。
//!
//! 架构（参考 rust-pipeline-factor-pattern skill）：
//!   - 公共基础：imb/rate 序列构造、8 段组合、统计辅助、幂迭代特征值
//!   - 每个系列一个 compute_hmXX_full(code, date) -> Vec<f32>：复刻原版受挂单量影响的因子逻辑
//!   - compute_orderbook_imb_refactor_full：串联所有系列，拼接输出
//!   - PyO3 包装 + pipeline 包装（factor_pipeline.rs）
//!
//! 每个系列的输出结构（要求 7）：
//!   对 rate2(÷十档和) 口径：[原版A, 改造B2, A−B2, |A−B2|]
//!   对 rate1(÷total全量) 口径：[原版A, 改造B1, A−B1, |A−B1|]
//!   两种口径各算一遍 → 8 段拼接（原版 A 在两组各出现一次）。
//!
//! 已实现系列：
//!   - hm32 量价的适中变化：haha 差值矩阵最大特征值 + 特征向量 11 统计 × 5 时段

use crate::fast_csv_reader::{
    read_market_fast_inner, read_trade_fast_inner, MarketRecord, TradeRecord,
};
use pyo3::prelude::*;

// ============================================================================
// 时段定义（with_afternoon_adjust=false，time_sec 为真实时钟）
// 09:30=34200 10:00=36000 11:30=41400 13:00=46800 14:30=52200 14:57=53820
// ============================================================================

const T_OPEN: f32 = 34200.0;
const T_OPEN30_END: f32 = 36000.0;
const T_MORN_END: f32 = 41400.0;
const T_AFTN_START: f32 = 46800.0;
const T_CLOSE30_START: f32 = 52200.0;
const T_END: f32 = 53820.0;

#[inline]
fn tod(t: f32) -> f32 {
    t.rem_euclid(86400.0)
}

/// 时段判定：t∈[t0,t1] 且排除午休 (11:30,13:00)。
#[inline]
fn in_segment(t: f32, t0: f32, t1: f32) -> bool {
    t >= t0 && t <= t1 && !(t > T_MORN_END && t < T_AFTN_START)
}

const SEGMENTS: [(&str, f32, f32); 5] = [
    ("allday", T_OPEN, T_END),
    ("open30", T_OPEN, T_OPEN30_END),
    ("morning", T_OPEN, T_MORN_END),
    ("afternoon", T_AFTN_START, T_END),
    ("close30", T_CLOSE30_START, T_END),
];

// ============================================================================
// 统计辅助（纯 f32，NaN/inf 显式过滤）
// ============================================================================

fn filter_valid(v: &[f32]) -> Vec<f32> {
    v.iter().copied().filter(|x| x.is_finite()).collect()
}

fn mean_v(v: &[f32]) -> f32 {
    if v.is_empty() {
        f32::NAN
    } else {
        v.iter().sum::<f32>() / v.len() as f32
    }
}

fn std_v(v: &[f32]) -> f32 {
    let n = v.len();
    if n < 2 {
        return f32::NAN;
    }
    let m = mean_v(v);
    let sq: f32 = v.iter().map(|x| (x - m).powi(2)).sum();
    (sq / (n - 1) as f32).sqrt()
}

fn skew_v(v: &[f32]) -> f32 {
    let n = v.len();
    if n < 3 {
        return f32::NAN;
    }
    let m = mean_v(v);
    let m2: f32 = v.iter().map(|x| (x - m).powi(2)).sum::<f32>() / n as f32;
    let m3: f32 = v.iter().map(|x| (x - m).powi(3)).sum::<f32>() / n as f32;
    if m2 <= 0.0 {
        return f32::NAN;
    }
    let g1 = m3 / m2.powf(1.5);
    let nf = n as f32;
    let adj = g1 * ((nf - 1.0).powf(1.5)) / ((nf - 2.0) * nf.sqrt());
    if adj.is_finite() {
        adj
    } else {
        f32::NAN
    }
}

fn kurt_v(v: &[f32]) -> f32 {
    // 超额峰度（pandas kurtosis，Fisher）
    let n = v.len();
    if n < 4 {
        return f32::NAN;
    }
    let m = mean_v(v);
    let nf = n as f32;
    let s2: f32 = v.iter().map(|x| (x - m).powi(2)).sum::<f32>();
    let s4: f32 = v.iter().map(|x| (x - m).powi(4)).sum::<f32>();
    let denom = (nf - 1.0) * (nf - 2.0) * (nf - 3.0);
    if denom == 0.0 || s2 == 0.0 {
        return f32::NAN;
    }
    let k = nf * ((nf + 1.0) * s4 - 3.0 * (nf - 1.0).powi(2) * s2 * s2 / (nf * nf))
        / (denom * (s2 / nf).powi(2));
    if k.is_finite() {
        k
    } else {
        f32::NAN
    }
}

fn autocorr_v(v: &[f32], lag: usize) -> f32 {
    let n = v.len();
    if lag == 0 || n <= lag {
        return f32::NAN;
    }
    let a = &v[..n - lag];
    let b = &v[lag..];
    corr_v(a, b)
}

fn trend_v(v: &[f32]) -> f32 {
    let n = v.len();
    if n < 2 {
        return f32::NAN;
    }
    let xs: Vec<f32> = (1..=n).map(|x| x as f32).collect();
    corr_v(&xs, v)
}

fn quantile_sorted(sorted: &[f32], q: f32) -> f32 {
    let n = sorted.len();
    if n == 0 {
        return f32::NAN;
    }
    let pos = q * (n - 1) as f32;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    if lo == hi {
        sorted[lo]
    } else {
        sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo as f32)
    }
}

fn corr_v(a: &[f32], b: &[f32]) -> f32 {
    // 逐位配对，要求两侧都有限
    let mut pa = Vec::with_capacity(a.len());
    let mut pb = Vec::with_capacity(a.len());
    let n = a.len().min(b.len());
    for k in 0..n {
        if a[k].is_finite() && b[k].is_finite() {
            pa.push(a[k]);
            pb.push(b[k]);
        }
    }
    let m = pa.len();
    if m < 2 {
        return f32::NAN;
    }
    let ma = pa.iter().sum::<f32>() / m as f32;
    let mb = pb.iter().sum::<f32>() / m as f32;
    let (mut cov, mut va, mut vb) = (0.0f32, 0.0f32, 0.0f32);
    for k in 0..m {
        let da = pa[k] - ma;
        let db = pb[k] - mb;
        cov += da * db;
        va += da * da;
        vb += db * db;
    }
    let den = (va * vb).sqrt();
    if den > 0.0 {
        cov / den
    } else {
        f32::NAN
    }
}

// ============================================================================
// imb / rate 序列构造
// ============================================================================

/// 单侧 rate 序列：对给定快照子集 idx，取 side(ask/bid) 第 level 档，按 kind 归一。
/// kind: "orig"=绝对量, "r1"=÷total_{side}_vol, "r2"=÷Σ{side}_vol1..10
fn rate_seq(
    market: &[MarketRecord],
    idx: &[usize],
    side: &str,
    level: usize,
    kind: &str,
) -> Vec<f32> {
    let li = level - 1;
    idx.iter()
        .map(|&i| {
            let m = &market[i];
            let raw = if side == "ask" {
                m.ask_vols[li]
            } else {
                m.bid_vols[li]
            };
            match kind {
                "orig" => raw,
                "r1" => {
                    let tot = if side == "ask" {
                        m.total_ask_vol
                    } else {
                        m.total_bid_vol
                    };
                    if tot > 0.0 {
                        raw / tot
                    } else {
                        f32::NAN
                    }
                }
                _ => {
                    // r2
                    let s10: f32 = if side == "ask" {
                        m.ask_vols.iter().sum()
                    } else {
                        m.bid_vols.iter().sum()
                    };
                    if s10 > 0.0 {
                        raw / s10
                    } else {
                        f32::NAN
                    }
                }
            }
        })
        .collect()
}

/// 联合 imb 序列：direction*(Σ{side}_k 归一差)。direction=+1 → ask−bid；−1 → bid−ask。
/// kind: "orig"=(Σa−Σb)/(Σa+Σb), "r1"=Σa/total_ask−Σb/total_bid, "r2"=Σa/Σa10−Σb/Σb10
fn imb_seq(
    market: &[MarketRecord],
    idx: &[usize],
    level: usize,
    kind: &str,
    direction: f32,
) -> Vec<f32> {
    idx.iter()
        .map(|&i| {
            let m = &market[i];
            let bid_k: f32 = m.bid_vols[..level].iter().sum();
            let ask_k: f32 = m.ask_vols[..level].iter().sum();
            let v = match kind {
                "orig" => {
                    if ask_k + bid_k > 0.0 {
                        (ask_k - bid_k) / (ask_k + bid_k)
                    } else {
                        f32::NAN
                    }
                }
                "r1" => {
                    if m.total_ask_vol > 0.0 && m.total_bid_vol > 0.0 {
                        ask_k / m.total_ask_vol - bid_k / m.total_bid_vol
                    } else {
                        f32::NAN
                    }
                }
                _ => {
                    let a10: f32 = m.ask_vols.iter().sum();
                    let b10: f32 = m.bid_vols.iter().sum();
                    if a10 > 0.0 && b10 > 0.0 {
                        ask_k / a10 - bid_k / b10
                    } else {
                        f32::NAN
                    }
                }
            };
            direction * v
        })
        .collect()
}

// ============================================================================
// 8 段组合工具
// ============================================================================

/// 把三套等长因子值（orig/r1/r2）组合成 8 段输出：
///   r2 组：[orig, r2, orig−r2, |orig−r2|]
///   r1 组：[orig, r1, orig−r1, |orig−r1|]
fn combine_8panels(orig: &[f32], r1: &[f32], r2: &[f32]) -> Vec<f32> {
    let n = orig.len();
    debug_assert_eq!(r1.len(), n);
    debug_assert_eq!(r2.len(), n);
    let mut out = Vec::with_capacity(n * 8);
    out.extend_from_slice(orig);
    out.extend_from_slice(r2);
    for i in 0..n {
        out.push(orig[i] - r2[i]);
    }
    for i in 0..n {
        out.push((orig[i] - r2[i]).abs());
    }
    out.extend_from_slice(orig);
    out.extend_from_slice(r1);
    for i in 0..n {
        out.push(orig[i] - r1[i]);
    }
    for i in 0..n {
        out.push((orig[i] - r1[i]).abs());
    }
    out
}

/// 8 段名字后缀（与 combine_8panels 输出顺序严格一致）。
const PANEL_SUFFIX: [&str; 8] = [
    "r2_orig",
    "r2_rate",
    "r2_diff",
    "r2_absdiff",
    "r1_orig",
    "r1_rate",
    "r1_diff",
    "r1_absdiff",
];

// ============================================================================
// 线性代数辅助：幂迭代求对称矩阵最大特征值 + 特征向量
// ============================================================================

fn max_eig_with_vec(mat: &[f32], n: usize, iters: usize) -> (f32, Vec<f32>) {
    if n == 0 {
        return (f32::NAN, vec![]);
    }
    let mut v = vec![1.0f32 / (n as f32).sqrt(); n];
    let mut eig = f32::NAN;
    for _ in 0..iters {
        let mut mv = vec![0.0f32; n];
        for i in 0..n {
            let row = &mat[i * n..(i + 1) * n];
            let mut s = 0.0f32;
            for j in 0..n {
                s += row[j] * v[j];
            }
            mv[i] = s;
        }
        let nrm = mv.iter().map(|x| x * x).sum::<f32>().sqrt();
        if !nrm.is_finite() || nrm == 0.0 {
            return (f32::NAN, v);
        }
        for i in 0..n {
            v[i] = mv[i] / nrm;
        }
        eig = nrm;
    }
    (eig, v)
}

/// 两两绝对差值矩阵 |x_i − x_j|，n×n 行优先。
fn diff_matrix(x: &[f32]) -> Vec<f32> {
    let n = x.len();
    let mut m = vec![0.0f32; n * n];
    for i in 0..n {
        for j in 0..n {
            m[i * n + j] = (x[i] - x[j]).abs();
        }
    }
    m
}

// ============================================================================
// hm32：量价的适中变化 —— haha 差值矩阵最大特征值 + 特征向量 11 统计
// 原版 res4 = haha(ask_vol1, bid_vol1)：把 bid 差值矩阵的下三角(含对角)混入 ask 差值矩阵，
// 算混合矩阵最大特征值 + 对应特征向量的 11 个统计量。受挂单量影响 = res4。
// ============================================================================

const HM32_STATS: [&str; 11] = [
    "eigenvalue",
    "vec_mean",
    "vec_std",
    "vec_skew",
    "vec_kurt",
    "vec_max",
    "vec_min",
    "vec_median",
    "vec_q1",
    "vec_q3",
    "vec_iqr",
];

fn haha_features(a: &[f32], b: &[f32], n: usize, iters: usize) -> [f32; 11] {
    let mut out = [f32::NAN; 11];
    if n < 2 {
        return out;
    }
    // b 的下三角(含对角)混入 a
    let mut mixed = a.to_vec();
    for i in 0..n {
        for j in 0..=i {
            mixed[i * n + j] = b[i * n + j];
        }
    }
    let (eig, vec) = max_eig_with_vec(&mixed, n, iters);
    out[0] = eig;
    let valid = filter_valid(&vec);
    let nv = valid.len();
    if nv == 0 {
        return out;
    }
    out[1] = mean_v(&valid);
    if nv >= 2 {
        out[2] = std_v(&valid);
        out[5] = valid.iter().copied().fold(f32::INFINITY, f32::min);
        out[6] = valid.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut s = valid.clone();
        s.sort_unstable_by(|x, y| x.partial_cmp(y).unwrap());
        out[7] = quantile_sorted(&s, 0.5);
        out[8] = quantile_sorted(&s, 0.25);
        out[9] = quantile_sorted(&s, 0.75);
        out[10] = out[9] - out[8];
    }
    if nv >= 3 {
        out[3] = skew_v(&valid);
    }
    if nv >= 4 {
        out[4] = kurt_v(&valid);
    }
    out
}

/// 原地 z-score 标准化（mean0/std1），用于 norm 口径让特征值与 rate 同量级。
fn zscore_inplace(v: &mut [f32]) {
    let valid: Vec<f32> = v.iter().copied().filter(|x| x.is_finite()).collect();
    if valid.len() < 2 {
        return;
    }
    let m = mean_v(&valid);
    let s = std_v(&valid);
    if s > 0.0 && s.is_finite() {
        for x in v.iter_mut() {
            if x.is_finite() {
                *x = (*x - m) / s;
            }
        }
    }
}

/// 单时段单口径的 hm32 原始因子（11 个）：haha(ask_seq, bid_seq)。
/// kind: "raw"=绝对量, "norm"=绝对量z-score归一化, "r1"=÷total, "r2"=÷十档和
/// 关键：rate 口径在 total=0/十档全0 处产生 NaN，差值矩阵两两组合会让单个 NaN 污染
/// 整行整列 → 特征值 NaN。故先配对剔除 NaN 快照，再采样、算 diff_matrix。
fn hm32_one_panel(market: &[MarketRecord], idx: &[usize], kind: &str) -> [f32; 11] {
    const MAXN: usize = 200;
    // raw/norm 基于 orig 序列(绝对量)；r1/r2 用对应归一
    let base = if kind == "r1" || kind == "r2" {
        kind
    } else {
        "orig"
    };
    let ask_all = rate_seq(market, idx, "ask", 1, base);
    let bid_all = rate_seq(market, idx, "bid", 1, base);
    // 配对剔除 NaN
    let mut ask_v: Vec<f32> = Vec::with_capacity(idx.len());
    let mut bid_v: Vec<f32> = Vec::with_capacity(idx.len());
    for k in 0..idx.len() {
        if ask_all[k].is_finite() && bid_all[k].is_finite() {
            ask_v.push(ask_all[k]);
            bid_v.push(bid_all[k]);
        }
    }
    let n = ask_v.len();
    if n < 2 {
        return [f32::NAN; 11];
    }
    // 等距采样到 MAXN
    let (mut sa, mut sb) = if n <= MAXN {
        (ask_v, bid_v)
    } else {
        let mut a = Vec::with_capacity(MAXN);
        let mut b = Vec::with_capacity(MAXN);
        for k in 0..MAXN {
            let i = k * (n - 1) / (MAXN - 1);
            a.push(ask_v[i]);
            b.push(bid_v[i]);
        }
        (a, b)
    };
    if kind == "norm" {
        zscore_inplace(&mut sa);
        zscore_inplace(&mut sb);
    }
    let m = sa.len();
    let ma = diff_matrix(&sa);
    let mb = diff_matrix(&sb);
    haha_features(&ma, &mb, m, 60)
}

const HM32_LEN: usize = HM32_STATS.len() * SEGMENTS.len() * PANEL_SUFFIX.len() * 2; // 11*5*8*2=880 (raw块+norm块)

fn hm32_names() -> Vec<String> {
    let mut names = Vec::with_capacity(HM32_LEN);
    // 顺序与 compute_hm32_full 输出对齐：raw块(8段) 后接 norm块(8段)，每段内 panel×seg×stat
    for &blk in ["raw", "norm"].iter() {
        for &panel in PANEL_SUFFIX.iter() {
            for &(seg, _, _) in SEGMENTS.iter() {
                for &stat in HM32_STATS.iter() {
                    names.push(format!("hm32_{}_{}_{}_{}", blk, stat, seg, panel));
                }
            }
        }
    }
    names
}

pub fn compute_hm32_full(code: &str, date: i64) -> std::io::Result<Vec<f32>> {
    let market = read_market_fast_inner(code, date, false, false, usize::MAX)?;
    // 全天盘内快照索引
    let all_idx: Vec<usize> = market
        .iter()
        .enumerate()
        .filter(|(_, m)| m.last_prc != 0.0 && in_segment(tod(m.time_sec), T_OPEN, T_END))
        .map(|(i, _)| i)
        .collect();
    let mut out = Vec::with_capacity(HM32_LEN);
    if all_idx.is_empty() {
        out.resize(HM32_LEN, f32::NAN);
        return Ok(out);
    }
    // 四套口径：raw(绝对量)、norm(绝对量z-score)、r1(÷total)、r2(÷十档和)，各 11 stat × 5 seg = 55
    let cap = HM32_STATS.len() * SEGMENTS.len();
    let mut raw55 = Vec::with_capacity(cap);
    let mut norm55 = Vec::with_capacity(cap);
    let mut r1_55 = Vec::with_capacity(cap);
    let mut r2_55 = Vec::with_capacity(cap);
    for &(seg, t0, t1) in SEGMENTS.iter() {
        let idx: Vec<usize> = all_idx
            .iter()
            .copied()
            .filter(|&i| in_segment(tod(market[i].time_sec), t0, t1))
            .collect();
        if idx.is_empty() {
            raw55.extend_from_slice(&[f32::NAN; 11]);
            norm55.extend_from_slice(&[f32::NAN; 11]);
            r1_55.extend_from_slice(&[f32::NAN; 11]);
            r2_55.extend_from_slice(&[f32::NAN; 11]);
        } else {
            raw55.extend_from_slice(&hm32_one_panel(&market, &idx, "raw"));
            norm55.extend_from_slice(&hm32_one_panel(&market, &idx, "norm"));
            r1_55.extend_from_slice(&hm32_one_panel(&market, &idx, "r1"));
            r2_55.extend_from_slice(&hm32_one_panel(&market, &idx, "r2"));
        }
    }
    // raw 块（绝对量原版）8 段 + norm 块（归一化原版）8 段
    out = combine_8panels(&raw55, &r1_55, &r2_55);
    out.extend(combine_8panels(&norm55, &r1_55, &r2_55));
    Ok(out)
}

// ============================================================================
// hm46：奇偶数偏好 —— ab_vol3_rate (3档双侧 imb) 的 10 统计 × 5 时段
// 原版 ab_vol3_rate = ask_vol3/(|ask_vol3|+|bid_vol3|)（已是归一化 imb，norm 类）
// 改造 r1/r2 用 ÷total / ÷十档和 口径。last_prc 分桶用 POINT1=0.2, POINT2=0.4。
// 注：简化 hm46 原版的 ask_vol2/3 diff 预处理，直接用原值（保证三套口径一致可比）。
// ============================================================================

const HM46_STATS: [&str; 10] = [
    "mean", "std", "skew", "max", "min", "corr_prc", "small1", "small2", "large1", "large2",
];
const HM46_LEN: usize = HM46_STATS.len() * SEGMENTS.len() * PANEL_SUFFIX.len(); // 10*5*8=400
const HM46_POINT1: f32 = 0.2;
const HM46_POINT2: f32 = 0.4;

/// 3 档双侧 imb 序列。kind: "orig"=ab_vol3_rate原版, "r1"=÷total, "r2"=÷十档和
fn hm46_seq(market: &[MarketRecord], idx: &[usize], kind: &str) -> Vec<f32> {
    idx.iter()
        .map(|&i| {
            let m = &market[i];
            let a3 = m.ask_vols[2];
            let b3 = m.bid_vols[2];
            match kind {
                "orig" => {
                    let denom = a3.abs() + b3.abs();
                    if denom > 0.0 {
                        a3 / denom
                    } else {
                        f32::NAN
                    }
                }
                "r1" => {
                    if m.total_ask_vol > 0.0 && m.total_bid_vol > 0.0 {
                        a3 / m.total_ask_vol - b3 / m.total_bid_vol
                    } else {
                        f32::NAN
                    }
                }
                _ => {
                    let a10: f32 = m.ask_vols.iter().sum();
                    let b10: f32 = m.bid_vols.iter().sum();
                    if a10 > 0.0 && b10 > 0.0 {
                        a3 / a10 - b3 / b10
                    } else {
                        f32::NAN
                    }
                }
            }
        })
        .collect()
}

/// last_prc 分桶均值：对满足 pred(prc) 的快照取 seq 均值（配对过滤 NaN）。
fn bucket_mean(seq: &[f32], prc: &[f32], pred: impl Fn(f32) -> bool) -> f32 {
    let n = seq.len().min(prc.len());
    let (mut sum, mut cnt) = (0.0f32, 0u32);
    for k in 0..n {
        if seq[k].is_finite() && prc[k].is_finite() && pred(prc[k]) {
            sum += seq[k];
            cnt += 1;
        }
    }
    if cnt > 0 {
        sum / cnt as f32
    } else {
        f32::NAN
    }
}

/// 单时段单口径 hm46 因子（10 个）。
fn hm46_one_panel(market: &[MarketRecord], idx: &[usize], kind: &str) -> [f32; 10] {
    let mut out = [f32::NAN; 10];
    let seq = hm46_seq(market, idx, kind);
    let prc: Vec<f32> = idx.iter().map(|&i| market[i].last_prc).collect();
    let valid = filter_valid(&seq);
    if valid.is_empty() {
        return out;
    }
    out[0] = mean_v(&valid);
    if valid.len() >= 2 {
        out[1] = std_v(&valid);
        out[3] = valid.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        out[4] = valid.iter().copied().fold(f32::INFINITY, f32::min);
    }
    if valid.len() >= 3 {
        out[2] = skew_v(&valid);
    }
    out[5] = corr_v(&seq, &prc); // corr_prc
                                 // last_prc 分桶（POINT1=0.2, POINT2=0.4）
    let mut prc_sorted: Vec<f32> = prc.iter().copied().filter(|x| x.is_finite()).collect();
    prc_sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    let q_small1 = quantile_sorted(&prc_sorted, HM46_POINT1);
    let q_small2 = quantile_sorted(&prc_sorted, HM46_POINT2);
    let q_large1 = quantile_sorted(&prc_sorted, 1.0 - HM46_POINT1);
    let q_large2 = quantile_sorted(&prc_sorted, 1.0 - HM46_POINT2);
    out[6] = bucket_mean(&seq, &prc, |p| p < q_small1);
    out[7] = bucket_mean(&seq, &prc, |p| p < q_small2);
    out[8] = bucket_mean(&seq, &prc, |p| p > q_large1);
    out[9] = bucket_mean(&seq, &prc, |p| p > q_large2);
    out
}

fn hm46_names() -> Vec<String> {
    let mut names = Vec::with_capacity(HM46_LEN);
    for &panel in PANEL_SUFFIX.iter() {
        for &(seg, _, _) in SEGMENTS.iter() {
            for &stat in HM46_STATS.iter() {
                names.push(format!("hm46_{}_{}_{}", stat, seg, panel));
            }
        }
    }
    names
}

pub fn compute_hm46_full(code: &str, date: i64) -> std::io::Result<Vec<f32>> {
    let market = read_market_fast_inner(code, date, false, false, usize::MAX)?;
    let all_idx: Vec<usize> = market
        .iter()
        .enumerate()
        .filter(|(_, m)| m.last_prc != 0.0 && in_segment(tod(m.time_sec), T_OPEN, T_END))
        .map(|(i, _)| i)
        .collect();
    let cap = HM46_STATS.len() * SEGMENTS.len();
    let mut orig50 = Vec::with_capacity(cap);
    let mut r1_50 = Vec::with_capacity(cap);
    let mut r2_50 = Vec::with_capacity(cap);
    for &(seg, t0, t1) in SEGMENTS.iter() {
        let idx: Vec<usize> = all_idx
            .iter()
            .copied()
            .filter(|&i| in_segment(tod(market[i].time_sec), t0, t1))
            .collect();
        if idx.is_empty() {
            orig50.extend_from_slice(&[f32::NAN; 10]);
            r1_50.extend_from_slice(&[f32::NAN; 10]);
            r2_50.extend_from_slice(&[f32::NAN; 10]);
        } else {
            orig50.extend_from_slice(&hm46_one_panel(&market, &idx, "orig"));
            r1_50.extend_from_slice(&hm46_one_panel(&market, &idx, "r1"));
            r2_50.extend_from_slice(&hm46_one_panel(&market, &idx, "r2"));
        }
    }
    Ok(combine_8panels(&orig50, &r1_50, &r2_50))
}

// ============================================================================
// hm72：增量分解 LZC —— abi1/abi3 (1/3档双侧 imb) 的 LZ76 复杂度 8 字段
// 原版 lzc_abi = lz_complexity_detailed(abi, [0.33,0.66])，返回 8 字段：
//   length_mean/std/skew/kurt/max/autocorr/index_corr + lz_complexity
// 自实现 simple LZ76（值与原版 suffix_automaton 一致，仅性能差）+ discretize 分桶。
// abitotal 跳过：total/total=1 → r1 退化 0，r2 无十档概念。
// ============================================================================

const HM72_STATS: [&str; 8] = [
    "length_mean",
    "length_std",
    "length_skew",
    "length_kurt",
    "length_max",
    "length_autocorr",
    "length_index_corr",
    "lz_complexity",
];
const HM72_LEN: usize = 2 * HM72_STATS.len() * PANEL_SUFFIX.len(); // 2档×8字段×8段=128

/// 按 quantiles 分位把序列分桶为符号（与 lz_complexity_detailed 的 discretize 一致）。
fn discretize_seq(seq: &[f32], quantiles: &[f64]) -> Vec<u8> {
    if seq.is_empty() {
        return vec![];
    }
    let mut sorted = seq.to_vec();
    sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    let thresholds: Vec<f32> = quantiles
        .iter()
        .map(|&q| quantile_sorted(&sorted, q as f32))
        .collect();
    seq.iter()
        .map(|&x| {
            let mut b = 0u8;
            for &t in &thresholds {
                if x > t {
                    b += 1;
                }
            }
            b
        })
        .collect()
}

/// LZ76 分解（simple O(n²)，值与 suffix_automaton 一致）。
/// 返回 (phrase 数, 各 phrase 长度, 各 phrase 起点位置)。
fn lz76_decompose(sym: &[u8]) -> (usize, Vec<usize>, Vec<usize>) {
    let n = sym.len();
    if n == 0 {
        return (0, vec![], vec![]);
    }
    let mut lengths = Vec::new();
    let mut positions = Vec::new();
    let mut i = 0usize;
    while i < n {
        let mut l = 1usize;
        loop {
            if i + l > n {
                break;
            }
            if l == 1 {
                // 扩展到 l=2 需 sym[i] 在历史 sym[..i) 中出现过
                if i > 0 && sym[..i].contains(&sym[i]) {
                    l += 1;
                } else {
                    break;
                }
                continue;
            }
            // l>=2: pattern sym[i..i+l-1]（长 l-1）需在历史 sym[..i+l-1) 出现
            let pat_len = l - 1;
            let hist_end = i + l - 1;
            if pat_len > hist_end {
                break;
            }
            let pat = &sym[i..i + pat_len];
            let mut found = false;
            let mut s = 0usize;
            // s < i 排除 pat 自身位置的 self-match（否则 l 无限扩展→complexity 退化）；
            // 允许重叠匹配（s+pat_len 可 > i，延伸进 pat 区，LZ76 标准）。
            while s + pat_len <= hist_end && s < i {
                if &sym[s..s + pat_len] == pat {
                    found = true;
                    break;
                }
                s += 1;
            }
            if found {
                l += 1;
            } else {
                break;
            }
        }
        let pl = l.min(n - i).max(1);
        lengths.push(pl);
        positions.push(i);
        i += pl;
    }
    (lengths.len(), lengths, positions)
}

/// LZC detailed 8 字段（顺序与原版 to_dict 一致）。
fn lzc_detailed_8(seq: &[f32], quantiles: &[f64]) -> [f32; 8] {
    let mut out = [f32::NAN; 8];
    if seq.len() < 2 {
        return out;
    }
    let sym = discretize_seq(seq, quantiles);
    let (complexity, lengths, positions) = lz76_decompose(&sym);
    if complexity == 0 {
        return out;
    }
    let lf: Vec<f32> = lengths.iter().map(|&x| x as f32).collect();
    let nl = lf.len();
    out[7] = complexity as f32; // lz_complexity
    out[0] = mean_v(&lf); // length_mean
    if nl >= 2 {
        out[1] = std_v(&lf); // length_std
        out[4] = lf.iter().copied().fold(f32::NEG_INFINITY, f32::max); // length_max
    }
    if nl >= 3 {
        out[2] = skew_v(&lf); // length_skew
    }
    if nl >= 4 {
        out[3] = kurt_v(&lf); // length_kurt
    }
    out[5] = autocorr_v(&lf, 1); // length_autocorr
    let posf: Vec<f32> = positions.iter().map(|&x| x as f32).collect();
    out[6] = corr_v(&lf, &posf); // length_index_corr
    out
}

/// 等距采样到 maxn（LZ76 simple O(n²)，序列长时降采样保性能）。
fn sample_down(v: &[f32], maxn: usize) -> Vec<f32> {
    let n = v.len();
    if n <= maxn {
        return v.to_vec();
    }
    (0..maxn).map(|k| v[k * (n - 1) / (maxn - 1)]).collect()
}

fn hm72_names() -> Vec<String> {
    let mut names = Vec::with_capacity(HM72_LEN);
    // 顺序对齐：lv(2) × panel(8) × stat(8)
    for &lv in &[1usize, 3] {
        for &panel in PANEL_SUFFIX.iter() {
            for &stat in HM72_STATS.iter() {
                names.push(format!("hm72_abi{}_{}_{}", lv, stat, panel));
            }
        }
    }
    names
}

pub fn compute_hm72_full(code: &str, date: i64) -> std::io::Result<Vec<f32>> {
    let market = read_market_fast_inner(code, date, false, false, usize::MAX)?;
    let all_idx: Vec<usize> = market
        .iter()
        .enumerate()
        .filter(|(_, m)| m.last_prc != 0.0 && in_segment(tod(m.time_sec), T_OPEN, T_END))
        .map(|(i, _)| i)
        .collect();
    let mut out = Vec::with_capacity(HM72_LEN);
    if all_idx.is_empty() {
        out.resize(HM72_LEN, f32::NAN);
        return Ok(out);
    }
    let qs = [0.33f64, 0.66];
    for &lv in &[1usize, 3] {
        let orig_seq = filter_valid(&imb_seq(&market, &all_idx, lv, "orig", 1.0));
        let r1_seq = filter_valid(&imb_seq(&market, &all_idx, lv, "r1", 1.0));
        let r2_seq = filter_valid(&imb_seq(&market, &all_idx, lv, "r2", 1.0));
        let orig8 = lzc_detailed_8(&sample_down(&orig_seq, 2000), &qs).to_vec();
        let r1_8 = lzc_detailed_8(&sample_down(&r1_seq, 2000), &qs).to_vec();
        let r2_8 = lzc_detailed_8(&sample_down(&r2_seq, 2000), &qs).to_vec();
        out.extend(combine_8panels(&orig8, &r1_8, &r2_8));
    }
    Ok(out)
}

// ============================================================================
// hm21：阁中帝子今何在 —— 卖价下调/买价上调时刻的 ask_66/bid_83 统计差
// ask_66 = market_ask/(主动买量+market_ask)；bid_83 同理。受 ask_vol1/bid_vol1 影响。
// 原版 19 个 abs(ask统计 - bid统计)：corr1/2/3 + distance + label 的 10 统计 + corr1-3_label + distance_label
// raw(绝对量 market_ask) + norm(ask_66 z-score) 双块。
// ============================================================================

const HM21_STATS: [&str; 19] = [
    "corr1",
    "corr2",
    "corr3",
    "distance",
    "max_label",
    "min_label",
    "mean_label",
    "std_label",
    "sum_label",
    "skew_label",
    "kurt_label",
    "q75_label",
    "q25_label",
    "median_label",
    "iqr_label",
    "corr1_label",
    "corr2_label",
    "corr3_label",
    "distance_label",
];
const HM21_LEN: usize = HM21_STATS.len() * PANEL_SUFFIX.len() * 2; // 19*8*2=304

/// 单侧 1 档挂单量按 kind 归一。kind: "orig"=绝对量, "r1"=÷total, "r2"=÷十档和
fn vol_of(m: &MarketRecord, side: &str, kind: &str) -> f32 {
    let raw = if side == "ask" {
        m.ask_vols[0]
    } else {
        m.bid_vols[0]
    };
    match kind {
        "orig" => raw,
        "r1" => {
            let tot = if side == "ask" {
                m.total_ask_vol
            } else {
                m.total_bid_vol
            };
            if tot > 0.0 {
                raw / tot
            } else {
                f32::NAN
            }
        }
        _ => {
            let s10 = if side == "ask" {
                m.ask_vols.iter().sum::<f32>()
            } else {
                m.bid_vols.iter().sum::<f32>()
            };
            if s10 > 0.0 {
                raw / s10
            } else {
                f32::NAN
            }
        }
    }
}

fn zscore_vec(v: &[f32]) -> Vec<f32> {
    let mut out = v.to_vec();
    zscore_inplace(&mut out);
    out
}

/// hm21 的 19 统计（对 ask_66 或 bid_83 序列）。
fn hm21_stats_19(seq: &[f32], labels: &[usize], max_label: usize) -> [f32; 19] {
    let mut out = [f32::NAN; 19];
    let n = seq.len();
    if n < 4 {
        return out;
    }
    out[0] = autocorr_v(seq, 1);
    out[1] = autocorr_v(seq, 2);
    out[2] = autocorr_v(seq, 3);
    let q75i = (0.75 * n as f32) as usize;
    let q25i = (0.25 * n as f32) as usize;
    out[3] = (labels[q75i.min(n - 1)] - labels[q25i.min(n - 1)]) as f32;
    // 按 seq 值排序，label 跟着，归一化
    let mut pairs: Vec<(f32, f32)> = seq
        .iter()
        .zip(labels.iter())
        .filter(|(s, _)| s.is_finite())
        .map(|(&s, &l)| (s, l as f32 / max_label.max(1) as f32))
        .collect();
    pairs.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let sl: Vec<f32> = pairs.iter().map(|(_, l)| *l).collect();
    let m = sl.len();
    if m == 0 {
        return out;
    }
    out[4] = sl.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    out[5] = sl.iter().copied().fold(f32::INFINITY, f32::min);
    out[6] = mean_v(&sl);
    if m >= 2 {
        out[7] = std_v(&sl);
    }
    out[8] = sl.iter().sum::<f32>();
    if m >= 3 {
        out[9] = skew_v(&sl);
    }
    if m >= 4 {
        out[10] = kurt_v(&sl);
    }
    let mut srt = sl.clone();
    srt.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    out[11] = quantile_sorted(&srt, 0.75);
    out[12] = quantile_sorted(&srt, 0.25);
    out[13] = quantile_sorted(&srt, 0.5);
    out[14] = out[11] - out[12];
    out[15] = autocorr_v(&sl, 1);
    out[16] = autocorr_v(&sl, 2);
    out[17] = autocorr_v(&sl, 3);
    let d75 = (0.75 * m as f32) as usize;
    let d25 = (0.25 * m as f32) as usize;
    out[18] = sl[d75.min(m - 1)] - sl[d25.min(m - 1)];
    out
}

fn hm21_names() -> Vec<String> {
    let mut names = Vec::with_capacity(HM21_LEN);
    for &blk in ["raw", "norm"].iter() {
        for &panel in PANEL_SUFFIX.iter() {
            for &stat in HM21_STATS.iter() {
                names.push(format!("hm21_{}_{}_{}", blk, stat, panel));
            }
        }
    }
    names
}

pub fn compute_hm21_full(code: &str, date: i64) -> std::io::Result<Vec<f32>> {
    let market_all = read_market_fast_inner(code, date, false, false, usize::MAX)?;
    let trade_all = read_trade_fast_inner(code, date, false, false, usize::MAX)?;
    let market: Vec<MarketRecord> = market_all
        .iter()
        .filter(|m| m.last_prc != 0.0 && in_segment(tod(m.time_sec), T_OPEN, T_END))
        .cloned()
        .collect();
    let n = market.len();
    let mut out = Vec::with_capacity(HM21_LEN);
    if n < 2 {
        out.resize(HM21_LEN, f32::NAN);
        return Ok(out);
    }
    let max_label = n - 1;
    // trade-label groupby（flag 66=主买, 83=主卖）
    let mut t66 = vec![0.0f32; n];
    let mut t83 = vec![0.0f32; n];
    for tr in &trade_all {
        if in_segment(tod(tr.time_sec), T_OPEN, T_END) {
            let lab = market.partition_point(|m| m.time_sec <= tr.time_sec);
            if lab > 0 && lab <= n {
                let l = lab - 1;
                if tr.flag == 66 {
                    t66[l] += tr.volume;
                } else if tr.flag == 83 {
                    t83[l] += tr.volume;
                }
            }
        }
    }
    let mut raw_orig = Vec::new();
    let mut raw_r1 = Vec::new();
    let mut raw_r2 = Vec::new();
    let mut norm_orig = Vec::new();
    let mut norm_r1 = Vec::new();
    let mut norm_r2 = Vec::new();
    for kind in ["orig", "r1", "r2"] {
        let mut ask_vols = Vec::new();
        let mut ask_labels = Vec::new();
        let mut bid_vols = Vec::new();
        let mut bid_labels = Vec::new();
        for i in 1..n {
            if market[i].ask_prcs[0] < market[i - 1].ask_prcs[0] {
                ask_vols.push(vol_of(&market[i], "ask", kind));
                ask_labels.push(i);
            }
            if market[i].bid_prcs[0] > market[i - 1].bid_prcs[0] {
                bid_vols.push(vol_of(&market[i], "bid", kind));
                bid_labels.push(i);
            }
        }
        let nan19 = vec![f32::NAN; 19];
        if ask_vols.len() <= 15 || bid_vols.len() <= 15 {
            // hm21 原版要求点数 > 15
            match kind {
                "orig" => {
                    raw_orig.extend(&nan19);
                    norm_orig.extend(&nan19);
                }
                "r1" => {
                    raw_r1.extend(&nan19);
                    norm_r1.extend(&nan19);
                }
                _ => {
                    raw_r2.extend(&nan19);
                    norm_r2.extend(&nan19);
                }
            }
            continue;
        }
        let ask_66: Vec<f32> = ask_vols
            .iter()
            .zip(ask_labels.iter())
            .map(|(&v, &l)| {
                let d = t66[l] + v;
                if d > 0.0 {
                    v / d
                } else {
                    f32::NAN
                }
            })
            .collect();
        let bid_83: Vec<f32> = bid_vols
            .iter()
            .zip(bid_labels.iter())
            .map(|(&v, &l)| {
                let d = t83[l] + v;
                if d > 0.0 {
                    v / d
                } else {
                    f32::NAN
                }
            })
            .collect();
        let ask_s = hm21_stats_19(&ask_66, &ask_labels, max_label);
        let bid_s = hm21_stats_19(&bid_83, &bid_labels, max_label);
        let abs19: Vec<f32> = ask_s
            .iter()
            .zip(bid_s.iter())
            .map(|(&a, &b)| (a - b).abs())
            .collect();
        let ask_n = zscore_vec(&ask_66);
        let bid_n = zscore_vec(&bid_83);
        let ask_sn = hm21_stats_19(&ask_n, &ask_labels, max_label);
        let bid_sn = hm21_stats_19(&bid_n, &bid_labels, max_label);
        let abs19n: Vec<f32> = ask_sn
            .iter()
            .zip(bid_sn.iter())
            .map(|(&a, &b)| (a - b).abs())
            .collect();
        match kind {
            "orig" => {
                raw_orig.extend(&abs19);
                norm_orig.extend(&abs19n);
            }
            "r1" => {
                raw_r1.extend(&abs19);
                norm_r1.extend(&abs19n);
            }
            _ => {
                raw_r2.extend(&abs19);
                norm_r2.extend(&abs19n);
            }
        }
    }
    out.extend(combine_8panels(&raw_orig, &raw_r1, &raw_r2));
    out.extend(combine_8panels(&norm_orig, &norm_r1, &norm_r2));
    Ok(out)
}

// ============================================================================
// hm91：可观测挂单 —— obs_ask_ratio/obs_bid_ratio (单侧可见度) 10统计 + bid_ask_corr
// 原版 obs_ask_ratio = Σ十档/total_ask（卖侧十档可见度，[0,1]，已是 rate1 类）
// 三套口径：orig=十档/total, r1=单档/total, r2=单档/十档和（分子分母都变，不退化）
// 受影响：obs_ask 10统计 + obs_bid 10统计 + bid_ask_corr，× 5 时段 = 105，× 8 段 = 840
// ============================================================================

const HM91_FEATS: [&str; 21] = [
    "ask_mean",
    "ask_std",
    "ask_skew",
    "ask_kurt",
    "ask_iqr",
    "ask_median",
    "ask_max",
    "ask_min",
    "ask_autocorr1",
    "ask_trend",
    "bid_mean",
    "bid_std",
    "bid_skew",
    "bid_kurt",
    "bid_iqr",
    "bid_median",
    "bid_max",
    "bid_min",
    "bid_autocorr1",
    "bid_trend",
    "bid_ask_corr",
];
const HM91_LEN: usize = HM91_FEATS.len() * SEGMENTS.len() * PANEL_SUFFIX.len(); // 21*5*8=840

/// 单侧 obs 比例序列。kind: "orig"=Σ十档/total, "r1"=单档/total, "r2"=单档/Σ十档
fn obs_ratio(market: &[MarketRecord], idx: &[usize], side: &str, kind: &str) -> Vec<f32> {
    idx.iter()
        .map(|&i| {
            let m = &market[i];
            let s10 = if side == "ask" {
                m.ask_vols.iter().sum::<f32>()
            } else {
                m.bid_vols.iter().sum::<f32>()
            };
            let v1 = if side == "ask" {
                m.ask_vols[0]
            } else {
                m.bid_vols[0]
            };
            let tot = if side == "ask" {
                m.total_ask_vol
            } else {
                m.total_bid_vol
            };
            match kind {
                "orig" => {
                    if tot > 0.0 {
                        s10 / tot
                    } else {
                        f32::NAN
                    }
                }
                "r1" => {
                    if tot > 0.0 {
                        v1 / tot
                    } else {
                        f32::NAN
                    }
                }
                _ => {
                    if s10 > 0.0 {
                        v1 / s10
                    } else {
                        f32::NAN
                    }
                }
            }
        })
        .collect()
}

/// 序列 10 统计：mean/std/skew/kurt/iqr/median/max/min/autocorr1/trend。
fn stats10(v: &[f32]) -> [f32; 10] {
    let mut out = [f32::NAN; 10];
    let valid = filter_valid(v);
    let n = valid.len();
    if n == 0 {
        return out;
    }
    out[0] = mean_v(&valid);
    if n >= 2 {
        out[1] = std_v(&valid);
        out[6] = valid.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        out[7] = valid.iter().copied().fold(f32::INFINITY, f32::min);
        let mut s = valid.clone();
        s.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        out[4] = quantile_sorted(&s, 0.75) - quantile_sorted(&s, 0.25);
        out[5] = quantile_sorted(&s, 0.5);
        out[9] = trend_v(&valid);
    }
    if n >= 3 {
        out[2] = skew_v(&valid);
    }
    if n >= 4 {
        out[3] = kurt_v(&valid);
    }
    out[8] = autocorr_v(&valid, 1);
    out
}

fn hm91_names() -> Vec<String> {
    let mut names = Vec::with_capacity(HM91_LEN);
    for &panel in PANEL_SUFFIX.iter() {
        for &(seg, _, _) in SEGMENTS.iter() {
            for &feat in HM91_FEATS.iter() {
                names.push(format!("hm91_{}_{}_{}", feat, seg, panel));
            }
        }
    }
    names
}

pub fn compute_hm91_full(code: &str, date: i64) -> std::io::Result<Vec<f32>> {
    let market = read_market_fast_inner(code, date, false, false, usize::MAX)?;
    let all_idx: Vec<usize> = market
        .iter()
        .enumerate()
        .filter(|(_, m)| m.last_prc != 0.0 && in_segment(tod(m.time_sec), T_OPEN, T_END))
        .map(|(i, _)| i)
        .collect();
    let cap = HM91_FEATS.len() * SEGMENTS.len();
    let mut orig105 = Vec::with_capacity(cap);
    let mut r1_105 = Vec::with_capacity(cap);
    let mut r2_105 = Vec::with_capacity(cap);
    for kind in ["orig", "r1", "r2"] {
        for &(seg, t0, t1) in SEGMENTS.iter() {
            let idx: Vec<usize> = all_idx
                .iter()
                .copied()
                .filter(|&i| in_segment(tod(market[i].time_sec), t0, t1))
                .collect();
            if idx.is_empty() {
                let block = vec![f32::NAN; HM91_FEATS.len()];
                match kind {
                    "orig" => orig105.extend(&block),
                    "r1" => r1_105.extend(&block),
                    _ => r2_105.extend(&block),
                }
                continue;
            }
            let ask = obs_ratio(&market, &idx, "ask", kind);
            let bid = obs_ratio(&market, &idx, "bid", kind);
            let ask10 = stats10(&ask);
            let bid10 = stats10(&bid);
            let corr = corr_v(&ask, &bid); // bid_ask_corr
            let mut block = Vec::with_capacity(HM91_FEATS.len());
            block.extend_from_slice(&ask10);
            block.extend_from_slice(&bid10);
            block.push(corr);
            match kind {
                "orig" => orig105.extend(&block),
                "r1" => r1_105.extend(&block),
                _ => r2_105.extend(&block),
            }
        }
    }
    Ok(combine_8panels(&orig105, &r1_105, &r2_105))
}

// ============================================================================
// hm11：百鸟朝凤 —— 高挂单时段（5档和>80%分位）的成交统计 ask/bid 差
// df2_ask/bid = 5档和，减 80% 分位，取>0 时刻；trade 经 merge_asof(forward,容差) 锚定。
// 12 差值 = turnover(std/median/max ÷ to) × all/buy/sell + wait_median × all/buy/sell
// 受影响：5档和口径（orig/r1/r2）决定"哪些时刻是高挂单"，× 5时段 × 2容差(60s/30s) = 120
// 简化：省略原版 asks>90% 大单二次筛选（trade 侧，与挂单量改造无关，三套共用）。
// ============================================================================

const HM11_STATS: [&str; 12] = [
    "to_all_std",
    "to_all_median",
    "to_all_max",
    "to_buy_std",
    "to_buy_median",
    "to_buy_max",
    "to_sell_std",
    "to_sell_median",
    "to_sell_max",
    "wait_all_median",
    "wait_buy_median",
    "wait_sell_median",
];
const HM11_LEN: usize = HM11_STATS.len() * SEGMENTS.len() * 2 * PANEL_SUFFIX.len(); // 12*5*2*8=960

/// 5 档挂单量和按 kind 归一。kind: "orig"=绝对量, "r1"=÷total, "r2"=÷十档和
fn sum5(m: &MarketRecord, side: &str, kind: &str) -> f32 {
    let vols = if side == "ask" {
        &m.ask_vols[..5]
    } else {
        &m.bid_vols[..5]
    };
    let s5: f32 = vols.iter().sum();
    match kind {
        "orig" => s5,
        "r1" => {
            let tot = if side == "ask" {
                m.total_ask_vol
            } else {
                m.total_bid_vol
            };
            if tot > 0.0 {
                s5 / tot
            } else {
                f32::NAN
            }
        }
        _ => {
            let s10 = if side == "ask" {
                m.ask_vols.iter().sum::<f32>()
            } else {
                m.bid_vols.iter().sum::<f32>()
            };
            if s10 > 0.0 {
                s5 / s10
            } else {
                f32::NAN
            }
        }
    }
}

/// 高挂单时刻（5档和 > 80% 分位），返回 time_sec 列表（已按时段 idx 顺序）。
fn high_vol_times(market: &[MarketRecord], idx: &[usize], side: &str, kind: &str) -> Vec<f32> {
    let vals: Vec<f32> = idx.iter().map(|&i| sum5(&market[i], side, kind)).collect();
    let mut sorted: Vec<f32> = vals.iter().copied().filter(|v| v.is_finite()).collect();
    sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    if sorted.is_empty() {
        return vec![];
    }
    let thr = quantile_sorted(&sorted, 0.8);
    idx.iter()
        .zip(vals.iter())
        .filter(|(_, v)| **v > thr)
        .map(|(&i, _)| market[i].time_sec)
        .collect()
}

/// trade 锚定到高挂单时刻（forward merge_asof，容差 tol 秒）。返回 (trade_idx, wait)。
fn merge_asof_forward(trade: &[TradeRecord], high: &[f32], tol: f32) -> Vec<(usize, f32)> {
    let mut out = Vec::new();
    let mut hi = 0usize;
    for (ti, t) in trade.iter().enumerate() {
        while hi < high.len() && high[hi] < t.time_sec {
            hi += 1;
        }
        if hi < high.len() && high[hi] <= t.time_sec + tol {
            out.push((ti, high[hi] - t.time_sec));
        }
    }
    out
}

/// 12 成交统计（turnover std/median/max ÷ to × all/buy/sell + wait_median × all/buy/sell）。
fn hm11_stats12(trade: &[TradeRecord], anchored: &[(usize, f32)]) -> [f32; 12] {
    let mut out = [f32::NAN; 12];
    if anchored.is_empty() {
        return out;
    }
    let mut tov = [Vec::new(), Vec::new(), Vec::new()]; // all, buy(66), sell(83)
    let mut wai = [Vec::new(), Vec::new(), Vec::new()];
    for &(ti, w) in anchored {
        let tr = &trade[ti];
        tov[0].push(tr.turnover);
        wai[0].push(w);
        if tr.flag == 66 {
            tov[1].push(tr.turnover);
            wai[1].push(w);
        } else if tr.flag == 83 {
            tov[2].push(tr.turnover);
            wai[2].push(w);
        }
    }
    for g in 0..3 {
        let to: f32 = tov[g].iter().sum();
        let base = g * 3;
        if to > 0.0 && tov[g].len() >= 2 {
            let m = mean_v(&tov[g]);
            let sq: f32 = tov[g].iter().map(|x| (x - m).powi(2)).sum::<f32>() / tov[g].len() as f32;
            out[base] = sq.sqrt() / to; // std/to
            let mut s = tov[g].clone();
            s.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
            out[base + 1] = quantile_sorted(&s, 0.5) / to; // median/to
            out[base + 2] = s[s.len() - 1] / to; // max/to
        }
        if !wai[g].is_empty() {
            let mut sw = wai[g].clone();
            sw.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
            out[9 + g] = quantile_sorted(&sw, 0.5); // wait_median
        }
    }
    out
}

fn hm11_names() -> Vec<String> {
    let mut names = Vec::with_capacity(HM11_LEN);
    for &panel in PANEL_SUFFIX.iter() {
        for &(seg, _, _) in SEGMENTS.iter() {
            for tol in ["tol60", "tol30"].iter() {
                for &stat in HM11_STATS.iter() {
                    names.push(format!("hm11_{}_{}_{}_{}", stat, seg, tol, panel));
                }
            }
        }
    }
    names
}

pub fn compute_hm11_full(code: &str, date: i64) -> std::io::Result<Vec<f32>> {
    let market_all = read_market_fast_inner(code, date, false, false, usize::MAX)?;
    let trade_all = read_trade_fast_inner(code, date, false, false, usize::MAX)?;
    let market: Vec<MarketRecord> = market_all
        .iter()
        .filter(|m| m.last_prc != 0.0 && in_segment(tod(m.time_sec), T_OPEN, T_END))
        .cloned()
        .collect();
    let trade: Vec<TradeRecord> = trade_all
        .iter()
        .filter(|t| in_segment(tod(t.time_sec), T_OPEN, T_END))
        .cloned()
        .collect();
    let n = market.len();
    let cap = HM11_STATS.len() * SEGMENTS.len() * 2;
    let mut orig120 = Vec::with_capacity(cap);
    let mut r1_120 = Vec::with_capacity(cap);
    let mut r2_120 = Vec::with_capacity(cap);
    for kind in ["orig", "r1", "r2"] {
        for &(seg, t0, t1) in SEGMENTS.iter() {
            let seg_idx: Vec<usize> = (0..n)
                .filter(|&i| in_segment(tod(market[i].time_sec), t0, t1))
                .collect();
            for &tol in &[60.0f32, 30.0] {
                if seg_idx.is_empty() {
                    let block = vec![f32::NAN; HM11_STATS.len()];
                    match kind {
                        "orig" => orig120.extend(&block),
                        "r1" => r1_120.extend(&block),
                        _ => r2_120.extend(&block),
                    }
                    continue;
                }
                let ask_high = high_vol_times(&market, &seg_idx, "ask", kind);
                let bid_high = high_vol_times(&market, &seg_idx, "bid", kind);
                let ask_anc = merge_asof_forward(&trade, &ask_high, tol);
                let bid_anc = merge_asof_forward(&trade, &bid_high, tol);
                let ask_s = hm11_stats12(&trade, &ask_anc);
                let bid_s = hm11_stats12(&trade, &bid_anc);
                let diff12: Vec<f32> = ask_s
                    .iter()
                    .zip(bid_s.iter())
                    .map(|(&a, &b)| a - b)
                    .collect();
                match kind {
                    "orig" => orig120.extend(&diff12),
                    "r1" => r1_120.extend(&diff12),
                    _ => r2_120.extend(&diff12),
                }
            }
        }
    }
    Ok(combine_8panels(&orig120, &r1_120, &r2_120))
}

// ============================================================================
// hm10：黑粉铁粉 —— 十档挂单量 diff 与主动成交的 corr 均值（res_ask_pri/bid_pri）
// df2_ask/bid = 十档 vol 按 kind 归一后再 diff；与 buy/sell × before/after 成交算 corr，取 10 档均值。
// 简化：只做 diff 版（省略 cumsum/slope/slope_diff/1m频）。8/配置 × 5时段 = 40，× 8段 = 320。
// ============================================================================

const HM10_STATS: [&str; 8] = [
    "ask_buy_before",
    "ask_sell_before",
    "ask_buy_after",
    "ask_sell_after",
    "bid_buy_before",
    "bid_sell_before",
    "bid_buy_after",
    "bid_sell_after",
];
const HM10_LEN: usize = HM10_STATS.len() * SEGMENTS.len() * PANEL_SUFFIX.len(); // 8*5*8=320

/// 第 k 档（0-based）按 kind 归一。
fn vol_k(m: &MarketRecord, k: usize, side: &str, kind: &str) -> f32 {
    let raw = if side == "ask" {
        m.ask_vols[k]
    } else {
        m.bid_vols[k]
    };
    match kind {
        "orig" => raw,
        "r1" => {
            let tot = if side == "ask" {
                m.total_ask_vol
            } else {
                m.total_bid_vol
            };
            if tot > 0.0 {
                raw / tot
            } else {
                f32::NAN
            }
        }
        _ => {
            let s10 = if side == "ask" {
                m.ask_vols.iter().sum::<f32>()
            } else {
                m.bid_vols.iter().sum::<f32>()
            };
            if s10 > 0.0 {
                raw / s10
            } else {
                f32::NAN
            }
        }
    }
}

/// 10 档 corr 的均值。
fn mean10_corr(diff10: &[Vec<f32>], vol: &[f32]) -> f32 {
    let mut cs = Vec::with_capacity(10);
    for k in 0..10 {
        let c = corr_v(&diff10[k], vol);
        if c.is_finite() {
            cs.push(c);
        }
    }
    if cs.is_empty() {
        f32::NAN
    } else {
        mean_v(&cs)
    }
}

fn hm10_names() -> Vec<String> {
    let mut names = Vec::with_capacity(HM10_LEN);
    for &panel in PANEL_SUFFIX.iter() {
        for &(seg, _, _) in SEGMENTS.iter() {
            for &stat in HM10_STATS.iter() {
                names.push(format!("hm10_{}_{}_{}", stat, seg, panel));
            }
        }
    }
    names
}

pub fn compute_hm10_full(code: &str, date: i64) -> std::io::Result<Vec<f32>> {
    let market_all = read_market_fast_inner(code, date, false, false, usize::MAX)?;
    let trade_all = read_trade_fast_inner(code, date, false, false, usize::MAX)?;
    let market: Vec<MarketRecord> = market_all
        .iter()
        .filter(|m| m.last_prc != 0.0 && in_segment(tod(m.time_sec), T_OPEN, T_END))
        .cloned()
        .collect();
    let n = market.len();
    let cap = HM10_STATS.len() * SEGMENTS.len();
    let mut orig40 = Vec::with_capacity(cap);
    let mut r1_40 = Vec::with_capacity(cap);
    let mut r2_40 = Vec::with_capacity(cap);
    if n < 3 {
        orig40.resize(cap, f32::NAN);
        r1_40.resize(cap, f32::NAN);
        r2_40.resize(cap, f32::NAN);
        return Ok(combine_8panels(&orig40, &r1_40, &r2_40));
    }
    // trade-label groupby（flag 66=主买, 83=主卖）
    let mut buy_vol = vec![0.0f32; n];
    let mut sell_vol = vec![0.0f32; n];
    for tr in &trade_all {
        if in_segment(tod(tr.time_sec), T_OPEN, T_END) {
            let lab = market.partition_point(|m| m.time_sec <= tr.time_sec);
            if lab > 0 && lab <= n {
                let l = lab - 1;
                if tr.flag == 66 {
                    buy_vol[l] += tr.volume;
                } else if tr.flag == 83 {
                    sell_vol[l] += tr.volume;
                }
            }
        }
    }
    for kind in ["orig", "r1", "r2"] {
        for &(seg, t0, t1) in SEGMENTS.iter() {
            let seg_idx: Vec<usize> = (0..n)
                .filter(|&i| in_segment(tod(market[i].time_sec), t0, t1))
                .collect();
            let mut block = [f32::NAN; 8];
            if seg_idx.len() >= 3 {
                let m = seg_idx.len();
                // 十档 diff（按 seg_idx 顺序）
                let mut ask_diff = vec![vec![f32::NAN; m]; 10];
                let mut bid_diff = vec![vec![f32::NAN; m]; 10];
                for k in 0..10 {
                    for i in 1..m {
                        ask_diff[k][i] = vol_k(&market[seg_idx[i]], k, "ask", kind)
                            - vol_k(&market[seg_idx[i - 1]], k, "ask", kind);
                        bid_diff[k][i] = vol_k(&market[seg_idx[i]], k, "bid", kind)
                            - vol_k(&market[seg_idx[i - 1]], k, "bid", kind);
                    }
                }
                // buy/sell before/after（before=当前label, after=label+1）
                let buy_before: Vec<f32> = seg_idx.iter().map(|&i| buy_vol[i]).collect();
                let sell_before: Vec<f32> = seg_idx.iter().map(|&i| sell_vol[i]).collect();
                let buy_after: Vec<f32> = seg_idx
                    .iter()
                    .map(|&i| if i + 1 < n { buy_vol[i + 1] } else { f32::NAN })
                    .collect();
                let sell_after: Vec<f32> = seg_idx
                    .iter()
                    .map(|&i| if i + 1 < n { sell_vol[i + 1] } else { f32::NAN })
                    .collect();
                block[0] = mean10_corr(&ask_diff, &buy_before);
                block[1] = mean10_corr(&ask_diff, &sell_before);
                block[2] = mean10_corr(&ask_diff, &buy_after);
                block[3] = mean10_corr(&ask_diff, &sell_after);
                block[4] = mean10_corr(&bid_diff, &buy_before);
                block[5] = mean10_corr(&bid_diff, &sell_before);
                block[6] = mean10_corr(&bid_diff, &buy_after);
                block[7] = mean10_corr(&bid_diff, &sell_after);
            }
            match kind {
                "orig" => orig40.extend_from_slice(&block),
                "r1" => r1_40.extend_from_slice(&block),
                _ => r2_40.extend_from_slice(&block),
            }
        }
    }
    Ok(combine_8panels(&orig40, &r1_40, &r2_40))
}

// ============================================================================
// hm73：八戒照镜子 —— vol2-5 对价格波动率的 OLS 残差统计 + ask/bid 残差双侧 corr
// volatility = last_prc.pct_change.rolling(20).std；resi = vol − (a+b·volatility)
// 简化：聚焦残差统计 + 双侧 corr（hm73 唯一买卖耦合），省略 trend_2d/rank/get_features_factors。
// 受影响：vol2-5 每档 × {ask mean, ask std, bid mean, bid std, ask-bid corr} = 20 × 5时段 = 100
// ============================================================================

const HM73_FEATS: [&str; 20] = [
    "v2_ask_mean",
    "v2_ask_std",
    "v2_bid_mean",
    "v2_bid_std",
    "v2_ask_bid_corr",
    "v3_ask_mean",
    "v3_ask_std",
    "v3_bid_mean",
    "v3_bid_std",
    "v3_ask_bid_corr",
    "v4_ask_mean",
    "v4_ask_std",
    "v4_bid_mean",
    "v4_bid_std",
    "v4_ask_bid_corr",
    "v5_ask_mean",
    "v5_ask_std",
    "v5_bid_mean",
    "v5_bid_std",
    "v5_ask_bid_corr",
];
const HM73_LEN: usize = HM73_FEATS.len() * SEGMENTS.len() * PANEL_SUFFIX.len(); // 20*5*8=800

/// OLS 残差：resi = y − (a+b·x)，b = cov(x,y)/var(x)，a = ȳ − b·x̄。
fn ols_residuals(x: &[f32], y: &[f32]) -> Vec<f32> {
    let n = x.len().min(y.len());
    let mut out = vec![f32::NAN; n];
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    for i in 0..n {
        if x[i].is_finite() && y[i].is_finite() {
            xs.push(x[i]);
            ys.push(y[i]);
        }
    }
    let m = xs.len();
    if m < 2 {
        return out;
    }
    let mx = xs.iter().sum::<f32>() / m as f32;
    let my = ys.iter().sum::<f32>() / m as f32;
    let (mut cov, mut vx) = (0.0f32, 0.0f32);
    for k in 0..m {
        cov += (xs[k] - mx) * (ys[k] - my);
        vx += (xs[k] - mx).powi(2);
    }
    if vx <= 0.0 {
        return out;
    }
    let b = cov / vx;
    let a = my - b * mx;
    for i in 0..n {
        if x[i].is_finite() && y[i].is_finite() {
            out[i] = y[i] - (a + b * x[i]);
        }
    }
    out
}

/// rolling std（窗口 w，含当前，ddof=0）。
fn rolling_std(x: &[f32], w: usize) -> Vec<f32> {
    let n = x.len();
    let mut out = vec![f32::NAN; n];
    if w == 0 {
        return out;
    }
    for i in 0..n {
        if i + 1 >= w {
            let win = &x[i + 1 - w..=i];
            let valid: Vec<f32> = win.iter().copied().filter(|v| v.is_finite()).collect();
            if valid.len() >= 2 {
                let m = mean_v(&valid);
                let sq: f32 = valid.iter().map(|v| (v - m).powi(2)).sum::<f32>();
                out[i] = (sq / valid.len() as f32).sqrt();
            }
        }
    }
    out
}

fn hm73_names() -> Vec<String> {
    let mut names = Vec::with_capacity(HM73_LEN);
    for &panel in PANEL_SUFFIX.iter() {
        for &(seg, _, _) in SEGMENTS.iter() {
            for &feat in HM73_FEATS.iter() {
                names.push(format!("hm73_{}_{}_{}", feat, seg, panel));
            }
        }
    }
    names
}

pub fn compute_hm73_full(code: &str, date: i64) -> std::io::Result<Vec<f32>> {
    let market = read_market_fast_inner(code, date, false, false, usize::MAX)?;
    let all_idx: Vec<usize> = market
        .iter()
        .enumerate()
        .filter(|(_, m)| m.last_prc != 0.0 && in_segment(tod(m.time_sec), T_OPEN, T_END))
        .map(|(i, _)| i)
        .collect();
    let na = all_idx.len();
    let cap = HM73_FEATS.len() * SEGMENTS.len();
    let mut orig100 = Vec::with_capacity(cap);
    let mut r1_100 = Vec::with_capacity(cap);
    let mut r2_100 = Vec::with_capacity(cap);
    if na < 22 {
        orig100.resize(cap, f32::NAN);
        r1_100.resize(cap, f32::NAN);
        r2_100.resize(cap, f32::NAN);
        return Ok(combine_8panels(&orig100, &r1_100, &r2_100));
    }
    // volatility = pct_change.rolling(20).std（基于全天盘内）
    let prc: Vec<f32> = all_idx.iter().map(|&i| market[i].last_prc).collect();
    let mut pct = vec![f32::NAN; na];
    for i in 1..na {
        if prc[i].is_finite() && prc[i - 1] > 0.0 {
            pct[i] = prc[i] / prc[i - 1] - 1.0;
        }
    }
    let volat = rolling_std(&pct, 20);
    for kind in ["orig", "r1", "r2"] {
        for &(seg, t0, t1) in SEGMENTS.iter() {
            // seg_pos: all_idx 中时段内的位置（0..na）
            let seg_pos: Vec<usize> = (0..na)
                .filter(|&p| in_segment(tod(market[all_idx[p]].time_sec), t0, t1))
                .collect();
            let mut block = vec![f32::NAN; HM73_FEATS.len()];
            if seg_pos.len() >= 22 {
                let vol_seg: Vec<f32> = seg_pos.iter().map(|&p| volat[p]).collect();
                for (j, &k) in [2usize, 3, 4, 5].iter().enumerate() {
                    let ask_seg: Vec<f32> = seg_pos
                        .iter()
                        .map(|&p| vol_k(&market[all_idx[p]], k - 1, "ask", kind))
                        .collect();
                    let bid_seg: Vec<f32> = seg_pos
                        .iter()
                        .map(|&p| vol_k(&market[all_idx[p]], k - 1, "bid", kind))
                        .collect();
                    let ask_resi = ols_residuals(&vol_seg, &ask_seg);
                    let bid_resi = ols_residuals(&vol_seg, &bid_seg);
                    let base = j * 5;
                    block[base] = mean_v(&filter_valid(&ask_resi));
                    block[base + 1] = std_v(&filter_valid(&ask_resi));
                    block[base + 2] = mean_v(&filter_valid(&bid_resi));
                    block[base + 3] = std_v(&filter_valid(&bid_resi));
                    block[base + 4] = corr_v(&ask_resi, &bid_resi);
                }
            }
            match kind {
                "orig" => orig100.extend_from_slice(&block),
                "r1" => r1_100.extend_from_slice(&block),
                _ => r2_100.extend_from_slice(&block),
            }
        }
    }
    Ok(combine_8panels(&orig100, &r1_100, &r2_100))
}

// ============================================================================
// hm79：交流之苦 —— bid/ask 十档协方差结构特征（自实现简化版）
// 原版 orderbook_volume_cov_factors 的 15 特征（DMatrix/Schur/特征值）是 pyo3+私有，
// 纯 Rust 不可调且不改现有 → 自实现核心 4 个：ask 方差和、bid 方差和、ask-bid 跨侧协方差和、
// 20×20 双侧相关矩阵上三角均值（无量纲，三套可比）。× 5时段 = 20，× 8段 = 160。
// ============================================================================

const HM79_FEATS: [&str; 4] = [
    "ask_var_sum",
    "bid_var_sum",
    "ask_bid_cov_sum",
    "total_corr_mean",
];
const HM79_LEN: usize = HM79_FEATS.len() * SEGMENTS.len() * PANEL_SUFFIX.len(); // 4*5*8=160

fn var_v(x: &[f32]) -> f32 {
    let valid = filter_valid(x);
    if valid.len() < 2 {
        return f32::NAN;
    }
    let m = mean_v(&valid);
    valid.iter().map(|v| (v - m).powi(2)).sum::<f32>() / valid.len() as f32
}

fn cov_v(x: &[f32], y: &[f32]) -> f32 {
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    for k in 0..x.len().min(y.len()) {
        if x[k].is_finite() && y[k].is_finite() {
            xs.push(x[k]);
            ys.push(y[k]);
        }
    }
    if xs.len() < 2 {
        return f32::NAN;
    }
    let mx = mean_v(&xs);
    let my = mean_v(&ys);
    xs.iter()
        .zip(ys.iter())
        .map(|(a, b)| (a - mx) * (b - my))
        .sum::<f32>()
        / xs.len() as f32
}

fn hm79_names() -> Vec<String> {
    let mut names = Vec::with_capacity(HM79_LEN);
    for &panel in PANEL_SUFFIX.iter() {
        for &(seg, _, _) in SEGMENTS.iter() {
            for &feat in HM79_FEATS.iter() {
                names.push(format!("hm79_{}_{}_{}", feat, seg, panel));
            }
        }
    }
    names
}

pub fn compute_hm79_full(code: &str, date: i64) -> std::io::Result<Vec<f32>> {
    let market = read_market_fast_inner(code, date, false, false, usize::MAX)?;
    let all_idx: Vec<usize> = market
        .iter()
        .enumerate()
        .filter(|(_, m)| m.last_prc != 0.0 && in_segment(tod(m.time_sec), T_OPEN, T_END))
        .map(|(i, _)| i)
        .collect();
    let cap = HM79_FEATS.len() * SEGMENTS.len();
    let mut orig20 = Vec::with_capacity(cap);
    let mut r1_20 = Vec::with_capacity(cap);
    let mut r2_20 = Vec::with_capacity(cap);
    for kind in ["orig", "r1", "r2"] {
        for &(seg, t0, t1) in SEGMENTS.iter() {
            let seg_idx: Vec<usize> = all_idx
                .iter()
                .copied()
                .filter(|&i| in_segment(tod(market[i].time_sec), t0, t1))
                .collect();
            let mut block = [f32::NAN; HM79_FEATS.len()];
            if seg_idx.len() >= 5 {
                // 20 列：10 ask + 10 bid（各档按 kind 归一）
                let mut cols: Vec<Vec<f32>> = Vec::with_capacity(20);
                for k in 0..10 {
                    cols.push(
                        seg_idx
                            .iter()
                            .map(|&i| vol_k(&market[i], k, "ask", kind))
                            .collect(),
                    );
                }
                for k in 0..10 {
                    cols.push(
                        seg_idx
                            .iter()
                            .map(|&i| vol_k(&market[i], k, "bid", kind))
                            .collect(),
                    );
                }
                let avs: f32 = (0..10)
                    .map(|k| var_v(&cols[k]))
                    .filter(|v| v.is_finite())
                    .sum();
                let bvs: f32 = (0..10)
                    .map(|k| var_v(&cols[10 + k]))
                    .filter(|v| v.is_finite())
                    .sum();
                let abc: f32 = (0..10)
                    .map(|k| cov_v(&cols[k], &cols[10 + k]))
                    .filter(|v| v.is_finite())
                    .sum();
                let mut corrs = Vec::new();
                for i in 0..20 {
                    for j in (i + 1)..20 {
                        let c = corr_v(&cols[i], &cols[j]);
                        if c.is_finite() {
                            corrs.push(c);
                        }
                    }
                }
                let tcm = mean_v(&corrs);
                block = [avs, bvs, abc, tcm];
            }
            match kind {
                "orig" => orig20.extend_from_slice(&block),
                "r1" => r1_20.extend_from_slice(&block),
                _ => r2_20.extend_from_slice(&block),
            }
        }
    }
    Ok(combine_8panels(&orig20, &r1_20, &r2_20))
}

// ============================================================================
// hm49：龙生九子 —— 跨日 imb1 持续性（今日 vs 昨日 corr）+ 今日 imb1 统计
// 原版 rolling_corr 找相似窗口 + get_stats/get_stats2 太繁，简化为跨日 imb 关联核心。
// prev_date 用 date−1 近似（跨月/节假日读不到数据则跨日 corr 全 NaN）。
// 受影响：crossday_corr(1) + 今日 imb1 的 6 统计 × 5 时段(30) = 31/套，× 3 套，× 8 段 = 248
// ============================================================================

const HM49_STATS6: [&str; 6] = ["mean", "std", "skew", "kurt", "autocorr1", "trend"];
const HM49_LEN: usize = (1 + HM49_STATS6.len() * SEGMENTS.len()) * PANEL_SUFFIX.len(); // 31*8=248

fn hm49_names() -> Vec<String> {
    let mut names = Vec::with_capacity(HM49_LEN);
    for &panel in PANEL_SUFFIX.iter() {
        names.push(format!("hm49_crossday_corr_{}", panel));
        for &(seg, _, _) in SEGMENTS.iter() {
            for &stat in HM49_STATS6.iter() {
                names.push(format!("hm49_{}_{}_{}", stat, seg, panel));
            }
        }
    }
    names
}

pub fn compute_hm49_full(code: &str, date: i64) -> std::io::Result<Vec<f32>> {
    let market = read_market_fast_inner(code, date, false, false, usize::MAX)?;
    let all_idx: Vec<usize> = market
        .iter()
        .enumerate()
        .filter(|(_, m)| m.last_prc != 0.0 && in_segment(tod(m.time_sec), T_OPEN, T_END))
        .map(|(i, _)| i)
        .collect();
    // 昨日 market（date−1 近似，跨月可能读不到 → 跨日 corr NaN）
    let yest_market = read_market_fast_inner(code, date - 1, false, false, usize::MAX).ok();
    let yest_idx: Vec<usize> = match &yest_market {
        Some(ym) => ym
            .iter()
            .enumerate()
            .filter(|(_, m)| m.last_prc != 0.0 && in_segment(tod(m.time_sec), T_OPEN, T_END))
            .map(|(i, _)| i)
            .collect(),
        None => vec![],
    };
    let ncol = 1 + HM49_STATS6.len() * SEGMENTS.len();
    let mut orig_v = Vec::with_capacity(ncol);
    let mut r1_v = Vec::with_capacity(ncol);
    let mut r2_v = Vec::with_capacity(ncol);
    for kind in ["orig", "r1", "r2"] {
        let today_imb_all = imb_seq(&market, &all_idx, 1, kind, 1.0);
        let yest_imb_all = if let Some(ym) = &yest_market {
            imb_seq(ym, &yest_idx, 1, kind, 1.0)
        } else {
            vec![]
        };
        let cross_corr = corr_v(&today_imb_all, &yest_imb_all);
        let mut block = Vec::with_capacity(ncol);
        block.push(cross_corr);
        for &(seg, t0, t1) in SEGMENTS.iter() {
            let seg_idx: Vec<usize> = all_idx
                .iter()
                .copied()
                .filter(|&i| in_segment(tod(market[i].time_sec), t0, t1))
                .collect();
            let imb_seg = imb_seq(&market, &seg_idx, 1, kind, 1.0);
            let valid = filter_valid(&imb_seg);
            block.push(mean_v(&valid));
            block.push(std_v(&valid));
            block.push(skew_v(&valid));
            block.push(kurt_v(&valid));
            block.push(autocorr_v(&valid, 1));
            block.push(trend_v(&valid));
        }
        match kind {
            "orig" => orig_v = block,
            "r1" => r1_v = block,
            _ => r2_v = block,
        }
    }
    Ok(combine_8panels(&orig_v, &r1_v, &r2_v))
}

// ============================================================================
// 串联汇总（后续系列逐个加入）
// ============================================================================

pub fn compute_orderbook_imb_refactor_full(code: &str, date: i64) -> std::io::Result<Vec<f32>> {
    let mut out = Vec::new();
    out.extend(compute_hm32_full(code, date)?);
    out.extend(compute_hm46_full(code, date)?);
    out.extend(compute_hm72_full(code, date)?);
    out.extend(compute_hm21_full(code, date)?);
    out.extend(compute_hm91_full(code, date)?);
    out.extend(compute_hm11_full(code, date)?);
    out.extend(compute_hm10_full(code, date)?);
    out.extend(compute_hm73_full(code, date)?);
    out.extend(compute_hm79_full(code, date)?);
    out.extend(compute_hm49_full(code, date)?);
    Ok(out)
}

pub fn orderbook_imb_refactor_names() -> Vec<String> {
    let mut names = Vec::new();
    names.extend(hm32_names());
    names.extend(hm46_names());
    names.extend(hm72_names());
    names.extend(hm21_names());
    names.extend(hm91_names());
    names.extend(hm11_names());
    names.extend(hm10_names());
    names.extend(hm73_names());
    names.extend(hm79_names());
    names.extend(hm49_names());
    names
}

// ============================================================================
// PyO3 包装（单股调试入口，错误抛异常）
// ============================================================================

#[pyfunction]
pub fn py_compute_orderbook_imb_refactor(code: &str, date: i64) -> PyResult<Vec<f32>> {
    compute_orderbook_imb_refactor_full(code, date)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("{:?}", e)))
}

#[pyfunction]
pub fn py_orderbook_imb_refactor_names() -> Vec<String> {
    orderbook_imb_refactor_names()
}
