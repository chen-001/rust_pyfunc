//! 蒸馏思想因子（distill）—— 600 维日频横截面因子。
//!
//! 核心思想：集合竞价 = Few-Shot Teacher，连续竞价 = Student。
//! 通过 4 个分支刻画 "投资者学习 → 遗忘" 曲线。
//!
//! NaN 策略：只要 P_auct 可用（集合竞价段 ≥5 条），全部 600 因子均为有限值。
//! 所有统计辅助函数在样本不足时返回 0.0（或 Hurst 返回 0.5）而非 NaN。
//! epsilon 5 维在 teacher 信号缺失时用降级归一化方式仍产出有限值。
//!
//! 三层结构：compute_distill_full（核心）+ pipeline_distill（批量）+ py_distill（调试）。
//! 详见 docs/superpowers/specs/2026-07-09-distillation-factor-design.md。

use crate::fast_csv_reader::{
    read_market_fast_inner, read_trade_fast_inner, MarketRecord, TradeRecord,
};
use pyo3::prelude::*;

// ============================================================================
// 常量
// ============================================================================

/// A:100 + B:140 + D:240 + E:121 = 601。
pub const OUT_LEN: usize = 601;

const AUCT_START: f32 = 33300.0;
const AUCT_END: f32 = 33900.0;
const AUCT_SIGMA_START: f32 = 33600.0;
const SESSION_START: f32 = 34200.0;
const SESSION_END: f32 = 48420.0;
const SEC_PER_MIN: f32 = 60.0;
const MIN_AUCT_RECORDS: usize = 5;

/// 分支 A 窗口网格。
const WINDOWS_A: [usize; 7] = [3, 5, 10, 15, 30, 60, 120];
/// 分支 A 突破阈值。
const THRESHOLDS_A: [f32; 6] = [0.001, 0.003, 0.005, 0.01, 0.02, 0.03];
/// 分支 B 窗口。
const WINDOWS_B: [usize; 5] = [3, 5, 15, 30, 60];
/// 分支 D 时段结束（分钟）。
const SEG_ENDS_D: [usize; 6] = [15, 30, 60, 120, 180, 237];
const SEG_MIDS_D: [f32; 6] = [7.5, 22.5, 45.0, 90.0, 150.0, 208.5];
/// 大单阈值分位。
const LARGE_ORDER_QUANTILES: [f64; 3] = [0.90, 0.95, 0.99];
const HAWKES_TRIGGER_WIN: f32 = 60.0;

// ============================================================================
// 数据结构
// ============================================================================

#[derive(Clone, Debug)]
struct TeacherSignals {
    p_auct: f32,
    v_auct: f32,
    sigma_auct: f32,
    delta_auct: f32,
    delta_yc: f32,
}

struct MinuteData {
    n: usize,
    price: Vec<f32>,
    volume: Vec<f32>,
    abr: Vec<f32>,
    ofi: Vec<f32>,
    logret: Vec<f32>,
}

// ============================================================================
// 统计辅助函数 —— 样本不足时返回 0.0（或 0.5 for Hurst），绝不返回 NaN
// ============================================================================

fn mean(v: &[f32]) -> f32 {
    let valid: Vec<f32> = v.iter().copied().filter(|x| x.is_finite()).collect();
    if valid.is_empty() {
        return 0.0;
    }
    valid.iter().sum::<f32>() / valid.len() as f32
}

fn std_dev(v: &[f32]) -> f32 {
    let valid: Vec<f32> = v.iter().copied().filter(|x| x.is_finite()).collect();
    let n = valid.len();
    if n < 2 {
        return 0.0;
    }
    let m = valid.iter().sum::<f32>() / n as f32;
    let var = valid.iter().map(|x| (x - m).powi(2)).sum::<f32>() / (n - 1) as f32;
    var.max(0.0).sqrt()
}

fn skewness(v: &[f32]) -> f32 {
    let valid: Vec<f32> = v.iter().copied().filter(|x| x.is_finite()).collect();
    let n = valid.len();
    if n < 3 {
        return 0.0;
    }
    let m = valid.iter().sum::<f32>() / n as f32;
    let s = std_dev(&valid);
    if s <= 0.0 {
        return 0.0;
    }
    let m3 = valid.iter().map(|x| (x - m).powi(3)).sum::<f32>() / n as f32;
    let g1 = m3 / s.powi(3);
    let nf = n as f32;
    g1 * ((nf - 1.0).powf(1.5)) / ((nf - 2.0) * nf.sqrt())
}

fn kurtosis(v: &[f32]) -> f32 {
    let valid: Vec<f32> = v.iter().copied().filter(|x| x.is_finite()).collect();
    let n = valid.len();
    if n < 4 {
        return 0.0;
    }
    let m = valid.iter().sum::<f32>() / n as f32;
    let var = valid.iter().map(|x| (x - m).powi(2)).sum::<f32>() / n as f32;
    if var <= 0.0 {
        return 0.0;
    }
    let m4 = valid.iter().map(|x| (x - m).powi(4)).sum::<f32>() / n as f32;
    let nf = n as f32;
    let raw_k = m4 / var.powi(2) - 3.0;
    raw_k * ((nf + 1.0) * (nf - 1.0)) / ((nf - 2.0) * (nf - 3.0))
}

fn quantile(v: &[f32], q: f64) -> f32 {
    let mut sorted: Vec<f32> = v.iter().copied().filter(|x| x.is_finite()).collect();
    if sorted.is_empty() {
        return 0.0;
    }
    sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    let n = sorted.len();
    let pos = q * (n as f64 - 1.0);
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    if lo == hi {
        sorted[lo.min(n - 1)]
    } else {
        let frac = pos - lo as f64;
        sorted[lo] * (1.0 - frac as f32) + sorted[hi] * (frac as f32)
    }
}

/// 相关：有效配对 <2 时返回 0.0。
fn corr(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    if n < 2 {
        return 0.0;
    }
    let pairs: Vec<(f32, f32)> = (0..n)
        .filter(|&i| a[i].is_finite() && b[i].is_finite())
        .map(|i| (a[i], b[i]))
        .collect();
    let m = pairs.len();
    if m < 2 {
        return 0.0;
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
        0.0
    }
}

/// 带滞后的相关：corr(a[0..n-lag], b[lag..n])。
fn lagged_corr(a: &[f32], b: &[f32], lag: usize) -> f32 {
    let n = a.len().min(b.len());
    if n <= lag + 1 {
        return 0.0;
    }
    corr(&a[..n - lag], &b[lag..n])
}

/// R/S 法 Hurst 指数；样本不足返回 0.5（随机游走中性值）。
fn hurst_rs(v: &[f32]) -> f32 {
    let valid: Vec<f32> = v.iter().copied().filter(|x| x.is_finite()).collect();
    let n = valid.len();
    if n < 4 {
        return 0.5;
    }
    let m = valid.iter().sum::<f32>() / n as f32;
    let dev: Vec<f32> = valid.iter().map(|x| x - m).collect();
    let mut cum = vec![0.0f32; n];
    cum[0] = dev[0];
    for i in 1..n {
        cum[i] = cum[i - 1] + dev[i];
    }
    let range = cum.iter().copied().fold(f32::NEG_INFINITY, f32::max)
        - cum.iter().copied().fold(f32::INFINITY, f32::min);
    let s = std_dev(&valid);
    if s <= 0.0 || range <= 0.0 {
        return 0.5;
    }
    let rs = range / s;
    if rs <= 0.0 {
        return 0.5;
    }
    let h = rs.ln() / (n as f32).ln();
    h.clamp(0.0, 1.0)
}

fn gini(v: &[f32]) -> f32 {
    let mut valid: Vec<f32> = v
        .iter()
        .copied()
        .filter(|x| x.is_finite() && *x >= 0.0)
        .collect();
    let n = valid.len();
    if n < 2 {
        return 0.0;
    }
    valid.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    let sum: f32 = valid.iter().copied().sum();
    if sum <= 0.0 {
        return 0.0;
    }
    let weighted_sum: f32 = valid
        .iter()
        .enumerate()
        .map(|(i, &x)| (i as f32 + 1.0) * x)
        .sum();
    let nf = n as f32;
    (2.0 * weighted_sum) / (nf * sum) - (nf + 1.0) / nf
}

fn permutation_entropy(perm: &[usize]) -> f32 {
    let n = perm.len();
    if n < 2 {
        return 0.0;
    }
    let mut ranks = vec![0usize; n];
    for &p in perm {
        if p >= 1 && p <= n {
            ranks[p - 1] += 1;
        }
    }
    let total = n as f32;
    let mut h = 0.0f32;
    for &c in &ranks {
        if c > 0 {
            let p = c as f32 / total;
            h -= p * p.ln();
        }
    }
    let max_h = (n as f32).ln();
    if max_h > 0.0 {
        h / max_h
    } else {
        0.0
    }
}

// ============================================================================
// 指数衰减拟合 —— 始终返回有限值
// ============================================================================

/// 拟合 |e(t)| = |e0| * exp(-t/tau)，对 ln|e| 做 OLS。
/// 返回 (tau, e0, resid_std, r2)，全部有限（样本不足用退化值）。
fn fit_exp_decay(t: &[f32], e: &[f32]) -> (f32, f32, f32, f32) {
    let n = t.len().min(e.len());
    let mut xs: Vec<f32> = Vec::with_capacity(n);
    let mut ys: Vec<f32> = Vec::with_capacity(n);
    let mut e0_sign = 1.0f32;
    for i in 0..n {
        let ei = e[i];
        if ei.is_finite() && ei.abs() > 1e-10 {
            if xs.is_empty() {
                e0_sign = if ei >= 0.0 { 1.0 } else { -1.0 };
            }
            xs.push(t[i]);
            ys.push(ei.abs().ln());
        }
    }
    let m = xs.len();
    if m == 0 {
        return (240.0, 0.0, 0.0, 0.0);
    }
    if m == 1 {
        return (240.0, e0_sign * ys[0].exp(), 0.0, 0.0);
    }
    let mf = m as f32;
    let mx = xs.iter().sum::<f32>() / mf;
    let my = ys.iter().sum::<f32>() / mf;
    let mut sxy = 0.0f32;
    let mut sxx = 0.0f32;
    let mut syy = 0.0f32;
    for i in 0..m {
        let dx = xs[i] - mx;
        let dy = ys[i] - my;
        sxy += dx * dy;
        sxx += dx * dx;
        syy += dy * dy;
    }
    if sxx < 1e-20 {
        return (240.0, e0_sign * my.exp(), 0.0, 0.0);
    }
    let slope = sxy / sxx;
    let intercept = my - slope * mx;
    let tau = if slope.abs() > 1e-10 {
        (-1.0 / slope).clamp(0.5, 240.0)
    } else {
        240.0
    };
    let e0 = e0_sign * intercept.exp();
    let mut resid_sq = 0.0f32;
    for i in 0..m {
        let pred = intercept + slope * xs[i];
        let resid = ys[i] - pred;
        resid_sq += resid * resid;
    }
    let resid_std = (resid_sq / mf).sqrt();
    let r2 = if syy > 1e-20 {
        (1.0 - resid_sq / syy).clamp(0.0, 1.0)
    } else {
        0.0
    };
    (tau, e0, resid_std, r2)
}

// ============================================================================
// Teacher 信号提取
// ============================================================================

fn extract_auction_teacher(
    market_raw: &[MarketRecord],
    trade: &[TradeRecord],
    yclose: f32,
) -> TeacherSignals {
    let mut auct_mids: Vec<(f32, f32)> = Vec::new();
    for m in market_raw {
        let day_sec = m.time_sec % 86400.0;
        if day_sec < AUCT_START || day_sec > AUCT_END {
            continue;
        }
        let ask1 = m.ask_prcs[0];
        let bid1 = m.bid_prcs[0];
        if ask1 > 0.0 && bid1 > 0.0 {
            auct_mids.push((day_sec, (ask1 + bid1) / 2.0));
        }
    }
    if auct_mids.len() < MIN_AUCT_RECORDS {
        return TeacherSignals {
            p_auct: f32::NAN,
            v_auct: f32::NAN,
            sigma_auct: f32::NAN,
            delta_auct: f32::NAN,
            delta_yc: f32::NAN,
        };
    }
    auct_mids.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    let mut p_auct = f32::NAN;
    for &(ds, mid) in auct_mids.iter().rev() {
        if ds >= 33840.0 {
            p_auct = mid;
            break;
        }
    }
    if !p_auct.is_finite() {
        p_auct = auct_mids.last().unwrap().1;
    }

    let sigma_mids: Vec<f32> = auct_mids
        .iter()
        .filter(|(ds, _)| *ds >= AUCT_SIGMA_START)
        .map(|(_, mid)| *mid)
        .collect();
    let sigma_auct = if sigma_mids.len() >= 3 {
        let s = std_dev(&sigma_mids);
        if s > 0.0 {
            s
        } else {
            f32::NAN
        }
    } else {
        f32::NAN
    };

    let mid_open = auct_mids
        .iter()
        .find(|(ds, _)| *ds >= AUCT_SIGMA_START)
        .map(|(_, mid)| *mid);
    let delta_auct = match mid_open {
        Some(mo) if mo > 0.0 && p_auct > 0.0 => (p_auct / mo).ln(),
        _ => 0.0, // 降级：中性
    };

    let mut v_auct = 0.0f32;
    let mut found = false;
    for t in trade {
        let day_sec = t.time_sec % 86400.0;
        if day_sec >= SESSION_START && day_sec < SESSION_START + 5.0 {
            v_auct += t.volume;
            found = true;
        }
    }
    if !found {
        for t in trade {
            let day_sec = t.time_sec % 86400.0;
            if day_sec >= SESSION_START && day_sec < SESSION_START + 30.0 {
                v_auct += t.volume;
                found = true;
            }
        }
    }
    let v_auct = if found { v_auct } else { f32::NAN };

    let delta_yc = if yclose.is_finite() && yclose > 0.0 && p_auct > 0.0 {
        (p_auct / yclose).ln()
    } else {
        f32::NAN
    };

    TeacherSignals {
        p_auct,
        v_auct,
        sigma_auct,
        delta_auct,
        delta_yc,
    }
}

// ============================================================================
// Student 数据（1 分钟重采样）
// ============================================================================

fn resample_minutes(market: &[MarketRecord], trade: &[TradeRecord]) -> MinuteData {
    let n_min = (((SESSION_END - SESSION_START) / SEC_PER_MIN).ceil() as usize).max(1);

    let mut price = vec![f32::NAN; n_min];
    let mut volume = vec![0.0f32; n_min];
    let mut active_buy = vec![0.0f32; n_min];
    let mut active_sell = vec![0.0f32; n_min];
    let mut bid_start = vec![f32::NAN; n_min];
    let mut ask_start = vec![f32::NAN; n_min];
    let mut bid_end = vec![f32::NAN; n_min];
    let mut ask_end = vec![f32::NAN; n_min];

    for m in market {
        let day_sec = m.time_sec % 86400.0;
        if day_sec < SESSION_START || day_sec >= SESSION_END {
            continue;
        }
        let idx = ((day_sec - SESSION_START) / SEC_PER_MIN) as usize;
        if idx >= n_min {
            continue;
        }
        if m.last_prc > 0.0 {
            price[idx] = m.last_prc;
        }
        if bid_start[idx].is_nan() {
            bid_start[idx] = m.total_bid_vol;
            ask_start[idx] = m.total_ask_vol;
        }
        bid_end[idx] = m.total_bid_vol;
        ask_end[idx] = m.total_ask_vol;
    }

    let mut last = f32::NAN;
    for i in 0..n_min {
        if price[i].is_finite() {
            last = price[i];
        } else {
            price[i] = last;
        }
    }

    for t in trade {
        let day_sec = t.time_sec % 86400.0;
        if day_sec < SESSION_START || day_sec >= SESSION_END {
            continue;
        }
        let idx = ((day_sec - SESSION_START) / SEC_PER_MIN) as usize;
        if idx >= n_min {
            continue;
        }
        volume[idx] += t.volume;
        match t.flag {
            66 => active_buy[idx] += t.volume,
            83 => active_sell[idx] += t.volume,
            _ => {}
        }
    }

    let mut abr = vec![0.0f32; n_min];
    let mut ofi = vec![0.0f32; n_min];
    for i in 0..n_min {
        let total = active_buy[i] + active_sell[i];
        if total > 0.0 {
            abr[i] = active_buy[i] / total;
        }
        if bid_start[i].is_finite() && bid_end[i].is_finite() {
            ofi[i] = (bid_end[i] - bid_start[i]) - (ask_end[i] - ask_start[i]);
        }
    }

    let mut logret = vec![0.0f32; n_min];
    for i in 1..n_min {
        if price[i].is_finite() && price[i - 1].is_finite() && price[i - 1] > 0.0 {
            logret[i] = (price[i] / price[i - 1]).ln();
        }
    }

    MinuteData {
        n: n_min,
        price,
        volume,
        abr,
        ofi,
        logret,
    }
}

// ============================================================================
// 5 维 epsilon 序列 —— 只要 P_auct 可用，全部 5 维产出有限值
// ============================================================================

struct EpsilonSeries {
    eps_price: Vec<f32>,
    eps_vol: Vec<f32>,
    eps_belief: Vec<f32>,
    eps_sigma: Vec<f32>,
    eps_yc: Vec<f32>,
}

fn compute_epsilons(teacher: &TeacherSignals, md: &MinuteData) -> EpsilonSeries {
    let n = md.n;
    let p_auct = teacher.p_auct;
    let delta_sign = if teacher.delta_auct.is_finite() && teacher.delta_auct != 0.0 {
        teacher.delta_auct.signum()
    } else {
        1.0 // 降级：正方向
    };
    let delta_yc = teacher.delta_yc;

    // 降级归一化基准
    let vol_norm = if teacher.v_auct.is_finite() && teacher.v_auct > 0.0 {
        teacher.v_auct
    } else {
        // 降级：用全天平均分钟成交量
        let m = mean(&md.volume);
        if m > 0.0 {
            m
        } else {
            1.0
        }
    };

    // eps_sigma 降级归一化基准
    let raw_rolling_std: Vec<f32> = (0..n)
        .map(|i| {
            if i + 1 < 5 {
                std_dev(&md.logret[..=i])
            } else {
                std_dev(&md.logret[i + 1 - 5..=i])
            }
        })
        .collect();
    let sigma_norm = if teacher.sigma_auct.is_finite() && teacher.sigma_auct > 0.0 {
        teacher.sigma_auct
    } else {
        // 降级：用 rolling_std 的均值
        let m = mean(&raw_rolling_std);
        if m > 0.0 {
            m
        } else {
            1.0
        }
    };

    // eps_yc 降级基准
    let yc_scale = if delta_yc.is_finite() && delta_yc.abs() > 1e-8 {
        delta_yc.abs()
    } else {
        1.0 // 降级：不缩放
    };
    let yclose_fallback = if delta_yc.is_finite() && p_auct > 0.0 {
        p_auct / delta_yc.exp() // 还原 yclose
    } else {
        p_auct // 降级：用 P_auct 作参考
    };

    let mut eps_price = vec![0.0f32; n];
    let mut eps_vol = vec![0.0f32; n];
    let mut eps_belief = vec![0.0f32; n];
    let mut eps_sigma = vec![0.0f32; n];
    let mut eps_yc = vec![0.0f32; n];

    for i in 0..n {
        let p = md.price[i];
        if p.is_finite() && p > 0.0 && p_auct > 0.0 {
            let lr = (p / p_auct).ln();
            eps_price[i] = lr;
            eps_belief[i] = delta_sign * lr;
            if yclose_fallback > 0.0 {
                eps_yc[i] = (p / yclose_fallback).ln() / yc_scale;
            }
        }
        eps_vol[i] = md.volume[i] / vol_norm;
        eps_sigma[i] = raw_rolling_std[i] / sigma_norm;
    }

    EpsilonSeries {
        eps_price,
        eps_vol,
        eps_belief,
        eps_sigma,
        eps_yc,
    }
}

// ============================================================================
// 分支 A：单维曲线拟合（100 因子）
// ============================================================================

mod branch_a {
    use super::*;

    pub const COUNT: usize = 100;

    pub fn compute(teacher: &TeacherSignals, md: &MinuteData) -> Vec<f32> {
        let mut out = Vec::with_capacity(COUNT);
        let p_auct = teacher.p_auct;

        let eps: Vec<f32> = if p_auct > 0.0 {
            md.price
                .iter()
                .map(|&p| if p > 0.0 { (p / p_auct).ln() } else { 0.0 })
                .collect()
        } else {
            vec![0.0; md.n]
        };
        let t_axis: Vec<f32> = (0..md.n).map(|i| i as f32).collect();

        // (a) 基础拟合：7 窗口 × 5 = 35
        for &k in &WINDOWS_A {
            let end = k.min(md.n);
            let (tau, eps0, resid_std, r2) = fit_exp_decay(&t_axis[..end], &eps[..end]);
            let hurst = hurst_rs(&eps[..end]);
            out.push(tau);
            out.push(eps0);
            out.push(resid_std);
            out.push(r2);
            out.push(hurst);
        }

        // (b) 统计降维：15
        out.push(mean(&eps));
        out.push(std_dev(&eps));
        out.push(
            eps.iter()
                .copied()
                .filter(|x| x.is_finite())
                .fold(f32::INFINITY, f32::min),
        );
        out.push(
            eps.iter()
                .copied()
                .filter(|x| x.is_finite())
                .fold(f32::NEG_INFINITY, f32::max),
        );
        out.push(skewness(&eps));
        out.push(kurtosis(&eps));
        out.push(quantile(&eps, 0.05));
        out.push(quantile(&eps, 0.10));
        out.push(quantile(&eps, 0.25));
        out.push(quantile(&eps, 0.50));
        out.push(quantile(&eps, 0.75));
        out.push(quantile(&eps, 0.90));
        out.push(quantile(&eps, 0.95));
        let q75 = quantile(&eps, 0.75);
        let q25 = quantile(&eps, 0.25);
        out.push(q75 - q25); // iqr
        let sd = std_dev(&eps);
        out.push(if mean(&eps).abs() > 1e-10 {
            sd / mean(&eps).abs()
        } else {
            0.0
        }); // cv

        // (c) 突破特征：6 阈值 × 3 = 18
        let end_30 = 30.min(md.n);
        for &thr in &THRESHOLDS_A {
            let upper = p_auct * (1.0 + thr);
            let lower = p_auct * (1.0 - thr);
            let mut cnt = 0.0f32;
            let mut first_time = 0.0f32;
            let mut prev_above = false;
            let mut prev_below = false;
            for i in 0..end_30 {
                let p = md.price[i];
                if !p.is_finite() || p <= 0.0 {
                    continue;
                }
                let above = p > upper;
                let below = p < lower;
                if (above && !prev_above) || (below && !prev_below) {
                    cnt += 1.0;
                    if cnt == 1.0 {
                        first_time = i as f32;
                    }
                }
                prev_above = above;
                prev_below = below;
            }
            let rebound = if cnt > 0.0 {
                let ft = first_time as usize;
                let end_r = (ft + 5).min(md.n);
                let mut max_dev = 0.0f32;
                for i in ft..end_r {
                    let p = md.price[i];
                    if p.is_finite() && p_auct > 0.0 {
                        max_dev = max_dev.max(((p - p_auct) / p_auct).abs());
                    }
                }
                max_dev
            } else {
                0.0
            };
            out.push(cnt);
            out.push(first_time);
            out.push(rebound);
        }

        // (d) 收敛速度：7 窗口 × 2 = 14
        for &k in &WINDOWS_A {
            let end = k.min(md.n);
            let (tau, _, _, _) = fit_exp_decay(&t_axis[..end], &eps[..end]);
            // 收敛速度 = 1/tau
            let speed = if tau > 0.0 { 1.0 / tau } else { 0.0 };
            out.push(speed);
            // half_life ratio = tau_full / tau_k（越小越快收敛到局部模式）
            let (tau_full, _, _, _) = fit_exp_decay(&t_axis, &eps);
            let ratio = if tau > 0.0 { tau_full / tau } else { 0.0 };
            out.push(ratio);
        }

        // (e) 路径形态：7
        let max_dev = eps
            .iter()
            .copied()
            .filter(|x| x.is_finite())
            .fold(0.0f32, |a, x| a.max(x.abs()));
        out.push(max_dev);
        let abs_area: f32 = eps
            .iter()
            .map(|x| if x.is_finite() { x.abs() } else { 0.0 })
            .sum::<f32>()
            / md.n as f32;
        out.push(abs_area);
        let mut sign_changes = 0;
        let mut prev_sign: i8 = 0;
        for &x in &eps {
            if !x.is_finite() || x == 0.0 {
                continue;
            }
            let s: i8 = if x > 0.0 { 1 } else { -1 };
            if prev_sign != 0 && s != prev_sign {
                sign_changes += 1;
            }
            prev_sign = s;
        }
        out.push(sign_changes as f32);
        // reversal count（穿越零点的次数）
        out.push(sign_changes as f32);
        // max run（最长连续同号段）
        let mut max_run = 0;
        let mut cur_run = 0;
        let mut prev_s: i8 = 0;
        for &x in &eps {
            if !x.is_finite() || x == 0.0 {
                continue;
            }
            let s: i8 = if x > 0.0 { 1 } else { -1 };
            if s == prev_s {
                cur_run += 1;
            } else {
                cur_run = 1;
            }
            max_run = max_run.max(cur_run);
            prev_s = s;
        }
        out.push(max_run as f32);
        // trend strength
        out.push(corr(&t_axis, &eps));
        // pct above zero
        let above = eps.iter().filter(|x| x.is_finite() && **x > 0.0).count();
        out.push(above as f32 / md.n as f32);

        // (f) 自相关：6
        out.push(lagged_corr(&eps, &eps, 1));
        out.push(lagged_corr(&eps, &eps, 5));
        out.push(lagged_corr(&eps, &eps, 15));
        let ac1 = lagged_corr(&eps, &eps, 1);
        out.push(ac1.abs());
        // autocorr decay = ac1 - ac5
        let ac5 = lagged_corr(&eps, &eps, 5);
        out.push((ac1 - ac5).abs());
        // cross-zero rate
        out.push(sign_changes as f32 / md.n as f32);

        // (g) 波动率结构：5
        let early_vol = std_dev(&eps[..30.min(md.n)]);
        let late_start = (md.n as f32 * 0.8) as usize;
        let late_vol = std_dev(&eps[late_start.min(md.n)..]);
        out.push(early_vol);
        out.push(late_vol);
        out.push(if late_vol > 1e-10 {
            early_vol / late_vol
        } else {
            0.0
        });
        out.push(skewness(&eps[..30.min(md.n)]));
        let early_var = std_dev(&eps[..15.min(md.n)]);
        let mid_var = std_dev(&eps[15.min(md.n)..60.min(md.n)]);
        out.push(if mid_var > 1e-10 {
            early_var / mid_var
        } else {
            0.0
        });

        assert_eq!(out.len(), COUNT);
        out
    }

    pub fn names() -> Vec<String> {
        let mut out = Vec::with_capacity(COUNT);
        let metrics = ["tau", "eps0", "resid_std", "r2", "hurst_eps"];
        for &k in &WINDOWS_A {
            for m in &metrics {
                out.push(format!("dA_{}_k{}", m, k));
            }
        }
        for s in &[
            "eps_mean", "eps_std", "eps_min", "eps_max", "eps_skew", "eps_kurt", "eps_q05",
            "eps_q10", "eps_q25", "eps_q50", "eps_q75", "eps_q90", "eps_q95", "eps_iqr", "eps_cv",
        ] {
            out.push(format!("dA_{}", s));
        }
        let tl = ["01", "03", "05", "1", "2", "3"];
        for t in &tl {
            out.push(format!("dA_break_cnt_thr{}", t));
            out.push(format!("dA_break_first_time_thr{}", t));
            out.push(format!("dA_break_rebound_thr{}", t));
        }
        for &k in &WINDOWS_A {
            out.push(format!("dA_conv_speed_k{}", k));
            out.push(format!("dA_tau_ratio_k{}", k));
        }
        for s in &[
            "max_dev",
            "abs_area",
            "sign_changes",
            "reversal_count",
            "max_run",
            "trend_strength",
            "pct_above",
        ] {
            out.push(format!("dA_{}", s));
        }
        for s in &[
            "ac1",
            "ac5",
            "ac15",
            "ac1_abs",
            "ac_decay",
            "cross_zero_rate",
        ] {
            out.push(format!("dA_{}", s));
        }
        for s in &[
            "vol_early",
            "vol_late",
            "vol_ratio",
            "early_skew",
            "var_ratio",
        ] {
            out.push(format!("dA_{}", s));
        }
        assert_eq!(out.len(), COUNT);
        out
    }
}

// ============================================================================
// 分支 B：多维独立遗忘（140 因子）
// ============================================================================

mod branch_b {
    use super::*;

    pub const COUNT: usize = 140;

    pub fn compute(eps: &EpsilonSeries, md: &MinuteData) -> Vec<f32> {
        let mut out = Vec::with_capacity(COUNT);
        let n = md.n;
        let t_axis: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let dim_series: [&[f32]; 5] = [
            &eps.eps_price,
            &eps.eps_vol,
            &eps.eps_belief,
            &eps.eps_sigma,
            &eps.eps_yc,
        ];

        // (a) 每维基础：5 × 10 = 50
        let mut taus: Vec<f32> = Vec::with_capacity(5);
        for di in 0..5 {
            let series = dim_series[di];
            let (tau, eps0, _, _) = fit_exp_decay(&t_axis, series);
            let hurst = hurst_rs(series);
            let half_life = if tau > 0.0 { tau * 0.6931472 } else { 0.0 };
            let r1 = if n > 1 { series[1].abs() } else { 0.0 };
            let r15 = if n > 15 { series[15].abs() } else { 0.0 };
            let r60 = if n > 60 { series[60].abs() } else { 0.0 };
            let dmean = mean(series);
            let dstd = std_dev(series);
            let dmax = series
                .iter()
                .copied()
                .filter(|x| x.is_finite())
                .fold(0.0f32, |a, x| a.max(x.abs()));
            taus.push(tau);
            out.push(tau);
            out.push(eps0);
            out.push(hurst);
            out.push(half_life);
            out.push(r1);
            out.push(r15);
            out.push(r60);
            out.push(dmean);
            out.push(dstd);
            out.push(dmax);
        }

        // (b) 细化窗口：5 × 5 = 25
        for di in 0..5 {
            let series = dim_series[di];
            for &k in &WINDOWS_B {
                let end = k.min(n);
                let (tau_k, _, _, _) = fit_exp_decay(&t_axis[..end], &series[..end]);
                out.push(tau_k);
            }
        }

        // (c) 跨维统计：10
        out.push(mean(&taus));
        out.push(std_dev(&taus));
        let tmin = taus.iter().copied().fold(f32::INFINITY, f32::min);
        let tmax = taus.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        out.push(tmin);
        out.push(tmax);
        out.push(tmax - tmin);
        let tm = mean(&taus);
        out.push(if tm.abs() > 1e-10 {
            std_dev(&taus) / tm.abs()
        } else {
            0.0
        });
        out.push(skewness(&taus));
        out.push(kurtosis(&taus));
        out.push(gini(
            &taus.iter().copied().map(|t| t.abs()).collect::<Vec<_>>(),
        ));
        out.push(quantile(&taus, 0.5));

        // (d) 遗忘顺序：6
        let mut tau_pairs: Vec<(f32, usize)> =
            taus.iter().enumerate().map(|(i, &t)| (t, i + 1)).collect();
        tau_pairs.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        out.push(tau_pairs[0].1 as f32); // first forgotten
        out.push(tau_pairs[4].1 as f32); // last forgotten
        let perm: Vec<usize> = tau_pairs.iter().map(|(_, d)| *d).collect();
        out.push(permutation_entropy(&perm));
        out.push(if tmin.abs() > 1e-10 { tmax / tmin } else { 0.0 }); // extremeness
        out.push(gini(
            &taus.iter().copied().map(|t| t.abs()).collect::<Vec<_>>(),
        )); // balance
        let dominance = if tmax > 1e-10 {
            (tmax - tmin) / tmax
        } else {
            0.0
        };
        out.push(dominance);

        // (e) 跨维相关：C(5,2)=10 pairs × 3 = 30
        for i in 0..5 {
            for j in (i + 1)..5 {
                out.push(corr(dim_series[i], dim_series[j]));
                out.push(lagged_corr(dim_series[i], dim_series[j], 5));
                let ci = corr(dim_series[i], dim_series[j]);
                let ci5 = lagged_corr(dim_series[i], dim_series[j], 5);
                out.push((ci - ci5).abs()); // spread
            }
        }

        // (f) 残差扩展：5 × 3 = 15
        for di in 0..5 {
            let series = dim_series[di];
            let r30 = if n > 30 { series[30].abs() } else { 0.0 };
            let r120 = if n > 120 { series[120].abs() } else { 0.0 };
            let rend = series[n - 1].abs();
            out.push(r30);
            out.push(r120);
            out.push(rend);
        }

        // (g) 多维复合：4
        let tau_pv = if taus[1] > 0.0 {
            taus[0] / taus[1]
        } else {
            0.0
        };
        let tau_pb = if taus[2] > 0.0 {
            taus[0] / taus[2]
        } else {
            0.0
        };
        let tau_ps = if taus[3] > 0.0 {
            taus[0] / taus[3]
        } else {
            0.0
        };
        let tau_weighted = (taus[0] * 2.0 + taus[1] + taus[2] + taus[3] + taus[4]) / 6.0;
        out.push(tau_pv);
        out.push(tau_pb);
        out.push(tau_ps);
        out.push(tau_weighted);

        assert_eq!(out.len(), COUNT);
        out
    }

    pub fn names() -> Vec<String> {
        let mut out = Vec::with_capacity(COUNT);
        let dims = ["price", "vol", "belief", "sigma", "yc"];
        for d in &dims {
            for m in &[
                "tau",
                "eps0",
                "hurst",
                "half_life",
                "residual_1min",
                "residual_15min",
                "residual_60min",
                "mean",
                "std",
                "max_abs",
            ] {
                out.push(format!("dB_{}_{}", m, d));
            }
        }
        for d in &dims {
            for &k in &WINDOWS_B {
                out.push(format!("dB_tau_{}_k{}", d, k));
            }
        }
        for s in &[
            "tau_mean",
            "tau_std",
            "tau_min",
            "tau_max",
            "tau_range",
            "tau_cv",
            "tau_skew",
            "tau_kurt",
            "tau_gini",
            "tau_median",
        ] {
            out.push(format!("dB_{}", s));
        }
        for s in &[
            "first_forgotten_dim",
            "last_forgotten_dim",
            "forgetting_order_entropy",
            "dim_extremeness",
            "dim_balance",
            "dominance_ratio",
        ] {
            out.push(format!("dB_{}", s));
        }
        for i in 0..5 {
            for j in (i + 1)..5 {
                out.push(format!("dB_corr_d{}_d{}", i + 1, j + 1));
                out.push(format!("dB_lagcorr_d{}_d{}", i + 1, j + 1));
                out.push(format!("dB_corr_spread_d{}_d{}", i + 1, j + 1));
            }
        }
        for d in &dims {
            for m in &["residual_30min", "residual_120min", "residual_end"] {
                out.push(format!("dB_{}_{}", m, d));
            }
        }
        for s in &[
            "tau_ratio_price_vol",
            "tau_ratio_price_belief",
            "tau_ratio_price_sigma",
            "tau_weighted_mean",
        ] {
            out.push(format!("dB_{}", s));
        }
        assert_eq!(out.len(), COUNT);
        out
    }
}

// ============================================================================
// 分支 D：分阶段衰减（240 因子）
// ============================================================================

mod branch_d {
    use super::*;

    pub const COUNT: usize = 240;

    // 20 种 teacher 组合，维度索引 P=0, V=1, σ=2, Δ=3, δ=4
    const COMBOS: [&[usize]; 20] = [
        &[0],
        &[1],
        &[2],
        &[3],
        &[4],
        &[0, 3],
        &[0, 4],
        &[1, 3],
        &[1, 4],
        &[2, 3],
        &[0, 1, 3],
        &[0, 1, 4],
        &[0, 3, 4],
        &[1, 3, 4],
        &[0, 1, 2],
        &[0, 1, 3, 4],
        &[0, 1, 2, 3],
        &[0, 2, 3, 4],
        &[1, 2, 3, 4],
        &[0, 1, 2, 3, 4],
    ];

    pub fn compute(eps: &EpsilonSeries, md: &MinuteData) -> Vec<f32> {
        let mut out = Vec::with_capacity(COUNT);
        let n = md.n;
        let dim_signals: [&[f32]; 5] = [
            &eps.eps_price,
            &eps.eps_vol,
            &eps.eps_sigma,
            &eps.eps_belief,
            &eps.eps_yc,
        ];

        let mut all_ics: Vec<[f32; 6]> = Vec::with_capacity(20);

        for combo in &COMBOS {
            let mut combined = vec![0.0f32; n];
            for i in 0..n {
                let mut sum = 0.0f32;
                for &di in combo.iter() {
                    sum += dim_signals[di][i];
                }
                combined[i] = sum / combo.len() as f32;
            }
            let mut ics = [0.0f32; 6];
            let mut seg_starts = [0usize; 6];
            for si in 1..6 {
                seg_starts[si] = SEG_ENDS_D[si - 1];
            }
            for si in 0..6 {
                let s = seg_starts[si];
                let e = SEG_ENDS_D[si].min(n);
                if e <= s + 1 {
                    ics[si] = 0.0;
                    continue;
                }
                let base_p = md.price[s];
                let mut cr = vec![0.0f32; e - s];
                if base_p.is_finite() && base_p > 0.0 {
                    for j in s..e {
                        if md.price[j].is_finite() && md.price[j] > 0.0 {
                            cr[j - s] = (md.price[j] / base_p).ln();
                        }
                    }
                }
                ics[si] = corr(&combined[s..e], &cr);
            }
            all_ics.push(ics);
        }

        // (a) IC 度量：6 × 20 = 120
        for ics in &all_ics {
            for &ic in ics {
                out.push(ic);
            }
        }

        // (b) 衰减曲线：20 × 6 = 120
        for ics in &all_ics {
            let ts: Vec<f32> = (0..6).map(|i| SEG_MIDS_D[i]).collect();
            let (tau, _, _, _) = fit_exp_decay(&ts, ics);
            let end_over_start = if ics[0].abs() > 1e-10 {
                ics[5] / ics[0]
            } else {
                0.0
            };
            let mut sign_changes = 0;
            let mut prev_sign: i8 = 0;
            for &ic in ics {
                let s: i8 = if ic > 1e-10 {
                    1
                } else if ic < -1e-10 {
                    -1
                } else {
                    0
                };
                if prev_sign != 0 && s != 0 && s != prev_sign {
                    sign_changes += 1;
                }
                if s != 0 {
                    prev_sign = s;
                }
            }
            let shape = if sign_changes == 0 {
                0.0
            } else if sign_changes <= 2 {
                1.0
            } else {
                2.0
            };
            let slope = (ics[5] - ics[0]) / 5.0;
            let area: f32 = ics.iter().sum::<f32>() / 6.0;
            // curvature = mean of second differences
            let mut sec_diff = 0.0f32;
            for i in 1..5 {
                sec_diff += ics[i + 1] - 2.0 * ics[i] + ics[i - 1];
            }
            let curvature = sec_diff / 4.0;
            out.push(tau);
            out.push(end_over_start);
            out.push(shape);
            out.push(slope);
            out.push(area);
            out.push(curvature);
        }

        assert_eq!(out.len(), COUNT);
        out
    }

    pub fn names() -> Vec<String> {
        let mut out = Vec::with_capacity(COUNT);
        for ti in 1..=6 {
            for cj in 1..=20 {
                out.push(format!("dD_IC_T{}_C{}", ti, cj));
            }
        }
        for cj in 1..=20 {
            out.push(format!("dD_decay_tau_C{}", cj));
            out.push(format!("dD_end_over_start_C{}", cj));
            out.push(format!("dD_shape_C{}", cj));
            out.push(format!("dD_slope_C{}", cj));
            out.push(format!("dD_ic_area_C{}", cj));
            out.push(format!("dD_curvature_C{}", cj));
        }
        assert_eq!(out.len(), COUNT);
        out
    }
}

// ============================================================================
// 分支 E：双层 Hawkes 蒸馏（120 因子）
// ============================================================================

mod branch_e {
    use super::*;

    pub const COUNT: usize = 121;
    const L1_END: f32 = 300.0;
    const L2_END: f32 = SESSION_END - SESSION_START;

    #[derive(Clone, Copy)]
    struct HawkesParams {
        lambda: f32,
        eta: f32,
    }

    fn estimate_hawkes(event_times: &[f32], span: f32) -> HawkesParams {
        let n = event_times.len();
        if n == 0 || span <= 0.0 {
            return HawkesParams {
                lambda: 0.0,
                eta: 0.0,
            };
        }
        let lambda = n as f32 / span;
        let mut total_trig = 0usize;
        let mut lo = 0usize;
        for hi in 0..n {
            while event_times[hi] - event_times[lo] > HAWKES_TRIGGER_WIN {
                lo += 1;
            }
            // [lo, hi] 范围内的事件都在 hi 的触发窗口内
            total_trig += hi - lo; // 不含 hi 自身
        }
        let eta = total_trig as f32 / n as f32;
        HawkesParams { lambda, eta }
    }

    pub fn compute(trade: &[TradeRecord]) -> Vec<f32> {
        let mut out = Vec::with_capacity(COUNT);

        let mut buy_events: Vec<(f32, f32)> = Vec::new();
        let mut sell_events: Vec<(f32, f32)> = Vec::new();
        for t in trade {
            let day_sec = t.time_sec % 86400.0;
            if day_sec < SESSION_START || day_sec >= SESSION_END {
                continue;
            }
            let offset = day_sec - SESSION_START;
            match t.flag {
                66 => buy_events.push((offset, t.volume)),
                83 => sell_events.push((offset, t.volume)),
                _ => {}
            }
        }

        let buy_thr = compute_thresholds(&buy_events);
        let sell_thr = compute_thresholds(&sell_events);

        // 6 种事件：buy_10, sell_10, buy_5, sell_5, buy_1, sell_1
        let event_defs: [(bool, f32); 6] = [
            (true, buy_thr[0]),
            (false, sell_thr[0]),
            (true, buy_thr[1]),
            (false, sell_thr[1]),
            (true, buy_thr[2]),
            (false, sell_thr[2]),
        ];
        let event_times: Vec<Vec<f32>> = event_defs
            .iter()
            .map(|&(is_buy, thr)| {
                let src = if is_buy { &buy_events } else { &sell_events };
                src.iter()
                    .filter(|(_, v)| *v >= thr)
                    .map(|(t, _)| *t)
                    .collect()
            })
            .collect();

        // (a) Hawkes 参数：6 × 4 = 24
        let mut h1 = Vec::with_capacity(6);
        let mut h2 = Vec::with_capacity(6);
        for times in &event_times {
            let l1: Vec<f32> = times.iter().copied().filter(|&t| t < L1_END).collect();
            let l2: Vec<f32> = times
                .iter()
                .copied()
                .filter(|&t| t >= L1_END && t < L2_END)
                .collect();
            let p1 = estimate_hawkes(&l1, L1_END);
            let p2 = estimate_hawkes(&l2, L2_END - L1_END);
            out.push(p1.lambda);
            out.push(p1.eta);
            out.push(p2.lambda);
            out.push(p2.eta);
            h1.push(p1);
            h2.push(p2);
        }

        // (b) 衍生 per-event：6 × 5 = 30
        for i in 0..6 {
            let absorption = h1[i].lambda * h1[i].eta;
            let propagation = h2[i].lambda * h2[i].eta;
            let learning = if propagation.abs() > 1e-10 {
                absorption / propagation
            } else {
                0.0
            };
            let eta_decay = h1[i].eta - h2[i].eta;
            let lambda_ratio = if h2[i].lambda.abs() > 1e-10 {
                h1[i].lambda / h2[i].lambda
            } else {
                0.0
            };
            out.push(absorption);
            out.push(propagation);
            out.push(learning);
            out.push(eta_decay);
            out.push(lambda_ratio);
        }

        // (c) 跨事件聚合：20
        let eta1_buy = [h1[0].eta, h1[2].eta, h1[4].eta];
        let eta1_sell = [h1[1].eta, h1[3].eta, h1[5].eta];
        let eta2_buy = [h2[0].eta, h2[2].eta, h2[4].eta];
        let eta2_sell = [h2[1].eta, h2[3].eta, h2[5].eta];
        out.push(cv(&eta1_buy));
        out.push(cv(&eta1_sell));
        out.push(cv(&eta2_buy));
        out.push(cv(&eta2_sell));
        out.push((eta1_buy[0] - eta1_buy[1]).abs()); // consistency 10 vs 5 buy
        out.push((eta1_buy[1] - eta1_buy[2]).abs()); // consistency 5 vs 1 buy
        out.push(asymmetry(h1[0].eta, h1[1].eta)); // asymmetry 10
        out.push(asymmetry(h1[2].eta, h1[3].eta)); // asymmetry 5
        out.push(asymmetry(h1[4].eta, h1[5].eta)); // asymmetry 1
        let abs_vals: Vec<f32> = (0..6).map(|i| h1[i].lambda * h1[i].eta).collect();
        let prop_vals: Vec<f32> = (0..6).map(|i| h2[i].lambda * h2[i].eta).collect();
        let learn_vals: Vec<f32> = (0..6)
            .map(|i| {
                let p = prop_vals[i];
                if p.abs() > 1e-10 {
                    abs_vals[i] / p
                } else {
                    0.0
                }
            })
            .collect();
        out.push(mean(&abs_vals));
        out.push(mean(&prop_vals));
        out.push(mean(&learn_vals));
        out.push(mean(
            &(0..6).map(|i| h1[i].eta - h2[i].eta).collect::<Vec<_>>(),
        ));
        out.push(mean(
            &(0..6)
                .map(|i| {
                    if h2[i].lambda.abs() > 1e-10 {
                        h1[i].lambda / h2[i].lambda
                    } else {
                        0.0
                    }
                })
                .collect::<Vec<_>>(),
        ));
        // direction balance
        out.push(
            (eta1_buy[0] + eta1_buy[1] + eta1_buy[2]).abs()
                - (eta1_sell[0] + eta1_sell[1] + eta1_sell[2]).abs(),
        );
        // size monotonicity
        out.push((eta1_buy[0] - eta1_buy[2]).abs());
        // rank correlation eta1
        out.push(corr(&eta1_buy, &eta1_sell));
        // rank correlation eta2
        out.push(corr(&eta2_buy, &eta2_sell));
        // cross layer correlation
        let all_eta1: Vec<f32> = h1.iter().map(|p| p.eta).collect();
        let all_eta2: Vec<f32> = h2.iter().map(|p| p.eta).collect();
        out.push(corr(&all_eta1, &all_eta2));
        // net flow asymmetry
        let net_flow: Vec<f32> = (0..3)
            .map(|i| h1[i * 2].lambda - h1[i * 2 + 1].lambda)
            .collect();
        out.push(mean(&net_flow));
        // layer1 eta vs layer2 eta ratio
        let eta_ratio: Vec<f32> = (0..6)
            .map(|i| {
                if h2[i].eta.abs() > 1e-10 {
                    h1[i].eta / h2[i].eta
                } else {
                    0.0
                }
            })
            .collect();
        out.push(mean(&eta_ratio));

        // (d) 形状分类：10
        let mean_eta1 = mean(&all_eta1);
        let mean_eta2 = mean(&all_eta2);
        let mean_lam1 = mean(&h1.iter().map(|p| p.lambda).collect::<Vec<_>>());
        let mean_lam2 = mean(&h2.iter().map(|p| p.lambda).collect::<Vec<_>>());
        let pattern = if mean_eta1 > 0.5 && mean_lam1 > mean_lam2 {
            1.0
        } else if mean_lam1 < mean_lam2 && mean_eta2 > mean_eta1 {
            2.0
        } else if mean_eta1.signum() != mean_eta2.signum() {
            3.0
        } else {
            0.0
        };
        out.push(pattern);
        out.push(mean_eta1.max(mean_eta2));
        let l1_counts: Vec<f32> = event_times
            .iter()
            .map(|t| t.iter().filter(|&&x| x < L1_END).count() as f32)
            .collect();
        let l2_counts: Vec<f32> = event_times
            .iter()
            .map(|t| t.iter().filter(|&&x| x >= L1_END && x < L2_END).count() as f32)
            .collect();
        let total_l1: f32 = l1_counts.iter().sum();
        let total_l2: f32 = l2_counts.iter().sum();
        out.push(if total_l2 > 0.0 {
            total_l1 / total_l2
        } else {
            0.0
        });
        let event_counts: Vec<f32> = event_times.iter().map(|t| t.len() as f32).collect();
        out.push(cv(&event_counts));
        // burst ratio = max single event count / total
        let max_cnt = event_counts.iter().copied().fold(0.0f32, f32::max);
        let total_cnt: f32 = event_counts.iter().sum();
        out.push(if total_cnt > 0.0 {
            max_cnt / total_cnt
        } else {
            0.0
        });
        // timing asymmetry = corr(event_index, count)
        let idx_axis: Vec<f32> = (0..6).map(|i| i as f32).collect();
        out.push(corr(&idx_axis, &event_counts));
        // density gradient
        out.push(if mean_lam2.abs() > 1e-10 {
            mean_lam1 / mean_lam2
        } else {
            0.0
        });
        // peak timing (argmax of event counts)
        out.push(
            event_counts
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap()
                .0 as f32,
        );
        // concentration index = gini of event counts
        out.push(gini(&event_counts));

        // (e) 时间间隔特征：6 × 2 = 12
        for times in &event_times {
            if times.len() >= 2 {
                let intervals: Vec<f32> = times.windows(2).map(|w| w[1] - w[0]).collect();
                out.push(mean(&intervals));
                out.push(cv(&intervals));
            } else {
                out.push(0.0);
                out.push(0.0);
            }
        }

        // (f) 成交量特征：6 × 2 = 12
        for (ei, &(is_buy, _)) in event_defs.iter().enumerate() {
            let src = if is_buy { &buy_events } else { &sell_events };
            let thr = event_defs[ei].1;
            let vols: Vec<f32> = src
                .iter()
                .filter(|(_, v)| *v >= thr)
                .map(|(_, v)| *v)
                .collect();
            out.push(mean(&vols));
            out.push(if !vols.is_empty() {
                std_dev(&vols)
            } else {
                0.0
            });
        }

        // (g) 突发检测：12
        // 优化：用滑动窗口 O(n) 而非 O(n²)
        let max_burst_buy = sliding_window_max_count(
            &buy_events.iter().map(|(t, _)| *t).collect::<Vec<_>>(),
            HAWKES_TRIGGER_WIN,
        );
        let max_burst_sell = sliding_window_max_count(
            &sell_events.iter().map(|(t, _)| *t).collect::<Vec<_>>(),
            HAWKES_TRIGGER_WIN,
        );
        out.push(max_burst_buy);
        out.push(max_burst_sell);
        out.push(if max_burst_sell > 0.0 {
            max_burst_buy / max_burst_sell
        } else {
            0.0
        });
        // calm ratio: fraction of 1-min bins with 0 events
        let mut busy_bins = vec![false; md_n_minutes()];
        for (offset, _) in &buy_events {
            let idx = (offset / SEC_PER_MIN) as usize;
            if idx < busy_bins.len() {
                busy_bins[idx] = true;
            }
        }
        for (offset, _) in &sell_events {
            let idx = (offset / SEC_PER_MIN) as usize;
            if idx < busy_bins.len() {
                busy_bins[idx] = true;
            }
        }
        let calm = busy_bins.iter().filter(|&&b| !b).count();
        out.push(calm as f32 / busy_bins.len().max(1) as f32);
        // burst asymmetry per threshold
        out.push(asymmetry(h1[0].lambda, h1[1].lambda));
        out.push(asymmetry(h2[0].lambda, h2[1].lambda));
        // persistence = corr(count_layer1, count_layer2) across events
        out.push(corr(&l1_counts, &l2_counts));
        // burst decay = eta1 - eta2 averaged
        out.push(mean(
            &(0..6).map(|i| h1[i].eta - h2[i].eta).collect::<Vec<_>>(),
        ));
        // burst frequency = total events / total time
        let total_events: f32 = event_counts.iter().sum();
        out.push(total_events / L2_END);
        // burst intensity = mean event size across all events
        let all_vols: Vec<f32> = buy_events
            .iter()
            .chain(sell_events.iter())
            .map(|(_, v)| *v)
            .collect();
        out.push(mean(&all_vols));
        // burst spread = max interval - min interval across events
        let mean_intervals: Vec<f32> = event_times
            .iter()
            .map(|t| {
                if t.len() >= 2 {
                    mean(&t.windows(2).map(|w| w[1] - w[0]).collect::<Vec<_>>())
                } else {
                    0.0
                }
            })
            .collect();
        let max_int = mean_intervals.iter().copied().fold(0.0f32, f32::max);
        let min_int = mean_intervals.iter().copied().fold(f32::INFINITY, f32::min);
        out.push(max_int - min_int);
        // layer delay = mean layer2 time - mean layer1 time
        let l1_mean_time = mean(
            &event_times
                .iter()
                .flat_map(|t| t.iter().filter(|&&x| x < L1_END).copied())
                .collect::<Vec<_>>(),
        );
        let l2_mean_time = mean(
            &event_times
                .iter()
                .flat_map(|t| t.iter().filter(|&&x| x >= L1_END && x < L2_END).copied())
                .collect::<Vec<_>>(),
        );
        out.push(l2_mean_time - l1_mean_time);
        // cross layer amplification
        out.push(if mean_lam1.abs() > 1e-10 {
            mean_lam2 / mean_lam1
        } else {
            0.0
        });

        assert_eq!(out.len(), COUNT);
        out
    }

    fn md_n_minutes() -> usize {
        ((SESSION_END - SESSION_START) / SEC_PER_MIN).ceil() as usize
    }

    fn compute_thresholds(events: &[(f32, f32)]) -> [f32; 3] {
        let mut vols: Vec<f32> = events
            .iter()
            .map(|(_, v)| *v)
            .filter(|v| *v > 0.0)
            .collect();
        if vols.is_empty() {
            return [1.0, 1.0, 1.0]; // 降级：默认阈值
        }
        vols.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        [
            quantile(&vols, LARGE_ORDER_QUANTILES[0]),
            quantile(&vols, LARGE_ORDER_QUANTILES[1]),
            quantile(&vols, LARGE_ORDER_QUANTILES[2]),
        ]
    }

    fn cv(v: &[f32]) -> f32 {
        let m = mean(v);
        let s = std_dev(v);
        if m.abs() > 1e-10 {
            s / m.abs()
        } else {
            0.0
        }
    }

    fn asymmetry(a: f32, b: f32) -> f32 {
        let sum = a + b;
        if sum.abs() > 1e-10 {
            (a - b).abs() / sum.abs()
        } else {
            0.0
        }
    }

    /// 滑动窗口最大计数：对已排序事件时间，找窗口 [t, t+win] 内的最大事件数。
    /// 双指针 O(n)。
    fn sliding_window_max_count(events: &[f32], win: f32) -> f32 {
        if events.is_empty() {
            return 0.0;
        }
        let mut max_cnt = 0usize;
        let mut lo = 0usize;
        for hi in 0..events.len() {
            while events[hi] - events[lo] > win {
                lo += 1;
            }
            let cnt = hi - lo + 1;
            if cnt > max_cnt {
                max_cnt = cnt;
            }
        }
        max_cnt as f32
    }

    pub fn names() -> Vec<String> {
        let mut out = Vec::with_capacity(COUNT);
        let evts = ["buy_10", "sell_10", "buy_5", "sell_5", "buy_1", "sell_1"];
        for e in &evts {
            out.push(format!("dE_lambda1_{}", e));
            out.push(format!("dE_eta1_{}", e));
            out.push(format!("dE_lambda2_{}", e));
            out.push(format!("dE_eta2_{}", e));
        }
        for e in &evts {
            out.push(format!("dE_absorption_{}", e));
            out.push(format!("dE_propagation_{}", e));
            out.push(format!("dE_learning_{}", e));
            out.push(format!("dE_eta_decay_{}", e));
            out.push(format!("dE_lambda_ratio_{}", e));
        }
        for s in &[
            "eta_cv_buy",
            "eta_cv_sell",
            "eta_cv_buy2",
            "eta_cv_sell2",
            "consist_10_5_buy",
            "consist_5_1_buy",
            "asym_10",
            "asym_5",
            "asym_1",
            "abs_mean",
            "prop_mean",
            "learn_mean",
            "eta_decay_mean",
            "lambda_ratio_mean",
            "dir_balance",
            "size_mono",
            "rank_corr_eta1",
            "rank_corr_eta2",
            "cross_layer_corr",
            "net_flow",
            "eta_ratio_mean",
        ] {
            out.push(format!("dE_{}", s));
        }
        for s in &[
            "pattern_label",
            "pattern_score",
            "two_layer_balance",
            "event_cv",
            "burst_ratio",
            "timing_asym",
            "density_gradient",
            "peak_timing",
            "concentration_idx",
            "calm_ratio",
        ] {
            out.push(format!("dE_{}", s));
        }
        for e in &evts {
            out.push(format!("dE_interval_mean_{}", e));
            out.push(format!("dE_interval_cv_{}", e));
        }
        for e in &evts {
            out.push(format!("dE_vol_mean_{}", e));
            out.push(format!("dE_vol_std_{}", e));
        }
        for s in &[
            "max_burst_buy",
            "max_burst_sell",
            "burst_ratio_bs",
            "burst_asym_10",
            "burst_asym_2",
            "persistence",
            "burst_decay",
            "burst_freq",
            "burst_intensity",
            "burst_spread",
            "layer_delay",
            "layer_amplification",
        ] {
            out.push(format!("dE_{}", s));
        }
        // 补齐到 120
        assert_eq!(out.len(), COUNT);
        out
    }
}

// ============================================================================
// 名字
// ============================================================================

pub fn distill_names() -> Vec<String> {
    let mut names = Vec::with_capacity(OUT_LEN);
    names.extend(branch_a::names());
    names.extend(branch_b::names());
    names.extend(branch_d::names());
    names.extend(branch_e::names());
    assert_eq!(names.len(), OUT_LEN);
    names
}

// ============================================================================
// 核心计算
// ============================================================================

pub fn compute_distill_full(
    code: &str,
    date: i64,
    prev_date: Option<i64>,
) -> std::io::Result<Vec<f32>> {
    // 优化：只读一次 market_raw（含集合竞价，不过滤涨跌停，不平移）
    // market（连续竞价专用）从 market_raw 内存过滤得到，避免二次读 CSV
    let market_raw = read_market_fast_inner(code, date, true, false, usize::MAX)?;

    // 从 market_raw 过滤出连续竞价段（去掉集合竞价 + 去掉涨跌停 + 做下午平移）
    let afternoon_start: f32 = 13.0 * 3600.0;
    let afternoon_end: f32 = 14.0 * 3600.0 + 57.0 * 60.0;
    let morning_start: f32 = 9.0 * 3600.0 + 30.0 * 60.0;
    let morning_end: f32 = 11.0 * 3600.0 + 30.0 * 60.0;
    let shift: f32 = 90.0 * 60.0;
    let market: Vec<MarketRecord> = market_raw
        .iter()
        .filter_map(|m| {
            // 过滤涨跌停
            if m.ask_prcs[0] == 0.0 || m.bid_prcs[0] == 0.0 {
                return None;
            }
            let day_sec = m.time_sec % 86400.0;
            // 连续竞价段平移
            let new_time = if day_sec >= afternoon_start && day_sec <= afternoon_end {
                m.time_sec - shift
            } else if day_sec >= morning_start && day_sec <= morning_end {
                m.time_sec
            } else {
                return None; // 集合竞价段或其他
            };
            let mut rec = *m;
            rec.time_sec = new_time;
            Some(rec)
        })
        .collect();

    let trade = read_trade_fast_inner(code, date, false, true, usize::MAX)?;

    // 昨收：从前一交易日的 market 数据读最后一条 last_prc
    let yclose = match prev_date {
        Some(pd) => match read_market_fast_inner(code, pd, false, true, usize::MAX) {
            Ok(mp) => mp
                .iter()
                .rev()
                .find(|m| m.last_prc > 0.0)
                .map(|m| m.last_prc)
                .unwrap_or(f32::NAN),
            Err(_) => f32::NAN,
        },
        None => f32::NAN,
    };

    let teacher = extract_auction_teacher(&market_raw, &trade, yclose);
    if teacher.p_auct.is_nan() {
        return Ok(vec![0.0; OUT_LEN]);
    }

    let md = resample_minutes(&market, &trade);
    let eps = compute_epsilons(&teacher, &md);

    let mut out = Vec::with_capacity(OUT_LEN);
    out.extend(branch_a::compute(&teacher, &md));
    out.extend(branch_b::compute(&eps, &md));
    out.extend(branch_d::compute(&eps, &md));
    out.extend(branch_e::compute(&trade));

    assert_eq!(out.len(), OUT_LEN);
    Ok(out)
}

// ============================================================================
// PyO3 接口
// ============================================================================

#[pyfunction]
#[pyo3(signature = (code, date, prev_date=None))]
pub fn py_distill(
    py: Python<'_>,
    code: &str,
    date: i64,
    prev_date: Option<i64>,
) -> PyResult<Vec<f32>> {
    compute_distill_full(code, date, prev_date)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("{:?}", e)))
}

#[pyfunction]
pub fn py_distill_names() -> Vec<String> {
    distill_names()
}
