//! Agent交互验证 — 新增因子计算 (TQ, PC, SA, RI, ET, CD, LF)
//!
//! 因子顺序：TQ → PC → SA → RI → ET → CD (per-agent) → LF (cross-agent)

use std::f64;


// Local utility functions (mirror mod.rs)
#[inline]
fn safe_div(a: f64, b: f64) -> f64 {
    if b.abs() < 1e-15 || !b.is_finite() { 0.0 } else { a / b }
}
#[inline]
fn cap_val(v: f64, limit: f64) -> f64 {
    if !v.is_finite() { return 0.0; }
    v.max(-limit).min(limit)
}
fn safe_mean(x: &[f64]) -> f64 {
    if x.is_empty() { return 0.0; }
    let sum: f64 = x.iter().filter(|v| v.is_finite()).sum();
    let count = x.iter().filter(|v| v.is_finite()).count();
    if count == 0 { 0.0 } else { sum / count as f64 }
}
fn safe_std(x: &[f64]) -> f64 {
    if x.len() < 2 { return 0.0; }
    let valid: Vec<f64> = x.iter().filter(|v| v.is_finite()).copied().collect();
    if valid.len() < 2 { return 0.0; }
    let n = valid.len() as f64;
    let mean = valid.iter().sum::<f64>() / n;
    let var = valid.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
    var.sqrt()
}

const TQ_PER_HORIZON: usize = 6;
const PC_PER_HORIZON: usize = 6;
const SA_PER_AGENT: usize = 9;
const RI_PER_HORIZON: usize = 2; // ret_diff, hitrate_diff
const ET_PER_AGENT: usize = 4;
const CD_PER_AGENT: usize = 5;
const LF_PER_PAIR: usize = 7;

const N_TQ_H: usize = 5;
const N_PC_H: usize = 5;
const N_RI_H: usize = 8;
const N_STATES: usize = 10;

const TQ_HORIZONS: [f64; N_TQ_H] = [1.0, 3.0, 5.0, 10.0, 30.0];
const PC_HORIZONS: [f64; N_PC_H] = [1.0, 3.0, 5.0, 10.0, 30.0];
const RI_HORIZONS: [f64; N_RI_H] = [1.0, 3.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0];

const PER_AGENT_NEW: usize = N_TQ_H * TQ_PER_HORIZON
    + N_PC_H * PC_PER_HORIZON
    + SA_PER_AGENT
    + N_STATES * N_RI_H * RI_PER_HORIZON
    + ET_PER_AGENT
    + CD_PER_AGENT;

pub fn extra_factor_count(n_agents: usize) -> usize {
    let n_pairs = n_agents * (n_agents - 1) / 2;
    n_agents * PER_AGENT_NEW + n_pairs * LF_PER_PAIR
}

// ============================================================
// 数学工具函数
// ============================================================

fn hurst_rs(data: &[f64]) -> f64 {
    let n = data.len();
    if n < 16 {
        return 0.5;
    }
    let mut scales: Vec<usize> = Vec::new();
    let mut s: usize = 4;
    while s <= n / 2 && scales.len() < 20 {
        scales.push(s);
        s = ((s as f64) * 1.5) as usize + 1;
    }
    if scales.len() < 3 {
        return 0.5;
    }

    let mut log_scales = Vec::with_capacity(scales.len());
    let mut log_rs = Vec::with_capacity(scales.len());

    for &scale in &scales {
        let n_seg = n / scale;
        let mut rs_sum = 0.0;
        for seg in 0..n_seg {
            let seg_start = seg * scale;
            let seg_end = seg_start + scale;
            let mean: f64 = data[seg_start..seg_end].iter().sum::<f64>() / scale as f64;

            let mut cum_dev = 0.0f64;
            let mut cum_min = 0.0f64;
            let mut cum_max = 0.0f64;
            let mut var_sum = 0.0f64;
            for &v in &data[seg_start..seg_end] {
                let dev = v - mean;
                cum_dev += dev;
                cum_min = cum_min.min(cum_dev);
                cum_max = cum_max.max(cum_dev);
                var_sum += dev * dev;
            }
            let r_val = cum_max - cum_min;
            let s_val = (var_sum / scale as f64).sqrt().max(1e-15);
            rs_sum += r_val / s_val;
        }
        let rs_avg = rs_sum / n_seg as f64;
        if rs_avg > 0.0 {
            log_scales.push((scale as f64).ln());
            log_rs.push(rs_avg.ln());
        }
    }

    if log_scales.len() < 3 {
        return 0.5;
    }
    // Linear regression: log(RS) = H * log(scale) + C
    let n_pts = log_scales.len() as f64;
    let sx: f64 = log_scales.iter().sum();
    let sy: f64 = log_rs.iter().sum();
    let sxx: f64 = log_scales.iter().map(|&x| x * x).sum();
    let sxy: f64 = log_scales.iter().zip(log_rs.iter()).map(|(&x, &y)| x * y).sum();
    let den = n_pts * sxx - sx * sx;
    if den.abs() < 1e-15 {
        return 0.5;
    }
    let slope = (n_pts * sxy - sx * sy) / den;
    slope.max(0.0).min(1.0)
}

fn fractal_dimension(data: &[f64]) -> f64 {
    let n = data.len();
    if n < 2 {
        return 1.0;
    }
    let mut total_len = 0.0f64;
    for i in 1..n {
        total_len += (data[i] - data[i - 1]).abs();
    }
    let first = data[0];
    let max_dist = data.iter().fold(0.0f64, |acc, &v| acc.max((v - first).abs()));
    if max_dist < 1e-15 || total_len < 1e-15 {
        return 1.0;
    }
    (total_len.ln() / max_dist.ln()).min(2.0)
}

fn permutation_entropy(data: &[f64], m: usize) -> f64 {
    let n = data.len();
    if n < m {
        return 1.0;
    }
    let n_patterns = n - m + 1;
    use std::collections::HashMap;
    let mut counts: HashMap<Vec<usize>, usize> = HashMap::new();

    let mut window: Vec<(f64, usize)> = (0..m).map(|i| (data[i], i)).collect();

    for i in 0..n_patterns {
        if i > 0 {
            // slide window
            for j in 0..m - 1 {
                window[j] = window[j + 1];
            }
            window[m - 1] = (data[i + m - 1], i + m - 1);
        }
        // sort by value to get rank pattern
        let mut indices: Vec<usize> = (0..m).collect();
        indices.sort_by(|&a, &b| {
            window[a].0.partial_cmp(&window[b].0).unwrap_or(std::cmp::Ordering::Equal)
        });
        *counts.entry(indices).or_insert(0) += 1;
    }

    let total = n_patterns as f64;
    let mut entropy = 0.0f64;
    for &count in counts.values() {
        let p = count as f64 / total;
        if p > 0.0 {
            entropy -= p * p.ln();
        }
    }
    // Normalize by max entropy = ln(m!)
    let max_entropy = (1..=m).map(|k| (k as f64).ln()).sum::<f64>();
    if max_entropy < 1e-15 {
        1.0
    } else {
        entropy / max_entropy
    }
}

fn quadratic_convexity(data: &[f64]) -> f64 {
    // Fit y = a*t² + b*t + c, return a (normalized by std)
    let n = data.len();
    if n < 4 {
        return 0.0;
    }
    let mean_y: f64 = data.iter().sum::<f64>() / n as f64;
    let st = data.iter().map(|&v| (v - mean_y).powi(2)).sum::<f64>() / n as f64;
    let std_y = st.sqrt();
    if std_y < 1e-15 {
        return 0.0;
    }
    // Normalize to zero mean
    let y: Vec<f64> = data.iter().map(|&v| (v - mean_y) / std_y).collect();
    let t: Vec<f64> = (0..n).map(|i| (i as f64 - (n - 1) as f64 / 2.0) / ((n - 1) as f64 / 2.0)).collect();

    // Solve for a, b, c in y = a*t² + b*t + c via least squares
    let mut st2 = 0.0f64;
    let mut st3 = 0.0f64;
    let mut st4 = 0.0f64;
    let mut syt = 0.0f64;
    let mut syt2 = 0.0f64;
    let mut sy = 0.0f64;
    let mut st_sum = 0.0f64;

    for i in 0..n {
        let ti = t[i];
        let yi = y[i];
        st2 += ti * ti;
        st3 += ti * ti * ti;
        st4 += ti * ti * ti * ti;
        syt += yi * ti;
        syt2 += yi * ti * ti;
        sy += yi;
        st_sum += ti;
    }
    let n_f = n as f64;
    // Solve 3x3 system for [c, b, a]:
    // [n    Σt   Σt²] [c]   [Σy ]
    // [Σt   Σt²  Σt³] [b] = [Σyt]
    // [Σt²  Σt³  Σt⁴] [a]   [Σyt²]
    // Use Cramer's rule or simple Gaussian elimination
    let a11 = n_f; let a12 = st_sum; let a13 = st2;
    let a21 = st_sum; let a22 = st2; let a23 = st3;
    let a31 = st2; let a32 = st3; let a33 = st4;
    let b1 = sy; let b2 = syt; let b3 = syt2;

    // Determinant
    let det = a11 * (a22 * a33 - a23 * a32)
        - a12 * (a21 * a33 - a23 * a31)
        + a13 * (a21 * a32 - a22 * a31);
    if det.abs() < 1e-15 {
        return 0.0;
    }
    // a = det_a / det
    let det_a = a11 * (a22 * b3 - b2 * a32)
        - a12 * (a21 * b3 - b2 * a31)
        + b1 * (a21 * a32 - a22 * a31);
    det_a / det
}

fn runs_test_z(seq: &[bool]) -> f64 {
    let n = seq.len();
    if n < 4 {
        return 0.0;
    }
    let n1 = seq.iter().filter(|&&b| b).count();
    let n0 = n - n1;
    if n1 < 2 || n0 < 2 {
        return 0.0;
    }
    let mut runs = 1usize;
    for i in 1..n {
        if seq[i] != seq[i - 1] {
            runs += 1;
        }
    }
    let n_f = n as f64;
    let n1_f = n1 as f64;
    let n0_f = n0 as f64;
    let exp_runs = 1.0 + 2.0 * n1_f * n0_f / n_f;
    let var_runs = (2.0 * n1_f * n0_f * (2.0 * n1_f * n0_f - n_f)) / (n_f * n_f * (n_f - 1.0));
    let std_runs = var_runs.max(1e-15).sqrt();
    (runs as f64 - exp_runs) / std_runs
}

fn transition_entropy(seq: &[bool]) -> f64 {
    // 1st-order Markov transition entropy
    let n = seq.len();
    if n < 4 {
        return 1.0;
    }
    // Count transitions: (prev, curr) -> count
    let mut t00 = 0usize;
    let mut t01 = 0usize;
    let mut t10 = 0usize;
    let mut t11 = 0usize;
    for i in 1..n {
        match (seq[i - 1], seq[i]) {
            (false, false) => t00 += 1,
            (false, true) => t01 += 1,
            (true, false) => t10 += 1,
            (true, true) => t11 += 1,
        }
    }
    let from0 = (t00 + t01) as f64;
    let from1 = (t10 + t11) as f64;
    let total = (n - 1) as f64;
    let mut entropy = 0.0f64;

    let p00 = if from0 > 0.0 { t00 as f64 / total } else { 0.0 };
    let p01 = if from0 > 0.0 { t01 as f64 / total } else { 0.0 };
    let p10 = if from1 > 0.0 { t10 as f64 / total } else { 0.0 };
    let p11 = if from1 > 0.0 { t11 as f64 / total } else { 0.0 };

    for &p in &[p00, p01, p10, p11] {
        if p > 0.0 {
            entropy -= p * p.ln();
        }
    }
    // Max entropy = ln(4) for 4 possible transitions
    let max_entropy = 4.0f64.ln();
    if max_entropy < 1e-15 {
        1.0
    } else {
        entropy / max_entropy
    }
}

fn lz_complexity_bits(seq: &[bool]) -> f64 {
    let n = seq.len();
    if n < 4 {
        return 0.5;
    }
    // Convert to byte string for simpler LZ
    let bytes: Vec<u8> = seq.iter().map(|&b| if b { 1u8 } else { 0u8 }).collect();

    let mut i = 0usize;
    let mut vocab_count = 0usize;
    while i < n {
        let mut l = 1usize;
        while i + l <= n {
            let pattern = &bytes[i..i + l];
            let rest = &bytes[0..i + l - 1];
            if !contains_subsequence(rest, pattern) {
                vocab_count += 1;
                i += l;
                break;
            }
            l += 1;
        }
        if i + l > n {
            vocab_count += 1;
            break;
        }
    }
    let c = vocab_count as f64;
    (c * (n as f64).ln() / n as f64).min(1.0)
}

fn contains_subsequence(haystack: &[u8], needle: &[u8]) -> bool {
    if needle.is_empty() {
        return true;
    }
    if haystack.len() < needle.len() {
        return false;
    }
    for start in 0..=haystack.len() - needle.len() {
        if haystack[start..start + needle.len()] == needle[..] {
            return true;
        }
    }
    false
}

fn hilbert_approx(signal: &[f64]) -> Vec<f64> {
    let n = signal.len();
    let half = 15usize;
    let mut result = vec![0.0f64; n];
    for i in 0..n {
        let mut s = 0.0f64;
        for k in 1..=half {
            if k % 2 == 1 {
                let w = 0.54 - 0.46 * (std::f64::consts::PI * k as f64 / half as f64).cos();
                let coef = 2.0 / (std::f64::consts::PI * k as f64) * w;
                if i >= k {
                    s += coef * signal[i - k];
                }
                if i + k < n {
                    s -= coef * signal[i + k];
                }
            }
        }
        result[i] = s;
    }
    result
}

fn plv_and_lead(sig_a: &[f64], sig_b: &[f64]) -> (f64, f64) {
    let n = sig_a.len().min(sig_b.len());
    if n < 10 {
        return (0.0, 0.0);
    }
    let hilb_a = hilbert_approx(&sig_a[..n]);
    let hilb_b = hilbert_approx(&sig_b[..n]);

    let mut sum_re = 0.0f64;
    let mut sum_im = 0.0f64;
    let mut lead_sum = 0.0f64;

    let offset = 5usize; // skip filter transient
    let start = offset;
    let end = n - offset;

    for i in start..end {
        let phase_a = hilb_a[i].atan2(sig_a[i]);
        let phase_b = hilb_b[i].atan2(sig_b[i]);
        let diff = phase_a - phase_b;
        sum_re += diff.cos();
        sum_im += diff.sin();
        lead_sum += diff;
    }
    let count = (end - start) as f64;
    if count < 1.0 {
        return (0.0, 0.0);
    }
    let plv = (sum_re.powi(2) + sum_im.powi(2)).sqrt() / count;
    let lead = lead_sum / count;
    (plv.min(1.0), lead)
}

fn transfer_entropy_dir(x: &[f64], y: &[f64]) -> f64 {
    // TE(X→Y): how much does knowing X's past help predict Y?
    // Discrete symbols: 3 levels (-1, 0, +1)
    // Use k=1 (one lag)
    let n = x.len().min(y.len());
    if n < 20 {
        return 0.0;
    }

    fn sym(v: f64) -> i8 {
        if v > 0.3 {
            1
        } else if v < -0.3 {
            -1
        } else {
            0
        }
    }

    use std::collections::HashMap;
    // Count: (y_t, y_{t-1}, x_{t-1})
    let mut p_y_yx = HashMap::new();
    let mut p_yx = HashMap::new();
    let mut p_y_y = HashMap::new();
    let mut p_y = HashMap::new();

    for t in 1..n {
        let yt = sym(y[t]);
        let yp = sym(y[t - 1]);
        let xp = sym(x[t - 1]);

        *p_y_yx.entry((yt, yp, xp)).or_insert(0usize) += 1;
        *p_yx.entry((yp, xp)).or_insert(0usize) += 1;
        *p_y_y.entry((yt, yp)).or_insert(0usize) += 1;
        *p_y.entry(yp).or_insert(0usize) += 1;
    }

    let total = (n - 1) as f64;
    let mut te = 0.0f64;
    for (&(yt, yp, xp), &count) in &p_y_yx {
        let p_joint = count as f64 / total;
        let count_yx = *p_yx.get(&(yp, xp)).unwrap_or(&0) as f64;
        let count_yy = *p_y_y.get(&(yt, yp)).unwrap_or(&0) as f64;
        let count_y = *p_y.get(&yp).unwrap_or(&0) as f64;

        if count_yx > 0.0 && count_yy > 0.0 && count_y > 0.0 {
            let p_yt_yx = count as f64 / count_yx;
            let p_yt_yp = count_yy / count_y;
            if p_yt_yx > 0.0 && p_yt_yp > 0.0 {
                te += p_joint * (p_yt_yx / p_yt_yp).ln();
            }
        }
    }
    te.max(0.0)
}

fn event_sync_q(ts_a: &[i64], ts_b: &[i64]) -> f64 {
    // Simplified Event Synchronization Q
    let n_a = ts_a.len();
    if n_a < 5 || ts_b.is_empty() {
        return 0.0;
    }
    // Adaptive tau: half the minimum inter-event interval
    let mut min_interval = i64::MAX;
    for i in 1..n_a {
        min_interval = min_interval.min(ts_a[i] - ts_a[i - 1]);
    }
    let tau = (min_interval / 2).max(1_000_000_i64); // at least 1ms

    let mut sync_count = 0i64;
    for &t in ts_a {
        // Find nearest B event
        match ts_b.binary_search(&t) {
            Ok(_) => sync_count += 1,
            Err(pos) => {
                let mut nearest_dist = i64::MAX;
                if pos > 0 {
                    nearest_dist = nearest_dist.min((t - ts_b[pos - 1]).abs());
                }
                if pos < ts_b.len() {
                    nearest_dist = nearest_dist.min((ts_b[pos] - t).abs());
                }
                if nearest_dist <= tau {
                    sync_count += 1;
                }
            }
        }
    }
    // Q relative to random expectation (simplified: Q = 2*(sync_count - E)/n_A)
    // Expected sync count: 2 * tau * n_A * rate_B
    let rate_b = ts_b.len() as f64 / ((ts_b.last().unwrap_or(&1) - ts_b.first().unwrap_or(&0)).max(1) as f64);
    let expected = 2.0 * tau as f64 * n_a as f64 * rate_b / 1_000_000_000_f64;
    let var_expected = expected.max(1.0);
    (sync_count as f64 - expected) / var_expected.sqrt()
}

// ============================================================
// 状态指标计算 (用于 RI)
// ============================================================

struct StateArrays {
    // A: 价格波动类
    realized_vol_5min: Vec<f64>,
    trend_strength: Vec<f64>,
    ret_autocorr_1s: Vec<f64>,
    price_vs_vwap: Vec<f64>,
    daily_range_pos: Vec<f64>,
    // B: 成交量类
    vol_intensity: Vec<f64>,
    vol_trend: Vec<f64>,
    trade_size_ratio: Vec<f64>,
    buy_vol_ratio: Vec<f64>,
    // C: 盘口类
    norm_spread: Vec<f64>,
    ob_imbalance_state: Vec<f64>,
    depth_slope: Vec<f64>,
    visible_order_ratio: Vec<f64>,
    weighted_spread: Vec<f64>,
    depth_consumption: Vec<f64>,
    // D: 订单ID/时间类
    order_id_gap: Vec<f64>,
    order_id_gap_trend: Vec<f64>,
    event_rate: Vec<f64>,
}

fn compute_state_arrays(
    mkt_ts: &[i64],
    mkt_pr: &[f64],
    mkt_vo: &[f64],
    mkt_fl: &[i32],
    ob_spread: &[f64],
    ob_imbalance: &[f64],
    ob_depth: &[f64],
    ob_idx_for_trade: &[usize],
    ob_bid_vol1: &[f64],
    ob_ask_vol1: &[f64],
    ob_bid1: &[f64],
    ob_ask1: &[f64],
    bid_order_ids: &[i64],
    ask_order_ids: &[i64],
    n_mkt: usize,
    n_ob: usize,
) -> StateArrays {
    let mut sa = StateArrays {
        realized_vol_5min: vec![0.0; n_mkt],
        trend_strength: vec![0.0; n_mkt],
        ret_autocorr_1s: vec![0.0; n_mkt],
        price_vs_vwap: vec![0.0; n_mkt],
        daily_range_pos: vec![0.0; n_mkt],
        vol_intensity: vec![0.0; n_mkt],
        vol_trend: vec![0.0; n_mkt],
        trade_size_ratio: vec![0.0; n_mkt],
        buy_vol_ratio: vec![0.0; n_mkt],
        norm_spread: vec![0.0; n_mkt],
        ob_imbalance_state: vec![0.0; n_mkt],
        depth_slope: vec![0.0; n_mkt],
        visible_order_ratio: vec![0.0; n_mkt],
        weighted_spread: vec![0.0; n_mkt],
        depth_consumption: vec![0.0; n_mkt],
        order_id_gap: vec![0.0; n_mkt],
        order_id_gap_trend: vec![0.0; n_mkt],
        event_rate: vec![0.0; n_mkt],
    };

    let window_5min = 300 * 1_000_000_000_i64;
    let window_1min = 60 * 1_000_000_000_i64;
    let window_30min = 1800 * 1_000_000_000_i64;
    let window_1s = 1_000_000_000_i64;

    // Daily range
    let (day_min, day_max) = if n_mkt > 0 {
        let min_p = mkt_pr.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_p = mkt_pr.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        (min_p, max_p)
    } else {
        (0.0, 0.0)
    };
    let day_range = day_max - day_min;

    // VWAP
    let mut cum_to: f64 = 0.0;
    let mut cum_vo: f64 = 0.0;

    for i in 0..n_mkt {
        cum_to += mkt_pr[i] * mkt_vo[i];
        cum_vo += mkt_vo[i];
        let vwap = if cum_vo > 0.0 { cum_to / cum_vo } else { mkt_pr[i] };
        sa.price_vs_vwap[i] = if vwap > 0.0 { (mkt_pr[i] - vwap) / vwap } else { 0.0 };
        sa.daily_range_pos[i] = if day_range > 1e-15 { (mkt_pr[i] - day_min) / day_range } else { 0.5 };

        // Realized vol (past 5min)
        let t = mkt_ts[i];
        let lo = match mkt_ts.binary_search(&(t - window_5min)) {
            Ok(x) => x,
            Err(x) => x,
        };
        if i > lo + 5 {
            let rets: Vec<f64> = (lo + 1..=i).map(|j| {
                if mkt_pr[j - 1] > 0.0 { (mkt_pr[j] - mkt_pr[j - 1]) / mkt_pr[j - 1] } else { 0.0 }
            }).collect();
            if rets.len() >= 5 {
                let mean_ret = rets.iter().sum::<f64>() / rets.len() as f64;
                let var = rets.iter().map(|&r| (r - mean_ret).powi(2)).sum::<f64>() / rets.len() as f64;
                sa.realized_vol_5min[i] = var.sqrt();
                // Trend strength
                let total_ret = if mkt_pr[lo] > 0.0 { (mkt_pr[i] - mkt_pr[lo]) / mkt_pr[lo] } else { 0.0 };
                let vol_scaled = sa.realized_vol_5min[i].max(1e-15) * ((i - lo) as f64 / 1_000_000_000_f64).sqrt();
                sa.trend_strength[i] = if vol_scaled > 1e-15 { total_ret.abs() / vol_scaled } else { 0.0 };
            }
        }

        // Ret autocorr 1s
        let lo_1s = match mkt_ts.binary_search(&(t - window_1s)) {
            Ok(x) => x,
            Err(x) => x,
        };
        if i > lo_1s + 4 {
            let rets_1s: Vec<f64> = (lo_1s + 1..=i).map(|j| {
                if mkt_pr[j - 1] > 0.0 { (mkt_pr[j] - mkt_pr[j - 1]) / mkt_pr[j - 1] } else { 0.0 }
            }).collect();
            if rets_1s.len() >= 5 {
                let n_r = rets_1s.len();
                let rets_a = &rets_1s[..n_r - 1];
                let rets_b = &rets_1s[1..];
                let m_a = rets_a.iter().sum::<f64>() / rets_a.len() as f64;
                let m_b = rets_b.iter().sum::<f64>() / rets_b.len() as f64;
                let mut num = 0.0; let mut den_a = 0.0; let mut den_b = 0.0;
                for k in 0..n_r - 1 {
                    num += (rets_a[k] - m_a) * (rets_b[k] - m_b);
                    den_a += (rets_a[k] - m_a).powi(2);
                    den_b += (rets_b[k] - m_b).powi(2);
                }
                sa.ret_autocorr_1s[i] = if den_a * den_b > 1e-30 { num / (den_a * den_b).sqrt() } else { 0.0 };
            }
        }

        // Vol intensity: 1min volume / 30min average
        let lo_v = match mkt_ts.binary_search(&(t - window_1min)) {
            Ok(x) => x,
            Err(x) => x,
        };
        let lo_v30 = match mkt_ts.binary_search(&(t - window_30min)) {
            Ok(x) => x,
            Err(x) => x,
        };
        if hi_lo_diff(lo_v, i) > 0 {
            let vol_1m = mkt_vo[lo_v..i].iter().sum::<f64>();
            let time_1m = (t - mkt_ts[lo_v]).max(1) as f64;
            let vol_rate = vol_1m / time_1m;
            let span_30 = (i - lo_v30).max(1) as f64;
            let vol_30 = mkt_vo[lo_v30..i].iter().sum::<f64>() / span_30;
            sa.vol_intensity[i] = if vol_30 > 0.0 { vol_rate / vol_30 } else { 1.0 };
        }

        // Order ID gap
        let ob_i = ob_idx_for_trade[i];
        if ob_i < n_ob {
            sa.order_id_gap[i] = (ask_order_ids[i] - bid_order_ids[i]) as f64;
            sa.norm_spread[i] = if ob_bid1[ob_i] > 0.0 {
                (ob_ask1[ob_i] - ob_bid1[ob_i]) / ob_bid1[ob_i]
            } else {
                0.0
            };
            sa.ob_imbalance_state[i] = ob_imbalance[ob_i];
            sa.depth_slope[i] = if ob_depth[ob_i] > 0.0 {
                (ob_bid_vol1[ob_i] + ob_ask_vol1[ob_i]) / ob_depth[ob_i]
            } else {
                1.0
            };
            sa.visible_order_ratio[i] = if ob_bid_vol1[ob_i] + ob_ask_vol1[ob_i] > 0.0 {
                (ob_bid_vol1[ob_i] + ob_ask_vol1[ob_i]) / (ob_bid_vol1[ob_i] + ob_ask_vol1[ob_i]).max(1.0)
            } else {
                0.0
            };
            // Weighted spread approximation using bid1/ask1
            let w_bid = ob_bid1[ob_i];
            let w_ask = ob_ask1[ob_i];
            sa.weighted_spread[i] = if w_bid > 0.0 { (w_ask - w_bid) / w_bid } else { 0.0 };
        }

        // Buy vol ratio in window
        let lo_w = match mkt_ts.binary_search(&(t - window_1s)) {
            Ok(x) => x,
            Err(x) => x,
        };
        if i > lo_w {
            let n_buy = mkt_fl[lo_w..i].iter().filter(|&&f| f == 66).count();
            let n_tot = i - lo_w;
            sa.buy_vol_ratio[i] = n_buy as f64 / n_tot.max(1) as f64;
            sa.event_rate[i] = n_tot as f64 / (t - mkt_ts[lo_w]).max(1) as f64;
        }
    }

    // Post-pass: vol trend (needs sequential windows)
    for i in 0..n_mkt {
        let t = mkt_ts[i];
        // 5 x 1min windows
        let mut vol_rates: Vec<f64> = Vec::new();
        for w in 0..5 {
            let lo = match mkt_ts.binary_search(&(t - (w + 1) as i64 * window_1min)) {
                Ok(x) => x, Err(x) => x,
            };
            let hi = match mkt_ts.binary_search(&(t - w as i64 * window_1min)) {
                Ok(x) => x, Err(x) => x,
            };
            if hi > lo + 1 && lo < i {
                let vol = mkt_vo[lo..hi].iter().sum::<f64>();
                let dt = (mkt_ts[hi - 1] - mkt_ts[lo]).max(1) as f64;
                vol_rates.push(vol / dt);
            }
        }
        if vol_rates.len() >= 3 {
            let n = vol_rates.len() as f64;
            let mean_v = vol_rates.iter().sum::<f64>() / n;
            let x_mean = (n - 1.0) / 2.0;
            let mut num = 0.0; let mut den = 0.0;
            for (j, v) in vol_rates.iter().enumerate() {
                let dx = j as f64 - x_mean;
                num += dx * (v - mean_v);
                den += dx * dx;
            }
            sa.vol_trend[i] = if den > 1e-15 { num / den } else { 0.0 };
        }

        // Trade size ratio
        let lo_t = match mkt_ts.binary_search(&(t - window_1min)) {
            Ok(x) => x, Err(x) => x,
        };
        if i > lo_t + 1 {
            let avg_vol = mkt_vo[lo_t..i].iter().sum::<f64>() / (i - lo_t) as f64;
            let lo_30 = match mkt_ts.binary_search(&(t - window_30min)) {
                Ok(x) => x, Err(x) => x,
            };
            if lo_30 < lo_t {
                let hist_avg = mkt_vo[lo_30..lo_t].iter().sum::<f64>() / (lo_t - lo_30).max(1) as f64;
                sa.trade_size_ratio[i] = if hist_avg > 0.0 { avg_vol / hist_avg } else { 1.0 };
            }
        }

        // Depth consumption
        let ob_i = ob_idx_for_trade[i];
        let lo_dc = match mkt_ts.binary_search(&(t - window_1s)) {
            Ok(x) => x, Err(x) => x,
        };
        if i > lo_dc && ob_i < n_ob {
            let total_vol = mkt_vo[lo_dc..i].iter().sum::<f64>();
            let depth1 = ob_bid_vol1[ob_i] + ob_ask_vol1[ob_i];
            sa.depth_consumption[i] = if depth1 > 0.0 { total_vol / depth1 } else { 0.0 };
        }
    }

    // Order ID gap trend
    for i in 0..n_mkt {
        let t = mkt_ts[i];
        let lo = match mkt_ts.binary_search(&(t - window_5min)) {
            Ok(x) => x, Err(x) => x,
        };
        if i > lo + 5 {
            let gaps: Vec<f64> = (lo..i).map(|j| sa.order_id_gap[j]).collect();
            let n = gaps.len() as f64;
            let mean_g = gaps.iter().sum::<f64>() / n;
            let x_mean = (n - 1.0) / 2.0;
            let mut num = 0.0; let mut den = 0.0;
            for (j, &g) in gaps.iter().enumerate() {
                let dx = j as f64 - x_mean;
                num += dx * (g - mean_g);
                den += dx * dx;
            }
            sa.order_id_gap_trend[i] = if den > 1e-15 { num / den } else { 0.0 };
        }
    }

    sa
}

#[inline]
fn hi_lo_diff(lo: usize, hi: usize) -> usize {
    if hi > lo { hi - lo } else { 0 }
}

// ============================================================
// 主计算函数
// ============================================================

pub fn compute_extra_factors(
    mkt_ts: &[i64],
    mkt_pr: &[f64],
    mkt_vo: &[f64],
    mkt_fl: &[i32],
    ob_spread: &[f64],
    ob_imbalance: &[f64],
    ob_depth: &[f64],
    ob_idx_for_trade: &[usize],
    ob_bid_vol1: &[f64],
    ob_ask_vol1: &[f64],
    ob_bid1: &[f64],
    ob_ask1: &[f64],
    bid_order_ids: &[i64],
    ask_order_ids: &[i64],
    per_agent_idx: &[Vec<i64>],
    per_agent_dir: &[Vec<i32>],
    per_agent_vol: &[Vec<f64>],
    per_agent_sign: &[Vec<f64>],
    fwd_prices: &[Vec<f64>],
    fwd_indices: &[Vec<i64>],
    fwd_horizons_sec: &[f64],
) -> Vec<f64> {
    let n_agents = per_agent_idx.len();
    let n_mkt = mkt_ts.len();
    let n_ob = ob_spread.len();
    let n_pairs = n_agents * (n_agents - 1) / 2;
    let n_fwd_h = fwd_horizons_sec.len();

    let total = extra_factor_count(n_agents);
    let mut result = vec![0.0f64; total];

    // Helper: find horizon index in fwd_horizons_sec
    let find_h_idx = |h: f64| -> Option<usize> {
        fwd_horizons_sec.iter().position(|&x| (x - h).abs() < 1e-9)
    };

    // ===========================================================
    // PILLAR 7: 择时质量 (TQ)
    // ===========================================================
    for a in 0..n_agents {
        let idxs_a = &per_agent_idx[a];
        let sign_a = &per_agent_sign[a];
        let n_trades = idxs_a.len();
        if n_trades < 2 {
            continue;
        }

        for (th, &h_sec) in TQ_HORIZONS.iter().enumerate() {
            if let Some(h_idx) = find_h_idx(h_sec) {
                let base = a * PER_AGENT_NEW + th * TQ_PER_HORIZON;

                let mut pre_momentum = vec![0.0f64; n_trades];
                let mut entry_pct = vec![0.0f64; n_trades];
                let mut adverse = vec![0.0f64; n_trades];
                let mut capture = vec![0.0f64; n_trades];
                let mut rr_eff = vec![0.0f64; n_trades];
                let mut time_profit = vec![0.0f64; n_trades];

                let h_ns = (h_sec * 1_000_000_000_f64) as i64;
                let pre_ns = (h_sec * 1_000_000_000_f64 * 0.5) as i64; // pre-trade window

                for k in 0..n_trades {
                    let i = idxs_a[k] as usize;
                    let ep = mkt_pr[i];
                    if ep <= 0.0 {
                        continue;
                    }

                    // Pre-trade momentum (normalized by return vol)
                    let t = mkt_ts[i];
                    let lo_pre = match mkt_ts.binary_search(&(t - pre_ns)) {
                        Ok(x) => x, Err(x) => x,
                    };
                    if i > lo_pre + 2 && mkt_pr[lo_pre] > 0.0 {
                        let rets: Vec<f64> = (lo_pre+1..=i).map(|jj| {
                            if mkt_pr[jj-1] > 0.0 { (mkt_pr[jj] - mkt_pr[jj-1]) / mkt_pr[jj-1] } else { 0.0 }
                        }).collect();
                        let ret_vol = safe_std(&rets);
                        let raw_ret = (ep - mkt_pr[lo_pre]) / mkt_pr[lo_pre];
                        pre_momentum[k] = if ret_vol > 1e-15 {
                            raw_ret / (ret_vol * ((i - lo_pre) as f64).sqrt())
                        } else {
                            0.0
                        };
                    }

                    // Entry percentile in [t-pre, t+h] window
                    let lo_all = match mkt_ts.binary_search(&(t - pre_ns)) {
                        Ok(x) => x, Err(x) => x,
                    };
                    let hi_all = match mkt_ts.binary_search(&(t + h_ns)) {
                        Ok(x) => x + 1, Err(x) => x,
                    };
                    if hi_all > lo_all && hi_all <= n_mkt {
                        let window_prices = &mkt_pr[lo_all..hi_all];
                        let w_min = window_prices.iter().fold(f64::INFINITY, |a, &b| a.min(b));
                        let w_max = window_prices.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                        if w_max > w_min {
                            entry_pct[k] = ((ep - w_min) / (w_max - w_min)).max(0.0).min(1.0);
                        }
                    }

                    // Adverse instant (500ms)
                    let adv_ns = 500 * 1_000_000_i64;
                    let j_adv = match mkt_ts.binary_search(&(t + adv_ns)) {
                        Ok(x) => x, Err(x) => x,
                    };
                    if j_adv < n_mkt && j_adv > i && mkt_pr[j_adv] > 0.0 {
                        adverse[k] = (mkt_pr[j_adv] - ep) / ep * sign_a[k];
                    }

                    // Capture ratio, R/R efficiency, time to profit
                    let fp = fwd_prices[h_idx][i];
                    let fi = fwd_indices[h_idx][i];
                    if fp.is_finite() && fi >= 0 && fi as usize > i {
                        let end_i = fi as usize;
                        let seg = &mkt_pr[i..=end_i];
                        let ar = (fp - ep) / ep * sign_a[k];
                        if ar > 0.0 {
                            let mfe = seg.iter().fold(0.0f64, |a, &b| {
                                let r = if sign_a[k] > 0.0 { (b - ep) / ep } else { (ep - b) / ep };
                                a.max(r.max(0.0))
                            });
                            capture[k] = if mfe > 1e-15 { ar / mfe } else { 1.0 };
                        }
                        let mfe = seg.iter().fold(0.0f64, |a, &b| {
                            let r = if sign_a[k] > 0.0 { (b - ep) / ep } else { (ep - b) / ep };
                            a.max(r.max(0.0))
                        });
                        let mae = seg.iter().fold(0.0f64, |a, &b| {
                            let r = if sign_a[k] > 0.0 { (ep - b) / ep } else { (b - ep) / ep };
                            a.max(r.max(0.0))
                        });
                        rr_eff[k] = if mfe + mae > 1e-15 { mfe / (mfe + mae) } else { 0.5 };

                        // Time to first profit
                        for (jj, &seg_p) in seg.iter().enumerate() {
                            let r = if sign_a[k] > 0.0 { (seg_p - ep) / ep } else { (ep - seg_p) / ep };
                            if r > 0.0 {
                                time_profit[k] = jj as f64 / seg.len() as f64;
                                break;
                            }
                        }
                    }
                }

                result[base] = safe_mean(&pre_momentum);
                result[base + 1] = safe_mean(&entry_pct);
                result[base + 2] = safe_mean(&adverse);
                result[base + 3] = safe_mean(&capture);
                result[base + 4] = safe_mean(&rr_eff);
                result[base + 5] = safe_mean(&time_profit);
            }
        }
    }

    // ===========================================================
    // PILLAR 8: 路径形态 (PC)
    // ===========================================================
    for a in 0..n_agents {
        let idxs_a = &per_agent_idx[a];
        let sign_a = &per_agent_sign[a];
        let n_trades = idxs_a.len();
        if n_trades < 2 {
            continue;
        }

        for (th, &h_sec) in PC_HORIZONS.iter().enumerate() {
            if let Some(h_idx) = find_h_idx(h_sec) {
                let base = a * PER_AGENT_NEW + N_TQ_H * TQ_PER_HORIZON + th * PC_PER_HORIZON;

                let mut hurst_vals = vec![0.0f64; n_trades];
                let mut fractal_vals = vec![0.0f64; n_trades];
                let mut perm_ent_vals = vec![0.0f64; n_trades];
                let mut convexity_vals = vec![0.0f64; n_trades];
                let mut capture_vals = vec![0.0f64; n_trades];
                let mut pain_vals = vec![0.0f64; n_trades];

                for k in 0..n_trades {
                    let i = idxs_a[k] as usize;
                    let ep = mkt_pr[i];
                    let fi = fwd_indices[h_idx][i];
                    if fi < 0 || fi as usize <= i || ep <= 0.0 {
                        continue;
                    }
                    let end_i = fi as usize;
                    let seg = &mkt_pr[i..=end_i];

                    // Aligned returns (subsample if too many points for heavy ops)
                    let aligned_full: Vec<f64> = seg.iter().map(|&p| {
                        if sign_a[k] > 0.0 { (p - ep) / ep } else { (ep - p) / ep }
                    }).collect();
                    let aligned_subsampled: Vec<f64> = if aligned_full.len() > 200 {
                        let step = aligned_full.len() / 200;
                        (0..aligned_full.len()).step_by(step.max(1)).take(200).map(|j| aligned_full[j]).collect()
                    } else {
                        aligned_full.clone()
                    };

                    hurst_vals[k] = if aligned_subsampled.len() >= 16 { hurst_rs(&aligned_subsampled) } else { 0.5 };
                    fractal_vals[k] = fractal_dimension(&aligned_subsampled);
                    perm_ent_vals[k] = if aligned_subsampled.len() >= 4 { permutation_entropy(&aligned_subsampled, 3) } else { 1.0 };
                    convexity_vals[k] = if aligned_subsampled.len() >= 8 { quadratic_convexity(&aligned_subsampled) } else { 0.0 };

                    // Capture ratio and pain ratio use the full aligned array (cheap ops)
                    let mfe = aligned_full.iter().fold(0.0f64, |a, &b| a.max(b));
                    let final_ret = aligned_full.last().copied().unwrap_or(0.0);
                    // capture_vals already computed below
                    let pain_ticks = aligned_full.iter().filter(|&&x| x < -1e-10).count();
                    pain_vals[k] = pain_ticks as f64 / aligned_full.len().max(1) as f64;

                    capture_vals[k] = if mfe > 1e-15 && final_ret > 0.0 { final_ret / mfe } else { 0.0 };
                }

                result[base] = safe_mean(&hurst_vals);
                result[base + 1] = safe_mean(&fractal_vals);
                result[base + 2] = safe_mean(&perm_ent_vals);
                result[base + 3] = safe_mean(&convexity_vals);
                result[base + 4] = safe_mean(&capture_vals);
                result[base + 5] = safe_mean(&pain_vals);
            }
        }
    }

    // ===========================================================
    // PILLAR 9: 连胜连败 (SA)
    // ===========================================================
    for a in 0..n_agents {
        let idxs_a = &per_agent_idx[a];
        let sign_a = &per_agent_sign[a];
        let n_trades = idxs_a.len();
        if n_trades < 4 {
            continue;
        }
        let base_sa = a * PER_AGENT_NEW + N_TQ_H * TQ_PER_HORIZON + N_PC_H * PC_PER_HORIZON;

        // Use 1s forward for win/loss determination
        let h1_idx = match find_h_idx(1.0) {
            Some(x) => x, None => continue,
        };

        let mut wins: Vec<bool> = Vec::with_capacity(n_trades);
        let mut rets: Vec<f64> = Vec::with_capacity(n_trades);
        for k in 0..n_trades {
            let i = idxs_a[k] as usize;
            let fp = fwd_prices[h1_idx][i];
            if fp.is_finite() && mkt_pr[i] > 0.0 {
                let ar = (fp - mkt_pr[i]) / mkt_pr[i] * sign_a[k];
                wins.push(ar > 0.0);
                rets.push(ar);
            }
        }

        if wins.len() < 4 {
            continue;
        }

        let n_w = wins.len();
        // Conditional probabilities
        let mut w_after_w = 0usize;
        let mut w_after_l = 0usize;
        let mut after_w = 0usize;
        let mut after_l = 0usize;
        let mut w_after_2w = 0usize;
        let mut after_2w = 0usize;

        for k in 1..n_w {
            if wins[k - 1] { after_w += 1; if wins[k] { w_after_w += 1; } }
            else { after_l += 1; if wins[k] { w_after_l += 1; } }
            if k >= 2 && wins[k - 2] && wins[k - 1] {
                after_2w += 1;
                if wins[k] { w_after_2w += 1; }
            }
        }

        let p_ww = if after_w > 0 { w_after_w as f64 / after_w as f64 } else { 0.0 };
        let p_lw = if after_l > 0 { w_after_l as f64 / after_l as f64 } else { 0.0 };
        let p_2ww = if after_2w > 0 { w_after_2w as f64 / after_2w as f64 } else { 0.0 };

        result[base_sa] = p_ww;
        result[base_sa + 1] = p_lw;
        result[base_sa + 2] = p_2ww;
        result[base_sa + 3] = runs_test_z(&wins);
        result[base_sa + 4] = transition_entropy(&wins);
        result[base_sa + 5] = lz_complexity_bits(&wins);
        result[base_sa + 6] = p_lw - p_ww; // GF score
        // Ret autocorr
        if rets.len() >= 4 {
            let n_r = rets.len();
            let m_ret = rets.iter().sum::<f64>() / n_r as f64;
            let mut num = 0.0; let mut den = 0.0;
            for k in 0..n_r - 1 {
                num += (rets[k] - m_ret) * (rets[k + 1] - m_ret);
                den += (rets[k] - m_ret).powi(2);
            }
            result[base_sa + 7] = if den > 1e-15 { num / den } else { 0.0 };
        }
        // Conditional vol ratio
        let rets_w: Vec<f64> = (0..n_w-1).filter(|&k| wins[k]).map(|k| rets[k+1]).collect();
        let rets_l: Vec<f64> = (0..n_w-1).filter(|&k| !wins[k]).map(|k| rets[k+1]).collect();
        let std_w = safe_std(&rets_w);
        let std_l = safe_std(&rets_l);
        result[base_sa + 8] = if std_l > 1e-15 { std_w / std_l } else { 1.0 };
    }

    // ===========================================================
    // Precompute all 10 state arrays in O(n_mkt) single pass
    // ===========================================================
    let window_5min_ri = 300 * 1_000_000_000_i64;
    let window_1min_ri = 60 * 1_000_000_000_i64;
    let window_1s_ri = 1_000_000_000_i64;

    let mut lo_5min_ri = vec![0usize; n_mkt];
    let mut lo_1min_ri = vec![0usize; n_mkt];
    let mut lo_1s_ri = vec![0usize; n_mkt];
    {
        let mut j5: usize = 0; let mut j1m: usize = 0; let mut j1s: usize = 0;
        for i in 0..n_mkt {
            let t = mkt_ts[i];
            while j5 < i && mkt_ts[j5] < t - window_5min_ri { j5 += 1; }
            while j1m < i && mkt_ts[j1m] < t - window_1min_ri { j1m += 1; }
            while j1s < i && mkt_ts[j1s] < t - window_1s_ri { j1s += 1; }
            lo_5min_ri[i] = j5;
            lo_1min_ri[i] = j1m;
            lo_1s_ri[i] = j1s;
        }
    }
    let (day_min, day_max) = if n_mkt > 0 {
        let min_p = mkt_pr.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_p = mkt_pr.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        (min_p, max_p)
    } else { (0.0, 0.0) };
    let day_range = day_max - day_min;

    // 10 state arrays using online sliding-window O(n_mkt) algorithms
    let n_states_ri = 10usize;
    let mut state_arrays_2d: Vec<Vec<f64>> = vec![vec![0.0f64; n_mkt]; n_states_ri];

    // Online stats: running sum and sum-of-squares for returns
    let mut rets_queue: std::collections::VecDeque<f64> = std::collections::VecDeque::new();
    let mut ret_sum: f64 = 0.0;
    let mut ret_sum2: f64 = 0.0;
    let mut to_sum: f64 = 0.0;
    let mut vo_sum: f64 = 0.0;
    let mut to_queue: std::collections::VecDeque<(f64, f64)> = std::collections::VecDeque::new(); // (price*vol, vol) for sliding VWAP
    let mut prev_lo5: usize = 0;
    // Additional online accumulators for volume/frequency stats
    let mut vol_1m_queue: std::collections::VecDeque<f64> = std::collections::VecDeque::new();
    let mut vol_1m_sum: f64 = 0.0;
    let mut buy_1s_queue: std::collections::VecDeque<bool> = std::collections::VecDeque::new();
    let mut buy_1s_count: usize = 0;
    let mut prev_lo1m: usize = 0;
    let mut prev_lo1s: usize = 0;

    for i in 0..n_mkt {
        let lo5 = lo_5min_ri[i];
        let lo1m = lo_1min_ri[i];
        let lo1s = lo_1s_ri[i];

        // Add new return at position i
        if i > 0 {
            let r = if mkt_pr[i-1] > 0.0 { (mkt_pr[i] - mkt_pr[i-1]) / mkt_pr[i-1] } else { 0.0 };
            rets_queue.push_back(r);
            ret_sum += r;
            ret_sum2 += r * r;
            to_queue.push_back((mkt_pr[i] * mkt_vo[i], mkt_vo[i]));
            to_sum += mkt_pr[i] * mkt_vo[i];
            vo_sum += mkt_vo[i];
        }

        // Remove old returns that fell out of the 5min window
        while prev_lo5 < lo5 && !rets_queue.is_empty() {
            let old_r = rets_queue.pop_front().unwrap();
            ret_sum -= old_r;
            ret_sum2 -= old_r * old_r;
            let (old_to, old_vo) = to_queue.pop_front().unwrap();
            to_sum -= old_to;
            vo_sum -= old_vo;
            prev_lo5 += 1;
        }
        // Also catch up if lo5 jumped ahead
        while prev_lo5 < lo5 {
            prev_lo5 += 1;
        }

        let n_ret = rets_queue.len();
        // State 0: realized_vol
        if n_ret >= 5 {
            let nf = n_ret as f64;
            state_arrays_2d[0][i] = ((ret_sum2 - ret_sum*ret_sum/nf) / nf).max(0.0).sqrt();
        }
        // State 1: trend_strength
        if n_ret >= 5 && mkt_pr[lo5] > 0.0 {
            let ret = (mkt_pr[i] - mkt_pr[lo5]) / mkt_pr[lo5];
            let vol = state_arrays_2d[0][i].max(1e-15);
            let adj = vol * (n_ret as f64).sqrt();
            state_arrays_2d[1][i] = if adj > 1e-15 { ret.abs() / adj } else { 0.0 };
        }
        // State 2: price_vs_vwap
        if vo_sum > 0.0 {
            let vwap = to_sum / vo_sum;
            state_arrays_2d[2][i] = if vwap > 0.0 { (mkt_pr[i] - vwap) / vwap } else { 0.0 };
        }
        // State 3: daily_range_pos
        state_arrays_2d[3][i] = if day_range > 1e-15 { (mkt_pr[i] - day_min) / day_range } else { 0.5 };
        // State 4: vol_intensity (online)
        // Maintain sliding 1-min volume
        vol_1m_queue.push_back(mkt_vo[i]);
        vol_1m_sum += mkt_vo[i];
        while prev_lo1m < lo1m && !vol_1m_queue.is_empty() {
            vol_1m_sum -= vol_1m_queue.pop_front().unwrap();
            prev_lo1m += 1;
        }
        while prev_lo1m < lo1m { prev_lo1m += 1; }
        if i > lo1m {
            let vol_1m = vol_1m_sum;
            let dt = (mkt_ts[i] - mkt_ts[lo1m]).max(1) as f64;
            let rate = vol_1m / dt;
            let vol_30 = if n_ret > 5 { state_arrays_2d[0][i] } else { rate };
            state_arrays_2d[4][i] = if vol_30 > 0.0 { rate / vol_30 } else { 1.0 };
        } else { state_arrays_2d[4][i] = 1.0; }
        // State 5: buy_vol_ratio (online)
        let is_buy = mkt_fl[i] == 66;
        buy_1s_queue.push_back(is_buy);
        if is_buy { buy_1s_count += 1; }
        while prev_lo1s < lo1s && !buy_1s_queue.is_empty() {
            if buy_1s_queue.pop_front().unwrap() { buy_1s_count -= 1; }
            prev_lo1s += 1;
        }
        while prev_lo1s < lo1s { prev_lo1s += 1; }
        if i > lo1s {
            state_arrays_2d[5][i] = (buy_1s_count as f64) / (i - lo1s).max(1) as f64;
        } else { state_arrays_2d[5][i] = 0.5; }
        // States 6-9: order book
        let ob_i = ob_idx_for_trade[i];
        if ob_i < n_ob {
            state_arrays_2d[6][i] = if ob_bid1[ob_i] > 0.0 { (ob_ask1[ob_i] - ob_bid1[ob_i]) / ob_bid1[ob_i] } else { 0.0 };
            state_arrays_2d[7][i] = ob_imbalance[ob_i];
            if i > lo1s {
                let vol = mkt_vo[lo1s..i].iter().sum::<f64>();
                let d1 = ob_bid_vol1[ob_i] + ob_ask_vol1[ob_i];
                state_arrays_2d[8][i] = if d1 > 0.0 { vol / d1 } else { 0.0 };
            }
            state_arrays_2d[9][i] = (ask_order_ids[i] - bid_order_ids[i]) as f64;
        }
    }

    // ===========================================================
    // PILLAR 10: 状态依赖 (RI) — optimized: lazy state computation
    // ===========================================================
    // Helper: compute a single state value at a given market index

    // Precompute window boundaries using two-pointer (O(n_mkt))
    let window_5min = 300 * 1_000_000_000_i64;
    let window_1min_i = 60 * 1_000_000_000_i64;
    let window_1s_i = 1_000_000_000_i64;

    let mut lo_5min = vec![0usize; n_mkt];
    let mut lo_1min = vec![0usize; n_mkt];
    let mut lo_1s = vec![0usize; n_mkt];
    let mut j5: usize = 0;
    let mut j1m: usize = 0;
    let mut j1s: usize = 0;
    for i in 0..n_mkt {
        let t = mkt_ts[i];
        while j5 < i && mkt_ts[j5] < t - window_5min { j5 += 1; }
        while j1m < i && mkt_ts[j1m] < t - window_1min_i { j1m += 1; }
        while j1s < i && mkt_ts[j1s] < t - window_1s_i { j1s += 1; }
        lo_5min[i] = j5;
        lo_1min[i] = j1m;
        lo_1s[i] = j1s;
    }

    // Day range and cumulative values for state computation
    let (day_min, day_max) = if n_mkt > 0 {
        let min_p = mkt_pr.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_p = mkt_pr.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        (min_p, max_p)
    } else { (0.0, 0.0) };
    let day_range = day_max - day_min;
    let mut cum_to_state: f64 = 0.0;
    let mut cum_vo_state: f64 = 0.0;
    for i in 0..n_mkt.min(10000) {
        cum_to_state += mkt_pr[i] * mkt_vo[i];
        cum_vo_state += mkt_vo[i];
    }
    // Approximate full-day cum by sampling
    if n_mkt > 10000 {
        let step = n_mkt / 10000;
        for i in (10000..n_mkt).step_by(step.max(1)) {
            cum_to_state += mkt_pr[i] * mkt_vo[i] * step as f64;
            cum_vo_state += mkt_vo[i] * step as f64;
        }
    }

    // Simplified state names (most important ones)
    let ri_simple_states: [&str; 10] = [
        "realized_vol", "trend_strength", "price_vs_vwap", "daily_range_pos",
        "vol_intensity", "buy_vol_ratio",
        "norm_spread", "ob_imb", "depth_consumption", "order_id_gap"
    ];

    for a in 0..n_agents {
        let idxs_a = &per_agent_idx[a];
        let sign_a = &per_agent_sign[a];
        let n_trades = idxs_a.len();
        if n_trades < 10 { continue; }

        for (s_idx, &state_name) in ri_simple_states.iter().enumerate() {
            // Look up state values from precomputed array
            let s_arr = &state_arrays_2d[s_idx];
            let mut trade_states: Vec<(usize, f64)> = Vec::with_capacity(n_trades);
            for k in 0..n_trades {
                let idx = idxs_a[k] as usize;
                let val = s_arr[idx];
                if val.is_finite() {
                    trade_states.push((k, val));
                }
            }
            if trade_states.len() < 10 { continue; }
            trade_states.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            let n = trade_states.len();
            let mid = n / 2;

            for (rh, &h_sec) in RI_HORIZONS.iter().enumerate() {
                if let Some(h_idx) = find_h_idx(h_sec) {
                    let base = a * PER_AGENT_NEW
                        + N_TQ_H * TQ_PER_HORIZON
                        + N_PC_H * PC_PER_HORIZON
                        + SA_PER_AGENT
                        + s_idx * N_RI_H * RI_PER_HORIZON
                        + rh * RI_PER_HORIZON;

                    let mut low_rets = Vec::with_capacity(mid);
                    let mut low_hits = 0usize;
                    for &(k, _) in &trade_states[..mid] {
                        let i = idxs_a[k] as usize;
                        let fp = fwd_prices[h_idx][i];
                        if fp.is_finite() && mkt_pr[i] > 0.0 {
                            let ar = (fp - mkt_pr[i]) / mkt_pr[i] * sign_a[k];
                            low_rets.push(ar);
                            if ar > 0.0 { low_hits += 1; }
                        }
                    }
                    let mut high_rets = Vec::with_capacity(n - mid);
                    let mut high_hits = 0usize;
                    for &(k, _) in &trade_states[mid..] {
                        let i = idxs_a[k] as usize;
                        let fp = fwd_prices[h_idx][i];
                        if fp.is_finite() && mkt_pr[i] > 0.0 {
                            let ar = (fp - mkt_pr[i]) / mkt_pr[i] * sign_a[k];
                            high_rets.push(ar);
                            if ar > 0.0 { high_hits += 1; }
                        }
                    }
                    result[base] = safe_mean(&high_rets) - safe_mean(&low_rets);
                    result[base + 1] = if !high_rets.is_empty() && !low_rets.is_empty() {
                        (high_hits as f64 / high_rets.len() as f64) - (low_hits as f64 / low_rets.len() as f64)
                    } else { 0.0 };
                }
            }
        }
    }    // ===========================================================
    // PILLAR 11: 极端尾部 (ET)
    // ===========================================================
    for a in 0..n_agents {
        let idxs_a = &per_agent_idx[a];
        let sign_a = &per_agent_sign[a];
        let n_trades = idxs_a.len();
        if n_trades < 4 {
            continue;
        }
        let base_et = a * PER_AGENT_NEW
            + N_TQ_H * TQ_PER_HORIZON
            + N_PC_H * PC_PER_HORIZON
            + SA_PER_AGENT
            + N_STATES * N_RI_H * RI_PER_HORIZON;

        // Use 1s forward
        let h1_idx = match find_h_idx(1.0) {
            Some(x) => x, None => continue,
        };
        let mut rets: Vec<f64> = Vec::with_capacity(n_trades);
        for k in 0..n_trades {
            let i = idxs_a[k] as usize;
            let fp = fwd_prices[h1_idx][i];
            if fp.is_finite() && mkt_pr[i] > 0.0 {
                rets.push((fp - mkt_pr[i]) / mkt_pr[i] * sign_a[k]);
            }
        }
        if rets.len() < 10 {
            continue;
        }
        rets.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = rets.len();
        let top10 = (n as f64 * 0.9) as usize;
        let bot10 = (n as f64 * 0.1) as usize;

        let top_mean = if top10 < n { safe_mean(&rets[top10..]) } else { 0.0 };
        let bot_mean = if bot10 > 0 { safe_mean(&rets[..bot10]) } else { 0.0 };

        result[base_et] = if bot_mean.abs() > 1e-15 { top_mean / bot_mean.abs() } else { 0.0 };
        result[base_et + 1] = if bot10 > 0 { safe_mean(&rets[..bot10]) } else { 0.0 }; // Expected shortfall
        result[base_et + 2] = rets.last().copied().unwrap_or(0.0); // max
        result[base_et + 3] = rets.first().copied().unwrap_or(0.0); // min
    }

    // ===========================================================
    // PILLAR 12: 拥挤度 (CD)
    // ===========================================================
    for a in 0..n_agents {
        let idxs_a = &per_agent_idx[a];
        let sign_a = &per_agent_sign[a];
        let n_trades = idxs_a.len();
        if n_trades < 4 {
            continue;
        }
        let base_cd = a * PER_AGENT_NEW
            + N_TQ_H * TQ_PER_HORIZON
            + N_PC_H * PC_PER_HORIZON
            + SA_PER_AGENT
            + N_STATES * N_RI_H * RI_PER_HORIZON
            + ET_PER_AGENT;

        let h1_idx = match find_h_idx(1.0) {
            Some(x) => x, None => continue,
        };

        let crowd_window = 1_000_000_000_i64; // 1s window
        let mut crowding_ratios = vec![0.0f64; n_trades];
        let mut rets_1s = vec![0.0f64; n_trades];

        for k in 0..n_trades {
            let i = idxs_a[k] as usize;
            let t = mkt_ts[i];
            let fp = fwd_prices[h1_idx][i];
            if fp.is_finite() && mkt_pr[i] > 0.0 {
                rets_1s[k] = (fp - mkt_pr[i]) / mkt_pr[i] * sign_a[k];
            }
            // Count other agents trading in same direction within 1s
            let mut same = 0usize;
            let mut total_other = 0usize;
            for b in 0..n_agents {
                if b == a { continue; }
                let idxs_b = &per_agent_idx[b];
                for (pos_b, &j) in idxs_b.iter().enumerate() {
                    let t_b = mkt_ts[j as usize];
                    if (t_b - t).abs() <= crowd_window / 2 {
                        total_other += 1;
                        if per_agent_sign[b][pos_b] * sign_a[k] > 0.0 {
                            same += 1;
                        }
                        break;
                    }
                }
            }
            crowding_ratios[k] = if total_other > 0 { same as f64 / total_other as f64 } else { 0.0 };
        }

        let median_cr = {
            let mut crs: Vec<f64> = crowding_ratios.iter().filter(|v| v.is_finite()).copied().collect();
            crs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            if crs.is_empty() { 0.0 } else { crs[crs.len() / 2] }
        };

        let crowd_high: Vec<bool> = crowding_ratios.iter().map(|&c| c > median_cr).collect();
        let crowd_low: Vec<bool> = crowding_ratios.iter().map(|&c| c <= median_cr).collect();

        let crow_ret: Vec<f64> = rets_1s.iter().zip(crowd_high.iter()).filter(|(_, &h)| h).map(|(&r, _)| r).collect();
        let solo_ret: Vec<f64> = rets_1s.iter().zip(crowd_low.iter()).filter(|(_, &h)| h).map(|(&r, _)| r).collect();

        let crow_hit = crow_ret.iter().filter(|&&r| r > 0.0).count();
        let solo_hit = solo_ret.iter().filter(|&&r| r > 0.0).count();

        result[base_cd] = safe_mean(&crowding_ratios);
        result[base_cd + 1] = if !crow_ret.is_empty() { crow_hit as f64 / crow_ret.len() as f64 } else { 0.0 };
        result[base_cd + 2] = if !solo_ret.is_empty() { solo_hit as f64 / solo_ret.len() as f64 } else { 0.0 };
        result[base_cd + 3] = result[base_cd + 1] - result[base_cd + 2];
        // Herding corr
        let herding: Vec<f64> = (0..n_trades).filter_map(|k| {
            let mut same_count = 0usize;
            let mut total = 0usize;
            for b in 0..n_agents {
                if b == a { continue; }
                // Check if agent b traded near the same time
                let t = mkt_ts[idxs_a[k] as usize];
                for &j in &per_agent_idx[b] {
                    if (mkt_ts[j as usize] - t).abs() <= crowd_window / 2 {
                        total += 1;
                        break;
                    }
                }
            }
            if total > 0 { Some(same_count as f64 / total as f64) } else { None }
        }).collect();
        result[base_cd + 4] = safe_mean(&herding);
    }

    // ===========================================================
    // PILLAR 13: 领先滞后 (LF) — cross-agent
    // ===========================================================
    let base_lf = n_agents * PER_AGENT_NEW;
    for a in 0..n_agents {
        for b in (a + 1)..n_agents {
            let idx_a = &per_agent_idx[a];
            let idx_b = &per_agent_idx[b];
            let sign_a = &per_agent_sign[a];
            let sign_b = &per_agent_sign[b];
            let p_idx = a * (2 * n_agents - a - 1) / 2 + (b - a - 1);
            let base = base_lf + p_idx * LF_PER_PAIR;

            if idx_a.len() < 5 || idx_b.len() < 5 {
                continue;
            }

            // Aggregate directions into 1-second bins for signal processing
            let t0 = mkt_ts[0];
            let t_end = mkt_ts[n_mkt - 1];
            let n_bins = ((t_end - t0) / 1_000_000_000_i64 + 1).max(1) as usize;
            let n_bins_clipped = n_bins.min(14400); // max 4 hours
            let mut sig_a = vec![0.0f64; n_bins_clipped];
            let mut sig_b = vec![0.0f64; n_bins_clipped];

            for (pos, &i) in idx_a.iter().enumerate() {
                let bin = ((mkt_ts[i as usize] - t0) / 1_000_000_000_i64).min(n_bins_clipped as i64 - 1).max(0) as usize;
                sig_a[bin] += sign_a[pos];
            }
            for (pos, &i) in idx_b.iter().enumerate() {
                let bin = ((mkt_ts[i as usize] - t0) / 1_000_000_000_i64).min(n_bins_clipped as i64 - 1).max(0) as usize;
                sig_b[bin] += sign_b[pos];
            }
            // Clip to [-1, 1]
            for v in sig_a.iter_mut() { *v = v.max(-1.0).min(1.0); }
            for v in sig_b.iter_mut() { *v = v.max(-1.0).min(1.0); }

            // TE
            let te_ab = transfer_entropy_dir(&sig_a, &sig_b);
            let te_ba = transfer_entropy_dir(&sig_b, &sig_a);
            result[base] = te_ab - te_ba; // te_asymmetry

            // Event Sync Q
            let ts_a: Vec<i64> = idx_a.iter().map(|&i| mkt_ts[i as usize]).collect();
            let ts_b: Vec<i64> = idx_b.iter().map(|&i| mkt_ts[i as usize]).collect();
            let q_ab = event_sync_q(&ts_a, &ts_b);
            let q_ba = event_sync_q(&ts_b, &ts_a);
            result[base + 1] = q_ab - q_ba; // es_q_asymmetry

            // Simplified PLV: use cross-correlation peak timing instead of Hilbert
            let n_sig = sig_a.len().min(sig_b.len());
            let mut xcorr_max = 0.0f64;
            let mut best_lag: i64 = 0;
            let max_lag = 10i64; // +/-10 second lags
            for lag in -max_lag..=max_lag {
                let mut corr = 0.0f64;
                let mut count = 0usize;
                for i_val in 0..n_sig {
                    let j_val = i_val as i64 + lag;
                    if j_val >= 0 && j_val < n_sig as i64 {
                        corr += sig_a[i_val] * sig_b[j_val as usize];
                        count += 1;
                    }
                }
                let avg = if count > 0 { corr / count as f64 } else { 0.0 };
                if avg > xcorr_max { xcorr_max = avg; best_lag = lag; }
            }
            result[base + 2] = xcorr_max; // plv approximation via max xcorr
            result[base + 3] = best_lag as f64 / max_lag as f64; // normalized lead (-1 to 1)

            // Feedback lead: A's forward return sign predicts B's subsequent return
            let h1_idx = match find_h_idx(1.0) {
                Some(x) => x, None => continue,
            };
            let mut feedback_ret = vec![0.0f64; 0];
            let mut feedback_hit = vec![0.0f64; 0];
            for k in 0..idx_a.len() {
                let i = idx_a[k] as usize;
                let fp = fwd_prices[h1_idx][i];
                if !fp.is_finite() || mkt_pr[i] <= 0.0 {
                    continue;
                }
                let r_a = (fp - mkt_pr[i]) / mkt_pr[i] * sign_a[k];
                let sgn = if r_a > 0.0 { 1.0 } else { -1.0 };
                // Find B's trades in next 5s
                let t_a = mkt_ts[i];
                let t_end_b = t_a + 5 * 1_000_000_000_i64;
                let j_start = match mkt_ts.binary_search(&t_a) {
                    Ok(x) => x, Err(x) => x,
                };
                let j_end = match mkt_ts.binary_search(&t_end_b) {
                    Ok(x) => x + 1, Err(x) => x,
                };
                // Find B trades in this window
                for &j in idx_b.iter() {
                    let j_u = j as usize;
                    if j_u >= j_start && j_u < j_end && j_u < n_mkt {
                        let fp_b = fwd_prices[h1_idx][j_u];
                        if fp_b.is_finite() && mkt_pr[j_u] > 0.0 {
                            if let Some(b_pos) = idx_b.iter().position(|&x| x == j) {
                                let r_b = (fp_b - mkt_pr[j_u]) / mkt_pr[j_u] * sign_b[b_pos];
                                feedback_ret.push(sgn * r_b);
                                feedback_hit.push(if (sgn > 0.0 && r_b > 0.0) || (sgn < 0.0 && r_b < 0.0) { 1.0 } else { 0.0 });
                            }
                        }
                        break; // only first B trade in window
                    }
                }
            }
            if feedback_ret.len() >= 4 {
                result[base + 4] = safe_mean(&feedback_ret);
                result[base + 5] = safe_mean(&feedback_hit);
            }

            // Feedback TE
            if feedback_ret.len() >= 10 {
                let mut sig_fb: Vec<f64> = feedback_ret.iter().map(|&r| r.signum()).collect();
                // Ensure same length for TE computation
                let min_len = sig_a.len().min(sig_fb.len());
                sig_fb.truncate(min_len);
                let sig_a_trunc = &sig_a[..min_len];
                result[base + 6] = transfer_entropy_dir(sig_a_trunc, &sig_fb);
            }
        }
    }

    result
}
