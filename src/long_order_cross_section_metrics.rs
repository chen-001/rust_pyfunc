//! 樫截面因子：漫长订单（hm27 思路的横截面化，全维度展开版）
//!
//! # 核心思想
//! 每个 LimitOrder 被 0~N 笔成交吃掉，「存续时长」= 末笔 − 首笔（秒）。
//! 漫长订单（分批、跨较长时间成交）往往是隐藏大单 / 冰山单 / 机构分仓的代理。
//! 全市场统一阈值档 + 对 ln(总成交额) 回归中性化，剥离流动性共线。
//!
//! # 38 个因子（10 个独立逻辑族）
//! ## 族1 漫长占比 (0-11)：4 阈值档 {30,60,180,600}s × {卖,买,净} = 12
//! ## 族2 大单方向 (12-15)：4 口径 {p80,p90,abs50w,abs100w} 净不平衡 = 4
//! ## 族3 大单耐心度 (16-17)：corr(dur, amt) 卖/买 = 2
//! ## 族4 核心资金 (18-20)：漫长∩大单 交集 = 3
//! ## 族5 份额背离 (21-24)：漫长份额 + 笔数-金额背离 卖/买 = 4
//! ## 族6 时段耐心 (25-28)：开盘/收盘 × {60,180}s 净不平衡 = 4
//! ## 族7 存续均值 (29-32)：非单点均值 + 金额加权均值 卖/买 = 4
//! ## 族8 耐心溢价 (33-34)：dur_p95/p99 = 2
//! ## 族9 分仓粒度 (35-36)：漫长订单平均成交笔数 卖/买 = 2
//! ## 族10 首笔时机 (37)：漫长卖单首笔相对时机 = 1

use crate::fast_csv_reader::{read_trade_fast_inner, TradeRecord};
use pyo3::prelude::*;
use rayon::prelude::*;
use std::collections::BTreeSet;
use std::fs;

/// 每只股票输出因子数（38→31：剔除 30s 整档 / cross_60_p80 / cross_180_p80 /
/// cnt_amt_div_ask/buy 等高相关冗余，剩余最高截面相关 0.894）。
pub const N_FACTORS: usize = 31;

/// 漫长订单阈值档（秒）。保留 60/180/600 三档（30s 与 60s 相关>0.94 已剔）。
const THR_SECS: [f64; 3] = [60.0, 180.0, 600.0];
/// 开盘/收盘窗口长度（微秒）= 30 分钟
const WINDOW_US: i64 = 30 * 60 * 1_000_000;

/// 每个因子是否需要 ln(总成交额) 流动性中性化。
/// 净额类（lsi/big_net/时段lsi/时机）天然去共线，不做回归。
const NEED_NEUTRALIZE: [bool; N_FACTORS] = [
    // 族1: lsp,lbp,lsi × 3档(60/180/600)
    true, true, false, true, true, false, true, true, false,
    // 族2: big_net 4口径
    false, false, false, false,
    // 族3: dac 卖/买
    true, true,
    // 族4: 核心资金 cross_60_abs
    true,
    // 族5: 漫长份额 卖/买
    true, true,
    // 族6: 时段lsi 4
    false, false, false, false,
    // 族7: 存续均值 4
    true, true, true, true,
    // 族8: 耐心溢价 2
    true, true,
    // 族9: 分仓粒度 2
    true, true,
    // 族10: 首笔时机 1
    false,
];

#[derive(Clone, Copy, Default)]
struct Agg {
    first_us: i64,
    last_us: i64,
    amt: f64,
    cnt: i64,
    init: bool,
}

pub fn list_codes(date: i64, subdir: &str) -> Vec<String> {
    let dir = format!("/ssd_data/stock/{date}/{subdir}");
    let mut set = BTreeSet::new();
    if let Ok(entries) = fs::read_dir(&dir) {
        for e in entries.flatten() {
            let name = e.file_name().into_string().unwrap_or_default();
            if let Some(code) = name.split('_').next() {
                if code.chars().all(|c| c.is_ascii_digit()) {
                    set.insert(code.to_string());
                }
            }
        }
    }
    set.into_iter().collect()
}

/// 按 order id 聚合（可选时间窗口），返回 HashMap。
/// window: Some((lo,hi))，lo=0 表示上界窗口(开盘 time_us<hi)，hi=MAX 表示下界窗口(收盘 time_us>lo)。
fn aggregate(
    trades: &[TradeRecord],
    get_id: impl Fn(&TradeRecord) -> i64,
    window: Option<(i64, i64)>,
) -> std::collections::HashMap<i64, Agg> {
    let mut m: std::collections::HashMap<i64, Agg> =
        std::collections::HashMap::with_capacity(trades.len() / 2 + 1);
    for t in trades {
        if t.flag != 66 && t.flag != 83 {
            continue;
        }
        if let Some((lo, hi)) = window {
            if lo > 0 && t.time_us <= lo {
                continue;
            }
            if hi < i64::MAX && t.time_us >= hi {
                continue;
            }
        }
        let id = get_id(t);
        if id <= 0 {
            continue;
        }
        let e = m.entry(id).or_default();
        if !e.init {
            e.first_us = t.time_us;
            e.last_us = t.time_us;
            e.amt = t.turnover as f64;
            e.cnt = 1;
            e.init = true;
        } else {
            if t.time_us < e.first_us {
                e.first_us = t.time_us;
            }
            if t.time_us > e.last_us {
                e.last_us = t.time_us;
            }
            e.amt += t.turnover as f64;
            e.cnt += 1;
        }
    }
    m
}

/// 提取 (duration_秒, amount, 成交笔数) 配对数组（同序）。
fn extract3(m: &std::collections::HashMap<i64, Agg>) -> (Vec<f64>, Vec<f64>, Vec<i64>) {
    let n = m.len();
    let mut dur = Vec::with_capacity(n);
    let mut amt = Vec::with_capacity(n);
    let mut cnt = Vec::with_capacity(n);
    for e in m.values() {
        dur.push((e.last_us - e.first_us) as f64 / 1e6);
        amt.push(e.amt);
        cnt.push(e.cnt);
    }
    (dur, amt, cnt)
}

/// Σ(dur>thr 的 amount)
fn sum_amt_above(dur: &[f64], amt: &[f64], thr: f64) -> f64 {
    dur.iter()
        .zip(amt.iter())
        .filter(|(d, _)| **d > thr)
        .map(|(_, a)| *a)
        .sum()
}
/// Σ(dur>dthr 且 amt>athr 的 amount)
fn sum_amt_cross(dur: &[f64], amt: &[f64], dthr: f64, athr: f64) -> f64 {
    dur.iter()
        .zip(amt.iter())
        .filter(|(d, a)| **d > dthr && **a > athr)
        .map(|(_, a)| *a)
        .sum()
}
fn percentile(v: &[f64], q: f64) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    let mut s: Vec<f64> = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = ((s.len() - 1) as f64 * q).round() as usize;
    s[idx.min(s.len() - 1)]
}

fn corr(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len();
    if n < 3 {
        return 0.0;
    }
    let nf = n as f64;
    let mx = x.iter().sum::<f64>() / nf;
    let my = y.iter().sum::<f64>() / nf;
    let (mut cov, mut vx, mut vy) = (0.0f64, 0.0f64, 0.0f64);
    for i in 0..n {
        let dx = x[i] - mx;
        let dy = y[i] - my;
        cov += dx * dy;
        vx += dx * dx;
        vy += dy * dy;
    }
    let d = (vx * vy).sqrt();
    if d < 1e-12 {
        0.0
    } else {
        cov / d
    }
}

fn mean_pos_dur(dur: &[f64]) -> f64 {
    let mut s = 0.0f64;
    let mut n = 0u64;
    for &d in dur {
        if d > 0.0 {
            s += d;
            n += 1;
        }
    }
    if n == 0 {
        0.0
    } else {
        s / n as f64
    }
}

/// 漫长(dur>thr)订单的平均成交笔数（分仓粒度）。
fn mean_cnt_long(cnt: &[i64], dur: &[f64], thr: f64) -> f64 {
    let mut s = 0.0f64;
    let mut n = 0u64;
    for (c, d) in cnt.iter().zip(dur.iter()) {
        if *d > thr {
            s += *c as f64;
            n += 1;
        }
    }
    if n == 0 {
        0.0
    } else {
        s / n as f64
    }
}

fn window_lsi(
    ask: &std::collections::HashMap<i64, Agg>,
    bid: &std::collections::HashMap<i64, Agg>,
    total_amt: f64,
    thr: f64,
) -> f64 {
    let lsp: f64 = ask
        .values()
        .filter(|e| (e.last_us - e.first_us) as f64 / 1e6 > thr)
        .map(|e| e.amt)
        .sum::<f64>()
        / total_amt;
    let lbp: f64 = bid
        .values()
        .filter(|e| (e.last_us - e.first_us) as f64 / 1e6 > thr)
        .map(|e| e.amt)
        .sum::<f64>()
        / total_amt;
    lbp - lsp
}

/// per-stock 算 38 个原始因子 + ln(总成交额)。
fn per_stock(trades: &[TradeRecord]) -> Option<([f64; N_FACTORS], f64)> {
    if trades.len() < 50 {
        return None;
    }
    let mut total_amt = 0.0f64;
    let mut first_us = i64::MAX;
    let mut last_us = i64::MIN;
    for t in trades {
        if t.flag != 66 && t.flag != 83 {
            continue;
        }
        total_amt += t.turnover as f64;
        if t.time_us < first_us {
            first_us = t.time_us;
        }
        if t.time_us > last_us {
            last_us = t.time_us;
        }
    }
    if total_amt <= 0.0 || first_us >= last_us {
        return None;
    }

    let ask_map = aggregate(trades, |t| t.ask_order, None);
    let bid_map = aggregate(trades, |t| t.bid_order, None);
    let (adur, aamt, acnt) = extract3(&ask_map);
    let (bdur, bamt, bcnt) = extract3(&bid_map);
    if aamt.is_empty() || bamt.is_empty() {
        return None;
    }

    let mut f = [0.0f64; N_FACTORS];
    let a_p80 = percentile(&aamt, 0.80);
    let a_p90 = percentile(&aamt, 0.90);
    let b_p80 = percentile(&bamt, 0.80);
    let b_p90 = percentile(&bamt, 0.90);

    // 族1 (0-11): 4 档 lsp/lbp/lsi
    for (i, &thr) in THR_SECS.iter().enumerate() {
        let lsp = sum_amt_above(&adur, &aamt, thr) / total_amt;
        let lbp = sum_amt_above(&bdur, &bamt, thr) / total_amt;
        f[i * 3] = lsp;
        f[i * 3 + 1] = lbp;
        f[i * 3 + 2] = lbp - lsp;
    }

    // 族2 (9-12): 大单方向不平衡（4 口径）
    let big = |amt: &[f64], thr: f64| amt.iter().filter(|&&a| a > thr).sum::<f64>() / total_amt;
    f[9] = big(&bamt, b_p80) - big(&aamt, a_p80);
    f[10] = big(&bamt, b_p90) - big(&aamt, a_p90);
    f[11] = big(&bamt, 5e5) - big(&aamt, 5e5);
    f[12] = big(&bamt, 1e6) - big(&aamt, 1e6);

    // 族3 (13-14): 大单耐心度
    f[13] = corr(&adur, &aamt);
    f[14] = corr(&bdur, &bamt);

    // 族4 (15): 漫长∩大单 核心资金（仅绝对口径；p80 口径与 lsp 镜像已剔）
    f[15] = sum_amt_cross(&adur, &aamt, 60.0, 5e5) / total_amt;

    // 族5 (21-24): 漫长份额 + 笔数-金额背离
    let pos_a: f64 = aamt
        .iter()
        .zip(adur.iter())
        .filter(|(_, d)| **d > 0.0)
        .map(|(a, _)| *a)
        .sum();
    let pos_b: f64 = bamt
        .iter()
        .zip(bdur.iter())
        .filter(|(_, d)| **d > 0.0)
        .map(|(a, _)| *a)
        .sum();
    let l60_a = sum_amt_above(&adur, &aamt, 60.0);
    let l60_b = sum_amt_above(&bdur, &bamt, 60.0);
    f[16] = if pos_a > 0.0 { l60_a / pos_a } else { 0.0 };
    f[17] = if pos_b > 0.0 { l60_b / pos_b } else { 0.0 };

    // 族6 (25-28): 开盘/收盘 × {60,180}s 净不平衡
    let open_hi = first_us + WINDOW_US;
    let close_lo = last_us - WINDOW_US;
    let oa = aggregate(trades, |t| t.ask_order, Some((0, open_hi)));
    let ob = aggregate(trades, |t| t.bid_order, Some((0, open_hi)));
    let ca = aggregate(trades, |t| t.ask_order, Some((close_lo, i64::MAX)));
    let cb = aggregate(trades, |t| t.bid_order, Some((close_lo, i64::MAX)));
    f[18] = window_lsi(&oa, &ob, total_amt, 60.0);
    f[19] = window_lsi(&ca, &cb, total_amt, 60.0);
    f[20] = window_lsi(&oa, &ob, total_amt, 180.0);
    f[21] = window_lsi(&ca, &cb, total_amt, 180.0);

    // 族7 (29-32): 非单点均值 + 金额加权均值
    f[22] = mean_pos_dur(&adur);
    f[23] = mean_pos_dur(&bdur);
    let sa: f64 = aamt.iter().sum();
    let sb: f64 = bamt.iter().sum();
    f[24] = if sa > 0.0 {
        adur.iter().zip(aamt.iter()).map(|(d, a)| d * a).sum::<f64>() / sa
    } else {
        0.0
    };
    f[25] = if sb > 0.0 {
        bdur.iter().zip(bamt.iter()).map(|(d, a)| d * a).sum::<f64>() / sb
    } else {
        0.0
    };

    // 族8 (33-34): 耐心溢价原始量（横截面再中性化+z）
    f[26] = percentile(&adur, 0.95);
    f[27] = percentile(&adur, 0.99);

    // 族9 (35-36): 漫长订单分仓粒度
    f[28] = mean_cnt_long(&acnt, &adur, 60.0);
    f[29] = mean_cnt_long(&bcnt, &bdur, 60.0);

    // 族10 (37): 漫长卖单首笔时机（0=开盘, 1=收盘）
    let span = (last_us - first_us) as f64;
    if span > 0.0 {
        let mut s = 0.0f64;
        let mut c = 0u64;
        for e in ask_map.values() {
            if (e.last_us - e.first_us) as f64 / 1e6 > 60.0 {
                s += (e.first_us - first_us) as f64;
                c += 1;
            }
        }
        f[30] = if c > 0 { (s / c as f64) / span } else { 0.5 };
    } else {
        f[30] = 0.5;
    }

    for v in f.iter_mut() {
        if !v.is_finite() {
            *v = 0.0;
        }
    }
    Some((f, total_amt.ln()))
}

// ============================================================
// 横截面运算
// ============================================================

fn neutralize_vs(y: &[f64], x: &[f64]) -> Vec<f64> {
    let finite: Vec<(f64, f64)> = y
        .iter()
        .zip(x.iter())
        .filter(|(a, b)| a.is_finite() && b.is_finite())
        .map(|(a, b)| (*a, *b))
        .collect();
    if finite.len() < 3 {
        return y.to_vec();
    }
    let n = finite.len() as f64;
    let mx = finite.iter().map(|(_, x)| x).sum::<f64>() / n;
    let my = finite.iter().map(|(y, _)| y).sum::<f64>() / n;
    let (mut sxy, mut sxx) = (0.0f64, 0.0f64);
    for (yv, xv) in &finite {
        sxy += (xv - mx) * (yv - my);
        sxx += (xv - mx) * (xv - mx);
    }
    if sxx < 1e-12 {
        return y.to_vec();
    }
    let b = sxy / sxx;
    let a = my - b * mx;
    y.iter()
        .zip(x.iter())
        .map(|(yv, xv)| yv - (a + b * xv))
        .collect()
}

fn zscore(vals: &mut [f64]) {
    let finite: Vec<f64> = vals.iter().filter(|v| v.is_finite()).copied().collect();
    if finite.len() < 2 {
        for v in vals.iter_mut() {
            if !v.is_finite() {
                *v = 0.0;
            }
        }
        return;
    }
    let mean = finite.iter().sum::<f64>() / finite.len() as f64;
    let var = finite.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / finite.len() as f64;
    let std = var.sqrt();
    if std < 1e-12 {
        for v in vals.iter_mut() {
            *v = 0.0;
        }
        return;
    }
    for v in vals.iter_mut() {
        *v = if v.is_finite() { (*v - mean) / std } else { 0.0 };
    }
}

/// 核心唯一真相源：读全市场 → per-stock 因子 → 中性化 + z-score → (codes, vals)。
pub fn compute_long_order_full(date: i64) -> std::io::Result<(Vec<String>, Vec<f32>)> {
    let codes = list_codes(date, "transaction");
    let results: Vec<Option<([f64; N_FACTORS], f64)>> = codes
        .par_iter()
        .map(|code| {
            let trades = read_trade_fast_inner(code, date, false, true, usize::MAX).ok()?;
            per_stock(&trades)
        })
        .collect();

    let mut valid_codes: Vec<String> = Vec::new();
    let mut feats: Vec<[f64; N_FACTORS]> = Vec::new();
    let mut ln_tots: Vec<f64> = Vec::new();
    for (code, r) in codes.iter().zip(results.iter()) {
        if let Some((f, ln)) = r {
            valid_codes.push(code.clone());
            feats.push(*f);
            ln_tots.push(*ln);
        }
    }
    let n = valid_codes.len();
    if n == 0 {
        return Ok((Vec::new(), Vec::new()));
    }

    for j in 0..N_FACTORS {
        let col: Vec<f64> = feats.iter().map(|f| f[j]).collect();
        let mut processed = if NEED_NEUTRALIZE[j] {
            neutralize_vs(&col, &ln_tots)
        } else {
            col
        };
        zscore(&mut processed);
        for (i, &v) in processed.iter().enumerate() {
            feats[i][j] = v;
        }
    }

    let mut out_vals = Vec::with_capacity(n * N_FACTORS);
    for f in &feats {
        out_vals.extend(f.iter().map(|&v| v as f32));
    }
    Ok((valid_codes, out_vals))
}

/// 因子名（与 N_FACTORS 严格对齐）。
pub fn long_order_names() -> Vec<String> {
    vec![
        // 族1 漫长占比（3档×3：60/180/600s）
        "lsp_60".into(), "lbp_60".into(), "lsi_60".into(),
        "lsp_180".into(), "lbp_180".into(), "lsi_180".into(),
        "lsp_600".into(), "lbp_600".into(), "lsi_600".into(),
        // 族2 大单方向（4口径）
        "big_net_p80".into(), "big_net_p90".into(), "big_net_abs50w".into(), "big_net_abs100w".into(),
        // 族3 大单耐心度
        "dac_ask".into(), "dac_bid".into(),
        // 族4 核心资金（绝对口径；p80 口径与 lsp 镜像已剔）
        "cross_60_abs".into(),
        // 族5 漫长份额
        "long_share_ask".into(), "long_share_buy".into(),
        // 族6 时段耐心
        "open_lsi_60".into(), "close_lsi_60".into(), "open_lsi_180".into(), "close_lsi_180".into(),
        // 族7 存续均值
        "mean_dur_pos_ask".into(), "mean_dur_pos_buy".into(),
        "amt_w_dur_ask".into(), "amt_w_dur_buy".into(),
        // 族8 耐心溢价
        "persist_p95".into(), "persist_p99".into(),
        // 族9 分仓粒度
        "long_cnt_ask".into(), "long_cnt_buy".into(),
        // 族10 首笔时机
        "long_first_offset".into(),
    ]
}

#[pyfunction]
pub fn py_long_order(py: Python<'_>, date: i64) -> PyResult<(Vec<String>, Vec<f32>)> {
    compute_long_order_full(date)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("{e:?}")))
}

#[pyfunction]
pub fn py_long_order_names() -> Vec<String> {
    long_order_names()
}

/// 调试用：返回原始（未中性化）因子值 + ln(总成交额)。
pub fn collect_raw(date: i64) -> std::io::Result<(Vec<String>, Vec<f32>, Vec<f32>)> {
    let codes = list_codes(date, "transaction");
    let results: Vec<Option<([f64; N_FACTORS], f64)>> = codes
        .par_iter()
        .map(|code| {
            let trades = read_trade_fast_inner(code, date, false, true, usize::MAX).ok()?;
            per_stock(&trades)
        })
        .collect();
    let mut vc = Vec::new();
    let mut vals = Vec::new();
    let mut lns = Vec::new();
    for (code, r) in codes.iter().zip(results.iter()) {
        if let Some((f, ln)) = r {
            vc.push(code.clone());
            vals.extend(f.iter().map(|&v| v as f32));
            lns.push(*ln as f32);
        }
    }
    Ok((vc, vals, lns))
}

#[pyfunction]
pub fn py_long_order_raw(
    py: Python<'_>,
    date: i64,
) -> PyResult<(Vec<String>, Vec<f32>, Vec<f32>)> {
    collect_raw(date).map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("{e:?}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_names_count() {
        assert_eq!(long_order_names().len(), N_FACTORS);
        assert_eq!(NEED_NEUTRALIZE.len(), N_FACTORS);
    }

    #[test]
    fn test_neutralize_removes_linear_trend() {
        let x: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let y: Vec<f64> = x.iter().map(|xv| 2.0 * xv + 1.0).collect();
        let resid = neutralize_vs(&y, &x);
        let mx = x.iter().sum::<f64>() / x.len() as f64;
        let mr = resid.iter().sum::<f64>() / resid.len() as f64;
        let cov = x
            .iter()
            .zip(resid.iter())
            .map(|(a, b)| (a - mx) * (b - mr))
            .sum::<f64>();
        assert!(cov.abs() < 1e-6);
    }

    #[test]
    fn test_corr_basic() {
        assert!((corr(&[1.0, 2.0, 3.0, 4.0], &[1.0, 2.0, 3.0, 4.0]) - 1.0).abs() < 1e-9);
    }
}
