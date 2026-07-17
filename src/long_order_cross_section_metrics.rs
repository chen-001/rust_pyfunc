//! 樫截面因子：漫长订单（hm27 思路的横截面化）
//!
//! # 核心思想
//! 每个 LimitOrder 被 0~N 笔成交吃掉，其「存续时长」= 末笔成交 − 首笔成交（秒）。
//! 漫长订单（分批、跨较长时间成交）往往是隐藏大单 / 冰山单 / 机构分仓的代理。
//!
//! # 与原 hm27 的区别
//! hm27 是 per-stock 自适应阈值（每只股票用自身 mean+k·σ）。
//! 本模块用**全市场统一标准**（绝对时长档 60s / 180s），并对因子做**流动性中性化**
//! （对 ln(总成交额) 回归取残差），剥离「冷门股订单天然慢」的流动性共线。
//!
//! # 因子布局（N_FACTORS = 16）
//! | idx | name                 | 中性化 | 含义 |
//! |-----|----------------------|--------|------|
//! | 0,1 | lsp_60, lbp_60       | 是     | 漫长卖/买单成交额占比 (dur>60s) |
//! | 2   | lsi_60               | 否     | 漫长买卖净不平衡 (天然去共线) |
//! | 3,4 | lsp_180, lbp_180     | 是     | 同上，180s 档 |
//! | 5   | lsi_180              | 否     | 180s 档净不平衡 |
//! | 6,7 | dac_ask, dac_bid     | 是     | 大单耐心度 corr(dur, amt) |
//! | 8   | big_net              | 否     | 大单(自身p80)方向不平衡 |
//! | 9   | cross_big_long_60    | 是     | 漫长∩大单 交集占比 |
//! | 10  | mean_dur_pos_ask     | 是     | 非单点订单平均存续(卖) |
//! | 11  | long_share_ask       | 是     | 漫长份额 = dur>60 amt / dur>0 amt |
//! | 12  | cnt_amt_div_ask      | 是     | 笔数占比 − 金额占比 背离 |
//! | 13  | open_lsi_60          | 否     | 开盘30min 净不平衡 |
//! | 14  | close_lsi_60         | 否     | 收盘30min 净不平衡 |
//! | 15  | long_persist_z       | 是     | 耐心溢价 (dur_p95 中性化) |
//!
//! # 性能
//! - per-stock：用 HashMap 聚合订单（order id 不可枚举，必须哈希，仅在股内用）
//! - 横截面：只处理每股 O(1) 的 16 个聚合量，顺序访问，零随机访问
//! - 见 AGENTS.md「高性能规范」：HashMap 只在 per-stock 内（非跨股热路径），符合原则

use crate::fast_csv_reader::{read_trade_fast_inner, TradeRecord};
use pyo3::prelude::*;
use rayon::prelude::*;
use std::collections::BTreeSet;
use std::fs;

/// 每只股票输出因子数。
pub const N_FACTORS: usize = 16;

/// 漫长订单阈值档（秒）。两档给出不同「耐心强度」。
const THR_SECS: [f64; 2] = [60.0, 180.0];

/// 每个因子是否需要 ln(总成交额) 流动性中性化。
/// 净不平衡类（lsi / big_net / 时段lsi）天然去共线，不做回归。
const NEED_NEUTRALIZE: [bool; N_FACTORS] = [
    true, true, false,  // 0,1,2 : lsp60, lbp60, lsi60
    true, true, false,  // 3,4,5 : lsp180, lbp180, lsi180
    true, true,         // 6,7   : dac_ask, dac_bid
    false,              // 8     : big_net
    true,               // 9     : cross_big_long_60
    true, true,         // 10,11 : mean_dur_pos_ask, long_share_ask
    true,               // 12    : cnt_amt_div_ask
    false, false,       // 13,14 : open_lsi_60, close_lsi_60
    true,               // 15    : long_persist_z
];

/// 开盘 / 收盘窗口长度（微秒）= 30 分钟
const WINDOW_US: i64 = 30 * 60 * 1_000_000;

// ============================================================
// per-stock 聚合
// ============================================================

/// 单个订单的聚合状态。
#[derive(Clone, Copy, Default)]
struct Agg {
    first_us: i64,
    last_us: i64,
    amt: f64,
    cnt: i64,
    init: bool,
}

/// 列出某天某子目录下所有股票代码（横截面枚举用）。
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

/// 按 order id 聚合（可选时间窗口过滤），返回 HashMap。
/// `get_id` 取 ask_order 或 bid_order；`in_window` 为 true 时只统计窗口内成交。
fn aggregate(
    trades: &[TradeRecord],
    get_id: impl Fn(&TradeRecord) -> i64,
    window: Option<(i64, i64)>,
) -> std::collections::HashMap<i64, Agg> {
    let mut m: std::collections::HashMap<i64, Agg> = std::collections::HashMap::with_capacity(trades.len() / 2 + 1);
    for t in trades {
        if t.flag != 66 && t.flag != 83 {
            continue;
        }
        if let Some((lo, hi)) = window {
            // lo=0 表示「< hi」的上界窗口（开盘）；hi=i64::MAX 表示「> lo」的下界窗口（收盘）
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

/// 从聚合 map 提取 (duration_秒, amount) 配对数组（同序，供 corr/筛选用）。
fn extract(m: &std::collections::HashMap<i64, Agg>) -> (Vec<f64>, Vec<f64>) {
    let n = m.len();
    let mut dur = Vec::with_capacity(n);
    let mut amt = Vec::with_capacity(n);
    for e in m.values() {
        dur.push((e.last_us - e.first_us) as f64 / 1e6);
        amt.push(e.amt);
    }
    (dur, amt)
}

/// 经验分位数（排序法，per-stock 订单数万级，开销可接受）。
fn percentile(v: &[f64], q: f64) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    let mut s: Vec<f64> = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = ((s.len() - 1) as f64 * q).round() as usize;
    s[idx.min(s.len() - 1)]
}

/// 皮尔逊相关系数（数组须同序）。
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

/// per-stock 算 16 个原始因子 + ln(总成交额)。
/// 返回 None 表示数据不足。
fn per_stock(trades: &[TradeRecord]) -> Option<([f64; N_FACTORS], f64)> {
    if trades.len() < 50 {
        return None;
    }
    // 全天总量 + 首末时间
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

    // 全天 ask / bid 聚合
    let ask_map = aggregate(trades, |t| t.ask_order, None);
    let bid_map = aggregate(trades, |t| t.bid_order, None);
    let (adur, aamt) = extract(&ask_map);
    let (bdur, bamt) = extract(&bid_map);
    if aamt.is_empty() || bamt.is_empty() {
        return None;
    }

    let mut f = [0.0f64; N_FACTORS];

    // 大单阈值（自身 p80）
    let a_p80 = percentile(&aamt, 0.80);
    let b_p80 = percentile(&bamt, 0.80);

    // [0..6] 各时长档的 lsp / lbp / lsi
    for (i, &thr) in THR_SECS.iter().enumerate() {
        let lsp: f64 = adur
            .iter()
            .zip(aamt.iter())
            .filter(|(d, _)| **d > thr)
            .map(|(_, a)| *a)
            .sum::<f64>()
            / total_amt;
        let lbp: f64 = bdur
            .iter()
            .zip(bamt.iter())
            .filter(|(d, _)| **d > thr)
            .map(|(_, a)| *a)
            .sum::<f64>()
            / total_amt;
        f[i * 3] = lsp;
        f[i * 3 + 1] = lbp;
        f[i * 3 + 2] = lbp - lsp;
    }

    // [6,7] 大单耐心度 corr(dur, amt)
    f[6] = corr(&adur, &aamt);
    f[7] = corr(&bdur, &bamt);

    // [8] 大单方向不平衡（自身 p80）
    let big_sell = aamt.iter().filter(|&&a| a > a_p80).sum::<f64>() / total_amt;
    let big_buy = bamt.iter().filter(|&&a| a > b_p80).sum::<f64>() / total_amt;
    f[8] = big_buy - big_sell;

    // [9] 漫长(>60s) ∩ 大单(>p80) 交集占比（卖单）
    f[9] = adur
        .iter()
        .zip(aamt.iter())
        .filter(|(d, a)| **d > 60.0 && **a > a_p80)
        .map(|(_, a)| *a)
        .sum::<f64>()
        / total_amt;

    // [10] 非单点订单平均存续（卖单）
    let pos: Vec<f64> = adur.iter().filter(|&&d| d > 0.0).copied().collect();
    f[10] = if pos.is_empty() {
        0.0
    } else {
        pos.iter().sum::<f64>() / pos.len() as f64
    };

    // [11] 漫长份额 = dur>60 amt / dur>0 amt（卖单）
    let l60 = adur
        .iter()
        .zip(aamt.iter())
        .filter(|(d, _)| **d > 60.0)
        .map(|(_, a)| *a)
        .sum::<f64>();
    let lpos = aamt
        .iter()
        .zip(adur.iter())
        .filter(|(_, d)| **d > 0.0)
        .map(|(a, _)| *a)
        .sum::<f64>();
    f[11] = if lpos > 0.0 { l60 / lpos } else { 0.0 };

    // [12] 笔数占比 − 金额占比 背离（卖单, 60s）
    let n_ord = adur.len().max(1) as f64;
    let cnt60 = adur.iter().filter(|&&d| d > 60.0).count() as f64 / n_ord;
    f[12] = cnt60 - l60 / total_amt;

    // [13,14] 开盘/收盘 30min 净不平衡（60s 档）
    let open_hi = first_us + WINDOW_US; // time_us < open_hi
    let close_lo = last_us - WINDOW_US; // time_us > close_lo
    let oa = aggregate(trades, |t| t.ask_order, Some((0, open_hi)));
    let ob = aggregate(trades, |t| t.bid_order, Some((0, open_hi)));
    let ca = aggregate(trades, |t| t.ask_order, Some((close_lo, i64::MAX)));
    let cb = aggregate(trades, |t| t.bid_order, Some((close_lo, i64::MAX)));
    f[13] = window_lsi(&oa, &ob, total_amt, 60.0);
    f[14] = window_lsi(&ca, &cb, total_amt, 60.0);

    // [15] 耐心溢价原始量 = dur_p95（卖单），横截面再中性化+z
    f[15] = percentile(&adur, 0.95);

    // 防御：确保所有值有限
    for v in f.iter_mut() {
        if !v.is_finite() {
            *v = 0.0;
        }
    }
    Some((f, total_amt.ln()))
}

/// 窗口内漫长买卖净不平衡（避免再写一遍筛选逻辑）。
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

// ============================================================
// 横截面运算
// ============================================================

/// 对 ln(总成交额) 线性回归取残差（流动性中性化）。
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

/// 横截面 z-score（仅用有限值统计，非有限值置 0）。
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

/// 核心唯一真相源：读全市场 → per-stock 因子 → 流动性中性化 + z-score → (codes, vals)。
pub fn compute_long_order_full(date: i64) -> std::io::Result<(Vec<String>, Vec<f32>)> {
    let codes = list_codes(date, "transaction");

    // ①+② rayon 并行读全市场逐笔 + per-stock 算因子
    let results: Vec<Option<([f64; N_FACTORS], f64)>> = codes
        .par_iter()
        .map(|code| {
            let trades = read_trade_fast_inner(code, date, false, true, usize::MAX).ok()?;
            per_stock(&trades)
        })
        .collect();

    // 组装有效股票
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

    // ③ 横截面：每列可选中性化 + z-score
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

    // ④ fan-out
    let mut out_vals = Vec::with_capacity(n * N_FACTORS);
    for f in &feats {
        out_vals.extend(f.iter().map(|&v| v as f32));
    }
    Ok((valid_codes, out_vals))
}

/// 因子名（与 N_FACTORS 严格对齐）。
pub fn long_order_names() -> Vec<String> {
    vec![
        "lsp_60".to_string(),
        "lbp_60".to_string(),
        "lsi_60".to_string(),
        "lsp_180".to_string(),
        "lbp_180".to_string(),
        "lsi_180".to_string(),
        "dac_ask".to_string(),
        "dac_bid".to_string(),
        "big_net".to_string(),
        "cross_big_long_60".to_string(),
        "mean_dur_pos_ask".to_string(),
        "long_share_ask".to_string(),
        "cnt_amt_div_ask".to_string(),
        "open_lsi_60".to_string(),
        "close_lsi_60".to_string(),
        "long_persist_z".to_string(),
    ]
}

// ============================================================
// Python 单日调试入口
// ============================================================

#[pyfunction]
pub fn py_long_order(py: Python<'_>, date: i64) -> PyResult<(Vec<String>, Vec<f32>)> {
    compute_long_order_full(date)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("{e:?}")))
}

#[pyfunction]
pub fn py_long_order_names() -> Vec<String> {
    long_order_names()
}

/// 调试用：返回原始（未中性化）因子值 + ln(总成交额)，供 sandbox 验证计算正确性。
/// 返回 (codes, raw_vals[n*16], ln_tots[n])。
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
    }

    #[test]
    fn test_neutralize_removes_linear_trend() {
        // y = 2x + 1 + noise，残差应与 x 近似不相关
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
        assert!((corr(&[1.0, 2.0, 3.0, 4.0], &[4.0, 3.0, 2.0, 1.0]) + 1.0).abs() < 1e-9);
    }
}
