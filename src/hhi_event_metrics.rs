//! A股创新高/新低 HHI 事件因子（hhi_event）—— 140 维日频横截面因子。
//!
//! 把「局部 HHI」做法搬到 A 股 Level2 逐笔成交上：以订单号（ask_order / bid_order）作为分组对象，
//! 度量创新高（UP）/ 创新低（DOWN）事件前后各 W 秒窗口内成交量分布的集中度变化。
//!
//! 事件：running max / min 被突破的成交（当天第一笔有效成交不算事件）。
//! 窗口：W ∈ {3, 30, 60, 120, 150, 300, 900} 秒；每个「方向 × W」组内做非重叠筛选（t > last + 2W）。
//! 每个保留事件：diff = HHI(after) - HHI(before)，
//!   before = [τ-W, τ]（含事件成交），after = (τ, τ+W]（不含事件成交，空则丢弃）。
//! 输出 = 2 方向 × 2 侧 × 7 窗口 × 5 统计量（mean/skew/max/min/sum）= 140。
//!
//! 三层结构：compute_hhi_event_full（核心）+ pipeline_hhi_event（批量）+ py_hhi_event（调试）。
//! 规格：/tmp/hhi_spec.md（§3 的浮点运算顺序是硬要求，两边逐位一致）。
//!
//! 所有参数用 const 写死，不引入新 Params（避免改 TaskMessage::Init）。

use crate::fast_csv_reader::{read_trade_fast_inner, TradeRecord};
use pyo3::prelude::*;
use std::collections::HashMap;

// ============================================================================
// 常量
// ============================================================================

/// 2 方向 × 2 侧 × 7 窗口 × 5 统计量。
pub const OUT_LEN: usize = 140;

/// 窗口（秒）。
const WINDOWS: [i64; 7] = [3, 30, 60, 120, 150, 300, 900];

/// 统计量（顺序即输出下标顺序）。
const STATS: [&str; 5] = ["mean", "skew", "max", "min", "sum"];

const US_PER_SEC: i64 = 1_000_000;
const DAY_US: i64 = 86_400_000_000;
/// 09:30:00（当天零点起微秒）。
const SESSION_START_US: i64 = 34_200_000_000;
/// 14:57:00（当天零点起微秒）。
const SESSION_END_US: i64 = 53_820_000_000;

/// 侧：0 = ask_order，1 = bid_order。
const SIDE_ASK: usize = 0;
const SIDE_BID: usize = 1;

// ============================================================================
// 辅助
// ============================================================================

/// 第一个 t >= x 的下标（t 已升序）。
#[inline]
fn lower_bound(t: &[i64], x: i64) -> usize {
    t.partition_point(|&v| v < x)
}

/// 第一个 t > x 的下标（t 已升序）。
#[inline]
fn upper_bound(t: &[i64], x: i64) -> usize {
    t.partition_point(|&v| v <= x)
}

/// 单个区间内的 HHI（按 side 选分组键）。
///
/// 浮点顺序（规格 §3）：组内 q_g 按时间升序累加 volume；
/// Q 与 HHI 都按分组键（order id）升序累加。
fn hhi_of(recs: &[TradeRecord], side: usize) -> f64 {
    let mut g: HashMap<i64, f64> = HashMap::new();
    for r in recs {
        let k = if side == SIDE_ASK { r.ask_order } else { r.bid_order };
        *g.entry(k).or_insert(0.0) += r.volume as f64;
    }
    let mut kv: Vec<(i64, f64)> = g.into_iter().collect();
    kv.sort_unstable_by_key(|x| x.0);

    let mut q = 0.0f64;
    for &(_, v) in kv.iter() {
        q += v;
    }
    let mut acc = 0.0f64;
    for &(_, v) in kv.iter() {
        let p = v / q;
        acc += p * p;
    }
    acc
}

/// 5 个统计量：[mean, skew, max, min, sum]。
///
/// 空序列 → 5 个 NaN；skew 需要 n >= 3，否则 NaN。
/// 浮点顺序（规格 §3）：朴素顺序累加；skew 沿用 rust_pyfunc nanskew 的排列。
fn stats5(v: &[f64]) -> [f64; 5] {
    let n = v.len();
    if n == 0 {
        return [f64::NAN; 5];
    }
    let mut sum = 0.0f64;
    for &x in v {
        sum += x;
    }
    let mean = sum / n as f64;

    let skew = if n < 3 {
        f64::NAN
    } else {
        let mut m2 = 0.0f64;
        let mut m3 = 0.0f64;
        for &x in v {
            let d = x - mean;
            m2 += d * d;
            m3 += d * d * d;
        }
        m2 /= n as f64;
        m3 /= n as f64;
        if m2 <= 0.0 {
            f64::NAN
        } else {
            let nf = n as f64;
            let a = (nf * (nf - 1.0)).sqrt() / (nf - 2.0);
            a * m3 / m2.powf(1.5)
        }
    };

    let mut mx = v[0];
    let mut mn = v[0];
    for &x in v {
        if x > mx {
            mx = x;
        }
        if x < mn {
            mn = x;
        }
    }

    [mean, skew, mx, mn, sum]
}

// ============================================================================
// 核心计算 + 名字 + PyO3
// ============================================================================

/// 纯计算内核：records 必须已按 time_us 稳定升序（且已过筛）。
pub fn compute_hhi_event_from_records(recs: &[TradeRecord]) -> Vec<f32> {
    let n = recs.len();
    let t: Vec<i64> = recs.iter().map(|r| r.time_us).collect();

    // 事件：running max / min 被突破；i == 0 不算事件。
    let mut up_events: Vec<usize> = Vec::new();
    let mut down_events: Vec<usize> = Vec::new();
    if n > 0 {
        let mut mx = recs[0].price;
        let mut mn = recs[0].price;
        for (i, r) in recs.iter().enumerate().skip(1) {
            let p = r.price;
            if p > mx {
                mx = p;
                up_events.push(i);
            }
            if p < mn {
                mn = p;
                down_events.push(i);
            }
        }
    }

    let mut out: Vec<f32> = Vec::with_capacity(OUT_LEN);
    for events in [&up_events, &down_events] {
        for side in [SIDE_ASK, SIDE_BID] {
            for &w in WINDOWS.iter() {
                let w_us = w * US_PER_SEC;

                // 非重叠筛选：第一个必留，之后 t > last + 2W（严格大于）。
                let mut diffs: Vec<f64> = Vec::new();
                let mut last: Option<i64> = None;
                for &i in events.iter() {
                    let tau = t[i];
                    match last {
                        None => last = Some(tau),
                        Some(l) if tau > l + 2 * w_us => last = Some(tau),
                        Some(_) => continue,
                    }
                    let lo_b = lower_bound(&t, tau - w_us);
                    let hi_b = upper_bound(&t, tau);
                    let lo_a = upper_bound(&t, tau);
                    let hi_a = upper_bound(&t, tau + w_us);
                    // after 区间内没有任何有效成交 → 丢弃该事件
                    if lo_a >= hi_a {
                        continue;
                    }
                    let d = hhi_of(&recs[lo_a..hi_a], side) - hhi_of(&recs[lo_b..hi_b], side);
                    diffs.push(d);
                }

                for v in stats5(&diffs).iter() {
                    out.push(*v as f32);
                }
            }
        }
    }

    debug_assert_eq!(out.len(), OUT_LEN);
    out
}

/// 核心唯一真相源：pipeline 和 Python 入口的共同调用点。
pub fn compute_hhi_event_full(code: &str, date: i64) -> std::io::Result<Vec<f32>> {
    // 第 3 个参数 false = 丢掉 flag==32 撤单；第 4 个 false = 不做下午平移。
    let raw = read_trade_fast_inner(code, date, false, false, usize::MAX)?;
    let mut recs: Vec<TradeRecord> = raw
        .into_iter()
        .filter(|r| {
            if !(r.price > 0.0 && r.volume > 0.0) {
                return false;
            }
            let day_us = r.time_us.rem_euclid(DAY_US);
            day_us >= SESSION_START_US && day_us <= SESSION_END_US
        })
        .collect();
    // 稳定升序：保持同时间戳的原文件顺序。
    recs.sort_by_key(|r| r.time_us);

    Ok(compute_hhi_event_from_records(&recs))
}

/// 因子名（与 compute_hhi_event_full 输出严格对齐，单一源）。
pub fn hhi_event_names() -> Vec<String> {
    let mut names = Vec::with_capacity(OUT_LEN);
    for dir in ["up", "down"] {
        for side in ["ask", "bid"] {
            for &w in WINDOWS.iter() {
                for st in STATS.iter() {
                    names.push(format!("{}_{}_hhi{}_{}", dir, side, w, st));
                }
            }
        }
    }
    assert_eq!(names.len(), OUT_LEN);
    names
}

/// 单股单日调试用，错误抛异常（可见完整栈）。
#[pyfunction]
pub fn py_hhi_event(py: Python<'_>, code: &str, date: i64) -> PyResult<Vec<f32>> {
    compute_hhi_event_full(code, date)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("{:?}", e)))
}

/// Python 拿因子名。
#[pyfunction]
pub fn py_hhi_event_names() -> Vec<String> {
    hhi_event_names()
}
