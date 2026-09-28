//! A股创新高/新低「熵变」因子（etp_event）—— 60 维日频横截面因子。
//!
//! hm54（学术_行为因子/factor5.py）的现代重算版，与 hhi_event 是姊妹因子：
//! 事件定义相同（价格突破 running max / running min），但窗口与测度不同。
//!
//! - 基准分布 base：从开盘到当前时刻的累计 order -> volume。
//! - 窗口分布 win：base + 事件后 W 秒内（t_sec <= t_sec[i] + W）的成交。
//! - 测度：香农熵 H = -Σ p·ln(p)，熵变 = H(win) - H(base)。
//! - 无「非重叠筛选」；3 个窗口 {3, 30, 300}；输出 2 方向 × 2 侧 × 3 窗口 × 5 统计量 = 60。
//!
//! 与 rp.calculate_shannon_entropy_change / factor5.py 逐条一致（规格 /tmp/etp_spec.md）：
//! - 第一笔成交不算事件（extreme 初值取 price[0]，i==0 只更新 base/cum）。
//! - we 窗口右端指针跨事件持久单调、不重置。
//! - entropy 按键（order id）升序累加；base / win 用 BTreeMap 天然有序。
//! - 每个 (方向, 侧, W) 组合有独立的 base / cum / we。
//!
//! 三层结构：compute_etp_event_full（核心）+ pipeline_etp_event（批量）+ py_etp_event（调试）。
//! 参数全部 const 写死，不新增 Params（避免改 TaskMessage::Init）。

use crate::fast_csv_reader::{read_trade_fast_inner, TradeRecord};
use pyo3::prelude::*;
use std::collections::BTreeMap;

// ============================================================================
// 常量
// ============================================================================

/// 2 方向 × 2 侧 × 3 窗口 × 5 统计量。
pub const OUT_LEN: usize = 60;

/// 窗口（秒）。
const WINDOWS: [i64; 3] = [3, 30, 300];

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

/// 香农熵 H = -Σ p·ln(p)，按键（order id）升序累加（规格 §2）。
fn entropy(m: &BTreeMap<i64, f64>, total: f64) -> f64 {
    let mut e = 0.0f64;
    for &v in m.values() {
        let p = v / total;
        e -= p * p.ln();
    }
    e
}

/// 5 个统计量：[mean, skew, max, min, sum]（与 hhi_event 的 stats5 完全相同）。
///
/// 空序列 → 5 个 NaN；skew 需要 n >= 3，否则 NaN。
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

/// 一条逐笔熵变序列（长度 n，非事件点 NaN）。
///
/// t_sec[i] = time_us[i] / 1_000_000（i64 整数除法，向零截断）；
/// up=true 用 running max（创新高），false 用 running min（创新低）。
fn etp_series(recs: &[TradeRecord], t_sec: &[i64], side: usize, up: bool, w: i64) -> Vec<f64> {
    let n = recs.len();
    let mut result = vec![f64::NAN; n];
    if n == 0 {
        return result;
    }

    let mut base: BTreeMap<i64, f64> = BTreeMap::new();
    let mut cum = 0.0f64;
    let mut we = 0usize;
    // 第一笔成交只用来初始化极值，不算事件。
    let mut extreme = recs[0].price as f64;

    for i in 0..n {
        let key = key_of(&recs[i], side);
        *base.entry(key).or_insert(0.0) += recs[i].volume as f64;
        cum += recs[i].volume as f64;

        if i == 0 {
            continue;
        }

        let p = recs[i].price as f64;
        let is_event = if up { p > extreme } else { p < extreme };
        if !is_event {
            continue;
        }
        extreme = p;

        let base_ent = if cum > 0.0 { entropy(&base, cum) } else { 0.0 };
        let end_time = t_sec[i] + w;
        // we 跨事件持久：只前进、不重置。
        while we + 1 < n && t_sec[we + 1] <= end_time {
            we += 1;
        }
        if we > i {
            let mut win = base.clone();
            let mut win_total = cum;
            for j in (i + 1)..=we {
                let kj = key_of(&recs[j], side);
                *win.entry(kj).or_insert(0.0) += recs[j].volume as f64;
                win_total += recs[j].volume as f64;
            }
            if win_total > 0.0 {
                result[i] = entropy(&win, win_total) - base_ent;
            }
        }
    }

    result
}

#[inline]
fn key_of(r: &TradeRecord, side: usize) -> i64 {
    if side == SIDE_ASK {
        r.ask_order
    } else {
        r.bid_order
    }
}

// ============================================================================
// 核心计算 + 名字 + PyO3
// ============================================================================

/// 纯计算内核：records 必须已按 time_us 稳定升序（且已过筛）。
pub fn compute_etp_event_from_records(recs: &[TradeRecord]) -> Vec<f32> {
    let t_sec: Vec<i64> = recs.iter().map(|r| r.time_us / US_PER_SEC).collect();

    let mut out: Vec<f32> = Vec::with_capacity(OUT_LEN);
    for up in [true, false] {
        for side in [SIDE_ASK, SIDE_BID] {
            for &w in WINDOWS.iter() {
                let series = etp_series(recs, &t_sec, side, up, w);
                // 取事件点的值（非 NaN；事件点窗口为空时该点仍是 NaN，不参与统计）
                let vals: Vec<f64> = series.into_iter().filter(|v| !v.is_nan()).collect();
                for v in stats5(&vals).iter() {
                    out.push(*v as f32);
                }
            }
        }
    }

    debug_assert_eq!(out.len(), OUT_LEN);
    out
}

/// 核心唯一真相源：pipeline 和 Python 入口的共同调用点。
pub fn compute_etp_event_full(code: &str, date: i64) -> std::io::Result<Vec<f32>> {
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

    Ok(compute_etp_event_from_records(&recs))
}

/// 因子名（与 compute_etp_event_full 输出严格对齐，单一源）。
pub fn etp_event_names() -> Vec<String> {
    let mut names = Vec::with_capacity(OUT_LEN);
    for dir in ["up", "down"] {
        for side in ["ask", "bid"] {
            for &w in WINDOWS.iter() {
                for st in STATS.iter() {
                    names.push(format!("{}_{}_etp{}_{}", dir, side, w, st));
                }
            }
        }
    }
    assert_eq!(names.len(), OUT_LEN);
    names
}

/// 单股单日调试用，错误抛异常（可见完整栈）。
#[pyfunction]
pub fn py_etp_event(py: Python<'_>, code: &str, date: i64) -> PyResult<Vec<f32>> {
    compute_etp_event_full(code, date)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("{:?}", e)))
}

/// Python 拿因子名。
#[pyfunction]
pub fn py_etp_event_names() -> Vec<String> {
    etp_event_names()
}
