//! hm46 奇偶数偏好 —— 旧框架（design_whatever_out.TickDataBase）的 Rust 复现。
//!
//! 旧文件：/home/chenzongwei/pythoncode/成果hm46_奇偶数偏好/hm46_okay.py
//!   run_factor_out(..., get_many=0, read_full=0, change_time=1, cross_section=False)
//!   → read_market_data_single() 走 `get_many==0 && read_full==0` 那条分支：
//!     只做 deal_data，返回**单个** DataFrame（index 是 pd.to_datetime(datetime) 列，
//!     不是 exchtime），不做任何时段切分。所以 Rust 侧读盘口快照、自己按钟点切窗口。
//!   get_factor 整体包在 try/except 里，出任何错都返回 None（上游补整条 NaN）。
//!
//! ════════════════════════════════════════════════════════════════════════
//! 一、get_factor() 的算法（逐行对齐旧代码）
//! ════════════════════════════════════════════════════════════════════════
//!   1. df = between_time("09:30:00", "14:57:00")，只保留 9 列：
//!        last_prc, ask_prc2, bid_prc2, ask_vol2, bid_vol2, ask_prc3, bid_prc3,
//!        ask_vol3, bid_vol3
//!      between_time 是**微秒精度的闭区间**，14:57:00.000000 算在内、之后的不算。
//!   2. 过滤（shift(1) 是在**过滤前**的整段上做的，第 0 行 shift 出来是 NaN，
//!      所以第 0 行必然被丢掉）：
//!        ask_prc2 == ask_prc2.shift(1)  且 bid_prc2、ask_prc3、bid_prc3 同理
//!        且四个价格 != 0、四个量 != 0
//!   3. **过滤之后**才做差分：ask_vol2 = ask_vol2 - ask_vol2.shift(1)（其余三个同理），
//!      过滤后第 0 行的差是 NaN。
//!   4. 行数 >= 5 才算，否则整条 None。然后：
//!        ask_vol23_rate = ask_vol2 / (|ask_vol2| + |ask_vol3|)
//!        bid_vol23_rate = bid_vol2 / (|bid_vol2| + |bid_vol3|)
//!        ab_vol3_rate   = ask_vol3 / (|ask_vol3| + |bid_vol3|)
//!        last_prc       = (last_prc - min) / (max - min)     ← 最小最大都在过滤后的整段上取
//!   5. 五个窗口各自 one()：
//!        all       整段
//!        open30    09:30:00 ~ 10:00:00
//!        morning   09:30:00 ~ 11:30:00
//!        afternoon 13:00:00 ~ 14:57:00
//!        close30   14:30:00 ~ 14:57:00
//!      窗口是在过滤后的帧上按 index 再切一次，所以边界两侧都含。
//!   6. one(df) 出 30 个值：三个 rate 各自的 mean/std/skew/max/min/corr(last_prc)，
//!      以及 last_prc 落在「最小的 20%/40%」和「最大的 20%/40%」两个子集里的
//!      三个 rate 的均值（6 个分位阈值都按窗口内的 last_prc 现算）。
//!      顺序见 METRICS，与旧代码的 res 列表一一对应。
//!
//!   names_single = 30 个指标 × 5 个窗口（all/open30/morning/afternoon/close30），共 150 个。
//!   names_save 见 hm46_okay.py；其中 7 个带 `_fold` 的（截面均值距离化）由因子库那一层
//!   生成，不在本文件的输出里。
//!
//! ════════════════════════════════════════════════════════════════════════
//! 二、pandas 语义对照
//! ════════════════════════════════════════════════════════════════════════
//!   mean/std/max/min/skew 都 skipna；std 用 ddof=1（样本 < 2 给 NaN）；
//!   skew 是 pandas 2.3 的无偏口径 g1*sqrt(n(n-1))/(n-2)（n < 3 给 NaN）；
//!   quantile 默认线性插值，(n-1)*q 位置；
//!   corr 是成对完整观测的 Pearson（样本 < 2 或某列方差为 0 给 NaN）。
//!   非有限值一律写成 NaN，与旧框架落盘时的 replace(inf, nan) 口径一致。

use std::fs;

use crate::data_paths;

// market_data.csv 的列下标（0-based，与文件头一致）：
//   symbol,datetime,date,time,exchtime,localtime,last_prc,status,prev_close,open,high,low,
//   high_limited,low_limited,volume,turnover,num_trades,weighted_ask_prc,weighted_bid_prc,
//   total_ask_vol,total_bid_vol,ask_prc1,ask_vol1,bid_prc1,bid_vol1,
//   ask_prc2,ask_vol2,bid_prc2,bid_vol2,ask_prc3,ask_vol3,bid_prc3,bid_vol3,...
const COL_EXCHTIME: usize = 4;
const COL_LAST_PRC: usize = 6;
const COL_ASK_PRC2: usize = 25;
const COL_ASK_VOL2: usize = 26;
const COL_BID_PRC2: usize = 27;
const COL_BID_VOL2: usize = 28;
const COL_ASK_PRC3: usize = 29;
const COL_ASK_VOL3: usize = 30;
const COL_BID_PRC3: usize = 31;
const COL_BID_VOL3: usize = 32;

const DAY_US: i64 = 86_400_000_000;
const OPEN_US: i64 = (9 * 3600 + 30 * 60) * 1_000_000; // 09:30:00
const AM_END_US: i64 = (11 * 3600 + 30 * 60) * 1_000_000; // 11:30:00
const PM_START_US: i64 = 13 * 3600 * 1_000_000; // 13:00:00
const LAST_US: i64 = (14 * 3600 + 57 * 60) * 1_000_000; // 14:57:00
const CLOSE_START_US: i64 = (14 * 3600 + 30 * 60) * 1_000_000; // 14:30:00
const OPEN30_END_US: i64 = 10 * 3600 * 1_000_000; // 10:00:00

const POINT1: f64 = 0.2;
const POINT2: f64 = 0.4;

/// 每个窗口 30 个指标
const PER_SESSION: usize = 30;
pub const N_OUT: usize = PER_SESSION * 5;

const METRICS: [&str; PER_SESSION] = [
    "ask_vol23_rate_mean",
    "bid_vol23_rate_mean",
    "ab_vol3_rate_mean",
    "ask_vol23_rate_std",
    "bid_vol23_rate_std",
    "ab_vol3_rate_std",
    "ask_vol23_rate_skew",
    "bid_vol23_rate_skew",
    "ab_vol3_rate_skew",
    "ask_vol23_rate_max",
    "bid_vol23_rate_max",
    "ab_vol3_rate_max",
    "ask_vol23_rate_min",
    "bid_vol23_rate_min",
    "ab_vol3_rate_min",
    "ask_vol23_rate_corr_last_prc",
    "bid_vol23_rate_corr_last_prc",
    "ab_vol3_rate_corr_last_prc",
    "last_prc_small1_ask_vol23_rate",
    "last_prc_small2_ask_vol23_rate",
    "last_prc_small1_bid_vol23_rate",
    "last_prc_small2_bid_vol23_rate",
    "last_prc_small1_ab_vol3_rate",
    "last_prc_small2_ab_vol3_rate",
    "last_prc_large1_ask_vol23_rate",
    "last_prc_large2_ask_vol23_rate",
    "last_prc_large1_bid_vol23_rate",
    "last_prc_large2_bid_vol23_rate",
    "last_prc_large1_ab_vol3_rate",
    "last_prc_large2_ab_vol3_rate",
];

/// 五个窗口（当天零点起的微秒，闭区间两端），顺序就是 names 的拼接顺序
const WINDOWS: [(&str, i64, i64); 5] = [
    ("all", 0, DAY_US),
    ("open30", OPEN_US, OPEN30_END_US),
    ("morning", OPEN_US, AM_END_US),
    ("afternoon", PM_START_US, LAST_US),
    ("close30", CLOSE_START_US, LAST_US),
];

pub fn names() -> Vec<String> {
    let mut out = Vec::with_capacity(N_OUT);
    for (w, _, _) in WINDOWS.iter() {
        for m in METRICS.iter() {
            out.push(format!("{}_{}", m, w));
        }
    }
    out
}

// ============================================================================
// pandas 统计量
// ============================================================================

fn mean_skip(x: &[f64]) -> f64 {
    let mut n = 0usize;
    let mut s = 0.0f64;
    for &v in x.iter() {
        if v.is_nan() {
            continue;
        }
        n += 1;
        s += v;
    }
    if n == 0 {
        return f64::NAN;
    }
    s / n as f64
}

/// Series.std(ddof=1, skipna=True)，两遍算法
fn std_skip(x: &[f64]) -> f64 {
    let n = x.iter().filter(|v| !v.is_nan()).count();
    if n < 2 {
        return f64::NAN;
    }
    let nf = n as f64;
    let m = x.iter().filter(|v| !v.is_nan()).sum::<f64>() / nf;
    let mut s = 0.0f64;
    for &v in x.iter() {
        if v.is_nan() {
            continue;
        }
        let d = v - m;
        s += d * d;
    }
    (s / (nf - 1.0)).sqrt()
}

/// pandas 2.3 的 Series.skew()（无偏口径），skipna
fn skew_skip(x: &[f64]) -> f64 {
    let vals: Vec<f64> = x.iter().copied().filter(|v| !v.is_nan()).collect();
    let n = vals.len();
    if n < 3 {
        return f64::NAN;
    }
    let nf = n as f64;
    let m = vals.iter().sum::<f64>() / nf;
    let mut m2 = 0.0f64;
    let mut m3 = 0.0f64;
    for &v in vals.iter() {
        let d = v - m;
        m2 += d * d;
        m3 += d * d * d;
    }
    m2 /= nf;
    m3 /= nf;
    if m2 <= 0.0 {
        return f64::NAN;
    }
    m3 / m2.powf(1.5) * (nf * (nf - 1.0)).sqrt() / (nf - 2.0)
}

fn max_skip(x: &[f64]) -> f64 {
    let mut best = f64::NAN;
    for &v in x.iter() {
        if v.is_nan() {
            continue;
        }
        if best.is_nan() || v > best {
            best = v;
        }
    }
    best
}

fn min_skip(x: &[f64]) -> f64 {
    let mut best = f64::NAN;
    for &v in x.iter() {
        if v.is_nan() {
            continue;
        }
        if best.is_nan() || v < best {
            best = v;
        }
    }
    best
}

/// Series.quantile(q) 默认线性插值
fn quantile_skip(x: &[f64], q: f64) -> f64 {
    let mut v: Vec<f64> = x.iter().copied().filter(|t| !t.is_nan()).collect();
    let n = v.len();
    if n == 0 {
        return f64::NAN;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let pos = (n as f64 - 1.0) * q;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    if lo == hi {
        return v[lo];
    }
    v[lo] + (v[hi] - v[lo]) * (pos - lo as f64)
}

/// DataFrame.corr() 的 Pearson（成对完整观测）
fn pearson(x: &[f64], y: &[f64]) -> f64 {
    let mut n = 0usize;
    let mut sx = 0.0f64;
    let mut sy = 0.0f64;
    for i in 0..x.len() {
        if x[i].is_nan() || y[i].is_nan() {
            continue;
        }
        n += 1;
        sx += x[i];
        sy += y[i];
    }
    if n < 2 {
        return f64::NAN;
    }
    let nf = n as f64;
    let mx = sx / nf;
    let my = sy / nf;
    let mut vx = 0.0f64;
    let mut vy = 0.0f64;
    let mut cxy = 0.0f64;
    for i in 0..x.len() {
        if x[i].is_nan() || y[i].is_nan() {
            continue;
        }
        let dx = x[i] - mx;
        let dy = y[i] - my;
        vx += dx * dx;
        vy += dy * dy;
        cxy += dx * dy;
    }
    if vx <= 0.0 || vy <= 0.0 {
        return f64::NAN;
    }
    cxy / (vx.sqrt() * vy.sqrt())
}

/// 子集上的 skipna 均值：sel[i] 为真才纳入
fn mean_where(x: &[f64], sel: &[bool]) -> f64 {
    let mut n = 0usize;
    let mut s = 0.0f64;
    for i in 0..x.len() {
        if !sel[i] || x[i].is_nan() {
            continue;
        }
        n += 1;
        s += x[i];
    }
    if n == 0 {
        return f64::NAN;
    }
    s / n as f64
}

#[inline]
fn finite_or_nan(v: f64) -> f64 {
    if v.is_finite() {
        v
    } else {
        f64::NAN
    }
}

/// 旧代码的 one(df)：窗口内 30 个指标
fn one(ask: &[f64], bid: &[f64], ab: &[f64], lp: &[f64]) -> [f64; PER_SESSION] {
    if ask.is_empty() {
        return [f64::NAN; PER_SESSION];
    }
    let q1 = quantile_skip(lp, POINT1);
    let q2 = quantile_skip(lp, POINT2);
    let q8 = quantile_skip(lp, 1.0 - POINT1);
    let q6 = quantile_skip(lp, 1.0 - POINT2);
    let n = lp.len();
    let mut s1 = vec![false; n];
    let mut s2 = vec![false; n];
    let mut l1 = vec![false; n];
    let mut l2 = vec![false; n];
    for i in 0..n {
        s1[i] = lp[i] < q1;
        s2[i] = lp[i] < q2;
        l1[i] = lp[i] > q8;
        l2[i] = lp[i] > q6;
    }
    [
        mean_skip(ask),
        mean_skip(bid),
        mean_skip(ab),
        std_skip(ask),
        std_skip(bid),
        std_skip(ab),
        skew_skip(ask),
        skew_skip(bid),
        skew_skip(ab),
        max_skip(ask),
        max_skip(bid),
        max_skip(ab),
        min_skip(ask),
        min_skip(bid),
        min_skip(ab),
        pearson(ask, lp),
        pearson(bid, lp),
        pearson(ab, lp),
        mean_where(ask, &s1),
        mean_where(ask, &s2),
        mean_where(bid, &s1),
        mean_where(bid, &s2),
        mean_where(ab, &s1),
        mean_where(ab, &s2),
        mean_where(ask, &l1),
        mean_where(ask, &l2),
        mean_where(bid, &l1),
        mean_where(bid, &l2),
        mean_where(ab, &l1),
        mean_where(ab, &l2),
    ]
}

/// 一行盘口快照，只留旧代码真正用到的那几个量（全部 f64）。
#[derive(Clone, Copy)]
struct RawRow {
    /// 当天北京时间的微秒（exchtime 微秒 + 8 小时，再对一天取模）
    day_us: i64,
    /// last_prc, ask_prc2, bid_prc2, ask_vol2, bid_vol2, ask_prc3, bid_prc3, ask_vol3, bid_vol3
    v: [f64; 9],
}

fn parse_f64(bytes: &[u8]) -> f64 {
    std::str::from_utf8(bytes)
        .ok()
        .and_then(|s| s.trim().parse::<f64>().ok())
        .unwrap_or(f64::NAN)
}

/// 拆一行成 61 个字段（不足的补空切片）
#[inline]
fn split_fields(line: &[u8]) -> [&[u8]; 61] {
    let mut fields: [&[u8]; 61] = [&[][..]; 61];
    let mut start = 0usize;
    let mut col = 0usize;
    for (i, &b) in line.iter().enumerate() {
        if b == b',' {
            if col < 61 {
                fields[col] = &line[start..i];
            }
            col += 1;
            start = i + 1;
        }
    }
    if col < 61 {
        fields[col] = &line[start..];
    }
    fields
}

#[inline]
fn parse_line(line: &[u8]) -> Option<RawRow> {
    if line.is_empty() {
        return None;
    }
    let f = split_fields(line);
    let exchtime = parse_f64(f[COL_EXCHTIME]);
    if !exchtime.is_finite() {
        return None;
    }
    let ex_us = exchtime.round() as i64;
    if ex_us == 0 {
        return None;
    }
    let vals = [
        parse_f64(f[COL_LAST_PRC]),
        parse_f64(f[COL_ASK_PRC2]),
        parse_f64(f[COL_BID_PRC2]),
        parse_f64(f[COL_ASK_VOL2]),
        parse_f64(f[COL_BID_VOL2]),
        parse_f64(f[COL_ASK_PRC3]),
        parse_f64(f[COL_BID_PRC3]),
        parse_f64(f[COL_ASK_VOL3]),
        parse_f64(f[COL_BID_VOL3]),
    ];
    Some(RawRow {
        day_us: (ex_us + 8 * 3600 * 1_000_000).rem_euclid(DAY_US),
        v: vals,
    })
}

/// 自己读盘口快照，价格与量都保持 f64。
/// 旧代码是 pandas.read_csv（f64）＋ pd.to_datetime(datetime) 建索引；这批数据里
/// exchtime 与 datetime 完全一致且没有 exchtime==0 的行，所以这里按 exchtime 取钟点。
/// 用库里的 read_market_fast_inner（f32）会差一点点：实测 300750@20231020 的
/// ab_vol3_rate_corr_last_prc_close30 相对误差 1.0e-4，刚好压线过不了对账。
fn read_market_f64(code: &str, date: i64) -> Option<Vec<RawRow>> {
    let filename = format!("{}_{}_market_data.csv", code, date);
    let mut path = None;
    for root in data_paths::level2_roots() {
        let p = root.join(date.to_string()).join("market_data").join(&filename);
        if p.exists() {
            path = Some(p);
            break;
        }
    }
    let data = fs::read(path?).ok()?;

    let mut out: Vec<RawRow> = Vec::with_capacity(data.len() / 200 + 1);
    let mut start = 0usize;
    for i in 0..data.len() {
        if data[i] == b'\n' {
            let line = &data[start..i];
            let line = if line.last() == Some(&b'\r') {
                &line[..line.len() - 1]
            } else {
                line
            };
            if let Some(r) = parse_line(line) {
                out.push(r);
            }
            start = i + 1;
        }
    }
    if start < data.len() {
        let line = &data[start..];
        let line = if line.last() == Some(&b'\r') {
            &line[..line.len() - 1]
        } else {
            line
        };
        if let Some(r) = parse_line(line) {
            out.push(r);
        }
    }
    Some(out)
}

/// 逐股入口：算不出来返回 None（上游会补 NaN，保证因子个数恒定）。
pub fn compute_stock(code: &str, date: i64) -> Option<Vec<f32>> {
    let recs = read_market_f64(code, date)?;
    if recs.is_empty() {
        return None;
    }

    // ── 第 1 步：取 [09:30:00, 14:57:00] 的行，按文件顺序（时间升序） ──
    let mut rows: Vec<([f64; 9], i64)> = Vec::with_capacity(recs.len());
    for r in recs.iter() {
        if r.day_us < OPEN_US || r.day_us > LAST_US {
            continue;
        }
        rows.push((r.v, r.day_us));
    }
    if rows.is_empty() {
        return None;
    }

    // ── 第 2 步：过滤（shift(1) 在过滤前的整段上做，第 0 行必然被丢） ──
    let mut kept: Vec<([f64; 9], i64)> = Vec::with_capacity(rows.len());
    let mut prev: Option<[f64; 9]> = None;
    for (cur, day) in rows.iter() {
        if let Some(p) = prev {
            let ok = cur[1] == p[1]
                && cur[2] == p[2]
                && cur[5] == p[5]
                && cur[6] == p[6]
                && cur[1] != 0.0
                && cur[2] != 0.0
                && cur[5] != 0.0
                && cur[6] != 0.0
                && cur[3] != 0.0
                && cur[4] != 0.0
                && cur[7] != 0.0
                && cur[8] != 0.0;
            if ok {
                kept.push((*cur, *day));
            }
        }
        prev = Some(*cur);
    }
    if kept.len() < 5 {
        return None;
    }

    // ── 第 3 步：过滤之后再差分 ──
    let m = kept.len();
    let mut ask_rate = vec![f64::NAN; m];
    let mut bid_rate = vec![f64::NAN; m];
    let mut ab_rate = vec![f64::NAN; m];
    for i in 1..m {
        let a2 = kept[i].0[3] - kept[i - 1].0[3];
        let b2 = kept[i].0[4] - kept[i - 1].0[4];
        let a3 = kept[i].0[7] - kept[i - 1].0[7];
        let b3 = kept[i].0[8] - kept[i - 1].0[8];
        ask_rate[i] = a2 / (a2.abs() + a3.abs());
        bid_rate[i] = b2 / (b2.abs() + b3.abs());
        ab_rate[i] = a3 / (a3.abs() + b3.abs());
    }

    // ── 第 4 步：last_prc 在过滤后的整段上归一化 ──
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for (r, _) in kept.iter() {
        if r[0] < lo {
            lo = r[0];
        }
        if r[0] > hi {
            hi = r[0];
        }
    }
    let span = hi - lo;
    let mut lp = vec![f64::NAN; m];
    for i in 0..m {
        lp[i] = (kept[i].0[0] - lo) / span;
    }

    // ── 第 5、6 步：五个窗口各 30 个指标 ──
    let mut out = Vec::with_capacity(N_OUT);
    for (_w, wlo, whi) in WINDOWS.iter() {
        let mut a: Vec<f64> = Vec::new();
        let mut b: Vec<f64> = Vec::new();
        let mut ab: Vec<f64> = Vec::new();
        let mut l: Vec<f64> = Vec::new();
        for i in 0..m {
            if kept[i].1 < *wlo || kept[i].1 > *whi {
                continue;
            }
            a.push(ask_rate[i]);
            b.push(bid_rate[i]);
            ab.push(ab_rate[i]);
            l.push(lp[i]);
        }
        let vals = one(&a, &b, &ab, &l);
        for v in vals.iter() {
            out.push(finite_or_nan(*v) as f32);
        }
    }
    debug_assert_eq!(out.len(), N_OUT);
    Some(out)
}

/// hm46 全是逐股因子，没有横截面版本。
pub fn compute_cross(_date: i64) -> Option<(Vec<String>, Vec<f32>)> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_layout() {
        let n = names();
        assert_eq!(n.len(), N_OUT);
        assert_eq!(n[0], "ask_vol23_rate_mean_all");
        assert_eq!(n[29], "last_prc_large2_ab_vol3_rate_all");
        assert_eq!(n[30], "ask_vol23_rate_mean_open30");
        assert_eq!(n[149], "last_prc_large2_ab_vol3_rate_close30");
    }

    #[test]
    fn quantile_linear() {
        let v = [0.0, 1.0, 2.0, 3.0, 4.0];
        assert!((quantile_skip(&v, 0.2) - 0.8).abs() < 1e-12);
        assert!((quantile_skip(&v, 0.5) - 2.0).abs() < 1e-12);
        let w = [1.0, 2.0, f64::NAN, 4.0];
        assert!((quantile_skip(&w, 0.5) - 2.0).abs() < 1e-12);
    }

    #[test]
    fn skew_is_unbiased() {
        let v = [0.1, 0.5, -0.3, 0.9, 0.2, -0.7];
        assert!((skew_skip(&v) - (-0.1306398578424021)).abs() < 1e-12);
    }
}
