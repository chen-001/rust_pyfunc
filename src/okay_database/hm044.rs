//! hm044 撤单专区 —— 旧框架（design_whatever_out.TickDataBase）的 Rust 复现。
//!
//! 旧文件：/home/chenzongwei/pythoncode/成果hm43_成果hm44_撤单专区/hm44_okay.py
//! 参数：get_many=0 / read_full=0 / change_time=1 / use_old=False
//!
//! ════════════════════════════════════════════════════════════════════
//! 一、这个系列要的是**撤单行**，所以不能过滤 flag==32
//! ════════════════════════════════════════════════════════════════════
//! 旧代码是 read_trade_data_single(with_retreat=1)，也就是把撤单行留着；
//! 后面再自己分：flag==32 是撤单，其中 ask_order==0 算「买方撤单」、bid_order==0 算「卖方撤单」；
//! flag!=32 是成交，其中 flag==66 算「主买成交」。
//! change_time=1 → exchtime 被转成时间戳；dura = exchtime，再做 (dura-df.dura.min()).dt.total_seconds()。
//!
//! ════════════════════════════════════════════════════════════════════
//! 二、先按分钟装桶，再在分钟序列上算指标
//! ════════════════════════════════════════════════════════════════════
//! 1) 逐笔先切 [09:30:00, 14:57:00]（闭区间）。
//! 2) resample("1min")：1 分钟一桶，label=left、closed=left（桶标签是桶起点，
//!    桶内是 [T, T+1min)）；每个桶调 one()：
//!      桶内行数 <= 10 → 该桶记 NaN（后面 dropna 掉）
//!      否则：dura = 桶内 exchtime 减去**桶内最早一笔**的秒数
//!            buy  = dtw(买方撤单的 dura, 主买成交的 dura)
//!            sell = dtw(卖方撤单的 dura, 主买成交的 dura)   ← 旧代码写的也是 ac_buy，照抄
//!      再把 ±inf 换成 NaN 并 dropna（撤单为空时 dtw 返回 inf，于是整桶被丢掉）
//! 3) 六个窗口切在这张「分钟桶」表上（桶标签闭区间）：
//!      allday 全部、open30 [09:30,10:00]、morning [09:30,11:30]、
//!      afternoon [13:00,14:57]、close30 [14:30,14:57]、silent [10:00,14:30]
//! 4) 每个窗口 sing()：桶数 <= 5 → 整段 32 个 NaN；否则算 32 个指标
//!      buy/sell 各自的 mean, std(ddof=1), skew(pandas 无偏), max, min,
//!      corr1..corr5（该列与自身滞后 1..5 桶的相关系数，成对完整观测）,
//!      dtw1..dtw5（该列去掉前 k 个 / 去掉后 k 个的 DTW 距离）
//!      外加 corr_buy_sell（buy 与 sell 的相关系数）与 dtw_buy_sell
//!    共 32 个，顺序 = [buy_mean, sell_mean, buy_std, sell_std, buy_skew, sell_skew,
//!      buy_max, sell_max, buy_min, sell_min, buy_corr1, sell_corr1, buy_corr2, sell_corr2,
//!      buy_corr3, sell_corr3, buy_corr4, sell_corr4, buy_corr5, sell_corr5,
//!      buy_dtw1, sell_dtw1, buy_dtw2, sell_dtw2, buy_dtw3, sell_dtw3,
//!      buy_dtw4, sell_dtw4, buy_dtw5, sell_dtw5, corr_buy_sell, dtw_buy_sell]
//! 6 个时段 × 32 = 192 个值，names 的顺序是 allday/open30/morning/afternoon/close30/silent。

use std::fs::File;
use std::io::Read;

const SEC_NS: i64 = 1_000_000_000;
const DAY_NS: i64 = 86_400 * SEC_NS;
const MIN_NS: i64 = 60 * SEC_NS;
const BJ_OFFSET_NS: i64 = 8 * 3600 * SEC_NS;

const fn hm(h: i64, m: i64) -> i64 {
    (h * 3600 + m * 60) * SEC_NS
}

const TRADE_LO: i64 = hm(9, 30);
const TRADE_HI: i64 = hm(14, 57);

/// 六个时段（分钟桶标签的闭区间），顺序与旧代码 res 的拼接顺序一致
const SESSIONS: [(i64, i64); 6] = [
    (0, DAY_NS),             // allday
    (hm(9, 30), hm(10, 0)),  // open30
    (hm(9, 30), hm(11, 30)), // morning
    (hm(13, 0), hm(14, 57)), // afternoon
    (hm(14, 30), hm(14, 57)), // close30
    (hm(10, 0), hm(14, 30)), // silent
];

const PER_SESSION: usize = 32;
pub const N_OUT: usize = PER_SESSION * 6; // 192

const BASE_NAMES: [&str; PER_SESSION] = [
    "buy_mean", "sell_mean", "buy_std", "sell_std", "buy_skew", "sell_skew", "buy_max", "sell_max",
    "buy_min", "sell_min", "buy_corr1", "sell_corr1", "buy_corr2", "sell_corr2", "buy_corr3",
    "sell_corr3", "buy_corr4", "sell_corr4", "buy_corr5", "sell_corr5", "buy_dtw1", "sell_dtw1",
    "buy_dtw2", "sell_dtw2", "buy_dtw3", "sell_dtw3", "buy_dtw4", "sell_dtw4", "buy_dtw5",
    "sell_dtw5", "corr_buy_sell", "dtw_buy_sell",
];
const SESSION_NAMES: [&str; 6] = ["allday", "open30", "morning", "afternoon", "close30", "silent"];

pub fn names() -> Vec<String> {
    let mut out = Vec::with_capacity(N_OUT);
    for s in SESSION_NAMES.iter() {
        for m in BASE_NAMES.iter() {
            out.push(format!("{}_{}", m, s));
        }
    }
    out
}

// ============================================================================
// 读逐笔（**不过滤** flag==32）
// ============================================================================

const TRD_COL_EXCHTIME: usize = 4;
const TRD_COL_FLAG: usize = 10;
const TRD_COL_ASK_ORDER: usize = 13;
const TRD_COL_BID_ORDER: usize = 14;
const TRD_NCOL: usize = 15;

#[inline]
fn parse_i64_at(b: &[u8]) -> i64 {
    let mut i = 0usize;
    while i < b.len() && b[i] == b' ' {
        i += 1;
    }
    let mut neg = false;
    if i < b.len() && (b[i] == b'-' || b[i] == b'+') {
        neg = b[i] == b'-';
        i += 1;
    }
    let mut v: i64 = 0;
    while i < b.len() {
        let c = b[i];
        if !(b'0'..=b'9').contains(&c) {
            break;
        }
        v = v * 10 + (c - b'0') as i64;
        i += 1;
    }
    if neg {
        -v
    } else {
        v
    }
}

#[derive(Clone, Copy)]
struct Row {
    clock: i64,
    flag: i64,
    ask_order: i64,
    bid_order: i64,
}

fn read_trades(code: &str, date: i64) -> Option<Vec<Row>> {
    let filename = format!("{}_{}_transaction.csv", code, date);
    let mut path = None;
    for root in crate::data_paths::level2_roots() {
        let p = root.join(date.to_string()).join("transaction").join(&filename);
        if p.exists() {
            path = Some(p);
            break;
        }
    }
    let path = path?;
    let mut content = String::new();
    File::open(&path).ok()?.read_to_string(&mut content).ok()?;
    let mut out: Vec<Row> = Vec::with_capacity(content.len() / 120 + 1);
    let mut lines = content.lines();
    lines.next();
    for line in lines {
        if line.is_empty() {
            continue;
        }
        let bytes = line.as_bytes();
        let mut f: [&[u8]; TRD_NCOL] = [&[][..]; TRD_NCOL];
        let mut start = 0usize;
        let mut col = 0usize;
        for (i, &b) in bytes.iter().enumerate() {
            if b == b',' {
                if col < TRD_NCOL {
                    f[col] = &bytes[start..i];
                }
                col += 1;
                start = i + 1;
            }
        }
        if col < TRD_NCOL {
            f[col] = &bytes[start..];
        }
        out.push(Row {
            clock: (parse_i64_at(f[TRD_COL_EXCHTIME]) * 1000 + BJ_OFFSET_NS).rem_euclid(DAY_NS),
            flag: parse_i64_at(f[TRD_COL_FLAG]),
            ask_order: parse_i64_at(f[TRD_COL_ASK_ORDER]),
            bid_order: parse_i64_at(f[TRD_COL_BID_ORDER]),
        });
    }
    Some(out)
}

// ============================================================================
// pandas 语义的小工具
// ============================================================================

/// pandas 的相关系数（逐行复刻 algos.pyx 的 nancorr：Welford 在线算法，成对完整观测）
fn pearson(x: &[f64], y: &[f64]) -> f64 {
    let mut nobs = 0.0f64;
    let (mut meanx, mut meany) = (0.0f64, 0.0f64);
    let (mut ssqdmx, mut ssqdmy, mut covxy) = (0.0f64, 0.0f64, 0.0f64);
    for i in 0..x.len() {
        if !x[i].is_finite() || !y[i].is_finite() {
            continue;
        }
        let vx = x[i];
        let vy = y[i];
        nobs += 1.0;
        let dx = vx - meanx;
        let dy = vy - meany;
        meanx += 1.0 / nobs * dx;
        meany += 1.0 / nobs * dy;
        ssqdmx += (vx - meanx) * dx;
        ssqdmy += (vy - meany) * dy;
        covxy += (vx - meanx) * dy;
    }
    if nobs < 1.0 {
        return f64::NAN;
    }
    let divisor = (ssqdmx * ssqdmy).sqrt();
    if divisor != 0.0 {
        covxy / divisor
    } else {
        f64::NAN
    }
}

/// pandas Series.mean()（NaN 跳过；全 NaN 给 NaN）
fn nanmean(x: &[f64]) -> f64 {
    let mut s = 0.0f64;
    let mut c = 0usize;
    for &v in x.iter() {
        if v.is_finite() {
            s += v;
            c += 1;
        }
    }
    if c == 0 {
        f64::NAN
    } else {
        s / c as f64
    }
}

/// pandas Series.std()（ddof=1；样本<2 给 NaN）
fn nanstd(x: &[f64]) -> f64 {
    let n = x.iter().filter(|v| v.is_finite()).count();
    if n < 2 {
        return f64::NAN;
    }
    let mean = nanmean(x);
    let mut s = 0.0f64;
    for &v in x.iter() {
        if v.is_finite() {
            s += (v - mean) * (v - mean);
        }
    }
    let m2 = s / n as f64;
    (m2 * n as f64 / (n as f64 - 1.0)).sqrt()
}

/// pandas Series.skew()（无偏：m3/m2^1.5 * sqrt(n(n-1))/(n-2)，样本<3 或方差为 0 给 NaN）
fn nanskew(x: &[f64]) -> f64 {
    let n = x.iter().filter(|v| v.is_finite()).count();
    if n < 3 {
        return f64::NAN;
    }
    let mean = nanmean(x);
    let mut m2 = 0.0f64;
    let mut m3 = 0.0f64;
    for &v in x.iter() {
        if v.is_finite() {
            let d = v - mean;
            m2 += d * d;
            m3 += d * d * d;
        }
    }
    m2 /= n as f64;
    m3 /= n as f64;
    if m2 == 0.0 {
        return f64::NAN;
    }
    let g = m3 / m2.powf(1.5);
    g * ((n * (n - 1)) as f64).sqrt() / (n as f64 - 2.0)
}

fn nanmax(x: &[f64]) -> f64 {
    let mut m = f64::NAN;
    for &v in x.iter() {
        if v.is_finite() && (m.is_nan() || v > m) {
            m = v;
        }
    }
    m
}

fn nanmin(x: &[f64]) -> f64 {
    let mut m = f64::NAN;
    for &v in x.iter() {
        if v.is_finite() && (m.is_nan() || v < m) {
            m = v;
        }
    }
    m
}

/// 与 crate 里 `rp.dtw_distance` 同一套算法的**纯 Rust**实现。
///
/// ！！不能直接调 `crate::time_series::dtw_distance`：那是 #[pyfunction]、返回 PyResult，
/// 而 worker 二进制不带 libpython，链路上只要碰到 PyErr/PyResult，链接就会缺
/// PyExc_ValueError 这类符号，整个 worker 构建挂在 link 这一步（rewrite-04 已实测）。
/// 算法逐行抄自 src/time_series/mod.rs 的 dtw_distance（radius=None、不限时那条路径）：
/// 一侧为空返回 inf，两侧都空返回 0。
fn dtw(s1: &[f64], s2: &[f64]) -> f64 {
    let n1 = s1.len();
    let n2 = s2.len();
    let width = n2 + 1;
    let mut mat = vec![f64::INFINITY; (n1 + 1) * width];
    mat[0] = 0.0;
    for i in 1..=n1 {
        for j in 1..=n2 {
            let cost = (s1[i - 1] - s2[j - 1]).abs();
            let best = mat[(i - 1) * width + j]
                .min(mat[i * width + j - 1])
                .min(mat[(i - 1) * width + j - 1]);
            mat[i * width + j] = cost + best;
        }
    }
    mat[n1 * width + n2]
}

// ============================================================================
// 一个时段的 32 个指标
// ============================================================================

fn session_metrics(buy: &[f64], sell: &[f64]) -> [f64; PER_SESSION] {
    let n = buy.len();
    if n <= 5 {
        return [f64::NAN; PER_SESSION];
    }
    let mut out = [f64::NAN; PER_SESSION];
    out[0] = nanmean(buy);
    out[1] = nanmean(sell);
    out[2] = nanstd(buy);
    out[3] = nanstd(sell);
    out[4] = nanskew(buy);
    out[5] = nanskew(sell);
    out[6] = nanmax(buy);
    out[7] = nanmax(sell);
    out[8] = nanmin(buy);
    out[9] = nanmin(sell);

    // corrwith(shift(k))：成对完整观测（滞后 k 的前 k 个点自然被丢掉）
    for k in 1..=5usize {
        {
            let a: Vec<f64> = buy[k..].iter().map(|&v| v).collect();
            let b: Vec<f64> = buy[..n - k].iter().map(|&v| v).collect();
            out[10 + 2 * (k - 1)] = pearson(&a, &b);
        }
        {
            let a: Vec<f64> = sell[k..].iter().map(|&v| v).collect();
            let b: Vec<f64> = sell[..n - k].iter().map(|&v| v).collect();
            out[11 + 2 * (k - 1)] = pearson(&a, &b);
        }
        {
            let a: Vec<f64> = buy[..n - k].iter().map(|&v| v).collect();
            let b: Vec<f64> = buy[k..].iter().map(|&v| v).collect();
            out[20 + 2 * (k - 1)] = dtw(&a, &b);
        }
        {
            let a: Vec<f64> = sell[..n - k].iter().map(|&v| v).collect();
            let b: Vec<f64> = sell[k..].iter().map(|&v| v).collect();
            out[21 + 2 * (k - 1)] = dtw(&a, &b);
        }
    }
    out[30] = pearson(buy, sell);
    out[31] = dtw(buy, sell);
    out
}

/// 逐股入口：算不出来返回 None（上游补 NaN，保证因子个数恒定）。
pub fn compute_stock(code: &str, date: i64) -> Option<Vec<f32>> {
    let rows_all = read_trades(code, date)?;
    // 1) 先切 [09:30:00, 14:57:00]
    let rows: Vec<&Row> = rows_all
        .iter()
        .filter(|r| r.clock >= TRADE_LO && r.clock <= TRADE_HI)
        .collect();
    if rows.is_empty() {
        return None;
    }
    // 2) 1 分钟装桶（label=left、closed=left）：桶按标签升序（pandas resample 会排序），
    //    桶内保持原行序
    let mut map: std::collections::HashMap<i64, Vec<&Row>> = std::collections::HashMap::new();
    let mut keys: Vec<i64> = Vec::new();
    for r in rows.iter() {
        let b = r.clock.div_euclid(MIN_NS) * MIN_NS;
        match map.get_mut(&b) {
            Some(v) => v.push(r),
            None => {
                map.insert(b, vec![r]);
                keys.push(b);
            }
        }
    }
    keys.sort_unstable();
    let mut bin_t: Vec<i64> = Vec::new();
    let mut buy: Vec<f64> = Vec::new();
    let mut sell: Vec<f64> = Vec::new();
    for b in keys.iter() {
        let v = &map[b];
        if v.len() <= 10 {
            continue; // 该桶记 NaN，随后被 dropna 掉
        }
        let t0 = v.iter().map(|r| r.clock).min().unwrap();
        let dura: Vec<f64> = v.iter().map(|r| (r.clock - t0) as f64 / 1e9).collect();
        let mut rt_buy: Vec<f64> = Vec::new();
        let mut rt_sell: Vec<f64> = Vec::new();
        let mut ac_buy: Vec<f64> = Vec::new();
        for (i, r) in v.iter().enumerate() {
            if r.flag == 32 {
                if r.ask_order == 0 {
                    rt_buy.push(dura[i]);
                }
                if r.bid_order == 0 {
                    rt_sell.push(dura[i]);
                }
            } else if r.flag == 66 {
                ac_buy.push(dura[i]);
            }
        }
        let bv = dtw(&rt_buy, &ac_buy);
        let sv = dtw(&rt_sell, &ac_buy);
        // replace(±inf -> NaN).dropna()：任一列非有限，该桶整行丢掉
        if !bv.is_finite() || !sv.is_finite() {
            continue;
        }
        bin_t.push(*b);
        buy.push(bv);
        sell.push(sv);
    }
    // 3) 六个窗口 + 4) 每段 32 个指标
    let mut out: Vec<f32> = Vec::with_capacity(N_OUT);
    for &(lo, hi) in SESSIONS.iter() {
        let mut b: Vec<f64> = Vec::new();
        let mut s: Vec<f64> = Vec::new();
        for i in 0..bin_t.len() {
            if bin_t[i] >= lo && bin_t[i] <= hi {
                b.push(buy[i]);
                s.push(sell[i]);
            }
        }
        out.extend(session_metrics(&b, &s).iter().map(|&v| v as f32));
    }
    debug_assert_eq!(out.len(), N_OUT);
    Some(out)
}

/// hm044 全是逐股因子，没有横截面版本。
pub fn compute_cross(_date: i64) -> Option<(Vec<String>, Vec<f32>)> {
    None
}
