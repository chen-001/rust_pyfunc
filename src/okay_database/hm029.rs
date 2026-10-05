//! hm29 订单编号趋势 —— 旧框架（design_whatever_out.TickDataBase）的 Rust 复现。
//!
//! 旧代码要点（逐条对齐）：
//!   data = read_trade_data_single(with_retreat=0)   → 丢掉 flag==32 的撤单行
//!   注意这里**不做** adjust_afternoon 的下午平移，窗口用的是真实钟点，
//!   所以 Rust 侧用 with_afternoon_adjust=false 读，自己按钟点筛。
//!   每个窗口内取四列 [num, exchtime, ask_order, bid_order]，num 是窗口内的行号，
//!   exchtime 是相对窗口内最早一笔的秒数。
//!   输出 10 个相关系数：num×ask、num×bid、ask×bid、以及对应的 rank 版，
//!   再加 seconds×ask、seconds×bid 与它们的 rank 版。
//!   五个时段顺序：allday、open30、morning、afternoon、close30，共 50 个值。
//!
//! 窗口边界用 between_time，两端都含，所以这里用闭区间。

use crate::fast_csv_reader::{read_trade_fast_inner, TradeRecord};

/// 五个时段（真实钟点的闭区间，单位秒）
const SESSIONS: [(&str, i64, i64); 5] = [
    ("allday", 9 * 3600 + 30 * 60, 14 * 3600 + 57 * 60),
    ("open30", 9 * 3600 + 30 * 60, 10 * 3600),
    ("morning", 9 * 3600 + 30 * 60, 11 * 3600 + 30 * 60),
    ("afternoon", 13 * 3600, 14 * 3600 + 57 * 60),
    ("close30", 14 * 3600 + 30 * 60, 14 * 3600 + 57 * 60),
];

/// 每个时段 10 个指标，五个时段共 50 个
const PER_SESSION: usize = 10;
pub const N_OUT: usize = PER_SESSION * 5;

const METRIC_NAMES: [&str; PER_SESSION] = [
    "corr_num_ask",
    "corr_num_bid",
    "corr_ask_bid",
    "corr_rank_num_ask",
    "corr_rank_num_bid",
    "corr_rank_ask_bid",
    "corr_seconds_ask",
    "corr_seconds_bid",
    "corr_rank_seconds_ask",
    "corr_rank_seconds_bid",
];

pub fn names() -> Vec<String> {
    let mut out = Vec::with_capacity(N_OUT);
    for (sess, _, _) in SESSIONS.iter() {
        for m in METRIC_NAMES.iter() {
            out.push(format!("{}_{}", m, sess));
        }
    }
    out
}

/// 北京时间的当日微秒数（time_us 已含 8 小时偏移）。
///
/// 必须用微秒而不是整秒：pandas 的 between_time 是精确到纳秒的闭区间，
/// 用整秒截断会把 14:57:00.3 这种本该被排除的行放进来。
#[inline]
fn day_us(time_us: i64) -> i64 {
    time_us.rem_euclid(86_400_000_000)
}

/// pandas 的 rank(method="average")：升序，并列取平均名次。
/// 输入里的 NaN 位置返回 NaN（对应 DataFrame.rank 的 na_option="keep"）。
fn rank_average(values: &[f64]) -> Vec<f64> {
    let n = values.len();
    let mut idx: Vec<usize> = (0..n).filter(|&i| values[i].is_finite()).collect();
    idx.sort_by(|&a, &b| values[a].partial_cmp(&values[b]).unwrap_or(std::cmp::Ordering::Equal));
    let mut out = vec![f64::NAN; n];
    let mut i = 0usize;
    while i < idx.len() {
        let mut j = i + 1;
        while j < idx.len() && values[idx[j]] == values[idx[i]] {
            j += 1;
        }
        // 名次从 1 开始，并列取 [i+1, j] 的平均
        let avg = ((i + 1) as f64 + j as f64) / 2.0;
        for k in i..j {
            out[idx[k]] = avg;
        }
        i = j;
    }
    out
}

/// pandas 的 DataFrame.corr()：Pearson，成对完整观测（任一为 NaN 就丢掉这一对）。
/// 任一侧标准差为 0 或有效样本 < 2 时返回 NaN。
fn pearson(x: &[f64], y: &[f64]) -> f64 {
    let mut n = 0usize;
    let mut sx = 0.0f64;
    let mut sy = 0.0f64;
    let mut sxx = 0.0f64;
    let mut syy = 0.0f64;
    let mut sxy = 0.0f64;
    for i in 0..x.len() {
        if !x[i].is_finite() || !y[i].is_finite() {
            continue;
        }
        n += 1;
        sx += x[i];
        sy += y[i];
        sxx += x[i] * x[i];
        syy += y[i] * y[i];
        sxy += x[i] * y[i];
    }
    if n < 2 {
        return f64::NAN;
    }
    let nf = n as f64;
    let cov = sxy - sx * sy / nf;
    let vx = sxx - sx * sx / nf;
    let vy = syy - sy * sy / nf;
    if vx <= 0.0 || vy <= 0.0 {
        return f64::NAN;
    }
    cov / (vx.sqrt() * vy.sqrt())
}

/// 窗口内算 10 个相关系数，顺序与 METRIC_NAMES 一致。
fn session_metrics(rows: &[&TradeRecord], lo: i64, hi: i64) -> [f64; PER_SESSION] {
    let mut num: Vec<f64> = Vec::new();
    let mut secs: Vec<f64> = Vec::new();
    let mut ask: Vec<f64> = Vec::new();
    let mut bid: Vec<f64> = Vec::new();
    for r in rows.iter() {
        if !(lo * 1_000_000..=hi * 1_000_000).contains(&day_us(r.time_us)) {
            continue;
        }
        num.push(num.len() as f64);
        secs.push(r.time_us as f64 / 1e6);
        // 空的下单编号在 Python 侧是 NaN，这里用 0 表示缺失（委托编号不会是 0）
        ask.push(if r.ask_order == 0 { f64::NAN } else { r.ask_order as f64 });
        bid.push(if r.bid_order == 0 { f64::NAN } else { r.bid_order as f64 });
    }
    if num.len() < 2 {
        return [f64::NAN; PER_SESSION];
    }
    // exchtime 转成「相对窗口内最早一笔的秒数」
    let t0 = secs.iter().cloned().fold(f64::INFINITY, f64::min);
    for v in secs.iter_mut() {
        *v -= t0;
    }
    let r_num = rank_average(&num);
    let r_sec = rank_average(&secs);
    let r_ask = rank_average(&ask);
    let r_bid = rank_average(&bid);
    [
        pearson(&num, &ask),
        pearson(&num, &bid),
        pearson(&ask, &bid),
        pearson(&r_num, &r_ask),
        pearson(&r_num, &r_bid),
        pearson(&r_ask, &r_bid),
        pearson(&secs, &ask),
        pearson(&secs, &bid),
        pearson(&r_sec, &r_ask),
        pearson(&r_sec, &r_bid),
    ]
}

/// 逐股入口：算不出来返回 None（上游会补 NaN，保证因子个数恒定）。
pub fn compute_stock(code: &str, date: i64) -> Option<Vec<f32>> {
    // with_afternoon_adjust=false：旧代码用真实钟点窗口，不做下午平移。
    //
    // 注意：这里不能靠 read_trade_fast_inner 的 with_retreat 来丢撤单。
    // 成交明细里 flag 列 2025 年前后写法不一样（早年是 "32"，2025 年是 "32.0"），
    // 库里那条过滤是按字节比 "32"，对 "32.0" 不生效。所以读到之后自己按数值过滤。
    let trades = read_trade_fast_inner(code, date, true, false, usize::MAX).ok()?;
    if trades.is_empty() {
        return None;
    }
    let rows: Vec<&TradeRecord> = trades.iter().filter(|r| r.flag != 32).collect();
    if rows.is_empty() {
        return None;
    }
    let mut out = Vec::with_capacity(N_OUT);
    for (_sess, lo, hi) in SESSIONS.iter() {
        out.extend(session_metrics(&rows, *lo, *hi).iter().map(|&v| v as f32));
    }
    debug_assert_eq!(out.len(), N_OUT);
    Some(out)
}

/// hm29 全是逐股因子，没有横截面版本。
pub fn compute_cross(_date: i64) -> Option<(Vec<String>, Vec<f32>)> {
    None
}
