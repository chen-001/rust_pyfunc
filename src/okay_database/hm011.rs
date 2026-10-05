//! hm11「百鸟朝凤」（旧框架 design_whatever_out.TickDataBase）的 Rust 复现。
//!
//! 旧文件：/home/chenzongwei/pythoncode/成果hm11_百鸟朝凤/hm11_okay.py
//!   框架参数 get_many=0 / read_full=0 / change_time=1，也就是「一整天的逐笔 + 一整天的盘口」
//!   各自一张表，窗口由旧代码自己切。
//!
//! ════════════════════════════════════════════════════════════════════
//! 旧代码逐步对照（写代码前先抄在这里，方便复核）
//! ════════════════════════════════════════════════════════════════════
//! 1) df  = read_trade_data_single(with_retreat=0)  → 丢掉 flag==32 的撤单行
//!    df2 = read_market_data_single()
//! 2) 盘口先切 [09:30:00, 14:57:00]：
//!      ask5 = ask_vol1..5 逐行求和，bid5 同理
//!      q80 取窗口内该序列的 0.8 分位数（线性插值）
//!      只留 ask5 - q80 > 0 的快照（bid 同理），按原行号打 label（0 起）
//!    逐笔同样切 [09:30:00, 14:57:00]，按 exchtime 排序（原文件本来就单调，
//!    pandas 的 sort_index 对单调索引直接返回 None 不改顺序，所以等价于保持原顺序）
//!    按 ask_order / bid_order 分组求 turnover 之和，只留 > 0.9 分位数的那些委托号
//!    —— 这两个集合是「当日窗口内成交额排前 10% 的委托号」。
//! 3) 两次 merge_asof（forward，容差 60 秒记 param60，30 秒记 param30）：
//!    每笔成交找「时间不早于它、且离它不超过容差」的第一条快照，
//!    wait = 快照时间 - 成交时间（秒），label 就是那条快照的序号；
//!    找不到就丢（旧代码 dropna）。asks 用 ask5 过滤后的快照，bids 用 bid5 过滤后的快照。
//! 4) 五个时段（all / open30 / morning / afternoon / close30）分别算 12 个量再相减：
//!    先在成交里按 label 分组，组内取 turnover 最大的那一笔（稳定排序后取 first），
//!    再只留 ask_order 或 bid_order 落在第 3 步那两个集合里的行，
//!    然后算 turnover 的 std(ddof=1) / median / max 除以分母，以及
//!    max / wait 中位数。分母：all 用「整段 ask 侧成交额之和」，
//!    buy 用本窗口主动买之和，sell 用本窗口主动卖之和。
//!    ！！注意两处旧代码的怪癖，必须照抄：
//!      (a) all 的分母用的是 haha(df, df2_ask) 整段（不切窗口）的成交额之和，
//!          也就是说 open30/close30 这些窗口里 turnover_all_* 也除以「全天」总额；
//!      (b) 每个时段 ask 侧的 12 个变量在 guard 没过时**保留上一时段的值**
//!          （Python 函数作用域泄漏）。bid 侧没有这个问题，guard 没过就直接写 12 个 nan。
//!          只有「bid 侧 guard 过了而 ask 侧从来没有过值」时，旧代码才会 NameError，
//!          被最外层 except 吞掉、整条返回 None，这里用 None 复现。
//! 5) 输出 = param60 的 (12 × 5) 拼上 param30 的 (12 × 5)，共 120 个值。
//!
//! ════════════════════════════════════════════════════════════════════
//! 精度：逐笔与盘口都用 f64 解析
//! ════════════════════════════════════════════════════════════════════
//! · 撤单过滤仍按规范自己 filter flag != 32（不靠 read_trade_fast_inner 的 with_retreat：
//!   库里那条判断按字节比 "32"，2025 年的明细写的是 "32.0"，过滤不生效）。
//! · 盘口五档量是整数，但要和 0.8 分位数做严格大于比较，f32 会把边界上的快照选错；
//!   逐笔 turnover 同时用于「组内排序取最大」和 0.9 分位数比较，也一样不能降精度。
//!   所以两者都在本文件里按 f64 解析 CSV（列索引与 fast_csv_reader 的常量一致）。

use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::Read;

/// 逐笔 CSV 的列索引（0 基；2018 年及以前末尾多一列 num，不影响前 15 列）
const TRD_COL_EXCHTIME: usize = 4;
const TRD_COL_TURNOVER: usize = 9;
const TRD_COL_FLAG: usize = 10;
const TRD_COL_ASK_ORDER: usize = 13;
const TRD_COL_BID_ORDER: usize = 14;
const TRD_NCOL: usize = 15;

/// 盘口 CSV 的列索引（0 基）
const MKT_COL_EXCHTIME: usize = 4;
const MKT_ASK_VOL1: usize = 22; // 每档 4 列：ask_prc, ask_vol, bid_prc, bid_vol
const MKT_BID_VOL1: usize = 24;
const MKT_NCOL: usize = 61;

const DAY_US: i64 = 86_400_000_000;
const SEC_US: i64 = 1_000_000;
const LD_0930: i64 = (9 * 3600 + 30 * 60) * SEC_US;
const LD_1000: i64 = 10 * 3600 * SEC_US;
const LD_1130: i64 = (11 * 3600 + 30 * 60) * SEC_US;
const LD_1300: i64 = 13 * 3600 * SEC_US;
const LD_1430: i64 = (14 * 3600 + 30 * 60) * SEC_US;
const LD_1457: i64 = (14 * 3600 + 57 * 60) * SEC_US;

/// 五个时段（真实钟点的微秒闭区间，旧代码用 between_time，两端都含）
const SESSION_NAMES: [&str; 5] = ["all", "open30", "morning", "afternoon", "close30"];
const SESSION_SPANS: [(i64, i64); 5] = [
    (LD_0930, LD_1457),
    (LD_0930, LD_1000),
    (LD_0930, LD_1130),
    (LD_1300, LD_1457),
    (LD_1430, LD_1457),
];

/// 每个时段的 12 个量（顺序与旧代码 res_* 列表一致）
const BASE_NAMES: [&str; 12] = [
    "turnover_all_std",
    "turnover_all_median",
    "turnover_all_max",
    "turnover_buy_std",
    "turnover_buy_median",
    "turnover_buy_max",
    "turnover_sell_std",
    "turnover_sell_median",
    "turnover_sell_max",
    "turnover_all_max_adj",
    "turnover_buy_max_adj",
    "turnover_sell_max_adj",
];

const PARAMS: [&str; 2] = ["param60", "param30"];
/// merge_asof 的 forward 容差（微秒），与 PARAMS 一一对应
const TOLS_US: [i64; 2] = [60 * SEC_US, 30 * SEC_US];

const PER_SESSION: usize = 12;
pub const N_OUT: usize = PER_SESSION * 5 * 2;

pub fn names() -> Vec<String> {
    let mut out = Vec::with_capacity(N_OUT);
    for p in PARAMS.iter() {
        for s in SESSION_NAMES.iter() {
            for b in BASE_NAMES.iter() {
                out.push(format!("{}_{}_{}", b, s, p));
            }
        }
    }
    out
}

#[inline]
fn day_us(t: i64) -> i64 {
    t.rem_euclid(DAY_US)
}

/// 钟点是否落在闭区间 [lo, hi] 内（微秒精度，不能按整秒截断）
#[inline]
fn in_span(t: i64, lo: i64, hi: i64) -> bool {
    let d = day_us(t);
    d >= lo && d <= hi
}

// ============================================================================
// 解析辅助
// ============================================================================

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

/// 空单元格按 pandas 的读法给 NaN
#[inline]
fn parse_f64_at(b: &[u8]) -> f64 {
    if b.is_empty() {
        return f64::NAN;
    }
    std::str::from_utf8(b)
        .ok()
        .and_then(|s| s.trim().parse::<f64>().ok())
        .unwrap_or(f64::NAN)
}

/// 委托编号列：空串在 pandas 里是 NaN（分组时被丢掉、isin 恒为 False），用 i64::MIN 表示
#[inline]
fn parse_order_at(b: &[u8]) -> i64 {
    if b.is_empty() {
        return i64::MIN;
    }
    parse_i64_at(b)
}

fn path_of(code: &str, date: i64, kind: &str) -> Option<std::path::PathBuf> {
    let filename = format!("{}_{}_{}.csv", code, date, kind);
    for root in crate::data_paths::level2_roots() {
        let p = root.join(date.to_string()).join(kind).join(&filename);
        if p.exists() {
            return Some(p);
        }
    }
    None
}

// ============================================================================
// 逐笔 CSV（f64）
// ============================================================================

#[derive(Clone, Copy, Default)]
struct TradeRow {
    time_us: i64,
    turnover: f64,
    flag: i64,
    ask_order: i64,
    bid_order: i64,
}

/// 读 transaction CSV（f64）。表头之后每一行都算，后面自己 filter flag != 32。
fn read_trade_f64(code: &str, date: i64) -> Option<Vec<TradeRow>> {
    let path = path_of(code, date, "transaction")?;
    let mut content = String::new();
    File::open(&path).ok()?.read_to_string(&mut content).ok()?;

    let mut out: Vec<TradeRow> = Vec::with_capacity(content.len() / 120 + 1);
    let mut lines = content.lines();
    lines.next(); // 表头
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
        out.push(TradeRow {
            time_us: parse_i64_at(f[TRD_COL_EXCHTIME]) + 8 * 3600 * SEC_US,
            turnover: parse_f64_at(f[TRD_COL_TURNOVER]),
            flag: parse_i64_at(f[TRD_COL_FLAG]),
            ask_order: parse_order_at(f[TRD_COL_ASK_ORDER]),
            bid_order: parse_order_at(f[TRD_COL_BID_ORDER]),
        });
    }
    Some(out)
}

// ============================================================================
// 盘口 CSV（f64，只要五档量之和）
// ============================================================================

#[derive(Clone, Copy, Default)]
struct MktRow {
    time_us: i64,
    ask5: f64,
    bid5: f64,
}

/// 五档量按列顺序求和，与 pandas `df2[[...]].sum(axis=1)` 一致（有 NaN 时跳过，
/// 五个全 NaN 才是 NaN）。
#[inline]
fn sum5(f: &[&[u8]; MKT_NCOL], base: usize) -> f64 {
    let mut s = 0.0f64;
    let mut any = false;
    for k in 0..5 {
        let v = parse_f64_at(f[base + k * 4]);
        if !v.is_nan() {
            s += v;
            any = true;
        }
    }
    if any {
        s
    } else {
        f64::NAN
    }
}

fn read_market_f64(code: &str, date: i64) -> Option<Vec<MktRow>> {
    let path = path_of(code, date, "market_data")?;
    let mut content = String::new();
    File::open(&path).ok()?.read_to_string(&mut content).ok()?;

    let mut out: Vec<MktRow> = Vec::with_capacity(content.len() / 220 + 1);
    let mut lines = content.lines();
    lines.next(); // 表头
    for line in lines {
        if line.is_empty() {
            continue;
        }
        let bytes = line.as_bytes();
        let mut f: [&[u8]; MKT_NCOL] = [&[][..]; MKT_NCOL];
        let mut start = 0usize;
        let mut col = 0usize;
        for (i, &b) in bytes.iter().enumerate() {
            if b == b',' {
                if col < MKT_NCOL {
                    f[col] = &bytes[start..i];
                }
                col += 1;
                start = i + 1;
            }
        }
        if col < MKT_NCOL {
            f[col] = &bytes[start..];
        }
        out.push(MktRow {
            time_us: parse_i64_at(f[MKT_COL_EXCHTIME]) + 8 * 3600 * SEC_US,
            ask5: sum5(&f, MKT_ASK_VOL1),
            bid5: sum5(&f, MKT_BID_VOL1),
        });
    }
    Some(out)
}

// ============================================================================
// pandas 语义的小工具
// ============================================================================

/// pandas Series.quantile(q)（默认线性插值）。NaN 跳过。
///
/// 分段写法与 numpy 的 _lerp 一致：t < 0.5 用 a + (b-a)*t，否则用 b - (b-a)*(1-t)。
/// 实测这样和 pandas 的结果逐位相同（换成 a + (b-a)*t 会在末位差一点，
/// 而这里的分位数要拿去做严格大于比较，边界上不能差）。
fn quantile_linear(vals: &[f64], q: f64) -> f64 {
    let mut v: Vec<f64> = vals.iter().cloned().filter(|x| !x.is_nan()).collect();
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = v.len();
    if n == 1 {
        return v[0];
    }
    let idx = (n as f64 - 1.0) * q;
    let lo = idx.floor();
    let hi = idx.ceil();
    let loi = lo as usize;
    let hii = hi as usize;
    if loi == hii {
        return v[loi];
    }
    let t = idx - lo;
    if t < 0.5 {
        v[loi] + (v[hii] - v[loi]) * t
    } else {
        v[hii] - (v[hii] - v[loi]) * (1.0 - t)
    }
}

/// numpy nanmedian：偶数个取中间两个的算术平均（(a+b)/2，不是 a+(b-a)/2）。
fn median_of(v: &mut [f64]) -> f64 {
    let n = v.len();
    if n == 0 {
        return f64::NAN;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
}

// ============================================================================
// 合并与统计
// ============================================================================

/// merge_asof(forward) 之后的一行。label 是 ask 侧/bid 侧快照序号（两侧各自编号）。
#[derive(Clone, Copy)]
struct MRow {
    time_us: i64,
    turnover: f64,
    flag: i64,
    ask_order: i64,
    bid_order: i64,
    label: usize,
    wait_sec: f64,
}

/// 对每笔成交找「时间不早于它、且离它不超过 tol_us」的第一条快照，并算出 wait。
///
/// 旧代码两个分支：快照本来就按时间单调时直接 merge_asof；
/// 不单调时会抛异常，异常分支把右侧 sort_values('merge_key') 之后重来（label 仍是原行号）。
/// 这里用稳定的 sort_by_key 复现后者，快照本来就有序时是空操作。
fn merge(trades: &[TradeRow], snap_times: &[i64], tol_us: i64) -> Vec<MRow> {
    let mut pairs: Vec<(i64, usize)> = snap_times.iter().copied().enumerate().map(|(i, t)| (t, i)).collect();
    if pairs.windows(2).any(|w| w[0].0 > w[1].0) {
        pairs.sort_by_key(|p| p.0);
    }
    let times: Vec<i64> = pairs.iter().map(|p| p.0).collect();

    let mut out: Vec<MRow> = Vec::with_capacity(trades.len());
    for t in trades.iter() {
        let pos = times.partition_point(|&s| s < t.time_us);
        if pos >= times.len() {
            continue;
        }
        let dt_us = times[pos] - t.time_us;
        if dt_us > tol_us {
            continue;
        }
        out.push(MRow {
            time_us: t.time_us,
            turnover: t.turnover,
            flag: t.flag,
            ask_order: t.ask_order,
            bid_order: t.bid_order,
            label: pairs[pos].1,
            wait_sec: (dt_us * 1000) as f64 / 1e9,
        });
    }
    out
}

/// 一组「按 label 取 turnover 最大的那一笔，再按委托号集合过滤」之后的行。
///
/// 旧代码是 sort_values(['label','turnover'], ascending=[True,False]) 之后 groupby('label').first()。
/// pandas 多键排序走 lexsort_indexer → np.lexsort，是稳定排序，所以并列时保留原顺序，
/// 这里用 Rust 的稳定 sort_by 对应。
fn grouped(win: &[MRow], want: i8, asks: &HashSet<i64>, bids: &HashSet<i64>) -> Vec<MRow> {
    let mut idx: Vec<usize> = (0..win.len())
        .filter(|&i| match want {
            1 => win[i].flag == 66,
            2 => win[i].flag == 83,
            _ => true,
        })
        .collect();
    idx.sort_by(|&a, &b| {
        win[a]
            .label
            .cmp(&win[b].label)
            .then_with(|| {
                win[b]
                    .turnover
                    .partial_cmp(&win[a].turnover)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    });
    let mut out: Vec<MRow> = Vec::new();
    let mut last: Option<usize> = None;
    for &i in idx.iter() {
        let r = win[i];
        if last == Some(r.label) {
            continue;
        }
        last = Some(r.label);
        if asks.contains(&r.ask_order) || bids.contains(&r.bid_order) {
            out.push(r);
        }
    }
    out
}

/// turnover 的 std(ddof=1)/median/max 各自除以 denom，外加 wait 的中位数。
/// 空组按 pandas 的读法全给 NaN。
fn col_stats(rows: &[MRow], denom: f64) -> (f64, f64, f64, f64) {
    let n = rows.len();
    if n == 0 {
        return (f64::NAN, f64::NAN, f64::NAN, f64::NAN);
    }
    let mut tv: Vec<f64> = rows.iter().map(|r| r.turnover).collect();
    let std = if n > 1 {
        let mean = tv.iter().sum::<f64>() / n as f64;
        let var = tv.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / (n - 1) as f64;
        var.sqrt()
    } else {
        f64::NAN
    };
    tv.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let med = if n % 2 == 1 {
        tv[n / 2]
    } else {
        (tv[n / 2 - 1] + tv[n / 2]) / 2.0
    };
    let mx = tv[n - 1];
    let mut wv: Vec<f64> = rows.iter().map(|r| r.wait_sec).collect();
    let wmed = median_of(&mut wv);
    (std / denom, med / denom, mx / denom, wmed)
}

/// 一个 side（ask 或 bid）在一个窗口内的 12 个量。guard 没过返回 None。
fn side_stats(
    win: &[MRow],
    asks: &HashSet<i64>,
    bids: &HashSet<i64>,
    to_all: f64,
) -> Option<[f64; PER_SESSION]> {
    let mut n_buy = 0usize;
    let mut n_sell = 0usize;
    let mut to_buy = 0.0f64;
    let mut to_sell = 0.0f64;
    for r in win.iter() {
        if r.flag == 66 {
            n_buy += 1;
            to_buy += r.turnover;
        } else if r.flag == 83 {
            n_sell += 1;
            to_sell += r.turnover;
        }
    }
    if n_buy == 0 || n_sell == 0 {
        return None;
    }
    let ga = grouped(win, 0, asks, bids);
    let gb = grouped(win, 1, asks, bids);
    let gs = grouped(win, 2, asks, bids);
    let (a_std, a_med, a_max, a_wait) = col_stats(&ga, to_all);
    let (b_std, b_med, b_max, b_wait) = col_stats(&gb, to_buy);
    let (s_std, s_med, s_max, s_wait) = col_stats(&gs, to_sell);
    Some([
        a_std,
        a_med,
        a_max,
        b_std,
        b_med,
        b_max,
        s_std,
        s_med,
        s_max,
        a_max / a_wait,
        b_max / b_wait,
        s_max / s_wait,
    ])
}

/// 一个容差下的 60 个值。ask 侧变量沿用上一时段是旧代码的实际行为，见文件头注释 (b)。
fn run_param(
    win: &[TradeRow],
    snaps_ask: &[i64],
    snaps_bid: &[i64],
    asks: &HashSet<i64>,
    bids: &HashSet<i64>,
    tol_us: i64,
) -> Option<Vec<f64>> {
    let m_ask = merge(win, snaps_ask, tol_us);
    let m_bid = merge(win, snaps_bid, tol_us);
    let to_all: f64 = m_ask.iter().map(|r| r.turnover).sum();

    let mut state_a: Option<[f64; PER_SESSION]> = None;
    let mut res: Vec<f64> = Vec::with_capacity(PER_SESSION * 5);
    for (lo, hi) in SESSION_SPANS.iter() {
        let wa: Vec<MRow> = m_ask
            .iter()
            .filter(|r| in_span(r.time_us, *lo, *hi))
            .copied()
            .collect();
        let wb: Vec<MRow> = m_bid
            .iter()
            .filter(|r| in_span(r.time_us, *lo, *hi))
            .copied()
            .collect();
        let a = side_stats(&wa, asks, bids, to_all);
        let b = side_stats(&wb, asks, bids, to_all);
        let a_eff = match a {
            Some(v) => {
                state_a = Some(v);
                Some(v)
            }
            None => state_a,
        };
        match (a_eff, b) {
            // bid 侧过了而 ask 侧从来没有值 → 旧代码 NameError → 整条 None
            (None, Some(_)) => return None,
            (_, None) => res.extend(std::iter::repeat(f64::NAN).take(PER_SESSION)),
            (Some(av), Some(bv)) => {
                for i in 0..PER_SESSION {
                    res.push(av[i] - bv[i]);
                }
            }
        }
    }
    Some(res)
}

/// 窗口内按委托号分组求成交额之和，只留 > 0.9 分位数的那些委托号。
fn top_orders(win: &[TradeRow], is_ask: bool) -> HashSet<i64> {
    let mut m: HashMap<i64, f64> = HashMap::new();
    for t in win.iter() {
        let k = if is_ask { t.ask_order } else { t.bid_order };
        if k == i64::MIN {
            continue;
        }
        *m.entry(k).or_insert(0.0) += t.turnover;
    }
    if m.is_empty() {
        return HashSet::new();
    }
    let sums: Vec<f64> = m.values().copied().collect();
    let q = quantile_linear(&sums, 0.9);
    m.into_iter().filter(|(_, v)| *v > q).map(|(k, _)| k).collect()
}

/// 逐股的一天：算不出来返回 None（上游补 NaN，保证因子个数恒定）。
pub fn compute_stock(code: &str, date: i64) -> Option<Vec<f32>> {
    let trades_all = read_trade_f64(code, date)?;
    let trades: Vec<TradeRow> = trades_all.into_iter().filter(|r| r.flag != 32).collect();
    if trades.is_empty() {
        return None;
    }
    let mkt = read_market_f64(code, date)?;

    // 盘口：先切窗口，再按 ask5/bid5 的 0.8 分位数筛快照（保留原行号顺序）
    let mut w_time: Vec<i64> = Vec::new();
    let mut w_ask5: Vec<f64> = Vec::new();
    let mut w_bid5: Vec<f64> = Vec::new();
    for m in mkt.iter() {
        if in_span(m.time_us, LD_0930, LD_1457) {
            w_time.push(m.time_us);
            w_ask5.push(m.ask5);
            w_bid5.push(m.bid5);
        }
    }
    let qa = quantile_linear(&w_ask5, 0.8);
    let qb = quantile_linear(&w_bid5, 0.8);
    let snaps_ask: Vec<i64> = (0..w_time.len())
        .filter(|&i| w_ask5[i] - qa > 0.0)
        .map(|i| w_time[i])
        .collect();
    let snaps_bid: Vec<i64> = (0..w_time.len())
        .filter(|&i| w_bid5[i] - qb > 0.0)
        .map(|i| w_time[i])
        .collect();
    // 旧代码最外层要求两侧都快照非空，否则 res1/res2 都没定义、返回 None
    if snaps_ask.is_empty() || snaps_bid.is_empty() {
        return None;
    }

    // 逐笔：切窗口（原顺序即按时间有序）
    let win: Vec<TradeRow> = trades
        .iter()
        .filter(|r| in_span(r.time_us, LD_0930, LD_1457))
        .copied()
        .collect();
    let asks = top_orders(&win, true);
    let bids = top_orders(&win, false);

    let mut out: Vec<f32> = Vec::with_capacity(N_OUT);
    for tol in TOLS_US.iter() {
        let r = run_param(&win, &snaps_ask, &snaps_bid, &asks, &bids, *tol)?;
        out.extend(r.iter().map(|&v| v as f32));
    }
    debug_assert_eq!(out.len(), N_OUT);
    Some(out)
}

/// hm11 全是逐股因子，没有横截面版本。
pub fn compute_cross(_date: i64) -> Option<(Vec<String>, Vec<f32>)> {
    None
}
