//! hm12「梦想三级跳」的 Rust 复现（旧框架 design_whatever_out.TickDataBase）。
//!
//! 旧文件：/home/chenzongwei/pythoncode/成果hm12_成果hm13_梦想三级跳/hm12_okay.py
//!   Go.get_factor 参数 get_many=0 / read_full=0 / change_time=1，
//!   也就是「一整天的逐笔 + 一整天的盘口」各自一张表，窗口由旧代码自己切。
//!
//! ════════════════════════════════════════════════════════════════════
//! 旧代码逐步对照
//! ════════════════════════════════════════════════════════════════════
//! 1) df  = read_trade_data_single(with_retreat=0)  → 丢掉 flag==32 的撤单行
//!    df2 = read_market_data_single()               → 10 档盘口（注意：**不过滤 last_prc**）
//! 2) df2 先切 [09:30:00, 14:57:00]，按行号打 label（0 起），
//!    再 drop_duplicates(subset=['exchtime']) 得到 df2_label（同一时刻只留第一行）
//! 3) A = concat([df, df2_label]).sort_values(['exchtime'])；A.label = ffill
//!    → 每笔成交拿到「排序后位于它前面的最后一个快照」的 label，也就是不超过它的
//!      最近一个快照；成交早于首个快照时为 NaN。
//!    ！！这里 sort_values 是单键，走 pandas nargsort → DatetimeArray.argsort('quicksort')
//!      → numpy aquicksort_datetime。它对相等的键**不是稳定排序**，平局顺序由分区过程决定。
//!      平局归属会改变成交落在哪个快照上，所以下面 np_argsort_quicksort 是逐行复刻，
//!      不能换成 sort_by_key（这一段与 hm013.rs 同源）。
//! 4) A 再切一次 [09:30:00, 14:57:00]，按 label 分组
//! 5) DF66 = flag==66 的成交，DF83 = flag==83 的成交（盘口行 flag 是 NaN，自动排除）
//!    volumes_buy[label]  = DF66 组内 volume 之和
//!    turnovers_buy[label]= DF66 组内 turnover 之和（DF83 同理）
//!    price3_asks = turnovers_buy / volumes_buy，dropna（只丢 NaN，inf 留着）
//!    price1_asks = 盘口 ask_prc1 按上面的 label 对齐
//!    注意 turnover/volume 是「这个快照下所有 66 成交的成交额之和 / 成交量之和」，
//!    是「现实价格」；ask_prc1/bid_prc1 是「最理想价格」。
//! 6) 五个时段各算一个量：
//!      x = |price3_asks - price1_asks|            （只看该时段内、且两侧都非空的 label）
//!      y = |price1_bids - price3_bids|
//!      d = x / (x + y)，分母恰好为 0 的先换成 NaN
//!      mean4_p1 = d 的均值（跳过 NaN；两个子集任一为空则整段 NaN）
//!    时段顺序 all / open30 / morning / afternoon / close30，共 5 个值。
//!
//! names（names_single）：all_mean4_p1, open30_mean4_p1, morning_mean4_p1,
//!   afternoon_mean4_p1, close30_mean4_p1
//! names_save 就是这 5 个，names_in_db 为空。
//!
//! 精度：逐笔的 volume/turnover 与盘口的 ask_prc1/bid_prc1 都按 f64 自解析 CSV
//! （列索引与 fast_csv_reader 的常量一致），与 hm013 同一套读法。

use std::collections::HashMap;
use std::fs::File;
use std::io::Read;

/// 逐笔 CSV 的列索引（0 基，与 fast_csv_reader 一致）
const TRD_COL_EXCHTIME: usize = 4;
const TRD_COL_VOLUME: usize = 8;
const TRD_COL_TURNOVER: usize = 9;
const TRD_COL_FLAG: usize = 10;
const TRD_NCOL: usize = 15;

/// 盘口 CSV 的列索引（0 基，与 fast_csv_reader 的常量一致）
const MKT_COL_EXCHTIME: usize = 4;
const MKT_COL_ASK_PRC1: usize = 21;
const MKT_COL_BID_PRC1: usize = 23;
const MKT_NCOL: usize = 61;

const DAY_US: i64 = 86_400_000_000;
const SEC_US: i64 = 1_000_000;
const LD_0930: i64 = (9 * 3600 + 30 * 60) * SEC_US;
const LD_1000: i64 = 10 * 3600 * SEC_US;
const LD_1130: i64 = (11 * 3600 + 30 * 60) * SEC_US;
const LD_1300: i64 = 13 * 3600 * SEC_US;
const LD_1430: i64 = (14 * 3600 + 30 * 60) * SEC_US;
const LD_1457: i64 = (14 * 3600 + 57 * 60) * SEC_US;

const SESSION_NAMES: [&str; 5] = ["all", "open30", "morning", "afternoon", "close30"];
/// 五个时段（真实钟点的微秒闭区间，旧代码用 between_time，两端都含）
const SESSIONS: [(i64, i64); 5] = [
    (LD_0930, LD_1457),
    (LD_0930, LD_1000),
    (LD_0930, LD_1130),
    (LD_1300, LD_1457),
    (LD_1430, LD_1457),
];
pub const N_OUT: usize = 5;

pub fn names() -> Vec<String> {
    SESSION_NAMES
        .iter()
        .map(|s| format!("{}_mean4_p1", s))
        .collect()
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

/// 把一行按逗号切成最多 n 段
#[inline]
fn split_cols<'a>(line: &'a str, buf: &mut [&'a [u8]; 64], n: usize) -> usize {
    let bytes = line.as_bytes();
    for slot in buf.iter_mut().take(n) {
        *slot = &[][..];
    }
    let mut start = 0usize;
    let mut col = 0usize;
    for (i, &b) in bytes.iter().enumerate() {
        if b == b',' {
            if col < n {
                buf[col] = &bytes[start..i];
            }
            col += 1;
            start = i + 1;
        }
    }
    if col < n {
        buf[col] = &bytes[start..];
    }
    col
}

fn find_file(code: &str, date: i64, subdir: &str, suffix: &str) -> Option<std::path::PathBuf> {
    let filename = format!("{}_{}_{}", code, date, suffix);
    for root in crate::data_paths::level2_roots() {
        let p = root.join(date.to_string()).join(subdir).join(&filename);
        if p.exists() {
            return Some(p);
        }
    }
    None
}

// ============================================================================
// 读取
// ============================================================================

#[derive(Clone, Copy, Default)]
struct TradeRow {
    time_us: i64,
    volume: f64,
    turnover: f64,
    flag: f64,
}

fn read_trade_f64(code: &str, date: i64) -> Option<Vec<TradeRow>> {
    let path = find_file(code, date, "transaction", "transaction.csv")?;
    let mut content = String::new();
    File::open(&path).ok()?.read_to_string(&mut content).ok()?;

    let mut out: Vec<TradeRow> = Vec::with_capacity(content.len() / 120 + 1);
    let mut lines = content.lines();
    lines.next(); // 表头
    for line in lines {
        if line.is_empty() {
            continue;
        }
        let mut buf: [&[u8]; 64] = [&[][..]; 64];
        split_cols(line, &mut buf, TRD_NCOL);
        out.push(TradeRow {
            time_us: parse_i64_at(buf[TRD_COL_EXCHTIME]) + 8 * 3600 * SEC_US,
            volume: parse_f64_at(buf[TRD_COL_VOLUME]),
            turnover: parse_f64_at(buf[TRD_COL_TURNOVER]),
            flag: parse_f64_at(buf[TRD_COL_FLAG]),
        });
    }
    Some(out)
}

#[derive(Clone, Copy, Default)]
struct SnapRow {
    time_us: i64,
    ask_prc1: f64,
    bid_prc1: f64,
}

fn read_market_f64(code: &str, date: i64) -> Option<Vec<SnapRow>> {
    let path = find_file(code, date, "market_data", "market_data.csv")?;
    let mut content = String::new();
    File::open(&path).ok()?.read_to_string(&mut content).ok()?;

    let mut out: Vec<SnapRow> = Vec::with_capacity(content.len() / 200 + 1);
    let mut lines = content.lines();
    lines.next(); // 表头
    for line in lines {
        if line.is_empty() {
            continue;
        }
        let mut buf: [&[u8]; 64] = [&[][..]; 64];
        split_cols(line, &mut buf, MKT_NCOL);
        out.push(SnapRow {
            time_us: parse_i64_at(buf[MKT_COL_EXCHTIME]) + 8 * 3600 * SEC_US,
            ask_prc1: parse_f64_at(buf[MKT_COL_ASK_PRC1]),
            bid_prc1: parse_f64_at(buf[MKT_COL_BID_PRC1]),
        });
    }
    Some(out)
}

// ============================================================================
// numpy aquicksort_datetime 的逐行复刻（与 hm013.rs 同源）
// ============================================================================

#[inline]
fn msb(mut u: u64) -> i32 {
    let mut d = 0i32;
    while u > 1 {
        u >>= 1;
        d += 1;
    }
    d
}

/// numpy aheapsort_（索引版堆排序），只在与 quicksort 深度耗尽时才会用到
fn aheapsort(v: &[i64], idx: &mut [usize], lo: usize, n: usize) {
    if n < 2 {
        return;
    }
    let mut n = n;
    let mut l = n >> 1;
    while l > 0 {
        let tmp = idx[lo + l - 1];
        let mut i = l;
        let mut j = l << 1;
        while j <= n {
            if j < n && v[idx[lo + j - 1]] < v[idx[lo + j]] {
                j += 1;
            }
            if v[tmp] < v[idx[lo + j - 1]] {
                idx[lo + i - 1] = idx[lo + j - 1];
                i = j;
                j += j;
            } else {
                break;
            }
        }
        idx[lo + i - 1] = tmp;
        l -= 1;
    }
    while n > 1 {
        let tmp = idx[lo + n - 1];
        idx[lo + n - 1] = idx[lo];
        n -= 1;
        let mut i = 1usize;
        let mut j = 2usize;
        while j <= n {
            if j < n && v[idx[lo + j - 1]] < v[idx[lo + j]] {
                j += 1;
            }
            if v[tmp] < v[idx[lo + j - 1]] {
                idx[lo + i - 1] = idx[lo + j - 1];
                i = j;
                j += j;
            } else {
                break;
            }
        }
        idx[lo + i - 1] = tmp;
    }
}

/// numpy `aquicksort_<datetime_tag>`（npysort/quicksort.cpp）的逐行复刻。
/// 平局顺序必须与它一致，否则成交落在哪个快照上就会变。
fn np_argsort_quicksort(v: &[i64]) -> Vec<usize> {
    let num = v.len();
    let mut idx: Vec<usize> = (0..num).collect();
    if num <= 1 {
        return idx;
    }
    const SMALL_QUICKSORT: usize = 15;

    let mut stack: Vec<(usize, usize, i32)> = Vec::with_capacity(64);
    let mut pl = 0usize;
    let mut pr = num - 1;
    let mut cdepth: i32 = msb(num as u64) * 2;

    loop {
        if cdepth < 0 {
            aheapsort(v, &mut idx, pl, pr - pl + 1);
        } else {
            while pr - pl > SMALL_QUICKSORT {
                let pm = pl + ((pr - pl) >> 1);
                if v[idx[pm]] < v[idx[pl]] {
                    idx.swap(pm, pl);
                }
                if v[idx[pr]] < v[idx[pm]] {
                    idx.swap(pr, pm);
                }
                if v[idx[pm]] < v[idx[pl]] {
                    idx.swap(pm, pl);
                }
                let vp = v[idx[pm]];
                let mut pi = pl;
                let mut pj = pr - 1;
                idx.swap(pm, pj);
                loop {
                    pi += 1;
                    while v[idx[pi]] < vp {
                        pi += 1;
                    }
                    pj -= 1;
                    while vp < v[idx[pj]] {
                        pj -= 1;
                    }
                    if pi >= pj {
                        break;
                    }
                    idx.swap(pi, pj);
                }
                let pk = pr - 1;
                idx.swap(pi, pk);
                if pi - pl < pr - pi {
                    stack.push((pi + 1, pr, cdepth - 1));
                    pr = pi - 1;
                } else {
                    stack.push((pl, pi - 1, cdepth - 1));
                    pl = pi + 1;
                }
                cdepth -= 1;
            }
            let mut pi = pl + 1;
            while pi <= pr {
                let vi = idx[pi];
                let vp = v[vi];
                let mut pj = pi;
                let mut pk = pi - 1;
                while pj > pl && vp < v[idx[pk]] {
                    idx[pj] = idx[pk];
                    pj -= 1;
                    pk -= 1;
                }
                idx[pj] = vi;
                pi += 1;
            }
        }
        match stack.pop() {
            Some((npl, npr, d)) => {
                pl = npl;
                pr = npr;
                cdepth = d;
            }
            None => break,
        }
    }
    idx
}

// ============================================================================
// 指标
// ============================================================================

/// 一个快照在某个时段里的 (label, |现实价 - 理想价|)。
fn side_dist(
    snaps: &[(i64, f64, f64, i64)], // (time_us, ask_prc1, bid_prc1, label) 去重后的快照
    lo: i64,
    hi: i64,
    sums_vol: &HashMap<i64, f64>,
    sums_to: &HashMap<i64, f64>,
    is_ask: bool,
) -> Vec<(i64, f64)> {
    let mut out = Vec::new();
    for &(t, ask1, bid1, label) in snaps.iter() {
        if !in_span(t, lo, hi) {
            continue;
        }
        let (v, to) = match (sums_vol.get(&label), sums_to.get(&label)) {
            (Some(a), Some(b)) => (*a, *b),
            _ => continue,
        };
        let p3 = to / v;
        // 旧代码的 (turnovers/volumes).dropna()：只丢 NaN，inf 留着
        if p3.is_nan() {
            continue;
        }
        let p1 = if is_ask { ask1 } else { bid1 };
        let d = if is_ask { p3 - p1 } else { p1 - p3 };
        out.push((label, d.abs()));
    }
    out
}

/// pandas 的 mean()：跳过 NaN；全 NaN / 空返回 NaN
fn nanmean(v: &[f64]) -> f64 {
    let mut n = 0usize;
    let mut s = 0.0f64;
    for &x in v {
        if !x.is_nan() {
            n += 1;
            s += x;
        }
    }
    if n == 0 {
        f64::NAN
    } else {
        s / n as f64
    }
}

/// 逐股的一天：算不出来返回 None（上游补 NaN，保证因子个数恒定）。
pub fn compute_stock(code: &str, date: i64) -> Option<Vec<f32>> {
    // 先全读进来，再自己按数值过滤 flag==32（"32.0" 也要能过滤掉）
    let trades_all = read_trade_f64(code, date)?;
    let trades: Vec<&TradeRow> = trades_all.iter().filter(|r| r.flag != 32.0).collect();
    // 盘口：不过滤 last_prc（旧代码这里用的是 read_market_data_single）
    let mkt_all = read_market_f64(code, date)?;
    let mkt: Vec<SnapRow> = mkt_all
        .into_iter()
        .filter(|r| in_span(r.time_us, LD_0930, LD_1457))
        .collect();

    // label = 窗口内的行号；同一 exchtime 只留第一行（concat 里用的是去重后的那份）
    let mut dedup: Vec<(i64, f64, f64, i64)> = Vec::new();
    {
        let mut seen: std::collections::HashSet<i64> = std::collections::HashSet::new();
        for (i, r) in mkt.iter().enumerate() {
            if seen.insert(r.time_us) {
                dedup.push((r.time_us, r.ask_prc1, r.bid_prc1, i as i64));
            }
        }
    }

    // concat([逐笔, 去重盘口]).sort_values('exchtime') 后 ffill
    let n_t = trades.len();
    let mut times: Vec<i64> = Vec::with_capacity(n_t + dedup.len());
    times.extend(trades.iter().map(|r| r.time_us));
    times.extend(dedup.iter().map(|d| d.0));
    let perm = np_argsort_quicksort(&times);

    // 每笔成交拿到「排在它前面的最后一个盘口行」的 label；早于首个盘口的是 NaN（用 -1 表示）
    let mut labels: Vec<i64> = vec![-1; n_t];
    {
        let mut cur: i64 = -1;
        for &i in perm.iter() {
            if i >= n_t {
                cur = dedup[i - n_t].3;
            } else {
                labels[i] = cur;
            }
        }
    }

    // 窗口内、按 label 分组求和（flag==66 / flag==83 各一套）
    let mut v66: HashMap<i64, f64> = HashMap::new();
    let mut t66: HashMap<i64, f64> = HashMap::new();
    let mut v83: HashMap<i64, f64> = HashMap::new();
    let mut t83: HashMap<i64, f64> = HashMap::new();
    for (i, t) in trades.iter().enumerate() {
        if !in_span(t.time_us, LD_0930, LD_1457) {
            continue;
        }
        let lab = labels[i];
        if lab < 0 {
            continue;
        }
        if t.flag == 66.0 {
            *v66.entry(lab).or_insert(0.0) += t.volume;
            *t66.entry(lab).or_insert(0.0) += t.turnover;
        } else if t.flag == 83.0 {
            *v83.entry(lab).or_insert(0.0) += t.volume;
            *t83.entry(lab).or_insert(0.0) += t.turnover;
        }
    }

    let mut out: Vec<f32> = Vec::with_capacity(N_OUT);
    for (lo, hi) in SESSIONS.iter() {
        let a = side_dist(&dedup, *lo, *hi, &v66, &t66, true);
        let b = side_dist(&dedup, *lo, *hi, &v83, &t83, false);
        if a.is_empty() || b.is_empty() {
            out.push(f32::NAN);
            continue;
        }
        // d = x/(x+y)，分母恰好为 0 的换成 NaN；x 或 y 缺一边的也是 NaN，均值会跳过
        let bidx: HashMap<i64, f64> = b.iter().cloned().collect();
        let mut d: Vec<f64> = Vec::new();
        for &(lab, x) in a.iter() {
            if let Some(&y) = bidx.get(&lab) {
                let den = x + y;
                d.push(if den == 0.0 { f64::NAN } else { x / den });
            }
        }
        out.push(nanmean(&d) as f32);
    }
    debug_assert_eq!(out.len(), N_OUT);
    Some(out)
}

/// hm12 全是逐股因子，没有横截面版本。
pub fn compute_cross(_date: i64) -> Option<(Vec<String>, Vec<f32>)> {
    None
}
