//! hm13「隐藏订单深度」（旧框架 design_whatever_out.TickDataBase）的 Rust 复现。
//!
//! 旧文件：/home/chenzongwei/pythoncode/成果hm12_成果hm13_梦想三级跳/hm13_okay.py
//!   run_base 里的 Go.get_factor，参数 get_many=0 / read_full=0 / change_time=1，
//!   也就是「一整天的逐笔 + 一整天的盘口」各自一张表，窗口由旧代码自己切。
//!
//! ════════════════════════════════════════════════════════════════════
//! 旧代码逐步对照（写代码前先抄在这里，方便复核）
//! ════════════════════════════════════════════════════════════════════
//! 1) df  = read_trade_data_single(with_retreat=0)  → 丢掉 flag==32 的撤单行
//!    df2 = read_market_data_single()               → 10 档盘口
//! 2) df2 先切 [09:30:00, 14:57:00]，按行号打 label（0 起），
//!    再 drop_duplicates(subset=['exchtime']) 得到 df2_label（同一时刻只留第一行）
//! 3) A = concat([df, df2_label]).sort_values(['exchtime'])；A.label = ffill
//!    → 每笔成交拿到「排序后位于它前面的最后一个快照」的 label，
//!      也就是不超过它的最近一个快照；成交早于首个快照时为 NaN。
//!    ！！这里 sort_values 是单键，走 pandas nargsort → DatetimeArray.argsort('quicksort')
//!      → numpy aquicksort_datetime。它对相等的键**不是稳定排序**，平局顺序由分区过程决定。
//!      平局归属会改变因子值（实测相对差可达 3e-3，远超验收的 1e-4），
//!      所以下面 np_argsort_quicksort 是逐行复刻，不能换成 sort_by_key。
//! 4) df 再切一次窗口 [09:30:00, 14:57:00]，turn = 窗口内全部成交量（含 label 为 NaN 的行）
//! 5) 按 bid_order / ask_order 分组求 volume、turnover 之和
//!    （分组在窗口过滤之后、dropna 之前做，所以 label 为 NaN 的行也计入分组和）
//! 6) 每笔成交按自己的 label 取快照，把 11 档（第 11 档价 = 第 10 档价 ±0.01、
//!    第 11 档量 = turn）按「吃掉 U，也就是同一 bid_order 的总成交量，需要走到第几档」
//!    算加权均价 price2；price2_asks 取 max(ask_prc1, price2)，
//!    price2_bids 取 min(bid_prc1, price2，其中恰好为 0 的先换成 1e10)。
//!    real_buy = 该 bid_order 的成交额之和 / U，real_sell 同理按 ask_order。
//! 7) 五个时段各算 corr(price2_asks - real_buy, U) 与 corr(price2_bids - real_sell, W)，
//!    时段顺序 all / open30 / morning / afternoon / close30，每段两个值，共 10 个。
//!
//! names（names_single）：all_asks, all_bids, open30_asks, open30_bids, morning_asks,
//!   morning_bids, afternoon_asks, afternoon_bids, close30_asks, close30_bids
//! names_save：all_bids, close30_bids, all_asks, open30_bids
//!
//! ════════════════════════════════════════════════════════════════════
//! 精度：两个都不能用 f32 的读取器
//! ════════════════════════════════════════════════════════════════════
//! · 撤单过滤仍按规范自己 filter flag != 32（不靠 read_trade_fast_inner 的 with_retreat：
//!   库里那条判断按字节比 "32"，2025 年的明细写的是 "32.0"，过滤不生效）。
//! · 但成交额与盘口价格都**不能**用 fast_csv_reader 的 f32：
//!   price2 - real_buy 的量级只有 5e-3，而 real_buy = 组内成交额之和 / 组内成交量之和。
//!   实测 000001@20250303 只把盘口价格降到 f32，10 个值里最大相对误差就 9.75e-5（阈值 1e-4）；
//!   600519@20250304 再叠上 f32 成交额，afternoon_asks 直接超：f64 是 9.8918443966e-05，
//!   f32 是 9.8903042386e-05（相对差 1.56e-4）。
//!   所以逐笔与盘口都在本文件里按 f64 解析 CSV（列索引与 fast_csv_reader 的常量一致）。

use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::Read;

/// 逐笔 CSV 的列索引（0 基，与 fast_csv_reader 一致；2018 年及以前末尾多一列 num，不影响前 15 列）
const TRD_COL_EXCHTIME: usize = 4;
const TRD_COL_VOLUME: usize = 8;
const TRD_COL_TURNOVER: usize = 9;
const TRD_COL_FLAG: usize = 10;
const TRD_COL_ASK_ORDER: usize = 13;
const TRD_COL_BID_ORDER: usize = 14;
const TRD_NCOL: usize = 15;

/// 盘口 CSV 的列索引（0 基，与 fast_csv_reader 的常量一致）
const MKT_COL_EXCHTIME: usize = 4;
const MKT_ASK_PRC_BASE: usize = 21; // ask_prc1=21, ask_vol1=22, bid_prc1=23, bid_vol1=24, ...
const MKT_NCOL: usize = 61;

const DAY_US: i64 = 86_400_000_000;
const SEC_US: i64 = 1_000_000;
const LD_0930: i64 = (9 * 3600 + 30 * 60) * SEC_US;
const LD_1000: i64 = 10 * 3600 * SEC_US;
const LD_1130: i64 = (11 * 3600 + 30 * 60) * SEC_US;
const LD_1300: i64 = 13 * 3600 * SEC_US;
const LD_1430: i64 = (14 * 3600 + 30 * 60) * SEC_US;
const LD_1457: i64 = (14 * 3600 + 57 * 60) * SEC_US;

/// 五个时段（真实钟点的微秒闭区间，旧代码用的是 between_time，两端都含）
const SESSIONS: [(i64, i64); 5] = [
    (LD_0930, LD_1457),
    (LD_0930, LD_1000),
    (LD_0930, LD_1130),
    (LD_1300, LD_1457),
    (LD_1430, LD_1457),
];

const SESSION_NAMES: [&str; 5] = ["all", "open30", "morning", "afternoon", "close30"];
const PER_SESSION: usize = 2;
pub const N_OUT: usize = PER_SESSION * 5;

pub fn names() -> Vec<String> {
    let mut out = Vec::with_capacity(N_OUT);
    for s in SESSION_NAMES.iter() {
        out.push(format!("{}_asks", s));
        out.push(format!("{}_bids", s));
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
// 逐笔 CSV 的 f64 读取
// ============================================================================

/// 一条逐笔成交（只留计算要用的列）
#[derive(Clone, Copy, Default)]
struct TradeRow {
    time_us: i64,
    volume: f64,
    turnover: f64,
    flag: i64,
    bid_order: i64,
    ask_order: i64,
}

/// 读 transaction CSV（f64）。行集与 pandas 读整张表一致：
/// 表头之后每一行都算，exchtime==0 的行也保留（旧代码里它排在最早的 08:00，
/// 会被后面 between_time 丢掉，但它参与了排序数组，保留才和旧代码的平局顺序一致）。
fn read_trade_f64(code: &str, date: i64) -> Option<Vec<TradeRow>> {
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
            volume: parse_f64_at(f[TRD_COL_VOLUME]),
            turnover: parse_f64_at(f[TRD_COL_TURNOVER]),
            flag: parse_i64_at(f[TRD_COL_FLAG]),
            bid_order: parse_i64_at(f[TRD_COL_BID_ORDER]),
            ask_order: parse_i64_at(f[TRD_COL_ASK_ORDER]),
        });
    }
    Some(out)
}

// ============================================================================
// 盘口 CSV 的 f64 读取
// ============================================================================

/// 一天的盘口快照（只留计算要用的列，价格保持 f64）
#[derive(Clone, Copy, Default)]
struct MktRow {
    time_us: i64,
    ask_prc: [f64; 10],
    ask_vol: [f64; 10],
    bid_prc: [f64; 10],
    bid_vol: [f64; 10],
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

/// 读 market_data CSV（f64），跳过表头与 exchtime==0 的行。
/// 行集与 pandas 读整张表的行集一致（旧代码不过滤涨跌停，这里也不过滤）。
fn read_market_f64(code: &str, date: i64) -> Option<Vec<MktRow>> {
    let filename = format!("{}_{}_market_data.csv", code, date);
    let mut path = None;
    for root in crate::data_paths::level2_roots() {
        let p = root.join(date.to_string()).join("market_data").join(&filename);
        if p.exists() {
            path = Some(p);
            break;
        }
    }
    let path = path?;

    let mut content = String::new();
    File::open(&path).ok()?.read_to_string(&mut content).ok()?;

    let mut out: Vec<MktRow> = Vec::with_capacity(content.len() / 200 + 1);
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

        let ex = parse_i64_at(f[MKT_COL_EXCHTIME]);
        if ex == 0 {
            continue;
        }
        let mut row = MktRow {
            time_us: ex + 8 * 3600 * SEC_US,
            ..Default::default()
        };
        for k in 0..10 {
            let base = MKT_ASK_PRC_BASE + k * 4;
            row.ask_prc[k] = parse_f64_at(f[base]);
            row.ask_vol[k] = parse_f64_at(f[base + 1]);
            row.bid_prc[k] = parse_f64_at(f[base + 2]);
            row.bid_vol[k] = parse_f64_at(f[base + 3]);
        }
        out.push(row);
    }
    Some(out)
}

// ============================================================================
// numpy aquicksort_datetime 的逐行复刻
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
    // C 里 a = tosort - 1，对应这里的 a[k] = idx[lo + k - 1]
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
///
/// 平局顺序必须与它一致：pandas 单键 sort_values 走 DatetimeArray.argsort('quicksort')，
/// 落到这个函数；不稳定的分区过程决定了同一时刻的成交与快照谁在前，
/// 而谁在前又决定了成交取哪一档快照，实测能让最终相关系数差 3e-3。
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
                // 大的一半压栈，小的一半继续循环（与 C 版一致）
                if pi - pl < pr - pi {
                    stack.push((pi + 1, pr, cdepth - 1));
                    pr = pi - 1;
                } else {
                    stack.push((pl, pi - 1, cdepth - 1));
                    pl = pi + 1;
                }
                cdepth -= 1;
            }
            // 插入排序（对相等元素稳定，和 C 版一致）
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

/// 11 档（第 11 档是人为补的）累计量价走出来的加权均价。
///
/// 旧代码的写法是：
///   first  = 累计量第一次严格超过 amount 的那一档留 amount，其余置 NaN
///            （注意 keep_first_non_nan_mask 会把这一档之后的也都置 NaN，
///              所以最后只剩这一个非 NaN 值，不是「前面的档都留下」）
///   money  = 上一档累计金额 + 本档价 × (amount - 上一档累计量)   （第 1 档没有上一档，为 NaN）
///   price2 = money.sum() / amount        ← pandas 的 sum 跳过 NaN，全 NaN 时返回 0
/// 也就是标准的「吃穿前 k-1 档、余量按第 k 档价格算」。
/// 超过是严格大于，恰好等于 amount 不算。
fn price2_levels(prc: &[f64; 11], vol: &[f64; 11], amount: f64) -> f64 {
    let mut cum_v = [0.0f64; 11];
    let mut cum_m = [0.0f64; 11];
    let (mut sv, mut sm) = (0.0f64, 0.0f64);
    let (mut nan_v, mut nan_m) = (false, false);
    for k in 0..11 {
        if vol[k].is_nan() {
            nan_v = true;
        } else {
            sv += vol[k];
        }
        let m = prc[k] * vol[k];
        if m.is_nan() {
            nan_m = true;
        } else {
            sm += m;
        }
        cum_v[k] = if nan_v { f64::NAN } else { sv };
        cum_m[k] = if nan_m { f64::NAN } else { sm };
    }

    let mut kstar: Option<usize> = None;
    for k in 0..11 {
        if cum_v[k] > amount {
            kstar = Some(k);
            break;
        }
    }
    let kstar = match kstar {
        Some(k) => k,
        None => return 0.0 / amount,
    };
    if kstar == 0 {
        // 第 1 档就没有上一档，money 是 NaN，pandas 求和得 0
        return 0.0 / amount;
    }
    let a = cum_m[kstar - 1];
    let b = prc[kstar] * (amount - cum_v[kstar - 1]);
    if a.is_nan() || b.is_nan() {
        // 该档金额是 NaN → pandas 求和得 0
        return 0.0 / amount;
    }
    (a + b) / amount
}

#[inline]
fn select_max(a: f64, b: f64) -> f64 {
    (a + b + (a - b).abs()) / 2.0
}

#[inline]
fn select_min(a: f64, b: f64) -> f64 {
    (a + b - (a - b).abs()) / 2.0
}

/// pandas DataFrame.corr() 的 Pearson：逐行复刻 pandas/_libs/algos.pyx 的 nancorr。
///
/// 它用的是 Welford 在线算法，不是「sum(x²) - sum(x)²/n」那种一次遍历公式。
/// 正常数据上两者只差 1e-16 量级，但在**退化情形**差得很远：停牌股那类某一列几乎恒定，
/// 一次遍历公式会因抵消误差给出一个假的正方差，算出 0 或垃圾值；Welford 给出正好 0，
/// divisor == 0 → NaN。旧代码在这种股票上就是 NaN，所以必须用同一套算法。
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
    // corr() 的 min_periods 默认是 1
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

/// 逐股的一天：算不出来返回 None（上游补 NaN，保证因子个数恒定）。
pub fn compute_stock(code: &str, date: i64) -> Option<Vec<f32>> {
    // 先全读进来，再自己按数值过滤 flag==32（"32.0" 也要能过滤掉）
    let trades_all = read_trade_f64(code, date)?;
    let trades: Vec<&TradeRow> = trades_all.iter().filter(|r| r.flag != 32).collect();
    if trades.is_empty() {
        return None;
    }

    // 盘口：窗口内全部快照（旧代码不过滤涨跌停，这里也不过滤）
    let mkt_all = read_market_f64(code, date)?;
    let mkt: Vec<MktRow> = mkt_all
        .into_iter()
        .filter(|r| in_span(r.time_us, LD_0930, LD_1457))
        .collect();
    if mkt.is_empty() {
        return None;
    }

    // ── label：快照按行号编号，同一 exchtime 只留第一行（drop_duplicates）
    let mut snap_time: Vec<i64> = Vec::new();
    let mut snap_label: Vec<i64> = Vec::new();
    {
        let mut seen: HashSet<i64> = HashSet::with_capacity(mkt.len());
        for (i, r) in mkt.iter().enumerate() {
            if seen.insert(r.time_us) {
                snap_time.push(r.time_us);
                snap_label.push(i as i64);
            }
        }
    }

    // ── concat([逐笔, 快照]).sort_values(['exchtime']) 后 ffill
    let n_t = trades.len();
    let mut times: Vec<i64> = Vec::with_capacity(n_t + snap_time.len());
    times.extend(trades.iter().map(|r| r.time_us));
    times.extend(snap_time.iter().copied());
    let perm = np_argsort_quicksort(&times);

    let mut labels: Vec<i64> = vec![-1; n_t];
    {
        let mut cur: i64 = -1;
        for &i in perm.iter() {
            if i >= n_t {
                cur = snap_label[i - n_t];
            } else {
                labels[i] = cur;
            }
        }
    }

    // ── 窗口内的逐笔（分组和、turn 都只统计窗口内）
    let mut idx_win: Vec<usize> = Vec::new();
    for (i, t) in trades.iter().enumerate() {
        if in_span(t.time_us, LD_0930, LD_1457) {
            idx_win.push(i);
        }
    }
    if idx_win.is_empty() {
        return None;
    }

    let mut turn = 0.0f64;
    for &i in idx_win.iter() {
        turn += trades[i].volume;
    }

    let mut bvol: HashMap<i64, f64> = HashMap::new();
    let mut bto: HashMap<i64, f64> = HashMap::new();
    let mut avol: HashMap<i64, f64> = HashMap::new();
    let mut ato: HashMap<i64, f64> = HashMap::new();
    for &i in idx_win.iter() {
        let t = trades[i];
        if t.bid_order != 0 {
            *bvol.entry(t.bid_order).or_insert(0.0) += t.volume;
            *bto.entry(t.bid_order).or_insert(0.0) += t.turnover;
        }
        if t.ask_order != 0 {
            *avol.entry(t.ask_order).or_insert(0.0) += t.volume;
            *ato.entry(t.ask_order).or_insert(0.0) += t.turnover;
        }
    }

    // ── 每行：time_us, x_asks, y_asks(=U), x_bids, y_bids(=W)
    let mut rows: Vec<(i64, f64, f64, f64, f64)> = Vec::with_capacity(idx_win.len());
    for &i in idx_win.iter() {
        let lab = labels[i];
        if lab < 0 {
            continue;
        }
        let t = trades[i];
        if t.bid_order == 0 || t.ask_order == 0 {
            continue;
        }
        let (u, w) = match (bvol.get(&t.bid_order), avol.get(&t.ask_order)) {
            (Some(a), Some(b)) => (*a, *b),
            _ => continue,
        };
        let real_buy = bto.get(&t.bid_order).copied().unwrap_or(f64::NAN) / u;
        let real_sell = ato.get(&t.ask_order).copied().unwrap_or(f64::NAN) / w;

        let s = &mkt[lab as usize];
        let mut ap = [0.0f64; 11];
        let mut av = [0.0f64; 11];
        let mut bp = [0.0f64; 11];
        let mut bv = [0.0f64; 11];
        for k in 0..10 {
            ap[k] = s.ask_prc[k];
            av[k] = s.ask_vol[k];
            bp[k] = s.bid_prc[k];
            bv[k] = s.bid_vol[k];
        }
        ap[10] = s.ask_prc[9] + 0.01;
        av[10] = turn;
        bp[10] = s.bid_prc[9] - 0.01;
        bv[10] = turn;

        // asks：max(第 1 档卖价, 走出来的均价) - real_buy
        let x_asks = select_max(s.ask_prc[0], price2_levels(&ap, &av, u)) - real_buy;
        // bids：min(第 1 档买价, 走出来的均价，恰好为 0 的先换成 1e10) - real_sell
        // 注意 1e10 不能省成「直接用第 1 档买价」：select_min 里出现 (price+1e10) - 1e10
        // 这种大数相减，f64 会掉出约 2.3e-7 的舍入，旧代码就是这么算的，必须一致。
        let p2b = price2_levels(&bp, &bv, w);
        let p2b = if p2b == 0.0 { 1e10 } else { p2b };
        let x_bids = select_min(s.bid_prc[0], p2b) - real_sell;

        rows.push((t.time_us, x_asks, u, x_bids, w));
    }

    // ── 五个时段各两个相关系数
    let mut out: Vec<f32> = Vec::with_capacity(N_OUT);
    for (lo, hi) in SESSIONS.iter() {
        let mut xa: Vec<f64> = Vec::new();
        let mut ya: Vec<f64> = Vec::new();
        let mut xb: Vec<f64> = Vec::new();
        let mut yb: Vec<f64> = Vec::new();
        for r in rows.iter() {
            if !in_span(r.0, *lo, *hi) {
                continue;
            }
            xa.push(r.1);
            ya.push(r.2);
            xb.push(r.3);
            yb.push(r.4);
        }
        if xa.is_empty() {
            out.push(f32::NAN);
            out.push(f32::NAN);
        } else {
            out.push(pearson(&xa, &ya) as f32);
            out.push(pearson(&xb, &yb) as f32);
        }
    }
    debug_assert_eq!(out.len(), N_OUT);
    Some(out)
}

/// hm13 全是逐股因子，没有横截面版本。
pub fn compute_cross(_date: i64) -> Option<(Vec<String>, Vec<f32>)> {
    None
}
