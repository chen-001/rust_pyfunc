//! hm10「黑粉铁粉」（旧框架 design_whatever_out.TickDataBase）的 Rust 复现。
//!
//! 旧文件：/home/chenzongwei/pythoncode/成果hm10_黑粉铁粉/hm10_okay.py
//!   run_base 的 get_factor，参数 get_many=0 / read_full=0 / change_time=1
//!   → 一整天逐笔一张表（df）、一整天盘口一张表（df2），窗口由旧代码自己切。
//!
//! ════════════════════════════════════════════════════════════════════
//! 旧代码逐步对照（写代码前先抄在这里，方便复核）
//! ════════════════════════════════════════════════════════════════════
//! 1) df  = read_trade_data_single(with_retreat=0)  → 丢掉 flag==32 的撤单行
//!    df2 = read_market_data_single()               → 10 档盘口
//! 2) df1b = df[['exchtime','volume','flag']]
//!    df1b_66 = df1b[flag==66].resample('1T').volume.sum().replace(0,nan).dropna()
//!    df1b_83 同理。索引是「分钟起点」，只留分钟量不为 0 的分钟。
//! 3) df2 先切 [09:30:00, 14:57:00]，再 resample('1T').last().dropna() 得到 df2b
//!    （每分钟最后一行，整行没有空值的分钟才留下）。两个帧各自按行号打 label：
//!    df2 的 label 是「窗口内第几行」，df2b 的 label 是「第几个有效分钟」。
//! 4) df2_label = df2[['exchtime','label']] 按 exchtime 去重（留第一次出现）
//!    df2b_label = df2b[['exchtime','label']] 的 (分钟, label)
//! 5) df1b_66 = concat([df1b_66, df2b_label]).sort_index()
//!              .between_time('09:30','14:57') 然后 label 列 ffill
//!    → 每个分钟拿到「不超过它的最近一个 df2b 分钟」的 label（1m 分支用）
//! 6) df2 再看相邻两行的 ask_prc1 / bid_prc1 有没有变：
//!    df2 = df2[(ask_prc1.diff()==0) & (bid_prc1.diff()==0)]  ← 这就是 3s 那套盘口
//!    注意 diff 是在 09:30~14:57 这个切片上做的，所以第一行必然被去掉。
//! 7) df = concat([df, df2_label]).sort_values(['exchtime']) 然后 label 列 ffill
//!    → 每笔成交拿到「不超过它的最近一个快照」的 label（3s 分支用）
//!    ！！sort_values 是单键，走 pandas nargsort → DatetimeArray.argsort('quicksort')
//!      → numpy aquicksort_datetime，对相等的键**不稳定**。同一时刻的成交与快照谁在前
//!      决定了成交拿到哪个 label，所以下面 np_argsort_quicksort 是逐行复刻，不能换成
//!      sort_by_key。
//! 8) 一趟 one(df, df2, start, end) 里：
//!      df = df.between_time(start,end)；df2 = df2.between_time(start,end)
//!      df2 = df2.set_index('label')；df 按 flag 拆 66 / 83
//!      1m 模式则用 df1b_66 / df1b_83（先 between_time，再 set_index('label').dropna()）
//!    然后 8 组指标，每组 4 个相关系数（X ∈ {买前, 卖前, 买后, 卖后}）：
//!      ask_pri:         corr(ask_vol{k}.diff(), X) 对 k=1..10 取平均
//!      bid_pri:         同上用 bid_vol
//!      ask:             corr(diff(ask_vol).cumsum(axis=1){k}, X) 取平均
//!      bid:             同上用 bid_vol
//!      ask_slope:       corr(z(ask_vol5) - z(ask_vol1), X)，z 是窗口内标准化
//!      bid_slope:       同上用 bid_vol
//!      ask_slope_diff:  corr((z5-z1).diff(), X)
//!      bid_slope_diff:  同上用 bid_vol
//!    X 的「前 / 后」是 groupby('label').volume.sum() 的 shift(-1)：
//!    后 = 组内 label 升序的下一个**存在**的 label，不是下一个快照。
//!
//! names（names_single，共 320 个）：
//!   ['3s_','1m_'] × ['all_','open30_','morning_','afternoon_','close30_'] ×
//!   ['ask_pri_','bid_pri_','ask_','bid_','ask_slope_','bid_slope_','ask_slope_diff_','bid_slope_diff_']
//!   × ['mean_buy_before','mean_sell_before','mean_buy_after','mean_sell_after']
//!
//! ════════════════════════════════════════════════════════════════════
//! 为什么 names() 有 320 个、compute_stock 却返回 416 个值
//! ════════════════════════════════════════════════════════════════════
//! 旧代码最后的 return 是
//!   res_3s_all + res_3s_open30 + res_3s_morning + res_3s_afternoon + res_3s_close30
//! + res_1m_all + res_1m_open30 + res_1m_morning
//! + res_1m_all + res_1m_open30 + res_1m_morning   ← 这三段被重复加了一次
//! + res_1m_afternoon + res_1m_close30
//! 共 13 段 × 32 = 416 个值，但 names_single 只有 320 个。
//! 旧框架写 base 时是 `zip(names_single, res)`，只取前 320 个，于是
//!   位置 256..287（名字是 1m_afternoon_*）装的是 res_1m_all 的值，
//!   位置 288..319（名字是 1m_close30_*）装的是 res_1m_open30 的值。
//! 历史数据就是这么落库的，所以：
//!   · compute_stock 返回完整的 416 个值（与旧代码逐值对齐，verify_series 比的就是 416）；
//!   · names() 只声明 320 个名字（真实列集，names_save 都在这 320 个里）。
//!   · 流水线按 expected_result_length = len(names) 截断，正好复刻旧框架的 zip 行为。
//! 不能把 names() 写成 416 个：那会让 256..319 出现重名，colblk 落盘按名建列会打架。
//!
//! ════════════════════════════════════════════════════════════════════
//! 精度：不能用 f32 的读取器
//! ════════════════════════════════════════════════════════════════════
//! · 撤单过滤自己按数值筛 flag != 32（不靠 read_trade_fast_inner 的 with_retreat：
//!   库里那条判断按字节比 "32"，2025 年的明细写的是 "32.0"，过滤不生效）。
//! · 全程 f64 自己解析 CSV：这一系列是「盘口量的一阶/二阶差分」与成交量的相关系数，
//!   盘口量是小整数、差值更小，f32 的读取器会把量列压成 f32 再相减，
//!   相对误差落在 1e-5~1e-4 量级（验收阈值就是 1e-4）。hm013 已有同样的教训，
//!   这里逐笔与盘口都按 f64 解析（列索引与 fast_csv_reader 的常量一致）。

use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::Read;

// ── 逐笔 CSV 列索引（0 基；2018 年及以前末尾多一列 num，不影响前 15 列） ──
const TRD_COL_EXCHTIME: usize = 4;
const TRD_COL_VOLUME: usize = 8;
const TRD_COL_FLAG: usize = 10;

// ── 盘口 CSV 列索引（0 基） ──
// 表头：... ask_prc1(=21), ask_vol1(22), bid_prc1(23), bid_vol1(24), ask_prc2(25) ...
// 即每档 4 列，ask_prc1 在第 22 列（1 基）。2018 年文件末尾多一列 num，不影响前面的列。
const MKT_COL_EXCHTIME: usize = 4;
const MKT_ASK_PRC_BASE: usize = 21;

const SEC_US: i64 = 1_000_000;
const DAY_US: i64 = 86_400_000_000;
const MIN_US: i64 = 60 * SEC_US;
/// exchtime 是 epoch 微秒，旧代码 deal_data 里统一 +8 小时换成北京钟点
const BJS_OFFSET_US: i64 = 8 * 3600 * SEC_US;

const LD_0930: i64 = (9 * 3600 + 30 * 60) * SEC_US;
const LD_1000: i64 = 10 * 3600 * SEC_US;
const LD_1130: i64 = (11 * 3600 + 30 * 60) * SEC_US;
const LD_1300: i64 = 13 * 3600 * SEC_US;
const LD_1430: i64 = (14 * 3600 + 30 * 60) * SEC_US;
const LD_1457: i64 = (14 * 3600 + 57 * 60) * SEC_US;

/// 五个时段（真实钟点的微秒闭区间；旧代码用 between_time，两端都含）
const SESSIONS: [(i64, i64); 5] = [
    (LD_0930, LD_1457),
    (LD_0930, LD_1000),
    (LD_0930, LD_1130),
    (LD_1300, LD_1457),
    (LD_1430, LD_1457),
];
const SESSION_TAGS: [&str; 5] = ["all", "open30", "morning", "afternoon", "close30"];

/// 8 组指标 × 4 个 X = 每时段 32 个值
const GROUP_TAGS: [&str; 8] = [
    "ask_pri",
    "bid_pri",
    "ask",
    "bid",
    "ask_slope",
    "bid_slope",
    "ask_slope_diff",
    "bid_slope_diff",
];
const X_TAGS: [&str; 4] = [
    "mean_buy_before",
    "mean_sell_before",
    "mean_buy_after",
    "mean_sell_after",
];
const X_BUY_BEFORE: usize = 0;
const X_SELL_BEFORE: usize = 1;
const X_BUY_AFTER: usize = 2;
const X_SELL_AFTER: usize = 3;

const PER_SESSION: usize = 32;
/// names() 的长度（真实列集，与旧代码 names_single 对齐）
pub const N_NAMES: usize = 30 * PER_SESSION;
/// compute_stock 返回的长度（旧代码 return 出来的 13 段 × 32；见文件头说明）
pub const N_OUT: usize = 13 * PER_SESSION;

pub fn names() -> Vec<String> {
    let mut out = Vec::with_capacity(N_NAMES);
    for mode in ["3s", "1m"] {
        for s in SESSION_TAGS.iter() {
            for g in GROUP_TAGS.iter() {
                for x in X_TAGS.iter() {
                    out.push(format!("{}_{}_{}_{}", mode, s, g, x));
                }
            }
        }
    }
    out
}

#[inline]
fn day_us(t: i64) -> i64 {
    t.rem_euclid(DAY_US)
}

/// 钟点是否落在微秒闭区间内（between_time 精确到微秒，不能按整秒截断）
#[inline]
fn in_span(t: i64, lo: i64, hi: i64) -> bool {
    let d = day_us(t);
    d >= lo && d <= hi
}

/// 分钟起点（当日微秒）
#[inline]
fn minute_start(d: i64) -> i64 {
    (d / MIN_US) * MIN_US
}

// ============================================================================
// CSV 解析
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

fn level2_path(code: &str, date: i64, kind: &str) -> Option<std::path::PathBuf> {
    let filename = format!("{}_{}_{}.csv", code, date, kind);
    for root in crate::data_paths::level2_roots() {
        let p = root.join(date.to_string()).join(kind).join(&filename);
        if p.exists() {
            return Some(p);
        }
    }
    None
}

/// 把一行按逗号切成若干段，返回段数（缓冲区固定 64 段，够 62 列的盘口）
#[inline]
fn split_into<'a>(bytes: &'a [u8], buf: &mut [&'a [u8]; 64]) -> usize {
    let mut start = 0usize;
    let mut n = 0usize;
    for (i, &b) in bytes.iter().enumerate() {
        if b == b',' {
            if n < 64 {
                buf[n] = &bytes[start..i];
            }
            n += 1;
            start = i + 1;
        }
    }
    if n < 64 {
        buf[n] = &bytes[start..];
    }
    (n + 1).min(64)
}

/// 一条逐笔成交（只留计算要用的列）
#[derive(Clone, Copy, Default)]
struct TradeRow {
    t_us: i64,
    volume: f64,
    flag: i64,
}

/// 读 transaction CSV（f64）。行集与旧框架 pd.read_csv(usecols=[...]) 一致：
/// 表头之后每一行都算，exchtime==0 的行也保留（它排在 08:00，落在窗口外，
/// 但参与了 sort_values 的排序数组，保留才和旧代码的平局顺序一致）。
fn read_trade_f64(code: &str, date: i64) -> Option<Vec<TradeRow>> {
    let path = level2_path(code, date, "transaction")?;
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
        let mut f: [&[u8]; 64] = [&[][..]; 64];
        let nf = split_into(bytes, &mut f);
        if nf <= TRD_COL_FLAG {
            continue;
        }
        out.push(TradeRow {
            t_us: parse_i64_at(f[TRD_COL_EXCHTIME]) + BJS_OFFSET_US,
            volume: parse_f64_at(f[TRD_COL_VOLUME]),
            flag: parse_i64_at(f[TRD_COL_FLAG]),
        });
    }
    Some(out)
}

/// 一条盘口快照（只留计算要用的档位）
#[derive(Clone, Copy, Default)]
struct SnapRow {
    t_us: i64,
    ask_prc1: f64,
    bid_prc1: f64,
    ask_vol: [f64; 10],
    bid_vol: [f64; 10],
}

struct MarketData {
    /// 窗口 [09:30:00, 14:57:00] 内的全部快照，按文件顺序，label = 下标
    window: Vec<SnapRow>,
    /// 窗口内每分钟一行：`resample('1T').last().dropna()` 的结果，按分钟升序
    minutes: Vec<(i64, SnapRow)>,
}

/// 读 market_data CSV（f64），只保留窗口内的行。
///
/// 为什么先判窗口再聚合分钟：旧代码是 `between_time` 之后才 `resample`，
/// 14:57:00.5 这种行不属于窗口、也就不该出现在 14:57 这一分钟的 last() 里。
/// 分钟聚合只吃窗口内的行。
fn read_market_f64(code: &str, date: i64) -> Option<MarketData> {
    let path = level2_path(code, date, "market_data")?;
    let mut content = String::new();
    File::open(&path).ok()?.read_to_string(&mut content).ok()?;

    let mut lines = content.lines();
    let header = lines.next()?;
    let ncol = header.split(',').count();

    let mut window: Vec<SnapRow> = Vec::with_capacity(content.len() / 300 + 1);
    // 分钟 -> (该分钟各列的最后一个非空值, 出现过的非空列掩码)
    let mut acc: HashMap<i64, (SnapRow, u64)> = HashMap::new();

    for line in lines {
        if line.is_empty() {
            continue;
        }
        let bytes = line.as_bytes();
        let mut f: [&[u8]; 64] = [&[][..]; 64];
        let nf = split_into(bytes, &mut f);
        if nf <= MKT_ASK_PRC_BASE + 39 {
            continue;
        }
        let ex = parse_i64_at(f[MKT_COL_EXCHTIME]);
        if ex == 0 {
            // exchtime==0 换出来是 1970-01-01 08:00，落在窗口外；旧代码里也在窗口外
            continue;
        }
        let t_us = ex + BJS_OFFSET_US;
        let d = day_us(t_us);
        if d < LD_0930 || d > LD_1457 {
            continue;
        }
        window.push(snap_from_fields(&f, t_us));

        // resample('1T').last()：逐列取「本分钟最后一个非空值」
        let minute = minute_start(d);
        let e = acc
            .entry(minute)
            .or_insert_with(|| (SnapRow { t_us: minute, ..Default::default() }, 0u64));
        if !f[MKT_ASK_PRC_BASE].is_empty() {
            e.0.ask_prc1 = parse_f64_at(f[MKT_ASK_PRC_BASE]);
        }
        if !f[MKT_ASK_PRC_BASE + 2].is_empty() {
            e.0.bid_prc1 = parse_f64_at(f[MKT_ASK_PRC_BASE + 2]);
        }
        for k in 0..10 {
            let base = MKT_ASK_PRC_BASE + k * 4;
            if !f[base + 1].is_empty() {
                e.0.ask_vol[k] = parse_f64_at(f[base + 1]);
            }
            if !f[base + 3].is_empty() {
                e.0.bid_vol[k] = parse_f64_at(f[base + 3]);
            }
        }
        let mut mask = 0u64;
        for c in 0..nf.min(64) {
            if !f[c].is_empty() {
                mask |= 1u64 << c;
            }
        }
        e.1 |= mask;
    }

    // dropna()：整行（全部列）在分钟内一次都没出现过值的分钟要丢掉
    let full_mask: u64 = if ncol >= 64 { u64::MAX } else { (1u64 << ncol) - 1 };
    let mut minutes: Vec<(i64, SnapRow)> = acc
        .into_iter()
        .filter(|(_, (_, mask))| *mask == full_mask)
        .map(|(m, (row, _))| (m, row))
        .collect();
    minutes.sort_by_key(|&(m, _)| m);

    Some(MarketData { window, minutes })
}

fn snap_from_fields(f: &[&[u8]], t_us: i64) -> SnapRow {
    let mut r = SnapRow { t_us, ..Default::default() };
    r.ask_prc1 = parse_f64_at(f[MKT_ASK_PRC_BASE]);
    r.bid_prc1 = parse_f64_at(f[MKT_ASK_PRC_BASE + 2]);
    for k in 0..10 {
        let base = MKT_ASK_PRC_BASE + k * 4;
        r.ask_vol[k] = parse_f64_at(f[base + 1]);
        r.bid_vol[k] = parse_f64_at(f[base + 3]);
    }
    r
}

// ============================================================================
// numpy aquicksort_datetime 的逐行复刻（与 hm013 一致）
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
///
/// 平局顺序必须与它一致：pandas 单键 sort_values 走 DatetimeArray.argsort('quicksort')，
/// 落到这个函数；同一时刻的成交与快照谁在前，决定了那笔成交拿到哪个快照的 label。
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
// 统计（pandas 语义）
// ============================================================================

/// pandas DataFrame.corr() 的 Pearson：逐行复刻 pandas/_libs/algos.pyx 的 nancorr。
///
/// 成对完整观测（任一为 NaN 就丢掉这一对）；用 Welford 在线算法。
/// 任一侧平方和为 0（某列恒定）或 divisor 为 0 时返回 NaN，与旧代码一致。
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

/// Series.mean()：跳过 NaN，全 NaN 返回 NaN
fn mean_skipna(v: &[f64]) -> f64 {
    let mut n = 0usize;
    let mut s = 0.0f64;
    for &x in v.iter() {
        if x.is_finite() {
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

/// Series.std()：ddof=1，跳过 NaN，样本 < 2 返回 NaN
fn std_skipna(v: &[f64]) -> f64 {
    let mut n = 0usize;
    let mut s = 0.0f64;
    for &x in v.iter() {
        if x.is_finite() {
            n += 1;
            s += x;
        }
    }
    if n < 2 {
        return f64::NAN;
    }
    let m = s / n as f64;
    let mut ss = 0.0f64;
    for &x in v.iter() {
        if x.is_finite() {
            ss += (x - m) * (x - m);
        }
    }
    (ss / (n - 1) as f64).sqrt()
}

// ============================================================================
// label 组装
// ============================================================================

/// concat([逐笔, df2_label]).sort_values(['exchtime']) 之后 ffill label。
/// 返回每笔成交（原始顺序）拿到的快照 label，-1 表示 NaN。
fn assign_trade_labels(trades: &[TradeRow], df2_label: &[(i64, i64)]) -> Vec<i64> {
    let n_t = trades.len();
    let mut times: Vec<i64> = Vec::with_capacity(n_t + df2_label.len());
    times.extend(trades.iter().map(|r| r.t_us));
    times.extend(df2_label.iter().map(|&(t, _)| t));
    let perm = np_argsort_quicksort(&times);

    let mut labels = vec![-1i64; n_t];
    let mut cur: i64 = -1;
    for &i in perm.iter() {
        if i >= n_t {
            cur = df2_label[i - n_t].1;
        } else {
            labels[i] = cur;
        }
    }
    labels
}

/// 一天里每分钟的 66 / 83 成交量（分钟起点 -> 量），按分钟升序返回，和旧代码
/// `resample('1T').volume.sum().replace(0,nan).dropna()` 等价。
/// 注意：用的是整天（含窗口外的成交），因为分钟量的口径是整个分钟；
/// 窗口筛选在取时段时再做。
fn trade_minute_sums(trades: &[TradeRow]) -> (Vec<(i64, f64)>, Vec<(i64, f64)>) {
    let mut m66: HashMap<i64, f64> = HashMap::new();
    let mut m83: HashMap<i64, f64> = HashMap::new();
    for t in trades.iter() {
        let m = minute_start(day_us(t.t_us));
        let slot = match t.flag {
            66 => &mut m66,
            83 => &mut m83,
            _ => continue,
        };
        *slot.entry(m).or_insert(0.0) += t.volume;
    }
    let sort_drop = |m: HashMap<i64, f64>| -> Vec<(i64, f64)> {
        let mut v: Vec<(i64, f64)> = m.into_iter().filter(|(_, x)| *x != 0.0).collect();
        v.sort_by_key(|&(k, _)| k);
        v
    };
    (sort_drop(m66), sort_drop(m83))
}

/// 分钟量那条序列（df1b_66 / df1b_83）拿到 label。
///
/// ！！这里必须逐行复刻 pd.concat + sort_index 的行为：
///   pd.concat([df1b_66, df2b_label]) 是**按行堆叠**、不按 index 对齐（两个帧的分钟
///   一样时会出现两行：一行有 volume、一行有 label）。sort_index 是单键 quicksort
///   （numpy aquicksort_datetime），同一分钟里 volume 行和 label 行谁在前由它决定，
///   而谁在前决定了 ffill 后那一分钟拿到的是「本分钟的 label」还是「上一分钟的 label」。
///   实测 09:31 的成交量行排在 09:31 的 label 行前面，于是它拿到的是 09:30 的 label。
fn assign_minute_labels(minutes: &[i64], df2b_label: &[(i64, i64)]) -> Vec<i64> {
    let n_v = minutes.len();
    let mut times: Vec<i64> = Vec::with_capacity(n_v + df2b_label.len());
    times.extend(minutes.iter().copied());
    times.extend(df2b_label.iter().map(|&(t, _)| t));
    let perm = np_argsort_quicksort(&times);

    let mut labels = vec![-1i64; n_v];
    let mut cur: i64 = -1;
    for &i in perm.iter() {
        if i >= n_v {
            cur = df2b_label[i - n_v].1;
        } else {
            labels[i] = cur;
        }
    }
    labels
}

/// `groupby('label').volume.sum()` 与它的 `shift(-1)`
struct LabelSums {
    labels: Vec<i64>,
    before: Vec<f64>,
    after: Vec<f64>,
}

impl LabelSums {
    fn build(m: &HashMap<i64, f64>) -> LabelSums {
        // groupby 默认按 label 升序；这里显式排序，保证结果确定
        let mut labels: Vec<i64> = m.keys().copied().collect();
        labels.sort_unstable();
        let before: Vec<f64> = labels.iter().map(|l| m[l]).collect();
        let mut after = Vec::with_capacity(labels.len());
        for i in 0..labels.len() {
            after.push(if i + 1 < labels.len() {
                before[i + 1]
            } else {
                f64::NAN
            });
        }
        LabelSums { labels, before, after }
    }

    #[inline]
    fn pos(&self, l: i64) -> Option<usize> {
        self.labels.binary_search(&l).ok()
    }
}

/// 3s 分支：窗口内成交按 label 汇总（X ∈ {买,卖} × {前,后}）
struct TradeAgg {
    buy: LabelSums,
    sell: LabelSums,
    n66: usize,
    n83: usize,
}

fn trade_agg_3s(trades: &[TradeRow], labels: &[i64], lo: i64, hi: i64) -> TradeAgg {
    let mut buy: HashMap<i64, f64> = HashMap::new();
    let mut sell: HashMap<i64, f64> = HashMap::new();
    let mut n66 = 0usize;
    let mut n83 = 0usize;
    for (i, t) in trades.iter().enumerate() {
        if !in_span(t.t_us, lo, hi) {
            continue;
        }
        match t.flag {
            66 => n66 += 1,
            83 => n83 += 1,
            _ => continue,
        }
        let l = labels[i];
        if l < 0 {
            continue;
        }
        let slot = if t.flag == 66 { &mut buy } else { &mut sell };
        *slot.entry(l).or_insert(0.0) += t.volume;
    }
    TradeAgg { buy: LabelSums::build(&buy), sell: LabelSums::build(&sell), n66, n83 }
}

/// 1m 分支：分钟量按 label 汇总。分钟量序列与 label 已经在外面用
/// assign_minute_labels 对齐好了，这里只取时段、丢掉 label 为 NaN 的分钟
/// （对应旧代码的 between_time + set_index('label').dropna()）。
/// 分钟本来就按升序，浮点累加顺序确定。
fn trade_agg_1m(
    m66: &[(i64, f64)],
    lab66: &[i64],
    m83: &[(i64, f64)],
    lab83: &[i64],
    lo: i64,
    hi: i64,
) -> TradeAgg {
    let mut buy: HashMap<i64, f64> = HashMap::new();
    let mut sell: HashMap<i64, f64> = HashMap::new();
    let n66 = accumulate_1m(&mut buy, m66, lab66, lo, hi);
    let n83 = accumulate_1m(&mut sell, m83, lab83, lo, hi);
    TradeAgg { buy: LabelSums::build(&buy), sell: LabelSums::build(&sell), n66, n83 }
}

/// 返回落进时段的分钟数（= 旧代码 dropna 之后的行数）
fn accumulate_1m(
    dst: &mut HashMap<i64, f64>,
    minutes: &[(i64, f64)],
    labels: &[i64],
    lo: i64,
    hi: i64,
) -> usize {
    let mut cnt = 0usize;
    for (i, &(m, v)) in minutes.iter().enumerate() {
        if m < lo || m > hi || labels[i] < 0 {
            continue;
        }
        cnt += 1;
        *dst.entry(labels[i]).or_insert(0.0) += v;
    }
    cnt
}

// ============================================================================
// 快照帧与时段切片
// ============================================================================

/// 一套「快照帧」：labels + 每档量（按时间升序）
struct Frame {
    times: Vec<i64>,
    labels: Vec<i64>,
    ask: Vec<[f64; 10]>,
    bid: Vec<[f64; 10]>,
}

impl Frame {
    fn push(&mut self, t: i64, l: i64, r: &SnapRow) {
        self.times.push(t);
        self.labels.push(l);
        self.ask.push(r.ask_vol);
        self.bid.push(r.bid_vol);
    }

    /// 对应 `df2.between_time(start, end).set_index('label')`
    fn slice(&self, lo: i64, hi: i64) -> (Vec<i64>, Vec<[f64; 10]>, Vec<[f64; 10]>) {
        let mut labels = Vec::new();
        let mut ask = Vec::new();
        let mut bid = Vec::new();
        for i in 0..self.times.len() {
            if !in_span(self.times[i], lo, hi) {
                continue;
            }
            labels.push(self.labels[i]);
            ask.push(self.ask[i]);
            bid.push(self.bid[i]);
        }
        (labels, ask, bid)
    }
}

/// df2 的「价差未变」子集：ask_prc1 与 bid_prc1 都和前一行相等。
/// diff 在窗口切片上做，所以第一行一定被剔除。
fn price_stable_frame(window: &[SnapRow]) -> Frame {
    let mut f = Frame { times: Vec::new(), labels: Vec::new(), ask: Vec::new(), bid: Vec::new() };
    for i in 1..window.len() {
        if window[i].ask_prc1 == window[i - 1].ask_prc1
            && window[i].bid_prc1 == window[i - 1].bid_prc1
        {
            f.push(window[i].t_us, i as i64, &window[i]);
        }
    }
    f
}

/// df2b：每分钟一行，label = 该分钟在 df2b 里的行号
fn minute_frame(minutes: &[(i64, SnapRow)]) -> Frame {
    let mut f = Frame { times: Vec::new(), labels: Vec::new(), ask: Vec::new(), bid: Vec::new() };
    for (i, (m, r)) in minutes.iter().enumerate() {
        f.push(*m, i as i64, r);
    }
    f
}

// ============================================================================
// 一段 one(...) 的 32 个值
// ============================================================================

/// pandas `DataFrame.cumsum(axis=1)`：NaN 当 0 参与累加，但输出位置上仍是 NaN
fn cumsum_rows(src: &[[f64; 10]]) -> Vec<[f64; 10]> {
    let mut out = vec![[f64::NAN; 10]; src.len()];
    for i in 0..src.len() {
        let mut s = 0.0f64;
        for k in 0..10 {
            if src[i][k].is_nan() {
                out[i][k] = f64::NAN;
            } else {
                s += src[i][k];
                out[i][k] = s;
            }
        }
    }
    out
}

/// 10 列各算一次 corr(列, X) 再 np.mean；任一为 NaN → NaN
fn mean_col_corr(cols: &[[f64; 10]], xs: &[f64]) -> f64 {
    let n = xs.len();
    let mut acc = 0.0f64;
    let mut col: Vec<f64> = vec![f64::NAN; n];
    for k in 0..10 {
        for i in 0..n {
            col[i] = cols[i][k];
        }
        acc += pearson(&col, xs);
    }
    acc / 10.0
}

/// 单列的标准化（(x - mean) / std，ddof=1，跳 NaN）后的「第 5 档 - 第 1 档」
fn std_slope(vol: &[[f64; 10]]) -> Vec<f64> {
    let n = vol.len();
    let c1: Vec<f64> = (0..n).map(|i| vol[i][0]).collect();
    let c5: Vec<f64> = (0..n).map(|i| vol[i][4]).collect();
    let (m1, s1) = (mean_skipna(&c1), std_skipna(&c1));
    let (m5, s5) = (mean_skipna(&c5), std_skipna(&c5));
    (0..n).map(|i| (c5[i] - m5) / s5 - (c1[i] - m1) / s1).collect()
}

/// 行方向的一阶差分（第一行 NaN）
fn row_diff(v: &[f64]) -> Vec<f64> {
    let mut out = vec![f64::NAN; v.len()];
    for i in 1..v.len() {
        out[i] = v[i] - v[i - 1];
    }
    out
}

// ============================================================================
// 逐股入口
// ============================================================================

/// 一段 one(...)：snap 是时段内的快照，agg 是同时段成交按 label 的汇总。
/// 返回 32 个值，顺序与 names 里 8 组 × 4 个 X 一致。
fn session_metrics(
    snap_labels: &[i64],
    ask: &[[f64; 10]],
    bid: &[[f64; 10]],
    agg: &TradeAgg,
) -> [f64; 32] {
    let mut out = [f64::NAN; PER_SESSION];
    let n = snap_labels.len();
    // 旧代码的两个前置判断：df_66 / df_83 都非空，且时段内快照非空
    if n == 0 || agg.n66 == 0 || agg.n83 == 0 {
        return out;
    }

    // X 四列：买前 / 卖前 / 买后 / 卖后
    let mut xv: [Vec<f64>; 4] = [
        vec![f64::NAN; n],
        vec![f64::NAN; n],
        vec![f64::NAN; n],
        vec![f64::NAN; n],
    ];
    for i in 0..n {
        let l = snap_labels[i];
        if let Some(p) = agg.buy.pos(l) {
            xv[X_BUY_BEFORE][i] = agg.buy.before[p];
            xv[X_BUY_AFTER][i] = agg.buy.after[p];
        }
        if let Some(p) = agg.sell.pos(l) {
            xv[X_SELL_BEFORE][i] = agg.sell.before[p];
            xv[X_SELL_AFTER][i] = agg.sell.after[p];
        }
    }

    // 一阶差分（第一行 NaN）
    let mut a_pri = vec![[f64::NAN; 10]; n];
    let mut b_pri = vec![[f64::NAN; 10]; n];
    for i in 1..n {
        for k in 0..10 {
            a_pri[i][k] = ask[i][k] - ask[i - 1][k];
            b_pri[i][k] = bid[i][k] - bid[i - 1][k];
        }
    }
    let a_cum = cumsum_rows(&a_pri);
    let b_cum = cumsum_rows(&b_pri);
    let a_slope = std_slope(ask);
    let b_slope = std_slope(bid);
    let a_slope_d = row_diff(&a_slope);
    let b_slope_d = row_diff(&b_slope);

    // 8 组 × 4 个 X
    let mut g = 0usize;
    for cols in [&a_pri, &b_pri, &a_cum, &b_cum] {
        for x in 0..4 {
            out[g * 4 + x] = mean_col_corr(cols, &xv[x]);
        }
        g += 1;
    }
    for v in [&a_slope, &b_slope, &a_slope_d, &b_slope_d] {
        for x in 0..4 {
            out[g * 4 + x] = pearson(v, &xv[x]);
        }
        g += 1;
    }
    debug_assert_eq!(g, 8);
    out
}

/// 逐股的一天：算不出来返回 None（上游补 NaN，保证因子个数恒定）。
pub fn compute_stock(code: &str, date: i64) -> Option<Vec<f32>> {
    // 先全读进来，再自己按数值过滤 flag==32（"32.0" 也要能过滤掉）
    let trades_all = read_trade_f64(code, date)?;
    let trades: Vec<TradeRow> = trades_all.into_iter().filter(|r| r.flag != 32).collect();

    let md = read_market_f64(code, date)?;
    if md.window.is_empty() {
        // 旧代码 `if df2.shape[0]>0:` 不成立 → get_factor 返回 None
        return None;
    }

    // 3s 的 label（逐笔 ← 快照）
    let mut seen: HashSet<i64> = HashSet::with_capacity(md.window.len());
    let mut df2_label: Vec<(i64, i64)> = Vec::new();
    for (i, r) in md.window.iter().enumerate() {
        if seen.insert(r.t_us) {
            df2_label.push((r.t_us, i as i64));
        }
    }
    let labels = assign_trade_labels(&trades, &df2_label);

    // 1m 的两套索引：分钟量序列 + 它从 df2b 拿到的 label
    let (m66, m83) = trade_minute_sums(&trades);
    let df2b_label: Vec<(i64, i64)> = md
        .minutes
        .iter()
        .enumerate()
        .map(|(i, &(m, _))| (m, i as i64))
        .collect();
    let min66: Vec<i64> = m66.iter().map(|&(m, _)| m).collect();
    let min83: Vec<i64> = m83.iter().map(|&(m, _)| m).collect();
    let lab66 = assign_minute_labels(&min66, &df2b_label);
    let lab83 = assign_minute_labels(&min83, &df2b_label);

    let frame_f = price_stable_frame(&md.window);
    let frame_b = minute_frame(&md.minutes);

    let mut out: Vec<f32> = Vec::with_capacity(N_OUT);

    // ── 3s：五个时段
    for (lo, hi) in SESSIONS.iter() {
        let agg = trade_agg_3s(&trades, &labels, *lo, *hi);
        let (sl, sa, sb) = frame_f.slice(*lo, *hi);
        let vals = session_metrics(&sl, &sa, &sb, &agg);
        out.extend(vals.iter().map(|&v| v as f32));
    }

    // ── 1m：all / open30 / morning（对价差未变的那套快照）
    let mut first_three: Vec<[f64; 32]> = Vec::with_capacity(3);
    for (lo, hi) in SESSIONS[0..3].iter() {
        let agg = trade_agg_1m(&m66, &lab66, &m83, &lab83, *lo, *hi);
        let (sl, sa, sb) = frame_f.slice(*lo, *hi);
        first_three.push(session_metrics(&sl, &sa, &sb, &agg));
    }
    // 旧代码把这三段又算了一遍（结果完全相同），照旧输出两轮，见文件头说明
    for v in first_three.iter() {
        out.extend(v.iter().map(|&x| x as f32));
    }
    for v in first_three.iter() {
        out.extend(v.iter().map(|&x| x as f32));
    }

    // ── 1m：afternoon / close30（这两段旧代码传的是 df2b）
    for (lo, hi) in SESSIONS[3..5].iter() {
        let agg = trade_agg_1m(&m66, &lab66, &m83, &lab83, *lo, *hi);
        let (sl, sa, sb) = frame_b.slice(*lo, *hi);
        let vals = session_metrics(&sl, &sa, &sb, &agg);
        out.extend(vals.iter().map(|&v| v as f32));
    }

    debug_assert_eq!(out.len(), N_OUT);
    Some(out)
}

/// hm10 全是逐股因子，没有横截面版本。
pub fn compute_cross(_date: i64) -> Option<(Vec<String>, Vec<f32>)> {
    None
}
