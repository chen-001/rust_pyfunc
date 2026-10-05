//! hm55「成交失衡与盘口失衡」的 Rust 复现。
//!
//! 旧文件：/home/chenzongwei/pythoncode/成果hm55_成交失衡与盘口失衡/hm55_okay.py
//!   run_base 里的 Go.get_factor，参数 get_many=0 / read_full=0 / change_time=1，
//!   也就是「一整天逐笔 + 一整天盘口」各自一张表，窗口由旧代码自己切。
//!
//! ════════════════════════════════════════════════════════════════════
//! 一、旧代码逐步对照（写代码前先抄在这里，方便复核）
//! ════════════════════════════════════════════════════════════════════
//! 1) trade  = read_trade_data_single(with_retreat=1)
//!      ！！这里是 with_retreat=**1**，也就是**不做任何撤单过滤**，flag==32 的行照样留着。
//!      （规范里那条「自己 filter flag != 32」是给 with_retreat=0 的系列用的，本系列不适用。）
//!    market = read_market_data_single()（10 档盘口 61 列 + code）
//! 2) 两个窗口，各切一次逐笔与盘口，都是 between_time 闭区间、精确到微秒：
//!      open30  = [09:30:00, 10:00:00]
//!      close30 = [14:30:00, 14:57:00]
//! 3) single(trade, market) 把窗口内的逐笔与盘口拼起来，算 76 个相关系数（再取绝对值 → 152 个）。
//!    res = single(open30) + single(close30)，共 304 个值，顺序就是 names()。
//!
//! ════════════════════════════════════════════════════════════════════
//! 二、single(trade, market) 的每一步
//! ════════════════════════════════════════════════════════════════════
//! · 时间：trade.exchtime / market.exchtime 先 `to_numpy(dtype=float)`（datetime64[ns] → float64
//!   纳秒），再各自减去**本窗口内自己的最小值**、除以 1e9，得到「相对秒」。
//!   两边的基准不同（逐笔用逐笔窗口的第一笔，盘口用盘口窗口的第一个快照），所以拼起来排序时
//!   不是真实的先后，而是两个相对坐标的混排 —— 这一点必须照抄。
//! · 盘口 12 列：want_diff_1/5/10 与各自的 abs、volume（相邻快照累计量之差 /100）、last_prc。
//!   注意 want_diff_5 / want_diff_10 的分子分母在 pandas 里是**一路从左往右加**的，
//!   不是「先各自求和再相减」，f64 下这两种写法可以差 1 ulp，所以下面按同样的顺序写。
//! · `pd.concat([trade, market]).sort_values(by=['exchtime'])`：单键排序走 pandas nargsort
//!   → `np.argsort(kind='quicksort')`。numpy 2.x 在支持 AVX512 的机器上对 float64 走
//!   x86-simd-sort 的 argsort（不是 npysort 的 introsort），**并列顺序不稳定**，
//!   而本系列并列极多（逐笔只有 7613 个不同的 exchtime / 19212 行），并列里谁在前
//!   直接决定每笔成交落到哪个快照区间，实测换稳定排序 304 个值里有 298 个超差。
//!   所以这里把 exchtime 映射成保序的 int64 名次码，再走 hm039.rs 已逐行复刻并与
//!   numpy 2.3.5 对拍通过的 xss_argsort_i64（见下方第三段）。
//! · market_label：盘口行按窗口内行号 0..M-1 编号，排序后 ffill，逐笔就拿到「排在它前面的
//!   最后一个快照」的编号；排在第一个快照之前的逐笔拿到 NaN，groupby 会丢掉它们。
//! · dropna(subset=['flag']) 只留下逐笔行；trade.volume /= 100。
//! · 按 market_label 分组，每组算 4 个 DTW（fast_dtw_distance 的 radius=None 版本，纯 Rust 抄一份，
//!   worker 链路不能调 #[pyfunction]）：
//!     dtw_act_vol  = DTW(flag==66 的 volume,  flag==83 的 volume)
//!     dtw_act_dura = DTW(flag==66 的相对秒,   flag==83 的相对秒)
//!     dtw_back_vol = DTW(ask_order==0 的 volume, bid_order==0 的 volume)
//!     dtw_back_dura= DTW(ask_order==0 的相对秒,  bid_order==0 的相对秒)
//!   任一侧序列为空就取 NaN。注意 flag==32（撤单）的行也参与 ask_order==0 / bid_order==0 的判断。
//! · twins_f = concat([组结果, 盘口 12 列], axis=1)（按 label 对齐）；
//!   twins_b 是同一张表但组结果整体下移一格（trade_b.index = trade.index + 1），
//!   也就是「第 L 个快照 对 第 L-1 组逐笔」。两张表各取 pandas 的成对完整观测 Pearson。
//! · 76 个相关系数 = 12 列的严格上三角（第 0 行 11 个、第 1 行 10 个、第 2 行 9 个、第 3 行 8 个）
//!   先在 f 上取一遍、再在 b 上取一遍；再整体取绝对值接在后面。
//!
//! ════════════════════════════════════════════════════════════════════
//! 三、排序并列顺序（本系列的关键）
//! ════════════════════════════════════════════════════════════════════
//! 做法与 hm066.rs 完全一致：把 f64 的 exchtime 相对秒映射成保序的 int64 名次码
//! （同值同码、不同值不同码），再调 xss_argsort_i64。xss 的每一步只做比较与 min/max，
//! 名次码上的判定序列与原值完全一致，所以输出置换与原值上的 argsort 完全相同。
//! pandas 的 nargsort 还会把 NaN 整体挪到最后，pandas_argsort_asc 里照抄。
//!
//! ════════════════════════════════════════════════════════════════════
//! 四、时间与精度
//! ════════════════════════════════════════════════════════════════════
//! · 读出来的 exchtime 是 f64 微秒，旧框架 `pd.to_timedelta(exchtime/1e6, unit='s')`
//!   把它变成 ns，再用 `+ Timestamp('1970-01-01 08:00:00')` 加 8 小时。
//!   pandas 这一步是 `<int64_t>(x * 1e9)`（x = exchtime/1e6 的 f64，**截断**），
//!   实测它与「先把 f64 乘 1e9 再截断」差最多 128ns，与「精确乘 1e9 再截断」差 ≤1ns。
//!   这里用精确乘积（u128 整数运算，见 mul_1e9_trunc），误差 1ns 即 1e-9 秒，对下游可忽略。
//! · 窗口判定用 pandas 的微秒口径（`_get_time_micros`）：当日微秒 = (ns/1000) % 86400e6，
//!   闭区间两端都含。
//! · 旧代码里 304 个值全是相关系数，量级 O(1)，f32 输出足够（对照阈值 1e-4）。
//!
//! ════════════════════════════════════════════════════════════════════
//! 五、重名（304 个位置名里有 48 个重名，必须处理）
//! ════════════════════════════════════════════════════════════════════
//! 旧 names_single 有 304 项但只有 256 个不同的名字：`want_diff_K_abs` 这一列本身参与相关，
//! 得到的名字 `..._corr_want_diff_K_abs` 与「`..._corr_want_diff_K` 取绝对值」的名字**完全一样**。
//! 每个窗口 24 对，两个窗口共 48 对，每对恰好 2 次。
//!
//! 旧框架 cal_factor_one_day 里是
//!     pd.DataFrame({**{'code':..,'date':..}, **{k: v for k, v in zip(names_single, res)}})
//! 字典推导式遇重名是**后者覆盖前者**，所以旧库里那个名字下的值是**后出现的那一次**
//! （也就是 abs 半区、|corr(dtw, want_diff_K)|），不是 base 半区那个。
//! 实测 000001@20250303：`f_dtw_back_vol_corr_want_diff_1_abs_close30` 旧值 0.057939，
//! base 半区 -0.029781、abs 半区 0.057939 —— 确实是后者。
//! names_save 里有 4 个正是这种重名（f/b 各两个），值必须取后出现的那一次。
//!
//! 列式 store 不允许重名，所以这里把**早出现的那一次**加 `_dup2` 后缀
//! （值不变、位置不变、个数不变），把不带后缀的名字留给**最后一次出现**，
//! 这样 names_save 里的名字拿到的就是旧库那个值。48 个 `_dup2` 名字都不在 names_save 里。

use std::collections::HashMap;
use std::fs::File;
use std::io::Read;

// ── 逐笔 CSV 列号（0 基，与 fast_csv_reader 的常量一致）
const TRD_COL_EXCHTIME: usize = 4;
const TRD_COL_VOLUME: usize = 8;
const TRD_COL_FLAG: usize = 10;
const TRD_COL_ASK_ORDER: usize = 13;
const TRD_COL_BID_ORDER: usize = 14;
const TRD_NCOL: usize = 15;

// ── 盘口 CSV 列号（0 基）：ask_vol1=22, bid_vol1=24, ask_vol2=26, ...
const MKT_COL_EXCHTIME: usize = 4;
const MKT_COL_LAST_PRC: usize = 6;
const MKT_COL_VOLUME: usize = 14;
const MKT_COL_ASKVOL1: usize = 22;
const MKT_NCOL: usize = 61;

const DAY_US: i64 = 86_400_000_000;
const NS8H: i64 = 28_800_000_000_000;

/// 两个窗口（当日微秒，闭区间）
const OPEN30: (i64, i64) = ((9 * 3600 + 30 * 60) * 1_000_000, 10 * 3600 * 1_000_000);
const CLOSE30: (i64, i64) = (
    (14 * 3600 + 30 * 60) * 1_000_000,
    (14 * 3600 + 57 * 60) * 1_000_000,
);

/// 盘口 8 个派生列 + 4 个 DTW 列，顺序即相关系数矩阵的行列顺序
const COLS: [&str; 12] = [
    "dtw_act_vol",
    "dtw_act_dura",
    "dtw_back_vol",
    "dtw_back_dura",
    "want_diff_1",
    "want_diff_5",
    "want_diff_10",
    "want_diff_1_abs",
    "want_diff_5_abs",
    "want_diff_10_abs",
    "volume",
    "last_prc",
];

const PER_BLOCK: usize = 76; // f 38 + b 38
const PER_WINDOW: usize = 152; // 76 原值 + 76 绝对值
pub const N_OUT: usize = PER_WINDOW * 2;

pub fn names() -> Vec<String> {
    let mut base: Vec<String> = Vec::with_capacity(PER_BLOCK);
    for tag in ["f", "b"] {
        for i in 0..4 {
            for j in (i + 1)..12 {
                base.push(format!("{}_{}_corr_{}", tag, COLS[i], COLS[j]));
            }
        }
    }
    debug_assert_eq!(base.len(), PER_BLOCK);
    let mut per_window: Vec<String> = base.clone();
    per_window.extend(base.iter().map(|n| format!("{n}_abs")));
    dedup_keep_last(&mut per_window);
    let mut out: Vec<String> = Vec::with_capacity(N_OUT);
    out.extend(per_window.iter().map(|n| format!("{n}_open30")));
    out.extend(per_window.iter().map(|n| format!("{n}_close30")));
    out
}

/// 重名处理（见文件头第五节）：不带后缀的名字留给**最后一次出现**的位置
/// （旧框架的字典推导式是后者覆盖前者），早出现的位置加 `_dup2`。
/// 位置与个数都不变，只改名字。
fn dedup_keep_last(names: &mut [String]) {
    let mut last: HashMap<String, usize> = HashMap::with_capacity(names.len());
    for (i, n) in names.iter().enumerate() {
        last.insert(n.clone(), i);
    }
    let mut seen: HashMap<String, usize> = HashMap::new();
    for i in 0..names.len() {
        let n = names[i].clone();
        if last[&n] != i {
            let c = seen.entry(n.clone()).or_insert(1);
            *c += 1;
            names[i] = format!("{n}_dup{c}");
        }
    }
}

// ============================================================================
// 解析小工具
// ============================================================================

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

/// 拆一行 CSV 的字段（列数不足的按空串）
fn split_fields<'a>(line: &'a [u8], f: &mut [&'a [u8]]) {
    let mut start = 0usize;
    let mut col = 0usize;
    for (i, &b) in line.iter().enumerate() {
        if b == b',' {
            if col < f.len() {
                f[col] = &line[start..i];
            }
            col += 1;
            start = i + 1;
        }
    }
    if col < f.len() {
        f[col] = &line[start..];
    }
}

fn find_file(code: &str, date: i64, kind: &str, suffix: &str) -> Option<std::path::PathBuf> {
    let filename = format!("{}_{}_{}.csv", code, date, suffix);
    for root in crate::data_paths::level2_roots() {
        let p = root.join(date.to_string()).join(kind).join(&filename);
        if p.exists() {
            return Some(p);
        }
    }
    None
}

/// 把 f64 微秒的 exchtime 变成旧框架那条链路上的「当日纳秒」（已加 8 小时）。
///
/// 旧框架：`pd.to_timedelta(exchtime/1e6, unit='s') + Timestamp('1970-01-01 08:00:00')`，
/// 即 `<int64_t>(x * 1e9)`（x = exchtime/1e6 的 f64，向零截断）再加 8 小时的 ns。
/// 这里用整数精确算 x*1e9 的截断（与 pandas 最多差 1ns，见文件头）。
fn ns_total(exchtime_us: f64) -> i64 {
    if !exchtime_us.is_finite() {
        return i64::MIN; // 与 NaT 一样：不落在任何窗口里
    }
    let x = exchtime_us / 1_000_000.0;
    mul_1e9_trunc(x) + NS8H
}

/// floor(x * 1e9)，x 为有限正 f64，用整数运算精确求（x = mant * 2^(exp-52)）。
fn mul_1e9_trunc(x: f64) -> i64 {
    if x <= 0.0 {
        return 0;
    }
    let bits = x.to_bits();
    let exp = (((bits >> 52) & 0x7ff) as i32) - 1023;
    let mant = (bits & ((1u64 << 52) - 1)) | (1u64 << 52);
    let e = exp - 52;
    let num = (mant as u128) * 1_000_000_000u128;
    let v = if e >= 0 {
        num << (e as u32)
    } else {
        num >> ((-e) as u32)
    };
    v as i64
}

/// 当日微秒（pandas `_get_time_micros` 的口径）
#[inline]
fn clock_us(ns_total: i64) -> i64 {
    (ns_total / 1000).rem_euclid(DAY_US)
}

#[inline]
fn in_span(ns_total: i64, span: (i64, i64)) -> bool {
    let t = clock_us(ns_total);
    t >= span.0 && t <= span.1
}

// ============================================================================
// 读数据
// ============================================================================

/// 一条逐笔成交（只留计算要用的列）。flag 用 f64：旧代码里它经过 concat 之后就是 float64，
/// 空值要当 NaN 处理（dropna(subset=['flag']) 会丢掉这种行）。
#[derive(Clone, Copy, Default)]
struct TradeRow {
    ns: i64,
    volume: f64,
    flag: f64,
    ask_order: f64,
    bid_order: f64,
}

/// 一条盘口快照
#[derive(Clone, Copy, Default)]
struct MktRow {
    ns: i64,
    ask_vol: [f64; 10],
    bid_vol: [f64; 10],
    volume: f64,
    last_prc: f64,
}

fn read_trades(code: &str, date: i64) -> Option<Vec<TradeRow>> {
    let path = find_file(code, date, "transaction", "transaction")?;
    let mut content = String::new();
    File::open(&path).ok()?.read_to_string(&mut content).ok()?;

    let mut out: Vec<TradeRow> = Vec::with_capacity(content.len() / 120 + 1);
    let mut lines = content.lines();
    lines.next();
    let mut f: [&[u8]; TRD_NCOL] = [&[][..]; TRD_NCOL];
    for line in lines {
        if line.is_empty() {
            continue;
        }
        split_fields(line.as_bytes(), &mut f);
        out.push(TradeRow {
            ns: ns_total(parse_f64_at(f[TRD_COL_EXCHTIME])),
            volume: parse_f64_at(f[TRD_COL_VOLUME]),
            flag: parse_f64_at(f[TRD_COL_FLAG]),
            ask_order: parse_f64_at(f[TRD_COL_ASK_ORDER]),
            bid_order: parse_f64_at(f[TRD_COL_BID_ORDER]),
        });
    }
    Some(out)
}

fn read_market(code: &str, date: i64) -> Option<Vec<MktRow>> {
    let path = find_file(code, date, "market_data", "market_data")?;
    let mut content = String::new();
    File::open(&path).ok()?.read_to_string(&mut content).ok()?;

    let mut out: Vec<MktRow> = Vec::with_capacity(content.len() / 200 + 1);
    let mut lines = content.lines();
    lines.next();
    let mut f: [&[u8]; MKT_NCOL] = [&[][..]; MKT_NCOL];
    for line in lines {
        if line.is_empty() {
            continue;
        }
        split_fields(line.as_bytes(), &mut f);
        let mut row = MktRow {
            ns: ns_total(parse_f64_at(f[MKT_COL_EXCHTIME])),
            volume: parse_f64_at(f[MKT_COL_VOLUME]),
            last_prc: parse_f64_at(f[MKT_COL_LAST_PRC]),
            ..Default::default()
        };
        for k in 0..10 {
            row.ask_vol[k] = parse_f64_at(f[MKT_COL_ASKVOL1 + k * 4]);
            row.bid_vol[k] = parse_f64_at(f[MKT_COL_ASKVOL1 + 2 + k * 4]);
        }
        out.push(row);
    }
    Some(out)
}

// ============================================================================
// 保序名次码 + pandas nargsort（与 hm066.rs 同一套）
// ============================================================================

/// f64 → 保序 int64 名次码：同值同码、不同值不同码。xss 只做比较与 min/max，
/// 所以名次码上的排序置换与原值完全一致。入参必须不含 NaN。
fn rank_codes(vals: &[f64]) -> Vec<i64> {
    let n = vals.len();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| {
        vals[a]
            .partial_cmp(&vals[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut codes = vec![0i64; n];
    let mut cur = 0i64;
    for k in 0..n {
        if k > 0 && vals[idx[k]] != vals[idx[k - 1]] {
            cur += 1;
        }
        codes[idx[k]] = cur;
    }
    codes
}

/// pandas `sort_values(col)`（单键、升序、kind='quicksort'）的 indexer：
/// 非 NaN 部分走 `values.argsort(kind='quicksort')`，NaN 的下标整体拼在最后。
fn pandas_argsort_asc(vals: &[f64]) -> Vec<usize> {
    let n = vals.len();
    let mut non_nan: Vec<usize> = Vec::with_capacity(n);
    let mut nan_idx: Vec<usize> = Vec::new();
    for i in 0..n {
        if vals[i].is_nan() {
            nan_idx.push(i);
        } else {
            non_nan.push(i);
        }
    }
    let sub_vals: Vec<f64> = non_nan.iter().map(|&i| vals[i]).collect();
    let codes = rank_codes(&sub_vals);
    let order = xss_argsort_i64(&codes);
    let mut indexer: Vec<usize> = order.iter().map(|&o| non_nan[o as usize]).collect();
    indexer.extend(nan_idx);
    indexer
}

// ============================================================================
// 指标
// ============================================================================

/// rp.fast_dtw_distance 的 radius=None 分支（src/time_series/mod.rs 的 compute_dtw），
/// 纯 Rust 抄一份：worker 链路不能调 #[pyfunction]。两行滚动数组、标准 DTW。
fn dtw(s1: &[f64], s2: &[f64]) -> f64 {
    if s1.is_empty() || s2.is_empty() {
        return f64::NAN;
    }
    let n = s2.len();
    let mut prev = vec![f64::INFINITY; n + 1];
    let mut curr = vec![f64::INFINITY; n + 1];
    prev[0] = 0.0;
    for i in 1..=s1.len() {
        curr[0] = f64::INFINITY;
        let a = s1[i - 1];
        for j in 1..=n {
            let cost = (a - s2[j - 1]).abs();
            curr[j] = cost + prev[j].min(curr[j - 1]).min(prev[j - 1]);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[n]
}

/// pandas `DataFrame.corr()` 的 Pearson：成对完整观测，逐行复刻
/// pandas/_libs/algos.pyx 的 nancorr（Welford 在线算法，与 hm013 / hm066 同一份）。
fn pearson(x: &[f64], y: &[f64]) -> f64 {
    let mut nobs = 0.0f64;
    let (mut meanx, mut meany) = (0.0f64, 0.0f64);
    let (mut ssqdmx, mut ssqdmy, mut covxy) = (0.0f64, 0.0f64, 0.0f64);
    for i in 0..x.len() {
        if x[i].is_nan() || y[i].is_nan() {
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

// ============================================================================
// 一个窗口的 152 个值
// ============================================================================

/// 单个窗口（open30 或 close30）：返回 152 个值（76 相关 + 76 绝对值）。
/// 任一窗口为空时旧代码会抛异常、整个 get_factor 返回 None，这里也返回 None。
fn single(trades: &[TradeRow], tw: &[usize], mkt: &[MktRow], mw: &[usize]) -> Option<Vec<f32>> {
    if tw.is_empty() || mw.is_empty() {
        return None;
    }
    let nt = tw.len();
    let nm = mw.len();

    // ── 相对秒（各自减自己的最小值）
    let tmin = tw.iter().map(|&i| trades[i].ns).min()?;
    let mmin = mw.iter().map(|&i| mkt[i].ns).min()?;
    let mut times: Vec<f64> = Vec::with_capacity(nt + nm);
    for &i in tw.iter() {
        times.push((trades[i].ns as f64 - tmin as f64) / 1e9);
    }
    for &i in mw.iter() {
        times.push((mkt[i].ns as f64 - mmin as f64) / 1e9);
    }

    // ── 排序 + ffill 出每笔成交的快照编号
    let perm = pandas_argsort_asc(&times);
    let mut labels: Vec<i64> = vec![-1; nt];
    {
        let mut cur: i64 = -1;
        for &k in perm.iter() {
            if k >= nt {
                cur = (k - nt) as i64;
            } else {
                labels[k] = cur;
            }
        }
    }

    // ── 按 label 分组（perm 顺序里 label 单调不减，扫一遍即可）
    let mut dtw_f: Vec<[f64; 4]> = vec![[f64::NAN; 4]; nm];
    let mut buf: Vec<usize> = Vec::new(); // 组内逐笔（perm 顺序）
    let mut flush = |lab: i64, buf: &mut Vec<usize>, out: &mut Vec<[f64; 4]>| {
        if lab < 0 || buf.is_empty() {
            buf.clear();
            return;
        }
        let mut buy_v: Vec<f64> = Vec::new();
        let mut sell_v: Vec<f64> = Vec::new();
        let mut buy_t: Vec<f64> = Vec::new();
        let mut sell_t: Vec<f64> = Vec::new();
        let mut bb_v: Vec<f64> = Vec::new();
        let mut ba_v: Vec<f64> = Vec::new();
        let mut bb_t: Vec<f64> = Vec::new();
        let mut ba_t: Vec<f64> = Vec::new();
        for &ti in buf.iter() {
            let r = &trades[ti];
            let t = (r.ns as f64 - tmin as f64) / 1e9;
            if r.flag == 66.0 {
                buy_v.push(r.volume / 100.0);
                buy_t.push(t);
            }
            if r.flag == 83.0 {
                sell_v.push(r.volume / 100.0);
                sell_t.push(t);
            }
            if r.ask_order == 0.0 {
                bb_v.push(r.volume / 100.0);
                bb_t.push(t);
            }
            if r.bid_order == 0.0 {
                ba_v.push(r.volume / 100.0);
                ba_t.push(t);
            }
        }
        let v = [
            dtw(&buy_v, &sell_v),
            dtw(&buy_t, &sell_t),
            dtw(&bb_v, &ba_v),
            dtw(&bb_t, &ba_t),
        ];
        if (lab as usize) < out.len() {
            out[lab as usize] = v;
        }
        buf.clear();
    };

    let mut cur_lab: i64 = -1;
    for &k in perm.iter() {
        if k >= nt {
            flush(cur_lab, &mut buf, &mut dtw_f);
            cur_lab = (k - nt) as i64;
        } else {
            let ti = tw[k];
            if labels[k] >= 0 && !trades[ti].flag.is_nan() {
                buf.push(ti);
            }
        }
    }
    flush(cur_lab, &mut buf, &mut dtw_f);

    // ── 盘口 8 个派生列（窗口内行序）
    let mut mcols: Vec<[f64; 8]> = Vec::with_capacity(nm);
    for (j, &mi) in mw.iter().enumerate() {
        let r = &mkt[mi];
        let (b1, b2, b3, b4, b5) = (
            r.bid_vol[0],
            r.bid_vol[1],
            r.bid_vol[2],
            r.bid_vol[3],
            r.bid_vol[4],
        );
        let (a1, a2, a3, a4, a5) = (
            r.ask_vol[0],
            r.ask_vol[1],
            r.ask_vol[2],
            r.ask_vol[3],
            r.ask_vol[4],
        );
        // want_diff_1
        let wd1 = (b1 - a1) / (b1 + a1);
        // want_diff_5：分子分母都按 pandas 的左结合顺序累加
        let num5 = (((((((((b1 + b2) + b3) + b4) + b5) - a1) - a2) - a3) - a4) - a5);
        let den5 = (((((((((b1 + b2) + b3) + b4) + b5) + a1) + a2) + a3) + a4) + a5);
        let wd5 = num5 / den5;
        // want_diff_10
        let mut num10 = b1 + b2 + b3 + b4 + b5;
        let mut den10 = num10;
        for k in 5..10 {
            num10 += r.bid_vol[k];
            den10 += r.bid_vol[k];
        }
        for k in 0..10 {
            num10 -= r.ask_vol[k];
            den10 += r.ask_vol[k];
        }
        let wd10 = num10 / den10;
        let vol = if j == 0 {
            f64::NAN
        } else {
            (r.volume - mkt[mw[j - 1]].volume) / 100.0
        };
        mcols.push([wd1, wd5, wd10, wd1.abs(), wd5.abs(), wd10.abs(), vol, r.last_prc]);
    }

    // ── twins_f：M 行（label 0..M-1）；twins_b：M+1 行（组结果整体下移一格）
    let mut cf: Vec<Vec<f64>> = vec![Vec::with_capacity(nm); 12];
    let mut cb: Vec<Vec<f64>> = vec![Vec::with_capacity(nm + 1); 12];
    for l in 0..nm {
        for k in 0..4 {
            cf[k].push(dtw_f[l][k]);
            cb[k].push(if l == 0 { f64::NAN } else { dtw_f[l - 1][k] });
        }
        for k in 0..8 {
            cf[4 + k].push(mcols[l][k]);
            cb[4 + k].push(mcols[l][k]);
        }
    }
    for k in 0..4 {
        cb[k].push(dtw_f[nm - 1][k]);
    }
    for k in 4..12 {
        cb[k].push(f64::NAN);
    }

    // ── 76 个相关系数（12 列严格上三角：f 一遍、b 一遍），再整体取绝对值
    let mut res: Vec<f64> = Vec::with_capacity(PER_WINDOW);
    for tag in 0..2 {
        let c = if tag == 0 { &cf } else { &cb };
        for i in 0..4 {
            for j in (i + 1)..12 {
                res.push(pearson(&c[i], &c[j]));
            }
        }
    }
    debug_assert_eq!(res.len(), PER_BLOCK);
    let abs_part: Vec<f64> = res.iter().map(|v| v.abs()).collect();
    res.extend(abs_part);
    Some(res.into_iter().map(|v| v as f32).collect())
}

/// 逐股的一天：算不出来返回 None（上游补 NaN，保证因子个数恒定）。
pub fn compute_stock(code: &str, date: i64) -> Option<Vec<f32>> {
    let trades = read_trades(code, date)?;
    if trades.is_empty() {
        return None;
    }
    let mkt = read_market(code, date)?;
    if mkt.is_empty() {
        return None;
    }

    let tw_open: Vec<usize> = (0..trades.len())
        .filter(|&i| in_span(trades[i].ns, OPEN30))
        .collect();
    let mw_open: Vec<usize> = (0..mkt.len()).filter(|&i| in_span(mkt[i].ns, OPEN30)).collect();
    let tw_close: Vec<usize> = (0..trades.len())
        .filter(|&i| in_span(trades[i].ns, CLOSE30))
        .collect();
    let mw_close: Vec<usize> = (0..mkt.len())
        .filter(|&i| in_span(mkt[i].ns, CLOSE30))
        .collect();

    let mut out: Vec<f32> = Vec::with_capacity(N_OUT);
    out.extend(single(&trades, &tw_open, &mkt, &mw_open)?);
    out.extend(single(&trades, &tw_close, &mkt, &mw_close)?);
    debug_assert_eq!(out.len(), N_OUT);
    Some(out)
}

/// hm55 全是逐股因子，没有横截面版本。
pub fn compute_cross(_date: i64) -> Option<(Vec<String>, Vec<f32>)> {
    None
}

// ============================================================================
// ============================================================================
// numpy x86-simd-sort argsort(int64) 的逐行复刻
// ============================================================================
//
// 8 通道寄存器用定长数组表示；掩码用 u8（bit l 对应通道 l）。
// 所有比较都是 int64 有符号比较。

const LANES: usize = 8;
const I64MAX: i64 = i64::MAX;

#[inline]
fn swap_perm(scale: usize) -> [usize; LANES] {
    match scale {
        2 => [1, 0, 3, 2, 5, 4, 7, 6],
        4 => [2, 3, 0, 1, 6, 7, 4, 5],
        _ => [4, 5, 6, 7, 0, 1, 2, 3],
    }
}

#[inline]
fn reverse_perm(scale: usize) -> [usize; LANES] {
    match scale {
        2 => [1, 0, 3, 2, 5, 4, 7, 6],
        4 => [3, 2, 1, 0, 7, 6, 5, 4],
        _ => [7, 6, 5, 4, 3, 2, 1, 0],
    }
}

#[inline]
fn perm_k(k: &[i64; LANES], p: &[usize; LANES]) -> [i64; LANES] {
    let mut o = [0i64; LANES];
    for l in 0..LANES {
        o[l] = k[p[l]];
    }
    o
}

#[inline]
fn perm_i(i: &[i64; LANES], p: &[usize; LANES]) -> [i64; LANES] {
    perm_k(i, p)
}

/// 只按键的 cmp_merge：0 -> min，1 -> max
fn cmp_merge_k(k1: &[i64; LANES], k2: &[i64; LANES], mask: u8) -> [i64; LANES] {
    let mut o = [0i64; LANES];
    for l in 0..LANES {
        let (a, b) = (k1[l], k2[l]);
        let (lo, hi) = if a < b { (a, b) } else { (b, a) };
        o[l] = if (mask >> l) & 1 == 1 { hi } else { lo };
    }
    o
}

/// 带下标的 cmp_merge（xss-network-keyvaluesort.hpp）
fn cmp_merge_ki(
    k1: &[i64; LANES],
    i1: &[i64; LANES],
    k2: &[i64; LANES],
    i2: &[i64; LANES],
    mask: u8,
) -> ([i64; LANES], [i64; LANES]) {
    let mut ko = [0i64; LANES];
    let mut io = [0i64; LANES];
    for l in 0..LANES {
        let (a, b) = (k1[l], k2[l]);
        let (lo, hi) = if a < b { (a, b) } else { (b, a) };
        ko[l] = if (mask >> l) & 1 == 1 { hi } else { lo };
        // 结果来自 in1 就保留下标 1，否则用下标 2
        io[l] = if ko[l] == a { i1[l] } else { i2[l] };
    }
    (ko, io)
}

/// COEX：稳定的比较交换
fn coex(
    k1: &[i64; LANES],
    i1: &[i64; LANES],
    k2: &[i64; LANES],
    i2: &[i64; LANES],
) -> ([i64; LANES], [i64; LANES], [i64; LANES], [i64; LANES]) {
    let mut ko1 = [0i64; LANES];
    let mut ko2 = [0i64; LANES];
    let mut io1 = [0i64; LANES];
    let mut io2 = [0i64; LANES];
    for l in 0..LANES {
        let (a, b) = (k1[l], k2[l]);
        let (lo, hi) = if a < b { (a, b) } else { (b, a) };
        ko1[l] = lo;
        ko2[l] = hi;
        let eq = lo == a;
        io1[l] = if eq { i1[l] } else { i2[l] };
        io2[l] = if eq { i2[l] } else { i1[l] };
    }
    (ko1, io1, ko2, io2)
}

fn sort_reg_8lanes(k: &[i64; LANES], i: &[i64; LANES]) -> ([i64; LANES], [i64; LANES]) {
    let (mut k, mut i) = cmp_merge_ki(k, i, &perm_k(k, &reverse_perm(2)), &perm_i(i, &reverse_perm(2)), 0xAA);
    let t = cmp_merge_ki(&k, &i, &perm_k(&k, &reverse_perm(4)), &perm_i(&i, &reverse_perm(4)), 0xCC);
    k = t.0;
    i = t.1;
    let t = cmp_merge_ki(&k, &i, &perm_k(&k, &swap_perm(2)), &perm_i(&i, &swap_perm(2)), 0xAA);
    k = t.0;
    i = t.1;
    let t = cmp_merge_ki(&k, &i, &perm_k(&k, &reverse_perm(8)), &perm_i(&i, &reverse_perm(8)), 0xF0);
    k = t.0;
    i = t.1;
    let t = cmp_merge_ki(&k, &i, &perm_k(&k, &swap_perm(4)), &perm_i(&i, &swap_perm(4)), 0xCC);
    k = t.0;
    i = t.1;
    let t = cmp_merge_ki(&k, &i, &perm_k(&k, &swap_perm(2)), &perm_i(&i, &swap_perm(2)), 0xAA);
    (t.0, t.1)
}

fn bitonic_merge_reg_8lanes(k: &[i64; LANES], i: &[i64; LANES]) -> ([i64; LANES], [i64; LANES]) {
    let t = cmp_merge_ki(k, i, &perm_k(k, &swap_perm(8)), &perm_i(i, &swap_perm(8)), 0xF0);
    let t = cmp_merge_ki(&t.0, &t.1, &perm_k(&t.0, &swap_perm(4)), &perm_i(&t.1, &swap_perm(4)), 0xCC);
    cmp_merge_ki(&t.0, &t.1, &perm_k(&t.0, &swap_perm(2)), &perm_i(&t.1, &swap_perm(2)), 0xAA)
}

fn bitonic_clean_n_vec(keys: &mut [[i64; LANES]], idxs: &mut [[i64; LANES]], base: usize, num_vecs: usize) {
    let mut num = num_vecs / 2;
    while num >= 2 {
        let mut j = 0usize;
        while j < num_vecs {
            for i in 0..num / 2 {
                let a = base + j + i;
                let b = base + j + i + num / 2;
                let t = coex(&keys[a], &idxs[a], &keys[b], &idxs[b]);
                keys[a] = t.0;
                idxs[a] = t.1;
                keys[b] = t.2;
                idxs[b] = t.3;
            }
            j += num;
        }
        num /= 2;
    }
}

fn bitonic_merge_n_vec(keys: &mut [[i64; LANES]], idxs: &mut [[i64; LANES]], base: usize, num_vecs: usize) {
    if num_vecs == 2 {
        let j = base + 1;
        keys[j].reverse();
        idxs[j].reverse();
        let t = coex(&keys[base], &idxs[base], &keys[j], &idxs[j]);
        keys[base] = t.0;
        idxs[base] = t.1;
        keys[j] = t.2;
        idxs[j] = t.3;
        keys[j].reverse();
        idxs[j].reverse();
    } else if num_vecs > 2 {
        for i in 0..num_vecs / 2 {
            let a = base + i;
            let j = base + num_vecs - i - 1;
            keys[j].reverse();
            idxs[j].reverse();
            let t = coex(&keys[a], &idxs[a], &keys[j], &idxs[j]);
            keys[a] = t.0;
            idxs[a] = t.1;
            keys[j] = t.2;
            idxs[j] = t.3;
            keys[j].reverse();
            idxs[j].reverse();
        }
    }
    bitonic_clean_n_vec(keys, idxs, base, num_vecs);
    for i in 0..num_vecs {
        let t = bitonic_merge_reg_8lanes(&keys[base + i], &idxs[base + i]);
        keys[base + i] = t.0;
        idxs[base + i] = t.1;
    }
}

fn bitonic_fullmerge_n_vec(
    keys: &mut [[i64; LANES]],
    idxs: &mut [[i64; LANES]],
    base: usize,
    num_vecs: usize,
    num_per: usize,
) {
    if num_per > num_vecs {
        return;
    }
    for i in 0..num_vecs / num_per {
        bitonic_merge_n_vec(keys, idxs, base + i * num_per, num_per);
    }
    bitonic_fullmerge_n_vec(keys, idxs, base, num_vecs, num_per * 2);
}

/// argsort_n&lt;vtype, argtype, 256&gt; 的 numVecs 会按 N 递归减半。
fn argsort_n_vec(arr: &[i64], ind: &mut [i64], n: usize, num_vecs: usize) {
    let mut num_vecs = num_vecs;
    while num_vecs > 1 && n * 2 <= num_vecs * LANES {
        num_vecs /= 2;
    }
    let mut keys = vec![[0i64; LANES]; num_vecs];
    let mut idxs = vec![[0i64; LANES]; num_vecs];
    let half = num_vecs / 2;
    for i in 0..half {
        for l in 0..LANES {
            let v = ind[i * LANES + l];
            idxs[i][l] = v;
            keys[i][l] = arr[v as usize];
        }
    }
    for i in half..num_vecs {
        let num_to_read = if n > i * LANES { (n - i * LANES).min(LANES) } else { 0 };
        for l in 0..LANES {
            if l < num_to_read {
                let v = ind[i * LANES + l];
                idxs[i][l] = v;
                keys[i][l] = arr[v as usize];
            } else {
                // indexType::zmm_max() / keyType::zmm_max()：补齐的通道放最大值
                idxs[i][l] = I64MAX;
                keys[i][l] = I64MAX;
            }
        }
    }
    for i in 0..num_vecs {
        let t = sort_reg_8lanes(&keys[i], &idxs[i]);
        keys[i] = t.0;
        idxs[i] = t.1;
    }
    bitonic_fullmerge_n_vec(&mut keys, &mut idxs, 0, num_vecs, 2);
    for i in 0..half {
        for l in 0..LANES {
            ind[i * LANES + l] = idxs[i][l];
        }
    }
    for i in half..num_vecs {
        let num_to_write = if n > i * LANES { (n - i * LANES).min(LANES) } else { 0 };
        for l in 0..num_to_write {
            ind[i * LANES + l] = idxs[i][l];
        }
    }
}

fn argsort_n(arr: &[i64], ind: &mut [i64], left: usize, n: usize) {
    let mut sub: Vec<i64> = ind[left..left + n].to_vec();
    argsort_n_vec(arr, &mut sub, n, 256 / LANES);
    ind[left..left + n].copy_from_slice(&sub);
}

/// get_pivot_64bit（numlanes=8 的 median-of-8 分支）
fn get_pivot(arr: &[i64], ind: &[i64], left: usize, right: usize) -> i64 {
    if right - left >= LANES {
        let size = (right - left) / LANES;
        // _mm512_set_epi64(v1..v8)：通道 0 放最后一个参数
        let rand_vec: [i64; LANES] = [
            arr[ind[left + 8 * size] as usize],
            arr[ind[left + 7 * size] as usize],
            arr[ind[left + 6 * size] as usize],
            arr[ind[left + 5 * size] as usize],
            arr[ind[left + 4 * size] as usize],
            arr[ind[left + 3 * size] as usize],
            arr[ind[left + 2 * size] as usize],
            arr[ind[left + 1 * size] as usize],
        ];
        // vtype::sort_vec：只按键的排序网络
        let mut k = rand_vec;
        k = cmp_merge_k(&k, &perm_k(&k, &reverse_perm(2)), 0xAA);
        k = cmp_merge_k(&k, &perm_k(&k, &reverse_perm(4)), 0xCC);
        k = cmp_merge_k(&k, &perm_k(&k, &swap_perm(2)), 0xAA);
        k = cmp_merge_k(&k, &perm_k(&k, &reverse_perm(8)), 0xF0);
        k = cmp_merge_k(&k, &perm_k(&k, &swap_perm(4)), 0xCC);
        k = cmp_merge_k(&k, &perm_k(&k, &swap_perm(2)), 0xAA);
        return k[4];
    }
    arr[ind[right] as usize]
}

/// partition_vec（argsort 版：>= pivot 的压缩写到右侧，< pivot 的写到左侧，通道序保持）
fn partition_vec(
    ind: &mut [i64],
    l_store: usize,
    r_store_plus: usize,
    arg_vec: &[i64; LANES],
    curr_vec: &[i64; LANES],
    pivot: i64,
) -> usize {
    let mut gt = [false; LANES];
    let mut amount = 0usize;
    for l in 0..LANES {
        gt[l] = curr_vec[l] >= pivot;
        if gt[l] {
            amount += 1;
        }
    }
    let mut p = l_store;
    for l in 0..LANES {
        if !gt[l] {
            ind[p] = arg_vec[l];
            p += 1;
        }
    }
    let mut p = r_store_plus - amount;
    for l in 0..LANES {
        if gt[l] {
            ind[p] = arg_vec[l];
            p += 1;
        }
    }
    amount
}

fn argpartition_scalar(
    arr: &[i64],
    ind: &mut [i64],
    left_in: usize,
    right_in: usize,
    pivot: i64,
    smallest: &mut i64,
    biggest: &mut i64,
) -> usize {
    let mut left = left_in;
    let mut right = right_in;
    let rest = (right - left) % LANES;
    for _ in 0..rest {
        let v = arr[ind[left] as usize];
        *smallest = (*smallest).min(v);
        *biggest = (*biggest).max(v);
        if !(v < pivot) {
            right -= 1;
            ind.swap(left, right);
        } else {
            left += 1;
        }
    }
    if left == right {
        return left;
    }
    if right - left == LANES {
        let mut av = [0i64; LANES];
        av.copy_from_slice(&ind[left..left + LANES]);
        let mut cv = [0i64; LANES];
        for l in 0..LANES {
            cv[l] = arr[av[l] as usize];
        }
        let amount = partition_vec(ind, left, left + LANES, &av, &cv, pivot);
        for l in 0..LANES {
            *smallest = (*smallest).min(cv[l]);
            *biggest = (*biggest).max(cv[l]);
        }
        return left + (LANES - amount);
    }
    let mut av_left = [0i64; LANES];
    av_left.copy_from_slice(&ind[left..left + LANES]);
    let mut cv_left = [0i64; LANES];
    for l in 0..LANES {
        cv_left[l] = arr[av_left[l] as usize];
    }
    let mut av_right = [0i64; LANES];
    av_right.copy_from_slice(&ind[right - LANES..right]);
    let mut cv_right = [0i64; LANES];
    for l in 0..LANES {
        cv_right[l] = arr[av_right[l] as usize];
    }
    let mut r_store = right - LANES;
    let mut l_store = left;
    let mut left = left + LANES;
    let mut right = right - LANES;
    while right != left {
        let (av, cv) = if (r_store + LANES) - right < left - l_store {
            right -= LANES;
            let mut av = [0i64; LANES];
            av.copy_from_slice(&ind[right..right + LANES]);
            let mut cv = [0i64; LANES];
            for l in 0..LANES {
                cv[l] = arr[av[l] as usize];
            }
            (av, cv)
        } else {
            let mut av = [0i64; LANES];
            av.copy_from_slice(&ind[left..left + LANES]);
            let mut cv = [0i64; LANES];
            for l in 0..LANES {
                cv[l] = arr[av[l] as usize];
            }
            left += LANES;
            (av, cv)
        };
        let amount = partition_vec(ind, l_store, r_store + LANES, &av, &cv, pivot);
        for l in 0..LANES {
            *smallest = (*smallest).min(cv[l]);
            *biggest = (*biggest).max(cv[l]);
        }
        r_store -= amount;
        l_store += LANES - amount;
    }
    let amount = partition_vec(ind, l_store, r_store + LANES, &av_left, &cv_left, pivot);
    for l in 0..LANES {
        *smallest = (*smallest).min(cv_left[l]);
        *biggest = (*biggest).max(cv_left[l]);
    }
    l_store += LANES - amount;
    let amount = partition_vec(ind, l_store, l_store + LANES, &av_right, &cv_right, pivot);
    for l in 0..LANES {
        *smallest = (*smallest).min(cv_right[l]);
        *biggest = (*biggest).max(cv_right[l]);
    }
    l_store + LANES - amount
}

const NUM_UNROLL: usize = 4;

fn argpartition_unrolled(
    arr: &[i64],
    ind: &mut [i64],
    left_in: usize,
    right_in: usize,
    pivot: i64,
    smallest: &mut i64,
    biggest: &mut i64,
) -> usize {
    if right_in - left_in <= 8 * NUM_UNROLL * LANES {
        return argpartition_scalar(arr, ind, left_in, right_in, pivot, smallest, biggest);
    }
    let mut left = left_in;
    let mut right = right_in;
    let rest = (right - left) % (NUM_UNROLL * LANES);
    for _ in 0..rest {
        let v = arr[ind[left] as usize];
        *smallest = (*smallest).min(v);
        *biggest = (*biggest).max(v);
        if !(v < pivot) {
            right -= 1;
            ind.swap(left, right);
        } else {
            left += 1;
        }
    }
    if left == right {
        return left;
    }
    let mut kv_left = [[0i64; LANES]; NUM_UNROLL];
    let mut iv_left = [[0i64; LANES]; NUM_UNROLL];
    let mut kv_right = [[0i64; LANES]; NUM_UNROLL];
    let mut iv_right = [[0i64; LANES]; NUM_UNROLL];
    for ii in 0..NUM_UNROLL {
        iv_left[ii].copy_from_slice(&ind[left + LANES * ii..left + LANES * (ii + 1)]);
        for l in 0..LANES {
            kv_left[ii][l] = arr[iv_left[ii][l] as usize];
        }
        let s = right - LANES * (NUM_UNROLL - ii);
        iv_right[ii].copy_from_slice(&ind[s..s + LANES]);
        for l in 0..LANES {
            kv_right[ii][l] = arr[iv_right[ii][l] as usize];
        }
    }
    let mut r_store = right - LANES;
    let mut l_store = left;
    let mut left = left + NUM_UNROLL * LANES;
    let mut right = right - NUM_UNROLL * LANES;

    macro_rules! upd {
        ($cv:expr) => {
            for l in 0..LANES {
                *smallest = (*smallest).min($cv[l]);
                *biggest = (*biggest).max($cv[l]);
            }
        };
    }

    while right != left {
        let mut avs = [[0i64; LANES]; NUM_UNROLL];
        let mut cvs = [[0i64; LANES]; NUM_UNROLL];
        if (r_store + LANES) - right < left - l_store {
            right -= NUM_UNROLL * LANES;
            for ii in 0..NUM_UNROLL {
                avs[ii].copy_from_slice(&ind[right + ii * LANES..right + (ii + 1) * LANES]);
            }
        } else {
            for ii in 0..NUM_UNROLL {
                avs[ii].copy_from_slice(&ind[left + ii * LANES..left + (ii + 1) * LANES]);
            }
            left += NUM_UNROLL * LANES;
        }
        for ii in 0..NUM_UNROLL {
            for l in 0..LANES {
                cvs[ii][l] = arr[avs[ii][l] as usize];
            }
            let amount = partition_vec(ind, l_store, r_store + LANES, &avs[ii], &cvs[ii], pivot);
            upd!(cvs[ii]);
            l_store += LANES - amount;
            r_store -= amount;
        }
    }
    for ii in 0..NUM_UNROLL {
        let amount = partition_vec(ind, l_store, r_store + LANES, &iv_left[ii], &kv_left[ii], pivot);
        upd!(kv_left[ii]);
        l_store += LANES - amount;
        r_store -= amount;
    }
    for ii in 0..NUM_UNROLL {
        let amount = partition_vec(ind, l_store, r_store + LANES, &iv_right[ii], &kv_right[ii], pivot);
        upd!(kv_right[ii]);
        l_store += LANES - amount;
        r_store -= amount;
    }
    l_store
}

fn argsort_rec(arr: &[i64], ind: &mut [i64], left: usize, right: i64, max_iters: i64) {
    if max_iters <= 0 {
        // x86-simd-sort 这里退回 std::sort（libstdc++ introsort）。本复刻用「按值稳定排序」
        // 代替：实测 4 组验收数据的 48 个候选数组一次都没走到这里。
        let mut sub: Vec<i64> = ind[left..=(right as usize)].to_vec();
        sub.sort_by(|&a, &b| arr[a as usize].cmp(&arr[b as usize]));
        ind[left..=(right as usize)].copy_from_slice(&sub);
        return;
    }
    if (right + 1) as usize - left <= 256 {
        argsort_n(arr, ind, left, (right + 1) as usize - left);
        return;
    }
    let (l, r) = (left, right as usize);
    let pivot = get_pivot(arr, ind, l, r);
    let mut smallest = I64MAX;
    let mut biggest = i64::MIN;
    let pivot_index = argpartition_unrolled(arr, ind, l, r + 1, pivot, &mut smallest, &mut biggest);
    if pivot != smallest {
        argsort_rec(arr, ind, left, pivot_index as i64 - 1, max_iters - 1);
    }
    if pivot != biggest {
        argsort_rec(arr, ind, pivot_index, right, max_iters - 1);
    }
}

/// numpy 2.3.5 在 AVX512 机器上 `np.argsort(int64_array, kind="quicksort")` 的结果（升序下标）。
fn xss_argsort_i64(arr: &[i64]) -> Vec<i64> {
    let n = arr.len();
    let mut ind: Vec<i64> = (0..n as i64).collect();
    if n <= 1 {
        return ind;
    }
    // std::is_sorted(arr, arr+n, arr[a] < arr[b]) 的提前返回
    if (1..n).all(|i| arr[i - 1] <= arr[i]) {
        return ind;
    }
    let mut log2 = 0i64;
    while (1i64 << (log2 + 1)) <= n as i64 {
        log2 += 1;
    }
    let max_iters = 2 * log2;
    argsort_rec(arr, &mut ind, 0, n as i64 - 1, max_iters);
    ind
}
