//! hm23「主被转换订单」—— 旧框架（design_whatever_out.TickDataBase）的 Rust 复现。
//!
//! 旧文件：/home/chenzongwei/pythoncode/成果hm23_主被转换订单/hm23_okay.py
//!   run_base 里 Go.get_factor 用的是 read_trade_data_full_market()，也就是**一天全市场**
//!   的逐笔成交合成一张表，再由自己切窗口；run_factor_out 传的是 cross_section=True，
//!   所以旧框架是一天调一次 get_factor、返回一列一列的 pandas Series（index 是股票代码）。
//!
//! ════════════════════════════════════════════════════════════════════
//! 为什么这个系列只能做横截面（compute_cross）
//! ════════════════════════════════════════════════════════════════════
//! 最后的输出是 `(asks[i].rank() - bids[i].rank()).abs()`，其中 asks / bids 是
//! group_by('code') 之后每只股票一行的表，rank() 是**跨股票**排名。逐股单独算拿不到
//! 别的股票的值，排名无从谈起，所以走 kind='cross_section' 的 compute_cross。
//!
//! ════════════════════════════════════════════════════════════════════
//! 旧代码逐步对照（写代码前抄在这里，方便复核）
//! ════════════════════════════════════════════════════════════════════
//! 1) df = read_trade_data_full_market()      // lazy=1，数据是**先全读进内存再 .lazy()**
//!    列：symbol(→code) exchtime price volume turnover flag localtime ask_order bid_order
//!    exchtime 先 round(0)（Float64 → 整数微秒），再 deal_data 转成 Datetime：
//!         datetime = 1970-01-01 08:00:00 + exchtime 微秒
//!    所以「当天钟点」= (exchtime + 8h) mod 24h。
//!    ！！注意 read_trade_data_full_market 用的是自己的 with_retreat 默认值 1（保留撤单），
//!      但旧代码紧接着自己 filter(flag != 32)，等价于**丢掉撤单行**。
//!      所以 Rust 侧读的时候保留 32、随后按数值筛掉（不能靠 read_trade_fast_inner 的
//!      with_retreat=false，那条判断按字节比 "32"，2025 年文件里写的是 "32.0"，筛不掉）。
//! 2) 切窗口（pl.col('exchtime').dt.time() 与 pl.time(...) 比，微秒精度）：
//!      df（先切一次，两端都含）：09:30:00 <= t <= 14:57:00
//!      all       = df
//!      open30    : 09:30:00 <  t <  10:00:00
//!      morning   : 09:30:00 <= t <  11:30:00
//!      afternoon : 13:00:00 <= t <  14:57:00
//!      close30   : 14:30:00 <= t <  14:57:00
//!    子窗口都落在外层区间之内，所以窗口边界就是各自那两条。
//! 3) sing(df)（每个时段各算一遍）：
//!      a) total_volume = volume.sum().over('code')      // 时段内该股票的总成交量（分母）
//!      b) asks = df.unique(["code","ask_order","flag"])
//!                 .filter(flag.count().over(["code","ask_order"]) > 1)
//!                 .sort(["exchtime"])
//!                 .with_columns(exchtime.diff().dt.total_seconds().over("code")   -> exchtime_diff_code
//!                               exchtime.diff(1000).dt.total_seconds()            -> exchtime_diff)
//!         bids 同理，键换成 bid_order。
//!         · unique 保留每组**第一次出现**的那一行（按读入顺序；polars 的 unique 输出顺序
//!           不稳定，见下面「不可复现」一节）。
//!         · filter 里的 count 是在去重后的表上按 (code, ask_order) 数行数，>1 才留。
//!         · diff(...).dt.total_seconds() 返回 **Int64 整秒**（微秒差 // 1e6，不是浮点秒）。
//!         · exchtime_diff 是**全局**滞后 1000 行，不是逐股。
//!      c) group_by('code').agg(...) 出 11 个量（顺序就是 names 的顺序）：
//!           exchtime_diff_mean / exchtime_diff_code_mean
//!           exchtime_diff_std  / exchtime_diff_code_std      // ddof=1
//!           exchtime_diff_skew / exchtime_diff_code_skew     // bias=True，即 m3/m2^1.5
//!           volume_ratio_{sum,mean,std,skew}                 // volume/total_volume
//!           exchtime_diff_and_code = corr(exchtime_diff, exchtime_diff_code)
//!         再 .drop_nulls()：mean/std/sum 里出现 **null**（不是 NaN）的股票整行丢掉。
//! 4) 对每一路：asks[i].rank() - bids[i].rank() 取绝对值（pandas rank：升序、并列取平均、
//!    NaN 保持 NaN；两边的 index 取并集，只在一边出现的算 NaN）。
//! 5) 输出 5 个时段 × 11 个量 = 55 个值，顺序 all / open30 / morning / afternoon / close30。
//!
//! ════════════════════════════════════════════════════════════════════
//! 不可复现的 4 个量（重要，别把它当成算错了）
//! ════════════════════════════════════════════════════════════════════
//! `exchtime_diff` 是 sort(["exchtime"]) 之后**全局滞后 1000 行**的差。同一时刻的那些行
//! 谁先谁后，决定了每个位置上的 t[i] 与 t[i-1000] 具体取到哪两行；而这个并列次序来自
//! polars 的 `unique` 输出顺序，它在多线程下**每个进程都可能不同**（实测 POLARS_MAX_THREADS=1
//! 又是另一个值）。于是 `exchtime_diff` 跟着变。出问题的是每个时段的
//! exchtime_diff_mean / exchtime_diff_std / exchtime_diff_skew / exchtime_diff_and_code
//! 这 4 个量，共 20 列；其余 35 列（exchtime_diff_code_* 与 volume_ratio_*）只看
//! 「同一只股票内部的时间序列」或「同一只股票内部的量」，与全市场并列次序无关，因而稳定。
//!
//! 实测数字（2026-10 复现时跑的）：
//!   · 子集 106 只（20250303）：同一份 hm23_okay.py 连跑两次，5830 个格子里 **72 个不同**，
//!     幅度 1~3，全部落在那 4 个量上；其余 35 列两次完全一致。
//!   · 全市场 6066 只（000001@20250303）：连跑两次，55 个值里 **20 个不同**，
//!     最大差 313（afternoon_exchtime_diff_and_code：2009 vs 1696）。
//!   · 本实现 vs 旧代码：子集上差异只在那 4 个量（单次运行 2~4 格）；
//!     全市场上「新 vs 旧第一次」差 21 格、「新 vs 旧第二次」差 20 格，
//!     多出来的唯一一格是 all_volume_ratio_std（10 vs 9），来自 polars 的 f32 累加
//!     与本地 f64 累加的一次排名翻转。
//!
//! 结论与处理口径：`names_save` 的 18 个入选名里有 **10 个**落在上面这 4 个不稳定量里
//! （morning/all/afternoon 的 exchtime_diff_mean、_std、_skew、_and_code），
//! 旧代码本身就没有唯一答案，本实现给出的是其中**确定的那一个**取值，按「接受 + 写清楚」处理，
//! 不剔除这些列；exchtime_diff_code_* 与 volume_ratio_* 仍然要精确对上，已对上。
//! 另外 `verify_series.py` 用 rtol=1e-4 比的是「排名差」，一个名次就是 1e-3 量级，
//! 所以在这 20 列上这个阈值不可能通过，属预期，不要据此改实现。
//!
//! ════════════════════════════════════════════════════════════════════
//! 精度取舍
//! ════════════════════════════════════════════════════════════════════
//! 全部中间量用 f64（旧代码里 volume_ratio_* 是 f32、exchtime_diff_* 是 i64/f64，
//! 实测按 f64 累加算出来的排名与旧代码逐格一致）。唯一照抄 numpy 的是
//! volume_ratio_sum：旧代码走 polars 的 f32 sum，这里用 numpy 的 pairwise sum，
//! 是为了让「两个股票的值是否严格相等」这件事和实测结果对得上。
//! 逐笔读入后按 CSV 的 index 列升序重排（旧代码 read_csv 之后 sort(['index'])；
//! 文件里的行序并不总是 index 升序，实测 24 万行里会有 2 行错位）。

use std::collections::HashMap;

use rayon::prelude::*;

use crate::data_paths;

const SEC_US: i64 = 1_000_000;
const DAY_US: i64 = 86_400_000_000;
/// exchtime（epoch 微秒，UTC）→ 北京时间的加数。
const OFF_US: i64 = 8 * 3600 * SEC_US;

const N_WINDOWS: usize = 5;
const PER_WINDOW: usize = 11;
pub const N_OUT: usize = N_WINDOWS * PER_WINDOW; // 55

/// 五个时段：(下界, 下界含否, 上界, 上界含否)，单位是当天零点起的微秒。
const WINDOWS: [(i64, bool, i64, bool); N_WINDOWS] = [
    // all: 09:30:00 <= t <= 14:57:00
    ((9 * 3600 + 30 * 60) * SEC_US, true, (14 * 3600 + 57 * 60) * SEC_US, true),
    // open30: 09:30:00 < t < 10:00:00
    ((9 * 3600 + 30 * 60) * SEC_US, false, 10 * 3600 * SEC_US, false),
    // morning: 09:30:00 <= t < 11:30:00
    ((9 * 3600 + 30 * 60) * SEC_US, true, (11 * 3600 + 30 * 60) * SEC_US, false),
    // afternoon: 13:00:00 <= t < 14:57:00
    (13 * 3600 * SEC_US, true, (14 * 3600 + 57 * 60) * SEC_US, false),
    // close30: 14:30:00 <= t < 14:57:00
    ((14 * 3600 + 30 * 60) * SEC_US, true, (14 * 3600 + 57 * 60) * SEC_US, false),
];

const WINDOW_TAGS: [&str; N_WINDOWS] = ["all", "open30", "morning", "afternoon", "close30"];
const METRIC_NAMES: [&str; PER_WINDOW] = [
    "exchtime_diff_mean",
    "exchtime_diff_code_mean",
    "exchtime_diff_std",
    "exchtime_diff_code_std",
    "exchtime_diff_skew",
    "exchtime_diff_code_skew",
    "volume_ratio_sum",
    "volume_ratio_mean",
    "volume_ratio_std",
    "volume_ratio_skew",
    "exchtime_diff_and_code",
];

pub fn names() -> Vec<String> {
    let mut out = Vec::with_capacity(N_OUT);
    for tag in WINDOW_TAGS.iter() {
        for m in METRIC_NAMES.iter() {
            out.push(format!("{}_{}", tag, m));
        }
    }
    out
}

/// polars 里空字符串的 Int64 是 null；用一个不会被真实委托号占用的哨兵值代替，
/// 这样「所有 null 归一组」的行为和 polars 一致。
const NULL_ORDER: i64 = i64::MIN;

struct Raw {
    idx: i64,
    ex: i64,
    vol: f32,
    flag: i32,
    ask: i64,
    bid: i64,
}

/// 解析一个字段的整数前缀（遇到非数字就停）。
#[inline]
fn parse_i64_prefix(b: &[u8]) -> i64 {
    let mut neg = false;
    let mut i = 0usize;
    while i < b.len() && b[i] == b' ' {
        i += 1;
    }
    if i < b.len() && (b[i] == b'-' || b[i] == b'+') {
        neg = b[i] == b'-';
        i += 1;
    }
    let mut val: i64 = 0;
    while i < b.len() {
        let c = b[i];
        if !c.is_ascii_digit() {
            break;
        }
        val = val * 10 + (c - b'0') as i64;
        i += 1;
    }
    if neg {
        -val
    } else {
        val
    }
}

#[inline]
fn parse_f64(b: &[u8]) -> f64 {
    std::str::from_utf8(b)
        .ok()
        .and_then(|s| s.trim().parse::<f64>().ok())
        .unwrap_or(0.0)
}

#[inline]
fn parse_order(b: &[u8]) -> i64 {
    if b.is_empty() {
        NULL_ORDER
    } else {
        parse_i64_prefix(b)
    }
}

/// 读一只股票当天的逐笔，按 CSV 的 index 列升序返回。
///
/// 列序（transaction.csv）：symbol,datetime,date,time,exchtime,localtime,index,price,
///                          volume,turnover,flag,order_type,func_code,ask_order,bid_order[,num]
/// 旧代码 read_csv 之后 sort(['index'])，文件行序并不总是 index 升序，所以这里也排一次。
fn read_stock(code: &str, date: i64) -> Option<Vec<Raw>> {
    let filename = format!("{}_{}_transaction.csv", code, date);
    let mut found = None;
    for root in data_paths::level2_roots() {
        let p = root.join(date.to_string()).join("transaction").join(&filename);
        if p.exists() {
            found = Some(p);
            break;
        }
    }
    let path = found?;
    let content = std::fs::read(&path).ok()?;

    let body_start = content
        .iter()
        .position(|&b| b == b'\n')
        .map(|p| p + 1)
        .unwrap_or(0);
    let mut out: Vec<Raw> = Vec::with_capacity(content.len() / 90);
    for line in content[body_start..].split(|&b| b == b'\n') {
        if line.is_empty() {
            continue;
        }
        let line = if line.last() == Some(&b'\r') {
            &line[..line.len() - 1]
        } else {
            line
        };
        if line.is_empty() {
            continue;
        }
        let mut fields: [&[u8]; 15] = [&[][..]; 15];
        let mut col = 0usize;
        let mut start = 0usize;
        for (i, &b) in line.iter().enumerate() {
            if b == b',' {
                if col < 15 {
                    fields[col] = &line[start..i];
                }
                col += 1;
                start = i + 1;
            }
        }
        if col < 15 {
            fields[col] = &line[start..];
        }
        // exchtime round(0) 之后是整数微秒；0 表示无效行（表头/异常），与旧框架一致地跳过
        let ex_f = parse_f64(fields[4]);
        if ex_f == 0.0 {
            continue;
        }
        out.push(Raw {
            idx: parse_i64_prefix(fields[6]),
            ex: ex_f.round() as i64,
            vol: parse_f64(fields[8]) as f32,
            flag: parse_i64_prefix(fields[10]) as i32,
            ask: parse_order(fields[13]),
            bid: parse_order(fields[14]),
        });
    }
    out.sort_by_key(|r| r.idx);
    Some(out)
}

/// 去重之后留下来的行（ask 或 bid 一路）。
struct Surv {
    ex: i64,
    vol: f32,
}

/// 一只股票在五个时段上的中间量。
struct CodeAgg {
    /// 每个时段的总成交量（volume_ratio 的分母），f64 顺序累加。
    total: [f64; N_WINDOWS],
    /// [时段][方向 0=ask 1=bid]，行按 index 升序。
    surv: [Vec<Surv>; N_WINDOWS * 2],
}

impl CodeAgg {
    fn new() -> Self {
        CodeAgg {
            total: [0.0; N_WINDOWS],
            surv: Default::default(),
        }
    }
}

fn slot(w: usize, side: usize) -> usize {
    w * 2 + side
}

/// 读一只股票、算五个时段的中间量。读不到文件返回 None。
fn process_code(code: &str, date: i64) -> Option<CodeAgg> {
    let raw = read_stock(code, date)?;
    if raw.is_empty() {
        return None;
    }
    let mut agg = CodeAgg::new();
    for w in 0..N_WINDOWS {
        let (lo, lo_incl, hi, hi_incl) = WINDOWS[w];
        // 窗口内、且 flag != 32 的行，保持 index 序
        let mut rows: Vec<(i64, f32, i32, i64, i64)> = Vec::new();
        for r in raw.iter() {
            if r.flag == 32 {
                continue;
            }
            let t = (r.ex + OFF_US).rem_euclid(DAY_US);
            let ok_lo = if lo_incl { t >= lo } else { t > lo };
            let ok_hi = if hi_incl { t <= hi } else { t < hi };
            if ok_lo && ok_hi {
                rows.push((r.ex, r.vol, r.flag, r.ask, r.bid));
            }
        }
        let mut tot = 0.0f64;
        for v in rows.iter() {
            tot += v.1 as f64;
        }
        agg.total[w] = tot;

        for side in 0..2usize {
            // unique(["code", key, "flag"])：保留第一次出现
            let mut seen: HashMap<(i64, i32), ()> = HashMap::new();
            let mut uniq: Vec<&(i64, f32, i32, i64, i64)> = Vec::with_capacity(rows.len());
            for r in rows.iter() {
                let k = if side == 0 { (r.3, r.2) } else { (r.4, r.2) };
                if seen.insert(k, ()).is_none() {
                    uniq.push(r);
                }
            }
            // filter(count(flag).over([code, key]) > 1)
            let mut cnt: HashMap<i64, u32> = HashMap::new();
            for r in uniq.iter() {
                let k = if side == 0 { r.3 } else { r.4 };
                *cnt.entry(k).or_insert(0) += 1;
            }
            let mut surv: Vec<Surv> = Vec::new();
            for r in uniq.iter() {
                let k = if side == 0 { r.3 } else { r.4 };
                if cnt[&k] > 1 {
                    surv.push(Surv { ex: r.0, vol: r.1 });
                }
            }
            agg.surv[slot(w, side)] = surv;
        }
    }
    Some(agg)
}

// ───────────────────────── 统计量（与参考实现逐式对齐） ─────────────────────────

/// numpy 的 pairwise summation（loops_utils.h）。
/// volume_ratio_sum 在旧代码里是 polars 的 f32 顺序和，这里用 numpy 的成对和，
/// 是为了让「两个股票的值是否严格相等」与实测结果一致。
fn np_pairwise_sum(a: &[f64]) -> f64 {
    const BLOCK: usize = 128;
    let n = a.len();
    if n < 8 {
        let mut res = 0.0f64;
        for &v in a {
            res += v;
        }
        return res;
    }
    if n <= BLOCK {
        let mut r = [a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]];
        let mut i = 8usize;
        let stop = n - (n % 8);
        while i < stop {
            for j in 0..8 {
                r[j] += a[i + j];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n {
            res += a[i];
            i += 1;
        }
        res
    } else {
        let mut n2 = n / 2;
        n2 -= n2 % 8;
        np_pairwise_sum(&a[..n2]) + np_pairwise_sum(&a[n2..])
    }
}

fn mean_of(v: &[f64]) -> Option<f64> {
    if v.is_empty() {
        return None;
    }
    let mut s = 0.0f64;
    for &x in v {
        s += x;
    }
    Some(s / v.len() as f64)
}

/// ddof=1，两趟算法（先均值再平方和），<2 个有效值返回 None（对应 polars 的 null）。
fn std_of(v: &[f64]) -> Option<f64> {
    if v.len() < 2 {
        return None;
    }
    let n = v.len() as f64;
    let mut mu = 0.0f64;
    for &x in v {
        mu += x;
    }
    mu /= n;
    let mut ss = 0.0f64;
    for &x in v {
        let d = x - mu;
        ss += d * d;
    }
    Some((ss / (n - 1.0)).sqrt())
}

/// 总体偏度 m3/m2^1.5（polars skew 的 bias=True）。m2<=0 返回 NaN。
fn skew_of(v: &[f64]) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    let n = v.len() as f64;
    let mut m = 0.0f64;
    for &x in v {
        m += x;
    }
    m /= n;
    let mut m2 = 0.0f64;
    let mut m3 = 0.0f64;
    for &x in v {
        let d = x - m;
        let dd = d * d;
        m2 += dd;
        m3 += dd * d;
    }
    m2 /= n;
    m3 /= n;
    if !(m2 > 0.0) {
        return f64::NAN;
    }
    m3 / (m2.sqrt() * m2)
}

/// 成对完整观测的 Pearson 相关；有效对数 <2 或某一侧方差为 0 返回 NaN。
fn corr_of(x: &[Option<f64>], y: &[Option<f64>]) -> f64 {
    let mut n = 0usize;
    let mut sx = 0.0f64;
    let mut sy = 0.0f64;
    for i in 0..x.len() {
        let (Some(a), Some(b)) = (x[i], y[i]) else {
            continue;
        };
        n += 1;
        sx += a;
        sy += b;
    }
    if n < 2 {
        return f64::NAN;
    }
    let nf = n as f64;
    let mx = sx / nf;
    let my = sy / nf;
    let mut cov = 0.0f64;
    let mut vx = 0.0f64;
    let mut vy = 0.0f64;
    for i in 0..x.len() {
        let (Some(a), Some(b)) = (x[i], y[i]) else {
            continue;
        };
        cov += (a - mx) * (b - my);
        vx += (a - mx) * (a - mx);
        vy += (b - my) * (b - my);
    }
    if vx <= 0.0 || vy <= 0.0 {
        return f64::NAN;
    }
    cov / (vx * vy).sqrt()
}

/// pandas Series.rank()：升序、并列取平均、NaN 保持 NaN。
fn rank_average(values: &[f64]) -> Vec<f64> {
    let n = values.len();
    let mut idx: Vec<usize> = (0..n).filter(|&i| !values[i].is_nan()).collect();
    idx.sort_by(|&a, &b| {
        values[a]
            .partial_cmp(&values[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut out = vec![f64::NAN; n];
    let mut i = 0usize;
    while i < idx.len() {
        let mut j = i + 1;
        while j < idx.len() && values[idx[j]] == values[idx[i]] {
            j += 1;
        }
        let avg = ((i + 1) as f64 + j as f64) / 2.0;
        for k in i..j {
            out[idx[k]] = avg;
        }
        i = j;
    }
    out
}

/// 横截面入口：一次算一天全市场，返回 (股票代码, 扁平因子值)。
///
/// 值按股票分块排列（每只股票连续 55 个），与 `dispatch_cross` 的
/// `vals.chunks(expected_len).zip(codes)` 对齐。
pub fn compute_cross(date: i64) -> Option<(Vec<String>, Vec<f32>)> {
    let codes = data_paths::list_stock_codes(date, "transaction");
    if codes.is_empty() {
        return None;
    }
    let aggs: Vec<Option<CodeAgg>> = codes
        .par_iter()
        .map(|code| process_code(code, date))
        .collect();
    if std::env::var("HM023_DEBUG").is_ok() {
        let ok_n = aggs.iter().filter(|a| a.is_some()).count();
        let s0: usize = aggs
            .iter()
            .filter_map(|a| a.as_ref())
            .map(|a| a.surv[0].len())
            .sum();
        eprintln!(
            "[hm023] codes={} ok={} surv(all,ask)={} raw_first={:?}",
            codes.len(),
            ok_n,
            s0,
            codes.first()
        );
    }

    let n_codes = codes.len();
    // 每个时段每个方向的 11 个量：先算股票 -> 值，再按并集排名
    let mut cols: Vec<Vec<Option<f64>>> = vec![vec![None; n_codes]; N_OUT];
    let mut used: Vec<bool> = vec![false; n_codes];

    // 分母按股票取（旧代码是 volume.sum().over('code')）
    let mut per_code_total: Vec<[f64; N_WINDOWS]> = vec![[0.0; N_WINDOWS]; n_codes];
    for (c, agg) in aggs.iter().enumerate() {
        if let Some(a) = agg {
            per_code_total[c] = a.total;
        }
    }

    for w in 0..N_WINDOWS {
        let mut per_side: Vec<Vec<(usize, [f64; PER_WINDOW])>> = Vec::with_capacity(2);
        for side in 0..2usize {
            // 拼成一天全市场的行：股票按代码升序、股票内部按 index 升序（与参考实现一致）
            let mut ci: Vec<u32> = Vec::new();
            let mut ex: Vec<i64> = Vec::new();
            let mut vol: Vec<f32> = Vec::new();
            for (c, agg) in aggs.iter().enumerate() {
                let Some(a) = agg else { continue };
                for s in a.surv[slot(w, side)].iter() {
                    ci.push(c as u32);
                    ex.push(s.ex);
                    vol.push(s.vol);
                }
            }
            // 稳定排序：并列时保持 (股票升序, index 升序)
            let mut perm: Vec<u32> = (0..ex.len() as u32).collect();
            perm.sort_by_key(|&i| ex[i as usize]);
            let ci_s: Vec<u32> = perm.iter().map(|&i| ci[i as usize]).collect();
            let ex_s: Vec<i64> = perm.iter().map(|&i| ex[i as usize]).collect();
            let vol_s: Vec<f32> = perm.iter().map(|&i| vol[i as usize]).collect();
            per_side.push(side_metrics_multi(&ci_s, &ex_s, &vol_s, &per_code_total, w));
        }
        let ask = &per_side[0];
        let bid = &per_side[1];
        let mut ask_map: HashMap<usize, [f64; PER_WINDOW]> = HashMap::new();
        let mut bid_map: HashMap<usize, [f64; PER_WINDOW]> = HashMap::new();
        for (c, v) in ask.iter() {
            ask_map.insert(*c, *v);
        }
        for (c, v) in bid.iter() {
            bid_map.insert(*c, *v);
        }
        let mut union: Vec<usize> = ask_map.keys().chain(bid_map.keys()).copied().collect();
        union.sort_unstable();
        union.dedup();
        for c in union.iter() {
            used[*c] = true;
        }
        for m in 0..PER_WINDOW {
            let va: Vec<f64> = union
                .iter()
                .map(|c| ask_map.get(c).map(|v| v[m]).unwrap_or(f64::NAN))
                .collect();
            let vb: Vec<f64> = union
                .iter()
                .map(|c| bid_map.get(c).map(|v| v[m]).unwrap_or(f64::NAN))
                .collect();
            let ra = rank_average(&va);
            let rb = rank_average(&vb);
            let col = &mut cols[w * PER_WINDOW + m];
            for (k, &c) in union.iter().enumerate() {
                let d = (ra[k] - rb[k]).abs();
                col[c] = if d.is_nan() { None } else { Some(d) };
            }
        }
    }

    // 输出：参与过任何一路任何时段的股票，按代码升序
    let keep: Vec<usize> = (0..n_codes).filter(|&c| used[c]).collect();
    let mut out = vec![f64::NAN; keep.len() * N_OUT];
    for (k, &c) in keep.iter().enumerate() {
        for m in 0..N_OUT {
            if let Some(v) = cols[m][c] {
                out[k * N_OUT + m] = v;
            }
        }
    }
    let flat: Vec<f32> = out.into_iter().map(|v| v as f32).collect();
    debug_assert_eq!(flat.len(), keep.len() * N_OUT);
    let out_codes: Vec<String> = keep.iter().map(|&c| codes[c].clone()).collect();
    Some((out_codes, flat))
}

/// 与 `side_metrics` 同义，但分母按股票各自取（旧代码 total_volume 是 over('code')）。
fn side_metrics_multi(
    rows_ci: &[u32],
    rows_ex: &[i64],
    rows_vol: &[f32],
    per_code_total: &[[f64; N_WINDOWS]],
    w: usize,
) -> Vec<(usize, [f64; PER_WINDOW])> {
    let n = rows_ex.len();
    if n == 0 {
        return Vec::new();
    }
    let codes_n = per_code_total.len();
    let mut dg: Vec<Option<f64>> = Vec::with_capacity(n);
    for i in 0..n {
        if i >= 1000 {
            dg.push(Some(((rows_ex[i] - rows_ex[i - 1000]) / SEC_US) as f64));
        } else {
            dg.push(None);
        }
    }
    let mut dc: Vec<Option<f64>> = Vec::with_capacity(n);
    let mut last_ex: Vec<i64> = vec![i64::MIN; codes_n];
    let mut seen_code: Vec<bool> = vec![false; codes_n];
    for i in 0..n {
        let ci = rows_ci[i] as usize;
        if seen_code[ci] {
            dc.push(Some(((rows_ex[i] - last_ex[ci]) / SEC_US) as f64));
        } else {
            seen_code[ci] = true;
            dc.push(None);
        }
        last_ex[ci] = rows_ex[i];
    }

    let mut counts = vec![0u32; codes_n];
    for &c in rows_ci {
        counts[c as usize] += 1;
    }
    let mut starts = vec![0usize; codes_n + 1];
    for c in 0..codes_n {
        starts[c + 1] = starts[c] + counts[c] as usize;
    }
    let mut cursor = starts.clone();
    let mut order: Vec<u32> = vec![0; n];
    for i in 0..n {
        let c = rows_ci[i] as usize;
        order[cursor[c]] = i as u32;
        cursor[c] += 1;
    }

    let mut out: Vec<(usize, [f64; PER_WINDOW])> = Vec::new();
    for c in 0..codes_n {
        let (s, e) = (starts[c], starts[c + 1]);
        if s == e {
            continue;
        }
        let idx = &order[s..e];
        let xg_all: Vec<Option<f64>> = idx.iter().map(|&i| dg[i as usize]).collect();
        let xc_all: Vec<Option<f64>> = idx.iter().map(|&i| dc[i as usize]).collect();
        let xg: Vec<f64> = xg_all.iter().filter_map(|v| *v).collect();
        let xc: Vec<f64> = xc_all.iter().filter_map(|v| *v).collect();
        let tv = per_code_total[c][w];
        let vr: Vec<f64> = idx
            .iter()
            .map(|&i| rows_vol[i as usize] as f64 / tv)
            .collect();
        let (Some(m_dg_mean), Some(m_dc_mean), Some(m_dg_std), Some(m_dc_std)) = (
            mean_of(&xg),
            mean_of(&xc),
            std_of(&xg),
            std_of(&xc),
        ) else {
            continue;
        };
        if vr.is_empty() {
            continue;
        }
        let vals = [
            m_dg_mean,
            m_dc_mean,
            m_dg_std,
            m_dc_std,
            skew_of(&xg),
            skew_of(&xc),
            np_pairwise_sum(&vr),
            mean_of(&vr).unwrap_or(f64::NAN),
            std_of(&vr).unwrap_or(f64::NAN),
            skew_of(&vr),
            corr_of(&xg_all, &xc_all),
        ];
        out.push((c, vals));
    }
    out
}

/// hm23 是横截面系列，没有逐股实现。
pub fn compute_stock(_code: &str, _date: i64) -> Option<Vec<f32>> {
    None
}
