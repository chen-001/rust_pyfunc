//! hm41「野火」（旧框架 design_whatever_out.TickDataBase）的 Rust 复现。
//!
//! 旧文件：/home/chenzongwei/pythoncode/成果hm41_野火/hm41_okay.py
//!   run_base 里的 Go.get_factor，参数 get_many=0 / read_full=0 / change_time=1，
//!   也就是「一整天的盘口」一张表，五个时段由旧代码自己 between_time 切。
//!   change_time=1 在旧框架里只做 exchtime 的单位换算（epoch 微秒 → Timestamp），
//!   没有 adjust_afternoon 的下午平移，所以窗口是真实钟点、两端都含。
//!
//! ════════════════════════════════════════════════════════════════════
//! 旧代码逐步对照（写代码前先抄在这里，方便复核）
//! ════════════════════════════════════════════════════════════════════
//! 1) df = read_market_data_single()，然后 df = df[df.last_prc != 0]
//! 2) read_market_data_single_price_vol_pair：把 10 档 ask_prc / ask_vol 熔成长表，
//!    price==0 的行丢掉，number 就是档位 1..10，最后按 (exchtime, number) 升序排序
//!    （多键 sort_values 走 np.lexsort，是稳定排序）。bids 同理。
//! 3) 五个时段（真实钟点的闭区间）：
//!      allday   [09:30, 14:57]   open30 [09:30, 10:00]   morning [09:30, 11:30]
//!      afternoon[13:00, 14:57]   close30[14:30, 14:57]
//! 4) 每个时段里切两档：cut_num=6（档位 1~5，名字后缀 _num5）与 cut_num=11（档位 1~10，
//!    后缀 _num10）。每一次 sing1 产出 36 个量（见下面的 METRIC 清单）。
//! 5) 排序后的 asks/bids 各自：
//!      first = drop_duplicates(subset="price", keep="first")  按 price 去重留最早一条
//!      last  = drop_duplicates(subset="price", keep="last")   留最晚一条
//!    然后按 price 对齐算 36 个量。
//!
//! 指标清单（sing1 返回的顺序，也是 names 里每个 (时段, 档位) 块的前 34 个名字）：
//!   0 asks_dura_mean | 1 bids_dura_mean | 2 asks_dura_std | 3 bids_dura_std
//!   4 asks_dura_max  | 5 bids_dura_max  | 6 asks_dura_skew| 7 bids_dura_skew
//!   8 asks_dura_corr | 9 bids_dura_corr
//!   10 asks_first_sum| 11 asks_last_sum | 12 bids_first_sum| 13 bids_last_sum
//!   14 asks_dura_sum | 15 bids_dura_sum            ← 这两个名字里**没有**
//!   16 ab_first      | 17 ab_last
//!   18 asks_dura_sum_abs | 19 bids_dura_sum_abs | 20 ab_first_abs | 21 ab_last_abs
//!   22 asks_bids_dura_sum_plus | 23 asks_bids_dura_sum_minus
//!   24 asks_dura_sum_rate | 25 bids_dura_sum_rate | 26 ab_first_rate | 27 ab_last_rate
//!   28 corr_asks_first | 29 corr_asks_last | 30 corr_bids_first | 31 corr_bids_last
//!   32 corr_asks_first_last | 33 corr_bids_first_last
//!   34 corr_first_asks_bids | 35 corr_last_asks_bids
//!
//! 每个量的定义：
//!   dura      = (last.exchtime - first.exchtime) 的秒数，按 price 对齐
//!   dura_mean/std/max/skew = 上面这个序列的统计量
//!   dura_corr = corr(price, dura)
//!   first_sum/last_sum = 该侧 vol 列之和（first 用 first 的，last 用 last 的）
//!   dura_sum  = last_sum - first_sum
//!   ab_first  = asks_first_sum - bids_first_sum；ab_last 同理
//!   *_abs / plus / minus = 对应的绝对值与和差绝对值
//!   dada_sum_rate = dura_sum / (last_sum + first_sum)
//!   ab_first_rate = asks_first_sum / (asks_first_sum + bids_first_sum)
//!   ab_last_rate  = asks_last_sum  / (asks_last_sum  + bids_last_sum)
//!       ！！注意不是 (asks-bids)/(asks+bids)，旧代码写的是占比；写错会让值从 0.5 变成 -0.003
//!   corr_asks_first = asks_first.corr().iloc[0,1]
//!       ！！asks_first 的列是 [exchtime, number, vol]，所以这个量是
//!         corr(exchtime 的纳秒整数值, number)，不是 corr(price, ...)。
//!         pandas 2.x 的 DataFrame.corr 会把 datetime64[ns] 当 float64（= 自 epoch 的纳秒数）
//!         参与计算，这里必须照做。
//!   corr_asks_first_last = corr(price, last_vol - first_vol)，按 price 内连接对齐
//!   corr_first_asks_bids = corr(price, asks_first_vol - bids_first_vol)，按 price 内连接
//!
//! ════════════════════════════════════════════════════════════════════
//! ！！这里有一个必须原样复刻的旧代码错位
//! ════════════════════════════════════════════════════════════════════
//! 旧代码里 sing1 返回的是 **36** 个值，但 names 里每个 (时段, 档位) 只声明了 **34** 个名字
//! （漏掉了 asks_dura_sum 与 bids_dura_sum）。旧框架 cal_factor_one_stock /
//! cal_factor_one_day 都是 `zip(res, names_single)`，zip 按短的截断，所以
//! **从第 14 位（ab_first）起，名字与值整体错位 2 格**，最后两个值被丢掉。
//! 举例：存进因子库的 `ab_first_num10_open30` 实际装的是 asks_dura_sum。
//!
//! 为了让新链路存进因子库的列值和旧链路逐个一致，这里：
//!   · names() 返回 340 个名字（= 旧 names_single 原样）
//!   · compute_stock() 返回完整的 360 个原始值（= 旧 get_factor 的原始返回）
//! pipeline 侧按 expected_result_length = 340 截断，正好等于旧 zip 的截断行为；
//! 验证脚本比的是 compute_stock 的原始返回 360 对旧 get_factor 的 360。
//! **不要**顺手把 360 改成 340，那样值会整体前移两格、和因子库对不上。
//!
//! 精度：全程 f64 自解析 CSV（与 hm013 同一套读法）。盘口价格默认是 0.0，
//! 空单元格按 NaN，`price != 0` 的判断对 NaN 成立（pandas 里 NaN != 0 也是 True），
//! 所以 NaN 行要照样保留。

use std::collections::HashSet;
use std::fs::File;
use std::io::Read;

/// 盘口 CSV 的列索引（0 基，与 fast_csv_reader 的常量一致）
const MKT_COL_EXCHTIME: usize = 4;
const MKT_COL_LAST_PRC: usize = 6;
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

/// 五个时段（真实钟点的微秒闭区间，旧代码用 between_time，两端都含）
const SESSIONS: [(&str, i64, i64); 5] = [
    ("allday", LD_0930, LD_1457),
    ("open30", LD_0930, LD_1000),
    ("morning", LD_0930, LD_1130),
    ("afternoon", LD_1300, LD_1457),
    ("close30", LD_1430, LD_1457),
];

/// 每个 (时段, 档位) 组声明的因子名个数（旧 names_single 每块的个数）
const PER_BLOCK: usize = 34;
/// sing1 实际返回的原始值个数（旧代码每块多算了 2 个，名字里没有）
const RAW_PER_SING: usize = 36;
/// names() 的总长度 = 5 时段 × 2 档位 × 34
pub const N_NAMES: usize = PER_BLOCK * 2 * 5;

/// 两个档位切分：cut_num=6 → 档位 1~5（num5）；cut_num=11 → 档位 1~10（num10）
const CUTS: [i64; 2] = [6, 11];
const CUT_SUFFIX: [&str; 2] = ["num5", "num10"];

/// names 里每块的 34 个名字，顺序与旧 names 完全一致
const BLOCK_NAMES: [&str; PER_BLOCK] = [
    "asks_dura_mean",
    "bids_dura_mean",
    "asks_dura_std",
    "bids_dura_std",
    "asks_dura_max",
    "bids_dura_max",
    "asks_dura_skew",
    "bids_dura_skew",
    "asks_dura_corr",
    "bids_dura_corr",
    "asks_first_sum",
    "asks_last_sum",
    "bids_first_sum",
    "bids_last_sum",
    "ab_first",
    "ab_last",
    "asks_dura_sum_abs",
    "bids_dura_sum_abs",
    "ab_first_abs",
    "ab_last_abs",
    "asks_bids_dura_sum_plus",
    "asks_bids_dura_sum_minus",
    "asks_dura_sum_rate",
    "bids_dura_sum_rate",
    "ab_first_rate",
    "ab_last_rate",
    "corr_asks_first",
    "corr_asks_last",
    "corr_bids_first",
    "corr_bids_last",
    "corr_asks_first_last",
    "corr_bids_first_last",
    "corr_first_asks_bids",
    "corr_last_asks_bids",
];

/// 因子名：时段在外层，档位（num5 再 num10）在里层，与旧 names 的拼接顺序一致。
pub fn names() -> Vec<String> {
    let mut out = Vec::with_capacity(N_NAMES);
    for (sess, _, _) in SESSIONS.iter() {
        for cut in 0..2 {
            for b in BLOCK_NAMES.iter() {
                out.push(format!("{}_{}_{}", b, CUT_SUFFIX[cut], sess));
            }
        }
    }
    debug_assert_eq!(out.len(), N_NAMES);
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
// 盘口 CSV 的 f64 读取
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

/// 一条盘口快照（只留计算要用的列）
#[derive(Clone, Copy, Default)]
struct Snapshot {
    /// epoch 微秒 + 8 小时
    time_us: i64,
    ask_prc: [f64; 10],
    ask_vol: [f64; 10],
    bid_prc: [f64; 10],
    bid_vol: [f64; 10],
}

/// 读 market_data CSV（f64）。行集与 pandas 读整张表一致；exchtime==0 的行也读进来，
/// 它们会被时段窗口滤掉（旧代码路径完全一样）。
fn read_market_f64(code: &str, date: i64) -> Option<Vec<Snapshot>> {
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

    let mut out: Vec<Snapshot> = Vec::with_capacity(content.len() / 200 + 1);
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

        // 旧代码 df = df[df.last_prc != 0]（NaN != 0 也是 True，这里保持一致）
        if parse_f64_at(f[MKT_COL_LAST_PRC]) == 0.0 {
            continue;
        }

        let mut row = Snapshot {
            time_us: parse_i64_at(f[MKT_COL_EXCHTIME]) + 8 * 3600 * SEC_US,
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
// 长表（熔成长表并按 exchtime, number 稳定排序）
// ============================================================================

/// 长表的一行：一个快照的一个档位
#[derive(Clone, Copy)]
struct BookRow {
    time_us: i64,
    number: i64,
    price: f64,
    vol: f64,
}

/// 把快照摊成长表：档位 1..10，price==0 的行丢掉，再按 (time_us, number) 稳定排序。
///
/// 等价于旧代码的 melt → `asks[asks.price != 0]` → `sort_values(["exchtime","number"])`。
/// 多键 sort_values 走 np.lexsort（稳定），所以同等键的先后 = 原表行序；
/// 下面按「快照顺序 × 档位顺序」入表，同一档位内仍是原表行序，与 melt 的顺序一致。
fn build_book(snaps: &[Snapshot], is_ask: bool) -> Vec<BookRow> {
    let mut out: Vec<BookRow> = Vec::with_capacity(snaps.len() * 10);
    for s in snaps.iter() {
        for k in 0..10usize {
            let (p, v) = if is_ask {
                (s.ask_prc[k], s.ask_vol[k])
            } else {
                (s.bid_prc[k], s.bid_vol[k])
            };
            // pandas: df[df.price != 0]，NaN != 0 为 True，所以 NaN 行保留
            if p == 0.0 {
                continue;
            }
            out.push(BookRow {
                time_us: s.time_us,
                number: (k + 1) as i64,
                price: p,
                vol: v,
            });
        }
    }
    out.sort_by_key(|r| (r.time_us, r.number));
    out
}

// ============================================================================
// pandas 语义的小工具
// ============================================================================

/// pandas DataFrame.corr() / Series.corr() 的 Pearson：逐行复刻 pandas/_libs/algos.pyx
/// 的 nancorr（Welford 在线算法，min_periods 默认 1）。
/// 正常数据上与一次遍历公式只差 1e-16，但退化情形（某列恒定）差别很大，必须一致。
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

/// pandas Series.mean()：跳过 NaN；全 NaN / 空返回 NaN
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

/// pandas Series.std()（ddof=1）：两遍算法，n<2 返回 NaN
fn nanstd(v: &[f64]) -> f64 {
    let mut n = 0usize;
    let mut s = 0.0f64;
    for &x in v {
        if !x.is_nan() {
            n += 1;
            s += x;
        }
    }
    if n < 2 {
        return f64::NAN;
    }
    let avg = s / n as f64;
    let mut sq = 0.0f64;
    for &x in v {
        if !x.is_nan() {
            let d = avg - x;
            sq += d * d;
        }
    }
    (sq / (n as f64 - 1.0)).sqrt()
}

/// pandas Series.max()：跳过 NaN；空返回 NaN
fn nanmax(v: &[f64]) -> f64 {
    let mut m = f64::NEG_INFINITY;
    let mut n = 0usize;
    for &x in v {
        if !x.is_nan() {
            n += 1;
            if x > m {
                m = x;
            }
        }
    }
    if n == 0 {
        f64::NAN
    } else {
        m
    }
}

/// pandas Series.sum()：跳过 NaN；空返回 0.0
fn nansum(v: &[f64]) -> f64 {
    let mut s = 0.0f64;
    for &x in v {
        if !x.is_nan() {
            s += x;
        }
    }
    s
}

#[inline]
fn zero_out_fperr(x: f64) -> f64 {
    if x.abs() < 1e-14 {
        0.0
    } else {
        x
    }
}

/// pandas Series.skew()：调整后的 Fisher-Pearson G1，
///   result = (n*sqrt(n-1)/(n-2)) * (m3 / m2^1.5)
/// 其中 m2/m3 是二阶/三阶中心矩，且各自先按 _zero_out_fperr 把 |x|<1e-14 抹成 0；
/// m2==0 时返回 0；n<3 返回 NaN。
fn nanskew(v: &[f64]) -> f64 {
    let vals: Vec<f64> = v.iter().cloned().filter(|x| !x.is_nan()).collect();
    let n = vals.len();
    if n == 0 {
        return f64::NAN;
    }
    let count = n as f64;
    let mean = vals.iter().sum::<f64>() / count;
    let mut m2 = 0.0f64;
    let mut m3 = 0.0f64;
    for &x in vals.iter() {
        let a = x - mean;
        m2 += a * a;
        m3 += a * a * a;
    }
    let m2 = zero_out_fperr(m2);
    let m3 = zero_out_fperr(m3);
    if count < 3.0 {
        return f64::NAN;
    }
    if m2 == 0.0 {
        return 0.0;
    }
    (count * (count - 1.0).sqrt() / (count - 2.0)) * (m3 / m2.powf(1.5))
}

// ============================================================================
// 单块（一个时段 + 一个档位切分）的 36 个原始值
// ============================================================================

/// 按 price 去重后的两侧（first = 留最早一条，last = 留最晚一条），按 price 对齐。
struct Dedup {
    price: Vec<f64>,
    time_us: Vec<i64>,
    number: Vec<i64>,
    vol: Vec<f64>,
}

/// drop_duplicates(subset="price", keep="first"|"last")，输入必须已按
/// (time_us, number) 升序排好。行序就是去重后的顺序。
fn dedup_price(rows: &[BookRow], keep_first: bool) -> Dedup {
    let mut seen: HashSet<u64> = HashSet::with_capacity(rows.len());
    let mut d = Dedup {
        price: Vec::new(),
        time_us: Vec::new(),
        number: Vec::new(),
        vol: Vec::new(),
    };
    if keep_first {
        for r in rows.iter() {
            if seen.insert(r.price.to_bits()) {
                d.price.push(r.price);
                d.time_us.push(r.time_us);
                d.number.push(r.number);
                d.vol.push(r.vol);
            }
        }
    } else {
        for r in rows.iter().rev() {
            if seen.insert(r.price.to_bits()) {
                d.price.push(r.price);
                d.time_us.push(r.time_us);
                d.number.push(r.number);
                d.vol.push(r.vol);
            }
        }
    }
    d
}

/// 两侧按 price 内连接：返回 (left_price, left_vol, right_vol)，顺序按 left。
fn inner_join_vol(lp: &[f64], lv: &[f64], rp: &[f64], rv: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    use std::collections::HashMap;
    let mut m: HashMap<u64, f64> = HashMap::with_capacity(rp.len());
    for i in 0..rp.len() {
        m.insert(rp[i].to_bits(), rv[i]);
    }
    let mut a = Vec::new();
    let mut b = Vec::new();
    let mut c = Vec::new();
    for i in 0..lp.len() {
        if let Some(&v) = m.get(&lp[i].to_bits()) {
            a.push(lp[i]);
            b.push(lv[i]);
            c.push(v);
        }
    }
    (a, b, c)
}

/// 一侧的 4 个统计量与 3 个相关系数。
struct Side {
    dura_mean: f64,
    dura_std: f64,
    dura_max: f64,
    dura_skew: f64,
    dura_corr: f64,
    first_sum: f64,
    last_sum: f64,
    corr_first: f64,
    corr_last: f64,
    corr_first_last: f64,
}

fn side_metrics(f: &Dedup, l: &Dedup) -> Side {
    // ！！必须按 price 对齐，不能按位置：keep="first" 与 keep="last" 留下来的行序不同
    // （某个 price 最早出现的位置可能晚于另一个 price 最晚出现的位置）。
    // 旧代码 (asks_last.exchtime - asks_first.exchtime) 是按 index（= price）对齐的。
    use std::collections::HashMap;
    let mut lidx: HashMap<u64, usize> = HashMap::with_capacity(l.price.len());
    for i in 0..l.price.len() {
        lidx.insert(l.price[i].to_bits(), i);
    }
    let n = f.price.len();
    let mut dura: Vec<f64> = Vec::with_capacity(n);
    for i in 0..n {
        // (last.exchtime - first.exchtime).dt.total_seconds()，按 price 取对应那一行
        match lidx.get(&f.price[i].to_bits()) {
            Some(&j) => dura.push((l.time_us[j] - f.time_us[i]) as f64 / 1e6),
            None => dura.push(f64::NAN),
        }
    }
    // corr(exchtime, number)：exchtime 在 pandas 里是 datetime64[ns]，
    // DataFrame.corr 会把它当成「自 epoch 的纳秒数」的 float64。
    let f_ns: Vec<f64> = f.time_us.iter().map(|&t| t as f64 * 1000.0).collect();
    let l_ns: Vec<f64> = l.time_us.iter().map(|&t| t as f64 * 1000.0).collect();
    let f_num: Vec<f64> = f.number.iter().map(|&x| x as f64).collect();
    let l_num: Vec<f64> = l.number.iter().map(|&x| x as f64).collect();

    let (jp, jf, jl) = inner_join_vol(&f.price, &f.vol, &l.price, &l.vol);
    let mut diff: Vec<f64> = Vec::with_capacity(jp.len());
    for i in 0..jp.len() {
        diff.push(jl[i] - jf[i]);
    }

    Side {
        dura_mean: nanmean(&dura),
        dura_std: nanstd(&dura),
        dura_max: nanmax(&dura),
        dura_skew: nanskew(&dura),
        dura_corr: pearson(&f.price, &dura),
        first_sum: nansum(&f.vol),
        last_sum: nansum(&l.vol),
        corr_first: pearson(&f_ns, &f_num),
        corr_last: pearson(&l_ns, &l_num),
        corr_first_last: pearson(&jp, &diff),
    }
}

/// 旧代码的 sing1：窗口内两侧都非空才算，返回 36 个值；否则返回 34 个 NaN。
fn sing1(asks_win: &[BookRow], bids_win: &[BookRow], cut: i64) -> Vec<f64> {
    if asks_win.is_empty() || bids_win.is_empty() {
        return vec![f64::NAN; PER_BLOCK];
    }
    let af_rows: Vec<BookRow> = asks_win.iter().cloned().filter(|r| r.number < cut).collect();
    let bf_rows: Vec<BookRow> = bids_win.iter().cloned().filter(|r| r.number < cut).collect();

    let af = dedup_price(&af_rows, true);
    let al = dedup_price(&af_rows, false);
    let bf = dedup_price(&bf_rows, true);
    let bl = dedup_price(&bf_rows, false);

    let a = side_metrics(&af, &al);
    let b = side_metrics(&bf, &bl);

    // corr_first_asks_bids / corr_last_asks_bids：按 price 内连接后取 vol 之差
    let (p1, av, bv) = inner_join_vol(&af.price, &af.vol, &bf.price, &bf.vol);
    let mut d1: Vec<f64> = Vec::with_capacity(p1.len());
    for i in 0..p1.len() {
        d1.push(av[i] - bv[i]);
    }
    let (p2, av2, bv2) = inner_join_vol(&al.price, &al.vol, &bl.price, &bl.vol);
    let mut d2: Vec<f64> = Vec::with_capacity(p2.len());
    for i in 0..p2.len() {
        d2.push(av2[i] - bv2[i]);
    }
    let corr_first_asks_bids = pearson(&p1, &d1);
    let corr_last_asks_bids = pearson(&p2, &d2);

    let asks_dura_sum = a.last_sum - a.first_sum;
    let bids_dura_sum = b.last_sum - b.first_sum;
    let ab_first = a.first_sum - b.first_sum;
    let ab_last = a.last_sum - b.last_sum;

    vec![
        a.dura_mean,
        b.dura_mean,
        a.dura_std,
        b.dura_std,
        a.dura_max,
        b.dura_max,
        a.dura_skew,
        b.dura_skew,
        a.dura_corr,
        b.dura_corr,
        a.first_sum,
        a.last_sum,
        b.first_sum,
        b.last_sum,
        asks_dura_sum,
        bids_dura_sum,
        ab_first,
        ab_last,
        asks_dura_sum.abs(),
        bids_dura_sum.abs(),
        ab_first.abs(),
        ab_last.abs(),
        (asks_dura_sum + bids_dura_sum).abs(),
        (asks_dura_sum - bids_dura_sum).abs(),
        asks_dura_sum / (a.last_sum + a.first_sum),
        bids_dura_sum / (b.last_sum + b.first_sum),
        // ！！ab_first_rate 不是 (a-b)/(a+b)，旧代码写的是 a/(a+b)：
        //     ab_first_rate = asks_first_sum / (asks_first_sum + bids_first_sum)
        //   实测这一条写错会让值从 0.498 变成 -0.003，属于必错项。
        a.first_sum / (a.first_sum + b.first_sum),
        a.last_sum / (a.last_sum + b.last_sum),
        a.corr_first,
        a.corr_last,
        b.corr_first,
        b.corr_last,
        a.corr_first_last,
        b.corr_first_last,
        corr_first_asks_bids,
        corr_last_asks_bids,
    ]
}

/// 逐股的一天：算不出来返回 None（上游补 NaN，保证因子个数恒定）。
///
/// 返回的是**旧 get_factor 的原始 360 个值**（每块 36 个 × 5 时段 × 2 档位），
/// 不是 340。原因见文件头「旧代码错位」一节，千万别顺手截断。
pub fn compute_stock(code: &str, date: i64) -> Option<Vec<f32>> {
    let snaps = read_market_f64(code, date)?;
    if snaps.is_empty() {
        return None;
    }
    let asks_all = build_book(&snaps, true);
    let bids_all = build_book(&snaps, false);
    if asks_all.is_empty() || bids_all.is_empty() {
        return None;
    }

    let mut out: Vec<f64> = Vec::with_capacity(RAW_PER_SING * 2 * 5);
    for (_sess, lo, hi) in SESSIONS.iter() {
        let aw: Vec<BookRow> = asks_all
            .iter()
            .cloned()
            .filter(|r| in_span(r.time_us, *lo, *hi))
            .collect();
        let bw: Vec<BookRow> = bids_all
            .iter()
            .cloned()
            .filter(|r| in_span(r.time_us, *lo, *hi))
            .collect();
        for cut in CUTS.iter() {
            out.extend(sing1(&aw, &bw, *cut));
        }
    }
    Some(out.into_iter().map(|v| v as f32).collect())
}

/// hm41 全是逐股因子，没有横截面版本。
pub fn compute_cross(_date: i64) -> Option<(Vec<String>, Vec<f32>)> {
    None
}
