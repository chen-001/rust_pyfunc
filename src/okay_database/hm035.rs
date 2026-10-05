//! hm035「显现后交易」—— 旧框架（design_whatever_out.TickDataBase）的 Rust 复现。
//!
//! 旧文件：/home/chenzongwei/pythoncode/成果hm35_显现后交易/hm35_okay.py
//!
//! 旧代码要点（逐条对齐）：
//!
//! 【调用参数】get_many=0、read_full=0、change_time=1 → read_trade_data_single 返回单个 DataFrame，
//!   列含 exchtime（已换成北京时间 Timestamp）、price、volume、flag；with_retreat=0 先丢掉 flag==32。
//!
//! 【排序】df = df.sort_values(['price','exchtime']) —— 价格升序、同价按时间升序（稳定排序）。
//!   下面所有窗口都是从这张排好序的 df 上切出来的，所以每个窗口内部仍是「价格升序」的顺序。
//!   ！！这一点决定了 two 处取值，不能只按时间想：
//!     min_time = df.exchtime.iloc[0]      = 该窗口里**价格最低**那一组中时间最早的一笔
//!     max_time = df0.exchtime.iloc[0]     = 该窗口里价格最高那一组中，**按帧顺序**排在最前的一笔
//!   单个窗口是一整块，所以「价格最低组」等价于「先找最小价、再在其中取最早时间」，
//!   不需要真的排序。唯一例外是 silent 窗口：它是两段 between_time 结果 concat 起来的，
//!   帧顺序是「10:00~11:30 整块」在前、「13:01~14:30 整块」在后，
//!   两段各自价格有序但整体不是，所以 iloc[0] 取的是前一段里的最低价那一笔；
//!   max_time 则取「先出现最高价的那一段」里最高价组的最早一笔。见以下实现里的 blocks。
//!
//! 【每个窗口的 5 个指标】one(df)：
//!   df.volume = df.volume / df.volume.sum()      ← 用**该窗口自己**的总量归一化
//!   one_time = min(min_time, max_time)，two_time = max(...)
//!   last_volume = Σ volume[t >= two_time]
//!   min_volume  = Σ volume[t <= min_time]
//!   max_volume  = Σ volume[t <= max_time]
//!   one_volume  = Σ volume[t <= one_time]
//!   two_volume  = Σ volume[one_time <= t <= two_time]
//!   窗口为空（0 行）时返回 5 个 NaN。
//!
//! 【窗口与分组】6 个时间段 × 3 种口径（full=全部、66=主买、83=主卖），共 18 组，每组 5 个值：
//!   allday    = 全天不筛（连 09:25 集合竞价那些也算）
//!   open30    = [09:30:00, 10:00:00]
//!   morning   = [09:30:00, 11:30:00]
//!   afternoon = [13:00:00, 14:57:00]
//!   close30   = [14:30:00, 14:57:00]
//!   silent    = [10:00:00, 11:30:00] ∪ [13:01:00, 14:30:00]（两块拼接）
//!   边界闭区间、精确到微秒。注意这里**不做**下午平移，全部用真实钟点。
//!   full 口径不筛 flag，其余按 flag==66 / flag==83 筛。
//!
//! names 顺序与旧代码 names_single 一致：先 allday 的 full/66/83，再 open30 …，最后 silent，
//! 每组内顺序是 last_volume / min_volume / max_volume / one_volume / two_volume，共 90 个。
//!
//! ════════════════════════════════════════════════════════════════════
//! 精度：price 必须按 f64 读
//! ════════════════════════════════════════════════════════════════════
//! min_time / max_time 是靠 `price == price.max()` 这种**严格相等**分出来的，
//! 用 f32 读 price 会让两个本来不同的价格塌成同一个数，把不属于最高价组的成交算进去，
//! 时间点跟着错，输出整个歪掉。所以这里自己按 f64 解析 CSV，列索引与 fast_csv_reader 一致。

use std::fs::File;
use std::io::Read;

/// 逐笔 CSV 的列索引（0 基，与 fast_csv_reader 的常量一致）
const TRD_COL_EXCHTIME: usize = 4;
const TRD_COL_PRICE: usize = 7;
const TRD_COL_VOLUME: usize = 8;
const TRD_COL_FLAG: usize = 10;
const TRD_NCOL: usize = 15;

/// 秒 → 微秒
const US: i64 = 1_000_000;

/// 6 个时间段，每个由 1~2 个「块」组成；块是有序的，silent 的两块不能合并看
struct Window {
    name: &'static str,
    spans: &'static [(i64, i64)],
}

const WINDOWS: [Window; 6] = [
    Window {
        name: "allday",
        spans: &[(i64::MIN, i64::MAX)],
    },
    Window {
        name: "open30",
        spans: &[(34200 * US, 36000 * US)],
    },
    Window {
        name: "morning",
        spans: &[(34200 * US, 41400 * US)],
    },
    Window {
        name: "afternoon",
        spans: &[(46800 * US, 53820 * US)],
    },
    Window {
        name: "close30",
        spans: &[(52200 * US, 53820 * US)],
    },
    Window {
        // 两段拼接，帧顺序 = 前一段全部在前
        name: "silent",
        spans: &[(36000 * US, 41400 * US), (46860 * US, 52200 * US)],
    },
];

/// 三种口径：full 不筛 flag，66/83 各筛一种
const KINDS: [(&str, i32); 3] = [("full", -1), ("66", 66), ("83", 83)];

const METRIC_NAMES: [&str; 5] = [
    "last_volume",
    "min_volume",
    "max_volume",
    "one_volume",
    "two_volume",
];

const PER_GROUP: usize = 5;
pub const N_OUT: usize = PER_GROUP * 6 * 3;

/// 一条逐笔成交（排序视图里用得到的字段）
#[derive(Clone, Copy)]
struct R3 {
    /// 北京时间的当日微秒数
    t: i64,
    price: f64,
    volume: f64,
}

pub fn names() -> Vec<String> {
    let mut out = Vec::with_capacity(N_OUT);
    for w in WINDOWS.iter() {
        for (kind, _) in KINDS.iter() {
            for m in METRIC_NAMES.iter() {
                out.push(format!("{}_{}_{}", m, w.name, kind));
            }
        }
    }
    out
}

#[inline]
fn day_us(time_us: i64) -> i64 {
    time_us.rem_euclid(86_400_000_000)
}

/// 一组块（silent 两块，其余一块）算 5 个指标。
/// 规则见文件头：min_time 取第一块最低价组的最早时间；max_time 取第一个含最高价的块里最早的一笔。
fn one(blocks: &[Vec<R3>]) -> [f64; PER_GROUP] {
    let mut tot = 0.0f64;
    let mut pmax = f64::NEG_INFINITY;
    let mut n = 0usize;
    for b in blocks.iter() {
        for r in b.iter() {
            n += 1;
            if r.volume.is_finite() {
                tot += r.volume;
            }
            if r.price.is_finite() && r.price > pmax {
                pmax = r.price;
            }
        }
    }
    if n == 0 {
        return [f64::NAN; PER_GROUP];
    }
    // ---- min_time：第一块里的最低价组 ----
    let mut min_time = f64::NAN;
    for b in blocks.iter() {
        if b.is_empty() {
            continue;
        }
        let mut pmin = f64::INFINITY;
        for r in b.iter() {
            if r.price.is_finite() && r.price < pmin {
                pmin = r.price;
            }
        }
        let mut t = i64::MAX;
        for r in b.iter() {
            if r.price == pmin && r.t < t {
                t = r.t;
            }
        }
        if t != i64::MAX {
            min_time = t as f64;
        }
        break;
    }
    // ---- max_time：第一个含最高价的行所在的块，该组里最早的一笔 ----
    let mut max_time = f64::NAN;
    for b in blocks.iter() {
        if b.is_empty() {
            continue;
        }
        let mut t = i64::MAX;
        for r in b.iter() {
            if r.price == pmax && r.t < t {
                t = r.t;
            }
        }
        if t != i64::MAX {
            max_time = t as f64;
            break;
        }
    }
    // 两边有一边是 NaT 时 pandas 的 min/max 都给 NaT，所有比较都 False → 5 个和都是 0
    let (lo, hi) = if min_time.is_finite() && max_time.is_finite() {
        (min_time.min(max_time), min_time.max(max_time))
    } else {
        (f64::NAN, f64::NAN)
    };
    let (mut last, mut minv, mut maxv, mut onev, mut twov) = (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for b in blocks.iter() {
        for r in b.iter() {
            let v = r.volume / tot; // pandas 先整列归一化再求和，NaN 会被 sum 跳过
            if !v.is_finite() {
                continue;
            }
            let t = r.t as f64;
            if t >= hi {
                last += v;
            }
            if t <= min_time {
                minv += v;
            }
            if t <= max_time {
                maxv += v;
            }
            if t <= lo {
                onev += v;
            }
            if t >= lo && t <= hi {
                twov += v;
            }
        }
    }
    [last, minv, maxv, onev, twov]
}

// ============================================================================
// 逐笔 CSV 的 f64 读取
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

/// 空单元格按 pandas 的读法给 NaN；其余交给 std 的正确舍入解析（与 pandas 的 strtod 一致）
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

/// 读 transaction CSV：返回 (当日微秒数, price, volume, flag)，保持文件行序。
/// 撤单不在这里过滤（下面按数值筛 flag==32），与 pandas 读整张表的行集一致。
fn read_trade_f64(code: &str, date: i64) -> Option<Vec<(i64, f64, f64, i32)>> {
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

    let mut out = Vec::with_capacity(content.len() / 120 + 1);
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
        let t_us = day_us(parse_i64_at(f[TRD_COL_EXCHTIME]) + 8 * 3600 * US);
        out.push((
            t_us,
            parse_f64_at(f[TRD_COL_PRICE]),
            parse_f64_at(f[TRD_COL_VOLUME]),
            parse_i64_at(f[TRD_COL_FLAG]) as i32,
        ));
    }
    Some(out)
}

/// 逐股入口：算不出来返回 None（上游会补 NaN，保证因子个数恒定）。
pub fn compute_stock(code: &str, date: i64) -> Option<Vec<f32>> {
    let trades = read_trade_f64(code, date)?;
    // 旧代码 with_retreat=0 → 丢掉 flag==32（撤单）。不能靠库里的字节比较（2025 年前后写的是 "32.0"）。
    let rows: Vec<(i64, f64, f64, i32)> = trades.into_iter().filter(|r| r.3 != 32).collect();
    if rows.is_empty() {
        return None;
    }

    let mut out = Vec::with_capacity(N_OUT);
    for w in WINDOWS.iter() {
        for (_, flag) in KINDS.iter() {
            // 每个 span 一个块，块内保持文件行序（one() 只依赖集合与块的先后，不依赖块内顺序）
            let blocks: Vec<Vec<R3>> = w
                .spans
                .iter()
                .map(|&(lo, hi)| {
                    rows.iter()
                        .filter(|r| r.0 >= lo && r.0 <= hi && (*flag < 0 || r.3 == *flag))
                        .map(|r| R3 {
                            t: r.0,
                            price: r.1,
                            volume: r.2,
                        })
                        .collect()
                })
                .collect();
            out.extend(one(&blocks).iter().map(|&v| v as f32));
        }
    }
    debug_assert_eq!(out.len(), N_OUT);
    Some(out)
}

/// hm35 全是逐股因子，没有横截面版本。
pub fn compute_cross(_date: i64) -> Option<(Vec<String>, Vec<f32>)> {
    None
}
