//! hm5「整点订单」（旧框架 design_whatever_out.TickDataBase）的 Rust 复现。
//!
//! 旧文件：/home/chenzongwei/pythoncode/成果hm5_整点订单/hm5_okay.py
//!   run_base 里的 Go.get_factor，参数 get_many=0 / read_full=0 / change_time=1，
//!   也就是「一整天的逐笔」一张表，窗口由旧代码自己切。
//!
//! ════════════════════════════════════════════════════════════════════
//! 旧代码逐步对照（写代码前先抄在这里，方便复核）
//! ════════════════════════════════════════════════════════════════════
//! 1) df = read_trade_data_single(with_retreat=0) → 丢掉 flag==32 的撤单行。
//!    change_time=1 只做 deal_data：把 exchtime 从「epoch 微秒」换成 Timestamp，
//!    不做 adjust_afternoon 的下午平移，所以 Rust 侧按真实钟点切窗口。
//! 2) df = df.set_index(['exchtime']).between_time('09:30:00','14:57:00')，两端都含。
//!    再 assign minute / second / microsecond（取自 DatetimeIndex）。
//! 3) 五个时段（都是 df 的子集，闭区间）：
//!      all       09:30:00 ~ 14:57:00
//!      open30    09:30:00 ~ 10:00:00
//!      close30   14:30:00 ~ 14:57:00
//!      morning   09:30:00 ~ 11:30:00
//!      afternoon 13:00:00 ~ 14:57:00
//!    主买 act_buy = flag==66，主卖 act_sell = flag==83（每个时段各切一份）。
//! 4) 四个周期，各给一个「整点」子集：
//!      1s  microsecond < 250000      （每秒的前 0.25 秒）
//!      1m  second < 15               （每分钟的前 15 秒）
//!      5m  minute % 5 == 0
//!      15m minute % 15 == 0
//! 5) 每个 (周期, 时段) 出 6 个指标，先算原值：
//!      cjbs_diff_x      = 子集行数 / 时段行数
//!      cjbs_diff_buy_x  = 子集主买行数 / 时段主买行数
//!      cjbs_diff_sell_x = 子集主卖行数 / 时段主卖行数
//!      mys_rate_x       = 子集 turnover 之和 / 时段 turnover 之和
//!      mys_buy_rate_x   = 同上，限主买
//!      mys_sell_rate_x  = 同上，限主卖
//! 6) **只有 1s 那 30 个值**再做 abs(值 - 0.25)（1m / 5m / 15m 的原值直接输出）。
//!    这是旧代码的主要信号：整点（每秒前 0.25 秒）的成交占比偏离 0.25 多少。
//! 7) 输出顺序 = 1s(30) + 1m(30) + 5m(30) + 15m(30)，每个周期内按时段
//!    all / open30 / close30 / morning / afternoon，每个时段内按上面 6 个指标的先后。
//!    共 120 个，与 names_single 逐个对齐。
//!
//! names_save（要保存的 base 名）：旧文件里 44 个，已原样搬到 hm5_okay_database.py。
//!
//! ════════════════════════════════════════════════════════════════════
//! 几个必须踩准的点
//! ════════════════════════════════════════════════════════════════════
//! · 撤单过滤自己按数值做，不靠 read_trade_fast_inner 的 with_retreat：
//!   库里那条判断按字节比 "32"，而 2025 年的明细写的是 "32.0"，过滤不生效。
//! · turnover 用 f64 解析，不用 TradeRecord 的 f32。旧代码里 turnover 是 float64
//!   求和，1s 那 30 个值又要减 0.25 再取绝对值，比例越接近 0.25 误差被放得越大。
//!   实测 600519@20250304 用 f32 时 120 个值里最大相对误差 8.5e-6（阈值 1e-4），
//!   虽仍能过，但 min|值| 只有 1.19e-4，余量不到一个数量级；换 f64 后误差降到
//!   求和顺序带来的 1e-16 量级。所以本文件自己按 f64 解析 CSV。
//! · 时间用整微秒比较就够：pandas 的 between_time 走 indexer_between_time，
//!   比较的是 *微秒*（_get_time_micros），纳秒被截掉。pandas 从「微秒整数」
//!   造出来的 Timestamp 因为有 to_timedelta(float) 的一步，个别值会差 1 微秒，
//!   但只可能发生在小数部分不是二进制精确值的时候；而窗口边界与
//!   microsecond<250000 这些判断的临界值都是 .000000 / .250000 秒，它们是精确
//!   可表示的，所以整微秒比较与旧代码逐值一致（四组对照实测最大相对误差 0.0）。
//! · 空 turnover 单元格在 pandas 里是 NaN，.sum() 会跳过它，但该行仍计入 shape[0]。
//!   这里同样：计数照加，求和跳过 NaN。

use std::fs::File;
use std::io::Read;

/// 逐笔 CSV 的列索引（0 基，与 fast_csv_reader 一致；2018 年及以前末尾多一列 num，不影响前 15 列）
const TRD_COL_EXCHTIME: usize = 4;
const TRD_COL_TURNOVER: usize = 9;
const TRD_COL_FLAG: usize = 10;
const TRD_NCOL: usize = 15;

const DAY_US: i64 = 86_400_000_000;
const SEC_US: i64 = 1_000_000;
const MIN_US: i64 = 60 * SEC_US;

const D_0930: i64 = (9 * 3600 + 30 * 60) * SEC_US;
const D_1000: i64 = 10 * 3600 * SEC_US;
const D_1130: i64 = (11 * 3600 + 30 * 60) * SEC_US;
const D_1300: i64 = 13 * 3600 * SEC_US;
const D_1430: i64 = (14 * 3600 + 30 * 60) * SEC_US;
const D_1457: i64 = (14 * 3600 + 57 * 60) * SEC_US;

/// 五个时段：后缀、窗口（毫秒级微秒闭区间）。
/// 顺序必须与旧 names_single 的拼接顺序一致（all / open30 / close30 / morning / afternoon）。
const SESSIONS: [(&str, i64, i64); 5] = [
    ("", D_0930, D_1457),
    ("_open30", D_0930, D_1000),
    ("_close30", D_1430, D_1457),
    ("_morning", D_0930, D_1130),
    ("_afternoon", D_1300, D_1457),
];

/// 四个周期，顺序即输出顺序
const PERIODS: [&str; 4] = ["1s", "1m", "5m", "15m"];

/// 每个时段 6 个指标，顺序即输出顺序
const METRICS: [&str; 6] = [
    "cjbs_diff",
    "cjbs_diff_buy",
    "cjbs_diff_sell",
    "mys_rate",
    "mys_buy_rate",
    "mys_sell_rate",
];

const PER_PERIOD: usize = METRICS.len() * SESSIONS.len(); // 30
pub const N_OUT: usize = PER_PERIOD * PERIODS.len(); // 120

pub fn names() -> Vec<String> {
    let mut out = Vec::with_capacity(N_OUT);
    for period in PERIODS.iter() {
        for (suf, _, _) in SESSIONS.iter() {
            for m in METRICS.iter() {
                out.push(format!("{}_{}{}", m, period, suf));
            }
        }
    }
    out
}

/// 北京时间的当日微秒数（time_us 已含 8 小时偏移）
#[inline]
fn day_us(t: i64) -> i64 {
    t.rem_euclid(DAY_US)
}

/// 该行是否落在第 p 个周期的「整点」子集里。
/// 0=1s（每秒前 0.25 秒）、1=1m（每分钟前 15 秒）、2=5m、3=15m。
#[inline]
fn in_period(d: i64, p: usize) -> bool {
    match p {
        0 => d % SEC_US < 250_000,
        1 => (d / SEC_US) % 60 < 15,
        2 => ((d / MIN_US) % 60) % 5 == 0,
        _ => ((d / MIN_US) % 60) % 15 == 0,
    }
}

/// pandas 的 sum 跳过 NaN；空单元格解析出来的 NaN 不计入求和（该行的计数照算）。
#[inline]
fn add(x: &mut f64, v: f64) {
    if !v.is_nan() {
        *x += v;
    }
}

/// 一个时段的累加器。
///
/// base 六个：时段内总行数、主买行数、主卖行数、总成交额、主买成交额、主卖成交额。
/// per[p] 同样六个：该周期「整点」子集里的对应量。
#[derive(Clone, Copy, Default)]
struct Acc {
    base: [f64; 6],
    per: [[f64; 6]; 4],
}

/// 一条逐笔成交（只留计算要用的列）
#[derive(Clone, Copy, Default)]
struct TradeRow {
    time_us: i64,
    turnover: f64,
    flag: i64,
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

/// 读 transaction CSV（f64 的 turnover）。
///
/// 行集与 pandas 读整张表一致：表头之后每一行都算（含 exchtime 为空／0 的行，
/// 它们会被窗口过滤掉，但保留才与旧代码的行集一致）。
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
            turnover: parse_f64_at(f[TRD_COL_TURNOVER]),
            flag: parse_i64_at(f[TRD_COL_FLAG]),
        });
    }
    Some(out)
}

/// 逐股的一天。文件读不到返回 None（上游补整条 NaN，保证因子个数恒定）。
pub fn compute_stock(code: &str, date: i64) -> Option<Vec<f32>> {
    let all = read_trade_f64(code, date)?;
    // 撤单过滤按数值做（"32.0" 也要能过滤掉）
    let rows: Vec<&TradeRow> = all.iter().filter(|r| r.flag != 32).collect();

    let mut acc = [Acc::default(); 5];
    for r in rows.iter() {
        let d = day_us(r.time_us);
        let is_buy = r.flag == 66;
        let is_sell = r.flag == 83;
        for (si, (_, lo, hi)) in SESSIONS.iter().enumerate() {
            if d < *lo || d > *hi {
                continue;
            }
            let a = &mut acc[si];
            a.base[0] += 1.0;
            add(&mut a.base[3], r.turnover);
            if is_buy {
                a.base[1] += 1.0;
                add(&mut a.base[4], r.turnover);
            }
            if is_sell {
                a.base[2] += 1.0;
                add(&mut a.base[5], r.turnover);
            }
            for pi in 0..4 {
                if !in_period(d, pi) {
                    continue;
                }
                let p = &mut a.per[pi];
                p[0] += 1.0;
                add(&mut p[3], r.turnover);
                if is_buy {
                    p[1] += 1.0;
                    add(&mut p[4], r.turnover);
                }
                if is_sell {
                    p[2] += 1.0;
                    add(&mut p[5], r.turnover);
                }
            }
        }
    }

    let mut out: Vec<f32> = Vec::with_capacity(N_OUT);
    for (pi, _) in PERIODS.iter().enumerate() {
        for (si, _) in SESSIONS.iter().enumerate() {
            let a = &acc[si];
            let p = &a.per[pi];
            // 顺序：cjbs_diff / cjbs_diff_buy / cjbs_diff_sell / mys_rate / mys_buy_rate / mys_sell_rate
            let raw = [
                p[0] / a.base[0],
                p[1] / a.base[1],
                p[2] / a.base[2],
                p[3] / a.base[3],
                p[4] / a.base[4],
                p[5] / a.base[5],
            ];
            for v in raw.iter() {
                // 只有 1s 做 abs(值 - 0.25) 的变换
                let v = if pi == 0 { (v - 0.25).abs() } else { *v };
                out.push(v as f32);
            }
        }
    }
    debug_assert_eq!(out.len(), N_OUT);
    Some(out)
}

/// hm5 全是逐股因子，没有横截面版本。
pub fn compute_cross(_date: i64) -> Option<(Vec<String>, Vec<f32>)> {
    None
}
