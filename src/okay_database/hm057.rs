//! hm57「本福特定律（早一版）」——旧框架（design_whatever_out.TickDataBase）的 Rust 复现。
//!
//! 旧文件：/home/chenzongwei/pythoncode/成果hm57_hm69_本福特定律/hm57_okay.py
//! 数据来源：Level2 逐笔。get_many=0 / read_full=0 / change_time=1，逐股系列。
//!
//! ════════════════════════════════════════════════════════════════════
//! 一、旧 get_factor 逐步对照（写代码前先抄在这里，方便复核）
//! ════════════════════════════════════════════════════════════════════
//! 1) trade_data = read_trade_data_single(with_retreat=1)
//!      → **保留** flag==32 的撤单行（后面要用撤单）。
//! 2) trade_data = trade_data.set_index('exchtime').between_time('09:30:00','14:57:00')
//!      注意：**没有** dwo.adjust_afternoon！窗口用的是真实钟点，闭区间，微秒精度。
//!      09:15~09:25 的集合竞价与 14:57~15:00 的收盘竞价都被这一刀切掉。
//! 3) 另切四个子集 open30 / morning / afternoon / close30（真实钟点）：
//!      09:30~10:00 / 09:30~11:30 / 13:00~14:57 / 14:30~14:57
//! 4) 关键一处旧代码的写法：**`one(df)` 收了参数 df 但从来不 use，函数体里全程引用外层
//!    的 trade_data**。所以 res1..res5 = one(trade_data)+one(open30)+one(morning)
//!    +one(afternoon)+one(close30) 这五块**数值完全相同**，只是名字带了不同的时段后缀。
//!    对账必须照这个来（跑一遍 280 个值、复制五份），不能改回「各自切窗口再算」。
//! 5) one() 里：10 条序列（成交量/成交金额/主买量/主卖量/买单编号/卖单编号/
//!    买单撤单量/卖单撤单量/买单撤单编号/卖单撤单编号）各 20 个统计量，
//!    再加 4 组按位相减（ask&bid_order_diff[_abs]、buy_cancels&sell_cancels_diff[_abs]），
//!    共 280 个。序列的取法与 20 个统计量的算式与 hm069 逐条相同（见下面的说明）。
//! 6) get_factor 外面套了 try/except 把异常吞掉 → 出错就当算不出来（整条 None）。
//!
//! ════════════════════════════════════════════════════════════════════
//! 二、与 hm069 共用的三处要点（照抄，别自己另推）
//! ════════════════════════════════════════════════════════════════════
//! · 时间戳：change_time=1 走的是 pandas 的
//!     pd.to_timedelta(exchtime_us / 1e6, unit='s') + Timestamp('1970-01-01 08:00:00')
//!   即 base=向零截断、frac=rint(frac*1e9)/1e9、ns=base*1e9+(i64)(frac*1e9)，再 +8 小时。
//!   这条链带 ±114ns 噪声（实测 000001@20250303 有 72787/76488 行与「微秒×1000」不同，
//!   约一半为负），而 between_time 比的是**向下取整到微秒**的当日时刻，
//!   不照抄就会改变「正好卡在 09:30:00.000000 / 14:57:00.000000」那些行的窗口归属。
//! · 首字符：positive.astype(str).str.slice(0,1) 取的是字符串首字符，不是首位有效数字。
//!   x>=1e16 或 x<1e-4 时 Python/numpy 用科学计数法（首字符=首位有效数字）；
//!   1e-4<=x<1 时是 '0.001' 这种定点写法（首字符 '0'）；其余取整数部分首位。
//!   '0'/'inf'/'nan' 这些首字符不进 '1'..'9' 的直方图柱，但**要计入分母**。
//! · 统计量：Series.corr/autocorr 走 np.corrcoef 两遍法；DataFrame.corr 走
//!   libalgos.nancorr 的 Welford（除数写成 sqrt(ssqdmx*ssqdmy)，掩码用 isfinite）；
//!   std 是 nanvar 两遍法 ddof=1；skew 是 nanskew（count==2 时除零给 ±inf，照原样）；
//!   median 去 NaN 后排序取中位。空序列 → 20 个全 NaN。
//!
//! ════════════════════════════════════════════════════════════════════
//! 三、精度与已知差异
//! ════════════════════════════════════════════════════════════════════
//! 求和/均值用顺序累加，与 numpy 的成对求和差在 1e-16 相对量级；四组对账实测 1400 个值
//! 全部一致。volume/turnover/order 号只用「是否 > 0」与「首字符」，不做量级算术。

use crate::data_paths;

const N_STATS: usize = 20;
const N_SERIES: usize = 10;
const N_EXTRA: usize = 4;
const PER_WINDOW: usize = (N_SERIES + N_EXTRA) * N_STATS; // 280
const N_WINDOWS: usize = 5;
/// 总共 1400 个（五份 280，数值相同、名字不同）
pub const N_OUT: usize = PER_WINDOW * N_WINDOWS;

const STAT_NAMES: [&str; N_STATS] = [
    "correlation",
    "deviation_trend",
    "deviation_trend_abs",
    "deviation_std",
    "deviation_abs_std",
    "deviation_skew",
    "deviation_abs_skew",
    "deviation_max",
    "deviation_abs_max",
    "deviation_min",
    "deviation_abs_min",
    "deviation_abs_mean",
    "deviation_median",
    "deviation_abs_median",
    "deviation_autocorr1",
    "deviation_abs_autocorr1",
    "deviation_autocorr2",
    "deviation_abs_autocorr2",
    "deviation_autocorr3",
    "deviation_abs_autocorr3",
];

const SERIES_NAMES: [&str; N_SERIES] = [
    "volume",
    "turnover",
    "volume66",
    "volume83",
    "bid_order",
    "ask_order",
    "buy_cancels",
    "sell_cancels",
    "buy_cancels_order",
    "sell_cancels_order",
];

const EXTRA_NAMES: [&str; N_EXTRA] = [
    "ask&bid_order_diff",
    "ask&bid_order_diff_abs",
    "buy_cancels&sell_cancels_diff",
    "buy_cancels&sell_cancels_diff_abs",
];

const WINDOW_TAGS: [&str; N_WINDOWS] = ["all", "_open30", "_morning", "_afternoon", "_close30"];

pub fn names() -> Vec<String> {
    let mut base: Vec<String> = Vec::with_capacity((N_SERIES + N_EXTRA) * N_STATS);
    for s in SERIES_NAMES.iter() {
        for st in STAT_NAMES.iter() {
            base.push(format!("{}_{}", s, st));
        }
    }
    for e in EXTRA_NAMES.iter() {
        for st in STAT_NAMES.iter() {
            base.push(format!("{}_{}", e, st));
        }
    }
    let mut out = Vec::with_capacity(N_OUT);
    for (w, tag) in WINDOW_TAGS.iter().enumerate() {
        for b in base.iter() {
            if w == 0 {
                out.push(format!("all_{}", b));
            } else {
                out.push(format!("{}{}", b, tag));
            }
        }
    }
    out
}

const DAY_NS: i64 = 86_400_000_000_000;
const OFFSET_NS: i64 = 8 * 3600 * 1_000_000_000;
const US: i64 = 1_000;
const SEC_US: i64 = 1_000_000;
const D_0930: i64 = (9 * 3600 + 30 * 60) * SEC_US;
const D_1457: i64 = (14 * 3600 + 57 * 60) * SEC_US;

/// pandas 的「epoch 微秒（f64）→ 纳秒（i64）」（conversion.pyx cast_from_unit_vectorized）。
/// 入参 `val` 是 **exchtime_us / 1e6（秒）**，不是微秒。
#[inline]
fn pandas_ns(val: f64) -> i64 {
    let base = val as i64;
    let frac = val - base as f64;
    let frac = (frac * 1e9).round_ties_even() / 1e9;
    base * 1_000_000_000 + (frac * 1e9) as i64
}

/// 一个交易日里落在 [09:30:00, 14:57:00] 的逐笔（保留撤单行）。
struct Row {
    price: f64,
    volume: f64,
    turnover: f64,
    flag: i64,
    bid: i64,
    ask: i64,
}

fn read_rows(code: &str, date: i64) -> Option<Vec<Row>> {
    const COL_EXCHTIME: usize = 4;
    const COL_PRICE: usize = 7;
    const COL_VOLUME: usize = 8;
    const COL_TURNOVER: usize = 9;
    const COL_FLAG: usize = 10;
    const COL_ASK_ORDER: usize = 13;
    const COL_BID_ORDER: usize = 14;

    let filename = format!("{}_{}_transaction.csv", code, date);
    let mut data = None;
    for root in data_paths::level2_roots() {
        let path = root
            .join(date.to_string())
            .join("transaction")
            .join(&filename);
        if let Ok(bytes) = std::fs::read(&path) {
            data = Some(bytes);
            break;
        }
    }
    let data = data?;

    let mut rows: Vec<Row> = Vec::new();
    for (lineno, line) in data.split(|&b| b == b'\n').enumerate() {
        if lineno == 0 || line.is_empty() {
            continue;
        }
        let mut fields: [&[u8]; 15] = [&[][..]; 15];
        let mut commas = 0usize;
        let mut start = 0usize;
        for (i, &b) in line.iter().enumerate() {
            if b == b',' {
                if commas < 15 {
                    fields[commas] = &line[start..i];
                }
                commas += 1;
                start = i + 1;
            }
        }
        if commas < 15 {
            fields[commas] = &line[start..];
        }
        if commas + 1 < 15 {
            continue;
        }
        let exch_us = us_prefix(fields[COL_EXCHTIME]);
        if exch_us <= 0 {
            continue; // 空 exchtime 在 pandas 里是 NaT，between_time 选不上
        }
        let ns = pandas_ns(exch_us as f64 / 1e6) + OFFSET_NS;
        let tod = ns.rem_euclid(DAY_NS) / US;
        // 唯一的窗口：真实钟点 [09:30:00, 14:57:00]，闭区间、微秒精度
        if !(D_0930..=D_1457).contains(&tod) {
            continue;
        }
        rows.push(Row {
            price: num(fields[COL_PRICE]),
            volume: num(fields[COL_VOLUME]),
            turnover: num(fields[COL_TURNOVER]),
            flag: int_prefix(fields[COL_FLAG]),
            bid: int_prefix(fields[COL_BID_ORDER]),
            ask: int_prefix(fields[COL_ASK_ORDER]),
        });
    }
    Some(rows)
}

fn trim_ascii(raw: &[u8]) -> &[u8] {
    let mut s = 0usize;
    let mut e = raw.len();
    while s < e && raw[s].is_ascii_whitespace() {
        s += 1;
    }
    while e > s && raw[e - 1].is_ascii_whitespace() {
        e -= 1;
    }
    &raw[s..e]
}

fn int_prefix(raw: &[u8]) -> i64 {
    let b = trim_ascii(raw);
    let mut i = 0usize;
    let mut neg = false;
    if i < b.len() && (b[i] == b'-' || b[i] == b'+') {
        neg = b[i] == b'-';
        i += 1;
    }
    let mut v: i64 = 0;
    while i < b.len() && b[i].is_ascii_digit() {
        v = v * 10 + (b[i] - b'0') as i64;
        i += 1;
    }
    if neg {
        -v
    } else {
        v
    }
}

fn us_prefix(raw: &[u8]) -> i64 {
    let b = trim_ascii(raw);
    if !b.contains(&b'e') && !b.contains(&b'E') {
        return int_prefix(b);
    }
    std::str::from_utf8(b)
        .ok()
        .and_then(|s| s.parse::<f64>().ok())
        .map(|v| v.round() as i64)
        .unwrap_or(0)
}

fn num(raw: &[u8]) -> f64 {
    let b = trim_ascii(raw);
    if b.is_empty() {
        return f64::NAN;
    }
    parse_f64_pandas(b)
}

fn pow10_table() -> &'static [f64; 309] {
    static T: std::sync::OnceLock<[f64; 309]> = std::sync::OnceLock::new();
    T.get_or_init(|| {
        let mut t = [1.0f64; 309];
        for (k, v) in t.iter_mut().enumerate() {
            *v = format!("1e{}", k).parse::<f64>().unwrap();
        }
        t
    })
}

/// pandas C 解析器 `precise_xstrtod` 的逐行复刻（与 hm033/hm069/hm090 里的同名函数一致）。
fn parse_f64_pandas(b: &[u8]) -> f64 {
    const MAX_DIGITS: usize = 17;
    let n = b.len();
    let mut p = 0usize;
    while p < n && matches!(b[p], b' ' | b'\t' | b'\r' | b'\n') {
        p += 1;
    }
    let mut negative = false;
    if p < n && (b[p] == b'-' || b[p] == b'+') {
        negative = b[p] == b'-';
        p += 1;
    }
    let mut number = 0.0f64;
    let mut exponent: i32 = 0;
    let mut num_digits = 0usize;
    let mut num_decimals = 0usize;
    while p < n && b[p].is_ascii_digit() {
        if num_digits < MAX_DIGITS {
            number = number * 10.0 + (b[p] - b'0') as f64;
            num_digits += 1;
        } else {
            exponent += 1;
        }
        p += 1;
    }
    if p < n && b[p] == b'.' {
        p += 1;
        while num_digits < MAX_DIGITS && p < n && b[p].is_ascii_digit() {
            number = number * 10.0 + (b[p] - b'0') as f64;
            p += 1;
            num_digits += 1;
            num_decimals += 1;
        }
        if num_digits >= MAX_DIGITS {
            while p < n && b[p].is_ascii_digit() {
                p += 1;
            }
        }
        exponent -= num_decimals as i32;
    }
    if num_digits == 0 {
        return 0.0;
    }
    if negative {
        number = -number;
    }
    if p < n && (b[p] | 0x20) == b'e' {
        p += 1;
        let mut eneg = false;
        if p < n && (b[p] == b'-' || b[p] == b'+') {
            eneg = b[p] == b'-';
            p += 1;
        }
        let mut nn: i32 = 0;
        let mut nd = 0usize;
        while nd < MAX_DIGITS && p < n && b[p].is_ascii_digit() {
            nn = nn * 10 + (b[p] - b'0') as i32;
            nd += 1;
            p += 1;
        }
        if eneg {
            exponent -= nn;
        } else {
            exponent += nn;
        }
    }
    if exponent > 308 {
        f64::INFINITY
    } else if exponent > 0 {
        number * pow10_table()[exponent as usize]
    } else if exponent < -308 {
        if exponent < -616 {
            0.0
        } else {
            let t = pow10_table();
            (number / t[(-308 - exponent) as usize]) / t[308]
        }
    } else {
        number / pow10_table()[(-exponent) as usize]
    }
}

/// `str(x)[0]`：见文件头第二节。
#[inline]
fn first_char(x: f64) -> u8 {
    if !x.is_finite() {
        return b'i';
    }
    if x >= 1.0e16 || x < 0.0001 {
        let s = format!("{:e}", x);
        return s.as_bytes()[0];
    }
    if x >= 1.0 {
        let n = x as u64;
        let k = n.ilog10();
        return b'0' + (n / 10u64.pow(k)) as u8;
    }
    b'0'
}

#[inline]
fn valid_count(x: &[f64]) -> usize {
    x.iter().filter(|v| !v.is_nan()).count()
}

fn nanmean(x: &[f64]) -> f64 {
    let c = valid_count(x);
    if c == 0 {
        return f64::NAN;
    }
    let s: f64 = x.iter().map(|v| if v.is_nan() { 0.0 } else { *v }).sum();
    s / c as f64
}

fn nanvar(x: &[f64], ddof: usize) -> f64 {
    let c = valid_count(x);
    if c <= ddof {
        return f64::NAN;
    }
    let avg = nanmean(x);
    let mut s = 0.0f64;
    for v in x.iter() {
        if !v.is_nan() {
            let d = avg - *v;
            s += d * d;
        }
    }
    s / (c - ddof) as f64
}

fn nanstd(x: &[f64], ddof: usize) -> f64 {
    nanvar(x, ddof).sqrt()
}

fn nanskew(x: &[f64]) -> f64 {
    let c = valid_count(x);
    let mean = if c == 0 {
        f64::NAN
    } else {
        let s: f64 = x.iter().map(|v| if v.is_nan() { 0.0 } else { *v }).sum();
        s / c as f64
    };
    let mut m2 = 0.0f64;
    let mut m3 = 0.0f64;
    for v in x.iter() {
        let adj = if v.is_nan() { 0.0 } else { *v - mean };
        m2 += adj * adj;
        m3 += adj * adj * adj;
    }
    if m2.abs() < 1e-14 {
        m2 = 0.0;
    }
    if m3.abs() < 1e-14 {
        m3 = 0.0;
    }
    if m2 == 0.0 {
        return 0.0;
    }
    let cnt = c as f64;
    (cnt * (cnt - 1.0).sqrt() / (cnt - 2.0)) * (m3 / m2.powf(1.5))
}

fn nanmax(x: &[f64]) -> f64 {
    let mut m = f64::NAN;
    for v in x.iter() {
        if !v.is_nan() && (m.is_nan() || *v > m) {
            m = *v;
        }
    }
    m
}

fn nanmin(x: &[f64]) -> f64 {
    let mut m = f64::NAN;
    for v in x.iter() {
        if !v.is_nan() && (m.is_nan() || *v < m) {
            m = *v;
        }
    }
    m
}

fn nanmedian(x: &[f64]) -> f64 {
    let mut v: Vec<f64> = x.iter().cloned().filter(|z| !z.is_nan()).collect();
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
}

/// Series.corr → nanops.nancorr → np.corrcoef（两遍法），成对丢 NaN。
fn corrcoef(x: &[f64], y: &[f64]) -> f64 {
    let mut xs: Vec<f64> = Vec::new();
    let mut ys: Vec<f64> = Vec::new();
    for i in 0..x.len() {
        if !x[i].is_nan() && !y[i].is_nan() {
            xs.push(x[i]);
            ys.push(y[i]);
        }
    }
    let n = xs.len();
    if n < 2 {
        return f64::NAN;
    }
    let nf = n as f64;
    let mx = xs.iter().sum::<f64>() / nf;
    let my = ys.iter().sum::<f64>() / nf;
    let mut sxx = 0.0f64;
    let mut syy = 0.0f64;
    let mut sxy = 0.0f64;
    for i in 0..n {
        let dx = xs[i] - mx;
        let dy = ys[i] - my;
        sxx += dx * dx;
        syy += dy * dy;
        sxy += dx * dy;
    }
    if sxx <= 0.0 || syy <= 0.0 {
        return f64::NAN;
    }
    sxy / (sxx.sqrt() * syy.sqrt())
}

/// DataFrame.corr → libalgos.nancorr 的 Welford 版（掩码用 isfinite）。
fn corr_welford(x: &[f64], y: &[f64]) -> f64 {
    let mut nobs: i64 = 0;
    let mut meanx = 0.0f64;
    let mut meany = 0.0f64;
    let mut ssqdmx = 0.0f64;
    let mut ssqdmy = 0.0f64;
    let mut covxy = 0.0f64;
    for i in 0..x.len() {
        if x[i].is_finite() && y[i].is_finite() {
            let vx = x[i];
            let vy = y[i];
            nobs += 1;
            let dx = vx - meanx;
            let dy = vy - meany;
            meanx += 1.0 / nobs as f64 * dx;
            meany += 1.0 / nobs as f64 * dy;
            ssqdmx += (vx - meanx) * dx;
            ssqdmy += (vy - meany) * dy;
            covxy += (vx - meanx) * dy;
        }
    }
    let divisor = (ssqdmx * ssqdmy).sqrt();
    if divisor != 0.0 {
        covxy / divisor
    } else {
        f64::NAN
    }
}

fn autocorr(dev: &[f64], lag: usize) -> f64 {
    let n = dev.len();
    let x: Vec<f64> = dev[lag..].to_vec();
    let y: Vec<f64> = dev[..n - lag].to_vec();
    corrcoef(&x, &y)
}

/// 一条序列在一个时段里的 20 个统计量。
fn series_stats(vals: &[f64]) -> [f64; N_STATS] {
    let mut out = [f64::NAN; N_STATS];
    let mut total = 0usize;
    let mut hits = [0usize; 9];
    for &v in vals.iter() {
        if v > 0.0 {
            total += 1;
            let c = first_char(v);
            if (b'1'..=b'9').contains(&c) {
                hits[(c - b'1') as usize] += 1;
            }
        }
    }
    if total == 0 {
        return out;
    }
    let mut obs = [f64::NAN; 9];
    for k in 0..9 {
        if hits[k] > 0 {
            obs[k] = hits[k] as f64 / total as f64;
        }
    }
    let theo: [f64; 9] = std::array::from_fn(|k| (1.0 + 1.0 / (k as f64 + 1.0)).log10());
    let mut dev = [0.0f64; 9];
    let mut devabs = [0.0f64; 9];
    for k in 0..9 {
        dev[k] = obs[k] - theo[k];
        devabs[k] = dev[k].abs();
    }
    let xs: [f64; 9] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];

    out[0] = corrcoef(&obs, &theo);
    out[1] = corr_welford(&xs, &dev);
    out[2] = corr_welford(&xs, &devabs);
    out[3] = nanstd(&dev, 1);
    out[4] = nanstd(&devabs, 1);
    out[5] = nanskew(&dev);
    out[6] = nanskew(&devabs);
    out[7] = nanmax(&dev);
    out[8] = nanmax(&devabs);
    out[9] = nanmin(&dev);
    out[10] = nanmin(&devabs);
    out[11] = nanmean(&devabs);
    out[12] = nanmedian(&dev);
    out[13] = nanmedian(&devabs);
    out[14] = autocorr(&dev, 1);
    out[15] = autocorr(&devabs, 1);
    out[16] = autocorr(&dev, 2);
    out[17] = autocorr(&devabs, 2);
    out[18] = autocorr(&dev, 3);
    out[19] = autocorr(&devabs, 3);
    out
}

fn collect(rows: &[Row], which: usize, out: &mut Vec<f64>) {
    out.clear();
    for r in rows.iter() {
        match which {
            0 => {
                if r.price != 0.0 {
                    out.push(r.volume)
                }
            }
            1 => {
                if r.price != 0.0 {
                    out.push(r.turnover)
                }
            }
            2 => {
                if r.flag == 66 && r.price != 0.0 {
                    out.push(r.volume)
                }
            }
            3 => {
                if r.flag == 83 && r.price != 0.0 {
                    out.push(r.volume)
                }
            }
            4 => {
                if r.price != 0.0 {
                    out.push(r.bid as f64)
                }
            }
            5 => {
                if r.price != 0.0 {
                    out.push(r.ask as f64)
                }
            }
            6 => {
                if r.flag == 32 && r.bid > 0 {
                    out.push(r.volume)
                }
            }
            7 => {
                if r.flag == 32 && r.ask > 0 {
                    out.push(r.volume)
                }
            }
            8 => {
                if r.flag == 32 && r.bid > 0 {
                    out.push(r.bid as f64)
                }
            }
            _ => {
                if r.flag == 32 && r.ask > 0 {
                    out.push(r.ask as f64)
                }
            }
        }
    }
}

/// 逐股入口：算不出来返回 None（上游会补 NaN，保证因子个数恒定）。
pub fn compute_stock(code: &str, date: i64) -> Option<Vec<f32>> {
    let rows = read_rows(code, date)?;
    let mut buf: Vec<f64> = Vec::new();
    let mut per = [[f64::NAN; N_STATS]; N_SERIES];
    for s in 0..N_SERIES {
        collect(&rows, s, &mut buf);
        per[s] = series_stats(&buf);
    }
    let mut block: Vec<f32> = Vec::with_capacity(PER_WINDOW);
    for s in 0..N_SERIES {
        block.extend(per[s].iter().map(|&v| v as f32));
    }
    for k in 0..N_STATS {
        block.push((per[4][k] - per[5][k]) as f32);
    }
    for k in 0..N_STATS {
        block.push((per[4][k] - per[5][k]).abs() as f32);
    }
    for k in 0..N_STATS {
        block.push((per[8][k] - per[9][k]) as f32);
    }
    for k in 0..N_STATS {
        block.push((per[8][k] - per[9][k]).abs() as f32);
    }
    debug_assert_eq!(block.len(), PER_WINDOW);

    // 旧代码 one(df) 里收的 df 从来没被用过，五段算的都是同一份 trade_data
    let mut out = Vec::with_capacity(N_OUT);
    for _ in 0..N_WINDOWS {
        out.extend_from_slice(&block);
    }
    if out.len() != N_OUT {
        return None;
    }
    Some(out)
}

/// hm57 全是逐股因子，没有横截面版本。
pub fn compute_cross(_date: i64) -> Option<(Vec<String>, Vec<f32>)> {
    None
}
