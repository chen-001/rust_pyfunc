//! hm47「死亡半衰期」的 Rust 复现（旧框架 design_whatever_out.TickDataBase）。
//!
//! 旧文件：/home/chenzongwei/pythoncode/成果hm47_死亡半衰期/hm47_okay.py
//!   Go.get_factor 参数 get_many=0 / read_full=0 / change_time=1，
//!   也就是「一整天的逐笔」一张表，窗口由旧代码自己切。逐股系列。
//!
//! ════════════════════════════════════════════════════════════════════
//! 一、旧代码逐步对照
//! ════════════════════════════════════════════════════════════════════
//! 1) df = read_trade_data_single(with_retreat=0)
//!    · pd.read_csv(<root>/stock/<date>/transaction/<code>_<date>_transaction.csv,
//!      usecols=[date,exchtime,price,volume,turnover,flag,localtime,ask_order,bid_order])
//!      行序 = 文件序，exchtime 是「带 .0 的浮点串」的微秒数。
//!    · with_retreat=0 → data[data["flag"] != 32]（数值比较，"32.0" 也过滤得掉）。
//!    · change_time=1 → deal_data：exchtime 换成 Timestamp
//!      （epoch 微秒 /1e6 秒 + 1970-01-01 08:00:00，走 cast_from_unit_vectorized，末位有 ns 噪声）。
//! 2) df = df.set_index("exchtime").between_time("09:30", "14:57").reset_index()
//!    （两端闭区间，微秒/纳秒精确比较，不能按整秒截断）。
//! 3) 在这张表上算 12 个「半衰期」列（rp 的两个 #[pyfunction]，见第二节）：
//!      energyM1old / N1old / N2old = find_half_energy_time(exchtime_ns, price, 1/5/30)
//!      energyM1/N1/N2              = find_half_extreme_time(..., 方向默认)
//!      energyM1/N1/N2_up/_down     = find_half_extreme_time(..., "pos"/"neg")
//!    再把每一列里的 0 换成 NaN（replace(0, np.nan)）。
//! 4) 五个窗口（all 就是上面那张全表，其余是它的子集）：
//!      open30    [09:30:00, 10:00:00]
//!      morning   [09:30:00, 11:30:00]
//!      afternoon [13:00:00, 14:57:00]
//!      close30   [14:30:00, 14:57:00]
//!    每个窗口 one(df) 返回 138 个值，顺序见第三节；五段首尾拼成 690 个。
//!
//! 二、one(df) 里的四组与 138 个值
//!    df_large = df[volume >= volume.quantile(0.9)]
//!    df_small = df[(volume <= volume.quantile(0.5)) & (volume > 100)]
//!    df_66    = df[flag == 66]      df_83 = df[flag == 83]
//!    · 前 18 个 = corr(volume, energyX)，df[["volume","energyX"]].corr().iloc[0,1]
//!      走 pandas libalgos.nancorr（Welford 在线算法、掩码 isfinite），
//!      六个能量列 × {全体, 66, 83} 两组顺序见 CORR_KEYS。
//!    · 后 120 个 = 四组 × 六个方向列(N1_up,N2_up,M1_up,N1_down,N2_down,M1_down)
//!      × 五个统计量(std, max, median, autocorr1, autocorr2)。
//!      std 是 pandas 样本标准差 ddof=1、max/median 跳过 NaN，
//!      autocorr(k) = Series.corr(shift(k)) → nanops.nancorr → **np.corrcoef**（两遍法），
//!      与上面的 Welford 不是同一条路，不能互相代用。
//!
//! ！！异常语义：旧代码整段包在 try/except 里，`Series.autocorr(k)` 在
//!   `lag >= len(组)` 时会抛 ValueError（被吞掉），于是**整只股票日返回 None**。
//!   由于每组都要算 autocorr2，等价于「五个窗口里任一窗口的任一组少于 3 行 →
//!   整只股票日算不出来」。本实现照抄：返回 None。
//!
//! ════════════════════════════════════════════════════════════════════
//! 三、两个 rp 函数与超时
//! ════════════════════════════════════════════════════════════════════
//! rp.find_half_energy_time（src/time_series/mod.rs:1586）与 rp.find_half_extreme_time
//! （同文件 1379 行起）都是 #[pyfunction]，worker 不带 libpython 不能在链路里调，
//! 所以把两者的**纯 Rust 主体逐字抄**进本文件（把 PyReadonlyArray1 换成 &[f64]、
//! PyResult 去掉；两条长度不一致的 PyErr 分支走不到）。src/time_series/mod.rs 一行没动。
//!
//! ！！超时：旧代码两个函数都传 `timeout_seconds=1`，实现里每处理 100 个外层元素
//!   查一次 `Instant::now()`，超时就**整列返回 NaN**。这个方法本来就是 rp 的，
//!   要照抄（不抄的话真触发超时的那天会整列对不上）。本文件保留同样的检查点。
//!
//! ════════════════════════════════════════════════════════════════════
//! 四、几个必须逐位对齐的 pandas 细节
//! ════════════════════════════════════════════════════════════════════
//! · CSV 浮点走 pandas C 解析器 precise_xstrtod 的复刻（parse_f64_pandas）。
//!   与 Rust 的正确舍入 str::parse 在 turnover 上有 1 ulp 差异（见 hm090.rs 的说明），
//!   price 上本数据没差，但统一用它更稳。
//! · exchtime 的纳秒值照抄 conversion.pyx 的 cast_from_unit_vectorized
//!   （base=(int64)秒、frac=np.round(秒-base,9)、ns=base*1e9+(int64)(frac*1e9)）。
//! · Series.std()：非 NaN 计数的样本标准差（ddof=1）。
//! · Series.quantile(q)：线性插值 pos=q*(n-1)。
//! · df[["a","b"]].corr() → libalgos.nancorr 的 Welford（掩码 isfinite、min_periods=1）。
//! · Series.autocorr(k) → nanops.nancorr → np.corrcoef 的两遍法，
//!   成对丢 NaN，结果 clip 到 [-1,1]。

use crate::data_paths;
use crate::features_f64 as ff;
use std::fs::File;
use std::io::Read;
use std::time::{Duration, Instant};

// ============================================================================
// 常量
// ============================================================================

/// 逐笔 CSV 的列索引（0 基，与 fast_csv_reader 一致；2018 年及以前末尾多一列 num）
const TRD_COL_EXCHTIME: usize = 4;
const TRD_COL_PRICE: usize = 7;
const TRD_COL_VOLUME: usize = 8;
const TRD_COL_TURNOVER: usize = 9;
const TRD_COL_FLAG: usize = 10;
const TRD_NCOL: usize = 16;

const DAY_NS: i64 = 86_400_000_000_000;
const H8_NS: i64 = 8 * 3600 * 1_000_000_000;

const fn hms(h: i64, m: i64, s: i64) -> i64 {
    (h * 3600 + m * 60 + s) * 1_000_000_000
}

/// 12 个能量列的下标与含义（顺序 = res 里 corr 段引用的顺序无关，只为可读）
const E_M1OLD: usize = 0;
const E_N1OLD: usize = 1;
const E_N2OLD: usize = 2;
const E_M1: usize = 3;
const E_N1: usize = 4;
const E_N2: usize = 5;
const E_M1_UP: usize = 6;
const E_M1_DOWN: usize = 7;
const E_N1_UP: usize = 8;
const E_N1_DOWN: usize = 9;
const E_N2_UP: usize = 10;
const E_N2_DOWN: usize = 11;
const N_ENERGY: usize = 12;

/// 五个窗口（真实钟点、纳秒闭区间，顺序 = 值的拼接顺序）
const SESSIONS: [(&str, i64, i64); 5] = [
    ("all", hms(9, 30, 0), hms(14, 57, 0)),
    ("open30", hms(9, 30, 0), hms(10, 0, 0)),
    ("morning", hms(9, 30, 0), hms(11, 30, 0)),
    ("afternoon", hms(13, 0, 0), hms(14, 57, 0)),
    ("close30", hms(14, 30, 0), hms(14, 57, 0)),
];

/// 18 个 corr： (名字里的能量列名, 能量列下标, 组后缀)
const CORR_KEYS: [(&str, usize, &str); 18] = [
    ("N1old", E_N1OLD, ""),
    ("N2old", E_N2OLD, ""),
    ("M1old", E_M1OLD, ""),
    ("N1", E_N1, ""),
    ("N2", E_N2, ""),
    ("M1", E_M1, ""),
    ("N1", E_N1, "66"),
    ("N1", E_N1, "83"),
    ("N2", E_N2, "66"),
    ("N2", E_N2, "83"),
    ("M1", E_M1, "66"),
    ("M1", E_M1, "83"),
    ("N1old", E_N1OLD, "66"),
    ("N1old", E_N1OLD, "83"),
    ("N2old", E_N2OLD, "66"),
    ("N2old", E_N2OLD, "83"),
    ("M1old", E_M1OLD, "66"),
    ("M1old", E_M1OLD, "83"),
];

/// 四组的顺序 = res 里 120 个 half_* 的顺序
const GRP4: [&str; 4] = ["large", "small", "66", "83"];
/// 六个方向列的顺序 = res 里每组的列序。
/// 名字是 `half_{统计量}_{能量列}_{组}_{up|down}_{窗口}`，
/// 注意**组名在 up/down 前面**（旧代码的变量名如 half_std_M1_66_up 就是这个形状）。
const DIR_COLS: [(&str, &str, usize); 6] = [
    ("N1", "up", E_N1_UP),
    ("N2", "up", E_N2_UP),
    ("M1", "up", E_M1_UP),
    ("N1", "down", E_N1_DOWN),
    ("N2", "down", E_N2_DOWN),
    ("M1", "down", E_M1_DOWN),
];
/// 五个统计量的顺序
const STATS5: [&str; 5] = ["std", "max", "median", "autocorr1", "autocorr2"];

/// 一个窗口 138 个值
const PER_WINDOW: usize = 18 + 4 * 6 * 5;
/// 五个窗口共 690 个
pub const N_OUT: usize = PER_WINDOW * 5;

// ============================================================================
// 名字
// ============================================================================

/// 690 个因子名，顺序与旧代码 names 的构造完全一致。
pub fn names() -> Vec<String> {
    let mut base: Vec<String> = Vec::with_capacity(PER_WINDOW);
    for (nm, _, grp) in CORR_KEYS.iter() {
        base.push(if grp.is_empty() {
            format!("corr_volume_{}", nm)
        } else {
            format!("corr_volume_{}_{}", nm, grp)
        });
    }
    for g in GRP4.iter() {
        for (c, d, _) in DIR_COLS.iter() {
            for s in STATS5.iter() {
                base.push(format!("half_{}_{}_{}_{}", s, c, g, d));
            }
        }
    }
    let mut out = Vec::with_capacity(N_OUT);
    for (s, _, _) in SESSIONS.iter() {
        for b in base.iter() {
            out.push(format!("{}_{}", b, s));
        }
    }
    out
}

// ============================================================================
// 读取
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

/// 10 的幂表（与 pandas/tokenizer.c 里那张写死的 e[] 表逐位一致）。
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

/// pandas C 解析器 `precise_xstrtod` 的逐行复刻（与 hm090.rs / hm012.rs 同一份）。
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

/// 空单元格按 pandas 的读法给 NaN
#[inline]
fn parse_f64_at(b: &[u8]) -> f64 {
    let mut s = 0usize;
    let mut e = b.len();
    while s < e && matches!(b[s], b' ' | b'\t' | b'\r' | b'\n') {
        s += 1;
    }
    while e > s && matches!(b[e - 1], b' ' | b'\t' | b'\r' | b'\n') {
        e -= 1;
    }
    if s >= e {
        return f64::NAN;
    }
    parse_f64_pandas(&b[s..e])
}

#[inline]
fn split_cols<'a>(line: &'a str, buf: &mut [&'a [u8]; 32], n: usize) -> usize {
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

/// pandas `pd.to_timedelta(微秒/1e6, unit="s") + Timestamp("1970-01-01 08:00:00")`
/// 的纳秒值（已含 8 小时偏移）。逐行复刻 conversion.pyx 的 cast_from_unit_vectorized。
#[inline]
fn pandas_time_ns(us: f64) -> i64 {
    if us.is_nan() {
        return i64::MIN;
    }
    let val = us / 1e6;
    let base = val.trunc();
    let frac = val - base;
    let frac = (frac * 1e9).round_ties_even() / 1e9;
    (base as i64) * 1_000_000_000 + (frac * 1e9) as i64
}

#[derive(Clone, Copy)]
struct Row {
    us: f64,
    price: f64,
    volume: f64,
    flag: f64,
}

/// 读 transaction CSV（列索引与 fast_csv_reader 一致），行序 = 文件序。
/// 返回 (ns, price, volume, flag) 四列，ns 已含 8 小时偏移。
fn read_trade(code: &str, date: i64) -> Option<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>)> {
    let filename = format!("{}_{}_transaction.csv", code, date);
    let mut path = None;
    for root in data_paths::level2_roots() {
        let p = root.join(date.to_string()).join("transaction").join(&filename);
        if p.exists() {
            path = Some(p);
            break;
        }
    }
    let path = path?;
    let mut content = String::new();
    File::open(&path).ok()?.read_to_string(&mut content).ok()?;

    let mut ns: Vec<f64> = Vec::with_capacity(content.len() / 120 + 1);
    let mut price: Vec<f64> = Vec::with_capacity(content.len() / 120 + 1);
    let mut volume: Vec<f64> = Vec::with_capacity(content.len() / 120 + 1);
    let mut flag: Vec<f64> = Vec::with_capacity(content.len() / 120 + 1);

    let mut lines = content.lines();
    lines.next(); // 表头
    for line in lines {
        if line.is_empty() {
            continue;
        }
        let mut buf: [&[u8]; 32] = [&[][..]; 32];
        split_cols(line, &mut buf, TRD_NCOL);
        let us = parse_f64_at(buf[TRD_COL_EXCHTIME]);
        let fl = parse_f64_at(buf[TRD_COL_FLAG]);
        // 旧代码 with_retreat=0：data[data["flag"] != 32]（NaN 也留）
        if fl == 32.0 {
            continue;
        }
        ns.push(pandas_time_ns(us) as f64 + H8_NS as f64);
        price.push(parse_f64_at(buf[TRD_COL_PRICE]));
        volume.push(parse_f64_at(buf[TRD_COL_VOLUME]));
        flag.push(fl);
        let _ = parse_f64_at(buf[TRD_COL_TURNOVER]);
    }
    Some((ns, price, volume, flag))
}

// ============================================================================
// 两个 rp 函数的纯 Rust 主体（src/time_series/mod.rs 逐字抄，去掉 pyo3 外壳）
// ============================================================================

const TIMEOUT_SECONDS: f64 = 1.0;

#[inline]
fn timed_out(start: Instant) -> bool {
    start.elapsed() > Duration::from_secs_f64(TIMEOUT_SECONDS)
}

/// find_half_extreme_time 的纯 Rust 主体（direction: "pos" / "neg" / 其它=全部）。
fn half_extreme_time(times_ns: &[f64], prices: &[f64], time_window: f64, direction: &str) -> Vec<f64> {
    let start = Instant::now();
    let times: Vec<f64> = times_ns.iter().map(|&x| x / 1.0e9).collect();
    let n = times.len();
    let mut result = vec![time_window; n];
    if timed_out(start) {
        return vec![f64::NAN; n];
    }
    for i in 0..n {
        if i % 100 == 0 && timed_out(start) {
            return vec![f64::NAN; n];
        }
        let current_time = times[i];
        let current_price = prices[i];
        if !current_price.is_finite() {
            result[i] = f64::NAN;
            continue;
        }
        let mut max_up = 0.0f64;
        let mut max_down = 0.0f64;
        for j in i..n {
            let time_diff = times[j] - current_time;
            if time_diff > time_window {
                break;
            }
            if !prices[j].is_finite() {
                continue;
            }
            let price_ratio = (prices[j] - current_price) / current_price;
            if price_ratio > max_up {
                max_up = price_ratio;
            } else if price_ratio < -max_down {
                max_down = -price_ratio;
            }
        }
        let (target_ratio, dir_value) = match direction {
            "pos" => {
                if max_up <= 0.0 {
                    result[i] = f64::NAN;
                    continue;
                }
                (max_up, 1.0)
            }
            "neg" => {
                if max_down <= 0.0 {
                    result[i] = f64::NAN;
                    continue;
                }
                (max_down, -1.0)
            }
            _ => {
                if max_up > max_down {
                    (max_up, 1.0)
                } else {
                    (max_down, -1.0)
                }
            }
        };
        if target_ratio <= 0.0 {
            continue;
        }
        let half_ratio = target_ratio / 2.0 * dir_value;
        for j in i..n {
            let time_diff = times[j] - current_time;
            if time_diff > time_window {
                break;
            }
            if !prices[j].is_finite() {
                continue;
            }
            let price_ratio = (prices[j] - current_price) / current_price;
            if (dir_value > 0.0 && price_ratio >= half_ratio)
                || (dir_value < 0.0 && price_ratio <= half_ratio)
            {
                result[i] = time_diff;
                break;
            }
        }
    }
    if timed_out(start) {
        return vec![f64::NAN; n];
    }
    result
}

/// find_half_energy_time 的纯 Rust 主体（旧代码 direction 一律走默认 "ignore"）。
fn half_energy_time(times_ns: &[f64], prices: &[f64], time_window: f64, direction: &str) -> Vec<f64> {
    let start = Instant::now();
    let times: Vec<f64> = times_ns.iter().map(|&x| x / 1.0e9).collect();
    let n = times.len();
    let mut result = vec![time_window; n];
    if timed_out(start) {
        return vec![f64::NAN; n];
    }
    for i in 0..n {
        if i % 100 == 0 && timed_out(start) {
            return vec![f64::NAN; n];
        }
        let current_time = times[i];
        let current_price = prices[i];
        if !current_price.is_finite() {
            result[i] = f64::NAN;
            continue;
        }
        let mut final_energy = 0.0f64;
        for j in i..n {
            if j == i {
                continue;
            }
            let time_diff = times[j] - current_time;
            if time_diff < time_window {
                continue;
            }
            if !prices[j].is_finite() {
                continue;
            }
            let price_change = prices[j] - current_price;
            match direction {
                "pos" if price_change <= 0.0 => {
                    result[i] = f64::NAN;
                    break;
                }
                "neg" if price_change >= 0.0 => {
                    result[i] = f64::NAN;
                    break;
                }
                _ => {}
            }
            final_energy = price_change.abs() / current_price;
            break;
        }
        if final_energy <= 0.0 {
            result[i] = 0.0;
            continue;
        }
        let half_energy = final_energy / 2.0;
        for j in i..n {
            if j == i {
                continue;
            }
            let time_diff = times[j] - current_time;
            if time_diff > time_window {
                break;
            }
            if !prices[j].is_finite() {
                continue;
            }
            let price_ratio = (prices[j] - current_price).abs() / current_price;
            if price_ratio >= half_energy {
                result[i] = time_diff;
                break;
            }
        }
    }
    if timed_out(start) {
        return vec![f64::NAN; n];
    }
    result
}

// ============================================================================
// pandas 口径的统计量
// ============================================================================

/// numpy 的 pairwise_sum（PW_BLOCKSIZE = 128），逐行复刻。
fn pairwise_sum(a: &[f64]) -> f64 {
    let n = a.len();
    if n < 8 {
        let mut r = 0.0f64;
        for i in 0..n {
            r += a[i];
        }
        return r;
    }
    if n <= 128 {
        let mut r = [a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]];
        let mut i = 8usize;
        let stop = n - (n % 8);
        while i < stop {
            for k in 0..8 {
                r[k] += a[i + k];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n {
            res += a[i];
            i += 1;
        }
        return res;
    }
    let mut n2 = n / 2;
    n2 -= n2 % 8;
    pairwise_sum(&a[..n2]) + pairwise_sum(&a[n2..])
}

/// Series.median()：非 NaN 排序取中位；偶数个取两数均值。
fn nan_median(x: &[f64]) -> f64 {
    let mut xs: Vec<f64> = x.iter().filter(|v| !v.is_nan()).copied().collect();
    if xs.is_empty() {
        return f64::NAN;
    }
    xs.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = xs.len();
    if n % 2 == 1 {
        xs[n / 2]
    } else {
        (xs[n / 2 - 1] + xs[n / 2]) / 2.0
    }
}

/// Series.quantile(q)：线性插值 pos = q*(n-1)，跳过 NaN。
fn nan_quantile(x: &[f64], q: f64) -> f64 {
    let mut xs: Vec<f64> = x.iter().filter(|v| !v.is_nan()).copied().collect();
    if xs.is_empty() {
        return f64::NAN;
    }
    xs.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = xs.len();
    if n == 1 {
        return xs[0];
    }
    let pos = q * (n - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = (lo + 1).min(n - 1);
    let frac = pos - lo as f64;
    // numpy 的 _lerp：t < 0.5 用 a + (b-a)*t，否则用 b - (b-a)*(1-t)
    let d = xs[hi] - xs[lo];
    if frac < 0.5 {
        xs[lo] + d * frac
    } else {
        xs[hi] - d * (1.0 - frac)
    }
}

/// pandas `Series.autocorr(k)` = Series.corr(shift(k)) → nanops.nancorr → np.corrcoef。
/// 两遍法、成对丢 NaN、结果 clip 到 [-1,1]。
fn autocorr_lag(x: &[f64], lag: usize) -> f64 {
    let n = x.len();
    if n <= lag {
        return f64::NAN;
    }
    let mut a: Vec<f64> = Vec::with_capacity(n - lag);
    let mut b: Vec<f64> = Vec::with_capacity(n - lag);
    for i in lag..n {
        let (u, v) = (x[i], x[i - lag]);
        if u.is_nan() || v.is_nan() {
            continue;
        }
        a.push(u);
        b.push(v);
    }
    let m = a.len();
    if m < 2 {
        return f64::NAN;
    }
    let ma = pairwise_sum(&a) / m as f64;
    let mb = pairwise_sum(&b) / m as f64;
    let mut sxy = 0.0f64;
    let mut sxx = 0.0f64;
    let mut syy = 0.0f64;
    for i in 0..m {
        let xa = a[i] - ma;
        let yb = b[i] - mb;
        sxy += xa * yb;
        sxx += xa * xa;
        syy += yb * yb;
    }
    // np.corrcoef：c = dot/(n-1)，再 c01 / sqrt(c00) / sqrt(c11)，最后 clip
    let nf = (m - 1) as f64;
    let c00 = sxx / nf;
    let c11 = syy / nf;
    let c01 = sxy / nf;
    let r = (c01 / c00.sqrt()) / c11.sqrt();
    if r.is_nan() {
        f64::NAN
    } else {
        r.clamp(-1.0, 1.0)
    }
}

/// pandas `DataFrame.corr()` 的 Pearson：逐行复刻 pandas/_libs/algos.pyx 的 nancorr
/// （Welford 在线算法、掩码 isfinite、min_periods=1）。
fn pearson_welford(x: &[f64], y: &[f64]) -> f64 {
    let mut nobs = 0.0f64;
    let (mut meanx, mut meany) = (0.0f64, 0.0f64);
    let (mut ssqdmx, mut ssqdmy, mut covxy) = (0.0f64, 0.0f64, 0.0f64);
    for i in 0..x.len() {
        if !x[i].is_finite() || !y[i].is_finite() {
            continue;
        }
        let (vx, vy) = (x[i], y[i]);
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
// 主流程
// ============================================================================

struct Frame {
    ns: Vec<i64>,
    price: Vec<f64>,
    volume: Vec<f64>,
    flag: Vec<f64>,
    energy: Vec<[f64; N_ENERGY]>,
}

/// 一个窗口的 138 个值。返回 None 表示旧代码在这一段抛了异常（整个股票日算不出来）。
fn one(fr: &Frame, sel: &[usize]) -> Option<Vec<f64>> {
    let q90 = nan_quantile(
        &sel.iter().map(|&i| fr.volume[i]).collect::<Vec<f64>>(),
        0.9,
    );
    let q50 = nan_quantile(
        &sel.iter().map(|&i| fr.volume[i]).collect::<Vec<f64>>(),
        0.5,
    );
    let large: Vec<usize> = sel
        .iter()
        .copied()
        .filter(|&i| fr.volume[i] >= q90)
        .collect();
    let small: Vec<usize> = sel
        .iter()
        .copied()
        .filter(|&i| fr.volume[i] <= q50 && fr.volume[i] > 100.0)
        .collect();
    let g66: Vec<usize> = sel.iter().copied().filter(|&i| fr.flag[i] == 66.0).collect();
    let g83: Vec<usize> = sel.iter().copied().filter(|&i| fr.flag[i] == 83.0).collect();
    let groups: [&Vec<usize>; 4] = [&large, &small, &g66, &g83];
    // ！！pandas 2.x 的 Series.autocorr 不再抛 ValueError（1.x 才会），
    // 组太短时只是走到 np.corrcoef 的 0/0 返回 NaN，所以这里**不能**设行数门槛。
    // 实测 600519@20250304 / 300750@20231020 的旧输出各有 150 个 NaN 且照常返回 690 个值。

    let mut out: Vec<f64> = Vec::with_capacity(PER_WINDOW);
    // 18 个 corr（DataFrame.corr → Welford）
    let vol = |idx: &[usize]| -> Vec<f64> { idx.iter().map(|&i| fr.volume[i]).collect() };
    let en = |idx: &[usize], c: usize| -> Vec<f64> { idx.iter().map(|&i| fr.energy[i][c]).collect() };
    for (_, c, grp) in CORR_KEYS.iter() {
        let idx: &[usize] = match *grp {
            "" => sel,
            "66" => &g66,
            _ => &g83,
        };
        out.push(pearson_welford(&vol(idx), &en(idx, *c)));
    }
    // 120 个 half_*
    for g in groups.iter() {
        for (_, _, c) in DIR_COLS.iter() {
            let x: Vec<f64> = g.iter().map(|&i| fr.energy[i][*c]).collect();
            out.push(ff::col_std(&x));
            out.push(ff::col_max(&x));
            out.push(nan_median(&x));
            out.push(autocorr_lag(&x, 1));
            out.push(autocorr_lag(&x, 2));
        }
    }
    debug_assert_eq!(out.len(), PER_WINDOW);
    Some(out)
}

/// 逐股的一天：算不出来返回 None（上游补 NaN，保证因子个数恒定）。
pub fn compute_stock(code: &str, date: i64) -> Option<Vec<f32>> {
    let (ns_all, price_all, volume_all, flag_all) = read_trade(code, date)?;
    let n_rows = ns_all.len();

    // between_time("09:30", "14:57")
    let mut keep: Vec<usize> = Vec::with_capacity(n_rows);
    for i in 0..n_rows {
        let d = (ns_all[i] as i64).rem_euclid(DAY_NS);
        if d >= hms(9, 30, 0) && d <= hms(14, 57, 0) {
            keep.push(i);
        }
    }
    if keep.is_empty() {
        return None;
    }

    let n = keep.len();
    let ns: Vec<i64> = keep.iter().map(|&i| ns_all[i] as i64).collect();
    let price: Vec<f64> = keep.iter().map(|&i| price_all[i]).collect();
    let volume: Vec<f64> = keep.iter().map(|&i| volume_all[i]).collect();
    let flag: Vec<f64> = keep.iter().map(|&i| flag_all[i]).collect();
    // df.exchtime.to_numpy(dtype=float) 再 /1e9 → 秒
    let times_f: Vec<f64> = ns.iter().map(|&v| v as f64).collect();

    let mut energy: Vec<[f64; N_ENERGY]> = vec![[f64::NAN; N_ENERGY]; n];
    {
        let mut put = |c: usize, v: Vec<f64>| {
            debug_assert_eq!(v.len(), n);
            for i in 0..n {
                // replace(0, np.nan)
                energy[i][c] = if v[i] == 0.0 { f64::NAN } else { v[i] };
            }
        };
        put(E_M1OLD, half_energy_time(&times_f, &price, 1.0, "ignore"));
        put(E_N1OLD, half_energy_time(&times_f, &price, 5.0, "ignore"));
        put(E_N2OLD, half_energy_time(&times_f, &price, 30.0, "ignore"));
        put(E_M1, half_extreme_time(&times_f, &price, 1.0, "ignore"));
        put(E_N1, half_extreme_time(&times_f, &price, 5.0, "ignore"));
        put(E_N2, half_extreme_time(&times_f, &price, 30.0, "ignore"));
        put(E_M1_UP, half_extreme_time(&times_f, &price, 1.0, "pos"));
        put(E_M1_DOWN, half_extreme_time(&times_f, &price, 1.0, "neg"));
        put(E_N1_UP, half_extreme_time(&times_f, &price, 5.0, "pos"));
        put(E_N1_DOWN, half_extreme_time(&times_f, &price, 5.0, "neg"));
        put(E_N2_UP, half_extreme_time(&times_f, &price, 30.0, "pos"));
        put(E_N2_DOWN, half_extreme_time(&times_f, &price, 30.0, "neg"));
    }

    let fr = Frame { ns, price, volume, flag, energy };
    let mut out: Vec<f64> = Vec::with_capacity(N_OUT);
    for (_, lo, hi) in SESSIONS.iter() {
        let sel: Vec<usize> = (0..n)
            .filter(|&i| {
                let d = fr.ns[i].rem_euclid(DAY_NS);
                d >= *lo && d <= *hi
            })
            .collect();
        out.extend_from_slice(&one(&fr, &sel)?);
    }
    debug_assert_eq!(out.len(), N_OUT);
    Some(out.into_iter().map(|v| v as f32).collect())
}

/// hm47 全是逐股因子，没有横截面版本。
pub fn compute_cross(_date: i64) -> Option<(Vec<String>, Vec<f32>)> {
    None
}
