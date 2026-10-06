//! hm62「闪现开团」的 Rust 复现：大单附近小单的时间间隔 + 半能量时间 → 240 个统计量 × 4 个变体。
//!
//! 旧文件：/home/chenzongwei/pythoncode/成果hm62_闪现开团/hm62_okay.py
//! 形态：dwo.TickDataBase 的 Go.get_factor，get_many=0 / read_full=0 / change_time=1，
//!   也就是「一整天的逐笔」一张表，窗口由旧代码自己切。属于**逐股**系列。
//!
//! ════════════════════════════════════════════════════════════════════
//! 0) 浮点一律走 parse_f64_pandas（pandas 的 precise_xstrtod 复刻），**不要**用 str::parse。
//!    旧框架是 pd.read_csv 读进来的，不是正确舍入；实测 000001@20250303 有 260/76488 个
//!    turnover 与正确舍入差 1 ulp，而本系列下游有 round(2) 比阈值、quantile 切割、nunique
//!    这类操作，1 ulp 就能把掩码翻面（hm012 就是这么差出 1.4e-4~7.1e-3 的）。
//!
//! 一、旧代码逐步对照
//! ════════════════════════════════════════════════════════════════════
//! 1) df = read_trade_data_single(with_retreat=0)
//!    · hf_reader 用 usecols=[date,exchtime,price,volume,turnover,flag,localtime,
//!      ask_order,bid_order] 读 /ssd_data/stock/<date>/transaction/<code>_<date>_transaction.csv，
//!      行序 = 文件序。
//!    · with_retreat=0 → data[data["flag"] != 32]（pandas 的数值比较，"32.0" 也过滤得掉）。
//!    · change_time=1 → deal_data：加一列 delay = localtime − exchtime，
//!      并把 exchtime / localtime 换成 Timestamp（epoch 微秒 /1e6 秒 + 1970-01-01 08:00:00）。
//!    · 再加一列 code。
//! 2) df = dwo.adjust_afternoon(df)：按 exchtime 切 [09:30:00, 11:30:00] 与
//!    [13:00:00, 14:57:00]（between_time，微秒级闭区间），下午整段 −90 分钟，
//!    再 concat([上午, 下午]).reset_index()；**不重新排序**，所以行序是「上午整块在前、
//!    下午整块在后」。后面两个 rp 函数都按数组顺序扫，这个块顺序必须照抄。
//! 3) sing(df, only_after, large_to_large) 一次返回 240 个值：
//!      gap_small          = 大单到「附近目标单」的时间间隔（flag_filter="ignore"）
//!      gap_small_sameflag = 同上，flag_filter="same"
//!      gap_small_diffflag = 同上，flag_filter="diff"
//!      half_energy        = find_half_energy_time(exchtime, price, 15)
//!      flag_distance      = gap_small_sameflag − gap_small_diffflag
//!      flag_distance_abs  = |flag_distance|
//!    · 参数固定：large_quantile=0.9, small_quantile=0.5, near_number=15,
//!      exclude_same_time=True, order_type="small"；only_after / large_to_large 由变体给。
//!    · df = df.dropna()（**整表**丢 NaN，不是只看新列），再切 df_66 = df[flag==66]、
//!      df_83 = df[flag==83]。
//!    · 三个组（all / 66 / 83）各取 5 个新列 × 14 个统计量 = 210，再加 10 个列对相关 = 30。
//! 4) 4 个变体按 res1(False,False) + res2(False,True) + res3(True,False) + res4(True,True)
//!    拼接，共 960 个值；名字后缀块顺序正是
//!    _与小单间隔_前后都有 / _与大单间隔_前后都有 / _与小单间隔_只算后面 / _与大单间隔_只算后面。
//! 5) run_factor_out(names_single=names, names_save=names_save_unfold)：names_single 是
//!    960 项，get_factor 原样返回这 960 项（verify_series 对的就是这 960 项，早期版本
//!    误按 names_save 的 55 项对账，长度就对不上）；names_save（56 项）去掉尾部 `_fold`
//!    后 55 个不同的名字，才是本系列要存进因子库的 base，由 hm62_okay_database.py 里的
//!    write_selected_base 从这 960 列里挑。
//!
//! ════════════════════════════════════════════════════════════════════
//! 二、两个 rp 函数
//! ════════════════════════════════════════════════════════════════════
//! rp.find_half_energy_time 与 rp.calculate_large_order_nearby_small_order_time_gap 都是
//! #[pyfunction]（src/time_series/mod.rs 1586 / 1958 行起）。worker 二进制不带 libpython，
//! 链路里不能调，所以把两者的**纯 Rust 主体逐行抄**进本文件（只去掉 pyo3 外壳：
//! PyReadonlyArray1 → &[f64]、PyResult → 直接返回、`Ok(v)` → `v`、
//! find_half_energy_time 的 timeout_seconds 恒为 None 所以整段超时机制删掉、
//! 两条长度不一致的 PyErr 分支改成返回空 Vec，本用法走不到）。
//! src/time_series/mod.rs 一行没动。
//!
//! ════════════════════════════════════════════════════════════════════
//! 三、统计量口径
//! ════════════════════════════════════════════════════════════════════
//! dropna 之后参与统计的列一律无 NaN，所以直接用已验证过的 pandas 口径实现：
//!   mean/std/skew/kurt/max/min → crate::features_f64 的 col_*（与 pandas 对齐，
//!     已由 lib/okay_database/verify_features_f64_switches.py 覆盖）
//!   median → 偶数个取 (a+b)/2（= np.median 的 np.mean(两个)）
//!   sum    → numpy 的 pairwise_sum（PW_BLOCKSIZE=128），不是顺序累加
//!   corr / autocorr → pandas **Series.corr 走 np.corrcoef（两遍法）**，
//!     不是 DataFrame.corr 的 Welford。autocorr(k) = corr(x[k:], x[:n-k])。
//!
//! ════════════════════════════════════════════════════════════════════
//! 四、与旧代码的已知差异
//! ════════════════════════════════════════════════════════════════════
//! · 旧代码 `Series.autocorr(k)` 在 k >= len(组) 时会抛 ValueError，被 get_factor 的
//!   except 吞掉 → **整只股票日返回 None（960 个全 NaN）**。本实现同样处理：
//!   只要 all / 66 / 83 任一组少于 4 行，整个股票日返回 None。
//! · 旧代码里 read_trade_data_single 的 try/except 会把异常吞成 None，本实现里
//!   文件不存在或列数不够时同样返回 None。

use crate::data_paths;
use crate::features_f64 as ff;

/// 逐笔 CSV 的列索引（0 基，与 fast_csv_reader 一致；2018 年及以前末尾多一列 num）
const TRD_COL_DATE: usize = 2;
const TRD_COL_EXCHTIME: usize = 4;
const TRD_COL_LOCALTIME: usize = 5;
const TRD_COL_PRICE: usize = 7;
const TRD_COL_VOLUME: usize = 8;
const TRD_COL_TURNOVER: usize = 9;
const TRD_COL_FLAG: usize = 10;
const TRD_COL_ASK_ORDER: usize = 13;
const TRD_COL_BID_ORDER: usize = 14;
const TRD_NCOL: usize = 15;

const DAY_US: i64 = 86_400_000_000;
const SEC_US: i64 = 1_000_000;
const H8_NS: i64 = 8 * 3600 * 1_000_000_000;
const PM_SHIFT_NS: i64 = 90 * 60 * 1_000_000_000;
const AM_START_US: i64 = (9 * 3600 + 30 * 60) * SEC_US;
const AM_END_US: i64 = (11 * 3600 + 30 * 60) * SEC_US;
const PM_START_US: i64 = 13 * 3600 * SEC_US;
const PM_END_US: i64 = (14 * 3600 + 57 * 60) * SEC_US;

/// 每个变体 240 个：3 组 × (5 列 × 14 统计量) + 3 组 × 10 个列对相关。
const PER_VARIANT: usize = 3 * (5 * 14) + 3 * 10;
/// 4 个变体共 960 个值。
pub const N_ALL: usize = PER_VARIANT * 4;

const STATS14: [&str; 14] = [
    "mean", "std", "skew", "kurt", "max", "min", "median", "sum",
    "autocorr1", "autocorr2", "autocorr3",
    "corr_with_price", "corr_with_volume", "corr_with_halfenergy",
];
const COLS5: [&str; 5] = [
    "gap_small", "gap_small_sameflag", "gap_small_diffflag", "flag_distance", "flag_distance_abs",
];
const CORR10: [&str; 10] = [
    "corr_gs_gss", "corr_gs_gsd", "corr_gs_fd", "corr_gs_fda", "corr_gss_gsd",
    "corr_gss_fd", "corr_gss_fda", "corr_gsd_fd", "corr_gsd_fda", "corr_fd_fda",
];
const GROUPS3: [&str; 3] = ["all", "66", "83"];
const SUF4: [&str; 4] = [
    "_与小单间隔_前后都有", "_与大单间隔_前后都有", "_与小单间隔_只算后面", "_与大单间隔_只算后面",
];
/// 4 个变体的 (only_after, large_to_large)，顺序对应 SUF4。
const VARIANTS: [(bool, bool); 4] = [(false, false), (false, true), (true, false), (true, true)];
/// 5 个基础列在 one() 里的顺序 = COLS5。
const PAIRS10: [(usize, usize); 10] = [
    (0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4),
];



/// 960 个因子名，顺序与旧代码 names 的构造完全一致。
pub fn names() -> Vec<String> {
    let mut base: Vec<String> = Vec::with_capacity(PER_VARIANT);
    for g in GROUPS3.iter() {
        for c in COLS5.iter() {
            for s in STATS14.iter() {
                base.push(format!("{}_{}_{}", g, c, s));
            }
        }
    }
    for g in GROUPS3.iter() {
        for c in CORR10.iter() {
            base.push(format!("{}_{}", g, c));
        }
    }
    let mut out = Vec::with_capacity(N_ALL);
    for suf in SUF4.iter() {
        for b in base.iter() {
            out.push(format!("{}{}", b, suf));
        }
    }
    out
}

// ============================================================================
// 逐笔 CSV 读取 + adjust_afternoon
// ============================================================================

#[derive(Clone, Copy, Default)]
struct Raw {
    /// exchtime（epoch 微秒，CSV 里是 "1740964500130000.0" 这种浮点串）
    us: f64,
    price: f64,
    volume: f64,
    flag: f64,
    /// dropna 要看的其余原始列里是否出现过 NaN（date/localtime/turnover/ask/bid/delay）
    extra_nan: bool,
}

/// 去掉字段两端的空白（含行尾的 `\r`）。
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

/// pandas 的浮点读取：空串给 NaN，其余走 pandas C 解析器 `precise_xstrtod` 的复刻。
///
/// ！！不能用 `str::parse::<f64>()`（正确舍入）：旧框架是 `pd.read_csv` 读进来的，
/// 用的是 `_libs/src/parser/tokenizer.c` 的 precise_xstrtod —— 只累加**前 17 位有效数字**，
/// 多出来的整数位转成指数、多出来的小数位直接丢掉，最后用一张写死的 10 的幂做一次乘或除。
/// 实测 000001@20250303 有 260/76488 个 turnover 与正确舍入差 1 ulp。
/// 本系列下游有 `.round(2) > 0`、`quantile` 分位切割、`nunique` 这类「恰好相等 / 比阈值 /
/// 排名」的操作，1 ulp 就能把掩码翻面（hm012 在 000001@20250303 上就因此差 1.4e-4~7.1e-3，
/// 换成这份解析才过）。逐位等价更重要，所以照抄。
fn parse_f64_at(b: &[u8]) -> f64 {
    let t = trim_ascii(b);
    if t.is_empty() {
        return f64::NAN;
    }
    parse_f64_pandas(t)
}

/// pandas C 解析器 `precise_xstrtod`（_libs/src/parser/tokenizer.c）的逐行复刻。
/// 与 src/okay_database/hm033.rs / hm047.rs / hm090.rs 里的同名函数一致，这里复制一份
/// 免得跨系列互相依赖。
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
    // 指数段（本数据没有 'e'，按源码照抄）
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

/// 10 的幂表：用 Rust 自己的正确舍入解析 "1e{k}" 建出来，
/// 与 pandas/tokenizer.c 里那张写死的 e[] 表逐位一致。
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

/// 读 transaction CSV，丢掉 flag == 32（撤单）。行序 = 文件序。
fn read_trade(code: &str, date: i64) -> Option<Vec<Raw>> {
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
    let content = std::fs::read_to_string(&path).ok()?;

    let mut out: Vec<Raw> = Vec::with_capacity(content.len() / 150 + 1);
    let mut lines = content.lines();
    lines.next()?; // 表头
    for line in lines {
        if line.is_empty() {
            continue;
        }
        let bytes = line.as_bytes();
        let mut f: [&[u8]; TRD_NCOL] = [&b""[..]; TRD_NCOL];
        let mut col = 0usize;
        let mut start = 0usize;
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
            col += 1;
        }
        if col < TRD_NCOL {
            continue; // 列数不够的行：pandas 会报错，这里跳过保护一下
        }

        let flag = parse_f64_at(f[TRD_COL_FLAG]);
        if flag == 32.0 {
            continue; // with_retreat=0：pandas 的 df[df.flag != 32]
        }
        let us = parse_f64_at(f[TRD_COL_EXCHTIME]);
        let localtime = parse_f64_at(f[TRD_COL_LOCALTIME]);
        let price = parse_f64_at(f[TRD_COL_PRICE]);
        let volume = parse_f64_at(f[TRD_COL_VOLUME]);
        let turnover = parse_f64_at(f[TRD_COL_TURNOVER]);
        let ask_order = parse_f64_at(f[TRD_COL_ASK_ORDER]);
        let bid_order = parse_f64_at(f[TRD_COL_BID_ORDER]);
        let d_date = parse_f64_at(f[TRD_COL_DATE]);
        // delay = localtime - exchtime（deal_data 里在换算成时间戳之前算的）
        let delay = localtime - us;
        let extra_nan = d_date.is_nan()
            || localtime.is_nan()
            || turnover.is_nan()
            || ask_order.is_nan()
            || bid_order.is_nan()
            || delay.is_nan();

        out.push(Raw { us, price, volume, flag, extra_nan });
    }
    Some(out)
}

/// pandas `pd.to_timedelta(微秒/1e6, unit="s") + Timestamp("1970-01-01 08:00:00")`
/// 的纳秒值（已含 8 小时偏移）。逐行复刻 conversion.pyx 的 cast_from_unit_vectorized，
/// 与 hm066.rs 里那份相同（实测与 pandas 逐行相等）。
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

/// adjust_afternoon 之后的整张表（上午整块在前、下午整块在后，下午时刻 −90 分钟）。
struct Frame {
    /// exchtime 的纳秒值（含 8 小时偏移、下午已平移），后面按 float 用
    ns: Vec<i64>,
    price: Vec<f64>,
    volume: Vec<f64>,
    flag: Vec<i64>,
    /// 是否有原始列是 NaN（dropna 会把这些行丢掉）
    nan_any: Vec<bool>,
}

fn build_frame(raw: &[Raw]) -> Frame {
    let mut fr = Frame {
        ns: Vec::with_capacity(raw.len()),
        price: Vec::with_capacity(raw.len()),
        volume: Vec::with_capacity(raw.len()),
        flag: Vec::with_capacity(raw.len()),
        nan_any: Vec::with_capacity(raw.len()),
    };
    let mut push = |r: &Raw, shift: i64| {
        let ns = pandas_time_ns(r.us) + H8_NS + shift;
        fr.ns.push(ns);
        fr.price.push(r.price);
        fr.volume.push(r.volume);
        fr.flag.push(r.flag.round() as i64);
        fr.nan_any.push(
            r.extra_nan || r.us.is_nan() || r.price.is_nan() || r.volume.is_nan() || r.flag.is_nan(),
        );
    };
    for r in raw.iter() {
        let t = (r.us + 8.0 * 3600.0 * 1e6).rem_euclid(DAY_US as f64);
        let t = t as i64;
        if t >= AM_START_US && t <= AM_END_US {
            push(r, 0);
        }
    }
    for r in raw.iter() {
        let t = (r.us + 8.0 * 3600.0 * 1e6).rem_euclid(DAY_US as f64);
        let t = t as i64;
        if t >= PM_START_US && t <= PM_END_US {
            push(r, -PM_SHIFT_NS);
        }
    }
    fr
}

// ============================================================================
// pandas 口径的统计量（hm066.rs 里那份的复制）
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

/// pandas nanops 的求和：先把 NaN 换成 fill（默认 0.0），再走 numpy 的 pairwise_sum
fn nanops_sum(v: &[f64], fill: f64) -> f64 {
    let buf: Vec<f64> = v.iter().map(|&x| if x.is_nan() { fill } else { x }).collect();
    pairwise_sum(&buf)
}

/// pandas `Series.median()`：非 NaN 排序后取中位；偶数个取 (a+b)/2（= np.median 的
/// np.mean(两个)）。
fn nan_median(v: &[f64]) -> f64 {
    let mut xs: Vec<f64> = v.iter().cloned().filter(|x| !x.is_nan()).collect();
    let n = xs.len();
    if n == 0 {
        return f64::NAN;
    }
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    if n % 2 == 1 {
        xs[n / 2]
    } else {
        (xs[n / 2 - 1] + xs[n / 2]) / 2.0
    }
}

/// pandas `Series.corr`：走 np.corrcoef 的两遍法（先减均值再点乘，ddof=1）。
/// 分母为 0 或样本不足 2 时给 NaN（与 np.corrcoef 的 0/0 一致）。
fn corr2(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len();
    if n < 2 || b.len() != n {
        return f64::NAN;
    }
    let ma = pairwise_sum(a) / n as f64;
    let mb = pairwise_sum(b) / n as f64;
    let mut sxy = 0.0f64;
    let mut sxx = 0.0f64;
    let mut syy = 0.0f64;
    for i in 0..n {
        let xa = a[i] - ma;
        let yb = b[i] - mb;
        sxy += xa * yb;
        sxx += xa * xa;
        syy += yb * yb;
    }
    if !(sxx > 0.0) || !(syy > 0.0) {
        return f64::NAN;
    }
    sxy / (sxx.sqrt() * syy.sqrt())
}

/// pandas `Series.autocorr(k)` = corr(x, x.shift(k))，对齐后是 (x[k:], x[:n-k])。
fn autocorr_lag(x: &[f64], lag: usize) -> f64 {
    let n = x.len();
    if n <= lag {
        return f64::NAN;
    }
    corr2(&x[lag..], &x[..n - lag])
}

/// 旧代码 one(df, col) 的 14 个值，顺序与 STATS14 一致。
fn one(x: &[f64], price: &[f64], volume: &[f64], halfenergy: &[f64]) -> [f64; 14] {
    let mut r = [f64::NAN; 14];
    if x.is_empty() {
        // pandas：mean/std/skew/kurt/max/min/median 给 NaN、sum 给 0.0、
        // autocorr 会抛异常（本实现的调用方已保证组内 >= 4 行）
        r[7] = 0.0;
        return r;
    }
    r[0] = ff::col_mean(x);
    r[1] = ff::col_std(x);
    r[2] = ff::col_skew(x);
    r[3] = ff::col_kurt(x);
    r[4] = ff::col_max(x);
    r[5] = ff::col_min(x);
    r[6] = nan_median(x);
    r[7] = nanops_sum(x, 0.0);
    r[8] = autocorr_lag(x, 1);
    r[9] = autocorr_lag(x, 2);
    r[10] = autocorr_lag(x, 3);
    r[11] = corr2(x, price);
    r[12] = corr2(x, volume);
    r[13] = corr2(x, halfenergy);
    r
}

// ============================================================================
// 主流程
// ============================================================================

/// 一个变体 240 个值。任一组不足 4 行时返回 None（对应旧代码 autocorr 抛异常被吞成 None）。
fn sing(fr: &Frame, only_after: bool, large_to_large: bool) -> Option<Vec<f64>> {
    let n = fr.ns.len();
    if n == 0 {
        return None;
    }
    let ns_f: Vec<f64> = fr.ns.iter().map(|&v| v as f64).collect();
    let gs = large_order_nearby_small_order_time_gap(
        &fr.volume, &ns_f, 0.9, 0.5, 15, true, "small", Some(&fr.flag), "ignore",
        only_after, large_to_large,
    );
    let gss = large_order_nearby_small_order_time_gap(
        &fr.volume, &ns_f, 0.9, 0.5, 15, true, "small", Some(&fr.flag), "same",
        only_after, large_to_large,
    );
    let gsd = large_order_nearby_small_order_time_gap(
        &fr.volume, &ns_f, 0.9, 0.5, 15, true, "small", Some(&fr.flag), "diff",
        only_after, large_to_large,
    );
    if gs.len() != n || gss.len() != n || gsd.len() != n {
        return None;
    }
    let he = half_energy_time(&ns_f, &fr.price, 15.0, "ignore");
    if he.len() != n {
        return None;
    }
    let fd: Vec<f64> = (0..n).map(|i| gss[i] - gsd[i]).collect();
    let fda: Vec<f64> = fd.iter().map(|v| v.abs()).collect();

    // df.dropna()：整表丢 NaN。原始列里只有 exchtime/price/volume/flag 与
    // date/localtime/turnover/ask_order/bid_order/delay 可能是 NaN，前面已记进 nan_any。
    let keep: Vec<usize> = (0..n)
        .filter(|&i| {
            !fr.nan_any[i]
                && !gs[i].is_nan()
                && !gss[i].is_nan()
                && !gsd[i].is_nan()
                && !he[i].is_nan()
        })
        .collect();

    let cols: [&Vec<f64>; 5] = [&gs, &gss, &gsd, &fd, &fda];
    let mut out: Vec<f64> = Vec::with_capacity(PER_VARIANT);
    for (g, name) in GROUPS3.iter().enumerate() {
        let sel: Vec<usize> = keep
            .iter()
            .copied()
            .filter(|&i| match g {
                0 => true,
                1 => fr.flag[i] == 66,
                _ => fr.flag[i] == 83,
            })
            .collect();
        let _ = name;
        if sel.len() < 4 {
            // 旧代码 Series.autocorr(3) 会抛 ValueError → get_factor 返回 None → 整行 NaN
            return None;
        }
        for c in 0..5 {
            let x: Vec<f64> = sel.iter().map(|&i| cols[c][i]).collect();
            let p: Vec<f64> = sel.iter().map(|&i| fr.price[i]).collect();
            let v: Vec<f64> = sel.iter().map(|&i| fr.volume[i]).collect();
            let h: Vec<f64> = sel.iter().map(|&i| he[i]).collect();
            out.extend_from_slice(&one(&x, &p, &v, &h));
        }
    }
    for g in 0..3 {
        let sel: Vec<usize> = keep
            .iter()
            .copied()
            .filter(|&i| match g {
                0 => true,
                1 => fr.flag[i] == 66,
                _ => fr.flag[i] == 83,
            })
            .collect();
        if sel.len() < 4 {
            return None;
        }
        for &(a, b) in PAIRS10.iter() {
            let x: Vec<f64> = sel.iter().map(|&i| cols[a][i]).collect();
            let y: Vec<f64> = sel.iter().map(|&i| cols[b][i]).collect();
            out.push(corr2(&x, &y));
        }
    }
    Some(out)
}

fn all_values(code: &str, date: i64) -> Option<Vec<f64>> {
    let raw = read_trade(code, date)?;
    let fr = build_frame(&raw);
    let mut all: Vec<f64> = Vec::with_capacity(N_ALL);
    for &(oa, l2l) in VARIANTS.iter() {
        let v = sing(&fr, oa, l2l)?;
        all.extend_from_slice(&v);
    }
    if all.len() != N_ALL {
        return None;
    }
    Some(all)
}

/// 逐股入口：返回旧 names_single 的 960 个值（names() 的顺序逐个对齐）。
pub fn compute_stock(code: &str, date: i64) -> Option<Vec<f32>> {
    let vals = all_values(code, date)?;
    Some(vals.iter().map(|&v| v as f32).collect())
}

/// hm62 是逐股系列，没有横截面。
pub fn compute_cross(_date: i64) -> Option<(Vec<String>, Vec<f32>)> {
    None
}

fn half_energy_time(
    times: &[f64],
    prices: &[f64],
    time_window: f64,
    direction: &str,
) -> Vec<f64> {
    // 提取数组数据
    let times: Vec<f64> = times.iter().map(|&x| x / 1.0e9).collect();
    let n = times.len();

    // 预分配结果向量，初始值为 time_window
    let mut result = vec![time_window; n];

    // 计算每个时间点的半能量时间（使用单线程处理）
    for i in 0..n {

        let current_time = times[i];
        let current_price = prices[i];

        // 检查价格是否为NaN或Inf
        if !current_price.is_finite() {
            result[i] = f64::NAN;
            continue;
        }

        let mut final_energy = 0.0;

        // 首先计算time_window秒后的最终能量
        for j in i..n {
            // 跳过当前点
            if j == i {
                continue;
            }

            // 检查时间间隔
            let time_diff = times[j] - current_time;
            if time_diff < time_window {
                continue;
            }

            // 检查价格是否为NaN或Inf
            if !prices[j].is_finite() {
                continue;
            }

            // 获取价格变动
            let price_change = prices[j] - current_price;

            // 根据方向参数筛选
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

            // 计算价格变动比率的绝对值
            final_energy = price_change.abs() / current_price;
            break;
        }

        // 如果最终能量为0，设置为0.0并返回
        if final_energy <= 0.0 {
            result[i] = 0.0;
            continue;
        }

        // 计算一半能量的阈值
        let half_energy = final_energy / 2.0;

        // 再次遍历，找到第一次达到一半能量的时间
        for j in i..n {
            if j == i {
                continue;
            }

            let time_diff = times[j] - current_time;
            if time_diff > time_window {
                break;
            }

            // 检查价格是否为NaN或Inf
            if !prices[j].is_finite() {
                continue;
            }

            // 计算当前时刻的能量
            let price_ratio = (prices[j] - current_price).abs() / current_price;

            // 如果达到一半能量
            if price_ratio >= half_energy {
                result[i] = time_diff;
                break;
            }
        }
        // 如果没有找到达到一半能量的时间，保持默认值 time_window
    }

    result
}

fn large_order_nearby_small_order_time_gap(
    volumes: &[f64],
    exchtimes: &[f64],
    large_quantile: f64,
    small_quantile: f64,
    near_number: i64,
    exclude_same_time: bool,
    order_type: &str,
    flags: Option<&[i64]>,
    flag_filter: &str,
    only_after: bool,
    large_to_large: bool,
) -> Vec<f64> {
    // 转换为Rust类型处理
    let n = volumes.len() as i64;

    // 如果输入数组为空，直接返回空结果
    if n == 0 {
        return Vec::new();
    }

    // 确保输入数组长度一致
    if exchtimes.len() as i64 != n {
        return Vec::new(); // 本用法走不到（长度总是自洽）
    }

    // 处理flags参数
    let flags_vec = if let Some(flags_array) = flags {
        // 确保flags长度与volumes和exchtimes一致
        if flags_array.len() as i64 != n {
            return Vec::new(); // 本用法走不到（长度总是自洽）
        }
        flags_array.to_vec()
    } else {
        // 如果没有提供flags，则创建默认值
        vec![0; n as usize]
    };

    // 转换时间戳为秒单位，并复制为向量
    let times: Vec<f64> = exchtimes.iter().map(|&x| x / 1.0e9).collect();
    let volumes_vec: Vec<f64> = volumes.iter().cloned().collect();

    // 计算分位点
    let mut volumes_sorted = volumes_vec.clone();
    volumes_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let large_threshold_idx = ((n as f64) * large_quantile).ceil() as i64;
    let small_threshold_idx = ((n as f64) * small_quantile).floor() as i64;

    // 确保索引不越界
    let large_threshold_idx = if large_threshold_idx >= n {
        n - 1
    } else {
        large_threshold_idx
    };
    let small_threshold_idx = if small_threshold_idx >= n {
        n - 1
    } else {
        small_threshold_idx
    };

    let large_threshold = volumes_sorted[large_threshold_idx as usize];
    let small_threshold = volumes_sorted[small_threshold_idx as usize];

    // 标记大单和目标订单
    let mut is_large_order = vec![false; n as usize];
    let mut is_target_order = vec![false; n as usize];

    for i in 0..(n as usize) {
        if volumes[i] >= large_threshold {
            is_large_order[i] = true;
        }

        if large_to_large {
            // 当large_to_large=true时，目标订单就是大单
            is_target_order[i] = is_large_order[i];
        } else {
            // 原有逻辑：根据order_type标记目标订单
            match order_type {
                "small" => {
                    // 标记小单
                    if volumes[i] <= small_threshold {
                        is_target_order[i] = true;
                    }
                }
                "mid" => {
                    // 标记中间订单
                    if volumes[i] > small_threshold && volumes[i] < large_threshold {
                        is_target_order[i] = true;
                    }
                }
                "full" => {
                    // 标记所有小于large_threshold的订单
                    if volumes[i] < large_threshold {
                        is_target_order[i] = true;
                    }
                }
                _ => {
                    // 默认为"small"，标记小单
                    if volumes[i] <= small_threshold {
                        is_target_order[i] = true;
                    }
                }
            }
        }
    }

    // 结果数组，初始化为NaN
    let mut result = vec![f64::NAN; n as usize];

    // 对每个大单计算与临近小单的时间间隔
    for i in 0..n as usize {
        if !is_large_order[i] {
            continue; // 跳过非大单
        }

        let large_time = times[i];
        let mut time_gaps = Vec::new();

        // 获取当前大单的flag
        let large_flag = flags_vec[i];

        // 查找前面的目标订单（当only_after=true时跳过）
        if !only_after {
            let mut before_count = 0;
            for j in (0..i).rev() {
                if is_target_order[j] {
                    // 根据flag_filter判断是否满足条件
                    let flag_match = match flag_filter {
                        "same" => flags_vec[j] == large_flag,
                        "diff" => flags_vec[j] != large_flag,
                        _ => true, // "ignore"或其他值，忽略flag判断
                    };

                    if flag_match {
                        let time_diff = (large_time - times[j]).abs();
                        // 如果排除相同时间戳的订单，且时间差为0，则跳过
                        if exclude_same_time && time_diff == 0.0 {
                            continue;
                        }
                        time_gaps.push(time_diff);
                        before_count += 1;
                        if before_count >= near_number as i64 {
                            break;
                        }
                    }
                }
            }
        }

        // 查找后面的目标订单
        let mut after_count = 0;
        for j in (i + 1)..n as usize {
            if is_target_order[j] {
                // 根据flag_filter判断是否满足条件
                let flag_match = match flag_filter {
                    "same" => flags_vec[j] == large_flag,
                    "diff" => flags_vec[j] != large_flag,
                    _ => true, // "ignore"或其他值，忽略flag判断
                };

                if flag_match {
                    let time_diff = (times[j] - large_time).abs();
                    // 如果排除相同时间戳的订单，且时间差为0，则跳过
                    if exclude_same_time && time_diff == 0.0 {
                        continue;
                    }
                    time_gaps.push(time_diff);
                    after_count += 1;
                    if after_count >= near_number as i64 {
                        break;
                    }
                }
            }
        }

        // 如果找到了至少一个小单，选择最小的near_number个间隔计算均值
        if !time_gaps.is_empty() {
            // 对时间间隔进行排序
            time_gaps.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            // 取出最小的near_number个（或全部如果数量不足）
            let count = std::cmp::min(near_number as usize, time_gaps.len());
            let min_gaps: Vec<f64> = time_gaps.iter().take(count).cloned().collect();

            // 计算这些最小间隔的均值
            let avg_gap = min_gaps.iter().sum::<f64>() / min_gaps.len() as f64;
            result[i] = avg_gap;
        }
    }
    result
}

