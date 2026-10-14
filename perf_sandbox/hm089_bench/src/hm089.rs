//! hm089「发起进攻」—— 旧框架模块级 go(date, code) 的 Rust 复现。
//!
//! 旧文件：/home/chenzongwei/pythoncode/成果hm89_发起进攻/hm89_okay.py
//!
//! ════════════════════════════════════════════════════════════════════
//! 一、旧 go 做的事
//! ════════════════════════════════════════════════════════════════════
//! 1) trade_data  = rp.adjust_afternoon(rp.read_trade(code, date))
//!    market_data = rp.adjust_afternoon(rp.read_market(code, date))
//!    取 trade 的 exchtime/price/volume/turnover/flag 与 market 的 exchtime 和 10 档
//!    bid_prc/bid_vol/ask_prc/ask_vol，共 50 个数组。
//!    **exchtime 用的是 `df.exchtime.astype(np.int64)`，也就是 pandas Timestamp 的纳秒值**
//!    （带 ±120ns 的换算噪声，见下面 pandas_ns_of），不是微秒。
//! 2) 四个变体各调一次大挂单微观结构函数（后三个是同一函数的参数变体，第四个换 v3）：
//!      expanded  k1_horizontal=2.0  k2_vertical=5.0   window=100 decay=0.3
//!      expanded  k1_horizontal=1.0  k2_vertical=2.5   window=100 decay=0.5
//!      expanded  k1_horizontal=1.5  k2_vertical=3.5   window=100 decay=0.3
//!      v3        k1_horizontal=2.0  k2_vertical=5.0   window=100 decay=0.3
//!    每个变体返回「9 个 (事件数 × 21) 矩阵 + 9 组列名」，
//!    组合顺序固定 = horizontal/vertical/both × bid/ask/both。
//! 3) 每个矩阵各自再过一次 rp.get_features_factors(df)（**默认参数**：开 lyapunov、
//!    开 threshold_counts、无 max_min、无 append），21 列 → 672 个值。9 个块 = 6048。
//! 4) 四个变体拼成 4 × 6048 = 24192 个值，再按 names_save 取 68 个返回。
//!
//! ════════════════════════════════════════════════════════════════════
//! 二、为什么不用 pipeline_observable_order 之类现成流水线
//! ════════════════════════════════════════════════════════════════════
//! 库里没有这条链的现成实现（observable_order 是另一套按低占比分段的构造）。
//! 能复用的只有两个大挂单 pyfunction 的**算法体**：2026-10-06 在各文件里加了纯 Rust 入口
//! `allo_tris_expanded_inner` / `allo_tris_expanded_v3_inner`（不改 pyfunction 一行），
//! 本文件直接调它们，所以 8000 行的微观结构计算不用重写。
//!
//! ════════════════════════════════════════════════════════════════════
//! 三、取值口径上的三个必须照抄的地方
//! ════════════════════════════════════════════════════════════════════
//! · 时间戳：`pd.to_timedelta(exchtime_us / 1e6, unit="s") + Timestamp("1970-01-01 08:00:00")`
//!   之后 `.astype(np.int64)`。这是 pandas 的 cast_from_unit_vectorized（hm064.rs 的
//!   pandas_ns_of 同式），先除 1e6 变成秒的 f64，再拆整数秒 + rint(frac*1e9)/1e9，
//!   自带 ±120ns 噪声；微观结构里的时间比较落在纳秒上，用「微秒×1000」会差几个纳秒。
//! · 浮点解析：走 pandas 的 precise_xstrtod 复刻（parse_f64_pandas，见下），
//!   read_market 的过滤就是「ask_prc1 / bid_prc1 精确等于 0」。
//! · 空块：某一天某个组合可能一个事件都没有，旧代码对 0 行的 DataFrame 仍然给出 672 个值，
//!   其中只有 trend / mean_above_p90 / mean_below_p10 / lz_complexity 四组是 0.0，其余全 NaN
//!   （实测确认：非 NaN 恰好 84 个、值全是 0）。这里照抄，不能直接给全 NaN。
//!
//! 验证：hm89_okay.py 的四组日期股票，68 个值全部一致（rtol 1e-4）。

use crate::allo_microstructure::{self, AlloBlock};
use crate::allo_microstructure_v3::{self, AlloBlockV3};
use crate::data_paths;
use crate::features_f64::{self, FeatureOpts};
use ndarray::Array2;

/// 21 个微观结构指标（顺序即旧文件 metrics 列表，也是矩阵的列序）。
#[allow(dead_code)]
const METRICS: [&str; 21] = [
    "M1_relative_prominence",
    "M3_flicker_frequency",
    "M7_queue_loitering_duration",
    "M8_frontrun_passive",
    "M9_frontrun_active",
    "M10_ally_retreat_rate",
    "M11a_attack_skewness_opponent",
    "M11b_attack_skewness_ally",
    "M12a_peak_latency_ratio_opponent",
    "M12b_peak_latency_ratio_ally",
    "M13a_courage_acceleration_opponent",
    "M13b_courage_acceleration_ally",
    "M14a_rhythm_entropy_opponent",
    "M14b_rhythm_entropy_ally",
    "M15_fox_tiger_index",
    "M16_shadow_projection_ratio",
    "M17_gravitational_redshift",
    "M19_shielding_thickness_ratio",
    "M20_oxygen_saturation",
    "M21_suffocation_integral",
    "M22_local_survivor_bias",
];

/// 9 种组合（detection_mode × side_filter），顺序是硬契约。
#[allow(dead_code)]
const PREFIXES: [&str; 9] = [
    "horizontal_bid",
    "horizontal_ask",
    "horizontal_both",
    "vertical_bid",
    "vertical_ask",
    "vertical_both",
    "both_bid",
    "both_ask",
    "both_both",
];

/// 4 个变体的名字前缀，顺序即旧文件 tags。
#[allow(dead_code)]
const TAGS: [&str; 4] = ["attack", "optimized", "loose1", "loose2"];

/// 4 个变体的参数：(用 v3 版本?, k1_horizontal, k2_vertical, decay_threshold)。
const VARIANTS: [(bool, f64, f64, f64); 4] = [
    (false, 2.0, 5.0, 0.3),
    (false, 1.0, 2.5, 0.5),
    (false, 1.5, 3.5, 0.3),
    (true, 2.0, 5.0, 0.3),
];

/// 窗口快照数（四个变体都是 100）。
const WINDOW_SIZE: usize = 100;

/// 每个块（21 列）经 get_features_factors 之后的取值个数：18 统计 × 21 + 210 相关 + lyapunov 21 + 复杂度 3 × 21。
const PER_BLOCK: usize = 18 * 21 + 21 * 20 / 2 + 21 + 3 * 21;
/// 每只股票的总长度 = 4 个变体 × 9 个块 × 672。
const FULL: usize = 4 * 9 * PER_BLOCK;

/// 与旧库 `get_features_factors` 默认参数一致的口径（关 curvature）。
const OPTS: FeatureOpts = FeatureOpts {
    with_threshold_counts: true,
    with_curvature: false,
    with_max_min: false,
    with_lyapunov_exponent: true,
    with_complexity: true,
};

const NAMES: [&str; 68] = [
    "attack_both_ask_M11a_attack_skewness_opponent_p5",
    "attack_both_ask_M14b_rhythm_entropy_ally_entropy_1d",
    "attack_both_ask_M22_local_survivor_bias_median",
    "attack_both_ask_M3_flicker_frequency_cv",
    "attack_both_bid_M22_local_survivor_bias_p25",
    "attack_both_bid_M7_queue_loitering_duration_corr_both_bid_M21_suffocation_integral",
    "attack_both_both_M15_fox_tiger_index_p75",
    "attack_both_both_M8_frontrun_passive_median",
    "attack_vertical_ask_M12b_peak_latency_ratio_ally_mean_above_p90",
    "attack_vertical_ask_M21_suffocation_integral_period_diff",
    "attack_vertical_ask_M22_local_survivor_bias_median",
    "loose1_both_ask_M1_relative_prominence_corr_both_ask_M21_suffocation_integral",
    "loose1_both_bid_M14a_rhythm_entropy_opponent_corr_both_bid_M16_shadow_projection_ratio",
    "loose1_both_bid_M3_flicker_frequency_autocorr1",
    "loose1_both_bid_M7_queue_loitering_duration_corr_both_bid_M21_suffocation_integral",
    "loose1_both_bid_M7_queue_loitering_duration_p75",
    "loose1_both_both_M11b_attack_skewness_ally_kurt",
    "loose1_both_both_M12b_peak_latency_ratio_ally_autocorr1_abs",
    "loose1_both_both_M20_oxygen_saturation_entropy_1d",
    "loose1_both_both_M22_local_survivor_bias_mean_above_p90",
    "loose1_both_both_M22_local_survivor_bias_mean_above_p90_dup2",
    "loose1_vertical_ask_M15_fox_tiger_index_p75",
    "loose1_vertical_ask_M7_queue_loitering_duration_corr_vertical_ask_M16_shadow_projection_ratio",
    "loose1_vertical_both_M11b_attack_skewness_ally_kurt",
    "loose1_vertical_both_M12b_peak_latency_ratio_ally_corr_vertical_both_M17_gravitational_redshift",
    "loose1_vertical_both_M7_queue_loitering_duration_period_diff",
    "loose2_both_ask_M22_local_survivor_bias_p25",
    "loose2_both_bid_M15_fox_tiger_index_p95",
    "loose2_both_bid_M22_local_survivor_bias_median",
    "loose2_both_bid_M22_local_survivor_bias_p5",
    "loose2_both_both_M22_local_survivor_bias_p5",
    "loose2_vertical_ask_M22_local_survivor_bias_median",
    "loose2_vertical_bid_M16_shadow_projection_ratio_mean",
    "loose2_vertical_bid_M22_local_survivor_bias_median",
    "optimized_both_ask_M11b_attack_skewness_ally_p95",
    "optimized_both_ask_M12b_peak_latency_ratio_ally_period_diff",
    "optimized_both_ask_M13b_courage_acceleration_ally_corr_both_ask_M17_gravitational_redshift",
    "optimized_both_ask_M20_oxygen_saturation_entropy_1d",
    "optimized_both_ask_M22_local_survivor_bias_mean_above_p90",
    "optimized_both_ask_M7_queue_loitering_duration_iqr",
    "optimized_both_bid_M11b_attack_skewness_ally_std",
    "optimized_both_bid_M20_oxygen_saturation_autocorr1_abs",
    "optimized_both_bid_M20_oxygen_saturation_p95",
    "optimized_both_bid_M3_flicker_frequency_corr_both_bid_M20_oxygen_saturation",
    "optimized_both_bid_M3_flicker_frequency_corr_both_bid_M21_suffocation_integral",
    "optimized_both_bid_M9_frontrun_active_period_ratio",
    "optimized_both_both_M11b_attack_skewness_ally_entropy_1d",
    "optimized_both_both_M17_gravitational_redshift_corr_both_both_M20_oxygen_saturation",
    "optimized_both_both_M17_gravitational_redshift_p25",
    "optimized_both_both_M3_flicker_frequency_corr_both_both_M13b_courage_acceleration_ally",
    "optimized_both_both_M3_flicker_frequency_p25",
    "optimized_both_both_M7_queue_loitering_duration_p75",
    "optimized_vertical_ask_M11a_attack_skewness_opponent_p5",
    "optimized_vertical_ask_M1_relative_prominence_corr_vertical_ask_M21_suffocation_integral",
    "optimized_vertical_ask_M1_relative_prominence_corr_vertical_ask_M3_flicker_frequency",
    "optimized_vertical_ask_M1_relative_prominence_trend",
    "optimized_vertical_ask_M20_oxygen_saturation_p95",
    "optimized_vertical_ask_M3_flicker_frequency_corr_vertical_ask_M16_shadow_projection_ratio",
    "optimized_vertical_bid_M14b_rhythm_entropy_ally_corr_vertical_bid_M15_fox_tiger_index",
    "optimized_vertical_bid_M16_shadow_projection_ratio_p95",
    "optimized_vertical_bid_M20_oxygen_saturation_mean",
    "optimized_vertical_bid_M7_queue_loitering_duration_trend",
    "optimized_vertical_both_M10_ally_retreat_rate_corr_vertical_both_M16_shadow_projection_ratio",
    "optimized_vertical_both_M11b_attack_skewness_ally_mean_above_p90",
    "optimized_vertical_both_M11b_attack_skewness_ally_p95",
    "optimized_vertical_both_M12b_peak_latency_ratio_ally_corr_vertical_both_M17_gravitational_redshift",
    "optimized_vertical_both_M16_shadow_projection_ratio_corr_vertical_both_M17_gravitational_redshift",
    "optimized_vertical_both_M3_flicker_frequency_corr_vertical_both_M11b_attack_skewness_ally",
];

const IDX: [usize; 68] = [
    4815, 5347, 4745, 4915, 4178, 4465, 5537, 5400, 3033, 3001, 2729, 17196, 16682, 16360,
    16561, 16277, 17563, 17733, 18120, 17828, 17828, 14945, 15213, 15547, 15984, 15752, 22994,
    22358, 22217, 22301, 23645, 20873, 20175, 20201, 10927, 11055, 11299, 11400, 11108, 10943,
    10129, 10350, 10266, 10494, 10495, 10399, 12061, 12003, 11566, 11831, 11551, 11573, 8847,
    9132, 9114, 9009, 8922, 9147, 8624, 8247, 8082, 8339, 9885, 9751, 9583, 9936, 9981, 9811,
];

pub fn names() -> Vec<String> {
    NAMES.iter().map(|s| s.to_string()).collect()
}

// ============================================================================
// 读数据：f64 + pandas 口径
// ============================================================================

/// C 的 rint：舍到最近的整数，正好 .5 时取偶数（与 numpy 的 rint 一致）。
#[inline]
fn rint(x: f64) -> f64 {
    if !x.is_finite() {
        return x;
    }
    let f = x.floor();
    let d = x - f;
    if d < 0.5 {
        f
    } else if d > 0.5 {
        f + 1.0
    } else if (f as i64) % 2 == 0 {
        f
    } else {
        f + 1.0
    }
}

/// 复刻 `pd.to_timedelta(exchtime_us / 1e6, unit="s").astype("int64")` 的纳秒值
/// （与 hm064.rs / hm078.rs 的同名函数同式）。不带那 8 小时偏移，调用方自己补。
#[inline]
fn pandas_ns_of(exchtime_us: i64) -> i64 {
    let val = (exchtime_us as f64) / 1_000_000.0;
    let base = val as i64; // C 的转 int64 = 向零截断
    let frac = val - (base as f64);
    let fr = rint(frac * 1e9) / 1e9;
    base * 1_000_000_000 + (fr * 1e9) as i64
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

/// 取字段的整数前缀（exchtime 是 `1740960939000000.0` 这种浮点串）。
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

/// pandas C 解析器 `precise_xstrtod` 的逐行复刻（hm033/hm078/hm090 里各有一份）。
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

fn num(raw: &[u8]) -> f64 {
    let b = trim_ascii(raw);
    if b.is_empty() {
        return f64::NAN;
    }
    parse_f64_pandas(b)
}

/// 按逗号把一行切成至多 `N` 个字段（返回 (字段, 逗号数)）。
fn split_fields<'a>(line: &'a [u8], out: &mut [&'a [u8]], n: usize) -> usize {
    let mut commas = 0usize;
    let mut start = 0usize;
    for (i, &b) in line.iter().enumerate() {
        if b == b',' {
            if commas < n {
                out[commas] = &line[start..i];
            }
            commas += 1;
            start = i + 1;
        }
    }
    if commas < n {
        out[commas] = &line[start..];
    }
    commas + 1
}

/// 读文件的字节（在 level2 根目录里按 <date>/<subdir>/<code>_<date>_<subdir>.csv 找）。
fn read_csv_bytes(code: &str, date: i64, subdir: &str) -> Option<Vec<u8>> {
    let filename = format!("{}_{}_{}.csv", code, date, subdir);
    for root in data_paths::level2_roots() {
        let path = root.join(date.to_string()).join(subdir).join(&filename);
        let __t = std::time::Instant::now();
        let r = std::fs::read(&path);
        features_f64::T_FILE_READ.fetch_add(__t.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
        if let Ok(bytes) = r {
            return Some(bytes);
        }
    }
    None
}

/// 窗口常量（adjust_afternoon 的两段，微秒级闭区间）。
const DAY_NS: i64 = 86_400 * 1_000_000_000;
const OFFSET_NS: i64 = 8 * 3600 * 1_000_000_000;
const SHIFT_NS: i64 = 90 * 60 * 1_000_000_000;
const LD_0930: i64 = (9 * 3600 + 30 * 60) * 1_000_000_000;
const LD_1130: i64 = (11 * 3600 + 30 * 60) * 1_000_000_000;
const LD_1300: i64 = 13 * 3600 * 1_000_000_000;
const LD_1457: i64 = (14 * 3600 + 57 * 60) * 1_000_000_000;

/// 读全天逐笔，复刻 `rp.read_trade`（丢 flag==32）+ `rp.adjust_afternoon`。
/// 返回 (exchtime 纳秒, price, volume, turnover, flag)，行序 = 上午全部在前、下午全部在后。
fn read_trade_day(code: &str, date: i64) -> Option<(Vec<i64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<i32>)> {
    const COL_EXCHTIME: usize = 4;
    const COL_PRICE: usize = 7;
    const COL_VOLUME: usize = 8;
    const COL_TURNOVER: usize = 9;
    const COL_FLAG: usize = 10;
    const NCOLS: usize = 15;

    let data = read_csv_bytes(code, date, "transaction")?;
    let mut morning: Vec<(i64, f64, f64, f64, i32)> = Vec::new();
    let mut afternoon: Vec<(i64, f64, f64, f64, i32)> = Vec::new();
    for (lineno, line) in data.split(|&b| b == b'\n').enumerate() {
        if lineno == 0 || line.is_empty() {
            continue;
        }
        let mut fields: [&[u8]; NCOLS] = [&[][..]; NCOLS];
        let nf = split_fields(line, &mut fields, NCOLS);
        if nf < NCOLS {
            continue;
        }
        let exchtime_us = int_prefix(fields[COL_EXCHTIME]);
        if exchtime_us == 0 {
            continue;
        }
        let flag = int_prefix(fields[COL_FLAG]) as i32;
        if flag == 32 {
            continue; // read_trade 默认 with_retreat=0
        }
        let tns = OFFSET_NS + pandas_ns_of(exchtime_us);
        let tod = tns.rem_euclid(DAY_NS);
        let row = (
            tns,
            num(fields[COL_PRICE]),
            num(fields[COL_VOLUME]),
            num(fields[COL_TURNOVER]),
            flag,
        );
        if (LD_0930..=LD_1130).contains(&tod) {
            morning.push(row);
        } else if (LD_1300..=LD_1457).contains(&tod) {
            afternoon.push((tns - SHIFT_NS, row.1, row.2, row.3, row.4));
        }
    }
    morning.extend(afternoon);
    let mut t = Vec::with_capacity(morning.len());
    let mut p = Vec::with_capacity(morning.len());
    let mut v = Vec::with_capacity(morning.len());
    let mut to = Vec::with_capacity(morning.len());
    let mut f = Vec::with_capacity(morning.len());
    for (a, b, c, d, e) in morning {
        t.push(a);
        p.push(b);
        v.push(c);
        to.push(d);
        f.push(e);
    }
    Some((t, p, v, to, f))
}

/// 一天的盘口：exchtime 纳秒 + 10 档 bid/ask 价量。
struct MarketDay {
    exchtime: Vec<i64>,
    bid_prc: [Vec<f64>; 10],
    bid_vol: [Vec<f64>; 10],
    ask_prc: [Vec<f64>; 10],
    ask_vol: [Vec<f64>; 10],
}

/// 读全天盘口，复刻 `rp.read_market`（丢 ask_prc1==0 或 bid_prc1==0 的行）+ `rp.adjust_afternoon`。
fn read_market_day(code: &str, date: i64) -> Option<MarketDay> {
    const COL_EXCHTIME: usize = 4;
    const COL_ASK_PRC_BASE: usize = 21; // ask_prc1
    const NCOLS: usize = 61;
    const LEVELS: usize = 10;

    let data = read_csv_bytes(code, date, "market_data")?;
    // 时间戳是 1.7e18 量级的纳秒，超出 f64 能精确表示整数的范围（2^53），必须用 i64 存。
    let mut morning: Vec<(i64, [f64; 40])> = Vec::new();
    let mut afternoon: Vec<(i64, [f64; 40])> = Vec::new();
    for (lineno, line) in data.split(|&b| b == b'\n').enumerate() {
        if lineno == 0 || line.is_empty() {
            continue;
        }
        let mut fields: [&[u8]; NCOLS] = [&[][..]; NCOLS];
        let nf = split_fields(line, &mut fields, NCOLS);
        if nf < NCOLS {
            continue;
        }
        // read_market 的 with_high_low_limited=0
        if num(fields[COL_ASK_PRC_BASE]) == 0.0 || num(fields[COL_ASK_PRC_BASE + 2]) == 0.0 {
            continue;
        }
        let exchtime_us = int_prefix(fields[COL_EXCHTIME]);
        if exchtime_us == 0 {
            continue;
        }
        let tns = OFFSET_NS + pandas_ns_of(exchtime_us);
        let tod = tns.rem_euclid(DAY_NS);
        let mut row = [0.0f64; 40];
        for i in 0..LEVELS {
            let base = COL_ASK_PRC_BASE + i * 4;
            row[i * 4] = num(fields[base]); // ask_prc{i+1}
            row[i * 4 + 1] = num(fields[base + 1]); // ask_vol{i+1}
            row[i * 4 + 2] = num(fields[base + 2]); // bid_prc{i+1}
            row[i * 4 + 3] = num(fields[base + 3]); // bid_vol{i+1}
        }
        if (LD_0930..=LD_1130).contains(&tod) {
            morning.push((tns, row));
        } else if (LD_1300..=LD_1457).contains(&tod) {
            afternoon.push((tns - SHIFT_NS, row));
        }
    }
    morning.extend(afternoon);

    let n = morning.len();
    let mut out = MarketDay {
        exchtime: Vec::with_capacity(n),
        bid_prc: std::array::from_fn(|_| Vec::with_capacity(n)),
        bid_vol: std::array::from_fn(|_| Vec::with_capacity(n)),
        ask_prc: std::array::from_fn(|_| Vec::with_capacity(n)),
        ask_vol: std::array::from_fn(|_| Vec::with_capacity(n)),
    };
    for (tns, row) in morning.iter() {
        out.exchtime.push(*tns);
        for i in 0..LEVELS {
            out.ask_prc[i].push(row[i * 4]);
            out.ask_vol[i].push(row[i * 4 + 1]);
            out.bid_prc[i].push(row[i * 4 + 2]);
            out.bid_vol[i].push(row[i * 4 + 3]);
        }
    }
    Some(out)
}

// ============================================================================
// 一块矩阵 → 672 个值
// ============================================================================

/// 0 事件时旧代码给出的 672 个值：只有 trend / mean_above_p90 / mean_below_p10 /
/// lz_complexity 四组是 0.0（实测确认：非 NaN 恰好 84 个、值全是 0），其余全 NaN。
/// 列名用一个 2 行的假矩阵从同一条特征通路取，保证与正常路径的取名逐字一致。
fn empty_block(n_features: usize, col_names: &[String]) -> Vec<f64> {
    let dummy = Array2::<f64>::zeros((2, n_features));
    let (_, fnames) =
        features_f64::get_features_factors_rust_full_ex(&dummy.view(), col_names, OPTS, None);
    fnames
        .iter()
        .map(|nm| {
            if nm.ends_with("_trend")
                || nm.ends_with("_mean_above_p90")
                || nm.ends_with("_mean_below_p10")
                || nm.ends_with("_lz_complexity")
            {
                0.0
            } else {
                f64::NAN
            }
        })
        .collect()
}

/// 把一块 (n_events × n_features) 的 row-major 矩阵过一遍 f64 特征通路，得到 672 个值。
fn block_features(
    n_events: usize,
    n_features: usize,
    values: &[f64],
    col_names: &[String],
) -> Vec<f64> {
    if n_events == 0 {
        return empty_block(n_features, col_names);
    }
    match Array2::from_shape_vec((n_events, n_features), values.to_vec()) {
        Ok(m) => features_f64::get_features_factors_rust_full_ex(&m.view(), col_names, OPTS, None).0,
        Err(_) => empty_block(n_features, col_names),
    }
}

pub fn compute_stock(code: &str, date: i64) -> Option<Vec<f32>> {
    let (t_exch, t_price, t_vol, t_turn, t_flag) = read_trade_day(code, date)?;
    let mk = read_market_day(code, date)?;
    if t_exch.is_empty() || mk.exchtime.is_empty() {
        return None;
    }
    let bid_prc: [&[f64]; 10] = std::array::from_fn(|i| mk.bid_prc[i].as_slice());
    let bid_vol: [&[f64]; 10] = std::array::from_fn(|i| mk.bid_vol[i].as_slice());
    let ask_prc: [&[f64]; 10] = std::array::from_fn(|i| mk.ask_prc[i].as_slice());
    let ask_vol: [&[f64]; 10] = std::array::from_fn(|i| mk.ask_vol[i].as_slice());

    let mut full: Vec<f64> = Vec::with_capacity(FULL);
    for &(is_v3, k1, k2, decay) in VARIANTS.iter() {
        let blocks: Vec<(usize, usize, Vec<f64>, Vec<String>)> = if is_v3 {
            allo_microstructure_v3::allo_tris_expanded_v3_inner(
                &t_exch,
                &t_price,
                &t_vol,
                &t_turn,
                &t_flag,
                &mk.exchtime,
                &bid_prc,
                &bid_vol,
                &ask_prc,
                &ask_vol,
                k1,
                k2,
                WINDOW_SIZE,
                decay,
            )
            .into_iter()
            .map(|b: AlloBlockV3| (b.n_events, b.n_features, b.values, b.names))
            .collect()
        } else {
            allo_microstructure::allo_tris_expanded_inner(
                &t_exch,
                &t_price,
                &t_vol,
                &t_turn,
                &t_flag,
                &mk.exchtime,
                &bid_prc,
                &bid_vol,
                &ask_prc,
                &ask_vol,
                k1,
                k2,
                WINDOW_SIZE,
                decay,
            )
            .into_iter()
            .map(|b: AlloBlock| (b.n_events, b.n_features, b.values, b.names))
            .collect()
        };
        for (n_events, n_features, values, col_names) in blocks {
            full.extend(block_features(n_events, n_features, &values, &col_names));
        }
    }
    if full.len() != FULL {
        return None;
    }
    Some(IDX.iter().map(|&i| full[i] as f32).collect())
}

pub fn compute_cross(_date: i64) -> Option<(Vec<String>, Vec<f32>)> {
    None
}


/// 计时版入口：算法与 compute_stock 逐行相同，只在阶段边界读表。
pub struct Timings {
    pub total_ns: u128,
    pub file_read_ns: u128,
    pub read_trade_ns: u128,
    pub read_market_ns: u128,
    pub micro_ns: Vec<u128>,
    pub n_events: Vec<usize>,
    pub feat_ns: Vec<u128>,
}

pub fn compute_stock_timed(code: &str, date: i64) -> Option<(Vec<f32>, Timings)> {
    use std::sync::atomic::Ordering;
    let t_all = std::time::Instant::now();
    let t0 = std::time::Instant::now();
    let (t_exch, t_price, t_vol, t_turn, t_flag) = read_trade_day(code, date)?;
    let read_trade = t0.elapsed().as_nanos();
    let t1 = std::time::Instant::now();
    let mk = read_market_day(code, date)?;
    let read_market = t1.elapsed().as_nanos();
    if t_exch.is_empty() || mk.exchtime.is_empty() {
        return None;
    }
    let bid_prc: [&[f64]; 10] = std::array::from_fn(|i| mk.bid_prc[i].as_slice());
    let bid_vol: [&[f64]; 10] = std::array::from_fn(|i| mk.bid_vol[i].as_slice());
    let ask_prc: [&[f64]; 10] = std::array::from_fn(|i| mk.ask_prc[i].as_slice());
    let ask_vol: [&[f64]; 10] = std::array::from_fn(|i| mk.ask_vol[i].as_slice());

    let mut micro_ns: Vec<u128> = Vec::new();
    let mut n_events: Vec<usize> = Vec::new();
    let mut feat_ns: Vec<u128> = Vec::new();
    let mut full: Vec<f64> = Vec::with_capacity(FULL);
    for &(is_v3, k1, k2, decay) in VARIANTS.iter() {
        let tm = std::time::Instant::now();
        let blocks: Vec<(usize, usize, Vec<f64>, Vec<String>)> = if is_v3 {
            allo_microstructure_v3::allo_tris_expanded_v3_inner(
                &t_exch, &t_price, &t_vol, &t_turn, &t_flag, &mk.exchtime,
                &bid_prc, &bid_vol, &ask_prc, &ask_vol, k1, k2, WINDOW_SIZE, decay,
            )
            .into_iter()
            .map(|b: AlloBlockV3| (b.n_events, b.n_features, b.values, b.names))
            .collect()
        } else {
            allo_microstructure::allo_tris_expanded_inner(
                &t_exch, &t_price, &t_vol, &t_turn, &t_flag, &mk.exchtime,
                &bid_prc, &bid_vol, &ask_prc, &ask_vol, k1, k2, WINDOW_SIZE, decay,
            )
            .into_iter()
            .map(|b: AlloBlock| (b.n_events, b.n_features, b.values, b.names))
            .collect()
        };
        micro_ns.push(tm.elapsed().as_nanos());
        let __ne: usize = blocks.iter().map(|b| b.0).sum();
        n_events.push(__ne);
        for (n_events, n_features, values, col_names) in blocks {
            let tf = std::time::Instant::now();
            full.extend(block_features(n_events, n_features, &values, &col_names));
            feat_ns.push(tf.elapsed().as_nanos());
        }
    }
    if full.len() != FULL {
        return None;
    }
    let out: Vec<f32> = IDX.iter().map(|&i| full[i] as f32).collect();
    let timings = Timings {
        total_ns: t_all.elapsed().as_nanos(),
        file_read_ns: features_f64::T_FILE_READ.load(Ordering::Relaxed) as u128,
        read_trade_ns: read_trade,
        read_market_ns: read_market,
        micro_ns,
        n_events,
        feat_ns,
    };
    Some((out, timings))
}
