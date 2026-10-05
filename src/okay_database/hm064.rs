//! hm64「银河落九天」的 Rust 复现（旧框架 design_whatever_out.TickDataBase）。
//!
//! 旧文件：/home/chenzongwei/pythoncode/成果hm64_银河落九天/hm64_okay.py
//!
//! ════════════════════════════════════════════════════════════════════
//! 旧代码逐步对照（写代码前先抄在这里，方便复核）
//! ════════════════════════════════════════════════════════════════════
//! 1) df = read_trade_data_single(with_retreat=0) → 丢掉 flag==32 的撤单行；
//!    参数 get_many=0 / read_full=0 / change_time=1，也就是一整天一张表，
//!    窗口由旧代码自己切。
//! 2) df = dwo.adjust_afternoon(df)：只保留真实钟点 [09:30:00, 11:30:00] 与
//!    [13:00:00, 14:57:00] 两段（微秒闭区间），下午段整体前移 90 分钟，
//!    结果按「上午段在前、下午段在后」拼接。
//! 3) df_morning = between_time('09:30:00','11:30:00')、df_afternoon =
//!    between_time('11:30:00','13:27:00')，都是在**平移之后**的时间轴上切的。
//!    因为平移后下午段起点正好是 11:30:00，所以
//!      上午段 = 上午的行 + 恰好 13:00:00.000000 的下午行
//!      下午段 = 下午的行 + 恰好 11:30:00.000000 的上午行
//!    边界那一笔在上午段与下午段里**各出现一次**（旧代码就是这么算的）。
//! 4) 三个时段各自调 one(df)：全部 / flag==66 / flag==83 三套，
//!    再各加一版 flag_different=True、一版 with_forth=True、以及两者的组合，
//!    共 8 套变体；每套调 rp.trade_peak_analysis(..., top_tier1=0.01,
//!    top_tier2=0.1, SECONDS=30, ...) 得到「每行一个高峰 × 17 个特征」的矩阵。
//! 5) 每个矩阵按 get_res 压成 110 个数（8 个列间相关 + 6 组 × 17 列统计），
//!    8 套变体 = 880，再拼 3 个时段 = 2640。
//!
//! names_single（2640 个，旧文件 107~109 行）：
//!   110 个基础名 × 8 个变体后缀 × 3 个时段后缀；
//!   变体后缀顺序 = _全部 / _66 / _83 / _diff / _包含向前 / _66_包含向前 /
//!   _83_包含向前 / _diff_包含向前；时段后缀顺序 = _全天 / _上午 / _下午。
//!   ！！std 那一组的**名字顺序**与值顺序不同：旧文件 line 107 里是
//!     ...'时间趋势_std','时间重心_std','时间自相关_std'，
//!   而 line 93 的取值顺序是 ...'时间趋势_std','时间自相关_std','时间重心_std'。
//!   位置一一对应，所以列名 '时间自相关_std' 上挂的是「成交量加权时间距离的 std」。
//!   本文件照抄这个名字顺序（names()）与取值顺序（get_res），保持与旧库一致。
//!
//! ════════════════════════════════════════════════════════════════════
//! 关键坑：旧代码喂给 trade_peak_analysis 的时间是 pandas Timestamp 的纳秒
//! ════════════════════════════════════════════════════════════════════
//! 旧代码传的是 df.exchtime.to_numpy(int)，即 datetime64[ns] 的纳秒整数。
//! 这个数不是「微秒 × 1000」，而是 pandas 把 exchtime（微秒）先除以 1e6 得到
//! **秒的浮点数**、再经 cast_from_unit_vectorized 转出来的，带 ±120ns 的舍入噪声：
//!
//!     base = (int64) trunc(val)              # val = exchtime_us / 1e6
//!     frac = val - (double)base
//!     frac = rint(frac * 1e9) / 1e9          # np.round(frac, 9)，p = log10(1e9) = 9
//!     ns   = base * 1_000_000_000 + (int64)(frac * 1e9)
//!     （源码：pandas/_libs/tslibs/conversion.pyx cast_from_unit_vectorized /
//!       precision_from_unit）
//!
//! 这 100ns 级的噪声不是可以忽略的：trade_peak_analysis 用
//! |Δt| <= 30e9 ns 判断小峰，Δt 的真值恰好等于 30.000000 秒时，纳秒噪声
//! 会让「这一笔算不算小峰」翻面，实测 000001@20250303 的 2640 个值里有 14 个
//! 会因此变化（最大相对差 39%）。所以必须照上式复刻，不能图省事用微秒×1000
//! （本文件 pandas_ns_of() 就是这段）。
//!
//! 另外两处必须留意的等价/差异：
//! · 撤单过滤自己按数值筛 flag != 32（库里 with_retreat 按字节比 "32"，
//!   2025 年的明细写的是 "32.0"，那个过滤不生效），与规范一致。
//! · 下午平移量：exchtime 毫秒级精度平移由时间戳整数减法完成，这里直接把
//!   纳秒值减 5400e9，与原样一致。
//!
//! ════════════════════════════════════════════════════════════════════
//! get_res 的 110 个量：pandas 语义逐条复刻
//! ════════════════════════════════════════════════════════════════════
//! 旧代码在 DataFrame 上取 corr / mean / std / skew / kurt / corrwith / trend_2d：
//! · 8 个列间相关：df.corr() → np.corrcoef，样本<2 或某列方差为 0 → NaN
//! · mean/std：numpy 两遍算法；std 是 ddof=1，count-1<=0 → NaN
//! · skew：pandas nanops.nanskew，先对 m2/m3 做 |x|<1e-14 清零，m2==0 → 0，
//!   count<3 → NaN，再乘 n*sqrt(n-1)/(n-2)
//! · kurt：pandas nanops.nankurt，m2/m4 先清零，分子 n(n+1)(n-1)m4、
//!   分母 (n-2)(n-3)m2²，各再清零一次，分母为 0 → 0，count<4 → NaN，
//!   最后减 3(n-1)²/((n-2)(n-3))
//! · corrwith(df.shift(1))：逐列 corr(x[1:], x[:-1])
//! · trend_2d(axis=0)：与 src/time_series/trend_mod.rs 的 calculate_trend_1d 同式
//!   （过滤非有限值、下标从 1 起、顺序累加）；这个函数是 #[pyfunction]，
//!   worker 链路不能调，所以在这里另写了一份纯 Rust 的同式实现。
//! 求和用顺序累加：本轮实测（四组日期股票、2640 值逐个比）最大相对差 4e-12，
//! 远小于验收的 1e-4；这里不逐列搬 numpy 的成对求和。
//!
//! 验证：hm64_okay.py 对 20250303:000001 / 20250304:600519 / 20180601:000002 /
//! 20231020:300750 四组，2640 值全部一致（rtol 1e-4）。

use crate::fast_csv_reader::read_trade_fast_inner;

/// 每个时段的因子数 = 110 × 8 套变体。
pub const PER_SEG: usize = 880;
/// 每只股票输出的因子数 = 880 × 3 个时段。
pub const N_OUT: usize = 2640;

/// 时间窗口（秒），与旧代码 SECONDS=30 一致。
const SECONDS_NS: i64 = 30 * 1_000_000_000;
const TOP_TIER1: f64 = 0.01;
const TOP_TIER2: f64 = 0.1;

const US: i64 = 1_000_000;
const NS: i64 = 1_000_000_000;
const DAY_US: i64 = 86_400_000_000;
/// 原始 exchtime（UTC 微秒）到「北京时区时间戳」的偏移。
const OFFSET_US: i64 = 28_800_000_000;
/// adjust_afternoon 的下午前移量。
const SHIFT_US: i64 = 5_400_000_000;

const LD_0930: i64 = (9 * 3600 + 30 * 60) * US;
const LD_1130: i64 = (11 * 3600 + 30 * 60) * US;
const LD_1300: i64 = 13 * 3600 * US;
const LD_1327: i64 = (13 * 3600 + 27 * 60) * US;
const LD_1457: i64 = (14 * 3600 + 57 * 60) * US;

/// 17 个特征的取值顺序（旧代码 df 的列顺序，硬契约）。
const FEAT17: [&str; 17] = [
    "小峰成交量总和比值",
    "小峰平均成交量比值",
    "小峰个数",
    "时间间隔均值秒",
    "成交量时间相关系数",
    "DTW距离",
    "成交量变异系数",
    "成交量偏度",
    "成交量峰度",
    "成交量趋势",
    "成交量自相关",
    "时间变异系数",
    "时间偏度",
    "时间峰度",
    "时间趋势",
    "时间自相关",
    // 库里的特征名是「成交量加权时间距离」，旧代码解包时叫「时间重心」。
    // 名字按旧代码的写法走。
    "时间重心",
];

/// 8 个相关对的列下标（旧代码 get_res 的前 8 个返回值）。
const PAIRS: [(usize, usize); 8] = [
    (0, 3),
    (1, 3),
    (2, 3),
    (6, 11),
    (7, 12),
    (8, 13),
    (9, 14),
    (10, 15),
];

/// 8 套高峰定义变体的名字后缀，顺序 = 旧代码 get_res 的拼接顺序。
const VARIANT_SUFFIX: [&str; 8] = [
    "_全部",
    "_66",
    "_83",
    "_diff",
    "_包含向前",
    "_66_包含向前",
    "_83_包含向前",
    "_diff_包含向前",
];

/// 3 个时段后缀，顺序 = res1 + res2 + res3。
const SEG_SUFFIX: [&str; 3] = ["_全天", "_上午", "_下午"];

/// 相关对的名字里，3 号特征写作「时间间隔均值」（不带「秒」），
/// 其余位置与 FEAT17 同名（旧文件 line 107 的前 8 项就是这样）。
fn corr_name(i: usize) -> &'static str {
    if i == 3 {
        "时间间隔均值"
    } else {
        FEAT17[i]
    }
}

/// 110 个基础名（顺序 = 旧代码 names 的前 110 项）。
///
/// 顺序是 corr(8) + mean(17) + std(17) + skew(17) + kurt(17) + autocorr(17) + trend(17)，
/// 其中 std 那一组用 std_names()（15/16 两号互换），其余五组按 FEAT17 顺序。
fn base_names() -> Vec<String> {
    let mut v = Vec::with_capacity(110);
    for &(a, b) in PAIRS.iter() {
        v.push(format!("corr_{}_{}", corr_name(a), corr_name(b)));
    }
    for s in ["_mean"] {
        for f in FEAT17.iter() {
            v.push(format!("{}{}", f, s));
        }
    }
    v.extend(std_names());
    for s in ["_skew", "_kurt", "_autocorr", "_trend"] {
        for f in FEAT17.iter() {
            v.push(format!("{}{}", f, s));
        }
    }
    debug_assert_eq!(v.len(), 110);
    v
}

/// std 那一组的名字顺序（旧文件里唯一一处与取值顺序不一致的地方）。
fn std_names() -> Vec<String> {
    let mut v: Vec<String> = FEAT17.iter().map(|f| format!("{}_std", f)).collect();
    // 旧文件 names 里这两号互换：..., 时间趋势_std, 时间重心_std, 时间自相关_std
    v.swap(15, 16);
    v
}

pub fn names() -> Vec<String> {
    let base = base_names(); // 110 个
    let mut out = Vec::with_capacity(N_OUT);
    for seg in SEG_SUFFIX.iter() {
        for var in VARIANT_SUFFIX.iter() {
            for n in base.iter() {
                out.push(format!("{}{}{}", n, var, seg));
            }
        }
    }
    debug_assert_eq!(out.len(), N_OUT);
    out
}

/// 横截面版本：本系列是逐股系列。
pub fn compute_cross(_date: i64) -> Option<(Vec<String>, Vec<f32>)> {
    None
}

// ============================================================================
// pandas 的浮点→纳秒转换（cast_from_unit_vectorized，单位 s → ns）
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

/// 复刻 `pd.to_timedelta(exchtime_us / 1e6, unit='s').to_numpy(int)` 的逐行结果。
///
/// val 是「秒」的 f64（= exchtime_us / 1e6）；m=1e9、p=9。
#[inline]
fn pandas_ns_of(exchtime_us: i64) -> i64 {
    let val = (exchtime_us as f64) / 1_000_000.0;
    let base = val as i64; // C 的 int64 转换 = 向零截断
    let frac = val - (base as f64);
    // np.round(frac, 9)：rint(frac * 1e9) / 1e9，rint 是「四舍六入五成双」
    let fr = rint(frac * 1e9) / 1e9;
    base * NS + (fr * 1e9) as i64
}

// ============================================================================
// 一个时段的一条记录（已平移后的时间轴）
// ============================================================================

#[derive(Clone, Copy)]
struct Row {
    /// 平移后的纳秒时间戳（就是旧代码 trade_peak_analysis 拿到的那种值）
    ns: i64,
    /// 平移后的当日钟点（微秒），只用来切上午/下午两段
    tod: i64,
    vol: f64,
    flag: i32,
}

/// 读全天逐笔，做撤单过滤 + adjust_afternoon 平移，并切出三个时段。
fn build_segments(code: &str, date: i64) -> Option<[Vec<Row>; 3]> {
    let trades = read_trade_fast_inner(code, date, true, false, usize::MAX).ok()?;

    let mut morning: Vec<Row> = Vec::new();
    let mut afternoon: Vec<Row> = Vec::new();
    for r in trades.iter() {
        if r.flag == 32 {
            continue; // 旧代码 with_retreat=0 丢掉撤单行
        }
        let raw_us = r.time_us - OFFSET_US;
        let tod = (r.time_us).rem_euclid(DAY_US);
        let ns = pandas_ns_of(raw_us) + OFFSET_US * 1000;
        if tod >= LD_0930 && tod <= LD_1130 {
            morning.push(Row {
                ns,
                tod,
                vol: r.volume as f64,
                flag: r.flag,
            });
        } else if tod >= LD_1300 && tod <= LD_1457 {
            afternoon.push(Row {
                ns: ns - SHIFT_US * 1000,
                tod: tod - SHIFT_US,
                vol: r.volume as f64,
                flag: r.flag,
            });
        }
    }

    // 旧代码 df = concat([上午段, 下午段])：全天 = 上午的行在前、下午的行在后
    let mut allday: Vec<Row> = Vec::with_capacity(morning.len() + afternoon.len());
    allday.extend_from_slice(&morning);
    allday.extend_from_slice(&afternoon);

    // 上午段 = 平移后钟点在 [09:30:00, 11:30:00] 的行（含恰好 11:30:00 的下午行）
    let mor_seg: Vec<Row> = allday
        .iter()
        .copied()
        .filter(|r| r.tod >= LD_0930 && r.tod <= LD_1130)
        .collect();
    // 下午段 = 平移后钟点在 [11:30:00, 13:27:00] 的行（含恰好 11:30:00 的上午行）
    let aft_seg: Vec<Row> = allday
        .iter()
        .copied()
        .filter(|r| r.tod >= LD_1130 && r.tod <= LD_1327)
        .collect();

    Some([allday, mor_seg, aft_seg])
}

// ============================================================================
// trade_peak_analysis 的纯 Rust 版（语义与 src/trade_peak_analysis.rs 逐字一致，
// 只是把「全表扫描窗口」换成按时间二分出的窗口，结果相同，快很多）
// ============================================================================

fn window_lo(sorted: bool, t: &[i64], center: i64, win: i64) -> usize {
    if sorted {
        t.partition_point(|&x| x < center - win)
    } else {
        0
    }
}

fn window_hi(sorted: bool, t: &[i64], center: i64, win: i64) -> usize {
    if sorted {
        t.partition_point(|&x| x <= center + win)
    } else {
        t.len()
    }
}

fn is_sorted(t: &[i64]) -> bool {
    t.windows(2).all(|w| w[0] <= w[1])
}

/// 每个高峰一行 17 个特征；返回 (行数, 行优先的扁平矩阵)。
fn peak_features(
    t: &[i64],
    vol: &[f64],
    flag: &[i32],
    top_tier1: f64,
    top_tier2: f64,
    win_ns: i64,
    flag_different: bool,
    with_forth: bool,
) -> Vec<Vec<f64>> {
    let n = t.len();
    if n == 0 {
        return Vec::new();
    }
    let sorted = is_sorted(t);

    let mut v_sorted: Vec<f64> = vol.to_vec();
    v_sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let tier1 = v_sorted[(((n as f64) * top_tier1) as usize).min(n - 1)];
    let tier2 = v_sorted[(((n as f64) * top_tier2) as usize).min(n - 1)];

    let mut peaks: Vec<usize> = Vec::new();
    for i in 0..n {
        if vol[i] < tier1 {
            continue;
        }
        let lo = window_lo(sorted, t, t[i], win_ns);
        let hi = window_hi(sorted, t, t[i], win_ns);
        let mut local = true;
        for j in lo..hi {
            if j == i {
                continue;
            }
            if vol[j] > vol[i] {
                local = false;
                break;
            }
        }
        if local {
            peaks.push(i);
        }
    }

    let mut out: Vec<Vec<f64>> = Vec::with_capacity(peaks.len());
    for &pi in peaks.iter() {
        let peak_time = t[pi];
        let peak_volume = vol[pi];
        let peak_flag = flag[pi];
        let lo = window_lo(sorted, t, peak_time, win_ns);
        let hi = window_hi(sorted, t, peak_time, win_ns);

        let mut minor: Vec<f64> = Vec::new();
        let mut tdiffs: Vec<f64> = Vec::new();
        for j in lo..hi {
            if j == pi {
                continue;
            }
            let dt = (t[j] - peak_time).abs();
            if dt > win_ns {
                continue;
            }
            let should = if with_forth { true } else { t[j] > peak_time };
            if !should || vol[j] < tier2 {
                continue;
            }
            if flag_different && flag[j] == peak_flag {
                continue;
            }
            minor.push(vol[j]);
            tdiffs.push(dt as f64 / 1e9);
        }
        out.push(calculate_features(peak_volume, &minor, &tdiffs));
    }
    out
}

// ---- 以下 17 个特征的计算照抄 src/trade_peak_analysis.rs（旧代码调的就是它）----

fn calculate_features(peak_volume: f64, minor_peaks: &[f64], time_diffs: &[f64]) -> Vec<f64> {
    let mut features = vec![0.0; 17];
    if minor_peaks.is_empty() {
        return features;
    }
    let n = minor_peaks.len();
    let minor_sum: f64 = minor_peaks.iter().sum();
    let minor_mean = minor_sum / n as f64;

    features[0] = minor_sum / peak_volume;
    features[1] = minor_mean / peak_volume;
    features[2] = n as f64;
    let time_mean: f64 = time_diffs.iter().sum::<f64>() / n as f64;
    features[3] = time_mean;
    features[4] = correlation(minor_peaks, time_diffs);
    features[5] = dtw_distance_simple(minor_peaks, time_diffs);
    let minor_std = calculate_std(minor_peaks, minor_mean);
    features[6] = if minor_mean.abs() > f64::EPSILON {
        minor_std / minor_mean
    } else {
        0.0
    };
    features[7] = calculate_skewness(minor_peaks, minor_mean, minor_std);
    features[8] = calculate_kurtosis(minor_peaks, minor_mean, minor_std);
    features[9] = calculate_trend(minor_peaks);
    features[10] = calculate_autocorr(minor_peaks);
    let time_std = calculate_std(time_diffs, time_mean);
    features[11] = if time_mean.abs() > f64::EPSILON {
        time_std / time_mean
    } else {
        0.0
    };
    features[12] = calculate_skewness(time_diffs, time_mean, time_std);
    features[13] = calculate_kurtosis(time_diffs, time_mean, time_std);
    features[14] = calculate_trend(time_diffs);
    features[15] = calculate_autocorr(time_diffs);
    features[16] = calculate_volume_weighted_time_distance(minor_peaks, time_diffs);
    features
}

fn correlation(x: &[f64], y: &[f64]) -> f64 {
    if x.len() != y.len() || x.len() < 2 {
        return 0.0;
    }
    let n = x.len() as f64;
    let x_mean = x.iter().sum::<f64>() / n;
    let y_mean = y.iter().sum::<f64>() / n;
    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;
    for i in 0..x.len() {
        let dx = x[i] - x_mean;
        let dy = y[i] - y_mean;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }
    if var_x.abs() < f64::EPSILON || var_y.abs() < f64::EPSILON {
        return 0.0;
    }
    cov / (var_x.sqrt() * var_y.sqrt())
}

fn dtw_distance_simple(s1: &[f64], s2: &[f64]) -> f64 {
    if s1.is_empty() || s2.is_empty() {
        return 0.0;
    }
    let len1 = s1.len();
    let len2 = s2.len();
    if len1 > 100 || len2 > 100 {
        let min_len = len1.min(len2);
        let mut sum = 0.0;
        for i in 0..min_len {
            sum += (s1[i] - s2[i]).powi(2);
        }
        return sum.sqrt();
    }
    let mut dp = vec![vec![f64::INFINITY; len2]; len1];
    dp[0][0] = (s1[0] - s2[0]).abs();
    for i in 1..len1 {
        dp[i][0] = dp[i - 1][0] + (s1[i] - s2[0]).abs();
    }
    for j in 1..len2 {
        dp[0][j] = dp[0][j - 1] + (s1[0] - s2[j]).abs();
    }
    for i in 1..len1 {
        for j in 1..len2 {
            let cost = (s1[i] - s2[j]).abs();
            dp[i][j] = cost + dp[i - 1][j].min(dp[i][j - 1]).min(dp[i - 1][j - 1]);
        }
    }
    dp[len1 - 1][len2 - 1]
}

fn calculate_std(data: &[f64], mean: f64) -> f64 {
    if data.len() < 2 {
        return 0.0;
    }
    let variance: f64 =
        data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (data.len() - 1) as f64;
    variance.sqrt()
}

fn calculate_skewness(data: &[f64], mean: f64, std: f64) -> f64 {
    if data.len() < 3 || std.abs() < f64::EPSILON {
        return 0.0;
    }
    let n = data.len() as f64;
    let skew: f64 = data
        .iter()
        .map(|&x| ((x - mean) / std).powi(3))
        .sum::<f64>()
        / n;
    if data.len() > 2 {
        let adj_factor = (n * (n - 1.0)).sqrt() / (n - 2.0);
        skew * adj_factor
    } else {
        skew
    }
}

fn calculate_kurtosis(data: &[f64], mean: f64, std: f64) -> f64 {
    if data.len() < 4 || std.abs() < f64::EPSILON {
        return 0.0;
    }
    let n = data.len() as f64;
    let kurt: f64 = data
        .iter()
        .map(|&x| ((x - mean) / std).powi(4))
        .sum::<f64>()
        / n;
    if data.len() > 3 {
        ((n - 1.0) / ((n - 2.0) * (n - 3.0))) * ((n + 1.0) * kurt - 3.0 * (n - 1.0))
    } else {
        kurt - 3.0
    }
}

fn calculate_trend(data: &[f64]) -> f64 {
    if data.len() < 2 {
        return 0.0;
    }
    let indices: Vec<f64> = (0..data.len()).map(|i| i as f64).collect();
    correlation(data, &indices)
}

fn calculate_autocorr(data: &[f64]) -> f64 {
    if data.len() < 2 {
        return 0.0;
    }
    let x1 = &data[0..data.len() - 1];
    let x2 = &data[1..data.len()];
    correlation(x1, x2)
}

fn calculate_volume_weighted_time_distance(minor_peaks: &[f64], time_diffs: &[f64]) -> f64 {
    if minor_peaks.is_empty() || time_diffs.is_empty() || minor_peaks.len() != time_diffs.len() {
        return 0.0;
    }
    let total_volume: f64 = minor_peaks.iter().sum();
    if total_volume.abs() < f64::EPSILON {
        return 0.0;
    }
    let weighted_sum: f64 = minor_peaks
        .iter()
        .zip(time_diffs.iter())
        .map(|(&volume, &time_diff)| volume * time_diff)
        .sum();
    weighted_sum / total_volume
}

// ============================================================================
// get_res：N×17 矩阵 → 110 个统计量（pandas 语义复刻）
// ============================================================================

fn seq_sum(v: &[f64]) -> f64 {
    let mut s = 0.0;
    for &x in v {
        s += x;
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

/// pandas 的 notna：只有 NaN 算缺失，inf 算有效。
fn valid_of(c: &[f64]) -> Vec<f64> {
    c.iter().copied().filter(|x| !x.is_nan()).collect()
}

fn col_mean(c: &[f64]) -> f64 {
    let v = valid_of(c);
    if v.is_empty() {
        return f64::NAN;
    }
    seq_sum(&v) / v.len() as f64
}

fn col_std(c: &[f64]) -> f64 {
    let v = valid_of(c);
    let n = v.len();
    if n < 2 {
        return f64::NAN;
    }
    let avg = seq_sum(&v) / n as f64;
    let mut sq = 0.0;
    for &x in v.iter() {
        let d = avg - x;
        sq += d * d;
    }
    (sq / (n as f64 - 1.0)).sqrt()
}

fn col_skew(c: &[f64]) -> f64 {
    let v = valid_of(c);
    let n = v.len();
    if n < 3 {
        return f64::NAN;
    }
    let nf = n as f64;
    let mu = seq_sum(&v) / nf;
    let mut m2 = 0.0;
    let mut m3 = 0.0;
    for &x in v.iter() {
        let d = x - mu;
        m2 += d * d;
        m3 += (d * d) * d;
    }
    let m2 = zero_out_fperr(m2);
    let m3 = zero_out_fperr(m3);
    if m2 == 0.0 {
        return 0.0;
    }
    (nf * (nf - 1.0).sqrt() / (nf - 2.0)) * (m3 / m2.powf(1.5))
}

fn col_kurt(c: &[f64]) -> f64 {
    let v = valid_of(c);
    let n = v.len();
    if n < 4 {
        return f64::NAN;
    }
    let nf = n as f64;
    let mu = seq_sum(&v) / nf;
    let mut m2 = 0.0;
    let mut m4 = 0.0;
    for &x in v.iter() {
        let d = x - mu;
        let d2 = d * d;
        m2 += d2;
        m4 += d2 * d2;
    }
    let m2 = zero_out_fperr(m2);
    let m4 = zero_out_fperr(m4);
    let adj = 3.0 * (nf - 1.0) * (nf - 1.0) / ((nf - 2.0) * (nf - 3.0));
    let numerator = zero_out_fperr(((nf * (nf + 1.0)) * (nf - 1.0)) * m4);
    let denominator = zero_out_fperr(((nf - 2.0) * (nf - 3.0)) * (m2 * m2));
    if denominator == 0.0 {
        return 0.0;
    }
    numerator / denominator - adj
}

/// pandas 的 Pearson 相关（np.corrcoef 语义）：成对非 NaN，样本<2 或方差为 0 → NaN。
fn pearson(a: &[f64], b: &[f64]) -> f64 {
    let mut n = 0usize;
    let mut sx = 0.0;
    let mut sy = 0.0;
    for i in 0..a.len().min(b.len()) {
        if a[i].is_nan() || b[i].is_nan() {
            continue;
        }
        n += 1;
        sx += a[i];
        sy += b[i];
    }
    if n < 2 {
        return f64::NAN;
    }
    let nf = n as f64;
    let mx = sx / nf;
    let my = sy / nf;
    let mut sxx = 0.0;
    let mut syy = 0.0;
    let mut sxy = 0.0;
    for i in 0..a.len().min(b.len()) {
        if a[i].is_nan() || b[i].is_nan() {
            continue;
        }
        let dx = a[i] - mx;
        let dy = b[i] - my;
        sxx += dx * dx;
        syy += dy * dy;
        sxy += dx * dy;
    }
    if sxx == 0.0 || syy == 0.0 {
        return f64::NAN;
    }
    sxy / (sxx.sqrt() * syy.sqrt())
}

/// src/time_series/trend_mod.rs::calculate_trend_1d 的同式实现（纯 Rust，worker 可调）。
fn trend_1d(data: &[f64]) -> f64 {
    let mut values: Vec<f64> = Vec::with_capacity(data.len());
    let mut indices: Vec<f64> = Vec::with_capacity(data.len());
    for (i, &val) in data.iter().enumerate() {
        if val.is_finite() {
            values.push(val);
            indices.push((i + 1) as f64);
        }
    }
    let n = values.len();
    if n == 0 || n == 1 {
        return 0.0;
    }
    let nf = n as f64;
    let mean_x = values.iter().sum::<f64>() / nf;
    let mean_y = indices.iter().sum::<f64>() / nf;
    let mut covariance = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;
    for i in 0..n {
        let dx = values[i] - mean_x;
        let dy = indices[i] - mean_y;
        covariance += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }
    if var_x == 0.0 || var_y == 0.0 {
        return 0.0;
    }
    covariance / (var_x.sqrt() * var_y.sqrt())
}

/// N×17 矩阵 → 110 个数：8 个相关对 + 6 组 × 17 列统计（顺序 = 旧代码 get_res）。
fn get_res(m: &[Vec<f64>]) -> Vec<f64> {
    let mut out = vec![f64::NAN; 110];
    let nrow = m.len();
    if nrow == 0 {
        return out;
    }
    let cols: Vec<Vec<f64>> = (0..17).map(|j| m.iter().map(|r| r[j]).collect()).collect();

    for (k, &(a, b)) in PAIRS.iter().enumerate() {
        out[k] = pearson(&cols[a], &cols[b]);
    }
    for j in 0..17 {
        let c = &cols[j];
        out[8 + j] = col_mean(c);
        out[25 + j] = col_std(c);
        out[42 + j] = col_skew(c);
        out[59 + j] = col_kurt(c);
        // corrwith(df.shift(1))：第 i 行配第 i-1 行
        out[76 + j] = pearson(&c[1..], &c[..nrow - 1]);
        out[93 + j] = trend_1d(c);
    }
    out
}

// ============================================================================
// 一个时段 → 880 个值
// ============================================================================

fn push_segment(seg: &[Row], out: &mut Vec<f32>) {
    let n = seg.len();
    let n66 = seg.iter().filter(|r| r.flag == 66).count();
    let n83 = seg.iter().filter(|r| r.flag == 83).count();
    if n <= 30 || n66 <= 30 || n83 <= 30 {
        // 旧代码：整段返回 880 个 NaN
        out.extend(std::iter::repeat(f32::NAN).take(PER_SEG));
        return;
    }
    let t: Vec<i64> = seg.iter().map(|r| r.ns).collect();
    let vol: Vec<f64> = seg.iter().map(|r| r.vol).collect();
    let flag: Vec<i32> = seg.iter().map(|r| r.flag).collect();

    let mut sub_t: Vec<i64> = Vec::new();
    let mut sub_v: Vec<f64> = Vec::new();
    let mut sub_f: Vec<i32> = Vec::new();

    for &(only66, only83, flag_diff, with_forth) in VARIANT_SPEC.iter() {
        let (tt, vv, ff): (&[i64], &[f64], &[i32]) = if !only66 && !only83 {
            (t.as_slice(), vol.as_slice(), flag.as_slice())
        } else {
            sub_t.clear();
            sub_v.clear();
            sub_f.clear();
            let want = if only66 { 66 } else { 83 };
            for i in 0..n {
                if flag[i] == want {
                    sub_t.push(t[i]);
                    sub_v.push(vol[i]);
                    sub_f.push(flag[i]);
                }
            }
            (sub_t.as_slice(), sub_v.as_slice(), sub_f.as_slice())
        };
        let m = peak_features(
            tt, vv, ff, TOP_TIER1, TOP_TIER2, SECONDS_NS, flag_diff, with_forth,
        );
        let res = get_res(&m);
        for v in res.into_iter() {
            out.push(v as f32);
        }
    }
}

/// 8 套变体：(只看 flag==66, 只看 flag==83, flag_different, with_forth)
const VARIANT_SPEC: [(bool, bool, bool, bool); 8] = [
    (false, false, false, false),
    (true, false, false, false),
    (false, true, false, false),
    (false, false, true, false),
    (false, false, false, true),
    (true, false, false, true),
    (false, true, false, true),
    (false, false, true, true),
];

/// 单股单日：2640 个因子。
pub fn compute_stock(code: &str, date: i64) -> Option<Vec<f32>> {
    let segs = build_segments(code, date)?;
    let mut out: Vec<f32> = Vec::with_capacity(N_OUT);
    for seg in segs.iter() {
        push_segment(seg, &mut out);
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_count_and_unique() {
        let n = names();
        assert_eq!(n.len(), N_OUT);
        let mut s = n.clone();
        s.sort();
        s.dedup();
        assert_eq!(s.len(), N_OUT);
    }

    #[test]
    fn names_std_swap() {
        let n = names();
        assert_eq!(n[40], "时间重心_std_全部_全天");
        assert_eq!(n[41], "时间自相关_std_全部_全天");
        assert_eq!(n[24], "时间重心_mean_全部_全天");
        assert_eq!(n[23], "时间自相关_mean_全部_全天");
    }

    #[test]
    fn pandas_ns_matches_known_cases() {
        // 1740965404510000 μs → pandas 给 1740965404509999989（比微秒×1000 少 11ns）
        assert_eq!(pandas_ns_of(1740965404510000), 1740965404509999989);
        // 整秒不带误差
        assert_eq!(pandas_ns_of(1740965400000000), 1740965400000000000);
    }

    #[test]
    fn rint_half_to_even() {
        assert_eq!(rint(0.5), 0.0);
        assert_eq!(rint(1.5), 2.0);
        assert_eq!(rint(2.5), 2.0);
        assert_eq!(rint(3.5), 4.0);
    }
}
