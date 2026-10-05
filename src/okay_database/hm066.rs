//! hm066 你追我赶 —— 旧框架（design_whatever_out.TickDataBase）的 Rust 复现。
//!
//! 旧文件：/home/chenzongwei/pythoncode/成果hm66_你追我赶/hm66_okay.py
//!   run_base 里的 Go.get_factor，参数 get_many=0 / read_full=0 / change_time=1，
//!   也就是「一整天的逐笔」一张表，窗口由旧代码自己切。
//!
//! ════════════════════════════════════════════════════════════════════
//! 一、整段 get_factor 是包在 try/except 里的
//! ════════════════════════════════════════════════════════════════════
//! 旧代码 `except Exception as e: ...` 直接吞掉异常、返回 None，整只股票全天 644 个值
//! 全落 NaN。所以下面任何一步算不出来都返回 None（上游补整条 NaN，因子个数恒定）。
//! 实测会走到这一条的情形：adjust_afternoon 之后一张表都没有（文件不存在、或当天
//! 只有集合竞价没有连续竞价成交），此时 `tradesort.exchtime.iloc[0]` 抛 IndexError。
//!
//! ════════════════════════════════════════════════════════════════════
//! 二、时间轴
//! ════════════════════════════════════════════════════════════════════
//! 1) read_trade_data_single(with_retreat=0)：先丢 flag==32 的撤单行。
//! 2) dwo.adjust_afternoon(df, only_inday=1)：取 [09:30:00, 11:30:00] 与
//!    [13:00:00, 14:57:00] 两段，**下午段整体减 90 分钟**，再 concat（上午段在前）。
//!    两段之外的行全丢。所以所有时刻都落在平移后的 09:30:00 ~ 13:27:00 上。
//! 3) between_time 是精确到微秒的闭区间。本机 pandas 走 `_get_time_micros`，
//!    比较用的就是**微秒**（纳秒被截掉），所以这里一律用当日微秒数比较。
//!
//! ════════════════════════════════════════════════════════════════════
//! 三、`tradesort=trade0.sort_values('price')` 的并列顺序（本系列的关键）
//! ════════════════════════════════════════════════════════════════════
//! 旧代码取 `tradesort.exchtime.iloc[0]` 与 `.iloc[-1]`，也就是**排序后的第一行/最后一行**
//! 的 exchtime，再用它切出「高低价之间」的那张表。价格只有两位小数，并列极多
//! （000001@20250303 的最低价有 251 行、跨越 34 秒），并列里挑中谁完全由 numpy 的
//! argsort 决定，换稳定排序结果就不一样（实测四组对照全部不同）。
//!
//! 本机 numpy 2.3.5 对 float64 走 x86-simd-sort 的 AVX-512 argsort（与 int64 同一份
//! 模板 xss-common-argsort.h），hm039.rs 已经把 int64 那一条逐行复刻并对照通过。
//! 这里直接沿用那段复刻，做法是**先把价格映射成保序的 int64 名次码**再排序：
//! 名次码与原值同序、相等价同码，而 xss 的每一步只做 `</>=/==` 与 min/max，
//! 所以判定序列完全一致、输出置换完全相同。实测四组验收数据
//! `np.argsort(float64 价格,'quicksort')` 与 `np.argsort(int64 名次码,'quicksort')`
//! 的置换逐位相同（000001/600519/000002/300750 全 True），这条等价关系是可靠的。
//! 另外 pandas 的 nargsort 会把 NaN 摘出来放最后，这里照抄。
//!
//! ════════════════════════════════════════════════════════════════════
//! 四、每个 (数据集, 列对) 出 39 个值
//! ════════════════════════════════════════════════════════════════════
//! 数据集 2 个：t0（adjust_afternoon 后的全表）与 trade（在「最低价那行的时刻」到
//! 「最高价那行的时刻」之间的子表）。
//! 每个数据集都先做同一步加工：
//!   price  = |price - price.shift(1)| 之后 cumsum()/sum()   （第 0 行是 NaN）
//!   volume = volume.cumsum()/volume.sum()
//!   volume_66 = (volume*(flag==66)).cumsum()/sum()，volume_83 同理
//!   exchtime_float = exchtime.to_numpy(float)，再 (x-min)/(max-min)
//! 注意 cumsum() 是 skipna：NaN 位置上结果仍是 NaN，后面的累积和照常接上。
//! 列对 8 个（顺序即输出顺序）：
//!   price/volume, price/volume_66, price/volume_83, volume_66/volume_83,
//!   exchtime_float/volume, exchtime_float/volume_66, exchtime_float/volume_83,
//!   exchtime_float/price
//! 每个列对 39 个值，顺序见 BASE39。
//!   corr_all / corra / corrb 是 pandas 的成对完整观测 Pearson（这里用与 hm013 同一份
//!   Welford 实现），corra 只取 col1>col2 的行、corrb 只取 col1<col2 的行。
//!   a,b = rp.segment_and_correlate(col1, col2, min_length=10)：按 (col1>col2) 的真假
//!   分段，长度 >= 10 的段算段内 Pearson，分别收进 a（真）与 b（假）。段里有 NaN 时
//!   段相关系数是 NaN，但**照样入列**（只有方差为 0 时被丢掉）。
//!   a_dis=|a-corra|，b_dis=|b-corrb|；四个 DTW 距离照 rp.dtw_distance 的算法抄
//!   （空序列参与时返回 inf，与旧框架一致，见下方说明）。
//!   统计量 mean/std(ddof=1)/max/min/median/skew/kurt 全部按 pandas nanops 的算法
//!   （skew/kurt 是 bias=False 的修正 Fisher 系数，且都带 `_zero_out_fperr`）。
//!   corr1 是 Series.autocorr(1) = corr(x, x.shift(1))。
//!
//! ════════════════════════════════════════════════════════════════════
//! 五、names（644 个，本文件 names() 必须逐个对上）
//! ════════════════════════════════════════════════════════════════════
//! 前 20 个是 corr_price_volume_{low,high}{10,20,30,40,50}_{fullday,highlow}。
//! 这 20 个在旧代码里是 `子表.corr().iloc[0,1]`，而子表带着 exchtime / date /
//! localtime / code / delay 等一堆列，corr() 出来的矩阵第 0 行是 exchtime、
//! 第 1 列是 date —— 一个 datetime 列对 int 列，pandas 一律给 NaN。
//! 实测四组 20 个全是 NaN（空子表时 corr() 仍是同样形状的 NaN 矩阵，也不抛异常），
//! 所以这里直接输出 20 个 NaN。
//! 其后 312 + 312 个 = 8 列对 × 39 × {_全天, _高低价之间}，_全天 对应 t0、_高低价之间
//! 对应 trade。
//!
//! ════════════════════════════════════════════════════════════════════
//! 六、精度与读取
//! ════════════════════════════════════════════════════════════════════
//! · 撤单过滤按规范自己 filter flag != 32（不靠 read_trade_fast_inner 的 with_retreat：
//!   库里那条判断按字节比 "32"，2025 年的明细写的是 "32.0"，过滤不生效）。
//! · 逐笔的 price / volume 全部按 f64 自己解析 CSV：volume 在旧代码里是 int64，
//!   cumsum/sum 之后再相除；用 f32 会把归一化累计量在小数位上打偏，而 39 个值里
//!   有大量「归一化累计量之差的统计量」，f32 的 1e-7 相对误差会被抵消放大。
//! · exchtime_float 必须逐位复刻旧代码的纳秒值，不能图省事用微秒：
//!   旧代码是 `exchtime.to_numpy(float)`，先由 pandas 把「微秒/1e6 秒」转成 int64 纳秒
//!   （cast_from_unit_vectorized：base*1e9 + int64(round(frac,9)*1e9)），再转成 float64。
//!   1.7e18 的纳秒转 f64 时 ulp 是 256ns，这个取整本身就有几十纳秒的量化；
//!   segment_and_correlate 的分段又是按 `exchtime_float > price` 的真假切的，
//!   实测把它换成精确的微秒值会让 000002@20180601 的分段整个变样
//!   （a_mean 从 -0.00407 变成 0.99593），对账直接不过。
//!   所以这里按 pandas 的公式把纳秒一位不差地算出来，再转 f64。

use std::fs::File;
use std::io::Read;

/// 逐笔 CSV 的列索引（0 基，与 fast_csv_reader 一致；2018 年及以前末尾多一列 num，不影响前 15 列）
const TRD_COL_EXCHTIME: usize = 4;
const TRD_COL_PRICE: usize = 7;
const TRD_COL_VOLUME: usize = 8;
const TRD_COL_FLAG: usize = 10;
const TRD_NCOL: usize = 15;

const DAY_NS: i64 = 86_400_000_000_000;
const DAY_US: i64 = DAY_NS / 1000;
const SEC_US: i64 = 1_000_000;
const MORNING_LO: i64 = (9 * 3600 + 30 * 60) * SEC_US;
const MORNING_HI: i64 = (11 * 3600 + 30 * 60) * SEC_US;
const AFTER_LO: i64 = 13 * 3600 * SEC_US;
const AFTER_HI: i64 = (14 * 3600 + 57 * 60) * SEC_US;
/// adjust_afternoon 把下午段整体减 90 分钟（在纳秒上做，与 pandas 一致）
const AFTER_SHIFT_NS: i64 = 90 * 60 * 1_000_000_000;

/// 每个列对 39 个值，8 个列对 × 2 个数据集 + 20 个前缀 NaN
const PER_PAIR: usize = 39;
const N_PAIRS: usize = 8;
pub const N_OUT: usize = 20 + PER_PAIR * N_PAIRS * 2; // 644

const PAIR_NAMES: [&str; N_PAIRS] = [
    "price_volume",
    "price_volume_66",
    "price_volume_83",
    "volume_66_volume_83",
    "exchtime_float_volume",
    "exchtime_float_volume_66",
    "exchtime_float_volume_83",
    "exchtime_float_price",
];

/// 每个列对的 39 个指标名，顺序与旧代码 one() 的返回顺序逐个对齐
const BASE39: [&str; PER_PAIR] = [
    "dtw_ab",
    "dtw_ab_dis",
    "dtw_a_dis",
    "dtw_b_dis",
    "corr_all",
    "corra",
    "corrb",
    "a_mean",
    "b_mean",
    "a_dis_mean",
    "b_dis_mean",
    "a_std",
    "b_std",
    "a_dis_std",
    "b_dis_std",
    "a_max",
    "b_max",
    "a_dis_max",
    "b_dis_max",
    "a_min",
    "b_min",
    "a_dis_min",
    "b_dis_min",
    "a_median",
    "b_median",
    "a_dis_median",
    "b_dis_median",
    "a_skew",
    "b_skew",
    "a_dis_skew",
    "b_dis_skew",
    "a_kurt",
    "b_kurt",
    "a_dis_kurt",
    "b_dis_kurt",
    "a_corr1",
    "b_corr1",
    "a_dis_corr1",
    "b_dis_corr1",
];

/// 前 20 个名字（值恒为 NaN，见文件头第五节）
const RES0_NAMES: [&str; 20] = [
    "corr_price_volume_low10_fullday",
    "corr_price_volume_high10_fullday",
    "corr_price_volume_low20_fullday",
    "corr_price_volume_high20_fullday",
    "corr_price_volume_low30_fullday",
    "corr_price_volume_high30_fullday",
    "corr_price_volume_low40_fullday",
    "corr_price_volume_high40_fullday",
    "corr_price_volume_low50_fullday",
    "corr_price_volume_high50_fullday",
    "corr_price_volume_low10_highlow",
    "corr_price_volume_high10_highlow",
    "corr_price_volume_low20_highlow",
    "corr_price_volume_high20_highlow",
    "corr_price_volume_low30_highlow",
    "corr_price_volume_high30_highlow",
    "corr_price_volume_low40_highlow",
    "corr_price_volume_high40_highlow",
    "corr_price_volume_low50_highlow",
    "corr_price_volume_high50_highlow",
];

pub fn names() -> Vec<String> {
    let mut out: Vec<String> = Vec::with_capacity(N_OUT);
    for n in RES0_NAMES.iter() {
        out.push((*n).to_string());
    }
    for suf in ["_全天", "_高低价之间"].iter() {
        for p in PAIR_NAMES.iter() {
            for m in BASE39.iter() {
                out.push(format!("{}_on_{}{}", m, p, suf));
            }
        }
    }
    out
}

// ============================================================================
// 读取与 adjust_afternoon
// ============================================================================

#[derive(Clone, Copy, Default)]
struct Raw {
    /// pandas 把 exchtime 转成的纳秒时间戳（未含 8 小时偏移的原始 ns，见 pandas_time_ns）
    time_ns: i64,
    price: f64,
    volume: f64,
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

/// pandas `pd.to_timedelta(微秒/1e6, unit="s")` 得到的时间戳（纳秒，未含 8 小时偏移）。
///
/// 逐行复刻 pandas/_libs/tslibs/conversion.pyx 的 cast_from_unit_vectorized
/// （unit='s'、out_unit='ns' 时 m = 1e9、p = 9）：
///     base = <int64_t>val                // 秒的整数部分，向零截断
///     frac = val - base
///     frac = np.round(frac, 9)           // = rint(frac * 1e9) / 1e9
///     out  = base * m + <int64_t>(frac * m)
/// 那一步 np.round(frac, 9) 是关键：不去掉它，frac*1e9 会差到 1 纳秒，四组实测都对不上。
/// 实测五份逐笔文件（2025 与 2018 各若干）这个公式与 pandas 的输出逐行完全相等。
#[inline]
fn pandas_time_ns(us: f64) -> i64 {
    if us.is_nan() {
        return i64::MIN; // 对应 NaT，后面按「不在任何窗口内」丢掉这一行
    }
    let val = us / 1e6;
    let base = val.trunc();
    let frac = val - base;
    let frac = (frac * 1e9).round_ties_even() / 1e9;
    (base as i64) * 1_000_000_000 + (frac * 1e9) as i64
}

fn read_trade_f64(code: &str, date: i64) -> Option<Vec<Raw>> {
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

    let mut out: Vec<Raw> = Vec::with_capacity(content.len() / 110 + 1);
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
        out.push(Raw {
            time_ns: pandas_time_ns(parse_f64_at(f[TRD_COL_EXCHTIME])),
            price: parse_f64_at(f[TRD_COL_PRICE]),
            volume: parse_f64_at(f[TRD_COL_VOLUME]),
            flag: parse_i64_at(f[TRD_COL_FLAG]),
        });
    }
    Some(out)
}

/// adjust_afternoon 之后的整张表：行序 = 上午段（原序）+ 下午段（原序，时刻已 -90 分钟）
#[derive(Default)]
struct Frame {
    t: Vec<i64>,
    price: Vec<f64>,
    volume: Vec<f64>,
    flag: Vec<i64>,
}

fn adjust_afternoon(raw: &[Raw]) -> Frame {
    let mut f = Frame::default();
    f.t.reserve(raw.len());
    // 窗口判定走 pandas between_time：它比的是**微秒**（_get_time_micros 把纳秒截掉）
    let day_us_of = |ns: i64| -> i64 {
        if ns == i64::MIN {
            -1
        } else {
            (ns + 8 * 3600 * 1_000_000_000).rem_euclid(DAY_NS) / 1000
        }
    };
    // 上午段在前
    for r in raw.iter() {
        if r.flag == 32 {
            continue;
        }
        let d = day_us_of(r.time_ns);
        if d >= MORNING_LO && d <= MORNING_HI {
            f.t.push(r.time_ns);
            f.price.push(r.price);
            f.volume.push(r.volume);
            f.flag.push(r.flag);
        }
    }
    // 下午段在后，整体前移 90 分钟
    for r in raw.iter() {
        if r.flag == 32 {
            continue;
        }
        let d = day_us_of(r.time_ns);
        if d >= AFTER_LO && d <= AFTER_HI {
            f.t.push(r.time_ns - AFTER_SHIFT_NS);
            f.price.push(r.price);
            f.volume.push(r.volume);
            f.flag.push(r.flag);
        }
    }
    f
}

// ============================================================================
// 保序名次码 + pandas nargsort（升序，NaN 排最后）
// ============================================================================

/// 把一组非 NaN 的 f64 映射成保序的 int64 名次码：同值同码、不同值不同码。
/// xss 的每一步只做比较与 min/max，所以名次码上的排序置换与原值完全一致。
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

/// pandas `sort_values(col)`（单键、升序、kind='quicksort'）的 indexer。
/// pandas/core/sorting.py 的 nargsort：非 NaN 部分走 `values.argsort(kind='quicksort')`，
/// NaN 的下标整体拼在最后。
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
// pandas 的统计量
// ============================================================================

/// pandas `DataFrame.corr()` / `Series.corr()` 的 Pearson：成对完整观测，
/// 逐行复刻 pandas/_libs/algos.pyx 的 nancorr（Welford 在线算法，与 hm013 同一份）。
fn welford_pearson(x: &[f64], y: &[f64]) -> f64 {
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

/// numpy 的 pairwise_sum（PW_BLOCKSIZE = 128），逐行复刻。
///
/// pandas 的 `.sum()` 走这条路径，求和顺序会改变最后几位；而本系列里
/// 「归一化累计量之间谁大谁小」决定了 segment_and_correlate 的分段
/// （见文件头第四节），x 上 1e-14 的差别就能让某一行的归属翻掉，
/// 进而把段相关系数改动 1e-7 量级。所以这里必须与 numpy 逐位一致，
/// 不能图省事用顺序累加。实测 000001@20250303 的 corrb 因此从
/// 0.9822125486349061 变回旧框架的 0.982212377751107。
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
        let mut r = [
            a[0] + a[1],
            a[2] + a[3],
            a[4] + a[5],
            a[6] + a[7],
            a[8] + a[9],
            a[10] + a[11],
            a[12] + a[13],
            a[14] + a[15],
        ];
        let mut i = 16usize;
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

/// pandas `_zero_out_fperr`：把 |x| < 1e-14 的浮点误差抹成 0
#[inline]
fn zero_out_fperr(x: f64) -> f64 {
    if x.abs() < 1e-14 {
        0.0
    } else {
        x
    }
}

fn nan_mean(v: &[f64]) -> f64 {
    let n = v.iter().filter(|x| !x.is_nan()).count();
    if n == 0 {
        f64::NAN
    } else {
        nanops_sum(v, 0.0) / n as f64
    }
}

/// pandas nanstd（ddof=1）：两遍算法，count - ddof <= 0 时 NaN
fn nan_std(v: &[f64]) -> f64 {
    let n = v.iter().filter(|x| !x.is_nan()).count();
    let d = n as f64 - 1.0;
    if n == 0 || d <= 0.0 {
        return f64::NAN;
    }
    let avg = nanops_sum(v, 0.0) / n as f64;
    let sq: Vec<f64> = v
        .iter()
        .map(|&x| if x.is_nan() { 0.0 } else { (avg - x) * (avg - x) })
        .collect();
    (pairwise_sum(&sq) / d).sqrt()
}

fn nan_max(v: &[f64]) -> f64 {
    let mut r = f64::NAN;
    for &x in v.iter() {
        if x.is_nan() {
            continue;
        }
        if r.is_nan() || x > r {
            r = x;
        }
    }
    r
}

fn nan_min(v: &[f64]) -> f64 {
    let mut r = f64::NAN;
    for &x in v.iter() {
        if x.is_nan() {
            continue;
        }
        if r.is_nan() || x < r {
            r = x;
        }
    }
    r
}

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

/// pandas `DataFrame.skew()`（bias=False，修正 Fisher-Pearson G1）
fn nan_skew(v: &[f64]) -> f64 {
    let n = v.iter().filter(|x| !x.is_nan()).count();
    if n == 0 {
        return f64::NAN;
    }
    let cnt = n as f64;
    let mean = nanops_sum(v, 0.0) / cnt;
    // pandas：values 里 NaN 先换成 0，减完均值后再把 NaN 位置置 0，再求和
    let a2: Vec<f64> = v
        .iter()
        .map(|&x| {
            if x.is_nan() {
                0.0
            } else {
                let d = x - mean;
                d * d
            }
        })
        .collect();
    let a3: Vec<f64> = v
        .iter()
        .map(|&x| {
            if x.is_nan() {
                0.0
            } else {
                let d = x - mean;
                d * d * d
            }
        })
        .collect();
    let m2 = zero_out_fperr(pairwise_sum(&a2));
    let m3 = zero_out_fperr(pairwise_sum(&a3));
    if n < 3 {
        return f64::NAN;
    }
    if m2 == 0.0 {
        return 0.0;
    }
    (cnt * (cnt - 1.0).sqrt() / (cnt - 2.0)) * (m3 / m2.powf(1.5))
}

/// pandas `DataFrame.kurt()`（bias=False，修正 Fisher 超额峰度）
fn nan_kurt(v: &[f64]) -> f64 {
    let n = v.iter().filter(|x| !x.is_nan()).count();
    if n == 0 {
        return f64::NAN;
    }
    let cnt = n as f64;
    let mean = nanops_sum(v, 0.0) / cnt;
    let a2: Vec<f64> = v
        .iter()
        .map(|&x| {
            if x.is_nan() {
                0.0
            } else {
                let d = x - mean;
                d * d
            }
        })
        .collect();
    let a4: Vec<f64> = v
        .iter()
        .map(|&x| {
            if x.is_nan() {
                0.0
            } else {
                let d = x - mean;
                let d2 = d * d;
                d2 * d2
            }
        })
        .collect();
    let m2 = pairwise_sum(&a2);
    let m4 = pairwise_sum(&a4);
    if n < 4 {
        return f64::NAN;
    }
    let adj = 3.0 * (cnt - 1.0).powi(2) / ((cnt - 2.0) * (cnt - 3.0));
    let numerator = zero_out_fperr(cnt * (cnt + 1.0) * (cnt - 1.0) * m4);
    let denominator = zero_out_fperr((cnt - 2.0) * (cnt - 3.0) * m2 * m2);
    if denominator == 0.0 {
        return 0.0;
    }
    numerator / denominator - adj
}

/// pandas `Series.autocorr(1)` = corr(x, x.shift(1))，成对完整观测
fn autocorr1(v: &[f64]) -> f64 {
    let n = v.len();
    let mut y: Vec<f64> = Vec::with_capacity(n);
    for i in 0..n {
        y.push(if i >= 1 { v[i - 1] } else { f64::NAN });
    }
    welford_pearson(v, &y)
}

// ============================================================================
// rp.dtw_distance（纯 Rust，不用 #[pyfunction] 那条：worker 不带 libpython）
// ============================================================================
//
// 算法照 src/time_series/mod.rs 的 dtw_distance（radius=None、无超时）抄：
// 成本矩阵从 [0][0]=0 起，其余初值 +inf，递推取左/上/左上三者最小。
// 注意 f64::min 会忽略 NaN，所以段相关系数里有 NaN 时结果也是这套语义。
// 空序列参与时返回矩阵角落里那个 +inf（或 [0][0]=0），与旧框架一致。
fn dtw_distance(s1: &[f64], s2: &[f64]) -> f64 {
    let n1 = s1.len();
    let n2 = s2.len();
    let mut m = vec![vec![f64::INFINITY; n2 + 1]; n1 + 1];
    m[0][0] = 0.0;
    for i in 1..=n1 {
        for j in 1..=n2 {
            let cost = (s1[i - 1] - s2[j - 1]).abs();
            m[i][j] = cost + m[i - 1][j].min(m[i][j - 1].min(m[i - 1][j - 1]));
        }
    }
    m[n1][n2]
}

// ============================================================================
// rp.segment_and_correlate（纯 Rust 复刻）
// ============================================================================

/// 段内 Pearson；分母为 0（含 NaN 情形下分母为 NaN 时不等于 0）时返回 None
fn seg_corr(x: &[f64], y: &[f64]) -> Option<f64> {
    let n = x.len();
    if n != y.len() || n < 2 {
        return None;
    }
    let nx = n as f64;
    let mean_x: f64 = x.iter().sum::<f64>() / nx;
    let mean_y: f64 = y.iter().sum::<f64>() / nx;
    let (mut cov, mut vx, mut vy) = (0.0f64, 0.0f64, 0.0f64);
    for i in 0..n {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        cov += dx * dy;
        vx += dx * dx;
        vy += dy * dy;
    }
    let den = (vx * vy).sqrt();
    if den == 0.0 {
        None
    } else {
        Some(cov / den)
    }
}

/// 按 (a>b) 的真假分段，长度 >= min_length(10) 的段算相关系数；
/// 真段进 a_greater，假段进 b_greater。相关系数算不出来（分母 0）的段两个都不进。
fn segment_and_correlate(a: &[f64], b: &[f64], min_length: usize) -> (Vec<f64>, Vec<f64>) {
    let n = a.len();
    let mut ag: Vec<f64> = Vec::new();
    let mut bg: Vec<f64> = Vec::new();
    if n < 2 {
        return (ag, bg);
    }
    let mut segs: Vec<(usize, usize, bool)> = Vec::new();
    let mut cur_start = 0usize;
    let mut cur_greater = a[0] > b[0];
    for i in 1..n {
        let g = a[i] > b[i];
        if g != cur_greater {
            if i - cur_start >= min_length {
                segs.push((cur_start, i, cur_greater));
            }
            cur_start = i;
            cur_greater = g;
        }
    }
    if n - cur_start >= min_length {
        segs.push((cur_start, n, cur_greater));
    }
    for (s, e, g) in segs {
        if let Some(c) = seg_corr(&a[s..e], &b[s..e]) {
            if g {
                ag.push(c);
            } else {
                bg.push(c);
            }
        }
    }
    (ag, bg)
}

// ============================================================================
// 数据集加工
// ============================================================================

#[derive(Clone, Copy)]
enum Col {
    Price,
    Volume,
    Volume66,
    Volume83,
    Exchtime,
}

#[derive(Default)]
struct Prep {
    price: Vec<f64>,
    volume: Vec<f64>,
    volume_66: Vec<f64>,
    volume_83: Vec<f64>,
    exchtime: Vec<f64>,
}

impl Prep {
    fn col(&self, c: Col) -> &Vec<f64> {
        match c {
            Col::Price => &self.price,
            Col::Volume => &self.volume,
            Col::Volume66 => &self.volume_66,
            Col::Volume83 => &self.volume_83,
            Col::Exchtime => &self.exchtime,
        }
    }
}

/// 累计和 / 总和（skipna：NaN 位置结果仍是 NaN，后面的累积和接上）
fn cumsum_over_sum(v: &[f64]) -> Vec<f64> {
    let mut out = vec![f64::NAN; v.len()];
    let mut acc = 0.0f64;
    for i in 0..v.len() {
        if v[i].is_nan() {
            out[i] = f64::NAN;
        } else {
            acc += v[i];
            out[i] = acc;
        }
    }
    // 除数用 numpy 的 pairwise 求和（pandas 的 .sum()），不是顺序累加的那个 acc
    let total = nanops_sum(v, 0.0);
    // 总和为 0 时 pandas 得到 0/0 = NaN 或 x/0 = ±inf，这里靠除法自然产生
    for x in out.iter_mut() {
        *x /= total;
    }
    out
}

fn prep(f: &Frame, idx: &[usize]) -> Prep {
    let n = idx.len();
    let mut p = Prep {
        price: Vec::with_capacity(n),
        volume: Vec::with_capacity(n),
        volume_66: Vec::with_capacity(n),
        volume_83: Vec::with_capacity(n),
        exchtime: Vec::with_capacity(n),
    };
    // price: |diff| 后 cumsum/sum
    let mut dprice = vec![f64::NAN; n];
    for k in 1..n {
        dprice[k] = (f.price[idx[k]] - f.price[idx[k - 1]]).abs();
    }
    p.price = cumsum_over_sum(&dprice);
    // volume / volume_66 / volume_83
    let v: Vec<f64> = idx.iter().map(|&i| f.volume[i]).collect();
    let v66: Vec<f64> = idx
        .iter()
        .map(|&i| if f.flag[i] == 66 { f.volume[i] } else { 0.0 })
        .collect();
    let v83: Vec<f64> = idx
        .iter()
        .map(|&i| if f.flag[i] == 83 { f.volume[i] } else { 0.0 })
        .collect();
    p.volume = cumsum_over_sum(&v);
    p.volume_66 = cumsum_over_sum(&v66);
    p.volume_83 = cumsum_over_sum(&v83);
    // exchtime_float: (x - min)/(max - min)
    let t: Vec<f64> = idx.iter().map(|&i| f.t[i] as f64).collect();
    let mut tmin = f64::INFINITY;
    let mut tmax = f64::NEG_INFINITY;
    for &x in t.iter() {
        if x < tmin {
            tmin = x;
        }
        if x > tmax {
            tmax = x;
        }
    }
    let span = tmax - tmin;
    p.exchtime = t.iter().map(|&x| (x - tmin) / span).collect();
    p
}

// ============================================================================
// 每个列对的 39 个值
// ============================================================================

fn one(x: &[f64], y: &[f64]) -> [f64; PER_PAIR] {
    let n = x.len();
    let corr_all = welford_pearson(x, y);
    // corra：col1 > col2 的行；corrb：col1 < col2 的行（NaN 比较为假，天然被排除）
    let mut gx: Vec<f64> = Vec::new();
    let mut gy: Vec<f64> = Vec::new();
    let mut lx: Vec<f64> = Vec::new();
    let mut ly: Vec<f64> = Vec::new();
    for i in 0..n {
        if x[i] > y[i] {
            gx.push(x[i]);
            gy.push(y[i]);
        } else if x[i] < y[i] {
            lx.push(x[i]);
            ly.push(y[i]);
        }
    }
    let corra = welford_pearson(&gx, &gy);
    let corrb = welford_pearson(&lx, &ly);

    let (a, b) = segment_and_correlate(x, y, 10);
    let a_dis: Vec<f64> = a.iter().map(|&v| (v - corra).abs()).collect();
    let b_dis: Vec<f64> = b.iter().map(|&v| (v - corrb).abs()).collect();

    let mut out = [f64::NAN; PER_PAIR];
    out[0] = dtw_distance(&a, &b);
    out[1] = dtw_distance(&a_dis, &b_dis);
    out[2] = dtw_distance(&a, &a_dis);
    out[3] = dtw_distance(&b, &b_dis);
    out[4] = corr_all;
    out[5] = corra;
    out[6] = corrb;

    let stats: [[f64; 4]; 8] = [
        [
            nan_mean(&a),
            nan_mean(&b),
            nan_mean(&a_dis),
            nan_mean(&b_dis),
        ],
        [nan_std(&a), nan_std(&b), nan_std(&a_dis), nan_std(&b_dis)],
        [nan_max(&a), nan_max(&b), nan_max(&a_dis), nan_max(&b_dis)],
        [nan_min(&a), nan_min(&b), nan_min(&a_dis), nan_min(&b_dis)],
        [
            nan_median(&a),
            nan_median(&b),
            nan_median(&a_dis),
            nan_median(&b_dis),
        ],
        [
            nan_skew(&a),
            nan_skew(&b),
            nan_skew(&a_dis),
            nan_skew(&b_dis),
        ],
        [
            nan_kurt(&a),
            nan_kurt(&b),
            nan_kurt(&a_dis),
            nan_kurt(&b_dis),
        ],
        [
            autocorr1(&a),
            autocorr1(&b),
            autocorr1(&a_dis),
            autocorr1(&b_dis),
        ],
    ];
    let mut k = 7usize;
    for st in stats.iter() {
        for v in st.iter() {
            out[k] = *v;
            k += 1;
        }
    }
    debug_assert_eq!(k, PER_PAIR);
    out
}

/// 第 i 个列对的两个列（与 PAIR_NAMES 同序）
const PAIR_COLS: [(Col, Col); N_PAIRS] = [
    (Col::Price, Col::Volume),
    (Col::Price, Col::Volume66),
    (Col::Price, Col::Volume83),
    (Col::Volume66, Col::Volume83),
    (Col::Exchtime, Col::Volume),
    (Col::Exchtime, Col::Volume66),
    (Col::Exchtime, Col::Volume83),
    (Col::Exchtime, Col::Price),
];

/// 逐股的一天。文件读不到、或 adjust_afternoon 之后一张表都没有 → None（上游补整条 NaN）。
pub fn compute_stock(code: &str, date: i64) -> Option<Vec<f32>> {
    let raw = read_trade_f64(code, date)?;
    let f = adjust_afternoon(&raw);
    if f.t.is_empty() {
        return None;
    }

    // 排序取「最低价那行」与「最高价那行」的时刻，再切出高低价之间的子表
    let order = pandas_argsort_asc(&f.price);
    let low = f.t[order[0]];
    let high = f.t[*order.last().unwrap()];
    let start = low.min(high);
    let end = low.max(high);

    let all_idx: Vec<usize> = (0..f.t.len()).collect();
    let hl_idx: Vec<usize> = (0..f.t.len())
        .filter(|&i| f.t[i] >= start && f.t[i] <= end)
        .collect();

    let p_all = prep(&f, &all_idx);
    let p_hl = prep(&f, &hl_idx);

    let mut out: Vec<f32> = Vec::with_capacity(N_OUT);
    for _ in 0..20 {
        out.push(f32::NAN);
    }
    for (c1, c2) in PAIR_COLS.iter() {
        let vals = one(p_all.col(*c1), p_all.col(*c2));
        out.extend(vals.iter().map(|&v| v as f32));
    }
    for (c1, c2) in PAIR_COLS.iter() {
        let vals = one(p_hl.col(*c1), p_hl.col(*c2));
        out.extend(vals.iter().map(|&v| v as f32));
    }
    debug_assert_eq!(out.len(), N_OUT);
    Some(out)
}

/// hm66 全是逐股因子，没有横截面版本。
pub fn compute_cross(_date: i64) -> Option<(Vec<String>, Vec<f32>)> {
    None
}

// ============================================================================
// 下面是 hm039.rs 里那份「numpy x86-simd-sort argsort(int64) 的逐行复刻」原文照搬，
// 一个字都没改。本文件用它排保序名次码（见文件头第三节）。
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
