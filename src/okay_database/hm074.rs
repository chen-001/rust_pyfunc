//! hm074 落叶像雪 —— 旧框架（design_whatever_out.TickDataBase）的 Rust 复现，**横截面形态**。
//!
//! 旧文件：/home/chenzongwei/pythoncode/成果hm74_落叶像雪/hm74_okay.py
//!   run_base 里的 Go.get_factor，参数 get_many=0 / read_full=0 / change_time=1，
//!   cross_section=False（旧代码自己是逐股跑的），整段包在 try/except 里。
//!
//! ════════════════════════════════════════════════════════════════════════
//! 零、为什么这个系列必须是 compute_cross，不是 compute_stock
//! ════════════════════════════════════════════════════════════════════════
//! 旧的 hm74_okay.py 虽然挂在 TickDataBase 下，但每个「股票日」做的是全市场的动作：
//!   1) 读当天**全市场**的分钟表（4 个字段，每个 240 行 × 5792 列）；
//!   2) 用 corrwith 挑出与该股分钟相关性最高的 10 只票；
//!   3) 把这 11 只票的盘口快照各读一遍、各自 resample 成 3 秒；
//!   4) 两条相关系数序列（10 个 1 分钟相关 + 10 个 3 秒相关）做 89 个指标。
//! 照原样逐股做，全历史要重读 11 倍的盘口快照，约 350TB 读盘、78 小时机时。
//! 改成一天算一次、当天中间结果复用后，盘口只需每天读一遍（约 15GB/天 → 39TB 全历史）。
//! 所以本系列走 kind='cross_section' 的 compute_cross，返回股票优先的扁平值。
//!
//! ════════════════════════════════════════════════════════════════════════
//! 一、旧代码逐步对照（写代码前先抄在这里，方便复核）
//! ════════════════════════════════════════════════════════════════════════
//! run_base 里三组，每组 89 个指标，共 267 个：
//!   组1 col='volume'   df = read_minute_data('volume')               后缀 ___volume
//!   组2 col='last_prc' df = read_minute_data('close')                后缀 ___price
//!   组3 col='abi1'     df = (ask_vol1-bid_vol1)/(ask_vol1+bid_vol1)  后缀 ___abi1
//! 注意：分钟数据的 hdf5_dir 参数（/nas197/binary/stock/sz_alpha/factor_base）只用于
//! 因子存储，真正读分钟的是 FactorDesign 内建的读者，目录是 /ssd_data/data/1min_factor_text；
//! 与 hm017.rs 用的是同一份数据（同一天逐值比对过）。
//!
//! one(df, col) 逐步：
//!   1) corrs_1m = df.corrwith(df[code]).dropna().sort_values(ascending=False)
//!                   .drop(code).head(10)
//!      —— 与目标股分钟相关性最高的 10 只票，是一个「代码 → 相关系数」的 Series。
//!   2) 目标股 3 秒序列：
//!        df3 = adjust_afternoon(read_market(code,date).set_index('exchtime'))
//!                .resample('3s').last().ffill()
//!      组3 再补一列 abi1=(ask_vol1-bid_vol1)/(ask_vol1+bid_vol1)。
//!   3) 10 只 peer 同样处理，得到 tops。
//!   4) corrs_3s：
//!        col='volume'   → tops_df = concat(peer['volume'].diff()) ，
//!                         corrs_3s = tops_df.corrwith(df3['volume'].diff())
//!        col='last_prc' → 同理但用 pct_change()
//!        其余（abi1）  → 直接用原列对接
//!   5) 89 个指标全部由这两条 10 元素序列（corrs_1m / corrs_3s）产生，外加
//!      top1/top5 的 ret 与 volume。
//!
//! ════════════════════════════════════════════════════════════════════════
//! 二、输出排布（横截面系列的约定，别搞反）
//! ════════════════════════════════════════════════════════════════════════
//! compute_cross 返回 (codes, vals)，vals 是**股票优先**：每只股票连续 267 个值，
//! 顺序是 组1(volume) 89 个 → 组2(price) 89 个 → 组3(abi1) 89 个，与旧代码
//! `res = res1 + res2 + res3` 一致。expected_len = names().len() = 267。
//! codes 是分钟数据 symbol_map.csv 的全名单（20250303 那天 5792 只），
//! 算不出来的股票（分钟整列全空、盘口文件缺失、11 只票里有任何一只没数据）直接不输出，
//! 与旧代码 try/except 吞掉异常返回 None 的效果一致（写盘时那些 cell 落 NaN）。
//!
//! ════════════════════════════════════════════════════════════════════════
//! 三、分钟相关系数（corrs_1m）：为什么可以做成稠密点积
//! ════════════════════════════════════════════════════════════════════════
//! pandas 的 `df.corrwith(df[code])` 对每列走 `Series.corr` → nanops.nancorr：
//! 先按「两边都不是 NaN」取成对完整观测，再 `np.corrcoef`（两遍法）——
//! 均值用 np.average（numpy 的成对求和），协方差是 Σ(中心化乘积)/(n-1)，
//! 最后 `(c/sa)/sb` 再 clip 到 [-1,1]。
//! 实测本数据每天的 NaN 结构极其干净：少数列整列为空，其余列**都**只在第 237/238/239
//! 行（一天最后三根，旧框架 get_field_min_data 置的）上为 NaN。于是大多数列共享同一个
//! 有效行集合，成对完整观测退化成「同一组行上的稠密点积」，一天 5792 列的相关可以一次性
//! 算完，不必逐股重算。
//! 实现里仍然保留一般路径：先取「出现次数最多的那个有效行掩码」当主导掩码，建稠密矩阵；
//! 掩码不同的列走逐对的一般算法。这样即使某天出现个别股票缺前几分钟也仍然正确。
//!
//! ════════════════════════════════════════════════════════════════════════
//! 四、3 秒序列：网格与 ffill 的精确语义
//! ════════════════════════════════════════════════════════════════════════
//! · adjust_afternoon 取 [09:30:00, 11:30:00] 与 [13:00:00, 14:57:00] 两段，下午段整体
//!   减 90 分钟再拼到上午段后面。between_time 是**精确到微秒**的闭区间。
//! · 时间戳先按 pandas 的方式从「exchtime 微秒」转成纳秒（含 ±250ns 的量化），
//!   再加 8 小时，再取当天微秒判窗口，与 hm066.rs 的 pandas_time_ns 同一套。
//! · resample('3s').last()：网格从 floor(第一笔时刻, 3s) 到 floor(最后一笔时刻, 3s)，
//!   步长 3 秒；每个桶取**桶内最后一行**（本数据无 NaN，skipna 与否无差别）。
//!   实测 688981@20250303 的 adj 区间是 09:30:03~13:26:59，网格就是 09:30:03~13:26:57，
//!   4739 个桶，两端都是 floor 而不是 ceil。
//! · .ffill()：桶内没有数据的补成前一个非空桶的值（每列各自 ffill）。
//! · 拼接 10 个 peer 时 pandas 用 axis=1 的并集索引对齐，桶的编号是**绝对**的
//!   （floor(纳秒/3e9)），所以这里直接用全局桶号，区间外的位置等价于 NaN，
//!   相关系数只落在两边区间的交集上。
//!
//! ════════════════════════════════════════════════════════════════════════
//! 五、pct_change / diff 的确切写法
//! ════════════════════════════════════════════════════════════════════════
//! · diff()    = x - x.shift(1)
//! · pct_change() 在 pandas 2.3.3 里默认 fill_method 是 lib.no_default，会被改写成 'pad'
//!   （先 ffill 再算），因此是 `x / x.shift(1) - 1`，**不是** (x-x.shift)/x.shift。
//!   本数据的 3 秒序列已经 ffill 过，pad 是空操作，但算式顺序必须照抄。
//!
//! ════════════════════════════════════════════════════════════════════════
//! 六、数值口径
//! ════════════════════════════════════════════════════════════════════════
//! · 逐笔/盘口的价格与成交量都按 **f64** 自己解析 CSV：volume 是当日累计量，量级到 1e8，
//!   f32 的 24 位尾数会把「3 秒增量」（差值约 1e3~1e4）打偏到 1e-3 量级，超过 1e-4 验收线。
//! · 求和一律走 numpy 的 pairwise_sum（PW_BLOCKSIZE=128），与 pandas 的 .sum()/.mean()
//!   同一条路径；10 元素序列落在 8 路展开的分支里。
//! · mean/std/skew/kurt 照 pandas nanops（std 是两遍法 ddof=1；skew/kurt 是 bias=False
//!   的修正 Fisher 系数，且都带 _zero_out_fperr）。
//! · Series.corr / autocorr 走 np.corrcoef（两遍法 + clip 到 [-1,1]），
//!   Series.cov 走 np.cov(a,b,ddof=1)[0,1]。
//! · rp.trend：丢掉非有限值后与「原始位置 +1」做 Pearson，样本 <2 或任一侧方差为 0 给 0.0。

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, LazyLock, Mutex};

use rayon::prelude::*;

/// 分钟数据每天 240 根 K 线
const MIN_PER_DAY: usize = 240;
/// 旧框架 get_field_min_data 把每根日线索引 % 240 >= 237 的三根置 NaN
const MIN_NAN_FROM: usize = 237;
/// corrs_1m / corrs_3s 取前多少只 peer
const TOP_NUMBER: usize = 10;
/// 每组指标个数
const N_METRIC: usize = 89;
/// 输出因子个数 = 3 组 × 89
pub const N_OUT: usize = N_METRIC * 3;
/// 3 秒网格的纳秒步长
const THREE_S_NS: i64 = 3_000_000_000;
/// 北京时区偏移
const BJ_NS: i64 = 8 * 3600 * 1_000_000_000;
const DAY_NS: i64 = 86_400_000_000_000;
/// between_time 的两段窗口（当天微秒，闭区间）
const AM_LO_US: i64 = 34_200 * 1_000_000; // 09:30:00
const AM_HI_US: i64 = 41_400 * 1_000_000; // 11:30:00
const PM_LO_US: i64 = 46_800 * 1_000_000; // 13:00:00
const PM_HI_US: i64 = 53_820 * 1_000_000; // 14:57:00
/// 下午段整体平移 -90 分钟
const AFTERNOON_SHIFT_NS: i64 = 90 * 60 * 1_000_000_000;

/// 89 个指标名，顺序与旧代码 res 列表逐个对齐（来自 hm74_okay.py 的 names）
const METRIC_NAMES: [&str; N_METRIC] = [
    "corr_1m3s",
    "cov_1m3s",
    "corr_abs_1m3s",
    "cov_abs_1m3s",
    "trend_diff_1m3s",
    "trend_diff_abs_1m3s",
    "cdr_total",
    "cdrs_mean",
    "cdrs_std",
    "cdrs_skew",
    "cdrs_kurt",
    "cdrs_max",
    "cdrs_min",
    "cdrs_autocorr",
    "cdrs_autocorr_abs",
    "cdrs_trend",
    "cdrs_trend_abs",
    "corr1m_mean",
    "corr1m_std",
    "corr1m_skew",
    "corr1m_kurt",
    "corr1m_max",
    "corr1m_min",
    "corr1m_autocorr",
    "corr1m_autocorr_abs",
    "corr1m_trend",
    "corr1m_trend_abs",
    "corr3s_mean",
    "corr3s_std",
    "corr3s_skew",
    "corr3s_kurt",
    "corr3s_max",
    "corr3s_min",
    "corr3s_autocorr",
    "corr3s_autocorr_abs",
    "corr3s_trend",
    "corr3s_trend_abs",
    "corr1m3s_std_diff",
    "corr1m3s_skew_diff",
    "corr1m3s_kurt_diff",
    "corr1m3s_max_diff",
    "corr1m3s_min_diff",
    "corr1m3s_autocorr_diff",
    "corr1m3s_autocorr_abs_diff",
    "corr1m3s_trend_diff",
    "corr1m3s_trend_abs_diff",
    "corr1m3s_std_diff_abs",
    "corr1m3s_skew_diff_abs",
    "corr1m3s_kurt_diff_abs",
    "corr1m3s_max_diff_abs",
    "corr1m3s_min_diff_abs",
    "corr1m3s_autocorr_diff_abs",
    "corr1m3s_autocorr_abs_diff_abs",
    "corr1m3s_trend_diff_abs",
    "corr1m3s_trend_abs_diff_abs",
    "top1_ret_1m",
    "top1_ret_3s",
    "top5_ret_1m",
    "top5_ret_3s",
    "ret_self",
    "top1_ret_diff_1m3s",
    "top5_ret_diff_1m3s",
    "top1_ret_diff_1m3s_abs",
    "top5_ret_diff_1m3s_abs",
    "top1_ret_diff_1m_self",
    "top5_ret_diff_1m_self",
    "top1_ret_diff_1m_self_abs",
    "top5_ret_diff_1m_self_abs",
    "top1_ret_diff_3s_self",
    "top5_ret_diff_3s_self",
    "top1_ret_diff_3s_self_abs",
    "top5_ret_diff_3s_self_abs",
    "top1_volume_1m",
    "top1_volume_3s",
    "top5_volume_1m",
    "top5_volume_3s",
    "volume_self",
    "top1_volume_ratio_1m3s",
    "top5_volume_ratio_1m3s",
    "top1_volume_ratio_1m_self",
    "top5_volume_ratio_1m_self",
    "top1_volume_ratio_3s_self",
    "top5_volume_ratio_3s_self",
    "top1_volume_ratio_1m3s_around1abs",
    "top5_volume_ratio_1m3s_around1abs",
    "top1_volume_ratio_1m_self_around1abs",
    "top5_volume_ratio_1m_self_around1abs",
    "top1_volume_ratio_3s_self_around1abs",
    "top5_volume_ratio_3s_self_around1abs",
];

/// 三组的后缀，顺序就是旧代码 res1 + res2 + res3 的顺序
const GROUP_SUFFIX: [&str; 3] = ["volume", "price", "abi1"];

pub fn names() -> Vec<String> {
    let mut out: Vec<String> = Vec::with_capacity(N_OUT);
    for suf in GROUP_SUFFIX.iter() {
        for m in METRIC_NAMES.iter() {
            out.push(format!("{}___{}", m, suf));
        }
    }
    debug_assert_eq!(out.len(), N_OUT);
    out
}

// ============================================================================
// pandas / numpy 的小工具
// ============================================================================

/// numpy 的 pairwise_sum（PW_BLOCKSIZE = 128），与 hm066.rs 用的是同一套实现。
/// pandas 的 .sum()/.mean()/.std() 都走这条路径，求和顺序会改变最后几位。
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

/// pandas nanops 的求和：先把 NaN 换成 fill（默认 0.0），再走 pairwise_sum
fn nanops_sum(v: &[f64], fill: f64) -> f64 {
    let buf: Vec<f64> = v.iter().map(|&x| if x.is_nan() { fill } else { x }).collect();
    pairwise_sum(&buf)
}

#[inline]
fn zero_out_fperr(x: f64) -> f64 {
    if x.abs() < 1e-14 {
        0.0
    } else {
        x
    }
}

#[inline]
fn count_valid(v: &[f64]) -> usize {
    v.iter().filter(|x| !x.is_nan()).count()
}

fn nan_mean(v: &[f64]) -> f64 {
    let n = count_valid(v);
    if n == 0 {
        f64::NAN
    } else {
        nanops_sum(v, 0.0) / n as f64
    }
}

/// pandas nanstd（ddof=1）：两遍算法，count - ddof <= 0 时 NaN
fn nan_std(v: &[f64]) -> f64 {
    let n = count_valid(v);
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

fn nan_sum(v: &[f64]) -> f64 {
    nanops_sum(v, 0.0)
}

/// pandas nanops.nanskew（bias=False 的修正 Fisher-Pearson G1）
fn nan_skew(v: &[f64]) -> f64 {
    let n = count_valid(v);
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

/// pandas nanops.nankurt（bias=False 的修正 Fisher 超额峰度）
fn nan_kurt(v: &[f64]) -> f64 {
    let n = count_valid(v);
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

/// np.corrcoef(a, b)[0,1]：两遍法（np.average 的成对求和 → 中心化 → 点积/(n-1)）
/// → `(c/sa)/sb` → clip 到 [-1,1]。成对完整观测由调用方事先丢掉 NaN。
fn np_corrcoef(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len();
    debug_assert_eq!(n, b.len());
    if n == 0 {
        return f64::NAN;
    }
    let fact = n as f64 - 1.0;
    let ma = pairwise_sum(a) / n as f64;
    let mb = pairwise_sum(b) / n as f64;
    let (mut daa, mut dbb, mut dab) = (0.0f64, 0.0f64, 0.0f64);
    for i in 0..n {
        let x = a[i] - ma;
        let y = b[i] - mb;
        daa += x * x;
        dbb += y * y;
        dab += x * y;
    }
    let caa = daa / fact;
    let cbb = dbb / fact;
    let cab = dab / fact;
    let sa = caa.sqrt();
    let sb = cbb.sqrt();
    let r = cab / sa / sb;
    if r.is_nan() {
        f64::NAN
    } else {
        r.clamp(-1.0, 1.0)
    }
}

/// np.cov(a, b, ddof=1)[0,1]
fn np_cov(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len();
    if n == 0 {
        return f64::NAN;
    }
    let fact = n as f64 - 1.0;
    let ma = pairwise_sum(a) / n as f64;
    let mb = pairwise_sum(b) / n as f64;
    let mut dab = 0.0f64;
    for i in 0..n {
        dab += (a[i] - ma) * (b[i] - mb);
    }
    dab / fact
}

/// Series.corr(other)：成对完整观测 + np.corrcoef
fn series_corr(a: &[f64], b: &[f64]) -> f64 {
    let mut x: Vec<f64> = Vec::with_capacity(a.len());
    let mut y: Vec<f64> = Vec::with_capacity(a.len());
    for i in 0..a.len().min(b.len()) {
        if !a[i].is_nan() && !b[i].is_nan() {
            x.push(a[i]);
            y.push(b[i]);
        }
    }
    if x.is_empty() {
        return f64::NAN;
    }
    np_corrcoef(&x, &y)
}

/// Series.cov(other)：成对完整观测 + np.cov(ddof=1)
fn series_cov(a: &[f64], b: &[f64]) -> f64 {
    let mut x: Vec<f64> = Vec::with_capacity(a.len());
    let mut y: Vec<f64> = Vec::with_capacity(a.len());
    for i in 0..a.len().min(b.len()) {
        if !a[i].is_nan() && !b[i].is_nan() {
            x.push(a[i]);
            y.push(b[i]);
        }
    }
    if x.is_empty() {
        return f64::NAN;
    }
    np_cov(&x, &y)
}

/// Series.autocorr(1) = corr(x, x.shift(1))
fn autocorr1(v: &[f64]) -> f64 {
    let n = v.len();
    if n == 0 {
        return f64::NAN;
    }
    let mut y: Vec<f64> = Vec::with_capacity(n);
    for i in 0..n {
        y.push(if i >= 1 { v[i - 1] } else { f64::NAN });
    }
    series_corr(v, &y)
}

/// rp.trend：丢掉非有限值后与「原始位置 + 1」做 Pearson；样本 < 2 或方差为 0 给 0.0。
fn rp_trend(x: &[f64]) -> f64 {
    let mut values: Vec<f64> = Vec::new();
    let mut idx: Vec<f64> = Vec::new();
    for (i, &v) in x.iter().enumerate() {
        if v.is_finite() {
            values.push(v);
            idx.push((i + 1) as f64);
        }
    }
    let n = values.len();
    if n < 2 {
        return 0.0;
    }
    let mean_x: f64 = values.iter().sum::<f64>() / n as f64;
    let mean_y: f64 = idx.iter().sum::<f64>() / n as f64;
    let (mut cov, mut var_x, mut var_y) = (0.0f64, 0.0f64, 0.0f64);
    for i in 0..n {
        let dx = values[i] - mean_x;
        let dy = idx[i] - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }
    if var_x == 0.0 || var_y == 0.0 {
        return 0.0;
    }
    cov / (var_x.sqrt() * var_y.sqrt())
}

/// pandas `pd.to_timedelta(微秒/1e6, unit="s")` 得到的时间戳（纳秒，未含 8 小时偏移）。
/// 与 hm066.rs 的 pandas_time_ns 同一份实现（cast_from_unit_vectorized）。
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

// ============================================================================
// 分钟数据：元信息 + 一天的整块读取（与 hm017.rs 同一套）
// ============================================================================

struct MinMeta {
    date_to_dayidx: std::collections::HashMap<i64, usize>,
    /// 按 symbol_map.csv 的 pos 排列的股票代码
    codes: Vec<String>,
}

static MIN_META: LazyLock<Mutex<std::collections::HashMap<PathBuf, Arc<MinMeta>>>> =
    LazyLock::new(|| Mutex::new(std::collections::HashMap::new()));

fn load_min_meta(dir: &Path) -> Option<MinMeta> {
    let cal = fs::read_to_string(dir.join("calendar_map.csv")).ok()?;
    let mut date_to_dayidx = std::collections::HashMap::new();
    let mut day = 0usize;
    for line in cal.lines() {
        let t = line.trim();
        if t.is_empty() {
            continue;
        }
        if let Ok(d) = t.parse::<i64>() {
            date_to_dayidx.insert(d, day);
            day += 1;
        }
    }
    let sym = fs::read_to_string(dir.join("symbol_map.csv")).ok()?;
    let mut pairs: Vec<(usize, String)> = Vec::new();
    for line in sym.lines() {
        let t = line.trim();
        if t.is_empty() {
            continue;
        }
        let parts: Vec<&str> = t.split(',').collect();
        if parts.len() < 2 {
            continue;
        }
        let pos: usize = match parts[1].trim().parse() {
            Ok(p) => p,
            Err(_) => continue,
        };
        pairs.push((pos, format!("{:0>6}", parts[0].trim())));
    }
    if pairs.is_empty() {
        return None;
    }
    let width = pairs.iter().map(|(p, _)| *p).max().unwrap() + 1;
    let mut codes = vec![String::new(); width];
    for (p, c) in pairs {
        codes[p] = c;
    }
    Some(MinMeta {
        date_to_dayidx,
        codes,
    })
}

fn min_meta() -> Option<Arc<MinMeta>> {
    let dir = crate::data_paths::minute_dir();
    let mut cache = MIN_META.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(m) = cache.get(&dir) {
        return Some(m.clone());
    }
    let m = Arc::new(load_min_meta(&dir)?);
    cache.insert(dir, m.clone());
    Some(m)
}

/// 读某字段某天的整块 240 行 × ncols 列，行优先铺平；最后三根置 NaN。
#[cfg(feature = "hdf5")]
fn read_day_block(field: &str, day_idx: usize, ncols: usize) -> Option<Vec<f64>> {
    use hdf5_metno as hdf5;
    use ndarray::s;

    let path = crate::data_paths::minute_file(&format!("{field}.h5"));
    let file = hdf5::File::open(&path).ok()?;
    let ds = file.dataset("data").ok()?;
    let shape = ds.shape();
    if shape.len() != 2 || ncols > shape[1] {
        return None;
    }
    let rs = day_idx * MIN_PER_DAY;
    if rs >= shape[0] {
        return None;
    }
    let re = std::cmp::min(rs + MIN_PER_DAY, shape[0]);
    let arr: ndarray::Array2<f64> = ds.read_slice_2d(s![rs..re, 0..ncols]).ok()?;
    let rows = re - rs;
    let mut v = vec![f64::NAN; MIN_PER_DAY * ncols];
    for i in 0..rows {
        if i >= MIN_NAN_FROM {
            break;
        }
        for j in 0..ncols {
            v[i * ncols + j] = arr[[i, j]];
        }
    }
    Some(v)
}

#[cfg(not(feature = "hdf5"))]
fn read_day_block(_field: &str, _day_idx: usize, _ncols: usize) -> Option<Vec<f64>> {
    None
}

// ============================================================================
// 盘口快照 → 3 秒序列
// ============================================================================

/// 一只票当天 adjust_afternoon + resample('3s').last().ffill() 之后的 4 列。
/// 桶号是**绝对**的（floor(纳秒 / 3e9)），方便直接按桶对齐。
struct Mkt3 {
    first_bin: i64,
    last_prc: Vec<f64>,
    volume: Vec<f64>,
    ask_vol1: Vec<f64>,
    bid_vol1: Vec<f64>,
}

impl Mkt3 {
    #[inline]
    fn n(&self) -> usize {
        self.last_prc.len()
    }
    #[inline]
    fn last_bin(&self) -> i64 {
        self.first_bin + self.n() as i64 - 1
    }
    /// 收益率 ret = last/last.shift(-n) - 1；空表返回 NaN
    fn ret(&self) -> f64 {
        if self.n() == 0 {
            return f64::NAN;
        }
        self.last_prc[self.n() - 1] / self.last_prc[0] - 1.0
    }
    fn volume_max(&self) -> f64 {
        nan_max(&self.volume)
    }
    /// 该组的 3 秒序列：0=volume 用 diff，1=last_prc 用 pct_change，2=abi1 用原列
    fn series(&self, group: usize) -> Vec<f64> {
        let n = self.n();
        match group {
            0 => {
                let mut v = vec![f64::NAN; n];
                for i in 1..n {
                    v[i] = self.volume[i] - self.volume[i - 1];
                }
                v
            }
            1 => self.last_prc.clone(),
            _ => {
                let mut v = vec![f64::NAN; n];
                for i in 0..n {
                    let a = self.ask_vol1[i];
                    let b = self.bid_vol1[i];
                    v[i] = (a - b) / (a + b);
                }
                v
            }
        }
    }
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

/// 读一只票的盘口快照 → 3 秒序列（f64，自己解析 CSV，不用 f32 的 fast reader）。
fn read_mkt3(code: &str, date: i64) -> Option<Mkt3> {
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
    let content = fs::read(&path).ok()?;

    // 第一遍：按行取出 adjust_afternoon 后的 (桶号, 4 列)
    let mut bins: Vec<i64> = Vec::with_capacity(6000);
    let mut cols: Vec<[f64; 4]> = Vec::with_capacity(6000);
    for (li, line) in content.split(|&b| b == b'\n').enumerate() {
        if li == 0 {
            continue; // 表头
        }
        let line = if line.last() == Some(&b'\r') {
            &line[..line.len() - 1]
        } else {
            line
        };
        if line.is_empty() {
            continue;
        }
        // market_data 有 61 列，只要 4(exchtime) / 6(last_prc) / 14(volume) / 22(ask_vol1) / 24(bid_vol1)
        let mut f: [&[u8]; 25] = [&[][..]; 25];
        let mut start = 0usize;
        let mut col = 0usize;
        for (i, &b) in line.iter().enumerate() {
            if b == b',' {
                if col < 25 {
                    f[col] = &line[start..i];
                }
                col += 1;
                start = i + 1;
                if col > 24 {
                    break;
                }
            }
        }
        if col <= 24 {
            f[col] = &line[start..];
        }
        let us = parse_f64_at(f[4]);
        if us.is_nan() {
            continue;
        }
        let ns = pandas_time_ns(us);
        if ns == i64::MIN {
            continue;
        }
        let total = ns + BJ_NS;
        let us_day = total.rem_euclid(DAY_NS) / 1000;
        let adj = if (AM_LO_US..=AM_HI_US).contains(&us_day) {
            total
        } else if (PM_LO_US..=PM_HI_US).contains(&us_day) {
            total - AFTERNOON_SHIFT_NS
        } else {
            continue;
        };
        bins.push(adj.div_euclid(THREE_S_NS));
        cols.push([
            parse_f64_at(f[6]),
            parse_f64_at(f[14]),
            parse_f64_at(f[22]),
            parse_f64_at(f[24]),
        ]);
    }
    if bins.is_empty() {
        return None;
    }

    let first = *bins.iter().min().unwrap();
    let last = *bins.iter().max().unwrap();
    let n = (last - first + 1) as usize;
    let mut last_prc = vec![f64::NAN; n];
    let mut volume = vec![f64::NAN; n];
    let mut ask_vol1 = vec![f64::NAN; n];
    let mut bid_vol1 = vec![f64::NAN; n];
    // group_last(skipna=True)：同一个桶里后面那行覆盖前面那行，但 NaN 不覆盖
    for (b, c) in bins.iter().zip(cols.iter()) {
        let k = (b - first) as usize;
        if !c[0].is_nan() {
            last_prc[k] = c[0];
        }
        if !c[1].is_nan() {
            volume[k] = c[1];
        }
        if !c[2].is_nan() {
            ask_vol1[k] = c[2];
        }
        if !c[3].is_nan() {
            bid_vol1[k] = c[3];
        }
    }
    // ffill（每列各自）
    for i in 1..n {
        if last_prc[i].is_nan() {
            last_prc[i] = last_prc[i - 1];
        }
        if volume[i].is_nan() {
            volume[i] = volume[i - 1];
        }
        if ask_vol1[i].is_nan() {
            ask_vol1[i] = ask_vol1[i - 1];
        }
        if bid_vol1[i].is_nan() {
            bid_vol1[i] = bid_vol1[i - 1];
        }
    }
    Some(Mkt3 {
        first_bin: first,
        last_prc,
        volume,
        ask_vol1,
        bid_vol1,
    })
}

/// 两只票的 3 秒序列在**共同桶区间**上做 Series.corr（成对完整观测 + np.corrcoef）。
fn corr_common(t: &Mkt3, ts: &[f64], p: &Mkt3, ps: &[f64]) -> f64 {
    let lo = t.first_bin.max(p.first_bin);
    let hi = t.last_bin().min(p.last_bin());
    if hi < lo {
        return f64::NAN;
    }
    let mut a: Vec<f64> = Vec::new();
    let mut b: Vec<f64> = Vec::new();
    let mut bin = lo;
    while bin <= hi {
        let x = ts[(bin - t.first_bin) as usize];
        let y = ps[(bin - p.first_bin) as usize];
        if !x.is_nan() && !y.is_nan() {
            a.push(x);
            b.push(y);
        }
        bin += 1;
    }
    if a.is_empty() {
        return f64::NAN;
    }
    np_corrcoef(&a, &b)
}

// ============================================================================
// 分钟横截面相关：每只票的 top10 peer
// ============================================================================

/// 一列的有效行掩码（240 位）
type Mask = [u64; 4];

#[inline]
fn mask_of(mat: &[f64], ncols: usize, j: usize) -> Mask {
    let mut m: Mask = [0u64; 4];
    for i in 0..MIN_PER_DAY {
        if !mat[i * ncols + j].is_nan() {
            m[i / 64] |= 1u64 << (i % 64);
        }
    }
    m
}

#[inline]
fn mask_rows(m: &Mask) -> Vec<usize> {
    let mut out = Vec::with_capacity(240);
    for i in 0..MIN_PER_DAY {
        if m[i / 64] >> (i % 64) & 1 == 1 {
            out.push(i);
        }
    }
    out
}

/// 逐对完整观测的 Pearson（一般路径）
fn pairwise_pearson(mat: &[f64], ncols: usize, j: usize, k: usize, rows_j: &[usize]) -> f64 {
    let mut a: Vec<f64> = Vec::with_capacity(rows_j.len());
    let mut b: Vec<f64> = Vec::with_capacity(rows_j.len());
    for &i in rows_j.iter() {
        let x = mat[i * ncols + j];
        let y = mat[i * ncols + k];
        if !y.is_nan() {
            a.push(x);
            b.push(y);
        }
    }
    if a.is_empty() {
        return f64::NAN;
    }
    np_corrcoef(&a, &b)
}

/// 对一组分钟矩阵，给每一列算「与该列相关性最高的 10 列」。返回 vec[j] = Some([(列号, 相关系数);<=10])
fn top_pairs(mat: &[f64], ncols: usize) -> Vec<Option<Vec<(usize, f64)>>> {
    // 1) 每列的掩码，统计出现次数挑主导掩码（只统计至少 2 个有效行的列）
    let masks: Vec<Mask> = (0..ncols).map(|j| mask_of(mat, ncols, j)).collect();
    let mut counts: std::collections::HashMap<Mask, usize> = std::collections::HashMap::new();
    for (j, &m) in masks.iter().enumerate() {
        if mask_rows(&m).len() >= 2 {
            *counts.entry(m).or_insert(0) += 1;
        }
        let _ = j;
    }
    let dom: Option<Mask> = counts
        .into_iter()
        .max_by_key(|(_, c)| *c)
        .map(|(m, _)| m);
    let dom = match dom {
        Some(m) => m,
        None => return vec![None; ncols],
    };
    let rows = mask_rows(&dom);
    let nr = rows.len();
    let mut dom_cols: Vec<usize> = Vec::new();
    for j in 0..ncols {
        if masks[j] == dom {
            dom_cols.push(j);
        }
    }
    let nd = dom_cols.len();
    // 列主序：z[col_idx * nr + k]
    let mut z = vec![0.0f64; nd * nr];
    let mut ss = vec![0.0f64; nd];
    let mut col_index = vec![usize::MAX; ncols];
    for (c, &j) in dom_cols.iter().enumerate() {
        col_index[j] = c;
    }
    for (c, &j) in dom_cols.iter().enumerate() {
        let mut sum = 0.0f64;
        let mut buf = Vec::with_capacity(nr);
        for &i in rows.iter() {
            let v = mat[i * ncols + j];
            buf.push(v);
            sum += v;
        }
        let mean = pairwise_sum(&buf) / nr as f64;
        let mut s2 = 0.0f64;
        for k in 0..nr {
            let d = buf[k] - mean;
            z[c * nr + k] = d;
            s2 += d * d;
        }
        ss[c] = s2;
    }
    // 非主导列的有效行
    let others: Vec<usize> = (0..ncols)
        .filter(|&j| masks[j] != dom && mask_rows(&masks[j]).len() >= 2)
        .collect();
    let other_rows: Vec<Vec<usize>> = others.iter().map(|&j| mask_rows(&masks[j])).collect();
    let other_pos: std::collections::HashMap<usize, usize> =
        others.iter().enumerate().map(|(p, &j)| (j, p)).collect();

    let fact = nr as f64 - 1.0;

    (0..ncols)
        .into_par_iter()
        .map(|j| {
            if masks[j] == dom {
                let cj = col_index[j];
                let sj = ss[cj];
                let mut acc: Vec<(usize, f64)> = Vec::with_capacity(nd);
                let zj = &z[cj * nr..cj * nr + nr];
                for (c, &k) in dom_cols.iter().enumerate() {
                    if k == j {
                        continue;
                    }
                    let zk = &z[c * nr..c * nr + nr];
                    let mut dot = 0.0f64;
                    for t in 0..nr {
                        dot += zj[t] * zk[t];
                    }
                    let cab = dot / fact;
                    let sa = (sj / fact).sqrt();
                    let sb = (ss[c] / fact).sqrt();
                    let r = cab / sa / sb;
                    let r = if r.is_nan() { f64::NAN } else { r.clamp(-1.0, 1.0) };
                    if !r.is_nan() {
                        acc.push((k, r));
                    }
                }
                for &k in others.iter() {
                    let e = other_pos[&k];
                    let r = pairwise_pearson(mat, ncols, j, k, &other_rows[e]);
                    if !r.is_nan() {
                        acc.push((k, r));
                    }
                }
                if acc.is_empty() {
                    return None;
                }
                // 降序；并列时按列号升序（numpy quicksort 的并列顺序不可复现，这里取确定的一种）
                acc.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap().then(a.0.cmp(&b.0)));
                acc.truncate(TOP_NUMBER);
                Some(acc)
            } else {
                let rows_j = mask_rows(&masks[j]);
                if rows_j.len() < 2 {
                    return None;
                }
                let mut acc: Vec<(usize, f64)> = Vec::with_capacity(ncols);
                for k in 0..ncols {
                    if k == j {
                        continue;
                    }
                    let r = pairwise_pearson(mat, ncols, j, k, &rows_j);
                    if !r.is_nan() {
                        acc.push((k, r));
                    }
                }
                if acc.is_empty() {
                    return None;
                }
                acc.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap().then(a.0.cmp(&b.0)));
                acc.truncate(TOP_NUMBER);
                Some(acc)
            }
        })
        .collect()
}

// ============================================================================
// 一组的 89 个指标
// ============================================================================

/// 一只票在一个组上的 89 个指标。任何一步拿不到（peer 缺盘口、序列为空）返回 None。
fn group_metrics(
    t_idx: usize,
    pairs: &[(usize, f64)],
    mkt: &[Option<Mkt3>],
    group: usize,
) -> Option<[f64; N_METRIC]> {
    let t = mkt[t_idx].as_ref()?;
    if t.n() == 0 || pairs.is_empty() {
        return None;
    }
    let ts = t.series(group);

    // peer 的 3 秒序列与 corrs_3s
    let mut corr1m: Vec<f64> = Vec::with_capacity(pairs.len());
    let mut corr3s: Vec<f64> = Vec::with_capacity(pairs.len());
    let mut peer_ret: Vec<f64> = Vec::with_capacity(pairs.len());
    let mut peer_vmax: Vec<f64> = Vec::with_capacity(pairs.len());
    for &(k, c1) in pairs.iter() {
        let p = mkt[k].as_ref()?;
        if p.n() == 0 {
            return None;
        }
        let ps = p.series(group);
        let c3 = corr_common(t, &ts, p, &ps);
        corr1m.push(c1);
        corr3s.push(c3);
        peer_ret.push(p.ret());
        peer_vmax.push(p.volume_max());
    }

    let corr_1m3s = series_corr(&corr1m, &corr3s);
    let cov_1m3s = series_cov(&corr1m, &corr3s);
    let trend_diff_1m3s = rp_trend(&corr1m) - rp_trend(&corr3s);
    let cdr_total = nan_sum(&corr3s) / nan_sum(&corr1m);
    let cdrs: Vec<f64> = corr3s
        .iter()
        .zip(corr1m.iter())
        .map(|(&a, &b)| a / b)
        .collect();

    let c_mean = nan_mean(&cdrs);
    let c_std = nan_std(&cdrs);
    let c_skew = nan_skew(&cdrs);
    let c_kurt = nan_kurt(&cdrs);
    let c_max = nan_max(&cdrs);
    let c_min = nan_min(&cdrs);
    let c_auto = autocorr1(&cdrs);
    let c_trend = rp_trend(&cdrs);

    let m_mean = nan_mean(&corr1m);
    let m_std = nan_std(&corr1m);
    let m_skew = nan_skew(&corr1m);
    let m_kurt = nan_kurt(&corr1m);
    let m_max = nan_max(&corr1m);
    let m_min = nan_min(&corr1m);
    let m_auto = autocorr1(&corr1m);
    let m_trend = rp_trend(&corr1m);

    let s_mean = nan_mean(&corr3s);
    let s_std = nan_std(&corr3s);
    let s_skew = nan_skew(&corr3s);
    let s_kurt = nan_kurt(&corr3s);
    let s_max = nan_max(&corr3s);
    let s_min = nan_min(&corr3s);
    let s_auto = autocorr1(&corr3s);
    let s_trend = rp_trend(&corr3s);

    let std_diff = m_std / m_mean - s_std / s_mean;
    let skew_diff = m_skew - s_skew;
    let kurt_diff = m_kurt - s_kurt;
    let max_diff = m_max / m_mean - s_max / s_mean;
    let min_diff = m_min / m_mean - s_min / s_mean;
    let auto_diff = m_auto - s_auto;
    let auto_abs_diff = m_auto.abs() - s_auto.abs();
    let trend_diff = m_trend - s_trend;
    let trend_abs_diff = m_trend.abs() - s_trend.abs();

    // new_top_sort：corrs_3s 降序（NaN 排最后），index 是 10 只 peer 里的位置
    let mut order: Vec<usize> = (0..corr3s.len()).collect();
    order.sort_by(|&a, &b| {
        let (x, y) = (corr3s[a], corr3s[b]);
        match (x.is_nan(), y.is_nan()) {
            (true, true) => a.cmp(&b),
            (true, false) => std::cmp::Ordering::Greater,
            (false, true) => std::cmp::Ordering::Less,
            _ => y.partial_cmp(&x).unwrap().then(a.cmp(&b)),
        }
    });

    let top1_ret_1m = peer_ret[0];
    let top1_ret_3s = peer_ret[order[0]];
    let top5_ret_1m = {
        let mut s = 0.0f64;
        for i in 0..5.min(peer_ret.len()) {
            s += peer_ret[i];
        }
        s / 5.0
    };
    let top5_ret_3s = {
        let mut s = 0.0f64;
        for i in 0..5.min(order.len()) {
            s += peer_ret[order[i]];
        }
        s / 5.0
    };
    let ret_self = t.ret();

    let top1_volume_1m = peer_vmax[0];
    let top1_volume_3s = peer_vmax[order[0]];
    let top5_volume_1m = {
        let mut s = 0.0f64;
        for i in 0..5.min(peer_vmax.len()) {
            s += peer_vmax[i];
        }
        s / 5.0
    };
    let top5_volume_3s = {
        let mut s = 0.0f64;
        for i in 0..5.min(order.len()) {
            s += peer_vmax[order[i]];
        }
        s / 5.0
    };
    let volume_self = t.volume_max();

    let mut r = [0.0f64; N_METRIC];
    let mut p = 0usize;
    let mut push = |v: f64, r: &mut [f64; N_METRIC], p: &mut usize| {
        r[*p] = v;
        *p += 1;
    };
    push(corr_1m3s, &mut r, &mut p);
    push(cov_1m3s, &mut r, &mut p);
    push(corr_1m3s.abs(), &mut r, &mut p);
    push(cov_1m3s.abs(), &mut r, &mut p);
    push(trend_diff_1m3s, &mut r, &mut p);
    push(trend_diff_1m3s.abs(), &mut r, &mut p);
    push(cdr_total, &mut r, &mut p);
    push(c_mean, &mut r, &mut p);
    push(c_std, &mut r, &mut p);
    push(c_skew, &mut r, &mut p);
    push(c_kurt, &mut r, &mut p);
    push(c_max, &mut r, &mut p);
    push(c_min, &mut r, &mut p);
    push(c_auto, &mut r, &mut p);
    push(c_auto.abs(), &mut r, &mut p);
    push(c_trend, &mut r, &mut p);
    push(c_trend.abs(), &mut r, &mut p);
    push(m_mean, &mut r, &mut p);
    push(m_std, &mut r, &mut p);
    push(m_skew, &mut r, &mut p);
    push(m_kurt, &mut r, &mut p);
    push(m_max, &mut r, &mut p);
    push(m_min, &mut r, &mut p);
    push(m_auto, &mut r, &mut p);
    push(m_auto.abs(), &mut r, &mut p);
    push(m_trend, &mut r, &mut p);
    push(m_trend.abs(), &mut r, &mut p);
    push(s_mean, &mut r, &mut p);
    push(s_std, &mut r, &mut p);
    push(s_skew, &mut r, &mut p);
    push(s_kurt, &mut r, &mut p);
    push(s_max, &mut r, &mut p);
    push(s_min, &mut r, &mut p);
    push(s_auto, &mut r, &mut p);
    push(s_auto.abs(), &mut r, &mut p);
    push(s_trend, &mut r, &mut p);
    push(s_trend.abs(), &mut r, &mut p);
    push(std_diff, &mut r, &mut p);
    push(skew_diff, &mut r, &mut p);
    push(kurt_diff, &mut r, &mut p);
    push(max_diff, &mut r, &mut p);
    push(min_diff, &mut r, &mut p);
    push(auto_diff, &mut r, &mut p);
    push(auto_abs_diff, &mut r, &mut p);
    push(trend_diff, &mut r, &mut p);
    push(trend_abs_diff, &mut r, &mut p);
    push(std_diff.abs(), &mut r, &mut p);
    push(skew_diff.abs(), &mut r, &mut p);
    push(kurt_diff.abs(), &mut r, &mut p);
    push(max_diff.abs(), &mut r, &mut p);
    push(min_diff.abs(), &mut r, &mut p);
    push(auto_diff.abs(), &mut r, &mut p);
    push(auto_abs_diff.abs(), &mut r, &mut p);
    push(trend_diff.abs(), &mut r, &mut p);
    push(trend_abs_diff.abs(), &mut r, &mut p);
    push(top1_ret_1m, &mut r, &mut p);
    push(top1_ret_3s, &mut r, &mut p);
    push(top5_ret_1m, &mut r, &mut p);
    push(top5_ret_3s, &mut r, &mut p);
    push(ret_self, &mut r, &mut p);
    push(top1_ret_1m - top1_ret_3s, &mut r, &mut p);
    push(top5_ret_1m - top5_ret_3s, &mut r, &mut p);
    push((top1_ret_1m - top1_ret_3s).abs(), &mut r, &mut p);
    push((top5_ret_1m - top5_ret_3s).abs(), &mut r, &mut p);
    push(top1_ret_1m - ret_self, &mut r, &mut p);
    push(top5_ret_1m - ret_self, &mut r, &mut p);
    push((top1_ret_1m - ret_self).abs(), &mut r, &mut p);
    push((top5_ret_1m - ret_self).abs(), &mut r, &mut p);
    push(top1_ret_3s - ret_self, &mut r, &mut p);
    push(top5_ret_3s - ret_self, &mut r, &mut p);
    push((top1_ret_3s - ret_self).abs(), &mut r, &mut p);
    push((top5_ret_3s - ret_self).abs(), &mut r, &mut p);
    push(top1_volume_1m, &mut r, &mut p);
    push(top1_volume_3s, &mut r, &mut p);
    push(top5_volume_1m, &mut r, &mut p);
    push(top5_volume_3s, &mut r, &mut p);
    push(volume_self, &mut r, &mut p);
    push(top1_volume_1m / top1_volume_3s, &mut r, &mut p);
    push(top5_volume_1m / top5_volume_3s, &mut r, &mut p);
    push(top1_volume_1m / volume_self, &mut r, &mut p);
    push(top5_volume_1m / volume_self, &mut r, &mut p);
    push(top1_volume_3s / volume_self, &mut r, &mut p);
    push(top5_volume_3s / volume_self, &mut r, &mut p);
    push((top1_volume_1m / top1_volume_3s - 1.0).abs(), &mut r, &mut p);
    push((top5_volume_1m / top5_volume_3s - 1.0).abs(), &mut r, &mut p);
    push((top1_volume_1m / volume_self - 1.0).abs(), &mut r, &mut p);
    push((top5_volume_1m / volume_self - 1.0).abs(), &mut r, &mut p);
    push((top1_volume_3s / volume_self - 1.0).abs(), &mut r, &mut p);
    push((top5_volume_3s / volume_self - 1.0).abs(), &mut r, &mut p);
    if p != N_METRIC {
        return None;
    }
    Some(r)
}

// ============================================================================
// 入口
// ============================================================================

/// hm74 是横截面系列，没有逐股版本。
pub fn compute_stock(_code: &str, _date: i64) -> Option<Vec<f32>> {
    None
}

/// 一天一次：读全市场分钟表挑 peer、读全市场盘口快照一次、逐股算 267 个值。
pub fn compute_cross(date: i64) -> Option<(Vec<String>, Vec<f32>)> {
    let meta = min_meta()?;
    let day_idx = *meta.date_to_dayidx.get(&date)?;
    let ncols = meta.codes.len();

    let volume = read_day_block("volume", day_idx, ncols)?;
    let close = read_day_block("close", day_idx, ncols)?;
    let ask1 = read_day_block("ask_vol1", day_idx, ncols)?;
    let bid1 = read_day_block("bid_vol1", day_idx, ncols)?;
    let tot = ncols * MIN_PER_DAY;
    let mut abi1 = vec![f64::NAN; tot];
    for i in 0..tot {
        let a = ask1[i];
        let b = bid1[i];
        abi1[i] = (a - b) / (a + b);
    }

    let g0 = top_pairs(&volume, ncols);
    let g1 = top_pairs(&close, ncols);
    let g2 = top_pairs(&abi1, ncols);
    let groups: [&Vec<Option<Vec<(usize, f64)>>>; 3] = [&g0, &g1, &g2];

    // 全市场盘口快照：一天只读一次
    let mkt: Vec<Option<Mkt3>> = (0..ncols)
        .into_par_iter()
        .map(|j| read_mkt3(&meta.codes[j], date))
        .collect();

    let rows: Vec<Option<Vec<f32>>> = (0..ncols)
        .into_par_iter()
        .map(|j| {
            let mut out: Vec<f32> = Vec::with_capacity(N_OUT);
            for (gi, g) in groups.iter().enumerate() {
                let pairs = g[j].as_ref()?;
                let m = group_metrics(j, pairs, &mkt, gi)?;
                out.extend(m.iter().map(|&v| v as f32));
            }
            Some(out)
        })
        .collect();

    let mut codes: Vec<String> = Vec::new();
    let mut vals: Vec<f32> = Vec::new();
    for (j, r) in rows.into_iter().enumerate() {
        if let Some(v) = r {
            if v.len() != N_OUT {
                continue;
            }
            codes.push(meta.codes[j].clone());
            vals.extend(v);
        }
    }
    Some((codes, vals))
}
