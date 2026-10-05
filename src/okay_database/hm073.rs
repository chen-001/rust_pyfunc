//! hm73「八戒照镜子」的 Rust 复现（旧框架 design_whatever_out.TickDataBase，逐股系列）。
//!
//! 旧文件：/home/chenzongwei/pythoncode/成果hm73_八戒照镜子/hm73_okay.py
//!   调用参数 get_many=0 / read_full=0 / change_time=1（见旧文件 run_base 里 run_factor_out 的
//!   实参），也就是「一整天的盘口快照」一张表，窗口由旧代码自己切。
//!
//! ════════════════════════════════════════════════════════════════════
//! 一、旧代码逐步对照（写代码前先抄在这里，方便复核）
//! ════════════════════════════════════════════════════════════════════
//! 1) df = read_market_data_single()  → 一天的盘口快照（全部列，不做涨跌停过滤）
//!    deal_data 把 exchtime 换成 Timestamp = to_timedelta(exchtime/1e6, 's') + 1970-01-01 08:00
//! 2) df = dwo.adjust_afternoon(df)：按 exchtime 的钟点保留
//!        [09:30:00, 11:30:00] ∪ [13:00:00, 14:57:00]（两端都含，微秒/纳秒精度）
//!    下午那一段整体左移 90 分钟，然后 pd.concat([上午, 下午]) → reset_index()
//!    行序 = 上午（文件序）在前、下午（文件序）在后
//! 3) volatility = last_prc.pct_change().rolling(20).std()  （ddof=1，min_periods=20）
//!    —— pct_change 是 x[i]/x[i-1] - 1，首行 NaN；前 20 行 volatility 是 NaN
//! 4) volume = volume - volume.shift(1)   （首行 NaN）
//! 5) 8 个量 ask_vol2..5 / bid_vol2..5，各算：
//!      corr      = Series.corr(该量, volatility)   ← 成对完整观测
//!      resi      = rp.ols_residuals(X=[volatility], y=该量)  （带截距）
//!      resi_abs  = |resi|
//!      resi_adj  = resi * (corr >= 0 ? 1 : -1)      ← 注意 corr 是 NaN 时判 False，符号取 -1
//!      rank      = rp.pandas_series_rank(该量) / max   （average、升序、NaN 保持）
//!      resi_rank / resi_adj_rank 同理
//!      rank_diff      = resi_rank - rank
//!      adj_rank_diff  = resi_adj_rank - rank
//! 6) 行方向（axis=1）的 trend：对 [ask_vol2_resi, ask_vol3_resi, ask_vol4_resi, ask_vol5_resi]
//!    与 bid 的四个、以及 adjusted 版各算一个「值与 1 基位置的 Pearson」→ 4 条序列
//!      ask_resi_diff / bid_resi_diff / ask_resi_adjusted_diff / bid_resi_adjusted_diff
//!    全为 NaN 的行 trend 给 0.0（不是 NaN），所以这 4 条序列永远有限
//!
//! 7) 一列一列地算（cols 共 40 个 = 4 个档位 × ask/bid × 5 个变体）：
//!      q80 = 该列的 0.8 分位、q20 = 0.2 分位（线性插值，跳过 NaN）
//!      large = 该列 >= q80 的行、small = 该列 <= q20 的行（NaN 不入选）
//!      以 col_short（如 ask_vol2，即去掉后面的变体后缀）与 volatility 为原料：
//!        large_volume_rate / small_volume_rate = 子集 volume 和 / 全体 volume 和
//!        large_order_rate / small_order_rate   = 子集 col_short 均值 / 全体均值
//!        large/small_volatility_mean            = 子集 volatility 均值
//!        large/small_volatility_rate            = 子集均值 / 全体均值
//!        large/small_corr_volatility_order      = 子集内 corr(col_short, volatility)
//!        *_standardized                         = 上面的 corr 减去全体的 corr
//!      然后 res_twins = get_features_factors(两列的 DataFrame)：
//!        变体名里含 "adjusted" 就用 adjusted 那两条，否则用原始那两条
//!      12 + 26 = 38 个值/列，共 1520
//! 8) res = corrs(8) + res_four(60) + ress(1520) = 1588，顺序即旧文件里那份字面 names 清单。
//!    names 清单已逐项核对：程序生成的顺序与旧文件第 426 行那份 1588 项的字面清单完全一致。
//!
//! ════════════════════════════════════════════════════════════════════
//! 二、几个必须照抄的语义（踩过的地方）
//! ════════════════════════════════════════════════════════════════════
//! · 分位数：pandas Series.quantile 走 np.percentile(..., method='linear')，
//!   虚拟下标是 (n-1)*q（numpy 对 linear 专门用这个式子，不用 n*q+(1-q)-1），
//!   插值用 numpy 的 _lerp：t < 0.5 时 a + (b-a)*t，t >= 0.5 时 b - (b-a)*(1-t)。
//!   这里必须逐步照抄：large/small 是「>= / <= 分位点」的硬掩码，
//!   分位点算错 1 ulp 就可能把一行挪进或挪出 large，直接改变 large_volume_rate。
//!   （实测 000001@20250303 上本实现与 pandas 的 q80/q20 逐位相同。）
//! · pandas 的 mean/sum/var/skew/kurt 走 numpy 的成对求和（PW_BLOCKSIZE=128、8 路展开），
//!   NaN 先填 0 再求和、按有效观测数计数。这里照抄 pairwise_sum，
//!   因为 resi 的均值数学上恰好是 0，顺序求和会把它算成 1e-11 量级的噪声。
//! · 三种相关系数在旧代码里各走一条实现，不能混：
//!     Series.corr / DataFrame.corr  → pandas 的 Welford 在线算法（corr 与符号翻转有关，
//!                                    所以照抄 Welford，见 hm013.rs 里那份 pearson）
//!     rp.corrwith(df, df.shift(1))  → src/statistics/mod.rs 的 calculate_correlation_optimized
//!                                    （先求均值、再按 32 个一批分块累加）
//!     rp.fast_correlation_matrix_v2_df → src/statistics/fast_correlation_v2.rs 的
//!                                     compute_correlation_from_clean_data（8 路展开顺序累加，
//!                                     均值走 Kahan 求和）。本系列传进去的两列永远有限，
//!                                     所以走的是 clean 分支；含 NaN 的分支也照抄了，以防万一。
//! · trend：非有限值先丢掉、位置用 1 基、样本 <2 或任一侧方差为 0 时给 0.0（不是 NaN）。
//! · 撤单过滤与逐笔数据本系列完全不用，只读 market_data。
//!
//! ════════════════════════════════════════════════════════════════════
//! 三、已知的数值差异与影响面（都不影响验收）
//! ════════════════════════════════════════════════════════════════════
//! · rolling(20).std() 逐行复刻了 pandas 的 roll_var（见 rolling_std 的注释）：
//!   两遍法只差最后一位，但这个量要做回归自变量、残差再去做 rank，
//!   1 ulp 就足以让名次变掉、进而让 rank_diff 那一族超 1e-4。
//! · rp.ols_residuals 必须用 ndarray 的 .dot() 照抄（见 ols_residual 的注释），
//!   顺序求和会让残差的并列结构变掉、进而让 rank 变掉。
//! · 三种 Pearson 之间彼此的末位差在 1e-15 相对量级；只有当某个 corr 恰好落在
//!   0 附近（|corr| < 1e-15）才可能翻转 resi_adjusted 的符号，实测没有这种列。
//! · names_save 里有 `_fold` 结尾的名字（旧代码 dwo.get_abs 的结果，即「与当日截面均值的距离」），
//!   新流水线按后缀自动生成，本文件不参与。
//!
//! ════════════════════════════════════════════════════════════════════
//! 四、names_single 有 1588 个。逐项比过旧文件那份字面清单，没有重名，不需要 _dup 后缀。

use std::fs::File;
use std::io::Read;

use ndarray::{Array1, Array2};

// ============================================================================
// 盘口 CSV 的列索引（0 基，与文件表头一致；2018 及更早的文件末尾多一列 num）
// ============================================================================
const COL_EXCHTIME: usize = 4;
const COL_LAST_PRC: usize = 6;
const COL_VOLUME: usize = 14;
const COL_ASK_VOL2: usize = 26;
const COL_BID_VOL2: usize = 28;
const COL_ASK_VOL3: usize = 30;
const COL_BID_VOL3: usize = 32;
const COL_ASK_VOL4: usize = 34;
const COL_BID_VOL4: usize = 36;
const COL_ASK_VOL5: usize = 38;
const COL_BID_VOL5: usize = 40;
const MAX_COLS: usize = 64;

/// 8 个量，顺序即旧代码 corrs 列表的顺序（也是 names 里 corr 块的顺序）
const SERIES_COLS: [usize; 8] = [
    COL_ASK_VOL2,
    COL_ASK_VOL3,
    COL_ASK_VOL4,
    COL_ASK_VOL5,
    COL_BID_VOL2,
    COL_BID_VOL3,
    COL_BID_VOL4,
    COL_BID_VOL5,
];
const SERIES_NAMES: [&str; 8] = [
    "ask_vol2", "ask_vol3", "ask_vol4", "ask_vol5", "bid_vol2", "bid_vol3", "bid_vol4", "bid_vol5",
];

/// 每个量的 5 个变体，顺序即旧代码 cols 列表里每个量的 5 项
const VARIANT5: [&str; 5] = [
    "resi",
    "resi_abs",
    "resi_adjusted",
    "resi_rank_diff",
    "resi_adjusted_rank_diff",
];

/// get_features_factors 每个量算的 12 个统计量，顺序即旧代码 res 列表
const STATS12: [&str; 12] = [
    "mean",
    "mean_abs",
    "std",
    "std_abs",
    "skew",
    "skew_abs",
    "kurt",
    "kurt_abs",
    "autocorr1",
    "autocorr1_abs",
    "trend",
    "trend_abs",
];

/// 每列 one() 里那 12 个基础量的后缀，顺序即旧代码 res 列表（和 STATS12 不是一回事）
const BLOCK12: [&str; 12] = [
    "large_volume_rate",
    "small_volume_rate",
    "large_order_rate",
    "small_order_rate",
    "large_volatility_mean",
    "small_volatility_mean",
    "large_volatility_rate",
    "small_volatility_rate",
    "large_corr_volatility_order",
    "small_corr_volatility_order",
    "large_corr_volatility_order_standardized",
    "small_corr_volatility_order_standardized",
];

const FOUR_COLS: [&str; 4] = [
    "ask_resi_diff",
    "bid_resi_diff",
    "ask_resi_adjusted_diff",
    "bid_resi_adjusted_diff",
];

const N_CORR: usize = 8;
const N_FOUR: usize = 12 * 4 + 4 * 3; // 60
const N_TWINS: usize = 12 * 2 + 2; // 26
const N_PER_COL: usize = 12 + N_TWINS; // 38
pub const N_OUT: usize = N_CORR + N_FOUR + 40 * N_PER_COL; // 1588

// ============================================================================
// 时间常量（纳秒）
// ============================================================================
const NS_PER_SEC: i64 = 1_000_000_000;
const DAY_NS: i64 = 86_400 * NS_PER_SEC;
const OFFSET_8H_NS: i64 = 8 * 3600 * NS_PER_SEC;
/// 下午整体左移 90 分钟（adjust_afternoon）
const SHIFT90_NS: i64 = 90 * 60 * NS_PER_SEC;

const LD_0930: i64 = (9 * 3600 + 30 * 60) * NS_PER_SEC;
const LD_1130: i64 = (11 * 3600 + 30 * 60) * NS_PER_SEC;
const LD_1300: i64 = 13 * 3600 * NS_PER_SEC;
const LD_1457: i64 = (14 * 3600 + 57 * 60) * NS_PER_SEC;

// ============================================================================
// 名称清单
// ============================================================================

/// get_features_factors 的名称块（与 stats_block 的取值顺序严格对齐）
fn stats_names(cols: &[&str], prefix: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::with_capacity(12 * cols.len() + cols.len() * (cols.len() - 1));
    for s in STATS12.iter() {
        for c in cols.iter() {
            out.push(format!("{}{}_{}", prefix, c, s));
        }
    }
    for i in 0..cols.len() {
        for j in (i + 1)..cols.len() {
            out.push(format!("{}{}_corr_{}", prefix, cols[i], cols[j]));
        }
    }
    for i in 0..cols.len() {
        for j in (i + 1)..cols.len() {
            out.push(format!("{}{}_corr_{}_abs", prefix, cols[i], cols[j]));
        }
    }
    out
}

pub fn names() -> Vec<String> {
    let mut out: Vec<String> = Vec::with_capacity(N_OUT);
    for s in SERIES_NAMES.iter() {
        out.push(format!("corr_{}_volatility", s));
    }
    out.extend(stats_names(&FOUR_COLS, ""));

    for level in 0..4 {
        for side in 0..2 {
            let side_name = if side == 0 { "ask" } else { "bid" };
            let short = format!("{}_vol{}", side_name, level + 2);
            for variant in VARIANT5.iter() {
                let col = format!("{}_{}", short, variant);
                for s in BLOCK12.iter() {
                    out.push(format!("{}_{}", col, s));
                }
                let twins: [&str; 2] = if col.contains("adjusted") {
                    ["ask_resi_adjusted_diff", "bid_resi_adjusted_diff"]
                } else {
                    ["ask_resi_diff", "bid_resi_diff"]
                };
                for s in STATS12.iter() {
                    out.push(format!("{}_{}_{}", col, twins[0], s));
                    out.push(format!("{}_{}_{}", col, twins[1], s));
                }
                out.push(format!("{}_{}_corr_{}", col, twins[0], twins[1]));
                out.push(format!("{}_{}_corr_{}_abs", col, twins[0], twins[1]));
            }
        }
    }
    debug_assert_eq!(out.len(), N_OUT);
    out
}

// ============================================================================
// pandas / numpy 的小工具（语义照抄，见文件头）
// ============================================================================

#[inline]
fn is_ok(v: f64) -> bool {
    !v.is_nan()
}

/// numpy 的成对求和（PW_BLOCKSIZE=128、8 路展开）。
fn pairwise_sum(a: &[f64]) -> f64 {
    const PW_BLOCKSIZE: usize = 128;
    let n = a.len();
    if n < 8 {
        let mut res = 0.0f64;
        for &x in a.iter() {
            res += x;
        }
        return res;
    }
    if n <= PW_BLOCKSIZE {
        let mut r = [a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]];
        let mut i = 8usize;
        while i + 8 <= n {
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

/// pandas nanops 里的 _zero_out_fperr：|v| < 1e-14 归零
#[inline]
fn zero_out_fperr(v: f64) -> f64 {
    if v.abs() < 1e-14 {
        0.0
    } else {
        v
    }
}

/// 把 NaN 换成 0 的一份拷贝 + 有效观测数（pandas 的 skipna 路径都是这么干的）
fn filled(v: &[f64]) -> (Vec<f64>, usize) {
    let mut out = Vec::with_capacity(v.len());
    let mut cnt = 0usize;
    for &x in v.iter() {
        if x.is_nan() {
            out.push(0.0);
        } else {
            out.push(x);
            cnt += 1;
        }
    }
    (out, cnt)
}

/// Series.mean()（skipna）
fn nan_mean(v: &[f64]) -> f64 {
    let (f, cnt) = filled(v);
    if cnt == 0 {
        return f64::NAN;
    }
    pairwise_sum(&f) / cnt as f64
}

/// Series.sum()（skipna，min_count=0：全 NaN 给 0）
fn nan_sum(v: &[f64]) -> f64 {
    let (f, _) = filled(v);
    pairwise_sum(&f)
}

/// Series.std(ddof=1)
fn nan_std(v: &[f64]) -> f64 {
    let (f, cnt) = filled(v);
    if cnt <= 1 {
        return f64::NAN;
    }
    let avg = pairwise_sum(&f) / cnt as f64;
    let mut sqr: Vec<f64> = Vec::with_capacity(f.len());
    for (i, &x) in f.iter().enumerate() {
        if v[i].is_nan() {
            sqr.push(0.0);
        } else {
            sqr.push((avg - x) * (avg - x));
        }
    }
    (pairwise_sum(&sqr) / (cnt - 1) as f64).sqrt()
}

/// Series.skew()（bias=False，修正 Fisher-Pearson）
fn nan_skew(v: &[f64]) -> f64 {
    let (f, cnt) = filled(v);
    if cnt < 3 {
        return f64::NAN;
    }
    let mean = pairwise_sum(&f) / cnt as f64;
    let mut a2: Vec<f64> = Vec::with_capacity(f.len());
    let mut a3: Vec<f64> = Vec::with_capacity(f.len());
    for (i, &x) in f.iter().enumerate() {
        if v[i].is_nan() {
            a2.push(0.0);
            a3.push(0.0);
        } else {
            let d = x - mean;
            a2.push(d * d);
            a3.push(d * d * d);
        }
    }
    let m2 = zero_out_fperr(pairwise_sum(&a2));
    let m3 = zero_out_fperr(pairwise_sum(&a3));
    if m2 == 0.0 {
        return 0.0;
    }
    let c = cnt as f64;
    (c * (c - 1.0).sqrt() / (c - 2.0)) * (m3 / m2.powf(1.5))
}

/// Series.kurt()（bias=False，超额峰度）
fn nan_kurt(v: &[f64]) -> f64 {
    let (f, cnt) = filled(v);
    if cnt < 4 {
        return f64::NAN;
    }
    let mean = pairwise_sum(&f) / cnt as f64;
    let mut a2: Vec<f64> = Vec::with_capacity(f.len());
    let mut a4: Vec<f64> = Vec::with_capacity(f.len());
    for (i, &x) in f.iter().enumerate() {
        if v[i].is_nan() {
            a2.push(0.0);
            a4.push(0.0);
        } else {
            let d = x - mean;
            let d2 = d * d;
            a2.push(d2);
            a4.push(d2 * d2);
        }
    }
    let m2 = pairwise_sum(&a2);
    let m4 = pairwise_sum(&a4);
    let c = cnt as f64;
    let adj = 3.0 * (c - 1.0) * (c - 1.0) / ((c - 2.0) * (c - 3.0));
    let numerator = zero_out_fperr(c * (c + 1.0) * (c - 1.0) * m4);
    let denominator = zero_out_fperr((c - 2.0) * (c - 3.0) * m2 * m2);
    if denominator == 0.0 {
        return 0.0;
    }
    numerator / denominator - adj
}

/// Series.quantile(q)：pandas 走 np.percentile(..., method='linear')。
/// 虚拟下标 (n-1)*q；插值照 numpy 的 _lerp（t>=0.5 走 b-(b-a)*(1-t)）。
fn nan_quantile(v: &[f64], q: f64) -> f64 {
    let mut ok: Vec<f64> = v.iter().copied().filter(|&x| is_ok(x)).collect();
    if ok.is_empty() {
        return f64::NAN;
    }
    ok.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = ok.len();
    let virt = (n as f64 - 1.0) * q;
    let prev_f = virt.floor();
    let mut pi = prev_f as i64;
    let mut ni = pi + 1;
    if virt >= n as f64 - 1.0 {
        pi = -1;
        ni = -1;
    }
    if virt < 0.0 {
        pi = 0;
        ni = 0;
    }
    let idx = |i: i64| -> usize {
        if i < 0 {
            (n as i64 + i) as usize
        } else {
            i as usize
        }
    };
    let pv = ok[idx(pi)];
    let nv = ok[idx(ni)];
    let gamma = virt - prev_f;
    let diff = nv - pv;
    let mut res = pv + diff * gamma;
    if gamma >= 0.5 {
        res = nv - diff * (1.0 - gamma);
    }
    res
}

/// pandas 的 Welford 在线 Pearson（Series.corr / DataFrame.corr 走这条）。
/// 见 hm013.rs 的同名实现；divisor == 0 时给 NaN。
fn pearson_welford(x: &[f64], y: &[f64]) -> f64 {
    let mut nobs = 0.0f64;
    let (mut mx, mut my) = (0.0f64, 0.0f64);
    let (mut sxx, mut syy, mut sxy) = (0.0f64, 0.0f64, 0.0f64);
    for i in 0..x.len().min(y.len()) {
        if !x[i].is_finite() || !y[i].is_finite() {
            continue;
        }
        let vx = x[i];
        let vy = y[i];
        nobs += 1.0;
        let dx = vx - mx;
        let dy = vy - my;
        mx += 1.0 / nobs * dx;
        my += 1.0 / nobs * dy;
        sxx += (vx - mx) * dx;
        syy += (vy - my) * dy;
        sxy += (vx - mx) * dy;
    }
    if nobs < 1.0 {
        return f64::NAN;
    }
    let divisor = (sxx * syy).sqrt();
    if divisor != 0.0 {
        sxy / divisor
    } else {
        f64::NAN
    }
}

/// rp.corrwith 那条：先求均值、再按 32 个一批分块累加。
/// 照抄 src/statistics/mod.rs 的 calculate_correlation_optimized。
fn pearson_corrwith(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len().min(y.len());
    if n == 0 {
        return f64::NAN;
    }
    let mut count = 0usize;
    let (mut sum_x, mut sum_y) = (0.0f64, 0.0f64);
    let mut pairs: Vec<(f64, f64)> = Vec::with_capacity(n);
    for i in 0..n {
        if x[i].is_nan() || y[i].is_nan() {
            continue;
        }
        pairs.push((x[i], y[i]));
        sum_x += x[i];
        sum_y += y[i];
        count += 1;
    }
    if count < 2 {
        return f64::NAN;
    }
    let mean_x = sum_x / count as f64;
    let mean_y = sum_y / count as f64;
    let (mut cov, mut var_x, mut var_y) = (0.0f64, 0.0f64, 0.0f64);
    for chunk in pairs.chunks(32) {
        let (mut lc, mut lx, mut ly) = (0.0f64, 0.0f64, 0.0f64);
        for &(xi, yi) in chunk.iter() {
            let dx = xi - mean_x;
            let dy = yi - mean_y;
            lc += dx * dy;
            lx += dx * dx;
            ly += dy * dy;
        }
        cov += lc;
        var_x += lx;
        var_y += ly;
    }
    if var_x.abs() < f64::EPSILON || var_y.abs() < f64::EPSILON {
        return f64::NAN;
    }
    cov / (var_x.sqrt() * var_y.sqrt())
}

/// Kahan 补偿求和（fast_correlation_v2 里算列均值用的那个）
fn kahan_sum(v: &[f64]) -> f64 {
    let mut sum = 0.0f64;
    let mut comp = 0.0f64;
    for &x in v.iter() {
        let y = x - comp;
        let t = sum + y;
        comp = (t - sum) - y;
        sum = t;
    }
    sum
}

/// rp.fast_correlation_matrix_v2_df(df, max_workers=1)[i,j]。
/// 本系列传进去的都是 trend 出来的序列（永远有限），所以走 clean 分支。
fn pearson_fast(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len().min(y.len());
    let mut px: Vec<f64> = Vec::with_capacity(n);
    let mut py: Vec<f64> = Vec::with_capacity(n);
    for i in 0..n {
        if !x[i].is_nan() {
            px.push(x[i]);
        }
        if !y[i].is_nan() {
            py.push(y[i]);
        }
    }
    if px.len() == n && py.len() == n {
        // clean 分支：8 路展开的顺序累加
        if n < 2 {
            return f64::NAN;
        }
        let mi = kahan_sum(&px) / n as f64;
        let mj = kahan_sum(&py) / n as f64;
        let (mut cov, mut vi, mut vj) = (0.0f64, 0.0f64, 0.0f64);
        let chunks = n / 8;
        let rem = n % 8;
        for c in 0..chunks {
            let base = c * 8;
            for k in 0..8 {
                let idx = base + k;
                let a = px[idx] - mi;
                let b = py[idx] - mj;
                cov += a * b;
                vi += a * a;
                vj += b * b;
            }
        }
        for k in 0..rem {
            let idx = chunks * 8 + k;
            let a = px[idx] - mi;
            let b = py[idx] - mj;
            cov += a * b;
            vi += a * a;
            vj += b * b;
        }
        if vi.abs() < f64::EPSILON || vj.abs() < f64::EPSILON {
            return f64::NAN;
        }
        return cov / (vi.sqrt() * vj.sqrt());
    }
    // 含 NaN 的分支：共同有效观测、Kahan 求均值、顺序累加（本系列走不到，保底用）
    let mut pairs: Vec<(f64, f64)> = Vec::with_capacity(n);
    for i in 0..n {
        if !x[i].is_nan() && !y[i].is_nan() {
            pairs.push((x[i], y[i]));
        }
    }
    if pairs.len() < 2 {
        return f64::NAN;
    }
    let xs: Vec<f64> = pairs.iter().map(|p| p.0).collect();
    let ys: Vec<f64> = pairs.iter().map(|p| p.1).collect();
    let mi = kahan_sum(&xs) / pairs.len() as f64;
    let mj = kahan_sum(&ys) / pairs.len() as f64;
    let (mut cov, mut vi, mut vj) = (0.0f64, 0.0f64, 0.0f64);
    for &(a, b) in pairs.iter() {
        let da = a - mi;
        let db = b - mj;
        cov += da * db;
        vi += da * da;
        vj += db * db;
    }
    if vi.abs() < f64::EPSILON || vj.abs() < f64::EPSILON {
        return f64::NAN;
    }
    cov / (vi.sqrt() * vj.sqrt())
}

/// rp.trend_2d 的一维版本：值与 1 基位置的 Pearson；非有限值丢掉；
/// 样本 < 2 或任一侧方差为 0 给 0.0（不是 NaN）。
fn trend_1d(v: &[f64]) -> f64 {
    let mut values: Vec<f64> = Vec::new();
    let mut idx: Vec<f64> = Vec::new();
    for (i, &x) in v.iter().enumerate() {
        if x.is_finite() {
            values.push(x);
            idx.push((i + 1) as f64);
        }
    }
    let n = values.len();
    if n < 2 {
        return 0.0;
    }
    let mean_x = values.iter().sum::<f64>() / n as f64;
    let mean_y = idx.iter().sum::<f64>() / n as f64;
    let (mut cov, mut vx, mut vy) = (0.0f64, 0.0f64, 0.0f64);
    for i in 0..n {
        let dx = values[i] - mean_x;
        let dy = idx[i] - mean_y;
        cov += dx * dy;
        vx += dx * dx;
        vy += dy * dy;
    }
    if vx == 0.0 || vy == 0.0 {
        return 0.0;
    }
    cov / (vx.sqrt() * vy.sqrt())
}

// ============================================================================
// pandas 的 Rolling.std（roll_var 的逐行复刻）
// ============================================================================

/// add_var：pandas/_libs/window/aggregations.pyx 里的 add_var（带 Kahan 补偿）
#[allow(clippy::too_many_arguments)]
fn add_var(
    val: f64,
    nobs: &mut f64,
    mean_x: &mut f64,
    ssqdm_x: &mut f64,
    comp: &mut f64,
    num_same: &mut i64,
    prev_value: &mut f64,
) {
    if val.is_nan() {
        return;
    }
    *nobs += 1.0;
    if val == *prev_value {
        *num_same += 1;
    } else {
        *num_same = 1;
    }
    *prev_value = val;

    // Welford + Kahan：prev_mean 与 y 都用**更新前**的补偿量
    let prev_mean = *mean_x - *comp;
    let y = val - *comp;
    let t = y - *mean_x;
    *comp = t + *mean_x - y;
    let delta = t;
    if *nobs != 0.0 {
        *mean_x = *mean_x + delta / *nobs;
    } else {
        *mean_x = 0.0;
    }
    *ssqdm_x = *ssqdm_x + (val - prev_mean) * (val - *mean_x);
}

/// remove_var：同上的 remove 版本（用的是**另一个**补偿量 compensation_remove）
fn remove_var(val: f64, nobs: &mut f64, mean_x: &mut f64, ssqdm_x: &mut f64, comp: &mut f64) {
    if val.is_nan() {
        return;
    }
    *nobs -= 1.0;
    if *nobs != 0.0 {
        let prev_mean = *mean_x - *comp;
        let y = val - *comp;
        let t = y - *mean_x;
        *comp = t + *mean_x - y;
        let delta = t;
        *mean_x = *mean_x - delta / *nobs;
        *ssqdm_x = *ssqdm_x - (val - prev_mean) * (val - *mean_x);
    } else {
        *mean_x = 0.0;
        *ssqdm_x = 0.0;
    }
}

/// calc_var：窗口里全是同一个值（num_consecutive_same_value >= nobs）时给 0，不是 0±舍入
fn calc_var(minp: i64, ddof: i64, nobs: f64, ssqdm_x: f64, num_same: i64) -> f64 {
    if nobs >= minp as f64 && nobs > ddof as f64 {
        if nobs == 1.0 || num_same as f64 >= nobs {
            0.0
        } else {
            ssqdm_x / (nobs - ddof as f64)
        }
    } else {
        f64::NAN
    }
}

/// Series.rolling(window).std() 的逐行复刻（pandas 的 roll_var + 外层 np.sqrt）。
///
/// 为什么必须逐行复刻而不能用两遍法：这个量只差最后一两位，但它要拿去做回归的自变量，
/// 残差再拿去做 rank。实测 000001@20250303 上两遍法与 pandas 的 volatility 有 4310/4719
/// 个值差 1 ulp，直接让 ask_vol3 的残差名次变了 171 行（最大 1.3e-3），
/// rank_diff 那一族全部超 1e-4；换成这份复刻后逐位相同。
///
/// 结构照抄 roll_var：固定窗口下 start/end 都单调，所以走的是「先按 start 的推进删、
/// 再按 end 的推进加」那条分支，状态（nobs/mean_x/ssqdm_x/两个补偿量）一直留着。
fn rolling_std(values: &[f64], window: usize, minp: i64, ddof: i64) -> Vec<f64> {
    let n = values.len();
    let mut out = vec![f64::NAN; n];
    if n == 0 {
        return out;
    }
    let mut nobs = 0.0f64;
    let mut mean_x = 0.0f64;
    let mut ssqdm_x = 0.0f64;
    let mut comp_add = 0.0f64;
    let mut comp_rem = 0.0f64;
    let mut num_same: i64 = 0;
    let mut prev_value = 0.0f64;
    let (mut prev_s, mut prev_e) = (0usize, 0usize);

    for i in 0..n {
        let s = (i + 1).saturating_sub(window);
        let e = i + 1;
        if i == 0 || s >= prev_e {
            prev_value = values[s];
            num_same = 0;
            nobs = 0.0;
            mean_x = 0.0;
            ssqdm_x = 0.0;
            comp_add = 0.0;
            comp_rem = 0.0;
            for j in s..e {
                add_var(
                    values[j],
                    &mut nobs,
                    &mut mean_x,
                    &mut ssqdm_x,
                    &mut comp_add,
                    &mut num_same,
                    &mut prev_value,
                );
            }
        } else {
            for j in prev_s..s {
                remove_var(values[j], &mut nobs, &mut mean_x, &mut ssqdm_x, &mut comp_rem);
            }
            for j in prev_e..e {
                add_var(
                    values[j],
                    &mut nobs,
                    &mut mean_x,
                    &mut ssqdm_x,
                    &mut comp_add,
                    &mut num_same,
                    &mut prev_value,
                );
            }
        }
        let v = calc_var(minp, ddof, nobs, ssqdm_x, num_same);
        // 外层是 np.sqrt：负数给 NaN，0 给 0
        out[i] = if v.is_nan() { f64::NAN } else { v.sqrt() };
        prev_s = s;
        prev_e = e;
    }
    out
}

/// rp.pandas_series_rank(x)：average、升序、NaN 保持（相等判据 |a-b| < 1e-15）
fn rank_avg(v: &[f64]) -> Vec<f64> {
    let n = v.len();
    let mut pairs: Vec<(f64, usize)> = Vec::with_capacity(n);
    for (i, &x) in v.iter().enumerate() {
        if !x.is_nan() {
            pairs.push((x, i));
        }
    }
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let mut out = vec![f64::NAN; n];
    let mut i = 0usize;
    while i < pairs.len() {
        let cur = pairs[i].0;
        let mut j = i;
        while j < pairs.len() && (pairs[j].0 - cur).abs() < 1e-15 {
            j += 1;
        }
        let avg = ((i + 1) + j) as f64 / 2.0;
        for k in i..j {
            out[pairs[k].1] = avg;
        }
        i = j;
    }
    out
}

/// rank / rank.max()（pandas 的 max 跳过 NaN；全 NaN 时 max 是 NaN）
fn rank_norm(v: &[f64]) -> Vec<f64> {
    let r = rank_avg(v);
    let mut mx = f64::NAN;
    for &x in r.iter() {
        if x.is_nan() {
            continue;
        }
        if mx.is_nan() || x > mx {
            mx = x;
        }
    }
    r.iter().map(|&x| x / mx).collect()
}

/// rp.ols_residuals(X=[x], y)：带截距的一元回归残差。
/// 样本过滤只按 is_nan（inf 也算有效样本，和旧代码一致）。
///
/// ！！正规方程与预测值都必须用 ndarray 的 `.dot()`，不能自己写顺序求和：
/// 旧代码的残差要拿去做 rank，而残差的并列（同一份盘口快照重复出现时
/// resi 会逐位相等）决定了平均名次。自己顺序求和的 X'X 与 ndarray 用的
/// matrixmultiply 差在末位，实测 000001@20250303 上 resi 只差 1e-8 绝对值
/// （相对 1e-13），却让 74 行的名次变了，rank_diff 那一族直接超 1e-4。
/// 这里照抄 src/statistics/mod.rs 的写法（同一份 ndarray + solve_linear_system3），
/// 两条路径就是同一段代码，逐位一致。
fn ols_residual(x: &[f64], y: &[f64]) -> Vec<f64> {
    let n = x.len().min(y.len());
    let mut out = vec![f64::NAN; n];
    let mut idx: Vec<usize> = Vec::new();
    for i in 0..n {
        if x[i].is_nan() || y[i].is_nan() {
            continue;
        }
        idx.push(i);
    }
    if idx.is_empty() {
        return out;
    }
    let m = idx.len();
    // 设计矩阵 [1, x]
    let mut design = Array2::<f64>::ones((m, 2));
    let mut yv = Array1::<f64>::zeros(m);
    for (row, &i) in idx.iter().enumerate() {
        design[[row, 1]] = x[i];
        yv[row] = y[i];
    }
    let xt_x = design.t().dot(&design);
    let xt_y = design.t().dot(&yv);
    let coeff = crate::statistics::solve_linear_system3(&xt_x.view(), &xt_y.view());
    let y_pred = design.dot(&coeff);
    for (row, &i) in idx.iter().enumerate() {
        out[i] = y[i] - y_pred[row];
    }
    out
}

/// get_features_factors 每列的 12 个统计量（kind 对应 STATS12 的下标）
fn stat_value(kind: usize, v: &[f64]) -> f64 {
    match kind {
        0 => nan_mean(v),
        1 => nan_mean(v).abs(),
        2 => nan_std(v),
        3 => nan_std(v).abs(),
        4 => nan_skew(v),
        5 => nan_skew(v).abs(),
        6 => nan_kurt(v),
        7 => nan_kurt(v).abs(),
        8 => {
            if v.len() < 2 {
                f64::NAN
            } else {
                pearson_corrwith(&v[1..], &v[..v.len() - 1])
            }
        }
        9 => {
            if v.len() < 2 {
                f64::NAN
            } else {
                pearson_corrwith(&v[1..], &v[..v.len() - 1]).abs()
            }
        }
        10 => trend_1d(v),
        11 => trend_1d(v).abs(),
        _ => unreachable!(),
    }
}

/// get_features_factors(DataFrame, with_abs=True)：12*k 个统计量 + k*(k-1) 个成对相关
fn stats_block(cols: &[&[f64]]) -> Vec<f64> {
    let k = cols.len();
    let mut out: Vec<f64> = Vec::with_capacity(12 * k + k * (k - 1));
    for kind in 0..12 {
        for c in cols.iter() {
            out.push(stat_value(kind, c));
        }
    }
    for i in 0..k {
        for j in (i + 1)..k {
            out.push(pearson_fast(cols[i], cols[j]));
        }
    }
    for i in 0..k {
        for j in (i + 1)..k {
            out.push(pearson_fast(cols[i], cols[j]).abs());
        }
    }
    out
}

// ============================================================================
// CSV 读取
// ============================================================================

#[inline]
fn split_into<'a>(bytes: &'a [u8], buf: &mut [&'a [u8]; MAX_COLS]) -> usize {
    let mut start = 0usize;
    let mut n = 0usize;
    for (i, &b) in bytes.iter().enumerate() {
        if b == b',' {
            if n < MAX_COLS {
                buf[n] = &bytes[start..i];
            }
            n += 1;
            start = i + 1;
        }
    }
    if n < MAX_COLS {
        buf[n] = &bytes[start..];
    }
    (n + 1).min(MAX_COLS)
}

/// 空单元格按 pandas 的读法给 NaN
#[inline]
fn parse_f64(b: &[u8]) -> f64 {
    if b.is_empty() {
        return f64::NAN;
    }
    std::str::from_utf8(b)
        .ok()
        .and_then(|s| s.trim().parse::<f64>().ok())
        .unwrap_or(f64::NAN)
}

/// 一行盘口。只留计算要用的列。
#[derive(Clone, Copy)]
struct Row {
    last_prc: f64,
    volume_raw: f64,
    v: [f64; 8],
}

/// 复刻到 adjust_afternoon 为止：读出 [09:30:00, 11:30:00] ∪ [13:00:00, 14:57:00] 的行。
/// 行序 = 上午（文件序）在前、下午（文件序）在后，和 pd.concat([df1, df2]) 一致。
fn read_rows(code: &str, date: i64) -> Option<Vec<Row>> {
    let filename = format!("{}_{}_market_data.csv", code, date);
    let mut path = None;
    for root in crate::data_paths::level2_roots() {
        let p = root
            .join(date.to_string())
            .join("market_data")
            .join(&filename);
        if p.exists() {
            path = Some(p);
            break;
        }
    }
    let path = path?;

    let mut content = String::new();
    File::open(&path).ok()?.read_to_string(&mut content).ok()?;

    let mut morning: Vec<Row> = Vec::new();
    let mut afternoon: Vec<Row> = Vec::new();
    let mut lines = content.lines();
    lines.next(); // 表头
    for raw in lines {
        let line = raw.trim_end_matches('\r');
        if line.is_empty() {
            continue;
        }
        let mut f: [&[u8]; MAX_COLS] = [&[][..]; MAX_COLS];
        let nf = split_into(line.as_bytes(), &mut f);
        if nf <= COL_BID_VOL5 {
            continue;
        }
        let ex = parse_f64(f[COL_EXCHTIME]);
        if !ex.is_finite() {
            continue;
        }
        // deal_data：to_timedelta(exchtime/1e6, unit='s') + 1970-01-01 08:00:00
        // 先除 1e6 再乘 1e9 再四舍五入，和 pandas 的运算次序一致
        let ns = (ex / 1_000_000.0 * NS_PER_SEC as f64).round();
        if !ns.is_finite() {
            continue;
        }
        let t = ns as i64 + OFFSET_8H_NS;
        let day = t.rem_euclid(DAY_NS);
        let dst = if (LD_0930..=LD_1130).contains(&day) {
            &mut morning
        } else if (LD_1300..=LD_1457).contains(&day) {
            &mut afternoon
        } else {
            continue;
        };
        let mut v = [0.0f64; 8];
        for (k, &ci) in SERIES_COLS.iter().enumerate() {
            v[k] = parse_f64(f[ci]);
        }
        dst.push(Row {
            last_prc: parse_f64(f[COL_LAST_PRC]),
            volume_raw: parse_f64(f[COL_VOLUME]),
            v,
        });
    }

    let mut rows = morning;
    rows.extend(afternoon);
    if rows.is_empty() {
        return None;
    }
    Some(rows)
}

// ============================================================================
// 单股单日
// ============================================================================

/// 一列的 `one()` 块：12 个基础量 + 26 个 twins 量
fn one_block(
    col: &[f64],
    col_short: &[f64],
    volatility: &[f64],
    volume: &[f64],
    twins_ask: &[f64],
    twins_bid: &[f64],
) -> Vec<f64> {
    let q80 = nan_quantile(col, 0.8);
    let q20 = nan_quantile(col, 0.2);

    let mut li: Vec<usize> = Vec::new();
    let mut si: Vec<usize> = Vec::new();
    for i in 0..col.len() {
        if col[i] >= q80 {
            li.push(i);
        }
        if col[i] <= q20 {
            si.push(i);
        }
    }

    let volume_sum = nan_sum(volume);
    let order_mean = nan_mean(col_short);
    let volatility_mean = nan_mean(volatility);
    let corr_total = pearson_welford(col_short, volatility);

    let gather = |idx: &[usize], src: &[f64]| -> Vec<f64> {
        idx.iter().map(|&i| src[i]).collect()
    };
    let lv = gather(&li, volume);
    let sv = gather(&si, volume);
    let lo = gather(&li, col_short);
    let so = gather(&si, col_short);
    let lvol = gather(&li, volatility);
    let svol = gather(&si, volatility);

    let large_volume_rate = nan_sum(&lv) / volume_sum;
    let small_volume_rate = nan_sum(&sv) / volume_sum;
    let large_order_rate = nan_mean(&lo) / order_mean;
    let small_order_rate = nan_mean(&so) / order_mean;
    let large_volatility_mean = nan_mean(&lvol);
    let small_volatility_mean = nan_mean(&svol);
    let large_volatility_rate = large_volatility_mean / volatility_mean;
    let small_volatility_rate = small_volatility_mean / volatility_mean;
    let large_corr = pearson_welford(&lo, &lvol);
    let small_corr = pearson_welford(&so, &svol);

    let mut out: Vec<f64> = Vec::with_capacity(N_PER_COL);
    out.push(large_volume_rate);
    out.push(small_volume_rate);
    out.push(large_order_rate);
    out.push(small_order_rate);
    out.push(large_volatility_mean);
    out.push(small_volatility_mean);
    out.push(large_volatility_rate);
    out.push(small_volatility_rate);
    out.push(large_corr);
    out.push(small_corr);
    out.push(large_corr - corr_total);
    out.push(small_corr - corr_total);
    out.extend(stats_block(&[twins_ask, twins_bid]));
    debug_assert_eq!(out.len(), N_PER_COL);
    out
}

/// 逐股的一天：算不出来返回 None（上游补 NaN，保证因子个数恒定）。
pub fn compute_stock(code: &str, date: i64) -> Option<Vec<f32>> {
    let rows = read_rows(code, date)?;
    let n = rows.len();

    let mut last_prc = vec![0.0f64; n];
    let mut vol_raw = vec![0.0f64; n];
    let mut vcols: Vec<Vec<f64>> = (0..8).map(|_| vec![0.0f64; n]).collect();
    for (i, r) in rows.iter().enumerate() {
        last_prc[i] = r.last_prc;
        vol_raw[i] = r.volume_raw;
        for k in 0..8 {
            vcols[k][i] = r.v[k];
        }
    }

    // pct_change + rolling(20).std()（ddof=1、min_periods=20）
    let mut pc = vec![f64::NAN; n];
    for i in 1..n {
        pc[i] = last_prc[i] / last_prc[i - 1] - 1.0;
    }
    let volatility = rolling_std(&pc, 20, 20, 1);

    // volume 差分
    let mut volume = vec![f64::NAN; n];
    for i in 1..n {
        volume[i] = vol_raw[i] - vol_raw[i - 1];
    }

    // 8 个量各自的派生序列
    let mut corrs = vec![f64::NAN; 8];
    // 每个量的 5 个变体：[resi, resi_abs, resi_adjusted, resi_rank_diff, resi_adjusted_rank_diff]
    let mut variants: Vec<[Vec<f64>; 5]> = Vec::with_capacity(8);
    for j in 0..8 {
        let col = &vcols[j];
        let corr = pearson_welford(col, &volatility);
        corrs[j] = corr;
        let sign = if corr >= 0.0 { 1.0 } else { -1.0 };
        let resi = ols_residual(&volatility, col);
        let resi_abs: Vec<f64> = resi.iter().map(|x| x.abs()).collect();
        let resi_adj: Vec<f64> = resi.iter().map(|x| x * sign).collect();
        let vol_rank = rank_norm(col);
        let resi_rank = rank_norm(&resi);
        let adj_rank = rank_norm(&resi_adj);
        let rd: Vec<f64> = (0..n).map(|i| resi_rank[i] - vol_rank[i]).collect();
        let ard: Vec<f64> = (0..n).map(|i| adj_rank[i] - vol_rank[i]).collect();
        variants.push([resi, resi_abs, resi_adj, rd, ard]);
    }

    // 行方向 trend：ask 的 4 个 + bid 的 4 个，各一组（原始 / adjusted）
    let mut tn_ask = vec![0.0f64; n];
    let mut tn_bid = vec![0.0f64; n];
    let mut ta_ask = vec![0.0f64; n];
    let mut ta_bid = vec![0.0f64; n];
    let mut buf = [0.0f64; 4];
    for i in 0..n {
        for k in 0..4 {
            buf[k] = variants[k][0][i];
        }
        tn_ask[i] = trend_1d(&buf);
        for k in 0..4 {
            buf[k] = variants[k + 4][0][i];
        }
        tn_bid[i] = trend_1d(&buf);
        for k in 0..4 {
            buf[k] = variants[k][2][i];
        }
        ta_ask[i] = trend_1d(&buf);
        for k in 0..4 {
            buf[k] = variants[k + 4][2][i];
        }
        ta_bid[i] = trend_1d(&buf);
    }

    let mut out: Vec<f64> = Vec::with_capacity(N_OUT);
    out.extend(corrs.iter().copied());
    out.extend(stats_block(&[&tn_ask[..], &tn_bid[..], &ta_ask[..], &ta_bid[..]]));

    for level in 0..4 {
        for side in 0..2 {
            let j = if side == 0 { level } else { level + 4 };
            let col_short = &vcols[j];
            for vi in 0..5 {
                let col = &variants[j][vi];
                let use_adj = vi == 2 || vi == 4;
                let (tw_a, tw_b) = if use_adj {
                    (&ta_ask, &ta_bid)
                } else {
                    (&tn_ask, &tn_bid)
                };
                out.extend(one_block(
                    col,
                    col_short,
                    &volatility,
                    &volume,
                    tw_a,
                    tw_b,
                ));
            }
        }
    }
    debug_assert_eq!(out.len(), N_OUT);

    Some(out.into_iter().map(|x| x as f32).collect())
}

/// hm73 全是逐股因子，没有横截面版本。
pub fn compute_cross(_date: i64) -> Option<(Vec<String>, Vec<f32>)> {
    None
}
