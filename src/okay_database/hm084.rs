//! hm84 雪国列车 —— 旧框架（design_whatever_out）的 Rust 复现。
//!
//! 旧代码（/home/chenzongwei/pythoncode/成果hm84_雪国列车/hm84_okay.py 的 go()）要点：
//!
//!   PARAM=15（15 秒一档）；PARAM_LESS=0.9；PARAM_BIG_POS=0.7；PARAM_BIG_NEG=-0.3；
//!   PARAM_DECAY_THRESH=0.5
//!
//! 1. df = adjust_afternoon(read_market(code,date).set_index(['exchtime'])).resample('15s').last()
//!    · read_market 默认 with_high_low_limited=0 → 丢掉 ask_prc1==0 或 bid_prc1==0 的快照
//!    · adjust_afternoon 保留 [09:30:00,11:30:00] 与 [13:00:00,14:57:00]，下午整体前移 90 分钟
//!    · resample('15s') 的档 = floor(时间 / 15s)，15s 整除一天，所以与 epoch 对齐一致
//!    · .last() 是**逐列**取该档内最后一个非 NaN
//! 2. df.volume = df.volume.diff().fillna(0)
//! 3. df['local_volume'] = df.volume / df.volume.rolling(60).mean()   ← 用的是**未归一化**的 volume
//!    （local_volume 排在 df 列末，所以 lims_cols 的最后一项是它）
//! 4. lims_cols = 含 '_vol' 且不含 'total' 的列 = ask_vol1,bid_vol1,...,ask_vol10,bid_vol10,local_volume（21 行）
//!    lims = df[lims_cols].T / 100   → 21 × T
//!    lims_corrs = correlation_matrix_df(lims) → 对 lims 的**列**求相关，得到 T × T
//!    （rp.correlation_matrix_df 就是 fast_correlation_matrix_v2_df，21 个样本按列求 Pearson）
//! 5. df.volume = df.volume / df.volume.sum()
//! 6. trade = adjust_afternoon(read_trade(code,date).set_index(['exchtime']))（read_trade 默认丢 flag==32）
//!    trade_66 / trade_83 = 按 flag 过滤后 resample('15s').volume.sum().fillna(0)
//!    rate66_3s_past = trade_66 / (trade_66+trade_83).replace(inf,1).fillna(0)
//!      ！！注意方法链只作用在**分母**上，结果本身不 fillna：分母为 0 时结果是 NaN
//!    rate66_15s_past / rate66_60s_past：先 rolling(5)/rolling(20).sum() 再相除，同上
//!    *_future = *_past.shift(-1) / shift(-5) / shift(-20)
//! 7. morph_decay_steps：lims_corrs 上三角（k=1）里第一处 < 0.5 的**绝对列号**（找不到给 NaN）
//!    rate66_at_decay：列号 = clip(i + steps_i, 0, T-1)，取 rate66_3s_past 在该列时间标签上的值
//! 8. volume_3s_past = df.volume.resample('15s').sum().fillna(0)（已在 15s 网格上，等于原值）
//!    local_volume_3s_past 同理（NaN 被 sum 成 0）
//! 9. 对 20 个序列（rate66/local_volume/volume × 3s/15s/60s × past/future，加 morph_decay_steps、
//!    rate66_at_decay）各跑一次 corrwith_diff，每次产出 10 列：
//!      corr_full / corr_less / corr_big_pos / corr_big_neg / mean_big_pos / mean_big_neg /
//!      big_pos_early / big_pos_late / big_neg_early / big_neg_late
//!    · diff_matrix[i][j] = |price_i - price_j|，索引/列都是该序列的时间标签
//!    · corr_full  = corrwith(diff_matrix, lims_corrs)          （逐列 Pearson，成对完整观测）
//!    · corr_less  = corrwith(diff_matrix, lims_corrs_less)     （lims_corrs >= 0.9 处为 NaN）
//!    · big_pos    = lims_corrs.where(0.7 < c < 0.9)，big_neg 同（c < -0.3）
//!    · diff_big_pos = big_pos * diff_matrix；corr_big_pos = corrwith(diff_big_pos, lims_corrs)
//!    · mean_big_pos = diff_big_pos.mean()（逐列 skipna 均值）
//!    · big_pos_stack = big_pos.stack().sort_values(ascending=False)...head(100)
//!      big_neg_stack = big_neg.stack().sort_values(ascending=False)...tail(100)
//!      → 取 early_time=min(两列时间)、late_time=max(...)，再从 price 里按标签取出这些时刻的值
//! 10. res（T × 200）交给 rp.get_features_factors(res, with_corr=False,
//!     with_threshold_counts=False, with_lyapunov_exponent=False)，每列 19 个特征：
//!     mean median std skew kurt p5 p25 p75 p95 iqr cv autocorr1 autocorr1_abs trend
//!     period_diff period_ratio lz_complexity entropy_1d max_range_product
//! 11. 最后按 names_save 取 69 个值返回（顺序即 names()）。
//!
//! 已知的取舍（都实测过，见对账结论）：
//! · 求和顺序：pandas 的 mean/std 走 numpy 成对求和，本文件用 f64 顺序累加。948 个数的相对
//!   误差量级 1e-13，远低于 1e-4 验收线，只在「均值本身接近 0」的比值型特征上放大，实测未越线。
//! · big_pos/big_neg 的 stack→sort_values 并列顺序：pandas 是 argsort(kind='quicksort')[::-1]，
//!   本文件用「按值稳定排序再反转」（即 top100 用值降序 + 原始位置降序，bottom100 用值升序 +
//!   原始位置升序）。矩阵是对称的，每个值天然成对出现，实测两者的时间集合一致。
//! · 时间只为分档用，不比较纳秒：pandas 的 exchtime→Timestamp 有 ±256ns 噪声，实测本样本上
//!   分档结果与「精确微秒分档」完全一致。

use std::fs::File;
use std::io::Read;

// ============================================================================
// 常量与输出名表
// ============================================================================

const N_SERIES: usize = 20;
const N_METRICS: usize = 10;
const N_COLS: usize = N_SERIES * N_METRICS; // 200
const N_FEAT: usize = 19;
const N_OUT: usize = 69;

/// 一档 = 15 秒（微秒）
const BAR_US: i64 = 15_000_000;
const DAY_US: i64 = 86_400_000_000;
/// 东八区偏移（微秒）：CSV 的 exchtime 是 UTC epoch 微秒
const EXCH_OFFSET_US: i64 = 8 * 3600 * 1_000_000;
/// 下午时段前移 90 分钟
const LUNCH_US: i64 = 90 * 60 * 1_000_000;

const MORNING_START_US: i64 = (9 * 3600 + 30 * 60) * 1_000_000;
const MORNING_END_US: i64 = (11 * 3600 + 30 * 60) * 1_000_000;
const AFTERNOON_START_US: i64 = (13 * 3600) * 1_000_000;
const AFTERNOON_END_US: i64 = (14 * 3600 + 57 * 60) * 1_000_000;

const PARAM_LESS: f64 = 0.9;
const PARAM_BIG_POS: f64 = 0.7;
const PARAM_BIG_NEG: f64 = -0.3;
const PARAM_DECAY_THRESH: f64 = 0.5;

// market_data CSV 的列序号（与 fast_csv_reader 一致）
const MKT_COL_EXCHTIME: usize = 4;
const MKT_COL_VOLUME: usize = 14;
const MKT_COL_ASK_PRC_BASE: usize = 21; // ask_prc1=21, ask_vol1=22, bid_prc1=23, bid_vol1=24, ...

// 19 个特征的后缀顺序（= get_features_names 的顺序）
const FEAT_SUFFIX: [&str; N_FEAT] = [
    "mean",
    "median",
    "std",
    "skew",
    "kurt",
    "p5",
    "p25",
    "p75",
    "p95",
    "iqr",
    "cv",
    "autocorr1",
    "autocorr1_abs",
    "trend",
    "period_diff",
    "period_ratio",
    "lz_complexity",
    "entropy_1d",
    "max_range_product",
];

pub static NAMES: [&str; N_OUT] = [
    "big_neg_early_rate66_15s_future_p75",
    "big_neg_early_rate66_3s_future_entropy_1d",
    "big_neg_early_rate66_3s_future_median",
    "big_neg_early_rate66_3s_future_p75",
    "big_neg_early_rate66_60s_future_p25",
    "big_neg_early_rate66_60s_future_p75",
    "big_neg_early_rate66_at_decay_median",
    "big_neg_late_rate66_15s_future_p75",
    "big_neg_late_volume_60s_future_median",
    "big_pos_early_morph_decay_steps_mean",
    "big_pos_early_rate66_15s_past_median",
    "big_pos_early_rate66_3s_future_iqr",
    "big_pos_early_rate66_3s_future_skew",
    "big_pos_early_rate66_60s_future_p95",
    "big_pos_early_rate66_60s_past_p75",
    "big_pos_early_volume_60s_future_median",
    "big_pos_late_rate66_3s_future_max_range_product",
    "big_pos_late_rate66_3s_past_entropy_1d",
    "big_pos_late_rate66_3s_past_mean",
    "big_pos_late_volume_3s_future_trend",
    "corr_big_pos_local_volume_15s_future_autocorr1_abs",
    "corr_big_pos_morph_decay_steps_autocorr1",
    "corr_big_pos_rate66_3s_future_entropy_1d",
    "corr_big_pos_rate66_at_decay_entropy_1d",
    "corr_big_pos_volume_15s_future_median",
    "corr_big_pos_volume_3s_future_p25",
    "corr_full_local_volume_15s_past_cv",
    "corr_full_local_volume_3s_past_cv",
    "corr_full_local_volume_3s_past_iqr",
    "corr_full_local_volume_3s_past_std",
    "corr_full_volume_15s_future_autocorr1_abs",
    "corr_less_local_volume_3s_future_iqr",
    "corr_less_local_volume_3s_past_iqr",
    "corr_less_local_volume_3s_past_p5",
    "corr_less_rate66_60s_future_iqr",
    "corr_less_rate66_60s_future_p95",
    "corr_less_volume_3s_past_period_ratio",
    "mean_big_neg_local_volume_60s_future_entropy_1d",
    "mean_big_neg_rate66_15s_future_median",
    "mean_big_neg_rate66_15s_future_p25",
    "mean_big_neg_rate66_3s_past_iqr",
    "mean_big_neg_rate66_3s_past_median",
    "mean_big_neg_rate66_3s_past_p75",
    "mean_big_neg_volume_3s_future_autocorr1_abs",
    "mean_big_neg_volume_3s_past_autocorr1_abs",
    "mean_big_neg_volume_60s_future_autocorr1_abs",
    "mean_big_neg_volume_60s_future_entropy_1d",
    "mean_big_neg_volume_60s_future_p95",
    "mean_big_neg_volume_60s_future_skew",
    "mean_big_pos_local_volume_15s_past_max_range_product",
    "mean_big_pos_local_volume_3s_future_entropy_1d",
    "mean_big_pos_local_volume_60s_past_entropy_1d",
    "mean_big_pos_morph_decay_steps_p25",
    "mean_big_pos_rate66_15s_future_p95",
    "mean_big_pos_rate66_15s_past_p95",
    "mean_big_pos_rate66_60s_past_entropy_1d",
    "mean_big_pos_volume_15s_future_p25",
    "mean_big_pos_volume_15s_past_p5",
    "mean_big_pos_volume_3s_future_autocorr1_abs",
    "mean_big_pos_volume_3s_future_mean",
    "mean_big_pos_volume_3s_future_p25",
    "mean_big_pos_volume_3s_future_p95",
    "mean_big_pos_volume_3s_past_autocorr1",
    "mean_big_pos_volume_3s_past_mean",
    "mean_big_pos_volume_3s_past_p25",
    "mean_big_pos_volume_60s_future_period_ratio",
    "mean_big_pos_volume_60s_future_std",
    "mean_big_pos_volume_60s_past_entropy_1d",
    "mean_big_pos_volume_60s_past_p25",
];

/// (名字, 特征序号 0..18, res 列序号 0..199)
pub static TABLE: [(&str, usize, usize); N_OUT] = [
    ("big_neg_early_rate66_15s_future_p75", 7, 38),
    ("big_neg_early_rate66_3s_future_entropy_1d", 17, 18),
    ("big_neg_early_rate66_3s_future_median", 1, 18),
    ("big_neg_early_rate66_3s_future_p75", 7, 18),
    ("big_neg_early_rate66_60s_future_p25", 6, 58),
    ("big_neg_early_rate66_60s_future_p75", 7, 58),
    ("big_neg_early_rate66_at_decay_median", 1, 198),
    ("big_neg_late_rate66_15s_future_p75", 7, 39),
    ("big_neg_late_volume_60s_future_median", 1, 119),
    ("big_pos_early_morph_decay_steps_mean", 0, 186),
    ("big_pos_early_rate66_15s_past_median", 1, 26),
    ("big_pos_early_rate66_3s_future_iqr", 9, 16),
    ("big_pos_early_rate66_3s_future_skew", 3, 16),
    ("big_pos_early_rate66_60s_future_p95", 8, 56),
    ("big_pos_early_rate66_60s_past_p75", 7, 46),
    ("big_pos_early_volume_60s_future_median", 1, 116),
    ("big_pos_late_rate66_3s_future_max_range_product", 18, 17),
    ("big_pos_late_rate66_3s_past_entropy_1d", 17, 7),
    ("big_pos_late_rate66_3s_past_mean", 0, 7),
    ("big_pos_late_volume_3s_future_trend", 13, 77),
    ("corr_big_pos_local_volume_15s_future_autocorr1_abs", 12, 152),
    ("corr_big_pos_morph_decay_steps_autocorr1", 11, 182),
    ("corr_big_pos_rate66_3s_future_entropy_1d", 17, 12),
    ("corr_big_pos_rate66_at_decay_entropy_1d", 17, 192),
    ("corr_big_pos_volume_15s_future_median", 1, 92),
    ("corr_big_pos_volume_3s_future_p25", 6, 72),
    ("corr_full_local_volume_15s_past_cv", 10, 140),
    ("corr_full_local_volume_3s_past_cv", 10, 120),
    ("corr_full_local_volume_3s_past_iqr", 9, 120),
    ("corr_full_local_volume_3s_past_std", 2, 120),
    ("corr_full_volume_15s_future_autocorr1_abs", 12, 90),
    ("corr_less_local_volume_3s_future_iqr", 9, 131),
    ("corr_less_local_volume_3s_past_iqr", 9, 121),
    ("corr_less_local_volume_3s_past_p5", 5, 121),
    ("corr_less_rate66_60s_future_iqr", 9, 51),
    ("corr_less_rate66_60s_future_p95", 8, 51),
    ("corr_less_volume_3s_past_period_ratio", 15, 61),
    ("mean_big_neg_local_volume_60s_future_entropy_1d", 17, 175),
    ("mean_big_neg_rate66_15s_future_median", 1, 35),
    ("mean_big_neg_rate66_15s_future_p25", 6, 35),
    ("mean_big_neg_rate66_3s_past_iqr", 9, 5),
    ("mean_big_neg_rate66_3s_past_median", 1, 5),
    ("mean_big_neg_rate66_3s_past_p75", 7, 5),
    ("mean_big_neg_volume_3s_future_autocorr1_abs", 12, 75),
    ("mean_big_neg_volume_3s_past_autocorr1_abs", 12, 65),
    ("mean_big_neg_volume_60s_future_autocorr1_abs", 12, 115),
    ("mean_big_neg_volume_60s_future_entropy_1d", 17, 115),
    ("mean_big_neg_volume_60s_future_p95", 8, 115),
    ("mean_big_neg_volume_60s_future_skew", 3, 115),
    ("mean_big_pos_local_volume_15s_past_max_range_product", 18, 144),
    ("mean_big_pos_local_volume_3s_future_entropy_1d", 17, 134),
    ("mean_big_pos_local_volume_60s_past_entropy_1d", 17, 164),
    ("mean_big_pos_morph_decay_steps_p25", 6, 184),
    ("mean_big_pos_rate66_15s_future_p95", 8, 34),
    ("mean_big_pos_rate66_15s_past_p95", 8, 24),
    ("mean_big_pos_rate66_60s_past_entropy_1d", 17, 44),
    ("mean_big_pos_volume_15s_future_p25", 6, 94),
    ("mean_big_pos_volume_15s_past_p5", 5, 84),
    ("mean_big_pos_volume_3s_future_autocorr1_abs", 12, 74),
    ("mean_big_pos_volume_3s_future_mean", 0, 74),
    ("mean_big_pos_volume_3s_future_p25", 6, 74),
    ("mean_big_pos_volume_3s_future_p95", 8, 74),
    ("mean_big_pos_volume_3s_past_autocorr1", 11, 64),
    ("mean_big_pos_volume_3s_past_mean", 0, 64),
    ("mean_big_pos_volume_3s_past_p25", 6, 64),
    ("mean_big_pos_volume_60s_future_period_ratio", 15, 114),
    ("mean_big_pos_volume_60s_future_std", 2, 114),
    ("mean_big_pos_volume_60s_past_entropy_1d", 17, 104),
    ("mean_big_pos_volume_60s_past_p25", 6, 104),
];


pub fn names() -> Vec<String> {
    NAMES.iter().map(|s| s.to_string()).collect()
}

pub fn compute_cross(_date: i64) -> Option<(Vec<String>, Vec<f32>)> {
    None
}

// ============================================================================
// 基础工具
// ============================================================================

#[inline]
fn bin_of(t_us: i64) -> i64 {
    t_us.div_euclid(BAR_US)
}

/// adjust_afternoon（only_inday=1）的精确微秒版本：返回平移后的时间，非盘中返回 None。
#[inline]
fn afternoon_adjust(t_us: i64) -> Option<i64> {
    let tod = t_us.rem_euclid(DAY_US);
    if tod >= MORNING_START_US && tod <= MORNING_END_US {
        Some(t_us)
    } else if tod >= AFTERNOON_START_US && tod <= AFTERNOON_END_US {
        Some(t_us - LUNCH_US)
    } else {
        None
    }
}

/// 一条时间序列：在连续档区间 [b0, b1] 上有值，区间外是 NaN。
#[derive(Clone)]
struct Ser {
    b0: i64,
    b1: i64,
    v: Vec<f64>,
}

impl Ser {
    #[inline]
    fn get(&self, b: i64) -> f64 {
        if b < self.b0 || b > self.b1 {
            f64::NAN
        } else {
            self.v[(b - self.b0) as usize]
        }
    }
}

/// 两条序列档区间的并集
fn union_range(a: (i64, i64), b: (i64, i64)) -> (i64, i64) {
    (a.0.min(b.0), a.1.max(b.1))
}

/// 两条序列档区间的交集；无交集返回 None
fn inter_range(a: (i64, i64), b: (i64, i64)) -> Option<(i64, i64)> {
    let lo = a.0.max(b.0);
    let hi = a.1.min(b.1);
    if lo > hi {
        None
    } else {
        Some((lo, hi))
    }
}

/// a / (a+b).replace(inf,1).fillna(0)（fillna 只作用在分母上，结果不 fillna）
fn ratio66(a: &Ser, b: &Ser) -> Ser {
    let (lo, hi) = union_range((a.b0, a.b1), (b.b0, b.b1));
    let n = (hi - lo + 1) as usize;
    let mut v = vec![f64::NAN; n];
    for i in 0..n {
        let bb = lo + i as i64;
        let num = a.get(bb);
        let mut den = a.get(bb) + b.get(bb);
        if den == f64::INFINITY {
            den = 1.0;
        }
        if den.is_nan() {
            den = 0.0;
        }
        v[i] = num / den;
    }
    Ser { b0: lo, b1: hi, v }
}

/// shift(-k)：档不变，值整体上移 k 位，尾部落 NaN
fn shift_up(s: &Ser, k: i64) -> Ser {
    let n = s.v.len();
    let mut v = vec![f64::NAN; n];
    for i in 0..n {
        let j = i as i64 + k;
        if j >= 0 && (j as usize) < n {
            v[i] = s.v[j as usize];
        }
    }
    Ser {
        b0: s.b0,
        b1: s.b1,
        v,
    }
}

/// rolling(w).sum()：min_periods=w，窗口内非 NaN 个数不足 w 给 NaN
fn rolling_sum(s: &Ser, w: usize) -> Ser {
    let n = s.v.len();
    let mut v = vec![f64::NAN; n];
    for i in 0..n {
        if i + 1 < w {
            continue;
        }
        let mut sum = 0.0f64;
        let mut cnt = 0usize;
        for j in (i + 1 - w)..=i {
            if !s.v[j].is_nan() {
                sum += s.v[j];
                cnt += 1;
            }
        }
        if cnt == w {
            v[i] = sum;
        }
    }
    Ser {
        b0: s.b0,
        b1: s.b1,
        v,
    }
}

// ============================================================================
// 读市场快照（自己解析，取 f64 的 volume 与 20 档挂单量）
// ============================================================================

struct Market {
    b0: i64,
    /// 每档的 volume（原值，未 diff）
    volume: Vec<f64>,
    /// 每档 20 档挂单量，顺序 ask_vol1,bid_vol1,ask_vol2,bid_vol2,...
    vols: Vec<[f64; 20]>,
}

fn parse_f64(b: &[u8]) -> f64 {
    if b.is_empty() {
        return f64::NAN;
    }
    std::str::from_utf8(b)
        .ok()
        .and_then(|s| s.trim().parse::<f64>().ok())
        .unwrap_or(f64::NAN)
}

/// CSV 里的 exchtime 写成 "1740960939000000.0"，要按整数前缀解析（同 fast_csv_reader::parse_i64_fast）
fn parse_i64(b: &[u8]) -> i64 {
    let mut i = 0usize;
    let mut neg = false;
    while i < b.len() && (b[i] == b' ' || b[i] == b'\t') {
        i += 1;
    }
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

fn read_market(code: &str, date: i64) -> std::io::Result<Market> {
    let filename = format!("{}_{}_market_data.csv", code, date);
    let mut path = None;
    for root in crate::data_paths::level2_roots() {
        let p = root.join(date.to_string()).join("market_data").join(&filename);
        if p.exists() {
            path = Some(p);
            break;
        }
    }
    let path = path.ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("market_data 未找到: {filename}"),
        )
    })?;
    let mut content = String::new();
    File::open(&path)?.read_to_string(&mut content)?;

    // 第一遍：解析并做 adjust_afternoon，记录档区间
    struct Row {
        bin: i64,
        volume: f64,
        vols: [f64; 20],
    }
    let mut rows: Vec<Row> = Vec::new();
    let mut b0 = i64::MAX;
    let mut b1 = i64::MIN;
    let mut first = true;
    for line in content.split('\n') {
        if first {
            first = false;
            continue;
        }
        let line = line.strip_suffix('\r').unwrap_or(line);
        if line.is_empty() {
            continue;
        }
        let f: Vec<&[u8]> = line.as_bytes().split(|&c| c == b',').collect();
        if f.len() < 61 {
            continue;
        }
        let ask_prc1 = parse_f64(f[MKT_COL_ASK_PRC_BASE]);
        let bid_prc1 = parse_f64(f[MKT_COL_ASK_PRC_BASE + 2]);
        if ask_prc1 == 0.0 || bid_prc1 == 0.0 {
            continue;
        }
        let exch = parse_i64(f[MKT_COL_EXCHTIME]);
        if exch == 0 {
            continue;
        }
        let t = match afternoon_adjust(exch + EXCH_OFFSET_US) {
            Some(t) => t,
            None => continue,
        };
        let bin = bin_of(t);
        let mut vols = [f64::NAN; 20];
        for k in 0..10 {
            let base = MKT_COL_ASK_PRC_BASE + k * 4;
            vols[k * 2] = parse_f64(f[base + 1]);
            vols[k * 2 + 1] = parse_f64(f[base + 3]);
        }
        if bin < b0 {
            b0 = bin;
        }
        if bin > b1 {
            b1 = bin;
        }
        rows.push(Row {
            bin,
            volume: parse_f64(f[MKT_COL_VOLUME]),
            vols,
        });
    }
    if rows.is_empty() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "market_data 空",
        ));
    }
    let n = (b1 - b0 + 1) as usize;
    let mut volume = vec![f64::NAN; n];
    let mut vols = vec![[f64::NAN; 20]; n];
    // resample('15s').last()：逐列取该档内最后一个非 NaN
    for r in rows.iter() {
        let i = (r.bin - b0) as usize;
        if !r.volume.is_nan() {
            volume[i] = r.volume;
        }
        for k in 0..20 {
            if !r.vols[k].is_nan() {
                vols[i][k] = r.vols[k];
            }
        }
    }
    Ok(Market { b0, volume, vols })
}

// ============================================================================
// 读逐笔成交 → 按 15s 分档求 flag=66 / 83 的成交量
// ============================================================================

/// 返回 (b0, sums) 连续档区间上的分档量和；无数据返回 None
fn trade_bins(code: &str, date: i64, flag: i32) -> std::io::Result<Option<(i64, Vec<f64>)>> {
    // with_afternoon_adjust=false：时间不过滤，自己在下面按精确微秒做 afternoon 平移；
    // 读取时保留 flag==32（with_retreat=true），再按解析出的整数 flag 过滤，
    // 因为新数据里撤单标记可能写成 "32.0"，库里的字节比较不生效。
    let trades = crate::fast_csv_reader::read_trade_fast_inner(code, date, true, false, usize::MAX)?;
    let mut b0 = i64::MAX;
    let mut b1 = i64::MIN;
    let mut pairs: Vec<(i64, f64)> = Vec::new();
    for r in trades.iter() {
        if r.flag == 32 || r.flag != flag {
            continue;
        }
        let t = match afternoon_adjust(r.time_us) {
            Some(t) => t,
            None => continue,
        };
        let bin = bin_of(t);
        if bin < b0 {
            b0 = bin;
        }
        if bin > b1 {
            b1 = bin;
        }
        pairs.push((bin, r.volume as f64));
    }
    if pairs.is_empty() {
        return Ok(None);
    }
    let n = (b1 - b0 + 1) as usize;
    let mut sums = vec![0.0f64; n];
    for (bin, vol) in pairs {
        sums[(bin - b0) as usize] += vol;
    }
    Ok(Some((b0, sums)))
}

// ============================================================================
// lims_corrs：fast_correlation_matrix_v2 的逐项复刻
// ============================================================================

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

/// 两列的 Pearson：照抄 fast_correlation_v2 的两条分支（clean / 含 NaN）
fn fast_pearson(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len();
    let vx: Vec<f64> = x.iter().filter(|v| !v.is_nan()).copied().collect();
    let vy: Vec<f64> = y.iter().filter(|v| !v.is_nan()).copied().collect();
    if vx.len() < 1 || vy.len() < 1 {
        return f64::NAN;
    }
    if vx.len() == n && vy.len() == n {
        // clean 分支
        if n < 2 {
            return f64::NAN;
        }
        let mi = kahan_sum(&vx) / n as f64;
        let mj = kahan_sum(&vy) / n as f64;
        let (mut cov, mut vi, mut vj) = (0.0f64, 0.0f64, 0.0f64);
        let chunks = n / 8;
        let rem = n % 8;
        for c in 0..chunks {
            let base = c * 8;
            for k in 0..8 {
                let a = vx[base + k] - mi;
                let b = vy[base + k] - mj;
                cov += a * b;
                vi += a * a;
                vj += b * b;
            }
        }
        for k in 0..rem {
            let idx = chunks * 8 + k;
            let a = vx[idx] - mi;
            let b = vy[idx] - mj;
            cov += a * b;
            vi += a * a;
            vj += b * b;
        }
        if vi.abs() < f64::EPSILON || vj.abs() < f64::EPSILON {
            return f64::NAN;
        }
        return cov / (vi.sqrt() * vj.sqrt());
    }
    // 含 NaN 的分支：共同有效观测
    let mut pairs: Vec<(f64, f64)> = Vec::new();
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

/// calculate_correlation_optimized（dataframe_corrwith_single_thread 的逐列内核）
fn corr_optimized(pairs: &[(f64, f64)]) -> f64 {
    let n = pairs.len();
    if n < 2 {
        return f64::NAN;
    }
    let mut sum_x = 0.0f64;
    let mut sum_y = 0.0f64;
    for &(a, b) in pairs.iter() {
        sum_x += a;
        sum_y += b;
    }
    let mean_x = sum_x / n as f64;
    let mean_y = sum_y / n as f64;
    let (mut cov, mut vx, mut vy) = (0.0f64, 0.0f64, 0.0f64);
    for chunk in pairs.chunks(32) {
        let (mut lc, mut lx, mut ly) = (0.0f64, 0.0f64, 0.0f64);
        for &(a, b) in chunk.iter() {
            let dx = a - mean_x;
            let dy = b - mean_y;
            lc += dx * dy;
            lx += dx * dx;
            ly += dy * dy;
        }
        cov += lc;
        vx += lx;
        vy += ly;
    }
    if vx.abs() < f64::EPSILON || vy.abs() < f64::EPSILON {
        return f64::NAN;
    }
    cov / (vx.sqrt() * vy.sqrt())
}

// ============================================================================
// 单列特征（get_features_factors，关闭 corr/threshold_counts/lyapunov/abs/max_min）
// ============================================================================

fn col_mean(v: &[f64]) -> (f64, usize) {
    let mut s = 0.0f64;
    let mut n = 0usize;
    for &x in v.iter() {
        if !x.is_nan() {
            s += x;
            n += 1;
        }
    }
    if n == 0 {
        (f64::NAN, 0)
    } else {
        (s / n as f64, n)
    }
}

fn col_valid(v: &[f64]) -> Vec<f64> {
    v.iter().filter(|x| !x.is_nan()).copied().collect()
}

fn col_std(v: &[f64]) -> f64 {
    let (m, n) = col_mean(v);
    if n < 2 {
        return f64::NAN;
    }
    let mut s = 0.0f64;
    for &x in v.iter() {
        if !x.is_nan() {
            let d = x - m;
            s += d * d;
        }
    }
    (s / (n - 1) as f64).sqrt()
}

fn col_skew(v: &[f64]) -> f64 {
    let (m, n) = col_mean(v);
    if n < 3 {
        return f64::NAN;
    }
    let nf = n as f64;
    let mut m2 = 0.0f64;
    let mut m3 = 0.0f64;
    for &x in v.iter() {
        if !x.is_nan() {
            let d = x - m;
            m2 += d * d;
            m3 += d * d * d;
        }
    }
    m2 /= nf;
    m3 /= nf;
    let g1 = m3 / m2.powf(1.5);
    g1 * (nf * (nf - 1.0)).sqrt() / (nf - 2.0)
}

fn col_kurt(v: &[f64]) -> f64 {
    let (m, n) = col_mean(v);
    if n < 4 {
        return f64::NAN;
    }
    let nf = n as f64;
    let mut m2 = 0.0f64;
    let mut m4 = 0.0f64;
    for &x in v.iter() {
        if !x.is_nan() {
            let d = x - m;
            m2 += d * d;
            m4 += d * d * d * d;
        }
    }
    m2 /= nf;
    m4 /= nf;
    let r = m4 / (m2 * m2) - 3.0;
    ((nf - 1.0) / ((nf - 2.0) * (nf - 3.0))) * ((nf + 1.0) * r + 6.0)
}

fn col_median(valid: &[f64]) -> f64 {
    let n = valid.len();
    if n == 0 {
        return f64::NAN;
    }
    let mut s = valid.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    if n % 2 == 1 {
        s[n / 2]
    } else {
        (s[n / 2 - 1] + s[n / 2]) / 2.0
    }
}

/// pandas quantile(interpolation='linear')
fn col_quantile(sorted: &[f64], q: f64) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return f64::NAN;
    }
    if n == 1 {
        return sorted[0];
    }
    let pos = q * (n - 1) as f64;
    let lo = pos.floor();
    let frac = pos - lo;
    let i = lo as usize;
    if i + 1 >= n {
        sorted[n - 1]
    } else {
        sorted[i] + frac * (sorted[i + 1] - sorted[i])
    }
}

/// trend_2d(axis=0) 的单列版（照抄 time_series::trend_mod::calculate_trend_1d）
fn col_trend(v: &[f64]) -> f64 {
    let mut values: Vec<f64> = Vec::new();
    let mut idx: Vec<f64> = Vec::new();
    for (i, &x) in v.iter().enumerate() {
        if x.is_finite() {
            values.push(x);
            idx.push((i + 1) as f64);
        }
    }
    let n = values.len();
    if n <= 1 {
        return 0.0;
    }
    let mx = values.iter().sum::<f64>() / n as f64;
    let my = idx.iter().sum::<f64>() / n as f64;
    let (mut cov, mut vx, mut vy) = (0.0f64, 0.0f64, 0.0f64);
    for i in 0..n {
        let dx = values[i] - mx;
        let dy = idx[i] - my;
        cov += dx * dy;
        vx += dx * dx;
        vy += dy * dy;
    }
    if vx == 0.0 || vy == 0.0 {
        return 0.0;
    }
    cov / (vx.sqrt() * vy.sqrt())
}

/// lz_complexity(seq, [0.33,0.66])（照抄 lz_complexity.rs 的离散化 + 归一化）
fn col_lz(seq: &[f64]) -> f64 {
    let n = seq.len();
    if n == 0 {
        return 0.0;
    }
    let mut values: Vec<f64> = seq.to_vec();
    values.sort_unstable_by(|a, b| match (a.is_nan(), b.is_nan()) {
        (true, true) => std::cmp::Ordering::Equal,
        (true, false) => std::cmp::Ordering::Greater,
        (false, true) => std::cmp::Ordering::Less,
        (false, false) => a.partial_cmp(b).unwrap(),
    });
    let qs = [0.33f64, 0.66f64];
    let mut thresholds = [0.0f64; 2];
    for (k, &q) in qs.iter().enumerate() {
        let idx = ((n - 1) as f64 * q) as usize;
        thresholds[k] = values[idx.min(n - 1)];
    }
    let mut disc: Vec<u8> = Vec::with_capacity(n);
    for &val in seq.iter() {
        let mut sym = 2usize;
        for i in 0..2 {
            if val <= thresholds[i] {
                sym = i;
                break;
            }
        }
        disc.push((sym + 1) as u8);
    }
    let complexity = crate::lz_complexity::calculate_lz_complexity(&disc);
    if n <= 1 {
        return 0.0;
    }
    let mut seen = [false; 4];
    for &s in disc.iter() {
        seen[s as usize] = true;
    }
    let k_eff = seen.iter().filter(|&&x| x).count() as f64;
    if k_eff < 3.0 {
        return f64::NAN;
    }
    complexity as f64 * ((n as f64).ln() / k_eff.ln()) / n as f64
}

/// calculate_binned_entropy_1d（equal_width）
fn col_entropy(seq: &[f64]) -> f64 {
    let n = seq.len();
    if n == 0 {
        return 0.0;
    }
    let n_bins = ((n as f64).log2().ceil() as usize) + 1;
    if n_bins == 0 {
        return 0.0;
    }
    let filtered: Vec<f64> = seq.iter().filter(|x| !x.is_nan()).copied().collect();
    if filtered.is_empty() || filtered.len() == 1 {
        return 0.0;
    }
    let mn = filtered.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let mx = filtered.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    if (mx - mn).abs() < f64::EPSILON {
        return 0.0;
    }
    let bin_width = (mx - mn) / n_bins as f64;
    let mut counts: Vec<usize> = vec![0; n_bins + 1];
    for &val in seq.iter() {
        if val.is_nan() {
            counts[n_bins] += 1;
        } else {
            let mut bi = ((val - mn) / bin_width).floor() as usize;
            if bi >= n_bins {
                bi = n_bins - 1;
            }
            counts[bi] += 1;
        }
    }
    let total = n as f64;
    let mut ent = 0.0f64;
    for &c in counts.iter() {
        if c > 0 {
            let p = c as f64 / total;
            ent += -p * p.ln();
        }
    }
    ent
}

/// find_max_range_product 的复刻（照抄 sequence/mod.rs），返回 abs(x-y)/n
fn col_max_range_product(seq: &[f64]) -> f64 {
    let n = seq.len();
    if n < 2 {
        return 0.0;
    }
    let mut max_product = f64::NEG_INFINITY;
    let mut res = (0i64, 0i64);
    let mut left = 0usize;
    let mut right = n - 1;
    while left < right {
        let product = seq[left].min(seq[right]) * (right - left) as f64;
        if product > max_product {
            max_product = product;
            res = (left as i64, right as i64);
        }
        if seq[left] < seq[right] {
            left += 1;
        } else {
            right -= 1;
        }
    }
    for i in 0..n - 1 {
        let product = seq[i].min(seq[i + 1]);
        if product > max_product {
            max_product = product;
            res = (i as i64, (i + 1) as i64);
        }
    }
    ((res.0 - res.1).abs()) as f64 / n as f64
}

/// 把 19 个特征写进 feats[feat * N_COLS + col]
fn col_features(v: &[f64], n_rows: usize, col: usize, feats: &mut [f64]) {
    let (m, _cnt) = col_mean(v);
    let valid = col_valid(v);
    let mut sorted = valid.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let sd = col_std(v);
    let p5 = col_quantile(&sorted, 0.05);
    let p25 = col_quantile(&sorted, 0.25);
    let p75 = col_quantile(&sorted, 0.75);
    let p95 = col_quantile(&sorted, 0.95);

    let mut put = |fi: usize, val: f64| {
        feats[fi * N_COLS + col] = val;
    };
    put(0, m);
    put(1, col_median(&sorted));
    put(2, sd);
    put(3, col_skew(v));
    put(4, col_kurt(v));
    put(5, p5);
    put(6, p25);
    put(7, p75);
    put(8, p95);
    put(9, p75 - p25);
    put(10, sd / (m.abs() + 1e-8));

    // autocorr1 = corrwith(x, x.shift(1))：配对 (v[i], v[i-1])
    let mut pairs: Vec<(f64, f64)> = Vec::new();
    for i in 1..v.len() {
        if !v[i].is_nan() && !v[i - 1].is_nan() {
            pairs.push((v[i], v[i - 1]));
        }
    }
    let ac = corr_optimized(&pairs);
    put(11, ac);
    put(12, ac.abs());
    put(13, col_trend(v));

    // period_diff / period_ratio
    let split = n_rows / 3;
    let first = if split == 0 {
        f64::NAN
    } else {
        col_mean(&v[..split]).0
    };
    let last = if split == 0 {
        col_mean(v).0
    } else {
        col_mean(&v[n_rows - split..]).0
    };
    put(14, last - first);
    put(15, last / (first.abs() + 1e-8));

    put(16, col_lz(v));
    put(17, col_entropy(v));
    put(18, col_max_range_product(v));
}

// ============================================================================
// 主流程
// ============================================================================

struct PriceSeries {
    b0: i64,
    b1: i64,
    v: Vec<f64>,
}

impl PriceSeries {
    #[inline]
    fn get(&self, b: i64) -> f64 {
        if b < self.b0 || b > self.b1 {
            f64::NAN
        } else {
            self.v[(b - self.b0) as usize]
        }
    }
}

/// 调试开关：设了 HM084_DEBUG 就把中间的档数打到 stderr
fn dbg_on() -> bool {
    std::env::var("HM084_DEBUG").is_ok()
}

pub fn compute_stock(code: &str, date: i64) -> Option<Vec<f32>> {
    let dbg = dbg_on();
    let mkt = match read_market(code, date) {
        Ok(m) => m,
        Err(e) => {
            if dbg {
                eprintln!("[hm084] read_market 失败 {} {}: {}", code, date, e);
            }
            return None;
        }
    };
    let t66 = match trade_bins(code, date, 66).ok().flatten() {
        Some(v) => v,
        None => {
            if dbg {
                eprintln!("[hm084] flag66 无数据 {} {}", code, date);
            }
            return None;
        }
    };
    let t83 = match trade_bins(code, date, 83).ok().flatten() {
        Some(v) => v,
        None => {
            if dbg {
                eprintln!("[hm084] flag83 无数据 {} {}", code, date);
            }
            return None;
        }
    };

    let t_mkt = mkt.volume.len();
    let g0 = mkt.b0;
    let g1 = mkt.b0 + t_mkt as i64 - 1;
    if dbg {
        eprintln!(
            "[hm084] {} {}: 市场档数 {} [{}..{}], flag66 档 {} [{}..{}], flag83 档 {} [{}..{}]",
            code,
            date,
            t_mkt,
            g0,
            g1,
            t66.1.len(),
            t66.0,
            t66.0 + t66.1.len() as i64 - 1,
            t83.1.len(),
            t83.0,
            t83.0 + t83.1.len() as i64 - 1
        );
    }

    // ---- volume / local_volume ----
    let mut vol = mkt.volume.clone();
    for i in (1..t_mkt).rev() {
        vol[i] = if vol[i].is_nan() || vol[i - 1].is_nan() {
            0.0
        } else {
            vol[i] - vol[i - 1]
        };
    }
    if t_mkt > 0 {
        vol[0] = 0.0;
    }
    let mut local = vec![f64::NAN; t_mkt];
    for i in 0..t_mkt {
        if i + 1 >= 60 {
            let s: f64 = vol[i + 1 - 60..=i].iter().sum();
            local[i] = vol[i] / (s / 60.0);
        }
    }
    let total: f64 = vol.iter().sum();
    let mut vol_norm = vec![f64::NAN; t_mkt];
    for i in 0..t_mkt {
        vol_norm[i] = vol[i] / total;
    }

    // ---- lims_corrs（T × T，行主序） ----
    let lims = {
        // 21 × T
        let mut m = vec![0.0f64; 21 * t_mkt];
        for b in 0..t_mkt {
            for k in 0..20 {
                m[k * t_mkt + b] = mkt.vols[b][k] / 100.0;
            }
            m[20 * t_mkt + b] = local[b] / 100.0;
        }
        m
    };
    // lims 是 21 行 × T 列，相关矩阵按**列**求（每个时间点是一列，21 个观测）
    let mut corr = vec![f64::NAN; t_mkt * t_mkt];
    {
        let mut ci: Vec<f64> = vec![0.0; 21];
        let mut cj: Vec<f64> = vec![0.0; 21];
        for i in 0..t_mkt {
            corr[i * t_mkt + i] = 1.0;
            for k in 0..21 {
                ci[k] = lims[k * t_mkt + i];
            }
            for j in (i + 1)..t_mkt {
                for k in 0..21 {
                    cj[k] = lims[k * t_mkt + j];
                }
                let c = fast_pearson(&ci, &cj);
                corr[i * t_mkt + j] = c;
                corr[j * t_mkt + i] = c;
            }
        }
    }

    // ---- 逐笔分档 ----
    let ser66 = Ser {
        b0: t66.0,
        b1: t66.0 + t66.1.len() as i64 - 1,
        v: t66.1,
    };
    let ser83 = Ser {
        b0: t83.0,
        b1: t83.0 + t83.1.len() as i64 - 1,
        v: t83.1,
    };
    let r3_past = ratio66(&ser66, &ser83);
    let r3_future = shift_up(&r3_past, 1);
    let r15_past = ratio66(&rolling_sum(&ser66, 5), &rolling_sum(&ser83, 5));
    let r15_future = shift_up(&r15_past, 5);
    let r60_past = ratio66(&rolling_sum(&ser66, 20), &rolling_sum(&ser83, 20));
    let r60_future = shift_up(&r60_past, 20);

    // ---- 成交量的 15s 档（已在 15s 网格上；resample('15s').sum().fillna(0) 等于把 NaN 抹成 0） ----
    let mk_ser = |vals: Vec<f64>| -> Ser {
        Ser {
            b0: g0,
            b1: g1,
            v: vals.into_iter().map(|x| if x.is_nan() { 0.0 } else { x }).collect(),
        }
    };
    let v3_past = mk_ser(vol_norm.clone());
    let v3_future = shift_up(&v3_past, 1);
    let v15_past = rolling_sum(&v3_past, 5);
    let v15_future = shift_up(&v15_past, 5);
    let v60_past = rolling_sum(&v3_past, 20);
    let v60_future = shift_up(&v60_past, 20);

    let l3_past = mk_ser(local.clone());
    let l3_future = shift_up(&l3_past, 1);
    let l15_past = rolling_sum(&l3_past, 5);
    let l15_future = shift_up(&l15_past, 5);
    let l60_past = rolling_sum(&l3_past, 20);
    let l60_future = shift_up(&l60_past, 20);

    // ---- morph_decay_steps / rate66_at_decay ----
    let mut morph = vec![f64::NAN; t_mkt];
    let mut at_decay = vec![f64::NAN; t_mkt];
    for i in 0..t_mkt {
        let mut found: Option<usize> = None;
        for j in (i + 1)..t_mkt {
            if corr[i * t_mkt + j] < PARAM_DECAY_THRESH {
                found = Some(j);
                break;
            }
        }
        if let Some(j) = found {
            morph[i] = j as f64;
            let mut ti = i as i64 + j as i64;
            if ti > (t_mkt - 1) as i64 {
                ti = (t_mkt - 1) as i64;
            }
            at_decay[i] = r3_past.get(g0 + ti);
        }
    }
    let morph_ser = Ser {
        b0: g0,
        b1: g1,
        v: morph,
    };
    let at_decay_ser = Ser {
        b0: g0,
        b1: g1,
        v: at_decay,
    };

    let prices: Vec<PriceSeries> = vec![
        PriceSeries { b0: r3_past.b0, b1: r3_past.b1, v: r3_past.v.clone() },
        PriceSeries { b0: r3_future.b0, b1: r3_future.b1, v: r3_future.v.clone() },
        PriceSeries { b0: r15_past.b0, b1: r15_past.b1, v: r15_past.v.clone() },
        PriceSeries { b0: r15_future.b0, b1: r15_future.b1, v: r15_future.v.clone() },
        PriceSeries { b0: r60_past.b0, b1: r60_past.b1, v: r60_past.v.clone() },
        PriceSeries { b0: r60_future.b0, b1: r60_future.b1, v: r60_future.v.clone() },
        PriceSeries { b0: v3_past.b0, b1: v3_past.b1, v: v3_past.v.clone() },
        PriceSeries { b0: v3_future.b0, b1: v3_future.b1, v: v3_future.v.clone() },
        PriceSeries { b0: v15_past.b0, b1: v15_past.b1, v: v15_past.v.clone() },
        PriceSeries { b0: v15_future.b0, b1: v15_future.b1, v: v15_future.v.clone() },
        PriceSeries { b0: v60_past.b0, b1: v60_past.b1, v: v60_past.v.clone() },
        PriceSeries { b0: v60_future.b0, b1: v60_future.b1, v: v60_future.v.clone() },
        PriceSeries { b0: l3_past.b0, b1: l3_past.b1, v: l3_past.v.clone() },
        PriceSeries { b0: l3_future.b0, b1: l3_future.b1, v: l3_future.v.clone() },
        PriceSeries { b0: l15_past.b0, b1: l15_past.b1, v: l15_past.v.clone() },
        PriceSeries { b0: l15_future.b0, b1: l15_future.b1, v: l15_future.v.clone() },
        PriceSeries { b0: l60_past.b0, b1: l60_past.b1, v: l60_past.v.clone() },
        PriceSeries { b0: l60_future.b0, b1: l60_future.b1, v: l60_future.v.clone() },
        PriceSeries { b0: morph_ser.b0, b1: morph_ser.b1, v: morph_ser.v.clone() },
        PriceSeries { b0: at_decay_ser.b0, b1: at_decay_ser.b1, v: at_decay_ser.v.clone() },
    ];

    // ---- 全局行轴 ----
    let mut b_lo = g0;
    let mut b_hi = g1;
    for p in prices.iter() {
        b_lo = b_lo.min(p.b0);
        b_hi = b_hi.max(p.b1);
    }
    let n_rows = (b_hi - b_lo + 1) as usize;

    // big_pos / big_neg（作用在 lims_corrs 上）
    let mut big_pos = vec![f64::NAN; t_mkt * t_mkt];
    let mut big_neg = vec![f64::NAN; t_mkt * t_mkt];
    let mut corr_less = vec![f64::NAN; t_mkt * t_mkt];
    for i in 0..t_mkt * t_mkt {
        let c = corr[i];
        if c < PARAM_LESS && c > PARAM_BIG_POS {
            big_pos[i] = c;
        }
        if c < PARAM_LESS && c < PARAM_BIG_NEG {
            big_neg[i] = c;
        }
        if c < PARAM_LESS {
            corr_less[i] = c;
        }
    }

    // ---- 每个序列产出 10 列 ----
    let mut feats = vec![f64::NAN; N_FEAT * N_COLS];
    let mut colbuf = vec![f64::NAN; n_rows];
    for (si, p) in prices.iter().enumerate() {
        let t = (p.b1 - p.b0 + 1) as usize;
        // diff 矩阵（行主序）
        let mut d = vec![f64::NAN; t * t];
        for i in 0..t {
            for j in 0..t {
                if !p.v[i].is_nan() && !p.v[j].is_nan() {
                    d[i * t + j] = (p.v[i] - p.v[j]).abs();
                }
            }
        }
        // 与 lims_corrs 的行/列交集
        let inter = inter_range((p.b0, p.b1), (g0, g1));
        let (i0, i1) = match inter {
            Some(r) => r,
            None => (1, 0),
        };

        // corr_full / corr_less：逐列 Pearson
        let mut cf = vec![f64::NAN; t];
        let mut cl = vec![f64::NAN; t];
        let mut cbp = vec![f64::NAN; t];
        let mut cbn = vec![f64::NAN; t];
        let mut mbp = vec![f64::NAN; t];
        let mut mbn = vec![f64::NAN; t];
        if i0 <= i1 {
            for j in 0..t {
                let bj = p.b0 + j as i64;
                if bj < i0 || bj > i1 {
                    continue;
                }
                let gi = (bj - g0) as usize;
                let mut pa: Vec<(f64, f64)> = Vec::new();
                let mut pb: Vec<(f64, f64)> = Vec::new();
                let mut pc: Vec<(f64, f64)> = Vec::new();
                let mut pd: Vec<(f64, f64)> = Vec::new();
                for i in 0..t {
                    let bi = p.b0 + i as i64;
                    if bi < i0 || bi > i1 {
                        continue;
                    }
                    let row = (bi - g0) as usize;
                    // corr_optimized 只接受已经筛掉 NaN 的配对，NaN 必须在入队前丢掉
                    let dv = d[i * t + j];
                    if dv.is_nan() {
                        continue;
                    }
                    let c = corr[row * t_mkt + gi];
                    if c.is_nan() {
                        continue;
                    }
                    pa.push((dv, c));
                    let c2 = corr_less[row * t_mkt + gi];
                    if !c2.is_nan() {
                        pb.push((dv, c2));
                    }
                    let bpos = big_pos[row * t_mkt + gi];
                    if !bpos.is_nan() {
                        pc.push((bpos * dv, c));
                    }
                    let bneg = big_neg[row * t_mkt + gi];
                    if !bneg.is_nan() {
                        pd.push((bneg * dv, c));
                    }
                }
                cf[j] = corr_optimized(&pa);
                cl[j] = corr_optimized(&pb);
                cbp[j] = corr_optimized(&pc);
                cbn[j] = corr_optimized(&pd);
            }
        }
        // mean_big_pos / mean_big_neg（逐列 skipna 均值，作用在整个 diff 矩阵上）
        for j in 0..t {
            let (mut s1, mut n1) = (0.0f64, 0usize);
            let (mut s2, mut n2) = (0.0f64, 0usize);
            let bj = p.b0 + j as i64;
            if bj < g0 || bj > g1 {
                continue;
            }
            let gj = (bj - g0) as usize;
            for i in 0..t {
                let bi = p.b0 + i as i64;
                if bi < g0 || bi > g1 {
                    continue;
                }
                let row = (bi - g0) as usize;
                let dv = d[i * t + j];
                if !dv.is_nan() {
                    let bp = big_pos[row * t_mkt + gj];
                    if !bp.is_nan() {
                        s1 += bp * dv;
                        n1 += 1;
                    }
                    let bn = big_neg[row * t_mkt + gj];
                    if !bn.is_nan() {
                        s2 += bn * dv;
                        n2 += 1;
                    }
                }
            }
            mbp[j] = if n1 == 0 { f64::NAN } else { s1 / n1 as f64 };
            mbn[j] = if n2 == 0 { f64::NAN } else { s2 / n2 as f64 };
        }
        // big_pos / big_neg 的 top100 / bottom100
        let (pos_early, pos_late) = stack_pick(&big_pos, t_mkt, true);
        let (neg_early, neg_late) = stack_pick(&big_neg, t_mkt, false);
        let mut bp_early = vec![f64::NAN; t];
        let mut bp_late = vec![f64::NAN; t];
        let mut bn_early = vec![f64::NAN; t];
        let mut bn_late = vec![f64::NAN; t];
        for i in 0..t {
            let bi = p.b0 + i as i64;
            let gi = bi - g0;
            if pos_early.contains(&gi) {
                bp_early[i] = p.v[i];
            }
            if pos_late.contains(&gi) {
                bp_late[i] = p.v[i];
            }
            if neg_early.contains(&gi) {
                bn_early[i] = p.v[i];
            }
            if neg_late.contains(&gi) {
                bn_late[i] = p.v[i];
            }
        }

        // 落到全局行轴
        let base = (p.b0 - b_lo) as usize;
        for (mi, src) in [
            &cf, &cl, &cbp, &cbn, &mbp, &mbn, &bp_early, &bp_late, &bn_early, &bn_late,
        ]
        .iter()
        .enumerate()
        {
            for i in 0..n_rows {
                colbuf[i] = f64::NAN;
            }
            for i in 0..t {
                colbuf[base + i] = src[i];
            }
            col_features(&colbuf, n_rows, si * N_METRICS + mi, &mut feats);
        }
    }

    let mut out = Vec::with_capacity(N_OUT);
    for &(_, fi, ci) in TABLE.iter() {
        out.push(feats[fi * N_COLS + ci] as f32);
    }
    Some(out)
}

/// big_pos / big_neg 的 stack→sort_values 取 head/tail 100，返回 (early 档集合, late 档集合)。
///
/// top = true 取最大的 100 个（值降序、并列按原始位置降序）；
/// top = false 取最小的 100 个（值升序、并列按原始位置升序）。
/// 只返回去重后的档号集合，因为下游只用 isin。
fn stack_pick(m: &[f64], t: usize, top: bool) -> (Vec<i64>, Vec<i64>) {
    let mut items: Vec<(f64, usize)> = Vec::new();
    for i in 0..t {
        for j in 0..t {
            let v = m[i * t + j];
            if !v.is_nan() {
                items.push((v, i * t + j));
            }
        }
    }
    if top {
        items.sort_by(|a, b| {
            b.0.partial_cmp(&a.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(b.1.cmp(&a.1))
        });
    } else {
        items.sort_by(|a, b| {
            a.0.partial_cmp(&b.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.1.cmp(&b.1))
        });
    }
    let k = items.len().min(100);
    let mut early: Vec<i64> = Vec::new();
    let mut late: Vec<i64> = Vec::new();
    for &(_, pos) in items[..k].iter() {
        let i = pos / t;
        let j = pos % t;
        let (e, l) = if i <= j { (i, j) } else { (j, i) };
        let e = e as i64;
        let l = l as i64;
        if !early.contains(&e) {
            early.push(e);
        }
        if !late.contains(&l) {
            late.push(l);
        }
    }
    (early, late)
}
