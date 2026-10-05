//! hm1（hot_change2）—— 旧框架（design_whatever_out.FactorDesign，**分钟**数据）的 Rust 复现。
//!
//! 旧文件：/home/chenzongwei/pythoncode/成果hm0_成果hm1_hot_change/hm1_okay.py
//!   run_base 的 Go.get_factor，参数 get_many=**3** / read_full=0 / change_time=1，
//!   Go.check_my_kind == "minute"，所以是一天一次、一次全市场，走 compute_cross。
//!
//! ════════════════════════════════════════════════════════════════════════
//! 一、names 的来源（旧文件里 names 是拼出来的）
//! ════════════════════════════════════════════════════════════════════════
//!   11 个基础名 × 5 段后缀，段序 = [allday, morning, afternoon, amountbig, amountsmall]
//!   （names1 + names2 + names3 + names6 + names7，注意 names4/names5 在这个系列里没定义，
//!     也就是**没有** open30 / close30 两段）。共 55 个。
//!
//! ════════════════════════════════════════════════════════════════════════
//! 二、get_many=3 的 5 段权重
//! ════════════════════════════════════════════════════════════════════════
//!   0 allday    = a
//!   1 morning   = a * nums.where(nums <= 120, nan)   ← 第 m(1..120) 行乘 m，其余 NaN
//!   2 afternoon = a * nums.where(nums >  120, nan)   ← 第 m(121..240) 行乘 m，其余 NaN
//!   3 amountbig = a * amounts_big
//!   4 amountsmall = a * amounts_small
//! 其中（FactorDesign.__init__ 里算的，用的是**原始 amount**，不带任何权重）：
//!   amounts_big   = where(amount_raw - 该股当日 amount 中位数 > 0, 1, nan)
//!   amounts_small = where(amount_raw - 该股当日 amount 中位数 < 0, 1, nan)
//! 中位数是 `amounts.groupby("new").median()`，即**每只股票在当天 240 行上的中位数**
//! （只一天，所以就是列中位数；amount 本身在第 237~239 行已被框架置 NaN）。
//!
//! ════════════════════════════════════════════════════════════════════════
//! 三、旧代码逐步对照
//! ════════════════════════════════════════════════════════════════════════
//!   one(highs, lows, field, order) 里，对 down ∈ {0, 1}：
//!     down=0: rushs = (highs / lows.shift(1)  - 1).rank(axis=1)   ← 每分钟跨股票排名
//!     down=1: rushs = (lows  / highs.shift(1) - 1).rank(axis=1)
//!     judge = rushs - rushs.shift(1)
//!     speed = judge.abs()
//!     pos = sign(judge) →replace(0,1)→replace(-1,nan)                ⇒ judge >= 0 才为 1
//!     neg = sign(judge) →replace(0,-1)→replace(1,nan)→replace(-1,1)  ⇒ judge <= 0 才为 1
//!     is_fast1 = sign(speed.T - speed.T.median())→replace(0,1)→replace(-1,nan)→.T
//!                 ⇒ speed >= 同一分钟全市场中位数 才为 1（逐格，不是整行）
//!     is_slow1 = 同一套 replace 的慢档                              ⇒ speed <= 中位数 才为 1
//!     afp = amounts * is_fast1 * pos    （1/NaN 相乘就是取掩码，NaN 传播）
//!     afn = amounts * is_fast1 * neg
//!     asp = amounts * is_slow1 * pos
//!     asn = amounts * is_slow1 * neg
//!     五个逐列求和（DataFrame.sum()，axis=0，对 240 行顺序累加、跳过 NaN）：
//!         ams = [sum(afp), sum(afn), sum(asp), sum(asn), sum(amounts)]
//!     返回 [ams0/ams4, ams1/ams4, ams2/ams4, ams3/ams4]
//!   on_ud(0) + on_ud(1) 拼成 8 个（下标 0..3 是 down=0，4..7 是 down=1）。
//!
//!   每个 order 取这些位置（5 个字段各调一次 one()）：
//!     amount              → [son0, son3, son5, son6] = [d0.afp, d0.asn, d1.afn, d1.asp]
//!     act_buy_amount_sum  → [son0]                   = [d0.afp]
//!     act_sell_amount_sum → [son0, son3, son5]       = [d0.afp, d0.asn, d1.afn]
//!     up_tick_count       → [son0, son2]             = [d0.afp, d0.asp]
//!     down_tick_count     → [son0]                   = [d0.afp]
//!   共 11 个，除以各自的 ams[4]（那一段该字段自己的总和）。
//!   由名字可反推：up1=d0.afp up3=d0.asp up4=d0.asn down2=d1.afn down3=d1.asp。
//!
//! ════════════════════════════════════════════════════════════════════════
//! 四、数值口径
//! ════════════════════════════════════════════════════════════════════════
//! · DataFrame.sum()（axis=0，多列 C 顺序）numpy 走**顺序累加**而不是成对求和
//!   （实测：np.sum(M, axis=0) 在 (5792,2) 的 C 顺序数组上给出顺序累加的结果），
//!   所以这一层按行号 0..239 顺序累加、跳过 NaN。
//! · 中位数：nanmedian，偶数个取中间两个的均值。
//! · 非有限值一律写 NaN（与旧框架落盘口径一致）。
//!
//! ════════════════════════════════════════════════════════════════════════
//! 五、输出排布
//! ════════════════════════════════════════════════════════════════════════
//!   vals 是股票优先：第 i 只股票连续 55 个值，段序 allday → morning → afternoon
//!   → amountbig → amountsmall，段内就是上面那 11 个。

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, LazyLock, Mutex};

use rayon::prelude::*;

const MIN_PER_DAY: usize = 240;
const MIN_NAN_FROM: usize = 237;
/// 11 个基础名 × 5 段
pub const N_OUT: usize = 55;
/// 每段的输出个数
const PER_SEG: usize = 11;

/// 七个分钟字段，下标固定
const F_HIGH: usize = 0;
const F_LOW: usize = 1;
const F_AMOUNT: usize = 2;
const F_BUY: usize = 3;
const F_SELL: usize = 4;
const F_UPTICK: usize = 5;
const F_DOWNTICK: usize = 6;
const FIELDS: [&str; 7] = [
    "high",
    "low",
    "amount",
    "act_buy_amount_sum",
    "act_sell_amount_sum",
    "up_tick_count",
    "down_tick_count",
];

/// 五个段后缀，顺序就是 names 的拼法
const SEGS: [&str; 5] = ["allday", "morning", "afternoon", "amountbig", "amountsmall"];

/// 11 个基础名
const BASE: [&str; 11] = [
    "hot_change_on_stock_rate_amount_up1",
    "hot_change_on_stock_rate_amount_up4",
    "hot_change_on_stock_rate_amount_down2",
    "hot_change_on_stock_rate_amount_down3",
    "hot_change_on_stock_rate_act_buy_amount_sum_up1",
    "hot_change_on_stock_rate_act_sell_amount_sum_up1",
    "hot_change_on_stock_rate_act_sell_amount_sum_up4",
    "hot_change_on_stock_rate_act_sell_amount_sum_down2",
    "hot_change_on_stock_rate_up_tick_count_up1",
    "hot_change_on_stock_rate_up_tick_count_up3",
    "hot_change_on_stock_rate_down_tick_count_up1",
];

pub fn names() -> Vec<String> {
    let mut out: Vec<String> = Vec::with_capacity(N_OUT);
    for seg in SEGS.iter() {
        for b in BASE.iter() {
            out.push(format!("{}_{}", b, seg));
        }
    }
    debug_assert_eq!(out.len(), N_OUT);
    out
}

// ============================================================================
// 分钟数据：元信息 + 一天的整块读取
// ============================================================================

struct MinMeta {
    date_to_dayidx: HashMap<i64, usize>,
    codes: Vec<String>,
}

static MIN_META: LazyLock<Mutex<HashMap<PathBuf, Arc<MinMeta>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

fn load_min_meta(dir: &Path) -> Option<MinMeta> {
    let cal = fs::read_to_string(dir.join("calendar_map.csv")).ok()?;
    let mut date_to_dayidx = HashMap::new();
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
// 小工具
// ============================================================================

/// 第 seg 段的逐行权重（0..239 行）：allday/amountbig/amountsmall 由调用方另行处理，
/// 这里只负责 allday(0)/morning(1)/afternoon(2) 三种分钟加权。
fn minute_weight(seg: usize, row: usize) -> f64 {
    let m = (row + 1) as f64;
    match seg {
        0 => 1.0,
        1 => {
            if row + 1 <= 120 {
                m
            } else {
                f64::NAN
            }
        }
        _ => {
            if row + 1 > 120 {
                m
            } else {
                f64::NAN
            }
        }
    }
}

/// 逐行乘权重（权重 NaN 的行整行 NaN）
fn apply_row_weight(src: &[f64], n: usize, w: &[f64]) -> Vec<f64> {
    let mut out = vec![f64::NAN; src.len()];
    for row in 0..MIN_PER_DAY {
        let ww = w[row];
        if ww.is_nan() {
            continue;
        }
        let base = row * n;
        for j in 0..n {
            out[base + j] = src[base + j] * ww;
        }
    }
    out
}

/// 逐格乘掩码（amountbig / amountsmall 是 240×n 的逐格 1/NaN，不是逐行权重）
fn apply_elem_mask(src: &[f64], m: &[f64]) -> Vec<f64> {
    src.iter().zip(m.iter()).map(|(a, b)| a * b).collect()
}

/// 逐行 rank（pandas DataFrame.rank(axis=1)，升序、并列取平均、NaN 保留）
fn rank_rows(mat: &[f64], nrows: usize, ncols: usize) -> Vec<f64> {
    let rows: Vec<(usize, Vec<(usize, f64)>)> = (0..nrows)
        .into_par_iter()
        .map(|i| {
            let mut items: Vec<(usize, f64)> = Vec::with_capacity(ncols);
            for j in 0..ncols {
                let v = mat[i * ncols + j];
                if !v.is_nan() {
                    items.push((j, v));
                }
            }
            items.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            let m = items.len();
            let mut res: Vec<(usize, f64)> = Vec::with_capacity(m);
            let mut k = 0usize;
            while k < m {
                let mut e = k + 1;
                while e < m && items[e].1 == items[k].1 {
                    e += 1;
                }
                let avg = ((k + 1) as f64 + e as f64) / 2.0;
                for t in k..e {
                    res.push((items[t].0, avg));
                }
                k = e;
            }
            (i, res)
        })
        .collect();
    let mut out = vec![f64::NAN; nrows * ncols];
    for (i, res) in rows {
        for (j, r) in res {
            out[i * ncols + j] = r;
        }
    }
    out
}

/// 逐列中位数（pandas groupby("new").median()：跳过 NaN，偶数取中间两个的均值）
fn column_median(mat: &[f64], n: usize) -> Vec<f64> {
    let res: Vec<(usize, f64)> = (0..n)
        .into_par_iter()
        .map(|j| {
            let mut v: Vec<f64> = Vec::with_capacity(MIN_PER_DAY);
            for row in 0..MIN_PER_DAY {
                let x = mat[row * n + j];
                if !x.is_nan() {
                    v.push(x);
                }
            }
            if v.is_empty() {
                return (j, f64::NAN);
            }
            v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let m = v.len();
            let r = if m % 2 == 1 {
                v[m / 2]
            } else {
                (v[m / 2 - 1] + v[m / 2]) / 2.0
            };
            (j, r)
        })
        .collect();
    let mut out = vec![f64::NAN; n];
    for (j, v) in res {
        out[j] = v;
    }
    out
}

/// 逐行中位数（每分钟跨股票，对应旧代码 speed.T.median()）
fn row_median_nan(mat: &[f64], n: usize) -> Vec<f64> {
    let rows: Vec<(usize, f64)> = (0..MIN_PER_DAY)
        .into_par_iter()
        .map(|row| {
            let base = row * n;
            let mut v: Vec<f64> = Vec::with_capacity(n);
            for j in 0..n {
                let x = mat[base + j];
                if !x.is_nan() {
                    v.push(x);
                }
            }
            if v.is_empty() {
                return (row, f64::NAN);
            }
            v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let m = v.len();
            let r = if m % 2 == 1 {
                v[m / 2]
            } else {
                (v[m / 2 - 1] + v[m / 2]) / 2.0
            };
            (row, r)
        })
        .collect();
    let mut out = vec![f64::NAN; MIN_PER_DAY];
    for (row, v) in rows {
        out[row] = v;
    }
    out
}

/// 逐列顺序累加（pandas DataFrame.sum()，跳过 NaN）
fn column_sum(mat: &[f64], n: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; n];
    for row in 0..MIN_PER_DAY {
        let base = row * n;
        for j in 0..n {
            let v = mat[base + j];
            if !v.is_nan() {
                out[j] += v;
            }
        }
    }
    out
}

/// 掩码逐列顺序累加：只在 m[row*n+j] 为 1.0 且值非 NaN 时累加
fn masked_column_sum(mat: &[f64], m: &[f64], n: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; n];
    for row in 0..MIN_PER_DAY {
        let base = row * n;
        for j in 0..n {
            if m[base + j] == 1.0 {
                let v = mat[base + j];
                if !v.is_nan() {
                    out[j] += v;
                }
            }
        }
    }
    out
}

#[inline]
fn finite_or_nan(v: f64) -> f64 {
    if v.is_finite() {
        v
    } else {
        f64::NAN
    }
}

// ============================================================================
// 一方向（down=0/1）的四个掩码
// ============================================================================

struct DirMasks {
    /// fast & pos / fast & neg / slow & pos / slow & neg，都是 1.0 或 NaN 的 240×n
    afp: Vec<f64>,
    afn: Vec<f64>,
    asp: Vec<f64>,
    asn: Vec<f64>,
}

fn dir_masks(high: &[f64], low: &[f64], down: bool, n: usize) -> DirMasks {
    // rushs
    let mut ratio = vec![f64::NAN; MIN_PER_DAY * n];
    for row in 1..MIN_PER_DAY {
        for j in 0..n {
            let (x, y) = if down {
                (low[row * n + j], high[(row - 1) * n + j])
            } else {
                (high[row * n + j], low[(row - 1) * n + j])
            };
            ratio[row * n + j] = x / y - 1.0;
        }
    }
    let rushs = rank_rows(&ratio, MIN_PER_DAY, n);
    let mut judge = vec![f64::NAN; MIN_PER_DAY * n];
    let mut speed = vec![f64::NAN; MIN_PER_DAY * n];
    for row in 1..MIN_PER_DAY {
        for j in 0..n {
            let d = rushs[row * n + j] - rushs[(row - 1) * n + j];
            judge[row * n + j] = d;
            speed[row * n + j] = d.abs();
        }
    }
    let med = row_median_nan(&speed, n);

    let mut afp = vec![f64::NAN; MIN_PER_DAY * n];
    let mut afn = vec![f64::NAN; MIN_PER_DAY * n];
    let mut asp = vec![f64::NAN; MIN_PER_DAY * n];
    let mut asn = vec![f64::NAN; MIN_PER_DAY * n];
    for row in 0..MIN_PER_DAY {
        let md = med[row];
        if md.is_nan() {
            continue;
        }
        let base = row * n;
        for j in 0..n {
            let d = judge[base + j];
            let s = speed[base + j];
            if d.is_nan() || s.is_nan() {
                continue;
            }
            let fast = s >= md;
            let slow = s <= md;
            let pos = d >= 0.0;
            let neg = d <= 0.0;
            if fast && pos {
                afp[base + j] = 1.0;
            }
            if fast && neg {
                afn[base + j] = 1.0;
            }
            if slow && pos {
                asp[base + j] = 1.0;
            }
            if slow && neg {
                asn[base + j] = 1.0;
            }
        }
    }
    DirMasks { afp, afn, asp, asn }
}

// ============================================================================
// 入口
// ============================================================================

pub fn compute_stock(_code: &str, _date: i64) -> Option<Vec<f32>> {
    None
}

pub fn compute_cross(date: i64) -> Option<(Vec<String>, Vec<f32>)> {
    let meta = min_meta()?;
    let day = *meta.date_to_dayidx.get(&date)?;
    let codes = meta.codes.clone();
    let n = codes.len();
    if n == 0 {
        return None;
    }

    let raw: Vec<Vec<f64>> = FIELDS
        .iter()
        .map(|f| read_day_block(f, day, n))
        .collect::<Option<Vec<_>>>()?;

    // amountbig / amountsmall 掩码（原始 amount，按股票当天列中位数）
    let med_amount = column_median(&raw[F_AMOUNT], n);
    let mut big = vec![f64::NAN; MIN_PER_DAY * n];
    let mut small = vec![f64::NAN; MIN_PER_DAY * n];
    for row in 0..MIN_PER_DAY {
        for j in 0..n {
            let m = med_amount[j];
            if m.is_nan() {
                continue;
            }
            let v = raw[F_AMOUNT][row * n + j];
            let d = v - m;
            if d > 0.0 {
                big[row * n + j] = 1.0;
            } else if d < 0.0 {
                small[row * n + j] = 1.0;
            }
        }
    }

    let mut out = vec![f64::NAN; n * N_OUT];
    for seg in 0..SEGS.len() {
        // 该段的 7 个加权字段
        let w: Vec<f64> = (0..MIN_PER_DAY)
            .map(|row| minute_weight(seg, row))
            .collect();
        let mut f: Vec<Vec<f64>> = Vec::with_capacity(FIELDS.len());
        for k in 0..FIELDS.len() {
            let v = match seg {
                0 | 1 | 2 => apply_row_weight(&raw[k], n, &w),
                3 => apply_elem_mask(&raw[k], &big),
                _ => apply_elem_mask(&raw[k], &small),
            };
            f.push(v);
        }

        let d0 = dir_masks(&f[F_HIGH], &f[F_LOW], false, n);
        let d1 = dir_masks(&f[F_HIGH], &f[F_LOW], true, n);

        // 每个字段的 5 个和：afp, afn, asp, asn(各方向) + 总和
        // 先把需要的量算出来
        let tot_amount = column_sum(&f[F_AMOUNT], n);
        let tot_buy = column_sum(&f[F_BUY], n);
        let tot_sell = column_sum(&f[F_SELL], n);
        let tot_up = column_sum(&f[F_UPTICK], n);
        let tot_down = column_sum(&f[F_DOWNTICK], n);

        let amount_d0_afp = masked_column_sum(&f[F_AMOUNT], &d0.afp, n);
        let amount_d0_asn = masked_column_sum(&f[F_AMOUNT], &d0.asn, n);
        let amount_d1_afn = masked_column_sum(&f[F_AMOUNT], &d1.afn, n);
        let amount_d1_asp = masked_column_sum(&f[F_AMOUNT], &d1.asp, n);

        let buy_d0_afp = masked_column_sum(&f[F_BUY], &d0.afp, n);

        let sell_d0_afp = masked_column_sum(&f[F_SELL], &d0.afp, n);
        let sell_d0_asn = masked_column_sum(&f[F_SELL], &d0.asn, n);
        let sell_d1_afn = masked_column_sum(&f[F_SELL], &d1.afn, n);

        let up_d0_afp = masked_column_sum(&f[F_UPTICK], &d0.afp, n);
        let up_d0_asp = masked_column_sum(&f[F_UPTICK], &d0.asp, n);

        let down_d0_afp = masked_column_sum(&f[F_DOWNTICK], &d0.afp, n);

        for j in 0..n {
            let base = seg * PER_SEG;
            let idx = j * N_OUT + base;
            out[idx] = finite_or_nan(amount_d0_afp[j] / tot_amount[j]);
            out[idx + 1] = finite_or_nan(amount_d0_asn[j] / tot_amount[j]);
            out[idx + 2] = finite_or_nan(amount_d1_afn[j] / tot_amount[j]);
            out[idx + 3] = finite_or_nan(amount_d1_asp[j] / tot_amount[j]);
            out[idx + 4] = finite_or_nan(buy_d0_afp[j] / tot_buy[j]);
            out[idx + 5] = finite_or_nan(sell_d0_afp[j] / tot_sell[j]);
            out[idx + 6] = finite_or_nan(sell_d0_asn[j] / tot_sell[j]);
            out[idx + 7] = finite_or_nan(sell_d1_afn[j] / tot_sell[j]);
            out[idx + 8] = finite_or_nan(up_d0_afp[j] / tot_up[j]);
            out[idx + 9] = finite_or_nan(up_d0_asp[j] / tot_up[j]);
            out[idx + 10] = finite_or_nan(down_d0_afp[j] / tot_down[j]);
        }
    }

    let vals: Vec<f32> = out.into_iter().map(|v| v as f32).collect();
    debug_assert_eq!(vals.len(), codes.len() * N_OUT);
    Some((codes, vals))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_layout() {
        let n = names();
        assert_eq!(n.len(), N_OUT);
        assert_eq!(n[0], "hot_change_on_stock_rate_amount_up1_allday");
        assert_eq!(n[10], "hot_change_on_stock_rate_down_tick_count_up1_allday");
        assert_eq!(n[11], "hot_change_on_stock_rate_amount_up1_morning");
        assert_eq!(n[22], "hot_change_on_stock_rate_amount_up1_afternoon");
        assert_eq!(n[33], "hot_change_on_stock_rate_amount_up1_amountbig");
        assert_eq!(n[44], "hot_change_on_stock_rate_amount_up1_amountsmall");
        assert_eq!(n[54], "hot_change_on_stock_rate_down_tick_count_up1_amountsmall");
    }

    #[test]
    fn minute_weights() {
        assert_eq!(minute_weight(0, 0), 1.0);
        assert_eq!(minute_weight(1, 0), 1.0);
        assert_eq!(minute_weight(1, 119), 120.0);
        assert!(minute_weight(1, 120).is_nan());
        assert!(minute_weight(2, 119).is_nan());
        assert_eq!(minute_weight(2, 120), 121.0);
    }

    #[test]
    fn col_sum_is_sequential() {
        // 顺序累加：[1.0, 1e-16 * (K-1)] 的和是 1.0（成对求和会给出 1.0000000000005782）
        let k = 5792;
        let mut mat = vec![1e-16f64; k];
        mat[0] = 1.0;
        let s = column_sum(&mat, 1);
        assert_eq!(s[0], 1.0);
    }
}
