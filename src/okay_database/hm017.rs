//! hm17 色差 —— 旧框架（design_whatever_out.FactorDesign，check_my_kind="minute"）的 Rust 复现。
//!
//! 旧文件：/home/chenzongwei/pythoncode/成果hm17_色差/hm17_okay.py
//!   run_factor_out(..., get_many=0, read_full=0, cross_section=False, only_cal_these_days=1)
//!   分钟系列的执行方式是 `cal_factor_one_day(date)`：**一天一次、一次全市场**，
//!   每次 new 一个 `Go(base_dir, date, date, 1, "hm17_base", read_full=0, get_many=0)`，
//!   所以 start=end=date、look_back_window=1 ⇒ 只看当天那 240 根分钟线。
//!   因此本系列走 kind='cross_section' 的 compute_cross。
//!
//! ════════════════════════════════════════════════════════════════════════
//! 一、旧代码逐步对照（写代码前先抄在这里，方便复核）
//! ════════════════════════════════════════════════════════════════════════
//!   1. 六个分钟字段，每个都是 240 行 × 全部股票的 DataFrame（index 是 "new"=日期）：
//!        ask_size = ask_size_10_mean    ask_vwap = ask_vwap_mean    ask_do = act_sell_amount_sum
//!        bid_size = bid_size_10_mean    bid_vwap = bid_vwap_mean    bid_do = act_buy_amount_sum
//!      注意 get_field_min_data 会把 `index % 240 >= 237` 的三根（一天里最后三根）置 NaN。
//!
//!   2. 每个字段按 5 种权重切成 5 份（旧代码里的局部函数 get_many）：
//!        v0 = a                                          （all）
//!        v1 = a * nums.where(nums <= 30  , nan)          （open30）
//!        v2 = a * nums.where(nums <= 120 , nan)          （morning）
//!        v3 = a * nums.where(nums >  120 , nan)          （afternoon）
//!        v4 = a * nums.where(nums >  210 , nan)          （close30）
//!      ！！nums 是 1..240 的分钟序号，`where(cond, nan)` **保留的是序号本身**，
//!      不是 1.0。所以这里是「按分钟序号加权」，不是简单的取/不取。
//!      这一点最容易看错——第一版原型就是按 1.0/NaN 写的，除了 all 以外全对不上。
//!
//!   3. 三个 kind 是同一批字段的三种角色排列：
//!        kind1: ask(r,g,b)=(size, vwap, do)     bid=(size, vwap, do)
//!        kind2: ask=(vwap, do, size)            bid=(vwap, do, size)
//!        kind3: ask=(size, do, vwap)            bid=(size, do, vwap)
//!
//!   4. rgb_to_cielab(r,g,b)：把三个 DataFrame 当成一张 RGB 图，逐列（每只股票）算 CIE-Lab
//!      色差序列：
//!        r,g,b 各自先 r.rank()（axis=0，升序、并列取平均、NaN 保留）再 /255
//!        x = (0.4124562r + 0.3575761g + 0.1804375b) / 0.95047
//!        y =  0.2126729r + 0.7151522g + 0.0721750b
//!        z = (0.0193330r + 0.1191920g + 0.9503041b) / 1.08883
//!        l = 116*y^(1/3) - 16，若 l < 6/29 换成 903.3*y
//!        a = 500*(x^(1/3) - y^(1/3))，b = 200*(y^(1/3) - z^(1/3))
//!        delta_e = sqrt((l-l.shift(1))^2 + (a-a.shift(1))^2 + (b-b.shift(1))^2)
//!      第一行（shift 之后是 NaN）恒为 NaN。
//!
//!   5. one(r1,g1,b1, r2,g2,b2)：
//!        asks = cielab(三个 ask 字段)，bids = cielab(三个 bid 字段)
//!        mean_diff = asks.mean().rank() - bids.mean().rank()
//!        std_diff  = asks.std().rank()  - bids.std().rank()
//!      mean/std 是**沿 240 行**（每只股票自己的一天），后面的 rank 是**横截面**（跨股票）。
//!      mean/std 都是 skipna；std 用 ddof=1，有效样本 < 2 给 NaN。
//!      rank 的 NaN 保留，NaN - NaN 还是 NaN。
//!
//!   6. 输出顺序：kind1 的 5 个权重 × (mean_diff, std_diff) = 10 个，
//!      再 kind2 的 10 个、kind3 的 10 个，最后把前 30 个取绝对值再接在后面，共 60 个。
//!
//! ════════════════════════════════════════════════════════════════════════
//! 二、输出排布（横截面系列的约定，别搞反）
//! ════════════════════════════════════════════════════════════════════════
//! compute_cross 返回 (codes, vals)，vals 是**股票优先**：每只股票连续 60 个值，
//! 第 i 只股票的值在 vals[i*60 .. i*60+60]，与 worker 侧
//! `vals.chunks(expected_len).zip(codes.iter())` 的分块方式一致。
//! expected_len = names().len() = 60。
//!
//! codes 是分钟数据 symbol_map.csv 里的全名单（20240102 那天是 5792 只），
//! 与旧代码 `i.to_frame(date).T` 出来的列完全一致；写盘时由流水线按标准名单再对齐。
//!
//! ════════════════════════════════════════════════════════════════════════
//! 三、数值口径
//! ════════════════════════════════════════════════════════════════════════
//! · 逐列均值/标准差都按行号从 0 到 239 **顺序累加**（与 pandas/numpy 沿 axis=0
//!   的朴素累加一致），标准差用两遍算法（先均值再求偏差平方和），与 pandas 实测一致。
//! · x^(1/3) 用 `powf(1.0/3.0)`，与 Python 的 `** (1/3)` 走同一个 libm pow。
//! · delta_e 的平方和按 (l 项 + a 项) + b 项的次序相加，与旧代码的表达式一致。
//! · 非有限值一律写成 NaN，与旧框架落盘时的 replace(inf, nan) 口径一致。

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, LazyLock, Mutex};

use rayon::prelude::*;

/// 分钟数据每天 240 根 K 线
const MIN_PER_DAY: usize = 240;
/// 旧框架 get_field_min_data 把每根日线索引 % 240 >= 237 的三根置 NaN
const MIN_NAN_FROM: usize = 237;
/// 输出因子个数 = 3 个 kind × 5 个权重 × 2 个统计量 × 2（原值 + 绝对值）
pub const N_OUT: usize = 60;

/// 六个分钟字段，下标固定：0..2 是 ask 那三个，3..5 是 bid 那三个
const FIELDS: [&str; 6] = [
    "ask_size_10_mean",
    "ask_vwap_mean",
    "act_sell_amount_sum",
    "bid_size_10_mean",
    "bid_vwap_mean",
    "act_buy_amount_sum",
];

/// 三个 kind：(ask 的 (r,g,b) 字段下标, bid 的 (r,g,b) 字段下标)
const KINDS: [[[usize; 3]; 2]; 3] = [
    [[0, 1, 2], [3, 4, 5]],
    [[1, 2, 0], [4, 5, 3]],
    [[0, 2, 1], [3, 5, 4]],
];

/// 十个基础名，顺序就是旧代码 names 里的前 10 个
const BASE: [&str; 10] = [
    "all_mean_diff",
    "all_std_diff",
    "open30_mean_diff",
    "open30_std_diff",
    "morning_mean_diff",
    "morning_std_diff",
    "afternoon_mean_diff",
    "afternoon_std_diff",
    "close30_mean_diff",
    "close30_std_diff",
];

pub fn names() -> Vec<String> {
    let mut out: Vec<String> = Vec::with_capacity(N_OUT);
    for k in 1..=3 {
        for b in BASE.iter() {
            out.push(format!("{}_kind{}", b, k));
        }
    }
    let head = out.clone();
    for n in head.iter() {
        out.push(format!("{}_abs", n));
    }
    debug_assert_eq!(out.len(), N_OUT);
    out
}

// ============================================================================
// 分钟数据：元信息 + 一天的整块读取
// ============================================================================

struct MinMeta {
    date_to_dayidx: HashMap<i64, usize>,
    /// 按 symbol_map.csv 的 pos 排列的股票代码
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
        // 表头非数字，跳过；能 parse 的才占一个 day_idx
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
// 小工具
// ============================================================================

/// 某个权重档的逐行权重（240 个）：v0 全是 1.0，其余是「分钟序号本身或 NaN」。
fn weight_of(variant: usize) -> Vec<f64> {
    let mut w = vec![1.0f64; MIN_PER_DAY];
    let cond = |m: usize| -> bool {
        match variant {
            1 => m <= 30,
            2 => m <= 120,
            3 => m > 120,
            4 => m > 210,
            _ => true,
        }
    };
    for m in 0..MIN_PER_DAY {
        if !cond(m + 1) {
            w[m] = f64::NAN;
        } else if variant != 0 {
            w[m] = (m + 1) as f64;
        }
    }
    w
}

/// 逐列 rank（pandas DataFrame.rank，axis=0，升序、并列取平均、NaN 保留）。
/// 输入输出都是行优先的 nrows × ncols。
fn rank_columns(mat: &[f64], nrows: usize, ncols: usize) -> Vec<f64> {
    let mut out = vec![f64::NAN; nrows * ncols];
    let cols: Vec<(usize, Vec<(f64, usize)>)> = (0..ncols)
        .into_par_iter()
        .map(|j| {
            let mut items: Vec<(f64, usize)> = Vec::with_capacity(nrows);
            for i in 0..nrows {
                let v = mat[i * ncols + j];
                if !v.is_nan() {
                    items.push((v, i));
                }
            }
            (j, items)
        })
        .collect();
    // 排序与写回都按列独立，结果与串行一致
    let filled: Vec<(usize, Vec<(usize, f64)>)> = cols
        .into_par_iter()
        .map(|(j, mut items)| {
            items.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            let m = items.len();
            let mut res: Vec<(usize, f64)> = Vec::with_capacity(m);
            let mut i = 0usize;
            while i < m {
                let mut k = i + 1;
                while k < m && items[k].0 == items[i].0 {
                    k += 1;
                }
                let avg = ((i + 1) as f64 + k as f64) / 2.0;
                for t in i..k {
                    res.push((items[t].1, avg));
                }
                i = k;
            }
            (j, res)
        })
        .collect();
    for (j, res) in filled {
        for (i, r) in res {
            out[i * ncols + j] = r;
        }
    }
    out
}

/// 一维 rank（pandas Series.rank），NaN 保留
fn rank_average(values: &[f64]) -> Vec<f64> {
    let n = values.len();
    let mut idx: Vec<usize> = (0..n).filter(|&i| !values[i].is_nan()).collect();
    idx.sort_by(|&a, &b| {
        values[a]
            .partial_cmp(&values[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut out = vec![f64::NAN; n];
    let mut i = 0usize;
    while i < idx.len() {
        let mut j = i + 1;
        while j < idx.len() && values[idx[j]] == values[idx[i]] {
            j += 1;
        }
        let avg = ((i + 1) as f64 + j as f64) / 2.0;
        for k in i..j {
            out[idx[k]] = avg;
        }
        i = j;
    }
    out
}

/// 单格 CIE-Lab 的 (l, a, b)
#[inline]
fn lab(r: f64, g: f64, b: f64) -> (f64, f64, f64) {
    let r = r / 255.0;
    let g = g / 255.0;
    let b = b / 255.0;
    let x = (0.4124562 * r + 0.3575761 * g + 0.1804375 * b) / 0.95047;
    let y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b;
    let z = (0.0193330 * r + 0.1191920 * g + 0.9503041 * b) / 1.08883;
    let third = 1.0f64 / 3.0;
    let cx = x.powf(third);
    let cy = y.powf(third);
    let cz = z.powf(third);
    let l0 = 116.0 * cy - 16.0;
    let l = if l0 >= 6.0 / 29.0 { l0 } else { 903.3 * y };
    (l, 500.0 * (cx - cy), 200.0 * (cy - cz))
}

/// 三张 rank 矩阵 → 每只股票的 delta_e 序列 → 逐列均值与样本标准差（ddof=1，skipna）。
/// 均值那一遍按行号顺序累加，方差那一遍也是，和 pandas/numpy 沿 axis=0 的口径一致。
fn delta_mean_std(rr: &[f64], rg: &[f64], rb: &[f64], ncols: usize) -> (Vec<f64>, Vec<f64>) {
    let per_col: Vec<(f64, f64)> = (0..ncols)
        .into_par_iter()
        .map(|j| {
            let mut de = [f64::NAN; MIN_PER_DAY];
            let mut prev: Option<(f64, f64, f64)> = None;
            for i in 0..MIN_PER_DAY {
                let idx = i * ncols + j;
                let cur = lab(rr[idx], rg[idx], rb[idx]);
                if let Some(p) = prev {
                    let d1 = cur.0 - p.0;
                    let d2 = cur.1 - p.1;
                    let d3 = cur.2 - p.2;
                    de[i] = (d1 * d1 + d2 * d2 + d3 * d3).sqrt();
                }
                prev = Some(cur);
            }
            let mut sum = 0.0f64;
            let mut cnt = 0usize;
            for i in 0..MIN_PER_DAY {
                if de[i].is_nan() {
                    continue;
                }
                sum += de[i];
                cnt += 1;
            }
            if cnt == 0 {
                return (f64::NAN, f64::NAN);
            }
            let mean = sum / cnt as f64;
            if cnt < 2 {
                return (mean, f64::NAN);
            }
            let mut ss = 0.0f64;
            for i in 0..MIN_PER_DAY {
                if de[i].is_nan() {
                    continue;
                }
                let d = de[i] - mean;
                ss += d * d;
            }
            (mean, (ss / (cnt as f64 - 1.0)).sqrt())
        })
        .collect();
    let mut mean = vec![f64::NAN; ncols];
    let mut sd = vec![f64::NAN; ncols];
    for (j, (m, s)) in per_col.into_iter().enumerate() {
        mean[j] = m;
        sd[j] = s;
    }
    (mean, sd)
}

/// 非有限值（inf/-inf）写成 NaN，与旧框架落盘时的 replace(inf, nan) 一致
#[inline]
fn finite_or_nan(v: f64) -> f64 {
    if v.is_finite() {
        v
    } else {
        f64::NAN
    }
}

// ============================================================================
// 入口
// ============================================================================

/// hm17 是分钟系列，一天一次全市场，没有逐股实现。
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

    // 六个字段各读一整块（240 × n），最后三根已经是 NaN
    let mut fields: Vec<Vec<f64>> = Vec::with_capacity(FIELDS.len());
    for f in FIELDS.iter() {
        fields.push(read_day_block(f, day, n)?);
    }

    let mut out = vec![f64::NAN; n * N_OUT];
    for variant in 0..5usize {
        let w = weight_of(variant);
        // 六个字段在这一档上的逐列 rank（三个 kind 是同一批字段的排列，rank 只算一次）
        let ranks: Vec<Vec<f64>> = fields
            .par_iter()
            .map(|f| {
                // 权重是按**行**（分钟）给的，不是按元素给的：先用行号取权重再逐行乘
                let mut masked = vec![f64::NAN; f.len()];
                for i in 0..MIN_PER_DAY {
                    let ww = w[i];
                    if ww.is_nan() {
                        continue;
                    }
                    let base = i * n;
                    for k in base..base + n {
                        let v = f[k];
                        if !v.is_nan() {
                            masked[k] = v * ww;
                        }
                    }
                }
                rank_columns(&masked, MIN_PER_DAY, n)
            })
            .collect();

        for (ki, kind) in KINDS.iter().enumerate() {
            let ask = &kind[0];
            let bid = &kind[1];
            let (am, asd) = delta_mean_std(&ranks[ask[0]], &ranks[ask[1]], &ranks[ask[2]], n);
            let (bm, bsd) = delta_mean_std(&ranks[bid[0]], &ranks[bid[1]], &ranks[bid[2]], n);
            let arm = rank_average(&am);
            let brm = rank_average(&bm);
            let ars = rank_average(&asd);
            let brs = rank_average(&bsd);
            let base = ki * 10 + variant * 2;
            for i in 0..n {
                let md = finite_or_nan(arm[i] - brm[i]);
                let sd = finite_or_nan(ars[i] - brs[i]);
                out[i * N_OUT + base] = md;
                out[i * N_OUT + base + 1] = sd;
                out[i * N_OUT + 30 + base] = finite_or_nan(md.abs());
                out[i * N_OUT + 30 + base + 1] = finite_or_nan(sd.abs());
            }
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
        assert_eq!(n[0], "all_mean_diff_kind1");
        assert_eq!(n[1], "all_std_diff_kind1");
        assert_eq!(n[2], "open30_mean_diff_kind1");
        assert_eq!(n[10], "all_mean_diff_kind2");
        assert_eq!(n[20], "all_mean_diff_kind3");
        assert_eq!(n[30], "all_mean_diff_kind1_abs");
        assert_eq!(n[59], "close30_std_diff_kind3_abs");
    }

    #[test]
    fn weights_are_minute_numbers() {
        let w0 = weight_of(0);
        assert!(w0.iter().all(|v| *v == 1.0));
        let w1 = weight_of(1);
        assert_eq!(w1[0], 1.0);
        assert_eq!(w1[29], 30.0);
        assert!(w1[30].is_nan());
        let w3 = weight_of(3);
        assert!(w3[119].is_nan());
        assert_eq!(w3[120], 121.0);
        assert_eq!(w3[239], 240.0);
        let w4 = weight_of(4);
        assert!(w4[209].is_nan());
        assert_eq!(w4[210], 211.0);
    }

    #[test]
    fn rank_matches_pandas() {
        // 列 0: [3, 1, 1, NaN]  → [3, 1.5, 1.5, NaN]
        let mat = vec![3.0, 10.0, 1.0, 20.0, 1.0, 30.0, f64::NAN, 40.0];
        let r = rank_columns(&mat, 4, 2);
        assert_eq!(r[0], 3.0);
        assert_eq!(r[2], 1.5);
        assert_eq!(r[4], 1.5);
        assert!(r[6].is_nan());
        assert_eq!(r[1], 1.0);
        assert_eq!(r[3], 2.0);
        assert_eq!(r[5], 3.0);
        assert_eq!(r[7], 4.0);
    }

    #[test]
    fn rank1_average() {
        let v = [2.0, 1.0, 1.0, f64::NAN, 3.0];
        let r = rank_average(&v);
        assert_eq!(r[0], 3.0);
        assert_eq!(r[1], 1.5);
        assert_eq!(r[2], 1.5);
        assert!(r[3].is_nan());
        assert_eq!(r[4], 4.0);
    }
}
