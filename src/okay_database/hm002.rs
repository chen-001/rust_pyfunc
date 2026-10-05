//! hm2（stranded_trades）—— 旧框架（design_whatever_out.FactorDesign，**分钟**数据）的 Rust 复现。
//!
//! 旧文件：/home/chenzongwei/pythoncode/成果hm2_ask_mutation/hm2_okay.py
//!   run_base 的 Go.get_factor，参数 get_many=**4** / read_full=0，一天一次全市场 ⇒ compute_cross。
//!
//! ════════════════════════════════════════════════════════════════════════
//! 一、names 的来源（拼出来的）
//! ════════════════════════════════════════════════════════════════════════
//!   names = ['stranded_trades_buy', 'stranded_trades_sell']
//!   段序 = names1 + names4 + names2 + names3 + names5 + names6 + names7
//!        = allday, open30, morning, afternoon, close30, amountbig, amountsmall
//!   与 get_many=4 的段顺序**完全一致**（旧文件里特意按这个顺序拼的）。
//!   每段两个值：先 buy 再 sell。共 14 个。
//!
//! ════════════════════════════════════════════════════════════════════════
//! 二、get_many=4 的 7 段权重
//! ════════════════════════════════════════════════════════════════════════
//!   0 allday    = a
//!   1 open30    = a * nums.where(nums <=  30, nan)   ← 第 m 行乘 m（不是简单取窗口）
//!   2 morning   = a * nums.where(nums <= 120, nan)
//!   3 afternoon = a * nums.where(nums >  120, nan)
//!   4 close30   = a * nums.where(nums >  210, nan)
//!   5 amountbig = a * amounts_big     （amount_raw - 该股当日列中位数 > 0）
//!   6 amountsmall = a * amounts_small （< 0）
//!
//! ════════════════════════════════════════════════════════════════════════
//! 三、旧代码逐步对照
//! ════════════════════════════════════════════════════════════════════════
//!   nums = self.nums.reset_index(drop=True)   ← 240×n，第 r 行全是 r+1
//!   one1(do):
//!       do = do.reset_index(drop=True)
//!       corr1 = do.corrwith(nums)             ← 逐股票与「分钟序号」的 Pearson（两遍法）
//!       do    = do / do.rolling(5).mean()     ← 除以自身 5 行滚动均值（min_periods=5，Kahan）
//!       corr2 = do.corrwith(nums)
//!       return corr2.rank(pct=True) - corr1.abs().rank(pct=True)
//!   每段对 buy / sell 各调一次 one1，拼成 [buy, sell]。
//!
//!   corrwith(other) 在 pandas 里的实现（frame.py，method='pearson' + other 是 DataFrame）：
//!       left  = left + right*0      # 把两边 NaN 的位置并集化
//!       right = right + left*0
//!       ldem  = left  - left.mean()
//!       rdem  = right - right.mean()
//!       num   = (ldem * rdem).sum()
//!       dom   = (left.count() - 1) * left.std() * right.std()
//!       res   = num / dom
//!   其中 mean/std 都是 skipna；std 是 ddof=1 的两遍法
//!   （nanvar：avg = Σv/count，再 Σ(avg-v)²/(count-1)，NaN 位置记 0）。
//!   逐列求和（axis=0）在 C 顺序多列数组上是**顺序累加**，不是成对求和。
//!
//!   corr1.abs() 后 rank(pct=True)：先按平均名次升序排名（NaN 保留），再除以该列非 NaN 的个数。
//!
//! ════════════════════════════════════════════════════════════════════════
//! 四、输出排布
//! ════════════════════════════════════════════════════════════════════════
//!   vals 股票优先：第 i 只股票连续 14 个值 = 7 段 × (buy, sell)。

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, LazyLock, Mutex};

use rayon::prelude::*;

const MIN_PER_DAY: usize = 240;
const MIN_NAN_FROM: usize = 237;
pub const N_OUT: usize = 14;
const PER_SEG: usize = 2;

const F_AMOUNT: usize = 2;
const FIELDS: [&str; 3] = ["act_buy_amount_sum", "act_sell_amount_sum", "amount"];
const IDX_BUY: usize = 0;
const IDX_SELL: usize = 1;

const SEGS: [&str; 7] = [
    "allday",
    "open30",
    "morning",
    "afternoon",
    "close30",
    "amountbig",
    "amountsmall",
];
const BASE: [&str; 2] = ["stranded_trades_buy", "stranded_trades_sell"];

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
// 分钟数据读取
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

/// get_many=4 前 5 段的分钟序号权重（amountbig/amountsmall 由调用方另行处理）
fn minute_weight(seg: usize, row: usize) -> f64 {
    let m = (row + 1) as f64;
    let ok = match seg {
        0 => true,
        1 => row + 1 <= 30,
        2 => row + 1 <= 120,
        3 => row + 1 > 120,
        _ => row + 1 > 210,
    };
    if !ok {
        return f64::NAN;
    }
    if seg == 0 {
        1.0
    } else {
        m
    }
}

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

/// 逐列中位数（跳过 NaN，偶数取中间两个的均值）
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

/// 一维 rank（method='average'、升序、NaN 保留）；pct=true 时再除以非 NaN 个数
fn rank_average_pct(values: &[f64], pct: bool) -> Vec<f64> {
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
    if pct {
        let cnt = idx.len() as f64;
        if cnt > 0.0 {
            for v in out.iter_mut() {
                if !v.is_nan() {
                    *v /= cnt;
                }
            }
        }
    }
    out
}

/// 逐列顺序累加（跳过 NaN）
#[inline]
fn col_nansum(v: &[f64]) -> f64 {
    let mut s = 0.0f64;
    for &x in v {
        if !x.is_nan() {
            s += x;
        }
    }
    s
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
// rolling(5).mean()（pandas roll_mean 的 Kahan 逐字复刻）
// ============================================================================

/// pandas aggregations.pyx: add_mean / remove_mean / calc_mean 的逐字转写。
#[derive(Clone, Copy)]
struct RollState {
    nobs: i64,
    sum_x: f64,
    neg_ct: i64,
    compensation: f64,
    num_consecutive_same_value: i64,
    prev_value: f64,
}

impl RollState {
    fn new() -> Self {
        RollState {
            nobs: 0,
            sum_x: 0.0,
            neg_ct: 0,
            compensation: 0.0,
            num_consecutive_same_value: 0,
            prev_value: 0.0,
        }
    }
    #[inline]
    fn add(&mut self, val: f64) {
        if val == val {
            self.nobs += 1;
            let y = val - self.compensation;
            let t = self.sum_x + y;
            self.compensation = t - self.sum_x - y;
            self.sum_x = t;
            if val.is_sign_negative() {
                self.neg_ct += 1;
            }
            if val == self.prev_value {
                self.num_consecutive_same_value += 1;
            } else {
                self.num_consecutive_same_value = 1;
            }
            self.prev_value = val;
        }
    }
    #[inline]
    fn remove(&mut self, val: f64) {
        if val == val {
            self.nobs -= 1;
            let y = -val - self.compensation;
            let t = self.sum_x + y;
            self.compensation = t - self.sum_x - y;
            self.sum_x = t;
            if val.is_sign_negative() {
                self.neg_ct -= 1;
            }
        }
    }
    #[inline]
    fn mean(&self, minp: i64) -> f64 {
        if self.nobs >= minp && self.nobs > 0 {
            let mut result = self.sum_x / (self.nobs as f64);
            if self.num_consecutive_same_value >= self.nobs {
                result = self.prev_value;
            } else if self.neg_ct == 0 && result < 0.0 {
                result = 0.0;
            } else if self.neg_ct == self.nobs && result > 0.0 {
                result = 0.0;
            }
            result
        } else {
            f64::NAN
        }
    }
}

/// 固定窗口 window、min_periods=window 的滚动均值，逐字复刻 roll_mean 的增量路径。
/// values 长度 N（一天 240 行），输出同长。
/// 注意 pandas `_prep_values` 会先把 inf 换成 NaN（rolling.py 第 375-378 行），这里照做。
fn roll_mean(vals: &[f64], window: usize) -> Vec<f64> {
    let vals: Vec<f64> = vals
        .iter()
        .map(|&x| if x.is_infinite() { f64::NAN } else { x })
        .collect();
    let vals: &[f64] = &vals;
    let n = vals.len();
    let minp = window as i64;
    let mut out = vec![f64::NAN; n];
    let mut st_add = RollState::new();
    let mut st_remove = RollState::new();
    for i in 0..n {
        let s = i.saturating_sub(window - 1);
        let e = i + 1;
        let prev_s = if i == 0 { 0 } else { (i - 1).saturating_sub(window - 1) };
        let prev_e = i;
        if i == 0 || s >= prev_e {
            st_add = RollState::new();
            st_remove = RollState::new();
            st_add.prev_value = vals[s];
            for j in s..e {
                st_add.add(vals[j]);
            }
        } else {
            for j in prev_s..s {
                st_remove.remove(vals[j]);
            }
            for j in prev_e..e {
                st_add.add(vals[j]);
            }
        }
        out[i] = st_add.mean(minp);
    }
    out
}

// ============================================================================
// corrwith：逐股票与「分钟序号 1..240」的 Pearson
// ============================================================================

/// 对一只股票的 240 个值，算 pandas DataFrame.corrwith(DataFrame) 的 pearson 结果。
fn corrwith_nums(do_col: &[f64]) -> f64 {
    // left = left + right*0 ：right（分钟序号）无 NaN，所以只需按 left 的 NaN 掩码
    let mut cnt = 0usize;
    let mut sl = 0.0f64;
    let mut sr = 0.0f64;
    for t in 0..MIN_PER_DAY {
        let l = do_col[t];
        if !l.is_nan() {
            cnt += 1;
            sl += l;
            sr += (t + 1) as f64;
        }
    }
    if cnt == 0 {
        return f64::NAN;
    }
    let c = cnt as f64;
    let lm = sl / c;
    let rm = sr / c;
    let mut num = 0.0f64;
    let mut sql = 0.0f64;
    let mut sqr = 0.0f64;
    for t in 0..MIN_PER_DAY {
        let l = do_col[t];
        if !l.is_nan() {
            let r = (t + 1) as f64;
            num += (l - lm) * (r - rm);
            let dl = lm - l;
            let dr = rm - r;
            sql += dl * dl;
            sqr += dr * dr;
        }
    }
    let d = c - 1.0;
    let dom = d * (sql / d).sqrt() * (sqr / d).sqrt();
    num / dom
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

    let med_amount = column_median(&raw[F_AMOUNT], n);
    let mut big = vec![f64::NAN; MIN_PER_DAY * n];
    let mut small = vec![f64::NAN; MIN_PER_DAY * n];
    for row in 0..MIN_PER_DAY {
        for j in 0..n {
            let m = med_amount[j];
            if m.is_nan() {
                continue;
            }
            let d = raw[F_AMOUNT][row * n + j] - m;
            if d > 0.0 {
                big[row * n + j] = 1.0;
            } else if d < 0.0 {
                small[row * n + j] = 1.0;
            }
        }
    }

    let mut out = vec![f64::NAN; n * N_OUT];
    for seg in 0..SEGS.len() {
        let w: Vec<f64> = (0..MIN_PER_DAY)
            .map(|row| minute_weight(seg, row))
            .collect();
        let (buy, sell) = match seg {
            0..=4 => (
                apply_row_weight(&raw[IDX_BUY], n, &w),
                apply_row_weight(&raw[IDX_SELL], n, &w),
            ),
            5 => (
                apply_elem_mask(&raw[IDX_BUY], &big),
                apply_elem_mask(&raw[IDX_SELL], &big),
            ),
            _ => (
                apply_elem_mask(&raw[IDX_BUY], &small),
                apply_elem_mask(&raw[IDX_SELL], &small),
            ),
        };

        for (k, mat) in [(0usize, &buy), (1usize, &sell)] {
            // 1) 与分钟序号的相关系数
            let corr1: Vec<f64> = (0..n)
                .into_par_iter()
                .map(|j| {
                    let col: Vec<f64> = (0..MIN_PER_DAY).map(|r| mat[r * n + j]).collect();
                    corrwith_nums(&col)
                })
                .collect();
            // 2) 除以自身 5 行滚动均值
            let mut ratio = vec![f64::NAN; MIN_PER_DAY * n];
            let cols: Vec<(usize, Vec<f64>)> = (0..n)
                .into_par_iter()
                .map(|j| {
                    let col: Vec<f64> = (0..MIN_PER_DAY).map(|r| mat[r * n + j]).collect();
                    let rm = roll_mean(&col, 5);
                    let mut v = vec![f64::NAN; MIN_PER_DAY];
                    for r in 0..MIN_PER_DAY {
                        v[r] = col[r] / rm[r];
                    }
                    (j, v)
                })
                .collect();
            for (j, v) in cols {
                for r in 0..MIN_PER_DAY {
                    ratio[r * n + j] = v[r];
                }
            }
            let corr2: Vec<f64> = (0..n)
                .into_par_iter()
                .map(|j| {
                    let col: Vec<f64> = (0..MIN_PER_DAY).map(|r| ratio[r * n + j]).collect();
                    corrwith_nums(&col)
                })
                .collect();
            let r2 = rank_average_pct(&corr2, true);
            let abs1: Vec<f64> = corr1.iter().map(|v| v.abs()).collect();
            let r1 = rank_average_pct(&abs1, true);
            for j in 0..n {
                out[j * N_OUT + seg * PER_SEG + k] = finite_or_nan(r2[j] - r1[j]);
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
        assert_eq!(n[0], "stranded_trades_buy_allday");
        assert_eq!(n[1], "stranded_trades_sell_allday");
        assert_eq!(n[2], "stranded_trades_buy_open30");
        assert_eq!(n[6], "stranded_trades_buy_afternoon");
        assert_eq!(n[8], "stranded_trades_buy_close30");
        assert_eq!(n[10], "stranded_trades_buy_amountbig");
        assert_eq!(n[12], "stranded_trades_buy_amountsmall");
    }

    #[test]
    fn minute_weights() {
        assert_eq!(minute_weight(0, 0), 1.0);
        assert_eq!(minute_weight(1, 29), 30.0);
        assert!(minute_weight(1, 30).is_nan());
        assert_eq!(minute_weight(2, 119), 120.0);
        assert!(minute_weight(3, 119).is_nan());
        assert_eq!(minute_weight(4, 210), 211.0);
    }
}
