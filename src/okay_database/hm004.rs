//! hm4（turning_bet）—— 旧框架（design_whatever_out.FactorDesign，**分钟**数据）的 Rust 复现。
//!
//! 旧文件：/home/chenzongwei/pythoncode/成果hm4_turning_bet/hm4_okay.py
//!   run_base 的 Go.get_factor，参数 get_many=**0** / read_full=0。
//!   get_many=0 且 read_full=0 ⇒ new_get_field_min_data 直接返回单个 DataFrame（没有分段、
//!   没有分钟序号加权），一天一次全市场 ⇒ compute_cross。
//!
//! ════════════════════════════════════════════════════════════════════════
//! 一、names
//! ════════════════════════════════════════════════════════════════════════
//!   names = ['turning_bet_5','turning_bet_7','turning_bet_9','turning_bet_11','turning_bet_13']
//!   与 get_factor 返回的 [two(5),two(7),two(9),two(11),two(13)] 一一对应。共 5 个。
//!
//! ════════════════════════════════════════════════════════════════════════
//! 二、旧代码逐步对照
//! ════════════════════════════════════════════════════════════════════════
//!   two(param):  shift = param // 2
//!     one(price, do1, do2, is_high):
//!       is_high 时：end_daily = price - price.rolling(param).max().shift(-shift)
//!                   is_end    = np.where(end_daily >= 0, 1, nan)
//!       否则     ：end_daily = price - price.rolling(param).min().shift(-shift)
//!                   is_end    = np.where(end_daily <= 0, 1, nan)
//!       do_left1  = do1.rolling(shift).sum().shift(1)
//!       do_right1 = do1.rolling(shift).sum().shift(-shift)
//!       do_left2  = do2.rolling(shift).sum().shift(1)
//!       do_right2 = do2.rolling(shift).sum().shift(-shift)
//!       lr_rate = do_left1/(do_left1+do_left2)
//!                 / ( (do_right1/(do_right1+do_right2)) + do_left1/(do_left1+do_left2) )
//!       lr_rate = lr_rate * is_end                 # DataFrame * ndarray，逐格
//!       return lr_rate.quantile(0.5).rank()        # 每只股票 240 行上的中位数，再跨股票排名
//!     return (one(high,sell,buy,1) - one(low,buy,sell,0)).to_frame(end_date).T
//!   return [two(5), two(7), two(9), two(11), two(13)]
//!
//!   rolling 的默认 min_periods = window，所以窗口内必须**全非 NaN** 才出值。
//!
//! ════════════════════════════════════════════════════════════════════════
//! 三、数值口径（照抄 pandas window/aggregations.pyx）
//! ════════════════════════════════════════════════════════════════════════
//! · roll_sum 用 Kahan 补偿的增量 add/remove（compensation_add 跨迭代保留），
//!   并带 `num_consecutive_same_value >= nobs → prev_value*nobs` 与 `nobs==0 → 0` 两条规则，
//!   本文件的 RollSum 是它的逐字转写。
//! · roll_max / roll_min 是单调队列；min_periods==window 时非 NaN 计数必须满窗口，
//!   所以等价于「窗口全非 NaN 才取窗口最大值」，本文件按这个等价实现。
//! · rolling 前 pandas `_prep_values` 会把 inf 换成 NaN，本文件照做。
//! · 分位数：np.percentile(non-NaN, 50, method='linear')。
//! · rank：pandas Series.rank()，升序、并列取平均、NaN 保留。
//!
//! ════════════════════════════════════════════════════════════════════════
//! 四、输出排布
//! ════════════════════════════════════════════════════════════════════════
//!   vals 股票优先：第 i 只股票连续 5 个值，顺序 [5, 7, 9, 11, 13]。

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, LazyLock, Mutex};

use rayon::prelude::*;

const MIN_PER_DAY: usize = 240;
const MIN_NAN_FROM: usize = 237;
pub const N_OUT: usize = 5;

const F_HIGH: usize = 0;
const F_LOW: usize = 1;
const F_BUY: usize = 2;
const F_SELL: usize = 3;
const FIELDS: [&str; 4] = ["high", "low", "act_buy_amount_sum", "act_sell_amount_sum"];
const PARAMS: [usize; 5] = [5, 7, 9, 11, 13];

pub fn names() -> Vec<String> {
    PARAMS.iter().map(|p| format!("turning_bet_{}", p)).collect()
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
// rolling：pandas window/aggregations.pyx 的逐字复刻
// ============================================================================

/// pandas `_prep_values`：rolling 之前把 inf 换成 NaN
#[inline]
fn prep(vals: &[f64]) -> Vec<f64> {
    vals.iter()
        .map(|&x| if x.is_infinite() { f64::NAN } else { x })
        .collect()
}

/// 固定窗口、min_periods=window 的滚动求和（Kahan 补偿 + 全同值回退）
fn roll_sum(vals: &[f64], window: usize) -> Vec<f64> {
    let v = prep(vals);
    let n = v.len();
    let minp = window as i64;
    let mut out = vec![f64::NAN; n];
    let mut sum_x = 0.0f64;
    let mut comp_add = 0.0f64;
    let mut comp_rem = 0.0f64;
    let mut nobs: i64 = 0;
    let mut prev_value = 0.0f64;
    let mut n_same: i64 = 0;
    for i in 0..n {
        let s = i.saturating_sub(window - 1);
        let e = i + 1;
        let prev_s = if i == 0 { 0 } else { (i - 1).saturating_sub(window - 1) };
        let prev_e = i;
        if i == 0 || s >= prev_e {
            prev_value = v[s];
            n_same = 0;
            sum_x = 0.0;
            comp_add = 0.0;
            comp_rem = 0.0;
            nobs = 0;
            for j in s..e {
                let val = v[j];
                if val == val {
                    nobs += 1;
                    let y = val - comp_add;
                    let t = sum_x + y;
                    comp_add = t - sum_x - y;
                    sum_x = t;
                    if val == prev_value {
                        n_same += 1;
                    } else {
                        n_same = 1;
                    }
                    prev_value = val;
                }
            }
        } else {
            for j in prev_s..s {
                let val = v[j];
                if val == val {
                    nobs -= 1;
                    let y = -val - comp_rem;
                    let t = sum_x + y;
                    comp_rem = t - sum_x - y;
                    sum_x = t;
                }
            }
            for j in prev_e..e {
                let val = v[j];
                if val == val {
                    nobs += 1;
                    let y = val - comp_add;
                    let t = sum_x + y;
                    comp_add = t - sum_x - y;
                    sum_x = t;
                    if val == prev_value {
                        n_same += 1;
                    } else {
                        n_same = 1;
                    }
                    prev_value = val;
                }
            }
        }
        out[i] = if nobs == 0 {
            0.0
        } else if nobs >= minp {
            if n_same >= nobs {
                prev_value * (nobs as f64)
            } else {
                sum_x
            }
        } else {
            f64::NAN
        };
    }
    out
}

/// 固定窗口、min_periods=window 的滚动最大/最小。
/// nobs 必须满窗口（min_periods==window）⇒ 窗口全非 NaN 才出值。
fn roll_maxmin(vals: &[f64], window: usize, is_max: bool) -> Vec<f64> {
    let v = prep(vals);
    let n = v.len();
    let mut out = vec![f64::NAN; n];
    for i in 0..n {
        let s = i.saturating_sub(window - 1);
        let e = i + 1;
        if e - s < window {
            continue;
        }
        let mut acc = if is_max { f64::NEG_INFINITY } else { f64::INFINITY };
        let mut ok = true;
        for j in s..e {
            let x = v[j];
            if x.is_nan() {
                ok = false;
                break;
            }
            if is_max {
                if x > acc {
                    acc = x;
                }
            } else if x < acc {
                acc = x;
            }
        }
        out[i] = if ok { acc } else { f64::NAN };
    }
    out
}

/// DataFrame.shift(k)：正数向后（下）移，负数向前（上）移
fn shift_vec(v: &[f64], k: i64) -> Vec<f64> {
    let n = v.len() as i64;
    let mut out = vec![f64::NAN; v.len()];
    for i in 0..n {
        let src = i - k;
        if src >= 0 && src < n {
            out[i as usize] = v[src as usize];
        }
    }
    out
}

/// np.percentile(non-NaN, 100*q, method='linear')
fn quantile_linear(values: &[f64], q: f64) -> f64 {
    let mut v: Vec<f64> = values.iter().copied().filter(|x| !x.is_nan()).collect();
    let n = v.len();
    if n == 0 {
        return f64::NAN;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let virt = q * (n as f64 - 1.0);
    let prev = virt.floor();
    let mut lo = prev as i64;
    let mut hi = lo + 1;
    if lo < 0 {
        lo = 0;
    }
    if hi > n as i64 - 1 {
        hi = n as i64 - 1;
    }
    let gamma = virt - prev;
    let a = v[lo as usize];
    let b = v[hi as usize];
    a + gamma * (b - a)
}

/// 一维 rank（升序、并列取平均、NaN 保留）
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

/// 取二维矩阵的第 j 列（行优先 240×n）
#[inline]
fn col(mat: &[f64], n: usize, j: usize) -> Vec<f64> {
    (0..MIN_PER_DAY).map(|r| mat[r * n + j]).collect()
}

#[inline]
fn finite_or_nan(v: f64) -> f64 {
    if v.is_finite() {
        v
    } else {
        f64::NAN
    }
}

/// one(price, do1, do2, is_high) 对一只股票算出的标量
fn one_stock(price: &[f64], do1: &[f64], do2: &[f64], is_high: bool, param: usize) -> f64 {
    let shift = param / 2;
    let p = prep(price);
    let ext = if is_high {
        roll_maxmin(&p, param, true)
    } else {
        roll_maxmin(&p, param, false)
    };
    let ext_sh = shift_vec(&ext, -(shift as i64));
    let mut is_end = vec![f64::NAN; MIN_PER_DAY];
    for t in 0..MIN_PER_DAY {
        let d = p[t] - ext_sh[t];
        let hit = if is_high { d >= 0.0 } else { d <= 0.0 };
        if hit {
            is_end[t] = 1.0;
        }
    }
    let s1 = roll_sum(do1, shift);
    let s2 = roll_sum(do2, shift);
    let dl1 = shift_vec(&s1, 1);
    let dr1 = shift_vec(&s1, -(shift as i64));
    let dl2 = shift_vec(&s2, 1);
    let dr2 = shift_vec(&s2, -(shift as i64));
    let mut lr = vec![f64::NAN; MIN_PER_DAY];
    for t in 0..MIN_PER_DAY {
        let a = dl1[t];
        let b = dl2[t];
        let c = dr1[t];
        let d = dr2[t];
        let left = a / (a + b);
        let right = c / (c + d);
        lr[t] = (left / (right + left)) * is_end[t];
    }
    quantile_linear(&lr, 0.5)
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

    // 每个 param：先各自对 high / low 的那串标量跨股票排名，再相减
    // （旧代码是 one(...).rank() - one(...).rank()，rank 必须在各 Series 上分别做）
    let mut out = vec![f64::NAN; n * N_OUT];
    for (k, &param) in PARAMS.iter().enumerate() {
        let hi_vals: Vec<(usize, f64)> = (0..n)
            .into_par_iter()
            .map(|j| {
                let high = col(&raw[F_HIGH], n, j);
                let buy = col(&raw[F_BUY], n, j);
                let sell = col(&raw[F_SELL], n, j);
                (j, one_stock(&high, &sell, &buy, true, param))
            })
            .collect();
        let lo_vals: Vec<(usize, f64)> = (0..n)
            .into_par_iter()
            .map(|j| {
                let low = col(&raw[F_LOW], n, j);
                let buy = col(&raw[F_BUY], n, j);
                let sell = col(&raw[F_SELL], n, j);
                (j, one_stock(&low, &buy, &sell, false, param))
            })
            .collect();
        let mut hv = vec![f64::NAN; n];
        let mut lv = vec![f64::NAN; n];
        for (j, v) in hi_vals {
            hv[j] = v;
        }
        for (j, v) in lo_vals {
            lv[j] = v;
        }
        let rh = rank_average(&hv);
        let rl = rank_average(&lv);
        for j in 0..n {
            out[j * N_OUT + k] = finite_or_nan(rh[j] - rl[j]);
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
        assert_eq!(n, vec!["turning_bet_5", "turning_bet_7", "turning_bet_9", "turning_bet_11", "turning_bet_13"]);
    }

    #[test]
    fn roll_sum_kahan_basic() {
        let v = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let s = roll_sum(&v, 3);
        assert!(s[0].is_nan());
        assert!(s[1].is_nan());
        assert_eq!(s[2], 6.0);
        assert_eq!(s[5], 15.0);
    }

    #[test]
    fn roll_max_full_window_only() {
        let v = [1.0, 5.0, f64::NAN, 3.0];
        let m = roll_maxmin(&v, 2, true);
        assert!(m[0].is_nan());
        assert_eq!(m[1], 5.0);
        assert!(m[2].is_nan());
        assert!(m[3].is_nan());
    }

    #[test]
    fn shift_direction() {
        let v = [1.0, 2.0, 3.0, 4.0];
        assert_eq!(shift_vec(&v, 1), vec![f64::NAN, 1.0, 2.0, 3.0]);
        let s = shift_vec(&v, -1);
        assert_eq!(s[0], 2.0);
        assert_eq!(s[2], 4.0);
        assert!(s[3].is_nan());
    }
}
