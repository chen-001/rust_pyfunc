//! hm026 兴业转移熵 —— 旧框架（design_whatever_out.TickDataBase）的 Rust 复现。
//!
//! ════════════════════════════════════════════════════════════════════════
//! 旧代码要点（逐条对齐 hm26_okay.py 的 run_base）
//! ════════════════════════════════════════════════════════════════════════
//!
//! 【框架参数】get_many=0 / read_full=0 / change_time=1 / cross_section=True。
//!   Go 继承 dwo.TickDataBase，get_factor 里：
//!     df = self.read_trade_data_full_market(lazy=0).filter(pl.col('flag')!=32)
//!   注意 read_trade_data_full_market 忽略 symbol 参数，一次读**当天全市场**的
//!   transaction CSV。但后面所有指标都是 `group_by("code")` 或按列（分钟宽表）算的，
//!   **每只股票的结果只取决于它自己的数据**，与其他股票无关。所以逐股复现是等价的。
//!
//! 【成交读入与时间】旧代码的读入链：
//!   pl.read_csv(..., schema=dtypes, ignore_errors=True).sort(["index"])
//!   → exchtime.round(0)
//!   → deal_data(via_polars=1)：exchtime 变成 Datetime = 1970-01-01 08:00:00 + exchtime 微秒
//!   → filter(flag != 32)
//!   Rust 侧用 read_trade_fast_inner(code, date, true, false, usize::MAX) 拿到同一批行
//!   （time_us 就是「epoch 微秒 + 8 小时」，和上面的 Datetime 完全对应），
//!   然后自己按 flag != 32 过滤（库里那个 with_retreat 判断对 "32.0" 不生效）。
//!
//! 【分钟数据】read_minute_data 走的是 FactorDesign → MinFactorReader，
//!   数据源写死在 /ssd_data/data/1min_factor_text（与传进来的 hdf5_dir 无关），
//!   每天 240 根 K 线，并且 get_field_min_data 把 index % 240 >= 237 的三根置 NaN。
//!   注意库里的 read_minute_field 是把**最后 4 根**置 NaN，多灭了一根 236，
//!   这里不能直接用它，必须自己按 237..239 置 NaN。
//!
//! 【输出 97 个值】分成五段，顺序就是旧代码 names_single 的顺序：
//!   1) one(df)   7 个：每约 1% 成交量刻度处的「间隔秒数 / 涨跌幅」统计
//!   2) two(df)   4 个：把间隔秒数按 1/10/30/60 秒分桶、乘上价格涨跌方向后的熵
//!   3) three()   4 个：日内分钟收益×量占比的同号连段，取「负转正且合计为负」的幅度统计
//!   4) four()    2 个：振幅与分钟收益的相关、分钟量与正收益的相关
//!   5) five()   80 个：分钟收益与分钟量的双向转移熵（k=1..10 × c=2/5/10/20）
//!
//! 【段 1/2 的「刻度行」语义】旧代码把原始逐笔的 volume 列先换成「累计成交量占比」，
//!   再造 99 行 (code, k/100) 的刻度行，拼起来按 (code, volume) 排序，
//!   对 cut 列 forward_fill(limit=1) 后 drop_nulls(["cut","exchtime"])。
//!   实质是：某一笔成交留下 ⟺ 排在它前面紧挨着的那一行是刻度行。
//!   等价条件（累计占比单调不减）：设 k* 为满足 k/100 < frac_i 的最大 k ∈ [1,99]，
//!   当且仅当 frac_{i-1} < k*/100 时第 i 笔留下。见 select_percent_rows。
//!
//! 【段 1/2 里的 dtype】volume 是 Float32，cum_sum 和 sum 都在 Float32 里算，
//!   所以累计占比是 f32（不是 f64）。price 是 Float32，pct_change 用
//!   (x - prev)/prev 也在 f32 里算。这些都必须照做，否则选行会漂。
//!
//! 【已知与旧代码的差异】旧代码的 .round() 是 polars 默认的「四舍六入五取偶」，
//!   这里用 round_half_even 对齐。

use crate::fast_csv_reader::{read_trade_fast_inner, TradeRecord};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, LazyLock, Mutex};

/// 输出因子个数（= 旧代码 names_single 的长度）
pub const N_OUT: usize = 97;
/// 分钟数据每天 240 根 K 线
const MIN_PER_DAY: usize = 240;
/// 旧框架 get_field_min_data 把每根日线索引 % 240 >= 237 的三根置 NaN
const MIN_NAN_FROM: usize = 237;

// ============================================================================
// 因子名
// ============================================================================

/// 转移熵的参数网格：k = 1..10，c ∈ {2,5,10,20}，共 40 个
const TE_KS: [usize; 10] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
const TE_CS: [usize; 4] = [2, 5, 10, 20];

pub fn names() -> Vec<String> {
    let mut out: Vec<String> = [
        "time_std",
        "time_mean",
        "price_std",
        "price_mean",
        "time_corr",
        "price_corr",
        "corr_pe",
        "buc_entropy1",
        "buc_entropy10",
        "buc_entropy30",
        "buc_entropy60",
        "negv_mean",
        "negv_max",
        "negvwgt_mean",
        "negvwgt_max",
        "rhl_cor",
        "voh_cor",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    for prefix in ["ret_to_vol", "vol_to_ret"] {
        for k in TE_KS.iter() {
            for c in TE_CS.iter() {
                out.push(format!("{prefix}_k{k}_c{c}"));
            }
        }
    }
    debug_assert_eq!(out.len(), N_OUT);
    out
}

// ============================================================================
// 分钟数据：单股单日、单列读取
// ============================================================================

struct MinMeta {
    date_to_dayidx: HashMap<i64, usize>,
    code_to_col: HashMap<String, usize>,
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
    let mut code_to_col = HashMap::new();
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
        let code = format!("{:0>6}", parts[0].trim());
        code_to_col.insert(code, pos);
    }
    Some(MinMeta {
        date_to_dayidx,
        code_to_col,
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

/// 当天该股在分钟数据里的列号（不在名单里返回 None）
fn minute_col_of(code: &str) -> Option<usize> {
    let meta = min_meta()?;
    meta.code_to_col.get(code).copied()
}

/// 日期在分钟数据里的 day_idx
fn minute_dayidx(date: i64) -> Option<usize> {
    let meta = min_meta()?;
    meta.date_to_dayidx.get(&date).copied()
}

#[cfg(feature = "hdf5")]
mod h5col {
    use super::*;
    use hdf5_metno as hdf5;
    use ndarray::s;

    thread_local! {
        static MIN_FILES: std::cell::RefCell<HashMap<String, hdf5::File>> =
            std::cell::RefCell::new(HashMap::new());
    }

    /// 读某字段某股某天的 240 根分钟值，最后三根（索引 237..239）置 NaN。
    pub fn read_col(field: &str, day_idx: usize, col: usize) -> Option<Vec<f64>> {
        let path = crate::data_paths::minute_file(&format!("{field}.h5"));
        let key = path.to_string_lossy().into_owned();
        let mut out: Option<Vec<f64>> = None;
        MIN_FILES.with(|cell| {
            let mut cache = cell.borrow_mut();
            if !cache.contains_key(&key) {
                match hdf5::File::open(&path) {
                    Ok(f) => {
                        cache.insert(key.clone(), f);
                    }
                    Err(_) => return,
                }
            }
            let file = match cache.get(&key) {
                Some(f) => f,
                None => return,
            };
            let ds = match file.dataset("data") {
                Ok(d) => d,
                Err(_) => return,
            };
            let shape = ds.shape();
            if shape.len() != 2 || col >= shape[1] {
                return;
            }
            let nrows = shape[0];
            let rs = day_idx * MIN_PER_DAY;
            if rs >= nrows {
                return;
            }
            let re = std::cmp::min(rs + MIN_PER_DAY, nrows);
            let arr: ndarray::Array2<f64> = match ds.read_slice_2d(s![rs..re, col..col + 1]) {
                Ok(a) => a,
                Err(_) => return,
            };
            let mut v: Vec<f64> = arr.column(0).to_vec();
            v.resize(MIN_PER_DAY, f64::NAN);
            for i in MIN_NAN_FROM..MIN_PER_DAY {
                v[i] = f64::NAN;
            }
            out = Some(v);
        });
        out
    }
}

#[cfg(not(feature = "hdf5"))]
mod h5col {
    pub fn read_col(_field: &str, _day_idx: usize, _col: usize) -> Option<Vec<f64>> {
        None
    }
}

/// 一次性取该股的 close / volume / high / low 四列分钟值
fn minute_quads(code: &str, date: i64) -> Option<[Vec<f64>; 4]> {
    let day = minute_dayidx(date)?;
    let col = minute_col_of(code)?;
    let close = h5col::read_col("close", day, col)?;
    let vol = h5col::read_col("volume", day, col)?;
    let high = h5col::read_col("high", day, col)?;
    let low = h5col::read_col("low", day, col)?;
    Some([close, vol, high, low])
}

// ============================================================================
// 通用小工具
// ============================================================================

/// polars 的 .round() 默认是 half-to-even
fn round_half_even(x: f64) -> f64 {
    let f = x.floor();
    let d = x - f;
    if d > 0.5 {
        f + 1.0
    } else if d < 0.5 {
        f
    } else if (f as i64) % 2 == 0 {
        f
    } else {
        f + 1.0
    }
}

fn mean_of(v: &[f64]) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    let mut s = 0.0;
    for &x in v {
        s += x;
    }
    s / v.len() as f64
}

/// 样本标准差（ddof=1），与 polars / pandas 的默认一致
fn sample_std(v: &[f64]) -> f64 {
    let n = v.len();
    if n < 2 {
        return f64::NAN;
    }
    let m = mean_of(v);
    let mut s = 0.0;
    for &x in v {
        let d = x - m;
        s += d * d;
    }
    (s / (n as f64 - 1.0)).sqrt()
}

/// 成对完整观测的 Pearson 相关；任一为 NaN 的对丢掉；有效对 < 2 或某侧方差为 0 返回 NaN
fn pearson_pairwise(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len().min(y.len());
    let mut xs: Vec<f64> = Vec::with_capacity(n);
    let mut ys: Vec<f64> = Vec::with_capacity(n);
    for i in 0..n {
        if !x[i].is_nan() && !y[i].is_nan() {
            xs.push(x[i]);
            ys.push(y[i]);
        }
    }
    let k = xs.len();
    if k < 2 {
        return f64::NAN;
    }
    let mx = mean_of(&xs);
    let my = mean_of(&ys);
    let mut sxy = 0.0;
    let mut sxx = 0.0;
    let mut syy = 0.0;
    for i in 0..k {
        let a = xs[i] - mx;
        let b = ys[i] - my;
        sxy += a * b;
        sxx += a * a;
        syy += b * b;
    }
    if !(sxx > 0.0) || !(syy > 0.0) {
        return f64::NAN;
    }
    sxy / (sxx.sqrt() * syy.sqrt())
}

/// 序列与其滞后一项的 Pearson 相关（对应 polars 的 pl.corr(col, col.shift(1))）
fn lag1_corr(v: &[f64]) -> f64 {
    if v.len() < 2 {
        return f64::NAN;
    }
    let a = &v[1..];
    let b = &v[..v.len() - 1];
    pearson_pairwise(a, b)
}

// ============================================================================
// 段 1 / 段 2：按 1% 成交量刻度抽行，再算统计量与分桶熵
// ============================================================================

/// 选中「累计成交量占比跨过 1% 刻度」的那一笔，返回 (time_us, price)。
///
/// 见文件头【段 1/2 的「刻度行」语义】。累计占比用 Float32 算，与 polars 一致。
fn select_percent_rows(rows: &[&TradeRecord]) -> Vec<(i64, f32)> {
    let mut total: f32 = 0.0;
    for r in rows.iter() {
        total += r.volume;
    }
    if !(total > 0.0) || !total.is_finite() {
        return Vec::new();
    }
    let mut out: Vec<(i64, f32)> = Vec::with_capacity(128);
    let mut cum: f32 = 0.0;
    let mut prev_frac = f64::NEG_INFINITY;
    for r in rows.iter() {
        cum += r.volume;
        let frac = (cum / total) as f64;
        if !frac.is_finite() {
            continue;
        }
        // k* = 满足 k/100.0 < frac 的最大 k ∈ [0,99]
        let mut k = (frac * 100.0).floor() as i64;
        if k > 99 {
            k = 99;
        }
        if k < 0 {
            k = 0;
        }
        while k < 99 && ((k + 1) as f64) / 100.0 < frac {
            k += 1;
        }
        while k > 0 && (k as f64) / 100.0 >= frac {
            k -= 1;
        }
        if k >= 1 && prev_frac < (k as f64) / 100.0 {
            out.push((r.time_us, r.price));
        }
        prev_frac = frac;
    }
    out
}

/// 段 1：7 个量。
///
/// 旧代码在选中行之后：
///   exchtime ← diff().dt.total_seconds()  （Int64，截断到整秒）
///   price    ← pct_change()               （Float32，(x-prev)/prev）
///   把 price 的 ±inf 换成 null，drop_nulls() 再 drop_nans()
/// 然后按 code 聚合。
fn stage_one(kept: &[(i64, f32)]) -> [f64; 7] {
    let mut sec: Vec<f64> = Vec::new();
    let mut pct: Vec<f64> = Vec::new();
    for i in 1..kept.len() {
        let d_us = kept[i].0 - kept[i - 1].0;
        let s = (d_us / 1_000_000) as f64;
        let p = (kept[i].1 - kept[i - 1].1) / kept[i - 1].1; // f32 运算
        let pf = p as f64;
        if pf.is_finite() {
            sec.push(s);
            pct.push(pf);
        }
    }
    // 注意不要在这里提前返回 NaN：只剩 1 行有效观测时 polars 仍会给出 time_mean，
    // 其余统计量由各自的函数按「样本不足」返回 NaN。
    [
        sample_std(&sec),
        mean_of(&sec),
        sample_std(&pct),
        mean_of(&pct),
        lag1_corr(&sec),
        lag1_corr(&pct),
        pearson_pairwise(&pct, &sec),
    ]
}

/// 段 2：把间隔秒数除以 div 后四舍六入五取偶，乘上价格涨跌方向，做频次分布的熵（自然对数）。
fn buc_entropy(kept: &[(i64, f32)], div: f64) -> f64 {
    let mut vals: Vec<f64> = Vec::with_capacity(kept.len());
    for i in 1..kept.len() {
        let d_us = kept[i].0 - kept[i - 1].0;
        let s = (d_us / 1_000_000) as f64;
        let p = (kept[i].1 - kept[i - 1].1) / kept[i - 1].1;
        if !p.is_finite() {
            continue;
        }
        let sign = if p > 0.0 {
            1.0
        } else if p < 0.0 {
            -1.0
        } else {
            0.0
        };
        vals.push(round_half_even(s / div) * sign);
    }
    if vals.is_empty() {
        return f64::NAN;
    }
    // 按取值统计频次（p 非有限的那一行已经在上面跳过）
    // 注意：buc 可能同时出现 +0.0 和 -0.0（四舍五入到 0 再乘 -1），
    // polars 的 group_by 把两者当同一个值，所以这里也按数值归一化后再取键。
    let mut counts: HashMap<u64, usize> = HashMap::new();
    for &v in vals.iter() {
        let key = if v == 0.0 { 0.0f64.to_bits() } else { v.to_bits() };
        counts.entry(key).and_modify(|e| *e += 1).or_insert(1);
    }
    let n = vals.len() as f64;
    // 排序后遍历，保证多次运行结果一致
    let mut keys: Vec<u64> = counts.keys().copied().collect();
    keys.sort_unstable();
    // 旧代码的 buc_prob 列是「每个取值的频次 / 该股总行数」，它不是概率分布
    // （各值加起来等于 Σ c_j² / N），所以 pl.col("buc_prob").entropy() 里那次
    // 「按列和归一化」会真的改变结果，这里必须照做。
    let mut col_sum = 0.0f64;
    for &k in keys.iter() {
        let c = counts[&k] as f64;
        col_sum += c * (c / n);
    }
    if !(col_sum > 0.0) {
        return f64::NAN;
    }
    let mut h = 0.0;
    for &k in keys.iter() {
        let c = counts[&k] as f64;
        let p = (c / n) / col_sum;
        if p > 0.0 {
            h -= c * (p * p.ln());
        }
    }
    h
}

fn stage_two(kept: &[(i64, f32)]) -> [f64; 4] {
    [
        buc_entropy(kept, 1.0),
        buc_entropy(kept, 10.0),
        buc_entropy(kept, 30.0),
        buc_entropy(kept, 60.0),
    ]
}

// ============================================================================
// 段 3：分钟收益×量占比的同号连段
// ============================================================================

/// 旧代码在逐股这一列上做的事：rtn = close/close.shift(1)-1，vols 归一化后相乘，
/// dropna，然后按「同号」把连续段各自求和，记下段长。
/// 再找「上一段为负、这一段为正、且两段合计仍为负」的位置，统计其幅度。
fn stage_three(close: &[f64], vol: &[f64]) -> [f64; 4] {
    let mut vsum = 0.0f64;
    for &v in vol.iter() {
        if !v.is_nan() {
            vsum += v;
        }
    }
    let mut rv: Vec<f64> = Vec::with_capacity(MIN_PER_DAY);
    if vsum != 0.0 && vsum.is_finite() {
        for i in 0..MIN_PER_DAY {
            let c1 = close[i];
            let c0 = if i == 0 { f64::NAN } else { close[i - 1] };
            let rtn = if c1.is_nan() || c0.is_nan() {
                f64::NAN
            } else {
                c1 / c0 - 1.0
            };
            let x = rtn * (vol[i] / vsum);
            if !x.is_nan() {
                rv.push(x);
            }
        }
    }
    // 同号连段
    let mut res: Vec<f64> = Vec::new();
    let mut timev: Vec<i64> = Vec::new();
    let mut pos = 0usize;
    while pos < rv.len() {
        let n = rv.len() - pos;
        let mut fir = rv[pos];
        let mut t = 1usize;
        if n > 1 {
            let mut i = 1usize;
            loop {
                if i >= n {
                    break;
                }
                if fir * rv[pos + i] > 0.0 {
                    fir += rv[pos + i];
                    t += 1;
                    if t == n {
                        if fir != 0.0 {
                            res.push(fir);
                            timev.push(t as i64);
                        }
                        pos += t;
                        break;
                    }
                } else {
                    if fir != 0.0 {
                        res.push(fir);
                        timev.push(t as i64);
                    }
                    pos += t;
                    break;
                }
                i += 1;
            }
        } else {
            if fir != 0.0 {
                res.push(fir);
                timev.push(t as i64);
            }
            pos += t;
        }
    }
    // 负转正、合计为负
    let mut rvv: Vec<f64> = Vec::new();
    let mut wgt: Vec<f64> = Vec::new();
    for i in 1..res.len() {
        if res[i] > 0.0 && res[i - 1] < 0.0 && (res[i - 1] + res[i]) < 0.0 {
            let a = (res[i - 1] + res[i]).abs();
            rvv.push(a);
            wgt.push(a / ((timev[i - 1] + timev[i]) as f64));
        }
    }
    let negv_mean = mean_of(&rvv);
    let negv_max = rvv.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let negvwgt_mean = mean_of(&wgt);
    let negvwgt_max = wgt.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    [
        if rvv.is_empty() { f64::NAN } else { negv_mean },
        if rvv.is_empty() { f64::NAN } else { negv_max },
        if wgt.is_empty() { f64::NAN } else { negvwgt_mean },
        if wgt.is_empty() { f64::NAN } else { negvwgt_max },
    ]
}

// ============================================================================
// 段 4：振幅与收益的相关、量与正收益的相关
// ============================================================================

fn stage_four(high: &[f64], low: &[f64], close: &[f64], vol: &[f64]) -> [f64; 2] {
    let mut hl = vec![f64::NAN; MIN_PER_DAY];
    let mut ret = vec![f64::NAN; MIN_PER_DAY];
    let mut ret_pos = vec![0.0f64; MIN_PER_DAY];
    for i in 0..MIN_PER_DAY {
        hl[i] = high[i] / low[i] - 1.0;
        if i >= 1 && !close[i].is_nan() && !close[i - 1].is_nan() {
            ret[i] = close[i] / close[i - 1] - 1.0;
        }
        ret_pos[i] = if ret[i] > 0.0 { ret[i] } else { 0.0 };
    }
    [
        pearson_pairwise(&hl, &ret),
        pearson_pairwise(vol, &ret_pos),
    ]
}

// ============================================================================
// 段 5：分钟收益与分钟量的双向转移熵
// ============================================================================

fn stage_five(close: &[f64], vol: &[f64]) -> Vec<f64> {
    // pandas: closes = closes.pct_change()  → close[i]/close[i-1] - 1
    let mut cp = vec![f64::NAN; MIN_PER_DAY];
    for i in 1..MIN_PER_DAY {
        if !close[i].is_nan() && !close[i - 1].is_nan() {
            cp[i] = close[i] / close[i - 1] - 1.0;
        }
    }
    // merge_many(...).dropna()：只保留两列都非 NaN 的 (date, code)
    let mut xa: Vec<f64> = Vec::with_capacity(MIN_PER_DAY);
    let mut ya: Vec<f64> = Vec::with_capacity(MIN_PER_DAY);
    let mut xb: Vec<f64> = Vec::with_capacity(MIN_PER_DAY);
    let mut yb: Vec<f64> = Vec::with_capacity(MIN_PER_DAY);
    for i in 0..MIN_PER_DAY {
        if !cp[i].is_nan() && !vol[i].is_nan() {
            xa.push(cp[i]);
            ya.push(vol[i]);
            xb.push(vol[i]);
            yb.push(cp[i]);
        }
    }
    let mut out: Vec<f64> = Vec::with_capacity(80);
    for k in TE_KS.iter() {
        for c in TE_CS.iter() {
            out.push(crate::time_series::transfer_entropy(
                xa.clone(),
                ya.clone(),
                *k,
                *c,
            ));
        }
    }
    for k in TE_KS.iter() {
        for c in TE_CS.iter() {
            out.push(crate::time_series::transfer_entropy(
                xb.clone(),
                yb.clone(),
                *k,
                *c,
            ));
        }
    }
    out
}

// ============================================================================
// 入口
// ============================================================================

/// 逐股入口。算不出来（既没有成交也没有分钟数据）返回 None，上游会补 NaN。
pub fn compute_stock(code: &str, date: i64) -> Option<Vec<f32>> {
    let mut out = vec![f32::NAN; N_OUT];
    let mut any = false;

    // 段 1 / 段 2：来自逐笔成交
    if let Ok(trades) = read_trade_fast_inner(code, date, true, false, usize::MAX) {
        any = true;
        let rows: Vec<&TradeRecord> = trades.iter().filter(|r| r.flag != 32).collect();
        if !rows.is_empty() {
            let kept = select_percent_rows(&rows);
            for (i, v) in stage_one(&kept).iter().enumerate() {
                out[i] = *v as f32;
            }
            for (i, v) in stage_two(&kept).iter().enumerate() {
                out[7 + i] = *v as f32;
            }
        }
    }

    // 段 3 / 段 4 / 段 5：来自分钟数据
    if let Some(q) = minute_quads(code, date) {
        any = true;
        let (close, vol, high, low) = (&q[0], &q[1], &q[2], &q[3]);
        for (i, v) in stage_three(close, vol).iter().enumerate() {
            out[11 + i] = *v as f32;
        }
        for (i, v) in stage_four(high, low, close, vol).iter().enumerate() {
            out[15 + i] = *v as f32;
        }
        for (i, v) in stage_five(close, vol).iter().enumerate() {
            out[17 + i] = *v as f32;
        }
    }

    if any {
        Some(out)
    } else {
        None
    }
}

/// 横截面入口：只为新旧对照验证保留（旧代码的 get_factor 返回按 code 索引的 Series，
/// 对照脚本会走横截面分支）。生产流水线用的是 per_stock。
pub fn compute_cross(date: i64) -> Option<(Vec<String>, Vec<f32>)> {
    use rayon::prelude::*;
    let codes = crate::data_paths::list_stock_codes(date, "transaction");
    if codes.is_empty() {
        return None;
    }
    let per: Vec<Vec<f32>> = codes
        .par_iter()
        .map(|c| compute_stock(c, date).unwrap_or_else(|| vec![f32::NAN; N_OUT]))
        .collect();
    let mut vals: Vec<f32> = Vec::with_capacity(codes.len() * N_OUT);
    for v in per.iter() {
        vals.extend_from_slice(v);
    }
    Some((codes, vals))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_layout() {
        let n = names();
        assert_eq!(n.len(), N_OUT);
        assert_eq!(n[0], "time_std");
        assert_eq!(n[6], "corr_pe");
        assert_eq!(n[7], "buc_entropy1");
        assert_eq!(n[11], "negv_mean");
        assert_eq!(n[16], "voh_cor");
        assert_eq!(n[17], "ret_to_vol_k1_c2");
        assert_eq!(n[56], "ret_to_vol_k10_c20");
        assert_eq!(n[57], "vol_to_ret_k1_c2");
        assert_eq!(n[96], "vol_to_ret_k10_c20");
    }

    #[test]
    fn rounding_is_half_even() {
        assert_eq!(round_half_even(0.5), 0.0);
        assert_eq!(round_half_even(1.5), 2.0);
        assert_eq!(round_half_even(2.5), 2.0);
        assert_eq!(round_half_even(-1.5), -2.0);
        assert_eq!(round_half_even(-2.5), -2.0);
        assert_eq!(round_half_even(3.4), 3.0);
    }
}
