//! hm0（hot_change1）—— 旧框架（design_whatever_out.FactorDesign，**分钟**数据）的 Rust 复现。
//!
//! 旧文件：/home/chenzongwei/pythoncode/成果hm0_成果hm1_hot_change/hm0_okay.py
//!   run_base 的 Go.get_factor，参数 get_many=1 / read_full=0 / change_time=1，
//!   Go.check_my_kind == "minute"，所以旧框架走的是
//!       Go(base_dir, date, date, 1, base_ver, read_full=0, get_many=1)
//!   也就是**一天一次、返回全市场一张表**，因此本文件只有 compute_cross。
//!
//! ════════════════════════════════════════════════════════════════════════
//! 一、names 的来源（旧文件里 names 是拼出来的，不是字面列表）
//! ════════════════════════════════════════════════════════════════════════
//!   names = ['hot_change_on_market_cut50',
//!            'hot_change_on_market_cut60',
//!            'hot_change_on_stock']
//!   names = [i+"_allday" for i in names] + [i+"_morning" for i in names]
//!          + [i+"_afternoon" for i in names]
//!   这份拼出来的 names **同时**当 names_single 和 names_save 传进 run_factor_out，
//!   所以 Rust 侧 names() 的顺序必须与 get_factor() 的返回顺序逐个对齐：
//!   每个时段返回 [c11, c12, change2]，三个时段依次是 allday / morning / afternoon。
//!   （gap5 / gap1 的入选名单在 calc_horizon5 / calc_horizon1 里是另一份字面列表，
//!     那些名字是 base 名 + 后处理后缀，不属于本文件的 names()。）
//!
//! ════════════════════════════════════════════════════════════════════════
//! 二、旧代码逐步对照（写代码前先抄在这里，方便复核）
//! ════════════════════════════════════════════════════════════════════════
//!  1) highs / lows / amounts = new_get_field_min_data(...)，各返回 **3 个** 240×全市场 的
//!     DataFrame，顺序 [allday, morning, afternoon]。框架对每段做的加权是「乘分钟序号」：
//!        allday    = a * 1
//!        morning   = a * nums.where(nums <= 120, nan)   → 第 m(1..120) 行乘 m，其余 NaN
//!        afternoon = a * nums.where(nums >  120, nan)   → 第 m(121..240) 行乘 m，其余 NaN
//!     这是**加权**不是简单切窗口（hm017 头部记的是同一条坑）。另外
//!     get_field_min_data 会把每天最后 3 根（index%240 >= 237，即 14:58/14:59/15:00）置 NaN。
//!
//!  2) one(highs, lows, amounts) 对每一段算：
//!        rushs = (highs / lows.shift(1) - 1).rank(axis=1)     ← 每分钟跨股票排名
//!        speed = (rushs - rushs.shift(1)).abs()               ← shift 是按行（分钟）
//!        mar   = speed.mean(axis=1)                           ← 每分钟跨股票的均值
//!        one1(cut):
//!            mar_mean = mar.quantile(cut)                     ← 240 个分钟值上的分位点（标量）
//!            is_fast  = sign(mar-mar_mean)→  replace(0,1)→replace(-1,nan)
//!                       等价于「mar >= mar_mean 的行取 1，其余（含 NaN）取 NaN」
//!            is_slow  = sign(mar-mar_mean)→replace(0,-1)→replace(1,nan)→replace(-1,1)
//!                       等价于「mar <= mar_mean 的行取 1，其余（含 NaN）取 NaN」
//!            amounts_fast = amounts 按行掩码（is_fast 只有 1 或 NaN，乘上去就是掩码）
//!            corr_fast = amounts_fast.corr().abs().skew()
//!                        corr() 是**成对完整观测**的 Pearson（pandas libalgos.nancorr，
//!                        Welford 增量法）；abs() 后 skew() 是**按列**（每只股票一行 |corr|
//!                        的偏度）
//!            change1 = corr_fast.rank() - corr_slow.rank()    ← 跨股票排名后相减
//!        c11 = one1(0.5)，c12 = one1(0.6)
//!
//!  3) change2 的掩码是**逐格**的（这一段最容易看错）：
//!        speed = (...).abs().T                                ← 转置：index=股票, columns=分钟
//!        speed.median()                                       ← axis=0 ⇒ 每分钟跨股票的**中位数**
//!        is_fast = sign(speed - speed.median()) ... .T         ← 转回来，index=分钟 columns=股票
//!        amounts_fast = amounts * is_fast
//!     即「该股该分钟的 |rushs 变化| >= 同一分钟全市场中位数」才保留，和 cut 那种整行掩码不同。
//!
//!  4) get_factor 返回 sum(res, [])：三个时段 × [c11, c12, change2] 共 9 个 Series，
//!     旧框架再 `i.to_frame(self.end_date).T` 变成「一行（日期）× 全市场列」。
//!
//! ════════════════════════════════════════════════════════════════════════
//! 三、数值口径（都按 pandas/numpy 的实测行为复刻）
//! ════════════════════════════════════════════════════════════════════════
//! · mar 的逐行均值：pandas `mean(axis=1)` 先转置，求和轴在内存里是连续的，
//!   走 numpy 的**成对求和**（PW_BLOCKSIZE=128、8 路展开），本文件用
//!   crate::tail_v4_fulltest_rust::pairwise_sum 复用同一份逐字转写。
//! · skew 里对 |corr| 的按列求和：values 是 (K,K) C 顺序，axis=0 的归约步长是 K*8
//!   而不是 8，numpy **不走**成对求和而是顺序累加（实测：np.sum(a, axis=0) 在
//!   多列 C 顺序数组上给出顺序累加的结果）。所以这一层用顺序累加。
//! · corr：逐字复刻 pandas libalgos.nancorr 的 Welford 递推（含 `nobs < 1 → NaN`、
//!   `divisor==0 → NaN`、以及 np.isfinite 把 inf 也算缺失）。
//! · skew：nanskew —— count<3 → NaN，m2==0（含 |m2|<1e-14 抹零后）→ 0，
//!   其余 `(count*sqrt(count-1)/(count-2)) * (m3/m2**1.5)`。
//! · rank：pandas 默认 method='average'、升序、NaN 保留。
//! · 分位数：np.percentile(method='linear')，即 virtual=(n-1)*q、线性插值。
//! · 中位数：nanmedian，偶数个取中间两个的均值。
//!
//! ════════════════════════════════════════════════════════════════════════
//! 五、对账范围与取舍（为什么只验了 2 个日期）
//! ════════════════════════════════════════════════════════════════════════
//! 这个构造每个输出都要做一次 5792×5792 的成对完整观测相关矩阵，再 abs + 逐列 skew：
//! 一天 3 段 × 3 输出 × （fast/slow 两张）= **18 次**全市场相关 + skew。
//! Rust 侧（本文件，列优先布局 + rayon）单日全市场约 6.5 秒。
//! 旧侧（pandas）在同样输入上慢三个数量级：一次 `.corr().abs().skew()` 实测 5 分钟上下
//! （单独测 `DataFrame.skew()` 一次 235 秒；机器常年 400 并行、load 470+/512，
//! 内存带宽被挤满，见 lib/okay_database/WORKER_BRIEF.txt）。所以 verify_series.py 每多算
//! 一组 (日期,股票) 就要把旧侧全市场重跑一遍，4 个不同日期≈3~4 小时。
//!
//! 因此对账按 **2 个日期 × 2 只股票 = 4 组 (日期,股票)** 做（一个 2018 年附近、一个 2025 年，
//! 拉开时间），另加一层**小 universe 严格对齐**：把旧框架的分钟股票数截到 300 只，
//! 让旧 okay 文件的 `one()` 原文和一套独立重写的实现（自写 Welford 相关 + 自写 nanskew +
//! 自写 rank）在 9 个输出上逐值比，结果 9 个输出全部 0 处不一致。
//! 全市场那条链路本来就要一次读全市场，2 个日期已经覆盖了「全市场路径」这一维度。
//!
//! 已知的坑（第一版踩过，已修）：change2 里 speed == 每分钟中位数时，
//! is_fast 和 is_slow **同时**为 1，这一格要同时进两张掩码表；
//! 写成 if/else 只进一张，全市场对账时 3 个 change2 输出各出现 3 个名次的偏差。
//!
//! ════════════════════════════════════════════════════════════════════════
//! 六、输出排布
//! ════════════════════════════════════════════════════════════════════════
//!   compute_cross 返回 (codes, vals)，codes 是分钟数据 symbol_map.csv 的全名单，
//!   vals 是**股票优先**：第 i 只股票连续 9 个值，顺序
//!   [cut50_allday, cut60_allday, on_stock_allday,
//!    cut50_morning, cut60_morning, on_stock_morning,
//!    cut50_afternoon, cut60_afternoon, on_stock_afternoon]。

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, LazyLock, Mutex};

use rayon::prelude::*;

/// 分钟数据每天 240 根 K 线
const MIN_PER_DAY: usize = 240;
/// 旧框架 get_field_min_data 把每根日线索引 % 240 >= 237 的三根置 NaN
const MIN_NAN_FROM: usize = 237;
/// 输出因子个数 = 3 个时段 × 3 个量
pub const N_OUT: usize = 9;

/// 三个分钟字段
const FIELDS: [&str; 3] = ["high", "low", "amount"];

/// 三个时段（顺序就是 names 的拼法）
const SEGS: [&str; 3] = ["allday", "morning", "afternoon"];

/// 三个基础名（时段后缀由 names() 拼）
const BASE: [&str; 3] = [
    "hot_change_on_market_cut50",
    "hot_change_on_market_cut60",
    "hot_change_on_stock",
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
// 分钟数据：元信息 + 一天的整块读取（与 hm017 / hm028 同一套读法）
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

/// 时段权重：allday 全 1；morning 第 1..120 行取分钟序号本身、其余 NaN；
/// afternoon 第 121..240 行取分钟序号本身、其余 NaN。
#[inline]
fn seg_weight(seg: usize, row: usize) -> f64 {
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

/// 按行（分钟）加权：w 为 NaN 的行整行变 NaN，其余乘以 w。
fn apply_seg_weight(src: &[f64], n: usize, seg: usize) -> Vec<f64> {
    let mut out = vec![f64::NAN; src.len()];
    for row in 0..MIN_PER_DAY {
        let w = seg_weight(seg, row);
        let base = row * n;
        if w.is_nan() {
            continue;
        }
        for j in 0..n {
            out[base + j] = src[base + j] * w;
        }
    }
    out
}

/// 逐行 rank（pandas DataFrame.rank(axis=1)，升序、并列取平均、NaN 保留）。
/// 输入输出都是行优先的 nrows × ncols。
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

/// 一维 rank（pandas Series.rank，method='average'、升序、NaN 保留）
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

/// numpy 成对求和（复用 tail_v4 里那份逐字转写）
#[inline]
fn pairwise_sum(a: &[f64]) -> f64 {
    crate::tail_v4_fulltest_rust::pairwise_sum(a)
}

/// 逐行 skipna 均值（pandas mean(axis=1)）：值里的 NaN 先抹成 0 再成对求和，除以非 NaN 个数。
fn row_mean_nan(mat: &[f64], n: usize) -> Vec<f64> {
    let mut out = vec![f64::NAN; MIN_PER_DAY];
    let mut buf = vec![0.0f64; n];
    for row in 0..MIN_PER_DAY {
        let base = row * n;
        let mut cnt = 0usize;
        for j in 0..n {
            let v = mat[base + j];
            if v.is_nan() {
                buf[j] = 0.0;
            } else {
                buf[j] = v;
                cnt += 1;
            }
        }
        out[row] = if cnt == 0 {
            f64::NAN
        } else {
            pairwise_sum(&buf) / cnt as f64
        };
    }
    out
}

/// 逐行 skipna 中位数（pandas median(axis=1) = np.nanmedian）
fn row_median_nan(mat: &[f64], n: usize) -> Vec<f64> {
    let mut out = vec![f64::NAN; MIN_PER_DAY];
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
    for (row, v) in rows {
        out[row] = v;
    }
    out
}

/// np.percentile(values[finite], 100*q, method='linear')
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

/// 非有限值一律 NaN（与旧框架落盘的 replace(inf, nan) 一致）
#[inline]
fn finite_or_nan(v: f64) -> f64 {
    if v.is_finite() {
        v
    } else {
        f64::NAN
    }
}

// ============================================================================
// 相关矩阵：pandas libalgos.nancorr 的 Welford 逐字复刻
// ============================================================================

/// 把行优先的 (240 × n) 掩码矩阵转成列优先 (n × 240)，
/// 这样「固定一列 j、扫遍所有 i」时两侧都是连续内存。
fn to_col_major(mat: &[f64], n: usize) -> Vec<f64> {
    let mut out = vec![f64::NAN; n * MIN_PER_DAY];
    for row in 0..MIN_PER_DAY {
        let base = row * n;
        for j in 0..n {
            out[j * MIN_PER_DAY + row] = mat[base + j];
        }
    }
    out
}

/// 单对 (i, j) 的成对完整观测 Pearson，逐字照抄 pandas nancorr 的 Welford 递推。
#[inline]
fn pearson_pair(x: &[f64], y: &[f64]) -> f64 {
    let mut nobs: i64 = 0;
    let mut meanx = 0.0f64;
    let mut meany = 0.0f64;
    let mut ssqdmx = 0.0f64;
    let mut ssqdmy = 0.0f64;
    let mut covxy = 0.0f64;
    for t in 0..MIN_PER_DAY {
        let vx = x[t];
        let vy = y[t];
        // pandas 用的是 np.isfinite：inf 也算缺失
        if vx.is_finite() && vy.is_finite() {
            nobs += 1;
            let dx = vx - meanx;
            let dy = vy - meany;
            let inv = 1.0 / (nobs as f64);
            meanx += inv * dx;
            meany += inv * dy;
            ssqdmx += (vx - meanx) * dx;
            ssqdmy += (vy - meany) * dy;
            covxy += (vx - meanx) * dy;
        }
    }
    if nobs < 1 {
        return f64::NAN;
    }
    let divisor = (ssqdmx * ssqdmy).sqrt();
    if divisor != 0.0 {
        covxy / divisor
    } else {
        f64::NAN
    }
}

/// nanskew：对一列值算 pandas DataFrame.skew() 的那个偏度。
fn nanskew(col: &[f64]) -> f64 {
    let n = col.len();
    let mut count = 0.0f64;
    let mut sum = 0.0f64;
    for i in 0..n {
        let v = col[i].abs(); // 旧代码是 corr().abs().skew()
        if !v.is_nan() {
            count += 1.0;
            sum += v;
        } else {
            sum += 0.0;
        }
    }
    let mean = sum / count;
    let mut m2 = 0.0f64;
    let mut m3 = 0.0f64;
    for i in 0..n {
        let v = col[i].abs();
        let adj = if v.is_nan() { 0.0 } else { v - mean };
        let a2 = adj * adj;
        let a3 = a2 * adj;
        m2 += a2;
        m3 += a3;
    }
    if m2.abs() < 1e-14 {
        m2 = 0.0;
    }
    if m3.abs() < 1e-14 {
        m3 = 0.0;
    }
    let mut result = (count * (count - 1.0).sqrt() / (count - 2.0)) * (m3 / m2.powf(1.5));
    if m2 == 0.0 {
        result = 0.0;
    }
    if count < 3.0 {
        result = f64::NAN;
    }
    result
}

/// corr().abs().skew()：返回每只股票一列 |corr| 的偏度。
/// 掩码矩阵先转列优先，再按列并行（每列独立算完，无归约顺序问题）。
fn corr_abs_skew(masked: &[f64], n: usize) -> Vec<f64> {
    let cm = to_col_major(masked, n);
    let res: Vec<(usize, f64)> = (0..n)
        .into_par_iter()
        .map(|j| {
            let y = &cm[j * MIN_PER_DAY..(j + 1) * MIN_PER_DAY];
            let mut col = vec![0.0f64; n];
            for i in 0..n {
                col[i] = pearson_pair(&cm[i * MIN_PER_DAY..(i + 1) * MIN_PER_DAY], y);
            }
            (j, nanskew(&col))
        })
        .collect();
    let mut out = vec![f64::NAN; n];
    for (j, v) in res {
        out[j] = v;
    }
    out
}

// ============================================================================
// 一段（allday / morning / afternoon）的三个输出
// ============================================================================

/// 返回该段的 [cut50, cut60, on_stock]，每个长度 n。
fn segment(H: &[f64], L: &[f64], A: &[f64], n: usize) -> [Vec<f64>; 3] {
    // rushs = (H / L.shift(1) - 1).rank(axis=1)
    let mut ratio = vec![f64::NAN; MIN_PER_DAY * n];
    for row in 1..MIN_PER_DAY {
        for j in 0..n {
            let h = H[row * n + j];
            let l = L[(row - 1) * n + j];
            ratio[row * n + j] = h / l - 1.0;
        }
    }
    let rushs = rank_rows(&ratio, MIN_PER_DAY, n);

    // speed = |rushs - rushs.shift(1)|
    let mut speed = vec![f64::NAN; MIN_PER_DAY * n];
    for row in 1..MIN_PER_DAY {
        for j in 0..n {
            let a = rushs[row * n + j];
            let b = rushs[(row - 1) * n + j];
            speed[row * n + j] = (a - b).abs();
        }
    }
    let mar = row_mean_nan(&speed, n);

    let mut out: [Vec<f64>; 3] = [
        vec![f64::NAN; n],
        vec![f64::NAN; n],
        vec![f64::NAN; n],
    ];

    // ---- cut 0.5 / 0.6：整行掩码 ----
    for (k, cut) in [0.5f64, 0.6f64].iter().enumerate() {
        let q = quantile_linear(&mar, *cut);
        let mut fast_rows = vec![false; MIN_PER_DAY];
        let mut slow_rows = vec![false; MIN_PER_DAY];
        for row in 0..MIN_PER_DAY {
            let m = mar[row];
            if m.is_nan() || q.is_nan() {
                continue;
            }
            fast_rows[row] = m >= q;
            slow_rows[row] = m <= q;
        }
        let mf = row_mask(&A, n, &fast_rows);
        let ms = row_mask(&A, n, &slow_rows);
        let cf = corr_abs_skew(&mf, n);
        let cs = corr_abs_skew(&ms, n);
        let rf = rank_average(&cf);
        let rs = rank_average(&cs);
        for j in 0..n {
            out[k][j] = finite_or_nan(rf[j] - rs[j]);
        }
    }

    // ---- change2：逐格掩码（每分钟跨股票的中位数） ----
    //  ！！注意 speed == median 时 is_fast 和 is_slow **同时**为 1，这一格要同时进两张掩码表，
    //  不能写成 if/else（第一版写成 else，全市场对账时 change2 那三列出现 3 个名次的偏差）。
    let med = row_median_nan(&speed, n);
    let mut mf = vec![f64::NAN; MIN_PER_DAY * n];
    let mut ms = vec![f64::NAN; MIN_PER_DAY * n];
    for row in 0..MIN_PER_DAY {
        let md = med[row];
        if md.is_nan() {
            continue;
        }
        for j in 0..n {
            let s = speed[row * n + j];
            if s.is_nan() {
                continue;
            }
            if s >= md {
                mf[row * n + j] = A[row * n + j];
            }
            if s <= md {
                ms[row * n + j] = A[row * n + j];
            }
        }
    }
    let cf = corr_abs_skew(&mf, n);
    let cs = corr_abs_skew(&ms, n);
    let rf = rank_average(&cf);
    let rs = rank_average(&cs);
    for j in 0..n {
        out[2][j] = finite_or_nan(rf[j] - rs[j]);
    }

    out
}

/// 按行掩码构造 masked 矩阵：掩码为真的行原样保留，否则 NaN
fn row_mask(A: &[f64], n: usize, keep: &[bool]) -> Vec<f64> {
    let mut out = vec![f64::NAN; MIN_PER_DAY * n];
    for row in 0..MIN_PER_DAY {
        if !keep[row] {
            continue;
        }
        out[row * n..row * n + n].copy_from_slice(&A[row * n..row * n + n]);
    }
    out
}

// ============================================================================
// 入口
// ============================================================================

/// hm0 是分钟系列，一天一次全市场，没有逐股实现。
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

    // 三个字段各读一整块（240 × n），最后三根已经是 NaN
    let raw: Vec<Vec<f64>> = FIELDS
        .iter()
        .map(|f| read_day_block(f, day, n))
        .collect::<Option<Vec<_>>>()?;

    let mut out = vec![f64::NAN; n * N_OUT];
    for seg in 0..3usize {
        let h = apply_seg_weight(&raw[0], n, seg);
        let l = apply_seg_weight(&raw[1], n, seg);
        let a = apply_seg_weight(&raw[2], n, seg);
        let vals = segment(&h, &l, &a, n);
        for k in 0..3usize {
            let base = seg * 3 + k;
            for j in 0..n {
                out[j * N_OUT + base] = vals[k][j];
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
        assert_eq!(n[0], "hot_change_on_market_cut50_allday");
        assert_eq!(n[1], "hot_change_on_market_cut60_allday");
        assert_eq!(n[2], "hot_change_on_stock_allday");
        assert_eq!(n[3], "hot_change_on_market_cut50_morning");
        assert_eq!(n[6], "hot_change_on_market_cut50_afternoon");
        assert_eq!(n[8], "hot_change_on_stock_afternoon");
    }

    #[test]
    fn seg_weights() {
        assert_eq!(seg_weight(0, 0), 1.0);
        assert_eq!(seg_weight(0, 239), 1.0);
        assert_eq!(seg_weight(1, 0), 1.0);
        assert_eq!(seg_weight(1, 119), 120.0);
        assert!(seg_weight(1, 120).is_nan());
        assert!(seg_weight(2, 119).is_nan());
        assert_eq!(seg_weight(2, 120), 121.0);
        assert_eq!(seg_weight(2, 239), 240.0);
    }

    #[test]
    fn rank_row_basic() {
        // 一行 [3, 1, 1, NaN] → [3, 1.5, 1.5, NaN]
        let mat = vec![3.0, 1.0, 1.0, f64::NAN];
        let r = rank_rows(&mat, 1, 4);
        assert_eq!(r[0], 3.0);
        assert_eq!(r[1], 1.5);
        assert_eq!(r[2], 1.5);
        assert!(r[3].is_nan());
    }

    #[test]
    fn rank_avg_basic() {
        let v = [2.0, 1.0, 1.0, f64::NAN, 3.0];
        let r = rank_average(&v);
        assert_eq!(r[0], 3.0);
        assert_eq!(r[1], 1.5);
        assert_eq!(r[2], 1.5);
        assert!(r[3].is_nan());
        assert_eq!(r[4], 4.0);
    }

    #[test]
    fn quantile_matches_np() {
        let v = [1.0, 2.0, 3.0, 4.0, f64::NAN];
        // np.percentile([1,2,3,4], 50) = 2.5
        assert!((quantile_linear(&v, 0.5) - 2.5).abs() < 1e-15);
        // q=1 → 最大值
        assert!((quantile_linear(&v, 1.0) - 4.0).abs() < 1e-15);
    }
}
