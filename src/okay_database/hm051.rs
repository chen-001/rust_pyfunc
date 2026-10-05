//! hm51「同频伙伴」——旧框架（design_whatever_out.FactorDesign，**分钟**数据）的 Rust 复现。
//!
//! 旧文件：/home/chenzongwei/pythoncode/成果hm51_同频伙伴/hm51_okay.py
//!   run_base 的 Go.get_factor，get_many=0 / read_full=0 / change_time=1，
//!   Go.check_my_kind == "minute"（继承 dwo.FactorDesign），所以旧框架走的是
//!       Go(base_dir, date, date, look_back_window, base_ver, read_full=0, get_many=0)
//!   一天一次、返回全市场一张表，因此本文件只有 compute_cross，没有 compute_stock。
//!
//! ════════════════════════════════════════════════════════════════════
//! 一、旧代码逐步对照（写代码前抄在这里，方便复核）
//! ════════════════════════════════════════════════════════════════════
//! 5 个原始字段：close、act_buy_vol_sum、act_sell_vol_sum、ask_size_1_mean、bid_size_1_mean
//! （注意旧代码第 44 行 volumes 也是读 "close"，不是成交量）
//!
//! new_get_field_min_data(name) = add_day_date_from_back(get_field_min_data(name))
//!   · get_field_min_data 里 `field_data[field_data.index % 240 >= 237] = np.nan`，
//!     把每天**最后 3 根** K 线（14:58/14:59/15:00）置 NaN。是 3 根不是 4 根，
//!     库里 rp.read_minute_field 置的是 4 根（236..239），所以本文件自己读 h5。
//!   · add_day_date_from_back 只是把行号换成交易日，一天一次调用时行序不变。
//!   · 行 0..119 是上午（09:31~11:30），行 120..239 是下午（13:01~15:00）。
//!
//! 派生量（顺序与旧代码一致）：
//!   rets       = close / close.shift(1) - 1            （行 0 是 NaN；不做 ffill）
//!   volumes    = close / close.sum()                   （列内 skipna 求和后归一）
//!   act_buy    = act_buy_vol_sum / act_buy_vol_sum.sum()
//!   act_sell   = act_sell_vol_sum / act_sell_vol_sum.sum()
//!   ab_ratio   = act_buy / volumes
//!   imbalance1 = (ask_size_1_mean - bid_size_1_mean) / (ask_size_1_mean + bid_size_1_mean)
//!
//! get_five(dc, drs)（drs = [rets, volumes, act_buy, act_sell, ab_ratio]）：
//!   corr = np.log(dc.iloc[:120, :]).replace([inf, -inf], nan).corr()
//!   corr_large = (corr >= corr.quantile(0.99)).astype(int).replace(0, nan)   # 1.0 / NaN
//!   corr_small = (corr <= corr.quantile(0.01)).astype(int).replace(0, nan)
//!   对每个 dr：
//!       dr_morning   = dr.iloc[:120, :].mean()      # 每列一个数（Series，index=股票）
//!       dr_afternoon = dr.iloc[120:, :].mean()
//!       rl_pri_morning   = (corr_large * pd.DataFrame({k: dr_morning for k in corr_large.columns})).mean()
//!       rl_pri_afternoon = (corr_large * pd.DataFrame({k: dr_afternoon for k in ...})).mean()
//!       rl_abn_morning   = rl_pri_morning   - dr_afternoon
//!       rl_abn_afternoon = rl_pri_afternoon - dr_afternoon
//!       rs_* 同理，用 corr_small
//!       # 注意：`pd.DataFrame({k: dr_morning for k in corr_large.columns})` 里每个列
//!       # 放的都是**同一个 Series**，所以它的行索引 = 股票、第 (i, j) 格的值是
//!       # dr_morning[第 i 行的股票]，跟列号 j 无关。于是
//!       #   rl_pri_morning[j] = Σ_{i: corr_large[i,j]==1 且 dr_morning[i] 非 NaN} dr_morning[i] / 个数
//!       #（.mean() 沿 axis=0、skipna，只跳过 NaN，inf 照算）
//!   get_five 返回 9 组 × 5 个 Series，组顺序：
//!       large_pri_morning, small_pri_morning, large_abn_morning, small_abn_morning,
//!       large_pri_afternoon, small_pri_afternoon, large_abn_afternoon, small_abn_afternoon,
//!       dr_afternoon
//!   get_factor 对 dc 依次取 [rets, volumes, act_buy, act_sell, ab_ratio, imbalance1]，
//!   把 6 × 45 = 270 个 Series 拼起来返回。
//!
//! names_single = 45 个基础名（组为外层、dr 为内层）× 6 个后缀块，后缀块的顺序是
//!   corr_on_ret, corr_on_volume, corr_on_act_buy, corr_on_act_sell, corr_on_ab_ratio,
//!   corr_on_imbalance1，与 dc 的顺序一一对应。names() 必须与之一致。
//! 实测 names 里没有重名（45 个基础名互不相同，6 个后缀互不相同），所以不需要 _dup2 后缀。
//!
//! ════════════════════════════════════════════════════════════════════
//! 二、相关系数必须逐位复刻 pandas（这是本系列最难的地方）
//! ════════════════════════════════════════════════════════════════════
//! pandas 2.3.3 的 DataFrame.corr()（method="pearson"）走 pandas/_libs/algos.pyx 的
//! `nancorr`，用的是 **Welford 在线算法**，不是两遍法：
//!     for i in range(N):
//!         if mask[i,xi] and mask[i,yi]:
//!             vx, vy = mat[i,xi], mat[i,yi]
//!             nobs += 1
//!             dx = vx - meanx; dy = vy - meany
//!             meanx += 1./nobs * dx
//!             meany += 1./nobs * dy
//!             ssqdmx += (vx - meanx) * dx
//!             ssqdmy += (vy - meany) * dy
//!             covxy  += (vx - meanx) * dy
//!     divisor = sqrt(ssqdmx * ssqdmy)
//!     r = covxy / divisor   （divisor == 0 时 NaN）
//! 外层是 `for xi in range(K): for yi in range(xi+1)`，每对只算一次、
//! 同时写 result[xi,yi] 和 result[yi,xi]，所以**矩阵是对称的**（本文件按 `low[a][b]`
//! 存下三角，a >= b）。
//! 两遍法和 Welford 在近常量列上差别很大（实测同一对股票：两遍法 -1.22e-15、
//! pandas 返回 NaN），而掩码只取每列前 1% 的 pair，差一点就会改掉因子值，
//! 所以这里逐行照抄 Welford（实测与 pandas 逐位一致，见第三节）。
//! mask 用 np.isfinite，所以是「非 NaN 且非 ±inf」。
//!
//! 分位数：pandas DataFrame.quantile(0.99) 对每列走
//!   _nanpercentile_1d → np.percentile(非 NaN 值, 99.0, method="linear")
//! numpy 的 linear 是 Hyndman&Fan 第 7 种：
//!   virtual = n*q + (alpha + q*(1-alpha-beta)) - 1   （alpha=beta=1）
//!   prev = floor(virtual)，next = prev+1，gamma = virtual - prev
//!   r = a + (b-a)*gamma ；当 gamma >= 0.5 时改用 r = b - (b-a)*(1-gamma)
//!   （numpy 的 `_lerp` 先算前式再按 where=t>=0.5 覆盖，两个式子分开算）
//!   virtual >= n-1 取最大值，virtual < 0 取最小值，n=0 给 NaN
//! 本文件把这一整套照抄，包括那个 0.5 分支。
//!
//! ════════════════════════════════════════════════════════════════════
//! 三、实测与已知偏差
//! ════════════════════════════════════════════════════════════════════
//! · Welford 相关系数：与 pandas 逐位一致（20250303 的 500 只股票 × 6 个 dc，
//!   非 NaN 位置 bit 相同、NaN 位置也相同）。
//! · np.log 与 glibc log（Rust f64::ln 走的就是 glibc log）：实测约 0.01%~0.2% 的
//!   输入差 1 ulp。用真实数据比过：6 个 dc 的 log 分别用 numpy 和 libm 算，
//!   800 只股票上相关系数掩码（0.99/0.01 两个分位）差异为 0。留在文件里备案。
//! · 列内求和（volumes.sum() / 列均值）用简单顺序累加，numpy 用成对求和，
//!   相对差 ~1e-16，只影响 log 输入的末位，量级与上一条相同。
//! · 旧代码把 inf 直接写进因子（分钟分支没有 replace(inf, nan)）。本文件同行为，
//!   不做额外清洗。
//!
//! ════════════════════════════════════════════════════════════════════
//! 四、输出排布（横截面系列的约定）
//! ════════════════════════════════════════════════════════════════════
//! compute_cross 返回 (codes, vals)，vals **股票优先**：第 i 只股票的 270 个值在
//! vals[i*270 .. i*270+270]，与 worker 侧 `vals.chunks(expected_len).zip(codes)` 一致。
//! codes 取分钟数据 symbol_map.csv 的全量列（与旧框架 info_dict["symbol"] 同序）。

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, LazyLock, Mutex};

use rayon::prelude::*;

/// 每天 240 根 1min K 线
const N_MIN: usize = 240;
/// 旧框架 `field_data.index % 240 >= 237` 置 NaN 的起始位置（14:58/14:59/15:00）
const NAN_FROM: usize = 237;
/// 上午根数（corr 与 dr_morning 用前 120 根）
const N_MORNING: usize = 120;
/// 每个 dc 的 9 组
const N_GROUP: usize = 9;
/// 每组 5 个 dr
const N_DR: usize = 5;
/// dc 个数
const N_DC: usize = 6;
/// 每个 dc 内部的因子数 = 9 × 5
const N_DC_OUT: usize = N_GROUP * N_DR;
/// 每只股票输出的因子数 = 6 × 45
const N_OUT: usize = N_DC * N_DC_OUT;

/// 组名（get_five 的返回顺序）
const GROUPS: [&str; N_GROUP] = [
    "large_pri_morning",
    "small_pri_morning",
    "large_abn_morning",
    "small_abn_morning",
    "large_pri_afternoon",
    "small_pri_afternoon",
    "large_abn_afternoon",
    "small_abn_afternoon",
    "dr_afternoon",
];
/// drs = [rets, volumes, act_buy, act_sell, ab_ratio]
const DR_NAMES: [&str; N_DR] = ["rets", "volumes", "act_buy", "act_sell", "ab_ratio"];
/// 后缀块顺序 = dc 顺序 [rets, volumes, act_buy, act_sell, ab_ratio, imbalance1]
const DC_SUFFIX: [&str; N_DC] = [
    "ret",
    "volume",
    "act_buy",
    "act_sell",
    "ab_ratio",
    "imbalance1",
];

/// 因子名顺序必须与 compute_cross 的返回值顺序一致。
/// 组为外层、dr 为内层，再套 6 个后缀块。
pub fn names() -> Vec<String> {
    let mut v = Vec::with_capacity(N_OUT);
    for s in DC_SUFFIX {
        for g in GROUPS {
            for d in DR_NAMES {
                v.push(format!("{d}_{g}corr_on_{s}"));
            }
        }
    }
    v
}

// ============================================================================
// 分钟数据读取（自己读 h5，因为库里的 read_minute_field 会把最后 4 根置 NaN）
// ============================================================================

struct MinuteMeta {
    date_to_idx: HashMap<i64, usize>,
    codes: Vec<String>,
    total_rows: usize,
}

static META_CACHE: LazyLock<Mutex<HashMap<PathBuf, Arc<MinuteMeta>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

#[cfg(feature = "hdf5")]
fn get_meta() -> Option<Arc<MinuteMeta>> {
    let dir = crate::data_paths::minute_dir();
    let mut cache = META_CACHE.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(m) = cache.get(&dir) {
        return Some(m.clone());
    }
    let meta = Arc::new(load_meta(&dir)?);
    cache.insert(dir, meta.clone());
    Some(meta)
}

#[cfg(feature = "hdf5")]
fn load_meta(dir: &std::path::Path) -> Option<MinuteMeta> {
    // calendar_map.csv：表头 date_min，之后每行一个交易日，行号 = day_idx
    let cal = std::fs::read_to_string(dir.join("calendar_map.csv")).ok()?;
    let mut date_to_idx = HashMap::new();
    let mut day_idx = 0usize;
    for line in cal.lines() {
        let s = line.trim();
        if s.is_empty() {
            continue;
        }
        if let Ok(d) = s.parse::<i64>() {
            date_to_idx.insert(d, day_idx);
            day_idx += 1;
        }
    }
    // symbol_map.csv：symbol,pos；pos 就是列号
    let sym = std::fs::read_to_string(dir.join("symbol_map.csv")).ok()?;
    let mut pairs: Vec<(usize, String)> = Vec::new();
    let mut max_pos = 0usize;
    for line in sym.lines() {
        let s = line.trim();
        if s.is_empty() {
            continue;
        }
        let mut it = s.split(',');
        let (Some(symbol), Some(pos_s)) = (it.next(), it.next()) else {
            continue;
        };
        let Ok(pos) = pos_s.trim().parse::<usize>() else {
            continue;
        };
        if pos > max_pos {
            max_pos = pos;
        }
        pairs.push((pos, symbol.trim().to_string()));
    }
    if pairs.is_empty() {
        return None;
    }
    let mut codes = vec![String::new(); max_pos + 1];
    for (pos, symbol) in pairs {
        codes[pos] = symbol;
    }
    let path = crate::data_paths::minute_file("close.h5");
    let total_rows = hdf5_open_total_rows(&path).unwrap_or(0);
    Some(MinuteMeta {
        date_to_idx,
        codes,
        total_rows,
    })
}

#[cfg(feature = "hdf5")]
fn hdf5_open_total_rows(path: &std::path::Path) -> Option<usize> {
    let f = hdf5::File::open(path).ok()?;
    let ds = f.dataset("data").ok()?;
    ds.shape().into_iter().next()
}

#[cfg(not(feature = "hdf5"))]
fn get_meta() -> Option<Arc<MinuteMeta>> {
    None
}

/// 读某个字段某一天的分钟矩阵，返回**行主序**的扁平 f64（行 = 分钟，列 = 股票）。
/// 不做任何 NaN 处理（调用方自己按 NAN_FROM 置 NaN）。
#[cfg(feature = "hdf5")]
fn read_field_day(field: &str, day_idx: usize, n_codes: usize) -> Option<Vec<f64>> {
    use ndarray::s;
    let path = crate::data_paths::minute_file(&format!("{field}.h5"));
    let file = hdf5::File::open(&path).ok()?;
    let ds = file.dataset("data").ok()?;
    let shape = ds.shape();
    let total = shape[0];
    let ncol = shape[1].min(n_codes);
    let row_start = day_idx * N_MIN;
    if row_start >= total || ncol == 0 {
        return None;
    }
    let row_end = (row_start + N_MIN).min(total);
    let arr: ndarray::Array2<f64> = ds.read_slice_2d(s![row_start..row_end, 0..ncol]).ok()?;
    let n_rows = arr.nrows();
    // 列不足 n_codes 时右侧补 NaN，保证扁平长度恒为 n_rows * n_codes
    let mut out = vec![f64::NAN; n_rows * n_codes];
    for r in 0..n_rows {
        for c in 0..ncol {
            out[r * n_codes + c] = arr[(r, c)];
        }
    }
    Some(out)
}

#[cfg(not(feature = "hdf5"))]
fn read_field_day(_field: &str, _day_idx: usize, _n_codes: usize) -> Option<Vec<f64>> {
    None
}

#[cfg(feature = "hdf5")]
use hdf5_metno as hdf5;

// ============================================================================
// 小工具
// ============================================================================

/// 把每天最后 3 根 K 线置 NaN（旧框架 get_field_min_data 的做法）。
fn nan_tail_rows(a: &mut [f64], n_rows: usize, n_codes: usize) {
    for m in NAN_FROM..n_rows.min(N_MIN) {
        for j in 0..n_codes {
            a[m * n_codes + j] = f64::NAN;
        }
    }
}

/// pandas `DataFrame.sum()`：按列、沿行序跳过 NaN 累加（numpy 成对求和，这里顺序累加）。
fn col_nansum(a: &[f64], n_rows: usize, n_codes: usize) -> Vec<f64> {
    let mut s = vec![0.0f64; n_codes];
    for m in 0..n_rows {
        let row = &a[m * n_codes..m * n_codes + n_codes];
        for j in 0..n_codes {
            let v = row[j];
            if !v.is_nan() {
                s[j] += v;
            }
        }
    }
    s
}

/// pandas `DataFrame.mean()`（axis=0，skipna）：每列在 [m0, m1) 行上的均值。
fn col_nanmean(a: &[f64], n_rows: usize, n_codes: usize, m0: usize, m1: usize) -> Vec<f64> {
    let mut s = vec![0.0f64; n_codes];
    let mut c = vec![0usize; n_codes];
    for m in m0..m1.min(n_rows) {
        let row = &a[m * n_codes..m * n_codes + n_codes];
        for j in 0..n_codes {
            let v = row[j];
            if !v.is_nan() {
                s[j] += v;
                c[j] += 1;
            }
        }
    }
    (0..n_codes)
        .map(|j| if c[j] == 0 { f64::NAN } else { s[j] / c[j] as f64 })
        .collect()
}

/// np.log(...) 之后 `.replace([inf,-inf], nan)`，再摆成「股票优先」的 [j*m1 + m]。
fn log_cols(row_major: &[f64], n_rows: usize, n_codes: usize, m1: usize) -> Vec<f64> {
    let mut out = vec![f64::NAN; n_codes * m1];
    for m in 0..m1.min(n_rows) {
        for j in 0..n_codes {
            let v = row_major[m * n_codes + j];
            let l = v.ln();
            out[j * m1 + m] = if l.is_finite() { l } else { f64::NAN };
        }
    }
    out
}

/// f64 的全序比较：列里已经滤掉 NaN，partial_cmp 不会返回 None。
/// （并列取哪一个是无所谓的，order statistic 的值相同。）
fn cmp_f64(a: &f64, b: &f64) -> std::cmp::Ordering {
    a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
}

/// numpy `_compute_virtual_index` + `_get_indexes` + `_lerp` 的 linear 分位数。
/// `v` 是该列的非 NaN 值（顺序任意），会被就地打乱。
fn quantile_linear(v: &mut [f64], n: usize, q: f64) -> f64 {
    if n == 0 {
        return f64::NAN;
    }
    if n == 1 {
        return v[0];
    }
    let virtual_index = (n as f64) * q + (1.0 + q * (1.0 - 1.0 - 1.0)) - 1.0;
    if virtual_index >= (n - 1) as f64 {
        let (_, max_v, _) = v.select_nth_unstable_by(n - 1, cmp_f64);
        return *max_v;
    }
    if virtual_index < 0.0 {
        let (_, min_v, _) = v.select_nth_unstable_by(0, cmp_f64);
        return *min_v;
    }
    let prev = virtual_index.floor();
    let gamma = virtual_index - prev;
    let pi = prev as usize;
    let ni = pi + 1;
    let (_, a, _) = v.select_nth_unstable_by(pi, cmp_f64);
    let a = *a;
    let (_, b, _) = v[ni..].select_nth_unstable_by(0, cmp_f64);
    let b = *b;
    let diff = b - a;
    let mut r = a + diff * gamma;
    if gamma >= 0.5 {
        r = b - diff * (1.0 - gamma);
    }
    r
}

// ============================================================================
// Welford 相关系数（逐行照抄 pandas/_libs/algos.pyx 的 nancorr）
// ============================================================================

/// 返回下三角 `low`，low[xi][yi] = corr(xi, yi)（xi >= yi）。
/// logl 是「股票优先」的 [j*n + m] 排布，n = 用于相关的根数（120）。
fn corr_low_triangle(logl: &[f64], n_codes: usize, n: usize) -> Vec<Vec<f64>> {
    (0..n_codes)
        .into_par_iter()
        .map(|xi| {
            let a = &logl[xi * n..xi * n + n];
            let mut row = vec![f64::NAN; xi + 1];
            for yi in 0..=xi {
                let b = &logl[yi * n..yi * n + n];
                let mut nobs: i64 = 0;
                let mut meanx = 0.0f64;
                let mut meany = 0.0f64;
                let mut ssqdmx = 0.0f64;
                let mut ssqdmy = 0.0f64;
                let mut covxy = 0.0f64;
                for i in 0..n {
                    let vx = a[i];
                    let vy = b[i];
                    if vx.is_finite() && vy.is_finite() {
                        nobs += 1;
                        let inv = 1.0 / (nobs as f64);
                        let dx = vx - meanx;
                        let dy = vy - meany;
                        meanx += inv * dx;
                        meany += inv * dy;
                        ssqdmx += (vx - meanx) * dx;
                        ssqdmy += (vy - meany) * dy;
                        covxy += (vx - meanx) * dy;
                    }
                }
                row[yi] = if nobs < 1 {
                    f64::NAN
                } else {
                    let divisor = (ssqdmx * ssqdmy).sqrt();
                    if divisor != 0.0 {
                        covxy / divisor
                    } else {
                        f64::NAN
                    }
                };
            }
            row
        })
        .collect()
}

// ============================================================================
// 横截面入口
// ============================================================================

/// 一天全市场：返回 (股票代码, 股票优先排布的因子值)。
pub fn compute_cross(date: i64) -> Option<(Vec<String>, Vec<f32>)> {
    let meta = get_meta()?;
    let day_idx = *meta.date_to_idx.get(&date)?;
    let n_codes = meta.codes.len();
    if n_codes == 0 {
        return None;
    }

    let close_raw = read_field_day("close", day_idx, n_codes)?;
    let ab_raw = read_field_day("act_buy_vol_sum", day_idx, n_codes)?;
    let as_raw = read_field_day("act_sell_vol_sum", day_idx, n_codes)?;
    let bid1_raw = read_field_day("bid_size_1_mean", day_idx, n_codes)?;
    let ask1_raw = read_field_day("ask_size_1_mean", day_idx, n_codes)?;
    let n_rows = close_raw.len() / n_codes;
    if n_rows == 0
        || ab_raw.len() / n_codes != n_rows
        || as_raw.len() / n_codes != n_rows
        || bid1_raw.len() / n_codes != n_rows
        || ask1_raw.len() / n_codes != n_rows
    {
        return None;
    }

    // 1) 每天最后 3 根置 NaN
    let mut close = close_raw;
    let mut act_buy_raw = ab_raw;
    let mut act_sell_raw = as_raw;
    let mut bid1 = bid1_raw;
    let mut ask1 = ask1_raw;
    nan_tail_rows(&mut close, n_rows, n_codes);
    nan_tail_rows(&mut act_buy_raw, n_rows, n_codes);
    nan_tail_rows(&mut act_sell_raw, n_rows, n_codes);
    nan_tail_rows(&mut bid1, n_rows, n_codes);
    nan_tail_rows(&mut ask1, n_rows, n_codes);

    // 2) 派生量
    let mut rets = vec![f64::NAN; n_rows * n_codes];
    for m in 1..n_rows {
        for j in 0..n_codes {
            let i = m * n_codes + j;
            rets[i] = close[i] / close[i - n_codes] - 1.0;
        }
    }
    let close_sum = col_nansum(&close, n_rows, n_codes);
    let ab_sum = col_nansum(&act_buy_raw, n_rows, n_codes);
    let as_sum = col_nansum(&act_sell_raw, n_rows, n_codes);
    let mut volumes = vec![f64::NAN; n_rows * n_codes];
    let mut act_buy = vec![f64::NAN; n_rows * n_codes];
    let mut act_sell = vec![f64::NAN; n_rows * n_codes];
    let mut ab_ratio = vec![f64::NAN; n_rows * n_codes];
    let mut imbalance1 = vec![f64::NAN; n_rows * n_codes];
    for i in 0..n_rows * n_codes {
        let j = i % n_codes;
        volumes[i] = close[i] / close_sum[j];
        act_buy[i] = act_buy_raw[i] / ab_sum[j];
        act_sell[i] = act_sell_raw[i] / as_sum[j];
        ab_ratio[i] = act_buy[i] / volumes[i];
        imbalance1[i] = (ask1[i] - bid1[i]) / (ask1[i] + bid1[i]);
    }

    // drs 索引：0=rets 1=volumes 2=act_buy 3=act_sell 4=ab_ratio
    let drs: [&Vec<f64>; N_DR] = [&rets, &volumes, &act_buy, &act_sell, &ab_ratio];
    // dcs 索引与 DC_SUFFIX 对齐
    let dcs: [&Vec<f64>; N_DC] = [
        &rets,
        &volumes,
        &act_buy,
        &act_sell,
        &ab_ratio,
        &imbalance1,
    ];

    // 3) 每个 dr 的上午 / 下午列均值
    let mut dm: Vec<Vec<f64>> = Vec::with_capacity(N_DR);
    let mut da: Vec<Vec<f64>> = Vec::with_capacity(N_DR);
    for k in 0..N_DR {
        dm.push(col_nanmean(drs[k], n_rows, n_codes, 0, N_MORNING));
        da.push(col_nanmean(drs[k], n_rows, n_codes, N_MORNING, n_rows));
    }

    // 4) 每个 dc 一次：相关系数 → 分位数 → 掩码 → 加权均值
    //    每个 dc 的输出按 [j*45 + (g*5 + k)] 排布，最后再搬到全局的股票优先布局。
    let n_corr = N_MORNING.min(n_rows);
    let per_dc: Vec<Vec<f64>> = dcs
        .par_iter()
        .map(|dc| {
            let logl = log_cols(dc, n_rows, n_codes, n_corr);
            let low = corr_low_triangle(&logl, n_codes, n_corr);
            let mut vals = vec![f64::NAN; n_codes * N_DC_OUT];
            vals.par_chunks_mut(N_DC_OUT)
                .enumerate()
                .for_each(|(j, chunk)| {
                    // 第 j 列（下三角里取，i>=j 取 low[i][j]，i<j 取 low[j][i]）
                    let mut col: Vec<f64> = Vec::with_capacity(n_codes);
                    for i in 0..n_codes {
                        let v = if i >= j { low[i][j] } else { low[j][i] };
                        if !v.is_nan() {
                            col.push(v);
                        }
                    }
                    let n = col.len();
                    let thr_large = quantile_linear(&mut col, n, 0.99);
                    let thr_small = quantile_linear(&mut col, n, 0.01);

                    // 一趟扫完整列，累加 5 个 dr × (上午/下午) × (large/small) 的和与个数
                    let mut sum_lm = [0.0f64; N_DR];
                    let mut cnt_lm = [0usize; N_DR];
                    let mut sum_sm = [0.0f64; N_DR];
                    let mut cnt_sm = [0usize; N_DR];
                    let mut sum_la = [0.0f64; N_DR];
                    let mut cnt_la = [0usize; N_DR];
                    let mut sum_sa = [0.0f64; N_DR];
                    let mut cnt_sa = [0usize; N_DR];
                    for i in 0..n_codes {
                        let v = if i >= j { low[i][j] } else { low[j][i] };
                        if v.is_nan() {
                            continue;
                        }
                        let big = v >= thr_large;
                        let small = v <= thr_small;
                        if !big && !small {
                            continue;
                        }
                        for k in 0..N_DR {
                            if big {
                                let m = dm[k][i];
                                if !m.is_nan() {
                                    sum_lm[k] += m;
                                    cnt_lm[k] += 1;
                                }
                                let a = da[k][i];
                                if !a.is_nan() {
                                    sum_la[k] += a;
                                    cnt_la[k] += 1;
                                }
                            }
                            if small {
                                let m = dm[k][i];
                                if !m.is_nan() {
                                    sum_sm[k] += m;
                                    cnt_sm[k] += 1;
                                }
                                let a = da[k][i];
                                if !a.is_nan() {
                                    sum_sa[k] += a;
                                    cnt_sa[k] += 1;
                                }
                            }
                        }
                    }
                    let mean = |s: f64, c: usize| if c == 0 { f64::NAN } else { s / c as f64 };
                    for k in 0..N_DR {
                        let mlm = mean(sum_lm[k], cnt_lm[k]);
                        let msm = mean(sum_sm[k], cnt_sm[k]);
                        let mla = mean(sum_la[k], cnt_la[k]);
                        let msa = mean(sum_sa[k], cnt_sa[k]);
                        let d = da[k][j];
                        chunk[0 * N_DR + k] = mlm;
                        chunk[1 * N_DR + k] = msm;
                        chunk[2 * N_DR + k] = mlm - d;
                        chunk[3 * N_DR + k] = msm - d;
                        chunk[4 * N_DR + k] = mla;
                        chunk[5 * N_DR + k] = msa;
                        chunk[6 * N_DR + k] = mla - d;
                        chunk[7 * N_DR + k] = msa - d;
                        chunk[8 * N_DR + k] = d;
                    }
                });
            vals
        })
        .collect();

    let mut out = vec![f64::NAN; n_codes * N_OUT];
    for (d, vals) in per_dc.iter().enumerate() {
        for j in 0..n_codes {
            for t in 0..N_DC_OUT {
                out[j * N_OUT + d * N_DC_OUT + t] = vals[j * N_DC_OUT + t];
            }
        }
    }

    let flat: Vec<f32> = out.into_iter().map(|v| v as f32).collect();
    debug_assert_eq!(flat.len(), n_codes * N_OUT);
    Some((meta.codes.clone(), flat))
}

/// hm51 是分钟（横截面）系列，没有逐股实现。
pub fn compute_stock(_code: &str, _date: i64) -> Option<Vec<f32>> {
    None
}
