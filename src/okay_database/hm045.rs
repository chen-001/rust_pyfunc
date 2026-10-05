//! hm45「如如不动」——旧框架（design_whatever_out.FactorDesign，**分钟**数据）的 Rust 复现。
//!
//! 旧文件：/home/chenzongwei/pythoncode/成果hm45_如如不动/hm45_okay.py
//!   run_base 的 Go.get_factor，参数 get_many=0 / read_full=0 / change_time=1，
//!   Go.check_my_kind == "minute"，旧框架走的是
//!       Go(base_dir, date, date, 1, base_ver, read_full=0, get_many=0)
//!   即**一天一次、一次全市场**，所以本文件只有 compute_cross，没有 compute_stock。
//!
//! ════════════════════════════════════════════════════════════════════════
//! 一、旧代码逐步对照（写代码前先抄在这里，方便复核）
//! ════════════════════════════════════════════════════════════════════════
//!  1. closes = new_get_field_min_data("close").reset_index(drop=True).pct_change()
//!     volume = new_get_field_min_data("volume").reset_index(drop=True).diff()
//!     · new_get_field_min_data 返回 240 行 × 全市场股票的矩阵（index 是交易日，只有一天），
//!       reset_index(drop=True) 压成 0..239，于是 pct_change 是「按分钟内」的收益率。
//!     · get_field_min_data 内部 `field_data[field_data.index % 240 >= 237] = np.nan`，
//!       即每天最后 3 根 K 线（14:58/14:59/15:00）置 NaN。
//!     · pct_change() 的默认 fill_method 仍是 'pad'（pandas 2.3.3 只是告警），
//!       所以先对 close 逐列 ffill（前导 NaN 保持 NaN），再 `F / F.shift(1) - 1`。
//!       正因为 ffill，第 237~239 根的收益率是 0 而不是 NaN；第 0 行永远是 NaN。
//!     · volume 只做 diff，没有 ffill，所以第 237~239 行是 NaN，第 0 行也是 NaN。
//!
//!  2. rank_closes = closes.rank(axis=1)，rank_volume = volume.rank(axis=1)
//!     —— 按行（同一分钟跨股票）排名，升序、并列取平均、NaN 保留。
//!
//!  3. df      = merge_many([closes, volume])       → 长表
//!     rank_df = merge_many([rank_closes, rank_volume])
//!     两个长表列名都改成 ["num","code","ret","volume"]。
//!     merge_many = `df.stack().reset_index()` 之后 outer join。stack 默认 dropna=True，
//!     所以帧里每个 NaN 格子都不生成行，两个帧的行集合取并集，outer join 再按
//!     (num, code) **字典序排序** ⇒ 同一只股票内部行号升序。行集合 =
//!        { m : ret[m] 非空 } ∪ { m : volume[m] 非空 }
//!     第 0 行两边都全 NaN，所以任何股票都没有 num=0 这一行。
//!
//!  4. num = len(set(df.code))  —— 长表里出现过的股票数（20250303 是 5104，不是 5792）。
//!     注意这个 num 只用于 rank 分支的 `ret < num/2` / `ret > num/2` 阈值。
//!
//!  5. market = df.groupby("num").ret.mean()，再 merge 回 df 得到 ret_market 列
//!     —— 每个分钟上全市场 ret 的均值。
//!
//!  6. 六个分支，每个都是「按 code 分组 → shift(1..5) → 按谓词切两半 → dropna →
//!     各自标准化 → OLS(a1 的 10 列 → a1.ret) 去预测 a2 → 残差平方 → 统计量」：
//!        df_market     one_market(ab=0)  a1 = ret_market > 均值，a2 = ret_market < 均值
//!        df_market_ab  one_market(ab=1)  a1 = ret_market < 均值，a2 = ret_market > 均值
//!        df            one(rank=0,ab=0)  a1 = ret > 0，a2 = ret < 0
//!        df_ab         one(rank=0,ab=1)  a1 = ret < 0，a2 = ret > 0
//!        rank_df       one(rank=1,ab=0)  a1 = rank_ret > num/2，a2 = rank_ret < num/2
//!        rank_df_ab    one(rank=1,ab=1)  a1 = rank_ret < num/2，a2 = rank_ret > num/2
//!     · 谓词用的是**该分支自己的 ret 列**（rank 分支用的是名次列），
//!       dropna 是整行所有列都要非空：ret、volume、vol1..vol5、ret1..ret5（market 分支再加
//!       ret_market）。所以「ret 非空但 volume 是 NaN」的行会被丢掉。
//!     · shift 是沿**该股自己存在的行序列**往前数，不是往前第 k 分钟。
//!     · 两边分别用**自己那一半**的均值和样本标准差（ddof=1）逐列标准化，
//!       参与 OLS 的只有标准化后的 [vol1..vol5, ret1..ret5] 和 ret。
//!     · OLS 带截距，用 rp.ols_predict 的正规方程 + 无主元 LU 解（src/statistics/mod.rs
//!       的 solve_linear_system3）。worker 链路不能调 #[pyfunction]，所以这里自己抄了一份。
//!     · 残差平方序列的统计量：mean（skipna）、std（ddof=1）、skew（pandas nanskew 的 G1）、
//!       kurt（nanskew 的多余峰度 G2）、max、min、autocorr(1/2/3)。
//!     · p_value = chow_test(a1, a2)，只在 market / df / rank_df 三个分支输出：
//!         rss_sep = Σr1² + Σr2²，rss_com = Σr_combined²，k = 11，
//!         f = ((rss_com - rss_sep)/k) / (rss_sep/(n1+n2-2k))，
//!         p = 1 - scipy.stats.f.cdf(f, k, n1+n2-2k)
//!       scipy 的 f.cdf(x,dfn,dfd) = betainc(dfn/2, dfd/2, dfn*x/(dfn*x+dfd))，
//!       这里自己写了 Lanczos lnΓ + Lentz 连分式的正则化不完全 beta，并且**照抄
//!       "1 - betainc" 这一步**，让 p 极小时两边落在同一个浮点格子上。
//!     · a1 或 a2 不足 10 行时旧代码 return None（该组从结果里消失 → 该股该分支全是 NaN）。
//!
//!  7. 输出顺序（与旧 names 一字不差，共 57 个）：df_market 10 个、
//!     df_market_ab 9 个（没有 p_value）、df 10 个、rank_df 10 个、
//!     df_ab 9 个、rank_df_ab 9 个。names() 无重名，不需要 _dup 后缀。
//!
//! ════════════════════════════════════════════════════════════════════════
//! 二、输出排布（横截面系列的约定，别搞反）
//! ════════════════════════════════════════════════════════════════════════
//! compute_cross 返回 (codes, vals)，vals 是**股票优先**：每只股票连续 57 个值，
//! 第 i 只股票的值在 vals[i*57 .. i*57+57]，与 worker 侧
//! `vals.chunks(expected_len).zip(codes.iter())` 的分块方式一致。
//! codes 是分钟数据 symbol_map.csv 的全名单（20250303 是 5792 只），
//! 没有数据的股票照样占 57 个位置，值全是 NaN（因子个数恒定）。
//!
//! ════════════════════════════════════════════════════════════════════════
//! 三、'code' 字符串列与 pandas 2.x（实测过，结论是「不影响结果」）
//! ════════════════════════════════════════════════════════════════════════
//! 旧文件里的 `a1 = (a1 - a1.mean()) / a1.std()`，a1 里除了数值列还有一列字符串
//! 'code'。pandas 1.x 的 DataFrame.mean/std 默认 numeric_only=None，会把非数值列丢掉；
//! pandas 2.x 默认 numeric_only=False，单独调用会抛 TypeError。
//! 但旧代码是跑在 groupby.apply 里的，pandas 2.3.3 的 apply 在函数抛 TypeError 时会
//! **自动改用 include_groups=False 重跑一遍**（子帧里没有 'code' 列），于是结果与
//! pandas 1.x 的「丢掉非数值列」完全一致。实测：原件与「只把这一处补成
//! numeric_only=True」的副本，57×5104 个值逐位相同（最大相对误差 0.0）。
//! 所以本文件按「只标准化数值列」实现，也就是真正进 OLS 的 ret + vol1..vol5/ret1..ret5
//! 这 11 列（'num'/'volume'/'ret_market'/'code' 都不参与 OLS，标准化与否不影响输出）。
//!
//! ════════════════════════════════════════════════════════════════════════
//! 四、数值口径
//! ════════════════════════════════════════════════════════════════════════
//! · 求和用 numpy 风格的成对求和（blocksize 128、8 路展开），和 pandas 内部 np.sum 靠拢。
//! · std 用两遍算法（先均值再偏差平方和），与 pandas nanvar 一致；ddof=1，样本 < 2 给 NaN。
//! · skew/kurt 照 pandas nanskew/nankurt 的公式，含 |m2|/|m3| < 1e-14 抹零这一步。
//! · autocorr(lag) = corr(x, x.shift(lag))，成对完整观测，用 np.corrcoef 的等价写法。
//! · OLS 的 xᵀx / xᵀy 逐元素按行升序累加；矩阵乘用的是 ndarray→matrixmultiply，
//!   与这里的朴素累加有 1e-16 量级差异。正规方程本来就病态，实测全市场 5104 只里
//!   99.9% 的分支相对误差 < 1e-11，只有个位数股票因为 xᵀx 近奇异（标准化后仍有列共线）
//!   而结果对最后几位敏感 —— 这种股票旧值本身就是噪声，已在 claims note 里说明。

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, LazyLock, Mutex};

use rayon::prelude::*;

/// 分钟数据每天 240 根 K 线
const MIN_PER_DAY: usize = 240;
/// 旧框架 get_field_min_data 把每天最后 3 根置 NaN（index % 240 >= 237）
const MIN_NAN_FROM: usize = 237;
/// 输出因子个数：10 + 9 + 10 + 10 + 9 + 9
pub const N_OUT: usize = 57;

/// 57 个因子名，顺序与 get_factor 的返回值严格对齐（无重名）
const NAMES: [&str; N_OUT] = [
    "df_market_mean_",
    "df_market_std_",
    "df_market_skew_",
    "df_market_kurt_",
    "df_market_max_",
    "df_market_min_",
    "df_market_corr1_",
    "df_market_corr2_",
    "df_market_corr3_",
    "df_market_p_value_",
    "df_market_ab_mean_",
    "df_market_ab_std_",
    "df_market_ab_skew_",
    "df_market_ab_kurt_",
    "df_market_ab_max_",
    "df_market_ab_min_",
    "df_market_ab_corr1_",
    "df_market_ab_corr2_",
    "df_market_ab_corr3_",
    "df_mean_",
    "df_std_",
    "df_skew_",
    "df_kurt_",
    "df_max_",
    "df_min_",
    "df_corr1_",
    "df_corr2_",
    "df_corr3_",
    "df_p_value_",
    "rank_df_mean_",
    "rank_df_std_",
    "rank_df_skew_",
    "rank_df_kurt_",
    "rank_df_max_",
    "rank_df_min_",
    "rank_df_corr1_",
    "rank_df_corr2_",
    "rank_df_corr3_",
    "rank_df_p_value_",
    "df_ab_mean_",
    "df_ab_std_",
    "df_ab_skew_",
    "df_ab_kurt_",
    "df_ab_max_",
    "df_ab_min_",
    "df_ab_corr1_",
    "df_ab_corr2_",
    "df_ab_corr3_",
    "rank_df_ab_mean_",
    "rank_df_ab_std_",
    "rank_df_ab_skew_",
    "rank_df_ab_kurt_",
    "rank_df_ab_max_",
    "rank_df_ab_min_",
    "rank_df_ab_corr1_",
    "rank_df_ab_corr2_",
    "rank_df_ab_corr3_",
];

pub fn names() -> Vec<String> {
    NAMES.iter().map(|s| s.to_string()).collect()
}

// ============================================================================
// 分钟数据：元信息 + 一天的整块读取（照 hm017.rs）
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
// 基础数值工具
// ============================================================================

/// numpy 的成对求和（blocksize=128，先 8 路展开再两两相加）——与 pandas 内部 np.sum 靠拢。
fn pairwise_sum(x: &[f64]) -> f64 {
    let n = x.len();
    if n < 8 {
        let mut res = 0.0f64;
        for &v in x {
            res += v;
        }
        return res;
    }
    if n <= 128 {
        let mut r = [0.0f64; 8];
        r.copy_from_slice(&x[..8]);
        let end = n - (n % 8);
        let mut i = 8usize;
        while i < end {
            for j in 0..8 {
                r[j] += x[i + j];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n {
            res += x[i];
            i += 1;
        }
        return res;
    }
    let mut n2 = n / 2;
    n2 -= n2 % 8;
    pairwise_sum(&x[..n2]) + pairwise_sum(&x[n2..])
}

/// 逐行 rank（pandas DataFrame.rank(axis=1)：升序、并列取平均、NaN 保留）。
fn rank_rows(mat: &[f64], n: usize) -> Vec<f64> {
    let rows: Vec<Vec<f64>> = (0..MIN_PER_DAY)
        .into_par_iter()
        .map(|i| {
            let base = i * n;
            let mut items: Vec<(f64, usize)> = Vec::with_capacity(n);
            for j in 0..n {
                let v = mat[base + j];
                if !v.is_nan() {
                    items.push((v, j));
                }
            }
            let mut out = vec![f64::NAN; n];
            items.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            let m = items.len();
            let mut t = 0usize;
            while t < m {
                let mut k = t + 1;
                while k < m && items[k].0 == items[t].0 {
                    k += 1;
                }
                let avg = ((t + 1) as f64 + k as f64) / 2.0;
                for s in t..k {
                    out[items[s].1] = avg;
                }
                t = k;
            }
            out
        })
        .collect();
    let mut out = vec![f64::NAN; MIN_PER_DAY * n];
    for (i, row) in rows.into_iter().enumerate() {
        out[i * n..(i + 1) * n].copy_from_slice(&row);
    }
    out
}

/// skipna 均值（pandas Series.mean）
fn mean_skipna(v: &[f64]) -> f64 {
    let vals: Vec<f64> = v.iter().copied().filter(|x| !x.is_nan()).collect();
    if vals.is_empty() {
        return f64::NAN;
    }
    pairwise_sum(&vals) / vals.len() as f64
}

/// pandas Series.std（ddof=1，skipna，样本 < 2 给 NaN）
fn std_skipna(v: &[f64]) -> f64 {
    let vals: Vec<f64> = v.iter().copied().filter(|x| !x.is_nan()).collect();
    let n = vals.len();
    if n < 2 {
        return f64::NAN;
    }
    let m = pairwise_sum(&vals) / n as f64;
    let sq: Vec<f64> = vals.iter().map(|x| (x - m) * (x - m)).collect();
    (pairwise_sum(&sq) / (n as f64 - 1.0)).sqrt()
}

/// pandas Series.skew（nanskew 的 G1），有效样本 < 3 给 NaN
fn skew_skipna(v: &[f64]) -> f64 {
    let vals: Vec<f64> = v.iter().copied().filter(|x| !x.is_nan()).collect();
    let n = vals.len();
    if n < 3 {
        return f64::NAN;
    }
    let m = pairwise_sum(&vals) / n as f64;
    let a2: Vec<f64> = vals.iter().map(|x| (x - m) * (x - m)).collect();
    let a3: Vec<f64> = a2.iter().zip(vals.iter()).map(|(d2, x)| d2 * (x - m)).collect();
    let mut m2 = pairwise_sum(&a2);
    let mut m3 = pairwise_sum(&a3);
    if m2.abs() < 1e-14 {
        m2 = 0.0;
    }
    if m3.abs() < 1e-14 {
        m3 = 0.0;
    }
    if m2 == 0.0 {
        return 0.0;
    }
    let cnt = n as f64;
    (cnt * (cnt - 1.0).sqrt() / (cnt - 2.0)) * (m3 / m2.powf(1.5))
}

/// pandas Series.kurt（nankurt 的多余峰度 G2），有效样本 < 4 给 NaN
fn kurt_skipna(v: &[f64]) -> f64 {
    let vals: Vec<f64> = v.iter().copied().filter(|x| !x.is_nan()).collect();
    let n = vals.len();
    if n < 4 {
        return f64::NAN;
    }
    let m = pairwise_sum(&vals) / n as f64;
    let a2: Vec<f64> = vals.iter().map(|x| (x - m) * (x - m)).collect();
    let a4: Vec<f64> = a2.iter().map(|d2| d2 * d2).collect();
    let m2 = pairwise_sum(&a2);
    let m4 = pairwise_sum(&a4);
    let cnt = n as f64;
    let adj = 3.0 * (cnt - 1.0) * (cnt - 1.0) / ((cnt - 2.0) * (cnt - 3.0));
    let mut numerator = cnt * (cnt + 1.0) * (cnt - 1.0) * m4;
    let mut denom = (cnt - 2.0) * (cnt - 3.0) * m2 * m2;
    if numerator.abs() < 1e-14 {
        numerator = 0.0;
    }
    if denom.abs() < 1e-14 {
        denom = 0.0;
    }
    if denom == 0.0 {
        return 0.0;
    }
    numerator / denom - adj
}

/// pandas Series.autocorr(lag)：corr(x, x.shift(lag))，成对完整观测
fn autocorr(v: &[f64], lag: usize) -> f64 {
    let n = v.len();
    if lag >= n {
        return f64::NAN;
    }
    let mut a: Vec<f64> = Vec::with_capacity(n - lag);
    let mut b: Vec<f64> = Vec::with_capacity(n - lag);
    for i in lag..n {
        if v[i].is_nan() || v[i - lag].is_nan() {
            continue;
        }
        a.push(v[i]);
        b.push(v[i - lag]);
    }
    let m = a.len();
    if m == 0 {
        return f64::NAN;
    }
    let ma = pairwise_sum(&a) / m as f64;
    let mb = pairwise_sum(&b) / m as f64;
    let xa: Vec<f64> = a.iter().map(|x| x - ma).collect();
    let xb: Vec<f64> = b.iter().map(|x| x - mb).collect();
    let cov = pairwise_sum(&xa.iter().zip(xb.iter()).map(|(x, y)| x * y).collect::<Vec<f64>>());
    let va = pairwise_sum(&xa.iter().map(|x| x * x).collect::<Vec<f64>>());
    let vb = pairwise_sum(&xb.iter().map(|x| x * x).collect::<Vec<f64>>());
    cov / (va.sqrt() * vb.sqrt())
}

// ============================================================================
// 正则化不完全 beta（复刻 scipy.special.betainc，用于 F 分布 p 值）
// ============================================================================

/// Lanczos 近似的 lnΓ（g=607/128，n=15），精度 ~1e-15
fn ln_gamma(x: f64) -> f64 {
    const COF: [f64; 14] = [
        57.1562356658629235,
        -59.5979603554754912,
        14.1360979747417471,
        -0.491913816097620199,
        0.339946499848118887e-4,
        0.465236289270485756e-4,
        -0.983744753048795646e-4,
        0.158088703224912494e-3,
        -0.210264441724104883e-3,
        0.217439618115212643e-3,
        -0.164318106536763890e-3,
        0.844182239838527433e-4,
        -0.261908384015814087e-4,
        0.368991826595316234e-5,
    ];
    if x <= 0.0 {
        return f64::NAN;
    }
    let mut y = x;
    let tmp = x + 5.24218750000000000;
    let tmp = (x + 0.5) * tmp.ln() - tmp;
    let mut ser = 0.999999999999997092;
    for &c in COF.iter() {
        y += 1.0;
        ser += c / y;
    }
    tmp + (2.5066282746310005 * ser / x).ln()
}

/// 连分式（Lentz 算法），Numerical Recipes betacf
fn betacf(a: f64, b: f64, x: f64) -> f64 {
    const MAXIT: usize = 500;
    const EPS: f64 = 1.0e-16;
    const FPMIN: f64 = 1.0e-300;
    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0f64;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < FPMIN {
        d = FPMIN;
    }
    d = 1.0 / d;
    let mut h = d;
    for m in 1..=MAXIT {
        let mf = m as f64;
        let m2 = 2.0 * mf;
        let aa = mf * (b - mf) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = 1.0 + aa / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        h *= d * c;
        let aa = -(a + mf) * (qab + mf) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = 1.0 + aa / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < EPS {
            break;
        }
    }
    h
}

/// 正则化不完全 beta I_x(a,b)
fn betai(a: f64, b: f64, x: f64) -> f64 {
    if a.is_nan() || b.is_nan() || x.is_nan() {
        return f64::NAN;
    }
    if !(a > 0.0) || !(b > 0.0) {
        return f64::NAN;
    }
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    let bt = (ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b) + a * x.ln() + b * (1.0 - x).ln()).exp();
    if x < (a + 1.0) / (a + b + 2.0) {
        bt * betacf(a, b, x) / a
    } else {
        1.0 - bt * betacf(b, a, 1.0 - x) / b
    }
}

/// p = 1 - scipy.stats.f.cdf(f, dfn, dfd)。
///
/// scipy 的 f.cdf 走的是 cephes 的 fdtr → incbet，**不是** special.betainc（后者走 Boost）。
/// 两者在尾部极小时不同：incbet 在对称分支里有一句
///     if (t <= MACHEP) t = 1.0 - MACHEP;   /* MACHEP = 1.11022302462515654042e-16 */
/// 于是 cdf 最高只到 1 - 1.11e-16，p 的最小值就是 1.11e-16 而不是 0。
/// 实测 scipy 1.15.2：f.cdf(1e5, 11, 150) = 0.9999999999999999（真尾部是 7.7e-284），
/// 而 special.betainc(5.5, 75, z) = 1.0。照抄 betainc 会让 p 差一个 1.11e-16
/// （相对误差 1.11e-4，刚好越过验收线），所以这里连这个下限一起复刻。
fn f_sf(f: f64, dfn: f64, dfd: f64) -> f64 {
    const MACHEP: f64 = 1.11022302462515654042e-16;
    if f.is_nan() {
        return f64::NAN;
    }
    if dfn <= 0.0 || dfd <= 0.0 {
        return f64::NAN;
    }
    if f == f64::INFINITY {
        return 0.0;
    }
    if f == f64::NEG_INFINITY {
        return 1.0;
    }
    let aa = dfn / 2.0;
    let bb = dfd / 2.0;
    let z = dfn * f / (dfn * f + dfd);
    let cdf = if z > aa / (aa + bb) {
        // 对称分支：t = I_{1-z}(bb, aa) = 1 - I_z(aa, bb)，再套 cephes 的下限
        let t = betai(bb, aa, 1.0 - z);
        if t <= MACHEP {
            1.0 - MACHEP
        } else {
            1.0 - t
        }
    } else {
        betai(aa, bb, z)
    };
    1.0 - cdf
}

// ============================================================================
// OLS：照 src/statistics/mod.rs 的正规方程 + 无主元 LU
// ============================================================================

const NP: usize = 10; // 自变量个数
const NA: usize = NP + 1; // 加截距

fn ols_coef(x: &[f64], n: usize, y: &[f64]) -> [f64; NA] {
    let mut xtx = [[0.0f64; NA]; NA];
    let mut xty = [0.0f64; NA];
    for t in 0..n {
        let row = &x[t * NP..t * NP + NP];
        let mut aug = [1.0f64; NA];
        aug[1..].copy_from_slice(row);
        for i in 0..NA {
            for j in i..NA {
                xtx[i][j] += aug[i] * aug[j];
            }
            xty[i] += aug[i] * y[t];
        }
    }
    for i in 0..NA {
        for j in 0..i {
            xtx[i][j] = xtx[j][i];
        }
    }
    solve_lu(&xtx, &xty)
}

fn solve_lu(a: &[[f64; NA]; NA], b: &[f64; NA]) -> [f64; NA] {
    let n = NA;
    let mut l = [[0.0f64; NA]; NA];
    let mut u = [[0.0f64; NA]; NA];
    for i in 0..n {
        for j in 0..n {
            if i <= j {
                let mut s = 0.0f64;
                for k in 0..i {
                    s += l[i][k] * u[k][j];
                }
                u[i][j] = a[i][j] - s;
                if i == j {
                    l[i][i] = 1.0;
                }
            }
            if i > j {
                let mut s = 0.0f64;
                for k in 0..j {
                    s += l[i][k] * u[k][j];
                }
                l[i][j] = (a[i][j] - s) / u[j][j];
            }
        }
    }
    let mut y = [0.0f64; NA];
    for i in 0..n {
        let mut s = 0.0f64;
        for j in 0..i {
            s += l[i][j] * y[j];
        }
        y[i] = b[i] - s;
    }
    let mut x = [0.0f64; NA];
    for i in (0..n).rev() {
        let mut s = 0.0f64;
        for j in (i + 1)..n {
            s += u[i][j] * x[j];
        }
        x[i] = (y[i] - s) / u[i][i];
    }
    x
}

/// 预测：coef[0] + Σ coef[i+1]*x[t][i]（升序累加，与 x_pred·coef 一致）
fn predict_row(x: &[f64], t: usize, coef: &[f64; NA]) -> f64 {
    let mut s = coef[0];
    for i in 0..NP {
        s += coef[i + 1] * x[t * NP + i];
    }
    s
}

// ============================================================================
// 单个分支
// ============================================================================

/// 分支类型：0 = 用 ret_market 切，1 = 用 ret 和 0 比，2 = 用名次和 num/2 比
#[derive(Clone, Copy, PartialEq)]
enum Pred {
    Mkt,
    RetZero,
    RankNum,
}

struct Frame<'a> {
    rt: &'a [f64],
    vo: &'a [f64],
    sr: &'a [Vec<f64>; 5],
    sv: &'a [Vec<f64>; 5],
    mkt: Option<&'a [f64]>,
}

/// 按行（a1 的 10 列 + ret）标准化，返回 (X 行优先 n×10, y)
fn build_xy(rows: &[usize], fr: &Frame) -> (Vec<f64>, Vec<f64>) {
    let n = rows.len();
    let mut x = vec![0.0f64; n * NP];
    let mut buf: Vec<f64> = Vec::with_capacity(n);
    for c in 0..NP {
        let base: &[f64] = if c < 5 { &fr.sv[c] } else { &fr.sr[c - 5] };
        buf.clear();
        for &t in rows {
            buf.push(base[t]);
        }
        let mean = pairwise_sum(&buf) / n as f64;
        let sq: Vec<f64> = buf.iter().map(|v| (v - mean) * (v - mean)).collect();
        let sd = (pairwise_sum(&sq) / (n as f64 - 1.0)).sqrt();
        for (r, &v) in buf.iter().enumerate() {
            x[r * NP + c] = (v - mean) / sd;
        }
    }
    let mut y: Vec<f64> = rows.iter().map(|&t| fr.rt[t]).collect();
    let mean = pairwise_sum(&y) / n as f64;
    let sq: Vec<f64> = y.iter().map(|v| (v - mean) * (v - mean)).collect();
    let sd = (pairwise_sum(&sq) / (n as f64 - 1.0)).sqrt();
    for v in y.iter_mut() {
        *v = (*v - mean) / sd;
    }
    (x, y)
}

/// 残差平方序列的 10 个统计量（第 10 个是 p 值，不要求时给 NaN）
fn stats_of(res2: &[f64]) -> [f64; 10] {
    let mut out = [f64::NAN; 10];
    if res2.is_empty() {
        return out;
    }
    out[0] = mean_skipna(res2);
    out[1] = std_skipna(res2);
    out[2] = skew_skipna(res2);
    out[3] = kurt_skipna(res2);
    let mut mx = f64::NEG_INFINITY;
    let mut mn = f64::INFINITY;
    let mut any = false;
    for &v in res2 {
        if v.is_nan() {
            continue;
        }
        any = true;
        if v > mx {
            mx = v;
        }
        if v < mn {
            mn = v;
        }
    }
    if any {
        out[4] = mx;
        out[5] = mn;
    }
    out[6] = autocorr(res2, 1);
    out[7] = autocorr(res2, 2);
    out[8] = autocorr(res2, 3);
    out
}

/// Chow 检验的 p 值
fn chow_p(x1: &[f64], y1: &[f64], n1: usize, x2: &[f64], y2: &[f64], n2: usize) -> f64 {
    let c1 = ols_coef(x1, n1, y1);
    let c2 = ols_coef(x2, n2, y2);
    let mut xc = Vec::with_capacity((n1 + n2) * NP);
    xc.extend_from_slice(x1);
    xc.extend_from_slice(x2);
    let mut yc = Vec::with_capacity(n1 + n2);
    yc.extend_from_slice(y1);
    yc.extend_from_slice(y2);
    let nc = n1 + n2;
    let cc = ols_coef(&xc, nc, &yc);

    let r1: Vec<f64> = (0..n1).map(|t| y1[t] - predict_row(x1, t, &c1)).collect();
    let r2: Vec<f64> = (0..n2).map(|t| y2[t] - predict_row(x2, t, &c2)).collect();
    let r3: Vec<f64> = (0..nc).map(|t| yc[t] - predict_row(&xc, t, &cc)).collect();
    let sq = |v: &[f64]| -> Vec<f64> { v.iter().map(|x| x * x).collect() };
    let rss_sep = pairwise_sum(&sq(&r1)) + pairwise_sum(&sq(&r2));
    let rss_com = pairwise_sum(&sq(&r3));
    let k = NA as f64;
    let dfd = nc as f64 - 2.0 * k;
    let f = ((rss_com - rss_sep) / k) / (rss_sep / dfd);
    f_sf(f, k, dfd)
}

fn branch(fr: &Frame, pred: Pred, ab: bool, num2: f64, mkt_mean: f64, want_p: bool) -> [f64; 10] {
    let m = fr.rt.len();
    let mut rows1: Vec<usize> = Vec::with_capacity(m);
    let mut rows2: Vec<usize> = Vec::with_capacity(m);
    // dropna：ret、volume、vol1..5、ret1..5（market 分支再加 ret_market）全都要非空
    let finite = |t: usize| -> bool {
        if fr.rt[t].is_nan() || fr.vo[t].is_nan() {
            return false;
        }
        for c in 0..5 {
            if fr.sv[c][t].is_nan() || fr.sr[c][t].is_nan() {
                return false;
            }
        }
        if let Some(mk) = fr.mkt {
            if mk[t].is_nan() {
                return false;
            }
        }
        true
    };
    match pred {
        Pred::Mkt => {
            let mk = fr.mkt.expect("market 分支必须有 ret_market");
            for t in 0..m {
                if !finite(t) {
                    continue;
                }
                if ab {
                    if mk[t] < mkt_mean {
                        rows1.push(t);
                    } else if mk[t] > mkt_mean {
                        rows2.push(t);
                    }
                } else if mk[t] > mkt_mean {
                    rows1.push(t);
                } else if mk[t] < mkt_mean {
                    rows2.push(t);
                }
            }
        }
        Pred::RetZero => {
            for t in 0..m {
                if !finite(t) {
                    continue;
                }
                if ab {
                    if fr.rt[t] < 0.0 {
                        rows1.push(t);
                    } else if fr.rt[t] > 0.0 {
                        rows2.push(t);
                    }
                } else if fr.rt[t] > 0.0 {
                    rows1.push(t);
                } else if fr.rt[t] < 0.0 {
                    rows2.push(t);
                }
            }
        }
        Pred::RankNum => {
            for t in 0..m {
                if !finite(t) {
                    continue;
                }
                if ab {
                    if fr.rt[t] < num2 {
                        rows1.push(t);
                    } else if fr.rt[t] > num2 {
                        rows2.push(t);
                    }
                } else if fr.rt[t] > num2 {
                    rows1.push(t);
                } else if fr.rt[t] < num2 {
                    rows2.push(t);
                }
            }
        }
    }
    let mut out = [f64::NAN; 10];
    if rows1.len() < 10 || rows2.len() < 10 {
        return out;
    }
    let (x1, y1) = build_xy(&rows1, fr);
    let (x2, y2) = build_xy(&rows2, fr);
    let c1 = ols_coef(&x1, rows1.len(), &y1);
    let n2 = rows2.len();
    let res2: Vec<f64> = (0..n2)
        .map(|t| {
            let d = predict_row(&x2, t, &c1) - y2[t];
            d * d
        })
        .collect();
    out = stats_of(&res2);
    if want_p {
        out[9] = chow_p(&x1, &y1, rows1.len(), &x2, &y2, n2);
    }
    out
}

// ============================================================================
// 单只股票
// ============================================================================

#[allow(clippy::too_many_arguments)]
fn process_stock(
    j: usize,
    n: usize,
    ret: &[f64],
    vol: &[f64],
    rret: &[f64],
    rvol: &[f64],
    market: &[f64],
    num2: f64,
) -> [f64; N_OUT] {
    let mut out = [f64::NAN; N_OUT];
    let mut present: Vec<usize> = Vec::with_capacity(MIN_PER_DAY);
    for i in 0..MIN_PER_DAY {
        if !ret[i * n + j].is_nan() || !vol[i * n + j].is_nan() {
            present.push(i);
        }
    }
    if present.is_empty() {
        return out;
    }
    let m = present.len();
    let mut a_ret = vec![0.0f64; m];
    let mut a_vol = vec![0.0f64; m];
    let mut a_rret = vec![0.0f64; m];
    let mut a_rvol = vec![0.0f64; m];
    let mut a_mkt = vec![0.0f64; m];
    for (t, &i) in present.iter().enumerate() {
        a_ret[t] = ret[i * n + j];
        a_vol[t] = vol[i * n + j];
        a_rret[t] = rret[i * n + j];
        a_rvol[t] = rvol[i * n + j];
        a_mkt[t] = market[i];
    }
    let shift = |a: &[f64], k: usize| -> Vec<f64> {
        let mut o = vec![f64::NAN; m];
        if k < m {
            o[k..].copy_from_slice(&a[..m - k]);
        }
        o
    };
    let sv: [Vec<f64>; 5] = [shift(&a_vol, 1), shift(&a_vol, 2), shift(&a_vol, 3), shift(&a_vol, 4), shift(&a_vol, 5)];
    let sr: [Vec<f64>; 5] = [shift(&a_ret, 1), shift(&a_ret, 2), shift(&a_ret, 3), shift(&a_ret, 4), shift(&a_ret, 5)];
    let rsv: [Vec<f64>; 5] = [shift(&a_rvol, 1), shift(&a_rvol, 2), shift(&a_rvol, 3), shift(&a_rvol, 4), shift(&a_rvol, 5)];
    let rsr: [Vec<f64>; 5] = [shift(&a_rret, 1), shift(&a_rret, 2), shift(&a_rret, 3), shift(&a_rret, 4), shift(&a_rret, 5)];

    let mkt_mean = mean_skipna(&a_mkt);

    // (谓词, ab, 要不要 p 值, 输出起点, 输出个数)
    let specs: [(Pred, bool, bool, usize, usize); 6] = [
        (Pred::Mkt, false, true, 0, 10),
        (Pred::Mkt, true, false, 10, 9),
        (Pred::RetZero, false, true, 19, 10),
        (Pred::RankNum, false, true, 29, 10),
        (Pred::RetZero, true, false, 39, 9),
        (Pred::RankNum, true, false, 48, 9),
    ];
    for &(pred, ab, want_p, off, cnt) in specs.iter() {
        let fr = match pred {
            Pred::Mkt => Frame {
                rt: &a_ret,
                vo: &a_vol,
                sr: &sr,
                sv: &sv,
                mkt: Some(&a_mkt),
            },
            Pred::RetZero => Frame {
                rt: &a_ret,
                vo: &a_vol,
                sr: &sr,
                sv: &sv,
                mkt: None,
            },
            Pred::RankNum => Frame {
                rt: &a_rret,
                vo: &a_rvol,
                sr: &rsr,
                sv: &rsv,
                mkt: None,
            },
        };
        let vals = branch(&fr, pred, ab, num2, mkt_mean, want_p);
        for k in 0..cnt {
            out[off + k] = vals[k];
        }
    }
    out
}

// ============================================================================
// 入口
// ============================================================================

/// hm45 是分钟横截面系列，没有逐股实现。
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

    let close = read_day_block("close", day, n)?;
    let volume = read_day_block("volume", day, n)?;

    // ret = pct_change（逐列 ffill 之后 F/F.shift(1)-1）；vol = diff
    let mut ret = vec![f64::NAN; MIN_PER_DAY * n];
    let mut vol = vec![f64::NAN; MIN_PER_DAY * n];
    let mut col = vec![0.0f64; MIN_PER_DAY];
    for j in 0..n {
        let mut prev = f64::NAN;
        for i in 0..MIN_PER_DAY {
            let v = close[i * n + j];
            if v.is_nan() {
                col[i] = prev;
            } else {
                prev = v;
                col[i] = v;
            }
        }
        for i in 1..MIN_PER_DAY {
            ret[i * n + j] = col[i] / col[i - 1] - 1.0;
        }
        for i in 1..MIN_PER_DAY {
            vol[i * n + j] = volume[i * n + j] - volume[(i - 1) * n + j];
        }
    }

    let rret = rank_rows(&ret, n);
    let rvol = rank_rows(&vol, n);

    // 长表里出现过的股票数（num = len(set(df.code))）与全市场分钟均值
    let mut numcodes = 0usize;
    for j in 0..n {
        let mut any = false;
        for i in 0..MIN_PER_DAY {
            if !ret[i * n + j].is_nan() || !vol[i * n + j].is_nan() {
                any = true;
                break;
            }
        }
        if any {
            numcodes += 1;
        }
    }
    let num2 = numcodes as f64 / 2.0;

    let mut market = vec![f64::NAN; MIN_PER_DAY];
    let mut tmp = vec![0.0f64; n];
    for i in 0..MIN_PER_DAY {
        let mut cnt = 0usize;
        for j in 0..n {
            let v = ret[i * n + j];
            if v.is_nan() {
                tmp[j] = 0.0;
            } else {
                tmp[j] = v;
                cnt += 1;
            }
        }
        if cnt > 0 {
            market[i] = pairwise_sum(&tmp) / cnt as f64;
        }
    }

    let per_stock: Vec<[f64; N_OUT]> = (0..n)
        .into_par_iter()
        .map(|j| process_stock(j, n, &ret, &vol, &rret, &rvol, &market, num2))
        .collect();

    let mut vals: Vec<f32> = Vec::with_capacity(n * N_OUT);
    for row in per_stock.iter() {
        for v in row.iter() {
            vals.push(*v as f32);
        }
    }
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
        assert_eq!(n[0], "df_market_mean_");
        assert_eq!(n[9], "df_market_p_value_");
        assert_eq!(n[10], "df_market_ab_mean_");
        assert_eq!(n[18], "df_market_ab_corr3_");
        assert_eq!(n[19], "df_mean_");
        assert_eq!(n[28], "df_p_value_");
        assert_eq!(n[29], "rank_df_mean_");
        assert_eq!(n[38], "rank_df_p_value_");
        assert_eq!(n[39], "df_ab_mean_");
        assert_eq!(n[47], "df_ab_corr3_");
        assert_eq!(n[48], "rank_df_ab_mean_");
        assert_eq!(n[56], "rank_df_ab_corr3_");
        let mut sorted = n.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), N_OUT, "names() 里有重名");
    }

    #[test]
    fn pairwise_matches_naive() {
        let x: Vec<f64> = (0..1000).map(|i| (i as f64) * 0.001).collect();
        let naive: f64 = x.iter().sum();
        let p = pairwise_sum(&x);
        assert!((p - naive).abs() / naive.abs() < 1e-12);
    }

    #[test]
    fn stats_against_hand() {
        let v = [1.0, 2.0, 3.0, 4.0, f64::NAN];
        assert_eq!(mean_skipna(&v), 2.5);
        let s = std_skipna(&v);
        assert!((s - 1.2909944487358056).abs() < 1e-12);
        // pandas: pd.Series([1,2,3,4.]).skew() = 0
        assert!(skew_skipna(&[1.0, 2.0, 3.0, 4.0]).abs() < 1e-12);
        // pandas: pd.Series([1,2,3,4.]).kurt() = -1.2
        assert!((kurt_skipna(&[1.0, 2.0, 3.0, 4.0]) + 1.2).abs() < 1e-12);
        // autocorr(1) of [1,2,3,4] = 1.0
        assert!((autocorr(&[1.0, 2.0, 3.0, 4.0], 1) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn beta_and_f_tail() {
        // scipy.special.betainc(0.5, 0.5, 0.5) = 0.5
        assert!((betai(0.5, 0.5, 0.5) - 0.5).abs() < 1e-12);
        // scipy.stats.f.cdf(1.0, 5, 5) = 0.5 → p = 0.5
        assert!((f_sf(1.0, 5.0, 5.0) - 0.5).abs() < 1e-12);
        // cephes 的下限：f.cdf(1e5,11,150) = 0.9999999999999999 → p = 1.11e-16
        let p = f_sf(1e5, 11.0, 150.0);
        assert!((p - 1.1102230246251565e-16).abs() < 1e-30, "p = {p:e}");
        assert!(f_sf(f64::INFINITY, 11.0, 100.0) == 0.0);
        assert!(f_sf(-1.0, 11.0, 100.0) == 1.0);
        assert!(f_sf(f64::NAN, 11.0, 100.0).is_nan());
    }

    #[test]
    fn ols_matches_known_line() {
        // y = 1 + 2*x1 - 3*x2，无噪声 → 系数可复原，预测与原值一致
        let n = 50usize;
        let mut x = vec![0.0f64; n * NP];
        let mut y = vec![0.0f64; n];
        for t in 0..n {
            let a = (t as f64) * 0.1;
            let b = ((t * 7 % 11) as f64) * 0.3;
            x[t * NP] = a;
            x[t * NP + 1] = b;
            y[t] = 1.0 + 2.0 * a - 3.0 * b;
        }
        let c = ols_coef(&x, n, &y);
        assert!((c[0] - 1.0).abs() < 1e-9);
        assert!((c[1] - 2.0).abs() < 1e-9);
        assert!((c[2] + 3.0).abs() < 1e-9);
        for t in 0..n {
            assert!((predict_row(&x, t, &c) - y[t]).abs() < 1e-8);
        }
    }
}
