//! hm28「天机不可泄露」——旧框架（design_whatever_out.FactorDesign，**分钟**数据）的 Rust 复现。
//!
//! 旧文件：/home/chenzongwei/pythoncode/成果hm28_天机不可泄露/hm28_okay.py
//!   run_base 的 Go.get_factor，参数 get_many=0 / read_full=0 / change_time=1，
//!   Go.check_my_kind == "minute"，所以旧框架走的是
//!       Go(base_dir, date, date, 1, base_ver, read_full=0, get_many=0)
//!   也就是**一天一次、返回全市场一张表**，因此本文件只有 compute_cross，没有 compute_stock。
//!
//! ════════════════════════════════════════════════════════════════════
//! 一、旧代码逐步对照（写代码前先抄在这里，方便复核）
//! ════════════════════════════════════════════════════════════════════
//! 1) closes = FactorDesign.new_get_field_min_data("close").reset_index(drop=True).pct_change()
//!    volume = FactorDesign.new_get_field_min_data("volume").reset_index(drop=True)
//!    · new_get_field_min_data 返回 240 行 × 全市场股票的矩阵，index 是交易日（这里只有一天），
//!      reset_index(drop=True) 把它压成 0..239 的整数索引 —— 于是 pct_change 是**整天首尾相接**的，
//!      只不过只有一天，等价于按分钟内做。
//!    · get_field_min_data 内部有一步 `field_data[field_data.index % 240 >= 237] = np.nan`，
//!      即每天**最后 3 根** K 线（14:58/14:59/15:00）置 NaN。注意是 3 根不是 4 根，
//!      库里 rp.read_minute_field 置的是 4 根（236..239），差这一根会让第二个因子从
//!      0.225174 变成 0.223800（相对差 6e-3，过不了 1e-4 的验收），所以本文件自己读 h5，
//!      不用 rp 的那个函数。
//!    · pct_change() 的默认 fill_method 仍是 'pad'（pandas 2.3.3 只是告警），
//!      所以先对 close 做 ffill（前导 NaN 保持 NaN）再 `F / F.shift(1) - 1`。
//!      正因为 ffill，第 237~239 根的收益率是 0 而不是 NaN。
//!
//! 2) sing1(closes, volume)：
//!      volume = volume.diff()                     ← 只有 sing1 对成交量做差分
//!      df = merge_many([closes, volume])          ← 两个宽表按 (行号, 股票) 外连接成长表
//!      df.columns = ["num","code","ret","volume"]
//!      按 code 分组，每组：
//!        a = a.assign(vol1=a.volume.shift(1), ..., vol5=a.volume.shift(5),
//!                     ret1=a.ret.shift(1),   ..., ret5=a.ret.shift(5))
//!        a = a[a.ret != 0].dropna()               ← 只留价格真的变了的分钟，且 12 列全非空
//!        if a.shape[0] > 20:
//!            algt.ret = sign(a.ret).astype(int).replace(-1, 0)      （1=涨，0=跌）
//!            lgt = smf.logit("ret~vol1..vol5+ret1..ret5", data=algt).fit(disp=0)
//!            ols = smf.ols  ("ret~vol1..vol5+ret1..ret5", data=a).fit()
//!            返回 lgt.prsquared 与 ols.rsquared；任何异常 → 两个都 NaN
//!      → 因子 1（旧 names_single[0] = "ols_lgt_allday"）= lgt_r - ols_r
//!
//! 3) sing2(closes, volume)：
//!      rank_closes = closes.rank(axis=1)          ← **按行**（同一分钟跨股票）的名次
//!      rank_volume = volume.rank(axis=1)
//!      df      = merge_many([closes, volume])
//!      rank_df = merge_many([rank_closes, rank_volume])
//!      两组各自按 code 分组、shifts、a[a.ret != 0].dropna()、a.shape[0] > 20，
//!      只跑 **OLS**（sing2 的 one() 里那段 algt 是死代码，算完没人用，也不会抛异常）：
//!          ols  = OLS(真实 ret  ~ vol1..vol5 + ret1..ret5)
//!          olsr = OLS(名次  ret  ~ 名次 vol1..vol5 + 名次 ret1..ret5)
//!      → 因子 2（旧 names_single[1] = "lgt_rank_nodiff_allday"）= df.ols_r - rank_df.ols_r
//!    · 名次是横截面量，所以这个系列必须一次算一天全市场（compute_cross），逐股算不出来。
//!
//! 4) get_factor 最后 return sing1 + sing2，共 2 个 Series（index 都是股票代码）。
//!    旧框架 names_save = ['lgt_rank_nodiff_allday', 'ols_lgt_allday']，
//!    也就是位置 1 存成 lgt_rank_nodiff_allday、位置 0 存成 ols_lgt_allday。
//!    names() 的顺序必须与返回值一致：["ols_lgt_allday", "lgt_rank_nodiff_allday"]。
//!
//! ════════════════════════════════════════════════════════════════════
//! 二、merge_many 的语义（决定 shift 到底跨不跨行）
//! ════════════════════════════════════════════════════════════════════
//! merge_many 是 `df.stack().reset_index()` 之后再 outer join。`stack()` 默认 dropna=True，
//! 所以帧里每一个 NaN 格子都不生成行；两个帧的行集合取并集。
//!   单股的单日行集合 = { m : ret[m] 非空 } ∪ { m : 第二列[m] 非空 }
//! 实测（20250303 全市场 5792 只）：这个并集里的行号在每只股票内部都是升序且不重复，
//! 所以 shift(k) 就是「沿时间往前数第 k 个**存在的**行」，不是往前第 k 分钟。
//! 本文件的 build_design 就按「先取 presence 序列，再沿序列做 shift」实现。
//!
//! 另外 `a[a.ret != 0].dropna()` 里的 dropna() 是整行 12 列全都要非空：
//! ret、volume（第二列）、vol1..vol5、ret1..ret5。
//!
//! ════════════════════════════════════════════════════════════════════
//! 三、两个统计量必须逐位复刻 statsmodels（这是本系列最难的地方）
//! ════════════════════════════════════════════════════════════════════
//! 【OLS 的 R²】statsmodels 的 OLS.fit() 用 np.linalg.pinv（SVD）解，R² = 1 - ssr/centered_tss。
//!   这里改用「列归一化 + 带列主元的 Householder QR」，用 Q^T y 的尾部平方和直接得到 ssr
//!   （列缩放不改变列空间，所以残差数学上完全相同）。实测全市场 5104 只与 statsmodels
//!   的最大相对误差 7.7e-12，远低于 1e-4 的门槛。
//!
//! 【logit 的 McFadden 伪 R²】statsmodels 的 Logit.fit() 默认 method='newton'：
//!   · 起始值 start_params = 全 0（Logit 没有 start_params 属性，exog 非空 → `[0.0]*k`）
//!   · score = Xᵀ(y-p)/n，hess = -Xᵀdiag(p(1-p))X/n，迭代公式 beta ← beta - solve(hess, score)
//!   · **每次迭代前给 Hessian 的对角线加 1e-10**（_fit_newton 的 ridge_factor），
//!     但存进 mle_retvals 的那份 Hessian 是**不加** ridge 的
//!   · 终止条件 `maxiter < 35 且 any(|Δbeta| > 1e-8)`，tol 默认 1e-8
//!   · 拟合完之后 LikelihoodModel.fit 还要算 `Hinv = np.linalg.inv(-retvals['Hessian'])/nobs`，
//!     **这一步会在 Hessian 恰好奇异时抛 LinAlgError**，被旧代码的 except 吞成 NaN。
//!     实测：完全不收敛的那些股票（完全分离，p 精确等于 0/1，w 全为 0）正是走这条路
//!     —— 20250303 全市场 5104 只里有 25 只，旧代码给的就是 NaN。
//!     所以本文件在两处都做「LU 主元恰好为 0 → 整个 logit 失败」的判定。
//!   · prsquared = 1 - llf/llnull，llf = Σ log(1/(1+exp(-(2y-1)·Xbeta)))；
//!     llnull 旧框架是自己跑 nm+bfgs 拟合截距模型得到的，这里用解析值
//!     n·[p·ln p + (1-p)·ln(1-p)] 代替。实测两者差 ~1e-11（相对），可以忽略。
//!
//! 实测全市场（20250303，旧框架原值对照）：
//!   · ols_lgt_allday：NaN 个数 67 = 67，非 NaN 位置最大相对误差 2.3e-6
//!   · lgt_rank_nodiff_allday：NaN 个数 42 = 42，非 NaN 位置最大相对误差 1.4e-11
//! 67 = 42（样本不足 20 行，整条被 groupby.apply 丢掉）+ 25（logit 奇异）。
//!
//! ════════════════════════════════════════════════════════════════════
//! 四、输出排布（横截面系列的约定，别搞反）
//! ════════════════════════════════════════════════════════════════════
//! compute_cross 返回 (codes, vals)，vals 是**股票优先**：第 i 只股票的值在
//! vals[i*2 .. i*2+2]，与 worker 侧 `vals.chunks(expected_len).zip(codes.iter())` 一致。
//! codes 取分钟数据 symbol_map 的全量列（含全 NaN 的列，它们的值就是 NaN），
//! 这样与旧框架写盘时的股票全集一致。

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, LazyLock, Mutex};

use rayon::prelude::*;

/// 每天 240 根 1min K 线（09:31~11:30 + 13:01~15:00）
const N_MIN: usize = 240;
/// 旧框架 `field_data.index % 240 >= 237` 置 NaN 的起始位置（14:58/14:59/15:00）
const NAN_FROM: usize = 237;
/// 回归列数：截距 + vol1..vol5 + ret1..ret5
const K: usize = 11;
/// 旧代码 `if a.shape[0] > 20`
const MIN_ROWS: usize = 20;
/// 每个股票输出的因子数
const N_OUT: usize = 2;

/// 因子名顺序必须与 compute_cross 的返回值顺序一致
/// （旧 names_single = ['ols_lgt_allday', 'lgt_rank_nodiff_allday']）。
pub fn names() -> Vec<String> {
    vec![
        "ols_lgt_allday".to_string(),
        "lgt_rank_nodiff_allday".to_string(),
    ]
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
    // calendar_map.csv：表头之后每行一个 date_min，行号 = day_idx
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
    // 总行数：拿 close.h5 的 shape
    let path = crate::data_paths::minute_file("close.h5");
    let total_rows = hdf5_open_total_rows(&path).unwrap_or(0);
    Some(MinuteMeta { date_to_idx, codes, total_rows })
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

/// pandas `Series.diff()`：第一行 NaN，其余相减（NaN 传染）。
fn diff_rows(src: &[f64], n_rows: usize, n_codes: usize) -> Vec<f64> {
    let mut out = vec![f64::NAN; n_rows * n_codes];
    for m in 1..n_rows {
        for j in 0..n_codes {
            let i = m * n_codes + j;
            out[i] = src[i] - src[i - n_codes];
        }
    }
    out
}

/// pandas `DataFrame.pct_change()`（fill_method='pad'，pandas 2.3.3 的默认）：
/// 先逐列 ffill（前导 NaN 保持 NaN），再 `F / F.shift(1) - 1`。
fn pct_change_ffill(close: &[f64], n_rows: usize, n_codes: usize) -> Vec<f64> {
    let mut f = close.to_vec();
    for m in 1..n_rows {
        for j in 0..n_codes {
            let i = m * n_codes + j;
            if f[i].is_nan() {
                f[i] = f[i - n_codes];
            }
        }
    }
    let mut out = vec![f64::NAN; n_rows * n_codes];
    for m in 1..n_rows {
        for j in 0..n_codes {
            let i = m * n_codes + j;
            // 与 pandas 的 `data / shifted - 1` 同序（先除后减）
            out[i] = f[i] / f[i - n_codes] - 1.0;
        }
    }
    out
}

/// pandas `DataFrame.rank(axis=1)`：默认 method='average'、升序、NaN 保持 NaN。
/// 名次从 1 开始，并列取平均名次。
fn rank_rows(src: &[f64], n_rows: usize, n_codes: usize) -> Vec<f64> {
    let mut out = vec![f64::NAN; n_rows * n_codes];
    let mut buf: Vec<(f64, usize)> = Vec::with_capacity(n_codes);
    for m in 0..n_rows {
        buf.clear();
        for j in 0..n_codes {
            let v = src[m * n_codes + j];
            if !v.is_nan() {
                buf.push((v, j));
            }
        }
        // 并列名次只与相等关系有关，排序是否稳定都不影响结果；
        // 但为了同一输入每次结果一致，按 (值, 列号) 排全序。
        buf.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        let n = buf.len();
        let mut s = 0usize;
        while s < n {
            let mut e = s + 1;
            while e < n && buf[e].0 == buf[s].0 {
                e += 1;
            }
            let avg = ((s + 1) as f64 + e as f64) / 2.0;
            for t in s..e {
                out[m * n_codes + buf[t].1] = avg;
            }
            s = e;
        }
    }
    out
}

// ============================================================================
// 线性代数
// ============================================================================

/// 不带主元检查的 LU 分解；返回 true 表示「主元恰好为 0」（numpy 的
/// np.linalg.solve / np.linalg.inv 走 dgetrf，遇到这种情况抛 LinAlgError）。
/// 逐行照抄 LAPACK dgetf2 的做法：取剩余列绝对值最大的行做主元（并列取最靠前），
/// 乘子用 `A[i][j] * (1/A[j][j])`（不是除法），再做秩 1 更新。
fn lu_zero_pivot(h: &[[f64; K]; K]) -> bool {
    let mut a = *h;
    for j in 0..K {
        let mut p = j;
        let mut best = a[j][j].abs();
        for i in (j + 1)..K {
            let v = a[i][j].abs();
            if v > best {
                best = v;
                p = i;
            }
        }
        if a[p][j] == 0.0 {
            return true;
        }
        if p != j {
            a.swap(j, p);
        }
        let inv = 1.0 / a[j][j];
        for i in (j + 1)..K {
            a[i][j] *= inv;
        }
        for k in (j + 1)..K {
            let t = a[j][k];
            if t != 0.0 {
                for i in (j + 1)..K {
                    a[i][k] -= a[i][j] * t;
                }
            }
        }
    }
    false
}

/// 解 K×K 线性方程组（LU，部分主元）。主元恰好为 0 时返回 None。
fn lu_solve(h: &[[f64; K]; K], b: &[f64; K]) -> Option<[f64; K]> {
    let mut a = *h;
    let mut rhs = *b;
    for j in 0..K {
        let mut p = j;
        let mut best = a[j][j].abs();
        for i in (j + 1)..K {
            let v = a[i][j].abs();
            if v > best {
                best = v;
                p = i;
            }
        }
        if a[p][j] == 0.0 {
            return None;
        }
        if p != j {
            a.swap(j, p);
            rhs.swap(j, p);
        }
        let inv = 1.0 / a[j][j];
        for i in (j + 1)..K {
            a[i][j] *= inv;
        }
        for k in (j + 1)..K {
            let t = a[j][k];
            if t != 0.0 {
                for i in (j + 1)..K {
                    a[i][k] -= a[i][j] * t;
                }
            }
        }
        for i in (j + 1)..K {
            rhs[i] -= a[i][j] * rhs[j];
        }
    }
    let mut x = [0.0f64; K];
    for i in (0..K).rev() {
        let mut s = rhs[i];
        for k in (i + 1)..K {
            s -= a[i][k] * x[k];
        }
        x[i] = s / a[i][i];
    }
    Some(x)
}

/// OLS 的 R²（截距已在设计矩阵里，centered_tss）。
///
/// `x` 是行主序的 n×K 设计矩阵。做法：列归一化（不改变列空间，只改善条件数）
/// → 带列主元的 Householder QR → ssr 直接取 Qᵀy 尾部的平方和。
fn ols_r2(x: &[f64], n: usize, y: &[f64]) -> f64 {
    // 列归一化 + 转成列主序
    let mut scale = [1.0f64; K];
    for c in 0..K {
        let mut s = 0.0f64;
        for i in 0..n {
            let v = x[i * K + c];
            s += v * v;
        }
        let nrm = s.sqrt();
        scale[c] = if nrm > 0.0 { 1.0 / nrm } else { 1.0 };
    }
    let mut a = vec![0.0f64; K * n];
    for c in 0..K {
        for i in 0..n {
            a[c * n + i] = x[i * K + c] * scale[c];
        }
    }
    let mut qty = y.to_vec();
    let mut diag = [0.0f64; K];
    let mut v = vec![0.0f64; n];
    for j in 0..K {
        // 选主元列：剩余行上的平方和最大者（并列取最靠前的列）
        let mut best = j;
        let mut bestv = -1.0f64;
        for p in j..K {
            let mut s = 0.0f64;
            for i in j..n {
                let t = a[p * n + i];
                s += t * t;
            }
            if s > bestv {
                bestv = s;
                best = p;
            }
        }
        if best != j {
            for i in 0..n {
                a.swap(j * n + i, best * n + i);
            }
        }
        let mut nrm = 0.0f64;
        for i in j..n {
            let t = a[j * n + i];
            nrm += t * t;
        }
        let nrm = nrm.sqrt();
        if nrm == 0.0 {
            diag[j] = 0.0;
            continue;
        }
        let x0 = a[j * n + j];
        let alpha = if x0 >= 0.0 { -nrm } else { nrm };
        let m = n - j;
        for i in 0..m {
            v[i] = a[j * n + j + i];
        }
        v[0] -= alpha;
        let mut vtv = 0.0f64;
        for i in 0..m {
            vtv += v[i] * v[i];
        }
        if vtv == 0.0 {
            diag[j] = 0.0;
            continue;
        }
        let beta = 2.0 / vtv;
        for c in j..K {
            let mut dot = 0.0f64;
            for i in 0..m {
                dot += v[i] * a[c * n + j + i];
            }
            let f = beta * dot;
            for i in 0..m {
                a[c * n + j + i] -= f * v[i];
            }
        }
        let mut dot = 0.0f64;
        for i in 0..m {
            dot += v[i] * qty[j + i];
        }
        let f = beta * dot;
        for i in 0..m {
            qty[j + i] -= f * v[i];
        }
        diag[j] = alpha;
    }
    // 秩的判定：列已归一化，用最大对角元的 1e-12 做相对阈值
    let mut mx = 0.0f64;
    for j in 0..K {
        let v = diag[j].abs();
        if v > mx {
            mx = v;
        }
    }
    let tol = mx * 1e-12;
    let mut r = 0usize;
    while r < K && diag[r].abs() > tol {
        r += 1;
    }
    let mut ssr = 0.0f64;
    for i in r..n {
        ssr += qty[i] * qty[i];
    }
    let mut mean = 0.0f64;
    for i in 0..n {
        mean += y[i];
    }
    mean /= n as f64;
    let mut tss = 0.0f64;
    for i in 0..n {
        let d = y[i] - mean;
        tss += d * d;
    }
    1.0 - ssr / tss
}

/// statsmodels Logit 的 McFadden 伪 R²（见文件头第三节）。
/// 返回 None 表示旧代码会抛异常 → 整条 NaN。
fn logit_prsq(x: &[f64], n: usize, y: &[f64]) -> Option<f64> {
    const MAXIT: usize = 35;
    const TOL: f64 = 1e-8;
    const RIDGE: f64 = 1e-10;

    let nf = n as f64;
    let mut beta = [0.0f64; K];
    let mut old = [f64::INFINITY; K];
    let mut p = vec![0.0f64; n];
    let mut it = 0usize;
    while it < MAXIT && (0..K).any(|c| (beta[c] - old[c]).abs() > TOL) {
        for i in 0..n {
            let mut lp = 0.0f64;
            for c in 0..K {
                lp += x[i * K + c] * beta[c];
            }
            p[i] = 1.0 / (1.0 + (-lp).exp());
        }
        let mut h = [[0.0f64; K]; K];
        let mut sc = [0.0f64; K];
        for i in 0..n {
            let pi = p[i];
            let w = pi * (1.0 - pi);
            let yi = y[i];
            let base = i * K;
            for c in 0..K {
                let xc = x[base + c];
                sc[c] += xc * (yi - pi);
                for d in c..K {
                    h[c][d] += xc * x[base + d] * w;
                }
            }
        }
        for c in 0..K {
            for d in c..K {
                let v = -h[c][d] / nf;
                h[c][d] = v;
                h[d][c] = v;
            }
            h[c][c] += RIDGE;
            sc[c] /= nf;
        }
        let delta = lu_solve(&h, &sc)?;
        old = beta;
        for c in 0..K {
            beta[c] = old[c] - delta[c];
        }
        it += 1;
    }

    // 收尾：statsmodels 之后要 np.linalg.inv(-Hessian)，奇异就抛异常
    for i in 0..n {
        let mut lp = 0.0f64;
        for c in 0..K {
            lp += x[i * K + c] * beta[c];
        }
        p[i] = 1.0 / (1.0 + (-lp).exp());
    }
    let mut hf = [[0.0f64; K]; K];
    for i in 0..n {
        let pi = p[i];
        let w = pi * (1.0 - pi);
        let base = i * K;
        for c in 0..K {
            let xc = x[base + c];
            for d in c..K {
                hf[c][d] += xc * x[base + d] * w;
            }
        }
    }
    for c in 0..K {
        for d in c..K {
            let v = -hf[c][d] / nf;
            hf[c][d] = v;
            hf[d][c] = v;
        }
    }
    if lu_zero_pivot(&hf) {
        return None;
    }

    let mut llf = 0.0f64;
    let mut ysum = 0.0f64;
    for i in 0..n {
        let mut lp = 0.0f64;
        for c in 0..K {
            lp += x[i * K + c] * beta[c];
        }
        let z = (2.0 * y[i] - 1.0) * lp;
        let cdf = 1.0 / (1.0 + (-z).exp());
        llf += cdf.ln();
        ysum += y[i];
    }
    let m = ysum / nf;
    let llnull = nf * (m * m.ln() + (1.0 - m) * (1.0 - m).ln());
    Some(1.0 - llf / llnull)
}

// ============================================================================
// 长表组装（merge_many + groupby + shift + 两次筛选）
// ============================================================================

/// 把 (a, b) 两个「宽表列」拼成长表并选出进回归的行，写出 X（行主序 n×K）与 y。
/// 语义与旧代码的
///     df = merge_many([a, b]); a = a.assign(shift...); a = a[a.ret != 0].dropna()
/// 完全一致（a 是 ret 那一列，b 是 volume 那一列）。
/// 返回 false 表示行数不足 20（旧代码 one() 返回 None，整条被丢掉）。
fn build_design(
    a: &[f64],
    b: &[f64],
    n_rows: usize,
    n_codes: usize,
    j: usize,
    seq: &mut Vec<(f64, f64)>,
    xs: &mut Vec<f64>,
    ys: &mut Vec<f64>,
) -> bool {
    seq.clear();
    xs.clear();
    ys.clear();
    // presence 序列：stack() 丢掉 NaN，两个帧 outer join 取并集
    for m in 0..n_rows {
        let i = m * n_codes + j;
        if !a[i].is_nan() || !b[i].is_nan() {
            seq.push((a[i], b[i]));
        }
    }
    let k = seq.len();
    if k <= 5 {
        return false;
    }
    for t in 5..k {
        let (r, v) = seq[t];
        if r.is_nan() || r == 0.0 || v.is_nan() {
            continue;
        }
        let mut ok = true;
        for d in 1..=5 {
            let (rd, vd) = seq[t - d];
            if rd.is_nan() || vd.is_nan() {
                ok = false;
                break;
            }
        }
        if !ok {
            continue;
        }
        xs.push(1.0);
        for d in 1..=5 {
            xs.push(seq[t - d].1); // vol1..vol5
        }
        for d in 1..=5 {
            xs.push(seq[t - d].0); // ret1..ret5
        }
        ys.push(r);
    }
    ys.len() > MIN_ROWS
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
    let vol_raw = read_field_day("volume", day_idx, n_codes)?;
    let n_rows = close_raw.len() / n_codes;
    let n_vol = vol_raw.len() / n_codes;
    if n_rows == 0 || n_vol != n_rows {
        return None;
    }

    // 1) 每天最后 3 根 K 线置 NaN（旧框架 get_field_min_data 的做法）
    let mut close = close_raw;
    let mut vol = vol_raw;
    for m in NAN_FROM..n_rows {
        for j in 0..n_codes {
            let i = m * n_codes + j;
            close[i] = f64::NAN;
            vol[i] = f64::NAN;
        }
    }

    // 2) 收益率（pct_change 默认先 ffill）、成交量差分
    let ret = pct_change_ffill(&close, n_rows, n_codes);
    let vdiff = diff_rows(&vol, n_rows, n_codes);

    // 3) 横截面名次（rank(axis=1)）
    let ret_rank = rank_rows(&ret, n_rows, n_codes);
    let vol_rank = rank_rows(&vol, n_rows, n_codes);

    // 4) 逐股三个回归
    let pairs: Vec<(f64, f64)> = (0..n_codes)
        .into_par_iter()
        .map(|j| {
            let mut seq: Vec<(f64, f64)> = Vec::with_capacity(n_rows);
            let mut xs: Vec<f64> = Vec::with_capacity(n_rows * K);
            let mut ys: Vec<f64> = Vec::with_capacity(n_rows);
            // 因子 1：sing1 —— 带差分的成交量
            let v1 = if build_design(
                &ret, &vdiff, n_rows, n_codes, j, &mut seq, &mut xs, &mut ys,
            ) {
                let n = ys.len();
                // 旧代码：algt.ret = np.sign(a.ret).astype(int).replace(-1, 0)
                // （ret==0 的行已经在上面筛掉了，所以就是 涨=1 / 跌=0）
                let yb: Vec<f64> = ys.iter().map(|&v| if v > 0.0 { 1.0 } else { 0.0 }).collect();
                match logit_prsq(&xs, n, &yb) {
                    Some(p) => p - ols_r2(&xs, n, &ys),
                    None => f64::NAN,
                }
            } else {
                f64::NAN
            };
            // 因子 2：sing2 —— 原成交量与横截面名次两版 OLS 之差
            let ols2 = if build_design(
                &ret, &vol, n_rows, n_codes, j, &mut seq, &mut xs, &mut ys,
            ) {
                Some(ols_r2(&xs, ys.len(), &ys))
            } else {
                None
            };
            let olsr = if build_design(
                &ret_rank, &vol_rank, n_rows, n_codes, j, &mut seq, &mut xs, &mut ys,
            ) {
                Some(ols_r2(&xs, ys.len(), &ys))
            } else {
                None
            };
            let v2 = match (ols2, olsr) {
                (Some(a), Some(b)) => a - b,
                _ => f64::NAN,
            };
            (v1, v2)
        })
        .collect();

    let mut flat: Vec<f32> = Vec::with_capacity(n_codes * N_OUT);
    for (v1, v2) in pairs {
        flat.push(v1 as f32);
        flat.push(v2 as f32);
    }
    debug_assert_eq!(flat.len(), n_codes * N_OUT);
    Some((meta.codes.clone(), flat))
}

/// hm28 是分钟（横截面）系列，没有逐股实现。
pub fn compute_stock(_code: &str, _date: i64) -> Option<Vec<f32>> {
    None
}
