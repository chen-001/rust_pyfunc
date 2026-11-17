//! tail_v4_fulltest_dataset —— fulltest 回测底数据的读盘与预处理全量下沉。
//!
//! 被替换的 Python 段：
//!   `PortfolioRetSummary.__init__`（fulltest_whatever.py 1375-1433）
//!   `prepare_residual_variables`（1537-1555）
//!   `prepare_ret_residual` / `get_residual_by_variables_new` / `get_residual`（1506-1599）
//!   `bare_vars.get_return_data` / `get_restrict_data` / `get_index_ret`
//!   `bare_szba.get_szba_fields`
//!
//! 全流程只用 Rust：HDF5 读盘（hdf5-metno）+ 窗口滚动求和 + 逐日分位缩尾 +
//! 前移一天 + 逐日 OLS 残差（SVD 伪逆）+ 残差截面秩。
//!
//! 输出（date-major，形状 (T, S)，与 Python 的 (S, T) 转置对应）：
//!   ret / ret_sum / restrict / index_ret / ret_resi_raw / ret_resi_raw_1d / ret_resi

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use arrow::array::{Array, Date32Array, Date64Array, Float32Array, Float64Array, Int32Array, StringArray};
use arrow::datatypes::DataType;
use chrono::{Duration, NaiveDate};
use hdf5_metno as hdf5;
use ndarray::Array2;
use nalgebra::{DMatrix, DVector};
use numpy::PyArray;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

const NAN: f64 = f64::NAN;

/// 六位代码 → 带后缀（与 design_whatever.bare_vars.add_symbol_suffix 一致）。
fn add_symbol_suffix(s: &str) -> String {
    if s.contains('.') {
        return s.to_string();
    }
    match s.chars().next() {
        Some('0') | Some('3') => format!("{s}.SZ"),
        Some('6') => format!("{s}.SH"),
        Some('8') => format!("{s}.BJ"),
        _ => format!("{s}.error"),
    }
}

fn format_date(d: i32) -> String {
    format!("{:04}-{:02}-{:02}", d / 10000, (d / 100) % 100, d % 100)
}

fn parse_date_str(s: &str) -> i32 {
    s.replace('-', "").parse().unwrap_or(0)
}

/// calendar_map.csv 里 [start, end] 区间的整数日期（升序）。
fn read_dates(data_dir: &Path, start: i32, end: i32) -> Result<Vec<i32>, String> {
    let path = data_dir.join("calendar_map.csv");
    let file = File::open(&path).map_err(|e| format!("打开 {} 失败: {e}", path.display()))?;
    let mut out = Vec::new();
    for (idx, line) in BufReader::new(file).lines().enumerate() {
        let line = line.map_err(|e| format!("读取 calendar_map 第 {idx} 行失败: {e}"))?;
        let t = line.trim();
        if idx == 0 || t.is_empty() {
            continue;
        }
        let d: i32 = t
            .parse()
            .map_err(|_| format!("calendar_map 第 {idx} 行不是整数日期: {t}"))?;
        if d >= start && d <= end {
            out.push(d);
        }
    }
    Ok(out)
}

/// symbol_map.csv 的全列序（六位代码）。
fn read_stocks(data_dir: &Path) -> Result<Vec<String>, String> {
    let path = data_dir.join("symbol_map.csv");
    let file = File::open(&path).map_err(|e| format!("打开 {} 失败: {e}", path.display()))?;
    let mut out = Vec::new();
    for (idx, line) in BufReader::new(file).lines().enumerate() {
        let line = line.map_err(|e| format!("读取 symbol_map 第 {idx} 行失败: {e}"))?;
        let t = line.trim().trim_matches('"');
        if idx == 0 || t.is_empty() {
            continue;
        }
        let code = t.split(',').next().unwrap_or(t).trim().trim_matches('"');
        out.push(code.to_string());
    }
    Ok(out)
}

fn strip_stock_suffix(s: &str) -> &str {
    match s.find('.') {
        Some(i) => &s[..i],
        None => s,
    }
}

fn parse_calendar_map(path: &Path) -> Result<HashMap<i32, usize>, String> {
    let file = File::open(path).map_err(|e| format!("打开 {} 失败: {e}", path.display()))?;
    let mut out = HashMap::new();
    for (idx, line) in BufReader::new(file).lines().enumerate() {
        let line = line.map_err(|e| format!("读取 calendar_map 第 {idx} 行失败: {e}"))?;
        let t = line.trim();
        if idx == 0 || t.is_empty() {
            continue;
        }
        let d: i32 = t
            .parse()
            .map_err(|_| format!("calendar_map 第 {idx} 行不是整数日期: {t}"))?;
        out.insert(d, idx - 1);
    }
    Ok(out)
}

fn parse_symbol_map(path: &Path) -> Result<HashMap<String, usize>, String> {
    let file = File::open(path).map_err(|e| format!("打开 {} 失败: {e}", path.display()))?;
    let mut out = HashMap::new();
    for (idx, line) in BufReader::new(file).lines().enumerate() {
        let line = line.map_err(|e| format!("读取 symbol_map 第 {idx} 行失败: {e}"))?;
        let t = line.trim().trim_matches('"');
        if idx == 0 || t.is_empty() {
            continue;
        }
        let code = t.split(',').next().unwrap_or(t).trim().trim_matches('"');
        out.insert(code.to_string(), idx - 1);
    }
    Ok(out)
}

/// 读一个 vars 字段并按 (dates, stocks) 对齐成 date-major 面板。
///
/// dates 是整数日期（升序），stocks 可带 .SZ/.SH/.BJ 后缀（会自动去后缀查 symbol_map）。
/// 缺失日期/股票一律填 NaN，与 pandas 的 reindex 同义。
fn read_h5_aligned(
    data_dir: &Path,
    field: &str,
    dates: &[i32],
    stocks: &[String],
) -> Result<Vec<f64>, String> {
    let date_to_row = parse_calendar_map(&data_dir.join("calendar_map.csv"))?;
    let symbol_to_col = parse_symbol_map(&data_dir.join("symbol_map.csv"))?;
    let rows: Vec<Option<usize>> = dates.iter().map(|d| date_to_row.get(d).copied()).collect();
    let cols: Vec<Option<usize>> = stocks
        .iter()
        .map(|s| symbol_to_col.get(strip_stock_suffix(s)).copied())
        .collect();

    let n_t = dates.len();
    let n_s = stocks.len();
    let mut out = vec![NAN; n_t * n_s];

    let path = data_dir.join(format!("{field}.h5"));
    let file = hdf5::File::open(&path).map_err(|e| format!("打开 {} 失败: {e}", path.display()))?;
    let ds = file
        .dataset("data")
        .map_err(|e| format!("{} 缺少 data dataset: {e}", path.display()))?;
    let full: Array2<f64> = ds
        .read_2d()
        .map_err(|e| format!("读取 {} 失败: {e}", path.display()))?;

    for (t, row) in rows.iter().enumerate() {
        let Some(r) = *row else { continue };
        if r >= full.nrows() {
            continue;
        }
        for (j, col) in cols.iter().enumerate() {
            let Some(c) = *col else { continue };
            if c >= full.ncols() {
                continue;
            }
            let v = full[(r, c)];
            out[t * n_s + j] = if v.is_finite() { v } else { NAN };
        }
    }
    Ok(out)
}

/// 指数 forward 收益：(open_t / open_{t-1} - 1) 再前移一天。
///
/// 与 `bare_vars.get_index_ret` 完全同构：**读全历史算差分**，再按目标日期取。
/// 不能只在 [start, end] 区间算，否则区间首日缺前一天的 open、末日缺下一天的 open，
/// 这两天的收益会变成 NaN（实测 hedge 指标因此差 7e-3）。
fn read_index_forward_ret(
    data_dir: &Path,
    field: &str,
    column: &str,
    dates: &[i32],
) -> Result<Vec<f64>, String> {
    let date_to_row = parse_calendar_map(&data_dir.join("calendar_map.csv"))?;
    let symbol_to_col = parse_symbol_map(&data_dir.join("symbol_map.csv"))?;
    let col = symbol_to_col
        .get(column)
        .or_else(|| symbol_to_col.get(strip_stock_suffix(column)))
        .copied()
        .ok_or_else(|| format!("symbol_map 里没有 {column}"))?;

    let path = data_dir.join(format!("{field}.h5"));
    let file = hdf5::File::open(&path).map_err(|e| format!("打开 {} 失败: {e}", path.display()))?;
    let ds = file
        .dataset("data")
        .map_err(|e| format!("{} 缺少 data dataset: {e}", path.display()))?;
    let full: Array2<f64> = ds
        .read_2d()
        .map_err(|e| format!("读取 {} 失败: {e}", path.display()))?;

    let mut ret_by_row = vec![NAN; full.nrows()];
    for r in 1..full.nrows() {
        let a = full[(r - 1, col)];
        let b = full[(r, col)];
        if a.is_finite() && b.is_finite() && a != 0.0 {
            ret_by_row[r - 1] = b / a - 1.0;
        }
    }
    let mut out = vec![NAN; dates.len()];
    for (t, d) in dates.iter().enumerate() {
        if let Some(&r) = date_to_row.get(d) {
            if r < ret_by_row.len() {
                out[t] = ret_by_row[r];
            }
        }
    }
    Ok(out)
}

/// 读一列（指数开盘价）并按日期对齐，返回 (T,)。
fn read_h5_series(
    data_dir: &Path,
    field: &str,
    column: &str,
    dates: &[i32],
) -> Result<Vec<f64>, String> {
    let date_to_row = parse_calendar_map(&data_dir.join("calendar_map.csv"))?;
    let symbol_to_col = parse_symbol_map(&data_dir.join("symbol_map.csv"))?;
    // index_data 的 symbol_map 是带后缀的单列（000852.SH），Base/SzBa 是六位两列，
    // 两种格式都要能查：先按原样查，再按去后缀查。
    let col = symbol_to_col
        .get(column)
        .or_else(|| symbol_to_col.get(strip_stock_suffix(column)))
        .copied()
        .ok_or_else(|| format!("symbol_map 里没有 {column}"))?;

    let path = data_dir.join(format!("{field}.h5"));
    let file = hdf5::File::open(&path).map_err(|e| format!("打开 {} 失败: {e}", path.display()))?;
    let ds = file
        .dataset("data")
        .map_err(|e| format!("{} 缺少 data dataset: {e}", path.display()))?;
    let full: Array2<f64> = ds
        .read_2d()
        .map_err(|e| format!("读取 {} 失败: {e}", path.display()))?;

    let mut out = vec![NAN; dates.len()];
    for (t, d) in dates.iter().enumerate() {
        if let Some(&r) = date_to_row.get(d) {
            if r < full.nrows() && col < full.ncols() {
                let v = full[(r, col)];
                out[t] = if v.is_finite() { v } else { NAN };
            }
        }
    }
    Ok(out)
}

/// pandas `rolling(window, min_periods=1).sum()` 沿日期方向，逐股票。
fn rolling_sum(p: &[f64], n_t: usize, n_s: usize, window: usize) -> Vec<f64> {
    let mut out = vec![NAN; n_t * n_s];
    for j in 0..n_s {
        for t in 0..n_t {
            let lo = t.saturating_sub(window - 1);
            let mut s = 0.0f64;
            let mut cnt = 0usize;
            for k in lo..=t {
                let v = p[k * n_s + j];
                if !v.is_nan() {
                    s += v;
                    cnt += 1;
                }
            }
            out[t * n_s + j] = if cnt == 0 { NAN } else { s };
        }
    }
    out
}

/// numpy 的线性插值分位数（基于 `select_nth_unstable_by` O(n) 局部选择，与全排序逐位一致）。
fn quantile_linear_select(vals: &mut [f64], q: f64) -> f64 {
    let n = vals.len();
    if n == 0 {
        return NAN;
    }
    if n == 1 {
        return vals[0];
    }
    let pos = q * (n - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    let (_, &mut vlo, right) = vals.select_nth_unstable_by(lo, |a, b| a.total_cmp(b));
    if lo == hi {
        vlo
    } else {
        let vhi = right
            .iter()
            .copied()
            .min_by(|a, b| a.total_cmp(b))
            .unwrap_or(vlo);
        let w = pos - lo as f64;
        vlo * (1.0 - w) + vhi * w
    }
}

/// 逐日缩尾：先算当日 2.5% / 97.5% 分位（零 clone + O(n) 局部选择），再把越界值截到边界。
fn winsorize_cols(p: &mut [f64], n_t: usize, n_s: usize) {
    let mut buf: Vec<f64> = Vec::with_capacity(n_s);
    for t in 0..n_t {
        buf.clear();
        for j in 0..n_s {
            let v = p[t * n_s + j];
            if !v.is_nan() {
                buf.push(v);
            }
        }
        if buf.is_empty() {
            continue;
        }
        let lo = quantile_linear_select(&mut buf, 0.025);
        let hi = quantile_linear_select(&mut buf, 0.975);
        for j in 0..n_s {
            let v = p[t * n_s + j];
            if v.is_nan() {
                continue;
            }
            if v > hi {
                p[t * n_s + j] = hi;
            } else if v < lo {
                p[t * n_s + j] = lo;
            }
        }
    }
}

/// pandas `shift(-k, axis=1)`：新表第 t 列 = 原表第 t+k 列，末尾补 NaN。
fn shift_cols_left(p: &[f64], n_t: usize, n_s: usize, k: usize) -> Vec<f64> {
    let mut out = vec![NAN; n_t * n_s];
    for t in 0..n_t {
        let src = t + k;
        if src < n_t {
            out[t * n_s..(t + 1) * n_s].copy_from_slice(&p[src * n_s..(src + 1) * n_s]);
        }
    }
    out
}

/// pandas `shift(1, axis=0)`：新表第 t 行 = 原表第 t-1 行（沿日期下移一格）。
fn shift_rows_down(p: &[f64], n_t: usize, n_s: usize) -> Vec<f64> {
    let mut out = vec![NAN; n_t * n_s];
    for t in 1..n_t {
        out[t * n_s..(t + 1) * n_s].copy_from_slice(&p[(t - 1) * n_s..t * n_s]);
    }
    out
}

/// 逐日 OLS 残差：`y ~ 1 + x0..x9`，样本 = y 与所有 x 都非 NaN 的股票，样本数 > 10 才算。
///
/// 优先走 11×11 正规方程 Cholesky（条件数良好时与 SVD 机器精度一致且快 ~15 倍），
/// 若遇不满秩或病态矩阵自动回退原 SVD 伪逆，确保任意极端交易日结果一致。
fn ols_residual(y: &[f64], xs: &[Vec<f64>], n_t: usize, n_s: usize) -> Vec<f64> {
    let k = xs.len() + 1; // const + 风格因子
    let mut out = vec![NAN; n_t * n_s];
    let mut xbuf: Vec<f64> = Vec::with_capacity(n_s * k);
    let mut ybuf: Vec<f64> = Vec::with_capacity(n_s);
    let mut idx: Vec<usize> = Vec::with_capacity(n_s);
    let mut gram = vec![0.0f64; k * k];
    let mut rhs = vec![0.0f64; k];
    for t in 0..n_t {
        idx.clear();
        ybuf.clear();
        xbuf.clear();
        for j in 0..n_s {
            let v = y[t * n_s + j];
            if v.is_nan() {
                continue;
            }
            let mut ok = true;
            for x in xs.iter() {
                if x[t * n_s + j].is_nan() {
                    ok = false;
                    break;
                }
            }
            if !ok {
                continue;
            }
            idx.push(j);
            ybuf.push(v);
            xbuf.push(1.0);
            for x in xs.iter() {
                xbuf.push(x[t * n_s + j]);
            }
        }
        let n = ybuf.len();
        if n <= 10 {
            continue;
        }
        gram.fill(0.0);
        rhs.fill(0.0);
        for i in 0..n {
            let row = &xbuf[i * k..(i + 1) * k];
            let yi = ybuf[i];
            for a in 0..k {
                let ra = row[a];
                rhs[a] += ra * yi;
                for b in a..k {
                    gram[a * k + b] += ra * row[b];
                }
            }
        }
        for a in 0..k {
            for b in 0..a {
                gram[a * k + b] = gram[b * k + a];
            }
        }
        let gmat = DMatrix::from_row_slice(k, k, &gram);
        let rvec = DVector::from_row_slice(&rhs);
        let beta_opt = if let Some(chol) = gmat.cholesky() {
            let b = chol.solve(&rvec);
            // 校验残差正交性防极端病态，若法方程残差偏大则回退 SVD
            let g_err = (&DMatrix::from_row_slice(k, k, &gram) * &b - &rvec).norm();
            let r_norm = rvec.norm().max(1.0);
            if b.iter().all(|v| v.is_finite()) && g_err <= 1e-7 * r_norm {
                Some(b)
            } else {
                None
            }
        } else {
            None
        };
        let beta = match beta_opt {
            Some(b) => b,
            None => {
                let xmat = DMatrix::from_row_slice(n, k, &xbuf);
                let yvec = DVector::from_row_slice(&ybuf);
                let svd = xmat.svd(true, true);
                match svd.solve(&yvec, 1e-15) {
                    Ok(b) => b,
                    Err(_) => continue,
                }
            }
        };
        let b_slice = beta.as_slice();
        for (m, &j) in idx.iter().enumerate() {
            let row = &xbuf[m * k..(m + 1) * k];
            let mut pred = 0.0f64;
            for c in 0..k {
                pred += row[c] * b_slice[c];
            }
            out[t * n_s + j] = ybuf[m] - pred;
        }
    }
    out
}

/// pandas `rank(axis=0, method="average", pct=True)` 的单日原地实现。
fn rank_pct_average_in_place(col: &mut [f64]) {
    let vals = col.to_vec();
    let mut idx: Vec<u32> = Vec::with_capacity(vals.len());
    for (i, v) in vals.iter().enumerate() {
        if !v.is_nan() {
            idx.push(i as u32);
        }
    }
    let n = idx.len();
    if n == 0 {
        return;
    }
    idx.sort_by(|&a, &b| {
        vals[a as usize]
            .partial_cmp(&vals[b as usize])
            .expect("NaN 已过滤")
    });
    let denom = n as f64;
    let mut k = 0usize;
    while k < n {
        let mut j = k;
        let v = vals[idx[k] as usize];
        while j + 1 < n && vals[idx[j + 1] as usize] == v {
            j += 1;
        }
        let pct = ((k + 1) as f64 + (j + 1) as f64) / 2.0 / denom;
        for m in k..=j {
            let i = idx[m] as usize;
            col[i] = pct;
        }
        k = j + 1;
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
struct CachedFulltestDataset {
    n_t: usize,
    n_s: usize,
    dates: Vec<i32>,
    stocks: Vec<String>,
    ret: Vec<f64>,
    ret_sum: Vec<f64>,
    restrict: Vec<f64>,
    ret_resi_raw: Vec<f64>,
    ret_resi_raw_1d: Vec<f64>,
    ret_resi: Vec<f64>,
    index_ret: Vec<f64>,
}

fn fulltest_dataset_cache_key(
    vars: &Path,
    szba: &Path,
    indexd: &Path,
    index_name: &str,
    start_date: i32,
    end_date: i32,
    gap: usize,
    style_fields: &[&str],
) -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    #[cfg(unix)]
    use std::os::unix::fs::MetadataExt;

    let mut h = DefaultHasher::new();
    "fulltest_ds_v1".hash(&mut h);
    vars.to_string_lossy().hash(&mut h);
    szba.to_string_lossy().hash(&mut h);
    indexd.to_string_lossy().hash(&mut h);
    index_name.hash(&mut h);
    start_date.hash(&mut h);
    end_date.hash(&mut h);
    gap.hash(&mut h);

    let mut tracked_paths = vec![
        vars.join("calendar_map.csv"),
        vars.join("symbol_map.csv"),
        vars.join("S_DQ_RET_VWAP_0930_1000.h5"),
        vars.join("S_RESTRICT.h5"),
        indexd.join("calendar_map.csv"),
        indexd.join("symbol_map.csv"),
        indexd.join("S_DQ_OPEN.h5"),
        szba.join("calendar_map.csv"),
        szba.join("symbol_map.csv"),
    ];
    for f in style_fields {
        tracked_paths.push(szba.join(format!("{f}.h5")));
    }
    for p in tracked_paths {
        if let Ok(meta) = std::fs::metadata(&p) {
            meta.len().hash(&mut h);
            #[cfg(unix)]
            {
                meta.mtime().hash(&mut h);
                meta.mtime_nsec().hash(&mut h);
            }
        } else {
            0u64.hash(&mut h);
        }
    }
    format!("{:016x}", h.finish())
}

fn fulltest_dataset_cache_dir() -> std::path::PathBuf {
    if let Ok(custom) = std::env::var("RUST_PYFUNC_FULLTEST_CACHE_DIR") {
        if !custom.trim().is_empty() {
            return std::path::PathBuf::from(custom);
        }
    }
    let shared = Path::new("/home/chenzongwei/pythoncode/_tail_v2_shared/fulltest_dataset_cache");
    if std::fs::create_dir_all(shared).is_ok() {
        return shared.to_path_buf();
    }
    let fallback = Path::new("/tmp/rp_fulltest_dataset_cache");
    let _ = std::fs::create_dir_all(fallback);
    fallback.to_path_buf()
}

fn build_fulltest_dataset_uncached(
    vars: &Path,
    szba: &Path,
    indexd: &Path,
    index_name: &str,
    start_date: i32,
    end_date: i32,
    gap: usize,
    style_fields: &[&str],
) -> Result<CachedFulltestDataset, String> {
    let dates = read_dates(vars, start_date, end_date)?;
    let stocks: Vec<String> = read_stocks(vars)?
        .into_iter()
        .map(|s| add_symbol_suffix(&s))
        .collect();
    let n_t = dates.len();
    let n_s = stocks.len();

    let mut ret = read_h5_aligned(vars, "S_DQ_RET_VWAP_0930_1000", &dates, &stocks)?;
    for v in ret.iter_mut() {
        if v.is_infinite() {
            *v = NAN;
        }
    }
    let restrict = read_h5_aligned(vars, "S_RESTRICT", &dates, &stocks)?;
    let index_ret = read_index_forward_ret(
        indexd,
        "S_DQ_OPEN",
        &format!("{index_name}.SH"),
        &dates,
    )?;

    let ret_sum = rolling_sum(&ret, n_t, n_s, gap);
    let mut w5 = ret_sum.clone();
    winsorize_cols(&mut w5, n_t, n_s);
    let winsor_5d = shift_cols_left(&w5, n_t, n_s, gap - 1);

    let mut w1 = ret.clone();
    winsorize_cols(&mut w1, n_t, n_s);
    let winsor_1d = w1;

    let mut styles: Vec<Vec<f64>> = Vec::with_capacity(style_fields.len());
    for f in style_fields.iter() {
        let raw = read_h5_aligned(szba, f, &dates, &stocks)?;
        styles.push(shift_rows_down(&raw, n_t, n_s));
    }

    let ret_resi_raw = ols_residual(&winsor_5d, &styles, n_t, n_s);
    let ret_resi_raw_1d = ols_residual(&winsor_1d, &styles, n_t, n_s);
    let mut ret_resi = ret_resi_raw.clone();
    for t in 0..n_t {
        rank_pct_average_in_place(&mut ret_resi[t * n_s..(t + 1) * n_s]);
    }

    Ok(CachedFulltestDataset {
        n_t,
        n_s,
        dates,
        stocks,
        ret,
        ret_sum,
        restrict,
        ret_resi_raw,
        ret_resi_raw_1d,
        ret_resi,
        index_ret,
    })
}

#[pyfunction]
#[pyo3(signature = (vars_dir, szba_dir, index_dir, index_name, start_date, end_date, gap))]
pub fn tail_v4_fulltest_dataset_rust(
    py: Python<'_>,
    vars_dir: &str,
    szba_dir: &str,
    index_dir: &str,
    index_name: &str,
    start_date: i32,
    end_date: i32,
    gap: usize,
) -> PyResult<PyObject> {
    const STYLE_FIELDS: [&str; 10] = [
        "size",
        "beta",
        "momentum",
        "residual_volatility",
        "book_to_price",
        "liquidity",
        "non_linear_size",
        "leverage",
        "earnings_yield",
        "growth",
    ];
    let vars_s = vars_dir.to_string();
    let szba_s = szba_dir.to_string();
    let index_s = index_dir.to_string();
    let idx_name_s = index_name.to_string();

    let ds_cached = py
        .allow_threads(move || -> Result<CachedFulltestDataset, String> {
            let vars = Path::new(&vars_s);
            let szba = Path::new(&szba_s);
            let indexd = Path::new(&index_s);
            let disable_cache = std::env::var("RUST_PYFUNC_DISABLE_FULLTEST_DS_CACHE")
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false);
            if disable_cache {
                return build_fulltest_dataset_uncached(
                    vars,
                    szba,
                    indexd,
                    &idx_name_s,
                    start_date,
                    end_date,
                    gap,
                    &STYLE_FIELDS,
                );
            }

            let key = fulltest_dataset_cache_key(
                vars,
                szba,
                indexd,
                &idx_name_s,
                start_date,
                end_date,
                gap,
                &STYLE_FIELDS,
            );
            let cache_dir = fulltest_dataset_cache_dir();
            let bin_path = cache_dir.join(format!("ft_ds_{key}.bin"));
            let lock_path = cache_dir.join(format!("ft_ds_{key}.lock"));

            // 快速无锁命中检查
            if let Ok(bytes) = std::fs::read(&bin_path) {
                if let Ok(decoded) = zstd::decode_all(&bytes[..]) {
                    if let Ok(ds) = bincode::deserialize::<CachedFulltestDataset>(&decoded) {
                        return Ok(ds);
                    }
                }
            }

            // 排他锁保护：防止 32~64 个 fulltest worker 同时未命中并发读 13 个 H5
            let lock_file = std::fs::OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(false)
                .open(&lock_path)
                .ok();
            #[cfg(unix)]
            if let Some(ref lf) = lock_file {
                use std::os::unix::io::AsRawFd;
                unsafe {
                    libc::flock(lf.as_raw_fd(), libc::LOCK_EX);
                }
            }

            // 拿到锁后二次检查缓存
            if let Ok(bytes) = std::fs::read(&bin_path) {
                if let Ok(decoded) = zstd::decode_all(&bytes[..]) {
                    if let Ok(ds) = bincode::deserialize::<CachedFulltestDataset>(&decoded) {
                        return Ok(ds);
                    }
                }
            }

            let ds = build_fulltest_dataset_uncached(
                vars,
                szba,
                indexd,
                &idx_name_s,
                start_date,
                end_date,
                gap,
                &STYLE_FIELDS,
            )?;

            if let Ok(raw_bytes) = bincode::serialize(&ds) {
                if let Ok(compressed) = zstd::encode_all(&raw_bytes[..], 1) {
                    let tmp_path =
                        cache_dir.join(format!("ft_ds_{key}.{}.tmp", std::process::id()));
                    if std::fs::write(&tmp_path, &compressed).is_ok() {
                        let _ = std::fs::rename(&tmp_path, &bin_path);
                    }
                }
            }
            Ok(ds)
        })
        .map_err(PyValueError::new_err)?;

    let n_t = ds_cached.n_t;
    let n_s = ds_cached.n_s;
    let dict = PyDict::new(py);
    let mk = |data: Vec<f64>| {
        PyArray::from_owned_array(py, Array2::from_shape_vec((n_t, n_s), data).expect("形状"))
    };
    dict.set_item("ret", mk(ds_cached.ret))?;
    dict.set_item("ret_sum", mk(ds_cached.ret_sum))?;
    dict.set_item("restrict", mk(ds_cached.restrict))?;
    dict.set_item("ret_resi_raw", mk(ds_cached.ret_resi_raw))?;
    dict.set_item("ret_resi_raw_1d", mk(ds_cached.ret_resi_raw_1d))?;
    dict.set_item("ret_resi", mk(ds_cached.ret_resi))?;
    dict.set_item(
        "index_ret",
        PyArray::from_owned_array(py, ndarray::Array1::from(ds_cached.index_ret)),
    )?;
    dict.set_item(
        "dates",
        PyList::new(py, ds_cached.dates.iter().map(|&d| format_date(d))),
    )?;
    dict.set_item("stocks", PyList::new(py, ds_cached.stocks.iter()))?;
    Ok(dict.into())
}


/// 读一个新格式因子 parquet（`date` 列 + 每只股票一列），对齐到 (stocks, dates)。
///
/// 返回 (n_stocks, n_t) 的 float64 矩阵；缺失位置为 NaN，非有限值也置 NaN
/// （与 `_read_factor_to_template` 的 inf→NaN 一致）。
#[pyfunction]
#[pyo3(signature = (factor_path, dates, stocks))]
pub fn tail_v4_fulltest_factor_rust(
    py: Python<'_>,
    factor_path: &str,
    dates: Vec<String>,
    stocks: Vec<String>,
) -> PyResult<PyObject> {
    let n_t = dates.len();
    let n_s = stocks.len();
    let date_pos: HashMap<i32, usize> = dates
        .iter()
        .enumerate()
        .map(|(i, d)| (parse_date_str(d), i))
        .collect();
    let stock_pos: HashMap<&str, usize> = stocks
        .iter()
        .enumerate()
        .map(|(i, s)| (s.as_str(), i))
        .collect();

    let mut out = vec![NAN; n_t * n_s];
    let file = File::open(factor_path)
        .map_err(|e| PyValueError::new_err(format!("打开因子文件失败 {factor_path}: {e}")))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|e| PyValueError::new_err(format!("解析 parquet 失败 {factor_path}: {e}")))?;
    let reader = builder
        .build()
        .map_err(|e| PyValueError::new_err(format!("构建 parquet reader 失败 {factor_path}: {e}")))?;

    let epoch = NaiveDate::from_ymd_opt(1970, 1, 1).expect("1970-01-01 合法");
    for batch in reader {
        let batch =
            batch.map_err(|e| PyValueError::new_err(format!("读取 parquet 批次失败: {e}")))?;
        let schema = batch.schema();
        let date_idx = schema
            .index_of("date")
            .map_err(|_| PyValueError::new_err(format!("{factor_path} 缺少 date 列")))?;
        let date_col = batch.column(date_idx);
        // pandas 写出来的 date 列可能是 date32[day]、也可能被 arrow-rs 解析成别的物理类型，
        // 这里按实际类型取值，统一折成 yyyymmdd 整数。
        let mut ymd_of_row: Vec<Option<i32>> = Vec::with_capacity(batch.num_rows());
        if let Some(a) = date_col.as_any().downcast_ref::<Date32Array>() {
            for i in 0..a.len() {
                ymd_of_row.push(Some(
                    (epoch + Duration::days(a.value(i) as i64))
                        .format("%Y%m%d")
                        .to_string()
                        .parse()
                        .expect("yyyyMMdd"),
                ));
            }
        } else if let Some(a) = date_col.as_any().downcast_ref::<Date64Array>() {
            for i in 0..a.len() {
                let ms = a.value(i);
                ymd_of_row.push(
                    chrono::DateTime::from_timestamp_millis(ms)
                        .map(|dt| dt.format("%Y%m%d").to_string().parse().expect("yyyyMMdd")),
                );
            }
        } else if let Some(a) = date_col.as_any().downcast_ref::<StringArray>() {
            for i in 0..a.len() {
                ymd_of_row.push(Some(parse_date_str(a.value(i))));
            }
        } else if let Some(a) = date_col.as_any().downcast_ref::<Int32Array>() {
            for i in 0..a.len() {
                ymd_of_row.push(Some(a.value(i)));
            }
        } else {
            return Err(PyValueError::new_err(format!(
                "date 列类型不支持: {}",
                date_col.data_type()
            )));
        }

        for row in 0..ymd_of_row.len() {
            let Some(ymd) = ymd_of_row[row] else { continue };
            let Some(&t) = date_pos.get(&ymd) else { continue };
            for (ci, field) in schema.fields().iter().enumerate() {
                if ci == date_idx {
                    continue;
                }
                let Some(&j) = stock_pos.get(field.name().as_str()) else {
                    continue;
                };
                let col = batch.column(ci);
                // arrow 的 null 位置 value() 会返回 0.0，必须先判空，否则缺失值会变成 0。
                let v = if let Some(a) = col.as_any().downcast_ref::<Float64Array>() {
                    if a.is_null(row) {
                        NAN
                    } else {
                        a.value(row)
                    }
                } else if let Some(a) = col.as_any().downcast_ref::<Float32Array>() {
                    if a.is_null(row) {
                        NAN
                    } else {
                        a.value(row) as f64
                    }
                } else {
                    continue;
                };
                out[t * n_s + j] = if v.is_finite() { v } else { NAN };
            }
        }
    }

    let dict = PyDict::new(py);
    dict.set_item(
        "values",
        PyArray::from_owned_array(
            py,
            Array2::from_shape_vec((n_t, n_s), out).expect("形状与元素数一致"),
        ),
    )?;
    Ok(dict.into())
}
