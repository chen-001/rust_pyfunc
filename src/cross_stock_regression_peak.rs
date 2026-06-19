//! 跨股票回归因子 — 峰谷窗口版本（主卖/主买密度峰值确定个性化训练和预测窗口）
//!
//! 每只股票找到两个关键时刻：
//! - x = 主卖密度最大时刻 → 开盘至 x 为自回归训练窗口
//! - y = 主买密度最大时刻 → y 至收盘为迁移预测窗口
//!
//! 复用二次型展开加速（每对 (i,j) 仅 ~45 次浮点运算）

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use super::cross_stock_regression::{extract_38_stats, ols_with_intercept_safe};

// ============================================================
// 复用 fast 版本的预计算结构体（拷贝，避免跨模块依赖）
// ============================================================

#[derive(Clone)]
struct StockPrecomputed {
    sum_y: f64,
    sum_y2: f64,
    ss_tot: f64,
    q_upper: Vec<f64>,
    sx: Vec<f64>,
    sxy: Vec<f64>,
}

impl StockPrecomputed {
    fn new(n_lags: usize) -> Self {
        StockPrecomputed {
            sum_y: 0.0,
            sum_y2: 0.0,
            ss_tot: 0.0,
            q_upper: vec![0.0; n_lags * (n_lags + 1) / 2],
            sx: vec![0.0; n_lags],
            sxy: vec![0.0; n_lags],
        }
    }

    fn compute_from(&mut self, x: &ArrayView2<f64>, y: &ArrayView1<f64>) {
        let t = x.nrows();
        let n_lags = self.sx.len();
        if t < 3 {
            return;
        }
        self.sum_y = y.iter().filter(|v| v.is_finite()).sum();
        self.sum_y2 = y.iter().filter(|v| v.is_finite()).map(|v| v * v).sum();
        let y_mean = self.sum_y / t as f64;
        self.ss_tot = y.iter().filter(|v| v.is_finite()).map(|&v| (v - y_mean).powi(2)).sum();
        for a in 0..n_lags {
            self.sx[a] = x.column(a).iter().filter(|v| v.is_finite()).sum();
            self.sxy[a] = x.column(a).iter().zip(y.iter())
                .filter(|(xv, yv)| xv.is_finite() && yv.is_finite())
                .map(|(xv, yv)| xv * yv).sum();
            let idx_aa = a * (a + 1) / 2 + a;
            self.q_upper[idx_aa] = x.column(a).iter().filter(|v| v.is_finite()).map(|v| v * v).sum();
            for b in a + 1..n_lags {
                let idx_ab = b * (b + 1) / 2 + a;
                self.q_upper[idx_ab] = x.column(a).iter().zip(x.column(b).iter())
                    .filter(|(xa, xb)| xa.is_finite() && xb.is_finite())
                    .map(|(xa, xb)| xa * xb).sum();
            }
        }
    }

    #[inline]
    fn compute_ss_res(&self, beta: &ArrayView1<f64>, t_valid: f64) -> f64 {
        let b0 = if beta[0].is_finite() { beta[0] } else { return f64::NAN };
        let n_lags = self.sx.len();
        let t1 = t_valid * b0 * b0 - 2.0 * b0 * self.sum_y + self.sum_y2;
        let mut t2 = 0.0;
        for a in 0..n_lags {
            let ba = beta[a + 1];
            if !ba.is_finite() { return f64::NAN; }
            let idx_aa = a * (a + 1) / 2 + a;
            t2 += self.q_upper[idx_aa] * ba * ba;
            for b in a + 1..n_lags {
                let bb = beta[b + 1];
                if !bb.is_finite() { continue; }
                let idx_ab = b * (b + 1) / 2 + a;
                t2 += 2.0 * self.q_upper[idx_ab] * ba * bb;
            }
        }
        let mut t3 = 0.0;
        for a in 0..n_lags {
            let ba = beta[a + 1];
            if !ba.is_finite() { continue; }
            t3 += ba * (b0 * self.sx[a] - self.sxy[a]);
        }
        t3 *= 2.0;
        t1 + t2 + t3
    }
}

// ============================================================
// 辅助：从一列数据 + 窗口边界构建滞后设计矩阵
// ============================================================

fn build_lagged_window(
    data_col: &ArrayView1<f64>,
    start: usize,
    end: usize,
    n_lags: usize,
) -> (Array2<f64>, Array1<f64>) {
    let len = end - start + 1;
    let t_valid = len.saturating_sub(n_lags);
    let mut x = Array2::<f64>::zeros((t_valid, n_lags));
    let mut y = Array1::<f64>::zeros(t_valid);
    for row in 0..t_valid {
        let actual_row = start + row + n_lags;
        y[row] = data_col[actual_row];
        for lag in 0..n_lags {
            x[[row, lag]] = data_col[actual_row - 1 - lag];
        }
    }
    (x, y)
}

// ============================================================
// 主力函数：峰谷窗口 autoreg
// ============================================================

#[pyfunction]
#[pyo3(signature = (data_fit, data_pred, fit_ends, pred_starts, n_lags))]
pub fn cross_stock_autoreg_peak_38_fast(
    py: Python,
    data_fit: PyReadonlyArray2<f64>,
    data_pred: PyReadonlyArray2<f64>,
    fit_ends: PyReadonlyArray1<i64>,
    pred_starts: PyReadonlyArray1<i64>,
    n_lags: usize,
) -> PyResult<Py<PyArray2<f64>>> {
    let df = data_fit.as_array();
    let dp = data_pred.as_array();
    let fe = fit_ends.as_array();
    let ps = pred_starts.as_array();
    let n_stocks = df.ncols();
    let t = df.nrows();

    // ---- OLS 拟合（每只股票用自己的训练窗口） ----
    let mut betas = Array2::<f64>::zeros((n_lags + 1, n_stocks));
    let mut r2_pri = Array1::<f64>::zeros(n_stocks);
    for j in 0..n_stocks {
        let end = fe[j].max(n_lags as i64 + 1).min(t as i64 - 1) as usize;
        if end <= n_lags {
            for k in 0..n_lags + 1 { betas[[k, j]] = f64::NAN; }
            r2_pri[j] = f64::NAN;
            continue;
        }
        let col = df.column(j);
        let (x_j, y_j) = build_lagged_window(&col, 0, end, n_lags);
        let (coef, r2) = ols_with_intercept_safe(&x_j.view(), &y_j.view());
        for k in 0..n_lags + 1 { betas[[k, j]] = coef[k]; }
        r2_pri[j] = r2;
    }

    // ---- 预计算每只股票的预测窗口二次型 ----
    let mut precomputed: Vec<StockPrecomputed> = Vec::with_capacity(n_stocks);
    let mut valid_lengths = vec![0usize; n_stocks];
    for i in 0..n_stocks {
        let start = ps[i].max(0).min(t as i64 - n_lags as i64 - 1) as usize;
        let end = t - 1;
        let col = dp.column(i);
        let (x_i, y_i) = build_lagged_window(&col, start, end, n_lags);
        let mut sp = StockPrecomputed::new(n_lags);
        sp.compute_from(&x_i.view(), &y_i.view());
        valid_lengths[i] = x_i.nrows();
        precomputed.push(sp);
    }

    // ---- 快速 R² 计算 ----
    let mut r2s = Array2::<f64>::from_elem((n_stocks, n_stocks), f64::NAN);
    for j in 0..n_stocks {
        let beta_j = betas.column(j);
        for i in 0..n_stocks {
            let pc = &precomputed[i];
            if pc.ss_tot <= 0.0 || valid_lengths[i] < 3 {
                continue;
            }
            let ss_res = pc.compute_ss_res(&beta_j, valid_lengths[i] as f64);
            if ss_res.is_finite() {
                r2s[[i, j]] = 1.0 - ss_res / pc.ss_tot;
            }
        }
    }

    let stats = extract_38_stats(&r2s.view(), &r2_pri.view());
    Ok(stats.into_pyarray(py).to_owned())
}

// ============================================================
// 主力函数：峰谷窗口 crossvar
// ============================================================

#[pyfunction]
#[pyo3(signature = (y_fit, x_fit, y_pred, x_pred, fit_ends, pred_starts, n_lags))]
pub fn cross_stock_crossvar_peak_38_fast(
    py: Python,
    y_fit: PyReadonlyArray2<f64>,
    x_fit: PyReadonlyArray2<f64>,
    y_pred: PyReadonlyArray2<f64>,
    x_pred: PyReadonlyArray2<f64>,
    fit_ends: PyReadonlyArray1<i64>,
    pred_starts: PyReadonlyArray1<i64>,
    n_lags: usize,
) -> PyResult<Py<PyArray2<f64>>> {
    let yf = y_fit.as_array();
    let xf = x_fit.as_array();
    let yp = y_pred.as_array();
    let xp = x_pred.as_array();
    let fe = fit_ends.as_array();
    let ps = pred_starts.as_array();
    let n_stocks = yf.ncols();
    let t = yf.nrows();
    let total_lags = 2 * n_lags;

    // ---- OLS 拟合 ----
    let mut betas = Array2::<f64>::zeros((total_lags + 1, n_stocks));
    let mut r2_pri = Array1::<f64>::zeros(n_stocks);
    for j in 0..n_stocks {
        let end = fe[j].max((n_lags + 1) as i64).min(t as i64 - 1) as usize;
        if end <= n_lags {
            for k in 0..total_lags + 1 { betas[[k, j]] = f64::NAN; }
            r2_pri[j] = f64::NAN;
            continue;
        }
        let y_col = yf.column(j);
        let x_col = xf.column(j);
        let len = end + 1;
        let t_valid = len.saturating_sub(n_lags);
        let mut x_j = Array2::<f64>::zeros((t_valid, total_lags));
        let mut y_j = Array1::<f64>::zeros(t_valid);
        for row in 0..t_valid {
            let r = row + n_lags;
            y_j[row] = y_col[r];
            for lag in 0..n_lags {
                x_j[[row, lag]] = y_col[r - 1 - lag];
                x_j[[row, lag + n_lags]] = x_col[r - 1 - lag];
            }
        }
        let (coef, r2) = ols_with_intercept_safe(&x_j.view(), &y_j.view());
        for k in 0..total_lags + 1 { betas[[k, j]] = coef[k]; }
        r2_pri[j] = r2;
    }

    // ---- 预计算预测窗口 ----
    let mut precomputed: Vec<StockPrecomputed> = Vec::with_capacity(n_stocks);
    let mut valid_lengths = vec![0usize; n_stocks];
    for i in 0..n_stocks {
        let start = ps[i].max(0).min(t as i64 - n_lags as i64 - 1) as usize;
        let end = t - 1;
        let y_col = yp.column(i);
        let x_col = xp.column(i);
        let len = end - start + 1;
        let t_valid = len.saturating_sub(n_lags);
        let mut x_i = Array2::<f64>::zeros((t_valid, total_lags));
        let mut y_i = Array1::<f64>::zeros(t_valid);
        for row in 0..t_valid {
            let r = start + row + n_lags;
            y_i[row] = y_col[r];
            for lag in 0..n_lags {
                x_i[[row, lag]] = y_col[r - 1 - lag];
                x_i[[row, lag + n_lags]] = x_col[r - 1 - lag];
            }
        }
        let mut sp = StockPrecomputed::new(total_lags);
        sp.compute_from(&x_i.view(), &y_i.view());
        valid_lengths[i] = t_valid;
        precomputed.push(sp);
    }

    // ---- 快速 R² ----
    let mut r2s = Array2::<f64>::from_elem((n_stocks, n_stocks), f64::NAN);
    for j in 0..n_stocks {
        let beta_j = betas.column(j);
        for i in 0..n_stocks {
            let pc = &precomputed[i];
            if pc.ss_tot <= 0.0 || valid_lengths[i] < 3 {
                continue;
            }
            let ss_res = pc.compute_ss_res(&beta_j, valid_lengths[i] as f64);
            if ss_res.is_finite() {
                r2s[[i, j]] = 1.0 - ss_res / pc.ss_tot;
            }
        }
    }

    let stats = extract_38_stats(&r2s.view(), &r2_pri.view());
    Ok(stats.into_pyarray(py).to_owned())
}

// ============================================================
// 峰谷查找：滚动窗口 trimmed mean 峰值（Rust SIMD 加速）
// ============================================================

/// 对每只股票计算滚动窗口 trimmed mean 的峰值位置。
///
/// - sell_vol: (T, N) 主动卖出原始成交量
/// - buy_vol:  (T, N) 主动买入原始成交量
/// - window:   窗口长度（默认60）
/// - min_len:  最小有效窗口长度（默认10）
///
/// 返回: (fit_ends_i64, pred_starts_i64) 两个 i64 数组，长度均为 N
#[pyfunction]
#[pyo3(signature = (sell_vol, buy_vol, window=60, min_len=10))]
pub fn find_density_peaks(
    py: Python,
    sell_vol: PyReadonlyArray2<f64>,
    buy_vol: PyReadonlyArray2<f64>,
    window: usize,
    min_len: usize,
) -> PyResult<(Py<PyArray1<i64>>, Py<PyArray1<i64>>)> {
    let sv = sell_vol.as_array();
    let bv = buy_vol.as_array();
    let t = sv.nrows();
    let n_stocks = sv.ncols();

    let mut fit_ends = Array1::<i64>::from_elem(n_stocks, (window - 1) as i64);
    let mut pred_starts = Array1::<i64>::from_elem(n_stocks, (t - window) as i64);

    // 复用缓冲区，避免每次分配
    let mut buf = vec![0.0f64; window];

    for i in 0..n_stocks {
        // ---- 主卖密度：窗口右端对齐 m ----
        let mut sell_best = (window - 1) as i64;
        let mut sell_max = -1.0f64;

        for m in min_len..t {
            let start = if m > window { m - window } else { 0 };
            let wlen = m - start;
            let k = (wlen / 2).max(1);

            let mut pos_count = 0usize;
            let mut valid_len = 0usize;
            for j in start..m {
                let v = sv[[j, i]];
                buf[valid_len] = v;
                valid_len += 1;
                if v > 0.0 { pos_count += 1; }
            }
            if pos_count < min_len { continue; }
            if valid_len == 0 { continue; }

            let slice = &mut buf[..valid_len];
            slice.sort_unstable_by(|a, b| b.partial_cmp(a).unwrap());

            let mean: f64 = slice[..k].iter().sum::<f64>() / k as f64;
            if mean > sell_max {
                sell_max = mean;
                sell_best = m as i64;
            }
        }
        fit_ends[i] = sell_best;

        // ---- 主买密度：窗口左端对齐 m ----
        let mut buy_best = (t - window) as i64;
        let mut buy_max = -1.0f64;

        let max_m = t.saturating_sub(min_len);
        for m in 0..max_m {
            let end = if m + window < t { m + window } else { t };
            let wlen = end - m;
            let k = (wlen / 2).max(1);

            let mut pos_count = 0usize;
            let mut valid_len = 0usize;
            for j in m..end {
                let v = bv[[j, i]];
                buf[valid_len] = v;
                valid_len += 1;
                if v > 0.0 { pos_count += 1; }
            }
            if pos_count < min_len { continue; }
            if valid_len == 0 { continue; }
            let slice = &mut buf[..valid_len];
            slice.sort_unstable_by(|a, b| b.partial_cmp(a).unwrap());

            let mean: f64 = slice[..k].iter().sum::<f64>() / k as f64;
            if mean > buy_max {
                buy_max = mean;
                buy_best = m as i64;
            }
        }
        pred_starts[i] = buy_best;
    }

    Ok((fit_ends.into_pyarray(py).to_owned(), pred_starts.into_pyarray(py).to_owned()))
}
