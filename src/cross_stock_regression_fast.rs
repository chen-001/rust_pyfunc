//! 跨股票回归因子计算 — 快速版本（使用二次型展开替代逐时间点比较）
//!
//! 关键优化：R² = 1 - SS_res / SS_tot
//!
//! SS_res = Σ_t (X_i[t]·β_j - y_i[t])²
//!        = T·b0² - 2·b0·Σy + Σy²        (t1)
//!        + βᵀ·Q_i·β                       (t2)
//!        + 2·βᵀ·(b0·Sx - Sxy)             (t3)
//!
//! 其中 Q_i = X_iᵀ·X_i, Sx_i = Σ_t X_i[t], Sxy_i = Σ_t y_i[t]·X_i[t]
//!
//! 预计算后，每对 (i,j) 仅需 ~45 浮点操作（vs 原来 ~700）

use ndarray::{s, Array1, Array2, ArrayView1, ArrayView2};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

// 复用 cross_stock_regression.rs 中的辅助函数
use super::cross_stock_regression::{
    build_lagged_design, build_stacked_design, extract_38_stats, ols_with_intercept_safe,
};

// ============================================================
// 预计算结构体：每只股票的二次型数据
// ============================================================

#[derive(Clone)]
struct StockPrecomputed {
    sum_y: f64,
    sum_y2: f64,
    ss_tot: f64,
    // Q_i: 对称矩阵的上三角，行优先，长度 = n_lags * (n_lags+1) / 2
    q_upper: Vec<f64>,
    sx: Vec<f64>,   // n_lags
    sxy: Vec<f64>,  // n_lags
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

    /// 从 (T, n_lags) 的设计矩阵 X 和 (T,) 的 y 预计算
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
            self.sxy[a] = x
                .column(a)
                .iter()
                .zip(y.iter())
                .filter(|(xv, yv)| xv.is_finite() && yv.is_finite())
                .map(|(xv, yv)| xv * yv)
                .sum();
            // Q_i 对角线
            let mut idx = a * (a + 1) / 2 + a;
            self.q_upper[idx] = x
                .column(a)
                .iter()
                .filter(|v| v.is_finite())
                .map(|v| v * v)
                .sum();
            // Q_i 上三角非对角线
            for b in a + 1..n_lags {
                let idx = b * (b + 1) / 2 + a;
                self.q_upper[idx] = x
                    .column(a)
                    .iter()
                    .zip(x.column(b).iter())
                    .filter(|(xa, xb)| xa.is_finite() && xb.is_finite())
                    .map(|(xa, xb)| xa * xb)
                    .sum();
            }
        }
    }

    /// 计算 SS_res = Σ (X·β - y)²，使用预计算数据
    /// β: [intercept, β[1], ..., β[n_lags]]
    #[inline]
    fn compute_ss_res(&self, beta: &ArrayView1<f64>, t_valid: f64) -> f64 {
        let b0 = if beta[0].is_finite() { beta[0] } else { return f64::NAN };
        let n_lags = self.sx.len();

        // t1 = T·b0² - 2·b0·Σy + Σy²
        let t1 = t_valid * b0 * b0 - 2.0 * b0 * self.sum_y + self.sum_y2;

        // t2 = βᵀ·Q·β
        let mut t2 = 0.0;
        for a in 0..n_lags {
            let ba = beta[a + 1];
            if !ba.is_finite() { return f64::NAN; }
            // 对角线项: Q_aa * ba²
            let idx_aa = a * (a + 1) / 2 + a;
            t2 += self.q_upper[idx_aa] * ba * ba;
            // 非对角线项: 2 * Q_ab * ba * bb
            for b in a + 1..n_lags {
                let bb = beta[b + 1];
                if !bb.is_finite() { continue; }
                let idx_ab = b * (b + 1) / 2 + a;
                t2 += 2.0 * self.q_upper[idx_ab] * ba * bb;
            }
        }

        // t3 = 2·βᵀ·(b0·Sx - Sxy)
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
// 快速 autoreg 函数
// ============================================================

#[pyfunction]
#[pyo3(signature = (data_prev, data_today, n_lags))]
pub fn cross_stock_autoreg_38_fast(
    py: Python,
    data_prev: PyReadonlyArray2<f64>,
    data_today: PyReadonlyArray2<f64>,
    n_lags: usize,
) -> PyResult<Py<PyArray2<f64>>> {
    let dp0 = data_prev.as_array();
    let dt0 = data_today.as_array();
    let n_stocks = dp0.ncols();
    // prev/today 经 clean 删除全 NaN 行后行数可能不同，取较小者统一截断，避免切片越界 panic
    let t = dp0.nrows().min(dt0.nrows());
    let dp = dp0.slice(s![..t, ..]);
    let dt = dt0.slice(s![..t, ..]);
    let t_valid = t - n_lags;

    if t_valid < 3 || n_stocks == 0 {
        return Ok(Array2::<f64>::from_elem((38, n_stocks), f64::NAN)
            .into_pyarray(py)
            .to_owned());
    }

    // ---- 第一步：OLS 拟合（不变）----
    let mut betas = Array2::<f64>::zeros((n_lags + 1, n_stocks));
    let mut r2_pri = Array1::<f64>::zeros(n_stocks);

    for j in 0..n_stocks {
        let col = dp.column(j);
        let x_j = build_lagged_design(&col, n_lags);
        let y_j = col.slice(s![n_lags..]);
        let (coef, r2) = ols_with_intercept_safe(&x_j.view(), &y_j);
        for k in 0..n_lags + 1 {
            betas[[k, j]] = coef[k];
        }
        r2_pri[j] = r2;
    }

    // ---- 第二步：构建今日设计矩阵 + 预计算 ----
    let (x_today_stacked, y_today_stacked) = build_stacked_design(&dt, n_lags);
    let t_f64 = t_valid as f64;

    // 提取每个股票的设计矩阵并预计算
    let mut precomputed: Vec<StockPrecomputed> = Vec::with_capacity(n_stocks);
    for i in 0..n_stocks {
        let offset = i * t_valid;
        let x_i = x_today_stacked.slice(s![offset..offset + t_valid, ..]);
        let y_i = y_today_stacked.slice(s![offset..offset + t_valid]);
        let mut sp = StockPrecomputed::new(n_lags);
        sp.compute_from(&x_i, &y_i);
        precomputed.push(sp);
    }

    // ---- 第三步：快速 R² 计算 ----
    let mut r2s = Array2::<f64>::from_elem((n_stocks, n_stocks), f64::NAN);

    for j in 0..n_stocks {
        let beta_j = betas.column(j);
        for i in 0..n_stocks {
            let pc = &precomputed[i];
            if pc.ss_tot <= 0.0 {
                continue;
            }
            let ss_res = pc.compute_ss_res(&beta_j, t_f64);
            if ss_res.is_finite() {
                r2s[[i, j]] = 1.0 - ss_res / pc.ss_tot;
            }
        }
    }

    // ---- 第四步：提取统计量（不变）----
    let stats = extract_38_stats(&r2s.view(), &r2_pri.view());
    Ok(stats.into_pyarray(py).to_owned())
}

// ============================================================
// 快速 crossvar 函数
// ============================================================

#[pyfunction]
#[pyo3(signature = (y_prev, x_prev, y_today, x_today, n_lags))]
pub fn cross_stock_crossvar_38_fast(
    py: Python,
    y_prev: PyReadonlyArray2<f64>,
    x_prev: PyReadonlyArray2<f64>,
    y_today: PyReadonlyArray2<f64>,
    x_today: PyReadonlyArray2<f64>,
    n_lags: usize,
) -> PyResult<Py<PyArray2<f64>>> {
    let yp0 = y_prev.as_array();
    let xp0 = x_prev.as_array();
    let yt0 = y_today.as_array();
    let xt0 = x_today.as_array();
    let n_stocks = yp0.ncols();
    // 四个矩阵行数可能不同(同上)，取最小行数统一截断
    let t = yp0.nrows().min(xp0.nrows()).min(yt0.nrows()).min(xt0.nrows());
    let yp = yp0.slice(s![..t, ..]);
    let xp = xp0.slice(s![..t, ..]);
    let yt = yt0.slice(s![..t, ..]);
    let xt = xt0.slice(s![..t, ..]);
    let t_valid = t - n_lags;
    let total_lags = 2 * n_lags;

    if t_valid < 3 || n_stocks == 0 {
        return Ok(Array2::<f64>::from_elem((38, n_stocks), f64::NAN)
            .into_pyarray(py)
            .to_owned());
    }

    // ---- OLS 拟合 ----
    let mut betas = Array2::<f64>::zeros((total_lags + 1, n_stocks));
    let mut r2_pri = Array1::<f64>::zeros(n_stocks);

    for j in 0..n_stocks {
        let y_col = yp.column(j);
        let x_col = xp.column(j);
        let mut x_j = Array2::<f64>::zeros((t_valid, total_lags));
        for lag in 0..n_lags {
            for row in 0..t_valid {
                x_j[[row, lag]] = y_col[row + n_lags - 1 - lag];
                x_j[[row, lag + n_lags]] = x_col[row + n_lags - 1 - lag];
            }
        }
        let y_j = y_col.slice(s![n_lags..]);
        let (coef, r2) = ols_with_intercept_safe(&x_j.view(), &y_j);
        for k in 0..total_lags + 1 {
            betas[[k, j]] = coef[k];
        }
        r2_pri[j] = r2;
    }

    // ---- 构建今日设计矩阵 + 预计算 ----
    // 先构建堆叠设计矩阵，再拆成每个股票
    let mut x_stacked = Array2::<f64>::zeros((n_stocks * t_valid, total_lags));
    let mut y_stacked = Array1::<f64>::zeros(n_stocks * t_valid);

    for s in 0..n_stocks {
        let offset = s * t_valid;
        let ycol = yt.column(s);
        let xcol = xt.column(s);
        for lag in 0..n_lags {
            for row in 0..t_valid {
                x_stacked[[offset + row, lag]] = ycol[row + n_lags - 1 - lag];
                x_stacked[[offset + row, lag + n_lags]] = xcol[row + n_lags - 1 - lag];
            }
        }
        for row in 0..t_valid {
            y_stacked[offset + row] = ycol[row + n_lags];
        }
    }

    let t_f64 = t_valid as f64;
    let mut precomputed: Vec<StockPrecomputed> = Vec::with_capacity(n_stocks);
    for i in 0..n_stocks {
        let offset = i * t_valid;
        let x_i = x_stacked.slice(s![offset..offset + t_valid, ..]);
        let y_i = y_stacked.slice(s![offset..offset + t_valid]);
        let mut sp = StockPrecomputed::new(total_lags);
        sp.compute_from(&x_i, &y_i);
        precomputed.push(sp);
    }

    // ---- 快速 R² 计算 ----
    let mut r2s = Array2::<f64>::from_elem((n_stocks, n_stocks), f64::NAN);

    for j in 0..n_stocks {
        let beta_j = betas.column(j);
        for i in 0..n_stocks {
            let pc = &precomputed[i];
            if pc.ss_tot <= 0.0 {
                continue;
            }
            let ss_res = pc.compute_ss_res(&beta_j, t_f64);
            if ss_res.is_finite() {
                r2s[[i, j]] = 1.0 - ss_res / pc.ss_tot;
            }
        }
    }

    let stats = extract_38_stats(&r2s.view(), &r2_pri.view());
    Ok(stats.into_pyarray(py).to_owned())
}


// ============================================================
// 加权版本：autoreg + crossvar（用成交量对数做 R² 矩阵统计量加权）
// ============================================================

use super::cross_stock_regression::extract_38_stats_weighted;

/// 跨股票自回归因子（加权版）
/// weights: (N,) 正权重向量，通常为 log(总成交量)
#[pyfunction]
#[pyo3(signature = (data_fit, data_pred, n_lags, weights))]
pub fn cross_stock_autoreg_38_weighted(
    py: Python,
    data_fit: PyReadonlyArray2<f64>,
    data_pred: PyReadonlyArray2<f64>,
    n_lags: usize,
    weights: PyReadonlyArray1<f64>,
) -> PyResult<Py<PyArray2<f64>>> {
    let df = data_fit.as_array();
    let dp = data_pred.as_array();
    let w = weights.as_array();
    let n_stocks = df.ncols();
    let t = df.nrows();
    let t_valid = t - n_lags;
    if t_valid < 3 || n_stocks == 0 {
        return Ok(Array2::<f64>::from_elem((38, n_stocks), f64::NAN).into_pyarray(py).to_owned());
    }

    // OLS 拟合
    let mut betas = Array2::<f64>::zeros((n_lags + 1, n_stocks));
    let mut r2_pri = Array1::<f64>::zeros(n_stocks);
    for j in 0..n_stocks {
        let col = df.column(j);
        let x_j = build_lagged_design(&col, n_lags);
        let y_j = col.slice(s![n_lags..]);
        let (coef, r2) = ols_with_intercept_safe(&x_j.view(), &y_j);
        for k in 0..n_lags + 1 { betas[[k, j]] = coef[k]; }
        r2_pri[j] = r2;
    }

    // 构建今日设计矩阵 + 预计算
    let (x_today_stacked, y_today_stacked) = build_stacked_design(&dp, n_lags);
    let t_f64 = t_valid as f64;
    let mut precomputed: Vec<StockPrecomputed> = Vec::with_capacity(n_stocks);
    for i in 0..n_stocks {
        let offset = i * t_valid;
        let x_i = x_today_stacked.slice(s![offset..offset + t_valid, ..]);
        let y_i = y_today_stacked.slice(s![offset..offset + t_valid]);
        let mut sp = StockPrecomputed::new(n_lags);
        sp.compute_from(&x_i, &y_i);
        precomputed.push(sp);
    }

    // 快速 R²
    let mut r2s = Array2::<f64>::from_elem((n_stocks, n_stocks), f64::NAN);
    for j in 0..n_stocks {
        let beta_j = betas.column(j);
        for i in 0..n_stocks {
            let pc = &precomputed[i];
            if pc.ss_tot <= 0.0 { continue; }
            let ss_res = pc.compute_ss_res(&beta_j, t_f64);
            if ss_res.is_finite() { r2s[[i, j]] = 1.0 - ss_res / pc.ss_tot; }
        }
    }

    let stats = extract_38_stats_weighted(&r2s.view(), &r2_pri.view(), &w);
    Ok(stats.into_pyarray(py).to_owned())
}


/// 跨股票交叉变量回归因子（加权版）
#[pyfunction]
#[pyo3(signature = (y_fit, x_fit, y_pred, x_pred, n_lags, weights))]
pub fn cross_stock_crossvar_38_weighted(
    py: Python,
    y_fit: PyReadonlyArray2<f64>,
    x_fit: PyReadonlyArray2<f64>,
    y_pred: PyReadonlyArray2<f64>,
    x_pred: PyReadonlyArray2<f64>,
    n_lags: usize,
    weights: PyReadonlyArray1<f64>,
) -> PyResult<Py<PyArray2<f64>>> {
    let yf = y_fit.as_array();
    let xf = x_fit.as_array();
    let yp = y_pred.as_array();
    let xp = x_pred.as_array();
    let w = weights.as_array();
    let n_stocks = yf.ncols();
    let t_valid = yf.nrows() - n_lags;
    let total_lags = 2 * n_lags;
    if t_valid < 3 || n_stocks == 0 {
        return Ok(Array2::<f64>::from_elem((38, n_stocks), f64::NAN).into_pyarray(py).to_owned());
    }

    // OLS
    let mut betas = Array2::<f64>::zeros((total_lags + 1, n_stocks));
    let mut r2_pri = Array1::<f64>::zeros(n_stocks);
    for j in 0..n_stocks {
        let y_col = yf.column(j); let x_col = xf.column(j);
        let mut x_j = Array2::<f64>::zeros((t_valid, total_lags));
        for lag in 0..n_lags {
            for row in 0..t_valid {
                x_j[[row, lag]] = y_col[row + n_lags - 1 - lag];
                x_j[[row, lag + n_lags]] = x_col[row + n_lags - 1 - lag];
            }
        }
        let y_j = y_col.slice(s![n_lags..]);
        let (coef, r2) = ols_with_intercept_safe(&x_j.view(), &y_j);
        for k in 0..total_lags + 1 { betas[[k, j]] = coef[k]; }
        r2_pri[j] = r2;
    }

    // 设计矩阵 + 预计算
    let mut x_stacked = Array2::<f64>::zeros((n_stocks * t_valid, total_lags));
    let mut y_stacked = Array1::<f64>::zeros(n_stocks * t_valid);
    for s in 0..n_stocks {
        let offset = s * t_valid;
        let ycol = yp.column(s); let xcol = xp.column(s);
        for lag in 0..n_lags {
            for row in 0..t_valid {
                x_stacked[[offset + row, lag]] = ycol[row + n_lags - 1 - lag];
                x_stacked[[offset + row, lag + n_lags]] = xcol[row + n_lags - 1 - lag];
            }
        }
        for row in 0..t_valid { y_stacked[offset + row] = ycol[row + n_lags]; }
    }

    let t_f64 = t_valid as f64;
    let mut precomputed: Vec<StockPrecomputed> = Vec::with_capacity(n_stocks);
    for i in 0..n_stocks {
        let offset = i * t_valid;
        let x_i = x_stacked.slice(s![offset..offset + t_valid, ..]);
        let y_i = y_stacked.slice(s![offset..offset + t_valid]);
        let mut sp = StockPrecomputed::new(total_lags);
        sp.compute_from(&x_i, &y_i);
        precomputed.push(sp);
    }

    // R²
    let mut r2s = Array2::<f64>::from_elem((n_stocks, n_stocks), f64::NAN);
    for j in 0..n_stocks {
        let beta_j = betas.column(j);
        for i in 0..n_stocks {
            let pc = &precomputed[i];
            if pc.ss_tot <= 0.0 { continue; }
            let ss_res = pc.compute_ss_res(&beta_j, t_f64);
            if ss_res.is_finite() { r2s[[i, j]] = 1.0 - ss_res / pc.ss_tot; }
        }
    }

    let stats = extract_38_stats_weighted(&r2s.view(), &r2_pri.view(), &w);
    Ok(stats.into_pyarray(py).to_owned())
}
