//! 跨股票回归因子计算：用前一日回归系数在今日数据上做迁移预测。
//!
//! 核心想法：对每只股票用前一日数据拟合自回归模型得到系数，然后用这些系数
//! 在今日数据上对每只股票做预测，计算 N×N 的交叉 R² 矩阵，从中提取 38 个统计量。
//!
//! 函数：
//! - cross_stock_autoreg_38 : 纯自回归（5/15阶滞后），对应 hm59 的 sing1/sing2
//! - cross_stock_crossvar_38: 交叉变量回归，对应 hm59 的 sing3

use ndarray::{s, Array1, Array2, ArrayView1, ArrayView2};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;

// ============================================================
// 辅助：最小二乘求解（从 statistics::solve_linear_system3 拷贝，避免跨模块依赖）
// ============================================================
pub fn solve_linear_system_safe(a: &ArrayView2<f64>, b: &ArrayView1<f64>) -> Array1<f64> {
    let n = a.nrows();
    const RIDGE: f64 = 1e-8;
    const MIN_PIVOT: f64 = 1e-12;

    // 添加 ridge 正则化
    let mut a_reg = a.to_owned();
    for i in 0..n {
        a_reg[[i, i]] += RIDGE * a_reg[[i, i]].abs().max(RIDGE);
    }

    let mut l = Array2::<f64>::zeros((n, n));
    let mut u = Array2::<f64>::zeros((n, n));

    for i in 0..n {
        for j in 0..n {
            if i <= j {
                u[[i, j]] = a_reg[[i, j]] - (0..i).map(|k| l[[i, k]] * u[[k, j]]).sum::<f64>();
                if i == j {
                    l[[i, i]] = 1.0;
                    if u[[i, i]].abs() < MIN_PIVOT {
                        u[[i, i]] = if u[[i, i]] >= 0.0 { MIN_PIVOT } else { -MIN_PIVOT };
                    }
                }
            }
            if i > j {
                l[[i, j]] =
                    (a_reg[[i, j]] - (0..j).map(|k| l[[i, k]] * u[[k, j]]).sum::<f64>()) / u[[j, j]];
                if !l[[i, j]].is_finite() {
                    l[[i, j]] = 0.0;
                }
            }
        }
    }

    let mut y = Array1::<f64>::zeros(n);
    for i in 0..n {
        y[i] = b[i] - (0..i).map(|j| l[[i, j]] * y[j]).sum::<f64>();
        if !y[i].is_finite() { y[i] = 0.0; }
    }

    let mut x = Array1::<f64>::zeros(n);
    for i in (0..n).rev() {
        x[i] = (y[i] - (i + 1..n).map(|j| u[[i, j]] * x[j]).sum::<f64>()) / u[[i, i]];
        if !x[i].is_finite() { x[i] = 0.0; }
    }
    x
}

// ============================================================
// OLS 拟合：x (n, p) → 加截距项 → 返回 [intercept, coef0, coef1, ..., r2]
// ============================================================
pub fn ols_with_intercept_safe(x: &ArrayView2<f64>, y: &ArrayView1<f64>) -> (Array1<f64>, f64) {
    let n = x.nrows();
    let p = x.ncols() + 1;

    // 统计有效行数
    let mut valid_rows = Vec::with_capacity(n);
    for i in 0..n {
        if y[i].is_finite() {
            let mut all_finite = true;
            for a in 0..x.ncols() {
                if !x[[i, a]].is_finite() {
                    all_finite = false;
                    break;
                }
            }
            if all_finite {
                valid_rows.push(i);
            }
        }
    }

    let n_valid = valid_rows.len();
    if n_valid < p + 2 {
        return (Array1::from_elem(p, f64::NAN), f64::NAN);
    }

    let mut xt_x = Array2::<f64>::zeros((p, p));
    let mut xt_y = Array1::<f64>::zeros(p);
    let mut y_sum = 0.0;

    for &i in &valid_rows {
        let yi = y[i];
        y_sum += yi;
        xt_y[0] += yi;
        xt_x[[0, 0]] += 1.0;
        for a in 1..p {
            let xia = x[[i, a - 1]];
            xt_y[a] += xia * yi;
            xt_x[[0, a]] += xia;
            xt_x[[a, 0]] += xia;
            for b in 1..=a {
                xt_x[[a, b]] += xia * x[[i, b - 1]];
                if a != b {
                    xt_x[[b, a]] = xt_x[[a, b]];
                }
            }
        }
    }

    let coef = solve_linear_system_safe(&xt_x.view(), &xt_y.view());

    // 检查系数是否有效
    if coef.iter().any(|&c| !c.is_finite()) {
        return (Array1::from_elem(p, f64::NAN), f64::NAN);
    }

    // R²
    let y_mean = y_sum / n_valid as f64;
    let mut ss_tot = 0.0;
    let mut ss_res = 0.0;
    for &i in &valid_rows {
        let yi = y[i];
        ss_tot += (yi - y_mean).powi(2);
        let mut pred = coef[0];
        for a in 1..p {
            pred += coef[a] * x[[i, a - 1]];
        }
        ss_res += (yi - pred).powi(2);
    }
    let r2 = if ss_tot > 0.0 { 1.0 - ss_res / ss_tot } else { f64::NAN };

    (coef, r2)
}

// ============================================================
// 从一列数据构建滞后设计矩阵
// data: (T,) 一列数据  → X: (T - n_lags, n_lags) 不含截距项
// ============================================================
pub fn build_lagged_design(data: &ArrayView1<f64>, n_lags: usize) -> Array2<f64> {
    let t = data.len();
    let n_rows = t - n_lags;
    let mut x = Array2::<f64>::zeros((n_rows, n_lags));
    for lag in 0..n_lags {
        for row in 0..n_rows {
            x[[row, lag]] = data[row + n_lags - 1 - lag];
        }
    }
    x
}

// ============================================================
// 从 2D 数据构建所有股票的滞后设计矩阵（堆叠）
// data: (T, N) → 对每列 n，构建 (T-n_lags, n_lags) 后纵向堆叠
// 返回: X_stacked (N*(T-n_lags), n_lags), y_stacked (N*(T-n_lags),)
// ============================================================
pub fn build_stacked_design(
    data: &ArrayView2<f64>,
    n_lags: usize,
) -> (Array2<f64>, Array1<f64>) {
    let t = data.nrows();
    let n_stocks = data.ncols();
    let t_valid = t - n_lags;
    let n_rows = n_stocks * t_valid;

    let mut x_stacked = Array2::<f64>::zeros((n_rows, n_lags));
    let mut y_stacked = Array1::<f64>::zeros(n_rows);

    for s in 0..n_stocks {
        let offset = s * t_valid;
        for lag in 0..n_lags {
            for row in 0..t_valid {
                x_stacked[[offset + row, lag]] = data[[row + n_lags - 1 - lag, s]];
            }
        }
        for row in 0..t_valid {
            y_stacked[offset + row] = data[[row + n_lags, s]];
        }
    }

    (x_stacked, y_stacked)
}

// ============================================================
// 从一行数据计算 Pearson R²（给定预测值和实际值）
// y_pred: (T,) 预测值, y_actual: (T,) 实际值
// ============================================================
fn compute_r2(y_pred: &ArrayView1<f64>, y_actual: &ArrayView1<f64>) -> f64 {
    let n = y_pred.len();
    if n < 2 {
        return f64::NAN;
    }

    // y_actual 的均值和总平方和
    let y_mean = y_actual.mean().unwrap_or(0.0);
    let ss_tot: f64 = y_actual.iter().map(|&v| (v - y_mean).powi(2)).sum();
    if ss_tot <= 0.0 {
        return f64::NAN;
    }

    // 残差平方和
    let ss_res: f64 = y_pred
        .iter()
        .zip(y_actual.iter())
        .map(|(&p, &a)| (p - a).powi(2))
        .sum();

    1.0 - ss_res / ss_tot
}

// ============================================================
// 从 r2s 矩阵提取 38 个统计量
// r2s: (N, N) 交叉 R² 矩阵, r2s[i][j] = 用股票 j 的系数预测股票 i 的 R²
// r2_pri: (N,) 各股票的自拟合 R²
// 返回: (38, N) 统计量矩阵
// ============================================================
pub fn extract_38_stats(r2s: &ArrayView2<f64>, r2_pri: &ArrayView1<f64>) -> Array2<f64> {
    let n = r2s.nrows();
    let mut stats = Array2::<f64>::zeros((38, n));

    // r2s * r2s^T（用于 corr_in_out）
    // corr_in_out[s] = 第 s 列的 r2s 和第 s 行的 r2s 之间的相关系数
    //   = corr(r2s[:, s], r2s[s, :]^T)
    // 即：r2s 的第 s 列（列 s）和 r2st 的第 s 列（即行 s 的转置）之间的相关性

    for s in 0..n {
        // ---- 提取第 s 行和第 s 列 ----
        let row_s = r2s.row(s);
        let col_s = r2s.column(s);

        // 有效值（排除 NaN）
        let mut valid_vals_row = Vec::with_capacity(n);
        let mut valid_vals_col = Vec::with_capacity(n);
        let mut corr_pairs = Vec::with_capacity(n);
        for k in 0..n {
            let r = row_s[k];
            let c = col_s[k];
            if r.is_finite() && c.is_finite() {
                corr_pairs.push((r, c));
            }
            if r.is_finite() {
                valid_vals_row.push(r);
            }
            if c.is_finite() {
                valid_vals_col.push(c);
            }
        }

        // ---- corr_in_out ----
        stats[[0, s]] = pearson_corr_pairs(&corr_pairs);

        // ---- r2_in_uprate, r2_out_uprate, trans_out_uprate ----
        let r2p = if r2_pri[s].is_finite() { r2_pri[s] } else { f64::NAN };
        let n_valid = valid_vals_row.len().max(1);
        let mut up_in = 0;
        let mut up_out = 0;
        let mut trans_out = 0;
        for k in 0..n {
            let r = row_s[k];
            let c = col_s[k];
            if r.is_finite() {
                if r > 0.0 && r > r2p {
                    up_in += 1;
                }
                if c.is_finite() && r > c {
                    trans_out += 1;
                }
            }
            if c.is_finite() {
                if c > 0.0 && c > r2p {
                    up_out += 1;
                }
            }
        }
        stats[[1, s]] = up_in as f64 / n_valid as f64;
        stats[[2, s]] = up_out as f64 / n_valid as f64;
        stats[[3, s]] = trans_out as f64 / n_valid as f64;

        // ---- 行统计量（in_*）和列统计量（out_*）和 trans_* ----
        // in: row_s (stock s 被其他股票预测)
        // out: col_s (stock s 预测其他股票)
        // trans: row_s - col_s
        // in_ab: row_s - r2_pri[s] (减去自己的自R²)
        // out_ab: col_s - r2_pri[s]

        let (in_mean, in_std, in_skew, in_kurt) = compute_moments(&valid_vals_row);
        let (out_mean, out_std, out_skew, out_kurt) = compute_moments(&valid_vals_col);

        // trans = row - col, 需要行列配对
        let mut trans_vals = Vec::with_capacity(corr_pairs.len());
        for &(r, c) in &corr_pairs {
            trans_vals.push(r - c);
        }
        let (trans_mean, trans_std, trans_skew, trans_kurt) = compute_moments(&trans_vals);

        // in_ab = row - r2_pri[s], out_ab = col - r2_pri[j] for each j
        let mut in_ab_vals = Vec::with_capacity(n);
        let mut out_ab_vals = Vec::with_capacity(n);
        for k in 0..n {
            let r = row_s[k];
            let c = col_s[k];
            let pri_k = if r2_pri[k].is_finite() { r2_pri[k] } else { f64::NAN };
            if r.is_finite() {
                in_ab_vals.push(r - r2p);
            }
            if c.is_finite() {
                out_ab_vals.push(c - pri_k);
            }
        }
        let (in_ab_mean, in_ab_std, in_ab_skew, in_ab_kurt) = compute_moments(&in_ab_vals);
        let (out_ab_mean, out_ab_std, out_ab_skew, out_ab_kurt) = compute_moments(&out_ab_vals);

        // 均值: stats[4..9]
        stats[[4, s]] = in_mean;
        stats[[5, s]] = out_mean;
        stats[[6, s]] = in_ab_mean;
        stats[[7, s]] = out_ab_mean;
        stats[[8, s]] = trans_mean;

        // 标准差: stats[9..14]
        stats[[9, s]] = in_std;
        stats[[10, s]] = out_std;
        stats[[11, s]] = in_ab_std;
        stats[[12, s]] = out_ab_std;
        stats[[13, s]] = trans_std;

        // 偏度: stats[14..19]
        stats[[14, s]] = in_skew;
        stats[[15, s]] = out_skew;
        stats[[16, s]] = in_ab_skew;
        stats[[17, s]] = out_ab_skew;
        stats[[18, s]] = trans_skew;

        // 峰度: stats[19..24]
        stats[[19, s]] = in_kurt;
        stats[[20, s]] = out_kurt;
        stats[[21, s]] = in_ab_kurt;
        stats[[22, s]] = out_ab_kurt;
        stats[[23, s]] = trans_kurt;

        // 中位数: stats[24..29]
        stats[[24, s]] = median(&valid_vals_row);
        stats[[25, s]] = median(&valid_vals_col);
        stats[[26, s]] = median(&in_ab_vals);
        stats[[27, s]] = median(&out_ab_vals);
        stats[[28, s]] = median(&trans_vals);

        // 最大值: stats[29..33]
        stats[[29, s]] = max_finite(&valid_vals_row);
        stats[[30, s]] = max_finite(&in_ab_vals);
        stats[[31, s]] = max_finite(&out_ab_vals);
        stats[[32, s]] = max_finite(&trans_vals);

        // 最小值: stats[33..38]
        stats[[33, s]] = min_finite(&valid_vals_row);
        stats[[34, s]] = min_finite(&valid_vals_col);
        stats[[35, s]] = min_finite(&in_ab_vals);
        stats[[36, s]] = min_finite(&out_ab_vals);
        stats[[37, s]] = min_finite(&trans_vals);
    }

    stats
}

// ============================================================
// 加权版 38 统计量提取
// ============================================================

/// 从 r2s 矩阵提取 38 个加权统计量
/// weights: (N,) 正权重向量，log(总成交量)
/// 行统计量（in_*）：以 weights[k] 加权 r2s[s, k]
/// 列统计量（out_*）：以 weights[k] 加权 r2s[k, s]
pub fn extract_38_stats_weighted(r2s: &ArrayView2<f64>, r2_pri: &ArrayView1<f64>, weights: &ArrayView1<f64>) -> Array2<f64> {
    let n = r2s.nrows();
    let mut stats = Array2::<f64>::zeros((38, n));

    // 总权重
    let sum_w: f64 = weights.iter().filter(|v| v.is_finite() && **v > 0.0).sum();
    if sum_w <= 0.0 {
        return stats;  // all NaN
    }

    for s in 0..n {
        let row_s = r2s.row(s);
        let col_s = r2s.column(s);
        let ws = if weights[s].is_finite() && weights[s] > 0.0 { weights[s] } else { 0.0 };

        // 收集行列有效值及对应权重
        let mut pairs_row: Vec<(f64, f64)> = Vec::with_capacity(n);
        let mut pairs_col: Vec<(f64, f64)> = Vec::with_capacity(n);
        let mut trans_pairs: Vec<(f64, f64)> = Vec::with_capacity(n);
        let mut in_ab_pairs: Vec<(f64, f64)> = Vec::with_capacity(n);
        let mut out_ab_pairs: Vec<(f64, f64)> = Vec::with_capacity(n);

        for k in 0..n {
            let r = row_s[k];
            let c = col_s[k];
            let wk = if weights[k].is_finite() && weights[k] > 0.0 { weights[k] } else { 0.0 };
            if r.is_finite() && wk > 0.0 {
                pairs_row.push((r, wk));
            }
            if c.is_finite() && wk > 0.0 {
                pairs_col.push((c, wk));
            }
            if r.is_finite() && c.is_finite() && wk > 0.0 {
                trans_pairs.push((r - c, wk));
            }
            let pri_k = if r2_pri[k].is_finite() { r2_pri[k] } else { f64::NAN };
            let r2p = if r2_pri[s].is_finite() { r2_pri[s] } else { f64::NAN };
            if r.is_finite() && r2p.is_finite() && wk > 0.0 {
                in_ab_pairs.push((r - r2p, wk));
            }
            if c.is_finite() && pri_k.is_finite() && wk > 0.0 {
                out_ab_pairs.push((c - pri_k, wk));
            }
        }

        // corr_in_out: 仍用未加权 Pearson（因为两列长度相同，等权合理）
        let mut corr_pairs = Vec::with_capacity(n);
        for k in 0..n {
            let r = row_s[k]; let c = col_s[k];
            if r.is_finite() && c.is_finite() { corr_pairs.push((r, c)); }
        }
        stats[[0, s]] = pearson_corr_pairs(&corr_pairs);

        // uprate: 仍用未加权比例
        let r2p = if r2_pri[s].is_finite() { r2_pri[s] } else { f64::NAN };
        let n_valid = pairs_row.len().max(1);
        let mut up_in = 0usize; let mut up_out = 0usize; let mut trans_out = 0usize;
        for k in 0..n {
            let r = row_s[k]; let c = col_s[k];
            if r.is_finite() { if r > 0.0 && r > r2p { up_in += 1; } if c.is_finite() && r > c { trans_out += 1; } }
            if c.is_finite() && c > 0.0 && c > r2p { up_out += 1; }
        }
        stats[[1, s]] = up_in as f64 / n_valid as f64;
        stats[[2, s]] = up_out as f64 / n_valid as f64;
        stats[[3, s]] = trans_out as f64 / n_valid as f64;

        // 加权矩统计
        let (im, io, ik, ikt) = weighted_moments(&pairs_row);
        let (om, oo, ok, okt) = weighted_moments(&pairs_col);
        let (tm, to, tk, tkt) = weighted_moments(&trans_pairs);
        let (iam, iao, iak, iakt) = weighted_moments(&in_ab_pairs);
        let (oam, oao, oak, oakt) = weighted_moments(&out_ab_pairs);

        stats[[4, s]] = im;  stats[[5, s]] = om;
        stats[[6, s]] = iam; stats[[7, s]] = oam; stats[[8, s]] = tm;
        stats[[9, s]] = io;  stats[[10, s]] = oo;
        stats[[11, s]] = iao; stats[[12, s]] = oao; stats[[13, s]] = to;
        stats[[14, s]] = ik; stats[[15, s]] = ok;
        stats[[16, s]] = iak; stats[[17, s]] = oak; stats[[18, s]] = tk;
        stats[[19, s]] = ikt; stats[[20, s]] = okt;
        stats[[21, s]] = iakt; stats[[22, s]] = oakt; stats[[23, s]] = tkt;

        // 加权中位数
        stats[[24, s]] = weighted_median(&pairs_row);
        stats[[25, s]] = weighted_median(&pairs_col);
        stats[[26, s]] = weighted_median(&in_ab_pairs);
        stats[[27, s]] = weighted_median(&out_ab_pairs);
        stats[[28, s]] = weighted_median(&trans_pairs);

        // 最大值
        stats[[29, s]] = pairs_row.iter().map(|(v,_)| *v).fold(f64::NAN, |a,v| if v.is_finite() && (a.is_nan() || v>a) {v} else {a});
        stats[[30, s]] = in_ab_pairs.iter().map(|(v,_)| *v).fold(f64::NAN, |a,v| if v.is_finite() && (a.is_nan() || v>a) {v} else {a});
        stats[[31, s]] = out_ab_pairs.iter().map(|(v,_)| *v).fold(f64::NAN, |a,v| if v.is_finite() && (a.is_nan() || v>a) {v} else {a});
        stats[[32, s]] = trans_pairs.iter().map(|(v,_)| *v).fold(f64::NAN, |a,v| if v.is_finite() && (a.is_nan() || v>a) {v} else {a});

        // 最小值
        stats[[33, s]] = pairs_row.iter().map(|(v,_)| *v).fold(f64::NAN, |a,v| if v.is_finite() && (a.is_nan() || v<a) {v} else {a});
        stats[[34, s]] = pairs_col.iter().map(|(v,_)| *v).fold(f64::NAN, |a,v| if v.is_finite() && (a.is_nan() || v<a) {v} else {a});
        stats[[35, s]] = in_ab_pairs.iter().map(|(v,_)| *v).fold(f64::NAN, |a,v| if v.is_finite() && (a.is_nan() || v<a) {v} else {a});
        stats[[36, s]] = out_ab_pairs.iter().map(|(v,_)| *v).fold(f64::NAN, |a,v| if v.is_finite() && (a.is_nan() || v<a) {v} else {a});
        stats[[37, s]] = trans_pairs.iter().map(|(v,_)| *v).fold(f64::NAN, |a,v| if v.is_finite() && (a.is_nan() || v<a) {v} else {a});
    }
    stats
}

/// 加权矩：返回 (mean, std, skew, kurt)
fn weighted_moments(pairs: &[(f64, f64)]) -> (f64, f64, f64, f64) {
    let n = pairs.len();
    if n < 3 { return (f64::NAN, f64::NAN, f64::NAN, f64::NAN); }
    let sum_w: f64 = pairs.iter().map(|(_, w)| w).sum();
    if sum_w <= 0.0 { return (f64::NAN, f64::NAN, f64::NAN, f64::NAN); }
    let mean = pairs.iter().map(|(v, w)| v * w).sum::<f64>() / sum_w;
    let m2 = pairs.iter().map(|(v, w)| w * (v - mean).powi(2)).sum::<f64>() / sum_w;
    if m2 <= 0.0 { return (mean, 0.0, f64::NAN, f64::NAN); }
    let std = (m2 * n as f64 / (n as f64 - 1.0)).sqrt();
    let m3 = pairs.iter().map(|(v, w)| w * (v - mean).powi(3)).sum::<f64>() / sum_w;
    let m4 = pairs.iter().map(|(v, w)| w * (v - mean).powi(4)).sum::<f64>() / sum_w;
    let skew = if std > 0.0 { (n as f64).sqrt() * m3 / (std * std * std) * (n as f64 - 1.0).sqrt() / (n as f64).sqrt() } else { f64::NAN };
    let kurt = if std > 0.0 { n as f64 * m4 / (m2 * m2 * (n as f64 - 1.0)) - 3.0 } else { f64::NAN };
    (mean, std, skew, kurt)
}

/// 加权中位数
fn weighted_median(pairs: &[(f64, f64)]) -> f64 {
    if pairs.is_empty() { return f64::NAN; }
    let mut sorted: Vec<(f64, f64)> = pairs.to_vec();
    sorted.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let total_w: f64 = sorted.iter().map(|(_, w)| w).sum();
    if total_w <= 0.0 { return f64::NAN; }
    let half = total_w / 2.0;
    let mut cum = 0.0;
    for (v, w) in &sorted {
        cum += w;
        if cum >= half { return *v; }
    }
    sorted.last().unwrap().0
}

// ============================================================
// 统计辅助函数
// ============================================================

/// Pearson 相关系数（成对数据）
fn pearson_corr_pairs(pairs: &[(f64, f64)]) -> f64 {
    let n = pairs.len();
    if n < 3 {
        return f64::NAN;
    }
    let mean_x = pairs.iter().map(|p| p.0).sum::<f64>() / n as f64;
    let mean_y = pairs.iter().map(|p| p.1).sum::<f64>() / n as f64;
    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;
    for &(x, y) in pairs {
        let dx = x - mean_x;
        let dy = y - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }
    if var_x <= 0.0 || var_y <= 0.0 {
        return f64::NAN;
    }
    cov / (var_x.sqrt() * var_y.sqrt())
}

/// 计算前四阶矩（均值、标准差、偏度、峰度）
fn compute_moments(vals: &[f64]) -> (f64, f64, f64, f64) {
    let n = vals.len();
    if n < 3 {
        return (f64::NAN, f64::NAN, f64::NAN, f64::NAN);
    }
    let mean = vals.iter().sum::<f64>() / n as f64;
    let m2: f64 = vals.iter().map(|&v| (v - mean).powi(2)).sum();
    if m2 <= 0.0 {
        return (mean, 0.0, f64::NAN, f64::NAN);
    }
    let std = (m2 / (n as f64 - 1.0)).sqrt();
    let m3: f64 = vals.iter().map(|&v| (v - mean).powi(3)).sum();
    let m4: f64 = vals.iter().map(|&v| (v - mean).powi(4)).sum();
    // 样本偏度（使用 np.skew 的公式）
    let skew = if std > 0.0 {
        (n as f64).sqrt() * m3 / (m2.powf(1.5))
    } else {
        f64::NAN
    };
    // 超额峰度（pandas kurt 默认使用 Pearson/Fisher 定义，即超额峰度）
    let kurt = if std > 0.0 {
        (n as f64) * m4 / (m2 * m2) - 3.0
    } else {
        f64::NAN
    };
    (mean, std, skew, kurt)
}

fn median(vals: &[f64]) -> f64 {
    let mut finite: Vec<f64> = vals.iter().filter(|v| v.is_finite()).copied().collect();
    if finite.len() < 2 {
        return f64::NAN;
    }
    finite.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    let n = finite.len();
    if n % 2 == 1 {
        finite[n / 2]
    } else {
        (finite[n / 2 - 1] + finite[n / 2]) / 2.0
    }
}

fn max_finite(vals: &[f64]) -> f64 {
    vals.iter().fold(f64::NAN, |acc, &v| {
        if v.is_finite() && (acc.is_nan() || v > acc) {
            v
        } else {
            acc
        }
    })
}

fn min_finite(vals: &[f64]) -> f64 {
    vals.iter().fold(f64::NAN, |acc, &v| {
        if v.is_finite() && (acc.is_nan() || v < acc) {
            v
        } else {
            acc
        }
    })
}

// ============================================================
// 主函数：纯自回归跨股票因子
// ============================================================

/// 跨股票自回归因子（用前一日系数在今日数据上做迁移预测）
///
/// data_prev: (T, N) 前一日数据（已标准化，无 NaN 行）
/// data_today: (T, N) 今日数据（已标准化，无 NaN 行）
/// n_lags: 滞后阶数（5 或 15）
///
/// 返回: (38, N) 每只股票的 38 个统计量
#[pyfunction]
#[pyo3(signature = (data_prev, data_today, n_lags))]
pub fn cross_stock_autoreg_38(
    py: Python,
    data_prev: PyReadonlyArray2<f64>,
    data_today: PyReadonlyArray2<f64>,
    n_lags: usize,
) -> PyResult<Py<PyArray2<f64>>> {
    let dp = data_prev.as_array();
    let dt = data_today.as_array();
    let n_stocks = dp.ncols();
    let t_valid = dp.nrows() - n_lags;

    if t_valid < 3 {
        return Ok(Array2::<f64>::from_elem((38, n_stocks), f64::NAN)
            .into_pyarray(py)
            .to_owned());
    }

    // ---- 第一步：对每只股票用前一日数据拟合 OLS ----
    let mut betas = Array2::<f64>::zeros((n_lags + 1, n_stocks)); // (intercept + lags, N)
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

    // ---- 第二步：构建今日设计矩阵 + 转 f32 预计算 ----
    let (x_today_stacked, y_today_stacked) = build_stacked_design(&dt, n_lags);

    // y 转 f32 预计算 ss_tot（R² 分母）
    let y_f32: Vec<f32> = y_today_stacked.as_slice().unwrap().iter().map(|&v| v as f32).collect();
    let mut y_ss_tot = vec![0.0f32; n_stocks];
    for i in 0..n_stocks {
        let offset = i * t_valid;
        let chunk = &y_f32[offset..offset + t_valid];
        let mean = chunk.iter().sum::<f32>() / t_valid as f32;
        y_ss_tot[i] = chunk.iter().map(|&v| (v - mean) * (v - mean)).sum();
    }

    // ---- 第三步：逐列计算交叉 R²（dot 用 ndarray/f64，R² 用 f32 裸切片） ----
    let mut r2s_f32 = vec![f32::NAN; n_stocks * n_stocks];

    for j in 0..n_stocks {
        let coef_j = betas.column(j);
        let intercept = coef_j[0];
        let coef_no_intercept = coef_j.slice(s![1..]);
        // ndarray 的 dot 是优化过的 BLAS 级实现
        let y_pred_all = x_today_stacked.dot(&coef_no_intercept).mapv(|v| v + intercept);
        let y_pred_slice = y_pred_all.as_slice().unwrap();

        for i in 0..n_stocks {
            let ss_tot = y_ss_tot[i];
            if ss_tot <= 0.0f32 {
                continue;
            }
            let offset = i * t_valid;
            let mut ss_res = 0.0f32;
            for k in 0..t_valid {
                let diff = y_pred_slice[offset + k] as f32 - y_f32[offset + k];
                ss_res += diff * diff;
            }
            r2s_f32[i * n_stocks + j] = 1.0f32 - ss_res / ss_tot;
        }
    }

    // 转回 f64 提取统计量
    let r2s_f64: Vec<f64> = r2s_f32.iter().map(|&v| v as f64).collect();
    let r2s = ArrayView2::from_shape((n_stocks, n_stocks), &r2s_f64).unwrap();
    // ---- 第四步：提取 38 个统计量 ----
    let stats = extract_38_stats(&r2s.view(), &r2_pri.view());
    Ok(stats.into_pyarray(py).to_owned())
}

// ============================================================
// 主函数：交叉变量回归跨股票因子（对应 hm59 的 sing3）
// ============================================================

/// 跨股票交叉变量回归因子
///
/// y_prev: (T, N) 前一日因变量
/// x_prev: (T, N) 前一日自变量
/// y_today: (T, N) 今日因变量
/// x_today: (T, N) 今日自变量
/// n_lags: 每变量的滞后阶数（通常为 5）
///
/// 模型：y ~ y_lag1..n_lags + x_lag1..n_lags
///
/// 返回: (38, N)
#[pyfunction]
#[pyo3(signature = (y_prev, x_prev, y_today, x_today, n_lags))]
pub fn cross_stock_crossvar_38(
    py: Python,
    y_prev: PyReadonlyArray2<f64>,
    x_prev: PyReadonlyArray2<f64>,
    y_today: PyReadonlyArray2<f64>,
    x_today: PyReadonlyArray2<f64>,
    n_lags: usize,
) -> PyResult<Py<PyArray2<f64>>> {
    let yp = y_prev.as_array();
    let xp = x_prev.as_array();
    let yt = y_today.as_array();
    let xt = x_today.as_array();

    let n_stocks = yp.ncols();
    let t_valid = yp.nrows() - n_lags;
    let total_lags = 2 * n_lags;

    if t_valid < 3 {
        return Ok(Array2::<f64>::from_elem((38, n_stocks), f64::NAN)
            .into_pyarray(py)
            .to_owned());
    }

    // ---- 第一步：对每只股票拟合 OLS (prev day) ----
    // y_prev ~ y_lag1..n_lags + x_lag1..n_lags
    let mut betas = Array2::<f64>::zeros((total_lags + 1, n_stocks));
    let mut r2_pri = Array1::<f64>::zeros(n_stocks);

    for j in 0..n_stocks {
        let y_col = yp.column(j);
        let x_col = xp.column(j);
        // 构建滞后设计矩阵（y_lags + x_lags，不含截距）
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

    // ---- 第二步：构建今日数据的堆叠设计矩阵 ----
    // y_today 的滞后 + x_today 的滞后
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
    let y_f32: Vec<f32> = y_stacked.as_slice().unwrap().iter().map(|&v| v as f32).collect();
    let mut y_ss_tot = vec![0.0f32; n_stocks];
    for i in 0..n_stocks {
        let offset = i * t_valid;
        let chunk = &y_f32[offset..offset + t_valid];
        let mean = chunk.iter().sum::<f32>() / t_valid as f32;
        y_ss_tot[i] = chunk.iter().map(|&v| (v - mean) * (v - mean)).sum();
    }

    // ---- 第四步：逐列计算交叉 R²（dot 用 ndarray/f64，R² 用 f32 裸切片） ----
    let mut r2s_f32 = vec![f32::NAN; n_stocks * n_stocks];

    for j in 0..n_stocks {
        let coef_j = betas.column(j);
        let intercept = coef_j[0];
        let coef_no_intercept = coef_j.slice(s![1..]);
        let y_pred_all = x_stacked.dot(&coef_no_intercept).mapv(|v| v + intercept);
        let y_pred_slice = y_pred_all.as_slice().unwrap();

        for i in 0..n_stocks {
            let ss_tot = y_ss_tot[i];
            if ss_tot <= 0.0f32 {
                continue;
            }
            let offset = i * t_valid;
            let mut ss_res = 0.0f32;
            for k in 0..t_valid {
                let diff = y_pred_slice[offset + k] as f32 - y_f32[offset + k];
                ss_res += diff * diff;
            }
            r2s_f32[i * n_stocks + j] = 1.0f32 - ss_res / ss_tot;
        }
    }

    // 转回 f64 提取统计量
    let r2s_f64: Vec<f64> = r2s_f32.iter().map(|&v| v as f64).collect();
    let r2s = ArrayView2::from_shape((n_stocks, n_stocks), &r2s_f64).unwrap();

    let stats = extract_38_stats(&r2s.view(), &r2_pri.view());
    Ok(stats.into_pyarray(py).to_owned())
}
