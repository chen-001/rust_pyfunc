//! dev_sandbox_neutralize: 生产标准「因子合成前预处理」的 Rust 对齐实现。
//!
//! 对齐对象: `/home/chenzongwei/pythoncode/回归校准/preprocess_factor_standalone.py`
//! 的 `fill_and_rank_factors` (2026-08-07 生产口径), 逐函数逐行一致:
//!
//!   rank pct → 行业 OLS 回归填充(base=size, ind2→ind1→ind0)
//!           → 行业 2 级中位填充(median 基于填充前原始值)
//!           → 限制股置空(restrict==0 保留) → rank pct
//!           → 含截距 10 风格 OLS 残差(样本>10, y 常数截面置 0.5) → 残差 rank pct
//!
//! 关键语义细节 (与 Python 版逐条对齐):
//! - rank pct = pandas `rank(axis=1, pct=True)`: 平均秩, pct=rank/n_non_nan
//! - 行业 OLS 填充: 每日因子 unique 值 <10 跳过该日; 每行业样本 >=10 才回归;
//!   同一行业层内各行业码基于该日原始行独立填充(拷贝写回)
//! - 行业中位填充: 三层 median 全部基于填充前的 fv (median_source 固定),
//!   行业码 NaN 位置先置 NaN 且不参与填充; 末级全市场中位仅对行业码非NaN位置生效
//! - 残差回归: X=[1, 10风格], 风格与 base 均先 rank pct (对应 load_context 的 _ranked);
//!   X'X 只在因子有效子集上计算 (numpy lstsq 语义, 与 Rust 旧 neutralize 的全截面
//!   预计算矩阵不同); y 截面 unique==1 时残差整体置 0.5
//!
//! 用法:
//!   import dev_sandbox_neutralize as dn
//!   resid_rank = dn.neutralize_std_pipeline(
//!       factor, industry, restrict, barra_list, ind_base_list)  # 全部 (T,N) f64 ndarray
//!   # barra_list: [10风格原始值], ind_base_list: [size原始值]; 内部自动 rank pct
//!
//! 输入约定 (对应 preprocess_factor_standalone.load_context):
//!   factor    (T,N) 原始因子值
//!   industry  (T,N) 申万行业码矩阵 (如 170201.0; NaN=无行业)
//!   restrict  (T,N) 限制股矩阵 (0=可交易; 其余置空)
//!   barra_list  10个 (T,N) 风格原始值 (residual_volatility..non_linear_size)
//!   ind_base_list [size] (T,N) 行业填充 base 原始值

use nalgebra::DMatrix;
use ndarray::{Array1, Array2};
use numpy::{IntoPyArray, PyReadonlyArray2, PyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// NaN 安全比较: NaN 视为最大, 保证全序 (partial_cmp 遇 NaN 返回 None 会 panic)。
fn cmp_f64(a: &f64, b: &f64) -> std::cmp::Ordering {
    match (a.is_nan(), b.is_nan()) {
        (true, true) => std::cmp::Ordering::Equal,
        (true, false) => std::cmp::Ordering::Greater,
        (false, true) => std::cmp::Ordering::Less,
        (false, false) => a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal),
    }
}

/// numpy lstsq 语义的最小二乘解: coef = pinv(X) @ y (SVD, rcond 截断同 numpy 默认)。
fn lstsq(x: &Array2<f64>, y: &[f64]) -> Vec<f64> {
    let (n, k) = x.dim();
    let xm = DMatrix::from_row_slice(n, k, x.as_slice().unwrap());
    let ym = DMatrix::from_column_slice(n, 1, y);
    let svd = xm.clone().svd(true, true);
    let u = svd.u.expect("svd u");
    let vt = svd.v_t.expect("svd vt");
    let s = svd.singular_values;
    let s_max = s.iter().cloned().fold(0.0_f64, f64::max);
    let rcond = s_max * (n.max(k) as f64) * 2.22e-16;
    let uty = u.transpose() * ym; // (k,1)
    let mut coef = DMatrix::zeros(k, 1);
    for i in 0..k {
        let si = s[i];
        if si > rcond {
            coef[(i, 0)] = uty[(i, 0)] / si;
        }
    }
    (vt.transpose() * coef).column(0).iter().cloned().collect()
}

/// pandas `rank(axis=1, pct=True)` 语义: 行内非NaN平均秩, pct = rank / n_non_nan。
fn rank_pct_row(vals: &[f64]) -> Vec<f64> {
    let n = vals.len();
    let mut indexed: Vec<(usize, f64)> = vals
        .iter()
        .enumerate()
        .filter(|(_, v)| !v.is_nan())
        .map(|(i, &v)| (i, v))
        .collect();
    indexed.sort_by(|a, b| cmp_f64(&a.1, &b.1));
    let n_valid = indexed.len();
    let mut ranks = vec![f64::NAN; n];
    if n_valid == 0 {
        return ranks;
    }
    let mut i = 0;
    while i < n_valid {
        let mut j = i;
        while j + 1 < n_valid && indexed[j + 1].1 == indexed[i].1 {
            j += 1;
        }
        let avg_rank = ((i + 1) + (j + 1)) as f64 / 2.0;
        for item in &indexed[i..=j] {
            ranks[item.0] = avg_rank / n_valid as f64;
        }
        i = j + 1;
    }
    ranks
}

fn rank_pct_all(values: &mut Array2<f64>) {
    for mut row in values.rows_mut() {
        let ranked = rank_pct_row(row.as_slice().unwrap());
        row.assign(&Array1::from_vec(ranked));
    }
}

/// 行业 OLS 回归填充 (fill_factor_by_ind_reg, base 已 rank pct)。
fn fill_ind_reg(fv: &mut Array2<f64>, base_list: &[Array2<f64>], ind3: &Array2<f64>) {
    let (t, n) = fv.dim();
    let n_base = base_list.len();
    let ind2 = ind3.map(|&v| (v / 100.0).floor());
    let ind1 = ind3.map(|&v| (v / 10000.0).floor());
    let ind0 = ind1.map(|&v| if v.is_nan() { 0.0 } else { 1.0 });

    for ind_level in [&ind2, &ind1, &ind0] {
        for idx in 0..t {
            // 拷贝当日行, 避免持有 fv 视图导致的借用冲突
            let mut fv_row: Vec<f64> = fv.row(idx).iter().copied().collect();
            let mut uniq = fv_row.clone();
            uniq.sort_by(cmp_f64);
            uniq.dedup_by(|a, b| (a.is_nan() && b.is_nan()) || a == b);
            if uniq.len() < 10 {
                continue;
            }
            let ind_row = ind_level.row(idx);
            let mut codes: Vec<f64> = ind_row.iter().copied().filter(|v| !v.is_nan()).collect();
            codes.sort_by(cmp_f64);
            codes.dedup_by(|a, b| (a.is_nan() && b.is_nan()) || a == b);
            for &code in &codes {
                // mask 内原始列索引
                let cols: Vec<usize> = (0..n).filter(|&j| ind_row[j] == code).collect();
                let m_count = cols.len();
                // _f = factor_row[cols] 拷贝; _b = base[k][cols] (m, K)
                let mut f_masked: Vec<f64> = cols.iter().map(|&j| fv_row[j]).collect();
                let mut b_masked: Vec<Vec<f64>> = (0..m_count)
                    .map(|mi| (0..n_base).map(|k| base_list[k][[idx, cols[mi]]]).collect())
                    .collect();
                // _not_nan = ~isnan(_f) & ~isnan(_b).any(axis=1)
                let not_nan: Vec<bool> = (0..m_count)
                    .map(|i| !f_masked[i].is_nan() && b_masked[i].iter().all(|&v| !v.is_nan()))
                    .collect();
                let n_obs = not_nan.iter().filter(|&&b| b).count();
                if n_obs >= 10 {
                    let mut x_obs = Array2::<f64>::zeros((n_obs, n_base + 1));
                    let mut y_obs = Vec::with_capacity(n_obs);
                    let mut obs_i = 0;
                    for i in 0..m_count {
                        if not_nan[i] {
                            x_obs[[obs_i, 0]] = 1.0;
                            for k in 0..n_base {
                                x_obs[[obs_i, k + 1]] = b_masked[i][k];
                            }
                            y_obs.push(f_masked[i]);
                            obs_i += 1;
                        }
                    }
                    let coef = lstsq(&x_obs, &y_obs);
                    // _f[~_not_nan] = [1, _b_miss] @ coef
                    for i in 0..m_count {
                        if !not_nan[i] {
                            let mut pred = coef[0];
                            for k in 0..n_base {
                                pred += coef[k + 1] * b_masked[i][k];
                            }
                            f_masked[i] = pred;
                        }
                    }
                    for (mi, &ci) in cols.iter().enumerate() {
                        fv_row[ci] = f_masked[mi];
                    }
                }
            }
            for j in 0..n {
                fv[[idx, j]] = fv_row[j];
            }
        }
    }
}

/// _fill_by_group_median: 行内分组中位填充 (median 来自 median_source, 固定不重算)。
fn fill_by_group_median(
    values: &mut Array2<f64>,
    codes: &Array2<f64>,
    valid_mask: Option<&Array2<f64>>,
    median_source: &Array2<f64>,
) {
    let (t, n) = values.dim();
    for idx in 0..t {
        let mut row = values.row_mut(idx);
        let source_row = median_source.row(idx);
        let nan_mask: Vec<bool> = row.iter().map(|v| v.is_nan()).collect();
        if !nan_mask.iter().any(|&b| b) {
            continue;
        }
        let codes_row = codes.row(idx);
        let mut uniq: Vec<f64> = codes_row.iter().copied().filter(|v| !v.is_nan()).collect();
        uniq.sort_by(cmp_f64);
        uniq.dedup_by(|a, b| (a.is_nan() && b.is_nan()) || a == b);
        for &code in &uniq {
            let m: Vec<bool> = (0..n)
                .map(|j| {
                    codes_row[j] == code
                        && valid_mask.map_or(true, |vm| vm[[idx, j]] == 1.0)
                })
                .collect();
            let mut sv: Vec<f64> = (0..n)
                .filter(|&j| m[j] && !source_row[j].is_nan())
                .map(|j| source_row[j])
                .collect();
            if !sv.is_empty() {
                sv.sort_by(cmp_f64);
                let med = if sv.len() % 2 == 1 {
                    sv[sv.len() / 2]
                } else {
                    (sv[sv.len() / 2 - 1] + sv[sv.len() / 2]) / 2.0
                };
                for j in 0..n {
                    if nan_mask[j] && m[j] {
                        row[j] = med;
                    }
                }
            }
        }
    }
}

/// get_residual: 逐日对基准因子(已 rank pct)做含截距 OLS 取残差。
/// 样本数 >10 才回归; y 截面 unique==1 时残差整体置 0.5。
fn get_residual(fv: &Array2<f64>, bench: &[Array2<f64>]) -> Array2<f64> {
    let (t, n) = fv.dim();
    let mut resid = Array2::<f64>::from_elem((t, n), f64::NAN);
    for idx in 0..t {
        let row = fv.row(idx);
        if !row.iter().any(|v| !v.is_nan()) {
            continue; // dropna(how="all")
        }
        let f_v: Vec<f64> = row.iter().copied().collect();
        let k = bench.len();
        let valid: Vec<bool> = (0..n)
            .map(|j| f_v[j].is_finite() && bench.iter().all(|b| b[[idx, j]].is_finite()))
            .collect();
        let n_valid = valid.iter().filter(|&&b| b).count();
        if n_valid <= 10 {
            continue;
        }
        let y: Vec<f64> = (0..n).filter(|&j| valid[j]).map(|j| f_v[j]).collect();
        let mut uniq = y.clone();
        uniq.sort_by(cmp_f64);
        uniq.dedup_by(|a, b| (a.is_nan() && b.is_nan()) || a == b);
        let mut resid_row = vec![f64::NAN; n];
        if uniq.len() == 1 {
            for j in 0..n {
                if valid[j] {
                    resid_row[j] = 0.5;
                }
            }
        } else {
            let mut x = Array2::<f64>::zeros((n_valid, k + 1));
            let mut vi = 0;
            for j in 0..n {
                if valid[j] {
                    x[[vi, 0]] = 1.0;
                    for c in 0..k {
                        x[[vi, c + 1]] = bench[c][[idx, j]];
                    }
                    vi += 1;
                }
            }
            let coef = lstsq(&x, &y);
            let mut vi = 0;
            for j in 0..n {
                if valid[j] {
                    let mut pred = coef[0];
                    for c in 0..k {
                        pred += coef[c + 1] * bench[c][[idx, j]];
                    }
                    resid_row[j] = y[vi] - pred;
                    vi += 1;
                }
            }
        }
        for j in 0..n {
            resid[[idx, j]] = resid_row[j];
        }
    }
    resid
}

/// 生产标准预处理主入口 (对齐 preprocess_factor_standalone.fill_and_rank_factors)。
#[pyfunction]
#[pyo3(signature = (factor, industry, restrict, barra_list, ind_base_list))]
fn neutralize_std_pipeline<'py>(
    py: Python<'py>,
    factor: PyReadonlyArray2<'py, f64>,
    industry: PyReadonlyArray2<'py, f64>,
    restrict: PyReadonlyArray2<'py, f64>,
    barra_list: Vec<PyReadonlyArray2<'py, f64>>,
    ind_base_list: Vec<PyReadonlyArray2<'py, f64>>,
) -> PyResult<Py<PyArray2<f64>>> {
    let (t, n) = factor.as_array().dim();
    for (name, arr) in [
        ("industry", industry.as_array()),
        ("restrict", restrict.as_array()),
    ] {
        if arr.dim() != (t, n) {
            return Err(PyValueError::new_err(format!(
                "{} shape {:?} != factor {:?}",
                name,
                arr.dim(),
                (t, n)
            )));
        }
    }
    for (i, b) in barra_list.iter().enumerate() {
        if b.as_array().dim() != (t, n) {
            return Err(PyValueError::new_err(format!(
                "barra_list[{}] shape {:?} != factor {:?}",
                i,
                b.as_array().dim(),
                (t, n)
            )));
        }
    }

    let fv = factor.as_array().to_owned();
    let ind3 = industry.as_array().to_owned();
    let rest = restrict.as_array().to_owned();
    let barra: Vec<Array2<f64>> = barra_list.iter().map(|b| b.as_array().to_owned()).collect();
    let ind_base: Vec<Array2<f64>> =
        ind_base_list.iter().map(|b| b.as_array().to_owned()).collect();

    let output = py.allow_threads(move || -> Result<Array2<f64>, String> {
        // 1. factor / barra / ind_base 均 rank pct (barra 与 base 对应 load_context 的 _ranked)
        let mut fv_ranked = fv.clone();
        rank_pct_all(&mut fv_ranked);
        let mut barra_ranked: Vec<Array2<f64>> = barra.clone();
        for b in barra_ranked.iter_mut() {
            rank_pct_all(b);
        }
        let mut ind_base_ranked: Vec<Array2<f64>> = ind_base.clone();
        for b in ind_base_ranked.iter_mut() {
            rank_pct_all(b);
        }

        // 2. 行业 OLS 回归填充
        fill_ind_reg(&mut fv_ranked, &ind_base_ranked, &ind3);

        // 3. 行业 2 级中位填充
        let mut fv_filled = fv_ranked.clone();
        let ind1 = ind3.map(|&v| (v / 10000.0).floor());
        // fv_v[np.isnan(ind1)] = np.nan
        for i in 0..(t * n) {
            if ind1.as_slice().unwrap()[i].is_nan() {
                fv_filled.as_slice_mut().unwrap()[i] = f64::NAN;
            }
        }
        let median_source = fv_filled.clone();
        let ind2 = ind3.map(|&v| (v / 100.0).floor());
        fill_by_group_median(&mut fv_filled, &ind2, None, &median_source);
        fill_by_group_median(&mut fv_filled, &ind1, None, &median_source);
        // 全市场级: codes 全 0, valid_mask = ~isnan(ind1)
        let zeros = Array2::<f64>::zeros((t, n));
        let ind1_mask = ind1.map(|&v| if v.is_nan() { 0.0 } else { 1.0 });
        fill_by_group_median(&mut fv_filled, &zeros, Some(&ind1_mask), &median_source);

        // 4. 限制股置空: fv_restricted[restrict==0] = filled
        let mut fv_restricted = Array2::<f64>::from_elem((t, n), f64::NAN);
        for i in 0..(t * n) {
            if rest.as_slice().unwrap()[i] == 0.0 {
                fv_restricted.as_slice_mut().unwrap()[i] = fv_filled.as_slice().unwrap()[i];
            }
        }

        // 5. rank pct
        rank_pct_all(&mut fv_restricted);

        // 6. 残差 (含截距, 10 风格 rank pct)
        let resid = get_residual(&fv_restricted, &barra_ranked);

        // 7. 残差 rank pct
        let mut resid_rank = resid.clone();
        rank_pct_all(&mut resid_rank);
        Ok(resid_rank)
    })
    .map_err(|e| PyValueError::new_err(e))?;

    Ok(output.into_pyarray(py).to_owned())
}

#[pymodule]
fn dev_sandbox_neutralize(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(neutralize_std_pipeline, m)?)?;
    Ok(())
}
