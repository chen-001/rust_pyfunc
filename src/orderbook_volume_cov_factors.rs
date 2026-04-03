use nalgebra::DMatrix;
use ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;

/// 从大矩阵中提取子矩阵（按行索引和列索引）
/// DMatrix::from_vec 期望 column-major，所以按列优先收集
fn extract_submatrix(mat: &DMatrix<f64>, rows: &[usize], cols: &[usize]) -> DMatrix<f64> {
    let nr = rows.len();
    let nc = cols.len();
    let mut data = Vec::with_capacity(nr * nc);
    for &c in cols {
        for &r in rows {
            data.push(mat[(r, c)]);
        }
    }
    DMatrix::from_vec(nr, nc, data)
}

/// 计算对数行列式，使用 Cholesky 分解
fn log_det_cholesky(mat: &DMatrix<f64>) -> Option<f64> {
    let chol = mat.clone().cholesky()?;
    let log_det: f64 = 2.0 * chol.l().diagonal().iter().map(|x| x.ln()).sum::<f64>();
    if log_det.is_finite() {
        Some(log_det)
    } else {
        None
    }
}

/// Schur 补因子计算: log(det_Y) - log(det_schur)
fn schur_factor(cov: &DMatrix<f64>, x_idx: &[usize], y_idx: &[usize]) -> Option<f64> {
    let sigma_xx = extract_submatrix(cov, x_idx, x_idx);
    let sigma_yy = extract_submatrix(cov, y_idx, y_idx);
    let sigma_xy = extract_submatrix(cov, x_idx, y_idx);

    let log_det_y = log_det_cholesky(&sigma_yy)?;
    if log_det_y < (-23.0) {
        return None;
    }

    let xx_chol = sigma_xx.clone().cholesky()?;
    let log_det_xx: f64 = 2.0 * xx_chol.l().diagonal().iter().map(|x| x.ln()).sum::<f64>();
    if !log_det_xx.is_finite() || log_det_xx < (-23.0) {
        return None;
    }

    let solved = xx_chol.solve(&sigma_xy);
    let schur = &sigma_yy - &sigma_xy.transpose() * &solved;

    let log_det_schur = log_det_cholesky(&schur)?;
    if log_det_schur < (-23.0) {
        return None;
    }

    let _ = log_det_xx;
    Some(log_det_y - log_det_schur)
}

/// 计算特征值的对数和（过滤掉过小的特征值）
fn log_sum_eigenvalues(eigenvalues: &[f64]) -> f64 {
    eigenvalues
        .iter()
        .filter(|&&x| x > 1e-10)
        .map(|x| x.ln())
        .sum()
}

/// 提取特征值到 Vec
fn extract_eigenvalues(mat: &DMatrix<f64>) -> Vec<f64> {
    mat.clone()
        .symmetric_eigen()
        .eigenvalues
        .iter()
        .copied()
        .collect()
}

/// 使用预计算的全矩阵特征值，计算特征值对数和差
fn eigval_diff_precomputed(eig_full: &[f64], cov_split: &DMatrix<f64>) -> Option<f64> {
    let sum_full = log_sum_eigenvalues(eig_full);
    let eig_split = extract_eigenvalues(cov_split);
    let sum_split = log_sum_eigenvalues(&eig_split);

    if !sum_full.is_finite() || !sum_split.is_finite() {
        return None;
    }
    Some(sum_split - sum_full)
}

/// 构建 tridiagonal 掩码
fn build_adjacent_mask(d: usize, cov: &DMatrix<f64>) -> DMatrix<f64> {
    let mut m = DMatrix::zeros(d, d);
    for i in 0..d {
        m[(i, i)] = cov[(i, i)];
        if i + 1 < d {
            m[(i, i + 1)] = cov[(i, i + 1)];
            m[(i + 1, i)] = cov[(i + 1, i)];
        }
    }
    m
}

/// 构建反对角线零掩码矩阵
fn build_anti_diag_mask(n_half: usize, d: usize, cov: &DMatrix<f64>) -> DMatrix<f64> {
    let mut m = cov.clone();
    for j in 0..n_half {
        let k = d - 1 - j;
        m[(j, k)] = 0.0;
        m[(k, j)] = 0.0;
    }
    m
}

/// 将两组索引之间的交叉块置零（含对称）
fn zero_cross_blocks(mat: &mut DMatrix<f64>, rows: &[usize], cols: &[usize]) {
    for &i in rows {
        for &j in cols {
            mat[(i, j)] = 0.0;
            mat[(j, i)] = 0.0;
        }
    }
}

/// 从20x20协方差矩阵计算全部15个因子
/// 优化：预计算共享的 eigendecomposition，避免重复计算
fn compute_all_factors(cov20: &DMatrix<f64>) -> [Option<f64>; 15] {
    let mut results = [None; 15];

    // 预计算共享的 eigendecomposition（避免 eig(cov20) 被重复调用4次）
    let eig20_vals = extract_eigenvalues(cov20);

    // === 20x20 Schur 因子 (3个) ===
    results[0] = schur_factor(
        cov20,
        &(0..10).collect::<Vec<_>>(),
        &(10..20).collect::<Vec<_>>(),
    );
    results[1] = schur_factor(
        cov20,
        &(0..20).step_by(2).collect::<Vec<_>>(),
        &(1..20).step_by(2).collect::<Vec<_>>(),
    );
    results[2] = schur_factor(
        cov20,
        &(0..5).chain(10..15).collect::<Vec<_>>(),
        &(5..10).chain(15..20).collect::<Vec<_>>(),
    );

    // === 20x20 Eigval 因子 (4个，共享 eig20_vals) ===
    let bid_low: Vec<usize> = (0..5).collect();
    let ask_high: Vec<usize> = (15..20).collect();
    let bid_high: Vec<usize> = (5..10).collect();
    let ask_low: Vec<usize> = (10..15).collect();

    // 4. bla_cpl
    {
        let mut cov_split = cov20.clone();
        zero_cross_blocks(&mut cov_split, &bid_high, &ask_low);
        zero_cross_blocks(&mut cov_split, &bid_low, &ask_low);
        zero_cross_blocks(&mut cov_split, &bid_high, &ask_high);
        results[3] = eigval_diff_precomputed(&eig20_vals, &cov_split);
    }

    // 5. bha_cpl
    {
        let mut cov_split = cov20.clone();
        zero_cross_blocks(&mut cov_split, &bid_low, &ask_high);
        zero_cross_blocks(&mut cov_split, &bid_low, &ask_low);
        zero_cross_blocks(&mut cov_split, &bid_high, &ask_high);
        results[4] = eigval_diff_precomputed(&eig20_vals, &cov_split);
    }

    // 6. adj_cpl
    {
        let cov_split = build_adjacent_mask(20, cov20);
        results[5] = eigval_diff_precomputed(&eig20_vals, &cov_split);
    }

    // 7. diag_spl
    {
        let cov_split = build_anti_diag_mask(10, 20, cov20);
        results[6] = eigval_diff_precomputed(&eig20_vals, &cov_split);
    }

    // === 提取 bid/ask 10x10 子矩阵及预计算特征值 ===
    let bid_rows: Vec<usize> = (0..10).collect();
    let cov_bid = extract_submatrix(cov20, &bid_rows, &bid_rows);
    let ask_rows: Vec<usize> = (10..20).collect();
    let cov_ask = extract_submatrix(cov20, &ask_rows, &ask_rows);

    let eig_bid_vals = extract_eigenvalues(&cov_bid);
    let eig_ask_vals = extract_eigenvalues(&cov_ask);

    // === bid 10x10 Schur 因子 (2个) ===
    results[7] = schur_factor(
        &cov_bid,
        &(0..10).step_by(2).collect::<Vec<_>>(),
        &(1..10).step_by(2).collect::<Vec<_>>(),
    );
    results[8] = schur_factor(
        &cov_bid,
        &(0..5).collect::<Vec<_>>(),
        &(5..10).collect::<Vec<_>>(),
    );

    // === bid 10x10 Eigval 因子 (2个，共享 eig_bid_vals) ===
    {
        let cov_split = build_adjacent_mask(10, &cov_bid);
        results[9] = eigval_diff_precomputed(&eig_bid_vals, &cov_split);
    }
    {
        let cov_split = build_anti_diag_mask(5, 10, &cov_bid);
        results[10] = eigval_diff_precomputed(&eig_bid_vals, &cov_split);
    }

    // === ask 10x10 Schur 因子 (2个) ===
    results[11] = schur_factor(
        &cov_ask,
        &(0..10).step_by(2).collect::<Vec<_>>(),
        &(1..10).step_by(2).collect::<Vec<_>>(),
    );
    results[12] = schur_factor(
        &cov_ask,
        &(0..5).collect::<Vec<_>>(),
        &(5..10).collect::<Vec<_>>(),
    );

    // === ask 10x10 Eigval 因子 (2个，共享 eig_ask_vals) ===
    {
        let cov_split = build_adjacent_mask(10, &cov_ask);
        results[13] = eigval_diff_precomputed(&eig_ask_vals, &cov_split);
    }
    {
        let cov_split = build_anti_diag_mask(5, 10, &cov_ask);
        results[14] = eigval_diff_precomputed(&eig_ask_vals, &cov_split);
    }

    results
}

/// 列名生成
fn get_column_names() -> Vec<String> {
    vec![
        "ba_spl".into(),
        "oe_spl".into(),
        "hl_spl".into(),
        "bla_cpl".into(),
        "bha_cpl".into(),
        "adj_cpl".into(),
        "diag_spl".into(),
        "b_oe".into(),
        "b_hl".into(),
        "b_adj".into(),
        "b_diag".into(),
        "a_oe".into(),
        "a_hl".into(),
        "a_adj".into(),
        "a_diag".into(),
        "oe_bma".into(),
        "oe_dabs".into(),
        "hl_bma".into(),
        "hl_dabs".into(),
        "adj_bma".into(),
        "adj_dabs".into(),
        "diag_bma".into(),
        "diag_dabs".into(),
    ]
}

/// 盘口量协方差因子计算
///
/// 对盘口10档买卖量差分数据，通过滑动窗口计算协方差矩阵，
/// 然后对协方差矩阵做不同方式的分割/掩码，得到15个基础因子和8个衍生因子。
///
/// 使用增量协方差优化，每个窗口的更新仅需 O(D²) 而非 O(T·D²)。
#[pyfunction]
pub fn orderbook_volume_cov_factors(
    py: Python<'_>,
    bid_volumes: PyReadonlyArray2<f64>,
    ask_volumes: PyReadonlyArray2<f64>,
    window_size: usize,
) -> PyResult<(Vec<String>, Py<PyArray2<f64>>)> {
    let bid = bid_volumes.as_array();
    let ask = ask_volumes.as_array();
    let (n, ncols_bid) = bid.dim();
    let (_, ncols_ask) = ask.dim();

    if ncols_bid != 10 || ncols_ask != 10 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "bid_volumes 和 ask_volumes 必须各为 10 列",
        ));
    }

    let d = 20;
    let n_windows = if n >= window_size {
        n - window_size + 1
    } else {
        0
    };

    let col_names = get_column_names();
    let ncols = col_names.len();

    if n_windows == 0 {
        let result = PyArray2::<f64>::zeros(py, [0, ncols], false);
        return Ok((col_names, result.into()));
    }

    // 差分数据: diff_data[i] = volumes[i] - volumes[i-1], diff_data[0] = 0
    let mut diff_data = vec![0.0f64; n * d];
    for i in 1..n {
        for j in 0..10 {
            diff_data[i * d + j] = bid[(i, j)] - bid[(i - 1, j)];
            diff_data[i * d + 10 + j] = ask[(i, j)] - ask[(i - 1, j)];
        }
    }

    let mut result_data = vec![f64::NAN; n_windows * ncols];

    // 初始窗口 [0, window_size) 的累积量
    let mut sum_x = vec![0.0f64; d];
    let mut sum_xx = vec![0.0f64; d * d];

    for t in 0..window_size {
        for i in 0..d {
            let vi = diff_data[t * d + i];
            sum_x[i] += vi;
            for j in 0..=i {
                let vj = diff_data[t * d + j];
                sum_xx[i * d + j] += vi * vj;
            }
        }
    }

    let ws_f = window_size as f64;
    let ddof = (window_size - 1) as f64;

    // 预分配协方差矩阵缓冲区（column-major），跨窗口复用
    let mut cov_colmaj = vec![0.0f64; d * d];

    for w in 0..n_windows {
        // 直接构建 column-major 协方差矩阵
        // column-major: cov_colmaj[col * d + row] = cov(row, col)
        let mut has_nan_inf = false;
        for j in 0..d {
            let sj = sum_x[j];
            for i in 0..d {
                let si = sum_x[i];
                // sum_xx 只存下三角 (lo <= hi)
                let (lo, hi) = if i >= j { (j, i) } else { (i, j) };
                let sxx = sum_xx[hi * d + lo];
                let cov_val = (sxx - si * sj / ws_f) / ddof;
                if !cov_val.is_finite() {
                    has_nan_inf = true;
                    break;
                }
                cov_colmaj[j * d + i] = cov_val;
            }
            if has_nan_inf {
                break;
            }
        }

        if !has_nan_inf {
            let cov = DMatrix::from_vec(d, d, cov_colmaj.clone());
            let factors = compute_all_factors(&cov);

            for i in 0..15 {
                if let Some(v) = factors[i] {
                    result_data[w * ncols + i] = v;
                }
            }

            // 衍生因子
            let b_oe = factors[7];
            let a_oe = factors[11];
            let b_hl = factors[8];
            let a_hl = factors[12];
            let b_adj = factors[9];
            let a_adj = factors[13];
            let b_diag = factors[10];
            let a_diag = factors[14];

            if let (Some(b), Some(a)) = (b_oe, a_oe) {
                result_data[w * ncols + 15] = b - a;
                result_data[w * ncols + 16] = (b - a).abs();
            }
            if let (Some(b), Some(a)) = (b_hl, a_hl) {
                result_data[w * ncols + 17] = b - a;
                result_data[w * ncols + 18] = (b - a).abs();
            }
            if let (Some(b), Some(a)) = (b_adj, a_adj) {
                result_data[w * ncols + 19] = b - a;
                result_data[w * ncols + 20] = (b - a).abs();
            }
            if let (Some(b), Some(a)) = (b_diag, a_diag) {
                result_data[w * ncols + 21] = b - a;
                result_data[w * ncols + 22] = (b - a).abs();
            }
        }

        // 增量更新
        if w < n_windows - 1 {
            let old_t = w;
            let new_t = w + window_size;
            for i in 0..d {
                let vi_old = diff_data[old_t * d + i];
                sum_x[i] -= vi_old;
                for j in 0..=i {
                    let vj_old = diff_data[old_t * d + j];
                    sum_xx[i * d + j] -= vi_old * vj_old;
                }
            }
            for i in 0..d {
                let vi_new = diff_data[new_t * d + i];
                sum_x[i] += vi_new;
                for j in 0..=i {
                    let vj_new = diff_data[new_t * d + j];
                    sum_xx[i * d + j] += vi_new * vj_new;
                }
            }
        }
    }

    // row-major data -> ndarray -> PyArray2
    let arr = Array2::from_shape_vec((n_windows, ncols), result_data)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("reshape failed: {}", e)))?;
    let result_array = arr.into_pyarray(py);
    Ok((col_names, result_array.into()))
}
