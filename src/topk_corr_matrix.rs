use ndarray::{Array2, ArrayView2};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;

/// 计算每分钟涨幅最大/最小的top_k只股票与所有股票在过去window分钟的相关系数统计量
///
/// 参数说明：
/// ----------
/// select_matrix : numpy.ndarray (n_minutes, n_stocks)
///     用于选择top_k股票的矩阵（如收益率矩阵），第t行第s列是第t分钟第s只股票的值
/// corr_matrix : numpy.ndarray (n_minutes, n_stocks)
///     用于计算相关系数的矩阵（如收益率矩阵或成交量矩阵），shape必须与select_matrix相同
/// top_k : usize
///     每分钟选择的股票数量（如100）
/// window : usize
///     计算相关系数时的回看窗口长度（如10）
/// top_n : usize
///     计算最大top_n个相关系数的均值（如10）
/// select_largest : bool
///     true=选择值最大的top_k只; false=选择值最小的top_k只
///
/// 返回值：
/// -------
/// (numpy.ndarray, numpy.ndarray, numpy.ndarray)
///     三个与输入矩阵相同shape的矩阵:
///     - corr_mean: 相关系数的均值
///     - corr_top_n_mean: 最大top_n个相关系数的均值
///     - corr_skew: 相关系数的偏度
///
/// Python调用示例：
/// ```python
/// import rust_pyfunc
/// import numpy as np
///
/// rets = np.random.randn(240, 5000)
/// corr_mean, corr_top_mean, corr_skew = rust_pyfunc.topk_corr_matrix(rets, rets, 100, 10, 10, True)
/// ```
#[pyfunction]
#[pyo3(signature = (select_matrix, corr_matrix, top_k, window, top_n, select_largest))]
pub fn topk_corr_matrix(
    py: Python,
    select_matrix: PyReadonlyArray2<f64>,
    corr_matrix: PyReadonlyArray2<f64>,
    top_k: usize,
    window: usize,
    top_n: usize,
    select_largest: bool,
) -> PyResult<(Py<PyArray2<f64>>, Py<PyArray2<f64>>, Py<PyArray2<f64>>)> {
    let sel = select_matrix.as_array();
    let corr = corr_matrix.as_array();

    let (n_rows, n_cols) = (sel.nrows(), sel.ncols());

    if corr.nrows() != n_rows || corr.ncols() != n_cols {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "select_matrix和corr_matrix的shape必须相同",
        ));
    }

    if top_k == 0 || window < 2 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "top_k必须>0，window必须>=2",
        ));
    }

    let top_n = top_n.min(top_k);

    let mut out_mean = Array2::<f64>::from_elem((n_rows, n_cols), f64::NAN);
    let mut out_topn = Array2::<f64>::from_elem((n_rows, n_cols), f64::NAN);
    let mut out_skew = Array2::<f64>::from_elem((n_rows, n_cols), f64::NAN);

    for t in window.saturating_sub(1)..n_rows {
        let start = if t + 1 >= window { t + 1 - window } else { 0 };

        // 选择第t分钟值最大/最小的top_k只股票
        let selected = select_topk_indices(&sel, t, top_k, select_largest);
        if selected.is_empty() {
            continue;
        }

        // 提取selected股票在[start..=t]窗口内的corr_matrix序列
        let win_len = t + 1 - start;
        let selected_series: Vec<Vec<f64>> = selected
            .iter()
            .map(|&col| {
                (start..=t).map(|r| corr[[r, col]]).collect()
            })
            .collect();

        // 对每只股票计算与selected股票的相关系数
        for s in 0..n_cols {
            let stock_series: Vec<f64> = (start..=t).map(|r| corr[[r, s]]).collect();

            // 计算与每个selected股票的相关系数
            let mut corrs: Vec<f64> = Vec::with_capacity(selected.len());
            for sel_s in &selected_series {
                let c = pearson_corr_nan(&stock_series, sel_s, win_len);
                if !c.is_nan() {
                    corrs.push(c);
                }
            }

            if corrs.is_empty() {
                continue;
            }

            let n = corrs.len() as f64;
            let mean = corrs.iter().sum::<f64>() / n;
            out_mean[[t, s]] = mean;

            // 最大top_n个的均值: 部分排序
            let actual_top_n = top_n.min(corrs.len());
            partial_sort_descending(&mut corrs, actual_top_n);
            out_topn[[t, s]] = corrs[..actual_top_n].iter().sum::<f64>() / actual_top_n as f64;

            // 偏度
            if corrs.len() >= 3 {
                out_skew[[t, s]] = compute_skewness(&corrs, mean);
            }
        }
    }

    Ok((
        out_mean.into_pyarray(py).to_owned(),
        out_topn.into_pyarray(py).to_owned(),
        out_skew.into_pyarray(py).to_owned(),
    ))
}

/// 选择第t行中值最大（或最小）的top_k个列索引（跳过NaN）
fn select_topk_indices(
    matrix: &ArrayView2<f64>,
    t: usize,
    top_k: usize,
    select_largest: bool,
) -> Vec<usize> {
    let n_cols = matrix.ncols();
    let mut indexed: Vec<(usize, f64)> = Vec::with_capacity(n_cols);
    for c in 0..n_cols {
        let v = matrix[[t, c]];
        if !v.is_nan() {
            indexed.push((c, v));
        }
    }

    if indexed.len() <= top_k {
        return indexed.iter().map(|&(i, _)| i).collect();
    }

    // 部分排序获取前top_k
    if select_largest {
        indexed.select_nth_unstable_by(top_k - 1, |a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });
    } else {
        indexed.select_nth_unstable_by(top_k - 1, |a, b| {
            a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    indexed[..top_k].iter().map(|&(i, _)| i).collect()
}

/// 计算两个序列的Pearson相关系数（跳过任一为NaN的位置）
#[inline]
fn pearson_corr_nan(x: &[f64], y: &[f64], _len: usize) -> f64 {
    let mut mean_x = 0.0;
    let mut mean_y = 0.0;
    let mut m2_x = 0.0;
    let mut m2_y = 0.0;
    let mut cov = 0.0;
    let mut count = 0u64;

    for i in 0..x.len() {
        let vx = x[i];
        let vy = y[i];
        if vx.is_nan() || vy.is_nan() {
            continue;
        }
        count += 1;
        let dx = vx - mean_x;
        let dy = vy - mean_y;
        mean_x += dx / count as f64;
        mean_y += dy / count as f64;
        let new_dx = vx - mean_x;
        let new_dy = vy - mean_y;
        m2_x += dx * new_dx;
        m2_y += dy * new_dy;
        cov += dx * new_dy;
    }

    if count < 2 {
        return f64::NAN;
    }

    let denom = (m2_x * m2_y).sqrt();
    if denom <= f64::EPSILON {
        return f64::NAN;
    }
    (cov / denom).clamp(-1.0, 1.0)
}

/// 部分排序：将最大的n个元素放到前面
fn partial_sort_descending(arr: &mut [f64], n: usize) {
    if n >= arr.len() {
        arr.sort_unstable_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        return;
    }
    arr.select_nth_unstable_by(n - 1, |a, b| {
        b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal)
    });
    // 前n个已经是最大的，但不一定排序
}

/// 计算偏度 (Fisher's skewness)
#[inline]
fn compute_skewness(data: &[f64], mean: f64) -> f64 {
    let n = data.len() as f64;
    let m2: f64 = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n;
    let m3: f64 = data.iter().map(|&x| (x - mean).powi(3)).sum::<f64>() / n;

    if m2 <= f64::EPSILON {
        return f64::NAN;
    }
    let std = m2.sqrt();
    m3 / std.powi(3)
}
