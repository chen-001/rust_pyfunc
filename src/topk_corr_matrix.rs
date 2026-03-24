use ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;

/// 计算每分钟涨幅最大/最小的指定排名范围内股票与所有股票在过去window分钟的相关系数统计量
///
/// 参数说明：
/// ----------
/// select_matrix : numpy.ndarray (n_minutes, n_stocks)
///     用于选择股票的矩阵（如收益率矩阵），第t行第s列是第t分钟第s只股票的值
/// corr_matrix : numpy.ndarray (n_minutes, n_stocks)
///     用于计算相关系数的矩阵（如收益率矩阵或成交量矩阵），shape必须与select_matrix相同
/// rank_start : usize
///     排名起始位置（从0开始，包含），如0表示第1名
/// rank_end : usize
///     排名结束位置（不包含），如100表示取第1~100名
/// window : usize
///     计算相关系数时的回看窗口长度（如10）
/// top_n : usize
///     计算最大top_n个相关系数的均值（如10）
/// select_largest : bool
///     true=选择值最大的; false=选择值最小的
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
/// # 选最大的第1~100名（等同于之前的top_k=100）
/// corr_mean, corr_top_mean, corr_skew = rust_pyfunc.topk_corr_matrix(rets, rets, 0, 100, 10, 10, True)
/// # 选最大的第101~200名
/// corr_mean, corr_top_mean, corr_skew = rust_pyfunc.topk_corr_matrix(rets, rets, 100, 200, 10, 10, True)
/// ```
#[pyfunction]
#[pyo3(signature = (select_matrix, corr_matrix, rank_start, rank_end, window, top_n, select_largest))]
pub fn topk_corr_matrix(
    py: Python,
    select_matrix: PyReadonlyArray2<f64>,
    corr_matrix: PyReadonlyArray2<f64>,
    rank_start: usize,
    rank_end: usize,
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

    if rank_start >= rank_end || window < 2 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "需要rank_start < rank_end，且window必须>=2",
        ));
    }

    let range_size = rank_end - rank_start;

    // 将corr_matrix转为行主序连续存储，提升缓存命中率
    let corr_data: Vec<f64> = if corr.is_standard_layout() {
        corr.as_slice().unwrap().to_vec()
    } else {
        let mut buf = vec![0.0f64; n_rows * n_cols];
        for r in 0..n_rows {
            for c in 0..n_cols {
                buf[r * n_cols + c] = corr[[r, c]];
            }
        }
        buf
    };

    // 将sel也转为连续存储
    let sel_data: Vec<f64> = if sel.is_standard_layout() {
        sel.as_slice().unwrap().to_vec()
    } else {
        let mut buf = vec![0.0f64; n_rows * n_cols];
        for r in 0..n_rows {
            for c in 0..n_cols {
                buf[r * n_cols + c] = sel[[r, c]];
            }
        }
        buf
    };

    let mut out_mean = Array2::<f64>::from_elem((n_rows, n_cols), f64::NAN);
    let mut out_topn = Array2::<f64>::from_elem((n_rows, n_cols), f64::NAN);
    let mut out_skew = Array2::<f64>::from_elem((n_rows, n_cols), f64::NAN);

    // 预分配工作缓冲区（避免每次循环内重新分配）
    let mut indexed_buf: Vec<(usize, f64)> = Vec::with_capacity(n_cols);
    let mut corrs_buf: Vec<f64> = Vec::with_capacity(range_size);

    // 预分配selected股票的预计算统计量缓冲区
    let mut sel_sum: Vec<f64> = vec![0.0; range_size];
    let mut sel_sum_sq: Vec<f64> = vec![0.0; range_size];
    let mut sel_valid_count: Vec<u32> = vec![0; range_size];
    // 存储selected股票的窗口数据（列主序：sel_win_data[k * window + w]）
    let mut sel_win_data: Vec<f64> = vec![0.0; range_size * window];
    let mut selected_indices: Vec<usize> = Vec::with_capacity(range_size);

    for t in (window - 1)..n_rows {
        let start = t + 1 - window;
        let win_len = window; // = t + 1 - start

        // 选择第t分钟值最大/最小的top_k只股票
        indexed_buf.clear();
        let sel_row = &sel_data[t * n_cols..(t + 1) * n_cols];
        for c in 0..n_cols {
            let v = sel_row[c];
            if !v.is_nan() {
                indexed_buf.push((c, v));
            }
        }

        if indexed_buf.is_empty() {
            continue;
        }

        let n_valid = indexed_buf.len();
        // 实际可用的排名范围
        let actual_start = rank_start.min(n_valid);
        let actual_end = rank_end.min(n_valid);
        if actual_start >= actual_end {
            continue;
        }

        // 先用select_nth_unstable_by将前rank_end个元素放到前面
        if actual_end < n_valid {
            if select_largest {
                indexed_buf.select_nth_unstable_by(actual_end - 1, |a, b| {
                    b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                });
            } else {
                indexed_buf.select_nth_unstable_by(actual_end - 1, |a, b| {
                    a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
                });
            }
        }
        // 再在前actual_end个元素中，将前actual_start个放到最前面
        if actual_start > 0 && actual_start < actual_end {
            if select_largest {
                indexed_buf[..actual_end].select_nth_unstable_by(actual_start - 1, |a, b| {
                    b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                });
            } else {
                indexed_buf[..actual_end].select_nth_unstable_by(actual_start - 1, |a, b| {
                    a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
                });
            }
        }

        // 提取selected索引（rank_start..rank_end范围内的元素）
        let actual_k = actual_end - actual_start;
        selected_indices.clear();
        for i in actual_start..actual_end {
            selected_indices.push(indexed_buf[i].0);
        }

        // 预计算selected股票的窗口数据和统计量（均值、平方和）
        let mut valid_sel_count = 0usize;
        for (ki, &col) in selected_indices.iter().enumerate() {
            let mut sum = 0.0f64;
            let mut sum_sq = 0.0f64;
            let mut cnt = 0u32;
            for w in 0..win_len {
                let v = corr_data[(start + w) * n_cols + col];
                sel_win_data[ki * window + w] = v;
                if !v.is_nan() {
                    sum += v;
                    sum_sq += v * v;
                    cnt += 1;
                }
            }
            sel_sum[ki] = sum;
            sel_sum_sq[ki] = sum_sq;
            sel_valid_count[ki] = cnt;
            if cnt >= 2 {
                valid_sel_count += 1;
            }
        }

        if valid_sel_count == 0 {
            continue;
        }

        // 对每只股票计算与selected股票的相关系数
        for s in 0..n_cols {
            // 提取股票s在窗口内的数据，同时计算统计量
            let base = start * n_cols + s;
            let mut s_sum = 0.0f64;
            let mut s_sum_sq = 0.0f64;
            let mut s_cnt = 0u32;
            let mut has_nan = false;

            for w in 0..win_len {
                let v = corr_data[base + w * n_cols];
                if v.is_nan() {
                    has_nan = true;
                    break;
                }
                s_sum += v;
                s_sum_sq += v * v;
                s_cnt += 1;
            }

            if !has_nan && s_cnt == win_len as u32 {
                // 快速路径：股票s无NaN
                let s_mean = s_sum / win_len as f64;
                let s_var = s_sum_sq - s_sum * s_mean; // = sum((x-mean)^2) * (不除n)

                if s_var <= f64::EPSILON {
                    continue;
                }

                corrs_buf.clear();
                for ki in 0..actual_k {
                    if sel_valid_count[ki] < 2 {
                        continue;
                    }

                    // 检查selected股票是否也全部无NaN（大部分情况）
                    if sel_valid_count[ki] == win_len as u32 {
                        // 两者均无NaN: 直接计算
                        let y_sum = sel_sum[ki];
                        let y_sum_sq = sel_sum_sq[ki];
                        let y_mean = y_sum / win_len as f64;
                        let y_var = y_sum_sq - y_sum * y_mean;

                        if y_var <= f64::EPSILON {
                            continue;
                        }

                        let mut cross = 0.0f64;
                        let sel_base = ki * window;
                        for w in 0..win_len {
                            cross += corr_data[base + w * n_cols] * sel_win_data[sel_base + w];
                        }
                        // corr = (cross - n*mean_x*mean_y) / sqrt(var_x * var_y)
                        let cov = cross - win_len as f64 * s_mean * y_mean;
                        let denom = (s_var * y_var).sqrt();
                        let c = (cov / denom).clamp(-1.0, 1.0);
                        corrs_buf.push(c);
                    } else {
                        // selected有NaN: Welford逐点计算
                        let c = pearson_corr_with_nan_y(
                            &corr_data,
                            base,
                            n_cols,
                            &sel_win_data[ki * window..ki * window + win_len],
                            win_len,
                        );
                        if !c.is_nan() {
                            corrs_buf.push(c);
                        }
                    }
                }

                if corrs_buf.is_empty() {
                    continue;
                }

                compute_stats(
                    &mut corrs_buf,
                    top_n,
                    t,
                    s,
                    &mut out_mean,
                    &mut out_topn,
                    &mut out_skew,
                );
            } else {
                // 慢速路径：股票s有NaN
                corrs_buf.clear();
                for ki in 0..actual_k {
                    if sel_valid_count[ki] < 2 {
                        continue;
                    }
                    let c = pearson_corr_both_nan(
                        &corr_data,
                        base,
                        n_cols,
                        &sel_win_data[ki * window..ki * window + win_len],
                        win_len,
                    );
                    if !c.is_nan() {
                        corrs_buf.push(c);
                    }
                }

                if corrs_buf.is_empty() {
                    continue;
                }

                compute_stats(
                    &mut corrs_buf,
                    top_n,
                    t,
                    s,
                    &mut out_mean,
                    &mut out_topn,
                    &mut out_skew,
                );
            }
        }
    }

    Ok((
        out_mean.into_pyarray(py).to_owned(),
        out_topn.into_pyarray(py).to_owned(),
        out_skew.into_pyarray(py).to_owned(),
    ))
}

/// 计算相关系数的统计量并写入输出矩阵
#[inline(always)]
fn compute_stats(
    corrs: &mut Vec<f64>,
    top_n: usize,
    t: usize,
    s: usize,
    out_mean: &mut Array2<f64>,
    out_topn: &mut Array2<f64>,
    out_skew: &mut Array2<f64>,
) {
    let n = corrs.len() as f64;
    let mean = corrs.iter().sum::<f64>() / n;
    out_mean[[t, s]] = mean;

    // 最大top_n个的均值
    let actual_top_n = top_n.min(corrs.len());
    corrs.select_nth_unstable_by(actual_top_n - 1, |a, b| {
        b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal)
    });
    out_topn[[t, s]] = corrs[..actual_top_n].iter().sum::<f64>() / actual_top_n as f64;

    // 偏度
    if corrs.len() >= 3 {
        let mut m2 = 0.0f64;
        let mut m3 = 0.0f64;
        for &c in corrs.iter() {
            let d = c - mean;
            let d2 = d * d;
            m2 += d2;
            m3 += d2 * d;
        }
        m2 /= n;
        m3 /= n;
        if m2 > f64::EPSILON {
            let std = m2.sqrt();
            out_skew[[t, s]] = m3 / (std * std * std);
        }
    }
}

/// x无NaN，y可能有NaN的相关系数
#[inline]
fn pearson_corr_with_nan_y(
    data: &[f64],
    x_base: usize,
    x_stride: usize,
    y: &[f64],
    len: usize,
) -> f64 {
    let mut sum_x = 0.0f64;
    let mut sum_y = 0.0f64;
    let mut sum_xx = 0.0f64;
    let mut sum_yy = 0.0f64;
    let mut sum_xy = 0.0f64;
    let mut count = 0u32;

    for w in 0..len {
        let vy = y[w];
        if vy.is_nan() {
            continue;
        }
        let vx = data[x_base + w * x_stride];
        sum_x += vx;
        sum_y += vy;
        sum_xx += vx * vx;
        sum_yy += vy * vy;
        sum_xy += vx * vy;
        count += 1;
    }

    if count < 2 {
        return f64::NAN;
    }

    let n = count as f64;
    let cov = sum_xy - sum_x * sum_y / n;
    let var_x = sum_xx - sum_x * sum_x / n;
    let var_y = sum_yy - sum_y * sum_y / n;
    let denom = (var_x * var_y).sqrt();

    if denom <= f64::EPSILON {
        return f64::NAN;
    }
    (cov / denom).clamp(-1.0, 1.0)
}

/// 两者都可能有NaN的相关系数
#[inline]
fn pearson_corr_both_nan(
    data: &[f64],
    x_base: usize,
    x_stride: usize,
    y: &[f64],
    len: usize,
) -> f64 {
    let mut sum_x = 0.0f64;
    let mut sum_y = 0.0f64;
    let mut sum_xx = 0.0f64;
    let mut sum_yy = 0.0f64;
    let mut sum_xy = 0.0f64;
    let mut count = 0u32;

    for w in 0..len {
        let vx = data[x_base + w * x_stride];
        let vy = y[w];
        if vx.is_nan() || vy.is_nan() {
            continue;
        }
        sum_x += vx;
        sum_y += vy;
        sum_xx += vx * vx;
        sum_yy += vy * vy;
        sum_xy += vx * vy;
        count += 1;
    }

    if count < 2 {
        return f64::NAN;
    }

    let n = count as f64;
    let cov = sum_xy - sum_x * sum_y / n;
    let var_x = sum_xx - sum_x * sum_x / n;
    let var_y = sum_yy - sum_y * sum_y / n;
    let denom = (var_x * var_y).sqrt();

    if denom <= f64::EPSILON {
        return f64::NAN;
    }
    (cov / denom).clamp(-1.0, 1.0)
}
