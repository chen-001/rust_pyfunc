//! 回测预处理：横截面 rank + 缺失值中位数填充 + rolling 统计 block。
//!
//! 与 `tail_v3_rank_roll_block_f32` 的唯一区别是中间多了一步「缺失 rank 填充」：
//!   1. 按日期做横截面**平均名次** rank（`rank_axis1_average_f32_serial`）；
//!   2. 把「当天 Restrict==0（可交易）但因子缺值」的股票的 rank 填成
//!      `(当日有效只数 + 1) / 2`；停牌/涨跌停/未上市保持 NaN；
//!   3. **不再二次 rank**，直接在填好的 rank 上做 rolling mean/max/min/std/autocorr。
//!
//! 输出 slot 布局（每个窗口固定 5 位，纯加法：前 4 位的位置与取值不变）：
//!   slot 0                    = 填好的 rank 矩阵本身
//!   slot 1 + 5*i + 0/1/2/3/4  = 第 i 个窗口的 mean / max / min / std / autocorr
//!
//! 这正是 tail_v5 引擎 v7 路径（`process_v7_variant`）的预处理顺序，本函数供
//! fulltest 侧复现同一口径使用，使 fulltest 的中性化 IC 与引擎一致。

use ndarray::{Array2, Array3};
use numpy::{IntoPyArray, PyArray3, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::tail_v2_rank_roll_factor::rolling_stats_f32_serial;
use crate::tail_v5_pipeline::rank_and_fill_missing_cross_sectional_median;

/// 滚动自相关：在 rank + 缺失填充后的矩阵上，对每个日期取长度 `window` 的滚动窗口，
/// 算该列与其滞后一期（上一行）的 Pearson 相关系数。
///
/// 与旧框架 `tail_whatever.autocorr_df`（即 `son.corrwith(son.shift(1))`）同一口径：
///   · 窗口是 [row-window+1, row] 共 window 行，与 mean/max/min/std 的窗口口径一致；
///   · 窗口内相邻行两两配对，共 window-1 对；
///   · 只保留两侧都非 NaN 的配对（成对完整观测）；
///   · 有效配对少于 2 对、或任一侧中心化平方和为 0 时输出 NaN；
///   · 第 window-1 行之前没有完整窗口，输出 NaN。
///
/// 累计量用 f64，避免 rank 值较大（上千）时中心化平方和相减丢有效数字。
pub(crate) fn rolling_autocorr_f32_serial(ranked: &Array2<f32>, window: usize) -> Array2<f32> {
    let n_rows = ranked.nrows();
    let n_cols = ranked.ncols();
    let mut out = Array2::<f32>::from_elem((n_rows, n_cols), f32::NAN);
    if window < 2 {
        return out;
    }
    let mut buf = vec![0.0f64; window];
    for col_idx in 0..n_cols {
        for row_idx in (window - 1)..n_rows {
            let start = row_idx + 1 - window;
            for (offset, slot) in buf.iter_mut().enumerate() {
                *slot = ranked[[start + offset, col_idx]] as f64;
            }
            let mut count = 0usize;
            let mut sum_x = 0.0f64;
            let mut sum_y = 0.0f64;
            for t in 0..(window - 1) {
                let x = buf[t + 1];
                let y = buf[t];
                if !x.is_nan() && !y.is_nan() {
                    count += 1;
                    sum_x += x;
                    sum_y += y;
                }
            }
            if count < 2 {
                continue;
            }
            let mean_x = sum_x / count as f64;
            let mean_y = sum_y / count as f64;
            let mut sxx = 0.0f64;
            let mut syy = 0.0f64;
            let mut sxy = 0.0f64;
            for t in 0..(window - 1) {
                let x = buf[t + 1];
                let y = buf[t];
                if !x.is_nan() && !y.is_nan() {
                    let dx = x - mean_x;
                    let dy = y - mean_y;
                    sxx += dx * dx;
                    syy += dy * dy;
                    sxy += dx * dy;
                }
            }
            if sxx > 0.0 && syy > 0.0 {
                out[[row_idx, col_idx]] = (sxy / (sxx * syy).sqrt()) as f32;
            }
        }
    }
    out
}

fn rank_fill_roll_block_f32(
    data: &Array2<f32>,
    restrict: &Array2<f32>,
    windows: &[usize],
) -> Result<Array3<f32>, String> {
    if data.dim() != restrict.dim() {
        return Err(format!(
            "data 与 restrict 形状不一致: {:?} vs {:?}",
            data.dim(),
            restrict.dim()
        ));
    }
    let ranked = rank_and_fill_missing_cross_sectional_median(data, restrict);
    let mut arrays = vec![ranked.clone()];
    for &window in windows {
        if window == 0 {
            return Err("window 必须大于 0".to_string());
        }
        let min_periods = std::cmp::max(1, window / 2);
        let (mean, max, min, std) = rolling_stats_f32_serial(&ranked, window, min_periods);
        let autocorr = rolling_autocorr_f32_serial(&ranked, window);
        arrays.push(mean);
        arrays.push(max);
        arrays.push(min);
        arrays.push(std);
        arrays.push(autocorr);
    }

    let (n_rows, n_cols) = ranked.dim();
    let n_slots = arrays.len();
    let mut block = Array3::<f32>::from_elem((n_rows, n_cols, n_slots), f32::NAN);
    for (slot_idx, array) in arrays.into_iter().enumerate() {
        for row_idx in 0..n_rows {
            for col_idx in 0..n_cols {
                block[[row_idx, col_idx, slot_idx]] = array[[row_idx, col_idx]];
            }
        }
    }
    Ok(block)
}

#[pyfunction]
#[pyo3(signature = (data, restrict, windows))]
pub fn tail_v5_rank_fill_roll_block_f32<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, f32>,
    restrict: PyReadonlyArray2<'py, f32>,
    windows: Vec<usize>,
) -> PyResult<Py<PyArray3<f32>>> {
    let data_view = data.as_array();
    let restrict_view = restrict.as_array();
    let data_owned =
        Array2::<f32>::from_shape_vec(data_view.dim(), data_view.iter().copied().collect())
            .map_err(|_| PyValueError::new_err("data 形状无效"))?;
    let restrict_owned = Array2::<f32>::from_shape_vec(
        restrict_view.dim(),
        restrict_view.iter().copied().collect(),
    )
    .map_err(|_| PyValueError::new_err("restrict 形状无效"))?;

    let output = py.allow_threads(|| {
        rank_fill_roll_block_f32(&data_owned, &restrict_owned, &windows)
            .map_err(PyValueError::new_err)
    })?;

    Ok(output.into_pyarray(py).to_owned())
}
