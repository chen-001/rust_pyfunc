//! 回测预处理：横截面 rank + 缺失值中位数填充 + rolling 统计 block。
//!
//! 与 `tail_v3_rank_roll_block_f32` 的唯一区别是中间多了一步「缺失 rank 填充」：
//!   1. 按日期做横截面**平均名次** rank（`rank_axis1_average_f32_serial`）；
//!   2. 把「当天 Restrict==0（可交易）但因子缺值」的股票的 rank 填成
//!      `(当日有效只数 + 1) / 2`；停牌/涨跌停/未上市保持 NaN；
//!   3. **不再二次 rank**，直接在填好的 rank 上做 rolling mean/max/min/std。
//!
//! 这正是 tail_v5 引擎 v7 路径（`process_v7_variant`）的预处理顺序，本函数供
//! fulltest 侧复现同一口径使用，使 fulltest 的中性化 IC 与引擎一致。

use ndarray::{Array2, Array3};
use numpy::{IntoPyArray, PyArray3, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::tail_v2_rank_roll_factor::rolling_stats_f32_serial;
use crate::tail_v5_pipeline::rank_and_fill_missing_cross_sectional_median;

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
        arrays.push(mean);
        arrays.push(max);
        arrays.push(min);
        arrays.push(std);
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
