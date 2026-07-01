//! follow2026 回测专用：对已 rank+rolling 的因子 block 同时做 raw 和 neu gap1 回测。
//!
//! 函数 `neutralize_and_backtest_gap1` 接收已经过 rank + rolling 处理的因子 block（3D），
//! 内部不做 re-rank。流程：
//!   1. raw gap1 回测（不做中性化）
//!   2. 用 style_cube 做中性化
//!   3. neu gap1 回测
//! 返回 raw / neu 两套 summary 和 ic 时序。

use numpy::{
    IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::tail_v2_backtest_block::backtest_block_f32_with_parallel;
use crate::tail_v2_block_neutralizer::neutralize_block_f32_out_with_parallel;

/// 对已 rank+rolling 的因子 block 做 raw + neu gap1 回测。
///
/// 输入因子 block 已经过 rank + rolling 处理，本函数 **不做** re-rank。
/// `rank_before` 参数仅控制中性化步骤内部是否对截面因子值做 rank（属于中性化流程，
/// 与因子预处理阶段的 rank+rolling 无关）。
///
/// # 参数
/// - `factor_block`: [n_dates, n_stocks, n_factors] f32，已 rank+rolled
/// - `style_cube`:   [n_dates, n_stocks, n_style] f32，Barra 风格因子（用于中性化）
/// - `ret_array`:    [n_dates, n_stocks] f32，日收益率
/// - `ret_sum_array`:[n_dates, n_stocks] f32，累计收益率
/// - `restrict_array`:[n_dates, n_stocks] f32，可交易掩码（0=可交易）
/// - `index_ret`:    [n_dates] f32，指数日收益率
/// - `gap`:          持仓间隔（默认 1）
/// - `portf_num`:    分组数（默认 10）
/// - `rank_before`:  中性化前是否对因子做截面 rank（默认 true）
/// - `min_valid`:    每天最少有效股票数（默认 12）
///
/// # 返回
/// (raw_summary, raw_ic, neu_summary, neu_ic)
/// - summary: [n_factors, 10]
///   - [0] ic_mean, [1] ir, [2] ann_return, [3] sharpe, [4] max_dd,
///   - [5] date_size, [6] coverage, [7] hedge_annualized_return, [8] hedge_sharpe, [9] hedge_max_dd
/// - ic: [n_ic_dates, n_factors]
#[pyfunction]
#[pyo3(signature = (factor_block, style_cube, ret_array, ret_sum_array, restrict_array, index_ret, gap=1, portf_num=10, rank_before=true, min_valid=12))]
#[allow(clippy::type_complexity)]
pub fn neutralize_and_backtest_gap1<'py>(
    py: Python<'py>,
    factor_block: PyReadonlyArray3<'py, f32>,
    style_cube: PyReadonlyArray3<'py, f32>,
    ret_array: PyReadonlyArray2<'py, f32>,
    ret_sum_array: PyReadonlyArray2<'py, f32>,
    restrict_array: PyReadonlyArray2<'py, f32>,
    index_ret: PyReadonlyArray1<'py, f32>,
    gap: usize,
    portf_num: usize,
    rank_before: bool,
    min_valid: usize,
) -> PyResult<(
    Py<PyArray2<f64>>,
    Py<PyArray2<f64>>,
    Py<PyArray2<f64>>,
    Py<PyArray2<f64>>,
)> {
    let factor = factor_block.as_array();
    let style = style_cube.as_array();
    let ret = ret_array.as_array();
    let ret_sum = ret_sum_array.as_array();
    let restrict = restrict_array.as_array();
    let index = index_ret.as_array();

    let (raw_summary, raw_ic, neu_summary, neu_ic) = py.allow_threads(|| {
        // 1) raw gap1 回测（不做中性化）
        let (raw_summary, raw_ic) = backtest_block_f32_with_parallel(
            factor, ret, ret_sum, restrict, index, gap, portf_num, true,
        )
        .map_err(PyValueError::new_err)?;

        // 2) 中性化（style_cube 对 factor_block 做回归取残差）
        let neu_block = neutralize_block_f32_out_with_parallel(
            style, factor, rank_before, min_valid, true,
        )
        .map_err(PyValueError::new_err)?;

        // 3) neu gap1 回测
        let (neu_summary, neu_ic) = backtest_block_f32_with_parallel(
            neu_block.view(), ret, ret_sum, restrict, index, gap, portf_num, true,
        )
        .map_err(PyValueError::new_err)?;

        Ok::<_, pyo3::PyErr>((raw_summary, raw_ic, neu_summary, neu_ic))
    })?;

    Ok((
        raw_summary.into_pyarray(py).to_owned(),
        raw_ic.into_pyarray(py).to_owned(),
        neu_summary.into_pyarray(py).to_owned(),
        neu_ic.into_pyarray(py).to_owned(),
    ))
}
