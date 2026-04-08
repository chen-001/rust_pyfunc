use std::cmp::Ordering;

use ndarray::{Array2, ArrayView1, ArrayView2, ArrayView3};
use numpy::{
    IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

const EPS: f64 = 1e-12;

trait FloatLike: Copy + PartialOrd + Send + Sync {
    fn to_f64(self) -> f64;
    fn is_nan(self) -> bool {
        self.to_f64().is_nan()
    }
}

impl FloatLike for f64 {
    #[inline]
    fn to_f64(self) -> f64 {
        self
    }
}

impl FloatLike for f32 {
    #[inline]
    fn to_f64(self) -> f64 {
        self as f64
    }
}

#[inline]
fn cmp_nan_last<T: FloatLike>(a: T, b: T) -> Ordering {
    match (a.is_nan(), b.is_nan()) {
        (true, true) => Ordering::Equal,
        (true, false) => Ordering::Greater,
        (false, true) => Ordering::Less,
        (false, false) => a.partial_cmp(&b).unwrap_or(Ordering::Equal),
    }
}

fn pearson_corr(x: &[f64], y: &[f64]) -> f64 {
    if x.len() != y.len() || x.len() < 2 {
        return f64::NAN;
    }
    let n = x.len() as f64;
    let mean_x = x.iter().sum::<f64>() / n;
    let mean_y = y.iter().sum::<f64>() / n;
    let mut var_x = 0.0;
    let mut var_y = 0.0;
    let mut cov = 0.0;
    for idx in 0..x.len() {
        let dx = x[idx] - mean_x;
        let dy = y[idx] - mean_y;
        var_x += dx * dx;
        var_y += dy * dy;
        cov += dx * dy;
    }
    if var_x <= EPS || var_y <= EPS {
        return f64::NAN;
    }
    cov / (var_x.sqrt() * var_y.sqrt())
}

fn nanmean(values: &[f64]) -> f64 {
    let mut sum = 0.0;
    let mut count = 0usize;
    for &value in values {
        if !value.is_nan() {
            sum += value;
            count += 1;
        }
    }
    if count == 0 {
        f64::NAN
    } else {
        sum / count as f64
    }
}

fn nanstd(values: &[f64]) -> f64 {
    let mean = nanmean(values);
    if mean.is_nan() {
        return f64::NAN;
    }
    let mut sq_sum = 0.0;
    let mut count = 0usize;
    for &value in values {
        if !value.is_nan() {
            let delta = value - mean;
            sq_sum += delta * delta;
            count += 1;
        }
    }
    if count == 0 {
        f64::NAN
    } else {
        (sq_sum / count as f64).sqrt()
    }
}

fn annualized_sharpe(values: &[f64]) -> f64 {
    let std = nanstd(values);
    if std.is_nan() || std <= EPS {
        return f64::NAN;
    }
    nanmean(values) / std * 250.0_f64.sqrt()
}

fn max_drawdown_from_returns(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    let mut cumulative = 0.0;
    let mut peak = f64::NEG_INFINITY;
    let mut max_dd = 0.0;
    for &value in values {
        if !value.is_nan() {
            cumulative += value;
        }
        if cumulative > peak {
            peak = cumulative;
        }
        let drawdown = peak - cumulative;
        if drawdown > max_dd {
            max_dd = drawdown;
        }
    }
    max_dd
}

fn validate_shapes<T: FloatLike>(
    factor: &ArrayView3<'_, T>,
    ret: &ArrayView2<'_, T>,
    ret_sum: &ArrayView2<'_, T>,
    restrict: &ArrayView2<'_, T>,
    index: &ArrayView1<'_, T>,
    gap: usize,
    portf_num: usize,
) -> PyResult<(usize, usize, usize, usize)> {
    if gap == 0 {
        return Err(PyValueError::new_err("gap 必须大于 0"));
    }
    if portf_num == 0 {
        return Err(PyValueError::new_err("portf_num 必须大于 0"));
    }
    if factor.ndim() != 3 {
        return Err(PyValueError::new_err("factor_block 必须是三维数组"));
    }
    let n_dates = factor.shape()[0];
    let n_stocks = factor.shape()[1];
    let n_factors = factor.shape()[2];
    if ret.shape() != [n_dates, n_stocks]
        || ret_sum.shape() != [n_dates, n_stocks]
        || restrict.shape() != [n_dates, n_stocks]
        || index.len() != n_dates
    {
        return Err(PyValueError::new_err(
            "factor_block, ret_array, ret_sum_array, restrict_array, index_ret 的形状不匹配",
        ));
    }
    Ok((n_dates, n_stocks, n_factors, n_dates / gap))
}

fn precompute_future_orders<T: FloatLike>(
    ret_sum: &ArrayView2<'_, T>,
    gap: usize,
    n_dates: usize,
    n_stocks: usize,
) -> Vec<Vec<u32>> {
    let mut orders = vec![Vec::<u32>::new(); n_dates];
    for date_idx in 0..n_dates {
        if (date_idx + 1) % gap != 0 {
            continue;
        }
        let mut order = (0..n_stocks as u32).collect::<Vec<_>>();
        order.sort_by(|&lhs, &rhs| {
            cmp_nan_last(
                ret_sum[[date_idx, lhs as usize]],
                ret_sum[[date_idx, rhs as usize]],
            )
        });
        orders[date_idx] = order;
    }
    orders
}

fn precompute_trade_masks<T: FloatLike>(
    ret: &ArrayView2<'_, T>,
    restrict: &ArrayView2<'_, T>,
    n_dates: usize,
    n_stocks: usize,
) -> (Vec<u8>, Vec<u8>, Vec<usize>) {
    let mut ret_valid_today = vec![0u8; n_dates * n_stocks];
    let mut hold_open = vec![0u8; n_dates * n_stocks];
    let mut valid_universe_counts = vec![0usize; n_dates];
    for date_idx in 0..n_dates {
        let mut valid_count = 0usize;
        for stock_idx in 0..n_stocks {
            let flat_idx = date_idx * n_stocks + stock_idx;
            let is_open = restrict[[date_idx, stock_idx]].to_f64() == 0.0;
            hold_open[flat_idx] = if is_open { 1 } else { 0 };
            let ret_valid = !ret[[date_idx, stock_idx]].is_nan();
            ret_valid_today[flat_idx] = if ret_valid { 1 } else { 0 };
            if is_open && ret_valid {
                valid_count += 1;
            }
        }
        valid_universe_counts[date_idx] = valid_count;
    }
    (ret_valid_today, hold_open, valid_universe_counts)
}

fn backtest_block_impl<T: FloatLike>(
    factor: ArrayView3<'_, T>,
    ret: ArrayView2<'_, T>,
    ret_sum: ArrayView2<'_, T>,
    restrict: ArrayView2<'_, T>,
    index: ArrayView1<'_, T>,
    gap: usize,
    portf_num: usize,
) -> PyResult<(Array2<f64>, Array2<f64>)> {
    let (n_dates, n_stocks, n_factors, n_ic_dates) =
        validate_shapes(&factor, &ret, &ret_sum, &restrict, &index, gap, portf_num)?;

    let future_orders = precompute_future_orders(&ret_sum, gap, n_dates, n_stocks);
    let (ret_valid_today, hold_open, valid_universe_counts) =
        precompute_trade_masks(&ret, &restrict, n_dates, n_stocks);

    let (summary_rows, ic_columns): (Vec<[f64; 10]>, Vec<Vec<f64>>) = (0..n_factors)
        .into_par_iter()
        .map(|factor_idx| {
            let mut current_hold_date = usize::MAX;
            let mut current_signal_order = Vec::<u32>::with_capacity(n_stocks);
            let mut current_signal_values = Vec::<T>::with_capacity(n_stocks);
            let mut valid_trade = vec![false; n_stocks];
            let mut valid_with_future = vec![false; n_stocks];
            let mut signal_rank_per_stock = vec![f64::NAN; n_stocks];
            let mut paired_signal = Vec::<f64>::with_capacity(n_stocks);
            let mut paired_future = Vec::<f64>::with_capacity(n_stocks);
            let mut group_sum = vec![0.0; portf_num];
            let mut group_count = vec![0usize; portf_num];
            let mut ls_returns = vec![0.0; n_dates];
            let mut hedge_returns = vec![0.0; n_dates];
            let mut coverage_values = vec![f64::NAN; n_dates];
            let mut ic_values = vec![f64::NAN; n_ic_dates];

            let mut ic_sum = 0.0;
            let mut ic_count = 0usize;

            for date_idx in 0..n_dates {
                let hold_date = (date_idx / gap) * gap;
                if hold_date != current_hold_date {
                    current_hold_date = hold_date;
                    current_signal_values.clear();
                    for stock_idx in 0..n_stocks {
                        current_signal_values.push(factor[[hold_date, stock_idx, factor_idx]]);
                    }
                    current_signal_order.clear();
                    current_signal_order.extend(0..n_stocks as u32);
                    current_signal_order.sort_by(|&lhs, &rhs| {
                        cmp_nan_last(
                            current_signal_values[lhs as usize],
                            current_signal_values[rhs as usize],
                        )
                    });
                }

                let ic_slot = if (date_idx + 1) % gap == 0 {
                    Some(date_idx / gap)
                } else {
                    None
                };

                let mut valid_count = 0usize;
                let valid_universe = valid_universe_counts[date_idx];
                for &stock in &current_signal_order {
                    let stock_idx = stock as usize;
                    let signal_value = current_signal_values[stock_idx];
                    if signal_value.is_nan() {
                        break;
                    }
                    let is_valid = hold_open[hold_date * n_stocks + stock_idx] != 0
                        && ret_valid_today[date_idx * n_stocks + stock_idx] != 0;
                    valid_trade[stock_idx] = is_valid;
                    if is_valid {
                        valid_count += 1;
                    }
                }

                if valid_count < portf_num {
                    if let Some(slot) = ic_slot {
                        ic_values[slot] = f64::NAN;
                    }
                    continue;
                }

                group_sum.fill(0.0);
                group_count.fill(0);
                if ic_slot.is_some() {
                    valid_with_future.fill(false);
                    signal_rank_per_stock.fill(f64::NAN);
                }

                let mut seen_valid = 0usize;
                let mut future_valid_count = 0usize;
                let mut pos = 0usize;
                while pos < current_signal_order.len() {
                    let stock_idx = current_signal_order[pos] as usize;
                    let signal_value = current_signal_values[stock_idx];
                    if signal_value.is_nan() {
                        break;
                    }
                    let mut end = pos + 1;
                    while end < current_signal_order.len() {
                        let next_idx = current_signal_order[end] as usize;
                        let next_value = current_signal_values[next_idx];
                        if next_value.is_nan() || signal_value.to_f64() != next_value.to_f64() {
                            break;
                        }
                        end += 1;
                    }

                    let mut valid_in_group = 0usize;
                    for order_pos in pos..end {
                        let member_idx = current_signal_order[order_pos] as usize;
                        if valid_trade[member_idx] {
                            valid_in_group += 1;
                        }
                    }

                    if valid_in_group > 0 {
                        let avg_rank =
                            ((seen_valid + 1 + seen_valid + valid_in_group) as f64) / 2.0;
                        let mut group_id =
                            ((avg_rank / valid_count as f64) * portf_num as f64).ceil() as usize;
                        if group_id < 1 {
                            group_id = 1;
                        }
                        if group_id > portf_num {
                            group_id = portf_num;
                        }
                        let group_slot = group_id - 1;

                        for order_pos in pos..end {
                            let member_idx = current_signal_order[order_pos] as usize;
                            if !valid_trade[member_idx] {
                                continue;
                            }
                            group_sum[group_slot] += ret[[date_idx, member_idx]].to_f64();
                            group_count[group_slot] += 1;
                            if ic_slot.is_some()
                                && !ret_sum[[date_idx, member_idx]].is_nan()
                            {
                                valid_with_future[member_idx] = true;
                                signal_rank_per_stock[member_idx] = avg_rank;
                                future_valid_count += 1;
                            }
                        }
                        seen_valid += valid_in_group;
                    }
                    pos = end;
                }

                if let Some(slot) = ic_slot {
                    paired_signal.clear();
                    paired_future.clear();

                    let ic_value = if future_valid_count >= 2 {
                        let future_order = &future_orders[date_idx];
                        let mut seen_future_valid = 0usize;
                        let mut future_pos = 0usize;
                        while future_pos < future_order.len() {
                            let stock_idx = future_order[future_pos] as usize;
                            let future_value = ret_sum[[date_idx, stock_idx]];
                            if future_value.is_nan() {
                                break;
                            }
                            let mut future_end = future_pos + 1;
                            while future_end < future_order.len() {
                                let next_idx = future_order[future_end] as usize;
                                let next_value = ret_sum[[date_idx, next_idx]];
                                if next_value.is_nan()
                                    || future_value.to_f64() != next_value.to_f64()
                                {
                                    break;
                                }
                                future_end += 1;
                            }

                            let mut valid_in_group = 0usize;
                            for order_pos in future_pos..future_end {
                                let member_idx = future_order[order_pos] as usize;
                                if valid_with_future[member_idx] {
                                    valid_in_group += 1;
                                }
                            }

                            if valid_in_group > 0 {
                                let avg_rank = ((seen_future_valid + 1
                                    + seen_future_valid
                                    + valid_in_group)
                                    as f64)
                                    / 2.0;
                                for order_pos in future_pos..future_end {
                                    let member_idx = future_order[order_pos] as usize;
                                    if valid_with_future[member_idx] {
                                        paired_signal.push(signal_rank_per_stock[member_idx]);
                                        paired_future.push(avg_rank);
                                    }
                                }
                                seen_future_valid += valid_in_group;
                            }
                            future_pos = future_end;
                        }
                        pearson_corr(&paired_signal, &paired_future)
                    } else {
                        f64::NAN
                    };

                    ic_values[slot] = ic_value;
                    if !ic_value.is_nan() {
                        ic_sum += ic_value;
                        ic_count += 1;
                    }
                }

                let ic_mean_so_far = if ic_count == 0 {
                    f64::NAN
                } else {
                    ic_sum / ic_count as f64
                };
                let (long_idx, short_idx) =
                    if ic_mean_so_far.is_nan() || ic_mean_so_far >= 0.0 {
                        (portf_num - 1, 0usize)
                    } else {
                        (0usize, portf_num - 1)
                    };

                let long_ret = if group_count[long_idx] == 0 {
                    0.0
                } else {
                    group_sum[long_idx] / group_count[long_idx] as f64
                };
                let short_ret = if group_count[short_idx] == 0 {
                    0.0
                } else {
                    group_sum[short_idx] / group_count[short_idx] as f64
                };

                ls_returns[date_idx] = long_ret - short_ret;
                hedge_returns[date_idx] = long_ret - index[date_idx].to_f64();
                coverage_values[date_idx] = if valid_universe == 0 {
                    f64::NAN
                } else {
                    valid_count as f64 / valid_universe as f64
                };
            }

            let ic_mean = nanmean(&ic_values);
            let ic_std = nanstd(&ic_values);
            let ir = if ic_std.is_nan() || ic_std <= EPS {
                f64::NAN
            } else {
                ic_mean.abs() / ic_std * (250.0 / gap as f64).sqrt()
            };
            let summary = [
                ic_mean,
                ir,
                nanmean(&ls_returns) * 250.0,
                annualized_sharpe(&ls_returns),
                max_drawdown_from_returns(&ls_returns),
                n_dates as f64,
                nanmean(&coverage_values),
                nanmean(&hedge_returns) * 250.0,
                annualized_sharpe(&hedge_returns),
                max_drawdown_from_returns(&hedge_returns),
            ];
            (summary, ic_values)
        })
        .unzip();

    let mut summary_array = Array2::<f64>::from_elem((n_factors, 10), f64::NAN);
    for (factor_idx, row) in summary_rows.iter().enumerate() {
        for metric_idx in 0..10 {
            summary_array[[factor_idx, metric_idx]] = row[metric_idx];
        }
    }

    let mut ic_array = Array2::<f64>::from_elem((n_ic_dates, n_factors), f64::NAN);
    for (factor_idx, ic_values) in ic_columns.iter().enumerate() {
        for (ic_idx, value) in ic_values.iter().enumerate() {
            ic_array[[ic_idx, factor_idx]] = *value;
        }
    }

    Ok((summary_array, ic_array))
}

#[pyfunction]
#[pyo3(signature = (factor_block, ret_array, ret_sum_array, restrict_array, index_ret, gap, portf_num=10))]
pub fn tail_v2_backtest_block<'py>(
    py: Python<'py>,
    factor_block: PyReadonlyArray3<f64>,
    ret_array: PyReadonlyArray2<f64>,
    ret_sum_array: PyReadonlyArray2<f64>,
    restrict_array: PyReadonlyArray2<f64>,
    index_ret: PyReadonlyArray1<f64>,
    gap: usize,
    portf_num: usize,
) -> PyResult<(Py<PyArray2<f64>>, Py<PyArray2<f64>>)> {
    let factor = factor_block.as_array();
    let ret = ret_array.as_array();
    let ret_sum = ret_sum_array.as_array();
    let restrict = restrict_array.as_array();
    let index = index_ret.as_array();
    let (summary, ic) = py.allow_threads(|| {
        backtest_block_impl(
            factor,
            ret,
            ret_sum,
            restrict,
            index,
            gap,
            portf_num,
        )
    })?;
    Ok((summary.into_pyarray(py).to_owned(), ic.into_pyarray(py).to_owned()))
}

#[pyfunction]
#[pyo3(signature = (factor_block, ret_array, ret_sum_array, restrict_array, index_ret, gap, portf_num=10))]
pub fn tail_v2_backtest_block_f32<'py>(
    py: Python<'py>,
    factor_block: PyReadonlyArray3<f32>,
    ret_array: PyReadonlyArray2<f32>,
    ret_sum_array: PyReadonlyArray2<f32>,
    restrict_array: PyReadonlyArray2<f32>,
    index_ret: PyReadonlyArray1<f32>,
    gap: usize,
    portf_num: usize,
) -> PyResult<(Py<PyArray2<f64>>, Py<PyArray2<f64>>)> {
    let factor = factor_block.as_array();
    let ret = ret_array.as_array();
    let ret_sum = ret_sum_array.as_array();
    let restrict = restrict_array.as_array();
    let index = index_ret.as_array();
    let (summary, ic) = py.allow_threads(|| {
        backtest_block_impl(
            factor,
            ret,
            ret_sum,
            restrict,
            index,
            gap,
            portf_num,
        )
    })?;
    Ok((summary.into_pyarray(py).to_owned(), ic.into_pyarray(py).to_owned()))
}
