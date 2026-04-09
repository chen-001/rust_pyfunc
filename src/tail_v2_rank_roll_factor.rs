use std::cmp::Ordering;
use std::collections::VecDeque;

use ndarray::{Array2, Array3};
use numpy::{IntoPyArray, PyArray2, PyArray3, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

fn rank_average_row(row: &[f32]) -> Vec<f32> {
    let mut indexed = row
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, value)| !value.is_nan())
        .collect::<Vec<_>>();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));

    let mut ranked = vec![f32::NAN; row.len()];
    let mut start = 0usize;
    while start < indexed.len() {
        let value = indexed[start].1;
        let mut end = start + 1;
        while end < indexed.len() && indexed[end].1 == value {
            end += 1;
        }
        let avg_rank = ((start + 1 + end) as f64 / 2.0) as f32;
        for item in indexed.iter().take(end).skip(start) {
            ranked[item.0] = avg_rank;
        }
        start = end;
    }
    ranked
}

fn rank_axis1_average_f32(data: &Array2<f32>) -> Array2<f32> {
    let (n_rows, n_cols) = data.dim();
    let mut flat = vec![f32::NAN; n_rows * n_cols];
    flat.par_chunks_mut(n_cols)
        .enumerate()
        .for_each(|(row_idx, out_row)| {
            let row = data.row(row_idx);
            let ranked = rank_average_row(row.as_slice().unwrap_or(&[]));
            out_row.copy_from_slice(&ranked);
        });
    Array2::from_shape_vec((n_rows, n_cols), flat).unwrap()
}

fn rank_axis1_average_f32_serial(data: &Array2<f32>) -> Array2<f32> {
    let (n_rows, n_cols) = data.dim();
    let mut flat = vec![f32::NAN; n_rows * n_cols];
    for row_idx in 0..n_rows {
        let row = data.row(row_idx);
        let ranked = rank_average_row(row.as_slice().unwrap_or(&[]));
        let start = row_idx * n_cols;
        flat[start..start + n_cols].copy_from_slice(&ranked);
    }
    Array2::from_shape_vec((n_rows, n_cols), flat).unwrap()
}

fn rolling_stats_for_column(
    ranked: &Array2<f32>,
    col_idx: usize,
    window: usize,
    min_periods: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let n_rows = ranked.nrows();
    let mut mean_out = vec![f32::NAN; n_rows];
    let mut max_out = vec![f32::NAN; n_rows];
    let mut min_out = vec![f32::NAN; n_rows];
    let mut std_out = vec![f32::NAN; n_rows];

    let mut sum = 0.0f64;
    let mut sumsq = 0.0f64;
    let mut count = 0usize;
    let mut max_deque = VecDeque::<(usize, f32)>::new();
    let mut min_deque = VecDeque::<(usize, f32)>::new();

    for row_idx in 0..n_rows {
        let value = ranked[[row_idx, col_idx]];
        if !value.is_nan() {
            let value64 = value as f64;
            sum += value64;
            sumsq += value64 * value64;
            count += 1;

            while let Some((_, tail_val)) = max_deque.back() {
                if *tail_val <= value {
                    max_deque.pop_back();
                } else {
                    break;
                }
            }
            max_deque.push_back((row_idx, value));

            while let Some((_, tail_val)) = min_deque.back() {
                if *tail_val >= value {
                    min_deque.pop_back();
                } else {
                    break;
                }
            }
            min_deque.push_back((row_idx, value));
        }

        if row_idx >= window {
            let leave_idx = row_idx - window;
            let leave_value = ranked[[leave_idx, col_idx]];
            if !leave_value.is_nan() {
                let leave64 = leave_value as f64;
                sum -= leave64;
                sumsq -= leave64 * leave64;
                count -= 1;
            }
        }

        let valid_start = (row_idx + 1).saturating_sub(window);
        while let Some((idx, _)) = max_deque.front() {
            if *idx < valid_start {
                max_deque.pop_front();
            } else {
                break;
            }
        }
        while let Some((idx, _)) = min_deque.front() {
            if *idx < valid_start {
                min_deque.pop_front();
            } else {
                break;
            }
        }

        if count >= min_periods {
            let mean = sum / count as f64;
            mean_out[row_idx] = mean as f32;
            max_out[row_idx] = max_deque.front().map(|item| item.1).unwrap_or(f32::NAN);
            min_out[row_idx] = min_deque.front().map(|item| item.1).unwrap_or(f32::NAN);
            if count > 1 {
                let variance = ((sumsq - (sum * sum) / count as f64) / (count as f64 - 1.0)).max(0.0);
                std_out[row_idx] = variance.sqrt() as f32;
            }
        }
    }

    (mean_out, max_out, min_out, std_out)
}

fn rolling_stats_f32(
    ranked: &Array2<f32>,
    window: usize,
    min_periods: usize,
) -> (Array2<f32>, Array2<f32>, Array2<f32>, Array2<f32>) {
    let n_rows = ranked.nrows();
    let n_cols = ranked.ncols();
    let col_outputs = (0..n_cols)
        .into_par_iter()
        .map(|col_idx| rolling_stats_for_column(ranked, col_idx, window, min_periods))
        .collect::<Vec<_>>();

    let mut mean = Array2::<f32>::from_elem((n_rows, n_cols), f32::NAN);
    let mut max = Array2::<f32>::from_elem((n_rows, n_cols), f32::NAN);
    let mut min = Array2::<f32>::from_elem((n_rows, n_cols), f32::NAN);
    let mut std = Array2::<f32>::from_elem((n_rows, n_cols), f32::NAN);

    for (col_idx, (mean_col, max_col, min_col, std_col)) in col_outputs.into_iter().enumerate() {
        for row_idx in 0..n_rows {
            mean[[row_idx, col_idx]] = mean_col[row_idx];
            max[[row_idx, col_idx]] = max_col[row_idx];
            min[[row_idx, col_idx]] = min_col[row_idx];
            std[[row_idx, col_idx]] = std_col[row_idx];
        }
    }
    (mean, max, min, std)
}

fn rolling_stats_f32_serial(
    ranked: &Array2<f32>,
    window: usize,
    min_periods: usize,
) -> (Array2<f32>, Array2<f32>, Array2<f32>, Array2<f32>) {
    let n_rows = ranked.nrows();
    let n_cols = ranked.ncols();
    let mut mean = Array2::<f32>::from_elem((n_rows, n_cols), f32::NAN);
    let mut max = Array2::<f32>::from_elem((n_rows, n_cols), f32::NAN);
    let mut min = Array2::<f32>::from_elem((n_rows, n_cols), f32::NAN);
    let mut std = Array2::<f32>::from_elem((n_rows, n_cols), f32::NAN);

    for col_idx in 0..n_cols {
        let (mean_col, max_col, min_col, std_col) =
            rolling_stats_for_column(ranked, col_idx, window, min_periods);
        for row_idx in 0..n_rows {
            mean[[row_idx, col_idx]] = mean_col[row_idx];
            max[[row_idx, col_idx]] = max_col[row_idx];
            min[[row_idx, col_idx]] = min_col[row_idx];
            std[[row_idx, col_idx]] = std_col[row_idx];
        }
    }

    (mean, max, min, std)
}

pub(crate) fn rank_roll_block_f32_with_parallel(
    data: &Array2<f32>,
    windows: &[usize],
    parallel: bool,
) -> Result<Array3<f32>, String> {
    let ranked = if parallel {
        rank_axis1_average_f32(data)
    } else {
        rank_axis1_average_f32_serial(data)
    };
    let mut arrays = vec![ranked.clone()];
    for &window in windows {
        if window == 0 {
            return Err("window 必须大于 0".to_string());
        }
        let min_periods = std::cmp::max(1, window / 2);
        let (mean, max, min, std) = if parallel {
            rolling_stats_f32(&ranked, window, min_periods)
        } else {
            rolling_stats_f32_serial(&ranked, window, min_periods)
        };
        arrays.push(mean);
        arrays.push(max);
        arrays.push(min);
        arrays.push(std);
    }

    let n_rows = ranked.nrows();
    let n_cols = ranked.ncols();
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
#[pyo3(signature = (data, windows))]
pub fn tail_v2_rank_roll_factor_f32<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<f32>,
    windows: Vec<usize>,
) -> PyResult<Vec<Py<PyArray2<f32>>>> {
    let data_view = data.as_array();
    let data_owned = Array2::<f32>::from_shape_vec(data_view.dim(), data_view.iter().copied().collect())
        .map_err(|_| PyValueError::new_err("data 形状无效"))?;

    let outputs = py.allow_threads(|| {
        let ranked = rank_axis1_average_f32(&data_owned);
        let mut arrays = vec![ranked.clone()];
        for &window in &windows {
            if window == 0 {
                return Err(PyValueError::new_err("window 必须大于 0"));
            }
            let min_periods = std::cmp::max(1, window / 2);
            let (mean, max, min, std) = rolling_stats_f32(&ranked, window, min_periods);
            arrays.push(mean);
            arrays.push(max);
            arrays.push(min);
            arrays.push(std);
        }
        Ok::<Vec<Array2<f32>>, PyErr>(arrays)
    })?;

    Ok(outputs
        .into_iter()
        .map(|array| array.into_pyarray(py).to_owned())
        .collect())
}

#[pyfunction]
#[pyo3(signature = (data, windows))]
pub fn tail_v3_rank_roll_block_f32<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<f32>,
    windows: Vec<usize>,
) -> PyResult<Py<PyArray3<f32>>> {
    let data_view = data.as_array();
    let data_owned = Array2::<f32>::from_shape_vec(data_view.dim(), data_view.iter().copied().collect())
        .map_err(|_| PyValueError::new_err("data 形状无效"))?;

    let output = py.allow_threads(|| {
        rank_roll_block_f32_with_parallel(&data_owned, &windows, true)
            .map_err(PyValueError::new_err)
    })?;

    Ok(output.into_pyarray(py).to_owned())
}
