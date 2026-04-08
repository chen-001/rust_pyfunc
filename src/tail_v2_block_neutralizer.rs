use nalgebra::{DMatrix, DVector};
use ndarray::{Array3, ArrayView3};
use numpy::{IntoPyArray, PyArray3, PyReadonlyArray3};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

trait FloatLike: Copy + Send + Sync {
    fn to_f64(self) -> f64;
    fn is_nan(self) -> bool {
        self.to_f64().is_nan()
    }
}

trait OutputLike: Copy + Send + Sync {
    fn nan() -> Self;
    fn from_f64(value: f64) -> Self;
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

impl OutputLike for f64 {
    #[inline]
    fn nan() -> Self {
        f64::NAN
    }

    #[inline]
    fn from_f64(value: f64) -> Self {
        value
    }
}

impl OutputLike for f32 {
    #[inline]
    fn nan() -> Self {
        f32::NAN
    }

    #[inline]
    fn from_f64(value: f64) -> Self {
        value as f32
    }
}

fn rank_valid_values(values: &[f64]) -> Vec<f64> {
    let mut indexed: Vec<(usize, f64)> = values.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    let mut ranked = vec![f64::NAN; values.len()];
    for (rank, (original_idx, _)) in indexed.iter().enumerate() {
        ranked[*original_idx] = (rank + 1) as f64;
    }
    ranked
}

fn solve_beta(xtx_values: &[f64], xty_values: &[f64], n_features: usize) -> Option<DVector<f64>> {
    let xtx = DMatrix::from_row_slice(n_features, n_features, xtx_values);
    let xty = DVector::from_vec(xty_values.to_vec());
    if let Some(chol) = xtx.clone().cholesky() {
        return Some(chol.solve(&xty));
    }
    xtx.lu().solve(&xty)
}

fn neutralize_impl<T: FloatLike, O: OutputLike>(
    style: ArrayView3<'_, T>,
    factor: ArrayView3<'_, T>,
    rank_before: bool,
    min_valid: usize,
) -> PyResult<Array3<O>> {
    if style.ndim() != 3 || factor.ndim() != 3 {
        return Err(PyValueError::new_err("style_cube 和 factor_block 都必须是三维数组"));
    }
    if style.shape()[0] != factor.shape()[0] || style.shape()[1] != factor.shape()[1] {
        return Err(PyValueError::new_err(
            "style_cube 与 factor_block 的日期轴和股票轴长度必须一致",
        ));
    }

    let n_dates = factor.shape()[0];
    let n_stocks = factor.shape()[1];
    let n_factors = factor.shape()[2];
    let n_features = style.shape()[2];

    let mut output_flat = vec![O::nan(); n_dates * n_stocks * n_factors];
    output_flat
        .par_chunks_mut(n_stocks * n_factors)
        .enumerate()
        .for_each(|(date_idx, day_output)| {
            let mut style_valid_stock_indices = Vec::with_capacity(n_stocks);
            let mut style_valid_rows = Vec::with_capacity(n_stocks * n_features);
            for stock_idx in 0..n_stocks {
                let mut row_valid = true;
                for feature_idx in 0..n_features {
                    if style[[date_idx, stock_idx, feature_idx]].is_nan() {
                        row_valid = false;
                        break;
                    }
                }
                if !row_valid {
                    continue;
                }
                style_valid_stock_indices.push(stock_idx);
                for feature_idx in 0..n_features {
                    style_valid_rows.push(style[[date_idx, stock_idx, feature_idx]].to_f64());
                }
            }

            if style_valid_stock_indices.len() < min_valid {
                return;
            }

            let mut valid_stock_indices = Vec::with_capacity(style_valid_stock_indices.len());
            let mut valid_style_positions = Vec::with_capacity(style_valid_stock_indices.len());
            let mut y_values = Vec::with_capacity(style_valid_stock_indices.len());
            let mut xtx_values = vec![0.0; n_features * n_features];
            let mut xty_values = vec![0.0; n_features];

            for factor_idx in 0..n_factors {
                valid_stock_indices.clear();
                valid_style_positions.clear();
                y_values.clear();

                for (style_pos, &stock_idx) in style_valid_stock_indices.iter().enumerate() {
                    let factor_value = factor[[date_idx, stock_idx, factor_idx]];
                    if factor_value.is_nan() {
                        continue;
                    }
                    valid_stock_indices.push(stock_idx);
                    valid_style_positions.push(style_pos);
                    y_values.push(factor_value.to_f64());
                }

                if valid_stock_indices.len() < min_valid {
                    continue;
                }

                let y_ranked = if rank_before {
                    rank_valid_values(&y_values)
                } else {
                    y_values.clone()
                };

                xtx_values.fill(0.0);
                xty_values.fill(0.0);
                for (row_idx, &style_pos) in valid_style_positions.iter().enumerate() {
                    let row_offset = style_pos * n_features;
                    let y_value = y_ranked[row_idx];
                    for i in 0..n_features {
                        let xi = style_valid_rows[row_offset + i];
                        xty_values[i] += xi * y_value;
                        let base = i * n_features;
                        for j in 0..=i {
                            xtx_values[base + j] += xi * style_valid_rows[row_offset + j];
                        }
                    }
                }
                for i in 0..n_features {
                    for j in 0..i {
                        xtx_values[j * n_features + i] = xtx_values[i * n_features + j];
                    }
                }

                let beta = if let Some(solution) = solve_beta(&xtx_values, &xty_values, n_features) {
                    solution
                } else {
                    continue;
                };
                for (row_idx, stock_idx) in valid_stock_indices.iter().enumerate() {
                    let row_offset = valid_style_positions[row_idx] * n_features;
                    let mut fitted = 0.0;
                    for feature_idx in 0..n_features {
                        fitted += style_valid_rows[row_offset + feature_idx] * beta[feature_idx];
                    }
                    day_output[*stock_idx * n_factors + factor_idx] =
                        O::from_f64(y_ranked[row_idx] - fitted);
                }
            }
        });

    Array3::from_shape_vec((n_dates, n_stocks, n_factors), output_flat)
        .map_err(|_| PyValueError::new_err("neutralize 输出形状构造失败"))
}

#[pyfunction]
#[pyo3(signature = (style_cube, factor_block, rank_before=true, min_valid=12))]
pub fn tail_v2_neutralize_block<'py>(
    py: Python<'py>,
    style_cube: PyReadonlyArray3<f64>,
    factor_block: PyReadonlyArray3<f64>,
    rank_before: bool,
    min_valid: usize,
) -> PyResult<Py<PyArray3<f64>>> {
    let style = style_cube.as_array();
    let factor = factor_block.as_array();
    let output = py.allow_threads(|| neutralize_impl::<f64, f64>(style, factor, rank_before, min_valid))?;
    Ok(output.into_pyarray(py).to_owned())
}

#[pyfunction]
#[pyo3(signature = (style_cube, factor_block, rank_before=true, min_valid=12))]
pub fn tail_v2_neutralize_block_f32<'py>(
    py: Python<'py>,
    style_cube: PyReadonlyArray3<f32>,
    factor_block: PyReadonlyArray3<f32>,
    rank_before: bool,
    min_valid: usize,
) -> PyResult<Py<PyArray3<f64>>> {
    let style = style_cube.as_array();
    let factor = factor_block.as_array();
    let output = py.allow_threads(|| neutralize_impl::<f32, f64>(style, factor, rank_before, min_valid))?;
    Ok(output.into_pyarray(py).to_owned())
}

#[pyfunction]
#[pyo3(signature = (style_cube, factor_block, rank_before=true, min_valid=12))]
pub fn tail_v2_neutralize_block_f32_out<'py>(
    py: Python<'py>,
    style_cube: PyReadonlyArray3<f32>,
    factor_block: PyReadonlyArray3<f32>,
    rank_before: bool,
    min_valid: usize,
) -> PyResult<Py<PyArray3<f32>>> {
    let style = style_cube.as_array();
    let factor = factor_block.as_array();
    let output = py.allow_threads(|| neutralize_impl::<f32, f32>(style, factor, rank_before, min_valid))?;
    Ok(output.into_pyarray(py).to_owned())
}
