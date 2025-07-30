use pyo3::prelude::*;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use std::collections::HashMap;

#[pyfunction]
pub fn calculate_entropy_1d(py: Python, data: PyReadonlyArray1<f64>) -> PyResult<f64> {
    let data = data.as_array();
    
    if data.is_empty() {
        return Ok(0.0);
    }
    
    let mut counts: HashMap<String, usize> = HashMap::new();
    let total = data.len();
    
    for &value in data.iter() {
        let key = if value.is_nan() {
            "NaN".to_string()
        } else {
            format!("{:.10}", value)
        };
        *counts.entry(key).or_insert(0) += 1;
    }
    
    let entropy = counts.values()
        .map(|&count| {
            let prob = count as f64 / total as f64;
            if prob > 0.0 {
                -prob * prob.ln()
            } else {
                0.0
            }
        })
        .sum::<f64>();
    
    Ok(entropy)
}

#[pyfunction]
pub fn calculate_entropy_2d(
    py: Python,
    data: PyReadonlyArray2<f64>,
    axis: Option<i32>,
) -> PyResult<PyObject> {
    let data = data.as_array();
    
    match axis {
        Some(0) => {
            let (_, cols) = data.dim();
            let mut entropies: Vec<f64> = Vec::with_capacity(cols);
            
            for col in 0..cols {
                let column = data.column(col);
                entropies.push(calculate_entropy_for_slice(&column.to_vec()));
            }
            
            Ok(entropies.into_pyarray(py).into())
        }
        Some(1) => {
            let (rows, _) = data.dim();
            let mut entropies: Vec<f64> = Vec::with_capacity(rows);
            
            for row in 0..rows {
                let row_data = data.row(row);
                entropies.push(calculate_entropy_for_slice(&row_data.to_vec()));
            }
            
            Ok(entropies.into_pyarray(py).into())
        }
        None => {
            let flattened: Vec<f64> = data.iter().cloned().collect();
            let entropy = calculate_entropy_for_slice(&flattened);
            Ok(entropy.into_py(py))
        }
        _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "axis must be None, 0, or 1"
        )),
    }
}

fn calculate_entropy_for_slice(data: &[f64]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    
    let mut counts: HashMap<String, usize> = HashMap::new();
    let total = data.len();
    
    for &value in data.iter() {
        let key = if value.is_nan() {
            "NaN".to_string()
        } else {
            format!("{:.10}", value)
        };
        *counts.entry(key).or_insert(0) += 1;
    }
    
    counts.values()
        .map(|&count| {
            let prob = count as f64 / total as f64;
            if prob > 0.0 {
                -prob * prob.ln()
            } else {
                0.0
            }
        })
        .sum::<f64>()
}

#[pyfunction]
pub fn calculate_entropy_discrete_1d(py: Python, data: PyReadonlyArray1<i64>) -> PyResult<f64> {
    let data = data.as_array();
    
    if data.is_empty() {
        return Ok(0.0);
    }
    
    let mut counts: HashMap<i64, usize> = HashMap::new();
    let total = data.len();
    
    for &value in data.iter() {
        *counts.entry(value).or_insert(0) += 1;
    }
    
    let entropy = counts.values()
        .map(|&count| {
            let prob = count as f64 / total as f64;
            if prob > 0.0 {
                -prob * prob.ln()
            } else {
                0.0
            }
        })
        .sum::<f64>();
    
    Ok(entropy)
}

#[pyfunction]
pub fn calculate_entropy_discrete_2d(
    py: Python,
    data: PyReadonlyArray2<i64>,
    axis: Option<i32>,
) -> PyResult<PyObject> {
    let data = data.as_array();
    
    match axis {
        Some(0) => {
            let (_, cols) = data.dim();
            let mut entropies: Vec<f64> = Vec::with_capacity(cols);
            
            for col in 0..cols {
                let column = data.column(col);
                entropies.push(calculate_entropy_for_discrete_slice(&column.to_vec()));
            }
            
            Ok(entropies.into_pyarray(py).into())
        }
        Some(1) => {
            let (rows, _) = data.dim();
            let mut entropies: Vec<f64> = Vec::with_capacity(rows);
            
            for row in 0..rows {
                let row_data = data.row(row);
                entropies.push(calculate_entropy_for_discrete_slice(&row_data.to_vec()));
            }
            
            Ok(entropies.into_pyarray(py).into())
        }
        None => {
            let flattened: Vec<i64> = data.iter().cloned().collect();
            let entropy = calculate_entropy_for_discrete_slice(&flattened);
            Ok(entropy.into_py(py))
        }
        _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "axis must be None, 0, or 1"
        )),
    }
}

fn calculate_entropy_for_discrete_slice(data: &[i64]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    
    let mut counts: HashMap<i64, usize> = HashMap::new();
    let total = data.len();
    
    for &value in data.iter() {
        *counts.entry(value).or_insert(0) += 1;
    }
    
    counts.values()
        .map(|&count| {
            let prob = count as f64 / total as f64;
            if prob > 0.0 {
                -prob * prob.ln()
            } else {
                0.0
            }
        })
        .sum::<f64>()
}

#[pyfunction]
pub fn calculate_binned_entropy_1d(
    py: Python, 
    data: PyReadonlyArray1<f64>,
    n_bins: usize,
    bin_method: Option<&str>
) -> PyResult<f64> {
    let data = data.as_array();
    
    if data.is_empty() {
        return Ok(0.0);
    }
    
    if n_bins == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "n_bins must be greater than 0"
        ));
    }
    
    let method = bin_method.unwrap_or("equal_width");
    
    let data_vec: Vec<f64> = data.to_vec();
    let bin_indices = match method {
        "equal_width" => equal_width_binning(&data_vec, n_bins)?,
        "equal_frequency" => equal_frequency_binning(&data_vec, n_bins)?,
        _ => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "bin_method must be 'equal_width' or 'equal_frequency'"
        )),
    };
    
    let mut counts: HashMap<usize, usize> = HashMap::new();
    let total = bin_indices.len();
    
    for &bin_idx in bin_indices.iter() {
        *counts.entry(bin_idx).or_insert(0) += 1;
    }
    
    let entropy = counts.values()
        .map(|&count| {
            let prob = count as f64 / total as f64;
            if prob > 0.0 {
                -prob * prob.ln()
            } else {
                0.0
            }
        })
        .sum::<f64>();
    
    Ok(entropy)
}

#[pyfunction]
pub fn calculate_binned_entropy_2d(
    py: Python,
    data: PyReadonlyArray2<f64>,
    n_bins: usize,
    bin_method: Option<&str>,
    axis: Option<i32>,
) -> PyResult<PyObject> {
    let data = data.as_array();
    
    if n_bins == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "n_bins must be greater than 0"
        ));
    }
    
    let method = bin_method.unwrap_or("equal_width");
    
    match axis {
        Some(0) => {
            let (_, cols) = data.dim();
            let mut entropies: Vec<f64> = Vec::with_capacity(cols);
            
            for col in 0..cols {
                let column = data.column(col);
                let column_vec: Vec<f64> = column.to_vec();
                
                if column_vec.is_empty() {
                    entropies.push(0.0);
                    continue;
                }
                
                let bin_indices = match method {
                    "equal_width" => equal_width_binning(&column_vec, n_bins)?,
                    "equal_frequency" => equal_frequency_binning(&column_vec, n_bins)?,
                    _ => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "bin_method must be 'equal_width' or 'equal_frequency'"
                    )),
                };
                
                let entropy = calculate_entropy_from_bins(&bin_indices);
                entropies.push(entropy);
            }
            
            Ok(entropies.into_pyarray(py).into())
        }
        Some(1) => {
            let (rows, _) = data.dim();
            let mut entropies: Vec<f64> = Vec::with_capacity(rows);
            
            for row in 0..rows {
                let row_data = data.row(row);
                let row_vec: Vec<f64> = row_data.to_vec();
                
                if row_vec.is_empty() {
                    entropies.push(0.0);
                    continue;
                }
                
                let bin_indices = match method {
                    "equal_width" => equal_width_binning(&row_vec, n_bins)?,
                    "equal_frequency" => equal_frequency_binning(&row_vec, n_bins)?,
                    _ => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "bin_method must be 'equal_width' or 'equal_frequency'"
                    )),
                };
                
                let entropy = calculate_entropy_from_bins(&bin_indices);
                entropies.push(entropy);
            }
            
            Ok(entropies.into_pyarray(py).into())
        }
        None => {
            let flattened: Vec<f64> = data.iter().cloned().collect();
            
            if flattened.is_empty() {
                return Ok(0.0.into_py(py));
            }
            
            let bin_indices = match method {
                "equal_width" => equal_width_binning(&flattened, n_bins)?,
                "equal_frequency" => equal_frequency_binning(&flattened, n_bins)?,
                _ => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "bin_method must be 'equal_width' or 'equal_frequency'"
                )),
            };
            
            let entropy = calculate_entropy_from_bins(&bin_indices);
            Ok(entropy.into_py(py))
        }
        _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "axis must be None, 0, or 1"
        )),
    }
}

fn equal_width_binning(data: &[f64], n_bins: usize) -> Result<Vec<usize>, PyErr> {
    let filtered_data: Vec<f64> = data.iter().filter(|&&x| !x.is_nan()).cloned().collect();
    
    if filtered_data.is_empty() {
        return Ok(vec![0; data.len()]);
    }
    
    if filtered_data.len() == 1 {
        return Ok(vec![0; data.len()]);
    }
    
    let min_val = filtered_data.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max_val = filtered_data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    
    if (max_val - min_val).abs() < f64::EPSILON {
        return Ok(vec![0; data.len()]);
    }
    
    let bin_width = (max_val - min_val) / n_bins as f64;
    
    let mut bin_indices = Vec::with_capacity(data.len());
    
    for &value in data.iter() {
        if value.is_nan() {
            bin_indices.push(n_bins);
        } else {
            let bin_idx = ((value - min_val) / bin_width).floor() as usize;
            let bin_idx = if bin_idx >= n_bins { n_bins - 1 } else { bin_idx };
            bin_indices.push(bin_idx);
        }
    }
    
    Ok(bin_indices)
}

fn equal_frequency_binning(data: &[f64], n_bins: usize) -> Result<Vec<usize>, PyErr> {
    let mut filtered_data: Vec<(f64, usize)> = data.iter()
        .enumerate()
        .filter(|(_, &x)| !x.is_nan())
        .map(|(i, &x)| (x, i))
        .collect();
    
    if filtered_data.is_empty() {
        return Ok(vec![0; data.len()]);
    }
    
    filtered_data.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    
    let bin_size = filtered_data.len() as f64 / n_bins as f64;
    let mut bin_indices = vec![0; data.len()];
    
    for (rank, &(_, original_idx)) in filtered_data.iter().enumerate() {
        let bin_idx = ((rank as f64 / bin_size).floor() as usize).min(n_bins - 1);
        bin_indices[original_idx] = bin_idx;
    }
    
    for (i, &value) in data.iter().enumerate() {
        if value.is_nan() {
            bin_indices[i] = n_bins;
        }
    }
    
    Ok(bin_indices)
}

fn calculate_entropy_from_bins(bin_indices: &[usize]) -> f64 {
    if bin_indices.is_empty() {
        return 0.0;
    }
    
    let mut counts: HashMap<usize, usize> = HashMap::new();
    let total = bin_indices.len();
    
    for &bin_idx in bin_indices.iter() {
        *counts.entry(bin_idx).or_insert(0) += 1;
    }
    
    counts.values()
        .map(|&count| {
            let prob = count as f64 / total as f64;
            if prob > 0.0 {
                -prob * prob.ln()
            } else {
                0.0
            }
        })
        .sum::<f64>()
}