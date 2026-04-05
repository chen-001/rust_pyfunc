use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArrayDyn};
use pyo3::prelude::*;

/// DCT-II: 计算 values 的前 k 阶 DCT 系数
fn dct_first_k(values: &[f64], k: usize) -> Vec<f64> {
    let n = values.len() as f64;
    (0..k)
        .map(|kk| {
            let mut sum = 0.0;
            for (i, &v) in values.iter().enumerate() {
                if v.is_finite() {
                    let angle = std::f64::consts::PI * (i as f64 + 0.5) * kk as f64 / n.max(1.0);
                    sum += v * angle.cos();
                }
            }
            sum
        })
        .collect()
}

/// 一维 DCT: 输入 1D 数组，输出前 k 阶 DCT 系数
#[pyfunction]
pub fn dct_1d(
    py: Python,
    values: PyReadonlyArrayDyn<f64>,
    k: usize,
) -> PyResult<Py<PyArray1<f64>>> {
    let data = values.as_array();
    let flat: Vec<f64> = data.iter().cloned().collect();
    let result = dct_first_k(&flat, k);
    Ok(PyArray1::from_vec(py, result).to_owned())
}

/// 二维 DCT: 输入 (n, m) 矩阵，对每列分别计算前 k 阶 DCT，输出 (k, m) 矩阵
#[pyfunction]
pub fn dct_2d(
    py: Python,
    matrix: PyReadonlyArrayDyn<f64>,
    k: usize,
) -> PyResult<Py<PyArray2<f64>>> {
    let data = matrix.as_array();
    let shape = data.shape();
    if shape.len() != 2 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "输入必须是二维矩阵 (n, m)",
        ));
    }
    let n = shape[0];
    let m = shape[1];
    let flat: Vec<f64> = data.iter().cloned().collect();

    let mut result = vec![0.0f64; k * m];
    for col in 0..m {
        let col_data: Vec<f64> = (0..n).map(|row| flat[row * m + col]).collect();
        let dct = dct_first_k(&col_data, k);
        for (j, &v) in dct.iter().enumerate() {
            result[j * m + col] = v;
        }
    }

    use ndarray::Array2;
    let arr = Array2::from_shape_vec((k, m), result)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("数组形状错误: {}", e)))?;
    Ok(arr.into_pyarray(py).to_owned())
}
