use numpy::{PyArray2, PyReadonlyArray3};
use pyo3::prelude::*;
use ndarray::Array2;

/// 计算3D数组各(i,j)位置的偏态值
///
/// 输入: shape (n, n, r*r) 的3D数组
/// 输出: shape (n, n) 的2D偏态值矩阵
///
/// 偏态公式: skew = E[(X-μ)³] / σ³
#[pyfunction]
#[pyo3(signature = (arr))]
pub fn skew_numba(arr: PyReadonlyArray3<f64>) -> PyResult<Py<PyArray2<f64>>> {
    let arr = arr.as_array();
    let (n, _n2, rr) = arr.dim();

    // 创建结果矩阵
    let mut result = Array2::zeros((n, n));

    for i in 0..n {
        for j in 0..n {
            // 提取当前(i,j)位置的r*r个值
            let slice = arr.slice(ndarray::s![i, j, ..]);

            // 计算均值
            let mean: f64 = slice.iter().sum::<f64>() / (rr as f64);

            // 计算方差（样本方差）和三阶矩
            let mut var_sum = 0.0;
            let mut moment3_sum = 0.0;
            for &x in slice.iter() {
                let diff = x - mean;
                var_sum += diff * diff;
                moment3_sum += diff * diff * diff;
            }

            let variance = var_sum / (rr as f64);
            let moment3 = moment3_sum / (rr as f64);

            // 计算偏态
            let std = variance.sqrt();
            if std > 1e-10 {
                result[[i, j]] = moment3 / (std * std * std);
            } else {
                result[[i, j]] = 0.0;
            }
        }
    }

    let py = Python::with_gil(|py| {
        let array = PyArray2::from_array(py, &result);
        Ok::<_, PyErr>(array.to_owned())
    })?;

    Ok(py)
}
