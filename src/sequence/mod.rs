use pyo3::prelude::*;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use ndarray::Array1;


/// 识别数组中的连续相等值段，并为每个段分配唯一标识符。
/// 每个连续相等的值构成一个段，第一个段标识符为1，第二个为2，以此类推。
///
/// 参数说明：
/// ----------
/// arr : numpy.ndarray
///     输入数组，类型为float64
///
/// 返回值：
/// -------
/// numpy.ndarray
///     与输入数组等长的整数数组，每个元素表示该位置所属段的标识符
///
/// Python调用示例：
/// ```python
/// import numpy as np
/// from rust_pyfunc import identify_segments
///
/// # 创建测试数组
/// arr = np.array([1.0, 1.0, 2.0, 2.0, 2.0, 1.0], dtype=np.float64)
/// segments = identify_segments(arr)
/// print(f"段标识: {segments}")  # 输出: [1, 1, 2, 2, 2, 3]
///
/// # 解释结果：
/// # - 第一段 [1.0, 1.0] 标识为1
/// # - 第二段 [2.0, 2.0, 2.0] 标识为2
/// # - 第三段 [1.0] 标识为3
/// ```
#[pyfunction]
#[pyo3(signature = (arr))]
pub fn identify_segments(arr: PyReadonlyArray1<f64>) -> PyResult<Py<PyArray1<i32>>> {
    let arr_view = arr.as_array();
    let n = arr_view.len();
    let mut segments = Array1::zeros(n);
    let mut current_segment = 1;

    for i in 1..n {
        if arr_view[i] != arr_view[i - 1] {
            current_segment += 1;
        }
        segments[i] = current_segment;
    }

    Ok(segments.into_pyarray(arr.py()).to_owned())
}



/// 在数组中找到一对索引(x, y)，使得min(arr[x], arr[y]) * |x-y|的值最大。
/// 这个函数可以用来找到数组中距离最远的两个元素，同时考虑它们的最小值。
///
/// 参数说明：
/// ----------
/// arr : numpy.ndarray
///     输入数组，类型为float64
///
/// 返回值：
/// -------
/// tuple
///     返回一个元组(x, y, max_product)，其中x和y是使得乘积最大的索引对，max_product是最大乘积
///
/// Python调用示例：
/// ```python
/// import numpy as np
/// from rust_pyfunc import find_max_range_product
///
/// # 创建测试数组
/// arr = np.array([4.0, 2.0, 1.0, 3.0], dtype=np.float64)
/// x, y, max_product = find_max_range_product(arr)
/// 
/// print(f"最大乘积出现在索引 {x} 和 {y}")
/// print(f"对应的值为 {arr[x]} 和 {arr[y]}")
/// print(f"最大乘积为: {max_product}")
///
/// # 例如，如果x=0, y=3，那么：
/// # min(arr[0], arr[3]) * |0-3| = min(4.0, 3.0) * 3 = 3.0 * 3 = 9.0
/// ```
#[pyfunction]
#[pyo3(signature = (arr))]
pub fn find_max_range_product(arr: PyReadonlyArray1<f64>) -> PyResult<(i64, i64, f64)> {
    let arr_view = arr.as_array();
    let n = arr_view.len();
    
    if n < 2 {
        return Ok((0, 0, 0.0));
    }

    let mut max_product = f64::NEG_INFINITY;
    let mut result = (0i64, 0i64);
    let mut left = 0;
    let mut right = n - 1;

    while left < right {
        let product = arr_view[left].min(arr_view[right]) * (right - left) as f64;
        if product > max_product {
            max_product = product;
            result = (left as i64, right as i64);
        }

        if arr_view[left] < arr_view[right] {
            left += 1;
        } else {
            right -= 1;
        }
    }

    for i in 0..n-1 {
        let product = arr_view[i].min(arr_view[i+1]) * 1.0;
        if product > max_product {
            max_product = product;
            result = (i as i64, (i+1) as i64);
        }
    }
    
    Ok((result.0, result.1, max_product))
}