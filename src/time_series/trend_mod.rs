use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyList;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// 计算输入数组与自然数序列(1, 2, ..., n)之间的皮尔逊相关系数。
/// 这个函数可以用来判断一个序列的趋势性，如果返回值接近1表示强上升趋势，接近-1表示强下降趋势。
///
/// 参数说明：
/// ----------
/// arr : 输入数组
///     可以是以下类型之一：
///     - numpy.ndarray (float64或int64类型)
///     - Python列表 (float或int类型)
///
/// 返回值：
/// -------
/// float
///     输入数组与自然数序列的皮尔逊相关系数。
///     如果输入数组为空或方差为零，则返回0.0。
///
/// Python调用示例：
/// ```python
/// import numpy as np
/// from rust_pyfunc import trend
///
/// # 使用numpy数组
/// arr1 = np.array([1.0, 2.0, 3.0, 4.0])  # 完美上升趋势
/// result1 = trend(arr1)  # 返回接近1.0
///
/// # 使用Python列表
/// arr2 = [4, 3, 2, 1]  # 完美下降趋势
/// result2 = trend(arr2)  # 返回接近-1.0
///
/// # 无趋势序列
/// arr3 = [1, 1, 1, 1]
/// result3 = trend(arr3)  # 返回0.0
/// ```
#[pyfunction]
#[pyo3(signature = (arr))]
pub fn trend(arr: &PyAny) -> PyResult<f64> {
    let py = arr.py();

    // 尝试将输入转换为Vec<f64>
    let arr_vec: Vec<f64> = if arr.is_instance_of::<PyList>()? {
        let list = arr.downcast::<PyList>()?;
        let mut result = Vec::with_capacity(list.len());
        for item in list.iter() {
            if let Ok(val) = item.extract::<f64>() {
                result.push(val);
            } else if let Ok(val) = item.extract::<i64>() {
                result.push(val as f64);
            } else {
                return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                    "List elements must be numeric (float or int)",
                ));
            }
        }
        result
    } else {
        // 尝试将输入转换为numpy数组
        let numpy = py.import("numpy")?;
        let arr = numpy.call_method1("asarray", (arr,))?;
        let arr = arr.call_method1("astype", ("float64",))?;
        arr.extract::<Vec<f64>>()?
    };

    // 过滤掉NaN值，同时保持对应的索引
    let valid_pairs: Vec<(f64, f64)> = arr_vec
        .iter()
        .enumerate()
        .filter_map(|(i, &val)| {
            if val.is_finite() {
                Some((val, (i + 1) as f64))
            } else {
                None
            }
        })
        .collect();

    let n = valid_pairs.len();

    if n == 0 || n == 1 {
        return Ok(0.0);
    }

    // 分离有效的值和对应的索引
    let (values, indices): (Vec<f64>, Vec<f64>) = valid_pairs.into_iter().unzip();

    // 计算均值
    let mean_x: f64 = values.iter().sum::<f64>() / n as f64;
    let mean_y: f64 = indices.iter().sum::<f64>() / n as f64;

    // 计算协方差和标准差
    let mut covariance: f64 = 0.0;
    let mut var_x: f64 = 0.0;
    let mut var_y: f64 = 0.0;

    for i in 0..n {
        let diff_x = values[i] - mean_x;
        let diff_y = indices[i] - mean_y;

        covariance += diff_x * diff_y;
        var_x += diff_x * diff_x;
        var_y += diff_y * diff_y;
    }

    // 避免除以零
    if var_x == 0.0 || var_y == 0.0 {
        return Ok(0.0);
    }

    // 计算相关系数
    let correlation = covariance / (var_x.sqrt() * var_y.sqrt());

    Ok(correlation)
}

/// 这是trend函数的高性能版本，专门用于处理numpy.ndarray类型的float64数组。
/// 使用了显式的SIMD指令和缓存优化处理，比普通版本更快。
///
/// 参数说明：
/// ----------
/// arr : numpy.ndarray
///     输入数组，必须是float64类型
///
/// 返回值：
/// -------
/// float
///     输入数组与自然数序列的皮尔逊相关系数
///
/// Python调用示例：
/// ```python
/// import numpy as np
/// from rust_pyfunc import trend_fast
///
/// # 创建一个大型数组进行测试
/// arr = np.array([float(i) for i in range(1000000)], dtype=np.float64)
/// result = trend_fast(arr)  # 会比trend函数快很多
/// print(f"趋势系数: {result}")  # 对于这个例子，应该非常接近1.0
/// ```
#[pyfunction]
#[pyo3(signature = (arr))]
pub fn trend_fast(arr: PyReadonlyArray1<f64>) -> PyResult<f64> {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx") {
            unsafe {
                return trend_fast_avx(arr);
            }
        }
    }

    // 如果不支持AVX或不是x86_64架构，回退到标量版本
    trend_fast_scalar(arr)
}

/// AVX-optimized implementation of trend_fast
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn trend_fast_avx(arr: PyReadonlyArray1<f64>) -> PyResult<f64> {
    let x = arr.as_array();
    let n = x.len();

    if n == 0 {
        return Ok(0.0);
    }

    // 预计算一些常量
    let n_f64 = n as f64;
    let var_y = (n_f64 * n_f64 - 1.0) / 12.0; // 自然数序列的方差有解析解

    // 使用AVX指令，每次处理4个双精度数
    const CHUNK_SIZE: usize = 4;
    let main_iter = n / CHUNK_SIZE;
    let remainder = n % CHUNK_SIZE;

    // 初始化SIMD寄存器
    let mut sum_x = _mm256_setzero_pd();
    let mut sum_xy = _mm256_setzero_pd();
    let mut sum_x2 = _mm256_setzero_pd();

    // 主循环，每次处理4个元素
    for chunk in 0..main_iter {
        let base_idx = chunk * CHUNK_SIZE;

        // 加载4个连续的元素到AVX寄存器
        let x_vec = _mm256_loadu_pd(x.as_ptr().add(base_idx));

        // 生成自然数序列 [i+1, i+2, i+3, i+4]
        let indices = _mm256_set_pd(
            (base_idx + 4) as f64,
            (base_idx + 3) as f64,
            (base_idx + 2) as f64,
            (base_idx + 1) as f64,
        );

        // 累加x值
        sum_x = _mm256_add_pd(sum_x, x_vec);

        // 计算与自然数序列的乘积
        sum_xy = _mm256_add_pd(sum_xy, _mm256_mul_pd(x_vec, indices));

        // 计算平方和
        sum_x2 = _mm256_add_pd(sum_x2, _mm256_mul_pd(x_vec, x_vec));
    }

    // 水平求和AVX寄存器
    let mut sum_x_arr = [0.0f64; 4];
    let mut sum_xy_arr = [0.0f64; 4];
    let mut sum_x2_arr = [0.0f64; 4];

    _mm256_storeu_pd(sum_x_arr.as_mut_ptr(), sum_x);
    _mm256_storeu_pd(sum_xy_arr.as_mut_ptr(), sum_xy);
    _mm256_storeu_pd(sum_x2_arr.as_mut_ptr(), sum_x2);

    let mut total_sum_x = sum_x_arr.iter().sum::<f64>();
    let mut total_sum_xy = sum_xy_arr.iter().sum::<f64>();
    let mut total_sum_x2 = sum_x2_arr.iter().sum::<f64>();

    // 处理剩余元素
    let start_remainder = main_iter * CHUNK_SIZE;
    for i in 0..remainder {
        let idx = start_remainder + i;
        let xi = x[idx];
        total_sum_x += xi;
        total_sum_xy += xi * (idx + 1) as f64;
        total_sum_x2 += xi * xi;
    }

    // 计算均值
    let mean_x = total_sum_x / n_f64;

    // 计算协方差和方差
    let covariance = total_sum_xy - mean_x * n_f64 * (n_f64 + 1.0) / 2.0;
    let var_x = total_sum_x2 - mean_x * mean_x * n_f64;

    // 避免除以零
    if var_x == 0.0 || var_y == 0.0 {
        return Ok(0.0);
    }

    // 计算相关系数
    Ok(covariance / (var_x.sqrt() * var_y.sqrt()))
}

/// Scalar fallback implementation of trend_fast
fn trend_fast_scalar(arr: PyReadonlyArray1<f64>) -> PyResult<f64> {
    let x = arr.as_array();
    let n = x.len();

    if n == 0 {
        return Ok(0.0);
    }

    // 预计算一些常量
    let n_f64 = n as f64;
    let var_y = (n_f64 * n_f64 - 1.0) / 12.0; // 自然数序列的方差有解析解

    // 使用L1缓存友好的块大小
    const CHUNK_SIZE: usize = 16; // 通常L1缓存行大小为64字节，一个f64是8字节
    let main_iter = n / CHUNK_SIZE;
    let remainder = n % CHUNK_SIZE;

    let mut sum_x = 0.0;
    let mut sum_xy = 0.0;
    let mut sum_x2 = 0.0;

    // 主循环，每次处理16个元素
    for chunk in 0..main_iter {
        let base_idx = chunk * CHUNK_SIZE;
        let mut chunk_sum_x = 0.0;
        let mut chunk_sum_xy = 0.0;
        let mut chunk_sum_x2 = 0.0;

        // 在每个块内使用展开的循环
        // 将16个元素分成4组，每组4个元素
        for i in 0..4 {
            let offset = i * 4;
            let idx = base_idx + offset;

            // 加载4个连续的元素
            let x0 = x[idx];
            let x1 = x[idx + 1];
            let x2 = x[idx + 2];
            let x3 = x[idx + 3];

            // 累加x值
            chunk_sum_x += x0 + x1 + x2 + x3;

            // 计算与自然数序列的乘积
            chunk_sum_xy += x0 * (idx + 1) as f64
                + x1 * (idx + 2) as f64
                + x2 * (idx + 3) as f64
                + x3 * (idx + 4) as f64;

            // 计算平方和
            chunk_sum_x2 += x0 * x0 + x1 * x1 + x2 * x2 + x3 * x3;
        }

        // 更新全局累加器
        sum_x += chunk_sum_x;
        sum_xy += chunk_sum_xy;
        sum_x2 += chunk_sum_x2;
    }

    // 处理剩余元素
    let start_remainder = main_iter * CHUNK_SIZE;
    for i in 0..remainder {
        let idx = start_remainder + i;
        let xi = x[idx];
        sum_x += xi;
        sum_xy += xi * (idx + 1) as f64;
        sum_x2 += xi * xi;
    }

    // 计算均值
    let mean_x = sum_x / n_f64;

    // 计算协方差和方差
    let covariance = sum_xy - mean_x * n_f64 * (n_f64 + 1.0) / 2.0;
    let var_x = sum_x2 - mean_x * mean_x * n_f64;

    // 避免除以零
    if var_x == 0.0 || var_y == 0.0 {
        return Ok(0.0);
    }

    // 计算相关系数
    Ok(covariance / (var_x.sqrt() * var_y.sqrt()))
}

/// 计算二维数组各行或各列的趋势性
///
/// 参数说明：
/// ----------
/// arr : numpy.ndarray
///     二维数组
/// axis : int
///     计算轴，0表示对每列计算趋势，1表示对每行计算趋势
///
/// 返回值：
/// -------
/// numpy.ndarray
///     一维数组，包含每行或每列的趋势值
///
/// Python调用示例：
/// ```python
/// import numpy as np
/// from rust_pyfunc import trend_2d
///
/// # 创建示例数据
/// data = np.array([[1.0, 2.0, 3.0, 4.0],
///                  [4.0, 3.0, 2.0, 1.0],
///                  [1.0, 3.0, 2.0, 4.0]])
///
/// # 计算每行的趋势
/// row_trends = trend_2d(data, axis=1)
///
/// # 计算每列的趋势
/// col_trends = trend_2d(data, axis=0)
/// ```
#[pyfunction]
pub fn trend_2d(arr: PyReadonlyArray2<f64>, axis: i32) -> PyResult<Vec<f64>> {
    let arr = arr.as_array();
    let (rows, cols) = arr.dim();

    let mut results = Vec::new();

    match axis {
        0 => {
            // 对每列计算趋势
            for col in 0..cols {
                let col_data: Vec<f64> = (0..rows).map(|row| arr[[row, col]]).collect();
                let trend_val = calculate_trend_1d(&col_data);
                results.push(trend_val);
            }
        }
        1 => {
            // 对每行计算趋势
            for row in 0..rows {
                let row_data: Vec<f64> = (0..cols).map(|col| arr[[row, col]]).collect();
                let trend_val = calculate_trend_1d(&row_data);
                results.push(trend_val);
            }
        }
        _ => {
            return Err(PyValueError::new_err("axis must be 0 or 1"));
        }
    }

    Ok(results)
}

/// 计算一维数组的趋势性（内部辅助函数）
fn calculate_trend_1d(data: &[f64]) -> f64 {
    // 过滤掉NaN值，同时保持对应的索引
    let valid_pairs: Vec<(f64, f64)> = data
        .iter()
        .enumerate()
        .filter_map(|(i, &val)| {
            if val.is_finite() {
                Some((val, (i + 1) as f64))
            } else {
                None
            }
        })
        .collect();

    let n = valid_pairs.len();

    if n == 0 || n == 1 {
        return 0.0;
    }

    // 分离有效的值和对应的索引
    let (values, indices): (Vec<f64>, Vec<f64>) = valid_pairs.into_iter().unzip();

    // 计算均值
    let mean_x: f64 = values.iter().sum::<f64>() / n as f64;
    let mean_y: f64 = indices.iter().sum::<f64>() / n as f64;

    // 计算协方差和标准差
    let mut covariance: f64 = 0.0;
    let mut var_x: f64 = 0.0;
    let mut var_y: f64 = 0.0;

    for i in 0..n {
        let diff_x = values[i] - mean_x;
        let diff_y = indices[i] - mean_y;

        covariance += diff_x * diff_y;
        var_x += diff_x * diff_x;
        var_y += diff_y * diff_y;
    }

    // 避免除以零
    if var_x == 0.0 || var_y == 0.0 {
        return 0.0;
    }

    // 计算相关系数
    covariance / (var_x.sqrt() * var_y.sqrt())
}
