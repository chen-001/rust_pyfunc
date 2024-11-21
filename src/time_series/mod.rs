use pyo3::prelude::*;
use pyo3::types::PyList;
use numpy::PyReadonlyArray1;
use ndarray::Array1;
use std::collections::HashMap;

/// 计算两个序列之间的动态时间规整(DTW)距离。
/// DTW是一种衡量两个时间序列相似度的算法，可以处理不等长的序列。
/// 它通过寻找两个序列之间的最佳对齐方式来计算距离。
///
/// 参数说明：
/// ----------
/// s1 : array_like
///     第一个输入序列
/// s2 : array_like
///     第二个输入序列
/// radius : int, optional
///     Sakoe-Chiba半径，用于限制规整路径，可以提高计算效率。
///     如果不指定，则不使用路径限制。
///
/// 返回值：
/// -------
/// float
///     两个序列之间的DTW距离，值越小表示序列越相似
///
/// Python调用示例：
/// ```python
/// from rust_pyfunc import dtw_distance
///
/// # 计算两个序列的DTW距离
/// s1 = [1.0, 2.0, 3.0, 4.0]
/// s2 = [1.0, 2.0, 2.5, 3.0, 4.0]
/// 
/// # 不使用radius限制
/// dist1 = dtw_distance(s1, s2)
/// print(f"不限制路径的DTW距离: {dist1}")
///
/// # 使用radius=1限制规整路径
/// dist2 = dtw_distance(s1, s2, radius=1)
/// print(f"使用radius=1的DTW距离: {dist2}")
/// ```
#[pyfunction]
#[pyo3(signature = (s1, s2, radius=None))]
pub fn dtw_distance(s1: Vec<f64>, s2: Vec<f64>, radius: Option<usize>) -> PyResult<f64> {
    // let radius_after_default = set_c(radius);
    let len_s1 = s1.len();
    let len_s2 = s2.len();
    let mut warp_dist_mat = vec![vec![f64::INFINITY; len_s2 + 1]; len_s1 + 1];
    warp_dist_mat[0][0] = 0.0;

    for i in 1..=len_s1 {
        for j in 1..=len_s2 {
            match radius {
                Some(_) => {
                    if !sakoe_chiba_window(i, j, radius.unwrap()) {
                        continue;
                    }
                }
                None => {}
            }
            let cost = (s1[i - 1] - s2[j - 1]).abs() as f64;
            warp_dist_mat[i][j] = cost
                + warp_dist_mat[i - 1][j]
                    .min(warp_dist_mat[i][j - 1].min(warp_dist_mat[i - 1][j - 1]));
        }
    }
    Ok(warp_dist_mat[len_s1][len_s2])
}


/// 计算从序列x到序列y的转移熵（Transfer Entropy）。
/// 转移熵衡量了一个时间序列对另一个时间序列的影响程度，是一种非线性的因果关系度量。
/// 具体来说，它测量了在已知x的过去k个状态的情况下，对y的当前状态预测能力的提升程度。
///
/// 参数说明：
/// ----------
/// x_ : array_like
///     源序列，用于预测目标序列
/// y_ : array_like
///     目标序列，我们要预测的序列
/// k : int
///     历史长度，考虑过去k个时间步的状态
/// c : int
///     离散化的类别数，将连续值离散化为c个等级
///
/// 返回值：
/// -------
/// float
///     从x到y的转移熵值。值越大表示x对y的影响越大。
///
/// Python调用示例：
/// ```python
/// import numpy as np
/// from rust_pyfunc import transfer_entropy
///
/// # 创建两个相关的时间序列
/// x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
/// y = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])  # y比x滞后一个时间步
///
/// # 计算转移熵
/// k = 2  # 考虑过去2个时间步
/// c = 4  # 将数据离散化为4个等级
/// te = transfer_entropy(x, y, k, c)
/// print(f"从x到y的转移熵: {te}")  # 应该得到一个正值，表示x确实影响y
///
/// # 反向计算
/// te_reverse = transfer_entropy(y, x, k, c)
/// print(f"从y到x的转移熵: {te_reverse}")  # 应该比te小，因为y不影响x
/// ```
#[pyfunction]
#[pyo3(signature = (x_, y_, k, c))]
pub fn transfer_entropy(x_: Vec<f64>, y_: Vec<f64>, k: usize, c: usize) -> f64 {
    let x = discretize(x_, c);
    let y = discretize(y_, c);
    let n = x.len();
    let mut joint_prob = HashMap::new();
    let mut conditional_prob = HashMap::new();
    let mut marginal_prob = HashMap::new();

    // 计算联合概率 p(x_{t-k}, y_t)
    for t in k..n {
        let key = (format!("{:.6}", x[t - k]), format!("{:.6}", y[t]));
        *joint_prob.entry(key).or_insert(0) += 1;
        *marginal_prob.entry(format!("{:.6}", y[t])).or_insert(0) += 1;
    }

    // 计算条件概率 p(y_t | x_{t-k})
    for t in k..n {
        let key = (format!("{:.6}", x[t - k]), format!("{:.6}", y[t]));
        let count = joint_prob.get(&key).unwrap_or(&0);
        let conditional_key = format!("{:.6}", x[t - k]);

        // 计算条件概率
        if let Some(total_count) = marginal_prob.get(&conditional_key) {
            let prob = *count as f64 / *total_count as f64;
            *conditional_prob
                .entry((conditional_key.clone(), format!("{:.6}", y[t])))
                .or_insert(0.0) += prob;
        }
    }

    // 计算转移熵
    let mut te = 0.0;
    for (key, &count) in joint_prob.iter() {
        let (x_state, y_state) = key;
        let p_xy = count as f64 / (n - k) as f64;
        let p_y_given_x = conditional_prob
            .get(&(x_state.clone(), y_state.clone()))
            .unwrap_or(&0.0);
        let p_y = marginal_prob.get(y_state).unwrap_or(&0);

        if *p_y > 0 {
            te += p_xy * (p_y_given_x / *p_y as f64).log2();
        }
    }

    te
}


#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

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
                    "List elements must be numeric (float or int)"
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

    let n = arr_vec.len();
    
    if n == 0 {
        return Ok(0.0);
    }

    // 创建自然数序列 1,2,3...n
    let natural_seq: Vec<f64> = (1..=n).map(|x| x as f64).collect();

    // 计算均值
    let mean_x: f64 = arr_vec.iter().sum::<f64>() / n as f64;
    let mean_y: f64 = natural_seq.iter().sum::<f64>() / n as f64;

    // 计算协方差和标准差
    let mut covariance: f64 = 0.0;
    let mut var_x: f64 = 0.0;
    let mut var_y: f64 = 0.0;

    for i in 0..n {
        let diff_x = arr_vec[i] - mean_x;
        let diff_y = natural_seq[i] - mean_y;
        
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
    if is_x86_feature_detected!("avx") {
        unsafe {
            return trend_fast_avx(arr);
        }
    }
    
    // 如果不支持AVX，回退到标量版本
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
    let var_y = (n_f64 * n_f64 - 1.0) / 12.0;  // 自然数序列的方差有解析解

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
            (base_idx + 1) as f64
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
    let var_y = (n_f64 * n_f64 - 1.0) / 12.0;  // 自然数序列的方差有解析解

    // 使用L1缓存友好的块大小
    const CHUNK_SIZE: usize = 16;  // 通常L1缓存行大小为64字节，一个f64是8字节
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


// fn set_k(b: Option<usize>) -> usize {
//     match b {
//         Some(value) => value, // 如果b不是None，则c等于b的值加1
//         None => 2,            // 如果b是None，则c等于1
//     }
// }


fn sakoe_chiba_window(i: usize, j: usize, radius: usize) -> bool {
    (i.saturating_sub(radius) <= j) && (j <= i + radius)
}


/// Discretizes a sequence of numbers into c categories.
///
/// Parameters
/// ----------
/// data_ : array_like
///     The input sequence.
/// c : int
///     The number of categories.
///
/// Returns
/// -------
/// Array1<f64>
///     The discretized sequence.
fn discretize(data_: Vec<f64>, c: usize) -> Array1<f64> {
    let data = Array1::from_vec(data_);
    let mut sorted_indices: Vec<usize> = (0..data.len()).collect();
    sorted_indices.sort_by(|&i, &j| data[i].partial_cmp(&data[j]).unwrap());

    let mut discretized = Array1::zeros(data.len());
    let chunk_size = data.len() / c;

    for i in 0..c {
        let start = i * chunk_size;
        let end = if i == c - 1 {
            data.len()
        } else {
            (i + 1) * chunk_size
        };
        for j in start..end {
            discretized[sorted_indices[j]] = i + 1; // 类别从 1 开始
        }
    }
    let discretized_f64: Array1<f64> =
        Array1::from(discretized.iter().map(|&x| x as f64).collect::<Vec<f64>>());

    discretized_f64
}