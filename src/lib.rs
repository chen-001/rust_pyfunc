// use pyo3::prelude::*;
// use pyo3::types::{PyList, PyModule};
// use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
// use ndarray::{s, Array1, Array2, ArrayView1, ArrayView2};
// use std::collections::{HashMap, HashSet};

// #[cfg(target_arch = "x86_64")]
// use core::arch::x86_64::*;

// /// 计算输入数组与自然数序列(1, 2, ..., n)之间的皮尔逊相关系数。
// /// 这个函数可以用来判断一个序列的趋势性，如果返回值接近1表示强上升趋势，接近-1表示强下降趋势。
// ///
// /// 参数说明：
// /// ----------
// /// arr : 输入数组
// ///     可以是以下类型之一：
// ///     - numpy.ndarray (float64或int64类型)
// ///     - Python列表 (float或int类型)
// ///
// /// 返回值：
// /// -------
// /// float
// ///     输入数组与自然数序列的皮尔逊相关系数。
// ///     如果输入数组为空或方差为零，则返回0.0。
// ///
// /// Python调用示例：
// /// ```python
// /// import numpy as np
// /// from rust_pyfunc import trend
// ///
// /// # 使用numpy数组
// /// arr1 = np.array([1.0, 2.0, 3.0, 4.0])  # 完美上升趋势
// /// result1 = trend(arr1)  # 返回接近1.0
// ///
// /// # 使用Python列表
// /// arr2 = [4, 3, 2, 1]  # 完美下降趋势
// /// result2 = trend(arr2)  # 返回接近-1.0
// ///
// /// # 无趋势序列
// /// arr3 = [1, 1, 1, 1]
// /// result3 = trend(arr3)  # 返回0.0
// /// ```
// #[pyfunction]
// #[pyo3(signature = (arr))]
// fn trend(arr: &PyAny) -> PyResult<f64> {
//     let py = arr.py();
    
//     // 尝试将输入转换为Vec<f64>
//     let arr_vec: Vec<f64> = if arr.is_instance_of::<PyList>()? {
//         let list = arr.downcast::<PyList>()?;
//         let mut result = Vec::with_capacity(list.len());
//         for item in list.iter() {
//             if let Ok(val) = item.extract::<f64>() {
//                 result.push(val);
//             } else if let Ok(val) = item.extract::<i64>() {
//                 result.push(val as f64);
//             } else {
//                 return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
//                     "List elements must be numeric (float or int)"
//                 ));
//             }
//         }
//         result
//     } else {
//         // 尝试将输入转换为numpy数组
//         let numpy = py.import("numpy")?;
//         let arr = numpy.call_method1("asarray", (arr,))?;
//         let arr = arr.call_method1("astype", ("float64",))?;
//         arr.extract::<Vec<f64>>()?
//     };

//     let n = arr_vec.len();
    
//     if n == 0 {
//         return Ok(0.0);
//     }

//     // 创建自然数序列 1,2,3...n
//     let natural_seq: Vec<f64> = (1..=n).map(|x| x as f64).collect();

//     // 计算均值
//     let mean_x: f64 = arr_vec.iter().sum::<f64>() / n as f64;
//     let mean_y: f64 = natural_seq.iter().sum::<f64>() / n as f64;

//     // 计算协方差和标准差
//     let mut covariance: f64 = 0.0;
//     let mut var_x: f64 = 0.0;
//     let mut var_y: f64 = 0.0;

//     for i in 0..n {
//         let diff_x = arr_vec[i] - mean_x;
//         let diff_y = natural_seq[i] - mean_y;
        
//         covariance += diff_x * diff_y;
//         var_x += diff_x * diff_x;
//         var_y += diff_y * diff_y;
//     }

//     // 避免除以零
//     if var_x == 0.0 || var_y == 0.0 {
//         return Ok(0.0);
//     }

//     // 计算相关系数
//     let correlation = covariance / (var_x.sqrt() * var_y.sqrt());
    
//     Ok(correlation)
// }

// /// 这是trend函数的高性能版本，专门用于处理numpy.ndarray类型的float64数组。
// /// 使用了显式的SIMD指令和缓存优化处理，比普通版本更快。
// ///
// /// 参数说明：
// /// ----------
// /// arr : numpy.ndarray
// ///     输入数组，必须是float64类型
// ///
// /// 返回值：
// /// -------
// /// float
// ///     输入数组与自然数序列的皮尔逊相关系数
// ///
// /// Python调用示例：
// /// ```python
// /// import numpy as np
// /// from rust_pyfunc import trend_fast
// ///
// /// # 创建一个大型数组进行测试
// /// arr = np.array([float(i) for i in range(1000000)], dtype=np.float64)
// /// result = trend_fast(arr)  # 会比trend函数快很多
// /// print(f"趋势系数: {result}")  # 对于这个例子，应该非常接近1.0
// /// ```
// #[pyfunction]
// #[pyo3(signature = (arr))]
// fn trend_fast(arr: PyReadonlyArray1<f64>) -> PyResult<f64> {
//     if is_x86_feature_detected!("avx") {
//         unsafe {
//             return trend_fast_avx(arr);
//         }
//     }
    
//     // 如果不支持AVX，回退到标量版本
//     trend_fast_scalar(arr)
// }

// /// AVX-optimized implementation of trend_fast
// #[cfg(target_arch = "x86_64")]
// #[target_feature(enable = "avx")]
// unsafe fn trend_fast_avx(arr: PyReadonlyArray1<f64>) -> PyResult<f64> {
//     let x = arr.as_array();
//     let n = x.len();
    
//     if n == 0 {
//         return Ok(0.0);
//     }

//     // 预计算一些常量
//     let n_f64 = n as f64;
//     let var_y = (n_f64 * n_f64 - 1.0) / 12.0;  // 自然数序列的方差有解析解

//     // 使用AVX指令，每次处理4个双精度数
//     const CHUNK_SIZE: usize = 4;
//     let main_iter = n / CHUNK_SIZE;
//     let remainder = n % CHUNK_SIZE;

//     // 初始化SIMD寄存器
//     let mut sum_x = _mm256_setzero_pd();
//     let mut sum_xy = _mm256_setzero_pd();
//     let mut sum_x2 = _mm256_setzero_pd();

//     // 主循环，每次处理4个元素
//     for chunk in 0..main_iter {
//         let base_idx = chunk * CHUNK_SIZE;
        
//         // 加载4个连续的元素到AVX寄存器
//         let x_vec = _mm256_loadu_pd(x.as_ptr().add(base_idx));
        
//         // 生成自然数序列 [i+1, i+2, i+3, i+4]
//         let indices = _mm256_set_pd(
//             (base_idx + 4) as f64,
//             (base_idx + 3) as f64,
//             (base_idx + 2) as f64,
//             (base_idx + 1) as f64
//         );

//         // 累加x值
//         sum_x = _mm256_add_pd(sum_x, x_vec);
        
//         // 计算与自然数序列的乘积
//         sum_xy = _mm256_add_pd(sum_xy, _mm256_mul_pd(x_vec, indices));
        
//         // 计算平方和
//         sum_x2 = _mm256_add_pd(sum_x2, _mm256_mul_pd(x_vec, x_vec));
//     }

//     // 水平求和AVX寄存器
//     let mut sum_x_arr = [0.0f64; 4];
//     let mut sum_xy_arr = [0.0f64; 4];
//     let mut sum_x2_arr = [0.0f64; 4];
    
//     _mm256_storeu_pd(sum_x_arr.as_mut_ptr(), sum_x);
//     _mm256_storeu_pd(sum_xy_arr.as_mut_ptr(), sum_xy);
//     _mm256_storeu_pd(sum_x2_arr.as_mut_ptr(), sum_x2);

//     let mut total_sum_x = sum_x_arr.iter().sum::<f64>();
//     let mut total_sum_xy = sum_xy_arr.iter().sum::<f64>();
//     let mut total_sum_x2 = sum_x2_arr.iter().sum::<f64>();

//     // 处理剩余元素
//     let start_remainder = main_iter * CHUNK_SIZE;
//     for i in 0..remainder {
//         let idx = start_remainder + i;
//         let xi = x[idx];
//         total_sum_x += xi;
//         total_sum_xy += xi * (idx + 1) as f64;
//         total_sum_x2 += xi * xi;
//     }

//     // 计算均值
//     let mean_x = total_sum_x / n_f64;

//     // 计算协方差和方差
//     let covariance = total_sum_xy - mean_x * n_f64 * (n_f64 + 1.0) / 2.0;
//     let var_x = total_sum_x2 - mean_x * mean_x * n_f64;

//     // 避免除以零
//     if var_x == 0.0 || var_y == 0.0 {
//         return Ok(0.0);
//     }

//     // 计算相关系数
//     Ok(covariance / (var_x.sqrt() * var_y.sqrt()))
// }

// /// Scalar fallback implementation of trend_fast
// fn trend_fast_scalar(arr: PyReadonlyArray1<f64>) -> PyResult<f64> {
//     let x = arr.as_array();
//     let n = x.len();
    
//     if n == 0 {
//         return Ok(0.0);
//     }

//     // 预计算一些常量
//     let n_f64 = n as f64;
//     let var_y = (n_f64 * n_f64 - 1.0) / 12.0;  // 自然数序列的方差有解析解

//     // 使用L1缓存友好的块大小
//     const CHUNK_SIZE: usize = 16;  // 通常L1缓存行大小为64字节，一个f64是8字节
//     let main_iter = n / CHUNK_SIZE;
//     let remainder = n % CHUNK_SIZE;

//     let mut sum_x = 0.0;
//     let mut sum_xy = 0.0;
//     let mut sum_x2 = 0.0;

//     // 主循环，每次处理16个元素
//     for chunk in 0..main_iter {
//         let base_idx = chunk * CHUNK_SIZE;
//         let mut chunk_sum_x = 0.0;
//         let mut chunk_sum_xy = 0.0;
//         let mut chunk_sum_x2 = 0.0;

//         // 在每个块内使用展开的循环
//         // 将16个元素分成4组，每组4个元素
//         for i in 0..4 {
//             let offset = i * 4;
//             let idx = base_idx + offset;
            
//             // 加载4个连续的元素
//             let x0 = x[idx];
//             let x1 = x[idx + 1];
//             let x2 = x[idx + 2];
//             let x3 = x[idx + 3];

//             // 累加x值
//             chunk_sum_x += x0 + x1 + x2 + x3;

//             // 计算与自然数序列的乘积
//             chunk_sum_xy += x0 * (idx + 1) as f64
//                          + x1 * (idx + 2) as f64
//                          + x2 * (idx + 3) as f64
//                          + x3 * (idx + 4) as f64;

//             // 计算平方和
//             chunk_sum_x2 += x0 * x0 + x1 * x1 + x2 * x2 + x3 * x3;
//         }

//         // 更新全局累加器
//         sum_x += chunk_sum_x;
//         sum_xy += chunk_sum_xy;
//         sum_x2 += chunk_sum_x2;
//     }

//     // 处理剩余元素
//     let start_remainder = main_iter * CHUNK_SIZE;
//     for i in 0..remainder {
//         let idx = start_remainder + i;
//         let xi = x[idx];
//         sum_x += xi;
//         sum_xy += xi * (idx + 1) as f64;
//         sum_x2 += xi * xi;
//     }

//     // 计算均值
//     let mean_x = sum_x / n_f64;

//     // 计算协方差和方差
//     let covariance = sum_xy - mean_x * n_f64 * (n_f64 + 1.0) / 2.0;
//     let var_x = sum_x2 - mean_x * mean_x * n_f64;

//     // 避免除以零
//     if var_x == 0.0 || var_y == 0.0 {
//         return Ok(0.0);
//     }

//     // 计算相关系数
//     Ok(covariance / (var_x.sqrt() * var_y.sqrt()))
// }

// /// Formats the sum of two numbers as string.
// #[pyfunction]
// #[pyo3(signature = (a, b))]
// fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
//     Ok((a + b).to_string())
// }

// // fn set_k(b: Option<usize>) -> usize {
// //     match b {
// //         Some(value) => value, // 如果b不是None，则c等于b的值加1
// //         None => 2,            // 如果b是None，则c等于1
// //     }
// // }


// fn sakoe_chiba_window(i: usize, j: usize, radius: usize) -> bool {
//     (i.saturating_sub(radius) <= j) && (j <= i + radius)
// }

// /// 计算两个序列之间的动态时间规整(DTW)距离。
// /// DTW是一种衡量两个时间序列相似度的算法，可以处理不等长的序列。
// /// 它通过寻找两个序列之间的最佳对齐方式来计算距离。
// ///
// /// 参数说明：
// /// ----------
// /// s1 : array_like
// ///     第一个输入序列
// /// s2 : array_like
// ///     第二个输入序列
// /// radius : int, optional
// ///     Sakoe-Chiba半径，用于限制规整路径，可以提高计算效率。
// ///     如果不指定，则不使用路径限制。
// ///
// /// 返回值：
// /// -------
// /// float
// ///     两个序列之间的DTW距离，值越小表示序列越相似
// ///
// /// Python调用示例：
// /// ```python
// /// from rust_pyfunc import dtw_distance
// ///
// /// # 计算两个序列的DTW距离
// /// s1 = [1.0, 2.0, 3.0, 4.0]
// /// s2 = [1.0, 2.0, 2.5, 3.0, 4.0]
// /// 
// /// # 不使用radius限制
// /// dist1 = dtw_distance(s1, s2)
// /// print(f"不限制路径的DTW距离: {dist1}")
// ///
// /// # 使用radius=1限制规整路径
// /// dist2 = dtw_distance(s1, s2, radius=1)
// /// print(f"使用radius=1的DTW距离: {dist2}")
// /// ```
// #[pyfunction]
// #[pyo3(signature = (s1, s2, radius=None))]
// fn dtw_distance(s1: Vec<f64>, s2: Vec<f64>, radius: Option<usize>) -> PyResult<f64> {
//     // let radius_after_default = set_c(radius);
//     let len_s1 = s1.len();
//     let len_s2 = s2.len();
//     let mut warp_dist_mat = vec![vec![f64::INFINITY; len_s2 + 1]; len_s1 + 1];
//     warp_dist_mat[0][0] = 0.0;

//     for i in 1..=len_s1 {
//         for j in 1..=len_s2 {
//             match radius {
//                 Some(_) => {
//                     if !sakoe_chiba_window(i, j, radius.unwrap()) {
//                         continue;
//                     }
//                 }
//                 None => {}
//             }
//             let cost = (s1[i - 1] - s2[j - 1]).abs() as f64;
//             warp_dist_mat[i][j] = cost
//                 + warp_dist_mat[i - 1][j]
//                     .min(warp_dist_mat[i][j - 1].min(warp_dist_mat[i - 1][j - 1]));
//         }
//     }
//     Ok(warp_dist_mat[len_s1][len_s2])
// }

// /// Discretizes a sequence of numbers into c categories.
// ///
// /// Parameters
// /// ----------
// /// data_ : array_like
// ///     The input sequence.
// /// c : int
// ///     The number of categories.
// ///
// /// Returns
// /// -------
// /// Array1<f64>
// ///     The discretized sequence.
// fn discretize(data_: Vec<f64>, c: usize) -> Array1<f64> {
//     let data = Array1::from_vec(data_);
//     let mut sorted_indices: Vec<usize> = (0..data.len()).collect();
//     sorted_indices.sort_by(|&i, &j| data[i].partial_cmp(&data[j]).unwrap());

//     let mut discretized = Array1::zeros(data.len());
//     let chunk_size = data.len() / c;

//     for i in 0..c {
//         let start = i * chunk_size;
//         let end = if i == c - 1 {
//             data.len()
//         } else {
//             (i + 1) * chunk_size
//         };
//         for j in start..end {
//             discretized[sorted_indices[j]] = i + 1; // 类别从 1 开始
//         }
//     }
//     let discretized_f64: Array1<f64> =
//         Array1::from(discretized.iter().map(|&x| x as f64).collect::<Vec<f64>>());

//     discretized_f64
// }

// /// 计算从序列x到序列y的转移熵（Transfer Entropy）。
// /// 转移熵衡量了一个时间序列对另一个时间序列的影响程度，是一种非线性的因果关系度量。
// /// 具体来说，它测量了在已知x的过去k个状态的情况下，对y的当前状态预测能力的提升程度。
// ///
// /// 参数说明：
// /// ----------
// /// x_ : array_like
// ///     源序列，用于预测目标序列
// /// y_ : array_like
// ///     目标序列，我们要预测的序列
// /// k : int
// ///     历史长度，考虑过去k个时间步的状态
// /// c : int
// ///     离散化的类别数，将连续值离散化为c个等级
// ///
// /// 返回值：
// /// -------
// /// float
// ///     从x到y的转移熵值。值越大表示x对y的影响越大。
// ///
// /// Python调用示例：
// /// ```python
// /// import numpy as np
// /// from rust_pyfunc import transfer_entropy
// ///
// /// # 创建两个相关的时间序列
// /// x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
// /// y = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])  # y比x滞后一个时间步
// ///
// /// # 计算转移熵
// /// k = 2  # 考虑过去2个时间步
// /// c = 4  # 将数据离散化为4个等级
// /// te = transfer_entropy(x, y, k, c)
// /// print(f"从x到y的转移熵: {te}")  # 应该得到一个正值，表示x确实影响y
// ///
// /// # 反向计算
// /// te_reverse = transfer_entropy(y, x, k, c)
// /// print(f"从y到x的转移熵: {te_reverse}")  # 应该比te小，因为y不影响x
// /// ```
// #[pyfunction]
// #[pyo3(signature = (x_, y_, k, c))]
// fn transfer_entropy(x_: Vec<f64>, y_: Vec<f64>, k: usize, c: usize) -> f64 {
//     let x = discretize(x_, c);
//     let y = discretize(y_, c);
//     let n = x.len();
//     let mut joint_prob = HashMap::new();
//     let mut conditional_prob = HashMap::new();
//     let mut marginal_prob = HashMap::new();

//     // 计算联合概率 p(x_{t-k}, y_t)
//     for t in k..n {
//         let key = (format!("{:.6}", x[t - k]), format!("{:.6}", y[t]));
//         *joint_prob.entry(key).or_insert(0) += 1;
//         *marginal_prob.entry(format!("{:.6}", y[t])).or_insert(0) += 1;
//     }

//     // 计算条件概率 p(y_t | x_{t-k})
//     for t in k..n {
//         let key = (format!("{:.6}", x[t - k]), format!("{:.6}", y[t]));
//         let count = joint_prob.get(&key).unwrap_or(&0);
//         let conditional_key = format!("{:.6}", x[t - k]);

//         // 计算条件概率
//         if let Some(total_count) = marginal_prob.get(&conditional_key) {
//             let prob = *count as f64 / *total_count as f64;
//             *conditional_prob
//                 .entry((conditional_key.clone(), format!("{:.6}", y[t])))
//                 .or_insert(0.0) += prob;
//         }
//     }

//     // 计算转移熵
//     let mut te = 0.0;
//     for (key, &count) in joint_prob.iter() {
//         let (x_state, y_state) = key;
//         let p_xy = count as f64 / (n - k) as f64;
//         let p_y_given_x = conditional_prob
//             .get(&(x_state.clone(), y_state.clone()))
//             .unwrap_or(&0.0);
//         let p_y = marginal_prob.get(y_state).unwrap_or(&0);

//         if *p_y > 0 {
//             te += p_xy * (p_y_given_x / *p_y as f64).log2();
//         }
//     }

//     te
// }

// /// 普通最小二乘(OLS)回归。
// /// 用于拟合线性回归模型 y = Xβ + ε，其中β是要估计的回归系数。
// ///
// /// 参数说明：
// /// ----------
// /// x : numpy.ndarray
// ///     设计矩阵，形状为(n_samples, n_features)
// /// y : numpy.ndarray
// ///     响应变量，形状为(n_samples,)
// /// calculate_r2 : bool, optional
// ///     是否计算R²值，默认为True
// ///
// /// 返回值：
// /// -------
// /// numpy.ndarray
// ///     回归系数β
// ///
// /// Python调用示例：
// /// ```python
// /// import numpy as np
// /// from rust_pyfunc import ols
// ///
// /// # 准备训练数据
// /// X = np.array([[1, 1], [1, 2], [1, 3]], dtype=np.float64)  # 包含一个常数项和一个特征
// /// y = np.array([2, 4, 6], dtype=np.float64)  # 目标变量
// ///
// /// # 拟合模型
// /// coefficients = ols(X, y)
// /// print(f"回归系数: {coefficients}")  # 预期输出接近[0, 2]，表示y ≈ 0 + 2x
// /// ```
// #[pyfunction]
// #[pyo3(signature = (x, y, calculate_r2=true))]
// fn ols(
//     py: Python,
//     x: PyReadonlyArray2<f64>,
//     y: PyReadonlyArray1<f64>,
//     calculate_r2: Option<bool>,
// ) -> PyResult<Py<PyArray1<f64>>> {
//     let x: ArrayView2<f64> = x.as_array();
//     let y: ArrayView1<f64> = y.as_array();

//     // 创建带有截距项的设计矩阵
//     let mut x_with_intercept = Array2::ones((x.nrows(), x.ncols() + 1));
//     x_with_intercept.slice_mut(s![.., 1..]).assign(&x);

//     // 计算 (X^T * X)^(-1) * X^T * y
//     let xt_x = x_with_intercept.t().dot(&x_with_intercept);
//     let xt_y = x_with_intercept.t().dot(&y);
//     let coefficients = solve_linear_system3(&xt_x.view(), &xt_y.view());

//     let mut result = coefficients.to_vec();

//     // 如果需要计算R方
//     if calculate_r2.unwrap_or(true) {
//         // 计算R方
//         let y_mean = y.mean().unwrap();
//         let y_pred = x_with_intercept.dot(&coefficients);
//         let ss_tot: f64 = y.iter().map(|&yi| (yi - y_mean).powi(2)).sum();
//         let ss_res: f64 = (&y - &y_pred).map(|e| e.powi(2)).sum();
//         let r_squared = 1.0 - (ss_res / ss_tot);
//         result.push(r_squared);
//     }

//     // 将结果转换为 Python 数组
//     Ok(Array1::from(result).into_pyarray(py).to_owned())
// }

//     /// 使用已有数据和响应变量，对新的数据点进行OLS线性回归预测。
// ///
// /// 参数说明：
// /// ----------
// /// x : numpy.ndarray
// ///     原始设计矩阵，形状为(n_samples, n_features)
// /// y : numpy.ndarray
// ///     原始响应变量，形状为(n_samples,)
// /// x_pred : numpy.ndarray
// ///     需要预测的新数据点，形状为(m_samples, n_features)
// ///
// /// 返回值：
// /// -------
// /// numpy.ndarray
// ///     预测值，形状为(m_samples,)
// ///
// /// Python调用示例：
// /// ```python
// /// import numpy as np
// /// from rust_pyfunc import ols_predict
// ///
// /// # 准备训练数据
// /// X_train = np.array([[1, 1], [1, 2], [1, 3]], dtype=np.float64)
// /// y_train = np.array([2, 4, 6], dtype=np.float64)
// ///
// /// # 准备预测数据
// /// X_pred = np.array([[1, 4], [1, 5]], dtype=np.float64)
// ///
// /// # 进行预测
// /// predictions = ols_predict(X_train, y_train, X_pred)
// /// print(f"预测值: {predictions}")  # 预期输出接近[8, 10]
// /// ```
// #[pyfunction]
// #[pyo3(signature = (x, y, x_pred))]
// fn ols_predict(
//     py: Python,
//     x: PyReadonlyArray2<f64>,
//     y: PyReadonlyArray1<f64>,
//     x_pred: PyReadonlyArray2<f64>,
// ) -> PyResult<Py<PyArray1<f64>>> {
//     let x: ArrayView2<f64> = x.as_array();
//     let y: ArrayView1<f64> = y.as_array();
//     let x_pred: ArrayView2<f64> = x_pred.as_array();

//     // 创建带有截距项的设计矩阵
//     let mut x_with_intercept = Array2::ones((x.nrows(), x.ncols() + 1));
//     x_with_intercept.slice_mut(s![.., 1..]).assign(&x);

//     // 计算回归系数
//     let xt_x = x_with_intercept.t().dot(&x_with_intercept);
//     let xt_y = x_with_intercept.t().dot(&y);
//     let coefficients = solve_linear_system3(&xt_x.view(), &xt_y.view());

//     // 为预测数据创建带有截距项的设计矩阵
//     let mut x_pred_with_intercept = Array2::ones((x_pred.nrows(), x_pred.ncols() + 1));
//     x_pred_with_intercept.slice_mut(s![.., 1..]).assign(&x_pred);

//     // 计算预测值
//     let predictions = x_pred_with_intercept.dot(&coefficients);

//     // 将预测结果转换为 Python 数组
//     Ok(predictions.into_pyarray(py).to_owned())
// }

// fn solve_linear_system3(a: &ArrayView2<f64>, b: &ArrayView1<f64>) -> Array1<f64> {
//     let mut l = Array2::<f64>::zeros((a.nrows(), a.ncols()));
//     let mut u = Array2::<f64>::zeros((a.nrows(), a.ncols()));

//     // LU decomposition
//     for i in 0..a.nrows() {
//         for j in 0..a.ncols() {
//             if i <= j {
//                 u[[i, j]] = a[[i, j]] - (0..i).map(|k| l[[i, k]] * u[[k, j]]).sum::<f64>();
//                 if i == j {
//                     l[[i, i]] = 1.0;
//                 }
//             }
//             if i > j {
//                 l[[i, j]] =
//                     (a[[i, j]] - (0..j).map(|k| l[[i, k]] * u[[k, j]]).sum::<f64>()) / u[[j, j]];
//             }
//         }
//     }

//     // Forward substitution
//     let mut y = Array1::<f64>::zeros(b.len());
//     for i in 0..b.len() {
//         y[i] = b[i] - (0..i).map(|j| l[[i, j]] * y[j]).sum::<f64>();
//     }

//     // Backward substitution
//     let mut x = Array1::<f64>::zeros(b.len());
//     for i in (0..b.len()).rev() {
//         x[i] = (y[i] - (i + 1..b.len()).map(|j| u[[i, j]] * x[j]).sum::<f64>()) / u[[i, i]];
//     }

//     x
// }

// /// 识别数组中的连续相等值段，并为每个段分配唯一标识符。
// /// 每个连续相等的值构成一个段，第一个段标识符为1，第二个为2，以此类推。
// ///
// /// 参数说明：
// /// ----------
// /// arr : numpy.ndarray
// ///     输入数组，类型为float64
// ///
// /// 返回值：
// /// -------
// /// numpy.ndarray
// ///     与输入数组等长的整数数组，每个元素表示该位置所属段的标识符
// ///
// /// Python调用示例：
// /// ```python
// /// import numpy as np
// /// from rust_pyfunc import identify_segments
// ///
// /// # 创建测试数组
// /// arr = np.array([1.0, 1.0, 2.0, 2.0, 2.0, 1.0], dtype=np.float64)
// /// segments = identify_segments(arr)
// /// print(f"段标识: {segments}")  # 输出: [1, 1, 2, 2, 2, 3]
// ///
// /// # 解释结果：
// /// # - 第一段 [1.0, 1.0] 标识为1
// /// # - 第二段 [2.0, 2.0, 2.0] 标识为2
// /// # - 第三段 [1.0] 标识为3
// /// ```
// #[pyfunction]
// #[pyo3(signature = (arr))]
// fn identify_segments(arr: PyReadonlyArray1<f64>) -> PyResult<Py<PyArray1<i32>>> {
//     let arr_view = arr.as_array();
//     let n = arr_view.len();
//     let mut segments = Array1::zeros(n);
//     let mut current_segment = 1;

//     for i in 1..n {
//         if arr_view[i] != arr_view[i - 1] {
//             current_segment += 1;
//         }
//         segments[i] = current_segment;
//     }

//     Ok(segments.into_pyarray(arr.py()).to_owned())
// }

// /// 计算序列中每个位置结尾的最长连续子序列长度，其中子序列的最大值在该位置。
// ///
// /// 参数说明：
// /// ----------
// /// s : array_like
// ///     输入序列，一个数值列表
// ///
// /// 返回值：
// /// -------
// /// list
// ///     与输入序列等长的整数列表，每个元素表示以该位置结尾且最大值在该位置的最长连续子序列长度
// ///
// /// Python调用示例：
// /// ```python
// /// from rust_pyfunc import max_range_loop
// ///
// /// # 测试序列
// /// seq = [1.0, 2.0, 3.0, 2.0, 1.0]
// ///
// /// # 计算最大值范围
// /// ranges = max_range_loop(seq)
// /// print(f"最大值范围: {ranges}")  # 输出: [1, 2, 3, 1, 1]
// ///
// /// # 解释结果：
// /// # - ranges[0] = 1: 序列[1.0]的长度，最大值1.0在位置0
// /// # - ranges[1] = 2: 序列[1.0, 2.0]的长度，最大值2.0在位置1
// /// # - ranges[2] = 3: 序列[1.0, 2.0, 3.0]的长度，最大值3.0在位置2
// /// # - ranges[3] = 1: 序列[2.0]的长度，最大值2.0在位置3
// /// # - ranges[4] = 1: 序列[1.0]的长度，最大值1.0在位置4
// /// ```
// #[pyfunction]
// #[pyo3(signature = (s))]
// fn max_range_loop(s: Vec<f64>) -> Vec<i32> {
//     let mut maxranges = Vec::with_capacity(s.len());
//     let mut stack = Vec::new();

//     for i in 0..s.len() {
//         while let Some(&j) = stack.last() {
//             if s[j] < s[i] {
//                 maxranges.push(i as i32 - j as i32);
//                 break;
//             }
//             stack.pop();
//         }
//         if stack.is_empty() {
//             maxranges.push(i as i32 + 1);
//         }
//         stack.push(i);
//     }

//     maxranges
// }

// /// Given a sentence, return a dictionary where the keys are the words in the sentence
// /// and the values are their frequencies.
// ///
// /// For example, given the sentence `"The quick brown fox"`, the function will return
// /// `{"The": 1, "quick": 1, "brown": 1, "fox": 1}`, since each word appears once.
// fn sentence_to_word_count(sentence: &str) -> HashMap<String, usize> {
//     let words: Vec<String> = sentence
//         .to_lowercase() // 转为小写，确保不区分大小写
//         .replace(".", "") // 去掉句末的句点
//         .split_whitespace() // 分词
//         .map(|s| s.to_string()) // 转换为 String
//         .collect();

//     let mut word_count = HashMap::new();
//     for word in words {
//         *word_count.entry(word).or_insert(0) += 1;
//     }

//     word_count
// }

// /// 将两个句子转换为词频向量。
// /// 生成的向量长度相同，等于两个句子中不同单词的总数。
// /// 向量中的每个位置对应一个单词，值表示该单词在句子中出现的次数。
// ///
// /// 参数说明：
// /// ----------
// /// sentence1 : str
// ///     第一个输入句子
// /// sentence2 : str
// ///     第二个输入句子
// ///
// /// 返回值：
// /// -------
// /// tuple
// ///     返回一个元组(vector1, vector2)，其中：
// ///     - vector1: 第一个句子的词频向量
// ///     - vector2: 第二个句子的词频向量
// ///     两个向量长度相同，每个位置对应词表中的一个单词
// ///
// /// Python调用示例：
// /// ```python
// /// from rust_pyfunc import vectorize_sentences
// ///
// /// # 准备两个测试句子
// /// s1 = "The quick brown fox"
// /// s2 = "The lazy brown dog"
// ///
// /// # 转换为词频向量
// /// v1, v2 = vectorize_sentences(s1, s2)
// /// print(f"句子1的词频向量: {v1}")  # 例如：[1, 1, 1, 1, 0]
// /// print(f"句子2的词频向量: {v2}")  # 例如：[1, 0, 1, 0, 1]
// ///
// /// # 解释结果：
// /// # 假设合并的词表为 ["brown", "fox", "quick", "the", "lazy"]
// /// # v1 = [1, 1, 1, 1, 0] 表示 brown, fox, quick, the 各出现一次，lazy未出现
// /// # v2 = [1, 0, 0, 1, 1] 表示 brown, the, lazy 各出现一次，fox和quick未出现
// /// ```
// #[pyfunction]
// #[pyo3(signature = (sentence1, sentence2))]
// fn vectorize_sentences(sentence1: &str, sentence2: &str) -> (Vec<usize>, Vec<usize>) {
//     let count1 = sentence_to_word_count(sentence1);
//     let count2 = sentence_to_word_count(sentence2);

//     let mut all_words: HashSet<String> = HashSet::new();
//     all_words.extend(count1.keys().cloned());
//     all_words.extend(count2.keys().cloned());

//     let mut vector1 = Vec::new();
//     let mut vector2 = Vec::new();

//     for word in &all_words {
//         vector1.push(count1.get(word).unwrap_or(&0).clone());
//         vector2.push(count2.get(word).unwrap_or(&0).clone());
//     }

//     (vector1, vector2)
// }

// /// 将多个句子转换为词频向量列表。
// /// 生成的所有向量长度相同，等于所有句子中不同单词的总数。
// /// 每个向量中的每个位置对应一个单词，值表示该单词在对应句子中出现的次数。
// ///
// /// 参数说明：
// /// ----------
// /// sentences : list[str]
// ///     输入句子列表，每个元素是一个字符串
// ///
// /// 返回值：
// /// -------
// /// list[list[int]]
// ///     返回词频向量列表，其中：
// ///     - 每个向量对应一个输入句子
// ///     - 所有向量长度相同，等于所有句子中不同单词的总数
// ///     - 向量中的每个值表示对应单词在该句子中的出现次数
// ///
// /// Python调用示例：
// /// ```python
// /// from rust_pyfunc import vectorize_sentences_list
// ///
// /// # 准备测试句子列表
// /// sentences = [
// ///     "The quick brown fox",
// ///     "The lazy brown dog",
// ///     "A quick brown fox jumps"
// /// ]
// ///
// /// # 转换为词频向量列表
// /// vectors = vectorize_sentences_list(sentences)
// ///
// /// # 打印每个句子的词频向量
// /// for i, vec in enumerate(vectors):
// ///     print(f"句子{i+1}的词频向量: {vec}")
// ///
// /// # 示例输出解释：
// /// # 假设合并后的词表为 ["a", "brown", "dog", "fox", "jumps", "lazy", "quick", "the"]
// /// # 第一个句子: [0, 1, 0, 1, 0, 0, 1, 1]  # "The quick brown fox"
// /// # 第二个句子: [0, 1, 1, 0, 0, 1, 0, 1]  # "The lazy brown dog"
// /// # 第三个句子: [1, 1, 0, 1, 1, 0, 1, 0]  # "A quick brown fox jumps"
// /// ```
// #[pyfunction]
// #[pyo3(signature = (sentences))]
// fn vectorize_sentences_list(sentences: Vec<&str>) -> Vec<Vec<usize>> {
//     let mut all_words: HashSet<String> = HashSet::new();
//     let mut counts: Vec<HashMap<String, usize>> = Vec::new();

//     // 收集所有单词并计算每个句子的单词频率
//     for sentence in sentences {
//         let count = sentence_to_word_count(sentence);
//         all_words.extend(count.keys().cloned());
//         counts.push(count);
//     }

//     let mut vectors = Vec::new();

//     // 为每个句子构建向量
//     for count in counts {
//         let mut vector = Vec::new();
//         for word in &all_words {
//             vector.push(count.get(word).unwrap_or(&0).clone());
//         }
//         vectors.push(vector);
//     }

//     vectors
// }



// /// 计算两个句子之间的Jaccard相似度。
// /// Jaccard相似度是两个集合交集大小除以并集大小，用于衡量两个句子的相似程度。
// /// 这里将每个句子视为单词集合，忽略单词出现的顺序和频率。
// ///
// /// 参数说明：
// /// ----------
// /// sentence1 : str
// ///     第一个输入句子
// /// sentence2 : str
// ///     第二个输入句子
// ///
// /// 返回值：
// /// -------
// /// float
// ///     返回两个句子的Jaccard相似度，范围在[0, 1]之间：
// ///     - 1表示两个句子完全相同（包含相同的单词集合）
// ///     - 0表示两个句子完全不同（没有共同单词）
// ///     - 中间值表示部分相似
// ///
// /// Python调用示例：
// /// ```python
// /// from rust_pyfunc import jaccard_similarity
// ///
// /// # 测试完全相同的句子
// /// s1 = "The quick brown fox"
// /// s2 = "The quick brown fox"
// /// sim1 = jaccard_similarity(s1, s2)
// /// print(f"完全相同的句子相似度: {sim1}")  # 输出: 1.0
// ///
// /// # 测试部分相同的句子
// /// s3 = "The lazy brown dog"
// /// sim2 = jaccard_similarity(s1, s3)
// /// print(f"部分相同的句子相似度: {sim2}")  # 输出: 0.4 (2个共同词 / 5个不同词)
// ///
// /// # 测试完全不同的句子
// /// s4 = "Hello world example"
// /// sim3 = jaccard_similarity(s1, s4)
// /// print(f"完全不同的句子相似度: {sim3}")  # 输出: 0.0
// ///
// /// # 注意：结果会忽略大小写和标点符号
// /// s5 = "THE QUICK BROWN FOX!"
// /// sim4 = jaccard_similarity(s1, s5)
// /// print(f"大小写不同的相似度: {sim4}")  # 输出: 1.0
// /// ```
// #[pyfunction]
// #[pyo3(signature = (str1, str2))]
// fn jaccard_similarity(str1: &str, str2: &str) -> f64 {
//     // 将字符串分词并转换为集合
//     let set1: HashSet<&str> = str1.split_whitespace().collect();
//     let set2: HashSet<&str> = str2.split_whitespace().collect();

//     // 计算交集和并集
//     let intersection: HashSet<_> = set1.intersection(&set2).cloned().collect();
//     let union: HashSet<_> = set1.union(&set2).cloned().collect();

//     // 计算 Jaccard 相似度
//     if union.is_empty() {
//         0.0
//     } else {
//         intersection.len() as f64 / union.len() as f64
//     }
// }

// /// 计算序列中每个位置结尾的最长连续子序列长度，其中子序列的最小值在该位置。
// ///
// /// 参数说明：
// /// ----------
// /// s : array_like
// ///     输入序列，一个数值列表
// ///
// /// 返回值：
// /// -------
// /// list
// ///     与输入序列等长的整数列表，每个元素表示以该位置结尾且最小值在该位置的最长连续子序列长度
// ///
// /// Python调用示例：
// /// ```python
// /// from rust_pyfunc import min_range_loop
// ///
// /// # 测试序列
// /// seq = [1.0, 2.0, 3.0, 2.0, 1.0]
// ///
// /// # 计算最小值范围
// /// ranges = min_range_loop(seq)
// /// print(f"最小值范围: {ranges}")  # 输出: [1, 2, 3, 1, 5]
// ///
// /// # 解释结果：
// /// # - ranges[0] = 1: 序列[1.0]的长度，最小值1.0在位置0
// /// # - ranges[1] = 2: 序列[1.0, 2.0]的长度，最小值1.0不在位置1
// /// # - ranges[2] = 3: 序列[1.0, 2.0, 3.0]的长度，最小值1.0不在位置2
// /// # - ranges[3] = 1: 序列[2.0]的长度，最小值2.0在位置3
// /// # - ranges[4] = 5: 序列[1.0, 2.0, 3.0, 2.0, 1.0]的长度，最小值1.0在位置4
// /// ```
// #[pyfunction]
// #[pyo3(signature = (s))]
// fn min_range_loop(s: Vec<f64>) -> Vec<i32> {
//     let mut minranges = Vec::with_capacity(s.len());
//     let mut stack = Vec::new();

//     for i in 0..s.len() {
//         while let Some(&j) = stack.last() {
//             if s[j] < s[i] {
//                 minranges.push(i as i32 - j as i32);
//                 break;
//             }
//             stack.pop();
//         }
//         if stack.is_empty() {
//             minranges.push(i as i32 + 1);
//         }
//         stack.push(i);
//     }

//     minranges
// }

// /// 在数组中找到一对索引(x, y)，使得min(arr[x], arr[y]) * |x-y|的值最大。
// /// 这个函数可以用来找到数组中距离最远的两个元素，同时考虑它们的最小值。
// ///
// /// 参数说明：
// /// ----------
// /// arr : numpy.ndarray
// ///     输入数组，类型为float64
// ///
// /// 返回值：
// /// -------
// /// tuple
// ///     返回一个元组(x, y)，其中x和y是使得乘积最大的索引对
// ///
// /// Python调用示例：
// /// ```python
// /// import numpy as np
// /// from rust_pyfunc import find_max_range_product
// ///
// /// # 创建测试数组
// /// arr = np.array([4.0, 2.0, 1.0, 3.0], dtype=np.float64)
// /// x, y = find_max_range_product(arr)
// /// 
// /// # 计算最大乘积
// /// max_product = min(arr[x], arr[y]) * abs(x - y)
// /// print(f"最大乘积出现在索引 {x} 和 {y}")
// /// print(f"对应的值为 {arr[x]} 和 {arr[y]}")
// /// print(f"最大乘积为: {max_product}")
// ///
// /// # 例如，如果x=0, y=3，那么：
// /// # min(arr[0], arr[3]) * |0-3| = min(4.0, 3.0) * 3 = 3.0 * 3 = 9.0
// /// ```
// #[pyfunction]
// #[pyo3(signature = (arr))]
// fn find_max_range_product(arr: PyReadonlyArray1<f64>) -> PyResult<(i64, i64)> {
//     let arr_view = arr.as_array();
//     let n = arr_view.len();
    
//     if n < 2 {
//         return Ok((0, 0));
//     }

//     let mut max_product = f64::NEG_INFINITY;
//     let mut result = (0i64, 0i64);
//     let mut left = 0;
//     let mut right = n - 1;

//     while left < right {
//         let product = arr_view[left].min(arr_view[right]) * (right - left) as f64;
//         if product > max_product {
//             max_product = product;
//             result = (left as i64, right as i64);
//         }

//         // 如果左边的值较小，尝试增加左边界以获得更大的值
//         if arr_view[left] < arr_view[right] {
//             left += 1;
//         } else {
//             // 如果右边的值较小或相等，尝试减小右边界
//             right -= 1;
//         }
//     }

//     // 再检查一遍所有相邻的对
//     for i in 0..n-1 {
//         let product = arr_view[i].min(arr_view[i+1]) * 1.0;
//         if product > max_product {
//             max_product = product;
//             result = (i as i64, (i+1) as i64);
//         }
//     }
    
//     Ok(result)
// }

// #[pymodule]
// fn rust_pyfunc(_py: Python, m: &PyModule) -> PyResult<()> {
//     m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
//     m.add_function(wrap_pyfunction!(dtw_distance, m)?)?;
//     m.add_function(wrap_pyfunction!(transfer_entropy, m)?)?;
//     m.add_function(wrap_pyfunction!(ols, m)?)?;
//     m.add_function(wrap_pyfunction!(ols_predict, m)?)?;
//     m.add_function(wrap_pyfunction!(min_range_loop, m)?)?;
//     m.add_function(wrap_pyfunction!(max_range_loop, m)?)?;
//     m.add_function(wrap_pyfunction!(vectorize_sentences, m)?)?;
//     m.add_function(wrap_pyfunction!(vectorize_sentences_list, m)?)?;
//     m.add_function(wrap_pyfunction!(jaccard_similarity, m)?)?;
//     m.add_function(wrap_pyfunction!(identify_segments, m)?)?;
//     m.add_function(wrap_pyfunction!(trend, m)?)?;
//     m.add_function(wrap_pyfunction!(trend_fast, m)?)?;
//     m.add_function(wrap_pyfunction!(find_max_range_product, m)?)?;
//     m.add_function(wrap_pyfunction!(text::min_word_edit_distance, m)?)?;
//     m.add_function(wrap_pyfunction!(text::normalized_diff, m)?)?;
//     Ok(())
// }

use pyo3::prelude::*;

pub mod text;
pub mod sequence;
pub mod time_series;
pub mod statistics;

/// Formats the sum of two numbers as string.
#[pyfunction]
#[pyo3(signature = (a, b))]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

/// A Python module implemented in Rust.
#[pymodule]
fn rust_pyfunc(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    m.add_function(wrap_pyfunction!(time_series::dtw_distance, m)?)?;
    m.add_function(wrap_pyfunction!(time_series::transfer_entropy, m)?)?;
    m.add_function(wrap_pyfunction!(statistics::ols, m)?)?;
    m.add_function(wrap_pyfunction!(statistics::ols_predict, m)?)?;
    m.add_function(wrap_pyfunction!(statistics::min_range_loop, m)?)?;
    m.add_function(wrap_pyfunction!(statistics::max_range_loop, m)?)?;
    m.add_function(wrap_pyfunction!(text::vectorize_sentences, m)?)?;
    m.add_function(wrap_pyfunction!(text::vectorize_sentences_list, m)?)?;
    m.add_function(wrap_pyfunction!(text::jaccard_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(sequence::identify_segments, m)?)?;
    m.add_function(wrap_pyfunction!(time_series::trend, m)?)?;
    m.add_function(wrap_pyfunction!(time_series::trend_fast, m)?)?;
    m.add_function(wrap_pyfunction!(sequence::find_max_range_product, m)?)?;
    m.add_function(wrap_pyfunction!(text::min_word_edit_distance, m)?)?;
    // m.add_function(wrap_pyfunction!(text::normalized_diff, m)?)?;
    Ok(())
}