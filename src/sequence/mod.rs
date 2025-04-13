use ndarray::Array1;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
// use std::collections::HashMap;
use std::f64;
use std::cmp::Ordering;
use std::collections::HashMap;

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
/// # 例如，如果x=0, y=3那么：
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



/// 计算二维方阵的最大特征值和对应的特征向量
/// 使用幂迭代法计算，不使用并行计算
///
/// 参数说明：
/// ----------
/// matrix : numpy.ndarray
///     输入二维方阵，类型为float64
///
/// 返回值：
/// -------
/// tuple
///     返回一个元组(eigenvalue, eigenvector)，
///     eigenvalue是最大特征值（float64），
///     eigenvector是对应的特征向量（numpy.ndarray）
///
/// Python调用示例：
/// ```python
/// import numpy as np
/// from rust_pyfunc import compute_max_eigenvalue
///
/// # 创建测试矩阵
/// matrix = np.array([[4.0, -1.0], 
///                    [-1.0, 3.0]], dtype=np.float64)
/// eigenvalue, eigenvector = compute_max_eigenvalue(matrix)
/// print(f"最大特征值: {eigenvalue}")
/// print(f"对应的特征向量: {eigenvector}")
/// ```
#[pyfunction]
#[pyo3(signature = (matrix))]
pub fn compute_max_eigenvalue(matrix: PyReadonlyArray2<f64>) -> PyResult<(f64, Py<PyArray1<f64>>)> {
    let matrix_view = matrix.as_array();
    let shape = matrix_view.shape();
    
    if shape[0] != shape[1] {
        return Err(PyValueError::new_err("输入必须是方阵"));
    }
    
    let n = shape[0];
    let mut v = Array1::<f64>::ones(n);
    v.mapv_inplace(|x| x / (n as f64).sqrt());
    
    let max_iter = 30;
    let tolerance = 1e-4;
    let mut eigenvalue: f64;
    let mut prev_eigenvalue: f64;
    
    // 预分配内存并确保内存对齐
    let mut new_v = Array1::<f64>::zeros(n);
    let mut temp = Array1::<f64>::zeros(n);
    
    // 预计算第一次矩阵向量乘积并存储在temp中
    matrix_view.dot(&v).assign_to(&mut temp);
    eigenvalue = v.dot(&temp);
    
    for _ in 0..max_iter {
        prev_eigenvalue = eigenvalue;
        
        // 使用预分配的数组进行矩阵向量乘法
        matrix_view.dot(&v).assign_to(&mut new_v);
        
        // 快速计算范数
        let norm = new_v.dot(&new_v).sqrt();
        if norm < 1e-5 {
            break;
        }
        
        // 原地归一化
        new_v.mapv_inplace(|x| x / norm);
        
        // 计算瑞利商
        matrix_view.dot(&new_v).assign_to(&mut temp);
        eigenvalue = new_v.dot(&temp);
        
        // 交换向量引用
        std::mem::swap(&mut v, &mut new_v);
        
        // 收敛检查
        let rel_error = (eigenvalue - prev_eigenvalue).abs();
        if rel_error < tolerance * eigenvalue.abs() {
            break;
        }
    }
    
    Ok((eigenvalue, v.into_pyarray(matrix.py()).to_owned()))
}

/// 计算价格变化后的香农熵变
/// 
/// 参数说明：
/// ----------
/// exchtime : numpy.ndarray
///     交易时间数组，纳秒时间戳，类型为float64
/// order : numpy.ndarray
///     订单机构ID数组，类型为int64
/// volume : numpy.ndarray
///     成交量数组，类型为float64
/// price : numpy.ndarray
///     价格数组，类型为float64
/// window_seconds : float
///     计算香农熵变的时间窗口，单位为秒
/// top_k : Optional[int]
///     如果提供，则只计算价格最高的k个点的熵变，默认为None（计算所有价格创新高点）
///
/// 返回值：
/// -------
/// numpy.ndarray
///     香农熵变数组，类型为float64。只在价格创新高时计算熵变，其他时刻为NaN。
///     熵变值表示在价格创新高时，从当前时刻到未来window_seconds时间窗口内，
///     交易分布的变化程度。正值表示分布变得更分散，负值表示分布变得更集中。
///
/// 示例：
/// -------
/// >>> import numpy as np
/// >>> from rust_pyfunc import calculate_shannon_entropy_change
/// >>> 
/// >>> # 创建测试数据
/// >>> exchtime = np.array([1e9, 2e9, 3e9, 4e9], dtype=np.float64)  # 时间戳（纳秒）
/// >>> order = np.array([100, 200, 300, 400], dtype=np.int64)  # 机构ID
/// >>> volume = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float64)
/// >>> price = np.array([100.0, 102.0, 101.0, 103.0], dtype=np.float64)
/// >>> 
/// >>> # 计算3秒窗口的香农熵变
/// >>> entropy_changes = calculate_shannon_entropy_change(exchtime, order, volume, price, 3.0)
/// >>> print(entropy_changes)  # 只有价格为100.0、102.0和103.0的位置有非NaN值
#[pyfunction]
#[pyo3(signature = (exchtime, order, volume, price, window_seconds, top_k=None))]
pub fn calculate_shannon_entropy_change(
    exchtime: PyReadonlyArray1<f64>,
    order: PyReadonlyArray1<i64>,
    volume: PyReadonlyArray1<f64>,
    price: PyReadonlyArray1<f64>,
    window_seconds: f64,
    top_k: Option<usize>,
) -> PyResult<Py<PyArray1<f64>>> {
    let py = exchtime.py();
    
    // 转换为ndarray视图
    let exchtime = exchtime.as_array();
    // 将时间戳从纳秒转换为秒
    let exchtime: Vec<i64> = exchtime.iter().map(|&x| (x / 1.0e9) as i64).collect();
    let order = order.as_array();
    let volume = volume.as_array();
    let price = price.as_array();
    let n = exchtime.len();
    
    // 检查所有输入数组长度是否一致
    if order.len() != n || volume.len() != n || price.len() != n {
        return Err(PyValueError::new_err("所有输入数组长度必须相同"));
    }
    
    // 创建结果数组，初始化为NaN
    let mut result = Array1::from_elem(n, f64::NAN);
    
    // 将window_seconds转换为秒
    let window_nanos = window_seconds as i64;
    
    // 使用HashMap存储数据
    let mut base_volumes: HashMap<i64, f64> = HashMap::new();
    let mut window_volumes: HashMap<i64, f64> = HashMap::new();
    
    // 跟踪最高价格
    let mut max_price = f64::NEG_INFINITY;
    let mut window_end = 0;
    let mut cumulative_volume = 0.0;
    
    // 存储价格创新高的点
    let mut high_points: Vec<(usize, f64)> = Vec::new();
    
    // 处理每个时间点
    for i in 0..n {
        let current_time = exchtime[i] as i64;
        let current_price = price[i];
        
        // 更新基准分布
        *base_volumes.entry(order[i]).or_insert(0.0) += volume[i];
        cumulative_volume += volume[i];
        
        // 检查是否是价格创新高
        if current_price > max_price {
            max_price = current_price;
            high_points.push((i, current_price));
            
            // 如果未指定top_k或未收集足够数量的高点，直接计算熵变
            if top_k.is_none() {
                // 计算基准熵
                let base_entropy = if cumulative_volume > 0.0 {
                    calculate_entropy_hashmap(&base_volumes, cumulative_volume)
                } else {
                    0.0
                };
                
                // 更新窗口结束位置
                let window_end_time = current_time + window_nanos;
                while window_end + 1 < n {
                    let next_time = exchtime[window_end + 1] as i64;
                    match next_time.cmp(&window_end_time) {
                        Ordering::Less | Ordering::Equal => window_end += 1,
                        Ordering::Greater => break,
                    }
                }
                
                if window_end > i {
                    // 计算窗口分布
                    window_volumes.clear();
                    window_volumes.extend(base_volumes.iter().map(|(&k, &v)| (k, v)));
                    let mut window_volume = cumulative_volume;
                    
                    // 累加窗口内的数据
                    for j in (i + 1)..=window_end {
                        *window_volumes.entry(order[j]).or_insert(0.0) += volume[j];
                        window_volume += volume[j];
                    }
                    
                    // 计算窗口熵
                    if window_volume > 0.0 {
                        let window_entropy = calculate_entropy_hashmap(&window_volumes, window_volume);
                        result[i] = window_entropy - base_entropy;
                    }
                }
            }
        }
    }
    
    // 如果指定了top_k，只计算价格最高的k个点
    if let Some(k) = top_k {
        if k > 0 && !high_points.is_empty() {
            // 按价格降序排序
            high_points.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
            
            // 只保留前k个点（或者所有点，如果点数少于k）
            let k = k.min(high_points.len());
            let top_points: Vec<usize> = high_points[0..k].iter().map(|&(idx, _)| idx).collect();
            
            // 重置结果数组（因为之前已经计算过了）
            result = Array1::from_elem(n, f64::NAN);
            
            // 重新计算这些点的熵变
            for &i in &top_points {
                let current_time = exchtime[i] as i64;
                
                // 计算到此点为止的基准分布
                let mut local_base_volumes: HashMap<i64, f64> = HashMap::new();
                let mut local_cumulative_volume = 0.0;
                
                for j in 0..=i {
                    *local_base_volumes.entry(order[j]).or_insert(0.0) += volume[j];
                    local_cumulative_volume += volume[j];
                }
                
                // 计算基准熵
                let base_entropy = if local_cumulative_volume > 0.0 {
                    calculate_entropy_hashmap(&local_base_volumes, local_cumulative_volume)
                } else {
                    0.0
                };
                
                // 确定窗口结束位置
                let mut local_window_end = i;
                let window_end_time = current_time + window_nanos;
                while local_window_end + 1 < n {
                    let next_time = exchtime[local_window_end + 1] as i64;
                    match next_time.cmp(&window_end_time) {
                        Ordering::Less | Ordering::Equal => local_window_end += 1,
                        Ordering::Greater => break,
                    }
                }
                
                if local_window_end > i {
                    // 计算窗口分布
                    window_volumes.clear();
                    window_volumes.extend(local_base_volumes.iter().map(|(&k, &v)| (k, v)));
                    let mut window_volume = local_cumulative_volume;
                    
                    // 累加窗口内的数据
                    for j in (i + 1)..=local_window_end {
                        *window_volumes.entry(order[j]).or_insert(0.0) += volume[j];
                        window_volume += volume[j];
                    }
                    
                    // 计算窗口熵
                    if window_volume > 0.0 {
                        let window_entropy = calculate_entropy_hashmap(&window_volumes, window_volume);
                        result[i] = window_entropy - base_entropy;
                    }
                }
            }
        }
    }
    
    // 将第一个值设为NaN
    if n > 0 {
        result[0] = f64::NAN;
    }
    
    Ok(result.into_pyarray(py).to_owned())
}

#[inline(always)]
fn fast_ln(x: f64) -> f64 {
    // 快速对数计算
    x.ln()
}

#[inline(always)]
fn calculate_entropy_hashmap(volumes: &HashMap<i64, f64>, total_volume: f64) -> f64 {
    let mut entropy = 0.0;
    
    for &vol in volumes.values() {
        if vol > 0.0 {
            let p = vol / total_volume;
            entropy -= p * fast_ln(p);
        }
    }
    
    entropy
}

/// 在价格创新低时计算香农熵变
/// 
/// 参数:
/// * exchtime: 交易时间数组
/// * order: 订单号数组
/// * volume: 成交量数组
/// * price: 价格数组
/// * window_seconds: 时间窗口大小（秒）
/// * bottom_k: 如果提供，则只计算价格最低的k个点的熵变，默认为None（计算所有价格创新低点）
/// 
/// 返回:
/// * 香农熵变数组，只在价格创新低时有值，其他位置为NaN
#[pyfunction]
#[pyo3(signature = (exchtime, order, volume, price, window_seconds, bottom_k=None))]
pub fn calculate_shannon_entropy_change_at_low(
    exchtime: PyReadonlyArray1<f64>,
    order: PyReadonlyArray1<i64>,
    volume: PyReadonlyArray1<f64>,
    price: PyReadonlyArray1<f64>,
    window_seconds: f64,
    bottom_k: Option<usize>,
) -> PyResult<Py<PyArray1<f64>>> {
    use std::collections::HashMap;
    
    let py = exchtime.py();
    
    // 转换为ndarray视图
    let exchtime = exchtime.as_array();
    // 将时间戳从纳秒转换为秒
    let exchtime: Vec<i64> = exchtime.iter().map(|&x| (x / 1.0e9) as i64).collect();
    let order = order.as_array();
    let volume = volume.as_array();
    let price = price.as_array();
    let n = exchtime.len();
    
    // 检查所有输入数组长度是否一致
    if order.len() != n || volume.len() != n || price.len() != n {
        return Err(PyValueError::new_err("所有输入数组长度必须相同"));
    }
    
    // 创建结果数组，初始化为NaN
    let mut result = Array1::from_elem(n, f64::NAN);
    
    // 将window_seconds转换为秒
    let window_nanos = window_seconds as i64;
    
    // 使用HashMap存储数据
    let mut base_volumes: HashMap<i64, f64> = HashMap::new();
    let mut window_volumes: HashMap<i64, f64> = HashMap::new();
    
    // 跟踪最低价格
    let mut min_price = f64::INFINITY;
    let mut window_end = 0;
    let mut cumulative_volume = 0.0;
    
    // 存储价格创新低的点
    let mut low_points: Vec<(usize, f64)> = Vec::new();
    
    // 处理每个时间点
    for i in 0..n {
        let current_time = exchtime[i] as i64;
        let current_price = price[i];
        
        // 更新基准分布
        *base_volumes.entry(order[i]).or_insert(0.0) += volume[i];
        cumulative_volume += volume[i];
        
        // 检查是否是价格创新低
        if current_price < min_price {
            min_price = current_price;
            low_points.push((i, current_price));
            
            // 如果未指定bottom_k或未收集足够数量的低点，直接计算熵变
            if bottom_k.is_none() {
                // 计算基准熵
                let base_entropy = if cumulative_volume > 0.0 {
                    calculate_entropy_hashmap(&base_volumes, cumulative_volume)
                } else {
                    0.0
                };
                
                // 更新窗口结束位置
                let window_end_time = current_time + window_nanos;
                while window_end + 1 < n {
                    let next_time = exchtime[window_end + 1] as i64;
                    match next_time.cmp(&window_end_time) {
                        Ordering::Less | Ordering::Equal => window_end += 1,
                        Ordering::Greater => break,
                    }
                }
                
                if window_end > i {
                    // 计算窗口分布
                    window_volumes.clear();
                    window_volumes.extend(base_volumes.iter().map(|(&k, &v)| (k, v)));
                    let mut window_volume = cumulative_volume;
                    
                    // 累加窗口内的数据
                    for j in (i + 1)..=window_end {
                        *window_volumes.entry(order[j]).or_insert(0.0) += volume[j];
                        window_volume += volume[j];
                    }
                    
                    // 计算窗口熵
                    if window_volume > 0.0 {
                        let window_entropy = calculate_entropy_hashmap(&window_volumes, window_volume);
                        result[i] = window_entropy - base_entropy;
                    }
                }
            }
        }
    }
    
    // 如果指定了bottom_k，只计算价格最低的k个点
    if let Some(k) = bottom_k {
        if k > 0 && !low_points.is_empty() {
            // 按价格升序排序
            low_points.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
            
            // 只保留前k个点（或者所有点，如果点数少于k）
            let k = k.min(low_points.len());
            let bottom_points: Vec<usize> = low_points[0..k].iter().map(|&(idx, _)| idx).collect();
            
            // 重置结果数组（因为之前已经计算过了）
            result = Array1::from_elem(n, f64::NAN);
            
            // 重新计算这些点的熵变
            for &i in &bottom_points {
                let current_time = exchtime[i] as i64;
                
                // 计算到此点为止的基准分布
                let mut local_base_volumes: HashMap<i64, f64> = HashMap::new();
                let mut local_cumulative_volume = 0.0;
                
                for j in 0..=i {
                    *local_base_volumes.entry(order[j]).or_insert(0.0) += volume[j];
                    local_cumulative_volume += volume[j];
                }
                
                // 计算基准熵
                let base_entropy = if local_cumulative_volume > 0.0 {
                    calculate_entropy_hashmap(&local_base_volumes, local_cumulative_volume)
                } else {
                    0.0
                };
                
                // 确定窗口结束位置
                let mut local_window_end = i;
                let window_end_time = current_time + window_nanos;
                while local_window_end + 1 < n {
                    let next_time = exchtime[local_window_end + 1] as i64;
                    match next_time.cmp(&window_end_time) {
                        Ordering::Less | Ordering::Equal => local_window_end += 1,
                        Ordering::Greater => break,
                    }
                }
                
                if local_window_end > i {
                    // 计算窗口分布
                    window_volumes.clear();
                    window_volumes.extend(local_base_volumes.iter().map(|(&k, &v)| (k, v)));
                    let mut window_volume = local_cumulative_volume;
                    
                    // 累加窗口内的数据
                    for j in (i + 1)..=local_window_end {
                        *window_volumes.entry(order[j]).or_insert(0.0) += volume[j];
                        window_volume += volume[j];
                    }
                    
                    // 计算窗口熵
                    if window_volume > 0.0 {
                        let window_entropy = calculate_entropy_hashmap(&window_volumes, window_volume);
                        result[i] = window_entropy - base_entropy;
                    }
                }
            }
        }
    }
    
    // 将第一个值设为NaN
    if n > 0 {
        result[0] = f64::NAN;
    }
    
    Ok(result.into_pyarray(py).to_owned())
}

/// 计算最速曲线（投掷线）并返回x_series对应的y坐标
/// 
/// 参数说明：
/// ----------
/// x1 : float
///     起点x坐标
/// y1 : float
///     起点y坐标
/// x2 : float
///     终点x坐标
/// y2 : float
///     终点y坐标
/// x_series : numpy.ndarray
///     需要计算y坐标的x点序列
///
/// 返回值：
/// -------
/// numpy.ndarray
///     与x_series相对应的y坐标值数组
///
/// Python调用示例：
/// ```python
/// import numpy as np
/// import pandas as pd
/// from rust_pyfunc import brachistochrone_curve
///
/// # 创建x序列
/// x_vals = pd.Series(np.linspace(0, 5, 100))
/// # 计算从点(0,0)到点(5,-3)的最速曲线
/// y_vals = brachistochrone_curve(0.0, 0.0, 5.0, -3.0, x_vals)
/// ```
#[pyfunction]
#[pyo3(signature = (x1, y1, x2, y2, x_series))]
pub fn brachistochrone_curve(
    x1: f64, 
    y1: f64, 
    x2: f64, 
    y2: f64, 
    x_series: PyReadonlyArray1<f64>
) -> PyResult<Py<PyArray1<f64>>> {
    let py = x_series.py();
    let x_view = x_series.as_array();
    let n = x_view.len();
    
    // 确保x1 < x2
    let (x1, y1, x2, y2) = if x1 > x2 {
        (x2, y2, x1, y1)
    } else {
        (x1, y1, x2, y2)
    };
    
    // 转换坐标使起点位于原点
    let x_offset = x1;
    let y_offset = y1;
    let x2_shifted = x2 - x_offset;
    let y2_shifted = y2 - y_offset;
    
    // 如果y2_shifted > 0，需要翻转坐标系，使粒子沿着重力方向移动
    let (y2_shifted, flip) = if y2_shifted > 0.0 {
        (-y2_shifted, true)
    } else {
        (y2_shifted, false)
    };
    
    // 使用Nelder-Mead算法寻找最优R和theta
    let initial_r = (x2_shifted * x2_shifted + y2_shifted * y2_shifted).sqrt() / 2.0;
    let initial_theta = std::f64::consts::PI;
    let (r, theta_max) = optimize_r_theta(initial_r, initial_theta, x2_shifted, y2_shifted);
    
    // 创建结果数组
    let mut result = Array1::from_elem(n, f64::NAN);
    
    // 计算x_series对应的y值
    for (i, &x) in x_view.iter().enumerate() {
        let x_shifted = x - x_offset;
        
        // 超出范围的值设为NaN
        if x_shifted < 0.0 || x_shifted > x2_shifted {
            result[i] = f64::NAN;
            continue;
        }
        
        // 求解theta
        match solve_theta_for_x(r, theta_max, x_shifted) {
            Some(theta) => {
                // 计算对应的y值
                let mut y = -r * (1.0 - theta.cos());
                if flip {
                    y = -y;
                }
                result[i] = y + y_offset;
            },
            None => {
                result[i] = f64::NAN;
            }
        }
    }
    
    Ok(result.into_pyarray(py).to_owned())
}

// 内部函数：优化R和theta参数
fn optimize_r_theta(initial_r: f64, initial_theta: f64, x_target: f64, y_target: f64) -> (f64, f64) {
    // 目标函数：计算参数方程得到的终点与目标点的距离平方
    fn objective(r: f64, theta: f64, x_target: f64, y_target: f64) -> f64 {
        let x_end = r * (theta - theta.sin());
        let y_end = -r * (1.0 - theta.cos());
        (x_end - x_target).powi(2) + (y_end - y_target).powi(2)
    }
    
    // 使用简化版的Nelder-Mead算法进行优化
    // 实际实现中，这里可能需要一个更复杂的优化算法库
    let mut r = initial_r;
    let mut theta = initial_theta;
    let mut step_size_r = initial_r * 0.1;
    let mut step_size_theta = 0.1;
    let tolerance = 1e-6;
    let max_iterations = 100;
    
    for _ in 0..max_iterations {
        let current_error = objective(r, theta, x_target, y_target);
        
        // 尝试在r方向上移动
        let r_plus = objective(r + step_size_r, theta, x_target, y_target);
        let r_minus = objective(r - step_size_r, theta, x_target, y_target);
        
        if r_plus < current_error {
            r += step_size_r;
        } else if r_minus < current_error {
            r -= step_size_r;
        } else {
            step_size_r *= 0.5;
        }
        
        // 尝试在theta方向上移动
        let theta_plus = objective(r, theta + step_size_theta, x_target, y_target);
        let theta_minus = objective(r, theta - step_size_theta, x_target, y_target);
        
        if theta_plus < current_error {
            theta += step_size_theta;
        } else if theta_minus < current_error {
            theta -= step_size_theta;
        } else {
            step_size_theta *= 0.5;
        }
        
        // 检查是否已经收敛
        if step_size_r < tolerance && step_size_theta < tolerance {
            break;
        }
    }
    
    // 返回优化后的r和theta
    (r, theta)
}

// 内部函数：为给定的x值求解theta
fn solve_theta_for_x(r: f64, theta_max: f64, x: f64) -> Option<f64> {
    // 使用二分法求解方程 r*(theta - sin(theta)) = x
    
    // 设置搜索范围
    let mut low = 0.0;
    let mut high = theta_max;
    let tolerance = 1e-6;
    let max_iterations = 50;
    
    for _ in 0..max_iterations {
        let mid = (low + high) / 2.0;
        let x_mid = r * (mid - mid.sin());
        
        if (x_mid - x).abs() < tolerance {
            return Some(mid);
        }
        
        if x_mid < x {
            low = mid;
        } else {
            high = mid;
        }
    }
    
    // 如果未收敛，检查最终结果是否足够接近
    let final_mid = (low + high) / 2.0;
    let final_x = r * (final_mid - final_mid.sin());
    
    if (final_x - x).abs() < tolerance * 10.0 {
        Some(final_mid)
    } else {
        None
    }
}


