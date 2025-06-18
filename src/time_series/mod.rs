use pyo3::prelude::*;
use pyo3::types::PyList;
use numpy::{PyArray1, PyReadonlyArray1, IntoPyArray};
use ndarray::Array1;
use pyo3::exceptions::PyValueError;
use std::time::Instant;
use std::error::Error;
use std::fmt;
use std::collections::HashMap;

// 导入优化版模块
pub mod fast_extreme;
pub mod super_extreme;
pub mod retreat_advance;
pub mod retreat_advance_v2;

// 定义超时错误结构
#[derive(Debug)]
struct TimeoutError {
    message: String,
    duration: f64,
}

impl Error for TimeoutError {}

impl fmt::Display for TimeoutError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{} ({} seconds)", self.message, self.duration)
    }
}

// 实现从 TimeoutError 到 PyErr 的转换
impl From<TimeoutError> for PyErr {
    fn from(err: TimeoutError) -> PyErr {
        PyValueError::new_err(format!("{}({}s)", err.message, err.duration))
    }
}

// use std::collections::VecDeque;
// use std::collections::BTreeMap;

/// DTW（动态时间规整）是一种测量两个时间序列相似度的方法。
/// 该算法计算两个可能长度不同、tempo不同的时间序列间的最优匹配。
///
/// 参数说明：
/// ----------
/// s1 : array_like
///     第一个时间序列
/// s2 : array_like
///     第二个时间序列
/// radius : int, optional
///     Sakoe-Chiba半径，用于限制规整路径，可以提高计算效率。
///     如果不指定，则不使用路径限制。
/// timeout_seconds : float, optional
///     计算超时时间，单位为秒。如果函数执行时间超过此值，将抛出TimeoutError异常。
///     默认为None，表示无超时限制。
///
/// 返回值：
/// -------
/// float
///     两个序列之间的DTW距离。值越小表示序列越相似。
///
/// 异常：
/// -----
/// TimeoutError
///     当计算时间超过timeout_seconds指定的秒数时抛出
///
/// Python调用示例：
/// ```python
/// import numpy as np
/// from rust_pyfunc import dtw_distance
///
/// # 创建两个测试序列
/// s1 = [1.0, 2.0, 3.0, 4.0, 5.0]
/// s2 = [1.0, 2.0, 2.5, 3.5, 4.0, 5.0]
///
/// # 计算完整DTW距离
/// dist1 = dtw_distance(s1, s2)
/// print(f"DTW距离: {dist1}")
///
/// # 使用radius=1限制规整路径
/// dist2 = dtw_distance(s1, s2, radius=1)
/// print(f"使用radius=1的DTW距离: {dist2}")
///
/// # 设置超时时间为1秒
/// try:
///     dist3 = dtw_distance(s1, s2, timeout_seconds=1.0)
///     print(f"DTW距离: {dist3}")
/// except RuntimeError as e:
///     print(f"超时错误: {e}")
/// ```
#[pyfunction]
#[pyo3(signature = (s1, s2, radius=None, timeout_seconds=None))]
pub fn dtw_distance(s1: Vec<f64>, s2: Vec<f64>, radius: Option<usize>, timeout_seconds: Option<f64>) -> PyResult<f64> {
    // 记录开始时间
    let start_time = Instant::now();
    
    // 检查超时的闭包函数
    let check_timeout = |timeout: Option<f64>| -> Result<(), TimeoutError> {
        if let Some(timeout) = timeout {
            let elapsed = start_time.elapsed().as_secs_f64();
            if elapsed > timeout {
                return Err(TimeoutError {
                    message: "DTW距离计算超时".to_string(),
                    duration: elapsed,
                });
            }
        }
        Ok(())
    };
    
    let len_s1 = s1.len();
    let len_s2 = s2.len();
    let mut warp_dist_mat = vec![vec![f64::INFINITY; len_s2 + 1]; len_s1 + 1];
    warp_dist_mat[0][0] = 0.0;

    // 检查初始化后是否超时
    check_timeout(timeout_seconds)?;

    for i in 1..=len_s1 {
        // 每行开始时检查一次超时
        check_timeout(timeout_seconds)?;
        
        for j in 1..=len_s2 {
            // 对于大型序列，每100次计算检查一次超时
            if (i * len_s2 + j) % 100 == 0 {
                check_timeout(timeout_seconds)?;
            }
            
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
    
    // 最终检查一次超时
    check_timeout(timeout_seconds)?;
    
    Ok(warp_dist_mat[len_s1][len_s2])
}

/// 优化版DTW距离函数，使用以下技术提升性能：
/// 1. 使用一维数组代替二维数组，减少内存分配和间接访问
/// 2. 提前计算常用值，减少重复计算
/// 3. 对于窗口计算进行更高效的实现
/// 4. 优化内存访问模式，提高缓存命中率
/// 5. 智能初始化窗口内单元格，避免无限值问题
/// 6. 自动调整radius大小，确保计算结果有效
/// 
/// ```python
/// import numpy as np
/// from rust_pyfunc import fast_dtw_distance, dtw_distance
/// 
/// # 创建两个时间序列
/// s1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
/// s2 = np.array([1.1, 2.1, 3.1, 4.1, 5.1])
/// 
/// # 比较两种实现的结果和性能
/// import time
/// 
/// start = time.time()
/// result1 = dtw_distance(s1, s2)
/// time1 = time.time() - start
/// 
/// start = time.time()
/// result2 = fast_dtw_distance(s1, s2)
/// time2 = time.time() - start
/// 
/// print(f"标准DTW距离: {result1:.6f}, 耗时: {time1:.6f}秒")
/// print(f"快速DTW距离: {result2:.6f}, 耗时: {time2:.6f}秒")
/// print(f"加速比: {time1/time2:.2f}倍")
/// ```
#[pyfunction]
#[pyo3(signature = (s1, s2, radius=None, timeout_seconds=None))]
pub fn fast_dtw_distance(s1: Vec<f64>, s2: Vec<f64>, radius: Option<usize>, timeout_seconds: Option<f64>) -> PyResult<f64> {
    // 记录开始时间
    let start_time = Instant::now();
    
    // 检查超时的闭包函数
    let check_timeout = |timeout: Option<f64>| -> Result<(), TimeoutError> {
        if let Some(timeout) = timeout {
            let elapsed = start_time.elapsed().as_secs_f64();
            if elapsed > timeout {
                return Err(TimeoutError {
                    message: "快速DTW距离计算超时".to_string(),
                    duration: elapsed,
                });
            }
        }
        Ok(())
    };
    
    let len_s1 = s1.len();
    let len_s2 = s2.len();
    
    // 当任一输入序列长度为0时，返回NaN
    if len_s1 == 0 || len_s2 == 0 {
        return Ok(f64::NAN);
    }
    
    // 如果radius为None，直接使用原始算法，保证结果不变
    if radius.is_none() {
        return compute_dtw(&s1, &s2, None, &check_timeout, timeout_seconds);
    }
    
    // 尝试使用提供的radius计算，如果结果为inf，则自动增加radius重试
    let mut current_radius = radius;
    let mut result = compute_dtw(&s1, &s2, current_radius, &check_timeout, timeout_seconds)?;
    
    // 如果结果为inf并且有设置radius，则自动增加radius重试
    // 最多尝试3次，每次将radius增加为序列长度的1/4, 1/2, 全长
    let max_attempts = 3;
    let mut attempt = 0;
    
    while result.is_infinite() && attempt < max_attempts && current_radius.is_some() {
        attempt += 1;
        
        // 逐步增加radius：1/4序列长度 -> 1/2序列长度 -> 全长
        let new_radius = match attempt {
            1 => len_s1.max(len_s2) / 4,
            2 => len_s1.max(len_s2) / 2,
            _ => len_s1.max(len_s2),
        };
        
        // 确保新radius大于当前radius
        if let Some(r) = current_radius {
            if new_radius <= r {
                continue;
            }
        }
        
        current_radius = Some(new_radius);
        result = compute_dtw(&s1, &s2, current_radius, &check_timeout, timeout_seconds)?;
    }
    
    Ok(result)
}

/// 内部DTW计算函数，支持带窗口约束的计算
fn compute_dtw<F>(s1: &[f64], s2: &[f64], radius: Option<usize>, check_timeout: &F, timeout_seconds: Option<f64>) -> PyResult<f64> 
where F: Fn(Option<f64>) -> Result<(), TimeoutError> {
    let len_s1 = s1.len();
    let len_s2 = s2.len();
    
    // 当任一输入序列长度为0时，返回NaN
    if len_s1 == 0 || len_s2 == 0 {
        return Ok(f64::NAN);
    }
    
    // 只使用两行的一维数组而不是完整的二维矩阵，节省内存
    let mut prev_row = vec![f64::INFINITY; len_s2 + 1];
    let mut curr_row = vec![f64::INFINITY; len_s2 + 1];
    
    // 初始化第一个元素为0
    prev_row[0] = 0.0;
    
    // 改进的初始化：为第一行提供更好的初始值
    if let Some(r) = radius {
        // 只有在使用窗口约束时才需要特殊初始化
        let init_range = 1..=(r.min(len_s2));
        let mut cum_cost = 0.0;
        
        for j in init_range {
            cum_cost += (s1[0] - s2[j-1]).abs();
            prev_row[j] = cum_cost; // 累积第一行的代价
        }
    }
    
    // 检查初始化后是否超时
    check_timeout(timeout_seconds)?;
    
    for i in 1..=len_s1 {
        // 每行开始时检查一次超时
        check_timeout(timeout_seconds)?;
        
        // 提前将当前序列值缓存，减少每次循环中的索引查找
        let s1_val = s1[i - 1];
        
        // 初始化当前行的第一个元素
        // 改进：当使用radius时，为第一列提供更好的初始值
        if let Some(r) = radius {
            if i <= r {
                // 在窗口范围内的第一列，累积代价
                let mut cum_cost = 0.0;
                for k in 0..i {
                    cum_cost += (s1[k] - s2[0]).abs();
                }
                curr_row[0] = cum_cost;
            } else {
                curr_row[0] = f64::INFINITY;
            }
        } else {
            curr_row[0] = f64::INFINITY;
        }
        
        // 对于半径限制，直接计算起始和结束范围
        let (j_start, j_end) = match radius {
            Some(r) => (1.max(i.saturating_sub(r)), (i + r).min(len_s2) + 1),
            None => (1, len_s2 + 1),
        };
        
        for j in j_start..j_end {
            // 对于大型序列，每500次计算检查一次超时
            if (i * len_s2 + j) % 500 == 0 {
                check_timeout(timeout_seconds)?;
            }
            
            // 直接计算cost，避免额外的函数调用
            let cost = (s1_val - s2[j - 1]).abs();
            
            // 使用min方法链式比较，避免嵌套min调用
            curr_row[j] = cost + prev_row[j]
                .min(curr_row[j - 1])
                .min(prev_row[j - 1]);
        }
        
        // 交换行，避免额外的内存分配
        std::mem::swap(&mut prev_row, &mut curr_row);
    }
    
    // 最终检查一次超时
    check_timeout(timeout_seconds)?;
    
    // 由于我们交换了行，结果在prev_row的最后一个元素中
    Ok(prev_row[len_s2])
}

/// 超级优化版DTW距离函数，使用以下高级技术提升性能：
/// 1. 内存预分配 - 减少运行时内存分配
/// 2. 更精细的内存访问优化 - 提高缓存命中率
/// 3. 基于启发式的跳过技术 - 避免不必要的计算
/// 4. 提前退出策略 - 当部分结果已超过最优值时提前终止
/// 5. 更稀疏的超时检查 - 减少检查开销
/// 
/// ```python
/// import numpy as np
/// from rust_pyfunc import super_dtw_distance, fast_dtw_distance, dtw_distance
/// 
/// # 创建两个时间序列
/// s1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
/// s2 = np.array([1.1, 2.1, 3.1, 4.1, 5.1])
/// 
/// # 比较三种实现的结果和性能
/// import time
/// 
/// start = time.time()
/// result1 = dtw_distance(s1, s2)
/// time1 = time.time() - start
/// 
/// start = time.time()
/// result2 = fast_dtw_distance(s1, s2)
/// time2 = time.time() - start
/// 
/// start = time.time()
/// result3 = super_dtw_distance(s1, s2)
/// time3 = time.time() - start
/// 
/// print(f"标准DTW距离: {result1:.6f}, 耗时: {time1:.6f}秒")
/// print(f"快速DTW距离: {result2:.6f}, 耗时: {time2:.6f}秒")
/// print(f"超级DTW距离: {result3:.6f}, 耗时: {time3:.6f}秒")
/// print(f"fast加速比: {time1/time2:.2f}倍")
/// print(f"super加速比: {time1/time3:.2f}倍")
/// ```
#[pyfunction]
#[pyo3(signature = (s1, s2, radius=None, timeout_seconds=None, lower_bound_pruning=true, early_termination_threshold=None))]
pub fn super_dtw_distance(
    s1: Vec<f64>, 
    s2: Vec<f64>, 
    radius: Option<usize>, 
    timeout_seconds: Option<f64>,
    lower_bound_pruning: bool,
    early_termination_threshold: Option<f64>
) -> PyResult<f64> {
    // 记录开始时间
    let start_time = Instant::now();
    
    // 检查输入序列长度
    let len_s1 = s1.len();
    let len_s2 = s2.len();
    
    // 特殊情况快速处理
    if len_s1 == 0 || len_s2 == 0 {
        return Ok(0.0);
    }
    
    // 优化1: 确保s1是较短的序列，减少内存占用和计算量
    if len_s1 > len_s2 {
        return super_dtw_distance(s2, s1, radius, timeout_seconds, lower_bound_pruning, early_termination_threshold);
    }
    
    // 检查超时的闭包函数 - 减少检查频率
    let check_timeout = |timeout: Option<f64>, iteration: usize| -> Result<(), TimeoutError> {
        // 每1000次迭代检查一次超时，进一步降低检查频率
        if iteration % 1000 == 0 {
            if let Some(timeout) = timeout {
                let elapsed = start_time.elapsed().as_secs_f64();
                if elapsed > timeout {
                    return Err(TimeoutError {
                        message: "超级DTW距离计算超时".to_string(),
                        duration: elapsed,
                    });
                }
            }
        }
        Ok(())
    };
    
    // 优化2: 启发式下界剪枝 - 计算曼哈顿距离作为DTW的下界
    if lower_bound_pruning {
        // 计算序列间的最小差异（曼哈顿距离）作为DTW的下界
        let lb_kim = (s1[0] - s2[0]).abs() + (s1[len_s1-1] - s2[len_s2-1]).abs();
        
        // 如果已知阈值并且下界已超过阈值，提前返回
        if let Some(threshold) = early_termination_threshold {
            if lb_kim > threshold {
                return Ok(f64::INFINITY);
            }
        }
    }
    
    // 优化3: 内存预分配 - 只使用两行的一维数组，一次性分配好内存
    let mut prev_row = vec![f64::INFINITY; len_s2 + 1];
    let mut curr_row = vec![f64::INFINITY; len_s2 + 1];
    
    prev_row[0] = 0.0;
    
    // 检查初始化后是否超时
    check_timeout(timeout_seconds, 0)?;
    
    // 优化4: 滑动窗口的高效实现
    let r = match radius {
        Some(r) => r,
        None => len_s2,  // 如果没有指定半径，使用全序列长度
    };
    
    // 总迭代次数计数，用于超时检查
    let mut iter_count = 0;
    
    // 缓存s2的值，减少重复索引查找
    let s2_vals: Vec<f64> = s2.iter().copied().collect();
    
    for i in 1..=len_s1 {
        // 优化5: 提前将当前序列值缓存
        let s1_val = s1[i - 1];
        
        // 初始化当前行的第一个元素
        curr_row[0] = f64::INFINITY;
        
        // 计算当前行的有效列范围
        let j_start = 1.max(i.saturating_sub(r));
        let j_end = (i + r).min(len_s2) + 1;
        
        // 设置提前终止阈值的本地变量
        let mut local_min = f64::INFINITY;
        let early_abandon = early_termination_threshold.is_some();
        
        for j in j_start..j_end {
            iter_count += 1;
            
            // 稀疏超时检查
            check_timeout(timeout_seconds, iter_count)?;
            
            // 优化6: 缓存访问和直接计算cost
            let cost = (s1_val - s2_vals[j - 1]).abs();
            
            let min_prev = prev_row[j].min(curr_row[j - 1]).min(prev_row[j - 1]);
            let candidate = cost + min_prev;
            
            // 优化7: 记录当前行的最小值，用于提前终止检查
            if early_abandon && candidate < local_min {
                local_min = candidate;
            }
            
            curr_row[j] = candidate;
        }
        
        // 优化8: 基于提前终止阈值的剪枝
        if early_abandon {
            let threshold = early_termination_threshold.unwrap();
            if local_min > threshold {
                return Ok(f64::INFINITY); // 已超过阈值，不可能是最优解
            }
        }
        
        // 交换行，避免额外的内存分配
        std::mem::swap(&mut prev_row, &mut curr_row);
    }
    
    // 最终检查一次超时
    check_timeout(timeout_seconds, iter_count + 1)?;
    
    // 结果在prev_row的最后一个元素中
    Ok(prev_row[len_s2])
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

/// 查找时间序列中价格在指定时间窗口内为局部最大值的点。
/// 
/// 参数说明：
/// ----------
/// times : array_like
///     时间戳数组（单位：秒）
/// prices : array_like
///     价格数组
/// window : float
///     时间窗口大小（单位：秒）
/// 
/// 返回值：
/// -------
/// numpy.ndarray
///     布尔数组，True表示该点的价格大于指定时间窗口内的所有价格
/// 
/// Python调用示例：
/// ```python
/// import numpy as np
/// from rust_pyfunc import find_local_peaks_within_window
/// 
/// # 创建示例数据
/// times = np.array([0.0, 10.0, 20.0, 30.0, 40.0])  # 时间戳（秒）
/// prices = np.array([1.0, 3.0, 2.0, 1.5, 1.0])     # 价格
/// window = 100.0  # 时间窗口大小（秒）
/// 
/// # 查找局部最大值点
/// peaks = find_local_peaks_within_window(times, prices, window)
/// # 获取满足条件的数据
/// result_times = times[peaks]
/// result_prices = prices[peaks]
/// ```
#[pyfunction]
pub fn find_local_peaks_within_window(times: PyReadonlyArray1<f64>, prices: PyReadonlyArray1<f64>, window: f64) -> PyResult<Vec<bool>> {
    let times = times.as_array();
    let times: Vec<f64> = times.iter().map(|&x| x / 1.0e9).collect();
    let prices = prices.as_array();
    let n = times.len();
    let mut result = vec![false; n];
    
    // 对每个点，检查之后window秒内是否存在更大的价格
    for i in 0..n {
        let current_time = times[i];
        let mut is_peak = true;
        
        // 检查之后的点
        for j in (i + 1)..n {
            // 如果时间差超过window秒，退出内层循环
            if times[j] - current_time > window {
                break;
            }
            // 如果找到更大的价格，说明当前点不是局部最大值
            if prices[j] > prices[i] {
                is_peak = false;
                break;
            }
        }
        
        result[i] = is_peak;
    }
    
    // 最后一个点总是局部最大值（因为之后没有点了）
    if n > 0 {
        result[n-1] = true;
    }
    
    Ok(result)
}

/// 计算每一行在其后time_window秒内具有相同volume（及可选相同price）的行的volume总和。
/// 
/// 参数说明：
/// ----------
/// times : array_like
///     时间戳数组（单位：秒）
/// prices : array_like
///     价格数组
/// volumes : array_like
///     成交量数组
/// time_window : float, optional, default=0.1
///     时间窗口（单位：秒）
/// check_price : bool, optional, default=True
///     是否检查价格是否相同
/// filter_frequent_volumes : bool, optional, default=False
///     是否过滤频繁出现的相同volume值
/// 
/// 返回值：
/// -------
/// numpy.ndarray
///     每一行在其后time_window秒内具有相同条件的行的volume总和
///     如果filter_frequent_volumes=True，则出现频率超过30%的volume值对应的行会被设为NaN
/// 
/// Python调用示例：
/// ```python
/// import pandas as pd
/// import numpy as np
/// from rust_pyfunc import find_follow_volume_sum_same_price
/// 
/// # 创建示例DataFrame
/// df = pd.DataFrame({
///     'exchtime': [1.0, 1.05, 1.08, 1.15, 1.2],
///     'price': [10.0, 10.0, 10.0, 11.0, 10.0],
///     'volume': [100, 100, 100, 200, 100]
/// })
/// 
/// # 计算follow列
/// df['follow'] = find_follow_volume_sum_same_price(
///     df['exchtime'].values,
///     df['price'].values,
///     df['volume'].values
/// )
/// ```
/// 计算每一行在其后time_window秒内具有相同volume（及可选相同price）的行的volume总和。
/// 
/// 参数说明：
/// ----------
/// times : array_like
///     时间戳数组（单位：秒）
/// prices : array_like
///     价格数组
/// volumes : array_like
///     成交量数组
/// time_window : float, optional, default=0.1
///     时间窗口（单位：秒）
/// check_price : bool, optional, default=True
///     是否检查价格是否相同，默认为True。设为False时只检查volume是否相同。
/// filter_ratio : float, optional, default=0.0
///     要过滤的volume数值比例，默认为0（不过滤）。如果大于0，则过滤出现频率最高的前 filter_ratio 比例的volume种类，对应的行会被设为NaN。
/// 
/// 返回值：
/// -------
/// numpy.ndarray
///     每一行在其后time_window秒内（包括当前行）具有相同条件的行的volume总和。
///     如果filter_frequent_volumes=True，则出现频率超过30%的volume值对应的行会被设为NaN。
#[pyfunction]
#[pyo3(signature = (times, prices, volumes, time_window=0.1, check_price=true, filter_ratio=0.0, timeout_seconds=None))]
pub fn find_follow_volume_sum_same_price(
    times: PyReadonlyArray1<f64>,
    prices: PyReadonlyArray1<f64>,
    volumes: PyReadonlyArray1<f64>,
    time_window: f64,
    check_price: bool,
    filter_ratio: f64,
    timeout_seconds: Option<f64>
) -> PyResult<Vec<f64>> {
    // 记录开始时间（用于超时检查）
    let start_time = std::time::Instant::now();
    
    // 检查超时的辅助函数
    let check_timeout = |timeout: Option<f64>| -> Result<(), TimeoutError> {
        if let Some(timeout) = timeout {
            let elapsed = start_time.elapsed().as_secs_f64();
            if elapsed > timeout {
                return Err(TimeoutError {
                    message: "查找相同价格后续成交量计算超时".to_string(),
                    duration: elapsed,
                });
            }
        }
        Ok(())
    };
    
    // 准备数据
    let times = times.as_array();
    let times: Vec<f64> = times.iter().map(|&x| x / 1.0e9).collect();
    let prices = prices.as_array();
    let volumes = volumes.as_array();
    let n = times.len();
    let mut result = vec![0.0; n];
    
    // 导入OrderedFloat以便使用浮点数作为BTreeMap的键
    use ordered_float::OrderedFloat;
    
    // 第1步：计算每个点在time_window内的volume总和并标记无匹配的点为NaN
    for i in 0..n {
        // 定期检查超时（每100次计算）
        if i % 100 == 0 {
            if let Err(_) = check_timeout(timeout_seconds) {
                return Ok(vec![f64::NAN; n]); // 如果超时，返回全NaN的数组
            }
        }
        
        let current_time = times[i];
        let current_price = prices[i];
        let current_volume = volumes[i];
        let mut sum = current_volume; // 包含当前点的成交量
        let mut has_match = false; // 记录是否有匹配
        
        // 检查之后的点
        for j in (i + 1)..n {
            // 如果时间差超过time_window秒，退出内层循环
            if times[j] - current_time > time_window {
                break;
            }
            
            // 根据check_price参数决定是否检查价格
            let price_match = !check_price || (prices[j] - current_price).abs() < 1e-10;
            let volume_match = (volumes[j] - current_volume).abs() < 1e-10;
            
            if price_match && volume_match {
                has_match = true;
                sum += volumes[j];
            }
        }
        
        // 如果没有找到匹配项，则设为NaN
        if !has_match {
            result[i] = f64::NAN;
        } else {
            result[i] = sum;
        }
    }
    
    // 第2步：如果需要过滤频繁出现的volume，统计非NaN点的volume出现频率
    // 检查处理完第一步后是否超时
    if let Err(_) = check_timeout(timeout_seconds) {
        return Ok(vec![f64::NAN; n]);
    }
    
    if filter_ratio > 0.0 {
        let mut volume_counts: std::collections::BTreeMap<OrderedFloat<f64>, usize> = std::collections::BTreeMap::new();
        
        // 只统计非NaN点的volume频率
        for i in 0..n {
            // 如果该点是NaN，则跳过不统计
            if result[i].is_nan() {
                continue;
            }
            
            let current_volume = volumes[i];
            *volume_counts.entry(OrderedFloat(current_volume)).or_insert(0) += 1;
        }
        
        // 过滤出现频率最高的前30%的volume类型
        if !volume_counts.is_empty() {
            // 将volume按出现频率从高到低排序
            let mut volume_freq: Vec<(OrderedFloat<f64>, usize)> = volume_counts
                .iter()
                .map(|(k, v)| (*k, *v))
                .collect();
                
            // 按频率降序排序
            volume_freq.sort_by(|a, b| b.1.cmp(&a.1));
            
            // 计算需要过滤的volume种类数量（根据filter_ratio参数）
            let total_types = volume_freq.len();
            let filter_count = (total_types as f64 * filter_ratio).ceil() as usize;
            
            // 确保至少过滤一种如果有多种类型
            let filter_count = if filter_count == 0 && total_types > 0 { 1 } else { filter_count };
            
            // 选取出现频率最高的前几种volume类型
            let volume_to_filter: Vec<f64> = volume_freq
                .iter()
                .take(filter_count)
                .map(|(vol, _)| vol.into_inner())
                .collect();
            
            // 将这些高频率volume对应的行设为NaN
            if !volume_to_filter.is_empty() {
                for i in 0..n {
                    // 定期检查超时（每100次计算）
                    if i % 100 == 0 {
                        if let Err(_) = check_timeout(timeout_seconds) {
                            return Ok(vec![f64::NAN; n]); // 如果超时，返回全NaN的数组
                        }
                    }
                    
                    // 浮点数比较需要小心处理
                    if volume_to_filter.iter().any(|&v| (v - volumes[i]).abs() < 1e-10) {
                        // 使用f64::NAN表示NaN
                        result[i] = f64::NAN;
                    }
                }
            }
        }
    }
    
    Ok(result)
}
#[pyfunction]
#[pyo3(signature = (times, prices, volumes, flags, time_window=0.1))]
pub fn find_follow_volume_sum_same_price_and_flag(
    times: PyReadonlyArray1<f64>,
    prices: PyReadonlyArray1<f64>,
    volumes: PyReadonlyArray1<f64>,
    flags: PyReadonlyArray1<i64>,
    time_window: f64
) -> PyResult<Vec<f64>> {
    let times = times.as_array();
    let times: Vec<f64> = times.iter().map(|&x| x / 1.0e9).collect();
    let prices = prices.as_array();
    let volumes = volumes.as_array();
    let flags = flags.as_array();
    let n = times.len();
    let mut result = vec![0.0; n];
    
    // 对每个点，检查之后time_window秒内的点
    for i in 0..n {
        let current_time = times[i];
        let current_price = prices[i];
        let current_volume = volumes[i];
        let current_flag = flags[i];
        let mut sum = current_volume; // 包含当前点的成交量
        
        // 检查之后的点
        for j in (i + 1)..n {
            // 如果时间差超过time_window秒，退出内层循环
            if times[j] - current_time > time_window {
                break;
            }
            // 如果价格和成交量都相同，加入总和
            if (prices[j] - current_price).abs() < 1e-10 && 
               (volumes[j] - current_volume).abs() < 1e-10 &&
               flags[j] == current_flag {
                sum += volumes[j];
            }
        }
        
        result[i] = sum;
    }
    
    Ok(result)
}

/// 标记每一行在其后0.1秒内具有相同price和volume的行组。
/// 对于同一个时间窗口内的相同交易组，标记相同的组号。
/// 组号从1开始递增，每遇到一个新的交易组就分配一个新的组号。
/// 
/// 参数说明：
/// ----------
/// times : array_like
///     时间戳数组（单位：秒）
/// prices : array_like
///     价格数组
/// volumes : array_like
///     成交量数组
/// time_window : float, optional
///     时间窗口大小（单位：秒），默认为0.1
/// 
/// 返回值：
/// -------
/// numpy.ndarray
///     整数数组，表示每行所属的组号。0表示不属于任何组。
/// 
/// Python调用示例：
/// ```python
/// import pandas as pd
/// import numpy as np
/// from rust_pyfunc import mark_follow_groups
/// 
/// # 创建示例DataFrame
/// df = pd.DataFrame({
///     'exchtime': [1.0, 1.05, 1.08, 1.15, 1.2],
///     'price': [10.0, 10.0, 10.0, 11.0, 10.0],
///     'volume': [100, 100, 100, 200, 100]
/// })
/// 
/// # 标记协同交易组
/// df['group'] = mark_follow_groups(
///     df['exchtime'].values,
///     df['price'].values,
///     df['volume'].values
/// )
/// print(df)
/// #    exchtime  price  volume  group
/// # 0     1.00   10.0    100      1  # 第一组的起始点
/// # 1     1.05   10.0    100      1  # 属于第一组
/// # 2     1.08   10.0    100      1  # 属于第一组
/// # 3     1.15   11.0    200      2  # 第二组的起始点
/// # 4     1.20   10.0    100      3  # 第三组的起始点
/// ```
#[pyfunction]
#[pyo3(signature = (times, prices, volumes, time_window=0.1))]
pub fn mark_follow_groups(
    times: PyReadonlyArray1<f64>,
    prices: PyReadonlyArray1<f64>,
    volumes: PyReadonlyArray1<f64>,
    time_window: f64
) -> PyResult<Vec<i32>> {
    let times = times.as_array();
    let times: Vec<f64> = times.iter().map(|&x| x / 1.0e9).collect();
    let prices = prices.as_array();
    let volumes = volumes.as_array();
    let n = times.len();
    let mut result = vec![0; n];
    let mut current_group = 0;
    
    // 对每个未标记的点，检查是否可以形成新组
    for i in 0..n {
        // 如果当前点已经被标记，跳过
        if result[i] != 0 {
            continue;
        }
        
        let current_time = times[i];
        let current_price = prices[i];
        let current_volume = volumes[i];
        let mut has_group = false;
        
        // 检查之后的点，看是否有相同的交易
        for j in i..n {
            // 如果时间差超过time_window秒，退出内层循环
            if j > i && times[j] - current_time > time_window {
                break;
            }
            
            // 如果价格和成交量都相同
            if (prices[j] - current_price).abs() < 1e-10 && 
               (volumes[j] - current_volume).abs() < 1e-10 {
                // 如果还没有分配组号，分配新组号
                if !has_group {
                    current_group += 1;
                    has_group = true;
                }
                // 标记这个点属于当前组
                result[j] = current_group;
            }
        }
    }
    
    Ok(result)
}

/// 标记每一行在其后time_window秒内具有相同flag、price和volume的行组。
/// 对于同一个时间窗口内的相同交易组，标记相同的组号。
/// 组号从1开始递增，每遇到一个新的交易组就分配一个新的组号。
/// 
/// 参数说明：
/// ----------
/// times : array_like
///     时间戳数组（单位：秒）
/// prices : array_like
///     价格数组
/// volumes : array_like
///     成交量数组
/// flags : array_like
///     主买卖标志数组
/// time_window : float, optional
///     时间窗口大小（单位：秒），默认为0.1
/// 
/// 返回值：
/// -------
/// numpy.ndarray
///     整数数组，表示每行所属的组号。0表示不属于任何组。
/// 
/// Python调用示例：
/// ```python
/// import pandas as pd
/// import numpy as np
/// from rust_pyfunc import mark_follow_groups_with_flag
/// 
/// # 创建示例DataFrame
/// df = pd.DataFrame({
///     'exchtime': [1.0, 1.05, 1.08, 1.15, 1.2],
///     'price': [10.0, 10.0, 10.0, 11.0, 10.0],
///     'volume': [100, 100, 100, 200, 100],
///     'flag': [66, 66, 66, 83, 66]
/// })
/// 
/// # 标记协同交易组
/// df['group'] = mark_follow_groups_with_flag(
///     df['exchtime'].values,
///     df['price'].values,
///     df['volume'].values,
///     df['flag'].values
/// )
/// print(df)
/// #    exchtime  price  volume  flag  group
/// # 0     1.00   10.0    100    66      1  # 第一组的起始点
/// # 1     1.05   10.0    100    66      1  # 属于第一组
/// # 2     1.08   10.0    100    66      1  # 属于第一组
/// # 3     1.15   11.0    200    83      2  # 第二组的起始点
/// # 4     1.20   10.0    100    66      3  # 第三组的起始点
/// ```
#[pyfunction]
#[pyo3(signature = (times, prices, volumes, flags, time_window=0.1))]
pub fn mark_follow_groups_with_flag(
    times: PyReadonlyArray1<f64>,
    prices: PyReadonlyArray1<f64>,
    volumes: PyReadonlyArray1<f64>,
    flags: PyReadonlyArray1<i64>,
    time_window: f64
) -> PyResult<Vec<i32>> {
    let times = times.as_array();
    let times: Vec<f64> = times.iter().map(|&x| x / 1.0e9).collect();
    let prices = prices.as_array();
    let volumes = volumes.as_array();
    let flags = flags.as_array();
    let n = times.len();
    let mut result = vec![0; n];
    let mut current_group = 0;
    
    // 对每个未标记的点，检查是否可以形成新组
    for i in 0..n {
        // 如果当前点已经被标记，跳过
        if result[i] != 0 {
            continue;
        }
        
        let current_time = times[i];
        let current_price = prices[i];
        let current_volume = volumes[i];
        let current_flag = flags[i];
        let mut has_group = false;
        
        // 检查之后的点，看是否有相同的交易
        for j in i..n {
            // 如果时间差超过time_window秒，退出内层循环
            if j > i && times[j] - current_time > time_window {
                break;
            }
            
            // 如果价格、成交量和标志都相同
            if (prices[j] - current_price).abs() < 1e-10 && 
               (volumes[j] - current_volume).abs() < 1e-10 &&
               flags[j] == current_flag {
                // 如果还没有分配组号，分配新组号
                if !has_group {
                    current_group += 1;
                    has_group = true;
                }
                // 标记这个点属于当前组
                result[j] = current_group;
            }
        }
    }
    
    Ok(result)
}

/// 计算每一行在其后指定时间窗口内的价格变动能量，并找出首次达到最终能量一半时所需的时间。
/// 
/// 参数说明：
/// ----------
/// times : array_like
///     时间戳数组（单位：秒）
/// prices : array_like
///     价格数组
/// time_window : float, optional
///     时间窗口大小（单位：秒），默认为5.0
/// 
/// 返回值：
/// -------
/// numpy.ndarray
///     浮点数数组，表示每行达到最终能量一半所需的时间（秒）。
///     如果在时间窗口内未达到一半能量，或者最终能量为0，则返回time_window值。
/// 
/// Python调用示例：
/// ```python
/// import pandas as pd
/// import numpy as np
/// from rust_pyfunc import find_half_energy_time
/// 
/// # 创建示例DataFrame
/// df = pd.DataFrame({
///     'exchtime': [1.0, 1.1, 1.2, 1.3, 1.4],
///     'price': [10.0, 10.2, 10.5, 10.3, 10.1]
/// })
/// 
/// # 计算达到一半能量所需时间
/// df['half_energy_time'] = find_half_energy_time(
///     df['exchtime'].values,
///     df['price'].values,
///     time_window=5.0
/// )
/// print(df)
/// #    exchtime  price  half_energy_time
/// # 0      1.0   10.0              2.1  # 在2.1秒时达到5秒能量的一半
/// # 1      1.1   10.2              1.9  # 在1.9秒时达到5秒能量的一半
/// # 2      1.2   10.5              1.8  # 在1.8秒时达到5秒能量的一半
/// # 3      1.3   10.3              1.7  # 在1.7秒时达到5秒能量的一半
/// # 4      1.4   10.1              5.0  # 未达到5秒能量的一半
/// ```
/// 计算每个时间点价格达到时间窗口内最大变动一半所需的时间。
/// 
/// 该函数首先在每个时间点的后续时间窗口内找到价格的最大上涨和下跌幅度，
/// 然后确定主要方向（上涨或下跌），最后计算价格首次达到该方向最大变动一半时所需的时间。
/// 
/// # 参数说明
/// 
/// * `times` - 时间戳数组（单位：秒）
/// * `prices` - 价格数组
/// * `time_window` - 时间窗口大小（单位：秒），默认为5.0
/// 
/// # 返回值
/// 
/// 浮点数数组，表示每个时间点达到最大变动一半所需的时间（秒）。
/// 如果在时间窗口内未达到一半变动，则返回time_window值。
/// 
/// # 特殊情况处理
/// 
/// * 当价格为NaN或Inf时，对应结果为NaN
/// * 当时间点后续数据不足时，返回time_window
/// * 当最大价格变动为0时，返回time_window
/// 
/// # 性能
/// 
/// 该函数使用并行处理加速计算，在大规模数据集上比等效的Python实现快约5-8倍。
/// 
/// # 示例
/// 
/// ```python
/// import pandas as pd
/// import numpy as np
/// from rust_pyfunc import find_half_extreme_time
/// 
/// # 创建示例DataFrame
/// df = pd.DataFrame({
///     'exchtime': [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0],
///     'price': [10.0, 10.2, 10.5, 10.3, 10.1, 10.0, 9.8, 9.5, 9.3, 9.2, 9.0]
/// })
/// 
/// # 计算达到最大变动一半所需时间
/// df['half_extreme_time'] = find_half_extreme_time(
///     df['exchtime'].values,
///     df['price'].values,
///     time_window=1.0  # 1秒时间窗口
/// )
/// print(df)
/// ```
/// 
/// 输出结果：
/// ```
///     exchtime  price  half_extreme_time
/// 0        1.0   10.0               0.3  # 在0.3秒时达到最大上涨(0.5)的一半(0.25)
/// 1        1.1   10.2               0.3  # 在0.3秒时达到最大上涨(0.3)的一半(0.15)
/// 2        1.2   10.5               1.0  # 最大变动为下跌，但未达到一半
/// 3        1.3   10.3               0.4  # 在0.4秒时达到最大下跌(0.5)的一半(0.25)
/// 4        1.4   10.1               0.3  # 在0.3秒时达到最大下跌(0.6)的一半(0.3)
/// 5        1.5   10.0               0.2  # 在0.2秒时达到最大下跌(0.5)的一半(0.25)
/// 6        1.6    9.8               0.3  # 在0.3秒时达到最大下跌(0.8)的一半(0.4)
/// 7        1.7    9.5               0.2  # 在0.2秒时达到最大下跌(0.7)的一半(0.35)
/// 8        1.8    9.3               0.2  # 在0.2秒时达到最大下跌(0.5)的一半(0.25)
/// 9        1.9    9.2               0.1  # 在0.1秒时达到最大下跌(0.2)的一半(0.1)
/// 10       2.0    9.0               1.0  # 时间窗口内没有后续数据
/// ```
/// 
/// # 实际股票数据应用场景
/// 
/// ```python
/// import pandas as pd
/// import numpy as np
/// from rust_pyfunc import find_half_extreme_time
/// 
/// # 读取股票分钟数据
/// df = pd.read_csv('stock_data.csv')
/// df['time'] = pd.to_datetime(df['datetime']).astype('int64') // 10**9
/// 
/// # 计算每个时间点在未来30分钟(1800秒)内达到最大变动一半所需的时间
/// df['half_extreme_time'] = find_half_extreme_time(
///     df['time'].values,
///     df['close'].values,
///     time_window=1800.0
/// )
/// 
/// # 分析结果
/// print(f"平均达到半程时间: {df['half_extreme_time'].mean():.2f} 秒")
/// print(f"中位达到半程时间: {df['half_extreme_time'].median():.2f} 秒")
/// ```
#[pyfunction]
#[pyo3(signature = (times, prices, time_window=5.0, direction="ignore", timeout_seconds=None))]
pub fn find_half_extreme_time(
    times: PyReadonlyArray1<f64>,
    prices: PyReadonlyArray1<f64>,
    time_window: f64,
    direction: &str,
    timeout_seconds: Option<f64>
) -> PyResult<Vec<f64>> {
    // 记录开始时间
    let start_time = Instant::now();
    
    // 检查超时的闭包函数
    let check_timeout = |timeout: Option<f64>| -> Result<(), TimeoutError> {
        if let Some(timeout) = timeout {
            let elapsed = start_time.elapsed().as_secs_f64();
            if elapsed > timeout {
                return Err(TimeoutError {
                    message: "半极端时间计算超时".to_string(),
                    duration: elapsed,
                });
            }
        }
        Ok(())
    };
    
    // 提取数组数据
    let times = times.as_array();
    let times: Vec<f64> = times.iter().map(|&x| x / 1.0e9).collect();
    let prices = prices.as_array();
    let n = times.len();
    
    // 预分配结果向量，初始值为 time_window
    let mut result = vec![time_window; n];
    
    // 检查初始化后是否超时
    if let Err(_) = check_timeout(timeout_seconds) {
        // 如果已经超时，返回全NaN数组
        return Ok(vec![f64::NAN; n]);
    }
    
    // 计算每个时间点的半极端时间（使用单线程处理）
    let chunk_size = 100; // 每处理100个元素检查一次超时
    
    for i in 0..n {
        // 每处理chunk_size个元素检查一次超时
        if i % chunk_size == 0 {
            if let Err(_) = check_timeout(timeout_seconds) {
                // 如果超时，返回全NaN数组
                return Ok(vec![f64::NAN; n]);
            }
        }
        
        let current_time = times[i];
        let current_price = prices[i];
        
        // 检查价格是否为NaN或Inf
        if !current_price.is_finite() {
            result[i] = f64::NAN;
            continue;
        }
        
        let mut max_up = 0.0;      // 最大上涨幅度
        let mut max_down = 0.0;    // 最大下跌幅度
        
        // 第一次遍历：找到时间窗口内的最大上涨和下跌幅度
        for j in i..n {
            let time_diff = times[j] - current_time;
            if time_diff > time_window {
                break;
            }
            
            // 检查价格是否为NaN或Inf
            if !prices[j].is_finite() {
                continue; // 跳过无效价格
            }
            
            // 计算价格变动比率
            let price_ratio = (prices[j] - current_price) / current_price;
            
            // 更新最大上涨和下跌幅度
            if price_ratio > max_up {
                max_up = price_ratio;
            } else if price_ratio < -max_down {
                max_down = -price_ratio;
            }
        }
        
        // 确定主要方向（上涨或下跌），根据方向参数筛选
        let (target_ratio, dir_value) = match direction {
            "pos" => {
                // 只考虑上涨
                if max_up <= 0.0 {
                    // 没有上涨，设置为 NaN
                    result[i] = f64::NAN;
                    continue;
                }
                (max_up, 1.0) // 上涨
            },
            "neg" => {
                // 只考虑下跌
                if max_down <= 0.0 {
                    // 没有下跌，设置为 NaN
                    result[i] = f64::NAN;
                    continue;
                }
                 (max_down, -1.0) // 下跌
            },
            _ => {
                // 全部方向，选择变动更大的
                if max_up > max_down {
                    (max_up, 1.0)  // 上涨
                } else {
                    (max_down, -1.0) // 下跌
                }
            }
        };
        
        // 如果目标变动为0，保持默认值并继续
        if target_ratio <= 0.0 {
            continue;
        }
        
        let half_ratio = target_ratio / 2.0 * dir_value;
        
        // 第二次遍历：找到首次达到一半变动的时间
        for j in i..n {
            let time_diff = times[j] - current_time;
            if time_diff > time_window {
                break;
            }
            
            // 检查价格是否为NaN或Inf
            if !prices[j].is_finite() {
                continue; // 跳过无效价格
            }
            
            // 计算当前时刻的价格变动比率
            let price_ratio = (prices[j] - current_price) / current_price;
            
            // 检查是否达到目标变动的一半
            if (dir_value > 0.0 && price_ratio >= half_ratio) || 
               (dir_value < 0.0 && price_ratio <= half_ratio) {
                result[i] = time_diff;
                break; // 找到后跳出循环
            }
        }
        // 如果没有找到达到一半变动的时间，保持默认值
    }
    
    // 最终检查一次超时
    if let Err(_) = check_timeout(timeout_seconds) {
        return Ok(vec![f64::NAN; n]);
    }
    
    Ok(result)
}

/// 计算每个时间点价格达到时间窗口内最终能量一半所需的时间。
/// 
/// 该函数首先计算时间窗口结束时的能量（价格变动的绝对值比率），
/// 然后计算第一次达到该能量一半所需的时间。
/// 
/// # 参数说明
/// 
/// * `times` - 时间戳数组（单位：秒）
/// * `prices` - 价格数组
/// * `time_window` - 时间窗口大小（单位：秒），默认为5.0
/// 
/// # 返回值
/// 
/// 浮点数数组，表示每个时间点达到最终能量一半所需的时间（秒）。
/// 如果在时间窗口内未达到一半能量，则返回time_window值。
/// 如果最终能量为0，则返回0。
/// 
/// # 特殊情况处理
/// 
/// * 当价格为NaN或Inf时，对应结果为NaN
/// * 当最终能量为0时，结果为0
/// * 当时间窗口内无法计算出最终能量时，结果为time_window
/// 
/// # 性能
/// 
/// 该函数使用并行处理加速计算，在大规模数据集上比等效的Python实现快约20-100倍。
/// 
/// # 示例
/// 
/// ```python
/// import pandas as pd
/// import numpy as np
/// from rust_pyfunc import find_half_energy_time
/// 
/// # 创建示例DataFrame
/// df = pd.DataFrame({
///     'exchtime': [1.0, 1.1, 1.2, 1.3, 1.4],
///     'price': [10.0, 10.2, 10.5, 10.3, 10.1]
/// })
/// 
/// # 计算达到一半能量所需时间
/// df['half_energy_time'] = find_half_energy_time(
///     df['exchtime'].values,
///     df['price'].values,
///     time_window=5.0
/// )
/// print(df)
/// ```
#[pyfunction]
#[pyo3(signature = (times, prices, time_window=5.0, direction="ignore", timeout_seconds=None))]
pub fn find_half_energy_time(
    times: PyReadonlyArray1<f64>,
    prices: PyReadonlyArray1<f64>,
    time_window: f64,
    direction: &str,
    timeout_seconds: Option<f64>
) -> PyResult<Vec<f64>> {
    // 记录开始时间
    let start_time = Instant::now();
    
    // 检查超时的闭包函数
    let check_timeout = |timeout: Option<f64>| -> Result<(), TimeoutError> {
        if let Some(timeout) = timeout {
            let elapsed = start_time.elapsed().as_secs_f64();
            if elapsed > timeout {
                return Err(TimeoutError {
                    message: "半能量时间计算超时".to_string(),
                    duration: elapsed,
                });
            }
        }
        Ok(())
    };
    
    // 提取数组数据
    let times = times.as_array();
    let times: Vec<f64> = times.iter().map(|&x| x / 1.0e9).collect();
    let prices = prices.as_array();
    let n = times.len();
    
    // 预分配结果向量，初始值为 time_window
    let mut result = vec![time_window; n];
    
    // 检查初始化后是否超时
    if let Err(_) = check_timeout(timeout_seconds) {
        // 如果已经超时，返回全NaN数组
        return Ok(vec![f64::NAN; n]);
    }
    
    // 计算每个时间点的半能量时间（使用单线程处理）
    let chunk_size = 100; // 每处理100个元素检查一次超时
    
    for i in 0..n {
        // 每处理chunk_size个元素检查一次超时
        if i % chunk_size == 0 {
            if let Err(_) = check_timeout(timeout_seconds) {
                // 如果超时，返回全NaN数组
                return Ok(vec![f64::NAN; n]);
            }
        }
        
        let current_time = times[i];
        let current_price = prices[i];
        
        // 检查价格是否为NaN或Inf
        if !current_price.is_finite() {
            result[i] = f64::NAN;
            continue;
        }
        
        let mut final_energy = 0.0;
        
        // 首先计算time_window秒后的最终能量
        for j in i..n {
            // 跳过当前点
            if j == i {
                continue;
            }
            
            // 检查时间间隔
            let time_diff = times[j] - current_time;
            if time_diff < time_window {
                continue;
            }
            
            // 检查价格是否为NaN或Inf
            if !prices[j].is_finite() {
                continue;
            }
            
            // 获取价格变动
            let price_change = prices[j] - current_price;
            
            // 根据方向参数筛选
            match direction {
                "pos" if price_change <= 0.0 => {
                    result[i] = f64::NAN;
                    break;
                },
                "neg" if price_change >= 0.0 => {
                    result[i] = f64::NAN;
                    break;
                },
                _ => {}
            }
            
            // 计算价格变动比率的绝对值
            final_energy = price_change.abs() / current_price;
            break;
        }
        
        // 如果最终能量为0，设置为0.0并返回
        if final_energy <= 0.0 {
            result[i] = 0.0;
            continue;
        }
        
        // 计算一半能量的阈值
        let half_energy = final_energy / 2.0;
        
        // 再次遍历，找到第一次达到一半能量的时间
        for j in i..n {
            if j == i {
                continue;
            }
            
            let time_diff = times[j] - current_time;
            if time_diff > time_window {
                break;
            }
            
            // 检查价格是否为NaN或Inf
            if !prices[j].is_finite() {
                continue;
            }
            
            // 计算当前时刻的能量
            let price_ratio = (prices[j] - current_price).abs() / current_price;
            
            // 如果达到一半能量
            if price_ratio >= half_energy {
                result[i] = time_diff;
                break;
            }
        }
        // 如果没有找到达到一半能量的时间，保持默认值 time_window
    }
    
    // 最终检查一次超时
    if let Err(_) = check_timeout(timeout_seconds) {
        return Ok(vec![f64::NAN; n]);
    }
    
    Ok(result)
}

/// 计算每个大单与其临近小单之间的时间间隔均值。
/// 
/// 参数说明：
/// ----------
/// volumes : numpy.ndarray
///     交易量数组
/// exchtimes : numpy.ndarray
///     交易时间数组（单位：纳秒）
/// large_quantile : float
///     大单的分位点阈值
/// small_quantile : float
///     小单的分位点阈值
/// near_number : int
///     每个大单要考虑的临近小单数量
/// exclude_same_time : bool, optional
///     是否排除时间戳与大单完全相同的小单，默认为False
///     当为True时，计算附近小单时不包含时间戳与大单完全相同的小单
///     当为False时，包含与大单时间戳完全相同的小单
/// 
/// 返回值：
/// -------
/// numpy.ndarray
///     浮点数数组，与输入volumes等长。对于大单，返回其与临近小单的时间间隔均值（秒）；
///     对于非大单，返回NaN。
/// 
/// Python调用示例：
/// ```python
/// import pandas as pd
/// import numpy as np
/// from rust_pyfunc import calculate_large_order_nearby_small_order_time_gap
/// 
/// # 创建示例DataFrame
/// df = pd.DataFrame({
///     'exchtime': [1.0e9, 1.1e9, 1.2e9, 1.3e9, 1.4e9],  # 纳秒时间戳
///     'volume': [100, 10, 200, 20, 150]
/// })
/// 
/// # 计算大单与临近小单的时间间隔
/// df['time_gap'] = calculate_large_order_nearby_small_order_time_gap(
///     df['volume'].values,
///     df['exchtime'].values,
///     large_quantile=0.7,  # 70%分位点以上为大单
///     small_quantile=0.3,  # 30%分位点以下为小单
///     near_number=2        # 每个大单考虑最近的2个小单
/// )
/// print(df)
/// #    exchtime  volume    time_gap
/// # 0      1.0e9    100        NaN  # 不是大单
/// # 1      1.1e9     10        NaN  # 不是大单
/// # 2      1.2e9    200       0.15  # 大单，与附近2个小单的时间间隔均值
/// # 3      1.3e9     20        NaN  # 不是大单
/// # 4      1.4e9    150        NaN  # 不是大单
/// 
/// # 计算大单与临近小单的时间间隔（排除时间戳相同的小单）
/// df['time_gap_exc'] = calculate_large_order_nearby_small_order_time_gap(
///     df['volume'].values,
///     df['exchtime'].values,
///     large_quantile=0.7,  # 70%分位点以上为大单
///     small_quantile=0.3,  # 30%分位点以下为小单
///     near_number=2,       # 每个大单考虑最近的2个小单
///     exclude_same_time=True  # 排除时间戳相同的小单
/// )
/// ```
/// 计算滚动DTW距离：计算son中每一行与其前n分钟片段和dragon的DTW距离。
/// 
/// 参数说明：
/// ----------
/// son : array_like
///     主要时间序列，将在此序列上滚动计算DTW距离
/// dragon : array_like
///     参考时间序列，用于计算DTW距离的模板
/// exchtime : array_like
///     时间戳数组，必须与son长度相同
/// minute_back : int
///     滚动窗口大小，以分钟为单位，表示每次计算使用的历史数据长度
/// 
/// 返回值：
/// -------
/// numpy.ndarray
///     与son等长的数组，包含每个点的DTW距离，其中部分位置可能为NaN
///     （如果相应位置的历史数据不足以计算DTW距离）
/// 
/// Python调用示例：
/// ```python
/// import pandas as pd
/// import numpy as np
/// from rust_pyfunc import rolling_dtw_distance
/// 
/// # 准备数据
/// times = pd.date_range('2023-01-01', periods=100, freq='T')
/// son = pd.Series(np.sin(np.linspace(0, 10, 100)), index=times)
/// dragon = pd.Series([0, 0.5, 1, 0.5, 0, -0.5, -1, -0.5, 0]) # 一个波形模板
/// 
/// # 计算滚动DTW距离
/// # 每个点与其前5分钟数据和dragon的DTW距离
/// result = rolling_dtw_distance(son.values, dragon.values, times.astype(np.int64).values, 5)
/// dtw_series = pd.Series(result, index=times)
/// ```
#[pyfunction]
#[pyo3(signature = (son, dragon, exchtime, minute_back))]
pub fn rolling_dtw_distance(
    son: PyReadonlyArray1<f64>,
    dragon: PyReadonlyArray1<f64>,
    exchtime: PyReadonlyArray1<f64>,
    minute_back: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    // 转换为Rust类型
    let son_array = son.as_array();
    let dragon_array = dragon.as_array();
    let exchtime_array = exchtime.as_array();
    
    // 检查输入数据长度是否一致
    if son_array.len() != exchtime_array.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "son和exchtime长度必须相同"
        ));
    }
    
    // 对dragon进行标准化
    let dragon_vec: Vec<f64> = dragon_array.iter().cloned().collect();
    let dragon_mean = dragon_vec.iter().sum::<f64>() / dragon_vec.len() as f64;
    let dragon_std = (dragon_vec.iter().map(|&x| (x - dragon_mean).powi(2)).sum::<f64>() / dragon_vec.len() as f64).sqrt();
    
    let dragon_normalized: Vec<f64> = dragon_vec.iter()
                                          .map(|&x| (x - dragon_mean) / dragon_std)
                                          .collect();
    
    // 结果数组，初始化为NaN
    let mut result = Array1::from_elem(son_array.len(), f64::NAN);
    
    // 定义一分钟的纳秒数
    let one_minute_ns: f64 = 60.0 * 1_000_000_000.0;
    
    // 使用 rayon 并行处理
    use rayon::prelude::*;
    use rayon::ThreadPoolBuilder;
    
    // 创建一个线程数受限的线程池
    let max_threads = 10;
    
    // 创建自定义线程池
    let pool = ThreadPoolBuilder::new()
        .num_threads(max_threads)
        .build()
        .unwrap();
    
    // 使用自定义线程池执行并行计算
    let results: Vec<_> = pool.install(|| {
        // 将son和exchtime数据转换为向量，以便在线程间共享
        let son_vec: Vec<f64> = son_array.iter().cloned().collect();
        let exchtime_vec: Vec<f64> = exchtime_array.iter().cloned().collect();
        let dragon_norm = dragon_normalized.clone();
        
        // 在指定线程池中并行计算
        (0..son_array.len())
            .into_par_iter()
            .map(|i| {
                let current_time = exchtime_vec[i];
                let start_time = current_time - (minute_back * one_minute_ns);
                
                // 收集当前时间点前minute_back分钟内的数据
                let mut segment: Vec<f64> = Vec::new();
                for j in 0..=i {
                    if exchtime_vec[j] > start_time && exchtime_vec[j] <= current_time {
                        segment.push(son_vec[j]);
                    }
                }
                
                // 如果分段数据和dragon都至少有2个点，则计算DTW距离
                if segment.len() > 1 && dragon_norm.len() > 1 {
                    // 对segment进行标准化
                    let segment_mean = segment.iter().sum::<f64>() / segment.len() as f64;
                    let segment_std = (segment.iter().map(|&x| (x - segment_mean).powi(2)).sum::<f64>() / segment.len() as f64).sqrt();
                    
                    // 确保std不为零，避免除以零的错误
                    if segment_std > 0.0 {
                        let segment_normalized: Vec<f64> = segment.iter()
                                                          .map(|&x| (x - segment_mean) / segment_std)
                                                          .collect();
                        
                        // 计算DTW距离
                        match fast_dtw_distance(segment_normalized, dragon_norm.clone(), None, Some(1.0)) {
                            Ok(distance) => {
                                // 返回计算结果
                                return (i, distance);
                            },
                            Err(_) => {
                                // 如果计算失败，保持NaN值
                            }
                        }
                    }
                }
                // 未计算成功时返回原始值
                (i, f64::NAN)
            })
            .collect()
    });
    
    // 将并行结果填入结果数组
    for (idx, val) in results {
        result[idx] = val;
    }
    
    // 将结果转换为NumPy数组返回
    Ok(result.into_pyarray(son.py()).to_owned())
}

#[pyfunction]
#[pyo3(signature = (volumes, exchtimes, large_quantile, small_quantile, near_number, exclude_same_time=false, order_type="small", flags=None, flag_filter="ignore"))]
pub fn calculate_large_order_nearby_small_order_time_gap(
    volumes: PyReadonlyArray1<f64>,
    exchtimes: PyReadonlyArray1<f64>,
    large_quantile: f64,
    small_quantile: f64,
    near_number: i64,
    exclude_same_time: bool,
    order_type: &str,
    flags: Option<PyReadonlyArray1<i64>>,
    flag_filter: &str
) -> PyResult<Vec<f64>> {
    // 转换为Rust类型处理
    let volumes = volumes.as_array();
    let exchtimes = exchtimes.as_array();
    let n = volumes.len() as i64;
    
    // 如果输入数组为空，直接返回空结果
    if n == 0 {
        return Ok(Vec::new());
    }
    
    // 确保输入数组长度一致
    if exchtimes.len() as i64 != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "volumes和exchtimes的长度必须一致"
        ));
    }
    
    // 处理flags参数
    let flags_vec = if let Some(flags_array) = flags {
        // 确保flags长度与volumes和exchtimes一致
        if flags_array.len() as i64 != n {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "flags的长度必须与volumes和exchtimes一致"
            ));
        }
        flags_array.as_slice().unwrap_or(&[]).to_vec()
    } else {
        // 如果没有提供flags，则创建默认值
        vec![0; n as usize]
    };
    
    // 转换时间戳为秒单位，并复制为向量
    let times: Vec<f64> = exchtimes.iter().map(|&x| x / 1.0e9).collect();
    let volumes_vec: Vec<f64> = volumes.iter().cloned().collect();
    
    // 计算分位点
    let mut volumes_sorted = volumes_vec.clone();
    volumes_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    
    let large_threshold_idx = ((n as f64) * large_quantile).ceil() as i64;
    let small_threshold_idx = ((n as f64) * small_quantile).floor() as i64;
    
    // 确保索引不越界
    let large_threshold_idx = if large_threshold_idx >= n { n - 1 } else { large_threshold_idx };
    let small_threshold_idx = if small_threshold_idx >= n { n - 1 } else { small_threshold_idx };
    
    let large_threshold = volumes_sorted[large_threshold_idx as usize];
    let small_threshold = volumes_sorted[small_threshold_idx as usize];
    
    // 标记大单和目标订单
    let mut is_large_order = vec![false; n as usize];
    let mut is_target_order = vec![false; n as usize];
    
    for i in 0..(n as usize) {
        if volumes[i] >= large_threshold {
            is_large_order[i] = true;
        }
        
        match order_type {
            "small" => {
                // 标记小单
                if volumes[i] <= small_threshold {
                    is_target_order[i] = true;
                }
            },
            "mid" => {
                // 标记中间订单
                if volumes[i] > small_threshold && volumes[i] < large_threshold {
                    is_target_order[i] = true;
                }
            },
            "full" => {
                // 标记所有小于large_threshold的订单
                if volumes[i] < large_threshold {
                    is_target_order[i] = true;
                }
            },
            _ => {
                // 默认为"small"，标记小单
                if volumes[i] <= small_threshold {
                    is_target_order[i] = true;
                }
            }
        }
    }
    
    // 结果数组，初始化为NaN
    let mut result = vec![f64::NAN; n as usize];
    
    // 对每个大单计算与临近小单的时间间隔
    for i in 0..n as usize {
        if !is_large_order[i] {
            continue; // 跳过非大单
        }
        
        let large_time = times[i];
        let mut time_gaps = Vec::new();
        
        // 获取当前大单的flag
        let large_flag = flags_vec[i];
        
        // 查找前面的目标订单
        let mut before_count = 0;
        for j in (0..i).rev() {
            if is_target_order[j] {
                // 根据flag_filter判断是否满足条件
                let flag_match = match flag_filter {
                    "same" => flags_vec[j] == large_flag,
                    "diff" => flags_vec[j] != large_flag,
                    _ => true, // "ignore"或其他值，忽略flag判断
                };
                
                if flag_match {
                    let time_diff = (large_time - times[j]).abs();
                    // 如果排除相同时间戳的订单，且时间差为0，则跳过
                    if exclude_same_time && time_diff == 0.0 {
                        continue;
                    }
                    time_gaps.push(time_diff);
                    before_count += 1;
                    if before_count >= near_number as i64 {
                        break;
                    }
                }
            }
        }
        
        // 查找后面的目标订单
        let mut after_count = 0;
        for j in (i+1)..n as usize {
            if is_target_order[j] {
                // 根据flag_filter判断是否满足条件
                let flag_match = match flag_filter {
                    "same" => flags_vec[j] == large_flag,
                    "diff" => flags_vec[j] != large_flag,
                    _ => true, // "ignore"或其他值，忽略flag判断
                };
                
                if flag_match {
                    let time_diff = (times[j] - large_time).abs();
                    // 如果排除相同时间戳的订单，且时间差为0，则跳过
                    if exclude_same_time && time_diff == 0.0 {
                        continue;
                    }
                    time_gaps.push(time_diff);
                    after_count += 1;
                    if after_count >= near_number as i64 {
                        break;
                    }
                }
            }
        }
        
        // 如果找到了至少一个小单，选择最小的near_number个间隔计算均值
        if !time_gaps.is_empty() {
            // 对时间间隔进行排序
            time_gaps.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            
            // 取出最小的near_number个（或全部如果数量不足）
            let count = std::cmp::min(near_number as usize, time_gaps.len());
            let min_gaps: Vec<f64> = time_gaps.iter().take(count).cloned().collect();
            
            // 计算这些最小间隔的均值
            let avg_gap = min_gaps.iter().sum::<f64>() / min_gaps.len() as f64;
            result[i] = avg_gap;
        }
    }
    Ok(result)
}

