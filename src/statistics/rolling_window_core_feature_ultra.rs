use ndarray::Array1;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

/// 超级轻量级优化版滚动窗口核心特征提取
///
/// 基于性能测试结果，专注于最有效的优化：
/// 1. 极致的内联优化
/// 2. 最小化内存分配
/// 3. CPU缓存友好的数据访问模式
/// 4. 编译器友好的代码结构
///
/// 去掉所有复杂缓存机制，专注于算法核心优化
///
/// 参数说明：
/// ----------
/// values : numpy.ndarray
///     输入的一维数组，必须是float64类型
/// window_size : int, 可选
///     滚动窗口的大小，默认为5。必须>=2且<=序列长度
///
/// 返回值：
/// -------
/// tuple[numpy.ndarray, numpy.ndarray]
///     返回两个数组：
///     - 第一个数组：核心特征序列，每个位置对应该窗口中最重要的特征值
///     - 第二个数组：次要特征序列，每个位置对应该窗口中最不重要的特征值
#[pyfunction]
#[pyo3(signature = (values, window_size=5))]
pub fn rolling_window_core_feature_ultra(
    py: Python,
    values: PyReadonlyArray1<f64>,
    window_size: Option<usize>,
) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
    let values = values.as_array();
    let window_size = window_size.unwrap_or(5);
    let n = values.len();

    // 参数验证
    if window_size < 2 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "窗口大小必须至少为2",
        ));
    }

    if window_size > n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "窗口大小不能超过序列长度",
        ));
    }

    // 初始化结果数组
    let mut core_features = Array1::from_elem(n, f64::NAN);
    let mut minor_features = Array1::from_elem(n, f64::NAN);

    if n < window_size {
        return Ok((
            core_features.into_pyarray(py).to_owned(),
            minor_features.into_pyarray(py).to_owned(),
        ));
    }

    // 预分配所有需要的缓冲区，避免运行时内存分配
    let num_windows = n - window_size + 1;
    let mut base_correlations = Vec::with_capacity(num_windows);
    let mut masked_correlations = Vec::with_capacity(num_windows);
    let mut position_impacts = Vec::with_capacity(window_size);

    // 对每个窗口进行分析（从第window_size-1个位置开始）
    for current_idx in (window_size - 1)..n {
        let window_start = current_idx - window_size + 1;

        // 清空复用的向量
        base_correlations.clear();
        position_impacts.clear();

        // 计算基准相关性：当前窗口与所有其他窗口的相关系数
        compute_base_correlations_ultra(&values, window_start, window_size, &mut base_correlations);

        if base_correlations.is_empty() {
            continue;
        }

        // 分析每个位置的重要性
        for mask_pos in 0..window_size {
            // 清空复用的向量
            masked_correlations.clear();

            // 计算masked窗口与其他窗口的相关性
            compute_masked_correlations_ultra(
                &values,
                window_start,
                window_size,
                mask_pos,
                &mut masked_correlations,
            );

            // 计算相关性变化程度
            let impact =
                calculate_correlation_impact_ultra(&base_correlations, &masked_correlations);
            position_impacts.push((mask_pos, impact));
        }

        // 找出最重要和最不重要的位置
        let (core_pos, minor_pos) = find_extreme_positions(&position_impacts);

        // 存储结果
        if let Some(pos) = core_pos {
            core_features[current_idx] = values[window_start + pos];
        }

        if let Some(pos) = minor_pos {
            minor_features[current_idx] = values[window_start + pos];
        }
    }

    Ok((
        core_features.into_pyarray(py).to_owned(),
        minor_features.into_pyarray(py).to_owned(),
    ))
}

/// 超快速基准相关性计算（内联优化）
#[inline(always)]
fn compute_base_correlations_ultra(
    values: &ndarray::ArrayView1<f64>,
    current_window_start: usize,
    window_size: usize,
    base_correlations: &mut Vec<f64>,
) {
    let n = values.len();
    let num_windows = n - window_size + 1;

    for other_start in 0..num_windows {
        if other_start == current_window_start {
            continue;
        }

        let corr =
            compute_correlation_ultra_fast(values, current_window_start, other_start, window_size);

        if !corr.is_nan() {
            base_correlations.push(corr);
        }
    }
}

/// 超快速masked相关性计算（内联优化）
#[inline(always)]
fn compute_masked_correlations_ultra(
    values: &ndarray::ArrayView1<f64>,
    current_window_start: usize,
    window_size: usize,
    mask_pos: usize,
    masked_correlations: &mut Vec<f64>,
) {
    let n = values.len();
    let num_windows = n - window_size + 1;

    for other_start in 0..num_windows {
        if other_start == current_window_start {
            continue;
        }

        let corr = compute_correlation_masked_ultra_fast(
            values,
            current_window_start,
            other_start,
            window_size,
            mask_pos,
        );

        if !corr.is_nan() {
            masked_correlations.push(corr);
        }
    }
}

/// 超快速相关性计算（终极优化版）
/// 专门处理稀疏数据和零方差情况
#[inline(always)]
fn compute_correlation_ultra_fast(
    values: &ndarray::ArrayView1<f64>,
    window1_start: usize,
    window2_start: usize,
    window_size: usize,
) -> f64 {
    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    let mut sum_xy = 0.0;
    let mut sum_x2 = 0.0;
    let mut sum_y2 = 0.0;
    let mut count = 0;

    // 最小方差阈值，用于处理稀疏数据
    const EPSILON: f64 = 1e-12;

    for i in 0..window_size {
        let xi = unsafe { *values.uget(window1_start + i) };
        let yi = unsafe { *values.uget(window2_start + i) };

        // 快速NaN检查（比.is_nan()更快）
        if xi == xi && yi == yi {
            // NaN != NaN
            sum_x += xi;
            sum_y += yi;
            sum_xy += xi * yi;
            sum_x2 += xi * xi;
            sum_y2 += yi * yi;
            count += 1;
        }
    }

    if count < 2 {
        return f64::NAN;
    }

    let n = count as f64;

    // 计算方差
    let numerator = n * sum_xy - sum_x * sum_y;
    let var_x = n * sum_x2 - sum_x * sum_x;
    let var_y = n * sum_y2 - sum_y * sum_y;

    // 处理零方差或接近零方差的情况（稀疏数据的核心问题）
    let var_x_is_zero = var_x.abs() < EPSILON;
    let var_y_is_zero = var_y.abs() < EPSILON;

    if var_x_is_zero && var_y_is_zero {
        // 两个窗口都是常数：如果值相同则完全相关，否则无关
        let mean_x = sum_x / n;
        let mean_y = sum_y / n;
        if (mean_x - mean_y).abs() < EPSILON {
            return 1.0; // 两个相同的常数窗口，完全相关
        } else {
            return 0.0; // 两个不同的常数窗口，无相关
        }
    } else if var_x_is_zero || var_y_is_zero {
        // 只有一个窗口是常数：无相关性
        return 0.0;
    }

    // 正常情况：计算皮尔逊相关系数
    let denom_product = var_x * var_y;
    if denom_product <= 0.0 {
        return 0.0; // 防御性编程：不应该到这里
    }

    let correlation = numerator / denom_product.sqrt();

    // 快速范围限制并处理数值误差
    if correlation.is_nan() {
        return 0.0;
    }
    correlation.clamp(-1.0, 1.0)
}

/// 超快速masked相关性计算（终极优化版）
/// 专门处理稀疏数据和零方差情况
#[inline(always)]
fn compute_correlation_masked_ultra_fast(
    values: &ndarray::ArrayView1<f64>,
    window1_start: usize,
    window2_start: usize,
    window_size: usize,
    mask_pos: usize,
) -> f64 {
    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    let mut sum_xy = 0.0;
    let mut sum_x2 = 0.0;
    let mut sum_y2 = 0.0;
    let mut count = 0;

    // 最小方差阈值，用于处理稀疏数据
    const EPSILON: f64 = 1e-12;

    // 手动展开循环，跳过mask位置
    for i in 0..window_size {
        if i == mask_pos {
            continue;
        }

        let xi = unsafe { *values.uget(window1_start + i) };
        let yi = unsafe { *values.uget(window2_start + i) };

        // 快速NaN检查
        if xi == xi && yi == yi {
            sum_x += xi;
            sum_y += yi;
            sum_xy += xi * yi;
            sum_x2 += xi * xi;
            sum_y2 += yi * yi;
            count += 1;
        }
    }

    if count < 2 {
        return f64::NAN;
    }

    let n = count as f64;

    // 计算方差
    let numerator = n * sum_xy - sum_x * sum_y;
    let var_x = n * sum_x2 - sum_x * sum_x;
    let var_y = n * sum_y2 - sum_y * sum_y;

    // 处理零方差或接近零方差的情况（稀疏数据的核心问题）
    let var_x_is_zero = var_x.abs() < EPSILON;
    let var_y_is_zero = var_y.abs() < EPSILON;

    if var_x_is_zero && var_y_is_zero {
        // 两个masked窗口都是常数：如果值相同则完全相关，否则无关
        let mean_x = sum_x / n;
        let mean_y = sum_y / n;
        if (mean_x - mean_y).abs() < EPSILON {
            return 1.0; // 两个相同的常数窗口，完全相关
        } else {
            return 0.0; // 两个不同的常数窗口，无相关
        }
    } else if var_x_is_zero || var_y_is_zero {
        // 只有一个masked窗口是常数：无相关性
        return 0.0;
    }

    // 正常情况：计算皮尔逊相关系数
    let denom_product = var_x * var_y;
    if denom_product <= 0.0 {
        return 0.0; // 防御性编程：不应该到这里
    }

    let correlation = numerator / denom_product.sqrt();

    // 快速范围限制并处理数值误差
    if correlation.is_nan() {
        return 0.0;
    }
    correlation.clamp(-1.0, 1.0)
}

/// 超快速相关性影响计算
/// 改进稀疏数据下的稳健性
#[inline(always)]
fn calculate_correlation_impact_ultra(
    base_correlations: &[f64],
    masked_correlations: &[f64],
) -> f64 {
    if base_correlations.len() != masked_correlations.len() || base_correlations.is_empty() {
        return f64::NAN;
    }

    let mut sum_diff_sq = 0.0;
    let mut sum_abs_diff = 0.0;
    let mut count = 0;
    const EPSILON: f64 = 1e-12;

    // 计算多种差异度量以提高稳健性
    for (&base, &masked) in base_correlations.iter().zip(masked_correlations.iter()) {
        // 跳过NaN值或无意义的对比
        if base == base && masked == masked {
            // 快速NaN检查
            let diff = base - masked;
            let abs_diff = diff.abs();

            // 忽略微小差异（可能是数值误差）
            if abs_diff > EPSILON {
                sum_diff_sq += diff * diff;
                sum_abs_diff += abs_diff;
                count += 1;
            }
        }
    }

    if count == 0 {
        // 如果所有相关性都相同（包括都是0.0的情况），
        // 说明这个位置的影响很小
        return 0.0;
    }

    // 对于稀疏数据，使用平均绝对差异作为主要度量
    // 这比RMSE更稳健，特别是对于大量接近零的相关性
    let mean_abs_diff = sum_abs_diff / count as f64;
    let rmse = (sum_diff_sq / count as f64).sqrt();

    // 使用加权组合：对于稀疏数据，平均绝对差异更重要
    0.7 * mean_abs_diff + 0.3 * rmse
}

/// 快速查找极值位置
#[inline(always)]
fn find_extreme_positions(position_impacts: &[(usize, f64)]) -> (Option<usize>, Option<usize>) {
    if position_impacts.is_empty() {
        return (None, None);
    }

    let mut min_impact = f64::INFINITY;
    let mut max_impact = f64::NEG_INFINITY;
    let mut core_pos = None;
    let mut minor_pos = None;

    for &(pos, impact) in position_impacts.iter() {
        if impact == impact {
            // 快速NaN检查
            if impact < min_impact {
                min_impact = impact;
                core_pos = Some(pos);
            }
            if impact > max_impact {
                max_impact = impact;
                minor_pos = Some(pos);
            }
        }
    }

    (core_pos, minor_pos)
}
