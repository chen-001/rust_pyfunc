use pyo3::prelude::*;
use numpy::{PyArray1, PyReadonlyArray1, IntoPyArray};
use std::collections::HashMap;
use rayon::prelude::*;

/// 按日期对因子值进行分组
/// 
/// # Arguments
/// * `dates` - 日期时间戳数组
/// * `factors` - 因子值数组
/// * `groups_num` - 分组数量，默认为10
/// 
/// # Returns
/// * 每个观测值对应的分组号(1到groups_num)
#[pyfunction]
#[pyo3(signature = (dates, factors, groups_num = 10))]
pub fn factor_grouping<'py>(
    py: Python<'py>,
    dates: PyReadonlyArray1<i64>,
    factors: PyReadonlyArray1<f64>,
    groups_num: usize,
) -> PyResult<Py<PyArray1<i32>>> {
    let dates_slice = dates.as_slice()?;
    let factors_slice = factors.as_slice()?;
    
    if dates_slice.len() != factors_slice.len() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "dates and factors must have the same length"
        ));
    }
    
    let n = dates_slice.len();
    let mut result = vec![0i32; n];
    
    // 按日期分组数据
    let mut date_groups: HashMap<i64, Vec<(usize, f64)>> = HashMap::new();
    
    for (idx, (&date, &factor)) in dates_slice.iter().zip(factors_slice.iter()).enumerate() {
        if !factor.is_nan() {
            date_groups.entry(date).or_insert_with(Vec::new).push((idx, factor));
        }
    }
    
    // 并行处理每个日期的分组
    let grouped_results: Vec<_> = date_groups
        .into_par_iter()
        .map(|(_, mut indices_factors)| {
            // 按因子值排序
            indices_factors.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            
            let count = indices_factors.len();
            if count == 0 {
                return Vec::new();
            }
            
            // 使用更精确的分组逻辑，模拟Python版本
            let each_group_size = ((count as f64) / (groups_num as f64)).round() as usize;
            let each_group_size = if each_group_size == 0 { 1 } else { each_group_size };
            
            let mut group_assignments = Vec::new();
            
            // 创建分组列表，和Python版本完全一致
            let mut group_list = Vec::new();
            for group_id in 1..=groups_num {
                for _ in 0..each_group_size {
                    group_list.push(group_id as i32);
                }
            }
            
            // 如果分组列表长度不足，用最后一组补齐
            while group_list.len() < count {
                group_list.push(groups_num as i32);
            }
            
            // 截取到所需长度
            group_list.truncate(count);
            
            // 分配分组
            for (i, &(original_idx, _)) in indices_factors.iter().enumerate() {
                group_assignments.push((original_idx, group_list[i]));
            }
            
            group_assignments
        })
        .collect();
        
    // 将结果写入结果数组
    for group_assignments in grouped_results {
        for (idx, group) in group_assignments {
            result[idx] = group;
        }
    }
    
    Ok(result.into_pyarray(py).to_owned())
}


/// 按日期计算ret和fac的分组相关系数
/// 
/// # Arguments
/// * `dates` - 日期时间戳数组
/// * `ret` - 收益率数组
/// * `fac` - 因子值数组
/// 
/// # Returns
/// * (unique_dates, full_corr, low_corr, high_corr) - 四个数组
///   - unique_dates: 唯一日期
///   - full_corr: 每日全体数据的ret和fac排序值相关系数
///   - low_corr: 每日fac小于中位数部分的ret和fac排序值相关系数
///   - high_corr: 每日fac大于中位数部分的ret和fac排序值相关系数
#[pyfunction]
pub fn factor_correlation_by_date<'py>(
    py: Python<'py>,
    dates: PyReadonlyArray1<i64>,
    ret: PyReadonlyArray1<f64>,
    fac: PyReadonlyArray1<f64>,
) -> PyResult<(Py<PyArray1<i64>>, Py<PyArray1<f64>>, Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
    let dates_slice = dates.as_slice()?;
    let ret_slice = ret.as_slice()?;
    let fac_slice = fac.as_slice()?;
    
    let n = dates_slice.len();
    if n != ret_slice.len() || n != fac_slice.len() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "dates, ret, and fac must have the same length"
        ));
    }
    
    // 按日期分组数据
    let mut date_groups: HashMap<i64, Vec<(f64, f64)>> = HashMap::new();
    
    for i in 0..n {
        let date = dates_slice[i];
        let ret_val = ret_slice[i];
        let fac_val = fac_slice[i];
        
        // 只处理非NaN的数据
        if !ret_val.is_nan() && !fac_val.is_nan() {
            date_groups.entry(date).or_insert_with(Vec::new).push((ret_val, fac_val));
        }
    }
    
    // 对每个日期计算相关系数
    let mut results: Vec<(i64, f64, f64, f64)> = date_groups
        .into_iter()
        .map(|(date, data)| {
            if data.len() < 2 {
                return (date, f64::NAN, f64::NAN, f64::NAN);
            }
            
            // 计算fac的中位数
            let mut fac_vals: Vec<f64> = data.iter().map(|(_, f)| *f).collect();
            fac_vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let median = if fac_vals.len() % 2 == 0 {
                let mid = fac_vals.len() / 2;
                (fac_vals[mid - 1] + fac_vals[mid]) / 2.0
            } else {
                fac_vals[fac_vals.len() / 2]
            };
            
            // 1. 全体数据的相关系数
            let full_corr = calculate_rank_correlation(&data);
            
            // 2. fac小于中位数部分的相关系数
            let low_data: Vec<(f64, f64)> = data.iter()
                .filter(|(_, f)| *f < median)
                .cloned()
                .collect();
            let low_corr = if low_data.len() < 2 {
                f64::NAN
            } else {
                calculate_rank_correlation(&low_data)
            };
            
            // 3. fac大于中位数部分的相关系数
            let high_data: Vec<(f64, f64)> = data.iter()
                .filter(|(_, f)| *f > median)
                .cloned()
                .collect();
            let high_corr = if high_data.len() < 2 {
                f64::NAN
            } else {
                calculate_rank_correlation(&high_data)
            };
            
            (date, full_corr, low_corr, high_corr)
        })
        .collect();
    
    // 按日期排序
    results.sort_by_key(|&(date, _, _, _)| date);
    
    // 分离结果
    let unique_dates: Vec<i64> = results.iter().map(|&(date, _, _, _)| date).collect();
    let full_corr: Vec<f64> = results.iter().map(|&(_, full, _, _)| full).collect();
    let low_corr: Vec<f64> = results.iter().map(|&(_, _, low, _)| low).collect();
    let high_corr: Vec<f64> = results.iter().map(|&(_, _, _, high)| high).collect();
    
    Ok((
        unique_dates.into_pyarray(py).to_owned(),
        full_corr.into_pyarray(py).to_owned(),
        low_corr.into_pyarray(py).to_owned(),
        high_corr.into_pyarray(py).to_owned(),
    ))
}

/// 计算排序值相关系数的辅助函数
fn calculate_rank_correlation(data: &[(f64, f64)]) -> f64 {
    if data.len() < 2 {
        return f64::NAN;
    }
    
    // 获取ret和fac的排序
    let ret_ranks = get_ranks(&data.iter().map(|(r, _)| *r).collect::<Vec<f64>>());
    let fac_ranks = get_ranks(&data.iter().map(|(_, f)| *f).collect::<Vec<f64>>());
    
    // 计算皮尔逊相关系数
    pearson_correlation(&ret_ranks, &fac_ranks)
}

/// 获取数组的排序值
fn get_ranks(values: &[f64]) -> Vec<f64> {
    let n = values.len();
    let mut indexed_values: Vec<(usize, f64)> = values.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    
    // 按值排序
    indexed_values.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    
    let mut ranks = vec![0.0; n];
    let mut i = 0;
    while i < n {
        let current_value = indexed_values[i].1;
        let mut j = i;
        
        // 找到所有相同值的范围
        while j < n && indexed_values[j].1 == current_value {
            j += 1;
        }
        
        // 计算平均排名
        let avg_rank = ((i + 1) + j) as f64 / 2.0;
        
        // 为所有相同值分配平均排名
        for k in i..j {
            ranks[indexed_values[k].0] = avg_rank;
        }
        
        i = j;
    }
    
    ranks
}

/// 计算皮尔逊相关系数
fn pearson_correlation(x: &[f64], y: &[f64]) -> f64 {
    if x.len() != y.len() || x.len() < 2 {
        return f64::NAN;
    }
    
    let n = x.len() as f64;
    let sum_x: f64 = x.iter().sum();
    let sum_y: f64 = y.iter().sum();
    let sum_xx: f64 = x.iter().map(|&v| v * v).sum();
    let sum_yy: f64 = y.iter().map(|&v| v * v).sum();
    let sum_xy: f64 = x.iter().zip(y.iter()).map(|(&a, &b)| a * b).sum();
    
    let numerator = n * sum_xy - sum_x * sum_y;
    let denominator = ((n * sum_xx - sum_x * sum_x) * (n * sum_yy - sum_y * sum_y)).sqrt();
    
    if denominator == 0.0 {
        f64::NAN
    } else {
        numerator / denominator
    }
}

// #[cfg(test)]
// mod tests {
//     use super::*;
//     use numpy::PyArray1;
//     use pyo3::Python;
    
//     #[test]
//     fn test_factor_grouping() {
//         Python::with_gil(|py| {
//             // 测试数据：两个日期，每个日期有5个因子值
//             let dates = vec![20220101i64, 20220101, 20220101, 20220101, 20220101,
//                            20220102, 20220102, 20220102, 20220102, 20220102];
//             let factors = vec![1.0, 2.0, 3.0, 4.0, 5.0, 
//                              5.0, 4.0, 3.0, 2.0, 1.0];
            
//             let dates_array = PyArray1::from_vec(py, dates);
//             let factors_array = PyArray1::from_vec(py, factors);
            
//             let result = factor_grouping(py, dates_array, factors_array, 5).unwrap();
//             let result_vec = result.to_vec().unwrap();
            
//             println!("Result: {:?}", result_vec);
//             // 应该返回分组号，每个日期内按因子值排序分组
//         });
//     }
// }