use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

/// 计算pandas Series的排名 (单线程版本)
///
/// # Arguments
/// * `data` - 输入的一维数组数据
/// * `method` - 排名方法: "average", "min", "max", "first", "dense"
/// * `ascending` - 是否升序排列
/// * `na_option` - NaN处理方式: "keep", "top", "bottom"
///
/// # Returns
/// * 排名结果的一维数组
#[pyfunction]
pub fn pandas_series_rank(
    py: Python,
    data: PyReadonlyArray1<f64>,
    method: Option<&str>,
    ascending: Option<bool>,
    na_option: Option<&str>,
) -> PyResult<PyObject> {
    let data_array = data.as_array();
    let data_vec: Vec<f64> = data_array.to_vec();

    let method = method.unwrap_or("average");
    let ascending = ascending.unwrap_or(true);
    let na_option = na_option.unwrap_or("keep");

    let result = rank_series_single_thread(data_vec, method, ascending, na_option);

    Ok(PyArray1::from_vec(py, result).to_object(py))
}

/// 单线程排名算法实现
fn rank_series_single_thread(
    data: Vec<f64>,
    method: &str,
    ascending: bool,
    na_option: &str,
) -> Vec<f64> {
    let n = data.len();
    if n == 0 {
        return vec![];
    }

    // 创建索引-值对，用于排序
    let indexed_data: Vec<(usize, f64)> =
        data.iter().enumerate().map(|(i, &val)| (i, val)).collect();

    // 分离NaN和非NaN值
    let mut nan_indices = vec![];
    let mut valid_data = vec![];

    for (i, val) in indexed_data.iter() {
        if val.is_nan() {
            nan_indices.push(*i);
        } else {
            valid_data.push((*i, *val));
        }
    }

    // 对非NaN值进行排序
    if ascending {
        valid_data.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    } else {
        valid_data.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    }

    // 初始化结果数组
    let mut result = vec![f64::NAN; n];

    // 处理NaN值
    handle_nan_values(&mut result, &nan_indices, &valid_data, method, na_option);

    // 处理非NaN值的排名
    if !valid_data.is_empty() {
        let rank_offset = match na_option {
            "top" => nan_indices.len() as f64,
            _ => 0.0,
        };

        assign_ranks(&mut result, &valid_data, method, rank_offset);
    }

    result
}

/// 处理NaN值的排名
fn handle_nan_values(
    result: &mut Vec<f64>,
    nan_indices: &[usize],
    valid_data: &[(usize, f64)],
    method: &str,
    na_option: &str,
) {
    match na_option {
        "keep" => {
            // NaN保持为NaN，已经在初始化时设置
        }
        "top" => {
            if !nan_indices.is_empty() {
                match method {
                    "average" => {
                        let avg_rank = (1 + nan_indices.len()) as f64 / 2.0;
                        for &idx in nan_indices.iter() {
                            result[idx] = avg_rank;
                        }
                    }
                    "min" => {
                        for &idx in nan_indices.iter() {
                            result[idx] = 1.0;
                        }
                    }
                    "max" => {
                        let max_rank = nan_indices.len() as f64;
                        for &idx in nan_indices.iter() {
                            result[idx] = max_rank;
                        }
                    }
                    "first" => {
                        for (rank, &idx) in nan_indices.iter().enumerate() {
                            result[idx] = (rank + 1) as f64;
                        }
                    }
                    "dense" => {
                        for &idx in nan_indices.iter() {
                            result[idx] = 1.0;
                        }
                    }
                    _ => panic!("不支持的method: {}", method),
                }
            }
        }
        "bottom" => {
            if !nan_indices.is_empty() {
                let nan_start_rank = valid_data.len() + 1;
                match method {
                    "average" => {
                        let avg_rank = (nan_start_rank as f64
                            + (nan_start_rank + nan_indices.len() - 1) as f64)
                            / 2.0;
                        for &idx in nan_indices.iter() {
                            result[idx] = avg_rank;
                        }
                    }
                    "min" => {
                        for &idx in nan_indices.iter() {
                            result[idx] = nan_start_rank as f64;
                        }
                    }
                    "max" => {
                        let max_rank = (nan_start_rank + nan_indices.len() - 1) as f64;
                        for &idx in nan_indices.iter() {
                            result[idx] = max_rank;
                        }
                    }
                    "first" => {
                        for (rank, &idx) in nan_indices.iter().enumerate() {
                            result[idx] = (nan_start_rank + rank) as f64;
                        }
                    }
                    "dense" => {
                        let dense_rank = if valid_data.is_empty() {
                            1.0
                        } else {
                            // 计算非NaN值中有多少个不同的值
                            let mut unique_values =
                                valid_data.iter().map(|(_, val)| *val).collect::<Vec<_>>();
                            unique_values.sort_by(|a, b| {
                                a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                            });
                            unique_values.dedup_by(|a, b| (*a - *b).abs() < 1e-15);
                            (unique_values.len() + 1) as f64
                        };
                        for &idx in nan_indices.iter() {
                            result[idx] = dense_rank;
                        }
                    }
                    _ => panic!("不支持的method: {}", method),
                }
            }
        }
        _ => panic!("不支持的na_option: {}", na_option),
    }
}

/// 分配非NaN值的排名
fn assign_ranks(
    result: &mut Vec<f64>,
    valid_data: &[(usize, f64)],
    method: &str,
    rank_offset: f64,
) {
    match method {
        "average" => {
            assign_average_ranks(result, valid_data, rank_offset);
        }
        "min" => {
            assign_min_ranks(result, valid_data, rank_offset);
        }
        "max" => {
            assign_max_ranks(result, valid_data, rank_offset);
        }
        "first" => {
            assign_first_ranks(result, valid_data, rank_offset);
        }
        "dense" => {
            assign_dense_ranks(result, valid_data, rank_offset);
        }
        _ => panic!("不支持的method: {}", method),
    }
}

/// 分配平均排名
fn assign_average_ranks(result: &mut Vec<f64>, valid_data: &[(usize, f64)], rank_offset: f64) {
    let mut i = 0;
    while i < valid_data.len() {
        let current_val = valid_data[i].1;
        let mut j = i;

        // 找到所有相等的值
        while j < valid_data.len() && (valid_data[j].1 - current_val).abs() < 1e-15 {
            j += 1;
        }

        // 计算平均排名
        let avg_rank = rank_offset + ((i + 1) + j) as f64 / 2.0;

        // 分配排名
        for k in i..j {
            result[valid_data[k].0] = avg_rank;
        }

        i = j;
    }
}

/// 分配最小排名
fn assign_min_ranks(result: &mut Vec<f64>, valid_data: &[(usize, f64)], rank_offset: f64) {
    let mut i = 0;
    while i < valid_data.len() {
        let current_val = valid_data[i].1;
        let min_rank = rank_offset + (i + 1) as f64;
        let mut j = i;

        // 找到所有相等的值并分配最小排名
        while j < valid_data.len() && (valid_data[j].1 - current_val).abs() < 1e-15 {
            result[valid_data[j].0] = min_rank;
            j += 1;
        }

        i = j;
    }
}

/// 分配最大排名
fn assign_max_ranks(result: &mut Vec<f64>, valid_data: &[(usize, f64)], rank_offset: f64) {
    let mut i = 0;
    while i < valid_data.len() {
        let current_val = valid_data[i].1;
        let mut j = i;

        // 找到所有相等的值
        while j < valid_data.len() && (valid_data[j].1 - current_val).abs() < 1e-15 {
            j += 1;
        }

        // 计算最大排名
        let max_rank = rank_offset + j as f64;

        // 分配排名
        for k in i..j {
            result[valid_data[k].0] = max_rank;
        }

        i = j;
    }
}

/// 分配第一次出现排名
fn assign_first_ranks(result: &mut Vec<f64>, valid_data: &[(usize, f64)], rank_offset: f64) {
    for (rank, &(idx, _)) in valid_data.iter().enumerate() {
        result[idx] = rank_offset + (rank + 1) as f64;
    }
}

/// 分配密集排名
fn assign_dense_ranks(result: &mut Vec<f64>, valid_data: &[(usize, f64)], rank_offset: f64) {
    let mut i = 0;
    let mut dense_rank = 1.0 + rank_offset;

    while i < valid_data.len() {
        let current_val = valid_data[i].1;
        let mut j = i;

        // 找到所有相等的值并分配相同的密集排名
        while j < valid_data.len() && (valid_data[j].1 - current_val).abs() < 1e-15 {
            result[valid_data[j].0] = dense_rank;
            j += 1;
        }

        dense_rank += 1.0;
        i = j;
    }
}
