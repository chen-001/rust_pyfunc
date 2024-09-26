use pyo3::prelude::*;
use ndarray::Array1;
use std::collections::HashMap;

/// Formats the sum of two numbers as string.
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}




fn sakoe_chiba_window(i: usize, j: usize, radius: usize, len_s1: usize, len_s2: usize) -> bool {
    (i.saturating_sub(radius) <= j) && (j <= i + radius)
}

fn set_k(b: Option<usize>) -> usize {
    match b {
        Some(value) => value, // 如果b不是None，则c等于b的值加1
        None => 2,                // 如果b是None，则c等于1
    }
}

// #[pyo3(signature = (s1, s2, radius=None))]
/// 计算x和y之间的动态时间规整DTW距离，x和y为长度可以不相等的两个序列，计算他们的相似性
/// radius为可选参数，指定了Sakoe-Chiba半径，如果不指定，则默认不考虑Sakoe-Chiba半径
#[pyfunction]
fn dtw_distance(s1: Vec<f64>, s2: Vec<f64>, radius: Option<usize>) -> PyResult<f64> {
    // let radius_after_default = set_c(radius);
    let len_s1 = s1.len();
    let len_s2 = s2.len();
    let mut warp_dist_mat = vec![vec![f64::INFINITY; len_s2 + 1]; len_s1 + 1];
    warp_dist_mat[0][0] = 0.0;

    for i in 1..=len_s1 {
        for j in 1..=len_s2 {
            match radius {
                Some(_) => {
                    if !sakoe_chiba_window(i, j, radius.unwrap(), len_s1, len_s2) {
                        continue;
                    }
                    },
                None => {},
            }
            let cost = (s1[i - 1] - s2[j - 1]).abs() as f64;
            warp_dist_mat[i][j] = cost + warp_dist_mat[i - 1][j].min(
                warp_dist_mat[i][j - 1].min(warp_dist_mat[i - 1][j - 1])
            );
        }
    }
    Ok(warp_dist_mat[len_s1][len_s2])
}



fn discretize(data_: Vec<f64>, c: usize) -> Array1<f64> {
    let data=Array1::from_vec(data_);
    let mut sorted_indices: Vec<usize> = (0..data.len()).collect();
    sorted_indices.sort_by(|&i, &j| data[i].partial_cmp(&data[j]).unwrap());

    let mut discretized = Array1::zeros(data.len());
    let chunk_size = data.len() / c;

    for i in 0..c {
        let start = i * chunk_size;
        let end = if i == c - 1 { data.len() } else { (i + 1) * chunk_size };
        for j in start..end {
            discretized[sorted_indices[j]] = i + 1; // 类别从 1 开始
        }
    }
    let discretized_f64: Array1<f64> = Array1::from(discretized.iter().map(|&x| x as f64).collect::<Vec<f64>>());

    discretized_f64
}



/// 计算x到y的转移熵，即确定x的过去k期状态后，y的当期状态的不确定性的减少程度
/// 这里将x和y序列分块以离散化，c为分块数量
#[pyfunction]
fn transfer_entropy(x_: Vec<f64>, y_: Vec<f64>, k: usize, c:usize) -> f64 {
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
            *conditional_prob.entry((conditional_key.clone(), format!("{:.6}", y[t]))).or_insert(0.0) += prob;
        }
    }

    // 计算转移熵
    let mut te = 0.0;
    for (key, &count) in joint_prob.iter() {
        let (x_state, y_state) = key;
        let p_xy = count as f64 / (n - k) as f64;
        let p_y_given_x = conditional_prob.get(&(x_state.clone(), y_state.clone())).unwrap_or(&0.0);
        let p_y = marginal_prob.get(y_state).unwrap_or(&0);

        if *p_y > 0 {
            te += p_xy * (p_y_given_x / *p_y as f64).log2();
        }
    }

    te
}

// fn main() {
//     let k = 1;

//     let te = calculate_transfer_entropy(&x, &y, k);
//     println!("Transfer Entropy from X to Y: {}", te);
// }

/// A Python module implemented in Rust.
#[pymodule]
fn rust_pyfunc(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    m.add_function(wrap_pyfunction!(dtw_distance, m)?)?;
    m.add_function(wrap_pyfunction!(transfer_entropy, m)?)?;
    Ok(())
}