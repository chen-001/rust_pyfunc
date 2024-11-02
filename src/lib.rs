use pyo3::prelude::*;
// use ndarray::Array1;
// use nalgebra::{DMatrix, DVector};
use ndarray::{s, Array1, Array2, ArrayView1, ArrayView2};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};

use std::collections::{HashMap, HashSet};
// use std::time::Instant;

/// Formats the sum of two numbers as string.
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

fn sakoe_chiba_window(i: usize, j: usize, radius: usize) -> bool {
    (i.saturating_sub(radius) <= j) && (j <= i + radius)
}

fn set_k(b: Option<usize>) -> usize {
    match b {
        Some(value) => value, // 如果b不是None，则c等于b的值加1
        None => 2,            // 如果b是None，则c等于1
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

/// 计算x到y的转移熵，即确定x的过去k期状态后，y的当期状态的不确定性的减少程度
/// 这里将x和y序列分块以离散化，c为分块数量
#[pyfunction]
fn transfer_entropy(x_: Vec<f64>, y_: Vec<f64>, k: usize, c: usize) -> f64 {
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

// #[pyfunction]
// fn ols(x: Vec<Vec<f64>>, y: Vec<f64>) -> (Vec<f64>, f64) {
//     // 将 Vec<Vec<f64>> 转换为 DMatrix
//     let num_rows = x.len();
//     let num_cols = if num_rows > 0 { x[0].len() } else { 0 };
//     let mut x_matrix = DMatrix::zeros(num_rows, num_cols);

//     for (i, row) in x.iter().enumerate() {
//         for (j, &value) in row.iter().enumerate() {
//             x_matrix[(i, j)] = value;
//         }
//     }

//     // 将 Vec<f64> 转换为 DVector
//     let y_vector = DVector::from_vec(y);

//     // 添加常数项
//     let mut x_with_intercept = DMatrix::zeros(num_rows, num_cols + 1);
//     for i in 0..num_rows {
//         x_with_intercept[(i, 0)] = 1.0; // 常数项
//         for j in 0..num_cols {
//             x_with_intercept[(i, j + 1)] = x_matrix[(i, j)]; // 复制原有自变量
//         }
//     }

//     let xt = x_with_intercept.transpose();
//     let xtx = xt.clone() * &x_with_intercept; // 计算 (X'X)
//     let xtx_inv = xtx.try_inverse().expect("Failed to invert matrix");
//     let xt_y = xt.clone() * y_vector.clone(); // 计算 (X'y)
//     let beta = xtx_inv * xt_y; // 计算 OLS 系数

//     // 计算预测值
//     let y_hat = x_with_intercept * beta.clone();

//     // 计算 R²
//     let ss_total: f64 = y_vector.iter().map(|yi| (yi - y_vector.mean()).powi(2)).sum();
//     let ss_residual: f64 = y_vector.iter().zip(y_hat.iter()).map(|(yi, y_hat_i)| (yi - y_hat_i).powi(2)).sum();
//     let r_squared = 1.0 - (ss_residual / ss_total);
//     let beta_out = beta.as_slice().to_vec();

//     (beta_out, r_squared)
// }

// #[pyfunction]
// fn ols2(x: Vec<Vec<f64>>, y: Vec<f64>) -> (Vec<f64>, f64) {
//     // 将 Vec<Vec<f64>> 转换为 DMatrix
//     let num_rows = x.len();
//     let num_cols = if num_rows > 0 { x[0].len() } else { 0 };

//     // 使用 DMatrix::from_vec 构造矩阵
//     let x_matrix = DMatrix::from_vec(num_rows, num_cols, x.iter().flat_map(|row| row.iter().cloned()).collect::<Vec<_>>());

//     // 将 Vec<f64> 转换为 DVector
//     let y_vector = DVector::from_vec(y);

//     // 添加常数项
//     let mut x_with_intercept = DMatrix::zeros(num_rows, num_cols + 1);
//     x_with_intercept.column_mut(0).fill(1.0); // 填充第一列为常数项
//     x_with_intercept.columns_mut(1, num_cols).copy_from(&x_matrix);

//     // 计算 (X'X) 和 (X'y)
//     let xt = x_with_intercept.transpose();
//     let xtx = &xt * &x_with_intercept; // 计算 (X'X)
//     let xty = &xt * &y_vector; // 计算 (X'y)

//     // 计算逆矩阵
//     let xtx_inv = xtx.try_inverse().expect("Failed to invert matrix");

//     // 计算 OLS 系数
//     let beta = xtx_inv * xty;

//     // 计算预测值
//     let y_hat = x_with_intercept * beta.clone();

//     // 计算 R²
//     let ss_total: f64 = y_vector.iter().map(|yi| (yi - y_vector.mean()).powi(2)).sum();
//     let ss_residual: f64 = y_vector.iter()
//         .zip(y_hat.iter())
//         .map(|(yi, y_hat_i)| (yi - y_hat_i).powi(2))
//         .sum();
//     let r_squared = 1.0 - (ss_residual / ss_total);

//     // 将 beta 转换为 Vec<f64>
//     let beta_out = beta.as_slice().to_vec();

//     (beta_out, r_squared)
// }

// #[pyfunction]
// fn ols_regression(x: Vec<Vec<f64>>, y: Vec<f64>) -> Vec<f64> {
//     let start = Instant::now();

//     // 创建带有截距项的设计矩阵
//     let mut x_with_intercept = Vec::with_capacity(x.len());
//     for row in x.iter() {
//         let mut new_row = vec![1.0]; // 添加截距项
//         new_row.extend(row.iter().cloned());
//         x_with_intercept.push(new_row);
//     }
//     println!("Time to create design matrix: {:?}", start.elapsed());

//     let matrix_start = Instant::now();
//     // 将输入数据转换为nalgebra矩阵和向量
//     let x_matrix = DMatrix::from_fn(x_with_intercept.len(), x_with_intercept[0].len(), |i, j| x_with_intercept[i][j]);
//     let y_vector = DVector::from_column_slice(&y);
//     println!("Time to create nalgebra structures: {:?}", matrix_start.elapsed());

//     let calc_start = Instant::now();
//     // 计算 (X^T * X)^(-1) * X^T * y
//     let x_transpose = x_matrix.transpose();
//     let x_transpose_x = &x_transpose * &x_matrix;
//     let x_transpose_x_inv = x_transpose_x.try_inverse().unwrap();
//     let coefficients = &x_transpose_x_inv * &x_transpose * &y_vector;
//     println!("Time to calculate coefficients: {:?}", calc_start.elapsed());

//     let r_squared_start = Instant::now();
//     // 计算R方
//     let y_mean = y_vector.mean();
//     let y_pred = &x_matrix * &coefficients;
//     let ss_tot = y_vector.iter().map(|&yi| (yi - y_mean).powi(2)).sum::<f64>();
//     let ss_res = (&y_vector - &y_pred).map(|e| e.powi(2)).sum();
//     let r_squared = 1.0 - (ss_res / ss_tot);
//     println!("Time to calculate R-squared: {:?}", r_squared_start.elapsed());

//     // 将系数和R方组合成一个向量
//     let mut result = coefficients.data.as_vec().to_owned();
//     result.push(r_squared);

//     println!("Total time: {:?}", start.elapsed());

//     result
// }

// #[pyfunction]
// fn ols_regression2(x: Vec<Vec<f64>>, y: Vec<f64>) -> Vec<f64> {
//     let start = Instant::now();

//     // 创建带有截距项的设计矩阵
//     let x_with_intercept: Array2<f64> = Array2::from_shape_fn((x.len(), x[0].len() + 1), |(i, j)| {
//         if j == 0 { 1.0 } else { x[i][j-1] }
//     });
//     let y_array = Array1::from_vec(y.to_vec());
//     println!("Time to create design matrix: {:?}", start.elapsed());

//     let calc_start = Instant::now();
//     // 计算 (X^T * X)^(-1) * X^T * y
//     let xt_x = x_with_intercept.t().dot(&x_with_intercept);
//     let xt_y = x_with_intercept.t().dot(&y_array);
//     let coefficients = solve_linear_system(&xt_x.view(), &xt_y.view());
//     println!("Time to calculate coefficients: {:?}", calc_start.elapsed());

//     let r_squared_start = Instant::now();
//     // 计算R方
//     let y_mean = y_array.mean().unwrap();
//     let y_pred = x_with_intercept.dot(&coefficients);
//     let ss_tot: f64 = y_array.iter().map(|&yi| (yi - y_mean).powi(2)).sum();
//     let ss_res: f64 = (&y_array - &y_pred).map(|e| e.powi(2)).sum();
//     let r_squared = 1.0 - (ss_res / ss_tot);
//     println!("Time to calculate R-squared: {:?}", r_squared_start.elapsed());

//     // 将系数和R方组合成一个向量
//     let mut result = coefficients.to_vec();
//     result.push(r_squared);

//     println!("Total time: {:?}", start.elapsed());

//     result
// }

// fn solve_linear_system(a: &ArrayView2<f64>, b: &ArrayView1<f64>) -> Array1<f64> {
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
//                 l[[i, j]] = (a[[i, j]] - (0..j).map(|k| l[[i, k]] * u[[k, j]]).sum::<f64>()) / u[[j, j]];
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
//         x[i] = (y[i] - (i+1..b.len()).map(|j| u[[i, j]] * x[j]).sum::<f64>()) / u[[i, i]];
//     }

//     x
// }

#[pyfunction]
fn ols(
    py: Python,
    x: PyReadonlyArray2<f64>,
    y: PyReadonlyArray1<f64>,
) -> PyResult<Py<PyArray1<f64>>> {
    // let start = Instant::now();

    let x: ArrayView2<f64> = x.as_array();
    let y: ArrayView1<f64> = y.as_array();

    // if y.len() == 0 {
    //     return Ok(None.into_py(py));
    // }

    // 创建带有截距项的设计矩阵
    let mut x_with_intercept = Array2::ones((x.nrows(), x.ncols() + 1));
    x_with_intercept.slice_mut(s![.., 1..]).assign(&x);
    // println!("Time to create design matrix: {:?}", start.elapsed());

    // let calc_start = Instant::now();
    // 计算 (X^T * X)^(-1) * X^T * y
    let xt_x = x_with_intercept.t().dot(&x_with_intercept);
    let xt_y = x_with_intercept.t().dot(&y);
    let coefficients = solve_linear_system3(&xt_x.view(), &xt_y.view());
    // println!("Time to calculate coefficients: {:?}", calc_start.elapsed());

    // let r_squared_start = Instant::now();
    // 计算R方
    let y_mean = y.mean().unwrap();
    let y_pred = x_with_intercept.dot(&coefficients);
    let ss_tot: f64 = y.iter().map(|&yi| (yi - y_mean).powi(2)).sum();
    let ss_res: f64 = (&y - &y_pred).map(|e| e.powi(2)).sum();
    let r_squared = 1.0 - (ss_res / ss_tot);
    // println!("Time to calculate R-squared: {:?}", r_squared_start.elapsed());

    // 将系数和R方组合成一个向量
    let mut result = coefficients.to_vec();
    result.push(r_squared);

    // println!("Total time: {:?}", start.elapsed());

    // 将结果转换为 Python 数组
    Ok(Array1::from(result).into_pyarray(py).to_owned())
}

fn solve_linear_system3(a: &ArrayView2<f64>, b: &ArrayView1<f64>) -> Array1<f64> {
    let mut l = Array2::<f64>::zeros((a.nrows(), a.ncols()));
    let mut u = Array2::<f64>::zeros((a.nrows(), a.ncols()));

    // LU decomposition
    for i in 0..a.nrows() {
        for j in 0..a.ncols() {
            if i <= j {
                u[[i, j]] = a[[i, j]] - (0..i).map(|k| l[[i, k]] * u[[k, j]]).sum::<f64>();
                if i == j {
                    l[[i, i]] = 1.0;
                }
            }
            if i > j {
                l[[i, j]] =
                    (a[[i, j]] - (0..j).map(|k| l[[i, k]] * u[[k, j]]).sum::<f64>()) / u[[j, j]];
            }
        }
    }

    // Forward substitution
    let mut y = Array1::<f64>::zeros(b.len());
    for i in 0..b.len() {
        y[i] = b[i] - (0..i).map(|j| l[[i, j]] * y[j]).sum::<f64>();
    }

    // Backward substitution
    let mut x = Array1::<f64>::zeros(b.len());
    for i in (0..b.len()).rev() {
        x[i] = (y[i] - (i + 1..b.len()).map(|j| u[[i, j]] * x[j]).sum::<f64>()) / u[[i, i]];
    }

    x
}

#[pyfunction]
fn min_range_loop(s: Vec<f64>) -> Vec<i32> {
    let mut minranges = Vec::with_capacity(s.len());
    let mut stack = Vec::new();

    for i in 0..s.len() {
        while let Some(&j) = stack.last() {
            if s[j] < s[i] {
                minranges.push((i as i32 - j as i32));
                break;
            }
            stack.pop();
        }
        if stack.is_empty() {
            minranges.push(i as i32 + 1);
        }
        stack.push(i);
    }

    minranges
}

#[pyfunction]
fn max_range_loop(s: Vec<f64>) -> Vec<i32> {
    let mut maxranges = Vec::with_capacity(s.len());
    let mut stack = Vec::new();

    for i in 0..s.len() {
        while let Some(&j) = stack.last() {
            if s[j] > s[i] {
                maxranges.push((i as i32 - j as i32));
                break;
            }
            stack.pop();
        }
        if stack.is_empty() {
            maxranges.push(i as i32 + 1);
        }
        stack.push(i);
    }

    maxranges
}

// fn main() {
//     let k = 1;

//     let te = calculate_transfer_entropy(&x, &y, k);
//     println!("Transfer Entropy from X to Y: {}", te);
// }

/// A Python module implemented in Rust.
// #[pymodule]
// fn rust_pyfunc(m: &Bound<'_, PyModule>) -> PyResult<()> {
//     m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
//     m.add_function(wrap_pyfunction!(dtw_distance, m)?)?;
//     m.add_function(wrap_pyfunction!(transfer_entropy, m)?)?;
//     m.add_function(wrap_pyfunction!(ols, m)?)?;
//     m.add_function(wrap_pyfunction!(ols2, m)?)?;
//     m.add_function(wrap_pyfunction!(ols_regression, m)?)?;
//     m.add_function(wrap_pyfunction!(ols_regression2, m)?)?;
//     m.add_function(wrap_pyfunction!(ols_regression3, m)?)?;
//     Ok(())
// }

// fn transpose(matrix: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
//     let nrows = matrix.len();
//     let ncols = matrix[0].len();

//     (0..ncols)
//         .map(|j| (0..nrows).map(|i| matrix[i][j]).collect())
//         .collect()
// }

// fn matmul(a: &Vec<Vec<f64>>, b: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
//     let nrows_a = a.len();
//     let ncols_a = a[0].len();
//     let ncols_b = b[0].len();

//     (0..nrows_a)
//         .map(|i| {
//             (0..ncols_b)
//                 .map(|j| (0..ncols_a).map(|k| &a[i][k] * &b[k][j]).sum())
//                 .collect()
//         })
//         .collect()
// }

// fn inv(matrix: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
//     let n = matrix.len();

//     let mut identity_matrix: Vec<Vec<f64>> = vec![vec![0.0; n]; n];

//     for i in 0..n {
//         let max_index = (i + 1..n)
//             .max_by_key(|&j| matrix[j][i].abs())
//             .expect("Matrix is not invertible");

//         if matrix[max_index][i] == 0.0 {
//             panic!("Matrix is not invertible");
//         }

//         identity_matrix.swap(i, max_index);

//         let diagonal_element = &matrix[i][i];
//         for j in 0..n {
//             matrix[i][j] /= diagonal_element;
//             identity_matrix[i][j] /= diagonal_element;
//         }

//         for j in 0..n {
//             if j != i {
//                 let factor = &matrix[j][i];
//                 for k in 0..n {
//                     matrix[j][k] -= factor * &matrix[i][k];
//                     identity_matrix[j][k] -= factor * &identity_matrix[i][k];
//                 }
//             }
//         }
//     }

//     identity_matrix
// }

// fn ols_regression(x: &Vec<Vec<f64>>, y: &Vec<f64>) -> Vec<f64> {
//     let n = x.len();
//     let k = x[0].len();

//     let mut x_with_intercept: Vec<Vec<f64>> = vec![vec![1.0; k + 1]; n];
//     for i in 0..n {
//         x_with_intercept[i][1..k + 1].copy_from_slice(&x[i]);
//     }

//     let xt = transpose(&x_with_intercept);
//     let xtx_inv = inv(&matmul(&xt, &x_with_intercept));
//     let beta = matmul(&matmul(&xtx_inv, &xt), &y);

//     beta
// }

fn sentence_to_word_count(sentence: &str) -> HashMap<String, usize> {
    let words: Vec<String> = sentence
        .to_lowercase() // 转为小写，确保不区分大小写
        .replace(".", "") // 去掉句末的句点
        .split_whitespace() // 分词
        .map(|s| s.to_string()) // 转换为 String
        .collect();

    let mut word_count = HashMap::new();
    for word in words {
        *word_count.entry(word).or_insert(0) += 1;
    }

    word_count
}

#[pyfunction]
fn vectorize_sentences(sentence1: &str, sentence2: &str) -> (Vec<usize>, Vec<usize>) {
    let count1 = sentence_to_word_count(sentence1);
    let count2 = sentence_to_word_count(sentence2);

    let mut all_words: HashSet<String> = HashSet::new();
    all_words.extend(count1.keys().cloned());
    all_words.extend(count2.keys().cloned());

    let mut vector1 = Vec::new();
    let mut vector2 = Vec::new();

    for word in &all_words {
        vector1.push(count1.get(word).unwrap_or(&0).clone());
        vector2.push(count2.get(word).unwrap_or(&0).clone());
    }

    (vector1, vector2)
}


#[pymodule]
fn rust_pyfunc(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    m.add_function(wrap_pyfunction!(dtw_distance, m)?)?;
    m.add_function(wrap_pyfunction!(transfer_entropy, m)?)?;
    // m.add_function(wrap_pyfunction!(ols, m)?)?;
    // m.add_function(wrap_pyfunction!(ols2, m)?)?;
    // m.add_function(wrap_pyfunction!(ols_regression, m)?)?;
    // m.add_function(wrap_pyfunction!(ols_regression2, m)?)?;
    m.add_function(wrap_pyfunction!(ols, m)?)?;
    m.add_function(wrap_pyfunction!(min_range_loop, m)?)?;
    m.add_function(wrap_pyfunction!(max_range_loop, m)?)?;
    // m.add_function(wrap_pyfunction!(ols_regression, m)?)?;
    m.add_function(wrap_pyfunction!(vectorize_sentences, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vectorize_sentences_happy_path() {
        let (vec1, vec2) = vectorize_sentences("We expect demand to increase we", "We expect worldwide demand to increase");

        // (vec1, vec2)
        println!("vec1: {:?}, vec2: {:?}", vec1, vec2);
        assert_eq!(vec1, vec![0, 0]); // No unique integer mapping
        assert_eq!(vec2, vec![0, 0]); // No unique integer mapping
    }
    // test_vectorize_sentences_happy_path();
}
