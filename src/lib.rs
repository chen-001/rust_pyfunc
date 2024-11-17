use pyo3::prelude::*;
// use ndarray::Array1;
// use nalgebra::{DMatrix, DVector};
use ndarray::{s, Array1, Array2, ArrayView1, ArrayView2};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
// use pyo3::types::PyString;

use std::collections::{HashMap, HashSet};
// use std::time::Instant;

/// Formats the sum of two numbers as string.
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
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
                minranges.push(i as i32 - j as i32);
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
                maxranges.push(i as i32 - j as i32);
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

// #[pyfunction]
#[pyfunction]
fn vectorize_sentences_list(sentences: Vec<&str>) -> Vec<Vec<usize>> {
    let mut all_words: HashSet<String> = HashSet::new();
    let mut counts: Vec<HashMap<String, usize>> = Vec::new();

    // 收集所有单词并计算每个句子的单词频率
    for sentence in sentences {
        let count = sentence_to_word_count(sentence);
        all_words.extend(count.keys().cloned());
        counts.push(count);
    }

    let mut vectors = Vec::new();

    // 为每个句子构建向量
    for count in counts {
        let mut vector = Vec::new();
        for word in &all_words {
            vector.push(count.get(word).unwrap_or(&0).clone());
        }
        vectors.push(vector);
    }

    vectors
}


#[pyfunction]
fn jaccard_similarity(str1: &str, str2: &str) -> f64 {
    // 将字符串分词并转换为集合
    let set1: HashSet<&str> = str1.split_whitespace().collect();
    let set2: HashSet<&str> = str2.split_whitespace().collect();

    // 计算交集和并集
    let intersection: HashSet<_> = set1.intersection(&set2).cloned().collect();
    let union: HashSet<_> = set1.union(&set2).cloned().collect();

    // 计算 Jaccard 相似度
    if union.is_empty() {
        0.0
    } else {
        intersection.len() as f64 / union.len() as f64
    }
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
    m.add_function(wrap_pyfunction!(vectorize_sentences_list, m)?)?;
    m.add_function(wrap_pyfunction!(jaccard_similarity, m)?)?;
    Ok(())
}

// #[cfg(test)]
// mod tests {
//     use super::*;

//     #[test]
//     fn test_vectorize_sentences_happy_path() {
//         let (vec1, vec2) = vectorize_sentences("We expect demand to increase we", "We expect worldwide demand to increase");

//         // (vec1, vec2)
//         println!("vec1: {:?}, vec2: {:?}", vec1, vec2);
//         assert_eq!(vec1, vec![0, 0]); // No unique integer mapping
//         assert_eq!(vec2, vec![0, 0]); // No unique integer mapping
//     }
//     // test_vectorize_sentences_happy_path();
// }
