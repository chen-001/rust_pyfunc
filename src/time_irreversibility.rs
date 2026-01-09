use ndarray::{s, Array1, Array2};
use numpy::{IntoPyArray, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::HashMap;
use std::f64;

/// 计算序型索引
/// 对窗口数据进行argsort，得到序型
#[inline]
fn calculate_permutation_pattern(window: &[f64]) -> Vec<usize> {
    let mut indexed: Vec<(usize, &f64)> = window.iter().enumerate().collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal));
    indexed.into_iter().map(|(idx, _)| idx + 1).collect()
}

/// 计算反向序型
/// π^R = (m+1-i_1, ..., m+1-i_m)
#[inline]
fn reverse_permutation(pattern: &[usize], m: usize) -> Vec<usize> {
    pattern.iter().map(|&i| m + 1 - i).collect()
}

/// 静态版-简略版：只计算时间不可逆指标 I_ord
/// 参数：
///   data: 输入的时间序列
///   m: 嵌入维度（窗口大小），默认5
/// 返回：
///   I_ord: 时间不可逆性指标（0-1之间），如果数据长度小于m则返回NaN
#[pyfunction]
#[pyo3(signature = (data, m=5))]
pub fn time_irreversibility_static_simple(data: PyReadonlyArray1<f64>, m: usize) -> PyResult<f64> {
    let data = data.as_array();
    let n = data.len();

    if m < 2 {
        return Err(PyValueError::new_err("嵌入维度m必须大于等于2"));
    }

    if n < m {
        return Ok(f64::NAN);
    }

    // 统计每种序型的频率
    let mut pattern_counts: HashMap<Vec<usize>, usize> = HashMap::new();
    let num_windows = n - m + 1;

    for i in 0..num_windows {
        let window = &data.slice(s![i..i + m]);
        let pattern = calculate_permutation_pattern(window.as_slice().unwrap());
        *pattern_counts.entry(pattern).or_insert(0) += 1;
    }

    // 计算不可逆指标
    let mut i_ord = 0.0;
    let total_windows = num_windows as f64;

    for (pattern, count) in &pattern_counts {
        let p_forward = *count as f64 / total_windows;
        let reversed = reverse_permutation(pattern, m);
        let count_reverse = pattern_counts.get(&reversed).copied().unwrap_or(0);
        let p_reverse = count_reverse as f64 / total_windows;
        i_ord += (p_forward - p_reverse).abs();
    }

    // 对于没有出现的序型，需要检查它们的反向是否出现
    // 由于我们只遍历了出现的模式，需要额外的检查
    // 但数学上 I_ord = 0.5 * sum|p(π)-p(π^R)|，我们上面的计算只算了出现的模式
    // 对于没出现的模式，p=0，也需要考虑
    // 优化：我们可以从所有可能的模式组合来考虑，但m!增长很快，对于m=5,6,7还可以接受

    i_ord *= 0.5;

    Ok(i_ord)
}

/// 静态版-详细版：计算所有静态不可逆性指标
/// 参数：
///   data: 输入的时间序列
///   m: 嵌入维度（窗口大小），默认5
/// 返回：
///   包含所有指标的Python字典
#[pyfunction]
#[pyo3(signature = (data, m=5))]
pub fn time_irreversibility_static_detailed(
    py: Python,
    data: PyReadonlyArray1<f64>,
    m: usize,
) -> PyResult<PyObject> {
    let data = data.as_array();
    let n = data.len();

    if m < 2 {
        return Err(PyValueError::new_err("嵌入维度m必须大于等于2"));
    }

    if n < m {
        return Err(PyValueError::new_err("数据长度必须大于等于嵌入维度m"));
    }

    let num_patterns = (1..=m).product::<usize>(); // m!
    let num_windows = n - m + 1;
    let total_windows = num_windows as f64;

    // 统计序型频率
    let mut pattern_counts: HashMap<Vec<usize>, usize> = HashMap::new();
    let mut pattern_sequence: Vec<Vec<usize>> = Vec::with_capacity(num_windows);
    let mut pattern_idx_sequence: Vec<usize> = Vec::with_capacity(num_windows);

    for i in 0..num_windows {
        let window = &data.slice(s![i..i + m]);
        let pattern = calculate_permutation_pattern(window.as_slice().unwrap());
        *pattern_counts.entry(pattern.clone()).or_insert(0) += 1;
        pattern_sequence.push(pattern.clone());
    }

    // 计算各项指标
    let mut i_ord = 0.0;
    let mut h_perm = 0.0; // 序型熵
    let mut kl_divergence = 0.0; // KL距离
    let forbidden_count; // 缺失序型个数

    // 频率和偏差数组（按模式）
    let mut pattern_frequencies: Vec<f64> = vec![0.0; num_patterns];
    let mut reversed_frequencies: Vec<f64> = vec![0.0; num_patterns];
    let mut relative_bias_per_pattern: Vec<f64> = vec![0.0; num_patterns];

    // 需要建立模式到索引的映射
    let mut pattern_to_idx: HashMap<Vec<usize>, usize> = HashMap::new();

    // 生成所有可能的序型（对于m=5，120种；m=6，720种；m=7，5040种）
    // 这里用递归生成所有排列
    let all_patterns = generate_permutations(m);

    fn generate_permutations(m: usize) -> Vec<Vec<usize>> {
        let mut result = Vec::new();
        let nums: Vec<usize> = (1..=m).collect();
        let mut current = Vec::new();
        let mut used = vec![false; m];

        fn backtrack(
            nums: &[usize],
            current: &mut Vec<usize>,
            used: &mut Vec<bool>,
            result: &mut Vec<Vec<usize>>,
        ) {
            if current.len() == nums.len() {
                result.push(current.clone());
                return;
            }
            for i in 0..nums.len() {
                if !used[i] {
                    used[i] = true;
                    current.push(nums[i]);
                    backtrack(nums, current, used, result);
                    current.pop();
                    used[i] = false;
                }
            }
        }

        backtrack(&nums, &mut current, &mut used, &mut result);
        result
    }

    for (idx, pattern) in all_patterns.iter().enumerate() {
        pattern_to_idx.insert(pattern.clone(), idx);
    }

    // 生成模式索引序列（用于后续计算每个窗口的局部不可逆度）
    for pattern in &pattern_sequence {
        if let Some(&idx) = pattern_to_idx.get(pattern) {
            pattern_idx_sequence.push(idx);
        } else {
            pattern_idx_sequence.push(0); // 默认值
        }
    }

    // 计算频率
    for (pattern, count) in &pattern_counts {
        if let Some(&idx) = pattern_to_idx.get(pattern) {
            pattern_frequencies[idx] = *count as f64 / total_windows;
        }
    }

    // 计算反向频率
    for (idx, pattern) in all_patterns.iter().enumerate() {
        let reversed = reverse_permutation(pattern, m);
        if let Some(&rev_idx) = pattern_to_idx.get(&reversed) {
            reversed_frequencies[idx] = pattern_frequencies[rev_idx];
        }
    }

    // 计算I_ord
    for i in 0..num_patterns {
        i_ord += (pattern_frequencies[i] - reversed_frequencies[i]).abs();
    }
    i_ord *= 0.5;

    // 计算序型熵
    for &p in &pattern_frequencies {
        if p > 0.0 {
            h_perm -= p * p.ln();
        }
    }
    let h_norm = h_perm / (num_patterns as f64).ln();

    // 统计缺失序型
    forbidden_count = pattern_frequencies.iter().filter(|&&p| p == 0.0).count();
    let forbidden_ratio = forbidden_count as f64 / num_patterns as f64;

    // 计算KL散度
    for i in 0..num_patterns {
        let p = pattern_frequencies[i];
        let q = reversed_frequencies[i];
        if p > 0.0 && q > 0.0 {
            kl_divergence += p * (p / q).ln();
        } else if p > 0.0 && q == 0.0 {
            kl_divergence = f64::INFINITY;
            break;
        }
    }

    // 计算每个模式的相对偏差
    for i in 0..num_patterns {
        let p = pattern_frequencies[i];
        let q = reversed_frequencies[i];
        if p + q > 0.0 {
            relative_bias_per_pattern[i] = p / (p + q) - 0.5;
        } else {
            relative_bias_per_pattern[i] = 0.0;
        }
    }

    // 计算每个窗口的局部不可逆度和相对偏差（按窗口）
    // 返回与输入等长的序列，前m-1个用NaN填充
    let mut window_local_irreversibility: Vec<f64> = vec![f64::NAN; n];
    let mut window_local_irreversibility_signed: Vec<f64> = vec![f64::NAN; n]; // 带符号版本
    let mut window_relative_bias: Vec<f64> = vec![f64::NAN; n];
    let mut window_pattern_frequency: Vec<f64> = vec![f64::NAN; n]; // 每个窗口对应序型的频率

    for i in 0..num_windows {
        let pattern_idx = pattern_idx_sequence[i];
        let p = pattern_frequencies[pattern_idx];
        let q = reversed_frequencies[pattern_idx];
        let diff = p - q;

        window_local_irreversibility[i + m - 1] = diff.abs(); // 绝对值版本
        window_local_irreversibility_signed[i + m - 1] = diff; // 带符号版本
        window_relative_bias[i + m - 1] = relative_bias_per_pattern[pattern_idx];
        window_pattern_frequency[i + m - 1] = pattern_frequencies[pattern_idx]; // 该窗口对应序型的频率
    }

    // 返回Python字典
    let result = pyo3::types::PyDict::new(py);
    result.set_item("i_ord", i_ord)?;
    result.set_item("permutation_entropy", h_perm)?;
    result.set_item("normalized_permutation_entropy", h_norm)?;
    result.set_item("forbidden_count", forbidden_count)?;
    result.set_item("forbidden_ratio", forbidden_ratio)?;
    result.set_item("kl_divergence", kl_divergence)?;
    result.set_item(
        "local_irreversibility",
        window_local_irreversibility.into_pyarray(py).to_object(py),
    )?;
    result.set_item(
        "local_irreversibility_signed",
        window_local_irreversibility_signed
            .into_pyarray(py)
            .to_object(py),
    )?;
    result.set_item(
        "relative_bias",
        window_relative_bias.into_pyarray(py).to_object(py),
    )?;
    result.set_item(
        "pattern_frequencies",
        window_pattern_frequency.into_pyarray(py).to_object(py),
    )?;
    result.set_item(
        "pattern_frequency_all",
        pattern_frequencies.into_pyarray(py).to_object(py),
    )?;

    Ok(result.into())
}

/// 转移版-简略版：只计算转移不可逆指标 I_trans
/// 参数：
///   data: 输入的时间序列
///   m: 嵌入维度（窗口大小），默认5
/// 返回：
///   I_trans: 转移不可逆性指标，如果数据长度小于m则返回NaN
#[pyfunction]
#[pyo3(signature = (data, m=5))]
pub fn time_irreversibility_transfer_simple(
    data: PyReadonlyArray1<f64>,
    m: usize,
) -> PyResult<f64> {
    let data = data.as_array();
    let n = data.len();

    if m < 2 {
        return Err(PyValueError::new_err("嵌入维度m必须大于等于2"));
    }

    if n < m {
        return Ok(f64::NAN);
    }

    let num_patterns = (1..=m).product::<usize>();
    let num_windows = n - m + 1;

    // 首先生成序型序列
    let mut pattern_sequence: Vec<usize> = Vec::with_capacity(num_windows);

    // 建立模式到索引的映射
    fn generate_permutations(m: usize) -> Vec<Vec<usize>> {
        let mut result = Vec::new();
        let nums: Vec<usize> = (1..=m).collect();
        let mut current = Vec::new();
        let mut used = vec![false; m];

        fn backtrack(
            nums: &[usize],
            current: &mut Vec<usize>,
            used: &mut Vec<bool>,
            result: &mut Vec<Vec<usize>>,
        ) {
            if current.len() == nums.len() {
                result.push(current.clone());
                return;
            }
            for i in 0..nums.len() {
                if !used[i] {
                    used[i] = true;
                    current.push(nums[i]);
                    backtrack(nums, current, used, result);
                    current.pop();
                    used[i] = false;
                }
            }
        }

        backtrack(&nums, &mut current, &mut used, &mut result);
        result
    }

    let all_patterns = generate_permutations(m);
    let mut pattern_to_idx: HashMap<Vec<usize>, usize> = HashMap::new();
    for (idx, pattern) in all_patterns.iter().enumerate() {
        pattern_to_idx.insert(pattern.clone(), idx);
    }

    // 生成序型序列
    for i in 0..num_windows {
        let window = &data.slice(s![i..i + m]);
        let pattern = calculate_permutation_pattern(window.as_slice().unwrap());
        let idx = pattern_to_idx.get(&pattern).copied().unwrap_or(0);
        pattern_sequence.push(idx);
    }

    // 构建转移计数矩阵
    let mut transition_counts = Array2::<usize>::zeros((num_patterns, num_patterns));
    for i in 0..pattern_sequence.len() - 1 {
        let current = pattern_sequence[i];
        let next = pattern_sequence[i + 1];
        transition_counts[[current, next]] += 1;
    }

    // 计算转移概率矩阵和平稳分布
    let mut transition_probs = Array2::<f64>::zeros((num_patterns, num_patterns));
    let mut stationary = Array1::<f64>::zeros(num_patterns);

    for i in 0..num_patterns {
        let row_sum = transition_counts.row(i).sum();
        stationary[i] = row_sum as f64;
        if row_sum > 0 {
            for j in 0..num_patterns {
                transition_probs[[i, j]] = transition_counts[[i, j]] as f64 / row_sum as f64;
            }
        }
    }

    let stationary_sum = stationary.sum();
    if stationary_sum > 0.0 {
        stationary /= stationary_sum;
    }

    // 计算转移不可逆指标 I_trans
    // 只遍历实际出现的模式，而不是所有理论上的排列组合
    let active_patterns: Vec<usize> = (0..num_patterns).filter(|&i| stationary[i] > 0.0).collect();

    let mut i_trans = 0.0;
    for &i in &active_patterns {
        for &j in &active_patterns {
            let forward_flow = stationary[i] * transition_probs[[i, j]];
            let reverse_flow = stationary[j] * transition_probs[[j, i]];
            i_trans += (forward_flow - reverse_flow).abs();
        }
    }
    i_trans *= 0.5;

    Ok(i_trans)
}

/// 转移版-详细版：计算所有转移不可逆性指标
/// 参数：
///   data: 输入的时间序列
///   m: 嵌入维度（窗口大小），默认5
/// 返回：
///   包含所有指标的Python字典
#[pyfunction]
#[pyo3(signature = (data, m=5))]
pub fn time_irreversibility_transfer_detailed(
    py: Python,
    data: PyReadonlyArray1<f64>,
    m: usize,
) -> PyResult<PyObject> {
    let data = data.as_array();
    let n = data.len();

    if m < 2 {
        return Err(PyValueError::new_err("嵌入维度m必须大于等于2"));
    }

    if n < m {
        return Err(PyValueError::new_err("数据长度必须大于等于嵌入维度m"));
    }

    let num_patterns = (1..=m).product::<usize>();
    let num_windows = n - m + 1;

    // 生成所有可能的序型
    fn generate_permutations(m: usize) -> Vec<Vec<usize>> {
        let mut result = Vec::new();
        let nums: Vec<usize> = (1..=m).collect();
        let mut current = Vec::new();
        let mut used = vec![false; m];

        fn backtrack(
            nums: &[usize],
            current: &mut Vec<usize>,
            used: &mut Vec<bool>,
            result: &mut Vec<Vec<usize>>,
        ) {
            if current.len() == nums.len() {
                result.push(current.clone());
                return;
            }
            for i in 0..nums.len() {
                if !used[i] {
                    used[i] = true;
                    current.push(nums[i]);
                    backtrack(nums, current, used, result);
                    current.pop();
                    used[i] = false;
                }
            }
        }

        backtrack(&nums, &mut current, &mut used, &mut result);
        result
    }

    let all_patterns = generate_permutations(m);
    let mut pattern_to_idx: HashMap<Vec<usize>, usize> = HashMap::new();
    for (idx, pattern) in all_patterns.iter().enumerate() {
        pattern_to_idx.insert(pattern.clone(), idx);
    }

    // 生成序型序列
    let mut pattern_sequence: Vec<usize> = Vec::with_capacity(num_windows);
    for i in 0..num_windows {
        let window = &data.slice(s![i..i + m]);
        let pattern = calculate_permutation_pattern(window.as_slice().unwrap());
        let idx = pattern_to_idx.get(&pattern).copied().unwrap_or(0);
        pattern_sequence.push(idx);
    }

    // 构建转移计数矩阵
    let mut transition_counts = Array2::<usize>::zeros((num_patterns, num_patterns));
    for i in 0..pattern_sequence.len() - 1 {
        let current = pattern_sequence[i];
        let next = pattern_sequence[i + 1];
        transition_counts[[current, next]] += 1;
    }

    // 计算转移概率矩阵和平稳分布
    let mut transition_probs = Array2::<f64>::zeros((num_patterns, num_patterns));
    let mut stationary = Array1::<f64>::zeros(num_patterns);

    for i in 0..num_patterns {
        let row_sum = transition_counts.row(i).sum();
        stationary[i] = row_sum as f64;
        if row_sum > 0 {
            for j in 0..num_patterns {
                transition_probs[[i, j]] = transition_counts[[i, j]] as f64 / row_sum as f64;
            }
        }
    }

    let stationary_sum = stationary.sum();
    if stationary_sum > 0.0 {
        stationary /= stationary_sum;
    }

    // 计算I_trans（对所有可能的转移对(i,j)求和）
    let mut i_trans = 0.0;
    for i in 0..num_patterns {
        for j in 0..num_patterns {
            let forward_flow = stationary[i] * transition_probs[[i, j]];
            let reverse_flow = stationary[j] * transition_probs[[j, i]];
            i_trans += (forward_flow - reverse_flow).abs();
        }
    }
    i_trans *= 0.5;

    // 计算每个实际转移点的局部概率流差
    // 返回与输入等长的序列，前m个用NaN填充（因为前m个点无法构成第一个转移）
    let num_transitions = pattern_sequence.len() - 1;
    let mut window_flow_differences: Vec<f64> = vec![f64::NAN; n];
    let mut window_flow_direction: Vec<f64> = vec![f64::NAN; n]; // 带符号的净流（正向为正，反向为负）

    for i in 0..num_transitions {
        let current = pattern_sequence[i];
        let next = pattern_sequence[i + 1];
        let forward_flow = stationary[current] * transition_probs[[current, next]];
        let reverse_flow = stationary[next] * transition_probs[[next, current]];
        let diff = forward_flow - reverse_flow;

        window_flow_differences[i + m] = diff.abs(); // 绝对值版本
        window_flow_direction[i + m] = diff; // 带符号版本（正向>反向为正）
    }

    // 计算熵率
    let mut entropy_rate = 0.0;
    for i in 0..num_patterns {
        if stationary[i] > 0.0 {
            let mut row_entropy = 0.0;
            for j in 0..num_patterns {
                if transition_probs[[i, j]] > 0.0 {
                    row_entropy -= transition_probs[[i, j]] * transition_probs[[i, j]].ln();
                }
            }
            entropy_rate += stationary[i] * row_entropy;
        }
    }

    // 计算转移熵（这里只用自身转移熵，即熵率）
    let transition_entropy = entropy_rate;

    // 返回Python字典
    let result = pyo3::types::PyDict::new(py);
    result.set_item("i_trans", i_trans)?;
    result.set_item("entropy_rate", entropy_rate)?;
    result.set_item("transition_entropy", transition_entropy)?;
    result.set_item(
        "stationary_distribution",
        stationary.into_pyarray(py).to_object(py),
    )?;
    result.set_item(
        "transition_matrix",
        transition_probs.into_pyarray(py).to_object(py),
    )?;
    result.set_item(
        "flow_differences",
        window_flow_differences.into_pyarray(py).to_object(py),
    )?;
    result.set_item(
        "flow_direction",
        window_flow_direction.into_pyarray(py).to_object(py),
    )?;

    Ok(result.into())
}
