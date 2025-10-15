use numpy::PyReadonlyArray1;
use pyo3::prelude::*;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

// 用于f64的包装器，实现Hash和Eq
#[derive(Debug, Clone, Copy)]
struct F64Key(f64);

impl PartialEq for F64Key {
    fn eq(&self, other: &Self) -> bool {
        self.0.to_bits() == other.0.to_bits()
    }
}

impl Eq for F64Key {}

impl Hash for F64Key {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.to_bits().hash(state);
    }
}

// 统计信息收集结构
#[derive(Debug, Default)]
struct LZDetailedStats {
    phrase_lengths: Vec<usize>,
    phrase_positions: Vec<usize>,
}

impl LZDetailedStats {
    fn new() -> Self {
        Self::default()
    }

    fn add_phrase(&mut self, phrase: &[u8], position: usize) {
        let length = phrase.len();
        self.phrase_lengths.push(length);
        self.phrase_positions.push(position);
    }

    fn calculate_autocorr(&self, data: &[f64]) -> f64 {
        if data.len() < 2 {
            return 0.0;
        }

        let n = data.len();
        let mean = data.iter().sum::<f64>() / n as f64;

        let mut numerator = 0.0;
        let mut denominator = 0.0;

        for i in 1..n {
            let x_dev = data[i] - mean;
            let y_dev = data[i-1] - mean;
            numerator += x_dev * y_dev;
        }

        for i in 0..n {
            let dev = data[i] - mean;
            denominator += dev * dev;
        }

        if denominator == 0.0 {
            0.0
        } else {
            numerator / denominator
        }
    }

    fn calculate_skewness(&self, data: &[f64]) -> f64 {
        if data.len() < 3 {
            return 0.0;
        }

        let n = data.len() as f64;
        let mean = data.iter().sum::<f64>() / n;

        let m2 = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n;
        let m3 = data.iter().map(|&x| (x - mean).powi(3)).sum::<f64>() / n;

        if m2 == 0.0 {
            0.0
        } else {
            m3 / m2.powf(1.5)
        }
    }

    fn calculate_kurtosis(&self, data: &[f64]) -> f64 {
        if data.len() < 4 {
            return 0.0;
        }

        let n = data.len() as f64;
        let mean = data.iter().sum::<f64>() / n;

        let m2 = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n;
        let m4 = data.iter().map(|&x| (x - mean).powi(4)).sum::<f64>() / n;

        if m2 == 0.0 {
            0.0
        } else {
            m4 / m2.powi(2) - 3.0  // 减去3得到超额峰度
        }
    }

    fn calculate_correlation(&self, x: &[f64], y: &[f64]) -> f64 {
        if x.len() != y.len() || x.len() < 2 {
            return 0.0;
        }

        let n = x.len() as f64;
        let mean_x = x.iter().sum::<f64>() / n;
        let mean_y = y.iter().sum::<f64>() / n;

        let mut numerator = 0.0;
        let mut var_x = 0.0;
        let mut var_y = 0.0;

        for i in 0..x.len() {
            let x_dev = x[i] - mean_x;
            let y_dev = y[i] - mean_y;
            numerator += x_dev * y_dev;
            var_x += x_dev * x_dev;
            var_y += y_dev * y_dev;
        }

        if var_x == 0.0 || var_y == 0.0 {
            0.0
        } else {
            numerator / (var_x * var_y).sqrt()
        }
    }

    fn to_dict(&self, py: Python, lz_complexity: f64) -> PyResult<pyo3::Py<pyo3::types::PyDict>> {
        let dict = pyo3::types::PyDict::new(py);

        // 转换长度数据为f64数组
        let lengths_f64: Vec<f64> = self.phrase_lengths.iter().map(|&x| x as f64).collect();
        let positions_f64: Vec<f64> = self.phrase_positions.iter().map(|&x| x as f64).collect();

        // 计算长度统计量
        let length_mean = lengths_f64.iter().sum::<f64>() / lengths_f64.len() as f64;
        let length_var = lengths_f64.iter().map(|&x| (x - length_mean).powi(2)).sum::<f64>() / lengths_f64.len() as f64;
        let length_std = length_var.sqrt();
        let length_skew = self.calculate_skewness(&lengths_f64);
        let length_kurt = self.calculate_kurtosis(&lengths_f64);
        let length_max = lengths_f64.iter().fold(0.0_f64, |a, &b| a.max(b));
        let length_autocorr = self.calculate_autocorr(&lengths_f64);

        // 计算相关系数
        let length_index_corr = self.calculate_correlation(&lengths_f64, &positions_f64);

        // 填充字典
        dict.set_item("length_mean", length_mean)?;
        dict.set_item("length_std", length_std)?;
        dict.set_item("length_skew", length_skew)?;
        dict.set_item("length_kurt", length_kurt)?;
        dict.set_item("length_max", length_max)?;
        dict.set_item("length_autocorr", length_autocorr)?;
        dict.set_item("length_index_corr", length_index_corr)?;
        dict.set_item("lz_complexity", lz_complexity)?;

        Ok(dict.into())
    }
}

/// LZ76增量分解复杂度详细分析计算
///
/// 参数:
/// - seq: 输入序列，可以是1D numpy数组
/// - quantiles: 分位数列表，用于连续变量离散化，None表示序列已经是离散的
/// - normalize: 是否归一化结果
///
/// 返回:
/// - 包含详细统计信息的字典
#[pyfunction]
#[pyo3(signature = (seq, quantiles=None, normalize=true))]
pub fn lz_complexity_detailed(
    seq: PyReadonlyArray1<f64>,
    quantiles: Option<Vec<f64>>,
    normalize: bool,
) -> PyResult<PyObject> {
    let seq_view = seq.as_array();
    let n = seq_view.len();

    if n == 0 {
        return Python::with_gil(|py| {
            let dict = pyo3::types::PyDict::new(py);
            Ok(dict.into())
        });
    }

    // 将序列转换为离散化的符号序列
    let discrete_seq = if let Some(ref q) = quantiles {
        discretize_sequence(&seq_view, q)?
    } else {
        // 优化的符号映射 - 对于常见离散值使用快速路径
        let mut symbols = Vec::with_capacity(n);

        // 预检查是否是简单的0/1序列（最常见情况）
        let is_binary = seq_view.iter().all(|&x| x == 0.0 || x == 1.0);

        if is_binary {
            // 快速路径：直接映射0->0, 1->1
            for &val in seq_view.iter() {
                symbols.push(if val == 0.0 { 0 } else { 1 });
            }
        } else {
            // 一般情况：使用HashMap，但预分配合理容量
            let mut symbol_map = HashMap::with_capacity(16);
            let mut next_symbol = 0u8;

            for &val in seq_view.iter() {
                let key = F64Key(val);
                match symbol_map.get(&key) {
                    Some(&symbol) => symbols.push(symbol),
                    None => {
                        symbol_map.insert(key, next_symbol);
                        symbols.push(next_symbol);
                        next_symbol += 1;
                    }
                }
            }
        }
        symbols
    };

    // 计算LZ复杂度和详细统计
    let (complexity, stats) = calculate_lz_complexity_detailed(&discrete_seq);

    // 如果需要归一化，计算归一化值
    let final_complexity = if normalize {
        let k_eff = get_unique_symbol_count(&discrete_seq) as f64;
        if n <= 1 {
            0.0
        } else {
            let log_n_base_k = (n as f64).ln() / k_eff.ln();
            complexity as f64 * log_n_base_k / n as f64
        }
    } else {
        complexity as f64
    };

    // 将结果转换为字典
    Python::with_gil(|py| {
        let dict = stats.to_dict(py, final_complexity)?;
        Ok(dict.into())
    })
}

/// 使用分位数离散化序列 - 优化版本
fn discretize_sequence(seq: &ndarray::ArrayView1<f64>, quantiles: &[f64]) -> PyResult<Vec<u8>> {
    let n = seq.len();

    // 优化：避免克隆整个序列，直接对原始数据进行分位数计算
    let mut values: Vec<f64> = seq.iter().cloned().collect();

    // 使用更快的排序算法
    values.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

    // 计算分位数阈值
    let mut thresholds = Vec::with_capacity(quantiles.len());
    for &q in quantiles {
        if q < 0.0 || q > 1.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "分位数必须在0到1之间",
            ));
        }
        let idx = ((n - 1) as f64 * q) as usize;
        thresholds.push(values[idx]);
    }

    // 优化离散化过程 - 使用二分查找
    let mut discrete_seq = Vec::with_capacity(n);
    for &val in seq.iter() {
        // 使用迭代器的find方法来找到正确的符号
        let symbol = thresholds
            .iter()
            .enumerate()
            .find(|(_, &threshold)| val <= threshold)
            .map(|(i, _)| i as u8)
            .unwrap_or(thresholds.len() as u8);

        discrete_seq.push(symbol + 1); // 从1开始编号
    }

    Ok(discrete_seq)
}

/// 计算LZ复杂度核心算法 - 带详细统计的版本
fn calculate_lz_complexity_detailed(seq: &[u8]) -> (usize, LZDetailedStats) {
    let n = seq.len();
    if n == 0 {
        return (0, LZDetailedStats::new());
    }

    if n <= 64 {
        return calculate_lz_complexity_simple_detailed(seq);
    }

    calculate_lz_complexity_suffix_automaton_detailed(seq)
}

#[derive(Clone)]
struct SamState {
    len: usize,
    link: Option<usize>,
    transitions: Vec<(u8, usize)>,
}

impl SamState {
    fn new(len: usize) -> Self {
        Self {
            len,
            link: None,
            transitions: Vec::with_capacity(2),
        }
    }

    #[inline]
    fn get(&self, c: u8) -> Option<usize> {
        self.transitions
            .iter()
            .find_map(|&(ch, state)| if ch == c { Some(state) } else { None })
    }

    #[inline]
    fn set(&mut self, c: u8, state: usize) {
        for (ch, target_state) in &mut self.transitions {
            if *ch == c {
                *target_state = state;
                return;
            }
        }
        self.transitions.push((c, state));
    }
}

struct SuffixAutomaton {
    states: Vec<SamState>,
    last: usize,
}

impl SuffixAutomaton {
    fn with_capacity(capacity: usize) -> Self {
        let mut states = Vec::with_capacity(capacity.max(2));
        states.push(SamState::new(0));
        Self { states, last: 0 }
    }

    #[inline]
    fn next_state(&self, state: usize, c: u8) -> Option<usize> {
        self.states[state].get(c)
    }

    fn extend(&mut self, c: u8) {
        let cur_index = self.states.len();
        let cur_len = self.states[self.last].len + 1;
        self.states.push(SamState::new(cur_len));

        let mut p_opt = Some(self.last);
        while let Some(p_idx) = p_opt {
            if self.states[p_idx].get(c).is_some() {
                break;
            }
            self.states[p_idx].set(c, cur_index);
            p_opt = self.states[p_idx].link;
        }

        if let Some(p_idx) = p_opt {
            let q_idx = self.states[p_idx].get(c).expect("transition must exist");
            if self.states[p_idx].len + 1 == self.states[q_idx].len {
                self.states[cur_index].link = Some(q_idx);
            } else {
                let clone_idx = self.states.len();
                let mut cloned_state = self.states[q_idx].clone();
                cloned_state.len = self.states[p_idx].len + 1;
                self.states.push(cloned_state);

                self.states[q_idx].link = Some(clone_idx);
                self.states[cur_index].link = Some(clone_idx);

                let mut current_opt = Some(p_idx);
                while let Some(current) = current_opt {
                    if self.states[current].get(c) == Some(q_idx) {
                        self.states[current].set(c, clone_idx);
                        current_opt = self.states[current].link;
                    } else {
                        break;
                    }
                }
            }
        } else {
            self.states[cur_index].link = Some(0);
        }

        self.last = cur_index;
    }
}

/// 使用后缀自动机计算LZ复杂度 - 带详细统计的版本
fn calculate_lz_complexity_suffix_automaton_detailed(seq: &[u8]) -> (usize, LZDetailedStats) {
    let n = seq.len();
    if n == 0 {
        return (0, LZDetailedStats::new());
    }

    let mut sam = SuffixAutomaton::with_capacity(2 * n);
    let mut complexity = 0;
    let mut i = 0;
    let mut stats = LZDetailedStats::new();

    while i < n {
        let mut state = 0;
        let mut j = i;

        while j < n {
            if let Some(next_state) = sam.next_state(state, seq[j]) {
                state = next_state;
                j += 1;
            } else {
                break;
            }
        }

        if j == n {
            stats.add_phrase(&seq[i..], i);
            complexity += 1;
            break;
        }

        stats.add_phrase(&seq[i..=j], i);
        complexity += 1;
        let phrase_end = j + 1;
        for &symbol in &seq[i..phrase_end] {
            sam.extend(symbol);
        }
        i = phrase_end;
    }

    (complexity, stats)
}

/// 小序列LZ复杂度计算（针对<=64元素的优化）- 带详细统计的版本
fn calculate_lz_complexity_simple_detailed(seq: &[u8]) -> (usize, LZDetailedStats) {
    let n = seq.len();
    if n == 0 {
        return (0, LZDetailedStats::new());
    }

    let mut complexity = 0;
    let mut i = 0;
    let mut stats = LZDetailedStats::new();

    // 对于小序列，使用简单但高效的算法
    while i < n {
        let mut j = i + 1;

        // 找到最长的前缀匹配
        while j <= n {
            let sub_len = j - i;
            let search_end = j - 1;

            if search_end < sub_len {
                break;
            }

            // 检查子串是否在前面出现过
            let mut found = false;
            for start_pos in 0..=(search_end - sub_len) {
                if seq[start_pos..start_pos + sub_len] == seq[i..j] {
                    found = true;
                    break;
                }
            }

            if found && j < n {
                j += 1;
            } else {
                break;
            }
        }

        stats.add_phrase(&seq[i..j], i);
        complexity += 1;
        i = j;
    }

    (complexity, stats)
}

/// 获取序列中唯一符号的数量
fn get_unique_symbol_count(seq: &[u8]) -> usize {
    let mut unique_symbols = std::collections::HashSet::new();
    for &symbol in seq {
        unique_symbols.insert(symbol);
    }
    unique_symbols.len()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lz_complexity_detailed_discrete() {
        // 测试离散序列
        let result = lz_complexity_detailed(
            numpy::array![0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0].view(),
            None,
            true,
        ).unwrap();

        // 结果应该是一个字典
        assert!(result.is_instance_of::<pyo3::types::PyDict>());
    }

    #[test]
    fn test_discretization_detailed() {
        let seq = numpy::array![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0].view();
        let result = lz_complexity_detailed(seq, Some(vec![0.5]), true).unwrap();

        assert!(result.is_instance_of::<pyo3::types::PyDict>());
    }
}