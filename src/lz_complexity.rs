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

/// LZ76增量分解复杂度计算
///
/// 参数:
/// - seq: 输入序列，可以是1D numpy数组
/// - quantiles: 分位数列表，用于连续变量离散化，None表示序列已经是离散的
/// - normalize: 是否归一化结果
///
/// 返回:
/// - LZ复杂度值
#[pyfunction]
#[pyo3(signature = (seq, quantiles=None, normalize=true))]
pub fn lz_complexity(
    seq: PyReadonlyArray1<f64>,
    quantiles: Option<Vec<f64>>,
    normalize: bool,
) -> PyResult<f64> {
    let seq_view = seq.as_array();
    let n = seq_view.len();

    if n == 0 {
        return Ok(0.0);
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

    // 计算LZ复杂度
    let complexity = calculate_lz_complexity(&discrete_seq);

    if !normalize {
        return Ok(complexity as f64);
    }

    // 归一化
    let k_eff = get_unique_symbol_count(&discrete_seq) as f64;
    if n <= 1 {
        return Ok(0.0);
    }

    // 如果使用了分位数离散化，检查唯一符号数量是否足够
    // 分位数列表长度为 m 时，期望至少有 m+1 个类别
    let min_required_categories = quantiles.as_ref().map(|q| q.len() as f64 + 1.0).unwrap_or(2.0);
    if k_eff < min_required_categories {
        return Ok(f64::NAN);
    }

    // 归一化公式: c * log_base(n) / n，其中log_base(n) = ln(n) / ln(base)
    let log_n_base_k = (n as f64).ln() / k_eff.ln();
    Ok(complexity as f64 * log_n_base_k / n as f64)
}

/// 使用分位数离散化序列 - 优化版本
fn discretize_sequence(seq: &ndarray::ArrayView1<f64>, quantiles: &[f64]) -> PyResult<Vec<u8>> {
    let n = seq.len();

    // 优化：避免克隆整个序列，直接对原始数据进行分位数计算
    let mut values: Vec<f64> = seq.iter().cloned().collect();

    // 使用更快的排序算法
    // 处理 NaN 值：将 NaN 排在最后，避免 partial_cmp 返回 None 导致 panic
    values.sort_unstable_by(|a, b| match (a.is_nan(), b.is_nan()) {
        (true, true) => std::cmp::Ordering::Equal,
        (true, false) => std::cmp::Ordering::Greater,
        (false, true) => std::cmp::Ordering::Less,
        (false, false) => a.partial_cmp(b).unwrap(),
    });

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

/// 计算LZ复杂度核心算法 - 后缀自动机版本
fn calculate_lz_complexity(seq: &[u8]) -> usize {
    let n = seq.len();
    if n == 0 {
        return 0;
    }

    if n <= 64 {
        return calculate_lz_complexity_simple(seq);
    }

    calculate_lz_complexity_suffix_automaton(seq)
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

/// 使用后缀自动机计算LZ复杂度
fn calculate_lz_complexity_suffix_automaton(seq: &[u8]) -> usize {
    let n = seq.len();
    if n == 0 {
        return 0;
    }

    let mut sam = SuffixAutomaton::with_capacity(2 * n);
    let mut complexity = 0;
    let mut i = 0;

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
            complexity += 1;
            break;
        }

        complexity += 1;
        let phrase_end = j + 1;
        for &symbol in &seq[i..phrase_end] {
            sam.extend(symbol);
        }
        i = phrase_end;
    }

    complexity
}

/// 后缀数组优化的LZ复杂度计算
#[allow(dead_code)]
fn calculate_lz_complexity_suffix_optimized(seq: &[u8]) -> usize {
    let n = seq.len();
    if n == 0 {
        return 0;
    }

    let mut complexity = 0;
    let mut i = 0;

    // 预计算所有位置的字符出现位置，加速查找
    let mut char_positions = Vec::with_capacity(256);
    for _ in 0..256 {
        char_positions.push(Vec::new());
    }
    for (pos, &ch) in seq.iter().enumerate() {
        char_positions[ch as usize].push(pos);
    }

    while i < n {
        let mut j = i + 1;
        let mut found_match = true;

        // 寻找最长的可匹配前缀
        while j <= n && found_match {
            let current_len = j - i;
            let search_end = j - 1;

            if current_len > search_end {
                break;
            }

            // 检查子串是否在前面出现过
            found_match = check_substring_exists_optimized(seq, i, j, search_end, &char_positions);

            if found_match {
                j += 1;
            }
        }

        complexity += 1;
        i = j;
    }

    complexity
}

/// 优化的子串存在性检查
#[allow(dead_code)]
fn check_substring_exists_optimized(
    seq: &[u8],
    start: usize,
    end: usize,
    search_end: usize,
    char_positions: &Vec<Vec<usize>>,
) -> bool {
    let sub_len = end - start;
    let first_char = seq[start];

    // 如果第一个字符在搜索范围内都没有出现，直接返回false
    for &pos in &char_positions[first_char as usize] {
        if pos < start && pos + sub_len <= search_end {
            // 检查从这个位置开始的完整匹配
            if check_full_match(seq, pos, start, sub_len) {
                return true;
            }
        }
    }

    false
}

/// 检查完整匹配
#[allow(dead_code)]
fn check_full_match(seq: &[u8], pos1: usize, pos2: usize, len: usize) -> bool {
    // 对于短模式使用展开比较
    match len {
        1 => seq[pos1] == seq[pos2],
        2 => seq[pos1] == seq[pos2] && seq[pos1 + 1] == seq[pos2 + 1],
        3 => {
            seq[pos1] == seq[pos2]
                && seq[pos1 + 1] == seq[pos2 + 1]
                && seq[pos1 + 2] == seq[pos2 + 2]
        }
        4 => {
            seq[pos1] == seq[pos2]
                && seq[pos1 + 1] == seq[pos2 + 1]
                && seq[pos1 + 2] == seq[pos2 + 2]
                && seq[pos1 + 3] == seq[pos2 + 3]
        }
        _ => {
            // 对于长模式使用切片比较
            &seq[pos1..pos1 + len] == &seq[pos2..pos2 + len]
        }
    }
}

/// 高性能子串搜索 - 专门为LZ76优化
#[allow(dead_code)]
fn contains_substring_optimized(text: &[u8], pattern: &[u8], search_end: usize) -> bool {
    let pat_len = pattern.len();
    if pat_len == 0 || pat_len > search_end {
        return false;
    }

    // 对于短模式，使用展开循环
    if pat_len == 1 {
        let target = pattern[0];
        return text[..search_end].contains(&target);
    }

    if pat_len == 2 {
        let first = pattern[0];
        let second = pattern[1];
        for i in 0..=(search_end - 2) {
            if text[i] == first && text[i + 1] == second {
                return true;
            }
        }
        return false;
    }

    if pat_len == 3 {
        let first = pattern[0];
        let second = pattern[1];
        let third = pattern[2];
        for i in 0..=(search_end - 3) {
            if text[i] == first && text[i + 1] == second && text[i + 2] == third {
                return true;
            }
        }
        return false;
    }

    // 对于长模式，使用标准库优化
    for i in 0..=(search_end - pat_len) {
        if &text[i..i + pat_len] == pattern {
            return true;
        }
    }

    false
}

/// 高效的子串搜索函数 - 优化版本
#[allow(dead_code)]
fn contains_substring(text: &[u8], pattern: &[u8], search_end: usize) -> bool {
    if pattern.is_empty() || pattern.len() > search_end {
        return false;
    }

    let pat_len = pattern.len();

    // 对于短模式使用两向字符串搜索
    if pat_len <= 4 {
        return contains_substring_short(text, pattern, search_end);
    }

    // 对于长模式使用优化的搜索
    for start_pos in 0..=(search_end - pat_len) {
        if &text[start_pos..start_pos + pat_len] == pattern {
            return true;
        }
    }

    false
}

/// 短模式子串搜索（针对长度<=4的模式优化）
#[allow(dead_code)]
fn contains_substring_short(text: &[u8], pattern: &[u8], search_end: usize) -> bool {
    let pat_len = pattern.len();

    if pat_len == 1 {
        // 单字符搜索
        let target = pattern[0];
        return text[..search_end].contains(&target);
    }

    if pat_len == 2 {
        // 双字符搜索 - 使用位操作加速
        let target1 = pattern[0] as u16;
        let target2 = pattern[1] as u16;
        let target = (target1 << 8) | target2;

        for i in 0..=(search_end - 2) {
            let current = ((text[i] as u16) << 8) | (text[i + 1] as u16);
            if current == target {
                return true;
            }
        }
        return false;
    }

    // 3-4字符搜索
    for start_pos in 0..=(search_end - pat_len) {
        let mut matches = true;
        for k in 0..pat_len {
            if text[start_pos + k] != pattern[k] {
                matches = false;
                break;
            }
        }
        if matches {
            return true;
        }
    }

    false
}

/// 精确的LZ复杂度计算（用于小序列）
#[allow(dead_code)]
fn calculate_lz_complexity_exact(seq: &[u8]) -> usize {
    let n = seq.len();
    if n == 0 {
        return 0;
    }

    let mut complexity = 0;
    let mut i = 0;

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
            for start_pos in 0..=search_end - sub_len {
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

        complexity += 1;
        i = j;
    }

    complexity
}

/// 检查子串seq[i:j]是否在前面seq[:j-1]中出现过 - 超高性能版本
#[allow(dead_code)]
fn is_substring_in_prefix_optimized(
    seq: &[u8],
    start: usize,
    end: usize,
    lookup_table: &mut [usize],
) -> bool {
    let sub_len = end - start;
    let search_end = end - 1; // 对应Python的s[:j-1]

    if sub_len == 0 || search_end < sub_len {
        return false;
    }

    // 使用预查找表加速搜索
    if sub_len == 1 {
        let target = seq[start];
        // 使用查找表快速定位
        if lookup_table[target as usize] > 0 && lookup_table[target as usize] < start {
            return true;
        }

        // 线性搜索并更新查找表
        for i in 0..search_end {
            if seq[i] == target {
                lookup_table[target as usize] = i;
                if i < start {
                    return true;
                }
            }
        }
        return false;
    }

    // 对于长度为2的子串，使用双字符匹配
    if sub_len == 2 {
        let first = seq[start];
        let second = seq[start + 1];

        for i in 0..=(search_end - 2) {
            if seq[i] == first && seq[i + 1] == second {
                return true;
            }
        }
        return false;
    }

    // 对于较长的子串，使用标准库的窗口搜索（已高度优化）
    if let Some(_pos) = seq[0..search_end]
        .windows(sub_len)
        .position(|window| window == &seq[start..end])
    {
        return true;
    }

    false
}

/// 检查子串seq[i:j]是否在前面seq[:j-1]中出现过
/// 优化版本：使用更高效的子串搜索
#[allow(dead_code)]
fn is_substring_in_prefix(seq: &[u8], start: usize, end: usize) -> bool {
    let sub_len = end - start;
    let search_end = end - 1; // 对应Python的s[:j-1]

    if sub_len == 0 || search_end < sub_len {
        return false;
    }

    // 短子串使用直接比较，长子串使用更高效的算法
    if sub_len <= 4 {
        // 对于短子串，直接遍历比较更快
        for i in 0..=search_end - sub_len {
            let mut found = true;
            for k in 0..sub_len {
                if seq[i + k] != seq[start + k] {
                    found = false;
                    break;
                }
            }
            if found {
                return true;
            }
        }
    } else {
        // 对于长子串，使用标准库的窗口搜索
        if seq[0..search_end]
            .windows(sub_len)
            .any(|window| window == &seq[start..end])
        {
            return true;
        }
    }

    false
}

/// 小序列LZ复杂度计算（针对<=64元素的优化）
fn calculate_lz_complexity_simple(seq: &[u8]) -> usize {
    let n = seq.len();
    if n == 0 {
        return 0;
    }

    let mut complexity = 0;
    let mut i = 0;

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

        complexity += 1;
        i = j;
    }

    complexity
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
    fn test_lz_complexity_discrete() {
        // 测试离散序列
        let seq = vec![0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0];
        let result = lz_complexity(
            numpy::array![0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0].view(),
            None,
            true,
        )
        .unwrap();

        // 应该和Python版本结果相近
        assert!(result > 0.0);
    }

    #[test]
    fn test_discretization() {
        let seq = numpy::array![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0].view();
        let result = lz_complexity(seq, Some(vec![0.5]), true).unwrap();

        assert!(result > 0.0);
    }
}
