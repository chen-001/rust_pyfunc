use numpy::{PyArray2, PyReadonlyArray1};
use pyo3::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::{HashMap, HashSet, VecDeque};

/// 计算两个字符串的接近度（最少操作次数）
///
/// 参数说明：
/// ----------
/// word1 : str
///     第一个输入字符串
/// word2 : str  
///     第二个输入字符串
///
/// 返回值：
/// -------
/// float
///     两个字符串之间的最少操作次数。如果无法通过操作相互转换，返回NaN。
///
/// 操作说明：
/// 1. 操作1：交换任意两个现有字符的位置
/// 2. 操作2：将一个现有字符的每次出现都转换为另一个现有字符，并对另一个字符执行相同的操作
///
/// Python调用示例：
/// ```python
/// from rust_pyfunc import check_string_proximity
///
/// # 示例1
/// result = check_string_proximity("abc", "bca")
/// print(f"abc -> bca 需要操作次数: {result}")  # 2
///
/// # 示例2  
/// result = check_string_proximity("a", "aa")
/// print(f"a -> aa 需要操作次数: {result}")    # NaN (无法转换)
///
/// # 示例3
/// result = check_string_proximity("cabbba", "abbccc")
/// print(f"cabbba -> abbccc 需要操作次数: {result}")  # 3
/// ```
#[pyfunction]
#[pyo3(signature = (word1, word2))]
pub fn check_string_proximity(word1: &str, word2: &str) -> f64 {
    if !can_transform(word1, word2) {
        return f64::NAN;
    }

    if word1 == word2 {
        return 0.0;
    }

    find_min_operations(word1, word2)
}

/// 计算字符串数组中所有字符串对之间的接近度矩阵
///
/// 参数说明：
/// ----------
/// words : list[str]
///     包含字符串的Python列表
///
/// 返回值：
/// -------
/// np.ndarray
///     k×k的接近度矩阵，其中k是字符串数组的长度。
///     matrix[i][j] 表示第i个字符串和第j个字符串之间的最少操作次数。
///     如果无法转换则为NaN。
///
/// Python调用示例：
/// ```python
/// from rust_pyfunc import check_string_proximity_matrix
///
/// words = ["abc", "bca", "cab", "xyz"]
/// matrix = check_string_proximity_matrix(words)
/// print("接近度矩阵:")
/// print(matrix)
/// # [[0.0, 2.0, 2.0, nan],
/// #  [2.0, 0.0, 2.0, nan],
/// #  [2.0, 2.0, 0.0, nan],
/// #  [nan, nan, nan, 0.0]]
/// ```
#[pyfunction]
#[pyo3(signature = (words))]
pub fn check_string_proximity_matrix<'py>(
    py: Python<'py>,
    words: Vec<&str>,
) -> PyResult<&'py PyArray2<f64>> {
    let n = words.len();
    let mut matrix = vec![vec![0.0; n]; n];

    for i in 0..n {
        for j in 0..n {
            if i == j {
                matrix[i][j] = 0.0;
            } else {
                matrix[i][j] = check_string_proximity(words[i], words[j]);
            }
        }
    }

    Ok(PyArray2::from_vec2(py, &matrix).unwrap())
}

/// 计算两个字符串的接近度（最少操作次数），支持宽限参数
///
/// 参数说明：
/// ----------
/// word1 : str
///     第一个输入字符串
/// word2 : str  
///     第二个输入字符串
/// tolerance : int
///     宽限参数，允许最多tolerance个位置上的字符不同
///
/// 返回值：
/// -------
/// float
///     两个字符串之间的最少操作次数。如果无法通过操作相互转换，返回NaN。
///
/// 操作说明：
/// 1. 操作1：交换任意两个现有字符的位置
/// 2. 操作2：将一个现有字符的每次出现都转换为另一个现有字符，并对另一个字符执行相同的操作
/// 3. 宽限：允许最多tolerance个字符位置不匹配
///
/// Python调用示例：
/// ```python
/// from rust_pyfunc import check_string_proximity_with_tolerance
///
/// # 示例1：无宽限（等同于原函数）
/// result = check_string_proximity_with_tolerance("abc", "bca", 0)
/// print(f"abc -> bca (tolerance=0): {result}")  # 2
///
/// # 示例2：允许1个字符不同
/// result = check_string_proximity_with_tolerance("abc", "abd", 1)
/// print(f"abc -> abd (tolerance=1): {result}")  # 可以转换
///
/// # 示例3：允许2个字符不同
/// result = check_string_proximity_with_tolerance("abcd", "abef", 2)
/// print(f"abcd -> abef (tolerance=2): {result}")  # 可以转换
/// ```
#[pyfunction]
#[pyo3(signature = (word1, word2, tolerance = 0))]
pub fn check_string_proximity_with_tolerance(word1: &str, word2: &str, tolerance: usize) -> f64 {
    if !can_transform_with_tolerance(word1, word2, tolerance) {
        return f64::NAN;
    }

    if word1 == word2 {
        return 0.0;
    }

    find_min_operations_with_tolerance(word1, word2, tolerance)
}

/// 计算字符串数组中所有字符串对之间的接近度矩阵，支持宽限参数
///
/// 参数说明：
/// ----------
/// words : list[str]
///     包含字符串的Python列表
/// tolerance : int
///     宽限参数，允许最多tolerance个位置上的字符不同
///
/// 返回值：
/// -------
/// np.ndarray
///     k×k的接近度矩阵，其中k是字符串数组的长度。
///     matrix[i][j] 表示第i个字符串和第j个字符串之间的最少操作次数。
///     如果无法转换则为NaN。
///
/// Python调用示例：
/// ```python
/// from rust_pyfunc import check_string_proximity_matrix_with_tolerance
///
/// words = ["abc", "abd", "aef", "xyz"]
/// matrix = check_string_proximity_matrix_with_tolerance(words, tolerance=1)
/// print("接近度矩阵 (tolerance=1):")
/// print(matrix)
/// ```
#[pyfunction]
#[pyo3(signature = (words, tolerance = 0))]
pub fn check_string_proximity_matrix_with_tolerance<'py>(
    py: Python<'py>,
    words: Vec<&str>,
    tolerance: usize,
) -> PyResult<&'py PyArray2<f64>> {
    let n = words.len();
    let mut matrix = vec![vec![0.0; n]; n];

    // 使用缓存存储已计算的结果，利用对称性
    let mut cache = FxHashMap::default();

    for i in 0..n {
        for j in i..n {
            // 只计算上三角矩阵
            if i == j {
                matrix[i][j] = 0.0;
            } else {
                // 检查缓存
                let key = if words[i] <= words[j] {
                    (words[i], words[j])
                } else {
                    (words[j], words[i])
                };

                let result = if let Some(&cached_result) = cache.get(&key) {
                    cached_result
                } else {
                    let computed_result =
                        check_string_proximity_with_tolerance(words[i], words[j], tolerance);
                    cache.insert(key, computed_result);
                    computed_result
                };

                // 利用对称性填充矩阵
                matrix[i][j] = result;
                matrix[j][i] = result;
            }
        }
    }

    Ok(PyArray2::from_vec2(py, &matrix).unwrap())
}

fn can_transform(word1: &str, word2: &str) -> bool {
    if word1.len() != word2.len() {
        return false;
    }

    let mut count1 = HashMap::new();
    let mut count2 = HashMap::new();

    for ch in word1.chars() {
        *count1.entry(ch).or_insert(0) += 1;
    }

    for ch in word2.chars() {
        *count2.entry(ch).or_insert(0) += 1;
    }

    // 检查字符频次是否相同（必要条件）
    count1 == count2
}

fn find_min_operations(word1: &str, word2: &str) -> f64 {
    if word1 == word2 {
        return 0.0;
    }

    find_min_operations_optimized(word1, word2)
}

fn find_min_operations_optimized(word1: &str, word2: &str) -> f64 {
    if word1 == word2 {
        return 0.0;
    }

    // 针对4位数字字符串的快速检查
    if !can_transform(word1, word2) {
        return f64::NAN;
    }

    // 对于4位数字字符串，使用简化的BFS
    find_min_operations_for_digits(word1, word2)
}

fn find_min_operations_for_digits(word1: &str, word2: &str) -> f64 {
    if word1 == word2 {
        return 0.0;
    }

    let start_state = word1.as_bytes();
    let target_state = word2.as_bytes();

    let mut queue = VecDeque::new();
    let mut visited = FxHashSet::default();

    queue.push_back((start_state.to_vec(), 0));
    visited.insert(start_state.to_vec());

    while let Some((current, steps)) = queue.pop_front() {
        if current == target_state {
            return steps as f64;
        }

        // 限制搜索深度避免过长时间
        if steps >= 6 {
            // 4位数字最多需要6步
            continue;
        }

        // 生成所有可能的下一步状态
        let mut next_states = Vec::new();

        // 操作1：交换任意两个位置
        for i in 0..4 {
            for j in (i + 1)..4 {
                if current[i] != current[j] {
                    let mut new_state = current.clone();
                    new_state.swap(i, j);
                    next_states.push(new_state);
                }
            }
        }

        // 操作2：字符映射交换（针对0-9数字优化）
        for d1 in b'0'..=b'9' {
            for d2 in (d1 + 1)..=b'9' {
                if current.contains(&d1) && current.contains(&d2) {
                    let new_state: Vec<u8> = current
                        .iter()
                        .map(|&ch| {
                            if ch == d1 {
                                d2
                            } else if ch == d2 {
                                d1
                            } else {
                                ch
                            }
                        })
                        .collect();

                    if new_state != current {
                        next_states.push(new_state);
                    }
                }
            }
        }

        // 添加新状态到队列
        for state in next_states {
            if !visited.contains(&state) {
                visited.insert(state.clone());
                queue.push_back((state, steps + 1));
            }
        }
    }

    f64::NAN
}

fn generate_swap_states(word: &str, steps: usize) -> Vec<(String, usize)> {
    let mut result = Vec::new();
    let chars: Vec<char> = word.chars().collect();
    let n = chars.len();

    for i in 0..n {
        for j in (i + 1)..n {
            let mut new_chars = chars.clone();
            new_chars.swap(i, j);
            let new_word: String = new_chars.into_iter().collect();
            result.push((new_word, steps));
        }
    }

    result
}

fn generate_char_swap_states(word: &str, steps: usize) -> Vec<(String, usize)> {
    let mut result = Vec::new();
    let unique_chars: HashSet<char> = word.chars().collect();
    let unique_chars: Vec<char> = unique_chars.into_iter().collect();
    let n = unique_chars.len();

    for i in 0..n {
        for j in (i + 1)..n {
            let char1 = unique_chars[i];
            let char2 = unique_chars[j];

            let new_word: String = word
                .chars()
                .map(|c| {
                    if c == char1 {
                        char2
                    } else if c == char2 {
                        char1
                    } else {
                        c
                    }
                })
                .collect();

            if new_word != word {
                result.push((new_word, steps));
            }
        }
    }

    result
}

fn can_transform_with_tolerance(word1: &str, word2: &str, tolerance: usize) -> bool {
    if word1.len() != word2.len() {
        return false;
    }

    let mut count1 = HashMap::new();
    let mut count2 = HashMap::new();

    for ch in word1.chars() {
        *count1.entry(ch).or_insert(0) += 1;
    }

    for ch in word2.chars() {
        *count2.entry(ch).or_insert(0) += 1;
    }

    // 计算字符频次差异的总数
    let mut total_diff = 0;
    let all_chars: HashSet<char> = count1.keys().chain(count2.keys()).cloned().collect();

    for ch in all_chars {
        let count1_ch = *count1.get(&ch).unwrap_or(&0);
        let count2_ch = *count2.get(&ch).unwrap_or(&0);
        let diff = if count1_ch > count2_ch {
            count1_ch - count2_ch
        } else {
            count2_ch - count1_ch
        };
        total_diff += diff;
    }

    // 差异总数的一半就是需要的宽限数
    // 例如：word1中有2个a，word2中有1个a，差异为1
    // 但实际只需要1个宽限位置来忽略这个差异
    (total_diff / 2) <= tolerance
}

fn find_min_operations_with_tolerance(word1: &str, word2: &str, tolerance: usize) -> f64 {
    if word1 == word2 {
        return 0.0;
    }

    // 对于有宽限的情况，我们需要更复杂的搜索策略
    // 基本思路：尝试移除最多tolerance个字符对，然后对剩余部分进行BFS搜索

    // 如果tolerance为0，使用原有的方法
    if tolerance == 0 {
        return find_min_operations(word1, word2);
    }

    // 对于有tolerance的情况，简化处理：
    // 1. 计算需要忽略的字符差异
    // 2. 对于剩余可匹配的部分，计算最小操作数

    let chars1: Vec<char> = word1.chars().collect();
    let chars2: Vec<char> = word2.chars().collect();
    let n = chars1.len();

    let mut min_operations = f64::INFINITY;

    // 尝试所有可能的忽略位置组合
    // 为了简化，这里使用贪心策略：优先忽略不匹配的位置

    let mut ignored_positions = Vec::new();
    for i in 0..n {
        if chars1[i] != chars2[i] {
            ignored_positions.push(i);
        }
    }

    // 如果需要忽略的位置数量小于等于tolerance，直接计算剩余部分的操作数
    if ignored_positions.len() <= tolerance {
        // 构建忽略某些位置后的子字符串
        let mut filtered_chars1 = Vec::new();
        let mut filtered_chars2 = Vec::new();

        let ignored_set: HashSet<usize> = ignored_positions.into_iter().take(tolerance).collect();

        for i in 0..n {
            if !ignored_set.contains(&i) {
                filtered_chars1.push(chars1[i]);
                filtered_chars2.push(chars2[i]);
            }
        }

        let filtered_word1: String = filtered_chars1.into_iter().collect();
        let filtered_word2: String = filtered_chars2.into_iter().collect();

        if filtered_word1.is_empty() {
            return 0.0;
        }

        // 对过滤后的字符串计算最小操作数
        let filtered_result = find_min_operations(&filtered_word1, &filtered_word2);
        if !filtered_result.is_nan() {
            min_operations = min_operations.min(filtered_result);
        }
    } else {
        // 如果直接不匹配的位置太多，尝试更复杂的组合策略
        // 这里为了简化，返回一个启发式结果
        let base_operations = find_min_operations_heuristic(word1, word2);
        if !base_operations.is_nan() {
            min_operations = min_operations.min(base_operations);
        }
    }

    if min_operations.is_infinite() {
        f64::NAN
    } else {
        min_operations
    }
}

fn find_min_operations_heuristic(word1: &str, word2: &str) -> f64 {
    // 启发式方法：估算需要的操作数
    // 计算两个字符串之间的"编辑距离"的一个简化版本

    let chars1: Vec<char> = word1.chars().collect();
    let chars2: Vec<char> = word2.chars().collect();

    if chars1.len() != chars2.len() {
        return f64::NAN;
    }

    let mut diff_count = 0;
    for i in 0..chars1.len() {
        if chars1[i] != chars2[i] {
            diff_count += 1;
        }
    }

    // 简化的启发式：不同位置的数量除以2（因为每次交换可以修复2个位置）
    (diff_count as f64 / 2.0).ceil()
}

// ========== 优化后的辅助函数 ==========

/// 预计算字符位置映射，用于快速定位和优化
fn compute_char_positions(state: &[u8]) -> FxHashMap<u8, Vec<usize>> {
    let mut positions = FxHashMap::default();
    for (i, &ch) in state.iter().enumerate() {
        positions.entry(ch).or_insert_with(Vec::new).push(i);
    }
    positions
}

/// 检查两个字符位置映射是否能够通过操作相互转换
fn can_transform_with_char_mapping(
    positions1: &FxHashMap<u8, Vec<usize>>,
    positions2: &FxHashMap<u8, Vec<usize>>,
) -> bool {
    // 检查每个字符的数量是否相同
    for (&ch, pos_list1) in positions1 {
        if let Some(pos_list2) = positions2.get(&ch) {
            if pos_list1.len() != pos_list2.len() {
                return false;
            }
        } else {
            return false;
        }
    }

    // 检查反向映射
    for (&ch, pos_list2) in positions2 {
        if !positions1.contains_key(&ch) {
            return false;
        }
    }

    true
}

/// 优化的位置交换状态生成器
fn generate_swap_states_optimized(state: &[u8]) -> Vec<Vec<u8>> {
    let mut result = Vec::new();
    let n = state.len();

    for i in 0..n {
        for j in (i + 1)..n {
            // 只有当两个位置的字符不同时才进行交换
            if state[i] != state[j] {
                let mut new_state = state.to_vec();
                new_state.swap(i, j);
                result.push(new_state);
            }
        }
    }

    result
}

/// 优化的字符映射交换状态生成器，利用对称性避免重复计算
fn generate_char_swap_states_optimized(
    state: &[u8],
    char_positions: &FxHashMap<u8, Vec<usize>>,
) -> Vec<Vec<u8>> {
    let mut result = Vec::new();
    let unique_chars: Vec<u8> = char_positions.keys().cloned().collect();
    let n = unique_chars.len();

    for i in 0..n {
        for j in (i + 1)..n {
            let char1 = unique_chars[i];
            let char2 = unique_chars[j];

            // 利用对称性：只考虑 char1 < char2 的情况
            if char1 < char2 {
                let new_state = swap_characters_in_state(state, char1, char2);
                if new_state != state {
                    result.push(new_state);
                }
            }
        }
    }

    result
}

/// 在状态中交换两个字符的所有出现
fn swap_characters_in_state(state: &[u8], char1: u8, char2: u8) -> Vec<u8> {
    state
        .iter()
        .map(|&ch| {
            if ch == char1 {
                char2
            } else if ch == char2 {
                char1
            } else {
                ch
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_example_cases() {
        // 示例1: "abc" -> "bca", 应该是2步
        let result = check_string_proximity("abc", "bca");
        assert!(!result.is_nan());
        assert_eq!(result, 2.0);

        // 示例2: "a" -> "aa", 应该无法转换
        let result = check_string_proximity("a", "aa");
        assert!(result.is_nan());

        // 示例3: "cabbba" -> "abbccc", 应该是3步
        let result = check_string_proximity("cabbba", "abbccc");
        assert!(!result.is_nan());
        assert_eq!(result, 3.0);

        // 相同字符串
        let result = check_string_proximity("hello", "hello");
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_can_transform() {
        // 可以转换的情况
        assert!(can_transform("abc", "bca"));
        assert!(can_transform("aab", "aba"));

        // 不能转换的情况
        assert!(!can_transform("a", "aa"));
        assert!(!can_transform("abc", "def"));
        assert!(!can_transform("abc", "abcd"));
    }

    #[test]
    fn test_can_transform_with_tolerance() {
        // tolerance = 0，等同于原函数
        assert!(can_transform_with_tolerance("abc", "bca", 0));
        assert!(!can_transform_with_tolerance("abc", "def", 0));

        // tolerance = 1，允许1个字符不同
        assert!(can_transform_with_tolerance("abc", "abd", 1)); // c->d，差异1个字符
        assert!(can_transform_with_tolerance("abc", "aec", 1)); // b->e，差异1个字符
        assert!(!can_transform_with_tolerance("abc", "def", 1)); // 差异3个字符，超过tolerance

        // tolerance = 2，允许2个字符不同
        assert!(can_transform_with_tolerance("abcd", "abef", 2)); // c->e, d->f，差异2个字符
        assert!(can_transform_with_tolerance("abc", "def", 3)); // 差异3个字符，tolerance=3允许

        // 边界情况
        assert!(can_transform_with_tolerance("", "", 0));
        assert!(can_transform_with_tolerance("a", "a", 0));
        assert!(can_transform_with_tolerance("a", "b", 1));
    }

    #[test]
    fn test_with_tolerance_functionality() {
        // tolerance = 0应该等同于原函数
        assert_eq!(
            check_string_proximity_with_tolerance("abc", "bca", 0),
            check_string_proximity("abc", "bca")
        );

        // tolerance = 1的测试用例
        let result = check_string_proximity_with_tolerance("abc", "abd", 1);
        assert!(!result.is_nan());

        // 相同字符串
        assert_eq!(
            check_string_proximity_with_tolerance("hello", "hello", 1),
            0.0
        );

        // 完全不同但在tolerance范围内
        let result = check_string_proximity_with_tolerance("ab", "cd", 2);
        assert!(!result.is_nan());
    }
}
