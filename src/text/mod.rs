use pyo3::prelude::*;
use std::collections::{HashMap, HashSet};

/// 将两个句子转换为词频向量。
/// 生成的向量长度相同，等于两个句子中不同单词的总数。
/// 向量中的每个位置对应一个单词，值表示该单词在句子中出现的次数。
///
/// 参数说明：
/// ----------
/// sentence1 : str
///     第一个输入句子
/// sentence2 : str
///     第二个输入句子
///
/// 返回值：
/// -------
/// tuple
///     返回一个元组(vector1, vector2)，其中：
///     - vector1: 第一个句子的词频向量
///     - vector2: 第二个句子的词频向量
///     两个向量长度相同，每个位置对应词表中的一个单词
///
/// Python调用示例：
/// ```python
/// from rust_pyfunc import vectorize_sentences
///
/// # 准备两个测试句子
/// s1 = "The quick brown fox"
/// s2 = "The lazy brown dog"
///
/// # 转换为词频向量
/// v1, v2 = vectorize_sentences(s1, s2)
/// print(f"句子1的词频向量: {v1}")  # 例如：[1, 1, 1, 1, 0]
/// print(f"句子2的词频向量: {v2}")  # 例如：[1, 0, 1, 0, 1]
///
/// # 解释结果：
/// # 假设合并的词表为 ["brown", "fox", "quick", "the", "lazy"]
/// # v1 = [1, 1, 1, 1, 0] 表示 brown, fox, quick, the 各出现一次，lazy未出现
/// # v2 = [1, 0, 0, 1, 1] 表示 brown, the, lazy 各出现一次，fox和quick未出现
/// ```
#[pyfunction]
#[pyo3(signature = (sentence1, sentence2))]
pub fn vectorize_sentences(sentence1: &str, sentence2: &str) -> (Vec<usize>, Vec<usize>) {
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

/// 将多个句子转换为词频向量列表。
/// 生成的所有向量长度相同，等于所有句子中不同单词的总数。
/// 每个向量中的每个位置对应一个单词，值表示该单词在对应句子中出现的次数。
///
/// 参数说明：
/// ----------
/// sentences : list[str]
///     输入句子列表，每个元素是一个字符串
///
/// 返回值：
/// -------
/// list[list[int]]
///     返回词频向量列表，其中：
///     - 每个向量对应一个输入句子
///     - 所有向量长度相同，等于所有句子中不同单词的总数
///     - 向量中的每个值表示对应单词在该句子中的出现次数
///
/// Python调用示例：
/// ```python
/// from rust_pyfunc import vectorize_sentences_list
///
/// # 准备测试句子列表
/// sentences = [
///     "The quick brown fox",
///     "The lazy brown dog",
///     "A quick brown fox jumps"
/// ]
///
/// # 转换为词频向量列表
/// vectors = vectorize_sentences_list(sentences)
///
/// # 打印每个句子的词频向量
/// for i, vec in enumerate(vectors):
///     print(f"句子{i+1}的词频向量: {vec}")
///
/// # 示例输出解释：
/// # 假设合并后的词表为 ["a", "brown", "dog", "fox", "jumps", "lazy", "quick", "the"]
/// # 第一个句子: [0, 1, 0, 1, 0, 0, 1, 1]  # "The quick brown fox"
/// # 第二个句子: [0, 1, 1, 0, 0, 1, 0, 1]  # "The lazy brown dog"
/// # 第三个句子: [1, 1, 0, 1, 1, 0, 1, 0]  # "A quick brown fox jumps"
/// ```
#[pyfunction]
#[pyo3(signature = (sentences))]
pub fn vectorize_sentences_list(sentences: Vec<&str>) -> Vec<Vec<usize>> {
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


/// 计算两个句子之间的Jaccard相似度。
/// Jaccard相似度是两个集合交集大小除以并集大小，用于衡量两个句子的相似程度。
/// 这里将每个句子视为单词集合，忽略单词出现的顺序和频率。
///
/// 参数说明：
/// ----------
/// sentence1 : str
///     第一个输入句子
/// sentence2 : str
///     第二个输入句子
///
/// 返回值：
/// -------
/// float
///     返回两个句子的Jaccard相似度，范围在[0, 1]之间：
///     - 1表示两个句子完全相同（包含相同的单词集合）
///     - 0表示两个句子完全不同（没有共同单词）
///     - 中间值表示部分相似
///
/// Python调用示例：
/// ```python
/// from rust_pyfunc import jaccard_similarity
///
/// # 测试完全相同的句子
/// s1 = "The quick brown fox"
/// s2 = "The quick brown fox"
/// sim1 = jaccard_similarity(s1, s2)
/// print(f"完全相同的句子相似度: {sim1}")  # 输出: 1.0
///
/// # 测试部分相同的句子
/// s3 = "The lazy brown dog"
/// sim2 = jaccard_similarity(s1, s3)
/// print(f"部分相同的句子相似度: {sim2}")  # 输出: 0.4 (2个共同词 / 5个不同词)
///
/// # 测试完全不同的句子
/// s4 = "Hello world example"
/// sim3 = jaccard_similarity(s1, s4)
/// print(f"完全不同的句子相似度: {sim3}")  # 输出: 0.0
///
/// # 注意：结果会忽略大小写和标点符号
/// s5 = "THE QUICK BROWN FOX!"
/// sim4 = jaccard_similarity(s1, s5)
/// print(f"大小写不同的相似度: {sim4}")  # 输出: 1.0
/// ```
#[pyfunction]
#[pyo3(signature = (str1, str2))]
pub fn jaccard_similarity(str1: &str, str2: &str) -> f64 {
    // 预处理文本
    let str1 = preprocess_text(str1);
    let str2 = preprocess_text(str2);

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

/// 预处理文本，转为小写并删除标点符号和非字母数字字符。
fn preprocess_text(text: &str) -> String {
    // 转为小写
    let text = text.to_lowercase();
    // 删除标点符号和非字母数字字符
    let text: String = text.chars()
        .filter(|c| c.is_alphanumeric() || c.is_whitespace())
        .collect();
    // 去除多余空格（可选，已通过split_whitespace处理）
    text
}

/// Given a sentence, return a dictionary where the keys are the words in the sentence
/// and the values are their frequencies.
///
/// For example, given the sentence `"The quick brown fox"`, the function will return
/// `{"The": 1, "quick": 1, "brown": 1, "fox": 1}`, since each word appears once.
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

/// 计算将一个句子转换为另一个句子所需的最少单词操作次数（添加/删除）。
///
/// # 参数
/// * `str1` - 源句子
/// * `str2` - 目标句子
///
/// # 示例
/// ```python
/// from rust_pyfunc import min_word_edit_distance
///
/// # 示例1：添加一个单词
/// da = "We expect demand to increase"
/// db = "We expect worldwide demand to increase"
/// print(min_word_edit_distance(da, db))  # 输出: 1 (添加 "worldwide")
///
/// # 示例2：多次修改
/// dc = "We expect weakness in sales"
/// print(min_word_edit_distance(da, dc))  # 输出: 6 (删除3个单词，添加3个单词)
/// ```
#[pyfunction]
#[pyo3(signature = (str1, str2))]
pub fn min_word_edit_distance(str1: &str, str2: &str) -> usize {
    // 预处理文本
    let str1 = preprocess_text(str1);
    let str2 = preprocess_text(str2);

    // 将句子分割成单词数组
    let words1: Vec<&str> = str1.split_whitespace().collect();
    let words2: Vec<&str> = str2.split_whitespace().collect();

    // 创建两个HashSet来存储单词
    let set1: HashSet<&str> = words1.iter().copied().collect();
    let set2: HashSet<&str> = words2.iter().copied().collect();

    // 计算需要删除的单词数（在str1中但不在str2中的单词）
    let deletions = set1.difference(&set2).count();
    
    // 计算需要添加的单词数（在str2中但不在str1中的单词）
    let additions = set2.difference(&set1).count();

    // 总的编辑距离是删除和添加操作的总和
    deletions + additions
}