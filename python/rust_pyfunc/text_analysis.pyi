"""文本处理函数类型声明"""
from typing import List, Tuple
import numpy as np

def vectorize_sentences(sentence1: str, sentence2: str) -> Tuple[List[int], List[int]]:
    """将两个句子转换为词频向量。
    生成的向量长度相同，等于两个句子中不同单词的总数。
    向量中的每个位置对应一个单词，值表示该单词在句子中出现的次数。

    参数说明：
    ----------
    sentence1 : str
        第一个输入句子
    sentence2 : str
        第二个输入句子

    返回值：
    -------
    tuple
        返回一个元组(vector1, vector2)，其中：
        - vector1: 第一个句子的词频向量
        - vector2: 第二个句子的词频向量
        两个向量长度相同，每个位置对应词表中的一个单词
    """
    ...

def jaccard_similarity(str1: str, str2: str) -> float:
    """计算两个句子之间的Jaccard相似度。
    Jaccard相似度是两个集合交集大小除以并集大小，用于衡量两个句子的相似程度。
    这里将每个句子视为单词集合，忽略单词出现的顺序和频率。

    参数说明：
    ----------
    str1 : str
        第一个输入句子
    str2 : str
        第二个输入句子

    返回值：
    -------
    float
        Jaccard相似度值，范围为[0, 1]，1表示完全相似，0表示完全不相似
    """
    ...

def min_word_edit_distance(str1: str, str2: str) -> int:
    """计算两个字符串之间的最小编辑距离（Levenshtein距离）。
    编辑距离是指通过插入、删除或替换字符将一个字符串转换为另一个字符串所需的最小操作次数。

    参数说明：
    ----------
    str1 : str
        第一个输入字符串
    str2 : str
        第二个输入字符串

    返回值：
    -------
    int
        最小编辑距离，非负整数
    """
    ...

def vectorize_sentences_list(sentences: List[str]) -> List[List[int]]:
    """将多个句子转换为词频向量矩阵。
    
    参数说明：
    ----------
    sentences : List[str]
        句子列表
        
    返回值：
    -------
    List[List[int]]
        词频向量矩阵，每行对应一个句子的词频向量
    """
    ...

def check_string_proximity(word1: str, word2: str) -> float:
    """计算两个字符串的接近度（最少操作次数）。
    
    通过两种操作判断字符串是否接近：
    1. 操作1：交换任意两个现有字符的位置
    2. 操作2：将一个现有字符的每次出现都转换为另一个现有字符，并对另一个字符执行相同的操作
    
    参数说明：
    ----------
    word1 : str
        第一个输入字符串
    word2 : str  
        第二个输入字符串
        
    返回值：
    -------
    float
        两个字符串之间的最少操作次数。如果无法通过操作相互转换，返回NaN。
    """
    ...

def check_string_proximity_matrix(words: List[str]) -> np.ndarray:
    """计算字符串数组中所有字符串对之间的接近度矩阵。
    
    参数说明：
    ----------
    words : List[str]
        包含字符串的Python列表
        
    返回值：
    -------
    np.ndarray
        k×k的接近度矩阵，其中k是字符串数组的长度。
        matrix[i][j] 表示第i个字符串和第j个字符串之间的最少操作次数。
        如果无法转换则为NaN。
    """
    ...

def check_string_proximity_with_tolerance(word1: str, word2: str, tolerance: int = 0) -> float:
    """计算两个字符串的接近度（最少操作次数），支持宽限参数。
    
    通过两种操作判断字符串是否接近：
    1. 操作1：交换任意两个现有字符的位置
    2. 操作2：将一个现有字符的每次出现都转换为另一个现有字符，并对另一个字符执行相同的操作
    3. 宽限：允许最多tolerance个字符位置不匹配
    
    参数说明：
    ----------
    word1 : str
        第一个输入字符串
    word2 : str  
        第二个输入字符串
    tolerance : int, optional
        宽限参数，允许最多tolerance个位置上的字符不同，默认为0
        
    返回值：
    -------
    float
        两个字符串之间的最少操作次数。如果无法通过操作相互转换，返回NaN。
    """
    ...

def check_string_proximity_matrix_with_tolerance(words: List[str], tolerance: int = 0) -> np.ndarray:
    """计算字符串数组中所有字符串对之间的接近度矩阵，支持宽限参数。
    
    参数说明：
    ----------
    words : List[str]
        包含字符串的Python列表
    tolerance : int, optional
        宽限参数，允许最多tolerance个位置上的字符不同，默认为0
        
    返回值：
    -------
    np.ndarray
        k×k的接近度矩阵，其中k是字符串数组的长度。
        matrix[i][j] 表示第i个字符串和第j个字符串之间的最少操作次数。
        如果无法转换则为NaN。
    """
    ...