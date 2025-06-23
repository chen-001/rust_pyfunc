"""文本处理函数类型声明"""
from typing import List, Tuple

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