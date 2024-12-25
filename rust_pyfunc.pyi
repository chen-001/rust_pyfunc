from typing import List, Optional, Tuple, Union
import numpy as np
from numpy.typing import NDArray

def trend(arr: Union[NDArray[np.float64], List[Union[float, int]]]) -> float:
    """计算输入数组与自然数序列(1, 2, ..., n)之间的皮尔逊相关系数。
    这个函数可以用来判断一个序列的趋势性，如果返回值接近1表示强上升趋势，接近-1表示强下降趋势。

    参数说明：
    ----------
    arr : 输入数组
        可以是以下类型之一：
        - numpy.ndarray (float64或int64类型)
        - Python列表 (float或int类型)

    返回值：
    -------
    float
        输入数组与自然数序列的皮尔逊相关系数。
        如果输入数组为空或方差为零，则返回0.0。
    """
    ...

def trend_fast(arr: NDArray[np.float64]) -> float:
    """这是trend函数的高性能版本，专门用于处理numpy.ndarray类型的float64数组。
    使用了显式的SIMD指令和缓存优化处理，比普通版本更快。

    参数说明：
    ----------
    arr : numpy.ndarray
        输入数组，必须是float64类型

    返回值：
    -------
    float
        输入数组与自然数序列的皮尔逊相关系数
    """
    ...

def identify_segments(arr: NDArray[np.float64]) -> NDArray[np.int32]:
    """识别数组中的连续相等值段，并为每个段分配唯一标识符。
    每个连续相等的值构成一个段，第一个段标识符为1，第二个为2，以此类推。

    参数说明：
    ----------
    arr : numpy.ndarray
        输入数组，类型为float64

    返回值：
    -------
    numpy.ndarray
        与输入数组等长的整数数组，每个元素表示该位置所属段的标识符
    """
    ...

def find_max_range_product(arr: NDArray[np.float64]) -> Tuple[int, int, float]:
    """在数组中找到一对索引(x, y)，使得min(arr[x], arr[y]) * |x-y|的值最大。
    这个函数可以用来找到数组中距离最远的两个元素，同时考虑它们的最小值。

    参数说明：
    ----------
    arr : numpy.ndarray
        输入数组，类型为float64

    返回值：
    -------
    tuple
        返回一个元组(x, y, max_product)，其中x和y是使得乘积最大的索引对，max_product是最大乘积
    """
    ...

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
        返回两个句子的Jaccard相似度，范围在[0, 1]之间：
        - 1表示两个句子完全相同（包含相同的单词集合）
        - 0表示两个句子完全不同（没有共同单词）
        - 中间值表示部分相似
    """
    ...

def min_word_edit_distance(str1: str, str2: str) -> int:
    """计算将一个句子转换为另一个句子所需的最少单词操作次数（添加/删除）。

    参数说明：
    ----------
    str1 : str
        源句子
    str2 : str
        目标句子

    返回值：
    -------
    int
        最少需要的单词操作次数
    """
    ...

def dtw_distance(s1: List[float], s2: List[float], radius: Optional[int] = None) -> float:
    """计算两个序列之间的动态时间规整(DTW)距离。
    DTW是一种衡量两个时间序列相似度的算法，可以处理不等长的序列。
    它通过寻找两个序列之间的最佳对齐方式来计算距离。

    参数说明：
    ----------
    s1 : array_like
        第一个输入序列
    s2 : array_like
        第二个输入序列
    radius : int, optional
        Sakoe-Chiba半径，用于限制规整路径，可以提高计算效率。
        如果不指定，则不使用路径限制。

    返回值：
    -------
    float
        两个序列之间的DTW距离，值越小表示序列越相似
    """
    ...

def transfer_entropy(x_: List[float], y_: List[float], k: int, c: int) -> float:
    """计算从序列x到序列y的转移熵（Transfer Entropy）。
    转移熵衡量了一个时间序列对另一个时间序列的影响程度，是一种非线性的因果关系度量。
    具体来说，它测量了在已知x的过去k个状态的情况下，对y的当前状态预测能力的提升程度。

    参数说明：
    ----------
    x_ : array_like
        源序列，用于预测目标序列
    y_ : array_like
        目标序列，我们要预测的序列
    k : int
        历史长度，考虑过去k个时间步的状态
    c : int
        离散化的类别数，将连续值离散化为c个等级

    返回值：
    -------
    float
        从x到y的转移熵值。值越大表示x对y的影响越大。
    """
    ...

def ols(x: NDArray[np.float64], y: NDArray[np.float64], calculate_r2: bool = True) -> NDArray[np.float64]:
    """普通最小二乘(OLS)回归。
    用于拟合线性回归模型 y = Xβ + ε，其中β是要估计的回归系数。

    参数说明：
    ----------
    x : numpy.ndarray
        设计矩阵，形状为(n_samples, n_features)
    y : numpy.ndarray
        响应变量，形状为(n_samples,)
    calculate_r2 : bool, optional
        是否计算R²值，默认为True

    返回值：
    -------
    numpy.ndarray
        回归系数β
    """
    ...

def ols_predict(x: NDArray[np.float64], y: NDArray[np.float64], x_pred: NDArray[np.float64]) -> NDArray[np.float64]:
    """使用已有数据和响应变量，对新的数据点进行OLS线性回归预测。

    参数说明：
    ----------
    x : numpy.ndarray
        原始设计矩阵，形状为(n_samples, n_features)
    y : numpy.ndarray
        原始响应变量，形状为(n_samples,)
    x_pred : numpy.ndarray
        需要预测的新数据点，形状为(m_samples, n_features)

    返回值：
    -------
    numpy.ndarray
        预测值，形状为(m_samples,)
    """
    ...

def max_range_loop(s: List[float], allow_equal: bool = False) -> List[int]:
    """计算序列中每个位置结尾的最长连续子序列长度，其中子序列的最大值在该位置。

    参数说明：
    ----------
    s : array_like
        输入序列，一个数值列表
    allow_equal : bool, 默认为False
        是否允许相等。如果为True，则当前位置的值大于前面的值时计入长度；
        如果为False，则当前位置的值大于等于前面的值时计入长度。

    返回值：
    -------
    list
        与输入序列等长的整数列表，每个元素表示以该位置结尾且最大值在该位置的最长连续子序列长度
    """
    ...

def min_range_loop(s: List[float], allow_equal: bool = False) -> List[int]:
    """计算序列中每个位置结尾的最长连续子序列长度，其中子序列的最小值在该位置。

    参数说明：
    ----------
    s : array_like
        输入序列，一个数值列表
    allow_equal : bool, 默认为False
        是否允许相等。如果为True，则当前位置的值小于前面的值时计入长度；
        如果为False，则当前位置的值小于等于前面的值时计入长度。

    返回值：
    -------
    list
        与输入序列等长的整数列表，每个元素表示以该位置结尾且最小值在该位置的最长连续子序列长度
    """
    ...

def find_local_peaks_within_window(times: NDArray[np.float64], prices: NDArray[np.float64], window: float) -> NDArray[np.bool_]:
    """
    查找时间序列中价格在指定时间窗口内为局部最大值的点。

    参数说明：
    ----------
    times : array_like
        时间戳数组（单位：秒）
    prices : array_like
        价格数组
    window : float
        时间窗口大小（单位：秒）

    返回值：
    -------
    numpy.ndarray
        布尔数组，True表示该点的价格大于指定时间窗口内的所有价格
    """
    ...

def rolling_window_stat(
    times: np.ndarray,
    values: np.ndarray,
    window: float,
    stat_type: str,
    include_current: bool = True,
) -> np.ndarray:
    """计算滚动窗口统计量

    参数：
        times (np.ndarray): 时间序列
        values (np.ndarray): 值序列
        window (float): 窗口大小（纳秒）
        stat_type (str): 统计量类型，可选值：
            - "mean": 均值
            - "sum": 求和
            - "max": 最大值
            - "min": 最小值
            - "last": 最后一个值
            - "std": 标准差
            - "median": 中位数
            - "count": 计数
            - "rank": 排名
            - "skew": 偏度
            - "trend_time": 与时间序列的相关系数
            - "trend_oneton": 与1到n序列的相关系数（忽略时间间隔）
        include_current (bool): 是否包含当前时间点的值

    返回：
        np.ndarray: 每个时间点的统计量
    """
    pass
