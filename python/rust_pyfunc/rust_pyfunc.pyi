from typing import List, Optional, Tuple, Union, Dict
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
    """计算时间序列在指定时间窗口内向后滚动的统计量。
    对于每个时间点，计算该点之后指定时间窗口内所有数据的指定统计量。

    参数说明：
    ----------
    times : np.ndarray
        时间戳数组（单位：秒）
    values : np.ndarray
        数值数组
    window : float
        时间窗口大小（单位：秒）
    stat_type : str
        统计量类型，可选值：
        - "mean": 均值
        - "sum": 总和
        - "max": 最大值
        - "min": 最小值
        - "last": 时间窗口内最后一个值
        - "std": 标准差
        - "median": 中位数
        - "count": 数据点数量
        - "rank": 分位数（0到1之间）
        - "skew": 偏度
        - "trend_time": 与时间序列的相关系数
        - "trend_oneton": 与1到n序列的相关系数（时间间隔）
    include_current : bool, default=True
        是否包含当前时间点的值。如果为False，则计算时会排除当前时间点的值。

    返回值：
    -------
    np.ndarray
        与输入时间序列等长的数组，包含每个时间点对应的统计量。
        对于无效的计算结果（如窗口内数据点不足），返回NaN。

    示例：
    -------
    >>> import numpy as np
    >>> from rust_pyfunc import rolling_window_stat
    >>> times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    >>> values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> window = 2.0
    >>> rolling_window_stat(times, values, window, "mean")
    array([2.0, 2.5, 3.5, 4.5, 5.0])

    注意：
    -----
    1. 时间窗口是向后滚动的，即对于每个时间点t，计算[t, t+window]范围内的统计量
    2. 当include_current=False时，计算范围为(t, t+window]
    3. 对于不同的统计类型，可能需要不同数量的有效数据点才能计算结果
    4. 所有时间单位都是秒
    """
    pass

class PriceTree:
    """价格树结构，用于分析价格序列的层次关系和分布特征。
    
    这是一个二叉树结构，每个节点代表一个价格水平，包含该价格的成交量和时间信息。
    树的构建基于价格的大小关系，支持快速的价格查找和区间统计。
    """
    
    def __init__(self) -> None:
        """初始化一个空的价格树。"""
        ...
    
    def build_tree(
        self,
        times: NDArray[np.int64],
        prices: NDArray[np.float64],
        volumes: NDArray[np.float64]
    ) -> None:
        """根据时间序列、价格序列和成交量序列构建价格树。

        参数说明：
        ----------
        times : numpy.ndarray
            时间戳序列，Unix时间戳格式
        prices : numpy.ndarray
            价格序列
        volumes : numpy.ndarray
            成交量序列
        """
        ...

    def get_tree_structure(self) -> str:
        """获取树的结构字符串表示。

        返回值：
        -------
        str
            树结构的字符串表示
        """
        ...

    def get_tree_statistics(self) -> List[Tuple[str, str]]:
        """获取树的统计特征。

        返回值：
        -------
        List[Tuple[str, str]]
            包含多个统计指标的列表，每个元素为(指标名称, 指标值)对
        """
        ...

    @property
    def min_depth(self) -> int:
        """获取树的最小深度"""
        ...

    @property
    def max_depth(self) -> int:
        """获取树的最大深度"""
        ...

    @property
    def avg_depth(self) -> float:
        """获取树的平均深度"""
        ...

    @property
    def total_nodes(self) -> int:
        """获取树的总节点数"""
        ...

    @property
    def leaf_nodes(self) -> int:
        """获取叶子节点数"""
        ...

    @property
    def internal_nodes(self) -> int:
        """获取内部节点数"""
        ...

    @property
    def leaf_internal_ratio(self) -> float:
        """获取叶子节点与内部节点的比率"""
        ...

    @property
    def degree_one_nodes(self) -> int:
        """获取度为1的节点数"""
        ...

    @property
    def degree_two_nodes(self) -> int:
        """获取度为2的节点数"""
        ...

    @property
    def degree_ratio(self) -> float:
        """获取度为1和度为2节点的比率"""
        ...

    @property
    def avg_balance_factor(self) -> float:
        """获取平均平衡因子"""
        ...

    @property
    def max_balance_factor(self) -> int:
        """获取最大平衡因子"""
        ...

    @property
    def skewness(self) -> float:
        """获取树的倾斜度"""
        ...

    @property
    def avg_path_length(self) -> float:
        """获取平均路径长度"""
        ...

    @property
    def max_path_length(self) -> int:
        """获取最大路径长度"""
        ...

    @property
    def avg_subtree_nodes(self) -> float:
        """获取平均子树节点数"""
        ...

    @property
    def max_subtree_nodes(self) -> int:
        """获取最大子树节点数"""
        ...

    @property
    def min_width(self) -> int:
        """获取树的最小宽度"""
        ...

    @property
    def max_width(self) -> int:
        """获取树的最大宽度"""
        ...

    @property
    def avg_width(self) -> float:
        """获取树的平均宽度"""
        ...

    @property
    def completeness(self) -> float:
        """获取树的完整度"""
        ...

    @property
    def density(self) -> float:
        """获取树的密度"""
        ...

    @property
    def critical_nodes(self) -> int:
        """获取关键节点数"""
        ...

    @property
    def asl(self) -> float:
        """获取平均查找长度(ASL)"""
        ...

    @property
    def wpl(self) -> float:
        """获取加权路径长度(WPL)"""
        ...

    @property
    def diameter(self) -> int:
        """获取树的直径"""
        ...

    @property
    def total_volume(self) -> float:
        """获取总成交量"""
        ...

    @property
    def avg_volume_per_node(self) -> float:
        """获取每个节点的平均成交量"""
        ...

    @property
    def price_range(self) -> Tuple[float, float]:
        """获取价格范围"""
        ...

    @property
    def time_range(self) -> Tuple[int, int]:
        """获取时间范围"""
        ...

    def get_all_features(self) -> Dict[str, Dict[str, Union[float, int]]]:
        """获取所有树的特征。

        返回值：
        -------
        Dict[str, Dict[str, Union[float, int]]]
            包含所有特征的嵌套字典，按类别组织
        """
        ...

def compute_max_eigenvalue(matrix: NDArray[np.float64]) -> Tuple[float, NDArray[np.float64]]:
    """计算二维方阵的最大特征值和对应的特征向量。
    使用幂迭代法计算，不使用并行计算。

    参数说明：
    ----------
    matrix : numpy.ndarray
        输入二维方阵，类型为float64

    返回值：
    -------
    tuple
        返回一个元组(eigenvalue, eigenvector)，
        eigenvalue是最大特征值（float64），
        eigenvector是对应的特征向量（numpy.ndarray）

    示例：
    -------
    >>> import numpy as np
    >>> from rust_pyfunc import compute_max_eigenvalue
    >>> 
    >>> # 创建测试矩阵
    >>> matrix = np.array([[4.0, -1.0], 
    ...                    [-1.0, 3.0]], dtype=np.float64)
    >>> eigenvalue, eigenvector = compute_max_eigenvalue(matrix)
    >>> print(f"最大特征值: {eigenvalue}")
    >>> print(f"对应的特征向量: {eigenvector}")
    """
    ...

def find_follow_volume_sum_same_price(times: NDArray[np.float64], prices: NDArray[np.float64], volumes: NDArray[np.float64], time_window: float = 0.1) -> NDArray[np.float64]:
    """计算每一行在其后0.1秒内具有相同price和volume的行的volume总和。

    参数说明：
    ----------
    times : numpy.ndarray
        时间戳数组（单位：秒）
    prices : numpy.ndarray
        价格数组
    volumes : numpy.ndarray
        成交量数组
    time_window : float, optional
        时间窗口大小（单位：秒），默认为0.1

    返回值：
    -------
    numpy.ndarray
        每一行在其后time_window秒内（包括当前行）具有相同price和volume的行的volume总和

    示例：
    -------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from rust_pyfunc import find_follow_volume_sum
    >>> 
    >>> # 创建示例DataFrame
    >>> df = pd.DataFrame({
    ...     'exchtime': [1.0, 1.05, 1.08, 1.15, 1.2],
    ...     'price': [10.0, 10.0, 10.0, 11.0, 10.0],
    ...     'volume': [100, 100, 100, 200, 100]
    ... })
    >>> 
    >>> # 计算follow列
    >>> df['follow'] = find_follow_volume_sum(
    ...     df['exchtime'].values,
    ...     df['price'].values,
    ...     df['volume'].values
    ... )
    >>> print(df)
       exchtime  price  volume  follow
    0     1.00   10.0    100    300.0  # 1.0秒后0.1秒内有两个相同的交易
    1     1.05   10.0    100    200.0  # 1.05秒后0.1秒内有一个相同的交易
    2     1.08   10.0    100    100.0  # 1.08秒后0.1秒内没有相同的交易
    3     1.15   11.0    200    200.0  # 不同价格的交易
    4     1.20   10.0    100    100.0  # 最后一个交易
    """
    ...

def find_follow_volume_sum_same_price_and_flag(times: NDArray[np.float64], prices: NDArray[np.float64], volumes: NDArray[np.float64], flags: NDArray[np.int32], time_window: float = 0.1) -> NDArray[np.float64]:
    """计算每一行在其后0.1秒内具有相同flag、price和volume的行的volume总和。

    参数说明：
    ----------
    times : array_like
        时间戳数组（单位：秒）
    prices : array_like
        价格数组
    volumes : array_like
        成交量数组
    flags : array_like
        主买卖标志数组
    time_window : float, optional
        时间窗口大小（单位：秒），默认为0.1

    返回值：
    -------
    numpy.ndarray
        每一行在其后time_window秒内具有相同price和volume的行的volume总和

    示例：
    -------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from rust_pyfunc import find_follow_volume_sum
    >>> 
    >>> # 创建示例DataFrame
    >>> df = pd.DataFrame({
    ...     'exchtime': [1.0, 1.05, 1.08, 1.15, 1.2],
    ...     'price': [10.0, 10.0, 10.0, 11.0, 10.0],
    ...     'volume': [100, 100, 100, 200, 100],
    ...     'flag': [66, 66, 66, 83, 66]
    ... })
    >>> 
    >>> # 计算follow列
    >>> df['follow'] = find_follow_volume_sum_same_price_and_flag(
    ...     df['exchtime'].values,
    ...     df['price'].values,
    ...     df['volume'].values,
    ...     df['flag'].values
    ... )
    """
    ...

def mark_follow_groups(times: NDArray[np.float64], prices: NDArray[np.float64], volumes: NDArray[np.float64], time_window: float = 0.1) -> NDArray[np.int32]:
    """标记每一行在其后0.1秒内具有相同price和volume的行组。
    对于同一个时间窗口内的相同交易组，标记相同的组号。
    组号从1开始递增，每遇到一个新的交易组就分配一个新的组号。

    参数说明：
    ----------
    times : numpy.ndarray
        时间戳数组（单位：秒）
    prices : numpy.ndarray
        价格数组
    volumes : numpy.ndarray
        成交量数组
    time_window : float, optional
        时间窗口大小（单位：秒），默认为0.1

    返回值：
    -------
    numpy.ndarray
        整数数组，表示每行所属的组号。0表示不属于任何组。

    示例：
    -------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from rust_pyfunc import mark_follow_groups
    >>> 
    >>> # 创建示例DataFrame
    >>> df = pd.DataFrame({
    ...     'exchtime': [1.0, 1.05, 1.08, 1.15, 1.2],
    ...     'price': [10.0, 10.0, 10.0, 11.0, 10.0],
    ...     'volume': [100, 100, 100, 200, 100]
    ... })
    >>> 
    >>> # 标记协同交易组
    >>> df['group'] = mark_follow_groups(
    ...     df['exchtime'].values,
    ...     df['price'].values,
    ...     df['volume'].values
    ... )
    >>> print(df)
    #    exchtime  price  volume  group
    # 0     1.00   10.0    100      1  # 第一组的起始点
    # 1     1.05   10.0    100      1  # 属于第一组
    # 2     1.08   10.0    100      1  # 属于第一组
    # 3     1.15   11.0    200      2  # 第二组的起始点
    # 4     1.20   10.0    100      3  # 第三组的起始点
    """
    ...

def mark_follow_groups_with_flag(times: NDArray[np.float64], prices: NDArray[np.float64], volumes: NDArray[np.float64], flags: NDArray[np.int64], time_window: float = 0.1) -> NDArray[np.int32]:
    """标记每一行在其后time_window秒内具有相同flag、price和volume的行组。
    对于同一个时间窗口内的相同交易组，标记相同的组号。
    组号从1开始递增，每遇到一个新的交易组就分配一个新的组号。

    参数说明：
    ----------
    times : numpy.ndarray
        时间戳数组（单位：秒）
    prices : numpy.ndarray
        价格数组
    volumes : numpy.ndarray
        成交量数组
    flags : numpy.ndarray
        主买卖标志数组
    time_window : float, optional
        时间窗口大小（单位：秒），默认为0.1

    返回值：
    -------
    numpy.ndarray
        整数数组，表示每行所属的组号。0表示不属于任何组。

    示例：
    -------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from rust_pyfunc import mark_follow_groups_with_flag
    >>> 
    >>> # 创建示例DataFrame
    >>> df = pd.DataFrame({
    ...     'exchtime': [1.0, 1.05, 1.08, 1.15, 1.2],
    ...     'price': [10.0, 10.0, 10.0, 11.0, 10.0],
    ...     'volume': [100, 100, 100, 200, 100],
    ...     'flag': [66, 66, 66, 83, 66]
    ... })
    >>> 
    >>> # 标记协同交易组
    >>> df['group'] = mark_follow_groups_with_flag(
    ...     df['exchtime'].values,
    ...     df['price'].values,
    ...     df['volume'].values,
    ...     df['flag'].values
    ... )
    >>> print(df)
    #    exchtime  price  volume  flag  group
    # 0     1.00   10.0    100    66      1  # 第一组的起始点
    # 1     1.05   10.0    100    66      1  # 属于第一组
    # 2     1.08   10.0    100    66      1  # 属于第一组
    # 3     1.15   11.0    200    83      2  # 第二组的起始点
    # 4     1.20   10.0    100    66      3  # 第三组的起始点
    """
    ...

def find_half_energy_time(times: NDArray[np.float64], prices: NDArray[np.float64], time_window: float = 5.0) -> NDArray[np.float64]:
    """计算每一行在其后指定时间窗口内的价格变动能量，并找出首次达到最终能量一半时所需的时间。

    参数说明：
    ----------
    times : numpy.ndarray
        时间戳数组（单位：秒）
    prices : numpy.ndarray
        价格数组
    time_window : float, optional
        时间窗口大小（单位：秒），默认为5.0

    返回值：
    -------
    numpy.ndarray
        浮点数数组，表示每行达到最终能量一半所需的时间（秒）。
        如果在时间窗口内未达到一半能量，或者最终能量为0，则返回time_window值。

    示例：
    -------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from rust_pyfunc import find_half_energy_time
    >>> 
    >>> # 创建示例DataFrame
    >>> df = pd.DataFrame({
    ...     'exchtime': [1.0, 1.1, 1.2, 1.3, 1.4],
    ...     'price': [10.0, 10.2, 10.5, 10.3, 10.1]
    ... })
    >>> 
    >>> # 计算达到一半能量所需时间
    >>> df['half_energy_time'] = find_half_energy_time(
    ...     df['exchtime'].values,
    ...     df['price'].values,
    ...     time_window=5.0
    ... )
    >>> print(df)
    #    exchtime  price  half_energy_time
    # 0      1.0   10.0              2.1  # 在2.1秒时达到5秒能量的一半
    # 1      1.1   10.2              1.9  # 在1.9秒时达到5秒能量的一半
    # 2      1.2   10.5              1.8  # 在1.8秒时达到5秒能量的一半
    # 3      1.3   10.3              1.7  # 在1.7秒时达到5秒能量的一半
    # 4      1.4   10.1              5.0  # 未达到5秒能量的一半
    """
    ...

def calculate_shannon_entropy_change(exchtime: NDArray[np.float64], order: NDArray[np.int64], volume: NDArray[np.float64], price: NDArray[np.float64], window_seconds: float = 0.1, top_k: Optional[int] = None) -> NDArray[np.float64]:
    """计算价格创新高时的香农熵变化。

    参数说明：
    ----------
    exchtime : numpy.ndarray
        交易时间数组，纳秒时间戳，类型为float64
    order : numpy.ndarray
        订单机构ID数组，类型为int64
    volume : numpy.ndarray
        成交量数组，类型为float64
    price : numpy.ndarray
        价格数组，类型为float64
    window_seconds : float
        计算香农熵变的时间窗口，单位为秒
    top_k : Optional[int]
        如果提供，则只计算价格最高的k个点的熵变，默认为None（计算所有价格创新高点）

    返回值：
    -------
    numpy.ndarray
        香农熵变数组，类型为float64。只在价格创新高时计算熵变，其他时刻为NaN。
        熵变值表示在价格创新高时，从当前时刻到未来window_seconds时间窗口内，
        交易分布的变化程度。正值表示分布变得更分散，负值表示分布变得更集中。

    示例：
    -------
    >>> import numpy as np
    >>> from rust_pyfunc import calculate_shannon_entropy_change
    >>> 
    >>> # 创建测试数据
    >>> exchtime = np.array([1e9, 2e9, 3e9, 4e9], dtype=np.float64)  # 时间戳（纳秒）
    >>> order = np.array([100, 200, 300, 400], dtype=np.int64)  # 机构ID
    >>> volume = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float64)
    >>> price = np.array([100.0, 102.0, 101.0, 103.0], dtype=np.float64)
    >>> 
    >>> # 计算3秒窗口的香农熵变
    >>> entropy_changes = calculate_shannon_entropy_change(exchtime, order, volume, price, 3.0)
    >>> print(entropy_changes)  # 只有价格为100.0、102.0和103.0的位置有非NaN值
    >>>
    >>> # 只计算价格最高的2个点的熵变
    >>> entropy_changes = calculate_shannon_entropy_change(exchtime, order, volume, price, 3.0, top_k=2)
    """
    ...

def calculate_shannon_entropy_change_at_low(
    exchtime: NDArray[np.float64],
    order: NDArray[np.int64],
    volume: NDArray[np.float64],
    price: NDArray[np.float64],
    window_seconds: float,
    bottom_k: Optional[int] = None
) -> NDArray[np.float64]:
    """
    在价格创新低时计算香农熵变

    参数：
    ----------
    exchtime : numpy.ndarray
        交易时间数组，纳秒时间戳，类型为float64
    order : numpy.ndarray
        订单机构ID数组，类型为int64
    volume : numpy.ndarray
        成交量数组，类型为float64
    price : numpy.ndarray
        价格数组，类型为float64
    window_seconds : float
        计算香农熵变的时间窗口，单位为秒
    bottom_k : Optional[int]
        如果提供，则只计算价格最低的k个点的熵变，默认为None（计算所有价格创新低点）

    返回值：
    -------
    numpy.ndarray
        香农熵变数组，类型为float64。只在价格创新低时有值，其他位置为NaN。
        熵变值表示在价格创新低时，从当前时刻到未来window_seconds时间窗口内，
        交易分布的变化程度。正值表示分布变得更分散，负值表示分布变得更集中。

    示例：
    -------
    >>> import numpy as np
    >>> from rust_pyfunc import calculate_shannon_entropy_change_at_low
    >>> 
    >>> # 创建测试数据
    >>> exchtime = np.array([1e9, 2e9, 3e9, 4e9], dtype=np.float64)  # 时间戳（纳秒）
    >>> order = np.array([100, 200, 300, 400], dtype=np.int64)  # 机构ID
    >>> volume = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float64)
    >>> price = np.array([100.0, 102.0, 101.0, 103.0], dtype=np.float64)
    >>> 
    >>> # 计算3秒窗口的香农熵变
    >>> entropy_changes = calculate_shannon_entropy_change_at_low(exchtime, order, volume, price, 3.0)
    >>> print(entropy_changes)  # 只有价格为100.0、101.0的位置有非NaN值
    >>>
    >>> # 只计算价格最低的2个点的熵变
    >>> entropy_changes = calculate_shannon_entropy_change_at_low(exchtime, order, volume, price, 3.0, bottom_k=2)
    """
    ...
