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

def dtw_distance(s1: List[float], s2: List[float], radius: Optional[int] = None, timeout_seconds: Optional[float] = None) -> float:
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
    timeout_seconds : float, optional
        计算超时时间（秒）。如果计算时间超过此值，将抛出异常。
        默认为None，表示不设置超时。

    返回值：
    -------
    float
        两个序列之间的DTW距离，值越小表示序列越相似
    """
    ...

def fast_dtw_distance(s1: List[float], s2: List[float], radius: Optional[int] = None, timeout_seconds: Optional[float] = None) -> float:
    """计算两个序列之间的动态时间规整(DTW)距离的优化版本。
    此版本使用一维数组代替二维数组，减少内存分配和间接访问，提高计算效率。
    适合用于一般规模的时间序列比较，性能比标准版本提升1.7-2.0倍（无窗口限制）或
    2.4-4.6倍（有窗口限制）。
    
    具有以下优化特性：
    1. 智能初始化窗口内单元格，避免无限值问题
    2. 自动调整radius大小，当指定radius导致结果为inf时，会自动增大半径重试
    3. 针对行首和列首位置提供特殊处理，确保正确传播距离值

    参数说明：
    ----------
    s1 : array_like
        第一个输入序列
    s2 : array_like
        第二个输入序列
    radius : int, optional
        Sakoe-Chiba半径，用于限制规整路径，可以显著提高计算效率。
        如果不指定，则不使用路径限制。
        当指定radius太小导致结果为inf时，函数会自动增大radius值重试计算。
    timeout_seconds : float, optional
        计算超时时间（秒）。如果计算时间超过此值，将抛出异常。
        默认为None，表示不设置超时。

    返回值：
    -------
    float
        两个序列之间的DTW距离，值越小表示序列越相似。
        即使使用较小的radius值，函数也会尽量返回有效结果而不是inf。
        当任一输入序列长度为0时，返回NaN。
    """
    ...

def super_dtw_distance(s1: List[float], s2: List[float], radius: Optional[int] = None, timeout_seconds: Optional[float] = None, lower_bound_pruning: bool = True, early_termination_threshold: Optional[float] = None) -> float:
    """计算两个序列之间的动态时间规整(DTW)距离的超级优化版本。
    此版本使用多种高级优化技术，包括：
    1. 内存预分配 - 减少运行时内存分配
    2. 更精细的内存访问优化 - 提高缓存命中率
    3. 基于启发式的跳过技术 - 避免不必要的计算
    4. 提前退出策略 - 当部分结果已超过最优值时提前终止
    5. 更稀疏的超时检查 - 减少检查开销
    
    特别适合用于大规模时间序列数据或需要比较大量序列的应用场景。

    参数说明：
    ----------
    s1 : array_like
        第一个输入序列
    s2 : array_like
        第二个输入序列
    radius : int, optional
        Sakoe-Chiba半径，用于限制规整路径，可以显著提高计算效率。
        如果不指定，则不使用路径限制。
    timeout_seconds : float, optional
        计算超时时间（秒）。如果计算时间超过此值，将抛出异常。
        默认为None，表示不设置超时。
    lower_bound_pruning : bool, default=True
        是否使用下界剪枝技术。启用时，会计算序列间的简单距离作为DTW距离的下界，
        如果下界已经超过了早停阈值，则直接返回而不进行完整计算。
    early_termination_threshold : float, optional
        提前终止阈值。如果在计算过程中发现当前DTW距离已经超过此阈值，
        则提前终止计算并返回无穷大。适用于只关心小于某阈值的相似序列的场景。

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

def ols_residuals(x: NDArray[np.float64], y: NDArray[np.float64]) -> NDArray[np.float64]:
    """计算普通最小二乘(OLS)回归的残差序列。
    残差表示实际观测值与模型预测值之间的差异: ε = y - Xβ。

    参数说明：
    ----------
    x : numpy.ndarray
        设计矩阵，形状为(n_samples, n_features)
    y : numpy.ndarray
        响应变量，形状为(n_samples,)

    返回值：
    -------
    numpy.ndarray
        残差序列，形状为(n_samples,)
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

def find_follow_volume_sum_same_price(times: NDArray[np.float64], prices: NDArray[np.float64], volumes: NDArray[np.float64], time_window: float = 0.1, check_price: bool = True, filter_ratio: float = 0.0, timeout_seconds: Optional[float] = None) -> NDArray[np.float64]:
    """计算每一行在其后time_window秒内具有相同volume（及可选相同price）的行的volume总和。

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
    check_price : bool, optional
        是否检查价格是否相同，默认为True。设为False时只检查volume是否相同。
    filter_ratio : float, optional, default=0.0
        要过滤的volume数值比例，默认为0（不过滤）。如果大于0，则过滤出现频率最高的前 filter_ratio 比例的volume种类，对应的行会被设为NaN。
    timeout_seconds : float, optional, default=None
        计算超时时间（秒）。如果计算时间超过该值，函数将返回全NaN的数组。默认为None，表示不设置超时限制。

    返回值：
    -------
    numpy.ndarray
        每一行在其后time_window秒内（包括当前行）具有相同条件的行的volume总和。
        如果filter_ratio>0，则出现频率最高的前filter_ratio比例的volume值对应的行会被设为NaN。

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
    >>> # 计算follow列（默认检查price和volume）
    >>> df['follow'] = find_follow_volume_sum_same_price(
    ...     df['exchtime'].values,
    ...     df['price'].values,
    ...     df['volume'].values
    ... )
    >>>
    >>> # 不检查价格，只检查volume是否相同
    >>> df['follow_no_price_check'] = find_follow_volume_sum_same_price(
    ...     df['exchtime'].values,
    ...     df['price'].values,
    ...     df['volume'].values,
    ...     check_price=False
    ... )
    >>>
    >>> # 过滤频繁出现的volume值
    >>> df['follow_filtered'] = find_follow_volume_sum_same_price(
    ...     df['exchtime'].values,
    ...     df['price'].values,
    ...     df['volume'].values,
    ...     filter_ratio=0.3
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

def find_half_energy_time(times: NDArray[np.float64], prices: NDArray[np.float64], time_window: float = 5.0, direction: str = "ignore", timeout_seconds: Optional[float] = None) -> NDArray[np.float64]:
    """计算每一行在其后指定时间窗口内的价格变动能量，并找出首次达到最终能量一半时所需的时间。

    参数说明：
    ----------
    times : numpy.ndarray
        时间戳数组（单位：秒）
    prices : numpy.ndarray
        价格数组
    time_window : float, optional
    direction : str, optional
        计算方向，可选值为"pos"（只考虑上涨）、"neg"（只考虑下跌）、"ignore"（选择变动更大的方向），默认为"ignore"
    timeout_seconds : float, optional
        计算超时时间（秒）。如果计算时间超过该值，函数将返回全NaN的数组。默认为None，表示不设置超时限制
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

def find_half_extreme_time(times: NDArray[np.float64], prices: NDArray[np.float64], time_window: float = 5.0, direction: str = "ignore", timeout_seconds: Optional[float] = None) -> NDArray[np.float64]:
    """计算每个时间点价格达到时间窗口内最大变动一半所需的时间。

    该函数首先在每个时间点的后续时间窗口内找到价格的最大上涨和下跌幅度，
    然后确定主要方向（上涨或下跌），最后计算价格首次达到该方向最大变动一半时所需的时间。

    参数说明：
    ----------
    times : numpy.ndarray
        时间戳数组（单位：秒）
    prices : numpy.ndarray
        价格数组
    time_window : float, optional
        时间窗口大小（单位：秒），默认为5.0
    direction : str, optional
        计算方向，可选值为"pos"（只考虑上涨）、"neg"（只考虑下跌）、"ignore"（选择变动更大的方向），默认为"ignore"
    timeout_seconds : float, optional
        计算超时时间（秒）。如果计算时间超过该值，函数将返回全NaN的数组。默认为None，表示不设置超时限制

    返回值：
    -------
    numpy.ndarray
        浮点数数组，表示每行达到最大变动一半所需的时间（秒）。
        如果在时间窗口内未达到一半变动，则返回time_window值。

    示例：
    -------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from rust_pyfunc import find_half_extreme_time
    >>> 
    >>> # 创建示例DataFrame
    >>> df = pd.DataFrame({
    ...     'exchtime': [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0],
    ...     'price': [10.0, 10.2, 10.5, 10.3, 10.1, 10.0, 9.8, 9.5, 9.3, 9.2, 9.0]
    ... })
    >>> 
    >>> # 计算达到最大变动一半所需时间
    >>> df['half_extreme_time'] = find_half_extreme_time(
    ...     df['exchtime'].values,
    ...     df['price'].values,
    ...     time_window=1.0  # 1秒时间窗口
    ... )
    >>> print(df)
    #     exchtime  price  half_extreme_time
    # 0        1.0   10.0               0.3  # 在0.3秒时达到最大上涨(0.5)的一半(0.25)
    # 1        1.1   10.2               0.3  # 在0.3秒时达到最大上涨(0.3)的一半(0.15)
    # 2        1.2   10.5               1.0  # 最大变动为下跌，但未达到一半
    # 3        1.3   10.3               0.4  # 在0.4秒时达到最大下跌(0.5)的一半(0.25)
    # 4        1.4   10.1               0.3  # 在0.3秒时达到最大下跌(0.6)的一半(0.3)
    # 5        1.5   10.0               0.2  # 在0.2秒时达到最大下跌(0.5)的一半(0.25)
    # 6        1.6    9.8               0.3  # 在0.3秒时达到最大下跌(0.8)的一半(0.4)
    # 7        1.7    9.5               0.2  # 在0.2秒时达到最大下跌(0.7)的一半(0.35)
    # 8        1.8    9.3               0.2  # 在0.2秒时达到最大下跌(0.5)的一半(0.25)
    # 9        1.9    9.2               0.1  # 在0.1秒时达到最大下跌(0.2)的一半(0.1)
    # 10       2.0    9.0               1.0  # 时间窗口内没有后续数据
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

def calculate_base_entropy(exchtime: NDArray[np.float64], order: NDArray[np.int64], volume: NDArray[np.float64], index: int) -> float:
    """计算基准熵 - 基于到当前时间点为止的订单分布计算香农熵。

    参数说明：
    ----------
    exchtime : numpy.ndarray
        交易时间数组，纳秒时间戳，类型为float64
    order : numpy.ndarray
        订单机构ID数组，类型为int64
    volume : numpy.ndarray
        成交量数组，类型为float64
    index : int
        计算熵值的当前索引位置

    返回值：
    -------
    float
        基准熵值，表示到当前时间点为止的订单分布熵

    示例：
    -------
    >>> import numpy as np
    >>> from rust_pyfunc import calculate_base_entropy
    >>> 
    >>> # 创建测试数据
    >>> exchtime = np.array([1e9, 2e9, 3e9, 4e9], dtype=np.float64)  # 时间戳（纳秒）
    >>> order = np.array([100, 200, 100, 300], dtype=np.int64)  # 机构ID
    >>> volume = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float64)
    >>> 
    >>> # 计算索引2处的基准熵
    >>> base_entropy = calculate_base_entropy(exchtime, order, volume, 2)
    >>> print(f"基准熵: {base_entropy}")
    """
    ...

def calculate_window_entropy(exchtime: NDArray[np.float64], order: NDArray[np.int64], volume: NDArray[np.float64], index: int, window_seconds: float) -> float:
    """计算窗口熵 - 基于从当前时间点到未来指定时间窗口内的订单分布计算香农熵。

    参数说明：
    ----------
    exchtime : numpy.ndarray
        交易时间数组，纳秒时间戳，类型为float64
    order : numpy.ndarray
        订单机构ID数组，类型为int64
    volume : numpy.ndarray
        成交量数组，类型为float64
    index : int
        计算熵值的当前索引位置
    window_seconds : float
        向前查看的时间窗口大小，单位为秒

    返回值：
    -------
    float
        窗口熵值，表示从当前时间点到未来指定时间窗口内的订单分布熵

    示例：
    -------
    >>> import numpy as np
    >>> from rust_pyfunc import calculate_window_entropy
    >>> 
    >>> # 创建测试数据
    >>> exchtime = np.array([1e9, 2e9, 3e9, 4e9], dtype=np.float64)  # 时间戳（纳秒）
    >>> order = np.array([100, 200, 100, 300], dtype=np.int64)  # 机构ID
    >>> volume = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float64)
    >>> 
    >>> # 计算索引1处的3秒窗口熵
    >>> window_entropy = calculate_window_entropy(exchtime, order, volume, 1, 3.0)
    >>> print(f"窗口熵: {window_entropy}")
    >>> 
    >>> # 计算熵变（可以通过组合两个函数）
    >>> base = calculate_base_entropy(exchtime, order, volume, 1)
    >>> window = calculate_window_entropy(exchtime, order, volume, 1, 3.0)
    >>> entropy_change = window - base
    >>> print(f"熵变: {entropy_change}")
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

def brachistochrone_curve(x1: float, y1: float, x2: float, y2: float, x_series: NDArray[np.float64], timeout_seconds: Optional[float] = None) -> NDArray[np.float64]:
    """计算最速曲线（投掷线）并返回x_series对应的y坐标。
    
    最速曲线是指在重力作用下，一个质点从一点到另一点所需时间最短的路径，也被称为投掷线或摆线。
    其参数方程为：x = R(θ - sin θ), y = -R(1 - cos θ)。

    参数说明：
    ----------
    x1 : float
        起点x坐标
    y1 : float
        起点y坐标
    x2 : float
        终点x坐标
    y2 : float
        终点y坐标
    x_series : numpy.ndarray
        需要计算y坐标的x点序列
    timeout_seconds : float, optional
        计算超时时间，单位为秒。如果函数执行时间超过此值，将立即中断计算并抛出异常。默认值为None，表示无超时限制。

    返回值：
    -------
    numpy.ndarray
        与x_series相对应的y坐标值数组。对于超出曲线定义域的x值，返回NaN。
        
    异常：
    ------
    RuntimeError
        当计算时间超过timeout_seconds指定的秒数时抛出，错误信息包含具体的超时时长。

    示例：
    -------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from rust_pyfunc import brachistochrone_curve
    >>> 
    >>> # 创建x序列
    >>> x_vals = pd.Series(np.linspace(0, 5, 100))
    >>> # 计算从点(0,0)到点(5,-3)的最速曲线，设置5秒超时
    >>> try:
    >>>     y_vals = brachistochrone_curve(0.0, 0.0, 5.0, -3.0, x_vals, 5.0)
    >>> except RuntimeError as e:
    >>>     print(f"计算超时: {e}")
    >>>     
    >>> # 正常计算示例（不设置超时）
    >>> y_vals = brachistochrone_curve(0.0, 0.0, 5.0, -3.0, x_vals)
    >>> 
    >>> # 绘制曲线
    >>> import matplotlib.pyplot as plt
    >>> plt.figure(figsize=(10, 6))
    >>> plt.plot(x_vals, y_vals)
    >>> plt.scatter([0, 5], [0, -3], color='red', s=50)  # 标记起点和终点
    >>> plt.grid(True)
    >>> plt.title('最速曲线 (Brachistochrone Curve)')
    >>> plt.xlabel('x')
    >>> plt.ylabel('y')
    >>> plt.show()
    """
    ...

def brachistochrone_curve_v2(x1: float, y1: float, x2: float, y2: float, x_series: NDArray[np.float64], timeout_seconds: Optional[float] = None) -> NDArray[np.float64]:
    """计算最速曲线（投掷线）的修正版，确保终点严格一致。
    
    这是brachistochrone_curve函数的修正版，解决了原版函数可能存在的终点不一致问题。
    通过强制约束终点坐标，确保计算结果的数学正确性。最速曲线是指在重力作用下，
    一个质点从一点到另一点所需时间最短的路径，也被称为投掷线或摆线。
    其参数方程为：x = R(θ - sin θ), y = -R(1 - cos θ)。

    参数说明：
    ----------
    x1 : float
        起点x坐标
    y1 : float
        起点y坐标
    x2 : float
        终点x坐标
    y2 : float
        终点y坐标
    x_series : numpy.ndarray
        需要计算y坐标的x点序列
    timeout_seconds : float, optional
        计算超时时间，单位为秒。如果函数执行时间超过此值，将立即中断计算并抛出异常。默认值为None，表示无超时限制。

    返回值：
    -------
    numpy.ndarray
        与x_series相对应的y坐标值数组，确保起点和终点严格一致。
        对于超出曲线定义域的x值，返回NaN。
        
    异常：
    ------
    RuntimeError
        当计算时间超过timeout_seconds指定的秒数时抛出，错误信息包含具体的超时时长。

    特点：
    ------
    1. 严格的终点约束 - 确保曲线精确通过指定的起点和终点
    2. 改进的优化算法 - 使用更稳定的数值求解方法
    3. 特殊情况处理 - 正确处理垂直线、水平线和重合点等边界情况
    4. 提高的数值稳定性 - 减少计算误差和发散问题

    示例：
    -------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from rust_pyfunc import brachistochrone_curve_v2
    >>> 
    >>> # 创建x序列
    >>> x_vals = pd.Series(np.linspace(0, 5, 100))
    >>> # 计算从点(0,0)到点(5,-3)的最速曲线修正版，设置5秒超时
    >>> try:
    >>>     y_vals = brachistochrone_curve_v2(0.0, 0.0, 5.0, -3.0, x_vals, 5.0)
    >>> except RuntimeError as e:
    >>>     print(f"计算超时: {e}")
    >>>     
    >>> # 正常计算示例（不设置超时）
    >>> y_vals = brachistochrone_curve_v2(0.0, 0.0, 5.0, -3.0, x_vals)
    >>> 
    >>> # 验证终点是否精确匹配
    >>> x_end_index = np.argmin(np.abs(x_vals - 5.0))
    >>> y_end_computed = y_vals[x_end_index]
    >>> print(f"期望终点: (5.0, -3.0)")
    >>> print(f"计算终点: (5.0, {y_end_computed:.10f})")
    >>> 
    >>> # 绘制曲线
    >>> import matplotlib.pyplot as plt
    >>> plt.figure(figsize=(10, 6))
    >>> plt.plot(x_vals, y_vals, label='最速曲线修正版')
    >>> plt.scatter([0, 5], [0, -3], color='red', s=50, label='起点和终点')
    >>> plt.grid(True)
    >>> plt.title('最速曲线修正版 (Brachistochrone Curve V2)')
    >>> plt.xlabel('x')
    >>> plt.ylabel('y')
    >>> plt.legend()
    >>> plt.show()
    """
    ...

def rolling_volatility(prices: NDArray[np.float64], lookback: int, interval: int, min_periods: int = 2) -> NDArray[np.float64]:
    """计算价格序列的滚动波动率。

    对于每个位置，从该位置向前取n个点，间隔k个点取样，
    计算每个取出的价格点相对上一个价格点的对数收益率，
    然后计算这些收益率的标准差作为波动率。

    参数说明：
    ----------
    prices : numpy.ndarray
        价格序列，形状为(n_samples,)，类型为float64
    lookback : int
        向前查看的样本点数量
    interval : int
        取样间隔，每隔多少个点取一个样本
    min_periods : int, optional
        计算波动率所需的最小样本数，默认为2

    返回值：
    -------
    numpy.ndarray
        与输入序列等长的波动率序列，每个元素表示该位置的历史波动率。
        对于没有足够样本点的位置，返回NaN。

    示例：
    -------
    >>> import numpy as np
    >>> from rust_pyfunc import rolling_volatility
    >>> 
    >>> # 创建价格序列
    >>> prices = np.array([100.0, 102.0, 101.0, 103.0, 102.0, 104.0, 103.0])
    >>> 
    >>> # 计算滚动波动率，向前查看5个点，每隔1个点取样
    >>> vol = rolling_volatility(prices, 5, 1)
    >>> print(vol)
    """
    ...

def rolling_cv(values: NDArray[np.float64], lookback: int, interval: int, min_periods: int = 2) -> NDArray[np.float64]:
    """计算数值序列的滚动变异系数(CV)。

    对于每个位置，从该位置向前取n个点，间隔k个点取样，
    计算每个取出的价格点相对上一个价格点的对数收益率，
    然后计算这些收益率的变异系数（标准差除以均值绝对值）。

    变异系数是标准化的离散程度测量，可用于比较不同单位或数量级的数据。
    与波动率相比，变异系数考虑了均值信息，能更好地反映相对波动程度。

    参数说明：
    ----------
    values : numpy.ndarray
        价格序列，形状为(n_samples,)，类型为float64
    lookback : int
        向前查看的样本点数量
    interval : int
        取样间隔，每隔多少个点取一个样本
    min_periods : int, optional
        计算变异系数所需的最小样本数，默认为2

    返回值：
    -------
    numpy.ndarray
        与输入序列等长的变异系数序列，每个元素表示该位置的历史收益率变异系数。
        对于没有足够样本点的位置或收益率均值接近零的位置，返回NaN。

    示例：
    -------
    >>> import numpy as np
    >>> from rust_pyfunc import rolling_cv
    >>> 
    >>> # 创建价格序列
    >>> prices = np.array([100.0, 102.0, 101.0, 103.0, 102.0, 104.0, 103.0])
    >>> 
    >>> # 计算滚动变异系数，向前查看5个点，每隔1个点取样
    >>> cv = rolling_cv(prices, 5, 1)
    >>> print(cv)
    """
    ...

def rolling_qcv(values: NDArray[np.float64], lookback: int, interval: int, min_periods: int = 3) -> NDArray[np.float64]:
    """计算数值序列的滚动四分位变异系数(QCV)。

    对于每个位置，从该位置向前取n个点，间隔k个点取样，
    计算每个取出的价格点相对上一个价格点的对数收益率，
    然后计算这些收益率的四分位变异系数（四分位间距IQR除以中位数绝对值）。

    四分位变异系数是标准变异系数的稳健替代方案，对异常值和均值接近零的情况
    更具鲁棒性。当收益率分布接近对称分布时，QCV与传统CV趋于一致。

    参数说明：
    ----------
    values : numpy.ndarray
        价格序列，形状为(n_samples,)，类型为float64
    lookback : int
        向前查看的样本点数量
    interval : int
        取样间隔，每隔多少个点取一个样本
    min_periods : int, optional
        计算变异系数所需的最小样本数，默认为3（需要至少3个点计算有意义的IQR）

    返回值：
    -------
    numpy.ndarray
        与输入序列等长的四分位变异系数序列，每个元素表示该位置的历史收益率四分位变异系数。
        对于没有足够样本点的位置或收益率中位数接近零的位置，返回NaN。

    示例：
    -------
    >>> import numpy as np
    >>> from rust_pyfunc import rolling_qcv
    >>> 
    >>> # 创建价格序列
    >>> prices = np.array([100.0, 102.0, 101.0, 103.0, 102.0, 104.0, 103.0, 105.0, 106.0])
    >>> 
    >>> # 计算滚动四分位变异系数，向前查看5个点，每隔1个点取样
    >>> qcv = rolling_qcv(prices, 5, 1)
    >>> print(qcv)
    """
    ...

def calculate_large_order_nearby_small_order_time_gap(
    volumes: NDArray[np.float64],
    exchtimes: NDArray[np.float64],
    large_quantile: float,
    small_quantile: float,
    near_number: int,
    exclude_same_time: bool = False,
    order_type: str = "small",
    flags: Optional[NDArray[np.int32]] = None,
    flag_filter: str = "ignore"
) -> NDArray[np.float64]:
    """计算每个大单与其临近订单之间的时间间隔均值。

    参数说明：
    ----------
    volumes : numpy.ndarray
        交易量数组
    exchtimes : numpy.ndarray
        交易时间数组（单位：纳秒）
    large_quantile : float
        大单的分位点阈值
    small_quantile : float
        小单的分位点阈值
    near_number : int
        每个大单要考虑的临近订单数量
    exclude_same_time : bool, default=False
        是否排除与大单时间戳相同的订单
    order_type : str, default="small"
        指定与大单计算时间间隔的订单类型：
        - "small"：计算大单与小于small_quantile分位点的订单的时间间隔
        - "mid"：计算大单与位于small_quantile和large_quantile分位点之间的订单的时间间隔
        - "full"：计算大单与小于large_quantile分位点的所有订单的时间间隔
    flags : Optional[NDArray[np.int32]], default=None
        交易标志数组，通常66表示主动买入，83表示主动卖出
    flag_filter : str, default="ignore"
        指定如何根据交易标志筛选计算对象：
        - "same"：只计算与大单交易标志相同的订单的时间间隔
        - "diff"：只计算与大单交易标志不同的订单的时间间隔
        - "ignore"：忽略交易标志，计算所有符合条件的订单的时间间隔

    返回值：
    -------
    numpy.ndarray
        浮点数数组，与输入volumes等长。对于大单，返回其与临近目标订单的时间间隔均值（秒）；
        对于非大单，返回NaN。

    示例：
    -------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from rust_pyfunc import calculate_large_order_nearby_small_order_time_gap
    >>> 
    >>> # 创建示例DataFrame
    >>> df = pd.DataFrame({
    >>>     'exchtime': [1.0e9, 1.1e9, 1.2e9, 1.3e9, 1.4e9],  # 纳秒时间戳
    >>>     'volume': [100, 10, 200, 20, 150]
    >>> })
    >>> 
    >>> # 计算大单与临近小单的时间间隔
    >>> df['time_gap'] = calculate_large_order_nearby_small_order_time_gap(
    >>>     df['volume'].values,
    >>>     df['exchtime'].values,
    >>>     large_quantile=0.7,  # 70%分位点以上为大单
    >>>     small_quantile=0.3,  # 30%分位点以下为小单
    >>>     near_number=2        # 每个大单考虑最近的2个小单
    >>> )
    >>> print(df)
    """
    ...

def rolling_dtw_distance(son: NDArray[np.float64], dragon: NDArray[np.float64], exchtime: NDArray[np.float64], minute_back: float) -> NDArray[np.float64]:
    """计算滚动DTW距离：计算son中每一行与其前n分钟片段和dragon的DTW距离。

    参数说明：
    ----------
    son : array_like
        主要时间序列，将在此序列上滚动计算DTW距离
    dragon : array_like
        参考时间序列，用于计算DTW距离的模板
    exchtime : array_like
        时间戳数组，必须与son长度相同
    minute_back : int
        滚动窗口大小，以分钟为单位，表示每次计算使用的历史数据长度

    返回值：
    -------
    numpy.ndarray
        与son等长的数组，包含每个点的DTW距离，其中部分位置可能为NaN
        （如果相应位置的历史数据不足以计算DTW距离）

    示例：
    -------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from rust_pyfunc import rolling_dtw_distance
    >>> 
    >>> # 准备数据
    >>> times = pd.date_range('2023-01-01', periods=100, freq='T')
    >>> son = pd.Series(np.sin(np.linspace(0, 10, 100)), index=times)
    >>> dragon = pd.Series([0, 0.5, 1, 0.5, 0, -0.5, -1, -0.5, 0]) # 一个波形模板
    >>> 
    >>> # 计算滚动DTW距离
    >>> # 每个点与其前5分钟数据和dragon的DTW距离
    >>> result = rolling_dtw_distance(son.values, dragon.values, times.astype(np.int64).values, 5)
    >>> dtw_series = pd.Series(result, index=times)
    """
    ...

def dataframe_corrwith(df1: NDArray[np.float64], df2: NDArray[np.float64], axis: int = 0, drop_na: bool = True) -> NDArray[np.float64]:
    """计算两个数据框对应列的相关系数。

    这个函数类似于pandas中的df.corrwith(df1)，计算两个数据框中对应列之间的皮尔逊相关系数。
    相关系数范围为[-1, 1]，其中：
    - 1表示完全正相关
    - -1表示完全负相关
    - 0表示无相关性

    参数说明：
    ----------
    df1 : numpy.ndarray
        第一个数据框，形状为(n_rows, n_cols)，必须是float64类型
    df2 : numpy.ndarray
        第二个数据框，形状为(n_rows, m_cols)，必须是float64类型
    axis : int, 默认为0
        计算相关性的轴，默认为0（按列计算）。目前只支持按列计算。
    drop_na : bool, 默认为True
        是否忽略计算中的NaN值

    返回值：
    -------
    numpy.ndarray
        一维数组，长度为min(n_cols, m_cols)，包含对应列的相关系数

    示例：
    -------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from rust_pyfunc import dataframe_corrwith
    >>> 
    >>> # 创建两个数据框
    >>> df1 = pd.DataFrame({
    >>>     'A': [1.0, 2.0, 3.0, 4.0, 5.0],
    >>>     'B': [5.0, 4.0, 3.0, 2.0, 1.0],
    >>>     'C': [2.0, 4.0, 6.0, 8.0, 10.0]
    >>> })
    >>> df2 = pd.DataFrame({
    >>>     'A': [1.1, 2.2, 2.9, 4.1, 5.2],
    >>>     'B': [5.2, 4.1, 2.9, 2.1, 0.9],
    >>>     'D': [1.0, 2.0, 3.0, 4.0, 5.0]
    >>> })
    >>> 
    >>> # 计算相关系数
    >>> corr = dataframe_corrwith(df1.values, df2.values)
    >>> # 转换为Series以获得与pandas相同的输出格式
    >>> result = pd.Series(corr, index=['A', 'B', 'C'])
    >>> # 只保留有效对应列（A和B）
    >>> result = result.iloc[:2]
    >>> print(result)
    """
    ...

def fast_find_half_extreme_time(times: NDArray[np.float64], prices: NDArray[np.float64], time_window: float = 5.0, direction: str = "ignore", timeout_seconds: Optional[float] = None) -> NDArray[np.float64]:
    """计算每个时间点价格达到时间窗口内最大变动一半所需的时间（优化版本）。

    该函数是find_half_extreme_time的高性能优化版本，采用了以下优化技术：
    1. 预计算和缓存 - 避免重复计算时间差和比率
    2. 数据布局优化 - 改进内存访问模式
    3. 条件分支优化 - 减少分支预测失败
    4. 界限优化 - 提前确定搜索范围
    5. 算法优化 - 使用二分查找定位目标点

    参数说明：
    ----------
    times : numpy.ndarray
        时间戳数组（单位：秒）
    prices : numpy.ndarray
        价格数组
    time_window : float, optional
        时间窗口大小（单位：秒），默认为5.0
    direction : str, optional
        计算方向，可选值为"pos"（只考虑上涨）、"neg"（只考虑下跌）、"ignore"（选择变动更大的方向），默认为"ignore"
    timeout_seconds : float, optional
        计算超时时间（秒）。如果计算时间超过该值，函数将返回全NaN的数组。默认为None，表示不设置超时限制

    返回值：
    -------
    numpy.ndarray
        浮点数数组，表示每行达到最大变动一半所需的时间（秒）。
        如果在时间窗口内未达到一半变动，则返回time_window值。
        如果计算超时，则返回全为NaN的数组。

    示例：
    -------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from rust_pyfunc import fast_find_half_extreme_time
    >>> 
    >>> # 创建示例DataFrame
    >>> df = pd.DataFrame({
    ...     'exchtime': [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0],
    ...     'price': [10.0, 10.2, 10.5, 10.3, 10.1, 10.0, 9.8, 9.5, 9.3, 9.2, 9.0]
    ... })
    >>> 
    >>> # 计算达到最大变动一半所需时间（优化版本）
    >>> df['half_extreme_time'] = fast_find_half_extreme_time(
    ...     df['exchtime'].values,
    ...     df['price'].values,
    ...     time_window=1.0  # 1秒时间窗口
    ... )
    >>> print(df)
    """
    ...

def super_find_half_extreme_time(times: NDArray[np.float64], prices: NDArray[np.float64], time_window: float = 5.0, direction: str = "ignore", timeout_seconds: Optional[float] = None) -> NDArray[np.float64]:
    """计算每个时间点价格达到时间窗口内最大变动一半所需的时间（超级优化版本）。

    该函数是find_half_extreme_time的高度优化版本，针对大数据量设计，采用了以下优化技术：
    1. SIMD加速 - 利用向量化操作加速计算
    2. 高级缓存优化 - 通过预计算和数据布局进一步提高缓存命中率
    3. 直接内存操作 - 减少边界检查和间接访问
    4. 预先筛选 - 先过滤掉不可能的时间范围
    5. 多线程并行 - 在可能的情况下使用并行计算
    6. 二分查找 - 更高效地定位目标变动点

    参数说明：
    ----------
    times : numpy.ndarray
        时间戳数组（单位：秒）
    prices : numpy.ndarray
        价格数组
    time_window : float, optional
        时间窗口大小（单位：秒），默认为5.0
    direction : str, optional
        计算方向，可选值为"pos"（只考虑上涨）、"neg"（只考虑下跌）、"ignore"（选择变动更大的方向），默认为"ignore"
    timeout_seconds : float, optional
        计算超时时间（秒）。如果计算时间超过该值，函数将返回全NaN的数组。默认为None，表示不设置超时限制

    返回值：
    -------
    numpy.ndarray
        浮点数数组，表示每行达到最大变动一半所需的时间（秒）。
        如果在时间窗口内未达到一半变动，则返回time_window值。
        如果计算超时，则返回全为NaN的数组。

    示例：
    -------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from rust_pyfunc import super_find_half_extreme_time
    >>> 
    >>> # 创建示例DataFrame
    >>> df = pd.DataFrame({
    ...     'exchtime': [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0],
    ...     'price': [10.0, 10.2, 10.5, 10.3, 10.1, 10.0, 9.8, 9.5, 9.3, 9.2, 9.0]
    ... })
    >>> 
    >>> # 计算达到最大变动一半所需时间（超级优化版本）
    >>> df['half_extreme_time'] = super_find_half_extreme_time(
    ...     df['exchtime'].values,
    ...     df['price'].values,
    ...     time_window=1.0  # 1秒时间窗口
    ... )
    >>> print(df)
    """
    ...

def segment_and_correlate(
    a: NDArray[np.float64],
    b: NDArray[np.float64],
    min_length: int = 10
) -> Tuple[List[float], List[float]]:
    """序列分段和相关系数计算函数
    
    输入两个等长的序列，根据大小关系进行分段，然后计算每段内的相关系数。
    当a>b和b>a互相反超时会划分出新的段，只有长度大于等于min_length的段才会被计算。
    
    参数说明：
    ----------
    a : numpy.ndarray
        第一个序列，类型为float64
    b : numpy.ndarray  
        第二个序列，类型为float64，必须与a等长
    min_length : int, optional
        最小段长度，默认为10。只有长度大于等于此值的段才会被计算
        
    返回值：
    -------
    tuple
        返回一个元组(a_greater_corrs, b_greater_corrs)，其中：
        - a_greater_corrs: a > b的段中的相关系数列表
        - b_greater_corrs: b > a的段中的相关系数列表
        
    示例：
    -------
    >>> import numpy as np
    >>> from rust_pyfunc import segment_and_correlate
    >>> 
    >>> # 创建测试序列
    >>> a = np.array([1.0, 2.0, 3.0, 2.0, 1.0, 0.5, 1.5, 2.5, 3.5, 4.0, 3.0, 2.0, 1.0], dtype=np.float64)
    >>> b = np.array([0.5, 1.5, 2.5, 3.0, 2.0, 1.0, 2.0, 3.0, 2.5, 3.5, 4.0, 3.5, 2.5], dtype=np.float64)
    >>> 
    >>> # 计算相关系数，最小段长度为5  
    >>> a_greater_corrs, b_greater_corrs = segment_and_correlate(a, b, 5)
    >>> print(f"a > b段的相关系数: {a_greater_corrs}")
    >>> print(f"b > a段的相关系数: {b_greater_corrs}")
    """
    ...

def analyze_retreat_advance(
    trade_times: NDArray[np.float64],
    trade_prices: NDArray[np.float64], 
    trade_volumes: NDArray[np.float64],
    trade_flags: NDArray[np.float64],
    orderbook_times: NDArray[np.float64],
    orderbook_prices: NDArray[np.float64],
    orderbook_volumes: NDArray[np.float64],
    volume_percentile: Optional[float] = 99.0,
    time_window_minutes: Optional[float] = 1.0
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """分析股票交易中的"以退为进"现象
    
    该函数分析当价格触及某个局部高点后回落，然后在该价格的异常大挂单量消失后
    成功突破该价格的现象。
    
    参数说明：
    ----------
    trade_times : NDArray[np.float64]
        逐笔成交数据的时间戳序列（纳秒时间戳）
    trade_prices : NDArray[np.float64]
        逐笔成交数据的价格序列
    trade_volumes : NDArray[np.float64]
        逐笔成交数据的成交量序列
    trade_flags : NDArray[np.float64]
        逐笔成交数据的标志序列（买卖方向，正数表示买入，负数表示卖出）
    orderbook_times : NDArray[np.float64]
        盘口快照数据的时间戳序列（纳秒时间戳）
    orderbook_prices : NDArray[np.float64]
        盘口快照数据的价格序列
    orderbook_volumes : NDArray[np.float64]
        盘口快照数据的挂单量序列
    volume_percentile : Optional[float], default=99.0
        异常大挂单量的百分位数阈值，默认为99.0（即前1%）
    time_window_minutes : Optional[float], default=1.0
        检查异常大挂单量的时间窗口（分钟），默认为1.0分钟
    
    返回值：
    -------
    Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]
        包含6个数组的元组：
        - 过程期间的成交量
        - 过程期间首次观察到的价格x在盘口上的异常大挂单量
        - 过程开始后指定时间窗口内的成交量
        - 过程期间的主动买入成交量占比
        - 过程期间的价格种类数
        - 过程期间价格相对局部高点的最大下降比例
    
    Python调用示例：
    >>> import numpy as np
    >>> from rust_pyfunc import analyze_retreat_advance
    >>> 
    >>> # 准备逐笔成交数据
    >>> trade_times = np.array([9.30, 9.31, 9.32, 9.33, 9.34], dtype=np.float64)
    >>> trade_prices = np.array([10.0, 10.1, 10.2, 10.1, 10.0], dtype=np.float64)
    >>> trade_volumes = np.array([100, 200, 150, 300, 250], dtype=np.float64)
    >>> trade_flags = np.array([1, 1, 1, -1, -1], dtype=np.float64)
    >>> 
    >>> # 准备盘口快照数据
    >>> orderbook_times = np.array([9.30, 9.31, 9.32, 9.33, 9.34], dtype=np.float64)
    >>> orderbook_prices = np.array([10.0, 10.1, 10.2, 10.1, 10.0], dtype=np.float64)
    >>> orderbook_volumes = np.array([1000, 5000, 8000, 2000, 1500], dtype=np.float64)
    >>> 
    >>> # 分析"以退为进"现象
    >>> results = analyze_retreat_advance(
    ...     trade_times, trade_prices, trade_volumes, trade_flags,
    ...     orderbook_times, orderbook_prices, orderbook_volumes
    ... )
    >>> 
    >>> process_volumes, large_volumes, one_min_volumes, buy_ratios, price_counts, max_declines = results
    >>> print(f"找到 {len(process_volumes)} 个以退为进过程")
    """
    ...

def analyze_retreat_advance_v2(
    trade_times: NDArray[np.float64],
    trade_prices: NDArray[np.float64], 
    trade_volumes: NDArray[np.float64],
    trade_flags: NDArray[np.float64],
    orderbook_times: NDArray[np.float64],
    orderbook_prices: NDArray[np.float64],
    orderbook_volumes: NDArray[np.float64],
    volume_percentile: Optional[float] = 99.0,
    time_window_minutes: Optional[float] = 1.0,
    breakthrough_threshold: Optional[float] = 0.0,
    dedup_time_seconds: Optional[float] = 30.0
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """分析股票交易中的"以退为进"现象（纳秒版本）
    
    该函数分析当价格触及某个局部高点后回落，然后在该价格的异常大挂单量消失后
    成功突破该价格的现象。这是analyze_retreat_advance函数的改进版本，专门为处理
    纳秒级时间戳而优化，并包含局部高点去重功能。
    
    参数说明：
    ----------
    trade_times : NDArray[np.float64]
        逐笔成交数据的时间戳序列（纳秒时间戳）
    trade_prices : NDArray[np.float64]
        逐笔成交数据的价格序列
    trade_volumes : NDArray[np.float64]
        逐笔成交数据的成交量序列
    trade_flags : NDArray[np.float64]
        逐笔成交数据的标志序列（买卖方向），66表示主动买入，83表示主动卖出
    orderbook_times : NDArray[np.float64]
        盘口快照数据的时间戳序列（纳秒时间戳）
    orderbook_prices : NDArray[np.float64]
        盘口快照数据的价格序列
    orderbook_volumes : NDArray[np.float64]
        盘口快照数据的挂单量序列
    volume_percentile : Optional[float], default=99.0
        异常大挂单量的百分位数阈值，默认为99.0（即前1%）
    time_window_minutes : Optional[float], default=1.0
        检查异常大挂单量的时间窗口（分钟），默认为1.0分钟
    breakthrough_threshold : Optional[float], default=0.0
        突破阈值（百分比），默认为0.0（即只要高于局部高点任何幅度都算突破）
        例如：0.1表示需要高出局部高点0.1%才算突破
    dedup_time_seconds : Optional[float], default=30.0
        去重时间阈值（秒），默认为30.0。相同价格且时间间隔小于此值的局部高点将被视为重复
    
    返回值：
    -------
    Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]
        包含9个数组的元组：
        - 过程期间的成交量
        - 局部高点价格在盘口上时间最近的挂单量
        - 过程开始后指定时间窗口内的成交量
        - 过程期间的主动买入成交量占比
        - 过程期间的价格种类数
        - 过程期间价格相对局部高点的最大下降比例
        - 过程持续时间（秒）
        - 过程开始时间（纳秒时间戳）
        - 局部高点的价格
    
    特点：
    ------
    1. 纳秒级时间戳处理 - 专门优化处理纳秒级别的高精度时间戳
    2. 改进的局部高点识别 - 使用更准确的算法识别价格局部高点
    3. 可配置的局部高点去重功能 - 对相同价格且时间接近的局部高点进行去重，时间阈值可自定义
    4. 优化的异常挂单量检测 - 增强了对异常大挂单量的识别精度
    5. 可配置的突破条件 - 通过breakthrough_threshold参数自定义突破阈值
    6. 时间窗口控制 - 设置4小时最大搜索窗口，避免无限搜索
    
    Python调用示例：
    >>> import numpy as np
    >>> from rust_pyfunc import analyze_retreat_advance_v2
    >>> 
    >>> # 准备数据（纳秒时间戳）
    >>> trade_times = np.array([1661743800000000000, 1661743860000000000, 1661743920000000000], dtype=np.float64)
    >>> trade_prices = np.array([10.0, 10.1, 10.2], dtype=np.float64)
    >>> trade_volumes = np.array([100, 200, 150], dtype=np.float64)
    >>> trade_flags = np.array([66, 66, 83], dtype=np.float64)
    >>> 
    >>> orderbook_times = np.array([1661743800000000000, 1661743860000000000], dtype=np.float64)
    >>> orderbook_prices = np.array([10.0, 10.1], dtype=np.float64)
    >>> orderbook_volumes = np.array([1000, 5000], dtype=np.float64)
    >>> 
    >>> # 分析"以退为进"现象，使用2分钟时间窗口，0.1%突破阈值，60秒去重时间
    >>> results = analyze_retreat_advance_v2(
    ...     trade_times, trade_prices, trade_volumes, trade_flags,
    ...     orderbook_times, orderbook_prices, orderbook_volumes,
    ...     volume_percentile=95.0, time_window_minutes=2.0, breakthrough_threshold=0.1, dedup_time_seconds=60.0
    ... )
    >>> 
    >>> process_volumes, large_volumes, time_window_volumes, buy_ratios, price_counts, max_declines, process_durations, process_start_times, peak_prices = results
    >>> print(f"找到 {len(process_volumes)} 个以退为进过程")
    """
    ...
