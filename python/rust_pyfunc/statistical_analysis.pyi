"""统计分析函数类型声明"""
from typing import List, Optional, Tuple
import numpy as np
from numpy.typing import NDArray

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
    """在价格创新低时计算香农熵变化。

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
    bottom_k : Optional[int]
        如果提供，则只计算价格最低的k个点的熵变，默认为None（计算所有价格创新低点）

    返回值：
    -------
    numpy.ndarray
        香农熵变数组，类型为float64。只在价格创新低时有值，其他位置为NaN。
        熵变值表示在价格创新低时，从当前时刻到未来window_seconds时间窗口内，
        交易分布的变化程度。正值表示分布变得更分散，负值表示分布变得更集中。
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
    """
    ...

def factor_correlation_by_date(
    dates: NDArray[np.int64], 
    ret: NDArray[np.float64], 
    fac: NDArray[np.float64]
) -> tuple[NDArray[np.int64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """按日期计算ret和fac的分组相关系数
    
    对于每个日期，计算三种相关系数：
    1. 全体数据的ret和fac排序值的相关系数
    2. fac小于当日中位数部分的ret和fac排序值的相关系数
    3. fac大于当日中位数部分的ret和fac排序值的相关系数

    参数说明：
    ----------
    dates : NDArray[np.int64]
        日期数组，格式为YYYYMMDD（如20220101）
    ret : NDArray[np.float64]
        收益率数组
    fac : NDArray[np.float64]
        因子值数组
        
    返回值：
    -------
    tuple[NDArray[np.int64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]
        返回四个数组的元组：
        - 日期数组（去重后）
        - 全体数据的相关系数
        - 低因子组的相关系数
        - 高因子组的相关系数
    """
    ...

def factor_grouping(
    dates: NDArray[np.int64], 
    factors: NDArray[np.float64], 
    groups_num: int = 10
) -> NDArray[np.int32]:
    """按日期对因子值进行分组
    
    对于每个日期，将因子值按大小分为指定数量的组，返回每个观测值的分组号。
    
    参数说明：
    ----------
    dates : NDArray[np.int64]
        日期数组，格式为YYYYMMDD（如20220101）
    factors : NDArray[np.float64]
        因子值数组
    groups_num : int, default=10
        分组数量，默认为10
        
    返回值：
    -------
    NDArray[np.int32]
        分组号数组，值从1到groups_num，其中1表示因子值最小的组，groups_num表示因子值最大的组
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
    a : NDArray[np.float64]
        第一个序列
    b : NDArray[np.float64]
        第二个序列
    min_length : int, default=10
        段的最小长度，只有长度大于等于此值的段才计算相关系数
        
    返回值：
    -------
    Tuple[List[float], List[float]]
        返回两个列表的元组：
        - 第一个列表：a>b时段的相关系数
        - 第二个列表：b>a时段的相关系数
    """
    ...

def local_correlation(
    prices: NDArray[np.float64],
    volumes: NDArray[np.float64],
    window_size: int
) -> Tuple[NDArray[np.float64], List[str]]:
    """计算价格序列的局部相关性分析。
    
    对于每个价格点，向前取window_size个值作为局部序列，然后分别向前和向后搜索，
    找到与当前局部序列相关性最大和最小的位置，并计算间隔行数和volume总和。

    参数说明：
    ----------
    prices : NDArray[np.float64]
        价格序列，形状为(n,)
    volumes : NDArray[np.float64]
        成交量序列，形状为(n,)，与价格序列对应
    window_size : int
        局部序列的窗口大小，表示向前取多少个值

    返回值：
    -------
    Tuple[NDArray[np.float64], List[str]]
        返回二维数组和列名列表的元组：
        - 二维数组：n行12列，每行对应输入序列的一个位置
        - 列名列表：包含12个字符串，对应每一列的名称
        
        12列分别为：
        [0] 向后corr最大处间隔行数
        [1] 向后corr最大处间隔volume总和
        [2] 向后corr最小处间隔行数
        [3] 向后corr最小处间隔volume总和
        [4] 向后与corr最大处之间的corr最小处间隔行数
        [5] 向后与corr最大处之间的corr最小处间隔volume总和
        [6] 向前corr最大处间隔行数
        [7] 向前corr最大处间隔volume总和
        [8] 向前corr最小处间隔行数
        [9] 向前corr最小处间隔volume总和
        [10] 向前与corr最大处之间的corr最小处间隔行数
        [11] 向前与corr最大处之间的corr最小处间隔volume总和

    注意：
    -----
    - 如果corr最大处就是离当前行最近的位置，那么找不到它们之间的corr最小处，对应位置设置为NaN
    - 如果没有足够的数据计算相关性，对应位置也会设置为NaN
    """
    ...