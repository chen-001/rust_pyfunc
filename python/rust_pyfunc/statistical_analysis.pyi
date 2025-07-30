"""统计分析函数类型声明"""
from typing import List, Optional, Tuple, Union
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

def fast_correlation_matrix(
    data: NDArray[np.float64],
    method: str = "pearson",
    min_periods: int = 1,
    max_workers: int = 10
) -> NDArray[np.float64]:
    """快速计算相关性矩阵，类似于pandas的df.corr()功能。
    使用并行计算和优化算法大幅提升计算性能。

    参数说明：
    ----------
    data : NDArray[np.float64]
        输入数据矩阵，形状为(n_samples, n_features)，每列代表一个变量
    method : str, default="pearson"
        相关性计算方法，默认为'pearson'。目前只支持皮尔逊相关系数
    min_periods : int, default=1
        计算相关性所需的最小样本数，默认为1
    max_workers : int, default=10
        最大并行工作线程数，默认为10，设置为0则使用所有可用核心

    返回值：
    -------
    NDArray[np.float64]
        相关性矩阵，形状为(n_features, n_features)，对角线元素为1.0

    注意：
    -----
    - 函数使用并行计算和优化算法，性能比pandas.DataFrame.corr()快数倍
    - 自动处理NaN值
    - 相关性矩阵是对称的，对角线元素为1.0
    - 当样本数少于min_periods时，对应的相关系数为NaN
    """
    ...

def fast_correlation_matrix_v2(
    data: NDArray[np.float64],
    method: str = "pearson",
    min_periods: int = 1,
    max_workers: int = 10
) -> NDArray[np.float64]:
    """超快速计算相关性矩阵，进一步优化版本。
    采用SIMD优化、更好的内存访问模式和数值稳定性改进。

    参数说明：
    ----------
    data : NDArray[np.float64]
        输入数据矩阵，形状为(n_samples, n_features)，每列代表一个变量
    method : str, default="pearson"
        相关性计算方法，默认为'pearson'。目前只支持皮尔逊相关系数
    min_periods : int, default=1
        计算相关性所需的最小样本数，默认为1
    max_workers : int, default=10
        最大并行工作线程数，默认为10，设置为0则使用所有可用核心

    返回值：
    -------
    NDArray[np.float64]
        相关性矩阵，形状为(n_features, n_features)，对角线元素为1.0

    注意：
    -----
    - V2版本采用了多项优化：数据预处理、Kahan求和、循环展开、向量化计算
    - 内存访问模式优化，提高缓存命中率
    - 数值稳定性更好，减少浮点数累加误差
    - 对于大数据集性能可能进一步提升
    """
    ...

def calculate_entropy_1d(data: NDArray[np.float64]) -> float:
    """计算一维数组的熵。
    
    对数组中的值进行频次统计，然后计算香农熵：H = -∑(p * ln(p))，
    其中p是每个唯一值出现的概率。
    
    参数说明：
    ----------
    data : NDArray[np.float64]
        输入的一维数组
        
    返回值：
    -------
    float
        计算得到的香农熵值
        
    注意：
    -----
    - 空数组返回0.0
    - NaN值被单独计算为一个唯一值
    - 使用自然对数计算熵值
    """
    ...

def calculate_entropy_2d(
    data: NDArray[np.float64], 
    axis: Optional[int] = None
) -> Union[float, NDArray[np.float64]]:
    """计算二维数组的熵。
    
    可以按指定轴计算每行或每列的熵，或者计算整个数组的熵。
    
    参数说明：
    ----------
    data : NDArray[np.float64]
        输入的二维数组
    axis : Optional[int], default=None
        计算轴向：
        - None: 计算整个数组的熵，返回标量
        - 0: 计算每列的熵，返回形状为(n_cols,)的数组
        - 1: 计算每行的熵，返回形状为(n_rows,)的数组
        
    返回值：
    -------
    Union[float, NDArray[np.float64]]
        - axis=None时返回float
        - axis=0或1时返回一维数组
        
    异常：
    -----
    ValueError
        当axis不为None、0、1时抛出
        
    注意：
    -----
    - 使用并行计算提高性能
    - NaN值被单独计算为一个唯一值
    - 使用自然对数计算熵值
    """
    ...

def calculate_entropy_discrete_1d(data: NDArray[np.int64]) -> float:
    """计算一维离散数组的熵。
    
    专门为整数类型数据优化的熵计算函数，避免浮点数精度问题。
    
    参数说明：
    ----------
    data : NDArray[np.int64]
        输入的一维整数数组
        
    返回值：
    -------
    float
        计算得到的香农熵值
        
    注意：
    -----
    - 空数组返回0.0
    - 直接使用整数值作为键，避免浮点数格式化
    - 使用自然对数计算熵值
    """
    ...

def calculate_entropy_discrete_2d(
    data: NDArray[np.int64], 
    axis: Optional[int] = None
) -> Union[float, NDArray[np.float64]]:
    """计算二维离散数组的熵。
    
    专门为整数类型数据优化的熵计算函数，可以按指定轴计算每行或每列的熵。
    
    参数说明：
    ----------
    data : NDArray[np.int64]
        输入的二维整数数组
    axis : Optional[int], default=None
        计算轴向：
        - None: 计算整个数组的熵，返回标量
        - 0: 计算每列的熵，返回形状为(n_cols,)的数组
        - 1: 计算每行的熵，返回形状为(n_rows,)的数组
        
    返回值：
    -------
    Union[float, NDArray[np.float64]]
        - axis=None时返回float
        - axis=0或1时返回一维数组
        
    异常：
    -----
    ValueError
        当axis不为None、0、1时抛出
        
    注意：
    -----
    - 使用并行计算提高性能
    - 直接使用整数值作为键，避免浮点数格式化问题
    - 使用自然对数计算熵值
    """
    ...

def calculate_binned_entropy_1d(
    data: NDArray[np.float64], 
    n_bins: int,
    bin_method: Optional[str] = "equal_width"
) -> float:
    """计算一维数组的分箱熵。
    
    先将连续数据分箱，然后计算分箱后的熵值。这对于连续数据的熵计算更有意义。
    
    参数说明：
    ----------
    data : NDArray[np.float64]
        输入的一维数组
    n_bins : int
        分箱数量，必须大于0
    bin_method : Optional[str], default="equal_width"
        分箱方法：
        - "equal_width": 等宽分箱，每个分箱的区间长度相等
        - "equal_frequency": 等频分箱，每个分箱包含相近数量的数据点
        
    返回值：
    -------
    float
        分箱后的香农熵值
        
    异常：
    -----
    ValueError
        当n_bins <= 0或bin_method不支持时抛出
        
    注意：
    -----
    - 空数组返回0.0
    - NaN值被分配到单独的分箱（索引为n_bins）
    - 等宽分箱基于数据的最小值和最大值
    - 等频分箱尝试让每个分箱包含相近数量的数据点
    - 使用自然对数计算熵值
    - 熵值范围：0 到 ln(实际使用的分箱数)
    """
    ...

def calculate_binned_entropy_2d(
    data: NDArray[np.float64], 
    n_bins: int,
    bin_method: Optional[str] = "equal_width",
    axis: Optional[int] = None
) -> Union[float, NDArray[np.float64]]:
    """计算二维数组的分箱熵。
    
    先将连续数据分箱，然后按指定轴计算分箱后的熵值。
    
    参数说明：
    ----------
    data : NDArray[np.float64]
        输入的二维数组
    n_bins : int
        分箱数量，必须大于0
    bin_method : Optional[str], default="equal_width"
        分箱方法：
        - "equal_width": 等宽分箱，每个分箱的区间长度相等
        - "equal_frequency": 等频分箱，每个分箱包含相近数量的数据点
    axis : Optional[int], default=None
        计算轴向：
        - None: 计算整个数组的分箱熵，返回标量
        - 0: 计算每列的分箱熵，返回形状为(n_cols,)的数组
        - 1: 计算每行的分箱熵，返回形状为(n_rows,)的数组
        
    返回值：
    -------
    Union[float, NDArray[np.float64]]
        - axis=None时返回float
        - axis=0或1时返回一维数组
        
    异常：
    -----
    ValueError
        当n_bins <= 0、bin_method不支持或axis不为None、0、1时抛出
        
    注意：
    -----
    - 每行/列独立进行分箱和熵计算
    - NaN值被分配到单独的分箱
    - 等宽分箱基于每行/列数据的最小值和最大值
    - 等频分箱基于每行/列数据的排序位置
    - 使用自然对数计算熵值
    - 对于连续数据，这比直接计算熵更有意义
    """
    ...