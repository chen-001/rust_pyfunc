"""统计分析函数类型声明"""
from typing import List, Optional, Tuple, Union
import numpy as np
from numpy.typing import NDArray

def column_correlation_fast(array1: NDArray[np.float64], array2: NDArray[np.float64]) -> NDArray[np.float64]:
    """快速计算两个二维数组对应列的相关系数。

    使用高度优化的算法计算两个n×n数组对应列之间的皮尔逊相关系数。
    采用Welford's online算法确保数值稳定性，优化内存访问模式以提升性能。

    参数说明：
    ----------
    array1 : numpy.ndarray
        第一个输入数组，形状为(n, n)，dtype=float64
    array2 : numpy.ndarray
        第二个输入数组，形状为(n, n)，dtype=float64

    返回值：
    -------
    numpy.ndarray
        一维数组，形状为(n,)，包含每列的相关系数

    性能：
    ----
    - 时间复杂度: O(n²)
    - 空间复杂度: O(n)
    - 当n=5000时，执行时间<0.5秒
    """
    ...

def column_correlation_batch(array1: NDArray[np.float64], array2: NDArray[np.float64]) -> NDArray[np.float64]:
    """批量计算多列相关系数的优化版本。

    为了进一步提升性能，使用批量处理和更好的缓存局部性。

    参数说明：
    ----------
    array1 : numpy.ndarray
        第一个输入数组，形状为(n, n)，dtype=float64
    array2 : numpy.ndarray
        第二个输入数组，形状为(n, n)，dtype=float64

    返回值：
    -------
    numpy.ndarray
        一维数组，形状为(n,)，包含每列的相关系数
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

def rolling_correlation_mean(
    data: NDArray[np.float64],
    window_size: int,
    min_periods: Optional[int] = None,
    max_workers: int = 10
) -> NDArray[np.float64]:
    """滚动窗口计算相关性矩阵均值。
    
    对于输入数据的每一行，计算其过去window_size行的相关性矩阵，
    然后计算该相关性矩阵中所有值的均值。
    
    参数说明：
    ----------
    data : NDArray[np.float64]
        输入数据矩阵，形状为(n_samples, n_features)，每列代表一个变量
    window_size : int
        滚动窗口的大小，表示向前取多少行数据计算相关性矩阵
    min_periods : Optional[int], default=None
        计算相关性所需的最小样本数，默认为window_size
    max_workers : int, default=10
        最大并行工作线程数，默认为10，设置为0则使用所有可用核心
        
    返回值：
    -------
    NDArray[np.float64]
        一维数组，长度等于输入数据的行数，每个值为对应行的相关性矩阵均值
        
    注意：
    -----
    - 对于前window_size-1行，如果可用数据少于min_periods，则返回NaN
    - 从第window_size行开始，使用完整的窗口进行计算
    - 相关性矩阵的对角线元素为1.0，也会被包含在均值计算中
    - 如果窗口内的数据导致相关性矩阵无法计算（如常数列），对应位置返回NaN
    - 使用并行计算优化性能，可通过max_workers参数控制线程数
    - 自动处理NaN值，只使用有效的数据点进行计算
    
    性能特点：
    --------
    - 使用Rust实现，性能比纯Python版本快数倍到数十倍
    - 支持可配置的并行计算，充分利用多核CPU
    - 内存优化的滑动窗口算法，避免重复计算
    - 数值稳定的算法实现，使用Kahan求和提高精度
    
    示例：
    -----
    >>> import numpy as np
    >>> from rust_pyfunc import rolling_correlation_mean
    >>> 
    >>> # 创建测试数据
    >>> data = np.random.randn(1000, 20).astype(np.float64)
    >>> window_size = 50
    >>> 
    >>> # 计算滚动相关性矩阵均值
    >>> result = rolling_correlation_mean(data, window_size)
    >>> print(f"结果形状: {result.shape}")  # (1000,)
    >>> print(f"前50个值为NaN: {np.isnan(result[:49]).all()}")  # True
    >>> print(f"第50个值开始有效: {not np.isnan(result[49])}")  # True
    """
    ...

def rolling_correlation_skew(
    data: NDArray[np.float64],
    window_size: int,
    min_periods: Optional[int] = None,
    max_workers: int = 10
) -> NDArray[np.float64]:
    """滚动窗口计算相关性矩阵偏度。
    
    对于输入数据的每一行，计算其过去window_size行的相关性矩阵，
    然后计算该相关性矩阵中所有值的偏度（skewness）。
    偏度衡量相关性分布的不对称性。
    
    参数说明：
    ----------
    data : NDArray[np.float64]
        输入数据矩阵，形状为(n_samples, n_features)，每列代表一个变量
    window_size : int
        滚动窗口的大小，表示向前取多少行数据计算相关性矩阵
    min_periods : Optional[int], default=None
        计算相关性所需的最小样本数，默认为window_size
    max_workers : int, default=10
        最大并行工作线程数，默认为10，设置为0则使用所有可用核心
        
    返回值：
    -------
    NDArray[np.float64]
        一维数组，长度等于输入数据的行数，每个值为对应行的相关性矩阵偏度
        
    偏度解释：
    --------
    - 偏度 > 0：右偏（正偏），大部分相关性值较小，少数值较大
        表示市场中大多数资产相关性较低，但有少数资产高度相关
    - 偏度 < 0：左偏（负偏），大部分相关性值较大，少数值较小
        表示市场中大多数资产高度相关，但有少数资产相关性较低
    - 偏度 ≈ 0：接近对称分布，相关性分布较为均匀
    
    注意：
    -----
    - 需要至少3个有效的相关性值才能计算偏度
    - 对于前window_size-1行，如果可用数据少于min_periods，则返回NaN
    - 使用Fisher-Pearson标准化的偏度系数，包含样本校正
    - 自动处理NaN值，只使用有效的数据点进行计算
    - 使用并行计算优化性能，可通过max_workers参数控制线程数
    
    性能特点：
    --------
    - 使用Rust实现，性能比纯Python版本快数倍到数十倍
    - 支持可配置的并行计算，充分利用多核CPU
    - 内存优化的滑动窗口算法，避免重复计算
    - 数值稳定的算法实现，使用Kahan求和提高精度
    
    应用场景：
    --------
    - 金融风险管理：监测市场相关性结构的不对称性
    - 投资组合优化：识别相关性分布的极端情况
    - 市场状态分析：通过相关性偏度判断市场集中度
    - 危机传染分析：检测系统性风险的形成模式
    
    示例：
    -----
    >>> import numpy as np
    >>> from rust_pyfunc import rolling_correlation_skew
    >>> 
    >>> # 创建测试数据
    >>> data = np.random.randn(1000, 20).astype(np.float64)
    >>> window_size = 50
    >>> 
    >>> # 计算滚动相关性矩阵偏度
    >>> skew_result = rolling_correlation_skew(data, window_size)
    >>> 
    >>> # 分析结果
    >>> valid_skews = skew_result[~np.isnan(skew_result)]
    >>> print(f"偏度范围: {np.min(valid_skews):.4f} ~ {np.max(valid_skews):.4f}")
    >>> print(f"平均偏度: {np.mean(valid_skews):.4f}")
    >>> 
    >>> # 识别极端偏度时期
    >>> high_skew_mask = skew_result > np.nanpercentile(skew_result, 95)
    >>> print(f"高偏度时期: {np.sum(high_skew_mask)}个时点")
    """
    ...

def rolling_window_core_feature(
    values: NDArray[np.float64],
    window_size: int = 5
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """滚动窗口核心特征提取。
    
    对输入序列进行滚动窗口分析，识别每个窗口中最重要的特征位置（核心特征）
    和最不重要的特征位置。通过计算窗口间的相关性并分析mask效应来确定特征重要性。
    
    算法原理：
    1. 对每个滚动窗口，计算其与所有其他窗口的相关系数（基准相关性）
    2. 依次将窗口内每个位置设为NaN，重新计算相关系数
    3. 相关性变化最小的位置为最重要特征（核心代表性）
    4. 相关性变化最大的位置为最不重要特征
    
    参数说明：
    ----------
    values : NDArray[np.float64]
        输入的一维数组，必须是float64类型
    window_size : int, default=5
        滚动窗口的大小，必须>=2且<=序列长度
    
    返回值：
    -------
    Tuple[NDArray[np.float64], NDArray[np.float64]]
        返回两个数组的元组：
        - 第一个数组：核心特征序列，每个位置对应该窗口中最重要的特征值
        - 第二个数组：次要特征序列，每个位置对应该窗口中最不重要的特征值
        两个数组的前(window_size-1)个位置均为NaN
    
    特征重要性解释：
    --------------
    - 核心特征：移除后对窗口相关性影响最小的元素，代表窗口的核心代表性
    - 次要特征：移除后对窗口相关性影响最大的元素，代表窗口中的噪声成分
    
    注意：
    -----
    - 窗口大小必须至少为2，且不能超过序列长度
    - 对于前(window_size-1)个位置，返回NaN
    - 自动处理NaN值，只使用有效的数据点进行计算
    - 使用优化的相关系数计算，保证数值稳定性
    
    性能特点：
    --------
    - 使用Rust实现，性能比纯Python版本快数倍到数十倍
    - 预计算优化：避免重复计算窗口统计信息
    - 向量化操作：使用SIMD优化的向量运算
    - 内存优化：重用缓冲区，减少动态内存分配
    - 目标性能：10万长度序列在1秒内完成计算
    
    应用场景：
    --------
    - 时间序列特征工程：识别序列中的关键信息点
    - 异常检测：通过核心特征识别正常模式
    - 信号处理：提取信号的代表性特征
    - 金融数据分析：识别价格序列的关键驱动因素
    - 降噪处理：通过核心特征过滤噪声
    
    示例：
    -----
    >>> import numpy as np
    >>> from rust_pyfunc import rolling_window_core_feature
    >>> 
    >>> # 创建测试数据
    >>> data = np.random.randn(1000).astype(np.float64)
    >>> 
    >>> # 使用默认窗口大小5
    >>> core_features, minor_features = rolling_window_core_feature(data)
    >>> 
    >>> # 使用自定义窗口大小10
    >>> core_features, minor_features = rolling_window_core_feature(data, window_size=10)
    >>> 
    >>> # 分析结果
    >>> print(f"核心特征前10个值: {core_features[:10]}")
    >>> print(f"次要特征前10个值: {minor_features[:10]}")
    >>> 
    >>> # 验证有效值数量
    >>> valid_core = ~np.isnan(core_features)
    >>> print(f"有效核心特征数量: {np.sum(valid_core)}")
    >>> 
    >>> # 分析特征分布
    >>> core_std = np.nanstd(core_features)
    >>> minor_std = np.nanstd(minor_features)
    >>> print(f"核心特征标准差: {core_std:.4f}")
    >>> print(f"次要特征标准差: {minor_std:.4f}")
    """
    ...

def rolling_window_core_feature_ultra(
    values: NDArray[np.float64],
    window_size: int = 5
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """超级轻量级优化版滚动窗口核心特征提取。
    
    基于性能测试结果，专注于最有效的优化：
    1. 极致的内联优化
    2. 最小化内存分配
    3. CPU缓存友好的数据访问模式
    4. 编译器友好的代码结构
    
    去掉所有复杂缓存机制，专注于算法核心优化。
    
    算法原理：
    1. 对每个滚动窗口，计算其与所有其他窗口的相关系数（基准相关性）
    2. 依次将窗口内每个位置设为NaN，重新计算相关系数
    3. 相关性变化最小的位置为最重要特征（核心代表性）
    4. 相关性变化最大的位置为最不重要特征
    
    参数说明：
    ----------
    values : NDArray[np.float64]
        输入的一维数组，必须是float64类型
    window_size : int, default=5
        滚动窗口的大小，必须>=2且<=序列长度
    
    返回值：
    -------
    Tuple[NDArray[np.float64], NDArray[np.float64]]
        返回两个数组的元组：
        - 第一个数组：核心特征序列，每个位置对应该窗口中最重要的特征值
        - 第二个数组：次要特征序列，每个位置对应该窗口中最不重要的特征值
        两个数组的前(window_size-1)个位置均为NaN
    
    特征重要性解释：
    --------------
    - 核心特征：移除后对窗口相关性影响最小的元素，代表窗口的核心代表性
    - 次要特征：移除后对窗口相关性影响最大的元素，代表窗口中的噪声成分
    
    性能优化：
    --------
    - 使用unsafe内存访问，最大化访问速度
    - 激进的内联优化，减少函数调用开销
    - 最小化内存分配，重用缓冲区
    - CPU缓存友好的数据访问模式
    - 快速NaN检查，使用 xi == xi 模式
    - 向量预分配，避免运行时内存分配
    
    注意：
    -----
    - 此版本移除了复杂缓存机制，专注于算法核心性能
    - 通过简化设计获得更好的性能表现
    - 保证结果与原始版本完全一致
    - 适用于对性能要求极高的场景
    
    示例：
    -----
    >>> import numpy as np
    >>> from rust_pyfunc import rolling_window_core_feature_ultra
    >>> 
    >>> # 创建测试数据
    >>> data = np.random.randn(100000).astype(np.float64)
    >>> 
    >>> # 使用ultra版本进行高性能计算
    >>> core_features, minor_features = rolling_window_core_feature_ultra(data)
    >>> 
    >>> # 验证性能和结果一致性
    >>> print(f"数据长度: {len(data)}")
    >>> print(f"有效结果数量: {np.sum(~np.isnan(core_features))}")
    """
    ...

class HMMPredictionResult:
    """HMM趋势预测结果类。
    
    包含隐马尔科夫模型预测的所有结果，包括状态预测概率、更新后的状态概率、
    发射概率矩阵和转移概率矩阵的时间序列。
    """
    state_predictions: List[List[float]]
    """每步的状态预测概率序列，形状为(n_steps, 3)，对应[下跌, 震荡, 上涨]三种状态的概率"""
    
    updated_state_probs: List[List[float]]
    """每步更新后的状态概率序列，形状为(n_steps, 3)，基于真实观测更新后的状态概率"""
    
    emission_probs: List[List[List[float]]]
    """发射概率矩阵的时间序列，形状为(n_steps, 3, 3)，表示每个状态发射每种观测的概率"""
    
    transition_probs: List[List[List[float]]]
    """状态转移概率矩阵的时间序列，形状为(n_steps, 3, 3)，表示状态间转移概率"""
    
    def __repr__(self) -> str:
        """返回对象的字符串表示"""
        ...

def hmm_trend_prediction(
    prices: List[float],
    window: int = 30,
    slope_threshold: float = 0.0005,
    r2_threshold: float = 0.5,
    learning_rate: float = 0.1
) -> HMMPredictionResult:
    """基于线性回归趋势判断的隐马尔科夫模型价格预测。
    
    这是一个创新的预测方法，结合了线性回归的趋势判断能力和隐马尔科夫模型的状态建模能力。
    与传统的无监督HMM不同，此方法通过线性回归直接学习状态概率，无需迭代拟合。
    
    算法流程：
    1. 使用前10%的数据进行线性回归趋势分析，判断每个时刻的状态（上涨/下跌/震荡）
    2. 基于初始状态序列计算状态转移概率矩阵和发射概率矩阵
    3. 对剩余90%的数据逐步进行预测和模型参数在线更新：
       - 使用当前状态概率和转移矩阵预测下一时刻的状态概率
       - 观察真实价格变化，使用贝叶斯更新调整状态概率
       - 通过在线学习更新转移概率矩阵和发射概率矩阵
    
    状态定义：
    - 状态-1：下跌趋势（斜率 < -slope_threshold）
    - 状态 0：震荡趋势（|斜率| <= slope_threshold 或 R² < r2_threshold）
    - 状态 1：上涨趋势（斜率 > slope_threshold）
    
    参数说明：
    ----------
    prices : List[float]
        价格序列，由pandas.Series.to_numpy(float)转换而来的一维数组
    window : int, default=30
        线性回归的滑动窗口大小，用于趋势判断
    slope_threshold : float, default=0.0005
        斜率阈值，判断是否有明确趋势的临界值
    r2_threshold : float, default=0.5
        R²阈值，拟合优度低于此值视为震荡状态
    learning_rate : float, default=0.1
        在线学习率，控制模型参数更新的速度
    
    返回值：
    -------
    HMMPredictionResult
        包含以下字段的预测结果对象：
        - state_predictions：每步对三种状态的预测概率 [下跌, 震荡, 上涨]
        - updated_state_probs：每步结合真实观测后的状态概率更新
        - emission_probs：每步的发射概率矩阵演化 [状态][观测]
        - transition_probs：每步的状态转移概率矩阵演化
    
    异常：
    -----
    PyValueError
        当价格序列长度太短，无法进行有效的HMM分析时抛出
    
    注意：
    -----
    - 此方法不同于传统的无监督HMM，它通过线性回归直接学习状态，更适合金融时间序列
    - 模型参数会随着新数据不断在线更新，具有自适应能力
    - 价格序列会被转换为对数收益率进行分析，斜率直接表示收益率
    - 使用拉普拉斯平滑避免概率为0的情况
    - 适合处理非均匀时间间隔的交易数据
    
    性能特点：
    --------
    - 使用Rust实现，计算性能优异
    - 在线学习机制，适合实时预测场景
    - 数值稳定的算法实现，避免概率计算中的数值问题
    
    应用场景：
    --------
    - 股票价格趋势预测
    - 交易策略信号生成
    - 市场状态识别和转换分析
    - 风险管理中的趋势监控
    
    示例：
    -----
    >>> import numpy as np
    >>> import pandas as pd
    >>> from rust_pyfunc import hmm_trend_prediction
    >>> 
    >>> # 创建模拟价格序列
    >>> np.random.seed(42)
    >>> prices = np.cumsum(np.random.randn(1000) * 0.01) + 100.0
    >>> price_list = prices.tolist()
    >>> 
    >>> # 进行HMM趋势预测
    >>> result = hmm_trend_prediction(
    ...     prices=price_list,
    ...     window=30,
    ...     slope_threshold=0.001,
    ...     r2_threshold=0.6,
    ...     learning_rate=0.1
    ... )
    >>> 
    >>> # 查看预测结果
    >>> print(f"预测步数: {len(result.state_predictions)}")
    >>> print(f"首个预测: {result.state_predictions[0]}")  # [下跌概率, 震荡概率, 上涨概率]
    >>> 
    >>> # 分析状态转移概率的演化
    >>> final_transition = result.transition_probs[-1]
    >>> print("最终状态转移矩阵:")
    >>> for i, row in enumerate(final_transition):
    ...     print(f"状态{i-1}: {[f'{p:.3f}' for p in row]}")
    >>> 
    >>> # 转换为DataFrame便于分析
    >>> predictions_df = pd.DataFrame(
    ...     result.state_predictions,
    ...     columns=['下跌概率', '震荡概率', '上涨概率']
    ... )
    >>> updated_states_df = pd.DataFrame(
    ...     result.updated_state_probs,
    ...     columns=['更新后下跌概率', '更新后震荡概率', '更新后上涨概率']
    ... )
    """
    ...

def distances_to_frontier(
    r: NDArray[np.float64],
    group_size: int,
    drop_last: bool = True,
    ddof: int = 1,
    ridge: float = 1e-6
) -> NDArray[np.float64]:
    """计算收益序列中每个聚合块到马科维茨有效前沿的距离。

    基于马科维茨投资组合理论的有效前沿距离计算功能。给定单日3秒频率收益序列，
    按指定块大小聚合后计算每个资产点到有效前沿的最短距离。

    算法步骤：
    1. 数据分块聚合：将收益序列按指定大小分块，计算每块均值
    2. 协方差矩阵计算：计算块间样本协方差矩阵（带岭化保证正定性）
    3. 有效前沿构造：使用马科维茨无约束闭式解构造有效前沿
    4. 距离计算：使用KKT-λ四次方程法计算每个资产点到前沿的最短欧氏距离

    参数说明：
    ----------
    r : NDArray[np.float64]
        1D float64数组，单日3秒频率收益序列
    group_size : int
        每多少行聚合成一块（x），必须大于0
    drop_last : bool, default=True
        尾部不足group_size行时是否丢弃，True丢弃，False则报错
    ddof : int, default=1
        协方差/方差的自由度调整，0或1，默认1（样本协方差）
    ridge : float, default=1e-6
        岭化强度系数，用于保证协方差矩阵正定

    返回值：
    -------
    NDArray[np.float64]
        shape=(m,)的1D数组，包含每个资产点到有效前沿的距离
        其中m = floor(len(r) / group_size)（如果drop_last=True）

    异常：
    -----
    ValueError
        当输入参数无效时抛出：
        - group_size <= 0
        - 输入序列为空
        - drop_last=False且序列长度不能被group_size整除
        - 块大小 <= 自由度调整
        - 协方差矩阵不正定（可尝试增大ridge）
        - 有效前沿参数计算失败（Δ <= 0）

    数值提示：
    --------
    - 当 m >> group_size 时，协方差矩阵可能秩亏，需要通过增大ridge参数保证可逆性
    - 如果出现数值不稳定错误，建议将ridge增大10倍或100倍
    - 默认使用样本协方差（ddof=1），符合统计学习习惯

    性能特点：
    --------
    - 使用Rust实现，计算性能优异
    - 采用Cholesky分解避免显式矩阵求逆，数值稳定性好
    - 支持大规模数据处理，内存使用优化
    - 多项式求根采用高效算法，避免数值迭代

    应用场景：
    --------
    - 投资组合绩效评估：评估各时间段表现相对有效前沿的距离
    - 市场效率分析：通过距离分布判断市场效率变化
    - 风险管理：识别偏离有效前沿的异常时期
    - 资产配置优化：为动态调整提供量化依据

    示例：
    -----
    >>> import numpy as np
    >>> from rust_pyfunc import distances_to_frontier
    >>>
    >>> # 生成测试数据
    >>> np.random.seed(0)
    >>> r = 1e-4 * np.random.randn(4800).astype(np.float64)
    >>>
    >>> # 每1分钟聚合（20个3秒间隔）
    >>> distances = distances_to_frontier(r, group_size=20)
    >>> print(f"距离数组形状: {distances.shape}")  # (240,)
    >>> print(f"平均距离: {np.mean(distances):.6e}")
    >>>
    >>> # 每2分半聚合（50个3秒间隔）
    >>> distances2 = distances_to_frontier(r, group_size=50)
    >>> print(f"距离数组形状: {distances2.shape}")  # (96,)
    >>>
    >>> # 增大岭化系数处理病态数据
    >>> distances3 = distances_to_frontier(r, group_size=100, ridge=1e-4)
    >>> print(f"距离数组形状: {distances3.shape}")  # (48,)
    >>>
    >>> # 分析距离分布
    >>> import matplotlib.pyplot as plt
    >>> plt.hist(distances, bins=30, alpha=0.7)
    >>> plt.xlabel('到有效前沿的距离')
    >>> plt.ylabel('频次')
    >>> plt.title('距离分布直方图')
    >>> plt.show()

    注意：
    -----
    - 函数保证返回的距离值非负且有限
    - 在极少数情况下如果多项式求根失败，对应距离会设为0并发出警告
    - 所有计算都使用双精度浮点数，确保数值精度
    """
    ...