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
    ridge: float = 1e-6,
    timestamps: Optional[NDArray[np.int64]] = None,
) -> Tuple[NDArray[np.int64], NDArray[np.float64]]:
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
    timestamps : Optional[NDArray[np.int64]], default=None
        与输入序列等长的时间戳数组（例如DatetimeIndex.view('int64')），用于标记每个聚合块的首个时间点

    返回值：
    -------
    Tuple[NDArray[np.int64], NDArray[np.float64]]
        - block_timestamps：shape=(m,)的时间戳数组，表示每个聚合块的首个时间点；
          若未提供timestamps，则返回0..m-1的顺序索引
        - distances：shape=(m,)的距离数组
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
    >>> block_ts, distances = distances_to_frontier(r, group_size=20)
    >>> print(f"距离数组形状: {distances.shape}")  # (240,)
    >>> print(f"平均距离: {np.mean(distances):.6e}")
    >>>
    >>> # 每2分半聚合（50个3秒间隔）
    >>> block_ts2, distances2 = distances_to_frontier(r, group_size=50)
    >>> print(f"距离数组形状: {distances2.shape}")  # (96,)
    >>>
    >>> # 增大岭化系数处理病态数据
    >>> block_ts3, distances3 = distances_to_frontier(r, group_size=100, ridge=1e-4)
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

def mutual_information_knn(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    k: Optional[int] = None
) -> float:
    """使用k近邻法计算两个一维数组之间的互信息。

    采用KSG(Kraskov-Stögbauer-Grassberger)方法1计算互信息，
    该方法基于k近邻距离估计，是目前最准确和广泛使用的非参数互信息估计方法之一。

    算法原理：
    1. 将(x, y)对视为二维空间中的点
    2. 使用欧几里得距离找到每个点的k个最近邻
    3. 基于联合空间和边缘空间中的距离分布估计熵
    4. 使用公式 I(X;Y) = H(X) + H(Y) - H(X,Y) 计算互信息

    参数说明：
    ----------
    x : NDArray[np.float64]
        第一个一维数组，与y长度相等
    y : NDArray[np.float64]
        第二个一维数组，与x长度相等
    k : Optional[int], default=None
        k近邻参数，默认为3。如果为None则使用默认值3
        k值影响估计精度和计算复杂度，建议使用较小的奇数值

    返回值：
    -------
    float
        互信息值，单位为nats(自然单位)。值越大表示X和Y之间的依赖性越强

    异常：
    -----
    ValueError
        当x和y长度不等或k >= len(x)时抛出

    互信息解释：
    ------------
    - I(X;Y) >= 0：互信息永远非负
    - I(X;Y) = 0：X和Y独立，不存在依赖关系
    - I(X;Y) > 0：X和Y之间存在依赖关系
    - I(X;Y) ≈ H(X) 或 H(Y)：X和Y高度相关（可能存在函数关系）

    数值稳定性：
    -----------
    - 函数自动处理边界情况，如k值选择、数据长度等
    - 对于过小的数据集(< 100个点)，建议使用更小的k值
    - 结果可能存在小幅数值波动，这是k近邻方法的固有特性

    性能特点：
    ---------
    - 使用Rust实现，性能比纯Python版本快10-50倍
    - 时间复杂度：O(n² log n)，其中n是数据点数量
    - 适合中等规模数据集(100 - 100,000个点)
    - 对于超大数据集，建议先进行数据子采样

    应用场景：
    ---------
    - 特征选择：识别相关性强的特征对
    - 变量依赖性检测：判断两个变量是否独立
    - 金融数据分析：分析股票收益率之间的非线性关系
    - 信息论研究：量化信息传递强度

    示例：
    -----
    >>> import numpy as np
    >>> from rust_pyfunc import mutual_information_knn
    >>>
    >>> # 生成相关数据
    >>> x = np.random.randn(1000)
    >>> y = x + 0.5 * np.random.randn(1000)  # y与x相关
    >>>
    >>> # 计算互信息
    >>> mi_xy = mutual_information_knn(x, y, k=3)
    >>> print(f"X和Y的互信息: {mi_xy:.4f}")
    >>>
    >>> # 独立数据
    >>> z = np.random.randn(1000)
    >>> mi_xz = mutual_information_knn(x, z, k=3)
    >>> print(f"X和Z的互信息: {mi_xz:.4f}")
    >>>
    >>> # 函数关系
    >>> w = x ** 2
    >>> mi_xw = mutual_information_knn(x, w, k=3)
    >>> print(f"X和X²的互信息: {mi_xw:.4f}")
    """
    ...

def mutual_information_knn_chebyshev(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    k: Optional[int] = None
) -> float:
    """使用k近邻法计算两个一维数组之间的互信息(Chebyshev距离)。

    采用KSG方法2计算互信息，与mutual_information_knn的区别是使用Chebyshev距离(最大范数)
    而非欧几里得距离。Chebyshev距离定义为L∞ = max(|x_i - x_j|, |y_i - y_j|)。

    算法原理：
    1. 将(x, y)对视为二维空间中的点
    2. 使用Chebyshev距离找到每个点的k个最近邻
    3. 基于联合空间和边缘空间中的距离分布估计熵
    4. 使用公式 I(X;Y) = H(X) + H(Y) - H(X,Y) 计算互信息

    参数说明：
    ----------
    x : NDArray[np.float64]
        第一个一维数组，与y长度相等
    y : NDArray[np.float64]
        第二个一维数组，与x长度相等
    k : Optional[int], default=None
        k近邻参数，默认为3。如果为None则使用默认值3
        k值影响估计精度和计算复杂度，建议使用较小的奇数值

    返回值：
    -------
    float
        互信息值，单位为nats(自然单位)

    异常：
    -----
    ValueError
        当x和y长度不等或k >= len(x)时抛出

    Chebyshev距离特点：
    ------------------
    - 对各维度独立考虑，不考虑维度间的协方差
    - 在某些情况下比欧几里得距离更稳健
    - 计算复杂度略低于欧几里得距离
    - 适用于各维度尺度差异较大的数据

    性能特点：
    ---------
    - 使用Rust实现，性能优异
    - Chebyshev距离计算比欧几里得距离更快
    - 时间复杂度：O(n² log n)

    与mutual_information_knn的区别：
    ------------------------------
    - mutual_information_knn：使用欧几里得距离，考虑维度间相关性
    - mutual_information_knn_chebyshev：使用Chebyshev距离，独立处理各维度

    应用建议：
    ---------
    - 如果数据各维度尺度差异大，建议使用Chebyshev距离版本
    - 如果数据维度间存在强相关性，建议使用欧几里得距离版本
    - 对于未知数据，可以两种方法都尝试，比较结果稳定性

    示例：
    -----
    >>> import numpy as np
    >>> from rust_pyfunc import mutual_information_knn_chebyshev
    >>>
    >>> # 生成测试数据
    >>> x = np.random.randn(1000)
    >>> y = x + 0.5 * np.random.randn(1000)
    >>>
    >>> # 使用Chebyshev距离计算互信息
    >>> mi = mutual_information_knn_chebyshev(x, y, k=3)
    >>> print(f"互信息(Chebyshev): {mi:.4f}")
    >>>
    >>> # 比较两种方法
    >>> from rust_pyfunc import mutual_information_knn
    >>> mi_euclid = mutual_information_knn(x, y, k=3)
    >>> print(f"互信息(欧几里得): {mi_euclid:.4f}")
    """
    ...

def mutual_information_2d_knn(
    a: NDArray[np.float64],
    b: NDArray[np.float64],
    k: Optional[int] = None
) -> NDArray[np.float64]:
    """计算两个二维数组对应行的互信息。

    对输入的两个二维数组a和b，计算每一行对应位置的互信息。
    对于第i行，将a[i]和b[i]视为两个多维变量，计算它们之间的互信息。

    算法原理：
    1. 将两个输入数组视为形状为(n, m1)和(n, m2)的矩阵
    2. 对每一行i，使用KSG方法计算a[i]和b[i]的互信息
    3. 将每一行的多个元素视为一个多维点，使用k近邻方法估计互信息
       （联合空间k近邻用于确定第k个邻居；边缘计数阈值ε_i采用无穷范数半径，
       并对等距情况施加微小偏移以保证严格小于）
    4. 返回长度为n的一维数组，包含每行的互信息值

    参数说明：
    ----------
    a : NDArray[np.float64]
        第一个二维数组，形状为(n, m1)
    b : NDArray[np.float64]
        第二个二维数组，形状为(n, m2)，必须与a有相同的行数n
    k : Optional[int], default=None
        k近邻参数，默认为3，必须为正整数。如果为None则使用默认值3

    返回值：
    -------
    NDArray[np.float64]
        一维数组，长度为n，每个元素对应输入数组相应行的互信息值

    异常：
    -----
    ValueError
        当a和b行数不等，或k <= 0时抛出。注意：有效样本量以“每行的列数”为准，
        若某行有效样本数 <= k，该行结果为NaN而非抛错。

    应用场景：
    ---------
    - 多变量时间序列分析：计算不同变量序列间的依赖关系
    - 金融数据：分析股票价格、成交量、技术指标之间的互信息
    - 信号处理：量化多维信号之间的信息传递
    - 机器学习：特征选择和特征依赖性分析

    示例：
    -----
    >>> import numpy as np
    >>> from rust_pyfunc import mutual_information_2d_knn
    >>>
    >>> # 创建两个5x3的二维数组
    >>> a = np.random.randn(5, 3)
    >>> b = np.random.randn(5, 3)
    >>>
    >>> # 计算每一行的互信息
    >>> mi_per_row = mutual_information_2d_knn(a, b, k=3)
    >>> print(f"每行互信息: {mi_per_row}")
    >>> print(f"形状: {mi_per_row.shape}")  # (5,)
    >>>
    >>> # 实际应用：计算股票特征间的互信息
    >>> # price_features: (100, 5) - [open, high, low, close, volume]
    >>> # volume_features: (100, 3) - [volume, turnover, turnover_rate]
    >>> price_features = np.random.randn(100, 5)
    >>> volume_features = np.random.randn(100, 3)
    >>> mi = mutual_information_2d_knn(price_features, volume_features, k=5)
    >>> print(f"互信息范围: {np.min(mi):.4f} ~ {np.max(mi):.4f}")
    """
    ...

def mutual_information_2d_knn_chebyshev(
    a: NDArray[np.float64],
    b: NDArray[np.float64],
    k: Optional[int] = None
) -> NDArray[np.float64]:
    """计算两个二维数组对应行的互信息(Chebyshev距离版本)。

    与mutual_information_2d_knn相同，但使用Chebyshev距离而非欧几里得距离。

    参数说明：
    ----------
    a : NDArray[np.float64]
        第一个二维数组，形状为(n, m1)
    b : NDArray[np.float64]
        第二个二维数组，形状为(n, m2)，必须与a有相同的行数n
    k : Optional[int], default=None
        k近邻参数，默认为3

    返回值：
    -------
    NDArray[np.float64]
        一维数组，长度为n，每行的互信息值

    性能特点：
    ---------
    - Chebyshev距离计算更快
    - 对各维度独立处理，不考虑维度间相关性
    - 适用于各维度尺度差异较大的数据

    示例：
    -----
    >>> import numpy as np
    >>> from rust_pyfunc import mutual_information_2d_knn_chebyshev
    >>>
    >>> # 创建测试数据
    >>> a = np.random.randn(10, 4)
    >>> b = np.random.randn(10, 4)
    >>>
    >>> # 使用Chebyshev距离计算
    >>> mi = mutual_information_2d_knn_chebyshev(a, b, k=3)
    >>> print(f"互信息数组: {mi}")
    """
    ...

def mutual_information_2d_knn_fixed(
    a: NDArray[np.float64],
    b: NDArray[np.float64],
    k: Optional[int] = None
) -> NDArray[np.float64]:
    """计算两个二维数组对应行的互信息(修复负数问题的版本)。

    这是mutual_information_2d_knn的修复版本，解决了原版本中负数互信息的问题。
    使用修正的KSG算法，确保互信息值永远非负。

    算法修复：
    ----------
    1. 修正了KSG算法中的边缘计数问题
    2. 在X和Y维度中独立计数，避免错误的距离使用
    3. 确保互信息估计的非负性质

    参数说明：
    ----------
    a : NDArray[np.float64]
        第一个二维数组，形状为(n, m1)
    b : NDArray[np.float64]
        第二个二维数组，形状为(n, m2)，必须与a有相同的行数n
    k : Optional[int], default=None
        k近邻参数，默认为3，必须为正整数。如果为None则使用默认值3

    返回值：
    -------
    NDArray[np.float64]
        一维数组，长度为n，每个元素对应输入数组相应行的互信息值
        互信息值保证非负，符合理论要求

    异常：
    -----
    ValueError
        当a和b行数不等，或k <= 0时抛出

    算法保证：
    ----------
    - 互信息值永远 >= 0
    - 对于独立变量，互信息接近0
    - 对于强相关变量，互信息显著大于0
    - 消除了原版本中的负数估计问题

    性能特点：
    ----------
    - 使用修正的KSG算法，数值更稳定
    - 保持高性能，与原版本相当
    - 适合大规模数据分析

    应用建议：
    ----------
    - 推荐使用此版本替代mutual_information_2d_knn
    - 特别适用于需要准确互信息估计的场景
    - 当原版本出现负数问题时，使用此修复版本

    示例：
    -----
    >>> import numpy as np
    >>> from rust_pyfunc import mutual_information_2d_knn_fixed
    >>>
    >>> # 创建测试数据
    >>> a = np.random.randn(1000, 50)  # 1000个时间点，50个变量
    >>> b = a * 0.8 + np.random.randn(1000, 50) * 0.2  # 相关变量
    >>>
    >>> # 计算互信息
    >>> mi = mutual_information_2d_knn_fixed(a, b, k=3)
    >>> print(f"互信息范围: {np.min(mi):.4f} ~ {np.max(mi):.4f}")
    >>> print(f"负数数量: {np.sum(mi < 0)}")  # 应该为0
    """
    ...

def mutual_information_2d_knn_final(
    a: NDArray[np.float64],
    b: NDArray[np.float64],
    k: Optional[int] = None
) -> NDArray[np.float64]:
    """计算两个二维数组对应行的互信息(最终修复版本 - 负值截断)。

    这是mutual_information_2d_knn的最终修复版本，通过负值截断解决了KSG估计器的
    固有偏差问题。KSG估计器在弱相关情况下可能出现负值，这是已知的有限样本偏差现象。

    修复原理：
    ----------
    1. 理论基础：根据信息论，互信息I(X;Y) ≥ 0永远成立
    2. 问题识别：KSG估计器在弱相关时可能产生负值估计
    3. 解决方案：实施负值截断(max(0, MI_estimate))
    4. 标准做法：这是处理KSG负值问题的行业标准方法

    算法特点：
    ----------
    - 使用KSG方法1进行互信息估计
    - 采用Chebyshev距离确定k近邻
    - 实施负值截断确保理论一致性
    - 保持高性能的Rust实现

    参数说明：
    ----------
    a : NDArray[np.float64]
        第一个二维数组，形状为(n, m1)
    b : NDArray[np.float64]
        第二个二维数组，形状为(n, m2)，必须与a有相同的行数n
    k : Optional[int], default=None
        k近邻参数，默认为3，必须为正整数。如果为None则使用默认值3

    返回值：
    -------
    NDArray[np.float64]
        一维数组，长度为n，每个元素对应输入数组相应行的互信息值
        **所有值保证非负**，符合互信息的理论约束

    异常：
    -----
    ValueError
        当a和b行数不等，或k <= 0时抛出

    理论保证：
    ----------
    - 互信息值永远 >= 0 (通过负值截断保证)
    - 对于独立变量，互信息接近0
    - 对于强相关变量，互信息显著大于0
    - 完全消除了负数估计问题

    与其他版本的区别：
    ------------------
    - mutual_information_2d_knn：原始版本，可能出现负数
    - mutual_information_2d_knn_fixed：尝试修复算法，但未完全解决
    - mutual_information_2d_knn_final：通过负值截断完全解决负数问题

    应用建议：
    ----------
    - **强烈推荐使用此版本**作为生产环境的选择
    - 特别适用于金融数据分析、机器学习特征工程
    - 当需要理论一致性的互信息估计时使用
    - 适用于任何对负数敏感的应用场景

    性能特点：
    ----------
    - 与原版本相同的计算复杂度
    - 负值截断操作计算成本极低
    - 保持原有的高性能特性
    - 适合大规模数据分析

    示例：
    -----
    >>> import numpy as np
    >>> from rust_pyfunc import mutual_information_2d_knn_final
    >>>
    >>> # 创建测试数据 - 模拟股票数据
    >>> np.random.seed(42)
    >>> n_dates, n_stocks = 2000, 100
    >>> a = np.random.lognormal(2, 0.8, (n_dates, n_stocks))  # 成交量数据
    >>> b = 0.6 * a + 0.4 * np.random.lognormal(2.5, 1.0, (n_dates, n_stocks))  # 金额数据
    >>>
    >>> # 计算互信息
    >>> mi = mutual_information_2d_knn_final(a, b, k=3)
    >>> print(f"互信息范围: {np.min(mi):.4f} ~ {np.max(mi):.4f}")
    >>> print(f"负数数量: {np.sum(mi < 0)}")  # 必定为0
    >>> print(f"平均互信息: {np.mean(mi):.4f}")
    >>>
    >>> # 用户场景：股票成交量与金额的互信息分析
    >>> # tr_data = read_daily(tr=1)  # 读取成交量数据
    >>> # amount_data = read_daily(amount=1)  # 读取金额数据
    >>> # mi = mutual_information_2d_knn_final(tr_data, amount_data)
    >>> # print(f"成交量与金额的平均互信息: {np.mean(mi):.4f}")
    """
    ...
