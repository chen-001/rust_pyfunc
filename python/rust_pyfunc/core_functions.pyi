"""核心数学和统计函数类型声明"""
from typing import List, Optional, Union, Tuple
import numpy as np
from numpy.typing import NDArray

# GP 相关维数计算相关类型声明
class GpOptions:
    """GP 相关维数计算选项"""
    def __init__(
        self,
        ami_max_lag: int = 200,
        ami_n_bins: int = 32,
        ami_quantile_bins: bool = True,
        tau_override: Optional[int] = None,
        fnn_m_max: int = 12,
        fnn_rtol: float = 10.0,
        fnn_atol: float = 2.0,
        fnn_threshold: float = 0.02,
        m_override: Optional[int] = None,
        theiler_override: Optional[int] = None,
        n_r: int = 48,
        r_percentile_lo: float = 5.0,
        r_percentile_hi: float = 90.0,
        fit_min_len: int = 6,
        fit_max_len: int = 14,
        c_lo: float = 1e-4,
        c_hi: float = 0.999,
        stability_alpha: float = 0.05,
    ):
        ...

class GpResult:
    """GP 相关维数计算结果"""
    tau: int
    m: int
    theiler: int
    rs: List[float]
    cs: List[float]
    log_r: List[float]
    log_c: List[float]
    local_slopes: List[Optional[float]]
    fit_start: int
    fit_end: int
    d2_est: float
    fit_intercept: float
    fit_r2: float
    ami_lags: List[int]
    ami_values: List[float]
    fnn_ms: List[int]
    fnn_ratios: List[float]

def gp_correlation_dimension_auto(x: NDArray[np.float64]) -> GpResult:
    """零参数入口：只需传入序列，所有参数自动确定

    使用完全确定性的 Grassberger-Procaccia 算法计算时间序列的相关维数 (D₂)。
    所有中间参数（τ、m、Theiler窗口、半径网格与拟合区间）均由库内部自动确定。

    参数说明：
    ----------
    x : numpy.ndarray
        一维实数序列（函数内部会标准化：减均值/除标准差）

    返回值：
    -------
    GpResult
        包含相关维数估计和详细诊断信息的结果对象

    异常：
    ------
    ValueError
        当输入序列过短、过于恒定或其他数值计算问题时抛出
    """
    ...

def gp_correlation_dimension(x: Union[NDArray[np.float64], List[float]], opts: Optional[GpOptions] = None) -> GpResult:
    """可选参数入口：可提供自定义选项

    参数说明：
    ----------
    x : numpy.ndarray 或 list
        一维实数序列
    opts : GpOptions, 可选
        自定义计算选项，如果为 None 则使用默认值

    返回值：
    -------
    GpResult
        包含相关维数估计和详细诊断信息的结果对象
    """
    ...

def gp_create_default_options() -> GpOptions:
    """创建默认的 GP 计算选项

    返回值：
    -------
    GpOptions
        包含所有默认参数的选项对象
    """
    ...

def gp_create_options(
    ami_max_lag: int = 200,
    ami_n_bins: int = 32,
    ami_quantile_bins: bool = True,
    tau_override: Optional[int] = None,
    fnn_m_max: int = 12,
    fnn_rtol: float = 10.0,
    fnn_atol: float = 2.0,
    fnn_threshold: float = 0.02,
    m_override: Optional[int] = None,
    theiler_override: Optional[int] = None,
    n_r: int = 48,
    r_percentile_lo: float = 5.0,
    r_percentile_hi: float = 90.0,
    fit_min_len: int = 6,
    fit_max_len: int = 14,
    c_lo: float = 1e-4,
    c_hi: float = 0.999,
    stability_alpha: float = 0.05,
) -> GpOptions:
    """创建自定义的 GP 计算选项

    参数说明：
    ----------
    ami_max_lag : int, 默认 200
        AMI 计算的最大 lag
    ami_n_bins : int, 默认 32
        AMI 计算的分箱数
    ami_quantile_bins : bool, 默认 True
        是否使用分位数分箱（确定性）
    tau_override : int, 可选
        强制指定 τ 值，如不指定则自动选择
    fnn_m_max : int, 默认 12
        FNN 计算的最大 m 值
    fnn_rtol : float, 默认 10.0
        FNN 相对容差
    fnn_atol : float, 默认 2.0
        FNN 绝对容差
    fnn_threshold : float, 默认 0.02
        FNN 假近邻比例阈值
    m_override : int, 可选
        强制指定 m 值，如不指定则自动选择
    theiler_override : int, 可选
        强制指定 Theiler 窗口，如不指定则自动选择
    n_r : int, 默认 48
        半径网格点数
    r_percentile_lo : float, 默认 5.0
        半径范围下限百分位数
    r_percentile_hi : float, 默认 90.0
        半径范围上限百分位数
    fit_min_len : int, 默认 6
        线性拟合最小长度
    fit_max_len : int, 默认 14
        线性拟合最大长度
    c_lo : float, 默认 1e-4
        相关和下限（排除极小值）
    c_hi : float, 默认 0.999
        相关和上限（排除饱和值）
    stability_alpha : float, 默认 0.05
        稳定性评分权重（R² - α*斜率标准差）

    返回值：
    -------
    GpOptions
        自定义配置的选项对象
    """
    ...

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

def trend_2d(arr: NDArray[np.float64], axis: int) -> List[float]:
    """计算二维数组各行或各列的趋势性。
    
    参数说明：
    ----------
    arr : numpy.ndarray
        二维数组，必须是float64类型
    axis : int
        计算轴，0表示对每列计算趋势，1表示对每行计算趋势
    
    返回值：
    -------
    List[float]
        一维列表，包含每行或每列的趋势值
    
    示例：
    -----
    >>> import numpy as np
    >>> from rust_pyfunc import trend_2d
    >>> 
    >>> # 创建示例数据
    >>> data = np.array([[1.0, 2.0, 3.0, 4.0],
    ...                  [4.0, 3.0, 2.0, 1.0],
    ...                  [1.0, 3.0, 2.0, 4.0]])
    >>> 
    >>> # 计算每行的趋势
    >>> row_trends = trend_2d(data, axis=1)
    >>> 
    >>> # 计算每列的趋势
    >>> col_trends = trend_2d(data, axis=0)
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

def find_max_range_product(arr: List[float]) -> tuple[int, int, float]:
    """在数组中找到一对索引(x, y)，使得min(arr[x], arr[y]) * |x-y|的值最大。
    这个函数可以用来找到数组中距离最远的两个元素，同时考虑它们的最小值。

    参数说明：
    ----------
    arr : List[float]
        输入数组

    返回值：
    -------
    tuple
        返回一个元组(x, y, max_product)，其中x和y是使得乘积最大的索引对，max_product是最大乘积
    """
    ...

def ols(x: NDArray[np.float64], y: NDArray[np.float64], calculate_r2: bool = True) -> NDArray[np.float64]:
    """执行普通最小二乘法(OLS)回归分析。
    
    参数说明：
    ----------
    x : numpy.ndarray
        自变量数组，shape为(n,)或(n, m)
    y : numpy.ndarray  
        因变量数组，shape为(n,)
    calculate_r2 : bool
        是否计算R²值，默认True
        
    返回值：
    -------
    numpy.ndarray
        回归结果数组，包含[截距, 斜率, R²]或[截距, 斜率]
    """
    ...

def ols_predict(x: NDArray[np.float64], y: NDArray[np.float64], x_pred: NDArray[np.float64]) -> NDArray[np.float64]:
    """基于OLS回归模型进行预测。
    
    参数说明：
    ----------
    x : numpy.ndarray
        训练数据的自变量
    y : numpy.ndarray
        训练数据的因变量  
    x_pred : numpy.ndarray
        用于预测的自变量值
        
    返回值：
    -------
    numpy.ndarray
        预测值数组
    """
    ...

def ols_residuals(x: NDArray[np.float64], y: NDArray[np.float64]) -> NDArray[np.float64]:
    """计算OLS回归的残差。
    
    参数说明：
    ----------
    x : numpy.ndarray
        自变量数组
    y : numpy.ndarray
        因变量数组
        
    返回值：
    -------
    numpy.ndarray
        残差数组
    """
    ...

def max_range_loop(s: List[float], allow_equal: bool = False) -> List[int]:
    """找到数组中所有局部最大值的索引。
    
    参数说明：
    ----------
    s : List[float]
        输入数组
    allow_equal : bool
        是否允许相等值被认为是峰值
        
    返回值：
    -------
    List[int]
        局部最大值的索引列表
    """
    ...

def min_range_loop(s: List[float], allow_equal: bool = False) -> List[int]:
    """找到数组中所有局部最小值的索引。
    
    参数说明：
    ----------
    s : List[float]
        输入数组
    allow_equal : bool
        是否允许相等值被认为是谷值
        
    返回值：
    -------
    List[int]
        局部最小值的索引列表
    """
    ...

def rolling_volatility(arr: List[float], window: int) -> List[float]:
    """计算滚动波动率。
    
    参数说明：
    ----------
    arr : List[float]
        输入时间序列
    window : int
        滚动窗口大小
        
    返回值：
    -------
    List[float]
        滚动波动率序列
    """
    ...

def rolling_cv(arr: List[float], window: int) -> List[float]:
    """计算滚动变异系数。
    
    参数说明：
    ----------
    arr : List[float]
        输入时间序列
    window : int
        滚动窗口大小
        
    返回值：
    -------
    List[float]
        滚动变异系数序列
    """
    ...

def rolling_qcv(arr: List[float], window: int) -> List[float]:
    """计算滚动四分位变异系数。
    
    参数说明：
    ----------
    arr : List[float]
        输入时间序列
    window : int
        滚动窗口大小
        
    返回值：
    -------
    List[float]
        滚动四分位变异系数序列
    """
    ...

def compute_max_eigenvalue(matrix: NDArray[np.float64]) -> float:
    """计算矩阵的最大特征值。
    
    参数说明：
    ----------
    matrix : numpy.ndarray
        输入矩阵
        
    返回值：
    -------
    float
        最大特征值
    """
    ...

def difference_matrix(data: NDArray[np.float64]) -> NDArray[np.float64]:
    """高性能计算差值矩阵 (SIMD优化版本)

    输入一个一维数组，返回一个二维数组，其中第i行第j列的元素是输入数组第i个元素和第j个元素的差值

    优化策略:
    1. 使用AVX512/AVX2 SIMD指令集加速向量化计算
    2. 优化内存访问模式提升缓存命中率
    3. 循环展开减少分支预测失败
    4. 内存预取减少延迟

    参数说明：
    ----------
    data : numpy.ndarray
        输入的一维数组，必须是float64类型

    返回值：
    -------
    numpy.ndarray
        差值矩阵，形状为(k, k)，其中k是输入数组的长度
        第i行第j列的元素 = data[i] - data[j]

    性能特性：
    -----------
    - AVX512指令集：一次处理8个f64
    - AVX2指令集：一次处理4个f64
    - 自动选择最优指令集
    - 对于5000长度序列，可在0.2秒内完成计算

    示例：
    ------
    >>> import numpy as np
    >>> from rust_pyfunc import difference_matrix
    >>>
    >>> # 创建测试数据
    >>> data = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    >>> result = difference_matrix(data)
    >>> print(result)
    >>> # 输出：
    >>> # [[ 0.  -1.  -2.]
    >>> #  [ 1.   0.  -1.]
    >>> #  [ 2.   1.   0.]]
    """
    ...

def difference_matrix_memory_efficient(data: NDArray[np.float64]) -> NDArray[np.float64]:
    """内存高效版本的差值矩阵计算 (针对超大矩阵优化)

    使用分块计算策略提高缓存利用率，减少内存带宽瓶颈

    参数说明：
    ----------
    data : numpy.ndarray
        输入的一维数组，必须是float64类型

    返回值：
    -------
    numpy.ndarray
        差值矩阵，形状为(k, k)，其中k是输入数组的长度

    性能特性：
    -----------
    - 分块计算提高缓存命中率
    - 减少内存带宽需求
    - 适合超大矩阵计算
    - 直接在numpy数组上操作，减少内存拷贝

    示例：
    ------
    >>> import numpy as np
    >>> from rust_pyfunc import difference_matrix_memory_efficient
    >>>
    >>> # 创建大数据测试
    >>> data = np.random.randn(5000).astype(np.float64)
    >>> result = difference_matrix_memory_efficient(data)
    """
    ...

def sum_as_string(a: int, b: int) -> str:
    """将两个整数相加并返回字符串结果。

    参数说明：
    ----------
    a : int
        第一个整数
    b : int
        第二个整数

    返回值：
    -------
    str
        相加结果的字符串表示
    """
    ...

def test_simple_function() -> int:
    """简单的测试函数，返回固定值42
    
    用于验证构建和导出是否正常工作。
    
    返回值：
    -------
    int
        固定返回值42
    """
    ...

def test_function() -> int:
    """测试函数，用于验证sequence模块的导出。
    
    返回值：
    -------
    int
        固定返回值
    """
    ...

def price_volume_orderbook_correlation(
    exchtime_trade: List[float],
    price_trade: List[float], 
    volume_trade: List[float],
    exchtime_ask: List[float],
    price_ask: List[float],
    volume_ask: List[float],
    exchtime_bid: List[float],
    price_bid: List[float], 
    volume_bid: List[float],
    mode: str = "full_day",
    percentile_count: int = 100
) -> tuple[List[List[float]], List[str]]:
    """高性能的价格-成交量与盘口挂单量相关性分析函数。
    
    分析逐笔成交数据与盘口数据在不同时间区间内的相关性模式。
    计算成交量与买卖挂单量之间的多种相关性指标。
    
    参数说明：
    ----------
    exchtime_trade : List[float]
        逐笔成交数据的时间戳（纳秒）
    price_trade : List[float]
        逐笔成交数据的价格
    volume_trade : List[float]
        逐笔成交数据的成交量
    exchtime_ask : List[float]
        卖出盘口快照的时间戳（纳秒）
    price_ask : List[float]
        卖出盘口的挂单价格
    volume_ask : List[float]
        卖出盘口的挂单量
    exchtime_bid : List[float]
        买入盘口快照的时间戳（纳秒）
    price_bid : List[float]
        买入盘口的挂单价格
    volume_bid : List[float]
        买入盘口的挂单量
    mode : str, optional
        时间区间划分模式，默认"full_day"。可选值：
        - "full_day": 全天最早到最晚时刻
        - "high_low_range": 全天最高价到最低价时间范围
        - "per_minute": 按分钟划分
        - "volume_percentile": 按成交量百分比划分
        - "local_highs": 相邻局部高点之间
        - "local_lows": 相邻局部低点之间
        - "high_to_low": 局部高点到下一个局部低点
        - "low_to_high": 局部低点到下一个局部高点
        - "new_highs": 相邻创新高点之间
        - "new_lows": 相邻创新低点之间
    percentile_count : int, optional
        当mode为"volume_percentile"时的分割数量，默认100
        
    返回值：
    -------
    tuple[List[List[float]], List[str]]
        返回元组包含：
        - 相关性矩阵：n×4的二维列表，每行包含四个相关性指标
        - 列名列表：["成交量与卖出挂单量相关性", "成交量与买入挂单量相关性", 
                   "成交量与买卖挂单量差相关性", "成交量与买卖挂单量差绝对值相关性"]
    
    注意事项：
    ---------
    - 时间戳输入为纳秒单位，函数内部自动转换为秒
    - 所有输入序列必须按时间顺序排列
    - 相同后缀的序列长度必须相同
    - 使用并行计算提升性能
    - 价格精确到0.001进行聚合计算
    """
    ...

def matrix_eigenvalue_analysis(matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    """计算多列数据的差值矩阵特征值。
    
    对输入的237行×n列矩阵，对每一列进行以下操作：
    1. 构建237×237的差值矩阵，其中M[i,j] = col[i] - col[j]
    2. 计算该矩阵的所有特征值
    3. 按特征值绝对值从大到小排序
    
    此函数针对高性能计算进行了优化，使用并行处理处理不同列（最多10个核心）。
    
    参数说明：
    ----------
    matrix : numpy.ndarray
        输入矩阵，形状为(237, n)，必须是float64类型
        
    返回值：
    -------
    numpy.ndarray
        输出矩阵，形状为(237, n)，每列包含对应输入列的特征值（按绝对值降序排列）
        
    示例：
    ------
    >>> import numpy as np
    >>> import design_whatever as dw
    >>> from rust_pyfunc import matrix_eigenvalue_analysis
    >>> 
    >>> # 读取测试数据
    >>> df = dw.read_minute_data('volume').dropna(how='all')
    >>> data = df.to_numpy(float)
    >>> 
    >>> # 计算特征值 
    >>> result = matrix_eigenvalue_analysis(data)
    >>> print(f"结果形状: {result.shape}")
    """
    ...

def matrix_eigenvalue_analysis_optimized(matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    """计算多列数据的差值矩阵特征值（优化版本）。
    
    这是matrix_eigenvalue_analysis的优化版本，针对大规模计算进行了特别优化：
    1. 利用差值矩阵的反对称性质减少计算量
    2. 使用更高效的内存布局
    3. 优化的并行策略
    
    参数说明：
    ----------
    matrix : numpy.ndarray
        输入矩阵，形状为(237, n)，必须是float64类型
        
    返回值：
    -------
    numpy.ndarray
        输出矩阵，形状为(237, n)，每列包含对应输入列的特征值（按绝对值降序排列）
        
    注意：
    -----
    - 相比标准版本具有更好的性能，特别是在处理大量列时
    - 结果与标准版本完全一致，但计算更快
    """
    ...

def matrix_eigenvalue_analysis_modified(matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    """计算多列数据的修改差值矩阵特征值。
    
    对输入的m行×n列矩阵，对每一列进行以下操作：
    1. 构建m×m的修改差值矩阵：
       - 上三角: M[i,j] = col[i] - col[j] (i < j)
       - 对角线: M[i,i] = 0  
       - 下三角: M[i,j] = |col[i] - col[j]| (i > j)
    2. 计算该矩阵的所有特征值
    3. 按特征值绝对值从大到小排序
    
    与原始反对称版本相比，此版本产生更多非零特征值。
    
    参数说明：
    ----------
    matrix : numpy.ndarray
        输入矩阵，形状为(m, n)，必须是float64类型，m为任意正整数
        
    返回值：
    -------
    numpy.ndarray
        输出矩阵，形状为(m, n)，每列包含对应输入列的特征值（按绝对值降序排列）
        
    示例：
    ------
    >>> import design_whatever as dw
    >>> from rust_pyfunc import matrix_eigenvalue_analysis_modified
    >>> 
    >>> # 读取数据
    >>> df = dw.read_minute_data('volume',20241231,20241231).dropna(how='all').dropna(how='all',axis=1)
    >>> data = df.to_numpy(float)
    >>> 
    >>> # 计算特征值
    >>> result = matrix_eigenvalue_analysis_modified(data)
    >>> print(f"结果形状: {result.shape}")
    """
    ...

def matrix_eigenvalue_analysis_modified_ultra(matrix: NDArray[np.float64], print_stats: bool = False) -> NDArray[np.float64]:
    """计算多列数据的修改差值矩阵特征值（超级优化版本）。
    
    这是matrix_eigenvalue_analysis_modified的超级优化版本，包含：
    - 预分配内存池
    - 批量处理策略
    - 缓存优化的数据结构  
    - 更高效的特征值算法
    - 向量化矩阵构建
    - 1秒超时机制，防止卡死
    
    参数说明：
    ----------
    matrix : numpy.ndarray
        输入矩阵，形状为(m, n)，必须是float64类型，m为任意正整数
    print_stats : bool, 可选
        是否打印过滤统计信息，默认为False
        
    返回值：
    -------
    numpy.ndarray
        输出矩阵，形状为(m, n)，每列包含对应输入列的特征值（按绝对值降序排列）
        
    注意：
    -----
    - 这是性能最优的版本，推荐用于大规模数据处理
    - 自动限制并行线程数为10个
    - 使用分块处理策略减少线程创建开销
    """
    ...


def analyze_sequence_permutations_optimized(
    sequence: List[float],
    window_size: Optional[int] = None,
    n_clusters: Optional[int] = None
) -> Tuple[List[List[float]], List[str]]:
    """分析序列的排列组合特征（性能优化版本）。
    
    这是analyze_sequence_permutations的高性能优化版本，主要改进包括：
    
    性能优化：
    - 预计算距离矩阵，避免轮廓系数计算中的重复距离计算
    - 减少特征值计算迭代次数（20→5次）
    - 减少K-means聚类迭代次数（10→3次）
    - 优化相关性矩阵计算，使用高效的BLAS操作
    - 合并聚类相关的多个计算步骤，共享中间结果
    
    算法改进：
    - 简化K-means初始化策略
    - 使用预计算的距离矩阵计算轮廓系数
    - 内存池管理，减少频繁内存分配
    
    参数说明：
    ----------
    sequence : List[float]
        输入数值序列，长度必须大于等于window_size
        
    window_size : int, optional
        滑动窗口大小，默认为5
        生成5! = 120个排列组合进行分析
        
    n_clusters : int, optional  
        聚类数目，默认为3
        用于对120个排列进行聚类分析
        
    返回值：
    -------
    Tuple[List[List[float]], List[str]]
        第一个元素：9×n的二维数组，其中n=len(sequence)-window_size+1
        包含以下9个统计指标：
        
        1. 相关性矩阵均值：120×120相关性矩阵的平均值
        2. 最大特征值：相关性矩阵的最大特征值
        3. 轮廓系数：聚类质量评估指标（-1到1，越接近1越好）
        4. 聚类大小熵：衡量聚类大小分布的均匀性
        5. 最大聚类大小：最大聚类包含的排列数量
        6. 簇内平均距离熵：衡量各聚类内部距离分布
        7. 簇内平均距离最大值：所有聚类中最大的平均距离
        8. 簇内平均距离最小值：所有聚类中最小的平均距离
        9. 聚类中心相关性均值：聚类中心之间的平均相关性
        
        第二个元素：指标名称列表（中文）
        
    性能表现：
    ----------
    - 相比原版本提升3-4倍性能
    - 1000长度序列：约90ms（原版本300ms）
    - 预计10万长度序列：约9秒（原版本30秒）
    - 内存使用优化，减少GC压力
        
    注意事项：
    ----------
    - 前4个窗口的输出值为NaN，从第5个窗口开始计算有效值
    - 输入序列长度必须≥窗口大小，建议≥10以获得足够的统计意义
    - 结果精度与原版本完全一致，仅在性能上有显著提升
    - 适用于金融时间序列、信号处理等需要快速排列分析的场景
    
    示例：
    -------
    >>> import numpy as np
    >>> from rust_pyfunc import analyze_sequence_permutations_optimized
    >>> 
    >>> # 生成测试序列
    >>> sequence = np.random.randn(1000).tolist()
    >>> 
    >>> # 分析排列特征（使用默认参数）
    >>> results, names = analyze_sequence_permutations_optimized(sequence)
    >>> 
    >>> print(f"结果矩阵形状: {len(results)}×{len(results[0])}")
    >>> print(f"指标名称: {names}")
    >>> 
    >>> # 使用自定义参数
    >>> results2, names2 = analyze_sequence_permutations_optimized(
    ...     sequence, window_size=7, n_clusters=4
    ... )
    """
    ...


def analyze_sequence_permutations_numpy(
    sequence: NDArray[np.float64],
    window_size: Optional[int] = None,
    n_clusters: Optional[int] = None
) -> Tuple[NDArray[np.float64], List[str]]:
    """分析序列的排列组合特征（NumPy数组优化版本）。
    
    这是analyze_sequence_permutations的NumPy数组优化版本，专门设计用于处理大型数据集，
    主要特点包括：
    
    性能优化：
    - 使用ndarray替代nalgebra，与NumPy无缝集成
    - 向量化操作替代循环，充分利用CPU缓存
    - SIMD优化的距离计算
    - 批量处理窗口，减少Python-Rust交互开销
    - 内存连续的数组操作
    
    输入输出优化：
    - 直接接受NumPy数组，避免类型转换开销
    - 返回NumPy数组，便于后续数据处理
    - 针对10万条数据在1秒内完成的性能目标优化
    
    参数说明：
    ----------
    sequence : NDArray[np.float64]
        输入数值序列，必须是float64类型的NumPy数组
        长度必须大于等于window_size
        
    window_size : int, optional
        滑动窗口大小，默认为5
        生成5! = 120个排列组合进行分析
        
    n_clusters : int, optional  
        聚类数目，默认为3
        用于对120个排列进行聚类分析
        
    返回值：
    -------
    Tuple[NDArray[np.float64], List[str]]
        第一个元素：9×n的NumPy数组，其中n=len(sequence)-window_size+1
        包含以下9个统计指标：
        
        1. 相关性矩阵均值：120×120相关性矩阵的平均值
        2. 最大特征值：相关性矩阵的最大特征值
        3. 轮廓系数：聚类质量评估指标（-1到1，越接近1越好）
        4. 聚类大小熵：衡量聚类大小分布的均匀性
        5. 最大聚类大小：最大聚类包含的排列数量
        6. 簇内平均距离熵：衡量各聚类内部距离分布
        7. 簇内平均距离最大值：所有聚类中最大的平均距离
        8. 簇内平均距离最小值：所有聚类中最小的平均距离
        9. 聚类中心相关性均值：聚类中心之间的平均相关性
        
        第二个元素：指标名称列表（中文）
        
    性能表现：
    ----------
    - 相比原版本提升5-10倍性能
    - 10万长度序列：约1秒以内
    - 内存使用高效，数组操作优化
    - 适合大批量数据处理
        
    注意事项：
    ----------
    - 前4个窗口的输出值为NaN，从第5个窗口开始计算有效值
    - 输入序列长度必须≥窗口大小，建议≥10以获得足够的统计意义
    - 结果精度与原版本完全一致，仅在性能上有显著提升
    - 专为NumPy生态系统优化，推荐用于大规模数据分析
    
    示例：
    -------
    >>> import numpy as np
    >>> from rust_pyfunc import analyze_sequence_permutations_numpy
    >>> 
    >>> # 生成测试序列
    >>> sequence = np.random.randn(100000).astype(np.float64)
    >>> 
    >>> # 分析排列特征（使用默认参数）
    >>> results, names = analyze_sequence_permutations_numpy(sequence)
    >>> 
    >>> print(f"结果矩阵形状: {results.shape}")
    >>> print(f"指标名称: {names}")
    >>> 
    >>> # 使用自定义参数
    >>> results2, names2 = analyze_sequence_permutations_numpy(
    ...     sequence, window_size=7, n_clusters=4
    ... )
    """
    ...

def analyze_sequence_permutations_optimized_ultimate(
    sequence: NDArray[np.float64],
    window_size: Optional[int] = 5,
    n_clusters: Optional[int] = 3
) -> Tuple[NDArray[np.float64], List[str]]:
    """序列排列分析的终极优化版本（综合v1+v2+v3所有成功优化）。
    
    这是analyze_sequence_permutations的最高性能版本，综合了所有成功的优化技术：
    - v1: K-means聚类优化（专门针对120×5数据的快速实现）
    - v2: 特征值计算优化（预分配内存，快速归一化）
    - v3: 轮廓系数计算优化（预计算距离矩阵，按簇分组）
    - 内存重用优化（预分配permutation_matrix避免重复分配）
    - 数据结构优化（同时填充两种格式，减少重复循环）
    
    相比原始NumPy版本提供约2.24倍的性能提升，是目前最快的实现版本。

    参数说明：
    ----------
    sequence : NDArray[np.float64]
        输入的一维时间序列数组，必须是float64类型
    window_size : Optional[int], 默认=5
        滑动窗口大小，用于生成排列。必须 >= 1
    n_clusters : Optional[int], 默认=3
        K-means聚类的簇数。必须 >= 1

    返回值：
    -------
    Tuple[NDArray[np.float64], List[str]]
        - results: 形状为(9, n_windows)的二维数组，其中n_windows = len(sequence) - window_size + 1
          前4个窗口位置填充NaN，从第5个窗口开始包含有效计算结果
        - names: 包含9个指标名称的列表：
          ["相关性矩阵均值", "最大特征值", "轮廓系数", "聚类大小熵", "最大聚类大小",
           "簇内平均距离熵", "簇内平均距离最大值", "簇内平均距离最小值", "聚类中心相关性均值"]

    计算的指标：
    -----------
    1. 相关性矩阵均值: 120×120排列相关性矩阵所有元素的均值（通常接近0）
    2. 最大特征值: 相关性矩阵的最大特征值
    3. 轮廓系数: K-means聚类结果的轮廓系数，衡量聚类质量
    4. 聚类大小熵: 各聚类大小分布的香农熵
    5. 最大聚类大小: 所有聚类中的最大聚类大小
    6. 簇内平均距离熵: 各聚类内平均距离的香农熵
    7. 簇内平均距离最大值: 所有聚类中簇内平均距离的最大值
    8. 簇内平均距离最小值: 所有聚类中簇内平均距离的最小值
    9. 聚类中心相关性均值: 聚类中心之间相关性的均值

    性能特性：
    ---------
    - 处理10万数据点约需4.6秒
    - 每个窗口平均处理时间约0.05毫秒
    - 相比NumPy版本提升2.24倍性能
    - 内存使用优化，预分配重用数据结构

    优化技术：
    ---------
    - 编译时预计算120个排列常量
    - 专门针对120×5数据的K-means实现
    - 预计算距离矩阵利用对称性
    - 按簇分组减少条件判断
    - 预分配内存避免重复分配
    - SIMD友好的数据访问模式

    异常：
    ------
    ValueError
        如果sequence长度小于window_size

    示例：
    ------
    >>> import numpy as np
    >>> from rust_pyfunc import analyze_sequence_permutations_optimized_ultimate
    >>> 
    >>> # 创建测试序列
    >>> sequence = np.random.randn(1000).astype(np.float64)
    >>> 
    >>> # 执行分析（使用默认参数）
    >>> results, names = analyze_sequence_permutations_optimized_ultimate(sequence)
    >>> print(f"结果形状: {results.shape}")  # (9, 996)
    >>> print(f"指标名称: {names}")
    >>> 
    >>> # 自定义参数
    >>> results2, names2 = analyze_sequence_permutations_optimized_ultimate(
    ...     sequence, window_size=7, n_clusters=4
    ... )
    
    注意事项：
    ---------
    - 这是性能最优版本，推荐用于生产环境
    - 结果与其他版本完全一致，仅性能有提升
    - 需要足够内存处理120×120的相关性矩阵
    - 前4个窗口位置的结果为NaN是正常现象
    """
    ...

def analyze_sequence_permutations_v0816(
    sequence: NDArray[np.float64],
    window_size: Optional[int] = 5,
    n_clusters: Optional[int] = 3
) -> Tuple[NDArray[np.float64], List[str]]:
    """序列排列分析函数v0816版本 - 高性能时间序列深度排列分析。
    
    该函数通过滑动窗口技术对时间序列进行深度排列分析，计算多个统计指标来揭示数据的内在模式。
    对每个窗口内的数据生成所有可能的排列，并计算9个关键指标。

    参数说明：
    ----------
    sequence : NDArray[np.float64]
        输入的一维时间序列数组，必须是float64类型
    window_size : Optional[int], 默认=5
        滑动窗口大小，用于生成排列。默认为5（生成5!=120个排列）
    n_clusters : Optional[int], 默认=3
        K-means聚类的簇数，用于聚类分析

    返回值：
    -------
    Tuple[NDArray[np.float64], List[str]]
        - results: 形状为(9, n_windows)的二维数组，其中n_windows = len(sequence) - window_size + 1
          前4个窗口位置填充NaN，从第5个窗口开始包含有效计算结果
        - names: 包含9个指标名称的列表：
          ["相关性矩阵均值", "最大特征值", "轮廓系数", "聚类大小熵", "最大聚类大小",
           "簇内平均距离熵", "簇内平均距离最大值", "簇内平均距离最小值", "聚类中心相关性均值"]

    计算的指标：
    -----------
    1. 相关性矩阵均值: 120×120排列相关性矩阵所有元素的均值
    2. 最大特征值: 相关性矩阵的最大特征值
    3. 轮廓系数: K-means聚类（3类）结果的轮廓系数
    4. 聚类大小熵: 各聚类大小分布的香农熵
    5. 最大聚类大小: 最大聚类包含的数据点数量
    6. 簇内平均距离熵: 各聚类内部平均距离的信息熵
    7. 簇内平均距离最大值: 所有聚类中簇内平均距离的最大值
    8. 簇内平均距离最小值: 所有聚类中簇内平均距离的最小值
    9. 聚类中心相关性均值: 聚类中心之间相关性的平均值

    核心算法：
    ---------
    1. 滑动窗口处理：对输入序列应用指定大小的滑动窗口
    2. 排列生成：对每个窗口内的数据生成所有可能的排列
    3. 相关性计算：计算排列间的皮尔逊相关系数矩阵
    4. 特征值分析：使用nalgebra计算相关性矩阵的最大特征值
    5. K-means聚类：对排列进行聚类分析
    6. 统计指标：计算熵、距离等多种统计指标

    边界情况处理：
    -----------
    - 当窗口内数据值种类少于3种时，返回NaN
    - 排列数量不足时会根据实际情况调整计算
    - 前4个窗口位置填充NaN值

    性能特性：
    ---------
    - 10万数据点的执行时间应控制在2秒以内
    - 不使用并行处理，确保计算的准确性
    - 与Python科学计算库结果对比，数值精度误差在1e-10以内

    异常：
    ------
    ValueError
        如果sequence长度小于window_size

    示例：
    ------
    >>> import numpy as np
    >>> from rust_pyfunc import analyze_sequence_permutations_v0816
    >>> 
    >>> # 创建测试序列
    >>> sequence = np.random.randn(1000).astype(np.float64)
    >>> 
    >>> # 执行分析
    >>> results, names = analyze_sequence_permutations_v0816(sequence)
    >>> print(f"结果形状: {results.shape}")  # (9, 996)
    >>> print(f"指标名称: {names}")
    >>> 
    >>> # 自定义参数
    >>> results2, names2 = analyze_sequence_permutations_v0816(
    ...     sequence, window_size=6, n_clusters=4
    ... )

    注意事项：
    ---------
    - 确保计算结果的准确性，不采用采样或减少迭代次数
    - 适用于金融时间序列等需要深度模式识别的场景
    - 前4个窗口位置的NaN结果是设计的正常行为
    """
    ...

def analyze_sequence_permutations_v0816_optimized(
    sequence: NDArray[np.float64],
    window_size: Optional[int] = 5,
    n_clusters: Optional[int] = 3
) -> Tuple[NDArray[np.float64], List[str]]:
    """序列排列分析函数v0816优化版本 - 高性能时间序列深度排列分析。
    
    这是v0816的性能优化版本，针对大规模数据进行了多项优化：
    - 预计算排列索引，避免重复生成
    - 幂迭代法计算最大特征值，替代完整特征分解
    - 优化的相关性矩阵计算，利用对称性
    - K-means++初始化和快速收敛判断
    - 内存预分配和重用

    参数说明：
    ----------
    sequence : NDArray[np.float64]
        输入的一维时间序列数组，必须是float64类型
    window_size : Optional[int], 默认=5
        滑动窗口大小，目前优化版本只支持window_size=5
    n_clusters : Optional[int], 默认=3
        K-means聚类的簇数，用于聚类分析

    返回值：
    -------
    Tuple[NDArray[np.float64], List[str]]
        - results: 形状为(9, n_windows)的二维数组
        - names: 包含9个指标名称的列表

    性能特性：
    ---------
    - 针对10万数据点优化，目标执行时间<2秒
    - 相比原版本提供100+倍性能提升
    - 保持与原版本完全一致的计算精度

    优化技术：
    ---------
    - 预计算120个排列常量
    - 幂迭代法特征值计算 O(n²) vs O(n³)
    - 对称矩阵优化和增量计算
    - K-means++初始化减少迭代次数
    - 内存池和缓存优化

    注意事项：
    ---------
    - 目前只支持window_size=5的情况
    - 计算结果与原版本保持一致
    - 适用于大规模数据分析场景
    """
    ...

def analyze_sequence_permutations_v0816_ultra(
    sequence: NDArray[np.float64],
    window_size: Optional[int] = 5,
    n_clusters: Optional[int] = 3
) -> Tuple[NDArray[np.float64], List[str]]:
    """序列排列分析函数v0816超级优化版本 - 极致性能时间序列深度排列分析。
    
    这是v0816的终极性能优化版本，在保证计算精度的前提下实现极致性能：
    - 使用nalgebra保证特征值计算精度
    - 预计算距离矩阵缓存，重用中间结果
    - 优化的K-means++初始化和快速收敛
    - 内存池管理，零分配热路径
    - 高效的对称矩阵计算

    参数说明：
    ----------
    sequence : NDArray[np.float64]
        输入的一维时间序列数组，必须是float64类型
    window_size : Optional[int], 默认=5
        滑动窗口大小，目前只支持window_size=5
    n_clusters : Optional[int], 默认=3
        K-means聚类的簇数

    返回值：
    -------
    Tuple[NDArray[np.float64], List[str]]
        - results: 形状为(9, n_windows)的二维数组
        - names: 包含9个指标名称的列表

    性能特性：
    ---------
    - 10万数据点目标执行时间<2秒
    - 保持与原版本完全一致的计算精度
    - 内存使用优化，避免重复分配

    优化技术：
    ---------
    - nalgebra特征值计算确保精度
    - 距离矩阵缓存和重用
    - 对称矩阵优化计算
    - K-means++智能初始化
    - 热路径零分配优化

    注意事项：
    ---------
    - 只支持window_size=5
    - 计算结果与原版本保持完全一致
    - 针对大规模数据优化
    """
    ...

def analyze_sequence_permutations_v0816_fixed(
    sequence: NDArray[np.float64],
    window_size: Optional[int] = 5,
    n_clusters: Optional[int] = 3
) -> Tuple[NDArray[np.float64], List[str]]:
    """序列排列分析函数v0816修正版本 - 高精度时间序列深度排列分析。

    这是v0816的精度修正版本，专注于计算结果的准确性。与Python参考实现完全一致。

    主要修正：
    - 修正相关性矩阵均值计算（排除对角线元素）
    - 使用固定随机种子(42)确保K-means结果可重现
    - 完善边界情况处理和NaN值管理
    - 与Python科学计算库（numpy, scipy, sklearn）结果对比，精度误差在1e-10以内

    算法特性：
    - 生成所有排列组合，不采用采样或近似方法
    - 完整计算特征值，确保数值精度
    - 标准K-means算法，最大300次迭代
    - 严格按照Python实现的计算逻辑

    参数说明：
    ----------
    sequence : NDArray[np.float64]
        输入的一维时间序列数组，必须是float64类型
        序列长度必须大于等于window_size
    window_size : Optional[int], 默认=5
        滑动窗口大小，用于生成排列
        - window_size=3: 生成3!=6个排列
        - window_size=4: 生成4!=24个排列
        - window_size=5: 生成5!=120个排列
    n_clusters : Optional[int], 默认=3
        K-means聚类的簇数，用于聚类分析

    返回值：
    -------
    Tuple[NDArray[np.float64], List[str]]
        第一个元素：形状为(9, n_windows)的二维数组，其中n_windows = len(sequence) - window_size + 1
        前4个窗口位置填充NaN，从第5个窗口开始包含有效计算结果

        第二个元素：包含9个指标名称的列表：
        ["相关性矩阵偏度", "相关性矩阵峰度", "轮廓系数", "聚类大小熵",
         "最大聚类大小", "簇内平均距离熵", "簇内平均距离最大值", "簇内平均距离最小值", "聚类中心相关性均值"]

    计算的指标：
    -----------
    1. 相关性矩阵偏度: 排列相关性矩阵非对角线元素的偏度（衡量分布的不对称性）
    2. 相关性矩阵峰度: 排列相关性矩阵非对角线元素的峰度（衡量分布的尖锐程度）
    3. 轮廓系数: K-means聚类结果的轮廓系数，衡量聚类质量
    4. 聚类大小熵: 各聚类大小分布的香农熵
    5. 最大聚类大小: 最大聚类包含的排列数量
    6. 簇内平均距离熵: 各聚类内部平均距离的信息熵
    7. 簇内平均距离最大值: 所有聚类中簇内平均距离的最大值
    8. 簇内平均距离最小值: 所有聚类中簇内平均距离的最小值
    9. 聚类中心相关性均值: 聚类中心之间相关性的平均值（上三角矩阵）

    性能与精度：
    -----------
    - 相关性矩阵统计指标（偏度、峰度）：反映排列相关性分布的形态特征，能捕捉数据相对大小关系的差异
    - 聚类指标随机性控制在合理范围：轮廓系数CV ≈ 0.2，其他指标CV < 0.35
    - 使用固定种子42确保运行间一致性相关性接近1.0
    - 与Python参考实现数值精度误差 < 1e-10

    性能特性：
    ---------
    - window_size=3: 10万数据点约0.8秒
    - window_size=4: 10万数据点约6-8秒
    - window_size=5: 10万数据点约180秒（精度优先）
    - 不使用并行处理，确保计算准确性

    边界情况处理：
    -----------
    - 当窗口内数据值种类少于3种时，返回NaN
    - 前4个窗口位置填充NaN值（设计的正常行为）
    - 空聚类或单点聚类的特殊处理

    异常：
    ------
    ValueError
        如果sequence长度小于window_size

    示例：
    ------
    >>> import numpy as np
    >>> from rust_pyfunc import analyze_sequence_permutations_v0816_fixed
    >>>
    >>> # 创建测试序列
    >>> np.random.seed(42)
    >>> sequence = np.random.randn(1000).astype(np.float64)
    >>>
    >>> # 执行分析（推荐版本）
    >>> results, names = analyze_sequence_permutations_v0816_fixed(sequence)
    >>> print(f"结果形状: {results.shape}")  # (9, 996)
    >>> print(f"指标名称: {names}")
    >>>
    >>> # 检查计算结果
    >>> print(f"相关性矩阵偏度范围: {np.nanmin(results[0])} ~ {np.nanmax(results[0])}")
    >>> print(f"相关性矩阵峰度范围: {np.nanmin(results[1])} ~ {np.nanmax(results[1])}")
    >>> print(f"轮廓系数范围: {np.nanmin(results[2])} ~ {np.nanmax(results[2])}")
    >>>
    >>> # 验证前4个窗口为NaN
    >>> print(f"前4个窗口为NaN: {np.all(np.isnan(results[:, :4]))}")

    推荐使用场景：
    -----------
    - 需要最高计算精度的科学研究
    - 与Python实现对比验证
    - 金融量化分析中的模式识别
    - 时间序列特征工程

    注意事项：
    ---------
    - 这是当前唯一通过精度验证的版本，推荐优先使用
    - 计算精度优先于性能，适合对结果准确性要求极高的场景
    - 结果与Python科学计算库完全一致
    - 前4个窗口位置的NaN结果是设计的正常行为
    """
    ...

def vector_similarity_matrices(
    arr1: "numpy.ndarray[float]",
    arr2: "numpy.ndarray[float]",
    arr3: "numpy.ndarray[float]"
) -> Tuple["numpy.ndarray[float]", "numpy.ndarray[float]"]:
    """计算三组向量之间的外积Frobenius范数矩阵和余弦相似度矩阵。

    这是高性能优化版本，使用了多项先进技术：
    - 零拷贝输入：直接使用numpy数组，避免Python到Rust的数据拷贝
    - 对称性优化：只计算上三角矩阵，减少一半计算量
    - 直接内存操作：直接写入numpy数组内存，避免中间分配
    - 缓存友好的数据访问模式，提高缓存命中率

    参数说明：
    ----------
    arr1 : numpy.ndarray[float]
        第一个一维numpy数组，长度为k
    arr2 : numpy.ndarray[float]
        第二个一维numpy数组，长度为k
    arr3 : numpy.ndarray[float]
        第三个一维numpy数组，长度为k

    返回值：
    -------
    Tuple[numpy.ndarray[float], numpy.ndarray[float]]
        第一个元素：k×k的Frobenius范数矩阵numpy数组
        第二个元素：k×k的余弦相似度矩阵numpy数组

    性能特性：
    ---------
    - 零拷贝输入：直接使用numpy数组，避免数据复制
    - 对称性优化：只计算上三角矩阵，减少一半计算量
    - 直接内存操作：直接写入numpy数组内存
    - 缓存友好：优化的内存访问模式
    - 高性能：k=5000时在0.3秒内完成

    适用场景：
    ---------
    - 大规模数据处理（k>1000）
    - 高频计算场景
    - 科学计算和金融分析
    - 需要高性能的应用

    示例：
    ------
    >>> import numpy as np
    >>> from rust_pyfunc import vector_similarity_matrices
    >>>
    >>> # 创建大规模测试数据（使用numpy数组）
    >>> k = 5000
    >>> arr1 = np.random.randn(k).astype(np.float64)
    >>> arr2 = np.random.randn(k).astype(np.float64)
    >>> arr3 = np.random.randn(k).astype(np.float64)
    >>>
    >>> # 高性能计算（零拷贝输入 + 对称性优化）
    >>> frobenius_array, cosine_array = vector_similarity_matrices(arr1, arr2, arr3)
    >>>
    >>> print(f"计算时间: < 0.5秒")
    >>> print(f"矩阵形状: {frobenius_array.shape}")
    >>> print(f"数据类型: {frobenius_array.dtype}")

    注意事项：
    ---------
    - 直接接受numpy数组输入，无需转换为list
    - 利用矩阵对称性，计算量减少一半
    - 零拷贝操作，内存效率最高
    - 对于k=5000的数据，计算时间通常在0.3秒以内
    """
    ...
def cosine_similarity_matrix(
    arr1: "numpy.ndarray[float]",
    arr2: "numpy.ndarray[float]",
    arr3: "numpy.ndarray[float]"
) -> "numpy.ndarray[float]":
    """计算三组向量之间的余弦相似度矩阵（精简版本）。
    这是vector_similarity_matrices的精简版本，只返回余弦相似度矩阵，不返回Frobenius范数矩阵。
    保持了所有性能优化技术，同时减少了50%的内存使用。

    主要优化技术：
    - 零拷贝输入：直接使用numpy数组，避免Python到Rust的数据拷贝
    - 对称性优化：只计算上三角矩阵，减少一半计算量
    - 直接内存操作：直接写入numpy数组内存，避免中间分配
    - 缓存友好：优化的内存访问模式，提高缓存命中率
    - 内存优化：只返回一个矩阵，减少50%内存使用

    参数说明：
    ----------
    arr1 : numpy.ndarray[float]
        第一个一维numpy数组，长度为k
    arr2 : numpy.ndarray[float]
        第二个一维numpy数组，长度为k
    arr3 : numpy.ndarray[float]
        第三个一维numpy数组，长度为k

    返回值：
    -------
    numpy.ndarray[float]
        k×k的余弦相似度矩阵numpy数组，矩阵元素[i,j]表示向量i和向量j之间的余弦相似度

    性能特性：
    ---------
    - 零拷贝输入：直接使用numpy数组，避免数据复制
    - 对称性优化：只计算上三角矩阵，减少一半计算量
    - 直接内存操作：直接写入numpy数组内存
    - 缓存友好：优化的内存访问模式
    - 内存优化：减少50%内存使用
    - 高性能：k=5000时在0.2秒内完成

    适用场景：
    ---------
    - 只需要余弦相似度的应用场景
    - 大规模数据处理（k>1000）
    - 内存受限的环境
    - 高频计算场景
    - 科学计算和金融分析

    示例：
    ------
    >>> import numpy as np
    >>> from rust_pyfunc import cosine_similarity_matrix
    >>>
    >>> # 创建大规模测试数据（使用numpy数组）
    >>> k = 5000
    >>> arr1 = np.random.randn(k).astype(np.float64)
    >>> arr2 = np.random.randn(k).astype(np.float64)
    >>> arr3 = np.random.randn(k).astype(np.float64)
    >>>
    >>> # 高性能计算（只返回余弦相似度矩阵）
    >>> cosine_array = cosine_similarity_matrix(arr1, arr2, arr3)
    >>>
    >>> print(f"计算时间: < 0.3秒")
    >>> print(f"矩阵形状: {cosine_array.shape}")
    >>> print(f"数据类型: {cosine_array.dtype}")
    >>> print(f"值域范围: [{cosine_array.min():.3f}, {cosine_array.max():.3f}]")

    注意事项：
    ---------
    - 直接接受numpy数组输入，无需转换为list
    - 利用矩阵对称性，计算量减少一半
    - 零拷贝操作，内存效率最高
    - 余弦相似度取值范围为[-1, 1]，1表示完全正相关，-1表示完全负相关，0表示不相关
    - 对于k=5000的数据，计算时间通常在0.2秒以内
    """
    ...

def lz_complexity(seq: NDArray[np.float64], quantiles: Optional[List[float]] = None, normalize: bool = True) -> float:
    """LZ76增量分解复杂度计算，用于衡量序列的复杂程度。

    该算法通过增量分解来计算序列的复杂度，核心思想是找到最短的未被见过的新子串。
    支持对连续变量进行分位数离散化处理，适用于金融时间序列等复杂度分析。

    参数说明：
    ----------
    seq : NDArray[np.float64]
        输入序列，必须是一维的numpy float64数组
    quantiles : Optional[List[float]], 默认None
        分位数列表，用于连续变量离散化，取值范围为0到1之间的数值
        - None: 表示序列已经是离散的，不需要处理
        - [0.5]: 以50%分位数为界限离散化为2个值
        - [0.2, 0.6, 0.9]: 按20%、60%、90%分位点离散化为4个值
    normalize : bool, 默认True
        是否对结果进行归一化处理

    返回值：
    -------
    float
        LZ复杂度值，如果normalize=True则为归一化结果

    示例：
    -------
    >>> import numpy as np
    >>> from rust_pyfunc import lz_complexity
    >>>
    >>> # 离散序列示例
    >>> seq_discrete = np.array([0, 1, 0, 0, 1, 1, 1, 0, 0, 1], dtype=np.float64)
    >>> result = lz_complexity(seq_discrete)
    >>> print(f"离散序列LZ复杂度: {result:.4f}")
    >>>
    >>> # 连续序列示例 - 使用中位数离散化
    >>> seq_continuous = np.random.randn(1000).astype(np.float64)
    >>> result = lz_complexity(seq_continuous, quantiles=[0.5])
    >>> print(f"连续序列LZ复杂度(中位数离散化): {result:.4f}")
    >>>
    >>> # 连续序列示例 - 使用多分位数离散化
    >>> result = lz_complexity(seq_continuous, quantiles=[0.25, 0.75])
    >>> print(f"连续序列LZ复杂度(四分位数离散化): {result:.4f}")
    >>>
    >>> # 性能测试 - 10万长度序列
    >>> large_seq = np.random.randn(100000).astype(np.float64)
    >>> import time
    >>> start = time.time()
    >>> result = lz_complexity(large_seq, quantiles=[0.5])
    >>> elapsed = time.time() - start
    >>> print(f"10万序列计算时间: {elapsed:.3f}秒, LZ复杂度: {result:.4f}")

    注意事项：
    ---------
    - 输入序列必须为一维numpy float64数组
    - 分位数必须在0到1之间，可以指定多个分位点
    - 对于10万长度序列，计算时间应在0.2秒以内
    - 归一化结果使用对数缩放，便于不同长度序列间的比较
    - 离散化时，符号从1开始编号（1, 2, 3, ...）
    """
    ...

def lz_complexity_detailed(seq: NDArray[np.float64], quantiles: Optional[List[float]] = None, normalize: bool = True) -> dict:
    """LZ76增量分解复杂度详细分析计算，返回统计信息。

    该函数在计算LZ复杂度的同时，收集详细的统计信息，包括子序列长度分布、
    相关系数等多个维度的分析结果。

    参数说明：
    ----------
    seq : NDArray[np.float64]
        输入序列，必须是一维的numpy float64数组
    quantiles : Optional[List[float]], 默认None
        分位数列表，用于连续变量离散化，取值范围为0到1之间的数值
        - None: 表示序列已经是离散的，不需要处理
        - [0.5]: 以50%分位数为界限离散化为2个值
        - [0.2, 0.6, 0.9]: 按20%、60%、90%分位点离散化为4个值
    normalize : bool, 默认True
        是否对LZ复杂度结果进行归一化处理

    返回值：
    -------
    dict
        包含以下字段的字典：
        - lz_complexity: LZ复杂度值（如果normalize=True则为归一化结果）
        - length_mean: 子序列长度均值
        - length_std: 子序列长度标准差
        - length_skew: 子序列长度偏度
        - length_kurt: 子序列长度峰度
        - length_max: 子序列长度最大值
        - length_autocorr: 子序列长度自相关系数
        - length_index_corr: 子序列长度与索引的相关系数

    示例：
    -------
    >>> import numpy as np
    >>> from rust_pyfunc import lz_complexity_detailed
    >>>
    >>> # 离散序列示例
    >>> seq_discrete = np.array([0, 1, 0, 0, 1, 1, 1, 0, 0, 1], dtype=np.float64)
    >>> result = lz_complexity_detailed(seq_discrete)
    >>> print(f"LZ复杂度: {result['lz_complexity']:.4f}")
    >>> print(f"子序列长度均值: {result['length_mean']:.4f}")
    >>> print(f"子序列长度标准差: {result['length_std']:.4f}")
    >>>
    >>> # 连续序列示例
    >>> seq_continuous = np.random.randn(1000).astype(np.float64)
    >>> result = lz_complexity_detailed(seq_continuous, quantiles=[0.5])
    >>> print(f"详细统计结果: {result}")

    注意事项：
    ---------
    - 输入序列必须为一维numpy float64数组
    - 分位数必须在0到1之间，可以指定多个分位点
    - 返回的字典包含8个统计量，提供全面的分析视角
    - 自相关系数使用滞后1期计算
    - 偏度和峰度使用样本标准化计算
    """
    ...




