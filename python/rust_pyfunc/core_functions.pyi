"""核心数学和统计函数类型声明"""
from typing import List, Optional, Union
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