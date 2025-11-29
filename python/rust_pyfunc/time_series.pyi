"""时间序列分析函数类型声明"""
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
from numpy.typing import NDArray

def dtw_distance(s1: List[float], s2: List[float], radius: Optional[int] = None, timeout_seconds: Optional[float] = None) -> float:
    """计算两个时间序列之间的DTW(Dynamic Time Warping)距离。
    
    参数说明：
    ----------
    s1 : List[float]
        第一个时间序列
    s2 : List[float]
        第二个时间序列
    radius : Optional[int]
        约束带宽度，None表示无约束
    timeout_seconds : Optional[float]
        超时时间（秒），None表示无超时
        
    返回值：
    -------
    float
        DTW距离
    """
    ...

def fast_dtw_distance(s1: List[float], s2: List[float], radius: Optional[int] = None, timeout_seconds: Optional[float] = None) -> float:
    """高性能版本的DTW距离计算。
    
    参数说明：
    ----------
    s1 : List[float]
        第一个时间序列
    s2 : List[float]
        第二个时间序列
    radius : Optional[int]
        约束带宽度
    timeout_seconds : Optional[float]
        超时时间（秒）
        
    返回值：
    -------
    float
        DTW距离
    """
    ...

def super_dtw_distance(s1: List[float], s2: List[float], radius: Optional[int] = None, timeout_seconds: Optional[float] = None, lower_bound_pruning: bool = True, early_termination_threshold: Optional[float] = None) -> float:
    """超高性能DTW距离计算，包含多种优化技术。
    
    参数说明：
    ----------
    s1 : List[float]
        第一个时间序列
    s2 : List[float]
        第二个时间序列
    radius : Optional[int]
        约束带宽度
    timeout_seconds : Optional[float]
        超时时间（秒）
    lower_bound_pruning : bool
        是否启用下界剪枝
    early_termination_threshold : Optional[float]
        早期终止阈值
        
    返回值：
    -------
    float
        DTW距离
    """
    ...

def transfer_entropy(x_: List[float], y_: List[float], k: int, c: int) -> float:
    """计算两个时间序列之间的传递熵。
    
    参数说明：
    ----------
    x_ : List[float]
        源时间序列
    y_ : List[float]
        目标时间序列
    k : int
        历史长度
    c : int
        分箱数量
        
    返回值：
    -------
    float
        传递熵值
    """
    ...

def transfer_entropy_safe(x_: List[float], y_: List[float], k: int, c: int) -> float:
    """计算两个时间序列之间的传递熵（安全版本，可处理NaN值）。
    
    与原版transfer_entropy不同，此版本能够安全处理包含NaN值的数据。
    
    参数说明：
    ----------
    x_ : List[float]
        源时间序列，可以包含NaN值
    y_ : List[float]
        目标时间序列，可以包含NaN值
    k : int
        历史长度
    c : int
        分箱数量
        
    返回值：
    -------
    float
        传递熵值，如果数据不足或全为NaN则返回0.0
    """
    ...

def rolling_dtw_distance(ts1: List[float], ts2: List[float], window_size: int, step_size: int = 1, radius: Optional[int] = None) -> List[float]:
    """计算滚动DTW距离。
    
    参数说明：
    ----------
    ts1 : List[float]
        第一个时间序列
    ts2 : List[float]
        第二个时间序列
    window_size : int
        滚动窗口大小
    step_size : int
        步长，默认为1
    radius : Optional[int]
        DTW约束带宽度
        
    返回值：
    -------
    List[float]
        滚动DTW距离序列
    """
    ...

def find_local_peaks_within_window(
    times: NDArray[np.int64], 
    prices: NDArray[np.float64], 
    target_time: int, 
    time_window: int, 
    min_prominence: float = 0.01
) -> List[Tuple[int, float, float]]:
    """在指定时间窗口内寻找局部峰值。
    
    参数说明：
    ----------
    times : NDArray[np.int64]
        时间戳数组
    prices : NDArray[np.float64]
        价格数组
    target_time : int
        目标时间点
    time_window : int
        时间窗口大小
    min_prominence : float
        最小突出度
        
    返回值：
    -------
    List[Tuple[int, float, float]]
        峰值列表，每个元素为(时间, 价格, 突出度)
    """
    ...

def find_half_energy_time(
    times: NDArray[np.int64],
    prices: NDArray[np.float64],
    volumes: NDArray[np.float64],
    time_window_ns: int
) -> List[Tuple[int, int]]:
    """寻找半能量时间点。
    
    参数说明：
    ----------
    times : NDArray[np.int64]
        时间戳数组（纳秒）
    prices : NDArray[np.float64]
        价格数组
    volumes : NDArray[np.float64]
        成交量数组
    time_window_ns : int
        时间窗口大小（纳秒）
        
    返回值：
    -------
    List[Tuple[int, int]]
        半能量时间点列表
    """
    ...

def rolling_window_stat(
    times: NDArray[np.float64],
    values: NDArray[np.float64], 
    window_size: float,
    stat_type: str,
    include_current: bool = True
) -> NDArray[np.float64]:
    """计算向后滚动窗口统计量。
    
    参数说明：
    ----------
    times : NDArray[np.float64]
        时间数组
    values : NDArray[np.float64]
        数值数组
    window_size : float
        窗口大小（单位：秒）
    stat_type : str
        统计类型（"mean", "sum", "max", "min", "std", "median", "count", "rank", "skew", "trend_time", "trend_oneton", "last"）
    include_current : bool
        是否包含当前点
        
    返回值：
    -------
    NDArray[np.float64]
        滚动统计量数组
    """
    ...

def rolling_window_stat_backward(
    times: NDArray[np.float64],
    values: NDArray[np.float64], 
    window_size: float,
    stat_type: str,
    include_current: bool = True
) -> NDArray[np.float64]:
    """计算向前滚动窗口统计量。
    
    参数说明：
    ----------
    times : NDArray[np.float64]
        时间数组
    values : NDArray[np.float64]
        数值数组
    window_size : float
        窗口大小（单位：秒）
    stat_type : str
        统计类型（"mean", "sum", "max", "min", "std", "median", "count", "rank", "skew", "trend_time", "trend_oneton", "first", "last"）
    include_current : bool
        是否包含当前点
        
    返回值：
    -------
    NDArray[np.float64]
        滚动统计量数组
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
    """
    ...

def calculate_lyapunov_exponent(
    data: NDArray[np.floating],
    method: str = "auto",
    m: Optional[int] = None,
    tau: Optional[int] = None,
    max_t: int = 30,
    max_tau: int = 20,
    max_m: int = 10,
    mi_bins: int = 20,
    fnn_rtol: float = 15.0,
    fnn_atol: float = 2.0
) -> Dict[str, Any]:
    """计算时间序列的最大Lyapunov指数，用于量化系统对初始条件的敏感性。
    
    参数说明：
    ----------
    data : NDArray[np.floating]
        输入的时间序列数据（一维numpy数组）
    method : str, default="auto"
        参数选择方法：
        - "auto": 自动综合多种方法确定参数
        - "manual": 手动指定参数（必须提供m和tau）
        - "mutual_info": 使用互信息法确定tau
        - "autocorrelation": 使用自相关法确定tau
    m : Optional[int]
        嵌入维度。manual模式下必须指定
    tau : Optional[int]
        延迟时间。manual模式下必须指定
    max_t : int, default=30
        计算发散率序列的最大时间步长
    max_tau : int, default=20
        自动优化时τ的最大搜索范围
    max_m : int, default=10
        自动优化时m的最大搜索范围
    mi_bins : int, default=20
        互信息计算时的分箱数量
    fnn_rtol : float, default=15.0
        假最近邻法的相对容差阈值（百分比）
    fnn_atol : float, default=2.0
        假最近邻法的绝对容差阈值
        
    返回值：
    -------
    Dict[str, Any]
        包含以下键值的字典：
        - lyapunov_exponent: float - 最大Lyapunov指数
        - divergence_sequence: NDArray - 发散率序列
        - optimal_m: int - 使用的嵌入维度
        - optimal_tau: int - 使用的延迟时间
        - method_used: str - 实际使用的参数选择方法
        - intercept: float - 线性拟合的截距
        - r_squared: float - 线性拟合的决定系数
        - phase_space_size: int - 重构相空间的大小
        - data_length: int - 原始数据长度
        
    说明：
    -----
    Lyapunov指数的物理意义：
    - λ > 0: 混沌系统，初始条件敏感，长期不可预测
    - λ = 0: 临界状态或准周期系统  
    - λ < 0: 稳定系统，扰动会衰减
    
    预测时间范围: τ_pred ≈ 1/|λ|
    
    使用示例：
    --------
    # 自动模式（推荐）
    result = calculate_lyapunov_exponent(data)
    
    # 手动指定参数
    result = calculate_lyapunov_exponent(data, method="manual", m=5, tau=3)
    
    # 仅使用互信息法
    result = calculate_lyapunov_exponent(data, method="mutual_info")
    """
    ...

def rolling_lagged_regression(
    series: NDArray[np.float64],
    past_periods: int,
    future_periods: int
) -> NDArray[np.float64]:
    """滚动滞后自回归分析函数。

    对时间序列的每一行进行滚动计算，使用不同滞后阶数的自回归模型，
    分别计算过去期拟合优度和未来期预测准确度。

    参数说明：
    ----------
    series : NDArray[np.float64]
        输入的一维时间序列数据
    past_periods : int
        过去观察期数，用于拟合AR模型
    future_periods : int
        未来预测期数，必须 <= past_periods

    返回值：
    -------
    NDArray[np.float64]
        二维数组，形状为 (n, 2*x)，其中：
        - n 为输入序列长度
        - x = past_periods - future_periods + 1 为最大滞后阶数
        - 每行包含：[r_lag1_past, r_lag2_past, ..., r_lagx_past,
                    r_lag1_future, r_lag2_future, ..., r_lagx_future]
        - 对于无法计算的前几行，用 NaN 填充

    算法说明：
    --------
    对于序列中的每一行，该函数执行以下计算：
    1. 使用过去 past_periods 期的数据，拟合滞后 k 阶自回归模型 (k=1,2,...,x)
    2. 计算过去期拟合优度 r_lagk_past（R²值）
    3. 使用拟合的模型预测未来 future_periods 期数据
    4. 计算预测值与实际值的R²，记为 r_lagk_future
    5. 重复步骤1-4，直到滞后阶数达到最大值 x

    使用示例：
    --------
    # 假设有100期的时间序列数据
    import numpy as np
    import rust_pyfunc

    # 生成示例数据
    data = np.random.randn(100).astype(np.float64)

    # 使用过去20期拟合，预测未来5期
    result = rust_pyfunc.rolling_lagged_regression(data, 20, 5)

    # 结果维度: (100, 32) - 16个过去期R² + 16个未来期R²
    # 其中最大滞后阶数 = 20 - 5 + 1 = 16
    """
    ...

def rolling_lagged_regression_ridge(
    series: NDArray[np.float64],
    past_periods: int,
    future_periods: int,
    alpha: Optional[float] = None
) -> NDArray[np.float64]:
    """Ridge回归版本的滚动滞后自回归分析函数。

    对时间序列的每一行进行滚动计算，使用不同滞后阶数的Ridge自回归模型，
    分别计算过去期拟合优度和未来期预测准确度。Ridge回归通过L2正则化
    防止过拟合，特别适合高噪声的金融时间序列数据。

    参数说明：
    ----------
    series : NDArray[np.float64]
        输入的一维时间序列数据
    past_periods : int
        过去观察期数，用于拟合Ridge AR模型
    future_periods : int
        未来预测期数，必须 <= past_periods
    alpha : Optional[float], default=None
        Ridge正则化参数λ。如果为None，将自适应选择最优值
        - 较大的α：更强的正则化，更平滑的系数
        - 较小的α：更接近OLS回归
        - α=0：等价于标准OLS回归

    返回值：
    -------
    NDArray[np.float64]
        二维数组，形状为 (n, 2*x)，其中：
        - n 为输入序列长度
        - x = past_periods - future_periods + 1 为最大滞后阶数
        - 每行包含：[r_lag1_past, r_lag2_past, ..., r_lagx_past,
                    r_lag1_future, r_lag2_future, ..., r_lagx_future]
        - 对于无法计算的前几行，用 NaN 填充

    算法说明：
    --------
    Ridge回归的核心改进：
    1. 目标函数：min ||y - Xβ||² + α||β||²
    2. 正则化项α||β||²惩罚过大的系数
    3. 解析解：β_ridge = (X'X + αI)⁻¹X'y
    4. 自适应α选择：基于样本大小、特征数量和噪声水平

    相比标准OLS的优势：
    - 处理多重共线性：滞后变量间的高相关性
    - 防止过拟合：限制高阶项系数，提升泛化能力
    - 数值稳定性：避免接近奇异矩阵的问题
    - 偏差-方差平衡：通过α参数控制权衡点

    金融应用特点：
    - 适合高噪声数据：如挂单量、成交量等微观结构数据
    - 稳健参数估计：对数据扰动不敏感
    - 改善预测性能：Future R²更稳定，不会随滞后期急剧下降

    使用示例：
    --------
    # 基础用法：自动选择正则化参数
    import numpy as np
    import rust_pyfunc

    data = np.random.randn(200).astype(np.float64)
    result = rust_pyfunc.rolling_lagged_regression_ridge(data, 30, 20)

    # 指定正则化参数
    result_custom = rust_pyfunc.rolling_lagged_regression_ridge(
        data, 30, 20, alpha=1.0
    )

    # 对比OLS和Ridge的效果
    ols_result = rust_pyfunc.rolling_lagged_regression(data, 30, 20)
    ridge_result = rust_pyfunc.rolling_lagged_regression_ridge(data, 30, 20)

    注意事项：
    --------
    - 正则化参数的选择很关键，影响偏差-方差权衡
    - 对于低噪声数据，Ridge改进可能不明显
    - 计算时间略长于标准OLS版本
    - Future R²是评估模型质量的关键指标
    """
    ...

def rolling_lagged_regression_ridge_fast(
    series: NDArray[np.float64],
    past_periods: int,
    future_periods: int,
    alpha: Optional[float] = None
) -> NDArray[np.float64]:
    """滚动滞后自回归Ridge分析函数 - 高性能优化版本。

    这是rolling_lagged_regression_ridge的高性能优化版本，通过以下技术实现4-5倍性能提升：

    核心优化技术：
    ============
    1. **faer线性代数库** - 替换nalgebra，提供5-17倍矩阵运算性能提升
    2. **内存池和缓冲区重用** - 避免重复分配，减少20-30%内存开销
    3. **alpha参数缓存** - 缓存自适应参数计算，减少15-20%重复计算
    4. **Cholesky分解优化** - 对Ridge矩阵(X'X + αI)使用专门的对称正定分解
    5. **SIMD和缓存优化** - faer库的底层SIMD指令和缓存友好数据布局

    参数说明：
    ----------
    series : NDArray[np.float64]
        输入的一维时间序列数据
    past_periods : int
        过去观察期数，用于拟合Ridge AR模型
    future_periods : int
        未来预测期数，必须 <= past_periods
    alpha : Optional[float], default=None
        Ridge正则化参数λ。如果为None，将自适应选择最优值

    返回值：
    -------
    NDArray[np.float64]
        二维数组，形状和内容与原版本完全相同：
        - 形状：(n, 2*x)，其中 x = past_periods - future_periods + 1
        - 数值精度：与原版本误差 < 1e-10

    性能对比：
    --------
    测试数据：4741个数据点，past_periods=65, future_periods=50
    - 原版本：~0.94秒
    - 优化版本：~0.18-0.23秒 (4-5倍提升)
    - 内存使用：减少约40%

    应用场景：
    --------
    - 大规模时间序列分析：数据点 > 1000
    - 实时计算场景：需要快速响应
    - 批量处理：处理多个序列
    - 高频数据：如tick级别的金融数据

    使用示例：
    --------
    import numpy as np
    import rust_pyfunc as rp
    import time

    # 生成测试数据
    data = np.random.randn(5000).astype(np.float64)

    # 性能对比
    start = time.time()
    result_original = rp.rolling_lagged_regression_ridge(data, 65, 50)
    time_original = time.time() - start

    start = time.time()
    result_fast = rp.rolling_lagged_regression_ridge_fast(data, 65, 50)
    time_fast = time.time() - start

    print(f"原版本: {time_original:.3f}秒")
    print(f"优化版本: {time_fast:.3f}秒")
    print(f"提升倍数: {time_original/time_fast:.1f}x")

    # 验证结果一致性
    diff = np.abs(result_original - result_fast)
    max_diff = np.nanmax(diff)
    print(f"最大数值差异: {max_diff:.2e}")  # 应该 < 1e-10

    技术细节：
    --------
    1. **faer库优势**：
       - 专门为高性能科学计算设计
       - 原生支持SIMD向量化
       - 优化的缓存访问模式
       - 对称正定矩阵的专用算法

    2. **内存优化**：
       - 预分配工作缓冲区
       - 重用设计矩阵和目标向量
       - 避免临时对象创建

    3. **算法优化**：
       - Cholesky分解代替LU分解
       - 缓存alpha参数避免重复计算
       - 滑动窗口的增量计算

    注意事项：
    --------
    - 结果与原版本在数值精度上完全一致
    - 适用于所有原版本支持的场景
    - 无需修改现有代码接口
    - 推荐用于性能敏感的应用
    """
    ...

def rolling_lagged_regression_ridge_simd(
    series: NDArray[np.float64],
    past_periods: int,
    future_periods: int,
    alpha: Optional[float] = None
) -> NDArray[np.float64]:
    """滚动滞后自回归Ridge分析函数 - SIMD超高性能版本。

    这是rolling_lagged_regression_ridge的SIMD极致优化版本，通过硬件级向量化实现最大性能提升：

    SIMD优化技术：
    ============
    1. **AVX2向量化计算** - 同时处理4个双精度浮点数，大幅提升吞吐量
    2. **向量化R²计算** - SSE/AVX加速残差平方和与总平方和计算
    3. **SIMD点积运算** - 硬件加速的预测值计算和系数向量乘法
    4. **并行统计量计算** - 向量化的均值、方差等基础统计操作
    5. **内存对齐优化** - 确保数据按缓存行对齐，最大化内存带宽利用

    性能特征：
    --------
    - **自适应SIMD** - 根据数据大小和CPU特性自动选择最优算法
    - **CPU特性检测** - 运行时检测AVX2支持，无支持时自动回退
    - **缓存友好** - 优化数据布局减少缓存缺失
    - **零拷贝设计** - 最小化内存分配和数据复制

    参数说明：
    ----------
    series : NDArray[np.float64]
        输入的一维时间序列数据
    past_periods : int
        过去观察期数，用于拟合Ridge AR模型
    future_periods : int
        未来预测期数，必须 <= past_periods
    alpha : Optional[float], default=None
        Ridge正则化参数λ。如果为None，将使用自适应选择

    返回值：
    -------
    NDArray[np.float64]
        二维数组，与原版本格式完全兼容：
        - 形状：(n, 2*x)，其中 x = past_periods - future_periods + 1
        - 数值精度：与标准版本完全一致

    性能预期：
    --------
    根据CPU和数据特征的不同性能提升：
    - **小数据集** (n<1000): 1.5-2x 提升
    - **中等数据集** (1000<n<5000): 2-3x 提升
    - **大数据集** (n>5000): 3-4x 提升
    - **AVX2支持**: 额外20-50%提升

    使用示例：
    --------
    import numpy as np
    import rust_pyfunc as rp

    # SIMD版本调用
    result = rp.rolling_lagged_regression_ridge_simd(data, 65, 50)

    注意事项：
    --------
    - 在不支持AVX2的CPU上会自动回退到标量版本
    - 小数据集可能不会看到显著提升（SIMD开销）
    - 推荐用于计算密集型场景
    """
    ...

def rolling_lagged_regression_ridge_incremental(
    series: NDArray[np.float64],
    past_periods: int,
    future_periods: int,
    alpha: Optional[float] = None
) -> NDArray[np.float64]:
    """增量更新优化版本的滞后自回归分析（Ridge正则化）

    这是rolling_lagged_regression_ridge的增量更新极致优化版本，通过矩阵增量更新技术实现30-40%性能提升：

    核心算法优化：
    1. **增量矩阵更新** - 维护X'X和X'y矩阵的滑动更新，避免重复计算
    2. **滑动窗口优化** - 利用窗口重叠特性，每次只计算差量
    3. **O(n²)→O(n)复杂度** - 矩阵构建从二次复杂度降到线性
    4. **内存复用** - 最大化缓冲区复用，减少分配开销
    5. **数值稳定性** - 保持与原版本完全一致的数值精度

    算法原理：
    ----------
    滑动窗口从位置t到t+1时：
    - 移除最旧观测对X'X的贡献：X'X -= x_old * x_old'
    - 添加最新观测对X'X的贡献：X'X += x_new * x_new'
    - 类似地更新X'y向量
    - 直接在更新后的矩阵上求解Ridge回归

    参数：
    -----
    series : NDArray[np.float64]
        输入时间序列数据
    past_periods : int
        过去观察期数（用于拟合AR模型）
    future_periods : int
        未来预测期数（must <= past_periods）
    alpha : Optional[float]
        Ridge正则化参数。None时使用自适应选择

    返回：
    ------
    NDArray[np.float64]
        形状为(n, 2*k)的二维数组，其中k = past_periods - future_periods + 1
        - [:, 0:k]: 过去期各滞后阶数的拟合优度（R²）
        - [:, k:2k]: 未来期各滞后阶数的预测准确度（R²）

    性能特点：
    ---------
    - **显著加速**: 相比原版本提升30-40%
    - **内存高效**: 增量更新避免重复矩阵构建
    - **数值一致**: 与原版本保持完全相同的计算精度
    - **大数据友好**: 数据量越大，相对提升越明显

    使用示例：
    ---------
    ```python
    import numpy as np
    import rust_pyfunc as rp

    # 生成示例数据
    data = np.random.randn(5000).astype(np.float64)

    # 增量优化版本调用
    result = rp.rolling_lagged_regression_ridge_incremental(data, 65, 50)

    适用场景：
    --------
    - 大规模时间序列分析
    - 实时滚动回归计算
    - 高频金融数据分析
    - 需要长滑动窗口的场景
    """
    ...

def time_irreversibility_static_simple(
    data: NDArray[np.float64],
    m: int = 5
) -> float:
    """静态版-简略版：计算时间不可逆指标 I_ord。

    该函数基于序型（ordinal pattern）框架度量时间序列的不可逆性，只返回核心指标I_ord。
    通过比较正向和反向时间序列中各种序型出现的频率差异来度量时间箭头强度。

    计算原理：
    ----------
    给定时间序列 {x_t}_{t=1}^T 和嵌入维度 m：
    1. 构造长度为 m 的滑动窗口
    2. 将每个窗口转换为序型 π（元素按大小排序后的位置索引）
    3. 统计每种序型的频率 p(π)
    4. 计算反向序型 π^R = (m+1-i_1, ..., m+1-i_m)
    5. 不可逆指标：I_ord = 0.5 * Σ|p(π) - p(π^R)|

    参数：
    -----
    data : numpy.ndarray
        输入的时间序列数据（一维浮点数数组）
    m : int, optional
        嵌入维度（窗口大小），默认值为5（m>=2）

    返回值：
    -------
    float
        时间不可逆性指标 I_ord（取值范围[0,1]）
        - 0：完全可逆（时间对称）
        - 1：完全不可逆（时间不对称）
        - 典型金融时间序列：0.1-0.5

    计算复杂度：
    ------------
    - 时间：O(T * m log m)，T为数据长度
    - 空间：O(m!)，m!为可能的序型数量

    使用示例：
    ----------
    >>> import numpy as np
    >>> import rust_pyfunc as rp
    >>>
    >>> # 生成示例数据
    >>> data = np.random.randn(10000)
    >>>
    >>> # 计算不可逆指标（m=5）
    >>> i_ord = rp.time_irreversibility_static_simple(data, m=5)
    >>> print(f"时间不可逆性: {i_ord:.4f}")
    >>>
    >>> # 测试不同嵌入维度
    >>> for m in [2, 3, 4, 5]:
    ...     i_ord = rp.time_irreversibility_static_simple(data, m=m)
    ...     print(f"m={m}: I_ord = {i_ord:.4f}")

    性能说明：
    ----------
    - 13万条数据，m=5：约0.01秒
    - 使用高效排序算法和缓存优化
    """
    ...

def time_irreversibility_static_detailed(
    data: NDArray[np.float64],
    m: int = 5
) -> Dict[str, Any]:
    """静态版-详细版：计算所有静态时间不可逆性指标。

    该函数提供完整的时间不可逆性分析，返回包含多种相关指标的详细结果。
    除了核心指标I_ord外，还包括序型熵、缺失模式、KL散度等补充指标。

    计算指标：
    ----------
    1. **I_ord**: 核心不可逆指标（0.5 * Σ|p(π) - p(π^R)|）
    2. **Permutation Entropy**: 序型熵 H_perm = -Σp(π)log p(π)
    3. **Normalized Entropy**: 归一化熵 H_norm = H_perm / log(m!)
    4. **Forbidden Patterns**: 缺失序型数量（频率为0的模式）
    5. **Forbidden Ratio**: 缺失序型比例
    6. **KL Divergence**: KL散度 D_KL(p||p^R)
    7. **Local Irreversibility**: 各模式的局部不可逆度 Δ(π) = p(π) - p(π^R)
    8. **Relative Bias**: 相对偏差 B(π) = p(π)/(p(π)+p(π^R)) - 0.5
    9. **Pattern Frequencies**: 各序型频率分布

    参数：
    -----
    data : numpy.ndarray
        输入的时间序列数据（一维浮点数数组）
    m : int, optional
        嵌入维度（窗口大小），默认值为5（m>=2）

    返回值：
    -------
    Dict[str, Any]
        包含所有指标的字典：
        - 'i_ord': float - 核心不可逆指标
        - 'permutation_entropy': float - 序型熵
        - 'normalized_permutation_entropy': float - 归一化熵[0,1]
        - 'forbidden_count': int - 缺失序型个数
        - 'forbidden_ratio': float - 缺失序型比例
        - 'kl_divergence': float - KL散度
        - 'local_irreversibility': NDArray[np.float64] - 局部不可逆度（长度n，前m-1个为NaN）
        - 'local_irreversibility_signed': NDArray[np.float64] - 局部不可逆度带符号版（长度n，前m-1个为NaN）
        - 'relative_bias': NDArray[np.float64] - 相对偏差（长度n，前m-1个为NaN）
        - 'pattern_frequencies': NDArray[np.float64] - 序型频率（长度n，前m-1个为NaN）
        - 'pattern_frequency_all': NDArray[np.float64] - 所有序型频率汇总（长度m!）

    指标解读：
    ----------
    **不可逆性相关：**
    - I_ord: 越大说明时间箭头越强
    - Local Irreversibility: >0说明该模式更倾向正向时间（绝对值版本，>=0）
    - Local Irreversibility Signed: 带符号版本（正向>反向为正，反向>正向为负）

    **复杂性相关：**
    - Permutation Entropy: 越大说明系统越随机（接近白噪声）
    - Normalized Entropy: 0-1标准化，1表示完全随机
    - Forbidden Patterns: 越多说明确定性越强（混沌系统特征）

    **偏差相关：**
    - Relative Bias: 各模式的时间不对称偏好度
    - KL Divergence: 衡量正向/反向分布差异的信息论指标

    **序型频率相关：**
    - Pattern Frequencies: 每个窗口对应的序型在整个序列中的频率（可merge回原始序列）
    - Pattern Frequency All: 所有m!种序型的频率汇总统计

    序列对齐说明：
    ----------------
    - local_irreversibility和relative_bias的长度与输入序列相同（n）
    - 前m-1个值为NaN（无法构成完整窗口）
    - 第i个值（i>=m）对应从位置i-m+1开始的窗口

    使用示例：
    ----------
    >>> import numpy as np
    >>> import rust_pyfunc as rp
    >>> import matplotlib.pyplot as plt
    >>>
    >>> # 计算详细指标
    >>> data = np.random.randn(10000)
    >>> result = rp.time_irreversibility_static_detailed(data, m=5)
    >>>
    >>> # 输出核心指标
    >>> print(f"I_ord: {result['i_ord']:.4f}")
    >>> print(f"序型熵: {result['permutation_entropy']:.4f}")
    >>> print(f"local_irreversibility长度: {len(result['local_irreversibility'])}")
    >>> print(f"前5个值（NaN）: {result['local_irreversibility'][:5]}")
    >>> print(f"第6个值（第一个有效值）: {result['local_irreversibility'][5]:.6f}")
    >>>
    >>> # 比较绝对值和带符号版本
    >>> print(f"local_irreversibility_signed长度: {len(result['local_irreversibility_signed'])}")
    >>> print(f"范围: [{result['local_irreversibility_signed'].min():.6f}, {result['local_irreversibility_signed'].max():.6f}]")
    >>> print(f"正向模式个数: {(result['local_irreversibility_signed'] > 0).sum()}")
    >>> print(f"负向模式个数: {(result['local_irreversibility_signed'] < 0).sum()}")

    性能说明：
    ----------
    - 比简略版慢约2-3倍（需要计算所有衍生指标）
    - 13万条数据，m=5：约0.3-0.5秒
    - 额外开销主要来自模式枚举和统计计算
    """
    ...

def time_irreversibility_transfer_simple(
    data: NDArray[np.float64],
    m: int = 5
) -> float:
    """转移版-简略版：计算转移不可逆指标 I_trans。

    该函数基于序型马尔可夫链度量时间序列的转移不可逆性，只返回核心指标I_trans。
    通过比较正向和反向转移概率流的差异，捕捉动态演化层面的时间箭头。

    计算原理：
    ----------
    在序型序列 {π_t} 基础上：
    1. 构建转移计数矩阵 N_ij（从序型i到j的转移次数）
    2. 估计转移概率 P_ij = N_ij / Σ_k N_ik
    3. 计算经验平稳分布 μ_i = Σ_j N_ij / Σ_{k,l} N_kl
    4. 计算概率流差 J_ij = μ_i P_ij - μ_j P_ji
    5. 不可逆指标：I_trans = 0.5 * Σ|J_ij|

    物理意义：
    ----------
    - J_ij 表示正向时间i→j的净概率流
    - I_trans 越大，说明动态路径越不对称
    - 相比静态I_ord，能捕捉更丰富的动力学信息

    参数：
    -----
    data : numpy.ndarray
        输入的时间序列数据（一维浮点数数组）
    m : int, optional
        嵌入维度（窗口大小），默认值为5（m>=2）

    返回值：
    -------
    float
        转移不可逆性指标 I_trans（取值范围[0,1]）
        - 0：完全可逆（转移对称）
        - 1：完全不可逆（路径不对称）

    计算复杂度：
    ------------
    - 时间：O(T * (m log m + (m!)²))，T为数据长度
    - 空间：O((m!)²)，需要存储转移矩阵

    使用示例：
    ----------
    >>> import numpy as np
    >>> import rust_pyfunc as rp
    >>>
    >>> # 生成时间序列
    >>> data = np.random.randn(20000)
    >>>
    >>> # 计算转移不可逆指标（m=5）
    >>> i_trans = rp.time_irreversibility_transfer_simple(data, m=5)
    >>> print(f"转移不可逆性: {i_trans:.4f}")
    >>>
    >>> # 测试不同嵌入维度
    >>> for m in [2, 3, 4, 5]:
    ...     i_trans = rp.time_irreversibility_transfer_simple(data, m=m)
    ...     print(f"m={m}: I_trans = {i_trans:.4f}")

    性能说明：
    ----------
    - 13万条数据，m=5：约0.01秒
    - 比静态版稍慢（需要构建转移矩阵）
    - 内存使用：约 (m!)² * 8 字节
    """
    ...

def time_irreversibility_transfer_detailed(
    data: NDArray[np.float64],
    m: int = 5
) -> Dict[str, Any]:
    """转移版-详细版：计算所有转移时间不可逆性指标。

    该函数提供完整的转移不可逆性分析，返回包含多种动力学指标的详细结果。
    除了核心指标I_trans外，还包括熵率、平稳分布、转移矩阵等动态系统指标。

    计算指标：
    ----------
    1. **I_trans**: 转移不可逆指标（0.5 * Σ|μ_i P_ij - μ_j P_ji|）
    2. **Entropy Rate**: 熵率 h = -Σμ_i ΣP_ij log P_ij
    3. **Transition Entropy**: 转移熵（等同于熵率）
    4. **Stationary Distribution**: 经验平稳分布 μ_i
    5. **Transition Matrix**: 转移概率矩阵 P_ij
    6. **Flow Differences**: 概率流差矩阵 J_ij = μ_i P_ij - μ_j P_ji

    参数：
    -----
    data : numpy.ndarray
        输入的时间序列数据（一维浮点数数组）
    m : int, optional
        嵌入维度（窗口大小），默认值为5（m>=2）

    返回值：
    -------
    Dict[str, Any]
        包含所有指标的字典：
        - 'i_trans': float - 转移不可逆指标
        - 'entropy_rate': float - 熵率
        - 'transition_entropy': float - 转移熵
        - 'stationary_distribution': NDArray[np.float64] - 平稳分布（长度m!）
        - 'transition_matrix': NDArray[np.float64] - 转移矩阵（形状m!×m!）
        - 'flow_differences': NDArray[np.float64] - 概率流差绝对值（长度n，前m个为NaN）
        - 'flow_direction': NDArray[np.float64] - 概率流方向差值（长度n，前m个为NaN，有正负）

    指标解读：
    ----------
    **不可逆性相关：**
    - I_trans: 转移路径的时间不对称程度
    - Flow Differences: 各转移对的净概率流绝对值（>=0）
    - Flow Direction: 各转移对的有符号差值（正向>反向为正）

    **动力学相关：**
    - Entropy Rate: 系统演化的不确定性
    - Stationary Distribution: 长期行为中各状态的占比
    - Transition Matrix: 一步转移概率结构

    **复杂性度量：**
    - 高熵率：系统行为不可预测性强
    - 均匀平稳分布：各状态出现频率相似
    - 稀疏转移矩阵：确定性转移模式

    序列对齐说明：
    ----------------
    - flow_differences的长度与输入序列相同（n）
    - 前m个值为NaN（无法构成第一个转移）
    - 第i个值（i>m）对应从序型π_{i-m}到π_{i-m+1}的转移

    使用示例：
    ----------
    >>> import numpy as np
    >>> import rust_pyfunc as rp
    >>>
    >>> # 计算详细指标
    >>> data = np.random.randn(50000)
    >>> result = rp.time_irreversibility_transfer_detailed(data, m=5)
    >>>
    >>> # 输出核心指标
    >>> print(f"I_trans: {result['i_trans']:.4f}")
    >>> print(f"熵率: {result['entropy_rate']:.4f}")
    >>> print(f"flow_differences长度: {len(result['flow_differences'])}")
    >>> print(f"前5个值（NaN）: {result['flow_differences'][:5]}")
    >>> print(f"第6个值（第一个有效值）: {result['flow_differences'][5]:.6f}")
    >>>
    >>> # 比较绝对值和带符号版本
    >>> print(f"flow_direction长度: {len(result['flow_direction'])}")
    >>> print(f"flow_direction范围: [{result['flow_direction'].min():.6f}, {result['flow_direction'].max():.6f}]")
    >>> print(f"正向流个数: {(result['flow_direction'] > 0).sum()}")
    >>> print(f"反向流个数: {(result['flow_direction'] < 0).sum()}")

    性能说明：
    ----------
    - 比简略版慢约3-5倍（需要计算矩阵和熵）
    - 13万条数据，m=5：约0.3-0.5秒
    - 内存使用：约 (m!)² * 8 * 3 字节（矩阵、平稳分布、流差）
    - 对于m=5（120种模式），约需350KB内存
    - 对于m=6（720种模式），约需12MB内存
    - 对于m=7（5040种模式），约需600MB内存
    """
    ...