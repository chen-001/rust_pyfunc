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