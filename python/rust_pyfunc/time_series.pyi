"""时间序列分析函数类型声明"""
from typing import List, Optional, Tuple
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
    """计算滚动窗口统计量。
    
    参数说明：
    ----------
    times : NDArray[np.float64]
        时间数组
    values : NDArray[np.float64]
        数值数组
    window_size : float
        窗口大小
    stat_type : str
        统计类型（"mean", "std", "min", "max"等）
    include_current : bool
        是否包含当前点
        
    返回值：
    -------
    NDArray[np.float64]
        滚动统计量数组
    """
    ...