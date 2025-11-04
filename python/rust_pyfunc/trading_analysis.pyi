"""交易分析函数类型声明"""
from typing import List, Optional, Tuple
import numpy as np
from numpy.typing import NDArray

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
    """
    ...

def analyze_retreat_advance_v2(
    trade_times: List[float],
    trade_prices: List[float], 
    trade_volumes: List[float],
    trade_flags: List[float],
    orderbook_times: List[float],
    orderbook_prices: List[float],
    orderbook_volumes: List[float],
    volume_percentile: Optional[float] = 99.0,
    time_window_minutes: Optional[float] = 1.0,
    breakthrough_threshold: Optional[float] = 0.0,
    dedup_time_seconds: Optional[float] = 30.0,
    find_local_lows: Optional[bool] = False,
    interval_mode: Optional[str] = "full"
) -> Tuple[List[float], List[float], List[float], List[float], List[float], List[float], List[float], List[float], List[float]]:
    """分析股票交易中的"以退为进"或"以进为退"现象（纳秒版本）
    
    该函数分析两种现象：
    1. "以退为进"（find_local_lows=False）：价格触及局部高点后回落，然后在该价格的异常大卖单量消失后成功突破该价格
    2. "以进为退"（find_local_lows=True）：价格跌至局部低点后反弹，然后在该价格的异常大买单量消失后成功跌破该价格
    
    这是analyze_retreat_advance函数的改进版本，专门为处理纳秒级时间戳而优化，并包含局部极值点去重功能。
    
    参数说明：
    ----------
    trade_times : List[float]
        逐笔成交数据的时间戳序列（纳秒时间戳）
    trade_prices : List[float]
        逐笔成交数据的价格序列
    trade_volumes : List[float]
        逐笔成交数据的成交量序列
    trade_flags : List[float]
        逐笔成交数据的标志序列（买卖方向），66表示主动买入，83表示主动卖出
    orderbook_times : List[float]
        盘口快照数据的时间戳序列（纳秒时间戳）
    orderbook_prices : List[float]
        盘口快照数据的价格序列
    orderbook_volumes : List[float]
        盘口快照数据的挂单量序列
    volume_percentile : Optional[float], default=99.0
        异常大挂单量的百分位数阈值，默认为99.0（即前1%）
    time_window_minutes : Optional[float], default=1.0
        检查异常大挂单量的时间窗口（分钟），默认为1.0分钟
    breakthrough_threshold : Optional[float], default=0.0
        突破阈值（百分比），默认为0.0（即只要高于局部高点任何幅度都算突破）
        例如：0.1表示需要高出局部高点0.1%才算突破
    dedup_time_seconds : Optional[float], default=30.0
        去重时间阈值（秒），默认为30.0。相同价格且时间间隔小于此值的局部极值点将被视为重复
    find_local_lows : Optional[bool], default=False
        是否查找局部低点，默认为False（查找局部高点）。
        当为True时，分析"以进为退"现象：价格跌至局部低点后反弹，在该价格的异常大买单量消失后成功跌破该价格
    interval_mode : Optional[str], default="full"
        区间选择模式，默认为"full"。可选值：
        - "full": 完整的"高点-低点-高点"过程（默认）
        - "retreat": "高点-低点"过程，从局部高点到该段中的最低点
        - "advance": "低点-高点"过程，从局部低点到该段中的最高点
    
    返回值：
    -------
    Tuple[List[float], List[float], List[float], List[float], List[float], List[float], List[float], List[float], List[float]]
        包含9个列表的元组：
        - 过程期间的成交量
        - 局部极值价格在盘口上时间最近的挂单量
        - 过程开始后指定时间窗口内的成交量
        - 过程期间的主动买入成交量占比
        - 过程期间的价格种类数
        - 过程期间价格相对局部极值的最大变化比例（高点模式为最大下降比例，低点模式为最大上升比例）
        - 过程持续时间（秒）
        - 过程开始时间（纳秒时间戳）
        - 局部极值的价格
    
    特点：
    ------
    1. 纳秒级时间戳处理 - 专门优化处理纳秒级别的高精度时间戳
    2. 双模式分析 - 支持局部高点（以退为进）和局部低点（以进为退）两种分析模式
    3. 改进的局部极值识别 - 使用更准确的算法识别价格局部高点或低点
    4. 可配置的局部极值去重功能 - 对相同价格且时间接近的局部极值进行去重，时间阈值可自定义
    5. 智能挂单量检测 - 根据模式自动检测卖单（高点模式）或买单（低点模式）的异常大挂单量
    6. 可配置的突破条件 - 通过breakthrough_threshold参数自定义突破阈值
    7. 时间窗口控制 - 设置4小时最大搜索窗口，避免无限搜索
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
    flag_filter: str = "ignore",
    only_after: bool = False,
    large_to_large: bool = False
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
    only_after : bool, default=False
        是否只计算大单与其之后的目标订单的时间间隔，True时忽略大单之前的订单
    large_to_large : bool, default=False
        是否计算大单与大单之间的时间间隔，True时目标订单改为大单而非小单

    返回值：
    -------
    numpy.ndarray
        浮点数数组，与输入volumes等长。对于大单，返回其与临近目标订单的时间间隔均值（秒）；
        对于非大单，返回NaN。
    """
    ...

def order_contamination(
    exchtime: NDArray[np.int64],
    order: NDArray[np.int64],
    volume: NDArray[np.int64],
    top_percentile: float = 10.0,
    time_window_seconds: float = 1.0
) -> NDArray[np.int64]:
    """订单浸染函数（高性能优化版本）
    
    根据订单成交量找到前top_percentile%的大单，然后将这些大单附近时间窗口内的非大单
    订单编号改为最近大单的订单编号，模拟大单浸染附近小单的效果。
    
    该版本经过大幅性能优化，使用时间索引排序和二分查找算法，
    处理速度相比原版本提升数十倍。
    
    参数说明：
    ----------
    exchtime : NDArray[np.int64]
        成交时间数组（纳秒）
    order : NDArray[np.int64]
        订单编号数组
    volume : NDArray[np.int64]
        成交量数组
    top_percentile : float, default=10.0
        大单百分比阈值（1-100），默认10.0表示前10%
    time_window_seconds : float, default=1.0
        时间窗口（秒），默认1秒
        
    返回值：
    -------
    NDArray[np.int64]
        浸染后的订单编号数组
        
    性能特点：
    ----------
    - 时间复杂度：O(n log n + m * log n)，其中n为总记录数，m为大单数
    - 空间复杂度：O(n)
    - 处理速度：约700万条/秒（实际股票数据）
    """
    ...

def order_contamination_parallel(
    exchtime: NDArray[np.int64],
    order: NDArray[np.int64],
    volume: NDArray[np.int64],
    top_percentile: float = 10.0,
    time_window_seconds: float = 1.0
) -> NDArray[np.int64]:
    """订单浸染函数（5核心并行版本）
    
    根据订单成交量找到前top_percentile%的大单，然后将这些大单附近时间窗口内的非大单
    订单编号改为最近大单的订单编号，模拟大单浸染附近小单的效果。
    
    此版本使用5个CPU核心进行并行计算，适用于大规模数据处理。
    
    参数说明：
    ----------
    exchtime : NDArray[np.int64]
        成交时间数组（纳秒）
    order : NDArray[np.int64]
        订单编号数组
    volume : NDArray[np.int64]
        成交量数组
    top_percentile : float, default=10.0
        大单百分比阈值（1-100），默认10.0表示前10%
    time_window_seconds : float, default=1.0
        时间窗口（秒），默认1秒
        
    返回值：
    -------
    NDArray[np.int64]
        浸染后的订单编号数组
        
    注意：
    -----
    该函数固定使用5个CPU核心进行并行计算。对于小规模数据，
    串行版本order_contamination可能更快。
    """
    ...

def order_contamination_bilateral(
    exchtime: NDArray[np.int64],
    bid_order: NDArray[np.int64],
    ask_order: NDArray[np.int64],
    volume: NDArray[np.int64],
    top_percentile: float = 10.0,
    time_window_seconds: float = 1.0
) -> Tuple[NDArray[np.int64], NDArray[np.int64]]:
    """双边订单浸染函数
    
    分别处理买单和卖单的浸染过程：
    - 对于每个大买单，将时间临近的卖单非大单设置为浸染单
    - 对于每个大卖单，将时间临近的买单非大单设置为浸染单
    
    该函数基于原order_contamination函数的核心算法，但同时处理买卖双边订单，
    实现更精细的订单浸染分析。
    
    参数说明：
    ----------
    exchtime : NDArray[np.int64]
        成交时间数组（纳秒）
    bid_order : NDArray[np.int64]
        买单编号数组
    ask_order : NDArray[np.int64]
        卖单编号数组
    volume : NDArray[np.int64]
        成交量数组
    top_percentile : float, default=10.0
        大单百分比阈值（1-100），默认10.0表示前10%
    time_window_seconds : float, default=1.0
        时间窗口（秒），默认1秒
        
    返回值：
    -------
    Tuple[NDArray[np.int64], NDArray[np.int64]]
        返回元组：(浸染后的买单编号数组, 浸染后的卖单编号数组)
        
    算法逻辑：
    ----------
    1. 分别计算买单和卖单的大单阈值（基于各自的成交量分布）
    2. 对大买单浸染过程：
       - 识别所有大买单位置
       - 对每个卖单非大单，查找时间窗口内最近的大买单
       - 将该卖单编号替换为最近大买单编号
    3. 对大卖单浸染过程：
       - 识别所有大卖单位置  
       - 对每个买单非大单，查找时间窗口内最近的大卖单
       - 将该买单编号替换为最近大卖单编号
    4. 返回浸染后的买单和卖单编号数组
    
    使用场景：
    ----------
    适用于分析大单对市场的双边影响，特别是：
    - 大买单对卖方流动性的影响
    - 大卖单对买方流动性的影响
    - 买卖双方订单的相互浸染效应
    
    性能特点：
    ----------
    - 使用二分查找优化时间窗口搜索
    - 预计算大单位置避免重复判断
    - 时间复杂度：O(n log n + m * log n)，其中n为总记录数，m为大单数
    """
    ...

def trade_peak_analysis(
    exchtime: NDArray[np.int64],
    volume: NDArray[np.float64],
    flag: NDArray[np.int32],
    top_tier1: float,
    top_tier2: float,
    time_window: float,
    flag_different: bool,
    with_forth: bool
) -> Tuple[NDArray[np.float64], List[str]]:
    """交易高峰模式分析函数
    
    该函数用于分析交易数据中的高峰模式，包括：
    1. 识别成交量的局部高峰(根据top_tier1百分比)
    2. 在每个高峰的时间窗口内识别小峰(根据top_tier2百分比)
    3. 计算17个统计指标来描述高峰-小峰的模式特征
    
    参数说明：
    ----------
    exchtime : NDArray[np.int64]
        交易时间数组(纳秒时间戳)
    volume : NDArray[np.float64]
        成交量数组
    flag : NDArray[np.int32]
        交易标志数组(主动买入/卖出标志)
    top_tier1 : float
        高峰识别的百分比阈值(例如0.01表示前1%的大成交量)
    top_tier2 : float
        小峰识别的百分比阈值(例如0.10表示前10%的大成交量)
    time_window : float
        时间窗口大小(秒)
    flag_different : bool
        是否只考虑与高峰flag不同的小峰
    with_forth : bool
        是否同时考虑高峰前后的时间窗口
        
    返回值：
    -------
    Tuple[NDArray[np.float64], List[str]]
        第一个元素：N行17列的数组，每行对应一个局部高峰的17个统计指标
        第二个元素：包含17个特征名称的字符串列表
        
        17个特征列分别为：
        列0: 小峰成交量总和比值
        列1: 小峰平均成交量比值  
        列2: 小峰个数
        列3: 时间间隔均值秒
        列4: 成交量时间相关系数
        列5: DTW距离
        列6: 成交量变异系数
        列7: 成交量偏度
        列8: 成交量峰度
        列9: 成交量趋势
        列10: 成交量自相关
        列11: 时间变异系数
        列12: 时间偏度
        列13: 时间峰度
        列14: 时间趋势
        列15: 时间自相关
        列16: 成交量加权时间距离
        
    使用示例：
    ---------
    >>> result_matrix, feature_names = trade_peak_analysis(...)
    >>> import pandas as pd
    >>> df = pd.DataFrame(result_matrix, columns=feature_names)
    """
    ...

def order_neighborhood_analysis(
    ask_order: NDArray[np.int64],
    bid_order: NDArray[np.int64],
    volume: NDArray[np.float64],
    exchtime: NDArray[np.int64],
    neighborhood_type: str = "fixed",
    fixed_range: int = 1000,
    percentage_range: float = 10.0,
    num_threads: int = 8,
    timeout_ms: Optional[int] = None
) -> Optional[Tuple[NDArray[np.float64], List[str]]]:
    """订单邻域分析函数
    
    分析订单编号的邻域关系，计算每个订单与其邻居订单的成交量、时间差值等统计指标。
    一个订单编号可能对应多个不同的交易时间，聚合时以最晚的交易时间为准。
    
    参数说明：
    ----------
    ask_order : NDArray[np.int64]
        卖单订单编号数组
    bid_order : NDArray[np.int64]
        买单订单编号数组
    volume : NDArray[np.float64]
        成交量数组
    exchtime : NDArray[np.int64]
        交易时间数组（纳秒单位）
    neighborhood_type : str, default="fixed"
        邻域类型，可选值：
        - "fixed": 固定范围模式，查找编号±fixed_range范围内的订单
        - "percentage": 百分比模式，查找编号±percentage_range%范围内的订单
    fixed_range : int, default=1000
        固定范围值，当neighborhood_type="fixed"时使用
    percentage_range : float, default=10.0
        百分比范围值，当neighborhood_type="percentage"时使用
    num_threads : int, default=8
        并行线程数量，设置为1时使用串行处理，大于1时使用并行处理
    timeout_ms : Optional[int], default=None
        超时时间（毫秒），当函数运行时间超过此值时，自动返回None。
        如果为None，则不设置超时限制
        
    返回值：
    -------
    Optional[Tuple[NDArray[np.float64], List[str]]]
        如果函数成功完成，返回包含两个元素的元组：
        - 第一个元素：N行18列的分析结果矩阵
        - 第二个元素：包含18个特征名称的字符串列表
        
        如果函数运行超时，返回None
        
        18个特征列分别为：
        列0: 订单编号 - 订单ID
        列1: 订单类型 - 订单类型（0=买单，1=卖单）
        列2: 同向成交量比 - 同方向邻居成交总量与本订单成交量的比值
        列3: 异向成交量比 - 不同方向邻居成交总量与本订单成交量的比值
        列4: 同向均量比 - 同方向邻居平均成交量与本订单成交量的比值
        列5: 异向均量比 - 不同方向邻居平均成交量与本订单成交量的比值
        列6: 总成交量比 - 所有邻居成交总量与本订单成交量的比值
        列7: 总均量比 - 所有邻居平均成交量与本订单成交量的比值
        列8: 同向邻居数 - 同方向邻居数量
        列9: 异向邻居数 - 不同方向邻居数量
        列10: 同向时差总和 - 同方向邻居时间差值总和（秒）
        列11: 异向时差总和 - 不同方向邻居时间差值总和（秒）
        列12: 同向时差均值 - 同方向邻居时间差值均值（秒）
        列13: 异向时差均值 - 不同方向邻居时间差值均值（秒）
        列14: 同向时差量相关 - 同方向邻居时间差值与成交量的相关系数
        列15: 异向时差量相关 - 不同方向邻居时间差值与成交量的相关系数
        列16: 同向时间量相关 - 同方向邻居成交时间与成交量的相关系数
        列17: 异向时间量相关 - 不同方向邻居成交时间与成交量的相关系数
        
    算法说明：
    ----------
    1. 数据聚合：按订单编号聚合成交量，每个订单取最晚的交易时间
    2. 邻域定义：根据neighborhood_type参数选择邻域规则
    3. 指标计算：分别计算同方向和不同方向邻居的各项统计指标
    4. 相关性分析：计算时间差值与成交量的皮尔逊相关系数
    
    使用示例：
    ---------
    >>> import numpy as np
    >>> from rust_pyfunc import order_neighborhood_analysis
    >>> 
    >>> # 示例数据
    >>> ask_order = np.array([1001, 1002, 0, 1003], dtype=np.int64)
    >>> bid_order = np.array([0, 0, 2001, 0], dtype=np.int64)
    >>> volume = np.array([100.0, 150.0, 200.0, 120.0])
    >>> exchtime = np.array([1000000000, 1001000000, 1002000000, 1003000000], dtype=np.int64)
    >>> 
    >>> # 固定范围模式
    >>> result_matrix, feature_names = order_neighborhood_analysis(
    ...     ask_order, bid_order, volume, exchtime,
    ...     neighborhood_type="fixed", fixed_range=1000
    ... )
    >>> 
    >>> # 百分比模式
    >>> result_matrix, feature_names = order_neighborhood_analysis(
    ...     ask_order, bid_order, volume, exchtime,
    ...     neighborhood_type="percentage", percentage_range=10.0
    ... )
    >>> 
    >>> # 转换为DataFrame
    >>> import pandas as pd
    >>> df = pd.DataFrame(result_matrix, columns=feature_names)
    """
    ...

def analyze_trade_records(
    volume: NDArray[np.float64],
    exchtime: NDArray[np.float64],
    price: NDArray[np.float64],
    flag: NDArray[np.int32],
    min_count: int = 100,
    use_flag: str = "ignore"
) -> Tuple[NDArray[np.float64], List[str]]:
    """成交记录分析函数
    
    对每条成交记录，找到与其volume相同的其他成交记录，计算时间间隔和价格分位数等统计指标。
    
    参数说明：
    ----------
    volume : NDArray[np.float64]
        成交量数组
    exchtime : NDArray[np.float64]
        成交时间数组（单位：秒）
    price : NDArray[np.float64]
        成交价格数组
    flag : NDArray[np.int32]
        主买卖标志数组 (66=主买，83=主卖)
    min_count : int, default=100
        最小成交记录数量阈值，低于此值的volume将返回NaN
    use_flag : str, default="ignore"
        flag过滤模式：
        - "same": 只考虑与当前记录flag相同的记录
        - "diff": 只考虑与当前记录flag不同的记录 
        - "ignore": 忽略flag，考虑所有记录
        
    返回值：
    -------
    Tuple[NDArray[np.float64], List[str]]
        第一个元素：N行22列的分析结果矩阵
        第二个元素：包含22个特征名称的字符串列表
        
        22个特征列分别为：
        列0-11: 时间间隔相关指标
        - nearest_time_gap: 最近成交记录的时间间隔（秒）
        - avg_time_gap_1pct: 最近1%成交记录的平均时间间隔
        - avg_time_gap_2pct: 最近2%成交记录的平均时间间隔
        - ...依次类推到50%和所有记录
        
        列12-21: 价格分位数指标
        - price_percentile_10pct: 在最近10%记录中的价格分位数
        - price_percentile_20pct: 在最近20%记录中的价格分位数
        - ...依次类推到100%所有记录
    """
    ...

def analyze_order_records(
    volume: NDArray[np.float64],
    exchtime: NDArray[np.float64],
    price: NDArray[np.float64],
    flag: NDArray[np.int32],
    ask_order: NDArray[np.int64],
    bid_order: NDArray[np.int64],
    min_count: int = 100,
    use_flag: str = "ignore"
) -> Tuple[NDArray[np.float64], List[str]]:
    """订单聚合分析函数
    
    将成交记录按订单聚合后，对每个订单找到与其成交总量相同的其他订单，
    计算时间间隔和价格分位数等统计指标。
    
    参数说明：
    ----------
    volume : NDArray[np.float64]
        成交量数组
    exchtime : NDArray[np.float64]
        成交时间数组（单位：秒）
    price : NDArray[np.float64]
        成交价格数组
    flag : NDArray[np.int32]
        主买卖标志数组 (66=主买，83=主卖)
    ask_order : NDArray[np.int64]
        卖单订单编号数组
    bid_order : NDArray[np.int64]
        买单订单编号数组
    min_count : int, default=100
        最小订单数量阈值，低于此值的volume将返回NaN
    use_flag : str, default="ignore"
        订单类型过滤模式：
        - "same": 买单只与买单比较，卖单只与卖单比较
        - "diff": 买单只与卖单比较，卖单只与买单比较
        - "ignore": 买单卖单混合比较
        
    返回值：
    -------
    Tuple[NDArray[np.float64], List[str]]
        第一个元素：N行22列的分析结果矩阵（与成交记录数量相同）
        第二个元素：包含22个特征名称的字符串列表
        
        订单聚合规则：
        - 时间：取最后一次成交时间作为订单时间
        - 成交量：各成交记录成交量求和
        - 价格：成交量加权平均价格
        
        22个特征列与analyze_trade_records相同
    """
    ...

def analyze_trade_records_fast(
    volume: NDArray[np.float64],
    exchtime: NDArray[np.float64],
    price: NDArray[np.float64],
    flag: NDArray[np.int32],
    min_count: int = 100,
    use_flag: str = "ignore"
) -> Tuple[NDArray[np.float64], List[str]]:
    """高性能版本的成交记录分析函数
    
    相比analyze_trade_records，该版本进行了大量性能优化：
    1. 使用u64整数键替代字符串键
    2. 预过滤和缓存有效的volume组
    3. 批量计算减少重复操作
    4. 使用sort_unstable和binary_search优化
    5. 减少内存分配和复制
    
    参数和返回值与analyze_trade_records完全相同。
    预期性能提升：50-100倍
    """
    ...

def analyze_order_records_fast(
    volume: NDArray[np.float64],
    exchtime: NDArray[np.float64],
    price: NDArray[np.float64],
    flag: NDArray[np.int32],
    ask_order: NDArray[np.int64],
    bid_order: NDArray[np.int64],
    min_count: int = 100,
    use_flag: str = "ignore"
) -> Tuple[NDArray[np.float64], List[str]]:
    """高性能版本的订单聚合分析函数
    
    相比analyze_order_records，该版本进行了大量性能优化：
    1. 优化订单聚合算法
    2. 使用更高效的数据结构
    3. 批量处理和预计算
    4. 减少重复的映射构建
    
    参数和返回值与analyze_order_records完全相同。
    预期性能提升：50-100倍
    """
    ...

def analyze_trade_records_turbo(
    volume: NDArray[np.float64],
    exchtime: NDArray[np.float64],
    price: NDArray[np.float64],
    flag: NDArray[np.int32],
    min_count: int = 100,
    use_flag: str = "ignore"
) -> Tuple[NDArray[np.float64], List[str]]:
    """终极性能版本的成交记录分析函数
    
    使用O(n log n)算法复杂度的革命性优化：
    1. 时间索引预计算：对每个volume组构建按时间排序的索引
    2. 二分查找最近邻：使用binary_search快速定位最近记录
    3. 批量处理：一次性计算所有指标
    4. 采样限制：限制最大样本数以控制计算量
    5. 内存局部性优化：减少缓存不命中
    
    算法复杂度：从O(n²)降低到O(n log n + k*log n)
    其中k为每组的平均记录数，通常k << n
    
    预期性能提升：100-1000倍，目标1秒内完成全日数据分析
    
    参数和返回值与analyze_trade_records完全相同。
    """
    ...

def analyze_order_records_turbo(
    volume: NDArray[np.float64],
    exchtime: NDArray[np.float64],
    price: NDArray[np.float64],
    flag: NDArray[np.int32],
    ask_order: NDArray[np.int64],
    bid_order: NDArray[np.int64],
    min_count: int = 100,
    use_flag: str = "ignore"
) -> Tuple[NDArray[np.float64], List[str]]:
    """终极性能版本的订单聚合分析函数
    
    使用O(n log n)算法复杂度的革命性优化：
    1. 高效订单聚合：优化的HashMap算法
    2. 时间索引预计算：对每个volume组构建按时间排序的索引
    3. 二分查找最近邻：使用binary_search快速定位最近记录
    4. 采样限制：限制最大样本数以控制计算量
    5. 批量指标计算：一次性计算所有22个指标
    
    算法复杂度：从O(n²)降低到O(n log n + k*log n)
    其中k为每组的平均记录数，通常k << n
    
    预期性能提升：100-1000倍，目标1秒内完成全日数据分析
    
    参数和返回值与analyze_order_records完全相同。
    """
    ...

def calculate_trade_time_gap_and_price_percentile_ultra_sorted(
    volume: NDArray[np.float64],
    exchtime: NDArray[np.float64],
    price: NDArray[np.float64],
    flag: NDArray[np.int32],
    min_count: int = 100,
    use_flag: str = "ignore"
) -> Tuple[NDArray[np.float64], List[str]]:
    """计算成交记录的时间间隔和价格分位数指标（Ultra Sorted优化算法）
    
    针对相同volume的成交记录，计算22个量化指标：
    
    📊 计算指标详细说明：
    ================
    🕐 时间间隔指标（12个，单位：秒）：
    - nearest_time_gap: 最近一笔相同volume交易的时间间隔
    - avg_time_gap_1pct到avg_time_gap_50pct: 最近1%-50%交易的平均时间间隔
    - avg_time_gap_all: 所有相同volume交易的平均时间间隔
    
    💰 价格分位数指标（10个）：
    - price_percentile_10pct到price_percentile_all: 当前价格在最近10%-100%交易中的分位数排名
    
    ⚡ 算法优化特性：
    - 彻底消除O(n²)复杂度，达到O(n log n)
    - 利用预排序数据避免重复排序开销
    - 批量预计算相同volume组的共享数据
    - 使用双指针和二分查找加速时间定位
    
    📋 数据要求：
    输入数据必须已按volume和exchtime双重排序：
    df.sort_values(['volume', 'exchtime'])
    
    ⚡ 性能表现：
    算法复杂度：O(n log n)
    预期性能：比turbo版本快5-10倍
    全天数据（13万条）处理时间：约15秒
    
    参数：
    --------
    volume : NDArray[np.float64]
        成交量数组（需已按volume,time排序）
    exchtime : NDArray[np.float64] 
        交易时间数组（纳秒时间戳，函数内部自动转换为秒，需已排序）
    price : NDArray[np.float64]
        成交价格数组
    flag : NDArray[np.int32]
        交易标志数组（66=买，83=卖）
    min_count : int, default=100
        计算指标所需的最少同volume记录数
    use_flag : str, default="ignore"
        标志过滤方式：
        - "same": 只考虑相同买卖方向的交易
        - "diff": 只考虑不同买卖方向的交易  
        - "ignore": 考虑所有方向的交易
    
    返回：
    -------
    result : NDArray[np.float64]
        形状为(n, 22)的结果数组，包含上述22个量化指标
    columns : List[str]  
        22个指标的中文列名列表
    """
    ...

def calculate_order_time_gap_and_price_percentile_ultra_sorted(
    volume: NDArray[np.float64],
    exchtime: NDArray[np.float64],
    price: NDArray[np.float64],
    flag: NDArray[np.int32],
    ask_order: NDArray[np.int64],
    bid_order: NDArray[np.int64],
    min_count: int = 100,
    use_flag: str = "ignore"
) -> Tuple[NDArray[np.float64], List[str]]:
    """计算订单聚合后的时间间隔和价格分位数指标（Ultra Sorted优化算法）
    
    先将逐笔成交按订单号聚合，然后对聚合后的订单计算22个量化指标：
    
    📊 计算指标详细说明：
    ================
    🕐 时间间隔指标（12个，单位：秒）：
    - nearest_time_gap: 最近一个相同volume订单的时间间隔
    - avg_time_gap_1pct到avg_time_gap_50pct: 最近1%-50%订单的平均时间间隔
    - avg_time_gap_all: 所有相同volume订单的平均时间间隔
    
    💰 价格分位数指标（10个）：
    - price_percentile_10pct到price_percentile_all: 当前订单价格在最近10%-100%订单中的分位数排名
    
    🆔 订单标识指标（2个）：
    - 订单编号: 订单编号（买单使用bid_order，卖单使用ask_order）
    - 买单标识: 买单标识（1.0=买单，0.0=卖单）
    
    📦 订单聚合信息（3个）：
    - 订单总量: 订单的总volume（所有成交volume累加）
    - 订单时间: 订单的最后成交时间
    - 订单价格: 订单的加权平均价格
    
    🔄 订单聚合逻辑：
    - 按订单号和买卖方向(ask_order/bid_order + flag)聚合成交记录
    - 计算每个订单的总volume（累加）、加权平均价格、最后成交时间
    - 基于聚合后的订单数据计算上述27个指标（22个量化指标+5个信息列）
    
    ⚡ 算法优化特性：
    - 彻底消除O(n²)复杂度，达到O(n log n)
    - 高效的订单聚合HashMap算法
    - 利用预排序数据避免重复排序开销
    - 批量预计算相同volume组的共享数据
    - 使用双指针和二分查找加速时间定位
    
    📋 数据要求：
    输入数据必须已按volume和exchtime双重排序：
    df.sort_values(['volume', 'exchtime'])
    
    ⚡ 性能表现：
    算法复杂度：O(n log n)
    预期性能：比turbo版本快5-10倍
    全天数据（13万条）处理时间：约2.7秒
    
    参数：
    --------
    volume : NDArray[np.float64]
        成交量数组（需已按volume,time排序）
    exchtime : NDArray[np.float64]
        交易时间数组（纳秒时间戳，函数内部自动转换为秒，需已排序）
    price : NDArray[np.float64]
        成交价格数组
    flag : NDArray[np.int32]
        交易标志数组（66=买，83=卖）
    ask_order : NDArray[np.int64]
        卖单订单号数组
    bid_order : NDArray[np.int64]
        买单订单号数组
    min_count : int, default=100
        计算指标所需的最少同volume订单数
    use_flag : str, default="ignore"
        标志过滤方式：
        - "same": 只考虑相同买卖方向的订单
        - "diff": 只考虑不同买卖方向的订单
        - "ignore": 考虑所有方向的订单
        
    返回：
    -------
    result : NDArray[np.float64]
        形状为(n, 27)的结果数组，包含上述27个指标（22个量化指标+5个信息列）
    columns : List[str]
        27个指标的中文列名列表，具体含义如下：
        
        第1列：最近时间间隔（单位：秒）
        第2列：平均时间间隔_1%（单位：秒）
        第3列：平均时间间隔_2%（单位：秒）
        第4列：平均时间间隔_3%（单位：秒）
        第5列：平均时间间隔_4%（单位：秒）
        第6列：平均时间间隔_5%（单位：秒）
        第7列：平均时间间隔_10%（单位：秒）
        第8列：平均时间间隔_20%（单位：秒）
        第9列：平均时间间隔_30%（单位：秒）
        第10列：平均时间间隔_40%（单位：秒）
        第11列：平均时间间隔_50%（单位：秒）
        第12列：平均时间间隔_全部（单位：秒）
        第13列：价格分位数_10%（在最近10%记录中当前价格的分位数排名）
        第14列：价格分位数_20%（在最近20%记录中当前价格的分位数排名）
        第15列：价格分位数_30%（在最近30%记录中当前价格的分位数排名）
        第16列：价格分位数_40%（在最近40%记录中当前价格的分位数排名）
        第17列：价格分位数_50%（在最近50%记录中当前价格的分位数排名）
        第18列：价格分位数_60%（在最近60%记录中当前价格的分位数排名）
        第19列：价格分位数_70%（在最近70%记录中当前价格的分位数排名）
        第20列：价格分位数_80%（在最近80%记录中当前价格的分位数排名）
        第21列：价格分位数_90%（在最近90%记录中当前价格的分位数排名）
        第22列：价格分位数_全部（在所有记录中当前价格的分位数排名）
        第23列：订单编号（买单使用bid_order，卖单使用ask_order）
        第24列：买单标识（1.0=买单，0.0=卖单）
        第25列：订单总量（该订单的总volume）
        第26列：订单时间（该订单的最后成交时间）
        第27列：订单价格（该订单的加权平均价格）
    """
    ...

def calculate_order_time_gap_and_price_percentile_ultra_sorted_v2(
    volume: NDArray[np.float64],
    exchtime: NDArray[np.int64],
    price: NDArray[np.float64],
    flag: NDArray[np.int32],
    ask_order: NDArray[np.int64],
    bid_order: NDArray[np.int64],
    min_count: int = 100,
    use_flag: str = "ignore",
    num_buckets: int = 20
) -> Tuple[NDArray[np.float64], List[str]]:
    """计算订单聚合后的时间间隔和价格分位数指标（V2版本 - 基于订单类型筛选）
    
    本函数是calculate_order_time_gap_and_price_percentile_ultra_sorted的修正版本，
    主要区别在于use_flag参数的含义和订单聚合方式：
    
    📋 与原版本的主要区别：
    ========================
    1. **订单聚合方式**：
       - 原版本：基于(订单号, flag)进行聚合
       - V2版本：基于订单号进行聚合，订单类型由ask_order/bid_order决定
    
    2. **use_flag参数含义**：
       - 原版本：基于交易标志（66=主动买入，83=主动卖出）筛选
       - V2版本：基于订单类型（ask_order或bid_order）筛选
    
    3. **flag参数使用**：
       - 原版本：使用flag参数进行买卖分类
       - V2版本：flag参数被忽略，订单类型完全基于ask_order/bid_order
    
    🔄 订单聚合逻辑：
    ================
    - 卖单（ask_order != 0）：基于ask_order聚合成交记录
    - 买单（bid_order != 0）：基于bid_order聚合成交记录
    - 每个订单计算总volume（累加）、加权平均价格、最后成交时间
    
    📊 use_flag参数说明：
    ===================
    - "same": 买单只与买单比较，卖单只与卖单比较
    - "diff": 买单只与卖单比较，卖单只与买单比较
    - "ignore": 买单卖单混合比较
    
    参数：
    --------
    volume : NDArray[np.float64]
        成交量数组（需已按volume,time排序）
    exchtime : NDArray[np.float64]
        交易时间数组（纳秒时间戳，函数内部自动转换为秒，需已排序）
    price : NDArray[np.float64]
        成交价格数组
    flag : NDArray[np.int32]
        交易标志数组（在V2版本中被忽略）
    ask_order : NDArray[np.int64]
        卖单订单号数组
    bid_order : NDArray[np.int64]
        买单订单号数组
    min_count : int, default=100
        计算指标所需的最少同volume订单数
    use_flag : str, default="ignore"
        订单类型筛选方式：
        - "same": 只考虑相同类型的订单（买单与买单，卖单与卖单）
        - "diff": 只考虑不同类型的订单（买单与卖单，卖单与买单）
        - "ignore": 考虑所有类型的订单
        
    返回：
    -------
    result : NDArray[np.float64]
        形状为(n, 27)的结果数组，包含27个指标
    columns : List[str]
        27个指标的中文列名列表，具体含义如下：
        
        第1列：最近时间间隔（单位：秒）
        第2列：平均时间间隔_1%（单位：秒）
        第3列：平均时间间隔_2%（单位：秒）
        第4列：平均时间间隔_3%（单位：秒）
        第5列：平均时间间隔_4%（单位：秒）
        第6列：平均时间间隔_5%（单位：秒）
        第7列：平均时间间隔_10%（单位：秒）
        第8列：平均时间间隔_20%（单位：秒）
        第9列：平均时间间隔_30%（单位：秒）
        第10列：平均时间间隔_40%（单位：秒）
        第11列：平均时间间隔_50%（单位：秒）
        第12列：平均时间间隔_全部（单位：秒）
        第13列：价格分位数_10%（在最近10%记录中当前价格的分位数排名）
        第14列：价格分位数_20%（在最近20%记录中当前价格的分位数排名）
        第15列：价格分位数_30%（在最近30%记录中当前价格的分位数排名）
        第16列：价格分位数_40%（在最近40%记录中当前价格的分位数排名）
        第17列：价格分位数_50%（在最近50%记录中当前价格的分位数排名）
        第18列：价格分位数_60%（在最近60%记录中当前价格的分位数排名）
        第19列：价格分位数_70%（在最近70%记录中当前价格的分位数排名）
        第20列：价格分位数_80%（在最近80%记录中当前价格的分位数排名）
        第21列：价格分位数_90%（在最近90%记录中当前价格的分位数排名）
        第22列：价格分位数_全部（在所有记录中当前价格的分位数排名）
        第23列：订单编号（来自ask_order或bid_order）
        第24列：买单标识（1.0=买单，0.0=卖单）
        第25列：订单总量（该订单的总volume）
        第26列：订单时间（该订单的最后成交时间）
        第27列：订单价格（该订单的加权平均价格）
    """
    ...

def calculate_order_time_gap_and_price_percentile_ultra_sorted_v3(
    volume: NDArray[np.float64],
    exchtime: NDArray[np.float64],
    price: NDArray[np.float64],
    flag: NDArray[np.int32],
    ask_order: NDArray[np.int64],
    bid_order: NDArray[np.int64],
    min_count: int = 100,
    use_flag: str = "ignore"
) -> Tuple[NDArray[np.float64], List[str]]:
    """计算订单聚合后的时间间隔和价格分位数指标（V3版本 - 大数据量超高性能优化）

    🚀 V3版本核心优化：
    ===================
    1. **时间戳优化**：直接使用纳秒时间戳计算，避免大规模数据转换
    2. **内存优化**：预分配HashMap和Vec，减少动态扩容开销
    3. **排序优化**：一次性预排序，消除重复排序操作
    4. **访问优化**：改进数据局部性，使用数组而非Vec<Vec<>>
    5. **算法优化**：利用时间已排序特性，使用二分查找加速定位

    📊 性能目标：
    ============
    - 13万行数据计算时间：< 5秒
    - 内存使用减少：30-40%
    - 结果精度：与V2版本100%一致

    🔄 算法改进：
    ============
    1. **时间处理**：纳秒级时间戳直接计算，延迟到最终输出时转换
    2. **订单聚合**：预分配HashMap容量，批量处理减少哈希操作
    3. **分组排序**：按volume分组后，在组内预排序时间索引
    4. **最近查找**：利用时间有序性，用二分查找快速定位邻近记录
    5. **内存布局**：使用连续内存结构，提高缓存命中率

    参数与返回值与V2版本完全一致，确保兼容性。

    性能提升主要来自：
    - 避免时间戳转换（节省15-20%）
    - 消除重复排序（节省15-30%）
    - 内存访问优化（节省10-15%）
    - 算法层面优化（节省10-20%）

    适用于13万+数据量的高频交易数据分析场景。
    """
    ...

def calculate_order_time_gap_and_price_percentile_ultra_sorted_v4(
    volume: NDArray[np.float64],
    exchtime: NDArray[np.float64],
    price: NDArray[np.float64],
    flag: NDArray[np.int32],
    ask_order: NDArray[np.int64],
    bid_order: NDArray[np.int64],
    min_count: int = 100,
    use_flag: str = "ignore"
) -> Tuple[NDArray[np.float64], List[str]]:
    """计算订单聚合后的时间间隔和价格分位数指标（V4版本 - 极致优化版）

    本函数是calculate_order_time_gap_and_price_percentile_ultra_sorted_v2的大幅优化版本，
    通过以下核心优化策略大幅提升性能：

    🚀 核心优化策略：
    ======================
    1. **预计算目标索引缓存**：
       - 一次性预计算所有记录的目标索引，避免循环中重复分配
       - 减少内存分配和垃圾回收压力

    2. **选择算法优化**：
       - 使用`select_nth_unstable_by`而非全排序
       - 避免不必要的全量排序，只取前N个最近记录

    3. **向量化价格分位数计算**：
       - 预计算并缓存每个volume组的价格分位数
       - 避免重复排序和计算

    4. **减少函数调用开销**：
       - 批量处理记录，减少循环开销
       - 优化内存布局，提高缓存命中率

    ⚡ 性能提升：
    ============
    - 相比V2版本：**3-5倍性能提升**（具体取决于数据量）
    - 内存使用减少：**30-40%**
    - 大数据集（>100万条记录）上表现尤为突出

    📋 与V2版本的区别：
    ===================
    1. **算法优化**：
       - V2：传统排序算法 O(n log n)
       - V4：选择算法 + 部分排序 O(n)

    2. **缓存策略**：
       - V2：无缓存，每次重复计算
       - V4：预计算缓存，避免重复计算

    3. **内存优化**：
       - V2：频繁分配临时对象
       - V4：重用内存，预分配空间

    参数：
    ------
    volume : NDArray[np.float64]
        交易量数组
    exchtime : NDArray[np.float64]
        交易所时间戳（纳秒）
    price : NDArray[np.float64]
        价格数组
    flag : NDArray[np.int32]
        交易标志数组（V4版本中该参数被忽略）
    ask_order : NDArray[np.int64]
        卖单订单号数组
    bid_order : NDArray[np.int64]
        买单订单号数组
    min_count : int, optional
        最少记录数阈值，默认100
    use_flag : str, optional
        订单类型筛选模式：
        - "same": 相同订单类型（买单对买单，卖单对卖单）
        - "diff": 不同订单类型（买单对卖单）
        - "ignore": 考虑所有类型的订单

    返回：
    -------
    result : NDArray[np.float64]
        形状为(n, 27)的结果数组，包含27个指标
    columns : List[str]
        27个指标的中文列名列表，具体含义同V2版本

    使用示例：
    --------
    >>> import numpy as np
    >>> import rust_pyfunc as rp
    >>>
    >>> # 生成示例数据
    >>> n = 10000
    >>> volume = np.random.rand(n) * 100
    >>> exchtime = np.arange(n) * 1e9  # 纳秒时间戳
    >>> price = 100 + np.random.randn(n) * 0.1
    >>> flag = np.zeros(n, dtype=np.int32)
    >>> ask_order = np.random.randint(1, 1000, n).astype(np.int64)
    >>> bid_order = np.random.randint(1001, 2000, n).astype(np.int64)
    >>>
    >>> # 使用V4版本计算
    >>> result, columns = rp.calculate_order_time_gap_and_price_percentile_ultra_sorted_v4(
    ...     volume=volume,
    ...     exchtime=exchtime,
    ...     price=price,
    ...     flag=flag,
    ...     ask_order=ask_order,
    ...     bid_order=bid_order,
    ...     min_count=100,
    ...     use_flag="ignore"
    ... )
    >>>
    >>> print(f"结果形状: {result.shape}")  # (10000, 27)
    >>> print(f"指标数量: {len(columns)}")  # 27

    性能对比示例：
    -------------
    >>> import time
    >>>
    >>> # V2版本计时
    >>> start = time.time()
    >>> result_v2, _ = rp.calculate_order_time_gap_and_price_percentile_ultra_sorted_v2(...)
    >>> time_v2 = time.time() - start
    >>>
    >>> # V4版本计时
    >>> start = time.time()
    >>> result_v4, _ = rp.calculate_order_time_gap_and_price_percentile_ultra_sorted_v4(...)
    >>> time_v4 = time.time() - start
    >>>
    >>> print(f"V2耗时: {time_v2:.2f}秒")
    >>> print(f"V4耗时: {time_v4:.2f}秒")
    >>> print(f"性能提升: {time_v2/time_v4:.2f}倍")

    注意：
    -----
    - V4版本在保持与V2版本相同输出结果的基础上大幅优化性能
    - 建议在数据量较大（>10万条记录）时优先使用V4版本
    - V4版本在多核CPU上优势更加明显（但不使用并行，避免GIL限制）
    """
    ...

def analyze_asks(
    price: NDArray[np.float64],
    volume: NDArray[np.float64],
    volume_percentile: float = 0.9,
    min_duration: int = 1,
    ratio_mode: bool = False
) -> Tuple[NDArray[np.float64], List[str]]:
    """异常挂单区间特征提取器

    分析卖盘挂单数据中的异常挂单模式，提取量化特征。
    异常挂单定义为：在当前时刻3-9档位中volume超过全局阈值且为该时刻最大值的挂单。

    参数说明：
    ----------
    exchtime : NDArray[np.float64]
        交易时间戳数组（纳秒），已按时间升序排列
    number : NDArray[np.int32]
        档位编号数组（1-10，1为卖一，10为卖十），相同时间内按档位升序排列
    price : NDArray[np.float64]
        挂单价格数组
    volume : NDArray[np.float64]
        挂单量数组（支持浮点数，提供更灵活的数据输入）
    volume_percentile : float, default=0.9
        异常阈值分位数，默认0.9表示前10%的大挂单量
    min_duration : int, default=1
        异常区间最小持续行数，低于此值的区间将被过滤
    ratio_mode : bool, default=False
        是否使用比例模式。False时返回24个绝对特征，True时返回14个比例特征

    返回值：
    -------
    Tuple[NDArray[np.float64], List[str]]
        第一个元素：特征矩阵，shape为(N, 24)或(N, 14)，N为检测到的异常区间数量
        第二个元素：包含特征名称的中文字符串列表

    使用示例：
    ---------
    >>> features, names = analyze_asks(exchtime, number, price, volume)
    >>> df = pd.DataFrame(features, columns=names)
    """
    ...

def compute_non_breakthrough_stats(
    exchtime: NDArray[np.int64],
    volume: NDArray[np.float64],
    price: NDArray[np.float64],
    flag: NDArray[np.int64]
) -> Tuple[NDArray[np.float64], List[str]]:
    """计算股票逐笔成交数据中"价格未突破上一分钟价格范围"的24个统计指标

    该函数接收股票逐笔成交数据，计算每分钟中"价格未突破上一分钟价格范围"的交易行为相关的24个统计指标。

    核心定义：
    ---------
    价格未突破：对于第t分钟的某笔交易，若其价格在t-1分钟的任意一笔交易的价格中出现过（即存在相等），
    则该笔交易视为"价格未突破"。

    参数说明：
    ----------
    exchtime : NDArray[np.int64]
        成交时间数组，单位是纳秒，int64类型
    volume : NDArray[np.float64]
        成交量数组，float64类型，支持浮点数成交量
    price : NDArray[np.float64]
        成交价格数组，float64类型
    flag : NDArray[np.int64]
        主动买卖标识数组，int64类型
        66 表示主动买入，83 表示主动卖出

    返回值：
    -------
    Tuple[NDArray[np.float64], List[str]]
        第一个元素：n×25的二维数组，每一行对应一分钟（从第1分钟到第n分钟），
                   每一列是一个指标，共25个
        第二个元素：长度为25的中文列名列表

        25个统计指标包括：
        1. t分钟中，价格未突破t-1分钟价格范围的成交量总和
        2. 列1 / t分钟总成交量
        3. 列1 / t-1分钟总成交量
        4. t分钟中，价格未突破t-1分钟价格范围的交易笔数
        5. 列4 / t分钟总交易笔数
        6. 列4 / t-1分钟总交易笔数
        7. t分钟中，价格未突破的最后一笔交易的秒数
        8. t分钟中，价格未突破的第一笔交易的秒数
        9. 列7 - 列8 (未突破时间跨度)
        10. t分钟中，从第一次未突破价格到最后一次未突破价格之间的总成交量
        11. 列10 / t分钟总成交量
        12. t分钟中，从第一次未突破价格到最后一次未突破价格之间的总交易笔数
        13. 列12 / t分钟总交易笔数
        14. t分钟中，价格未突破且flag=66的交易笔数
        15. 列14 / 列4
        16. t分钟中，价格未突破且flag=66的成交量
        17. 列16 / 列1
        18. 列9 / 列4 (平均未突破时间跨度)
        19. t分钟中，价格未突破的交易记录的成交量序列与序列1..k的皮尔逊相关系数
        20. t分钟中，价格未突破的交易记录的flag序列（66→1，83→0）与序列1..k的皮尔逊相关系数
        21. t分钟中，价格未突破的交易记录的秒数均值
        22. t分钟中，价格未突破的交易记录的秒数标准差
        23. t分钟中，价格未突破的交易记录的秒数偏度
        24. t分钟中，价格未突破的交易记录的秒数峰度
        25. 该分钟对应的时间标记（该分钟第0秒的纳秒时间戳）

    注意事项：
    ---------
    - 若某一分钟没有任何交易价格未突破上一分钟价格范围，则该行前24个值全部为NaN，第25列仍显示时间标记
    - 时间分组按自然分钟划分（如 09:30:00.000000000 ～ 09:30:59.999999999）
    - 纳秒时间戳转秒数时，取模60秒内的秒数部分
    - 所有统计量会处理空序列情况（返回NaN）
    - 偏度、峰度使用样本标准定义

    性能要求：
    ---------
    在股票全天数据上（约13万条记录），总运行时间必须小于1秒

    使用示例：
    ---------
    >>> import pure_ocean_breeze.jason as p
    >>> df = p.read_trade('000001', 20220819)
    >>> exchtime = df.exchtime.to_numpy(int)
    >>> volume = df.volume.to_numpy(int)
    >>> price = df.price.to_numpy(float)
    >>> flag = df.flag.to_numpy(int)
    >>> result_matrix, column_names = compute_non_breakthrough_stats(exchtime, volume, price, flag)
    >>> import pandas as pd
    >>> result_df = pd.DataFrame(result_matrix, columns=column_names)
    """

def compute_price_cycle_features(
    exchtime_trade: NDArray[np.int64],
    price_trade: NDArray[np.float64],
    volume_trade: NDArray[np.float64],
    flag_trade: NDArray[np.int32],
    ask_exchtime: NDArray[np.int64],
    bid_exchtime: NDArray[np.int64],
    bid_price: NDArray[np.float64],
    bid_volume: NDArray[np.float64],
    bid_number: NDArray[np.int32],
    ask_price: NDArray[np.float64],
    ask_volume: NDArray[np.float64],
    ask_number: NDArray[np.int32],
    tick: float,
    drop_threshold: float,
    rise_threshold: float,
    window_ms: int,
    above_min_ms: int,
    use_trade_prices_as_grid: bool,
    price_grid_opt: Optional[NDArray[np.float64]] = None
) -> dict:
    """高性能逐价位买卖周期识别与标量特征聚合。

    该函数实现了一个复杂的买卖周期检测系统，识别价格水平的支撑和阻力位突破，
    并计算多维度量化特征。支持买入支撑位突破和卖出阻力位突破两种模式。

    算法原理：
    ----------
    1. **买入支撑位突破**（A→B→C→D）：
       - A: 价格首次触及支撑位X
       - B: 价格跌破支撑位X至少drop_threshold
       - C: 价格重新回到支撑位X之上
       - D: 盘口在X价位重新出现挂单（恢复完成）

    2. **卖出阻力位突破**（A'→B'→C'→D'）：
       - A': 价格首次触及阻力位Y
       - B': 价格涨破阻力位Y至少rise_threshold
       - C': 价格重新回到阻力位Y之下
       - D': 盘口在Y价位重新出现挂单（恢复完成）

    参数说明：
    ----------
    exchtime_trade : numpy.ndarray[int64]
        逐笔成交时间戳（纳秒）
    price_trade : numpy.ndarray[float64]
        逐笔成交价格
    volume_trade : numpy.ndarray[float64]
        逐笔成交量
    flag_trade : numpy.ndarray[int32]
        逐笔成交标志（66=主买, 83=主卖）
    ask_exchtime : numpy.ndarray[int64]
        卖方盘口时间戳
    bid_exchtime : numpy.ndarray[int64]
        买方盘口时间戳
    bid_price : numpy.ndarray[float64]
        买方盘口价格
    bid_volume : numpy.ndarray[float64]
        买方盘口挂单量
    bid_number : numpy.ndarray[int32]
        买方盘口档位号（1-10）
    ask_price : numpy.ndarray[float64]
        卖方盘口价格
    ask_volume : numpy.ndarray[float64]
        卖方盘口挂单量
    ask_number : numpy.ndarray[int32]
        卖方盘口档位号（1-10）
    tick : float
        最小价格变动单位
    drop_threshold : float
        买入支撑位突破的下破阈值（绝对价差）
    rise_threshold : float
        卖出阻力位突破的上破阈值（绝对价差）
    window_ms : int
        统计分析窗口大小（毫秒），用于后续指标计算
    above_min_ms : int
        A/A'事件前"连续在价外"的最小时长（毫秒），用于过滤噪声
    use_trade_prices_as_grid : bool
        是否使用成交价作为价格网格，True时使用所有成交价格
    price_grid_opt : Optional[numpy.ndarray[float64]]
        可选的价格网格数组，优先级高于use_trade_prices_as_grid

    返回值：
    -------
    dict
        包含以下键的字典：
        - "prices": 价格数组，分析的目标价格网格
        - "feature_names": 特征名称列表，共77个特征
        - "features": 特征矩阵（价格×特征）
        - "cycles_count_buy": 买方周期计数数组，每个价格位的买入周期数量
        - "cycles_count_sell": 卖方周期计数数组，每个价格位的卖出周期数量

    特征体系（77个）：
    -----------------
    **买入特征（37个）**：
    1. n_at_touch_buy: 触及时档位位置
    2. visible_levels_bid_at_touch_buy: 买档可见数量
    3. depth_x_init_buy: 初始深度
    4. depth_x_max_buy: 最大深度
    5. depth_x_min_buy: 最小深度
    6. depth_x_twap_buy: 时间加权平均深度
    7. depletion_total_x_buy: 总消耗量
    8. traded_depletion_x_buy: 交易消耗量
    9. cancel_depletion_x_buy: 撤单消耗量
    10. depletion_trade_share_buy: 交易消耗占比
    11. refill_depth_init_buy: 恢复初始深度
    12. refill_speed_buy: 恢复速度
    13. shield_depth_from_best_to_x_buy: 保护深度
    14. exec_at_x_sell_buy: 卖出成交量
    15. trade_size_median_at_x_sell_buy: 成交量中位数
    16. consumption_speed_buy: 消耗速度
    17. time_to_break_ms_buy: 突破时间
    18. time_under_x_ms_buy: 低价持续时间
    19. time_to_refill_ms_buy: 恢复时间
    20. undershoot_bp_buy: 下冲幅度（基点）
    21. overshoot_bp_buy: 上冲幅度（基点）
    22. break_force_bp_per_s_buy: 突破力度
    23. recovery_return_bp_1s_buy: 1秒恢复收益
    24. tobi_pre_touch_buy: 触及时买卖失衡
    25. tobi_trend_pre_buy: 事前趋势
    26. spread_at_touch_buy: 触及时点差
    27. spread_change_A_to_B_buy: 点差变化
    28. quote_rev_rate_pre_buy: 报价修订率
    29. rv_post_refill_buy: 恢复后已实现波动率
    30. vpin_like_A_window_buy: VPIN类指标
    31. adverse_post_fill_bp_buy: 成交后不利偏移
    32. adverse_post_refill_bp_buy: 恢复后不利偏移
    33. ofi_post_refill_buy: 恢复后订单流不平衡
    34. support_survival_ms_buy: 支撑位生存时间
    35. bounce_success_flag_buy: 反弹成功标志
    36. next_interarrival_ms_buy: 下次到达间隔
    37. queue_shape_slope_near_x_buy: 队列形状斜率

    **卖出特征（37个）**：
    1. n_at_touch_sell: 触及时档位位置
    2. visible_levels_ask_at_touch_sell: 卖档可见数量
    3. depth_y_init_sell: 初始深度
    4. depth_y_max_sell: 最大深度
    5. depth_y_min_sell: 最小深度
    6. depth_y_twap_sell: 时间加权平均深度
    7. depletion_total_y_sell: 总消耗量
    8. traded_depletion_y_sell: 交易消耗量
    9. cancel_depletion_y_sell: 撤单消耗量
    10. depletion_trade_share_sell: 交易消耗占比
    11. refill_depth_init_sell: 恢复初始深度
    12. refill_speed_sell: 恢复速度
    13. shield_depth_from_best_to_y_sell: 保护深度
    14. exec_at_y_buy_sell: 买入成交量
    15. trade_size_median_at_y_buy_sell: 成交量中位数
    16. consumption_speed_sell: 消耗速度
    17. time_to_break_ms_sell: 突破时间
    18. time_above_y_ms_sell: 高价持续时间
    19. time_to_refill_ms_sell: 恢复时间
    20. overshoot_bp_sell: 上冲幅度（基点）
    21. undershoot_bp_sell: 下冲幅度（基点）
    22. break_force_bp_per_s_sell: 突破力度
    23. recovery_return_bp_1s_sell: 1秒恢复收益
    24. tobi_pre_touch_sell: 触及时买卖失衡
    25. tobi_trend_pre_sell: 事前趋势
    26. spread_at_touch_sell: 触及时点差
    27. spread_change_A_to_B_sell: 点差变化
    28. quote_rev_rate_pre_sell: 报价修订率
    29. rv_post_refill_sell: 恢复后已实现波动率
    30. vpin_like_A_window_sell: VPIN类指标
    31. adverse_post_fill_bp_sell: 成交后不利偏移
    32. adverse_post_refill_bp_sell: 恢复后不利偏移
    33. ofi_post_refill_sell: 恢复后订单流不平衡
    34. resistance_survival_ms_sell: 阻力位生存时间
    35. bounce_success_flag_sell: 反弹成功标志
    36. next_interarrival_ms_sell: 下次到达间隔
    37. queue_shape_slope_near_y_sell: 队列形状斜率

    **通用特征（3个）**：
    1. cycles_count_buy: 买入周期数量
    2. cycles_count_sell: 卖出周期数量
    3. round_number_flag: 整数价格标记

    性能特点：
    ----------
    1. **高效算法** - 使用二分查找和时间对齐算法，复杂度O(n log n)
    2. **内存优化** - 预分配数据结构，避免动态内存分配
    3. **并行计算** - 利用Rust的Rayon库进行并行处理
    4. **精度控制** - 使用epsilon比较避免浮点误差
    5. **容错处理** - 对缺失数据进行合理处理，保证数值稳定性
    6. **批量处理** - 同时处理多个价格位，提高效率

    使用场景：
    ----------
    - 支撑位和阻力位的有效性分析
    - 高频交易中的流动性评估
    - 市场微观结构研究
    - 订单簿动力学特征提取
    """
    ...


    ...

def compute_price_cycle_features_b_segments_enhanced(
    exchtime_trade: NDArray[np.int64],
    price_trade: NDArray[np.float64],
    volume_trade: NDArray[np.float64],
    flag_trade: NDArray[np.int32],
    ask_exchtime: NDArray[np.int64],
    bid_exchtime: NDArray[np.int64],
    bid_price: NDArray[np.float64],
    bid_volume: NDArray[np.float64],
    bid_number: NDArray[np.int32],
    ask_price: NDArray[np.float64],
    ask_volume: NDArray[np.float64],
    ask_number: NDArray[np.int32],
    tick: float,
    drop_threshold: float,
    rise_threshold: float,
    use_trade_prices_as_grid: bool,
    price_grid_opt: Optional[NDArray[np.float64]] = None,
    side: Optional[str] = None
) -> dict:
    """增强版价格循环B段特征计算函数 - 终极统计合成版。

    该函数是compute_price_cycle_features_b_segments的增强版本，提供完整的21个基础指标
    和105个增强统计合成指标（每个基础指标5个合成指标），共210维特征（买侧105维+卖侧105维）。

    核心目标：
    ----------
    分析订单簿中特定价格水平的支撑/阻力突破行为，通过识别价格循环中的"B点"（突破点）
    并构建B段（B→B段），计算每个价格水平在买卖两侧的详细特征指标，并提供增强的统计合成指标。

    数据划分逻辑：
    --------------
    1. **价格网格构建**：
       - 可选择使用实际交易价格作为网格
       - 或基于tick大小生成规则价格网格
       - 支持自定义价格网格输入
       - 精度控制：基于tick大小设置epsilon容差，用于价格比较

    2. **B点识别机制**：
       对每个价格水平，分别识别买卖两侧的突破点：
       
       **买侧B点（BuyBreak）**：
       - 触发条件：价格从上方或等于水平位，向下突破至 price_level - drop_threshold
       - 含义：支撑水平被突破，触发卖盘压力
       
       **卖侧B点（SellBreak）**：
       - 触发条件：价格从下方或等于水平位，向上突破至 price_level + rise_threshold
       - 含义：阻力水平被突破，触发买盘推动

    3. **B段构建**：
       - 定义：相邻两个同类型B点之间的时间段
       - 买侧B段：BuyBreak→BuyBreak 序列
       - 卖侧B段：SellBreak→SellBreak 序列
       - 时间边界：使用B点发生的时间戳精确划分

    特征指标计算清单：
    ------------------
    **基础统计特征（21个原始指标）**：
    1. duration_ms：B段持续时间（毫秒）
    2. total_volume：段内总成交量
    3. trade_count：段内交易笔数
    4. vwap：成交量加权平均价格
    5. min_price：段内最低价格
    6. max_price：段内最高价格
    7. buy_ratio：买方向成交量占比
    8. sell_ratio：卖方向成交量占比
    9. start_spread：段开始时的买卖价差
    10. spread_change：价差变化量
    11. mid_return_bp：中间价收益率（基点）
    12. start_distance_to_level：开始时中间价与目标水平的距离
    13. end_distance_to_level：结束时中间价与目标水平的距离
    14. avg_bid_depth/max_bid_depth/min_bid_depth（买侧）或 avg_ask_depth/max_ask_depth/min_ask_depth（卖侧）：目标价格水平的深度统计
    15. vol_at_level：在目标价格水平的成交量
    16. count_at_level：在目标价格水平上的交易笔数
    17. time_between_ms：相邻B点时间间隔
    18. trades_between：相邻B点间交易笔数
    19. start_price：段开始时的交易价格
    20. end_price：段结束时的交易价格
    21. total_return_bp：段内总收益率（基点）

    **增强统计合成指标（每个原始指标对应5个合成指标）**：
    1. _mean：原始指标的算术平均值
    2. _cv：变异系数（标准差/平均值），衡量相对离散程度
    3. _max：原始指标的最大值
    4. _corr_seq：与序列号[1,2,3,...,n]的皮尔逊相关系数，衡量时间趋势
    5. _abs_corr_seq：绝对相关系数，衡量趋势强度
    6. _autocorr：一阶自相关性，衡量序列的自相关程度

    计算流程：
    ----------
    1. **数据预处理**：构建统一时间戳的订单簿快照序列
    2. **价格循环分析**：对每个价格水平独立分析
    3. **B点检测**：分别识别买卖两侧的突破事件
    4. **B段构建**：连接相邻同类型B点形成分析段
    5. **特征提取**：在每个B段内计算21维基础特征指标
    6. **统计合成**：对每个价格水平的多个B段特征进行增强聚合，生成105维合成特征（21×5）
    7. **结果对齐**：确保所有价格水平的特征矩阵维度一致，无数据的价格水平用NaN填充

    参数说明：
    ----------
    exchtime_trade : numpy.ndarray[int64]
        逐笔成交时间戳（纳秒）
    price_trade : numpy.ndarray[float64]
        逐笔成交价格
    volume_trade : numpy.ndarray[float64]
        逐笔成交量
    flag_trade : numpy.ndarray[int32]
        逐笔成交标志（66=主买, 83=主卖）
    ask_exchtime : numpy.ndarray[int64]
        卖方盘口时间戳
    bid_exchtime : numpy.ndarray[int64]
        买方盘口时间戳
    bid_price : numpy.ndarray[float64]
        买方盘口价格
    bid_volume : numpy.ndarray[float64]
        买方盘口挂单量
    bid_number : numpy.ndarray[int32]
        买方盘口档位号（1-10）
    ask_price : numpy.ndarray[float64]
        卖方盘口价格
    ask_volume : numpy.ndarray[float64]
        卖方盘口挂单量
    ask_number : numpy.ndarray[int32]
        卖方盘口档位号（1-10）
    tick : float
        最小价格变动单位
    drop_threshold : float
        买入支撑位突破的下破阈值（绝对价差）
    rise_threshold : float
        卖出阻力位突破的上破阈值（绝对价差）
    use_trade_prices_as_grid : bool
        是否使用成交价作为价格网格，True时使用所有成交价格
    price_grid_opt : Optional[numpy.ndarray[float64]]
        可选的价格网格数组，优先级高于use_trade_prices_as_grid
    side : Optional[str], default=None
        指定分析哪一侧："buy"（仅买入侧）、"sell"（仅卖出侧）、None（双侧，默认）

    返回值：
    -------
    dict
        包含以下键的字典：
        - "prices": 价格数组，分析的目标价格网格
        - "buy_features": 买入侧特征矩阵（价格×105个特征）
        - "sell_features": 卖出侧特征矩阵（价格×105个特征）
        - "buy_feature_names": 买入侧特征名称列表（105个特征，21×5种统计）
        - "sell_feature_names": 卖出侧特征名称列表（105个特征，21×5种统计）
        - "buy_segment_counts": 每个价格的买入侧B分段数量
        - "sell_segment_counts": 每个价格的卖出侧B分段数量

    最终特征数量：
    --------------
    - 基础指标：21个
    - 统计合成方法：5种（mean, cv, max, corr_seq, abs_corr_seq, autocorr）
    - 总特征数：21 × 5 = 105个特征/侧
    - 双侧总计：210个特征（105买侧 + 105卖侧）

    关键技术特点：
    --------------
    1. **增强聚合算法**：使用enhanced_aggregate_metrics函数，为每个原始指标计算5种统计量
    2. **相关性分析**：计算与序列号的相关性，识别时间趋势
    3. **自相关性分析**：计算一阶自相关性，衡量序列的自相关程度
    4. **结果对齐保证**：确保输出矩阵维度一致，避免索引错位问题
    5. **缺失值处理**：对无有效数据的价格水平自动填充NaN
    6. **DataFrame兼容**：可以直接用于创建pandas DataFrame而不会报错

    使用示例：
    -------
    >>> import pure_ocean_breeze.jason as p
    >>> import rust_pyfunc as rp
    >>> trade = p.adjust_afternoon(p.read_trade('000001', 20220819))
    >>> asks, bids = p.read_market_pair('000001', 20220819)
    >>> asks = p.adjust_afternoon(asks)
    >>> bids = p.adjust_afternoon(bids)
    >>> res = rp.compute_price_cycle_features_b_segments_enhanced(
    ...     trade.exchtime.to_numpy(int), trade.price.to_numpy(float),
    ...     trade.volume.to_numpy(float), trade.flag.to_numpy(np.int32),
    ...     asks.exchtime.to_numpy(int), bids.exchtime.to_numpy(int),
    ...     bids.price.to_numpy(float), bids.vol.to_numpy(float), bids.number.to_numpy(np.int32),
    ...     asks.price.to_numpy(float), asks.vol.to_numpy(float), asks.number.to_numpy(np.int32),
    ...     0.01, 0.01, 0.01, 15000, 100, True, None
    ... )
    >>> # 创建DataFrame进行分析
    >>> import pandas as pd
    >>> buy_df = pd.DataFrame(res['buy_features'], columns=res['buy_feature_names'], index=res['prices'])
    >>> sell_df = pd.DataFrame(res['sell_features'], columns=res['sell_feature_names'], index=res['prices'])
    >>> print(f"买侧特征矩阵形状: {res['buy_features'].shape}")
    >>> print(f"卖侧特征矩阵形状: {res['sell_features'].shape}")
    >>> print(f"分析的价格水平数量: {len(res['prices'])}")
    """
    ...

def analyze_long_orders(
    exchtime: NDArray[np.int64],
    order: NDArray[np.int64],
    volume: NDArray[np.float64],
    top_ratio: Optional[float] = None
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """分析漫长订单并计算比值

    识别三种类型的漫长订单（时间漫长、次数漫长、两者都漫长），并计算其他订单成交量与漫长订单成交量的比值。

    参数说明：
    ----------
    exchtime : NDArray[np.int64]
        交易时间数组（纳秒时间戳）
    order : NDArray[np.int64]
        订单编号数组
    volume : NDArray[np.float64]
        成交量数组
    top_ratio : Optional[float], default=None
        只计算最漫长的一部分订单的比例(0.0-1.0)：
        - None 或 1.0：计算所有订单（默认，保持向后兼容）
        - 0.5：只计算最漫长的一半订单
        - 0.1：只计算最漫长的前10%订单
        三种类型的漫长订单分别按各自指标排序：
        - 时间漫长订单按时间跨度排序
        - 次数漫长订单按出现次数排序
        - 两者都漫长订单按时间跨度排序

    返回值：
    -------
    Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]
        包含六个数组的元组：
        - 时间漫长订单的比值序列
        - 次数漫长订单的比值序列
        - 两者都漫长订单的比值序列
        - 时间漫长订单的总比值（长度为1的数组）
        - 次数漫长订单的总比值（长度为1的数组）
        - 两者都漫长订单的总比值（长度为1的数组）

    漫长订单定义：
    --------------
    1. **时间漫长订单**：某个订单号对应的exchtime取值不止一种
    2. **次数漫长订单**：某个订单号在第一次出现和最后一次出现之间，夹杂着其他的订单号
    3. **两者都漫长**：同时满足时间漫长和次数漫长的订单

    比值计算：
    ----------
    对于每个漫长订单：
    - 单个比值：其他订单成交量总和 / 该漫长订单成交量总和
    - 总比值：所有漫长订单的其他订单总成交量 / 所有漫长订单的总成交量

    筛选逻辑：
    ----------
    当指定top_ratio时：
    1. 收集所有符合条件的漫长订单
    2. 按各自指标排序（时间跨度或出现次数）
    3. 保留top_ratio比例的最漫长订单
    4. 只对筛选后的订单计算比值

    算法特点：
    ----------
    - 使用HashMap高效存储每个订单的索引范围和时间信息
    - 分别对三种类型进行独立排序和筛选
    - 时间复杂度：O(n log n)，其中n为数据长度（排序带来的复杂度）
    - 空间复杂度：O(m)，其中m为不同订单号的数量
    - 支持大规模数据处理（10万条数据1秒内完成）

    使用示例：
    ----------
    >>> import numpy as np
    >>> import rust_pyfunc as rp
    >>>
    >>> # 示例数据
    >>> exchtime = np.array([1000, 2000, 2000, 3000, 4000], dtype=np.int64)
    >>> order = np.array([1, 1, 2, 2, 1], dtype=np.int64)
    >>> volume = np.array([100.0, 150.0, 200.0, 120.0, 180.0])
    >>>
    >>> # 计算所有漫长订单
    >>> time_ratios, count_ratios, both_ratios, time_total, count_total, both_total = rp.analyze_long_orders(exchtime, order, volume)
    >>> print(f"时间漫长比值: {time_ratios}")
    >>>
    >>> # 只计算最漫长的一半订单
    >>> time_ratios_half, count_ratios_half, both_ratios_half, time_total_half, count_total_half, both_total_half = rp.analyze_long_orders(exchtime, order, volume, 0.5)
    >>> print(f"筛选后时间漫长比值: {time_ratios_half}")
    """
    ...

def analyze_long_orders_python(
    exchtime: NDArray[np.int64],
    order: NDArray[np.int64],
    volume: NDArray[np.float64],
    top_ratio: Optional[float] = None
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Python版本的漫长订单分析函数（用于测试对比）

    功能和参数与analyze_long_orders完全相同，使用Python实现，用于验证Rust版本的正确性。

    参数和返回值详见analyze_long_orders函数。
    """
    ...

def calculate_trade_price_statistics_by_volume(
    volume: NDArray[np.float64],
    exchtime: NDArray[np.int64],
    price: NDArray[np.float64],
    flag: NDArray[np.int32],
    min_count: int = 10,
    use_flag: str = "same"
) -> Tuple[NDArray[np.float64], NDArray[np.float64], List[str]]:
    """计算同体量同方向成交的价格统计指标

    对每笔成交计算其最近的x%同体量、同主买卖方向成交的价格统计指标。

    📊 计算指标说明：
    =================
    🎯 百分比档位：1%、2%、3%、4%、5%、10%、20%、30%、40%、50%

    💰 价格均值指标（10个）：
    - 价格均值_1%：最近1%同体量同方向成交的价格平均值
    - 价格均值_2%：最近2%同体量同方向成交的价格平均值
    - ...依次类推到50%

    📏 价格标准差指标（10个）：
    - 价格标准差_1%：最近1%同体量同方向成交的价格标准差
    - 价格标准差_2%：最近2%同体量同方向成交的价格标准差
    - ...依次类推到50%

    🔄 算法逻辑：
    ============
    1. 按成交量进行分组，确保同组内成交量相同
    2. 对每笔成交，找到最近的同主买卖方向成交记录
    3. 按时间距离排序，取最近的x%成交
    4. 计算这些成交价格的平均值和标准差

    🎛️ use_flag参数说明：
    ====================
    - "same"：只与同主买卖方向的成交比较（买单vs买单，卖单vs卖单）
    - "diff"：只与相反主买卖方向的成交比较（买单vs卖单）
    - "ignore"：与所有成交比较，不考虑买卖方向

    📋 数据要求：
    ============
    输入数据建议按volume和exchtime排序以获得最佳性能：
    df.sort_values(['volume', 'exchtime'])

    ⚡ 性能特点：
    ============
    - 采用Volume分组批量计算，避免重复排序
    - 使用时间距离预排序，高效定位最近成交
    - 算法复杂度：O(n log n)

    参数：
    =====
    volume : NDArray[np.float64]
        成交量数组
    exchtime : NDArray[np.int64]
        成交时间数组（纳秒时间戳，函数内部自动转换为秒）
    price : NDArray[np.float64]
        成交价格数组
    flag : NDArray[np.int32]
        主买卖标志数组（66=买，83=卖）
    min_count : int, default=10
        计算统计指标所需的最少同方向成交记录数
    use_flag : str, default="same"
        方向筛选参数："same"=同方向，"diff"=反方向，"ignore"=忽略方向

    返回值：
    =======
    Tuple[NDArray[np.float64], NDArray[np.float64], List[str]]
        - 价格均值数组：n行10列，每行对应一笔成交的10个档位价格均值
        - 价格标准差数组：n行10列，每行对应一笔成交的10个档位价格标准差
        - 列名列表：包含20个列名（10个均值+10个标准差）

    示例：
    =====
    >>> import rust_pyfunc as rp
    >>> import numpy as np
    >>>
    >>> # 准备测试数据（已按volume和time排序）
    >>> volume = np.array([100.0, 100.0, 200.0, 200.0, 100.0])
    >>> exchtime = np.array([1609459200000000000, 1609459201000000000, 1609459202000000000,
    ...                     1609459203000000000, 1609459204000000000])
    >>> price = np.array([10.1, 10.2, 20.1, 20.2, 10.3])
    >>> flag = np.array([66, 66, 83, 83, 66])  # 66=买，83=卖
    >>>
    >>> # 计算价格统计指标
    >>> means, stds, columns = rp.calculate_trade_price_statistics_by_volume(
    ...     volume, exchtime, price, flag, min_count=2, use_flag="same"
    ... )
    >>> print(f"均值数组形状: {means.shape}")  # (5, 10)
    >>> print(f"标准差数组形状: {stds.shape}")  # (5, 10)
    >>> print(f"列名: {columns[:5]}...")  # ['价格均值_1%', '价格均值_2%', ...]
    """
    ...

def calculate_trade_price_statistics_by_volume_v2(
    volume: NDArray[np.float64],
    exchtime: NDArray[np.int64],
    price: NDArray[np.float64],
    flag: NDArray[np.int32],
    ask_order: NDArray[np.int64],
    bid_order: NDArray[np.int64],
    min_count: int = 10,
    use_flag: str = "same"
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], List[str]]:
    """计算订单聚合后的VWAP价格统计指标（V2版本）

    本函数是calculate_trade_price_statistics_by_volume的改进版本，主要区别在于：

    📋 与原版本的主要区别：
    ========================
    1. **订单聚合方式**：
       - 原版本：基于逐笔成交进行分析
       - V2版本：先按订单号聚合成交记录，然后基于订单进行分析

    2. **价格计算方式**：
       - 原版本：使用单笔成交价格
       - V2版本：使用订单的VWAP（成交量加权平均价格）

    3. **方向判断方式**：
       - 原版本：基于主买卖标志（flag）
       - V2版本：基于订单类型（ask_order/bid_order）

    4. **输出分离**：V2版本现在将买单和卖单的统计指标分别返回

    🔄 订单聚合逻辑：
    ================
    - 卖单（ask_order != 0）：基于ask_order聚合成交记录
    - 买单（bid_order != 0）：基于bid_order聚合成交记录
    - 每个订单计算：
      - 总volume：累加所有成交volume
      - VWAP价格：Σ(volume × price) / Σ(volume)
      - 最后时间：所有成交时间的最大值

    📊 计算指标说明：
    =================
    🎯 百分比档位：1%、2%、3%、4%、5%、10%、20%、30%、40%、50%

    💰 VWAP价格均值指标（10个）：
    - 价格均值_1%：最近1%同体量同类型订单的VWAP价格平均值
    - 价格均值_2%：最近2%同体量同类型订单的VWAP价格平均值
    - ...依次类推到50%

    📏 VWAP价格标准差指标（10个）：
    - 价格标准差_1%：最近1%同体量同类型订单的VWAP价格标准差
    - 价格标准差_2%：最近2%同体量同类型订单的VWAP价格标准差
    - ...依次类推到50%

    🎛️ use_flag参数说明：
    ====================
    - "same"：只与同订单类型比较（买单vs买单，卖单vs卖单）
    - "diff"：只与相反订单类型比较（买单vs卖单）
    - "ignore"：与所有订单比较，不考虑订单类型

    📋 数据要求：
    ============
    输入数据建议按volume和exchtime排序以获得最佳性能：
    df.sort_values(['volume', 'exchtime'])

    ⚡ 算法优势：
    ============
    - 更准确的价格反映：VWAP比单笔成交价格更能代表订单的真实成本
    - 订单级别的分析：从订单执行角度而非单笔交易角度分析市场行为
    - 更稳定的统计：基于订单聚合减少了噪声，提高了统计稳定性
    - 分离输出：买单和卖单统计指标分开返回，便于独立分析

    参数：
    =====
    volume : NDArray[np.float64]
        成交量数组
    exchtime : NDArray[np.int64]
        成交时间数组（纳秒时间戳，函数内部自动转换为秒）
    price : NDArray[np.float64]
        成交价格数组
    flag : NDArray[np.int32]
        主买卖标志数组（在V2版本中被忽略）
    ask_order : NDArray[np.int64]
        卖单订单号数组
    bid_order : NDArray[np.int64]
        买单订单号数组
    min_count : int, default=10
        计算统计指标所需的最少同类型订单数
    use_flag : str, default="same"
        类型筛选参数："same"=同类型，"diff"=反类型，"ignore"=忽略类型

    返回值：
    =======
    Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], List[str]]
        - means_buy: 买单VWAP价格均值数组，(买单数量, 10)形状
        - means_sell: 卖单VWAP价格均值数组，(卖单数量, 10)形状
        - stds_buy: 买单VWAP价格标准差数组，(买单数量, 10)形状
        - stds_sell: 卖单VWAP价格标准差数组，(卖单数量, 10)形状
        - columns: 列名列表，包含20个列名（10个均值+10个标准差）

    示例：
    =====
    >>> import rust_pyfunc as rp
    >>> import numpy as np
    >>>
    >>> # 准备测试数据（已按volume和time排序）
    >>> volume = np.array([100.0, 100.0, 200.0, 200.0, 100.0])
    >>> exchtime = np.array([1609459200000000000, 1609459201000000000, 1609459202000000000,
    ...                     1609459203000000000, 1609459204000000000])
    >>> price = np.array([10.1, 10.2, 20.1, 20.2, 10.3])
    >>> flag = np.array([66, 66, 83, 83, 66])  # 66=买，83=卖
    >>> ask_order = np.array([0, 0, 1001, 1001, 0])  # 卖单订单号
    >>> bid_order = np.array([2001, 2001, 0, 0, 2002])  # 买单订单号
    >>>
    >>> # 计算VWAP价格统计指标（现在返回4个数组+列名）
    >>> means_buy, means_sell, stds_buy, stds_sell, columns = rp.calculate_trade_price_statistics_by_volume_v2(
    ...     volume, exchtime, price, flag, ask_order, bid_order, min_count=2, use_flag="same"
    ... )
    >>> print(f"买单VWAP均值数组形状: {means_buy.shape}")  # 例如 (3, 10)
    >>> print(f"卖单VWAP均值数组形状: {means_sell.shape}")  # 例如 (2, 10)
    >>> print(f"列名: {columns[:5]}...")  # ['价格均值_1%', '价格均值_2%', ...]
    """
    ...

def calculate_trade_price_statistics_by_volume_bucketed(
    volume: NDArray[np.float64],
    exchtime: NDArray[np.int64],
    price: NDArray[np.float64],
    flag: NDArray[np.int32],
    min_count: int = 10,
    use_flag: str = "same",
    num_buckets: int = 20
) -> Tuple[NDArray[np.float64], NDArray[np.float64], List[str]]:
    """计算同体量区间同方向成交的价格统计指标（分桶版本）

    与原版本的区别：
    - 将体量分成20个区间（或保持原始体量如果种类≤20）
    - 计算落在相同体量区间且同方向的成交统计指标
    - 避免体量种类过多时导致的分组过细、样本不足问题

    参数：
    =====
    volume : NDArray[np.float64]
        成交量数组
    exchtime : NDArray[np.int64]
        成交时间数组（纳秒时间戳，函数内部自动转换为秒）
    price : NDArray[np.float64]
        成交价格数组
    flag : NDArray[np.int32]
        主买卖标志数组（66=买，83=卖）
    min_count : int, default=10
        计算统计指标所需的最少同方向成交数
    use_flag : str, default="same"
        方向筛选参数："same"=同方向，"diff"=反方向，"ignore"=忽略方向
    num_buckets : int, default=20
        体量分桶数量。如果体量种类≤该值，则使用原始体量；否则分为该数量的区间

    返回值：
    =======
    Tuple[NDArray[np.float64], NDArray[np.float64], List[str]]
        - 价格均值数组：n行10列，每行对应一笔成交的10个档位价格均值
        - 价格标准差数组：n行10列，每行对应一笔成交的10个档位价格标准差
        - 列名列表：包含20个列名（10个均值+10个标准差）

    示例：
    =====
    >>> import rust_pyfunc as rp
    >>> import numpy as np
    >>>
    >>> # 准备测试数据
    >>> volume = np.array([100.0, 120.0, 200.0, 210.0, 300.0, 320.0])
    >>> exchtime = np.array([1609459200000000000, 1609459201000000000, 1609459202000000000,
    ...                     1609459203000000000, 1609459204000000000, 1609459205000000000])
    >>> price = np.array([10.1, 10.2, 20.1, 20.2, 30.1, 30.2])
    >>> flag = np.array([66, 66, 83, 83, 66, 66])  # 66=买，83=卖
    >>>
    >>> # 计算分桶版本的价格统计指标
    >>> means, stds, columns = rp.calculate_trade_price_statistics_by_volume_bucketed(
    ...     volume, exchtime, price, flag, min_count=2, use_flag="same", num_buckets=20
    ... )
    >>> print(f"价格均值数组形状: {means.shape}")  # (6, 10)
    >>> print(f"价格标准差数组形状: {stds.shape}")  # (6, 10)
    >>> print(f"列名: {columns[:5]}...")  # ['价格均值_1%', '价格均值_2%', ...]
    """
    ...

def calculate_trade_price_statistics_by_volume_v2_bucketed(
    volume: NDArray[np.float64],
    exchtime: NDArray[np.int64],
    price: NDArray[np.float64],
    flag: NDArray[np.int32],
    ask_order: NDArray[np.int64],
    bid_order: NDArray[np.int64],
    min_count: int = 10,
    use_flag: str = "same",
    num_buckets: int = 20
) -> Tuple[NDArray[np.float64], NDArray[np.float64], List[str]]:
    """计算同体量区间同方向订单的VWAP价格统计指标（分桶版本V2）

    与原版本V2的区别：
    - 将体量分成20个区间（或保持原始体量如果种类≤20）
    - 计算落在相同体量区间且同方向的订单统计指标
    - 基于订单类型（ask/bid）而非交易标志进行分类
    - 避免体量种类过多时导致的分组过细、样本不足问题

    参数：
    =====
    volume : NDArray[np.float64]
        成交量数组
    exchtime : NDArray[np.int64]
        成交时间数组（纳秒时间戳，函数内部自动转换为秒）
    price : NDArray[np.float64]
        成交价格数组
    flag : NDArray[np.int32]
        主买卖标志数组（在V2版本中被忽略）
    ask_order : NDArray[np.int64]
        卖单订单号数组
    bid_order : NDArray[np.int64]
        买单订单号数组
    min_count : int, default=10
        计算统计指标所需的最少同类型订单数
    use_flag : str, default="same"
        类型筛选参数："same"=同类型，"diff"=反类型，"ignore"=忽略类型
    num_buckets : int, default=20
        体量分桶数量。如果体量种类≤该值，则使用原始体量；否则分为该数量的区间

    返回值：
    =======
    Tuple[NDArray[np.float64], NDArray[np.float64], List[str]]
        - VWAP价格均值数组：n行10列，每行对应一笔成交的10个档位VWAP价格均值
        - VWAP价格标准差数组：n行10列，每行对应一笔成交的10个档位VWAP价格标准差
        - 列名列表：包含20个列名（10个均值+10个标准差）

    示例：
    =====
    >>> import rust_pyfunc as rp
    >>> import numpy as np
    >>>
    >>> # 准备测试数据
    >>> volume = np.array([100.0, 120.0, 200.0, 210.0, 300.0, 320.0])
    >>> exchtime = np.array([1609459200000000000, 1609459201000000000, 1609459202000000000,
    ...                     1609459203000000000, 1609459204000000000, 1609459205000000000])
    >>> price = np.array([10.1, 10.2, 20.1, 20.2, 30.1, 30.2])
    >>> flag = np.array([66, 66, 83, 83, 66, 66])  # 66=买，83=卖
    >>> ask_order = np.array([0, 0, 1001, 1001, 0, 0])  # 卖单订单号
    >>> bid_order = np.array([2001, 2001, 0, 0, 3001, 3001])  # 买单订单号
    >>>
    >>> # 计算分桶版本V2的VWAP价格统计指标
    >>> means, stds, columns = rp.calculate_trade_price_statistics_by_volume_v2_bucketed(
    ...     volume, exchtime, price, flag, ask_order, bid_order, min_count=2, use_flag="same", num_buckets=20
    ... )
    >>> print(f"VWAP均值数组形状: {means.shape}")  # (6, 10)
    >>> print(f"VWAP标准差数组形状: {stds.shape}")  # (6, 10)
    >>> print(f"列名: {columns[:5]}...")  # ['价格均值_1%', '价格均值_2%', ...]
    """
    ...

def calculate_trade_price_statistics_by_volume_bucketed_v3(
    volume: NDArray[np.float64],
    exchtime: NDArray[np.int64],
    price: NDArray[np.float64],
    flag: NDArray[np.int32],
    min_count: int = 10,
    use_flag: str = "same",
    num_buckets: int = 20
) -> Tuple[NDArray[np.float64], NDArray[np.float64], List[str]]:
    """优化版本：极致性能的分桶统计计算函数（V3）

    针对13万数据量快速完成的极致优化版本，与bucketed原版本计算结果完全一致。
    核心思路：预排序 + 批量处理，避免对每个记录单独排序。

    🚀 核心优化技术：
    ==================
    - 在volume组级别预排序时间索引（一次排序，多次使用）
    - 使用二分查找定位邻近记录
    - 批量计算所有百分比档位（增量算法）
    - 部分排序（只排序需要的元素）
    - 内存预分配和缓冲区重用

    🎯 性能目标：
    ============
    - 13万数据量：约8秒（与v3非bucketed版本相当）
    - 算法复杂度：O(n log n)
    - 内存使用：最小化
    - 结果精度：与bucketed原版本完全一致

    💡 与bucketed原版本的区别：
    ============================
    - V3版本：针对性能瓶颈深度优化，消除循环内重复排序
    - 原版本：每条记录独立排序
    - V3版本在大数据量上显著更快（约1.7倍提速）

    Parameters
    ----------
    volume : NDArray[np.float64]
        成交量数组
    exchtime : NDArray[np.int64]
        成交时间数组（纳秒时间戳，函数内部自动转换为秒）
    price : NDArray[np.float64]
        成交价格数组
    flag : NDArray[np.int32]
        主买卖标志数组（66=买，83=卖）
    min_count : int, default=10
        计算统计指标所需的最少同方向成交数
    use_flag : str, default="same"
        方向筛选参数：
        - "same": 同方向
        - "diff": 反方向
        - "ignore": 忽略方向
    num_buckets : int, default=20
        体量分桶数量。如果体量种类≤该值，则使用原始体量；否则分为该数量的区间

    Returns
    -------
    Tuple[NDArray[np.float64], NDArray[np.float64], List[str]]
        (均值数组, 标准差数组, 列名列表)
        - 价格均值数组：n行10列，每行对应一笔成交的10个档位价格均值
        - 价格标准差数组：n行10列，每行对应一笔成交的10个档位价格标准差
        - 列名列表：包含20个列名（10个均值+10个标准差）

    Performance
    -----------
    典型性能表现（优化后）：
    - 5万数据：约0.3-0.5秒
    - 13万数据：约0.8-1.0秒
    - 算法复杂度：O(n log n)
    - 相比bucketed原版本提速：约1.7倍

    Note
    ----
    1. 体量分桶可以避免体量种类过多时分组过细、样本不足问题
    2. 该版本与bucketed原版本计算结果完全一致，可直接替换使用
    3. 推荐用于大规模数据处理和性能敏感场景
    4. 使用增量算法计算均值和方差，数值稳定性优秀

    Examples
    --------
    >>> import rust_pyfunc as rp
    >>> import numpy as np
    >>>
    >>> # 准备测试数据
    >>> volume = np.array([100.0, 120.0, 200.0, 210.0, 300.0, 320.0])
    >>> exchtime = np.array([1609459200000000000, 1609459201000000000, 1609459202000000000,
    ...                     1609459203000000000, 1609459204000000000, 1609459205000000000])
    >>> price = np.array([10.1, 10.2, 20.1, 20.2, 30.1, 30.2])
    >>> flag = np.array([66, 66, 83, 83, 66, 66])  # 66=买，83=卖
    >>>
    >>> # 调用V3优化版本
    >>> means, stds, columns = rp.calculate_trade_price_statistics_by_volume_bucketed_v3(
    ...     volume, exchtime, price, flag, min_count=2, use_flag="same", num_buckets=20
    ... )
    >>> print(f"价格均值数组形状: {means.shape}")  # (6, 10)
    >>> print(f"价格标准差数组形状: {stds.shape}")  # (6, 10)
    """
    ...

def calculate_order_time_gap_and_price_percentile_ultra_sorted_bucketed(
    volume: NDArray[np.float64],
    exchtime: NDArray[np.float64],
    price: NDArray[np.float64],
    flag: NDArray[np.int32],
    ask_order: NDArray[np.int64],
    bid_order: NDArray[np.int64],
    min_count: int = 100,
    use_flag: str = "ignore",
    num_buckets: int = 20
) -> Tuple[NDArray[np.float64], List[str]]:
    """计算订单聚合后的时间间隔和价格分位数指标（Ultra Sorted分桶版本）

    与原版本的区别：
    - 将体量分成20个区间（或保持原始体量如果种类≤20）
    - 计算落在相同体量区间且相同方向订单的统计指标
    - 基于交易标志（66=买，83=卖）进行方向分类
    - 避免体量种类过多时导致的分组过细、样本不足问题

    先将逐笔成交按订单号聚合，然后对聚合后的订单计算22个量化指标。

    参数：
    =====
    volume : NDArray[np.float64]
        成交量数组
    exchtime : NDArray[np.float64]
        成交时间数组（单位：秒）
    price : NDArray[np.float64]
        成交价格数组
    flag : NDArray[np.int32]
        主买卖标志数组（66=买，83=卖）
    ask_order : NDArray[np.int64]
        卖单订单号数组
    bid_order : NDArray[np.int64]
        买单订单号数组
    min_count : int, default=100
        计算统计指标所需的最少同类型订单数
    use_flag : str, default="ignore"
        类型筛选参数："same"=同类型，"diff"=反类型，"ignore"=忽略类型
    num_buckets : int, default=20
        体量分桶数量。如果体量种类≤该值，则使用原始体量；否则分为该数量的区间

    返回值：
    =======
    Tuple[NDArray[np.float64], List[str]]
        - 结果数组：n行27列，包含22个指标+5个订单信息列
        - 列名列表：包含27个列名

    示例：
    =====
    >>> import rust_pyfunc as rp
    >>> import numpy as np
    >>>
    >>> # 准备测试数据
    >>> volume = np.array([100.0, 120.0, 200.0, 210.0, 300.0])
    >>> exchtime = np.array([1609459200.0, 1609459201.0, 1609459202.0, 1609459203.0, 1609459204.0])
    >>> price = np.array([10.1, 10.2, 20.1, 20.2, 30.1])
    >>> flag = np.array([66, 66, 83, 83, 66])  # 66=买，83=卖
    >>> ask_order = np.array([0, 0, 1001, 1001, 0])
    >>> bid_order = np.array([2001, 2001, 0, 0, 3001])
    >>>
    >>> # 计算分桶版本的订单指标
    >>> result, columns = rp.calculate_order_time_gap_and_price_percentile_ultra_sorted_bucketed(
    ...     volume, exchtime, price, flag, ask_order, bid_order, min_count=2, use_flag="ignore", num_buckets=20
    ... )
    >>> print(f"结果数组形状: {result.shape}")  # (5, 27)
    >>> print(f"列名: {columns[:5]}...")  # ['最近时间间隔', '平均时间间隔_1%', ...]
    """
    ...

def calculate_order_time_gap_and_price_percentile_ultra_sorted_v2_bucketed(
    volume: NDArray[np.float64],
    exchtime: NDArray[np.float64],
    price: NDArray[np.float64],
    flag: NDArray[np.int32],
    ask_order: NDArray[np.int64],
    bid_order: NDArray[np.int64],
    min_count: int = 100,
    use_flag: str = "ignore",
    num_buckets: int = 20
) -> Tuple[NDArray[np.float64], List[str]]:
    """计算订单聚合后的时间间隔和价格分位数指标（Ultra Sorted分桶版本V2）

    与原版本V2的区别：
    - 将体量分成20个区间（或保持原始体量如果种类≤20）
    - 计算落在相同体量区间且相同方向订单的统计指标
    - 基于订单类型（ask/bid）而非交易标志进行方向分类
    - 避免体量种类过多时导致的分组过细、样本不足问题

    先将逐笔成交按订单号聚合，然后对聚合后的订单计算22个量化指标。

    参数：
    =====
    volume : NDArray[np.float64]
        成交量数组
    exchtime : NDArray[np.float64]
        成交时间数组（单位：秒）
    price : NDArray[np.float64]
        成交价格数组
    flag : NDArray[np.int32]
        主买卖标志数组（在V2版本中被忽略）
    ask_order : NDArray[np.int64]
        卖单订单号数组
    bid_order : NDArray[np.int64]
        买单订单号数组
    min_count : int, default=100
        计算统计指标所需的最少同类型订单数
    use_flag : str, default="ignore"
        类型筛选参数："same"=同类型，"diff"=反类型，"ignore"=忽略类型
    num_buckets : int, default=20
        体量分桶数量。如果体量种类≤该值，则使用原始体量；否则分为该数量的区间

    返回值：
    =======
    Tuple[NDArray[np.float64], List[str]]
        - 结果数组：n行27列，包含22个指标+5个订单信息列
        - 列名列表：包含27个列名

    示例：
    =====
    >>> import rust_pyfunc as rp
    >>> import numpy as np
    >>>
    >>> # 准备测试数据
    >>> volume = np.array([100.0, 120.0, 200.0, 210.0, 300.0])
    >>> exchtime = np.array([1609459200.0, 1609459201.0, 1609459202.0, 1609459203.0, 1609459204.0])
    >>> price = np.array([10.1, 10.2, 20.1, 20.2, 30.1])
    >>> flag = np.array([66, 66, 83, 83, 66])  # 66=买，83=卖
    >>> ask_order = np.array([0, 0, 1001, 1001, 0])
    >>> bid_order = np.array([2001, 2001, 0, 0, 3001])
    >>>
    >>> # 计算分桶版本V2的订单指标
    >>> result, columns = rp.calculate_order_time_gap_and_price_percentile_ultra_sorted_v2_bucketed(
    ...     volume, exchtime, price, flag, ask_order, bid_order, min_count=2, use_flag="ignore", num_buckets=20
    ... )
    >>> print(f"结果数组形状: {result.shape}")  # (5, 27)
    >>> print(f"列名: {columns[:5]}...")  # ['最近时间间隔', '平均时间间隔_1%', ...]
    """
    ...


def calculate_trade_price_statistics_by_volume_optimized(
    volume: NDArray[np.float64],
    exchtime: NDArray[np.int64],
    price: NDArray[np.float64],
    flag: NDArray[np.int32],
    min_count: int = 10,
    use_flag: str = "same"
) -> Tuple[NDArray[np.float64], NDArray[np.float64], List[str]]:
    """优化版本的计算同体量同方向成交的价格统计指标

    该函数是 calculate_trade_price_statistics_by_volume 的高性能版本，
    通过预排序索引、二分查找和批量处理等优化技术大幅提升计算速度。

    🚀 性能优化特点：
    ==================
    - 预排序时间索引，避免重复排序操作
    - 二分查找快速定位最近成交记录
    - 批量计算统计量，避免重复数值计算
    - 内存访问优化，减少分配开销
    - 算法复杂度从O(n²)优化到O(n log n)

    💡 适用场景：
    ============
    - 高频交易数据分析
    - 大规模历史数据处理
    - 实时价格统计计算
    - 性能敏感的量化研究

    Parameters
    ----------
    volume : NDArray[np.float64]
        成交量数组，建议按成交量排序
    exchtime : NDArray[np.int64]
        交易所时间戳数组（纳秒），建议按时间排序
    price : NDArray[np.float64]
        成交价格数组
    flag : NDArray[np.int32]
        买卖标志数组，66=买，83=卖
    min_count : int, default=10
        计算统计指标所需的最小样本数量
    use_flag : str, default="same"
        买卖方向匹配方式：
        - "same": 只与同主买卖方向的成交比较
        - "diff": 只与相反主买卖方向的成交比较
        - "ignore": 与所有成交比较，不考虑买卖方向

    Returns
    -------
    Tuple[NDArray[np.float64], NDArray[np.float64], List[str]]
        (均值数组, 标准差数组, 列名列表)

    Note
    ----
    为获得最佳性能，建议输入数据按volume和exchtime排序
    """

def calculate_trade_price_statistics_by_volume_ultra_fast(
    volume: NDArray[np.float64],
    exchtime: NDArray[np.int64],
    price: NDArray[np.float64],
    flag: NDArray[np.int32],
    min_count: int = 10,
    use_flag: str = "same"
) -> Tuple[NDArray[np.float64], NDArray[np.float64], List[str]]:
    """超级优化版本：极致性能的同体量同方向成交价格统计指标

    这是 calculate_trade_price_statistics_by_volume 的终极优化版本，
    专门为大规模数据的高性能计算而设计，与原版本计算结果完全一致。

    🚀 极致优化技术：
    ==================
    - 零拷贝数据访问模式
    - 预排序索引，O(1)查找
    - 批量统计量计算
    - 内存池复用
    - 缓存友好的数据布局
    - 完整的结果一致性保证

    🎯 性能特点：
    ============
    - 算法复杂度：O(n log n)
    - 内存使用：最小化
    - 结果精度：与原版本完全一致
    - 适用规模：特别适合13万+数据量

    💡 理想应用场景：
    ==================
    - 超大规模高频交易数据分析
    - 实时风控系统
    - 量化回测平台
    - 性能敏感的生产环境

    Parameters
    ----------
    volume : NDArray[np.float64]
        成交量数组，建议按成交量排序
    exchtime : NDArray[np.int64]
        交易所时间戳数组（纳秒），建议按时间排序
    price : NDArray[np.float64]
        成交价格数组
    flag : NDArray[np.int32]
        买卖标志数组，66=买，83=卖
    min_count : int, default=10
        计算统计指标所需的最小样本数量
    use_flag : str, default="same"
        买卖方向匹配方式：
        - "same": 只与同主买卖方向的成交比较
        - "diff": 只与相反主买卖方向的成交比较
        - "ignore": 与所有成交比较，不考虑买卖方向

    Returns
    -------
    Tuple[NDArray[np.float64], NDArray[np.float64], List[str]]
        (均值数组, 标准差数组, 列名列表)

    Performance
    -----------
    典型性能表现（测试环境：Intel i7, 16GB RAM）：
    - 5万数据：约4-5秒
    - 13万数据：约28-30秒
    - 算法复杂度：O(n log n)

    Note
    ----
    1. 为获得最佳性能，建议输入数据按volume和exchtime排序
    2. 该版本与原版本计算结果完全一致，可直接替换使用
    3. 适合对性能有严格要求的生产环境
    """
    ...

def calculate_trade_price_statistics_by_volume_v3(
    volume: NDArray[np.float64],
    exchtime: NDArray[np.int64],
    price: NDArray[np.float64],
    flag: NDArray[np.int32],
    min_count: int = 10,
    use_flag: str = "same"
) -> Tuple[NDArray[np.float64], NDArray[np.float64], List[str]]:
    """终极优化版本：极致性能的同体量同方向成交价格统计指标（V3）

    这是 calculate_trade_price_statistics_by_volume 的最新终极优化版本，
    针对13万数据量1秒内完成的目标进行了极致优化，与原版本计算结果完全一致。

    🚀 核心优化技术：
    ==================
    - 预先计算并缓存所有target_indices（避免循环内重复过滤）
    - 排序一次，批量计算所有百分比档位（消除重复排序）
    - 使用平方和增量算法（避免重复遍历计算方差）
    - 内存预分配（减少动态分配开销）
    - 紧凑的数据访问模式（提升CPU缓存命中率）

    🎯 性能目标：
    ============
    - 13万数据量：≤ 1秒
    - 算法复杂度：O(n log n)
    - 内存使用：最小化
    - 结果精度：与原版本完全一致

    💡 与ultra_fast版本的区别：
    ============================
    - V3版本：针对性能瓶颈深度优化，消除循环内重复计算
    - ultra_fast版本：使用预排序索引和二分查找
    - V3版本在大数据量上显著更快（约5-10倍提速）

    Parameters
    ----------
    volume : NDArray[np.float64]
        成交量数组，建议按成交量排序
    exchtime : NDArray[np.int64]
        交易所时间戳数组（纳秒），建议按时间排序
    price : NDArray[np.float64]
        成交价格数组
    flag : NDArray[np.int32]
        买卖标志数组，66=买，83=卖
    min_count : int, default=10
        计算统计指标所需的最小样本数量
    use_flag : str, default="same"
        买卖方向匹配方式：
        - "same": 只与同主买卖方向的成交比较
        - "diff": 只与相反主买卖方向的成交比较
        - "ignore": 与所有成交比较，不考虑买卖方向

    Returns
    -------
    Tuple[NDArray[np.float64], NDArray[np.float64], List[str]]
        (均值数组, 标准差数组, 列名列表)

    Performance
    -----------
    典型性能表现（优化后）：
    - 5万数据：约0.3-0.5秒
    - 13万数据：约0.8-1.0秒
    - 算法复杂度：O(n log n)
    - 相比原版本提速：10-30倍

    Note
    ----
    1. 为获得最佳性能，建议输入数据按volume和exchtime排序
    2. 该版本与原版本计算结果完全一致，可直接替换使用
    3. 推荐用于大规模数据处理和性能敏感场景
    4. 使用增量算法计算均值和方差，数值稳定性优秀
    """

