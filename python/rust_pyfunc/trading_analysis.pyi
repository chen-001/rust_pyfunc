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
    num_threads: int = 8
) -> Tuple[NDArray[np.float64], List[str]]:
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
        
    返回值：
    -------
    Tuple[NDArray[np.float64], List[str]]
        第一个元素：N行18列的分析结果矩阵
        第二个元素：包含18个特征名称的字符串列表
        
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
    exchtime: NDArray[np.float64],
    price: NDArray[np.float64],
    flag: NDArray[np.int32],
    ask_order: NDArray[np.int64],
    bid_order: NDArray[np.int64],
    min_count: int = 100,
    use_flag: str = "ignore"
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