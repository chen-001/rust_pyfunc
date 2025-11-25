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
    """

def fit_hawkes_process(event_times: NDArray[np.float64], event_volumes: NDArray[np.float64], initial_guess: Optional[Tuple[float, float, float]] = None, max_iterations: int = 1000, tolerance: float = 1e-06, cluster_merge_threshold: float = 0.8, max_parent_search_window: int = 200, parent_time_threshold_factor: float = 10.0, merge_search_window: int = 500, merge_time_threshold_factor: float = 20.0, relax_factor_multiplier: float = 3.0) -> dict:
    """拟合Hawkes自激点过程模型并计算多种指标

    该函数使用指数核函数 φ(u) = α * exp(-β * u) 拟合Hawkes过程,
    计算模型参数和各种金融指标,用于分析逐笔成交数据的自激特性。

    参数说明：
    ----------
    event_times : numpy.ndarray
        事件时间戳数组(单位:秒),需要是升序排列
    event_volumes : numpy.ndarray
        事件对应的成交量数组
    initial_guess : Optional[Tuple[float, float, float]], optional
        参数初始猜测值 (mu, alpha, beta)
        - mu: 外生事件强度(基准强度)
        - alpha: 自激强度系数
        - beta: 核函数衰减率
        默认为None,使用启发式方法自动初始化
    max_iterations : int, optional
        EM算法最大迭代次数,默认为1000
    tolerance : float, optional
        收敛容差,默认为1e-6
    cluster_merge_threshold : float, optional
        簇合并阈值(0-1),数值越小越容易把事件并入已有簇(同时放宽父子搜索窗口),默认0.8保持原有行为
    max_parent_search_window : int, optional
        计算期望子节点数时的最大搜索窗口,默认200
    parent_time_threshold_factor : float, optional
        期望子节点计算的时间阈值因子(相对于1/beta),默认10.0
    merge_search_window : int, optional
        搜索候选父节点时的基础窗口大小,默认500
    merge_time_threshold_factor : float, optional
        合并判断的时间阈值因子(相对于1/beta),默认20.0
        值越大,时间窗口越宽,越容易形成大簇
    relax_factor_multiplier : float, optional
        cluster_merge_threshold对时间窗口的影响倍数,默认3.0

    返回值：
    -------
    dict
        包含以下字段的字典:
        - 'mu': 外生事件强度估计值
        - 'alpha': 自激强度系数估计值
        - 'beta': 核函数衰减率估计值
        - 'branching_ratio': 分枝率 n = α/β
        - 'mean_intensity': 无条件平均强度 Λ = μ/(1-n)
        - 'exogenous_intensity': 外生强度 = μ
        - 'endogenous_intensity': 内生强度 = Λ - μ
        - 'expected_cluster_size': 期望簇大小 = 1/(1-n)
        - 'half_life': 半衰期 = ln(2)/β
        - 'mean_parent_child_interval': 父子平均间隔 = 1/β
        - 'log_likelihood': 对数似然值
        - 'event_intensities': 每个事件时刻的强度值
        - 'root_probabilities': 每个事件是根节点(外生事件)的概率
        - 'expected_children': 每个事件的预期子女数
        - 'cluster_assignments': 每个事件所属的簇ID
        - 'cluster_sizes': 每个簇的大小
        - 'cluster_durations': 每个簇的持续时间
        - 'cluster_volumes': 每个簇的成交量总和

    关键指标解释：
    ---------------
    1. 分枝率(branching_ratio): 表示一个事件平均能触发多少个直接后代事件,
       也近似等于内生事件占总事件的比例。n接近1表示强烈的自激效应。

    2. 平均强度(mean_intensity): 单位时间内事件的平均发生次数,
       包含了外生和内生两部分的贡献。

    3. 期望簇大小(expected_cluster_size): 一次自激过程平均包含的事件数量,
       包含根事件和所有后代事件。

    4. 根概率(root_probabilities): 每个事件是外生独立事件(而非被触发)的概率,
       概率大的事件可视为簇的主要触发者。

    5. 预期子女数(expected_children): 每个事件预计会触发多少个后续事件,
       反映事件的"影响力"。

    示例：
    -------
    >>> import rust_pyfunc as rp
    >>> import numpy as np
    >>> # 模拟逐笔成交数据
    >>> times = np.cumsum(np.random.exponential(0.1, 1000))
    >>> volumes = np.random.lognormal(10, 1, 1000)
    >>> result = rp.fit_hawkes_process(times, volumes, cluster_merge_threshold=0.6)
    >>> print(f"分枝率: {result['branching_ratio']:.3f}")
    >>> print(f"期望簇大小: {result['expected_cluster_size']:.2f}")
    >>> print(f"平均强度: {result['mean_intensity']:.3f}")

    参考文献：
    ----------
    Hawkes, A. G. (1971). Spectra of some self-exciting and mutually exciting point processes.
    Biometrika, 58(1), 83-90.

    Laub, P. J., Taimre, T., & Pollett, P. K. (2015). Hawkes processes.
    arXiv preprint arXiv:1507.02822.
    """
    ...

def hawkes_event_indicators(event_times: NDArray[np.float64], event_volumes: NDArray[np.float64], event_prices: NDArray[np.float64], initial_guess: Optional[Tuple[float, float, float]] = None, max_iterations: int = 1000, tolerance: float = 1e-06, cluster_merge_threshold: float = 0.8, max_parent_search_window: int = 200, parent_time_threshold_factor: float = 10.0, merge_search_window: int = 500, merge_time_threshold_factor: float = 20.0, relax_factor_multiplier: float = 3.0) -> dict:
    """计算Hawkes过程的事件级指标

    该函数在fit_hawkes_process的基础上,增加了需要价格数据的指标计算。
    额外计算的指标主要用于分析每个事件对价格的影响。

    参数说明：
    ----------
    event_times : numpy.ndarray
        事件时间戳数组(单位:秒),需要是升序排列
    event_volumes : numpy.ndarray
        事件对应的成交量数组
    event_prices : numpy.ndarray
        事件对应的价格数组
    initial_guess : Optional[Tuple[float, float, float]], optional
        参数初始猜测值 (mu, alpha, beta),默认为None
    max_iterations : int, optional
        EM算法最大迭代次数,默认为1000
    tolerance : float, optional
        收敛容差,默认为1e-6
    cluster_merge_threshold : float, optional
        簇合并阈值(0-1),数值越小越容易把事件并入已有簇(同时放宽父子搜索窗口),默认0.8保持原有行为
    max_parent_search_window : int, optional
        计算期望子节点数时的最大搜索窗口,默认200
    parent_time_threshold_factor : float, optional
        期望子节点计算的时间阈值因子(相对于1/beta),默认10.0
    merge_search_window : int, optional
        搜索候选父节点时的基础窗口大小,默认500
    merge_time_threshold_factor : float, optional
        合并判断的时间阈值因子(相对于1/beta),默认20.0
        值越大,时间窗口越宽,越容易形成大簇
    relax_factor_multiplier : float, optional
        cluster_merge_threshold对时间窗口的影响倍数,默认3.0

    返回值：
    -------
    dict
        包含fit_hawkes_process的所有字段,以及：
        - 'cluster_price_changes': 每个簇的价格变化(簇结束时价格 - 簇开始时价格)
        - 'time_intervals': 连续事件间的时间间隔

    新指标解释：
    ------------
    1. 簇价格变化(cluster_price_changes): 每个成交簇从开始到结束的价格变化,
       反映该簇交易活动对价格的影响方向和幅度。

    2. 时间间隔(time_intervals): 连续成交事件之间的时间间隔,
       可用于分析市场活跃度的时间模式。

    示例：
    -------
    >>> import rust_pyfunc as rp
    >>> import numpy as np
    >>> # 读取真实逐笔成交数据
    >>> df = read_trade_data('000001', 20220101)
    >>> df['time_seconds'] = (df.exchtime - df.exchtime.min()).dt.total_seconds()
    >>> result = rp.hawkes_event_indicators(
    ...     df.time_seconds.to_numpy(),
    ...     df.volume.to_numpy(),
    ...     df.price.to_numpy()
    ... )
    >>> # 分析大簇的价格影响
    >>> large_clusters = np.array(result['cluster_sizes']) > 10
    >>> price_changes = np.array(result['cluster_price_changes'])[large_clusters]
    >>> print(f"大簇平均价格变化: {np.mean(price_changes):.4f}")
    """
    ...

def analyze_hawkes_indicators(
    mu: float,
    alpha: float,
    beta: float,
    branching_ratio: float,
    mean_intensity: float,
    expected_cluster_size: float,
    half_life: float,
    cluster_sizes: NDArray[np.int32],
) -> dict:
    """分析Hawkes过程指标并提供交易建议

    该函数基于Hawkes模型输出指标，自动分析市场微观结构特征，
    并提供量化交易和策略建议。

    参数说明：
    ----------
    mu : float
        外生事件强度（基准强度）
    alpha : float
        自激强度系数
    beta : float
        核函数衰减率
    branching_ratio : float
        分枝率 n = α/β
    mean_intensity : float
        无条件平均强度 Λ = μ/(1-n)
    expected_cluster_size : float
        期望簇大小 = 1/(1-n)
    half_life : float
        半衰期 = ln(2)/β
    cluster_sizes : numpy.ndarray
        每个簇的大小数组

    返回值：
    -------
    dict
        包含以下字段的字典：
        - 'branching_ratio'：分枝率
        - 'branching_level'：分枝强度等级（极强/强/中等/较弱/弱/极弱）
        - 'branching_interpretation'：分枝率详细解读
        - 'cluster_size_score'：簇规模评分（0-1，越高聚集性越强）
        - 'cluster_interpretation'：簇分布详细解读
        - 'market_memory_score'：市场记忆评分（0-1，越高记忆越短）
        - 'memory_interpretation'：市场记忆详细解读
        - 'overall_market_state'：整体市场状态综合评估
        - 'trading_suggestions'：具体交易建议列表
        - 'total_clusters'：总簇数量
        - 'large_clusters_10'：包含10+事件的簇数量
        - 'max_cluster_size'：最大簇的大小

    交易建议类别：
    ---------------
    1. 策略方向：趋势跟踪（强分枝）或均值回归（弱分枝）
    2. 时间框架：基于半衰期确定持仓周期
    3. 交易信号：大簇形成/结束时的入场/出场点
    4. 风险管理：根据聚集程度调整仓位大小

    示例：
    -------
    >>> import rust_pyfunc as rp
    >>> import numpy as np
    >>> # 先计算Hawkes指标
    >>> indicators = rp.fit_hawkes_process(times, volumes)
    >>> # 分析指标并获取建议
    >>> analysis = rp.analyze_hawkes_indicators(
    ...     indicators['mu'],
    ...     indicators['alpha'],
    ...     indicators['beta'],
    ...     indicators['branching_ratio'],
    ...     indicators['mean_intensity'],
    ...     indicators['expected_cluster_size'],
    ...     indicators['half_life'],
    ...     indicators['cluster_sizes']
    ... )
    >>> print(analysis['overall_market_state'])
    >>> for suggestion in analysis['trading_suggestions']:
    ...     print(suggestion)
    """
    ...
