"""交易分析函数类型声明"""

from typing import List, Optional, Tuple
import numpy as np
from numpy.typing import NDArray

def find_follow_volume_sum_same_price(
    times: NDArray[np.float64],
    prices: NDArray[np.float64],
    volumes: NDArray[np.float64],
    time_window: float = 0.1,
    check_price: bool = True,
    filter_ratio: float = 0.0,
    timeout_seconds: Optional[float] = None,
) -> NDArray[np.float64]:
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

def find_follow_volume_sum_same_price_and_flag(
    times: NDArray[np.float64],
    prices: NDArray[np.float64],
    volumes: NDArray[np.float64],
    flags: NDArray[np.int32],
    time_window: float = 0.1,
) -> NDArray[np.float64]:
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

def mark_follow_groups(
    times: NDArray[np.float64],
    prices: NDArray[np.float64],
    volumes: NDArray[np.float64],
    time_window: float = 0.1,
) -> NDArray[np.int32]:
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

def mark_follow_groups_with_flag(
    times: NDArray[np.float64],
    prices: NDArray[np.float64],
    volumes: NDArray[np.float64],
    flags: NDArray[np.int64],
    time_window: float = 0.1,
) -> NDArray[np.int32]:
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
    time_window_minutes: Optional[float] = 1.0,
) -> Tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
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

def reconstruct_limit_order_lifecycle(
    ticks_array: NDArray[np.float64],
    snaps_array: NDArray[np.float64],
) -> NDArray[np.float64]:
    """基于逐笔成交与盘口快照重建限价单生命周期特征。

    参数说明：
    ----------
    ticks_array : numpy.ndarray
        逐笔成交二维数组，列为exchtime, price, volume, turnover, flag, ask_order, bid_order
    snaps_array : numpy.ndarray
        盘口快照二维数组，列为exchtime + bid_prc1-10 + bid_vol1-10 + ask_prc1-10 + ask_vol1-10

    返回值：
    -------
    numpy.ndarray
        特征二维数组，每行对应一个(快照, 档位, 买卖方向)
    """


def reconstruct_limit_order_lifecycle_v2(
    ticks_array: NDArray[np.float64],
    snaps_array: NDArray[np.float64],
) -> NDArray[np.float64]:
    """基于逐笔成交与盘口快照重建限价单生命周期特征 (v2版本)。

    功能：通过匹配高频逐笔成交数据与盘口快照数据，使用订单ID作为追踪器，
    重建限价单生命周期指标。

    计算过程：
    ----------
    对于每个快照时刻 S_t 和每个档位 L (1-10档) 的买卖双侧：

    Phase A: 锚定 (Anchoring)
        - 在逐笔成交流中定位 S_t 的时间戳位置
        - 计算截至 S_t.exchtime 观察到的最大订单ID：
          MAX_BID_ID_t = 最大 bid_order ≤ S_t.exchtime
          MAX_ASK_ID_t = 最大 ask_order ≤ S_t.exchtime
        - 确定目标价格：
          Bid侧: P_target = BidPrc_L, ID_limit = MAX_BID_ID_t
          Ask侧: P_target = AskPrc_L, ID_limit = MAX_ASK_ID_t

    Phase B: 窗口扫描 (Window Scanning)
        - 从 S_t.exchtime 开始向前遍历逐笔成交数据
        - 直到触发终止条件（档位被突破）：
          Bid侧: 出现 tick.price < P_target（价格跌破目标档位）
          Ask侧: 出现 tick.price > P_target（价格涨破目标档位）

    Phase C: 交易匹配 (Trade Matching)
        在扫描窗口内，收集同时满足以下条件的逐笔成交：
        1. 价格匹配: tick.price == P_target
        2. 方向匹配:
           Bid侧: tick.flag == 83 (主动卖单击中被动买单)
           Ask侧: tick.flag == 66 (主动买单击中被动卖单)
        3. ID验证:
           Bid侧: tick.bid_order ≤ ID_limit (确认为快照前已存在的挂单)
           Ask侧: tick.ask_order ≤ ID_limit

    Phase D: 特征工程 (Feature Engineering)
        对收集到的匹配交易计算统计特征：

        Group 1 - 成交量统计 (恢复的流动性):
        - vol_sum: 成交量总和
        - vol_mean: 成交量均值
        - vol_std: 成交量标准差
        - vol_skew: 成交量偏度
        - vol_autocorr: 成交量滞后1阶自相关
        - vol_trend: 成交量与序列[1,2,...,N]的趋势相关性

        Group 2 - 订单ID统计 ("地质"年龄):
        - 预处理: ΔID = ID_limit - tick.passive_ID
        - id_count: 匹配交易数量
        - id_span: max(ΔID) - min(ΔID)
        - id_mean_diff: ΔID的均值
        - id_std_diff: ΔID的标准差
        - id_skew_diff: ΔID的偏度
        - id_trend: 绝对订单ID与交易序列索引的趋势相关性
          (正值=先吃老订单，负值=先吃新订单)

    参数说明：
    ----------
    ticks_array : numpy.ndarray
        逐笔成交二维数组 (N_ticks x 7)，列：
        - 0: exchtime (时间戳)
        - 1: price (成交价格)
        - 2: volume (成交量)
        - 3: turnover (成交金额)
        - 4: flag (交易标志: 66=主买, 83=主卖, 32=撤单)
        - 5: ask_order (卖单订单ID)
        - 6: bid_order (买单订单ID)

    snaps_array : numpy.ndarray
        盘口快照二维数组 (N_snaps x 41+)，列：
        - 0: exchtime (时间戳)
        - 1-10: bid_prc1-10 (买价1-10档)
        - 11-20: bid_vol1-10 (买量1-10档)
        - 21-30: ask_prc1-10 (卖价1-10档)
        - 31-40: ask_vol1-10 (卖量1-10档)

    返回值：
    -------
    numpy.ndarray
        特征二维数组 (N_snaps * 20 x 15)，每行对应一个(快照, 档位, 买卖方向)

        列索引 | 列名           | 说明
        ------|---------------|-------------------------------------------
        0     | timestamp     | 快照时间戳
        1     | side_flag     | 买卖方向 (0=Bid, 1=Ask)
        2     | level_index   | 档位 (1-10)
        3     | vol_sum       | 成交量总和
        4     | vol_mean      | 成交量均值
        5     | vol_std       | 成交量标准差
        6     | vol_skew      | 成交量偏度
        7     | vol_autocorr  | 成交量滞后1阶自相关
        8     | vol_trend     | 成交量趋势相关性
        9     | id_count      | 匹配订单数量
        10    | id_span       | 订单ID跨度
        11    | id_mean_diff  | 订单ID差值均值 (ID_limit - ID)
        12    | id_std_diff   | 订单ID差值标准差
        13    | id_skew_diff  | 订单ID差值偏度
        14    | id_trend      | 订单ID趋势相关性 (绝对ID vs 序列索引)

    注意：
    -----
    - 当某个档位没有匹配到交易时，vol_sum=0, id_count=0，其余统计量为NaN
    - 近档位(1-3档)通常有较高的匹配率，远档位(8-10档)匹配率较低
    """


def fit_hawkes_process(
    event_times: NDArray[np.float64],
    event_volumes: NDArray[np.float64],
    initial_guess: Optional[Tuple[float, float, float]] = None,
    max_iterations: int = 1000,
    tolerance: float = 1e-06,
    cluster_merge_threshold: float = 0.8,
    max_parent_search_window: int = 200,
    parent_time_threshold_factor: float = 10.0,
    merge_search_window: int = 500,
    merge_time_threshold_factor: float = 20.0,
    relax_factor_multiplier: float = 3.0,
) -> dict:
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

def hawkes_event_indicators(
    event_times: NDArray[np.float64],
    event_volumes: NDArray[np.float64],
    event_prices: NDArray[np.float64],
    initial_guess: Optional[Tuple[float, float, float]] = None,
    max_iterations: int = 1000,
    tolerance: float = 1e-06,
    cluster_merge_threshold: float = 0.8,
    max_parent_search_window: int = 200,
    parent_time_threshold_factor: float = 10.0,
    merge_search_window: int = 500,
    merge_time_threshold_factor: float = 20.0,
    relax_factor_multiplier: float = 3.0,
) -> dict:
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

def calculate_passive_order_features(
    trade_times: NDArray[np.int64],
    trade_flags: NDArray[np.int32],
    trade_bid_orders: NDArray[np.int64],
    trade_ask_orders: NDArray[np.int64],
    trade_volumes: NDArray[np.int64],
    market_times: NDArray[np.int64],
    compute_direction_ratio: bool = True,
    compute_flag_ratio: bool = True,
) -> Tuple[NDArray[np.float64], List[str]]:
    """计算被动订单特征

    对于每两个相邻的盘口快照之间的逐笔成交记录，识别被动方的订单编号，
    并计算以下统计特征：

    基础特征（42个）：
    - 全部/买单/卖单的被动订单编号统计特征（7×3）
    - 全部/买单/卖单的订单体量统计特征（7×3）

    方向比例特征（21个，可选）：
    - 每个被动订单前后50笔成交中，与自己成交方向相同的比例序列的统计特征
    - 全部/买单/卖单各计算一组（7×3）

    Flag比例特征（21个，可选）：
    - 每个被动订单前后50笔成交中，与自己主买主卖标识相同的比例序列的统计特征
    - 全部/买单/卖单各计算一组（7×3）

    统计特征包括：均值、标准差、偏度、峰度、自相关系数、趋势、LZ复杂度

    参数说明：
    ----------
    trade_times : numpy.ndarray[int64]
        逐笔成交时间戳（纳秒级）
    trade_flags : numpy.ndarray[int32]
        交易标志 (66=主买, 83=主卖)
    trade_bid_orders : numpy.ndarray[int64]
        买单订单编号
    trade_ask_orders : numpy.ndarray[int64]
        卖单订单编号
    trade_volumes : numpy.ndarray[int64]
        成交量
    market_times : numpy.ndarray[int64]
        盘口快照时间戳（纳秒级）
    compute_direction_ratio : bool, optional, default=True
        是否计算方向比例特征
    compute_flag_ratio : bool, optional, default=True
        是否计算flag比例特征

    返回值：
    -------
    Tuple[numpy.ndarray[float64], List[str]]
        一个元组，包含：
        - 特征数组: 形状为 (N-1, features) 的二维数组，其中N是盘口快照数
        - 列名列表: 包含特征列名的列表

        特征数量取决于参数：
        - 基础特征: 42个
        - 如果compute_direction_ratio=True: 额外21个
        - 如果compute_flag_ratio=True: 额外21个
        - 最多: 84个

    示例：
    -------
    >>> import rust_pyfunc as rp
    >>> import pure_ocean_breeze.jason as p
    >>>
    >>> # 读取数据
    >>> code = '600000'
    >>> date = 20220819
    >>> trade_data = p.adjust_afternoon(p.read_trade(code, date))
    >>> market_data = p.adjust_afternoon(p.read_market(code, date))
    >>>
    >>> # 准备数据
    >>> trade_times = trade_data['exchtime'].astype(np.int64).values
    >>> trade_flags = trade_data['flag'].astype(np.int32).values
    >>> trade_bid_orders = trade_data['bid_order'].values
    >>> trade_ask_orders = trade_data['ask_order'].values
    >>> trade_volumes = trade_data['volume'].values
    >>> market_times = market_data['exchtime'].astype(np.int64).values
    >>>
    >>> # 计算被动订单特征
    >>> features, column_names = rp.calculate_passive_order_features(
    ...     trade_times, trade_flags, trade_bid_orders,
    ...     trade_ask_orders, trade_volumes, market_times,
    ...     compute_direction_ratio=True,
    ...     compute_flag_ratio=True
    ... )
    >>> print(f"特征矩阵形状: {features.shape}")
    >>> print(f"列名: {column_names}")
    """
    ...


def calculate_microstructure_pattern_features(
    trade_times: NDArray[np.int64],
    trade_prices: NDArray[np.float64],
    trade_volumes: NDArray[np.float64],
    trade_flags: NDArray[np.int32],
    trade_bid_orders: NDArray[np.int64],
    trade_ask_orders: NDArray[np.int64],
    window_size: int = 100,
    histogram_bins: int = 10,
    marginal_window: int = 50,
) -> Tuple[NDArray[np.float64], List[str], List[float], List[str]]:
    """计算微观结构模式差异特征序列（原始版本）
    
    注意: 此版本已过时，建议使用 calculate_microstructure_pattern_features_optimized
    """
    ...

def calculate_microstructure_pattern_features_optimized(
    trade_times: NDArray[np.int64],
    trade_prices: NDArray[np.float64],
    trade_volumes: NDArray[np.float64],
    trade_flags: NDArray[np.int32],
    trade_bid_orders: NDArray[np.int64],
    trade_ask_orders: NDArray[np.int64],
    window_size: int = 100,
    histogram_bins: int = 10,
    marginal_window: int = 50,
    dtw_band_width: int = 10,
) -> Tuple[NDArray[np.float64], List[str], List[float], List[str]]:
    """计算微观结构模式差异特征序列（优化版本）
    
    该函数实现三步法构建日内特征序列：
    1. 模式定义：从逐笔成交数据中提取微观结构模式
    2. 差异度量：计算两个时间片段之间的模式差异
    3. 序列构造：通过移动时间窗生成日内特征序列
    
    模式类型（第一步）：
    -----------------
    标量模式：
    - order_id_gap: 订单代沟 |BuyID - SellID|
    - volume: 成交量
    - turnover: 成交金额（使用volume代替）
    - is_buy_flag: 是否主买 (1/0)
    
    直方图模式：
    - volume_hist: 成交量分布直方图
    - order_id_gap_hist: 订单代沟分布直方图
    
    排序序型：
    - volume_rank: 成交量排序序型
    - order_id_gap_rank: 订单代沟排序序型
    
    单独保留的price相关特征：
    - price_change_variance_ratio_cumul_vs_marginal: 价格变动的方差比（累积vs边际）
    
    差异度量（第二步）：
    -----------------
    标量差异：
    - mean_diff: 均值差异
    - dtw: 动态时间规整距离（使用欧氏距离优化）
    
    直方图差异：
    - wasserstein: Wasserstein距离
    - js_divergence: Jensen-Shannon散度
    - ks_stat: KS统计量
    
    序列差异：
    - edit_dist: 编辑距离（使用位并行Myers算法优化）
    - jaccard: Jaccard相似系数（转换为距离）
    
    序列构造方式（第三步）：
    ---------------------
    - moving_split: 移动切分点，对比[Start, t]与[t, End]
    - adjacent_roll: 孪生滑窗，对比[t-2W, t-W]与[t-W, t]
    - cumul_vs_marginal: 累积vs边际，对比[0, t]与[t-W, t]
    
    特征组合：
    ---------
    共生成 (4标量×2差异×3构造) + (2直方图×3差异×3构造) + 
          (2排序×2差异×3构造) + 1个单独price特征 = 55个日内特征序列
    
    标量特征：
    ---------
    对每个差异序列，计算最小值点和最大值点的以下特征：
    - time_diff: 前后两部分的时间长度差（秒）
    - vol_ratio_diff: 前后两部分的成交量占比差
    - buy_ratio_diff: 前后两部分的主买占比差
    
    共 55×6 = 330 个标量特征
    
    参数说明：
    ----------
    trade_times : NDArray[np.int64]
        逐笔成交时间戳（纳秒级）
    trade_prices : NDArray[np.float64]
        逐笔成交价格
    trade_volumes : NDArray[np.float64]
        逐笔成交量
    trade_flags : NDArray[np.int32]
        逐笔成交标志（66=主买, 83=主卖）
    trade_bid_orders : NDArray[np.int64]
        买单订单编号
    trade_ask_orders : NDArray[np.int64]
        卖单订单编号
    window_size : int, optional
        滑动窗口大小，默认100
    histogram_bins : int, optional
        直方图分箱数，默认10
    marginal_window : int, optional
        边际窗口大小，默认50
    dtw_band_width : int, optional
        DTW距离的Sakoe-Chiba带宽限制，默认10。较小的值计算更快但精度略低。
    
    优化说明：
    -----------
    相比原始版本，优化版本实现了以下改进：
    1. 编辑距离：使用Myers位并行算法（O(ceil(m/w)×n) vs O(m×n)）
    2. DTW距离：使用欧氏距离替代，复杂度从O(n×band_width)降至O(n)
    3. 直方图合并：使用前缀和优化，O(1)区间查询
    4. 标量特征：预计算前缀和，避免重复遍历
    5. 排序序型：使用归一化L1距离替代编辑距离（等长序列）
    
    性能提升：
    - 500条数据：约5-6倍加速
    - 1000条数据：约10-11倍加速
    - 数据量越大，加速效果越明显
    
    返回值：
    -------
    Tuple[NDArray[np.float64], List[str], List[float], List[str]]
        - 特征矩阵: 形状为 (n_trades, 55) 的二维数组，每列是一个日内特征序列
        - 特征名称列表: 55个特征序列的名称
        - 标量特征列表: 330个标量特征值
        - 标量特征名称列表: 330个标量特征的名称
    
    示例：
    -------
    >>> import pure_ocean_breeze.jason as p
    >>> import rust_pyfunc as rp
    >>> import numpy as np
    >>> 
    >>> # 读取数据
    >>> trade_data = p.adjust_afternoon(p.read_trade('000001', 20220819))
    >>> 
    >>> # 准备数据
    >>> trade_times = trade_data['exchtime'].astype(np.int64).values
    >>> trade_prices = trade_data['price'].values
    >>> trade_volumes = trade_data['volume'].astype(np.float64).values
    >>> trade_flags = trade_data['flag'].astype(np.int32).values
    >>> trade_bid_orders = trade_data['bid_order'].values
    >>> trade_ask_orders = trade_data['ask_order'].values
    >>> 
    >>> # 计算特征
    >>> features, names, scalar_features, scalar_names = rp.calculate_microstructure_pattern_features_optimized(
    ...     trade_times, trade_prices, trade_volumes, trade_flags,
    ...     trade_bid_orders, trade_ask_orders,
    ...     window_size=100, histogram_bins=10, marginal_window=50
    ... )
    >>> print(f"特征矩阵形状: {features.shape}")
    >>> print(f"特征序列数量: {len(names)}")
    >>> print(f"标量特征数量: {len(scalar_features)}")
    """
    ...


def compute_allo_microstructure_features(
    trade_exchtime: NDArray[np.int64],
    trade_price: NDArray[np.float64],
    trade_volume: NDArray[np.float64],
    trade_turnover: NDArray[np.float64],
    trade_flag: NDArray[np.int32],
    snap_exchtime: NDArray[np.int64],
    bid_prc1: NDArray[np.float64],
    bid_prc2: NDArray[np.float64],
    bid_prc3: NDArray[np.float64],
    bid_prc4: NDArray[np.float64],
    bid_prc5: NDArray[np.float64],
    bid_prc6: NDArray[np.float64],
    bid_prc7: NDArray[np.float64],
    bid_prc8: NDArray[np.float64],
    bid_prc9: NDArray[np.float64],
    bid_prc10: NDArray[np.float64],
    bid_vol1: NDArray[np.float64],
    bid_vol2: NDArray[np.float64],
    bid_vol3: NDArray[np.float64],
    bid_vol4: NDArray[np.float64],
    bid_vol5: NDArray[np.float64],
    bid_vol6: NDArray[np.float64],
    bid_vol7: NDArray[np.float64],
    bid_vol8: NDArray[np.float64],
    bid_vol9: NDArray[np.float64],
    bid_vol10: NDArray[np.float64],
    ask_prc1: NDArray[np.float64],
    ask_prc2: NDArray[np.float64],
    ask_prc3: NDArray[np.float64],
    ask_prc4: NDArray[np.float64],
    ask_prc5: NDArray[np.float64],
    ask_prc6: NDArray[np.float64],
    ask_prc7: NDArray[np.float64],
    ask_prc8: NDArray[np.float64],
    ask_prc9: NDArray[np.float64],
    ask_prc10: NDArray[np.float64],
    ask_vol1: NDArray[np.float64],
    ask_vol2: NDArray[np.float64],
    ask_vol3: NDArray[np.float64],
    ask_vol4: NDArray[np.float64],
    ask_vol5: NDArray[np.float64],
    ask_vol6: NDArray[np.float64],
    ask_vol7: NDArray[np.float64],
    ask_vol8: NDArray[np.float64],
    ask_vol9: NDArray[np.float64],
    ask_vol10: NDArray[np.float64],
    detection_mode: str = "both",
    side_filter: str = "both",
    k1_horizontal: float = 2.0,
    k2_vertical: float = 5.0,
    window_size: int = 100,
    decay_threshold: float = 0.5,
) -> Tuple[NDArray[np.float64], List[str]]:
    """
    计算非对称大挂单（ALLO）微观结构特征

    该函数检测"异常流动性聚集事件"(ALA)，并计算21个微观结构特征指标。

    参数：
    -----
    trade_exchtime : NDArray[np.int64]
        逐笔成交时间戳（纳秒）
    trade_price : NDArray[np.float64]
        逐笔成交价格
    trade_volume : NDArray[np.float64]
        逐笔成交量
    trade_turnover : NDArray[np.float64]
        逐笔成交金额
    trade_flag : NDArray[np.int32]
        逐笔成交标志（66=主买, 83=主卖）
    snap_exchtime : NDArray[np.int64]
        盘口快照时间戳（纳秒）
    bid_prc1-10 : NDArray[np.float64]
        买一到买十价格
    bid_vol1-10 : NDArray[np.float64]
        买一到买十挂单量
    ask_prc1-10 : NDArray[np.float64]
        卖一到卖十价格
    ask_vol1-10 : NDArray[np.float64]
        卖一到卖十挂单量
    detection_mode : str, optional
        检测模式："horizontal"、"vertical"、"both" 或 "tris"（默认"both"）
        - "horizontal": 单档位挂单量 > k1 * 其他档位总和
        - "vertical": 单档位挂单量 > k2 * 历史移动平均
        - "both": 同时满足横向或纵向条件之一
        - "tris": 返回所有三种模式的结果（horizontal/vertical/both），列名带前缀
    side_filter : str, optional
        买卖侧过滤："bid"、"ask"、"both" 或 "tris"（默认"both"）
        - "bid": 只检测买入侧的异常大挂单
        - "ask": 只检测卖出侧的异常大挂单
        - "both": 同时检测买卖两侧
        - "tris": 返回所有三种侧过滤的结果（bid/ask/both），列名带前缀
    k1_horizontal : float, optional
        横向阈值（默认2.0）
    k2_vertical : float, optional
        纵向阈值（默认5.0）
    window_size : int, optional
        纵向移动窗口大小（默认100）
    decay_threshold : float, optional
        事件结束的衰减阈值（默认0.5）

    返回：
    -----
    Tuple[NDArray[np.float64], List[str]]
        - 非tris模式: features_array形状为(n_events, 21)的特征矩阵
        - tris模式: features_array形状为(1, n_combinations*21)的均值特征矩阵
          当detection_mode="tris"且side_filter="tris"时，返回9组×21个特征=189列
          列名格式: "{detection_mode}_{side_filter}_{feature_name}"
        - feature_names: 特征名称列表

    特征说明（21个事件级特征）：
    -------------------------
    第一部分：巨石的物理属性
    - M1_relative_prominence: 相对凸度
    - M3_flicker_frequency: 闪烁频率

    第二部分：攻城战的流体力学
    - M7_queue_loitering_duration: 队列滞留时长

    第三部分：友军的生态结构
    - M8_frontrun_passive: 抢跑强度-挂单版
    - M9_frontrun_active: 抢跑强度-主买版
    - M10_ally_retreat_rate: 同侧撤单率

    第四部分：群体行为的时间形态学（对手攻击单）
    - M11a_attack_skewness_opponent: 攻击偏度-对手盘（正偏=闪电战，负偏=围攻战）
    - M12a_peak_latency_ratio_opponent: 峰值延迟率-对手盘（接近1=扫尾清场）
    - M13a_courage_acceleration_opponent: 勇气加速度-对手盘（正=信心增强，负=强弩之末）
    - M14a_rhythm_entropy_opponent: 节奏熵-对手盘（低熵=拆单算法，高熵=人类博弈）

    第四部分：群体行为的时间形态学（同侧抢跑单）
    - M11b_attack_skewness_ally: 攻击偏度-同侧
    - M12b_peak_latency_ratio_ally: 峰值延迟率-同侧
    - M13b_courage_acceleration_ally: 勇气加速度-同侧
    - M14b_rhythm_entropy_ally: 节奏熵-同侧

    第五部分：空间场论与距离效应
    - M15_fox_tiger_index: 狐假虎威指数
    - M16_shadow_projection_ratio: 阴影投射比
    - M17_gravitational_redshift: 引力红移速率
    - M19_shielding_thickness_ratio: 垫单厚度比

    第六部分：命运与结局
    - M20_oxygen_saturation: 氧气饱和度
    - M21_suffocation_integral: 窒息深度积分
    - M22_local_survivor_bias: 幸存者偏差-邻域版

    示例：
    -----
    >>> import pure_ocean_breeze.jason as p
    >>> import rust_pyfunc as rp
    >>> import numpy as np
    >>>
    >>> # 读取数据
    >>> trade_data = p.adjust_afternoon(p.read_trade('000001', 20220819))
    >>> market_data = p.adjust_afternoon(p.read_market('000001', 20220819))
    >>>
    >>> # 准备逐笔成交数据
    >>> trade_exchtime = trade_data['exchtime'].astype(np.int64).values
    >>> trade_price = trade_data['price'].values
    >>> trade_volume = trade_data['volume'].astype(np.float64).values
    >>> trade_turnover = trade_data['turnover'].values
    >>> trade_flag = trade_data['flag'].astype(np.int32).values
    >>>
    >>> # 准备盘口快照数据
    >>> snap_exchtime = market_data['exchtime'].astype(np.int64).values
    >>> bid_prc = [market_data[f'bid_prc{i}'].values for i in range(1, 11)]
    >>> bid_vol = [market_data[f'bid_vol{i}'].values for i in range(1, 11)]
    >>> ask_prc = [market_data[f'ask_prc{i}'].values for i in range(1, 11)]
    >>> ask_vol = [market_data[f'ask_vol{i}'].values for i in range(1, 11)]
    >>>
    >>> # 计算ALLO特征（只检测买入侧）
    >>> features, feature_names = rp.compute_allo_microstructure_features(
    ...     trade_exchtime, trade_price, trade_volume, trade_turnover, trade_flag,
    ...     snap_exchtime,
    ...     *bid_prc, *bid_vol, *ask_prc, *ask_vol,
    ...     detection_mode="both",
    ...     side_filter="bid",
    ...     k1_horizontal=2.0,
    ...     k2_vertical=5.0
    ... )
    >>> print(f"检测到 {features.shape[0]} 个ALA事件")
    >>> print(f"特征数: {features.shape[1]}")
    """
    ...

def compute_allo_microstructure_features_tris_expanded(
    trade_exchtime: NDArray[np.int64],
    trade_price: NDArray[np.float64],
    trade_volume: NDArray[np.float64],
    trade_turnover: NDArray[np.float64],
    trade_flag: NDArray[np.int32],
    snap_exchtime: NDArray[np.int64],
    bid_prc1: NDArray[np.float64],
    bid_prc2: NDArray[np.float64],
    bid_prc3: NDArray[np.float64],
    bid_prc4: NDArray[np.float64],
    bid_prc5: NDArray[np.float64],
    bid_prc6: NDArray[np.float64],
    bid_prc7: NDArray[np.float64],
    bid_prc8: NDArray[np.float64],
    bid_prc9: NDArray[np.float64],
    bid_prc10: NDArray[np.float64],
    bid_vol1: NDArray[np.float64],
    bid_vol2: NDArray[np.float64],
    bid_vol3: NDArray[np.float64],
    bid_vol4: NDArray[np.float64],
    bid_vol5: NDArray[np.float64],
    bid_vol6: NDArray[np.float64],
    bid_vol7: NDArray[np.float64],
    bid_vol8: NDArray[np.float64],
    bid_vol9: NDArray[np.float64],
    bid_vol10: NDArray[np.float64],
    ask_prc1: NDArray[np.float64],
    ask_prc2: NDArray[np.float64],
    ask_prc3: NDArray[np.float64],
    ask_prc4: NDArray[np.float64],
    ask_prc5: NDArray[np.float64],
    ask_prc6: NDArray[np.float64],
    ask_prc7: NDArray[np.float64],
    ask_prc8: NDArray[np.float64],
    ask_prc9: NDArray[np.float64],
    ask_prc10: NDArray[np.float64],
    ask_vol1: NDArray[np.float64],
    ask_vol2: NDArray[np.float64],
    ask_vol3: NDArray[np.float64],
    ask_vol4: NDArray[np.float64],
    ask_vol5: NDArray[np.float64],
    ask_vol6: NDArray[np.float64],
    ask_vol7: NDArray[np.float64],
    ask_vol8: NDArray[np.float64],
    ask_vol9: NDArray[np.float64],
    ask_vol10: NDArray[np.float64],
    k1_horizontal: float = 2.0,
    k2_vertical: float = 5.0,
    window_size: int = 100,
    decay_threshold: float = 0.5,
) -> Tuple[List[NDArray[np.float64]], List[List[str]]]:
    """计算ALA事件微观结构特征 - tris扩展版本（档位锚定）。
    
    返回9种(detection_mode, side_filter)组合的特征数组。
    """
    ...

def compute_allo_microstructure_features_tris_expanded_v2(
    trade_exchtime: NDArray[np.int64],
    trade_price: NDArray[np.float64],
    trade_volume: NDArray[np.float64],
    trade_turnover: NDArray[np.float64],
    trade_flag: NDArray[np.int32],
    snap_exchtime: NDArray[np.int64],
    bid_prc1: NDArray[np.float64],
    bid_prc2: NDArray[np.float64],
    bid_prc3: NDArray[np.float64],
    bid_prc4: NDArray[np.float64],
    bid_prc5: NDArray[np.float64],
    bid_prc6: NDArray[np.float64],
    bid_prc7: NDArray[np.float64],
    bid_prc8: NDArray[np.float64],
    bid_prc9: NDArray[np.float64],
    bid_prc10: NDArray[np.float64],
    bid_vol1: NDArray[np.float64],
    bid_vol2: NDArray[np.float64],
    bid_vol3: NDArray[np.float64],
    bid_vol4: NDArray[np.float64],
    bid_vol5: NDArray[np.float64],
    bid_vol6: NDArray[np.float64],
    bid_vol7: NDArray[np.float64],
    bid_vol8: NDArray[np.float64],
    bid_vol9: NDArray[np.float64],
    bid_vol10: NDArray[np.float64],
    ask_prc1: NDArray[np.float64],
    ask_prc2: NDArray[np.float64],
    ask_prc3: NDArray[np.float64],
    ask_prc4: NDArray[np.float64],
    ask_prc5: NDArray[np.float64],
    ask_prc6: NDArray[np.float64],
    ask_prc7: NDArray[np.float64],
    ask_prc8: NDArray[np.float64],
    ask_prc9: NDArray[np.float64],
    ask_prc10: NDArray[np.float64],
    ask_vol1: NDArray[np.float64],
    ask_vol2: NDArray[np.float64],
    ask_vol3: NDArray[np.float64],
    ask_vol4: NDArray[np.float64],
    ask_vol5: NDArray[np.float64],
    ask_vol6: NDArray[np.float64],
    ask_vol7: NDArray[np.float64],
    ask_vol8: NDArray[np.float64],
    ask_vol9: NDArray[np.float64],
    ask_vol10: NDArray[np.float64],
    k1_horizontal: float = 2.0,
    k2_vertical: float = 5.0,
    window_size: int = 100,
    decay_threshold: float = 0.5,
) -> Tuple[List[NDArray[np.float64]], List[List[str]]]:
    """计算ALA事件微观结构特征 v2版本（价格锚定）。
    
    v2版本改进：
    - 纵向触发：计算某价格的近期平均挂单量（而非某档位）
    - 事件追踪：按价格搜索，而非按档位
    - 结束条件：价格消失或挂单量衰减
    
    返回9种(detection_mode, side_filter)组合的特征数组。
    """
    ...

def compute_allo_microstructure_features_tris_expanded_v3(
    trade_exchtime: NDArray[np.int64],
    trade_price: NDArray[np.float64],
    trade_volume: NDArray[np.float64],
    trade_turnover: NDArray[np.float64],
    trade_flag: NDArray[np.int32],
    snap_exchtime: NDArray[np.int64],
    bid_prc1: NDArray[np.float64],
    bid_prc2: NDArray[np.float64],
    bid_prc3: NDArray[np.float64],
    bid_prc4: NDArray[np.float64],
    bid_prc5: NDArray[np.float64],
    bid_prc6: NDArray[np.float64],
    bid_prc7: NDArray[np.float64],
    bid_prc8: NDArray[np.float64],
    bid_prc9: NDArray[np.float64],
    bid_prc10: NDArray[np.float64],
    bid_vol1: NDArray[np.float64],
    bid_vol2: NDArray[np.float64],
    bid_vol3: NDArray[np.float64],
    bid_vol4: NDArray[np.float64],
    bid_vol5: NDArray[np.float64],
    bid_vol6: NDArray[np.float64],
    bid_vol7: NDArray[np.float64],
    bid_vol8: NDArray[np.float64],
    bid_vol9: NDArray[np.float64],
    bid_vol10: NDArray[np.float64],
    ask_prc1: NDArray[np.float64],
    ask_prc2: NDArray[np.float64],
    ask_prc3: NDArray[np.float64],
    ask_prc4: NDArray[np.float64],
    ask_prc5: NDArray[np.float64],
    ask_prc6: NDArray[np.float64],
    ask_prc7: NDArray[np.float64],
    ask_prc8: NDArray[np.float64],
    ask_prc9: NDArray[np.float64],
    ask_prc10: NDArray[np.float64],
    ask_vol1: NDArray[np.float64],
    ask_vol2: NDArray[np.float64],
    ask_vol3: NDArray[np.float64],
    ask_vol4: NDArray[np.float64],
    ask_vol5: NDArray[np.float64],
    ask_vol6: NDArray[np.float64],
    ask_vol7: NDArray[np.float64],
    ask_vol8: NDArray[np.float64],
    ask_vol9: NDArray[np.float64],
    ask_vol10: NDArray[np.float64],
    k1_horizontal: float = 2.0,
    k2_vertical: float = 5.0,
    window_size: int = 100,
    decay_threshold: float = 0.5,
) -> Tuple[List[NDArray[np.float64]], List[List[str]]]:
    """计算ALA事件微观结构特征 v3版本（性能优化版本）。
    
    v3版本改进：
    - 预计算滑动窗口平均值（O(1)查询）
    - 二分查找范围查询
    - 减少HashMap操作开销
    - 减少内存分配
    
    返回9种(detection_mode, side_filter)组合的特征数组。
    """
    ...
"""交易分析函数类型声明"""

from typing import List, Optional, Tuple
import numpy as np
from numpy.typing import NDArray

def compute_agent_trading_features(
    market_timestamps: NDArray[np.int64],
    market_prices: NDArray[np.float64],
    market_volumes: NDArray[np.float64],
    market_turnovers: NDArray[np.float64],
    market_flags: NDArray[np.int32],
    bid_order_ids: NDArray[np.int64],
    ask_order_ids: NDArray[np.int64],
    agent_market_indices: NDArray[np.uint64],
    agent_directions: NDArray[np.int32],
    agent_volumes: NDArray[np.float64],
    window_ms: int = 10000,
    grid_ms: int = 1000,
) -> Tuple[
    NDArray[np.float64], List[str],
    NDArray[np.float64], List[str],
    NDArray[np.float64], List[str]
]:
    """计算单Agent特征，返回事件级/时间栅格/参数轴三组结果。

    返回顺序：
    1) event_features, event_feature_names
    2) time_grid_features, time_grid_feature_names
    3) param_axis_features, param_axis_feature_names（单Agent版本为空矩阵）
    """
    ...

def compute_agent_trading_features_multi(
    market_timestamps: NDArray[np.int64],
    market_prices: NDArray[np.float64],
    market_volumes: NDArray[np.float64],
    market_turnovers: NDArray[np.float64],
    market_flags: NDArray[np.int32],
    bid_order_ids: NDArray[np.int64],
    ask_order_ids: NDArray[np.int64],
    all_agent_market_indices: List[NDArray[np.uint64]],
    all_agent_directions: List[NDArray[np.int32]],
    all_agent_volumes: List[NDArray[np.float64]],
    target_agent_idx: int,
    window_ms: int = 10000,
    grid_ms: int = 1000,
    param_window_ms: int = 2000,
    all_agent_params: Optional[List[float]] = None,
) -> Tuple[
    NDArray[np.float64], List[str],
    NDArray[np.float64], List[str],
    NDArray[np.float64], List[str]
]:
    """计算多Agent特征，返回事件级/时间栅格/参数轴三组结果。"""
    ...

def get_agent_feature_names() -> List[str]:
    """获取事件级特征列名（Base + 快速扩展）。"""
    ...

def get_agent_time_grid_feature_names() -> List[str]:
    """获取时间栅格特征列名。"""
    ...

def get_agent_param_axis_feature_names() -> List[str]:
    """获取参数轴特征列名。"""
    ...

def find_follow_volume_sum_same_price(
    times: NDArray[np.float64],
    prices: NDArray[np.float64],
    volumes: NDArray[np.float64],
    time_window: float = 0.1,
    check_price: bool = True,
    filter_ratio: float = 0.0,
    timeout_seconds: Optional[float] = None,
) -> NDArray[np.float64]:
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

[... 保留原有的函数声明 ...]
