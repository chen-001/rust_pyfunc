"""Agent交易模拟器类型声明

该模块提供基于Trait架构的Agent交易行为模拟器，支持多种类型的Agent。
"""

from typing import List, Dict, Any
import numpy as np
from numpy.typing import NDArray


def simulate_momentum_agents_py(
    timestamps: NDArray[np.int64],
    prices: NDArray[np.float64],
    volumes: NDArray[np.float64],
    flags: NDArray[np.int32],
    agent_names: List[str],
    lookback_ms_list: List[int],
    thresholds: List[float],
    is_percentages: List[bool],
    fixed_trade_sizes: List[int],
    cooldown_ms_list: List[int],
    allow_shorts: List[bool],
) -> List[Dict[str, Any]]:
    """模拟动量型Agent的交易行为
    
    动量型Agent根据回看窗口内的价格变化触发交易。
    
    触发规则：
    - 买入：价格变化 > 阈值
    - 卖出：价格变化 < -阈值（当allow_short=True时）
    
    参数说明：
    ----------
    timestamps : NDArray[np.int64]
        市场成交时间戳数组（毫秒）
    prices : NDArray[np.float64]
        市场成交价格数组
    volumes : NDArray[np.float64]
        市场成交量数组
    flags : NDArray[np.int32]
        市场成交方向，66=主买，83=主卖
    agent_names : List[str]
        Agent名称列表
    lookback_ms_list : List[int]
        每个Agent的回看时间窗口（毫秒）
    thresholds : List[float]
        每个Agent的价格变化阈值
    is_percentages : List[bool]
        是否使用百分比变化（True=百分比，False=绝对价格）
    fixed_trade_sizes : List[int]
        每个Agent的固定交易数量（股）
    cooldown_ms_list : List[int]
        每个Agent的冷却期（毫秒）
    allow_shorts : List[bool]
        每个Agent是否允许做空
    
    返回值：
    -------
    List[Dict[str, Any]]
        每个Agent的模拟结果列表，每个字典包含：
        - 'name': Agent名称
        - 'n_trades': 交易次数
        - 'total_buy_volume': 总买入量
        - 'total_sell_volume': 总卖出量
        - 'final_position': 最终持仓
        - 'market_indices': NDArray[np.uint64]，对应市场数据的索引
        - 'directions': NDArray[np.int32]，交易方向（66=买入，83=卖出）
        - 'volumes': NDArray[np.float64]，交易量
        - 'prices': NDArray[np.float64]，交易价格
    
    示例：
    -------
    >>> import rust_pyfunc as rp
    >>> import numpy as np
    >>> 
    >>> # 构造市场数据
    >>> n = 1000
    >>> timestamps = np.arange(n) * 1000  # 毫秒
    >>> prices = 10.0 + np.cumsum(np.random.randn(n) * 0.01)
    >>> volumes = np.random.uniform(100, 1000, n)
    >>> flags = np.random.choice([66, 83], n).astype(np.int32)
    >>> 
    >>> # 配置6个不同回看窗口的动量Agent
    >>> results = rp.simulate_momentum_agents_py(
    ...     timestamps, prices, volumes, flags,
    ...     agent_names=["3s", "1m", "5m", "10m", "30m", "60m"],
    ...     lookback_ms_list=[3000, 60000, 300000, 600000, 1800000, 3600000],
    ...     thresholds=[0.02, 0.05, 0.10, 0.15, 0.20, 0.30],
    ...     is_percentages=[False] * 6,
    ...     fixed_trade_sizes=[100] * 6,
    ...     cooldown_ms_list=[5000] * 6,
    ...     allow_shorts=[True] * 6
    ... )
    >>> 
    >>> # 查看结果
    >>> for r in results:
    ...     print(f"{r['name']}: {r['n_trades']} trades")
    >>> 
    >>> # 将结果传递给特征计算函数
    >>> features, column_names = rp.compute_agent_trading_features(
    ...     timestamps, prices, volumes, ...,  # 市场数据
    ...     results[0]['market_indices'].astype(np.uint64),
    ...     results[0]['directions'].astype(np.int32),
    ...     results[0]['volumes']
    ... )
    """
    ...


def simulate_buy_ratio_agents_py(
    timestamps: NDArray[np.int64],
    prices: NDArray[np.float64],
    volumes: NDArray[np.float64],
    flags: NDArray[np.int32],
    agent_names: List[str],
    lookback_ms_list: List[int],
    thresholds: List[float],
    fixed_trade_sizes: List[int],
    cooldown_ms_list: List[int],
    allow_shorts: List[bool],
) -> List[Dict[str, Any]]:
    """模拟主买占比型Agent的交易行为
    
    主买占比型Agent根据回看窗口内的主动买入成交量占比触发交易。
    
    触发规则：
    - 买入：主买占比 > 阈值（跟随主力资金）
    - 卖出：主买占比 < (1-阈值)（即主卖占比高）
    
    参数说明：
    ----------
    timestamps : NDArray[np.int64]
        市场成交时间戳数组（毫秒）
    prices : NDArray[np.float64]
        市场成交价格数组
    volumes : NDArray[np.float64]
        市场成交量数组
    flags : NDArray[np.int32]
        市场成交方向，66=主买，83=主卖
    agent_names : List[str]
        Agent名称列表
    lookback_ms_list : List[int]
        每个Agent的回看时间窗口（毫秒）
    thresholds : List[float]
        每个Agent的主买占比阈值（如0.67表示大于2/3）
    fixed_trade_sizes : List[int]
        每个Agent的固定交易数量（股）
    cooldown_ms_list : List[int]
        每个Agent的冷却期（毫秒）
    allow_shorts : List[bool]
        每个Agent是否允许做空
    
    返回值：
    -------
    List[Dict[str, Any]]
        每个Agent的模拟结果列表，格式与simulate_momentum_agents_py相同
    
    示例：
    -------
    >>> import rust_pyfunc as rp
    >>> import numpy as np
    >>> 
    >>> # 构造市场数据
    >>> timestamps = np.arange(300) * 1000
    >>> prices = 10.0 + np.cumsum(np.random.randn(300) * 0.001)
    >>> volumes = np.ones(300) * 100
    >>> 
    >>> # 构造不同主买占比的时段
    >>> flags = np.concatenate([
    ...     np.ones(150, dtype=np.int32) * 66,  # 前150条主买
    ...     np.ones(150, dtype=np.int32) * 83   # 后150条主卖
    ... ])
    >>> 
    >>> # 配置主买占比Agent
    >>> results = rp.simulate_buy_ratio_agents_py(
    ...     timestamps, prices, volumes, flags,
    ...     agent_names=["br_3s", "br_10s", "br_30s"],
    ...     lookback_ms_list=[3000, 10000, 30000],
    ...     thresholds=[0.60, 0.65, 0.70],
    ...     fixed_trade_sizes=[100] * 3,
    ...     cooldown_ms_list=[2000] * 3,
    ...     allow_shorts=[True] * 3
    ... )
    """
    ...


def simulate_thematic_agents_py(
    timestamps: NDArray[np.int64],
    prices: NDArray[np.float64],
    volumes: NDArray[np.float64],
    flags: NDArray[np.int32],
    agent_names: List[str],
    agent_types: List[str],
    short_window_ms_list: List[int],
    trend_window_ms_list: List[int],
    amount_thresholds: List[float],
    acceleration_factors: List[float],
    decay_factors: List[float],
    cooldown_ms_list: List[int],
) -> List[Dict[str, Any]]:
    """统一主题Agent入口：支持在一个列表里混合多种agent_type批量模拟"""
    ...


def simulate_bottom_fishing_agents_py(
    timestamps: NDArray[np.int64],
    prices: NDArray[np.float64],
    volumes: NDArray[np.float64],
    flags: NDArray[np.int32],
    agent_names: List[str],
    short_window_ms_list: List[int],
    trend_window_ms_list: List[int],
    cooldown_ms_list: List[int],
) -> List[Dict[str, Any]]:
    """模拟抄底型Agent交易行为（固定每次100股，允许卖空）"""
    ...


def simulate_follow_flow_agents_py(
    timestamps: NDArray[np.int64],
    prices: NDArray[np.float64],
    volumes: NDArray[np.float64],
    flags: NDArray[np.int32],
    agent_names: List[str],
    short_window_ms_list: List[int],
    trend_window_ms_list: List[int],
    amount_thresholds: List[float],
    cooldown_ms_list: List[int],
) -> List[Dict[str, Any]]:
    """模拟跟随型Agent交易行为（固定每次100股，允许卖空）"""
    ...


def simulate_acceleration_follow_agents_py(
    timestamps: NDArray[np.int64],
    prices: NDArray[np.float64],
    volumes: NDArray[np.float64],
    flags: NDArray[np.int32],
    agent_names: List[str],
    short_window_ms_list: List[int],
    trend_window_ms_list: List[int],
    amount_thresholds: List[float],
    acceleration_factors: List[float],
    cooldown_ms_list: List[int],
) -> List[Dict[str, Any]]:
    """模拟加速跟随型Agent交易行为（固定每次100股，允许卖空）"""
    ...


def simulate_exhaustion_reversal_agents_py(
    timestamps: NDArray[np.int64],
    prices: NDArray[np.float64],
    volumes: NDArray[np.float64],
    flags: NDArray[np.int32],
    agent_names: List[str],
    short_window_ms_list: List[int],
    trend_window_ms_list: List[int],
    amount_thresholds: List[float],
    decay_factors: List[float],
    cooldown_ms_list: List[int],
) -> List[Dict[str, Any]]:
    """模拟衰竭抄底型Agent交易行为（固定每次100股，允许卖空）"""
    ...
