"""主题聚类因子计算类型声明"""
from typing import List, Optional, Tuple
import numpy as np
from numpy.typing import NDArray


def theme_cluster_factors_from_minute(
    minute_data: NDArray[np.float64],
    segments: List[str],
    segment_bounds: List[Tuple[int, int]],
    k_values: List[int],
    align_methods: List[str],
    distance_metrics: List[str],
    distance_thresholds: List[float],
    lookback_values: List[int],
    n_threads: int = 8,
) -> Tuple[NDArray[np.float64], List[str]]:
    """
    从分钟数据计算主题聚类因子（Rust内部做时段拆分和特征提取）

    参数:
        minute_data: 分钟原始数据, shape=(n_days, 238, n_stocks, 20)
            字段顺序: open, close, amount, volume,
                      act_buy_amount_sum, act_sell_amount_sum,
                      act_buy_count_sum, act_sell_count_sum,
                      act_buy_vol_sum, act_sell_vol_sum,
                      up_tick_count, down_tick_count,
                      bid_size_1_mean, ask_size_1_mean,
                      bid_size_6_mean, ask_size_6_mean,
                      bid_vwap_mean, ask_vwap_mean,
                      bid_vol1, ask_vol1
        segments: 时段名称列表
        segment_bounds: 时段边界列表 [(start, end), ...]
        k_values: 聚类数列表
        align_methods: 对齐方法列表
        distance_metrics: 距离度量列表
        distance_thresholds: 距离阈值列表
        lookback_values: 回溯天数列表
        n_threads: 并行线程数

    返回:
        (因子矩阵, 列名列表)
    """
    ...


def theme_cluster_factors_batch(
    features_list: List[NDArray[np.float64]],
    micro_metrics_list: List[NDArray[np.float64]],
    k_values: List[int],
    align_methods: List[str],
    distance_metrics: List[str],
    distance_thresholds: List[float],
    lookback_values: List[int],
    n_threads: int = 8,
) -> Tuple[NDArray[np.float64], List[str]]:
    """
    批量计算主题聚类因子 (优化版本)

    参数:
        features_list: 多日特征矩阵列表 (按日期升序), 每个shape=(n_stocks, 12)
        micro_metrics_list: 多日微观指标矩阵列表 (按日期升序), 每个shape=(n_stocks, 8)
        k_values: 聚类数列表
        align_methods: 对齐方法列表 ("hungarian" / "overlap")
        distance_metrics: 距离度量列表 ("euclidean" / "cosine" / "mahalanobis")
        distance_thresholds: 距离阈值列表 (None用负数表示)
        lookback_values: 回溯天数列表
        n_threads: 并行线程数 (默认8, 最大10)

    返回:
        (因子矩阵, 列名列表)
        - 因子矩阵: shape=(n_stocks, n_factors * n_param_combos)
        - 列名列表: 每列的名称，格式为 {factor_name}__k{k}_{align}_{dist}_{thresh}_{lookback}
    """
    ...


def theme_cluster_factors_batch_multi_segments(
    segments_data: List[Tuple[str, List[NDArray[np.float64]], List[NDArray[np.float64]]]],
    k_values: List[int],
    align_methods: List[str],
    distance_metrics: List[str],
    distance_thresholds: List[float],
    lookback_values: List[int],
    n_threads: int = 8,
) -> Tuple[NDArray[np.float64], List[str]]:
    """
    批量计算主题聚类因子 (多时段并行版本)

    参数:
        segments_data: List[(segment_name, features_list, micro_list)]
            - segment_name: 时段名称字符串
            - features_list: List[ndarray], 每个shape=(n_stocks, 12)
            - micro_list: List[ndarray], 每个shape=(n_stocks, 8)
        k_values: 聚类数列表
        align_methods: 对齐方法列表
        distance_metrics: 距离度量列表
        distance_thresholds: 距离阈值列表 (None用负数表示)
        lookback_values: 回溯天数列表
        n_threads: 并行线程数 (默认8, 最大10)

    返回:
        (因子矩阵, 列名列表)
        - 因子矩阵: shape=(n_stocks, n_segments * n_combos * n_factors)
        - 列名列表: 格式为 {factor_name}__k{k}_{align}_{dist}_{thresh}_{lookback}__{segment}
    """
    ...


def clear_theme_cluster_cache() -> None:
    """清除聚类缓存"""
    ...


def get_theme_cluster_factor_names(
    segments: List[str],
    k_values: List[int],
    align_methods: List[str],
    distance_metrics: List[str],
    distance_thresholds: List[float],
    lookback_values: List[int],
) -> List[str]:
    """
    获取主题聚类因子的列名（不计算因子，仅生成列名）
    
    参数:
        segments: 时段列表 (如 ['all_day', 'open30'])
        k_values: 聚类数列表 (如 [20, 30])
        align_methods: 对齐方法列表 (如 ['hungarian', 'greedy'])
        distance_metrics: 距离度量列表 (如 ['euclidean', 'correlation'])
        distance_thresholds: 距离阈值列表 (如 [-1.0, 0.5], -1.0表示不使用阈值)
        lookback_values: 回溯天数列表 (如 [5, 10])
    
    返回:
        列名列表
    """
    ...


def theme_cluster_factors(
    features_list: List[NDArray[np.float64]],
    micro_metrics_list: List[NDArray[np.float64]],
    k: int,
    align_method: str,
    distance_metric: str,
    distance_threshold: float,
    lookback: int,
) -> NDArray[np.float64]:
    """
    计算所有主题聚类因子

    参数:
        features_list: 多日特征矩阵列表 (按日期升序), 每个shape=(n_stocks, 12)
            列顺序: daily_return, intraday_vol, log_total_amt, act_buy_ratio,
                    log_avg_trade_amt, min_ret_skew, up_tick_ratio,
                    bid_size1_mean, ask_size1_mean, vwap_diff,
                    last_bid_vol1, last_ask_vol1
        micro_metrics_list: 多日微观指标矩阵列表 (按日期升序), 每个shape=(n_stocks, 8)
            列顺序: act_buy_ratio, bid_ask_imbalance, vol_per_trade,
                    spread_ratio, depth_imbalance, vwap_deviation,
                    act_buy_vol_ratio, big_order_ratio
        k: 聚类数量
        align_method: 对齐方法 ("hungarian" / "overlap")
        distance_metric: 距离度量 ("euclidean" / "cosine")
        distance_threshold: 距离阈值 (<=0 表示不使用)
        lookback: 回溯天数
    
    返回:
        shape=(n_stocks, n_factors) 的因子矩阵
        因子顺序: F4_theme_return_mean, F5_yesterday_theme_return, F6_theme_return_trend,
                  F7_theme_size, F8_theme_amount, F9_theme_return, F10_theme_act_buy_ratio,
                  F11_return_rank_in_theme, F12_distance_to_center,
                  F13_switch_direction, F14_switch_strength,
                  F21_theme_entropy,
                  F23_theme_heat, F24_theme_momentum_persistence,
                  F25_theme_diffusion, F26_leader_stability,
                  [F17_x_deviation, F18_x_deviation_change] * n_micro_metrics
    """
    ...
