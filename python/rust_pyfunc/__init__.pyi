"""
rust_pyfunc - 高性能Python函数库
=================================

基于Rust实现的高性能Python函数库，提供数学计算、时间序列分析、
文本处理、并行计算等功能。

主要模块：
- core_functions: 核心数学和统计函数
- time_series: 时间序列分析函数  
- text_analysis: 文本处理函数
- parallel_computing: 并行计算和备份管理
- pandas_extensions: Pandas扩展函数
- tree_structures: 树结构相关类
"""

# 导入所有类型声明
from .core_functions import *
from .time_series import *
from .text_analysis import *
from .parallel_computing import *
from .pandas_extensions import *
from .pandas_correlation import *
from .tree_structures import *
from .trading_analysis import *
from .statistical_analysis import *
from .web_manager import *

# 版本信息
__version__ = "0.18.0"
__author__ = "chenzongwei"
__email__ = "noreply@example.com"

# 所有公开的函数和类
__all__ = [
    # 核心函数
    "trend",
    "trend_fast", 
    "identify_segments",
    "find_max_range_product",
    "ols",
    "ols_predict",
    "ols_residuals",
    "max_range_loop",
    "min_range_loop",
    "rolling_volatility",
    "rolling_cv", 
    "rolling_qcv",
    "compute_max_eigenvalue",
    "sum_as_string",
    "test_simple_function",
    "test_function",
    
    # 时间序列函数
    "dtw_distance",
    "fast_dtw_distance",
    "super_dtw_distance", 
    "transfer_entropy",
    "rolling_dtw_distance",
    "find_local_peaks_within_window",
    "find_half_energy_time",
    "rolling_window_stat",
    "find_half_extreme_time",
    "fast_find_half_extreme_time",
    "super_find_half_extreme_time",
    "brachistochrone_curve",
    "brachistochrone_curve_v2",
    "rolling_lagged_regression",
    "rolling_lagged_regression_ridge_fast",
    "rolling_lagged_regression_ridge_simd",
    "rolling_lagged_regression_ridge_incremental",
    
    # 文本分析函数
    "vectorize_sentences",
    "jaccard_similarity",
    "min_word_edit_distance",
    "vectorize_sentences_list",
    
    # 并行计算函数
    "run_pools_queue",
    "query_backup",
    "query_backup_fast",
    
    # Pandas扩展函数
    "dataframe_corrwith",
    "dataframe_corrwith_single_thread",
    "rank_axis1",
    "fast_merge",
    "fast_merge_mixed",
    "fast_inner_join_dataframes",
    
    # 树结构类
    "PriceTree",
    "RollingFutureAccessor",
    "PriceTreeViz",
    
    # 交易分析函数
    "find_follow_volume_sum_same_price",
    "find_follow_volume_sum_same_price_and_flag",
    "mark_follow_groups",
    "mark_follow_groups_with_flag",
    "analyze_retreat_advance",
    "analyze_retreat_advance_v2",
    "calculate_large_order_nearby_small_order_time_gap",
    "order_contamination",
    "order_contamination_parallel",
    "order_contamination_bilateral",
    "trade_peak_analysis",
    "order_neighborhood_analysis",
    "calculate_trade_time_gap_and_price_percentile_ultra_sorted",
    "calculate_order_time_gap_and_price_percentile_ultra_sorted",
    "analyze_asks",
    "compute_non_breakthrough_stats",
    "compute_price_cycle_features",
    "compute_price_cycle_features_b_segments",
    "compute_price_cycle_features_b_segments_enhanced",
    
    # 统计分析函数
    "calculate_base_entropy",
    "calculate_shannon_entropy_change",
    "calculate_shannon_entropy_change_at_low",
    "calculate_window_entropy",
    "factor_correlation_by_date",
    "factor_grouping",
    "segment_and_correlate",
    
    # Python定义的pandas扩展函数
    "corrwith",
    "rank_axis1_df",
    "rank_axis0_df",
    "fast_rank",
    "fast_rank_axis1", 
    "fast_rank_axis0",
    "fast_merge_df",
    "fast_inner_join_df",
    "fast_left_join_df",
    "fast_right_join_df",
    "fast_outer_join_df",
    "fast_join",
    "fast_merge_dataframe",
    
    # pandas相关性矩阵函数
    "fast_correlation_matrix_v2_df",
    "fast_corr_df",
    "correlation_matrix_df",
    
    # 因子中性化函数
    "batch_factor_neutralization",
    "batch_factor_neutralization_optimized",
    "batch_factor_neutralization_io_optimized", 
    "batch_factor_neutralization_simple_math_optimized",
    "batch_factor_neutralization_parallel_optimized",
    "batch_factor_neutralization_ultimate_optimized",
    "batch_factor_neutralization_simple_fallback",
    
    # Web管理器
    "BackupWebManager",
    "check_port_available",
    "find_available_port", 
    "start_web_manager",
    
    # 测试函数
    "haha",
    
]