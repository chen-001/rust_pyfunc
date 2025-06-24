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
from .tree_structures import *
from .trading_analysis import *
from .statistical_analysis import *

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
    
    # 文本分析函数
    "vectorize_sentences",
    "jaccard_similarity",
    "min_word_edit_distance",
    "vectorize_sentences_list",
    
    # 并行计算函数
    "run_pools",
    "query_backup", 
    "delete_backup",
    "backup_exists",
    "get_backup_info",
    
    # Pandas扩展函数
    "dataframe_corrwith",
    "rank_axis1",
    "fast_merge",
    "fast_merge_mixed",
    "fast_inner_join_dataframes",
    
    # 树结构类
    "PriceTree",
    "RollingFutureAccessor",
    
    # 交易分析函数
    "find_follow_volume_sum_same_price",
    "find_follow_volume_sum_same_price_and_flag",
    "mark_follow_groups",
    "mark_follow_groups_with_flag",
    "analyze_retreat_advance",
    "analyze_retreat_advance_v2",
    "calculate_large_order_nearby_small_order_time_gap",
    
    # 统计分析函数
    "calculate_base_entropy",
    "calculate_shannon_entropy_change",
    "calculate_shannon_entropy_change_at_low",
    "calculate_window_entropy",
    "factor_correlation_by_date",
    "factor_grouping",
    "segment_and_correlate",
]