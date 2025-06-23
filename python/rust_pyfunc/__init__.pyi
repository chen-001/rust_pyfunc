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
    
    # 时间序列函数
    "dtw_distance",
    "fast_dtw_distance",
    "super_dtw_distance", 
    "transfer_entropy",
    "rolling_dtw_distance",
    "find_local_peaks_within_window",
    "find_half_energy_time",
    "rolling_window_stat",
    
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
]