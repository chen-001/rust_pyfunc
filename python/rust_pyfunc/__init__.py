from importlib import import_module

from rust_pyfunc.rust_pyfunc import *
from rust_pyfunc import *
from rust_pyfunc.rolling_future import RollingFutureAccessor
from rust_pyfunc.rolling_past import RollingPastAccessor

_LAZY_EXPORTS = {
    "corrwith": (".pandas_corrwith", "corrwith"),
    "rank_axis1_df": (".pandas_rank", "rank_axis1_df"),
    "rank_axis0_df": (".pandas_rank", "rank_axis0_df"),
    "fast_rank": (".pandas_rank", "fast_rank"),
    "fast_rank_axis1": (".pandas_rank", "fast_rank_axis1"),
    "fast_rank_axis0": (".pandas_rank", "fast_rank_axis0"),
    "fast_merge_df": (".pandas_merge", "fast_merge_df"),
    "fast_inner_join_df": (".pandas_merge", "fast_inner_join_df"),
    "fast_left_join_df": (".pandas_merge", "fast_left_join_df"),
    "fast_right_join_df": (".pandas_merge", "fast_right_join_df"),
    "fast_outer_join_df": (".pandas_merge", "fast_outer_join_df"),
    "fast_join": (".pandas_merge", "fast_join"),
    "fast_merge_dataframe": (".pandas_merge", "fast_merge_dataframe"),
    "fast_correlation_matrix_v2_df": (".pandas_correlation", "fast_correlation_matrix_v2_df"),
    "fast_corr_df": (".pandas_correlation", "fast_corr_df"),
    "correlation_matrix_df": (".pandas_correlation", "correlation_matrix_df"),
    "aggregate_agent_feature_triplets_to_daily_table": (".agent_trading_daily", "aggregate_agent_feature_triplets_to_daily_table"),
    "get_agent_daily_feature_column_groups_for_get_features_factors": (".agent_trading_daily", "get_agent_daily_feature_column_groups_for_get_features_factors"),
    "pack_agent_daily_feature_table_for_get_features_factors": (".agent_trading_daily", "pack_agent_daily_feature_table_for_get_features_factors"),
    "compute_agent_trading_daily_feature_table": (".agent_trading_daily", "compute_agent_trading_daily_feature_table"),
    "compute_agent_trading_daily_feature_table_multi": (".agent_trading_daily", "compute_agent_trading_daily_feature_table_multi"),
    "compute_agent_trading_daily_feature_table_for_get_features_factors": (".agent_trading_daily", "compute_agent_trading_daily_feature_table_for_get_features_factors"),
    "compute_agent_trading_daily_feature_table_multi_for_get_features_factors": (".agent_trading_daily", "compute_agent_trading_daily_feature_table_multi_for_get_features_factors"),
    "THEME_FEATURE_EXPANSION_MINUTE_FIELDS": (".theme_feature_expansion_data", "THEME_FEATURE_EXPANSION_MINUTE_FIELDS"),
    "THEME_FEATURE_VECTOR_DIMENSIONS": (".theme_feature_expansion_data", "THEME_FEATURE_VECTOR_DIMENSIONS"),
    "prepare_minute_data_for_theme_feature_expansion": (".theme_feature_expansion_data", "prepare_minute_data_for_theme_feature_expansion"),
    "compute_theme_feature_expansion_from_minute_raw": (".theme_feature_expansion_data", "compute_theme_feature_expansion_from_minute_raw"),
    "compute_theme_feature_expansion_from_date": (".theme_feature_expansion_data", "compute_theme_feature_expansion_from_date"),
    "adjust_afternoon": (".trading_data_utils", "adjust_afternoon"),
    "read_trade": (".trading_data_utils", "read_trade"),
    "read_market": (".trading_data_utils", "read_market"),
    "read_market_pair": (".trading_data_utils", "read_market_pair"),
    "get_features_factors_single": (".trading_data_utils", "get_features_factors_single"),
    "get_features_factors": (".trading_data_utils", "get_features_factors"),
    "get_features_names": (".trading_data_utils", "get_features_names"),
    "abm_step_two": (".abm_analysis", "abm_step_two"),
    "abm_generate_names": (".abm_analysis", "abm_generate_names"),
    "TradingDay": (".trading_day", "TradingDay"),
    "last_trading_day_tricky": (".trading_day", "last_trading_day_tricky"),
    "next_trading_day_tricky": (".trading_day", "next_trading_day_tricky"),
    "last_n_trading_date": (".trading_day", "last_n_trading_date"),
    "PriceTreeViz": (".treevisual", "PriceTreeViz"),
    "haha": (".treevisual", "haha"),
    "treevisual": (".treevisual", None),
}

_LAZY_MODULE_EXPORTS = {
    "td": (".trading_day", "td"),
}


def __getattr__(name):
    if name in _LAZY_EXPORTS:
        module_path, attr = _LAZY_EXPORTS[name]
        module = import_module(module_path, __name__)
        value = module if attr is None else getattr(module, attr)
        globals()[name] = value
        return value
    if name in _LAZY_MODULE_EXPORTS:
        module_path, attr = _LAZY_MODULE_EXPORTS[name]
        module = import_module(module_path, __name__)
        value = getattr(module, attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_LAZY_EXPORTS) | set(_LAZY_MODULE_EXPORTS))
