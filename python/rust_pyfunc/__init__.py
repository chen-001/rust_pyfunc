from importlib import import_module

from rust_pyfunc.rust_pyfunc import *
from .rolling_future import RollingFutureAccessor
from .rolling_past import RollingPastAccessor
from .pandas_corrwith import corrwith
from .pandas_rank import rank_axis1_df, rank_axis0_df, fast_rank, fast_rank_axis1, fast_rank_axis0
from .pandas_merge import fast_merge_df, fast_inner_join_df, fast_left_join_df, fast_right_join_df, fast_outer_join_df, fast_join, fast_merge_dataframe
from .pandas_correlation import fast_correlation_matrix_v2_df, fast_corr_df, correlation_matrix_df
from .agent_trading_daily import (
    aggregate_agent_feature_triplets_to_daily_table,
    get_agent_daily_feature_column_groups_for_get_features_factors,
    pack_agent_daily_feature_table_for_get_features_factors,
    compute_agent_trading_daily_feature_table,
    compute_agent_trading_daily_feature_table_multi,
    compute_agent_trading_daily_feature_table_for_get_features_factors,
    compute_agent_trading_daily_feature_table_multi_for_get_features_factors,
)
from .theme_feature_expansion_data import (
    THEME_FEATURE_EXPANSION_MINUTE_FIELDS,
    THEME_FEATURE_VECTOR_DIMENSIONS,
    prepare_minute_data_for_theme_feature_expansion,
    compute_theme_feature_expansion_from_minute_raw,
    compute_theme_feature_expansion_from_date,
)
from .trading_data_utils import (
    adjust_afternoon,
    read_trade,
    read_market,
    read_market_pair,
    get_features_factors_single,
    get_features_factors,
    get_features_names,
)
from .abm_analysis import (
    abm_step_two,
    abm_generate_names,
)
from .trading_day import (
    TradingDay,
    td,
    last_trading_day_tricky,
    next_trading_day_tricky,
    last_n_trading_date,
)
from rust_pyfunc import *

_LAZY_TREEVISUAL_EXPORTS = {"PriceTreeViz", "haha", "treevisual"}


def __getattr__(name):
    if name in _LAZY_TREEVISUAL_EXPORTS:
        module = import_module(".treevisual", __name__)
        value = module if name == "treevisual" else getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(
        set(globals())
        | _LAZY_THEME_CLUSTER_EXPORTS
        | _LAZY_TREEVISUAL_EXPORTS
    )
