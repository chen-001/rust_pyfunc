"""主题聚类因子数据准备模块

提供 prepare_features_list 和 prepare_micro_metrics_list 两个函数,
将分钟数据转换为 theme_cluster_factors 所需的 np.ndarray 列表。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict

import design_whatever as dw

# 时段定义（基于240分钟数据）
TIME_SEGMENTS = {
    'all_day': (0, 240),
    'open30': (0, 30),
    'morning': (0, 120),
    'afternoon': (120, 240),
    'close30': (210, 240),
}

# 特征提取所需的分钟字段
_FEATURE_FIELDS = [
    'open', 'close', 'amount', 'volume',
    'act_buy_amount_sum', 'act_sell_amount_sum',
    'act_buy_count_sum', 'act_sell_count_sum',
    'up_tick_count', 'down_tick_count',
    'bid_size_1_mean', 'ask_size_1_mean',
    'bid_vwap_mean', 'ask_vwap_mean',
    'bid_vol1', 'ask_vol1',
]

# 微观指标额外需要的字段
_MICRO_EXTRA_FIELDS = [
    'act_buy_vol_sum', 'act_sell_vol_sum',
    'bid_size_6_mean', 'ask_size_6_mean',
]

# 微观指标列名顺序(与Rust端约定一致)
MICRO_METRIC_NAMES = [
    'act_buy_ratio', 'bid_ask_imbalance', 'vol_per_trade',
    'spread_ratio', 'depth_imbalance', 'vwap_deviation',
    'act_buy_vol_ratio', 'big_order_ratio',
]

# 特征列名顺序(与Rust端约定一致)
FEATURE_NAMES = [
    'daily_return', 'intraday_vol', 'log_total_amt', 'act_buy_ratio',
    'log_avg_trade_amt', 'min_ret_skew', 'up_tick_ratio',
    'bid_size1_mean', 'ask_size1_mean', 'vwap_diff',
    'last_bid_vol1', 'last_ask_vol1',
]


def _get_trading_dates(start_date: int, end_date: int) -> List[int]:
    """获取交易日列表"""
    td = dw.td
    start_pos = td.get_loc(start_date)
    end_pos = td.get_loc(end_date)
    return list(td.trading_days[start_pos:end_pos + 1])


def _last_n_trading_date(end_date: int, n_date: int) -> int:
    """获取end_date之前n_date天的那个交易日"""
    td = dw.td
    pos = td.get_loc(end_date) - n_date
    return td.trading_days[pos]


def _read_minute_fields(date: int, fields: List[str]) -> dict:
    """读取单日分钟字段"""
    return {f: dw.read_minute_data(f, date, date) for f in fields}


def _get_time_slice(data: dict, segment: str) -> dict:
    """获取指定时段的数据切片"""
    start, end = TIME_SEGMENTS[segment]
    return {k: v.iloc[start:end] for k, v in data.items()}


def _extract_daily_features_array(data: dict) -> Tuple[np.ndarray, pd.Index]:
    """
    从时段切片后的分钟数据提取12列特征, 返回 (n_stocks, 12) 的ndarray 和股票索引
    """
    open_ = data['open']
    close = data['close']
    amount = data['amount']
    volume = data['volume']
    act_buy_amt = data['act_buy_amount_sum']
    act_buy_cnt = data['act_buy_count_sum']
    act_sell_cnt = data['act_sell_count_sum']
    up_tick = data['up_tick_count']
    down_tick = data['down_tick_count']
    bid_size1 = data['bid_size_1_mean']
    ask_size1 = data['ask_size_1_mean']
    bid_vwap = data['bid_vwap_mean']
    ask_vwap = data['ask_vwap_mean']
    bid_vol1 = data['bid_vol1']
    ask_vol1 = data['ask_vol1']

    symbols = close.columns
    total_amt = amount.sum()
    
    # 找到最后一个有效分钟的行号
    valid_mask = close.notna().any(axis=1)
    last_valid_row = valid_mask[valid_mask].index[-1] if valid_mask.any() else len(close) - 1
    last_valid_iloc = close.index.get_loc(last_valid_row) if isinstance(last_valid_row, int) or hasattr(close.index, 'get_loc') else -1
    # 直接用iloc取最后一个有效行
    last_valid_iloc = valid_mask[valid_mask].shape[0] - 1 if valid_mask.any() else len(close) - 1
    # 简化：用位置索引找到最后一个非全NaN行
    last_valid_pos = valid_mask.values[::-1].argmax()  # 从后往前找第一个True
    last_valid_pos = len(close) - 1 - last_valid_pos  # 转回正向位置

    # 0: daily_return - 用第一个和最后一个有效分钟
    first_open = open_.iloc[0].values
    last_close = close.iloc[last_valid_pos].values
    daily_ret = np.where(first_open > 0, last_close / first_open - 1, np.nan)
    
    # 1: intraday_vol
    min_rets = np.log(close / close.shift(1))
    intraday_vol = min_rets.std().values
    # 2: log_total_amt
    ta = total_amt.values.copy().astype(float)
    ta[ta == 0] = np.nan
    log_total_amt = np.log(ta)
    # 3: act_buy_ratio
    act_buy_ratio = (act_buy_amt.sum() / (total_amt + 1e-10)).values
    # 4: log_avg_trade_amt
    total_cnt = (act_buy_cnt.sum() + act_sell_cnt.sum()).values.astype(float)
    total_cnt[total_cnt == 0] = np.nan
    log_avg_trade_amt = np.log(total_amt.values / total_cnt)
    # 5: min_ret_skew
    min_ret_skew = min_rets.skew().values
    # 6: up_tick_ratio
    total_tick = (up_tick.sum() + down_tick.sum()).values.astype(float)
    total_tick[total_tick == 0] = np.nan
    up_tick_ratio = up_tick.sum().values / total_tick
    up_tick_ratio = np.where(np.isnan(up_tick_ratio), 0.5, up_tick_ratio)
    # 7: bid_size1_mean
    bid_size1_mean = bid_size1.mean().values
    # 8: ask_size1_mean
    ask_size1_mean = ask_size1.mean().values
    # 9: vwap_diff
    vwap_diff = (bid_vwap.mean() - ask_vwap.mean()).values
    # 10: last_bid_vol1 - 用最后一个有效分钟
    last_bid_vol1 = bid_vol1.iloc[last_valid_pos].values if len(bid_vol1) > 0 else np.full(len(symbols), np.nan)
    # 11: last_ask_vol1
    last_ask_vol1 = ask_vol1.iloc[last_valid_pos].values if len(ask_vol1) > 0 else np.full(len(symbols), np.nan)

    result = np.column_stack([
        daily_ret, intraday_vol, log_total_amt, act_buy_ratio,
        log_avg_trade_amt, min_ret_skew, up_tick_ratio,
        bid_size1_mean, ask_size1_mean, vwap_diff,
        last_bid_vol1, last_ask_vol1,
    ]).astype(np.float64)

    return result, symbols


def _extract_micro_metrics_array(data: dict) -> Tuple[np.ndarray, pd.Index]:
    """
    从时段切片后的分钟数据提取8列微观指标, 返回 (n_stocks, 8) 的ndarray 和股票索引
    """
    amount = data['amount']
    act_buy_amt = data['act_buy_amount_sum']
    act_sell_amt = data['act_sell_amount_sum']
    act_buy_vol = data['act_buy_vol_sum']
    act_sell_vol = data['act_sell_vol_sum']
    act_buy_cnt = data['act_buy_count_sum']
    act_sell_cnt = data['act_sell_count_sum']
    bid_size1 = data['bid_size_1_mean']
    ask_size1 = data['ask_size_1_mean']
    bid_size6 = data['bid_size_6_mean']
    ask_size6 = data['ask_size_6_mean']
    bid_vwap = data['bid_vwap_mean']
    ask_vwap = data['ask_vwap_mean']

    symbols = amount.columns
    total_amt = amount.sum().values.astype(float)
    total_cnt = (act_buy_cnt.sum() + act_sell_cnt.sum()).values.astype(float)
    total_vol = (act_buy_vol.sum() + act_sell_vol.sum()).values.astype(float)
    total_cnt_safe = total_cnt.copy()
    total_cnt_safe[total_cnt_safe == 0] = np.nan

    # 0: act_buy_ratio
    m0 = act_buy_amt.sum().values / (total_amt + 1e-10)
    # 1: bid_ask_imbalance
    m1 = (act_buy_amt.sum().values - act_sell_amt.sum().values) / (total_amt + 1e-10)
    # 2: vol_per_trade
    total_vol_safe = total_vol.copy()
    total_vol_safe[total_vol_safe == 0] = np.nan
    m2 = np.log(total_vol / total_cnt_safe)
    # 3: spread_ratio
    depth_total = (bid_size1.sum() + ask_size1.sum()).values.astype(float)
    m3 = (ask_size1.sum().values - bid_size1.sum().values) / (depth_total + 1e-10)
    # 4: depth_imbalance
    depth6_total = (bid_size6.sum() + ask_size6.sum()).values.astype(float)
    m4 = (bid_size6.sum().values - ask_size6.sum().values) / (depth6_total + 1e-10)
    # 5: vwap_deviation
    m5 = (bid_vwap.mean() - ask_vwap.mean()).values
    # 6: act_buy_vol_ratio
    m6 = act_buy_vol.sum().values / (total_vol + 1e-10)
    # 7: big_order_ratio
    median_amt = amount.median()
    big_order_amt = amount.where(amount > median_amt, 0).sum()
    m7 = big_order_amt.values / (total_amt + 1e-10)

    result = np.column_stack([m0, m1, m2, m3, m4, m5, m6, m7]).astype(np.float64)
    return result, symbols


def prepare_features_list(
    date: int,
    lookback: int = 5,
    segment: str = 'all_day',
) -> Tuple[List[np.ndarray], List[int], np.ndarray]:
    """
    准备 theme_cluster_factors 所需的 features_list

    参数:
        date: 目标日期 (如 20220819)
        lookback: 回溯天数
        segment: 时段 ('all_day', 'open30', 'morning', 'afternoon', 'close30')

    返回:
        (features_list, dates, symbols)
        - features_list: List[ndarray], 每个shape=(n_stocks, 12), 按日期升序
        - dates: 交易日列表
        - symbols: 统一的股票代码数组 (用于最终构建DataFrame)
    """
    start_date = _last_n_trading_date(date, lookback)
    dates = _get_trading_dates(start_date, date)

    all_fields = list(set(_FEATURE_FIELDS))
    day_data = {}
    for d in dates:
        raw = _read_minute_fields(d, all_fields)
        day_data[d] = _get_time_slice(raw, segment)

    # 提取各天特征, 收集股票集合
    day_features = {}
    day_symbols = {}
    for d in dates:
        feat, syms = _extract_daily_features_array(day_data[d])
        day_features[d] = feat
        day_symbols[d] = syms

    # 取所有天都出现的股票交集, 保持顺序一致
    common_symbols = day_symbols[dates[0]]
    for d in dates[1:]:
        common_symbols = common_symbols.intersection(day_symbols[d])
    common_symbols = common_symbols.sort_values()

    # 对齐到统一股票列表
    features_list = []
    for d in dates:
        sym_idx = pd.Index(day_symbols[d])
        idx = np.array([sym_idx.get_loc(s) for s in common_symbols])
        features_list.append(day_features[d][idx])

    return features_list, dates, common_symbols.values


def prepare_micro_metrics_list(
    date: int,
    lookback: int = 5,
    segment: str = 'all_day',
) -> Tuple[List[np.ndarray], List[int], np.ndarray]:
    """
    准备 theme_cluster_factors 所需的 micro_metrics_list

    参数:
        date: 目标日期
        lookback: 回溯天数
        segment: 时段

    返回:
        (micro_list, dates, symbols)
        - micro_list: List[ndarray], 每个shape=(n_stocks, 8), 按日期升序
        - dates: 交易日列表
        - symbols: 统一的股票代码数组
    """
    start_date = _last_n_trading_date(date, lookback)
    dates = _get_trading_dates(start_date, date)

    all_fields = list(set(_FEATURE_FIELDS + _MICRO_EXTRA_FIELDS))
    day_data = {}
    for d in dates:
        raw = _read_minute_fields(d, all_fields)
        day_data[d] = _get_time_slice(raw, segment)

    day_micro = {}
    day_symbols = {}
    for d in dates:
        micro, syms = _extract_micro_metrics_array(day_data[d])
        day_micro[d] = micro
        day_symbols[d] = syms

    common_symbols = day_symbols[dates[0]]
    for d in dates[1:]:
        common_symbols = common_symbols.intersection(day_symbols[d])
    common_symbols = common_symbols.sort_values()

    micro_list = []
    for d in dates:
        sym_idx = pd.Index(day_symbols[d])
        idx = np.array([sym_idx.get_loc(s) for s in common_symbols])
        micro_list.append(day_micro[d][idx])

    return micro_list, dates, common_symbols.values


def prepare_all_data(
    date: int,
    lookback: int = 5,
    segment: str = 'all_day',
) -> Tuple[List[np.ndarray], List[np.ndarray], List[int], np.ndarray]:
    """
    一次性准备 features_list 和 micro_metrics_list (共享分钟数据读取, 避免重复IO)

    参数:
        date: 目标日期
        lookback: 回溯天数
        segment: 时段

    返回:
        (features_list, micro_list, dates, symbols)
    """
    start_date = _last_n_trading_date(date, lookback)
    dates = _get_trading_dates(start_date, date)
    
    # 一次性读取所有字段所有日数据
    all_fields = list(set(_FEATURE_FIELDS + _MICRO_EXTRA_FIELDS))
    all_data = {f: dw.read_minute_data(f, dates[0], dates[-1]) for f in all_fields}
    
    start_idx, end_idx = TIME_SEGMENTS[segment]
    
    day_features = {}
    day_micro = {}
    day_symbols_feat = {}
    day_symbols_micro = {}

    for d in dates:
        # 按天切片分钟数据
        sliced = {f: all_data[f].loc[d].iloc[start_idx:end_idx] for f in all_fields}
        feat, syms_f = _extract_daily_features_array(sliced)
        micro, syms_m = _extract_micro_metrics_array(sliced)
        day_features[d] = feat
        day_micro[d] = micro
        day_symbols_feat[d] = syms_f
        day_symbols_micro[d] = syms_m
    
    # 取所有天所有数据的股票交集
    common_symbols = day_symbols_feat[dates[0]]
    for d in dates:
        common_symbols = common_symbols.intersection(day_symbols_feat[d])
        common_symbols = common_symbols.intersection(day_symbols_micro[d])
    common_symbols = common_symbols.sort_values()
    
    # 使用get_indexer一次性获取所有索引
    common_idx = pd.Index(common_symbols)
    features_list = []
    micro_list = []
    for d in dates:
        idx_f = pd.Index(day_symbols_feat[d]).get_indexer(common_idx)
        idx_m = pd.Index(day_symbols_micro[d]).get_indexer(common_idx)
        features_list.append(day_features[d][idx_f])
        micro_list.append(day_micro[d][idx_m])

    return features_list, micro_list, dates, common_symbols.values


def prepare_all_data_multi_segments(
    date: int,
    lookback: int = 5,
    segments: List[str] = ['all_day'],
) -> Tuple[Dict[str, List[np.ndarray]], Dict[str, List[np.ndarray]], List[int], np.ndarray]:
    """
    一次性准备多个时段的 features_list 和 micro_metrics_list

    参数:
        date: 目标日期
        lookback: 回溯天数
        segments: 时段列表

    返回:
        (features_by_segment, micro_by_segment, dates, symbols)
        - features_by_segment: Dict[segment, List[ndarray]], 每个shape=(n_stocks, 12)
        - micro_by_segment: Dict[segment, List[ndarray]], 每个shape=(n_stocks, 8)
        - dates: 交易日列表
        - symbols: 统一的股票代码数组
    """
    start_date = _last_n_trading_date(date, lookback)
    dates = _get_trading_dates(start_date, date)
    
    # 一次性读取所有字段所有日数据
    all_fields = list(set(_FEATURE_FIELDS + _MICRO_EXTRA_FIELDS))
    all_data = {f: dw.read_minute_data(f, dates[0], dates[-1]) for f in all_fields}
    
    # 各时段的边界
    segment_bounds = {seg: TIME_SEGMENTS[seg] for seg in segments}
    
    # 按天按时段提取特征
    features_by_segment: Dict[str, Dict[int, np.ndarray]] = {seg: {} for seg in segments}
    micro_by_segment: Dict[str, Dict[int, np.ndarray]] = {seg: {} for seg in segments}
    symbols_by_segment: Dict[str, Dict[int, pd.Index]] = {seg: {} for seg in segments}
    
    for d in dates:
        for seg in segments:
            start_idx, end_idx = segment_bounds[seg]
            sliced = {f: all_data[f].loc[d].iloc[start_idx:end_idx] for f in all_fields}
            feat, syms_f = _extract_daily_features_array(sliced)
            micro, syms_m = _extract_micro_metrics_array(sliced)
            features_by_segment[seg][d] = feat
            micro_by_segment[seg][d] = micro
            symbols_by_segment[seg][d] = syms_f
    
    # 取所有时段所有天的股票交集
    common_symbols = None
    for seg in segments:
        seg_symbols = symbols_by_segment[seg][dates[0]]
        for d in dates:
            seg_symbols = seg_symbols.intersection(symbols_by_segment[seg][d])
        if common_symbols is None:
            common_symbols = seg_symbols
        else:
            common_symbols = common_symbols.intersection(seg_symbols)
    common_symbols = common_symbols.sort_values()
    
    # 对齐索引并构建最终输出
    common_idx = pd.Index(common_symbols)
    features_out: Dict[str, List[np.ndarray]] = {seg: [] for seg in segments}
    micro_out: Dict[str, List[np.ndarray]] = {seg: [] for seg in segments}
    
    for seg in segments:
        for d in dates:
            idx = pd.Index(symbols_by_segment[seg][d]).get_indexer(common_idx)
            features_out[seg].append(features_by_segment[seg][d][idx])
            micro_out[seg].append(micro_by_segment[seg][d][idx])
    
    return features_out, micro_out, dates, common_symbols.values


# 分钟级数据字段（用于Rust内部特征提取）
MINUTE_DATA_FIELDS = [
    'open', 'close', 'amount', 'volume',
    'act_buy_amount_sum', 'act_sell_amount_sum',
    'act_buy_count_sum', 'act_sell_count_sum',
    'act_buy_vol_sum', 'act_sell_vol_sum',
    'up_tick_count', 'down_tick_count',
    'bid_size_1_mean', 'ask_size_1_mean',
    'bid_size_6_mean', 'ask_size_6_mean',
    'bid_vwap_mean', 'ask_vwap_mean',
    'bid_vol1', 'ask_vol1',
]


def prepare_minute_data_for_theme_cluster(
    date: int,
    lookback: int = 5,
) -> Tuple[List[np.ndarray], List[int], np.ndarray]:
    """
    准备分钟级原始数据供Rust内部特征提取
    
    参数:
        date: 目标日期
        lookback: 回溯天数
    
    返回:
        (minute_data_list, dates, symbols)
        - minute_data_list: List[ndarray], 每个shape=(n_minutes, n_stocks, 20)
        - dates: 交易日列表
        - symbols: 股票代码数组
    """
    start_date = _last_n_trading_date(date, lookback)
    dates = _get_trading_dates(start_date, date)
    
    # 一次性读取所有字段所有日数据
    all_data = {f: dw.read_minute_data(f, dates[0], dates[-1]) for f in MINUTE_DATA_FIELDS}
    
    # 提取股票交集
    symbols = None
    for d in dates:
        day_symbols = all_data['close'].loc[d].columns
        if symbols is None:
            symbols = day_symbols
        else:
            symbols = symbols.intersection(day_symbols)
    symbols = symbols.sort_values()
    
    # 构建分钟数据数组
    n_minutes = 238  # 全天分钟数
    n_stocks = len(symbols)
    n_fields = len(MINUTE_DATA_FIELDS)
    
    minute_data_list = []
    for d in dates:
        # shape: (n_minutes, n_stocks, n_fields)
        day_arr = np.zeros((n_minutes, n_stocks, n_fields), dtype=np.float64)
        
        for f_idx, f in enumerate(MINUTE_DATA_FIELDS):
            df = all_data[f].loc[d][symbols]  # 按symbol顺序取列
            day_arr[:, :, f_idx] = df.values
        
        minute_data_list.append(day_arr)
    
    return minute_data_list, dates, symbols.values


def prepare_minute_data_for_theme_cluster_optimized(
    date: int,
    lookback: int = 5,
) -> Tuple[np.ndarray, List[int], np.ndarray]:
    """
    准备分钟级原始数据（单一大数组版本，减少内存开销）
    
    参数:
        date: 目标日期
        lookback: 回溯天数
    
    返回:
        (minute_data, dates, symbols)
        - minute_data: ndarray, shape=(n_days, n_minutes, n_stocks, 20)
        - dates: 交易日列表
        - symbols: 股票代码数组
    """
    start_date = _last_n_trading_date(date, lookback)
    dates = _get_trading_dates(start_date, date)
    
    # 一次性读取所有字段所有日数据
    all_data = {f: dw.read_minute_data(f, dates[0], dates[-1]) for f in MINUTE_DATA_FIELDS}
    
    # 提取股票交集
    symbols = None
    for d in dates:
        day_symbols = all_data['close'].loc[d].columns
        if symbols is None:
            symbols = day_symbols
        else:
            symbols = symbols.intersection(day_symbols)
    symbols = symbols.sort_values()
    
    # 构建单一大数组
    n_days = len(dates)
    n_minutes = 240  # 全天分钟数
    n_stocks = len(symbols)
    n_fields = len(MINUTE_DATA_FIELDS)
    
    minute_data = np.zeros((n_days, n_minutes, n_stocks, n_fields), dtype=np.float64)
    
    for d_idx, d in enumerate(dates):
        for f_idx, f in enumerate(MINUTE_DATA_FIELDS):
            df = all_data[f].loc[d][symbols]
            minute_data[d_idx, :, :, f_idx] = df.values
    
    return minute_data, dates, symbols.values


def compute_theme_cluster_factors_from_minute(
    date: int,
    lookback: int = 5,
    segments: List[str] = ['all_day'],
    k_values: List[int] = [20],
    align_methods: List[str] = ['hungarian'],
    distance_metrics: List[str] = ['euclidean'],
    distance_thresholds: List[float] = [-1.0],
    lookback_values: List[int] = [5],
    n_threads: int = 8,
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """
    从分钟数据一站式计算主题聚类因子（Rust内部做时段拆分和特征提取）
    
    参数:
        date: 目标日期
        lookback: 数据准备的回溯天数（自动取lookback_values的最大值）
        segments: 时段列表
        k_values: 聚类数列表
        align_methods: 对齐方法列表
        distance_metrics: 距离度量列表
        distance_thresholds: 距离阈值列表
        lookback_values: 因子计算的回溯天数列表
        n_threads: 并行线程数
    
    返回:
        (因子矩阵, 列名列表, 股票代码)
    """
    import rust_pyfunc as rp
    
    # 自动调整lookback以覆盖最大的lookback_values
    actual_lookback = max(lookback, max(lookback_values))
    
    # 准备分钟级原始数据
    minute_data, dates, symbols = prepare_minute_data_for_theme_cluster_optimized(
        date=date, lookback=actual_lookback
    )
    
    # 时段边界
    segment_bounds = [TIME_SEGMENTS[seg] for seg in segments]  # List[(start, end)]
    
    # 调用Rust函数
    result, col_names = rp.theme_cluster_factors_from_minute(
        minute_data=minute_data,
        segments=segments,
        segment_bounds=segment_bounds,
        k_values=k_values,
        align_methods=align_methods,
        distance_metrics=distance_metrics,
        distance_thresholds=distance_thresholds,
        lookback_values=lookback_values,
        n_threads=n_threads,
    )
    
    return result, col_names, symbols
