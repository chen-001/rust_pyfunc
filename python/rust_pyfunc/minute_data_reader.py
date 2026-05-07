from __future__ import annotations

from functools import lru_cache
from typing import List

import numpy as np
import pandas as pd


@lru_cache(maxsize=1)
def _get_minute_data_infos():
    import h5py

    _min_dir = "/ssd_data/data/1min_factor_text"
    _cal_map = pd.read_csv(f"{_min_dir}/calendar_map.csv", dtype={"date_min": int})
    _symbols = pd.read_csv(f"{_min_dir}/symbol_map.csv", dtype={"symbol": str})["symbol"].tolist()
    return _min_dir, _cal_map, _symbols, h5py


def read_minute_data(field: str, start_date: int, end_date: int) -> pd.DataFrame:
    """读取A股1分钟频率数据。

    数据路径: /ssd_data/data/1min_factor_text/{field}.h5
    交易日240根K线: 09:31-11:30 + 13:01-15:00
    每交易日最后4根K线(index % 240 >= 237)置为NaN

    可用字段:
      成交: act_buy_amount_sum, act_sell_amount_sum, act_buy_vol_sum, act_sell_vol_sum,
            act_buy_count_sum, act_sell_count_sum, volume, amount, turnover,
            up_tick_count, down_tick_count
      价格: close, high, low, open
      盘口均值: bid_size_1_mean, ask_size_1_mean, bid_size_6_mean, ask_size_6_mean,
               bid_size_10_mean, ask_size_10_mean, bid_vwap_mean, ask_vwap_mean,
               spread_over_tick_size_mean
      盘口快照: ask_prc1~10, ask_vol1~10, bid_prc1~10, bid_vol1~10
      加权均价: ask_vwap3, ask_vwap5, ask_vwap10, bid_vwap3, bid_vwap5, bid_vwap10

    Returns:
        pd.DataFrame: index=0..n-1, columns=股票代码(str,含前导零), values=float64
    """
    _min_dir, _cal_map, _symbols, h5py = _get_minute_data_infos()
    sp = _cal_map.index[_cal_map["date_min"] == start_date][0]
    ep = _cal_map.index[_cal_map["date_min"] == end_date][0] + 1
    with h5py.File(f"{_min_dir}/{field}.h5", "r") as f:
        arr = f["data"][sp * 240 : ep * 240, : len(_symbols)]
    df = pd.DataFrame(arr, columns=_symbols)
    df[df.index % 240 >= 237] = np.nan
    return df
