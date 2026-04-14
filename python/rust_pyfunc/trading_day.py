"""
交易日(TradingDay)工具模块 - 仅依赖 numpy/pandas

提供交易日历的加载、查询、推算等功能。
数据源: /ssd_data/data/basic_info/calendar.csv (无表头, 每行一个整数日期, 格式 %Y%m%d)

使用方式:
    import rust_pyfunc as rp
    rp.td.last_trading_day(20200101)       # 上一个交易日
    rp.td.next_trading_day(20200101)       # 下一个交易日
    rp.td.is_trading_day(20200101)         # 是否交易日
    rp.td.get_range(20200101, 20200301)    # 范围内的交易日数组
    rp.last_trading_day_tricky(20200101)   # 若非交易日则回退到上一个交易日
    rp.next_trading_day_tricky(20200101)   # 若非交易日则前进到下一个交易日
"""
import os
from datetime import datetime, timedelta

import numpy
import pandas


class TradingDay:
    """交易日历类, 从CSV文件加载交易日并支持各种查询操作

    Parameters
    ----------
    input_file : str
        交易日历CSV文件路径, 默认 /ssd_data/data/basic_info/calendar.csv
    start_date : int, optional
        起始日期, 格式 %Y%m%d, 如 20200101
    end_date : int, optional
        截止日期, 格式 %Y%m%d, 如 20201231

    Attributes
    ----------
    trading_days : numpy.ndarray[int]
        有序交易日数组
    date_map : dict[int, int]
        日期 -> 在 trading_days 中的索引
    """

    def __init__(self, input_file='/ssd_data/data/basic_info/calendar.csv',
                 start_date=None, end_date=None):
        self.trading_days = self._load_trading_days(input_file, start_date, end_date)
        self.date_map = dict(zip(self.trading_days, numpy.arange(self.length)))

    @staticmethod
    def _load_trading_days(path, start_date, end_date):
        df = pandas.read_csv(path, dtype=numpy.int64, names=["date"])
        trading_days = df["date"].sort_values()
        if start_date is not None or end_date is not None:
            mask = numpy.ones(len(trading_days), dtype=bool)
            if start_date is not None:
                mask &= trading_days.values >= start_date
            if end_date is not None:
                mask &= trading_days.values <= end_date
            trading_days = trading_days[mask]
        return trading_days.values

    @property
    def start_date(self):
        return self.trading_days[0]

    @property
    def end_date(self):
        return self.trading_days[-1]

    @property
    def length(self):
        return len(self.trading_days)

    def distance(self, start_date, end_date):
        """两个日期之间的交易日天数差"""
        return self.date_map[end_date] - self.date_map[start_date]

    def get_loc(self, date_):
        """获取某日期在交易日数组中的索引位置, 非交易日会抛 KeyError"""
        return self.date_map[date_]

    def is_trading_day(self, date_):
        """判断某日是否为交易日"""
        return date_ in self.date_map

    def last_trading_day(self, date_):
        """获取 date_ 之前的最近一个交易日"""
        return self.trading_days[self.trading_days < date_][-1]

    def next_trading_day(self, date_):
        """获取 date_ 之后的最近一个交易日"""
        return self.trading_days[self.trading_days > date_][0]

    def day_after_last(self, date_):
        """获取上一个交易日的下一个自然日"""
        last = self.last_trading_day(date_)
        dt = datetime.strptime(str(last), "%Y%m%d") + timedelta(days=1)
        return int(dt.strftime("%Y%m%d"))

    def trading_day_pair(self, date_):
        """获取包含 date_ 的最小交易日区间

        若 date_ 是交易日, 返回 (date_, next_trading_day(date_))
        否则返回 (last_trading_day(date_), next_trading_day(date_))
        """
        if self.is_trading_day(date_):
            return date_, self.next_trading_day(date_)
        return self.last_trading_day(date_), self.next_trading_day(date_)

    def get_range(self, start_date, end_date):
        """获取闭区间 [start_date, end_date] 内的所有交易日"""
        mask = (self.trading_days >= start_date) & (self.trading_days <= end_date)
        return self.trading_days[mask]


# ── 模块级单例 & 包装函数 ──────────────────────────────────────────────
td = TradingDay()


def last_trading_day_tricky(date):
    """若 date 非交易日则回退到上一个交易日, 否则原样返回"""
    if not td.is_trading_day(date):
        date = td.last_trading_day(date)
    return date


def next_trading_day_tricky(date):
    """若 date 非交易日则前进到下一个交易日, 否则原样返回"""
    if not td.is_trading_day(date):
        date = td.next_trading_day(date)
    return date


def last_n_trading_date(end_date, n_date):
    """从 end_date 往前数 n_date 个交易日, 返回那个日期"""
    pos = td.get_loc(end_date) - n_date
    return td.trading_days[pos]
