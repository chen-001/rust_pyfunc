from typing import Union, Literal, overload
import pandas as pd
import numpy as np
from rust_pyfunc.rust_pyfunc import rolling_window_stat

StatType = Literal["mean", "sum", "max", "min", "std", "median", "count", "rank", "skew", "trend_time", "trend_oneton", "last"]

@pd.api.extensions.register_dataframe_accessor("rolling_future")
@pd.api.extensions.register_series_accessor("rolling_future")
class RollingFutureAccessor:
    """用于在pandas DataFrame或Series上实现向后滚动窗口计算的访问器。
    
    支持的统计量类型：
    ---------------
    - mean: 计算后面窗口内的均值
    - sum: 计算后面窗口内的总和
    - max: 计算后面窗口内的最大值
    - min: 计算后面窗口内的最小值
    - std: 计算后面窗口内的标准差
    - median: 计算后面窗口内的中位数
    - count: 计算后面窗口内的数据点数量
    - rank: 计算当前值在后面窗口内的分位数（0到1之间）
    - skew: 计算后面窗口的偏度
    - trend_time: 计算后面窗口内数据序列与时间序列的相关系数
    - trend_oneton: 计算后面窗口内数据序列与1到n序列的相关系数（忽略时间间隔）
    - last: 计算后面窗口内的最后一个值
    
    注意：所有计算都不包括当前时间点的值，只考虑后面窗口内的值
    
    使用方法：
    ---------
    >>> import pandas as pd
    >>> from rust_pyfunc import rolling_future
    >>> # DataFrame示例
    >>> df = pd.DataFrame({
    ...     'time': pd.date_range('2024-01-01', periods=5, freq='s'),
    ...     'value': [1, 2, 3, 4, 5]
    ... })
    >>> df.set_index('time', inplace=True)
    >>> df.rolling_future('2s').mean()  # 计算每个时间点之后2秒内的均值
    >>> df.rolling_future('2s').rank()  # 计算每个值在后面2秒内的分位数
    >>> 
    >>> # Series示例
    >>> s = pd.Series([1, 2, 3, 4, 5], 
    ...               index=pd.date_range('2024-01-01', periods=5, freq='s'))
    >>> s.rolling_future('2s').mean()  # 计算每个时间点之后2秒内的均值
    >>> s.rolling_future('2s').trend_time()  # 计算后面2秒内的趋势
    """
    
    def __init__(self, pandas_obj: Union[pd.DataFrame, pd.Series]):
        """设置滚动窗口的大小。
        
        参数：
        -----
        window : str
            时间窗口大小，例如'5s'表示5秒，'1min'表示1分钟
        include_current : bool, default False
            是否在计算时包含当前行的值
            
        返回值：
        -------
        RollingFutureAccessor
            返回self以支持链式调用
        """
        self._obj = pandas_obj
        self._window = None
        self._include_current = False  # 默认不包含当前行
        
    def __call__(self, window: str, include_current: bool = False) -> "RollingFutureAccessor":
        """设置滚动窗口的大小。
        
        参数：
        -----
        window : str
            时间窗口大小，例如'5s'表示5秒，'1min'表示1分钟
        include_current : bool, default False
            是否在计算时包含当前行的值
            
        返回值：
        -------
        RollingFutureAccessor
            返回self以支持链式调用
        """
        self._window = pd.Timedelta(window).total_seconds()
        self._include_current = include_current
        return self
        
    def _apply_stat(self, column: str, stat_type: StatType) -> pd.Series:
        """对指定列应用统计计算。"""
        if not hasattr(self, '_window') or self._window is None:
            raise ValueError("必须先调用rolling_future(window)设置窗口大小")
            
        if not pd.api.types.is_datetime64_any_dtype(self._obj.index):
            raise ValueError("索引必须是datetime类型")
            
        # 转换为纳秒时间戳
        times = self._obj.index.astype(np.int64).to_numpy()  # pandas的时间戳默认就是纳秒
        window_ns = int(self._window * 1e9)  # 将秒转换为纳秒
        
        if isinstance(self._obj, pd.Series):
            values = self._obj.to_numpy().astype(np.float64)
            name = self._obj.name or column
        else:
            values = self._obj[column].to_numpy().astype(np.float64)
            name = column
        
        result = rolling_window_stat(times, values, window_ns, stat_type, self._include_current)
        return pd.Series(result, index=self._obj.index, name=name)
        
    @overload
    def mean(self) -> Union[pd.Series, pd.DataFrame]:
        """计算向后滚动窗口的均值。
        
        返回值：
        -------
        Union[pd.Series, pd.DataFrame]
            如果输入是Series，返回Series；如果输入是DataFrame，返回DataFrame
        """
        ...
        
    def mean(self) -> Union[pd.Series, pd.DataFrame]:
        """计算向后滚动窗口的均值。"""
        if isinstance(self._obj, pd.Series):
            return self._apply_stat("value", "mean")
        if len(self._obj.columns) == 1:
            return self._apply_stat(self._obj.columns[0], "mean")
        return pd.DataFrame({col: self._apply_stat(col, "mean") for col in self._obj.columns})
        
    @overload
    def sum(self) -> Union[pd.Series, pd.DataFrame]:
        """计算向后滚动窗口的总和。
        
        返回值：
        -------
        Union[pd.Series, pd.DataFrame]
            如果输入是Series，返回Series；如果输入是DataFrame，返回DataFrame
        """
        ...
        
    def sum(self) -> Union[pd.Series, pd.DataFrame]:
        """计算向后滚动窗口的总和。"""
        if isinstance(self._obj, pd.Series):
            return self._apply_stat("value", "sum")
        if len(self._obj.columns) == 1:
            return self._apply_stat(self._obj.columns[0], "sum")
        return pd.DataFrame({col: self._apply_stat(col, "sum") for col in self._obj.columns})
        
    @overload
    def max(self) -> Union[pd.Series, pd.DataFrame]:
        """计算向后滚动窗口的最大值。
        
        返回值：
        -------
        Union[pd.Series, pd.DataFrame]
            如果输入是Series，返回Series；如果输入是DataFrame，返回DataFrame
        """
        ...
        
    def max(self) -> Union[pd.Series, pd.DataFrame]:
        """计算向后滚动窗口的最大值。"""
        if isinstance(self._obj, pd.Series):
            return self._apply_stat("value", "max")
        if len(self._obj.columns) == 1:
            return self._apply_stat(self._obj.columns[0], "max")
        return pd.DataFrame({col: self._apply_stat(col, "max") for col in self._obj.columns})
        
    @overload
    def min(self) -> Union[pd.Series, pd.DataFrame]:
        """计算向后滚动窗口的最小值。
        
        返回值：
        -------
        Union[pd.Series, pd.DataFrame]
            如果输入是Series，返回Series；如果输入是DataFrame，返回DataFrame
        """
        ...
        
    def min(self) -> Union[pd.Series, pd.DataFrame]:
        """计算向后滚动窗口的最小值。"""
        if isinstance(self._obj, pd.Series):
            return self._apply_stat("value", "min")
        if len(self._obj.columns) == 1:
            return self._apply_stat(self._obj.columns[0], "min")
        return pd.DataFrame({col: self._apply_stat(col, "min") for col in self._obj.columns})
        
    @overload
    def std(self) -> Union[pd.Series, pd.DataFrame]:
        """计算向后滚动窗口的标准差。
        
        返回值：
        -------
        Union[pd.Series, pd.DataFrame]
            如果输入是Series，返回Series；如果输入是DataFrame，返回DataFrame
        """
        ...
        
    def std(self) -> Union[pd.Series, pd.DataFrame]:
        """计算向后滚动窗口的标准差。"""
        if isinstance(self._obj, pd.Series):
            return self._apply_stat("value", "std")
        if len(self._obj.columns) == 1:
            return self._apply_stat(self._obj.columns[0], "std")
        return pd.DataFrame({col: self._apply_stat(col, "std") for col in self._obj.columns})
        
    @overload
    def median(self) -> Union[pd.Series, pd.DataFrame]:
        """计算向后滚动窗口的中位数。
        
        返回值：
        -------
        Union[pd.Series, pd.DataFrame]
            如果输入是Series，返回Series；如果输入是DataFrame，返回DataFrame
        """
        ...
        
    def median(self) -> Union[pd.Series, pd.DataFrame]:
        """计算向后滚动窗口的中位数。"""
        if isinstance(self._obj, pd.Series):
            return self._apply_stat("value", "median")
        if len(self._obj.columns) == 1:
            return self._apply_stat(self._obj.columns[0], "median")
        return pd.DataFrame({col: self._apply_stat(col, "median") for col in self._obj.columns})
        
    @overload
    def count(self) -> Union[pd.Series, pd.DataFrame]:
        """计算向后滚动窗口内的数据点数量。
        
        返回值：
        -------
        Union[pd.Series, pd.DataFrame]
            如果输入是Series，返回Series；如果输入是DataFrame，返回DataFrame
        """
        ...
        
    def count(self) -> Union[pd.Series, pd.DataFrame]:
        """计算向后滚动窗口内的数据点数量。"""
        if isinstance(self._obj, pd.Series):
            return self._apply_stat("value", "count")
        if len(self._obj.columns) == 1:
            return self._apply_stat(self._obj.columns[0], "count")
        return pd.DataFrame({col: self._apply_stat(col, "count") for col in self._obj.columns})
        
    @overload
    def rank(self) -> Union[pd.Series, pd.DataFrame]:
        """计算当前值在后面窗口内的分位数（0到1之间）。
        
        返回值：
        -------
        Union[pd.Series, pd.DataFrame]
            如果输入是Series，返回Series；如果输入是DataFrame，返回DataFrame
        """
        ...
        
    def rank(self) -> Union[pd.Series, pd.DataFrame]:
        """计算当前值在后面窗口内的分位数（0到1之间）。"""
        if isinstance(self._obj, pd.Series):
            return self._apply_stat("value", "rank")
        if len(self._obj.columns) == 1:
            return self._apply_stat(self._obj.columns[0], "rank")
        return pd.DataFrame({col: self._apply_stat(col, "rank") for col in self._obj.columns})
        
    @overload
    def skew(self) -> Union[pd.Series, pd.DataFrame]:
        """计算向后滚动窗口的偏度。
        
        返回值：
        -------
        Union[pd.Series, pd.DataFrame]
            如果输入是Series，返回Series；如果输入是DataFrame，返回DataFrame
        """
        ...
        
    def skew(self) -> Union[pd.Series, pd.DataFrame]:
        """计算向后滚动窗口的偏度。"""
        if isinstance(self._obj, pd.Series):
            return self._apply_stat("value", "skew")
        if len(self._obj.columns) == 1:
            return self._apply_stat(self._obj.columns[0], "skew")
        return pd.DataFrame({col: self._apply_stat(col, "skew") for col in self._obj.columns})
        
    @overload
    def trend_time(self) -> Union[pd.Series, pd.DataFrame]:
        """计算向后滚动窗口内数据序列与时间序列的相关系数。
        
        返回值：
        -------
        Union[pd.Series, pd.DataFrame]
            如果输入是Series，返回Series；如果输入是DataFrame，返回DataFrame
        """
        ...
        
    def trend_time(self) -> Union[pd.Series, pd.DataFrame]:
        """计算向后滚动窗口内数据序列与时间序列的相关系数。"""
        if isinstance(self._obj, pd.Series):
            return self._apply_stat("value", "trend_time")
        if len(self._obj.columns) == 1:
            return self._apply_stat(self._obj.columns[0], "trend_time")
        return pd.DataFrame({col: self._apply_stat(col, "trend_time") for col in self._obj.columns})

    @overload
    def trend_oneton(self) -> Union[pd.Series, pd.DataFrame]:
        """计算向后滚动窗口内数据序列与1到n序列的相关系数（忽略时间间隔）。
        
        返回值：
        -------
        Union[pd.Series, pd.DataFrame]
            如果输入是Series，返回Series；如果输入是DataFrame，返回DataFrame
        """
        ...
        
    def trend_oneton(self) -> Union[pd.Series, pd.DataFrame]:
        """计算向后滚动窗口内数据序列与1到n序列的相关系数（忽略时间间隔）。"""
        if isinstance(self._obj, pd.Series):
            return self._apply_stat("value", "trend_oneton")
        if len(self._obj.columns) == 1:
            return self._apply_stat(self._obj.columns[0], "trend_oneton")
        return pd.DataFrame({col: self._apply_stat(col, "trend_oneton") for col in self._obj.columns})

    @overload
    def last(self) -> Union[pd.Series, pd.DataFrame]:
        """计算向后滚动窗口内的最后一个值。
        
        返回值：
        -------
        Union[pd.Series, pd.DataFrame]
            如果输入是Series，返回Series；如果输入是DataFrame，返回DataFrame
        """
        ...
        
    def last(self) -> Union[pd.Series, pd.DataFrame]:
        """计算向后滚动窗口内的最后一个值。"""
        if isinstance(self._obj, pd.Series):
            return self._apply_stat("value", "last")
        if len(self._obj.columns) == 1:
            return self._apply_stat(self._obj.columns[0], "last")
        return pd.DataFrame({col: self._apply_stat(col, "last") for col in self._obj.columns})

