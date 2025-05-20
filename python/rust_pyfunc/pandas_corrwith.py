"""
实现与pandas的corrwith方法相同功能的包装函数，处理两个DataFrame的列名匹配情况
"""

import pandas as pd
import numpy as np
from typing import Union, Optional, List
from rust_pyfunc.rust_pyfunc import dataframe_corrwith


def corrwith(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    axis: int = 0,
) -> pd.Series:
    """计算两个DataFrame对应列或行之间的相关系数。
    
    这个函数是pandas的corrwith方法的Rust实现包装，用于计算两个DataFrame中对应列（或行）
    之间的皮尔逊相关系数。会自动处理两个DataFrame的列名（或行名）匹配情况。
    
    参数：
    -----
    df1 : pd.DataFrame
        第一个数据框
    df2 : pd.DataFrame
        第二个数据框
    axis : int, 默认为0
        计算相关性的轴，0表示按列计算，1表示按行计算
        
    返回值：
    -------
    pd.Series
        包含对应列（或行）相关系数的Series，索引为共同的列名（或行名）
        
    示例：
    ------
    >>> df1 = pd.DataFrame({
    ...     'A': [1.0, 2.0, 3.0, 4.0, 5.0],
    ...     'B': [5.0, 4.0, 3.0, 2.0, 1.0],
    ...     'C': [2.0, 4.0, 6.0, 8.0, 10.0]
    ... })
    >>> df2 = pd.DataFrame({
    ...     'A': [1.1, 2.2, 2.9, 4.1, 5.2],
    ...     'B': [5.2, 4.1, 2.9, 2.1, 0.9],
    ...     'D': [1.0, 2.0, 3.0, 4.0, 5.0]
    ... })
    >>> corrwith(df1, df2)
    A    0.998867
    B    0.997492
    dtype: float64
    """
    cols = df1.columns.intersection(df2.columns)
    indexs = df1.index.intersection(df2.index)
    if axis == 0:
        df1 = df1.reindex(index=indexs, columns=cols).to_numpy(dtype=float)
        df2 = df2.reindex(index=indexs, columns=cols).to_numpy(dtype=float)
        return pd.Series(dataframe_corrwith(df1, df2, axis=0), index=cols)
    elif axis == 1:
        df1 = df1.T.reindex(index=cols, columns=indexs).to_numpy(dtype=float)
        df2 = df2.T.reindex(index=cols, columns=indexs).to_numpy(dtype=float)
        return pd.Series(dataframe_corrwith(df1, df2, axis=0), index=indexs)