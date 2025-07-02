"""Pandas扩展函数类型声明"""
from typing import List, Optional, Any
import pandas as pd
import numpy as np
from numpy.typing import NDArray

def dataframe_corrwith(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    axis: int = 0,
    drop_nan: bool = True,
    method: str = "pearson"
) -> pd.Series:
    """高性能DataFrame相关性计算。
    
    参数说明：
    ----------
    df1 : pd.DataFrame
        第一个DataFrame
    df2 : pd.DataFrame
        第二个DataFrame
    axis : int
        计算轴，0表示按列，1表示按行
    drop_nan : bool
        是否删除NaN值
    method : str
        相关性方法，支持"pearson", "spearman"
        
    返回值：
    -------
    pd.Series
        相关性结果
    """
    ...

def rank_axis1(df: pd.DataFrame, method: str = "average", ascending: bool = True) -> pd.DataFrame:
    """高性能按行排名函数。
    
    参数说明：
    ----------
    df : pd.DataFrame
        输入DataFrame
    method : str
        排名方法，支持"average", "min", "max", "first", "dense"
    ascending : bool
        是否升序排列
        
    返回值：
    -------
    pd.DataFrame
        排名结果DataFrame
    """
    ...

def fast_merge(
    left: pd.DataFrame,
    right: pd.DataFrame,
    left_on: str,
    right_on: str,
    how: str = "inner"
) -> pd.DataFrame:
    """高性能DataFrame合并函数。
    
    参数说明：
    ----------
    left : pd.DataFrame
        左侧DataFrame
    right : pd.DataFrame
        右侧DataFrame
    left_on : str
        左侧连接列名
    right_on : str
        右侧连接列名
    how : str
        连接方式，支持"inner", "left", "right", "outer"
        
    返回值：
    -------
    pd.DataFrame
        合并后的DataFrame
    """
    ...

def fast_merge_mixed(
    left: pd.DataFrame,
    right: pd.DataFrame,
    left_on: List[str],
    right_on: List[str],
    how: str = "inner"
) -> pd.DataFrame:
    """支持多列和混合类型的高性能合并。
    
    参数说明：
    ----------
    left : pd.DataFrame
        左侧DataFrame
    right : pd.DataFrame
        右侧DataFrame
    left_on : List[str]
        左侧连接列名列表
    right_on : List[str]
        右侧连接列名列表
    how : str
        连接方式
        
    返回值：
    -------
    pd.DataFrame
        合并后的DataFrame
    """
    ...

def fast_inner_join_dataframes(
    dfs: List[pd.DataFrame],
    on: str,
    suffixes: Optional[List[str]] = None
) -> pd.DataFrame:
    """多个DataFrame的高性能内连接。
    
    参数说明：
    ----------
    dfs : List[pd.DataFrame]
        DataFrame列表
    on : str
        连接列名
    suffixes : Optional[List[str]]
        列名后缀列表
        
    返回值：
    -------
    pd.DataFrame
        连接后的DataFrame
    """
    ...

# Python定义的函数类型声明

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
    """
    ...

def rank_axis1_df(
    df: pd.DataFrame,
    method: str = "average",
    ascending: bool = True,
    na_option: str = "keep"
) -> pd.DataFrame:
    """高性能的DataFrame rank函数，支持axis=1（沿行方向排名）"""
    ...

def rank_axis0_df(
    df: pd.DataFrame,
    method: str = "average", 
    ascending: bool = True,
    na_option: str = "keep"
) -> pd.DataFrame:
    """高性能的DataFrame rank函数，支持axis=0（沿列方向排名）"""
    ...

def fast_rank(
    arr: NDArray[np.float64],
    method: str = "average",
    ascending: bool = True
) -> NDArray[np.float64]:
    """快速排名函数"""
    ...

def fast_rank_axis1(
    arr: NDArray[np.float64],
    method: str = "average",
    ascending: bool = True
) -> NDArray[np.float64]:
    """沿axis=1快速排名"""
    ...

def fast_rank_axis0(
    arr: NDArray[np.float64], 
    method: str = "average",
    ascending: bool = True
) -> NDArray[np.float64]:
    """沿axis=0快速排名"""
    ...

def fast_merge_df(
    left: pd.DataFrame, 
    right: pd.DataFrame,
    on: Optional[str] = None,
    left_on: Optional[str] = None,
    right_on: Optional[str] = None,
    how: str = "inner"
) -> pd.DataFrame:
    """高性能的DataFrame merge函数"""
    ...

def fast_inner_join_df(
    left: pd.DataFrame,
    right: pd.DataFrame,
    on: str
) -> pd.DataFrame:
    """快速内连接的便捷函数"""
    ...

def fast_left_join_df(
    left: pd.DataFrame,
    right: pd.DataFrame,
    on: str
) -> pd.DataFrame:
    """快速左连接的便捷函数"""
    ...

def fast_right_join_df(
    left: pd.DataFrame,
    right: pd.DataFrame,
    on: str
) -> pd.DataFrame:
    """快速右连接的便捷函数"""
    ...

def fast_outer_join_df(
    left: pd.DataFrame,
    right: pd.DataFrame,
    on: str
) -> pd.DataFrame:
    """快速外连接的便捷函数"""
    ...

def fast_join(
    left: pd.DataFrame,
    right: pd.DataFrame,
    left_on: List[str],
    right_on: List[str],
    how: str = "inner"
) -> pd.DataFrame:
    """支持多列连接的快速join函数"""
    ...

def fast_merge_dataframe(
    left: pd.DataFrame,
    right: pd.DataFrame,
    left_on: List[str],
    right_on: List[str],
    how: str = "inner"
) -> pd.DataFrame:
    """高性能多列DataFrame合并函数"""
    ...