"""pandas DataFrame 相关性矩阵扩展函数类型声明"""
from typing import Optional
import pandas as pd


def fast_correlation_matrix_v2_df(
    df: pd.DataFrame,
    method: str = "pearson",
    min_periods: int = 1,
    max_workers: int = 10
) -> pd.DataFrame:
    """超快速计算DataFrame的相关性矩阵，进一步优化版本。
    
    这是rust_pyfunc.fast_correlation_matrix_v2的DataFrame封装版本，可以直接传入DataFrame
    并返回保持原有列名作为索引和列名的相关性矩阵DataFrame。
    
    参数说明：
    ----------
    df : pandas.DataFrame
        输入的DataFrame，每一列代表一个变量，会自动转换为float64类型
    method : str, optional
        相关性计算方法，默认为"pearson"。目前只支持皮尔逊相关系数
    min_periods : int, optional
        计算相关性所需的最小样本数，默认为1
    max_workers : int, optional
        最大并行工作线程数，默认为10，设置为0则使用所有可用核心
        
    返回值：
    -------
    pandas.DataFrame
        相关性矩阵DataFrame，行名和列名都是原DataFrame的列名
        矩阵是对称的，对角线元素为1.0
        
    性能特点：
    ----------
    1. 采用SIMD优化、更好的内存访问模式和数值稳定性改进
    2. V2版本使用数据预处理、Kahan求和、循环展开、向量化计算
    3. 内存访问模式优化，提高缓存命中率
    4. 数值稳定性更好，减少浮点数累加误差
    5. 对于大数据集性能比pandas.DataFrame.corr()快数倍
    
    示例：
    -------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from rust_pyfunc import fast_correlation_matrix_v2_df
    >>> 
    >>> # 创建测试DataFrame
    >>> np.random.seed(42)
    >>> df = pd.DataFrame({
    ...     'A': np.random.randn(1000),
    ...     'B': np.random.randn(1000),
    ...     'C': np.random.randn(1000),
    ...     'D': np.random.randn(1000)
    ... })
    >>> 
    >>> # 使用高性能相关性计算
    >>> corr_matrix = fast_correlation_matrix_v2_df(df)
    >>> print(corr_matrix)
    >>> #           A         B         C         D
    >>> # A  1.000000  0.012345 -0.023456  0.034567
    >>> # B  0.012345  1.000000  0.045678 -0.056789  
    >>> # C -0.023456  0.045678  1.000000  0.067890
    >>> # D  0.034567 -0.056789  0.067890  1.000000
    >>> 
    >>> # 等价于（但更快）：
    >>> # pandas_corr = df.corr(method='pearson', min_periods=1)
    
    注意事项：
    --------
    - 函数会自动处理NaN值
    - 相关性矩阵是对称的，对角线元素为1.0
    - 当样本数少于min_periods时，对应的相关系数为NaN
    - 输入DataFrame将被转换为float64类型进行计算
    - 如果DataFrame包含非数值列，需要事先过滤掉或转换
    """
    ...


def fast_corr_df(
    df: pd.DataFrame,
    method: str = "pearson",
    min_periods: int = 1,
    max_workers: int = 10
) -> pd.DataFrame:
    """fast_correlation_matrix_v2_df的简短别名。
    
    参数说明：
    ----------
    df : pandas.DataFrame
        输入的DataFrame，每一列代表一个变量
    method : str, optional
        相关性计算方法，默认为"pearson"
    min_periods : int, optional
        计算相关性所需的最小样本数，默认为1
    max_workers : int, optional
        最大并行工作线程数，默认为10
        
    返回值：
    -------
    pandas.DataFrame
        相关性矩阵DataFrame
        
    注意：
    -----
    这是fast_correlation_matrix_v2_df的别名函数，功能完全相同。
    """
    ...


def correlation_matrix_df(
    df: pd.DataFrame,
    method: str = "pearson",
    min_periods: int = 1,
    max_workers: int = 10
) -> pd.DataFrame:
    """fast_correlation_matrix_v2_df的描述性别名。
    
    参数说明：
    ----------
    df : pandas.DataFrame
        输入的DataFrame，每一列代表一个变量
    method : str, optional
        相关性计算方法，默认为"pearson"
    min_periods : int, optional
        计算相关性所需的最小样本数，默认为1
    max_workers : int, optional
        最大并行工作线程数，默认为10
        
    返回值：
    -------
    pandas.DataFrame
        相关性矩阵DataFrame
        
    注意：
    -----
    这是fast_correlation_matrix_v2_df的别名函数，功能完全相同。
    """
    ...