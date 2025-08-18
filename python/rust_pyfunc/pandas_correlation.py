"""
pandas DataFrame 相关性矩阵扩展函数
提供直接处理DataFrame的高性能相关性矩阵计算函数
"""

import pandas as pd
import numpy as np
from . import rust_pyfunc


def fast_correlation_matrix_v2_df(
    df: pd.DataFrame,
    method: str = "pearson",
    min_periods: int = 1,
    max_workers: int = 10
) -> pd.DataFrame:
    """
    超快速计算DataFrame的相关性矩阵，进一步优化版本
    
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
    >>> from rust_pyfunc.pandas_correlation import fast_correlation_matrix_v2_df
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
    >>> 
    >>> # 使用不同参数
    >>> corr_fast = fast_correlation_matrix_v2_df(df, max_workers=16, min_periods=10)
    
    与原生pandas对比：
    ------------------
    >>> # 性能测试示例
    >>> large_df = pd.DataFrame(np.random.randn(5000, 100))
    >>> 
    >>> # pandas方式（慢）
    >>> import time
    >>> start = time.time()
    >>> pandas_result = large_df.corr()
    >>> pandas_time = time.time() - start
    >>> 
    >>> # rust_pyfunc方式（快）
    >>> start = time.time()
    >>> rust_result = fast_correlation_matrix_v2_df(large_df)
    >>> rust_time = time.time() - start
    >>> 
    >>> print(f"pandas耗时: {pandas_time:.2f}秒")
    >>> print(f"rust_pyfunc耗时: {rust_time:.2f}秒") 
    >>> print(f"性能提升: {pandas_time/rust_time:.1f}倍")
    >>> 
    >>> # 验证结果一致性
    >>> print(f"结果差异最大值: {np.abs(pandas_result - rust_result).max().max():.10f}")
    
    注意事项：
    --------
    - 函数会自动处理NaN值
    - 相关性矩阵是对称的，对角线元素为1.0
    - 当样本数少于min_periods时，对应的相关系数为NaN
    - 输入DataFrame将被转换为float64类型进行计算
    - 如果DataFrame包含非数值列，需要事先过滤掉或转换
    """
    
    # 检查输入类型
    if not isinstance(df, pd.DataFrame):
        raise TypeError("输入必须是pandas DataFrame")
    
    if df.empty:
        # 空DataFrame返回空的相关性矩阵
        return pd.DataFrame(index=df.columns, columns=df.columns, dtype=float)
    
    if len(df.columns) == 0:
        # 没有列的DataFrame
        return pd.DataFrame(dtype=float)
        
    # 转换为numpy数组（float64类型）
    try:
        numpy_data = df.to_numpy(dtype=float)
    except (ValueError, TypeError) as e:
        raise ValueError(f"DataFrame无法转换为float64类型: {e}。请确保所有列都是数值类型或可转换为数值类型。")
    
    # 检查数据维度
    if numpy_data.shape[0] < min_periods:
        raise ValueError(f"样本数量({numpy_data.shape[0]})少于最小要求({min_periods})")
    
    # 调用Rust实现的相关性矩阵计算函数
    correlation_matrix = rust_pyfunc.fast_correlation_matrix_v2(
        numpy_data,
        method=method,
        min_periods=min_periods,
        max_workers=max_workers
    )
    
    # 构造结果DataFrame，使用原有列名作为行名和列名
    result_df = pd.DataFrame(
        correlation_matrix,
        index=df.columns,
        columns=df.columns
    )
    
    return result_df


# 为了方便使用，提供简短的别名
fast_corr_df = fast_correlation_matrix_v2_df
correlation_matrix_df = fast_correlation_matrix_v2_df