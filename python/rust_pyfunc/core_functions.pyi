"""核心数学和统计函数类型声明"""
from typing import List, Optional, Union
import numpy as np
from numpy.typing import NDArray

def trend(arr: Union[NDArray[np.float64], List[Union[float, int]]]) -> float:
    """计算输入数组与自然数序列(1, 2, ..., n)之间的皮尔逊相关系数。
    这个函数可以用来判断一个序列的趋势性，如果返回值接近1表示强上升趋势，接近-1表示强下降趋势。

    参数说明：
    ----------
    arr : 输入数组
        可以是以下类型之一：
        - numpy.ndarray (float64或int64类型)
        - Python列表 (float或int类型)

    返回值：
    -------
    float
        输入数组与自然数序列的皮尔逊相关系数。
        如果输入数组为空或方差为零，则返回0.0。
    """
    ...

def trend_fast(arr: NDArray[np.float64]) -> float:
    """这是trend函数的高性能版本，专门用于处理numpy.ndarray类型的float64数组。
    使用了显式的SIMD指令和缓存优化处理，比普通版本更快。

    参数说明：
    ----------
    arr : numpy.ndarray
        输入数组，必须是float64类型

    返回值：
    -------
    float
        输入数组与自然数序列的皮尔逊相关系数
    """
    ...

def identify_segments(arr: NDArray[np.float64]) -> NDArray[np.int32]:
    """识别数组中的连续相等值段，并为每个段分配唯一标识符。
    每个连续相等的值构成一个段，第一个段标识符为1，第二个为2，以此类推。

    参数说明：
    ----------
    arr : numpy.ndarray
        输入数组，类型为float64

    返回值：
    -------
    numpy.ndarray
        与输入数组等长的整数数组，每个元素表示该位置所属段的标识符
    """
    ...

def find_max_range_product(arr: List[float]) -> tuple[int, int, float]:
    """在数组中找到一对索引(x, y)，使得min(arr[x], arr[y]) * |x-y|的值最大。
    这个函数可以用来找到数组中距离最远的两个元素，同时考虑它们的最小值。

    参数说明：
    ----------
    arr : List[float]
        输入数组

    返回值：
    -------
    tuple
        返回一个元组(x, y, max_product)，其中x和y是使得乘积最大的索引对，max_product是最大乘积
    """
    ...

def ols(x: NDArray[np.float64], y: NDArray[np.float64], calculate_r2: bool = True) -> NDArray[np.float64]:
    """执行普通最小二乘法(OLS)回归分析。
    
    参数说明：
    ----------
    x : numpy.ndarray
        自变量数组，shape为(n,)或(n, m)
    y : numpy.ndarray  
        因变量数组，shape为(n,)
    calculate_r2 : bool
        是否计算R²值，默认True
        
    返回值：
    -------
    numpy.ndarray
        回归结果数组，包含[截距, 斜率, R²]或[截距, 斜率]
    """
    ...

def ols_predict(x: NDArray[np.float64], y: NDArray[np.float64], x_pred: NDArray[np.float64]) -> NDArray[np.float64]:
    """基于OLS回归模型进行预测。
    
    参数说明：
    ----------
    x : numpy.ndarray
        训练数据的自变量
    y : numpy.ndarray
        训练数据的因变量  
    x_pred : numpy.ndarray
        用于预测的自变量值
        
    返回值：
    -------
    numpy.ndarray
        预测值数组
    """
    ...

def ols_residuals(x: NDArray[np.float64], y: NDArray[np.float64]) -> NDArray[np.float64]:
    """计算OLS回归的残差。
    
    参数说明：
    ----------
    x : numpy.ndarray
        自变量数组
    y : numpy.ndarray
        因变量数组
        
    返回值：
    -------
    numpy.ndarray
        残差数组
    """
    ...

def max_range_loop(s: List[float], allow_equal: bool = False) -> List[int]:
    """找到数组中所有局部最大值的索引。
    
    参数说明：
    ----------
    s : List[float]
        输入数组
    allow_equal : bool
        是否允许相等值被认为是峰值
        
    返回值：
    -------
    List[int]
        局部最大值的索引列表
    """
    ...

def min_range_loop(s: List[float], allow_equal: bool = False) -> List[int]:
    """找到数组中所有局部最小值的索引。
    
    参数说明：
    ----------
    s : List[float]
        输入数组
    allow_equal : bool
        是否允许相等值被认为是谷值
        
    返回值：
    -------
    List[int]
        局部最小值的索引列表
    """
    ...

def rolling_volatility(arr: List[float], window: int) -> List[float]:
    """计算滚动波动率。
    
    参数说明：
    ----------
    arr : List[float]
        输入时间序列
    window : int
        滚动窗口大小
        
    返回值：
    -------
    List[float]
        滚动波动率序列
    """
    ...

def rolling_cv(arr: List[float], window: int) -> List[float]:
    """计算滚动变异系数。
    
    参数说明：
    ----------
    arr : List[float]
        输入时间序列
    window : int
        滚动窗口大小
        
    返回值：
    -------
    List[float]
        滚动变异系数序列
    """
    ...

def rolling_qcv(arr: List[float], window: int) -> List[float]:
    """计算滚动四分位变异系数。
    
    参数说明：
    ----------
    arr : List[float]
        输入时间序列
    window : int
        滚动窗口大小
        
    返回值：
    -------
    List[float]
        滚动四分位变异系数序列
    """
    ...

def compute_max_eigenvalue(matrix: NDArray[np.float64]) -> float:
    """计算矩阵的最大特征值。
    
    参数说明：
    ----------
    matrix : numpy.ndarray
        输入矩阵
        
    返回值：
    -------
    float
        最大特征值
    """
    ...

def sum_as_string(a: int, b: int) -> str:
    """将两个整数相加并返回字符串结果。
    
    参数说明：
    ----------
    a : int
        第一个整数
    b : int
        第二个整数
        
    返回值：
    -------
    str
        相加结果的字符串表示
    """
    ...

def test_simple_function() -> int:
    """简单的测试函数，返回固定值42
    
    用于验证构建和导出是否正常工作。
    
    返回值：
    -------
    int
        固定返回值42
    """
    ...