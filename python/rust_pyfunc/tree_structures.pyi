"""树结构相关类型声明"""
from typing import Dict, Union, Tuple, List
import numpy as np
import pandas as pd
from numpy.typing import NDArray

class PriceTree:
    """价格树结构，用于分析价格序列的层次关系和分布特征。
    
    这是一个二叉树结构，每个节点代表一个价格水平，包含该价格的成交量和时间信息。
    树的构建基于价格的大小关系，支持快速的价格查找和区间统计。
    """
    
    def __init__(self) -> None:
        """初始化一个空的价格树。"""
        ...
    
    def build_tree(
        self,
        times: NDArray[np.int64],
        prices: NDArray[np.float64],
        volumes: NDArray[np.float64]
    ) -> None:
        """根据时间序列、价格序列和成交量序列构建价格树。

        参数说明：
        ----------
        times : numpy.ndarray
            时间戳序列，Unix时间戳格式
        prices : numpy.ndarray
            价格序列
        volumes : numpy.ndarray
            成交量序列

        注意：
        -----
        三个数组的长度必须相同，且按时间顺序排列。
        """
        ...
    
    def query_price_range(
        self, 
        min_price: float, 
        max_price: float
    ) -> List[Tuple[float, float, int]]:
        """查询指定价格范围内的所有节点信息。

        参数说明：
        ----------
        min_price : float
            最小价格（包含）
        max_price : float
            最大价格（包含）

        返回值：
        -------
        List[Tuple[float, float, int]]
            返回列表，每个元素是(价格, 总成交量, 最早时间)的元组
        """
        ...
    
    def get_volume_at_price(self, price: float) -> float:
        """获取指定价格的总成交量。

        参数说明：
        ----------
        price : float
            查询价格

        返回值：
        -------
        float
            该价格的总成交量，如果价格不存在则返回0.0
        """
        ...
    
    def get_price_levels(self) -> List[float]:
        """获取所有价格水平。

        返回值：
        -------
        List[float]
            按升序排列的所有价格水平列表
        """
        ...
    
    @property
    def height(self) -> int:
        """获取树的高度"""
        ...

    @property
    def node_count(self) -> int:
        """获取节点总数"""
        ...

    @property
    def asl(self) -> float:
        """获取平均查找长度(ASL)"""
        ...

    @property
    def wpl(self) -> float:
        """获取加权路径长度(WPL)"""
        ...

    @property
    def diameter(self) -> int:
        """获取树的直径"""
        ...

    @property
    def total_volume(self) -> float:
        """获取总成交量"""
        ...

    @property
    def avg_volume_per_node(self) -> float:
        """获取每个节点的平均成交量"""
        ...

    @property
    def price_range(self) -> Tuple[float, float]:
        """获取价格范围"""
        ...

    @property
    def time_range(self) -> Tuple[int, int]:
        """获取时间范围"""
        ...

    def get_all_features(self) -> Dict[str, Dict[str, Union[float, int]]]:
        """获取所有树的特征。

        返回值：
        -------
        Dict[str, Dict[str, Union[float, int]]]
            包含树的各种统计特征的字典，包括：
            - structure: 树结构特征（高度、节点数、直径等）
            - performance: 性能特征（平均查找长度、加权路径长度等）
            - volume: 成交量特征（总量、平均量等）
            - price: 价格特征（范围、分布等）
            - time: 时间特征（范围、分布等）
        """
        ...

class RollingFutureAccessor:
    """滚动未来数据访问器。
    
    这是一个Pandas accessor类，用于为DataFrame和Series提供滚动未来窗口计算功能。
    通过.rolling_future属性访问，支持向后查看的滚动窗口统计分析。
    """
    
    def __init__(self, pandas_obj) -> None:
        """初始化滚动未来访问器。
        
        参数说明：
        ----------
        pandas_obj : pandas.DataFrame 或 pandas.Series
            要进行滚动计算的pandas对象
        """
        ...
    
    def __call__(self, window: int, include_current: bool = False) -> "RollingFutureAccessor":
        """设置滚动窗口的大小。
        
        参数说明：
        ----------
        window : int
            滚动窗口大小
        include_current : bool, 默认False
            是否包含当前行
            
        返回值：
        -------
        RollingFutureAccessor
            配置了窗口大小的访问器对象
        """
        ...
    
    def mean(self) -> Union[pd.Series, pd.DataFrame]:
        """计算向后滚动窗口的均值。
        
        返回值：
        -------
        pandas.Series 或 pandas.DataFrame
            滚动均值结果
        """
        ...
    
    def sum(self) -> Union[pd.Series, pd.DataFrame]:
        """计算向后滚动窗口的总和。
        
        返回值：
        -------
        pandas.Series 或 pandas.DataFrame
            滚动总和结果
        """
        ...
    
    def max(self) -> Union[pd.Series, pd.DataFrame]:
        """计算向后滚动窗口的最大值。
        
        返回值：
        -------
        pandas.Series 或 pandas.DataFrame
            滚动最大值结果
        """
        ...
    
    def min(self) -> Union[pd.Series, pd.DataFrame]:
        """计算向后滚动窗口的最小值。
        
        返回值：
        -------
        pandas.Series 或 pandas.DataFrame
            滚动最小值结果
        """
        ...
    
    def std(self) -> Union[pd.Series, pd.DataFrame]:
        """计算向后滚动窗口的标准差。
        
        返回值：
        -------
        pandas.Series 或 pandas.DataFrame
            滚动标准差结果
        """
        ...
    
    def median(self) -> Union[pd.Series, pd.DataFrame]:
        """计算向后滚动窗口的中位数。
        
        返回值：
        -------
        pandas.Series 或 pandas.DataFrame
            滚动中位数结果
        """
        ...
    
    def count(self) -> Union[pd.Series, pd.DataFrame]:
        """计算向后滚动窗口内的数据点数量。
        
        返回值：
        -------
        pandas.Series 或 pandas.DataFrame
            滚动计数结果
        """
        ...
    
    def rank(self) -> Union[pd.Series, pd.DataFrame]:
        """计算当前值在后面窗口内的分位数（0到1之间）。
        
        返回值：
        -------
        pandas.Series 或 pandas.DataFrame
            分位数排名结果
        """
        ...
    
    def skew(self) -> Union[pd.Series, pd.DataFrame]:
        """计算向后滚动窗口的偏度。
        
        返回值：
        -------
        pandas.Series 或 pandas.DataFrame
            滚动偏度结果
        """
        ...
    
    def trend_time(self) -> Union[pd.Series, pd.DataFrame]:
        """计算向后滚动窗口内数据序列与时间序列的相关系数。
        
        返回值：
        -------
        pandas.Series 或 pandas.DataFrame
            时间趋势相关系数结果
        """
        ...
    
    def trend_oneton(self) -> Union[pd.Series, pd.DataFrame]:
        """计算向后滚动窗口内数据序列与1到n序列的相关系数（忽略时间间隔）。
        
        返回值：
        -------
        pandas.Series 或 pandas.DataFrame
            1到n趋势相关系数结果
        """
        ...
    
    def last(self) -> Union[pd.Series, pd.DataFrame]:
        """计算向后滚动窗口内的最后一个值。
        
        返回值：
        -------
        pandas.Series 或 pandas.DataFrame
            滚动窗口最后值结果
        """
        ...

class PriceTreeViz:
    """价格树可视化类。
    
    用于构建和可视化价格数据的树结构，支持在Jupyter Notebook中展示。
    """
    
    def __init__(self) -> None:
        """构造函数，初始化价格树可视化对象。"""
        ...
    
    def build_tree(self, times: List[int], prices: List[float], volumes: List[float]) -> None:
        """构建价格树。
        
        参数说明：
        ----------
        times : List[int]
            时间戳列表
        prices : List[float]
            价格列表
        volumes : List[float]
            成交量列表
        """
        ...
    
    def get_tree_structure(self) -> Dict:
        """获取树结构。
        
        返回值：
        -------
        Dict
            树结构信息字典
        """
        ...
    
    def visualize(self) -> None:
        """在Jupyter Notebook中可视化价格树结构。"""
        ...
    
    def display_tree_stats(self) -> None:
        """显示价格树统计数据。"""
        ...

def haha() -> str:
    """测试函数"""
    ...