"""树结构相关类型声明"""
from typing import Dict, Union, Tuple, List
import numpy as np
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
    
    用于在滚动窗口中访问未来数据点，支持高效的时间序列前瞻分析。
    """
    
    def __init__(self, data: NDArray[np.float64], window_size: int) -> None:
        """初始化滚动未来访问器。
        
        参数说明：
        ----------
        data : NDArray[np.float64]
            时间序列数据
        window_size : int
            滚动窗口大小
        """
        ...
    
    def get_future_value(self, current_index: int, future_steps: int) -> float:
        """获取指定步数后的未来值。
        
        参数说明：
        ----------
        current_index : int
            当前索引位置
        future_steps : int
            未来步数
            
        返回值：
        -------
        float
            未来值，如果超出范围则返回NaN
        """
        ...
    
    def get_future_window(self, current_index: int, window_length: int) -> NDArray[np.float64]:
        """获取未来窗口的数据。
        
        参数说明：
        ----------
        current_index : int
            当前索引位置
        window_length : int
            未来窗口长度
            
        返回值：
        -------
        NDArray[np.float64]
            未来窗口数据数组
        """
        ...
    
    def rolling_future_mean(self, future_steps: int) -> NDArray[np.float64]:
        """计算滚动未来均值。
        
        参数说明：
        ----------
        future_steps : int
            未来步数
            
        返回值：
        -------
        NDArray[np.float64]
            滚动未来均值数组
        """
        ...
    
    def rolling_future_std(self, future_steps: int) -> NDArray[np.float64]:
        """计算滚动未来标准差。
        
        参数说明：
        ----------
        future_steps : int
            未来步数
            
        返回值：
        -------
        NDArray[np.float64]
            滚动未来标准差数组
        """
        ...