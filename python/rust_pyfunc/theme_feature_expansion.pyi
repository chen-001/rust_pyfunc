from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray


def compute_theme_feature_expansion_from_minute(
    minute_data: NDArray[np.float64],
    k: int = 30,
    summary_dim: int = 17,
    n_threads: int = 8,
) -> Tuple[List[NDArray[np.float64]], List[NDArray[np.float64]], List[str], List[str]]:
    """
    从分钟级四维数组中直接提取主题扩展向量特征和标量特征。

    参数:
        minute_data: shape=(n_days, n_minutes, n_stocks, 34)
        k: 单日单时段聚类数
        summary_dim: 聚类特征维度，可选3、7、17
        n_threads: 并行线程数，最大10

    返回:
        (
            向量二维数组列表,
            标量一维数组列表,
            向量名字列表,
            标量名字列表,
        )
    """
    ...


def get_theme_cluster_scatter_3d(
    minute_data: NDArray[np.float64],
    k: int = 30,
    n_threads: int = 8,
) -> Tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.float64], int]:
    """
    提取3维聚类散点数据用于可视化。

    返回 (coords, labels, centers, n_clusters)
    coords: n_stocks×3, 列为 [ret, log_amt, net_flow] 原始值
    labels: n_stocks, 无效标签为-1
    centers: n_clusters×3, 聚类内均值
    n_clusters: 实际聚类数
    """
    ...
