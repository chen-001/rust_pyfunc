from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray


def compute_theme_feature_expansion_from_minute(
    minute_data: NDArray[np.float64],
    k: int = 30,
    history_lookback: int = 5,
    n_threads: int = 8,
) -> Tuple[List[NDArray[np.float64]], List[NDArray[np.float64]], List[str], List[str]]:
    """
    从分钟级四维数组中直接提取主题扩展向量特征和标量特征。

    参数:
        minute_data: shape=(n_days, n_minutes, n_stocks, 34)
        k: 单日单时段聚类数
        history_lookback: 历史聚合长度
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
