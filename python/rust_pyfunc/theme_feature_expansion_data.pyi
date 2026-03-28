from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray

THEME_FEATURE_EXPANSION_MINUTE_FIELDS: List[str]
THEME_FEATURE_VECTOR_DIMENSIONS: Dict[str, int]


def prepare_minute_data_for_theme_feature_expansion(
    date: int,
    lookback: int = 2,
) -> Tuple[NDArray[np.float64], List[int], NDArray[np.str_]]:
    ...


def compute_theme_feature_expansion_from_minute_raw(
    minute_data: NDArray[np.float64],
    k: int = 30,
    summary_dim: int = 17,
    n_threads: int = 8,
) -> Tuple[List[NDArray[np.float64]], List[NDArray[np.float64]], List[str], List[str]]:
    ...


def compute_theme_feature_expansion_from_date(
    date: int,
    lookback: int = 2,
    k: int = 30,
    summary_dim: int = 17,
    n_threads: int = 8,
) -> Tuple[List[pd.DataFrame], List[pd.Series], List[str], List[str], NDArray[np.str_], List[int]]:
    ...
