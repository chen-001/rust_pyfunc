import os

import numpy as np
import rust_pyfunc as rp


DATE = 20220819


def _codes(n: int) -> list[str]:
    path = f"/ssd_data/stock/{DATE}/transaction"
    return sorted({name.split("_")[0] for name in os.listdir(path)})[:n]


def test_microstructure_3s_features_shape_and_ranges() -> None:
    names = rp.py_microstructure_3s_feature_names()
    values = np.asarray(rp.py_microstructure_3s_features("000001", DATE)).reshape(-1, len(names))
    assert values.shape == (4740, 12)
    for name in [
        "active_buy_volume_ratio",
        "observable_ratio_level",
        "near3_depth_share",
    ]:
        finite = values[np.isfinite(values[:, names.index(name)]), names.index(name)]
        assert finite.size > 0
        assert np.all((finite >= 0.0) & (finite <= 1.0))
    for name in ["book_imbalance10_level", "large_trade_direction_v2"]:
        finite = values[np.isfinite(values[:, names.index(name)]), names.index(name)]
        assert finite.size > 0
        assert np.all((finite >= -1.0) & (finite <= 1.0))


def test_microstructure_capm_reduced_output_contract() -> None:
    names = rp.py_microstructure_capm_names()
    assert len(names) == 9660
    assert len(set(names)) == len(names)
    codes, values = rp.py_microstructure_capm_codes(DATE, _codes(35))
    matrix = np.asarray(values, dtype=np.float32).reshape(len(codes), len(names))
    assert len(codes) >= 30
    assert matrix.shape == (len(codes), 9660)
    # 小样本刚超过横截面 30 股门槛，部分稀疏指标会因有效股票不足而保留 NaN。
    assert np.isfinite(matrix).mean() > 0.40
