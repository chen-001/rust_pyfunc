import os

import numpy as np
import rust_pyfunc as rp


DATE = 20220819
STALE_PREFIX_DATE = 20181018


def _codes(n: int, date: int = DATE) -> list[str]:
    path = f"/ssd_data/stock/{date}/transaction"
    return sorted(
        {
            code
            for name in os.listdir(path)
            if len(code := name.split("_")[0]) == 6 and code.isdigit()
        }
    )[:n]


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


def test_microstructure_capm_ignores_previous_day_prefix_records() -> None:
    # 20181018 的绝大多数原始文件首行混入 20181017 收盘记录。
    features = np.asarray(
        rp.py_microstructure_3s_features("000001", STALE_PREFIX_DATE),
        dtype=np.float32,
    )
    assert np.isfinite(features).sum() > 0

    names = rp.py_microstructure_capm_names()
    codes, values = rp.py_microstructure_capm_codes(
        STALE_PREFIX_DATE, _codes(35, STALE_PREFIX_DATE)
    )
    matrix = np.asarray(values, dtype=np.float32).reshape(len(codes), len(names))
    assert len(codes) >= 30
    assert np.isfinite(matrix).any()
