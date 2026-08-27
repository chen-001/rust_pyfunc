"""组合 store + 判定闭环的轻量冒烟测试（不依赖生产 /hdd store）。

运行: python -m pytest tests/test_supplement_pipeline_smoke.py -q
"""
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

rp = pytest.importorskip("rust_pyfunc")


def _patch_idx_factor_name(store_dir: Path, old: str, new: str) -> None:
    idx = store_dir / "factors.idx"
    data = bytearray(idx.read_bytes())
    pos = data.find(old.encode("utf-8"))
    assert pos >= 0, f"{old} not found in {idx}"
    assert len(old) == len(new)
    data[pos : pos + len(new)] = new.encode("utf-8")
    idx.write_bytes(data)


def test_combined_store_and_scatter_fast():
    root = Path(tempfile.mkdtemp(prefix="supplement_smoke_"))
    base = root / "base"
    supp = root / "supp"
    base.mkdir()
    supp.mkdir()
    assert rp.factor_store_v5_smoke_proj(str(base), 24, 2)[0]
    assert rp.factor_store_v5_smoke_proj(str(supp), 24, 1)[0]
    # smoke_proj 统一命名为 f0..fn；把 supp 的唯一因子改名为 g0，避免重名冲突。
    _patch_idx_factor_name(supp, "f0", "g0")

    reg = rp.factor_store_v5_register_group(str(root), "base", "base")
    assert reg["factor_count"] == 2
    reg = rp.factor_store_v5_register_group(str(root), "supp", "supp")
    assert reg["factor_count"] == 3

    info = rp.factor_store_v5_info(str(root))
    assert info["factor_names"] == ["f0", "f1", "g0"]
    assert info["is_projected"] is True

    for idx in range(3):
        max_diff, _, _ = rp.factor_store_v5_verify_scatter_fast(str(root), idx)
        assert max_diff == 0.0

    # 模板轴不一致的 group 应拒绝注册
    bad = root / "bad"
    bad.mkdir()
    rp.factor_store_v5_smoke_proj(str(bad), 25, 1)
    _patch_idx_factor_name(bad, "f0", "h0")
    with pytest.raises(ValueError, match="模板轴与已有组合不一致"):
        rp.factor_store_v5_register_group(str(root), "bad", "bad")


def test_tail_engine_signature_has_metrics_only():
    sig = getattr(rp.tail_backtest_engine, "__text_signature__", "") or ""
    assert "save_all_metrics" in sig


def test_evaluator_requires_expected_names():
    dw = pytest.importorskip("design_whatever")
    with pytest.raises(TypeError):
        dw.evaluate_supplement_factors(base_ver="x", supplement_ver="y")
