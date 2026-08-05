# -*- coding: utf-8 -*-
"""multi_factor_capm 横截面因子测试。

1. 常量与命名测试：N_FACTORS=49791、名称唯一、53 组合起点对齐
2. 确定性测试：同一天两次运行输出完全一致
3. （可选）数值对比：与 sandbox Python 参考实现对比 T1-y11 组合 39 列 mean
   需环境变量 MF_GRID_DIR 指向 sandbox 导出的网格目录（缺省跳过）
"""
import os
import numpy as np
import rust_pyfunc as rp

DATE = 20260717
N_FACTORS = 49791

# 53 组合起点（每组合 21 统计 × (15+8K) 列）
COMBO_STARTS = []
acc = 0
for name, k, n_y in [("T1", 3, 10), ("T2", 3, 12), ("T3", 3, 12), ("F1", 5, 10), ("F2", 5, 9)]:
    for _ in range(n_y):
        COMBO_STARTS.append(acc)
        acc += (15 + 8 * k) * 21
assert acc == N_FACTORS


def test_names_and_layout():
    names = rp.py_multi_factor_capm_names()
    assert len(names) == N_FACTORS
    assert len(set(names)) == N_FACTORS, "因子名必须唯一"
    # 每个组合起点对应预期前缀
    expect_prefixes = []
    for m, k, ys in [
        ("T1", 3, [0, 1, 2, 3, 5, 6, 8, 9, 11, 12]),
        ("T2", 3, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]),
        ("T3", 3, [0, 1, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13]),
        ("F1", 5, [0, 1, 2, 3, 5, 6, 8, 11, 12, 13]),
        ("F2", 5, [0, 1, 2, 3, 6, 8, 9, 11, 12]),
    ]:
        base = [f"active_buy_volume_ratio", "order_gap_signed_vw", "observable_ratio_level",
                "book_imbalance10_level", "observable_ratio_innovation", "book_imbalance10_innovation",
                "spread_bps", "near3_depth_share", "microprice_pressure_bps", "order_gap_magnitude",
                "large_trade_direction_v2", "price_log_return_3s", "log_volume_3s", "trade_arrival_clustering"]
        for y in ys:
            expect_prefixes.append(f"mfcapm_{m}_{base[y]}_")
    for i, pref in enumerate(expect_prefixes):
        assert names[COMBO_STARTS[i]].startswith(pref), f"组合{i} 起点不对: {names[COMBO_STARTS[i]]}"


def test_determinism():
    codes1, vals1 = rp.py_multi_factor_capm(DATE)
    codes2, vals2 = rp.py_multi_factor_capm(DATE)
    assert codes1 == codes2
    assert len(vals1) == len(codes1) * N_FACTORS
    v1 = np.array(vals1)
    v2 = np.array(vals2)
    both = np.isfinite(v1) & np.isfinite(v2)
    assert np.allclose(v1[both], v2[both], atol=0.0, rtol=0.0), "两次运行结果不一致（非确定性）"
    assert (np.isnan(v1) == np.isnan(v2)).all()


def test_basic_stats():
    codes, vals = rp.py_multi_factor_capm(DATE)
    assert len(codes) > 3000
    v = np.array(vals).reshape(len(codes), N_FACTORS)
    nan_ratio = np.isnan(v).mean(axis=0)
    # per-stock 列（beta/残差等）大部分股票应有值
    per_stock_nan = nan_ratio[: 34 * 819 + 19 * 1155]  # 近似：per-stock 在前
    assert per_stock_nan.mean() < 0.5, "per-stock 列 NaN 过多"


def test_against_py_ref():
    """与 sandbox Python 参考实现对比 T1-y11 组合 39 列 mean（需网格导出文件）。"""
    grid_dir = os.environ.get("MF_GRID_DIR")
    if not grid_dir:
        print("跳过数值对比（未设置 MF_GRID_DIR）")
        return
    import csv
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sandbox_multi_factor"))
    # 参考实现：读网格二进制，仅计算 T1-y11 组合（复用探索期验证过的逻辑，简化版只输出列均值）
    # 完整对比脚本见 sandbox_multi_factor/py_ref_T1_y11.py；此处只做端到端抽查：
    # 主项目输出中 T1-y11 组合的 39 列 mean 应与参考一致（误差 < 1e-4）
    codes, vals = rp.py_multi_factor_capm(DATE)
    names = rp.py_multi_factor_capm_names()
    start = 7 * 819  # T1-y11 = T1 第 8 个组合（ys 索引 7）
    col_idxs = [i for i in range(start, start + 39 * 21) if names[i].endswith("_mean")]
    assert len(col_idxs) == 39
    # 参考文件（探索期生成）：sandbox_multi_factor/py_ref_T1_y11_means.npy（旧布局）需重排
    ref_path = os.path.join(os.path.dirname(__file__), "..", "sandbox_multi_factor", "py_ref_T1_y11_means.npy")
    if not os.path.exists(ref_path):
        print("跳过数值对比（缺参考文件）")
        return
    py = np.load(ref_path)
    new_cols = []
    for c in range(39):
        if c < 6:
            new_cols.append(c)
        elif c < 12:
            new_cols.append(c - 6 + 17)
        elif c < 15:
            new_cols.append(c - 12 + 27)
        elif c < 18:
            new_cols.append(c - 15 + 36)
        elif c == 18:
            new_cols.append(6)
        elif c == 19:
            new_cols.append(7)
        elif c == 20:
            new_cols.append(8)
        elif c < 24:
            new_cols.append(c - 21 + 9)
        elif c == 24:
            new_cols.append(12)
        elif c < 28:
            new_cols.append(c - 25 + 13)
        elif c == 28:
            new_cols.append(16)
        elif c < 33:
            new_cols.append(c - 29 + 23)
        elif c < 36:
            new_cols.append(c - 33 + 30)
        else:
            new_cols.append(c - 36 + 33)
    py_new = py[new_cols]
    v = np.array(vals).reshape(len(codes), N_FACTORS)
    code_idx = {c: i for i, c in enumerate(codes)}
    rng = np.random.RandomState(0)
    sample = rng.choice(py_new.shape[1], 200, replace=False)
    for ci, idx in enumerate(col_idxs):
        a = v[sample, idx]  # 与参考列同序（sample 为参考的股票索引，与主项目 codes 顺序需一致）
        b = py_new[ci, sample]
        both = np.isfinite(a) & np.isfinite(b)
        if both.sum() > 50:
            assert np.nanmax(np.abs(a[both] - b[both])) < 1e-4, f"列{ci} 与参考不一致"


if __name__ == "__main__":
    test_names_and_layout()
    print("✓ 命名与布局")
    test_basic_stats()
    print("✓ 基本统计")
    test_against_py_ref()
    print("✓ 数值对比（如启用）")
    test_determinism()
    print("✓ 确定性（两次运行约 2 分钟）")
