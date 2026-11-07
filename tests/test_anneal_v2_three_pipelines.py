"""柜员游戏三个修正版的真实 pipeline 入口冒烟测试（需先 ./alter.sh）。

检查：直写列段的记录数、列数与名字一致；抽查列与单股 Python 入口逐位相同（per-stock 两个）；
横截面版只检查记录数与列数。临时目录放 HDD，结束后删除。
用法：python tests/test_anneal_v2_three_pipelines.py
"""
import shutil
import time
from pathlib import Path

import numpy as np
import rust_pyfunc as rp

ROOT = Path("/hdd/user_home_unsafe/chenzongwei/tmp_test_anneal_v2_three")
DATES, CODES = [20220819, 20241231], ["000001", "600519", "300750"]
TD = rp.td.trading_days.tolist()


def check_stock(pipeline, names, py_func):
    store = ROOT / pipeline
    t = time.time()
    rp.run_factor_pipeline(
        pipeline=pipeline, tasks=[[d, c] for d in DATES for c in CODES], n_jobs=6, backup_file="",
        expected_result_length=len(names), trading_days=TD, params=None, update_mode=True,
        bind_cores=False, mode="multiprocess", store_dir=str(store), store_factor_names=names,
        direct_write=True,
    )
    info = rp.factor_store_v5_info(str(store))
    assert info["record_count"] == len(DATES) * len(CODES), info
    assert info.get("storage_layout") == "direct_segments", info
    # 抽查若干列：store 里的值与 Python 入口逐位相同
    want = {(d, c): np.array(py_func(c, d), dtype=np.float32) for d in DATES for c in CODES}
    for col in [0, len(names) // 3, len(names) // 2, len(names) - 1]:
        vals = rp.factor_store_v5_read_factor(str(store), col)["factor"]
        assert len(vals) == len(DATES) * len(CODES)
        got = sorted(np.asarray(vals, dtype=np.float32).view(np.uint32).tolist())
        ref = sorted(np.array([want[k][col] for k in want], dtype=np.float32).view(np.uint32).tolist())
        assert got == ref, (pipeline, col)
    print(f"{pipeline}: 记录 {info['record_count']}，列 {len(names)}，抽查列与 Python 入口逐位相同，耗时 {time.time() - t:.1f}s")


def check_cross(pipeline, names):
    store = ROOT / pipeline
    t = time.time()
    rp.run_factor_pipeline_cross_section(
        pipeline=pipeline, tasks=[DATES[1]], n_jobs=100, expected_result_length=len(names),
        trading_days=TD, n_workers=2, update_mode=True, bind_cores=False, store_dir=str(store),
        store_factor_names=names, direct_write=True,
    )
    info = rp.factor_store_v5_info(str(store))
    assert info["record_count"] > 4000, info
    # 读回接口不返回空值行，所以按「读回行数 ≤ 记录数」并且整体有值来检查
    counts = [len(rp.factor_store_v5_read_factor(str(store), c)["factor"]) for c in (0, len(names) // 2, len(names) - 1)]
    print("抽查列读回行数", counts, "记录数", info["record_count"])
    assert all(0 < n <= info["record_count"] for n in counts), counts
    print(f"{pipeline}: 记录 {info['record_count']}，列 {len(names)}，耗时 {time.time() - t:.1f}s")


if __name__ == "__main__":
    shutil.rmtree(ROOT, ignore_errors=True)
    ROOT.mkdir(parents=True)
    try:
        check_stock("anneal_regime_excess", rp.py_anneal_regime_excess_names(), rp.py_anneal_regime_excess)
        check_stock("anneal_order_profile_v2", rp.py_anneal_order_profile_v2_names(), rp.py_anneal_order_profile_v2)
        check_cross("anneal_cross_peer_v2", rp.py_anneal_cross_peer_v2_names())
    finally:
        shutil.rmtree(ROOT, ignore_errors=True)
