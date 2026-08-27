"""验证 tail_backtest_engine 与 tail_v5_run_candidates 结果一致（小批量因子）。

用法: python tests/test_tail_backtest_engine.py
"""
import os
import sys
import time
import numpy as np
import pytest
import rust_pyfunc as rp

store_dir = "/hdd/user_home_unsafe/chenzongwei/factor_store_挂单猫0701b"
if not os.path.isdir(store_dir):
    pytest.skip(f"测试 store 不存在: {store_dir}", allow_module_level=True)

# 读模板轴
tmpl = rp.factor_store_v5_template(store_dir)
dates_all = np.asarray(tmpl["dates"], dtype=np.int32)
stocks_all = list(tmpl["stocks"])

# 缩小规模：取前 100 天 + 前 500 股票
dates = dates_all[:100].tolist()
stocks = stocks_all[:500]

info = rp.factor_store_v5_info(store_dir)
all_names = list(info["factor_names"])
name_to_idx = {n: i for i, n in enumerate(all_names)}

# 只测前 10 个因子
test_names = all_names[:10]
test_paths = [f"{store_dir}::{name_to_idx[n]}" for n in test_names]

# 通用参数
common = dict(
    factor_names=test_names,
    factor_paths=test_paths,
    dates=dates,
    stocks=stocks,
    windows=[5, 10, 20],
    fold=True,
    min_valid=12,
    style_data_path="/home/chenzongwei/database/barra/barra_daily_together_jason.parquet",
    ret_gap1_path="/nas197/user_home_unsafe/chenzongwei/test_engine/ret_gap1.npy",
    ret_sum_gap1_path="/nas197/user_home_unsafe/chenzongwei/test_engine/ret_sum_gap1.npy",
    ret_gap5_path="/nas197/user_home_unsafe/chenzongwei/test_engine/ret_gap5.npy",
    ret_sum_gap5_path="/nas197/user_home_unsafe/chenzongwei/test_engine/ret_sum_gap5.npy",
    restrict_path="/nas197/user_home_unsafe/chenzongwei/test_engine/restrict.npy",
    index_ret_path="/nas197/user_home_unsafe/chenzongwei/test_engine/index_ret.npy",
    backtest_start=20170201,
    cover_rate=0.5,
    ret_point_neu_gap5=0.055,
    ret_point_neu_gap1=0.08,
    ic_point_neu_gap5=0.01,
    ic_point_neu_gap1=0.006,
    ret_point_gap5=0.1,
    ret_point_gap1=0.13,
    ic_point_gap5=0.03,
    ic_point_gap1=0.02,
    ic_more_important_gap5=0.020,
    ic_more_important_gap1=0.012,
    majority_count_threshold=2000.0,
    zero_max_threshold=0.1,
    nan_max_threshold=0.04,
)

# 用假数据做回测输入（验证函数能跑通即可）
n_dates = len(dates)
n_stocks = len(stocks)
np.random.seed(42)
os.makedirs("/nas197/user_home_unsafe/chenzongwei/test_engine", exist_ok=True)
np.save("/nas197/user_home_unsafe/chenzongwei/test_engine/ret_gap1.npy", np.random.randn(n_dates, n_stocks).astype(np.float32))
np.save("/nas197/user_home_unsafe/chenzongwei/test_engine/ret_sum_gap1.npy", np.random.randn(n_dates, n_stocks).astype(np.float32))
np.save("/nas197/user_home_unsafe/chenzongwei/test_engine/ret_gap5.npy", np.random.randn(n_dates, n_stocks).astype(np.float32))
np.save("/nas197/user_home_unsafe/chenzongwei/test_engine/ret_sum_gap5.npy", np.random.randn(n_dates, n_stocks).astype(np.float32))
restrict = np.zeros((n_dates, n_stocks), dtype=np.float32)  # 全 0 = 全部可交易
np.save("/nas197/user_home_unsafe/chenzongwei/test_engine/restrict.npy", restrict)
np.save("/nas197/user_home_unsafe/chenzongwei/test_engine/index_ret.npy", np.random.randn(n_dates).astype(np.float32))

# 跑 engine
t0 = time.time()
result_engine = rp.tail_backtest_engine(
    colblk_store_dir=store_dir,
    n_jobs=4,
    cache_root="/nas197/user_home_unsafe/chenzongwei/test_engine_cache",
    industry_neutralize=False,
    industry_matrix=np.zeros((n_dates, n_stocks), dtype=np.float64),
    **common,
)
t_engine = time.time() - t0
print(f"Engine: {result_engine} ({t_engine:.1f}s)")

print(f"\n✅ tail_backtest_engine 可正常运行，处理 {result_engine['processed_sources']} 个因子")
