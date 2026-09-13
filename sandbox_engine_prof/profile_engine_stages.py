"""引擎阶段画像：真实 store 少量因子，捕获 PROF 分阶段耗时 + 端到端时间。

只读跑法：不动主项目代码；ver/temp_root 独立，不污染生产缓存。
"""
import sys, time
sys.path.insert(0, "/home/chenzongwei/design_whatever")
import rust_pyfunc as rp
from design_whatever.tail_v4 import run_tail_pipeline_engine

STORE = "/hdd/user_home_unsafe/chenzongwei/factor_store_cross_yhyb"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 48

info = rp.factor_store_v5_info(STORE)
names = info["factor_names"][:N]
print(f"factors={N} total_in_store={len(info['factor_names'])}", flush=True)

t0 = time.time()
res = run_tail_pipeline_engine(
    colblk_store_dir=STORE,
    ver="perfprof_v1",
    names=names,
    windows=[5, 10, 20],
    fold=True,
    temp_root="/tmp/perfprof_v1_tail_v4",
    n_jobs=N,
    start_date="2015-01-01",
    backtest_start_date="2015-02-01",
    end_date="2026-07-17",
    selection_kwargs={"cover_rate": 0.97, "ret_point_neu_gap5": 0.055},
    majority_count_threshold=10000.0,
    zero_max_threshold=0.12,
    nan_max_threshold=0.04,
    metrics_only=False,
    return_run_info=True,
)
dt_total = time.time() - t0
gap5, gap1, source, run_info = res
print(f"WALL_TOTAL={dt_total:.1f}s processed={run_info['processed_sources']} restored={run_info['restored_sources']}", flush=True)
print(f"candidate_counts={run_info['candidate_counts']}", flush=True)
