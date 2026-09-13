"""A/B 引擎对照：在 urgency store 上跑 N 个因子（与生产任务 194 同配置口径）。

用法: python ab_engine.py <ver> <N>
产物: temp_root=/tmp/{ver}_tail_v4 ；任务完成后把 task_results 快照到 /tmp/{ver}_task_results
"""
import sys, time, shutil, os
sys.path.insert(0, "/home/chenzongwei/design_whatever")
import json
import rust_pyfunc as rp
from design_whatever.tail_v4 import run_tail_pipeline_engine

ver = sys.argv[1]
N = int(sys.argv[2])
n_jobs = int(sys.argv[3]) if len(sys.argv) > 3 else N
STORE = "/hdd/user_home_unsafe/chenzongwei/factor_store_urgency_v1"
TMP = f"/tmp/{ver}_tail_v4"

info = rp.factor_store_v5_info(STORE)
names = info["factor_names"][:N]
print(f"ab {ver}: factors={N} n_jobs={n_jobs} TAIL_BATCH_B={os.environ.get('TAIL_BATCH_B','4')}", flush=True)

t0 = time.time()
res = run_tail_pipeline_engine(
    colblk_store_dir=STORE,
    ver=ver,
    names=names,
    windows=[5, 10, 20],
    fold=True,
    temp_root=TMP,
    n_jobs=n_jobs,
    start_date="2015-01-05",
    backtest_start_date="2015-02-01",
    end_date="2026-05-22",
    selection_kwargs={"cover_rate": 0.5, "ret_point_neu_gap5": 0.055},
    majority_count_threshold=10000.0,
    zero_max_threshold=0.1,
    nan_max_threshold=0.04,
    min_valid=12,
    index_name="000905",
    industry_neutralize=True,
    style_data_path="/home/chenzongwei/database/barra/barra_daily_together_jason.parquet",
    industry_data_path="/nas197/binary/stock/sz_alpha/csv/vars/Base/SW_IND_CODE.csv",
    metrics_only=True,
    return_run_info=True,
)
dt = time.time() - t0
gap5, gap1, source, run_info = res
print(f"WALL={dt:.1f}s processed={run_info['processed_sources']} cands={run_info['candidate_counts']}", flush=True)
print(f"selected gap5={len(gap5)} gap1={len(gap1)} source={len(source)}", flush=True)
snap = f"/tmp/{ver}_task_results"
if os.path.exists(snap):
    shutil.rmtree(snap)
shutil.copytree(f"{TMP}/task_results", snap)
print(f"snapshotted -> {snap}", flush=True)
print("AB_DONE", flush=True)
