"""cross_yhyb 新版引擎基线重跑（cross_yhyb_v2）。

目的：让初版因子的 selection audit / 纯 neu_IC 下限与补充因子使用同一引擎逻辑。
本脚本只做 candidate screening + audit，不做 fulltest/doc/gallery。
"""
import os
import design_whatever as dw

ver = "cross_yhyb_v2"
colblk_store_dir = "/hdd/user_home_unsafe/chenzongwei/factor_store_cross_yhyb"
temp_root = "/hdd/user_home_unsafe/chenzongwei/cross_yhyb_v2_tail_v4"
n_jobs = int(os.environ.get("FACTOR_N_JOBS", "200"))

print("=" * 70, flush=True)
print("cross_yhyb_v2 baseline rerun (new engine logic; yhyb event-rate factors need higher majority threshold)", flush=True)
print("store:", colblk_store_dir, "n_jobs:", n_jobs, flush=True)
print("=" * 70, flush=True)

gap5, gap1, source, run_info = dw.tail_pipeline_engine(
    colblk_store_dir=colblk_store_dir,
    ver=ver,
    names=None,
    windows=[5, 10, 20],
    fold=True,
    temp_root=temp_root,
    start_date="2015-01-01",
    backtest_start_date="2015-02-01",
    end_date="2026-07-17",
    n_jobs=n_jobs,
    do_base_part=False,
    do_fulltest_part=False,
    do_doc_part=False,
    selection_kwargs={"cover_rate": 0.97, "ret_point_neu_gap5": 0.055},
    majority_count_threshold=10000.0,
    zero_max_threshold=0.12,
    nan_max_threshold=0.04,
    force_restart=True,
    return_run_info=True,
)
print("RESULT", len(gap5), len(gap1), len(source), flush=True)
print("RUN_INFO", run_info, flush=True)
