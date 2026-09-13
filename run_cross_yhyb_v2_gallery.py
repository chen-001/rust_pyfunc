"""cross_yhyb_v2 补跑 fulltest 阶段（恢复 gallery 总览表格的指标列）。

背景：正式基线只跑了候选筛选（do_fulltest_part=False），gallery 总览表格的
指标列来自 factor_summary/{ver}_1st_son_*/summary.csv（fulltest 阶段产物），
因此只显示 因子名 + mean_corr + 两个空占位列。
本脚本复用同一 temp_root（force_restart=False，1916 个 task_result 全部恢复，
筛选秒过），只补 do_fulltest_part。
"""
import os
import design_whatever as dw

ver = "cross_yhyb_v2"
colblk_store_dir = "/hdd/user_home_unsafe/chenzongwei/factor_store_cross_yhyb"
temp_root = "/hdd/user_home_unsafe/chenzongwei/cross_yhyb_v2_tail_v4"
n_jobs = int(os.environ.get("FACTOR_N_JOBS", "200"))

print("=" * 70, flush=True)
print("cross_yhyb_v2 fulltest 补跑（恢复 gallery 指标列）", flush=True)
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
    do_fulltest_part=True,     # 关键：生成 factor_summary/{ver}_1st_son_*/summary.csv
    do_doc_part=False,
    selection_kwargs={"cover_rate": 0.97, "ret_point_neu_gap5": 0.055},
    majority_count_threshold=10000.0,
    zero_max_threshold=0.12,
    nan_max_threshold=0.04,
    force_restart=False,        # 复用缓存：task_result 全量恢复
    return_run_info=True,
)
print("RESULT", len(gap5), len(gap1), len(source), flush=True)
print("DONE", flush=True)