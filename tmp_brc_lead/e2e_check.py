"""端到端对账：只跑 brc-ref 指定的 6 个源因子，把引擎产出的 BRC_S/BRC_L 与其独立
Python 参考值对比（2016-01-01 ~ 2025-12-31、gap5、纯风格中性化、IC-only）。

门槛全部放到底（ic_point_neu_*=0.0），保证 _smooth_1 这类低 IC 变体也进 candidates 表。
"""
import numpy as np
import pandas as pd

import design_whatever as dw

STORE = "/hdd/user_home_unsafe/chenzongwei/factor_store_hot_stock_pool_v1_fix"
VER = "brc_e2e"
ROOT = f"/tmp/{VER}_tail_v4"
NAMES = [
    "x60y3_buy_cold_f18_lz_complexity",
    "x60y3_ba_cold_f26_corr_f34",
    "x15y10_buy_cold_f20_corr_f25",
    "x15y10_ba_cold_f15_corr_f28",
    "hotpool_ext_indpool_x60y3_buy_hot_f34_entropy_1d",
    "hotpool_ext_indpool_x15y10_buy_hot_f12_curvature",
]
REF = {
    "x60y3_buy_cold_f18_lz_complexity": (0.004947966042, 0.004174944264),
    "x60y3_ba_cold_f26_corr_f34": (-0.004201033821, -0.000192268650),
    "x15y10_buy_cold_f20_corr_f25": (0.000584873982, 0.009118755397),
    "x15y10_ba_cold_f15_corr_f28": (0.002489939844, -0.011868960748),
    "hotpool_ext_indpool_x60y3_buy_hot_f34_entropy_1d": (0.007201531898, 0.001668653791),
    "hotpool_ext_indpool_x15y10_buy_hot_f12_curvature": (-0.001852483841, 0.008946223174),
}

kwargs = {
    # 走完整回测（不触发 IC-only），这样 raw（rolled）通道也会产出 BRC。
    "cut1_rate": 0.9, "cut2_rate": 0.3, "raw_ret_rate": 0.175, "raw_ic_rate": 0.075,
    "ic_more_important_gap5": None, "ic_more_important_gap1": None,
    "cover_rate": 0.5,
    "ret_point_neu_gap5": -9.0, "ret_point_neu_gap1": -9.0,
    "ic_point_neu_gap5": 0.0, "ic_point_neu_gap1": 0.0,
    "ret_point_gap5": -9.0, "ret_point_gap1": -9.0,
    "ic_point_gap5": 0.0, "ic_point_gap1": 0.0,
    "corr_point_neu": 0.5, "corr_point": 0.8, "cut_num": 35,
}
dw.tail_pipeline_engine(
    colblk_store_dir=STORE, ver=VER, names=NAMES, windows=[5, 10, 20], fold=True,
    n_jobs=16, temp_root=ROOT, industry_neutralize=False,
    start_date="2016-01-01", backtest_start_date="2016-01-01", end_date="2025-12-31",
    do_base_part=False, do_fulltest_part=False, do_doc_part=False,
    selection_kwargs=kwargs, majority_count_threshold=500.0,
    zero_max_threshold=0.1, nan_max_threshold=0.04, force_restart=True,
)
d = pd.read_parquet(f"{ROOT}/metrics/summary_rolled_gap5_candidates.parquet")
print(f"raw(rolled) 表 {d.shape}")
print(f"{'因子':<58}{'表 BRC_S':>14}{'参考 BRC_S':>14}{'dS':>10}{'表 BRC_L':>14}{'参考 BRC_L':>14}{'dL':>10}")
worst = 0.0
for src, (rs, rl) in REF.items():
    for cand in (f"{src}_smooth_1", src):
        sub = d[d.factor_name == cand]
        if not len(sub):
            continue
        r = sub.iloc[0]
        ds, dl = r.BRC_S - rs, r.BRC_L - rl
        worst = max(worst, abs(ds), abs(dl))
        print(f"{cand:<58}{r.BRC_S:>14.12f}{rs:>14.12f}{ds:>10.1e}{r.BRC_L:>14.12f}{rl:>14.12f}{dl:>10.1e}")
        break
    else:
        print(f"{src:<58}  表里没有 _smooth_1 变体")
print(f"最大绝对偏差 = {worst:.3e}")
