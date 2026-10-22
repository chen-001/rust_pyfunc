"""核对：hm94 的 fulltest 存下来的中性化 IC，跟引擎筛选用的 IC 对不上的那 5 个因子，重新跑一遍。

背景：task 426（hm94 定稿重跑）跑完以后核对发现，45 个入选因子里有 5 个的
factor_summary/hm94_1st_son_neu/<因子>/5/summary.csv 里的 IC 跟
tail_v4 的 ic_ts/ic_neu_gap5.npy 差两个数量级，其余 40 个两边差 5% 以内。
这里用 run_manual_fulltest 把这 5 个（外加上两个对照）在 manual 版本里强制重算一遍
（fulltest_resume=False，覆盖 manual 版本），看重新算出来的值靠近哪一边。
只写 manual_* 目录，不碰 hm94 的任何交付产物。
"""
import design_whatever as dw

NAMES = [
    # 对不上的 5 个
    "ask_B_pre5_ct_b_mean_entropy_1d_mean_smooth_10",
    "bid_B_seg_ct_b_mean_cv_mean_smooth_10",
    "bid_B_pre5_ct_b_mean_p95_std_smooth_20",
    "bid_B_seg_pvol_std_fold_mean_smooth_10",
    "bid_B_seg_am_a_mean_cv_fold_min_smooth_10",
    # 对得上的 2 个，当对照
    "bid_B_seg_am_a_mean_p5_smooth_1",
    "bid_B_pre5_ivtrend_std_mean_smooth_10",
]

result = dw.run_manual_fulltest(
    factor_names=NAMES,
    source_dir="/hdd/user_home_unsafe/chenzongwei/factor_data/hm94",
    gaps=(5,),
    end_date="2025-12-31",
    fulltest_jobs=8,
    fulltest_resume=False,
    accumulate=False,
)
print("重算完成:", result["ver"], result["gap5_selected"], flush=True)
