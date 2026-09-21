"""受控 A/B：同一批因子、同一窗口，IC-only 模式 vs 完整回测模式，量一下引擎 wall time 差多少。

按 A→B→A→B 交替跑两轮，用第二轮（页缓存已热）做对照。
"""
import time

import design_whatever as dw
import rust_pyfunc as rp

STORE = "/hdd/user_home_unsafe/chenzongwei/factor_store_hot_stock_pool_v1_fix"
N_FACTORS = 200
N_JOBS = 100

names = list(rp.factor_store_v5_info(STORE)["factor_names"])[:N_FACTORS]

BASE = {
    "cover_rate": 0.5, "corr_point_neu": 0.5, "corr_point": 0.8, "cut_num": 35,
    "ic_point_neu_gap5": 0.01, "ic_point_neu_gap1": 0.006,
    "ret_point_neu_gap5": 0.055, "ret_point_neu_gap1": 0.08,
    "ret_point_gap5": 0.1, "ret_point_gap1": 0.13,
    "ic_point_gap5": 0.03, "ic_point_gap1": 0.02,
}
IC_ONLY = dict(BASE, cut1_rate=None, raw_ret_rate=None, raw_ic_rate=None,
               ic_more_important_gap5=None, ic_more_important_gap1=None, cut2_rate=1.0)
FULL = dict(BASE, cut1_rate=0.9, cut2_rate=0.3, raw_ret_rate=0.175, raw_ic_rate=0.075,
            ic_more_important_gap5=None, ic_more_important_gap1=None)


def run(tag, kwargs, ver):
    t = time.perf_counter()
    dw.tail_pipeline_engine(
        colblk_store_dir=STORE, ver=ver, names=names, windows=[5, 10, 20], fold=True,
        n_jobs=N_JOBS, temp_root=f"/tmp/ab_{ver}", industry_neutralize=False,
        start_date="2016-01-01", backtest_start_date="2016-01-01", end_date="2025-12-31",
        do_base_part=False, do_fulltest_part=False, do_doc_part=False,
        selection_kwargs=kwargs, majority_count_threshold=500.0,
        zero_max_threshold=0.1, nan_max_threshold=0.04, force_restart=True,
    )
    dt = time.perf_counter() - t
    print(f"  {tag:10s} ver={ver:16s} {dt:7.1f}s", flush=True)
    return dt


for rnd in (1, 2):
    print(f"--- 第 {rnd} 轮（A=IC-only, B=完整回测） ---", flush=True)
    a = run("IC-only", IC_ONLY, f"ab_ic{rnd}")
    b = run("完整回测", FULL, f"ab_full{rnd}")
    print(f"  比值 完整/IC-only = {b / a:.2f}x", flush=True)
