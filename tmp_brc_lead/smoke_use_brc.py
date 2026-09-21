"""use_brc 端到端冒烟：小因子子集 + 短窗口，验证 BRC 三列产出与按 BRC 选择。

不做 base / fulltest / doc，只跑引擎 + 筛选。
"""
import sys

import numpy as np
import pandas as pd

import design_whatever as dw
import rust_pyfunc as rp

STORE = "/hdd/user_home_unsafe/chenzongwei/factor_store_hot_stock_pool_v1_fix"
VER = "brc_smoke"
ROOT = f"/tmp/{VER}_tail_v4"

info = rp.factor_store_v5_info(STORE)
allnames = list(info["factor_names"])
print(f"store 列数 {len(allnames)}")
names = [n for n in allnames if "x60y3_buy_hot_f0" in n][:10] or allnames[:10]
print(f"本次测试因子 {len(names)} 个，例如 {names[0]}")

selection_kwargs = {
    "cut1_rate": None,
    "raw_ret_rate": None,
    "raw_ic_rate": None,
    "ic_more_important_gap5": None,
    "ic_more_important_gap1": None,
    "cover_rate": 0.5,
    "ic_point_neu_gap5": 0.01,
    "ic_point_neu_gap1": 0.006,
    "corr_point_neu": 0.5,
    "corr_point": 0.8,
    "cut_num": 5,
    "cut2_rate": 1.0,
    "brc_point_neu_gap5": -9.0,
    "brc_point_neu_gap1": -9.0,
}

for use_brc in (False, True):
    ver = f"{VER}_{'on' if use_brc else 'off'}"
    root = f"/tmp/{ver}_tail_v4"
    print("=" * 90)
    print(f"use_brc={use_brc}  ver={ver}")
    g5, g1, src, run_info = dw.tail_pipeline_engine(
        colblk_store_dir=STORE,
        ver=ver,
        names=names,
        windows=[5, 10, 20],
        fold=True,
        n_jobs=16,
        temp_root=root,
        industry_neutralize=False,
        start_date="2023-01-01",
        backtest_start_date="2023-02-01",
        end_date="2023-12-31",
        do_base_part=False,
        do_fulltest_part=False,
        do_doc_part=False,
        selection_kwargs=selection_kwargs,
        use_brc=use_brc,
        majority_count_threshold=500.0,
        zero_max_threshold=0.1,
        nan_max_threshold=0.04,
        force_restart=True,
        return_run_info=True,
    )
    print(f"  gap5_selected={g5}")
    print(f"  gap1_selected={g1}")
    p = f"{root}/metrics/summary_neu_gap5_candidates.parquet"
    d = pd.read_parquet(p)
    print(f"  表 {d.shape} 列={list(d.columns)}")
    for c in ("BRC", "BRC_S", "BRC_L"):
        if c in d.columns:
            v = d[c].to_numpy()
            print(f"    {c}: 非空 {np.isfinite(v).sum()}/{len(v)} 范围"
                  f"[{np.nanmin(v):.4f}, {np.nanmax(v):.4f}]")
    if "BRC" in d.columns and len(d):
        top = d.sort_values("BRC", ascending=False)[["factor_name", "BRC", "BRC_S", "BRC_L", "IC_mean"]].head(8)
        print(top.to_string(index=False))
        # 选择是否严格按 BRC 降序
        order = d.set_index("factor_name").loc[g5, "BRC"].to_numpy()
        print(f"  入选名单的 BRC 是否降序: {bool(np.all(np.diff(order) <= 1e-12))}")
