"""use_ssm 端到端冒烟：小因子子集 + 短窗口，验证 SSM 列产出与按 SSM 选择。

不做 base / fulltest / doc，只跑引擎 + 筛选。
"""
import json
import sys

import numpy as np
import pandas as pd

import design_whatever as dw
import rust_pyfunc as rp

STORE = "/hdd/user_home_unsafe/chenzongwei/factor_store_hm100"
VER = "ssm_smoke"
ROOT = "/tmp/ssm_smoke_tail_v4"

info = rp.factor_store_v5_info(STORE)
allnames = list(info["factor_names"])
print(f"store 列数 {len(allnames)}")
# 挑一批名字里有 capm 的，保证有足够横截面变化
names = [n for n in allnames if "residual_zscore" in n][:12] or allnames[:12]
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
}

for use_ssm in (False, True):
    ver = f"{VER}_{'on' if use_ssm else 'off'}"
    root = f"/tmp/{ver}_tail_v4"
    print("=" * 90)
    print(f"use_ssm={use_ssm}  ver={ver}")
    g5, g1, src, run_info = dw.tail_pipeline_engine(
        colblk_store_dir=STORE,
        ver=ver,
        names=names,
        windows=[5, 10, 20],
        fold=True,
        n_jobs=8,
        temp_root=root,
        industry_neutralize=False,
        start_date="2023-01-01",
        backtest_start_date="2023-02-01",
        end_date="2023-12-31",
        do_base_part=False,
        do_fulltest_part=False,
        do_doc_part=False,
        selection_kwargs=selection_kwargs,
        use_ssm=use_ssm,
        majority_count_threshold=500.0,
        zero_max_threshold=0.1,
        nan_max_threshold=0.04,
        force_restart=True,
        return_run_info=True,
    )
    print(f"  gap5_selected={g5}")
    print(f"  gap1_selected={g1}")
    for f in ("summary_neu_gap5_candidates.parquet", "summary_neu_gap5_all.parquet"):
        p = f"{root}/metrics/{f}"
        try:
            d = pd.read_parquet(p)
            has = "SSM" in d.columns
            s = d["SSM"].to_numpy() if has else None
            print(f"  {f}: {d.shape} 有SSM列={has}"
                  + (f"  SSM 非空 {np.isfinite(s).sum()}/{len(s)}"
                     f"  范围[{np.nanmin(s):.3f}, {np.nanmax(s):.3f}]" if has and np.isfinite(s).any() else ""))
            if has and f.endswith("candidates.parquet") and len(d):
                print("    按 SSM 降序前 5:")
                print(d[["factor_name", "IC_mean", "SSM"]].sort_values("SSM", ascending=False).head(5).to_string(index=False))
                print("    按 |IC| 降序前 5:")
                print(d[["factor_name", "IC_mean", "SSM"]].assign(a=lambda x: x.IC_mean.abs())
                      .sort_values("a", ascending=False).head(5)[["factor_name", "IC_mean", "SSM"]].to_string(index=False))
        except Exception as e:
            print(f"  {f}: 读取失败 {e}")
