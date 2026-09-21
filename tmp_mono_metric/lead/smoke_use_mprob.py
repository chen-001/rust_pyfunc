"""use_mprob 端到端冒烟：小因子子集 + 短窗口，验证 MPROB 列产出与按 MPROB 门槛选择。

不做 base / fulltest / doc，只跑引擎 + 筛选。跑完把 temp_root 交给 verifier 做端到端数值核对。
"""
import numpy as np
import pandas as pd

import design_whatever as dw
import rust_pyfunc as rp

STORE = "/hdd/user_home_unsafe/chenzongwei/factor_store_hm100"

info = rp.factor_store_v5_info(STORE)
allnames = list(info["factor_names"])
names = [n for n in allnames if "residual_zscore" in n][:12] or allnames[:12]
print(f"store 列数 {len(allnames)}，本次测试因子 {len(names)} 个，例如 {names[0]}")

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

base = dict(
    colblk_store_dir=STORE,
    names=names,
    windows=[5, 10, 20],
    fold=True,
    n_jobs=8,
    industry_neutralize=False,
    start_date="2023-01-01",
    backtest_start_date="2023-02-01",
    end_date="2023-12-31",
    do_base_part=False,
    do_fulltest_part=False,
    do_doc_part=False,
    selection_kwargs=selection_kwargs,
    majority_count_threshold=500.0,
    zero_max_threshold=0.1,
    nan_max_threshold=0.04,
    force_restart=True,
    return_run_info=True,
)

for use_mprob in (False, True):
    ver = f"mprob_smoke_{'on' if use_mprob else 'off'}"
    root = f"/tmp/{ver}_tail_v4"
    print("=" * 100)
    print(f"use_mprob={use_mprob}  ver={ver}  temp_root={root}")
    kwargs = dict(base, ver=ver, temp_root=root, use_mprob=use_mprob)
    if use_mprob:
        kwargs["selection_kwargs"] = dict(selection_kwargs, mprob_point_neu_gap5=-1.0,
                                          mprob_point_neu_gap1=-1.0)
    g5, g1, src, run_info = dw.tail_pipeline_engine(**kwargs)
    print(f"  gap5_selected={len(g5)}  gap1_selected={len(g1)}")
    d = pd.read_parquet(f"{root}/metrics/summary_neu_gap5_candidates.parquet")
    print(f"  summary_neu_gap5_candidates: {d.shape} 列={list(d.columns)}")
    for col in ("MPROB", "SSM"):
        if col in d.columns:
            s = d[col].to_numpy(dtype=float)
            print(f"    {col}: 非空 {np.isfinite(s).sum()}/{len(s)}"
                  f"  范围[{np.nanmin(s):.4f}, {np.nanmax(s):.4f}]")
    if "MPROB" in d.columns and len(d):
        print("    按 MPROB 降序前 5:")
        print(d[["factor_name", "IC_mean", "SSM", "MPROB"]]
              .sort_values("MPROB", ascending=False).head(5).to_string(index=False))
    if use_mprob:
        print(f"  [校验] 门槛 -1.0 时入选名单与 off 版是否一致（在脚本外层比较）")
