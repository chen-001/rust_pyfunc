"""用引擎自己那份筛选函数，在 hm104_mprob 的产物上复算不同 MPROB 门槛的入选名单。

目的：给出与 ssm2(0.5) / ssm3(0.9) 同档位的 MPROB 门槛对比；先用门槛 0.315 复算，
与引擎实际落盘的 selected/gap5_selected.parquet 逐项核对，确认复现口径正确。
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import design_whatever as dw
from design_whatever.tail_v2_screen import select_tail_v2_factors_with_audit
from design_whatever.tail_v3 import _resolve_selection_config

R = Path("/nas197/user_home_unsafe/chenzongwei/hm104_mprob_tail_v4")
GATES = [float(x) for x in sys.argv[1:]] or [0.315, 0.8, 0.9]

selection_kwargs = {
    "cut1_rate": None, "raw_ret_rate": None, "raw_ic_rate": None,
    "ic_more_important_gap5": None, "ic_more_important_gap1": None,
    "cover_rate": 0.5, "ic_point_neu_gap5": 0.01, "ic_point_neu_gap1": 0.006,
    "corr_point_neu": 0.5, "corr_point": 0.8, "cut_num": 35, "cut2_rate": 1.0,
}
cfg_base = _resolve_selection_config(selection_kwargs)
cfg_base.pop("use_ssm", None)
cfg_base.pop("ssm_point_neu_gap5", None)
cfg_base.pop("ssm_point_neu_gap1", None)


def load_wide(stem):
    m = np.load(R / "ic_ts" / f"{stem}.npy", mmap_mode="r")
    names = json.loads((R / "ic_ts" / f"{stem}_names.json").read_text(encoding="utf-8"))
    dates = np.load(R / "ic_ts" / f"{stem}_dates.npy", allow_pickle=False)
    return pd.DataFrame(np.asarray(m), index=dates[: m.shape[0]], columns=names)


print("载入 summary / ic_ts ...", flush=True)
s5n = pd.read_parquet(R / "metrics" / "summary_neu_gap5_candidates.parquet")
s5r = pd.read_parquet(R / "metrics" / "summary_rolled_gap5_candidates.parquet")
s1n = pd.read_parquet(R / "metrics" / "summary_neu_gap1_candidates.parquet")
s1r = pd.read_parquet(R / "metrics" / "summary_rolled_gap1_candidates.parquet")
i5n, i5r = load_wide("ic_neu_gap5"), load_wide("ic_rolled_gap5")
i1n, i1r = load_wide("ic_neu_gap1"), load_wide("ic_rolled_gap1")
print(f"neu_gap5 {s5n.shape} rolled_gap5 {s5r.shape} ic_neu_gap5 {i5n.shape}", flush=True)

actual = pd.read_parquet(R / "selected" / "gap5_selected.parquet")["factor_name"].tolist()
out = {}
for g in GATES:
    cfg = dict(cfg_base, use_mprob=True, mprob_point_neu_gap5=g, mprob_point_neu_gap1=g)
    print(f"\n=== MPROB 门槛 {g} ===", flush=True)
    g5, g1, src, audit = select_tail_v2_factors_with_audit(
        summary_gap5_raw=s5r, summary_gap5_neu=s5n,
        summary_gap1_raw=s1r, summary_gap1_neu=s1n,
        ic_wide_gap5_raw=i5r, ic_wide_gap5_neu=i5n,
        ic_wide_gap1_raw=i1r, ic_wide_gap1_neu=i1n, **cfg)
    out[g] = list(g5)
    neu = s5n[(s5n.ratio_mean >= 0.5) & (s5n.MPROB >= g)]
    print(f"  gap5 入选 {len(g5)} 个（中性化池 MPROB>={g} 共 {len(neu)} 个）", flush=True)
    if g == 0.315:
        print(f"  [复现校验] 与引擎实际落盘逐项一致: {list(g5) == actual}；集合相同: {set(g5) == set(actual)}")
    if g5:
        sub = s5n[s5n.factor_name.isin(g5)]
        print(f"  |IC| 中位 {sub.IC_mean.abs().median():.4f}  多头超额中位 "
              f"{sub.hedge_annualized_return.median()*100:.2f}%  多空年化中位 "
              f"{sub.annualized_return.median()*100:.2f}%  夏普中位 {sub.sharpe_ratio.median():.2f}  "
              f"MPROB 中位 {sub.MPROB.median():.3f}（最小 {sub.MPROB.min():.3f}）")

full = s5n
base = out.get(0.315, actual)
print("\n[各门槛与门槛 0.315 名单的重合]")
for g, lst in out.items():
    print(f"  {g:.3f}: {len(set(lst) & set(base))} / {len(lst)}")

# 与 ssm2 / ssm3 的重合
for tag, p in [("ssm2", "/nas197/user_home_unsafe/chenzongwei/hm104_ssm2_tail_v4"),
               ("ssm3", "/nas197/user_home_unsafe/chenzongwei/hm104_ssm3_tail_v4")]:
    other = set(pd.read_parquet(Path(p) / "selected" / "gap5_selected.parquet").factor_name)
    for g, lst in out.items():
        print(f"  门槛 {g:.3f} ∩ {tag} = {len(set(lst) & other)}")
