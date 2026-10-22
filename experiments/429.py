"""把 hm94 那 5 个 fulltest IC 与筛选 IC 对不上的因子，在 hm94 自己的目录里强制重算一遍。

背景（task 426 跑完核对时发现）：
  45 个入选因子里有 5 个，factor_summary/hm94_1st_son_neu/<因子>/5/summary.csv 里的
  中性化 IC 与 tail_v4 的 ic_ts/ic_neu_gap5.npy 差两到三个数量级；其余 40 个两边差 5% 以内。
  这 5 份里 4 份的时间戳是 2026-10-15 12:44~12:49，本次 fulltest 是「已恢复」而没有重算它们。

这个脚本只做三件事：
  1. 把这 5 个因子在 hm94_1st_son_{h5,neu} 下的旧输出目录改名备份（带时间戳）；
  2. 把 hm94_tail_v4/postprocess_fulltest/done 下对应的完成标记移走，让 resume 不再跳过；
  3. 调 tail_pipeline_v4_postprocess，只跑 fulltest（do_base_part=False、do_doc_part=False），
     把这几份算回来，然后打印新旧值与引擎值的对照。

不碰 base、不碰 doc、不碰 hf_data 与 dailybank。
"""
import json
import shutil
import time
from pathlib import Path

import pandas as pd

import design_whatever as dw

VER = "hm94"
STAMP = time.strftime("%Y%m%d_%H%M")
SUM = Path("/nas197/user_home_unsafe/chenzongwei/factor_summary")
TAIL = Path("/nas197/user_home_unsafe/chenzongwei/hm94_tail_v4")
DONE = TAIL / "postprocess_fulltest" / "done"
BACKUP = TAIL / "postprocess_fulltest" / f"stale_{STAMP}"

FACTORS = [
    "ask_B_pre5_ct_b_mean_entropy_1d_mean_smooth_10",
    "bid_B_seg_ct_b_mean_cv_mean_smooth_10",
    "bid_B_pre5_ct_b_mean_p95_std_smooth_20",
    "bid_B_seg_pvol_std_fold_mean_smooth_10",
    "bid_B_seg_am_a_mean_cv_fold_min_smooth_10",
]

BACKUP.mkdir(parents=True, exist_ok=True)

# 1. 旧输出目录挪走
for bucket in ("hm94_1st_son_h5", "hm94_1st_son_neu"):
    for name in FACTORS:
        src = SUM / bucket / name
        if src.exists():
            dst = BACKUP / f"{bucket}__{name}"
            shutil.move(str(src), str(dst))
            print(f"[备份] {src} → {dst}", flush=True)

# 2. 完成标记挪走
moved = 0
for marker in sorted(DONE.glob("*.json")):
    stem = marker.stem
    if any(stem.endswith(name) for name in FACTORS):
        shutil.move(str(marker), str(BACKUP / marker.name))
        moved += 1
print(f"[备份] 完成标记 {moved} 个", flush=True)

# 3. 只重跑 fulltest
selected = TAIL / "selected"
gap5 = pd.read_parquet(selected / "gap5_selected.parquet")["factor_name"].tolist()
gap1 = pd.read_parquet(selected / "gap1_selected.parquet")["factor_name"].tolist()
source_names = pd.read_parquet(selected / "source_selected.parquet")["source_factor"].tolist()
print(f"名单 gap5={len(gap5)} gap1={len(gap1)} source={len(source_names)}", flush=True)

dw.tail_pipeline_v4_postprocess(
    ver=VER,
    gap5_selected=gap5,
    gap1_selected=gap1,
    source_names=source_names,
    temp_root=str(TAIL),
    source_dir="/hdd/user_home_unsafe/chenzongwei/factor_data/hm94",
    do_base_part=False,
    do_fulltest_part=True,
    do_doc_part=False,
    no_pri=True,
    style_vars_dir="/ssd_data/data/vars",
    min_valid=12,
    start_date="2016-01-01",
    backtest_start_date="2016-01-01",
    end_date="2025-12-31",
    fulltest_jobs=8,
    fulltest_resume=True,
    industry_neutralize=False,
)

# 4. 新旧对照
eng = pd.read_parquet(
    TAIL / "metrics/summary_neu_gap5_candidates.parquet", columns=["factor_name", "IC_mean"]
).set_index("factor_name")["IC_mean"]
print("\n因子                                          引擎筛选IC   重算后IC    旧存档IC", flush=True)
for name in FACTORS:
    new = pd.read_csv(SUM / f"hm94_1st_son_neu/{name}/5/summary.csv").iloc[0]
    old = pd.read_csv(BACKUP / f"hm94_1st_son_neu__{name}/5/summary.csv").iloc[0]
    print(
        "%-46s %+.5f   %+.5f   %+.5f"
        % (name, eng.get(name, float("nan")), new["IC_mean"], old["IC_mean"]),
        flush=True,
    )
print("\n重算完成", flush=True)
