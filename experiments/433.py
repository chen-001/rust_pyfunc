"""fulltest 的 IC 改成日频以后，把 hm94 的全部 fulltest 结果重算一遍。

背景：2026-10-16 把 tail_v4_fulltest_rust 的 IC 采样从「每 H_hori 天一次」改成「每天一次」，
与 tail_pipeline_engine 的日频中性化 IC 对齐。已经落盘的 fulltest 结果都是旧口径算的，
要重算才能反映新口径。

做法：
  1. 把 postprocess_fulltest/done 下的完成标记与 factor_summary/hm94_1st_son_{h5,neu} 下的
     输出目录都挪到一个带时间戳的备份目录；
  2. 调 tail_pipeline_v4_postprocess(do_fulltest_part=True、do_base_part=False、do_doc_part=False)
     重算这 90 个 slot；
  3. 打印新旧 IC 的对照（每个因子的 |新 IC| 与 |旧 IC| 的比值分布）。
不碰 base、不碰 hf_data 与 dailybank。
"""
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

import design_whatever as dw

VER = "hm94"
SUM = Path("/nas197/user_home_unsafe/chenzongwei/factor_summary")
TAIL = Path("/nas197/user_home_unsafe/chenzongwei/hm94_tail_v4")
DONE = TAIL / "postprocess_fulltest" / "done"
STAMP = time.strftime("%Y%m%d_%H%M")
BACKUP = TAIL / "postprocess_fulltest" / f"pre_daily_ic_{STAMP}"
BACKUP.mkdir(parents=True, exist_ok=True)

selected = TAIL / "selected"
gap5 = pd.read_parquet(selected / "gap5_selected.parquet")["factor_name"].tolist()
gap1 = pd.read_parquet(selected / "gap1_selected.parquet")["factor_name"].tolist()
source_names = pd.read_parquet(selected / "source_selected.parquet")["source_factor"].tolist()

# 旧值留一份，用来算新旧比值
old = {}
for bucket, names, h in (("hm94_1st_son_h5", gap5, 5), ("hm94_1st_son_neu", gap5, 5),
                         ("hm94_1st_son_h5", gap1, 1), ("hm94_1st_son_neu", gap1, 1)):
    for n in names:
        f = SUM / bucket / n / str(h) / "summary.csv"
        if f.exists():
            old[(bucket, n, h)] = float(pd.read_csv(f).iloc[0]["IC_mean"])

moved_markers = 0
for m in sorted(DONE.glob("*.json")):
    shutil.move(str(m), str(BACKUP / m.name))
    moved_markers += 1
print(f"[备份] 完成标记 {moved_markers} 个 → {BACKUP}", flush=True)
for bucket in ("hm94_1st_son_h5", "hm94_1st_son_neu"):
    d = SUM / bucket
    for n in set(gap5 + gap1):
        sub = d / n
        if sub.exists():
            shutil.move(str(sub), str(BACKUP / f"{bucket}__{n}"))
print("[备份] 输出目录已挪走", flush=True)

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
    fulltest_jobs=16,
    fulltest_resume=True,
    industry_neutralize=False,
)

rows = []
for (bucket, n, h), old_ic in old.items():
    f = SUM / bucket / n / str(h) / "summary.csv"
    if not f.exists():
        rows.append((bucket, n, h, old_ic, np.nan, np.nan, np.nan)); continue
    d = pd.read_csv(f).iloc[0]
    rows.append((bucket, n, h, old_ic, float(d["IC_mean"]), float(d["IR"]), int(d["date_size"])))
df = pd.DataFrame(rows, columns=["bucket", "factor", "horizon", "旧IC", "新IC", "新IR", "date_size"])
df["比值"] = (df["新IC"].abs() / df["旧IC"].abs()).replace([np.inf, -np.inf], np.nan)
df.to_csv(BACKUP / "新旧IC对照.csv", index=False)
print("\n重算 %d 份，缺 %d 份" % (len(df.dropna(subset=["新IC"])), int(df["新IC"].isna().sum())), flush=True)
print("新/旧 |IC| 比值：中位 %.3f  最小 %.3f  最大 %.3f" % (df["比值"].median(), df["比值"].min(), df["比值"].max()), flush=True)
print("变化最大的 8 个：", flush=True)
print(df.reindex(df["比值"].sub(1).abs().sort_values(ascending=False).index).head(8).to_string(index=False), flush=True)
print("date_size 是否仍齐全:", sorted(df.date_size.dropna().unique()), flush=True)
