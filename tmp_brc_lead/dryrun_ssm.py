"""离线复现 hm104_ssm2 / ssm3 的入选名单（用改动前的 design_whatever 快照），
确认离线复现口径与引擎一致，再据此搭 BRC 对比。

用法：python dryrun_ssm.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "/tmp/dw_pristine")

import numpy as np
import pandas as pd

from design_whatever.tail_v2_screen import select_tail_v2_factors
from design_whatever.tail_v2_storage import make_tail_v2_paths
from design_whatever.tail_v4 import _load_ic_wide, _load_summary_json

NAS = Path("/nas197/user_home_unsafe/chenzongwei")
VER = "hm104_ssm3"
ROOT = NAS / f"{VER}_tail_v4"

paths = make_tail_v2_paths(VER, ROOT)
meta = json.loads((ROOT / "meta" / "tail_v4_config.json").read_text())
cfg = dict(meta["selection_kwargs"])
dates = np.load(ROOT / "meta" / "dates.npy", allow_pickle=False).astype(np.int32)
print("config:", {k: v for k, v in cfg.items() if "ssm" in k or "brc" in k or k in ("cut_num", "cut2_rate")})

summary = {
    f"{stage}_gap{gap}": _load_summary_json(
        paths.metrics_dir / f"summary_{stage}_gap{gap}_candidates.json"
    )
    for stage in ("rolled", "neu")
    for gap in (1, 5)
}
ic_wide = {
    f"{stage}_gap{gap}": _load_ic_wide(
        paths.ic_dir / f"ic_{stage}_gap{gap}.npy",
        paths.ic_dir / f"ic_{stage}_gap{gap}_names.json",
        dates,
        gap=gap,
    )
    for stage in ("rolled", "neu")
    for gap in (1, 5)
}
for k, v in summary.items():
    print(f"  summary_{k}: {v.shape}")
for k, v in ic_wide.items():
    print(f"  ic_{k}: {v.shape}")

engine_sel = pd.read_parquet(ROOT / "selected" / "gap5_selected.parquet")["factor_name"].tolist()
print(f"引擎实际入选 {len(engine_sel)} 个")


def run(ssm_thr, use_ssm=True):
    kw = dict(cfg)
    kw.pop("use_mprob", None)
    kw.pop("mprob_point_neu_gap5", None)
    kw.pop("mprob_point_neu_gap1", None)
    kw["use_ssm"] = use_ssm
    kw["ssm_point_neu_gap5"] = ssm_thr
    kw["ssm_point_neu_gap1"] = ssm_thr
    g5, g1, src = select_tail_v2_factors(
        summary_gap5_raw=summary["rolled_gap5"],
        summary_gap5_neu=summary["neu_gap5"],
        summary_gap1_raw=summary["rolled_gap1"],
        summary_gap1_neu=summary["neu_gap1"],
        ic_wide_gap5_raw=ic_wide["rolled_gap5"],
        ic_wide_gap5_neu=ic_wide["neu_gap5"],
        ic_wide_gap1_raw=ic_wide["rolled_gap1"],
        ic_wide_gap1_neu=ic_wide["neu_gap1"],
        **kw,
    )
    return g5, g1


for thr in (0.9, 0.5):
    g5, g1 = run(thr)
    same = set(g5) == set(engine_sel)
    print(f"ssm 门槛 {thr}: gap5={len(g5)} 与引擎名单集合相同={same} 交集={len(set(g5) & set(engine_sel))}")
    if not same:
        print("   离线有引擎无:", sorted(set(g5) - set(engine_sel))[:5])
        print("   引擎有离线无:", sorted(set(engine_sel) - set(g5))[:5])
    print("   顺序完全相同:", g5 == engine_sel)

# ssm1 = 纯 SSM 排序（无门槛、按 SSM 降序去重）
g5, g1 = run(-1.0)
print(f"ssm1（纯 SSM 排序，无门槛）: gap5={len(g5)}")
print("   top5:", g5[:5])
print("   与 ssm3 重合:", len(set(g5) & set(engine_sel)))
