"""预计算 hm104 上 |IC| / 多头超额 / ssm1 三个选法的前 35（从 hm104_mprob 完整表），
省得最终分析时再等。结果写 tmp_brc_lead/precomputed_picks.json。
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/chenzongwei/design_whatever")
from design_whatever.tail_v2_screen import _select_by_metric  # noqa: E402
from design_whatever.tail_v2_storage import make_tail_v2_paths  # noqa: E402
from design_whatever.tail_v4 import _load_ic_wide, _load_summary_json  # noqa: E402

NAS = Path("/nas197/user_home_unsafe/chenzongwei")
VER_FULL = "hm104_mprob"
R = NAS / f"{VER_FULL}_tail_v4"
OUT = Path("/home/chenzongwei/rust_pyfunc/tmp_brc_lead/precomputed_picks.json")
t0 = time.time()


def log(m):
    print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)


paths = make_tail_v2_paths(VER_FULL, R)
dates = np.load(R / "meta" / "dates.npy", allow_pickle=False).astype(np.int32)
full = _load_summary_json(paths.metrics_dir / "summary_neu_gap5_candidates.json")
pool = full[full.ratio_mean >= 0.5].copy()
ic5 = _load_ic_wide(
    paths.ic_dir / "ic_neu_gap5.npy", paths.ic_dir / "ic_neu_gap5_names.json", dates, gap=5
)
log(f"表 {full.shape} 池 {pool.shape} ic_wide {ic5.shape}")

out = {}
for key, metric, mv in (("ic", "IC_mean", 0.01), ("hedge", "hedge_annualized_return", 0.055),
                        ("ssm1", "SSM", -1.0)):
    t = time.time()
    r = _select_by_metric(pool, ic5, metric, mv, 0.5)
    out[key] = r.factor_name.tolist()[:35]
    log(f"  {key}: 去重后 {len(r)}，取前 {len(out[key])}，用时 {time.time()-t:.1f}s")
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
log("done")
