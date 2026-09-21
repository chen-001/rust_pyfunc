import json, sys, time
from pathlib import Path
sys.path.insert(0, "/tmp/dw_pristine")
import numpy as np, pandas as pd
t0=time.time()
def log(m): print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)
from design_whatever.tail_v2_screen import _select_by_metric
from design_whatever.tail_v2_storage import make_tail_v2_paths
from design_whatever.tail_v4 import _load_ic_wide, _load_summary_json
NAS = Path("/nas197/user_home_unsafe/chenzongwei"); VER="hm104_ssm3"; ROOT=NAS/f"{VER}_tail_v4"
paths = make_tail_v2_paths(VER, ROOT)
dates = np.load(ROOT/"meta"/"dates.npy", allow_pickle=False).astype(np.int32)
s5 = _load_summary_json(paths.metrics_dir/"summary_neu_gap5_candidates.json")
ic5 = _load_ic_wide(paths.ic_dir/"ic_neu_gap5.npy", paths.ic_dir/"ic_neu_gap5_names.json", dates, gap=5)
log(f"loaded {s5.shape} {ic5.shape}")
neu = s5[s5.ratio_mean>=0.5]
log(f"pool {neu.shape}")
for metric, mv in (("IC_mean",0.01), ("hedge_annualized_return",0.055), ("SSM",-1.0)):
    t=time.time()
    r=_select_by_metric(neu, ic5, metric, mv, 0.5)
    log(f"  {metric}: {len(r)} 用时 {time.time()-t:.1f}s")
