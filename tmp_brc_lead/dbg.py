import json, sys, time
from pathlib import Path
sys.path.insert(0, "/tmp/dw_pristine")
import numpy as np, pandas as pd
t0=time.time()
def log(m): print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)
from design_whatever.tail_v2_screen import select_tail_v2_factors
from design_whatever.tail_v2_storage import make_tail_v2_paths
from design_whatever.tail_v4 import _load_ic_wide, _load_summary_json
log("imports ok")
NAS = Path("/nas197/user_home_unsafe/chenzongwei"); VER="hm104_ssm3"; ROOT=NAS/f"{VER}_tail_v4"
paths = make_tail_v2_paths(VER, ROOT)
log(f"paths ok {paths.ic_dir}")
meta = json.loads((ROOT/"meta"/"tail_v4_config.json").read_text())
cfg = dict(meta["selection_kwargs"])
dates = np.load(ROOT/"meta"/"dates.npy", allow_pickle=False).astype(np.int32)
log(f"dates {dates.shape}")
for stage in ("rolled","neu"):
    for gap in (1,5):
        s=_load_summary_json(paths.metrics_dir/f"summary_{stage}_gap{gap}_candidates.json"); log(f"summary {stage} {gap} {s.shape}")
for stage in ("rolled","neu"):
    for gap in (1,5):
        ic=_load_ic_wide(paths.ic_dir/f"ic_{stage}_gap{gap}.npy", paths.ic_dir/f"ic_{stage}_gap{gap}_names.json", dates, gap=gap); log(f"ic {stage} {gap} {ic.shape}")
log("all loaded")
