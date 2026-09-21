"""用冒烟那轮（brc_smoke_on）验证：离线用 _select_by_metric 复现引擎的 use_brc 选择。"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/chenzongwei/design_whatever")
from design_whatever.tail_v2_screen import _select_by_metric  # noqa: E402
from design_whatever.tail_v2_storage import make_tail_v2_paths  # noqa: E402
from design_whatever.tail_v4 import _load_ic_wide, _load_summary_json  # noqa: E402

VER = "brc_smoke_on"
ROOT = Path(f"/tmp/{VER}_tail_v4")
paths = make_tail_v2_paths(VER, ROOT)
dates = np.load(ROOT / "meta" / "dates.npy", allow_pickle=False).astype(np.int32)
s = _load_summary_json(paths.metrics_dir / "summary_neu_gap5_candidates.json")
ic = _load_ic_wide(paths.ic_dir / "ic_neu_gap5.npy", paths.ic_dir / "ic_neu_gap5_names.json", dates, gap=5)
print(f"表 {s.shape}  ic {ic.shape}")
neu = s[s.ratio_mean >= 0.5].copy()
print(f"池 {neu.shape}  BRC 范围 [{neu.BRC.min():.4f}, {neu.BRC.max():.4f}]")
r = _select_by_metric(neu, ic, "BRC", -9.0, 0.5)
offline = r.factor_name.tolist()[:5]
engine = pd.read_parquet(ROOT / "selected" / "gap5_selected.parquet")["factor_name"].tolist()
print("引擎:", engine)
print("离线:", offline)
print("顺序完全相同:", engine == offline, " 集合相同:", set(engine) == set(offline))
print("引擎名单 BRC:", [round(float(neu.set_index('factor_name').loc[n, 'BRC']), 6) for n in engine])
print("池内 BRC 降序前 8:", neu.nlargest(8, "BRC")[["factor_name", "BRC", "IC_mean"]].to_string(index=False))
