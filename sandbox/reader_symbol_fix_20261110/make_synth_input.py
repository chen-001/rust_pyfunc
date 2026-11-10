"""准备“合成撤单因子”的输入：从一个真实 store 读出一个因子，做三列：
  plain_full     原值（对照，名字不含 cxl）
  cxl_masked     沪市（6 开头）20211105 之前设为缺失（模拟撤单因子，名字含 cxl）
  ctrl_masked    同样的缺失，但名字不含 cxl（默认规则下应被淘汰，用来对照）
写成 synth_input/{dates.npy, codes.txt, values.npy}，由 make_store 写成列式 store。
"""
import os
from pathlib import Path

import numpy as np
import rust_pyfunc as rp

HERE = Path(__file__).resolve().parent
SRC = "/hdd/user_home_unsafe/chenzongwei/factor_store_yupei_dist_lite"
FACTOR = "assort_flow_opp_t5_corr"
CUT = 20211105

info = rp.factor_store_v5_info(SRC)
col = list(info["factor_names"]).index(FACTOR)
tmpl = rp.factor_store_v5_template(SRC)
dates_axis = np.asarray(tmpl["dates"], dtype=np.int64)
codes_axis = list(tmpl["stocks"])
r = rp.factor_store_v5_read_factor(SRC, col)
d = dates_axis[np.asarray(r["date_id"])]
c = np.asarray(codes_axis, dtype=object)[np.asarray(r["code_id"])]
v = np.asarray(r["factor"], dtype=np.float32)
keep = np.isfinite(v)
d, c, v = d[keep], c[keep], v[keep]
order = np.lexsort((c.astype(str), d))
d, c, v = d[order], c[order], v[order]

sh = np.array([x.startswith("6") for x in c])
masked = v.copy()
masked[sh & (d < CUT)] = np.nan
vals = np.stack([v, masked, masked], axis=1).astype(np.float32)

out = HERE / "synth_input"
out.mkdir(exist_ok=True)
np.save(out / "dates.npy", d.astype(np.int64))
np.save(out / "values.npy", vals)
(out / "codes.txt").write_text("\n".join(x.split(".")[0] for x in c.tolist()) + "\n")  # 写入器会自己加交易所后缀
print("行数", len(d), "日期", d.min(), d.max(), "沪市早期被设为缺失的行", int((sh & (d < CUT)).sum()))
