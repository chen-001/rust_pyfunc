"""修复验证：直接调用真实的 select_tail_v2_factors，对比修复前后在 hm104 缓存上的入选名单。

修复点：名额为 0 的通道不参与归类（tail_v2_screen._merge_ranked）。
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from design_whatever.tail_v2_screen import select_tail_v2_factors

CFG = dict(
    cover_rate=0.5,
    ic_point_neu_gap5=0.01,
    ic_point_neu_gap1=0.006,
    corr_point_neu=0.5,
    corr_point=0.8,
    cut_num=35,
    cut1_rate=None,
    cut2_rate=1.0,
    raw_ret_rate=None,
    raw_ic_rate=None,
    ic_more_important_gap5=None,
    ic_more_important_gap1=None,
    use_ssm=True,
)


def load_wide(R, stem):
    m = np.load(R / "ic_ts" / f"{stem}.npy", allow_pickle=False)
    names = json.loads((R / "ic_ts" / f"{stem}_names.json").read_text(encoding="utf-8"))
    dates = np.load(R / "ic_ts" / f"{stem}_dates.npy", allow_pickle=False)
    return pd.DataFrame(m, index=dates[: m.shape[0]], columns=names)


for ver, gate in [("hm104_ssm2", 0.5), ("hm104_ssm3", 0.9)]:
    R = Path(f"/nas197/user_home_unsafe/chenzongwei/{ver}_tail_v4")
    sn = pd.read_parquet(R / "metrics" / "summary_neu_gap5_candidates.parquet")
    sr = pd.read_parquet(R / "metrics" / "summary_rolled_gap5_candidates.parquet")
    n1 = pd.read_parquet(R / "metrics" / "summary_neu_gap1_candidates.parquet")
    r1 = pd.read_parquet(R / "metrics" / "summary_rolled_gap1_candidates.parquet")
    g5n, g5r = load_wide(R, "ic_neu_gap5"), load_wide(R, "ic_rolled_gap5")
    g1n, g1r = load_wide(R, "ic_neu_gap1"), load_wide(R, "ic_rolled_gap1")

    cfg = dict(CFG, ssm_point_neu_gap5=gate, ssm_point_neu_gap1=gate)
    new5, new1, src = select_tail_v2_factors(
        summary_gap5_raw=sr, summary_gap5_neu=sn,
        summary_gap1_raw=r1, summary_gap1_neu=n1,
        ic_wide_gap5_raw=g5r, ic_wide_gap5_neu=g5n,
        ic_wide_gap1_raw=g1r, ic_wide_gap1_neu=g1n,
        **cfg,
    )
    old = pd.read_parquet(R / "selected" / "gap5_selected.parquet")["factor_name"].tolist()
    full = pd.read_parquet(R / "metrics" / "summary_neu_gap5_candidates.parquet")
    full["absic"] = full.IC_mean.abs()

    print("=" * 104)
    print(f"[{ver}] 门槛 {gate}")
    print(f"  旧名单 {len(old)} 个 → 新名单 {len(new5)} 个")
    print(f"  两者逐项一致: {new5 == old}    集合相同: {set(new5) == set(old)}")
    added = [n for n in new5 if n not in set(old)]
    removed = [n for n in old if n not in set(new5)]
    print(f"  新增 {len(added)} 个，移除 {len(removed)} 个")
    for tag, lst in [("新增", added), ("移除", removed)]:
        for n in lst[:8]:
            r = full[full.factor_name == n]
            if len(r):
                r = r.iloc[0]
                print(f"    {tag}: |IC|={abs(r.IC_mean):.4f} SSM={r.SSM:+.3f} 多空年化={r.annualized_return*100:6.2f}% 夏普={r.sharpe_ratio:5.2f}  {n[:62]}")
    for tag, lst in [("旧名单", old), ("新名单", new5)]:
        sub = full[full.factor_name.isin(lst)]
        print(f"  {tag}: |IC|中位 {sub.absic.median():.4f}  多空年化中位 {sub.annualized_return.median()*100:6.2f}%  夏普中位 {sub.sharpe_ratio.median():5.2f}")
