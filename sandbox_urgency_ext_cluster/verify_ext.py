"""sandbox_urgency_ext_cluster 验证脚本（逻辑 + 前瞻 IC + 残差 IC）。

用法：python verify_ext.py（dev 版）；速度基准需先 bash build.sh --release。
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/chenzongwei/.agents/skills/factor-data-reader")
import daily_data_reader as ddr  # noqa: E402

import sandbox_urgency_ext_cluster as sb  # noqa: E402

DATES = [20150824, 20180615, 20200203, 20210315, 20220819, 20230427, 20241231, 20260203]
NAMES = sb.py_urgency_ext_names()
N_FACTORS = len(NAMES)
assert N_FACTORS == 89

# 日频数据（前瞻口径）
closes = ddr.close()
gap1 = ddr.gap1()
gap5 = ddr.gap5()
money = ddr.money()
flow_cap = ddr.flow_cap()


def suffix(code):
    return code + (".SH" if code[0] in "69" else ".SZ")


def compute_day(date):
    codes, vals = sb.py_urgency_ext(date)
    arr = np.array(vals, dtype=np.float32).reshape(len(codes), N_FACTORS)
    idx = pd.to_datetime(str(date), format="%Y%m%d")
    cols = [suffix(c) for c in codes]
    # 只保留日频有数据的股票
    valid_cols = [c for c in cols if c in gap1.columns]
    if not valid_cols:
        return None, None, None
    pos = {c: i for i, c in enumerate(cols) if c in gap1.columns}
    sub = np.array([[arr[pos[c]] for c in valid_cols]])  # (1, n, F)
    return idx, valid_cols, sub[0]


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = ~(np.isnan(a) | np.isnan(b))
    if m.sum() < 30:
        return np.nan
    a, b = a[m], b[m]
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    if np.std(ra) == 0 or np.std(rb) == 0:
        return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    print(f"验证日期: {DATES}  因子数: {N_FACTORS}")
    # 逐日计算并缓存
    days = {}
    for d in DATES:
        idx, cols, arr = compute_day(d)
        if idx is None:
            print(f"  {d}: 无数据，跳过")
            continue
        days[d] = (idx, cols, arr)
        print(f"  {d}: {len(cols)} 股, NaN率={float(np.isnan(arr).mean()):.4f}")

    # 逐因子 IC（gap1 / gap5）+ 残差 IC
    rows = []
    for j, nm in enumerate(NAMES):
        ic1s, ic5s = [], []
        for d, (idx, cols, arr) in days.items():
            g1 = gap1.loc[idx, cols].to_numpy(dtype=float)
            g5 = gap5.loc[idx, cols].to_numpy(dtype=float)
            f = arr[:, j]
            ic1s.append(spearman(f, g1))
            ic5s.append(spearman(f, g5))
        ic1s = np.array([x for x in ic1s if not np.isnan(x)])
        ic5s = np.array([x for x in ic5s if not np.isnan(x)])
        rows.append({
            "name": nm,
            "group": nm.split("_")[-1],
            "ic1_mean": ic1s.mean() if len(ic1s) else np.nan,
            "ic1_dir": (ic1s > 0).mean() if len(ic1s) else np.nan,
            "ic5_mean": ic5s.mean() if len(ic5s) else np.nan,
            "ic5_dir": (ic5s > 0).mean() if len(ic5s) else np.nan,
        })
    res = pd.DataFrame(rows).sort_values("ic1_mean", key=lambda s: s.abs(), ascending=False)
    print("\n=== gap1 |IC| 前 12 ===")
    print(res.head(12).to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
    print("\n=== gap5 |IC| 前 12 ===")
    print(res.sort_values("ic5_mean", key=lambda s: s.abs(), ascending=False).head(12)
          .to_string(index=False, float_format=lambda x: f"{x:+.4f}"))

    # 残差 IC：控制 [money, flow_cap, n_urg] 后前 10 个 |ic1| 因子
    nurg_idx = NAMES.index("urgency_v1_ext_cluster_both_n_urg")
    print("\n=== 控制活跃度后残差 IC（|ic1| 前 10 因子，控制 money/flow_cap/n_urg）===")
    for _, r in res.head(10).iterrows():
        j = NAMES.index(r["name"])
        resid_ics = []
        for d, (idx, cols, arr) in days.items():
            f = arr[:, j]
            mo = money.loc[idx, cols].to_numpy(dtype=float)
            fc = flow_cap.loc[idx, cols].to_numpy(dtype=float)
            nu = arr[:, nurg_idx]
            g1 = gap1.loc[idx, cols].to_numpy(dtype=float)
            m = ~(np.isnan(f) | np.isnan(g1) | np.isnan(mo) | np.isnan(fc) | np.isnan(nu))
            if m.sum() < 100:
                continue
            X = np.column_stack([
                pd.Series(mo[m]).rank().to_numpy(),
                pd.Series(fc[m]).rank().to_numpy(),
                pd.Series(nu[m]).rank().to_numpy(),
            ])
            X = np.column_stack([np.ones(m.sum()), X])
            y = pd.Series(f[m]).rank().to_numpy()
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            resid = y - X @ beta
            resid_ics.append(spearman(resid, g1[m]))
        resid_ics = np.array([x for x in resid_ics if not np.isnan(x)])
        print(f"  {r['name'][:58]:60s} raw={r['ic1_mean']:+.4f}  resid={resid_ics.mean():+.4f}"
              f"  dir={(resid_ics > 0).mean():.2f}" if len(resid_ics) else f"  {r['name']} 样本不足")

    res.to_csv("/home/chenzongwei/rust_pyfunc/research_ext_ideas/sandbox_urg_cluster_ic.csv", index=False)
    print("\n已落盘 research_ext_ideas/sandbox_urg_cluster_ic.csv")


if __name__ == "__main__":
    main()
