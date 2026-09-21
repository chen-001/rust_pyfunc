"""对比 SSM 选出 vs |IC| 选出的因子：中性化 10 组收益的形状与幅度。

数据源：fulltest 产出的 neu 组收益 CSV（逐日），与引擎算 SSM 用的组收益同源。
"""
import numpy as np
import pandas as pd

SUM = "/nas197/user_home_unsafe/chenzongwei/hm100_ssm_tail_v4/metrics/summary_neu_gap5_candidates.parquet"
SEL = "/nas197/user_home_unsafe/chenzongwei/hm100_ssm_tail_v4/selected/gap5_selected.parquet"
NEU = "/nas197/user_home_unsafe/chenzongwei/factor_summary/hm100_ssm_1st_son_neu"

pd.set_option("display.width", 250)


def seg(r, i, j):
    s = np.asarray(r, float)[i - 1:j]
    d = np.diff(s)
    t = np.abs(d).sum()
    return (s[-1] - s[0]) / t if t > 0 else 0.0


def ssm(r):
    r = np.asarray(r, float).copy()
    if r[9] < r[0]:
        r = r[::-1]
    return min(seg(r, 1, 10), seg(r, 1, 5), seg(r, 6, 10), seg(r, 1, 4), seg(r, 7, 10))


def group_means(name, horizon="5"):
    p = f"{NEU}/{name}/{horizon}/group_return_{name}.csv"
    d = pd.read_csv(p, index_col=0)
    return d.to_numpy().mean(axis=0) * 1e4          # 时间平均，单位 bp


summ = pd.read_parquet(SUM)
sel = pd.read_parquet(SEL)["factor_name"].tolist()
top_ic = summ.assign(a=summ.IC_mean.abs()).nlargest(len(sel), "a")["factor_name"].tolist()

for tag, names in [("SSM 选出的 35 个", sel), ("|IC| 选出的前 35 个", top_ic)]:
    rows = []
    for n in names:
        try:
            r = group_means(n)
        except FileNotFoundError:
            continue
        s = summ.loc[summ.factor_name == n].iloc[0]
        rows.append(dict(factor=n[:58], ic=s.IC_mean, SSM_parquet=s.SSM, SSM_csv=ssm(r),
                         spread=r[9] - r[0], short_drop=r[4] - r[0], long_rise=r[9] - r[5],
                         r1=r[0], r5=r[4], r6=r[5], r10=r[9]))
    d = pd.DataFrame(rows)
    print("=" * 118)
    print(f"{tag}   （能读到组收益 CSV 的 {len(d)} 个）")
    print(f"  |IC| 中位 {d.ic.abs().median():.4f}   多空价差(r10-r1) 中位 {d.spread.median():7.1f} bp"
          f"   空头段落差 {d.short_drop.median():7.1f} bp   多头段涨幅 {d.long_rise.median():7.1f} bp")
    print(f"  SSM(parquet) 中位 {d.SSM_parquet.median():.3f}   SSM(用 CSV 组收益复算) 中位 {d.SSM_csv.median():.3f}"
          f"   两者最大偏差 {(d.SSM_parquet - d.SSM_csv).abs().max():.2e}")
    print(f"  多空价差 < 20bp 的占 {(d.spread < 20).mean() * 100:.0f}%   < 50bp 的占 {(d.spread < 50).mean() * 100:.0f}%")
    print(d[["ic", "SSM_parquet", "spread", "r1", "r5", "r6", "r10"]].describe().loc[
        ["mean", "50%", "min", "max"]].to_string())
    print()
    print("  前 8 个的曲线(bp)：")
    for _, x in d.head(8).iterrows():
        print(f"    ic={x.ic:+.4f} SSM={x.SSM_parquet:+.3f} 价差={x.spread:6.1f}  "
              f"[{x.r1:7.1f} {x.r5:7.1f} {x.r6:7.1f} {x.r10:7.1f}]  {x.factor}")
    d.to_csv(f"/home/chenzongwei/rust_pyfunc/tmp_mono_metric/lead/compare_{tag[:3].strip()}.csv", index=False)
