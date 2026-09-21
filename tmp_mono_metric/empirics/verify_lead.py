"""核对 lead 给的三个数 + ssm3 分档 + 口径稳健性（gap1 / 分时段）。"""
import os, numpy as np, pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))


def load(fn):
    df = pd.read_csv(os.path.join(HERE, fn))
    R = df[[f"r{i}" for i in range(1, 11)]].to_numpy(float)
    C = df[[f"c{i}" for i in range(1, 11)]].to_numpy(float)
    G = df[[f"gsd{i}" for i in range(1, 11)]].to_numpy(float)
    neg = df.ic.to_numpy() < 0
    Ror = np.where(neg[:, None], R[:, ::-1], R)
    Cor = np.where(neg[:, None], -C[:, ::-1], C)
    Gor = np.where(neg[:, None], G[:, ::-1], G)
    d = np.diff(Ror, axis=1)
    df["absic"] = df.ic.abs()
    df["lam_or"] = (Ror[:, 9] - Ror[:, 0]) / np.abs(d).sum(axis=1)
    df["lsh_or"] = (Ror[:, 4] - Ror[:, 0]) / np.abs(d[:, :4]).sum(axis=1)
    df["llo_or"] = (Ror[:, 9] - Ror[:, 5]) / np.abs(d[:, 5:]).sum(axis=1)
    df["ssm_or"] = np.minimum(df.lsh_or, df.llo_or)
    df["ssm3_or"] = np.minimum(df.ssm_or, df.lam_or)
    df["share_ic"] = Cor[:, :5].sum(axis=1) / Cor.sum(axis=1)
    df["ladder_ratio"] = Cor.sum(axis=1) / df.absic
    df["drop_short"] = np.abs(d[:, :4]).sum(axis=1)
    df["drop_long"] = np.abs(d[:, 5:]).sum(axis=1)
    df["drop_ratio"] = df.drop_short / df.drop_long
    df["gsd_short"] = Gor[:, 0]; df["gsd_long"] = Gor[:, 9]
    return df, Ror, Cor, Gor


df, Ror, Cor, Gor = load("metrics.csv")
top = df.nlargest(50, "absic")
print("=== lead 的三个数（|IC| 前 50）===")
print(f"1) 空头半边落差/多头半边落差: 中位数={top.drop_ratio.median():.3f} "
      f"均值={top.drop_ratio.mean():.3f} 空头更大占比={100*(top.drop_ratio>1).mean():.0f}% "
      f"(分母 d_long=0 的个数={int((top.drop_long==0).sum())})")
print(f"   绝对落差(bp): 空头半边 sum|Δ1..Δ4| 中位={top.drop_short.median():.1f}, "
      f"多头半边 sum|Δ6..Δ9| 中位={top.drop_long.median():.1f}")
print(f"2) share_ic: 中位数={top.share_ic.median():.4f} >0.5 占比={100*(top.share_ic>0.5).mean():.0f}%")
print(f"3) 半边完美单调: 空头 lsh_or==1 占比={100*(top.lsh_or>=0.99999).mean():.0f}% "
      f"多头 llo_or==1 占比={100*(top.llo_or>=0.99999).mean():.0f}%")
print(f"   全样本: 空头={100*(df.lsh_or>=0.99999).mean():.0f}% 多头={100*(df.llo_or>=0.99999).mean():.0f}%")

print("\n=== ssm3_or 分档 ===")
for lo, hi, n_ in [(0.07, 9, None), (0.05, 0.07, None), (0.0, 0.03, None)]:
    s = df[(df.absic >= lo) & (df.absic < hi)]
    print(f"|IC| in [{lo},{hi}) n={len(s)}: ssm3_or 中位={s.ssm3_or.median():.3f} "
          f"ssm3>=0.99 占比={100*(s.ssm3_or>=0.99).mean():.0f}% ssm3<0.5 占比={100*(s.ssm3_or<0.5).mean():.0f}% "
          f"lam_or 中位={s.lam_or.median():.3f}")
print("ssm3_or 恰好=1 的因子数:", int((df.ssm3_or >= 0.99999).sum()), "/", len(df))
print("ssm_or 与 ssm3_or 不等的因子数:", int((df.ssm3_or < df.ssm_or - 1e-9).sum()))
print("ladder_ratio<=0.9 的因子数:", int((df.ladder_ratio <= 0.9).sum()),
      " 最小 ladder_ratio:", round(float(df.ladder_ratio.min()), 4))

print("\n=== 口径稳健性 ===")
rows = []
for fn, tag in [("metrics.csv", "r5 2016-2025"), ("metrics_gap1.csv", "r1 2016-2025"),
                ("metrics_p1.csv", "r5 2016-2020"), ("metrics_p2.csv", "r5 2021-2025")]:
    d2, _, C2, G2 = load(fn)
    t2 = d2.nlargest(50, "absic")
    rows.append(dict(口径=tag, n=len(d2), IC中位=round(d2.absic.median(), 4),
                     top50_share中位=round(t2.share_ic.median(), 4),
                     top50_share_gt50=f"{100*(t2.share_ic>0.5).mean():.0f}%",
                     全样本share中位=round(d2.share_ic.median(), 4),
                     gsd空头减多头=round(float(np.nanmean(G2[:, 0] - G2[:, 9])), 1),
                     空头更离散占比=f"{100*np.nanmean(G2[:, 0] > G2[:, 9]):.0f}%",
                     ladder中位=round(d2.ladder_ratio.median(), 4)))
print(pd.DataFrame(rows).to_string(index=False))

print("\n=== 分 folder（r5 全期）===")
g = df.groupby("folder").agg(n=("ic", "size"), IC中位=("absic", "median"),
                             share中位=("share_ic", "median"),
                             ladder中位=("ladder_ratio", "median"))
print(g.round(4).to_string())
print("\n=== 分年 share_ic ===")
print("(需要分年重算，见 run_variant 输出)")
