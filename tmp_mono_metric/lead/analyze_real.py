"""把用户说的"空头更猛"拆成两个不同的量：幅度不对称 vs 秩贡献不对称。
r / gsd 列已经是 bp，不要再乘 1e4。
"""
import numpy as np
import pandas as pd

pd.set_option("display.width", 230)
d = pd.read_csv("/home/chenzongwei/rust_pyfunc/tmp_mono_metric/empirics/metrics_aug.csv")
R = d[[f"r{i}" for i in range(1, 11)]].to_numpy()
C = d[[f"c{i}" for i in range(1, 11)]].to_numpy()
G = d[[f"gsd{i}" for i in range(1, 11)]].to_numpy()
ic = d["ic"].to_numpy()
absic = np.abs(ic)
neg = ic < 0
Ro = np.where(neg[:, None], -R[:, ::-1], R)
Co = np.where(neg[:, None], -C[:, ::-1], C)
Go = np.where(neg[:, None], G[:, ::-1], G)
top = np.argsort(-absic)[:50]
mid = Ro[:, [4, 5]].mean(1)
short_drop = mid - Ro[:, 0]
long_rise = Ro[:, 9] - mid
ratio = short_drop / np.where(long_rise == 0, np.nan, long_rise)

print("=" * 104)
print("[Q1] 幅度不对称：空头半边落差 vs 多头半边落差（bp，方向已归正）")
print(f"  全样本 310:  空头落差中位 {np.nanmedian(short_drop):7.1f}bp   多头落差中位 {np.nanmedian(long_rise):7.1f}bp"
      f"   比值中位 {np.nanmedian(ratio):5.2f}   空头更大占 {(short_drop > long_rise).mean() * 100:.0f}%")
print(f"  |IC| 前 50:  空头落差中位 {np.nanmedian(short_drop[top]):7.1f}bp   多头落差中位 {np.nanmedian(long_rise[top]):7.1f}bp"
      f"   比值中位 {np.nanmedian(ratio[top]):5.2f}   空头更大占 {(short_drop[top] > long_rise[top]).mean() * 100:.0f}%")

print("=" * 104)
print("[Q2] 秩贡献不对称：IC 的阶梯质量有多少来自空头半边（方向归正后）")
tot = Co.sum(1)
sh = np.where(tot != 0, Co[:, :5].sum(1) / tot, np.nan)
q = np.percentile(sh[top], [5, 25, 50, 75, 95])
print(f"  全样本: 中位 {np.nanmedian(sh):.3f}   空头半边贡献更大的占 {(sh > 0.5).mean() * 100:.0f}%")
print(f"  |IC| 前 50: 中位 {np.nanmedian(sh[top]):.3f}  5%-95% 分位 "
      f"[{q[0]:.3f}, {q[1]:.3f}, {q[2]:.3f}, {q[3]:.3f}, {q[4]:.3f}]")
print(f"             空头半边贡献更大的占 {(sh[top] > 0.5).mean() * 100:.0f}%")

print("=" * 104)
print("[Q3] 组内离散度：第 1 组（空头）比第 10 组（多头）更整齐吗（bp）")
print(f"  全样本: gsd1 中位 {np.median(Go[:, 0]):7.1f}   gsd10 中位 {np.median(Go[:, 9]):7.1f}"
      f"   gsd1<gsd10 占 {(Go[:, 0] < Go[:, 9]).mean() * 100:.0f}%")
print(f"  |IC|前50: gsd1 中位 {np.median(Go[top, 0]):7.1f}   gsd10 中位 {np.median(Go[top, 9]):7.1f}"
      f"   gsd1<gsd10 占 {(Go[top, 0] < Go[top, 9]).mean() * 100:.0f}%")
print("  → 空头组内部反而更散（离散度更大），不是更整齐")

print("=" * 104)
print("[Q4] 两端的极端 3 组谁在撑 IC")
cs3, cl3 = Co[:, :3].sum(1), Co[:, 7:].sum(1)
print(f"  corr(|IC|, 空头3组) = {np.corrcoef(absic, cs3)[0, 1]:+.3f}    "
      f"corr(|IC|, 多头3组) = {np.corrcoef(absic, cl3)[0, 1]:+.3f}")
print(f"  |IC|前50: 空头3组贡献中位 {np.median(cs3[top]):.4f}    多头3组贡献中位 {np.median(cl3[top]):.4f}")

print("=" * 104)
print("[Q5] 高 IC 因子里哪一侧的阶梯更容易破位（方向归正后）")
lsh, llo, ssm = d["lsh_or"].to_numpy(), d["llo_or"].to_numpy(), d["ssm_or"].to_numpy()
lam = d["lam_or"].to_numpy()
for K, lab in [(50, "|IC| 前 50"), (100, "|IC| 前 100"), (310, "全样本")]:
    idx = np.argsort(-absic)[:K]
    print(f"  {lab:<12} 空头半边完美单调 {(lsh[idx] >= 0.99).mean() * 100:5.0f}%   "
          f"多头半边完美单调 {(llo[idx] >= 0.99).mean() * 100:5.0f}%   "
          f"多头半边破位(<0.5) {(llo[idx] < 0.5).mean() * 100:5.0f}%   "
          f"空头半边破位 {(lsh[idx] < 0.5).mean() * 100:5.0f}%")

print("=" * 104)
print("[Q6] |IC| 前 50 里破位的是哪一侧（前 8 个）")
idx = top[(llo[top] < 0.5) | (lsh[top] < 0.5)]
for i in idx[:8]:
    side = "多头破位" if llo[i] < 0.5 else "空头破位"
    print(f"  {side}  |IC|={absic[i]:.4f}  空头={lsh[i]:+.2f} 多头={llo[i]:+.2f} "
          f"SSM={ssm[i]:+.2f}  曲线(bp) " + " ".join(f"{x:6.1f}" for x in Ro[i]))
print(f"  合计 {len(idx)}/50 个，其中多头破位 {(llo[idx] < 0.5).sum()} 个，空头破位 {(lsh[idx] < 0.5).sum()} 个")

print("=" * 104)
print("[Q7] 换指标会不会换出一批货")
for nm, v in [("Lambda", lam), ("SSM", ssm), ("|Gamma|", np.abs(d['gamma'].to_numpy()))]:
    for K in (20, 50):
        ti, tv = set(np.argsort(-absic)[:K]), set(np.argsort(-v)[:K])
        print(f"  |IC| 前 {K:<3} ∩ {nm:<8} 前 {K:<3} = {len(ti & tv):>3}/{K}")
print(f"  |IC| 前 50 里 SSM 低于噪声 95% 分位(0.186) 的: {(ssm[top] < 0.186).sum()}/50")

print("=" * 104)
print("[Q8] 例子：|IC| 最高但阶梯指标差 / |IC| 中等但阶梯完美")
show = ["factor", "ic", "lam_or", "ssm_or", "gamma"] + [f"r{i}" for i in range(1, 11)]
sel = np.argsort(-absic)[:6]
print("\n|IC| 最高的 6 个：")
print(d.iloc[sel][show].to_string(index=False, float_format=lambda x: f"{x:7.2f}"))
print("\nSSM 最高且 |IC| 在前 100 的 6 个：")
cand = [i for i in np.argsort(-ssm) if i in set(np.argsort(-absic)[:100])][:6]
print(d.iloc[cand][show].to_string(index=False, float_format=lambda x: f"{x:7.2f}"))
