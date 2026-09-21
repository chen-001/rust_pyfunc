"""读 metrics.csv，算 A-E 的确切数字。

方向归正：若 IC<0，则把因子取负。此时组号反转（-f 的第 d 组 = f 的第 11-d 组），
组均收益 r'_d = r_{11-d}（收益本身不变号），分解贡献 c'_d = -c_{11-d}（因子秩百分位变号）。
"""
import os, numpy as np, pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(HERE, "metrics.csv"))
R = df[[f"r{i}" for i in range(1, 11)]].to_numpy(float)
C = df[[f"c{i}" for i in range(1, 11)]].to_numpy(float)
G = df[[f"gsd{i}" for i in range(1, 11)]].to_numpy(float)
neg = (df["ic"].to_numpy() < 0)
Ror = np.where(neg[:, None], R[:, ::-1], R)
Cor = np.where(neg[:, None], -C[:, ::-1], C)
Gor = np.where(neg[:, None], G[:, ::-1], G)

d = np.diff(Ror, axis=1)
df["lam_or"] = (Ror[:, 9] - Ror[:, 0]) / np.abs(d).sum(axis=1)
df["lsh_or"] = (Ror[:, 4] - Ror[:, 0]) / np.abs(d[:, :4]).sum(axis=1)
df["llo_or"] = (Ror[:, 9] - Ror[:, 5]) / np.abs(d[:, 5:]).sum(axis=1)
df["ssm_or"] = np.minimum(df.lsh_or, df.llo_or)
# lead 补充 1：三段口径（中间台阶 d[4] 也纳入）
df["ssm3_or"] = np.minimum(np.minimum(df.lam_or, df.lsh_or), df.llo_or)
df["ssm3"] = np.minimum(np.minimum(df["lambda"], df.lambda_short), df.lambda_long)
# lead 补充 2：阶梯部分占 IC 的比例
df["csum"] = C.sum(axis=1)
df["csum_or"] = Cor.sum(axis=1)
df["absic"] = df["ic"].abs()
df["ladder_ratio"] = df.csum_or / df.absic
df["within"] = 1.0 - df.ladder_ratio
df["share_ic2"] = Cor[:, :5].sum(axis=1) / Cor.sum(axis=1)
df["c_s3"] = Cor[:, :3].sum(axis=1)
df["c_l3"] = Cor[:, 7:].sum(axis=1)
assert np.allclose(df.share_ic, df.share_ic2)

L = []
p = L.append
p(f"样本：{len(df)} 个因子，来自 hm100-hm105；每个因子 {int(df.nd.median())} 个交易日（中位数），"
  f"共 {int(df.nd.sum())} 个（因子,日）样本。IC>0 的因子占 {100*(df.ic>0).mean():.1f}%。")
p(f"|IC| 分位 [50,75,90,100] = {np.round(np.percentile(df.absic,[50,75,90,100]),4)}")
p(f"分解残差 |IC - sum(c_d)|：均值 {df.resid_mean.abs().mean():.6f}，最大 {df.resid_mean.abs().max():.6f}；"
  f"相对 |IC| 均值的比例 {df.resid_mean.abs().mean()/df.absic.mean()*100:.2f}%")

p("\n=== A 镜像对称性 ===")
p(f"全局最大 |IC(-f) + IC(f)| = {df.max_mirror_dev.max()}")
p(f"逐日偏差 > 1e-12 的（因子,日）样本数 = {int(df.nmir_bad.sum())} / {int(df.nd.sum())}")
p(f"每因子最大偏差取值集合 = {np.unique(df.max_mirror_dev.to_numpy())}")
p(f"空组被跳过的因子 {int((df.nbad>0).sum())} 个，跳过天数合计 {int(df.nbad.sum())}")

p("\n=== B 前 50（按 |IC|）===")
top = df.nlargest(50, "absic")
for col in ["share", "share_ic"]:
    v = top[col].to_numpy(float)
    t = stats.ttest_1samp(v, 0.5)
    p(f"{col}: mean={v.mean():.4f} median={np.median(v):.4f} std={v.std(ddof=1):.4f} "
      f"t={t.statistic:.2f} p={t.pvalue:.2e} >0.5 的个数={int((v>0.5).sum())}/50 "
      f"25/75 分位={np.round(np.percentile(v,[25,75]),4)}")
v = df.share_ic.to_numpy(float)
p(f"全样本 share_ic: mean={v.mean():.4f} median={np.median(v):.4f} 25/75={np.round(np.percentile(v,[25,75]),4)} "
  f">0.5 的个数={int((v>0.5).sum())}/{len(v)}")
p(f"top50 中 IC<0 的个数 = {int((top.ic<0).sum())}")
p(f"top50 归正后 lambda_short 中位数={top.lsh_or.median():.4f} 均值={top.lsh_or.mean():.4f}")
p(f"top50 归正后 lambda_long  中位数={top.llo_or.median():.4f} 均值={top.llo_or.mean():.4f}")
p(f"top50 归正后 lsh>llo 的个数 = {int((top.lsh_or>top.llo_or).sum())}/50")
p(f"top50 raw lambda_short 中位数={top.lambda_short.median():.4f} raw lambda_long 中位数={top.lambda_long.median():.4f}")
p(f"全样本归正后 lambda_short 中位数={df.lsh_or.median():.4f} lambda_long 中位数={df.llo_or.median():.4f} "
  f"lsh>llo 个数={int((df.lsh_or>df.llo_or).sum())}/{len(df)}")

p("\n=== C 相关性 ===")
for a, b in [("ic", "lambda"), ("ic", "ssm"), ("ic", "gamma"),
             ("ic", "c_s3"), ("ic", "c_l3"),
             ("absic", "lam_or"), ("absic", "ssm_or"),
             ("absic", "c_s3"), ("absic", "c_l3"),
             ("ic", "share_ic"), ("absic", "share_ic"), ("absic", "ladder_ratio")]:
    x = df[a].to_numpy(float); y = df[b].to_numpy(float)
    m = np.isfinite(x) & np.isfinite(y)
    p(f"corr({a},{b}) pearson={stats.pearsonr(x[m],y[m]).statistic:.4f} "
      f"spearman={stats.spearmanr(x[m],y[m]).statistic:.4f} n={int(m.sum())}")

p("\n=== B' 阶梯占比（lead 补充）===")
p(f"ladder_ratio 分位 [5,25,50,75,95] = {np.round(np.nanpercentile(df.ladder_ratio,[5,25,50,75,95]),4)}")
p(f"top50(|IC|) ladder_ratio: 中位数={top.ladder_ratio.median():.4f} 均值={top.ladder_ratio.mean():.4f} "
  f"最小={top.ladder_ratio.min():.4f} >0.9 的个数={int((top.ladder_ratio>0.9).sum())}/50")
good = top[top.ladder_ratio > 0.9]
p(f"top50 中 ladder_ratio>0.9 的子集（{len(good)} 个）share_ic: 中位数={good.share_ic.median():.4f} "
  f"均值={good.share_ic.mean():.4f} >0.5 个数={int((good.share_ic>0.5).sum())}/{len(good)}")
p(f"top50 中 ladder_ratio<=0.9 的（{50-len(good)} 个）share_ic: 中位数={top[top.ladder_ratio<=0.9].share_ic.median():.4f} "
  f"均值={top[top.ladder_ratio<=0.9].share_ic.mean():.4f}")
p(f"ssm3_or 分位 [5,25,50,75,95] = {np.round(np.nanpercentile(df.ssm3_or,[5,25,50,75,95]),3)}; "
  f"top50 ssm3_or 中位数={top.ssm3_or.median():.4f} 均值={top.ssm3_or.mean():.4f}")
p(f"ssm_or 与 ssm3_or 差异：|ssm3-ssm| 中位数={np.nanmedian((df.ssm3_or-df.ssm_or).abs()):.4f} "
  f"最大={np.nanmax((df.ssm3_or-df.ssm_or).abs()):.4f}; ssm3<ssm 的因子数={int((df.ssm3_or<df.ssm_or-1e-9).sum())}")
p(f"全样本 ladder_ratio<=0.5 的因子数={int((df.ladder_ratio<=0.5).sum())}；"
  f"这些因子 share_ic 分位[5,50,95]={np.round(np.nanpercentile(df[df.ladder_ratio<=0.5].share_ic,[5,50,95]),3)}")

p(f"median |c_short3| = {df.c_s3.abs().median():.5f}  median |c_long3| = {df.c_l3.abs().median():.5f}  "
  f"median 比值(|c_s3|/|c_l3|) = {np.median(df.c_s3.abs()/df.c_l3.abs()):.4f}")
p(f"median |c1|={Cor[:,0].__abs__().mean():.5f} ... 各组 |c_d| 均值 = {np.round(np.abs(Cor).mean(axis=0),5)}")

p("\n=== D 组内离散度 ===")
p(f"归正后各组 r5 组内 std 均值(bp) = {np.round(np.nanmean(Gor,axis=0),1)}")
dif = Gor[:, 0] - Gor[:, 9]
p(f"归正后 空头组(第1组) - 多头组(第10组): 均值={np.nanmean(dif):.1f}bp 中位数={np.nanmedian(dif):.1f}bp "
  f"空头更离散的因子数={int(np.nansum(dif>0))}/{int(np.isfinite(dif).sum())}")
t = stats.ttest_rel(Gor[:, 0], Gor[:, 9], nan_policy="omit")
p(f"配对 t = {float(t.statistic):.2f}, p = {t.pvalue:.3e}; 比值中位数 = {np.nanmedian(Gor[:,0]/Gor[:,9]):.4f}")
p(f"未归正(raw)各组 std 均值(bp) = {np.round(np.nanmean(G,axis=0),1)}; "
  f"raw gsd1-gsd10 均值={np.nanmean(G[:,0]-G[:,9]):.1f} 空头更离散个数={int(np.nansum(G[:,0]>G[:,9]))}")
p(f"组内残差占比: mean(|resid|)/mean(|IC|) = {df.resid_mean.abs().mean()/df.absic.mean()*100:.2f}%")

p("\n=== E 例子 ===")
q80 = df.absic.quantile(0.80)
cand = df[df.absic >= q80].nsmallest(8, "ssm_or")
cols = ["factor", "ic", "ssm_or", "lam_or", "lsh_or", "llo_or"] + [f"r{i}" for i in range(1, 11)]
p(f"-- IC 前 20%（|IC|>={q80:.4f}）里 SSM 最差的 8 个 --")
p(cand[cols].round(4).to_string(index=False))
mid = df[(df.absic >= df.absic.quantile(0.40)) & (df.absic <= df.absic.quantile(0.60))]
best = mid.nlargest(6, "ssm_or")
p(f"-- IC 中等（40-60 分位，|IC| 在 {mid.absic.min():.4f}~{mid.absic.max():.4f}）里 SSM 最好的 6 个 --")
p(best[cols].round(4).to_string(index=False))
p(f"ssm_or 分位 [5,25,50,75,95] = {np.round(np.nanpercentile(df.ssm_or,[5,25,50,75,95]),3)}")
p(f"lam_or 分位 [5,25,50,75,95] = {np.round(np.nanpercentile(df.lam_or,[5,25,50,75,95]),3)}")
p(f"gamma 分位 [5,25,50,75,95] = {np.round(np.nanpercentile(df.gamma,[5,25,50,75,95]),3)}")

out = "\n".join(L)
print(out)
with open(os.path.join(HERE, "analysis_out.txt"), "w") as fh:
    fh.write(out + "\n")
df.to_csv(os.path.join(HERE, "metrics_aug.csv"), index=False)
