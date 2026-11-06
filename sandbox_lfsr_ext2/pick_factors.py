"""贪心去雷同 + 覆盖率过滤，输出拓展二候选名单与汇总。"""
import json, numpy as np, pandas as pd
O = "/hdd/user_home_unsafe/chenzongwei/sandbox_lfsr_ext2_out"
names = json.load(open(f"{O}/names_w1.json")); ix = {n: i for i, n in enumerate(names)}
keep = [l.strip() for l in open("keep_ext1.txt") if l.strip()]
summ = pd.read_parquet(f"{O}/summary.parquet").reindex(names)
C = np.load(f"{O}/abscorr.npy"); nanr = np.load(f"{O}/nanrate.npy")
score = summ["s5_g5_ir"].abs().fillna(0).values
bad = (nanr > 0.5) | (summ["cat"].values == "风格/计数") | np.isnan(summ["s5_g5_ir"].values)
order = [ix[k] for k in keep] + [j for j in np.argsort(-score) if names[j] not in keep and not bad[j]]
for th in (0.95, 0.9, 0.8):
    sel = []
    for j in order:
        if names[j] in keep or not sel or C[j, sel].max() <= th: sel.append(j)
    s = summ.iloc[sel]
    print(f"阈值 {th}: 保留 {len(sel)} 列；其中 ext1 原样 {len(keep)}，|IR|>0.3 有 {(s.s5_g5_ir.abs()>0.3).sum()}，>0.5 有 {(s.s5_g5_ir.abs()>0.5).sum()}")
    if th == 0.9:
        sel9 = sel
        print(s.groupby(["cat", "resid"]).size().unstack(fill_value=0).to_string())
names9 = [names[j] for j in sel9]
json.dump(names9, open(f"{O}/selected_names_th09.json", "w"), ensure_ascii=False)
# 新列和 ext1 保留因子的最大相关（看增量）
kidx = [ix[k] for k in keep] + [ix[k + "_r"] for k in keep]
new = [j for j in sel9 if names[j] not in keep]
mx = C[np.ix_(new, kidx)].max(1)
t = summ.iloc[new].assign(与ext1最大相关=mx, 缺失率=nanr[new])
print("\n新列与 ext1 保留因子的最大 |相关| 分布：", np.percentile(mx, [10, 50, 90]).round(2))
cols = ["cat", "w1_g5_ir", "s5_g5_ir", "s5_g1_ir", "s5_g5_ic", "与ext1最大相关", "缺失率"]
top = t[t["与ext1最大相关"] < 0.5].sort_values("s5_g5_ir", key=abs, ascending=False)[cols].head(25)
print("\n与 ext1 相关<0.5 的新列，按 |IR| 前 25："); print(top.round(3).to_string())
t[cols].to_parquet(f"{O}/selected_new_th09.parquet")
