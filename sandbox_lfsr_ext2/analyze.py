"""拓展二沙箱评估汇总：按类别比较 IC、残差版 vs 原始版、5 日合并 vs 5 日平均，并做去雷同。
只读 Rust 产物和 eval_ic.py 的 IC 表，不合成因子。
"""
import json, re, numpy as np, pandas as pd
from scipy.stats import rankdata

O = "/hdd/user_home_unsafe/chenzongwei/sandbox_lfsr_ext2_out"
keep = [l.strip() for l in open("keep_ext1.txt") if l.strip()]
names = json.load(open(f"{O}/names_w1.json"))
dates = [int(x) for x in open(f"{O}/dates.txt").read().split()]


def stat(tag, g):
    t = pd.read_parquet(f"{O}/ic_{tag}_{g}.parquet")
    m, s = t.mean(), t.std()
    return pd.DataFrame({"ic": m, "ir": m / s, "ndays": t.notna().sum()})


S = {f"{tag}_{g}": stat(tag, g) for tag in ["w1", "s5", "p5"] for g in ["g1", "g5"]}


def cat(n):
    b = n[:-2] if n.endswith("_r") else n
    if b in keep: return "ext1保留"
    if b.startswith("ctl_") or b in ("nsnap", "ntrade"): return "风格/计数"
    if b.startswith("s3_"): return "三态流"
    if b.startswith("tb_"): return "逐笔门槛流"
    if b.startswith("b_"):
        s = b.split("_")[1]
        mk = s in ("spdq50", "spdq80", "spdq95", "sprq50", "sprq80", "depq50", "depq80", "aimbq80", "bigmv", "btr95", "btr80")
        o = b.split("_")[-1]
        kind = "预测力" if o in ("e60", "e300", "end", "e60s") else "结构"
        return f"快照{'全市场门槛' if mk else '方向'}流-{kind}"
    if re.match(r"X\d+_c\d$", b): return "三元-8格频率"
    if re.match(r"X\d+_m", b): return "三元-8格条件收益"
    if re.match(r"X\d+_(oos|patH)", b): return "三元-样本外/熵"
    if re.match(r"X\d+_f\d+_(ac1|lks|lksx)$", b): return "三元函数-结构"
    if re.match(r"X\d+_f\d+_(e60|end)$", b): return "三元函数-预测力"
    return "其他"


df = pd.DataFrame(index=S["w1_g5"].index)
for k, v in S.items():
    df[k + "_ic"] = v["ic"]; df[k + "_ir"] = v["ir"]
df["cat"] = [cat(n) for n in df.index]
df["resid"] = [n.endswith("_r") for n in df.index]
df.to_parquet(f"{O}/summary.parquet")

# ---- 1) 类别概览（gap5，5 日平均口径，|ICIR|）----
def q(x): return pd.Series({"列数": len(x), "|IR|中位": x.abs().median(), "|IR|90分位": x.abs().quantile(.9),
                            "|IR|>0.3个数": (x.abs() > 0.3).sum(), "|IR|>0.5个数": (x.abs() > 0.5).sum()})
print("\n== 按类别：gap5、5日平均后 ICIR（原始版/残差版）==")
print(df.groupby(["cat", "resid"])["s5_g5_ir"].apply(q).unstack().round(3).to_string())

# ---- 2) 残差 vs 原始（同一列）----
base = [n for n in df.index if not n.endswith("_r") and n + "_r" in df.index]
cmp = pd.DataFrame({"raw": df.loc[base, "s5_g5_ir"].values, "res": df.loc[[b + "_r" for b in base], "s5_g5_ir"].values,
                    "cat": df.loc[base, "cat"].values}, index=base)
print("\n== 风格剔除前后 |ICIR|(gap5, 5日平均) 中位数 ==")
print(cmp.groupby("cat").apply(lambda x: pd.Series({"原始": x.raw.abs().median(), "残差": x.res.abs().median(),
      "残差更好占比": (x.res.abs() > x.raw.abs()).mean()})).round(3).to_string())
print("\n== ext1 保留因子：原始 vs 残差（gap5, 5日平均 ICIR）==")
print(cmp.loc[[k for k in keep if k in cmp.index], ["raw", "res"]].round(3).sort_values("raw").to_string())

# ---- 3) 5 日合并 vs 5 日平均（同 20 天）----
p5 = pd.read_parquet(f"{O}/ic_p5_g5.parquet")
s5 = pd.read_parquet(f"{O}/ic_s5_g5.parquet").loc[p5.index]
common = [c for c in p5.columns if c in s5.columns]
pm, sm = p5[common].mean() / p5[common].std(), s5[common].mean() / s5[common].std()
pc = pd.DataFrame({"pool": pm, "smooth": sm, "cat": [cat(c) for c in common]})
print("\n== 5日合并 vs 5日平均（同 20 天，gap5 ICIR）==")
print(pc.groupby("cat").apply(lambda x: pd.Series({"合并|IR|中位": x.pool.abs().median(), "平均|IR|中位": x.smooth.abs().median(),
      "合并更好占比": (x.pool.abs() > x.smooth.abs()).mean(), "Top50合并": x.pool.abs().nlargest(50).mean(),
      "Top50平均": x.smooth.abs().nlargest(50).mean()})).round(3).to_string())
pc.to_parquet(f"{O}/pool_vs_smooth.parquet")
