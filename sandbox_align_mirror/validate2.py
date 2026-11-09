"""方向 3 补充验证：高分到底来自镜像，还是来自字母组成/价格档位（只读 Rust 产物）。"""
import os
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[k] = "1"
import json
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "/home/chenzongwei/.agents/skills/daily-data-reader")
import daily_data_reader as ddr


def load(p):
    d = json.load(open(p))
    return pd.DataFrame(np.array(d["vals"], dtype=float), index=d["codes"], columns=d["names"])


def rc(a, b):
    x = pd.concat([a, b], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if len(x) < 50 or x.iloc[:, 0].nunique() < 2 or x.iloc[:, 1].nunique() < 2:
        return np.nan
    return x.iloc[:, 0].rank().corr(x.iloc[:, 1].rank())


def neutral(y, X):
    """y 的秩对 X 各列的秩做回归，返回残差（只用于验证，不是因子合成）"""
    df = pd.concat([y.rename("y"), X], axis=1).replace([np.inf, -np.inf], np.nan).dropna().rank()
    A = np.column_stack([np.ones(len(df)), df.drop(columns="y").values])
    beta, *_ = np.linalg.lstsq(A, df["y"].values, rcond=None)
    return pd.Series(df["y"].values - A @ beta, index=df.index)


A = load("results/mirror_20241231.json")
B = load("results/mirror_20241230.json")
r = ddr.ret(start=20241220, end=20250110)
ret1 = r.loc[20250102].reindex(A.index)
ret0 = r.loc[20241231].reindex(B.index)
cu = ddr.close_u(start=20241220, end=20250110)
tr = ddr.tr(start=20241220, end=20250110)
ctrl1 = pd.DataFrame({"invprice": 1 / cu.loc[20241231].reindex(A.index), "tr": tr.loc[20241231].reindex(A.index), "logamt": A["audit_logamt_all"]})
ctrl0 = pd.DataFrame({"invprice": 1 / cu.loc[20241230].reindex(B.index), "tr": tr.loc[20241230].reindex(B.index), "logamt": B["audit_logamt_all"]})

print("== 1. 原始比对分有多少由字母组成决定（perletter 对 全打乱基准均值 = perletter-excess）==")
rows = []
for c in sorted({x.rsplit("_", 1)[0] for x in A.columns if x.startswith("mir_")}):
    p, ex = A[f"{c}_perletter"], A[f"{c}_excess"]
    null = p - ex
    x = pd.concat([p, null], axis=1, keys=["p", "n"]).dropna()
    ratio = (x["p"] - x["n"]).std() / x["p"].std()
    rows.append(dict(combo=c, alpha=c.split("_")[1], clock=c.split("_")[2], win=c.split("_")[3],
                     rank_p_vs_null=rc(p, null), numeric_resid_over_p=ratio,
                     stab_p=rc(p, B[f"{c}_perletter"]), stab_excess=rc(ex, B[f"{c}_excess"])))
R = pd.DataFrame(rows)
print(R.groupby("alpha")[["rank_p_vs_null", "numeric_resid_over_p", "stab_p", "stab_excess"]].median().round(3).to_string())
print(R.groupby("win")[["rank_p_vs_null", "numeric_resid_over_p", "stab_p", "stab_excess"]].median().round(3).to_string())

print("\n== 2. 榜首 price 字母表高 IC 的来源：与价格倒数、换手、成交额的秩相关，以及中性化后 IC ==")
tops = ["mir_price_od_all_meancent", "mir_price_od_all_perletter", "mir_price_od_all_bitsD", "mir_price_od_all_bitsR",
        "mir_int_vo_all_meancent", "mir_jump_vo_all_perletter", "mir_int_tm_all_perletter"]
for c in tops:
    n1, n0 = neutral(A[c], ctrl1), neutral(B[c], ctrl0)
    print(f"{c:34s} invprice={rc(A[c], ctrl1.invprice):+.3f} tr={rc(A[c], ctrl1.tr):+.3f} logamt={rc(A[c], ctrl1.logamt):+.3f} "
          f"IC原始={rc(A[c], ret1):+.3f}/{rc(B[c], ret0):+.3f} 中性后IC={rc(n1, ret1):+.3f}/{rc(n0, ret0):+.3f} 中性后稳定={rc(n1, n0):+.3f}")

print("\n== 3. 去掉字母组成后的“纯镜像”特征（覆盖>80%，按跨日稳定性排序）==")
pure = ("zseg", "zfull", "excess", "dzR", "dzC", "dzD", "dzmax", "signed", "buyfirst", "center", "asym", "armvol", "npeaks")
pr = []
for c in A.columns:
    if not c.startswith("mir_") or c.rsplit("_", 1)[1] not in pure:
        continue
    if A[c].notna().sum() < 0.8 * len(A):
        continue
    n1, n0 = neutral(A[c], ctrl1), neutral(B[c], ctrl0)
    pr.append(dict(col=c, feat=c.rsplit("_", 1)[1], win=c.split("_")[3], stab=rc(A[c], B[c]), stab_neu=rc(n1, n0),
                   ic1=rc(A[c], ret1), ic0=rc(B[c], ret0), ic1_neu=rc(n1, ret1), ic0_neu=rc(n0, ret0),
                   amt=rc(A[c], ctrl1.logamt)))
P = pd.DataFrame(pr).set_index("col")
P["same_neu"] = np.sign(P.ic1_neu) == np.sign(P.ic0_neu)
print("纯镜像候选总数", len(P), " 中性后两日 IC 同号占比", round(P.same_neu.mean(), 3))
print(P.groupby("feat")[["stab", "stab_neu"]].median().round(3).sort_values("stab_neu", ascending=False).to_string())
print(P.sort_values("stab_neu", ascending=False).head(15).round(3).to_string())
Q = P[P.same_neu].copy()
Q["abs"] = (Q.ic1_neu.abs() + Q.ic0_neu.abs()) / 2
print("\n中性后两日 IC 同号、按 |IC| 排序前 15")
print(Q.sort_values("abs", ascending=False).head(15).round(3).to_string())
print("|IC| 随机参照：两日同号比例 50%；纯镜像 |中性 IC| 中位", round(((P.ic1_neu.abs() + P.ic0_neu.abs()) / 2).median(), 4))

print("\n== 4. 代表因子两两截面秩相关（20241231）==")
reps = ["mir_int_vo_all_meancent", "mir_int_tm_all_gapfrac", "mir_int_vo_all_zseg", "mir_int_tm_all_buyfirst", "mir_jump_vo_pm_zfull", "naive_anticorr_all"]
reps = [c for c in reps if c in A.columns]
print(A[reps].rank().corr().round(3).to_string())
for c in reps:
    print(f"{c:30s} cov={int(A[c].notna().sum())} stab={rc(A[c], B[c]):+.3f} IC={rc(A[c], ret1):+.3f}/{rc(B[c], ret0):+.3f}")

with pd.ExcelWriter("results/validate2_summary.xlsx") as w:
    R.round(4).to_excel(w, sheet_name="composition", index=False)
    P.round(4).to_excel(w, sheet_name="pure_mirror")
print("已写 results/validate2_summary.xlsx")
