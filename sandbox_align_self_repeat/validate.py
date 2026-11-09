"""方向 2 自比对：只在 Rust 产出的最终因子值上做探索验证，不做任何因子合成。

运行：python validate.py
产物：results/validate_columns.xlsx（每列的覆盖、分布、稳定性、次日相关、与朴素对照的相关）
"""
import os

for k in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[k] = "1"
import json
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/chenzongwei/.agents/skills/daily-data-reader")
import daily_data_reader as ddr

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def load(name):
    d = json.load(open(os.path.join(R, name)))
    return pd.DataFrame(np.array(d["vals"], dtype=float), index=d["codes"], columns=d["names"]), d


def spearman(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 30:
        return np.nan, int(m.sum())
    x, y = pd.Series(a[m]).rank().values, pd.Series(b[m]).rank().values
    if x.std() == 0 or y.std() == 0:
        return np.nan, int(m.sum())
    return float(np.corrcoef(x, y)[0, 1]), int(m.sum())


def cond_resid_ratio(a, b, bins=10):
    """数值角度：A 对 B 分箱取条件均值，残差离散度 / A 离散度。"""
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 100 or a[m].std() == 0:
        return np.nan
    q = pd.qcut(pd.Series(b[m]).rank(method="first"), bins, labels=False)
    s = pd.Series(a[m])
    return float((s - s.groupby(q.values).transform("mean")).std() / s.std())


d31, meta = load("d20241231.json")
d30, _ = load("d20241230.json")
shf, _ = load("d20241231_shuffle.json")
mkv, _ = load("d20241231_markov.json")
print("shape", d31.shape, d30.shape, shf.shape, mkv.shape, "symbol_map", meta["symbol_map"], meta["symbol_map_fnv"])

ret = ddr.ret(20241231, 20250102)
print("ret index", ret.index[:2].tolist(), "cols", list(ret.columns[:3]))
r = ret.loc[20250102]
code_map = {str(c).split(".")[0][:6]: c for c in r.index}
next_ret = pd.Series({c: r.get(code_map.get(c), np.nan) for c in d31.index}).astype(float)
print("next-day ret non-nan", next_ret.notna().sum(), "date", 20250102)

common = d31.index.intersection(d30.index)
A, P = d31.loc[common].values, d30.loc[common].values
log_n = np.log1p(np.array(meta["n_events"], dtype=float))
log_n_df = pd.DataFrame(log_n, index=meta["codes"], columns=["full", "pm", "tail30"])
idx_of = {c: i for i, c in enumerate(d31.columns)}

rows = []
for j, c in enumerate(d31.columns):
    a = d31.iloc[:, j].values
    fin = a[np.isfinite(a)]
    stab, _ = spearman(A[:, j], P[:, j])
    ic, ic_n = spearman(a, next_ret.reindex(d31.index).values)
    sa = mkv.iloc[:, j].reindex(d31.index).values
    sic, _ = spearman(sa, next_ret.reindex(d31.index).values)
    m = re.match(r"^(?P<seq>.+?)__(?P<win>full|pm|tail30)__(?P<feat>[^_].*)$", c)
    best_base, best_ln = np.nan, np.nan
    if m and not c.startswith("cmp_") and "base_" not in c:
        for b in ["lag1_match", "ac_excess_max", "lz78_norm", "run_mean", "entropy"]:
            k = f"{m['seq']}__{m['win']}__base_{b}" + ("__nrank" if c.endswith("__nrank") else "")
            if k in idx_of:
                v, _ = spearman(a, d31.iloc[:, idx_of[k]].values)
                if np.isfinite(v) and not (np.abs(v) <= np.abs(best_base) if np.isfinite(best_base) else False):
                    best_base = v
        best_ln, _ = spearman(a, log_n_df[m["win"]].reindex(d31.index).values)
    rows.append(
        {
            "col": c,
            "seq": m["seq"] if m else "",
            "win": m["win"] if m else "",
            "feat": m["feat"] if m else "",
            "coverage": len(fin),
            "zero_share": float((fin == 0).mean()) if len(fin) else np.nan,
            "p5": float(np.percentile(fin, 5)) if len(fin) else np.nan,
            "p50": float(np.percentile(fin, 50)) if len(fin) else np.nan,
            "p95": float(np.percentile(fin, 95)) if len(fin) else np.nan,
            "stability_1230_1231": stab,
            "ic_20250102": ic,
            "ic_n": ic_n,
            "ic_markov_null": sic,
            "max_corr_with_naive": best_base,
            "corr_with_log_n": best_ln,
        }
    )
S = pd.DataFrame(rows)

sel = S[~S.col.str.contains("base_")]
print("\n==== 空模型：真实 vs 字母打乱 vs 马尔可夫（cov 中位数，三时段） ====")
for win in ["full", "pm", "tail30"]:
    tb = []
    for q in sorted(set(sel[(sel.feat == "cov")].seq)):
        col = f"{q}__{win}__cov"
        if col not in idx_of:
            continue
        real, null, mk = d31[col].dropna(), shf[col].dropna(), mkv[col].dropna()
        if q.startswith("cmp_"):
            continue
        tb.append((q, len(real), round(float(real.median()), 4), round(float(null.median()), 4), round(float(mk.median()), 4), round(float((real > 0).mean()), 3), round(float((null > 0).mean()), 3), round(float((mk > 0).mean()), 3)))
    print(win)
    print(pd.DataFrame(tb, columns=["seq", "n", "real_cov_med", "shuffle_cov_med", "markov_cov_med", "real_pos", "shuffle_pos", "markov_pos"]).to_string(index=False))

print("\n==== 特征组概览（只看活跃度分组内排名列 __nrank） ====")
res = S[S.col.str.endswith("__nrank")].copy()
res["feat_base"] = res.feat.str.replace("__nrank", "", regex=False)
print(res.groupby("win")[["coverage", "stability_1230_1231"]].median().round(3).to_string())
print(res.groupby("feat_base").agg(cov=("coverage", "median"), stab=("stability_1230_1231", "median"), absic=("ic_20250102", lambda x: x.abs().median())).round(3).sort_values("stab", ascending=False).to_string())

print("\n==== |次日相关| 分布：真实 vs 空模型（全部 __nrank 列）——单日且有大量多重比较，只算很弱的证据 ====")
print(pd.DataFrame({"real": res.ic_20250102.abs().describe(percentiles=[.5, .9, .99]), "shuffle": res.ic_markov_null.abs().describe(percentiles=[.5, .9, .99])}).round(4).to_string())

good = res[(res.coverage >= 3500) & (res.stability_1230_1231 >= 0.3)].sort_values("stability_1230_1231", ascending=False)
print("\ncoverage>=3500 且 stab>=0.3 的列数:", len(good), "/", len(res))
print(good.sort_values("ic_20250102", key=lambda x: x.abs(), ascending=False).head(15)[["col", "coverage", "stability_1230_1231", "ic_20250102", "ic_markov_null", "max_corr_with_naive", "corr_with_log_n"]].round(3).to_string(index=False))

print("\n==== 两种增量（A 用活跃度前原值，序列 e_size_rc；B 为同序列同时段的朴素对照） ====")
inc = []
for win in ["full", "pm", "tail30"]:
    for feat, how in [("cov", "覆盖率=片段并集位置占比"), ("score_per_event", "总分/事件数(求和)"), ("len_mean", "片段长度取均值"), ("len_max", "片段长度取最大")]:
        ca = f"e_size_rc__{win}__{feat}"
        a = d31[ca].values
        an = d31[f"{ca}__nrank"].values
        for b in ["lz78_norm", "lag1_match", "ac_excess_max"]:
            bb = d31[f"e_size_rc__{win}__base_{b}"].values
            bn = d31[f"e_size_rc__{win}__base_{b}__nrank"].values
            inc.append({"win": win, "A": feat, "压法": how, "B": b, "数值:残差离散/A离散": cond_resid_ratio(a, bb), "数值(分组内排名后)": cond_resid_ratio(an, bn), "截面:秩相关": spearman(a, bb)[0], "截面(分组内排名后)": spearman(an, bn)[0], "A覆盖": int(np.isfinite(a).sum()), "A跨日": spearman(d31.loc[common, ca].values, d30.loc[common, ca].values)[0], "A~log_n": spearman(a, log_n_df[win].reindex(d31.index).values)[0]})
I = pd.DataFrame(inc)
print(I.round(3).to_string(index=False))

rep = ["e_size_rc__full__cov__nrank", "e_size_rc__tail30__cov__nrank", "e_size_rc__pm__cov__nrank", "e_lot_rc__full__cov__nrank", "cancel_sizegap_rc__full__cov__nrank", "e_oid_rc__full__tand_cov__nrank", "e_size_rc__full__lag_logmean__nrank", "cmp_event__full__cov", "cmp_event__tail_minus_full__cov"]
rep = [c for c in rep if c in d31.columns]
print("\n==== 代表因子两两截面秩相关 ====")
print(d31[rep].rank().corr().round(2).to_string())
for c in rep:
    row = S[S.col == c].iloc[0]
    print(c, "cov", row.coverage, "zero%", round(row.zero_share, 3), "stab", round(row.stability_1230_1231, 3), "ic", round(row.ic_20250102, 4), "naive", row.max_corr_with_naive if pd.isna(row.max_corr_with_naive) else round(row.max_corr_with_naive, 3), "logn", row.corr_with_log_n if pd.isna(row.corr_with_log_n) else round(row.corr_with_log_n, 3))

with pd.ExcelWriter(os.path.join(R, "validate_columns.xlsx")) as w:
    S.to_excel(w, sheet_name="每列检查", index=False)
    I.to_excel(w, sheet_name="两种增量", index=False)
print("saved validate_columns.xlsx", len(S))
