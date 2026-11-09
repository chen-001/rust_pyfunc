"""方向 3 镜像比对的验证（只读 Rust 产物，不参与任何因子合成）。
用法: python validate.py results/mirror_20241231.json results/mirror_20241230.json
输出写入 results/validate_summary.xlsx，关键数字打印到屏幕。"""
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
    return d["date"], pd.DataFrame(np.array(d["vals"], dtype=float), index=d["codes"], columns=d["names"])


def rank_corr(a, b):
    x = pd.concat([a, b], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if len(x) < 50 or x.iloc[:, 0].nunique() < 2 or x.iloc[:, 1].nunique() < 2:
        return np.nan, len(x)
    return x.iloc[:, 0].rank().corr(x.iloc[:, 1].rank()), len(x)


def next_ret(date, codes):
    r = ddr.ret(start=20241220, end=20250110)
    idx = r.index[r.index > date]
    ts = int(idx[0])
    return ts, r.loc[ts].reindex(codes)


def main():
    d1, A = load(sys.argv[1])
    d0, B = load(sys.argv[2])
    print("主日", d1, A.shape, "对照日", d0, B.shape)
    ts, ret1 = next_ret(d1, A.index)
    ts0, ret0 = next_ret(d0, B.index)
    print("次日收益日期", ts, ts0, "有值", ret1.notna().sum(), ret0.notna().sum())

    rows = []
    for c in A.columns:
        cov = int(A[c].notna().sum())
        stab, n_stab = rank_corr(A[c], B[c])
        ic1, _ = rank_corr(A[c], ret1)
        ic0, _ = rank_corr(B[c], ret0)
        rows.append(dict(col=c, cov=cov, cov_pct=cov / len(A), stab=stab, n_stab=n_stab, ic_20241231=ic1, ic_20241230=ic0,
                         ic_mean=np.nanmean([ic1, ic0]), ic_same_sign=bool(np.sign(ic1) == np.sign(ic0)),
                         p50=A[c].median(), std=A[c].std()))
    S = pd.DataFrame(rows).set_index("col")

    def fam(c):
        p = c.split("_")
        if p[0] in ("mir", "csrank", "peerx"):
            return dict(layer=p[0], alpha=p[1], clock=p[2], win=p[3], feat=p[4])
        if p[0] == "var":
            return dict(layer="var", alpha=p[1], clock=p[2], win=p[3], feat=p[5], variant=p[4])
        if p[0] == "naive":
            return dict(layer="naive", feat=p[1], win=p[2])
        return dict(layer=p[0], win=p[-1])

    S = S.join(pd.DataFrame([fam(c) for c in S.index], index=S.index))

    print("\n== 覆盖度与稳定性（按层）==")
    print(S.groupby("layer")[["cov", "stab"]].median().round(3))
    print("总列数", len(S), "全空列", int((S["cov"] == 0).sum()))

    mir = S[S.layer == "mir"]
    print("\n== 按特征汇总（mir 层；中位 |IC 均值| 与同号比例、稳定性）==")
    g = mir.groupby("feat").agg(stab=("stab", "median"), absic=("ic_mean", lambda x: x.abs().median()),
                                ic_mean=("ic_mean", "median"), same=("ic_same_sign", "mean"), cov=("cov", "median"))
    print(g.sort_values("absic", ascending=False).round(4).to_string())

    print("\n== 按时段、时钟、字母表（zseg 与 bits）==")
    for key in ("win", "clock", "alpha"):
        t = mir[mir.feat.isin(["zseg", "bits", "signed"])].groupby([key, "feat"]).agg(
            stab=("stab", "median"), ic=("ic_mean", "median"), absic=("ic_mean", lambda x: x.abs().median())).round(4)
        print(t.to_string())

    # 镜像对照：四种变换的归一 z 值 与 同一位置（全部用 A 当日的真实字段）
    print("\n== 镜像 vs 对照（zseg，全市场中位数 / M 比对照高的股票占比 / 平均 IC 绝对值）==")
    rec = []
    for c in sorted({x.rsplit("_", 1)[0] for x in mir.index}):
        p = c.split("_")
        z = {k: A[f"{c}_{n}"] for k, n in dict(M="zseg", R="zsegR", C="zsegC", D="zsegD").items()}
        row = dict(combo=c, alpha=p[1], clock=p[2], win=p[3])
        for k, v in z.items():
            row[f"med_{k}"] = v.median()
            row[f"ic_{k}"] = rank_corr(v, ret1)[0]
            row[f"stab_{k}"] = rank_corr(v, B[f"{c}_{dict(M='zseg',R='zsegR',C='zsegC',D='zsegD')[k]}"])[0]
        for k in "RCD":
            row[f"M_gt_{k}"] = float((z["M"] > z[k]).sum() / (z["M"].notna() & z[k].notna()).sum())
            row[f"corr_M_{k}"] = rank_corr(z["M"], z[k])[0]
        rec.append(row)
    C = pd.DataFrame(rec)
    show = ["med_M", "med_R", "med_C", "med_D", "M_gt_R", "M_gt_C", "M_gt_D", "corr_M_R", "corr_M_C", "corr_M_D",
            "ic_M", "ic_R", "ic_C", "ic_D", "stab_M", "stab_R", "stab_C", "stab_D"]
    print(C[show].abs().median().round(4).to_string(), "\n(上一列为 |x| 的中位)")
    print(C.groupby("win")[["med_M", "med_R", "med_C", "med_D", "M_gt_R", "M_gt_C", "M_gt_D"]].median().round(3).to_string())

    # 两种增量（对朴素替代物）
    print("\n== 两种增量 vs 朴素前后段反向相关 naive_anticorr ==")
    inc = []
    for w in ("all", "pm", "tail"):
        nv = A[f"naive_anticorr_{w}"]
        fb = A[f"naive_fbdiff_{w}"]
        for alpha in ("dir", "int", "osize", "jump", "book"):
            base = f"mir_{alpha}_tm_{w}"
            a = A[f"{base}_perletter"]
            x = pd.concat([a, nv], axis=1, keys=["a", "b"]).dropna()
            if len(x) < 200:
                continue
            bins = pd.qcut(x["b"], 20, labels=False, duplicates="drop")
            res = x["a"] - x.groupby(bins)["a"].transform("mean")
            ratio = res.std() / x["a"].std()
            row = dict(win=w, alpha=alpha, n=len(x), numeric_resid_ratio=ratio)
            for nm, col in (("max", "perletter"), ("center_mean", "meancent"), ("top3_mean", "top3cent"), ("seg_z", "zseg")):
                row[f"cs_{nm}_vs_anticorr"] = rank_corr(A[f"{base}_{col}"], nv)[0]
                row[f"cs_{nm}_vs_fbdiff"] = rank_corr(A[f"{base}_{col}"], fb)[0]
                row[f"cs_{nm}_vs_logamt"] = rank_corr(A[f"{base}_{col}"], A[f"audit_logamt_{w}"])[0]
                row[f"cs_{nm}_cov"] = int(A[f"{base}_{col}"].notna().sum())
                row[f"cs_{nm}_stab"] = rank_corr(A[f"{base}_{col}"], B[f"{base}_{col}"])[0]
            inc.append(row)
    I = pd.DataFrame(inc)
    print(I.round(3).to_string())

    # 变体
    print("\n== 参数变体（var 层，按变体：z 的稳定性中位 与 |IC| 中位）==")
    var = S[S.layer == "var"]
    print(var.groupby(["variant", "feat"]).agg(stab=("stab", "median"), absic=("ic_mean", lambda x: x.abs().median()), cov=("cov", "median")).round(4).to_string())

    print("\n== 朴素替代物自身（稳定性、IC）==")
    print(S[S.layer == "naive"][["cov", "stab", "ic_20241231", "ic_20241230"]].round(4).to_string())

    # 代表因子：稳定性高、方向两日同号、IC 绝对值大；排除明显由规模驱动的
    cand = S[(S.layer.isin(["mir", "csrank", "peerx", "var"])) & (S["cov"] > len(A) * 0.8) & (S.ic_same_sign)].copy()
    cand["amt_abs"] = [abs(rank_corr(A[c], A["audit_logamt_all"])[0]) for c in cand.index]
    good = cand[cand.amt_abs < 0.5].sort_values("ic_mean", key=lambda x: x.abs(), ascending=False)
    print("\n== 候选（覆盖>80%、两日 IC 同号、与成交额秩相关<0.5，按 |IC 均值| 排序）前 25 ==")
    print(good[["cov", "stab", "ic_20241231", "ic_20241230", "ic_mean", "amt_abs"]].head(25).round(4).to_string())

    with pd.ExcelWriter("results/validate_summary.xlsx") as w:
        S.round(5).to_excel(w, sheet_name="all_columns")
        C.round(4).to_excel(w, sheet_name="mirror_vs_controls", index=False)
        I.round(4).to_excel(w, sheet_name="two_increments", index=False)
        good.head(200).round(5).to_excel(w, sheet_name="candidates")
    print("\n已写 results/validate_summary.xlsx")


if __name__ == "__main__":
    main()
