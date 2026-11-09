"""方向 4（RNA 折叠）单日验证：只读 Rust 产出的最终因子值，不参与任何因子合成。
压成每股一个数的方式都由 Rust 列名给出：cons=字母均值，consmax=字母最大，consmed=字母中位数。"""
import os
os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
import glob, json, sys
import numpy as np, pandas as pd, h5py
from scipy.stats import rankdata
sys.path.insert(0, "/home/chenzongwei/.agents/skills/daily-data-reader")
import daily_data_reader as ddr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = f"{ROOT}/out"; RES = f"{ROOT}/results"


def load(run, date):
    m = json.load(open(f"{OUT}/{run}/{date}/meta.json"))
    v = np.fromfile(f"{OUT}/{run}/{date}/vals.bin", dtype="<f4").reshape(m["n_rows"], m["n_cols"])
    return pd.DataFrame(v, index=m["codes"], columns=m["names"])


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    k = np.isfinite(a) & np.isfinite(b)
    if k.sum() < 30 or np.ptp(a[k]) == 0 or np.ptp(b[k]) == 0:
        return np.nan
    return float(np.corrcoef(rankdata(a[k]), rankdata(b[k]))[0, 1])


def resid_ratio(A, B, bins=20):
    """数值角度：A 对 B 分 20 组做条件平均，残差的离散度 / A 的离散度。"""
    A, B = np.asarray(A, float), np.asarray(B, float)
    k = np.isfinite(A) & np.isfinite(B)
    A, B = A[k], B[k]
    if len(A) < 100 or A.std() == 0:
        return np.nan
    g = np.minimum((rankdata(B, method="average") - 1) / len(B) * bins, bins - 1).astype(int)
    mean = np.bincount(g, A, bins) / np.maximum(np.bincount(g, None, bins), 1)
    return float((A - mean[g]).std() / A.std())


d1 = load("run1", 20241231)
d0 = load("run1", 20241230)
common = d1.index.intersection(d0.index)
print("shape", d1.shape, "两日共同股票", len(common))
rets = ddr.ret(20241226, 20250102)
nxt = rets.loc[20250102].reindex(d1.index).astype(float)
today = rets.loc[20241231].reindex(d1.index).astype(float)
cap = np.log(ddr.flow_cap(20241231, 20241231).iloc[0].reindex(d1.index).astype(float))
tr = ddr.tr(20241231, 20241231).iloc[0].reindex(d1.index).astype(float)
print("次日收益有值", int(nxt.notna().sum()), "收益日", 20250102)

# ---------- 1. 每一列的体检 ----------
rows = []
for c in d1.columns:
    x = d1[c].values
    rows.append(dict(name=c, cover=int(np.isfinite(x).sum()), stab=spearman(d1.loc[common, c], d0.loc[common, c]),
                     ic_next=spearman(x, nxt), corr_today_ret=spearman(x, today), corr_cap=spearman(x, cap), corr_tr=spearman(x, tr),
                     std=float(np.nanstd(x)) if np.isfinite(x).sum() else np.nan))
tb = pd.DataFrame(rows).set_index("name")
print("全部列：有值股票数中位数", tb.cover.median(), "全 NaN 列", int((tb.cover == 0).sum()), "零离散列", int((tb["std"] == 0).sum()))


def fam(n):
    p = n.split("_")
    return p[0] if p[0] in ("cons", "consmax", "consmed", "consstd", "resid", "cx", "wx", "excess", "naive") else "raw"


tb["family"] = [fam(n) for n in tb.index]
print(tb.groupby("family").agg(n=("cover", "size"), cover=("cover", "median"), stab=("stab", "median"), absic=("ic_next", lambda s: s.abs().median())).round(4).to_string())

# ---------- 2. 平凡上界与打乱对照 ----------
print("\n[上界] 计数折叠 配对数/上界（字母较少一方的数量）")
for w in ["full", "pm", "tail"]:
    line = []
    for a in ["flow", "count", "ordjump", "cancel", "lob", "bigord"]:
        s = d1[f"{a}_{w}_cnt_pair_ratio"].dropna()
        line.append(f"{a}:均值{s.mean():.3f}/等于1占比{(s >= 0.9999).mean():.2f}/标准差{s.std():.3f}")
    print(w, " ".join(line))

print("\n[打乱对照] 实际减打乱（同批字母随机换位 3 次的均值）")
rec = []
for w in ["full", "pm", "tail"]:
    for a in ["flow", "count", "size", "speed", "ordjump", "cancel", "lob", "ret", "bigord", "volbar"]:
        for fold in ["cnt", "dec", "vol"]:
            for feat in ["score", "depth_mean", "span_mean"]:
                s = d1[f"excess_{a}_{w}_{fold}_{feat}"].dropna()
                raw_s = d1[f"{a}_{w}_{fold}_{feat}"].dropna()
                if len(s) < 30:
                    continue
                rec.append(dict(win=w, alpha=a, fold=fold, feat=feat, n=len(s), mean=s.mean(), sd=s.std(), t=s.mean() / (s.std() / np.sqrt(len(s))),
                                frac_pos=(s > 0).mean(), raw_mean=raw_s.mean(), raw_sd=raw_s.std(),
                                stab=spearman(d1.loc[common, f"excess_{a}_{w}_{fold}_{feat}"], d0.loc[common, f"excess_{a}_{w}_{fold}_{feat}"]),
                                ic=spearman(d1[f"excess_{a}_{w}_{fold}_{feat}"], nxt)))
ex = pd.DataFrame(rec)
print(ex.groupby(["fold", "feat"]).agg(mean=("mean", "median"), absT=("t", lambda s: s.abs().median()), frac_pos=("frac_pos", "median"), stab=("stab", "median"), absic=("ic", lambda s: s.abs().median())).round(4).to_string())
print(ex.groupby(["win", "feat"]).agg(mean=("mean", "median"), stab=("stab", "median")).round(4).to_string())
print(ex[(ex.alpha == "flow") & (ex.win == "full")].round(4).to_string())

# ---------- 3. 两种增量（数值角度 + 截面角度） ----------
pairs = [
    ("配对数占字母数 vs 上界", "cnt", "pairs_per_active", "bound"),
    ("计数配对比 vs 贪心括号配对", "cnt", "pair_ratio", "greedy_pair"),
    ("衰减得分 vs 翻转率", "dec", "score", "alternation"),
    ("短程配对比 vs 翻转率", "short", "pair_ratio", "alternation"),
    ("平均嵌套深度(衰减) vs 走势振幅", "dec", "depth_mean", "walk_range"),
    ("未配对净方向 vs 字母不平衡", "cnt", "unp_net", "letter_imb"),
    ("量加权得分 vs 量不平衡", "vol", "score", "vol_imb"),
    ("盈利配对比 vs 上界", "profit", "pair_ratio", "bound"),
    ("买先卖后占比 vs 累计走势终点", "dec", "bf_share", "walk_end"),
]
inc = []
for label, fold, feat, naive in pairs:
    for w in ["full", "pm", "tail"]:
        As, Bs = [], []
        for a in ["flow", "count", "size", "speed", "ordjump", "cancel", "lob", "ret", "bigord", "volbar"]:
            if feat == "pairs_per_active":
                A = d1[f"{a}_{w}_{fold}_pair_ratio"].values * d1[f"naive_{a}_{w}_bound"].values
            else:
                A = d1[f"{a}_{w}_{fold}_{feat}"].values
            As.append(A); Bs.append(d1[f"naive_{a}_{w}_{naive}"].values)
        ratio_all = resid_ratio(np.concatenate(As), np.concatenate(Bs))
        ratio_flow = resid_ratio(As[0], Bs[0])
        row = dict(pair=label, win=w, resid_ratio_pool=ratio_all, resid_ratio_flow=ratio_flow)
        cs_feat = "pair_ratio" if feat == "pairs_per_active" else feat
        for tag in ["cons", "consmax", "consmed"]:
            a_col, b_col = f"{tag}_{w}_{fold}_{cs_feat}", f"{tag}_naive_{w}_{naive}"
            row[f"cs_{tag}"] = spearman(d1[a_col], d1[b_col])
            row[f"cover_{tag}"] = int(d1[a_col].notna().sum())
            row[f"stab_{tag}"] = spearman(d1.loc[common, a_col], d0.loc[common, a_col])
        row["cs_flow_only"] = spearman(d1[f"flow_{w}_{fold}_{cs_feat}"], d1[f"naive_flow_{w}_{naive}"])
        inc.append(row)
inc = pd.DataFrame(inc)
print("\n[两种增量] resid_ratio=残差离散度/A离散度（池化=全部股票×10种字母，20分箱）；cs_*=按对应压法的截面秩相关")
print(inc.round(3).to_string())

# ---------- 4. 同分规则、粒度敏感性 ----------
ea = load("tie_early", 20241231)
b60 = load("bar60", 20241231)
tie = []
for f in ["cnt", "dec", "vol", "rna"]:
    for x in ["score", "depth_mean", "depth_max", "span_mean", "branch", "hairpin"]:
        c = f"flow_full_{f}_{x}"; cc = f"cons_full_{f}_{x}"
        tie.append(dict(fold=f, feat=x, flow=spearman(d1[c], ea[c]), cons=spearman(d1[cc], ea[cc])))
tie = pd.DataFrame(tie)
print("\n[同分规则] 配对同分时取最近 vs 取最远，同一列的截面秩相关（全天）")
print(tie.round(3).pivot(index="feat", columns="fold", values="flow").to_string())
print(tie.round(3).pivot(index="feat", columns="fold", values="cons").to_string())
bar = []
for f in ["cnt", "dec", "vol", "profit", "short"]:
    for x in ["score", "depth_mean", "span_mean", "unp_net"]:
        c = f"cons_full_{f}_{x}"
        bar.append(dict(fold=f, feat=x, corr=spearman(d1[c], b60[c])))
print("\n[粒度] 全天 30 秒 vs 60 秒字母，共识列截面秩相关")
print(pd.DataFrame(bar).round(3).pivot(index="feat", columns="fold", values="corr").to_string())

# ---------- 5. 与现有因子的最大相关（因子库抽样） ----------
L = "/nas197/user_home/chenzongwei/factor_library"
fac_files = sorted(glob.glob(f"{L}/dailybank/*/*.h5")) + sorted(glob.glob(f"{L}/factor_base/*/*.h5"))[:700]
existing = {}
for fp in fac_files:
    try:
        d = os.path.dirname(fp)
        cal = [int(x.split(",")[0]) for x in open(f"{d}/calendar_map.csv").read().splitlines()[1:]]
        sym = [x.split(",")[0] for x in open(f"{d}/symbol_map.csv").read().splitlines()[1:]]
        if 20241231 not in cal:
            continue
        with h5py.File(fp, "r") as h:
            arr = h["data"]
            row = arr[cal.index(20241231)]
        if row.ndim == 1 and len(row) >= len(sym):
            existing[fp.replace(L + "/", "")] = pd.Series(row[: len(sym)], index=sym).reindex(d1.index).astype(float).values
    except Exception:
        continue
print("\n现有因子抽样数", len(existing))
existing = {k: v for k, v in existing.items() if np.isfinite(v).sum() > 1000}
print("其中当天有值超过 1000 只", len(existing))
exm = np.array(list(existing.values()))
exk = list(existing)


def max_exist(col):
    x = d1[col].values
    cs = [spearman(x, exm[i]) for i in range(len(exk))]
    cs = np.array(cs)
    if not np.isfinite(cs).any():
        return np.nan, ""
    i = int(np.nanargmax(np.abs(cs)))
    return float(cs[i]), exk[i]


pick = tb[(tb.family.isin(["cons", "resid", "consmax", "consmed", "consstd", "wx", "cx"])) & (tb.cover >= 4000) & (tb["std"] > 0)]
print("\n[稳定性最高的合成列] 跨日秩相关前 25")
print(pick.sort_values("stab", ascending=False).head(25)[["cover", "stab", "ic_next", "corr_today_ret", "corr_cap", "corr_tr"]].round(3).to_string())
print("\n[残差列稳定性前 20]")
print(pick[pick.family == "resid"].sort_values("stab", ascending=False).head(20)[["cover", "stab", "ic_next", "corr_today_ret", "corr_cap", "corr_tr"]].round(3).to_string())
print("\n[残差列 |次日相关| 前 15]（单日，只能当很弱的证据）")
r = pick[(pick.family == "resid") & (pick.stab > 0.2)]
print(r.reindex(r.ic_next.abs().sort_values(ascending=False).index).head(15)[["cover", "stab", "ic_next", "corr_today_ret", "corr_cap", "corr_tr"]].round(3).to_string())

reps = json.load(open(f"{RES}/representatives.json")) if os.path.exists(f"{RES}/representatives.json") else []
if reps:
    zero = tb[tb["std"] == 0]
    print("\n[零离散列] 数量", len(zero), "按后缀", zero.index.str.split("_").str[-1].value_counts().to_dict(), "按前缀", zero.family.value_counts().to_dict(), "示例", list(zero.index[:6]))
    cap0 = np.log(ddr.flow_cap(20241230, 20241230).iloc[0].reindex(d0.index).astype(float))
    tr0 = ddr.tr(20241230, 20241230).iloc[0].reindex(d0.index).astype(float)

    def strip(x, ctrl):
        """验证专用：扣掉流通市值、换手、字母活跃占比后的残差，只用于判断是否是风格代理，不产出因子。"""
        x = np.asarray(x, float)
        C = np.column_stack(ctrl)
        k = np.isfinite(x) & np.isfinite(C).all(axis=1)
        out_ = np.full(len(x), np.nan)
        X = np.column_stack([np.ones(k.sum())] + [rankdata(C[k, j]) for j in range(C.shape[1])])
        y = rankdata(x[k])
        out_[k] = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
        return out_

    active1 = d1["cons_naive_full_active_share"].values
    active0 = d0["cons_naive_full_active_share"].values
    out = []
    for c in reps:
        v, name = max_exist(c)
        r1 = pd.Series(strip(d1[c].values, [cap.values, tr.values, active1]), index=d1.index)
        r0 = pd.Series(strip(d0[c].values, [cap0.values, tr0.values, active0]), index=d0.index)
        out.append(dict(name=c, cover=int(tb.loc[c, "cover"]), stab=tb.loc[c, "stab"], ic_next=tb.loc[c, "ic_next"], corr_today_ret=tb.loc[c, "corr_today_ret"],
                        corr_cap=tb.loc[c, "corr_cap"], corr_tr=tb.loc[c, "corr_tr"], corr_active=spearman(d1[c], active1), max_exist_corr=v, max_exist=name,
                        stab_ex_style=spearman(r1.loc[common], r0.loc[common]), ic_next_ex_style=spearman(r1.values, nxt)))
    rp = pd.DataFrame(out)
    print("\n[代表因子]\n", rp.round(3).to_string())
    cm = d1[reps].rank().corr()
    print("\n[代表因子两两截面秩相关]\n", cm.round(2).to_string())
    with pd.ExcelWriter(f"{RES}/validation_summary.xlsx") as w:
        tb.reset_index().to_excel(w, sheet_name="全部列体检", index=False)
        ex.to_excel(w, sheet_name="打乱对照", index=False)
        inc.to_excel(w, sheet_name="两种增量", index=False)
        tie.to_excel(w, sheet_name="同分规则敏感性", index=False)
        pd.DataFrame(bar).to_excel(w, sheet_name="粒度敏感性", index=False)
        rp.to_excel(w, sheet_name="代表因子", index=False)
        cm.reset_index().to_excel(w, sheet_name="代表因子相关", index=False)
