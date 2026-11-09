"""方向 1 验证脚本：只读 Rust 输出的因子值，不参与任何因子数值合成。
看覆盖、跨日稳定性、对次日收益的单日截面秩相关、两种增量、与现有因子的最大相关。
"""
import glob, json, os, random, re, sys, time
os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
import numpy as np, pandas as pd
sys.path.insert(0, "/home/chenzongwei/.agents/skills/daily-data-reader")
import daily_data_reader as ddr
import rust_pyfunc as rp

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
def load(d):
    j = json.load(open(f"{R}/d{d}.json"))
    return pd.DataFrame(np.array(j["vals"], dtype=float), index=j["codes"], columns=j["names"])
d31, d30, d27 = load(20241231), load(20241230), load(20241227)
print("shape", d31.shape, d30.shape, d27.shape)
ret = ddr.ret()
r0102, r1231 = ret.loc[20250102].reindex(d31.index), ret.loc[20241231].reindex(d30.index)

def rank_corr(a, b):
    """a,b 同索引。返回秩相关和共同有效数。"""
    m = a.notna() & b.notna()
    if m.sum() < 30 or a[m].nunique() < 3 or b[m].nunique() < 3:
        return np.nan, int(m.sum())
    return np.corrcoef(a[m].rank().values, b[m].rank().values)[0, 1], int(m.sum())

common = d31.index.intersection(d30.index).intersection(d27.index)
rows = []
for c in d31.columns:
    if c == "meta_n_trades": continue
    a31, a30, a27 = d31[c].reindex(common), d30[c].reindex(common), d27[c].reindex(common)
    stab_adj, _ = rank_corr(a31, a30)      # 相邻（两天输入共用 20241230）
    stab_dis, _ = rank_corr(a31, a27)      # 输入完全不重叠
    ic31, _ = rank_corr(d31[c], r0102)
    ic30, _ = rank_corr(d30[c], r1231)
    liq, _ = rank_corr(d31[c], d31["meta_n_trades"])
    rows.append(dict(name=c, coverage=int(d31[c].notna().sum()), std=float(d31[c].std()), stab_adjacent=stab_adj,
                     stab_nonoverlap=stab_dis, ic_next_20250102=ic31, ic_next_20241231=ic30, corr_liq=liq))
S = pd.DataFrame(rows).set_index("name")

# ---------- 解析列名以便分组看 ----------
def parse(n):
    m = re.match(r"^(all|pm|tail)_(t2x|t|v)_(\w+?)_(gl|gt|lc|ug)_(\w+)$", n)
    return m.groups() if m else None
P = pd.DataFrame({n: parse(n) for n in S.index if parse(n)}, index=["win", "clock", "alpha", "mode", "feat"]).T
for x in P.columns: S[x] = P[x]
pd.set_option("display.width", 200, "display.max_columns", 30, "display.max_rows", 200)
print("\n# 总览：逐列稳定性（不重叠输入，中位）与覆盖")
print(S.dropna(subset=["win"]).groupby(["win", "clock"])[["coverage", "stab_nonoverlap", "stab_adjacent", "corr_liq"]].median().round(3))
print("\n# 按字母表（全天 t 钟）中位数：")
print(S[(S.win == "all") & (S.clock == "t")].groupby(["alpha"])[["stab_nonoverlap", "ic_next_20250102", "ic_next_20241231", "corr_liq"]].median().round(3))
print("\n# 按特征（全天 t 钟 gl 模式）中位数：")
g = S[(S["mode"] == "gl") & (S.clock == "t")].groupby(["win", "feat"])[["stab_nonoverlap", "ic_next_20250102", "ic_next_20241231", "corr_liq"]].median().round(3)
print(g.to_string())

print("\n# 合成列全表")
comp = S[S.win.isna()].sort_index()
print(comp[["coverage", "stab_nonoverlap", "stab_adjacent", "ic_next_20250102", "ic_next_20241231", "corr_liq"]].round(3).to_string())

print("\n# 两日 IC 同号且绝对值都 >0.05 的列数:",
      int(((np.sign(S.ic_next_20250102) == np.sign(S.ic_next_20241231)) & (S.ic_next_20250102.abs() > .05) & (S.ic_next_20241231.abs() > .05)).sum()), "/", len(S))
top = S.assign(mn=np.minimum(S.ic_next_20250102.abs(), S.ic_next_20241231.abs()))
top = top[np.sign(top.ic_next_20250102) == np.sign(top.ic_next_20241231)].sort_values("mn", ascending=False).head(15)
print(top[["coverage", "stab_nonoverlap", "ic_next_20250102", "ic_next_20241231", "corr_liq"]].round(3).to_string())

# ---------- 增量（1）：数值角度，在“每股每时段每字母表”这一层 ----------
print("\n# 增量-数值角度：A 对 B 按 B 的 50 分位分箱取条件均值，报 残差离散度/A 离散度")
res = []
for (A, B, lab) in [("gl_score", "ug_score", "比对分对不许错位分"), ("gl_identity", "ug_identity", "比对相同率对逐位相同率")]:
    for w in ["all", "pm", "tail"]:
        ratios = []
        for al in ["volsig", "mag5", "mag5y", "size", "rhythm", "oid", "price", "bookflow", "bookshape", "cancel"]:
            a, b = d31[f"{w}_t_{al}_{A}"], d31[f"{w}_t_{al}_{B}"]
            m = a.notna() & b.notna()
            if m.sum() < 500: continue
            bins = pd.qcut(b[m], 50, labels=False, duplicates="drop")
            ratios.append((a[m] - a[m].groupby(bins).transform("mean")).std() / a[m].std())
        res.append(dict(pair=lab, win=w, resid_over_A_median=round(float(np.median(ratios)), 3), n_alphabets=len(ratios)))
print(pd.DataFrame(res).to_string(index=False))

# ---------- 增量（2）：截面角度，三种压法 ----------
print("\n# 增量-截面角度：同一批股票，A 与 B 分别跨 10 个字母表压成每股一个数后的截面秩相关")
out = []
for w in ["all", "pm", "tail"]:
    for ag in ["rankmean", "rankmax", "rankmin"]:
        for k, u in [("score", "ug_score"), ("identity", "ug_identity")]:
            A, B = f"{w}_cmp_{k}_{ag}", f"{w}_cmp_{u}_{ag}"
            rc, n = rank_corr(d31[A], d31[B])
            out.append(dict(win=w, compress=ag, A=k, B=u, rankcorr=round(rc, 3), n=n,
                            A_stab=round(S.loc[A, "stab_nonoverlap"], 3), B_stab=round(S.loc[B, "stab_nonoverlap"], 3)))
print(pd.DataFrame(out).to_string(index=False))
print("\n# 比对类合成因子 vs 普通分时量相关 / 成交量重心移动（同一批股票）")
o2 = []
for w in ["all", "pm", "tail"]:
    for A, B in [(f"{w}_cmp_score_rankmean", f"{w}_naive_volcorr"), (f"{w}_cmp_score_rankmean", f"{w}_naive_volrankcorr"),
                 (f"{w}_cmp_lag_mean_rankmean", f"{w}_naive_vol_com_shift")]:
        rc, n = rank_corr(d31[A], d31[B]); o2.append(dict(A=A, B=B, rankcorr=round(rc, 3), n=n))
print(pd.DataFrame(o2).to_string(index=False))

# ---------- 代表因子两两相关 ----------
reps = ["all_cmp_score_rankmean", "all_cmp_excess_rankmean", "all_cmp_lag_mean_rankmean", "all_cmp_lag_abs_rankmean",
        "all_cmp_dp_gain_rankmean", "all_cmp_score_rankmean_liqx", "pm_cmp_score_rankmean", "tail_cmp_score_rankmean",
        "tail_minus_all_score_rankmean", "all_t_oid_lc_span", "all_v_size_gl_score", "all_naive_volcorr", "meta_n_trades"]
print("\n# 代表因子两两秩相关（20241231）")
print(d31[reps].rank().corr().round(2).to_string())

# ---------- 与现有因子的最大相关（抽样） ----------
root = "/nas197/user_home/chenzongwei/factor_library/dailybank"
random.seed(7); t0 = time.time(); lib = {}
libs = sorted(os.listdir(root))
for v in libs:
    fs = sorted(glob.glob(f"{root}/{v}/*.h5"))
    for f in random.sample(fs, min(30, len(fs))):
        if time.time() - t0 > 240: break
        name = os.path.basename(f)[:-3]
        try:
            x = rp.read_factor(v, name, 20241231, 20241231, hdf5_dir_here=root)
            if len(x):
                lib[f"{v}/{name}"] = x.iloc[-1]
        except Exception as e:
            pass
print(f"\n# 现有因子抽样 {len(lib)} 个（共 {len(libs)} 个库，每库最多 30 个），耗时 {time.time()-t0:.0f}s")
L = pd.DataFrame(lib)
L.index = [str(i).split(".")[0].zfill(6) for i in L.index]
L = L.reindex(d31.index)
Lr = L.rank()
Lr = Lr.loc[:, Lr.notna().sum() >= 1000]
chk = ["all_cmp_score_rankmean", "all_cmp_excess_rankmean", "all_cmp_lag_mean_rankmean", "all_cmp_lag_abs_rankmean", "all_cmp_dp_gain_rankmean",
       "all_cmp_score_rankmean_liqx", "pm_cmp_score_rankmean", "tail_cmp_score_rankmean", "tail_minus_all_score_rankmean", "all_cmp_ug_score_rankmean"]
out = []
for c in chk:
    co = Lr.corrwith(d31[c].rank()).abs()
    out.append(dict(factor=c, max_abs_corr=round(float(co.max()), 3), which=co.idxmax(), n_lib=int(co.notna().sum())))
print(pd.DataFrame(out).to_string(index=False))
S.to_excel(f"{R}/validation_summary.xlsx")
print("written validation_summary.xlsx")
