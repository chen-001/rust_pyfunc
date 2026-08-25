"""switch_anneal 完整分析(主脚本, 收益口径已修正)

【收益口径(正确版, 2026-07-17 修订)】
- read_daily(ret=1) 第 D 行 = 当日收益(D-1收→D收), 不是前瞻收益! 不可直接对齐。
- 正确做法: close = read_daily(close=1);
  gap1 = close.shift(-1)/close - 1   # 第 D 行 = D→D+1 前瞻1日收益
  gap5 = close.shift(-5)/close - 1   # 第 D 行 = D→D+5 前瞻5日收益
- 因子日 D 与 gap1/gap5 的 D 行对齐; 20260717 为数据末日, gap1/gap5 全 NaN, 该日剔除。
- 旧口径(当日收益)仅在"前视偏差演示"小节保留作对照, 不得作为结论依据。

运行: python analyze_switch_anneal.py
"""
import json
import numpy as np
import pandas as pd
from scipy import stats

DATES = [20240104, 20240603, 20241008, 20260105, 20260717]
OUT_DIR = "/home/chenzongwei/rust_pyfunc/sandbox_switch_anneal/out"
from pure_ocean_breeze.jason.data.read_data import read_daily

close = read_daily(close=1)
gap1 = close.shift(-1) / close - 1   # 正确口径: 前瞻1日
gap5 = close.shift(-5) / close - 1   # 正确口径: 前瞻5日
ret_same = close / close.shift(1) - 1  # 仅用于前视偏差演示(错误口径)
amounts = pd.read_parquet(
    "/home/chenzongwei/database/daily_data/amounts.parquet"
).reindex(close.index)


def to_suffix(code: str) -> str:
    if code.startswith(("60", "68")):
        return code + ".SH"
    if code.startswith(("00", "30")):
        return code + ".SZ"
    return code + ".BJ"


frames = {}
cs_all = []
for d in DATES:
    js = json.load(open(f"{OUT_DIR}/out_{d}.json"))
    df = pd.DataFrame(js["d_vals"], index=js["d_codes"], columns=js["d_names"])
    df.index = pd.Index([to_suffix(c) for c in df.index], name="code")
    frames[d] = df
    for c in js["cs"]:
        c["c_vec"] = np.array(c["c_vec"], dtype=np.float64)
        cs_all.append({"date": d, **c})

ALL_F = [c for c in frames[20240104].columns if c != "n_min"]
CTRL = ["r0", "std", "ac1", "trend"]
FEATS = [c for c in ALL_F if c not in CTRL + ["mean", "morn_aft_diff"]]


def col_ic(df, col, ret_series, t):
    v = df[col].astype(float)
    r = ret_series.loc[t].reindex(df.index)
    m = v.notna() & r.notna() & np.isfinite(v) & np.isfinite(r)
    if m.sum() < 100:
        return np.nan
    return stats.spearmanr(v[m], r[m]).statistic


def col_amt_corr(df, col, t):
    v = df[col].astype(float)
    a = np.log(amounts.loc[t].reindex(df.index).astype(float))
    m = v.notna() & np.isfinite(v) & a.notna() & np.isfinite(a)
    if m.sum() < 100:
        return np.nan
    return stats.spearmanr(v[m], a[m]).statistic


def col_resic(df, col, ctrl_cols, ret_series, t):
    v = df[col].astype(float)
    r = ret_series.loc[t].reindex(df.index)
    base = df[ctrl_cols].astype(float)
    m = (
        v.notna() & r.notna() & np.isfinite(v) & np.isfinite(r)
        & np.isfinite(base).all(axis=1)
    )
    if m.sum() < 100:
        return np.nan
    X = np.column_stack([np.ones(m.sum()), base[m].values])
    beta, *_ = np.linalg.lstsq(X, v[m].values, rcond=None)
    resid = v[m].values - X @ beta
    return stats.spearmanr(resid, r[m].values).statistic


def summarize(ic_df, label):
    s = ic_df.mean()
    print(f"\n--- {label} 汇总 ---")
    for c in ic_df.columns:
        v = ic_df[c].dropna()
        print(
            f"  {c:15s} mean_ic={s[c]:+.4f} abs={ic_df[c].abs().mean():.4f} "
            f"icir={v.mean()/v.std():+.3f} pos={float((v>0).mean()):.2f} n_days={len(v)}"
        )


# ============ ① 方向D IC: 正确口径 gap1 / gap5 ============
print("=== ① 方向D 日 IC 明细(正确口径) ===")
ic_results = {}
for lab, ret_s in [("gap1 前瞻1日", gap1), ("gap5 前瞻5日", gap5)]:
    mat = []
    for d in DATES:
        t = pd.Timestamp(str(d))
        row = {c: col_ic(frames[d], c, ret_s, t) for c in ALL_F}
        mat.append(row)
    ic_df = pd.DataFrame(mat, index=[str(d) for d in DATES])
    ic_results[lab] = ic_df
    print(f"\n--- {lab}(20260717 无前瞻收益应全 NaN) ---")
    print(ic_df.round(4).to_string())
    summarize(ic_df, lab)

# ============ ② 成交额截面相关 ============
print("\n=== ② 成交额截面相关(5天均值, 与收益口径无关) ===")
amt_rows = pd.DataFrame(
    [{c: col_amt_corr(frames[d], c, pd.Timestamp(str(d))) for c in ALL_F} for d in DATES],
    index=[str(d) for d in DATES],
)
print(amt_rows.mean().round(4).to_string())

# ============ ③ 正交掉简单统计后的残差 IC(正确口径) ============
print("\n=== ③ 正交掉 [r0,std,ac1,trend] 后的残差 IC ===")
for lab, ret_s in [("gap1 前瞻1日", gap1), ("gap5 前瞻5日", gap5)]:
    rows = []
    for d in DATES:
        t = pd.Timestamp(str(d))
        rows.append(
            {c: col_resic(frames[d], c, CTRL, ret_s, t) for c in FEATS + ["mean", "morn_aft_diff"]}
        )
    res = pd.DataFrame(rows, index=[str(d) for d in DATES])
    print(f"\n--- {lab} ---")
    print(res.round(4).to_string())
    summarize(res, lab)

# ============ ④ 前视偏差演示(错误口径, 仅对照) ============
print("\n=== ④ 【前视偏差演示】同日收益口径(错误, 仅供对比, 不作结论) ===")
for f in ["ladder", "max_jump", "mean", "ac1", "r0"]:
    ics = []
    for d in DATES:
        t = pd.Timestamp(str(d))
        ic = col_ic(frames[d], f, ret_same, t)
        ics.append(ic)
    s = pd.Series(ics, dtype=float)
    print(
        f"  {f:15s} mean_ic={s.mean():+.4f} icir={s.mean()/s.std():+.3f} "
        f"pos={float((s>0).mean()):.2f}  (同日) vs gap1: mean_ic={ic_results['gap1 前瞻1日'][f].mean():+.4f}"
    )

# ============ ⑤ 特征面板相关(与收益无关, 结构结论) ============
p = pd.concat(frames, names=["date", "code"])
print("\n=== ⑤ 方向D 特征截面相关(面板5天, 与收益口径无关) ===")
print(p[ALL_F].corr(method="spearman").round(3).to_string())

# ============ ⑥ 方向C/B 时刻级 + 冗余 ============
cs_df = pd.DataFrame(cs_all)
print("\n=== ⑥ 方向C/B 时刻级(15 样本, 与收益口径无关) ===")
print(
    cs_df[
        ["date", "t", "r", "n", "c_r0", "c_steps70", "c_final_r", "c_gini",
         "c_top5pct", "c_kurt", "b_r0", "b_steps70", "b_ladder"]
    ].round(4).to_string()
)
for a, b in [
    ("r", "c_steps70"), ("c_top5pct", "c_steps70"), ("c_kurt", "c_steps70"),
    ("c_gini", "c_steps70"), ("c_gini", "r"),
]:
    print(f"  corr({a}, {b}) = {stats.spearmanr(cs_df[a], cs_df[b]).statistic:+.3f}")
cs_df["is_low"] = pd.qcut(cs_df["r"], 2, labels=False)
for g in [0, 1]:
    sub = cs_df[cs_df["is_low"] == g]["c_steps70"]
    print(
        f"  同r半区{g}: n={len(sub)} mean={sub.mean():.0f} std={sub.std():.0f} "
        f"CV={sub.std()/sub.mean():.3f}"
    )

# ============ ⑦ perm 对照(方向C 因果判定, 与收益口径无关) ============
def anneal_steps70(x, m_max, seed):
    n = len(x)
    mean = float(np.mean(x))
    sigma2 = float(np.mean((x - mean) ** 2))
    if sigma2 <= 0:
        return None
    rng = np.random.default_rng(seed)
    guess = np.sort(x).astype(np.float64)
    s = float(np.sum((guess - x) ** 2))
    denom = 2.0 * sigma2 * n
    ct_base = sigma2
    for t in range(m_max):
        i, j = rng.integers(0, n, 2)
        if i == j:
            j = rng.integers(0, n)
        ds = 2.0 * (guess[i] - guess[j]) * (x[i] - x[j])
        ct = ct_base * (1.0 - t / (m_max - 1))
        if ds < 0.0 or ds < ct:
            s += ds
            guess[i], guess[j] = guess[j], guess[i]
        cr = 1.0 - s / denom
        if cr >= 0.70:
            return t
    return None


print("\n=== ⑦ perm对照: 真实 c 向量 vs 同值集合随机排列(到达 r=0.70 步数) ===")
sel = sorted(cs_all, key=lambda c: c["c_top5pct"], reverse=True)[:2] + sorted(
    cs_all, key=lambda c: c["c_top5pct"]
)[:2]
for c in sel:
    vec = c["c_vec"]
    perms = [
        anneal_steps70(
            vec[np.random.default_rng(1234 + k).permutation(len(vec))], 500_000, 999 + k
        )
        for k in range(3)
    ]
    print(
        f"  {c['date']} t={c['t']:4d} top5pct={c['c_top5pct']:.3f} "
        f"真实steps70={c['c_steps70']} perm={perms}"
    )
