"""方向D逐股退火特征 + 方向C/B时刻级特征的 IC 评估(switch_anneal 探索)"""
import json
import numpy as np
import pandas as pd
from scipy import stats

DATES = [20240104, 20240603, 20241008, 20260105, 20260717]
OUT_DIR = "/home/chenzongwei/rust_pyfunc/sandbox_switch_anneal/out"

from pure_ocean_breeze.jason.data.read_data import read_daily

rets = read_daily(ret=1)
amounts = pd.read_parquet(
    "/home/chenzongwei/database/daily_data/amounts.parquet"
).reindex(rets.index)


def to_suffix(code: str) -> str:
    if code.startswith(("60", "68")):
        return code + ".SH"
    if code.startswith(("00", "30")):
        return code + ".SZ"
    return code + ".BJ"


# ---------- 加载方向 D ----------
frames = {}
cs_all = []
for d in DATES:
    js = json.load(open(f"{OUT_DIR}/out_{d}.json"))
    df = pd.DataFrame(js["d_vals"], index=js["d_codes"], columns=js["d_names"])
    df.index = pd.Index([to_suffix(c) for c in df.index], name="code")
    frames[d] = df
    for c in js["cs"]:
        cs_all.append({"date": d, **c})

# ---------- 日 IC ----------
def daily_ic(df, ret_row, amt_row):
    """返回 (ic, amt_corr) 序列: 对 df 每列"""
    r = ret_row.reindex(df.index)
    if r.notna().sum() < 50:
        return None, None
    ics, amts = {}, {}
    for col in df.columns:
        v = df[col].astype(float)
        mask = v.notna() & r.notna() & np.isfinite(v)
        if mask.sum() < 50:
            ics[col] = np.nan
            amts[col] = np.nan
            continue
        ics[col] = stats.spearmanr(v[mask], r[mask]).statistic
        a = np.log(amt_row.reindex(df.index).astype(float))
        m2 = mask & a.notna() & np.isfinite(a)
        amts[col] = stats.spearmanr(v[m2], a[m2]).statistic if m2.sum() > 50 else np.nan
    return ics, amts


ic_rows = []
amt_rows = []
for d in DATES:
    df = frames[d]
    t = pd.Timestamp(str(d))
    if t not in rets.index:
        print(f"WARN {d} not in rets")
        continue
    ics, amts = daily_ic(df, rets.loc[t], amounts.loc[t])
    ic_rows.append(ics)
    amt_rows.append(amts)

ic_df = pd.DataFrame(ic_rows, index=[str(d) for d in DATES])
amt_df = pd.DataFrame(amt_rows, index=[str(d) for d in DATES])
print("== 每日 IC(方向D 各特征 vs 次日收益)==")
print(ic_df.round(4).to_string())
print("\n== 每日 |IC| ==")
print(ic_df.abs().round(4).to_string())

# ---------- 汇总 ----------
summary = pd.DataFrame(
    {
        "mean_ic": ic_df.mean(),
        "mean_abs_ic": ic_df.abs().mean(),
        "std_ic": ic_df.std(),
        "icir": ic_df.mean() / ic_df.std(),
        "pos_frac": (ic_df > 0).mean(),
        "mean_amt_corr": amt_df.mean(),
        "max_amt_corr": amt_df.abs().max(),
    }
)
print("\n== 汇总(5天) ==\n", summary.round(4).to_string())

# ---------- 与成交额正交化后的残差 IC ----------
resic_rows = []
for d in DATES:
    df = frames[d]
    t = pd.Timestamp(str(d))
    r = rets.loc[t].reindex(df.index)
    a = np.log(amounts.loc[t].reindex(df.index).astype(float))
    res = {}
    for col in df.columns:
        v = df[col].astype(float)
        m = v.notna() & r.notna() & np.isfinite(v) & a.notna() & np.isfinite(a)
        if m.sum() < 50:
            res[col] = np.nan
            continue
        # OLS 残差
        X = np.column_stack([np.ones(m.sum()), a[m].values])
        beta, *_ = np.linalg.lstsq(X, v[m].values, rcond=None)
        resid = v[m].values - X @ beta
        res[col] = stats.spearmanr(resid, r[m].values).statistic
    resic_rows.append(res)
resic_df = pd.DataFrame(resic_rows, index=[str(d) for d in DATES])
print("\n== 正交化后残差 IC(日) ==\n", resic_df.round(4).to_string())
summary["resic_mean"] = resic_df.mean()
summary["resic_icir"] = resic_df.mean() / resic_df.std()
print("\n== 汇总(含残差IC) ==\n", summary.round(4).to_string())

# ---------- 特征间相关(面板) ----------
p = pd.concat(frames, names=["date", "code"])
print("\n== 方向D 特征截面相关(面板 5天) ==")
feat_cols = [c for c in p.columns if c != "n_min"]
print(p[feat_cols].corr(method="spearman").round(3).to_string())

# ---------- 方向 C 汇总表 ----------
cs_df = pd.DataFrame(cs_all)
print("\n== 方向C/B 时刻级结果(5天×3时刻) ==")
print(
    cs_df[
        ["date", "t", "r", "n", "c_r0", "c_steps50", "c_steps70", "c_steps90",
         "c_final_r", "c_gini", "c_top5pct", "c_kurt", "b_r0", "b_steps70", "b_steps90", "b_ladder"]
    ].round(4).to_string()
)
# r vs c_steps70 冗余度
print("\n== 方向C: r 与 c_steps70 的 Spearman(15点) ==")
print("corr(r, c_steps70) =", stats.spearmanr(cs_df["r"], cs_df["c_steps70"]).statistic.round(3))
print("corr(r, c_final_r) =", stats.spearmanr(cs_df["r"], cs_df["c_final_r"]).statistic.round(3))
print("corr(c_gini, c_steps70) =", stats.spearmanr(cs_df["c_gini"], cs_df["c_steps70"]).statistic.round(3))
print("corr(c_top5pct, c_steps70) =", stats.spearmanr(cs_df["c_top5pct"], cs_df["c_steps70"]).statistic.round(3))
print("corr(c_gini, r) =", stats.spearmanr(cs_df["c_gini"], cs_df["r"]).statistic.round(3))
print("corr(c_kurt, c_steps70) =", stats.spearmanr(cs_df["c_kurt"], cs_df["c_steps70"]).statistic.round(3))
# 同 r 水平下的步数离散度(旧agent的CV=0 检验)
cs_df["is_low"] = pd.qcut(cs_df["r"], 2, labels=False)
print("\n== 同 r 半区内 c_steps70 的 CV(旧agent检验) ==")
for g in [0, 1]:
    sub = cs_df[cs_df["is_low"] == g]["c_steps70"]
    print(f"  r半区{g}: n={len(sub)} mean={sub.mean():.0f} std={sub.std():.0f} CV={sub.std()/sub.mean():.3f}")
