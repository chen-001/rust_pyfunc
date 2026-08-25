"""方向D逐股退火特征 IC 评估(v2, 修正收益口径)

【口径更正】read_daily(ret=1) 第 D 行 = 当日收益(D-1收→D收), 不是前瞻收益!
正确: close = read_daily(close=1); gap1 = close.shift(-1)/close-1 (第D行=D→D+1);
      gap5 = close.shift(-5)/close-1 (第D行=D→D+5)。
本脚本同时给出"当日收益 IC"(旧错误口径, 仅作对照)与"前瞻 gap1/gap5 IC"(正确口径)。
"""
import json
import numpy as np
import pandas as pd
from scipy import stats

DATES = [20240104, 20240603, 20241008, 20260105, 20260717]
OUT_DIR = "/home/chenzongwei/rust_pyfunc/sandbox_switch_anneal/out"
from pure_ocean_breeze.jason.data.read_data import read_daily

close = read_daily(close=1)
ret_same = close / close.shift(1) - 1        # 旧错误口径: 当日收益
gap1 = close.shift(-1) / close - 1           # 正确: 前瞻1日
gap5 = close.shift(-5) / close - 1           # 正确: 前瞻5日
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
for d in DATES:
    js = json.load(open(f"{OUT_DIR}/out_{d}.json"))
    df = pd.DataFrame(js["d_vals"], index=js["d_codes"], columns=js["d_names"])
    df.index = pd.Index([to_suffix(c) for c in df.index], name="code")
    frames[d] = df


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


ALL_F = [c for c in frames[20240104].columns if c != "n_min"]
CTRL = ["r0", "std", "ac1", "trend"]
FEATS = [c for c in ALL_F if c not in CTRL + ["mean", "morn_aft_diff"]]

print("=== 日 IC 明细: 方向D 各特征 × [当日(旧口径) / gap1 / gap5] ===")
for lab, ret_s in [("当日(旧,仅对照)", ret_same), ("gap1(正确)", gap1), ("gap5(正确)", gap5)]:
    print(f"\n--- {lab} ---")
    mat = []
    for d in DATES:
        t = pd.Timestamp(str(d))
        row = {c: col_ic(frames[d], c, ret_s, t) for c in ALL_F}
        mat.append(row)
    ic_df = pd.DataFrame(mat, index=[str(d) for d in DATES])
    print(ic_df.round(4).to_string())
    s = ic_df.mean()
    print("汇总: " + "  ".join(
        f"{c}: mean_ic={s[c]:+.4f} abs={ic_df[c].abs().mean():.4f} "
        f"icir={s[c]/ic_df[c].std():+.3f} pos={float((ic_df[c]>0).mean()):.2f}"
        for c in ALL_F
    ))

print("\n=== 成交额截面相关(5天均值, 不变) ===")
amt_rows = pd.DataFrame(
    [{c: col_amt_corr(frames[d], c, pd.Timestamp(str(d))) for c in ALL_F} for d in DATES],
    index=[str(d) for d in DATES],
)
print(amt_rows.mean().round(4).to_string())

print("\n=== 正交掉 [r0,std,ac1,trend] 后的残差 IC(gap1 正确口径) ===")
rows = []
for d in DATES:
    t = pd.Timestamp(str(d))
    rows.append({c: col_resic(frames[d], c, CTRL, gap1, t) for c in FEATS + ["mean", "morn_aft_diff"]})
res1 = pd.DataFrame(rows, index=[str(d) for d in DATES])
print(res1.round(4).to_string())
s = res1.mean()
print("汇总: " + "  ".join(
    f"{c}: mean_ic={s[c]:+.4f} abs={res1[c].abs().mean():.4f} "
    f"icir={s[c]/res1[c].std():+.3f} pos={float((res1[c]>0).mean()):.2f}"
    for c in res1.columns
))

print("\n=== 正交后残差 IC(gap5 正确口径) ===")
rows = []
for d in DATES:
    t = pd.Timestamp(str(d))
    rows.append({c: col_resic(frames[d], c, CTRL, gap5, t) for c in FEATS + ["mean", "morn_aft_diff"]})
res5 = pd.DataFrame(rows, index=[str(d) for d in DATES])
print(res5.round(4).to_string())
s = res5.mean()
print("汇总: " + "  ".join(
    f"{c}: mean_ic={s[c]:+.4f} abs={res5[c].abs().mean():.4f} "
    f"icir={s[c]/res5[c].std():+.3f} pos={float((res5[c]>0).mean()):.2f}"
    for c in res5.columns
))
