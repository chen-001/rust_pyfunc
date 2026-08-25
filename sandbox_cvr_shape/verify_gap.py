"""收益口径校验与重算（用户指定官方口径）：
gap1 = close.shift(-1)/close - 1 的第 D 行 = D→D+1 前瞻收益。
对比：1) 与我此前用的 rets.iloc[pos+1] 是否逐值一致；2) 用官方口径重算 6 天 IC 汇总；
3) 顺带计算 gap5（5 日前瞻）作为稳健性补充。
"""
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from pure_ocean_breeze.jason.data.read_data import read_daily

DATES = [20240104, 20240603, 20240624, 20241008, 20260105, 20260605]
SDIR = "/home/chenzongwei/rust_pyfunc/sandbox_cvr_shape"

close = read_daily(close=1)
gap1 = close.shift(-1) / close - 1   # 官方口径：D 行 = D→D+1 前瞻 1 日
gap5 = close.shift(-5) / close - 1   # D 行 = D→D+5 前瞻 5 日
rets = read_daily(ret=1)             # 旧口径：D 行 = D-1→D 当日收益

amounts = pd.read_parquet("/home/chenzongwei/database/daily_data/amounts.parquet")
flowcaps = pd.read_parquet("/home/chenzongwei/database/daily_data/flow_caps.parquet")

def to_col(code):
    for suf in ("SZ", "SH", "BJ"):
        c = code + "." + suf
        if c in close.columns:
            return c
    return None

def sp(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 50:
        return np.nan
    return spearmanr(a[m], b[m]).statistic

def resid(y, X):
    X = np.asarray(X, dtype=float)
    m = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    if m.sum() < 100:
        return np.full_like(y, np.nan)
    XX = np.column_stack([np.ones(m.sum()), X[m]])
    beta, *_ = np.linalg.lstsq(XX, y[m], rcond=None)
    out = np.full_like(y, np.nan)
    out[m] = y[m] - XX @ beta
    return out

# ---------- 校验：官方 gap1 的 D 行 vs rets.iloc[pos+1] ----------
maxdiff = 0.0
for d in DATES:
    dt = pd.Timestamp(f"{d // 10000}-{(d // 100) % 100:02d}-{d % 100:02d}")
    pos = rets.index.get_loc(dt)
    a = gap1.loc[dt]
    b = rets.iloc[pos + 1]
    m = a.notna() & b.notna()
    maxdiff = max(maxdiff, np.nanmax(np.abs(a[m] - b[m])))
print(f"[校验] 官方gap1(D行) vs rets.iloc[D+1]: 最大绝对差 = {maxdiff:.2e}  -> 完全一致" if maxdiff < 1e-9 else f"[校验] 不一致! maxdiff={maxdiff}")

# ---------- 官方口径重算 IC ----------
per_day = []
for d in DATES:
    dt = pd.Timestamp(f"{d // 10000}-{(d // 100) % 100:02d}-{d % 100:02d}")
    j = json.load(open(f"{SDIR}/out_{d}.json"))
    df = pd.DataFrame(j["vals"], index=j["codes"], columns=j["names"])
    cols = [to_col(c) for c in df.index]
    keep = np.array([c is not None for c in cols], dtype=bool)
    df = df.iloc[keep]
    df.index = [c for c in cols if c is not None]
    g1 = gap1.loc[dt].reindex(df.index).to_numpy()
    g5 = gap5.loc[dt].reindex(df.index).to_numpy()
    same = rets.loc[dt].reindex(df.index).to_numpy()  # 当日收益（仅诊断用）
    amt = amounts.loc[dt].reindex(df.index).to_numpy()
    flow = flowcaps.loc[dt].reindex(df.index).to_numpy()
    log_amt = np.log(np.where(amt > 0, amt, np.nan))
    log_flow = np.log(np.where(flow > 0, flow, np.nan))
    row = {}
    for nm in df.columns:
        f = df[nm].to_numpy()
        row[(nm, "ic_gap1")] = sp(f, g1)
        row[(nm, "ic_gap5")] = sp(f, g5)
        row[(nm, "ic_same")] = sp(f, same)  # 诊断：与当日收益（前视口径）的对照
        row[(nm, "resid3_gap1")] = sp(resid(f, np.column_stack([log_amt, log_flow, same])), g1)
        row[(nm, "resid3_gap5")] = sp(resid(f, np.column_stack([log_amt, log_flow, same])), g5)
        row[(nm, "rho_amt")] = sp(f, log_amt)
    row[("__amt__", "ic_gap1")] = sp(log_amt, g1)
    row[("__amt__", "ic_gap5")] = sp(log_amt, g5)
    per_day.append((d, row))

def agg(fac, m):
    v = np.array([r[1][(fac, m)] for r in per_day], dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return (np.nan, np.nan, np.nan, 0, 0)
    return (np.nanmean(np.abs(v)), np.nanmean(v), np.nanmean(v) / np.nanstd(v) if np.nanstd(v) > 0 else np.nan, int((v > 0).sum()), len(v))

factors = sorted({k[0] for k in per_day[0][1] if not k[0].startswith("__")})
print("\n==== 官方口径 gap1 重算（D 行 = D→D+1 前瞻；对比旧表） ====")
print(f"{'factor':<15}{'mean|IC|':>9}{'IC_mean':>9}{'ICIR':>7}{'pos':>5}{'resid3':>9}{'rho_amt':>9}")
old = {"dcv_abs": (0.1687, -0.1687, -1.36, 0), "dcv_min": (0.1268, 0.1268, 0.96, 6), "hhi": (0.1157, 0.0451, 0.35, 4),
       "cum_open30": (0.0812, -0.0524, -0.68, 2), "cvr_int_cor": (0.0359, 0.0262, 0.80, 5)}
for fac in factors:
    a = agg(fac, "ic_gap1")
    r3 = agg(fac, "resid3_gap1")
    ra = agg(fac, "rho_amt")
    o = old.get(fac, ("", "", "", ""))
    print(f"{fac:<15}{a[0]:>9.4f}{a[1]:>9.4f}{a[2]:>7.2f}{a[3]:>3}/{a[4]:<3}{r3[0]:>9.4f}{ra[0]:>9.3f}  (旧表 {o[0]} / {o[3]}正)")

print("\n==== gap5（5 日前瞻）稳健性 ====")
print(f"{'factor':<15}{'mean|IC|':>9}{'IC_mean':>9}{'ICIR':>7}{'pos':>5}{'resid3':>9}")
for fac in ["dcv_abs", "dcv_min", "dcv_mean", "hhi", "top5", "cum_open30", "cum_close30", "t90", "cvr_int_cor", "cvr_lin_cor"]:
    a = agg(fac, "ic_gap5")
    r3 = agg(fac, "resid3_gap5")
    print(f"{fac:<15}{a[0]:>9.4f}{a[1]:>9.4f}{a[2]:>7.2f}{a[3]:>3}/{a[4]:<3}{r3[0]:>9.4f}")
a = agg("__amt__", "ic_gap5")
print(f"{'__amt__':<15}{a[0]:>9.4f}{a[1]:>9.4f}{a[2]:>7.2f}{a[3]:>3}/{a[4]:<3}")

# 每日明细（gap1）
print("\n==== 每日 IC（官方 gap1）关键因子 ====")
print("date     dcv_abs  dcv_min  hhi  cum_open30  cum_close30  t90  cvr_int_cor")
for d, row in per_day:
    print(f"{d}  " + "  ".join(f"{row[(k,'ic_gap1')]:>+9.4f}" for k in ["dcv_abs", "dcv_min", "hhi", "cum_open30", "cum_close30", "t90", "cvr_int_cor"]))
print("\n==== 每日残差³（gap1）====  (残差³ = 去log(成交额)+log(市值)+当日收益)")
for d, row in per_day:
    print(f"{d}  " + "  ".join(f"{row[(k,'resid3_gap1')]:>+9.4f}" for k in ["dcv_abs", "dcv_min", "hhi", "cum_open30", "cvr_int_cor"]))

pd.DataFrame({f"{d}": {k: v for k, v in r2.items()} for d, r2 in per_day}).to_csv(f"{SDIR}/ic_gap_official.csv")
print("\nsaved ic_gap_official.csv")
