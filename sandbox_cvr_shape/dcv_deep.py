"""dcv 系列深度检验：控制当日收益+规模后的增量 IC，以及关键因子描述统计。"""
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from pure_ocean_breeze.jason.data.read_data import read_daily

DATES = [20240104, 20240603, 20240624, 20241008, 20260105, 20260605]
SDIR = "/home/chenzongwei/rust_pyfunc/sandbox_cvr_shape"
KEYS = ["area_dev", "max_dev", "cum_open30", "cum_morning", "cum_close30", "t50", "t90",
        "hhi", "top5", "dcv_mean", "dcv_abs", "dcv_min", "dcv_max", "cvr_int_cor", "cvr_lin_cor"]

rets = read_daily(ret=1)
amounts = pd.read_parquet("/home/chenzongwei/database/daily_data/amounts.parquet")
flowcaps = pd.read_parquet("/home/chenzongwei/database/daily_data/flow_caps.parquet")

def to_col(code):
    for suf in ("SZ", "SH", "BJ"):
        c = code + "." + suf
        if c in rets.columns:
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

rows, desc_rows = [], []
for d in DATES:
    dt = pd.Timestamp(f"{d // 10000}-{(d // 100) % 100:02d}-{d % 100:02d}")
    j = json.load(open(f"{SDIR}/out_{d}.json"))
    df = pd.DataFrame(j["vals"], index=j["codes"], columns=j["names"])
    cols = [to_col(c) for c in df.index]
    keep = np.array([c is not None for c in cols], dtype=bool)
    df = df.iloc[keep]
    df.index = [c for c in cols if c is not None]
    pos = rets.index.get_loc(dt)
    y_same = rets.iloc[pos].reindex(df.index).to_numpy()
    y_next = rets.iloc[pos + 1].reindex(df.index).to_numpy()
    amt = amounts.loc[dt].reindex(df.index).to_numpy()
    flow = flowcaps.loc[dt].reindex(df.index).to_numpy()
    turn = amt / flow
    log_amt = np.log(np.where(amt > 0, amt, np.nan))
    log_turn = np.log(np.where((turn > 0) & np.isfinite(turn), turn, np.nan))
    log_flow = np.log(np.where((flow > 0) & np.isfinite(flow), flow, np.nan))
    r = {}
    for k in KEYS:
        f = df[k].to_numpy()
        r[(k, "ic_same")] = sp(f, y_same)
        r[(k, "ic_next")] = sp(f, y_next)
        f2 = resid(f, np.column_stack([log_amt, log_flow, y_same]))
        r[(k, "resid3_ic")] = sp(f2, y_next)
        f3 = resid(f, np.column_stack([log_amt, y_same]))
        r[(k, "resid2_ic")] = sp(f3, y_next)
        r[(k, "desc_mean")] = np.nanmean(f)
        r[(k, "desc_std")] = np.nanstd(f)
        r[(k, "desc_p5")] = np.nanpercentile(f, 5)
        r[(k, "desc_p50")] = np.nanpercentile(f, 50)
        r[(k, "desc_p95")] = np.nanpercentile(f, 95)
    rows.append((d, r))

def agg(key):
    v = np.array([r[1][key] for r in rows], dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return (np.nan, np.nan, np.nan, 0, 0)
    return (np.nanmean(np.abs(v)), np.nanmean(v), np.nanmean(v) / np.nanstd(v) if np.nanstd(v) > 0 else np.nan, int((v > 0).sum()), len(v))

print("==== 日均截面描述统计（6 天均值） ====")
print(f"{'factor':<14}{'mean':>9}{'std':>9}{'p5':>9}{'p50':>9}{'p95':>9}")
for k in KEYS:
    m = agg((k, "desc_mean"))
    s = agg((k, "desc_std"))
    p5 = agg((k, "desc_p5")); p50 = agg((k, "desc_p50")); p95 = agg((k, "desc_p95"))
    print(f"{k:<14}{m[0]:>9.4f}{s[0]:>9.4f}{p5[0]:>9.4f}{p50[0]:>9.4f}{p95[0]:>9.4f}")

print("\n==== IC 相关（next-day；resid3 = 控制 log(amount)+log(flowcap)+当日收益后） ====")
print(f"{'factor':<14}{'ic_same':>9}{'ic_next':>9}{'resid2_ic':>10}{'resid3_ic':>10}{'pos3/6':>8}")
for k in KEYS:
    a = agg((k, "ic_same"))
    b = agg((k, "ic_next"))
    c = agg((k, "resid2_ic"))
    d = agg((k, "resid3_ic"))
    print(f"{k:<14}{a[1]:>9.4f}{b[1]:>9.4f}{c[1]:>10.4f}{d[1]:>10.4f}{d[3]:>4}/{d[4]:<3}")

print("\n==== 每日明细 ====")
print("date     factor        ic_same  ic_next  resid3_ic")
for dd, r in rows:
    for k in KEYS:
        if k in ("dcv_mean", "dcv_abs", "dcv_min", "dcv_max", "cum_close30", "t90", "cvr_int_cor"):
            print(f"{dd}  {k:<14}{r[(k,'ic_same')]:>9.4f}{r[(k,'ic_next')]:>9.4f}{r[(k,'resid3_ic')]:>10.4f}")

pd.DataFrame([{f"{k}_{m}": r1[1][(k, m)] for k in KEYS for m in ("ic_same", "ic_next", "resid2_ic", "resid3_ic")} for d1, r1 in rows], index=DATES).to_csv(f"{SDIR}/dcv_deep.csv")
print("\nsaved dcv_deep.csv")
