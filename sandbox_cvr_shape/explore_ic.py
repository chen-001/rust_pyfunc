"""CVR/CTR 形状因子 IC 评估（sandbox 探索）。
读 Rust 输出的每日 JSON → 对齐 read_daily(ret=1) 收益 → Spearman IC、
与成交额/换手的截面相关、正交化残差 IC，汇总多日统计。
"""
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from pure_ocean_breeze.jason.data.read_data import read_daily

DATES = [20240104, 20240603, 20240624, 20241008, 20260105, 20260605]
DATES_DESC = DATES + [20260717]  # 描述统计多含末日后
SDIR = "/home/chenzongwei/rust_pyfunc/sandbox_cvr_shape"

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
    """截面最小二乘残差：y ~ X（X 含常数项）。自动剔除 NaN 行。"""
    X = np.asarray(X, dtype=float)
    m = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    if m.sum() < 100:
        return np.full_like(y, np.nan)
    XX = np.column_stack([np.ones(m.sum()), X[m]])
    beta, *_ = np.linalg.lstsq(XX, y[m], rcond=None)
    out = np.full_like(y, np.nan)
    out[m] = y[m] - XX @ beta
    return out

def load_day(d):
    dt = pd.Timestamp(f"{d // 10000}-{(d // 100) % 100:02d}-{d % 100:02d}")
    j = json.load(open(f"{SDIR}/out_{d}.json"))
    df = pd.DataFrame(j["vals"], index=j["codes"], columns=j["names"])
    cols = [to_col(c) for c in df.index]
    keep = np.array([c is not None for c in cols], dtype=bool)
    df = df.iloc[keep]
    cols = [c for c in cols if c is not None]
    df.index = cols
    return dt, df

# ---------- 描述统计（5+1 天，截面统计平均） ----------
desc_frames = []
for d in DATES_DESC:
    _, df = load_day(d)
    desc_frames.append(df)
allf = pd.concat(desc_frames, keys=DATES_DESC)
desc = allf.groupby(level=1).agg(["mean", "std", lambda x: x.quantile(0.05), lambda x: x.quantile(0.5), lambda x: x.quantile(0.95)]).T
desc.index = [f"{a}_{b}" for a, b in desc.index]
print("==== 形状因子横截面描述统计（6 天均值） ====")
print(desc.round(4).to_string())
desc.to_csv(f"{SDIR}/desc_stats.csv")

# ---------- 因子间平均截面相关 ----------
cors = []
for d in DATES:
    _, df = load_day(d)
    cors.append(df.corr(method="spearman"))
avg_corr = pd.concat(cors).groupby(level=0).mean()
print("\n==== 因子间平均截面 Spearman 相关（部分） ====")
print(avg_corr.loc[["area_dev", "cum_open30", "cum_morning", "cum_close30", "t50", "t90", "hhi", "top5", "dcv_mean", "dcv_abs", "dcv_min", "cvr_int_cor"], ["area_dev", "cum_open30", "cum_morning", "cum_close30", "t50", "t90", "hhi", "top5", "dcv_mean", "dcv_abs", "dcv_min", "cvr_int_cor"]].round(3).to_string())
avg_corr.to_csv(f"{SDIR}/factor_corr.csv")

# ---------- IC 评估 ----------
per_day = []
for d in DATES:
    dt, df = load_day(d)
    pos = rets.index.get_loc(dt)
    y_same = rets.iloc[pos].reindex(df.index).to_numpy()  # D 行：D-1→D（同期）
    y_next = rets.iloc[pos + 1].reindex(df.index).to_numpy()  # D+1 行：D→D+1（真前瞻）
    amt = amounts.loc[dt].reindex(df.index).to_numpy()
    flow = flowcaps.loc[dt].reindex(df.index).to_numpy()
    turn = amt / flow
    log_amt = np.log(np.where(amt > 0, amt, np.nan))
    log_turn = np.log(np.where((turn > 0) & np.isfinite(turn), turn, np.nan))
    log_flow = np.log(np.where((flow > 0) & np.isfinite(flow), flow, np.nan))
    row = {}
    for nm in df.columns:
        f = df[nm].to_numpy()
        row[(nm, "ic_same")] = sp(f, y_same)
        row[(nm, "ic_next")] = sp(f, y_next)
        row[(nm, "rho_logamt")] = sp(f, log_amt)
        row[(nm, "rho_logturn")] = sp(f, log_turn)
        f1 = resid(f, np.column_stack([log_amt]))
        f2 = resid(f, np.column_stack([log_amt, log_flow]))
        row[(nm, "resid_ic1")] = sp(f1, y_next)
        row[(nm, "resid_ic2")] = sp(f2, y_next)
    row[("__amount__", "ic_next")] = sp(log_amt, y_next)
    row[("__turnover__", "ic_next")] = sp(log_turn, y_next)
    row[("__amount__", "ic_same")] = sp(log_amt, y_same)
    row[("__turnover__", "ic_same")] = sp(log_turn, y_same)
    per_day.append((d, row))

factors = sorted({k[0] for k in per_day[0][1] if not k[0].startswith("__")})

def agg(fac, metric):
    vals = np.array([r[1][(fac, metric)] for r in per_day], dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return (np.nan, np.nan, np.nan, 0, 0)
    mean_abs = np.nanmean(np.abs(vals))
    ic_mean = np.nanmean(vals)
    ic_std = np.nanstd(vals)
    icir = ic_mean / ic_std if ic_std > 0 else np.nan
    return (mean_abs, ic_mean, icir, int((vals > 0).sum()), len(vals))

print("\n==== IC 汇总（next-day 前瞻；pos_days = 正天数/有效天数） ====")
lines = []
lines.append(f"{'factor':<16}{'mean|IC|':>9}{'IC_mean':>9}{'ICIR':>8}{'pos_days':>10}{'rho_amt':>9}{'rho_turn':>9}{'residIC1':>9}{'residIC2':>9}")
saved = {}
for fac in factors:
    a = agg(fac, "ic_next")
    rho = agg(fac, "rho_logamt")
    rt = agg(fac, "rho_logturn")
    r1 = agg(fac, "resid_ic1")
    r2 = agg(fac, "resid_ic2")
    lines.append(f"{fac:<16}{a[0]:>9.4f}{a[1]:>9.4f}{a[2]:>8.2f}{a[3]:>3}/{a[4]:<5}{rho[0]:>9.3f}{rt[0]:>9.3f}{r1[0]:>9.4f}{r2[0]:>9.4f}")
    saved[fac] = dict(ic_next=a, ic_same=agg(fac, "ic_same"), rho_amt=rho, rho_turn=rt, resid1=r1, resid2=r2)
for fac in ("__amount__", "__turnover__"):
    a = agg(fac, "ic_next")
    lines.append(f"BASE {fac[2:]:<9}{a[0]:>9.4f}{a[1]:>9.4f}{a[2]:>8.2f}{a[3]:>3}/{a[4]:<5}")
print("\n".join(lines))

print("\n==== daily IC (next-day) ====")
print("date     " + "  ".join(f"{f:>16}" for f in factors))
for d, row in per_day:
    print(f"{d}  " + "  ".join(f"{row[(f, 'ic_next')]:>16.4f}" for f in factors))

print("\n==== daily IC (同日 ret) ====")
print("date     " + "  ".join(f"{f:>16}" for f in ["area_dev", "cum_open30", "cum_morning", "cum_close30", "dcv_mean", "dcv_abs", "dcv_min", "cvr_int_cor", "t50", "t90"]))
for d, row in per_day:
    print(f"{d}  " + "  ".join(f"{row[(f, 'ic_same')]:>16.4f}" for f in ["area_dev", "cum_open30", "cum_morning", "cum_close30", "dcv_mean", "dcv_abs", "dcv_min", "cvr_int_cor", "t50", "t90"]))

json.dump({nm: {k: list(v) if isinstance(v, tuple) else v for k, v in d.items()} for nm, d in saved.items()},
          open(f"{SDIR}/ic_summary.json", "w"), indent=1, default=str)
pd.DataFrame({f"{d}": {k: v for k, v in row.items()} for d, row in per_day}).to_csv(f"{SDIR}/ic_daily.csv")
print("\nsaved ic_summary.json / ic_daily.csv / desc_stats.csv / factor_corr.csv")
