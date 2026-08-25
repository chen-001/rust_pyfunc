"""补充分析:量版因子正交残差 IC、共线性检查、每日 IC 明细、横截面描述统计。"""
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

BASE = "/home/chenzongwei/rust_pyfunc/sandbox_volnorm_portrait/data"
DATES = [20240104, 20240603, 20241008, 20260105, 20260717]
MIN_N = 100
from pure_ocean_breeze.jason.data.read_data import read_daily
r1 = read_daily(ret=1)
amt = pd.read_parquet("/home/chenzongwei/database/daily_data/amounts.parquet")

def to_ts(d):
    return pd.Timestamp(f"{d // 10000}-{(d // 100) % 100:02d}-{d % 100:02d}")

frames = {}
for d in DATES:
    j = json.load(open(f"{BASE}/out_{d}.json"))
    mat = np.array(j["vals"], dtype=float)
    f = pd.DataFrame(mat, index=j["codes"], columns=j["names"])
    day = pd.read_csv(f"{BASE}/day_{d}.csv", dtype={"code": str}).set_index("code")
    hm = pd.read_csv(f"{BASE}/hm20_{d}.csv", dtype={"code": str}).set_index("code")
    vr = np.nansum(day.values[:, :237], axis=1) / np.nansum(hm.values[:, :237], axis=1)
    f["_volratio"] = pd.Series(vr, index=day.index).reindex(f.index)
    ts = to_ts(d)
    f["_ret"] = r1.loc[ts].reindex([c + ".SZ" for c in f.index]).values
    f["_logamt"] = np.log(amt.loc[ts].reindex([c + ".SZ" for c in f.index]).values + 1.0)
    frames[d] = f

def mask(f, fac, base_cols):
    m = np.isfinite(f[fac]) & np.isfinite(f["_ret"]) & np.all(np.isfinite(f[base_cols]), axis=1)
    return m

def resid_ic(f, fac, base_cols):
    m = mask(f, fac, base_cols)
    if m.sum() < MIN_N:
        return None
    X = np.column_stack([np.ones(m.sum()), f.loc[m, base_cols].to_numpy(float)])
    y = f.loc[m, fac].to_numpy(float)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    rho, _ = spearmanr(resid, f.loc[m, "_ret"].to_numpy(float))
    return rho if np.isfinite(rho) else None

def cross_corr(f, a, b):
    m = np.isfinite(f[a]) & np.isfinite(f[b]) & np.isfinite(f["_ret"])
    if m.sum() < MIN_N:
        return None
    rho, _ = spearmanr(f.loc[m, a], f.loc[m, b])
    return rho if np.isfinite(rho) else None

CAND = [
    "1m-DevA-N20-int-cor", "1m-DevV-N20-int-cor", "1m-DevA-N10-int-cor",
    "sm-DevA-N20-int-cor", "sm-DevV-N20-int-cor", "10s-DevA-N20-int-cor", "10s-DevV-N20-int-cor",
    "1m-V-N20-int-cor", "sm-V-N20-int-cor", "10s-DevA-N10-int-cor",
    "1m-V-N20-int-dev", "sm-V-N10-int-dev", "10s-DevA-N20-int-dev",
]
B1 = ["_volratio", "_logamt"]
B2 = ["_volratio", "_logamt", "price-V-N20-lin-cor"]
B3 = ["price-V-N20-lin-cor"]  # 只对价格版

print("=== 量版 top 因子的残差 IC ===")
print(f"{'factor':26s} {'raw_mIC':>8s} {'raw_ICIR':>8s} | {'resid(vr+amt)':>14s} | {'resid(+price)':>14s} | {'resid(price only)':>18s}") 
for fac in CAND:
    ics = [day_ic for day_ic in []]
    raw = []
    for d in DATES:
        f = frames[d]
        m = np.isfinite(f[fac]) & np.isfinite(f["_ret"])
        if m.sum() >= MIN_N:
            rho, _ = spearmanr(f.loc[m, fac], f.loc[m, "_ret"])
            if np.isfinite(rho):
                raw.append(rho)
    raw = np.array(raw)
    raw_mic = np.abs(raw).mean()
    raw_icir = raw.mean() / raw.std(ddof=1) if len(raw) > 1 and raw.std(ddof=1) > 0 else np.nan

    r1_ics = [resid_ic(frames[d], fac, B1) for d in DATES]
    r1_ics = np.array([x for x in r1_ics if x is not None])
    r2_ics = [resid_ic(frames[d], fac, B2) for d in DATES]
    r2_ics = np.array([x for x in r2_ics if x is not None])
    r3_ics = [resid_ic(frames[d], fac, B3) for d in DATES]
    r3_ics = np.array([x for x in r3_ics if x is not None])

    def fmt(a):
        if len(a) == 0:
            return "n/a"
        icir = a.mean() / (a.std(ddof=1) if len(a) > 1 and a.std(ddof=1) > 0 else np.nan)
        return f"{a.mean():+.4f}({icir:+.2f})"
    print(f"{fac:26s} {raw_mic:8.4f} {raw_icir:+8.2f} | {fmt(r1_ics):>14s} | {fmt(r2_ics):>14s} | {fmt(r3_ics):>18s}")

print("\n=== 横截面相关性(5 天均值) ===")
PAIRS = [
    ("1m-DevA-N20-int-cor", "price-V-N20-lin-cor"),
    ("1m-DevA-N20-int-cor", "_volratio"),
    ("1m-DevA-N20-int-cor", "_logamt"),
    ("1m-V-N20-int-dev", "_volratio"),
    ("1m-V-N20-int-dev", "_logamt"),
    ("10s-DevA-N20-int-dev", "_volratio"),
    ("price-V-N20-lin-cor", "_volratio"),
]
for a, b in PAIRS:
    rhos = [cross_corr(frames[d], a, b) for d in DATES]
    rhos = [x for x in rhos if x is not None]
    print(f"{a:26s} vs {b:26s} rho={np.mean(rhos):+.3f}  (each: {[round(float(x),3) for x in rhos]})")

print("\n=== 每日 IC 明细(关键因子) ===")
KEYS = [
    "1m-DevA-N20-int-cor", "1m-DevV-N20-int-cor", "1m-DevA-N10-int-cor",
    "sm-DevA-N20-int-cor", "10s-DevA-N20-int-cor", "1m-V-N20-int-dev",
    "price-V-N20-lin-cor", "price-V-N20-lin-dev",
]
for fac in KEYS:
    row = []
    for d in DATES:
        f = frames[d]
        m = np.isfinite(f[fac]) & np.isfinite(f["_ret"])
        if m.sum() >= MIN_N:
            rho, _ = spearmanr(f.loc[m, fac], f.loc[m, "_ret"])
            row.append(round(float(rho), 4) if np.isfinite(rho) else None)
        else:
            row.append(None)
    print(f"{fac:26s} " + " ".join(f"{x:+.4f}" if x is not None else "  ----" for x in row))

print("\n=== 横截面描述统计(因子值,5 天平均) ===")
for fac in KEYS[:7]:
    stats = []
    for d in DATES:
        v = frames[d][fac].to_numpy(float)
        v = v[np.isfinite(v)]
        stats.append((len(v), np.nanstd(v), np.nanmedian(v)))
    n = np.mean([s[0] for s in stats])
    sd = np.mean([s[1] for s in stats])
    md = np.mean([s[2] for s in stats])
    print(f"{fac:26s} n={n:.0f} std={sd:.4f} median={md:.4f}")
