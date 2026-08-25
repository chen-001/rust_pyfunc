"""想法B 沙箱因子 IC 评估:5 日全市场 Spearman rank-IC + 基线对照 + 正交残差 IC。

输入:sandbox_volnorm_portrait/data/out_{date}.json(Rust 产物,120 列)
      read_daily(ret=1) 次日收益;amounts.parquet 日成交额;day_/hm20_ CSV 算放量比
输出:控制台汇总表 + data/ic_summary.json
"""
import json
import os
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

# ---- 载入全部因子列 ----
cols_order = None
frames = {}
for d in DATES:
    j = json.load(open(f"{BASE}/out_{d}.json"))
    cols_order = j["names"]
    mat = np.array(j["vals"], dtype=float)
    f = pd.DataFrame(mat, index=j["codes"], columns=j["names"])
    # 放量比:当天量/20日均量(分钟 CSV,0..236 桶)
    day = pd.read_csv(f"{BASE}/day_{d}.csv", dtype={"code": str}).set_index("code")
    hm = pd.read_csv(f"{BASE}/hm20_{d}.csv", dtype={"code": str}).set_index("code")
    vr = np.nansum(day.values[:, :237], axis=1) / np.nansum(hm.values[:, :237], axis=1)
    f["_volratio"] = pd.Series(vr, index=day.index).reindex(f.index)
    ts = to_ts(d)
    f["_ret"] = r1.loc[ts].reindex([c + ".SZ" for c in f.index]).values
    f["_logamt"] = np.log(amt.loc[ts].reindex([c + ".SZ" for c in f.index]).values + 1.0)
    frames[d] = f

factor_names = [c for c in cols_order]  # 120 列
base_names = ["_volratio", "_logamt"]

def day_ic(d, fac, port):
    f = frames[d]
    m = np.isfinite(f[fac]) & np.isfinite(f[port])
    if m.sum() < MIN_N:
        return None
    rho, _ = spearmanr(f.loc[m, fac], f.loc[m, port])
    return rho if np.isfinite(rho) else None

def collect(fac, port):
    ics = []
    for d in DATES:
        v = day_ic(d, fac, port)
        if v is not None:
            ics.append(v)
    ics = np.array(ics)
    if len(ics) == 0:
        return dict(fac=fac, port=port, n=0)
    icir = ics.mean() / (ics.std(ddof=1) if len(ics) > 1 and ics.std(ddof=1) > 0 else np.nan)
    return dict(
        fac=fac, port=port, n=len(ics),
        mean_ic=round(float(ics.mean()), 4), mean_abs_ic=round(float(np.abs(ics).mean()), 4),
        icir=round(float(icir), 3), pos_frac=round(float((ics > 0).mean()), 2),
        n_above_2pct=int((np.abs(ics) > 0.02).sum()), ics=[round(float(x), 4) for x in ics],
    )

rows = []
for fac in factor_names:
    ic = collect(fac, "_ret")
    rows.append(ic)
# 基线:放量、成交额
for b in base_names:
    ic = collect(b, "_ret")
    ic["fac"] = b
    rows.append(ic)
res = pd.DataFrame(rows).set_index("fac")
res.to_json(f"{BASE}/ic_summary.json")

# ---- 汇总表:按类别 ----
def show(title, idx_cond):
    sub = res[res.index.map(idx_cond)]
    keep = ["mean_ic", "mean_abs_ic", "icir", "pos_frac", "n_above_2pct"]
    print("\n== " + title + " ==")
    print(sub[keep].sort_values("mean_abs_ic", ascending=False).to_string())

print("全部列 mean|IC| 分布: ", res["mean_abs_ic"].describe().round(3).to_dict())
show("1分钟桶(int 拟合)", lambda s: s.startswith("1m-") and "-int-" in s)
show("1分钟桶(lin 拟合)", lambda s: s.startswith("1m-") and "-lin-" in s)
show("10秒桶(int 拟合)", lambda s: s.startswith("10s-") and "-int-" in s)
show("平滑(int 拟合)", lambda s: s.startswith("sm-") and "-int-" in s)
show("价格版对照", lambda s: s.startswith("price-"))
show("基线", lambda s: s in base_names)

# ---- 正交化:候选因子对(放量, log成交额)回归取残差,再算 IC ----
top = res.sort_values("mean_abs_ic", ascending=False)
print("\n== TOP 15 (mean|IC|) ==")
print(top[["mean_ic", "mean_abs_ic", "icir", "pos_frac", "n_above_2pct"]].head(15).to_string())

def resid_ic(d, fac):
    f = frames[d]
    base = np.column_stack(
        [f["_volratio"], f["_logamt"]]
    )
    y = f[fac].to_numpy(float)
    ok = np.isfinite(y) & np.all(np.isfinite(base), axis=1) & np.isfinite(f["_ret"])
    if ok.sum() < MIN_N:
        return None
    X = np.column_stack([np.ones(ok.sum()), base[ok]])
    coef, *_ = np.linalg.lstsq(X, y[ok], rcond=None)
    resid = y[ok] - X @ coef
    rho, _ = spearmanr(resid, f["_ret"].to_numpy(float)[ok])
    return rho if np.isfinite(rho) else None

print("\n== 正交残差 IC(对放量+log成交额回归后的残差,对照 top 因子) ==")
for fac in list(top.index[:10]):
    raw = collect(fac, "_ret")
    ics = []
    for d in DATES:
        v = resid_ic(d, fac)
        if v is not None:
            ics.append(v)
    if ics:
        ics = np.array(ics)
        icir = ics.mean() / (ics.std(ddof=1) if len(ics) > 1 and ics.std(ddof=1) > 0 else np.nan)
        print(f"{fac:24s} raw |IC|={raw['mean_abs_ic']:.4f} mean={raw['mean_ic']:.4f}  ->  resid |IC|={np.abs(ics).mean():.4f} mean={ics.mean():.4f} icir={icir:.3f} pos={np.mean(ics>0):.2f} ics={[round(float(x),4) for x in ics]}")
    else:
        print(f"{fac:24s} resid n=0")
