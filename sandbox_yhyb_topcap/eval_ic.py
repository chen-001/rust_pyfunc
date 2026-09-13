# -*- coding: utf-8 -*-
"""头部宇宙耦合因子的 IC 评估。

收益口径（重要）：read_daily(ret=1) 的 D 行是「当日收益」（D-1 收盘→D 收盘），
不是前瞻收益。正确做法：close = read_daily(close=1)；gap1 = close.shift(-1)/close - 1
（D 行 = D 到 D+1 的前瞻 1 日收益）、gap5 = close.shift(-5)/close - 1。
因子日 D（用 D 日 L2 数据算出的耦合因子）与 gap1/gap5 的 D 行对齐。
控制变量：D 日成交额（amounts.parquet）、D 日流通市值（flow_caps.parquet）、D 日事件数（rate）。
残差 IC：每股每日对 [1, log amount, log flow_cap, log(1+rate)] 截面 OLS 残差后再 Spearman。
"""
import json
import numpy as np
import pandas as pd
from pure_ocean_breeze.jason.data.read_data import read_daily
from scipy import stats

HERE = "/home/chenzongwei/rust_pyfunc/sandbox_yhyb_topcap"
DATES = [20240104, 20240603, 20241008, 20260105, 20260716]
SUF = lambda c: c + (".SH" if c.startswith(("6", "9")) else ".SZ")

close = read_daily(close=1)
gap1 = close.shift(-1) / close - 1
gap5 = close.shift(-5) / close - 1
amount = pd.read_parquet("/home/chenzongwei/database/daily_data/amounts.parquet")
flow_cap = pd.read_parquet("/home/chenzongwei/database/daily_data/flow_caps.parquet")


def spearman(f, r):
    m = f.notna() & r.notna() & np.isfinite(f)
    if m.sum() < 30:
        return np.nan, 0
    return stats.spearmanr(f[m], r[m]).statistic, int(m.sum())


def residualize(f, x1, x2, x3):
    """截面 OLS 残差：f ~ [1, x1, x2, x3]。"""
    m = f.notna() & x1.notna() & x2.notna() & x3.notna()
    if m.sum() < 30:
        return pd.Series(np.nan, index=f.index)
    X = np.column_stack([np.ones(m.sum()), x1[m], x2[m], x3[m]])
    y = f[m].values
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    out = pd.Series(np.nan, index=f.index)
    out[m] = resid
    return out


# 加载全部因子
frames = {}
for d in DATES:
    o = json.load(open(f"{HERE}/out_{d}.json"))
    codes = o["codes"]
    inds = o["inds"]
    events = o["events"]
    fnames = o["factor_names"]
    v = np.array(o["vals"], dtype=float)
    dd = pd.Timestamp(f"{d // 10000}-{(d // 100) % 100:02d}-{d % 100:02d}")
    idx = [SUF(c) for c in codes]
    df = pd.DataFrame(v, index=idx,
                      columns=[f"{e}__{f}" for e in events for f in fnames])
    df["_ind"] = inds
    df["_date"] = dd
    frames[d] = df

# 因子矩阵：[date][factor] -> Series
allcols = [c for c in frames[DATES[0]].columns if not c.startswith("_")]
cov = read_daily(close=1).loc[:, :].copy()  # unused, keep import tidy

rows = []
for d in DATES:
    dd = pd.Timestamp(f"{d // 10000}-{(d // 100) % 100:02d}-{d % 100:02d}")
    df = frames[d]
    g1 = gap1.loc[dd].reindex(df.index)
    g5 = gap5.loc[dd].reindex(df.index)
    am = amount.loc[dd].reindex(df.index)
    fc = flow_cap.loc[dd].reindex(df.index)
    df["_g1"] = g1
    df["_g5"] = g5
    df["_am"] = np.log(am + 1.0)
    df["_fc"] = np.log(fc + 1.0)
    for col in allcols:
        f = df[col]
        r = df["_g1"]
        ic, n = spearman(f, r)
        r5 = df["_g5"]
        ic5, _ = spearman(f, r5)
        # 残差 IC（控制 am/fc/rate）
        rate = df[f"{col.split('__')[0]}__rate"]
        resid = residualize(f, df["_am"], df["_fc"], np.log1p(rate))
        icr, _ = spearman(resid, r)
        # 残差 IC 只控制市值/成交额（不看 rate）
        resid2 = residualize(f, df["_am"], df["_fc"], pd.Series(0.0, index=df.index))
        icr2, _ = spearman(resid2, r)
        rows.append(dict(date=d, factor=col, ic=ic, ic5=ic5, ic_resid=icr, ic_resid_norate=icr2, n=n))

res = pd.DataFrame(rows)
res.to_csv(f"{HERE}/ic_results.csv", index=False)

# 汇总：mean IC / mean|IC| / ICIR / 正天占比
summ = []
for col in allcols:
    sub = res[res.factor == col]
    ic = sub.ic.dropna()
    ic5 = sub.ic5.dropna()
    icr = sub.ic_resid.dropna()
    icr2 = sub.ic_resid_norate.dropna()
    if len(ic) < 2:
        continue
    summ.append(dict(
        factor=col,
        n_days=len(ic),
        mean_ic=ic.mean(), mean_abs_ic=ic.abs().mean(), icir=ic.mean() / ic.std() if ic.std() > 0 else np.nan,
        pos_frac=(ic > 0).mean(),
        mean_ic5=ic5.mean(), mean_abs_ic5=ic5.abs().mean(),
        icir5=ic5.mean() / ic5.std() if ic5.std() > 0 else np.nan,
        pos_frac5=(ic5 > 0).mean(),
        mean_ic_resid=icr.mean(), mean_abs_resid=icr.abs().mean(), icir_resid=icr.mean() / icr.std() if icr.std() > 0 else np.nan,
        mean_ic_resid_norate=icr2.mean(), mean_abs_resid_norate=icr2.abs().mean(),
    ))
sm = pd.DataFrame(summ).sort_values("mean_abs_ic", ascending=False)
sm.to_csv(f"{HERE}/ic_summary.csv", index=False)
pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 200)
print(f"=== 共 {len(sm)} 个因子列（6 事件 × 14 因子） ===")
print(sm.to_string(index=False))
