"""拓展二沙箱评估：逐列 rank IC（gap1 / gap5），单日值、5 日平均值、5 日合并（pool）三种口径。
因子值全部来自 Rust 输出的 bin，这里只做检验，不合成因子。
产物：<O>/ic_w1.parquet（每天×每列）、ic_s5.parquet（5 日平均）、ic_p5.parquet（5 日合并，同日期）
"""
import json, sys, numpy as np, pandas as pd
from scipy.stats import rankdata
sys.path.insert(0, "/home/chenzongwei/.agents/skills/daily-data-reader")
import daily_data_reader as ddr

O = "/hdd/user_home_unsafe/chenzongwei/sandbox_lfsr_ext2_out"
dates = [int(x) for x in open(f"{O}/dates.txt").read().split()]
names = json.load(open(f"{O}/names_w1.json"))
names5 = json.load(open(f"{O}/names_w5.json"))
c = ddr.close(start=dates[0], end=20251231)
c.columns = [s[:6] for s in c.columns]
G = {"g1": c.shift(-1) / c - 1, "g5": c.shift(-5) / c - 1}


def load(d, w):
    codes = json.load(open(f"{O}/{d}_w{w}.codes.json"))
    m = np.fromfile(f"{O}/{d}_w{w}.bin", dtype=np.float32).reshape(len(codes), -1)
    return codes, m


def rank_ic(m, y):
    """每列与 y 的 Spearman（只在 y 有值的股票上；列内 NaN 剔除）"""
    ok = np.isfinite(y)
    m, y = m[ok], y[ok]
    x = rankdata(m, axis=0, nan_policy="omit").astype(np.float64)
    M = np.isfinite(x)
    x0 = np.where(M, x, 0.0)
    yr = rankdata(y).astype(np.float64)
    Mf = M.astype(np.float64)
    n = Mf.sum(0)
    sx, sxx = x0.sum(0), (x0 ** 2).sum(0)
    sy, syy, sxy = Mf.T @ yr, Mf.T @ (yr ** 2), x0.T @ yr
    with np.errstate(all="ignore"):
        cov = sxy / n - sx / n * sy / n
        r = cov / np.sqrt((sxx / n - (sx / n) ** 2) * (syy / n - (sy / n) ** 2))
    r[n < 500] = np.nan
    return r


def ics(codes, m, d):
    out = {}
    for g, df in G.items():
        y = df.loc[d].reindex(codes).values.astype(np.float64)
        out[g] = rank_ic(m, y)
    return out


rec_w1, rec_s5, rec_p5 = {g: {} for g in G}, {g: {} for g in G}, {g: {} for g in G}
hist = []  # 本段最近 5 天 (codes, m)
for i, d in enumerate(dates):
    if i % 25 == 0:
        hist = []
    codes, m = load(d, 1)
    for g, v in ics(codes, m, d).items():
        rec_w1[g][d] = v
    hist = (hist + [(codes, m)])[-5:]
    if len(hist) == 5:
        # 5 日平均：按当天股票对齐，至少 3 天有值
        idx = {cc: k for k, cc in enumerate(codes)}
        stack = np.full((5, len(codes), m.shape[1]), np.nan, dtype=np.float32)
        for t, (cs, mm) in enumerate(hist):
            rows = [(idx[cc], k) for k, cc in enumerate(cs) if cc in idx]
            a, b = zip(*rows)
            stack[t, list(a)] = mm[list(b)]
        cnt = np.isfinite(stack).sum(0)
        with np.errstate(all="ignore"):
            s5 = np.nanmean(stack, axis=0)
        s5[cnt < 3] = np.nan
        for g, v in ics(codes, s5, d).items():
            rec_s5[g][d] = v
        del stack, s5
    if (i % 25 + 1) % 5 == 0:
        try:
            cp, mp = load(d, 5)
            for g, v in ics(cp, mp, d).items():
                rec_p5[g][d] = v
        except FileNotFoundError:
            pass
    print(d, flush=True)

for tag, rec, nm in [("w1", rec_w1, names), ("s5", rec_s5, names), ("p5", rec_p5, names5)]:
    for g in G:
        if rec[g]:
            pd.DataFrame(rec[g], index=nm).T.to_parquet(f"{O}/ic_{tag}_{g}.parquet")
print("done")
