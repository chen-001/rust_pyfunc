"""独立复核：真实因子数据上验证 IC 分组贡献分解恒等式 + 空头半边占比。

数据来自 empirics teammate 生成的 returns_cache.npz（只读）。
"""
import glob
import numpy as np
import pandas as pd

CACHE = "/home/chenzongwei/rust_pyfunc/tmp_mono_metric/empirics/returns_cache.npz"
FOLDERS = ["/nas197/user_home_unsafe/chenzongwei/factor_data/hm100",
           "/nas197/user_home_unsafe/chenzongwei/factor_data/hm101"]

z = np.load(CACHE, allow_pickle=True)
dates = z["dates"]; codes = z["codes"]; r5 = z["r5"]
dpos = {int(d): i for i, d in enumerate(dates)}
cpos = {c: i for i, c in enumerate(codes)}


def avg_rank_1d(x):
    o = np.argsort(x, kind="stable"); r = np.empty(len(x))
    r[o] = np.arange(1, len(x) + 1, dtype=float)
    v, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    if (cnt > 1).any():
        s = np.zeros(len(cnt)); np.add.at(s, inv, r); r = (s / cnt)[inv]
    return r


def ladder(r):
    r = np.asarray(r, float); st = np.diff(r); tot = np.abs(st).sum()
    ts, tl = np.abs(st[:5]).sum(), np.abs(st[4:]).sum()
    ls = (r[5] - r[0]) / ts if ts else 0.0
    ll = (r[9] - r[4]) / tl if tl else 0.0
    return ((r[9] - r[0]) / tot if tot else 0.0), min(ls, ll)


rows = []
files = sorted(glob.glob(FOLDERS[0] + "/*.parquet"))[:8] + \
        sorted(glob.glob(FOLDERS[1] + "/*.parquet"))[:8]

for fp in files:
    df = pd.read_parquet(fp)
    df = df.rename(columns={"date": "_d"})
    df["_d"] = pd.to_datetime(df["_d"]).dt.strftime("%Y%m%d").astype(int)
    cols = [c for c in df.columns if c != "_d"]
    bare = np.array([c.split(".")[0] for c in cols])
    keep = np.array([c in cpos for c in bare])
    bare, cols = bare[keep], np.array(cols)[keep]
    colidx = np.array([cpos[c] for c in bare])
    df = df[["_d"] + list(cols)].set_index("_d")

    ics, cs, ws, shares, lams, ssms, gammas, groups = [], [], [], [], [], [], [], []
    for d, row in df.iterrows():
        i = dpos.get(int(d))
        if i is None:
            continue
        f = row.to_numpy(dtype=float)
        rr = r5[i, colidx]
        ok = ~np.isnan(f) & ~np.isnan(rr)
        if ok.sum() < 200:
            continue
        f, rr = f[ok], rr[ok]
        n = len(f)
        q = (avg_rank_1d(f) - 0.5) / n
        y = (avg_rank_1d(rr) - 0.5) / n
        ic = 12.0 * (q @ y / n - 0.25)
        g = np.clip((avg_rank_1d(f) * 10 // (n + 1)).astype(int), 0, 9)
        c = np.array([12.0 * (g == k).mean() * (q[g == k].mean() - .5) * (y[g == k].mean() - .5)
                      for k in range(10)])
        w = 0.0
        for k in range(10):
            m = g == k
            if m.sum() > 2:
                w += 12.0 * m.mean() * float(np.cov(q[m], y[m], bias=True)[0, 1])
        ics.append(ic); cs.append(c); ws.append(w)
        shares.append(c[:5].sum() / c.sum() if abs(c.sum()) > 1e-12 else np.nan)
        gm = np.array([rr[g == k].mean() for k in range(10)])
        la, ss = ladder(gm)
        lams.append(la); ssms.append(ss)
        groups.append(gm)
        ga = np.corrcoef(avg_rank_1d(np.arange(1, 11)), avg_rank_1d(gm))[0, 1]
        gammas.append(ga)

    C = np.array(cs)
    resid = float(np.max(np.abs(np.array(ics) - (C.sum(1) + np.array(ws)))))
    rows.append(dict(
        factor=fp.split("/")[-1].replace(".parquet", "")[:52],
        ic=float(np.mean(ics)),
        decomp_resid=resid,
        share=float(np.nanmean(shares)),
        lam=float(np.mean(lams)), ssm=float(np.mean(ssms)),
        gamma=float(np.mean(gammas)),
        r1_bp=float(np.mean([g[0] for g in groups]) * 1e4),
        r5_bp=float(np.mean([g[4] for g in groups]) * 1e4),
        r6_bp=float(np.mean([g[5] for g in groups]) * 1e4),
        r10_bp=float(np.mean([g[9] for g in groups]) * 1e4),
        c1_3=float(np.mean([c[:3].sum() for c in cs])),
        c8_10=float(np.mean([c[7:].sum() for c in cs])),
    ))
    print(f"done {rows[-1]['factor']}")

out = pd.DataFrame(rows)
pd.set_option("display.width", 220, "display.max_columns", 30)
print("\n" + "=" * 120)
print("[复核] 真实数据上的 IC 分解与阶梯指标")
print(out.to_string(index=False, float_format=lambda x: f"{x:8.4f}"))
print(f"\n分解恒等式最大残差 = {out['decomp_resid'].max():.3e}  （0 表示 IC = sum c_d + w 精确成立）")
print(f"空头半边占比 share: 均值 {out['share'].mean():.3f}  中位 {out['share'].median():.3f}")
print(f"corr(|IC|, share) = {np.corrcoef(out['ic'].abs(), out['share'])[0, 1]:+.3f}")
out.to_csv("/home/chenzongwei/rust_pyfunc/tmp_mono_metric/lead/verify_real.csv", index=False)
