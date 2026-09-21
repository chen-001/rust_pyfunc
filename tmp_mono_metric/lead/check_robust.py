"""查清：真实因子的 10 组曲线为什么抖得这么厉害？

对比四种"组收益"口径：
  A 时间平均（现在的做法，用户看图看到的）
  B 时间中位数
  C 截尾均值（每日组收益截到 1%/99% 再平均）
  D 每日各自算 Lambda 再平均（先看形状，再跨日平均）
"""
import glob
import numpy as np
import pandas as pd

CACHE = "/home/chenzongwei/rust_pyfunc/tmp_mono_metric/empirics/returns_cache.npz"
z = np.load(CACHE, allow_pickle=True)
dates, codes, r5 = z["dates"], z["codes"], z["r5"]
dpos = {int(d): i for i, d in enumerate(dates)}
cpos = {c: i for i, c in enumerate(codes)}


def ar(x):
    o = np.argsort(x, kind="stable"); r = np.empty(len(x))
    r[o] = np.arange(1, len(x) + 1, dtype=float)
    v, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    if (cnt > 1).any():
        s = np.zeros(len(cnt)); np.add.at(s, inv, r); r = (s / cnt)[inv]
    return r


def lam_of(gm):
    st = np.diff(gm); tot = np.abs(st).sum()
    return (gm[-1] - gm[0]) / tot if tot > 0 else 0.0


def ssm_of(gm):
    st = np.diff(gm)
    ts, tl = np.abs(st[:5]).sum(), np.abs(st[4:]).sum()
    ls = (gm[5] - gm[0]) / ts if ts else 0.0
    ll = (gm[9] - gm[4]) / tl if tl else 0.0
    return min(ls, ll)


files = (sorted(glob.glob("/nas197/user_home_unsafe/chenzongwei/factor_data/hm100/*.parquet"))[:4] +
         sorted(glob.glob("/nas197/user_home_unsafe/chenzongwei/factor_data/hm101/*.parquet"))[:2])

for fp in files:
    df = pd.read_parquet(fp)
    df = df.rename(columns={"date": "_d"})
    df["_d"] = pd.to_datetime(df["_d"]).dt.strftime("%Y%m%d").astype(int)
    cols = [c for c in df.columns if c != "_d"]
    bare = np.array([c.split(".")[0] for c in cols])
    keep = np.array([c in cpos for c in bare])
    cols = np.array(cols)[keep]
    colidx = np.array([cpos[c.split(".")[0]] for c in cols])
    df = df[["_d"] + list(cols)].set_index("_d")

    per_day, ics = [], []
    for d, row in df.iterrows():
        i = dpos.get(int(d))
        if i is None:
            continue
        f = row.to_numpy(float); rr = r5[i, colidx]
        ok = ~np.isnan(f) & ~np.isnan(rr)
        if ok.sum() < 300:
            continue
        f, rr = f[ok], rr[ok]
        n = len(f)
        g = np.clip((ar(f) * 10 // (n + 1)).astype(int), 0, 9)
        per_day.append([rr[g == k].mean() for k in range(10)])
        q = (ar(f) - .5) / n; y = (ar(rr) - .5) / n
        ics.append(12.0 * (q @ y / n - .25))

    P = np.array(per_day)                      # 日期 x 10
    A = P.mean(0); B = np.median(P, 0)
    C = np.clip(P, np.percentile(P, 1, axis=0), np.percentile(P, 99, axis=0)).mean(0)
    lam_daily = np.array([lam_of(row) for row in P])
    ssm_daily = np.array([ssm_of(row) for row in P])

    print("=" * 108)
    print(fp.split("/")[-1].replace(".parquet", "")[:70])
    print(f"  IC(每日秩相关均值) = {np.mean(ics):+.4f}")
    print(f"  单日组收益的标准差(典型) = {P.std(0).mean() * 1e4:8.1f} bp"
          f"   单日极差(1%-99%) = {(np.percentile(P, 99, 0) - np.percentile(P, 1, 0)).mean() * 1e4:8.1f} bp")
    for nm, v in [("A 时间平均", A), ("B 时间中位数", B), ("C 截尾均值", C)]:
        print(f"  {nm:<12} 组收益(bp) " + " ".join(f"{x * 1e4:7.1f}" for x in v)
              + f"   |  Lambda={lam_of(v):+.3f}  SSM={ssm_of(v):+.3f}")
    print(f"  D 每日Lambda均值 = {lam_daily.mean():+.3f}  (中位 {np.median(lam_daily):+.3f}, "
          f"标准差 {lam_daily.std():.3f})  每日SSM均值 = {ssm_daily.mean():+.3f}")
