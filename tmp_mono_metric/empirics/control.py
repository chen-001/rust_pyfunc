"""对照实验：用真实 r5、但因子换成已知结构的合成因子，看 share 会不会自己跑到 0.5 以外。

- perfect: f = y（与收益同序）→ 理论 share 恰为 0.5，用来验证代码。
- sym_sX : f = u + s*z，u=(rank(y)-0.5)/n，z~N(0,1)。构造关于 (u,z)->(1-u,-z) 对称，
           所以理论 share = 0.5，用来判断真实因子的 0.5565 是不是"IC 大小的机械产物"。
- het_lo / het_hi: 噪声方差随 u 变化（异方差），看这种不对称能把 share 推到哪边。
"""
import os, sys, numpy as np
from scipy.stats import rankdata

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from metrics_core import daily_stats

NC = 10
z = np.load(os.path.join(HERE, "returns_cache.npz"))
r5 = z["r5"]


def run(fgen, seed, reps):
    out = []
    for rep in range(reps):
        rng = np.random.default_rng(seed + rep)
        acc_r = np.zeros(NC); acc_c = np.zeros(NC); s_ic = 0.0; s_res = 0.0; nd = 0
        for i in range(r5.shape[0]):
            y = r5[i]
            m = np.isfinite(y)
            if m.sum() < 200:
                continue
            yv = y[m]
            st = daily_stats(fgen(rng, yv), yv)
            if st is None:
                continue
            acc_r += st["r"]; acc_c += st["c"]; s_ic += st["ic"]; s_res += st["resid"]; nd += 1
        r = acc_r / nd; c = acc_c / nd; ic = s_ic / nd
        d = np.diff(r)
        lam = (r[9] - r[0]) / np.abs(d).sum()
        lsh = (r[4] - r[0]) / np.abs(d[:4]).sum()
        llo = (r[9] - r[5]) / np.abs(d[5:]).sum()
        out.append((ic, c[:5].sum() / c.sum(), lam, lsh, llo, min(lsh, llo),
                    c.sum() / ic, r * 1e4))
    return out


def show(tag, res):
    ic = np.mean([x[0] for x in res]); sh = np.mean([x[1] for x in res])
    lam = np.mean([x[2] for x in res]); lsh = np.mean([x[3] for x in res])
    llo = np.mean([x[4] for x in res]); lr = np.mean([x[6] for x in res])
    print(f"{tag:14s} reps={len(res):3d} IC={ic:+.4f} share={sh:.4f} lambda={lam:+.3f} "
          f"lsh={lsh:+.3f} llo={llo:+.3f} ladder_ratio={lr:.4f}")


def gen_perfect(rng, yv):
    return yv


def gen_sym(s):
    def g(rng, yv):
        n = yv.size
        u = (rankdata(yv) - 0.5) / n
        return u + s * rng.standard_normal(n)
    return g


def gen_het(s, k):
    def g(rng, yv):
        n = yv.size
        u = (rankdata(yv) - 0.5) / n
        return u + s * (1 + k * (0.5 - u)) * rng.standard_normal(n)
    return g


show("perfect", run(gen_perfect, 11, 1))
for s in (3.0, 4.8, 8.0):
    show(f"sym_s{s}", run(gen_sym(s), 100, 20))
for k in (-0.8, 0.8):
    show(f"het_s4.8_k{k}", run(gen_het(4.8, k), 200, 10))

# 真实因子（top50 按 |IC|）的同口径对照
import pandas as pd
df = pd.read_csv(os.path.join(HERE, "metrics.csv"))
df["absic"] = df.ic.abs()
top = df.nlargest(50, "absic")
C = top[[f"c{i}" for i in range(1, 11)]].to_numpy(float)
R = top[[f"r{i}" for i in range(1, 11)]].to_numpy(float)
neg = top.ic.to_numpy() < 0
Cor = np.where(neg[:, None], -C[:, ::-1], C)
Ror = np.where(neg[:, None], R[:, ::-1], R)
d = np.diff(Ror, axis=1)
lam = (Ror[:, 9] - Ror[:, 0]) / np.abs(d).sum(axis=1)
lsh = (Ror[:, 4] - Ror[:, 0]) / np.abs(d[:, :4]).sum(axis=1)
llo = (Ror[:, 9] - Ror[:, 5]) / np.abs(d[:, 5:]).sum(axis=1)
print(f"{'REAL top50':14s} n= 50  IC={top.absic.mean():.4f} "
      f"share={top.share_ic.mean():.4f} lambda={lam.mean():+.3f} lsh={lsh.mean():+.3f} "
      f"llo={llo.mean():+.3f} ladder_ratio={(Cor.sum(1)/top.absic).mean():.4f}")
