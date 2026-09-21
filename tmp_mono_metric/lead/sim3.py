"""补充检验 3：IC 对"局部破位"的敏感度 vs 阶梯指标的敏感度。"""
import numpy as np

rng = np.random.default_rng(11)


def avg_rank(x):
    o = np.argsort(x, kind="stable"); r = np.empty(len(x), float)
    r[o] = np.arange(1, len(x) + 1, dtype=float)
    _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    if (cnt > 1).any():
        s = np.zeros(len(cnt)); np.add.at(s, inv, r); r = (s / cnt)[inv]
    return r


def spearman(x, y):
    a, b = avg_rank(x), avg_rank(y); a -= a.mean(); b -= b.mean()
    return float(a @ b / np.sqrt((a @ a) * (b @ b)))


def ladder(r):
    r = np.asarray(r, float); st = np.diff(r); tot = np.abs(st).sum()
    ts, tl = np.abs(st[:5]).sum(), np.abs(st[4:]).sum()
    ls = (r[5] - r[0]) / ts if ts else 0.0
    ll = (r[9] - r[4]) / tl if tl else 0.0
    return (r[9] - r[0]) / tot if tot else 0.0, min(ls, ll), spearman(np.arange(1, 11), r)


def sim_ic(gm, n_per=500, sigma=1.0, reps=200):
    f = np.concatenate([np.full(n_per, d) + rng.normal(0, .15, n_per) for d in range(10)])
    return float(np.mean([spearman(f, np.concatenate(
        [rng.normal(gm[d], sigma, n_per) for d in range(10)])) for _ in range(reps)]))


base = np.linspace(-1, 1, 10)
print("=" * 88)
print("[10] 把一个完美阶梯的第 10 组往下压 —— IC 掉得慢，阶梯指标掉得快")
print(f"  {'压下的幅度':>10}{'IC':>10}{'IC相对下降':>12}{'Lambda':>10}{'SSM':>10}{'Gamma':>10}")
ic0 = sim_ic(base)
for frac in [0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]:
    gm = base.copy()
    gm[9] = gm[8] - frac * 0.222          # 0.222 = 原步长
    lam, ssm, ga = ladder(gm)
    ic = sim_ic(gm)
    print(f"  {frac:>10.1f}{ic:10.4f}{(ic - ic0) / ic0 * 100:11.1f}%{lam:10.3f}{ssm:10.3f}{ga:10.3f}")

print("=" * 88)
print("[11] 对称性：把第 1 组往上抬（空头破位）—— 下降幅度应当一致")
print(f"  {'抬高的幅度':>10}{'IC':>10}{'IC相对下降':>12}{'Lambda':>10}{'SSM':>10}{'Gamma':>10}")
for frac in [0, 0.2, 0.4, 0.6, 0.8, 1.0]:
    gm = base.copy()
    gm[0] = gm[1] + frac * 0.222
    lam, ssm, ga = ladder(gm)
    ic = sim_ic(gm)
    print(f"  {frac:>10.1f}{ic:10.4f}{(ic - ic0) / ic0 * 100:11.1f}%{lam:10.3f}{ssm:10.3f}{ga:10.3f}")

print("=" * 88)
print("[12] 高 IC 因子里有多少其实阶梯不整齐（3000 个合成因子）")
pool = []
for _ in range(3000):
    p = rng.uniform(0.45, 1.0)
    s = rng.uniform(0.05, 2.0) * rng.choice([0.3, 1.0, 3.0])
    st = s * np.where(rng.random(9) < p, 1.0, -1.0)
    gm = np.concatenate([[0], np.cumsum(st)])
    gm = gm - gm.mean()
    lam, ssm, ga = ladder(gm)
    pool.append((sim_ic(gm, n_per=100, reps=8), ssm, ga, lam))
a = np.array(pool)
SSM95, G95 = 0.186, 0.552
top = np.argsort(-a[:, 0])[:50]
print(f"  corr(IC, SSM) = {np.corrcoef(a[:, 0], a[:, 1])[0, 1]:+.3f}   "
      f"corr(IC, Gamma) = {np.corrcoef(a[:, 0], a[:, 2])[0, 1]:+.3f}")
print(f"  IC 前 50 名里，SSM 低于噪声 95% 分位({SSM95}) 的有 "
      f"{(a[top, 1] < SSM95).sum()}/50 = {(a[top, 1] < SSM95).mean() * 100:.0f}%")
print(f"  IC 前 50 名里，Gamma 低于噪声 95% 分位({G95}) 的有 "
      f"{(a[top, 2] < G95).sum()}/50 = {(a[top, 2] < G95).mean() * 100:.0f}%")
print(f"  IC 前 50 名与 SSM 前 50 名的重合数 = "
      f"{len(set(top) & set(np.argsort(-a[:, 1])[:50]))}/50")
print(f"  IC 前 50 名与 Gamma 前 50 名的重合数 = "
      f"{len(set(top) & set(np.argsort(-a[:, 2])[:50]))}/50")
