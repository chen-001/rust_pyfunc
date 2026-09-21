"""补充检验（修正版）：重叠分段、多空对称性、阈值表、排序分歧。

定义（固定）：
  台阶 Δ_d = r_{d+1} - r_d, d=1..9
  Lambda_all   = (r10 - r1) / sum|Δ_1..Δ_9|          整体直线度
  Lambda_short = (r6  - r1) / sum|Δ_1..Δ_5|          空头段（组1..6，5个台阶）
  Lambda_long  = (r10 - r5) / sum|Δ_5..Δ_9|          多头段（组5..10，5个台阶）
  SSM          = min(Lambda_short, Lambda_long)      双边单调分
  Gamma        = Spearman(1..10, r_1..r_10)          分组秩相关
"""
import numpy as np

rng = np.random.default_rng(7)


def avg_rank(x):
    order = np.argsort(x, kind="stable")
    r = np.empty(len(x), float)
    r[order] = np.arange(1, len(x) + 1, dtype=float)
    _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    if (cnt > 1).any():
        s = np.zeros(len(cnt)); np.add.at(s, inv, r); r = (s / cnt)[inv]
    return r


def spearman(x, y):
    a, b = avg_rank(x), avg_rank(y)
    a = a - a.mean(); b = b - b.mean()
    return float(a @ b / np.sqrt((a @ a) * (b @ b)))


def ladder(r):
    r = np.asarray(r, float); st = np.diff(r); tot = np.abs(st).sum()
    lam = float((r[-1] - r[0]) / tot) if tot else 0.0
    ts, tl = np.abs(st[:5]).sum(), np.abs(st[4:]).sum()
    ls = float((r[5] - r[0]) / ts) if ts else 0.0
    ll = float((r[9] - r[4]) / tl) if tl else 0.0
    return dict(lam=lam, ls=ls, ll=ll, ssm=min(ls, ll), mid=float(st[4]),
                gamma=spearman(np.arange(1, 11), r))


def sim_ic(gm, n_per=600, sigma=1.0, reps=300):
    ng = len(gm)
    f = np.concatenate([np.full(n_per, d) + rng.normal(0, .15, n_per) for d in range(ng)])
    return float(np.mean([spearman(f, np.concatenate(
        [rng.normal(gm[d], sigma, n_per) for d in range(ng)])) for _ in range(reps)]))


print("=" * 84)
print("[6] 四种原型：IC 与阶梯形状指标的分离")
arch = {
    "A 均衡单调(线性)":            [-0.9, -0.7, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7, 0.9],
    "B 空头极强(空头半边x5)":      [-4.5, -3.5, -2.5, -1.5, -0.5, 0.1, 0.3, 0.5, 0.7, 0.9],
    "C 空头极强+第10组掉到第3名":   [-4.5, -3.5, -2.5, -1.5, -0.5, 0.1, 0.3, 0.5, 0.7, 0.4],
    "D 多头极强(多头半边x5)":      [-0.9, -0.7, -0.5, -0.3, -0.1, 0.5, 1.5, 2.5, 3.5, 4.5],
    "E 正中断裂(第6组低于第5组)":   [-4.5, -3.5, -2.5, -1.5, -0.5, -0.9, 0.1, 0.3, 0.5, 0.7],
    "F 空头乱(第1组不是最低)":      [-4.0, -4.5, -3.0, -1.5, -0.5, 0.1, 0.3, 0.5, 0.7, 0.9],
}
print(f"  {'原型':<28}{'IC':>9}{'Lambda':>9}{'空头':>8}{'多头':>8}{'SSM':>8}{'Gamma':>8}{'中段Δ5':>9}")
for k, v in arch.items():
    m = ladder(v)
    print(f"  {k:<28}{sim_ic(v):9.4f}{m['lam']:9.3f}{m['ls']:8.3f}{m['ll']:8.3f}"
          f"{m['ssm']:8.3f}{m['gamma']:8.3f}{m['mid']:9.2f}")

print("=" * 84)
print("[7] 同等强度信号加在空头 vs 多头，IC 是否系统性不同（400 次独立配对抽样）")
ns = 4000
ic_s, ic_l = [], []
for _ in range(400):
    f = rng.normal(size=ns)
    g = np.clip((avg_rank(f) * 10 // (ns + 1)).astype(int), 0, 9)
    base = rng.normal(size=ns) * 0.9
    rs = base.copy(); rs[g <= 2] -= 0.9
    rl = base.copy(); rl[g >= 7] += 0.9
    ic_s.append(spearman(f, rs)); ic_l.append(spearman(f, rl))
ic_s, ic_l = np.array(ic_s), np.array(ic_l)
d = ic_s - ic_l
print(f"  空头侧信号 IC = {ic_s.mean():.4f} (sd {ic_s.std():.4f})")
print(f"  多头侧信号 IC = {ic_l.mean():.4f} (sd {ic_l.std():.4f})")
print(f"  配对差值 = {d.mean():+.6f}   t = {d.mean() / (d.std() / np.sqrt(len(d))):+.2f}")
print("  → 差值统计上等于 0：IC 对两侧完全一视同仁")

print("=" * 84)
print("[8] 纯噪声零分布 → 实用阈值（20 万次）")
m = rng.normal(size=(200000, 10))
st = np.diff(m, axis=1)
vals = {
    "Lambda整体": (m[:, 9] - m[:, 0]) / np.abs(st).sum(1),
    "空头半边": (m[:, 5] - m[:, 0]) / np.abs(st[:, :5]).sum(1),
    "多头半边": (m[:, 9] - m[:, 4]) / np.abs(st[:, 4:]).sum(1),
}
vals["SSM=min"] = np.minimum(vals["空头半边"], vals["多头半边"])
vals["Gamma"] = np.array([spearman(np.arange(1, 11), row) for row in m[:40000]])
print(f"  {'指标':<14}{'5%':>9}{'50%':>9}{'95%':>9}{'99%':>9}{'P(>0.5)':>10}")
for nm, v in vals.items():
    q = np.percentile(v, [5, 50, 95, 99])
    print(f"  {nm:<14}{q[0]:9.3f}{q[1]:9.3f}{q[2]:9.3f}{q[3]:9.3f}{(v > 0.5).mean() * 100:9.2f}%")
print("  → SSM 的零分布中心在 -0.17 附近（min 运算天然偏低），及格线不能取 0")

print("=" * 84)
print("[9] 同一批因子里 IC 排序 vs 阶梯指标排序的分歧（3000 个合成因子）")
pool = []
for _ in range(3000):
    p = rng.uniform(0.5, 1.0)                 # 台阶方向正确的概率
    s = rng.uniform(0.05, 2.5)                # 台阶幅度
    steps = s * np.where(rng.random(9) < p, 1.0, -1.0)
    gm = np.concatenate([[0], np.cumsum(steps)]) + rng.normal(0, 0.01, 10)
    m = ladder(gm)
    pool.append((sim_ic(gm, n_per=120, reps=10), m["ssm"], m["gamma"], m["lam"]))
a = np.array(pool)
print(f"  corr(IC, Lambda_all) = {np.corrcoef(a[:, 0], a[:, 3])[0, 1]:+.3f}")
print(f"  corr(IC, SSM)        = {np.corrcoef(a[:, 0], a[:, 1])[0, 1]:+.3f}")
print(f"  corr(IC, Gamma)      = {np.corrcoef(a[:, 0], a[:, 2])[0, 1]:+.3f}")
for K in (20, 50, 100):
    ti = set(np.argsort(-a[:, 0])[:K]); ts = set(np.argsort(-a[:, 1])[:K])
    tg = set(np.argsort(-a[:, 2])[:K])
    print(f"  取前 {K:>3} 名: IC∩SSM {len(ti & ts) / K * 100:5.1f}%   "
          f"IC∩Gamma {len(ti & tg) / K * 100:5.1f}%   SSM∩Gamma {len(ts & tg) / K * 100:5.1f}%")
print("  → 高 IC 与高 SSM 是两批不同的因子，换指标会真的换出一批货")
