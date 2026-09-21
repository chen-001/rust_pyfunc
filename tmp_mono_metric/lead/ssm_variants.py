"""SSM 端点加重的三个改法，在真实 310 因子 + 构造形状上对比。

原版 V0 = min(Λ全, Λ空(1-5), Λ多(6-10))，端点台阶只占所在段 1/4 权重。
V1 = V0 再加两段更紧的首尾段（组1-4 / 组7-10，3 个台阶，端点权重升到 1/3）
V2 = V0 再加两段更紧的首尾段（组1-3 / 组8-10，2 个台阶，端点权重升到 1/2）
V3 = V0 再乘一个端点达标系数（端点不是极值时按缺口比例打折，不取 min）
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(2024)
pd.set_option("display.width", 240)


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


def seg(r, i, j):
    """组 i..j（1 起）这一段：净位移 / 该段台阶幅度之和。"""
    s = r[i - 1:j]
    d = np.diff(s)
    tot = np.abs(d).sum()
    return (s[-1] - s[0]) / tot if tot > 0 else 0.0


def end_margin(r):
    """两端是不是极值，以及领先多少个平均台阶。"""
    s = np.abs(np.diff(r)).sum() / 9.0
    d_short = r[1:].min() - r[0]      # 第1组比其余9组的最低值低多少
    d_long = r[-1] - r[:-1].max()     # 第10组比其余9组的最高值高多少
    return d_short, d_long, s


def variants(r):
    r = np.asarray(r, float)
    v0 = min(seg(r, 1, 10), seg(r, 1, 5), seg(r, 6, 10))
    v1 = min(v0, seg(r, 1, 4), seg(r, 7, 10))
    v2 = min(v1, seg(r, 1, 3), seg(r, 8, 10))
    d1, d10, s = end_margin(r)
    # 端点达标系数：两端都是极值时 =1，否则按缺口占多少个平均台阶打折
    gap = min(d1, d10)
    factor = 1.0 if gap >= 0 else max(0.0, 1.0 + gap / (3.0 * s))
    v3 = v0 * factor
    return dict(v0=v0, v1=v1, v2=v2, v3=v3, d_short=d1, d_long=d10, gap_ratio=gap / s)


# ------------------------------------------------------------------ 构造形状
print("=" * 108)
print("[1] 构造形状：看三种改法对'端点不是极值'的反应")
shapes = {
    "完美阶梯":              [0, 8, 16, 24, 32, 40, 48, 56, 64, 72],
    "第1组只差一点点(高2bp)":  [0, -2, 5, 15, 25, 35, 45, 55, 65, 75],
    "第10组只差一点点(低2bp)": [0, 8, 16, 24, 32, 40, 48, 56, 66, 64],
    "第1组明显不是最低":      [0, -20, 5, 15, 25, 35, 45, 55, 65, 75],
    "第10组明显不是最高":     [0, 8, 16, 24, 32, 40, 48, 56, 75, 50],
    "第10组掉到第3名":       [-45, -35, -25, -15, -5, 1, 3, 5, 7, 4],
    "中间破位(端点都好)":     [-45, -35, -25, -15, -5, -9, 1, 3, 5, 7],
}
print(f"  {'形状':<24}{'V0':>8}{'V1':>8}{'V2':>8}{'V3':>8}{'端点缺口/平均台阶':>18}")
for k, v in shapes.items():
    m = variants(v)
    print(f"  {k:<24}{m['v0']:8.3f}{m['v1']:8.3f}{m['v2']:8.3f}{m['v3']:8.3f}{m['gap_ratio']:18.2f}")

# ------------------------------------------------------------------ 真实因子
print("=" * 108)
d = pd.read_csv("/home/chenzongwei/rust_pyfunc/tmp_mono_metric/empirics/metrics_aug.csv")
R = d[[f"r{i}" for i in range(1, 11)]].to_numpy()
ic = d["ic"].to_numpy(); neg = ic < 0
Ro = np.where(neg[:, None], R[:, ::-1], R)
res = pd.DataFrame([variants(row) for row in Ro])
for k in ("v0", "v1", "v2", "v3", "gap_ratio"):
    d[k] = res[k].to_numpy()
d["absic"] = np.abs(ic)

print("[2] 310 个真实因子上的分布")
print(f"  {'指标':<6}{'5%':>9}{'25%':>9}{'50%':>9}{'75%':>9}{'95%':>9}{'=1.000 的个数':>16}")
for k in ("v0", "v1", "v2", "v3"):
    v = d[k].to_numpy()
    q = np.percentile(v, [5, 25, 50, 75, 95])
    print(f"  {k:<6}{q[0]:9.3f}{q[1]:9.3f}{q[2]:9.3f}{q[3]:9.3f}{q[4]:9.3f}{(np.abs(v - 1) < 1e-9).sum():16d}")
print()
print("[3] 改法把原来满分的因子打下来多少")
base = d.v0.to_numpy()
for k in ("v1", "v2"):
    v = d[k].to_numpy()
    m = np.abs(base - 1) < 1e-9
    print(f"  {k}: 原来 V0=1.000 的 {m.sum()} 个里，"
          f"<0.99 的 {(v[m] < 0.99).sum()} 个，<0.8 的 {(v[m] < 0.8).sum()} 个，"
          f"<0.5 的 {(v[m] < 0.5).sum()} 个，中位 {np.median(v[m]):.3f}")
print()
print("[4] 端点在真实因子里到底有多经常不是极值")
print(f"  第1组不是最低的因子: {(d.gap_ratio <= 0).sum()}/310 "
      f"（其中缺口不到 0.5 个平均台阶的 {((d.gap_ratio < 0) & (d.gap_ratio > -0.5)).sum()} 个）")
print(f"  端点缺口/平均台阶 分位 [5,25,50,75,95] = {np.round(np.percentile(d.gap_ratio, [5,25,50,75,95]), 3)}")
print()
print("[5] |IC| 前 50 在三种改法下的分数")
top = d.nlargest(50, "absic")
print(f"  V0 中位 {top.v0.median():.3f}   V1 中位 {top.v1.median():.3f}   "
      f"V2 中位 {top.v2.median():.3f}   V3 中位 {top.v3.median():.3f}")
print(f"  V0<0.5 的 {(top.v0 < 0.5).sum()} 个，V1<0.5 的 {(top.v1 < 0.5).sum()} 个，"
      f"V2<0.5 的 {(top.v2 < 0.5).sum()} 个，V3<0.5 的 {(top.v3 < 0.5).sum()} 个")
print()
print("[6] 分档：SSM 与 |IC| 的关系（用 V1）")
for lo, hi, lab in [(0.99, 1.01, "V1 ≥ 0.99"), (0.8, 0.99, "0.8 ≤ V1 < 0.99"),
                    (0.5, 0.8, "0.5 ≤ V1 < 0.8"), (-1.01, 0.5, "V1 < 0.5")]:
    m = (d.v1 >= lo) & (d.v1 < hi)
    if m.sum() == 0:
        continue
    print(f"  {lab:<20} n={m.sum():>3}   |IC| 中位 {d.absic[m].median():.4f}   "
          f"|IC|>0.05 占 {(d.absic[m] > 0.05).mean() * 100:3.0f}%")

# ------------------------------------------------------------------ 零分布
print("=" * 108)
print("[7] 纯噪声零分布（20 万次），三种改法各自的及格线")
m = rng.normal(size=(200000, 10))
rows = pd.DataFrame([variants(row) for row in m[:60000]])
print(f"  {'指标':<6}{'5%':>9}{'50%':>9}{'95%':>9}{'99%':>9}")
for j, k in enumerate(("v0", "v1", "v2", "v3")):
    v = rows.iloc[:, j].to_numpy()
    q = np.percentile(v, [5, 50, 95, 99])
    print(f"  {k:<6}{q[0]:9.3f}{q[1]:9.3f}{q[2]:9.3f}{q[3]:9.3f}")

# ------------------------------------------------------------------ 局部破位敏感度
print("=" * 108)
print("[8] 敏感度：把第 10 组从第 1 名往下压，IC 与各改法的反应")
base_curve = np.linspace(-1, 1, 10)


def sim_ic(gm, n_per=500, sigma=1.0, reps=150):
    f = np.concatenate([np.full(n_per, i) + rng.normal(0, .15, n_per) for i in range(10)])
    return float(np.mean([spearman(f, np.concatenate(
        [rng.normal(gm[i], sigma, n_per) for i in range(10)])) for _ in range(reps)]))


print(f"  {'压下的幅度':>10}{'IC':>10}{'V0':>9}{'V1':>9}{'V2':>9}{'V3':>9}")
ic0 = sim_ic(base_curve)
for frac in [0, 0.1, 0.2, 0.4, 0.6, 1.0]:
    gm = base_curve.copy()
    gm[9] = gm[8] - frac * 0.222
    m = variants(gm)
    print(f"  {frac:>10.1f}{sim_ic(gm):10.4f}{m['v0']:9.3f}{m['v1']:9.3f}{m['v2']:9.3f}{m['v3']:9.3f}")
print(f"  （IC 基准 {ic0:.4f}）")
d.to_csv("/home/chenzongwei/rust_pyfunc/tmp_mono_metric/lead/ssm_variants.csv", index=False)
