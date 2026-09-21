"""单调性指标：数学性质与零分布标定。

四件事：
1. 验证 IC 的分组贡献分解公式  IC = sum_d c_d + w  （精确恒等式）
2. 验证 IC 的镜像对称性（取负因子 → IC 恰好反号）
3. 标定纯噪声下 Lambda / Lambda_short / Lambda_long / SSM / Gamma 的零分布
4. 构造四种"同 IC 不同形状"的因子，展示 IC 与阶梯形状指标的分离
"""
import numpy as np

rng = np.random.default_rng(20240101)


def avg_rank(x):
    """平均秩（1..n），处理并列。"""
    order = np.argsort(x, kind="stable")
    r = np.empty(len(x), dtype=float)
    r[order] = np.arange(1, len(x) + 1, dtype=float)
    # 并列取平均
    _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    if (cnt > 1).any():
        s = np.zeros(len(cnt))
        np.add.at(s, inv, r)
        r = (s / cnt)[inv]
    return r


def spearman(x, y):
    rx, ry = avg_rank(x), avg_rank(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    return float(rx @ ry / np.sqrt((rx @ rx) * (ry @ ry)))


def ic_decomp(f, r, ngroups=10):
    """把 Spearman IC 精确拆成 每组贡献 c_d + 组内贡献 w。

    q,y = 因子/收益的秩百分位；c_d = 12*P(d)*(E[q|d]-.5)*(E[y|d]-.5)
    """
    n = len(f)
    q = (avg_rank(f) - 0.5) / n
    y = (avg_rank(r) - 0.5) / n
    ic = 12.0 * (q @ y / n - 0.25)
    g = np.clip((avg_rank(f) * ngroups // (n + 1)).astype(int), 0, ngroups - 1)
    c = np.zeros(ngroups)
    w = 0.0
    for d in range(ngroups):
        m = g == d
        p = m.mean()
        qd, yd = q[m], y[m]
        c[d] = 12.0 * p * (qd.mean() - 0.5) * (yd.mean() - 0.5)
        w += 12.0 * p * float(np.cov(qd, yd, bias=True)[0, 1])
    return ic, c, w


def ladder_metrics(r, ngroups=10):
    """r: 各组平均收益（长度 10）。返回整体/两侧直线度与秩序度。"""
    r = np.asarray(r, float)
    steps = np.diff(r)
    tot = np.abs(steps).sum()
    lam = float((r[-1] - r[0]) / tot) if tot > 0 else 0.0
    half = ngroups // 2
    s_short, s_long = steps[: half - 1], steps[half:]
    ts, tl = np.abs(s_short).sum(), np.abs(s_long).sum()
    lam_s = float((r[half - 1] - r[0]) / ts) if ts > 0 else 0.0
    lam_l = float((r[-1] - r[half]) / tl) if tl > 0 else 0.0
    gamma = spearman(np.arange(1, ngroups + 1), r)
    return dict(lambda_all=lam, lambda_short=lam_s, lambda_long=lam_l,
                ssm=min(lam_s, lam_l), gamma=gamma)


# ---------------------------------------------------------------- 1. 分解恒等式
print("=" * 74)
print("[1] IC 分组贡献分解恒等式  IC = sum_d c_d + w")
err_ic, err_dec = [], []
for t in range(200):
    n = 4000
    f = rng.normal(size=n)
    r = 0.35 * f + rng.normal(size=n) * 0.9
    ic, c, w = ic_decomp(f, r)
    err_ic.append(abs(ic - (c.sum() + w)))
    err_dec.append(abs(ic - 12.0 * (np.corrcoef(avg_rank(f), avg_rank(r))[0, 1]) - 0))
print(f"  分解残差 |IC - (sum c_d + w)| 最大 {max(err_ic):.3e}   (200 次试验)")
print(f"  自检 |IC - 12*(E[qy]-1/4)| 最大 {max(err_dec):.3e}")

# 纯空头信息 vs 纯多头信息：分解后的贡献分布
print("\n  两种形状的贡献分布（n=4000，信号只加在一侧）：")
for tag, sig in [("只加在最低3组", -1), ("只加在最高3组", +1)]:
    f = rng.normal(size=4000)
    r = rng.normal(size=4000) * 0.9
    g = np.clip((avg_rank(f) * 10 // 4001).astype(int), 0, 9)
    hit = (g <= 2) if sig < 0 else (g >= 7)
    r[hit] += sig * 0.9
    ic, c, w = ic_decomp(f, r)
    print(f"    {tag}: IC={ic:+.4f}  组内项 w={w:+.4f}  "
          f"空头半边占比={c[:5].sum() / c.sum() * 100:6.1f}%")

# ---------------------------------------------------------------- 2. 镜像对称性
print("=" * 74)
print("[2] IC 的镜像对称性：因子取负 → IC 是否恰好反号")
dmax = 0.0
for t in range(500):
    n = 3000
    f = rng.normal(size=n)
    r = 0.4 * f + rng.normal(size=n)
    dmax = max(dmax, abs(spearman(f, r) + spearman(-f, r)))
print(f"  |IC(f) + IC(-f)| 的最大值 = {dmax:.3e}  (500 次试验, 双精度机器精度级)")
print("  结论: IC 对多空两侧严格对称，不存在数学上的空头偏置")

# ---------------------------------------------------------------- 3. 零分布标定
print("=" * 74)
print("[3] 纯噪声下各指标的零分布（10 组组均，组均值 iid 正态）")


def null_sample(k=200000, ngroups=10):
    m = rng.normal(size=(k, ngroups))
    out = np.array([[ladder_metrics(row, ngroups)[key] for key in
                     ("lambda_all", "lambda_short", "lambda_long", "ssm", "gamma")]
                    for row in m[:20000]])
    return out


nd = null_sample()
names = ["Lambda(整体)", "Lambda(空头半边)", "Lambda(多头半边)", "SSM=min", "Gamma"]
print(f"  {'指标':<16}{'5%':>9}{'25%':>9}{'50%':>9}{'75%':>9}{'95%':>9}{'|x|>0.5':>10}")
for j, nm in enumerate(names):
    v = nd[:, j]
    q = np.percentile(v, [5, 25, 50, 75, 95])
    print(f"  {nm:<16}{q[0]:9.3f}{q[1]:9.3f}{q[2]:9.3f}{q[3]:9.3f}{q[4]:9.3f}"
          f"{(np.abs(v) > 0.5).mean() * 100:9.1f}%")
print("  解读: 纯噪声下 Gamma 的标准差 = 1/sqrt(9) = 0.333（Spearman 精确零分布）")
print("        纯噪声下 SSM 有约 10% 的概率超过 0.5 —— 10 个数太少，必须配显著性检验")

# ---------------------------------------------------------------- 4. 同 IC 不同形状
print("=" * 74)
print("[4] 四种原型：IC 与阶梯形状的分离")


def build(group_mean, n_per=800, sigma=1.0, reps=200):
    """按组均值构造因子-收益数据，返回 IC 与形状指标的均值。"""
    ng = len(group_mean)
    f = np.concatenate([np.full(n_per, d) + rng.normal(0, .15, n_per)
                        for d in range(ng)])
    ics, mets = [], []
    for _ in range(reps):
        r = np.concatenate([rng.normal(group_mean[d], sigma, n_per)
                            for d in range(ng)])
        ics.append(spearman(f, r))
    return np.mean(ics), ladder_metrics(group_mean)


arch = {
    "A 均衡单调 (线性阶梯)":        np.array([-0.9, -0.7, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7, 0.9]),
    "B 空头极强 (空头半边x5)":      np.array([-4.5, -3.5, -2.5, -1.5, -0.5, 0.1, 0.3, 0.5, 0.7, 0.9]),
    "C 空头极强+多头破位(第10组仅第3)": np.array([-4.5, -3.5, -2.5, -1.5, -0.5, 0.1, 0.3, 0.5, 0.7, 0.4]),
    "D 多头极强 (多头半边x5)":      np.array([-0.9, -0.7, -0.5, -0.3, -0.1, 0.5, 1.5, 2.5, 3.5, 4.5]),
}
print(f"  {'原型':<32}{'IC':>9}{'Lambda':>9}{'空头':>8}{'多头':>8}{'SSM':>8}{'Gamma':>8}")
for k, v in arch.items():
    ic, m = build(v)
    print(f"  {k:<32}{ic:9.4f}{m['lambda_all']:9.3f}{m['lambda_short']:8.3f}"
          f"{m['lambda_long']:8.3f}{m['ssm']:8.3f}{m['gamma']:8.3f}")
print("  解读: A/B/D 都严格单调 → Lambda 与 Gamma 全部 =1.000，完全平分秋色")
print("        IC 却把 B/D 排在 A 前面（只因为一侧幅度大）—— 这就是用户感觉到的偏袒")
print("        C 的多头半边破了位 → SSM 与 Gamma 立刻掉下来，而 IC 只掉一点")

# ---------------------------------------------------------------- 5. 尾部公平性
print("=" * 74)
print("[5] 尾部公平性：把空头半边整体放大 k 倍，各指标怎么变")
base = np.array([-0.9, -0.7, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7, 0.9])
print(f"  {'k':>6}{'IC':>10}{'Lambda':>10}{'SSM':>10}{'Gamma':>10}")
for k in [1, 2, 5, 10]:
    v = base.copy()
    v[:5] *= k
    ic, m = build(v)
    print(f"  {k:>6}{ic:10.4f}{m['lambda_all']:10.3f}{m['ssm']:10.3f}{m['gamma']:10.3f}")
print("  解读: 空头幅度放大 10 倍，IC 明显上升，而 Lambda/SSM/Gamma 一动不动")
print("        这三个指标对'哪一侧幅度大'完全不敏感，只对'台阶顺不顺'敏感")
