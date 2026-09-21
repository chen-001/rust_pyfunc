"""单因子：10 分组组均收益、Rank IC、单调性指标、IC 分组贡献分解、组内离散度。

约定
----
- 分组：按当日有效样本的因子平均秩 rf=1..n，组号 g = min(floor((rf-1)*10/n), 9)，g=0 是因子最小的一组。
- qf=(rf-0.5)/n 因子秩百分位，qy=(ry-0.5)/n 收益秩百分位。
- IC = Pearson(rf, ry) = Spearman(因子, 收益)。
- 分解：IC = 12*sum_d P(d)*(E[qf|d]-0.5)*(E[qy|d]-0.5) + 组内残差，c_d = 12*P(d)*(E[qf|d]-0.5)*(E[qy|d]-0.5)。
  成立前提：qf、qy 近似均匀分布在 (0,1)，标准差均为 1/sqrt(12)。
"""
import numpy as np
import pyarrow.parquet as pq
from scipy.stats import rankdata

NC = 10


def _pearson(a, b):
    a = a - a.mean()
    b = b - b.mean()
    return float(a @ b / np.sqrt((a @ a) * (b @ b)))


def load_factor(path, codes):
    """读 parquet，返回 (fdates int64, raw float32 (D, nsel), pos int64 (nsel,))。
    raw 的列 = 该文件与 codes 交集，按 codes 顺序；pos 是这些列在 codes 中的下标。"""
    names = pq.ParquetFile(path).schema_arrow.names
    naked = {n[:6]: n for n in names if n != "date"}
    sel = [naked[c] for c in codes if c in naked]
    tab = pq.read_table(path, columns=["date"] + sel)
    dv = tab.column("date").to_pandas().astype(str).values
    fdates = np.array([int(s[0:4] + s[5:7] + s[8:10]) for s in dv], dtype=np.int64)
    vals = tab.drop(["date"])
    raw = np.empty((vals.num_rows, vals.num_columns), dtype=np.float32)
    for j in range(vals.num_columns):
        raw[:, j] = vals.column(j).to_numpy(zero_copy_only=False)
    colidx = {c: i for i, c in enumerate(codes)}
    pos = np.array([colidx[c] for c in codes if c in naked], dtype=np.int64)
    return fdates, raw, pos


def daily_stats(fv, yv):
    """单日统计。返回 dict；样本不足或出现空组返回 None。"""
    n = fv.size
    if n < 200:
        return None
    rf = rankdata(fv)
    ry = rankdata(yv)
    g = np.minimum(((rf - 1) * NC / n).astype(np.int64), NC - 1)
    cnt = np.bincount(g, minlength=NC).astype(np.float64)
    if (cnt == 0).any():
        return None
    ic = _pearson(rf, ry)
    ic_mirror = _pearson(rankdata(-fv), ry)          # 独立重算：对因子取负后再排秩
    qf = (rf - 0.5) / n
    qy = (ry - 0.5) / n
    w = cnt / n
    gm = np.bincount(g, weights=yv, minlength=NC) / cnt
    ym = np.bincount(g, weights=qy, minlength=NC) / cnt
    qm = np.bincount(g, weights=qf, minlength=NC) / cnt
    c = 12.0 * w * (qm - 0.5) * (ym - 0.5)
    gsd = np.sqrt(np.maximum(
        np.bincount(g, weights=yv * yv, minlength=NC) / cnt - gm * gm, 0.0))
    return dict(ic=ic, ic_mirror=ic_mirror, r=gm, c=c, gsd=gsd,
                resid=ic - float(c.sum()))


def factor_metrics(path, codes, r5, r5dates):
    """算一个因子的全部指标。返回 dict 或 None。"""
    fdates, raw, pos = load_factor(path, codes)
    order = np.argsort(fdates)
    fs = fdates[order]
    ridx = np.flatnonzero(np.isin(r5dates, fs))
    if ridx.size < 200:
        return None
    rows = order[np.searchsorted(fs, r5dates[ridx])]
    F = raw[rows]
    R = r5[ridx][:, pos]

    acc_r = np.zeros(NC); acc_c = np.zeros(NC); acc_sd = np.zeros(NC)
    s_ic = 0.0; s_mir = 0.0; s_res = 0.0; nd = 0; nbad = 0; nmir_bad = 0
    max_mirror_dev = 0.0
    for i in range(F.shape[0]):
        f = F[i]; y = R[i]
        m = np.isfinite(f) & np.isfinite(y)
        if m.sum() < 200:
            continue
        st = daily_stats(f[m], y[m])
        if st is None:
            nbad += 1
            continue
        acc_r += st["r"]; acc_c += st["c"]; acc_sd += st["gsd"]
        s_ic += st["ic"]; s_mir += st["ic_mirror"]; s_res += st["resid"]
        dev = abs(st["ic"] + st["ic_mirror"])
        max_mirror_dev = max(max_mirror_dev, dev)
        nmir_bad += int(dev > 1e-12)
        nd += 1
    if nd < 200:
        return None
    r = acc_r / nd
    c = acc_c / nd
    gsd = acc_sd / nd
    ic = s_ic / nd
    d = np.diff(r)
    den = np.abs(d).sum()
    lam = (r[9] - r[0]) / den if den > 0 else np.nan
    d1, d2 = np.abs(d[:4]).sum(), np.abs(d[5:]).sum()
    l_short = (r[4] - r[0]) / d1 if d1 > 0 else np.nan
    l_long = (r[9] - r[5]) / d2 if d2 > 0 else np.nan
    ssm = min(l_short, l_long) if np.isfinite(l_short) and np.isfinite(l_long) else np.nan
    gamma = _pearson(rankdata(np.arange(1, 11, dtype=float)), rankdata(r))
    tot = c.sum()
    share = float(c[:5].sum() / tot) if tot != 0 else np.nan
    # 方向归正：把因子符号翻到 IC>0。此时组号反转（-f 的第 d 组 = f 的第 11-d 组）：
    # 组均收益 r'_d = r_{11-d}（收益本身不变号），分解贡献 c'_d = -c_{11-d}（因子秩百分位变号）。
    if ic < 0:
        c_or = -c[::-1]; r_or = r[::-1]
    else:
        c_or = c; r_or = r
    share_ic = float(c_or[:5].sum() / c_or.sum()) if c_or.sum() != 0 else np.nan
    return dict(nd=nd, nbad=nbad, nmir_bad=nmir_bad, ic=ic, ic_mirror_mean=s_mir / nd,
                resid=s_res / nd, max_mirror_dev=max_mirror_dev,
                r=r, r_or=r_or, c=c, c_or=c_or, gsd=gsd,
                lam=lam, l_short=l_short, l_long=l_long, ssm=ssm, gamma=gamma,
                share=share, share_ic=share_ic)
