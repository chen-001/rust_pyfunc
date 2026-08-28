"""RankIC 中性化方案探索 — Python 原型（小样本快速验证数学猜想）。

从零复刻生产 neutralize_std 链路（factor_neutralize_std.rs 语义），
与生产 rp.neutralize_std_block_py 对照验证理解正确，再实现各候选方案。
"""
import numpy as np
import pandas as pd
import rust_pyfunc as rp
import sys

sys.path.insert(0, "/home/chenzongwei/design_whatever")
import design_whatever as dw

STORE = "/hdd/user_home_unsafe/chenzongwei/factor_store_挂单猫0701c"
BARRA = "/home/chenzongwei/database/barra/barra_daily_together_jason.parquet"
BP = "/home/chenzongwei/pythoncode/_tail_v2_shared/backtest_inputs/000905_20170103_20260522_5438_6b8884a67f05d221/"
T, N = 2276, 5438


def load_all(n_days):
    """加载模板轴/行业/restrict/收益/barra（截取前 n_days 天）。"""
    tmpl = rp.factor_store_v5_template(STORE)
    dates_all = np.asarray(tmpl["dates"], dtype=np.int64)
    stocks = list(tmpl["stocks"])
    dates = dates_all[:n_days]
    ind = dw.tail_v4._load_industry_matrix(dates, stocks)
    restrict = np.load(BP + "restrict.npy")[:n_days].astype(np.float32)
    ret_sum1 = np.load(BP + "ret_sum_gap1.npy")[:n_days]
    ret_sum5 = np.load(BP + "ret_sum_gap5.npy")[:n_days]
    # barra 映射到模板轴 (T,N,10)：date->行、6位code->列，直接索引构造
    bar = pd.read_parquet(BARRA, columns=["date", "code"] + [f"value_{i}" for i in range(10)])
    bar["code"] = bar["code"].str[:6]
    bar = bar[bar["date"].isin(dates)]
    row_of = {d: i for i, d in enumerate(dates)}
    col_of = {c: i for i, c in enumerate(s[:6] for s in stocks)}
    barra = np.full((n_days, N, 10), np.nan)
    di = bar["date"].map(row_of).values
    ci = bar["code"].map(col_of).values
    ok = ~np.isnan(di) & ~np.isnan(ci)
    di, ci = di[ok].astype(int), ci[ok].astype(int)
    vals = bar[[f"value_{i}" for i in range(10)]].values[ok]
    barra[di, ci] = vals
    return dates, stocks, ind, restrict, ret_sum1, ret_sum5, barra


def read_factor_matrix(col_idx, n_days, T=T, N=N):
    d = rp.factor_store_v5_read_factor(STORE, col_idx)
    mat = np.full((n_days, N), np.nan, dtype=np.float32)
    ok = d["date_id"] < n_days
    mat[d["date_id"][ok], d["code_id"][ok]] = d["factor"][ok]
    return mat


def rank_pct(mat):
    """pandas rank(axis=1, pct=True) 语义：非 NaN 平均秩 / n_non_nan。"""
    return pd.DataFrame(mat).rank(axis=1, pct=True).values


def fill_ind_reg(fv, ind, size_ranked):
    """行业 OLS 填充（复刻 fill_ind_reg，3 层：ind2 -> ind1 -> ind0）。"""
    fv = fv.copy()
    ind1 = np.floor(ind / 10000.0)
    ind2 = np.floor(ind / 100.0)
    ind0 = np.where(np.isnan(ind1), 0.0, 1.0)
    for level in (ind2, ind1, ind0):
        for t in range(fv.shape[0]):
            row = fv[t]
            if len(np.unique(row[~np.isnan(row)])) + (1 if np.isnan(row).any() else 0) < 10:
                continue
            codes = level[t]
            order = np.argsort(codes, kind="stable")
            cs = codes[order]
            # 分组段
            start = 0
            while start < len(cs):
                c = cs[start]
                if np.isnan(c):
                    break
                end = start + 1
                while end < len(cs) and cs[end] == c:
                    end += 1
                idx = order[start:end]
                y = row[idx]; b = size_ranked[t, idx]
                nn = ~np.isnan(y) & ~np.isnan(b)
                if nn.sum() >= 10:
                    c0, c1 = np.polyfit(b[nn], y[nn], 1)
                    fill = ~nn
                    row[idx[fill]] = c0 + c1 * b[fill]
                start = end
    return fv


def group_median_fill(fv, codes, valid_mask, source):
    """分组中位填充（median 固定来自 source）。"""
    for t in range(fv.shape[0]):
        row = fv[t]; src = source[t]
        if not np.isnan(row).any():
            continue
        cs = codes[t]
        vm = np.ones(cs.shape) if valid_mask is None else valid_mask[t]
        order = np.argsort(cs, kind="stable")
        srt = cs[order]
        start = 0
        while start < len(srt):
            c = srt[start]
            if np.isnan(c):
                break
            end = start + 1
            while end < len(srt) and srt[end] == c:
                end += 1
            idx = order[start:end]
            sv = src[idx][(vm[idx] == 1.0) & ~np.isnan(src[idx])]
            if len(sv):
                med = np.median(sv)
                fill = np.isnan(row[idx]) & (vm[idx] == 1.0)
                row[idx[fill]] = med
            start = end


def get_residual(fv, barra, ind, industry=True):
    """逐日 OLS 残差（复刻 get_residual，industry 模式：X=[10风格, ind1 one-hot]）。"""
    resid = np.full_like(fv, np.nan)
    ind1 = np.floor(ind / 10000.0)
    k = barra.shape[2]
    for t in range(fv.shape[0]):
        y = fv[t]
        if np.isnan(y).all():
            continue
        icodes = np.unique(ind1[t][~np.isnan(ind1[t])])
        n_ind = len(icodes) if industry else 0
        valid = ~np.isnan(y) & np.all(~np.isnan(barra[t]), axis=1)
        nv = valid.sum()
        if nv <= 10:
            continue
        yv = y[valid]
        if len(np.unique(yv)) == 1:
            resid[t, valid] = 0.5
            continue
        X = barra[t][valid].copy()
        if industry:
            ic = ind1[t][valid]
            cols = []
            for c in icodes:
                cols.append((ic == c).astype(np.float64))
            X = np.concatenate([X] + [np.array(c)[:, None] for c in cols], axis=1)
        else:
            X = np.concatenate([np.ones((nv, 1)), X], axis=1)
        beta = np.linalg.solve(X.T @ X, X.T @ yv)
        resid[t, valid] = yv - X @ beta
    return resid


def neutralize_py(factor, ind, restrict, barra, industry=True):
    """完整复刻 neutralize_std_section（f64 路径）。barra 为原始风格值。"""
    barra_r = np.stack([rank_pct(barra[:, :, i].copy()) for i in range(barra.shape[2])], axis=2)
    size_ranked = barra_r[:, :, 2]
    fv = rank_pct(factor.astype(np.float64))
    fv = fill_ind_reg(fv, ind, size_ranked)
    ind1 = np.floor(ind / 10000.0)
    fv[np.isnan(ind1)] = np.nan
    filled = fv.copy()
    source = filled.copy()
    group_median_fill(filled, np.floor(ind / 100.0), None, source)
    group_median_fill(filled, ind1, None, source)
    group_median_fill(filled, np.zeros_like(ind), np.where(np.isnan(ind1), 0.0, 1.0), source)
    filled[restrict != 0] = np.nan
    a = rank_pct(filled)
    resid = get_residual(a, barra_r, ind, industry)
    return rank_pct(resid), resid, a


def spearman(x, y):
    """legacy_spearman_correlation 复刻：ordinal 秩（tie 稳定序不同秩）+ 秩差平方和。"""
    n = len(x)
    if n < 2:
        return np.nan
    rx = np.argsort(np.argsort(x, kind="stable"), kind="stable").astype(np.float64)
    ry = np.argsort(np.argsort(y, kind="stable"), kind="stable").astype(np.float64)
    d2 = ((rx - ry) ** 2).sum()
    return 1.0 - 6.0 * d2 / (n * (n * n - 1.0))


def backtest_ic(sig, ret_sum, restrict, gap, backtest_start=20170201, dates=None):
    """legacy_backtest_single_factor_with_effective 的 IC 序列复刻。"""
    ic_vals = []
    for t in range(1, sig.shape[0]):
        if dates is not None and dates[t] <= backtest_start:
            continue
        held = t - 1
        if (t - 1) % gap == 0:  # 近似 held 更新（effective 序列连续时等价）
            pass
        s = sig[held]; r = ret_sum[t]
        ok = np.isfinite(s) & np.isfinite(r) & (restrict[held] == 0)
        if (t) % gap == 0 and ok.sum() >= 2:
            ic_vals.append(spearman(r[ok], s[ok]))
    return np.array(ic_vals)


if __name__ == "__main__":
    n_days = 90
    dates, stocks, ind, restrict, ret_sum1, ret_sum5, barra = load_all(n_days)
    print("loaded:", n_days, "days")

    col = 0
    raw = read_factor_matrix(col, n_days)
    # 复刻 rank + 缺失中位秩填充（slot0 = _smooth_1）
    ranked = pd.DataFrame(raw).rank(axis=1).values
    med = (np.isfinite(ranked).sum(axis=1) + 1.0) / 2.0
    ranked = np.where(np.isnan(ranked), med[:, None], ranked)
    slot = ranked.astype(np.float32)

    # 生产基线
    prod = rp.neutralize_std_block_py(
        slot[:, :, None], ind, restrict,
        BARRA, dates.astype(np.int32).tolist(), stocks, True)
    prod = np.asarray(prod)[:, :, 0]
    # Python 复刻
    x_neu, resid, a = neutralize_py(slot, ind, restrict, barra, True)
    both = np.isfinite(prod) & np.isfinite(x_neu)
    print("复刻 vs 生产: 共同有限", both.sum(),
          "| NaN 模式一致:", np.array_equal(np.isfinite(prod), np.isfinite(x_neu)),
          "| max|diff|:", np.abs(prod[both].astype(np.float64) - x_neu[both]).max(),
          "| 秩相关:", spearman(prod[both].astype(np.float64), x_neu[both]))
