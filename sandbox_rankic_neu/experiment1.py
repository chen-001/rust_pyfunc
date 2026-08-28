"""RankIC 方案对照实验（Python 原型，小样本）。

方案:
  A  现状: spearman(neutralize(slot), r)
  B  备选: spearman(slot, neutralize(r))          —— 收益侧中性化(风格取信号日)
  B' 备选变体: 收益侧中性化不做行业填充(停牌股残差NaN不进截面)
  C  精确等价: spearman(resid, r)                  —— 省 resid 最终 rank
  C3 精确等价(无NaN slot): spearman(OLS_resid(slot), r) —— 省全部中间 rank/fill
  E/F/G cov 恒等式族(线性版, 理论对照)
"""
import numpy as np
import pandas as pd
import rust_pyfunc as rp
import sys

sys.path.insert(0, "/home/chenzongwei/design_whatever")
import design_whatever as dw
from prototype import (load_all, read_factor_matrix, neutralize_py, rank_pct,
                       fill_ind_reg, group_median_fill, get_residual, spearman)

BARRA = "/home/chenzongwei/database/barra/barra_daily_together_jason.parquet"


def neutralize_ret(r, ind, restrict, barra, industry=True, do_fill=True):
    """对收益做中性化（复刻同一链路）。do_fill=False: 跳过行业填充。"""
    barra_r = np.stack([rank_pct(barra[:, :, i].copy()) for i in range(barra.shape[2])], axis=2)
    size_ranked = barra_r[:, :, 2]
    fv = rank_pct(r.astype(np.float64))
    if do_fill:
        fv = fill_ind_reg(fv, ind, size_ranked)
        ind1 = np.floor(ind / 10000.0)
        fv[np.isnan(ind1)] = np.nan
        filled = fv.copy()
        source = filled.copy()
        group_median_fill(filled, np.floor(ind / 100.0), None, source)
        group_median_fill(filled, ind1, None, source)
        group_median_fill(filled, np.zeros_like(ind), np.where(np.isnan(ind1), 0.0, 1.0), source)
    else:
        ind1 = np.floor(ind / 10000.0)
        filled = fv.copy()
    filled[restrict != 0] = np.nan
    a = rank_pct(filled)
    resid = get_residual(a, barra_r, ind, industry)
    return rank_pct(resid), resid


def backtest_ic_pairs(sig, ret_sum, restrict, gap, backtest_start, dates):
    """返回 (信号截面, 收益截面) 对齐后的数组对列表（复刻 held 对齐）。"""
    eff = [t for t in range(1, sig.shape[0])
           if dates[t] > backtest_start and np.isfinite(sig[t - 1]).any()]
    pairs = []
    held = eff[0] - 1
    for local_t, t in enumerate(eff):
        if local_t % gap == 0:
            held = t - 1
        if (local_t + 1) % gap == 0:
            pairs.append((held, t))
    return pairs


def ic_series(sig, ret_sum, restrict, gap, backtest_start, dates):
    pairs = backtest_ic_pairs(sig, ret_sum, restrict, gap, backtest_start, dates)
    out = []
    for held, t in pairs:
        s = sig[held]; r = ret_sum[t]
        ok = np.isfinite(s) & np.isfinite(r) & (restrict[held] == 0)
        out.append(spearman(r[ok], s[ok]) if ok.sum() >= 2 else np.nan)
    return np.array(out)


def pearson(x, y):
    if len(x) < 2:
        return np.nan
    return np.corrcoef(x, y)[0, 1]


def ic_series_linear(sig, ret_sum, restrict, gap, backtest_start, dates, mode):
    """线性版 IC：mode 指定用哪侧的中性化残差/秩。"""
    pairs = backtest_ic_pairs(sig, ret_sum, restrict, gap, backtest_start, dates)
    out = []
    for held, t in pairs:
        s = sig[held]; r = ret_sum[t]
        ok = np.isfinite(s) & np.isfinite(r) & (restrict[held] == 0)
        if ok.sum() < 2:
            out.append(np.nan)
            continue
        out.append(pearson(s[ok], r[ok]))
    return np.array(out)


def cmp(a, b, label):
    m = ~(np.isnan(a) | np.isnan(b))
    d = a[m] - b[m]
    same = np.sum(d == 0.0)
    return (f"{label}: n={m.sum()} max|Δ|={np.abs(d).max():.3e} 逐位一致率={same/m.sum():.4f} "
            f"corr={np.corrcoef(a[m], b[m])[0,1]:.6f} 符号一致率={np.mean(np.sign(a[m])==np.sign(b[m])):.4f} "
            f"Δstd={d.std():.3e}")


def main():
    n_days = 240
    dates, stocks, ind, restrict, ret_sum1, ret_sum5, barra = load_all(n_days)
    print("loaded", n_days, "days", "N =", ind.shape[1])
    bt_start = 20170201
    cols = [0, 5, 20, 100]

    for gap, rsum in ((1, ret_sum1), (5, ret_sum5)):
        print(f"\n===== gap{gap} =====")
        r_neu_full, r_resid = neutralize_ret(rsum, ind, restrict, barra, True, do_fill=True)
        r_neu_nofill, _ = neutralize_ret(rsum, ind, restrict, barra, True, do_fill=False)
        for col in cols:
            raw = read_factor_matrix(col, n_days)
            ranked = pd.DataFrame(raw).rank(axis=1).values
            med = (np.isfinite(ranked).sum(axis=1) + 1.0) / 2.0
            ranked = np.where(np.isnan(ranked), med[:, None], ranked)
            slot = ranked.astype(np.float32)

            x_neu, resid, a = neutralize_py(slot, ind, restrict, barra, True)
            resid32 = resid.astype(np.float32)
            icA = ic_series(x_neu, rsum, restrict, gap, bt_start, dates)
            icB = ic_series(slot, r_neu_full, restrict, gap, bt_start, dates)
            icBp = ic_series(slot, r_neu_nofill, restrict, gap, bt_start, dates)
            icC = ic_series(resid32, rsum, restrict, gap, bt_start, dates)
            # C3: 直接对 slot 做 OLS 残差（跳过 rank/fill），valid 同口径
            barra_r = np.stack([rank_pct(barra[:, :, i].copy()) for i in range(barra.shape[2])], axis=2)
            fv3 = np.where(restrict == 0, slot.astype(np.float64), np.nan)
            resid3 = get_residual(fv3, barra_r, ind, True)
            icC3 = ic_series(resid3.astype(np.float32), rsum, restrict, gap, bt_start, dates)
            print(f"[col{col}]", cmp(icA, icB, "B  vs A"), sep="")
            print(f"        ", cmp(icA, icBp, "B' vs A"), sep="")
            print(f"        ", cmp(icA, icC, "C  vs A"), sep="")
            print(f"        ", cmp(icA, icC3, "C3 vs A"), sep="")
            print(f"        icA mean={np.nanmean(icA):+.4f}  icB mean={np.nanmean(icB):+.4f}  icB' mean={np.nanmean(icBp):+.4f}")


if __name__ == "__main__":
    main()
