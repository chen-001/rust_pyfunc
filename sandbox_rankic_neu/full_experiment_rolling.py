"""全量滚动 slot 验证 (2276 天 × 30 因子, slot 含 NaN 走填充路径)。

slot = rolling mean(window=20, min_periods=10) 作用在 rank 填充后的因子上
(复刻 _smooth_1 之外的真实派生 slot 形态: 时间维滚动, 前 9 天 NaN)。
"""
import numpy as np
import pandas as pd
import rust_pyfunc as rp
import sys

sys.path.insert(0, "/home/chenzongwei/design_whatever")
import design_whatever as dw
sys.path.insert(0, ".")
from prototype import load_all, read_factor_matrix
import dev_sandbox_rankic as ds

N_FACTORS = 30
BT = 20170201


def main():
    n_days = 2276
    print("loading...", flush=True)
    dates, stocks, ind, restrict, ret_sum1, ret_sum5, barra = load_all(n_days)
    ind = np.ascontiguousarray(ind)
    restrict = np.ascontiguousarray(restrict)
    barra = np.ascontiguousarray(barra)
    ret_sum1 = np.ascontiguousarray(ret_sum1)
    ret_sum5 = np.ascontiguousarray(ret_sum5)
    barra_r = np.asarray(ds.precompute_barra(barra))
    orders = ds.precompute_orders(ind)
    r_neu1, _ = ds.neutralize_ret(ret_sum1, ind, restrict, barra, True)
    r_neu5, _ = ds.neutralize_ret(ret_sum5, ind, restrict, barra, True)
    r_neu1 = np.asarray(r_neu1)
    r_neu5 = np.asarray(r_neu5)
    print("shared ready", flush=True)

    rows = []
    dates32 = dates.astype(np.int32).tolist()
    for col in range(N_FACTORS):
        raw = read_factor_matrix(col, n_days)
        ranked = pd.DataFrame(raw).rank(axis=1).values
        med = (np.isfinite(ranked).sum(axis=1) + 1.0) / 2.0
        ranked = np.where(np.isnan(ranked), med[:, None], ranked)
        slot = pd.DataFrame(ranked).rolling(20, min_periods=10).mean().values.astype(np.float32)
        slot = np.ascontiguousarray(slot)
        assert np.isnan(slot).sum() > 0, "slot 应含 NaN (窗口预热)"

        x_neu, _ = ds.neutralize_full(slot, ind, restrict, barra, True)
        x_neu = np.asarray(x_neu)
        resid_c = np.asarray(ds.neutralize_c_s(slot, ind, restrict, barra_r, True))
        resid_cpp = np.asarray(ds.neutralize_cpp(slot, ind, restrict, barra_r, orders, True))
        resid_c3e = np.asarray(ds.neutralize_c3_exact(slot, ind, restrict, barra_r, orders, True))

        for gap, rsum, rneu in ((1, ret_sum1, r_neu1), (5, ret_sum5, r_neu5)):
            icA = np.asarray(ds.ic_series_py(x_neu, rsum, restrict, dates32, BT, gap))
            icC = np.asarray(ds.ic_series_resid64_py(resid_c, rsum, restrict, dates32, BT, gap))
            icCPP = np.asarray(ds.ic_series_resid64_py(resid_cpp, rsum, restrict, dates32, BT, gap))
            icC3E = np.asarray(ds.ic_series_resid64_py(resid_c3e, rsum, restrict, dates32, BT, gap))
            icB = np.asarray(ds.ic_series_py(slot, rneu, restrict, dates32, BT, gap))
            rows.append((col, gap, "A_vs_C", icA, icC))
            rows.append((col, gap, "A_vs_CPP", icA, icCPP))
            rows.append((col, gap, "A_vs_C3E", icA, icC3E))
            rows.append((col, gap, "A_vs_B", icA, icB))
        print(f"col{col} done", flush=True)

    np.save("ic_compare_rows_rolling.npy", np.array(rows, dtype=object), allow_pickle=True)
    for gap in (1, 5):
        for tag in ("A_vs_C", "A_vs_CPP", "A_vs_C3E", "A_vs_B"):
            stats = []
            for col, g, t, a, b in rows:
                if g == gap and t == tag:
                    m = ~(np.isnan(a) | np.isnan(b))
                    d = a[m] - b[m]
                    stats.append((np.abs(d).max(), (d == 0).mean(),
                                  np.corrcoef(a[m], b[m])[0, 1],
                                  np.mean(np.sign(a[m]) == np.sign(b[m])), d.std()))
            s = np.array(stats)
            print(f"gap{gap} {tag}: n因子={len(s)} max|Δ|max={s[:,0].max():.3e} 逐位一致率均值={s[:,1].mean():.4f} "
                  f"corr均值={s[:,2].mean():.6f} 符号一致率均值={s[:,3].mean():.4f} Δstd均值={s[:,4].mean():.3e}", flush=True)


if __name__ == "__main__":
    main()
