"""全量验证：2276 天 × 30 因子 × gap1/gap5。

对每个因子:
  A  基线(生产链路复刻): x_neu = neutralize_full -> spearman(x_neu32, r)
  C' 精确等价: resid(f64) -> spearman f64 域秩 (应与 A 逐位一致)
  C3 近似: 跳过填充直接 OLS
  B  收益侧中性化: spearman(slot, r_neu)  (r_neu 全因子共享, 只算一次)
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

BARRA = "/home/chenzongwei/database/barra/barra_daily_together_jason.parquet"
N_FACTORS = 30
BT = 20170201


def main():
    n_days = 2276
    print("loading...", flush=True)
    dates, stocks, ind, restrict, ret_sum1, ret_sum5, barra = load_all(n_days)
    print("loaded", n_days, "days", flush=True)

    # 收益侧中性化: 只算一次, 全因子共享 (转连续布局, 对齐 Rust 端 row.as_slice)
    ret_sum1 = np.ascontiguousarray(ret_sum1)
    ret_sum5 = np.ascontiguousarray(ret_sum5)
    restrict_c = np.ascontiguousarray(restrict)
    slot_in = None
    barra_c = np.ascontiguousarray(barra)
    r_neu1, _ = ds.neutralize_ret(ret_sum1, ind, restrict_c, barra_c, True)
    r_neu5, _ = ds.neutralize_ret(ret_sum5, ind, restrict_c, barra_c, True)
    r_neu1 = np.asarray(r_neu1)
    r_neu5 = np.asarray(r_neu5)
    print("r_neu done", flush=True)

    rows = []
    for col in range(N_FACTORS):
        raw = read_factor_matrix(col, n_days)
        ranked = pd.DataFrame(raw).rank(axis=1).values
        med = (np.isfinite(ranked).sum(axis=1) + 1.0) / 2.0
        ranked = np.where(np.isnan(ranked), med[:, None], ranked)
        slot = ranked.astype(np.float32)

        x_neu, _ = ds.neutralize_full(slot, ind, restrict_c, barra_c, True)
        x_neu = np.asarray(x_neu)
        resid = np.asarray(ds.neutralize_c(slot, ind, restrict_c, barra_c, True))
        resid3 = np.asarray(ds.neutralize_c3(slot, ind, restrict_c, barra_c, True))
        dates32 = dates.astype(np.int32).tolist()

        for gap, rsum, rneu in ((1, ret_sum1, r_neu1), (5, ret_sum5, r_neu5)):
            icA = np.asarray(ds.ic_series_py(x_neu, rsum, restrict_c, dates32, BT, gap))
            icC = np.asarray(ds.ic_series_resid64_py(resid, rsum, restrict_c, dates32, BT, gap))
            icC3 = np.asarray(ds.ic_series_resid64_py(resid3.astype(np.float64), rsum, restrict_c, dates32, BT, gap))
            icB = np.asarray(ds.ic_series_py(slot, rneu, restrict_c, dates32, BT, gap))
            rows.append((col, gap, "A_vs_C", icA, icC))
            rows.append((col, gap, "A_vs_C3", icA, icC3))
            rows.append((col, gap, "A_vs_B", icA, icB))
        print(f"col{col} done", flush=True)

    np.save("ic_compare_rows.npy", np.array(rows, dtype=object), allow_pickle=True)
    # 汇总
    for gap in (1, 5):
        for tag in ("A_vs_C", "A_vs_C3", "A_vs_B"):
            stats = []
            for col, g, t, a, b in rows:
                if g == gap and t == tag:
                    m = ~(np.isnan(a) | np.isnan(b))
                    d = a[m] - b[m]
                    stats.append((
                        np.abs(d).max(),
                        (d == 0).mean(),
                        np.corrcoef(a[m], b[m])[0, 1],
                        np.mean(np.sign(a[m]) == np.sign(b[m])),
                        d.std(),
                        np.nanmean(a), np.nanmean(b),
                    ))
            s = np.array(stats)
            print(f"gap{gap} {tag}: n因子={len(s)} max|Δ|max={s[:,0].max():.3e} 逐位一致率均值={s[:,1].mean():.4f} "
                  f"corr均值={s[:,2].mean():.6f} 符号一致率均值={s[:,3].mean():.4f} Δstd均值={s[:,4].mean():.3e} "
                  f"icA均值={s[:,5].mean():+.4f} ic均值={s[:,6].mean():+.4f}", flush=True)


if __name__ == "__main__":
    main()
