"""差异归因: A(因子中性化) / B(收益中性化) / F(双边) / raw(无) 的关系。"""
import numpy as np
import pandas as pd
import rust_pyfunc as rp
import sys

sys.path.insert(0, "/home/chenzongwei/design_whatever")
import design_whatever as dw
sys.path.insert(0, ".")
from prototype import load_all, read_factor_matrix
import dev_sandbox_rankic as ds


def main():
    n_days = 240
    dates, stocks, ind, restrict, ret_sum1, ret_sum5, barra = load_all(n_days)
    ind = np.ascontiguousarray(ind); restrict = np.ascontiguousarray(restrict)
    barra = np.ascontiguousarray(barra)
    ret_sum1 = np.ascontiguousarray(ret_sum1); ret_sum5 = np.ascontiguousarray(ret_sum5)
    r_neu1, _ = ds.neutralize_ret(ret_sum1, ind, restrict, barra, True)
    r_neu5, _ = ds.neutralize_ret(ret_sum5, ind, restrict, barra, True)
    r_neu1 = np.asarray(r_neu1); r_neu5 = np.asarray(r_neu5)
    dates32 = dates.astype(np.int32).tolist()
    bt = 20170201

    print(f"{'col':>4} {'gap':>3} {'raw':>7} {'A':>7} {'B':>7} {'F':>7}  corr(A,B) corr(A,F) corr(A,raw) corr(B,F)")
    for col in [0, 5, 20, 100]:
        raw = read_factor_matrix(col, n_days)
        ranked = pd.DataFrame(raw).rank(axis=1).values
        med = (np.isfinite(ranked).sum(axis=1) + 1.0) / 2.0
        ranked = np.where(np.isnan(ranked), med[:, None], ranked)
        slot = np.ascontiguousarray(ranked.astype(np.float32))
        x_neu, _ = ds.neutralize_full(slot, ind, restrict, barra, True)
        x_neu = np.asarray(x_neu)
        for gap, rsum, rneu in ((1, ret_sum1, r_neu1), (5, ret_sum5, r_neu5)):
            icR = np.asarray(ds.ic_series_py(slot, rsum, restrict, dates32, bt, gap))
            icA = np.asarray(ds.ic_series_py(x_neu, rsum, restrict, dates32, bt, gap))
            icB = np.asarray(ds.ic_series_py(slot, rneu, restrict, dates32, bt, gap))
            icF = np.asarray(ds.ic_series_py(x_neu, rneu, restrict, dates32, bt, gap))
            m = ~(np.isnan(icA) | np.isnan(icB) | np.isnan(icF) | np.isnan(icR))
            c_ab = np.corrcoef(icA[m], icB[m])[0, 1]
            c_af = np.corrcoef(icA[m], icF[m])[0, 1]
            c_ar = np.corrcoef(icA[m], icR[m])[0, 1]
            c_bf = np.corrcoef(icB[m], icF[m])[0, 1]
            print(f"{col:>4} {gap:>3} {np.nanmean(icR):+7.4f} {np.nanmean(icA):+7.4f} "
                  f"{np.nanmean(icB):+7.4f} {np.nanmean(icF):+7.4f}  {c_ab:8.5f} {c_af:8.5f} {c_ar:8.5f} {c_bf:8.5f}")


if __name__ == "__main__":
    main()
