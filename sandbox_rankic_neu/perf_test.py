"""提速基准：各方案同输入同规模多次计时（release 构建）。

方案: prod(生产 rp.neutralize_std_block_py, 含 parquet 读)
      full(现状复刻) / c(省resid rank) / cpp(预计算排序+省rank) / c3e(无NaN快路径)
      ret(收益侧中性化一次)
"""
import numpy as np
import pandas as pd
import rust_pyfunc as rp
import sys
import time

sys.path.insert(0, "/home/chenzongwei/design_whatever")
import design_whatever as dw
sys.path.insert(0, ".")
from prototype import load_all, read_factor_matrix
import dev_sandbox_rankic as ds

BARRA = "/home/chenzongwei/database/barra/barra_daily_together_jason.parquet"


def bench(fn, *args, n=5, **kw):
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        r = fn(*args, **kw)
        ts.append(time.perf_counter() - t0)
    return r, min(ts), np.mean(ts)


def main():
    n_days = 2276
    print("loading...", flush=True)
    dates, stocks, ind, restrict, ret_sum1, ret_sum5, barra = load_all(n_days)
    ind = np.ascontiguousarray(ind)
    restrict = np.ascontiguousarray(restrict)
    barra = np.ascontiguousarray(barra)
    ret_sum1 = np.ascontiguousarray(ret_sum1)
    ret_sum5 = np.ascontiguousarray(ret_sum5)
    raw = read_factor_matrix(0, n_days)
    ranked = pd.DataFrame(raw).rank(axis=1).values
    med = (np.isfinite(ranked).sum(axis=1) + 1.0) / 2.0
    ranked = np.where(np.isnan(ranked), med[:, None], ranked)
    slot = np.ascontiguousarray(ranked.astype(np.float32))
    print("data ready", flush=True)

    orders = ds.precompute_orders(ind)
    t0 = time.perf_counter()
    orders = ds.precompute_orders(ind)
    print(f"precompute_orders: {time.perf_counter()-t0:.2f}s", flush=True)

    results = {}
    results["full(现状复刻)"] = bench(ds.neutralize_full, slot, ind, restrict, barra, True)
    results["c(省resid rank)"] = bench(ds.neutralize_c, slot, ind, restrict, barra, True)
    results["cpp(预计算排序)"] = bench(ds.neutralize_cpp, slot, ind, restrict, barra, orders, True)
    results["c3exact(无NaN快路径)"] = bench(ds.neutralize_c3_exact, slot, ind, restrict, barra, orders, True)
    results["ret(收益侧一次)"] = bench(ds.neutralize_ret, ret_sum1, ind, restrict, barra, True)
    results["prod(生产,含parquet读)"] = bench(
        rp.neutralize_std_block_py, slot[:, :, None], ind, restrict, BARRA,
        dates.astype(np.int32).tolist(), stocks, True, n=2)
    # IC 计算
    resid = np.asarray(results["c(省resid rank)"][0])
    dates32 = dates.astype(np.int32).tolist()
    results["ic_series(resid64)"] = bench(ds.ic_series_resid64_py, resid, ret_sum1, restrict, dates32, 20170201, 1)
    results["ic_series(xneu32)"] = bench(ds.ic_series_py, np.asarray(results["full(现状复刻)"][0][0]), ret_sum1, restrict, dates32, 20170201, 1)

    print("\n=== 2276天×5438股 单slot 耗时 (5次取min/mean, 秒) ===", flush=True)
    base_min = results["full(现状复刻)"][1]
    for k, (r, tmin, tmean) in results.items():
        print(f"{k:26s} min={tmin:8.3f} mean={tmean:8.3f}  相对full={tmean/base_min:5.2f}x", flush=True)


if __name__ == "__main__":
    main()
