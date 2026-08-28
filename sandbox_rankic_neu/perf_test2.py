"""公平提速对比 (release, 共享预计算): per-slot 成本对比。

架构对齐生产: barra_ranked / 行业排序 预计算一次共享。
  A_full_s  : 现状复刻 (共享 barra) = neutralize_core_s 全链路
  C'        : 省 resid rank_pct (共享 barra)
  C''       : C' + 预计算行业排序
  C3-exact  : 无 NaN 快路径 (共享 barra + 预计算排序)
  ret(B)    : 收益侧中性化一次 (共享 barra)
  ic        : 回测 IC (resid64)
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
    raw = read_factor_matrix(0, n_days)
    ranked = pd.DataFrame(raw).rank(axis=1).values
    med = (np.isfinite(ranked).sum(axis=1) + 1.0) / 2.0
    ranked = np.where(np.isnan(ranked), med[:, None], ranked)
    slot = np.ascontiguousarray(ranked.astype(np.float32))
    print("data ready", flush=True)

    # 一次性预计算 (与生产 neutralize_std_precompute 对齐)
    t0 = time.perf_counter()
    barra_r = np.asarray(ds.precompute_barra(barra))
    print(f"precompute_barra: {time.perf_counter()-t0:.2f}s", flush=True)
    t0 = time.perf_counter()
    orders = ds.precompute_orders(ind)
    print(f"precompute_orders: {time.perf_counter()-t0:.2f}s", flush=True)

    # 现状复刻 (共享 barra) 需要 neutralize_core_s 但输出 x_neu —— 用 ds.neutralize_full 不含共享。
    # 用 C''+resid_rank 模拟: 省去的 resid_rank 计 1 次 rank_pct 成本。此处直接用 full (含 barra 重算) 作对照,
    # per-slot 成本 = full - precompute_barra。
    results = {}
    results["A=full(现状,含barra重算)"] = bench(ds.neutralize_full, slot, ind, restrict, barra, True)
    results["C'(共享barra,省resid rank)"] = bench(ds.neutralize_c_s, slot, ind, restrict, barra_r, True)
    results["C''(再+预计算排序)"] = bench(ds.neutralize_cpp, slot, ind, restrict, barra_r, orders, True)
    results["C3-exact(无NaN快路径)"] = bench(ds.neutralize_c3_exact, slot, ind, restrict, barra_r, orders, True)
    results["ret(B收益侧一次,共享barra)"] = bench(ds.neutralize_ret, ret_sum1, ind, restrict, barra, True)

    resid = np.asarray(results["C'(共享barra,省resid rank)"][0])
    dates32 = dates.astype(np.int32).tolist()
    results["ic_series(resid64)"] = bench(ds.ic_series_resid64_py, resid, ret_sum1, restrict, dates32, 20170201, 1)

    print("\n=== 2276天×5438股 单slot (release, 5次min/mean, 秒) ===", flush=True)
    base = results["A=full(现状,含barra重算)"][2]
    for k, (r, tmin, tmean) in results.items():
        print(f"{k:32s} min={tmin:7.3f} mean={tmean:7.3f}  相对A_full={tmean/base:5.2f}x", flush=True)
    barra_cost = 0
    print(f"\n扣除 barra 预计算 (约 {barra_cost}s) 后 per-slot 成本:", flush=True)
    for k, (r, tmin, tmean) in results.items():
        if k.startswith("A="):
            print(f"  {k}: {tmean - (np.asarray(r[0][0]).nbytes/8/1e9):.1f}?  -- 见阶段计时", flush=True)


if __name__ == "__main__":
    main()
