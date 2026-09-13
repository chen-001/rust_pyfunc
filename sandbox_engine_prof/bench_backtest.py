"""回测 opt2 (4-pass radix + 单次排序融合) vs 生产 opt: 逐位校验 + 计时。"""
import sys, time, json
import numpy as np

sys.path.insert(0, "/home/chenzongwei/rust_pyfunc/sandbox_engine_opt")
sys.path.insert(0, "/home/chenzongwei/rust_pyfunc/sandbox_rankic_neu")
from prototype import load_all, read_factor_matrix
import dev_sandbox_engine_opt as dso

N_DAYS = 2276
BP = "/home/chenzongwei/pythoncode/_tail_v2_shared/backtest_inputs/000905_20170103_20260522_5438_6b8884a67f05d221/"
BT_START = 20170201


def engine_slot0(raw):
    ranked = np.asarray(dso.bench_rank_py(np.ascontiguousarray(raw))[1])
    med = (np.isfinite(ranked).sum(axis=1) + 1.0) / 2.0
    last_valid = -np.ones(raw.shape[1], dtype=int)
    for t in range(raw.shape[0]):
        for s in range(raw.shape[1]):
            if np.isfinite(raw[t, s]):
                last_valid[s] = t
        miss = (t - last_valid <= 20) & (last_valid >= 0)
        nan_ok = np.isnan(ranked[t]) & miss
        ranked[t, nan_ok] = med[t]
    return ranked.astype(np.float32)


def main():
    dates, stocks, ind, restrict, ret_sum1, ret_sum5, barra_raw = load_all(N_DAYS)
    restrict = np.ascontiguousarray(restrict.astype(np.float32))
    ret_sum1 = np.ascontiguousarray(ret_sum1.astype(np.float32))
    ret_sum5 = np.ascontiguousarray(ret_sum5.astype(np.float32))
    ret_g1 = np.ascontiguousarray(np.load(BP + "ret_gap1.npy")[:N_DAYS].astype(np.float32))
    ret_g5 = np.ascontiguousarray(np.load(BP + "ret_gap5.npy")[:N_DAYS].astype(np.float32))
    index_ret = np.ascontiguousarray(np.load(BP + "index_ret.npy")[:N_DAYS].astype(np.float32))
    raw = read_factor_matrix(0, N_DAYS).astype(np.float32)
    slot0 = np.ascontiguousarray(engine_slot0(raw))
    print("data ready", flush=True)

    out = {}
    for ic_only in (False, True):
        pre_ms, base_ms, opt_ms, sums, ic_base, ic_opt = dso.bench_backtest_py(
            slot0, ret_g1, ret_sum1, ret_g5, ret_sum5, restrict, index_ret,
            dates.astype(np.int32), BT_START, ic_only)
        pre2, opt2_ms, sums2, ics2 = dso.bench_backtest_opt2_py(
            slot0, ret_g1, ret_sum1, ret_g5, ret_sum5, restrict, index_ret,
            dates.astype(np.int32), BT_START, ic_only)
        s = np.array(sums)[20:]  # 生产 opt 部分 (前20为 baseline)
        s2 = np.array(sums2)
        eq_s = np.array_equal(np.nan_to_num(s, nan=0.0), np.nan_to_num(s2, nan=0.0))
        md_s = np.abs(np.nan_to_num(s, nan=0.0) - np.nan_to_num(s2, nan=0.0)).max()
        ic1, ic2 = np.array(ic_opt), np.array(ics2)
        eq_ic = np.array_equal(np.nan_to_num(ic1, nan=0.0), np.nan_to_num(ic2, nan=0.0))
        md_ic = np.abs(np.nan_to_num(ic1, nan=0.0) - np.nan_to_num(ic2, nan=0.0)).max()
        print(f"ic_only={ic_only}: opt={opt_ms:.0f}ms opt2={opt2_ms:.0f}ms "
              f"summary_eq={eq_s} maxdiff={md_s:.3e} ic_eq={eq_ic} ic_maxdiff={md_ic:.3e}", flush=True)
        out[f"ic_only={ic_only}"] = dict(opt_ms=opt_ms, opt2_ms=opt2_ms, pre_ms=pre_ms,
                                         pre2_ms=pre2, summary_eq=bool(eq_s), maxdiff=float(md_s),
                                         ic_eq=bool(eq_ic), ic_maxdiff=float(md_ic))
    with open("/tmp/sandbox_bt_bench.json", "w") as f:
        json.dump(out, f, indent=2)
    print("saved /tmp/sandbox_bt_bench.json", flush=True)


if __name__ == "__main__":
    main()
