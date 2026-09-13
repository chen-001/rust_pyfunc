"""引擎瓶颈分析与优化验证：真实数据(2276×5438)单 slot/单因子基准。

分阶段对比 baseline(生产原样复刻) vs opt(radix秩+收益秩预排序walk+行主序rolling+中性化预计算)。
"""
import numpy as np
import pandas as pd
import rust_pyfunc as rp
import sys, time, json

sys.path.insert(0, "/home/chenzongwei/design_whatever")
import design_whatever as dw
sys.path.insert(0, "/home/chenzongwei/rust_pyfunc/sandbox_rankic_neu")
from prototype import load_all, read_factor_matrix
import dev_sandbox_rankic as ds
import dev_sandbox_engine_opt as dso

BARRA = "/home/chenzongwei/database/barra/barra_daily_together_jason.parquet"
BP = "/home/chenzongwei/pythoncode/_tail_v2_shared/backtest_inputs/000905_20170103_20260522_5438_6b8884a67f05d221/"
BT_START = 20170201
N_DAYS = 2276


def load():
    print("loading real data ...", flush=True)
    t0 = time.time()
    dates, stocks, ind, restrict, ret_sum1, ret_sum5, barra_raw = load_all(N_DAYS)
    ind = np.ascontiguousarray(ind)
    restrict = np.ascontiguousarray(restrict.astype(np.float32))
    ret_sum1 = np.ascontiguousarray(ret_sum1.astype(np.float32))
    ret_sum5 = np.ascontiguousarray(ret_sum5.astype(np.float32))
    ret_g1 = np.ascontiguousarray(np.load(BP + "ret_gap1.npy")[:N_DAYS].astype(np.float32))
    ret_g5 = np.ascontiguousarray(np.load(BP + "ret_gap5.npy")[:N_DAYS].astype(np.float32))
    index_ret = np.ascontiguousarray(np.load(BP + "index_ret.npy")[:N_DAYS].astype(np.float32))
    barra_ranked = np.asarray(ds.precompute_barra(barra_raw))  # (T,N,10) rank pct
    print(f"data ready in {time.time()-t0:.1f}s", flush=True)
    return dict(dates=dates, stocks=stocks, ind=ind, restrict=restrict,
                ret_sum1=ret_sum1, ret_sum5=ret_sum5, ret_g1=ret_g1, ret_g5=ret_g5,
                index_ret=index_ret, barra_ranked=barra_ranked, barra_raw=barra_raw)


def engine_slot0(raw):
    """复刻引擎 slot0: rank + 缺失中位秩填充。"""
    ranked = np.asarray(dso.bench_rank_py(raw)[1])
    med = (np.isfinite(raw).sum(axis=1) + 1.0) / 2.0
    # 引擎 fill_missing: 近 20 日活跃的缺口填中位秩 (其实现与下方 Python 等价, 用 Rust 版保持一致性)
    # 简化: 用 Rust 的 rank 输出 + 同语义填充 —— 此处直接调用 ds 的 rank? 引擎版本是 f32 平均秩。
    active = raw
    last_valid = -np.ones(raw.shape[1], dtype=int)
    for t in range(raw.shape[0]):
        valid_count = 0
        for s in range(raw.shape[1]):
            if np.isfinite(ranked[t, s]):
                valid_count += 1
            if np.isfinite(active[t, s]):
                last_valid[s] = t
        if valid_count == 0:
            continue
        med_rank = (valid_count + 1.0) / 2.0
        miss = (t - last_valid <= 20) & (last_valid >= 0)
        nan_ok = np.isnan(ranked[t]) & miss
        ranked[t, nan_ok] = med_rank
    return ranked.astype(np.float32)


def main():
    D = load()
    raw = read_factor_matrix(0, N_DAYS).astype(np.float32)
    slot0 = engine_slot0(raw)
    print("slot0 ready", slot0.shape, flush=True)

    out = {}

    # 1) rank
    ms, _ = dso.bench_rank_py(np.ascontiguousarray(raw))
    out["rank_ms"] = ms
    print(f"rank: {ms:.0f} ms", flush=True)

    # 2) rolling serial vs rowmajor
    ranked_c = np.ascontiguousarray(slot0)
    for w in (5, 10, 20):
        a, b, outs = dso.bench_rolling_py(ranked_c, w)
        m1, x1, n1, s1, m2, x2, n2, s2 = outs
        eq = all(np.array_equal(np.nan_to_num(np.asarray(p), nan=0.0), np.nan_to_num(np.asarray(q), nan=0.0))
                 for p, q in [(m1, m2), (x1, x2), (n1, n2), (s1, s2)])
        print(f"rolling w={w}: serial={a:.0f}ms rowmajor={b:.0f}ms eq={eq}", flush=True)
        out[f"rolling_w{w}"] = dict(serial_ms=a, rowmajor_ms=b, eq=eq)

    # 3) backtest baseline vs opt (slot0; ic_only=False 完整口径)
    ic_only = False
    pre_ms, base_ms, opt_ms, sums, icb, ico = dso.bench_backtest_py(
        np.ascontiguousarray(slot0), D["ret_g1"], D["ret_sum1"], D["ret_g5"], D["ret_sum5"],
        D["restrict"], D["index_ret"], D["dates"].astype(np.int32), BT_START, ic_only)
    sums = np.array(sums)
    b_sum, o_sum = sums[:20], sums[20:]
    both_idx = ~np.isnan(b_sum)
    max_diff = np.abs(b_sum[both_idx] - o_sum[both_idx]).max() if both_idx.any() else 0.0
    icb, ico = np.array(icb), np.array(ico)
    ic_match = np.array_equal(np.nan_to_num(icb, nan=0.0), np.nan_to_num(ico, nan=0.0))
    ic_maxdiff = np.abs(np.nan_to_num(icb, nan=0.0) - np.nan_to_num(ico, nan=0.0)).max()
    print(f"backtest(slot0 full): precompute={pre_ms:.0f}ms base={base_ms:.0f}ms opt={opt_ms:.0f}ms"
          f" | summary maxdiff={max_diff:.3e} ic_match={ic_match} ic_maxdiff={ic_maxdiff:.3e} n_ic={len(icb)}", flush=True)
    out["backtest_slot0"] = dict(pre_ms=pre_ms, base_ms=base_ms, opt_ms=opt_ms,
                                 max_diff=float(max_diff), ic_match=bool(ic_match),
                                 ic_maxdiff=float(ic_maxdiff), n_ic=len(icb))

    # 侧face: 一个 rolling slot (w=5 mean) 也验证
    m5 = np.asarray(outs[0]) if False else None

    # 4) neutralize baseline vs v2 (slot0)
    pre_neu_ms, orders = dso.pre_neu_py(D["ind"], D["restrict"], D["barra_ranked"])
    print(f"precompute_neu: {pre_neu_ms:.0f}ms", flush=True)
    out["pre_neu_ms"] = pre_neu_ms
    ms_b, out_b, fb_b, _ = dso.bench_neutralize_py(
        np.ascontiguousarray(slot0), D["ind"], D["restrict"], D["barra_ranked"], orders, "baseline")
    ms_v2, out_v2, fb_v2, _ = dso.bench_neutralize_py(
        np.ascontiguousarray(slot0), D["ind"], D["restrict"], D["barra_ranked"], orders, "v2")
    out_b, out_v2 = np.asarray(out_b), np.asarray(out_v2)
    both = np.isfinite(out_b) & np.isfinite(out_v2)
    mx = np.abs(out_b[both].astype(np.float64) - out_v2[both].astype(np.float64)).max() if both.any() else 0.0
    print(f"neutralize(slot0): baseline={ms_b:.0f}ms v2={ms_v2:.0f}ms maxdiff={mx:.3e} (finite={both.sum()})", flush=True)
    out["neutralize_slot0"] = dict(base_ms=ms_b, v2_ms=ms_v2, maxdiff=float(mx), fallback=fb_v2)

    # 5) 与 rankic 金标准对照 (确认 baseline 复刻 == 生产 neutralize; rankic 用原始 barra)
    ref, _ = ds.neutralize_full(np.ascontiguousarray(slot0), D["ind"], D["restrict"], D["barra_raw"], True)
    ref = np.asarray(ref)
    b = np.isfinite(ref) & np.isfinite(out_b)
    mx2 = np.abs(ref[b].astype(np.float64) - out_b[b].astype(np.float64)).max() if b.any() else 0.0
    print(f"baseline vs rankic neutralize_full(maxdiff): {mx2:.3e}", flush=True)
    out["baseline_vs_rankic_maxdiff"] = float(mx2)

    with open("/tmp/engine_opt_bench.json", "w") as f:
        json.dump(out, f, indent=2)
    print("saved /tmp/engine_opt_bench.json", flush=True)


if __name__ == "__main__":
    main()
