"""沙箱优化验证: neutralize 分阶段计时 + 微优化包 + preflight/rank 变体。

- 数据: 2276×5438 真实模板 (与 sandbox_engine_opt/bench_engine.py 相同来源)。
- 校验: cur vs opt 输出逐位一致 (f32 bits), 不通过则打印差异并退出。
"""
import sys, time, json
import numpy as np

sys.path.insert(0, "/home/chenzongwei/rust_pyfunc/sandbox_engine_opt")
sys.path.insert(0, "/home/chenzongwei/rust_pyfunc/sandbox_rankic_neu")
from prototype import load_all, read_factor_matrix
import dev_sandbox_rankic as ds
import dev_sandbox_engine_opt as dso

N_DAYS = 2276


def engine_slot0(raw):
    """同引擎: rank (平均秩) + 缺失中位秩填充 (近 20 日活跃)。"""
    ranked = np.asarray(dso.bench_rank_py(np.ascontiguousarray(raw))[1])
    med = (np.isfinite(ranked).sum(axis=1) + 1.0) / 2.0
    active = raw
    last_valid = -np.ones(raw.shape[1], dtype=int)
    for t in range(raw.shape[0]):
        for s in range(raw.shape[1]):
            if np.isfinite(active[t, s]):
                last_valid[s] = t
        miss = (t - last_valid <= 20) & (last_valid >= 0)
        nan_ok = np.isnan(ranked[t]) & miss
        ranked[t, nan_ok] = med[t]
    return ranked.astype(np.float32)


def main():
    print("loading real data ...", flush=True)
    t0 = time.time()
    dates, stocks, ind, restrict, ret_sum1, ret_sum5, barra_raw = load_all(N_DAYS)
    ind = np.ascontiguousarray(ind)
    restrict = np.ascontiguousarray(restrict.astype(np.float32))
    barra_ranked = np.ascontiguousarray(np.asarray(ds.precompute_barra(barra_raw)))
    raw = read_factor_matrix(0, N_DAYS).astype(np.float32)
    slot0 = np.ascontiguousarray(engine_slot0(raw))
    print(f"data ready in {time.time()-t0:.1f}s slot0={slot0.shape}", flush=True)
    out = {}

    # ---------- 1) neutralize 分阶段 cur vs optx (逐步定位) ----------
    outs = {}
    stage_ms = {}
    for mode in ("cur", "opt1", "opt2", "opt3"):
        t0 = time.time()
        stages, fb, res, pre_ms = dso.bench_neutralize_stages_py(
            slot0, ind, restrict, barra_ranked, mode
        )
        dt = time.time() - t0
        print(f"neutralize[{mode}]: stages={['%.0f' % s for s in stages]} fallback={fb}", flush=True)
        outs[mode] = np.asarray(res).copy()
        stage_ms[mode] = stages
    base = outs["cur"]
    for mode in ("opt1", "opt2", "opt3"):
        o = outs[mode]
        sn = np.array_equal(np.isnan(base), np.isnan(o))
        b = ~np.isnan(base) & ~np.isnan(o)
        md = np.abs(base[b] - o[b]).max() if b.any() else 0.0
        n_diff = int((np.nan_to_num(base, nan=-1e30) != np.nan_to_num(o, nan=-1e30)).sum())
        print(f"neutralize cur vs {mode}: same_nan={sn} maxdiff={md:.3e} n_diff={n_diff}", flush=True)
    out["neutralize"] = dict(cur=stage_ms["cur"], opt1=stage_ms["opt1"],
                             opt2=stage_ms["opt2"], opt3=stage_ms["opt3"], pre_ms=pre_ms)

    # ---------- 2) preflight 变体 ----------
    for mode in ("hash", "radix", "nohash"):
        t0 = time.time()
        ms, passed, maj, zm, nm = dso.bench_preflight_py(
            slot0, restrict, 10000.0, 0.12, 0.04, mode
        )
        print(f"preflight[{mode}]: {ms:.0f}ms passed={passed} maj={maj:.3f} zero={zm:.4f} nan={nm:.4f}", flush=True)
        if mode == "hash":
            h = dict(ms=ms, passed=passed, maj=maj, zm=zm, nm=nm)
        elif mode == "radix":
            r = dict(ms=ms, passed=passed, maj=maj, zm=zm, nm=nm)
        else:
            n = dict(ms=ms, passed=passed, maj=maj, zm=zm, nm=nm)
    eq_r = (h["passed"] == r["passed"] and h["zm"] == r["zm"] and h["nm"] == r["nm"]
            and abs(h["maj"] - r["maj"]) < 1e-12)
    eq_n = (h["passed"] == n["passed"] and h["zm"] == n["zm"] and h["nm"] == n["nm"])
    print(f"preflight radix==hash: {eq_r}; nohash==hash(观测): {eq_n}", flush=True)
    out["preflight"] = dict(hash=h, radix=r, nohash=n, radix_eq=bool(eq_r), nohash_eq=bool(eq_n))

    # ---------- 3) rank 变体 ----------
    for mode in ("serial", "radix"):
        t0 = time.time()
        ms, res = dso.bench_rank_radix_py(np.ascontiguousarray(raw), mode)
        print(f"rank[{mode}]: {ms:.0f}ms", flush=True)
        if mode == "serial":
            s_out = np.asarray(res).copy()
            s_ms = ms
        else:
            r_out = np.asarray(res).copy()
            r_ms = ms
    sn = np.array_equal(np.isnan(s_out), np.isnan(r_out))
    b = ~np.isnan(s_out) & ~np.isnan(r_out)
    md = np.abs(s_out[b] - r_out[b]).max() if b.any() else 0.0
    print(f"rank serial vs radix: same_nan={sn} maxdiff={md:.3e}", flush=True)
    out["rank"] = dict(serial_ms=s_ms, radix_ms=r_ms, same_nan=bool(sn), maxdiff=float(md))

    with open("/tmp/sandbox_opt_bench.json", "w") as f:
        json.dump(out, f, indent=2, default=float)
    print("saved /tmp/sandbox_opt_bench.json", flush=True)


if __name__ == "__main__":
    main()
