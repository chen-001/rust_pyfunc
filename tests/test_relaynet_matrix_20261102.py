"""Exact matrix kernels: frozen baseline bits, independent formula, 50-thread speed."""
import os
os.environ.setdefault("RAYON_NUM_THREADS", "50")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import sys
from pathlib import Path
from time import perf_counter
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sandbox/relaynet_perf_20261102"))
import relaynet_perf_20261102 as engine


def oracle(x, kind):
    n = x.shape[1]
    out = np.full((n, n), np.nan)
    cols = x - x.mean(axis=0)
    sd = np.sqrt((cols * cols).sum(axis=0))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if kind == 1:
                both = (x[:, i] != 0) & (x[:, j] != 0)
                if both.sum() >= 10:
                    out[i, j] = np.mean((x[both, i] > 0) == (x[both, j] > 0))
            elif sd[i] > 0 and sd[j] > 0:
                if kind == 0:
                    num = np.sum(cols[:, i] * cols[:, j])
                else:
                    num = np.sum(cols[:-1, i] * cols[1:, j]) - np.sum(cols[1:, i] * cols[:-1, j])
                out[i, j] = num / (sd[i] * sd[j])
    return out


def test_kernels():
    rng = np.random.default_rng(7123)
    for shape in [(474, 137), (60, 257), (1, 9), (0, 5)]:
        x = rng.normal(size=shape)
        x[rng.random(shape) < .2] = 0
        x[:, 0] = 0
        x[:, 1] = 2
        x[:, 3] = x[:, 2]
        for kind in range(3):
            old = engine.matrix_probe(x, kind, "baseline")
            new = engine.matrix_probe(x, kind, "fast")
            assert np.array_equal(np.isnan(old), np.isnan(new)), (shape, kind, "NaN")
            good = ~np.isnan(old)
            assert np.array_equal(old[good].view(np.uint64), new[good].view(np.uint64)), (shape, kind, "bits")
            if shape[0] > 1:
                np.testing.assert_allclose(new[:8, :8], oracle(x[:, :8], kind), atol=3e-15, rtol=3e-14, equal_nan=True)
    x = rng.normal(size=(80, 17))
    x[:10, 4] = np.nan
    x[5, 7] = np.inf
    for kind in range(3):
        old = engine.matrix_probe(x, kind, "baseline")
        new = engine.matrix_probe(x, kind, "fast")
        np.testing.assert_array_equal(old, new)
    print("PASS: baseline finite bits and NaN masks match; independent Python formulas match")


def test_prep():
    # These private series have no registered factor or summary consumer.
    removed = {8, 15, 18, 23, 29, 31, 32, 33, 34, 40}
    keep = np.ones(41 * 474 + 4, dtype=bool)
    for k in removed:
        keep[k * 474:(k + 1) * 474] = False
    times = np.zeros(2)
    for date in [20220819, 20241231]:
        for code in ["000001", "600519", "300750"]:
            pair = []
            for i, variant in enumerate(["baseline", "fast"]):
                start = perf_counter()
                pair.append(engine.prep_probe(code, date, variant))
                times[i] += perf_counter() - start
            old, new = (v[keep] for v in pair)
            assert np.array_equal(np.isnan(old), np.isnan(new)), (code, date, "NaN")
            good = ~np.isnan(old)
            assert np.array_equal(old[good].view(np.uint64), new[good].view(np.uint64)), (code, date, "bits")
    print(f"PASS: six real stock/days prep used-series bits match; including IO baseline={times[0]:.3f}s fast={times[1]:.3f}s")


def test_directed():
    rng = np.random.default_rng(20051102)
    cases = [(0, [], True), (1, [], True)]
    for n in [2, 9, 31, 89]:
        for density in [.08, .5]:
            edges = [(u, v, float(np.float32(rng.uniform(.05, .95))))
                     for u in range(n) for v in range(n)
                     if u != v and rng.random() < density]
            cases.append((n, edges, True))
            cases.append((n, edges + [(0, 1, 1.0)], False))
    cases += [
        (4, [(0, 1, .5), (0, 2, .5), (1, 3, .5), (2, 3, .5)], True),
        (4, [(0, 1, 2.0**-40), (1, 2, .7)], False),
        (4, [(0, 1, -2.0**24), (1, 2, .7)], False),
    ]
    for n, edges, expected_safe in cases:
        safe, new = engine.directed_probe(n, edges, False)
        _, old = engine.directed_probe(n, edges, True)
        assert safe == expected_safe, (n, "branch")
        old, new = np.asarray(old), np.asarray(new)
        assert np.array_equal(np.isnan(old), np.isnan(new)), (n, "NaN")
        good = ~np.isnan(old)
        assert np.array_equal(old[good].view(np.uint64), new[good].view(np.uint64)), (n, "bits")
    print(f"PASS: {len(cases)} directed graphs, both exact fusion and fallback match original finite bits/NaN masks")


def test_binary_simd():
    rng = np.random.default_rng(10591)
    cases = [(0, []), (1, []), (4, [(0, 1, .5), (1, 2, .5), (2, 3, .5)])]
    for n in [9, 17, 31, 89, 127]:
        for density in [.08, .5, .95]:
            edges = [(u, v, .5) for u in range(n) for v in range(u + 1, n) if rng.random() < density]
            rng.shuffle(edges)
            cases.append((n, edges))
    for n, edges in cases:
        old = np.asarray(engine.binary_probe(n, edges, False))
        new = np.asarray(engine.binary_probe(n, edges, True))
        assert old.tobytes() == new.tobytes(), (n, "binary SIMD bits")
    print(f"PASS: {len(cases)} all-source binary Brandes graphs scalar/SIMD exact bits")


def test_small_stats():
    rng = np.random.default_rng(55012)
    flags = ["RELAYNET_BASIC_PARALLEL", "RELAYNET_TRI_SELECT", "RELAYNET_TRI_FUSE"]
    cases = [(1, []), (5, [(0, 1, .5), (1, 2, .7)]),
             (4, [(0, 1, -.25), (1, 2, -.5), (0, 2, .3), (2, 3, 0.0)])]
    for n in [9, 31, 89]:
        for density in [.1, .7]:
            edges = [(u, v, float(rng.choice([.125, .25, .5, .9])))
                     for u in range(n) for v in range(u + 1, n) if rng.random() < density]
            cases.append((n, edges))
    for n, edges in cases:
        reference = None
        for enabled in [(False, False, False), (True, False, False), (False, True, False),
                        (True, True, False), (False, False, True), (True, False, True),
                        (False, True, True), (True, True, True)]:
            for flag, active in zip(flags, enabled):
                if active:
                    os.environ[flag] = "1"
                else:
                    os.environ.pop(flag, None)
            actual = np.asarray(engine.stat_probe(n, edges, "fast"))
            if reference is None:
                reference = actual
            assert np.array_equal(np.isnan(reference), np.isnan(actual)), (n, enabled, "NaN")
            good = ~np.isnan(reference)
            assert np.array_equal(reference[good].view(np.uint64), actual[good].view(np.uint64)), (n, enabled, "bits")
    for flag in flags:
        os.environ.pop(flag, None)
    print(f"PASS: {len(cases)} graphs × eight basic/triangle variants, all full-stat finite bits/NaN masks match")


if __name__ == "__main__":
    test_kernels()
    test_prep()
    test_directed()
    test_binary_simd()
    test_small_stats()
    rng = np.random.default_rng(7123)
    x = rng.normal(size=(474, 2000))
    for kind in range(3):
        timings = []
        for variant in ["baseline", "fast"]:
            start = perf_counter()
            engine.matrix_probe(x, kind, variant)
            timings.append(perf_counter() - start)
        print(f"kind={kind}: baseline={timings[0]:.3f}s fast={timings[1]:.3f}s speedup={timings[0] / timings[1]:.2f}x")
