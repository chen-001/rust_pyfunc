"""Independent Python graph oracles and a small old/new kernel timing check."""
import heapq
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sandbox/relaynet_perf_20261102"))
import relaynet_perf_20261102 as engine


def reference(a, corrected):
    n = len(a)
    adj = [np.flatnonzero(np.isfinite(row)).tolist() for row in a]
    # Degeneracy ordering is independent of the production degree-bucket algorithm.
    left, degree, core, level = set(range(n)), list(map(len, adj)), [0] * n, 0
    while left:
        v = min(left, key=lambda j: (degree[j], j))
        level = max(level, degree[v])
        core[v] = level
        left.remove(v)
        for u in adj[v]:
            degree[u] -= u in left
    clustering = []
    for i, neighbors in enumerate(adj):
        total = 0.0
        for j in neighbors:
            for k in adj[j]:
                if k != i and np.isfinite(a[i, k]):
                    product = float(a[i, j]) * float(a[j, k]) * float(a[i, k])
                    if product > 0:
                        total += math.pow(product, 1 / 3)
        clustering.append(total / (len(neighbors) * (len(neighbors) - 1)) if len(neighbors) > 1 else 0.0)
    labels = list(range(n))
    for _ in range(100):
        changed = False
        for i, neighbors in enumerate(adj):
            if not neighbors:
                continue
            counts = {}
            for j in neighbors:
                label = labels[j]
                counts[label] = counts.get(label, 0.0) + float(a[i, j])
            best = min(counts, key=lambda label: (-counts[label], label))
            changed |= best != labels[i]
            labels[i] = best
        if not changed:
            break
    strength = [sum(float(a[i, j]) for j in adj[i]) for i in range(n)]
    pr = [1 / n] * n
    for _ in range(50):
        nex = [(1 - .85) / n if corrected else 0.] * n
        dangling = 0.
        for i, neighbors in enumerate(adj):
            if not corrected:
                nex[i] = (1 - .85) / n
            if strength[i] <= 0:
                dangling += pr[i]
            else:
                for j in neighbors:
                    nex[j] += .85 * pr[i] * float(a[i, j]) / strength[i]
        pr = [v + .85 * dangling / n for v in nex]
    return np.array([core, clustering, labels, pr])


def check():
    summary = engine.graph_probe()
    corrected = "corrected=true" in summary
    rng = np.random.default_rng(1937)
    if hasattr(engine, "binary_direction_state_probe"):
        cases = []
        for n in (3, 11, 31, 61):
            for density in (.07, .4, .9):
                edges = [(i, j, .5) for i in range(n) for j in range(i + 1, n)
                         if rng.random() < density]
                cases.append((n, edges, list(range(min(n, 7)))))
                shuffled = list(edges)
                rng.shuffle(shuffled)
                cases.append((n, shuffled, [0]))
        # Heterogeneous path counts exceed 2**53; shuffled IDs make queue order
        # different from parent ID order while all adjacency rows remain sorted.
        width, layers = 7, 45
        n = 1 + width * layers
        perm = rng.permutation(n)
        edges = [(0, 1 + j) for j in range(width)]
        for layer in range(1, layers):
            for j in range(width):
                selected = rng.choice(width, size=int(rng.integers(2, width + 1)), replace=False)
                edges += [(1 + (layer - 1) * width + int(k), 1 + layer * width + j) for k in selected]
        edges = sorted((*sorted((int(perm[i]), int(perm[j]))), .5) for i, j in edges)
        cases.append((n, edges, [int(perm[0]), int(perm[77])]))
        # Overflow is deliberate: sigma must become infinity at the same nodes.
        layers = 1030
        edges = [(0, 1, .5), (0, 2, .5)]
        for layer in range(1, layers):
            edges += [(1 + 2 * (layer - 1) + i, 1 + 2 * layer + j, .5)
                      for i in range(2) for j in range(2)]
        cases.append((1 + 2 * layers, edges, [0]))
        for n, edges, sources in cases:
            for src in sources:
                state_reference = engine.binary_direction_state_probe(n, edges, src, False, False)
                for force in (False, True):
                    fast = engine.binary_direction_state_probe(n, edges, src, True, force)
                    assert state_reference[0] == fast[0], "BFS queue order changed"
                    assert state_reference[2] == fast[2], "BFS distances changed"
                    np.testing.assert_equal(np.asarray(state_reference[1]).view(np.uint64),
                                            np.asarray(fast[1]).view(np.uint64))
        saved = {key: os.environ.get(key) for key in ("RELAYNET_BINARY_DIRECTION", "RELAYNET_BINARY_DIRECTION_FORCE",
                                                    "RELAYNET_BINARY_PREDECESSORS", "RELAYNET_BINARY_HOIST",
                                                    "RELAYNET_BINARY_PARENT_SIMD")}
        try:
            width, layers = 12, 18
            layered = [(i, j, .5) for layer in range(layers - 1)
                       for i in range(layer * width, (layer + 1) * width)
                       for j in range((layer + 1) * width, (layer + 2) * width)]
            for n, edges, _ in cases[:24] + [(width * layers, layered, [])]:
                os.environ.pop("RELAYNET_BINARY_DIRECTION", None)
                old = np.asarray(engine.binary_probe(n, edges, False))
                os.environ["RELAYNET_BINARY_DIRECTION"] = "1"
                os.environ["RELAYNET_BINARY_DIRECTION_FORCE"] = "1"
                for parents, hoist, simd in ((False, False, False), (True, False, False),
                                             (False, True, False), (True, True, False), (True, True, True)):
                    for key, enabled in (("RELAYNET_BINARY_PREDECESSORS", parents), ("RELAYNET_BINARY_HOIST", hoist),
                                         ("RELAYNET_BINARY_PARENT_SIMD", simd)):
                        if enabled:
                            os.environ[key] = "1"
                        else:
                            os.environ.pop(key, None)
                    fast = np.asarray(engine.binary_probe(n, edges, False))
                    np.testing.assert_equal(old.view(np.uint64), fast.view(np.uint64))
        finally:
            for key, value in saved.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
    if hasattr(engine, "directed_local_probe"):
        for n in (0, 1, 2, 11, 35, 70):
            for density in (.05, .4, .95):
                edges = [(i, j, float(np.float32(rng.uniform(.1, .9))))
                         for i in range(n) for j in range(n) if i != j and rng.random() < density]
                rng.shuffle(edges)
                old = np.asarray(engine.directed_local_probe(n, edges, True))
                fast = np.asarray(engine.directed_local_probe(n, edges, False))
                np.testing.assert_equal(fast.view(np.uint64), old.view(np.uint64))
                out, inc = [set() for _ in range(n)], [set() for _ in range(n)]
                for i, j, _ in edges:
                    out[i].add(j)
                    inc[j].add(i)
                expected = []
                for adj in (out, inc):
                    expected.append([len({i}.union(adj[i], *(adj[j] for j in adj[i]))) - 1
                                     for i in range(n)])
                np.testing.assert_equal(fast[:2], np.asarray(expected))
    if hasattr(engine, "binary_probe"):
        saved = os.environ.get("RELAYNET_BINARY_FULL_FRONTIER")
        cases = [(4, [(0, 1, .5), (0, 2, .5), (1, 3, .5), (2, 3, .5)])]
        for n in (3, 9, 24, 41):
            for density in (.12, .5, 1.):
                edges = [(i, j, .5) for i in range(n) for j in range(i + 1, n)
                         if i // ((n + 1) // 2) == j // ((n + 1) // 2) and rng.random() < density]
                cases.append((n, edges))
        try:
            for n, edges in cases:
                for simd in (False, True):
                    os.environ["RELAYNET_BINARY_FULL_FRONTIER"] = "1"
                    full = np.asarray(engine.binary_probe(n, edges, simd))
                    os.environ.pop("RELAYNET_BINARY_FULL_FRONTIER", None)
                    early = np.asarray(engine.binary_probe(n, edges, simd))
                    np.testing.assert_equal(early.view(np.uint64), full.view(np.uint64))
        finally:
            if saved is None:
                os.environ.pop("RELAYNET_BINARY_FULL_FRONTIER", None)
            else:
                os.environ["RELAYNET_BINARY_FULL_FRONTIER"] = saved
    for n in (1, 2, 9, 17, 25):
        raw = rng.uniform(.05, .99, (n, n)).astype(np.float32).astype(float)
        raw[rng.random((n, n)) < .35] = np.nan
        a = np.triu(raw, 1)
        a = a + a.T
        np.fill_diagonal(a, np.nan)
        actual = np.array(engine.graph_metrics_probe(a, False))
        np.testing.assert_allclose(actual, reference(a, corrected), rtol=1e-14, atol=1e-14)
        if corrected:
            np.testing.assert_allclose(actual[3].sum(), 1, rtol=1e-14, atol=1e-14)
    raw = rng.integers(-2, 7, (23, 23)).astype(float) / 10
    np.fill_diagonal(raw, np.nan)
    raw[0, 5:9] = np.nan
    for k in (1, 3, 10, 30):
        candidate = np.zeros(raw.shape, bool)
        for i, row in enumerate(raw):
            finite = [j for j in range(len(raw)) if j != i and np.isfinite(row[j])]
            if corrected:
                js = sorted(finite, key=lambda j: (-row[j], j))[:k]
            else:
                heap = []
                for j in finite:
                    if len(heap) < k:
                        heapq.heappush(heap, (-row[j], -j))
                    elif -heap[0][0] < row[j]:
                        heapq.heapreplace(heap, (-row[j], -j))
                js = [-item[1] for item in heap]
            candidate[i, js] = True
        for mutual in (False, True):
            keep = candidate & candidate.T if mutual else candidate | candidate.T
            expected = np.full(raw.shape, np.nan)
            for i in range(len(raw)):
                for j in range(i + 1, len(raw)):
                    if keep[i, j] and np.isfinite(raw[i, j]):
                        expected[i, j] = expected[j, i] = float(np.float32(raw[i, j]))
            np.testing.assert_equal(engine.graph_build_probe(raw, k, mutual), expected)
    if corrected:
        # Kruskal oracle: two disconnected nontrivial components plus one NaN isolate.
        n = 19
        s = np.full((n, n), np.nan)
        for start, stop in ((0, 9), (9, 18)):
            for i in range(start, stop):
                for j in range(i + 1, stop):
                    s[i, j] = s[j, i] = float(np.float32(rng.uniform(-.9, .9)))
        parent = list(range(n))
        def find(i):
            while parent[i] != i:
                i = parent[i]
            return i
        expected = np.full_like(s, np.nan)
        edges = sorted(((-s[i, j], i, j) for i in range(n) for j in range(i + 1, n)
                        if np.isfinite(s[i, j])))
        for neg_weight, i, j in edges:
            aroot, broot = find(i), find(j)
            if aroot != broot:
                parent[aroot] = broot
                expected[i, j] = expected[j, i] = -neg_weight
        np.testing.assert_equal(engine.graph_mst_probe(s), expected)
    n = 400
    raw = rng.uniform(.1, .9, (n, n)).astype(np.float32).astype(float)
    raw[rng.random((n, n)) > .25] = np.nan
    a = np.triu(raw, 1)
    a += a.T
    np.fill_diagonal(a, np.nan)
    timings = {}
    for old in (True, False):
        values = []
        for _ in range(3):
            start = time.perf_counter()
            engine.graph_metrics_probe(a, old)
            values.append(time.perf_counter() - start)
        timings["old" if old else "fast"] = float(np.median(values))
    print(json.dumps({"checks": summary, "python_oracles": "5 graphs + 8 knn", "seconds": timings,
                      "speedup": timings["old"] / timings["fast"]}))


if __name__ == "__main__":
    check()
