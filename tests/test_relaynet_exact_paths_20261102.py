"""Independent exhaustive-path and Floyd oracle for full exact graph statistics."""
import os
import argparse
import json
from pathlib import Path
import sys
import time

os.environ.setdefault('RAYON_NUM_THREADS', '50')
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'sandbox/relaynet_perf_20261102'))
import relaynet_perf_20261102 as engine


def oracle(n, edges):
    adj = [[] for _ in range(n)]
    distance = np.full((n, n), np.inf)
    np.fill_diagonal(distance, 0)
    for u, v, similarity in edges:
        cost = max(1 - float(np.float32(similarity)), 1e-9)
        adj[u].append((v, cost)); adj[v].append((u, cost))
        distance[u, v] = distance[v, u] = cost
    for k in range(n):
        distance = np.minimum(distance, distance[:, k, None] + distance[None, k, :])
    result = np.full((9, n), np.nan)
    result[7:] = 0
    for src in range(n):
        ds = distance[src, (np.arange(n) != src) & np.isfinite(distance[src])]
        if len(ds):
            result[:7, src] = [np.mean(1 / ds), ds.mean(), ds.max(), *np.quantile(ds, [.25, .75, .5, .9])]
        for target in range(src + 1, n):
            routes = []
            def walk(node, path, cost):
                if node == target:
                    routes.append((cost, path)); return
                for nxt, weight in adj[node]:
                    if nxt not in path:
                        walk(nxt, path + (nxt,), cost + weight)
            walk(src, (src,), 0.0)
            for row, measure in [(7, lambda route: route[0]), (8, lambda route: len(route[1]))]:
                if not routes: continue
                minimum = min(map(measure, routes))
                shortest = [path for route_cost, path in routes if measure((route_cost, path)) == minimum]
                for node in range(n):
                    result[row, node] += sum(node in path[1:-1] for path in shortest) / len(shortest)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph', type=Path)
    parser.add_argument('--experiments', action='store_true')
    parser.add_argument('--simd', action='store_true')
    args = parser.parse_args()
    args.experiments |= args.simd
    experiment_modes = (0, 16, 17, 32, 33, 64) if args.simd else (0, 1, 2, 3, 4, 5, 6, 7, 12, 13, 14, 15)
    cases = [(0, []), (1, []), (4, [(0, 1, .5), (1, 2, .5)]),
             (3, [(0, 1, 1.0), (1, 2, .5), (0, 2, .5)]),
             (4, [(0, 1, .5), (0, 2, .5), (1, 3, .5), (2, 3, .5)]),
             (5, [(0, 1, 1.0), (0, 2, 1.0), (1, 3, 1.0), (2, 3, 1.0)])]
    rng = np.random.default_rng(713)
    for n in range(2, 9):
        for repeat in range(5):
            edges = [(i, j, float(rng.choice([0., .125, .25, .5, .75, .875])))
                     for i in range(n) for j in range(i + 1, n) if rng.random() < .4]
            cases.append((n, edges))
    for n, edges in cases:
        actual = np.array(engine.exact_path_probe(n, edges)).reshape(9, n)
        expected = oracle(n, edges)
        np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13, equal_nan=True,
                                   err_msg=f'n={n}, edges={edges}')
        if args.experiments:
            for mode in experiment_modes:
                candidate = np.array(engine.exact_path_experiment_probe(n, edges, mode)).reshape(8, n)
                assert actual[:8].tobytes() == candidate.tobytes(), (mode, n, edges)
        for _ in range(3):
            repeated = np.array(engine.exact_path_probe(n, edges)).reshape(9, n)
            assert np.array_equal(actual, repeated, equal_nan=True)
    n = 120
    edges = [(i, i + 1, float(rng.uniform(.1, .9))) for i in range(n - 1)]
    result = np.array(engine.exact_path_probe(n, edges))
    counts = np.arange(n) * np.arange(n - 1, -1, -1)
    np.testing.assert_array_equal(result[7], counts)
    np.testing.assert_array_equal(result[8], counts)
    print(f'PASS {len(cases)} exhaustive all-pair graph cases + 120-node tree, repeated exact results')
    serial = np.array(engine.graph_batch_probe(1))
    saved_cache = os.environ.pop('RELAYNET_CANDIDATE_CACHE', None)
    try:
        for cached in (False, True):
            if cached: os.environ['RELAYNET_CANDIDATE_CACHE'] = '1'
            else: os.environ.pop('RELAYNET_CANDIDATE_CACHE', None)
            for batch in (1, 2, 4, 8):
                parallel = np.array(engine.graph_batch_probe(batch))
                assert serial.tobytes() == parallel.tobytes(), (cached, batch)
    finally:
        os.environ.pop('RELAYNET_CANDIDATE_CACHE', None)
        if saved_cache is not None: os.environ['RELAYNET_CANDIDATE_CACHE'] = saved_cache
    print('PASS batches 1/2/4/8 with candidate cache off/on: all columns bit-identical')
    if args.simd:
        for n in (17, 33):
            edges = [(i, j, float(rng.choice([.125, .5, .875, 1.])))
                     for i in range(n) for j in range(i + 1, n) if rng.random() < .8]
            reference = np.array(engine.exact_path_probe(n, edges))[:8]
            for mode in experiment_modes:
                candidate = np.array(engine.exact_path_experiment_probe(n, edges, mode))
                assert reference.tobytes() == candidate.tobytes(), (n, mode)
        print('PASS dense SIMD cases: ties, minimum-cost edges and partial vector tails')
    if args.graph:
        if args.experiments:
            reference = np.array(engine.weighted_graph_probe(str(args.graph), True))
            timings = {}
            modes = (-1,) + experiment_modes
            for mode in modes + tuple(reversed(modes)):
                started = time.perf_counter()
                actual = np.array(engine.weighted_graph_probe(str(args.graph), True)
                                  if mode == -1 else engine.weighted_graph_experiment_probe(str(args.graph), mode))
                elapsed = time.perf_counter() - started
                assert reference.tobytes() == actual.tobytes(), f'mode {mode} changed bits'
                timings.setdefault(mode, []).append(elapsed)
                print(json.dumps(dict(mode=mode, seconds=elapsed, bit_identical=True)), flush=True)
            suffix = '.weighted_simd.json' if args.simd else '.weighted_experiments.json'
            args.graph.with_suffix(suffix).write_text(json.dumps(timings, indent=2))
            return
        timings = {False: [], True: []}
        reference = None
        for parents in [False, True, True, False, False, True]:
            started = time.perf_counter()
            actual = np.array(engine.weighted_graph_probe(str(args.graph), parents))
            timings[parents].append(time.perf_counter() - started)
            if reference is None:
                reference = actual
            assert reference.tobytes() == actual.tobytes()
        result = dict(graph=str(args.graph), shape=reference.shape,
                      reverse_scan_seconds=timings[False], parent_array_seconds=timings[True],
                      bit_identical=True, threads=os.environ['RAYON_NUM_THREADS'])
        print(json.dumps(result))
        args.graph.with_suffix('.weighted_benchmark.json').write_text(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
