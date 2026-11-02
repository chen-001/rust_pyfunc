"""Fuse directed distances only when path reversal is exact in float64 arithmetic.

Apply after graph_patch.py, which restores the two original harmonic calls.
"""
from pathlib import Path
import argparse

p = argparse.ArgumentParser()
p.add_argument("--source", type=Path, default=Path(__file__).parent / "src/relaynet_fast.rs")
args = p.parse_args()
text = args.source.read_text()
old = "    let outh = dir_harmonic(g, false);\n    let inh = dir_harmonic(g, true);"
new = "    let (outh, inh) = dir_harmonic_pair_exact(g);"
if old not in text and new not in text:
    raise RuntimeError("directed harmonic call pair not found")
text = text.replace(old, new)
marker = "// BEGIN EXACT DIRECTED PAIR"
if marker in text:
    lo = text.index(marker)
    hi = text.index("// END EXACT DIRECTED PAIR", lo) + len("// END EXACT DIRECTED PAIR")
    text = text[:lo] + text[hi:]
text += r'''
// BEGIN EXACT DIRECTED PAIR
/// All positive edge costs are integer multiples of 2^-32, and even a
/// tentative n-edge path stays below 2^53 scaled units. Every addition is
/// therefore exact in f64; reversing a path preserves its exact bit pattern.
/// The 1e-9 cost floor fails this test and retains the two original passes.
fn dir_costs_are_reversible_exact(g: &DGraph) -> bool {
    const SCALE: f64 = 4294967296.0;
    const LIMIT: f64 = 9007199254740992.0;
    let mut max_scaled = 0.0f64;
    for adj in &g.out_adj {
        for &(_, sim) in adj {
            let scaled = (1.0 - sim as f64).max(1e-9) * SCALE;
            if !scaled.is_finite() || scaled <= 0.0 || scaled.fract() != 0.0 { return false; }
            max_scaled = max_scaled.max(scaled);
        }
    }
    max_scaled * (g.n as f64) < LIMIT
}

/// One flat all-pairs array avoids duplicate distance rows. Incoming sums
/// scan source nodes in their original ascending order, in cache-sized tiles.
fn dir_harmonic_pair_exact(g: &DGraph) -> (Vec<f64>, Vec<f64>) {
    let n = g.n;
    if n == 0 { return (Vec::new(), Vec::new()); }
    if g.out_adj.iter().all(Vec::is_empty) { return (vec![f64::NAN; n], vec![f64::NAN; n]); }
    if !dir_costs_are_reversible_exact(g) {
        return (dir_harmonic(g, false), dir_harmonic(g, true));
    }
    let mut distances = vec![f64::INFINITY; n * n];
    let mut outgoing = vec![f64::NAN; n];
    distances.par_chunks_mut(n).zip(outgoing.par_iter_mut()).enumerate().for_each(|(s, (d, out))| {
        d[s] = 0.0;
        if g.out_adj[s].is_empty() { return; }
        let mut heap = BinaryHeap::new();
        heap.push(State { d: 0.0, u: s });
        while let Some(State { d: du, u }) = heap.pop() {
            if du != d[u] { continue; }
            for &(v, sim) in &g.out_adj[u] {
                let w = (1.0 - sim as f64).max(1e-9);
                let candidate = du + w;
                if candidate < d[v as usize] {
                    d[v as usize] = candidate;
                    heap.push(State { d: candidate, u: v as usize });
                }
            }
        }
        let (mut h, mut count) = (0.0, 0usize);
        for (j, &x) in d.iter().enumerate() {
            if j != s && x.is_finite() && x > 0.0 { h += 1.0 / x; count += 1; }
        }
        if count > 0 { *out = h / count as f64; }
    });
    let mut incoming = vec![f64::NAN; n];
    const TILE: usize = 32;
    incoming.par_chunks_mut(TILE).enumerate().for_each(|(block, target)| {
        let lo = block * TILE;
        let mut sums = [0.0f64; TILE];
        let mut counts = [0usize; TILE];
        for source in 0..n {
            let row = &distances[source * n + lo..source * n + lo + target.len()];
            for (lane, &x) in row.iter().enumerate() {
                if source != lo + lane && x.is_finite() && x > 0.0 {
                    sums[lane] += 1.0 / x;
                    counts[lane] += 1;
                }
            }
        }
        for lane in 0..target.len() {
            if counts[lane] > 0 { target[lane] = sums[lane] / counts[lane] as f64; }
        }
    });
    (outgoing, incoming)
}

pub fn directed_probe(n: usize, edges: Vec<(usize, usize, f32)>, old: bool) -> (bool, Vec<Vec<f64>>) {
    let mut g = DGraph { n, out_adj: vec![Vec::new(); n], in_adj: vec![Vec::new(); n] };
    for (u, v, s) in edges { g.out_adj[u].push((v as u32, s)); g.in_adj[v].push((u as u32, s)); }
    let safe = dir_costs_are_reversible_exact(&g);
    let (outgoing, incoming) = if old {
        (dir_harmonic(&g, false), dir_harmonic(&g, true))
    } else {
        dir_harmonic_pair_exact(&g)
    };
    (safe, vec![outgoing, incoming])
}
// END EXACT DIRECTED PAIR
'''
args.source.write_text(text)

lib = args.source.parent / "lib.rs"
lib_text = lib.read_text()
if "fn directed_probe(" not in lib_text:
    lib_text += r'''

#[pyfunction]
fn directed_probe(py: Python, n: usize, edges: Vec<(usize, usize, f32)>, old: bool) -> (bool, Vec<Vec<f64>>) {
    py.allow_threads(|| relaynet_fast::directed_probe(n, edges, old))
}
'''
    lib_text = lib_text.replace("#[pymodule]", "#[pymodule]", 1)
    anchor = "    m.add_function(wrap_pyfunction!(run, m)?)?;"
    if anchor not in lib_text:
        raise RuntimeError("module registration anchor not found")
    lib_text = lib_text.replace(anchor, anchor + "\n    m.add_function(wrap_pyfunction!(directed_probe, m)?)?;", 1)
    lib.write_text(lib_text)
print(f"exact directed pair patch applied: {args.source}")
