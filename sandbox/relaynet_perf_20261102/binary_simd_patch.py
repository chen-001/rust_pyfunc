"""Exact lane-wise binary Brandes backprop; apply after exact_paths_patch.py.

Production selection remains scalar: the real-graph SIMD experiment was slower.
Both paths remain available through the binary_probe sandbox wrapper.
"""
from pathlib import Path
import argparse

p = argparse.ArgumentParser()
p.add_argument("--source", type=Path, default=Path(__file__).parent / "src/relaynet_fast.rs")
args = p.parse_args()
text = args.source.read_text()
start = text.index("fn betweenness_bin(")
brace = text.index("{", start)
depth = 1
end = brace + 1
while depth:
    depth += (text[end] == "{") - (text[end] == "}")
    end += 1
default = "false"
text = text[:start] + f'''fn betweenness_bin(g: &UGraph, _nsrc: usize) -> Vec<f64> {{
    binary_betweenness_run(g, {default})
}}''' + text[end:]
marker = "// BEGIN EXACT BINARY SIMD"
if marker in text:
    lo = text.index(marker)
    hi = text.index("// END EXACT BINARY SIMD", lo) + len("// END EXACT BINARY SIMD")
    text = text[:lo] + text[hi:]
text += r'''
// BEGIN EXACT BINARY SIMD
fn binary_betweenness_run(g: &UGraph, request_simd: bool) -> Vec<f64> {
    if let Some(bet) = forest_bet(g) { return bet; }
    let n = g.n;
    #[cfg(target_arch = "x86_64")]
    let use_simd = request_simd && std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("avx2");
    #[cfg(not(target_arch = "x86_64"))]
    let use_simd = false;
    // Preserve each original adjacency order, omitting unused edge weights.
    let mut offsets = Vec::with_capacity(n + 1);
    let mut neighbors = Vec::with_capacity(g.adj.iter().map(Vec::len).sum());
    offsets.push(0usize);
    for adj in &g.adj {
        neighbors.extend(adj.iter().map(|&(v, _)| v));
        offsets.push(neighbors.len());
    }
    const BLOCK: usize = 16;
    let parts: Vec<Vec<f64>> = (0..n.div_ceil(BLOCK)).into_par_iter().map(|block| {
        let mut dist = vec![-1i32; n]; let mut sigma = vec![0.0; n];
        let mut delta = vec![0.0; n]; let mut bet = vec![0.0; n];
        let mut queue = Vec::with_capacity(n);
        for src in block * BLOCK..((block + 1) * BLOCK).min(n) {
            if offsets[src] == offsets[src + 1] { continue; }
            dist.fill(-1); sigma.fill(0.0); delta.fill(0.0); queue.clear();
            dist[src] = 0; sigma[src] = 1.0; queue.push(src);
            let mut head = 0;
            while head < queue.len() {
                let u = queue[head]; head += 1;
                for &v in &neighbors[offsets[u]..offsets[u + 1]] {
                    let v = v as usize;
                    if dist[v] < 0 { dist[v] = dist[u] + 1; queue.push(v); }
                    if dist[v] == dist[u] + 1 { sigma[v] += sigma[u]; }
                }
            }
            for &u in queue.iter().rev() {
                let adj = &neighbors[offsets[u]..offsets[u + 1]];
                #[cfg(target_arch = "x86_64")]
                if use_simd && adj.len() >= 8 {
                    unsafe { binary_backprop_avx512(adj, &dist, &sigma, &mut delta, u); }
                    if u != src { bet[u] += delta[u]; }
                    continue;
                }
                binary_backprop_scalar(adj, &dist, &sigma, &mut delta, u);
                if u != src { bet[u] += delta[u]; }
            }
        }
        bet
    }).collect();
    let mut bet = vec![0.0; n];
    for part in parts { for (a, b) in bet.iter_mut().zip(part) { *a += b; } }
    for b in &mut bet { *b *= 0.5; }
    bet
}

#[inline]
fn binary_backprop_scalar(adj: &[u32], dist: &[i32], sigma: &[f64], delta: &mut [f64], u: usize) {
    for &v in adj {
        let v = v as usize;
        if dist[v] + 1 == dist[u] { delta[v] += (sigma[v] / sigma[u]) * (1.0 + delta[u]); }
    }
}

/// Graph builders supply unique neighbor IDs: eight lanes write eight distinct
/// delta values. Every lane uses separate divide, multiply and add instructions,
/// exactly preserving the scalar arithmetic and the reverse-BFS update order.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx2")]
unsafe fn binary_backprop_avx512(adj: &[u32], dist: &[i32], sigma: &[f64], delta: &mut [f64], u: usize) {
    use std::arch::x86_64::*;
    let du = _mm256_set1_epi32(dist[u] - 1);
    let su = _mm512_set1_pd(sigma[u]);
    let factor = _mm512_set1_pd(1.0 + delta[u]);
    let mut k = 0;
    while k + 8 <= adj.len() {
        let ids = _mm256_loadu_si256(adj.as_ptr().add(k) as *const __m256i);
        let dv = _mm256_i32gather_epi32::<4>(dist.as_ptr(), ids);
        let mask = _mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpeq_epi32(dv, du))) as u8;
        if mask != 0 {
            let sv = _mm512_mask_i32gather_pd::<8>(_mm512_setzero_pd(), mask, ids, sigma.as_ptr());
            let previous = _mm512_mask_i32gather_pd::<8>(_mm512_setzero_pd(), mask, ids, delta.as_ptr());
            let ratios = _mm512_div_pd(sv, su);
            let terms = _mm512_mul_pd(ratios, factor);
            let updated = _mm512_add_pd(previous, terms);
            _mm512_mask_i32scatter_pd::<8>(delta.as_mut_ptr(), mask, ids, updated);
        }
        k += 8;
    }
    binary_backprop_scalar(&adj[k..], dist, sigma, delta, u);
}

pub fn binary_probe(n: usize, edges: Vec<(usize, usize, f32)>, simd: bool) -> Vec<f64> {
    let mut g = UGraph { n, adj: vec![Vec::new(); n] };
    for (u, v, s) in edges { g.adj[u].push((v as u32, s)); g.adj[v].push((u as u32, s)); }
    binary_betweenness_run(&g, simd)
}

pub fn binary_graph_probe(path: &str, simd: bool) -> std::io::Result<Vec<f64>> {
    use std::io::Read;
    let mut f = std::io::BufReader::new(std::fs::File::open(path)?);
    let mut b = [0u8; 4]; f.read_exact(&mut b)?;
    let n = u32::from_le_bytes(b) as usize;
    let mut g = UGraph { n, adj: Vec::with_capacity(n) };
    for _ in 0..n {
        f.read_exact(&mut b)?; let m = u32::from_le_bytes(b) as usize;
        let mut adj = Vec::with_capacity(m);
        for _ in 0..m {
            f.read_exact(&mut b)?; let v = u32::from_le_bytes(b);
            f.read_exact(&mut b)?; let s = f32::from_le_bytes(b);
            adj.push((v, s));
        }
        g.adj.push(adj);
    }
    Ok(binary_betweenness_run(&g, simd))
}
// END EXACT BINARY SIMD
'''
args.source.write_text(text)
lib = args.source.parent / "lib.rs"
lib_text = lib.read_text()
if "fn binary_probe(" not in lib_text:
    lib_text += r'''

#[pyfunction]
fn binary_probe(py: Python, n: usize, edges: Vec<(usize, usize, f32)>, simd: bool) -> Vec<f64> {
    py.allow_threads(|| relaynet_fast::binary_probe(n, edges, simd))
}

#[pyfunction]
fn binary_graph_probe(py: Python, path: &str, simd: bool) -> PyResult<Vec<f64>> {
    py.allow_threads(|| relaynet_fast::binary_graph_probe(path, simd))
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
}
'''
    anchor = "    m.add_function(wrap_pyfunction!(run, m)?)?;"
    if anchor not in lib_text: raise RuntimeError("module registration anchor missing")
    lib_text = lib_text.replace(anchor, anchor + "\n    m.add_function(wrap_pyfunction!(binary_probe, m)?)?;\n    m.add_function(wrap_pyfunction!(binary_graph_probe, m)?)?;", 1)
    lib.write_text(lib_text)
print(f"binary SIMD experiment retained: {args.source}; production always scalar")
