"""Optional exact reuse: triangle quantile values are a subset of clustering work."""
from pathlib import Path

p = Path(__file__).parent / "src/relaynet_fast.rs"
s = p.read_text()
if "fn clustering_triangle_pair_exact(" in s:
    raise SystemExit("triangle fusion patch already applied")
old = "    cols.push(clustering_w(g));"
assert old in s
s = s.replace(old, '''    let fuse_triangle = std::env::var_os("RELAYNET_TRI_FUSE").is_some();
    let (clustering, triangle_p75_cache) = if fuse_triangle {
        clustering_triangle_pair_exact(g, select_triangle)
    } else { (clustering_w(g), Vec::new()) };
    cols.push(clustering);''', 1)
old = '''                // 三角形强度（正积才计入，负边三角形跳过）
                for &(k, sjk) in &g.adj[vi] {
                    let ki = k as usize;
                    if ki == i { continue; }
                    if (nbit[i][ki / 64] >> (ki % 64)) & 1 == 1 && ki > vi {
                        let ski = neighbor_weights[ki];
                        let prod = sij as f64 * sjk as f64 * ski as f64;
                        if prod > 0.0 { triw.push(prod.powf(1.0 / 3.0)); }
                    }
                }'''
assert old in s
s = s.replace(old, "                if !fuse_triangle {\n" + old + "\n                }", 1)
start = s.index("// tri_p75：")
lo = s.index("            if !triw.is_empty() {", start)
s = s[:lo] + s[lo:].replace("            if !triw.is_empty() {", "            if fuse_triangle { o[19] = triangle_p75_cache[i]; }\n            else if !triw.is_empty() {", 1)
s += r'''
// BEGIN EXACT TRIANGLE FUSION
fn clustering_triangle_pair_exact(g: &UGraph, select: bool) -> (Vec<f64>, Vec<f64>) {
    let n = g.n;
    let rows: Vec<(f64, f64)> = (0..n).into_par_iter().map_init(
        || (vec![0.0f32; n], vec![usize::MAX; n], Vec::<f64>::new()),
        |(weights, stamps, triw), i| {
            triw.clear();
            let di = g.deg(i);
            if di < 2 { return (0.0, f64::NAN); }
            for &(v, weight) in &g.adj[i] {
                weights[v as usize] = weight;
                stamps[v as usize] = i;
            }
            let mut acc = 0.0;
            // This is the original clustering loop, in its original order and
            // with its original product association. Quantile values are only
            // retained when k>j, exactly the original tri_p75 subset.
            for &(j, sij) in &g.adj[i] {
                for &(k, sjk) in &g.adj[j as usize] {
                    if k as usize != i && stamps[k as usize] == i {
                        let ski = weights[k as usize];
                        let prod = sij as f64 * sjk as f64 * ski as f64;
                        if prod > 0.0 {
                            let value = prod.powf(1.0 / 3.0);
                            acc += value;
                            if k > j { triw.push(value); }
                        }
                    }
                }
            }
            let p75 = if triw.is_empty() { f64::NAN }
            else if select { triangle_p75_select_exact(triw) }
            else {
                triw.sort_by(|x, y| x.total_cmp(y));
                let pos = 0.75 * (triw.len() - 1) as f64;
                let lo = pos.floor() as usize;
                let hi = (lo + 1).min(triw.len() - 1);
                triw[lo] + (triw[hi] - triw[lo]) * (pos - lo as f64)
            };
            (acc / (di as f64 * (di - 1) as f64), p75)
        },
    ).collect();
    rows.into_iter().unzip()
}

pub fn small_stats_graph_probe(path: &str) -> std::io::Result<Vec<Vec<f64>>> {
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
            f.read_exact(&mut b)?; let w = f32::from_le_bytes(b);
            adj.push((v, w));
        }
        g.adj.push(adj);
    }
    let sim = Array2::<f64>::zeros((n, n));
    let inds: Vec<i64> = (0..n).map(|i| (i % 3 + 1) as i64).collect();
    let mkt: Vec<f64> = (0..n).map(|i| i as f64 / n.max(1) as f64).collect();
    let leaders: Vec<bool> = (0..n).map(|i| i % 7 == 0).collect();
    Ok(stat_undir(&g, &sim, &inds, &mkt, &leaders, true))
}
// END EXACT TRIANGLE FUSION
'''
p.write_text(s)
lib = p.parent / "lib.rs"
t = lib.read_text()
if "fn small_stats_graph_probe(" not in t:
    t += r'''

#[pyfunction]
fn small_stats_graph_probe(py: Python, path: &str) -> PyResult<Vec<Vec<f64>>> {
    py.allow_threads(|| relaynet_fast::small_stats_graph_probe(path))
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
}
'''
    anchor = "    m.add_function(wrap_pyfunction!(run, m)?)?;"
    assert anchor in t
    t = t.replace(anchor, anchor + "\n    m.add_function(wrap_pyfunction!(small_stats_graph_probe, m)?)?;", 1)
    lib.write_text(t)
print("Applied optional exact clustering/triangle fusion and full-graph stat probe")
