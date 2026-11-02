"""Exact neighborhood optimization; run once against the isolated fast source."""
from pathlib import Path
import sys

p = Path(__file__).parent / "src/relaynet_fast.rs"
s = p.read_text()
start = s.index("fn stat_undir(")
end = s.index('\n/// 到"市场"', start)
f = s[start:end]
if "// Exact reusable neighborhood bitsets." in f:
    raise SystemExit("stats patch already applied")
a = f.index("    let mut reach2 = vec![0.0; n];")
b = f.index("    let (cid, sizes) = components(g);", a)
f = f[:a] + '''    // Exact reusable neighborhood bitsets. No paths or nodes are sampled.
    let nw = (n + 63) / 64;
    let mut nbit = vec![vec![0u64; nw]; n];
    for i in 0..n {
        for &(v, _) in &g.adj[i] {
            nbit[i][v as usize / 64] |= 1u64 << (v as usize % 64);
        }
    }
    let balls: Vec<Vec<u64>> = (0..n).into_par_iter().map(|i| {
        let mut ball = nbit[i].clone();
        for &(v, _) in &g.adj[i] {
            for (b, &w) in ball.iter_mut().zip(&nbit[v as usize]) { *b |= w; }
        }
        ball
    }).collect();
    cols.push((0..n).map(|i| {
        if g.adj[i].is_empty() { 0.0 }
        else { balls[i].iter().map(|w| w.count_ones() as usize).sum::<usize>() as f64 - 1.0 }
    }).collect());

''' + f[b:]
a = f.index("    let nw = (n + 63) / 64;", f.index("// ---- 追加的 7"))
b = f.index("    let degu:", a)
f = f[:a] + '''    // A two-hop ball covering its component has exactly the component's edges.
    let mut component_edges = vec![0usize; sizes.len()];
    for i in 0..n { component_edges[cid[i] as usize] += g.adj[i].len(); }
    for e in &mut component_edges { *e /= 2; }
''' + f[b:]
f = f.replace("let kcr = kcore(g);", "let kcr: Vec<u32> = cols[6].iter().map(|&x| x as u32).collect();")
f = f.replace('''        .map(|i| {
            let mut o = [f64::NAN; 28];''', '''        .map_init(|| vec![0.0f32; n], |neighbor_weights, i| {
            let mut o = [f64::NAN; 28];''')
f = f.replace("            let di = a.len() as f64;", "            for &(v, w) in a { neighbor_weights[v as usize] = w; }\n            let di = a.len() as f64;")
a = f.index("            // reach3/local_density 用位图并集")
b = f.index("            // ---- hhi", a)
f = f[:a] + '''            // Exact bitset union/intersection replaces repeated edge scans.
            {
                let ball = &balls[i];
                let ball_n: usize = ball.iter().map(|w| w.count_ones() as usize).sum();
                let component = cid[i] as usize;
                let (cnt3, e_in) = if ball_n == sizes[component] as usize {
                    (ball_n, component_edges[component])
                } else {
                    let mut b3 = ball.clone();
                    let mut twice_edges = 0usize;
                    for (wi, &word) in ball.iter().enumerate() {
                        let mut bits = word;
                        while bits != 0 {
                            let j = wi * 64 + bits.trailing_zeros() as usize;
                            bits &= bits - 1;
                            for w in 0..nw {
                                b3[w] |= nbit[j][w];
                                twice_edges += (ball[w] & nbit[j][w]).count_ones() as usize;
                            }
                        }
                    }
                    (b3.iter().map(|w| w.count_ones() as usize).sum(), twice_edges / 2)
                };
                o[13] = (cnt3 as i64 - 1 - if a.iter().any(|&(v,_)| v as usize == i) {1} else {0}) as f64;
                if ball_n >= 2 { o[26] = e_in as f64 / (ball_n * (ball_n - 1) / 2) as f64; }
            }
''' + f[b:]
f = f.replace('''                        if let Some(&(_, ski)) = a.iter().find(|&&(x, _)| x == k) {
                            let prod = sij as f64 * sjk as f64 * ski as f64;
                            if prod > 0.0 { triw.push(prod.powf(1.0 / 3.0)); }
                        }''', '''                        let ski = neighbor_weights[ki];
                        let prod = sij as f64 * sjk as f64 * ski as f64;
                        if prod > 0.0 { triw.push(prod.powf(1.0 / 3.0)); }''')
assert "a.iter().find(|&&(x, _)| x == k)" not in f
if "--compat" not in sys.argv:
    f = f.replace("dm_arr[v as usize] + w as f64", "dm_arr[v as usize] + (1.0 - w as f64).max(1e-9)")
p.write_text(s[:start] + f + s[end:])
