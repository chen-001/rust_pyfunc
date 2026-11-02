"""Apply exact graph kernels to the isolated copy; --compat preserves legacy bugs/sources."""
import argparse
from pathlib import Path

HERE = Path(__file__).resolve().parent


def function_span(text, name):
    start = text.index(f"fn {name}(")
    if text[max(0, start - 4):start] == "pub ":
        start -= 4
    brace = text.index("{", start)
    depth = 1
    end = brace + 1
    while depth:
        depth += (text[end] == "{") - (text[end] == "}")
        end += 1
    return start, end


def replace_function(text, name, replacement):
    start, end = function_span(text, name)
    return text[:start] + replacement.strip() + text[end:]


KCORE = r'''
fn kcore(g: &UGraph) -> Vec<u32> {
    // Degree buckets: every edge is visited once, with no repeated market-wide scan.
    let n = g.n;
    let mut degree: Vec<usize> = g.adj.iter().map(Vec::len).collect();
    let max_degree = degree.iter().copied().max().unwrap_or(0);
    let mut bins = vec![0usize; max_degree + 1];
    for &d in &degree { bins[d] += 1; }
    let mut start = 0;
    for count in &mut bins {
        let old = *count;
        *count = start;
        start += old;
    }
    let mut positions = vec![0usize; n];
    let mut vertices = vec![0usize; n];
    for v in 0..n {
        positions[v] = bins[degree[v]];
        vertices[positions[v]] = v;
        bins[degree[v]] += 1;
    }
    for d in (1..=max_degree).rev() { bins[d] = bins[d - 1]; }
    bins[0] = 0;
    for i in 0..n {
        let v = vertices[i];
        for &(u, _) in &g.adj[v] {
            let u = u as usize;
            if degree[u] > degree[v] {
                let d = degree[u];
                let pu = positions[u];
                let pw = bins[d];
                let w = vertices[pw];
                vertices.swap(pu, pw);
                positions[u] = pw;
                positions[w] = pu;
                bins[d] += 1;
                degree[u] -= 1;
            }
        }
    }
    degree.into_iter().map(|v| v as u32).collect()
}
'''

CLUSTERING = r'''
fn clustering_w(g: &UGraph) -> Vec<f64> {
    let n = g.n;
    (0..n).into_par_iter().map_init(
        || (vec![0.0f32; n], vec![usize::MAX; n]),
        |(weights, stamps), i| {
            let di = g.deg(i);
            if di < 2 { return 0.0; }
            for &(v, weight) in &g.adj[i] {
                weights[v as usize] = weight;
                stamps[v as usize] = i;
            }
            let mut acc = 0.0;
            // Keep the legacy loop, product, powf and addition order exactly.
            for &(j, sij) in &g.adj[i] {
                for &(k, sjk) in &g.adj[j as usize] {
                    if k as usize != i && stamps[k as usize] == i {
                        let ski = weights[k as usize];
                        let prod = sij as f64 * sjk as f64 * ski as f64;
                        if prod > 0.0 { acc += prod.powf(1.0 / 3.0); }
                    }
                }
            }
            acc / (di as f64 * (di - 1) as f64)
        },
    ).collect()
}
'''

COMMUNITIES = r'''
fn communities(g: &UGraph) -> Vec<u32> {
    let n = g.n;
    let mut lab: Vec<u32> = (0..n as u32).collect();
    let mut weights = vec![0.0f64; n];
    let mut marks = vec![false; n];
    let mut touched = Vec::with_capacity(n);
    for _ in 0..100 {
        let mut changed = false;
        for i in 0..n {
            if g.adj[i].is_empty() { continue; }
            touched.clear();
            for &(v, s) in &g.adj[i] {
                let l = lab[v as usize] as usize;
                if !marks[l] {
                    weights[l] = 0.0;
                    marks[l] = true;
                    touched.push(l);
                }
                weights[l] += s as f64;
            }
            let mut best = touched[0];
            for &l in &touched {
                let cmp = weights[l].partial_cmp(&weights[best]).unwrap_or(Ordering::Equal);
                if cmp == Ordering::Greater || (cmp == Ordering::Equal && l < best) {
                    best = l;
                }
                marks[l] = false;
            }
            if lab[i] != best as u32 { lab[i] = best as u32; changed = true; }
        }
        if !changed { break; }
    }
    lab
}
'''

PAGERANK = r'''
fn pagerank(g: &UGraph, d: f64, iters: usize) -> Vec<f64> {
    if !RELAY_GRAPH_CORRECT_FIXES { return relay_old_pagerank(g,d,iters); }
    let n = g.n;
    let strength: Vec<f64> = g.adj.iter().map(|a| a.iter().map(|&(_,s)| s as f64).sum()).collect();
    // The corrected push algorithm adds contributions in source-index order.
    // A sorted incoming row gives that same floating-point order independently per target.
    let incoming: Vec<Vec<(u32,f32)>> = g.adj.iter().map(|a| {
        let mut row = a.clone(); row.sort_unstable_by_key(|&(v,_)| v); row
    }).collect();
    let mut pr = vec![1.0 / n as f64;n];
    let mut next = vec![0.0;n];
    for _ in 0..iters {
        let mut dangling = 0.0;
        for i in 0..n { if strength[i] <= 0.0 { dangling += pr[i]; } }
        next.par_iter_mut().enumerate().for_each(|(i,out)| {
            let mut value = (1.0-d) / n as f64;
            for &(src,weight) in &incoming[i] {
                let src = src as usize;
                if strength[src] <= 0.0 { continue; }
                value += d * pr[src] * weight as f64 / strength[src];
            }
            *out = value + d * dangling / n as f64;
        });
        std::mem::swap(&mut pr,&mut next);
    }
    pr
}
'''

EIGENVEC = r'''
fn eigenvec(g: &UGraph, iters: usize) -> Vec<f64> {
    let n = g.n;
    let mut x = vec![1.0 / (n as f64).sqrt();n];
    let mut y = vec![0.0;n];
    for _ in 0..iters {
        y.par_iter_mut().enumerate().for_each(|(i,out)| {
            *out = g.adj[i].iter().map(|&(v,s)| s as f64 * x[v as usize]).sum();
        });
        // Keep the normalization sum in its original index order.
        let norm = y.iter().map(|v| v*v).sum::<f64>().sqrt();
        if norm <= 0.0 { break; }
        x.par_iter_mut().zip(y.par_iter()).for_each(|(out,&value)| *out = value / norm);
    }
    x
}
'''

BUILDERS = r'''
fn relay_candidate_bits(s: &Array2<f64>, k: usize, allow: Option<&[bool]>) -> Vec<Vec<u64>> {
    let n = s.nrows();
    let nw = (n + 63) / 64;
    (0..n).into_par_iter().map(|i| {
        let mut bits = vec![0u64; nw];
        if k == 0 || allow.map_or(false, |a| !a[i]) { return bits; }
        if RELAY_GRAPH_CORRECT_FIXES {
            // The worst retained candidate is minimum similarity, then largest code index.
            let mut heap: BinaryHeap<(std::cmp::Reverse<OrderedF>, u32)> = BinaryHeap::with_capacity(k);
            for j in 0..n {
                if j == i || allow.map_or(false, |a| !a[j]) { continue; }
                let v = s[[i,j]];
                if !v.is_finite() { continue; }
                if heap.len() < k {
                    heap.push((std::cmp::Reverse(OrderedF(v)), j as u32));
                } else {
                    let worst = heap.peek().unwrap();
                    if v > worst.0.0.0 || (v == worst.0.0.0 && (j as u32) < worst.1) {
                        heap.pop();
                        heap.push((std::cmp::Reverse(OrderedF(v)), j as u32));
                    }
                }
            }
            for (_, j) in heap { bits[j as usize / 64] |= 1u64 << (j as usize % 64); }
        } else {
            // Compatibility branch reproduces the existing maximum-heap bug exactly.
            let mut heap: BinaryHeap<(OrderedF, u32)> = BinaryHeap::with_capacity(k);
            for j in 0..n {
                if j == i || allow.map_or(false, |a| !a[j]) { continue; }
                let v = s[[i,j]];
                if !v.is_finite() { continue; }
                if heap.len() < k { heap.push((OrderedF(v), j as u32)); }
                else if heap.peek().unwrap().0.0 < v {
                    heap.pop(); heap.push((OrderedF(v), j as u32));
                }
            }
            for (_, j) in heap { bits[j as usize / 64] |= 1u64 << (j as usize % 64); }
        }
        bits
    }).collect()
}

fn relay_graph_from_bits(s: &Array2<f64>, bits: &[Vec<u64>], mutual: bool, threshold: Option<f64>) -> UGraph {
    let n = s.nrows();
    let nw = (n + 63) / 64;
    // Transpose only retained candidates; the inverse rows then have contiguous reads.
    let mut reverse = vec![vec![0u64;nw];n];
    for i in 0..n {
        for (word,&raw) in bits[i].iter().enumerate() {
            let mut remaining = raw;
            while remaining != 0 {
                let j = word * 64 + remaining.trailing_zeros() as usize;
                reverse[j][i / 64] |= 1u64 << (i % 64);
                remaining &= remaining - 1;
            }
        }
    }
    let adj = (0..n).into_par_iter().map(|i| {
        let mut a = Vec::new();
        for word in 0..nw {
            let mut keep = if mutual { bits[i][word] & reverse[i][word] } else { bits[i][word] | reverse[i][word] };
            // Increasing words/bits give exactly the original increasing-neighbor order.
            while keep != 0 {
                let j = word * 64 + keep.trailing_zeros() as usize;
                keep &= keep - 1;
                if i == j { continue; }
                // The original undirected builder reads the upper triangle for both directions.
                let v = s[[i.min(j), i.max(j)]];
                if v.is_finite() && threshold.map_or(true, |t| v > t) {
                    a.push((j as u32, v as f32));
                }
            }
        }
        a
    }).collect();
    UGraph { n, adj }
}
'''

MST = r'''
fn build_mst(s: &Array2<f64>) -> UGraph {
    if !RELAY_GRAPH_CORRECT_FIXES { return relay_old_build_mst(s); }
    let n = s.nrows();
    let mut intree = vec![false;n];
    let mut best = vec![f64::NEG_INFINITY;n];
    let mut parent = vec![usize::MAX;n];
    for _ in 0..n {
        let mut u = usize::MAX;
        let mut value = f64::NEG_INFINITY;
        for j in 0..n {
            if !intree[j] && best[j] > value { value = best[j]; u = j; }
        }
        // A disconnected finite-similarity component needs a fresh Prim root.
        if u == usize::MAX { u = (0..n).find(|&j| !intree[j]).unwrap(); }
        intree[u] = true;
        for j in 0..n {
            if !intree[j] {
                let v = s[[u,j]];
                if v.is_finite() && v > best[j] { best[j] = v; parent[j] = u; }
            }
        }
    }
    let mut adj = vec![Vec::new();n];
    for j in 0..n {
        if parent[j] != usize::MAX {
            let v = s[[j,parent[j]]] as f32;
            adj[j].push((parent[j] as u32,v)); adj[parent[j]].push((j as u32,v));
        }
    }
    UGraph { n,adj }
}
'''

PROBE = r'''
pub fn graph_metrics_probe(input: Array2<f64>, old: bool) -> Vec<Vec<f64>> {
    let n = input.nrows();
    let mut g = UGraph { n, adj: vec![Vec::new(); n] };
    for i in 0..n { for j in i+1..n {
        let w = input[[i,j]];
        if w.is_finite() { g.adj[i].push((j as u32,w as f32)); g.adj[j].push((i as u32,w as f32)); }
    }}
    let kc = if old { relay_old_kcore(&g) } else { kcore(&g) };
    let cc = if old { relay_old_clustering_w(&g) } else { clustering_w(&g) };
    let co = if old { relay_old_communities(&g) } else { communities(&g) };
    let pr = if old { relay_old_pagerank(&g,0.85,50) } else { pagerank(&g,0.85,50) };
    vec![kc.into_iter().map(|v| v as f64).collect(),cc,co.into_iter().map(|v| v as f64).collect(),pr]
}

pub fn graph_build_probe(input: Array2<f64>, k: usize, mutual: bool) -> Array2<f64> {
    let g = build_knn(&input,k,mutual);
    let mut out = Array2::from_elem((g.n,g.n),f64::NAN);
    for i in 0..g.n { for &(j,w) in &g.adj[i] { out[[i,j as usize]]=w as f64; } }
    out
}

pub fn graph_mst_probe(input: Array2<f64>) -> Array2<f64> {
    let g = build_mst(&input);
    let mut out = Array2::from_elem((g.n,g.n),f64::NAN);
    for i in 0..g.n { for &(j,w) in &g.adj[i] { out[[i,j as usize]]=w as f64; } }
    out
}

pub fn graph_probe() -> Result<String, String> {
    let mut checked = 0;
    for seed in 0..20u64 {
        let n = (seed as usize * 13 % 61) + 1;
        let mut g = UGraph { n, adj: vec![Vec::new(); n] };
        let mut rng = seed + 37;
        for i in 0..n {
            for j in i + 1..n {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                if (rng >> 32) % 7 <= seed % 7 {
                    let w = (((rng >> 24) % 19) as f32 - 4.0) / 20.0;
                    g.adj[i].push((j as u32, w)); g.adj[j].push((i as u32, w));
                }
            }
        }
        if kcore(&g) != relay_old_kcore(&g) { return Err(format!("kcore seed={seed}")); }
        if communities(&g) != relay_old_communities(&g) { return Err(format!("communities seed={seed}")); }
        let a = clustering_w(&g); let b = relay_old_clustering_w(&g);
        if a.iter().zip(&b).any(|(x,y)| x.to_bits() != y.to_bits()) {
            return Err(format!("clustering bit mismatch seed={seed}"));
        }
        let a = eigenvec(&g,50); let b = relay_old_eigenvec(&g,50);
        if a.iter().zip(&b).any(|(x,y)| x.to_bits() != y.to_bits()) {
            return Err(format!("eigenvec bit mismatch seed={seed}"));
        }
        if RELAY_GRAPH_CORRECT_FIXES {
            let a = pagerank(&g,0.85,50); let b = relay_correct_push_pagerank(&g,0.85,50);
            if a.iter().zip(&b).any(|(x,y)| x.to_bits() != y.to_bits()) {
                return Err(format!("pagerank pull/push bit mismatch seed={seed}"));
            }
            // Trees can have unsorted adjacency rows; incoming pull still preserves source order.
            for row in &mut g.adj { row.reverse(); }
            let a = pagerank(&g,0.85,50); let b = relay_correct_push_pagerank(&g,0.85,50);
            if a.iter().zip(&b).any(|(x,y)| x.to_bits() != y.to_bits()) {
                return Err(format!("pagerank unsorted bit mismatch seed={seed}"));
            }
        }
        checked += 1;
    }
    let n = 17;
    let s = Array2::from_shape_fn((n,n), |(i,j)| {
        if i == j { f64::NAN } else { ((i * 7 + j * 13) % 11) as f64 / 10.0 }
    });
    for k in [1,2,5,20] {
        for mutual in [false,true] {
            let fast = build_knn(&s,k,mutual);
            let expected = if RELAY_GRAPH_CORRECT_FIXES {
                let mut inset = vec![vec![false;n];n];
                for i in 0..n {
                    let mut js: Vec<usize> = (0..n).filter(|&j| j != i).collect();
                    js.sort_by(|&a,&b| s[[i,b]].total_cmp(&s[[i,a]]).then(a.cmp(&b)));
                    for j in js.into_iter().take(k) { inset[i][j]=true; }
                }
                let mut adj = vec![Vec::new();n];
                for i in 0..n { for j in i+1..n {
                    if if mutual { inset[i][j] && inset[j][i] } else { inset[i][j] || inset[j][i] } {
                        adj[i].push((j as u32,s[[i,j]] as f32)); adj[j].push((i as u32,s[[i,j]] as f32));
                    }
                }}
                UGraph {n,adj}
            } else { relay_old_build_knn(&s,k,mutual) };
            if fast.adj != expected.adj { return Err(format!("knn k={k} mutual={mutual}")); }
        }
    }
    let g = UGraph {n:2, adj:vec![vec![(1,1.0)],vec![(0,1.0)]]};
    let pr = pagerank(&g,0.85,50);
    if RELAY_GRAPH_CORRECT_FIXES {
        if pr.iter().any(|&p| (p - 0.5).abs() > 1e-14) { return Err(format!("pagerank={pr:?}")); }
    } else if pr != relay_old_pagerank(&g,0.85,50) { return Err("pagerank compatibility".into()); }
    let mut sim = Array2::from_elem((7,7),f64::NAN);
    for (i,j,w) in [(0,1,0.9),(1,2,0.8),(0,2,0.5),(3,4,0.7),(4,5,0.6),(3,5,0.4)] {
        sim[[i,j]]=w; sim[[j,i]]=w;
    }
    let forest = build_mst(&sim);
    let edge_count: usize = forest.adj.iter().map(Vec::len).sum::<usize>() / 2;
    if edge_count != if RELAY_GRAPH_CORRECT_FIXES {4} else {2} || !forest.adj[6].is_empty() {
        return Err(format!("forest edges={edge_count}"));
    }
    let full = dist_stats(&forest,n_src_of(forest.n)).0;
    let from_cache = mst_extra(&forest,&vec![0.5;forest.n],Some(&full));
    let recompute = mst_extra(&forest,&vec![0.5;forest.n],None);
    if from_cache.iter().flatten().zip(recompute.iter().flatten()).any(|(a,b)| a.to_bits()!=b.to_bits()) {
        return Err("mst cached harmonic mismatch".into());
    }
    Ok(format!("{checked} exact random graph checks; 8 top-k oracle checks; two-node PageRank; forest + cached-harmonic checks; corrected={RELAY_GRAPH_CORRECT_FIXES}"))
}
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compat", action="store_true")
    parser.add_argument("--target", type=Path, default=HERE / "src/relaynet_fast.rs")
    args = parser.parse_args()
    text = args.target.read_text()
    begin = "// RELAY_GRAPH_PATCH_HELPERS_BEGIN"
    end = "// RELAY_GRAPH_PATCH_HELPERS_END"
    if begin in text:
        a = text.index(begin)
        b = text.index(end, a) + len(end)
        text = text[:a] + text[b:]
    baseline = (HERE / "src/relaynet_baseline.rs").read_text()
    originals = []
    for name in ["kcore", "communities", "clustering_w", "build_knn", "build_mst", "pagerank", "eigenvec"]:
        a, b = function_span(baseline, name)
        originals.append(baseline[a:b].replace(f"fn {name}(", f"fn relay_old_{name}(", 1))
    a,b = function_span(baseline,"pagerank")
    push_reference = baseline[a:b].replace("fn pagerank(","fn relay_correct_push_pagerank(",1)
    push_reference = push_reference.replace("        let mut dangling = 0.0;", "        next.fill((1.0 - d) / n as f64);\n        let mut dangling = 0.0;")
    push_reference = push_reference.replace("            next[i] = (1.0 - d) / n as f64;", "")
    originals.append(push_reference)
    text = replace_function(text, "kcore", KCORE)
    text = replace_function(text, "communities", COMMUNITIES)
    text = replace_function(text, "clustering_w", CLUSTERING)
    text = replace_function(text, "eigenvec", EIGENVEC)
    text = replace_function(text, "build_mst", MST)
    text = replace_function(text, "build_knn", r'''
fn build_knn(s: &Array2<f64>, k: usize, mutual: bool) -> UGraph {
    let bits = relay_candidate_bits(s, k, None);
    relay_graph_from_bits(s, &bits, mutual, None)
}''')
    text = replace_function(text, "build_thresh_limited", r'''
fn build_thresh_limited(s: &Array2<f64>, t: f64, allow: &[bool]) -> UGraph {
    let n = s.nrows();
    let k = (n / 25).max(200).min(n.saturating_sub(1)).max(1);
    let bits = relay_candidate_bits(s, k, Some(allow));
    relay_graph_from_bits(s, &bits, false, Some(t))
}''')
    # Always start these from the snapshot so repeated application is safe.
    for name in ["pagerank", "dir_stats"]:
        a, b = function_span(baseline, name)
        f = baseline[a:b]
        for arr, base in [("next", "(1.0 - d) / n as f64"), ("next", "0.15 / n as f64"), ("next2", "0.15 / n as f64")]:
            old = f"            {arr}[i] = {base};"
            if old in f:
                f = f.replace(old, f"            if !RELAY_GRAPH_CORRECT_FIXES {{ {arr}[i] = {base}; }}")
                # Each PageRank loop has a distinct dangling declaration and exact next-array assignment.
                loop = f.index(f"if !RELAY_GRAPH_CORRECT_FIXES {{ {arr}[i]")
                pos = f.rfind("        let mut dangling = 0.0;", 0, loop)
                f = f[:pos] + f"        if RELAY_GRAPH_CORRECT_FIXES {{ {arr}.fill({base}); }}\n" + f[pos:]
        text = replace_function(text, name, f)
    text = replace_function(text,"pagerank",PAGERANK)
    text = replace_function(text, "n_src_of", "fn n_src_of(n: usize) -> usize { if RELAY_GRAPH_CORRECT_FIXES { n } else { (n / 25).max(10).min(n) } }")
    a,b = function_span(baseline,"mst_extra")
    mst_extra = baseline[a:b].replace(
        "fn mst_extra(g: &UGraph, mkt_corr: &[f64])", "fn mst_extra(g: &UGraph, mkt_corr: &[f64], precomputed_harm: Option<&[f64]>)"
    ).replace("    let (harm, _, _) = dist_stats(g, n_src_of(n));", """    let owned_harm;
    let harm = if let Some(h) = precomputed_harm { h } else {
        owned_harm = dist_stats(g, n_src_of(n)).0;
        &owned_harm
    };""")
    text = replace_function(text,"mst_extra",mst_extra)
    text = text.replace("mst_extra(&g, &mkt_corr)", "mst_extra(&g, &mkt_corr, if sp.full { Some(cols[9].as_slice()) } else { None })")
    if not args.compat:
        text = text.replace("/// 采样源的个数：max(10, n/25)，把全源 Dijkstra 压成 ~1/25", "/// 完整版本使用全部源节点；兼容回归分支保留旧源数。")
        text = text.replace("/// 采样最短路统计：只从 nsrc 个等距源跑 Dijkstra，\n/// 采样源出真实值，其余节点填采样源的平均（形状不变）", "/// 全节点精确最短路统计，不抽样或填补其他节点。")
        text = text.replace("/// 加权中介（Dijkstra 版 Brandes），nsrc 个等距抽样源近似", "/// 全节点加权中介（Dijkstra 版 Brandes）。")
        text = text.replace("// 出/入调和距离（沿出边 / 沿入边多源... 简单版: 从每点沿出边到所有点 —— 用反向 BFS 近似跳数）", "// 出/入调和距离：从每个节点沿对应方向求全部加权最短路。")
        text = text.replace("// is_art: 孩子数>0 且非根非叶 → 近似\"内部分叉节点\"", "// is_art 保留既有分叉节点定义（度 >= 3），不是严格割点判定。")
    helpers = "\n".join([begin, f"const RELAY_GRAPH_CORRECT_FIXES: bool = {str(not args.compat).lower()};", BUILDERS, *originals, PROBE, end])
    args.target.write_text(text + "\n" + helpers + "\n")
    print(f"Patched {args.target}; compatibility={args.compat}")


if __name__ == "__main__":
    main()
