"""Install exact all-source paths; positive costs, no source sampling or near ties."""
from pathlib import Path

p = Path(__file__).parent / 'src/relaynet_fast.rs'
s = p.read_text()

def replace_fn(name, replacement):
    global s
    start = s.index(f'fn {name}(')
    brace = s.index('{', start)
    depth = 1
    end = brace + 1
    while depth:
        depth += (s[end] == '{') - (s[end] == '}')
        end += 1
    s = s[:start] + replacement + s[end:]

helpers = r'''
// EXACT_PATHS_BEGIN
/// A forest has one positive-cost path per reachable pair. Its weighted and
/// binary betweenness equal the count of pairs in distinct branches at a node.
fn forest_bet(g: &UGraph) -> Option<Vec<f64>> {
    let n = g.n;
    let mut parent = vec![usize::MAX; n];
    let mut size = vec![1usize; n];
    let mut bet = vec![0.0; n];
    let mut order = Vec::with_capacity(n);
    let mut components = 0usize;
    for root in 0..n {
        if parent[root] != usize::MAX { continue; }
        components += 1;
        let first = order.len();
        parent[root] = root;
        order.push(root);
        let mut head = first;
        while head < order.len() {
            let u = order[head]; head += 1;
            for &(v, _) in &g.adj[u] {
                let v = v as usize;
                if parent[v] == usize::MAX { parent[v] = u; order.push(v); }
            }
        }
        let count = order.len() - first;
        for &u in order[first..].iter().rev() {
            if u != root { size[parent[u]] += size[u]; }
        }
        for &u in &order[first..] {
            let mut sum = 0u64;
            let mut pairs = 0u64;
            for &(v, _) in &g.adj[u] {
                let v = v as usize;
                let branch = if parent[v] == u { size[v] } else { count - size[u] } as u64;
                pairs += sum * branch; sum += branch;
            }
            bet[u] = pairs as f64;
        }
    }
    let edges = g.adj.iter().map(|a| a.len()).sum::<usize>() / 2;
    if edges == n.saturating_sub(components) { Some(bet) } else { None }
}

struct ExactPaths {
    columns: [Vec<f64>; 7],
    bet: Vec<f64>,
}
impl ExactPaths {
    fn distances(self) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let [h, f, e, q25, q75, q50, q90] = self.columns;
        (h, f, e, q25, q75, q50, q90)
    }
}

/// Reused for a fixed chunk of sources, so allocation and reduction order do
/// not depend on worker scheduling. No per-node predecessor vectors are needed.
struct PathScratch {
    dist: Vec<f64>, sigma: Vec<f64>, delta: Vec<f64>,
    order: Vec<usize>, heap: BinaryHeap<State>, values: Vec<f64>,
}
impl PathScratch {
    fn new(n: usize) -> Self {
        Self { dist: vec![f64::INFINITY; n], sigma: vec![0.0; n],
            delta: vec![0.0; n], order: Vec::with_capacity(n),
            heap: BinaryHeap::with_capacity(n), values: Vec::with_capacity(n) }
    }
    fn source(&mut self, g: &UGraph, src: usize, tree: bool, weighted: bool, bet: &mut [f64]) -> [f64; 7] {
        self.dist.fill(f64::INFINITY); self.dist[src] = 0.0;
        self.order.clear(); self.values.clear(); self.heap.clear();
        if weighted && !tree {
            self.sigma.fill(0.0); self.sigma[src] = 1.0; self.delta.fill(0.0);
        }
        if tree {
            self.order.push(src);
            let mut head = 0;
            while head < self.order.len() {
                let u = self.order[head]; head += 1;
                let du = self.dist[u];
                for &(v, sim) in &g.adj[u] {
                    let v = v as usize;
                    if !self.dist[v].is_finite() {
                        self.dist[v] = du + (1.0 - sim as f64).max(1e-9);
                        self.order.push(v);
                    }
                }
            }
        } else {
            self.heap.push(State { d: 0.0, u: src });
            while let Some(State { d: du, u }) = self.heap.pop() {
                if du != self.dist[u] { continue; }
                if weighted { self.order.push(u); }
                for &(v, sim) in &g.adj[u] {
                    let v = v as usize;
                    let nd = du + (1.0 - sim as f64).max(1e-9);
                    if nd < self.dist[v] {
                        self.dist[v] = nd;
                        self.heap.push(State { d: nd, u: v });
                        if weighted { self.sigma[v] = self.sigma[u]; }
                    } else if weighted && nd == self.dist[v] {
                        self.sigma[v] += self.sigma[u];
                    }
                }
            }
        }
        if weighted && !tree {
            for &u in self.order.iter().rev() {
                for &(v, sim) in &g.adj[u] {
                    let v = v as usize;
                    if self.dist[v] + (1.0 - sim as f64).max(1e-9) == self.dist[u] {
                        self.delta[v] += (self.sigma[v] / self.sigma[u]) * (1.0 + self.delta[u]);
                    }
                }
                if u != src { bet[u] += self.delta[u]; }
            }
        }
        let (mut h, mut f, mut e) = (0.0, 0.0, 0.0f64);
        // Preserve the original node-index summation order exactly.
        for (j, &d) in self.dist.iter().enumerate() {
            if j != src && d.is_finite() && d > 0.0 {
                h += 1.0 / d; f += d; e = e.max(d); self.values.push(d);
            }
        }
        if self.values.is_empty() { return [f64::NAN; 7]; }
        self.values.sort_unstable_by(|a, b| a.total_cmp(b));
        let count = self.values.len() as f64;
        [h / count, f / count, e, quantile_sorted(&self.values, 0.25),
            quantile_sorted(&self.values, 0.75), quantile_sorted(&self.values, 0.50),
            quantile_sorted(&self.values, 0.90)]
    }
}

fn exact_paths(g: &UGraph, weighted: bool) -> ExactPaths {
    let n = g.n;
    let forest = forest_bet(g);
    let tree = forest.is_some();
    // All n sources are used, including isolates (NaN distances, zero bet).
    // Fixed consecutive blocks give deterministic reduction across thread counts.
    const BLOCK: usize = 16;
    let parts: Vec<(Vec<[f64; 7]>, Vec<f64>)> = (0..n.div_ceil(BLOCK)).into_par_iter().map(|block| {
        let lo = block * BLOCK; let hi = (lo + BLOCK).min(n);
        let mut scratch = PathScratch::new(n);
        let mut bet = vec![0.0; n];
        let rows = (lo..hi).map(|src| scratch.source(g, src, tree, weighted, &mut bet)).collect();
        (rows, bet)
    }).collect();
    let mut columns: [Vec<f64>; 7] = std::array::from_fn(|_| Vec::with_capacity(n));
    let mut bet = forest.unwrap_or_else(|| vec![0.0; n]);
    for (rows, source_bet) in parts {
        for row in rows { for j in 0..7 { columns[j].push(row[j]); } }
        if weighted && !tree { for (a, b) in bet.iter_mut().zip(source_bet) { *a += b; } }
    }
    if weighted && !tree { for b in &mut bet { *b *= 0.5; } }
    ExactPaths { columns, bet }
}
// EXACT_PATHS_END
'''
if '// EXACT_PATHS_BEGIN' in s:
    a=s.index('// EXACT_PATHS_BEGIN'); b=s.index('// EXACT_PATHS_END',a)+len('// EXACT_PATHS_END')
    s=s[:a]+helpers.strip()+s[b:]
else:
    pos=s.index('fn dist_stats_q(');s=s[:pos]+helpers+'\n'+s[pos:]
replace_fn('dist_stats_q',r'''fn dist_stats_q(g: &UGraph, _nsrc: usize)
    -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>)
{
    exact_paths(g, false).distances()
}''')
replace_fn('betweenness_weighted',r'''fn betweenness_weighted(g: &UGraph, _nsrc: usize) -> Vec<f64> {
    exact_paths(g, true).bet
}''')
replace_fn('betweenness_bin',r'''fn betweenness_bin(g: &UGraph, _nsrc: usize) -> Vec<f64> {
    if let Some(bet) = forest_bet(g) { return bet; }
    let n = g.n;
    const BLOCK: usize = 16;
    let parts: Vec<Vec<f64>> = (0..n.div_ceil(BLOCK)).into_par_iter().map(|block| {
        let mut dist = vec![-1i32; n]; let mut sigma = vec![0.0; n];
        let mut delta = vec![0.0; n]; let mut bet = vec![0.0; n];
        let mut queue = Vec::with_capacity(n);
        for src in block * BLOCK..((block + 1) * BLOCK).min(n) {
            if g.adj[src].is_empty() { continue; }
            dist.fill(-1); sigma.fill(0.0); delta.fill(0.0); queue.clear();
            dist[src] = 0; sigma[src] = 1.0; queue.push(src);
            let mut head = 0;
            while head < queue.len() {
                let u = queue[head]; head += 1;
                for &(v, _) in &g.adj[u] {
                    let v = v as usize;
                    if dist[v] < 0 { dist[v] = dist[u] + 1; queue.push(v); }
                    if dist[v] == dist[u] + 1 { sigma[v] += sigma[u]; }
                }
            }
            for &u in queue.iter().rev() {
                for &(v, _) in &g.adj[u] {
                    let v = v as usize;
                    if dist[v] + 1 == dist[u] {
                        delta[v] += (sigma[v] / sigma[u]) * (1.0 + delta[u]);
                    }
                }
                if u != src { bet[u] += delta[u]; }
            }
        }
        bet
    }).collect();
    let mut bet = vec![0.0; n];
    for part in parts { for (a, b) in bet.iter_mut().zip(part) { *a += b; } }
    for b in &mut bet { *b *= 0.5; }
    bet
}''')
replace_fn('dijkstra',r'''fn dijkstra(g: &UGraph, src: usize) -> Vec<f64> {
    let mut d = vec![f64::INFINITY; g.n]; d[src] = 0.0;
    let mut heap = BinaryHeap::new(); heap.push(State { d: 0.0, u: src });
    while let Some(State { d: du, u }) = heap.pop() {
        if du != d[u] { continue; }
        for &(v, sim) in &g.adj[u] {
            let v = v as usize; let nd = du + (1.0 - sim as f64).max(1e-9);
            if nd < d[v] { d[v] = nd; heap.push(State { d: nd, u: v }); }
        }
    }
    d
}''')
old='    let (harm, farness, ecc, d25, d75, pdl50, dist_p90) = dist_stats_q(g, n_src_of(n));'
new='''    let exact = exact_paths(g, true);
    let [harm, farness, ecc, d25, d75, pdl50, dist_p90] = exact.columns;'''
if old in s: s=s.replace(old,new,1)
s=s.replace('    cols.push(betweenness_weighted(g, n_src_of(n)));','    cols.push(exact.bet);',1)
if 'pub fn exact_path_probe(' not in s:
    s += r'''
/// Independent Python-oracle interface: seven distances, weighted, binary bet.
pub fn exact_path_probe(n: usize, edges: Vec<(usize, usize, f32)>) -> Vec<Vec<f64>> {
    let mut g = UGraph { n, adj: vec![Vec::new(); n] };
    for (u, v, sim) in edges { g.adj[u].push((v as u32, sim)); g.adj[v].push((u as u32, sim)); }
    for a in &mut g.adj { a.sort_by_key(|&(v, _)| v); }
    let result = exact_paths(&g, true);
    let mut cols = Vec::from(result.columns); cols.push(result.bet);
    cols.push(betweenness_bin(&g, n)); cols
}
'''
# Exact comparisons also apply to virtual-market, leader-set, and directed paths.
s=s.replace('du > d[u] + 1e-12', 'du != d[u]')
s=s.replace('du + w < d[v as usize] - 1e-12', 'du + w < d[v as usize]')
s=s.replace('/// 采样最短路统计：只从 nsrc 个等距源跑 Dijkstra，\n/// 采样源出真实值，其余节点填采样源的平均（形状不变）',
            '/// 全源精确最短路：每个节点独立计算，不使用抽样或邻居均值代填。')
s=s.replace('/// 加权中介（Dijkstra 版 Brandes），nsrc 个等距抽样源近似',
            '/// 全源精确加权中介；距离和路径依赖在同一次遍历中计算。')
s=s.replace('/// 采样源的个数：max(10, n/25)，把全源 Dijkstra 压成 ~1/25',
            '/// 正式精确模式使用所有节点；历史对照模式仅供测试。')
s=s.replace('// 出/入调和距离（沿出边 / 沿入边多源... 简单版: 从每点沿出边到所有点 —— 用反向 BFS 近似跳数）',
            '// 出/入调和距离：每个节点的全源加权 Dijkstra，沿出边及入边分别计算。')
# Optional instrumentation adds no per-edge work and remains quiet by default.
a=s.index('fn stat_undir('); b=s.index('\n/// 到"市场"',a)
body=s[a:b]
if 'let profile_detail' not in body:
    body=body.replace('    let n = g.n;', '''    let n = g.n;
    let profile_detail = std::env::var_os("RLG_DETAIL_PROFILE").is_some();
    let mut profile_clock = std::time::Instant::now();
    let profile_mark = |clock: &mut std::time::Instant, label: &str| {
        if profile_detail {
            eprintln!("[rlg-kernel] {label} sec={:.6} nodes={n}", clock.elapsed().as_secs_f64());
        }
        *clock = std::time::Instant::now();
    };''',1)
    body=body.replace('    let exact = exact_paths(g, true);',
        '    profile_mark(&mut profile_clock, "basic_neighborhoods");\n    let exact = exact_paths(g, true);\n    profile_mark(&mut profile_clock, "weighted_paths_distances");')
    body=body.replace('    cols.push(betweenness_bin(g, n_src_of(n)));',
        '    cols.push(betweenness_bin(g, n_src_of(n)));\n    profile_mark(&mut profile_clock, "binary_betweenness");')
    body=body.replace('    cols.push(eigenvec(g, 50));',
        '    cols.push(eigenvec(g, 50));\n    profile_mark(&mut profile_clock, "pagerank_eigenvector");')
    body=body.replace('    cols.push(clustering_w(g));',
        '    cols.push(clustering_w(g));\n    profile_mark(&mut profile_clock, "clustering");')
    body=body.replace('    // ---- 追加的 7 个统计量（只 full 图有） ----',
        '    profile_mark(&mut profile_clock, "communities_market_leaders");\n    // ---- 追加的 7 个统计量（只 full 图有） ----')
    body=body.replace('    // 先把原有的 ndeg/hhi/jacc/leafnb/d25/d75/assort 7 个推回 cols',
        '    profile_mark(&mut profile_clock, "extended_neighborhoods");\n    // 先把原有的 ndeg/hhi/jacc/leafnb/d25/d75/assort 7 个推回 cols')
    s=s[:a]+body+s[b:]

p.write_text(s)
print('Applied exact all-source fused shortest paths and both Brandes variants')
