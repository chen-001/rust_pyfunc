"""Exact weighted Brandes parent arrays; apply after exact_paths_patch."""
from pathlib import Path
p=Path(__file__).parent/'src/relaynet_fast.rs';s=p.read_text()
if 'fn exact_paths_mode(' not in s:
    s=s.replace('order: Vec<usize>, heap: BinaryHeap<State>, values: Vec<f64>,',
        'order: Vec<usize>, heap: BinaryHeap<State>, values: Vec<f64>,\n    parent: Vec<u32>, head: Vec<u32>, pred_from: Vec<u32>, pred_next: Vec<u32>,')
    s=s.replace('heap: BinaryHeap::with_capacity(n), values: Vec::with_capacity(n) }',
        'heap: BinaryHeap::with_capacity(n), values: Vec::with_capacity(n),\n            parent: vec![u32::MAX; n], head: vec![u32::MAX; n],\n            pred_from: Vec::with_capacity(n), pred_next: Vec::with_capacity(n) }')
    s=s.replace('tree: bool, weighted: bool, bet: &mut [f64])',
        'tree: bool, weighted: bool, parents: bool, bet: &mut [f64])')
    s=s.replace('self.sigma.fill(0.0); self.sigma[src] = 1.0; self.delta.fill(0.0);',
        'self.sigma.fill(0.0); self.sigma[src] = 1.0; self.delta.fill(0.0);\n            if parents { self.parent.fill(u32::MAX); self.head.fill(u32::MAX); self.pred_from.clear(); self.pred_next.clear(); }')
    s=s.replace('if weighted { self.sigma[v] = self.sigma[u]; }', '''if weighted {
                            self.sigma[v] = self.sigma[u];
                            if parents { self.parent[v] = u as u32; self.head[v] = u32::MAX; }
                        }''')
    s=s.replace('} else if weighted && nd == self.dist[v] {\n                        self.sigma[v] += self.sigma[u];', '''} else if weighted && nd == self.dist[v] {
                        self.sigma[v] += self.sigma[u];
                        if parents {
                            self.pred_from.push(u as u32); self.pred_next.push(self.head[v]);
                            self.head[v] = (self.pred_from.len() - 1) as u32;
                        }''')
    old='''                for &(v, sim) in &g.adj[u] {
                    let v = v as usize;
                    if self.dist[v] + (1.0 - sim as f64).max(1e-9) == self.dist[u] {
                        self.delta[v] += (self.sigma[v] / self.sigma[u]) * (1.0 + self.delta[u]);
                    }
                }'''
    new='''                if parents {
                    let first = self.parent[u];
                    if first != u32::MAX {
                        let v = first as usize;
                        self.delta[v] += (self.sigma[v] / self.sigma[u]) * (1.0 + self.delta[u]);
                    }
                    let mut link = self.head[u];
                    while link != u32::MAX {
                        let v = self.pred_from[link as usize] as usize;
                        self.delta[v] += (self.sigma[v] / self.sigma[u]) * (1.0 + self.delta[u]);
                        link = self.pred_next[link as usize];
                    }
                } else {
                    for &(v, sim) in &g.adj[u] {
                        let v = v as usize;
                        if self.dist[v] + (1.0 - sim as f64).max(1e-9) == self.dist[u] {
                            self.delta[v] += (self.sigma[v] / self.sigma[u]) * (1.0 + self.delta[u]);
                        }
                    }
                }'''
    assert old in s;s=s.replace(old,new,1)
    s=s.replace('fn exact_paths(g: &UGraph, weighted: bool) -> ExactPaths {',
        'fn exact_paths(g: &UGraph, weighted: bool) -> ExactPaths { exact_paths_mode(g, weighted, true) }\n\nfn exact_paths_mode(g: &UGraph, weighted: bool, parents: bool) -> ExactPaths {',1)
    s=s.replace('scratch.source(g, src, tree, weighted, &mut bet)',
        'scratch.source(g, src, tree, weighted, parents, &mut bet)')
if 'fn rlg_dump_graph(' not in s:
    s += r'''
/// Profiling-only complete first graph dump; never changes the computation.
fn rlg_dump_graph(g: &UGraph) {
    let Some(path) = std::env::var_os("RLG_DUMP_GRAPH_PATH") else { return; };
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        use std::io::Write;
        let mut f = std::io::BufWriter::new(std::fs::File::create(path).unwrap());
        f.write_all(&(g.n as u32).to_le_bytes()).unwrap();
        for adj in &g.adj {
            f.write_all(&(adj.len() as u32).to_le_bytes()).unwrap();
            for &(v, sim) in adj { f.write_all(&v.to_le_bytes()).unwrap(); f.write_all(&sim.to_le_bytes()).unwrap(); }
        }
        f.flush().unwrap();
        eprintln!("[rlg-dump] complete full graph nodes={} edges={}", g.n, g.adj.iter().map(|a|a.len()).sum::<usize>()/2);
    });
}

pub fn weighted_graph_probe(path: &str, parents: bool) -> std::io::Result<Vec<Vec<f64>>> {
    use std::io::Read;
    let mut f = std::io::BufReader::new(std::fs::File::open(path)?);
    fn read4(f: &mut impl Read) -> std::io::Result<[u8; 4]> { let mut b=[0;4]; f.read_exact(&mut b)?; Ok(b) }
    let n = u32::from_le_bytes(read4(&mut f)?) as usize;
    let mut g = UGraph { n, adj: Vec::with_capacity(n) };
    for _ in 0..n {
        let count=u32::from_le_bytes(read4(&mut f)?) as usize; let mut adj=Vec::with_capacity(count);
        for _ in 0..count { adj.push((u32::from_le_bytes(read4(&mut f)?), f32::from_le_bytes(read4(&mut f)?))); }
        g.adj.push(adj);
    }
    let result = exact_paths_mode(&g, true, parents);
    let mut columns = Vec::from(result.columns); columns.push(result.bet); Ok(columns)
}
'''
    a=s.index('fn stat_undir('); brace=s.index('    let n = g.n;',a)
    s=s[:brace]+'    rlg_dump_graph(g);\n'+s[brace:]
p.write_text(s)
print('Applied exact first-parent + linked-tie predecessors; full graph profiling enabled by env only')
