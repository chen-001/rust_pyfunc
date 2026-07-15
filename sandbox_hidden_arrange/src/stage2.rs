//! 第二层：多关系图游戏（与 Python stage2 对照）
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use crate::{Window, Bank, average_rank, assign_by_rank};

static T_ADJ: AtomicU64 = AtomicU64::new(0);
static T_BISECT: AtomicU64 = AtomicU64::new(0);
static T_CAND: AtomicU64 = AtomicU64::new(0);
static T_QUERY: AtomicU64 = AtomicU64::new(0);
static T_KENDALL: AtomicU64 = AtomicU64::new(0);
static T_VOTE: AtomicU64 = AtomicU64::new(0);
static T_POOL: AtomicU64 = AtomicU64::new(0);
#[inline] fn acc(c: &AtomicU64, d: std::time::Duration) { c.fetch_add(d.as_nanos() as u64, Ordering::Relaxed); }
pub fn print_s2_timing() {
    println!("--- stage2 内部 ---");
    println!("  邻接表构建: {:.3}s", T_ADJ.load(Ordering::Relaxed) as f64 * 1e-9);
    println!("  graph_bisect: {:.3}s", T_BISECT.load(Ordering::Relaxed) as f64 * 1e-9);
    println!("  graph_candidate: {:.3}s", T_CAND.load(Ordering::Relaxed) as f64 * 1e-9);
    println!("  bank.query: {:.3}s", T_QUERY.load(Ordering::Relaxed) as f64 * 1e-9);
    println!("  normalized_kendall: {:.3}s", T_KENDALL.load(Ordering::Relaxed) as f64 * 1e-9);
    println!("  pool克隆: {:.3}s", T_POOL.load(Ordering::Relaxed) as f64 * 1e-9);
    println!("  vote: {:.3}s", T_VOTE.load(Ordering::Relaxed) as f64 * 1e-9);
}

pub const GRAPH_NAMES: [&str; 12] = [
    "TIME_R1", "TIME_R8", "TIME_R32", "ACTIVE_EXACT", "PASSIVE_EXACT",
    "PRICE_T0", "PRICE_T2", "SIDE_EXACT", "BOOK_K10", "BOOK_K30", "QUEUE_K10", "ARRIVAL_K10",
];
pub const SCOPES: [&str; 3] = ["LOCAL32", "MESO8_D2", "GLOBAL_D4"];
const PRICE_TICK: f64 = 0.01;

pub type EdgeSet = HashSet<(usize, usize)>;

fn add_edge(edges: &mut EdgeSet, u: usize, v: usize) {
    if u != v {
        let (a, b) = if u < v { (u, v) } else { (v, u) };
        edges.insert((a, b));
    }
}

fn knn_graph(win: &Window, feats: &[f64], nfeat: usize, k: usize, out: &mut EdgeSet) {
    let n = win.n;
    // z-score
    let mut mu = vec![0f64; nfeat];
    let mut sd = vec![0f64; nfeat];
    for j in 0..nfeat {
        let mut s = 0f64;
        for i in 0..n { s += feats[i * nfeat + j]; }
        mu[j] = s / n as f64;
    }
    for j in 0..nfeat {
        let mut s = 0f64;
        for i in 0..n { let d = feats[i * nfeat + j] - mu[j]; s += d * d; }
        sd[j] = (s / n as f64).sqrt();
        if sd[j] < 1e-12 { sd[j] = 1.0; }
    }
    let mut z = vec![0f64; n * nfeat];
    for i in 0..n {
        for j in 0..nfeat {
            z[i * nfeat + j] = (feats[i * nfeat + j] - mu[j]) / sd[j];
        }
    }
    // 距离矩阵 D2[i][j] = sq[i]+sq[j]-2*z[i]·z[j]
    let sq: Vec<f64> = (0..n).map(|i| {
        (0..nfeat).map(|j| z[i * nfeat + j] * z[i * nfeat + j]).sum::<f64>()
    }).collect();
    let kk = k.min(n - 1);
    for i in 0..n {
        // D2[i][j]
        let mut d2: Vec<(f64, usize)> = (0..n).map(|j| {
            let dot: f64 = (0..nfeat).map(|jj| z[i * nfeat + jj] * z[j * nfeat + jj]).sum();
            let v = sq[i] + sq[j] - 2.0 * dot;
            (if v < 0.0 { 0.0 } else { v }, j)
        }).collect();
        d2.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap().then(a.1.cmp(&b.1))); // stable: 同距按j升序
        let mut cnt = 0;
        for (_, j) in d2 {
            if j == i { continue; }
            add_edge(out, i, j);
            cnt += 1;
            if cnt >= kk { break; }
        }
    }
}

pub fn build_graphs(win: &Window) -> HashMap<&'static str, EdgeSet> {
    let n = win.n;
    let mut graphs: HashMap<&'static str, EdgeSet> = HashMap::new();
    for g in GRAPH_NAMES { graphs.insert(g, EdgeSet::new()); }
    // TIME_R1/R8/R32
    for &(d, name) in &[(1usize, "TIME_R1"), (8, "TIME_R8"), (32, "TIME_R32")] {
        let e = graphs.get_mut(name).unwrap();
        for i in 0..n {
            for j in (i + 1)..(i + d + 1).min(n) { add_edge(e, i, j); }
        }
    }
    // ACTIVE/PASSIVE_EXACT
    for &(col_fn, name) in &[
        ((|w: &Window, i: usize| w.active_order[i]) as fn(&Window, usize) -> i64, "ACTIVE_EXACT"),
        ((|w: &Window, i: usize| w.passive_order[i]) as fn(&Window, usize) -> i64, "PASSIVE_EXACT"),
    ] {
        let mut groups: HashMap<i64, Vec<usize>> = HashMap::new();
        for i in 0..n {
            let v = col_fn(win, i);
            if v < 0 { continue; }
            groups.entry(v).or_default().push(i);
        }
        let e = graphs.get_mut(name).unwrap();
        for members in groups.values() {
            for a in 0..members.len() {
                for b in (a + 1)..members.len() { add_edge(e, members[a], members[b]); }
            }
        }
    }
    // PRICE_T0 / PRICE_T2
    let mut by_price: HashMap<u64, Vec<usize>> = HashMap::new();
    for i in 0..n {
        let key = win.price[i].to_bits();
        by_price.entry(key).or_default().push(i);
    }
    let e0 = graphs.get_mut("PRICE_T0").unwrap();
    for members in by_price.values() {
        for a in 0..members.len() {
            for b in (a + 1)..members.len() { add_edge(e0, members[a], members[b]); }
        }
    }
    // PRICE_T2: 按价格排序滑窗
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| win.price[a].partial_cmp(&win.price[b]).unwrap());
    let sp: Vec<f64> = order.iter().map(|&i| win.price[i]).collect();
    let e2 = graphs.get_mut("PRICE_T2").unwrap();
    for a in 0..n {
        for b in (a + 1)..n {
            if sp[b] - sp[a] > 2.0 * PRICE_TICK + 1e-9 { break; }
            add_edge(e2, order[a], order[b]);
        }
    }
    // SIDE_EXACT
    let mut by_side: HashMap<i8, Vec<usize>> = HashMap::new();
    for i in 0..n { by_side.entry(win.active_side[i]).or_default().push(i); }
    let es = graphs.get_mut("SIDE_EXACT").unwrap();
    for members in by_side.values() {
        for a in 0..members.len() {
            for b in (a + 1)..members.len() { add_edge(es, members[a], members[b]); }
        }
    }
    // KNN
    knn_graph(win, &win.book_feats, 14, 10, graphs.get_mut("BOOK_K10").unwrap());
    knn_graph(win, &win.book_feats, 14, 30, graphs.get_mut("BOOK_K30").unwrap());
    knn_graph(win, &win.queue_feats, 10, 10, graphs.get_mut("QUEUE_K10").unwrap());
    knn_graph(win, &win.arrival_feats, 6, 10, graphs.get_mut("ARRIVAL_K10").unwrap());
    graphs
}

// UnionFind on dynamic node set
struct UF { p: HashMap<usize, usize> }
impl UF {
    fn new(nodes: &[usize]) -> Self {
        let mut p = HashMap::new();
        for &n in nodes { p.insert(n, n); }
        UF { p }
    }
    fn find(&mut self, x: usize) -> usize {
        let mut cur = x;
        while self.p[&cur] != cur {
            let nxt = self.p[&cur];
            self.p.insert(cur, nxt);
            cur = nxt;
        }
        cur
    }
    fn union(&mut self, a: usize, b: usize) {
        let ra = self.find(a); let rb = self.find(b);
        if ra != rb { self.p.insert(ra, rb); }
    }
    fn groups(mut self) -> Vec<Vec<usize>> {
        let mut g: HashMap<usize, Vec<usize>> = HashMap::new();
        let keys: Vec<usize> = self.p.keys().copied().collect();
        for k in keys {
            let r = self.find(k);
            g.entry(r).or_default().push(k);
        }
        g.into_values().collect()
    }
}

fn balance(mut a: Vec<usize>, mut b: Vec<usize>) -> (Vec<usize>, Vec<usize>) {
    a.sort(); b.sort();
    while a.len() > b.len() + 1 { b.push(a.pop().unwrap()); }
    while b.len() > a.len() + 1 { a.push(b.pop().unwrap()); }
    (a, b)
}

pub fn graph_bisect(nodes: &[usize], adj: &[Vec<usize>], buf: &mut Vec<usize>) -> (Vec<usize>, Vec<usize>) {
    let mut ns: Vec<usize> = nodes.to_vec();
    ns.sort();
    let n = ns.len();
    if n <= 1 { return (ns, vec![]); }
    // buf[x]=i+1 表示 x 在 ns 中的位置 i；0 表示不存在。调用方保证 buf 容量充足且初始为 0。
    for (i, &x) in ns.iter().enumerate() { buf[x] = i + 1; }
    let mut uf: Vec<usize> = (0..n).collect();
    let mut has_edge = false;
    for &i in &ns {
        let ia = buf[i] - 1;
        for &j in &adj[i] {
            let bj = buf[j];
            if bj != 0 {
                let ib = bj - 1;
                has_edge = true;
                let mut ra = ia; while uf[ra] != ra { uf[ra] = uf[uf[ra]]; ra = uf[ra]; }
                let mut rb = ib; while uf[rb] != rb { uf[rb] = uf[uf[rb]]; rb = uf[rb]; }
                if ra != rb { uf[ra] = rb; }
            }
        }
    }
    for &x in &ns { buf[x] = 0; }
    if !has_edge {
        let a: Vec<usize> = (0..n).step_by(2).map(|i| ns[i]).collect();
        let b: Vec<usize> = (1..n).step_by(2).map(|i| ns[i]).collect();
        return balance(a, b);
    }
    let mut g: HashMap<usize, Vec<usize>> = HashMap::new();
    for (i, &x) in ns.iter().enumerate() {
        let mut r = i; while uf[r] != r { uf[r] = uf[uf[r]]; r = uf[r]; }
        g.entry(r).or_default().push(x);
    }
    let mut comps: Vec<Vec<usize>> = g.into_values().collect();
    comps.sort_by(|a, b| b.len().cmp(&a.len()).then(a.iter().min().cmp(&b.iter().min())));
    let half = n as f64 / 2.0;
    let mut a: Vec<usize> = vec![];
    let mut b: Vec<usize> = vec![];
    for comp in comps {
        if (comp.len() as f64) > half {
            let mut cs = comp.clone(); cs.sort();
            let mid = cs.len() / 2;
            let (left, right) = cs.split_at(mid);
            if a.len() <= b.len() { a.extend_from_slice(left); b.extend_from_slice(right); }
            else { b.extend_from_slice(left); a.extend_from_slice(right); }
        } else {
            if a.len() <= b.len() { a.extend(comp.iter().copied()); }
            else { b.extend(comp.iter().copied()); }
        }
    }
    balance(a, b)
}

fn graph_candidate(g: &[f64], a: &[usize], b: &[usize], high_a: bool) -> Vec<f64> {
    let mut ab: Vec<usize> = a.iter().chain(b.iter()).copied().collect();
    let na = a.len(); let nb = b.len();
    let mut vals: Vec<f64> = ab.iter().map(|&i| g[i]).collect();
    vals.sort_by(|x, y| x.partial_cmp(y).unwrap());
    let (a_vals, b_vals): (Vec<f64>, Vec<f64>) = if high_a {
        (vals[nb..].to_vec(), vals[..nb].to_vec())
    } else {
        (vals[..na].to_vec(), vals[na..].to_vec())
    };
    let mut g2 = g.to_vec();
    let mut ao = a.to_vec();
    ao.sort_by(|&x, &y| g[x].partial_cmp(&g[y]).unwrap()); // stable 保持 A 顺序
    for (k, &i) in ao.iter().enumerate() { g2[i] = a_vals[k]; }
    let mut bo = b.to_vec();
    bo.sort_by(|&x, &y| g[x].partial_cmp(&g[y]).unwrap());
    for (k, &i) in bo.iter().enumerate() { g2[i] = b_vals[k]; }
    g2
}

fn split_blocks(n: usize, n_blocks: usize) -> Vec<Vec<usize>> {
    let base = n / n_blocks; let rem = n % n_blocks;
    let mut blocks = vec![]; let mut s = 0;
    for bi in 0..n_blocks {
        let sz = base + if bi < rem { 1 } else { 0 };
        blocks.push((s..s + sz).collect());
        s += sz;
    }
    blocks
}

fn normalized_kendall(r1: &[f64], r2: &[f64]) -> f64 {
    let n = r1.len();
    if n < 2 { return 0.0; }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| r1[a].partial_cmp(&r1[b]).unwrap());
    let s2: Vec<f64> = order.iter().map(|&i| r2[i]).collect();
    let tot = (n * (n - 1)) / 2;
    let mut disc = 0i64;
    for i in 0..n {
        for j in (i + 1)..n {
            if s2[j] < s2[i] { disc += 1; }
        }
    }
    disc as f64 / tot as f64
}

pub struct S2Branch {
    pub graph: String, pub scope: String,
    pub g: Vec<f64>, pub rho1: f64, pub rho_final: f64,
    pub trajectory: Vec<f64>,
    pub node_ds: Vec<f64>,
    pub accepts: Vec<bool>,
    pub n_queries: usize,
    pub path_distance: f64,
}

fn run_branch(g1: &[f64], y: &[f64], bank: &Bank, adj: &[Vec<usize>], scope: &str, buf: &mut Vec<usize>) -> S2Branch {
    use std::time::Instant;
    let n = y.len();
    let (n_blocks, max_depth) = match scope {
        "LOCAL32" => (32usize, 1usize),
        "MESO8_D2" => (8, 2),
        _ => (1, 4),
    };
    let blocks = if scope == "GLOBAL_D4" { vec![(0..n).collect::<Vec<_>>()] }
                  else { split_blocks(n, n_blocks) };
    let mut g = g1.to_vec();
    let t0 = Instant::now(); let mut rho_cur = bank.query(&g); acc(&T_QUERY, t0.elapsed());
    let rho1 = rho_cur;
    let mut trajectory = vec![rho1];
    let mut node_ds: Vec<f64> = vec![];
    let mut accepts: Vec<bool> = vec![];
    for block in &blocks {
        if block.len() < 2 { continue; }
        let mut cur_level: Vec<Vec<usize>> = vec![block.clone()];
        let mut d = 0;
        while d < max_depth && !cur_level.is_empty() {
            let mut next_level: Vec<Vec<usize>> = vec![];
            for nd in &cur_level {
                if nd.len() < 2 { continue; }
                let t0 = Instant::now(); let (a, b) = graph_bisect(nd, adj, buf); acc(&T_BISECT, t0.elapsed());
                let t0 = Instant::now();
                let g_ha = graph_candidate(&g, &a, &b, true);
                let g_hb = graph_candidate(&g, &a, &b, false);
                acc(&T_CAND, t0.elapsed());
                let t0 = Instant::now();
                let rho_ha = bank.query(&g_ha);
                let rho_hb = bank.query(&g_hb);
                acc(&T_QUERY, t0.elapsed());
                node_ds.push(rho_hb - rho_ha);
                let mut accepted = false;
                if rho_cur >= rho_ha && rho_cur >= rho_hb {}
                else if rho_ha >= rho_hb { g = g_ha; rho_cur = rho_ha; accepted = true; }
                else { g = g_hb; rho_cur = rho_hb; accepted = true; }
                accepts.push(accepted);
                trajectory.push(rho_cur);
                if !a.is_empty() { next_level.push(a); }
                if !b.is_empty() { next_level.push(b); }
            }
            cur_level = next_level; d += 1;
        }
    }
    let t0 = Instant::now(); let rho_final = bank.query(&g); acc(&T_QUERY, t0.elapsed());
    let t0 = Instant::now();
    let r1 = average_rank(g1); let r2 = average_rank(&g);
    let pd = normalized_kendall(&r1, &r2);
    acc(&T_KENDALL, t0.elapsed());
    let n_queries = node_ds.len() * 2;
    S2Branch { graph: String::new(), scope: scope.to_string(), g, rho1, rho_final,
        trajectory, node_ds, accepts, n_queries, path_distance: pd }
}

fn vote(branches: &[S2Branch], beta: f64, y: &[f64], rho1: f64) -> (Vec<f64>, f64) {
    let rhos: Vec<f64> = branches.iter().map(|b| b.rho_final).collect();
    let u: Vec<f64> = rhos.iter().map(|r| (r - rho1) / (1.0 - rho1 + 1e-12)).collect();
    let max_u = u.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = u.iter().map(|x| (beta * x - max_u).exp()).collect();
    let sum_e: f64 = exps.iter().sum();
    let n = y.len();
    let mut ranks = vec![0f64; n];
    for (b, wi) in branches.iter().zip(exps.iter()) {
        let br = average_rank(&b.g);
        for i in 0..n { ranks[i] += (wi / sum_e) * br[i]; }
    }
    let gv = assign_by_rank(&ranks, y);
    let bank = Bank::new(y.to_vec());
    let rv = bank.query(&gv);
    (gv, rv)
}

pub struct Stage2Result {
    pub g1: Vec<f64>, pub rho1: f64,
    pub g: Vec<f64>, pub rho: f64, pub best_name: String,
    pub branches: Vec<S2Branch>,
    pub votes: Vec<(Vec<f64>, f64)>,
}

impl Stage2Result {
    pub fn to_json(&self) -> String {
        let mut s = format!("{{\"rho2\":{:.4},\"s2_best_name\":\"{}\",\"s2_branches\":{{", self.rho, self.best_name);
        for (i, b) in self.branches.iter().enumerate() {
            if i > 0 { s.push(','); }
            s.push_str(&format!("\"{}_{}\":{{\"rho_final\":{:.4},\"path_distance\":{:.4}}}",
                b.graph, b.scope, b.rho_final, b.path_distance));
        }
        s.push_str("},\"s2_votes\":{");
        for (i, beta) in [1,2,5,10].iter().enumerate() {
            if i > 0 { s.push(','); }
            s.push_str(&format!("\"{}\":{:.4}", beta, self.votes[i].1));
        }
        s.push_str("}}");
        s
    }
}

pub fn build_adj_lists(graphs: &HashMap<&'static str, EdgeSet>, n: usize) -> HashMap<&'static str, Vec<Vec<usize>>> {
    let t_adji = std::time::Instant::now();
    let mut adjs: HashMap<&'static str, Vec<Vec<usize>>> = HashMap::new();
    for gname in GRAPH_NAMES.iter() {
        let edges = graphs.get(*gname).unwrap();
        let mut adj = vec![Vec::new(); n];
        for &(u, v) in edges.iter() { adj[u].push(v); adj[v].push(u); }
        adjs.insert(gname, adj);
    }
    acc(&T_ADJ, t_adji.elapsed());
    adjs
}

pub fn run_stage2_with_adjs(g1: &[f64], rho1: f64, y: &[f64], adjs: &HashMap<&'static str, Vec<Vec<usize>>>) -> Stage2Result {
    let bank = Bank::new(y.to_vec());
    let n = y.len();
    let mut id_buf = vec![0usize; n];
    let mut branches: Vec<S2Branch> = vec![];
    for gname in GRAPH_NAMES.iter() {
        let adj = &adjs[*gname];
        for scope in SCOPES.iter() {
            let mut br = run_branch(g1, y, &bank, adj, scope, &mut id_buf);
            br.graph = gname.to_string();
            branches.push(br);
        }
    }
    let gains: Vec<f64> = branches.iter().map(|b| b.rho_final - rho1).collect();
    let mut pos_idx: Vec<usize> = gains.iter().enumerate()
        .filter(|(_, &g)| g > 0.0).map(|(i, _)| i).collect();
    if pos_idx.len() > 24 {
        pos_idx.sort_by(|&a, &b| gains[b].partial_cmp(&gains[a]).unwrap());
        pos_idx.truncate(24);
    }
    let pool: Vec<&S2Branch> = if pos_idx.is_empty() {
        branches.iter().collect()
    } else {
        pos_idx.iter().map(|&i| &branches[i]).collect()
    };
    let t_pool0 = std::time::Instant::now();
    let owned_pool: Vec<S2Branch> = pool.iter().map(|&b| S2Branch {
        graph: b.graph.clone(), scope: b.scope.clone(),
        g: b.g.clone(), rho1: b.rho1, rho_final: b.rho_final,
        trajectory: b.trajectory.clone(), node_ds: b.node_ds.clone(),
        accepts: b.accepts.clone(), n_queries: b.n_queries, path_distance: b.path_distance
    }).collect();
    acc(&T_POOL, t_pool0.elapsed());
    let t_vote0 = std::time::Instant::now();
    let votes = vec![
        vote(&owned_pool, 1.0, y, rho1),
        vote(&owned_pool, 2.0, y, rho1),
        vote(&owned_pool, 5.0, y, rho1),
        vote(&owned_pool, 10.0, y, rho1),
    ];
    acc(&T_VOTE, t_vote0.elapsed());
    let mut best_single = &branches[0];
    for b in &branches[1..] {
        if b.rho_final > best_single.rho_final { best_single = b; }
    }
    let mut best = (String::from("STAGE1"), g1.to_vec(), rho1);
    if best_single.rho_final > best.2 {
        best = (format!("{}_{}", best_single.graph, best_single.scope), best_single.g.clone(), best_single.rho_final);
    }
    for (i, beta) in [1,2,5,10].iter().enumerate() {
        if votes[i].1 > best.2 { best = (format!("VOTE_B{}", beta), votes[i].0.clone(), votes[i].1); }
    }
    Stage2Result { g1: g1.to_vec(), rho1, g: best.1.clone(), rho: best.2, best_name: best.0.clone(), branches, votes }
}

pub fn run_stage2_with_graphs(g1: &[f64], rho1: f64, y: &[f64], graphs: &HashMap<&'static str, EdgeSet>) -> Stage2Result {
    let adjs = build_adj_lists(graphs, y.len());
    run_stage2_with_adjs(g1, rho1, y, &adjs)
}

pub fn run_stage2(g1: &[f64], rho1: f64, win: &Window, y: &[f64]) -> Stage2Result {
    let graphs = build_graphs(win);
    run_stage2_with_graphs(g1, rho1, y, &graphs)
}
