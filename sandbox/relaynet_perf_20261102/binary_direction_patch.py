"""Opt-in exact per-layer push/pull BFS; apply after binary_frontier_patch."""
from pathlib import Path

p=Path(__file__).parent/'src/relaynet_fast.rs'
s=p.read_text()
if 'fn binary_direction_run(' in s:
    raise SystemExit('binary direction experiment already applied')
old='''    binary_betweenness_run_mode(g, request_simd, std::env::var_os("RELAYNET_BINARY_FULL_FRONTIER").is_none())'''
new='''    let early_stop=std::env::var_os("RELAYNET_BINARY_FULL_FRONTIER").is_none();
    if std::env::var_os("RELAYNET_BINARY_DIRECTION").is_some() {
        binary_direction_run(g,request_simd,early_stop)
    } else { binary_betweenness_run_mode(g,request_simd,early_stop) }'''
assert old in s
s=s.replace(old,new,1)
s+=r'''
// BEGIN EXACT BINARY DIRECTION EXPERIMENT
struct BinaryDirectionGraph {
    offsets:Vec<usize>, neighbors:Vec<u32>, component:Vec<u32>, sizes:Vec<u32>,
    component_edges:Vec<usize>, sorted:bool,
}
impl BinaryDirectionGraph {
    fn new(g:&UGraph)->Self {
        let (component,sizes)=components(g);
        let mut component_edges=vec![0usize;sizes.len()];
        let mut offsets=Vec::with_capacity(g.n+1);
        let mut neighbors=Vec::with_capacity(g.adj.iter().map(Vec::len).sum());
        let mut sorted=true;
        offsets.push(0);
        for (i,row) in g.adj.iter().enumerate() {
            sorted &= row.windows(2).all(|pair|pair[0].0<pair[1].0);
            component_edges[component[i] as usize]+=row.len();
            neighbors.extend(row.iter().map(|&(v,_)|v)); offsets.push(neighbors.len());
        }
        Self{offsets,neighbors,component,sizes,component_edges,sorted}
    }
}

struct BinaryDirectionScratch {
    dist:Vec<i32>,sigma:Vec<f64>,delta:Vec<f64>,queue:Vec<usize>,rank:Vec<usize>,
    parents:Vec<usize>,found:Vec<(usize,usize)>,
}
impl BinaryDirectionScratch {
    fn new(n:usize)->Self {
        Self{dist:vec![-1;n],sigma:vec![0.0;n],delta:vec![0.0;n],queue:Vec::with_capacity(n),
            rank:vec![usize::MAX;n],parents:Vec::new(),found:Vec::with_capacity(n)}
    }
    fn bfs(&mut self,g:&BinaryDirectionGraph,src:usize,early_stop:bool,force_pull:bool) {
        self.dist.fill(-1);self.sigma.fill(0.0);self.delta.fill(0.0);self.queue.clear();self.rank.fill(usize::MAX);
        self.dist[src]=0;self.sigma[src]=1.0;self.queue.push(src);self.rank[src]=0;
        let cid=g.component[src] as usize;
        let mut remaining=g.sizes[cid] as usize-1;
        let mut unvisited_edges=g.component_edges[cid]-(g.offsets[src+1]-g.offsets[src]);
        let mut begin=0;
        while begin<self.queue.len() {
            if early_stop && remaining==0 { break; }
            let end=self.queue.len();
            let depth=self.dist[self.queue[begin]];
            let frontier_edges:usize=self.queue[begin..end].iter().map(|&v|g.offsets[v+1]-g.offsets[v]).sum();
            if g.sorted && remaining>0 && (force_pull || frontier_edges>unvisited_edges.saturating_mul(4)) {
                self.found.clear();
                for v in 0..self.dist.len() {
                    if self.dist[v]>=0 || g.component[v] as usize!=cid { continue; }
                    self.parents.clear();
                    for &u in &g.neighbors[g.offsets[v]..g.offsets[v+1]] {
                        let u=u as usize;
                        if self.dist[u]==depth { self.parents.push(u); }
                    }
                    if self.parents.is_empty() { continue; }
                    // The push kernel visits parents in queue order, not vertex-ID order.
                    self.parents.sort_unstable_by_key(|&u|self.rank[u]);
                    let mut paths=0.0;
                    for &u in &self.parents { paths+=self.sigma[u]; }
                    self.sigma[v]=paths;self.dist[v]=depth+1;
                    self.found.push((self.rank[self.parents[0]],v));
                    remaining-=1;unvisited_edges-=g.offsets[v+1]-g.offsets[v];
                }
                // With strictly sorted adjacency, discovery is lexicographic:
                // first parent queue position, then target vertex ID.
                self.found.sort_unstable();
                for &(_,v) in &self.found {self.rank[v]=self.queue.len();self.queue.push(v);}
            } else {
                for qi in begin..end {
                    let u=self.queue[qi];
                    for &v in &g.neighbors[g.offsets[u]..g.offsets[u+1]] {
                        let v=v as usize;
                        if self.dist[v]<0 {
                            self.dist[v]=depth+1;self.rank[v]=self.queue.len();self.queue.push(v);
                            remaining-=1;unvisited_edges-=g.offsets[v+1]-g.offsets[v];
                        }
                        if self.dist[v]==depth+1 {self.sigma[v]+=self.sigma[u];}
                    }
                }
            }
            begin=end;
        }
    }
}

fn binary_direction_run(g:&UGraph,request_simd:bool,early_stop:bool)->Vec<f64> {
    if let Some(bet)=forest_bet(g){return bet;}
    let n=g.n;
    let graph=BinaryDirectionGraph::new(g);
    let force_pull=std::env::var_os("RELAYNET_BINARY_DIRECTION_FORCE").is_some();
    #[cfg(target_arch="x86_64")]
    let use_simd=request_simd&&std::is_x86_feature_detected!("avx512f")&&std::is_x86_feature_detected!("avx2");
    #[cfg(not(target_arch="x86_64"))]
    let use_simd=false;
    const BLOCK:usize=16;
    let parts:Vec<Vec<f64>>=(0..n.div_ceil(BLOCK)).into_par_iter().map(|block|{
        let mut scratch=BinaryDirectionScratch::new(n);let mut bet=vec![0.0;n];
        for src in block*BLOCK..((block+1)*BLOCK).min(n) {
            if graph.offsets[src]==graph.offsets[src+1]{continue;}
            scratch.bfs(&graph,src,early_stop,force_pull);
            for &u in scratch.queue.iter().rev() {
                let adj=&graph.neighbors[graph.offsets[u]..graph.offsets[u+1]];
                #[cfg(target_arch="x86_64")]
                if use_simd&&adj.len()>=8 {
                    unsafe{binary_backprop_avx512(adj,&scratch.dist,&scratch.sigma,&mut scratch.delta,u);}
                    if u!=src{bet[u]+=scratch.delta[u];}continue;
                }
                binary_backprop_scalar(adj,&scratch.dist,&scratch.sigma,&mut scratch.delta,u);
                if u!=src{bet[u]+=scratch.delta[u];}
            }
        }
        bet
    }).collect();
    let mut bet=vec![0.0;n];
    for part in parts{for(a,b)in bet.iter_mut().zip(part){*a+=b;}}
    for b in &mut bet{*b*=0.5;}bet
}

pub fn binary_direction_state_probe(n:usize,edges:Vec<(usize,usize,f32)>,src:usize,direction:bool,force_pull:bool)
    ->(Vec<usize>,Vec<f64>,Vec<i32>)
{
    let mut g=UGraph{n,adj:vec![Vec::new();n]};
    for(u,v,w)in edges{g.adj[u].push((v as u32,w));g.adj[v].push((u as u32,w));}
    let graph=BinaryDirectionGraph::new(&g);
    let mut scratch=BinaryDirectionScratch::new(n);
    if direction {scratch.bfs(&graph,src,true,force_pull);}
    else {
        scratch.dist[src]=0;scratch.sigma[src]=1.0;scratch.queue.push(src);
        let mut head=0;
        while head<scratch.queue.len(){
            let u=scratch.queue[head];head+=1;
            for &(v,_) in &g.adj[u] {
                let v=v as usize;
                if scratch.dist[v]<0{scratch.dist[v]=scratch.dist[u]+1;scratch.queue.push(v);}
                if scratch.dist[v]==scratch.dist[u]+1{scratch.sigma[v]+=scratch.sigma[u];}
            }
        }
    }
    (scratch.queue,scratch.sigma,scratch.dist)
}
// END EXACT BINARY DIRECTION EXPERIMENT
'''
p.write_text(s)
lib=p.parent/'lib.rs';ls=lib.read_text()
if 'fn binary_direction_state_probe(' not in ls:
    ls+=r'''

#[pyfunction]
fn binary_direction_state_probe(py:Python,n:usize,edges:Vec<(usize,usize,f32)>,src:usize,direction:bool,force_pull:bool)
    ->(Vec<usize>,Vec<f64>,Vec<i32>)
{
    py.allow_threads(||relaynet_fast::binary_direction_state_probe(n,edges,src,direction,force_pull))
}
'''
    anchor='    m.add_function(wrap_pyfunction!(run, m)?)?;'
    ls=ls.replace(anchor,anchor+'\n    m.add_function(wrap_pyfunction!(binary_direction_state_probe, m)?)?;',1)
    lib.write_text(ls)
print('Applied opt-in exact layer push/pull; enable RELAYNET_BINARY_DIRECTION=1 only after benchmark')
