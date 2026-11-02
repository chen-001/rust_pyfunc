"""Optional whole-graph scheduling on the existing Rayon pool; default stays serial."""
from pathlib import Path
p=Path(__file__).parent/'src/relaynet_fast.rs';s=p.read_text()
start=s.index('    for sp in GRAPH_SPECS.iter() {', s.index('let mut graph_stats_cache:'))
end=s.index('    eprintln!("[rlg-stage] graphs',start)
if '// GRAPH_BATCH_LOOP_BEGIN' not in s:
    serial=s[start:end].rstrip()
    loop=r'''
    // GRAPH_BATCH_LOOP_BEGIN
    let graph_batch = std::env::var("RLG_GRAPH_BATCH").ok()
        .and_then(|v| v.parse::<usize>().ok()).unwrap_or(1).clamp(1, 8);
    if graph_batch == 1 {
SERIAL_LOOP
    } else {
        for specs in GRAPH_SPECS.chunks(graph_batch) {
            let batch_started = std::time::Instant::now();
            // SimBank owns non-Send Rc/RefCell caches. Populate them on this
            // thread, then share only immutable matrix references with Rayon.
            let mut sim_seconds = Vec::with_capacity(specs.len());
            let matrices: Vec<Rc<Array2<f64>>> = specs.iter().map(|sp| {
                let clock = std::time::Instant::now();
                let matrix = bank.sim(sp.fam, sp.sess);
                sim_seconds.push(clock.elapsed().as_secs_f64());
                matrix
            }).collect();
            let refs: Vec<&Array2<f64>> = matrices.iter().map(|m| m.as_ref()).collect();
            let outputs = relay_graph_batch(specs, &refs, &live, &inds,
                &mkt_corr, &lead_set, &mut graph_stats_cache);
            for (i, output) in outputs.into_iter().enumerate() {
                let sp = &specs[i];
                if output.cached { eprintln!("[rlg-cache] {} exact_graph_hit", sp.tag); }
                if let Some((edges, nodes)) = output.size {
                    eprintln!("[rlg-graph] {} edges={} nodes={}", sp.tag, edges, nodes);
                }
                let expect = spec_n_factors(sp);
                push_cols(output.columns, expect, &mut _dbg_names);
                eprintln!("[rlg-time] {} sim_sec={:.6} graph_sec={:.6} total_sec={:.6}",
                    sp.tag, sim_seconds[i], output.seconds, sim_seconds[i] + output.seconds);
            }
            eprintln!("[rlg-batch] graphs={} wall_sec={:.6} threads={}",
                specs.len(), batch_started.elapsed().as_secs_f64(), rayon::current_num_threads());
        }
    }
    // GRAPH_BATCH_LOOP_END

'''.replace('SERIAL_LOOP',serial)
    s=s[:start]+loop+s[end:]
helpers=r'''
// GRAPH_BATCH_HELPERS_BEGIN
type RelayGraphStatsCache = BTreeMap<(bool, u64), Vec<(Vec<Vec<(u32, f32)>>, Vec<Vec<f64>>)>>;

enum RelayGraphWork { Undirected(UGraph), Directed(DGraph) }
struct RelayGraphOutput {
    columns: Vec<Vec<f64>>, seconds: f64,
    size: Option<(usize, usize)>, cached: bool,
}
fn relay_graph_hash(g: &UGraph) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h=std::collections::hash_map::DefaultHasher::new();
    g.n.hash(&mut h);
    for row in &g.adj {
        row.len().hash(&mut h);
        for &(v,w) in row { v.hash(&mut h); w.to_bits().hash(&mut h); }
    }
    h.finish()
}

/// Parallelize independent graphs within the existing bounded compute pool.
/// All cache lookups/inserts and output assembly remain in specification order.
fn relay_graph_batch(specs: &[GraphSpec], sims: &[&Array2<f64>], live: &[bool],
    inds: &[i64], mkt: &[f64], leaders: &[bool], cache: &mut RelayGraphStatsCache)
    -> Vec<RelayGraphOutput>
{
    let prepared: Vec<(RelayGraphWork, u64, f64)> = specs.par_iter().enumerate().map(|(i,sp)| {
        let clock=std::time::Instant::now(); let sim=sims[i];
        let graph=match sp.kind {
            Kind::DirThr(t)=>RelayGraphWork::Directed(build_digraph(sim,t)),
            _=> {
                let g=match sp.kind {
                    Kind::Thr(t)=>build_thresh_limited(sim,t,live),
                    Kind::KnnU(k)=>prune_dead(&build_knn(sim,k,false),live),
                    Kind::KnnM(k)=>prune_dead(&build_knn(sim,k,true),live),
                    Kind::Mst=>prune_dead(&build_mst(sim),live),
                    Kind::IndThr(t)=>prune_dead(&build_ind_thresh(sim,inds,t),live),
                    Kind::DirThr(_)=>unreachable!(),
                };
                RelayGraphWork::Undirected(g)
            }
        };
        let hash=match &graph { RelayGraphWork::Undirected(g)=>relay_graph_hash(g), _=>0 };
        (graph,hash,clock.elapsed().as_secs_f64())
    }).collect();
    let count=specs.len();
    let mut base: Vec<Option<Vec<Vec<f64>>>>=(0..count).map(|_|None).collect();
    let mut duplicate=vec![None;count];
    let mut cached=vec![false;count];
    for i in 0..count {
        if let RelayGraphWork::Undirected(g)=&prepared[i].0 {
            let key=(specs[i].full,prepared[i].1);
            if let Some((_,cols))=cache.get(&key).and_then(|bucket| bucket.iter().find(|(adj,_)| *adj==g.adj)) {
                base[i]=Some(cols.clone()); cached[i]=true;
                continue;
            }
            duplicate[i]=(0..i).find(|&j| {
                specs[j].full==specs[i].full && prepared[j].1==prepared[i].1 &&
                matches!(&prepared[j].0,RelayGraphWork::Undirected(previous) if previous.adj==g.adj)
            });
            cached[i]=duplicate[i].is_some();
        }
    }
    let mut computed: Vec<Option<(Vec<Vec<f64>>,f64)>>=(0..count).into_par_iter().map(|i| {
        if cached[i] { return None; }
        let clock=std::time::Instant::now();
        let cols=match &prepared[i].0 {
            RelayGraphWork::Undirected(g)=>stat_undir(g,sims[i],inds,mkt,leaders,specs[i].full),
            RelayGraphWork::Directed(g)=>dir_stats(g),
        };
        Some((cols,clock.elapsed().as_secs_f64()))
    }).collect();
    let mut seconds: Vec<f64>=prepared.iter().map(|(_,_,time)|*time).collect();
    for i in 0..count {
        if let Some(j)=duplicate[i] {
            base[i]=Some(base[j].as_ref().expect("earlier batch result").clone());
        } else if let Some((cols,time))=computed[i].take() {
            seconds[i]+=time;
            if let RelayGraphWork::Undirected(g)=&prepared[i].0 {
                cache.entry((specs[i].full,prepared[i].1)).or_default().push((g.adj.clone(),cols.clone()));
            }
            base[i]=Some(cols);
        }
    }
    // MST extras reuse full-graph harmonics exactly as in the serial path.
    base.into_par_iter().enumerate().map(|(i,cols)| {
        let clock=std::time::Instant::now(); let mut cols=cols.expect("computed graph result");
        let size=match &prepared[i].0 {
            RelayGraphWork::Undirected(g)=> {
                if matches!(specs[i].kind,Kind::Mst) {
                    let extra=mst_extra(g,mkt,if specs[i].full {Some(cols[9].as_slice())} else {None});
                    cols.extend(extra);
                }
                let mut edges=0;let mut nodes=0;
                for u in 0..g.n {if live[u] && !g.adj[u].is_empty(){nodes+=1;edges+=g.adj[u].len();}}
                Some((edges/2,nodes))
            },
            _=>None,
        };
        RelayGraphOutput { columns:cols, seconds:seconds[i]+clock.elapsed().as_secs_f64(),size,cached:cached[i] }
    }).collect()
}
// GRAPH_BATCH_HELPERS_END
'''
if '// GRAPH_BATCH_HELPERS_BEGIN' in s:
    a=s.index('// GRAPH_BATCH_HELPERS_BEGIN');b=s.index('// GRAPH_BATCH_HELPERS_END',a)+len('// GRAPH_BATCH_HELPERS_END')
    s=s[:a]+helpers.strip()+s[b:]
else:s+='\n'+helpers
p.write_text(s)
print('Applied optional graph batches on same Rayon pool; RLG_GRAPH_BATCH defaults to 1')
