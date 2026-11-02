"""Exact, invocation-local reuse of unchanged top-k selections across graphs."""
from pathlib import Path
p = Path(__file__).parent/'src/relaynet_fast.rs'
s = p.read_text()
if '// CANDIDATE_CACHE_BEGIN' in s:
    raise SystemExit('already applied')
s=s.replace('    let g_main = build_thresh_limited(bank.sim("rd", Sess::Day).as_ref(), 0.20, &live);', '''    let mut candidate_cache = RelayCandidateCache::new(&live);
    let main_sim = bank.sim("rd", Sess::Day);
    let main_bits = candidate_cache.get(&main_sim, Kind::Thr(0.20));
    let g_main = match main_bits.as_deref() {
        Some(bits) => relay_graph_from_bits(main_sim.as_ref(), bits, false, Some(0.20)),
        None => build_thresh_limited(main_sim.as_ref(), 0.20, &live),
    };''',1)
s=s.replace('''        for c in cols {
            for i in 0..n {
                vals[i * f_total + col] = c[i];
            }
            col += 1;
        }''','''        if std::env::var_os("RELAYNET_ROW_OUTPUT").is_some() {
            for i in 0..n {
                let row = &mut vals[i * f_total + col .. i * f_total + col + cols.len()];
                for (k, source) in cols.iter().enumerate() { row[k] = source[i]; }
            }
            col += cols.len();
        } else {
            for c in cols {
                for i in 0..n { vals[i * f_total + col] = c[i]; }
                col += 1;
            }
        }''',1)
s=s.replace('''        let sim: &Array2<f64> = sim_rc.as_ref();
        match sp.kind {''','''        let sim: &Array2<f64> = sim_rc.as_ref();
        let candidate_bits = candidate_cache.get(&sim_rc, sp.kind);
        match sp.kind {''',1)
s=s.replace('''Kind::Thr(t) => build_thresh_limited(sim, t, &live),''','''Kind::Thr(t) => match candidate_bits.as_deref() {
                        Some(bits) => relay_graph_from_bits(sim, bits, false, Some(t)),
                        None => build_thresh_limited(sim, t, &live),
                    },''',1)
for kind, mutual in [('KnnU','false'),('KnnM','true')]:
    s=s.replace(f'let g0 = build_knn(sim, k, {mutual});',f'''let g0 = match candidate_bits.as_deref() {{
                            Some(bits) => relay_graph_from_bits(sim, bits, {mutual}, None),
                            None => build_knn(sim, k, {mutual}),
                        }};''',1)
s=s.replace('''            let refs: Vec<&Array2<f64>> = matrices.iter().map(|m| m.as_ref()).collect();
            let outputs = relay_graph_batch(specs, &refs, &live, &inds,
                &mkt_corr, &lead_set, &mut graph_stats_cache);''','''            let refs: Vec<&Array2<f64>> = matrices.iter().map(|m| m.as_ref()).collect();
            let candidate_owners: Vec<_> = matrices.iter().zip(specs).map(|(matrix,sp)|
                candidate_cache.get(matrix,sp.kind)).collect();
            let candidate_refs: Vec<_> = candidate_owners.iter().map(|b| b.as_deref().map(Vec::as_slice)).collect();
            let outputs = relay_graph_batch(specs, &refs, &live, &inds,
                &mkt_corr, &lead_set, &mut graph_stats_cache, Some(&candidate_refs));''',1)
s=s.replace('''inds: &[i64], mkt: &[f64], leaders: &[bool], cache: &mut RelayGraphStatsCache)
    -> Vec<RelayGraphOutput>''','''inds: &[i64], mkt: &[f64], leaders: &[bool], cache: &mut RelayGraphStatsCache,
    candidates: Option<&[Option<&[Vec<u64>]>]>) -> Vec<RelayGraphOutput>''',1)
s=s.replace('''        let clock=std::time::Instant::now(); let sim=sims[i];
        let graph=match sp.kind {''','''        let clock=std::time::Instant::now(); let sim=sims[i];
        let bits=candidates.and_then(|entries| entries[i]);
        let graph=match sp.kind {''',1)
s=s.replace('''Kind::Thr(t)=>build_thresh_limited(sim,t,live),''','''Kind::Thr(t)=>match bits { Some(b)=>relay_graph_from_bits(sim,b,false,Some(t)),
                        None=>build_thresh_limited(sim,t,live) },''',1)
for kind, mutual in [('KnnU','false'),('KnnM','true')]:
    s=s.replace(f'Kind::{kind}(k)=>prune_dead(&build_knn(sim,k,{mutual}),live),',f'''Kind::{kind}(k)=>prune_dead(&match bits {{
                        Some(b)=>relay_graph_from_bits(sim,b,{mutual},None),
                        None=>build_knn(sim,k,{mutual}) }},live),''',1)
# The existing small batch probe tests cache reuse, thresholds, kNN and dead nodes.
s=s.replace('''    let mut cache=RelayGraphStatsCache::new();let mut all=Vec::new();
    for chunk in specs.chunks(batch.max(1)) {''','''    let mut cache=RelayGraphStatsCache::new();let mut all=Vec::new();
    let sim=Rc::new(sim);let directed=Rc::new(directed);
    let mut candidate_cache=RelayCandidateCache::new(&live);
    for chunk in specs.chunks(batch.max(1)) {''',1)
s=s.replace('''        for result in relay_graph_batch(chunk,&refs,&live,&inds,&mkt,&leaders,&mut cache) { all.extend(result.columns); }''','''        let owners:Vec<_>=chunk.iter().map(|sp|candidate_cache.get(
            if matches!(sp.kind,Kind::DirThr(_)){&directed}else{&sim},sp.kind)).collect();
        let bits:Vec<_>=owners.iter().map(|b|b.as_deref().map(Vec::as_slice)).collect();
        for result in relay_graph_batch(chunk,&refs,&live,&inds,&mkt,&leaders,&mut cache,Some(&bits)) { all.extend(result.columns); }''',1)
s += '''
// CANDIDATE_CACHE_BEGIN
/// Scoped to one immutable day's matrices and live mask. Retaining each matrix
/// owner prevents address reuse; threshold and union/mutual mode do not affect
/// the directed top-k selection. No cache crosses dates or Python calls.
struct RelayCandidateCache<'a> {
    live:&'a [bool], enabled:bool,
    entries:BTreeMap<(usize,usize,bool),(Rc<Array2<f64>>,Rc<Vec<Vec<u64>>>)>,
}
impl<'a> RelayCandidateCache<'a> {
    fn new(live:&'a [bool])->Self { Self { live,
        enabled:std::env::var_os("RELAYNET_CANDIDATE_CACHE").is_some(),entries:BTreeMap::new() } }
    fn get(&mut self, matrix:&Rc<Array2<f64>>, kind:Kind)->Option<Rc<Vec<Vec<u64>>>> {
        if !self.enabled {return None;}
        let n=matrix.nrows();
        let (k,limited)=match kind {
            Kind::Thr(_)=>((n/25).max(200).min(n.saturating_sub(1)).max(1),true),
            Kind::KnnU(k)|Kind::KnnM(k)=>(k,false),
            _=>return None,
        };
        let key=(Rc::as_ptr(matrix) as usize,k,limited);
        if let Some((_,bits))=self.entries.get(&key){return Some(bits.clone());}
        let bits=Rc::new(relay_candidate_bits(matrix.as_ref(),k,if limited {Some(self.live)}else{None}));
        self.entries.insert(key,(matrix.clone(),bits.clone()));Some(bits)
    }
}
// CANDIDATE_CACHE_END
'''
s=s.replace("if matches!(sp.kind,Kind::DirThr(_)){&directed}else{&sim}).collect();", "if matches!(sp.kind,Kind::DirThr(_)){directed.as_ref()}else{sim.as_ref()}).collect();")
p.write_text(s)
print('Applied opt-in candidate selection cache and contiguous output writes')
