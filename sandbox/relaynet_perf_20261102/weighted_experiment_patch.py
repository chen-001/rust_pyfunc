"""Append isolated exact weighted-path experiments; production selection unchanged."""
from pathlib import Path
p=Path(__file__).parent/'src/relaynet_fast.rs';s=p.read_text()
# Copy the measured implementation so every experimental arithmetic change can
# be checked against the unchanged implementation in the very same binary.
a=s.index('    fn source(&mut self, g: &UGraph,');b=s.index('\n}\n\nfn exact_paths(',a)
method=s[a:b]
method=method.replace('fn source(', 'fn source_experiment(',1)
method=method.replace('bet: &mut [f64]) -> [f64; 7]', 'bet: &mut [f64], experiment: &WeightedExperiment) -> [f64; 7]',1)
method=method.replace('        self.dist.fill(f64::INFINITY); self.dist[src] = 0.0;', '''        let mode=experiment.mode;
        let mut unseen=if mode&4!=0 {experiment.component_size[src]-1} else {usize::MAX};
        let mut maximum=if unseen==0 {0.0} else {f64::INFINITY};
        let mut popped=0usize;
        self.dist.fill(f64::INFINITY); self.dist[src] = 0.0;''',1)
method=method.replace('                if weighted { self.order.push(u); }', '''                if weighted { self.order.push(u); }
                popped+=1;
                if mode&4!=0 && unseen==0 {
                    if mode&8!=0 && popped%128==0 {
                        maximum=self.dist.iter().copied().filter(|v|v.is_finite()).fold(0.0f64,f64::max);
                    }
                    // Monotonic floating-point addition guarantees every
                    // outgoing candidate exceeds every current target bound.
                    // Strict > keeps every exactly tied shortest path.
                    if du+experiment.minimum[u]>maximum {continue;}
                }''',1)
# Forward relaxation body appears only once in the original method.
start=method.index('                for &(v, sim) in &g.adj[u] {',method.index('            self.heap.push('))
brace=method.index('{',start);depth=1;end=brace+1
while depth:depth+=(method[end]=='{')-(method[end]=='}');end+=1
old=method[start:end]
modified=old.replace('                    if nd < self.dist[v] {', '''                    if nd < self.dist[v] {
                        if mode&4!=0 && self.dist[v].is_infinite() {unseen-=1;}''',1)
modified=modified.replace('                        self.dist[v] = nd;', '''                        self.dist[v] = nd;
                        if mode&4!=0 && unseen==0 && maximum.is_infinite() {
                            maximum=self.dist.iter().copied().filter(|v|v.is_finite()).fold(0.0f64,f64::max);
                        }''',1)
csr=modified.replace('for &(v, sim) in &g.adj[u] {\n                    let v = v as usize;\n                    let nd = du + (1.0 - sim as f64).max(1e-9);',
'''for edge in experiment.offsets[u]..experiment.offsets[u+1] {
                    let v = experiment.targets[edge] as usize;
                    let nd = du + experiment.costs[edge];''',1)
screened=modified.replace('                for &(v, sim) in &g.adj[u] {', '''                for chunk in g.adj[u].chunks(if mode&32!=0 {8} else {4}) {
                    let mask=if chunk.len()==8 {
                        unsafe {weighted_screen8(du,chunk,self.dist.as_ptr())}
                    } else if chunk.len()==4 {
                        unsafe {weighted_screen4(du,chunk,self.dist.as_ptr())}
                    } else {(1u32<<chunk.len())-1};
                    let mut remaining_lanes=mask;
                    while remaining_lanes!=0 {
                        let lane=remaining_lanes.trailing_zeros() as usize;
                        remaining_lanes&=remaining_lanes-1;
                        let (v,sim)=chunk[lane];''',1)+'\n                }'
method=method[:start]+'                if mode&2!=0 {\n'+csr+'\n                } else if mode&48!=0 {\n'+screened+'\n                } else {\n'+modified+'\n                }'+method[end:]
method=method.replace('        self.values.sort_unstable_by(|a, b| a.total_cmp(b));', '''        if mode&1!=0 {
            let count=self.values.len() as f64;
            let quantiles=weighted_select_quantiles(&mut self.values);
            return [h/count,f/count,e,quantiles[0],quantiles[1],quantiles[2],quantiles[3]];
        }
        self.values.sort_unstable_by(|a, b| a.total_cmp(b));''',1)
a=s.index('fn exact_paths_mode(');brace=s.index('{',a);depth=1;b=brace+1
while depth:depth+=(s[b]=='{')-(s[b]=='}');b+=1
runner=s[a:b]
runner=runner.replace('fn exact_paths_mode(g: &UGraph, weighted: bool, parents: bool)',
    'fn exact_paths_experiment(g: &UGraph, weighted: bool, parents: bool, mode: u8)',1)
runner=runner.replace('    let n = g.n;', '    let n = g.n;\n    let experiment=WeightedExperiment::new(g,mode);',1)
runner=runner.replace('scratch.source(g, src, tree, weighted, parents, &mut bet)',
    'scratch.source_experiment(g, src, tree, weighted, parents, &mut bet, &experiment)')
helper=r'''
// WEIGHTED_EXPERIMENT_BEGIN
struct WeightedExperiment {
    mode:u8, offsets:Vec<usize>, targets:Vec<u32>, costs:Vec<f64>,
    minimum:Vec<f64>, component_size:Vec<usize>,
}
impl WeightedExperiment {
    fn new(g:&UGraph,mode:u8)->Self {
        let mut mode=mode;
        #[cfg(target_arch="x86_64")]
        {
            if !std::is_x86_feature_detected!("avx2"){mode&=!16;}
            if !std::is_x86_feature_detected!("avx512f") || !std::is_x86_feature_detected!("avx2"){mode&=!32;}
        }
        #[cfg(not(target_arch="x86_64"))] {mode&=!48;}
        let mut result=Self {mode,offsets:Vec::new(),targets:Vec::new(),costs:Vec::new(),minimum:Vec::new(),component_size:Vec::new()};
        if mode&2!=0 {
            result.offsets.push(0);
            for row in &g.adj {
                for &(v,w) in row {result.targets.push(v);result.costs.push((1.0-w as f64).max(1e-9));}
                result.offsets.push(result.targets.len());
            }
        }
        if mode&4!=0 {
            result.minimum=g.adj.iter().map(|a|a.iter().map(|&(_,w)|(1.0-w as f64).max(1e-9)).fold(f64::INFINITY,f64::min)).collect();
            let (labels,sizes)=components(g);
            result.component_size=labels.iter().map(|&c|sizes[c as usize] as usize).collect();
        }
        result
    }
}

/// Select only the eight required order statistics. All values are finite and
/// positive; interpolation uses exactly the original operations and constants.
fn weighted_select_quantiles(values:&mut [f64])->[f64;4] {
    fn select(v:&mut [f64],base:usize,positions:&[usize]) {
        if positions.is_empty(){return;}
        let middle=positions.len()/2;let position=positions[middle]-base;
        let (left,_,right)=v.select_nth_unstable_by(position,|a,b|a.total_cmp(b));
        select(left,base,&positions[..middle]);
        select(right,base+position+1,&positions[middle+1..]);
    }
    let qs=[0.25,0.75,0.50,0.90];
    let mut positions=[0usize;8];
    for (i,&q) in qs.iter().enumerate(){let lo=(q*(values.len()-1) as f64).floor() as usize;positions[2*i]=lo;positions[2*i+1]=(lo+1).min(values.len()-1);}
    positions.sort_unstable();
    let mut distinct=0;
    for i in 0..8 {if distinct==0||positions[i]!=positions[distinct-1]{positions[distinct]=positions[i];distinct+=1;}}
    select(values,0,&positions[..distinct]);
    std::array::from_fn(|i|quantile_sorted(values,qs[i]))
}


// Screening may only reject an edge whose candidate is strictly greater than
// the current target distance. Accepted lanes retain the original scalar order.
// Even if targets repeat, later decreases cannot make a rejected lane useful.
#[cfg(target_arch="x86_64")]
#[target_feature(enable="avx2")]
#[inline]
unsafe fn weighted_screen4(du:f64,a:&[(u32,f32)],d:*const f64)->u32 {
    use std::arch::x86_64::*;
    let index=_mm_set_epi32(a[3].0 as i32,a[2].0 as i32,a[1].0 as i32,a[0].0 as i32);
    let sim=_mm256_cvtps_pd(_mm_set_ps(a[3].1,a[2].1,a[1].1,a[0].1));
    let cost=_mm256_max_pd(_mm256_sub_pd(_mm256_set1_pd(1.0),sim),_mm256_set1_pd(1e-9));
    let candidate=_mm256_add_pd(_mm256_set1_pd(du),cost);
    let target=_mm256_i32gather_pd::<8>(d,index);
    _mm256_movemask_pd(_mm256_cmp_pd::<_CMP_LE_OQ>(candidate,target)) as u32
}
#[cfg(target_arch="x86_64")]
#[target_feature(enable="avx512f")]
#[inline]
unsafe fn weighted_screen8(du:f64,a:&[(u32,f32)],d:*const f64)->u32 {
    use std::arch::x86_64::*;
    let index=_mm256_set_epi32(a[7].0 as i32,a[6].0 as i32,a[5].0 as i32,a[4].0 as i32,a[3].0 as i32,a[2].0 as i32,a[1].0 as i32,a[0].0 as i32);
    let sim=_mm512_cvtps_pd(_mm256_set_ps(a[7].1,a[6].1,a[5].1,a[4].1,a[3].1,a[2].1,a[1].1,a[0].1));
    let cost=_mm512_max_pd(_mm512_sub_pd(_mm512_set1_pd(1.0),sim),_mm512_set1_pd(1e-9));
    let candidate=_mm512_add_pd(_mm512_set1_pd(du),cost);
    let target=_mm512_i32gather_pd::<8>(index,d);
    _mm512_cmp_pd_mask::<_CMP_LE_OQ>(candidate,target) as u32
}
#[cfg(not(target_arch="x86_64"))]
unsafe fn weighted_screen4(_:f64,_:&[(u32,f32)],_:*const f64)->u32 {15}
#[cfg(not(target_arch="x86_64"))]
unsafe fn weighted_screen8(_:f64,_:&[(u32,f32)],_:*const f64)->u32 {255}

impl PathScratch {
METHOD
}

RUNNER

pub fn exact_path_experiment_probe(n:usize,edges:Vec<(usize,usize,f32)>,mode:u8)->Vec<Vec<f64>> {
    let mut g=UGraph{n,adj:vec![Vec::new();n]};
    for (u,v,w) in edges {g.adj[u].push((v as u32,w));g.adj[v].push((u as u32,w));}
    for a in &mut g.adj {a.sort_by_key(|&(v,_)|v);}
    let result=exact_paths_experiment(&g,true,true,mode);
    let mut out=Vec::from(result.columns);out.push(result.bet);out
}

pub fn weighted_graph_experiment_probe(path:&str,mode:u8)->std::io::Result<Vec<Vec<f64>>> {
    use std::io::Read;
    let mut f=std::io::BufReader::new(std::fs::File::open(path)?);
    fn read4(f:&mut impl Read)->std::io::Result<[u8;4]>{let mut b=[0;4];f.read_exact(&mut b)?;Ok(b)}
    let n=u32::from_le_bytes(read4(&mut f)?) as usize;
    let mut g=UGraph{n,adj:Vec::with_capacity(n)};
    for _ in 0..n {
        let count=u32::from_le_bytes(read4(&mut f)?) as usize;let mut row=Vec::with_capacity(count);
        for _ in 0..count {row.push((u32::from_le_bytes(read4(&mut f)?),f32::from_le_bytes(read4(&mut f)?)));}
        g.adj.push(row);
    }
    let result=exact_paths_experiment(&g,true,true,mode);
    let mut out=Vec::from(result.columns);out.push(result.bet);Ok(out)
}
// WEIGHTED_EXPERIMENT_END
'''.replace('METHOD',method).replace('RUNNER',runner)
if '// WEIGHTED_EXPERIMENT_BEGIN' in s:
    a=s.index('// WEIGHTED_EXPERIMENT_BEGIN');b=s.index('// WEIGHTED_EXPERIMENT_END',a)+len('// WEIGHTED_EXPERIMENT_END')
    s=s[:a]+helper.strip()+s[b:]
else:s+='\n'+helper
p.write_text(s)
print('Appended weighted experiments only; production paths unchanged')
