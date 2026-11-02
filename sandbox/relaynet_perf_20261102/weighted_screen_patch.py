"""Production candidate: compact AVX2 screening, opt-in and scalar fallback."""
from pathlib import Path
import re
p=Path(__file__).parent/'src/relaynet_fast.rs';s=p.read_text()
a=s.index('    fn source(&mut self, g: &UGraph,');b=s.index('\n}\n\nfn exact_paths(',a)
method=s[a:b]
header=method[:method.index('{')]
method=method.replace(header, '    fn source_screen4<const WEIGHTED: bool>(&mut self, g: &UGraph, src: usize, tree: bool, bet: &mut [f64]) -> [f64; 7] ',1)
method=re.sub(r'\bweighted\b','WEIGHTED',method)
method=re.sub(r'\bparents\b','true',method)
start=method.index('                for &(v, sim) in &g.adj[u] {',method.index('            self.heap.push('))
brace=method.index('{',start);depth=1;end=brace+1
while depth:depth+=(method[end]=='{')-(method[end]=='}');end+=1
loop=method[start:end]
loop=loop.replace('                for &(v, sim) in &g.adj[u] {', '''                for chunk in g.adj[u].chunks(4) {
                    let mut mask=if chunk.len()==4 {
                        unsafe {weighted_screen4(du,chunk,self.dist.as_ptr())}
                    } else {(1u32<<chunk.len())-1};
                    while mask!=0 {
                        let lane=mask.trailing_zeros() as usize;mask&=mask-1;
                        let (v,sim)=chunk[lane];''',1)+'\n                }'
method=method[:start]+loop+method[end:]
a=s.index('fn exact_paths_mode(');brace=s.index('{',a);depth=1;b=brace+1
while depth:depth+=(s[b]=='{')-(s[b]=='}');b+=1
runner=s[a:b]
runner=runner.replace('fn exact_paths_mode(g: &UGraph, weighted: bool, parents: bool)',
    'fn exact_paths_screen4(g: &UGraph, weighted: bool)',1)
runner=runner.replace('    let n = g.n;', '''    if !weighted_screen_guard(g) {return exact_paths_mode(g,weighted,true);}
    let n = g.n;''',1)
runner=runner.replace('scratch.source(g, src, tree, weighted, parents, &mut bet)',
    'if weighted {scratch.source_screen4::<true>(g,src,tree,&mut bet)} else {scratch.source_screen4::<false>(g,src,tree,&mut bet)}')
helper=r'''
// WEIGHTED_SCREEN_BEGIN
/// Hardware and unique-target guard make the SIMD lane assumptions explicit.
/// The fallback is the independently tested exact scalar implementation.
fn weighted_screen_guard(g:&UGraph)->bool {
    if g.n > i32::MAX as usize {return false;}
    #[cfg(target_arch="x86_64")]
    if !std::is_x86_feature_detected!("avx2"){return false;}
    #[cfg(not(target_arch="x86_64"))]
    {return false;}
    let mut seen=vec![usize::MAX;g.n];
    for (i,row) in g.adj.iter().enumerate() {
        for &(v,_) in row {
            let v=v as usize;
            if v>=g.n || seen[v]==i {return false;}
            seen[v]=i;
        }
    }
    true
}
impl PathScratch {
METHOD
}
RUNNER
// WEIGHTED_SCREEN_END
'''.replace('METHOD',method).replace('RUNNER',runner)
if '// WEIGHTED_SCREEN_BEGIN' in s:
    a=s.index('// WEIGHTED_SCREEN_BEGIN');b=s.index('// WEIGHTED_SCREEN_END',a)+len('// WEIGHTED_SCREEN_END')
    s=s[:a]+helper.strip()+s[b:]
else:s+='\n'+helper
old='fn exact_paths(g: &UGraph, weighted: bool) -> ExactPaths { exact_paths_mode(g, weighted, true) }'
new='''fn exact_paths(g: &UGraph, weighted: bool) -> ExactPaths {
    if std::env::var("RELAYNET_WEIGHTED_SIMD").as_deref()==Ok("1") {exact_paths_screen4(g,weighted)}
    else {exact_paths_mode(g,weighted,true)}
}'''
if old in s:s=s.replace(old,new,1)
# Mode64 explicitly benchmarks this compact candidate; all other experiment
# modes remain unchanged and cannot enable this kernel without the flag above.
s=s.replace('let result=exact_paths_experiment(&g,true,true,mode);',
    'let result=if mode==64 {exact_paths_screen4(&g,true)} else {exact_paths_experiment(&g,true,true,mode)};')
p.write_text(s)
print('Added compact AVX2 candidate; enable only with RELAYNET_WEIGHTED_SIMD=1')
