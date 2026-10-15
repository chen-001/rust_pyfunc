"""Compile actual old/new helpers and compare bits on masks, ties, NaN, +/-0 and block edges."""
from pathlib import Path
import subprocess, tempfile, json, math
r=Path(__file__).resolve().parents[1]
baseline=tempfile.TemporaryDirectory()
old=Path(baseline.name)/'src/yupei_pair'
old.mkdir(parents=True)
for name in ['net.rs','reduce.rs','reduce2.rs']:
    (old/name).write_bytes((r/'src/yupei_pair'/name).read_bytes())
subprocess.run(['patch','-R','-p1','-i',str(r/'lossless.patch')],cwd=baseline.name,check=True)
def item(s,name,kind='fn'):
    a=s.index(kind+' '+name); start=s.index('{',a); depth=1; end=start+1
    while depth:
        depth += (s[end]=='{')-(s[end]=='}'); end+=1
    return 'pub '+s[a:end]+'\n'
pre='use rayon::prelude::*; use std::time::Instant; struct C{thr:f64} fn soft()->C{C{thr:2.0}}\n'
mods=[]
for label,base in [('old',old),('new',r/'src/yupei_pair')]:
    a=(base/'reduce.rs').read_text();b=(base/'reduce2.rs').read_text()
    text=pre+item(a,'R1Role','struct')+item(a,'r1_role')
    text+=''.join(item(b,n)for n in ['row_stats16','solve_lin','row_stats_resid8','resid_stats'])
    text+=item(a,'topk_adj')
    if label=='new':
        text+=''.join(item(a,n)for n in ['topk_rows','topk_adj_from_rows','topk_adj_multi'])+item(b,'row_stats4')
        simd=(r/'src/yupei_pair/simd.rs').read_text()
        text+='mod simd {'+item(simd,'fkey')+item(simd,'pack_desc_idx_desc')+'}'
    mods.append('mod '+label+'{'+text+'}')
main=r'''
fn bits(a:&[f64],b:&[f64]) {assert_eq!(a.len(),b.len());for(x,y)in a.iter().zip(b){assert_eq!(x.to_bits(),y.to_bits());}}
fn main(){
 for n in [1usize,2,7,20,31,257] {
  for case in 0..4 {
   let ok:Vec<bool>=(0..n).map(|i|case==0||i%7!=0).collect();
   let row:Vec<f32>=(0..n).map(|j|match (j+case)%11{0=>f32::NAN,1=>-0.0,2=>0.0,3=>f32::INFINITY,_=>(j%13) as f32-6.0}).collect();
   for i in 0..n{let a=old::row_stats16(&row,i,n,&ok);let b=new::row_stats4(&row,i,n,&ok);for k in [0,6,8,11]{assert_eq!(a[k].to_bits(),b[k].to_bits());}}
   let mat:Vec<f32>=(0..n*n).map(|j|row[(j/n+j%n)%n]).collect();
   for k in [0,1,5,20,100] {let a=old::topk_adj(n,&mat,k,&ok);let b=new::topk_adj_multi(n,&mat,&[k],&ok);assert_eq!(a,b[0]);let rows=new::topk_rows(n,&mat,100,&ok);let shared=new::topk_adj_from_rows(n,&rows,&[k]);assert_eq!(a,shared[0]);}
   if n>2 {
    let x0:Vec<i32>=(0..n*n).map(|j|((j+1)%n) as i32).collect();
    let mut mask=ok.clone();mask.fill(true);
    let a=old::r1_role(n,&x0,&mat,&mat,&mask);let b=new::r1_role(n,&x0,&mat,&mat,&mask);
    bits(&a.count,&b.count);bits(&a.gain_mean,&b.gain_mean);bits(&a.z_mean,&b.z_mean);bits(&a.gain_max,&b.z_max);
   }
  }
  if n<7{continue;}
  let ok:Vec<_>=(0..n).map(|i|i%19!=0).collect();
  let rho:Vec<f32>=(0..n*n).map(|j|((j*19)%173) as f32/173.0).collect();
  let ln:Vec<_>=(0..n).map(|i|(i as f64+1.0).ln()).collect();let le:Vec<_>=(0..n).map(|i|(i as f64+2.0).sqrt()).collect();
  for different_masks in [false,true] {
   let weights:Vec<Vec<f32>>=(0..5).map(|k|(0..n*n).map(|j|if j%37==0 || (different_masks && k==2 && j%17==0){f32::NAN}else{(((j*31+k*11)%103) as f32-50.0)/73.0}).collect()).collect();
   let same=(0..n).all(|i|!ok[i]||(i+1..n).all(|j|!ok[j]||weights.iter().all(|w|w[i*n+j].is_finite()==weights[0][i*n+j].is_finite())));
   assert_eq!(same,!different_masks);
   let mut cache=None;
   for w in &weights{if !same {cache=None;}
    let a=old::resid_stats(n,w,&rho,&ln,&le,&ok);let b=new::resid_stats(n,w,&rho,&ln,&le,&ok,&mut cache);
    for(x,y)in a.iter().zip(b.iter()){bits(x,y);}
   }
  }
 }
 let v=[3.0f32,-1.0,7.0,0.0,2.0,9.0];let a=new::row_stats4(&v,0,6,&[true;6]);println!("{:?}",[a[0],a[6],a[8],a[11]]);
}
'''
deps=Path('/home/chenzongwei/rust_pyfunc/target/release-fast/deps')
rayon=next(deps.glob('librayon-*.rlib'))
with tempfile.TemporaryDirectory() as d:
    p=Path(d)/'edge.rs';p.write_text('#![allow(dead_code)]\n'+'\n'.join(mods)+main)
    subprocess.run(['/home/chenzongwei/.cargo/bin/rustc','--edition=2021','-O','-C','target-cpu=native','-L',str(deps),'--extern',f'rayon={rayon}',str(p),'-o',d+'/edge'],check=True)
    result=subprocess.check_output([d+'/edge'],text=True)
    v=sorted([-1.,7.,0.,2.,9.]);expected=[sum(v)/5,v[math.floor(.9*4+.5)],v[-1],sum(v)/5]
    assert json.loads(result)==expected,(result,expected)
print('PASS: row statistics/Python reference, graph ties/masks, shared explanation stats, regression common/different masks and 256-row block boundaries; raw f64 bits equal.')
