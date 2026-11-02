"""Opt-in invariant loads for exact binary loops; apply after binary_predecessor_patch."""
from pathlib import Path
p=Path(__file__).parent/'src/relaynet_fast.rs';s=p.read_text()
if 'fn binary_backprop_hoisted(' in s: raise SystemExit('binary invariant experiment already applied')
start=s.index('// BEGIN EXACT BINARY DIRECTION EXPERIMENT');end=s.index('// END EXACT BINARY DIRECTION EXPERIMENT',start)
f=s[start:end]
f=f.replace('record_parents:bool,','record_parents:bool,hoist_sigma:bool,',1)
f=f.replace('record_parents:false}','record_parents:false,hoist_sigma:false}',1)
f=f.replace('''                    let u=self.queue[qi];
                    for &v in''','''                    let u=self.queue[qi];
                    let sigma_u=self.sigma[u];
                    for &v in''',1)
f=f.replace('''                            self.sigma[v]+=self.sigma[u];''','''                            self.sigma[v]+=if self.hoist_sigma{sigma_u}else{self.sigma[u]};''',1)
f=f.replace('''    let record_predecessors=std::env::var_os("RELAYNET_BINARY_PREDECESSORS").is_some();''','''    let record_predecessors=std::env::var_os("RELAYNET_BINARY_PREDECESSORS").is_some();
    let hoist=std::env::var_os("RELAYNET_BINARY_HOIST").is_some();''',1)
f=f.replace('''        scratch.record_parents=record_predecessors;''','''        scratch.record_parents=record_predecessors;scratch.hoist_sigma=hoist;''',1)
f=f.replace('''                    let mut link=scratch.pred_head[u];
                    while link''','''                    let mut link=scratch.pred_head[u];
                    let sigma_u=scratch.sigma[u];let factor=1.0+scratch.delta[u];
                    while link''',1)
f=f.replace('''                        scratch.delta[v]+=(scratch.sigma[v]/scratch.sigma[u])*(1.0+scratch.delta[u]);''','''                        scratch.delta[v]+=(scratch.sigma[v]/sigma_u)*factor;''',1)
f=f.replace('''                binary_backprop_scalar(adj,&scratch.dist,&scratch.sigma,&mut scratch.delta,u);''','''                if hoist {binary_backprop_hoisted(adj,&scratch.dist,&scratch.sigma,&mut scratch.delta,u);}
                else {binary_backprop_scalar(adj,&scratch.dist,&scratch.sigma,&mut scratch.delta,u);}''',1)
s=s[:start]+f+s[end:]
s+=r'''
#[inline]
fn binary_backprop_hoisted(adj:&[u32],dist:&[i32],sigma:&[f64],delta:&mut[f64],u:usize){
    let parent_depth=dist[u]-1;
    let sigma_u=sigma[u];let factor=1.0+delta[u];
    for &v in adj{
        let v=v as usize;
        if dist[v]==parent_depth{delta[v]+=(sigma[v]/sigma_u)*factor;}
    }
}
'''
p.write_text(s)
print('Applied opt-in RELAYNET_BINARY_HOIST; source/target invariants retain arithmetic order')
