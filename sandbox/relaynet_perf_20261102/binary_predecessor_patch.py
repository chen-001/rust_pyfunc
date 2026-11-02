"""Opt-in binary Brandes predecessor lists; apply after binary_direction_patch."""
from pathlib import Path

p=Path(__file__).parent/'src/relaynet_fast.rs'
s=p.read_text()
if 'record_parents:bool' in s:
    raise SystemExit('binary predecessor experiment already applied')
start=s.index('// BEGIN EXACT BINARY DIRECTION EXPERIMENT')
end=s.index('// END EXACT BINARY DIRECTION EXPERIMENT',start)
f=s[start:end]
old='    parents:Vec<usize>,found:Vec<(usize,usize)>, '
if old not in f: old=old.rstrip()
assert old in f
f=f.replace(old,old+'''\n    pred_head:Vec<u32>,pred_from:Vec<u32>,pred_next:Vec<u32>,record_parents:bool,''',1)
old='rank:vec![usize::MAX;n],parents:Vec::new(),found:Vec::with_capacity(n)}'
assert old in f
f=f.replace(old,'''rank:vec![usize::MAX;n],parents:Vec::new(),found:Vec::with_capacity(n),
            pred_head:vec![u32::MAX;n],pred_from:Vec::new(),pred_next:Vec::new(),record_parents:false}''',1)
old='''        self.dist.fill(-1);self.sigma.fill(0.0);self.delta.fill(0.0);self.queue.clear();self.rank.fill(usize::MAX);'''
assert old in f
f=f.replace(old,old+'''
        if self.record_parents {self.pred_head.fill(u32::MAX);self.pred_from.clear();self.pred_next.clear();}''',1)
old='''                    self.sigma[v]=paths;self.dist[v]=depth+1;'''
assert old in f
f=f.replace(old,old+'''
                    if self.record_parents {
                        for &u in &self.parents {
                            self.pred_from.push(u as u32);self.pred_next.push(self.pred_head[v]);
                            self.pred_head[v]=(self.pred_from.len()-1) as u32;
                        }
                    }''',1)
old='''                        if self.dist[v]==depth+1 {self.sigma[v]+=self.sigma[u];}'''
assert old in f
f=f.replace(old,'''                        if self.dist[v]==depth+1 {
                            self.sigma[v]+=self.sigma[u];
                            if self.record_parents {
                                self.pred_from.push(u as u32);self.pred_next.push(self.pred_head[v]);
                                self.pred_head[v]=(self.pred_from.len()-1) as u32;
                            }
                        }''',1)
old='''    let force_pull=std::env::var_os("RELAYNET_BINARY_DIRECTION_FORCE").is_some();'''
assert old in f
f=f.replace(old,old+'''
    let record_predecessors=std::env::var_os("RELAYNET_BINARY_PREDECESSORS").is_some();''',1)
old='''        let mut scratch=BinaryDirectionScratch::new(n);let mut bet=vec![0.0;n];'''
assert old in f
f=f.replace(old,old+'''
        scratch.record_parents=record_predecessors;''',1)
old='''            for &u in scratch.queue.iter().rev() {
                let adj=&graph.neighbors[graph.offsets[u]..graph.offsets[u+1]];'''
assert old in f
f=f.replace(old,'''            for &u in scratch.queue.iter().rev() {
                if record_predecessors {
                    // For a fixed u, each predecessor updates a distinct delta[v].
                    // Reordering these independent writes leaves each target's
                    // additions in exactly the original reverse-queue order.
                    let mut link=scratch.pred_head[u];
                    while link!=u32::MAX {
                        let v=scratch.pred_from[link as usize] as usize;
                        scratch.delta[v]+=(scratch.sigma[v]/scratch.sigma[u])*(1.0+scratch.delta[u]);
                        link=scratch.pred_next[link as usize];
                    }
                    if u!=src {bet[u]+=scratch.delta[u];}
                    continue;
                }
                let adj=&graph.neighbors[graph.offsets[u]..graph.offsets[u+1]];''',1)
s=s[:start]+f+s[end:]
p.write_text(s)
print('Applied opt-in binary predecessor experiment; set RELAYNET_BINARY_PREDECESSORS=1 with DIRECTION=1')
