"""Verify graph-cache collisions by weight bits, including signed zero."""
from pathlib import Path

p = Path(__file__).parent / 'src/relaynet_fast.rs'
s = p.read_text()
if 'fn relay_graph_adjacency_equal(' in s:
    raise SystemExit('graph cache bit equality already applied')
assert s.count('*adj == g.adj') == 1
assert s.count('*adj==g.adj') == 1
assert s.count('previous.adj==g.adj') == 1
s = s.replace('*adj == g.adj', 'relay_graph_adjacency_equal(adj, &g.adj)')
s = s.replace('*adj==g.adj', 'relay_graph_adjacency_equal(adj, &g.adj)')
s = s.replace('previous.adj==g.adj', 'relay_graph_adjacency_equal(&previous.adj, &g.adj)')
s += '''
/// Hash collisions are verified with exact IDs and weight bits; +0 and -0 differ.
fn relay_graph_adjacency_equal(a:&[Vec<(u32,f32)>],b:&[Vec<(u32,f32)>])->bool {
    a.len()==b.len() && a.iter().zip(b).all(|(x,y)| x.len()==y.len() &&
        x.iter().zip(y).all(|(&(vx,wx),&(vy,wy))|vx==vy && wx.to_bits()==wy.to_bits()))
}
'''
p.write_text(s)
print('Graph-cache collision checks now compare f32 bits')
