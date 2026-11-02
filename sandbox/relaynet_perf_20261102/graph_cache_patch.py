"""Reuse complete statistics only after exact adjacency equality (hash is just an index)."""
from pathlib import Path
p = Path(__file__).parent / "src/relaynet_fast.rs"
s = p.read_text()
if "let mut graph_stats_cache:" in s:
    raise SystemExit("graph cache already applied")
s = s.replace("    let mut _dbg_names = Vec::new();", '''    let mut _dbg_names = Vec::new();
    let mut graph_stats_cache: BTreeMap<(bool, u64), Vec<(Vec<Vec<(u32, f32)>>, Vec<Vec<f64>>)>> = BTreeMap::new();''')
s = s.replace("                let mut cols = stat_undir(&g, sim, &inds, &mkt_corr, &lead_set, sp.full);", '''                let hash = {
                    use std::hash::{Hash, Hasher};
                    let mut h = std::collections::hash_map::DefaultHasher::new();
                    g.n.hash(&mut h);
                    for row in &g.adj {
                        row.len().hash(&mut h);
                        for &(v, w) in row { v.hash(&mut h); w.to_bits().hash(&mut h); }
                    }
                    h.finish()
                };
                let cache = graph_stats_cache.entry((sp.full, hash)).or_default();
                let mut cols = if let Some((_, cols)) = cache.iter().find(|(adj, _)| *adj == g.adj) {
                    eprintln!("[rlg-cache] {} exact_graph_hit", sp.tag);
                    cols.clone()
                } else {
                    let cols = stat_undir(&g, sim, &inds, &mkt_corr, &lead_set, sp.full);
                    cache.push((g.adj.clone(), cols.clone()));
                    cols
                };''')
p.write_text(s)
