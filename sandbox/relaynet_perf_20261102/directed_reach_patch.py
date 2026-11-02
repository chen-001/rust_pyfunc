"""Exact directed two-hop bitsets and parallel PageRank; apply after directed_patch."""
from pathlib import Path

p = Path(__file__).parent / "src/relaynet_fast.rs"
s = p.read_text()
if "fn dir_two_hop_exact(" in s:
    if "    let outreach = dir_two_hop_exact(&g.out_adj);" in s:
        raise SystemExit("directed local patch already applied")
    # graph_patch can restore dir_stats while leaving appended helpers intact.
    # Remove that stale helper block, then derive fresh legacy references below.
    lo = s.index("// BEGIN EXACT DIRECTED LOCAL")
    hi = s.index("// END EXACT DIRECTED LOCAL", lo) + len("// END EXACT DIRECTED LOCAL")
    s = s[:lo] + s[hi:]
start = s.index("fn dir_stats(")
end = s.index("\n/// 沿有向边", start)
f = s[start:end]
pr_start = f.index("    // 有向 PageRank")
pr_end = f.index("    let netpr:", pr_start)
old_pr = f[pr_start:pr_end]
reach_start = f.index("    let mut outreach =")
reach_end = f.index("    vec![", reach_start)
old_reach = f[reach_start:reach_end]
f = f[:reach_start] + '''    let reach_clock = std::time::Instant::now();
    let outreach = dir_two_hop_exact(&g.out_adj);
    let inreach = dir_two_hop_exact(&g.in_adj);
    if profile_detail { eprintln!("[rlg-dir-part] reach {:.6}",reach_clock.elapsed().as_secs_f64()); }
''' + f[reach_end:]
f = f[:pr_start] + '''    let profile_detail = std::env::var_os("RLG_DETAIL_PROFILE").is_some();
    let pr_clock = std::time::Instant::now();
    let (pr,pr2) = dir_pagerank_pair_exact(g,&outstr,&instr);
    if profile_detail { eprintln!("[rlg-dir-part] pagerank {:.6}",pr_clock.elapsed().as_secs_f64()); }
''' + f[pr_end:]
f = f.replace("    let (outh, inh) = dir_harmonic_pair_exact(g);", '''    let harmonic_clock = std::time::Instant::now();
    let (outh, inh) = dir_harmonic_pair_exact(g);
    if profile_detail { eprintln!("[rlg-dir-part] harmonic {:.6}",harmonic_clock.elapsed().as_secs_f64()); }''')
f = f.replace("    let n = g.n;", "    let n = g.n;\n    rlg_dump_digraph(g);", 1)
s = s[:start] + f + s[end:]
s += "\n// BEGIN EXACT DIRECTED LOCAL\n"
s += "fn dir_pagerank_legacy(g: &DGraph,outstr: &[f64],instr: &[f64]) -> (Vec<f64>,Vec<f64>) {\n    let n=g.n;\n" + old_pr + "    (pr,pr2)\n}\n"
s += "fn dir_reach_legacy(g: &DGraph) -> (Vec<f64>,Vec<f64>) {\n    let n=g.n;\n" + old_reach + "    (outreach,inreach)\n}\n"
s += r'''
fn dir_two_hop_exact(adj: &[Vec<(u32,f32)>]) -> Vec<f64> {
    let n = adj.len();
    if n == 0 { return Vec::new(); }
    let nw = n.div_ceil(64);
    let mut bits = vec![0u64;n*nw];
    bits.par_chunks_mut(nw).enumerate().for_each(|(i,row)| {
        for &(v,_) in &adj[i] { row[v as usize / 64] |= 1u64 << (v as usize % 64); }
    });
    (0..n).into_par_iter().map_init(|| vec![0u64;nw], |ball,i| {
        ball.copy_from_slice(&bits[i*nw..(i+1)*nw]);
        ball[i/64] |= 1u64 << (i%64);
        for &(v,_) in &adj[i] {
            let v = v as usize;
            if adj[v].len() < nw {
                for &(w,_) in &adj[v] { ball[w as usize / 64] |= 1u64 << (w as usize % 64); }
            } else {
                for (out,&word) in ball.iter_mut().zip(&bits[v*nw..(v+1)*nw]) { *out |= word; }
            }
        }
        ball.iter().map(|word| word.count_ones() as usize).sum::<usize>() as f64 - 1.0
    }).collect()
}

fn dir_pagerank_pair_exact(g: &DGraph,outstr: &[f64],instr: &[f64]) -> (Vec<f64>,Vec<f64>) {
    let n = g.n;
    // Sorted incoming source IDs reproduce the corrected serial push sum order.
    let incoming: Vec<_> = g.in_adj.iter().map(|a| { let mut a=a.clone(); a.sort_by_key(|&(v,_)|v); a }).collect();
    let reverse_incoming: Vec<_> = g.out_adj.iter().map(|a| { let mut a=a.clone(); a.sort_by_key(|&(v,_)|v); a }).collect();
    let mut pr = vec![1.0/n as f64;n]; let mut pr2=pr.clone();
    let mut next=vec![0.0;n]; let mut next2=vec![0.0;n];
    for _ in 0..50 {
        let (mut dangling,mut dangling2)=(0.0,0.0);
        for i in 0..n {
            if outstr[i]<=0.0 { dangling+=pr[i]; }
            if instr[i]<=0.0 { dangling2+=pr2[i]; }
        }
        next.par_iter_mut().zip(next2.par_iter_mut()).enumerate().for_each(|(i,(out,out2))| {
            let mut value=0.15/n as f64;
            for &(src,w) in &incoming[i] {
                let src=src as usize;
                if outstr[src]<=0.0 { continue; }
                value += 0.85*pr[src]*w as f64/outstr[src];
            }
            *out=value+0.85*dangling/n as f64;
            let mut value=0.15/n as f64;
            for &(src,w) in &reverse_incoming[i] {
                let src=src as usize;
                if instr[src]<=0.0 { continue; }
                value += 0.85*pr2[src]*w as f64/instr[src];
            }
            *out2=value+0.85*dangling2/n as f64;
        });
        std::mem::swap(&mut pr,&mut next); std::mem::swap(&mut pr2,&mut next2);
    }
    (pr,pr2)
}

fn directed_local_result(g: &DGraph,old: bool) -> Vec<Vec<f64>> {
    let outstr: Vec<f64>=g.out_adj.iter().map(|a|a.iter().map(|&(_,w)|w as f64).sum()).collect();
    let instr: Vec<f64>=g.in_adj.iter().map(|a|a.iter().map(|&(_,w)|w as f64).sum()).collect();
    let (pr,pr2)=if old { dir_pagerank_legacy(g,&outstr,&instr) } else { dir_pagerank_pair_exact(g,&outstr,&instr) };
    let (outreach,inreach)=if old { dir_reach_legacy(g) } else { (dir_two_hop_exact(&g.out_adj),dir_two_hop_exact(&g.in_adj)) };
    vec![outreach,inreach,pr,pr2]
}

pub fn directed_local_probe(n: usize,edges: Vec<(usize,usize,f32)>,old: bool) -> Vec<Vec<f64>> {
    let mut g=DGraph{n,out_adj:vec![Vec::new();n],in_adj:vec![Vec::new();n]};
    for (u,v,w) in edges { g.out_adj[u].push((v as u32,w));g.in_adj[v].push((u as u32,w)); }
    directed_local_result(&g,old)
}

fn rlg_dump_digraph(g: &DGraph) {
    let Some(path)=std::env::var_os("RLG_DUMP_DIGRAPH_PATH") else{return;};
    static ONCE:std::sync::Once=std::sync::Once::new();
    ONCE.call_once(|| {
        use std::io::Write;
        let mut f=std::io::BufWriter::new(std::fs::File::create(path).unwrap());
        f.write_all(&(g.n as u32).to_le_bytes()).unwrap();
        for row in &g.out_adj {
            f.write_all(&(row.len() as u32).to_le_bytes()).unwrap();
            for &(v,w) in row { f.write_all(&v.to_le_bytes()).unwrap(); f.write_all(&w.to_le_bytes()).unwrap(); }
        }
        f.flush().unwrap();
    });
}

pub fn directed_local_graph_probe(path: &str,old: bool) -> std::io::Result<Vec<Vec<f64>>> {
    use std::io::Read;
    let mut f=std::io::BufReader::new(std::fs::File::open(path)?);
    fn read4(f:&mut impl Read)->std::io::Result<[u8;4]>{let mut b=[0;4];f.read_exact(&mut b)?;Ok(b)}
    let n=u32::from_le_bytes(read4(&mut f)?) as usize;
    let mut g=DGraph{n,out_adj:Vec::with_capacity(n),in_adj:vec![Vec::new();n]};
    for u in 0..n {
        let count=u32::from_le_bytes(read4(&mut f)?) as usize;
        let mut row=Vec::with_capacity(count);
        for _ in 0..count {
            let v=u32::from_le_bytes(read4(&mut f)?);let w=f32::from_le_bytes(read4(&mut f)?);
            row.push((v,w));g.in_adj[v as usize].push((u as u32,w));
        }
        g.out_adj.push(row);
    }
    Ok(directed_local_result(&g,old))
}
// END EXACT DIRECTED LOCAL
'''
p.write_text(s)
lib=p.parent/'lib.rs'
ls=lib.read_text()
if 'fn directed_local_probe(' not in ls:
    ls += r'''

#[pyfunction]
fn directed_local_probe(py:Python,n:usize,edges:Vec<(usize,usize,f32)>,old:bool)->Vec<Vec<f64>> {
    py.allow_threads(||relaynet_fast::directed_local_probe(n,edges,old))
}

#[pyfunction]
fn directed_local_graph_probe(py:Python,path:&str,old:bool)->PyResult<Vec<Vec<f64>>> {
    py.allow_threads(||relaynet_fast::directed_local_graph_probe(path,old))
        .map_err(|e|pyo3::exceptions::PyIOError::new_err(e.to_string()))
}
'''
    anchor='    m.add_function(wrap_pyfunction!(run, m)?)?;'
    ls=ls.replace(anchor,anchor+'\n    m.add_function(wrap_pyfunction!(directed_local_probe, m)?)?;\n    m.add_function(wrap_pyfunction!(directed_local_graph_probe, m)?)?;',1)
    lib.write_text(ls)
print('Applied exact directed bitset reach + parallel pull PageRank + detailed timers')
