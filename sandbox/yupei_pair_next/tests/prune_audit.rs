//! Read-only audit instrumentation. Output sampling does not alter any factor.
use super::net::NetMats;
use std::io::Write;
pub fn capture(m: &NetMats, n: usize, ok: &[bool], date: i64, name: &str) {
    let Ok(root) = std::env::var("YUPEI_PRUNE_AUDIT") else { return; };
    let dir = std::path::Path::new(&root).join(date.to_string());
    std::fs::create_dir_all(&dir).unwrap();
    let fields: Vec<(&str, &[f32])> = vec![
        ("rho",&m.rho),("g",&m.g),("gs",&m.gs),("z",&m.z),("zs",&m.zs),
        ("ru",&m.ru),("rus",&m.rus),("sg",&m.sg),("sgs",&m.sgs),
        ("x0",&m.x0),("x0s",&m.x0s),("grel",&m.grel),("gsh",&m.gsh),
        ("gcov",&m.gcov),("gnpos",&m.gnpos),("ghhi",&m.ghhi),
        ("pb",&m.pb),("qc",&m.qc),("pbs",&m.pbs),("qcs",&m.qcs),
        ("exz",&m.exz),("exzs",&m.exzs)];
    let mut seed=741289u64;
    let mut draw=|| { seed ^= seed<<13; seed ^= seed>>7; seed ^= seed<<17; seed as usize % n };
    let mut pairs=Vec::new();
    while pairs.len()<4096 { let a=draw(); let b=draw(); if a!=b && ok[a] && ok[b] { pairs.push((a.min(b),a.max(b))); } }
    let mut bytes=Vec::new();
    for (_,v) in &fields { for &(b,c) in &pairs { bytes.extend_from_slice(&v[b*n+c].to_le_bytes()); } }
    std::fs::write(dir.join(format!("{name}_matrices.bin")),bytes).unwrap();
    let mut count=0u64; let mut different=[0u64;5]; let mut masks_differ=0u64;
    for b in 0..n { if !ok[b] {continue;} for c in b+1..n { if !ok[c] {continue;}
        let k=b*n+c;count+=1;
        for (j,(a,v)) in [(m.x0[k],m.x0s[k]),(m.g[k],m.gs[k]),(m.sg[k],m.pb[k].max(m.qc[k])),(m.sgs[k],m.pbs[k].max(m.qcs[k])),(m.ru[k],m.sg[k]+m.g[k])].iter().enumerate() {
            different[j]+=(a.to_bits()!=v.to_bits()) as u64;
        }
        let mask=m.g[k].is_finite();
        masks_differ+=([m.gs[k],m.z[k],m.zs[k],m.zs[k]-m.z[k]].iter().any(|v|v.is_finite()!=mask)) as u64;
    }}
    let mut triplebytes=Vec::new();let mut triple_total=0u64;let mut triple_diff=0u64;
    for &(b,c) in &pairs[..16] {
        let sb=m.e[b].sqrt();let sc=m.e[c].sqrt();
        let d=(m.e[b]+m.e[c]+2.0*(m.rho[b*n+c] as f64)*sb*sc).sqrt();
        let wb=(sb/d) as f32;let wc=(sc/d) as f32;
        let t=(2.0f32*m.rho[b*n+c])*(sb as f32)*(sc as f32);
        let den=(m.e[b]+m.e[c]+t as f64).max(1e-30).sqrt() as f32;
        let inv=1.0f32/den;
        let mut hs=Vec::new();let mut gs=Vec::new();
        for x in 0..n {
            if !ok[x] || x==b || x==c { hs.push(f32::NEG_INFINITY);gs.push(f32::NEG_INFINITY);continue; }
            let p=m.rho[x*n+b];let q=m.rho[x*n+c];
            let h=wb*p+wc*q; let g=h-p.max(q);
            let h2=(p*(sb as f32)+q*(sc as f32))*inv;
            triple_total+=1;triple_diff+=(h.to_bits()!=h2.to_bits()) as u64;
            hs.push(h);gs.push(g);
        }
        for v in hs.iter().chain(gs.iter()) {triplebytes.extend_from_slice(&v.to_le_bytes());}
    }
    std::fs::write(dir.join(format!("{name}_hg.bin")),triplebytes).unwrap();
    let summary=serde_json::json!({"date":date,"mat":name,"n":n,"active":ok.iter().filter(|&&x|x).count(),"fields":fields.iter().map(|v|v.0).collect::<Vec<_>>(),"samples":pairs,"all_valid_pairs":count,"differences":{"x0_x0s":different[0],"g_gs":different[1],"sg_max_pb_qc":different[2],"sgs_max_pbs_qcs":different[3],"ru_sg_plus_g":different[4]},"residual_valid_mask_differences":masks_differ,"sample_triples":triple_total,"net_r2_bits_differ":triple_diff});
    let mut f=std::fs::File::create(dir.join(format!("{name}.json"))).unwrap();
    write!(f,"{}",summary).unwrap();
}
