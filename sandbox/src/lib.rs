//! 开发沙箱：anneal_volume 性能分析。
//! 单版本（与主项目一致），加入逐阶段计时。

use memmap2::Mmap;
use ndarray::Array2;
use pyo3::prelude::*;
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;
use std::time::Instant;

// ============================================================================
const N_FACTORS: usize = 25;
const M_MAX_SCALAR: usize = 500_000;
const M_MAX_MINUTE: usize = 50_000;
const C1_FRAC: f64 = 1.0;
const SEED: u64 = 0x9E37_79B9_7F4A_7C15;
const N_MINUTES: usize = 237;
const N_MINUTE_COLS: usize = 75; // 3 versions × 25
const EXPECTED_LEN: usize = 5975;

fn adaptive_m_max(n: usize, cap: usize) -> usize { (n*n*10).max(2000).min(cap) }

const WINDOW_BOUNDS: [(f32,f32);6] = [
    (0.,14220.),(0.,1800.),(12600.,14220.),(1800.,12600.),(0.,7200.),(7200.,14220.),
];

// ============================================================================
// TradeRecord + CSV 读取
// ============================================================================
struct TradeRecord { time_sec: f32, volume: f32, flag: i32, bid_order: i64, ask_order: i64 }

fn resolve_path(date: i64, subdir: &str, fname: &str) -> std::io::Result<String> {
    if let Ok(p) = std::env::var("RUST_PYFUNC_LEVEL2_PATH") {
        let fp = Path::new(&p).join(date.to_string()).join(subdir).join(fname);
        if fp.exists() { return Ok(fp.to_string_lossy().into_owned()); }
    }
    for root in ["/ssd_data/stock", "/nas197/binary/stock/sz_alpha/stock"] {
        let fp = Path::new(root).join(date.to_string()).join(subdir).join(fname);
        if fp.exists() { return Ok(fp.to_string_lossy().into_owned()); }
    }
    Err(std::io::Error::new(std::io::ErrorKind::NotFound, "not found"))
}

#[inline] fn p_i64(b: &[u8]) -> i64 {
    let mut n = false; let mut i = 0;
    while i<b.len() && b[i]==b' ' { i+=1; }
    if i<b.len() && (b[i]==b'-'||b[i]==b'+') { n=b[i]==b'-'; i+=1; }
    let mut v: i64 = 0;
    while i<b.len() { let c=b[i]; if c<b'0'||c>b'9' {break;} v=v*10+(c-b'0') as i64; i+=1; }
    if n {-v} else {v}
}
#[inline] fn p_f32(b: &[u8]) -> f32 {
    let mut n=false; let mut i=0;
    if i<b.len()&&b[i]==b'-'{n=true;i+=1;}else if i<b.len()&&b[i]==b'+'{i+=1;}
    let mut ip=0.0f32;
    while i<b.len()&&b[i]>=b'0'&&b[i]<=b'9'{ip=ip*10.0+(b[i]-b'0')as f32;i+=1;}
    let mut fr=0.0f32; let mut fs=1.0f32;
    if i<b.len()&&b[i]==b'.'{i+=1;while i<b.len()&&b[i]>=b'0'&&b[i]<=b'9'{fr=fr*10.0+(b[i]-b'0')as f32;fs*=10.0;i+=1;}}
    let r=ip+fr/fs; if n{-r}else{r}
}

fn parse_line(l: &[u8]) -> Option<TradeRecord> {
    if l.is_empty() { return None; }
    let mut f: [&[u8];15] = [&[][..];15]; let mut s=0; let mut c=0;
    for (i,&b) in l.iter().enumerate() { if b==b',' { if c<15 {f[c]=&l[s..i];} c+=1; s=i+1; } }
    if c<15 { f[c]=&l[s..]; }
    if f[10] == b"32" { return None; }
    let flag = if f[10].is_empty() {0} else {p_i64(f[10]) as i32};
    let us = p_i64(f[4]); let ts = ((us+8*3600*1_000_000)/1_000_000) as f64;
    let doff = ((us/1_000_000)+8*3600).rem_euclid(86400);
    if doff < 9*3600+30*60 || doff > 14*3600+57*60 || (doff > 11*3600+30*60 && doff < 13*3600) { return None; }
    let ft = if doff >= 13*3600 { ts - (90*60) as f64 } else { ts };
    Some(TradeRecord { time_sec: ft as f32, volume: p_f32(f[8]), flag, bid_order: p_i64(f[14]), ask_order: p_i64(f[13]) })
}

fn parse_body(body: &[u8]) -> Vec<TradeRecord> {
    let mut out = Vec::with_capacity(body.len()/80+1);
    let mut s = 0;
    for i in 0..body.len() {
        if body[i] == b'\n' {
            let line = if i>s && body[i-1]==b'\r' { &body[s..i-1] } else { &body[s..i] };
            if let Some(r) = parse_line(line) { out.push(r); }
            s = i+1;
        }
    }
    if s < body.len() {
        let line = if body[s..].last()==Some(&b'\r') { &body[s..body.len()-1] } else { &body[s..] };
        if let Some(r) = parse_line(line) { out.push(r); }
    }
    out
}

/// 使用 mmap 读取（与主项目 read_trade_fast_inner 部分一致）
fn read_trades(code: &str, date: i64) -> std::io::Result<Vec<TradeRecord>> {
    let fname = format!("{}_{}_transaction.csv", code, date);
    let path = resolve_path(date, "transaction", &fname)?;
    let file = File::open(&path)?;
    let mmap = unsafe { Mmap::map(&file)? };
    let data = &mmap[..];
    let body = match data.iter().position(|&b| b==b'\n') {
        Some(p) => &data[p+1..], None => data
    };
    Ok(parse_body(body))
}

// ============================================================================
// PRNG + 统计
// ============================================================================
struct XorShift64 { state: u64 }
impl XorShift64 {
    fn new(s: u64) -> Self { Self { state: if s==0 { 0xDEAD_BEEF_DEAD_BEEF } else { s } } }
    #[inline] fn next_u64(&mut self) -> u64 { let mut x=self.state; x^=x<<13; x^=x>>7; x^=x<<17; self.state=x; x }
    #[inline] fn next_index(&mut self, n: usize) -> usize {
        if n==0 {0} else { ((self.next_u64()>>32) as u64).wrapping_mul(n as u64) as usize >> 32 }
    }
}

#[inline] fn std_ddof1(d: &[f32]) -> f32 {
    let n=d.len(); if n<2 { return f32::NAN }
    let m = d.iter().sum::<f32>()/n as f32;
    (d.iter().map(|&v|{let x=v-m;x*x}).sum::<f32>()/(n-1) as f32).sqrt()
}
fn pct_abs(d: &[f32], q: f64) -> f32 {
    let mut s: Vec<f32> = d.iter().map(|v|v.abs()).collect();
    s.sort_by(|a,b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n=s.len(); if n==0 {return f32::NAN} if n==1 {return s[0]}
    let pos=q*(n as f64-1.);let lo=pos.floor() as usize;let hi=(lo+1).min(n-1);let fr=pos-lo as f64;
    s[lo]*(1.-fr as f32)+s[hi]*fr as f32
}
fn runs_z(d: &[f32]) -> f32 {
    let n=d.len();if n<2{return f32::NAN}let n1=d.iter().filter(|&&v|v>=0.5).count();let n0=n-n1;
    if n0==0||n1==0{return f32::NAN}let mut r=1;
    for i in 1..n{if (d[i-1]>=0.5)!=(d[i]>=0.5){r+=1;}}
    let nt=n as f64;let er=2.*n0 as f64*n1 as f64/nt+1.;
    let vr=2.*n0 as f64*n1 as f64*(2.*n0 as f64*n1 as f64-nt)/(nt*nt*(nt-1.));
    if vr<1e-20{f32::NAN}else{((r as f64-er)/vr.sqrt())as f32}
}
fn hurst(d: &[f32]) -> f32 {
    let n=d.len();if n<16{return f32::NAN}let mut sz=vec![];let mut w=4;while w<=n/2{sz.push(w);w*=2;}if sz.len()<3{return f32::NAN}
    let mut lsz=vec![];let mut lrs=vec![];
    for &w in &sz{let nc=n/w;let mut rs=0.;let mut ct=0;
        for c in 0..nc{let ch=&d[c*w..(c+1)*w];let m:f64=ch.iter().map(|&v|v as f64).sum::<f64>()/w as f64;
            let mut cd=0.;let(mut mn,mut mx)=(f64::INFINITY,f64::NEG_INFINITY);let mut ss=0.;
            for &v in ch{let dd=v as f64-m;cd+=dd;if cd<mn{mn=cd}if cd>mx{mx=cd}ss+=dd*dd;}
            let rr=mx-mn;let s=(ss/w as f64).sqrt();if s>1e-10{rs+=rr/s;ct+=1;}}
        if ct>0{let a=rs/ct as f64;if a>0.{lsz.push((w as f64).ln());lrs.push(a.ln());}}}
    if lsz.len()<3{return f32::NAN}let ns=lsz.len();let mx=lsz.iter().sum::<f64>()/ns as f64;let my=lrs.iter().sum::<f64>()/ns as f64;
    let mut sxy=0.;let mut sxx=0.;for i in 0..ns{let dx=lsz[i]-mx;sxy+=dx*(lrs[i]-my);sxx+=dx*dx;}
    if sxx<1e-20{f32::NAN}else{(sxy/sxx)as f32}
}
fn dfa(d: &[f32]) -> f32 {
    let n=d.len();if n<16{return f32::NAN}let mean:f64=d.iter().map(|&v|v as f64).sum::<f64>()/n as f64;
    let mut y=vec![0.;n];y[0]=d[0]as f64-mean;for i in 1..n{y[i]=y[i-1]+d[i]as f64-mean;}
    let mut sz=vec![];let mut w=4;while w<=n/4{sz.push(w);w*=2;}if sz.len()<3{return f32::NAN}
    let mut lsz=vec![];let mut lf=vec![];
    for &w in &sz{let ns=n/w;let mut vs=0.;let mut ct=0;
        for s in 0..ns{let st=s*w;let mx=(0..w).map(|i|i as f64).sum::<f64>()/w as f64;let my=(0..w).map(|i|y[st+i]).sum::<f64>()/w as f64;
            let mut sxy=0.;let mut sxx=0.;for i in 0..w{let dx=i as f64-mx;sxy+=dx*(y[st+i]-my);sxx+=dx*dx;}
            if sxx<1e-20{continue;}let sl=sxy/sxx;let ic=my-sl*mx;let mut sr=0.;
            for i in 0..w{let p=sl*i as f64+ic;let r=y[st+i]-p;sr+=r*r;}vs+=sr/w as f64;ct+=1;}
        if ct>0{let f=(vs/ct as f64).sqrt();if f>1e-10{lsz.push((w as f64).ln());lf.push(f.ln());}}}
    if lsz.len()<3{return f32::NAN}let ns=lsz.len();let mx=lsz.iter().sum::<f64>()/ns as f64;let my=lf.iter().sum::<f64>()/ns as f64;
    let mut sxy=0.;let mut sxx=0.;for i in 0..ns{let dx=lsz[i]-mx;sxy+=dx*(lf[i]-my);sxx+=dx*dx;}
    if sxx<1e-20{f32::NAN}else{(sxy/sxx)as f32}
}
fn corrf(a:&[f32],b:&[f32])->f32{
    let n=a.len().min(b.len());if n<2{return f32::NAN}let ma=a.iter().sum::<f32>()/n as f32;let mb=b.iter().sum::<f32>()/n as f32;
    let mut sab=0.;let mut saa=0.;let mut sbb=0.;
    for i in 0..n{let da=(a[i]-ma)as f64;let db=(b[i]-mb)as f64;sab+=da*db;saa+=da*da;sbb+=db*db;}
    let d=(saa*sbb).sqrt();if d<1e-20{f32::NAN}else{(sab/d)as f32}
}
fn lin_slope(x:&[f32],y:&[f32])->f32{
    let n=x.len().min(y.len());if n<3{return f32::NAN}let mx=x.iter().sum::<f32>()/n as f32;let my=y.iter().sum::<f32>()/n as f32;
    let mut sxy=0.;let mut sxx=0.;for i in 0..n{let dx=(x[i]-mx)as f64;sxy+=dx*(y[i]-my)as f64;sxx+=dx*dx;}
    if sxx<1e-20{f32::NAN}else{(sxy/sxx)as f32}
}

// ============================================================================
// 订单聚合 + Side/Quantile
// ============================================================================
#[derive(Clone)] struct Agg { vol: f32, na: u32, np: u32, fidx: usize }

fn agg_orders(trades: &[TradeRecord]) -> (HashMap<i64,Agg>, HashMap<i64,Agg>) {
    let mut bid = HashMap::with_capacity(trades.len());
    let mut ask = HashMap::with_capacity(trades.len());
    for (i,t) in trades.iter().enumerate() {
        let be = bid.entry(t.bid_order).or_insert(Agg{vol:0.,na:0,np:0,fidx:i});
        be.vol += t.volume; match t.flag { 66=>be.na+=1, 83=>be.np+=1, _=>{} }
        let ae = ask.entry(t.ask_order).or_insert(Agg{vol:0.,na:0,np:0,fidx:i});
        ae.vol += t.volume; match t.flag { 83=>ae.na+=1, 66=>ae.np+=1, _=>{} }
    }
    (bid, ask)
}

#[derive(Clone,Copy,PartialEq,Eq,Hash)] enum Side { Bid,Ask,Mixed,Active,Passive,ActBid,ActAsk,PasBid,PasAsk }
impl Side { fn idx(self) -> usize { self as usize } }
#[derive(Clone,Copy,PartialEq)] enum Quant { All,Top10,Mid50,Bot40 }

fn segs() -> Vec<(usize,Side,Quant)> {
    let mut s = Vec::with_capacity(65);
    for &w in &[0,1,2,3,4,5] { for &si in &[Side::Bid,Side::Ask,Side::Mixed] { s.push((w,si,Quant::All)); } }
    for &si in &[Side::Bid,Side::Ask,Side::Mixed] { for &q in &[Quant::Top10,Quant::Mid50,Quant::Bot40] { s.push((0,si,q)); } }
    for &si in &[Side::Active,Side::Passive] { s.push((0,si,Quant::All)); }
    for &si in &[Side::Active,Side::Passive] { for &q in &[Quant::Top10,Quant::Mid50,Quant::Bot40] { s.push((0,si,q)); } }
    for &si in &[Side::ActBid,Side::PasBid,Side::ActAsk,Side::PasAsk] { for &q in &[Quant::Top10,Quant::Mid50,Quant::Bot40] { s.push((0,si,q)); } }
    for &si in &[Side::ActBid,Side::PasBid,Side::ActAsk,Side::PasAsk,Side::Active,Side::Passive] { s.push((2,si,Quant::All)); }
    for &si in &[Side::ActBid,Side::ActAsk,Side::PasBid,Side::PasAsk] { for &q in &[Quant::Top10,Quant::Bot40,Quant::Mid50] { s.push((2,si,q)); } }
    s
}

fn extract(bid: &HashMap<i64,Agg>, ask: &HashMap<i64,Agg>, side: Side) -> Vec<f32> {
    let mut triples: Vec<(usize,u8,f32)> = Vec::new();
    match side {
        Side::Bid => { for e in bid.values() { triples.push((e.fidx,0,e.vol)); } }
        Side::Ask => { for e in ask.values() { triples.push((e.fidx,0,e.vol)); } }
        Side::Mixed => { for e in bid.values() { triples.push((e.fidx,0,e.vol)); } for e in ask.values() { triples.push((e.fidx,1,e.vol)); } }
        Side::Active => {
            for e in bid.values() { if e.na>0&&e.np==0 { triples.push((e.fidx,0,e.vol)); } }
            for e in ask.values() { if e.na>0&&e.np==0 { triples.push((e.fidx,1,e.vol)); } }
        }
        Side::Passive => {
            for e in bid.values() { if e.np>0&&e.na==0 { triples.push((e.fidx,0,e.vol)); } }
            for e in ask.values() { if e.np>0&&e.na==0 { triples.push((e.fidx,1,e.vol)); } }
        }
        Side::ActBid => { for e in bid.values() { if e.na>0&&e.np==0 { triples.push((e.fidx,0,e.vol)); } } }
        Side::ActAsk => { for e in ask.values() { if e.na>0&&e.np==0 { triples.push((e.fidx,0,e.vol)); } } }
        Side::PasBid => { for e in bid.values() { if e.np>0&&e.na==0 { triples.push((e.fidx,0,e.vol)); } } }
        Side::PasAsk => { for e in ask.values() { if e.np>0&&e.na==0 { triples.push((e.fidx,0,e.vol)); } } }
    }
    triples.sort_by(|a,b| (a.0,a.1).cmp(&(b.0,b.1)));
    triples.into_iter().map(|(_,_,v)| v).collect()
}

fn qfilt(vols: &[f32], q: Quant) -> Vec<f32> {
    let n = vols.len();
    if n==0 || q==Quant::All { return vols.to_vec(); }
    let mut idx: Vec<(f32,usize)> = vols.iter().enumerate().map(|(i,&v)| (v,i)).collect();
    idx.sort_by(|a,b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    let (s,e) = match q {
        Quant::Top10 => { let k=((n as f64)*0.1).ceil() as usize; (0, k.max(1).min(n)) }
        Quant::Mid50 => { let a=((n as f64)*0.1) as usize; let b=((n as f64)*0.6) as usize; (a, b.min(n).max(a)) }
        Quant::Bot40 => { let k=((n as f64)*0.4) as usize; (n-k, n) }
        _ => (0,n),
    };
    let mut keep = vec![false; n];
    for &(_,i) in &idx[s..e] { keep[i] = true; }
    vols.iter().enumerate().filter(|(i,_)| keep[*i]).map(|(_,&v)| v).collect()
}

// ============================================================================
// 退火（与主项目完全一致）
// ============================================================================
const RS_MAX: usize = 2000; const DMAX: usize = 5000;
struct Buf { guess: Vec<f32>, smed: Vec<f32>, rs: Vec<f32>, dv: Vec<f32>, dt: Vec<f32>, gb: Vec<u8> }
impl Buf { fn new() -> Self { Self { guess:vec![], smed:vec![], rs:vec![], dv:vec![], dt:vec![], gb:vec![] } } }

fn anneal(tv: &[f32], mm: usize, buf: &mut Buf) -> [f32; N_FACTORS] {
    let n = tv.len(); if n<2 { return [f32::NAN; N_FACTORS]; }
    let mean: f64 = tv.iter().map(|&v| v as f64).sum::<f64>() / n as f64;
    let s2: f64 = tv.iter().map(|&v| { let d=v as f64-mean; d*d }).sum::<f64>() / n as f64;
    if s2 <= 0.0 { return [f32::NAN; N_FACTORS]; }
    buf.guess.clear(); buf.guess.extend_from_slice(tv);
    buf.guess.sort_by(|a,b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    // 中位数直接从已排序的 guess 取（节省一次 O(N log N) 排序）
    let med = (if n%2==0 { (buf.guess[n/2-1]+buf.guess[n/2])*0.5 } else { buf.guess[n/2] }) as f32;
    let mut s: f64 = (0..n).map(|k| { let d=buf.guess[k] as f64 - tv[k] as f64; d*d }).sum();
    let denom = 2.0*s2*n as f64; let inv_d = 1.0/denom; let r0 = 1.0 - s*inv_d;
    buf.rs.clear(); buf.dv.clear(); buf.dt.clear();
    let tol = (s2*n as f64*1e-10).max(1e-12);
    let ctb = s2*C1_FRAC; let inv_m = if mm>1 { 1.0/(mm as f64-1.0) } else { 0.0 };
    let stride = if mm<=RS_MAX { 1 } else { (mm+RS_MAX-1)/RS_MAX };
    let mut rng = XorShift64::new(SEED);
    let mut pr = r0; let mut fr = r0; let hl = (1.0+r0)*0.5;
    let (mut a1,mut a2,mut a3) = (usize::MAX,usize::MAX,usize::MAX);
    let mut ia = 0.0f64; let mut dc = 0u32; let mut rm = r0; let mut md = 0.0f64; let mut ps = 0usize; let mut mr = 0usize;
    let mut dn = 0u32; let (mut s1,mut s2,mut s3,mut s4) = (0.,0.,0.,0.);
    let ss = mm/3; let se1 = ss; let se2 = 2*ss;
    let (mut sn1,mut ss1,mut sq1,mut sn2,mut sd2,mut sn3,mut ss3,mut sq3) = (0u32,0.,0.,0u32,0u32,0u32,0.,0.);
    let mut rs2 = f64::NAN; let mut dcnt = 0usize; let mut t = 0usize;
    while t < mm {
        let i = rng.next_index(n); let mut j = rng.next_index(n); if j==i { j=rng.next_index(n); }
        if dcnt<DMAX && i!=j && buf.guess[i]!=buf.guess[j] {
            buf.dv.push(tv[j]-tv[i]); buf.dt.push(if j>i { (j-i) as f32 } else { (i-j) as f32 });
            buf.gb.push(if buf.guess[i] < med { 1 } else { 0 }); dcnt += 1;
        }
        if i != j {
            let gi=buf.guess[i]; let gj=buf.guess[j]; let ti=tv[i]; let tj=tv[j];
            let ds = 2.0f32*(gi-gj)*(ti-tj);
            let ct = ctb*(1.0-t as f64*inv_m).max(0.0_f64);
            if (ds as f64) < 0.0 || (ds as f64) < ct { s += ds as f64; buf.guess.swap(i,j); }
        }
        let cr = 1.0 - s*inv_d; fr = cr;
        if t%stride == 0 { buf.rs.push(cr as f32); }
        if t > 0 {
            let dv = cr-pr; dn+=1; s1+=dv; s2+=dv*dv; s3+=dv*dv*dv; s4+=dv*dv*dv*dv;
            if (cr as f32) < (pr as f32) { dc+=1; if t>=se1 && t<se2 { sd2+=1; } }
            if t<se1 { sn1+=1; ss1+=dv; sq1+=dv*dv; } else if t<se2 { sn2+=1; } else { sn3+=1; ss3+=dv; sq3+=dv*dv; }
        }
        if a1==usize::MAX && cr>=hl { a1=t; } if a2==usize::MAX && cr>=0.80 { a2=t; } if a3==usize::MAX && cr>=0.90 { a3=t; }
        ia += (1.0 - cr as f32) as f64;
        if cr >= rm { rm=cr; ps=t; } else { let uw=t-ps; if uw>mr { mr=uw; } }
        let dd = rm-cr; if dd > md { md = dd; }
        if t+1==se2 || (t+1==mm && rs2.is_nan()) { rs2 = cr; }
        pr = fr;
        if s <= tol { break; }
        t += 1;
    }
    while buf.rs.len() < 4 { buf.rs.push(fr as f32); }
    let mut f = [f32::NAN; N_FACTORS];
    f[0]=if a1!=usize::MAX {a1 as f32}else{mm as f32};f[1]=if a2!=usize::MAX{a2 as f32}else{mm as f32};f[2]=if a3!=usize::MAX{a3 as f32}else{mm as f32};
    f[3]=fr as f32;f[4]=ia as f32;f[5]=dc as f32;f[6]=md as f32;f[7]=mr as f32;
    if dn>=2{let nd=dn as f64;f[8]=((s2-s1*s1/nd)/(nd-1.)).max(0.).sqrt()as f32;}
    if sn1>=2{let nd=sn1 as f64;f[9]=((sq1-ss1*ss1/nd)/(nd-1.)).max(0.).sqrt()as f32;}
    if sn2>0{f[10]=sd2 as f32/sn2 as f32;}if!rs2.is_nan(){f[11]=(fr-rs2)as f32;}
    if sn3>=2{let nd=sn3 as f64;let st3=((sq3-ss3*ss3/nd)/(nd-1.)).max(0.).sqrt();
        if st3>1e-10&&sn1>=2{let nd1=sn1 as f64;f[12]=(((sq1-ss1*ss1/nd1)/(nd1-1.)).max(0.).sqrt()/st3)as f32;}}
    if buf.rs.len()>=4{let drs:Vec<f32>=(1..buf.rs.len()).map(|k|buf.rs[k]-buf.rs[k-1]).collect();
        if!drs.is_empty(){let p90=pct_abs(&drs,0.90);let bin:Vec<f32>=drs.iter().map(|&v|if v.abs()>=p90{1.0}else{0.0}).collect();f[13]=runs_z(&bin);}}
    if dn>=4{let nd=dn as f64;let mean=s1/nd;let m2=s2/nd-mean*mean;let m3=s3/nd-3.*mean*(s2/nd)+2.*mean*mean*mean;
        let m4=s4/nd-4.*mean*(s3/nd)+6.*mean*mean*(s2/nd)-3.*mean.powi(4);
        if m2>1e-20{let g1=m3/m2.powf(1.5);f[14]=(g1*((nd-1.)*nd).sqrt()/(nd-2.))as f32;
            let g2=m4/(m2*m2);f[15]=((((nd-1.)/((nd-2.)*(nd-3.)))*((nd+1.)*g2-3.*(nd-1.))))as f32;}}
    f[16]=hurst(&buf.rs);f[17]=dfa(&buf.rs);
    let k=buf.dv.len();
    if k>0{f[18]=buf.dv.iter().map(|d|d.abs()).sum::<f32>()/k as f32;f[19]=buf.dv.iter().filter(|d|**d>0.).count()as f32/k as f32;f[20]=std_ddof1(&buf.dv);
        if k>=2{f[21]=corrf(&buf.dv[..k-1],&buf.dv[1..]);}
        if k>=2{let flips=(0..k-1).filter(|&i|buf.dv[i].signum()!=buf.dv[i+1].signum()).count();f[22]=flips as f32/(k-1)as f32;}
        let p95=pct_abs(&buf.dv,0.95);let hidden=(0..k).filter(|&i|buf.gb[i]==1&&buf.dv[i].abs()>p95).count();f[23]=hidden as f32/k as f32;
        if k>=3{let ad:Vec<f32>=buf.dv.iter().map(|d|d.abs()).collect();f[24]=lin_slope(&buf.dt,&ad);}}
    f
}

// ============================================================================
// 完整计算 + 逐阶段计时
// ============================================================================

#[pyclass]
#[derive(Clone)]
struct StageReport {
    #[pyo3(get)] read_ms: f64,
    #[pyo3(get)] win_agg_ms: f64,
    #[pyo3(get)] scalar_ms: f64,
    #[pyo3(get)] minute_ms: f64,
    #[pyo3(get)] total_ms: f64,
    #[pyo3(get)] factors: Vec<f32>,
    #[pyo3(get)] n_trades: usize,
    #[pyo3(get)] n_nonempty_min: usize,
}

fn run_staged(code: &str, date: i64) -> std::io::Result<StageReport> {
    let t0 = Instant::now();

    // 阶段1: 读取
    let tr0 = Instant::now();
    let trades = read_trades(code, date)?;
    let read_ms = tr0.elapsed().as_secs_f64() * 1000.0;
    let nt = trades.len();
    if nt == 0 {
        return Ok(StageReport { read_ms, win_agg_ms:0., scalar_ms:0., minute_ms:0.,
            total_ms: t0.elapsed().as_secs_f64()*1000., factors: vec![f32::NAN; EXPECTED_LEN],
            n_trades:0, n_nonempty_min:0 });
    }
    let t_open = trades[0].time_sec;

    // 阶段2: 6窗口聚合
    let tr1 = Instant::now();
    let win_aggs: Vec<_> = WINDOW_BOUNDS.iter().map(|&(lo_s, hi_s)| {
        let lt = t_open+lo_s; let ht = t_open+hi_s;
        let lo = trades.partition_point(|t| t.time_sec < lt);
        let hi = trades.partition_point(|t| t.time_sec < ht);
        agg_orders(&trades[lo..hi])
    }).collect();
    let win_agg_ms = tr1.elapsed().as_secs_f64() * 1000.0;

    // 阶段3: 65标量片段
    let tr2 = Instant::now();
    let all_segs = segs();
    let mut buf = Buf::new();
    let mut out = Vec::with_capacity(EXPECTED_LEN);
    for &(wi, side, q) in &all_segs {
        let (b, a) = &win_aggs[wi];
        let vols = extract(b, a, side);
        let filt = qfilt(&vols, q);
        let ma = adaptive_m_max(filt.len(), M_MAX_SCALAR);
        let factors = anneal(&filt, ma, &mut buf);
        out.extend_from_slice(&factors);
    }
    let scalar_ms = tr2.elapsed().as_secs_f64() * 1000.0;

    // 阶段4: 237分钟矩阵
    let tr3 = Instant::now();
    let mut n_nonempty = 0usize;
    let mut matrix = Array2::zeros((N_MINUTES, N_MINUTE_COLS));
    // 预计算所有分钟边界（一次遍历，O(N)）
    let mut m_bounds = vec![(0usize, 0usize); N_MINUTES];
    {
        let mut m = 0usize; let mut ti = 0usize;
        while m < N_MINUTES && ti < nt {
            let lt = t_open + (m as f32)*60.;
            let ht = t_open + ((m+1) as f32)*60.;
            while ti<nt && trades[ti].time_sec < lt { ti+=1; }
            let lo = ti;
            while ti<nt && trades[ti].time_sec < ht { ti+=1; }
            m_bounds[m] = (lo, ti); m += 1;
        }
    }
    for m_idx in 0..N_MINUTES {
        let (lo, hi) = m_bounds[m_idx];
        if lo >= hi { continue; }
        n_nonempty += 1;
        let (bid, ask) = agg_orders(&trades[lo..hi]);
        for (vi, &ver) in [Side::Bid, Side::Ask, Side::Mixed].iter().enumerate() {
            let vols = extract(&bid, &ask, ver);
            let ma = adaptive_m_max(vols.len(), M_MAX_MINUTE);
            let factors = anneal(&vols, ma, &mut buf);
            for (fi, &val) in factors.iter().enumerate() { matrix[[m_idx, vi*N_FACTORS+fi]] = val; }
        }
    }
    let minute_ms = tr3.elapsed().as_secs_f64() * 1000.0;

    // 阶段5: 降维占位
    out.resize(EXPECTED_LEN, f32::NAN);

    let total_ms = t0.elapsed().as_secs_f64() * 1000.0;
    Ok(StageReport { read_ms, win_agg_ms, scalar_ms, minute_ms, total_ms,
        factors: out, n_trades: nt, n_nonempty_min: n_nonempty })
}

#[pyfunction]
fn profile(code: String, date: i64) -> PyResult<StageReport> {
    Ok(run_staged(&code, date)?)
}

/// 优化版：缓存 extract 结果 + 合并 qfilt 排序
fn run_optimized(code: &str, date: i64) -> std::io::Result<StageReport> {
    let t0 = Instant::now();

    // 阶段1: 读取（同 baseline）
    let tr0 = Instant::now();
    let trades = read_trades(code, date)?;
    let read_ms = tr0.elapsed().as_secs_f64() * 1000.0;
    let nt = trades.len();
    if nt == 0 {
        return Ok(StageReport { read_ms, win_agg_ms:0., scalar_ms:0., minute_ms:0.,
            total_ms: t0.elapsed().as_secs_f64()*1000., factors: vec![f32::NAN; EXPECTED_LEN],
            n_trades:0, n_nonempty_min:0 });
    }
    let t_open = trades[0].time_sec;

    // 阶段2: 6窗口聚合（同 baseline）
    let tr1 = Instant::now();
    let win_aggs: Vec<_> = WINDOW_BOUNDS.iter().map(|&(lo_s, hi_s)| {
        let lt = t_open+lo_s; let ht = t_open+hi_s;
        let lo = trades.partition_point(|t| t.time_sec < lt);
        let hi = trades.partition_point(|t| t.time_sec < ht);
        agg_orders(&trades[lo..hi])
    }).collect();
    let win_agg_ms = tr1.elapsed().as_secs_f64() * 1000.0;

    // 阶段3: 65标量片段 — 优化版（缓存 extract + 合并 qfilt sort）
    let tr2 = Instant::now();
    let all_segs = segs();
    let mut buf = Buf::new();
    let mut out = Vec::with_capacity(EXPECTED_LEN);

    // --- 优化1: 预提取所有唯一的 (window, side) extract 结果 ---
    // 找到所有唯一 (wi, side) pair
    let mut seen: HashMap<(usize, usize), usize> = HashMap::with_capacity(30);
    let mut extracts: Vec<Vec<f32>> = Vec::with_capacity(30);
    for &(wi, side, _) in &all_segs {
        let key = (wi, side.idx());
        if !seen.contains_key(&key) {
            let (b, a) = &win_aggs[wi];
            let v = extract(b, a, side);
            seen.insert(key, extracts.len());
            extracts.push(v);
        }
    }

    // --- 优化2: 对于同一份 volumes 的多个 quantile，预排序 ---
    struct QfiltEntry { sorted_idx: Vec<(f32, usize)>, n: usize }
    let mut qfilt_cache: HashMap<(usize, usize), QfiltEntry> = HashMap::with_capacity(20);

    for &(wi, side, q) in &all_segs {
        let ext_key = (wi, side.idx());
        let vols = &extracts[seen[&ext_key]];
        let filt = if q == Quant::All {
            vols.clone()
        } else {
            let entry = qfilt_cache.entry(ext_key).or_insert_with(|| {
                let mut idx: Vec<(f32, usize)> = vols.iter().enumerate().map(|(i, &v)| (v, i)).collect();
                idx.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
                QfiltEntry { sorted_idx: idx, n: vols.len() }
            });
            let (s, e) = match q {
                Quant::Top10 => { let k=((entry.n as f64)*0.1).ceil() as usize; (0, k.max(1).min(entry.n)) }
                Quant::Mid50 => { let a=((entry.n as f64)*0.1) as usize; let b=((entry.n as f64)*0.6) as usize; (a, b.min(entry.n).max(a)) }
                Quant::Bot40 => { let k=((entry.n as f64)*0.4) as usize; (entry.n-k, entry.n) }
                _ => (0, entry.n),
            };
            let mut keep = vec![false; entry.n];
            for &(_, i) in &entry.sorted_idx[s..e] { keep[i] = true; }
            vols.iter().enumerate().filter(|(i, _)| keep[*i]).map(|(_, &v)| v).collect()
        };
        let ma = adaptive_m_max(filt.len(), M_MAX_SCALAR);
        let factors = anneal(&filt, ma, &mut buf);
        out.extend_from_slice(&factors);
    }
    let scalar_ms = tr2.elapsed().as_secs_f64() * 1000.0;

    // 阶段4: 237分钟矩阵（同 baseline）
    let tr3 = Instant::now();
    let mut n_nonempty = 0usize;
    let mut matrix = Array2::zeros((N_MINUTES, N_MINUTE_COLS));
    let mut m_bounds = vec![(0usize, 0usize); N_MINUTES];
    { let mut m = 0usize; let mut ti = 0usize;
        while m < N_MINUTES && ti < nt {
            let lt = t_open + (m as f32)*60.; let ht = t_open + ((m+1) as f32)*60.;
            while ti<nt && trades[ti].time_sec < lt { ti+=1; } let lo = ti;
            while ti<nt && trades[ti].time_sec < ht { ti+=1; }
            m_bounds[m] = (lo, ti); m += 1;
        }
    }
    for m_idx in 0..N_MINUTES {
        let (lo, hi) = m_bounds[m_idx]; if lo >= hi { continue; } n_nonempty += 1;
        let (bid, ask) = agg_orders(&trades[lo..hi]);
        for (vi, &ver) in [Side::Bid, Side::Ask, Side::Mixed].iter().enumerate() {
            let vols = extract(&bid, &ask, ver);
            let ma = adaptive_m_max(vols.len(), M_MAX_MINUTE);
            let factors = anneal(&vols, ma, &mut buf);
            for (fi, &val) in factors.iter().enumerate() { matrix[[m_idx, vi*N_FACTORS+fi]] = val; }
        }
    }
    let minute_ms = tr3.elapsed().as_secs_f64() * 1000.0;

    out.resize(EXPECTED_LEN, f32::NAN);
    let total_ms = t0.elapsed().as_secs_f64() * 1000.0;
    Ok(StageReport { read_ms, win_agg_ms, scalar_ms, minute_ms, total_ms,
        factors: out, n_trades: nt, n_nonempty_min: n_nonempty })
}

#[pyfunction]
fn profile_opt(code: String, date: i64) -> PyResult<StageReport> {
    Ok(run_optimized(&code, date)?)
}

#[pymodule]
fn dev_sandbox(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<StageReport>()?;
    m.add_function(wrap_pyfunction!(profile, m)?)?;
    m.add_function(wrap_pyfunction!(profile_opt, m)?)?;
    Ok(())
}
