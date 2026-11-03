//! 非线性组合工厂 v4（sandbox_geffe）——列数恒定版
//! 规则：每股每天列数恒定；算不出来给 NaN。流不做剔除（剔除留给回测后的事后判断），
//! 三元组按固定索引取样，与股票无关。
//! 用法: geffe_sandbox <date> [out.json | - CODE...]
mod fast_csv_reader;
use fast_csv_reader::{read_trade_raw, read_market_raw, TradeRecord, MarketRecord};
use rayon::prelude::*;
use serde::Serialize;

const K: usize = 4;
const TRI_GLOBAL_CAP: usize = 3200;              // 三域三元组取样上限（均距取）
const DOM_CAP: [usize;3] = [1800, 900, 500];     // tick / market / aligned

#[derive(Serialize)]
struct Out { names: Vec<String>, codes: Vec<String>, vals: Vec<Vec<f32>> }

const NCOMB: usize = 5;

fn ctx_margin(s:&[u8])->f64{
    // 单遍滚动：c 为最近K位的上下文（低K位），命中表 O(1) 查询
    let n=s.len(); if n<2*K+60 {return f64::NAN;}
    let m=n-K; let half=m/2; let mask=(1usize<<K)-1;
    let mut cnt=[[0u32;2];1<<K];
    // c_t = s[t..t+K-1]；t 步的目标位是 s[t+K]
    let mut c=0usize; for t in 0..K.min(n){c=(c<<1)|s[t] as usize;}
    let mut hit=0u64; let mut tot=0u64; let mut sum1=0u64;
    for t in 0..m{
        let tgt=s[t+K];
        if t<half {cnt[c][tgt as usize]+=1;}
        else {hit+=(((cnt[c][1]>cnt[c][0]) as u8)==tgt) as u64; tot+=1; sum1+=tgt as u64;}
        c=((c<<1)|tgt as usize)&mask;   // 下一上下文 s[t+1..t+K]
    }
    if tot==0 {return f64::NAN;}
    let p=sum1 as f64/tot as f64;
    hit as f64/tot as f64 - p.max(1.0-p)
}
/// 与「把整条流旋转 r 位再做 lag1 比较」等价，但不再生成副本：
/// 原始写法比的是 s[i] 与 rot[i-1]=s[(i-1+r)%n]，下标滚动即可
#[inline] fn lag_agree_rot(s:&[u8],r:usize)->f64{
    let n=s.len(); if n<2 {return f64::NAN;}
    let mut j=r%n; let mut hit=0u64;
    for i in 1..n{ hit+=(s[i]==s[j]) as u64; j+=1; if j==n {j=0;} }
    hit as f64/(n-1) as f64
}
fn median(v:&mut [f64])->f64{
    if v.is_empty(){return f64::NAN;}
    let m=v.len()/2; *v.select_nth_unstable_by(m,|a,b|a.partial_cmp(b).unwrap()).1
}
fn pct(v:&mut Vec<f64>,num:usize,den:usize)->f64{
    if v.is_empty(){return f64::NAN;}
    let i=(v.len()*num)/den; *v.select_nth_unstable_by(i,|a,b|a.partial_cmp(b).unwrap()).1
}

fn bitop(op:usize, f:&[f64])->Vec<u8>{
    let n=f.len(); if n<3 {return Vec::new();}
    match op{
        0=>{let mut c=f.to_vec(); let m=median(&mut c); f.iter().map(|&x|(x>m) as u8).collect()},
        1=>{(1..n).map(|i|(f[i]>f[i-1]) as u8).collect()},
        2=>{let mut c=f.to_vec(); let q=pct(&mut c,3,4); f.iter().map(|&x|(x>q) as u8).collect()},
        _=>{(1..n).map(|i|((f[i]>f[i-1])!=(f[i-1]>(if i>1{f[i-2]}else{f[i-1]}))) as u8).collect()}
    }
}
const OPS:[&str;4]=["gt","df","q7","fl"];

#[derive(Clone)]
struct Dom{ names:Vec<String>, bits:Vec<Vec<u8>> }

// ---------- 三元组：抽样位压进 u64，一次算 64 个样本 ----------
// 同一「时段+域」里所有流都被 fld_streams 截断到同一长度 m，所以抽样步长、两个旋转量、
// 抽样点数对整域是同一套常量，可以每条流预先压好三份位图，三元组内循环只剩位运算和 popcount。
struct Packed{ sm:usize, nw:usize, tail:usize,
    p0:Vec<Vec<u64>>, p1:Vec<Vec<u64>>, p2:Vec<Vec<u64>> }

#[inline] fn comb_w<const C:u8>(a:u64,b:u64,c:u64)->u64{
    match C{0=>(a&b)^(b&c)^c,
            1=>(a&b)|((!a)&c),
            2=>(a&b)|(b&c)|(c&a),
            3=>a^b^c,
            _=>((!a)|b)&((!b)|c)}
}

/// 一趟取出三个偏移（0 / r1 / r2）的抽样位，各压成一份位图
fn pack_three(s:&[u8],step:usize,r1:usize,r2:usize,sm:usize)->(Vec<u64>,Vec<u64>,Vec<u64>){
    let m=s.len(); let nw=(sm+63)/64;
    let mut a=vec![0u64;nw]; let mut b=vec![0u64;nw]; let mut c=vec![0u64;nw];
    let (mut i0,mut i1,mut i2)=(0usize,r1,r2);
    for i in 0..sm{
        let sh=1u64<<(i&63); let w=i>>6;
        if s[i0]!=0 {a[w]|=sh;}
        if s[i1]!=0 {b[w]|=sh;}
        if s[i2]!=0 {c[w]|=sh;}
        i0+=step; i1+=step; i2+=step;
        if i0>=m{i0-=m;} if i1>=m{i1-=m;} if i2>=m{i2-=m;}
    }
    (a,b,c)
}

fn pack_domain(bits:&[Vec<u8>])->Option<Packed>{
    let m=bits.iter().map(|b|b.len()).min()?;
    if m<60 {return None;}
    let step=(m/12000).max(1); let sm=(m+step-1)/step;
    let r1=(m/3+7)%m; let r2=(m/3+11)%m;
    let nw=(sm+63)/64; let tail=sm-(nw-1)*64;
    let (mut p0,mut p1,mut p2)=(Vec::with_capacity(bits.len()),
        Vec::with_capacity(bits.len()),Vec::with_capacity(bits.len()));
    for s in bits{
        let (a,b,c)=pack_three(s,step,r1,r2,sm);
        p0.push(a); p1.push(b); p2.push(c);
    }
    Some(Packed{sm,nw,tail,p0,p1,p2})
}

#[inline]
fn triple_stats<const C:u8>(p:&Packed,i:usize,j:usize,k:usize)->([u64;3],[u64;3],u64){
    let (a0,b0,c0)=(&p.p0[i],&p.p0[j],&p.p0[k]);
    let (b1,c2)=(&p.p1[j],&p.p2[k]);
    let (mut hit,mut ehit,mut zsum)=([0u64;3],[0u64;3],0u64);
    for w in 0..p.nw{
        let msk=if w+1==p.nw&&p.tail<64 {(1u64<<p.tail)-1} else {!0u64};
        let a=a0[w]&msk; let b=b0[w]&msk; let c=c0[w]&msk;
        let bb=b1[w]&msk; let cc=c2[w]&msk;
        let z=comb_w::<C>(a,b,c)&msk;
        let x=comb_w::<C>(a,bb,cc)&msk;
        let cnt=msk.count_ones() as u64;
        zsum+=z.count_ones() as u64;
        hit[0]+=cnt-(z^a).count_ones() as u64;
        hit[1]+=cnt-(z^b).count_ones() as u64;
        hit[2]+=cnt-(z^c).count_ones() as u64;
        ehit[0]+=cnt-(x^a).count_ones() as u64;
        ehit[1]+=cnt-(x^b).count_ones() as u64;
        ehit[2]+=cnt-(x^c).count_ones() as u64;
    }
    (hit,ehit,zsum)
}

#[inline]
fn push_triple(p:&Packed,i:usize,j:usize,k:usize,feats:&mut Vec<f64>){
    let dm=p.sm as f64;
    for comb in 0..NCOMB{
        let (hit,ehit,zsum)=match comb{
            0=>triple_stats::<0>(p,i,j,k),
            1=>triple_stats::<1>(p,i,j,k),
            2=>triple_stats::<2>(p,i,j,k),
            3=>triple_stats::<3>(p,i,j,k),
            _=>triple_stats::<4>(p,i,j,k),
        };
        feats.push((hit[0] as f64/dm).max(hit[1] as f64/dm).max(hit[2] as f64/dm));
        feats.push(((hit[0] as f64-ehit[0] as f64)/dm)
            .max((hit[1] as f64-ehit[1] as f64)/dm)
            .max((hit[2] as f64-ehit[2] as f64)/dm));
        feats.push(zsum as f64/dm);
    }
}

fn fld_streams(prefix:&str, flds:&[(String,Vec<f64>)])->Dom{
    let mut names=Vec::new(); let mut bits=Vec::new();
    for (fnm,f) in flds{
        for op in 0..4{
            names.push(format!("{prefix}_{fnm}_{}",OPS[op]));
            bits.push(bitop(op,f));
        }
    }
    // 截齐到全域最短（diff/flip 少一位）
    let mlen=bits.iter().map(|b|b.len()).min().unwrap_or(0);
    for b in bits.iter_mut(){b.truncate(mlen);}
    Dom{names,bits}
}

fn build_tick(tr:&[&TradeRecord])->Dom{
    let flds:Vec<(String,Vec<f64>)>=vec![
        ("x".into(),tr.iter().map(|t|(t.flag==66) as i32 as f64).collect()),
        ("p".into(),tr.iter().map(|t|t.price).collect()),
        ("v".into(),tr.iter().map(|t|t.volume).collect()),
        ("a".into(),tr.iter().map(|t|t.turnover).collect()),
        ("d".into(),tr.iter().map(|t|(t.ask_order-t.bid_order).abs() as f64).collect()),
        ("r".into(),tr.windows(2).map(|w|(w[1].time_us-w[0].time_us) as f64).collect()),
        ("o".into(),tr.iter().map(|t|t.volume%100.0).collect()),
    ];
    fld_streams("t",&flds)
}
fn build_market(mk:&[&MarketRecord])->Dom{
    if mk.len()<3 {return Dom{names:Vec::new(),bits:Vec::new()};}
    let flds:Vec<(String,Vec<f64>)>=vec![
        ("i".into(),mk.iter().map(|s|{let d=s.total_bid_vol+s.total_ask_vol;if d>0.0{(s.total_bid_vol-s.total_ask_vol)/d}else{0.0}}).collect()),
        ("j".into(),mk.iter().map(|s|{let b=s.bid_vols[0]+s.bid_vols[1]+s.bid_vols[2];let a=s.ask_vols[0]+s.ask_vols[1]+s.ask_vols[2];if a+b>0.0{(b-a)/(a+b)}else{0.0}}).collect()),
        ("m".into(),mk.iter().map(|s|(s.ask_prcs[0]+s.bid_prcs[0])/2.0).collect()),
        ("s".into(),mk.iter().map(|s|s.ask_prcs[0]-s.bid_prcs[0]).collect()),
        ("b".into(),mk.iter().map(|s|s.total_bid_vol).collect()),
        ("a".into(),mk.iter().map(|s|s.total_ask_vol).collect()),
    ];
    fld_streams("m",&flds)
}
fn build_aligned(tr:&[&TradeRecord], mk:&[&MarketRecord])->Dom{
    if mk.len()<3||tr.is_empty() {return Dom{names:Vec::new(),bits:Vec::new()};}
    let n=mk.len();
    let mut ti=Vec::with_capacity(n); let mut j=0usize;
    for s in mk{while j+1<tr.len()&&tr[j+1].time_us<=s.time_us{j+=1;} ti.push(j);}
    let flds:Vec<(String,Vec<f64>)>=vec![
        ("d".into(),ti.iter().map(|&i|(tr[i].flag==66) as i32 as f64).collect()),
        ("p".into(),ti.iter().map(|&i|tr[i].price).collect()),
        ("g".into(),(0..n).map(|i|{let d=mk[i].total_bid_vol+mk[i].total_ask_vol;
            let ib=if d>0.0{((mk[i].total_bid_vol-mk[i].total_ask_vol)/d>0.0)}else{false};
            ((tr[ti[i]].flag==66)==ib) as i32 as f64}).collect()),
        ("u".into(),(0..n).map(|i|(tr[ti[i]].flag==66) as i32 as f64 * 
            (if mk[i].total_bid_vol>mk[i].total_ask_vol{1.0}else{0.0})).collect()),
    ];
    fld_streams("a",&flds)
}

/// 全域三元组：索引均距取样，固定确定性
fn triples_for(n:usize,cap:usize)->Vec<(usize,usize,usize)>{
    let mut v=Vec::new();
    for i in 0..n{for j in i+1..n{for k in j+1..n{v.push((i,j,k));}}}
    if v.len()>cap{
        let step=v.len() as f64/cap as f64;
        v=(0..cap).map(|t|v[(t as f64*step) as usize]).collect();
    }
    v
}
// 预计算的全域三元组清单（按最坏的满员流数：28/24/16）
fn all_tris()->[Vec<(usize,usize,usize)>;3]{
    [triples_for(28,DOM_CAP[0]),triples_for(24,DOM_CAP[1]),triples_for(16,DOM_CAP[2])]
}

const SESSIONS:[(&str,i64,i64);3]=[
    ("al",9*3600+30*60,14*3600+57*60),
    ("pm",13*3600,14*3600+57*60),
    ("tl",14*3600+27*60,14*3600+57*60),
];
#[inline] fn tod(us:i64)->i64{ (us/1_000_000).rem_euclid(86400) }

fn per_stock(trades:&[TradeRecord], mkt:&[MarketRecord], tris:&[Vec<(usize,usize,usize)>;3])
    ->Vec<f64>{
    let mut feats=Vec::with_capacity(23000);
    for (si,(sname,lo,hi)) in SESSIONS.iter().enumerate(){
        let _=si;
        let tr:Vec<&TradeRecord>=trades.iter()
            .filter(|t|{let h=tod(t.time_us); h>=*lo&&h<*hi}).collect();
        let mk:Vec<&MarketRecord>=mkt.iter()
            .filter(|s|{let h=tod(s.time_us); h>=*lo&&h<*hi}).collect();
        let doms=[build_tick(&tr),build_market(&mk),build_aligned(&tr,&mk)];
        for (di,d) in doms.iter().enumerate(){
            // 基础指标：p / lag1excess / cmg，每条流3个，空流给NaN
            let nmax=[28usize,24,16][di];
            for i in 0..nmax{
                if i<d.bits.len(){
                    let s=&d.bits[i];
                    if s.len()>=3{
                        let p=s.iter().map(|&v|v as usize).sum::<usize>() as f64/s.len() as f64;
                        feats.push(p); feats.push(lag_agree_rot(s,s.len()/3+7)); feats.push(ctx_margin(s));
                    }else{feats.extend_from_slice(&[f64::NAN;3]);}
                }else{feats.extend_from_slice(&[f64::NAN;3]);}
            }
            // 三元组：全域固定索引；流不存在或太短则 NaN
            match &pack_domain(&d.bits){
                Some(p)=>for &(i,j,k) in &tris[di]{
                    push_triple(p,i,j,k,&mut feats);
                },
                None=>feats.resize(feats.len()+tris[di].len()*NCOMB*3,f64::NAN),
            }
        }
    }
    feats
}

fn stock_codes_for_day(dir:&std::path::Path,symbol_map:&std::path::Path,date:i64)
    ->std::io::Result<Vec<String>>{
    use std::io::{Error,ErrorKind};
    let invalid=|e|Error::new(ErrorKind::InvalidData,e);
    let mut reader=csv::Reader::from_path(symbol_map).map_err(invalid)?;
    let col=reader.headers().map_err(invalid)?.iter().position(|h|h=="symbol")
        .ok_or_else(||Error::new(ErrorKind::InvalidData,"股票表缺少 symbol 列"))?;
    let mut stocks=std::collections::BTreeSet::new();
    for rec in reader.records(){
        let rec=rec.map_err(invalid)?;
        let code=rec.get(col).unwrap_or("").trim();
        if code.is_empty()||code.len()>6||!code.bytes().all(|b|b.is_ascii_digit()){
            return Err(Error::new(ErrorKind::InvalidData,"股票表含无效代码"));
        }
        stocks.insert(format!("{code:0>6}"));
    }
    if stocks.is_empty()||!dir.is_dir(){
        return Err(Error::new(ErrorKind::InvalidData,"白名单为空或目录不存在"));
    }
    Ok(stocks.into_iter().filter(|c|
        dir.join(format!("{c}_{date}_transaction.csv")).is_file()).collect())
}

fn all_names(tris:&[Vec<(usize,usize,usize)>;3])->Vec<String>{
    let mut names=Vec::with_capacity(23000);
    let domp=["t","m","a"];
    let base_flds:[&[&str];3]=[
        &["x","p","v","a","d","r","o"],
        &["i","j","m","s","b","a"],
        &["d","p","g","u"],
    ];
    for (sname,_,_) in SESSIONS.iter(){
        for di in 0..3{
            for fi in 0..base_flds[di].len(){
                for op in OPS{
                    let sn=format!("{}_{}_{}_{}",sname,domp[di],base_flds[di][fi],op);
                    names.push(format!("p_{sn}"));names.push(format!("lg_{sn}"));names.push(format!("cm_{sn}"));
                }
            }
            for &(i,j,k) in &tris[di]{
                let tn=format!("{}_{}_{}_{}_{}",sname,domp[di],i,j,k);
                for cb in ["gf","sl","mj","pa","im"]{
                    names.push(format!("{cb}_{tn}_lmx"));names.push(format!("{cb}_{tn}_lnm"));
                    names.push(format!("{cb}_{tn}_pz"));
                }
            }
        }
    }
    names
}

fn main(){
    let args:Vec<String>=std::env::args().collect();
    let date:i64=args.get(1).expect("用法: geffe_sandbox <date> [out.json|- CODE...]").parse().unwrap();
    let mode=args.get(2).cloned().unwrap_or_default();
    let root=std::env::var("RUST_PYFUNC_LEVEL2_ROOT").unwrap_or("/ssd_data/stock".into());
    let basic=std::env::var("RUST_PYFUNC_BASIC_INFO_DIR").unwrap_or("/ssd_data/data/basic_info".into());
    let tris=all_tris();
    let tris_ref=&tris;
    eprintln!("triples: t={} m={} a={} total={}",tris[0].len(),tris[1].len(),tris[2].len(),
        tris[0].len()+tris[1].len()+tris[2].len());
    let names=all_names(&tris);
    eprintln!("names={}",names.len());

    if mode=="-"{
        for c in &args[3..]{
            let t0=std::time::Instant::now();
            let tr=read_trade_raw(c,date).unwrap(); let mk=read_market_raw(c,date).unwrap();
            let t1=t0.elapsed();
            let f=per_stock(&tr,&mk,tris_ref);
            let nan=f.iter().filter(|v|v.is_nan()).count();
            eprintln!("{c} read={:?} calc={:?} feats={} nan={}",t1,t0.elapsed()-t1,f.len(),nan);
        }
        return;
    }

    rayon::ThreadPoolBuilder::new().num_threads(48).build_global().unwrap();
    let codes=stock_codes_for_day(
        &std::path::Path::new(&root).join(date.to_string()).join("transaction"),
        &std::path::Path::new(&basic).join("symbol_map.csv"),date)
        .expect("股票范围检查失败");
    eprintln!("codes={}",codes.len());
    let t_all=std::time::Instant::now();
    let stats=mode.is_empty();

    let results:Vec<(String,Vec<f64>,f64)>=codes.par_iter().map(|c|{
        let t0=std::time::Instant::now();
        let tr=read_trade_raw(c,date).unwrap_or_default();
        let mk=read_market_raw(c,date).unwrap_or_default();
        let f=per_stock(&tr,&mk,tris_ref);
        (c.clone(),f,t0.elapsed().as_secs_f64())
    }).collect();

    let el=t_all.elapsed();
    let tot_ms:Vec<f64>=results.iter().map(|(_,_,t)|*t*1000.0).collect();
    let mean=tot_ms.iter().sum::<f64>()/tot_ms.len() as f64;
    let mut s=tot_ms.clone(); s.sort_by(|a,b|a.partial_cmp(b).unwrap());
    let nan_share:Vec<f64>=results.iter().map(|(_,f,_)|f.iter().filter(|v|v.is_nan()).count() as f64/f.len().max(1) as f64).collect();
    let mut ns=nan_share.clone(); ns.sort_by(|a,b|a.partial_cmp(b).unwrap());
    eprintln!("wall={:?} stocks={} cols={}",el,results.len(),
        results.first().map(|(_,f,_)|f.len()).unwrap_or(0));
    eprintln!("per-stock ms: mean={:.0} p50={:.0} p90={:.0} max={:.0}",mean,s[s.len()/2],s[s.len()*9/10],s[s.len()-1]);
    eprintln!("nan share: p50={:.2} p90={:.2} max={:.2}",ns[ns.len()/2],ns[ns.len()*9/10],ns[ns.len()-1]);

    if !stats{
        let ncol=names.len();
        let mut vc=Vec::new(); let mut vv=Vec::new();
        for (c,f,_) in results{
            let mut row=f; row.resize(ncol,f64::NAN);
            vc.push(c); vv.push(row.into_iter().map(|x|x as f32).collect());
        }
        let out=Out{names,codes:vc,vals:vv};
        println!("{}",serde_json::to_string(&out).unwrap());
    }
}
