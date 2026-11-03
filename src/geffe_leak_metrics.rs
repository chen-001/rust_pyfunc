//! 线性反馈移位寄存器的非线性组合 —— 组合泄露因子（sandbox_geffe 正式版）
//!
//! 做法：把逐笔/盘口字段二值化成 0/1 位流（4 种二值化），在三个时段（全天/下午/尾盘）、
//! 三个域（成交/盘口/对齐）上各取一批三元组，用 5 种非线性组合函数（Geffe 生成器、
//! 选择器、多数表决、异或、蕴含）算出「组合后的位与各自输入的一致率」「相对错位基线的超额」
//! 「组合结果的偏置」三类统计量，用来找非线性组合之间的泄露。
//!
//! 输出列数恒定 144612 列，算不出来给 NaN。三元组按固定索引取样，与股票无关。
//! 计算口径与 sandbox_geffe 逐字节一致（三元组统计用位打包 + popcount，计的是整数命中次数，
//! 最后才除以抽样点数，所以换成位运算不改变数值）。
use pyo3::prelude::*;
use std::sync::OnceLock;

mod reader {
    //! 单股逐笔/盘口读取（f64 字段）。本因子要保留 f64 精度：主库 fast_csv_reader 的记录是 f32，
    //! 大成交量在 f32 下会丢整数位，volume%100 这类字段会变，所以这里自带一份读取器。
    use std::fs::File;
    use std::io::Read;

    const AFTERNOON_START_SEC: i64 = 13 * 3600;
    const AFTERNOON_END_SEC: i64 = 14 * 3600 + 57 * 60;
    const MORNING_START_SEC: i64 = 9 * 3600 + 30 * 60;
    const MORNING_END_SEC: i64 = 11 * 3600 + 30 * 60;
    const AFTERNOON_SHIFT_SEC: i64 = 90 * 60;
    const SHIFT_US: i64 = 90 * 60 * 1_000_000;
    const OFFSET_US: i64 = 8 * 3600 * 1_000_000;

    const COL_EXCHTIME: usize = 4;
    const COL_PRICE: usize = 7;
    const COL_VOLUME: usize = 8;
    const COL_TURNOVER: usize = 9;
    const COL_FLAG: usize = 10;
    const COL_ASK_ORDER: usize = 13;
    const COL_BID_ORDER: usize = 14;
    const COL_INDEX: usize = 6;

    const MKT_COL_EXCHTIME: usize = 4;
    const MKT_COL_LAST_PRC: usize = 6;
    const MKT_COL_VOLUME: usize = 14;
    const MKT_COL_TURNOVER: usize = 15;
    const MKT_COL_TOTAL_ASK_VOL: usize = 19;
    const MKT_COL_TOTAL_BID_VOL: usize = 20;
    const MKT_COL_ASK_PRC_BASE: usize = 21;

    #[derive(Clone, Copy, Debug, Default)]
    pub struct TradeRecord {
        pub time_us: i64,
        pub time_sec: f64,
        pub price: f64,
        pub volume: f64,
        pub turnover: f64,
        pub flag: i32,
        pub bid_order: i64,
        pub ask_order: i64,
        pub index: i64,
    }

    #[derive(Clone, Copy, Debug, Default)]
    pub struct MarketRecord {
        pub time_us: i64,
        pub time_sec: f64,
        pub last_prc: f64,
        pub volume: f64,
        pub turnover: f64,
        pub total_ask_vol: f64,
        pub total_bid_vol: f64,
        pub ask_prcs: [f64; 10],
        pub ask_vols: [f64; 10],
        pub bid_prcs: [f64; 10],
        pub bid_vols: [f64; 10],
    }

    fn resolve_stock_path(date: i64, subdir: &str, filename: &str) -> std::io::Result<String> {
        let roots = crate::data_paths::level2_roots();
        for root in &roots {
            let p = root.join(date.to_string()).join(subdir).join(filename);
            if p.exists() {
                return Ok(p.to_string_lossy().into_owned());
            }
        }
        Err(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("股票数据文件未找到: date={date}, subdir={subdir}, file={filename}"),
        ))
    }

    #[inline]
    fn parse_i64_fast(bytes: &[u8]) -> i64 {
        let mut neg = false;
        let mut i = 0;
        while i < bytes.len() && bytes[i] == b' ' { i += 1; }
        if i < bytes.len() && (bytes[i] == b'-' || bytes[i] == b'+') { neg = bytes[i] == b'-'; i += 1; }
        let mut val: i64 = 0;
        while i < bytes.len() {
            let c = bytes[i];
            if c < b'0' || c > b'9' { break; }
            val = val * 10 + (c - b'0') as i64;
            i += 1;
        }
        if neg { -val } else { val }
    }

    #[inline]
    fn parse_decimal_fast(bytes: &[u8]) -> f64 {
        let mut i = 0;
        let mut neg = false;
        if i < bytes.len() && bytes[i] == b'-' { neg = true; i += 1; }
        else if i < bytes.len() && bytes[i] == b'+' { i += 1; }
        let mut int_part = 0.0f64;
        while i < bytes.len() && bytes[i] >= b'0' && bytes[i] <= b'9' { int_part = int_part * 10.0 + (bytes[i] - b'0') as f64; i += 1; }
        let mut frac = 0.0f64; let mut scale = 1.0f64;
        if i < bytes.len() && bytes[i] == b'.' { i += 1; while i < bytes.len() && bytes[i] >= b'0' && bytes[i] <= b'9' { frac = frac * 10.0 + (bytes[i] - b'0') as f64; scale *= 10.0; i += 1; } }
        let mut r = int_part + frac / scale;
        if neg { r = -r; }
        r
    }

    #[inline]
    fn parse_f64_fast(bytes: &[u8]) -> f64 {
        let mut has_dot = false; let mut has_exp = false; let mut all_digits = true;
        for (i, &c) in bytes.iter().enumerate() {
            if c == b'.' { has_dot = true; continue; }
            if c == b'e' || c == b'E' { has_exp = true; continue; }
            if c == b'-' || c == b'+' { if i != 0 && bytes[i-1] != b'e' && bytes[i-1] != b'E' { all_digits = false; break; } continue; }
            if c < b'0' || c > b'9' { all_digits = false; break; }
        }
        if all_digits && !has_exp {
            if !has_dot { return parse_i64_fast(bytes) as f64; }
            return parse_decimal_fast(bytes);
        }
        std::str::from_utf8(bytes).ok().and_then(|s| s.trim().parse::<f64>().ok()).unwrap_or(f64::NAN)
    }

    #[inline]
    fn parse_line(line: &[u8], with_retreat: bool, with_afternoon_adjust: bool, code: &str, date: i64) -> Option<TradeRecord> {
        if line.is_empty() { return None; }
        let mut fields: [&[u8]; 15] = [&[][..]; 15];
        let mut start = 0; let mut col = 0;
        for (i, &b) in line.iter().enumerate() {
            if b == b',' { if col < 15 { fields[col] = &line[start..i]; } col += 1; start = i + 1; }
        }
        if col < 15 { fields[col] = &line[start..]; }
        // 非本股票 / 非本日期行直接丢（0=symbol, 2=date；date 解析成整数再比，覆盖 "20251031.0"）
        if fields[0] != code.as_bytes() || parse_i64_fast(fields[2]) != date { return None; }
        let flag_bytes = fields[COL_FLAG];
        // 先解析成数值再比较：部分日期的 flag 列写 "32.0"，按字节比 b"32" 会漏掉撤单
        let flag = if flag_bytes.is_empty() { 0 } else { parse_i64_fast(flag_bytes) as i32 };
        if !with_retreat && flag == 32 { return None; }
        let exchtime_us = parse_i64_fast(fields[COL_EXCHTIME]);
        let total_us = exchtime_us + OFFSET_US;
        let final_us = if with_afternoon_adjust {
            let day_offset = ((exchtime_us / 1_000_000) + 8 * 3600).rem_euclid(86400);
            if day_offset >= AFTERNOON_START_SEC && day_offset <= AFTERNOON_END_SEC { total_us - SHIFT_US }
            else if day_offset >= MORNING_START_SEC && day_offset <= MORNING_END_SEC { total_us }
            else { return None; }
        } else { total_us };
        let time_sec = (final_us as f64) / 1_000_000.0;
        Some(TradeRecord {
            time_us: final_us, time_sec: time_sec , price: parse_f64_fast(fields[COL_PRICE]) ,
            volume: parse_f64_fast(fields[COL_VOLUME]) , turnover: parse_f64_fast(fields[COL_TURNOVER]) ,
            flag, bid_order: parse_i64_fast(fields[COL_BID_ORDER]), ask_order: parse_i64_fast(fields[COL_ASK_ORDER]),
            index: parse_i64_fast(fields[COL_INDEX]),
        })
    }

    fn parse_chunk(data: &[u8], with_retreat: bool, adj: bool, code: &str, date: i64) -> Vec<TradeRecord> {
        let mut out = Vec::with_capacity(data.len() / 80 + 1);
        let mut start = 0;
        for i in 0..data.len() {
            if data[i] == b'\n' {
                let line = &data[start..i];
                let line = if line.last() == Some(&b'\r') { &line[..line.len()-1] } else { line };
                if let Some(r) = parse_line(line, with_retreat, adj, code, date) { out.push(r); }
                start = i + 1;
            }
        }
        if start < data.len() {
            let line = &data[start..];
            let line = if line.last() == Some(&b'\r') { &line[..line.len()-1] } else { line };
            if let Some(r) = parse_line(line, with_retreat, adj, code, date) { out.push(r); }
        }
        out
    }

    pub fn read_trade_fast(code: &str, date: i64) -> std::io::Result<Vec<TradeRecord>> {
        let filename = format!("{}_{}_transaction.csv", code, date);
        let path = resolve_stock_path(date, "transaction", &filename)?;
        let mut content = String::new();
        File::open(&path)?.read_to_string(&mut content)?;
        Ok(parse_chunk(content.as_bytes(), false, true, code, date))
    }

    #[inline]
    fn parse_market_line(line: &[u8], adj: bool, code: &str, date: i64) -> Option<MarketRecord> {
        if line.is_empty() { return None; }
        let mut fields: [&[u8]; 61] = [&[][..]; 61];
        let mut start = 0; let mut col = 0;
        for (i, &b) in line.iter().enumerate() {
            if b == b',' { if col < 61 { fields[col] = &line[start..i]; } col += 1; start = i + 1; }
        }
        if col < 61 { fields[col] = &line[start..]; }
        // 非本股票 / 非本日期行直接丢（0=symbol, 2=date）
        if fields[0] != code.as_bytes() || parse_i64_fast(fields[2]) != date { return None; }
        let ask_prc1 = parse_f64_fast(fields[MKT_COL_ASK_PRC_BASE]);
        let bid_prc1 = parse_f64_fast(fields[MKT_COL_ASK_PRC_BASE + 2]);
        if ask_prc1 == 0.0 || bid_prc1 == 0.0 { return None; }
        let exchtime_us = parse_i64_fast(fields[MKT_COL_EXCHTIME]);
        if exchtime_us == 0 { return None; }
        let total_us = exchtime_us + OFFSET_US;
        let final_us = if adj {
            let day_offset = ((exchtime_us / 1_000_000) + 8 * 3600).rem_euclid(86400);
            if day_offset >= AFTERNOON_START_SEC && day_offset <= AFTERNOON_END_SEC { total_us - SHIFT_US }
            else if day_offset >= MORNING_START_SEC && day_offset <= MORNING_END_SEC { total_us }
            else { return None; }
        } else { total_us };
        let time_sec = (final_us as f64) / 1_000_000.0;
        let final_time_sec = time_sec;
        let mut ask_prcs = [0f64; 10]; let mut ask_vols = [0f64; 10];
        let mut bid_prcs = [0f64; 10]; let mut bid_vols = [0f64; 10];
        for i in 0..10 {
            let base = MKT_COL_ASK_PRC_BASE + i * 4;
            ask_prcs[i] = parse_f64_fast(fields[base]) ;
            ask_vols[i] = parse_f64_fast(fields[base + 1]) ;
            bid_prcs[i] = parse_f64_fast(fields[base + 2]) ;
            bid_vols[i] = parse_f64_fast(fields[base + 3]) ;
        }
        Some(MarketRecord {
            time_us: final_us, time_sec: final_time_sec , last_prc: parse_f64_fast(fields[MKT_COL_LAST_PRC]) ,
            volume: parse_f64_fast(fields[MKT_COL_VOLUME]) , turnover: parse_f64_fast(fields[MKT_COL_TURNOVER]) ,
            total_ask_vol: parse_f64_fast(fields[MKT_COL_TOTAL_ASK_VOL]) , total_bid_vol: parse_f64_fast(fields[MKT_COL_TOTAL_BID_VOL]) ,
            ask_prcs, ask_vols, bid_prcs, bid_vols,
        })
    }

    fn parse_market_chunk(data: &[u8], adj: bool, code: &str, date: i64) -> Vec<MarketRecord> {
        let mut out = Vec::with_capacity(data.len() / 200 + 1);
        let mut start = 0;
        for i in 0..data.len() {
            if data[i] == b'\n' {
                let line = &data[start..i];
                let line = if line.last() == Some(&b'\r') { &line[..line.len()-1] } else { line };
                if let Some(r) = parse_market_line(line, adj, code, date) { out.push(r); }
                start = i + 1;
            }
        }
        if start < data.len() {
            let line = &data[start..];
            let line = if line.last() == Some(&b'\r') { &line[..line.len()-1] } else { line };
            if let Some(r) = parse_market_line(line, adj, code, date) { out.push(r); }
        }
        out
    }

    pub fn read_market_fast(code: &str, date: i64) -> std::io::Result<Vec<MarketRecord>> {
        let filename = format!("{}_{}_market_data.csv", code, date);
        let path = resolve_stock_path(date, "market_data", &filename)?;
        let mut content = String::new();
        File::open(&path)?.read_to_string(&mut content)?;
        Ok(parse_market_chunk(content.as_bytes(), true, code, date))
    }

    // ---- 不过滤不平移的原始读取（时段过滤交给调用方） ----
    pub fn read_trade_raw(code: &str, date: i64) -> std::io::Result<Vec<TradeRecord>> {
        let filename = format!("{}_{}_transaction.csv", code, date);
        let path = resolve_stock_path(date, "transaction", &filename)?;
        let mut content = String::new();
        File::open(&path)?.read_to_string(&mut content)?;
        Ok(parse_chunk(content.as_bytes(), false, false, code, date))
    }
    pub fn read_market_raw(code: &str, date: i64) -> std::io::Result<Vec<MarketRecord>> {
        let filename = format!("{}_{}_market_data.csv", code, date);
        let path = resolve_stock_path(date, "market_data", &filename)?;
        let mut content = String::new();
        File::open(&path)?.read_to_string(&mut content)?;
        Ok(parse_market_chunk(content.as_bytes(), false, code, date))
    }
}
use reader::{read_market_raw, read_trade_raw, MarketRecord, TradeRecord};

const K: usize = 4;
const DOM_CAP: [usize;3] = [1800, 900, 500];     // tick / market / aligned

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

/// 固定输出长度：3 时段 × (68 条流 × 3 指标 + 3200 三元组 × 5 组合 × 3 统计)
pub const OUT_LEN: usize = 144612;

fn tris() -> &'static [Vec<(usize, usize, usize)>; 3] {
    static T: OnceLock<[Vec<(usize, usize, usize)>; 3]> = OnceLock::new();
    T.get_or_init(all_tris)
}

/// 列名与输出严格对齐（进程内只算一次）
pub fn geffe_leak_names() -> Vec<String> {
    static N: OnceLock<Vec<String>> = OnceLock::new();
    N.get_or_init(|| all_names(tris())).clone()
}

fn per_stock_cached(trades: &[TradeRecord], mkt: &[MarketRecord]) -> Vec<f64> {
    per_stock(trades, mkt, tris())
}

/// 单股单日核心：读盘 → 计算，输出恒为 OUT_LEN 个 f32
pub fn compute_geffe_leak_full(code: &str, date: i64) -> std::io::Result<Vec<f32>> {
    let trades = read_trade_raw(code, date)?;
    let market = read_market_raw(code, date)?;
    Ok(compute_geffe_leak_from_records(&trades, &market))
}

/// 无 IO 内核：Python 传数组时也走这里，与读盘版逐位一致
pub fn compute_geffe_leak_from_records(
    trades: &[TradeRecord],
    market: &[MarketRecord],
) -> Vec<f32> {
    let feats = per_stock_cached(trades, market);
    debug_assert_eq!(feats.len(), OUT_LEN, "geffe_leak 列数必须恒定");
    feats.into_iter().map(|v| v as f32).collect()
}

/// 单股单日调试入口（错误抛异常，可见完整栈）
#[pyfunction]
pub fn py_geffe_leak(code: &str, date: i64) -> PyResult<Vec<f32>> {
    compute_geffe_leak_full(code, date)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("{e:?}")))
}

#[pyfunction]
pub fn py_geffe_leak_names() -> Vec<String> {
    geffe_leak_names()
}
