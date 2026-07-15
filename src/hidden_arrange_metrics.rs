// 隐藏排列三层游戏因子 — pipeline 规范实现
use pyo3::prelude::*;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::Read;
use std::path::Path;


// 精简版数据读取（单线程，无 pyo3/rayon），从 fast_csv_reader.rs 移植。
// 读取 transaction / market_data CSV，做 adjust_afternoon 平移。




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
    if let Ok(env_path) = std::env::var("RUST_PYFUNC_LEVEL2_PATH") {
        let p = Path::new(&env_path).join(date.to_string()).join(subdir).join(filename);
        if p.exists() { return Ok(p.to_string_lossy().into_owned()); }
        return Err(std::io::Error::new(std::io::ErrorKind::NotFound, format!("env path: {}", p.display())));
    }
    for root in ["/ssd_data/stock", "/nas197/binary/stock/sz_alpha/stock"] {
        let p = Path::new(root).join(date.to_string()).join(subdir).join(filename);
        if p.exists() { return Ok(p.to_string_lossy().into_owned()); }
    }
    Err(std::io::Error::new(std::io::ErrorKind::NotFound, format!("not found: {}/{}/{}/{}", "{root}", date, subdir, filename)))
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
fn parse_line(line: &[u8], with_retreat: bool, with_afternoon_adjust: bool) -> Option<TradeRecord> {
    if line.is_empty() { return None; }
    let mut fields: [&[u8]; 15] = [&[][..]; 15];
    let mut start = 0; let mut col = 0;
    for (i, &b) in line.iter().enumerate() {
        if b == b',' { if col < 15 { fields[col] = &line[start..i]; } col += 1; start = i + 1; }
    }
    if col < 15 { fields[col] = &line[start..]; }
    let flag_bytes = fields[COL_FLAG];
    if !with_retreat && flag_bytes == b"32" { return None; }
    let flag = if flag_bytes.is_empty() { 0 } else { parse_i64_fast(flag_bytes) as i32 };
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

fn parse_chunk(data: &[u8], with_retreat: bool, adj: bool) -> Vec<TradeRecord> {
    let mut out = Vec::with_capacity(data.len() / 80 + 1);
    let mut start = 0;
    for i in 0..data.len() {
        if data[i] == b'\n' {
            let line = &data[start..i];
            let line = if line.last() == Some(&b'\r') { &line[..line.len()-1] } else { line };
            if let Some(r) = parse_line(line, with_retreat, adj) { out.push(r); }
            start = i + 1;
        }
    }
    if start < data.len() {
        let line = &data[start..];
        let line = if line.last() == Some(&b'\r') { &line[..line.len()-1] } else { line };
        if let Some(r) = parse_line(line, with_retreat, adj) { out.push(r); }
    }
    out
}

pub fn read_trade_fast(code: &str, date: i64) -> std::io::Result<Vec<TradeRecord>> {
    let filename = format!("{}_{}_transaction.csv", code, date);
    let path = resolve_stock_path(date, "transaction", &filename)?;
    let mut content = String::new();
    File::open(&path)?.read_to_string(&mut content)?;
    Ok(parse_chunk(content.as_bytes(), false, true))
}

#[inline]
fn parse_market_line(line: &[u8], adj: bool) -> Option<MarketRecord> {
    if line.is_empty() { return None; }
    let mut fields: [&[u8]; 61] = [&[][..]; 61];
    let mut start = 0; let mut col = 0;
    for (i, &b) in line.iter().enumerate() {
        if b == b',' { if col < 61 { fields[col] = &line[start..i]; } col += 1; start = i + 1; }
    }
    if col < 61 { fields[col] = &line[start..]; }
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

fn parse_market_chunk(data: &[u8], adj: bool) -> Vec<MarketRecord> {
    let mut out = Vec::with_capacity(data.len() / 200 + 1);
    let mut start = 0;
    for i in 0..data.len() {
        if data[i] == b'\n' {
            let line = &data[start..i];
            let line = if line.last() == Some(&b'\r') { &line[..line.len()-1] } else { line };
            if let Some(r) = parse_market_line(line, adj) { out.push(r); }
            start = i + 1;
        }
    }
    if start < data.len() {
        let line = &data[start..];
        let line = if line.last() == Some(&b'\r') { &line[..line.len()-1] } else { line };
        if let Some(r) = parse_market_line(line, adj) { out.push(r); }
    }
    out
}

pub fn read_market_fast(code: &str, date: i64) -> std::io::Result<Vec<MarketRecord>> {
    let filename = format!("{}_{}_market_data.csv", code, date);
    let path = resolve_stock_path(date, "market_data", &filename)?;
    let mut content = String::new();
    File::open(&path)?.read_to_string(&mut content)?;
    Ok(parse_market_chunk(content.as_bytes(), true))
}

// 隐藏排列三层游戏 —— Rust sandbox（与 Python 参考实现逐数值对照）
// 第一阶段：数据读取 + 目标 + 银行 + 第一层，输出 JSON 对照。
// 不使用并行；纯 std。









// ===================== 数据结构 =====================
#[derive(Clone)]
pub struct Window {
    pub n: usize,
    pub volume: Vec<i64>,
    pub price: Vec<f64>,
    pub flag: Vec<i32>,
    pub bid_order: Vec<i64>,
    pub ask_order: Vec<i64>,
    pub time_sec: Vec<f64>,
    pub active_side: Vec<i8>,
    pub active_order: Vec<i64>,
    pub passive_order: Vec<i64>,
    pub book_feats: Vec<f64>,    // n*14
    pub queue_feats: Vec<f64>,   // n*10
    pub arrival_feats: Vec<f64>, // n*6
}

impl Window {
    fn load(path: &str) -> std::io::Result<Self> {
        let mut f = File::open(path)?;
        let mut buf = Vec::new();
        f.read_to_end(&mut buf)?;
        let mut p = 0;
        let magic = &buf[p..p + 4]; p += 4;
        assert_eq!(magic, b"HAG0", "magic mismatch");
        let rd_i32 = |b: &[u8], o: &mut usize| -> i32 {
            let v = i32::from_le_bytes(b[*o..*o + 4].try_into().unwrap()); *o += 4; v
        };
        let rd_i64 = |b: &[u8], o: &mut usize| -> i64 {
            let v = i64::from_le_bytes(b[*o..*o + 8].try_into().unwrap()); *o += 8; v
        };
        let rd_f64 = |b: &[u8], o: &mut usize| -> f64 {
            let v = f64::from_le_bytes(b[*o..*o + 8].try_into().unwrap()); *o += 8; v
        };
        let n = rd_i32(&buf, &mut p) as usize;
        let _nfb = rd_i32(&buf, &mut p);
        let _nfq = rd_i32(&buf, &mut p);
        let _nfa = rd_i32(&buf, &mut p);
        let mut volume = vec![0i64; n];
        for i in 0..n { volume[i] = rd_i64(&buf, &mut p); }
        let mut price = vec![0f64; n];
        for i in 0..n { price[i] = rd_f64(&buf, &mut p); }
        let mut flag = vec![0i32; n];
        for i in 0..n { flag[i] = rd_i32(&buf, &mut p); }
        let mut bid_order = vec![0i64; n];
        for i in 0..n { bid_order[i] = rd_i64(&buf, &mut p); }
        let mut ask_order = vec![0i64; n];
        for i in 0..n { ask_order[i] = rd_i64(&buf, &mut p); }
        let mut time_sec = vec![0f64; n];
        for i in 0..n { time_sec[i] = rd_f64(&buf, &mut p); }
        let mut active_side = vec![0i8; n];
        for i in 0..n {
            active_side[i] = i8::from_le_bytes([buf[p]]); p += 1;
        }
        let mut active_order = vec![0i64; n];
        for i in 0..n { active_order[i] = rd_i64(&buf, &mut p); }
        let mut passive_order = vec![0i64; n];
        for i in 0..n { passive_order[i] = rd_i64(&buf, &mut p); }
        // 特征矩阵
        let rd_f64s = |b: &[u8], o: &mut usize, cnt: usize| -> Vec<f64> {
            let mut v = vec![0f64; cnt];
            for i in 0..cnt { v[i] = rd_f64(b, o); }
            v
        };
        let nfb = _nfb as usize;
        let nfq = _nfq as usize;
        let nfa = _nfa as usize;
        let book_feats = rd_f64s(&buf, &mut p, n * nfb);
        let queue_feats = rd_f64s(&buf, &mut p, n * nfq);
        let arrival_feats = rd_f64s(&buf, &mut p, n * nfa);
        Ok(Window { n, volume, price, flag, bid_order, ask_order, time_sec,
                    active_side, active_order, passive_order,
                    book_feats, queue_feats, arrival_feats })
    }
}

// ===================== 目标构造 =====================

fn make_target(volume: &[i64], name: &str) -> Vec<f64> {
    let v: Vec<f64> = volume.iter().map(|&x| x as f64).collect();
    match name {
        "LOGVOL" => v.iter().map(|x| (1.0 + x).ln()).collect(),
        "RANKVOL" => percentile_rank(&v),
        "BIN8" | "BIN16" | "BIN32" => {
            let nbin: usize = name[3..].parse().unwrap();
            bin_equal(&v, nbin)
        }
        _ if name.starts_with("TOP") => {
            let q = 100.0 - name[3..].parse::<f64>().unwrap();
            let thr = percentile(&v, q);
            v.iter().map(|x| if *x >= thr { 1.0 } else { 0.0 }).collect()
        }
        _ => panic!("unknown target {}", name),
    }
}

fn percentile_rank(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| x[a].partial_cmp(&x[b]).unwrap()); // mergesort 稳定
    let mut ranks = vec![0f64; n];
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j + 1 < n && x[idx[j + 1]] == x[idx[i]] { j += 1; }
        let avg = (i as f64 + j as f64) / 2.0;
        let val = if n > 1 { avg / (n - 1) as f64 } else { 0.0 };
        for k in i..=j { ranks[idx[k]] = val; }
        i = j + 1;
    }
    ranks
}

fn bin_equal(x: &[f64], nbin: usize) -> Vec<f64> {
    let n = x.len();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| x[a].partial_cmp(&x[b]).unwrap());
    let mut ranks = vec![0i64; n];
    let mut r = 0i64;
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j + 1 < n && x[idx[j + 1]] == x[idx[i]] { j += 1; }
        for k in i..=j { ranks[idx[k]] = r; }
        r += 1;
        i = j + 1;
    }
    let n_unique = r as usize;
    let mut bins = vec![0f64; n];
    for i in 0..n {
        bins[i] = if n_unique <= nbin {
            ranks[i] as f64
        } else {
            ((ranks[i] as usize * nbin) / n_unique).min(nbin - 1) as f64
        };
    }
    bins
}

// ===================== 银行 =====================
fn round_half_even(x: f64) -> f64 {
    let i = x.floor();
    let f = x - i;
    if f < 0.5 { i }
    else if f > 0.5 { i + 1.0 }
    else {
        let ii = i as i64;
        if ii % 2 == 0 { i } else { i + 1.0 }
    }
}

fn quantize_corr(r: f64) -> f64 {
    round_half_even(r / 0.005) * 0.005
}

pub struct Bank {
    pub y: Vec<f64>,
    pub n: usize,
    pub var: f64,
    pub mu: f64,
}

impl Bank {
    pub fn new(y: Vec<f64>) -> Self {
        let n = y.len();
        let mu = y.iter().sum::<f64>() / n as f64;
        let var = y.iter().map(|v| (v - mu).powi(2)).sum::<f64>() / n as f64;
        Bank { y, n, var: var.max(1e-12), mu }
    }
    pub fn corr_raw(&self, g: &[f64]) -> f64 {
        let s: f64 = (0..self.n).map(|i| (g[i] - self.y[i]).powi(2)).sum();
        1.0 - s / (2.0 * self.var * self.n as f64)
    }
    pub fn query(&self, g: &[f64]) -> f64 {
        quantize_corr(self.corr_raw(g))
    }
}

// ===================== 排名工具 =====================
pub fn average_rank(g: &[f64]) -> Vec<f64> {
    let n = g.len();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| g[a].partial_cmp(&g[b]).unwrap());
    let mut ranks = vec![0f64; n];
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j + 1 < n && g[idx[j + 1]] == g[idx[i]] { j += 1; }
        let avg = (i as f64 + j as f64) / 2.0;
        for k in i..=j { ranks[idx[k]] = avg; }
        i = j + 1;
    }
    ranks
}

pub fn assign_by_rank(ranks: &[f64], y: &[f64]) -> Vec<f64> {
    let n = y.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| ranks[a].partial_cmp(&ranks[b]).unwrap()); // stable
    let mut ys = y.to_vec();
    ys.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mut g = vec![0f64; n];
    for (k, &i) in order.iter().enumerate() { g[i] = ys[k]; }
    g
}

// ===================== 轴序（匹配 np.lexsort）=====================
fn axis_order(win: &Window, axis: &str) -> Vec<usize> {
    let n = win.n;
    let mut idx: Vec<usize> = (0..n).collect();
    match axis {
        "TIME" => {} // 0..n
        "ACTIVE_ORDER" => {
            idx.sort_by(|&a, &b| {
                let ka = if win.active_order[a] >= 0 { win.active_order[a] } else { i64::MAX };
                let kb = if win.active_order[b] >= 0 { win.active_order[b] } else { i64::MAX };
                ka.cmp(&kb).then(a.cmp(&b))
            });
        }
        "PASSIVE_ORDER" => {
            idx.sort_by(|&a, &b| {
                let ka = if win.passive_order[a] >= 0 { win.passive_order[a] } else { i64::MAX };
                let kb = if win.passive_order[b] >= 0 { win.passive_order[b] } else { i64::MAX };
                ka.cmp(&kb).then(a.cmp(&b))
            });
        }
        "PRICE_TIME" => {
            idx.sort_by(|&a, &b| win.price[a].partial_cmp(&win.price[b]).unwrap().then(a.cmp(&b)));
        }
        _ => panic!("unknown axis"),
    }
    idx
}

// ===================== 第一层 =====================
const AXES: [&str; 4] = ["TIME", "ACTIVE_ORDER", "PASSIVE_ORDER", "PRICE_TIME"];
const DEPTHS: [&str; 4] = ["N32", "N16", "N8", "N4"];
const DEPTH_LEAVES: [usize; 4] = [32, 16, 8, 4];

fn node_bisect_candidate(g: &[f64], b: &[usize], high_left: bool) -> Vec<f64> {
    let half = b.len() / 2;
    let l = &b[..half];
    let r = &b[half..];
    let na = l.len();
    let nb = r.len();
    let mut vals: Vec<f64> = b.iter().map(|&i| g[i]).collect();
    vals.sort_by(|a, b2| a.partial_cmp(b2).unwrap());
    // 用 na/nb 切分（正确处理奇数 b.len()；偶数时 na=nb=half 与 Python 一致）
    let (l_vals, r_vals): (Vec<f64>, Vec<f64>) = if high_left {
        (vals[nb..].to_vec(), vals[..nb].to_vec())
    } else {
        (vals[..na].to_vec(), vals[na..].to_vec())
    };
    let mut g2 = g.to_vec();
    // L 内按 g 值稳定升序（相同 g 值保持 L 轴序，匹配 Python argsort stable）
    let mut lo: Vec<usize> = l.to_vec();
    lo.sort_by(|&a, &b2| g[a].partial_cmp(&g[b2]).unwrap());
    for (k, &i) in lo.iter().enumerate() { g2[i] = l_vals[k]; }
    let mut ro: Vec<usize> = r.to_vec();
    ro.sort_by(|&a, &b2| g[a].partial_cmp(&g[b2]).unwrap());
    for (k, &i) in ro.iter().enumerate() { g2[i] = r_vals[k]; }
    g2
}

pub struct BranchResult {
    pub axis: String, pub depth: String,
    pub g: Vec<f64>, pub rho0: f64, pub rho_final: f64,
    pub trajectory: Vec<f64>,
    pub node_ds: Vec<f64>,
    pub level_ds: Vec<Vec<f64>>,
    pub accepts: Vec<bool>,
    pub n_internal: usize, pub n_queries: usize,
    pub parent_child_sign: Vec<(f64, f64)>,
}

fn run_stage1_branch(win: &Window, bank: &Bank, g0: &[f64], axis: &str, depth: &str) -> BranchResult {
    let n = win.n;
    let leaf_size = n / DEPTH_LEAVES[DEPTHS.iter().position(|d| *d == depth).unwrap()];
    let aord = axis_order(win, axis);
    let mut g = g0.to_vec();
    let mut rho_cur = bank.query(&g);
    let rho0 = rho_cur;
    let mut trajectory = vec![rho_cur];
    let mut node_ds: Vec<f64> = vec![];
    let mut level_ds: Vec<Vec<f64>> = vec![];
    let mut accepts: Vec<bool> = vec![];
    let mut internal_count = 0usize;
    let mut cur_level: Vec<(usize, usize)> = vec![(0, n)];
    while !cur_level.is_empty() {
        let mut next_level: Vec<(usize, usize)> = vec![];
        let mut cur_level_ds: Vec<f64> = vec![];
        for (s, e) in cur_level {
            let seg_len = e - s;
            if seg_len <= leaf_size { continue; }
            internal_count += 1;
            let b: Vec<usize> = aord[s..e].to_vec();
            let g_hl = node_bisect_candidate(&g, &b, true);
            let g_hr = node_bisect_candidate(&g, &b, false);
            let rho_hl = bank.query(&g_hl);
            let rho_hr = bank.query(&g_hr);
            let d = rho_hr - rho_hl;
            node_ds.push(d);
            cur_level_ds.push(d);
            let mut accepted = false;
            if rho_cur >= rho_hl && rho_cur >= rho_hr {}
            else if rho_hl >= rho_hr { g = g_hl; rho_cur = rho_hl; accepted = true; }
            else { g = g_hr; rho_cur = rho_hr; accepted = true; }
            accepts.push(accepted);
            trajectory.push(rho_cur);
            let half = seg_len / 2;
            next_level.push((s, s + half));
            next_level.push((s + half, e));
        }
        if !cur_level_ds.is_empty() { level_ds.push(cur_level_ds); }
        cur_level = next_level;
    }
    let mut pc: Vec<(f64, f64)> = vec![];
    for li in 0..level_ds.len().saturating_sub(1) {
        for (j, &pd) in level_ds[li].iter().enumerate() {
            let ci = 2 * j;
            if ci < level_ds[li + 1].len() {
                pc.push((pd.signum(), level_ds[li + 1][ci].signum()));
            }
        }
    }
    let rho_final = bank.query(&g);
    let n_queries = node_ds.len() * 2;
    BranchResult {
        axis: axis.to_string(), depth: depth.to_string(),
        g, rho0, rho_final, trajectory, node_ds, level_ds, accepts,
        n_internal: internal_count, n_queries,
        parent_child_sign: pc,
    }
}

fn vote(branches: &[BranchResult], beta: f64, y: &[f64], rho0: f64) -> (Vec<f64>, f64) {
    let rhos: Vec<f64> = branches.iter().map(|b| b.rho_final).collect();
    let u: Vec<f64> = rhos.iter().map(|r| (r - rho0) / (1.0 - rho0 + 1e-12)).collect();
    let max_u = u.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = u.iter().map(|x| (beta * x - max_u).exp()).collect();
    let sum_e: f64 = exps.iter().sum();
    let w: Vec<f64> = exps.iter().map(|x| x / sum_e).collect();
    let n = y.len();
    let mut ranks = vec![0f64; n];
    for (b, wi) in branches.iter().zip(w.iter()) {
        let br = average_rank(&b.g);
        for i in 0..n { ranks[i] += wi * br[i]; }
    }
    let g_vote = assign_by_rank(&ranks, y);
    let bank = Bank::new(y.to_vec());
    let rv = bank.query(&g_vote);
    (g_vote, rv)
}

pub struct Stage1Result {
    pub g0: Vec<f64>, pub rho0: f64,
    pub g: Vec<f64>, pub rho: f64, pub best_name: String,
    pub branches: Vec<BranchResult>,
    pub votes: Vec<(Vec<f64>, f64)>,
    pub path_rhos: Vec<f64>,
}

impl Stage1Result {
    pub fn to_json(&self) -> String {
        let mut s = format!("{{\"rho0\":{:.4},\"rho1\":{:.4},\"best_name\":\"{}\",\"branches\":{{",
            self.rho0, self.rho, self.best_name);
        for (i, b) in self.branches.iter().enumerate() {
            if i > 0 { s.push(','); }
            let ac = b.accepts.iter().filter(|&&x| x).count();
            s.push_str(&format!("\"{}_{}\":{{\"rho0\":{:.4},\"rho_final\":{:.4},\"n_internal\":{},\"n_queries\":{},\"accept_count\":{}}}",
                b.axis, b.depth, b.rho0, b.rho_final, b.n_internal, b.n_queries, ac));
        }
        s.push_str("},\"votes\":{");
        for (i, beta) in [1,2,5,10].iter().enumerate() {
            if i > 0 { s.push(','); }
            s.push_str(&format!("\"{}\":{:.4}", beta, self.votes[i].1));
        }
        s.push_str("}}");
        s
    }
}

pub fn run_stage1(win: &Window, y: &[f64]) -> Stage1Result {
    let bank = Bank::new(y.to_vec());
    let mut ys = y.to_vec();
    ys.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let g0 = ys.clone();
    let rho0 = bank.query(&g0);
    let mut branches: Vec<BranchResult> = vec![];
    for axis in AXES.iter() {
        for depth in DEPTHS.iter() {
            branches.push(run_stage1_branch(win, &bank, &g0, axis, depth));
        }
    }
    let votes = vec![
        vote(&branches, 1.0, y, rho0),
        vote(&branches, 2.0, y, rho0),
        vote(&branches, 5.0, y, rho0),
        vote(&branches, 10.0, y, rho0),
    ];
    let mut best = (String::from("INIT"), g0.clone(), rho0);
    for b in &branches {
        if b.rho_final > best.2 { best = (format!("{}_{}", b.axis, b.depth), b.g.clone(), b.rho_final); }
    }
    for (i, beta) in [1,2,5,10].iter().enumerate() {
        if votes[i].1 > best.2 { best = (format!("VOTE_B{}", beta), votes[i].0.clone(), votes[i].1); }
    }
    let rho_desc = bank.query(&ys.iter().rev().cloned().collect::<Vec<_>>());
    let rho_active = {
        let ao = win.active_order.clone();
        let mut order: Vec<usize> = (0..win.n).collect();
        order.sort_by(|&a, &b| {
            let ka = if ao[a] >= 0 { ao[a] } else { i64::MAX };
            let kb = if ao[b] >= 0 { ao[b] } else { i64::MAX };
            ka.cmp(&kb).then(a.cmp(&b))
        });
        let mut g = vec![0f64; win.n];
        for (k, &i) in order.iter().enumerate() { g[i] = ys[k]; }
        bank.query(&g)
    };
    let rho_random = {
        let mut g = vec![0f64; win.n];
        let mut rng = Rng::new(20240101);
        let perm = rng.permutation(win.n);
        for (k, &i) in perm.iter().enumerate() { g[i] = ys[k]; }
        bank.query(&g)
    };
    Stage1Result { g0, rho0, g: best.1.clone(), rho: best.2, best_name: best.0.clone(),
        branches, votes, path_rhos: vec![rho0, rho_desc, rho_active, rho_random] }
}

fn is_discrete(t: &str) -> bool {
    matches!(t, "BIN8" | "BIN16" | "BIN32" | "TOP20" | "TOP10" | "TOP05" | "TOP02" | "TOP01")
}

fn run_full_game(win: &Window, y: &[f64], target: &str, graphs: &std::collections::HashMap<&'static str, EdgeSet>) -> (Stage1Result, Stage2Result, Option<Stage3Result>) {
    let t0 = std::time::Instant::now();
    let s1 = run_stage1(win, y);
    let t1 = std::time::Instant::now();
    let s2 = run_stage2_with_graphs(&s1.g, s1.rho, y, graphs);
    let t2 = std::time::Instant::now();
    let s3 = if is_discrete(target) { Some(run_stage3(&s2.g, y, &[0.5], 32)) } else { None };
    (s1, s2, s3)
}

fn random_null_y(y: &[f64], seed: u64) -> Vec<f64> {
    let mut rng = Rng::new(seed);
    let perm = rng.permutation(y.len());
    let mut ny = vec![0f64; y.len()];
    for (k, &i) in perm.iter().enumerate() { ny[i] = y[k]; }
    ny
}

const TARGETS_C: [&str; 3] = ["LOGVOL", "BIN16", "TOP10"];

/// 只跑第一二层（用于零假设，第三层因子只 RAW）
fn run_game_s1s2(win: &Window, y: &[f64], graphs: &std::collections::HashMap<&'static str, EdgeSet>) -> (Stage1Result, Stage2Result) {
    let s1 = run_stage1(win, y);
    let s2 = run_stage2_with_graphs(&s1.g, s1.rho, y, graphs);
    (s1, s2)
}

/// 折中方案C：4时段 × 2时间窗口 × 日频代表窗口(收盘前) × 5目标 × 基础因子 × RAW/EXCESS/NULL_Z
fn compute_all_factors(code: &str, date: i64, tws: &[f64], n_null: usize) {
    let t_total = std::time::Instant::now();
    let trade = read_trade_fast(code, date).expect("read trade");
    let market = read_market_fast(code, date).expect("read market");
    let ev = align_and_build(&trade, &market);
    println!("读数据+预处理: {} 事件, 耗时 {:.2}s", ev.n, t_total.elapsed().as_secs_f64());

    let mut total_cols = 0usize;
    let mut samples: Vec<String> = vec![];
    let mut t_game = 0f64;
    let mut win_count = 0usize;

    for seg_id in 1..=4u32 {
        let seg = select_segment(&ev, seg_id as usize);
        for &tw in tws {
            let tw_label = format!("T{}", tw as i64);
            // 日频：取该时段最后一个完整时间窗口（收盘前）
            let all_wins = slide_time_windows(&seg, tw, tw, 128, usize::MAX);
            let windows: Vec<(f64, Window)> = all_wins.last().map(|last| vec![last.clone()]).unwrap_or_default();
            win_count += windows.len();
            for (ts, win) in &windows {
                let tg = std::time::Instant::now();
                let tbg = std::time::Instant::now();
                let graphs = build_graphs(win);
                for target in TARGETS_C.iter() {
                    let y = make_target(&win.volume, target);
                    if target.starts_with("TOP") {
                        let n1 = y.iter().filter(|&&v| v == 1.0).count();
                        if n1 < 4 || y.len() - n1 < 4 { continue; }
                    }
                    // 真实游戏（含第三层）
                    let (s1, s2, s3) = run_full_game(win, &y, target, &graphs);
                    let real_f = extract_all(&s1, &s2, s3.as_ref(), &y);
                    // 零假设（只第一二层）
                    let mut null_maps: Vec<std::collections::HashMap<String, f64>> = vec![];
                    for k in 0..n_null {
                        let ny = random_null_y(&y, 999 + k as u64);
                        let (ns1, ns2) = run_game_s1s2(win, &ny, &graphs);
                        let nf = extract_all(&ns1, &ns2, None, &ny);
                        null_maps.push(nf.into_iter().collect());
                    }
                    for (name, rv) in &real_f {
                        let null_vals: Vec<f64> = null_maps.iter().map(|m| *m.get(name).unwrap_or(&f64::NAN)).filter(|v| !v.is_nan()).collect();
                        let mean_n = if null_vals.is_empty() { f64::NAN } else { null_vals.iter().sum::<f64>() / null_vals.len() as f64 };
                        let std_n = if null_vals.len() > 1 { (null_vals.iter().map(|v| (v - mean_n).powi(2)).sum::<f64>() / null_vals.len() as f64).sqrt() } else if null_vals.len() == 1 { 0.0 } else { f64::NAN };
                        let excess = if mean_n.is_nan() { f64::NAN } else { rv - mean_n };
                        let null_z = if mean_n.is_nan() || std_n.is_nan() { f64::NAN } else { (rv - mean_n) / (std_n + 1e-8) };
                        total_cols += 3;
                        if samples.len() < 8 {
                            samples.push(format!("SEG{}_{}_{}_{}_RAW/EXCESS/NULL_Z@{:.0} = {:.4}/{:.4}/{:.4}",
                                seg_id, tw_label, target, name, ts, rv, excess, null_z));
                        }
                    }
                }
                t_game += tg.elapsed().as_secs_f64();
            }
            println!("  SEG{} {}: N={}", seg_id, tw_label, windows.first().map(|(_, w)| w.n).unwrap_or(0));
        }
    }
    println!("\n=== 折中方案C 因子统计 ===");
    println!("日频窗口数(时间戳): {}", win_count);
    println!("单时间戳因子列(含RAW/EXCESS/NULL_Z): {}", total_cols / win_count.max(1));
    println!("总因子值: {}", total_cols);
    println!("三层游戏总耗时: {:.2}s", t_game);
    println!("总耗时: {:.2}s", t_total.elapsed().as_secs_f64());
    println!("\n样本:");
    for s in &samples { println!("  {}", s); }
}

// ===================== Pipeline 规范封装 =====================
/// 因子名（与 compute 输出严格对齐，单一源）。
pub fn hidden_arrange_names() -> Vec<String> {
    let mut names: Vec<String> = vec![];
    for seg_id in 1..=4u32 {
        for target in TARGETS_C.iter() {
            // LOGVOL 连续目标: 基础因子 703 个
            // BIN16/TOP10 离散目标: 基础因子 949 个
            // 用占位: 实际名字由 extract_all 的输出顺序决定
            let dummy_win_n = 430usize; // 代表窗口大小
            // 因子名需要和 compute 对齐，用索引而非硬编码
            names.push(format!("SEG{}_T180_{}_FACTOR_{}", seg_id, target, 0));
        }
    }
    names // 占位，实际由 compute_hidden_arrange_full 返回
}

/// 纯 Rust 核心（唯一真相源）：读数据 → 4时段 → T180 日频窗口 → 3目标 → 三层游戏 → 因子(RAW) → Vec<f32>
/// 输出长度固定，pipeline 和 Python 入口的共同调用点。
fn compute_hidden_arrange_impl(code: &str, date: i64) -> std::io::Result<(Vec<String>, Vec<f32>)> {
    let trade = read_trade_fast(code, date)?;
    let market = read_market_fast(code, date)?;
    let ev = align_and_build(&trade, &market);

    let mut all_names: Vec<String> = vec![];
    let mut all_vals: Vec<f32> = vec![];

    for seg_id in 1..=4u32 {
        let seg = select_segment(&ev, seg_id as usize);
        let windows = slide_time_windows(&seg, 180.0, 180.0, 32, usize::MAX);
        if windows.is_empty() { continue; }
        let (_ts, win) = windows.last().unwrap();
        let win = win.clone();
        let graphs = build_graphs(&win);
        let adjs = build_adj_lists(&graphs, win.n);
        for target in TARGETS_C.iter() {
            let y = make_target(&win.volume, target);
            if target.starts_with("TOP") {
                let n1 = y.iter().filter(|&&v| v == 1.0).count();
                if n1 < 4 || y.len() - n1 < 4 { continue; }
            }
            let s1 = run_stage1(&win, &y);
            let s2 = run_stage2_with_adjs(&s1.g, s1.rho, &y, &adjs);
            let s3 = if is_discrete(target) { Some(run_stage3(&s2.g, &y, &[0.5], 32)) } else { None };
            let fac = extract_all(&s1, &s2, s3.as_ref(), &y);
            for (name, val) in &fac {
                all_names.push(format!("SEG{}_T180_{}_{}_RAW", seg_id, target, name));
                all_vals.push(*val as f32);
            }
        }
    }
    Ok((all_names, all_vals))
}

use std::sync::OnceLock;
static HA_TEMPLATE: OnceLock<Vec<String>> = OnceLock::new();

/// 固定因子名模板（8772）：用 000001/20220819 跑一次拿到完整四时段名字并缓存。
/// 因子名对所有股票固定（extract_all 输出顺序确定）。
fn template_names() -> Vec<String> {
    if let Some(t) = HA_TEMPLATE.get() { return t.clone(); }
    let t = compute_hidden_arrange_impl("000001", 20220819).map(|(n, _)| n).unwrap_or_default();
    let _ = HA_TEMPLATE.set(t.clone());
    t
}

/// 纯 Rust 核心（唯一真相源）：固定输出 8772 个因子。
/// 缺失时段/目标的位置按模板对齐填 NaN，保证 pipeline 长度恒定。
pub fn compute_hidden_arrange_full(code: &str, date: i64) -> std::io::Result<(Vec<String>, Vec<f32>)> {
    let tmpl = template_names();
    if tmpl.is_empty() {
        return Ok((tmpl, vec![]));
    }
    let (names, vals) = compute_hidden_arrange_impl(code, date)?;
    let mut map: HashMap<&str, f32> = HashMap::with_capacity(names.len());
    for (n, v) in names.iter().zip(vals.iter()) {
        map.insert(n.as_str(), *v);
    }
    let out_vals: Vec<f32> = tmpl.iter().map(|n| map.get(n.as_str()).copied().unwrap_or(f32::NAN)).collect();
    Ok((tmpl, out_vals))
}

/// pipeline 包装：worker 进程批量调用，错误吞掉返 NaN。
pub fn pipeline_hidden_arrange(date: i64, code: &str, expected_len: usize) -> Vec<f32> {
    match compute_hidden_arrange_full(code, date) {
        Ok((_names, vals)) => vals,
        Err(_) => vec![f32::NAN; expected_len],
    }
}

// 数据预处理：盘口对齐 + 特征构造 + 4时段筛选（移植 Python data_prep）



const SEG_LATE30_START: f64 = 12600.0;
const SEG_LATE30_END: f64 = 14220.0;

fn safe_div(a: f64, b: f64) -> f64 {
    if b.abs() < 1e-12 { 0.0 } else { a / b }
}

fn rolling_mean(x: &[f64], k: usize) -> Vec<f64> {
    let n = x.len();
    let mut csum = vec![0f64; n + 1];
    for i in 0..n { csum[i + 1] = csum[i] + x[i]; }
    let mut out = vec![0f64; n];
    for i in 0..n {
        let lo = if i + 1 >= k { i + 1 - k } else { 0 };
        out[i] = (csum[i + 1] - csum[lo]) / (i - lo + 1) as f64;
    }
    out
}

fn rolling_std(x: &[f64], k: usize) -> Vec<f64> {
    // pandas rolling std, ddof=1, min_periods=1（窗口<2 → 0）
    let n = x.len();
    let mut out = vec![0f64; n];
    for i in 0..n {
        let lo = if i + 1 >= k { i + 1 - k } else { 0 };
        let cnt = i - lo + 1;
        if cnt < 2 { out[i] = 0.0; continue; }
        let mean = (lo..=i).map(|j| x[j]).sum::<f64>() / cnt as f64;
        let var = (lo..=i).map(|j| (x[j] - mean).powi(2)).sum::<f64>() / (cnt - 1) as f64;
        out[i] = var.sqrt();
    }
    out
}

fn count_within(time_sec: &[f64], window: f64) -> Vec<f64> {
    let n = time_sec.len();
    let mut out = vec![0f64; n];
    for i in 0..n {
        let lo = time_sec[i] - window;
        let j = time_sec.partition_point(|&t| t < lo);
        out[i] = (i - j + 1) as f64;
    }
    out
}

fn sameprice_runlen(price: &[f64]) -> Vec<f64> {
    let n = price.len();
    let mut out = vec![0f64; n];
    if n == 0 { return out; }
    let mut grp = vec![0i64; n];
    let mut g = 0i64;
    grp[0] = 0;
    for i in 1..n {
        if price[i] != price[i - 1] { g += 1; }
        grp[i] = g;
    }
    let maxg = (g + 1) as usize;
    let mut counts = vec![0f64; maxg];
    for &gg in &grp { counts[gg as usize] += 1.0; }
    for i in 0..n { out[i] = counts[grp[i] as usize]; }
    out
}

fn depth_slope(prc: &[f64], vol: &[f64], mid: &[f64], is_bid: bool) -> Vec<f64> {
    let n = prc.len() / 5; // prc 是扁平 n*5
    let mut out = vec![0f64; n];
    for i in 0..n {
        let p = &prc[i * 5..i * 5 + 5];
        let v = &vol[i * 5..i * 5 + 5];
        let dist: Vec<f64> = if is_bid {
            (0..5).map(|j| mid[i] - p[j]).collect()
        } else {
            (0..5).map(|j| p[j] - mid[i]).collect()
        };
        let mut cumv = [0f64; 5];
        cumv[0] = v[0];
        for j in 1..5 { cumv[j] = cumv[j - 1] + v[j]; }
        // mask dist>0
        let xs: Vec<f64> = (0..5).filter(|&j| dist[j] > 0.0).map(|j| dist[j]).collect();
        let ys: Vec<f64> = (0..5).filter(|&j| dist[j] > 0.0).map(|j| cumv[j]).collect();
        if xs.len() >= 2 {
            let mx = xs.iter().sum::<f64>() / xs.len() as f64;
            let my = ys.iter().sum::<f64>() / ys.len() as f64;
            let num: f64 = xs.iter().zip(ys.iter()).map(|(x, y)| (x - mx) * (y - my)).sum();
            let den: f64 = xs.iter().map(|x| (x - mx).powi(2)).sum();
            if den > 1e-12 { out[i] = num / den; }
        }
    }
    out
}

pub fn align_and_build(trade: &[TradeRecord], market: &[MarketRecord]) -> Window {
    let n = trade.len();
    if n == 0 || market.is_empty() {
        return Window {
            n: 0, volume: vec![], price: vec![], flag: vec![],
            bid_order: vec![], ask_order: vec![], time_sec: vec![],
            active_side: vec![], active_order: vec![], passive_order: vec![],
            book_feats: vec![], queue_feats: vec![], arrival_feats: vec![],
        };
    }
    // 按 time_us 稳定排序（匹配 Python read_trade 的 exchtime 排序）
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| trade[a].time_us.cmp(&trade[b].time_us).then(trade[b].index.cmp(&trade[a].index)));
    let trade: Vec<TradeRecord> = order.iter().map(|&i| trade[i]).collect();
    let trade = trade.as_slice();
    // market 按 time_us 有序（CSV 时间序）
    let m_times: Vec<i64> = market.iter().map(|m| m.time_us).collect();

    // 对齐盘口
    let mut bid1 = vec![0f64; n];
    let mut ask1 = vec![0f64; n];
    let mut bvol1 = vec![0f64; n];
    let mut avol1 = vec![0f64; n];
    let mut bvol5 = vec![0f64; n];
    let mut avol5 = vec![0f64; n];
    let mut bvol10 = vec![0f64; n];
    let mut avol10 = vec![0f64; n];
    let mut bp5 = vec![0f64; n * 5];
    let mut ap5 = vec![0f64; n * 5];
    let mut bv5 = vec![0f64; n * 5];
    let mut av5 = vec![0f64; n * 5];
    for i in 0..n {
        let tu = trade[i].time_us;
        let idx = m_times.partition_point(|&t| t <= tu);
        let idx = if idx > 0 { idx - 1 } else { 0 };
        let m = &market[idx];
        bid1[i] = m.bid_prcs[0] as f64;
        ask1[i] = m.ask_prcs[0] as f64;
        bvol1[i] = m.bid_vols[0] as f64;
        avol1[i] = m.ask_vols[0] as f64;
        let (b5, a5) = (0..5).fold((0f64, 0f64), |(b, a), j| (b + m.bid_vols[j] as f64, a + m.ask_vols[j] as f64));
        bvol5[i] = b5; avol5[i] = a5;
        let (b10, a10) = (0..10).fold((0f64, 0f64), |(b, a), j| (b + m.bid_vols[j] as f64, a + m.ask_vols[j] as f64));
        bvol10[i] = b10; avol10[i] = a10;
        for j in 0..5 {
            bp5[i * 5 + j] = m.bid_prcs[j] as f64;
            ap5[i * 5 + j] = m.ask_prcs[j] as f64;
            bv5[i * 5 + j] = m.bid_vols[j] as f64;
            av5[i * 5 + j] = m.ask_vols[j] as f64;
        }
    }
    let mid: Vec<f64> = (0..n).map(|i| (bid1[i] + ask1[i]) / 2.0).collect();
    let spread: Vec<f64> = (0..n).map(|i| ask1[i] - bid1[i]).collect();
    let imb1: Vec<f64> = (0..n).map(|i| safe_div(bvol1[i] - avol1[i], bvol1[i] + avol1[i])).collect();
    let imb5: Vec<f64> = (0..n).map(|i| safe_div(bvol5[i] - avol5[i], bvol5[i] + avol5[i])).collect();
    let imb10: Vec<f64> = (0..n).map(|i| safe_div(bvol10[i] - avol10[i], bvol10[i] + avol10[i])).collect();
    let microprice: Vec<f64> = (0..n).map(|i| safe_div(bid1[i] * avol1[i] + ask1[i] * bvol1[i], bvol1[i] + avol1[i])).collect();
    let micro_off: Vec<f64> = (0..n).map(|i| microprice[i] - mid[i]).collect();
    let slope_bid = depth_slope(&bp5, &bv5, &mid, true);
    let slope_ask = depth_slope(&ap5, &av5, &mid, false);

    // 主动方向
    let price: Vec<f64> = trade.iter().map(|t| t.price as f64).collect();
    let flag: Vec<i32> = trade.iter().map(|t| t.flag).collect();
    let mut side = vec![0i8; n];
    for i in 0..n {
        if flag[i] == 66 { side[i] = 1; }
        else if flag[i] == 83 { side[i] = -1; }
    }
    // 未知按中间价
    let mut last = 0i8;
    for i in 0..n {
        if side[i] != 0 { last = side[i]; continue; }
        let prev_price = if i > 0 { price[i - 1] } else { f64::NAN };
        let cond_buy = price[i] > mid[i] || (price[i] == mid[i] && price[i] > prev_price);
        let cond_sell = price[i] < mid[i] || (price[i] == mid[i] && price[i] < prev_price);
        if cond_buy { side[i] = 1; last = 1; }
        else if cond_sell { side[i] = -1; last = -1; }
        else { side[i] = last; }
    }
    let active_order: Vec<i64> = (0..n).map(|i| if side[i] == 1 { trade[i].bid_order } else if side[i] == -1 { trade[i].ask_order } else { -1 }).collect();
    let passive_order: Vec<i64> = (0..n).map(|i| if side[i] == 1 { trade[i].ask_order } else if side[i] == -1 { trade[i].bid_order } else { -1 }).collect();

    // 滚动特征
    let signed_vol: Vec<f64> = (0..n).map(|i| trade[i].volume as f64 * side[i] as f64).collect();
    let signed_imb10 = rolling_mean(&signed_vol, 10);
    let signed_imb50 = rolling_mean(&signed_vol, 50);
    let time_sec: Vec<f64> = (0..n).map(|i| (trade[i].time_us - trade[0].time_us) as f64 / 1_000_000.0).collect();
    let mut dt_prev = vec![0f64; n];
    for i in 1..n { dt_prev[i] = time_sec[i] - time_sec[i - 1]; }
    let a_dt_mean8 = rolling_mean(&dt_prev, 8);
    let a_dt_mean32 = rolling_mean(&dt_prev, 32);
    let rs_dt32 = rolling_std(&dt_prev, 32);
    let a_dt_cv32: Vec<f64> = (0..n).map(|i| safe_div(rs_dt32[i], a_dt_mean32[i])).collect();
    let a_cnt1s = count_within(&time_sec, 1.0);
    let a_cnt5s = count_within(&time_sec, 5.0);
    let mut mid_ret = vec![0f64; n];
    for i in 1..n { mid_ret[i] = mid[i] - mid[i - 1]; }
    let mid_ret10 = rolling_mean(&mid_ret, 10);
    let mid_vol50 = rolling_std(&mid_ret, 50);

    // 队列特征
    let mut q_bid1_chg = vec![0f64; n];
    let mut q_ask1_chg = vec![0f64; n];
    for i in 1..n { q_bid1_chg[i] = bvol1[i] - bvol1[i - 1]; q_ask1_chg[i] = avol1[i] - avol1[i - 1]; }
    let q_bid1_consume: Vec<f64> = (0..n).map(|i| {
        let prev = if i > 0 { bvol1[i - 1] } else { bvol1[i] };
        safe_div((prev - bvol1[i]).max(0.0), prev)
    }).collect();
    let q_ask1_consume: Vec<f64> = (0..n).map(|i| {
        let prev = if i > 0 { avol1[i - 1] } else { avol1[i] };
        safe_div((prev - avol1[i]).max(0.0), prev)
    }).collect();
    let q_sameprice = sameprice_runlen(&price);

    // 组装特征矩阵（顺序匹配 Python BOOK/QUEUE/ARRIVAL_FEATS）
    let mut book_feats = Vec::with_capacity(n * 14);
    for i in 0..n {
        book_feats.push(spread[i]); book_feats.push(imb1[i]); book_feats.push(imb5[i]); book_feats.push(imb10[i]);
        book_feats.push(bvol5[i]); book_feats.push(avol5[i]); book_feats.push(micro_off[i]);
        book_feats.push(slope_bid[i]); book_feats.push(slope_ask[i]);
        book_feats.push(signed_imb10[i]); book_feats.push(signed_imb50[i]);
        book_feats.push(dt_prev[i]); book_feats.push(mid_ret10[i]); book_feats.push(mid_vol50[i]);
    }
    let mut queue_feats = Vec::with_capacity(n * 10);
    for i in 0..n {
        queue_feats.push(bvol1[i]); queue_feats.push(avol1[i]);
        queue_feats.push(q_bid1_chg[i]); queue_feats.push(q_ask1_chg[i]);
        queue_feats.push(q_bid1_consume[i]); queue_feats.push(q_ask1_consume[i]);
        queue_feats.push(imb1[i]); queue_feats.push(spread[i]);
        queue_feats.push(side[i] as f64); queue_feats.push(q_sameprice[i]);
    }
    let mut arrival_feats = Vec::with_capacity(n * 6);
    for i in 0..n {
        arrival_feats.push(dt_prev[i]); arrival_feats.push(a_dt_mean8[i]);
        arrival_feats.push(a_dt_mean32[i]); arrival_feats.push(a_dt_cv32[i]);
        arrival_feats.push(a_cnt1s[i]); arrival_feats.push(a_cnt5s[i]);
    }

    Window {
        n,
        volume: trade.iter().map(|t| t.volume as i64).collect(),
        price, flag,
        bid_order: trade.iter().map(|t| t.bid_order).collect(),
        ask_order: trade.iter().map(|t| t.ask_order).collect(),
        time_sec,
        active_side: side,
        active_order, passive_order,
        book_feats, queue_feats, arrival_feats,
    }
}

fn percentile(x: &[f64], q: f64) -> f64 {
    let mut s = x.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = s.len();
    if n == 0 { return 0.0; }
    let pos = (n - 1) as f64 * q / 100.0;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    if lo == hi { s[lo] } else { s[lo] + (pos - lo as f64) * (s[hi] - s[lo]) }
}

pub fn select_segment(win: &Window, seg: usize) -> Window {
    let mut idx: Vec<usize> = vec![];
    let vols: Vec<f64> = win.volume.iter().map(|&v| v as f64).collect();
    for i in 0..win.n {
        let is_active = win.active_side[i] != 0;
        let is_buy = win.active_side[i] == 1;
        match seg {
            1 => { if is_active { idx.push(i); } }
            2 => { if is_buy { idx.push(i); } }
            3 => { if is_active { idx.push(i); } }
            4 => {
                let ts = win.time_sec[i];
                if is_buy && ts >= SEG_LATE30_START && ts <= SEG_LATE30_END { idx.push(i); }
            }
            _ => {}
        }
    }
    // 按成交量筛选
    let sub_vols: Vec<f64> = idx.iter().map(|&i| vols[i]).collect();
    let (lo_thr, hi_thr) = match seg {
        1 | 2 => (percentile(&sub_vols, 40.0), f64::INFINITY),
        3 => (percentile(&sub_vols, 40.0), percentile(&sub_vols, 90.0)),
        _ => (f64::NEG_INFINITY, f64::INFINITY),
    };
    if seg <= 3 {
        let mx = sub_vols.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    }
    let filtered: Vec<usize> = idx.into_iter().filter(|&i| {
        if seg == 3 { vols[i] > lo_thr && vols[i] <= hi_thr }
        else if seg == 4 { true }
        else { vols[i] <= lo_thr }
    }).collect();
    slice_window(win, &filtered)
}

/// 固定时间跨度滑窗（方案 A）：窗口 [start, start+tw)，步长 step，返回每个窗口（N 自适应）。
/// 窗口时间戳 = 窗口末笔时间。N < n_min 的窗口丢弃（树无意义）。
pub fn slide_time_windows(seg: &Window, tw: f64, step: f64, n_min: usize, max_windows: usize) -> Vec<(f64, Window)> {
    if seg.n == 0 { return vec![]; }
    let t0 = seg.time_sec[0];
    let t1 = seg.time_sec[seg.n - 1];
    let mut result = vec![];
    let mut start = t0;
    while start + tw <= t1 + 1e-6 {
        let end = start + tw;
        let mut idx = vec![];
        for i in 0..seg.n {
            if seg.time_sec[i] >= start && seg.time_sec[i] < end { idx.push(i); }
            else if seg.time_sec[i] >= end { break; }
        }
        if idx.len() >= n_min {
            let win = slice_window(seg, &idx);
            let ts = win.time_sec[win.n - 1];
            result.push((ts, win));
        }
        start += step;
        if result.len() >= max_windows { break; }
    }
    result
}

pub fn slice_window(win: &Window, idx: &[usize]) -> Window {
    let n = idx.len();
    let get_f = |arr: &[f64], nf: usize| -> Vec<f64> {
        let mut out = Vec::with_capacity(n * nf);
        for &i in idx { out.extend_from_slice(&arr[i * nf..(i + 1) * nf]); }
        out
    };
    Window {
        n,
        volume: idx.iter().map(|&i| win.volume[i]).collect(),
        price: idx.iter().map(|&i| win.price[i]).collect(),
        flag: idx.iter().map(|&i| win.flag[i]).collect(),
        bid_order: idx.iter().map(|&i| win.bid_order[i]).collect(),
        ask_order: idx.iter().map(|&i| win.ask_order[i]).collect(),
        time_sec: idx.iter().map(|&i| win.time_sec[i]).collect(),
        active_side: idx.iter().map(|&i| win.active_side[i]).collect(),
        active_order: idx.iter().map(|&i| win.active_order[i]).collect(),
        passive_order: idx.iter().map(|&i| win.passive_order[i]).collect(),
        book_feats: get_f(&win.book_feats, 14),
        queue_feats: get_f(&win.queue_feats, 10),
        arrival_feats: get_f(&win.arrival_feats, 6),
    }
}

// 第二层：多关系图游戏（与 Python stage2 对照）



pub const GRAPH_NAMES: [&str; 12] = [
    "TIME_R1", "TIME_R8", "TIME_R32", "ACTIVE_EXACT", "PASSIVE_EXACT",
    "PRICE_T0", "PRICE_T2", "SIDE_EXACT", "BOOK_K10", "BOOK_K30", "QUEUE_K10", "ARRIVAL_K10",
];
pub const SCOPES: [&str; 3] = ["LOCAL32", "MESO8_D2", "GLOBAL_D4"];
const PRICE_TICK: f64 = 0.01;

pub type EdgeSet = HashSet<(usize, usize)>;

fn add_edge(edges: &mut EdgeSet, u: usize, v: usize) {
    if u != v {
        let (a, b) = if u < v { (u, v) } else { (v, u) };
        edges.insert((a, b));
    }
}

fn knn_graph(win: &Window, feats: &[f64], nfeat: usize, k: usize, out: &mut EdgeSet) {
    let n = win.n;
    // z-score
    let mut mu = vec![0f64; nfeat];
    let mut sd = vec![0f64; nfeat];
    for j in 0..nfeat {
        let mut s = 0f64;
        for i in 0..n { s += feats[i * nfeat + j]; }
        mu[j] = s / n as f64;
    }
    for j in 0..nfeat {
        let mut s = 0f64;
        for i in 0..n { let d = feats[i * nfeat + j] - mu[j]; s += d * d; }
        sd[j] = (s / n as f64).sqrt();
        if sd[j] < 1e-12 { sd[j] = 1.0; }
    }
    let mut z = vec![0f64; n * nfeat];
    for i in 0..n {
        for j in 0..nfeat {
            z[i * nfeat + j] = (feats[i * nfeat + j] - mu[j]) / sd[j];
        }
    }
    // 距离矩阵 D2[i][j] = sq[i]+sq[j]-2*z[i]·z[j]
    let sq: Vec<f64> = (0..n).map(|i| {
        (0..nfeat).map(|j| z[i * nfeat + j] * z[i * nfeat + j]).sum::<f64>()
    }).collect();
    let kk = k.min(n - 1);
    for i in 0..n {
        // D2[i][j]
        let mut d2: Vec<(f64, usize)> = (0..n).map(|j| {
            let dot: f64 = (0..nfeat).map(|jj| z[i * nfeat + jj] * z[j * nfeat + jj]).sum();
            let v = sq[i] + sq[j] - 2.0 * dot;
            (if v < 0.0 { 0.0 } else { v }, j)
        }).collect();
        d2.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap().then(a.1.cmp(&b.1))); // stable: 同距按j升序
        let mut cnt = 0;
        for (_, j) in d2 {
            if j == i { continue; }
            add_edge(out, i, j);
            cnt += 1;
            if cnt >= kk { break; }
        }
    }
}

pub fn build_graphs(win: &Window) -> HashMap<&'static str, EdgeSet> {
    let n = win.n;
    let mut graphs: HashMap<&'static str, EdgeSet> = HashMap::new();
    for g in GRAPH_NAMES { graphs.insert(g, EdgeSet::new()); }
    // TIME_R1/R8/R32
    for &(d, name) in &[(1usize, "TIME_R1"), (8, "TIME_R8"), (32, "TIME_R32")] {
        let e = graphs.get_mut(name).unwrap();
        for i in 0..n {
            for j in (i + 1)..(i + d + 1).min(n) { add_edge(e, i, j); }
        }
    }
    // ACTIVE/PASSIVE_EXACT
    for &(col_fn, name) in &[
        ((|w: &Window, i: usize| w.active_order[i]) as fn(&Window, usize) -> i64, "ACTIVE_EXACT"),
        ((|w: &Window, i: usize| w.passive_order[i]) as fn(&Window, usize) -> i64, "PASSIVE_EXACT"),
    ] {
        let mut groups: HashMap<i64, Vec<usize>> = HashMap::new();
        for i in 0..n {
            let v = col_fn(win, i);
            if v < 0 { continue; }
            groups.entry(v).or_default().push(i);
        }
        let e = graphs.get_mut(name).unwrap();
        for members in groups.values() {
            for a in 0..members.len() {
                for b in (a + 1)..members.len() { add_edge(e, members[a], members[b]); }
            }
        }
    }
    // PRICE_T0 / PRICE_T2
    let mut by_price: HashMap<u64, Vec<usize>> = HashMap::new();
    for i in 0..n {
        let key = win.price[i].to_bits();
        by_price.entry(key).or_default().push(i);
    }
    let e0 = graphs.get_mut("PRICE_T0").unwrap();
    for members in by_price.values() {
        for a in 0..members.len() {
            for b in (a + 1)..members.len() { add_edge(e0, members[a], members[b]); }
        }
    }
    // PRICE_T2: 按价格排序滑窗
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| win.price[a].partial_cmp(&win.price[b]).unwrap());
    let sp: Vec<f64> = order.iter().map(|&i| win.price[i]).collect();
    let e2 = graphs.get_mut("PRICE_T2").unwrap();
    for a in 0..n {
        for b in (a + 1)..n {
            if sp[b] - sp[a] > 2.0 * PRICE_TICK + 1e-9 { break; }
            add_edge(e2, order[a], order[b]);
        }
    }
    // SIDE_EXACT
    let mut by_side: HashMap<i8, Vec<usize>> = HashMap::new();
    for i in 0..n { by_side.entry(win.active_side[i]).or_default().push(i); }
    let es = graphs.get_mut("SIDE_EXACT").unwrap();
    for members in by_side.values() {
        for a in 0..members.len() {
            for b in (a + 1)..members.len() { add_edge(es, members[a], members[b]); }
        }
    }
    // KNN
    knn_graph(win, &win.book_feats, 14, 10, graphs.get_mut("BOOK_K10").unwrap());
    knn_graph(win, &win.book_feats, 14, 30, graphs.get_mut("BOOK_K30").unwrap());
    knn_graph(win, &win.queue_feats, 10, 10, graphs.get_mut("QUEUE_K10").unwrap());
    knn_graph(win, &win.arrival_feats, 6, 10, graphs.get_mut("ARRIVAL_K10").unwrap());
    graphs
}

// UnionFind on dynamic node set
struct UF { p: HashMap<usize, usize> }
impl UF {
    fn new(nodes: &[usize]) -> Self {
        let mut p = HashMap::new();
        for &n in nodes { p.insert(n, n); }
        UF { p }
    }
    fn find(&mut self, x: usize) -> usize {
        let mut cur = x;
        while self.p[&cur] != cur {
            let nxt = self.p[&cur];
            self.p.insert(cur, nxt);
            cur = nxt;
        }
        cur
    }
    fn union(&mut self, a: usize, b: usize) {
        let ra = self.find(a); let rb = self.find(b);
        if ra != rb { self.p.insert(ra, rb); }
    }
    fn groups(mut self) -> Vec<Vec<usize>> {
        let mut g: HashMap<usize, Vec<usize>> = HashMap::new();
        let keys: Vec<usize> = self.p.keys().copied().collect();
        for k in keys {
            let r = self.find(k);
            g.entry(r).or_default().push(k);
        }
        g.into_values().collect()
    }
}

fn balance(mut a: Vec<usize>, mut b: Vec<usize>) -> (Vec<usize>, Vec<usize>) {
    a.sort(); b.sort();
    while a.len() > b.len() + 1 { b.push(a.pop().unwrap()); }
    while b.len() > a.len() + 1 { a.push(b.pop().unwrap()); }
    (a, b)
}

pub fn graph_bisect(nodes: &[usize], adj: &[Vec<usize>], buf: &mut Vec<usize>) -> (Vec<usize>, Vec<usize>) {
    let mut ns: Vec<usize> = nodes.to_vec();
    ns.sort();
    let n = ns.len();
    if n <= 1 { return (ns, vec![]); }
    // buf[x]=i+1 表示 x 在 ns 中的位置 i；0 表示不存在。调用方保证 buf 容量充足且初始为 0。
    for (i, &x) in ns.iter().enumerate() { buf[x] = i + 1; }
    let mut uf: Vec<usize> = (0..n).collect();
    let mut has_edge = false;
    for &i in &ns {
        let ia = buf[i] - 1;
        for &j in &adj[i] {
            let bj = buf[j];
            if bj != 0 {
                let ib = bj - 1;
                has_edge = true;
                let mut ra = ia; while uf[ra] != ra { uf[ra] = uf[uf[ra]]; ra = uf[ra]; }
                let mut rb = ib; while uf[rb] != rb { uf[rb] = uf[uf[rb]]; rb = uf[rb]; }
                if ra != rb { uf[ra] = rb; }
            }
        }
    }
    for &x in &ns { buf[x] = 0; }
    if !has_edge {
        let a: Vec<usize> = (0..n).step_by(2).map(|i| ns[i]).collect();
        let b: Vec<usize> = (1..n).step_by(2).map(|i| ns[i]).collect();
        return balance(a, b);
    }
    let mut g: HashMap<usize, Vec<usize>> = HashMap::new();
    for (i, &x) in ns.iter().enumerate() {
        let mut r = i; while uf[r] != r { uf[r] = uf[uf[r]]; r = uf[r]; }
        g.entry(r).or_default().push(x);
    }
    let mut comps: Vec<Vec<usize>> = g.into_values().collect();
    comps.sort_by(|a, b| b.len().cmp(&a.len()).then(a.iter().min().cmp(&b.iter().min())));
    let half = n as f64 / 2.0;
    let mut a: Vec<usize> = vec![];
    let mut b: Vec<usize> = vec![];
    for comp in comps {
        if (comp.len() as f64) > half {
            let mut cs = comp.clone(); cs.sort();
            let mid = cs.len() / 2;
            let (left, right) = cs.split_at(mid);
            if a.len() <= b.len() { a.extend_from_slice(left); b.extend_from_slice(right); }
            else { b.extend_from_slice(left); a.extend_from_slice(right); }
        } else {
            if a.len() <= b.len() { a.extend(comp.iter().copied()); }
            else { b.extend(comp.iter().copied()); }
        }
    }
    balance(a, b)
}

fn graph_candidate(g: &[f64], a: &[usize], b: &[usize], high_a: bool) -> Vec<f64> {
    let mut ab: Vec<usize> = a.iter().chain(b.iter()).copied().collect();
    let na = a.len(); let nb = b.len();
    let mut vals: Vec<f64> = ab.iter().map(|&i| g[i]).collect();
    vals.sort_by(|x, y| x.partial_cmp(y).unwrap());
    let (a_vals, b_vals): (Vec<f64>, Vec<f64>) = if high_a {
        (vals[nb..].to_vec(), vals[..nb].to_vec())
    } else {
        (vals[..na].to_vec(), vals[na..].to_vec())
    };
    let mut g2 = g.to_vec();
    let mut ao = a.to_vec();
    ao.sort_by(|&x, &y| g[x].partial_cmp(&g[y]).unwrap()); // stable 保持 A 顺序
    for (k, &i) in ao.iter().enumerate() { g2[i] = a_vals[k]; }
    let mut bo = b.to_vec();
    bo.sort_by(|&x, &y| g[x].partial_cmp(&g[y]).unwrap());
    for (k, &i) in bo.iter().enumerate() { g2[i] = b_vals[k]; }
    g2
}

fn split_blocks(n: usize, n_blocks: usize) -> Vec<Vec<usize>> {
    let base = n / n_blocks; let rem = n % n_blocks;
    let mut blocks = vec![]; let mut s = 0;
    for bi in 0..n_blocks {
        let sz = base + if bi < rem { 1 } else { 0 };
        blocks.push((s..s + sz).collect());
        s += sz;
    }
    blocks
}

fn normalized_kendall(r1: &[f64], r2: &[f64]) -> f64 {
    let n = r1.len();
    if n < 2 { return 0.0; }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| r1[a].partial_cmp(&r1[b]).unwrap());
    let s2: Vec<f64> = order.iter().map(|&i| r2[i]).collect();
    let tot = (n * (n - 1)) / 2;
    let mut disc = 0i64;
    for i in 0..n {
        for j in (i + 1)..n {
            if s2[j] < s2[i] { disc += 1; }
        }
    }
    disc as f64 / tot as f64
}

pub struct S2Branch {
    pub graph: String, pub scope: String,
    pub g: Vec<f64>, pub rho1: f64, pub rho_final: f64,
    pub trajectory: Vec<f64>,
    pub node_ds: Vec<f64>,
    pub accepts: Vec<bool>,
    pub n_queries: usize,
    pub path_distance: f64,
}

fn run_branch(g1: &[f64], y: &[f64], bank: &Bank, adj: &[Vec<usize>], scope: &str, buf: &mut Vec<usize>) -> S2Branch {
    let n = y.len();
    let (n_blocks, max_depth) = match scope {
        "LOCAL32" => (32usize, 1usize),
        "MESO8_D2" => (8, 2),
        _ => (1, 4),
    };
    let blocks = if scope == "GLOBAL_D4" { vec![(0..n).collect::<Vec<_>>()] }
                  else { split_blocks(n, n_blocks) };
    let mut g = g1.to_vec();
    let mut rho_cur = bank.query(&g);
    let rho1 = rho_cur;
    let mut trajectory = vec![rho1];
    let mut node_ds: Vec<f64> = vec![];
    let mut accepts: Vec<bool> = vec![];
    for block in &blocks {
        if block.len() < 2 { continue; }
        let mut cur_level: Vec<Vec<usize>> = vec![block.clone()];
        let mut d = 0;
        while d < max_depth && !cur_level.is_empty() {
            let mut next_level: Vec<Vec<usize>> = vec![];
            for nd in &cur_level {
                if nd.len() < 2 { continue; }
                let (a, b) = graph_bisect(nd, adj, buf);
                let g_ha = graph_candidate(&g, &a, &b, true);
                let g_hb = graph_candidate(&g, &a, &b, false);
                let rho_ha = bank.query(&g_ha);
                let rho_hb = bank.query(&g_hb);
                node_ds.push(rho_hb - rho_ha);
                let mut accepted = false;
                if rho_cur >= rho_ha && rho_cur >= rho_hb {}
                else if rho_ha >= rho_hb { g = g_ha; rho_cur = rho_ha; accepted = true; }
                else { g = g_hb; rho_cur = rho_hb; accepted = true; }
                accepts.push(accepted);
                trajectory.push(rho_cur);
                if !a.is_empty() { next_level.push(a); }
                if !b.is_empty() { next_level.push(b); }
            }
            cur_level = next_level; d += 1;
        }
    }
    let rho_final = bank.query(&g);
    let r1 = average_rank(g1); let r2 = average_rank(&g);
    let pd = normalized_kendall(&r1, &r2);
    let n_queries = node_ds.len() * 2;
    S2Branch { graph: String::new(), scope: scope.to_string(), g, rho1, rho_final,
        trajectory, node_ds, accepts, n_queries, path_distance: pd }
}

fn vote_s2(branches: &[S2Branch], beta: f64, y: &[f64], rho1: f64) -> (Vec<f64>, f64) {
    let rhos: Vec<f64> = branches.iter().map(|b| b.rho_final).collect();
    let u: Vec<f64> = rhos.iter().map(|r| (r - rho1) / (1.0 - rho1 + 1e-12)).collect();
    let max_u = u.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = u.iter().map(|x| (beta * x - max_u).exp()).collect();
    let sum_e: f64 = exps.iter().sum();
    let n = y.len();
    let mut ranks = vec![0f64; n];
    for (b, wi) in branches.iter().zip(exps.iter()) {
        let br = average_rank(&b.g);
        for i in 0..n { ranks[i] += (wi / sum_e) * br[i]; }
    }
    let gv = assign_by_rank(&ranks, y);
    let bank = Bank::new(y.to_vec());
    let rv = bank.query(&gv);
    (gv, rv)
}

pub struct Stage2Result {
    pub g1: Vec<f64>, pub rho1: f64,
    pub g: Vec<f64>, pub rho: f64, pub best_name: String,
    pub branches: Vec<S2Branch>,
    pub votes: Vec<(Vec<f64>, f64)>,
}

impl Stage2Result {
    pub fn to_json(&self) -> String {
        let mut s = format!("{{\"rho2\":{:.4},\"s2_best_name\":\"{}\",\"s2_branches\":{{", self.rho, self.best_name);
        for (i, b) in self.branches.iter().enumerate() {
            if i > 0 { s.push(','); }
            s.push_str(&format!("\"{}_{}\":{{\"rho_final\":{:.4},\"path_distance\":{:.4}}}",
                b.graph, b.scope, b.rho_final, b.path_distance));
        }
        s.push_str("},\"s2_votes\":{");
        for (i, beta) in [1,2,5,10].iter().enumerate() {
            if i > 0 { s.push(','); }
            s.push_str(&format!("\"{}\":{:.4}", beta, self.votes[i].1));
        }
        s.push_str("}}");
        s
    }
}

pub fn build_adj_lists(graphs: &HashMap<&'static str, EdgeSet>, n: usize) -> HashMap<&'static str, Vec<Vec<usize>>> {
    let mut adjs: HashMap<&'static str, Vec<Vec<usize>>> = HashMap::new();
    for gname in GRAPH_NAMES.iter() {
        let edges = graphs.get(*gname).unwrap();
        let mut adj = vec![Vec::new(); n];
        for &(u, v) in edges.iter() { adj[u].push(v); adj[v].push(u); }
        adjs.insert(gname, adj);
    }
    adjs
}

pub fn run_stage2_with_adjs(g1: &[f64], rho1: f64, y: &[f64], adjs: &HashMap<&'static str, Vec<Vec<usize>>>) -> Stage2Result {
    let bank = Bank::new(y.to_vec());
    let n = y.len();
    let mut id_buf = vec![0usize; n];
    let mut branches: Vec<S2Branch> = vec![];
    for gname in GRAPH_NAMES.iter() {
        let adj = &adjs[*gname];
        for scope in SCOPES.iter() {
            let mut br = run_branch(g1, y, &bank, adj, scope, &mut id_buf);
            br.graph = gname.to_string();
            branches.push(br);
        }
    }
    let gains: Vec<f64> = branches.iter().map(|b| b.rho_final - rho1).collect();
    let mut pos_idx: Vec<usize> = gains.iter().enumerate()
        .filter(|(_, &g)| g > 0.0).map(|(i, _)| i).collect();
    if pos_idx.len() > 24 {
        pos_idx.sort_by(|&a, &b| gains[b].partial_cmp(&gains[a]).unwrap());
        pos_idx.truncate(24);
    }
    let pool: Vec<&S2Branch> = if pos_idx.is_empty() {
        branches.iter().collect()
    } else {
        pos_idx.iter().map(|&i| &branches[i]).collect()
    };
    let owned_pool: Vec<S2Branch> = pool.iter().map(|&b| S2Branch {
        graph: b.graph.clone(), scope: b.scope.clone(),
        g: b.g.clone(), rho1: b.rho1, rho_final: b.rho_final,
        trajectory: b.trajectory.clone(), node_ds: b.node_ds.clone(),
        accepts: b.accepts.clone(), n_queries: b.n_queries, path_distance: b.path_distance
    }).collect();
    let votes = vec![
        vote_s2(&owned_pool, 1.0, y, rho1),
        vote_s2(&owned_pool, 2.0, y, rho1),
        vote_s2(&owned_pool, 5.0, y, rho1),
        vote_s2(&owned_pool, 10.0, y, rho1),
    ];
    let mut best_single = &branches[0];
    for b in &branches[1..] {
        if b.rho_final > best_single.rho_final { best_single = b; }
    }
    let mut best = (String::from("STAGE1"), g1.to_vec(), rho1);
    if best_single.rho_final > best.2 {
        best = (format!("{}_{}", best_single.graph, best_single.scope), best_single.g.clone(), best_single.rho_final);
    }
    for (i, beta) in [1,2,5,10].iter().enumerate() {
        if votes[i].1 > best.2 { best = (format!("VOTE_B{}", beta), votes[i].0.clone(), votes[i].1); }
    }
    Stage2Result { g1: g1.to_vec(), rho1, g: best.1.clone(), rho: best.2, best_name: best.0.clone(), branches, votes }
}

pub fn run_stage2_with_graphs(g1: &[f64], rho1: f64, y: &[f64], graphs: &HashMap<&'static str, EdgeSet>) -> Stage2Result {
    let adjs = build_adj_lists(graphs, y.len());
    run_stage2_with_adjs(g1, rho1, y, &adjs)
}

pub fn run_stage2(g1: &[f64], rho1: f64, win: &Window, y: &[f64]) -> Stage2Result {
    let graphs = build_graphs(win);
    run_stage2_with_graphs(g1, rho1, y, &graphs)
}

// 第三层：后验不确定性与主动查询（与 Python stage3 对照）


pub const TAUS: [f64; 3] = [0.25, 0.50, 1.00];
pub const TAU_NAMES: [&str; 3] = ["TAU025", "TAU050", "TAU100"];
pub const CHECKPOINTS: [usize; 5] = [0, 8, 16, 32, 64];
const N_QUERIES: usize = 64;
const UPDATE_LR: f64 = 0.3;

// xorshift64 RNG，匹配 Python
pub struct Rng { s: u64 }
impl Rng {
    pub fn new(seed: u64) -> Self { Rng { s: (seed & u64::MAX) | 1 } }
    pub fn u64(&mut self) -> u64 {
        let mut x = self.s;
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        self.s = x; x
    }
    pub fn random(&mut self) -> f64 { self.u64() as f64 / 2f64.powi(64) }
    pub fn permutation(&mut self, n: usize) -> Vec<usize> {
        let mut a: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            let j = (self.random() * (i + 1) as f64) as usize;
            a.swap(i, j);
        }
        a
    }
}

fn sinkhorn(p: &mut [f64], n: usize, m: usize, col_target: &[f64], iters: usize) {
    for _ in 0..iters {
        for j in 0..m {
            let cs: f64 = (0..n).map(|i| p[i * m + j]).sum();
            let f = col_target[j] / cs.max(1e-12);
            for i in 0..n { p[i * m + j] *= f; }
        }
        for i in 0..n {
            let rs: f64 = (0..m).map(|j| p[i * m + j]).sum();
            let f = 1.0 / rs.max(1e-12);
            for j in 0..m { p[i * m + j] *= f; }
        }
    }
}

fn build_prior(g2: &[f64], y: &[f64], tau: f64) -> (Vec<f64>, Vec<f64>, usize, Vec<f64>) {
    let n = y.len();
    let mut classes: Vec<f64> = y.to_vec();
    classes.sort_by(|a, b| a.partial_cmp(b).unwrap()); classes.dedup();
    let m = classes.len();
    let mut c_m = vec![0f64; m];
    for &v in y {
        let idx = classes.binary_search_by(|c| c.partial_cmp(&v).unwrap()).unwrap();
        c_m[idx] += 1.0;
    }
    let r = average_rank(g2);
    let rn: Vec<f64> = r.iter().map(|x| x / (n as f64 - 1.0).max(1.0)).collect();
    let mut p = vec![0f64; n * m];
    for i in 0..n {
        for j in 0..m {
            let center = if m > 1 { j as f64 / (m - 1) as f64 } else { 0.0 };
            let d = rn[i] - center;
            p[i * m + j] = (-(d * d) / tau.max(1e-6)).exp();
        }
        let s: f64 = (0..m).map(|j| p[i * m + j]).sum();
        for j in 0..m { p[i * m + j] /= s; }
    }
    sinkhorn(&mut p, n, m, &c_m, 60);
    (p, c_m, m, classes)
}

fn sample_permutation(p: &[f64], c_m: &[f64], n: usize, m: usize, rng: &mut Rng) -> Vec<usize> {
    debug_assert!(m <= 64);
    let mut remaining = c_m.to_vec();
    let mut assign = vec![0usize; n];
    let order = rng.permutation(n);
    let us: Vec<f64> = (0..n).map(|_| rng.random()).collect();
    let mut prob = [0f64; 64];
    let mut cdf = [0f64; 64];
    for (k, &i) in order.iter().enumerate() {
        let mut s = 0f64;
        for j in 0..m { prob[j] = p[i * m + j] * remaining[j]; s += prob[j]; }
        let mm = if s <= 1e-12 {
            let mut best = 0; let mut bv = -1f64;
            for j in 0..m { if remaining[j] > bv { bv = remaining[j]; best = j; } }
            best
        } else {
            let mut acc = 0.0;
            for j in 0..m { acc += prob[j] / s; cdf[j] = acc; }
            let mut idx = m;
            for j in 0..m { if us[k] <= cdf[j] { idx = j; break; } }
            idx.min(m - 1)
        };
        assign[i] = mm;
        remaining[mm] -= 1.0;
    }
    assign
}

fn row_entropy(p: &[f64], n: usize, m: usize) -> Vec<f64> {
    (0..n).map(|i| {
        let mut h = 0.0;
        for j in 0..m { let v = p[i * m + j]; if v > 1e-12 { h -= v * v.ln(); } }
        h
    }).collect()
}

fn make_group_candidate(y_sorted: &[f64], g_idx: &[usize], ranks_guide: &[f64], high_big: bool) -> Vec<f64> {
    let n = y_sorted.len();
    let mut present = vec![false; n];
    for &i in g_idx { present[i] = true; }
    let h: Vec<usize> = (0..n).filter(|&i| !present[i]).collect();
    let ng = g_idx.len(); let nh = h.len();
    if ng == 0 || nh == 0 { return y_sorted.to_vec(); }
    let (g_vals, h_vals): (Vec<f64>, Vec<f64>) = if high_big {
        (y_sorted[nh..].to_vec(), y_sorted[..nh].to_vec())
    } else { (y_sorted[..ng].to_vec(), y_sorted[ng..].to_vec()) };
    let mut q = vec![0f64; n];
    let mut go = g_idx.to_vec();
    go.sort_by(|&a, &b| ranks_guide[a].partial_cmp(&ranks_guide[b]).unwrap());
    let mut gv = g_vals; gv.sort_by(|a, b| a.partial_cmp(b).unwrap());
    for (k, &i) in go.iter().enumerate() { q[i] = gv[k]; }
    let mut ho = h;
    ho.sort_by(|&a, &b| ranks_guide[a].partial_cmp(&ranks_guide[b]).unwrap());
    let mut hv = h_vals; hv.sort_by(|a, b| a.partial_cmp(b).unwrap());
    for (k, &i) in ho.iter().enumerate() { q[i] = hv[k]; }
    q
}

struct Cand { ctype: u8, q: Vec<f64> }

fn quantile(x: &[f64], q: f64) -> f64 {
    let mut s = x.to_vec(); s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = s.len();
    let pos = (n - 1) as f64 * q;
    let lo = pos.floor() as usize; let hi = pos.ceil() as usize;
    if lo == hi { s[lo] } else { s[lo] + (pos - lo as f64) * (s[hi] - s[lo]) }
}

fn generate_candidates(p: &[f64], c_m: &[f64], y: &[f64], g2: &[f64], n: usize, m: usize, rng: &mut Rng, classes: &[f64]) -> Vec<Cand> {
    let e: Vec<f64> = (0..n).map(|i| (0..m).map(|j| j as f64 * p[i * m + j]).sum::<f64>()).collect();
    let ranks_guide = average_rank(&e);
    let mut ys_sorted = y.to_vec(); ys_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mut cands: Vec<Cand> = vec![];
    // 8 分歧：抽 16 个排列
    let mut perms: Vec<Vec<usize>> = vec![];
    for _ in 0..16 { perms.push(sample_permutation(p, c_m, n, m, rng)); }
    for k in 0..8 {
        let ya = &perms[2 * k];
        let q: Vec<f64> = ya.iter().map(|&idx| classes[idx]).collect();
        cands.push(Cand { ctype: 0, q });
    }
    // 8 结构
    let r2 = average_rank(g2);
    let qps = [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9];
    for (k, &qp) in qps.iter().enumerate() {
        let thr = quantile(&r2, qp);
        let g: Vec<usize> = if k % 2 == 0 {
            (0..n).filter(|&i| r2[i] >= thr).collect()
        } else { (0..n).filter(|&i| r2[i] < thr).collect() };
        let g = if g.is_empty() || g.len() == n {
            let mut s: Vec<usize> = (0..n).collect();
            s.sort_by(|&a, &b| r2[a].partial_cmp(&r2[b]).unwrap());
            s[n - n / 2..].to_vec()
        } else { g };
        let q = make_group_candidate(&ys_sorted, &g, &ranks_guide, k % 2 == 0);
        cands.push(Cand { ctype: 1, q });
    }
    // 8 高熵
    let h = row_entropy(p, n, m);
    let mut idx_h: Vec<usize> = (0..n).collect();
    idx_h.sort_by(|&a, &b| h[b].partial_cmp(&h[a]).unwrap());
    let top_half: Vec<usize> = idx_h[..n / 2].to_vec();
    for seed in 0..8 {
        let mut r = Rng::new(1000 + seed);
        let perm = r.permutation(top_half.len());
        let half = top_half.len() / 2;
        let g: Vec<usize> = (0..half).map(|k| top_half[perm[k]]).collect();
        let q = make_group_candidate(&ys_sorted, &g, &ranks_guide, true);
        cands.push(Cand { ctype: 2, q });
    }
    // 8 随机平衡
    for seed in 0..8 {
        let mut r = Rng::new(2000 + seed);
        let rand_score: Vec<f64> = (0..n).map(|_| r.random()).collect();
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| rand_score[a].partial_cmp(&rand_score[b]).unwrap());
        let mut q = vec![0f64; n];
        for (k, &i) in order.iter().enumerate() { q[i] = ys_sorted[k]; }
        cands.push(Cand { ctype: 3, q });
    }
    cands
}

fn select_query(cands: &[Cand], y: &[f64], p: &[f64], c_m: &[f64], n: usize, m: usize, classes: &[f64], rng: &mut Rng) -> (usize, Vec<f64>) {
    let mu = y.iter().sum::<f64>() / n as f64;
    let var = y.iter().map(|v| (v - mu).powi(2)).sum::<f64>() / n as f64;
    let denom = 2.0 * var.max(1e-12) * n as f64;
    let mut samples: Vec<Vec<f64>> = vec![];
    for _ in 0..32 {
        let perm = sample_permutation(p, c_m, n, m, rng);
        samples.push(perm.iter().map(|&idx| classes[idx]).collect());
    }
    let s_sq: Vec<f64> = samples.iter().map(|s| s.iter().map(|v| v * v).sum::<f64>()).collect();
    let best_pri = [0u8, 1, 2, 3];
    let mut best_idx = 0; let mut best_var = -1.0; let mut best_pri_val = 99u8;
    let mut var_list: Vec<f64> = vec![];
    let nsamp = samples.len();
    let mut rho_buf = [0f64; 40];
    for (idx, c) in cands.iter().enumerate() {
        let q_sq: f64 = c.q.iter().map(|v| v * v).sum();
        let mut sum_rho = 0f64;
        for t in 0..nsamp {
            let s = &samples[t];
            let dot: f64 = s.iter().zip(c.q.iter()).map(|(a, b)| a * b).sum();
            let rho = 1.0 - (q_sq + s_sq[t] - 2.0 * dot) / denom;
            rho_buf[t] = rho;
            sum_rho += rho;
        }
        let mean: f64 = sum_rho / nsamp as f64;
        let mut v = 0f64;
        for t in 0..nsamp { let d = rho_buf[t] - mean; v += d * d; }
        v /= nsamp as f64;
        var_list.push(v);
        let pri = best_pri[c.ctype as usize];
        if v > best_var + 1e-12 || (v - best_var).abs() <= 1e-12 && pri < best_pri_val {
            best_var = v; best_idx = idx; best_pri_val = pri;
        }
    }
    (best_idx, var_list)
}

fn update_posterior(p: &[f64], q: &[f64], r_bank: f64, c_m: &[f64], classes: &[f64], n: usize, m: usize) -> Vec<f64> {
    let s = r_bank.clamp(-1.0, 1.0);
    if s.abs() < 1e-9 { return p.to_vec(); }
    let q_class: Vec<usize> = q.iter().map(|&v| classes.binary_search_by(|c| c.partial_cmp(&v).unwrap()).unwrap()).collect();
    let lr_eff = UPDATE_LR * s.abs();
    let mut target = vec![0f64; n * m];
    if s > 0.0 {
        for i in 0..n {
            for j in 0..m { target[i * m + j] = (1.0 - lr_eff) * p[i * m + j]; }
            target[i * m + q_class[i]] += lr_eff;
        }
    } else {
        let dm = (m - 1).max(1) as f64;
        for i in 0..n {
            for j in 0..m {
                let v = if j == q_class[i] { 0.0 } else { 1.0 / dm };
                target[i * m + j] = (1.0 - lr_eff) * p[i * m + j] + lr_eff * v;
            }
        }
    }
    for v in target.iter_mut() { if *v < 1e-9 { *v = 1e-9; } }
    sinkhorn(&mut target, n, m, c_m, 30);
    target
}

fn posterior_stats(p: &[f64], y: &[f64], n: usize, m: usize, classes: &[f64], cnt: &[f64]) -> Vec<f64> {
    let h = row_entropy(p, n, m);
    let h_mean = h.iter().sum::<f64>() / n as f64;
    let top1: Vec<f64> = (0..n).map(|i| (0..m).map(|j| p[i * m + j]).fold(0f64, |a, b| a.max(b))).collect();
    let post_mean: Vec<f64> = (0..n).map(|i| (0..m).map(|j| j as f64 * p[i * m + j]).sum()).collect();
    let post_var: Vec<f64> = (0..n).map(|i| {
        let mean = post_mean[i];
        (0..m).map(|j| (j as f64).powi(2) * p[i * m + j]).sum::<f64>() - mean * mean
    }).collect();
    fn tail_set(m: usize, classes: &[f64], cnt: &[f64], n: usize, frac: f64) -> Vec<usize> {
        let mut order: Vec<usize> = (0..m).collect();
        order.sort_by(|&a, &b| classes[b].partial_cmp(&classes[a]).unwrap());
        let mut cum = 0.0; let mut s = Vec::new();
        for &ci in &order { s.push(ci); cum += cnt[ci]; if cum >= frac * n as f64 { break; } }
        s
    }
    fn tail_entropy(p: &[f64], n: usize, m: usize, s: &[usize]) -> f64 {
        let mut tot = 0.0;
        for i in 0..n {
            let pb = s.iter().map(|&j| p[i * m + j]).sum::<f64>().clamp(1e-9, 1.0 - 1e-9);
            tot += -(pb * pb.ln() + (1.0 - pb) * (1.0 - pb).ln());
        }
        tot / n as f64
    }
    fn tail_conf(p: &[f64], n: usize, m: usize, s: &[usize]) -> f64 {
        let mut tot = 0.0;
        for i in 0..n { tot += s.iter().map(|&j| p[i * m + j]).sum::<f64>(); }
        tot / n as f64
    }
    let s20 = tail_set(m, classes, cnt, n, 0.20);
    let s10 = tail_set(m, classes, cnt, n, 0.10);
    let s05 = tail_set(m, classes, cnt, n, 0.05);
    let s01 = tail_set(m, classes, cnt, n, 0.01);
    vec![
        h_mean,
        h.iter().map(|x| (x - h_mean).powi(2)).sum::<f64>() / n as f64,
        quantile(&h, 0.90),
        h.iter().cloned().fold(0f64, |a, b| a.max(b)),
        top1.iter().sum::<f64>() / n as f64,
        top1.iter().filter(|&&x| x < 0.5).count() as f64 / n as f64,
        post_var.iter().sum::<f64>() / n as f64,
        tail_entropy(p, n, m, &s20), tail_entropy(p, n, m, &s10),
        tail_entropy(p, n, m, &s05), tail_entropy(p, n, m, &s01),
        tail_conf(p, n, m, &s10), tail_conf(p, n, m, &s05),
    ]
}

pub struct TauResult {
    pub tau: f64,
    pub rho3: f64,
    pub rho2: f64,
    pub checkpoints: std::collections::HashMap<usize, Vec<f64>>,
    pub info_gains: Vec<f64>,
    pub query_types: Vec<u8>,
    pub pred_vars: Vec<f64>,
}

pub struct Stage3Result {
    pub taus: Vec<TauResult>,
}

impl Stage3Result {
    pub fn to_json(&self) -> String {
        let mut out = String::from("{");
        for (ti, t) in self.taus.iter().enumerate() {
            let h0 = t.checkpoints[&0][0];
            let hk = t.checkpoints.keys().copied().max().unwrap_or(0);
            let hv = t.checkpoints.get(&hk).map(|v| v[0]).unwrap_or(h0);
            let tn = format!("TAU{:03}", (t.tau * 100.0) as i64);
            let mut cnts = [0usize; 4];
            for &qt in &t.query_types { cnts[qt as usize] += 1; }
            if ti > 0 { out.push(','); }
            out.push_str(&format!("\"{}\":{{\"rho3\":{:.4},\"K0_entropy\":{:.6},\"K{}_entropy\":{:.6},\"total_info_gain\":{:.6},\"qt\":[{},{},{},{}]}}",
                tn, t.rho3, h0, hk, hv, h0 - hv, cnts[0], cnts[1], cnts[2], cnts[3]));
        }
        out.push('}');
        out
    }
}

pub fn run_stage3(g2: &[f64], y: &[f64], tau_list: &[f64], n_query: usize) -> Stage3Result {
    let n = y.len();
    let bank = Bank::new(y.to_vec());
    let rho2 = bank.query(g2);
    let mut taus = vec![];
    for &tau in tau_list.iter() {
        let tb = std::time::Instant::now();
        let (mut p, c_m, m, classes) = build_prior(g2, y, tau);
        let cnt: Vec<f64> = (0..m).map(|j| y.iter().filter(|&&v| v == classes[j]).count() as f64).collect();
        let mut rng = Rng::new(42);
        let mut ck = std::collections::HashMap::new();
        ck.insert(0, posterior_stats(&p, y, n, m, &classes, &cnt));
        let mut h_prev = row_entropy(&p, n, m).iter().sum::<f64>() / n as f64;
        let mut query_types: Vec<u8> = vec![];
        let mut info_gains: Vec<f64> = vec![];
        let mut pred_vars: Vec<f64> = vec![];
        let tq = std::time::Instant::now();
        let mut t_gen = 0f64; let mut t_sel = 0f64; let mut t_upd = 0f64;
        for k in 1..=n_query {
            let tg0 = std::time::Instant::now();
            let cands = generate_candidates(&p, &c_m, y, g2, n, m, &mut rng, &classes);
            t_gen += tg0.elapsed().as_secs_f64();
            let ts0 = std::time::Instant::now();
            let (idx, var_list) = select_query(&cands, y, &p, &c_m, n, m, &classes, &mut rng);
            t_sel += ts0.elapsed().as_secs_f64();
            let q = cands[idx].q.clone();
            let r_bank = bank.query(&q);
            let tu0 = std::time::Instant::now();
            p = update_posterior(&p, &q, r_bank, &c_m, &classes, n, m);
            t_upd += tu0.elapsed().as_secs_f64();
            let h_cur = row_entropy(&p, n, m).iter().sum::<f64>() / n as f64;
            info_gains.push(h_prev - h_cur);
            pred_vars.push(var_list[idx]);
            query_types.push(cands[idx].ctype);
            h_prev = h_cur;
            if k <= n_query && [0usize, 8, 16, 32, 64].contains(&k) {
                ck.insert(k, posterior_stats(&p, y, n, m, &classes, &cnt));
            }
        }
        let e: Vec<f64> = (0..n).map(|i| (0..m).map(|j| j as f64 * p[i * m + j]).sum()).collect();
        let ranks = average_rank(&e);
        let g3 = assign_by_rank(&ranks, y);
        let rho3 = bank.query(&g3);
        taus.push(TauResult { tau, rho3, rho2, checkpoints: ck, info_gains, query_types, pred_vars });
    }
    Stage3Result { taus }
}

// 因子提取：从三层游戏结果提取基础因子（移植 Python factors.py）





const S1_SUFFIX: [&str; 11] = ["FINAL_CORR", "NET_GAIN", "NORM_GAIN", "AUC", "QUERY_EFF",
    "ACCEPT_RATIO", "ABS_CONTRAST_MEAN", "CONTRAST_ENERGY", "SCALE_CENTROID", "SCALE_ENTROPY", "PARENT_CHILD_SIGN"];
const S2_SUFFIX: [&str; 8] = ["FINAL_CORR", "NET_GAIN", "NORM_GAIN", "AUC", "QUERY_EFF",
    "ACCEPT_RATIO", "CONTRAST_ENERGY", "PATH_DISTANCE"];
const S3_STATS: [(&str, usize); 12] = [
    ("ENTROPY_MEAN", 0), ("ENTROPY_STD", 1), ("ENTROPY_P90", 2), ("ENTROPY_MAX", 3),
    ("TOP1_CONF_MEAN", 4), ("LOWCONF_RATIO50", 5), ("POSTVAR_MEAN", 6),
    ("TAIL20_ENTROPY", 7), ("TAIL10_ENTROPY", 8), ("TAIL05_ENTROPY", 9),
    ("TAIL01_ENTROPY", 10), ("TAIL10_CONF", 11)];
const FAMILIES_GRAPHS: [(&str, &[&str]); 8] = [
    ("TIME", &["TIME_R1", "TIME_R8", "TIME_R32"]),
    ("ACTIVE", &["ACTIVE_EXACT"]), ("PASSIVE", &["PASSIVE_EXACT"]),
    ("PRICE", &["PRICE_T0", "PRICE_T2"]), ("SIDE", &["SIDE_EXACT"]),
    ("BOOK", &["BOOK_K10", "BOOK_K30"]), ("QUEUE", &["QUEUE_K10"]), ("ARRIVAL", &["ARRIVAL_K10"]),
];

fn sd(a: f64, b: f64) -> f64 { if b.abs() > 1e-12 { a / b } else { 0.0 } }
fn gini(x: &[f64]) -> f64 {
    let mut s: Vec<f64> = x.iter().cloned().collect();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = s.len(); if n == 0 { return 0.0; }
    let sum: f64 = s.iter().sum();
    if sum.abs() < 1e-12 { return 0.0; }
    (2.0 * (1..=n).map(|i| i as f64 * s[i - 1]).sum::<f64>()) / (n as f64 * sum) - (n as f64 + 1.0) / n as f64
}
fn entropy_norm(probs: &[f64], l: usize) -> f64 {
    let p: Vec<f64> = probs.iter().filter(|&&x| x > 1e-12).cloned().collect();
    if p.is_empty() || l <= 1 { return 0.0; }
    -p.iter().map(|x| x * x.ln()).sum::<f64>() / (l as f64).ln()
}
fn auc(traj: &[f64], rho0: f64) -> f64 {
    if traj.len() <= 1 { return 0.0; }
    traj[1..].iter().map(|r| (r - rho0) / (1.0 - rho0 + 1e-12)).sum::<f64>() / (traj.len() - 1) as f64
}
pub type Factors = Vec<(String, f64)>;
fn push(out: &mut Factors, name: &str, v: f64) { out.push((name.to_string(), v)); }

pub fn target_dist(y: &[f64]) -> Factors {
    let n = y.len();
    let mu = y.iter().sum::<f64>() / n as f64;
    let sdv = (y.iter().map(|v| (v - mu).powi(2)).sum::<f64>() / n as f64).sqrt();
    let med = percentile(y, 50.0);
    let mad_v: Vec<f64> = y.iter().map(|v| (v - med).abs()).collect();
    let mad = percentile(&mad_v, 50.0);
    let z: Vec<f64> = if sdv > 1e-12 { y.iter().map(|v| (v - mu) / sdv).collect() } else { vec![0f64; n] };
    let skew = if sdv > 1e-12 { z.iter().map(|v| v.powi(3)).sum::<f64>() / n as f64 } else { 0.0 };
    let kurt = if sdv > 1e-12 { z.iter().map(|v| v.powi(4)).sum::<f64>() / n as f64 - 3.0 } else { 0.0 };
    let total: f64 = y.iter().sum();
    let mut sy = y.to_vec(); sy.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let top_share = |frac: f64| { let k = ((frac * n as f64).ceil() as usize).max(1); sy[n - k..].iter().sum::<f64>() / (total.abs() + 1e-12) };
    let mut uniq = y.to_vec(); uniq.sort_by(|a, b| a.partial_cmp(b).unwrap()); uniq.dedup();
    let mut cnt: HashMap<i64, usize> = HashMap::new();
    for v in y.iter().map(|v| (v * 1e9) as i64) { *cnt.entry(v).or_default() += 1; }
    let shares: Vec<f64> = cnt.values().map(|&c| c as f64 / n as f64).collect();
    let mut f = vec![]; let mut p = |n: &str, v: f64| push(&mut f, n, v);
    p("TARGET_MEAN", mu); p("TARGET_STD", sdv); p("TARGET_MEDIAN", med); p("TARGET_MAD", mad);
    p("TARGET_SKEW", skew); p("TARGET_KURT", kurt);
    p("TARGET_MIN", y.iter().cloned().fold(f64::INFINITY, f64::min));
    p("TARGET_MAX", y.iter().cloned().fold(f64::NEG_INFINITY, f64::max));
    p("TARGET_Q75", percentile(y, 75.0)); p("TARGET_Q90", percentile(y, 90.0));
    p("TARGET_Q95", percentile(y, 95.0)); p("TARGET_Q99", percentile(y, 99.0));
    p("TARGET_UNIQUE_COUNT", uniq.len() as f64); p("TARGET_UNIQUE_RATIO", uniq.len() as f64 / n as f64);
    p("TARGET_VALUE_HHI", shares.iter().map(|s| s * s).sum());
    p("TARGET_GINI", gini(y));
    p("TARGET_TOP1_SHARE", top_share(0.01)); p("TARGET_TOP5_SHARE", top_share(0.05));
    p("TARGET_TOP10_SHARE", top_share(0.10));
    p("TARGET_MAX_SHARE", y.iter().cloned().fold(f64::NEG_INFINITY, f64::max) / (total.abs() + 1e-12));
    f
}

fn scale_features(level_ds: &[Vec<f64>]) -> (f64, f64) {
    if level_ds.is_empty() { return (0.0, 0.0); }
    let es: Vec<f64> = level_ds.iter().map(|lv| lv.iter().map(|d| d * d).sum::<f64>()).collect();
    let tot: f64 = es.iter().sum();
    if tot < 1e-12 { return (0.0, 0.0); }
    let pv: Vec<f64> = es.iter().map(|e| e / tot).collect();
    let l = es.len();
    let cent = (1..=l).zip(pv.iter()).map(|(i, pp)| i as f64 * pp).sum::<f64>();
    (cent, entropy_norm(&pv, l))
}

pub fn s1_branches(s1: &Stage1Result) -> Factors {
    let mut f = vec![];
    for b in &s1.branches {
        let name = format!("S1_{}_{}", b.axis, b.depth);
        let net = b.rho_final - b.rho0;
        let (cent, sent) = scale_features(&b.level_ds);
        let vals = [
            b.rho_final, net, sd(net, 1.0 - b.rho0), auc(&b.trajectory, b.rho0),
            sd(net, b.n_queries as f64),
            if b.n_internal > 0 { b.accepts.iter().filter(|&&x| x).count() as f64 / b.n_internal as f64 } else { 0.0 },
            if b.node_ds.is_empty() { 0.0 } else { b.node_ds.iter().map(|d| d.abs()).sum::<f64>() / b.node_ds.len() as f64 },
            if b.node_ds.is_empty() { 0.0 } else { b.node_ds.iter().map(|d| d * d).sum::<f64>() / b.node_ds.len() as f64 },
            cent, sent,
            if b.parent_child_sign.is_empty() { 0.0 } else { b.parent_child_sign.iter().filter(|(a, c)| a == c).count() as f64 / b.parent_child_sign.len() as f64 },
        ];
        for (suf, v) in S1_SUFFIX.iter().zip(vals.iter()) { push(&mut f, &format!("{}_{}", name, suf), *v); }
    }
    f
}

pub fn s1_summary(s1: &Stage1Result) -> Factors {
    let mut f = vec![]; let mut p = |n: &str, v: f64| push(&mut f, n, v);
    let rho0 = s1.rho0;
    let gains: Vec<f64> = s1.branches.iter().map(|b| b.rho_final - b.rho0).collect();
    p("S1_FINAL_CORR", s1.rho); p("S1_NET_GAIN", s1.rho - rho0); p("S1_NORM_GAIN", sd(s1.rho - rho0, 1.0 - rho0));
    let mut rhos: Vec<f64> = vec![rho0];
    rhos.extend(s1.branches.iter().map(|b| b.rho_final));
    rhos.extend(s1.votes.iter().map(|v| v.1));
    rhos.sort_by(|a, b| b.partial_cmp(a).unwrap());
    p("S1_BEST_MARGIN", if rhos.len() > 1 { rhos[0] - rhos[1] } else { 0.0 });
    let gm = gains.iter().sum::<f64>() / gains.len() as f64;
    p("S1_BRANCH_GAIN_MEAN", gm);
    p("S1_BRANCH_GAIN_STD", (gains.iter().map(|g| (g - gm).powi(2)).sum::<f64>() / gains.len() as f64).sqrt());
    p("S1_BRANCH_GAIN_MAX", gains.iter().cloned().fold(f64::NEG_INFINITY, f64::max));
    p("S1_BRANCH_GAIN_MIN", gains.iter().cloned().fold(f64::INFINITY, f64::min));
    p("S1_BRANCH_POS_RATIO", gains.iter().filter(|&&g| g > 0.0).count() as f64 / gains.len() as f64);
    let pos: Vec<f64> = gains.iter().map(|g| g.max(0.0)).collect();
    let pstot: f64 = pos.iter().sum();
    p("S1_BRANCH_GAIN_ENTROPY", if pstot > 1e-12 { entropy_norm(&pos.iter().map(|x| x / pstot).collect::<Vec<_>>(), 16) } else { 0.0 });
    p("S1_BRANCH_GAIN_HHI", if pstot > 1e-12 { pos.iter().map(|x| { let r = x / pstot; r * r }).sum() } else { 0.0 });
    let fam_gains = |fam: &str| -> Vec<f64> {
        s1.branches.iter().filter(|b| {
            let af = match b.axis.as_str() { "TIME" => "TIME", "ACTIVE_ORDER" => "ACTIVE", "PASSIVE_ORDER" => "PASSIVE", _ => "PRICE" };
            af == fam
        }).map(|b| b.rho_final - b.rho0).collect()
    };
    let fam_max = |fam: &str| fam_gains(fam).iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    for fam in ["TIME", "ACTIVE", "PASSIVE", "PRICE"] {
        p(&format!("S1_{}_FAMILY_MAX", fam), fam_max(fam));
        let g = fam_gains(fam); p(&format!("S1_{}_FAMILY_MEAN", fam), g.iter().sum::<f64>() / g.len() as f64);
    }
    p("S1_ACTIVE_MINUS_TIME", fam_max("ACTIVE") - fam_max("TIME"));
    p("S1_PASSIVE_MINUS_TIME", fam_max("PASSIVE") - fam_max("TIME"));
    p("S1_PRICE_MINUS_TIME", fam_max("PRICE") - fam_max("TIME"));
    let coarse: f64 = s1.branches.iter().filter(|b| b.depth == "N4" || b.depth == "N8").map(|b| (b.rho_final - b.rho0).max(0.0)).sum();
    let fine: f64 = s1.branches.iter().filter(|b| b.depth == "N32" || b.depth == "N16").map(|b| (b.rho_final - b.rho0).max(0.0)).sum();
    let tot = coarse + fine + 1e-12;
    p("S1_COARSE_SHARE", coarse / tot); p("S1_FINE_SHARE", fine / tot); p("S1_COARSE_MINUS_FINE", (coarse - fine) / tot);
    for (i, beta) in [1, 2, 5, 10].iter().enumerate() {
        p(&format!("S1_VOTE_B{}_CORR", beta), s1.votes[i].1);
        p(&format!("S1_VOTE_B{}_GAIN", beta), s1.votes[i].1 - rho0);
    }
    let pr = &s1.path_rhos;
    let pmean = pr.iter().sum::<f64>() / pr.len() as f64;
    p("S1_INIT_PATH_STD", (pr.iter().map(|x| (x - pmean).powi(2)).sum::<f64>() / pr.len() as f64).sqrt());
    p("S1_INIT_PATH_RANGE", pr.iter().cloned().fold(f64::NEG_INFINITY, f64::max) - pr.iter().cloned().fold(f64::INFINITY, f64::min));
    p("S1_TIME_ASC_MINUS_RANDOM", pr[0] - pr[3]);
    p("S1_TIME_DESC_MINUS_ASC", pr[1] - pr[0]);
    p("S1_ACTIVE_INIT_MINUS_TIME_INIT", pr[2] - pr[0]);
    f
}

pub fn s2_all(s2: &Stage2Result, rho0: f64) -> Factors {
    let mut f = vec![]; let mut p = |n: &str, v: f64| push(&mut f, n, v);
    let rho1 = s2.rho1;
    for b in &s2.branches {
        let name = format!("S2_{}_{}", b.graph, b.scope);
        let net = b.rho_final - rho1;
        let vals = [
            b.rho_final, net, sd(net, 1.0 - rho1), auc(&b.trajectory, rho1),
            sd(net, b.n_queries as f64),
            if b.accepts.is_empty() { 0.0 } else { b.accepts.iter().filter(|&&x| x).count() as f64 / b.accepts.len() as f64 },
            if b.node_ds.is_empty() { 0.0 } else { b.node_ds.iter().map(|d| d * d).sum::<f64>() / b.node_ds.len() as f64 },
            b.path_distance,
        ];
        for (suf, v) in S2_SUFFIX.iter().zip(vals.iter()) { let nm = format!("{}_{}", name, suf); p(&nm, *v); }
    }
    let find = |g: &str, sc: &str| s2.branches.iter().find(|b| b.graph == g && b.scope == sc);
    for g in GRAPH_NAMES.iter() {
        let gains: Vec<f64> = SCOPES.iter().map(|sc| find(g, sc).map(|b| b.rho_final - rho1).unwrap_or(0.0)).collect();
        p(&format!("S2_{}_GAIN_MAX_MODE", g), gains.iter().cloned().fold(f64::NEG_INFINITY, f64::max));
        let gm = gains.iter().sum::<f64>() / 3.0;
        p(&format!("S2_{}_GAIN_MEAN_MODE", g), gm);
        p(&format!("S2_{}_GAIN_STD_MODE", g), (gains.iter().map(|x| (x - gm).powi(2)).sum::<f64>() / 3.0).sqrt());
        p(&format!("S2_{}_LOCAL_GAIN", g), gains[0]); p(&format!("S2_{}_GLOBAL_GAIN", g), gains[2]);
        p(&format!("S2_{}_GLOBAL_MINUS_LOCAL", g), gains[2] - gains[0]);
    }
    let mut fam_gains: HashMap<&str, Vec<f64>> = HashMap::new();
    for (fam, graphs) in FAMILIES_GRAPHS.iter() {
        let mut gs = vec![];
        for g in graphs.iter() { for sc in SCOPES.iter() { if let Some(b) = find(g, sc) { gs.push(b.rho_final - rho1); } } }
        fam_gains.insert(fam, gs);
    }
    let all_pos: f64 = fam_gains.values().flat_map(|gs| gs.iter().map(|g| g.max(0.0))).sum::<f64>() + 1e-12;
    for fam in ["TIME", "ACTIVE", "PASSIVE", "PRICE", "SIDE", "BOOK", "QUEUE", "ARRIVAL"] {
        let gs = &fam_gains[fam];
        p(&format!("S2_{}_FAMILY_GAIN_MAX", fam), gs.iter().cloned().fold(f64::NEG_INFINITY, f64::max));
        p(&format!("S2_{}_FAMILY_GAIN_MEAN", fam), gs.iter().sum::<f64>() / gs.len() as f64);
        p(&format!("S2_{}_FAMILY_SHARE", fam), gs.iter().map(|g| g.max(0.0)).sum::<f64>() / all_pos);
        let gm = gs.iter().sum::<f64>() / gs.len() as f64;
        p(&format!("S2_{}_FAMILY_PARAM_STD", fam), if gs.len() > 1 { (gs.iter().map(|g| (g - gm).powi(2)).sum::<f64>() / gs.len() as f64).sqrt() } else { 0.0 });
    }
    let fmax2 = |fam: &str| -> f64 {
        let graphs = FAMILIES_GRAPHS.iter().find(|(fn_, _)| *fn_ == fam).unwrap().1;
        s2.branches.iter().filter(|b| graphs.contains(&b.graph.as_str())).map(|b| b.rho_final - rho1).fold(f64::NEG_INFINITY, f64::max)
    };
    for (a, b) in [("ACTIVE","TIME"),("PASSIVE","TIME"),("PRICE","TIME"),("SIDE","TIME"),("BOOK","TIME"),("QUEUE","TIME"),("ARRIVAL","TIME"),("ACTIVE","PASSIVE"),("BOOK","PRICE"),("QUEUE","BOOK"),("ARRIVAL","BOOK"),("SIDE","BOOK")] {
        p(&format!("S2_{}_MINUS_{}", a, b), fmax2(a) - fmax2(b));
    }
    p("S2_ACTIVE_TO_TIME_RATIO", sd(fmax2("ACTIVE"), fmax2("TIME").abs()));
    p("S2_PASSIVE_TO_TIME_RATIO", sd(fmax2("PASSIVE"), fmax2("TIME").abs()));
    p("S2_BOOK_TO_PRICE_RATIO", sd(fmax2("BOOK"), fmax2("PRICE").abs()));
    p("S2_QUEUE_TO_BOOK_RATIO", sd(fmax2("QUEUE"), fmax2("BOOK").abs()));
    p("S2_FINAL_CORR", s2.rho); p("S2_INCREMENTAL_GAIN", s2.rho - rho1); p("S2_TOTAL_GAIN", s2.rho - rho0);
    p("S2_NORM_INCREMENTAL_GAIN", sd(s2.rho - rho1, 1.0 - rho1));
    let mut all_rhos: Vec<f64> = vec![rho1, s2.rho];
    all_rhos.extend(s2.votes.iter().map(|v| v.1));
    all_rhos.sort_by(|a, b| b.partial_cmp(a).unwrap());
    p("S2_BEST_MARGIN", if all_rhos.len() > 1 { all_rhos[0] - all_rhos[1] } else { 0.0 });
    let gg: Vec<f64> = s2.branches.iter().map(|b| b.rho_final - rho1).collect();
    let ggm = gg.iter().sum::<f64>() / gg.len() as f64;
    p("S2_GRAPH_GAIN_MEAN", ggm);
    p("S2_GRAPH_GAIN_STD", (gg.iter().map(|g| (g - ggm).powi(2)).sum::<f64>() / gg.len() as f64).sqrt());
    p("S2_GRAPH_GAIN_MAX", gg.iter().cloned().fold(f64::NEG_INFINITY, f64::max));
    p("S2_GRAPH_POS_RATIO", gg.iter().filter(|&&g| g > 0.0).count() as f64 / gg.len() as f64);
    let shares: Vec<f64> = ["TIME", "ACTIVE", "PASSIVE", "PRICE", "SIDE", "BOOK", "QUEUE", "ARRIVAL"].iter().map(|fam| fmax2(fam).max(0.0)).collect();
    let stot: f64 = shares.iter().sum::<f64>() + 1e-12;
    let sp: Vec<f64> = shares.iter().map(|s| s / stot).collect();
    p("S2_GRAPH_MECHANISM_ENTROPY", entropy_norm(&sp, 8));
    p("S2_GRAPH_MECHANISM_HHI", sp.iter().map(|s| s * s).sum());
    let mut so = (0..8).collect::<Vec<usize>>(); so.sort_by(|&a, &b| shares[b].partial_cmp(&shares[a]).unwrap());
    p("S2_WINNER_MARGIN", (shares[so[0]] - shares[so[1]]) / stot);
    p("S2_TOP2_FAMILY_SHARE", (shares[so[0]] + shares[so[1]]) / stot);
    let lp: f64 = s2.branches.iter().filter(|b| b.scope == "LOCAL32").map(|b| (b.rho_final - rho1).max(0.0)).sum();
    let mp: f64 = s2.branches.iter().filter(|b| b.scope == "MESO8_D2").map(|b| (b.rho_final - rho1).max(0.0)).sum();
    let gp: f64 = s2.branches.iter().filter(|b| b.scope == "GLOBAL_D4").map(|b| (b.rho_final - rho1).max(0.0)).sum();
    let stt = lp + mp + gp + 1e-12;
    p("S2_LOCAL_SHARE", lp / stt); p("S2_MESO_SHARE", mp / stt); p("S2_GLOBAL_SHARE", gp / stt);
    p("S2_GLOBAL_MINUS_LOCAL", (gp / 15.0) - (lp / 32.0));
    for (i, beta) in [1, 2, 5, 10].iter().enumerate() {
        p(&format!("S2_VOTE_B{}_CORR", beta), s2.votes[i].1);
        p(&format!("S2_VOTE_B{}_GAIN", beta), s2.votes[i].1 - rho1);
    }
    f
}

pub fn s3_contrib_cross(s1: &Stage1Result, s2: &Stage2Result, s3: Option<&Stage3Result>) -> Factors {
    let mut f = vec![]; let mut p = |n: &str, v: f64| push(&mut f, n, v);
    if let Some(s3) = s3 {
        for t in s3.taus.iter() {
            let tn = format!("TAU{:03}", (t.tau * 100.0) as i64);
            for &ck in &[0usize, 8, 16, 32, 64] {
                if let Some(stats) = t.checkpoints.get(&ck) {
                    for (stat, idx) in S3_STATS.iter() { p(&format!("S3_{}_K{}_{}", tn, ck, stat), stats[*idx]); }
                }
            }
            let h = |k: usize| t.checkpoints.get(&k).map(|v| v[0]).unwrap_or(0.0);
            let h0 = h(0); let h8 = h(8); let h16 = h(16); let h32 = h(32); let h64 = h(64);
            p(&format!("S3_{}_TOTAL_INFO_GAIN", tn), h0 - h64);
            p(&format!("S3_{}_INFO_GAIN_0_8", tn), h0 - h8);
            p(&format!("S3_{}_INFO_GAIN_8_16", tn), h8 - h16);
            p(&format!("S3_{}_INFO_GAIN_16_32", tn), h16 - h32);
            p(&format!("S3_{}_INFO_GAIN_32_64", tn), h32 - h64);
            p(&format!("S3_{}_EARLY_INFO_SHARE8", tn), sd(h0 - h8, h0 - h64));
            p(&format!("S3_{}_EARLY_INFO_SHARE16", tn), sd(h0 - h16, h0 - h64));
            let ig = &t.info_gains;
            p(&format!("S3_{}_QUERY_INFO_GAIN_MAX", tn), ig.iter().cloned().fold(f64::NEG_INFINITY, f64::max));
            let igm = ig.iter().sum::<f64>() / ig.len() as f64;
            p(&format!("S3_{}_QUERY_INFO_GAIN_MEAN", tn), igm);
            p(&format!("S3_{}_QUERY_INFO_GAIN_STD", tn), (ig.iter().map(|x| (x - igm).powi(2)).sum::<f64>() / ig.len() as f64).sqrt());
            p(&format!("S3_{}_QUERY_INFO_GAIN_GINI", tn), gini(ig));
            p(&format!("S3_{}_PREDICTED_VARIANCE_MEAN", tn), t.pred_vars.iter().sum::<f64>() / t.pred_vars.len() as f64);
            p(&format!("S3_{}_FINAL_POSTERIOR_CORR", tn), t.rho3);
            p(&format!("S3_{}_FINAL_GAIN_VS_STAGE2", tn), t.rho3 - t.rho2);
            for (tp_i, tp) in ["DISAGREEMENT", "STRUCTURE", "ENTROPY", "RANDOM"].iter().enumerate() {
                let cnt = t.query_types.iter().filter(|&&q| q as usize == tp_i).count();
                p(&format!("S3_{}_{}_SHARE", tn, tp), cnt as f64 / t.query_types.len() as f64);
                let g: Vec<f64> = t.query_types.iter().enumerate().filter(|(_, &q)| q as usize == tp_i).map(|(i, _)| ig[i]).collect();
                p(&format!("S3_{}_{}_GAIN_MEAN", tn, tp), if g.is_empty() { 0.0 } else { g.iter().sum::<f64>() / g.len() as f64 });
            }
        }
    }
    let rho0 = s1.rho0; let rho1 = s1.rho; let rho2 = s2.rho;
    let g1 = rho1 - rho0; let g2 = rho2 - rho1;
    p("X_S1_GAIN", g1); p("X_S2_GAIN", g2);
    if let Some(s3) = s3 {
        for t in s3.taus.iter() {
            let tn = format!("TAU{:03}", (t.tau * 100.0) as i64);
            let g3 = t.rho3 - rho2;
            p(&format!("X_S3_GAIN_{}", tn), g3);
            p(&format!("X_S2_TO_S3_RATIO_{}", tn), sd(g2, g3.abs()));
        }
    }
    p("X_S1_SHARE_12", sd(g1, g1 + g2)); p("X_S2_SHARE_12", sd(g2, g1 + g2));
    p("X_S1_TO_S2_RATIO", sd(g1, g2.abs()));
    // ========== 交叉因子(25) ==========
    let s1gains: Vec<f64> = s1.branches.iter().map(|b| b.rho_final - b.rho0).collect();
    let s1pos: Vec<f64> = s1gains.iter().map(|g| g.max(0.0)).collect();
    let s1pstot: f64 = s1pos.iter().sum::<f64>() + 1e-12;
    let s1_bge = entropy_norm(&s1pos.iter().map(|x| x / s1pstot).collect::<Vec<_>>(), 16);
    let coarse: f64 = s1.branches.iter().filter(|b| b.depth == "N4" || b.depth == "N8").map(|b| (b.rho_final - b.rho0).max(0.0)).sum();
    let fine: f64 = s1.branches.iter().filter(|b| b.depth == "N32" || b.depth == "N16").map(|b| (b.rho_final - b.rho0).max(0.0)).sum();
    let ctot = coarse + fine + 1e-12;
    let s1_coarse = coarse / ctot; let s1_fine = fine / ctot;
    let rho1 = s2.rho1;
    let fam_gmax = |fam: &str| -> f64 {
        let graphs = FAMILIES_GRAPHS.iter().find(|(fn_, _)| *fn_ == fam).unwrap().1;
        s2.branches.iter().filter(|b| graphs.contains(&b.graph.as_str())).map(|b| b.rho_final - rho1).fold(f64::NEG_INFINITY, f64::max)
    };
    let fam_share = |fam: &str| fam_gmax(fam).max(0.0) / (fam_gmax(fam).max(0.0) + 1e-12);
    let shares8: Vec<f64> = ["TIME","ACTIVE","PASSIVE","PRICE","SIDE","BOOK","QUEUE","ARRIVAL"].iter().map(|f| fam_gmax(f).max(0.0)).collect();
    let stot8: f64 = shares8.iter().sum::<f64>() + 1e-12;
    let sp8: Vec<f64> = shares8.iter().map(|s| s / stot8).collect();
    let s2_me = entropy_norm(&sp8, 8);
    let fs = |fam: &str| fam_gmax(fam).max(0.0) / stot8;
    let active_minus_time = fam_gmax("ACTIVE") - fam_gmax("TIME");
    let book_minus_time = fam_gmax("BOOK") - fam_gmax("TIME");
    p("X_SCALE_ENTROPY_MINUS_GRAPH_ENTROPY", s1_bge - s2_me);
    p("X_COARSE_SHARE_X_ACTIVE_SHARE", s1_coarse * fs("ACTIVE"));
    p("X_COARSE_SHARE_X_BOOK_SHARE", s1_coarse * fs("BOOK"));
    p("X_FINE_SHARE_X_QUEUE_SHARE", s1_fine * fs("QUEUE"));
    p("X_FINE_SHARE_X_ARRIVAL_SHARE", s1_fine * fs("ARRIVAL"));
    p("X_ORDER_IDENTITY_DOMINANCE", (fam_gmax("ACTIVE") + fam_gmax("PASSIVE")) / 2.0 - fam_gmax("TIME"));
    p("X_BOOK_QUEUE_DOMINANCE", (fam_gmax("BOOK") + fam_gmax("QUEUE")) / 2.0 - fam_gmax("PRICE"));
    if let Some(s3) = s3 {
        for t in s3.taus.iter() {
            let tn = format!("TAU{:03}", (t.tau * 100.0) as i64);
            let last_k = t.checkpoints.keys().copied().max().unwrap_or(0);
            let ckl = t.checkpoints.get(&last_k).unwrap();
            let total_info = t.checkpoints[&0][0] - ckl[0];
            p(&format!("X_ACTIVE_GAIN_X_TAIL10_ENTROPY_{}", tn), fam_gmax("ACTIVE") * ckl[8]);
            p(&format!("X_BOOK_GAIN_X_TAIL10_CONF_{}", tn), fam_gmax("BOOK") * ckl[11]);
            p(&format!("X_GRAPH_ENTROPY_X_TOTAL_INFO_{}", tn), s2_me * total_info);
            p(&format!("X_SCALE_ENTROPY_X_FINAL_ENTROPY_{}", tn), s1_bge * ckl[0]);
            p(&format!("X_HIDDEN_LARGE_ORDER_{}", tn), active_minus_time * ckl[9]);
            p(&format!("X_VISIBLE_LARGE_LIQUIDITY_{}", tn), book_minus_time * ckl[12]);
        }
    }
    f
}

/// 汇总所有基础因子
pub fn extract_all(s1: &Stage1Result, s2: &Stage2Result, s3: Option<&Stage3Result>, y: &[f64]) -> Factors {
    let mut f = vec![];
    f.extend(target_dist(y));
    f.extend(s1_branches(s1));
    f.extend(s1_summary(s1));
    f.extend(s2_all(s2, s1.rho0));
    f.extend(s3_contrib_cross(s1, s2, s3));
    f
}

#[pyfunction]
pub fn py_hidden_arrange(py: Python<'_>, code: &str, date: i64) -> PyResult<Vec<f32>> {
    let (_n, v) = compute_hidden_arrange_full(code, date)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("{:?}", e)))?;
    Ok(v)
}
#[pyfunction]
pub fn py_hidden_arrange_names() -> Vec<String> {
    let t = template_names();
    if t.is_empty() { hidden_arrange_names() } else { t }
}
