//! 精简版数据读取（单线程，无 pyo3/rayon），从 fast_csv_reader.rs 移植。
//! 读取 transaction / market_data CSV，做 adjust_afternoon 平移。
use std::fs::File;
use std::io::Read;
use std::path::Path;

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
