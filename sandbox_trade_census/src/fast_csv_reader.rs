//! 逐笔成交读取（探索版）：只解析需要的列，不做下午平移，保留全部时间由调用方自行分桶。
//! 丢弃 flag=32 的集合竞价申报行（这些行 turnover 为 0，不是成交）。
use std::fs::File;
use std::io::Read;
use std::path::Path;

const COL_EXCHTIME: usize = 4;
const COL_PRICE: usize = 7;
const COL_VOLUME: usize = 8;
const COL_TURNOVER: usize = 9;
const COL_FLAG: usize = 10;
const OFFSET_US: i64 = 8 * 3600 * 1_000_000;

#[derive(Clone, Copy, Debug, Default)]
pub struct TradeRecord {
    pub day_sec: i64,
    pub price: f64,
    pub volume: f64,
    pub turnover: f64,
    pub flag: i32,
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
    Err(std::io::Error::new(std::io::ErrorKind::NotFound, format!("not found: {}/{}/{}", date, subdir, filename)))
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

/// 一行 15 列，只取时间/价/量/额/标志。flag=32 是集合竞价申报，丢弃。
#[inline]
fn parse_line(line: &[u8]) -> Option<TradeRecord> {
    if line.is_empty() { return None; }
    let mut fields: [&[u8]; 15] = [&[][..]; 15];
    let mut start = 0; let mut col = 0;
    for (i, &b) in line.iter().enumerate() {
        if b == b',' { if col < 15 { fields[col] = &line[start..i]; } col += 1; start = i + 1; }
    }
    if col < 15 { fields[col] = &line[start..]; }
    let flag_bytes = fields[COL_FLAG];
    if flag_bytes == b"32" { return None; }
    let exchtime_us = parse_i64_fast(fields[COL_EXCHTIME]);
    if exchtime_us == 0 { return None; }
    let day_sec = ((exchtime_us + OFFSET_US) / 1_000_000).rem_euclid(86400);
    Some(TradeRecord {
        day_sec,
        price: parse_f64_fast(fields[COL_PRICE]),
        volume: parse_f64_fast(fields[COL_VOLUME]),
        turnover: parse_f64_fast(fields[COL_TURNOVER]),
        flag: if flag_bytes.is_empty() { 0 } else { parse_i64_fast(flag_bytes) as i32 },
    })
}

fn parse_chunk(data: &[u8]) -> Vec<TradeRecord> {
    let mut out = Vec::with_capacity(data.len() / 80 + 1);
    let mut start = 0;
    let mut first = true;
    for i in 0..data.len() {
        if data[i] == b'\n' {
            let mut line = &data[start..i];
            if line.last() == Some(&b'\r') { line = &line[..line.len()-1]; }
            if first { first = false; }
            else if let Some(r) = parse_line(line) { out.push(r); }
            start = i + 1;
        }
    }
    if start < data.len() {
        let mut line = &data[start..];
        if line.last() == Some(&b'\r') { line = &line[..line.len()-1]; }
        if let Some(r) = parse_line(line) { out.push(r); }
    }
    out
}

pub fn read_trade_raw(code: &str, date: i64) -> std::io::Result<Vec<TradeRecord>> {
    let filename = format!("{}_{}_transaction.csv", code, date);
    let path = resolve_stock_path(date, "transaction", &filename)?;
    let mut content = String::new();
    File::open(&path)?.read_to_string(&mut content)?;
    Ok(parse_chunk(content.as_bytes()))
}
