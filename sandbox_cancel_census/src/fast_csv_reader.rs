//! 逐笔读取（撤单普查版）：保留撤单行（flag=32），连同方向字段与微秒时间戳一起返回。
//! 16 列: symbol,datetime,date,time,exchtime,localtime,index,price,volume,turnover,flag,order_type,func_code,ask_order,bid_order,num
use std::fs::File;
use std::io::Read;
use std::path::Path;

const COL_EXCHTIME: usize = 4;
const COL_VOLUME: usize = 8;
const COL_TURNOVER: usize = 9;
const COL_FLAG: usize = 10;
const COL_ASK_ORDER: usize = 13;
const COL_BID_ORDER: usize = 14;
const OFFSET_US: i64 = 8 * 3600 * 1_000_000;

#[derive(Clone, Copy, Debug, Default)]
pub struct Tick {
    pub us: i64,       // 东八区微秒，已加 8 小时
    pub vol: f64,
    pub turnover: f64,
    pub flag: i32,
    pub bid_order: i64,
    pub ask_order: i64,
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
    let mut i = 0;
    while i < bytes.len() && bytes[i] == b' ' { i += 1; }
    let mut val: i64 = 0;
    while i < bytes.len() {
        let c = bytes[i];
        if c < b'0' || c > b'9' { break; }
        val = val * 10 + (c - b'0') as i64;
        i += 1;
    }
    val
}

#[inline]
fn parse_f64_fast(bytes: &[u8]) -> f64 {
    // 20250303 起部分文件把整数写成小数（32.0），统一走小数解析
    let mut i = 0;
    let mut neg = false;
    if i < bytes.len() && bytes[i] == b'-' { neg = true; i += 1; }
    let mut int_part = 0.0f64;
    while i < bytes.len() && bytes[i] >= b'0' && bytes[i] <= b'9' { int_part = int_part * 10.0 + (bytes[i] - b'0') as f64; i += 1; }
    let mut frac = 0.0f64;
    let mut scale = 1.0f64;
    if i < bytes.len() && bytes[i] == b'.' {
        i += 1;
        while i < bytes.len() && bytes[i] >= b'0' && bytes[i] <= b'9' { frac = frac * 10.0 + (bytes[i] - b'0') as f64; scale *= 10.0; i += 1; }
    }
    let r = int_part + frac / scale;
    if neg { -r } else { r }
}

#[inline]
fn parse_line(line: &[u8]) -> Option<Tick> {
    if line.is_empty() { return None; }
    let mut fields: [&[u8]; 16] = [&[][..]; 16];
    let mut start = 0;
    let mut col = 0;
    for (i, &b) in line.iter().enumerate() {
        if b == b',' { if col < 16 { fields[col] = &line[start..i]; } col += 1; start = i + 1; }
    }
    if col < 16 { fields[col] = &line[start..]; }
    if col < 14 { return None; }
    let us = parse_i64_fast(fields[COL_EXCHTIME]);
    if us == 0 { return None; }
    let flag = parse_i64_fast(fields[COL_FLAG]) as i32;
    if flag != 32 && flag != 66 && flag != 83 { return None; }
    Some(Tick {
        us: us + OFFSET_US,
        vol: parse_f64_fast(fields[COL_VOLUME]),
        turnover: parse_f64_fast(fields[COL_TURNOVER]),
        flag,
        ask_order: parse_i64_fast(fields[COL_ASK_ORDER]),
        bid_order: parse_i64_fast(fields[COL_BID_ORDER]),
    })
}

pub fn read_ticks(code: &str, date: i64) -> std::io::Result<Vec<Tick>> {
    let filename = format!("{}_{}_transaction.csv", code, date);
    let path = resolve_stock_path(date, "transaction", &filename)?;
    let mut content = String::new();
    File::open(&path)?.read_to_string(&mut content)?;
    let data = content.as_bytes();
    let mut out = Vec::with_capacity(data.len() / 90 + 1);
    let mut start = 0;
    let mut first = true;
    for i in 0..data.len() {
        if data[i] == b'\n' {
            let mut line = &data[start..i];
            if line.last() == Some(&b'\r') { line = &line[..line.len() - 1]; }
            if first { first = false; } else if let Some(t) = parse_line(line) { out.push(t); }
            start = i + 1;
        }
    }
    if start < data.len() {
        let mut line = &data[start..];
        if line.last() == Some(&b'\r') { line = &line[..line.len() - 1]; }
        if let Some(t) = parse_line(line) { out.push(t); }
    }
    Ok(out)
}
