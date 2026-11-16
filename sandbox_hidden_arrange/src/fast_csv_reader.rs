//! ⚠️ 仅限独立 `sandbox_*` 探索 crate 使用（纯 std，无第三方依赖）！
//! 严禁把本文件或任何私有 CSV 读取逻辑复制进正式库 `src/`！
//! 升级并入 `src/` 时必须删除私有 reader，直接调用正式库统一入口：
//!   `use crate::fast_csv_reader::{read_trade_f64_inner, read_market_f64_inner, TradeRecordF64 as TradeRecord, MarketRecordF64 as MarketRecord};`
//!
//! 本模板已同步正式库的三项底层改进：
//!   1. 单次 `File::open`（不做前置 `exists()` 二次查找）
//!   2. 线程局部 `Vec<u8>` 原始字节缓冲复用（免每次堆重分配与 UTF-8 校验）
//!   3. 解析完立即 `POSIX_FADV_DONTNEED` 释放 OS 页缓存，防止全市场探索挤占系统缓存
#![allow(dead_code)]
use std::cell::RefCell;
use std::fs::File;
use std::io::Read;
use std::path::Path;

const RETAINED_CSV_BUF_CAP: usize = 8 * 1024 * 1024;
const SHRINK_CSV_BUF_THRESHOLD: usize = 16 * 1024 * 1024;

thread_local! {
    static CSV_READ_BUF: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
}

#[cfg(target_os = "linux")]
fn drop_page_cache_file(file: &File) {
    use std::os::unix::io::AsRawFd;
    const POSIX_FADV_DONTNEED: i32 = 4;
    extern "C" {
        fn posix_fadvise(fd: i32, offset: i64, len: i64, advice: i32) -> i32;
    }
    unsafe {
        let _ = posix_fadvise(file.as_raw_fd(), 0, 0, POSIX_FADV_DONTNEED);
    }
}

#[cfg(not(target_os = "linux"))]
fn drop_page_cache_file(_file: &File) {}

fn open_stock_file(date: i64, subdir: &str, filename: &str) -> std::io::Result<File> {
    let date_str = date.to_string();
    for env_key in ["RUST_PYFUNC_LEVEL2_ROOT", "RUST_PYFUNC_LEVEL2_PATH"] {
        if let Ok(env_path) = std::env::var(env_key) {
            let trimmed = env_path.trim();
            if !trimmed.is_empty() {
                let p = Path::new(trimmed).join(&date_str).join(subdir).join(filename);
                return File::open(&p);
            }
        }
    }
    let mut first_other_err: Option<std::io::Error> = None;
    for root in [
        "/nvme/level2",
        "/ssd_data/stock",
        "/nas197/binary/stock/sz_alpha/stock",
    ] {
        let p = Path::new(root).join(&date_str).join(subdir).join(filename);
        match File::open(&p) {
            Ok(f) => return Ok(f),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => {
                first_other_err.get_or_insert(e);
            }
        }
    }
    if let Some(err) = first_other_err {
        return Err(err);
    }
    Err(std::io::Error::new(
        std::io::ErrorKind::NotFound,
        format!("not found: date={date}, subdir={subdir}, file={filename}"),
    ))
}

fn with_stock_csv_bytes<T>(
    date: i64,
    subdir: &str,
    filename: &str,
    parse_fn: impl FnOnce(&[u8]) -> T,
) -> std::io::Result<T> {
    let mut file = open_stock_file(date, subdir, filename)?;
    let file_size = file.metadata().map(|m| m.len() as usize).unwrap_or(0);
    let res = CSV_READ_BUF.with(|cell| -> std::io::Result<T> {
        if let Ok(mut buf) = cell.try_borrow_mut() {
            buf.clear();
            let needed = file_size.saturating_add(64);
            if buf.capacity() < needed {
                buf.reserve(needed);
            }
            file.read_to_end(&mut buf)?;
            let out = parse_fn(&buf);
            buf.clear();
            if buf.capacity() > SHRINK_CSV_BUF_THRESHOLD {
                buf.shrink_to(RETAINED_CSV_BUF_CAP);
            }
            Ok(out)
        } else {
            let mut tmp = Vec::with_capacity(file_size.saturating_add(64));
            file.read_to_end(&mut tmp)?;
            Ok(parse_fn(&tmp))
        }
    });
    drop_page_cache_file(&file);
    res
}

const AFTERNOON_START_SEC: i64 = 13 * 3600;
const AFTERNOON_END_SEC: i64 = 14 * 3600 + 57 * 60;
const MORNING_START_SEC: i64 = 9 * 3600 + 30 * 60;
const MORNING_END_SEC: i64 = 11 * 3600 + 30 * 60;
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
    pub raw_sec: u32,
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
    pub raw_sec: u32,
    pub ask_prcs: [f64; 10],
    pub ask_vols: [f64; 10],
    pub bid_prcs: [f64; 10],
    pub bid_vols: [f64; 10],
}

#[inline]
fn normalize_symbol_digits(bytes: &[u8]) -> Option<&[u8]> {
    let mut s = bytes;
    while let [b' ', rest @ ..] = s {
        s = rest;
    }
    while let [rest @ .., b' '] = s {
        s = rest;
    }
    if let Some(dot) = s.iter().position(|&b| b == b'.') {
        if !s[dot + 1..].iter().all(|&b| b == b'0') {
            return None;
        }
        s = &s[..dot];
    }
    if s.is_empty() || !s.iter().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let first_nonzero = s.iter().position(|&b| b != b'0').unwrap_or(s.len() - 1);
    Some(&s[first_nonzero..])
}

#[inline]
fn symbol_field_matches(field: &[u8], code: &str) -> bool {
    let code = code.as_bytes();
    if field == code {
        return true;
    }
    match (normalize_symbol_digits(field), normalize_symbol_digits(code)) {
        (Some(a), Some(b)) => a == b,
        _ => false,
    }
}

#[inline]
fn parse_i64_fast(bytes: &[u8]) -> i64 {
    let mut neg = false;
    let mut i = 0;
    while i < bytes.len() && bytes[i] == b' ' {
        i += 1;
    }
    if i < bytes.len() && (bytes[i] == b'-' || bytes[i] == b'+') {
        neg = bytes[i] == b'-';
        i += 1;
    }
    let mut val: i64 = 0;
    while i < bytes.len() {
        let c = bytes[i];
        if c < b'0' || c > b'9' {
            break;
        }
        val = val * 10 + (c - b'0') as i64;
        i += 1;
    }
    if neg {
        -val
    } else {
        val
    }
}

#[inline]
fn parse_decimal_fast(bytes: &[u8]) -> f64 {
    let mut i = 0;
    let mut neg = false;
    if i < bytes.len() && bytes[i] == b'-' {
        neg = true;
        i += 1;
    } else if i < bytes.len() && bytes[i] == b'+' {
        i += 1;
    }
    let mut int_part = 0.0f64;
    while i < bytes.len() && bytes[i] >= b'0' && bytes[i] <= b'9' {
        int_part = int_part * 10.0 + (bytes[i] - b'0') as f64;
        i += 1;
    }
    let mut frac = 0.0f64;
    let mut scale = 1.0f64;
    if i < bytes.len() && bytes[i] == b'.' {
        i += 1;
        while i < bytes.len() && bytes[i] >= b'0' && bytes[i] <= b'9' {
            frac = frac * 10.0 + (bytes[i] - b'0') as f64;
            scale *= 10.0;
            i += 1;
        }
    }
    let mut r = int_part + frac / scale;
    if neg {
        r = -r;
    }
    r
}

#[inline]
fn parse_f64_fast(bytes: &[u8]) -> f64 {
    let mut has_dot = false;
    let mut has_exp = false;
    let mut all_digits = true;
    for (i, &c) in bytes.iter().enumerate() {
        if c == b'.' {
            has_dot = true;
            continue;
        }
        if c == b'e' || c == b'E' {
            has_exp = true;
            continue;
        }
        if c == b'-' || c == b'+' {
            if i != 0 && bytes[i - 1] != b'e' && bytes[i - 1] != b'E' {
                all_digits = false;
                break;
            }
            continue;
        }
        if c < b'0' || c > b'9' {
            all_digits = false;
            break;
        }
    }
    if all_digits && !has_exp {
        if !has_dot {
            return parse_i64_fast(bytes) as f64;
        }
        return parse_decimal_fast(bytes);
    }
    std::str::from_utf8(bytes)
        .ok()
        .and_then(|s| s.trim().parse::<f64>().ok())
        .unwrap_or(f64::NAN)
}

#[inline]
fn parse_line(
    line: &[u8],
    with_retreat: bool,
    with_afternoon_adjust: bool,
    code: &str,
    date: i64,
) -> Option<TradeRecord> {
    if line.is_empty() {
        return None;
    }
    let mut fields: [&[u8]; 15] = [&[][..]; 15];
    let mut start = 0;
    let mut col = 0;
    for (i, &b) in line.iter().enumerate() {
        if b == b',' {
            if col < 15 {
                fields[col] = &line[start..i];
            }
            col += 1;
            start = i + 1;
        }
    }
    if col < 15 {
        fields[col] = &line[start..];
    }
    if !symbol_field_matches(fields[0], code) || parse_i64_fast(fields[2]) != date {
        return None;
    }
    let flag_bytes = fields[COL_FLAG];
    let flag = if flag_bytes.is_empty() {
        0
    } else {
        parse_i64_fast(flag_bytes) as i32
    };
    if !with_retreat && flag == 32 {
        return None;
    }
    let exchtime_us = parse_i64_fast(fields[COL_EXCHTIME]);
    if exchtime_us == 0 {
        return None;
    }
    let total_us = exchtime_us + OFFSET_US;
    let day_offset = ((exchtime_us / 1_000_000) + 8 * 3600).rem_euclid(86400);
    let final_us = if with_afternoon_adjust {
        if day_offset >= AFTERNOON_START_SEC && day_offset <= AFTERNOON_END_SEC {
            total_us - SHIFT_US
        } else if day_offset >= MORNING_START_SEC && day_offset <= MORNING_END_SEC {
            total_us
        } else {
            return None;
        }
    } else {
        total_us
    };
    let time_sec = (final_us as f64) / 1_000_000.0;
    Some(TradeRecord {
        time_us: final_us,
        time_sec,
        price: parse_f64_fast(fields[COL_PRICE]),
        volume: parse_f64_fast(fields[COL_VOLUME]),
        turnover: parse_f64_fast(fields[COL_TURNOVER]),
        flag,
        bid_order: parse_i64_fast(fields[COL_BID_ORDER]),
        ask_order: parse_i64_fast(fields[COL_ASK_ORDER]),
        index: parse_i64_fast(fields[COL_INDEX]),
        raw_sec: day_offset as u32,
    })
}

fn parse_chunk(data: &[u8], with_retreat: bool, adj: bool, code: &str, date: i64) -> Vec<TradeRecord> {
    let mut out = Vec::with_capacity(data.len() / 80 + 1);
    let mut start = 0;
    for i in 0..data.len() {
        if data[i] == b'\n' {
            let line = &data[start..i];
            let line = if line.last() == Some(&b'\r') {
                &line[..line.len() - 1]
            } else {
                line
            };
            if let Some(r) = parse_line(line, with_retreat, adj, code, date) {
                out.push(r);
            }
            start = i + 1;
        }
    }
    if start < data.len() {
        let line = &data[start..];
        let line = if line.last() == Some(&b'\r') {
            &line[..line.len() - 1]
        } else {
            line
        };
        if let Some(r) = parse_line(line, with_retreat, adj, code, date) {
            out.push(r);
        }
    }
    out
}

pub fn read_trade_fast_inner(
    code: &str,
    date: i64,
    with_retreat: bool,
    with_afternoon_adjust: bool,
) -> std::io::Result<Vec<TradeRecord>> {
    let filename = format!("{}_{}_transaction.csv", code, date);
    with_stock_csv_bytes(date, "transaction", &filename, |bytes| {
        parse_chunk(bytes, with_retreat, with_afternoon_adjust, code, date)
    })
}

pub fn read_trade_fast(code: &str, date: i64) -> std::io::Result<Vec<TradeRecord>> {
    read_trade_fast_inner(code, date, false, true)
}

#[inline]
fn parse_market_line(line: &[u8], adj: bool, code: &str, date: i64) -> Option<MarketRecord> {
    if line.is_empty() {
        return None;
    }
    let mut fields: [&[u8]; 61] = [&[][..]; 61];
    let mut start = 0;
    let mut col = 0;
    for (i, &b) in line.iter().enumerate() {
        if b == b',' {
            if col < 61 {
                fields[col] = &line[start..i];
            }
            col += 1;
            start = i + 1;
        }
    }
    if col < 61 {
        fields[col] = &line[start..];
    }
    if !symbol_field_matches(fields[0], code) || parse_i64_fast(fields[2]) != date {
        return None;
    }
    let ask_prc1 = parse_f64_fast(fields[MKT_COL_ASK_PRC_BASE]);
    let bid_prc1 = parse_f64_fast(fields[MKT_COL_ASK_PRC_BASE + 2]);
    if ask_prc1 == 0.0 || bid_prc1 == 0.0 {
        return None;
    }
    let exchtime_us = parse_i64_fast(fields[MKT_COL_EXCHTIME]);
    if exchtime_us == 0 {
        return None;
    }
    let total_us = exchtime_us + OFFSET_US;
    let day_offset = ((exchtime_us / 1_000_000) + 8 * 3600).rem_euclid(86400);
    let final_us = if adj {
        if day_offset >= AFTERNOON_START_SEC && day_offset <= AFTERNOON_END_SEC {
            total_us - SHIFT_US
        } else if day_offset >= MORNING_START_SEC && day_offset <= MORNING_END_SEC {
            total_us
        } else {
            return None;
        }
    } else {
        total_us
    };
    let time_sec = (final_us as f64) / 1_000_000.0;
    let mut ask_prcs = [0f64; 10];
    let mut ask_vols = [0f64; 10];
    let mut bid_prcs = [0f64; 10];
    let mut bid_vols = [0f64; 10];
    for i in 0..10 {
        let base = MKT_COL_ASK_PRC_BASE + i * 4;
        ask_prcs[i] = parse_f64_fast(fields[base]);
        ask_vols[i] = parse_f64_fast(fields[base + 1]);
        bid_prcs[i] = parse_f64_fast(fields[base + 2]);
        bid_vols[i] = parse_f64_fast(fields[base + 3]);
    }
    Some(MarketRecord {
        time_us: final_us,
        time_sec,
        last_prc: parse_f64_fast(fields[MKT_COL_LAST_PRC]),
        volume: parse_f64_fast(fields[MKT_COL_VOLUME]),
        turnover: parse_f64_fast(fields[MKT_COL_TURNOVER]),
        total_ask_vol: parse_f64_fast(fields[MKT_COL_TOTAL_ASK_VOL]),
        total_bid_vol: parse_f64_fast(fields[MKT_COL_TOTAL_BID_VOL]),
        raw_sec: day_offset as u32,
        ask_prcs,
        ask_vols,
        bid_prcs,
        bid_vols,
    })
}

fn parse_market_chunk(data: &[u8], adj: bool, code: &str, date: i64) -> Vec<MarketRecord> {
    let mut out = Vec::with_capacity(data.len() / 200 + 1);
    let mut start = 0;
    for i in 0..data.len() {
        if data[i] == b'\n' {
            let line = &data[start..i];
            let line = if line.last() == Some(&b'\r') {
                &line[..line.len() - 1]
            } else {
                line
            };
            if let Some(r) = parse_market_line(line, adj, code, date) {
                out.push(r);
            }
            start = i + 1;
        }
    }
    if start < data.len() {
        let line = &data[start..];
        let line = if line.last() == Some(&b'\r') {
            &line[..line.len() - 1]
        } else {
            line
        };
        if let Some(r) = parse_market_line(line, adj, code, date) {
            out.push(r);
        }
    }
    out
}

pub fn read_market_fast_inner(
    code: &str,
    date: i64,
    with_afternoon_adjust: bool,
) -> std::io::Result<Vec<MarketRecord>> {
    let filename = format!("{}_{}_market_data.csv", code, date);
    with_stock_csv_bytes(date, "market_data", &filename, |bytes| {
        parse_market_chunk(bytes, with_afternoon_adjust, code, date)
    })
}

pub fn read_market_fast(code: &str, date: i64) -> std::io::Result<Vec<MarketRecord>> {
    read_market_fast_inner(code, date, true)
}
