//! 精简版数据读取（单线程，无 pyo3/rayon），从 fast_csv_reader.rs 移植。
//! 读取 transaction / market_data CSV，做 adjust_afternoon 平移。
//! 只保留因子实际用到的字段：盘口不解析第 6~10 档与 last_prc/volume，
//! 逐笔不解析 time_sec 与委托序号 index。
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

const MKT_COL_EXCHTIME: usize = 4;
const MKT_COL_TURNOVER: usize = 15;
const MKT_COL_TOTAL_ASK_VOL: usize = 19;
const MKT_COL_TOTAL_BID_VOL: usize = 20;
const MKT_COL_ASK_PRC_BASE: usize = 21;

#[derive(Clone, Copy, Debug, Default)]
pub struct TradeRecord {
    pub time_us: i64,
    pub price: f64,
    pub volume: f64,
    pub turnover: f64,
    pub flag: i32,
    pub bid_order: i64,
    pub ask_order: i64,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct MarketRecord {
    pub time_us: i64,
    pub turnover: f64,
    pub total_ask_vol: f64,
    pub total_bid_vol: f64,
    pub ask_prc1: f64,
    pub bid_prc1: f64,
    pub ask_vols: [f64; 5],
    pub bid_vols: [f64; 5],
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

/// 一遍扫描解析数字：整数走 i64 累加再转 f64，带小数点的走整数部分 + 小数部分/量纲。
/// 与旧版（先分类再分别走 parse_i64_fast / parse_decimal_fast）逐位相同；
/// 遇到空格、字母、指数等异常字符时退回 trim().parse()，与旧版兜底一致。
#[inline]
fn parse_f64_fast(bytes: &[u8]) -> f64 {
    let n = bytes.len();
    let mut i = 0;
    let mut neg = false;
    if i < n && (bytes[i] == b'-' || bytes[i] == b'+') { neg = bytes[i] == b'-'; i += 1; }
    let mut iv: i64 = 0;
    let mut int_part = 0.0f64;
    while i < n {
        let c = bytes[i];
        if c < b'0' || c > b'9' { break; }
        let d = (c - b'0') as i64;
        iv = iv.wrapping_mul(10).wrapping_add(d);
        int_part = int_part * 10.0 + d as f64;
        i += 1;
    }
    if i == n {
        return (if neg { -iv } else { iv }) as f64;
    }
    if bytes[i] == b'.' {
        i += 1;
        let mut frac = 0.0f64;
        let mut scale = 1.0f64;
        while i < n {
            let c = bytes[i];
            if c < b'0' || c > b'9' { break; }
            frac = frac * 10.0 + (c - b'0') as f64;
            scale *= 10.0;
            i += 1;
        }
        if i == n {
            let r = int_part + frac / scale;
            return if neg { -r } else { r };
        }
    }
    std::str::from_utf8(bytes).ok().and_then(|s| s.trim().parse::<f64>().ok()).unwrap_or(f64::NAN)
}

#[inline]
fn parse_line(line: &[u8], code: &str, date: i64) -> Option<TradeRecord> {
    if line.is_empty() { return None; }
    let mut fields: [&[u8]; 15] = [&[][..]; 15];
    let mut start = 0;
    let mut col = 0;
    for i in memchr::memchr_iter(b',', line) {
        if col < 15 { fields[col] = &line[start..i]; }
        col += 1;
        start = i + 1;
    }
    if col < 15 { fields[col] = &line[start..]; }
    // 非本股票 / 非本日期行直接丢（0=symbol, 2=date；date 解析成整数再比，覆盖 "20251031.0"）
    if fields[0] != code.as_bytes() || parse_i64_fast(fields[2]) != date { return None; }
    // 先解析成数值再比较：部分日期的 flag 列写 "32.0"，按字节比 b"32" 会漏掉撤单
    let flag = if fields[COL_FLAG].is_empty() { 0 } else { parse_i64_fast(fields[COL_FLAG]) as i32 };
    if flag == 32 { return None; }
    let exchtime_us = parse_i64_fast(fields[COL_EXCHTIME]);
    let total_us = exchtime_us + OFFSET_US;
    let day_offset = ((exchtime_us / 1_000_000) + 8 * 3600).rem_euclid(86400);
    let final_us = if day_offset >= AFTERNOON_START_SEC && day_offset <= AFTERNOON_END_SEC { total_us - SHIFT_US }
        else if day_offset >= MORNING_START_SEC && day_offset <= MORNING_END_SEC { total_us }
        else { return None; };
    Some(TradeRecord {
        time_us: final_us,
        price: parse_f64_fast(fields[COL_PRICE]),
        volume: parse_f64_fast(fields[COL_VOLUME]),
        turnover: parse_f64_fast(fields[COL_TURNOVER]),
        flag,
        bid_order: parse_i64_fast(fields[COL_BID_ORDER]),
        ask_order: parse_i64_fast(fields[COL_ASK_ORDER]),
    })
}

fn parse_chunk(data: &[u8], code: &str, date: i64) -> Vec<TradeRecord> {
    let mut out = Vec::with_capacity(data.len() / 130 + 1);
    let mut start = 0;
    for nl in memchr::memchr_iter(b'\n', data) {
        let line = &data[start..nl];
        let line = if line.last() == Some(&b'\r') { &line[..line.len() - 1] } else { line };
        if let Some(r) = parse_line(line, code, date) { out.push(r); }
        start = nl + 1;
    }
    if start < data.len() {
        let line = &data[start..];
        let line = if line.last() == Some(&b'\r') { &line[..line.len() - 1] } else { line };
        if let Some(r) = parse_line(line, code, date) { out.push(r); }
    }
    out
}

pub fn read_trade_fast(code: &str, date: i64) -> std::io::Result<Vec<TradeRecord>> {
    let filename = format!("{}_{}_transaction.csv", code, date);
    let path = resolve_stock_path(date, "transaction", &filename)?;
    let data = std::fs::read(&path)?;
    Ok(parse_chunk(&data, code, date))
}

#[inline]
fn parse_market_line(line: &[u8], code: &str, date: i64) -> Option<MarketRecord> {
    if line.is_empty() { return None; }
    // 只解析到第 41 列（第 5 档挂量的卖量序号位），后面的列不看，逗号扫到那里就停
    const NEED_MAX: usize = MKT_COL_ASK_PRC_BASE + 4 * 4 + 3;
    let mut fields: [&[u8]; 61] = [&[][..]; 61];
    let mut start = 0;
    let mut col = 0;
    for i in memchr::memchr_iter(b',', line) {
        if col > NEED_MAX { break; }
        fields[col] = &line[start..i];
        col += 1;
        start = i + 1;
    }
    if col <= NEED_MAX { fields[col] = &line[start..]; }
    // 非本股票 / 非本日期行直接丢（0=symbol, 2=date）
    if fields[0] != code.as_bytes() || parse_i64_fast(fields[2]) != date { return None; }
    let ask_prc1 = parse_f64_fast(fields[MKT_COL_ASK_PRC_BASE]);
    let bid_prc1 = parse_f64_fast(fields[MKT_COL_ASK_PRC_BASE + 2]);
    if ask_prc1 == 0.0 || bid_prc1 == 0.0 { return None; }
    let exchtime_us = parse_i64_fast(fields[MKT_COL_EXCHTIME]);
    if exchtime_us == 0 { return None; }
    let total_us = exchtime_us + OFFSET_US;
    let day_offset = ((exchtime_us / 1_000_000) + 8 * 3600).rem_euclid(86400);
    let final_us = if day_offset >= AFTERNOON_START_SEC && day_offset <= AFTERNOON_END_SEC { total_us - SHIFT_US }
        else if day_offset >= MORNING_START_SEC && day_offset <= MORNING_END_SEC { total_us }
        else { return None; };
    let mut ask_vols = [0f64; 5];
    let mut bid_vols = [0f64; 5];
    for i in 0..5 {
        let base = MKT_COL_ASK_PRC_BASE + i * 4;
        ask_vols[i] = parse_f64_fast(fields[base + 1]);
        bid_vols[i] = parse_f64_fast(fields[base + 3]);
    }
    Some(MarketRecord {
        time_us: final_us,
        turnover: parse_f64_fast(fields[MKT_COL_TURNOVER]),
        total_ask_vol: parse_f64_fast(fields[MKT_COL_TOTAL_ASK_VOL]),
        total_bid_vol: parse_f64_fast(fields[MKT_COL_TOTAL_BID_VOL]),
        ask_prc1,
        bid_prc1,
        ask_vols,
        bid_vols,
    })
}

fn parse_market_chunk(data: &[u8], code: &str, date: i64) -> Vec<MarketRecord> {
    let mut out = Vec::with_capacity(data.len() / 200 + 1);
    let mut start = 0;
    for nl in memchr::memchr_iter(b'\n', data) {
        let line = &data[start..nl];
        let line = if line.last() == Some(&b'\r') { &line[..line.len() - 1] } else { line };
        if let Some(r) = parse_market_line(line, code, date) { out.push(r); }
        start = nl + 1;
    }
    if start < data.len() {
        let line = &data[start..];
        let line = if line.last() == Some(&b'\r') { &line[..line.len() - 1] } else { line };
        if let Some(r) = parse_market_line(line, code, date) { out.push(r); }
    }
    out
}

pub fn read_market_fast(code: &str, date: i64) -> std::io::Result<Vec<MarketRecord>> {
    let filename = format!("{}_{}_market_data.csv", code, date);
    let path = resolve_stock_path(date, "market_data", &filename)?;
    let data = std::fs::read(&path)?;
    Ok(parse_market_chunk(&data, code, date))
}
