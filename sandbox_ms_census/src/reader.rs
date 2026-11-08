//! 读逐笔成交与快照时间。列按文件表头定位。
use std::fs;

pub struct Rec {
    pub cs: i64, // 东八区日内 10 毫秒序号（9:30:00 = 3,420,000）
    pub vol: f64,
    pub turnover: f64,
    pub dir: u8, // 0 主买，1 主卖
    pub oid: i64,
}

fn path(date: i64, sub: &str, code: &str) -> String {
    format!("/ssd_data/stock/{date}/{sub}/{code}_{date}_{sub}.csv")
}

/// 整数部分 + 按第一位小数四舍五入（exchtime 在部分日期带 .25/.75 的存储误差）。
#[inline]
fn parse_round(b: &[u8]) -> i64 {
    let mut i = 0;
    let mut v = 0i64;
    while i < b.len() && b[i].is_ascii_digit() {
        v = v * 10 + (b[i] - b'0') as i64;
        i += 1;
    }
    if i + 1 < b.len() && b[i] == b'.' && b[i + 1] >= b'5' {
        v += 1;
    }
    v
}

#[inline]
fn parse_f(b: &[u8]) -> f64 {
    let mut i = 0;
    let mut v = 0f64;
    while i < b.len() && b[i].is_ascii_digit() {
        v = v * 10.0 + (b[i] - b'0') as f64;
        i += 1;
    }
    if i < b.len() && b[i] == b'.' {
        i += 1;
        let mut s = 0.1f64;
        while i < b.len() && b[i].is_ascii_digit() {
            v += (b[i] - b'0') as f64 * s;
            s *= 0.1;
            i += 1;
        }
    }
    v
}

const DAY_US: i64 = 86_400_000_000;
const OFFSET_US: i64 = 8 * 3600 * 1_000_000;

fn col_of(header: &[u8], name: &[u8]) -> usize {
    header.split(|&c| c == b',').position(|f| f == name).unwrap_or_else(|| panic!("缺列 {}", String::from_utf8_lossy(name)))
}

pub fn read_trades(code: &str, date: i64) -> std::io::Result<Vec<Rec>> {
    let data = fs::read(path(date, "transaction", code))?;
    let nl = data.iter().position(|&c| c == b'\n').unwrap_or(data.len());
    let head = &data[..nl];
    let (c_ex, c_vol, c_to, c_fl, c_ask, c_bid) = (
        col_of(head, b"exchtime"),
        col_of(head, b"volume"),
        col_of(head, b"turnover"),
        col_of(head, b"flag"),
        col_of(head, b"ask_order"),
        col_of(head, b"bid_order"),
    );
    let mut out = Vec::with_capacity(data.len() / 100);
    let mut f: [&[u8]; 16] = [&[]; 16];
    for line in data[(nl + 1).min(data.len())..].split(|&c| c == b'\n') {
        let line = if line.last() == Some(&b'\r') { &line[..line.len() - 1] } else { line };
        if line.is_empty() {
            continue;
        }
        let mut n = 0;
        for p in line.split(|&c| c == b',') {
            if n < 16 {
                f[n] = p;
            }
            n += 1;
        }
        if n < 15 {
            continue;
        }
        let flag = parse_f(f[c_fl]) as i64;
        if flag != 66 && flag != 83 {
            continue;
        }
        let turnover = parse_f(f[c_to]);
        if !(turnover > 0.0) {
            continue;
        }
        let us = parse_round(f[c_ex]);
        if us == 0 {
            continue;
        }
        let day_us = (us + OFFSET_US).rem_euclid(DAY_US);
        let dir = if flag == 66 { 0 } else { 1 };
        let oid = parse_f(f[if flag == 66 { c_bid } else { c_ask }]) as i64;
        out.push(Rec { cs: (day_us + 5_000) / 10_000, vol: parse_f(f[c_vol]), turnover, dir, oid });
    }
    Ok(out)
}

/// 快照 time 字段（HHMMSSmmm）转成当日 10 毫秒序号。
pub fn read_snaps(code: &str, date: i64) -> std::io::Result<Vec<i64>> {
    let data = fs::read(path(date, "market_data", code))?;
    let nl = data.iter().position(|&c| c == b'\n').unwrap_or(data.len());
    let c_t = col_of(&data[..nl], b"time");
    let mut out = Vec::with_capacity(6000);
    for line in data[(nl + 1).min(data.len())..].split(|&c| c == b'\n') {
        if line.is_empty() {
            continue;
        }
        let Some(p) = line.split(|&c| c == b',').nth(c_t) else { continue };
        let t = parse_f(p) as i64;
        let (h, m, s, ms) = (t / 10_000_000, t / 100_000 % 100, t / 1000 % 100, t % 1000);
        out.push((h * 3600 + m * 60 + s) * 100 + ms / 10);
    }
    Ok(out)
}
