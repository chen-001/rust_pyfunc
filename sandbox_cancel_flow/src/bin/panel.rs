//! 全股「撤单-成交-挂单存量」30 秒格连续面板（panel_sandbox）
//!
//! 网格按交易所时间对齐到 09:30 起每 30 秒一格，只覆盖连续竞价。
//! 每只股票输出六条直接测出来的序列（新增挂单是这三条的线性组合，不单独当一列）：
//!   cbid  撤买量（flag 32 且 bid_order 非零）
//!   cask  撤卖量（flag 32 且 ask_order 非零）
//!   abuy  主动买成交量（flag 66）
//!   asell 主动卖成交量（flag 83）
//!   dbid  格内 total_bid_vol 的首尾差
//!   dask  格内 total_ask_vol 的首尾差
//! 用法: ./target/release/panel 20220819 000001,600519,...  或  ALL（取当日子样本，见 SIMPLE）
//! 结果一行一只股票，JSON。

use rayon::prelude::*;
use serde::Serialize;

const BUCKET_US: i64 = 30_000_000;
const OPEN_SEC: i64 = 9 * 3600 + 30 * 60;
const GRID: usize = 480; // 09:30-11:30 + 13:00-15:00 = 4h = 480 格

#[derive(Clone, Copy, Default)]
struct M { t: i64, tb: f64, ta: f64 }

#[derive(Serialize)]
struct Row {
    code: String,
    date: i64,
    n_bucket: usize,
    cbid: Vec<f64>,
    cask: Vec<f64>,
    abuy: Vec<f64>,
    asell: Vec<f64>,
    dbid: Vec<f64>,
    dask: Vec<f64>,
}

fn lsec(us: i64) -> i64 { ((us / 1_000_000) + 8 * 3600).rem_euclid(86400) }

fn bucket_of(us: i64) -> Option<usize> {
    let s = lsec(us);
    let (start, end) = if s >= OPEN_SEC && s <= 11 * 3600 + 30 * 60 { (OPEN_SEC, 11 * 3600 + 30 * 60) }
        else if s >= 13 * 3600 && s <= 15 * 3600 { (13 * 3600, 15 * 3600) }
        else { return None; };
    let k = ((s - start) / 30) as usize;
    let off = if start == OPEN_SEC { 0 } else { 240 };
    let idx = off + k;
    if s >= start && s < end && idx < GRID { Some(idx) } else { None }
}

fn load_m(path: &str) -> std::io::Result<Vec<M>> {
    let c = std::fs::read_to_string(path)?;
    let mut out = Vec::with_capacity(c.len() / 700);
    for line in c.lines().skip(1) {
        let f: Vec<&str> = line.split(',').collect();
        if f.len() < 25 { continue; }
        let t: i64 = f[4].parse().unwrap_or(0);
        if t == 0 { continue; }
        out.push(M { t, tb: f[20].parse().unwrap_or(0.0), ta: f[19].parse().unwrap_or(0.0) });
    }
    Ok(out)
}

fn one(code: &str, date: i64) -> Option<Row> {
    let mp = format!("/ssd_data/stock/{}/market_data/{}_{}_market_data.csv", date, code, date);
    let tp = format!("/ssd_data/stock/{}/transaction/{}_{}_transaction.csv", date, code, date);
    let mkt = load_m(&mp).ok()?;
    let txt = std::fs::read_to_string(&tp).ok()?;
    let mut cbid = vec![0f64; GRID];
    let mut cask = vec![0f64; GRID];
    let mut abuy = vec![0f64; GRID];
    let mut asell = vec![0f64; GRID];
    for line in txt.lines().skip(1) {
        let f: Vec<&str> = line.split(',').collect();
        if f.len() < 15 { continue; }
        let t: i64 = f[4].parse().unwrap_or(0);
        if t == 0 { continue; }
        let b = match bucket_of(t) { Some(b) => b, None => continue };
        let flag: i32 = f[10].parse().unwrap_or(0);
        let v: f64 = f[8].parse().unwrap_or(0.0);
        if flag == 32 {
            let bo: i64 = f[14].parse().unwrap_or(0);
            let ao: i64 = f[13].parse().unwrap_or(0);
            if bo > 0 { cbid[b] += v; } else if ao > 0 { cask[b] += v; }
        } else if flag == 66 { abuy[b] += v; } else if flag == 83 { asell[b] += v; }
    }
    // 每格用格内首尾快照算存量差
    let mut first_b = vec![f64::NAN; GRID];
    let mut last_b = vec![f64::NAN; GRID];
    let mut first_a = vec![f64::NAN; GRID];
    let mut last_a = vec![f64::NAN; GRID];
    for m in &mkt {
        let b = match bucket_of(m.t) { Some(b) => b, None => continue };
        if first_b[b].is_nan() { first_b[b] = m.tb; first_a[b] = m.ta; }
        last_b[b] = m.tb; last_a[b] = m.ta;
    }
    let dbid: Vec<f64> = (0..GRID).map(|b| last_b[b] - first_b[b]).collect();
    let dask: Vec<f64> = (0..GRID).map(|b| last_a[b] - first_a[b]).collect();
    let nb = (0..GRID).filter(|&b| !first_b[b].is_nan()).count();
    Some(Row { code: code.into(), date, n_bucket: nb, cbid, cask, abuy, asell, dbid, dask })
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let date: i64 = a.get(1).and_then(|s| s.parse().ok()).unwrap_or(20220819);
    let arg = a.get(2).cloned().unwrap_or_else(|| "ALL".into());
    let codes: Vec<String> = if arg == "ALL" {
        let dir = format!("/ssd_data/stock/{}/transaction", date);
        let mut v: Vec<String> = match std::fs::read_dir(&dir) {
            Ok(rd) => rd.flatten()
                .filter_map(|e| e.file_name().to_str().map(|s| s.to_string()))
                .filter_map(|n| n.split('_').next().map(|s| s.to_string()))
                .filter(|c| c.len() == 6 && c.bytes().all(|b| b.is_ascii_digit()))
                // 只要 A 股正股，ETF（5 开头）等品种的撤单量跟成交完全不是一个量级，混进来会把面板带偏
                .filter(|c| matches!(&c[..3], "000"|"001"|"002"|"003"|"300"|"301"|"600"|"601"|"603"|"605"|"688"))
                .collect(),
            Err(_) => Vec::new(),
        };
        v.sort(); v.dedup();
        // 子样本，先跑十分之一，避免一次读满盘
        let step: usize = std::env::var("PANEL_STEP").ok().and_then(|s| s.parse().ok()).unwrap_or(10);
        v.into_iter().step_by(step.max(1)).collect()
    } else {
        arg.split(',').map(|x| x.trim().to_string()).filter(|x| !x.is_empty()).collect()
    };
    eprintln!("panel_sandbox date={} codes={}", date, codes.len());
    let rows: Vec<Row> = codes.par_iter().filter_map(|c| one(c, date)).collect();
    for r in &rows { println!("{}", serde_json::to_string(r).unwrap()); }
    eprintln!("done rows={}", rows.len());
}
