//! 单日全市场成交普查（sandbox）。读一天全市场逐笔，输出每股成交笔数/金额/尾部占比，
//! 以及全市场分钟曲线与单笔金额直方图。构建与运行：
//!   cd sandbox_trade_census && cargo build --release
//!   ./target/release/trade_census 20240930 > out.json
mod fast_csv_reader;
use fast_csv_reader::read_trade_raw;
use rayon::prelude::*;
use serde::Serialize;

const SEC_0915: i64 = 9 * 3600 + 15 * 60;
const SEC_0930: i64 = 9 * 3600 + 30 * 60;
const SEC_1130: i64 = 11 * 3600 + 30 * 60;
const SEC_1300: i64 = 13 * 3600;
const SEC_1500: i64 = 15 * 3600;

const N_MIN: usize = 240;
const BIN_LO: f64 = 1.5;
const BIN_W: f64 = 0.01;
const N_BIN: usize = 700;

#[derive(Serialize)]
struct StockStat {
    n: u64,
    amount: f64,
    volume: f64,
    top1: f64,
    top10: f64,
    p50: f64,
    max_amt: f64,
    auc_amt: f64,
    auc_n: u64,
    close_amt: f64,
    close_n: u64,
}

#[derive(Serialize)]
struct Out {
    date: i64,
    n_codes: usize,
    n_valid: usize,
    n_fail: usize,
    codes: Vec<String>,
    stats: Vec<StockStat>,
    minute_cnt: Vec<u64>,
    minute_amt: Vec<f64>,
    hist_cnt: Vec<u64>,
    hist_amt: Vec<f64>,
    hist_lo: f64,
    hist_w: f64,
}

fn stock_universe(date: i64) -> std::io::Result<Vec<String>> {
    use std::io::{Error, ErrorKind};
    let basic = std::env::var("RUST_PYFUNC_BASIC_INFO_DIR").unwrap_or("/ssd_data/data/basic_info".into());
    let root = std::env::var("RUST_PYFUNC_LEVEL2_ROOT").unwrap_or("/ssd_data/stock".into());
    let dir = std::path::Path::new(&root).join(date.to_string()).join("transaction");
    let text = std::fs::read_to_string(std::path::Path::new(&basic).join("symbol_map.csv"))
        .map_err(|_| Error::new(ErrorKind::InvalidData, "股票主表缺失"))?;
    let mut stocks: Vec<String> = text.lines().skip(1).filter_map(|l| {
        let c = l.split(',').next().unwrap_or("").trim();
        (!c.is_empty() && c.len() <= 6 && c.bytes().all(|b| b.is_ascii_digit()))
            .then(|| format!("{c:0>6}"))
    }).collect();
    stocks.sort_unstable();
    stocks.dedup();
    if stocks.is_empty() || !dir.is_dir() {
        return Err(Error::new(ErrorKind::InvalidData, "股票白名单为空或原始目录不存在"));
    }
    Ok(stocks.into_iter()
        .filter(|c| dir.join(format!("{c}_{date}_transaction.csv")).is_file())
        .collect())
}

struct Part {
    stat: StockStat,
    minute_cnt: [u32; N_MIN],
    minute_amt: [f64; N_MIN],
    hist_cnt: [u32; N_BIN],
}

fn stat_one(date: i64, code: &str) -> Option<Part> {
    let trades = read_trade_raw(code, date).ok()?;
    let mut n = 0u64;
    let mut amount = 0.0f64;
    let mut volume = 0.0f64;
    let mut auc_amt = 0.0f64;
    let mut auc_n = 0u64;
    let mut close_amt = 0.0f64;
    let mut close_n = 0u64;
    let mut max_amt = 0.0f64;
    let mut minute = [0u32; N_MIN];
    let mut minute_amt = [0.0f64; N_MIN];
    let mut hist = [0u32; N_BIN];
    let mut amts: Vec<f64> = Vec::with_capacity(trades.len());
    for t in &trades {
        if !(t.turnover > 0.0) { continue; }
        n += 1;
        amount += t.turnover;
        volume += t.volume;
        if t.turnover > max_amt { max_amt = t.turnover; }
        amts.push(t.turnover);
        let s = t.day_sec;
        if s >= SEC_0930 && s < SEC_1130 {
            let i = ((s - SEC_0930) / 60) as usize;
            if i < N_MIN { minute[i] += 1; minute_amt[i] += t.turnover; }
        } else if s >= SEC_1300 && s <= SEC_1500 {
            let i = (120 + ((s - SEC_1300) / 60) as usize).min(N_MIN - 1);
            minute[i] += 1;
            minute_amt[i] += t.turnover;
        } else if s >= SEC_0915 && s < SEC_0930 {
            auc_amt += t.turnover;
            auc_n += 1;
        } else {
            close_amt += t.turnover;
            close_n += 1;
        }
        let b = ((t.turnover.log10() - BIN_LO) / BIN_W) as isize;
        hist[b.clamp(0, N_BIN as isize - 1) as usize] += 1;
    }
    if n == 0 { return None; }
    let n_us = n as usize;
    let k1 = (n_us / 100).max(1);
    let mid = n_us / 2;
    amts.select_nth_unstable_by(mid, |a, b| b.partial_cmp(a).unwrap());
    let p50 = amts[mid];
    amts.select_nth_unstable_by(k1 - 1, |a, b| b.partial_cmp(a).unwrap());
    let top1: f64 = amts[..k1].iter().sum::<f64>() / amount;
    let k10 = (n_us / 10).max(1);
    amts.select_nth_unstable_by(k10 - 1, |a, b| b.partial_cmp(a).unwrap());
    let top10: f64 = amts[..k10].iter().sum::<f64>() / amount;
    Some(Part { stat: StockStat { n, amount, volume, top1, top10, p50, max_amt, auc_amt, auc_n, close_amt, close_n }, minute_cnt: minute, minute_amt, hist_cnt: hist })
}

fn main() {
    let date: i64 = std::env::args().nth(1).expect("用法: trade_census <date>").parse().unwrap();
    let codes = stock_universe(date).expect("股票范围或原始目录检查失败");
    eprintln!("date={} codes={}", date, codes.len());

    let parts: Vec<Option<(String, Part)>> = codes
        .par_iter()
        .map(|c| stat_one(date, c).map(|p| (c.clone(), p)))
        .collect();

    let mut out = Out {
        date,
        n_codes: codes.len(),
        n_valid: 0,
        n_fail: 0,
        codes: Vec::new(),
        stats: Vec::new(),
        minute_cnt: vec![0; N_MIN],
        minute_amt: vec![0.0; N_MIN],
        hist_cnt: vec![0; N_BIN],
        hist_amt: vec![0.0; N_BIN],
        hist_lo: BIN_LO,
        hist_w: BIN_W,
    };
    for p in parts {
        match p {
            Some((code, p)) => {
                out.n_valid += 1;
                out.codes.push(code);
                out.stats.push(p.stat);
                for i in 0..N_MIN { out.minute_cnt[i] += p.minute_cnt[i] as u64; out.minute_amt[i] += p.minute_amt[i]; }
                for i in 0..N_BIN { out.hist_cnt[i] += p.hist_cnt[i] as u64; }
            }
            None => out.n_fail += 1,
        }
    }
    eprintln!("valid={} fail={}", out.n_valid, out.n_fail);
    println!("{}", serde_json::to_string(&out).unwrap());
}
