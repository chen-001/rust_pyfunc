//! 逐股票撤单表（时间口径，毫秒）+ 撤买撤卖不平衡 + 从未成交过就撤的比例
//! 构建: cargo build --release --bin play_table
//! 运行: PLAY_THREADS=16 ./target/release/play_table 20220819 > 表.csv
mod fast_csv_reader;

use fast_csv_reader::{read_trade_fast_retreat, TradeRecord};
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

const SYMBOL_MAP: &str = "/ssd_data/data/basic_info/symbol_map.csv";
const SESSION_START: i64 = 9 * 3600 + 30 * 60;
const SESSION_END: i64 = 13 * 3600 + 32 * 60;

fn read_symbol_map(path: &str) -> HashSet<String> {
    let txt = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("symbol_map: {}", e));
    let mut set = HashSet::new();
    for (k, line) in txt.lines().enumerate() {
        if k == 0 { continue; }
        if let Some(c) = line.split(',').next() {
            let c = c.trim();
            if c.len() == 6 && c.chars().all(|x| x.is_ascii_digit()) { set.insert(c.to_string()); }
        }
    }
    set
}

fn list_codes(date: i64, wl: &HashSet<String>) -> Vec<String> {
    let dir = format!("/ssd_data/stock/{}/transaction", date);
    let mut out = Vec::new();
    for e in std::fs::read_dir(&dir).unwrap_or_else(|e| panic!("read_dir: {}", e)) {
        let n = e.unwrap().file_name().to_string_lossy().to_string();
        if !n.ends_with("_transaction.csv") { continue; }
        let c = n.split('_').next().unwrap_or("").to_string();
        if wl.contains(&c) { out.push(c); }
    }
    out.sort();
    out
}

fn interp(vid: &[i64], vt: &[i64], id: i64) -> f64 {
    let n = vid.len();
    if n == 0 { return 0.0; }
    match vid.binary_search(&id) {
        Ok(k) => vt[k] as f64,
        Err(k) => {
            if k == 0 { return vt[0] as f64; }
            if k >= n { return vt[n - 1] as f64; }
            let (id0, id1) = (vid[k - 1], vid[k]);
            let (t0, t1) = (vt[k - 1] as f64, vt[k] as f64);
            if id1 == id0 { return t0; }
            t0 + (t1 - t0) * ((id - id0) as f64) / ((id1 - id0) as f64)
        }
    }
}

fn pct(v: &mut Vec<f64>, q: f64) -> f64 {
    if v.is_empty() { return f64::NAN; }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[(((v.len() - 1) as f64) * q).round() as usize]
}

fn per_stock(code: &str, date: i64) -> Option<String> {
    let recs: Vec<TradeRecord> = read_trade_fast_retreat(code, date).ok()?;
    let mut agg: Vec<(i64, i64)> = Vec::new();
    let mut cancels: Vec<(i64, i64, f64, bool)> = Vec::new(); // (被撤编号, 时间, 撤单量, 是否买方)
    let mut first_fill: HashMap<i64, i64> = HashMap::new();
    let mut seen: HashSet<i64> = HashSet::new();
    let mut turnover = 0.0f64;
    for r in &recs {
        let sod = (r.time_us / 1_000_000).rem_euclid(86400);
        if sod < SESSION_START || sod > SESSION_END { continue; }
        match r.flag {
            66 | 83 => {
                turnover += r.turnover;
                let (a, p) = if r.flag == 66 { (r.bid_order, r.ask_order) } else { (r.ask_order, r.bid_order) };
                if a > 0 { agg.push((a, r.time_us)); seen.insert(a); }
                if p > 0 { let e = first_fill.entry(p).or_insert(r.time_us); if r.time_us < *e { *e = r.time_us; } seen.insert(p); }
            }
            32 => {
                let (cid, is_buy) = if r.bid_order > 0 { (r.bid_order, true) } else { (r.ask_order, false) };
                if cid > 0 { cancels.push((cid, r.time_us, r.volume, is_buy)); }
            }
            _ => {}
        }
    }
    if agg.is_empty() || cancels.is_empty() || first_fill.is_empty() { return None; }
    agg.sort_unstable();
    let vid: Vec<i64> = agg.iter().map(|x| x.0).collect();
    let vt: Vec<i64> = agg.iter().map(|x| x.1).collect();

    let mut c_ms: Vec<f64> = Vec::with_capacity(cancels.len());
    let (mut buy_v, mut sell_v) = (0.0f64, 0.0f64);
    let mut never = 0f64;
    for &(cid, t, vol, is_buy) in &cancels {
        let d = (t as f64 - interp(&vid, &vt, cid)) / 1000.0;
        if d >= 0.0 { c_ms.push(d); }
        if is_buy { buy_v += vol; } else { sell_v += vol; }
        if !seen.contains(&cid) { never += 1.0; }
    }
    let mut w_ms: Vec<f64> = Vec::with_capacity(first_fill.len());
    for (&pid, &t) in &first_fill {
        let d = (t as f64 - interp(&vid, &vt, pid)) / 1000.0;
        if d >= 0.0 { w_ms.push(d); }
    }
    if c_ms.is_empty() || w_ms.is_empty() { return None; }
    let n_cancel_rows = cancels.len();
    let n_cancel_used = c_ms.len();
    let n_wait_orders = w_ms.len();
    let mut cp = c_ms.clone();
    let mut g2 = c_ms;
    let mut w2 = w_ms;
    let med_c = pct(&mut g2, 0.5);
    let p25 = pct(&mut cp, 0.25);
    let p75 = pct(&mut cp, 0.75);
    let med_w = pct(&mut w2, 0.5);
    let tot = buy_v + sell_v;
    let imb = if tot > 0.0 { (buy_v - sell_v) / tot } else { f64::NAN };
    let nf = never / n_cancel_rows as f64;
    Some(format!("{},{},{:.1},{:.1},{:.1},{:.1},{},{},{},{:.0},{:.4},{:.4}",
        code, date, med_c, p25, p75, med_w, n_cancel_rows, n_cancel_used, n_wait_orders, turnover, imb, nf))
}

fn main() {
    let date: i64 = std::env::args().nth(1).unwrap().parse().unwrap();
    let wl = read_symbol_map(SYMBOL_MAP);
    let codes = list_codes(date, &wl);
    let nt: usize = std::env::var("PLAY_THREADS").ok().and_then(|s| s.parse().ok()).unwrap_or(16);
    let next = AtomicUsize::new(0);
    let acc: Mutex<Vec<String>> = Mutex::new(Vec::new());
    std::thread::scope(|sc| {
        for _ in 0..nt {
            sc.spawn(|| loop {
                let k = next.fetch_add(1, Ordering::Relaxed);
                if k >= codes.len() { break; }
                if let Some(line) = per_stock(&codes[k], date) { acc.lock().unwrap().push(line); }
            });
        }
    });
    let mut rows = acc.into_inner().unwrap();
    rows.sort();
    println!("code,date,med_cancel_ms,p25_cancel_ms,p75_cancel_ms,med_wait_ms,n_cancel_rows,n_cancel_used,n_wait_orders,day_turnover,imb_cancel,never_frac");
    for r in rows { println!("{}", r); }
}
