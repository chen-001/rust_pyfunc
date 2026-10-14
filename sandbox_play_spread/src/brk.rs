//! 插值误差上界：撤单编号两侧最近的两个已知主动编号，中间隔多长时间
//! 构建: cargo build --release --bin play_brk
//! 运行: PLAY_THREADS=16 ./target/release/play_brk 20220819
mod fast_csv_reader;

use fast_csv_reader::{read_trade_fast_retreat, TradeRecord};
use std::collections::HashSet;
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

fn pct(v: &mut Vec<f64>, q: f64) -> f64 {
    if v.is_empty() { return f64::NAN; }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[(((v.len() - 1) as f64) * q).round() as usize]
}

/// 返回 (样本数, 区间宽度中位数ms, p90宽度ms, 寿命中位数ms)
fn per_stock(code: &str, date: i64) -> Option<(f64, f64, f64, f64)> {
    let recs: Vec<TradeRecord> = read_trade_fast_retreat(code, date).ok()?;
    let mut agg: Vec<(i64, i64)> = Vec::new();
    let mut cancels: Vec<(i64, i64)> = Vec::new();
    for r in &recs {
        let sod = (r.time_us / 1_000_000).rem_euclid(86400);
        if sod < SESSION_START || sod > SESSION_END { continue; }
        match r.flag {
            66 | 83 => {
                let a = if r.flag == 66 { r.bid_order } else { r.ask_order };
                if a > 0 { agg.push((a, r.time_us)); }
            }
            32 => {
                let cid = if r.bid_order > 0 { r.bid_order } else { r.ask_order };
                if cid > 0 { cancels.push((cid, r.time_us)); }
            }
            _ => {}
        }
    }
    if agg.len() < 1000 || cancels.is_empty() { return None; }
    agg.sort_unstable();
    let vid: Vec<i64> = agg.iter().map(|x| x.0).collect();
    let vt: Vec<i64> = agg.iter().map(|x| x.1).collect();
    let lo = vid[0];
    let hi = vid[vid.len() - 1];
    let mut widths: Vec<f64> = Vec::new();
    let mut lifes: Vec<f64> = Vec::new();
    for &(cid, t) in &cancels {
        if cid <= lo || cid >= hi { continue; }
        match vid.binary_search(&cid) {
            Ok(_) => continue,                       // 编号本身就是主动编号，跳过
            Err(k) => {
                let w = (vt[k] as f64 - vt[k - 1] as f64) / 1000.0;
                if w >= 0.0 { widths.push(w); }
                let life = (t as f64 - vt[k - 1] as f64) / 1000.0;
                lifes.push(life);
            }
        }
    }
    if widths.len() < 100 { return None; }
    let n = widths.len() as f64;
    let mut w2 = widths.clone();
    let mut w3 = widths;
    let mut l2 = lifes;
    let med_w = pct(&mut w2, 0.5);
    let p90_w = pct(&mut w3, 0.9);
    let med_l = pct(&mut l2, 0.5);
    Some((n, med_w, p90_w, med_l))
}

fn main() {
    let date: i64 = std::env::args().nth(1).unwrap().parse().unwrap();
    let wl = read_symbol_map(SYMBOL_MAP);
    let codes = list_codes(date, &wl);
    let nt: usize = std::env::var("PLAY_THREADS").ok().and_then(|s| s.parse().ok()).unwrap_or(16);
    let next = AtomicUsize::new(0);
    let acc: Mutex<Vec<(f64, f64, f64, f64)>> = Mutex::new(Vec::new());
    std::thread::scope(|sc| {
        for _ in 0..nt {
            sc.spawn(|| loop {
                let k = next.fetch_add(1, Ordering::Relaxed);
                if k >= codes.len() { break; }
                if let Some(v) = per_stock(&codes[k], date) { acc.lock().unwrap().push(v); }
            });
        }
    });
    let data = acc.into_inner().unwrap();
    let mut mw: Vec<f64> = data.iter().map(|x| x.1).collect();
    let mut p90w: Vec<f64> = data.iter().map(|x| x.2).collect();
    let mut ml: Vec<f64> = data.iter().map(|x| x.3).collect();
    let mut ratio: Vec<f64> = data.iter().map(|x| x.1 / x.3.max(1e-9)).collect();
    let mut nobs: Vec<f64> = data.iter().map(|x| x.0).collect();
    println!("{{\"date\":{},\"stocks\":{},\"bracket_med_ms_across_stocks\":{:.1},\"bracket_p90_ms_across_stocks\":{:.1},\"lifespan_med_ms_across_stocks\":{:.1},\"bracket_over_lifespan\":{:.4},\"cancel_rows_med\":{:.0}}}",
        date, data.len(), pct(&mut mw, 0.5), pct(&mut p90w, 0.5), pct(&mut ml, 0.5), pct(&mut ratio, 0.5), pct(&mut nobs, 0.5));
}
