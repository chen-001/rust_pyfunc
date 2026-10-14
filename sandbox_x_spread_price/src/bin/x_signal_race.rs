//! 三个候选信号同台比较：纯不平衡 / 不平衡乘点差 / 不平衡除以深度
mod fast_csv_reader;
use fast_csv_reader::{read_market_fast, MarketRecord};
use std::collections::HashSet;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

const FWD_US: i64 = 60_000_000;
const MAX_GAP_US: i64 = 95_000_000;
const MIN_OBS: usize = 200;
const SYMBOL_MAP: &str = "/ssd_data/data/basic_info/symbol_map.csv";

#[derive(Clone, Copy)]
struct Obs { imb: f64, sp_bps: f64, depth: f64, r_mid: f64, r_last: f64, price: f64 }

fn read_symbol_map(path: &str) -> HashSet<String> {
    let txt = std::fs::read_to_string(path).unwrap();
    let mut set = HashSet::new();
    for (k, line) in txt.lines().enumerate() {
        if k == 0 { continue; }
        if let Some(code) = line.split(',').next() { let c = code.trim(); if c.len() == 6 && c.chars().all(|x| x.is_ascii_digit()) { set.insert(c.to_string()); } }
    }
    set
}

fn list_codes(date: i64, wl: &HashSet<String>) -> Vec<String> {
    let dir = format!("/ssd_data/stock/{}/market_data", date);
    let mut out = Vec::new();
    for e in std::fs::read_dir(&dir).unwrap() {
        let n = e.unwrap().file_name().to_string_lossy().to_string();
        if !n.ends_with("_market_data.csv") { continue; }
        let c = n.split('_').next().unwrap_or("").to_string();
        if wl.contains(&c) { out.push(c); }
    }
    out.sort(); out
}

fn per_stock(recs: &[MarketRecord]) -> Vec<Obs> {
    let n = recs.len();
    let mut valid: Vec<usize> = Vec::with_capacity(n);
    for i in 0..n { let r = &recs[i]; if r.last_prc > 0.0 && r.ask_prcs[0] > 0.0 && r.bid_prcs[0] > 0.0 && r.ask_prcs[0] > r.bid_prcs[0] { valid.push(i); } }
    let mut out = Vec::with_capacity(valid.len());
    let mut j = 0usize;
    for (k, &i) in valid.iter().enumerate() {
        if j < k { j = k; }
        let t = recs[i].time_us;
        let sod = (t / 1_000_000).rem_euclid(86400);
        if sod >= 11 * 3600 + 29 * 60 && sod < 11 * 3600 + 30 * 60 { continue; }
        while j < valid.len() && recs[valid[j]].time_us < t + FWD_US { j += 1; }
        if j >= valid.len() { break; }
        if recs[valid[j]].time_us - t > MAX_GAP_US { continue; }
        let r = &recs[i]; let f = &recs[valid[j]];
        let mid = 0.5 * (r.ask_prcs[0] + r.bid_prcs[0]);
        let fmid = 0.5 * (f.ask_prcs[0] + f.bid_prcs[0]);
        let bid5: f64 = r.bid_vols[..5].iter().sum();
        let ask5: f64 = r.ask_vols[..5].iter().sum();
        if bid5 + ask5 <= 0.0 { continue; }
        out.push(Obs {
            imb: (bid5 - ask5) / (bid5 + ask5),
            sp_bps: (r.ask_prcs[0] - r.bid_prcs[0]) / mid * 1e4,
            depth: bid5 + ask5,
            r_mid: (fmid / mid - 1.0) * 1e4,
            r_last: (f.last_prc / r.last_prc - 1.0) * 1e4,
            price: r.last_prc,
        });
    }
    out
}

fn rank(v: &[f64]) -> Vec<f64> {
    let n = v.len(); let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| v[a].partial_cmp(&v[b]).unwrap_or(std::cmp::Ordering::Equal));
    let mut r = vec![0.0; n]; let mut i = 0;
    while i < n { let mut j = i; while j + 1 < n && v[idx[j + 1]] == v[idx[i]] { j += 1; }
        let avg = ((i + j) as f64) / 2.0 + 1.0; for k in i..=j { r[idx[k]] = avg; } i = j + 1; }
    r
}
fn pearson(x: &[f64], y: &[f64]) -> Option<f64> {
    let n = x.len() as f64; if n < 3.0 { return None; }
    let mx = x.iter().sum::<f64>() / n; let my = y.iter().sum::<f64>() / n;
    let (mut a, mut b, mut c) = (0.0, 0.0, 0.0);
    for i in 0..x.len() { let dx = x[i] - mx; let dy = y[i] - my; a += dx * dy; b += dx * dx; c += dy * dy; }
    if b <= 0.0 || c <= 0.0 { return None; }
    Some(a / (b.sqrt() * c.sqrt()))
}
fn spearman(x: &[f64], y: &[f64]) -> Option<f64> { pearson(&rank(x), &rank(y)) }

/// 分 5 组，返回第 5 组减第 1 组的均值、以及单调性（相邻组差为正的次数）
fn quintiles(x: &[f64], y: &[f64]) -> Option<(f64, [f64; 5])> {
    let n = x.len(); if n < 50 { return None; }
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| x[a].partial_cmp(&x[b]).unwrap_or(std::cmp::Ordering::Equal));
    let mut sums = [0.0f64; 5]; let mut cnts = [0.0f64; 5];
    for (r, &i) in idx.iter().enumerate() { let g = (r * 5 / n).min(4); sums[g] += y[i]; cnts[g] += 1.0; }
    let mut m = [0.0f64; 5];
    for g in 0..5 { if cnts[g] < 1.0 { return None; } m[g] = sums[g] / cnts[g]; }
    Some((m[4] - m[0], m))
}

#[derive(Clone, Copy)]
struct Acc { names: f64, ic: f64, ic2: f64, tb: f64, tb2: f64, mono: f64 }
impl Default for Acc { fn default() -> Self { Acc { names: 0.0, ic: 0.0, ic2: 0.0, tb: 0.0, tb2: 0.0, mono: 0.0 } } }

fn run(date: i64) {
    let wl = read_symbol_map(SYMBOL_MAP);
    let codes = list_codes(date, &wl);
    let nt: usize = std::env::var("XS_THREADS").ok().and_then(|s| s.parse().ok()).unwrap_or(12);
    let next = AtomicUsize::new(0);
    let collected: Mutex<Vec<(String, Vec<Obs>)>> = Mutex::new(Vec::new());
    std::thread::scope(|scope| {
        for _ in 0..nt { scope.spawn(|| loop {
            let k = next.fetch_add(1, Ordering::Relaxed);
            if k >= codes.len() { break; }
            let code = &codes[k];
            if let Ok(recs) = read_market_fast(code, date) { let o = per_stock(&recs); if o.len() >= MIN_OBS { collected.lock().unwrap().push((code.clone(), o)); } }
        }); }
    });
    let data = collected.into_inner().unwrap();
    // 信号 0: imb；1: imb * sp_bps；2: imb / depth
    let nsig = 4;
    let sigs: Vec<fn(&Obs) -> f64> = vec![
        |o: &Obs| o.imb,
        |o: &Obs| o.imb * o.sp_bps,
        |o: &Obs| if o.depth > 0.0 { o.imb / o.depth } else { 0.0 },
        |o: &Obs| if o.imb.abs() > 0.5 { o.imb } else { 0.0 },  // 只留极端不平衡
    ];
    let names = ["pure_imb", "imb_x_spread", "imb_div_depth", "imb_extreme_only"];
    let mut acc = vec![Acc::default(); nsig];
    let mut acc_last = vec![Acc::default(); nsig];
    for (_c, obs) in &data {
        for s in 0..nsig {
            let x: Vec<f64> = obs.iter().map(|o| sigs[s](o)).collect();
            let ym: Vec<f64> = obs.iter().map(|o| o.r_mid).collect();
            let yl: Vec<f64> = obs.iter().map(|o| o.r_last).collect();
            let a = &mut acc[s];
            if let Some(v) = spearman(&x, &ym) { a.ic += v; a.ic2 += v * v; }
            if let Some((tb, m)) = quintiles(&x, &ym) { a.tb += tb; a.tb2 += tb * tb;
                let mut mono = 0.0; for g in 0..4 { if m[g + 1] > m[g] { mono += 1.0; } } a.mono += mono; }
            if a.ic.is_finite() { a.names += 1.0; }
            let b = &mut acc_last[s];
            if let Some(v) = spearman(&x, &yl) { b.ic += v; b.ic2 += v * v; b.names += 1.0; }
        }
    }
    let mut parts: Vec<String> = Vec::new();
    for s in 0..nsig {
        let a = &acc[s]; let n = a.names.max(1.0);
        let m = a.ic / n; let se = ((a.ic2 / n - m * m).max(0.0) / n).sqrt();
        let tbm = a.tb / n; let tbse = ((a.tb2 / n - tbm * tbm).max(0.0) / n).sqrt();
        let b = &acc_last[s];
        parts.push(format!("{{\"name\":\"{}\",\"stock_days\":{:.0},\"ic_mid\":{:.4},\"ic_mid_se\":{:.4},\"ic_last\":{:.4},\"q5_minus_q1_bp\":{:.3},\"q5_minus_q1_se\":{:.3},\"mono_frac\":{:.3}}}",
            names[s], a.names, m, se, if b.names > 0.0 { b.ic / b.names } else { f64::NAN }, tbm, tbse, a.mono / n / 4.0));
    }
    println!("{{\"date\":{},\"stock_days\":{},\"signals\":[{}]}}", date, data.len(), parts.join(","));
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    assert!(!args.is_empty(), "usage: x_signal_race <date> [date...]");
    for a in &args { run(a.parse::<i64>().expect("date")); }
}