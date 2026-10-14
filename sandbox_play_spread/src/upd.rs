//! 按「过去 60 秒报价更新次数」分层，看价格梯度控制住报价活跃度之后还剩多少
//! 构建: cargo build --release --bin play_upd
//! 运行: PLAY_THREADS=16 ./target/release/play_upd 20220819 20241231
mod fast_csv_reader;

use fast_csv_reader::{read_market_fast, MarketRecord};
use std::collections::HashSet;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

const FWD_US: i64 = 60_000_000;
const MAX_GAP_US: i64 = 95_000_000;
const MIN_OBS: usize = 200;
const MIN_BUCKET: usize = 30;
const SYMBOL_MAP: &str = "/ssd_data/data/basic_info/symbol_map.csv";
const NB: usize = 5; // 0 / 1-2 / 3-5 / 6-15 / 16+

#[derive(Clone, Copy)]
struct Obs { imb: f64, r_mid: f64, price: f64, qup: f64 }

#[derive(Default, Clone, Copy)]
struct Sums { names: f64, obs: f64, ic: f64, ic2: f64, qup: f64 }

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
    let dir = format!("/ssd_data/stock/{}/market_data", date);
    let mut out = Vec::new();
    for e in std::fs::read_dir(&dir).unwrap_or_else(|e| panic!("read_dir: {}", e)) {
        let n = e.unwrap().file_name().to_string_lossy().to_string();
        if !n.ends_with("_market_data.csv") { continue; }
        let c = n.split('_').next().unwrap_or("").to_string();
        if wl.contains(&c) { out.push(c); }
    }
    out.sort();
    out
}

fn upd_bucket(q: f64) -> usize {
    if q == 0.0 { 0 } else if q <= 2.0 { 1 } else if q <= 5.0 { 2 } else if q <= 15.0 { 3 } else { 4 }
}

fn per_stock(recs: &[MarketRecord]) -> (Vec<Obs>, f64, f64) {
    let n = recs.len();
    let mut valid: Vec<usize> = Vec::with_capacity(n);
    for i in 0..n {
        let r = &recs[i];
        if r.last_prc > 0.0 && r.ask_prcs[0] > 0.0 && r.bid_prcs[0] > 0.0 && r.ask_prcs[0] > r.bid_prcs[0] { valid.push(i); }
    }
    let mut upd: Vec<i64> = Vec::new();
    for k in 1..valid.len() {
        let p = &recs[valid[k - 1]];
        let c = &recs[valid[k]];
        if p.ask_prcs[0] != c.ask_prcs[0] || p.bid_prcs[0] != c.bid_prcs[0] { upd.push(c.time_us); }
    }
    let day_turnover = recs.last().map(|r| r.turnover).unwrap_or(0.0);
    let mut out = Vec::new();
    let mut j = 0usize;
    let mut p = 0usize;
    let mut q = 0usize;
    for (k, &i) in valid.iter().enumerate() {
        if j < k { j = k; }
        let t = recs[i].time_us;
        let sod = (t / 1_000_000).rem_euclid(86400);
        if sod >= 11 * 3600 + 29 * 60 && sod < 11 * 3600 + 30 * 60 { continue; }
        while j < valid.len() && recs[valid[j]].time_us < t + FWD_US { j += 1; }
        if j >= valid.len() { break; }
        while p + 1 < k && recs[valid[p + 1]].time_us <= t - FWD_US { p += 1; }
        if p >= k || t - recs[valid[p]].time_us > MAX_GAP_US { continue; }
        while q < upd.len() && upd[q] <= t - FWD_US { q += 1; }
        let qup = (upd.len() - q) as f64;
        let r = &recs[i];
        let f = &recs[valid[j]];
        let bid5: f64 = r.bid_vols[..5].iter().sum();
        let ask5: f64 = r.ask_vols[..5].iter().sum();
        if bid5 + ask5 <= 0.0 || f.last_prc <= 0.0 { continue; }
        let mid = 0.5 * (r.ask_prcs[0] + r.bid_prcs[0]);
        let fmid = 0.5 * (f.ask_prcs[0] + f.bid_prcs[0]);
        if mid <= 0.0 || fmid <= 0.0 { continue; }
        out.push(Obs { imb: (bid5 - ask5) / (bid5 + ask5), r_mid: fmid / mid - 1.0, price: r.last_prc, qup });
    }
    (out, day_turnover, 0.0)
}

fn rank(v: &[f64]) -> Vec<f64> {
    let n = v.len();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| v[a].partial_cmp(&v[b]).unwrap_or(std::cmp::Ordering::Equal));
    let mut r = vec![0.0f64; n];
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j + 1 < n && v[idx[j + 1]] == v[idx[i]] { j += 1; }
        let avg = ((i + j) as f64) / 2.0 + 1.0;
        for k in i..=j { r[idx[k]] = avg; }
        i = j + 1;
    }
    r
}

fn spearman(x: &[f64], y: &[f64]) -> Option<f64> {
    let (rx, ry) = (rank(x), rank(y));
    let n = x.len() as f64;
    let mx = rx.iter().sum::<f64>() / n;
    let my = ry.iter().sum::<f64>() / n;
    let (mut sxy, mut sxx, mut syy) = (0.0, 0.0, 0.0);
    for i in 0..x.len() { let (dx, dy) = (rx[i] - mx, ry[i] - my); sxy += dx * dy; sxx += dx * dx; syy += dy * dy; }
    if sxx <= 0.0 || syy <= 0.0 { return None; }
    Some(sxy / (sxx.sqrt() * syy.sqrt()))
}

fn add(res: &mut [Sums], obs: &[Obs], labels: &[usize]) {
    for g in 0..res.len() {
        let idxs: Vec<usize> = (0..obs.len()).filter(|&k| labels[k] == g).collect();
        if idxs.len() < MIN_BUCKET { continue; }
        let x: Vec<f64> = idxs.iter().map(|&k| obs[k].imb).collect();
        let y: Vec<f64> = idxs.iter().map(|&k| obs[k].r_mid * 1e4).collect();
        let s = &mut res[g];
        s.names += 1.0;
        s.obs += idxs.len() as f64;
        if let Some(v) = spearman(&x, &y) { s.ic += v; s.ic2 += v * v; }
        for &k in &idxs { s.qup += obs[k].qup; }
    }
}

fn json_g(name: &str, s: &Sums, first: bool) -> String {
    let n = s.names.max(1.0);
    let m = s.ic / n;
    let var = (s.ic2 / n - m * m).max(0.0);
    let d = s.obs.max(1.0);
    format!("{}{{\"group\":\"{}\",\"stock_days\":{:.0},\"obs\":{:.0},\"ic\":{:.4},\"ic_se\":{:.4},\"mean_upd60\":{:.1}}}",
        if first { "" } else { "," }, name, s.names, s.obs, m, (var / n).sqrt(), s.qup / d)
}

fn run(date: i64) {
    let wl = read_symbol_map(SYMBOL_MAP);
    let codes = list_codes(date, &wl);
    let nt: usize = std::env::var("PLAY_THREADS").ok().and_then(|s| s.parse().ok()).unwrap_or(16);
    let next = AtomicUsize::new(0);
    let acc: Mutex<Vec<(Vec<Obs>, f64)>> = Mutex::new(Vec::new());
    let t0 = std::time::Instant::now();
    std::thread::scope(|sc| {
        for _ in 0..nt {
            sc.spawn(|| loop {
                let k = next.fetch_add(1, Ordering::Relaxed);
                if k >= codes.len() { break; }
                if let Ok(recs) = read_market_fast(&codes[k], date) {
                    let (o, tv, _) = per_stock(&recs);
                    if o.len() >= MIN_OBS { acc.lock().unwrap().push((o, tv)); }
                }
            });
        }
    });
    let data = acc.into_inner().unwrap();
    eprintln!("[{}] stock-days: {} in {:.1}s", date, data.len(), t0.elapsed().as_secs_f64());

    let mut res_all = vec![Sums::default(); NB];
    let mut res_px = vec![Sums::default(); 4 * NB];
    let mut upd_mean: Vec<f64> = Vec::new();
    let mut turn: Vec<f64> = Vec::new();
    let mut prc: Vec<f64> = Vec::new();
    for (obs, tv) in &data {
        let labels: Vec<usize> = obs.iter().map(|o| upd_bucket(o.qup)).collect();
        add(&mut res_all, obs, &labels);
        let mut prs: Vec<f64> = obs.iter().map(|o| o.price).collect();
        prs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mp = prs[prs.len() / 2];
        let gp = if mp < 5.0 { 0 } else if mp < 20.0 { 1 } else if mp < 100.0 { 2 } else { 3 };
        let two: Vec<usize> = labels.iter().map(|&v| gp * NB + v).collect();
        add(&mut res_px, obs, &two);
        upd_mean.push(obs.iter().map(|o| o.qup).sum::<f64>() / obs.len() as f64);
        turn.push((tv + 1.0).ln());
        prc.push(mp.ln());
    }
    let names = ["upd0", "upd1_2", "upd3_5", "upd6_15", "upd16p"];
    let px = ["px_lt5", "px_5to20", "px_20to100", "px_ge100"];
    let mut parts: Vec<String> = Vec::new();
    for i in 0..NB { parts.push(json_g(names[i], &res_all[i], parts.is_empty())); }
    for g in 0..4 { for b in 0..NB {
        parts.push(json_g(&format!("{}_{}", px[g], names[b]), &res_px[g * NB + b], false));
    } }
    let c_turn = spearman(&upd_mean, &turn).unwrap_or(f64::NAN);
    let c_prc = spearman(&upd_mean, &prc).unwrap_or(f64::NAN);
    println!("{{\"date\":{},\"stock_days\":{},\"spearman_upd_vs_turnover\":{:.3},\"spearman_upd_vs_price\":{:.3},\"groups\":[{}]}}",
        date, data.len(), c_turn, c_prc, parts.join(","));
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    assert!(!args.is_empty(), "usage: play_upd <date> [date...]");
    for a in &args { run(a.parse::<i64>().unwrap()); }
}
