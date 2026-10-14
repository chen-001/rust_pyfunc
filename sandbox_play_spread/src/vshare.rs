//! 低价股「报价没动」的时点：按快照数算还是按量算，差多少；以及按过去 60 秒成交量分层
//! 构建: cargo build --release --bin play_vshare
//! 运行: PLAY_THREADS=16 ./target/release/play_vshare 20220819 20241231
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

#[derive(Clone, Copy)]
struct Obs { imb: f64, r_mid: f64, vol_trail: f64, price: f64, fwd_vol: f64, trail_vol: f64 }

#[derive(Default, Clone, Copy)]
struct Sums { names: f64, obs: f64, ic: f64, ic2: f64, tvol: f64 }

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

fn per_stock(recs: &[MarketRecord]) -> Vec<Obs> {
    let n = recs.len();
    let mut valid: Vec<usize> = Vec::with_capacity(n);
    for i in 0..n {
        let r = &recs[i];
        if r.last_prc > 0.0 && r.ask_prcs[0] > 0.0 && r.bid_prcs[0] > 0.0 && r.ask_prcs[0] > r.bid_prcs[0] { valid.push(i); }
    }
    let mut out = Vec::new();
    let mut j = 0usize;
    let mut p = 0usize;
    for (k, &i) in valid.iter().enumerate() {
        if j < k { j = k; }
        let t = recs[i].time_us;
        let sod = (t / 1_000_000).rem_euclid(86400);
        if sod >= 11 * 3600 + 29 * 60 && sod < 11 * 3600 + 30 * 60 { continue; }
        while j < valid.len() && recs[valid[j]].time_us < t + FWD_US { j += 1; }
        if j >= valid.len() { break; }
        while p + 1 < k && recs[valid[p + 1]].time_us <= t - FWD_US { p += 1; }
        if p >= k || t - recs[valid[p]].time_us > MAX_GAP_US { continue; }
        let r = &recs[i];
        let f = &recs[valid[j]];
        let b = &recs[valid[p]];
        let bid5: f64 = r.bid_vols[..5].iter().sum();
        let ask5: f64 = r.ask_vols[..5].iter().sum();
        if bid5 + ask5 <= 0.0 || f.last_prc <= 0.0 || b.last_prc <= 0.0 { continue; }
        let mid = 0.5 * (r.ask_prcs[0] + r.bid_prcs[0]);
        let fmid = 0.5 * (f.ask_prcs[0] + f.bid_prcs[0]);
        let bmid = 0.5 * (b.ask_prcs[0] + b.bid_prcs[0]);
        if mid <= 0.0 || fmid <= 0.0 || bmid <= 0.0 { continue; }
        out.push(Obs {
            imb: (bid5 - ask5) / (bid5 + ask5),
            r_mid: fmid / mid - 1.0,
            vol_trail: (mid / bmid - 1.0).abs(),
            price: r.last_prc,
            fwd_vol: (f.volume - r.volume).max(0.0),
            trail_vol: (r.volume - b.volume).max(0.0),
        });
    }
    out
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

fn terciles(vals: &[f64]) -> Vec<usize> {
    let mut s: Vec<f64> = vals.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let q1 = s[s.len() / 3];
    let q2 = s[2 * s.len() / 3];
    vals.iter().map(|&v| if v <= q1 { 0 } else if v <= q2 { 1 } else { 2 }).collect()
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
        for &k in &idxs { s.tvol += obs[k].trail_vol; }
    }
}

fn run(date: i64) {
    let wl = read_symbol_map(SYMBOL_MAP);
    let codes = list_codes(date, &wl);
    let nt: usize = std::env::var("PLAY_THREADS").ok().and_then(|s| s.parse().ok()).unwrap_or(16);
    let next = AtomicUsize::new(0);
    let acc: Mutex<Vec<Vec<Obs>>> = Mutex::new(Vec::new());
    let t0 = std::time::Instant::now();
    std::thread::scope(|sc| {
        for _ in 0..nt {
            sc.spawn(|| loop {
                let k = next.fetch_add(1, Ordering::Relaxed);
                if k >= codes.len() { break; }
                if let Ok(recs) = read_market_fast(&codes[k], date) {
                    let o = per_stock(&recs);
                    if o.len() >= MIN_OBS { acc.lock().unwrap().push(o); }
                }
            });
        }
    });
    let data = acc.into_inner().unwrap();
    eprintln!("[{}] stock-days: {} in {:.1}s", date, data.len(), t0.elapsed().as_secs_f64());

    // 价格桶统计 + 过去成交量的三分位 IC
    let mut stat = vec![[0.0f64; 7]; 4];   // obs, quiet_obs, trail_all, trail_quiet, fwd_all, fwd_quiet, zero_trail
    let mut res = vec![Sums::default(); 12];
    let mut res2 = vec![Sums::default(); 24];  // 价格桶 x (过去60秒中间价静/动) x 到达量三档
    for obs in &data {
        let mut prs: Vec<f64> = obs.iter().map(|o| o.price).collect();
        prs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mp = prs[prs.len() / 2];
        let gp = if mp < 5.0 { 0 } else if mp < 20.0 { 1 } else if mp < 100.0 { 2 } else { 3 };
        for o in obs.iter() {
            let s = &mut stat[gp];
            s[0] += 1.0;
            s[2] += o.trail_vol;
            s[4] += o.fwd_vol;
            if o.vol_trail == 0.0 { s[1] += 1.0; s[3] += o.trail_vol; s[5] += o.fwd_vol; }
            if o.trail_vol == 0.0 { s[6] += 1.0; }
        }
        let tv: Vec<f64> = obs.iter().map(|o| o.trail_vol).collect();
        let lab = terciles(&tv);
        let labels: Vec<usize> = lab.iter().map(|&v| gp * 3 + v).collect();
        add(&mut res, obs, &labels);
        let labels2: Vec<usize> = lab.iter().enumerate()
            .map(|(k, &v)| gp * 6 + (if obs[k].vol_trail == 0.0 { 0 } else { 1 }) * 3 + v)
            .collect();
        add(&mut res2, obs, &labels2);
    }
    let px = ["px_lt5", "px_5to20", "px_20to100", "px_ge100"];
    let mut parts: Vec<String> = Vec::new();
    for g in 0..4 {
        let s = stat[g];
        let obs = s[0].max(1.0);
        parts.push(format!("{{\"group\":\"{}\",\"obs\":{:.0},\"quiet_frac\":{:.4},\"zero_trail_vol_frac\":{:.4},\"quiet_trail_vol_share\":{:.4},\"quiet_fwd_vol_share\":{:.4}}}",
            px[g], s[0], s[1] / obs, s[6] / obs, s[3] / s[2].max(1.0), s[5] / s[4].max(1.0)));
    }
    for g in 0..12 {
        let s = &res[g];
        let n = s.names.max(1.0);
        let m = s.ic / n;
        let var = (s.ic2 / n - m * m).max(0.0);
        let d = s.obs.max(1.0);
        parts.push(format!("{{\"group\":\"{}_v{}\",\"stock_days\":{:.0},\"obs\":{:.0},\"ic\":{:.4},\"ic_se\":{:.4},\"trail_vol_lot\":{:.0}}}",
            px[g / 3], g % 3 + 1, s.names, s.obs, m, (var / n).sqrt(), s.tvol / d));
    }
    for g in 0..4 { for z in 0..2 { for v in 0..3 {
        parts.push(format!("{{\"group\":\"{}_{}_v{}\",\"stock_days\":{:.0},\"obs\":{:.0},\"ic\":{:.4},\"ic_se\":{:.4}}}",
            px[g], if z == 0 { "quiet" } else { "active" }, v + 1,
            res2[g * 6 + z * 3 + v].names, res2[g * 6 + z * 3 + v].obs,
            res2[g * 6 + z * 3 + v].ic / res2[g * 6 + z * 3 + v].names.max(1.0),
            (res2[g * 6 + z * 3 + v].ic2 / res2[g * 6 + z * 3 + v].names.max(1.0)).sqrt().max(0.0)));
    } } }
    println!("{{\"date\":{},\"stock_days\":{},\"groups\":[{}]}}", date, data.len(), parts.join(","));
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    assert!(!args.is_empty(), "usage: play_vshare <date> [date...]");
    for a in &args { run(a.parse::<i64>().unwrap()); }
}
