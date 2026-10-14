//! sandbox 探索：不平衡预测力在「逐点波动归一」和「股票日常数归一」下差多少
//! （秩相关对股票日内的常数缩放不变，所以两者必须分开看）
//! 构建: cd sandbox_play_spread && ~/.cargo/bin/cargo build --release --bin play_vol
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
const VOL_FLOOR: f64 = 1e-4;

#[derive(Clone, Copy)]
struct Obs { imb: f64, r_mid: f64, vol_trail: f64, price: f64, day_turnover: f64 }

#[derive(Default, Clone, Copy)]
struct Sums {
    names: f64, obs: f64,
    ic: f64, ic2: f64,          // 原始
    icn: f64, icn2: f64,        // 除以逐点过去 60 秒波动
    icd: f64, icd2: f64,        // 除以股票日波动中位数
    vol: f64, absr: f64, zero_frac: f64,
}

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
    let day_turnover = recs.last().map(|r| r.turnover).unwrap_or(0.0);
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
            day_turnover,
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
    let mut sxy = 0.0; let mut sxx = 0.0; let mut syy = 0.0;
    for i in 0..x.len() { let (dx, dy) = (rx[i] - mx, ry[i] - my); sxy += dx * dy; sxx += dx * dx; syy += dy * dy; }
    if sxx <= 0.0 || syy <= 0.0 { return None; }
    Some(sxy / (sxx.sqrt() * syy.sqrt()))
}

fn add(res: &mut [Sums], obs: &[Obs], labels: &[usize], sigma_day: f64) {
    for g in 0..res.len() {
        let idxs: Vec<usize> = (0..obs.len()).filter(|&k| labels[k] == g).collect();
        if idxs.len() < MIN_BUCKET { continue; }
        let x: Vec<f64> = idxs.iter().map(|&k| obs[k].imb).collect();
        let y: Vec<f64> = idxs.iter().map(|&k| obs[k].r_mid * 1e4).collect();
        let yn: Vec<f64> = idxs.iter().map(|&k| obs[k].r_mid / obs[k].vol_trail.max(VOL_FLOOR)).collect();
        let yd: Vec<f64> = idxs.iter().map(|&k| obs[k].r_mid / sigma_day.max(VOL_FLOOR)).collect();
        let s = &mut res[g];
        s.names += 1.0;
        s.obs += idxs.len() as f64;
        if let Some(v) = spearman(&x, &y) { s.ic += v; s.ic2 += v * v; }
        if let Some(v) = spearman(&x, &yn) { s.icn += v; s.icn2 += v * v; }
        if let Some(v) = spearman(&x, &yd) { s.icd += v; s.icd2 += v * v; }
        for &k in &idxs {
            s.vol += obs[k].vol_trail * 1e4;
            s.absr += obs[k].r_mid.abs() * 1e4;
            if obs[k].vol_trail == 0.0 { s.zero_frac += 1.0; }
        }
    }
}

fn stats(sum: f64, sum2: f64, n: f64) -> (f64, f64) {
    let m = sum / n.max(1.0);
    let v = (sum2 / n.max(1.0) - m * m).max(0.0);
    (m, (v / n.max(1.0)).sqrt())
}

fn json_g(name: &str, s: &Sums, first: bool) -> String {
    let (ic, icse) = stats(s.ic, s.ic2, s.names);
    let (icn, icnse) = stats(s.icn, s.icn2, s.names);
    let (icd, icdse) = stats(s.icd, s.icd2, s.names);
    let d = s.obs.max(1.0);
    format!("{}{{\"group\":\"{}\",\"stock_days\":{:.0},\"obs\":{:.0},\"ic_raw\":{:.4},\"ic_raw_se\":{:.4},\"ic_norm_trail\":{:.4},\"ic_norm_trail_se\":{:.4},\"ic_norm_day\":{:.4},\"ic_norm_day_se\":{:.4},\"trail_vol_bp\":{:.2},\"absr_bp\":{:.2},\"zero_vol_frac\":{:.3}}}",
        if first { "" } else { "," }, name, s.names, s.obs, ic, icse, icn, icnse, icd, icdse, s.vol / d, s.absr / d, s.zero_frac / d)
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

    let mut res_px = vec![Sums::default(); 4];
    let mut res_all = vec![Sums::default(); 1];
    let mut res_q = vec![Sums::default(); 8];   // 价格桶 x (过去 60 秒中间价没动 / 动过)
    let mut res_qa = vec![Sums::default(); 2];  // 全体 x 静/动
    let mut res_pt = vec![Sums::default(); 12]; // 价格桶 x 成交额三分位（股票日层面）
    let mut tvals: Vec<f64> = data.iter().map(|o| o[0].day_turnover.ln()).collect();
    tvals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let tq1 = tvals[tvals.len() / 3];
    let tq2 = tvals[2 * tvals.len() / 3];
    for obs in &data {
        let mut vs: Vec<f64> = obs.iter().map(|o| o.vol_trail).collect();
        vs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let sigma_day = vs[vs.len() / 2];
        let mut prs: Vec<f64> = obs.iter().map(|o| o.price).collect();
        prs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mp = prs[prs.len() / 2];
        let gp = if mp < 5.0 { 0 } else if mp < 20.0 { 1 } else if mp < 100.0 { 2 } else { 3 };
        add(&mut res_px, obs, &vec![gp; obs.len()], sigma_day);
        add(&mut res_all, obs, &vec![0usize; obs.len()], sigma_day);
        let q: Vec<usize> = obs.iter().map(|o| gp * 2 + if o.vol_trail == 0.0 { 0 } else { 1 }).collect();
        add(&mut res_q, obs, &q, sigma_day);
        let qa: Vec<usize> = obs.iter().map(|o| if o.vol_trail == 0.0 { 0 } else { 1 }).collect();
        add(&mut res_qa, obs, &qa, sigma_day);
        let tn = obs[0].day_turnover.ln();
        let tb = if tn <= tq1 { 0 } else if tn <= tq2 { 1 } else { 2 };
        add(&mut res_pt, obs, &vec![gp * 3 + tb; obs.len()], sigma_day);
    }
    let px_names = ["px_lt5", "px_5to20", "px_20to100", "px_ge100"];
    let mut parts: Vec<String> = Vec::new();
    for i in 0..4 { parts.push(json_g(px_names[i], &res_px[i], parts.is_empty())); }
    parts.push(json_g("all", &res_all[0], false));
    for i in 0..4 { for j in 0..2 {
        parts.push(json_g(&format!("{}_{}", px_names[i], if j == 0 { "quiet" } else { "active" }), &res_q[i * 2 + j], false));
    } }
    parts.push(json_g("all_quiet", &res_qa[0], false));
    parts.push(json_g("all_active", &res_qa[1], false));
    for i in 0..4 { for j in 0..3 {
        parts.push(json_g(&format!("{}_tv{}", px_names[i], j + 1), &res_pt[i * 3 + j], false));
    } }
    println!("{{\"date\":{},\"stock_days\":{},\"groups\":[{}]}}", date, data.len(), parts.join(","));
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    assert!(!args.is_empty(), "usage: play_vol <date> [date...]");
    for a in &args { run(a.parse::<i64>().unwrap()); }
}
