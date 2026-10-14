//! 只做一件事：100 元以上桶，按股票日切波动三档（每档只切一次），再和六个时段窗交叉
//! 构建: cargo build --release --bin play_oct
//! 运行: PLAY_THREADS=16 ./target/release/play_oct 20220819 20241231
mod fast_csv_reader;

use fast_csv_reader::{read_market_fast, MarketRecord};
use std::collections::HashSet;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

const FWD_US: i64 = 60_000_000;
const MAX_GAP_US: i64 = 95_000_000;
const MIN_OBS_STOCKDAY: usize = 200;   // 股票日总观测下限
const MIN_OBS_CELL: usize = 10;         // 单格算 IC 的观测下限（不满足则该股票日不参与该格，计数报出来）
const MIN_STOCKDAYS: usize = 15;        // 判定用的每格股票日下限
const BASE: i64 = 9 * 3600 + 30 * 60;
const WIN: i64 = 40 * 60;
const NWIN: usize = 6;
const SYMBOL_MAP: &str = "/ssd_data/data/basic_info/symbol_map.csv";

#[derive(Clone, Copy)]
struct Obs { sod: i64, imb: f64, r_mid: f64, vol_trail: f64 }

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

fn per_stock(recs: &[MarketRecord]) -> (f64, Vec<Obs>) {
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
            sod,
            imb: (bid5 - ask5) / (bid5 + ask5),
            r_mid: fmid / mid - 1.0,
            vol_trail: (mid / bmid - 1.0).abs(),
        });
    }
    let mut prs: Vec<f64> = valid.iter().map(|&i| recs[i].last_prc).collect();
    prs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med_px = if prs.is_empty() { 0.0 } else { prs[prs.len() / 2] };
    (med_px, out)
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
    Some(sxy / (sxx * syy).sqrt())
}

fn run(date: i64) {
    let wl = read_symbol_map(SYMBOL_MAP);
    let codes = list_codes(date, &wl);
    let nt: usize = std::env::var("PLAY_THREADS").ok().and_then(|s| s.parse().ok()).unwrap_or(16);
    let next = AtomicUsize::new(0);
    let acc: Mutex<Vec<(f64, Vec<Obs>)>> = Mutex::new(Vec::new());
    let t0 = std::time::Instant::now();
    std::thread::scope(|sc| {
        for _ in 0..nt {
            sc.spawn(|| loop {
                let k = next.fetch_add(1, Ordering::Relaxed);
                if k >= codes.len() { break; }
                if let Ok(recs) = read_market_fast(&codes[k], date) {
                    let (px, o) = per_stock(&recs);
                    if o.len() >= MIN_OBS_STOCKDAY { acc.lock().unwrap().push((px, o)); }
                }
            });
        }
    });
    let all = acc.into_inner().unwrap();
    eprintln!("[{}] stock-days: {} in {:.1}s", date, all.len(), t0.elapsed().as_secs_f64());

    // 只留 100 元以上
    let hi: Vec<&(f64, Vec<Obs>)> = all.iter().filter(|(px, _)| *px >= 100.0).collect();
    // 每个股票日的波动水平：该日全部观测的中间价过去60秒绝对变动的中位数
    let mut day_vol: Vec<(f64, f64, &Vec<Obs>)> = hi.iter().map(|(px, o)| {
        let mut v: Vec<f64> = o.iter().map(|x| x.vol_trail).collect();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        (v[v.len() / 2], *px, o)
    }).collect();
    let mut vs: Vec<f64> = day_vol.iter().map(|x| x.0).collect();
    vs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let q1 = vs[vs.len() / 3];
    let q2 = vs[2 * vs.len() / 3];
    for d in day_vol.iter_mut() { d.0 = if d.0 <= q1 { 0.0 } else if d.0 <= q2 { 1.0 } else { 2.0 }; }
    let tercile_sizes = [day_vol.iter().filter(|d| d.0 == 0.0).count(), day_vol.iter().filter(|d| d.0 == 1.0).count(), day_vol.iter().filter(|d| d.0 == 2.0).count()];

    // 累加：档 x 窗
    let mut ic_sum = [[0.0f64; NWIN]; 3];
    let mut ic_sq = [[0.0f64; NWIN]; 3];
    let mut n_day = [[0.0f64; NWIN]; 3];
    let mut n_obs = [[0.0f64; NWIN]; 3];
    let mut n_day_excl = [[0.0f64; NWIN]; 3];   // 该格观测数不到 MIN_OBS_CELL 而被排除的股票日
    for (tier, _px, obs) in day_vol.iter() {
        let ti = *tier as usize;
        for w in 0..NWIN {
            let sel: Vec<&Obs> = obs.iter().filter(|o| {
                let k = (o.sod - BASE) / WIN;
                k >= 0 && k == w as i64 && o.sod >= BASE && o.sod < BASE + (NWIN as i64) * WIN
            }).collect();
            if sel.len() < MIN_OBS_CELL { n_day_excl[ti][w] += 1.0; continue; }
            let x: Vec<f64> = sel.iter().map(|o| o.imb).collect();
            let y: Vec<f64> = sel.iter().map(|o| o.r_mid * 1e4).collect();
            if let Some(v) = spearman(&x, &y) {
                ic_sum[ti][w] += v; ic_sq[ti][w] += v * v; n_day[ti][w] += 1.0; n_obs[ti][w] += sel.len() as f64;
            }
        }
    }
    let names = ["low", "mid", "high"];
    let mut parts: Vec<String> = Vec::new();
    for ti in 0..3 {
        for w in 0..NWIN {
            let n = n_day[ti][w];
            let (ic, se) = if n > 0.0 { let m = ic_sum[ti][w] / n; let var = (ic_sq[ti][w] / n - m * m).max(0.0); (m, (var / n).sqrt()) } else { (f64::NAN, f64::NAN) };
            let small = if n < MIN_STOCKDAYS as f64 { "格太小" } else { "" };
            parts.push(format!("{{\"vol\":\"{}\",\"win\":{},\"ic\":{:.4},\"ic_se\":{:.4},\"stock_days\":{:.0},\"obs\":{:.0},\"stock_days_excluded_small_cell\":{:.0},\"flag\":\"{}\"}}",
                names[ti], w + 1, ic, se, n, n_obs[ti][w], n_day_excl[ti][w], small));
        }
    }
    println!("{{\"date\":{},\"stock_days_gt100px\":{},\"tercile_sizes\":[{},{},{}],\"thresholds\":{{\"min_obs_cell\":{},\"min_stockdays\":{}}},\"cells\":[{}]}}",
        date, day_vol.len(), tercile_sizes[0], tercile_sizes[1], tercile_sizes[2], MIN_OBS_CELL, MIN_STOCKDAYS, parts.join(","));
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    assert!(!args.is_empty(), "usage: play_oct <date> [date...]");
    for a in &args { run(a.parse::<i64>().unwrap()); }
}
