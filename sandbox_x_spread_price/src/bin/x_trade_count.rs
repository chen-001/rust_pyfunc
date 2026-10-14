//! 对撞：过去 60 秒成交笔数 x 不平衡 -> 之后 60 秒收益（预期：笔数多的时点 IC 更高但尺度也更大）
//! 用法: XS_THREADS=10 x_trade_count 20220819 20241231
mod fast_csv_reader;

use fast_csv_reader::{read_market_fast, read_trade_fast, MarketRecord, TradeRecord};
use std::collections::HashSet;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

const FWD_US: i64 = 60_000_000;
const MAX_GAP_US: i64 = 95_000_000;
const MIN_OBS: usize = 200;
const MIN_BUCKET: usize = 30;
const SYMBOL_MAP: &str = "/ssd_data/data/basic_info/symbol_map.csv";

#[derive(Clone, Copy)]
struct Obs { imb: f64, r_mid: f64, cnt60: f64, price: f64 }

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
fn top_bottom(x: &[f64], y: &[f64]) -> Option<f64> {
    let n = x.len(); if n < 20 { return None; }
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| x[a].partial_cmp(&x[b]).unwrap_or(std::cmp::Ordering::Equal));
    let k = (n as f64 * 0.3).floor() as usize; if k == 0 { return None; }
    let lo: f64 = idx[..k].iter().map(|&i| y[i]).sum::<f64>() / k as f64;
    let hi: f64 = idx[n - k..].iter().map(|&i| y[i]).sum::<f64>() / k as f64;
    Some(hi - lo)
}

#[derive(Default, Clone, Copy)]
struct Acc { n: f64, ic: f64, ic2: f64, tb: f64, sig: f64, zero: f64 }

fn per_stock(mkt: &[MarketRecord], trd: &[TradeRecord]) -> Vec<Obs> {
    let mut trades: Vec<&TradeRecord> = trd.iter().filter(|t| t.price > 0.0 && t.volume > 0.0).collect();
    trades.sort_by_key(|t| t.time_us);
    let mut valid: Vec<usize> = Vec::new();
    for (i, r) in mkt.iter().enumerate() {
        if r.last_prc > 0.0 && r.ask_prcs[0] > 0.0 && r.bid_prcs[0] > 0.0 && r.ask_prcs[0] > r.bid_prcs[0] { valid.push(i); }
    }
    let mut out = Vec::new();
    if valid.is_empty() { return out; }
    let mut j = 0usize;
    let mut lo = 0usize;   // 过去 60 秒左端
    for (k, &i) in valid.iter().enumerate() {
        let r = &mkt[i];
        let t = r.time_us;
        let sod = (t / 1_000_000).rem_euclid(86400);
        if sod >= 11 * 3600 + 29 * 60 && sod < 11 * 3600 + 30 * 60 { continue; }
        if lo > k { lo = k; }
        // 过去 60 秒窗口内的逐笔索引区间 [a, b)
        while lo < trades.len() && trades[lo].time_us < t - FWD_US { lo += 1; }
        let mut hi = lo;
        while hi < trades.len() && trades[hi].time_us <= t { hi += 1; }
        let cnt60 = (hi - lo) as f64;
        if j < k { j = k; }
        while j < valid.len() && mkt[valid[j]].time_us < t + FWD_US { j += 1; }
        if j >= valid.len() { break; }
        let fj = valid[j];
        if mkt[fj].time_us - t > MAX_GAP_US { continue; }
        let f = &mkt[fj];
        let mid = 0.5 * (r.ask_prcs[0] + r.bid_prcs[0]);
        let fmid = 0.5 * (f.ask_prcs[0] + f.bid_prcs[0]);
        let bid5: f64 = r.bid_vols[..5].iter().sum();
        let ask5: f64 = r.ask_vols[..5].iter().sum();
        if bid5 + ask5 <= 0.0 { continue; }
        out.push(Obs {
            imb: (bid5 - ask5) / (bid5 + ask5),
            r_mid: (fmid / mid - 1.0) * 1e4,
            cnt60,
            price: r.last_prc,
        });
    }
    out
}

fn run(date: i64) {
    let wl = read_symbol_map(SYMBOL_MAP);
    let codes = list_codes(date, &wl);
    let nt: usize = std::env::var("XS_THREADS").ok().and_then(|s| s.parse().ok()).unwrap_or(10);
    let next = AtomicUsize::new(0);
    let collected: Mutex<Vec<(String, Vec<Obs>)>> = Mutex::new(Vec::new());
    std::thread::scope(|scope| {
        for _ in 0..nt { scope.spawn(|| loop {
            let k = next.fetch_add(1, Ordering::Relaxed);
            if k >= codes.len() { break; }
            let code = &codes[k];
            if let (Ok(mkt), Ok(trd)) = (read_market_fast(code, date), read_trade_fast(code, date)) {
                let o = per_stock(&mkt, &trd);
                if o.len() >= MIN_OBS { collected.lock().unwrap().push((code.clone(), o)); }
            }
        }); }
    });
    let mut data = collected.into_inner().unwrap();
    data.sort_by(|a, b| a.0.cmp(&b.0));
    eprintln!("[{}] stock-days: {}", date, data.len());

    // 每只股票日内按成交笔数三档（低/中/高），并记录档内边界值与不同取值个数
    let mut acc = vec![Acc::default(); 3];
    let mut bounds: Vec<(f64, f64)> = Vec::new();
    for (_c, obs) in &data {
        let mut v: Vec<f64> = obs.iter().map(|o| o.cnt60).collect();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let c1 = v[v.len() / 3]; let c2 = v[v.len() * 2 / 3];
        bounds.push((c1, c2));
        let lab: Vec<usize> = obs.iter().map(|o| if o.cnt60 <= c1 { 0 } else if o.cnt60 <= c2 { 1 } else { 2 }).collect();
        for g in 0..3 {
            let idxs: Vec<usize> = (0..obs.len()).filter(|&k| lab[k] == g).collect();
            if idxs.len() < MIN_BUCKET { continue; }
            let x: Vec<f64> = idxs.iter().map(|&k| obs[k].imb).collect();
            let y: Vec<f64> = idxs.iter().map(|&k| obs[k].r_mid).collect();
            let sd = (y.iter().map(|v| v * v).sum::<f64>() / y.len() as f64).sqrt();
            let a = &mut acc[g];
            a.n += 1.0;
            if let Some(v2) = spearman(&x, &y) { a.ic += v2; a.ic2 += v2 * v2; }
            if let Some(v2) = top_bottom(&x, &y) { a.tb += v2; }
            a.sig += sd;
            a.zero += y.iter().filter(|&&v| v == 0.0).count() as f64 / y.len() as f64;
        }
    }
    let names = ["cnt_low", "cnt_mid", "cnt_high"];
    let mut parts: Vec<String> = Vec::new();
    for g in 0..3 {
        let a = &acc[g]; let n = a.n.max(1.0);
        let m = a.ic / n; let se = ((a.ic2 / n - m * m).max(0.0) / n).sqrt();
        let sig = a.sig / n;
        parts.push(format!("{{\"name\":\"{}\",\"stock_days\":{:.0},\"ic_mid\":{:.4},\"ic_mid_se\":{:.4},\"tb_mid_bp\":{:.3},\"ret_scale_bp\":{:.2},\"tb_over_scale\":{:.4},\"zero_ret_share\":{:.3}}}",
            names[g], a.n, m, se, a.tb / n, sig, (a.tb / n) / sig.max(1e-9), a.zero / n));
    }
    // 档边界与并列程度
    let n = bounds.len() as f64;
    let m1 = bounds.iter().map(|b| b.0).sum::<f64>() / n;
    let m2 = bounds.iter().map(|b| b.1).sum::<f64>() / n;
    parts.push(format!("{{\"name\":\"cut_avg\",\"c1\":{:.2},\"c2\":{:.2}}}", m1, m2));
    // 全样本重值程度：cnt60 的不同取值个数占比（每股票日）
    let mut dup_ratio = 0.0f64; let mut dd = 0.0f64;
    let mut zero_share = 0.0f64;
    for (_c, obs) in &data {
        let mut v: Vec<i64> = obs.iter().map(|o| o.cnt60 as i64).collect();
        v.sort_unstable(); let total = v.len() as f64; v.dedup();
        dup_ratio += v.len() as f64 / total; dd += 1.0;
        zero_share += obs.iter().filter(|o| o.cnt60 == 0.0).count() as f64 / total;
    }
    parts.push(format!("{{\"name\":\"distinct_ratio\",\"value\":{:.4}}}", dup_ratio / dd.max(1.0)));
    parts.push(format!("{{\"name\":\"cnt_zero_share\",\"value\":{:.4}}}", zero_share / dd.max(1.0)));
    println!("{{\"date\":{},\"stock_days\":{},\"groups\":[{}]}}", date, data.len(), parts.join(","));
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    assert!(!args.is_empty(), "usage: x_trade_count <date> [date...]");
    for a in &args { run(a.parse::<i64>().expect("date")); }
}
