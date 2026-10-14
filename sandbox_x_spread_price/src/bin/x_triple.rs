//! 三个候选条件（过去 60 秒成交额 / 成交笔数 / 中间价波动）互相控制：谁在解释多空收益、谁只解释尺度
//! 用法: XS_THREADS=12 x_triple 20220819 [date...]
mod fast_csv_reader;

use fast_csv_reader::{read_market_fast, read_trade_fast, MarketRecord, TradeRecord};
use std::collections::HashSet;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

const FWD_US: i64 = 60_000_000;
const MAX_GAP_US: i64 = 95_000_000;
const MIN_OBS: usize = 200;
const MIN_BUCKET: usize = 20;
const SYMBOL_MAP: &str = "/ssd_data/data/basic_info/symbol_map.csv";

#[derive(Clone, Copy)]
struct Obs { imb: f64, r_mid: f64, amt60: f64, cnt60: f64, pv: f64, price: f64 }

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
/// 每只股票日内按 v 值三等分，返回每档的 (股票日数, 观测数, IC 均值, IC 平方和, 多空均值, 尺度均值, amt, cnt, pv)
#[derive(Default, Clone, Copy)]
struct Cell { n: f64, obs: f64, ic: f64, ic2: f64, tb: f64, sig: f64, amt: f64, cnt: f64, pv: f64 }

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

    // 三档 x 三档 x 三档：成交额 / 笔数 / 波动
    let mut cell = vec![Cell::default(); 27];
    // 观测级的三个量之间的秩相关（每股票日一次再平均）
    let (mut c1, mut c2, mut c3, mut cn) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for (_c, obs) in &data {
        let third = |key: fn(&Obs) -> f64, i: usize| -> Vec<f64> {
            let mut v: Vec<f64> = obs.iter().map(key).collect();
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let c1 = v[v.len() / 3]; let c2 = v[v.len() * 2 / 3];
            obs.iter().map(|o| { let x = key(o); if i == 0 { x } else { x } }).map(|x| if x <= c1 { 0.0 } else if x <= c2 { 1.0 } else { 2.0 }).collect()
        };
        let la = third(|o| o.amt60, 0);
        let lc = third(|o| o.cnt60, 0);
        let lp = third(|o| o.pv, 0);
        let av: Vec<f64> = obs.iter().map(|o| o.amt60).collect();
        let cv: Vec<f64> = obs.iter().map(|o| o.cnt60).collect();
        let pv: Vec<f64> = obs.iter().map(|o| o.pv).collect();
        if let Some(v) = spearman(&av, &cv) { c1 += v; }
        if let Some(v) = spearman(&av, &pv) { c2 += v; }
        if let Some(v) = spearman(&cv, &pv) { c3 += v; }
        cn += 1.0;
        for g in 0..27 {
            let ga = g / 9; let gc = (g / 3) % 3; let gp = g % 3;
            let idxs: Vec<usize> = (0..obs.len()).filter(|&k| la[k] as usize == ga && lc[k] as usize == gc && lp[k] as usize == gp).collect();
            if idxs.len() < MIN_BUCKET { continue; }
            let x: Vec<f64> = idxs.iter().map(|&k| obs[k].imb).collect();
            let y: Vec<f64> = idxs.iter().map(|&k| obs[k].r_mid).collect();
            let sd = (y.iter().map(|v| v * v).sum::<f64>() / y.len() as f64).sqrt();
            let s = &mut cell[g];
            s.n += 1.0; s.obs += idxs.len() as f64;
            if let Some(v) = spearman(&x, &y) { s.ic += v; s.ic2 += v * v; }
            if let Some(v) = top_bottom(&x, &y) { s.tb += v; }
            s.sig += sd;
            s.amt += idxs.iter().map(|&k| obs[k].amt60).sum::<f64>() / idxs.len() as f64;
            s.cnt += idxs.iter().map(|&k| obs[k].cnt60).sum::<f64>() / idxs.len() as f64;
            s.pv += idxs.iter().map(|&k| obs[k].pv).sum::<f64>() / idxs.len() as f64;
        }
    }
    let mut parts: Vec<String> = Vec::new();
    parts.push(format!("{{\"name\":\"corr_amt_cnt\",\"value\":{:.4}}}", c1 / cn.max(1.0)));
    parts.push(format!("{{\"name\":\"corr_amt_pv\",\"value\":{:.4}}}", c2 / cn.max(1.0)));
    parts.push(format!("{{\"name\":\"corr_cnt_pv\",\"value\":{:.4}}}", c3 / cn.max(1.0)));
    for g in 0..27 {
        let s = &cell[g];
        if s.n < 1.0 { continue; }
        let n = s.n;
        let m = s.ic / n;
        let se = ((s.ic2 / n - m * m).max(0.0) / n).sqrt();
        let sig = s.sig / n;
        parts.push(format!("{{\"name\":\"a{}c{}p{}\",\"stock_days\":{:.0},\"obs\":{:.0},\"ic_mid\":{:.4},\"ic_mid_se\":{:.4},\"tb_mid_bp\":{:.3},\"ret_scale_bp\":{:.2},\"tb_over_scale\":{:.4},\"amt60\":{:.0},\"cnt60\":{:.1},\"pv_bp\":{:.2}}}",
            g / 9, (g / 3) % 3, g % 3, n, s.obs, m, se, s.tb / n, sig, (s.tb / n) / sig.max(1e-9), s.amt / n, s.cnt / n, s.pv / n));
    }
    println!("{{\"date\":{},\"stock_days\":{},\"cells\":[{}]}}", date, data.len(), parts.join(","));
}

fn per_stock(mkt: &[MarketRecord], trd: &[TradeRecord]) -> Vec<Obs> {
    let mut trades: Vec<&TradeRecord> = trd.iter().filter(|t| t.price > 0.0 && t.volume > 0.0).collect();
    trades.sort_by_key(|t| t.time_us);
    // 成交额前缀和
    let mut cum = vec![0.0f64; trades.len() + 1];
    for (i, t) in trades.iter().enumerate() { cum[i + 1] = cum[i] + t.turnover; }
    let mut valid: Vec<usize> = Vec::new();
    for (i, r) in mkt.iter().enumerate() {
        if r.last_prc > 0.0 && r.ask_prcs[0] > 0.0 && r.bid_prcs[0] > 0.0 && r.ask_prcs[0] > r.bid_prcs[0] { valid.push(i); }
    }
    let mut out = Vec::new();
    if valid.is_empty() { return out; }
    let mut j = 0usize;
    let mut a = 0usize; let mut b = 0usize;
    for (k, &i) in valid.iter().enumerate() {
        let r = &mkt[i];
        let t = r.time_us;
        let sod = (t / 1_000_000).rem_euclid(86400);
        if sod >= 11 * 3600 + 29 * 60 && sod < 11 * 3600 + 30 * 60 { continue; }
        while a < trades.len() && trades[a].time_us < t - FWD_US { a += 1; }
        if b < a { b = a; }
        while b < trades.len() && trades[b].time_us <= t { b += 1; }
        let amt60 = cum[b] - cum[a];
        let cnt60 = (b - a) as f64;
        // 过去 60 秒中间价收益的标准差（逐点）
        let pstart = if k > 3 { k - 4 } else { 0 };
        let mut s = 0.0; let mut s2 = 0.0; let mut cnt = 0.0; let mut prev = f64::NAN;
        for w in pstart..=k {
            let m = 0.5 * (mkt[valid[w]].ask_prcs[0] + mkt[valid[w]].bid_prcs[0]);
            if m <= 0.0 { continue; }
            if prev.is_finite() { let x = (m / prev - 1.0) * 1e4; s += x; s2 += x * x; cnt += 1.0; }
            prev = m;
        }
        let pv = if cnt >= 3.0 { let mu = s / cnt; (((s2 / cnt - mu * mu).max(0.0)).sqrt()) } else { 0.0 };
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
        out.push(Obs { imb: (bid5 - ask5) / (bid5 + ask5), r_mid: (fmid / mid - 1.0) * 1e4, amt60, cnt60, pv, price: r.last_prc });
    }
    out
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    assert!(!args.is_empty(), "usage: x_triple <date> [date...]");
    for a in &args { run(a.parse::<i64>().expect("date")); }
}
