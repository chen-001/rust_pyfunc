//! sandbox 探索：撤单寿命 vs 成交等待（用编号推时间），单日全市场
//! 构建: cd sandbox_play_spread && ~/.cargo/bin/cargo build --release
//! 运行: PLAY_THREADS=16 ./target/release/play_cancel 20220819 20241231
mod fast_csv_reader;

use fast_csv_reader::{read_trade_fast_retreat, TradeRecord};
use std::collections::HashSet;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

const MIN_TRADES: usize = 1000;
const SYMBOL_MAP: &str = "/ssd_data/data/basic_info/symbol_map.csv";
const SESSION_START: i64 = 9 * 3600 + 30 * 60;
const SESSION_END: i64 = 13 * 3600 + 32 * 60; // 下午时段已平移，14:57 真实时间 -> 13:27

#[derive(Clone, Copy)]
struct Stat {
    price: f64,
    turnover: f64,
    trades: f64,
    cancels: f64,
    med_cancel: f64,
    med_wait: f64,
    p95_cancel: f64,
    neg_frac: f64,
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

/// 编号 -> 时间（µs），编号序列单调，落在两个已知编号之间就线性插值
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

fn median(v: &mut Vec<f64>) -> f64 {
    if v.is_empty() { return f64::NAN; }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn pct(v: &mut Vec<f64>, q: f64) -> f64 {
    if v.is_empty() { return f64::NAN; }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let i = (((v.len() - 1) as f64) * q).round() as usize;
    v[i]
}

fn per_stock(code: &str, date: i64) -> Option<Stat> {
    let recs: Vec<TradeRecord> = read_trade_fast_retreat(code, date).ok()?;
    let mut agg: Vec<(i64, i64)> = Vec::new();      // 主动方编号 -> 时间
    let mut pas: Vec<(i64, i64)> = Vec::new();      // 被动方编号 -> 成交时间
    let mut can: Vec<(i64, i64)> = Vec::new();      // 被撤单编号 -> 撤单时间
    let mut prices: Vec<f64> = Vec::new();
    let mut turnover = 0.0f64;
    for r in &recs {
        let sod = (r.time_us / 1_000_000).rem_euclid(86400);
        if sod < SESSION_START || sod > SESSION_END { continue; }
        match r.flag {
            66 => {
                if r.bid_order > 0 { agg.push((r.bid_order, r.time_us)); }
                if r.ask_order > 0 { pas.push((r.ask_order, r.time_us)); }
            }
            83 => {
                if r.ask_order > 0 { agg.push((r.ask_order, r.time_us)); }
                if r.bid_order > 0 { pas.push((r.bid_order, r.time_us)); }
            }
            32 => {
                let cid = if r.bid_order > 0 { r.bid_order } else { r.ask_order };
                if cid > 0 { can.push((cid, r.time_us)); }
            }
            _ => {}
        }
        if (r.flag == 66 || r.flag == 83) && r.price > 0.0 {
            prices.push(r.price);
            turnover += r.turnover;
        }
    }
    if agg.len() < MIN_TRADES { return None; }
    agg.sort_unstable();
    let vid: Vec<i64> = agg.iter().map(|x| x.0).collect();
    let vt: Vec<i64> = agg.iter().map(|x| x.1).collect();

    let mut cancel_sec: Vec<f64> = Vec::with_capacity(can.len());
    let mut neg = 0usize;
    for &(cid, t) in &can {
        let d = (t as f64 - interp(&vid, &vt, cid)) / 1e6;
        if d < 0.0 { neg += 1; }
        cancel_sec.push(d.max(0.0));
    }
    let mut wait_sec: Vec<f64> = Vec::with_capacity(pas.len());
    for &(pid, t) in &pas {
        let d = (t as f64 - interp(&vid, &vt, pid)) / 1e6;
        wait_sec.push(d.max(0.0));
    }
    if cancel_sec.is_empty() || wait_sec.is_empty() { return None; }
    let mut c2 = cancel_sec.clone();
    let mut w2 = wait_sec.clone();
    let med_cancel = median(&mut cancel_sec);
    let med_wait = median(&mut wait_sec);
    let p95_cancel = pct(&mut c2, 0.95);
    let price = median(&mut prices);
    Some(Stat {
        price,
        turnover,
        trades: w2.len() as f64,
        cancels: can.len() as f64,
        med_cancel,
        med_wait,
        p95_cancel,
        neg_frac: neg as f64 / (can.len().max(1)) as f64,
    })
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

fn spearman(x: &[f64], y: &[f64]) -> f64 {
    let (rx, ry) = (rank(x), rank(y));
    let n = x.len() as f64;
    let mx = rx.iter().sum::<f64>() / n;
    let my = ry.iter().sum::<f64>() / n;
    let mut sxy = 0.0; let mut sxx = 0.0; let mut syy = 0.0;
    for i in 0..x.len() {
        let (dx, dy) = (rx[i] - mx, ry[i] - my);
        sxy += dx * dy; sxx += dx * dx; syy += dy * dy;
    }
    if sxx <= 0.0 || syy <= 0.0 { return f64::NAN; }
    sxy / (sxx.sqrt() * syy.sqrt())
}

fn run(date: i64) {
    let wl = read_symbol_map(SYMBOL_MAP);
    let codes = list_codes(date, &wl);
    eprintln!("[{}] codes: {}", date, codes.len());
    let nt: usize = std::env::var("PLAY_THREADS").ok().and_then(|s| s.parse().ok()).unwrap_or(16);
    let next = AtomicUsize::new(0);
    let acc: Mutex<Vec<(String, Stat)>> = Mutex::new(Vec::new());
    let t0 = std::time::Instant::now();
    std::thread::scope(|sc| {
        for _ in 0..nt {
            sc.spawn(|| loop {
                let k = next.fetch_add(1, Ordering::Relaxed);
                if k >= codes.len() { break; }
                let c = &codes[k];
                if let Some(st) = per_stock(c, date) { acc.lock().unwrap().push((c.clone(), st)); }
            });
        }
    });
    let mut data = acc.into_inner().unwrap();
    data.sort_by(|a, b| a.0.cmp(&b.0));
    eprintln!("[{}] stocks kept: {} in {:.1}s", date, data.len(), t0.elapsed().as_secs_f64());

    let mut ratios: Vec<f64> = Vec::new();
    let mut prices: Vec<f64> = Vec::new();
    let mut turn: Vec<f64> = Vec::new();
    let mut ntr: Vec<f64> = Vec::new();
    let mut canshare: Vec<f64> = Vec::new();
    let mut med_cancels: Vec<f64> = Vec::new();
    let mut med_waits: Vec<f64> = Vec::new();
    let mut neg_frac: Vec<f64> = Vec::new();
    let mut named: Vec<String> = Vec::new();
    for (code, s) in &data {
        if s.med_wait <= 0.0 || s.med_cancel <= 0.0 { continue; }
        ratios.push((s.med_cancel / s.med_wait).ln());
        prices.push(s.price.ln());
        turn.push((s.turnover + 1.0).ln());
        ntr.push((s.trades).ln());
        canshare.push((s.cancels / s.trades).ln());
        med_cancels.push(s.med_cancel);
        med_waits.push(s.med_wait);
        neg_frac.push(s.neg_frac);
        if *code == "000001" || *code == "300750" {
            named.push(format!("{{\"code\":\"{}\",\"price\":{:.2},\"trades\":{:.0},\"cancels\":{:.0},\"med_cancel\":{:.1},\"med_wait\":{:.1},\"ratio\":{:.2}}}",
                code, s.price, s.trades, s.cancels, s.med_cancel, s.med_wait, s.med_cancel / s.med_wait));
        }
    }
    let mut r2 = ratios.clone();
    let mut mc = med_cancels.clone();
    let mut mw = med_waits.clone();
    let mut nf = neg_frac.clone();
    let (q10, q50, q90) = { let mut r3 = r2.clone(); (pct(&mut r3, 0.10), pct(&mut r3, 0.50), pct(&mut r3, 0.90)) };
    // 价格三分位下的比值
    let mut ord: Vec<usize> = (0..prices.len()).collect();
    ord.sort_by(|&a, &b| prices[a].partial_cmp(&prices[b]).unwrap());
    let k = ord.len() / 3;
    let mean_ln = |idx: &[usize]| idx.iter().map(|&i| ratios[i]).sum::<f64>() / idx.len().max(1) as f64;
    let low = mean_ln(&ord[..k]);
    let mid = mean_ln(&ord[k..2 * k]);
    let high = mean_ln(&ord[2 * k..]);
    println!(
        "{{\"date\":{},\"stocks\":{},\"med_cancel_s\":{:.1},\"med_wait_s\":{:.1},\"ratio_p10\":{:.2},\"ratio_p50\":{:.2},\"ratio_p90\":{:.2},\"ratio_lowpx\":{:.2},\"ratio_midpx\":{:.2},\"ratio_highpx\":{:.2},\"sp_ratio_price\":{:.3},\"sp_ratio_turnover\":{:.3},\"sp_ratio_trades\":{:.3},\"sp_ratio_cancelshare\":{:.3},\"neg_frac_mean\":{:.4},\"named\":[{}]}}",
        date, ratios.len(), median(&mut mc), median(&mut mw),
        q10.exp(), q50.exp(), q90.exp(),
        low.exp(), mid.exp(), high.exp(),
        spearman(&r2, &prices), spearman(&r2, &turn), spearman(&r2, &ntr), spearman(&r2, &canshare),
        nf.iter().sum::<f64>() / nf.len().max(1) as f64,
        named.join(",")
    );
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    assert!(!args.is_empty(), "usage: play_cancel <date> [date...]");
    for a in &args { run(a.parse::<i64>().unwrap()); }
}
