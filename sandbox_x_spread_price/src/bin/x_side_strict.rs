//! 成交方向条件的严格版：
//!  1) 方向只用「更早一段」的成交（滞后 LAG 到 LAG+60 秒），与当前快照的不平衡完全不重叠
//!  2) 同向 / 反向两组在同一个过去波动档里比较（用活跃度分层，不用「无成交」当基准）
//!  3) 报每组平均不平衡、组内高一组减低一组、组内不平衡上三成减下三成、条件命中率与无条件命中率
//!  4) 窗口扫描：方向窗口取 1/3/5/10/30 秒，看是单点冒尖还是整片高
//! 用法: XS_THREADS=12 XS_LAG_US=5000000 x_side_strict 20220819 [date...]
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

const HZ: [i64; 8] = [30_000_000, 60_000_000, 120_000_000, 180_000_000, 300_000_000, 600_000_000, 900_000_000, 1800_000_000];
#[derive(Clone, Copy)]
struct Obs { imb: f64, r_mid: f64, side: i32, pv: f64, price: f64, hz: [f64; 8] }

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

#[derive(Default, Clone)]
struct Cell { stock_days: f64, obs: f64, imb_sum: f64, ic: f64, ic2: f64,
              tb_imb: f64, tb_ret: f64, hit: f64, hit_n: f64, abs_ret: f64, hz_signed: [f64; 8], hz_n: [f64; 8] }

fn add_cell(c: &mut Cell, obs: &[Obs]) {
    if obs.len() < MIN_BUCKET { return; }
    c.stock_days += 1.0; c.obs += obs.len() as f64;
    let x: Vec<f64> = obs.iter().map(|o| o.imb).collect();
    let y: Vec<f64> = obs.iter().map(|o| o.r_mid).collect();
    c.imb_sum += x.iter().sum::<f64>() / x.len() as f64;
    if let Some(v) = spearman(&x, &y) { c.ic += v; c.ic2 += v * v; }
    // 按不平衡排序：上三成收益 − 下三成收益
    let mut idx: Vec<usize> = (0..obs.len()).collect();
    idx.sort_by(|&a, &b| x[a].partial_cmp(&x[b]).unwrap_or(std::cmp::Ordering::Equal));
    let k = (obs.len() as f64 * 0.3).floor() as usize;
    if k > 0 {
        let lo = idx[..k].iter().map(|&i| y[i]).sum::<f64>() / k as f64;
        let hi = idx[obs.len() - k..].iter().map(|&i| y[i]).sum::<f64>() / k as f64;
        c.tb_imb += hi - lo;
        // 按收益排序：高收益三成 − 低收益三成（组内「高的一组收益减低的一组」）
        let mut idy: Vec<usize> = (0..obs.len()).collect();
        idy.sort_by(|&a, &b| y[a].partial_cmp(&y[b]).unwrap_or(std::cmp::Ordering::Equal));
        let lo2 = idy[..k].iter().map(|&i| y[i]).sum::<f64>() / k as f64;
        let hi2 = idy[obs.len() - k..].iter().map(|&i| y[i]).sum::<f64>() / k as f64;
        c.tb_ret += hi2 - lo2;
    }
    for o in obs {
        if o.imb.abs() > 1e-9 { c.hit_n += 1.0; if (o.imb > 0.0) == (o.r_mid > 0.0) { c.hit += 1.0; } }
    }
    c.abs_ret += y.iter().map(|v| v.abs()).sum::<f64>() / y.len() as f64;
    for h in 0..8 {
        let mut sum = 0.0; let mut n = 0.0;
        for o in obs { let v = o.hz[h]; if v.is_finite() { sum += if o.imb >= 0.0 { v } else { -v }; n += 1.0; } }
        if n > 0.0 { c.hz_signed[h] += sum / n; c.hz_n[h] += 1.0; }
    }
}

fn run(date: i64) {
    let wl = read_symbol_map(SYMBOL_MAP);
    let codes = list_codes(date, &wl);
    let lag: i64 = std::env::var("XS_LAG_US").ok().and_then(|s| s.parse().ok()).unwrap_or(5_000_000);
    let nt: usize = std::env::var("XS_THREADS").ok().and_then(|s| s.parse().ok()).unwrap_or(10);
    let next = AtomicUsize::new(0);
    let collected: Mutex<Vec<(String, Vec<Obs>)>> = Mutex::new(Vec::new());
    std::thread::scope(|scope| {
        for _ in 0..nt { scope.spawn(|| loop {
            let k = next.fetch_add(1, Ordering::Relaxed);
            if k >= codes.len() { break; }
            let code = &codes[k];
            if let (Ok(mkt), Ok(trd)) = (read_market_fast(code, date), read_trade_fast(code, date)) {
                let o = per_stock(&mkt, &trd, lag);
                if o.len() >= MIN_OBS { collected.lock().unwrap().push((code.clone(), o)); }
            }
        }); }
    });
    let mut data = collected.into_inner().unwrap();
    data.sort_by(|a, b| a.0.cmp(&b.0));

    // 三组：同向 / 反向 / 该窗口无成交；同时在「过去波动三分位」内各算一遍
    let mut g: Vec<Cell> = vec![Cell::default(); 3];
    let mut gp: Vec<Cell> = vec![Cell::default(); 9];       // 波动档 x 组
    let mut uncond_hit = 0.0f64; let mut uncond_n = 0.0f64;
    let mut lag_trade_share = 0.0f64; let mut ld = 0.0f64;
    for (_c, obs) in &data {
        for (i, sel) in [1i32, -1i32, 0i32].iter().enumerate() {
            let v: Vec<Obs> = obs.iter().filter(|o| o.side == *sel).copied().collect();
            add_cell(&mut g[i], &v);
        }
        let mut pvs: Vec<f64> = obs.iter().map(|o| o.pv).collect();
        pvs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let c1 = pvs[pvs.len() / 3]; let c2 = pvs[pvs.len() * 2 / 3];
        for (i, sel) in [1i32, -1i32, 0i32].iter().enumerate() {
            let v: Vec<Obs> = obs.iter().filter(|o| o.side == *sel && o.pv <= c1).copied().collect();
            add_cell(&mut gp[0 * 3 + i], &v);
            let v: Vec<Obs> = obs.iter().filter(|o| o.side == *sel && o.pv > c1 && o.pv <= c2).copied().collect();
            add_cell(&mut gp[1 * 3 + i], &v);
            let v: Vec<Obs> = obs.iter().filter(|o| o.side == *sel && o.pv > c2).copied().collect();
            add_cell(&mut gp[2 * 3 + i], &v);
        }
        for o in obs { if o.imb.abs() > 1e-9 { uncond_n += 1.0; if (o.imb > 0.0) == (o.r_mid > 0.0) { uncond_hit += 1.0; } } }
        lag_trade_share += obs.iter().filter(|o| o.side != 0).count() as f64 / obs.len() as f64;
        ld += 1.0;
    }
    let jc = |name: &str, c: &Cell| -> String {
        let n = c.stock_days.max(1.0);
        let m = c.ic / n;
        let se = ((c.ic2 / n - m * m).max(0.0) / n).sqrt();
        format!("{{\"name\":\"{}\",\"stock_days\":{:.0},\"obs\":{:.0},\"mean_imb\":{:.4},\"ic_mid\":{:.4},\"ic_mid_se\":{:.4},\"tb_by_imb_bp\":{:.3},\"tb_by_ret_bp\":{:.3},\"cond_hit\":{:.4},\"abs_ret_bp\":{:.2},\"hz_signed\":[{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2}]}}",
            name, c.stock_days, c.obs, c.imb_sum / n, m, se, c.tb_imb / n, c.tb_ret / n,
            if c.hit_n > 0.0 { c.hit / c.hit_n } else { f64::NAN }, c.abs_ret / n,
            (0..8).map(|h| c.hz_signed[h] / c.hz_n[h].max(1.0)).collect::<Vec<f64>>()[0],
            (0..8).map(|h| c.hz_signed[h] / c.hz_n[h].max(1.0)).collect::<Vec<f64>>()[1],
            (0..8).map(|h| c.hz_signed[h] / c.hz_n[h].max(1.0)).collect::<Vec<f64>>()[2],
            (0..8).map(|h| c.hz_signed[h] / c.hz_n[h].max(1.0)).collect::<Vec<f64>>()[3],
            (0..8).map(|h| c.hz_signed[h] / c.hz_n[h].max(1.0)).collect::<Vec<f64>>()[4],
            (0..8).map(|h| c.hz_signed[h] / c.hz_n[h].max(1.0)).collect::<Vec<f64>>()[5],
            (0..8).map(|h| c.hz_signed[h] / c.hz_n[h].max(1.0)).collect::<Vec<f64>>()[6],
            (0..8).map(|h| c.hz_signed[h] / c.hz_n[h].max(1.0)).collect::<Vec<f64>>()[7])
    };
    let mut parts: Vec<String> = Vec::new();
    parts.push(format!("{{\"name\":\"lag_us\",\"value\":{}}}", lag));
    parts.push(format!("{{\"name\":\"lagged_trade_share\",\"value\":{:.4}}}", lag_trade_share / ld.max(1.0)));
    parts.push(format!("{{\"name\":\"uncond_hit\",\"value\":{:.4}}}", uncond_hit / uncond_n.max(1.0)));
    parts.push(jc("same", &g[0]));
    parts.push(jc("opposite", &g[1]));
    parts.push(jc("no_trade_in_window", &g[2]));
    let pn = ["pv_low", "pv_mid", "pv_high"];
    for b in 0..3 {
        parts.push(jc(&format!("{}__same", pn[b]), &gp[b * 3]));
        parts.push(jc(&format!("{}__opposite", pn[b]), &gp[b * 3 + 1]));
    }
    println!("{{\"date\":{},\"stock_days\":{},\"groups\":[{}]}}", date, data.len(), parts.join(","));
}

fn per_stock(mkt: &[MarketRecord], trd: &[TradeRecord], lag: i64) -> Vec<Obs> {
    let mut trades: Vec<&TradeRecord> = trd.iter().filter(|t| t.price > 0.0 && t.volume > 0.0).collect();
    trades.sort_by_key(|t| t.time_us);
    let mut valid: Vec<usize> = Vec::new();
    for (i, r) in mkt.iter().enumerate() {
        if r.last_prc > 0.0 && r.ask_prcs[0] > 0.0 && r.bid_prcs[0] > 0.0 && r.ask_prcs[0] > r.bid_prcs[0] { valid.push(i); }
    }
    let mut out = Vec::new();
    if valid.is_empty() || trades.is_empty() { return out; }
    // 每笔成交的方向：用成交当时的中间价（前面最近的有效快照）
    let mut side_of: Vec<i32> = Vec::with_capacity(trades.len());
    {
        let mut vi = 0usize;
        for tr in &trades {
            while vi + 1 < valid.len() && mkt[valid[vi + 1]].time_us <= tr.time_us { vi += 1; }
            let r = &mkt[valid[vi]];
            let mid = 0.5 * (r.ask_prcs[0] + r.bid_prcs[0]);
            side_of.push(if mid > 0.0 && tr.price >= mid { 1 } else { -1 });
        }
    }
    let mut j = 0usize;
    let mut a = 0usize; let mut b = 0usize;
    for (k, &i) in valid.iter().enumerate() {
        let r = &mkt[i];
        let t = r.time_us;
        let sod = (t / 1_000_000).rem_euclid(86400);
        if sod >= 11 * 3600 + 29 * 60 && sod < 11 * 3600 + 30 * 60 { continue; }
        // 方向只用滞后窗口 [t-lag-60s, t-lag] 内的成交
        while a < trades.len() && trades[a].time_us < t - lag - FWD_US { a += 1; }
        if b < a { b = a; }
        while b < trades.len() && trades[b].time_us <= t - lag { b += 1; }
        let mut last: i32 = 0;
        for w in a..b { if side_of[w] != 0 { last = side_of[w]; } }
        let side = last;
        // 过去 60 秒（到 t 为止）的中间价波动
        let pstart = if k > 20 { k - 20 } else { 0 };
        let mut s = 0.0; let mut s2 = 0.0; let mut cnt = 0.0; let mut prev = f64::NAN;
        for w in pstart..=k {
            let m = 0.5 * (mkt[valid[w]].ask_prcs[0] + mkt[valid[w]].bid_prcs[0]);
            if m <= 0.0 { continue; }
            if prev.is_finite() { let x = (m / prev - 1.0) * 1e4; s += x; s2 += x * x; cnt += 1.0; }
            prev = m;
        }
        let pv = if cnt >= 3.0 { let mu = s / cnt; ((s2 / cnt - mu * mu).max(0.0)).sqrt() } else { 0.0 };
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
        let mut hz = [f64::NAN; 8];
        {
            let mut hj = k;
            for (hi, &h) in HZ.iter().enumerate() {
                while hj < valid.len() && mkt[valid[hj]].time_us < t + h { hj += 1; }
                if hj < valid.len() && mkt[valid[hj]].time_us - t <= h + MAX_GAP_US {
                    let hm = 0.5 * (mkt[valid[hj]].ask_prcs[0] + mkt[valid[hj]].bid_prcs[0]);
                    hz[hi] = (hm / mid - 1.0) * 1e4;
                }
            }
        }
        out.push(Obs { imb: (bid5 - ask5) / (bid5 + ask5), r_mid: (fmid / mid - 1.0) * 1e4, side, pv, price: r.last_prc, hz });
    }
    out
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    assert!(!args.is_empty(), "usage: x_side_strict <date> [date...]");
    for a in &args { run(a.parse::<i64>().expect("date")); }
}