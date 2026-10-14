//! 对撞：不平衡指向的一侧最近有没有成交 —— 会不会提高排序准度（预期：不会，只提高幅度）
//! 用法: XS_THREADS=10 x_recent_side 20220819 20241231
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

const HZ: [i64; 8] = [30_000_000, 60_000_000, 120_000_000, 180_000_000, 300_000_000, 600_000_000, 900_000_000, 1800_000_000];

#[derive(Clone, Copy)]
struct Obs { imb: f64, r: [f64; 8], r_long: f64, r_10m: f64, side: i32, price: f64, sp_bps: f64 }
const PX: [f64; 4] = [5.0, 20.0, 100.0, f64::INFINITY];
const LONG_US: i64 = 300_000_000;
const LONG2_US: i64 = 600_000_000;

/// 每只股票日的信号命中率与平均不平衡符号，用来区分「机械」和「真信息」
fn hit_stats(obs: &[Obs], sel: i32) -> Option<(f64, f64, f64)> {
    let v: Vec<&Obs> = obs.iter().filter(|o| o.side == sel).collect();
    if v.len() < 30 { return None; }
    let mut hit = 0.0f64; let mut nz = 0.0f64; let mut imb_sum = 0.0f64;
    for o in &v {
        imb_sum += o.imb;
        if o.imb.abs() > 1e-9 { nz += 1.0; if (o.imb > 0.0) == (o.r[1] > 0.0) { hit += 1.0; } }
    }
    Some((hit / nz.max(1.0), if v.is_empty() { f64::NAN } else { imb_sum / v.len() as f64 }, nz / v.len() as f64))
}

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
fn absret(v: &[f64]) -> f64 { if v.is_empty() { f64::NAN } else { v.iter().map(|x| x.abs()).sum::<f64>() / v.len() as f64 } }

#[derive(Default, Clone, Copy)]
struct Acc { n: f64, ic: f64, ic2: f64, tb: f64, tb_long: f64, ab: f64, signed: f64, signed_se: f64, signed2: f64, signed_n: f64, h_hit: f64, h_imb: f64, h_nz: f64, signed10: f64, signed10_n: f64, hz_sum: [f64; 8], hz_n: [f64; 8] }

fn per_stock(mkt: &[MarketRecord], trd: &[TradeRecord]) -> Vec<Obs> {
    // 逐笔按时间排序（原始文件里已是时间序，保险再排一次）
    let mut trades: Vec<&TradeRecord> = trd.iter().collect();
    trades.sort_by_key(|t| t.time_us);
    let mut out = Vec::new();
    let mut last_side = 0i32;      // +1 主动买（价格偏向卖一），-1 主动卖，0 未知
    let mut last_t_us = i64::MIN;
    // 有效快照
    let mut valid: Vec<usize> = Vec::new();
    for (i, r) in mkt.iter().enumerate() {
        if r.last_prc > 0.0 && r.ask_prcs[0] > 0.0 && r.bid_prcs[0] > 0.0 && r.ask_prcs[0] > r.bid_prcs[0] { valid.push(i); }
    }
    if valid.is_empty() || trades.is_empty() { return out; }
    // 逐笔方向用「成交当时的中间价」：把每笔挂到它前面最近的那个有效快照上
    let mut mid_of_trade: Vec<i32> = Vec::with_capacity(trades.len());
    {
        let mut vi = 0usize;
        for tr in &trades {
            while vi + 1 < valid.len() && mkt[valid[vi + 1]].time_us <= tr.time_us { vi += 1; }
            let r = &mkt[valid[vi]];
            let mid = 0.5 * (r.ask_prcs[0] + r.bid_prcs[0]);
            if tr.price > 0.0 && tr.volume > 0.0 && mid > 0.0 {
                mid_of_trade.push(if tr.price >= mid { 1 } else { -1 });
            } else { mid_of_trade.push(0); }
        }
    }
    // 重建最后一次成交的方向（按逐笔自己的时间轴）；ti 与 mid_of_trade 一一对应
    let mut ti = 0usize;
    let mut j = 0usize;
    for (k, &i) in valid.iter().enumerate() {
        let r = &mkt[i];
        let t = r.time_us;
        let sod = (t / 1_000_000).rem_euclid(86400);
        if sod >= 11 * 3600 + 29 * 60 && sod < 11 * 3600 + 30 * 60 { continue; }
        // 把到当前时刻为止的逐笔推进，记录最近一笔的主动方向
        while ti < trades.len() && trades[ti].time_us <= t {
            let s = mid_of_trade[ti];
            if s != 0 { last_side = s; last_t_us = trades[ti].time_us; }
            ti += 1;
        }
        if j < k { j = k; }
        while j < valid.len() && mkt[valid[j]].time_us < t + FWD_US { j += 1; }
        if j >= valid.len() { break; }
        let fj = valid[j];
        if mkt[fj].time_us - t > MAX_GAP_US { continue; }
        let f = &mkt[fj];
        let mid = 0.5 * (r.ask_prcs[0] + r.bid_prcs[0]);
        let _fmid = 0.5 * (f.ask_prcs[0] + f.bid_prcs[0]);
        // 5 分钟后的中间价（同一有效快照序列里找）
        let mut lj = j;
        while lj < valid.len() && mkt[valid[lj]].time_us < t + LONG_US { lj += 1; }
        let r_long = if lj < valid.len() && mkt[valid[lj]].time_us - t <= LONG_US + MAX_GAP_US {
            let lm = 0.5 * (mkt[valid[lj]].ask_prcs[0] + mkt[valid[lj]].bid_prcs[0]);
            (lm / mid - 1.0) * 1e4
        } else { f64::NAN };
        let mut l2 = lj;
        while l2 < valid.len() && mkt[valid[l2]].time_us < t + LONG2_US { l2 += 1; }
        let r_10m = if l2 < valid.len() && mkt[valid[l2]].time_us - t <= LONG2_US + MAX_GAP_US {
            let lm2 = 0.5 * (mkt[valid[l2]].ask_prcs[0] + mkt[valid[l2]].bid_prcs[0]);
            (lm2 / mid - 1.0) * 1e4
        } else { f64::NAN };
        // 多窗口前瞻收益（每只股票日各自算，缺窗口置 NaN）
        let mut hz_r = [f64::NAN; 8];
        {
            let mut hj = k;
            for (hi, &h) in HZ.iter().enumerate() {
                while hj < valid.len() && mkt[valid[hj]].time_us < t + h { hj += 1; }
                if hj < valid.len() && mkt[valid[hj]].time_us - t <= h + MAX_GAP_US {
                    let hm = 0.5 * (mkt[valid[hj]].ask_prcs[0] + mkt[valid[hj]].bid_prcs[0]);
                    hz_r[hi] = (hm / mid - 1.0) * 1e4;
                }
            }
        }
        let bid5: f64 = r.bid_vols[..5].iter().sum();
        let ask5: f64 = r.ask_vols[..5].iter().sum();
        if bid5 + ask5 <= 0.0 { continue; }
        let imb = (bid5 - ask5) / (bid5 + ask5);
        // 最近一笔成交在 5 秒内才算「刚刚成交」；side 与不平衡同向记 +1，反向记 −1，无成交记 0
        let win: i64 = std::env::var("XS_SIDE_WIN_US").ok().and_then(|s| s.parse().ok()).unwrap_or(5_000_000);
        let side = if last_t_us != i64::MIN && t - last_t_us <= win {
            if (imb > 0.0 && last_side > 0) || (imb < 0.0 && last_side < 0) { 1 } else { -1 }
        } else { 0 };
        if !r_long.is_finite() || !r_10m.is_finite() { continue; }
        if !hz_r[4].is_finite() || !hz_r[7].is_finite() { continue; }
        let sp_bps = (r.ask_prcs[0] - r.bid_prcs[0]) / mid * 1e4;
        out.push(Obs { imb, r: hz_r, r_long, r_10m, side, price: r.last_prc, sp_bps });
    }
    out
}

fn run(date: i64) {
    let wl = read_symbol_map(SYMBOL_MAP);
    let codes = list_codes(date, &wl);
    let nt: usize = std::env::var("XS_THREADS").ok().and_then(|s| s.parse().ok()).unwrap_or(10);
    let next = AtomicUsize::new(0);
    let collected: Mutex<Vec<(String, Vec<Obs>)>> = Mutex::new(Vec::new());
    let t0 = std::time::Instant::now();
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
    eprintln!("[{}] stock-days: {} in {:.1}s", date, data.len(), t0.elapsed().as_secs_f64());

    // 三组：同向刚成交 / 反向刚成交 / 5 秒内无成交
    let mut acc = vec![Acc::default(); 3];
    let mut shares = [0f64; 3];
    for (_c, obs) in &data {
        for (g, sel) in [1i32, -1i32, 0i32].iter().enumerate() {
            let idxs: Vec<usize> = (0..obs.len()).filter(|&k| obs[k].side == *sel).collect();
            if idxs.len() < MIN_BUCKET { continue; }
            for &k in &idxs { shares[g] += 1.0; }
            let x: Vec<f64> = idxs.iter().map(|&k| obs[k].imb).collect();
            let y: Vec<f64> = idxs.iter().map(|&k| obs[k].r[1]).collect();
            let yl: Vec<f64> = idxs.iter().map(|&k| obs[k].r_long).collect();
            let xl: Vec<f64> = x.clone();
            let a = &mut acc[g];
            if let Some(v) = spearman(&x, &y) { a.ic += v; a.ic2 += v * v; }
            if let Some(v) = top_bottom(&x, &y) { a.n += 1.0; a.tb += v; }
            a.ab += absret(&y);
            if let Some(v) = top_bottom(&xl, &yl) { a.tb_long += v; }
            for (hi, h) in HZ.iter().enumerate() { let _ = h;
                let mut s = 0.0; let mut c = 0.0;
                for &k in &idxs { let v = obs[k].r[hi]; if v.is_finite() { s += if obs[k].imb >= 0.0 { v } else { -v }; c += 1.0; } }
                if c > 0.0 { a.hz_sum[hi] += s / c; a.hz_n[hi] += 1.0; }
            }
            let s10: f64 = idxs.iter().map(|&k| if obs[k].imb >= 0.0 { obs[k].r_10m } else { -obs[k].r_10m }).sum::<f64>() / idxs.len() as f64;
            a.signed10 += s10; a.signed10_n += 1.0;
            a.h_hit += hit_stats(obs, *sel).map(|z| z.0).unwrap_or(0.0);
            a.h_imb += hit_stats(obs, *sel).map(|z| z.1).unwrap_or(0.0);
            a.h_nz += hit_stats(obs, *sel).map(|z| z.2).unwrap_or(0.0);
            // 有方向收益：按不平衡符号持有的 60 秒收益（bp），正数表示方向对
            let s: f64 = idxs.iter().map(|&k| if obs[k].imb >= 0.0 { obs[k].r[1] } else { -obs[k].r[1] }).sum::<f64>() / idxs.len() as f64;
            a.signed += s; a.signed2 += s * s; a.signed_n += 1.0;
        }
    }
    let names = ["same_side_recent", "opposite_side_recent", "no_trade_5s"];
    // 按价格桶分别算：粗收益、一半点差、扣完净剩、能过门槛的占比
    let mut px_parts: Vec<String> = Vec::new();
    for pi in 0..4 {
        for (sel, nm) in [(1i32, "same"), (-1i32, "opposite")] {
            let mut gross = 0.0f64; let mut half = 0.0f64; let mut net = 0.0f64;
            let mut n = 0.0f64; let mut pass = 0.0f64; let mut pass_n = 0.0f64;
            for (_c, obs) in &data {
                let v: Vec<&Obs> = obs.iter().filter(|o| o.side == sel && o.price >= PX[pi] && o.price < PX[pi + 1] && o.r[1].is_finite()).collect();
                if v.len() < 20 { continue; }
                let g = v.iter().map(|o| if o.imb >= 0.0 { o.r[1] } else { -o.r[1] }).sum::<f64>() / v.len() as f64;
                let h = v.iter().map(|o| o.sp_bps).sum::<f64>() / v.len() as f64 / 2.0;
                gross += g; half += h; net += g - 2.0 * h; n += 1.0;
                for o in &v {
                    let one = if o.imb >= 0.0 { o.r[1] } else { -o.r[1] };
                    pass_n += 1.0; if one > 2.0 * o.sp_bps / 2.0 { pass += 1.0; }
                }
            }
            if n > 0.0 {
                px_parts.push(format!("{{\"name\":\"px{}_sq{}\",\"stock_days\":{:.0},\"gross_bp\":{:.3},\"half_spread_bp\":{:.3},\"net_bp\":{:.3},\"pass_share\":{:.4}}}",
                    pi, nm, n, gross / n, half / n, net / n, pass / pass_n.max(1.0)));
            }
        }
    }
    let mut parts: Vec<String> = Vec::new();
    let tot: f64 = shares.iter().sum();
    for p in px_parts { parts.push(p); }
    for g in 0..3 {
        let a = &acc[g]; let n = a.n.max(1.0);
        let m = a.ic / n; let se = ((a.ic2 / n - m * m).max(0.0) / n).sqrt();
        parts.push(format!("{{\"name\":\"{}\",\"stock_days\":{:.0},\"obs\":{:.0},\"obs_share\":{:.4},\"ic_mid\":{:.4},\"ic_mid_se\":{:.4},\"tb_mid_bp\":{:.3},\"absret_bp\":{:.3},\"signed_bp\":{:.3},\"signed_se\":{:.3},\"tb_long_bp\":{:.3},\"hit_rate\":{:.4},\"imb_mean\":{:.4},\"nz_share\":{:.4},\"signed_bp_10m\":{:.3},\"hz_signed_bp\":[{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2}]}}",
            names[g], a.n, shares[g], shares[g] / tot.max(1.0), m, se, a.tb / n, a.ab / n,
            a.signed / a.signed_n.max(1.0),
            ((a.signed2 / a.signed_n.max(1.0) - (a.signed / a.signed_n.max(1.0)).powi(2)).max(0.0) / a.signed_n.max(1.0)).sqrt(),
            a.tb_long / n, a.h_hit / n, a.h_imb / n, a.h_nz / n, a.signed10 / a.signed10_n.max(1.0),
            (0..8).map(|hi| a.hz_sum[hi] / a.hz_n[hi].max(1.0)).collect::<Vec<f64>>()[0],
            (0..8).map(|hi| a.hz_sum[hi] / a.hz_n[hi].max(1.0)).collect::<Vec<f64>>()[1],
            (0..8).map(|hi| a.hz_sum[hi] / a.hz_n[hi].max(1.0)).collect::<Vec<f64>>()[2],
            (0..8).map(|hi| a.hz_sum[hi] / a.hz_n[hi].max(1.0)).collect::<Vec<f64>>()[3],
            (0..8).map(|hi| a.hz_sum[hi] / a.hz_n[hi].max(1.0)).collect::<Vec<f64>>()[4],
            (0..8).map(|hi| a.hz_sum[hi] / a.hz_n[hi].max(1.0)).collect::<Vec<f64>>()[5],
            (0..8).map(|hi| a.hz_sum[hi] / a.hz_n[hi].max(1.0)).collect::<Vec<f64>>()[6],
            (0..8).map(|hi| a.hz_sum[hi] / a.hz_n[hi].max(1.0)).collect::<Vec<f64>>()[7]));
    }
    println!("{{\"date\":{},\"stock_days\":{},\"groups\":[{}]}}", date, data.len(), parts.join(","));
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    assert!(!args.is_empty(), "usage: x_recent_side <date> [date...]");
    for a in &args { run(a.parse::<i64>().expect("date")); }
}