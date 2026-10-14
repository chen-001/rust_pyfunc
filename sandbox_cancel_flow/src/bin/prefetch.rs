//! 成交价所在档位在快照里存在吗，存在时前后 3 秒的挂单增量长什么样（prefetch_sandbox）
//!
//! 对每一笔成交，按方向取对手盘十档（主动买 66 看卖档，主动卖 83 看买档），
//! 在成交时刻最近的一张快照里找价格等于成交价的那一档，再看下一张快照里还在不在。
//! 在的话算 dV_before = 该价挂单量(当下) - 该价挂单量(上一张)，
//! dV_after = 该价挂单量(下一张) - 该价挂单量(当下)，以及比值 dV_before/(dV_before+dV_after)。
//!
//! 构建: cd sandbox_cancel_flow && cargo build --release
//! 运行: ./target/release/prefetch 20220819 000001,600519,...

use rayon::prelude::*;
use serde::Serialize;

#[derive(Clone, Copy, Default)]
struct M { t: i64, ap: [f64; 10], av: [f64; 10], bp: [f64; 10], bv: [f64; 10] }
#[derive(Clone, Copy, Default)]
struct Tr { t: i64, price: f64, vol: f64, flag: i32 }

fn load_m(path: &str) -> std::io::Result<Vec<M>> {
    let c = std::fs::read_to_string(path)?;
    let mut out = Vec::with_capacity(c.len() / 700);
    for line in c.lines().skip(1) {
        let f: Vec<&str> = line.split(',').collect();
        if f.len() < 61 { continue; }
        let t: i64 = f[4].parse().unwrap_or(0);
        if t == 0 { continue; }
        let mut m = M { t, ..Default::default() };
        for i in 0..10 {
            let b = 21 + i * 4;
            m.ap[i] = f[b].parse().unwrap_or(0.0);
            m.av[i] = f[b + 1].parse().unwrap_or(0.0);
            m.bp[i] = f[b + 2].parse().unwrap_or(0.0);
            m.bv[i] = f[b + 3].parse().unwrap_or(0.0);
        }
        out.push(m);
    }
    Ok(out)
}

fn load_t(path: &str) -> std::io::Result<Vec<Tr>> {
    let c = std::fs::read_to_string(path)?;
    let mut out = Vec::with_capacity(c.len() / 120);
    for line in c.lines().skip(1) {
        let f: Vec<&str> = line.split(',').collect();
        if f.len() < 15 { continue; }
        let t: i64 = f[4].parse().unwrap_or(0);
        if t == 0 { continue; }
        let flag: i32 = f[10].parse().unwrap_or(0);
        if flag != 66 && flag != 83 { continue; }
        out.push(Tr { t, price: f[7].parse().unwrap_or(0.0), vol: f[8].parse().unwrap_or(0.0), flag });
    }
    Ok(out)
}

fn depth_at(m: &M, p: f64, look_ask: bool) -> Option<f64> {
    for i in 0..10 {
        let (pp, vv) = if look_ask { (m.ap[i], m.av[i]) } else { (m.bp[i], m.bv[i]) };
        if pp > 0.0 && (pp - p).abs() < 0.005 { return Some(vv); }
    }
    None
}

#[derive(Serialize)]
struct Out {
    code: String,
    n_trades: usize,
    used: usize,
    present_before: usize,
    present_after: usize,
    present_both: usize,
    present_both_pct: f64,
    absent_before_pct: f64,
    absent_after_pct: f64,
    denom_tiny: usize,
    denom_tiny_pct: f64,
    ratio_median: f64,
    ratio_q25: f64,
    ratio_q75: f64,
    ratio_in_0_1_pct: f64,
    dv_before_median: f64,
    dv_after_median: f64,
    trade_vol_median: f64,
}

fn q(v: &mut Vec<f64>, p: f64) -> f64 {
    if v.is_empty() { return f64::NAN; }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[(((v.len() as f64 - 1.0) * p).round() as usize).min(v.len() - 1)]
}

fn one(code: &str, date: i64) -> Option<Out> {
    let mp = format!("/ssd_data/stock/{}/market_data/{}_{}_market_data.csv", date, code, date);
    let tp = format!("/ssd_data/stock/{}/transaction/{}_{}_transaction.csv", date, code, date);
    let mkt = load_m(&mp).ok()?;
    let tr = load_t(&tp).ok()?;
    if mkt.len() < 200 { return None; }
    let mut pb = 0; let mut pa = 0; let mut pboth = 0; let mut used = 0; let mut tiny = 0;
    let mut ratios = Vec::new(); let mut dbs = Vec::new(); let mut das = Vec::new(); let mut vols = Vec::new();
    let mut j = 0usize;
    for t in &tr {
        while j + 1 < mkt.len() && mkt[j + 1].t <= t.t { j += 1; }
        if mkt[j].t > t.t { continue; }
        if j == 0 || j + 1 >= mkt.len() { continue; }
        // 只保留相邻快照间隔在 6 秒以内的样本，保证是「3 秒格」
        if mkt[j + 1].t - mkt[j].t > 6_000_000 || mkt[j].t - mkt[j - 1].t > 6_000_000 { continue; }
        used += 1;
        let look_ask = t.flag == 66;             // 主动买吃卖档，主动卖吃买档
        let cur = depth_at(&mkt[j], t.price, look_ask);
        let nxt = depth_at(&mkt[j + 1], t.price, look_ask);
        let prv = depth_at(&mkt[j - 1], t.price, look_ask);
        if prv.is_some() { pb += 1; }
        if nxt.is_some() { pa += 1; }
        vols.push(t.vol);
        if let (Some(c), Some(n)) = (cur, nxt) {
            pboth += 1;
            if let Some(p) = prv {
                let d_before = c - p;
                let d_after = n - c;
                dbs.push(d_before); das.push(d_after);
                let den = d_before + d_after;
                if den.abs() < 200.0 { tiny += 1; } else { ratios.push(d_before / den); }
            }
        }
    }
    let n = used.max(1) as f64;
    Some(Out {
        code: code.into(),
        n_trades: tr.len(),
        used,
        present_before: pb,
        present_after: pa,
        present_both: pboth,
        present_both_pct: 100.0 * pboth as f64 / n,
        absent_before_pct: 100.0 * (used - pb) as f64 / n,
        absent_after_pct: 100.0 * (used - pa) as f64 / n,
        denom_tiny: tiny,
        denom_tiny_pct: 100.0 * tiny as f64 / n,
        ratio_median: q(&mut ratios.clone(), 0.5),
        ratio_q25: q(&mut ratios.clone(), 0.25),
        ratio_q75: q(&mut ratios.clone(), 0.75),
        ratio_in_0_1_pct: 100.0 * ratios.iter().filter(|&&r| r >= 0.0 && r <= 1.0).count() as f64 / ratios.len().max(1) as f64,
        dv_before_median: q(&mut dbs, 0.5),
        dv_after_median: q(&mut das, 0.5),
        trade_vol_median: q(&mut vols, 0.5),
    })
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let date: i64 = a.get(1).and_then(|s| s.parse().ok()).unwrap_or(20220819);
    let codes: Vec<String> = a.get(2).map(|s| s.split(',').map(|x| x.trim().to_string()).collect())
        .unwrap_or_else(|| vec!["000001".into()]);
    let mut out: Vec<Out> = codes.par_iter().filter_map(|c| one(c, date)).collect();
    out.sort_by(|x, y| x.code.cmp(&y.code));
    eprintln!("prefetch_sandbox date={} stocks={}", date, out.len());
    println!("{}", serde_json::to_string(&out).unwrap());
}
