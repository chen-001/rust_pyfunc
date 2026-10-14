//! 重填比（refill）全市场检查（refill_sandbox）
//!
//! 对每一笔成交，按方向取对手盘十档（主动买 66 看卖档，主动卖 83 看买档），
//! 在成交时刻最近的那张快照里按价格找到成交价那一档，取挂单量 cur；下一张快照同价再取 nxt。
//!   重填比（只算价格还在的样本）= max(nxt - cur, 0) / 该笔成交量
//!   重填比（价格跳走记零）      = 同上，但 nxt 查不到时记 0
//! 两个版本一起报，用来判断「只算活下来的样本」这个选择偏差有多大。
//! 另出存活率：成交后同价还在的成交占比。
//!
//! 用法: ./target/release/refill 20220819 ALL   （ALL 取全市场 A 股正股）

use rayon::prelude::*;
use serde::Serialize;

#[derive(Clone, Copy, Default)]
struct M { t: i64, ap: [f64; 10], av: [f64; 10], bp: [f64; 10], bv: [f64; 10] }
#[derive(Clone, Copy, Default)]
struct Tr { t: i64, price: f64, vol: f64, turn: f64, flag: i32 }

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
        out.push(Tr {
            t, price: f[7].parse().unwrap_or(0.0), vol: f[8].parse().unwrap_or(0.0),
            turn: f[9].parse().unwrap_or(0.0), flag,
        });
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
    survival_pct: f64,
    refill_sel_mean: f64,
    refill_all_mean: f64,
    refill_sel_med: f64,
    refill_w_mean: f64,
    refill_trim_mean: f64,
    zero_share_pct: f64,
    mean_trade_vol: f64,
    turnover: f64,
}

fn median(v: &mut Vec<f64>) -> f64 {
    if v.is_empty() { return f64::NAN; }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn one(code: &str, date: i64) -> Option<Out> {
    let mp = format!("/ssd_data/stock/{}/market_data/{}_{}_market_data.csv", date, code, date);
    let tp = format!("/ssd_data/stock/{}/transaction/{}_{}_transaction.csv", date, code, date);
    let mkt = load_m(&mp).ok()?;
    let tr = load_t(&tp).ok()?;
    if mkt.len() < 200 { return None; }
    let mut used = 0usize; let mut both = 0usize; let mut with_cur = 0usize;
    let mut rs = 0.0; let mut ra = 0.0; let mut vsel = Vec::new(); let mut pairs: Vec<(f64, f64)> = Vec::new(); let mut nzero = 0usize;
    let mut vsum = 0.0; let mut turn = 0.0;
    let mut j = 0usize;
    for t in &tr {
        while j + 1 < mkt.len() && mkt[j + 1].t <= t.t { j += 1; }
        if mkt[j].t > t.t { continue; }
        if j == 0 || j + 1 >= mkt.len() { continue; }
        if mkt[j + 1].t - mkt[j].t > 6_000_000 || mkt[j].t - mkt[j - 1].t > 6_000_000 { continue; }
        used += 1;
        turn += t.turn;
        vsum += t.vol;
        let look_ask = t.flag == 66;
        let cur = depth_at(&mkt[j], t.price, look_ask);
        if cur.is_none() { continue; }
        with_cur += 1;
        let c = cur.unwrap();
        let vol = t.vol.max(1.0);
        match depth_at(&mkt[j + 1], t.price, look_ask) {
            Some(n) => {
                let r = (n - c).max(0.0) / vol;
                rs += r; ra += r; vsel.push(r);
                if r == 0.0 { nzero += 1; }
                pairs.push((r, vol));
                both += 1;
            }
            None => { ra += 0.0; }
        }
    }
    let den = with_cur.max(1) as f64;
    let volsum: f64 = pairs.iter().map(|x| x.1).sum();
    let wmean = if volsum > 0.0 { pairs.iter().map(|x| x.0 * x.1).sum::<f64>() / volsum } else { f64::NAN };
    let mut byvol = pairs.clone();
    byvol.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    let dropn = byvol.len() / 10;
    let tail = &byvol[dropn.min(byvol.len())..];
    let trim = if tail.is_empty() { f64::NAN } else { tail.iter().map(|x| x.0).sum::<f64>() / tail.len() as f64 };
    Some(Out {
        code: code.into(),
        n_trades: tr.len(),
        used,
        survival_pct: 100.0 * both as f64 / den,
        refill_sel_mean: rs / both.max(1) as f64,
        refill_all_mean: ra / den,
        refill_sel_med: median(&mut vsel),
        refill_w_mean: wmean,
        refill_trim_mean: trim,
        zero_share_pct: 100.0 * nzero as f64 / both.max(1) as f64,
        mean_trade_vol: vsum / used.max(1) as f64,
        turnover: turn,
    })
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let date: i64 = a.get(1).and_then(|s| s.parse().ok()).unwrap_or(20220819);
    let arg = a.get(2).cloned().unwrap_or_else(|| "ALL".into());
    let codes: Vec<String> = if arg == "ALL" {
        let dir = format!("/ssd_data/stock/{}/transaction", date);
        let mut v: Vec<String> = match std::fs::read_dir(&dir) {
            Ok(rd) => rd.flatten()
                .filter_map(|e| e.file_name().to_str().map(|s| s.to_string()))
                .filter_map(|n| n.split('_').next().map(|s| s.to_string()))
                .filter(|c| c.len() == 6 && c.bytes().all(|b| b.is_ascii_digit()))
                .filter(|c| matches!(&c[..3], "000"|"001"|"002"|"003"|"300"|"301"|"600"|"601"|"603"|"605"|"688"))
                .collect(),
            Err(_) => Vec::new(),
        };
        v.sort(); v.dedup();
        let step: usize = std::env::var("PANEL_STEP").ok().and_then(|s| s.parse().ok()).unwrap_or(1);
        v.into_iter().step_by(step.max(1)).collect()
    } else {
        arg.split(',').map(|x| x.trim().to_string()).filter(|x| !x.is_empty()).collect()
    };
    eprintln!("refill_sandbox date={} codes={}", date, codes.len());
    let rows: Vec<Out> = codes.par_iter().filter_map(|c| one(c, date)).collect();
    for r in &rows { println!("{}", serde_json::to_string(r).unwrap()); }
    eprintln!("done rows={}", rows.len());
}
