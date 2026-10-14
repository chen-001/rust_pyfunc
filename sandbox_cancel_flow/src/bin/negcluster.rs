//! 恒等式负残差在时间上是否聚集（negcluster_sandbox）
//!
//! 恒等式：新增买单 = 快照总买量之差 + 撤买量 + 主动卖成交量(flag 83)。
//! 逐格算，若为负记一笔。把负值格和全部格的「秒内位置、半小时位置、格长」分别打直方图，
//! 用来判断负值是快照与逐笔的时间戳没对齐（会聚在格边界或格长变大处），还是随机散落。
//! 用法: ./target/release/negcluster 20220819 ALL

use rayon::prelude::*;
use serde::Serialize;

#[derive(Clone, Copy, Default)]
struct M { t: i64, tb: f64 }
#[derive(Clone, Copy, Default)]
struct Tr { t: i64, vol: f64, flag: i32 }

#[derive(Serialize)]
struct Out {
    code: String,
    n_iv: usize,
    n_neg: usize,
    n_pos: usize,
    pos_sum: f64,
    neg_sum: f64,
    abs_sum: f64,
    dt_all: Vec<usize>,
    dt_neg: Vec<usize>,
    sec_all: Vec<usize>,
    sec_neg: Vec<usize>,
    hh_all: Vec<usize>,
    hh_neg: Vec<usize>,
}

fn lsec(us: i64) -> i64 { ((us / 1_000_000) + 8 * 3600).rem_euclid(86400) }
fn in_session(us: i64) -> bool {
    let s = lsec(us);
    (s >= 9 * 3600 + 30 * 60 && s <= 11 * 3600 + 30 * 60) || (s >= 13 * 3600 && s <= 15 * 3600)
}

fn load_m(path: &str) -> std::io::Result<Vec<M>> {
    let c = std::fs::read_to_string(path)?;
    let mut out = Vec::with_capacity(c.len() / 700);
    for line in c.lines().skip(1) {
        let f: Vec<&str> = line.split(',').collect();
        if f.len() < 25 { continue; }
        let t: i64 = f[4].parse().unwrap_or(0);
        if t == 0 || !in_session(t) { continue; }
        out.push(M { t, tb: f[20].parse().unwrap_or(0.0) });
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
        if t == 0 || !in_session(t) { continue; }
        let flag: i32 = f[10].parse().unwrap_or(0);
        let vol: f64 = f[8].parse().unwrap_or(0.0);
        if flag == 32 {
            let bo: i64 = f[14].parse().unwrap_or(0);
            let ao: i64 = f[13].parse().unwrap_or(0);
            if bo > 0 { out.push(Tr { t, vol, flag: 32 }); } else if ao > 0 { out.push(Tr { t, vol, flag: 0 }); }
        } else if flag == 66 {
            out.push(Tr { t, vol, flag: 66 });
        } else if flag == 83 {
            out.push(Tr { t, vol, flag: 83 });
        }
    }
    Ok(out)
}

fn one(code: &str, date: i64) -> Option<Out> {
    let mp = format!("/ssd_data/stock/{}/market_data/{}_{}_market_data.csv", date, code, date);
    let tp = format!("/ssd_data/stock/{}/transaction/{}_{}_transaction.csv", date, code, date);
    let mkt = load_m(&mp).ok()?;
    let tr = load_t(&tp).ok()?;
    if mkt.len() < 100 { return None; }
    let mut o = Out {
        code: code.into(), n_iv: 0, n_neg: 0, n_pos: 0, pos_sum: 0.0, neg_sum: 0.0, abs_sum: 0.0,
        dt_all: vec![0; 6], dt_neg: vec![0; 6],
        sec_all: vec![0; 60], sec_neg: vec![0; 60],
        hh_all: vec![0; 32], hh_neg: vec![0; 32],
    };
    let mut p = 0usize;
    for i in 1..mkt.len() {
        let (t0, t1) = (mkt[i - 1].t, mkt[i].t);
        let dt = t1 - t0;
        if dt <= 0 || dt > 60_000_000 { continue; }
        let mut cb = 0.0; let mut asell = 0.0;
        while p < tr.len() && tr[p].t <= t0 { p += 1; }
        let mut q = p;
        while q < tr.len() && tr[q].t <= t1 {
            if tr[q].flag == 32 { cb += tr[q].vol; } else if tr[q].flag == 83 { asell += tr[q].vol; }
            q += 1;
        }
        let nb = (mkt[i].tb - mkt[i - 1].tb) + cb + asell;
        let bin = if dt <= 3_000_000 { 0 } else if dt <= 4_000_000 { 1 } else if dt <= 5_000_000 { 2 }
                  else if dt <= 10_000_000 { 3 } else if dt <= 20_000_000 { 4 } else { 5 };
        let secs = lsec(t1) as usize;
        let secm = secs % 60;
        let hh = (secs / 1800).min(31);
        o.n_iv += 1;
        o.dt_all[bin] += 1; o.sec_all[secm] += 1; o.hh_all[hh] += 1;
        o.abs_sum += nb.abs();
        if nb > 0.0 { o.n_pos += 1; o.pos_sum += nb; } else if nb < 0.0 { o.neg_sum += -nb; }
        if nb < 0.0 {
            o.n_neg += 1;
            o.dt_neg[bin] += 1; o.sec_neg[secm] += 1; o.hh_neg[hh] += 1;
        }
    }
    Some(o)
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
    eprintln!("negcluster date={} codes={}", date, codes.len());
    let rows: Vec<Out> = codes.par_iter().filter_map(|c| one(c, date)).collect();
    for r in &rows { println!("{}", serde_json::to_string(r).unwrap()); }
    eprintln!("done rows={}", rows.len());
}
