//! 中间价自相关随采样尺度的曲线（rho_sandbox v2）
//!
//! 序参量 = 中间价 = (卖一价 + 买一价) / 2；控制参数 = 该快照前 60 秒成交量。
//! 分位在窗口内部算。尺度 = 3 / 10 / 30 / 60 秒，对应相隔 1 / 3 / 10 / 20 个快照。
//! 每个十分位报四个尺度上的自相关，外加 sigma。
//! 目的：看自相关在哪个尺度转正，转正后的衰减速度才是恢复速度；所有尺度都负就不必再做。
//!
//! 构建: cd sandbox_cancel_flow && cargo build --release
//! 运行: ./target/release/rho 20220819 000001,600519,...

use rayon::prelude::*;
use serde::Serialize;

const LAGS: [usize; 4] = [1, 3, 10, 20];
const LAG_SECS: [i64; 4] = [3, 9, 30, 60];

#[derive(Clone, Copy, Default)]
struct M { t: i64, ask: f64, bid: f64 }
#[derive(Clone, Copy, Default)]
struct Tr { t: i64, vol: f64 }

fn lsec(us: i64) -> i64 { ((us / 1_000_000) + 8 * 3600).rem_euclid(86400) }
fn in_flat(us: i64) -> bool {
    let s = lsec(us);
    (s >= 10 * 3600 && s < 11 * 3600) || (s >= 13 * 3600 + 1800 && s < 14 * 3600 + 1800)
}
fn in_day(us: i64) -> bool {
    let s = lsec(us);
    (s >= 9 * 3600 + 1800 && s <= 11 * 3600 + 1800) || (s >= 13 * 3600 && s <= 15 * 3600)
}

fn load_m(path: &str) -> std::io::Result<Vec<M>> {
    let c = std::fs::read_to_string(path)?;
    let mut out = Vec::with_capacity(c.len() / 600);
    for line in c.lines().skip(1) {
        let f: Vec<&str> = line.split(',').collect();
        if f.len() < 25 { continue; }
        let t: i64 = f[4].parse().unwrap_or(0);
        if t == 0 { continue; }
        let ask: f64 = f[21].parse().unwrap_or(0.0);
        let bid: f64 = f[23].parse().unwrap_or(0.0);
        if ask <= 0.0 || bid <= 0.0 { continue; }
        out.push(M { t, ask, bid });
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
        out.push(Tr { t, vol: f[8].parse().unwrap_or(0.0) });
    }
    Ok(out)
}

#[derive(Serialize)]
struct Dec { d: usize, n: usize, n_pairs: usize, mean_strength: f64, rho: Vec<f64>, sigma: f64 }
#[derive(Serialize)]
struct Win { name: String, n_snap: usize, deciles: Vec<Dec> }
#[derive(Serialize)]
struct Out { code: String, wins: Vec<Win> }

fn trailing_strength(mkt: &[M], tr: &[Tr], look_us: i64) -> Vec<f64> {
    let mut out = vec![0.0; mkt.len()];
    let (mut p, mut q) = (0usize, 0usize);
    let mut run = 0.0;
    for i in 0..mkt.len() {
        let t = mkt[i].t;
        while p < tr.len() && tr[p].t <= t - look_us { run -= tr[p].vol; p += 1; }
        while q < tr.len() && tr[q].t <= t { run += tr[q].vol; q += 1; }
        out[i] = run;
    }
    out
}

fn quantile(vals: &[f64], q: f64) -> f64 {
    if vals.is_empty() { return 0.0; }
    let mut v = vals.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[(((v.len() as f64 - 1.0) * q).round() as usize).min(v.len() - 1)]
}

fn corr(xs: &[f64], ys: &[f64]) -> f64 {
    if xs.len() < 30 { return f64::NAN; }
    let n = xs.len() as f64;
    let mx = xs.iter().sum::<f64>() / n;
    let my = ys.iter().sum::<f64>() / n;
    let (mut sxy, mut sxx, mut syy) = (0.0, 0.0, 0.0);
    for k in 0..xs.len() {
        let (a, b) = (xs[k] - mx, ys[k] - my);
        sxy += a * b; sxx += a * a; syy += b * b;
    }
    if sxx > 0.0 && syy > 0.0 { sxy / (sxx.sqrt() * syy.sqrt()) } else { f64::NAN }
}

fn analyse(mkt: &[M], strength: &[f64], filter: fn(i64) -> bool, name: &str) -> Win {
    let idx: Vec<usize> = (0..mkt.len()).filter(|&i| filter(mkt[i].t)).collect();
    if idx.len() < 200 { return Win { name: name.into(), n_snap: idx.len(), deciles: vec![] }; }
    let s: Vec<f64> = idx.iter().map(|&i| strength[i]).collect();
    let mut cuts = [0f64; 9];
    for k in 0..9 { cuts[k] = quantile(&s, (k + 1) as f64 / 10.0); }
    let dec_of: Vec<usize> = s.iter().map(|&v| { let mut d = 0; while d < 9 && v > cuts[d] { d += 1; } d }).collect();

    let mut ret = vec![f64::NAN; idx.len()];
    for w in 1..idx.len() {
        let (a, b) = (mkt[idx[w - 1]], mkt[idx[w]]);
        if b.t - a.t > 60_000_000 { continue; }
        let (ma, mb) = ((a.ask + a.bid) / 2.0, (b.ask + b.bid) / 2.0);
        if ma > 0.0 && mb > 0.0 { ret[w] = (mb / ma).ln(); }
    }
    // 每一格到下一格的间隔是否连续（用来挡住跨段的错配）
    let mut contig = vec![true; idx.len()];
    for w in 1..idx.len() { if mkt[idx[w]].t - mkt[idx[w - 1]].t > 60_000_000 { contig[w] = false; } }

    let mut deciles = Vec::with_capacity(10);
    for d in 0..10 {
        let mut n = 0usize; let mut sw = 0.0; let mut ss = 0.0;
        let mut xs: Vec<Vec<f64>> = (0..4).map(|_| Vec::new()).collect();
        let mut ys: Vec<Vec<f64>> = (0..4).map(|_| Vec::new()).collect();
        for i in 0..idx.len() {
            if dec_of[i] != d { continue; }
            n += 1; sw += s[i];
            if ret[i].is_finite() { ss += ret[i] * ret[i]; }
            for (li, &L) in LAGS.iter().enumerate() {
                // 只按起点所在分位取条件，终点不必同档，否则长尺度上配对太少
                if i + L >= idx.len() { continue; }
                if !ret[i].is_finite() || !ret[i + L].is_finite() { continue; }
                let mut ok = true;
                for k in i + 1..=i + L { if !contig[k] { ok = false; break; } }
                if !ok { continue; }
                xs[li].push(ret[i]); ys[li].push(ret[i + L]);
            }
        }
        let rho: Vec<f64> = (0..4).map(|li| corr(&xs[li], &ys[li])).collect();
        let np = if xs[0].is_empty() { 0 } else { xs.iter().map(|v| v.len()).max().unwrap_or(0) };
        deciles.push(Dec {
            d, n, n_pairs: np,
            mean_strength: if n > 0 { sw / n as f64 } else { 0.0 },
            rho,
            sigma: if n > 1 { (ss / n as f64).sqrt() } else { f64::NAN },
        });
    }
    Win { name: name.into(), n_snap: idx.len(), deciles }
}

fn one(code: &str, date: i64) -> Option<Out> {
    let mp = format!("/ssd_data/stock/{}/market_data/{}_{}_market_data.csv", date, code, date);
    let tp = format!("/ssd_data/stock/{}/transaction/{}_{}_transaction.csv", date, code, date);
    let mkt = load_m(&mp).ok()?;
    let tr = load_t(&tp).ok()?;
    if mkt.len() < 200 { return None; }
    let st = trailing_strength(&mkt, &tr, 60_000_000);
    Some(Out { code: code.into(), wins: vec![analyse(&mkt, &st, in_flat, "flat"), analyse(&mkt, &st, in_day, "allday")] })
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let date: i64 = a.get(1).and_then(|s| s.parse().ok()).unwrap_or(20220819);
    let codes: Vec<String> = a.get(2).map(|s| s.split(',').map(|x| x.trim().to_string()).collect())
        .unwrap_or_else(|| vec!["000001".into()]);
    let mut out: Vec<Out> = codes.par_iter().filter_map(|c| one(c, date)).collect();
    out.sort_by(|x, y| x.code.cmp(&y.code));
    eprintln!("rho_sandbox v2 date={} stocks={} lags_secs={:?}", date, out.len(), LAG_SECS);
    println!("{}", serde_json::to_string(&out).unwrap());
}
