//! sandbox 探索：撤单侧的量，单日全市场横截面
//! ① 从不成交就撤的比例（按笔数、按股数）
//! ② 成交过又撤：拆成「首次成交是主动成交」和「首次成交是被动成交」两组，各自算到撤单的间隔
//! ③ 撤买量/撤卖量不平衡、编号单位的年龄
//! 构建: cd sandbox_play_spread && ~/.cargo/bin/cargo build --release --bin play_cancel2
//! 运行: PLAY_THREADS=16 ./target/release/play_cancel2 20220819 20241231
mod fast_csv_reader;

use fast_csv_reader::{read_trade_fast_retreat, TradeRecord};
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

const MIN_TRADES: usize = 1000;
const SYMBOL_MAP: &str = "/ssd_data/data/basic_info/symbol_map.csv";
const SESSION_START: i64 = 9 * 3600 + 30 * 60;
const SESSION_END: i64 = 13 * 3600 + 32 * 60;

#[derive(Clone, Copy, Default)]
struct M {
    price: f64,
    turnover: f64,
    trades: f64,
    cancels: f64,
    imb_cancel: f64,
    never_frac: f64,
    ratio_id: f64,
    med_age_c: f64,
    med_age_w: f64,
    n_cancel: f64,
    n_partial: f64,
    med_gap_s: f64,
    // 新增：按主动/被动拆，以及按股数
    n_pass: f64,
    n_act: f64,
    med_gap_pass_s: f64,
    med_gap_act_s: f64,
    vol_cancel: f64,
    vol_never: f64,
    vol_pass: f64,
    vol_act: f64,
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

fn med(v: &mut Vec<f64>) -> f64 {
    if v.is_empty() { return f64::NAN; }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn pct(v: &mut Vec<f64>, q: f64) -> f64 {
    if v.is_empty() { return f64::NAN; }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[(((v.len() - 1) as f64) * q).round() as usize]
}

fn note(m: &mut HashMap<i64, (i64, bool)>, id: i64, t: i64, is_agg: bool) {
    match m.get(&id) {
        Some(&(t0, _)) if t0 <= t => {}
        _ => { m.insert(id, (t, is_agg)); }
    }
}

fn per_stock(code: &str, date: i64) -> Option<M> {
    let recs: Vec<TradeRecord> = read_trade_fast_retreat(code, date).ok()?;
    let mut seen: HashSet<i64> = HashSet::new();
    let mut cancels: Vec<(i64, i64, f64, bool)> = Vec::new();
    let mut fills: Vec<(i64, i64)> = Vec::new();
    let mut agg: Vec<(i64, i64)> = Vec::new();
    let mut all_first: HashMap<i64, (i64, bool)> = HashMap::new(); // 编号 -> (首次出现时间, 是否主动)
    let mut prices: Vec<f64> = Vec::new();
    let mut turnover = 0.0f64;
    for r in &recs {
        let sod = (r.time_us / 1_000_000).rem_euclid(86400);
        if sod < SESSION_START || sod > SESSION_END { continue; }
        match r.flag {
            66 | 83 => {
                let (a, p) = if r.flag == 66 { (r.bid_order, r.ask_order) } else { (r.ask_order, r.bid_order) };
                if a > 0 { seen.insert(a); agg.push((a, r.time_us)); note(&mut all_first, a, r.time_us, true); }
                if p > 0 { seen.insert(p); fills.push((p, r.time_us)); note(&mut all_first, p, r.time_us, false); }
                if r.price > 0.0 { prices.push(r.price); }
                turnover += r.turnover;
            }
            32 => {
                let (cid, is_buy) = if r.bid_order > 0 { (r.bid_order, true) } else { (r.ask_order, false) };
                if cid > 0 { cancels.push((cid, r.time_us, r.volume, is_buy)); }
            }
            _ => {}
        }
    }
    if agg.len() < MIN_TRADES || cancels.is_empty() || fills.is_empty() { return None; }
    let mut ai = 0usize;
    let mut age_c: Vec<f64> = Vec::with_capacity(cancels.len());
    let mut buy_v = 0.0f64;
    let mut sell_v = 0.0f64;
    let mut never = 0f64;
    let (mut vol_all, mut vol_never, mut vol_pass, mut vol_act) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    let mut gaps: Vec<f64> = Vec::new();
    let mut gaps_pass: Vec<f64> = Vec::new();
    let mut gaps_act: Vec<f64> = Vec::new();
    for &(cid, t, vol, is_buy) in &cancels {
        while ai < agg.len() && agg[ai].1 <= t { ai += 1; }
        let cur = if ai == 0 { agg[0].0 } else { agg[ai - 1].0 };
        age_c.push((cur - cid).max(0) as f64);
        if is_buy { buy_v += vol; } else { sell_v += vol; }
        vol_all += vol;
        match all_first.get(&cid) {
            None => { never += 1.0; vol_never += vol; }
            Some(&(t0, is_agg)) => {
                if t > t0 {
                    let g = (t - t0) as f64 / 1e6;
                    gaps.push(g);
                    if is_agg { gaps_act.push(g); vol_act += vol; } else { gaps_pass.push(g); vol_pass += vol; }
                } else {
                    // 同一时刻出现，方向不明，股数记到被动一侧的池子里不影响笔数口径
                    if is_agg { vol_act += vol; } else { vol_pass += vol; }
                }
            }
        }
    }
    let mut first_time: HashMap<i64, i64> = HashMap::new();
    for &(pid, t) in &fills {
        let e = first_time.entry(pid).or_insert(t);
        if t < *e { *e = t; }
    }
    let mut ai2 = 0usize;
    let mut age_w: Vec<f64> = Vec::new();
    for &(pid, t) in &fills {
        if first_time.get(&pid) != Some(&t) { continue; }
        while ai2 < agg.len() && agg[ai2].1 <= t { ai2 += 1; }
        let cur = if ai2 == 0 { agg[0].0 } else { agg[ai2 - 1].0 };
        age_w.push((cur - pid).max(0) as f64);
    }
    let n_partial = gaps.len() as f64;
    let mut g2 = gaps;
    let mut gp = gaps_pass.clone();
    let mut ga = gaps_act.clone();
    let mut ac = age_c;
    let mut aw = age_w;
    let med_age_c = med(&mut ac);
    let med_age_w = med(&mut aw);
    let tot = buy_v + sell_v;
    Some(M {
        price: med(&mut prices),
        turnover,
        trades: first_time.len() as f64,
        cancels: cancels.len() as f64,
        imb_cancel: if tot > 0.0 { (buy_v - sell_v) / tot } else { f64::NAN },
        never_frac: never / cancels.len() as f64,
        ratio_id: med_age_c / med_age_w.max(1.0),
        med_age_c,
        med_age_w,
        n_cancel: cancels.len() as f64,
        n_partial,
        med_gap_s: if g2.is_empty() { f64::NAN } else { med(&mut g2) },
        n_pass: gaps_pass.len() as f64,
        n_act: gaps_act.len() as f64,
        med_gap_pass_s: if gp.is_empty() { f64::NAN } else { med(&mut gp) },
        med_gap_act_s: if ga.is_empty() { f64::NAN } else { med(&mut ga) },
        vol_cancel: vol_all,
        vol_never,
        vol_pass,
        vol_act,
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
    let (mut sxy, mut sxx, mut syy) = (0.0, 0.0, 0.0);
    for i in 0..x.len() { let (dx, dy) = (rx[i] - mx, ry[i] - my); sxy += dx * dy; sxx += dx * dx; syy += dy * dy; }
    if sxx <= 0.0 || syy <= 0.0 { return f64::NAN; }
    sxy / (sxx.sqrt() * syy.sqrt())
}

/// 最小二乘残差（含常数项），返回 (残差, R²)
fn ols_resid(y: &[f64], xs: &[&[f64]]) -> (Vec<f64>, f64) {
    let n = y.len();
    let k = xs.len() + 1;
    let mut a = vec![vec![0.0f64; k + 1]; k];
    for i in 0..k {
        for j in 0..k {
            let mut s = 0.0;
            for t in 0..n {
                let xi = if i == 0 { 1.0 } else { xs[i - 1][t] };
                let xj = if j == 0 { 1.0 } else { xs[j - 1][t] };
                s += xi * xj;
            }
            a[i][j] = s;
        }
        let mut s = 0.0;
        for t in 0..n {
            let xi = if i == 0 { 1.0 } else { xs[i - 1][t] };
            s += xi * y[t];
        }
        a[i][k] = s;
    }
    for c in 0..k {
        let mut piv = c;
        for r in c + 1..k { if a[r][c].abs() > a[piv][c].abs() { piv = r; } }
        a.swap(c, piv);
        for r in c + 1..k {
            let f = a[r][c] / a[c][c];
            for cc in c..=k { a[r][cc] -= f * a[c][cc]; }
        }
    }
    let mut beta = vec![0.0f64; k];
    for c in (0..k).rev() {
        let mut s = a[c][k];
        for cc in c + 1..k { s -= a[c][cc] * beta[cc]; }
        beta[c] = s / a[c][c];
    }
    let my = y.iter().sum::<f64>() / n as f64;
    let (mut ss_res, mut ss_tot) = (0.0, 0.0);
    let mut out = vec![0.0f64; n];
    for t in 0..n {
        let mut p = beta[0];
        for i in 0..xs.len() { p += beta[i + 1] * xs[i][t]; }
        out[t] = y[t] - p;
        ss_res += out[t] * out[t];
        ss_tot += (y[t] - my) * (y[t] - my);
    }
    (out, 1.0 - ss_res / ss_tot.max(1e-12))
}

fn summarize(name: &str, vals: Vec<f64>) -> String {
    let mut a = vals.clone();
    let mut b = vals;
    let (p10, p50, p90, mean) = (pct(&mut a, 0.10), pct(&mut b, 0.50), pct(&mut b, 0.90), b.iter().sum::<f64>() / b.len().max(1) as f64);
    format!("\"{}\":{{\"n\":{},\"mean\":{:.4},\"p10\":{:.4},\"p50\":{:.4},\"p90\":{:.4}}}", name, a.len(), mean, p10, p50, p90)
}

fn run(dates: &[i64]) {
    let wl = read_symbol_map(SYMBOL_MAP);
    let mut per_date: Vec<(i64, Vec<(String, M)>)> = Vec::new();
    for &date in dates {
        let codes = list_codes(date, &wl);
        let nt: usize = std::env::var("PLAY_THREADS").ok().and_then(|s| s.parse().ok()).unwrap_or(16);
        let next = AtomicUsize::new(0);
        let acc: Mutex<Vec<(String, M)>> = Mutex::new(Vec::new());
        let t0 = std::time::Instant::now();
        std::thread::scope(|sc| {
            for _ in 0..nt {
                sc.spawn(|| loop {
                    let k = next.fetch_add(1, Ordering::Relaxed);
                    if k >= codes.len() { break; }
                    if let Some(m) = per_stock(&codes[k], date) { acc.lock().unwrap().push((codes[k].clone(), m)); }
                });
            }
        });
        let mut v = acc.into_inner().unwrap();
        v.sort_by(|a, b| a.0.cmp(&b.0));
        eprintln!("[{}] stocks: {} in {:.1}s", date, v.len(), t0.elapsed().as_secs_f64());

        let ok: Vec<&(String, M)> = v.iter().filter(|(_, m)| m.imb_cancel.is_finite() && m.ratio_id.is_finite()).collect();
        let imb: Vec<f64> = ok.iter().map(|x| x.1.imb_cancel).collect();
        let nf: Vec<f64> = ok.iter().map(|x| x.1.never_frac).collect();
        let ri: Vec<f64> = ok.iter().map(|x| x.1.ratio_id.ln()).collect();
        let px: Vec<f64> = ok.iter().map(|x| x.1.price.ln()).collect();
        let tv: Vec<f64> = ok.iter().map(|x| (x.1.turnover + 1.0).ln()).collect();
        let cs: Vec<f64> = ok.iter().map(|x| (x.1.cancels / x.1.trades).ln()).collect();

        let tot_c: f64 = v.iter().map(|x| x.1.n_cancel).sum();
        let tot_p: f64 = v.iter().map(|x| x.1.n_partial).sum();
        let tot_pass: f64 = v.iter().map(|x| x.1.n_pass).sum();
        let tot_act: f64 = v.iter().map(|x| x.1.n_act).sum();
        let vol_c: f64 = v.iter().map(|x| x.1.vol_cancel).sum();
        let vol_n: f64 = v.iter().map(|x| x.1.vol_never).sum();
        let vol_p: f64 = v.iter().map(|x| x.1.vol_pass).sum();
        let vol_a: f64 = v.iter().map(|x| x.1.vol_act).sum();
        let mut mps = v.iter().map(|x| x.1.med_gap_pass_s).filter(|x| x.is_finite()).collect::<Vec<f64>>();
        let mut mas = v.iter().map(|x| x.1.med_gap_act_s).filter(|x| x.is_finite()).collect::<Vec<f64>>();
        let mut allg = v.iter().map(|x| x.1.med_gap_s).filter(|x| x.is_finite()).collect::<Vec<f64>>();
        println!("{{\"date\":{},\"stocks\":{},\"cancel_rows\":{:.0},\"partial_rows\":{:.0},\"partial_frac_rows\":{:.4},\"partial_vol_share\":{:.4},\"never_vol_share\":{:.4},\"partial_pass_rows\":{:.0},\"partial_act_rows\":{:.0},\"act_share_rows\":{:.3},\"med_gap_all_s\":{:.2},\"med_gap_pass_s\":{:.2},\"med_gap_act_s\":{:.2},{},{},{},\"corr\":{{\"imb_vs_price\":{:.3},\"imb_vs_turnover\":{:.3},\"never_vs_price\":{:.3},\"never_vs_turnover\":{:.3},\"never_vs_cancelshare\":{:.3},\"ratioid_vs_turnover\":{:.3}}}}}",
            date, v.len(), tot_c, tot_p, tot_p / tot_c.max(1.0), (vol_p + vol_a) / vol_c.max(1.0), vol_n / vol_c.max(1.0),
            tot_pass, tot_act, tot_act / (tot_pass + tot_act).max(1.0),
            med(&mut allg), med(&mut mps), med(&mut mas),
            summarize("imb_cancel", imb.clone()), summarize("never_frac", nf.clone()), summarize("ratio_id", ri.iter().map(|x| x.exp()).collect()),
            spearman(&imb, &px), spearman(&imb, &tv), spearman(&nf, &px), spearman(&nf, &tv), spearman(&nf, &cs), spearman(&ri, &tv));
        per_date.push((date, v));
    }
    if per_date.len() == 2 {
        let (_, v0) = &per_date[0];
        let (_, v1) = &per_date[1];
        let m1: HashMap<&String, &M> = v1.iter().map(|(c, m)| (c, m)).collect();
        let (mut a_imb, mut b_imb, mut a_nf, mut b_nf) = (vec![], vec![], vec![], vec![]);
        let (mut r0, mut r1) = (vec![], vec![]);
        let (mut t0, mut t1) = (vec![], vec![]);
        let (mut p0, mut p1) = (vec![], vec![]);
        let (mut c0, mut c1) = (vec![], vec![]);
        for (c, m) in v0.iter() {
            if let Some(n) = m1.get(c) {
                if m.imb_cancel.is_finite() && n.imb_cancel.is_finite() && m.ratio_id > 0.0 && n.ratio_id > 0.0 {
                    a_imb.push(m.imb_cancel); b_imb.push(n.imb_cancel);
                    a_nf.push(m.never_frac); b_nf.push(n.never_frac);
                    r0.push(m.ratio_id.ln()); r1.push(n.ratio_id.ln());
                    t0.push((m.turnover + 1.0).ln()); t1.push((n.turnover + 1.0).ln());
                    p0.push(m.price.ln()); p1.push(n.price.ln());
                    c0.push((m.cancels / m.trades).ln()); c1.push((n.cancels / n.trades).ln());
                }
            }
        }
        let mean = |v: &Vec<f64>| v.iter().sum::<f64>() / v.len() as f64;
        let sd = |v: &Vec<f64>| { let m = mean(v); (v.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / v.len() as f64).sqrt() };
        let (e0, r2a) = ols_resid(&r0, &[&t0]);
        let (e1, r2b) = ols_resid(&r1, &[&t1]);
        let (f0, r2c) = ols_resid(&r0, &[&t0, &p0, &c0]);
        let (f1, r2d) = ols_resid(&r1, &[&t1, &p1, &c1]);
        let (nf_r0, _) = ols_resid(&a_nf, &[&p0, &t0]);
        let (nf_r1, _) = ols_resid(&b_nf, &[&p1, &t1]);
        println!("{{\"never_frac\":{{\"n\":{},\"raw_cross_day\":{:.3},\"resid_cross_day\":{:.3},\"raw_vs_price\":{:.3},\"resid_vs_price\":{:.3},\"raw_vs_turnover\":{:.3},\"resid_vs_turnover\":{:.3}}}}}",
            a_nf.len(),
            spearman(&a_nf, &b_nf), spearman(&nf_r0, &nf_r1),
            (spearman(&a_nf, &p0) + spearman(&b_nf, &p1)) / 2.0,
            (spearman(&nf_r0, &p0) + spearman(&nf_r1, &p1)) / 2.0,
            (spearman(&a_nf, &t0) + spearman(&b_nf, &t1)) / 2.0,
            (spearman(&nf_r0, &t0) + spearman(&nf_r1, &t1)) / 2.0);
        println!("{{\"cross_day\":{{\"n\":{},\"imb_cancel\":{:.3},\"never_frac\":{:.3},\"ratio_raw\":{:.3},\"resid_turnover\":{:.3},\"resid_turnover_price_cancel\":{:.3},\"r2_turnover\":{:.3},\"r2_full\":{:.3}}}}}",
            a_imb.len(), spearman(&a_imb, &b_imb), spearman(&a_nf, &b_nf), spearman(&r0, &r1),
            spearman(&e0, &e1), spearman(&f0, &f1), (r2a + r2b) / 2.0, (r2c + r2d) / 2.0);
    }
}

fn main() {
    let args: Vec<i64> = std::env::args().skip(1).map(|a| a.parse().unwrap()).collect();
    assert!(!args.is_empty(), "usage: play_cancel2 <date> [date...]");
    run(&args);
}
