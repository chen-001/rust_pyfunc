//! sandbox 探索：盘口前五档不平衡 x 点差宽窄 -> 之后 1 分钟收益（单日全市场，纯 std）
//! 构建: cd sandbox_play_spread && ~/.cargo/bin/cargo build --release
//! 运行: PLAY_THREADS=16 ./target/release/play_spread 20220819 20241231
mod fast_csv_reader;

use fast_csv_reader::{read_market_fast, MarketRecord};
use std::collections::HashSet;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

const TICK: f64 = 0.01;
const FWD_US: i64 = 60_000_000;
const MAX_GAP_US: i64 = 95_000_000;
const MIN_OBS: usize = 200;
const MIN_BUCKET: usize = 30;
const SYMBOL_MAP: &str = "/ssd_data/data/basic_info/symbol_map.csv";

#[derive(Clone, Copy)]
struct Obs {
    imb: f64,
    r_mid: f64,
    r_last: f64,
    spread_ticks: i32,
    spread_bps: f64,
    depth: f64,
    fwd_vol: f64,
    price: f64,
}

#[derive(Default, Clone, Copy)]
struct Sums {
    names: f64,
    obs: f64,
    icm: f64,
    icm2: f64,
    icl: f64,
    icl2: f64,
    lsp: f64,
    lsp2: f64,
    sp_bps: f64,
    sp_ticks: f64,
    absr: f64,
    depth: f64,
    fwd_vol: f64,
    price: f64,
}

fn read_symbol_map(path: &str) -> HashSet<String> {
    let txt = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("symbol_map {}: {}", path, e));
    let mut set = HashSet::new();
    for (k, line) in txt.lines().enumerate() {
        if k == 0 { continue; }
        if let Some(code) = line.split(',').next() {
            let c = code.trim();
            if c.len() == 6 && c.chars().all(|x| x.is_ascii_digit()) { set.insert(c.to_string()); }
        }
    }
    set
}

fn list_codes(date: i64, whitelist: &HashSet<String>) -> Vec<String> {
    let dir = format!("/ssd_data/stock/{}/market_data", date);
    let mut out = Vec::new();
    for e in std::fs::read_dir(&dir).unwrap_or_else(|e| panic!("read_dir {}: {}", dir, e)) {
        let name = e.unwrap().file_name().to_string_lossy().to_string();
        if !name.ends_with("_market_data.csv") { continue; }
        let code = name.split('_').next().unwrap_or("").to_string();
        if whitelist.contains(&code) { out.push(code); }
    }
    out.sort();
    out
}

/// 每只股票：有效快照 -> 60 秒后收益 + 前五档不平衡
fn per_stock(recs: &[MarketRecord]) -> Vec<Obs> {
    let n = recs.len();
    let mut valid: Vec<usize> = Vec::with_capacity(n);
    for i in 0..n {
        let r = &recs[i];
        if r.last_prc > 0.0 && r.ask_prcs[0] > 0.0 && r.bid_prcs[0] > 0.0 && r.ask_prcs[0] > r.bid_prcs[0] {
            valid.push(i);
        }
    }
    let mut out = Vec::with_capacity(valid.len());
    let mut j = 0usize;
    for (k, &i) in valid.iter().enumerate() {
        if j < k { j = k; }
        let t = recs[i].time_us;
        let sod = (t / 1_000_000).rem_euclid(86400);
        // 跨午休的 60 秒窗口丢掉（下午时段已平移到 11:30 之后，两边会直接相邻）
        if sod >= 11 * 3600 + 29 * 60 && sod < 11 * 3600 + 30 * 60 { continue; }
        while j < valid.len() && recs[valid[j]].time_us < t + FWD_US { j += 1; }
        if j >= valid.len() { break; }
        let fj = valid[j];
        if recs[fj].time_us - t > MAX_GAP_US { continue; }
        let r = &recs[i];
        let f = &recs[fj];
        let mid = 0.5 * (r.ask_prcs[0] + r.bid_prcs[0]);
        let fmid = 0.5 * (f.ask_prcs[0] + f.bid_prcs[0]);
        let bid5: f64 = r.bid_vols[..5].iter().sum();
        let ask5: f64 = r.ask_vols[..5].iter().sum();
        if bid5 + ask5 <= 0.0 { continue; }
        let sp_abs = r.ask_prcs[0] - r.bid_prcs[0];
        out.push(Obs {
            imb: (bid5 - ask5) / (bid5 + ask5),
            r_mid: fmid / mid - 1.0,
            r_last: f.last_prc / r.last_prc - 1.0,
            spread_ticks: (sp_abs / TICK).round().max(1.0) as i32,
            spread_bps: sp_abs / mid * 1e4,
            depth: bid5 + ask5,
            fwd_vol: (f.volume - r.volume).max(0.0),
            price: r.last_prc,
        });
    }
    out
}

fn rank(v: &[f64]) -> Vec<f64> {
    let n = v.len();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| v[a].partial_cmp(&v[b]).unwrap_or(std::cmp::Ordering::Equal));
    let mut r = vec![0.0f64; n];
    let mut i = 0usize;
    while i < n {
        let mut j = i;
        while j + 1 < n && v[idx[j + 1]] == v[idx[i]] { j += 1; }
        let avg = ((i + j) as f64) / 2.0 + 1.0;
        for k in i..=j { r[idx[k]] = avg; }
        i = j + 1;
    }
    r
}

fn pearson(x: &[f64], y: &[f64]) -> Option<f64> {
    let n = x.len() as f64;
    if n < 3.0 { return None; }
    let mx = x.iter().sum::<f64>() / n;
    let my = y.iter().sum::<f64>() / n;
    let mut sxy = 0.0; let mut sxx = 0.0; let mut syy = 0.0;
    for i in 0..x.len() {
        let dx = x[i] - mx; let dy = y[i] - my;
        sxy += dx * dy; sxx += dx * dx; syy += dy * dy;
    }
    if sxx <= 0.0 || syy <= 0.0 { return None; }
    Some(sxy / (sxx.sqrt() * syy.sqrt()))
}

fn spearman(x: &[f64], y: &[f64]) -> Option<f64> {
    let rx = rank(x);
    let ry = rank(y);
    pearson(&rx, &ry)
}

/// 按 x 排序，取上下各 30% 的 y 均值之差（bp）
fn top_bottom(x: &[f64], y: &[f64]) -> Option<f64> {
    let n = x.len();
    if n < 20 { return None; }
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| x[a].partial_cmp(&x[b]).unwrap_or(std::cmp::Ordering::Equal));
    let k = (n as f64 * 0.3).floor() as usize;
    if k == 0 { return None; }
    let lo: f64 = idx[..k].iter().map(|&i| y[i]).sum::<f64>() / k as f64;
    let hi: f64 = idx[n - k..].iter().map(|&i| y[i]).sum::<f64>() / k as f64;
    Some(hi - lo)
}

/// 把一只股票日的观测按 labels 分组，逐组算 IC / 多空差 / 描述量，累加进 res
fn add_stock_day(res: &mut [Sums], obs: &[Obs], labels: &[usize]) {
    for g in 0..res.len() {
        let idxs: Vec<usize> = (0..obs.len()).filter(|&k| labels[k] == g).collect();
        if idxs.len() < MIN_BUCKET { continue; }
        let x: Vec<f64> = idxs.iter().map(|&k| obs[k].imb).collect();
        let ym: Vec<f64> = idxs.iter().map(|&k| obs[k].r_mid * 1e4).collect();
        let yl: Vec<f64> = idxs.iter().map(|&k| obs[k].r_last * 1e4).collect();
        let s = &mut res[g];
        s.names += 1.0;
        s.obs += idxs.len() as f64;
        if let Some(v) = spearman(&x, &ym) { s.icm += v; s.icm2 += v * v; }
        if let Some(v) = spearman(&x, &yl) { s.icl += v; s.icl2 += v * v; }
        if let Some(v) = top_bottom(&x, &ym) { s.lsp += v; s.lsp2 += v * v; }
        for &k in &idxs {
            let o = &obs[k];
            s.sp_bps += o.spread_bps;
            s.sp_ticks += o.spread_ticks as f64;
            s.absr += o.r_mid.abs() * 1e4;
            s.depth += o.depth;
            s.fwd_vol += o.fwd_vol;
            s.price += o.price;
        }
    }
}

fn mean_se(sum: f64, sum2: f64, n: f64) -> (f64, f64) {
    if n < 1.0 { return (f64::NAN, f64::NAN); }
    let m = sum / n;
    let var = (sum2 / n - m * m).max(0.0);
    (m, (var / n).sqrt())
}

fn json_group(name: &str, s: &Sums) -> String {
    let (icm, icm_se) = mean_se(s.icm, s.icm2, s.names);
    let (icl, icl_se) = mean_se(s.icl, s.icl2, s.names);
    let (lsp, lsp_se) = mean_se(s.lsp, s.lsp2, s.names);
    let d = s.obs.max(1.0);
    let first = name.is_empty();
    format!(
        "{}{{\"group\":\"{}\",\"stock_days\":{:.0},\"obs\":{:.0},\"ic_mid\":{:.5},\"ic_mid_se\":{:.5},\"ic_last\":{:.5},\"ic_last_se\":{:.5},\"top_bot_bp\":{:.4},\"top_bot_se\":{:.4},\"spread_ticks\":{:.3},\"spread_bps\":{:.2},\"absr_bp\":{:.2},\"depth5\":{:.0},\"fwd_vol\":{:.0},\"price\":{:.2}}}",
        if first { "" } else { "," },
        name, s.names, s.obs, icm, icm_se, icl, icl_se, lsp, lsp_se,
        s.sp_ticks / d, s.sp_bps / d, s.absr / d, s.depth / d, s.fwd_vol / d, s.price / d
    )
}

fn tick_bucket(t: i32) -> usize { (t.clamp(1, 4) - 1) as usize }

fn run(date: i64) {
    let wl = read_symbol_map(SYMBOL_MAP);
    let codes = list_codes(date, &wl);
    eprintln!("[{}] codes in whitelist: {}", date, codes.len());
    let nt: usize = std::env::var("PLAY_THREADS").ok().and_then(|s| s.parse().ok()).unwrap_or(16);
    let next = AtomicUsize::new(0);
    let collected: Mutex<Vec<(String, Vec<Obs>)>> = Mutex::new(Vec::new());
    let t0 = std::time::Instant::now();
    std::thread::scope(|scope| {
        for _ in 0..nt {
            scope.spawn(|| loop {
                let k = next.fetch_add(1, Ordering::Relaxed);
                if k >= codes.len() { break; }
                let code = &codes[k];
                if let Ok(recs) = read_market_fast(code, date) {
                    let obs = per_stock(&recs);
                    if obs.len() >= MIN_OBS { collected.lock().unwrap().push((code.clone(), obs)); }
                }
            });
        }
    });
    let mut data = collected.into_inner().unwrap();
    data.sort_by(|a, b| a.0.cmp(&b.0));
    eprintln!("[{}] stock-days kept: {} in {:.1}s", date, data.len(), t0.elapsed().as_secs_f64());

    let n_tick = 4usize;
    let mut res_tick = vec![Sums::default(); n_tick];
    let mut res_all = vec![Sums::default(); 1];
    let mut res_sdgroup = vec![Sums::default(); 3];   // 股票日中位点差 1 / 2 / >=3 档
    let mut res_price = vec![Sums::default(); 4];     // 价格 <5 / 5-20 / 20-100 / >=100
    let mut total_obs = 0f64;

    for (_code, obs) in &data {
        let labels: Vec<usize> = obs.iter().map(|o| tick_bucket(o.spread_ticks)).collect();
        add_stock_day(&mut res_tick, obs, &labels);
        add_stock_day(&mut res_all, obs, &vec![0usize; obs.len()]);

        let mut ticks: Vec<i32> = obs.iter().map(|o| o.spread_ticks).collect();
        ticks.sort_unstable();
        let med_tick = ticks[ticks.len() / 2];
        let g = if med_tick <= 1 { 0 } else if med_tick == 2 { 1 } else { 2 };
        add_stock_day(&mut res_sdgroup, obs, &vec![g; obs.len()]);

        let mut prs: Vec<f64> = obs.iter().map(|o| o.price).collect();
        prs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let med_prc = prs[prs.len() / 2];
        let gp = if med_prc < 5.0 { 0 } else if med_prc < 20.0 { 1 } else if med_prc < 100.0 { 2 } else { 3 };
        add_stock_day(&mut res_price, obs, &vec![gp; obs.len()]);
        total_obs += obs.len() as f64;
    }

    let tick_names = ["1tick", "2ticks", "3ticks", "4+ticks"];
    let sd_names = ["sd_med1tick", "sd_med2ticks", "sd_med3+ticks"];
    let price_names = ["px_lt5", "px_5to20", "px_20to100", "px_ge100"];
    let mut parts: Vec<String> = Vec::new();
    for (i, n) in tick_names.iter().enumerate() { parts.push(json_group(n, &res_tick[i])); }
    for (i, n) in sd_names.iter().enumerate() { parts.push(json_group(n, &res_sdgroup[i])); }
    for (i, n) in price_names.iter().enumerate() { parts.push(json_group(n, &res_price[i])); }
    parts.push(json_group("all", &res_all[0]));
    println!("{{\"date\":{},\"stock_days\":{},\"obs\":{:.0},\"groups\":[{}]}}",
             date, data.len(), total_obs, parts.join(","));
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    assert!(!args.is_empty(), "usage: play_spread <date> [date...]");
    for a in &args { run(a.parse::<i64>().expect("date must be YYYYMMDD")); }
}
