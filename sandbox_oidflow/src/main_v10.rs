//! v10：价位级「耗尽—回补」机制 + 价格阶梯几何（sandbox）。
//! 与库里现有盘口工作的区别：全部按**价位**索引，不按档号。十档快照在价格跳档时，
//! 同一档号在不同时刻对应不同价位，按档号聚合会把不同价位混成一团，回补也就无从谈起。
//! 对象：每个 (价位, 侧) 的库存轨迹。事件 = 库存降到「安静水平」的 25% 以下 → 恢复到 50% 以上。
//! 每次事件带：掉下去的幅度、掉多久、回补比例、回补后又掉下去的节拍、事件在日内的位置。
//! 加工层：阈值谱 K=0.3/0.4/0.6/0.8/1.0 各出一遍；衰减指数拟合；霍克斯自激（衰减常数 + 分支比）；
//! 回补比例对时间位置的 1D 回归；跨事件间隔的 LZ 与熵；价格阶梯的空档宽度与孤岛计数。
//! 构建: cd /home/chenzongwei/rust_pyfunc/sandbox_oidflow && cargo build --release
//! 运行: ./target/release/oidflow_sandbox 20220819 > out_v10_20220819.json
mod fast_csv_reader;
mod v9_stats;
use rayon::prelude::*;
use serde::Serialize;
use std::collections::HashMap;
use v9_stats::*;

#[derive(Serialize)]
struct Out {
    names: Vec<String>,
    codes: Vec<String>,
    vals: Vec<Vec<f32>>,
    diag: Vec<Vec<f32>>,
}

fn stock_codes_for_day(
    dir: &std::path::Path,
    symbol_map: &std::path::Path,
    date: i64,
    subdir: &str,
) -> std::io::Result<Vec<String>> {
    use std::io::{Error, ErrorKind};
    let invalid = |e: &str| Error::new(ErrorKind::InvalidData, e.to_string());
    let text = std::fs::read_to_string(symbol_map).map_err(|e| invalid(&e.to_string()))?;
    let mut lines = text.lines();
    let header = lines.next().ok_or_else(|| invalid("股票表为空"))?;
    let col = header
        .split(',')
        .position(|h| h.trim() == "symbol")
        .ok_or_else(|| invalid("股票表缺少 symbol 列"))?;
    let mut stocks = std::collections::BTreeSet::new();
    for line in lines {
        let code = line.split(',').nth(col).unwrap_or("").trim();
        if code.is_empty() { continue; }
        if code.len() > 6 || !code.bytes().all(|b| b.is_ascii_digit()) {
            return Err(invalid("股票表含无效代码"));
        }
        stocks.insert(format!("{code:0>6}"));
    }
    if stocks.is_empty() || !dir.is_dir() {
        return Err(invalid("股票白名单为空或原始数据目录不存在"));
    }
    Ok(stocks.into_iter()
        .filter(|code| dir.join(format!("{code}_{date}_{subdir}.csv")).is_file())
        .collect())
}

fn list_codes(date: i64) -> std::io::Result<Vec<String>> {
    let root = std::env::var("RUST_PYFUNC_LEVEL2_ROOT").unwrap_or("/ssd_data/stock".into());
    let basic =
        std::env::var("RUST_PYFUNC_BASIC_INFO_DIR").unwrap_or("/ssd_data/data/basic_info".into());
    let dir = std::path::Path::new(&root).join(date.to_string()).join("market_data");
    let map = std::path::Path::new(&basic).join("symbol_map.csv");
    stock_codes_for_day(&dir, &map, date, "market_data")
}

#[derive(Clone, Copy, Default)]
struct Snap {
    t: f64,
    bid_p: [f64; 10],
    bid_v: [f64; 10],
    ask_p: [f64; 10],
    ask_v: [f64; 10],
    mid: f64,
}

fn read_snaps(code: &str, date: i64) -> Option<Vec<Snap>> {
    let root = std::env::var("RUST_PYFUNC_LEVEL2_ROOT").unwrap_or("/ssd_data/stock".into());
    let path = format!("{}/{}/market_data/{}_{}_market_data.csv", root, date, code, date);
    let text = std::fs::read_to_string(&path).ok()?;
    let mut out: Vec<Snap> = Vec::with_capacity(text.len() / 420);
    for line in text.lines().skip(1) {
        let f: Vec<&str> = line.split(',').collect();
        if f.len() < 61 { continue; }
        let t_us: i64 = match f[4].parse() { Ok(v) => v, Err(_) => continue };
        let mut s = Snap { t: t_us as f64 / 1e6, ..Default::default() };
        let mut ok = true;
        for l in 0..10 {
            let ap: f64 = match f[21 + 4 * l].parse() { Ok(v) => v, Err(_) => { ok = false; break } };
            let av: f64 = match f[22 + 4 * l].parse() { Ok(v) => v, Err(_) => { ok = false; break } };
            let bp: f64 = match f[23 + 4 * l].parse() { Ok(v) => v, Err(_) => { ok = false; break } };
            let bv: f64 = match f[24 + 4 * l].parse() { Ok(v) => v, Err(_) => { ok = false; break } };
            s.ask_p[l] = ap; s.ask_v[l] = av.max(0.0);
            s.bid_p[l] = bp; s.bid_v[l] = bv.max(0.0);
            if l == 0 { s.mid = (ap + bp) / 2.0; }
        }
        if !ok || !(s.mid > 0.0) { continue; }
        out.push(s);
    }
    if out.len() < 300 { return None; }
    out.sort_by(|a, b| a.t.partial_cmp(&b.t).unwrap());
    Some(out)
}

fn resample(snaps: &[Snap], step: f64) -> Vec<Snap> {
    let mut out: Vec<Snap> = Vec::new();
    let t0 = snaps[0].t;
    let mut cur = -1.0f64;
    for s in snaps {
        let b = ((s.t - t0) / step).floor();
        if b > cur { out.push(*s); cur = b; }
        else if let Some(last) = out.last_mut() { *last = *s; }
    }
    out
}

/// 一次回补事件。
#[derive(Clone, Copy, Default)]
struct Ev {
    side: u8,
    drop: f64,     // 库存掉的幅度（相对安静水平）
    dur: f64,      // 从掉下去到回补到 K 以上用了几个快照
    regain: f64,   // 回补到安静水平的多少
    pos: f64,      // 事件在日内的相对位置
    nlev: f64,     // 该价位在当天出现过的档号跨度
    quiet: f64,    // 安静水平
}

/// 对一个 (价位, 侧) 的库存序列做「耗尽—回补」识别，K 为回补阈值比例。
fn detect(seq: &[f64], side: u8, t0: f64, t1: f64, k: f64) -> Vec<Ev> {
    let n = seq.len();
    if n < 20 { return Vec::new(); }
    // 安静水平 = 当天该价位非零量的中位数
    let mut nz: Vec<f64> = seq.iter().cloned().filter(|x| *x > 0.0).collect();
    if nz.len() < 10 { return Vec::new(); }
    nz.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let quiet = quantile(&nz, 0.5);
    if !(quiet > 0.0) { return Vec::new(); }
    let low = quiet * 0.25;
    let mut out = Vec::new();
    let mut i = 1usize;
    while i + 2 < n {
        let prev = seq[i - 1];
        if prev < low * 0.5 || seq[i] > quiet * 0.4 || seq[i] >= low {
            i += 1;
            continue;
        }
        // 起点：seq[i] 掉到 low 以下（之前高于 0.4 quiet）
        let start = i;
        let mut j = i;
        while j < n && seq[j] < quiet * k { j += 1; }
        if j >= n { break; }
        if j - start > 2000 { i = j + 1; continue; }
        let dur = (j - start) as f64;
        let regain = seq[j] / quiet;
        let drop = 1.0 - seq[start] / quiet.max(1e-9);
        let pos = ((start as f64 + dur / 2.0) / n as f64).min(1.0);
        let _ = (t0, t1);
        out.push(Ev {
            side,
            drop: drop.max(0.0).min(1.0),
            dur,
            regain: regain.min(5.0),
            pos,
            nlev: 0.0,
            quiet,
        });
        i = j + 3;
    }
    out
}

/// 幂律/指数衰减拟合：对回补时间分布，用最小二乘拟合 log(生存函数) 的斜率。
fn decay_slope(dur: &[f64]) -> f64 {
    if dur.len() < 30 { return f64::NAN; }
    let mut s = dur.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mx = s[s.len() - 1].min(1e9);
    let nb = 30usize;
    let mut xs: Vec<f64> = Vec::new();
    let mut ys: Vec<f64> = Vec::new();
    for b in 0..nb {
        let x = mx * (b as f64 + 0.5) / nb as f64;
        let surv = s.iter().filter(|v| **v > x).count() as f64 / s.len() as f64;
        if surv > 0.0 && x > 0.0 {
            xs.push(x.ln());
            ys.push(surv.ln());
        }
    }
    if xs.len() < 5 { return f64::NAN; }
    let mx_ = mean(&xs);
    let my = mean(&ys);
    let mut sxy = 0.0;
    let mut sxx = 0.0;
    for i in 0..xs.len() {
        sxy += (xs[i] - mx_) * (ys[i] - my);
        sxx += (xs[i] - mx_) * (xs[i] - mx_);
    }
    if sxx.abs() < 1e-12 { return f64::NAN; }
    sxy / sxx
}

/// 霍克斯自激的粗估：事件间隔序列的分支比与衰减常数（用间隔的自相关与均值做矩估计）。
fn hawkes_moments(t: &[f64]) -> (f64, f64) {
    if t.len() < 50 { return (f64::NAN, f64::NAN); }
    let mut gaps: Vec<f64> = (1..t.len()).map(|i| (t[i] - t[i - 1]).max(0.0)).collect();
    gaps.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = quantile(&gaps, 0.5);
    if !(med > 0.0) { return (f64::NAN, f64::NAN); }
    // 分支比 n = 1 - sqrt(2 * Var / mean^2)（指数分布时 CV=1，n=0；聚簇越强 n 越大）
    let m = mean(&gaps);
    let v = std(&gaps).powi(2);
    let branch = if m > 0.0 { (1.0 - (2.0 * v / (m * m)).sqrt()).max(0.0) } else { f64::NAN };
    // 衰减常数 = 间隔中位数（秒）
    (branch, med)
}

/// 输出列名，顺序与 per-stock 里 put! 的顺序严格一致。
fn out_names() -> Vec<String> {
    let mut v: Vec<String> = Vec::new();
    let ks = [0.3f64, 0.4, 0.6, 0.8, 1.0];
    for &k in ks.iter() {
        let tag = format!("k{}", (k * 10.0) as i32);
        for s in ["n", "rate", "dur_p50", "dur_p90", "regain_mean", "regain_std", "drop_mean", "pos_mean", "slope"] {
            v.push(format!("{}_{}", tag, s));
        }
    }
    for s in ["regain_vs_pos_slope", "regain_vs_pos_r2", "regain_vs_drop_slope", "regain_vs_drop_r2",
              "dur_vs_drop_slope", "dur_vs_drop_r2", "hawkes_branch", "hawkes_decay_min",
              "gap_ac1", "gap_entropy", "gap_lz", "price_event_mean", "price_event_std",
              "price_event_max", "price_event_repeat_share"] {
        v.push(s.to_string());
    }
    for label in ["dur", "regain", "drop"] {
        for nm in ["mean", "std", "skew", "kurt", "p05", "p95", "cv", "ac1", "trend", "lz", "entropy"] {
            v.push(format!("{}_{}", label, nm));
        }
    }
    v.push("tick_size".to_string());
    for nm in ["mean", "std", "skew", "kurt", "p95", "cv", "ac1", "trend", "lz", "entropy", "max_range"] {
        v.push(format!("bid_step_{}", nm));
        v.push(format!("ask_step_{}", nm));
    }
    v.push("step_island_share".to_string());
    v.push("step_count".to_string());
    v.push("n_price_used".to_string());
    v
}

fn main() {
    let date: i64 = std::env::args().nth(1).expect("用法: oidflow_sandbox <date>")
        .parse().expect("date 必须是整数");
    let codes = list_codes(date).expect("股票范围或原始目录检查失败");
    eprintln!("codes = {}", codes.len());

    let rows: Vec<(String, Vec<f64>, [f64; 4])> = codes
        .par_iter()
        .filter_map(|c| {
            let snaps = read_snaps(c, date)?;
            let g = resample(&snaps, 3.0);
            let n = g.len();
            if n < 200 { return None; }
            let dt = if n > 2 { (g[n - 1].t - g[0].t) / (n - 1) as f64 } else { 3.0 };
            let t0 = g[0].t;
            let t1 = g[n - 1].t;

            // 按价位收集库存轨迹（侧, 价位）-> 量序列
            let mut seqs: HashMap<(u8, i64), Vec<f64>> = HashMap::new();
            let mut price_hits: HashMap<i64, usize> = HashMap::new();
            for s in g.iter() {
                for l in 0..10 {
                    for (side, p, vol) in [(0u8, s.bid_p[l], s.bid_v[l]), (1u8, s.ask_p[l], s.ask_v[l])] {
                        if !(p > 0.0) { continue; }
                        let key = (side, (p * 100.0).round() as i64);
                        let e = seqs.entry(key).or_insert_with(|| vec![0.0f64; n]);
                        let idx = ((s.t - t0) / dt.max(1e-9)).round() as usize;
                        let k = idx.min(n - 1);
                        e[k] += vol;
                    }
                    *price_hits.entry((s.bid_p[l] * 100.0).round() as i64).or_insert(0) += 1;
                }
            }
            let mut names: Vec<String> = Vec::new();
            let mut v: Vec<f64> = Vec::new();
            macro_rules! put { ($nm:expr, $x:expr) => { names.push($nm.to_string()); v.push($x); } }

            // 阈值谱：对每个 K 单独识别事件
            let ks = [0.3f64, 0.4, 0.6, 0.8, 1.0];
            let mut all_ev: Vec<Vec<Ev>> = Vec::new();
            for &k in ks.iter() {
                let mut evs: Vec<Ev> = Vec::new();
                for ((side, _), seq) in seqs.iter() {
                    let mut e = detect(seq, *side, t0, t1, k);
                    evs.append(&mut e);
                }
                all_ev.push(evs);
            }
            for (ki, &k) in ks.iter().enumerate() {
                let evs = &all_ev[ki];
                let tag = format!("k{}", (k * 10.0) as i32);
                if evs.len() < 20 {
                    for suffix in ["n", "rate", "dur_p50", "dur_p90", "regain_mean", "regain_std", "drop_mean", "pos_mean", "slope"] {
                        put!(format!("{}_{}", tag, suffix), f64::NAN);
                    }
                    continue;
                }
                let dur: Vec<f64> = evs.iter().map(|e| e.dur * dt).collect();
                let regain: Vec<f64> = evs.iter().map(|e| e.regain).collect();
                let drop: Vec<f64> = evs.iter().map(|e| e.drop).collect();
                let pos: Vec<f64> = evs.iter().map(|e| e.pos).collect();
                let mut ds = dur.clone();
                ds.sort_by(|a, b| a.partial_cmp(b).unwrap());
                put!(format!("{}_{}", tag, "n"), evs.len() as f64);
                put!(format!("{}_{}", tag, "rate"), evs.len() as f64 / seqs.len().max(1) as f64);
                put!(format!("{}_{}", tag, "dur_p50"), quantile(&ds, 0.5));
                put!(format!("{}_{}", tag, "dur_p90"), quantile(&ds, 0.9));
                put!(format!("{}_{}", tag, "regain_mean"), mean(&regain));
                put!(format!("{}_{}", tag, "regain_std"), std(&regain));
                put!(format!("{}_{}", tag, "drop_mean"), mean(&drop));
                put!(format!("{}_{}", tag, "pos_mean"), mean(&pos));
                put!(format!("{}_{}", tag, "slope"), decay_slope(&dur));
            }
            // 最宽阈值（K=1.0）事件集上的深度加工
            let ev = all_ev.last().unwrap();
            if ev.len() >= 50 {
                let dur: Vec<f64> = ev.iter().map(|e| e.dur * dt).collect();
                let regain: Vec<f64> = ev.iter().map(|e| e.regain).collect();
                let pos: Vec<f64> = ev.iter().map(|e| e.pos).collect();
                let drop: Vec<f64> = ev.iter().map(|e| e.drop).collect();
                // 回补比例 ~ 时间位置 的 1D 回归（带截距）
                let (sl, r2) = trend_of(&pos, &regain);
                put!("regain_vs_pos_slope", sl);
                put!("regain_vs_pos_r2", r2);
                let (sl2, r22) = trend_of(&drop, &regain);
                put!("regain_vs_drop_slope", sl2);
                put!("regain_vs_drop_r2", r22);
                let (sl3, r23) = trend_of(&drop, &dur);
                put!("dur_vs_drop_slope", sl3);
                put!("dur_vs_drop_r2", r23);
                // 事件时刻序列（按时间排序）与自激
                let mut tt: Vec<f64> = ev.iter().map(|e| e.pos * (t1 - t0) / 60.0).collect();
                tt.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let (branch, decay) = hawkes_moments(&tt);
                put!("hawkes_branch", branch);
                put!("hawkes_decay_min", decay);
                // 事件间隔的 LZ 与熵
                let gaps: Vec<f64> = (1..tt.len()).map(|i| tt[i] - tt[i - 1]).collect();
                let (act, en, lz) = gap_complexity(&gaps);
                put!("gap_ac1", act);
                put!("gap_entropy", en);
                put!("gap_lz", lz);
                // 事件集合的 21 统计（取 duration / regain / drop 三串）
                for (label, series) in [("dur", &dur), ("regain", &regain), ("drop", &drop)] {
                    let s = ts21(series);
                    for (k2, nm) in ["mean", "std", "skew", "kurt", "p05", "p95", "cv", "ac1", "trend", "lz", "entropy"].iter().enumerate() {
                        let idx = [0usize, 2, 3, 4, 5, 8, 10, 11, 13, 18, 19][k2];
                        put!(format!("{}_{}", label, nm), s[idx]);
                    }
                }
                // 事件是否「一地多次」：同一价位的重复次数分布
                let mut per_price: HashMap<(u8, i64), usize> = HashMap::new();
                for ((side, p), seq) in seqs.iter() {
                    let c = detect(seq, *side, t0, t1, 1.0).len();
                    per_price.insert((*side, *p), c);
                }
                let counts: Vec<f64> = per_price.values().map(|x| *x as f64).collect();
                put!("price_event_mean", mean(&counts));
                put!("price_event_std", std(&counts));
                put!("price_event_max", counts.iter().cloned().fold(0.0, f64::max));
                put!("price_event_repeat_share", counts.iter().filter(|x| **x >= 2.0).count() as f64 / counts.len().max(1) as f64);
            } else {
                for suffix in ["regain_vs_pos_slope","regain_vs_pos_r2","regain_vs_drop_slope","regain_vs_drop_r2",
                               "dur_vs_drop_slope","dur_vs_drop_r2","hawkes_branch","hawkes_decay_min",
                               "gap_ac1","gap_entropy","gap_lz","price_event_mean","price_event_std",
                               "price_event_max","price_event_repeat_share"] {
                    put!(suffix, f64::NAN);
                }
                for label in ["dur","regain","drop"] {
                    for nm in ["mean","std","skew","kurt","p05","p95","cv","ac1","trend","lz","entropy"] {
                        put!(format!("{}_{}", label, nm), f64::NAN);
                    }
                }
            }

            // 价格阶梯几何：十档价位之间的空档（跨越的最小价格单位数）
            let tick = estimate_tick(&g);
            let mut gaps_bid: Vec<f64> = Vec::new();
            let mut gaps_ask: Vec<f64> = Vec::new();
            let mut islands = 0.0f64;
            let mut steps = 0.0f64;
            for s in g.iter() {
                for l in 0..9 {
                    let bp = s.bid_p[l] - s.bid_p[l + 1];
                    let ap = s.ask_p[l + 1] - s.ask_p[l];
                    if bp > 0.0 && tick > 0.0 { gaps_bid.push(bp / tick); }
                    if ap > 0.0 && tick > 0.0 { gaps_ask.push(ap / tick); }
                }
                let mut nz = 0;
                for l in 0..10 { if s.bid_v[l] > 0.0 { nz += 1; } }
                if nz < 3 { islands += 1.0; }
                for l in 1..10 {
                    if s.bid_v[l] > 0.0 && s.bid_v[l - 1] <= 0.0 && s.bid_v[l - 2.min(l - 1)] > 0.0 { steps += 1.0; }
                }
            }
            let gs = ts21(&gaps_bid.iter().cloned().filter(|x| x.is_finite() && *x > 0.0).collect::<Vec<f64>>());
            let gsa = ts21(&gaps_ask.iter().cloned().filter(|x| x.is_finite() && *x > 0.0).collect::<Vec<f64>>());
            put!("tick_size", tick);
            for (k2, nm) in ["mean","std","skew","kurt","p95","cv","ac1","trend","lz","entropy","max_range"].iter().enumerate() {
                let idx = [0usize,2,3,4,8,10,11,13,18,19,20][k2];
                put!(format!("bid_step_{}", nm), gs[idx]);
                put!(format!("ask_step_{}", nm), gsa[idx]);
            }
            put!("step_island_share", islands / n as f64);
            put!("step_count", steps / n as f64);
            put!("n_price_used", seqs.len() as f64);

            let diag = [n as f64, snaps.len() as f64, seqs.len() as f64, dt];
            Some((c.clone(), v, diag))
        })
        .collect();

    let mut out = Out { names: Vec::new(), codes: Vec::new(), vals: Vec::new(), diag: Vec::new() };
    for (c, v, d) in rows {
        out.codes.push(c);
        out.vals.push(v.into_iter().map(|x| x as f32).collect());
        out.diag.push(d.iter().map(|&x| x as f32).collect());
    }
    let ncol = out.vals.first().map(|v| v.len()).unwrap_or(0);
    out.names = out_names();
    assert_eq!(out.names.len(), ncol, "v10 列名与输出数量不一致");
    eprintln!("valid = {} cols = {}", out.codes.len(), ncol);
    println!("{}", serde_json::to_string(&out).unwrap());
}

/// 单变量回归（带截距）：返回 (斜率, R²)。
fn trend_of(x: &[f64], y: &[f64]) -> (f64, f64) {
    let n = x.len().min(y.len());
    if n < 5 { return (f64::NAN, f64::NAN); }
    let (mx, my) = (mean(&x[..n]), mean(&y[..n]));
    let mut sxy = 0.0;
    let mut sxx = 0.0;
    for i in 0..n { sxy += (x[i] - mx) * (y[i] - my); sxx += (x[i] - mx).powi(2); }
    if sxx <= 1e-15 { return (f64::NAN, f64::NAN); }
    let sl = sxy / sxx;
    let mut sse = 0.0;
    let mut sst = 0.0;
    for i in 0..n {
        let fit = my + sl * (x[i] - mx);
        sse += (y[i] - fit).powi(2);
        sst += (y[i] - my).powi(2);
    }
    (sl, if sst > 1e-15 { 1.0 - sse / sst } else { f64::NAN })
}

fn gap_complexity(gaps: &[f64]) -> (f64, f64, f64) {
    if gaps.len() < 20 { return (f64::NAN, f64::NAN, f64::NAN); }
    let ac = autocorr1(gaps);
    let en = entropy_1d(gaps, 20);
    let m = mean(gaps);
    let lz = lz_complexity(&gaps.iter().map(|x| if *x >= m { 1u8 } else { 0u8 }).collect::<Vec<u8>>());
    (ac, en, lz)
}

/// 用一个 tick 的估计值：取相邻档位价格差的最小正值（常见间距的众数）。
fn estimate_tick(g: &[Snap]) -> f64 {
    let mut diffs: Vec<f64> = Vec::new();
    for s in g.iter() {
        for l in 0..9 {
            let d1 = s.bid_p[l] - s.bid_p[l + 1];
            let d2 = s.ask_p[l + 1] - s.ask_p[l];
            if d1 > 1e-9 { diffs.push(d1); }
            if d2 > 1e-9 { diffs.push(d2); }
        }
    }
    if diffs.is_empty() { return 0.0; }
    diffs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    quantile(&diffs, 0.05).max(1e-9)
}
