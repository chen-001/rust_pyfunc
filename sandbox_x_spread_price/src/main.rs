//! 对撞机 sandbox：前五档量不平衡 x 价格水平 x 基点点差 —— 谁在决定预测力
//! 目标：把「价格水平」和「点差」这两个高度相关的量拆开，看不平衡的预测力到底跟着谁单调。
//! 构建: cd sandbox_x_spread_price && ~/.cargo/bin/cargo build --release
//! 运行: XS_THREADS=12 ./target/release/x_spread_price 20220819 20241231
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
    time_us: i64,
    lt: i64,
    past_vol: f64,   // 过去 60 秒中间价收益标准差（快照当下就能知道，bp）   // 本地时刻（秒，含 −8h 偏移还原 + 下午 90 分钟平移还原）
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
    names: f64,     // 桶内股票日数（无条件）
    names_ic: f64,  // 其中算得出 IC 的股票日数
    obs: f64,
    sr_mid: f64, sr_mid2: f64,   // spearman(imb, r_mid)
    sr_last: f64, sr_last2: f64,
    tb_mid: f64, tb_mid2: f64,   // top30 - bottom30，mid 收益，bp
    tb_last: f64, tb_last2: f64,
    sp_bps: f64, sp_ticks: f64, price: f64, depth: f64, fwd_vol: f64, absr: f64,
    // v3 新增
    sp_abs: f64, tick_rel: f64, absimb: f64, r2_mid: f64,
    nic: f64, nic2: f64,   // 用该股票日 60 秒波动归一后的收益算的 IC
    tb_norm: f64, tb_norm2: f64,  // 收益除以该股票日 60 秒波动后的多空差（无量纲）
    pdiff: f64, pdiff2: f64,      // 配对差：同一股票日内 宽点差半区 IC − 窄点差半区 IC
    pdiff_n: f64,
    sig: f64,                     // 档内收益尺度（该档 r_mid 的标准差，bp）
    zero_ret: f64,                // 未来 60 秒收益恰好为 0 的观测占比（每个股票日贡献一个比例）
    tb_on_sig: f64, tb_on_sig2: f64,  // tb_mid_bp / 档内尺度
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
    // 过去 60 秒窗口的左端指针
    let mut p = 0usize;
    for (k, &i) in valid.iter().enumerate() {
        if j < k { j = k; }
        let t = recs[i].time_us;
        let sod = (t / 1_000_000).rem_euclid(86400);
        if sod >= 11 * 3600 + 29 * 60 && sod < 11 * 3600 + 30 * 60 { continue; }
        if p > k { p = k; }
        while p < k && recs[valid[p]].time_us < t - FWD_US { p += 1; }
        // 过去 60 秒中间价收益的标准差（bp）：只用中间价自己的变化，不受点差影响
        let past_vol = if k - p >= 5 {
            let mut s = 0.0; let mut s2 = 0.0; let mut cnt = 0.0;
            let mut prev_mid = f64::NAN;
            for w in p..=k {
                let rr = &recs[valid[w]];
                let m = 0.5 * (rr.ask_prcs[0] + rr.bid_prcs[0]);
                if m <= 0.0 { continue; }
                if prev_mid.is_finite() {
                    let x = (m / prev_mid - 1.0) * 1e4;
                    s += x; s2 += x * x; cnt += 1.0;
                }
                prev_mid = m;
            }
            if cnt >= 3.0 { let mu = s / cnt; (((s2 / cnt - mu * mu).max(0.0)).sqrt()) } else { f64::NAN }
        } else { f64::NAN };
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
            time_us: t,
            lt: (t / 1_000_000).rem_euclid(86400),
            past_vol,
            imb: (bid5 - ask5) / (bid5 + ask5),
            r_mid: (fmid / mid - 1.0) * 1e4,
            r_last: (f.last_prc / r.last_prc - 1.0) * 1e4,
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
    pearson(&rank(x), &rank(y))
}

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

fn add_obs(s: &mut Sums, o: &Obs) {
    s.obs += 1.0;
    s.sp_bps += o.spread_bps;
    s.sp_ticks += o.spread_ticks as f64;
    s.price += o.price;
    s.depth += o.depth;
    s.fwd_vol += o.fwd_vol;
    s.absr += o.r_mid.abs();
    s.sp_abs += o.spread_ticks as f64 * TICK;
    s.tick_rel += TICK / o.price;
    s.absimb += o.imb.abs();
    s.r2_mid += o.r_mid * o.r_mid;
}

/// 一只股票日内：按 labels 分组累加
fn add_stock_day(res: &mut [Sums], obs: &[Obs], labels: &[usize], sd_vol: f64) {
    add_stock_day_min(res, obs, labels, sd_vol, MIN_BUCKET);
}

fn add_stock_day_min(res: &mut [Sums], obs: &[Obs], labels: &[usize], sd_vol: f64, min_obs: usize) {
    // 守卫：标签越界会在下面被静默丢掉，这里当场报错
    let maxlab = labels.iter().copied().max().unwrap_or(0);
    assert!(
        maxlab < res.len(),
        "add_stock_day: 标签越界（max={} 但分组数组长度={}），越界观测会被静默丢弃",
        maxlab, res.len()
    );
    for g in 0..res.len() {
        let idxs: Vec<usize> = (0..obs.len()).filter(|&k| labels[k] == g).collect();
        if idxs.len() < min_obs { continue; }
        let x: Vec<f64> = idxs.iter().map(|&k| obs[k].imb).collect();
        let ym: Vec<f64> = idxs.iter().map(|&k| obs[k].r_mid).collect();
        let yl: Vec<f64> = idxs.iter().map(|&k| obs[k].r_last).collect();
        let s = &mut res[g];
        s.names += 1.0;
        let mut got_ic = false;
        if let Some(v) = spearman(&x, &ym) { s.sr_mid += v; s.sr_mid2 += v * v; got_ic = true; }
        if let Some(v) = spearman(&x, &yl) { s.sr_last += v; s.sr_last2 += v * v; }
        if let Some(v) = top_bottom(&x, &ym) { s.tb_mid += v; s.tb_mid2 += v * v; }
        if let Some(v) = top_bottom(&x, &yl) { s.tb_last += v; s.tb_last2 += v * v; }
        if got_ic { s.names_ic += 1.0; }
        for &k in &idxs { add_obs(s, &obs[k]); }
        s.zero_ret += idxs.iter().filter(|&&k| obs[k].r_mid == 0.0).count() as f64 / idxs.len() as f64;
        // 档内收益尺度 + 按档内尺度归一的 tb
        {
            let sd = (idxs.iter().map(|&k| obs[k].r_mid * obs[k].r_mid).sum::<f64>() / idxs.len() as f64).sqrt();
            if sd > 0.0 {
                let yn2: Vec<f64> = idxs.iter().map(|&k| obs[k].r_mid / sd).collect();
                if let Some(v) = top_bottom(&x, &yn2) { s.tb_on_sig += v; s.tb_on_sig2 += v * v; s.sig += sd; }
            }
        }
        // 该股票日全样本 60 秒波动（bp）
        let vol = if sd_vol > 0.0 { sd_vol } else { 1.0 };
        let yn: Vec<f64> = idxs.iter().map(|&k| obs[k].r_mid / vol).collect();
        if let Some(v) = spearman(&x, &yn) { s.nic += v; s.nic2 += v * v; }
        if let Some(v) = top_bottom(&x, &yn) { s.tb_norm += v; s.tb_norm2 += v * v; }
    }
}

fn mean_se(sum: f64, sum2: f64, n: f64) -> (f64, f64) {
    if n < 1.0 { return (f64::NAN, f64::NAN); }
    let m = sum / n;
    let var = (sum2 / n - m * m).max(0.0);
    (m, (var / n).sqrt())
}

fn json_group(name: &str, s: &Sums) -> String {
    let (rm, rse) = mean_se(s.sr_mid, s.sr_mid2, s.names_ic);
    let (rl, _rse2) = mean_se(s.sr_last, s.sr_last2, s.names);
    let (tm, tse) = mean_se(s.tb_mid, s.tb_mid2, s.names);
    let (tl, _tse2) = mean_se(s.tb_last, s.tb_last2, s.names);
    let (nic, _) = mean_se(s.nic, s.nic2, s.names_ic);
    let d = s.obs.max(1.0);
    let vol_bp = (s.r2_mid / d).sqrt();
    let ic_over_vol = if vol_bp > 0.0 { rm / vol_bp } else { f64::NAN };
    format!(
        "{{\"name\":\"{}\",\"stock_days\":{:.0},\"stock_days_ic\":{:.0},\"obs\":{:.0},\"ic_mid\":{:.4},\"ic_mid_se\":{:.4},\"ic_last\":{:.4},\"tb_mid_bp\":{:.3},\"tb_mid_se\":{:.3},\"tb_last_bp\":{:.3},\"spread_ticks\":{:.2},\"spread_bps\":{:.2},\"price\":{:.2},\"depth5\":{:.0},\"fwd_vol\":{:.0},\"absret_bp\":{:.2},\"sp_abs\":{:.3},\"tick_rel_bp\":{:.2},\"absimb\":{:.3},\"vol_bp\":{:.2},\"nic\":{:.4},\"ic_over_vol\":{:.4},\"tb_norm\":{:.4},\"tb_on_sig\":{:.4},\"zero_ret_share\":{:.4}}}",
        name, s.names, s.names_ic, s.obs, rm, rse, rl, tm, tse, tl,
        s.sp_ticks / d, s.sp_bps / d, s.price / d, s.depth / d, s.fwd_vol / d, s.absr / d,
        s.sp_abs / d, s.tick_rel / d * 1e4, s.absimb / d, vol_bp, nic, ic_over_vol,
        mean_se(s.tb_norm, s.tb_norm2, s.names).0,
        mean_se(s.tb_on_sig, s.tb_on_sig2, s.names).0,
        s.zero_ret / s.names.max(1.0)
    )
}
/// 在给定分组内去掉均值（时段去均值用）
fn demean_by(groups: &[usize], x: &[f64], ng: usize) -> Vec<f64> {
    let mut sum = vec![0.0f64; ng];
    let mut cnt = vec![0.0f64; ng];
    for i in 0..x.len() { sum[groups[i]] += x[i]; cnt[groups[i]] += 1.0; }
    let mut out = vec![0.0f64; x.len()];
    for i in 0..x.len() { out[i] = x[i] - sum[groups[i]] / cnt[groups[i]].max(1.0); }
    out
}

/// 组内残差化：x 对 [1, sp_bps, price(对数)] 做最小二乘，返回残差
fn residualize(x: &[f64], sp: &[f64], lnpx: &[f64]) -> Vec<f64> {
    let n = x.len();
    // 3x3 正规方程 (1, sp, lnpx)
    let (mut a11, mut a12, mut a13, mut a22, mut a23, mut a33) = (0.0f64, 0.0, 0.0, 0.0, 0.0, 0.0);
    let (mut b1, mut b2, mut b3) = (0.0f64, 0.0, 0.0);
    for i in 0..n {
        let (s, p, y) = (sp[i], lnpx[i], x[i]);
        a11 += 1.0; a12 += s; a13 += p;
        a22 += s * s; a23 += s * p; a33 += p * p;
        b1 += y; b2 += s * y; b3 += p * y;
    }
    // 高斯消元解 3x3
    let mut m = [[a11, a12, a13, b1], [a12, a22, a23, b2], [a13, a23, a33, b3]];
    for c in 0..3 {
        let mut piv = c;
        for r in c + 1..3 { if m[r][c].abs() > m[piv][c].abs() { piv = r; } }
        m.swap(c, piv);
        if m[c][c].abs() < 1e-12 { return x.to_vec(); }
        for r in 0..3 {
            if r == c { continue; }
            let f = m[r][c] / m[c][c];
            for k in c..4 { m[r][k] -= f * m[c][k]; }
        }
    }
    let beta = [m[0][3] / m[0][0], m[1][3] / m[1][1], m[2][3] / m[2][2]];
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        out.push(x[i] - (beta[0] + beta[1] * sp[i] + beta[2] * lnpx[i]));
    }
    out
}

/// 同一股票日内，点差高于自身中位 / 低于中位的两半分别算 IC，取配对差
fn paired_spread_diff(obs: &[Obs]) -> Option<f64> {
    let mut sp: Vec<f64> = obs.iter().map(|o| o.spread_bps).collect();
    sp.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = sp[sp.len() / 2];
    let mut lo_x = Vec::new(); let mut lo_y = Vec::new();
    let mut hi_x = Vec::new(); let mut hi_y = Vec::new();
    for o in obs {
        if o.spread_bps <= med { lo_x.push(o.imb); lo_y.push(o.r_mid); }
        else { hi_x.push(o.imb); hi_y.push(o.r_mid); }
    }
    if lo_x.len() < MIN_BUCKET || hi_x.len() < MIN_BUCKET { return None; }
    let a = spearman(&lo_x, &lo_y)?;
    let b = spearman(&hi_x, &hi_y)?;
    Some(b - a)
}

/// 名义观测数：同一秒内多个快照互相重复，按「唯一整秒」估算有效样本
fn eff_obs(obs: &[Obs]) -> f64 {
    let mut secs: Vec<i64> = obs.iter().map(|o| (o.time_us / 1_000_000) as i64).collect();
    secs.sort_unstable();
    secs.dedup();
    secs.len() as f64
}

fn corr2(n: f64, sx: f64, sy: f64, sxx: f64, syy: f64, sxy: f64) -> f64 {
    let num = n * sxy - sx * sy;
    let den = ((n * sxx - sx * sx) * (n * syy - sy * sy)).sqrt();
    if den > 0.0 { num / den } else { f64::NAN }
}

fn jnum(v: f64) -> String {
    if v.is_finite() { format!("{:.4}", v) } else { "null".to_string() }
}

fn run(date: i64) {
    let wl = read_symbol_map(SYMBOL_MAP);
    let codes = list_codes(date, &wl);
    let nt: usize = std::env::var("XS_THREADS").ok().and_then(|s| s.parse().ok()).unwrap_or(12);
    eprintln!("[{}] codes in whitelist: {}", date, codes.len());
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

    // 价格桶分界：按每只股票当天的中位价
    const PB: [f64; 4] = [5.0, 20.0, 100.0, f64::INFINITY];
    const PNAME: [&str; 4] = ["px_lt5", "px_5to20", "px_20to100", "px_ge100"];
    let pgroup = |med: f64| -> usize {
        if med < PB[0] { 0 } else if med < PB[1] { 1 } else if med < PB[2] { 2 } else { 3 }
    };

    // 全样本
    let mut res_all = vec![Sums::default(); 1];
    // 价格桶
    let mut res_px = vec![Sums::default(); 4];
    // 价格桶 x 点差五分位（在价格桶内按 sp_bps 五分位）
    let mut res_cell = vec![Sums::default(); 4 * 5];
    // 全样本点差五分位
    let mut res_spq = vec![Sums::default(); 5];
    // 股内自身中位点差：低于自身中位 / 高于
    let mut res_rel = vec![Sums::default(); 2];
    // 残差化 IC（x = imb 对 sp_bps + ln(price) 回归后残差）
    let mut res_fwl = vec![Sums::default(); 1];
    let mut res_fwl_px = vec![Sums::default(); 4];

    // 观测级相关矩阵（全样本，用 sum 累加）
    let (mut n_obs, mut s_px, mut s_sp, mut s_ib) = (0.0f64, 0.0, 0.0, 0.0);
    let (mut s_pp, mut s_ss, mut s_ii, mut s_ps, mut s_pi, mut s_si) = (0.0f64, 0.0, 0.0, 0.0, 0.0, 0.0);
    let (mut s_amt, mut s_aa, mut s_pp2, mut s_pa) = (0.0f64, 0.0, 0.0, 0.0);

    let mut res_tod = vec![Sums::default(); 6];          // 3 时段 x 2 (低价/高价)
    let mut res_pv = vec![Sums::default(); 3];           // 过去 60 秒波动三档（观测级）
    let mut res_pvz = vec![Sums::default(); 2];          // 过去 60 秒零波动 / 非零
    let mut res_imbq = vec![Sums::default(); 3];         // |imb| 三档（弱 / 中 / 强），档内算 IC
    let mut res_binsp = vec![Sums::default(); 8 * 5];    // 时段窗（8 个）× 窗内点差五分位
    let mut res_absq = vec![Sums::default(); 5];         // 按 sp_abs（元）五分位，而不是 bps
    let mut res_absq_cell = vec![Sums::default(); 20];   // sp_abs 五分位 x 价格桶
    let mut res_resid3 = vec![Sums::default(); 3];       // 去掉时段后点差三档
    let mut res_residpx = vec![Sums::default(); 12];     // 去掉时段后点差三档 x 价格桶 4
    let mut res_pvzpx = vec![Sums::default(); 8];        // 零/非零 x 价格桶 4
    let mut zero_cnt = [0f64; 4];                        // 各价格桶里零波动观测占比
    let mut res_pvpx = vec![Sums::default(); 12];        // 过去波动 3 x 价格桶 4
    let mut res_pvsp = vec![Sums::default(); 15];        // 过去波动 3 x 价格桶内点差五分位 5（高价股用）
    let mut res_ext = vec![Sums::default(); 4];          // 相对点差 2 x 极端不平衡 2
    let mut res_todpv = vec![Sums::default(); 18];       // 时段窗（6 个有数据）× 过去波动三档（股票日内切）
    let mut res_pv_nonzero = vec![Sums::default(); 3];   // 过去波动三档，但整日零波动的股票日单独剔除
    let mut d_allzero = 0f64; let mut d_all = 0f64;
    let mut res_imbsd = vec![Sums::default(); 3];        // 不平衡按股票日内三分位
    let mut res_nz = vec![Sums::default(); 3];           // 只看未来收益非零的观测：过去波动三档
    let mut res_nzpx = vec![Sums::default(); 12];        // 只看未来收益非零：过去波动 3 x 价格桶 4
    let mut res_vol = vec![Sums::default(); 3];
    let mut res_volpx = vec![Sums::default(); 12];
    let mut total_obs = 0f64;
    let mut eff_total = 0f64;

    let mut vols: Vec<f64> = data.iter().map(|(_c, obs)| (obs.iter().map(|o| o.r_mid * o.r_mid).sum::<f64>() / obs.len() as f64).sqrt()).collect();
    vols.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let vol_cut1 = vols[vols.len() / 3];
    let vol_cut2 = vols[vols.len() * 2 / 3];

    for (_code, obs) in &data {
        let sd_vol = (obs.iter().map(|o| o.r_mid * o.r_mid).sum::<f64>() / obs.len() as f64).sqrt();
        let mut prs: Vec<f64> = obs.iter().map(|o| o.price).collect();
        prs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let med_prc = prs[prs.len() / 2];
        let pg = pgroup(med_prc);

        add_stock_day(&mut res_all, obs, &vec![0usize; obs.len()], sd_vol);
        add_stock_day(&mut res_px, obs, &vec![pg; obs.len()], sd_vol);

        // 全样本点差五分位（按本股票日内 sp_bps 排序分 5 组）
        {
            let mut idx: Vec<usize> = (0..obs.len()).collect();
            idx.sort_by(|&a, &b| obs[a].spread_bps.partial_cmp(&obs[b].spread_bps).unwrap());
            let mut lab = vec![0usize; obs.len()];
            for (r, &i) in idx.iter().enumerate() { lab[i] = (r * 5 / obs.len()).min(4); }
            add_stock_day(&mut res_spq, obs, &lab, sd_vol);
            // 价格桶内同样五分位 -> cell
            let mut lab_cell = vec![0usize; obs.len()];
            for (r, &i) in idx.iter().enumerate() { lab_cell[i] = pg * 5 + (r * 5 / obs.len()).min(4); }
            add_stock_day(&mut res_cell, obs, &lab_cell, sd_vol);
        }

        // 相对自身中位点差
        {
            let mut sp: Vec<f64> = obs.iter().map(|o| o.spread_bps).collect();
            sp.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let med_sp = sp[sp.len() / 2];
            let lab: Vec<usize> = obs.iter().map(|o| if o.spread_bps <= med_sp { 0 } else { 1 }).collect();
            add_stock_day(&mut res_rel, obs, &lab, sd_vol);
        }

        if let Some(d) = paired_spread_diff(obs) {
            res_all[0].pdiff += d; res_all[0].pdiff2 += d * d; res_all[0].pdiff_n += 1.0;
        }

        // 按 sp_abs（绝对价差）五分位，而不是 bps
        {
            let mut idx: Vec<usize> = (0..obs.len()).collect();
            idx.sort_by(|&a, &b| obs[a].spread_ticks.partial_cmp(&obs[b].spread_ticks).unwrap());
            let mut lab = vec![0usize; obs.len()];
            for (r, &i2) in idx.iter().enumerate() { lab[i2] = (r * 5 / obs.len()).min(4); }
            add_stock_day(&mut res_absq, obs, &lab, sd_vol);
            let labc: Vec<usize> = lab.iter().map(|&q| q + 5 * pg).collect();
            add_stock_day(&mut res_absq_cell, obs, &labc, sd_vol);
        }

        // |imb| 三档：档内各自的 IC（并列被消除，口径与 pure_imb 一致）
        {
            let mut a: Vec<f64> = obs.iter().map(|o| o.imb.abs()).collect();
            a.sort_by(|x, y| x.partial_cmp(y).unwrap());
            let c1 = a[a.len() / 3]; let c2 = a[a.len() * 2 / 3];
            let lab: Vec<usize> = obs.iter().map(|o| { let v = o.imb.abs(); if v <= c1 { 0 } else if v <= c2 { 1 } else { 2 } }).collect();
            add_stock_day(&mut res_imbq, obs, &lab, sd_vol);
        }

        // 时段窗（本地时刻每 45 分钟一个窗，共 8 窗覆盖 09:00-15:00）× 窗内点差五分位
        {
            let bins: Vec<usize> = obs.iter().map(|o| ((((o.lt - 9 * 3600).max(0)) / (45 * 60)) as usize).min(7)).collect();
            // 窗内五分位：按 (窗, 该窗内 sp_bps 排序) 切
            let mut order: Vec<usize> = (0..obs.len()).collect();
            order.sort_by(|&a, &b| (bins[a], obs[a].spread_bps.partial_cmp(&obs[b].spread_bps).unwrap()).cmp(&(bins[b], std::cmp::Ordering::Equal)));
            let mut lab = vec![0usize; obs.len()];
            let mut i2 = 0usize;
            while i2 < order.len() {
                let b = bins[order[i2]];
                let mut j2 = i2;
                while j2 < order.len() && bins[order[j2]] == b { j2 += 1; }
                let n2 = j2 - i2;
                for (r, &idx) in order[i2..j2].iter().enumerate() { lab[idx] = b * 5 + (r * 5 / n2.max(1)).min(4); }
                i2 = j2;
            }
            add_stock_day(&mut res_binsp, obs, &lab, sd_vol);

            // 去掉时段：点差在（股票日 × 时段窗）内去均值，再在股票日内按残差三分位切
            let sp: Vec<f64> = obs.iter().map(|o| o.spread_bps).collect();
            let rsp = demean_by(&bins, &sp, 8);
            let mut idx: Vec<usize> = (0..obs.len()).collect();
            idx.sort_by(|&a, &b| rsp[a].partial_cmp(&rsp[b]).unwrap());
            let mut lab3 = vec![0usize; obs.len()];
            for (r, &i3) in idx.iter().enumerate() { lab3[i3] = (r * 3 / obs.len()).min(2); }
            add_stock_day(&mut res_resid3, obs, &lab3, sd_vol);
            let lab3px: Vec<usize> = lab3.iter().map(|&v| v + 3 * pg).collect();
            add_stock_day(&mut res_residpx, obs, &lab3px, sd_vol);
        }

        // 过去 60 秒是否零波动（低价股八成时点没有撮合让报价动过）
        {
            let lab: Vec<usize> = obs.iter().map(|o| if o.past_vol == 0.0 { 0 } else { 1 }).collect();
            add_stock_day(&mut res_pvz, obs, &lab, sd_vol);
            let labpx: Vec<usize> = lab.iter().map(|&z| z * 4 + pg).collect();
            add_stock_day(&mut res_pvzpx, obs, &labpx, sd_vol);
            for (k, o) in obs.iter().enumerate() { let _ = k; if o.past_vol == 0.0 { zero_cnt[pg] += 1.0; } }
        }

        // 整日过去波动全为 0 的股票日单独标记（这些日子不该塞进最低档）
        {
            let all_zero = obs.iter().all(|o| !o.past_vol.is_finite() || o.past_vol == 0.0);
            d_all += 1.0;
            if all_zero { d_allzero += 1.0; }
            if !all_zero {
                let mut v: Vec<f64> = obs.iter().map(|o| if o.past_vol.is_finite() { o.past_vol } else { 0.0 }).collect();
                v.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let c1 = v[v.len() / 3]; let c2 = v[v.len() * 2 / 3];
                let lab: Vec<usize> = obs.iter().map(|o| { let p = if o.past_vol.is_finite() { o.past_vol } else { 0.0 };
                    if p <= c1 { 0 } else if p <= c2 { 1 } else { 2 } }).collect();
                add_stock_day(&mut res_pv_nonzero, obs, &lab, sd_vol);
            }
        }

        // 时段窗（45 分钟）x 过去波动三档（都在股票日内切）
        {
            let mut v: Vec<f64> = obs.iter().map(|o| if o.past_vol.is_finite() { o.past_vol } else { 0.0 }).collect();
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let c1 = v[v.len() / 3]; let c2 = v[v.len() * 2 / 3];
            let lab: Vec<usize> = obs.iter().map(|o| {
                let p = if o.past_vol.is_finite() { o.past_vol } else { 0.0 };
                let vg = if p <= c1 { 0 } else if p <= c2 { 1 } else { 2 };
                let tb = ((((o.lt - 9 * 3600).max(0)) / (45 * 60)) as usize).min(7);
                if tb < 6 { vg * 6 + tb } else { 18 }
            }).collect();
            let usable: Vec<usize> = (0..obs.len()).filter(|&k| lab[k] < 18).collect();
            if usable.len() >= 12 * 6 {
                let obs2: Vec<Obs> = usable.iter().map(|&k| obs[k]).collect();
                let lab2: Vec<usize> = usable.iter().map(|&k| lab[k]).collect();
                add_stock_day_min(&mut res_todpv, &obs2, &lab2, sd_vol, 12);
            }
        }

        // 不平衡按股票日内三分位（每只股票同时出现在三档里）
        {
            let mut a: Vec<f64> = obs.iter().map(|o| o.imb.abs()).collect();
            a.sort_by(|x, y| x.partial_cmp(y).unwrap());
            let c1 = a[a.len() / 3]; let c2 = a[a.len() * 2 / 3];
            let lab: Vec<usize> = obs.iter().map(|o| { let v = o.imb.abs(); if v <= c1 { 0 } else if v <= c2 { 1 } else { 2 } }).collect();
            add_stock_day(&mut res_imbsd, obs, &lab, sd_vol);
        }

        // 过去 60 秒波动三档（每只股票日各切，三档同时出现在同一只股票里）；再对高价桶做波动 x 点差五分位
        {
            let mut v: Vec<f64> = obs.iter().map(|o| if o.past_vol.is_finite() { o.past_vol } else { 0.0 }).collect();
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let c1 = v[v.len() / 3];
            let c2 = v[v.len() * 2 / 3];
            let pv1 = c1; let pv2 = c2;
            let mut lab = vec![0usize; obs.len()];
            for (k, o) in obs.iter().enumerate() {
                let pvv = if o.past_vol.is_finite() { o.past_vol } else { 0.0 };
                let vg = if pvv <= pv1 { 0 } else if pvv <= pv2 { 1 } else { 2 };
                lab[k] = vg;
            }
            add_stock_day(&mut res_pv, obs, &lab, sd_vol);
            let labpx: Vec<usize> = lab.iter().map(|&v| v * 4 + pg).collect();
            add_stock_day(&mut res_pvpx, obs, &labpx, sd_vol);
            // 同一分组，但剔掉未来收益恰好为 0 的观测
            let nz: Vec<usize> = (0..obs.len()).filter(|&k| obs[k].r_mid != 0.0).collect();
            if nz.len() >= MIN_BUCKET {
                let labnz: Vec<usize> = nz.iter().map(|&k| lab[k]).collect();
                let obsnz: Vec<Obs> = nz.iter().map(|&k| obs[k]).collect();
                add_stock_day(&mut res_nz, &obsnz, &labnz, sd_vol);
                let labpxnz: Vec<usize> = nz.iter().map(|&k| lab[k] * 4 + pg).collect();
                add_stock_day(&mut res_nzpx, &obsnz, &labpxnz, sd_vol);
            }
            if pg == 3 {
                // 高价桶内再按点差五分位
                let mut idx: Vec<usize> = (0..obs.len()).collect();
                idx.sort_by(|&a, &b| obs[a].spread_bps.partial_cmp(&obs[b].spread_bps).unwrap());
                let mut lab2 = vec![0usize; obs.len()];
                for (r, &i2) in idx.iter().enumerate() {
                    let q = (r * 5 / obs.len()).min(4);
                    lab2[i2] = lab[i2] * 5 + q;
                }
                add_stock_day(&mut res_pvsp, obs, &lab2, sd_vol);
            }
        }

        {
            // 时段（3 段）x 价格（低价 vs 高价）
            let cheap = if pg <= 1 { 0usize } else { 1usize };
            let mut lab = vec![0usize; obs.len()];
            for (k, o) in obs.iter().enumerate() {
                let seg = if o.lt < 11 * 3600 + 30 * 60 { 0 } else if o.lt < 13 * 3600 + 30 * 60 { 1 } else { 2 };
                lab[k] = seg * 2 + cheap;
            }
            add_stock_day(&mut res_tod, obs, &lab, sd_vol);

            // 相对点差（高于自身中位 / 低于）x 极端不平衡（|imb| 是否在本股票日内前 20%）
            let mut ibs: Vec<f64> = obs.iter().map(|o| o.imb.abs()).collect();
            ibs.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let cut = ibs[ibs.len() * 8 / 10];
            let mut sps: Vec<f64> = obs.iter().map(|o| o.spread_bps).collect();
            sps.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let med_sp = sps[sps.len() / 2];
            let mut lab2 = vec![0usize; obs.len()];
            for (k, o) in obs.iter().enumerate() {
                let w = if o.spread_bps > med_sp { 0usize } else { 1usize };
                let e = if o.imb.abs() >= cut { 0usize } else { 1usize };
                lab2[k] = w * 2 + e;
            }
            add_stock_day(&mut res_ext, obs, &lab2, sd_vol);
        }

        {
            let vg = if sd_vol <= vol_cut1 { 0 } else if sd_vol <= vol_cut2 { 1 } else { 2 };
            add_stock_day(&mut res_vol, obs, &vec![vg; obs.len()], sd_vol);
            add_stock_day(&mut res_volpx, obs, &vec![vg * 4 + pg; obs.len()], sd_vol);
        }

        // FWL：剔掉点差与价格水平
        {
            let x: Vec<f64> = obs.iter().map(|o| o.imb).collect();
            let sp: Vec<f64> = obs.iter().map(|o| o.spread_bps).collect();
            let lp: Vec<f64> = obs.iter().map(|o| o.price.ln()).collect();
            let rx = residualize(&x, &sp, &lp);
            let rl: Vec<f64> = {
                // 收益也残差化，才是干净的偏相关
                let y: Vec<f64> = obs.iter().map(|o| o.r_mid).collect();
                residualize(&y, &sp, &lp)
            };
            if let Some(v) = pearson(&rx, &rl) {
                if v.is_finite() {
                    res_fwl[0].names += 1.0; res_fwl[0].obs += obs.len() as f64;
                    res_fwl[0].sr_mid += v; res_fwl[0].sr_mid2 += v * v;
                    let s = &mut res_fwl_px[pg];
                    s.names += 1.0; s.obs += obs.len() as f64;
                    s.sr_mid += v; s.sr_mid2 += v * v;
                }
            }
        }

        // 观测级相关
        for o in obs {
            let (lp, sp, ib) = (o.price.ln(), o.spread_bps, o.imb);
            n_obs += 1.0;
            s_px += lp; s_sp += sp; s_ib += ib;
            s_pp += lp * lp; s_ss += sp * sp; s_ii += ib * ib;
            s_ps += lp * sp; s_pi += lp * ib; s_si += sp * ib;
            s_amt += o.depth; s_aa += o.depth * o.depth;
            s_pp2 += o.price * o.price;
            s_pa += o.price * o.depth;
        }
        total_obs += obs.len() as f64;
        eff_total += eff_obs(obs);
    }
    let _ = (s_pp2, s_aa);
    let corr_px_sp = corr2(n_obs, s_px, s_sp, s_pp, s_ss, s_ps);
    let corr_px_ib = corr2(n_obs, s_px, s_ib, s_pp, s_ii, s_pi);
    let corr_sp_ib = corr2(n_obs, s_sp, s_ib, s_ss, s_ii, s_si);
    let corr_px_depth = corr2(n_obs, s_px, s_amt, s_pp, s_aa, s_pa);

    // 配对差全量 + 200 次随机对折：检验报出的误差要不要放大
    let mut pd_all: Vec<f64> = Vec::new();
    for (_c, obs) in &data {
        let am: Vec<usize> = (0..obs.len()).filter(|&k| obs[k].lt < 11 * 3600 + 30 * 60).collect();
        let pm: Vec<usize> = (0..obs.len()).filter(|&k| obs[k].lt >= 11 * 3600 + 30 * 60).collect();
        if am.len() < MIN_BUCKET || pm.len() < MIN_BUCKET { continue; }
        let xa: Vec<f64> = am.iter().map(|&k| obs[k].imb).collect();
        let ya: Vec<f64> = am.iter().map(|&k| obs[k].r_mid).collect();
        let xp: Vec<f64> = pm.iter().map(|&k| obs[k].imb).collect();
        let yp: Vec<f64> = pm.iter().map(|&k| obs[k].r_mid).collect();
        if let (Some(a), Some(b)) = (spearman(&xa, &ya), spearman(&xp, &yp)) { pd_all.push(b - a); }
    }
    let pd_mean = if pd_all.is_empty() { f64::NAN } else { pd_all.iter().sum::<f64>() / pd_all.len() as f64 };
    let pd_se = if pd_all.len() < 2 { f64::NAN } else {
        let v = pd_all.iter().map(|x| (x - pd_mean) * (x - pd_mean)).sum::<f64>() / (pd_all.len() as f64 - 1.0);
        (v / pd_all.len() as f64).sqrt()
    };
    let mut rng: u64 = 0x9E3779B97F4A7C15;
    let mut diffs: Vec<f64> = Vec::new();
    for _ in 0..200 {
        let (mut s0, mut n0, mut s1, mut n1) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
        for d in &pd_all {
            rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
            if rng & 1 == 0 { s0 += d; n0 += 1.0; } else { s1 += d; n1 += 1.0; }
        }
        if n0 > 0.0 && n1 > 0.0 { diffs.push(s0 / n0 - s1 / n1); }
    }
    let dif_sd = if diffs.is_empty() { f64::NAN } else {
        let m = diffs.iter().sum::<f64>() / diffs.len() as f64;
        (diffs.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / (diffs.len() as f64 - 1.0)).sqrt()
    };
    let mut parts_half = format!("{{\"name\":\"paired_ic_diff_all\",\"mean\":{:.5},\"se\":{:.5},\"n\":{:.0},\"split_sd\":{:.5},\"need_factor\":{:.3}}}",
        pd_mean, pd_se, pd_all.len() as f64, dif_sd, if pd_se.is_finite() && pd_se > 0.0 { dif_sd / (pd_se * 2f64.sqrt()) } else { f64::NAN });
    // 时段配对：每只股票算（午后 − 上午）的 IC 差，按价格桶分组；四格均值 + 股票间 SE
    let mut tod_grid = vec![Sums::default(); 4];
    let mut tod_d = vec![Sums::default(); 2];
    // 低价桶一跳占比
    let mut tick_hist = [0f64; 4]; // 低价桶里 spread_ticks = 1 / 2 / 3 / >=4 的观测数
    for (_c, obs) in &data {
        let mut prs: Vec<f64> = obs.iter().map(|o| o.price).collect();
        prs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let med = prs[prs.len() / 2];
        let pg = pgroup(med);
        let cheap = if pg <= 1 { 0usize } else { 1usize };
        let am: Vec<usize> = (0..obs.len()).filter(|&k| obs[k].lt < 11 * 3600 + 30 * 60).collect();
        let pm: Vec<usize> = (0..obs.len()).filter(|&k| obs[k].lt >= 11 * 3600 + 30 * 60).collect();
        if pg == 0 { for &k in obs.iter().map(|_| 0).collect::<Vec<usize>>().iter() { let _ = k; } for o in obs { if o.price < 5.0 { let b = (o.spread_ticks.clamp(1, 4) - 1) as usize; tick_hist[b] += 1.0; } } }
        if am.len() < MIN_BUCKET || pm.len() < MIN_BUCKET { continue; }
        let ic_of = |ix: &[usize]| -> Option<f64> {
            let x: Vec<f64> = ix.iter().map(|&k| obs[k].imb).collect();
            let y: Vec<f64> = ix.iter().map(|&k| obs[k].r_mid).collect();
            spearman(&x, &y)
        };
        let (a_ic, p_ic) = match (ic_of(&am), ic_of(&pm)) { (Some(a), Some(p)) => (a, p), _ => continue };
        let g = am.len();
        // 四个格：廉价/昂贵 × 上午/午后（同一个 Sums 结构，只是借用 ic_mid 字段）
        let cell = cheap * 2;
        tod_grid[cell].names += 1.0; tod_grid[cell].sr_mid += a_ic; tod_grid[cell].sr_mid2 += a_ic * a_ic; tod_grid[cell].obs += g as f64;
        tod_grid[cell + 1].names += 1.0; tod_grid[cell + 1].sr_mid += p_ic; tod_grid[cell + 1].sr_mid2 += p_ic * p_ic; tod_grid[cell + 1].obs += pm.len() as f64;
        let d = p_ic - a_ic;
        tod_d[cheap].names += 1.0; tod_d[cheap].sr_mid += d; tod_d[cheap].sr_mid2 += d * d;
    }
    let mut parts: Vec<String> = Vec::new();
    parts.push(format!("{{\"name\":\"corr_lnpx_vs_spreadbps\",\"value\":{}}}", jnum(corr_px_sp)));
    parts.push(format!("{{\"name\":\"corr_lnpx_vs_imb\",\"value\":{}}}", jnum(corr_px_ib)));
    parts.push(format!("{{\"name\":\"corr_spreadbps_vs_imb\",\"value\":{}}}", jnum(corr_sp_ib)));
    parts.push(format!("{{\"name\":\"corr_price_vs_depth5\",\"value\":{}}}", jnum(corr_px_depth)));
    for (i, n) in PNAME.iter().enumerate() { parts.push(json_group(n, &res_px[i])); }
    let qn = ["q1_narrow", "q2", "q3", "q4", "q5_wide"];
    let qn2 = qn;
    for i in 0..5 { parts.push(json_group(qn[i], &res_spq[i])); }
    for p in 0..4 {
        for q in 0..5 {
            parts.push(json_group(&format!("{}__{}", PNAME[p], qn[q]), &res_cell[p * 5 + q]));
        }
    }
    parts.push(json_group("rel_narrow_half", &res_rel[0]));
    parts.push(json_group("rel_wide_half", &res_rel[1]));
    for p in 0..4 { parts.push(format!("{{\"name\":\"zero_vol_share_{}\",\"value\":{:.4}}}", PNAME[p], zero_cnt[p] / res_px[p].obs.max(1.0))); }
    for p in 0..4 { parts.push(format!("{{\"name\":\"lowpx_tick_share_{}\",\"value\":{:.4}}}", p + 1, tick_hist[p] / tick_hist.iter().sum::<f64>().max(1.0))); }
    {
        let cn = ["pair_cheap_am", "pair_cheap_pm", "pair_dear_am", "pair_dear_pm"];
        for i in 0..4 {
            let s = &tod_grid[i];
            let n = s.names.max(1.0);
            let m = s.sr_mid / n;
            let se = ((s.sr_mid2 / n - m * m).max(0.0) / n).sqrt();
            parts.push(format!("{{\"name\":\"{}\",\"stock_days\":{:.0},\"ic_mid\":{:.4},\"ic_mid_se\":{:.4}}}", cn[i], s.names, m, se));
        }
        for i in 0..2 {
            let s = &tod_d[i];
            let n = s.names.max(1.0);
            let m = s.sr_mid / n;
            let se = ((s.sr_mid2 / n - m * m).max(0.0) / n).sqrt();
            parts.push(format!("{{\"name\":\"pair_diff_{}\",\"stock_days\":{:.0},\"mean\":{:.5},\"se\":{:.5}}}", if i == 0 { "cheap" } else { "dear" }, s.names, m, se));
        }
    }
    if !parts_half.is_empty() { parts.push(format!("{{\"name\":\"halves_check\",\"groups\":[{}]}}", parts_half)); }
    let bn = ["bin0_open", "bin1", "bin2", "bin3_mid", "bin4", "bin5", "bin6", "bin7_close"];
    let sq = ["q1n", "q2", "q3", "q4", "q5w"];
    for b in 0..8 { for q in 0..5 { parts.push(json_group(&format!("{}__{}", bn[b], sq[q]), &res_binsp[b * 5 + q])); } }
    let rn = ["resid_narrow", "resid_mid", "resid_wide"];
    for i in 0..3 { parts.push(json_group(rn[i], &res_resid3[i])); }
    for p in 0..4 { for i in 0..3 { parts.push(json_group(&format!("{}_{}", rn[i], PNAME[p]), &res_residpx[i + 3 * p])); } }
    parts.push(json_group("quiet_pastvol0", &res_pvz[0]));
    parts.push(json_group("active_pastvol_pos", &res_pvz[1]));
    for p in 0..4 { parts.push(json_group(&format!("quiet_{}", PNAME[p]), &res_pvzpx[p])); }
    for p in 0..4 { parts.push(json_group(&format!("active_{}", PNAME[p]), &res_pvzpx[4 + p])); }
    let an = ["abs_sp_q1", "abs_sp_q2", "abs_sp_q3", "abs_sp_q4", "abs_sp_q5"];
    for i in 0..5 { parts.push(json_group(an[i], &res_absq[i])); }
    for p in 0..4 { for q in 0..5 { parts.push(json_group(&format!("{}_{}", PNAME[p], an[q]), &res_absq_cell[q + 5 * p])); } }
    let inb = ["imb_weak", "imb_mid", "imb_strong"];
    for i in 0..3 { parts.push(json_group(inb[i], &res_imbq[i])); }
    parts.push(format!("{{\"name\":\"allzero_vol_days\",\"value\":{:.4}}}", d_allzero / d_all.max(1.0)));
    parts.push(format!("{{\"name\":\"pv_nonzero_stock_days\",\"value\":{:.0}}}", d_all - d_allzero));
    for i in 0..3 { parts.push(json_group(&format!("pv_nz_{}", i), &res_pv_nonzero[i])); }
    for tb in 0..6 {
        for vg in 0..3 {
            parts.push(json_group(&format!("todbin{}_pv{}", tb, vg), &res_todpv[vg * 6 + tb]));
        }
    }
    let inbsd = ["imb_sd_low", "imb_sd_mid", "imb_sd_high"];
    for i in 0..3 { parts.push(json_group(inbsd[i], &res_imbsd[i])); }
    let pnz = ["nz_pastvol_low", "nz_pastvol_mid", "nz_pastvol_high"];
    for i in 0..3 { parts.push(json_group(pnz[i], &res_nz[i])); }
    for v in 0..3 { for p in 0..4 { parts.push(json_group(&format!("{}_{}", pnz[v], PNAME[p]), &res_nzpx[v * 4 + p])); } }
    let pn = ["pastvol_low", "pastvol_mid", "pastvol_high"];
    for i in 0..3 { parts.push(json_group(pn[i], &res_pv[i])); }
    for v in 0..3 { for p in 0..4 { parts.push(json_group(&format!("{}_{}", pn[v], PNAME[p]), &res_pvpx[v * 4 + p])); } }
    for v in 0..3 { for q in 0..5 { parts.push(json_group(&format!("{}_px_ge100_{}", pn[v], qn2[q]), &res_pvsp[v * 5 + q])); } }
    let tn = ["tod_am_cheap", "tod_am_dear", "tod_mid_cheap", "tod_mid_dear", "tod_pm_cheap", "tod_pm_dear"];
    for i in 0..6 { parts.push(json_group(tn[i], &res_tod[i])); }
    let en = ["relwide_ext", "relwide_mid", "relnarrow_ext", "relnarrow_mid"];
    for i in 0..4 { parts.push(json_group(en[i], &res_ext[i])); }
    let vn = ["vol_low", "vol_mid", "vol_high"];
    for i in 0..3 { parts.push(json_group(vn[i], &res_vol[i])); }
    for v in 0..3 { for p in 0..4 { parts.push(json_group(&format!("{}_{}", vn[v], PNAME[p]), &res_volpx[v * 4 + p])); } }
    parts.push(json_group("fwl_all", &res_fwl[0]));
    for (i, n) in PNAME.iter().enumerate() { parts.push(json_group(&format!("fwl_{}", n), &res_fwl_px[i])); }
    {
        let n = res_all[0].pdiff_n.max(1.0);
        let m = res_all[0].pdiff / n;
        let se = ((res_all[0].pdiff2 / n - m * m).max(0.0) / n).sqrt();
        parts.push(format!("{{\"name\":\"paired_spread_ic_diff\",\"stock_days\":{:.0},\"mean\":{:.5},\"se\":{:.5}}}", res_all[0].pdiff_n, m, se));
    }
    parts.push(json_group("all", &res_all[0]));

    // 描述性：日内噪声与跳数/绝对点差的秩相关（每只股票日算一次，再取平均）
    {
        let (mut s1, mut s1n, mut s2, mut s2n, mut s3, mut s3n) = (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
        for (_c, obs) in &data {
            let ok: Vec<usize> = (0..obs.len()).filter(|&k| obs[k].past_vol.is_finite()).collect();
            if ok.len() < MIN_BUCKET { continue; }
            let pv: Vec<f64> = ok.iter().map(|&k| obs[k].past_vol).collect();
            let tk: Vec<f64> = ok.iter().map(|&k| obs[k].spread_ticks as f64).collect();
            let bp: Vec<f64> = ok.iter().map(|&k| obs[k].spread_bps).collect();
            let ab: Vec<f64> = ok.iter().map(|&k| obs[k].spread_ticks as f64 * TICK).collect();
            if let Some(v) = spearman(&pv, &tk) { s1 += v; s1n += 1.0; }
            if let Some(v) = spearman(&pv, &bp) { s2 += v; s2n += 1.0; }
            if let Some(v) = spearman(&pv, &ab) { s3 += v; s3n += 1.0; }
        }
        let a = if s1n > 0.0 { s1 / s1n } else { f64::NAN };
        let b = if s2n > 0.0 { s2 / s2n } else { f64::NAN };
        let c = if s3n > 0.0 { s3 / s3n } else { f64::NAN };
        parts.push(format!("{{\"name\":\"corr_noise_vs_ticks\",\"value\":{}}}", jnum(a)));
        parts.push(format!("{{\"name\":\"corr_noise_vs_spreadbps\",\"value\":{}}}", jnum(b)));
        parts.push(format!("{{\"name\":\"corr_noise_vs_spreadabs\",\"value\":{}}}", jnum(c)));
    }

    println!("{{\"date\":{},\"stock_days\":{},\"obs\":{:.0},\"eff_obs\":{:.0},\"groups\":[{}]}}",
             date, data.len(), total_obs, eff_total, parts.join(","));

    // 稳健性：每 60 秒只取一个不重叠窗口，重算全样本与价格桶的 IC
    let mut acc_all = Sums::default();
    let mut acc_px: Vec<Sums> = vec![Sums::default(); 4];
    let mut n_sub = [0usize; 4];
    for (_c, obs) in &data {
        let mut keep: Vec<Obs> = Vec::new();
        let mut next_t = i64::MIN;
        for o in obs {
            if o.time_us >= next_t { keep.push(*o); next_t = o.time_us + FWD_US; }
        }
        if keep.len() < 30 { continue; }
        let sd = (keep.iter().map(|o| o.r_mid * o.r_mid).sum::<f64>() / keep.len() as f64).sqrt();
        let mut prs: Vec<f64> = keep.iter().map(|o| o.price).collect();
        prs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let med = prs[prs.len() / 2];
        let pg = if med < PB[0] { 0 } else if med < PB[1] { 1 } else if med < PB[2] { 2 } else { 3 };
        if std::env::var("XS_DEBUG_SUB").is_ok() { n_sub[pg] += 1; }
        {
            let one = &mut [acc_all];
            add_stock_day(one, &keep, &vec![0usize; keep.len()], sd);
            acc_all = one[0];
        }
        {
            // 注意：分组标签必须落在同一长度的切片里，否则只累加第 0 组
            let one = &mut acc_px[pg..pg + 1];
            add_stock_day(one, &keep, &vec![0usize; keep.len()], sd);
        }
    }
    if std::env::var("XS_DEBUG_SUB").is_ok() { eprintln!("[{}] sub kept per price bucket: {:?}", date, n_sub); }
    // 不重叠样本上的过去波动三档 x 价格桶（每只股票日内切档）
    let mut sub_pv: Vec<Sums> = vec![Sums::default(); 3];
    let mut sub_pvpx: Vec<Sums> = vec![Sums::default(); 12];
    for (_c, obs) in &data {
        let mut keep: Vec<Obs> = Vec::new();
        let mut next_t = i64::MIN;
        for o in obs { if o.time_us >= next_t { keep.push(*o); next_t = o.time_us + FWD_US; } }
        if keep.len() < 30 { continue; }
        let mut v: Vec<f64> = keep.iter().map(|o| if o.past_vol.is_finite() { o.past_vol } else { 0.0 }).collect();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let c1 = v[v.len() / 3]; let c2 = v[v.len() * 2 / 3];
        let lab: Vec<usize> = keep.iter().map(|o| { let p = if o.past_vol.is_finite() { o.past_vol } else { 0.0 }; if p <= c1 { 0 } else if p <= c2 { 1 } else { 2 } }).collect();
        let sd = (keep.iter().map(|o| o.r_mid * o.r_mid).sum::<f64>() / keep.len() as f64).sqrt();
        let mut prs: Vec<f64> = keep.iter().map(|o| o.price).collect();
        prs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let med = prs[prs.len() / 2];
        let pg = if med < PB[0] { 0 } else if med < PB[1] { 1 } else if med < PB[2] { 2 } else { 3 };
        {
            let one = &mut sub_pv[0..3];
            add_stock_day(one, &keep, &lab, sd);
        }
        {
            let labpx: Vec<usize> = lab.iter().map(|&v| v * 4 + pg).collect();
            let one = &mut sub_pvpx[0..12];
            add_stock_day(one, &keep, &labpx, sd);
        }
    }
    let mut sp2: Vec<String> = Vec::new();
    sp2.push(json_group("sub_all", &acc_all));
    for i in 0..3 { sp2.push(json_group(&format!("sub_pastvol_{}", i), &sub_pv[i])); }
    for v in 0..3 { for p in 0..4 { sp2.push(json_group(&format!("sub_pastvol{}_{}", v, PNAME[p]), &sub_pvpx[v * 4 + p])); } }
    for (i, n) in PNAME.iter().enumerate() { sp2.push(json_group(&format!("sub_{}", n), &acc_px[i])); }
    println!("{{\"date\":{},\"kind\":\"subsample60\",\"groups\":[{}]}}", date, sp2.join(","));
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    assert!(!args.is_empty(), "usage: x_spread_price <date> [date...]");
    for a in &args { run(a.parse::<i64>().expect("date must be YYYYMMDD")); }
}