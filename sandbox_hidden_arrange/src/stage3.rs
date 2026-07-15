//! 第三层：后验不确定性与主动查询（与 Python stage3 对照）
use crate::{Bank, average_rank, assign_by_rank};
use std::sync::atomic::{AtomicU64, Ordering};

static T_S3_PRIOR: AtomicU64 = AtomicU64::new(0);
static T_S3_PSTATS: AtomicU64 = AtomicU64::new(0);
static T_S3_GEN: AtomicU64 = AtomicU64::new(0);
static T_S3_SEL: AtomicU64 = AtomicU64::new(0);
static T_S3_UPD: AtomicU64 = AtomicU64::new(0);
static T_S3_HROW: AtomicU64 = AtomicU64::new(0);
#[inline] fn acc3(c: &AtomicU64, d: std::time::Duration) { c.fetch_add(d.as_nanos() as u64, Ordering::Relaxed); }
pub fn print_s3_timing() {
    println!("--- stage3 内部 ---");
    println!("  build_prior: {:.3}s", T_S3_PRIOR.load(Ordering::Relaxed) as f64*1e-9);
    println!("  posterior_stats: {:.3}s", T_S3_PSTATS.load(Ordering::Relaxed) as f64*1e-9);
    println!("  generate_candidates: {:.3}s", T_S3_GEN.load(Ordering::Relaxed) as f64*1e-9);
    println!("  select_query: {:.3}s", T_S3_SEL.load(Ordering::Relaxed) as f64*1e-9);
    println!("  update_posterior: {:.3}s", T_S3_UPD.load(Ordering::Relaxed) as f64*1e-9);
    println!("  row_entropy(h_cur): {:.3}s", T_S3_HROW.load(Ordering::Relaxed) as f64*1e-9);
}

pub const TAUS: [f64; 3] = [0.25, 0.50, 1.00];
pub const TAU_NAMES: [&str; 3] = ["TAU025", "TAU050", "TAU100"];
pub const CHECKPOINTS: [usize; 5] = [0, 8, 16, 32, 64];
const N_QUERIES: usize = 64;
const UPDATE_LR: f64 = 0.3;

// xorshift64 RNG，匹配 Python
pub struct Rng { s: u64 }
impl Rng {
    pub fn new(seed: u64) -> Self { Rng { s: (seed & u64::MAX) | 1 } }
    pub fn u64(&mut self) -> u64 {
        let mut x = self.s;
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        self.s = x; x
    }
    pub fn random(&mut self) -> f64 { self.u64() as f64 / 2f64.powi(64) }
    pub fn permutation(&mut self, n: usize) -> Vec<usize> {
        let mut a: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            let j = (self.random() * (i + 1) as f64) as usize;
            a.swap(i, j);
        }
        a
    }
}

fn sinkhorn(p: &mut [f64], n: usize, m: usize, col_target: &[f64], iters: usize) {
    for _ in 0..iters {
        for j in 0..m {
            let cs: f64 = (0..n).map(|i| p[i * m + j]).sum();
            let f = col_target[j] / cs.max(1e-12);
            for i in 0..n { p[i * m + j] *= f; }
        }
        for i in 0..n {
            let rs: f64 = (0..m).map(|j| p[i * m + j]).sum();
            let f = 1.0 / rs.max(1e-12);
            for j in 0..m { p[i * m + j] *= f; }
        }
    }
}

fn build_prior(g2: &[f64], y: &[f64], tau: f64) -> (Vec<f64>, Vec<f64>, usize, Vec<f64>) {
    let n = y.len();
    let mut classes: Vec<f64> = y.to_vec();
    classes.sort_by(|a, b| a.partial_cmp(b).unwrap()); classes.dedup();
    let m = classes.len();
    let mut c_m = vec![0f64; m];
    for &v in y {
        let idx = classes.binary_search_by(|c| c.partial_cmp(&v).unwrap()).unwrap();
        c_m[idx] += 1.0;
    }
    let r = average_rank(g2);
    let rn: Vec<f64> = r.iter().map(|x| x / (n as f64 - 1.0).max(1.0)).collect();
    let mut p = vec![0f64; n * m];
    for i in 0..n {
        for j in 0..m {
            let center = if m > 1 { j as f64 / (m - 1) as f64 } else { 0.0 };
            let d = rn[i] - center;
            p[i * m + j] = (-(d * d) / tau.max(1e-6)).exp();
        }
        let s: f64 = (0..m).map(|j| p[i * m + j]).sum();
        for j in 0..m { p[i * m + j] /= s; }
    }
    sinkhorn(&mut p, n, m, &c_m, 60);
    (p, c_m, m, classes)
}

fn sample_permutation(p: &[f64], c_m: &[f64], n: usize, m: usize, rng: &mut Rng) -> Vec<usize> {
    debug_assert!(m <= 64);
    let mut remaining = c_m.to_vec();
    let mut assign = vec![0usize; n];
    let order = rng.permutation(n);
    let us: Vec<f64> = (0..n).map(|_| rng.random()).collect();
    let mut prob = [0f64; 64];
    let mut cdf = [0f64; 64];
    for (k, &i) in order.iter().enumerate() {
        let mut s = 0f64;
        for j in 0..m { prob[j] = p[i * m + j] * remaining[j]; s += prob[j]; }
        let mm = if s <= 1e-12 {
            let mut best = 0; let mut bv = -1f64;
            for j in 0..m { if remaining[j] > bv { bv = remaining[j]; best = j; } }
            best
        } else {
            let mut acc = 0.0;
            for j in 0..m { acc += prob[j] / s; cdf[j] = acc; }
            let mut idx = m;
            for j in 0..m { if us[k] <= cdf[j] { idx = j; break; } }
            idx.min(m - 1)
        };
        assign[i] = mm;
        remaining[mm] -= 1.0;
    }
    assign
}

fn row_entropy(p: &[f64], n: usize, m: usize) -> Vec<f64> {
    (0..n).map(|i| {
        let mut h = 0.0;
        for j in 0..m { let v = p[i * m + j]; if v > 1e-12 { h -= v * v.ln(); } }
        h
    }).collect()
}

fn make_group_candidate(y_sorted: &[f64], g_idx: &[usize], ranks_guide: &[f64], high_big: bool) -> Vec<f64> {
    let n = y_sorted.len();
    let mut present = vec![false; n];
    for &i in g_idx { present[i] = true; }
    let h: Vec<usize> = (0..n).filter(|&i| !present[i]).collect();
    let ng = g_idx.len(); let nh = h.len();
    if ng == 0 || nh == 0 { return y_sorted.to_vec(); }
    let (g_vals, h_vals): (Vec<f64>, Vec<f64>) = if high_big {
        (y_sorted[nh..].to_vec(), y_sorted[..nh].to_vec())
    } else { (y_sorted[..ng].to_vec(), y_sorted[ng..].to_vec()) };
    let mut q = vec![0f64; n];
    let mut go = g_idx.to_vec();
    go.sort_by(|&a, &b| ranks_guide[a].partial_cmp(&ranks_guide[b]).unwrap());
    let mut gv = g_vals; gv.sort_by(|a, b| a.partial_cmp(b).unwrap());
    for (k, &i) in go.iter().enumerate() { q[i] = gv[k]; }
    let mut ho = h;
    ho.sort_by(|&a, &b| ranks_guide[a].partial_cmp(&ranks_guide[b]).unwrap());
    let mut hv = h_vals; hv.sort_by(|a, b| a.partial_cmp(b).unwrap());
    for (k, &i) in ho.iter().enumerate() { q[i] = hv[k]; }
    q
}

struct Cand { ctype: u8, q: Vec<f64> }

fn quantile(x: &[f64], q: f64) -> f64 {
    let mut s = x.to_vec(); s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = s.len();
    let pos = (n - 1) as f64 * q;
    let lo = pos.floor() as usize; let hi = pos.ceil() as usize;
    if lo == hi { s[lo] } else { s[lo] + (pos - lo as f64) * (s[hi] - s[lo]) }
}

fn generate_candidates(p: &[f64], c_m: &[f64], y: &[f64], g2: &[f64], n: usize, m: usize, rng: &mut Rng, classes: &[f64]) -> Vec<Cand> {
    let e: Vec<f64> = (0..n).map(|i| (0..m).map(|j| j as f64 * p[i * m + j]).sum::<f64>()).collect();
    let ranks_guide = average_rank(&e);
    let mut ys_sorted = y.to_vec(); ys_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mut cands: Vec<Cand> = vec![];
    // 8 分歧：抽 16 个排列
    let mut perms: Vec<Vec<usize>> = vec![];
    for _ in 0..16 { perms.push(sample_permutation(p, c_m, n, m, rng)); }
    for k in 0..8 {
        let ya = &perms[2 * k];
        let q: Vec<f64> = ya.iter().map(|&idx| classes[idx]).collect();
        cands.push(Cand { ctype: 0, q });
    }
    // 8 结构
    let r2 = average_rank(g2);
    let qps = [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9];
    for (k, &qp) in qps.iter().enumerate() {
        let thr = quantile(&r2, qp);
        let g: Vec<usize> = if k % 2 == 0 {
            (0..n).filter(|&i| r2[i] >= thr).collect()
        } else { (0..n).filter(|&i| r2[i] < thr).collect() };
        let g = if g.is_empty() || g.len() == n {
            let mut s: Vec<usize> = (0..n).collect();
            s.sort_by(|&a, &b| r2[a].partial_cmp(&r2[b]).unwrap());
            s[n - n / 2..].to_vec()
        } else { g };
        let q = make_group_candidate(&ys_sorted, &g, &ranks_guide, k % 2 == 0);
        cands.push(Cand { ctype: 1, q });
    }
    // 8 高熵
    let h = row_entropy(p, n, m);
    let mut idx_h: Vec<usize> = (0..n).collect();
    idx_h.sort_by(|&a, &b| h[b].partial_cmp(&h[a]).unwrap());
    let top_half: Vec<usize> = idx_h[..n / 2].to_vec();
    for seed in 0..8 {
        let mut r = Rng::new(1000 + seed);
        let perm = r.permutation(top_half.len());
        let half = top_half.len() / 2;
        let g: Vec<usize> = (0..half).map(|k| top_half[perm[k]]).collect();
        let q = make_group_candidate(&ys_sorted, &g, &ranks_guide, true);
        cands.push(Cand { ctype: 2, q });
    }
    // 8 随机平衡
    for seed in 0..8 {
        let mut r = Rng::new(2000 + seed);
        let rand_score: Vec<f64> = (0..n).map(|_| r.random()).collect();
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| rand_score[a].partial_cmp(&rand_score[b]).unwrap());
        let mut q = vec![0f64; n];
        for (k, &i) in order.iter().enumerate() { q[i] = ys_sorted[k]; }
        cands.push(Cand { ctype: 3, q });
    }
    cands
}

fn select_query(cands: &[Cand], y: &[f64], p: &[f64], c_m: &[f64], n: usize, m: usize, classes: &[f64], rng: &mut Rng) -> (usize, Vec<f64>) {
    let mu = y.iter().sum::<f64>() / n as f64;
    let var = y.iter().map(|v| (v - mu).powi(2)).sum::<f64>() / n as f64;
    let denom = 2.0 * var.max(1e-12) * n as f64;
    let mut samples: Vec<Vec<f64>> = vec![];
    for _ in 0..32 {
        let perm = sample_permutation(p, c_m, n, m, rng);
        samples.push(perm.iter().map(|&idx| classes[idx]).collect());
    }
    let s_sq: Vec<f64> = samples.iter().map(|s| s.iter().map(|v| v * v).sum::<f64>()).collect();
    let best_pri = [0u8, 1, 2, 3];
    let mut best_idx = 0; let mut best_var = -1.0; let mut best_pri_val = 99u8;
    let mut var_list: Vec<f64> = vec![];
    let nsamp = samples.len();
    let mut rho_buf = [0f64; 40];
    for (idx, c) in cands.iter().enumerate() {
        let q_sq: f64 = c.q.iter().map(|v| v * v).sum();
        let mut sum_rho = 0f64;
        for t in 0..nsamp {
            let s = &samples[t];
            let dot: f64 = s.iter().zip(c.q.iter()).map(|(a, b)| a * b).sum();
            let rho = 1.0 - (q_sq + s_sq[t] - 2.0 * dot) / denom;
            rho_buf[t] = rho;
            sum_rho += rho;
        }
        let mean: f64 = sum_rho / nsamp as f64;
        let mut v = 0f64;
        for t in 0..nsamp { let d = rho_buf[t] - mean; v += d * d; }
        v /= nsamp as f64;
        var_list.push(v);
        let pri = best_pri[c.ctype as usize];
        if v > best_var + 1e-12 || (v - best_var).abs() <= 1e-12 && pri < best_pri_val {
            best_var = v; best_idx = idx; best_pri_val = pri;
        }
    }
    (best_idx, var_list)
}

fn update_posterior(p: &[f64], q: &[f64], r_bank: f64, c_m: &[f64], classes: &[f64], n: usize, m: usize) -> Vec<f64> {
    let s = r_bank.clamp(-1.0, 1.0);
    if s.abs() < 1e-9 { return p.to_vec(); }
    let q_class: Vec<usize> = q.iter().map(|&v| classes.binary_search_by(|c| c.partial_cmp(&v).unwrap()).unwrap()).collect();
    let lr_eff = UPDATE_LR * s.abs();
    let mut target = vec![0f64; n * m];
    if s > 0.0 {
        for i in 0..n {
            for j in 0..m { target[i * m + j] = (1.0 - lr_eff) * p[i * m + j]; }
            target[i * m + q_class[i]] += lr_eff;
        }
    } else {
        let dm = (m - 1).max(1) as f64;
        for i in 0..n {
            for j in 0..m {
                let v = if j == q_class[i] { 0.0 } else { 1.0 / dm };
                target[i * m + j] = (1.0 - lr_eff) * p[i * m + j] + lr_eff * v;
            }
        }
    }
    for v in target.iter_mut() { if *v < 1e-9 { *v = 1e-9; } }
    sinkhorn(&mut target, n, m, c_m, 30);
    target
}

fn posterior_stats(p: &[f64], y: &[f64], n: usize, m: usize, classes: &[f64], cnt: &[f64]) -> Vec<f64> {
    let h = row_entropy(p, n, m);
    let h_mean = h.iter().sum::<f64>() / n as f64;
    let top1: Vec<f64> = (0..n).map(|i| (0..m).map(|j| p[i * m + j]).fold(0f64, |a, b| a.max(b))).collect();
    let post_mean: Vec<f64> = (0..n).map(|i| (0..m).map(|j| j as f64 * p[i * m + j]).sum()).collect();
    let post_var: Vec<f64> = (0..n).map(|i| {
        let mean = post_mean[i];
        (0..m).map(|j| (j as f64).powi(2) * p[i * m + j]).sum::<f64>() - mean * mean
    }).collect();
    fn tail_set(m: usize, classes: &[f64], cnt: &[f64], n: usize, frac: f64) -> Vec<usize> {
        let mut order: Vec<usize> = (0..m).collect();
        order.sort_by(|&a, &b| classes[b].partial_cmp(&classes[a]).unwrap());
        let mut cum = 0.0; let mut s = Vec::new();
        for &ci in &order { s.push(ci); cum += cnt[ci]; if cum >= frac * n as f64 { break; } }
        s
    }
    fn tail_entropy(p: &[f64], n: usize, m: usize, s: &[usize]) -> f64 {
        let mut tot = 0.0;
        for i in 0..n {
            let pb = s.iter().map(|&j| p[i * m + j]).sum::<f64>().clamp(1e-9, 1.0 - 1e-9);
            tot += -(pb * pb.ln() + (1.0 - pb) * (1.0 - pb).ln());
        }
        tot / n as f64
    }
    fn tail_conf(p: &[f64], n: usize, m: usize, s: &[usize]) -> f64 {
        let mut tot = 0.0;
        for i in 0..n { tot += s.iter().map(|&j| p[i * m + j]).sum::<f64>(); }
        tot / n as f64
    }
    let s20 = tail_set(m, classes, cnt, n, 0.20);
    let s10 = tail_set(m, classes, cnt, n, 0.10);
    let s05 = tail_set(m, classes, cnt, n, 0.05);
    let s01 = tail_set(m, classes, cnt, n, 0.01);
    vec![
        h_mean,
        h.iter().map(|x| (x - h_mean).powi(2)).sum::<f64>() / n as f64,
        quantile(&h, 0.90),
        h.iter().cloned().fold(0f64, |a, b| a.max(b)),
        top1.iter().sum::<f64>() / n as f64,
        top1.iter().filter(|&&x| x < 0.5).count() as f64 / n as f64,
        post_var.iter().sum::<f64>() / n as f64,
        tail_entropy(p, n, m, &s20), tail_entropy(p, n, m, &s10),
        tail_entropy(p, n, m, &s05), tail_entropy(p, n, m, &s01),
        tail_conf(p, n, m, &s10), tail_conf(p, n, m, &s05),
    ]
}

pub struct TauResult {
    pub tau: f64,
    pub rho3: f64,
    pub rho2: f64,
    pub checkpoints: std::collections::HashMap<usize, Vec<f64>>,
    pub info_gains: Vec<f64>,
    pub query_types: Vec<u8>,
    pub pred_vars: Vec<f64>,
}

pub struct Stage3Result {
    pub taus: Vec<TauResult>,
}

impl Stage3Result {
    pub fn to_json(&self) -> String {
        let mut out = String::from("{");
        for (ti, t) in self.taus.iter().enumerate() {
            let h0 = t.checkpoints[&0][0];
            let hk = t.checkpoints.keys().copied().max().unwrap_or(0);
            let hv = t.checkpoints.get(&hk).map(|v| v[0]).unwrap_or(h0);
            let tn = format!("TAU{:03}", (t.tau * 100.0) as i64);
            let mut cnts = [0usize; 4];
            for &qt in &t.query_types { cnts[qt as usize] += 1; }
            if ti > 0 { out.push(','); }
            out.push_str(&format!("\"{}\":{{\"rho3\":{:.4},\"K0_entropy\":{:.6},\"K{}_entropy\":{:.6},\"total_info_gain\":{:.6},\"qt\":[{},{},{},{}]}}",
                tn, t.rho3, h0, hk, hv, h0 - hv, cnts[0], cnts[1], cnts[2], cnts[3]));
        }
        out.push('}');
        out
    }
}

pub fn run_stage3(g2: &[f64], y: &[f64], tau_list: &[f64], n_query: usize) -> Stage3Result {
    let n = y.len();
    let bank = Bank::new(y.to_vec());
    let rho2 = bank.query(g2);
    let mut taus = vec![];
    for &tau in tau_list.iter() {
        let t0 = std::time::Instant::now();
        let (mut p, c_m, m, classes) = build_prior(g2, y, tau);
        acc3(&T_S3_PRIOR, t0.elapsed());
        let cnt: Vec<f64> = (0..m).map(|j| y.iter().filter(|&&v| v == classes[j]).count() as f64).collect();
        let mut rng = Rng::new(42);
        let mut ck = std::collections::HashMap::new();
        let t0 = std::time::Instant::now();
        ck.insert(0, posterior_stats(&p, y, n, m, &classes, &cnt));
        acc3(&T_S3_PSTATS, t0.elapsed());
        let t0 = std::time::Instant::now();
        let mut h_prev = row_entropy(&p, n, m).iter().sum::<f64>() / n as f64;
        acc3(&T_S3_HROW, t0.elapsed());
        let mut query_types: Vec<u8> = vec![];
        let mut info_gains: Vec<f64> = vec![];
        let mut pred_vars: Vec<f64> = vec![];
        for k in 1..=n_query {
            let tg0 = std::time::Instant::now();
            let cands = generate_candidates(&p, &c_m, y, g2, n, m, &mut rng, &classes);
            acc3(&T_S3_GEN, tg0.elapsed());
            let ts0 = std::time::Instant::now();
            let (idx, var_list) = select_query(&cands, y, &p, &c_m, n, m, &classes, &mut rng);
            acc3(&T_S3_SEL, ts0.elapsed());
            let q = cands[idx].q.clone();
            let r_bank = bank.query(&q);
            let tu0 = std::time::Instant::now();
            p = update_posterior(&p, &q, r_bank, &c_m, &classes, n, m);
            acc3(&T_S3_UPD, tu0.elapsed());
            let t0 = std::time::Instant::now();
            let h_cur = row_entropy(&p, n, m).iter().sum::<f64>() / n as f64;
            acc3(&T_S3_HROW, t0.elapsed());
            info_gains.push(h_prev - h_cur);
            pred_vars.push(var_list[idx]);
            query_types.push(cands[idx].ctype);
            h_prev = h_cur;
            if k <= n_query && [0usize, 8, 16, 32, 64].contains(&k) {
                let t0 = std::time::Instant::now();
                ck.insert(k, posterior_stats(&p, y, n, m, &classes, &cnt));
                acc3(&T_S3_PSTATS, t0.elapsed());
            }
        }
        let e: Vec<f64> = (0..n).map(|i| (0..m).map(|j| j as f64 * p[i * m + j]).sum()).collect();
        let ranks = average_rank(&e);
        let g3 = assign_by_rank(&ranks, y);
        let rho3 = bank.query(&g3);
        taus.push(TauResult { tau, rho3, rho2, checkpoints: ck, info_gains, query_types, pred_vars });
    }
    Stage3Result { taus }
}
