//! 隐藏排列三层游戏 —— Rust sandbox（与 Python 参考实现逐数值对照）
//! 第一阶段：数据读取 + 目标 + 银行 + 第一层，输出 JSON 对照。
//! 不使用并行；纯 std。
use std::fs::File;
use std::io::{Read, Write};

mod fast_csv_reader;
mod data_prep;
mod stage2;
mod stage3;
mod factors;

// ===================== 数据结构 =====================
#[derive(Clone)]
pub struct Window {
    pub n: usize,
    pub volume: Vec<i64>,
    pub price: Vec<f64>,
    pub flag: Vec<i32>,
    pub bid_order: Vec<i64>,
    pub ask_order: Vec<i64>,
    pub time_sec: Vec<f64>,
    pub active_side: Vec<i8>,
    pub active_order: Vec<i64>,
    pub passive_order: Vec<i64>,
    pub book_feats: Vec<f64>,    // n*14
    pub queue_feats: Vec<f64>,   // n*10
    pub arrival_feats: Vec<f64>, // n*6
}

impl Window {
    fn load(path: &str) -> std::io::Result<Self> {
        let mut f = File::open(path)?;
        let mut buf = Vec::new();
        f.read_to_end(&mut buf)?;
        let mut p = 0;
        let magic = &buf[p..p + 4]; p += 4;
        assert_eq!(magic, b"HAG0", "magic mismatch");
        let rd_i32 = |b: &[u8], o: &mut usize| -> i32 {
            let v = i32::from_le_bytes(b[*o..*o + 4].try_into().unwrap()); *o += 4; v
        };
        let rd_i64 = |b: &[u8], o: &mut usize| -> i64 {
            let v = i64::from_le_bytes(b[*o..*o + 8].try_into().unwrap()); *o += 8; v
        };
        let rd_f64 = |b: &[u8], o: &mut usize| -> f64 {
            let v = f64::from_le_bytes(b[*o..*o + 8].try_into().unwrap()); *o += 8; v
        };
        let n = rd_i32(&buf, &mut p) as usize;
        let _nfb = rd_i32(&buf, &mut p);
        let _nfq = rd_i32(&buf, &mut p);
        let _nfa = rd_i32(&buf, &mut p);
        let mut volume = vec![0i64; n];
        for i in 0..n { volume[i] = rd_i64(&buf, &mut p); }
        let mut price = vec![0f64; n];
        for i in 0..n { price[i] = rd_f64(&buf, &mut p); }
        let mut flag = vec![0i32; n];
        for i in 0..n { flag[i] = rd_i32(&buf, &mut p); }
        let mut bid_order = vec![0i64; n];
        for i in 0..n { bid_order[i] = rd_i64(&buf, &mut p); }
        let mut ask_order = vec![0i64; n];
        for i in 0..n { ask_order[i] = rd_i64(&buf, &mut p); }
        let mut time_sec = vec![0f64; n];
        for i in 0..n { time_sec[i] = rd_f64(&buf, &mut p); }
        let mut active_side = vec![0i8; n];
        for i in 0..n {
            active_side[i] = i8::from_le_bytes([buf[p]]); p += 1;
        }
        let mut active_order = vec![0i64; n];
        for i in 0..n { active_order[i] = rd_i64(&buf, &mut p); }
        let mut passive_order = vec![0i64; n];
        for i in 0..n { passive_order[i] = rd_i64(&buf, &mut p); }
        // 特征矩阵
        let rd_f64s = |b: &[u8], o: &mut usize, cnt: usize| -> Vec<f64> {
            let mut v = vec![0f64; cnt];
            for i in 0..cnt { v[i] = rd_f64(b, o); }
            v
        };
        let nfb = _nfb as usize;
        let nfq = _nfq as usize;
        let nfa = _nfa as usize;
        let book_feats = rd_f64s(&buf, &mut p, n * nfb);
        let queue_feats = rd_f64s(&buf, &mut p, n * nfq);
        let arrival_feats = rd_f64s(&buf, &mut p, n * nfa);
        Ok(Window { n, volume, price, flag, bid_order, ask_order, time_sec,
                    active_side, active_order, passive_order,
                    book_feats, queue_feats, arrival_feats })
    }
}

// ===================== 目标构造 =====================
fn percentile(x: &[f64], q: f64) -> f64 {
    // 匹配 np.percentile linear 插值
    let n = x.len();
    if n == 0 { return 0.0; }
    let mut s: Vec<f64> = x.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let pos = (n - 1) as f64 * q / 100.0;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    if lo == hi { return s[lo]; }
    let frac = pos - lo as f64;
    s[lo] + frac * (s[hi] - s[lo])
}

fn make_target(volume: &[i64], name: &str) -> Vec<f64> {
    let v: Vec<f64> = volume.iter().map(|&x| x as f64).collect();
    match name {
        "LOGVOL" => v.iter().map(|x| (1.0 + x).ln()).collect(),
        "RANKVOL" => percentile_rank(&v),
        "BIN8" | "BIN16" | "BIN32" => {
            let nbin: usize = name[3..].parse().unwrap();
            bin_equal(&v, nbin)
        }
        _ if name.starts_with("TOP") => {
            let q = 100.0 - name[3..].parse::<f64>().unwrap();
            let thr = percentile(&v, q);
            v.iter().map(|x| if *x >= thr { 1.0 } else { 0.0 }).collect()
        }
        _ => panic!("unknown target {}", name),
    }
}

fn percentile_rank(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| x[a].partial_cmp(&x[b]).unwrap()); // mergesort 稳定
    let mut ranks = vec![0f64; n];
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j + 1 < n && x[idx[j + 1]] == x[idx[i]] { j += 1; }
        let avg = (i as f64 + j as f64) / 2.0;
        let val = if n > 1 { avg / (n - 1) as f64 } else { 0.0 };
        for k in i..=j { ranks[idx[k]] = val; }
        i = j + 1;
    }
    ranks
}

fn bin_equal(x: &[f64], nbin: usize) -> Vec<f64> {
    let n = x.len();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| x[a].partial_cmp(&x[b]).unwrap());
    let mut ranks = vec![0i64; n];
    let mut r = 0i64;
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j + 1 < n && x[idx[j + 1]] == x[idx[i]] { j += 1; }
        for k in i..=j { ranks[idx[k]] = r; }
        r += 1;
        i = j + 1;
    }
    let n_unique = r as usize;
    let mut bins = vec![0f64; n];
    for i in 0..n {
        bins[i] = if n_unique <= nbin {
            ranks[i] as f64
        } else {
            ((ranks[i] as usize * nbin) / n_unique).min(nbin - 1) as f64
        };
    }
    bins
}

// ===================== 银行 =====================
fn round_half_even(x: f64) -> f64 {
    let i = x.floor();
    let f = x - i;
    if f < 0.5 { i }
    else if f > 0.5 { i + 1.0 }
    else {
        let ii = i as i64;
        if ii % 2 == 0 { i } else { i + 1.0 }
    }
}

fn quantize_corr(r: f64) -> f64 {
    round_half_even(r / 0.005) * 0.005
}

pub struct Bank {
    pub y: Vec<f64>,
    pub n: usize,
    pub var: f64,
    pub mu: f64,
}

impl Bank {
    pub fn new(y: Vec<f64>) -> Self {
        let n = y.len();
        let mu = y.iter().sum::<f64>() / n as f64;
        let var = y.iter().map(|v| (v - mu).powi(2)).sum::<f64>() / n as f64;
        Bank { y, n, var: var.max(1e-12), mu }
    }
    pub fn corr_raw(&self, g: &[f64]) -> f64 {
        let s: f64 = (0..self.n).map(|i| (g[i] - self.y[i]).powi(2)).sum();
        1.0 - s / (2.0 * self.var * self.n as f64)
    }
    pub fn query(&self, g: &[f64]) -> f64 {
        quantize_corr(self.corr_raw(g))
    }
}

// ===================== 排名工具 =====================
pub fn average_rank(g: &[f64]) -> Vec<f64> {
    let n = g.len();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| g[a].partial_cmp(&g[b]).unwrap());
    let mut ranks = vec![0f64; n];
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j + 1 < n && g[idx[j + 1]] == g[idx[i]] { j += 1; }
        let avg = (i as f64 + j as f64) / 2.0;
        for k in i..=j { ranks[idx[k]] = avg; }
        i = j + 1;
    }
    ranks
}

pub fn assign_by_rank(ranks: &[f64], y: &[f64]) -> Vec<f64> {
    let n = y.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| ranks[a].partial_cmp(&ranks[b]).unwrap()); // stable
    let mut ys = y.to_vec();
    ys.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mut g = vec![0f64; n];
    for (k, &i) in order.iter().enumerate() { g[i] = ys[k]; }
    g
}

// ===================== 轴序（匹配 np.lexsort）=====================
fn axis_order(win: &Window, axis: &str) -> Vec<usize> {
    let n = win.n;
    let mut idx: Vec<usize> = (0..n).collect();
    match axis {
        "TIME" => {} // 0..n
        "ACTIVE_ORDER" => {
            idx.sort_by(|&a, &b| {
                let ka = if win.active_order[a] >= 0 { win.active_order[a] } else { i64::MAX };
                let kb = if win.active_order[b] >= 0 { win.active_order[b] } else { i64::MAX };
                ka.cmp(&kb).then(a.cmp(&b))
            });
        }
        "PASSIVE_ORDER" => {
            idx.sort_by(|&a, &b| {
                let ka = if win.passive_order[a] >= 0 { win.passive_order[a] } else { i64::MAX };
                let kb = if win.passive_order[b] >= 0 { win.passive_order[b] } else { i64::MAX };
                ka.cmp(&kb).then(a.cmp(&b))
            });
        }
        "PRICE_TIME" => {
            idx.sort_by(|&a, &b| win.price[a].partial_cmp(&win.price[b]).unwrap().then(a.cmp(&b)));
        }
        _ => panic!("unknown axis"),
    }
    idx
}

// ===================== 第一层 =====================
const AXES: [&str; 4] = ["TIME", "ACTIVE_ORDER", "PASSIVE_ORDER", "PRICE_TIME"];
const DEPTHS: [&str; 4] = ["N32", "N16", "N8", "N4"];
const DEPTH_LEAVES: [usize; 4] = [32, 16, 8, 4];

fn node_bisect_candidate(g: &[f64], b: &[usize], high_left: bool) -> Vec<f64> {
    let half = b.len() / 2;
    let l = &b[..half];
    let r = &b[half..];
    let na = l.len();
    let nb = r.len();
    let mut vals: Vec<f64> = b.iter().map(|&i| g[i]).collect();
    vals.sort_by(|a, b2| a.partial_cmp(b2).unwrap());
    // 用 na/nb 切分（正确处理奇数 b.len()；偶数时 na=nb=half 与 Python 一致）
    let (l_vals, r_vals): (Vec<f64>, Vec<f64>) = if high_left {
        (vals[nb..].to_vec(), vals[..nb].to_vec())
    } else {
        (vals[..na].to_vec(), vals[na..].to_vec())
    };
    let mut g2 = g.to_vec();
    // L 内按 g 值稳定升序（相同 g 值保持 L 轴序，匹配 Python argsort stable）
    let mut lo: Vec<usize> = l.to_vec();
    lo.sort_by(|&a, &b2| g[a].partial_cmp(&g[b2]).unwrap());
    for (k, &i) in lo.iter().enumerate() { g2[i] = l_vals[k]; }
    let mut ro: Vec<usize> = r.to_vec();
    ro.sort_by(|&a, &b2| g[a].partial_cmp(&g[b2]).unwrap());
    for (k, &i) in ro.iter().enumerate() { g2[i] = r_vals[k]; }
    g2
}

pub struct BranchResult {
    pub axis: String, pub depth: String,
    pub g: Vec<f64>, pub rho0: f64, pub rho_final: f64,
    pub trajectory: Vec<f64>,
    pub node_ds: Vec<f64>,
    pub level_ds: Vec<Vec<f64>>,
    pub accepts: Vec<bool>,
    pub n_internal: usize, pub n_queries: usize,
    pub parent_child_sign: Vec<(f64, f64)>,
}

fn run_stage1_branch(win: &Window, bank: &Bank, g0: &[f64], axis: &str, depth: &str) -> BranchResult {
    let n = win.n;
    let leaf_size = n / DEPTH_LEAVES[DEPTHS.iter().position(|d| *d == depth).unwrap()];
    let aord = axis_order(win, axis);
    let mut g = g0.to_vec();
    let mut rho_cur = bank.query(&g);
    let rho0 = rho_cur;
    let mut trajectory = vec![rho_cur];
    let mut node_ds: Vec<f64> = vec![];
    let mut level_ds: Vec<Vec<f64>> = vec![];
    let mut accepts: Vec<bool> = vec![];
    let mut internal_count = 0usize;
    let mut cur_level: Vec<(usize, usize)> = vec![(0, n)];
    while !cur_level.is_empty() {
        let mut next_level: Vec<(usize, usize)> = vec![];
        let mut cur_level_ds: Vec<f64> = vec![];
        for (s, e) in cur_level {
            let seg_len = e - s;
            if seg_len <= leaf_size { continue; }
            internal_count += 1;
            let b: Vec<usize> = aord[s..e].to_vec();
            let g_hl = node_bisect_candidate(&g, &b, true);
            let g_hr = node_bisect_candidate(&g, &b, false);
            let rho_hl = bank.query(&g_hl);
            let rho_hr = bank.query(&g_hr);
            let d = rho_hr - rho_hl;
            node_ds.push(d);
            cur_level_ds.push(d);
            let mut accepted = false;
            if rho_cur >= rho_hl && rho_cur >= rho_hr {}
            else if rho_hl >= rho_hr { g = g_hl; rho_cur = rho_hl; accepted = true; }
            else { g = g_hr; rho_cur = rho_hr; accepted = true; }
            accepts.push(accepted);
            trajectory.push(rho_cur);
            let half = seg_len / 2;
            next_level.push((s, s + half));
            next_level.push((s + half, e));
        }
        if !cur_level_ds.is_empty() { level_ds.push(cur_level_ds); }
        cur_level = next_level;
    }
    let mut pc: Vec<(f64, f64)> = vec![];
    for li in 0..level_ds.len().saturating_sub(1) {
        for (j, &pd) in level_ds[li].iter().enumerate() {
            let ci = 2 * j;
            if ci < level_ds[li + 1].len() {
                pc.push((pd.signum(), level_ds[li + 1][ci].signum()));
            }
        }
    }
    let rho_final = bank.query(&g);
    let n_queries = node_ds.len() * 2;
    BranchResult {
        axis: axis.to_string(), depth: depth.to_string(),
        g, rho0, rho_final, trajectory, node_ds, level_ds, accepts,
        n_internal: internal_count, n_queries,
        parent_child_sign: pc,
    }
}

fn vote(branches: &[BranchResult], beta: f64, y: &[f64], rho0: f64) -> (Vec<f64>, f64) {
    let rhos: Vec<f64> = branches.iter().map(|b| b.rho_final).collect();
    let u: Vec<f64> = rhos.iter().map(|r| (r - rho0) / (1.0 - rho0 + 1e-12)).collect();
    let max_u = u.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = u.iter().map(|x| (beta * x - max_u).exp()).collect();
    let sum_e: f64 = exps.iter().sum();
    let w: Vec<f64> = exps.iter().map(|x| x / sum_e).collect();
    let n = y.len();
    let mut ranks = vec![0f64; n];
    for (b, wi) in branches.iter().zip(w.iter()) {
        let br = average_rank(&b.g);
        for i in 0..n { ranks[i] += wi * br[i]; }
    }
    let g_vote = assign_by_rank(&ranks, y);
    let bank = Bank::new(y.to_vec());
    let rv = bank.query(&g_vote);
    (g_vote, rv)
}

pub struct Stage1Result {
    pub g0: Vec<f64>, pub rho0: f64,
    pub g: Vec<f64>, pub rho: f64, pub best_name: String,
    pub branches: Vec<BranchResult>,
    pub votes: Vec<(Vec<f64>, f64)>,
    pub path_rhos: Vec<f64>,
}

impl Stage1Result {
    pub fn to_json(&self) -> String {
        let mut s = format!("{{\"rho0\":{:.4},\"rho1\":{:.4},\"best_name\":\"{}\",\"branches\":{{",
            self.rho0, self.rho, self.best_name);
        for (i, b) in self.branches.iter().enumerate() {
            if i > 0 { s.push(','); }
            let ac = b.accepts.iter().filter(|&&x| x).count();
            s.push_str(&format!("\"{}_{}\":{{\"rho0\":{:.4},\"rho_final\":{:.4},\"n_internal\":{},\"n_queries\":{},\"accept_count\":{}}}",
                b.axis, b.depth, b.rho0, b.rho_final, b.n_internal, b.n_queries, ac));
        }
        s.push_str("},\"votes\":{");
        for (i, beta) in [1,2,5,10].iter().enumerate() {
            if i > 0 { s.push(','); }
            s.push_str(&format!("\"{}\":{:.4}", beta, self.votes[i].1));
        }
        s.push_str("}}");
        s
    }
}

pub fn run_stage1(win: &Window, y: &[f64]) -> Stage1Result {
    let bank = Bank::new(y.to_vec());
    let mut ys = y.to_vec();
    ys.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let g0 = ys.clone();
    let rho0 = bank.query(&g0);
    let mut branches: Vec<BranchResult> = vec![];
    for axis in AXES.iter() {
        for depth in DEPTHS.iter() {
            branches.push(run_stage1_branch(win, &bank, &g0, axis, depth));
        }
    }
    let votes = vec![
        vote(&branches, 1.0, y, rho0),
        vote(&branches, 2.0, y, rho0),
        vote(&branches, 5.0, y, rho0),
        vote(&branches, 10.0, y, rho0),
    ];
    let mut best = (String::from("INIT"), g0.clone(), rho0);
    for b in &branches {
        if b.rho_final > best.2 { best = (format!("{}_{}", b.axis, b.depth), b.g.clone(), b.rho_final); }
    }
    for (i, beta) in [1,2,5,10].iter().enumerate() {
        if votes[i].1 > best.2 { best = (format!("VOTE_B{}", beta), votes[i].0.clone(), votes[i].1); }
    }
    let rho_desc = bank.query(&ys.iter().rev().cloned().collect::<Vec<_>>());
    let rho_active = {
        let ao = win.active_order.clone();
        let mut order: Vec<usize> = (0..win.n).collect();
        order.sort_by(|&a, &b| {
            let ka = if ao[a] >= 0 { ao[a] } else { i64::MAX };
            let kb = if ao[b] >= 0 { ao[b] } else { i64::MAX };
            ka.cmp(&kb).then(a.cmp(&b))
        });
        let mut g = vec![0f64; win.n];
        for (k, &i) in order.iter().enumerate() { g[i] = ys[k]; }
        bank.query(&g)
    };
    let rho_random = {
        let mut g = vec![0f64; win.n];
        let mut rng = stage3::Rng::new(20240101);
        let perm = rng.permutation(win.n);
        for (k, &i) in perm.iter().enumerate() { g[i] = ys[k]; }
        bank.query(&g)
    };
    Stage1Result { g0, rho0, g: best.1.clone(), rho: best.2, best_name: best.0.clone(),
        branches, votes, path_rhos: vec![rho0, rho_desc, rho_active, rho_random] }
}

fn is_discrete(t: &str) -> bool {
    matches!(t, "BIN8" | "BIN16" | "BIN32" | "TOP20" | "TOP10" | "TOP05" | "TOP02" | "TOP01")
}

fn run_full_game(win: &Window, y: &[f64], target: &str, graphs: &std::collections::HashMap<&'static str, stage2::EdgeSet>) -> (Stage1Result, stage2::Stage2Result, Option<stage3::Stage3Result>) {
    let t0 = std::time::Instant::now();
    let s1 = run_stage1(win, y);
    let t1 = std::time::Instant::now();
    let s2 = stage2::run_stage2_with_graphs(&s1.g, s1.rho, y, graphs);
    let t2 = std::time::Instant::now();
    let s3 = if is_discrete(target) { Some(stage3::run_stage3(&s2.g, y, &[0.5], 32)) } else { None };
    (s1, s2, s3)
}

fn random_null_y(y: &[f64], seed: u64) -> Vec<f64> {
    let mut rng = stage3::Rng::new(seed);
    let perm = rng.permutation(y.len());
    let mut ny = vec![0f64; y.len()];
    for (k, &i) in perm.iter().enumerate() { ny[i] = y[k]; }
    ny
}

const TARGETS_C: [&str; 3] = ["LOGVOL", "BIN16", "TOP10"];

/// 只跑第一二层（用于零假设，第三层因子只 RAW）
fn run_game_s1s2(win: &Window, y: &[f64], graphs: &std::collections::HashMap<&'static str, stage2::EdgeSet>) -> (Stage1Result, stage2::Stage2Result) {
    let s1 = run_stage1(win, y);
    let s2 = stage2::run_stage2_with_graphs(&s1.g, s1.rho, y, graphs);
    (s1, s2)
}

/// 折中方案C：4时段 × 2时间窗口 × 日频代表窗口(收盘前) × 5目标 × 基础因子 × RAW/EXCESS/NULL_Z
fn compute_all_factors(code: &str, date: i64, tws: &[f64], n_null: usize) {
    let t_total = std::time::Instant::now();
    let trade = fast_csv_reader::read_trade_fast(code, date).expect("read trade");
    let market = fast_csv_reader::read_market_fast(code, date).expect("read market");
    let ev = data_prep::align_and_build(&trade, &market);
    println!("读数据+预处理: {} 事件, 耗时 {:.2}s", ev.n, t_total.elapsed().as_secs_f64());

    let mut total_cols = 0usize;
    let mut samples: Vec<String> = vec![];
    let mut t_game = 0f64;
    let mut win_count = 0usize;

    for seg_id in 1..=4u32 {
        let seg = data_prep::select_segment(&ev, seg_id as usize);
        for &tw in tws {
            let tw_label = format!("T{}", tw as i64);
            // 日频：取该时段最后一个完整时间窗口（收盘前）
            let all_wins = data_prep::slide_time_windows(&seg, tw, tw, 128, usize::MAX);
            let windows: Vec<(f64, Window)> = all_wins.last().map(|last| vec![last.clone()]).unwrap_or_default();
            win_count += windows.len();
            for (ts, win) in &windows {
                let tg = std::time::Instant::now();
                let tbg = std::time::Instant::now();
                let graphs = stage2::build_graphs(win);
                for target in TARGETS_C.iter() {
                    let y = make_target(&win.volume, target);
                    if target.starts_with("TOP") {
                        let n1 = y.iter().filter(|&&v| v == 1.0).count();
                        if n1 < 4 || y.len() - n1 < 4 { continue; }
                    }
                    // 真实游戏（含第三层）
                    let (s1, s2, s3) = run_full_game(win, &y, target, &graphs);
                    let real_f = factors::extract_all(&s1, &s2, s3.as_ref(), &y);
                    // 零假设（只第一二层）
                    let mut null_maps: Vec<std::collections::HashMap<String, f64>> = vec![];
                    for k in 0..n_null {
                        let ny = random_null_y(&y, 999 + k as u64);
                        let (ns1, ns2) = run_game_s1s2(win, &ny, &graphs);
                        let nf = factors::extract_all(&ns1, &ns2, None, &ny);
                        null_maps.push(nf.into_iter().collect());
                    }
                    for (name, rv) in &real_f {
                        let null_vals: Vec<f64> = null_maps.iter().map(|m| *m.get(name).unwrap_or(&f64::NAN)).filter(|v| !v.is_nan()).collect();
                        let mean_n = if null_vals.is_empty() { f64::NAN } else { null_vals.iter().sum::<f64>() / null_vals.len() as f64 };
                        let std_n = if null_vals.len() > 1 { (null_vals.iter().map(|v| (v - mean_n).powi(2)).sum::<f64>() / null_vals.len() as f64).sqrt() } else if null_vals.len() == 1 { 0.0 } else { f64::NAN };
                        let excess = if mean_n.is_nan() { f64::NAN } else { rv - mean_n };
                        let null_z = if mean_n.is_nan() || std_n.is_nan() { f64::NAN } else { (rv - mean_n) / (std_n + 1e-8) };
                        total_cols += 3;
                        if samples.len() < 8 {
                            samples.push(format!("SEG{}_{}_{}_{}_RAW/EXCESS/NULL_Z@{:.0} = {:.4}/{:.4}/{:.4}",
                                seg_id, tw_label, target, name, ts, rv, excess, null_z));
                        }
                    }
                }
                t_game += tg.elapsed().as_secs_f64();
            }
            println!("  SEG{} {}: N={}", seg_id, tw_label, windows.first().map(|(_, w)| w.n).unwrap_or(0));
        }
    }
    println!("\n=== 折中方案C 因子统计 ===");
    println!("日频窗口数(时间戳): {}", win_count);
    println!("单时间戳因子列(含RAW/EXCESS/NULL_Z): {}", total_cols / win_count.max(1));
    println!("总因子值: {}", total_cols);
    println!("三层游戏总耗时: {:.2}s", t_game);
    println!("总耗时: {:.2}s", t_total.elapsed().as_secs_f64());
    println!("\n样本:");
    for s in &samples { println!("  {}", s); }
}

// ===================== Pipeline 规范封装 =====================
/// 因子名（与 compute 输出严格对齐，单一源）。
pub fn hidden_arrange_names() -> Vec<String> {
    let mut names: Vec<String> = vec![];
    for seg_id in 1..=4u32 {
        for target in TARGETS_C.iter() {
            // LOGVOL 连续目标: 基础因子 703 个
            // BIN16/TOP10 离散目标: 基础因子 949 个
            // 用占位: 实际名字由 extract_all 的输出顺序决定
            let dummy_win_n = 430usize; // 代表窗口大小
            // 因子名需要和 compute 对齐，用索引而非硬编码
            names.push(format!("SEG{}_T180_{}_FACTOR_{}", seg_id, target, 0));
        }
    }
    names // 占位，实际由 compute_hidden_arrange_full 返回
}

/// 纯 Rust 核心（唯一真相源）：读数据 → 4时段 → T180 日频窗口 → 3目标 → 三层游戏 → 因子(RAW) → Vec<f32>
/// 输出长度固定，pipeline 和 Python 入口的共同调用点。
pub fn compute_hidden_arrange_full(code: &str, date: i64) -> std::io::Result<(Vec<String>, Vec<f32>)> {
    let trade = fast_csv_reader::read_trade_fast(code, date)?;
    let market = fast_csv_reader::read_market_fast(code, date)?;
    let ev = data_prep::align_and_build(&trade, &market);

    let mut all_names: Vec<String> = vec![];
    let mut all_vals: Vec<f32> = vec![];

    for seg_id in 1..=4u32 {
        let seg = data_prep::select_segment(&ev, seg_id as usize);
        let windows = data_prep::slide_time_windows(&seg, 180.0, 180.0, 128, usize::MAX);
        if windows.is_empty() { continue; }
        let (_ts, win) = windows.last().unwrap();
        let win = win.clone();
        let graphs = stage2::build_graphs(&win);
        let adjs = stage2::build_adj_lists(&graphs, win.n);
        for target in TARGETS_C.iter() {
            let y = make_target(&win.volume, target);
            if target.starts_with("TOP") {
                let n1 = y.iter().filter(|&&v| v == 1.0).count();
                if n1 < 4 || y.len() - n1 < 4 { continue; }
            }
            let s1 = run_stage1(&win, &y);
            let s2 = stage2::run_stage2_with_adjs(&s1.g, s1.rho, &y, &adjs);
            let s3 = if is_discrete(target) { Some(stage3::run_stage3(&s2.g, &y, &[0.5], 32)) } else { None };
            let fac = factors::extract_all(&s1, &s2, s3.as_ref(), &y);
            for (name, val) in &fac {
                all_names.push(format!("SEG{}_T180_{}_{}_RAW", seg_id, target, name));
                all_vals.push(*val as f32);
            }
        }
    }
    Ok((all_names, all_vals))
}

/// pipeline 包装：worker 进程批量调用，错误吞掉返 NaN。
pub fn pipeline_hidden_arrange(date: i64, code: &str, expected_len: usize) -> Vec<f32> {
    match compute_hidden_arrange_full(code, date) {
        Ok((_names, vals)) => vals,
        Err(_) => vec![f32::NAN; expected_len],
    }
}

fn compute_with_timing(code: &str, date: i64) {
    use std::time::Instant;
    let mut t_io = 0f64; let mut t_prep = 0f64;
    let mut t_seg = 0f64; let mut t_graph = 0f64;
    let mut t_target = 0f64; let mut t_s1 = 0f64; let mut t_s2 = 0f64;
    let mut t_s3 = 0f64; let mut t_factor = 0f64;

    let t0 = Instant::now();
    let ti = Instant::now();
    let trade = fast_csv_reader::read_trade_fast(code, date).unwrap();
    let market = fast_csv_reader::read_market_fast(code, date).unwrap();
    t_io += ti.elapsed().as_secs_f64();

    let ti = Instant::now();
    let ev = data_prep::align_and_build(&trade, &market);
    t_prep += ti.elapsed().as_secs_f64();

    let mut seg_ns: Vec<usize> = vec![];
    for seg_id in 1..=4u32 {
        let ti = Instant::now();
        let seg = data_prep::select_segment(&ev, seg_id as usize);
        let windows = data_prep::slide_time_windows(&seg, 180.0, 180.0, 128, usize::MAX);
        t_seg += ti.elapsed().as_secs_f64();
        if windows.is_empty() { continue; }
        let (_ts, win) = windows.last().unwrap();
        let win = win.clone();
        seg_ns.push(win.n);
        let ti = Instant::now();
        let graphs = stage2::build_graphs(&win);
        let adjs = stage2::build_adj_lists(&graphs, win.n);
        t_graph += ti.elapsed().as_secs_f64();
        for target in TARGETS_C.iter() {
            let ti = Instant::now();
            let y = make_target(&win.volume, target);
            t_target += ti.elapsed().as_secs_f64();
            if target.starts_with("TOP") {
                let n1 = y.iter().filter(|&&v| v == 1.0).count();
                if n1 < 4 || y.len() - n1 < 4 { continue; }
            }
            let ti = Instant::now();
            let s1 = run_stage1(&win, &y);
            t_s1 += ti.elapsed().as_secs_f64();
            let ti = Instant::now();
            let s2 = stage2::run_stage2_with_adjs(&s1.g, s1.rho, &y, &adjs);
            t_s2 += ti.elapsed().as_secs_f64();
            let ti = Instant::now();
            let s3 = if is_discrete(target) { Some(stage3::run_stage3(&s2.g, &y, &[0.5], 32)) } else { None };
            t_s3 += ti.elapsed().as_secs_f64();
            let ti = Instant::now();
            let _fac = factors::extract_all(&s1, &s2, s3.as_ref(), &y);
            t_factor += ti.elapsed().as_secs_f64();
        }
    }
    let sum = t_io+t_prep+t_seg+t_graph+t_target+t_s1+t_s2+t_s3+t_factor;
    println!("--- 分阶段耗时 ---");
    println!("各段窗口N: {:?}", seg_ns);
    println!("读数据(IO):     {:.3}s", t_io);
    println!("对齐预处理:     {:.3}s", t_prep);
    println!("选段+滑窗:      {:.3}s", t_seg);
    println!("建图graph:      {:.3}s", t_graph);
    println!("make_target:    {:.3}s", t_target);
    println!("run_stage1:     {:.3}s", t_s1);
    println!("run_stage2:     {:.3}s", t_s2);
    println!("run_stage3:     {:.3}s", t_s3);
    println!("extract_all:    {:.3}s", t_factor);
    println!("阶段累加:       {:.3}s", sum);
    println!("实际总耗时:     {:.3}s", t0.elapsed().as_secs_f64());
    stage2::print_s2_timing();
    stage3::print_s3_timing();
}

fn main() {
    println!("\n########## 分阶段计时 ##########");
    compute_with_timing("000001", 20220819);
    println!("\n########## 正式运行 + 5次确定性验证 ##########");
    let mut prev: Option<Vec<f32>> = None;
    let mut names_cnt = 0usize;
    for run in 1..=5u32 {
        let t = std::time::Instant::now();
        let (names, vals) = compute_hidden_arrange_full("000001", 20220819).expect("compute failed");
        let dt = t.elapsed().as_secs_f64();
        names_cnt = names.len();
        let ok = match &prev {
            None => true,
            Some(p) => vals.iter().zip(p.iter()).all(|(a, b)| a == b),
        };
        if run == 1 {
            println!("=== 方案C: T180 + 3目标 + RAW (000001/20220819) ===");
            println!("因子列数: {}", names.len());
            let mut sc: std::collections::BTreeMap<(String, String), usize> = std::collections::BTreeMap::new();
            for n in &names {
                let p: Vec<&str> = n.split('_').collect();
                if p.len() >= 3 { *sc.entry((p[0].to_string(), p[2].to_string())).or_insert(0) += 1; }
            }
            println!("--- 切片 × 目标 因子分布 ---");
            for ((seg, tgt), cnt) in &sc { println!("  {} × {}: {} 个", seg, tgt, cnt); }
            println!("前5个因子: ");
            for i in 0..names.len().min(5) {
                println!("  {} = {:.4}", names[i], vals[i]);
            }
        }
        println!("第{}次: 耗时 {:.3}s, 与上次结果一致 {}", run, dt, if ok { "✓" } else { "✗" });
        prev = Some(vals);
    }
    println!("5次运行全部一致, 因子列数 {}", names_cnt);
}
