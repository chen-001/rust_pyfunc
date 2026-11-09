//! RNA 折叠式买卖配对：单日全市场横截面探索（sandbox）。
//! 四层结构：字母化（letters.rs）→ 配对折叠（fold.rs）→ 结构特征（本文件 structure_features）→ 横截面合成（本文件 synthesize）。
//! 运行：./target/release/align_fold <date> <输出目录>
//! 环境变量：FOLD_THREADS（默认且最大 50）、FULL_BAR / PM_BAR / TAIL_BAR（秒，默认 30/30/15）、
//! TIEBREAK（latest / earliest）、RUST_PYFUNC_LEVEL2_ROOT、RUST_PYFUNC_LEVEL2_PATH、RUST_PYFUNC_BASIC_INFO_DIR。
mod fast_csv_reader;
mod fold;
mod letters;

use fast_csv_reader::{read_market_fast, read_trade_fast};
use fold::{Table, NEG};
use letters::{build_sequences, windows, Seq, Win, ALPHA_NAMES, N_ALPHA};
use rayon::prelude::*;
use std::collections::BTreeSet;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::time::Instant;

const FOLD_NAMES: [&str; 10] = ["cnt", "dec", "vol", "rna", "profit", "loss", "bf", "sf", "short", "h10"];
const N_FOLD: usize = FOLD_NAMES.len();
const FEAT_NAMES: [&str; 13] = [
    "score", "pair_ratio", "unp_net", "unp_center", "depth_max", "depth_mean", "span_mean", "branch", "hairpin",
    "stem", "bf_share", "pair_center", "gain_bps",
];
const N_F: usize = FEAT_NAMES.len();
const SHUF_FOLDS: [usize; 3] = [0, 1, 2]; // cnt / dec / vol 做打乱对照
const SHUF_FEATS: [usize; 3] = [0, 5, 6]; // score / depth_mean / span_mean
const N_SHUF_REP: usize = 3;
const NAIVE_NAMES: [&str; 9] = [
    "active_share", "buy_share", "bound", "letter_imb", "vol_imb", "alternation", "walk_range", "walk_end", "greedy_pair",
];
const N_NAIVE: usize = NAIVE_NAMES.len();
const PER_UNIT: usize = N_FOLD * N_F + SHUF_FOLDS.len() * SHUF_FEATS.len() + N_NAIVE;
const N_WIN: usize = 3;
const N_RAW: usize = N_WIN * N_ALPHA * PER_UNIT;
const H_DEFAULT: usize = 3;

#[inline]
fn base(win: usize, alpha: usize) -> usize {
    (win * N_ALPHA + alpha) * PER_UNIT
}
#[inline]
fn fold_col(win: usize, alpha: usize, fold: usize, feat: usize) -> usize {
    base(win, alpha) + fold * N_F + feat
}
#[inline]
fn shuf_col(win: usize, alpha: usize, sf: usize, feat: usize) -> usize {
    base(win, alpha) + N_FOLD * N_F + sf * SHUF_FEATS.len() + feat
}
#[inline]
fn naive_col(win: usize, alpha: usize, x: usize) -> usize {
    base(win, alpha) + N_FOLD * N_F + SHUF_FOLDS.len() * SHUF_FEATS.len() + x
}

fn raw_names() -> Vec<String> {
    let mut v = vec![String::new(); N_RAW];
    for (w, wn) in ["full", "pm", "tail"].iter().enumerate() {
        for (a, an) in ALPHA_NAMES.iter().enumerate() {
            for (f, fnm) in FOLD_NAMES.iter().enumerate() {
                for (x, xn) in FEAT_NAMES.iter().enumerate() {
                    v[fold_col(w, a, f, x)] = format!("{an}_{wn}_{fnm}_{xn}");
                }
            }
            for (i, &f) in SHUF_FOLDS.iter().enumerate() {
                for (j, &x) in SHUF_FEATS.iter().enumerate() {
                    v[shuf_col(w, a, i, j)] = format!("excess_{an}_{wn}_{}_{}", FOLD_NAMES[f], FEAT_NAMES[x]);
                }
            }
            for (x, xn) in NAIVE_NAMES.iter().enumerate() {
                v[naive_col(w, a, x)] = format!("naive_{an}_{wn}_{xn}");
            }
        }
    }
    v
}

// ---------- 配对折叠 + 结构特征 ----------

struct Ctx {
    side: Vec<i8>,
    strength: Vec<u8>,
    price: Vec<f32>,
    vint: Vec<i32>,
    active: Vec<bool>,
    decay: Vec<i32>,
    n_buy: i32,
    n_sell: i32,
    sum_vb: i64,
    sum_vs: i64,
}

fn make_ctx(s: &Seq) -> Ctx {
    let n = s.side.len();
    let mut sumv = 0f64;
    let mut cnt = 0f64;
    for i in 0..n {
        if s.side[i] != 0 { sumv += s.vol[i] as f64; cnt += 1.0; }
    }
    let mean = if cnt > 0.0 { sumv / cnt } else { 1.0 };
    let vint: Vec<i32> = s.vol.iter().map(|&v| ((v as f64 / mean * 100.0) as i32).clamp(1, 3000)).collect();
    let tau = (n as f64 / 8.0).max(1.0);
    let decay: Vec<i32> = (0..=n).map(|d| ((1000.0 * (-(d as f64) / tau).exp()).round() as i32).max(1)).collect();
    let (mut n_buy, mut n_sell, mut sum_vb, mut sum_vs) = (0i32, 0i32, 0i64, 0i64);
    for i in 0..n {
        if s.side[i] > 0 { n_buy += 1; sum_vb += vint[i] as i64; }
        else if s.side[i] < 0 { n_sell += 1; sum_vs += vint[i] as i64; }
    }
    Ctx {
        active: s.side.iter().map(|&x| x != 0).collect(),
        side: s.side.clone(),
        strength: s.strength.clone(),
        price: s.price.clone(),
        vint,
        decay,
        n_buy,
        n_sell,
        sum_vb,
        sum_vs,
    }
}

/// 按固定规则计算配对结构特征。
fn structure_features(c: &Ctx, partner: &[i32], score_norm: f32, out: &mut [f32]) {
    let n = partner.len();
    let (mut pairs, mut depth, mut maxd, mut sumd, mut top, mut hairpin, mut cont) = (0u32, 0u32, 0u32, 0u64, 0u32, 0u32, 0u32);
    let (mut span, mut bf, mut center, mut gain) = (0f64, 0u32, 0f64, 0f64);
    let (mut unp, mut unp_net, mut unp_pos) = (0u32, 0f64, 0f64);
    let mut prev_paired: i32 = -1;
    for p in 0..n {
        let q = partner[p];
        if q > p as i32 {
            depth += 1;
            pairs += 1;
            maxd = maxd.max(depth);
            sumd += depth as u64;
            if depth == 1 { top += 1; }
            let qu = q as usize;
            span += (qu - p) as f64;
            center += (p + qu) as f64 / 2.0;
            let (bi, si) = if c.side[p] > 0 { bf += 1; (p, qu) } else { (qu, p) };
            gain += ((c.price[si] / c.price[bi]) as f64 - 1.0) * 1e4;
            if p + 1 < qu && partner[p + 1] == q - 1 { cont += 1; }
            prev_paired = p as i32;
        } else if q >= 0 {
            if prev_paired == q { hairpin += 1; }
            prev_paired = p as i32;
            depth -= 1;
        } else if c.side[p] != 0 {
            unp += 1;
            unp_net += c.side[p] as f64;
            unp_pos += p as f64;
        }
    }
    let nf = n as f64;
    let active = (c.n_buy + c.n_sell) as f64;
    let bound = c.n_buy.min(c.n_sell) as f32;
    out[0] = score_norm;
    out[1] = pairs as f32 / bound;
    out[2] = (unp_net / active) as f32;
    out[3] = if unp > 0 { (unp_pos / unp as f64 / nf) as f32 } else { f32::NAN };
    if pairs > 0 {
        let pf = pairs as f64;
        out[4] = (maxd as f64 / pf) as f32;
        out[5] = (sumd as f64 / pf) as f32;
        out[6] = (span / pf / nf) as f32;
        out[7] = (top as f64 / pf) as f32;
        out[8] = (hairpin as f64 / pf) as f32;
        out[9] = (pf / (pf - cont as f64).max(1.0)) as f32;
        out[10] = (bf as f64 / pf) as f32;
        out[11] = (center / pf / nf) as f32;
        out[12] = (gain / pf) as f32;
    } else {
        out[4..13].iter_mut().for_each(|x| *x = f32::NAN);
    }
}

/// 跑一种折叠并按需求特征；返回归一化得分。
fn run_fold(kind: usize, c: &Ctx, t: &mut Table, partner: &mut [i32], latest: bool, out: Option<&mut [f32]>) -> f32 {
    macro_rules! go {
        ($h:expr, $ub:expr, $w:expr) => {{
            let w = $w;
            let n = c.side.len();
            let score = t.solve(n, $h, &c.active, &w);
            let ub: f64 = $ub;
            let sn = if ub > 0.0 { (score as f64 / ub) as f32 } else { f32::NAN };
            if let Some(o) = out {
                t.traceback(n, $h, &w, latest, partner);
                structure_features(c, &partner[..n], sn, o);
            }
            sn
        }};
    }
    let n = c.side.len();
    let side = &c.side[..];
    let price = &c.price[..];
    let unit = c.n_buy.min(c.n_sell) as f64;
    match kind {
        0 => go!(H_DEFAULT, unit, |k: usize, j: usize| if side[k] * side[j] == -1 { 1 } else { NEG }),
        1 => {
            let d = &c.decay[..];
            go!(H_DEFAULT, unit * 1000.0, |k: usize, j: usize| if side[k] * side[j] == -1 { d[j - k] } else { NEG })
        }
        2 => {
            let v = &c.vint[..];
            let ub = c.sum_vb.min(c.sum_vs) as f64;
            go!(H_DEFAULT, ub, |k: usize, j: usize| if side[k] * side[j] == -1 { v[k].min(v[j]) } else { NEG })
        }
        3 => {
            let st = &c.strength[..];
            go!(H_DEFAULT, unit * 3.0, |k: usize, j: usize| if side[k] * side[j] == -1 {
                st[k] as i32 + st[j] as i32 - 1
            } else { NEG })
        }
        4 | 5 => {
            let want_profit = kind == 4;
            go!(H_DEFAULT, unit, |k: usize, j: usize| {
                if side[k] * side[j] != -1 { return NEG; }
                let (b, s) = if side[k] > 0 { (price[k], price[j]) } else { (price[j], price[k]) };
                if (want_profit && s > b) || (!want_profit && s < b) { 1 } else { NEG }
            })
        }
        6 => go!(H_DEFAULT, unit, |k: usize, j: usize| if side[k] > 0 && side[j] < 0 { 1 } else { NEG }),
        7 => go!(H_DEFAULT, unit, |k: usize, j: usize| if side[k] < 0 && side[j] > 0 { 1 } else { NEG }),
        8 => {
            let l = (n / 8).max(H_DEFAULT + 1);
            go!(H_DEFAULT, unit, |k: usize, j: usize| if side[k] * side[j] == -1 && j - k <= l { 1 } else { NEG })
        }
        _ => go!(10usize, unit, |k: usize, j: usize| if side[k] * side[j] == -1 { 1 } else { NEG }),
    }
}

fn naive_features(c: &Ctx, s: &Seq, h: usize, out: &mut [f32]) {
    let n = c.side.len();
    let active = (c.n_buy + c.n_sell) as f64;
    out[0] = (active / n as f64) as f32;
    if active == 0.0 { return; }
    out[1] = (c.n_buy as f64 / active) as f32;
    out[2] = (c.n_buy.min(c.n_sell) as f64 / active) as f32;
    out[3] = ((c.n_buy - c.n_sell) as f64 / active) as f32;
    let (mut vb, mut vs) = (0f64, 0f64);
    for i in 0..n {
        if c.side[i] > 0 { vb += s.vol[i] as f64; } else if c.side[i] < 0 { vs += s.vol[i] as f64; }
    }
    out[4] = if vb + vs > 0.0 { ((vb - vs) / (vb + vs)) as f32 } else { f32::NAN };
    // 连续一步的简单统计：翻转率、累计走势的振幅和终点
    let (mut prev, mut flips, mut adj, mut walk, mut lo, mut hi) = (0i8, 0u32, 0u32, 0i32, 0i32, 0i32);
    let mut stack: Vec<usize> = Vec::new();
    let mut greedy = 0i32;
    for i in 0..n {
        let x = c.side[i];
        if x == 0 { continue; }
        if prev != 0 {
            adj += 1;
            if prev != x { flips += 1; }
        }
        prev = x;
        walk += x as i32;
        lo = lo.min(walk);
        hi = hi.max(walk);
        // 贪心括号匹配：只看栈顶能否闭合，作为动态规划的朴素替代物
        match stack.last() {
            Some(&t) if c.side[t] != x && i - t - 1 >= h => {
                stack.pop();
                greedy += 1;
            }
            _ => stack.push(i),
        }
    }
    out[5] = if adj > 0 { flips as f32 / adj as f32 } else { f32::NAN };
    out[6] = ((hi - lo) as f64 / active) as f32;
    out[7] = (walk as f64 / active) as f32;
    let b = c.n_buy.min(c.n_sell);
    out[8] = if b > 0 { greedy as f32 / b as f32 } else { f32::NAN };
}

fn permute(s: &Seq, seed: u64) -> Seq {
    let n = s.side.len();
    let mut idx: Vec<usize> = (0..n).collect();
    let mut x = seed | 1;
    for i in (1..n).rev() {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        idx.swap(i, (x % (i as u64 + 1)) as usize);
    }
    Seq {
        side: idx.iter().map(|&i| s.side[i]).collect(),
        strength: idx.iter().map(|&i| s.strength[i]).collect(),
        vol: idx.iter().map(|&i| s.vol[i]).collect(),
        price: idx.iter().map(|&i| s.price[i]).collect(),
    }
}

fn fnv(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

fn per_stock(code: &str, date: i64, wins: &[Win], latest: bool) -> Vec<f32> {
    let mut row = vec![f32::NAN; N_RAW];
    let tr = match read_trade_fast(code, date) { Ok(t) => t, Err(_) => return row };
    let mk = read_market_fast(code, date).unwrap_or_default();
    let mut t = Table::new();
    let mut partner = vec![-1i32; 1024];
    let mut buf = [f32::NAN; N_F];
    for (w, win) in wins.iter().enumerate() {
        let seqs = build_sequences(&tr, &mk, win);
        for (a, sq) in seqs.iter().enumerate() {
            let Some(s) = sq else { continue };
            let n = s.side.len();
            if partner.len() < n + 1 { partner.resize(n + 1, -1); }
            let c = make_ctx(s);
            let mut nv = [f32::NAN; N_NAIVE];
            naive_features(&c, s, H_DEFAULT, &mut nv);
            row[naive_col(w, a, 0)..naive_col(w, a, 0) + N_NAIVE].copy_from_slice(&nv);
            if n < 20 || c.n_buy < 3 || c.n_sell < 3 { continue; }
            for f in 0..N_FOLD {
                let col = fold_col(w, a, f, 0);
                run_fold(f, &c, &mut t, &mut partner[..n + 1], latest, Some(&mut row[col..col + N_F]));
            }
            // 打乱对照：同一批字母随机换位，保留每种字母的数量、量和价
            let code_seed = fnv(code.as_bytes()) ^ ((w as u64 + 1) << 40) ^ ((a as u64 + 1) << 32);
            let mut sums = [[0f64; 3]; 3];
            let mut cnts = [[0u32; 3]; 3];
            for rep in 0..N_SHUF_REP {
                let p = permute(s, code_seed.wrapping_add((rep as u64 + 1).wrapping_mul(0x9E3779B97F4A7C15)));
                let cp = make_ctx(&p);
                for (i, &f) in SHUF_FOLDS.iter().enumerate() {
                    run_fold(f, &cp, &mut t, &mut partner[..n + 1], latest, Some(&mut buf));
                    for (j, &x) in SHUF_FEATS.iter().enumerate() {
                        if buf[x].is_finite() { sums[i][j] += buf[x] as f64; cnts[i][j] += 1; }
                    }
                }
            }
            for (i, &f) in SHUF_FOLDS.iter().enumerate() {
                for (j, &x) in SHUF_FEATS.iter().enumerate() {
                    let v = row[fold_col(w, a, f, x)];
                    row[shuf_col(w, a, i, j)] = if cnts[i][j] > 0 && v.is_finite() {
                        v - (sums[i][j] / cnts[i][j] as f64) as f32
                    } else { f32::NAN };
                }
            }
        }
    }
    row
}

// ---------- 横截面合成层 ----------

/// 分位秩（并列取平均），空值保持空值；结果范围 0~1。
fn pct_rank(col: &[f32]) -> Vec<f32> {
    let mut idx: Vec<u32> = (0..col.len() as u32).filter(|&i| col[i as usize].is_finite()).collect();
    idx.sort_unstable_by(|&a, &b| col[a as usize].total_cmp(&col[b as usize]).then(a.cmp(&b)));
    let mut out = vec![f32::NAN; col.len()];
    let m = idx.len();
    if m < 2 { return out; }
    let mut i = 0;
    while i < m {
        let v = col[idx[i] as usize];
        let mut j = i + 1;
        while j < m && col[idx[j] as usize] == v { j += 1; }
        let r = ((i + j - 1) as f64 / 2.0 / (m - 1) as f64) as f32;
        for &x in &idx[i..j] { out[x as usize] = r; }
        i = j;
    }
    out
}

/// 对多个控制变量做横截面回归，返回残差。
fn ols_resid(y: &[f32], xs: &[Vec<f32>]) -> Vec<f32> {
    let p = xs.len() + 1;
    let n = y.len();
    let keep: Vec<usize> = (0..n).filter(|&i| y[i].is_finite() && xs.iter().all(|x| x[i].is_finite())).collect();
    let mut out = vec![f32::NAN; n];
    if keep.len() < 3 * p { return out; }
    let mut a = vec![0f64; p * (p + 1)];
    let mut row = vec![0f64; p];
    for &i in &keep {
        row[0] = 1.0;
        for (k, x) in xs.iter().enumerate() { row[k + 1] = x[i] as f64; }
        for r in 0..p {
            for c in 0..p { a[r * (p + 1) + c] += row[r] * row[c]; }
            a[r * (p + 1) + p] += row[r] * y[i] as f64;
        }
    }
    for r in 0..p { a[r * (p + 1) + r] += 1e-8; }
    // 带选主元的高斯消元
    for c in 0..p {
        let mut piv = c;
        for r in c + 1..p {
            if a[r * (p + 1) + c].abs() > a[piv * (p + 1) + c].abs() { piv = r; }
        }
        if piv != c {
            for x in 0..=p { a.swap(c * (p + 1) + x, piv * (p + 1) + x); }
        }
        let d = a[c * (p + 1) + c];
        if d.abs() < 1e-12 { return out; }
        for r in 0..p {
            if r == c { continue; }
            let f = a[r * (p + 1) + c] / d;
            for x in c..=p { a[r * (p + 1) + x] -= f * a[c * (p + 1) + x]; }
        }
    }
    let beta: Vec<f64> = (0..p).map(|r| a[r * (p + 1) + p] / a[r * (p + 1) + r]).collect();
    for &i in &keep {
        let mut pred = beta[0];
        for (k, x) in xs.iter().enumerate() { pred += beta[k + 1] * x[i] as f64; }
        out[i] = (y[i] as f64 - pred) as f32;
    }
    out
}

const CONTROLS: [usize; 4] = [2, 4, 5, 6]; // 平凡上界、量不平衡、翻转率、累计走势振幅

/// 合成：字母共识、残差、折叠间对比、时段对比。返回 (列名, 列)。
fn synthesize(raw: &[Vec<f32>], n_codes: usize) -> (Vec<String>, Vec<Vec<f32>>) {
    let col = |c: usize| -> Vec<f32> { raw.iter().map(|r| r[c]).collect() };
    // 需要先把每列字母版本转成分位秩，才能跨字母平均
    let targets: Vec<(usize, usize, usize, usize)> = (0..N_WIN)
        .flat_map(|w| (0..N_ALPHA).flat_map(move |a| (0..N_FOLD).flat_map(move |f| (0..N_F).map(move |x| (w, a, f, x)))))
        .collect();
    let ranks: Vec<Vec<f32>> = targets.par_iter().map(|&(w, a, f, x)| pct_rank(&col(fold_col(w, a, f, x)))).collect();
    let rk = |w: usize, a: usize, f: usize, x: usize| -> &Vec<f32> {
        &ranks[((w * N_ALPHA + a) * N_FOLD + f) * N_F + x]
    };
    let wn = ["full", "pm", "tail"];

    let mut names = Vec::new();
    let mut cols: Vec<Vec<f32>> = Vec::new();

    // 共识：同一时段、折叠、特征下，把 10 种字母的分位秩压成一个数，四种压法（至少 5 种字母有值）。
    // cons=均值，consmax=最大，consmed=中位数，consstd=字母之间的分歧度。
    let agg = |vals: &mut Vec<f32>| -> [f32; 4] {
        if vals.len() < 5 { return [f32::NAN; 4]; }
        vals.sort_unstable_by(|a, b| a.total_cmp(b));
        let m = vals.len();
        let mean = vals.iter().map(|&v| v as f64).sum::<f64>() / m as f64;
        let sd = (vals.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / m as f64).sqrt();
        let med = if m % 2 == 1 { vals[m / 2] as f64 } else { (vals[m / 2 - 1] as f64 + vals[m / 2] as f64) / 2.0 };
        [mean as f32, vals[m - 1], med as f32, sd as f32]
    };
    let triples: Vec<(usize, usize, usize)> = (0..N_WIN)
        .flat_map(|w| (0..N_FOLD).flat_map(move |f| (0..N_F).map(move |x| (w, f, x))))
        .collect();
    let cons4: Vec<[Vec<f32>; 4]> = triples.par_iter().map(|&(w, f, x)| {
        let mut out: [Vec<f32>; 4] = std::array::from_fn(|_| vec![f32::NAN; n_codes]);
        let mut vals = Vec::with_capacity(N_ALPHA);
        for i in 0..n_codes {
            vals.clear();
            for a in 0..N_ALPHA {
                let v = rk(w, a, f, x)[i];
                if v.is_finite() { vals.push(v); }
            }
            let r = agg(&mut vals);
            for q in 0..4 { out[q][i] = r[q]; }
        }
        out
    }).collect();
    let cons: Vec<Vec<f32>> = cons4.iter().map(|c| c[0].clone()).collect();

    // 朴素量做同样的四种压法，供“两种增量”按相同口径比较
    let naive_pairs: Vec<(usize, usize)> = (0..N_WIN).flat_map(|w| (0..N_NAIVE).map(move |x| (w, x))).collect();
    let naive_cons: Vec<[Vec<f32>; 4]> = naive_pairs.par_iter().map(|&(w, x)| {
        let rks: Vec<Vec<f32>> = (0..N_ALPHA).map(|a| pct_rank(&col(naive_col(w, a, x)))).collect();
        let mut out: [Vec<f32>; 4] = std::array::from_fn(|_| vec![f32::NAN; n_codes]);
        let mut vals = Vec::with_capacity(N_ALPHA);
        for i in 0..n_codes {
            vals.clear();
            for rk_a in &rks {
                if rk_a[i].is_finite() { vals.push(rk_a[i]); }
            }
            let r = agg(&mut vals);
            for q in 0..4 { out[q][i] = r[q]; }
        }
        out
    }).collect();

    // 残差：共识对 flow 字母的朴素量（上界、量不平衡、翻转率、累计振幅）的横截面回归残差
    let controls: Vec<Vec<Vec<f32>>> = (0..N_WIN)
        .map(|w| CONTROLS.iter().map(|&x| pct_rank(&col(naive_col(w, 0, x)))).collect())
        .collect();
    let resid: Vec<Vec<f32>> = triples.par_iter().enumerate().map(|(i, &(w, _, _))| ols_resid(&cons[i], &controls[w])).collect();

    let tags = ["cons", "consmax", "consmed", "consstd"];
    for (q, tag) in tags.iter().enumerate() {
        for (i, &(w, f, x)) in triples.iter().enumerate() {
            names.push(format!("{}_{}_{}_{}", tag, wn[w], FOLD_NAMES[f], FEAT_NAMES[x]));
            cols.push(cons4[i][q].clone());
        }
    }
    for (q, tag) in tags.iter().enumerate() {
        for (i, &(w, x)) in naive_pairs.iter().enumerate() {
            names.push(format!("{}_naive_{}_{}", tag, wn[w], NAIVE_NAMES[x]));
            cols.push(naive_cons[i][q].clone());
        }
    }
    for (i, &(w, f, x)) in triples.iter().enumerate() {
        names.push(format!("resid_{}_{}_{}", wn[w], FOLD_NAMES[f], FEAT_NAMES[x]));
        cols.push(resid[i].clone());
    }

    // 折叠间对比（只对归一化得分）
    let contrasts: [(&str, usize, usize); 4] = [("profit_minus_loss", 4, 5), ("bf_minus_sf", 6, 7), ("short_minus_cnt", 8, 0), ("dec_minus_cnt", 1, 0)];
    for w in 0..N_WIN {
        for a in 0..N_ALPHA {
            for &(nm, f1, f2) in &contrasts {
                names.push(format!("cx_{}_{}_{}", ALPHA_NAMES[a], wn[w], nm));
                let (c1, c2) = (fold_col(w, a, f1, 0), fold_col(w, a, f2, 0));
                cols.push(raw.iter().map(|r| r[c1] - r[c2]).collect());
            }
        }
    }
    // 时段对比：下午、尾盘的共识分位秩减全天
    let ci = |w: usize, f: usize, x: usize| (w * N_FOLD + f) * N_F + x;
    for (lab, w) in [("pm", 1usize), ("tail", 2usize)] {
        for f in 0..N_FOLD {
            for x in 0..N_F {
                names.push(format!("wx_{}_{}_{}", lab, FOLD_NAMES[f], FEAT_NAMES[x]));
                let (a, b) = (&cons[ci(w, f, x)], &cons[ci(0, f, x)]);
                cols.push(a.iter().zip(b).map(|(u, v)| u - v).collect());
            }
        }
    }
    (names, cols)
}

fn stock_codes(date: i64) -> Result<(Vec<String>, String, u64), String> {
    let root = std::env::var("RUST_PYFUNC_LEVEL2_ROOT").unwrap_or("/ssd_data/stock".into());
    let basic = std::env::var("RUST_PYFUNC_BASIC_INFO_DIR").unwrap_or("/ssd_data/data/basic_info".into());
    let map = Path::new(&basic).join("symbol_map.csv");
    let bytes = std::fs::read(&map).map_err(|e| format!("股票表读取失败 {}: {e}", map.display()))?;
    let mut reader = csv::Reader::from_reader(&bytes[..]);
    let column = reader.headers().map_err(|e| e.to_string())?.iter().position(|h| h == "symbol").ok_or("股票表缺少 symbol 列")?;
    let mut set = BTreeSet::new();
    for rec in reader.records() {
        let rec = rec.map_err(|e| e.to_string())?;
        let code = rec.get(column).unwrap_or("").trim();
        if code.is_empty() || code.len() > 6 || !code.bytes().all(|b| b.is_ascii_digit()) {
            return Err(format!("股票表含无效代码: {code}"));
        }
        set.insert(format!("{code:0>6}"));
    }
    if set.is_empty() { return Err("股票白名单为空".into()); }
    let dir = Path::new(&root).join(date.to_string()).join("transaction");
    if !dir.is_dir() { return Err(format!("原始数据目录不存在: {}", dir.display())); }
    let codes = set.into_iter().filter(|c| dir.join(format!("{c}_{date}_transaction.csv")).is_file()).collect();
    Ok((codes, map.display().to_string(), fnv(&bytes)))
}

fn env_i64(k: &str, d: i64) -> i64 {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let date: i64 = args.get(1).expect("用法: align_fold <date> <out_dir>").parse().unwrap();
    let out_dir = args.get(2).expect("用法: align_fold <date> <out_dir>").clone();
    let threads = (env_i64("FOLD_THREADS", 50) as usize).clamp(1, 50);
    let wins = windows(env_i64("FULL_BAR", 30), env_i64("PM_BAR", 30), env_i64("TAIL_BAR", 15));
    let latest = std::env::var("TIEBREAK").map(|v| v != "earliest").unwrap_or(true);
    let (codes, map_path, map_hash) = match stock_codes(date) {
        Ok(v) => v,
        Err(e) => { eprintln!("错误: {e}"); std::process::exit(2); }
    };
    eprintln!("codes = {}, threads = {threads}, 字母序列长度 = {:?}", codes.len(), wins.iter().map(|w| w.n()).collect::<Vec<_>>());
    let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
    let t0 = Instant::now();
    let (raw, (syn_names, syn_cols), t_stock) = pool.install(|| {
        let raw: Vec<Vec<f32>> = codes.par_iter().map(|c| per_stock(c, date, &wins, latest)).collect();
        let t_stock = t0.elapsed().as_secs_f64();
        let syn = synthesize(&raw, codes.len());
        (raw, syn, t_stock)
    });
    let t_all = t0.elapsed().as_secs_f64();
    eprintln!("per_stock = {t_stock:.2}s, 总计 = {t_all:.2}s");

    let day_dir = Path::new(&out_dir).join(date.to_string());
    std::fs::create_dir_all(&day_dir).unwrap();
    let mut names = raw_names();
    names.extend(syn_names);
    let n_cols = names.len();
    let mut w = BufWriter::with_capacity(1 << 20, std::fs::File::create(day_dir.join("vals.bin")).unwrap());
    for (i, r) in raw.iter().enumerate() {
        for v in r { w.write_all(&v.to_le_bytes()).unwrap(); }
        for c in &syn_cols { w.write_all(&c[i].to_le_bytes()).unwrap(); }
    }
    w.flush().unwrap();
    let win_info: Vec<serde_json::Value> = wins.iter().map(|x| serde_json::json!({"name": x.name, "bar_sec": x.bar, "n_letters": x.n()})).collect();
    let meta = serde_json::json!({
        "date": date, "names": names, "codes": codes, "n_rows": codes.len(), "n_cols": n_cols,
        "symbol_map": map_path, "symbol_map_fnv64": format!("{map_hash:016x}"),
        "threads": threads, "per_stock_sec": t_stock, "total_sec": t_all,
        "windows": win_info, "tiebreak_latest": latest,
    });
    std::fs::write(day_dir.join("meta.json"), serde_json::to_vec(&meta).unwrap()).unwrap();
    eprintln!("输出 {} 行 {} 列 → {}", codes.len(), n_cols, day_dir.display());
}
