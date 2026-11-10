//! 基因动态规划 · 方向 4「买卖配对折叠」：借 RNA 折叠——方向相反的行为字母可以配对，配对只能嵌套不能交叉，
//! 用动态规划求最优配对：外层是长周期来回，内层是短周期做 T，没配上的是当天净方向。
//! 由沙箱 sandbox_align_fold 第三版（fast 实现）原样搬入，因子值与沙箱逐字节一致，249,603 列，不做任何多因子合成。
//! 跨日配对同时读前一交易日的原始数据（不落盘）。
//! 列名约定：凡依赖撤单数据的列名都含 cxl，其余不含；撤单数据不存在的日期（沪市 20211105 前、深市 20160104 前）这些列写空值。
pub mod fold;
pub mod reader;
pub mod v3_feats;
pub mod v3_kernels;
pub mod v3_letters;

use hdf5_metno as hdf5;
use pyo3::prelude::*;
use reader::{read_market_fast, read_trade_fast, MarketRecord, TradeRecord};
use rayon::prelude::*;
use std::path::Path;
use std::time::Instant;
use v3_feats::{arc_feats, naive_feats, pair_feats, ARC_FEATS, NAIVE, PAIR_FEATS};
use v3_kernels::{cross_match, hpk, make_ctx, nested, prob_base, prob_fast, two_layer, Kern, Mode, PROB_FEATS, RULES};
use v3_letters::{base_variant, cancel_arcs, cxl_available, is_cxl, variants, windows, Bars, Seq, Win, BASE, HK_BARS, TIER_BARS, TIER_TAGS};

// 分段计时（只计时，不影响数值）：0 字母化 1 无交叉配对 2 二层 3 交叉匹配 4 配对概率 5 打乱对照 6 H型 7 跨日 8 联合 9 挂撤弧 10 特征
static TM: [std::sync::atomic::AtomicU64; 11] = [const { std::sync::atomic::AtomicU64::new(0) }; 11];
macro_rules! tm {
    ($i:expr, $e:expr) => {{
        let t = Instant::now();
        let r = $e;
        TM[$i].fetch_add(t.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
        r
    }};
}

const TASK_NAMES: [&str; 3] = ["all", "a", "b"];
const NV: usize = 38;
const NW: usize = 3;
const NG: usize = 2;
const NS: usize = NW * NG * NV; // 228 条序列
const NF: usize = 17; // 无交叉规则的特征数
const BLOCK: usize = 268;
const O_PK2: usize = 12 * NF; // 204
const O_CROSS: usize = O_PK2 + 19; // 223
const O_PROB: usize = O_CROSS + 19; // 242
const O_EXC: usize = O_PROB + 8; // 250
const O_NAIVE: usize = O_EXC + 9; // 259
const N_DIFFABLE: usize = O_NAIVE; // 259：参与时段差的列
const N_XS: usize = O_EXC; // 250：参与残差和行业中性的列
const EXC_RULES: [usize; 3] = [0, 1, 2];
const EXC_FEATS: [usize; 3] = [0, 5, 6];
const XD_F: usize = NF + 1; // 跨日配对：17 + 跨日配对占比
const CO_PAIRS: [(usize, usize); 6] = [(0, 5), (0, 6), (0, 4), (5, 6), (1, 2), (3, 7)]; // 基础字母下标
const CO_RULES: [usize; 3] = [0, 1, 2];
const WDIFF: [(&str, usize, usize); 3] = [("pm_minus_full", 1, 0), ("tail_minus_full", 2, 0), ("tail_minus_pm", 2, 1)];
const RDIFF: [(&str, usize, usize); 8] = [
    ("profit_minus_loss", 4 * NF + 1, 5 * NF + 1),
    ("bf_minus_sf", 6 * NF + 1, 7 * NF + 1),
    ("short_minus_cnt", 8 * NF + 1, 1),
    ("dec_minus_cnt", NF + 1, 1),
    ("pk2_minus_cnt", O_PK2 + 1, 1),
    ("cross_minus_dec", O_CROSS + 1, NF + 1),
    ("stack_minus_cnt", 11 * NF + 1, 1),
    ("gain_minus_profit", 10 * NF + 1, 4 * NF + 1),
];

const OFF_HK: usize = NS * BLOCK;
const OFF_XD: usize = OFF_HK + NW * NV * 19;
const OFF_CO: usize = OFF_XD + 16 * NW * 12 * XD_F;
const OFF_ARC: usize = OFF_CO + CO_PAIRS.len() * NW * CO_RULES.len() * NF;
const OFF_WD: usize = OFF_ARC + 3 * NW * 19;
const OFF_RD: usize = OFF_WD + 3 * NV * NG * N_DIFFABLE;
const ROW: usize = OFF_RD + NS * RDIFF.len();
const N_SYN: usize = 2 * NS * N_XS;

/// 拆分任务：0=全部，1=细粒度 g1 的序列块及其时段差/规则差/残差/行业中性，2=其余全部
fn task() -> usize {
    0 // 正式库只用「全部」：一次输出全部 249,603 列（沙箱里拆两任务只为测速）
}
fn tier_in_task(g: usize, t: usize) -> bool {
    t == 0 || (t == 1 && g == 0) || (t == 2 && g == 1)
}
/// 某一列属于哪个任务
fn col_in_task(col: usize, t: usize) -> bool {
    if t == 0 { return true; }
    let g_of_series = |s: usize| (s / NV) % NG;
    let g = if col < OFF_HK { Some(g_of_series(col / BLOCK)) }
        else if col < OFF_WD { None }
        else if col < OFF_RD { Some(((col - OFF_WD) / N_DIFFABLE) % NG) }
        else if col < ROW { Some(g_of_series((col - OFF_RD) / RDIFF.len())) }
        else { Some(g_of_series(((col - ROW) % (NS * N_XS)) / N_XS)) };
    match g { Some(g) => tier_in_task(g, t), None => t == 2 }
}

#[inline]
fn sidx(w: usize, g: usize, v: usize) -> usize {
    (w * NG + g) * NV + v
}

fn names() -> Vec<String> {
    let vs = variants();
    let wn: Vec<&str> = windows().iter().map(|w| w.name).collect();
    let mut out = vec![String::new(); ROW + N_SYN];
    let block_suffix = |c: usize| -> String {
        if c < O_PK2 { format!("{}_{}", RULES[c / NF], PAIR_FEATS[c % NF]) }
        else if c < O_CROSS { format!("pk2_{}", PAIR_FEATS[c - O_PK2]) }
        else if c < O_PROB { format!("cross_{}", PAIR_FEATS[c - O_CROSS]) }
        else if c < O_EXC { format!("prob_{}", PROB_FEATS[c - O_PROB]) }
        else if c < O_NAIVE { let q = c - O_EXC; format!("excess_{}_{}", RULES[EXC_RULES[q / 3]], PAIR_FEATS[EXC_FEATS[q % 3]]) }
        else { format!("naive_{}", NAIVE[c - O_NAIVE]) }
    };
    for w in 0..NW {
        for g in 0..NG {
            for v in 0..NV {
                let s = sidx(w, g, v);
                for c in 0..BLOCK {
                    out[s * BLOCK + c] = format!("{}_{}_{}_{}", vs[v].name, wn[w], TIER_TAGS[g], block_suffix(c));
                }
                for (q, (nm, _, _)) in RDIFF.iter().enumerate() {
                    out[OFF_RD + s * RDIFF.len() + q] = format!("rd_{}_{}_{}_{}", vs[v].name, wn[w], TIER_TAGS[g], nm);
                }
                for c in 0..N_XS {
                    out[ROW + s * N_XS + c] = format!("rs_{}_{}_{}_{}", vs[v].name, wn[w], TIER_TAGS[g], block_suffix(c));
                    out[ROW + NS * N_XS + s * N_XS + c] = format!("in_{}_{}_{}_{}", vs[v].name, wn[w], TIER_TAGS[g], block_suffix(c));
                }
            }
        }
        for v in 0..NV {
            for f in 0..19 {
                out[OFF_HK + (w * NV + v) * 19 + f] = format!("{}_{}_hk_hpk_{}", vs[v].name, wn[w], PAIR_FEATS[f]);
            }
        }
        for a in 0..16 {
            for r in 0..12 {
                for f in 0..XD_F {
                    let fname = if f < NF { PAIR_FEATS[f] } else { "straddle_share" };
                    out[OFF_XD + ((a * NW + w) * 12 + r) * XD_F + f] = format!("xd_{}_{}_{}_{}", BASE[a], wn[w], RULES[r], fname);
                }
            }
        }
        for (p, &(a, b)) in CO_PAIRS.iter().enumerate() {
            for (ri, &r) in CO_RULES.iter().enumerate() {
                for f in 0..NF {
                    out[OFF_CO + ((p * NW + w) * CO_RULES.len() + ri) * NF + f] = format!("co_{}_{}_{}_{}_{}", BASE[a], BASE[b], wn[w], RULES[r], PAIR_FEATS[f]);
                }
            }
        }
        for (si, sn) in ["buy", "sell", "both"].iter().enumerate() {
            for f in 0..19 {
                out[OFF_ARC + (si * NW + w) * 19 + f] = format!("cxlarc_{}_{}_{}", sn, wn[w], ARC_FEATS[f]);
            }
        }
    }
    for (d, (dn, _, _)) in WDIFF.iter().enumerate() {
        for v in 0..NV {
            for g in 0..NG {
                for c in 0..N_DIFFABLE {
                    out[OFF_WD + ((d * NV + v) * NG + g) * N_DIFFABLE + c] = format!("wd_{}_{}_{}_{}", dn, vs[v].name, TIER_TAGS[g], block_suffix(c));
                }
            }
        }
    }
    out
}

fn fnv(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
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

fn valid(c: &v3_kernels::Ctx) -> bool {
    c.n >= 20 && c.n_buy >= 3 && c.n_sell >= 3
}

/// 一条序列的 268 列
fn series_block(mode: Mode, k: &mut Kern, seq: &Seq, seed: u64, out: &mut [f32]) {
    let c = make_ctx(seq, None);
    naive_feats(&c, &mut out[O_NAIVE..BLOCK]);
    if !valid(&c) { return; }
    let mut cnt_pairs = Vec::new();
    for r in 0..12 {
        let (sn, pairs) = tm!(1, nested(mode, k, &c, r));
        tm!(10, pair_feats(&c, &pairs, sn, true, &mut out[r * NF..(r + 1) * NF]));
        if r == 0 { cnt_pairs = pairs; }
    }
    let (sn, p2) = tm!(2, two_layer(mode, k, &c, &cnt_pairs));
    pair_feats(&c, &p2, sn, false, &mut out[O_PK2..O_CROSS]);
    let (tot, pc) = tm!(3, cross_match(&c));
    let ub = 1000.0 * c.n_buy.min(c.n_sell) as f64;
    pair_feats(&c, &pc, (tot as f64 / ub) as f32, false, &mut out[O_CROSS..O_PROB]);
    let pr = tm!(4, match mode { Mode::Base => prob_base(&c, &mut k.prob), Mode::Fast => prob_fast(&c, &mut k.prob) });
    out[O_PROB..O_EXC].copy_from_slice(&pr);
    let mut sums = [[0f64; 3]; 3];
    let mut cnts = [[0u32; 3]; 3];
    let mut buf = [f32::NAN; NF];
    let t5 = Instant::now();
    for rep in 0..3u64 {
        let p = permute(seq, seed.wrapping_add((rep + 1).wrapping_mul(0x9E3779B97F4A7C15)));
        let cp = make_ctx(&p, None);
        for (i, &r) in EXC_RULES.iter().enumerate() {
            let (sn, pairs) = nested(mode, k, &cp, r);
            pair_feats(&cp, &pairs, sn, true, &mut buf);
            for (j, &f) in EXC_FEATS.iter().enumerate() {
                if buf[f].is_finite() { sums[i][j] += buf[f] as f64; cnts[i][j] += 1; }
            }
        }
    }
    TM[5].fetch_add(t5.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
    for (i, &r) in EXC_RULES.iter().enumerate() {
        for (j, &f) in EXC_FEATS.iter().enumerate() {
            let v = out[r * NF + f];
            out[O_EXC + i * 3 + j] = if cnts[i][j] > 0 && v.is_finite() { v - (sums[i][j] / cnts[i][j] as f64) as f32 } else { f32::NAN };
        }
    }
}

fn per_stock(code: &str, date: i64, prev: Option<i64>, mode: Mode) -> Vec<f32> {
    let mut row = vec![f32::NAN; ROW];
    v3_kernels::set_hpk_fast(mode == Mode::Fast);
    let tr: Vec<TradeRecord> = match read_trade_fast(code, date) { Ok(t) => t, Err(_) => return row };
    let mk: Vec<MarketRecord> = read_market_fast(code, date).unwrap_or_default();
    let cxl_ok = cxl_available(code, date);
    let (trp, mkp, cxl_ok_prev) = match prev {
        Some(p) => (read_trade_fast(code, p).ok(), read_market_fast(code, p).unwrap_or_default(), cxl_available(code, p)),
        None => (None, Vec::new(), false),
    };
    let vs = variants();
    let wins = windows();
    let code_seed = fnv(code.as_bytes());
    let mut k = Kern::new();
    let tk = task();
    // 两档粒度的字母序列（保留下来给 H 型假结、跨日、联合配对复用）
    let mut seqs: Vec<Vec<Option<Seq>>> = Vec::with_capacity(NW * NG);
    for w in 0..NW {
        for g in 0..NG {
            let win = wins[w].with_bar(TIER_BARS[w][g]);
            let ss: Vec<Option<Seq>> = tm!(0, {
                let bars = Bars::new(&tr, &mk, &win);
                vs.iter().map(|v| bars.as_ref().and_then(|b| b.seq(v, cxl_ok))).collect()
            });
            for v in 0..NV {
                if !tier_in_task(g, tk) { break; }
                if let Some(s) = &ss[v] {
                    let si = sidx(w, g, v);
                    series_block(mode, &mut k, s, code_seed ^ ((si as u64 + 1) << 24), &mut row[si * BLOCK..(si + 1) * BLOCK]);
                }
            }
            seqs.push(ss);
        }
    }
    // H 型假结：每个时段用约 120 个字母的粒度（全天 120 秒、下午 60 秒、尾盘 15 秒）
    for w in 0..NW {
        if tk == 1 { break; }
        let hk_seqs: Vec<Option<Seq>> = if HK_BARS[w] == TIER_BARS[w][1] { seqs[w * NG + 1].clone() }
            else if HK_BARS[w] == TIER_BARS[w][0] { seqs[w * NG].clone() }
            else {
                let b = Bars::new(&tr, &mk, &wins[w].with_bar(HK_BARS[w]));
                vs.iter().map(|v| b.as_ref().and_then(|b| b.seq(v, cxl_ok))).collect()
            };
        for v in 0..NV {
            let Some(s) = &hk_seqs[v] else { continue };
            let c = make_ctx(s, None);
            if !valid(&c) { continue; }
            let (best, pairs) = tm!(6, hpk(&c, &mut k.hpk));
            let sn = (best as f64 / c.n_buy.min(c.n_sell) as f64) as f32;
            let o = OFF_HK + (w * NV + v) * 19;
            pair_feats(&c, &pairs, sn, false, &mut row[o..o + 19]);
        }
    }
    // 跨日配对：昨天与今天同一时段的字母首尾相接（全天、下午 60 秒，尾盘 15 秒），只用基准阈值版本
    let t7 = Instant::now();
    if let (Some(trp), true) = (&trp, tk != 1) {
        for w in 0..NW {
            let g = if w == 2 { 0 } else { 1 };
            let bars_p = Bars::new(trp, &mkp, &wins[w].with_bar(TIER_BARS[w][g]));
            for a in 0..16 {
                let v = base_variant(a);
                let (Some(today), Some(bp)) = (&seqs[w * NG + g][v], &bars_p) else { continue };
                let Some(yday) = bp.seq(&vs[v], cxl_ok && cxl_ok_prev) else { continue };
                if is_cxl(a) && !(cxl_ok && cxl_ok_prev) { continue; }
                let cat = Seq::concat(&yday, today);
                let c = make_ctx(&cat, None);
                if !valid(&c) { continue; }
                let ny = yday.side.len();
                for r in 0..12 {
                    let (sn, pairs) = nested(mode, &mut k, &c, r);
                    let o = OFF_XD + ((a * NW + w) * 12 + r) * XD_F;
                    pair_feats(&c, &pairs, sn, true, &mut row[o..o + NF]);
                    if !pairs.is_empty() {
                        let st = pairs.iter().filter(|&&(x, y)| x < ny && y >= ny).count();
                        row[o + NF] = st as f32 / pairs.len() as f32;
                    }
                }
            }
        }
    }
    TM[7].fetch_add(t7.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
    let t8 = Instant::now();
    // 两字母联合配对：只有两种字母在两端都方向相反才可配对
    for w in 0..NW {
        if tk == 1 { break; }
        for (p, &(a, b)) in CO_PAIRS.iter().enumerate() {
            let (Some(sa), Some(sb)) = (&seqs[w * NG][base_variant(a)], &seqs[w * NG][base_variant(b)]) else { continue };
            let c = make_ctx(sa, Some(&sb.side));
            if !valid(&c) { continue; }
            for (ri, &r) in CO_RULES.iter().enumerate() {
                let (sn, pairs) = nested(mode, &mut k, &c, r);
                let o = OFF_CO + ((p * NW + w) * CO_RULES.len() + ri) * NF;
                pair_feats(&c, &pairs, sn, true, &mut row[o..o + NF]);
            }
        }
    }
    TM[8].fetch_add(t8.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
    let t9 = Instant::now();
    // 真实挂撤弧（撤单数据存在才算）
    if cxl_ok && !tr.is_empty() && tk != 1 {
        let day0 = tr[0].time_us - tr[0].time_us.rem_euclid(86_400_000_000);
        for w in 0..NW {
            let win: &Win = &wins[w];
            let arcs = cancel_arcs(&tr, win);
            let lo = day0 + win.adj_lo * 1_000_000;
            let len = (win.adj_hi - win.adj_lo) * 1_000_000;
            for (si, filt) in [Some(true), Some(false), None].iter().enumerate() {
                let sel: Vec<(i64, i64, bool, f64, bool, bool)> = arcs
                    .iter()
                    .filter(|a| filt.map_or(true, |b| a.buy == b))
                    .map(|a| (a.start_us, a.end_us, a.buy, a.vol, !a.clipped, a.unmapped))
                    .collect();
                let o = OFF_ARC + (si * NW + w) * 19;
                arc_feats(&sel, lo, len, TIER_BARS[w][0] * 1_000_000, &mut row[o..o + 19]);
            }
        }
    }
    TM[9].fetch_add(t9.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
    // 单股内的时段差与规则差
    for (d, &(_, wa, wb)) in WDIFF.iter().enumerate() {
        for v in 0..NV {
            for g in 0..NG {
                let (sa, sb) = (sidx(wa, g, v) * BLOCK, sidx(wb, g, v) * BLOCK);
                let o = OFF_WD + ((d * NV + v) * NG + g) * N_DIFFABLE;
                for c in 0..N_DIFFABLE {
                    row[o + c] = row[sa + c] - row[sb + c];
                }
            }
        }
    }
    for s in 0..NS {
        for (q, &(_, a, b)) in RDIFF.iter().enumerate() {
            row[OFF_RD + s * RDIFF.len() + q] = row[s * BLOCK + a] - row[s * BLOCK + b];
        }
    }
    row
}

// ---------------- 横截面变换（单因子）：分位秩后对朴素量回归取残差；分位秩减行业均值 ----------------

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

fn industry_neutral(y: &[f32], ind: &[i32]) -> Vec<f32> {
    let mut s = [0f64; 64];
    let mut c = [0u32; 64];
    for i in 0..y.len() {
        if y[i].is_finite() && ind[i] >= 0 { s[ind[i] as usize] += y[i] as f64; c[ind[i] as usize] += 1; }
    }
    (0..y.len()).map(|i| if y[i].is_finite() && ind[i] >= 0 { (y[i] as f64 - s[ind[i] as usize] / c[ind[i] as usize] as f64) as f32 } else { f32::NAN }).collect()
}

const CONTROLS: [usize; 4] = [2, 4, 5, 6]; // 平凡上界、量不平衡、翻转率、累计走势振幅（同一条序列的朴素量）

fn synthesize(rows: &[Vec<f32>], ind: &[i32]) -> Vec<Vec<f32>> {
    let col = |c: usize| -> Vec<f32> { rows.iter().map(|r| r[c]).collect() };
    let tk = task();
    let per: Vec<(Vec<Vec<f32>>, Vec<Vec<f32>>)> = (0..NS).into_par_iter().map(|s| {
        if !tier_in_task((s / NV) % NG, tk) { return (vec![Vec::new(); N_XS], vec![Vec::new(); N_XS]); }
        let ctrl: Vec<Vec<f32>> = CONTROLS.iter().map(|&x| pct_rank(&col(s * BLOCK + O_NAIVE + x))).collect();
        let mut rs = Vec::with_capacity(N_XS);
        let mut ne = Vec::with_capacity(N_XS);
        for c in 0..N_XS {
            let y = pct_rank(&col(s * BLOCK + c));
            rs.push(ols_resid(&y, &ctrl));
            ne.push(industry_neutral(&y, ind));
        }
        (rs, ne)
    }).collect();
    let mut out = Vec::with_capacity(N_SYN);
    let mut ne_all = Vec::with_capacity(NS * N_XS);
    for (rs, ne) in per {
        out.extend(rs);
        ne_all.extend(ne);
    }
    out.extend(ne_all);
    out
}

fn industry(codes: &[String], date: i64) -> Result<Vec<i32>, String> {
    let dir = crate::genedp_common::vars_file("SzBa");
    let cal: Vec<i64> = std::fs::read_to_string(format!("{dir}/calendar_map.csv")).map_err(|e| e.to_string())?
        .lines().skip(1).map(|l| l.split(',').next().unwrap().trim().parse().unwrap()).collect();
    let sym: Vec<String> = std::fs::read_to_string(format!("{dir}/symbol_map.csv")).map_err(|e| e.to_string())?
        .lines().skip(1).map(|l| l.split(',').next().unwrap().trim().to_string()).collect();
    let r = cal.iter().rposition(|&d| d <= date).ok_or("行业日历中没有早于该日的行")?;
    let f = hdf5::File::open(format!("{dir}/industry.h5")).map_err(|e| e.to_string())?;
    let ds = f.dataset("data").map_err(|e| e.to_string())?;
    let row: Vec<f64> = ds.read_slice_1d::<f64, _>((r, 0..sym.len())).map_err(|e| e.to_string())?.to_vec();
    let pos: std::collections::HashMap<&str, usize> = sym.iter().enumerate().map(|(i, s)| (s.as_str(), i)).collect();
    Ok(codes.iter().map(|c| match pos.get(c.as_str()) {
        Some(&i) if row[i].is_finite() && row[i] >= 0.0 && row[i] < 64.0 => row[i] as i32,
        _ => -1,
    }).collect())
}

fn stock_codes(root: &str, date: i64) -> Result<(Vec<String>, String, u64), String> {
    let map = crate::data_paths::basic_info_path("symbol_map.csv");
    let bytes = std::fs::read(&map).map_err(|e| format!("股票表读取失败 {}: {e}", map.display()))?;
    let mut reader = csv::Reader::from_reader(&bytes[..]);
    let column = reader.headers().map_err(|e| e.to_string())?.iter().position(|h| h == "symbol").ok_or("股票表缺少 symbol 列")?;
    let mut set = std::collections::BTreeSet::new();
    for rec in reader.records() {
        let rec = rec.map_err(|e| e.to_string())?;
        let code = rec.get(column).unwrap_or("").trim();
        if code.is_empty() || code.len() > 6 || !code.bytes().all(|b| b.is_ascii_digit()) { return Err(format!("股票表含无效代码: {code}")); }
        set.insert(format!("{code:0>6}"));
    }
    if set.is_empty() { return Err("股票白名单为空".into()); }
    let dir = Path::new(root).join(date.to_string()).join("transaction");
    if !dir.is_dir() { return Err(format!("原始数据目录不存在: {}", dir.display())); }
    let mut codes: Vec<String> = set.into_iter().filter(|c| dir.join(format!("{c}_{date}_transaction.csv")).is_file()).collect();
    Ok((codes, map.display().to_string(), fnv(&bytes)))
}

/// 全部列名（249,603 列）：原始层 ROW 列 + 残差与行业中性 N_SYN 列
pub fn genedp_buysell_fold_names() -> Vec<String> {
    names()
}

pub const N_FACTORS: usize = ROW + N_SYN;

/// 核心唯一真相源：读今天（和前一交易日）全市场原始数据 → 配对折叠特征 → 残差与行业内中性。
/// 返回 (股票代码, 行优先展平的因子值)，每股 N_FACTORS 列。用当前 rayon 线程池并行。
pub fn compute_genedp_buysell_fold_full(date: i64) -> std::io::Result<(Vec<String>, Vec<f32>)> {
    let root = crate::genedp_common::level2_root_string();
    let (codes, _map_path, _map_hash) = stock_codes(&root, date).map_err(crate::genedp_common::invalid)?;
    let ind = industry(&codes, date).map_err(|e| crate::genedp_common::invalid(format!("行业读取失败 {e}")))?;
    let prev = crate::genedp_common::prev_level2_date(date);
    let rows: Vec<Vec<f32>> = codes.par_iter().map(|c| per_stock(c, date, prev, Mode::Fast)).collect();
    let syn = synthesize(&rows, &ind);
    let ncol = N_FACTORS;
    let mut vals = vec![f32::NAN; rows.len() * ncol];
    vals.par_chunks_mut(ncol).enumerate().for_each(|(r, out)| {
        out[..ROW].copy_from_slice(&rows[r]);
        for (c, col) in syn.iter().enumerate() {
            out[ROW + c] = col[r];
        }
    });
    Ok((codes, vals))
}

/// 单日全市场调试入口（50 线程）：返回 (股票代码, 展平因子值)，每股 N_FACTORS 列
#[pyfunction]
pub fn py_genedp_buysell_fold(py: Python<'_>, date: i64) -> PyResult<(Vec<String>, Vec<f32>)> {
    py.allow_threads(|| crate::genedp_common::run_in_50_threads(|| compute_genedp_buysell_fold_full(date)))
        .and_then(|r| r)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("{e:?}")))
}

#[pyfunction]
pub fn py_genedp_buysell_fold_names() -> Vec<String> {
    genedp_buysell_fold_names()
}
