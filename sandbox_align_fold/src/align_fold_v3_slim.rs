//! 方向 4（RNA 折叠式买卖配对）第三版·精简版：单日全市场，67,467 列，不做任何多因子合成。
//! 由第三版完整版（249,603 列）按 20241231 全市场排名相关普查去重得到，保留列的数值与完整版逐字节相同：
//!   完全重复/常数列（未配对净方向、计数类规则的得分=配对比例、恒定列）→ 删；
//!   行业内中性、二层折叠、ordjump 的 0.5/2 倍门槛、块内 |相关|>0.95 的近似重复 → 删；
//!   对朴素量取残差整层、时段差「尾盘减下午」、6 种成交/撤单字母的 0.5 倍门槛、块内 0.85 以上近亲 → 删。
//! 被删的字母版本和二层折叠不再计算；残差与行业中性层整层不算，所以不再读行业。
//! 运行：align_fold_v3_slim <date> <输出目录>
//! 环境变量：FOLD_IMPL=base|fast（基准写法 / 提速写法，结果必须逐字节相同）、FOLD_THREADS（≤50）、
//!           RUST_PYFUNC_LEVEL2_ROOT / RUST_PYFUNC_LEVEL2_PATH / RUST_PYFUNC_BASIC_INFO_DIR、ONLY_CODES（逗号分隔，调试用）。
//! 列名约定：凡依赖撤单数据的列名都含 cxl，其余不含；撤单数据不存在的日期（沪市 20211105 前、深市 20160104 前）这些列写空值。
#[path = "fast_csv_reader.rs"]
mod fast_csv_reader;
#[path = "fold.rs"]
mod fold;
#[path = "v3_feats.rs"]
mod v3_feats;
#[path = "v3_kernels.rs"]
mod v3_kernels;
#[path = "v3_letters.rs"]
mod v3_letters;

use fast_csv_reader::{read_market_fast, read_trade_fast, MarketRecord, TradeRecord};
use rayon::prelude::*;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::time::Instant;
use v3_feats::{arc_feats, naive_feats, pair_feats, ARC_FEATS, NAIVE, PAIR_FEATS};
use v3_kernels::{cross_match, hpk, make_ctx, nested, prob_base, prob_fast, Kern, Mode, PROB_FEATS, RULES};
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
/// 精简版输出列数（固定）
const N_SLIM: usize = 67_467;

/// 拆分任务：0=全部，1=细粒度 g1 的序列块及其时段差/规则差/残差/行业中性，2=其余全部
fn task() -> usize {
    match std::env::var("TASK").as_deref() { Ok("a") => 1, Ok("b") => 2, _ => 0 }
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

// ---------------- 精简版保留规则（按列名判断；与 20241231 普查定下的 67,467 列名单逐一核对过） ----------------

/// 整个字母版本都删掉（不再计算）：6 种成交/撤单字母的 0.5 倍门槛、ordjump 的 0.5 和 2 倍门槛
const DROP_VARIANTS: [&str; 8] = ["flow_d05", "count_d05", "size_d05", "cxl_vol_d05", "volbar_d05", "cxl_cnt_d05", "ordjump_d05", "ordjump_d20"];
/// 得分恰好等于配对比例的规则（每对得分都是 1）
const COUNT_RULES: [&str; 8] = ["cnt", "profit", "loss", "bf", "sf", "short", "h10", "pk2"];
/// 块内 |相关| > 0.95 的近似重复
const NEAR_DUP: [&str; 7] = ["_cross_branch", "_cross_span_mean", "_gain_gain_bps", "_short_hairpin", "_prob_free_energy", "_excess_cnt_span_mean", "_excess_vol_span_mean"];
/// 块内 0.85 以上近亲（每组留一个代表，这里列的是被代替的那些）
const KIN_085: [&str; 34] = [
    "bf_span_mean", "cnt_span_mean", "cross_depth_max", "cross_depth_mean", "cross_hairpin", "cross_pair_center", "cross_span_p90", "dec_branch",
    "dec_pair_ratio", "dec_stem", "excess_dec_span_mean", "gain_span_p50", "h10_pair_ratio", "h10_span_mean", "h10_span_p50", "h10_stem",
    "loss_span_mean", "naive_active_share", "prob_exp_pair_ratio", "profit_span_mean", "rna_pair_ratio", "rna_score", "rna_span_mean", "rna_stem",
    "sf_span_mean", "short_branch", "short_pair_center", "short_pair_ratio", "short_span_p50", "short_stem", "stack_gain_bps", "stack_pair_ratio",
    "stack_span_mean", "vol_span_mean",
];

fn variant_kept(v: &str) -> bool {
    !DROP_VARIANTS.contains(&v)
}

/// 列名里是否出现某个字母版本（开头或以下划线隔开）
fn has_variant(n: &str, v: &str) -> bool {
    n.starts_with(&format!("{v}_")) || n.contains(&format!("_{v}_"))
}

/// 块内后缀：第一个 _g1_ / _g2_ 之后的部分（原始块与时段差适用）
fn block_suffix_of(n: &str) -> Option<&str> {
    let p = [n.find("_g1_"), n.find("_g2_")].into_iter().flatten().min()?;
    Some(&n[p + 4..])
}

fn col_kept(n: &str) -> bool {
    let layer = if n.starts_with("rs_") { "rs" } else if n.starts_with("in_") { "in" } else if n.starts_with("wd_") { "wd" }
        else if n.starts_with("rd_") { "rd" } else if n.starts_with("xd_") { "xd" } else if n.starts_with("co_") { "co" }
        else if n.starts_with("cxlarc_") { "arc" } else if n.contains("_hk_hpk_") { "hk" } else { "block" };
    // 第一档：完全重复与常数
    if n.ends_with("_unp_net") {
        let dup = match layer { "block" => true, "hk" => !n.contains("_full_"), _ => !n.ends_with("_cnt_unp_net") };
        if dup { return false; }
    }
    if n.ends_with("naive_buy_share") || n.ends_with("naive_walk_end") { return false; }
    if !n.contains("excess_") && COUNT_RULES.iter().any(|r| n.ends_with(&format!("_{r}_score"))) { return false; }
    if n.ends_with("_hk_hpk_score") || n.ends_with("_bf_bf_share") || n.ends_with("_sf_bf_share") || n.ends_with("_cross_pair_ratio") { return false; }
    // 第二档：整层或整块高度雷同
    if layer == "in" || layer == "rs" { return false; }
    if n.contains("_pk2_") || n.ends_with("pk2_minus_cnt") { return false; }
    if DROP_VARIANTS.iter().any(|v| has_variant(n, v)) { return false; }
    if NEAR_DUP.iter().any(|x| n.ends_with(x)) { return false; }
    // 精简档
    if n.starts_with("wd_tail_minus_pm") { return false; }
    if layer == "block" || layer == "wd" {
        if let Some(s) = block_suffix_of(n) {
            if KIN_085.contains(&s) { return false; }
        }
    }
    true
}

/// 精简版保留的列在完整版原始层（ROW 列）里的位置，按原顺序
fn kept_cols() -> Vec<usize> {
    names().iter().enumerate().take(ROW).filter(|(_, n)| col_kept(n)).map(|(i, _)| i).collect()
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
    let _ = cnt_pairs; // 二层折叠（pk2）在精简版里整块删除，不再计算
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
                if !variant_kept(&vs[v].name) { continue; }
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
            if !variant_kept(&vs[v].name) { continue; }
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
    let dir = "/ssd_data/data/vars/SzBa";
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
    let basic = std::env::var("RUST_PYFUNC_BASIC_INFO_DIR").unwrap_or("/ssd_data/data/basic_info".into());
    let map = Path::new(&basic).join("symbol_map.csv");
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
    if let Ok(only) = std::env::var("ONLY_CODES") {
        let keep: std::collections::BTreeSet<String> = only.split(',').map(|s| s.trim().to_string()).collect();
        codes.retain(|c| keep.contains(c));
    }
    Ok((codes, map.display().to_string(), fnv(&bytes)))
}

/// 上一个交易日：数据根目录下早于 date、且含 transaction 子目录的最大日期（只列一层目录）
fn prev_date(root: &str, date: i64) -> Option<i64> {
    std::fs::read_dir(root).ok()?
        .filter_map(|e| e.ok()?.file_name().to_str()?.parse::<i64>().ok())
        .filter(|&d| d < date && d > 19000000 && Path::new(root).join(d.to_string()).join("transaction").is_dir())
        .max()
}

fn peak_rss_mb() -> f64 {
    std::fs::read_to_string("/proc/self/status").ok()
        .and_then(|s| s.lines().find(|l| l.starts_with("VmHWM")).map(|l| l.split_whitespace().nth(1).unwrap().parse::<f64>().unwrap() / 1024.0))
        .unwrap_or(f64::NAN)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let date: i64 = args.get(1).expect("用法: align_fold_v3 <date> <out_dir>").parse().unwrap();
    let out_dir = args.get(2).expect("用法: align_fold_v3 <date> <out_dir>").clone();
    let mode = match std::env::var("FOLD_IMPL").as_deref() { Ok("base") => Mode::Base, _ => Mode::Fast };
    let threads = std::env::var("FOLD_THREADS").ok().and_then(|v| v.parse().ok()).unwrap_or(50usize).clamp(1, 50);
    let root = std::env::var("RUST_PYFUNC_LEVEL2_ROOT").unwrap_or("/ssd_data/stock".into());
    let (codes, map_path, map_hash) = match stock_codes(&root, date) {
        Ok(v) => v,
        Err(e) => { eprintln!("错误: {e}"); std::process::exit(2); }
    };
    let prev = prev_date(&root, date);
    eprintln!("codes = {}, threads = {threads}, mode = {}, prev = {:?}, 列数 = {}", codes.len(), if mode == Mode::Base { "base" } else { "fast" }, prev, N_SLIM);
    let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
    let t0 = Instant::now();
    let rows: Vec<Vec<f32>> = pool.install(|| codes.par_iter().map(|c| per_stock(c, date, prev, mode)).collect());
    let t_stock = t0.elapsed().as_secs_f64();
    let t_syn = 0.0f64; // 精简版没有横截面变换
    let tk = task();
    let keep = kept_cols();
    assert_eq!(keep.len(), N_SLIM, "精简版列数应恒为 {N_SLIM}");
    let full_names = names();
    let all_names: Vec<String> = keep.iter().map(|&c| full_names[c].clone()).collect();
    let day_dir = Path::new(&out_dir).join(date.to_string());
    std::fs::create_dir_all(&day_dir).unwrap();
    let mut w = BufWriter::with_capacity(64 << 20, std::fs::File::create(day_dir.join("vals.bin")).unwrap());
    let mut line: Vec<f32> = Vec::with_capacity(keep.len());
    for row in &rows {
        line.clear();
        line.extend(keep.iter().map(|&c| row[c]));
        let raw: &[u8] = unsafe { std::slice::from_raw_parts(line.as_ptr() as *const u8, line.len() * 4) };
        w.write_all(raw).unwrap();
    }
    w.flush().unwrap();
    drop(w);
    let t_all = t0.elapsed().as_secs_f64();
    let n_cxl = all_names.iter().filter(|n| n.contains("cxl")).count();
    let tm_names = ["letters", "nested", "two_layer", "cross_match", "prob", "shuffle", "hpk", "crossday", "cofold", "arcs", "pair_feats"];
    let tm_sec: serde_json::Map<String, serde_json::Value> = tm_names.iter().enumerate()
        .map(|(i, n)| (n.to_string(), serde_json::json!(TM[i].load(std::sync::atomic::Ordering::Relaxed) as f64 / 1e9))).collect();
    eprintln!("分段累计 CPU 秒: {:?}", tm_sec);
    let meta = serde_json::json!({
        "stage_cpu_sec": tm_sec,
        "date": date, "prev_date": prev, "names": all_names, "codes": codes, "n_rows": rows.len(), "n_cols": all_names.len(), "task": TASK_NAMES[tk],
        "n_cols_cxl": n_cxl, "layout": "float32 小端，行=股票，列按 names 顺序",
        "symbol_map": map_path, "symbol_map_fnv64": format!("{map_hash:016x}"), "threads": threads,
        "mode": if mode == Mode::Base { "base" } else { "fast" },
        "per_stock_sec": t_stock, "synth_sec": t_syn, "write_sec": t_all - t_stock - t_syn, "total_sec": t_all,
        "peak_rss_mb": peak_rss_mb(),
        "cxl_start": {"SH": 20211105, "SZ": 20160104},
        "windows": windows().iter().enumerate().map(|(i, x)| serde_json::json!({"name": x.name, "tier_bars_sec": TIER_BARS[i], "hk_bar_sec": HK_BARS[i]})).collect::<Vec<_>>(),
    });
    std::fs::write(day_dir.join("meta.json"), serde_json::to_vec(&meta).unwrap()).unwrap();
    eprintln!("per_stock = {t_stock:.2}s, 横截面变换 = {t_syn:.2}s, 写盘 = {:.2}s, 总计 = {t_all:.2}s, 峰值内存 = {:.0} MB, 输出 {} 行 {} 列（含 cxl {} 列）",
        t_all - t_stock - t_syn, peak_rss_mb(), rows.len(), all_names.len(), n_cxl);
}
