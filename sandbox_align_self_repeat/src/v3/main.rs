//! 方向 2「自比对找重复」第三版：单日全市场（同时重算前一交易日，用于跨日变化）。
//!
//! 层级
//!   第 1 层 字母化：53 条序列（19 条含撤单的混合序列 + 19 条只用成交的孪生版 + 15 条其他）
//!   第 2 层 自比对：带宽局部比对（align.rs；--ref 用参考实现，默认用逐位等价的提速实现）
//!   第 3 层 特征：每序列每时段 27 个特征 + 5 个朴素对照 = 32 个数
//!   第 4 层 横截面：同一模板多股重现（2 个特征）、活跃度/前一日流通市值/行业 三种组内排名、
//!          三种时段差、跨日差。不做任何多因子合成。
//!
//! 撤单约定：凡依赖撤单数据的列名字都含 cxl，其他列都不含。撤单只看标记 32。
//!   沪市（6 开头）20211105 前、深市（0/3 开头）20160104 前，cxl 列一律空值。
//!   非 cxl 列的事件流、活跃度分组都只用成交，与撤单数据无关。
//! 时段（下午已前移，从 9:30 起的秒）：全天 [0,14220)、下午 [7200,14220)=13:00-14:57、
//!   尾盘 [12420,14220)=14:27-14:57，均不含集合竞价。
//!
//! 运行：align_v3 <date> --out <prefix> [--ref]
//! 产物：<prefix>.f32（按列存放的 32 位浮点，小端，列数×股票数）+ <prefix>.meta.json
#[path = "../align.rs"]
mod align;
#[path = "../fast_csv_reader.rs"]
mod fast_csv_reader;

use align::{select_nonoverlap, self_align_affine, self_align_fast, self_align_ref, AlignOut, Seg};
use fast_csv_reader::{read_market_fast, read_trade_fast, MarketRecord, TradeRecord};
use hdf5_metno as hdf5;
use rayon::prelude::*;
use serde::Serialize;
use std::collections::HashMap;
use std::io::Write;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::time::Instant;

static PROF_READ: AtomicU64 = AtomicU64::new(0);
static PROF_LETTER: AtomicU64 = AtomicU64::new(0);
static PROF_DP: AtomicU64 = AtomicU64::new(0);
static PROF_FEAT: AtomicU64 = AtomicU64::new(0);
static PROF_CELLS: AtomicU64 = AtomicU64::new(0);
static PROF_SPEC_US: [AtomicU64; 64] = [const { AtomicU64::new(0) }; 64];
static PROF_SPEC_CELLS: [AtomicU64; 64] = [const { AtomicU64::new(0) }; 64];
static PROF_SPEC_EMIT: [AtomicU64; 64] = [const { AtomicU64::new(0) }; 64];

const THREADS: usize = 50;
const SEG_CAP: usize = 20_000;
const MIN_N: usize = 30;
const MOTIF_CAP: usize = 64;
const MOTIF_UNIT_MAX: usize = 16;
const SH_CXL_START: i64 = 20211105;
const SZ_CXL_START: i64 = 20160104;
const DAY_START_SEC: i64 = 9 * 3600 + 30 * 60;
const WINDOWS: [(&str, i64, i64); 3] = [("full", 0, 14_220), ("pm", 7_200, 14_220), ("tail30", 12_420, 14_220)];
const N_WIN: usize = 3;
const FEATS: [&str; 27] = [
    "cov", "cov_vol", "seg_k", "len_mean", "len_max", "score_per_event", "tand_cov", "tand_copy",
    "lag_modeshr", "lag_logmean", "lag_ent", "tgap_logmed", "tgap_cv", "gaprate", "mmrate", "cov_bs",
    "center", "cluster", "size_ratio", "impact_ratio", "motif_distinct", "motif_topshr",
    "cons_identity", "cons_loglen", "cons_copies", "xshare_frac", "xshare_log",
];
const N_F: usize = FEATS.len();
const F_XFRAC: usize = 25;
const F_XLOG: usize = 26;
const BASES: [&str; 5] = ["lag1_match", "ac_excess_max", "lz78_norm", "run_mean", "entropy"];
const N_B: usize = BASES.len();
const N_J: usize = N_F + N_B;

#[derive(Clone, Copy, PartialEq)]
enum Src {
    Mixed,
    Trade,
}

#[derive(Clone, Copy, PartialEq)]
enum Kind {
    Dir,
    Size,
    Lot,
    Gap,
    GapExact,
    Oid,
    SizeGap,
    SizeOid,
    Limit(bool), // true = 涨跌停状态下的事件，false = 非涨跌停
    Side(u8),    // 0 主买 1 主卖 2 撤单
    Big(u8),     // 0 大单全部 1 大单买 2 大单卖
    PxStep,
    Parent,
    BarVolFlow(i64),
    BarCxl(i64),
    BarLob(i64),
    BarPxPath(i64),
    BarDepth(i64),
}

#[derive(Clone, Copy)]
enum GapMode {
    Linear(i32),
    Affine,
}

struct Spec {
    name: String,
    kind: Kind,
    src: Src,
    max_lag: usize, // 0 = 序列长度的一半
    q: f64,
    gap: GapMode,
    compress: bool,
    cxl: bool,
}

fn specs() -> Vec<Spec> {
    use GapMode::*;
    use Kind::*;
    let ev: [(&str, Kind, usize, f64, GapMode, bool); 19] = [
        ("e_size_raw", Size, 128, 0.8, Linear(1), false),
        ("e_lot_raw", Lot, 128, 0.8, Linear(1), false),
        ("e_gapsec_raw", GapExact, 128, 0.8, Linear(1), false),
        ("e_dir_rc", Dir, 128, 0.8, Linear(1), true),
        ("e_size_rc", Size, 128, 0.8, Linear(1), true),
        ("e_lot_rc", Lot, 128, 0.8, Linear(1), true),
        ("e_gap_rc", Gap, 128, 0.8, Linear(1), true),
        ("e_gapsec_rc", GapExact, 128, 0.8, Linear(1), true),
        ("e_oid_rc", Oid, 128, 0.8, Linear(1), true),
        ("e_sizegap_rc", SizeGap, 128, 0.8, Linear(1), true),
        ("e_sizeoid_rc", SizeOid, 128, 0.8, Linear(1), true),
        ("e_size_rc_L32", Size, 32, 0.8, Linear(1), true),
        ("e_size_rc_L512", Size, 512, 0.8, Linear(1), true),
        ("e_size_rc_nogap", Size, 128, 0.8, Linear(1000), true),
        ("e_size_rc_q65", Size, 128, 0.65, Linear(1), true),
        ("e_size_rc_q90", Size, 128, 0.9, Linear(1), true),
        ("e_size_rc_affine", Size, 128, 0.8, Affine, true),
        ("nonlimit_sizegap_rc", Limit(false), 128, 0.8, Linear(1), true),
        ("atlimit_sizegap_rc", Limit(true), 128, 0.8, Linear(1), true),
    ];
    let mut v = Vec::new();
    for (src, prefix, cxl) in [(Src::Mixed, "cxl_", true), (Src::Trade, "tr_", false)] {
        for (n, k, l, q, g, rc) in ev.iter() {
            v.push(Spec { name: format!("{prefix}{n}"), kind: *k, src, max_lag: *l, q: *q, gap: *g, compress: *rc, cxl });
        }
    }
    let mut add = |name: &str, kind: Kind, src: Src, max_lag: usize, compress: bool, cxl: bool| {
        v.push(Spec { name: name.into(), kind, src, max_lag, q: 0.8, gap: Linear(1), compress, cxl });
    };
    add("cxl_only_sizegap_rc", Side(2), Src::Mixed, 128, true, true);
    add("buy_sizegap_rc", Side(0), Src::Trade, 128, true, false);
    add("sell_sizegap_rc", Side(1), Src::Trade, 128, true, false);
    add("bar15_volflow", BarVolFlow(15), Src::Trade, 0, false, false);
    add("bar60_volflow", BarVolFlow(60), Src::Trade, 0, false, false);
    add("cxl_bar15_size", BarCxl(15), Src::Mixed, 0, false, true);
    add("bar15_lob_rc", BarLob(15), Src::Trade, 0, true, false);
    add("tr_pxstep_rc", PxStep, Src::Trade, 128, true, false);
    add("bar15_pxpath_rc", BarPxPath(15), Src::Trade, 0, true, false);
    add("bar15_depth_rc", BarDepth(15), Src::Trade, 0, true, false);
    add("bar60_depth_rc", BarDepth(60), Src::Trade, 0, true, false);
    add("big_all_sizegap_rc", Big(0), Src::Trade, 128, true, false);
    add("big_buy_sizegap_rc", Big(1), Src::Trade, 128, true, false);
    add("big_sell_sizegap_rc", Big(2), Src::Trade, 128, true, false);
    add("parent_sizedur_rc", Parent, Src::Trade, 128, true, false);
    assert_eq!(v.len(), 53);
    for s in &v {
        assert_eq!(s.cxl, s.name.contains("cxl"), "命名约定：依赖撤单的序列名必须含 cxl，其他不含: {}", s.name);
    }
    v
}

#[inline]
fn raw_idx(s: usize, w: usize, j: usize) -> usize {
    (s * N_WIN + w) * N_J + j
}

fn cxl_available(code: &str, date: i64) -> bool {
    match code.as_bytes().first() {
        Some(b'6') => date >= SH_CXL_START,
        Some(b'0') | Some(b'3') => date >= SZ_CXL_START,
        _ => false,
    }
}

// ---------------------- 第 1 层 字母化 ----------------------

#[derive(Clone, Copy)]
struct Ev {
    t: i64,  // 距 9:30 的微秒（下午已前移）
    dir: u8, // 0 主买 1 主卖 2 撤买 3 撤卖
    vol: f64,
    amt: f64,
    oid: i64,
    px: f64, // 成交均价；撤单为 NaN
    lim: bool,
    dur: f64, // 只有母单用：持续秒数
}

/// records 已按 (时间, 序号) 排序。include_cxl=false 时完全不看撤单行（只用成交的序列与撤单数据无关）。
fn make_events(r: &[TradeRecord], include_cxl: bool) -> Vec<Ev> {
    let t0 = DAY_START_SEC * 1_000_000;
    let mut out: Vec<Ev> = Vec::with_capacity(r.len());
    for x in r.iter() {
        if x.volume <= 0.0 {
            continue;
        }
        let t = x.time_us.rem_euclid(86_400_000_000) - t0;
        match x.flag {
            66 | 83 => {
                let (dir, oid) = if x.flag == 66 { (0u8, x.bid_order) } else { (1u8, x.ask_order) };
                // 同一张主动单吃掉多个对手单：合并成一个事件
                if let Some(last) = out.last_mut() {
                    if last.dir == dir && last.oid == oid {
                        last.vol += x.volume;
                        last.amt += x.turnover;
                        last.px = last.amt / last.vol;
                        continue;
                    }
                }
                out.push(Ev { t, dir, vol: x.volume, amt: x.turnover, oid, px: x.turnover / x.volume, lim: false, dur: 0.0 });
            }
            32 if include_cxl => {
                // 撤单只看标记 32，不看价格（2025 年起沪市撤单价格非零）
                let (dir, oid) = if x.bid_order > 0 { (2u8, x.bid_order) } else { (3u8, x.ask_order) };
                if oid > 0 {
                    out.push(Ev { t, dir, vol: x.volume, amt: 0.0, oid, px: f64::NAN, lim: false, dur: 0.0 });
                }
            }
            _ => {}
        }
    }
    out
}

/// 涨跌停状态：以最近一笔成交价是否等于涨停价或跌停价判断（撤单沿用最近成交价）。
fn mark_limits(ev: &mut [Ev], up: f64, dn: f64) {
    let mut last = f64::NAN;
    for e in ev.iter_mut() {
        if e.px.is_finite() {
            last = e.px;
        }
        e.lim = last.is_finite() && ((up > 0.0 && last >= up - 1e-6) || (dn > 0.0 && last <= dn + 1e-6));
    }
}

/// 按分位切成 q 档（同值落同档，确定性）；非有限值落最高档。
fn qbucket(vals: &[f64], q: usize) -> Vec<u32> {
    let mut sorted: Vec<f64> = vals.iter().copied().filter(|v| v.is_finite()).collect();
    sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    if sorted.is_empty() {
        return vec![(q - 1) as u32; vals.len()];
    }
    let cuts: Vec<f64> = (1..q).map(|k| sorted[(k * sorted.len() / q).min(sorted.len() - 1)]).collect();
    vals.iter()
        .map(|v| if v.is_finite() { cuts.iter().filter(|c| *v > **c).count() as u32 } else { (q - 1) as u32 })
        .collect()
}

struct Seq {
    letters: Vec<u32>,
    k: usize,
    vol: Vec<f64>,
    buy: Vec<f64>,
    sell: Vec<f64>,
    t_s: Vec<f64>,
    px: Vec<f64>,
    tok: Vec<u64>, // 跨股票可比的原始记号（精确单量用原始手数，其余同字母）
}

fn compact(tokens: &[u64]) -> (Vec<u32>, usize) {
    let mut u = tokens.to_vec();
    u.sort_unstable();
    u.dedup();
    let letters = tokens.iter().map(|t| u.binary_search(t).unwrap() as u32).collect();
    (letters, u.len().max(1))
}

fn ffill(px: &mut [f64]) {
    let first = px.iter().copied().find(|v| v.is_finite()).unwrap_or(f64::NAN);
    let mut last = first;
    for v in px.iter_mut() {
        if v.is_finite() {
            last = *v;
        } else {
            *v = last;
        }
    }
}

fn gaps_of(ev: &[Ev]) -> Vec<f64> {
    (0..ev.len()).map(|i| if i == 0 { f64::NAN } else { (ev[i].t - ev[i - 1].t) as f64 / 1e6 }).collect()
}

/// 订单编号跳跃：主动单 = 与上一张主动单的编号差；撤单 = 被撤单相对已见最大编号的“年龄”。两类分别分档。
fn oid_buckets(ev: &[Ev], q: usize) -> Vec<u32> {
    let (mut prev, mut max_seen) = (0i64, 0i64);
    let (mut jump, mut is_cancel) = (Vec::with_capacity(ev.len()), Vec::with_capacity(ev.len()));
    for e in ev {
        if e.dir <= 1 {
            jump.push(if prev > 0 { (e.oid - prev).abs() as f64 } else { f64::NAN });
            prev = e.oid;
            is_cancel.push(false);
        } else {
            jump.push(if max_seen > 0 { (max_seen - e.oid).max(0) as f64 } else { f64::NAN });
            is_cancel.push(true);
        }
        max_seen = max_seen.max(e.oid);
    }
    let split = |flag: bool| -> Vec<f64> { jump.iter().zip(is_cancel.iter()).filter(|(_, c)| **c == flag).map(|(v, _)| *v).collect() };
    let (a, c) = (qbucket(&split(false), q), qbucket(&split(true), q));
    let (mut ia, mut ic) = (0, 0);
    is_cancel
        .iter()
        .map(|&f| {
            if f {
                ic += 1;
                c[ic - 1]
            } else {
                ia += 1;
                a[ia - 1]
            }
        })
        .collect()
}

fn seq_from_events(ev: &[Ev], letters: Vec<u32>, k: usize, tok: Option<Vec<u64>>) -> Seq {
    let mut px: Vec<f64> = ev.iter().map(|e| e.px).collect();
    ffill(&mut px);
    let tok = tok.unwrap_or_else(|| letters.iter().map(|&x| x as u64).collect());
    Seq {
        letters,
        k,
        vol: ev.iter().map(|e| e.vol).collect(),
        buy: ev.iter().map(|e| if e.dir == 0 { e.vol } else { 0.0 }).collect(),
        sell: ev.iter().map(|e| if e.dir == 1 { e.vol } else { 0.0 }).collect(),
        t_s: ev.iter().map(|e| e.t as f64 / 1e6).collect(),
        px,
        tok,
    }
}

/// 把连续相同字母折叠成一个（低复杂度屏蔽）。
fn run_compress(s: Seq) -> Seq {
    let mut o = Seq { letters: vec![], k: s.k, vol: vec![], buy: vec![], sell: vec![], t_s: vec![], px: vec![], tok: vec![] };
    for i in 0..s.letters.len() {
        if i > 0 && s.letters[i] == s.letters[i - 1] {
            *o.vol.last_mut().unwrap() += s.vol[i];
            *o.buy.last_mut().unwrap() += s.buy[i];
            *o.sell.last_mut().unwrap() += s.sell[i];
            *o.px.last_mut().unwrap() = s.px[i];
        } else {
            o.letters.push(s.letters[i]);
            o.vol.push(s.vol[i]);
            o.buy.push(s.buy[i]);
            o.sell.push(s.sell[i]);
            o.t_s.push(s.t_s[i]);
            o.px.push(s.px[i]);
            o.tok.push(s.tok[i]);
        }
    }
    o
}

struct EvCache {
    dir: Vec<u32>,
    size5: Vec<u32>,
    size3: Vec<u32>,
    gap5: Vec<u32>,
    gap3: Vec<u32>,
    oid5: Vec<u32>,
    oid3: Vec<u32>,
}

fn ev_cache(ev: &[Ev]) -> EvCache {
    let vols: Vec<f64> = ev.iter().map(|e| e.vol).collect();
    let gaps = gaps_of(ev);
    EvCache {
        dir: ev.iter().map(|e| e.dir as u32).collect(),
        size5: qbucket(&vols, 5),
        size3: qbucket(&vols, 3),
        gap5: qbucket(&gaps, 5),
        gap3: qbucket(&gaps, 3),
        oid5: oid_buckets(ev, 5),
        oid3: oid_buckets(ev, 3),
    }
}

fn zip(a: &[u32], ka: u32, b: &[u32]) -> Vec<u32> {
    a.iter().zip(b).map(|(x, y)| x * ka + y).collect()
}

fn size_gap_letters(c: &EvCache) -> Vec<u32> {
    let s = zip(&c.size3, 3, &c.gap3);
    zip(&c.dir, 9, &s)
}

fn build_event_seq(kind: Kind, ev: &[Ev], c: &EvCache) -> Option<Seq> {
    let (letters, k, tok): (Vec<u32>, usize, Option<Vec<u64>>) = match kind {
        Kind::Dir => (c.dir.clone(), 4, None),
        Kind::Size => (zip(&c.dir, 5, &c.size5), 20, None),
        Kind::Gap => (zip(&c.dir, 5, &c.gap5), 20, None),
        Kind::Oid => (zip(&c.dir, 5, &c.oid5), 20, None),
        Kind::SizeGap => (size_gap_letters(c), 36, None),
        Kind::SizeOid => {
            let s = zip(&c.size3, 3, &c.oid3);
            (zip(&c.dir, 9, &s), 36, None)
        }
        Kind::Lot => {
            let tok: Vec<u64> = ev.iter().map(|e| (e.dir as u64) << 40 | (e.vol.round() as u64)).collect();
            let (l, k) = compact(&tok);
            (l, k, Some(tok))
        }
        Kind::GapExact => {
            let g = gaps_of(ev);
            let sec: Vec<u32> = g.iter().map(|v| if v.is_finite() { v.round().clamp(0.0, 60.0) as u32 } else { 61 }).collect();
            (zip(&c.dir, 62, &sec), 248, None)
        }
        _ => return None,
    };
    Some(seq_from_events(ev, letters, k, tok))
}

/// 单量5档×间隔5档（子序列内部分档）。
fn size_gap25(sub: &[Ev]) -> Seq {
    let size = qbucket(&sub.iter().map(|e| e.vol).collect::<Vec<_>>(), 5);
    let gap = qbucket(&gaps_of(sub), 5);
    let letters: Vec<u32> = size.iter().zip(gap.iter()).map(|(s, g)| s * 5 + g).collect();
    seq_from_events(sub, letters, 25, None)
}

fn build_side_seq(side: u8, ev: &[Ev]) -> Seq {
    let sub: Vec<Ev> = ev.iter().copied().filter(|e| match side { 0 => e.dir == 0, 1 => e.dir == 1, _ => e.dir >= 2 }).collect();
    size_gap25(&sub)
}

fn build_limit_seq(at: bool, ev: &[Ev]) -> Option<Seq> {
    let sub: Vec<Ev> = ev.iter().copied().filter(|e| e.lim == at).collect();
    if sub.len() < MIN_N {
        return None;
    }
    let c = ev_cache(&sub);
    Some(seq_from_events(&sub, size_gap_letters(&c), 36, None))
}

/// 大单 = 本时段成交事件单量的 80% 分位及以上。
fn build_big_seq(which: u8, ev: &[Ev]) -> Option<Seq> {
    if ev.is_empty() {
        return None;
    }
    let mut v: Vec<f64> = ev.iter().map(|e| e.vol).collect();
    v.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    let thr = v[((v.len() as f64 * 0.8) as usize).min(v.len() - 1)];
    let sub: Vec<Ev> = ev.iter().copied().filter(|e| e.vol >= thr && (which == 0 || e.dir == which - 1)).collect();
    if sub.len() < MIN_N {
        return None;
    }
    if which == 0 {
        let size = qbucket(&sub.iter().map(|e| e.vol).collect::<Vec<_>>(), 3);
        let gap = qbucket(&gaps_of(&sub), 3);
        let letters: Vec<u32> = (0..sub.len()).map(|i| sub[i].dir as u32 * 9 + size[i] * 3 + gap[i]).collect();
        Some(seq_from_events(&sub, letters, 18, None))
    } else {
        Some(size_gap25(&sub))
    }
}

/// 价格档位变化：与上一笔成交相比变动几个最小价位（截到 ±3）× 方向。
fn build_pxstep_seq(ev: &[Ev]) -> Seq {
    let mut letters = Vec::with_capacity(ev.len());
    for i in 0..ev.len() {
        let d = if i == 0 { 0 } else { ((ev[i].px - ev[i - 1].px) / 0.01).round().clamp(-3.0, 3.0) as i32 };
        letters.push(ev[i].dir as u32 * 7 + (d + 3) as u32);
    }
    seq_from_events(ev, letters, 14, None)
}

/// 母单重建：同方向、间隔不超过 1 秒、价格变动不超过 1 个价位的连续主动单并成一张母单。
/// 字母 = 方向 × 母单量5档 × 持续时间3档。
fn build_parent_seq(ev: &[Ev]) -> Option<Seq> {
    let mut par: Vec<Ev> = Vec::new();
    let mut last_t = 0i64;
    let mut last_px = f64::NAN;
    for e in ev {
        if let Some(p) = par.last_mut() {
            if p.dir == e.dir && e.t - last_t <= 1_000_000 && (e.px - last_px).abs() <= 0.010_000_1 {
                p.vol += e.vol;
                p.amt += e.amt;
                p.px = p.amt / p.vol;
                p.dur = (e.t - p.t) as f64 / 1e6;
                last_t = e.t;
                last_px = e.px;
                continue;
            }
        }
        par.push(Ev { dur: 0.0, ..*e });
        last_t = e.t;
        last_px = e.px;
    }
    if par.len() < MIN_N {
        return None;
    }
    let v5 = qbucket(&par.iter().map(|e| e.vol).collect::<Vec<_>>(), 5);
    let d3 = qbucket(&par.iter().map(|e| e.dur).collect::<Vec<_>>(), 3);
    let letters: Vec<u32> = (0..par.len()).map(|i| par[i].dir as u32 * 15 + v5[i] * 3 + d3[i]).collect();
    Some(seq_from_events(&par, letters, 30, None))
}

struct BarStats {
    buy: Vec<f64>,
    sell: Vec<f64>,
    cancel: Vec<f64>,
    aggr: Vec<f64>,
    naggr: Vec<f64>,
    px: Vec<f64>,
}

fn bar_stats(ev: &[Ev], lo: i64, bar: i64, n: usize) -> BarStats {
    let mut s = BarStats { buy: vec![0.0; n], sell: vec![0.0; n], cancel: vec![0.0; n], aggr: vec![0.0; n], naggr: vec![0.0; n], px: vec![f64::NAN; n] };
    for e in ev {
        let b = ((e.t / 1_000_000 - lo) / bar) as usize;
        if b >= n {
            continue;
        }
        match e.dir {
            0 => s.buy[b] += e.vol,
            1 => s.sell[b] += e.vol,
            _ => s.cancel[b] += e.vol,
        }
        if e.dir <= 1 {
            s.aggr[b] += e.vol;
            s.naggr[b] += 1.0;
            s.px[b] = e.px;
        }
    }
    s
}

/// 每个时间格最后一个盘口快照（前向填充）。
fn bar_snapshots<'a>(mk: &'a [MarketRecord], lo: i64, hi: i64, bar: i64, n: usize) -> Vec<Option<&'a MarketRecord>> {
    let mut out: Vec<Option<&MarketRecord>> = vec![None; n];
    let t0 = DAY_START_SEC * 1_000_000;
    for m in mk {
        let sec = (m.time_us.rem_euclid(86_400_000_000) - t0) / 1_000_000;
        if sec < lo || sec >= hi {
            continue;
        }
        let b = ((sec - lo) / bar) as usize;
        if b < n {
            out[b] = Some(m);
        }
    }
    let mut last: Option<&MarketRecord> = out.iter().copied().flatten().next();
    for o in out.iter_mut() {
        if o.is_some() {
            last = *o;
        } else {
            *o = last;
        }
    }
    out
}

fn build_bar_seq(kind: Kind, ev: &[Ev], mk: &[MarketRecord], lo: i64, hi: i64) -> Option<Seq> {
    let bar = match kind {
        Kind::BarVolFlow(b) | Kind::BarCxl(b) | Kind::BarLob(b) | Kind::BarPxPath(b) | Kind::BarDepth(b) => b,
        _ => return None,
    };
    let n = ((hi - lo) / bar) as usize;
    let st = bar_stats(ev, lo, bar, n);
    let mut seqpx = st.px.clone();
    ffill(&mut seqpx);
    let vol = st.aggr.clone();
    let (letters, k): (Vec<u32>, usize) = match kind {
        Kind::BarVolFlow(_) => {
            let q = qbucket(&vol, 5);
            ((0..n).map(|i| {
                let net = st.buy[i] - st.sell[i];
                q[i] * 3 + if net > 0.0 { 2 } else if net < 0.0 { 0 } else { 1 }
            }).collect(), 15)
        }
        Kind::BarCxl(_) => {
            let cr: Vec<f64> = (0..n).map(|i| { let d = st.cancel[i] + vol[i]; if d > 0.0 { st.cancel[i] / d } else { 0.0 } }).collect();
            let az: Vec<f64> = (0..n).map(|i| if st.naggr[i] > 0.0 { vol[i] / st.naggr[i] } else { 0.0 }).collect();
            let (a, b) = (qbucket(&cr, 5), qbucket(&az, 5));
            (a.iter().zip(b.iter()).map(|(x, y)| x * 5 + y).collect(), 25)
        }
        Kind::BarPxPath(_) => {
            let v3 = qbucket(&vol, 3);
            ((0..n).map(|i| {
                let d = if i == 0 || !seqpx[i].is_finite() || !seqpx[i - 1].is_finite() { 0 } else { ((seqpx[i] - seqpx[i - 1]) / 0.01).round().clamp(-3.0, 3.0) as i32 };
                (d + 3) as u32 * 3 + v3[i]
            }).collect(), 21)
        }
        Kind::BarLob(_) | Kind::BarDepth(_) => {
            if mk.is_empty() {
                return None;
            }
            let snaps = bar_snapshots(mk, lo, hi, bar, n);
            if snaps.iter().all(|s| s.is_none()) {
                return None;
            }
            if let Kind::BarLob(_) = kind {
                let imb: Vec<f64> = snaps.iter().map(|s| s.map_or(f64::NAN, |m| { let d = m.total_bid_vol + m.total_ask_vol; if d > 0.0 { (m.total_bid_vol - m.total_ask_vol) / d } else { 0.0 } })).collect();
                let spr: Vec<f64> = snaps.iter().map(|s| s.map_or(f64::NAN, |m| { let (a1, b1) = (m.ask_prcs[0], m.bid_prcs[0]); if a1 > 0.0 && b1 > 0.0 { (a1 - b1) / ((a1 + b1) * 0.5) } else { f64::NAN } })).collect();
                let (a, b) = (qbucket(&imb, 5), qbucket(&spr, 3));
                (a.iter().zip(b.iter()).map(|(x, y)| x * 3 + y).collect(), 15)
            } else {
                // 十档形状：前五档挂单量占十档的比例，买卖各分 5 档
                let share = |v: &[f64; 10]| -> f64 { let t: f64 = v.iter().sum(); if t > 0.0 { v[..5].iter().sum::<f64>() / t } else { f64::NAN } };
                let sb: Vec<f64> = snaps.iter().map(|s| s.map_or(f64::NAN, |m| share(&m.bid_vols))).collect();
                let sa: Vec<f64> = snaps.iter().map(|s| s.map_or(f64::NAN, |m| share(&m.ask_vols))).collect();
                let (a, b) = (qbucket(&sb, 5), qbucket(&sa, 5));
                (a.iter().zip(b.iter()).map(|(x, y)| x * 5 + y).collect(), 25)
            }
        }
        _ => return None,
    };
    let t_s: Vec<f64> = (0..n).map(|i| (lo + i as i64 * bar) as f64).collect();
    let tok = letters.iter().map(|&x| x as u64).collect();
    Some(Seq { letters, k, vol, buy: st.buy, sell: st.sell, t_s, px: seqpx, tok })
}

// ---------------------- 第 3 层 特征 ----------------------

fn fnv64(xs: impl Iterator<Item = u64>) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for x in xs {
        h ^= x;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// 写入 27 个特征（xshare 两列留空，由横截面层填），返回本序列的模板记号（供跨股票比较）。
fn features(seq: &Seq, ao: &AlignOut, out: &mut [f32]) -> Vec<(u64, f32)> {
    let n = seq.letters.len();
    let (segs, cov) = select_nonoverlap(&ao.segs, n);
    let nf = n as f64;
    let total_vol: f64 = seq.vol.iter().sum();
    let mut f = [f64::NAN; N_F];
    let cnt_cov = cov.iter().filter(|c| **c).count() as f64;
    f[0] = cnt_cov / nf;
    let cov_vol: f64 = seq.vol.iter().zip(&cov).filter(|(_, c)| **c).map(|(v, _)| *v).sum();
    f[1] = if total_vol > 0.0 { cov_vol / total_vol } else { f64::NAN };
    f[2] = segs.len() as f64 / nf * 1000.0;
    f[5] = segs.iter().map(|s| s.score as f64).sum::<f64>() / nf;
    let (tb, ts): (f64, f64) = (seq.buy.iter().sum(), seq.sell.iter().sum());
    let cb: f64 = seq.buy.iter().zip(&cov).filter(|(_, c)| **c).map(|(v, _)| *v).sum();
    let cs: f64 = seq.sell.iter().zip(&cov).filter(|(_, c)| **c).map(|(v, _)| *v).sum();
    f[15] = if tb > 0.0 && ts > 0.0 { cb / tb - cs / ts } else { f64::NAN };
    f[6] = 0.0;
    let mut motifs = Vec::new();
    if !segs.is_empty() {
        let span = |s: &Seg| (s.ei - s.si + 1) as f64;
        let k = segs.len() as f64;
        let total_span: f64 = segs.iter().map(span).sum();
        f[3] = total_span / k;
        f[4] = segs.iter().map(span).fold(0.0, f64::max);
        let mut tand_mask = vec![false; n];
        let (mut copies, mut wc) = (0.0, 0.0);
        for s in &segs {
            let sp = span(s);
            if sp >= s.sl as f64 {
                for p in (s.si as usize)..=((s.ei + s.el) as usize).min(n - 1) {
                    tand_mask[p] = true;
                }
                copies += sp * (1.0 + sp / s.sl as f64);
                wc += sp;
            }
        }
        f[6] = tand_mask.iter().filter(|c| **c).count() as f64 / nf;
        f[7] = if wc > 0.0 { copies / wc } else { f64::NAN };
        let mut lag_w = vec![0.0f64; ao.max_lag + 2];
        let mut logsum = 0.0;
        for s in &segs {
            lag_w[s.sl as usize] += span(s);
            logsum += span(s) * (s.sl as f64).ln();
        }
        f[8] = lag_w.iter().copied().fold(0.0, f64::max) / total_span;
        f[9] = logsum / total_span;
        let ent: f64 = lag_w.iter().filter(|w| **w > 0.0).map(|w| { let p = w / total_span; -p * p.ln() }).sum();
        f[10] = if ao.max_lag > 1 { ent / (ao.max_lag as f64).ln() } else { f64::NAN };
        let mut tg: Vec<f64> = segs.iter().filter_map(|s| { let j = (s.si + s.sl) as usize; if j < n { Some(seq.t_s[j] - seq.t_s[s.si as usize]) } else { None } }).collect();
        if !tg.is_empty() {
            tg.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
            f[11] = (1.0 + tg[tg.len() / 2]).ln();
            let m = tg.iter().sum::<f64>() / tg.len() as f64;
            let v = tg.iter().map(|x| (x - m).powi(2)).sum::<f64>() / tg.len() as f64;
            f[12] = if m > 0.0 { v.sqrt() / m } else { f64::NAN };
        }
        let (ma, mi, ga): (f64, f64, f64) = segs.iter().fold((0.0, 0.0, 0.0), |a, s| (a.0 + s.matches as f64, a.1 + s.mism as f64, a.2 + s.gaps as f64));
        let col = ma + mi + ga;
        if col > 0.0 {
            f[13] = ga / col;
            f[14] = mi / col;
        }
        if cnt_cov > 0.0 {
            let mut c = 0.0;
            let mut bins = [0.0f64; 10];
            for (i, cv) in cov.iter().enumerate() {
                if *cv {
                    let r = i as f64 / (n.max(2) - 1) as f64;
                    c += r;
                    bins[((r * 10.0) as usize).min(9)] += 1.0;
                }
            }
            f[16] = c / cnt_cov;
            f[17] = bins.iter().copied().fold(0.0, f64::max) / cnt_cov;
            let outside = nf - cnt_cov;
            let outside_vol = total_vol - cov_vol;
            if outside > 0.0 && outside_vol > 0.0 && cov_vol > 0.0 {
                f[18] = (cov_vol / cnt_cov) / (outside_vol / outside);
            }
            let (mut imp_in, mut imp_out) = (0.0f64, 0.0f64);
            for i in 1..n {
                if seq.px[i] > 0.0 && seq.px[i - 1] > 0.0 {
                    let d = (seq.px[i] / seq.px[i - 1]).ln().abs();
                    if cov[i] { imp_in += d } else { imp_out += d }
                }
            }
            if outside_vol > 0.0 && cov_vol > 0.0 && imp_out > 0.0 {
                f[19] = (imp_in / cov_vol) / (imp_out / outside_vol);
            }
        }
        let mut hs: Vec<(u64, f64)> = segs.iter().map(|s| (fnv64(seq.letters[(s.si as usize)..=(s.ei as usize)].iter().map(|&x| x as u64)), span(s))).collect();
        hs.sort_unstable_by(|a, b| a.0.cmp(&b.0));
        let (mut distinct, mut best_sh, mut i) = (0usize, 0.0f64, 0usize);
        while i < hs.len() {
            let (mut j, mut sh) = (i, 0.0);
            while j < hs.len() && hs[j].0 == hs[i].0 {
                sh += hs[j].1;
                j += 1;
            }
            distinct += 1;
            best_sh = best_sh.max(sh);
            i = j;
        }
        f[20] = distinct as f64 / k;
        f[21] = best_sh / total_span;

        // 多重比对共同模板：取最高分片段，按周期把各份拷贝叠起来逐列取多数字母
        let top = segs[0];
        let p = (top.sl as usize).max(1);
        let (r0, r1) = (top.si as usize, ((top.ei + top.el) as usize).min(n - 1));
        let len = r1 - r0 + 1;
        let mut cols: Vec<(usize, u32)> = (r0..=r1).map(|pos| ((pos - r0) % p, seq.letters[pos])).collect();
        cols.sort_unstable();
        let mut agree = 0usize;
        let mut i = 0;
        while i < cols.len() {
            let c0 = cols[i].0;
            let mut best = 0usize;
            while i < cols.len() && cols[i].0 == c0 {
                let mut j = i;
                while j < cols.len() && cols[j] == cols[i] {
                    j += 1;
                }
                best = best.max(j - i);
                i = j;
            }
            agree += best;
        }
        f[22] = agree as f64 / len as f64;
        f[23] = (p as f64).ln();
        f[24] = len as f64 / p as f64;

        // 跨股票模板：每个片段的重复单元（前 min(周期,16) 个原始记号）
        for s in segs.iter().take(MOTIF_CAP) {
            let si = s.si as usize;
            let u = (s.sl as usize).min(MOTIF_UNIT_MAX).min(n - si);
            let h = fnv64(std::iter::once(u as u64).chain(seq.tok[si..si + u].iter().copied()));
            motifs.push((h, span(s) as f32));
        }
    }
    for (o, v) in out.iter_mut().zip(f.iter()) {
        *o = *v as f32;
    }
    motifs
}

fn baselines(seq: &Seq, ao: &AlignOut, out: &mut [f32]) {
    let n = seq.letters.len();
    let mut b = [f64::NAN; N_B];
    if ao.max_lag >= 1 {
        b[0] = ao.lag_hits[1] as f64 / (n - 1) as f64;
        let mut best = f64::NEG_INFINITY;
        for lag in 1..=ao.max_lag {
            let d = (n - lag) as f64;
            if d >= 8.0 {
                best = best.max(ao.lag_hits[lag] as f64 / d - ao.pp);
            }
        }
        if best.is_finite() {
            b[1] = best;
        }
    }
    // LZ78：字典大小 × log_k(n) / n；哈希表只查找不遍历
    let mut dict: HashMap<(u32, u32), u32> = HashMap::with_capacity(n / 4);
    let (mut cur, mut sz) = (0u32, 1u32);
    for &x in &seq.letters {
        if let Some(&id) = dict.get(&(cur, x)) {
            cur = id;
        } else {
            dict.insert((cur, x), sz);
            sz += 1;
            cur = 0;
        }
    }
    let mut used = vec![0u32; seq.k];
    for &x in &seq.letters {
        used[x as usize] += 1;
    }
    let kk = used.iter().filter(|c| **c > 0).count();
    if kk > 1 {
        b[2] = (sz as f64 - 1.0) * (n as f64).ln() / (kk as f64).ln() / n as f64;
        let ent: f64 = used.iter().filter(|c| **c > 0).map(|c| { let p = *c as f64 / n as f64; -p * p.ln() }).sum();
        b[4] = ent / (kk as f64).ln();
    }
    let runs = 1 + seq.letters.windows(2).filter(|w| w[0] != w[1]).count();
    b[3] = n as f64 / runs as f64;
    for (o, v) in out.iter_mut().zip(b.iter()) {
        *o = *v as f32;
    }
}

// ---------------------- 单股单日 ----------------------

struct StockOut {
    row: Vec<f32>,
    n_trade: [f32; N_WIN],
    n_mixed: [f32; N_WIN],
    motifs: Vec<Vec<(u64, f32)>>,
}

fn run_align(seq: &Seq, sp: &Spec, fast: bool) -> Option<AlignOut> {
    let lag = if sp.max_lag == 0 { seq.letters.len() / 2 } else { sp.max_lag };
    PROF_CELLS.fetch_add((seq.letters.len() * lag.min(seq.letters.len().saturating_sub(1))) as u64, Relaxed);
    match sp.gap {
        GapMode::Affine => self_align_affine(&seq.letters, seq.k, lag, sp.q, SEG_CAP, true, fast),
        GapMode::Linear(m) => {
            if fast {
                self_align_fast(&seq.letters, seq.k, lag, sp.q, m, SEG_CAP)
            } else {
                self_align_ref(&seq.letters, seq.k, lag, sp.q, m, SEG_CAP, true)
            }
        }
    }
}

fn per_stock(code: &str, date: i64, sp: &[Spec], fast: bool) -> Option<StockOut> {
    let t_read = Instant::now();
    let mut recs = read_trade_fast(code, date).ok()?;
    let mk = read_market_fast(code, date).unwrap_or_default();
    PROF_READ.fetch_add(t_read.elapsed().as_micros() as u64, Relaxed);
    let t_ev = Instant::now();
    recs.sort_by_key(|x| (x.time_us, x.index));
    let cxl_ok = cxl_available(code, date);
    let mut ev_t = make_events(&recs, false);
    if ev_t.is_empty() {
        return None;
    }
    let mut ev_m = if cxl_ok { make_events(&recs, true) } else { Vec::new() };
    let up = mk.iter().map(|m| m.high_limited).find(|v| *v > 0.0).unwrap_or(f64::NAN);
    let dn = mk.iter().map(|m| m.low_limited).find(|v| *v > 0.0).unwrap_or(f64::NAN);
    mark_limits(&mut ev_t, up, dn);
    mark_limits(&mut ev_m, up, dn);
    PROF_LETTER.fetch_add(t_ev.elapsed().as_micros() as u64, Relaxed);

    let ns = sp.len();
    let mut out = StockOut { row: vec![f32::NAN; ns * N_WIN * N_J], n_trade: [f32::NAN; N_WIN], n_mixed: [f32::NAN; N_WIN], motifs: vec![Vec::new(); ns * N_WIN] };
    for (w, &(_, lo, hi)) in WINDOWS.iter().enumerate() {
        let inw = |e: &&Ev| e.t >= lo * 1_000_000 && e.t < hi * 1_000_000;
        let wt: Vec<Ev> = ev_t.iter().filter(inw).copied().collect();
        let wm: Vec<Ev> = ev_m.iter().filter(inw).copied().collect();
        out.n_trade[w] = wt.len() as f32;
        if cxl_ok {
            out.n_mixed[w] = wm.len() as f32;
        }
        if wt.len() < MIN_N {
            continue;
        }
        let t_c = Instant::now();
        let ct = ev_cache(&wt);
        let cm = if cxl_ok && wm.len() >= MIN_N { Some(ev_cache(&wm)) } else { None };
        PROF_LETTER.fetch_add(t_c.elapsed().as_micros() as u64, Relaxed);
        for (s, spec) in sp.iter().enumerate() {
            if spec.cxl && !cxl_ok {
                continue; // 撤单数据开始之前：cxl 列保持空值
            }
            let t_l = Instant::now();
            let (list, cache) = match spec.src {
                Src::Trade => (&wt, Some(&ct)),
                Src::Mixed => (&wm, cm.as_ref()),
            };
            let seq = match spec.kind {
                Kind::Side(side) => {
                    if spec.src == Src::Mixed && cache.is_none() {
                        None
                    } else {
                        Some(build_side_seq(side, list))
                    }
                }
                Kind::Limit(at) => {
                    if cache.is_none() { None } else { build_limit_seq(at, list) }
                }
                Kind::Big(which) => build_big_seq(which, list),
                Kind::PxStep => Some(build_pxstep_seq(list)),
                Kind::Parent => build_parent_seq(list),
                Kind::BarVolFlow(_) | Kind::BarCxl(_) | Kind::BarLob(_) | Kind::BarPxPath(_) | Kind::BarDepth(_) => {
                    if spec.src == Src::Mixed && cache.is_none() { None } else { build_bar_seq(spec.kind, list, &mk, lo, hi) }
                }
                _ => cache.and_then(|c| build_event_seq(spec.kind, list, c)),
            };
            let Some(mut seq) = seq else { continue };
            if spec.compress {
                seq = run_compress(seq);
            }
            let is_bar = matches!(spec.kind, Kind::BarVolFlow(_) | Kind::BarCxl(_) | Kind::BarLob(_) | Kind::BarPxPath(_) | Kind::BarDepth(_));
            PROF_LETTER.fetch_add(t_l.elapsed().as_micros() as u64, Relaxed);
            if seq.letters.len() < if is_bar { 20 } else { MIN_N } {
                continue;
            }
            let t_d = Instant::now();
            let ao = run_align(&seq, spec, fast);
            let us = t_d.elapsed().as_micros() as u64;
            PROF_DP.fetch_add(us, Relaxed);
            PROF_SPEC_US[s].fetch_add(us, Relaxed);
            let lag = if spec.max_lag == 0 { seq.letters.len() / 2 } else { spec.max_lag };
            PROF_SPEC_CELLS[s].fetch_add((seq.letters.len() * lag.min(seq.letters.len().saturating_sub(1))) as u64, Relaxed);
            let Some(ao) = ao else { continue };
            PROF_SPEC_EMIT[s].fetch_add(ao.n_emitted, Relaxed);
            let t_f = Instant::now();
            let base = raw_idx(s, w, 0);
            out.motifs[s * N_WIN + w] = features(&seq, &ao, &mut out.row[base..base + N_F]);
            baselines(&seq, &ao, &mut out.row[base + N_F..base + N_J]);
            PROF_FEAT.fetch_add(t_f.elapsed().as_micros() as u64, Relaxed);
        }
    }
    Some(out)
}

// ---------------------- 第 4 层 横截面 ----------------------

/// 同值取平均名次，映射到 0~1，空值保持空。
fn rank01(y: &[f32]) -> Vec<f32> {
    let mut idx: Vec<usize> = (0..y.len()).filter(|&i| y[i].is_finite()).collect();
    let mut out = vec![f32::NAN; y.len()];
    if idx.len() < 2 {
        return out;
    }
    idx.sort_unstable_by(|&a, &b| y[a].partial_cmp(&y[b]).unwrap().then(a.cmp(&b)));
    let m = idx.len();
    let mut i = 0;
    while i < m {
        let mut j = i;
        while j + 1 < m && y[idx[j + 1]] == y[idx[i]] {
            j += 1;
        }
        let r = (i + j) as f32 * 0.5 / (m - 1) as f32;
        for k in i..=j {
            out[idx[k]] = r;
        }
        i = j + 1;
    }
    out
}

/// 按 x 从小到大分 bins 组，每组内部排名到 0~1（x 为空的股票输出空值）。
fn rank_in_bins(y: &[f32], x: &[f32], bins: usize) -> Vec<f32> {
    let mut idx: Vec<usize> = (0..y.len()).filter(|&i| x[i].is_finite()).collect();
    let mut out = vec![f32::NAN; y.len()];
    idx.sort_unstable_by(|&a, &b| x[a].partial_cmp(&x[b]).unwrap().then(a.cmp(&b)));
    let m = idx.len();
    for b in 0..bins {
        let chunk = &idx[b * m / bins..(b + 1) * m / bins];
        let sub: Vec<f32> = chunk.iter().map(|&i| y[i]).collect();
        for (&i, &v) in chunk.iter().zip(rank01(&sub).iter()) {
            out[i] = v;
        }
    }
    out
}

/// 行业内排名（行业编号为空的股票输出空值）。
fn rank_in_groups(y: &[f32], g: &[f32]) -> Vec<f32> {
    let mut idx: Vec<usize> = (0..y.len()).filter(|&i| g[i].is_finite()).collect();
    let mut out = vec![f32::NAN; y.len()];
    idx.sort_unstable_by(|&a, &b| g[a].partial_cmp(&g[b]).unwrap().then(a.cmp(&b)));
    let mut i = 0;
    while i < idx.len() {
        let mut j = i;
        while j < idx.len() && g[idx[j]] == g[idx[i]] {
            j += 1;
        }
        let chunk = &idx[i..j];
        let sub: Vec<f32> = chunk.iter().map(|&k| y[k]).collect();
        for (&k, &v) in chunk.iter().zip(rank01(&sub).iter()) {
            out[k] = v;
        }
        i = j;
    }
    out
}

struct Day {
    date: i64,
    codes: Vec<String>,
    outs: Vec<StockOut>,
}

/// 同一模板多股重现：同序列同时段内，一只股票的模板有多少其他股票也出现过。
fn fill_cross_stock(day: &mut Day, ns: usize) {
    let m = day.codes.len();
    let results: Vec<Vec<(usize, f32, f32)>> = (0..ns * N_WIN)
        .into_par_iter()
        .map(|sw| {
            let mut pairs: Vec<(u64, u32)> = Vec::new();
            for i in 0..m {
                let mut hs: Vec<u64> = day.outs[i].motifs[sw].iter().map(|x| x.0).collect();
                hs.sort_unstable();
                hs.dedup();
                pairs.extend(hs.into_iter().map(|h| (h, i as u32)));
            }
            pairs.sort_unstable();
            let mut uniq: Vec<(u64, u32)> = Vec::new();
            let mut i = 0;
            while i < pairs.len() {
                let mut j = i;
                while j < pairs.len() && pairs[j].0 == pairs[i].0 {
                    j += 1;
                }
                uniq.push((pairs[i].0, (j - i) as u32));
                i = j;
            }
            let mut res = Vec::new();
            for i in 0..m {
                let mo = &day.outs[i].motifs[sw];
                if mo.is_empty() {
                    continue;
                }
                let (mut wsh, mut wall, mut mx) = (0.0f64, 0.0f64, 0u32);
                for &(h, w) in mo {
                    let cnt = uniq[uniq.binary_search_by(|x| x.0.cmp(&h)).unwrap()].1;
                    wall += w as f64;
                    if cnt >= 2 {
                        wsh += w as f64;
                    }
                    mx = mx.max(cnt - 1);
                }
                res.push((i, (wsh / wall) as f32, (1.0 + mx as f64).ln() as f32));
            }
            res
        })
        .collect();
    for (sw, res) in results.into_iter().enumerate() {
        let (s, w) = (sw / N_WIN, sw % N_WIN);
        for (i, a, b) in res {
            day.outs[i].row[raw_idx(s, w, F_XFRAC)] = a;
            day.outs[i].row[raw_idx(s, w, F_XLOG)] = b;
        }
    }
}

fn raw_col(day: &Day, j: usize) -> Vec<f32> {
    day.outs.iter().map(|o| o.row[j]).collect()
}

fn act_ranks(day: &Day, n_raw: usize) -> Vec<Vec<f32>> {
    let x: Vec<Vec<f32>> = (0..N_WIN).map(|w| day.outs.iter().map(|o| (1.0 + o.n_trade[w]).ln()).collect()).collect();
    (0..n_raw).into_par_iter().map(|j| rank_in_bins(&raw_col(day, j), &x[(j / N_J) % N_WIN], 20)).collect()
}

// ---------------------- 输入：股票表、日期、日频数据 ----------------------

fn level2_root() -> String {
    std::env::var("RUST_PYFUNC_LEVEL2_PATH").unwrap_or("/ssd_data/stock".into())
}

fn symbol_list() -> std::io::Result<(Vec<String>, String, u64)> {
    use std::io::{Error, ErrorKind};
    let invalid = |e: csv::Error| Error::new(ErrorKind::InvalidData, e);
    let basic = std::env::var("RUST_PYFUNC_BASIC_INFO_DIR").unwrap_or("/ssd_data/data/basic_info".into());
    let path = std::path::Path::new(&basic).join("symbol_map.csv");
    let bytes = std::fs::read(&path)?;
    let hash = fnv64(bytes.iter().map(|b| *b as u64));
    let mut reader = csv::Reader::from_reader(bytes.as_slice());
    let column = reader.headers().map_err(invalid)?.iter().position(|h| h == "symbol").ok_or_else(|| Error::new(ErrorKind::InvalidData, "股票表缺少 symbol 列"))?;
    let mut set = std::collections::BTreeSet::new();
    for rec in reader.records() {
        let rec = rec.map_err(invalid)?;
        let code = rec.get(column).unwrap_or("").trim();
        if code.is_empty() || code.len() > 6 || !code.bytes().all(|b| b.is_ascii_digit()) {
            return Err(Error::new(ErrorKind::InvalidData, "股票表含无效代码"));
        }
        set.insert(format!("{code:0>6}"));
    }
    if set.is_empty() {
        return Err(Error::new(ErrorKind::InvalidData, "股票白名单为空"));
    }
    Ok((set.into_iter().collect(), path.display().to_string(), hash))
}

fn codes_on(all: &[String], date: i64) -> Vec<String> {
    let dir = std::path::Path::new(&level2_root()).join(date.to_string()).join("transaction");
    all.iter().filter(|c| dir.join(format!("{c}_{date}_transaction.csv")).is_file()).cloned().collect()
}

/// 原始数据目录里比 date 早的最近一个有逐笔成交的日期（不落盘、不用缓存）。
fn prev_day(date: i64) -> Option<i64> {
    let root = level2_root();
    let mut best: Option<i64> = None;
    for e in std::fs::read_dir(&root).ok()?.flatten() {
        if let Some(d) = e.file_name().to_str().and_then(|s| s.parse::<i64>().ok()) {
            if d < date && d > 19_000_000 && e.path().join("transaction").is_dir() {
                best = Some(best.map_or(d, |b: i64| b.max(d)));
            }
        }
    }
    best
}

/// 读日频 h5 的一行，按代码对齐。exact=true 时日历上必须正好有这一天，否则全空。
fn h5_row(dir: &str, field: &str, date: i64, exact: bool, codes: &[String]) -> Vec<f32> {
    let base = format!("/ssd_data/data/vars/{dir}");
    let cal: Vec<i64> = std::fs::read_to_string(format!("{base}/calendar_map.csv")).expect("读日历失败").lines().skip(1).filter_map(|l| l.trim().parse().ok()).collect();
    let sym: Vec<String> = std::fs::read_to_string(format!("{base}/symbol_map.csv")).expect("读代码表失败").lines().skip(1).map(|l| format!("{:0>6}", l.split(',').next().unwrap().trim())).collect();
    let r = cal.partition_point(|&d| d <= date);
    if r == 0 || (exact && cal[r - 1] != date) {
        return vec![f32::NAN; codes.len()];
    }
    let file = hdf5::File::open(format!("{base}/{field}.h5")).expect("打开 h5 失败");
    let row: ndarray::Array2<f64> = file.dataset("data").unwrap().read_slice_2d(ndarray::s![r - 1..r, ..sym.len()]).unwrap();
    let pos: HashMap<&str, usize> = sym.iter().enumerate().map(|(i, s)| (s.as_str(), i)).collect();
    codes.iter().map(|c| pos.get(c.as_str()).map_or(f32::NAN, |&p| row[[0, p]] as f32)).collect()
}

#[derive(Serialize)]
struct SpecMeta {
    name: String,
    cxl: bool,
    src: &'static str,
}

#[derive(Serialize)]
struct Meta {
    date: i64,
    prev_date: Option<i64>,
    layout: &'static str,
    n_codes: usize,
    n_cols: usize,
    windows: Vec<(String, i64, i64)>,
    sh_cxl_start: i64,
    sz_cxl_start: i64,
    symbol_map: String,
    symbol_map_fnv: String,
    cap_source: String,
    industry_source: String,
    specs: Vec<SpecMeta>,
    names: Vec<String>,
    codes: Vec<String>,
    cxl_available: Vec<bool>,
    n_trade: Vec<[f32; N_WIN]>,
    n_mixed: Vec<[f32; N_WIN]>,
    prev_cap: Vec<f32>,
    industry: Vec<f32>,
    dropped_today: usize,
    dropped_prev: usize,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let date: i64 = args.get(1).expect("用法: align_v3 <date> --out <prefix> [--ref]").parse().expect("日期须为 8 位整数");
    let prefix = args.iter().position(|a| a == "--out").and_then(|i| args.get(i + 1).cloned()).expect("缺少 --out");
    let fast = !args.iter().any(|a| a == "--ref");
    rayon::ThreadPoolBuilder::new().num_threads(THREADS).build_global().unwrap();
    let sp = specs();
    let ns = sp.len();
    let n_raw = ns * N_WIN * N_J;

    let t0 = Instant::now();
    let (all, symbol_map, hash) = symbol_list().expect("股票范围检查失败");
    let root = level2_root();
    if !std::path::Path::new(&root).join(date.to_string()).join("transaction").is_dir() {
        panic!("原始目录不存在: {root}/{date}/transaction");
    }
    let pdate = prev_day(date);
    let days: Vec<(i64, Vec<String>)> = std::iter::once((date, codes_on(&all, date))).chain(pdate.map(|p| (p, codes_on(&all, p)))).collect();
    eprintln!("date={date} prev={pdate:?} codes={} prev_codes={} threads={THREADS} mode={}", days[0].1.len(), days.get(1).map_or(0, |d| d.1.len()), if fast { "fast" } else { "ref" });

    // 两天的单股任务放在一起调度；提速模式下按文件大小从大到小排（结果按原位置放回，顺序无关）
    let mut tasks: Vec<(usize, usize)> = days.iter().enumerate().flat_map(|(d, (_, cs))| (0..cs.len()).map(move |i| (d, i))).collect();
    if fast {
        let size = |&(d, i): &(usize, usize)| -> u64 {
            let (dt, cs) = &days[d];
            std::fs::metadata(format!("{root}/{dt}/transaction/{}_{dt}_transaction.csv", cs[i])).map_or(0, |m| m.len())
        };
        let mut sized: Vec<(u64, (usize, usize))> = tasks.iter().map(|t| (size(t), *t)).collect();
        sized.sort_unstable_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
        tasks = sized.into_iter().map(|x| x.1).collect();
    }
    // 提速模式：每只股票单独成为一个可被窃取的任务（with_max_len(1)）。否则并行库会把相邻任务
    // 切成整块交给同一线程串行处理，按大小排序后最大的一批股票会挤在同一块里（实测墙钟翻倍）。
    let results: Vec<Option<StockOut>> = if fast {
        tasks.par_iter().with_max_len(1).map(|&(d, i)| per_stock(&days[d].1[i], days[d].0, &sp, fast)).collect()
    } else {
        tasks.par_iter().map(|&(d, i)| per_stock(&days[d].1[i], days[d].0, &sp, fast)).collect()
    };
    let mut slots: Vec<Vec<Option<StockOut>>> = days.iter().map(|(_, cs)| (0..cs.len()).map(|_| None).collect()).collect();
    for ((d, i), r) in tasks.into_iter().zip(results.into_iter()) {
        slots[d][i] = r;
    }
    let mut dropped = vec![0usize; days.len()];
    let mut dayv: Vec<Day> = Vec::new();
    for (d, sl) in slots.into_iter().enumerate() {
        let mut codes = Vec::new();
        let mut outs = Vec::new();
        for (c, o) in days[d].1.iter().zip(sl.into_iter()) {
            match o {
                Some(o) => {
                    codes.push(c.clone());
                    outs.push(o);
                }
                None => dropped[d] += 1,
            }
        }
        dayv.push(Day { date: days[d].0, codes, outs });
    }
    let t_ps = t0.elapsed().as_secs_f64();
    eprintln!("[profile] per-stock {:.1}s  thread-seconds read={:.0} letters={:.0} dp={:.0} features={:.0} cells={:.3e}", t_ps, PROF_READ.load(Relaxed) as f64 / 1e6, PROF_LETTER.load(Relaxed) as f64 / 1e6, PROF_DP.load(Relaxed) as f64 / 1e6, PROF_FEAT.load(Relaxed) as f64 / 1e6, PROF_CELLS.load(Relaxed) as f64);

    eprintln!("[profile] emitted segments={:.3e} finish(sort+dedup)={:.1}s", align::PROF_EMIT.load(Relaxed) as f64, align::PROF_FINISH_NS.load(Relaxed) as f64 / 1e9);
    if std::env::var("ALIGN_PROFILE_SPECS").is_ok() {
        let mut v: Vec<(u64, usize)> = (0..ns).map(|s| (PROF_SPEC_US[s].load(Relaxed), s)).collect();
        v.sort_unstable_by(|a, b| b.cmp(a));
        for (us, s) in v {
            let c = PROF_SPEC_CELLS[s].load(Relaxed) as f64;
            eprintln!("[spec] {:<24} {:>8.1}s  cells={:.2e}  ns/cell={:.2}  emitted={:.2e}", sp[s].name, us as f64 / 1e6, c, if c > 0.0 { us as f64 * 1e3 / c } else { 0.0 }, PROF_SPEC_EMIT[s].load(Relaxed) as f64);
        }
    }

    // 第 4 层
    let t1 = Instant::now();
    for d in dayv.iter_mut() {
        fill_cross_stock(d, ns);
    }
    let today = &dayv[0];
    let m = today.codes.len();
    let act = act_ranks(today, n_raw);
    let prev_cap = pdate.map_or(vec![f32::NAN; m], |p| h5_row("DailyDerived", "S_DQ_MV", p, true, &today.codes));
    let industry = h5_row("SzBa", "industry", date, false, &today.codes);
    let lcap: Vec<f32> = prev_cap.iter().map(|v| if *v > 0.0 { v.ln() } else { f32::NAN }).collect();
    let capr: Vec<Vec<f32>> = (0..n_raw).into_par_iter().map(|j| rank_in_bins(&raw_col(today, j), &lcap, 20)).collect();
    let indr: Vec<Vec<f32>> = (0..n_raw).into_par_iter().map(|j| rank_in_groups(&raw_col(today, j), &industry)).collect();
    // 跨日：今天的活跃度组内排名 − 前一交易日的（前一日按它自己的全市场重新计算）
    let act_prev: Option<(Vec<Vec<f32>>, Vec<Option<usize>>)> = dayv.get(1).map(|p| {
        let a = act_ranks(p, n_raw);
        let map: Vec<Option<usize>> = today.codes.iter().map(|c| p.codes.binary_search(c).ok()).collect();
        (a, map)
    });
    eprintln!("cross-section {:.1}s", t1.elapsed().as_secs_f64());

    // 组装列名与写盘（按列存放）
    let t2 = Instant::now();
    let mut names: Vec<String> = Vec::new();
    let base_name = |s: usize, w: &str, j: usize| -> String {
        let f = if j < N_F { FEATS[j].to_string() } else { format!("base_{}", BASES[j - N_F]) };
        format!("{}__{}__{}", sp[s].name, w, f)
    };
    let file = std::fs::File::create(format!("{prefix}.f32")).expect("创建输出失败");
    let mut wr = std::io::BufWriter::with_capacity(1 << 24, file);
    let mut put = |col: &[f32], wr: &mut std::io::BufWriter<std::fs::File>| {
        let mut buf = Vec::with_capacity(col.len() * 4);
        for v in col {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        wr.write_all(&buf).unwrap();
    };
    let idx3 = |j: usize| -> (usize, usize, usize) { (j / (N_WIN * N_J), (j / N_J) % N_WIN, j % N_J) };
    for j in 0..n_raw {
        let (s, w, k) = idx3(j);
        names.push(base_name(s, WINDOWS[w].0, k));
        put(&raw_col(today, j), &mut wr);
    }
    for (suffix, mat) in [("nrank_act", &act), ("nrank_cap", &capr), ("rank_ind", &indr)] {
        for j in 0..n_raw {
            let (s, w, k) = idx3(j);
            names.push(format!("{}__{suffix}", base_name(s, WINDOWS[w].0, k)));
            put(&mat[j], &mut wr);
        }
    }
    for (dname, a, b) in [("pm_minus_full", 1usize, 0usize), ("tail_minus_full", 2, 0), ("tail_minus_pm", 2, 1)] {
        for s in 0..ns {
            for k in 0..N_J {
                names.push(format!("{}__nrank_act", base_name(s, dname, k)));
                let (ca, cb) = (&act[raw_idx(s, a, k)], &act[raw_idx(s, b, k)]);
                let d: Vec<f32> = (0..m).map(|i| ca[i] - cb[i]).collect();
                put(&d, &mut wr);
            }
        }
    }
    for j in 0..n_raw {
        let (s, w, k) = idx3(j);
        names.push(format!("{}__nrank_act__d1", base_name(s, WINDOWS[w].0, k)));
        let d: Vec<f32> = match &act_prev {
            Some((ap, map)) => (0..m).map(|i| map[i].map_or(f32::NAN, |p| act[j][i] - ap[j][p])).collect(),
            None => vec![f32::NAN; m],
        };
        put(&d, &mut wr);
    }
    wr.flush().unwrap();
    let n_cols = names.len();
    let meta = Meta {
        date,
        prev_date: pdate,
        layout: "column-major f32 little-endian: n_cols blocks of n_codes values",
        n_codes: m,
        n_cols,
        windows: WINDOWS.iter().map(|w| (w.0.to_string(), w.1, w.2)).collect(),
        sh_cxl_start: SH_CXL_START,
        sz_cxl_start: SZ_CXL_START,
        symbol_map,
        symbol_map_fnv: format!("{hash:016x}"),
        cap_source: format!("DailyDerived/S_DQ_MV @ {pdate:?}"),
        industry_source: format!("SzBa/industry <= {date}"),
        specs: sp.iter().map(|s| SpecMeta { name: s.name.clone(), cxl: s.cxl, src: if s.src == Src::Mixed { "mixed" } else { "trade" } }).collect(),
        names,
        codes: today.codes.clone(),
        cxl_available: today.codes.iter().map(|c| cxl_available(c, date)).collect(),
        n_trade: today.outs.iter().map(|o| o.n_trade).collect(),
        n_mixed: today.outs.iter().map(|o| o.n_mixed).collect(),
        prev_cap,
        industry,
        dropped_today: dropped[0],
        dropped_prev: dropped.get(1).copied().unwrap_or(0),
    };
    let _ = dayv.iter().map(|d| d.date).count();
    serde_json::to_writer(std::io::BufWriter::new(std::fs::File::create(format!("{prefix}.meta.json")).unwrap()), &meta).unwrap();
    eprintln!("columns = {n_cols} codes = {m} write {:.1}s total {:.1}s", t2.elapsed().as_secs_f64(), t0.elapsed().as_secs_f64());
}
