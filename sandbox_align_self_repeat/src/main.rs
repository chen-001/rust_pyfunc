//! 方向 2「自比对找重复」单日全市场探索（sandbox）。
//!
//! 第 1 层 字母化：逐笔（成交按主动单合并 + 撤单）或 15/60 秒时间格 → 字母序列
//! 第 2 层 自比对：带宽限制的局部比对动态规划（align.rs）
//! 第 3 层 重复片段特征：22 个特征 + 5 个朴素对照
//! 第 4 层 横截面合成：活跃度分组内排名、按序列组合成、尾盘/下午相对全天
//!
//! 时段（前移后，从 9:30 起的秒）：全天 [0,14220)、下午 [7200,14220)=13:00-14:57、
//! 尾盘 [12420,14220)=14:27-14:57。三段都不含开盘和收盘集合竞价。
//!
//! 运行：align_self_repeat <date> [--shuffle | --markov] [--debug CODE] > out.json
mod align;
mod fast_csv_reader;

use align::{select_nonoverlap, self_align, AlignOut, Seg};
use fast_csv_reader::{read_market_fast, read_trade_fast, MarketRecord, TradeRecord};
use rayon::prelude::*;
use serde::Serialize;
use std::collections::HashMap;
use std::time::Instant;

use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
static PROF_READ: AtomicU64 = AtomicU64::new(0);
static PROF_LETTER: AtomicU64 = AtomicU64::new(0);
static PROF_DP: AtomicU64 = AtomicU64::new(0);
static PROF_FEAT: AtomicU64 = AtomicU64::new(0);
static PROF_CELLS: AtomicU64 = AtomicU64::new(0);

const THREADS: usize = 50;
const SEG_CAP: usize = 20_000;
const MIN_N: usize = 30;
const DAY_START_SEC: i64 = 9 * 3600 + 30 * 60;
const WINDOWS: [(&str, i64, i64); 3] = [("full", 0, 14_220), ("pm", 7_200, 14_220), ("tail30", 12_420, 14_220)];
const N_WIN: usize = 3;

const FEATS: [&str; 22] = [
    "cov", "cov_vol", "seg_k", "len_mean", "len_max", "score_per_event", "tand_cov", "tand_copy",
    "lag_modeshr", "lag_logmean", "lag_ent", "tgap_logmed", "tgap_cv", "gaprate", "mmrate", "cov_bs",
    "center", "cluster", "size_ratio", "impact_ratio", "motif_distinct", "motif_topshr",
];
const N_F: usize = FEATS.len();
const BASES: [&str; 5] = ["lag1_match", "ac_excess_max", "lz78_norm", "run_mean", "entropy"];
const N_B: usize = BASES.len();

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
    Side(u8), // 0 只主买 1 只主卖 2 只撤单，字母 = 单量5档×间隔5档
    BarVolFlow(i64),
    BarCancelSize(i64),
    BarLob(i64),
}

struct Spec {
    name: &'static str,
    kind: Kind,
    max_lag: usize, // 0 表示序列长度的一半
    gap_mult: i32,
    compress: bool,
    group: u8, // 0 事件字母 1 单边 2 时间格
}

const SPECS: [Spec; 21] = [
    Spec { name: "e_size_raw", kind: Kind::Size, max_lag: 128, gap_mult: 1, compress: false, group: 0 },
    Spec { name: "e_lot_raw", kind: Kind::Lot, max_lag: 128, gap_mult: 1, compress: false, group: 0 },
    Spec { name: "e_gapsec_raw", kind: Kind::GapExact, max_lag: 128, gap_mult: 1, compress: false, group: 0 },
    Spec { name: "e_dir_rc", kind: Kind::Dir, max_lag: 128, gap_mult: 1, compress: true, group: 0 },
    Spec { name: "e_size_rc", kind: Kind::Size, max_lag: 128, gap_mult: 1, compress: true, group: 0 },
    Spec { name: "e_lot_rc", kind: Kind::Lot, max_lag: 128, gap_mult: 1, compress: true, group: 0 },
    Spec { name: "e_gap_rc", kind: Kind::Gap, max_lag: 128, gap_mult: 1, compress: true, group: 0 },
    Spec { name: "e_gapsec_rc", kind: Kind::GapExact, max_lag: 128, gap_mult: 1, compress: true, group: 0 },
    Spec { name: "e_oid_rc", kind: Kind::Oid, max_lag: 128, gap_mult: 1, compress: true, group: 0 },
    Spec { name: "e_sizegap_rc", kind: Kind::SizeGap, max_lag: 128, gap_mult: 1, compress: true, group: 0 },
    Spec { name: "e_sizeoid_rc", kind: Kind::SizeOid, max_lag: 128, gap_mult: 1, compress: true, group: 0 },
    Spec { name: "e_size_rc_L32", kind: Kind::Size, max_lag: 32, gap_mult: 1, compress: true, group: 0 },
    Spec { name: "e_size_rc_L512", kind: Kind::Size, max_lag: 512, gap_mult: 1, compress: true, group: 0 },
    Spec { name: "e_size_rc_nogap", kind: Kind::Size, max_lag: 128, gap_mult: 1000, compress: true, group: 0 },
    Spec { name: "buy_sizegap_rc", kind: Kind::Side(0), max_lag: 128, gap_mult: 1, compress: true, group: 1 },
    Spec { name: "sell_sizegap_rc", kind: Kind::Side(1), max_lag: 128, gap_mult: 1, compress: true, group: 1 },
    Spec { name: "cancel_sizegap_rc", kind: Kind::Side(2), max_lag: 128, gap_mult: 1, compress: true, group: 1 },
    Spec { name: "bar15_volflow", kind: Kind::BarVolFlow(15), max_lag: 0, gap_mult: 1, compress: false, group: 2 },
    Spec { name: "bar60_volflow", kind: Kind::BarVolFlow(60), max_lag: 0, gap_mult: 1, compress: false, group: 2 },
    Spec { name: "bar15_cancelsize", kind: Kind::BarCancelSize(15), max_lag: 0, gap_mult: 1, compress: false, group: 2 },
    Spec { name: "bar15_lob_rc", kind: Kind::BarLob(15), max_lag: 0, gap_mult: 1, compress: true, group: 2 },
];
const N_S: usize = SPECS.len();
const N_RAW: usize = N_S * N_WIN * (N_F + N_B);
const GROUPS: [(&str, Option<u8>); 4] = [("all", None), ("event", Some(0)), ("side", Some(1)), ("bar", Some(2))];

#[inline]
fn raw_f(s: usize, w: usize, f: usize) -> usize {
    (s * N_WIN + w) * (N_F + N_B) + f
}
#[inline]
fn raw_b(s: usize, w: usize, b: usize) -> usize {
    (s * N_WIN + w) * (N_F + N_B) + N_F + b
}

// ---------------------- 事件与字母化（第 1 层） ----------------------

#[derive(Clone, Copy)]
struct Ev {
    t: i64, // 距 9:30 的微秒（下午已前移）
    dir: u8, // 0 主买 1 主卖 2 撤买 3 撤卖
    vol: f64,
    amt: f64,
    oid: i64,
}

fn make_events(mut r: Vec<TradeRecord>) -> Vec<Ev> {
    r.sort_by_key(|x| (x.time_us, x.index));
    let t0 = DAY_START_SEC * 1_000_000;
    let mut out: Vec<Ev> = Vec::with_capacity(r.len());
    for x in r.iter() {
        let t = x.time_us.rem_euclid(86_400_000_000) - t0;
        if x.volume <= 0.0 {
            continue;
        }
        match x.flag {
            66 | 83 => {
                let (dir, oid) = if x.flag == 66 { (0u8, x.bid_order) } else { (1u8, x.ask_order) };
                // 同一张主动单吃掉多个对手单：合并成一个事件，避免把“一笔大单”当成“重复”
                if let Some(last) = out.last_mut() {
                    if last.dir == dir && last.oid == oid {
                        last.vol += x.volume;
                        last.amt += x.turnover;
                        continue;
                    }
                }
                out.push(Ev { t, dir, vol: x.volume, amt: x.turnover, oid });
            }
            32 => {
                let (dir, oid) = if x.bid_order > 0 { (2u8, x.bid_order) } else { (3u8, x.ask_order) };
                if oid > 0 {
                    out.push(Ev { t, dir, vol: x.volume, amt: 0.0, oid });
                }
            }
            _ => {}
        }
    }
    out
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
}

fn compact(tokens: Vec<u64>) -> (Vec<u32>, usize) {
    let mut u = tokens.clone();
    u.sort_unstable();
    u.dedup();
    let letters = tokens.iter().map(|t| u.binary_search(t).unwrap() as u32).collect();
    (letters, u.len().max(1))
}

/// 成交价前向填充（撤单、空时间格没有价）。
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
    let mut g = Vec::with_capacity(ev.len());
    for i in 0..ev.len() {
        g.push(if i == 0 { f64::NAN } else { (ev[i].t - ev[i - 1].t) as f64 / 1e6 });
    }
    g
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
    let split = |flag: bool| -> Vec<f64> {
        jump.iter().zip(is_cancel.iter()).filter(|(_, c)| **c == flag).map(|(v, _)| *v).collect()
    };
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

fn seq_from_events(ev: &[Ev], letters: Vec<u32>, k: usize) -> Seq {
    let mut px: Vec<f64> = ev.iter().map(|e| if e.dir <= 1 && e.vol > 0.0 { e.amt / e.vol } else { f64::NAN }).collect();
    ffill(&mut px);
    Seq {
        letters,
        k,
        vol: ev.iter().map(|e| e.vol).collect(),
        buy: ev.iter().map(|e| if e.dir == 0 { e.vol } else { 0.0 }).collect(),
        sell: ev.iter().map(|e| if e.dir == 1 { e.vol } else { 0.0 }).collect(),
        t_s: ev.iter().map(|e| e.t as f64 / 1e6).collect(),
        px,
    }
}

/// 把连续相同字母折叠成一个（低复杂度屏蔽，对应生物里的同聚物压缩）。
fn run_compress(s: Seq) -> Seq {
    let mut o = Seq { letters: vec![], k: s.k, vol: vec![], buy: vec![], sell: vec![], t_s: vec![], px: vec![] };
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
        }
    }
    o
}

/// 事件级序列的公共分档，一个时段算一次，所有字母表共用。
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

fn build_event_seq(kind: Kind, ev: &[Ev], c: &EvCache) -> Option<Seq> {
    let zip = |a: &[u32], ka: u32, b: &[u32]| -> Vec<u32> { a.iter().zip(b).map(|(x, y)| x * ka + y).collect() };
    let (letters, k): (Vec<u32>, usize) = match kind {
        Kind::Dir => (c.dir.clone(), 4),
        Kind::Size => (zip(&c.dir, 5, &c.size5), 20),
        Kind::Gap => (zip(&c.dir, 5, &c.gap5), 20),
        Kind::Oid => (zip(&c.dir, 5, &c.oid5), 20),
        Kind::SizeGap => {
            let s: Vec<u32> = zip(&c.size3, 3, &c.gap3);
            (zip(&c.dir, 9, &s), 36)
        }
        Kind::SizeOid => {
            let s: Vec<u32> = zip(&c.size3, 3, &c.oid3);
            (zip(&c.dir, 9, &s), 36)
        }
        Kind::Lot => compact(ev.iter().map(|e| (e.dir as u64) << 40 | (e.vol.round() as u64)).collect()),
        Kind::GapExact => {
            let g = gaps_of(ev);
            let sec: Vec<u32> = g.iter().map(|v| if v.is_finite() { v.round().clamp(0.0, 60.0) as u32 } else { 61 }).collect();
            (zip(&c.dir, 62, &sec), 248)
        }
        _ => return None,
    };
    Some(seq_from_events(ev, letters, k))
}

fn build_side_seq(side: u8, ev: &[Ev]) -> Seq {
    let sub: Vec<Ev> = ev
        .iter()
        .copied()
        .filter(|e| match side {
            0 => e.dir == 0,
            1 => e.dir == 1,
            _ => e.dir >= 2,
        })
        .collect();
    let size = qbucket(&sub.iter().map(|e| e.vol).collect::<Vec<_>>(), 5);
    let gap = qbucket(&gaps_of(&sub), 5);
    let letters: Vec<u32> = size.iter().zip(gap.iter()).map(|(s, g)| s * 5 + g).collect();
    seq_from_events(&sub, letters, 25)
}

/// 时间格序列。第一个字母维度 × 第二个字母维度。
fn build_bar_seq(kind: Kind, ev: &[Ev], mk: &[MarketRecord], lo: i64, hi: i64) -> Option<Seq> {
    let bar = match kind {
        Kind::BarVolFlow(b) | Kind::BarCancelSize(b) | Kind::BarLob(b) => b,
        _ => return None,
    };
    let n = ((hi - lo) / bar) as usize;
    let (mut buy, mut sell, mut cancel, mut amt_v) = (vec![0.0; n], vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    let (mut px, mut naggr) = (vec![f64::NAN; n], vec![0.0; n]);
    for e in ev {
        let b = ((e.t / 1_000_000 - lo) / bar) as usize;
        if b >= n {
            continue;
        }
        match e.dir {
            0 => buy[b] += e.vol,
            1 => sell[b] += e.vol,
            _ => cancel[b] += e.vol,
        }
        if e.dir <= 1 {
            amt_v[b] += e.vol;
            naggr[b] += 1.0;
            if e.vol > 0.0 {
                px[b] = e.amt / e.vol;
            }
        }
    }
    let mut seqpx = px.clone();
    ffill(&mut seqpx);
    let vol: Vec<f64> = amt_v.clone();
    let (letters, k): (Vec<u32>, usize) = match kind {
        Kind::BarVolFlow(_) => {
            let q = qbucket(&vol, 5);
            let l = (0..n)
                .map(|i| {
                    let net = buy[i] - sell[i];
                    q[i] * 3 + if net > 0.0 { 2 } else if net < 0.0 { 0 } else { 1 }
                })
                .collect();
            (l, 15)
        }
        Kind::BarCancelSize(_) => {
            let cr: Vec<f64> = (0..n)
                .map(|i| {
                    let d = cancel[i] + vol[i];
                    if d > 0.0 { cancel[i] / d } else { 0.0 }
                })
                .collect();
            let az: Vec<f64> = (0..n).map(|i| if naggr[i] > 0.0 { vol[i] / naggr[i] } else { 0.0 }).collect();
            let (a, b) = (qbucket(&cr, 5), qbucket(&az, 5));
            (a.iter().zip(b.iter()).map(|(x, y)| x * 5 + y).collect(), 25)
        }
        Kind::BarLob(_) => {
            if mk.is_empty() {
                return None;
            }
            let (mut imb, mut spr) = (vec![f64::NAN; n], vec![f64::NAN; n]);
            let t0 = DAY_START_SEC * 1_000_000;
            for m in mk {
                let t = m.time_us.rem_euclid(86_400_000_000) - t0;
                let sec = t / 1_000_000;
                if sec < lo || sec >= hi {
                    continue;
                }
                let b = ((sec - lo) / bar) as usize;
                if b >= n {
                    continue;
                }
                let d = m.total_bid_vol + m.total_ask_vol;
                imb[b] = if d > 0.0 { (m.total_bid_vol - m.total_ask_vol) / d } else { 0.0 };
                let (a1, b1) = (m.ask_prcs[0], m.bid_prcs[0]);
                spr[b] = if a1 > 0.0 && b1 > 0.0 { (a1 - b1) / ((a1 + b1) * 0.5) } else { f64::NAN };
            }
            ffill(&mut imb);
            let (a, b) = (qbucket(&imb, 5), qbucket(&spr, 3));
            (a.iter().zip(b.iter()).map(|(x, y)| x * 3 + y).collect(), 15)
        }
        _ => return None,
    };
    let t_s: Vec<f64> = (0..n).map(|i| (lo + i as i64 * bar) as f64).collect();
    Some(Seq { letters, k, vol, buy, sell, t_s, px: seqpx })
}

// ---------------------- 特征（第 3 层） ----------------------

fn fnv(xs: &[u32]) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for &x in xs {
        h ^= x as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

fn features(seq: &Seq, ao: &AlignOut, out: &mut [f32]) {
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

    // 串联重复覆盖在没有片段时也有定义（为 0）
    f[6] = 0.0;
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

        let mut tg: Vec<f64> = segs
            .iter()
            .filter_map(|s| {
                let j = (s.si + s.sl) as usize;
                if j < n { Some(seq.t_s[j] - seq.t_s[s.si as usize]) } else { None }
            })
            .collect();
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

        let mut hs: Vec<(u64, f64)> = segs
            .iter()
            .map(|s| (fnv(&seq.letters[(s.si as usize)..=(s.ei as usize)]), span(s)))
            .collect();
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
    }
    for (o, v) in out.iter_mut().zip(f.iter()) {
        *o = *v as f32;
    }
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
    let used = {
        let mut c = vec![0u32; seq.k];
        for &x in &seq.letters {
            c[x as usize] += 1;
        }
        c
    };
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

/// 确定性洗牌（空模型）：只打乱字母顺序，量、时间不动。
fn shuffle(letters: &mut [u32], seed: u64) {
    let mut x = seed | 1;
    for i in (1..letters.len()).rev() {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        letters.swap(i, (x % (i as u64 + 1)) as usize);
    }
}

/// 一阶马尔可夫空模型：保留“相邻两个字母的转移规律”，其余随机（对应生物里的二核苷酸打乱）。
fn markov_resample(letters: &mut [u32], k: usize, seed: u64) {
    let mut succ: Vec<Vec<u32>> = vec![Vec::new(); k];
    for w in letters.windows(2) {
        succ[w[0] as usize].push(w[1]);
    }
    let mut x = seed | 1;
    for i in 1..letters.len() {
        let from = letters[i - 1] as usize;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        if !succ[from].is_empty() {
            letters[i] = succ[from][(x % succ[from].len() as u64) as usize];
        }
    }
}

fn per_stock(code: &str, date: i64, null_mode: u8, debug: bool) -> Option<(Vec<f32>, [f32; N_WIN])> {
    let t_read = Instant::now();
    let trades = read_trade_fast(code, date).ok()?;
    let mk = read_market_fast(code, date).unwrap_or_default();
    PROF_READ.fetch_add(t_read.elapsed().as_micros() as u64, Relaxed);
    let t_ev = Instant::now();
    let ev_all = make_events(trades);
    PROF_LETTER.fetch_add(t_ev.elapsed().as_micros() as u64, Relaxed);
    if ev_all.is_empty() {
        return None;
    }
    let mut row = vec![f32::NAN; N_RAW];
    let mut n_ev = [f32::NAN; N_WIN];
    for (w, &(wname, lo, hi)) in WINDOWS.iter().enumerate() {
        let ev: Vec<Ev> = ev_all.iter().copied().filter(|e| e.t >= lo * 1_000_000 && e.t < hi * 1_000_000).collect();
        n_ev[w] = ev.len() as f32;
        if ev.len() < MIN_N {
            continue;
        }
        let cache = ev_cache(&ev);
        for (s, sp) in SPECS.iter().enumerate() {
            let t_l = Instant::now();
            let seq = match sp.kind {
                Kind::Side(side) => Some(build_side_seq(side, &ev)),
                Kind::BarVolFlow(_) | Kind::BarCancelSize(_) | Kind::BarLob(_) => build_bar_seq(sp.kind, &ev, &mk, lo, hi),
                _ => build_event_seq(sp.kind, &ev, &cache),
            };
            let Some(mut seq) = seq else { continue };
            if sp.compress {
                seq = run_compress(seq);
            }
            if seq.letters.len() < if matches!(sp.group, 2) { 20 } else { MIN_N } {
                continue;
            }
            let seed = fnv(&[s as u32, w as u32, code.parse::<u32>().unwrap_or(7)]);
            match null_mode {
                1 => shuffle(&mut seq.letters, seed),
                2 => markov_resample(&mut seq.letters, seq.k, seed),
                _ => {}
            }
            let lag = if sp.max_lag == 0 { seq.letters.len() / 2 } else { sp.max_lag };
            PROF_LETTER.fetch_add(t_l.elapsed().as_micros() as u64, Relaxed);
            let t_d = Instant::now();
            PROF_CELLS.fetch_add((seq.letters.len() * lag.min(seq.letters.len() - 1)) as u64, Relaxed);
            let Some(ao) = self_align(&seq.letters, seq.k, lag, sp.gap_mult, SEG_CAP) else { continue };
            PROF_DP.fetch_add(t_d.elapsed().as_micros() as u64, Relaxed);
            let t_f = Instant::now();
            features(&seq, &ao, &mut row[raw_f(s, w, 0)..raw_f(s, w, 0) + N_F]);
            baselines(&seq, &ao, &mut row[raw_b(s, w, 0)..raw_b(s, w, 0) + N_B]);
            PROF_FEAT.fetch_add(t_f.elapsed().as_micros() as u64, Relaxed);
            if debug && w == 0 {
                let (picked, _) = select_nonoverlap(&ao.segs, seq.letters.len());
                eprintln!("[debug {code} {wname} {}] n={} k={} T={} kept={}", sp.name, seq.letters.len(), seq.k, ao.threshold, picked.len());
                for g in picked.iter().take(3) {
                    let a = &seq.letters[g.si as usize..=(g.ei as usize).min(g.si as usize + 40)];
                    eprintln!("   score={} first@{}..{} lag={}->{} ma={} mm={} gap={} t={:.0}s letters={:?}", g.score, g.si, g.ei, g.sl, g.el, g.matches, g.mism, g.gaps, seq.t_s[g.si as usize], a);
                }
            }
        }
    }
    Some((row, n_ev))
}

// ---------------------- 横截面合成（第 4 层） ----------------------

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

/// 活跃度分组内排名：按活跃度从低到高分 20 组，每组内部排名到 0~1。
/// 比线性回归残差稳健：大量为 0 或有极端值的列，回归残差会退化成活跃度本身（实测秩相关 ±1）。
fn rank_in_bins(y: &[f32], x: &[f32], bins: usize) -> Vec<f32> {
    let mut idx: Vec<usize> = (0..y.len()).filter(|&i| x[i].is_finite()).collect();
    let mut out = vec![f32::NAN; y.len()];
    idx.sort_unstable_by(|&a, &b| x[a].partial_cmp(&x[b]).unwrap().then(a.cmp(&b)));
    let m = idx.len();
    for b in 0..bins {
        let chunk = &idx[b * m / bins..(b + 1) * m / bins];
        let sub: Vec<f32> = chunk.iter().map(|&i| y[i]).collect();
        let r = rank01(&sub);
        for (&i, &v) in chunk.iter().zip(r.iter()) {
            out[i] = v;
        }
    }
    out
}

fn symbol_list() -> std::io::Result<(Vec<String>, String, u64)> {
    use std::io::{Error, ErrorKind};
    let invalid = |e: csv::Error| Error::new(ErrorKind::InvalidData, e);
    let basic = std::env::var("RUST_PYFUNC_BASIC_INFO_DIR").unwrap_or("/ssd_data/data/basic_info".into());
    let path = std::path::Path::new(&basic).join("symbol_map.csv");
    let bytes = std::fs::read(&path)?;
    let hash = bytes.iter().fold(0xcbf29ce484222325u64, |h, b| (h ^ *b as u64).wrapping_mul(0x100000001b3));
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

fn list_codes(date: i64) -> std::io::Result<(Vec<String>, String, u64)> {
    let root = std::env::var("RUST_PYFUNC_LEVEL2_PATH").unwrap_or("/ssd_data/stock".into());
    let dir = std::path::Path::new(&root).join(date.to_string()).join("transaction");
    if !dir.is_dir() {
        return Err(std::io::Error::new(std::io::ErrorKind::NotFound, format!("原始目录不存在 {}", dir.display())));
    }
    let (all, p, h) = symbol_list()?;
    let codes = all.into_iter().filter(|c| dir.join(format!("{c}_{date}_transaction.csv")).is_file()).collect();
    Ok((codes, p, h))
}

#[derive(Serialize)]
struct Out {
    date: i64,
    shuffle: bool,
    symbol_map: String,
    symbol_map_fnv: String,
    names: Vec<String>,
    codes: Vec<String>,
    n_events: Vec<Vec<f32>>,
    vals: Vec<Vec<f32>>,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let date: i64 = args.get(1).expect("用法: align_self_repeat <date> [--shuffle] [--debug CODE]").parse().expect("日期须为 8 位整数");
    let null_mode: u8 = if args.iter().any(|a| a == "--shuffle") { 1 } else if args.iter().any(|a| a == "--markov") { 2 } else { 0 };
    let do_shuffle = null_mode != 0;
    let debug = args.iter().position(|a| a == "--debug").and_then(|i| args.get(i + 1).cloned());
    rayon::ThreadPoolBuilder::new().num_threads(THREADS).build_global().unwrap();

    let t0 = Instant::now();
    let (mut codes, symbol_map, hash) = list_codes(date).expect("股票范围或原始目录检查失败");
    if let Some(c) = &debug {
        codes.retain(|x| x == c);
    }
    eprintln!("codes = {}  (threads = {THREADS})", codes.len());

    let res: Vec<Option<(Vec<f32>, [f32; N_WIN])>> = codes.par_iter().map(|c| per_stock(c, date, null_mode, debug.is_some())).collect();
    let mut valid = Vec::with_capacity(codes.len());
    let mut rows: Vec<Vec<f32>> = Vec::with_capacity(codes.len());
    let mut n_ev: Vec<[f32; N_WIN]> = Vec::with_capacity(codes.len());
    let mut dropped = 0usize;
    for (c, r) in codes.iter().zip(res.into_iter()) {
        match r {
            Some((a, b)) => {
                valid.push(c.clone());
                rows.push(a);
                n_ev.push(b);
            }
            None => dropped += 1,
        }
    }
    eprintln!("valid = {} dropped = {dropped}  per-stock {:.1}s", valid.len(), t0.elapsed().as_secs_f64());

    eprintln!("[profile] thread-seconds read={:.0} letters={:.0} dp={:.0} features={:.0}  cells={:.2e}", PROF_READ.load(Relaxed) as f64/1e6, PROF_LETTER.load(Relaxed) as f64/1e6, PROF_DP.load(Relaxed) as f64/1e6, PROF_FEAT.load(Relaxed) as f64/1e6, PROF_CELLS.load(Relaxed) as f64);

    // 第 4 层
    let t1 = Instant::now();
    let m = valid.len();
    let col = |j: usize| -> Vec<f32> { rows.iter().map(|r| r[j]).collect() };
    let x_log: Vec<Vec<f32>> = (0..N_WIN).map(|w| n_ev.iter().map(|n| (1.0 + n[w]).ln()).collect()).collect();

    // 活跃度分组内排名（特征和朴素对照都做，比较时才公平）
    let idx: Vec<(usize, usize, usize)> = (0..N_S).flat_map(|s| (0..N_WIN).flat_map(move |w| (0..N_F).map(move |f| (s, w, f)))).collect();
    let idx_b: Vec<(usize, usize, usize)> = (0..N_S).flat_map(|s| (0..N_WIN).flat_map(move |w| (0..N_B).map(move |b| (s, w, b)))).collect();
    let nrank: Vec<Vec<f32>> = idx.par_iter().map(|&(s, w, f)| rank_in_bins(&col(raw_f(s, w, f)), &x_log[w], 20)).collect();
    let nrank_b: Vec<Vec<f32>> = idx_b.par_iter().map(|&(s, w, b)| rank_in_bins(&col(raw_b(s, w, b)), &x_log[w], 20)).collect();
    let rk = |s: usize, w: usize, f: usize| -> &Vec<f32> { &nrank[(s * N_WIN + w) * N_F + f] };

    // 分组平均名次：至少一半的序列有值才给结果
    let mut comp: Vec<Vec<f32>> = Vec::new();
    let mut comp_names: Vec<String> = Vec::new();
    for (gname, g) in GROUPS.iter() {
        let members: Vec<usize> = (0..N_S).filter(|&s| g.map_or(true, |x| SPECS[s].group == x)).collect();
        for f in 0..N_F {
            let mut by_win: Vec<Vec<f32>> = Vec::with_capacity(N_WIN);
            for w in 0..N_WIN {
                let v: Vec<f32> = (0..m)
                    .map(|i| {
                        let (mut sum, mut cnt) = (0.0f32, 0usize);
                        for &s in &members {
                            let x = rk(s, w, f)[i];
                            if x.is_finite() {
                                sum += x;
                                cnt += 1;
                            }
                        }
                        if cnt * 2 >= members.len() { sum / cnt as f32 } else { f32::NAN }
                    })
                    .collect();
                comp_names.push(format!("cmp_{gname}__{}__{}", WINDOWS[w].0, FEATS[f]));
                by_win.push(v);
            }
            for (wn, w) in [("pm_minus_full", 1usize), ("tail_minus_full", 2usize)] {
                let d: Vec<f32> = (0..m).map(|i| by_win[w][i] - by_win[0][i]).collect();
                comp_names.push(format!("cmp_{gname}__{wn}__{}", FEATS[f]));
                by_win.push(d);
            }
            comp.extend(by_win);
        }
    }

    // 组装：原始特征 + 活跃度残差 + 朴素对照 + 合成
    let mut names: Vec<String> = Vec::new();
    for s in 0..N_S {
        for w in 0..N_WIN {
            for f in 0..N_F {
                names.push(format!("{}__{}__{}", SPECS[s].name, WINDOWS[w].0, FEATS[f]));
            }
        }
    }
    let n_raw_f = names.len();
    for s in 0..N_S {
        for w in 0..N_WIN {
            for f in 0..N_F {
                names.push(format!("{}__{}__{}__nrank", SPECS[s].name, WINDOWS[w].0, FEATS[f]));
            }
        }
    }
    for s in 0..N_S {
        for w in 0..N_WIN {
            for b in 0..N_B {
                names.push(format!("{}__{}__base_{}", SPECS[s].name, WINDOWS[w].0, BASES[b]));
            }
        }
    }
    for s in 0..N_S {
        for w in 0..N_WIN {
            for b in 0..N_B {
                names.push(format!("{}__{}__base_{}__nrank", SPECS[s].name, WINDOWS[w].0, BASES[b]));
            }
        }
    }
    names.extend(comp_names);
    let n_col = names.len();

    let vals: Vec<Vec<f32>> = (0..m)
        .map(|i| {
            let mut v = Vec::with_capacity(n_col);
            for j in 0..n_raw_f {
                let (s, w, f) = idx[j];
                v.push(rows[i][raw_f(s, w, f)]);
            }
            for r in &nrank {
                v.push(r[i]);
            }
            for s in 0..N_S {
                for w in 0..N_WIN {
                    for b in 0..N_B {
                        v.push(rows[i][raw_b(s, w, b)]);
                    }
                }
            }
            for r in &nrank_b {
                v.push(r[i]);
            }
            for c in &comp {
                v.push(c[i]);
            }
            v
        })
        .collect();
    eprintln!("columns = {n_col}  cross-section {:.1}s  total {:.1}s", t1.elapsed().as_secs_f64(), t0.elapsed().as_secs_f64());
    let t2 = Instant::now();
    let out = Out {
        date,
        shuffle: do_shuffle,
        symbol_map,
        symbol_map_fnv: format!("{hash:016x}"),
        names,
        codes: valid,
        n_events: n_ev.iter().map(|n| n.to_vec()).collect(),
        vals,
    };
    let stdout = std::io::stdout();
    let w = std::io::BufWriter::with_capacity(1 << 20, stdout.lock());
    serde_json::to_writer(w, &out).unwrap();
    eprintln!("write json {:.1}s", t2.elapsed().as_secs_f64());
}
