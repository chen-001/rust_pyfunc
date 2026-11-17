//! 基因动态规划 · 方向 1「今天对昨天」：同一只股票今天与昨天同一时段的行为字母序列做动态规划比对。
//! 由沙箱 sandbox_align_day_vs_day 第三版原样搬入（计算代码未改，因子值与沙箱逐字节一致）。
//! 只用今天和昨天两天的原始数据（昨天从原始数据重算），不落盘任何中间结果。
//!
//! 层级：
//!   1 字母化：3 个时段 × 4 档粒度 × (时间钟 11 + 成交量钟 7 + 等笔数钟 7 个字母表) + 4 个逐笔级字母表
//!   2 比对：宽松全局（带宽 1/4、1/8、1/2）、严格全局、首尾免罚、局部、不许错位
//!   3 路径特征：每条格子序列 55 列，每条逐笔序列 25 列，另有每档粒度 3 列朴素对照
//!   4 差值与变换：每个单因子的三种时段差；关键特征的流动性 / 行业 / 市值组内超额
//! 不做任何多因子合成。依赖撤单数据的列名字都含 cxl，撤单数据缺失的日子写空值。
pub mod align;
pub mod letters;
pub mod reader;

use align::{align_batch, align_scalar, Aln, BatchWork, Mode, Prob, Work, OP_D, OP_X};
use reader::{read_market_fast, read_trade_with_cancel, MarketRecord, TradeRecord};
use letters::*;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

const MAX_THREADS: usize = 50;
const SEC: i64 = 1_000_000;
const OPEN_US: i64 = (9 * 3600 + 30 * 60) * SEC;
const DAY_LEN: i64 = 14_220; // 09:30-11:30 + 13:00-14:57，下午已前移 90 分钟
/// 时段：(名字, 起始秒)。全天 09:30-14:57；下午 13:00-14:57；尾盘 14:27-14:57；都不含集合竞价
const WINDOWS: [(&str, i64); 3] = [("all", 0), ("pm", 7_200), ("tail", DAY_LEN - 1_800)];
/// 每个时段 4 档粒度（秒），名字里记作 1~4（1 最细）
const GRAINS: [[i64; 4]; 3] = [[15, 30, 60, 120], [10, 15, 30, 60], [3, 5, 10, 20]];
const CLOCK_TAG: [&str; 3] = ["t", "v", "n"];
const ALPHA_T: [usize; 11] = [VOLSIG, MAG5, MAG5Y, SIZEL, RHYTHM, OID, PRICE, BOOKFLOW, BOOKSHAPE, CXL, BIG];
const ALPHA_E: [usize; 7] = [VOLSIG, MAG5, SIZEL, RHYTHM, OID, PRICE, BIG];
const N_STREAM: usize = 25;
const EV_BAND: usize = 100;
const EV_MAX_RATIO: usize = 50;

const OPEN_LOOSE: i32 = 3;
const EXT_LOOSE: i32 = 1;
const OPEN_TIGHT: i32 = 8;
const EXT_TIGHT: i32 = 2;
const MAX_S: f64 = 4.0;

const GL_NAMES: [&str; 20] = ["score", "excess", "identity", "partial", "gap_open", "gap_bar",
    "lag_mean", "lag_abs", "lag_std", "lag_trend", "lag_h1_h2", "insert", "delete", "ins_del_asym",
    "insert_vw", "delete_vw", "lag_vw", "longest_run", "insert_center", "dp_gain"];
const SUB6: [usize; 6] = [0, 1, 2, 4, 6, 7];
const SUB6_NAMES: [&str; 6] = ["score", "excess", "identity", "gap_open", "lag_mean", "lag_abs"];
const LC_NAMES: [&str; 6] = ["score", "span", "start", "offset", "identity", "vol_cov"];
const N_FIX: usize = 55;
const N_EV: usize = 25;
const N_NAIVE: usize = 3;
const PER_WIN: usize = 4 * N_STREAM * N_FIX + 4 * N_EV + 4 * N_NAIVE; // 5612
const N_RAW: usize = PER_WIN * 3;
/// 做组内超额的关键特征（在单条序列内的偏移）
const KEY_FIX: [usize; 10] = [0, 1, 2, 4, 6, 7, 13, 19, 40, 41];
const KEY_EV: [usize; 7] = [0, 1, 2, 4, 6, 7, 13];
const GROUPS: [&str; 3] = ["liqx", "indx", "capx"];

/// 25 条格子序列：(时钟, 字母表)
fn streams() -> Vec<(usize, usize)> {
    let mut v: Vec<(usize, usize)> = ALPHA_T.iter().map(|&a| (0, a)).collect();
    for ck in 1..3 { for &a in ALPHA_E.iter() { v.push((ck, a)); } }
    v
}

fn grain_len(wi: usize, gi: usize) -> usize { ((DAY_LEN - WINDOWS[wi].1) / GRAINS[wi][gi]) as usize }

/// 撤单数据起始日：沪市 20211105，深市 20160104（逐日核对原始文件得到）。今天和昨天都要有撤单才算。
fn cxl_available(code: &str, date: i64, prev: i64) -> bool {
    let start = match code.as_bytes()[0] { b'6' => 20211105, b'0' | b'3' => 20160104, _ => i64::MAX };
    date >= start && prev >= start
}

fn basic_dir() -> String { crate::data_paths::basic_info_path("").to_string_lossy().trim_end_matches('/').to_string() }
fn list_codes(date: i64, subdir: &str) -> std::io::Result<Vec<String>> {
    let root = crate::genedp_common::level2_root_string();
    let sym_map = std::path::Path::new(&basic_dir()).join("symbol_map.csv");
    crate::genedp_common::stock_codes_whitelisted(&root, &sym_map, date, subdir)
        .map(|(codes, _, _)| codes)
        .map_err(crate::genedp_common::invalid)
}

/// 读日频 h5 某一天（最后一个 ≤ date 的行）给定股票的值。sym_csv 给出 代码→列号（pos 列）。
fn h5_row(h5: &str, cal: &str, sym: &str, date: i64, codes: &[String]) -> Result<Vec<f64>, String> {
    let dates: Vec<i64> = std::fs::read_to_string(cal).map_err(|e| format!("{cal}: {e}"))?
        .lines().skip(1).filter_map(|l| l.trim().parse().ok()).collect();
    let r = dates.partition_point(|&d| d <= date);
    if r == 0 { return Err(format!("{h5} 没有 {date} 及以前的行")); }
    let mut pos = std::collections::HashMap::new(); // 只做查找，不遍历
    for line in std::fs::read_to_string(sym).map_err(|e| format!("{sym}: {e}"))?.lines().skip(1) {
        let mut it = line.split(',');
        let (c, p) = (it.next().unwrap_or("").trim(), it.next().unwrap_or("").trim());
        if let Ok(p) = p.parse::<usize>() { pos.insert(format!("{c:0>6}"), p); }
    }
    let f = hdf5_metno::File::open(h5).map_err(|e| format!("{h5}: {e}"))?;
    let ds = f.dataset("data").map_err(|e| format!("{h5}: {e}"))?;
    let ncol = ds.shape()[1];
    let row: ndarray::Array1<f64> = ds.read_slice_1d(ndarray::s![r - 1, ..]).map_err(|e| format!("{h5}: {e}"))?;
    Ok(codes.iter().map(|c| match pos.get(c) { Some(&p) if p < ncol => row[p], _ => f64::NAN }).collect())
}

/// 一天一个时段的全部字母化输入
struct WinData { bars: Vec<Vec<Bars>>, cuts: Vec<[f64; 4]>, ev: EventSeqs }
struct DayData { win: Vec<WinData>, n_full: f64 }

fn split_sorted(mut all: Vec<TradeRecord>) -> (Vec<TradeRecord>, Vec<TradeRecord>) {
    if !all.windows(2).all(|w| w[0].time_us <= w[1].time_us) { all.sort_by_key(|t| t.time_us); }
    let (mut tr, mut ca) = (Vec::with_capacity(all.len()), Vec::with_capacity(all.len() / 4));
    for t in all { if t.flag == 32 { ca.push(t); } else { tr.push(t); } } // 撤单只看标记 32
    (tr, ca)
}
fn slice_by_time<T>(v: &[T], lo: i64, hi: i64, tm: impl Fn(&T) -> i64) -> &[T] {
    let a = v.partition_point(|x| tm(x) < lo);
    let b = v.partition_point(|x| tm(x) <= hi);
    &v[a..b.max(a)]
}

fn build_day(trades: &[TradeRecord], cancels: &[TradeRecord], mkt: &[MarketRecord], buf: &mut Vec<f64>) -> Option<DayData> {
    let t0 = trades.first()?.time_us;
    let day_us = t0 - t0.rem_euclid(86_400 * SEC);
    let mut win = Vec::with_capacity(3);
    let mut n_full = 0.0;
    for (wi, &(_, s)) in WINDOWS.iter().enumerate() {
        let start = OPEN_US + s * SEC;
        let (lo, hi) = (day_us + start, day_us + OPEN_US + DAY_LEN * SEC);
        let tr = slice_by_time(trades, lo, hi, |t| t.time_us);
        let ca = slice_by_time(cancels, lo, hi, |t| t.time_us);
        let mk = slice_by_time(mkt, lo, hi, |t| t.time_us);
        if wi == 0 { n_full = tr.len() as f64; }
        let big = big_flags(tr);
        let mut bars = vec![Vec::with_capacity(4), Vec::with_capacity(4), Vec::with_capacity(4)];
        let mut cuts = Vec::with_capacity(4);
        for gi in 0..4 {
            let l = grain_len(wi, gi);
            let tb = time_bars(tr, &big, ca, mk, start, GRAINS[wi][gi] * SEC, l, day_us);
            cuts.push(mag5_ref_cuts(&tb, buf));
            bars[0].push(tb);
            bars[1].push(event_clock_bars(tr, &big, l, Clock::Volume));
            bars[2].push(event_clock_bars(tr, &big, l, Clock::Count));
        }
        win.push(WinData { bars, cuts, ev: event_letters(tr, ca, buf) });
    }
    Some(DayData { win, n_full })
}

/// 路径特征。i 是今天的位置，j 是昨天的位置；位置都换成占各自序列长度的比例再相减。
#[allow(clippy::too_many_arguments)]
fn global_features(a: &[u8], b: &[u8], va: &[f64], vb: &[f64], ops: &[u8], r: &Aln, tbl: &[i32], ug: f64, e: f64, out: &mut [f32]) {
    let (n, m) = (a.len() as f64, b.len() as f64);
    let score = r.score as f64;
    out[0] = (score / (MAX_S * n)) as f32;
    out[1] = ((score - e) / (MAX_S * n)) as f32;
    out[19] = ((score - ug) / (MAX_S * n)) as f32;
    let (mut i, mut j) = (r.i0, r.j0);
    let (mut same, mut partial, mut pairs) = (0f64, 0f64, 0f64);
    let (mut opens, mut gap_pos, mut last_op) = (0f64, 0f64, 3u8);
    let (mut lag_s, mut lag_a, mut lag_s2, mut lag_n) = (0f64, 0f64, 0f64, 0f64);
    let (mut lv_sum, mut lv_w) = (0f64, 0f64);
    let (mut sx, mut sxx, mut sxy) = (0f64, 0f64, 0f64);
    let (mut h1, mut h1n, mut h2, mut h2n) = (0f64, 0f64, 0f64, 0f64);
    let (mut ins_n, mut del_n, mut ins_pos) = (0f64, 0f64, 0f64);
    let (mut ins_v, mut del_v) = (0f64, 0f64);
    let (mut run, mut best_run) = (0f64, 0f64);
    let half = a.len() / 2;
    for &op in ops {
        match op {
            OP_D => {
                let (x, y) = (a[i], b[j]);
                if x != 0 || y != 0 {
                    pairs += 1.0;
                    if x == y { same += 1.0; } else if tbl[x as usize * align::SIZE + y as usize] > 0 { partial += 1.0; }
                }
                if x != 0 && x == y { run += 1.0; if run > best_run { best_run = run; } }
                else if x != 0 || y != 0 { run = 0.0; }
                if x != 0 && y != 0 {
                    let lag = i as f64 / n - j as f64 / m;
                    let fi = i as f64 / n;
                    lag_s += lag; lag_a += lag.abs(); lag_s2 += lag * lag; lag_n += 1.0;
                    lv_sum += va[i] * lag; lv_w += va[i];
                    sx += fi; sxx += fi * fi; sxy += fi * lag;
                    if i < half { h1 += lag; h1n += 1.0; } else { h2 += lag; h2n += 1.0; }
                }
                i += 1; j += 1;
            }
            OP_X => {
                if last_op != OP_X { opens += 1.0; }
                gap_pos += 1.0;
                if a[i] != 0 { ins_n += 1.0; ins_pos += i as f64; ins_v += va[i]; }
                i += 1;
            }
            _ => {
                if last_op != op { opens += 1.0; }
                gap_pos += 1.0;
                if b[j] != 0 { del_n += 1.0; del_v += vb[j]; }
                j += 1;
            }
        }
        last_op = op;
    }
    if pairs > 0.0 { out[2] = (same / pairs) as f32; out[3] = (partial / pairs) as f32; }
    out[4] = (opens / n) as f32;
    out[5] = (gap_pos / (n + m)) as f32;
    if lag_n >= 5.0 {
        out[6] = (lag_s / lag_n) as f32;
        out[7] = (lag_a / lag_n) as f32;
        let var = (lag_s2 / lag_n - (lag_s / lag_n).powi(2)).max(0.0);
        out[8] = var.sqrt() as f32;
        let cov = sxy / lag_n - (sx / lag_n) * (lag_s / lag_n);
        let vx = sxx / lag_n - (sx / lag_n).powi(2);
        out[9] = if var > 1e-12 && vx > 1e-12 { (cov / (var * vx).sqrt()) as f32 } else { 0.0 };
        if h1n >= 2.0 && h2n >= 2.0 { out[10] = (h1 / h1n - h2 / h2n) as f32; }
        if lv_w > 0.0 { out[16] = (lv_sum / lv_w) as f32; }
    }
    let (na, nb) = (a.iter().filter(|&&x| x != 0).count() as f64, b.iter().filter(|&&x| x != 0).count() as f64);
    if na > 0.0 { out[11] = (ins_n / na) as f32; }
    if nb > 0.0 { out[12] = (del_n / nb) as f32; }
    out[13] = if ins_n + del_n > 0.0 { ((ins_n - del_n) / (ins_n + del_n)) as f32 } else { 0.0 };
    let (sa, sb): (f64, f64) = (va.iter().sum(), vb.iter().sum());
    if sa > 0.0 { out[14] = (ins_v / sa) as f32; }
    if sb > 0.0 { out[15] = (del_v / sb) as f32; }
    out[17] = (best_run / n) as f32;
    if ins_n > 0.0 { out[18] = (ins_pos / ins_n / n) as f32; }
}

fn local_features(a: &[u8], b: &[u8], va: &[f64], ops: &[u8], r: &Aln, out: &mut [f32]) {
    let (n, m) = (a.len() as f64, b.len() as f64);
    out[0] = (r.score as f64 / (MAX_S * n)) as f32;
    out[1] = ((r.i1 - r.i0) as f64 / n) as f32;
    out[2] = (r.i0 as f64 / n) as f32;
    out[3] = (r.i0 as f64 / n - r.j0 as f64 / m) as f32;
    let (mut i, mut j, mut same, mut pairs) = (r.i0, r.j0, 0f64, 0f64);
    for &op in ops {
        match op {
            OP_D => { if a[i] != 0 || b[j] != 0 { pairs += 1.0; if a[i] == b[j] { same += 1.0; } } i += 1; j += 1; }
            OP_X => i += 1,
            _ => j += 1,
        }
    }
    if pairs > 0.0 { out[4] = (same / pairs) as f32; }
    let s: f64 = va.iter().sum();
    if s > 0.0 { out[5] = (va[r.i0..r.i1].iter().sum::<f64>() / s) as f32; }
}

/// 不许错位的逐位得分、相同率，以及随机期望（两天字母频率给出，长度不等时按较短的一边）
fn ungapped(a: &[u8], b: &[u8], tbl: &[i32]) -> (f64, f64, f64) {
    let (n, m) = (a.len(), b.len());
    let (mut fa, mut fb) = ([0f64; align::SIZE], [0f64; align::SIZE]);
    for &x in a { fa[x as usize] += 1.0; }
    for &y in b { fb[y as usize] += 1.0; }
    let mut e = 0f64;
    for x in 0..align::SIZE { for y in 0..align::SIZE { e += fa[x] * fb[y] * tbl[x * align::SIZE + y] as f64; } }
    let e = e * n.min(m) as f64 / (n as f64 * m as f64);
    if n != m { return (f64::NAN, f64::NAN, e); }
    let (mut ug, mut same, mut pairs) = (0i64, 0f64, 0f64);
    for k in 0..n {
        let (x, y) = (a[k], b[k]);
        ug += tbl[x as usize * align::SIZE + y as usize] as i64;
        if x != 0 || y != 0 { pairs += 1.0; if x == y { same += 1.0; } }
    }
    (ug as f64, if pairs > 0.0 { same / pairs } else { f64::NAN }, e)
}

fn pearson(a: &[f64], b: &[f64]) -> f32 {
    let n = a.len() as f64;
    let (ma, mb) = (a.iter().sum::<f64>() / n, b.iter().sum::<f64>() / n);
    let (mut c, mut x, mut y) = (0.0, 0.0, 0.0);
    for (u, v) in a.iter().zip(b) { c += (u - ma) * (v - mb); x += (u - ma).powi(2); y += (v - mb).powi(2); }
    if x > 0.0 && y > 0.0 { (c / (x * y).sqrt()) as f32 } else { f32::NAN }
}
fn ranks(v: &[f64]) -> Vec<f64> {
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.sort_by(|&x, &y| v[x].total_cmp(&v[y]).then(x.cmp(&y)));
    let mut r = vec![0.0; v.len()];
    let mut i = 0;
    while i < idx.len() {
        let mut j = i;
        while j + 1 < idx.len() && v[idx[j + 1]] == v[idx[i]] { j += 1; }
        for k in i..=j { r[idx[k]] = (i + j) as f64 * 0.5; }
        i = j + 1;
    }
    r
}

#[derive(Default)]
struct Timers { read: AtomicU64, prep: AtomicU64, dp: AtomicU64, dp_ev: AtomicU64, feats: AtomicU64, trades: AtomicU64 }
#[inline]
fn add(c: &AtomicU64, t: Instant) { c.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed); }

/// 每个线程复用的缓冲
#[derive(Default)]
struct ThreadWork {
    sw: Work, bw16: BatchWork<16>, mw8: align::MaskWork<i32, u8, 8>, mw32: align::MaskWork<i16, u32, 32>,
    ops: Vec<Vec<u8>>, alns: Vec<Aln>, la: Vec<Vec<u8>>, lb: Vec<Vec<u8>>, buf: Vec<f64>,
}

#[derive(Clone, Copy, PartialEq)]
enum Kernel { Scalar, Batch }

/// 比对题的种类：(模式, 空位开启, 空位延长, 带宽分母)
const KINDS: [(Mode, i32, i32, usize); 6] = [
    (Mode::Global, OPEN_LOOSE, EXT_LOOSE, 4), (Mode::Global, OPEN_TIGHT, EXT_TIGHT, 4),
    (Mode::Semi, OPEN_LOOSE, EXT_LOOSE, 4), (Mode::Local, OPEN_LOOSE, EXT_LOOSE, 4),
    (Mode::Global, OPEN_LOOSE, EXT_LOOSE, 8), (Mode::Global, OPEN_LOOSE, EXT_LOOSE, 2),
];

/// 跑一组题。题目按带宽分组排好（同组同形状），批量实现每批最多 8/16/32 题。
fn run_probs(probs: &[Prob], bands: &[usize], tables: &[Vec<i32>], tabs: &[usize], kernel: Kernel, tw: &mut ThreadWork, event: bool) {
    let np = probs.len();
    if tw.ops.len() < np { tw.ops.resize(np, Vec::new()); }
    tw.alns.clear(); tw.alns.resize(np, Aln::default());
    match kernel {
        Kernel::Scalar => {
            for k in 0..np {
                let p = &probs[k];
                tw.alns[k] = align_scalar(p.a, p.b, &tables[tabs[k]], p.open, p.ext, bands[k], p.mode, &mut tw.sw, &mut tw.ops[k]);
            }
        }
        Kernel::Batch => {
            // 机器支持 512 位向量就用显式向量版：格子级 32 车道 16 位（分数范围安全时），逐笔级 8 车道 32 位；
            // 否则退回编译器自动向量化的版本。全部与基准逐字节相同。
            let simd = align::has_avx512();
            let lanes = if !simd { 16 } else if event { 8 } else { 32 };
            let mut s = 0;
            while s < np {
                let mut e = s + 1;
                while e < np && e - s < lanes && bands[e] == bands[s] && probs[e].a.len() == probs[s].a.len()
                    && probs[e].b.len() == probs[s].b.len() { e += 1; }
                let (n, m) = (probs[s].a.len(), probs[s].b.len());
                #[cfg(target_arch = "x86_64")]
                let res = if !simd { align_batch::<16>(&probs[s..e], bands[s], &mut tw.bw16, &mut tw.ops[s..e]) }
                    else if event { unsafe { align::align_batch8_ymm(&probs[s..e], bands[s], &mut tw.mw8, &mut tw.ops[s..e]) } }
                    else if align::grid_i16_safe(n, m, OPEN_TIGHT, EXT_TIGHT) {
                        unsafe { align::align_batch32_i16(&probs[s..e], bands[s], &mut tw.mw32, &mut tw.ops[s..e]) } }
                    else {
                        // 16 位不安全：分成不超过 16 题一批走 32 位版
                        let mut r = Vec::new();
                        let mut q = s;
                        while q < e { let qe = (q + 16).min(e);
                            r.extend(unsafe { align::align_batch16_avx512(&probs[q..qe], bands[s], &mut tw.bw16, &mut tw.ops[q..qe]) }); q = qe; }
                        r
                    };
                #[cfg(not(target_arch = "x86_64"))]
                let res = { let _ = (n, m); align_batch::<16>(&probs[s..e], bands[s], &mut tw.bw16, &mut tw.ops[s..e]) };
                tw.alns[s..e].copy_from_slice(&res);
                s = e;
            }
        }
    }
}

struct Ctx { tables: Vec<Vec<i32>>, ev_tables: Vec<Vec<i32>>, streams: Vec<(usize, usize)>, kernel: Kernel }

fn per_stock(code: &str, date: i64, prev: i64, ctx: &Ctx, tw: &mut ThreadWork, tm: &Timers) -> (Vec<f32>, f64) {
    let mut out = vec![f32::NAN; N_RAW];
    let t = Instant::now();
    let td = read_trade_with_cancel(code, date);
    let mt = read_market_fast(code, date).unwrap_or_default();
    let yd = read_trade_with_cancel(code, prev);
    let my = read_market_fast(code, prev).unwrap_or_default();
    add(&tm.read, t);
    let t = Instant::now();
    let Ok(td) = td else { return (out, f64::NAN) };
    let (tt, tc) = split_sorted(td);
    tm.trades.fetch_add(tt.len() as u64, Ordering::Relaxed);
    let mut buf = std::mem::take(&mut tw.buf);
    let today = build_day(&tt, &tc, &mt, &mut buf);
    let n_full = today.as_ref().map(|d| d.n_full).unwrap_or(0.0);
    let yest = yd.ok().and_then(|y| { let (yt, yc) = split_sorted(y); build_day(&yt, &yc, &my, &mut buf) });
    add(&tm.prep, t);
    let (Some(today), Some(yest)) = (today, yest) else { tw.buf = buf; return (out, n_full) };
    let cxl_ok = cxl_available(code, date, prev);
    // 字母缓冲先从 tw 里取出来，比对时 tw 还要可变借用
    let (mut la, mut lb) = (std::mem::take(&mut tw.la), std::mem::take(&mut tw.lb));
    if la.len() < N_STREAM { la.resize(N_STREAM, Vec::new()); lb.resize(N_STREAM, Vec::new()); }
    for wi in 0..3 {
        let (tw_, yw) = (&today.win[wi], &yest.win[wi]);
        let wbase = wi * PER_WIN;
        for gi in 0..4 {
            let l = grain_len(wi, gi);
            let t = Instant::now();
            let ycut = yw.cuts[gi];
            let mut active = [false; N_STREAM];
            for (s, &(ck, al)) in ctx.streams.iter().enumerate() {
                let ca = letters(&tw_.bars[ck][gi], &ycut, al, &mut la[s], &mut buf);
                let cb = letters(&yw.bars[ck][gi], &ycut, al, &mut lb[s], &mut buf);
                let min_active = (l / 20).max(10);
                active[s] = ca >= min_active && cb >= min_active && (al != CXL || cxl_ok);
            }
            // 题目按带宽分组：先全部 1/4 带宽的 4 种，再 1/8，再 1/2
            let mut probs: Vec<Prob> = Vec::with_capacity(N_STREAM * 6);
            let (mut bands, mut tabs, mut owner) = (Vec::new(), Vec::new(), Vec::new());
            for group in [&[0usize, 1, 2, 3][..], &[4][..], &[5][..]] {
                for (s, &(_, al)) in ctx.streams.iter().enumerate() {
                    if !active[s] { continue; }
                    for &k in group {
                        let (mode, open, ext, div) = KINDS[k];
                        probs.push(Prob { a: &la[s], b: &lb[s], grid: GRID[al], mode, open, ext });
                        bands.push(l / div); tabs.push(al); owner.push((s, k));
                    }
                }
            }
            add(&tm.prep, t);
            let t = Instant::now();
            run_probs(&probs, &bands, &ctx.tables, &tabs, ctx.kernel, tw, false);
            add(&tm.dp, t);
            let t = Instant::now();
            let mut pidx = [[usize::MAX; 6]; N_STREAM];
            for (q, &(s, k)) in owner.iter().enumerate() { pidx[s][k] = q; }
            for (s, &(ck, al)) in ctx.streams.iter().enumerate() {
                if !active[s] { continue; }
                let (a, b) = (&la[s][..], &lb[s][..]);
                let (va, vb) = (&tw_.bars[ck][gi].vol[..], &yw.bars[ck][gi].vol[..]);
                let tbl = &ctx.tables[al];
                let (ug, ugid, e) = ungapped(a, b, tbl);
                let slot = &mut out[wbase + (gi * N_STREAM + s) * N_FIX..][..N_FIX];
                let r = &tw.alns[pidx[s][0]];
                global_features(a, b, va, vb, &tw.ops[pidx[s][0]], r, tbl, ug, e, &mut slot[..20]);
                let mut tmp = [f32::NAN; 20];
                for (k, off) in [(1usize, 20usize), (2, 26), (4, 42), (5, 48)] {
                    tmp.iter_mut().for_each(|v| *v = f32::NAN);
                    let q = pidx[s][k];
                    global_features(a, b, va, vb, &tw.ops[q], &tw.alns[q], tbl, ug, e, &mut tmp);
                    for (d, &si) in SUB6.iter().enumerate() { slot[off + d] = tmp[si]; }
                }
                let rs = &tw.alns[pidx[s][2]];
                let (nf, mf) = (a.len() as f64, b.len() as f64);
                slot[32] = (rs.i0 as f64 / nf - rs.j0 as f64 / mf) as f32;
                slot[33] = ((nf - rs.i1 as f64) / nf - (mf - rs.j1 as f64) / mf) as f32;
                local_features(a, b, va, &tw.ops[pidx[s][3]], &tw.alns[pidx[s][3]], &mut slot[34..40]);
                slot[40] = (ug / (MAX_S * nf)) as f32;
                slot[41] = ugid as f32;
                slot[54] = ((r.score - tw.alns[pidx[s][1]].score) as f64 / (MAX_S * nf)) as f32;
            }
            add(&tm.feats, t);
            // 朴素对照：时间钟这一档的分时成交量
            let (va, vb) = (&tw_.bars[0][gi].vol, &yw.bars[0][gi].vol);
            let o = wbase + 4 * N_STREAM * N_FIX + 4 * N_EV + gi * N_NAIVE;
            let act = va.iter().filter(|&&v| v > 0.0).count().min(vb.iter().filter(|&&v| v > 0.0).count());
            if act >= (l / 20).max(10) {
                out[o] = pearson(va, vb);
                out[o + 1] = pearson(&ranks(va), &ranks(vb));
                let com = |v: &[f64]| { let s: f64 = v.iter().sum();
                    v.iter().enumerate().map(|(i, x)| i as f64 * x).sum::<f64>() / s / v.len() as f64 };
                out[o + 2] = (com(va) - com(vb)) as f32;
            }
        }
        // 逐笔级：宽松全局 + 局部，斜带宽 100 笔
        let t = Instant::now();
        let (te, ye) = (&tw_.ev, &yw.ev);
        let mut probs: Vec<Prob> = Vec::new();
        let (mut bands, mut tabs, mut owner) = (Vec::new(), Vec::new(), Vec::new());
        for ev in 0..4 {
            let (a, b) = (&te.letters[ev][..], &ye.letters[ev][..]);
            let (n, m) = (a.len(), b.len());
            if n < 10 || m < 10 || n.max(m) > EV_MAX_RATIO * n.min(m) { continue; }
            if ev == EV_CXLMIX && !cxl_ok { continue; }
            for (k, mode) in [(0usize, Mode::Global), (1, Mode::Local)] {
                probs.push(Prob { a, b, grid: EV_GRID[ev], mode, open: OPEN_LOOSE, ext: EXT_LOOSE });
                bands.push(EV_BAND); tabs.push(ev); owner.push((ev, k));
            }
        }
        run_probs(&probs, &bands, &ctx.ev_tables, &tabs, ctx.kernel, tw, true);
        add(&tm.dp_ev, t);
        let t = Instant::now();
        let mut q = 0;
        while q < owner.len() {
            let ev = owner[q].0;
            let (a, b) = (&te.letters[ev][..], &ye.letters[ev][..]);
            let tbl = &ctx.ev_tables[ev];
            let (_, _, e) = ungapped(a, b, tbl);
            let slot = &mut out[wbase + 4 * N_STREAM * N_FIX + ev * N_EV..][..N_EV];
            let mut tmp = [f32::NAN; 20];
            global_features(a, b, &te.vols[ev], &ye.vols[ev], &tw.ops[q], &tw.alns[q], tbl, f64::NAN, e, &mut tmp);
            slot[..19].copy_from_slice(&tmp[..19]);
            local_features(a, b, &te.vols[ev], &tw.ops[q + 1], &tw.alns[q + 1], &mut slot[19..25]);
            q += 2;
        }
        add(&tm.feats, t);
    }
    tw.buf = buf; tw.la = la; tw.lb = lb;
    (out, n_full)
}

fn raw_names(streams: &[(usize, usize)]) -> Vec<String> {
    let mut v = Vec::with_capacity(N_RAW);
    for &(win, _) in WINDOWS.iter() {
        for gi in 0..4 {
            for &(ck, al) in streams {
                let p = format!("{win}_{}{}_{}", CLOCK_TAG[ck], gi + 1, ALPHA_NAMES[al]);
                for n in GL_NAMES { v.push(format!("{p}_gl_{n}")); }
                for n in SUB6_NAMES { v.push(format!("{p}_gt_{n}")); }
                for n in SUB6_NAMES { v.push(format!("{p}_sg_{n}")); }
                v.push(format!("{p}_sg_lead_shift")); v.push(format!("{p}_sg_trail_shift"));
                for n in LC_NAMES { v.push(format!("{p}_lc_{n}")); }
                v.push(format!("{p}_ug_score")); v.push(format!("{p}_ug_identity"));
                for n in SUB6_NAMES { v.push(format!("{p}_b8_{n}")); }
                for n in SUB6_NAMES { v.push(format!("{p}_b2_{n}")); }
                v.push(format!("{p}_glgt_score_diff"));
            }
        }
        for ev in 0..4 {
            let p = format!("{win}_ev_{}", EV_NAMES[ev]);
            for n in &GL_NAMES[..19] { v.push(format!("{p}_gl_{n}")); }
            for n in LC_NAMES { v.push(format!("{p}_lc_{n}")); }
        }
        for gi in 0..4 {
            for n in ["volcorr", "volrankcorr", "vol_com_shift"] { v.push(format!("{win}_t{}_naive_{n}", gi + 1)); }
        }
    }
    v
}

/// 组内超额：减去同组均值（组内有效值 ≥5 才算）。组号 <0 表示未知。
fn group_demean(col: &[f32], group: &[i32], out: &mut [f32]) {
    let mut s = [0f64; 64]; let mut c = [0f64; 64];
    for i in 0..col.len() { let g = group[i]; if g >= 0 && col[i].is_finite() { s[g as usize] += col[i] as f64; c[g as usize] += 1.0; } }
    for i in 0..col.len() {
        let g = group[i];
        out[i] = if g >= 0 && col[i].is_finite() && c[g as usize] >= 5.0 { (col[i] as f64 - s[g as usize] / c[g as usize]) as f32 } else { f32::NAN };
    }
}
fn deciles(v: &[f64]) -> Vec<i32> {
    let mut idx: Vec<usize> = (0..v.len()).filter(|&i| v[i].is_finite()).collect();
    idx.sort_by(|&x, &y| v[x].total_cmp(&v[y]).then(x.cmp(&y)));
    let mut g = vec![-1i32; v.len()];
    let n = idx.len();
    for (r, &i) in idx.iter().enumerate() { g[i] = ((r * 10) / n.max(1)).min(9) as i32; }
    g
}

/// 关键特征（做组内超额的列）在原始层里的列号
fn key_cols() -> Vec<usize> {
    let mut key_cols = Vec::new();
    for wi in 0..3 {
        for gi in 0..4 { for s in 0..N_STREAM { for &k in KEY_FIX.iter() { key_cols.push(wi * PER_WIN + (gi * N_STREAM + s) * N_FIX + k); } } }
        for ev in 0..4 { for &k in KEY_EV.iter() { key_cols.push(wi * PER_WIN + 4 * N_STREAM * N_FIX + ev * N_EV + k); } }
    }
    key_cols
}

/// 全部列名（42,924 列）：原始层 → 三种时段差 → 关键特征的三种组内超额
pub fn genedp_day_vs_day_align_names() -> Vec<String> {
    let st = streams();
    let mut names = raw_names(&st);
    assert_eq!(names.len(), N_RAW);
    for tag in ["pm_minus_all", "tail_minus_all", "tail_minus_pm"] {
        for c in 0..PER_WIN { let rest = names[c].splitn(2, '_').nth(1).unwrap().to_string(); names.push(format!("{tag}_{rest}")); }
    }
    let key = key_cols();
    for g in GROUPS { for &c in &key { let nm = format!("{}_{g}", names[c]); names.push(nm); } }
    names
}

pub const N_FACTORS: usize = N_RAW + 3 * PER_WIN + 3 * (3 * (4 * N_STREAM * KEY_FIX.len() + 4 * KEY_EV.len()));

/// 核心唯一真相源：读今天和昨天全市场原始数据 → 每股比对特征 → 时段差 → 组内超额。
/// 返回 (股票代码, 行优先展平的因子值)，每股 N_FACTORS 列。用当前 rayon 线程池并行。
pub fn compute_genedp_day_vs_day_align_full(date: i64) -> std::io::Result<(Vec<String>, Vec<f32>)> {
    let prev = crate::genedp_common::prev_level2_date(date)
        .ok_or_else(|| crate::genedp_common::invalid(format!("{date} 之前没有 Level2 数据目录，无法和昨天比较")))?;
    let codes = list_codes(date, "transaction")?;
    let ind = h5_row(&crate::genedp_common::vars_file("SzBa/industry.h5"), &crate::genedp_common::vars_file("SzBa/calendar_map.csv"),
        &format!("{}/symbol_map.csv", basic_dir()), date, &codes).map_err(|e| crate::genedp_common::invalid(format!("读行业失败: {e}")))?;
    let cap = h5_row(&crate::genedp_common::vars_file("DailyDerived/S_DQ_MV.h5"), &crate::genedp_common::vars_file("DailyDerived/calendar_map.csv"),
        &crate::genedp_common::vars_file("DailyDerived/symbol_map.csv"), date, &codes).map_err(|e| crate::genedp_common::invalid(format!("读市值失败: {e}")))?;
    let st = streams();
    let ctx = Ctx { tables: GRID.iter().map(|&g| build_score_table_grid(g)).collect(),
        ev_tables: EV_GRID.iter().map(|&g| build_score_table_grid(g)).collect(), streams: st, kernel: Kernel::Batch };
    let tm = Timers::default();
    let res: Vec<(Vec<f32>, f64)> = codes.par_iter()
        .map_init(ThreadWork::default, |tw, c| per_stock(c, date, prev, &ctx, tw, &tm)).collect();
    let nrow = res.len();
    let key = key_cols();
    let ncol = N_FACTORS;
    let liq = deciles(&res.iter().map(|r| r.1).collect::<Vec<_>>());
    let indg: Vec<i32> = ind.iter().map(|&v| if v.is_finite() && v >= 1.0 && v < 64.0 { v as i32 } else { -1 }).collect();
    let capg = deciles(&cap);
    let mut mat = vec![f32::NAN; nrow * ncol];
    mat.par_chunks_mut(ncol).zip(res.par_iter()).for_each(|(row, (raw, _))| {
        row[..N_RAW].copy_from_slice(raw);
        let mut o = N_RAW;
        for (a, b) in [(1usize, 0usize), (2, 0), (2, 1)] {
            for c in 0..PER_WIN { row[o] = raw[a * PER_WIN + c] - raw[b * PER_WIN + c]; o += 1; }
        }
    });
    // 组内超额按列算（每列一趟全市场）
    let base = N_RAW + 3 * PER_WIN;
    let groups = [&liq, &indg, &capg];
    let cols: Vec<Vec<f32>> = (0..3 * key.len()).into_par_iter().map(|q| {
        let (g, c) = (q / key.len(), key[q % key.len()]);
        let col: Vec<f32> = (0..nrow).map(|i| res[i].0[c]).collect();
        let mut o = vec![f32::NAN; nrow];
        group_demean(&col, groups[g], &mut o);
        o
    }).collect();
    for (q, col) in cols.iter().enumerate() { for i in 0..nrow { mat[i * ncol + base + q] = col[i]; } }
    Ok((codes, mat))
}

/// 单日全市场调试入口（50 线程）：返回 (股票代码, 展平因子值)，每股 N_FACTORS 列
#[pyfunction]
pub fn py_genedp_day_vs_day_align(py: Python<'_>, date: i64) -> PyResult<(Vec<String>, Vec<f32>)> {
    py.allow_threads(|| crate::genedp_common::run_in_50_threads(|| compute_genedp_day_vs_day_align_full(date)))
        .and_then(|r| r)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("{e:?}")))
}

#[pyfunction]
pub fn py_genedp_day_vs_day_align_names() -> Vec<String> {
    genedp_day_vs_day_align_names()
}
