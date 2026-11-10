//! 方向 3「镜像比对」第三版：单日全市场（沙箱）。
//!
//! 用法: align_mirror_v3 <日期> [--engine base|fast] [--limit N] [--out 目录]
//!   base = 冻结的未提速版（打乱基准逐条算）；fast = 提速版（打乱基准 32 条一组并排算）。
//!   两种引擎的因子值必须逐字节相同。
//! 输出：{out}/v3_{日期}_{引擎}.f32（股票 × 列，float32 小端）、.names.json、.codes.json、.meta.json
//!
//! 列结构（全部单因子，不做多因子合成）：
//!   主表 114 组合 × 41 | 三种时段差 38 × 35 × 3 | 参数变体 | 跨日倒放 | 同行业倒放 | 横截面 | 参考列
//! 凡依赖撤单数据的列，名字含 cxl；沪市 20211105 前、深市 20160104 前这些列写空值。
mod batch;
mod dp;
mod full;
mod prep;
mod reader;

use dp::{Best, Cfg, Scratch, Tab, BASE};
use prep::*;
use rayon::prelude::*;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::time::Instant;

const MAX_THREADS: usize = 50;
const N_NULL: usize = 6;
const SEGS: usize = 16;
const MIN_ACTIVE: usize = 20;

// ==================== 计时 ====================
static T_READ: AtomicU64 = AtomicU64::new(0);
static T_PREP: AtomicU64 = AtomicU64::new(0);
static T_JOBS: AtomicU64 = AtomicU64::new(0);
static T_FULL: AtomicU64 = AtomicU64::new(0);
fn acc(c: &AtomicU64, t: Instant) {
    c.fetch_add(t.elapsed().as_micros() as u64, Relaxed);
}

// ==================== 打乱 ====================
struct Rng(u64);
impl Rng {
    #[inline]
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    #[inline]
    fn below(&mut self, n: usize) -> usize {
        ((self.next() >> 32).wrapping_mul(n as u64) >> 32) as usize
    }
}
fn shuffle_range(v: &mut [u8], r: &mut Rng) {
    for i in (1..v.len()).rev() {
        let j = r.below(i + 1);
        v.swap(i, j);
    }
}
fn shuffled(src: &[u8], segs: usize, r: &mut Rng) -> Vec<u8> {
    let mut out = src.to_vec();
    if segs <= 1 {
        shuffle_range(&mut out, r);
        return out;
    }
    let n = out.len();
    let len = n.div_ceil(segs).max(1);
    let mut i = 0;
    while i < n {
        let e = (i + len).min(n);
        shuffle_range(&mut out[i..e], r);
        i = e;
    }
    out
}
fn seed_of(code: &str, tag: u64) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in code.bytes() {
        h = (h ^ b as u64).wrapping_mul(0x100000001b3);
    }
    h ^ tag.wrapping_mul(0x9E3779B97F4A7C15)
}

// ==================== 比对任务（只算最高分的部分） ====================
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Kind {
    Nested,
    Forward,
    Rect,
}
struct Job {
    kind: Kind,
    a: Vec<u8>,
    s: Vec<u8>,
    tab: Tab,
    o: i32,
    e: i32,
    minsep: usize,
}
#[derive(Default)]
struct Jobs {
    v: Vec<Job>,
}
impl Jobs {
    fn push(&mut self, j: Job) -> usize {
        self.v.push(j);
        self.v.len() - 1
    }
    /// 6 条打乱序列的任务，返回首个下标
    #[allow(clippy::too_many_arguments)]
    fn nulls(&mut self, kind: Kind, a: &[u8], s: &[u8], tab: &Tab, cfg: &Cfg, segs: usize, seed: u64) -> usize {
        let mut r = Rng(seed);
        let first = self.v.len();
        for _ in 0..N_NULL {
            let sh = shuffled(s, segs, &mut r);
            self.push(Job { kind, a: a.to_vec(), s: sh, tab: *tab, o: cfg.o, e: cfg.e, minsep: cfg.minsep });
        }
        first
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Engine {
    Base,
    Fast,
}

fn run_jobs(jobs: &Jobs, eng: Engine) -> Vec<i32> {
    let t = Instant::now();
    let mut out = vec![0i32; jobs.v.len()];
    match eng {
        Engine::Base => {
            for (k, j) in jobs.v.iter().enumerate() {
                out[k] = match j.kind {
                    Kind::Nested => dp::nested_score(&j.s, &j.tab, j.o, j.e, j.minsep),
                    Kind::Forward => dp::forward_score(&j.s, &j.tab, j.o, j.e, j.minsep),
                    Kind::Rect => dp::rect_score(&j.a, &j.s, &j.tab, j.o),
                };
            }
        }
        Engine::Fast => {
            let mut groups: BTreeMap<(Kind, usize, usize, i32, i32, usize), Vec<usize>> = BTreeMap::new();
            for (k, j) in jobs.v.iter().enumerate() {
                groups.entry((j.kind, j.a.len(), j.s.len(), j.o, j.e, j.minsep)).or_default().push(k);
            }
            for (key, idx) in groups {
                for ch in idx.chunks(batch::L) {
                    let ss: Vec<&[u8]> = ch.iter().map(|&k| jobs.v[k].s.as_slice()).collect();
                    let tt: Vec<batch::Tab16> = ch.iter().map(|&k| batch::tab16(&jobs.v[k].tab)).collect();
                    let r = match key.0 {
                        Kind::Nested => batch::nested_score_batch(&ss, &tt, key.3 as i16, key.4 as i16, key.5),
                        Kind::Forward => batch::forward_score_batch(&ss, &tt, key.3 as i16, key.4 as i16, key.5),
                        Kind::Rect => {
                            let aa: Vec<&[u8]> = ch.iter().map(|&k| jobs.v[k].a.as_slice()).collect();
                            batch::rect_score_batch(&aa, &ss, &tt, key.3 as i16)
                        }
                    };
                    for (q, &k) in ch.iter().enumerate() {
                        out[k] = r[q];
                    }
                }
            }
        }
    }
    acc(&T_JOBS, t);
    out
}

// ==================== 完整路径比对任务 ====================
struct FullJob {
    shape: u8, // 0 嵌套 1 同向 2 两序列
    a: Vec<u8>,
    s: Vec<u8>,
    vol: Vec<f32>,
    tab: Tab,
    o: i32,
    e: i32,
    minsep: usize,
    center: bool,
}
#[derive(Default)]
struct FullJobs {
    v: Vec<FullJob>,
}
impl FullJobs {
    #[allow(clippy::too_many_arguments)]
    fn push(&mut self, shape: u8, a: &[u8], sq: &Seq, tab: Tab, o: i32, e: i32, minsep: usize, center: bool) -> usize {
        self.v.push(FullJob { shape, a: a.to_vec(), s: sq.s.clone(), vol: sq.vol.clone(), tab, o, e, minsep, center });
        self.v.len() - 1
    }
}
struct FullRes {
    best: Best,
    center: Vec<i32>,
}
fn run_full(jobs: &FullJobs, eng: Engine) -> Vec<FullRes> {
    let t = Instant::now();
    let mut out: Vec<Option<FullRes>> = (0..jobs.v.len()).map(|_| None).collect();
    if eng == Engine::Fast && full::available() {
        let mut groups: BTreeMap<(u8, usize, usize, i32, i32, usize), Vec<usize>> = BTreeMap::new();
        for (k, j) in jobs.v.iter().enumerate() {
            groups.entry((j.shape, j.a.len(), j.s.len(), j.o, j.e, j.minsep)).or_default().push(k);
        }
        for (key, idx) in groups {
            for ch in idx.chunks(batch::L) {
                let lanes: Vec<full::Lane> = ch
                    .iter()
                    .map(|&k| {
                        let j = &jobs.v[k];
                        full::Lane { a: &j.a, s: &j.s, vol: &j.vol, tab: &j.tab, center: j.center }
                    })
                    .collect();
                let r = full::run_group(key.0, &lanes, key.3, key.4, key.5);
                for (q, x) in r.into_iter().enumerate() {
                    out[ch[q]] = Some(FullRes { best: x.best, center: x.center });
                }
            }
        }
    } else {
        let mut sc = Scratch::default();
        for (k, j) in jobs.v.iter().enumerate() {
            let best = match j.shape {
                0 => dp::nested_full(&j.s, &j.vol, &j.tab, j.o, j.e, j.minsep, j.center, &mut sc),
                1 => dp::forward_full(&j.s, &j.vol, &j.tab, j.o, j.e, j.minsep, &mut sc),
                _ => dp::rect_full(&j.a, &j.s, &j.vol, &j.tab, j.o, &mut sc),
            };
            let center = if j.shape == 0 && j.center { sc.center.clone() } else { Vec::new() };
            out[k] = Some(FullRes { best, center });
        }
    }
    acc(&T_FULL, t);
    out.into_iter().map(|x| x.unwrap()).collect()
}

fn mean_std(v: &[i32]) -> (f32, f32) {
    let n = v.len() as f32;
    let m = v.iter().map(|&x| x as f32).sum::<f32>() / n;
    let s = (v.iter().map(|&x| (x as f32 - m).powi(2)).sum::<f32>() / n).sqrt();
    (m, s)
}
fn null_stat(res: &[i32], first: usize) -> (f32, f32) {
    mean_std(&res[first..first + N_NULL])
}
#[inline]
fn z_of(x: i32, ms: (f32, f32)) -> f32 {
    (x as f32 - ms.0) / ms.1.max(1.0)
}

// ==================== 特征清单 ====================
const FEATS: [&str; 41] = [
    "bits", "perletter", "zfull", "zseg", "excess", "stemlen", "matchfrac", "gapfrac", "center", "asym",
    "buyfirst", "armvol", "npeaks", "conc", "meancent", "top3cent", "pathvol", "matchvol", "bitsR", "bitsC",
    "bitsD", "zsegR", "zsegC", "zsegD", "bfR", "bfC", "bfD", "dzR", "dzC", "dzD", "dbfR", "dbfC", "dbfD", "midS",
    "midR", "volmedS", "volmedR", "highS", "highR", "lowS", "lowR",
];
/// 做时段差的特征：去掉 6 个规则差（27..33）
fn diff_feats() -> Vec<usize> {
    (0..27).chain(33..41).collect()
}
const VARIANTS: [(&str, Cfg); 10] = [
    ("gap1", Cfg { o: 1, e: 1, ..BASE }),
    ("gap4", Cfg { o: 4, e: 4, ..BASE }),
    ("sep1", Cfg { minsep: 1, ..BASE }),
    ("sep12", Cfg { minsep: 12, ..BASE }),
    ("sizeswap", Cfg { size_swap: true, ..BASE }),
    ("tierstrict", Cfg { tier_strict: true, ..BASE }),
    ("thA", Cfg { theta_mode: 1, ..BASE }),
    ("thB", Cfg { theta_mode: 2, ..BASE }),
    ("aff31", Cfg { o: 3, e: 1, ..BASE }),
    ("aff51", Cfg { o: 5, e: 1, ..BASE }),
];
/// 方向字母表只有一档，大小翻转与严格分档和基础版完全相同，不重复输出
fn variant_applies(a: usize, vi: usize) -> bool {
    !(a == A_DIR && (vi == 4 || vi == 5))
}
const VAR_FEATS: [(&str, usize); 6] =
    [("bits", 0), ("zseg", 1), ("center", 2), ("stemlen", 3), ("buyfirst", 4), ("matchfrac", 5)];
const X_ALPHAS: [usize; 6] = [0, 1, 2, 4, 5, 7];
const X_CLOCKS: [Clock; 2] = [Clock::Time, Clock::Vol];
const XDAY_FEATS: [&str; 10] =
    ["bits", "perletter", "zseg", "stemlen", "matchfrac", "center", "buyfirst", "zsegR", "zsegC", "zsegD"];
const IND_FEATS: [&str; 7] = ["bits", "perletter", "zseg", "stemlen", "matchfrac", "center", "buyfirst"];
const CS_N: usize = 18;

struct Combo {
    w: usize,
    c: usize,
    a: usize,
}
fn combos() -> Vec<Combo> {
    let mut v = Vec::new();
    for w in 0..3 {
        for (c, ck) in CLOCKS.iter().enumerate() {
            for &a in alphas_of(*ck) {
                v.push(Combo { w, c, a });
            }
        }
    }
    v
}
/// (时钟, 字母表) 组合，共 38 个
fn ac_pairs() -> Vec<(usize, usize)> {
    let mut v = Vec::new();
    for (c, ck) in CLOCKS.iter().enumerate() {
        for &a in alphas_of(*ck) {
            v.push((c, a));
        }
    }
    v
}
struct VarSpec {
    w: usize,
    c: usize,
    a: usize,
    vi: usize,
}
fn var_specs() -> Vec<VarSpec> {
    let mut v = Vec::new();
    for w in 0..3 {
        for c in 0..2 {
            for a in 0..10 {
                for vi in 0..VARIANTS.len() {
                    if variant_applies(a, vi) {
                        v.push(VarSpec { w, c, a, vi });
                    }
                }
            }
        }
    }
    v
}
struct XSpec {
    w: usize,
    c: usize,
    a: usize,
}
fn x_specs() -> Vec<XSpec> {
    let mut v = Vec::new();
    for w in 0..3 {
        for c in 0..2 {
            for &a in &X_ALPHAS {
                v.push(XSpec { w, c, a });
            }
        }
    }
    v
}

fn cname(c: usize) -> &'static str {
    CLOCK_NAMES[c]
}
fn build_names() -> (Vec<String>, usize, usize) {
    let mut n = Vec::new();
    let cs = combos();
    for k in &cs {
        for f in FEATS {
            n.push(format!("mir_{}_{}_{}_{}", ALPHA_NAMES[k.a], cname(k.c), WINS[k.w].name, f));
        }
    }
    let df = diff_feats();
    for (c, a) in ac_pairs() {
        for pair in ["pmall", "tailall", "tailpm"] {
            for &f in &df {
                n.push(format!("wd_{}_{}_{}_{}", ALPHA_NAMES[a], cname(c), pair, FEATS[f]));
            }
        }
    }
    for v in var_specs() {
        for (f, _) in VAR_FEATS {
            n.push(format!("var_{}_{}_{}_{}_{}", ALPHA_NAMES[v.a], cname(v.c), WINS[v.w].name, VARIANTS[v.vi].0, f));
        }
    }
    for x in x_specs() {
        for f in XDAY_FEATS {
            n.push(format!("xday_{}_{}_{}_{}", ALPHA_NAMES[x.a], cname(x.c), WINS[x.w].name, f));
        }
    }
    let n_stock = n.len();
    for x in x_specs() {
        for f in IND_FEATS {
            n.push(format!("ind_{}_{}_{}_{}", ALPHA_NAMES[x.a], cname(x.c), WINS[x.w].name, f));
        }
    }
    let n_before_cs = n.len();
    for k in &cs {
        for f in &FEATS[..CS_N] {
            for op in ["csrank", "peerx"] {
                n.push(format!("{}_{}_{}_{}_{}", op, ALPHA_NAMES[k.a], cname(k.c), WINS[k.w].name, f));
            }
        }
    }
    for w in WINS {
        n.push(format!("naive_anticorr_{}", w.name));
        n.push(format!("naive_fbdiff_{}", w.name));
        n.push(format!("audit_logamt_{}", w.name));
    }
    (n, n_stock, n_before_cs)
}

// ==================== 第 4 层：路径特征 ====================
fn center_peaks(center: &[i32], n: usize) -> (f32, f32, f32, f32) {
    let sep = (n / 5).max(4) as i64;
    let mut v: Vec<(i32, i64)> =
        center.iter().enumerate().filter(|(_, &x)| x > 0).map(|(i, &x)| (x, i as i64)).collect();
    if v.is_empty() {
        return (0.0, 0.0, 0.0, 0.0);
    }
    v.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
    let total: f64 = v.iter().map(|x| x.0 as f64).sum();
    let top = v[0].0;
    let mut picked: Vec<(i32, i64)> = Vec::with_capacity(8);
    for &(x, i) in &v {
        if picked.len() >= 8 {
            break;
        }
        if picked.iter().all(|&(_, k)| (k - i).abs() >= sep) {
            picked.push((x, i));
        }
    }
    let npeaks = picked.iter().filter(|p| p.0 * 2 >= top && p.0 >= 6).count() as f32;
    let top3 = picked.iter().take(3).map(|p| p.0 as f32).sum::<f32>() / 3.0 / n as f32;
    let conc = (v.iter().take(5).map(|x| x.0 as f64).sum::<f64>() / total) as f32;
    let mean = (total / (2 * n) as f64 / n as f64) as f32;
    (npeaks, conc, mean, top3)
}
fn bf(b: &Best) -> f32 {
    if b.score > 0 {
        b.c.bs as f32 / b.c.matched.max(1) as f32
    } else {
        0.0
    }
}
fn active(s: &[u8]) -> usize {
    s.iter().filter(|&&x| x >= 2).count()
}

struct MainJobs {
    mf: usize,
    ms: usize,
    rs: usize,
    cs: usize,
    ds: usize,
}
fn main_jobs(jobs: &mut Jobs, s: &[u8], t: usize, seed: u64) -> MainJobs {
    let tm = dp::make_tab(t, &BASE, true);
    let ti = dp::make_tab(t, &BASE, false);
    MainJobs {
        mf: jobs.nulls(Kind::Nested, &[], s, &tm, &BASE, 1, seed ^ 0x11),
        ms: jobs.nulls(Kind::Nested, &[], s, &tm, &BASE, SEGS, seed ^ 0x22),
        rs: jobs.nulls(Kind::Nested, &[], s, &ti, &BASE, SEGS, seed ^ 0x33),
        cs: jobs.nulls(Kind::Forward, &[], s, &tm, &BASE, SEGS, seed ^ 0x44),
        ds: jobs.nulls(Kind::Forward, &[], s, &ti, &BASE, SEGS, seed ^ 0x55),
    }
}

#[allow(clippy::too_many_arguments)]
fn analyze_main(sq: &Seq, t: usize, mj: &MainJobs, res: &[i32], fr: &[FullRes], fi: &[usize; 4], sc: &mut Scratch, out: &mut [f32]) {
    let s = &sq.s;
    let n = s.len();
    let nf = n as f32;
    let cfg = BASE;
    let tm = dp::make_tab(t, &cfg, true);
    let ti = dp::make_tab(t, &cfg, false);
    let cells = (n * (n - 1) / 2) as f64;
    let lam_m = dp::karlin_lambda(s, s, &tm);
    let lam_r = dp::karlin_lambda(s, s, &ti);
    let m = fr[fi[0]].best;
    let (npk, conc, meanc, top3) = center_peaks(&fr[fi[0]].center, n);
    let nfull = null_stat(res, mj.mf);
    let nseg = null_stat(res, mj.ms);
    let total: f64 = sq.vol.iter().map(|&x| x as f64).sum();
    out[0] = dp::bits(m.score, lam_m, cells);
    out[1] = m.score as f32 / nf;
    out[2] = z_of(m.score, nfull);
    let zseg = z_of(m.score, nseg);
    out[3] = zseg;
    out[4] = (m.score as f32 - nfull.0) / nf;
    if m.score > 0 {
        let c = m.c;
        let pairs = c.pairs.max(1) as f32;
        out[5] = c.pairs as f32 / nf;
        out[6] = c.matched as f32 / pairs;
        out[7] = c.gaps as f32 / (pairs + c.gaps as f32);
        out[8] = (c.i0 as f32 + c.j0 as f32) * 0.5 / nf;
        let (le, ll) = ((c.i0 as usize - m.i + 1) as f32, (m.j - c.j0 as usize + 1) as f32);
        out[9] = (ll - le) / (ll + le);
        out[10] = bf(&m);
        let vs = |a: usize, b: usize| -> f64 { sq.vol[a..=b].iter().map(|&x| x as f64).sum() };
        let ve = vs(m.i, c.i0 as usize);
        let vl = vs(c.j0 as usize, m.j);
        if ve > 0.0 && vl > 0.0 {
            out[11] = (vl / ve).ln() as f32;
        }
        if total > 0.0 {
            out[16] = ((ve + vl) / total) as f32;
            out[17] = (c.mv / total) as f32;
        }
    } else {
        for x in [5usize, 6, 7, 9, 10] {
            out[x] = 0.0;
        }
        if total > 0.0 {
            out[16] = 0.0;
            out[17] = 0.0;
        }
    }
    out[12] = npk;
    out[13] = conc;
    out[14] = meanc;
    out[15] = top3;
    let (r, c, d) = (fr[fi[1]].best, fr[fi[2]].best, fr[fi[3]].best);
    out[18] = dp::bits(r.score, lam_r, cells);
    out[19] = dp::bits(c.score, lam_m, cells);
    out[20] = dp::bits(d.score, lam_r, cells);
    let (zr, zc, zd) = (z_of(r.score, null_stat(res, mj.rs)), z_of(c.score, null_stat(res, mj.cs)), z_of(d.score, null_stat(res, mj.ds)));
    out[21] = zr;
    out[22] = zc;
    out[23] = zd;
    let bm = out[10];
    let (br, bc, bd) = (bf(&r), bf(&c), bf(&d));
    out[24] = br;
    out[25] = bc;
    out[26] = bd;
    out[27] = zseg - zr;
    out[28] = zseg - zc;
    out[29] = zseg - zd;
    out[30] = bm - br;
    out[31] = bm - bc;
    out[32] = bm - bd;
    // 固定切点：字母中点、量的中点、最高价、最低价
    let mut cuts = [n / 2; 4];
    let (mut cum, mut got) = (0f64, false);
    let (mut hi, mut lo) = (f32::MIN, f32::MAX);
    for i in 0..n {
        cum += sq.vol[i] as f64;
        if !got && total > 0.0 && cum >= total * 0.5 {
            cuts[1] = i;
            got = true;
        }
        let p = sq.px[i];
        if p.is_finite() {
            if p > hi {
                hi = p;
                cuts[2] = i;
            }
            if p < lo {
                lo = p;
                cuts[3] = i;
            }
        }
    }
    for (k, &cut) in cuts.iter().enumerate() {
        let (v, reach) = dp::extend_from_split(s, cut, &tm, cfg.o, sc);
        out[33 + 2 * k] = v as f32 / nf;
        out[34 + 2 * k] = reach as f32 / nf;
    }
}

fn analyze_variant(sq: &Seq, t: usize, cfg: &Cfg, first: usize, res: &[i32], m: &Best, out: &mut [f32]) {
    let s = &sq.s;
    let n = s.len();
    let tm = dp::make_tab(t, cfg, true);
    out[0] = dp::bits(m.score, dp::karlin_lambda(s, s, &tm), (n * (n - 1) / 2) as f64);
    out[1] = z_of(m.score, null_stat(res, first));
    if m.score > 0 {
        out[2] = (m.c.i0 as f32 + m.c.j0 as f32) * 0.5 / n as f32;
        out[3] = m.c.pairs as f32 / n as f32;
        out[4] = bf(m);
        out[5] = m.c.matched as f32 / m.c.pairs.max(1) as f32;
    } else {
        out[3] = 0.0;
        out[4] = 0.0;
        out[5] = 0.0;
    }
}

/// 两序列（跨日/同行业）：规则 0=M 倒序互换 1=R 只倒序 2=C 只互换 3=D 都不做
fn rect_a(x: &[u8], rule: usize) -> Vec<u8> {
    if rule <= 1 {
        x.iter().rev().copied().collect()
    } else {
        x.to_vec()
    }
}
fn rect_tab(t: usize, rule: usize) -> Tab {
    dp::make_tab(t, &BASE, rule == 0 || rule == 2)
}
/// 两序列的 M 特征 7 个
fn analyze_rect_m(a: &[u8], sq: &Seq, t: usize, null_first: usize, res: &[i32], m: &Best, out: &mut [f32]) {
    let tab = rect_tab(t, 0);
    let nb = sq.s.len() as f32;
    out[0] = dp::bits(m.score, dp::karlin_lambda(a, &sq.s, &tab), (a.len() * sq.s.len()) as f64);
    out[1] = m.score as f32 / nb;
    out[2] = z_of(m.score, null_stat(res, null_first));
    if m.score > 0 {
        out[3] = m.c.pairs as f32 / nb;
        out[4] = m.c.matched as f32 / m.c.pairs.max(1) as f32;
        out[5] = (m.c.j0 as f32 + m.j as f32) * 0.5 / nb;
        out[6] = bf(m);
    } else {
        out[3] = 0.0;
        out[4] = 0.0;
        out[6] = 0.0;
    }
}

// ==================== 每股：第一阶段 ====================
struct DayBars {
    bars: Vec<Vec<Bars>>, // [窗口][时钟]
    cxl_ok: bool,
}
fn day_bars(code: &str, date: i64) -> Option<(DayBars, Vec<WinData>)> {
    let t = Instant::now();
    let evs = reader::read_events(code, date).ok()?;
    let snaps = reader::read_snaps(code, date).unwrap_or_default();
    acc(&T_READ, t);
    let t = Instant::now();
    let day = prep::prep(evs, snaps, cxl_available(code, date));
    if day.trs.is_empty() {
        return None;
    }
    let wds: Vec<WinData> = WINS.iter().map(|w| window(&day, w)).collect();
    let bars = WINS
        .iter()
        .enumerate()
        .map(|(wi, w)| CLOCKS.iter().map(|&c| build_bars(&wds[wi], w, c)).collect())
        .collect();
    acc(&T_PREP, t);
    Some((DayBars { bars, cxl_ok: day.cxl_ok }, wds))
}
fn seq_ok(sq: &Seq, a: usize, cxl_ok: bool) -> bool {
    (!is_cxl(a) || cxl_ok) && sq.s.len() >= 2 && active(&sq.s) >= MIN_ACTIVE
}

struct StockOut {
    vals: Vec<f32>,
    refs: [f32; 9],
    /// 同行业模板用：36 组字母（无效为 None）
    ind: Vec<Option<Seq>>,
    ok: bool,
}

fn corr(a: &[f64], b: &[f64]) -> f32 {
    let n = a.len().min(b.len());
    if n < 5 {
        return f32::NAN;
    }
    let ma = a[..n].iter().sum::<f64>() / n as f64;
    let mb = b[..n].iter().sum::<f64>() / n as f64;
    let (mut c, mut va, mut vb) = (0.0, 0.0, 0.0);
    for i in 0..n {
        let (x, y) = (a[i] - ma, b[i] - mb);
        c += x * y;
        va += x * x;
        vb += y * y;
    }
    if va <= 0.0 || vb <= 0.0 {
        f32::NAN
    } else {
        (c / (va * vb).sqrt()) as f32
    }
}

fn per_stock(code: &str, date: i64, prev: Option<i64>, n_stock: usize, eng: Engine) -> StockOut {
    let mut vals = vec![f32::NAN; n_stock];
    let mut refs = [f32::NAN; 9];
    let Some((db, wds)) = day_bars(code, date) else {
        return StockOut { vals, refs, ind: (0..36).map(|_| None).collect(), ok: false };
    };
    let yb = prev.and_then(|p| day_bars(code, p).map(|x| x.0));
    let cs = combos();
    let mut jobs = Jobs::default();
    let mut fj = FullJobs::default();
    // ---- 主表 ----
    let mut main_seq: Vec<Option<(Seq, MainJobs, [usize; 4])>> = Vec::with_capacity(cs.len());
    for (ci, k) in cs.iter().enumerate() {
        let sq = encode(&db.bars[k.w][k.c], k.a, CLOCKS[k.c], 0);
        if seq_ok(&sq, k.a, db.cxl_ok) {
            let t = alpha_tiers(k.a);
            let mj = main_jobs(&mut jobs, &sq.s, t, seed_of(code, ci as u64 * 16));
            let (tm, ti) = (dp::make_tab(t, &BASE, true), dp::make_tab(t, &BASE, false));
            let (o, e, m) = (BASE.o, BASE.e, BASE.minsep);
            let fi = [
                fj.push(0, &[], &sq, tm, o, e, m, true),
                fj.push(0, &[], &sq, ti, o, e, m, false),
                fj.push(1, &[], &sq, tm, o, e, m, false),
                fj.push(1, &[], &sq, ti, o, e, m, false),
            ];
            main_seq.push(Some((sq, mj, fi)));
        } else {
            main_seq.push(None);
        }
    }
    // ---- 参数变体 ----
    let vs = var_specs();
    let mut var_seq: Vec<Option<(Seq, usize, usize)>> = Vec::with_capacity(vs.len());
    for (vi, v) in vs.iter().enumerate() {
        let cfg = VARIANTS[v.vi].1;
        let sq = encode(&db.bars[v.w][v.c], v.a, CLOCKS[v.c], cfg.theta_mode);
        if seq_ok(&sq, v.a, db.cxl_ok) {
            let tm = dp::make_tab(alpha_tiers(v.a), &cfg, true);
            let f = jobs.nulls(Kind::Nested, &[], &sq.s, &tm, &cfg, SEGS, seed_of(code, 100_000 + vi as u64 * 16));
            let g = fj.push(0, &[], &sq, tm, cfg.o, cfg.e, cfg.minsep, false);
            var_seq.push(Some((sq, f, g)));
        } else {
            var_seq.push(None);
        }
    }
    // ---- 跨日倒放 ----
    let xs = x_specs();
    let ycxl = prev.map(|p| cxl_available(code, p)).unwrap_or(false);
    let mut x_seq: Vec<Option<(Seq, Vec<u8>, [usize; 3], [usize; 4], usize)>> = Vec::with_capacity(xs.len());
    for (xi, x) in xs.iter().enumerate() {
        let item = (|| {
            let yb = yb.as_ref()?;
            let sq = encode(&db.bars[x.w][x.c], x.a, CLOCKS[x.c], 0);
            let ys = encode(&yb.bars[x.w][x.c], x.a, CLOCKS[x.c], 0);
            if !seq_ok(&sq, x.a, db.cxl_ok) || !seq_ok(&ys, x.a, ycxl && yb.cxl_ok) {
                return None;
            }
            let t = alpha_tiers(x.a);
            let mut act = [0usize; 3];
            let mut nul = [0usize; 4];
            for rule in 0..4 {
                let a = rect_a(&ys.s, rule);
                let tab = rect_tab(t, rule);
                if rule > 0 {
                    act[rule - 1] = jobs.push(Job { kind: Kind::Rect, a: a.clone(), s: sq.s.clone(), tab, o: BASE.o, e: BASE.o, minsep: 0 });
                }
                let cfg = Cfg { e: BASE.o, minsep: 0, ..BASE };
                nul[rule] = jobs.nulls(Kind::Rect, &a, &sq.s, &tab, &cfg, SEGS, seed_of(code, 200_000 + (xi * 4 + rule) as u64 * 16));
            }
            let am = rect_a(&ys.s, 0);
            let g = fj.push(2, &am, &sq, rect_tab(t, 0), BASE.o, BASE.o, 0, false);
            Some((sq, am, act, nul, g))
        })();
        x_seq.push(item);
    }
    let res = run_jobs(&jobs, eng);
    drop(jobs);
    let fr = run_full(&fj, eng);
    drop(fj);
    // ---- 完整路径与特征 ----
    let mut sc = Scratch::default();
    let nf = FEATS.len();
    for (ci, item) in main_seq.iter().enumerate() {
        if let Some((sq, mj, fi)) = item {
            analyze_main(sq, alpha_tiers(cs[ci].a), mj, &res, &fr, fi, &mut sc, &mut vals[ci * nf..(ci + 1) * nf]);
        }
    }
    let mut pos = cs.len() * nf;
    // 时段差
    let df = diff_feats();
    let idx_of = |w: usize, c: usize, a: usize| cs.iter().position(|k| k.w == w && k.c == c && k.a == a).unwrap();
    for (c, a) in ac_pairs() {
        let base = [idx_of(0, c, a), idx_of(1, c, a), idx_of(2, c, a)];
        for (x, y) in [(1usize, 0usize), (2, 0), (2, 1)] {
            for &f in &df {
                vals[pos] = vals[base[x] * nf + f] - vals[base[y] * nf + f];
                pos += 1;
            }
        }
    }
    for (vi, item) in var_seq.iter().enumerate() {
        if let Some((sq, first, g)) = item {
            let v = &vs[vi];
            analyze_variant(sq, alpha_tiers(v.a), &VARIANTS[v.vi].1, *first, &res, &fr[*g].best, &mut vals[pos..pos + 6]);
        }
        pos += 6;
    }
    for (xi, item) in x_seq.iter().enumerate() {
        if let Some((sq, am, act, nul, g)) = item {
            let t = alpha_tiers(xs[xi].a);
            let o = &mut vals[pos..pos + 10];
            analyze_rect_m(am, sq, t, nul[0], &res, &fr[*g].best, &mut o[..7]);
            for r in 0..3 {
                o[7 + r] = z_of(res[act[r]], null_stat(&res, nul[r + 1]));
            }
        }
        pos += 10;
    }
    assert_eq!(pos, n_stock);
    // ---- 参考列：朴素前后段对照与成交额 ----
    for wi in 0..3 {
        let bars = &db.bars[wi][0].bars;
        let x: Vec<f64> = bars
            .iter()
            .map(|b| {
                let t = b.bv + b.sv;
                if t > 0.0 {
                    ((b.bv - b.sv) / t) as f64
                } else {
                    0.0
                }
            })
            .collect();
        let m = x.len() / 2;
        let first: Vec<f64> = x[..m].iter().rev().copied().collect();
        refs[wi * 3] = -corr(&first, &x[m..]);
        let side = |r: &[Bar]| -> f64 {
            let (mut b, mut s) = (0f64, 0f64);
            for k in r {
                b += k.bv as f64;
                s += k.sv as f64;
            }
            if b + s > 0.0 {
                (b - s) / (b + s)
            } else {
                f64::NAN
            }
        };
        refs[wi * 3 + 1] = (side(&bars[..m]) - side(&bars[m..])) as f32;
        let amt: f64 = wds[wi].trs.iter().map(|t| t.v as f64 * t.p as f64).sum();
        refs[wi * 3 + 2] = if amt > 0.0 { amt.ln() as f32 } else { f32::NAN };
    }
    // ---- 同行业模板用的字母 ----
    let ind = xs
        .iter()
        .map(|x| {
            let sq = encode(&db.bars[x.w][x.c], x.a, CLOCKS[x.c], 0);
            if seq_ok(&sq, x.a, db.cxl_ok) {
                Some(sq)
            } else {
                None
            }
        })
        .collect();
    StockOut { vals, refs, ind, ok: true }
}

// ==================== 第二阶段：同行业模板倒放 ====================
fn template(counts: &[[u32; dp::K]], own: Option<&[u8]>) -> Vec<u8> {
    counts
        .iter()
        .enumerate()
        .map(|(i, c)| {
            let mut c = *c;
            if let Some(o) = own {
                if o[i] >= 2 {
                    c[o[i] as usize] -= 1;
                }
            }
            let mut best = 0usize;
            let mut bv = 0u32;
            for (k, &v) in c.iter().enumerate().skip(2) {
                if v > bv {
                    bv = v;
                    best = k;
                }
            }
            if bv == 0 {
                0
            } else {
                best as u8
            }
        })
        .collect()
}

fn industry_stage(codes: &[String], outs: &[StockOut], inds: &[Option<i64>], eng: Engine) -> Vec<Vec<f32>> {
    let xs = x_specs();
    // 计数表：[规格][行业] → 每个位置每个字母的股票数（只数非空闲、有效的股票）
    let mut ind_ids: Vec<i64> = inds.iter().flatten().copied().collect();
    ind_ids.sort();
    ind_ids.dedup();
    let ind_pos = |x: i64| ind_ids.binary_search(&x).unwrap();
    let counts: Vec<Vec<Vec<[u32; dp::K]>>> = (0..xs.len())
        .map(|xi| {
            let n = outs.iter().find_map(|o| o.ind.get(xi).and_then(|s| s.as_ref()).map(|s| s.s.len())).unwrap_or(0);
            let mut cnt = vec![vec![[0u32; dp::K]; n]; ind_ids.len()];
            for (k, o) in outs.iter().enumerate() {
                if let (Some(g), Some(Some(sq))) = (inds[k], o.ind.get(xi)) {
                    let gi = ind_pos(g);
                    for (i, &l) in sq.s.iter().enumerate() {
                        if l >= 2 {
                            cnt[gi][i][l as usize] += 1;
                        }
                    }
                }
            }
            cnt
        })
        .collect();
    codes
        .par_iter()
        .enumerate()
        .map(|(k, code)| {
            let mut out = vec![f32::NAN; xs.len() * IND_FEATS.len()];
            let Some(g) = inds[k] else { return out };
            let gi = ind_pos(g);
            let mut jobs = Jobs::default();
            let mut fj = FullJobs::default();
            let mut items = Vec::new();
            for (xi, x) in xs.iter().enumerate() {
                let Some(Some(sq)) = outs[k].ind.get(xi) else {
                    items.push(None);
                    continue;
                };
                let tpl = template(&counts[xi][gi], Some(&sq.s));
                if active(&tpl) < MIN_ACTIVE {
                    items.push(None);
                    continue;
                }
                let t = alpha_tiers(x.a);
                let a = rect_a(&tpl, 0);
                let cfg = Cfg { e: BASE.o, minsep: 0, ..BASE };
                let f = jobs.nulls(Kind::Rect, &a, &sq.s, &rect_tab(t, 0), &cfg, SEGS, seed_of(code, 300_000 + xi as u64 * 16));
                let g = fj.push(2, &a, sq, rect_tab(t, 0), BASE.o, BASE.o, 0, false);
                items.push(Some((a, f, g)));
            }
            let res = run_jobs(&jobs, eng);
            let fr = run_full(&fj, eng);
            for (xi, it) in items.iter().enumerate() {
                if let (Some((a, f, g)), Some(Some(sq))) = (it, outs[k].ind.get(xi)) {
                    let t = alpha_tiers(xs[xi].a);
                    analyze_rect_m(a, sq, t, *f, &res, &fr[*g].best, &mut out[xi * 7..xi * 7 + 7]);
                }
            }
            out
        })
        .collect()
}

// ==================== 第五层：横截面 ====================
fn cs_rank(col: &[f32]) -> Vec<f32> {
    let mut idx: Vec<usize> = (0..col.len()).filter(|&i| col[i].is_finite()).collect();
    idx.sort_by(|&a, &b| col[a].total_cmp(&col[b]).then(a.cmp(&b)));
    let mut out = vec![f32::NAN; col.len()];
    let m = idx.len();
    let mut i = 0;
    while i < m {
        let mut j = i + 1;
        while j < m && col[idx[j]] == col[idx[i]] {
            j += 1;
        }
        let r = ((i + j - 1) as f32 / 2.0 + 0.5) / m as f32;
        for &k in &idx[i..j] {
            out[k] = r;
        }
        i = j;
    }
    out
}
fn peer_excess(col: &[f32], bucket: &[usize], nb: usize) -> Vec<f32> {
    let mut groups: Vec<Vec<f32>> = vec![Vec::new(); nb];
    for i in 0..col.len() {
        if col[i].is_finite() && bucket[i] < nb {
            groups[bucket[i]].push(col[i]);
        }
    }
    let med: Vec<f32> = groups
        .iter_mut()
        .map(|g| {
            if g.is_empty() {
                return f32::NAN;
            }
            g.sort_by(|a, b| a.total_cmp(b));
            g[g.len() / 2]
        })
        .collect();
    (0..col.len()).map(|i| if bucket[i] < nb { col[i] - med[bucket[i]] } else { f32::NAN }).collect()
}

// ==================== 股票范围、日历、行业 ====================
fn list_codes(date: i64) -> std::io::Result<Vec<String>> {
    use std::io::{Error, ErrorKind};
    let root = std::env::var("RUST_PYFUNC_LEVEL2_ROOT").unwrap_or("/ssd_data/stock".into());
    let basic = std::env::var("RUST_PYFUNC_BASIC_INFO_DIR").unwrap_or("/ssd_data/data/basic_info".into());
    let dir = std::path::Path::new(&root).join(date.to_string()).join("transaction");
    let invalid = |e| Error::new(ErrorKind::InvalidData, e);
    let mut reader = csv::Reader::from_path(std::path::Path::new(&basic).join("symbol_map.csv")).map_err(invalid)?;
    let col = reader
        .headers()
        .map_err(invalid)?
        .iter()
        .position(|h| h == "symbol")
        .ok_or_else(|| Error::new(ErrorKind::InvalidData, "股票表缺少 symbol 列"))?;
    let mut stocks = std::collections::BTreeSet::new();
    for r in reader.records() {
        let r = r.map_err(invalid)?;
        let code = r.get(col).unwrap_or("").trim();
        if code.is_empty() || code.len() > 6 || !code.bytes().all(|b| b.is_ascii_digit()) {
            return Err(Error::new(ErrorKind::InvalidData, "股票表含无效代码"));
        }
        stocks.insert(format!("{code:0>6}"));
    }
    if stocks.is_empty() || !dir.is_dir() {
        return Err(Error::new(ErrorKind::InvalidData, "股票白名单为空或原始数据目录不存在"));
    }
    Ok(stocks.into_iter().filter(|c| dir.join(format!("{c}_{date}_transaction.csv")).is_file()).collect())
}

fn prev_trading_day(date: i64) -> Option<i64> {
    let s = std::fs::read_to_string("/nas197/binary/stock/sz_alpha/basic_info/calendar.csv").ok()?;
    s.lines().filter_map(|l| l.trim().parse::<i64>().ok()).filter(|&d| d < date).max()
}

/// 申万一级行业（当日值）；读不到返回空表
fn industries(date: i64) -> BTreeMap<String, i64> {
    let mut m = BTreeMap::new();
    let Ok(text) = std::fs::read_to_string("/nas197/binary/stock/sz_alpha/csv/vars/Base/SW_IND_CODE.csv") else {
        return m;
    };
    let mut lines = text.lines();
    let Some(header) = lines.next() else { return m };
    let key = format!("{}-{:02}-{:02}", date / 10000, date / 100 % 100, date % 100);
    let Some(col) = header.split(',').position(|h| h == key) else { return m };
    for l in lines {
        let mut it = l.split(',');
        let code = it.next().unwrap_or("");
        if let Some(v) = it.nth(col - 1) {
            if let Ok(x) = v.trim().parse::<f64>() {
                if x.is_finite() {
                    m.insert(code.split('.').next().unwrap_or("").to_string(), x as i64);
                }
            }
        }
    }
    m
}

/// 单线程测内核：同一批随机序列，标量逐条 vs 批量
fn bench() {
    let mut r = Rng(7);
    for n in [180usize, 474, 1000] {
        let seqs: Vec<Vec<u8>> = (0..32).map(|_| (0..n).map(|_| r.below(8) as u8).collect()).collect();
        let tab = dp::make_tab(3, &BASE, true);
        let t16 = batch::tab16(&tab);
        let refs: Vec<&[u8]> = seqs.iter().map(|v| v.as_slice()).collect();
        let tabs = vec![t16; 32];
        let cells = (32 * n * n / 2) as f64;
        for kind in 0..3 {
            let t = Instant::now();
            let a: Vec<i32> = seqs.iter().map(|s| match kind { 0 => dp::nested_score(s, &tab, 2, 2, 4), 1 => dp::forward_score(s, &tab, 2, 2, 4), _ => dp::rect_score(s, s, &tab, 2) }).collect();
            let ts = t.elapsed().as_secs_f64();
            let t = Instant::now();
            let b = match kind { 0 => batch::nested_score_batch(&refs, &tabs, 2, 2, 4), 1 => batch::forward_score_batch(&refs, &tabs, 2, 2, 4), _ => batch::rect_score_batch(&refs, &refs, &tabs, 2) };
            let tb = t.elapsed().as_secs_f64();
            let cf = if kind == 2 { 2.0 } else { 1.0 };
            println!("n={n} kind={kind} scalar={:.2}ns/cell batch={:.2}ns/cell same={}", ts / cells / cf * 1e9, tb / cells / cf * 1e9, a == b);
        }
        let vol = vec![1.0f32; n];
        let mut sc = Scratch::default();
        let t = Instant::now();
        let mut k = 0i64;
        for s in &seqs { k += dp::nested_full(s, &vol, &tab, 2, 2, 4, true, &mut sc).score as i64; }
        let a = t.elapsed().as_secs_f64() / cells * 1e9;
        let t = Instant::now();
        for s in &seqs { k += dp::nested_full(s, &vol, &tab, 2, 2, 4, false, &mut sc).score as i64; }
        let b = t.elapsed().as_secs_f64() / cells * 1e9;
        let t = Instant::now();
        for s in &seqs { k += dp::forward_full(s, &vol, &tab, 2, 2, 4, &mut sc).score as i64; }
        let c = t.elapsed().as_secs_f64() / cells * 1e9;
        let t = Instant::now();
        for s in &seqs { k += dp::rect_full(s, s, &vol, &tab, 2, &mut sc).score as i64; }
        let d = t.elapsed().as_secs_f64() / cells / 2.0 * 1e9;
        println!("n={n} full ns/cell: nested+center={a:.2} nested={b:.2} forward={c:.2} rect={d:.2} ({k})");
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.get(1).map(|s| s.as_str()) == Some("bench") {
        bench();
        return;
    }
    let date: i64 = args.get(1).expect("用法: align_mirror_v3 <日期> [--engine base|fast] [--limit N] [--out 目录]").parse().unwrap();
    let mut eng = Engine::Fast;
    let mut limit: Option<usize> = None;
    let mut out_dir = "results/v3".to_string();
    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--engine" => {
                eng = if args[i + 1] == "base" { Engine::Base } else { Engine::Fast };
                i += 2;
            }
            "--limit" => {
                limit = Some(args[i + 1].parse().unwrap());
                i += 2;
            }
            "--out" => {
                out_dir = args[i + 1].clone();
                i += 2;
            }
            _ => panic!("未知参数 {}", args[i]),
        }
    }
    rayon::ThreadPoolBuilder::new().num_threads(MAX_THREADS).build_global().unwrap();
    let t_all = Instant::now();
    let mut codes = match list_codes(date) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("股票范围或原始目录检查失败: {e}");
            std::process::exit(2);
        }
    };
    if let Some(k) = limit {
        codes.truncate(k);
    }
    let (names, n_stock, n_before_cs) = build_names();
    let prev = prev_trading_day(date);
    let ind_map = industries(date);
    eprintln!(
        "date={date} engine={} codes={} columns={} prev_day={:?} industry_labels={}",
        if eng == Engine::Base { "base" } else { "fast" },
        codes.len(),
        names.len(),
        prev,
        ind_map.len()
    );
    let t1 = Instant::now();
    let outs: Vec<StockOut> = codes.par_iter().map(|c| per_stock(c, date, prev, n_stock, eng)).collect();
    let stage1 = t1.elapsed().as_secs_f64();
    let t2 = Instant::now();
    let inds: Vec<Option<i64>> = codes.iter().zip(&outs).map(|(c, o)| if o.ok { ind_map.get(c).copied() } else { None }).collect();
    let ind_vals = industry_stage(&codes, &outs, &inds, eng);
    let stage2 = t2.elapsed().as_secs_f64();
    let t3 = Instant::now();
    let n = codes.len();
    let nf = FEATS.len();
    let cs = combos();
    let mut order: Vec<usize> = (0..n).filter(|&i| outs[i].refs[2].is_finite()).collect();
    order.sort_by(|&a, &b| outs[a].refs[2].total_cmp(&outs[b].refs[2]).then(a.cmp(&b)));
    const NB: usize = 20;
    let mut bucket = vec![NB; n];
    for (r, &i) in order.iter().enumerate() {
        bucket[i] = (r * NB / order.len().max(1)).min(NB - 1);
    }
    let pairs: Vec<(usize, usize)> = (0..cs.len()).flat_map(|ci| (0..CS_N).map(move |f| (ci, f))).collect();
    let extras: Vec<(Vec<f32>, Vec<f32>)> = pairs
        .par_iter()
        .map(|&(ci, f)| {
            let col: Vec<f32> = outs.iter().map(|o| o.vals[ci * nf + f]).collect();
            (cs_rank(&col), peer_excess(&col, &bucket, NB))
        })
        .collect();
    let ncol = names.len();
    let mut mat = vec![f32::NAN; n * ncol];
    mat.par_chunks_mut(ncol).enumerate().for_each(|(i, row)| {
        row[..n_stock].copy_from_slice(&outs[i].vals);
        row[n_stock..n_before_cs].copy_from_slice(&ind_vals[i]);
        let mut p = n_before_cs;
        for (a, b) in &extras {
            row[p] = a[i];
            row[p + 1] = b[i];
            p += 2;
        }
        row[p..p + 9].copy_from_slice(&outs[i].refs);
        assert_eq!(p + 9, ncol, "列数不恒定");
    });
    let stage3 = t3.elapsed().as_secs_f64();
    let t4 = Instant::now();
    std::fs::create_dir_all(&out_dir).unwrap();
    let tag = format!("{out_dir}/v3_{date}_{}", if eng == Engine::Base { "base" } else { "fast" });
    let mut bytes = Vec::with_capacity(mat.len() * 4);
    for x in &mat {
        bytes.extend_from_slice(&x.to_le_bytes());
    }
    std::fs::write(format!("{tag}.f32"), &bytes).unwrap();
    std::fs::write(format!("{tag}.names.json"), serde_json::to_string(&names).unwrap()).unwrap();
    std::fs::write(format!("{tag}.codes.json"), serde_json::to_string(&codes).unwrap()).unwrap();
    std::fs::write(
        format!("{tag}.meta.json"),
        serde_json::to_string(&serde_json::json!({"date": date, "n_codes": n, "n_cols": ncol, "prev_day": prev, "industry_labels": ind_map.len()})).unwrap(),
    )
    .unwrap();
    let wr = t4.elapsed().as_secs_f64();
    eprintln!(
        "timing: total={:.1}s stage1_per_stock={:.1}s stage2_industry={:.1}s stage3_cross_section={:.2}s write={:.1}s | thread_sum: read={:.0}s prep={:.0}s score_jobs={:.0}s full_paths={:.0}s",
        t_all.elapsed().as_secs_f64(),
        stage1,
        stage2,
        stage3,
        wr,
        T_READ.load(Relaxed) as f64 / 1e6,
        T_PREP.load(Relaxed) as f64 / 1e6,
        T_JOBS.load(Relaxed) as f64 / 1e6,
        T_FULL.load(Relaxed) as f64 / 1e6
    );
}
