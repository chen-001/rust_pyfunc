//! 方向 3「镜像比对」单日全市场探索（沙箱）。
//!
//! 想法：DNA 的反向互补。把一段行为序列倒序、买卖互换，再和后面的序列比对；
//! 得分高 = 后面的行为在“倒放”前面的行为（例如建仓后的派发）。
//!
//! 分层：
//!   第 1 层 字母化：  3 个时段 × 3 种时钟（时间/成交量/订单数）× 7 种字母表
//!   第 2 层 镜像变换：M 倒序+互换（主角）、R 只倒序、C 只互换、D 都不做（三个对照）
//!   第 3 层 比对：    嵌套动态规划一张表同时覆盖所有镜像中心；固定切点的外扩比对作补充
//!   第 4 层 路径特征：得分/比特分/打乱基准 z 值/臂长/错位/中心/先买后卖方向/多发夹…
//!   第 5 层 横截面：  全市场排名、同成交额桶超额
//!
//! 时段（相对 09:30、下午已前移 90 分钟，与正式库 anneal_regime 系列相同）：
//!   all  09:30-14:57（0..14220 秒）；pm 13:00-14:57（7200..14220，实际 117 分钟）；
//!   tail 14:27-14:57（12420..14220，最后 30 分钟连续竞价）。
//!   reader 只保留 09:30-11:30 与 13:00-14:57，三个时段都不含开盘/收盘集合竞价。
//!
//! 构建: cargo build --release -j 50
//! 运行: ./target/release/align_mirror_sandbox 20241231 > results/mirror_20241231.json
mod fast_csv_reader;
use fast_csv_reader::{read_market_fast, read_trade_fast, MarketRecord, TradeRecord};
use rayon::prelude::*;
use serde::Serialize;
use std::time::Instant;

const MAX_THREADS: usize = 50;
/// 打乱基准的次数（全打乱 / 分段内打乱各做这么多次）
const N_NULL: usize = 6;
const DAY_OPEN_SEC: i64 = 9 * 3600 + 30 * 60;
const US: i64 = 1_000_000;

// ==================== 时段与时钟 ====================

#[derive(Clone, Copy)]
struct Win {
    name: &'static str,
    lo: i64,
    hi: i64,
    bar_sec: i64,
    n_bar: usize,
}
const WINS: [Win; 3] = [
    Win { name: "all", lo: 0, hi: 14_220, bar_sec: 30, n_bar: 300 },
    Win { name: "pm", lo: 7_200, hi: 14_220, bar_sec: 30, n_bar: 200 },
    Win { name: "tail", lo: 12_420, hi: 14_220, bar_sec: 10, n_bar: 120 },
];

#[derive(Clone, Copy, PartialEq)]
enum Clock {
    Time,
    Vol,
    Ord,
}
const CLOCKS: [(Clock, &str); 3] = [(Clock::Time, "tm"), (Clock::Vol, "vo"), (Clock::Ord, "od")];

/// 字母表 id 与名称；a6 盘口只用时间时钟
const ALPHAS: [&str; 7] = ["dir", "int", "osize", "tempo", "jump", "price", "book"];

fn alpha_tiers(a: usize) -> usize {
    match a {
        0 => 1,
        1..=4 => 3,
        _ => 2,
    }
}

// ==================== 第 1 层：原始事件 → 分箱 → 字母 ====================

#[derive(Clone, Copy)]
struct Tr {
    t: i64, // 距 09:30 的微秒（下午已前移）
    d: i8,  // 1 主买 -1 主卖 0 其他
    v: f32,
    p: f32,
    aid: i64, // 主动方订单编号
    new_ord: bool,
    ord: u32,
}

#[derive(Clone, Copy, Default)]
struct Bar {
    bv: f32,
    sv: f32,
    n: u32,
    orders: u32,
    t0: i64,
    t1: i64,
    p0: f32,
    p1: f32,
    jump: f64,
    njump: u32,
}

impl Bar {
    #[inline]
    fn add(&mut self, tr: &Tr, v: f32, jump: Option<f64>) {
        if self.n == 0 {
            self.t0 = tr.t;
            self.p0 = tr.p;
        }
        self.t1 = tr.t;
        self.p1 = tr.p;
        self.n += 1;
        if tr.d > 0 {
            self.bv += v;
        } else if tr.d < 0 {
            self.sv += v;
        }
        if tr.new_ord {
            self.orders += 1;
            if let Some(j) = jump {
                self.jump += j;
                self.njump += 1;
            }
        }
    }
    #[inline]
    fn vol(&self) -> f32 {
        self.bv + self.sv
    }
}

fn prep_trades(raw: &[TradeRecord]) -> Vec<Tr> {
    let mut idx: Vec<usize> = (0..raw.len()).collect();
    idx.sort_by_key(|&i| (raw[i].time_us, raw[i].index));
    let mut out = Vec::with_capacity(raw.len());
    let mut prev = i64::MIN;
    let mut ord = 0u32;
    for &i in &idx {
        let r = &raw[i];
        if !(r.volume > 0.0 && r.price > 0.0) {
            continue;
        }
        let sec = (r.time_us.div_euclid(US)).rem_euclid(86_400) - DAY_OPEN_SEC;
        let t = sec * US + r.time_us.rem_euclid(US);
        let (d, aid) = match r.flag {
            66 => (1i8, r.bid_order),
            83 => (-1i8, r.ask_order),
            _ => (0i8, 0i64),
        };
        let new_ord = d != 0 && aid != prev;
        if new_ord {
            prev = aid;
            ord += 1;
        }
        out.push(Tr { t, d, v: r.volume as f32, p: r.price as f32, aid, new_ord, ord });
    }
    out
}

fn window_slice<'a>(trs: &'a [Tr], w: &Win) -> &'a [Tr] {
    let lo = w.lo * US;
    let hi = (w.hi + 1) * US;
    let a = trs.partition_point(|x| x.t < lo);
    let b = trs.partition_point(|x| x.t < hi);
    &trs[a..b]
}

fn n_bars(w: &Win, c: Clock) -> usize {
    match c {
        Clock::Time => ((w.hi - w.lo) / w.bar_sec) as usize,
        _ => w.n_bar,
    }
}

fn build_bars(trs: &[Tr], w: &Win, c: Clock) -> Vec<Bar> {
    let n = n_bars(w, c);
    let mut bars = vec![Bar::default(); n];
    if trs.is_empty() {
        return bars;
    }
    let mut prev_aid: Option<i64> = None;
    let jump_of = |tr: &Tr, prev: &mut Option<i64>| -> Option<f64> {
        if !tr.new_ord {
            return None;
        }
        let j = prev.map(|p| (tr.aid - p).abs() as f64);
        *prev = Some(tr.aid);
        j
    };
    match c {
        Clock::Time => {
            let wid = w.bar_sec * US;
            let lo = w.lo * US;
            for tr in trs {
                let i = (((tr.t - lo) / wid) as usize).min(n - 1);
                let j = jump_of(tr, &mut prev_aid);
                bars[i].add(tr, tr.v, j);
            }
        }
        Clock::Vol => {
            let total: f64 = trs.iter().map(|t| t.v as f64).sum();
            if total <= 0.0 {
                return bars;
            }
            let wid = total / n as f64;
            let mut cum = 0.0f64;
            for tr in trs {
                let j = jump_of(tr, &mut prev_aid);
                let (mut c0, c1) = (cum, cum + tr.v as f64);
                cum = c1;
                // 一笔大单跨多根时按比例摊开；下标只增不减，浮点误差也不会原地打转
                let mut i = ((c0 / wid) as usize).min(n - 1);
                let mut first = true;
                loop {
                    let end = if i == n - 1 { c1 } else { (((i + 1) as f64) * wid).min(c1) };
                    let part = (end - c0).max(0.0) as f32;
                    let mut t2 = *tr;
                    if !first {
                        t2.new_ord = false;
                    }
                    bars[i].add(&t2, part, if first { j } else { None });
                    first = false;
                    c0 = end;
                    if c0 >= c1 || i == n - 1 {
                        break;
                    }
                    i += 1;
                }
            }
        }
        Clock::Ord => {
            let o0 = trs[0].ord.saturating_sub(1);
            let o1 = trs[trs.len() - 1].ord;
            let tot = ((o1 - o0).max(1)) as u64;
            for tr in trs {
                let k = (tr.ord.saturating_sub(o0).saturating_sub(1)) as u64;
                let i = ((k * n as u64 / tot) as usize).min(n - 1);
                let j = jump_of(tr, &mut prev_aid);
                bars[i].add(tr, tr.v, j);
            }
        }
    }
    bars
}

/// 盘口：每根时间箱末的（买量合计 - 卖量合计）相对上一有效箱的变化；None = 无快照
fn build_book(mkt: &[(i64, f32)], w: &Win) -> Vec<Option<f32>> {
    let n = n_bars(w, Clock::Time);
    let mut last: Vec<Option<f32>> = vec![None; n];
    let lo = w.lo * US;
    let hi = (w.hi + 1) * US;
    let wid = w.bar_sec * US;
    let mut base: Option<f32> = None;
    for &(t, lv) in mkt {
        if t < lo || t >= hi {
            continue;
        }
        if base.is_none() {
            base = Some(lv);
        }
        let i = (((t - lo) / wid) as usize).min(n - 1);
        last[i] = Some(lv);
    }
    let mut out = vec![None; n];
    let mut prev = base;
    for i in 0..n {
        if let (Some(l), Some(p)) = (last[i], prev) {
            out[i] = Some(l - p);
            prev = Some(l);
        }
    }
    out
}

fn tier_thresholds(mut v: Vec<f32>, t: usize) -> Vec<f32> {
    if v.is_empty() || t <= 1 {
        return Vec::new();
    }
    v.sort_by(|a, b| a.total_cmp(b));
    (1..t).map(|k| v[v.len() * k / t]).collect()
}
#[inline]
fn tier_of(x: f32, th: &[f32]) -> usize {
    th.iter().take_while(|&&q| x >= q).count()
}
/// 字母：0 空闲、1 中性、2+2*tier 买/上方、3+2*tier 卖/下方
#[inline]
fn letter(side: i8, tier: usize) -> u8 {
    match side {
        0 => 1,
        s if s > 0 => (2 + 2 * tier) as u8,
        _ => (3 + 2 * tier) as u8,
    }
}

/// 把分箱变成字母序列。theta 是买卖失衡判定方向的门槛
fn encode(bars: &[Bar], book: &[Option<f32>], a: usize, c: Clock, theta: f32) -> Vec<u8> {
    let t = alpha_tiers(a);
    if a == 6 {
        // 盘口字母：买卖挂单差的变化方向 × 变化幅度
        let mut abs: Vec<f32> = book.iter().flatten().map(|x| x.abs()).collect();
        abs.sort_by(|x, y| x.total_cmp(y));
        let med = if abs.is_empty() { 0.0 } else { abs[abs.len() / 2] };
        return book
            .iter()
            .map(|x| match x {
                None => 0,
                Some(d) => {
                    if d.abs() < 0.3 * med || *d == 0.0 {
                        1
                    } else {
                        letter(if *d > 0.0 { 1 } else { -1 }, (d.abs() >= med) as usize)
                    }
                }
            })
            .collect();
    }
    let metric = |b: &Bar| -> f32 {
        let span = ((b.t1 - b.t0) as f32 / 1e6).max(1e-3);
        match a {
            1 | 5 => {
                if c == Clock::Time || a == 5 {
                    b.vol()
                } else {
                    1.0 / span
                }
            }
            2 => b.vol() / (b.orders.max(1) as f32),
            3 => {
                if b.n >= 2 {
                    span / (b.n - 1) as f32
                } else {
                    f32::NAN
                }
            }
            4 => {
                if b.njump > 0 {
                    (b.jump / b.njump as f64) as f32
                } else {
                    f32::NAN
                }
            }
            _ => 0.0,
        }
    };
    let vals: Vec<f32> = bars
        .iter()
        .filter(|b| b.n > 0)
        .map(&metric)
        .filter(|x| x.is_finite())
        .collect();
    let th = tier_thresholds(vals, t);
    bars.iter()
        .map(|b| {
            if b.n == 0 {
                return 0u8;
            }
            let side: i8 = if a == 5 {
                let d = b.p1 - b.p0;
                if d.abs() < 1e-6 {
                    0
                } else if d > 0.0 {
                    1
                } else {
                    -1
                }
            } else {
                let tot = b.bv + b.sv;
                if tot <= 0.0 {
                    0
                } else {
                    let im = (b.bv - b.sv) / tot;
                    if im > theta {
                        1
                    } else if im < -theta {
                        -1
                    } else {
                        0
                    }
                }
            };
            let m = metric(&b);
            let tier = if t <= 1 {
                0
            } else if m.is_finite() {
                tier_of(m, &th)
            } else {
                t / 2
            };
            letter(side, tier)
        })
        .collect()
}

// ==================== 第 2 层：计分表与镜像变换 ====================

const K: usize = 8;
type Tab = [[i32; K]; K];

#[derive(Clone, Copy)]
struct Cfg {
    gap: i32,
    minsep: usize,
    size_swap: bool,
    tier_strict: bool,
    theta: f32,
}
const BASE: Cfg = Cfg { gap: 2, minsep: 4, size_swap: false, tier_strict: false, theta: 0.2 };

/// 互换：买卖对调；size_swap 时再把大小档也翻过来（“大单建、小单出”）
#[inline]
fn comp(a: usize, t: usize, size_swap: bool) -> usize {
    if a < 2 {
        return a;
    }
    let tier = (a - 2) / 2;
    let side = (a - 2) % 2;
    let tier2 = if size_swap { t - 1 - tier } else { tier };
    2 + 2 * tier2 + (1 - side)
}

fn base_score(a: usize, b: usize, cfg: &Cfg) -> i32 {
    if a == 0 || b == 0 {
        return if a == b { 0 } else { -1 };
    }
    if a == 1 || b == 1 {
        return if a == b { 0 } else { -1 };
    }
    let (ta, sa) = ((a - 2) / 2, (a - 2) % 2);
    let (tb, sb) = ((b - 2) / 2, (b - 2) % 2);
    if sa != sb {
        return -3;
    }
    let d = ta.abs_diff(tb);
    if cfg.tier_strict {
        return if d == 0 { 2 } else { -1 };
    }
    match d {
        0 => 2,
        1 => 1,
        _ => 0,
    }
}

/// 生成“前面的字母 a 与后面的字母 b（已做变换）”的打分表
fn make_tab(t: usize, cfg: &Cfg, use_comp: bool) -> Tab {
    let mut tab = [[0i32; K]; K];
    for (a, row) in tab.iter_mut().enumerate() {
        for (b, cell) in row.iter_mut().enumerate() {
            let tb = if use_comp { comp(b, t.max(1), cfg.size_swap) } else { b };
            *cell = if tb < K { base_score(a, tb, cfg) } else { 0 };
        }
    }
    tab
}

/// 比对打分的 Karlin-Altschul λ（BLAST 的归一常数），只用字母组成解出；算不出返回 None
fn karlin_lambda(seq: &[u8], tab: &Tab) -> Option<f64> {
    let mut cnt = [0f64; K];
    for &a in seq {
        cnt[a as usize] += 1.0;
    }
    let n = seq.len() as f64;
    let f = |l: f64| -> f64 {
        let mut s = 0.0;
        for a in 0..K {
            for b in 0..K {
                if cnt[a] > 0.0 && cnt[b] > 0.0 {
                    s += cnt[a] * cnt[b] / (n * n) * (l * tab[a][b] as f64).exp();
                }
            }
        }
        s - 1.0
    };
    let mut e = 0.0;
    let mut has_pos = false;
    for a in 0..K {
        for b in 0..K {
            if cnt[a] > 0.0 && cnt[b] > 0.0 {
                e += cnt[a] * cnt[b] / (n * n) * tab[a][b] as f64;
                has_pos |= tab[a][b] > 0;
            }
        }
    }
    if e >= 0.0 || !has_pos {
        return None;
    }
    let (mut lo, mut hi) = (1e-4, 1.0);
    while f(hi) < 0.0 {
        hi *= 2.0;
        if hi > 64.0 {
            return None;
        }
    }
    for _ in 0..40 {
        let m = (lo + hi) / 2.0;
        if f(m) < 0.0 {
            lo = m
        } else {
            hi = m
        }
    }
    Some((lo + hi) / 2.0)
}

fn bits_of(score: i32, lambda: Option<f64>, n: usize) -> f32 {
    match lambda {
        Some(l) if n >= 2 => {
            let cells = (n * (n - 1) / 2) as f64;
            ((l * score as f64 - cells.ln()) / std::f64::consts::LN_2) as f32
        }
        _ => f32::NAN,
    }
}

// ==================== 第 3 层：比对动态规划 ====================

#[derive(Clone, Copy, Default)]
struct Cell {
    v: i32,
    i0: u16,
    j0: u16,
    pairs: u16,
    matched: u16,
    gaps: u16,
    bs: i16, // 配上的字母里，先买后卖 +1，先卖后买 -1
}

#[derive(Clone, Copy, Default)]
struct Best {
    score: i32,
    i: usize,
    j: usize,
    c: Cell,
}

struct Scratch {
    a: Vec<i32>,
    b: Vec<i32>,
    ja: Vec<u16>,
    jb: Vec<u16>,
    ca: Vec<Cell>,
    cb: Vec<Cell>,
    center: Vec<i32>,
    tmp: Vec<u8>,
}
impl Scratch {
    fn new() -> Self {
        Scratch {
            a: vec![],
            b: vec![],
            ja: vec![],
            jb: vec![],
            ca: vec![],
            cb: vec![],
            center: vec![],
            tmp: vec![],
        }
    }
}

/// 嵌套比对（M、R 用）：一张表覆盖所有镜像中心。
/// G(i,j) = 以 (i,j) 为最外层配对的、从中心向外生长的最佳局部得分；只返回最高分，供打乱基准用。
fn nested_score(s: &[u8], tab: &Tab, gap: i32, minsep: usize, sc: &mut Scratch) -> i32 {
    let n = s.len();
    sc.a.clear();
    sc.a.resize(n + 1, 0);
    sc.b.clear();
    sc.b.resize(n + 1, 0);
    let (mut prev, mut cur) = (&mut sc.a, &mut sc.b);
    let mut best = 0;
    for i in (0..n).rev() {
        let row = &tab[s[i] as usize];
        let start = i + minsep.max(1);
        if start >= n {
            std::mem::swap(&mut prev, &mut cur);
            continue;
        }
        cur[start - 1] = 0;
        let mut left = 0;
        for j in start..n {
            let v = (prev[j - 1] + row[s[j] as usize]).max(prev[j] - gap).max(left - gap).max(0);
            cur[j] = v;
            left = v;
            if v > best {
                best = v;
            }
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    best
}

/// 嵌套比对完整版：带路径信息，并记录每个镜像中心的最高分。center 长度 2n，下标 i0+j0。
fn nested_full(s: &[u8], tab: &Tab, gap: i32, minsep: usize, sc: &mut Scratch) -> Best {
    let n = s.len();
    sc.ca.clear();
    sc.ca.resize(n + 1, Cell::default());
    sc.cb.clear();
    sc.cb.resize(n + 1, Cell::default());
    sc.center.clear();
    sc.center.resize(2 * n + 1, 0);
    let (mut prev, mut cur) = (&mut sc.ca, &mut sc.cb);
    let center = &mut sc.center;
    let mut best = Best::default();
    for i in (0..n).rev() {
        let row = &tab[s[i] as usize];
        let start = i + minsep.max(1);
        if start >= n {
            std::mem::swap(&mut prev, &mut cur);
            continue;
        }
        cur[start - 1] = Cell::default();
        let mut left = Cell::default();
        for j in start..n {
            let p = row[s[j] as usize];
            let dg = prev[j - 1];
            let up = prev[j];
            let dv = dg.v + p;
            let uv = up.v - gap;
            let lv = left.v - gap;
            let mut c = Cell::default();
            if dv >= uv && dv >= lv {
                if dv > 0 {
                    c = if dg.v > 0 {
                        dg
                    } else {
                        Cell { i0: i as u16, j0: j as u16, ..Cell::default() }
                    };
                    c.v = dv;
                    c.pairs += 1;
                    if p > 0 {
                        c.matched += 1;
                        // 早的是买侧（偶数偏移）记 +1
                        let a = s[i] as usize;
                        if a >= 2 {
                            c.bs += if (a - 2) % 2 == 0 { 1 } else { -1 };
                        }
                    }
                }
            } else if uv >= lv {
                if uv > 0 {
                    c = up;
                    c.v = uv;
                    c.gaps += 1;
                }
            } else if lv > 0 {
                c = left;
                c.v = lv;
                c.gaps += 1;
            }
            cur[j] = c;
            left = c;
            if c.v > 0 {
                let k = c.i0 as usize + c.j0 as usize;
                if c.v > center[k] {
                    center[k] = c.v;
                }
                if c.v > best.score {
                    best = Best { score: c.v, i, j, c };
                }
            }
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    best
}

/// 同向比对（C、D 用）：前后两段都从左往右读，要求两段不重叠。
fn forward_score(s: &[u8], tab: &Tab, gap: i32, minsep: usize, sc: &mut Scratch) -> i32 {
    let n = s.len();
    for v in [&mut sc.a, &mut sc.b] {
        v.clear();
        v.resize(n + 1, 0);
    }
    for v in [&mut sc.ja, &mut sc.jb] {
        v.clear();
        v.resize(n + 1, 0);
    }
    let (mut prev, mut cur) = (&mut sc.a, &mut sc.b);
    let (mut pj, mut cj) = (&mut sc.ja, &mut sc.jb);
    let mut best = 0;
    let ms = minsep.max(1);
    for i in 0..n {
        let start = i + ms;
        if start >= n {
            break;
        }
        let row = &tab[s[i] as usize];
        cur[start - 1] = 0;
        let (mut left, mut lj) = (0i32, 0u16);
        for j in start..n {
            let p = row[s[j] as usize];
            let (dg, dj) = if j >= 1 && i >= 1 && j - 1 >= i - 1 + ms { (prev[j - 1], pj[j - 1]) } else { (0, 0) };
            let (up, uj) = if i >= 1 { (prev[j], pj[j]) } else { (0, 0) };
            // 两段不重叠：起点的 j0 必须在当前 i 之后
            let dv = if dg > 0 && dj as usize > i { dg + p } else { p };
            let dj2 = if dg > 0 && dj as usize > i { dj } else { j as u16 };
            let uv = if up > 0 && uj as usize > i { up - gap } else { 0 };
            let lv = if left > 0 && lj as usize > i { left - gap } else { 0 };
            let (mut v, mut nj) = (0i32, 0u16);
            if dv >= uv && dv >= lv {
                if dv > 0 {
                    v = dv;
                    nj = dj2;
                }
            } else if uv >= lv {
                if uv > 0 {
                    v = uv;
                    nj = uj;
                }
            } else if lv > 0 {
                v = lv;
                nj = lj;
            }
            cur[j] = v;
            cj[j] = nj;
            left = v;
            lj = nj;
            if v > best {
                best = v;
            }
        }
        std::mem::swap(&mut prev, &mut cur);
        std::mem::swap(&mut pj, &mut cj);
    }
    best
}

/// 固定切点的外扩比对：从切点向两边走，前段倒着读、后段顺着读（M 专用）。
/// 返回 (最高分, 走到最高分时覆盖的字母数)
fn extend_from_split(s: &[u8], c: usize, tab: &Tab, gap: i32, sc: &mut Scratch) -> (i32, usize) {
    let n = s.len();
    if c == 0 || c >= n {
        return (0, 0);
    }
    let la = c;
    let lb = n - c;
    sc.a.clear();
    sc.a.resize(lb + 1, 0);
    for y in 1..=lb {
        sc.a[y] = -(gap * y as i32);
    }
    let mut best = (0i32, 0usize);
    for x in 1..=la {
        let a = s[c - x] as usize;
        let row = &tab[a];
        let mut diag = sc.a[0];
        sc.a[0] = -(gap * x as i32);
        for y in 1..=lb {
            let up = sc.a[y];
            let v = (diag + row[s[c + y - 1] as usize]).max(up - gap).max(sc.a[y - 1] - gap);
            diag = up;
            sc.a[y] = v;
            if v > best.0 {
                best = (v, x + y);
            }
        }
    }
    best
}

// ==================== 打乱基准 ====================

struct Rng(u64);
impl Rng {
    #[inline]
    fn next(&mut self) -> u64 {
        // splitmix64，种子固定，结果确定
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
/// segs=1 全打乱（只保留字母比例）；segs>1 分段内打乱（保留慢趋势，只破坏细节顺序）
fn shuffled(src: &[u8], segs: usize, r: &mut Rng, out: &mut Vec<u8>) {
    out.clear();
    out.extend_from_slice(src);
    if segs <= 1 {
        shuffle_range(out, r);
        return;
    }
    let n = out.len();
    let len = n.div_ceil(segs).max(1);
    let mut i = 0;
    while i < n {
        let e = (i + len).min(n);
        shuffle_range(&mut out[i..e], r);
        i = e;
    }
}

fn mean_std(v: &[i32]) -> (f32, f32) {
    let n = v.len() as f32;
    let m = v.iter().map(|&x| x as f32).sum::<f32>() / n;
    let s = (v.iter().map(|&x| (x as f32 - m).powi(2)).sum::<f32>() / n).sqrt();
    (m, s)
}

fn null_scores(
    s: &[u8],
    tab: &Tab,
    cfg: &Cfg,
    nested: bool,
    segs: usize,
    seed: u64,
    sc: &mut Scratch,
) -> (f32, f32) {
    let mut r = Rng(seed);
    let mut scores = [0i32; N_NULL];
    let mut tmp = std::mem::take(&mut sc.tmp);
    for x in scores.iter_mut() {
        shuffled(s, segs, &mut r, &mut tmp);
        *x = if nested {
            nested_score(&tmp, tab, cfg.gap, cfg.minsep, sc)
        } else {
            forward_score(&tmp, tab, cfg.gap, cfg.minsep, sc)
        };
    }
    sc.tmp = tmp;
    mean_std(&scores)
}

#[inline]
fn z_of(x: i32, m: f32, s: f32) -> f32 {
    (x as f32 - m) / s.max(1.0)
}

// ==================== 第 4 层：路径特征 ====================

const FEATS: [&str; 35] = [
    "bits", "perletter", "zfull", "zseg", "excess", "stemlen", "matchfrac", "gapfrac", "center",
    "asym", "buyfirst", "armvol", "npeaks", "conc", "meancent", "top3cent", "signed",
    "bitsR", "bitsC", "bitsD", "zsegR", "zsegC", "zsegD", "dzR", "dzC", "dzD", "dzmax",
    "midS", "midR", "volmedS", "volmedR", "highS", "highR", "lowS", "lowR",
];

/// 在镜像中心曲线上贪心取若干个互相隔开的峰
fn center_peaks(center: &[i32], n: usize) -> (f32, f32, f32, f32) {
    let sep = (n / 5).max(4) as i64;
    let mut v: Vec<(i32, i64)> = center
        .iter()
        .enumerate()
        .filter(|(_, &x)| x > 0)
        .map(|(i, &x)| (x, i as i64))
        .collect();
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

#[allow(clippy::too_many_arguments)]
fn analyze(
    s: &[u8],
    bars: &[Bar],
    t: usize,
    cfg: &Cfg,
    seed: u64,
    controls: bool,
    sc: &mut Scratch,
    out: &mut [f32],
) {
    let n = s.len();
    let active = s.iter().filter(|&&x| x >= 2).count();
    if active < 20 {
        return;
    }
    let tm = make_tab(t, cfg, true);
    let ti = make_tab(t, cfg, false);
    let lam_m = karlin_lambda(s, &tm);
    let best = nested_full(s, &tm, cfg.gap, cfg.minsep, sc);
    let (npk, conc, meanc, top3) = center_peaks(&sc.center, n);
    let (mf, sf) = null_scores(s, &tm, cfg, true, 1, seed ^ 0x11, sc);
    let (ms, ss) = null_scores(s, &tm, cfg, true, 16, seed ^ 0x22, sc);
    let c = best.c;
    let nf = n as f32;
    let zseg = z_of(best.score, ms, ss);
    out[0] = bits_of(best.score, lam_m, n);
    out[1] = best.score as f32 / nf;
    out[2] = z_of(best.score, mf, sf);
    out[3] = zseg;
    out[4] = (best.score as f32 - mf) / nf;
    if best.score > 0 {
        let pairs = c.pairs.max(1) as f32;
        out[5] = c.pairs as f32 / nf;
        out[6] = c.matched as f32 / pairs;
        out[7] = c.gaps as f32 / (pairs + c.gaps as f32);
        out[8] = (c.i0 as f32 + c.j0 as f32) * 0.5 / nf;
        let (le, ll) = ((c.i0 as usize - best.i + 1) as f32, (best.j - c.j0 as usize + 1) as f32);
        out[9] = (ll - le) / (ll + le);
        let m = c.matched.max(1) as f32;
        out[10] = c.bs as f32 / m;
        let vol = |a: usize, b: usize| -> f32 { bars[a..=b].iter().map(|x| x.vol()).sum() };
        let ve = vol(best.i, c.i0 as usize);
        let vl = vol(c.j0 as usize, best.j);
        if ve > 0.0 && vl > 0.0 {
            out[11] = (vl / ve).ln();
        }
        out[16] = out[1] * out[10];
    } else {
        for x in [5usize, 6, 7, 9] {
            out[x] = 0.0;
        }
        out[10] = 0.0;
        out[16] = 0.0;
    }
    out[12] = npk;
    out[13] = conc;
    out[14] = meanc;
    out[15] = top3;
    if !controls {
        return;
    }
    let lam_r = karlin_lambda(s, &ti);
    let rs = nested_score(s, &ti, cfg.gap, cfg.minsep, sc);
    let cs = forward_score(s, &tm, cfg.gap, cfg.minsep, sc);
    let ds = forward_score(s, &ti, cfg.gap, cfg.minsep, sc);
    let (mr, sr) = null_scores(s, &ti, cfg, true, 16, seed ^ 0x33, sc);
    let (mc, sc2) = null_scores(s, &tm, cfg, false, 16, seed ^ 0x44, sc);
    let (md, sd) = null_scores(s, &ti, cfg, false, 16, seed ^ 0x55, sc);
    let (zr, zc, zd) = (z_of(rs, mr, sr), z_of(cs, mc, sc2), z_of(ds, md, sd));
    out[17] = bits_of(rs, lam_r, n);
    out[18] = bits_of(cs, lam_m, n);
    out[19] = bits_of(ds, lam_r, n);
    out[20] = zr;
    out[21] = zc;
    out[22] = zd;
    out[23] = zseg - zr;
    out[24] = zseg - zc;
    out[25] = zseg - zd;
    out[26] = zseg - zr.max(zc).max(zd);
    // 固定切点：字母中点、成交量中点、最高价位置、最低价位置
    let mut cuts = [n / 2, n / 2, n / 2, n / 2];
    let tot: f32 = bars.iter().map(|b| b.vol()).sum();
    let (mut cum, mut got) = (0f32, false);
    let (mut hi, mut lo) = (f32::MIN, f32::MAX);
    for (i, b) in bars.iter().enumerate() {
        cum += b.vol();
        if !got && cum >= tot * 0.5 {
            cuts[1] = i;
            got = true;
        }
        if b.n > 0 {
            if b.p1 > hi {
                hi = b.p1;
                cuts[2] = i;
            }
            if b.p1 < lo {
                lo = b.p1;
                cuts[3] = i;
            }
        }
    }
    for (k, &cut) in cuts.iter().enumerate() {
        let (sv, rch) = extend_from_split(s, cut, &tm, cfg.gap, sc);
        out[27 + 2 * k] = sv as f32 / nf;
        out[28 + 2 * k] = rch as f32 / nf;
    }
}

// ==================== 组合清单与每股计算 ====================

struct Combo {
    win: usize,
    clock: usize,
    alpha: usize,
}
fn combos() -> Vec<Combo> {
    let mut v = Vec::new();
    for win in 0..WINS.len() {
        for clock in 0..CLOCKS.len() {
            for alpha in 0..ALPHAS.len() {
                if alpha == 6 && clock != 0 {
                    continue;
                }
                v.push(Combo { win, clock, alpha });
            }
        }
    }
    v
}

const VARIANTS: [(&str, Cfg); 8] = [
    ("gap1", Cfg { gap: 1, ..BASE }),
    ("gap4", Cfg { gap: 4, ..BASE }),
    ("sep1", Cfg { minsep: 1, ..BASE }),
    ("sep12", Cfg { minsep: 12, ..BASE }),
    ("sizeswap", Cfg { size_swap: true, ..BASE }),
    ("tierstrict", Cfg { tier_strict: true, ..BASE }),
    ("theta0", Cfg { theta: 0.0, ..BASE }),
    ("theta05", Cfg { theta: 0.5, ..BASE }),
];
const VAR_FEATS: [(&str, usize); 6] =
    [("bits", 0), ("zseg", 3), ("signed", 16), ("center", 8), ("stemlen", 5), ("buyfirst", 10)];
const NAIVE: [&str; 2] = ["anticorr", "fbdiff"];
/// 进入横截面层的特征（FEATS 下标）
const CS_FEATS: [(&str, usize); 4] = [("zseg", 3), ("bits", 0), ("signed", 16), ("center", 8)];

fn build_names(cs: &[Combo]) -> (Vec<String>, usize) {
    let mut names = Vec::new();
    for c in cs {
        for f in FEATS {
            names.push(format!(
                "mir_{}_{}_{}_{}",
                ALPHAS[c.alpha], CLOCKS[c.clock].1, WINS[c.win].name, f
            ));
        }
    }
    for w in WINS {
        for (vn, _) in VARIANTS {
            for (fnm, _) in VAR_FEATS {
                names.push(format!("var_int_tm_{}_{}_{}", w.name, vn, fnm));
            }
        }
    }
    for w in WINS {
        for nv in NAIVE {
            names.push(format!("naive_{}_{}", nv, w.name));
        }
        names.push(format!("audit_logamt_{}", w.name));
    }
    let base = names.len();
    for c in cs {
        for (f, _) in CS_FEATS {
            for op in ["csrank", "peerx"] {
                names.push(format!(
                    "{}_{}_{}_{}_{}",
                    op, ALPHAS[c.alpha], CLOCKS[c.clock].1, WINS[c.win].name, f
                ));
            }
        }
    }
    (names, base)
}

fn seed_of(code: &str, c: u64) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in code.bytes() {
        h = (h ^ b as u64).wrapping_mul(0x100000001b3);
    }
    h ^ c.wrapping_mul(0x9E3779B97F4A7C15)
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

fn per_stock(code: &str, cs: &[Combo], ncol_base: usize, stage_io: &std::sync::atomic::AtomicU64) -> (Vec<f32>, bool) {
    let mut out = vec![f32::NAN; ncol_base];
    let t0 = Instant::now();
    let date = *DATE.get().unwrap();
    let raw = match read_trade_fast(code, date) {
        Ok(r) if !r.is_empty() => r,
        _ => return (out, false),
    };
    let mkt = read_market_fast(code, date).unwrap_or_default();
    stage_io.fetch_add(t0.elapsed().as_micros() as u64, std::sync::atomic::Ordering::Relaxed);
    let trs = prep_trades(&raw);
    drop(raw);
    let snaps = book_levels(&mkt);
    drop(mkt);
    let mut sc = Scratch::new();
    let nf = FEATS.len();
    let mut pos = 0;
    let mut wtrs: Vec<&[Tr]> = Vec::with_capacity(3);
    for w in &WINS {
        wtrs.push(window_slice(&trs, w));
    }
    let books: Vec<Vec<Option<f32>>> = WINS.iter().map(|w| build_book(&snaps, w)).collect();
    // 按时段×时钟缓存分箱，同一分箱喂给不同字母表
    let bars_cache: Vec<Vec<Vec<Bar>>> = WINS
        .iter()
        .enumerate()
        .map(|(wi, w)| CLOCKS.iter().map(|(c, _)| build_bars(wtrs[wi], w, *c)).collect())
        .collect();
    for (ci, c) in cs.iter().enumerate() {
        let bars = &bars_cache[c.win][c.clock];
        let clk = CLOCKS[c.clock].0;
        let s = encode(bars, &books[c.win], c.alpha, clk, BASE.theta);
        let seed = seed_of(code, ci as u64 + 1);
        analyze(&s, bars, alpha_tiers(c.alpha), &BASE, seed, true, &mut sc, &mut out[pos..pos + nf]);
        pos += nf;
    }
    let mut tmp = vec![f32::NAN; nf];
    for (wi, _w) in WINS.iter().enumerate() {
        let bars = &bars_cache[wi][0];
        for (vi, (_, cfg)) in VARIANTS.iter().enumerate() {
            tmp.iter_mut().for_each(|x| *x = f32::NAN);
            let s = encode(bars, &books[wi], 1, Clock::Time, cfg.theta);
            let seed = seed_of(code, 10_000 + (wi * 100 + vi) as u64);
            analyze(&s, bars, alpha_tiers(1), cfg, seed, false, &mut sc, &mut tmp);
            for (k, (_, idx)) in VAR_FEATS.iter().enumerate() {
                out[pos + k] = tmp[*idx];
            }
            pos += VAR_FEATS.len();
        }
    }
    // 朴素替代物：分时净买失衡的前后段反向相关；前后段失衡差；以及审计用成交额
    for wi in 0..WINS.len() {
        let bars = &bars_cache[wi][0];
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
        let second = &x[m..];
        out[pos] = -corr(&first, second);
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
        out[pos + 1] = (side(&bars[..m]) - side(&bars[m..])) as f32;
        let amt: f64 = wtrs[wi].iter().map(|t| (t.v * t.p) as f64).sum();
        out[pos + 2] = if amt > 0.0 { amt.ln() as f32 } else { f32::NAN };
        pos += 3;
    }
    debug_assert_eq!(pos, ncol_base);
    (out, true)
}

fn book_levels(mkt: &[MarketRecord]) -> Vec<(i64, f32)> {
    let mut v: Vec<(i64, f32)> = mkt
        .iter()
        .map(|m| {
            let sec = (m.time_us.div_euclid(US)).rem_euclid(86_400) - DAY_OPEN_SEC;
            (sec * US + m.time_us.rem_euclid(US), (m.total_bid_vol - m.total_ask_vol) as f32)
        })
        .collect();
    v.sort_by_key(|x| x.0);
    v
}

// ==================== 第 5 层：横截面 ====================

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
    (0..col.len())
        .map(|i| if bucket[i] < nb { col[i] - med[bucket[i]] } else { f32::NAN })
        .collect()
}

// ==================== 股票范围 ====================

fn stock_codes_for_day(
    dir: &std::path::Path,
    symbol_map: &std::path::Path,
    date: i64,
    subdir: &str,
) -> std::io::Result<Vec<String>> {
    use std::io::{Error, ErrorKind};
    let invalid = |e| Error::new(ErrorKind::InvalidData, e);
    let mut reader = csv::Reader::from_path(symbol_map).map_err(invalid)?;
    let column = reader
        .headers()
        .map_err(invalid)?
        .iter()
        .position(|h| h == "symbol")
        .ok_or_else(|| Error::new(ErrorKind::InvalidData, "股票表缺少 symbol 列"))?;
    let mut stocks = std::collections::BTreeSet::new();
    for record in reader.records() {
        let record = record.map_err(invalid)?;
        let code = record.get(column).unwrap_or("").trim();
        if code.is_empty() || code.len() > 6 || !code.bytes().all(|b| b.is_ascii_digit()) {
            return Err(Error::new(ErrorKind::InvalidData, "股票表含无效代码"));
        }
        stocks.insert(format!("{code:0>6}"));
    }
    if stocks.is_empty() || !dir.is_dir() {
        return Err(Error::new(ErrorKind::InvalidData, "股票白名单为空或原始数据目录不存在"));
    }
    Ok(stocks
        .into_iter()
        .filter(|code| dir.join(format!("{code}_{date}_{subdir}.csv")).is_file())
        .collect())
}

fn list_codes(date: i64, subdir: &str) -> std::io::Result<Vec<String>> {
    let root = std::env::var("RUST_PYFUNC_LEVEL2_ROOT").unwrap_or("/ssd_data/stock".into());
    let basic = std::env::var("RUST_PYFUNC_BASIC_INFO_DIR").unwrap_or("/ssd_data/data/basic_info".into());
    stock_codes_for_day(
        &std::path::Path::new(&root).join(date.to_string()).join(subdir),
        &std::path::Path::new(&basic).join("symbol_map.csv"),
        date,
        subdir,
    )
}

static DATE: std::sync::OnceLock<i64> = std::sync::OnceLock::new();

#[derive(Serialize)]
struct Out {
    date: i64,
    names: Vec<String>,
    codes: Vec<String>,
    vals: Vec<Vec<f32>>,
}

fn main() {
    let date: i64 = std::env::args().nth(1).expect("用法: align_mirror_sandbox <date>").parse().unwrap();
    DATE.set(date).unwrap();
    rayon::ThreadPoolBuilder::new().num_threads(MAX_THREADS).build_global().unwrap();
    let t_all = Instant::now();
    let codes = match list_codes(date, "transaction") {
        Ok(c) => c,
        Err(e) => {
            eprintln!("股票范围或原始目录检查失败: {e}");
            std::process::exit(2);
        }
    };
    // 可选第二参数：只算前 N 只（测速用）
    let codes: Vec<String> = match std::env::args().nth(2) {
        Some(k) => codes.into_iter().take(k.parse().unwrap()).collect(),
        None => codes,
    };
    let cs = combos();
    let (mut names, base) = build_names(&cs);
    eprintln!("codes = {}  columns = {}  base = {}", codes.len(), names.len(), base);

    let io_us = std::sync::atomic::AtomicU64::new(0);
    let t1 = Instant::now();
    let res: Vec<(Vec<f32>, bool)> = codes.par_iter().map(|c| per_stock(c, &cs, base, &io_us)).collect();
    let per_stock_sec = t1.elapsed().as_secs_f64();

    // ==== 横截面层 ====
    let t2 = Instant::now();
    let n = codes.len();
    let amt_idx = names.iter().position(|x| x == "audit_logamt_all").unwrap();
    let mut order: Vec<usize> = (0..n).filter(|&i| res[i].0[amt_idx].is_finite()).collect();
    order.sort_by(|&a, &b| res[a].0[amt_idx].total_cmp(&res[b].0[amt_idx]).then(a.cmp(&b)));
    const NB: usize = 20;
    let mut bucket = vec![NB; n];
    for (r, &i) in order.iter().enumerate() {
        bucket[i] = (r * NB / order.len().max(1)).min(NB - 1);
    }
    let nf = FEATS.len();
    let pairs: Vec<(usize, usize)> = (0..cs.len())
        .flat_map(|ci| CS_FEATS.iter().map(move |&(_, f)| (ci, f)))
        .collect();
    let extras: Vec<(Vec<f32>, Vec<f32>)> = pairs
        .par_iter()
        .map(|&(ci, f)| {
            let col: Vec<f32> = res.iter().map(|r| r.0[ci * nf + f]).collect();
            (cs_rank(&col), peer_excess(&col, &bucket, NB))
        })
        .collect();
    let mut vals: Vec<Vec<f32>> = res.into_iter().map(|(v, _)| v).collect();
    for (ei, _) in pairs.iter().enumerate() {
        let (a, b) = &extras[ei];
        for i in 0..n {
            vals[i].push(a[i]);
            vals[i].push(b[i]);
        }
    }
    let cs_sec = t2.elapsed().as_secs_f64();
    assert!(vals.iter().all(|v| v.len() == names.len()), "列数不恒定");

    let t3 = Instant::now();
    let out = Out { date, names: std::mem::take(&mut names), codes, vals };
    let stdout = std::io::stdout();
    let mut w = std::io::BufWriter::with_capacity(1 << 20, stdout.lock());
    serde_json::to_writer(&mut w, &out).unwrap();
    use std::io::Write;
    w.flush().unwrap();
    eprintln!(
        "timing: total={:.1}s  per_stock={:.1}s  read_cpu_sum={:.1}s  cross_section={:.2}s  json={:.1}s",
        t_all.elapsed().as_secs_f64(),
        per_stock_sec,
        io_us.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1e6,
        cs_sec,
        t3.elapsed().as_secs_f64()
    );
}
