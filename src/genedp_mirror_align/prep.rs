//! 第 1 层：原始事件 → 时段 → 分箱（5 种时钟）→ 字母（10 种字母表）。
use super::reader::{Ev, Snap};

pub const US: i64 = 1_000_000;

// ---------------- 事件 ----------------

#[derive(Clone, Copy)]
pub struct Tr {
    pub t: i64,
    pub pm: bool,
    pub d: i8,
    pub v: f32,
    pub p: f32,
    pub aid: i64,
    pub new_ord: bool,
    pub ord: u32,
}

#[derive(Clone, Copy)]
pub struct Cx {
    pub t: i64,
    pub pm: bool,
    /// +1 撤买单，-1 撤卖单
    pub side: i8,
    pub v: f32,
    /// 撤单时刻已出现的最大订单编号 - 被撤订单编号（撤单龄的代理）
    pub lag: f32,
}

pub struct Day {
    pub trs: Vec<Tr>,
    pub cxs: Vec<Cx>,
    pub snaps: Vec<Snap>,
    pub cxl_ok: bool,
}

/// 撤单数据是否可用：沪市 20211105 起，深市 20160104 起，其余（北交所等）不用
pub fn cxl_available(code: &str, date: i64) -> bool {
    match code.as_bytes().first() {
        Some(b'6') => date >= 20211105,
        Some(b'0') | Some(b'3') => date >= 20160104,
        _ => false,
    }
}

pub fn prep(mut evs: Vec<Ev>, mut snaps: Vec<Snap>, cxl_ok: bool) -> Day {
    evs.sort_by(|a, b| a.t.cmp(&b.t).then(a.index.cmp(&b.index)));
    snaps.sort_by(|a, b| a.t.cmp(&b.t));
    let mut trs = Vec::with_capacity(evs.len());
    let mut cxs = Vec::new();
    let mut prev = i64::MIN;
    let mut ord = 0u32;
    let mut maxid = 0i64;
    for e in &evs {
        if e.flag == 32 {
            if !cxl_ok || !(e.volume > 0.0) {
                continue;
            }
            let (side, id) = if e.bid > 0 {
                (1i8, e.bid)
            } else if e.ask > 0 {
                (-1i8, e.ask)
            } else {
                continue;
            };
            maxid = maxid.max(id);
            cxs.push(Cx { t: e.t, pm: e.pm, side, v: e.volume as f32, lag: (maxid - id) as f32 });
            continue;
        }
        maxid = maxid.max(e.bid).max(e.ask);
        if !(e.volume > 0.0 && e.price > 0.0) {
            continue;
        }
        let (d, aid) = match e.flag {
            66 => (1i8, e.bid),
            83 => (-1i8, e.ask),
            _ => (0i8, 0i64),
        };
        let new_ord = d != 0 && aid != prev;
        if new_ord {
            prev = aid;
            ord += 1;
        }
        trs.push(Tr { t: e.t, pm: e.pm, d, v: e.volume as f32, p: e.price as f32, aid, new_ord, ord });
    }
    Day { trs, cxs, snaps, cxl_ok }
}

// ---------------- 时段 ----------------

#[derive(Clone, Copy, PartialEq)]
pub enum WinKind {
    All,
    Pm,
    Tail,
}

#[derive(Clone, Copy)]
pub struct Win {
    pub name: &'static str,
    pub kind: WinKind,
    pub lo: i64,
    pub hi: i64,
    pub bar_sec: i64,
    pub n_bar: usize,
}

/// 全天 09:30-11:30 + 13:00-14:57；下午 13:00-14:57（只要下午记录）；尾盘 14:27-14:57
pub const WINS: [Win; 3] = [
    Win { name: "all", kind: WinKind::All, lo: 0, hi: 14_220, bar_sec: 30, n_bar: 300 },
    Win { name: "pm", kind: WinKind::Pm, lo: 7_200, hi: 14_220, bar_sec: 30, n_bar: 200 },
    Win { name: "tail", kind: WinKind::Tail, lo: 12_420, hi: 14_220, bar_sec: 10, n_bar: 120 },
];

impl Win {
    #[inline]
    pub fn contains(&self, t: i64, pm: bool) -> bool {
        match self.kind {
            WinKind::All => true,
            WinKind::Pm => pm,
            WinKind::Tail => pm && t >= self.lo * US,
        }
    }
    pub fn n_time(&self) -> usize {
        ((self.hi - self.lo) / self.bar_sec) as usize
    }
}

pub struct WinData {
    pub trs: Vec<Tr>,
    pub cxs: Vec<Cx>,
    pub snaps: Vec<Snap>,
}

pub fn window(day: &Day, w: &Win) -> WinData {
    WinData {
        trs: day.trs.iter().filter(|x| w.contains(x.t, x.pm)).copied().collect(),
        cxs: day.cxs.iter().filter(|x| w.contains(x.t, x.pm)).copied().collect(),
        snaps: day.snaps.iter().filter(|x| w.contains(x.t, x.pm)).copied().collect(),
    }
}

// ---------------- 分箱 ----------------

#[derive(Clone, Copy, PartialEq)]
pub enum Clock {
    Time,
    Vol,
    Ord,
    Ev,
    Cn,
}
pub const CLOCK_NAMES: [&str; 5] = ["tm", "vo", "od", "ev", "cn"];
pub const CLOCKS: [Clock; 5] = [Clock::Time, Clock::Vol, Clock::Ord, Clock::Ev, Clock::Cn];
pub const EV_CAP: usize = 1000;

#[derive(Clone, Copy, Default)]
pub struct Bar {
    pub bv: f32,
    pub sv: f32,
    pub n: u32,
    pub orders: u32,
    pub t0: i64,
    pub t1: i64,
    pub p0: f32,
    pub p1: f32,
    pub jump: f64,
    pub njump: u32,
    pub cb: f32,
    pub ca: f32,
    pub nc: u32,
    pub lag: f64,
    pub has_ev: bool,
    pub first_t: i64,
}

impl Bar {
    #[inline]
    fn touch(&mut self, t: i64) {
        if !self.has_ev {
            self.has_ev = true;
            self.first_t = t;
        }
    }
    #[inline]
    fn add_tr(&mut self, tr: &Tr, v: f32, jump: Option<f64>, count_order: bool) {
        self.touch(tr.t);
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
        if count_order && tr.new_ord {
            self.orders += 1;
            if let Some(j) = jump {
                self.jump += j;
                self.njump += 1;
            }
        }
    }
    #[inline]
    fn add_cx(&mut self, c: &Cx) {
        self.touch(c.t);
        if c.side > 0 {
            self.cb += c.v;
        } else {
            self.ca += c.v;
        }
        self.nc += 1;
        self.lag += c.lag as f64;
    }
    #[inline]
    pub fn tvol(&self) -> f32 {
        self.bv + self.sv
    }
}

pub struct Bars {
    pub bars: Vec<Bar>,
    /// 盘口：本箱末（买量合计-卖量合计）相对上一有快照箱的变化；None = 本箱无快照
    pub book: Vec<Option<f32>>,
}

fn jumps(trs: &[Tr]) -> Vec<Option<f64>> {
    let mut prev: Option<i64> = None;
    trs.iter()
        .map(|t| {
            if !t.new_ord {
                return None;
            }
            let j = prev.map(|p| (t.aid - p).abs() as f64);
            prev = Some(t.aid);
            j
        })
        .collect()
}

/// 按时间把事件归到箱：箱的起点是该箱第一条事件的时间，空箱沿用上一箱起点
fn starts_of(bars: &[Bar]) -> Vec<i64> {
    let mut s = Vec::with_capacity(bars.len());
    let mut last = i64::MIN;
    for b in bars {
        if b.has_ev {
            last = b.first_t;
        }
        s.push(last);
    }
    s
}
#[inline]
fn bar_by_time(starts: &[i64], t: i64) -> usize {
    starts.partition_point(|&s| s <= t).saturating_sub(1)
}

fn add_book(bars: &mut Bars, snaps: &[Snap], idx: impl Fn(&Snap) -> usize) {
    let n = bars.bars.len();
    let mut last: Vec<Option<f32>> = vec![None; n];
    let mut base: Option<f32> = None;
    for s in snaps {
        if base.is_none() {
            base = Some(s.lv);
        }
        last[idx(s).min(n - 1)] = Some(s.lv);
    }
    let mut prev = base;
    for i in 0..n {
        if let (Some(l), Some(p)) = (last[i], prev) {
            bars.book[i] = Some(l - p);
            prev = Some(l);
        }
    }
}

pub fn build_bars(wd: &WinData, w: &Win, c: Clock) -> Bars {
    let trs = &wd.trs;
    let n = match c {
        Clock::Time => w.n_time(),
        Clock::Vol | Clock::Ord | Clock::Cn => w.n_bar,
        Clock::Ev => {
            if trs.is_empty() {
                0
            } else {
                let o0 = trs[0].ord - trs[0].new_ord as u32;
                ((trs[trs.len() - 1].ord - o0) as usize).min(EV_CAP)
            }
        }
    };
    let mut out = Bars { bars: vec![Bar::default(); n], book: vec![None; n] };
    if n == 0 {
        return out;
    }
    let js = jumps(trs);
    let lo = w.lo * US;
    match c {
        Clock::Time => {
            let wid = w.bar_sec * US;
            let at = |t: i64| (((t - lo).max(0) / wid) as usize).min(n - 1);
            for (k, tr) in trs.iter().enumerate() {
                out.bars[at(tr.t)].add_tr(tr, tr.v, js[k], true);
            }
            for cx in &wd.cxs {
                out.bars[at(cx.t)].add_cx(cx);
            }
            add_book(&mut out, &wd.snaps, |s| at(s.t));
            return out;
        }
        Clock::Vol => {
            let total: f64 = trs.iter().map(|t| t.v as f64).sum();
            if total > 0.0 {
                let wid = total / n as f64;
                let mut cum = 0.0f64;
                for (k, tr) in trs.iter().enumerate() {
                    let (mut c0, c1) = (cum, cum + tr.v as f64);
                    cum = c1;
                    let mut i = ((c0 / wid) as usize).min(n - 1);
                    let mut first = true;
                    loop {
                        let end = if i == n - 1 { c1 } else { (((i + 1) as f64) * wid).min(c1) };
                        let part = (end - c0).max(0.0) as f32;
                        out.bars[i].add_tr(tr, part, if first { js[k] } else { None }, first);
                        first = false;
                        c0 = end;
                        if c0 >= c1 || i == n - 1 {
                            break;
                        }
                        i += 1;
                    }
                }
            }
        }
        Clock::Ord | Clock::Ev => {
            if !trs.is_empty() {
                let o0 = trs[0].ord - trs[0].new_ord as u32;
                let tot = (trs[trs.len() - 1].ord - o0).max(1) as u64;
                for (k, tr) in trs.iter().enumerate() {
                    let r = tr.ord.saturating_sub(o0).saturating_sub(1) as u64;
                    let i = ((r * n as u64 / tot) as usize).min(n - 1);
                    out.bars[i].add_tr(tr, tr.v, js[k], true);
                }
            }
        }
        Clock::Cn => {
            let tot = wd.cxs.len() as u64;
            if tot > 0 {
                for (k, cx) in wd.cxs.iter().enumerate() {
                    let i = ((k as u64 * n as u64 / tot) as usize).min(n - 1);
                    out.bars[i].add_cx(cx);
                }
                let starts = starts_of(&out.bars);
                for (k, tr) in trs.iter().enumerate() {
                    let i = bar_by_time(&starts, tr.t);
                    out.bars[i].add_tr(tr, tr.v, js[k], true);
                }
                add_book(&mut out, &wd.snaps, |s| bar_by_time(&starts, s.t));
            }
            return out;
        }
    }
    // 成交量 / 订单数 / 逐单时钟：撤单与快照按时间归箱
    let starts = starts_of(&out.bars);
    for cx in &wd.cxs {
        out.bars[bar_by_time(&starts, cx.t)].add_cx(cx);
    }
    add_book(&mut out, &wd.snaps, |s| bar_by_time(&starts, s.t));
    out
}

// ---------------- 字母 ----------------

pub const ALPHA_NAMES: [&str; 10] =
    ["dir", "int", "osize", "tempo", "jump", "price", "book", "cxlsize", "cxlage", "cxlmix"];
pub const A_DIR: usize = 0;
pub const A_INT: usize = 1;
pub const A_PRICE: usize = 5;
pub const A_BOOK: usize = 6;
pub const A_CXLSIZE: usize = 7;
pub const A_CXLMIX: usize = 9;

pub fn is_cxl(a: usize) -> bool {
    a >= 7
}

pub fn alpha_tiers(a: usize) -> usize {
    match a {
        0 => 1,
        1..=4 | 7 | 8 => 3,
        _ => 2,
    }
}

/// 字母：0 空闲、1 中性、2+2*tier 买侧、3+2*tier 卖侧
#[inline]
pub fn letter(side: i8, tier: usize) -> u8 {
    match side {
        0 => 1,
        s if s > 0 => (2 + 2 * tier) as u8,
        _ => (3 + 2 * tier) as u8,
    }
}

fn thresholds(mut v: Vec<f32>, t: usize) -> Vec<f32> {
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
fn median(mut v: Vec<f32>) -> f32 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(|a, b| a.total_cmp(b));
    v[v.len() / 2]
}

/// 门槛三档：0 = 基础，1 = 变体一，2 = 变体二（按字母表各自定义）
pub fn theta(a: usize, mode: usize) -> f32 {
    match a {
        A_PRICE => [0.0, 0.5, 1.0][mode],
        A_BOOK => [0.3, 0.0, 0.6][mode],
        _ => [0.2, 0.0, 0.5][mode],
    }
}

pub struct Seq {
    pub s: Vec<u8>,
    /// 每个字母的量（用于臂量与配对量）
    pub vol: Vec<f32>,
    /// 每个字母的末价（无成交为 NaN），用于最高/最低价切点
    pub px: Vec<f32>,
}

#[inline]
fn side_imb(b: f32, s: f32, th: f32) -> i8 {
    let tot = b + s;
    if tot <= 0.0 {
        return 0;
    }
    let im = (b - s) / tot;
    if im > th {
        1
    } else if im < -th {
        -1
    } else {
        0
    }
}

pub fn encode(bars: &Bars, a: usize, c: Clock, mode: usize) -> Seq {
    let bs = &bars.bars;
    let n = bs.len();
    let t = alpha_tiers(a);
    let th = theta(a, mode);
    let px: Vec<f32> = bs.iter().map(|b| if b.n > 0 { b.p1 } else { f32::NAN }).collect();
    let vol: Vec<f32> = bs
        .iter()
        .map(|b| match a {
            7 | 8 => b.cb + b.ca,
            9 => b.tvol() + b.cb + b.ca,
            _ => b.tvol(),
        })
        .collect();
    let mut s = vec![0u8; n];
    match a {
        A_BOOK => {
            let med = median(bars.book.iter().flatten().map(|x| x.abs()).collect());
            for i in 0..n {
                if let Some(d) = bars.book[i] {
                    s[i] = if *&d == 0.0 || d.abs() < th * med {
                        1
                    } else {
                        letter(if d > 0.0 { 1 } else { -1 }, (d.abs() >= med) as usize)
                    };
                }
            }
        }
        A_PRICE => {
            // 逐单时钟比相邻字母的末价，其余时钟比箱内首末价
            let mut dp = vec![f32::NAN; n];
            let mut last = f32::NAN;
            for i in 0..n {
                if bs[i].n == 0 {
                    continue;
                }
                dp[i] = if c == Clock::Ev {
                    if last.is_finite() {
                        bs[i].p1 - last
                    } else {
                        bs[i].p1 - bs[i].p0
                    }
                } else {
                    bs[i].p1 - bs[i].p0
                };
                last = bs[i].p1;
            }
            let med = median(dp.iter().filter(|x| x.is_finite() && **x != 0.0).map(|x| x.abs()).collect());
            let vth = thresholds(bs.iter().filter(|b| b.n > 0).map(|b| b.tvol()).collect(), 2);
            for i in 0..n {
                if bs[i].n == 0 {
                    continue;
                }
                let d = dp[i];
                let side = if d.abs() < 1e-6 || d.abs() < th * med {
                    0
                } else if d > 0.0 {
                    1
                } else {
                    -1
                };
                s[i] = letter(side, tier_of(bs[i].tvol(), &vth));
            }
        }
        7 | 8 => {
            let metric = |b: &Bar| if a == 7 { b.cb + b.ca } else { (b.lag / b.nc.max(1) as f64) as f32 };
            let vth = thresholds(bs.iter().filter(|b| b.nc > 0).map(metric).collect(), t);
            for i in 0..n {
                let b = &bs[i];
                if b.nc == 0 {
                    continue;
                }
                s[i] = letter(side_imb(b.cb, b.ca, th), tier_of(metric(b), &vth));
            }
        }
        A_CXLMIX => {
            for i in 0..n {
                let b = &bs[i];
                let q = [b.bv, b.sv, b.cb, b.ca];
                let sum: f32 = q.iter().sum();
                if b.n == 0 && b.nc == 0 || sum <= 0.0 {
                    continue;
                }
                let mut k = 0;
                for j in 1..4 {
                    if q[j] > q[k] {
                        k = j;
                    }
                }
                let second = (0..4).filter(|&j| j != k).map(|j| q[j]).fold(0.0f32, f32::max);
                s[i] = if q[k] - second < th * sum && th > 0.0 || q[k] == second {
                    1
                } else {
                    letter(if k % 2 == 0 { 1 } else { -1 }, k / 2)
                };
            }
        }
        _ => {
            // 方向 × 档：int 强度、osize 平均单量、tempo 成交间隔、jump 订单编号跳跃
            let metric = |i: usize| -> f32 {
                let b = &bs[i];
                let span = ((b.t1 - b.t0) as f32 / 1e6).max(1e-3);
                match a {
                    A_INT => {
                        if c == Clock::Time {
                            b.tvol()
                        } else {
                            1.0 / span
                        }
                    }
                    2 => b.tvol() / b.orders.max(1) as f32,
                    3 => {
                        if c == Clock::Ev {
                            // 逐单时钟：与上一张单之间的间隔
                            let mut k = i;
                            let mut g = f32::NAN;
                            while k > 0 {
                                k -= 1;
                                if bs[k].n > 0 {
                                    g = (b.t0 - bs[k].t1) as f32 / 1e6;
                                    break;
                                }
                            }
                            g
                        } else if b.n >= 2 {
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
            let ms: Vec<f32> = (0..n).map(|i| if bs[i].n > 0 { metric(i) } else { f32::NAN }).collect();
            let vth = thresholds(ms.iter().filter(|x| x.is_finite()).copied().collect(), t);
            for i in 0..n {
                let b = &bs[i];
                if b.n == 0 {
                    continue;
                }
                let tier = if t <= 1 {
                    0
                } else if ms[i].is_finite() {
                    tier_of(ms[i], &vth)
                } else {
                    t / 2
                };
                s[i] = letter(side_imb(b.bv, b.sv, th), tier);
            }
        }
    }
    Seq { s, vol, px }
}

/// 各时钟用哪些字母表
pub fn alphas_of(c: Clock) -> &'static [usize] {
    match c {
        Clock::Time | Clock::Vol | Clock::Ord => &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        Clock::Ev => &[0, 2, 3, 4, 5],
        Clock::Cn => &[7, 8, 9],
    }
}
