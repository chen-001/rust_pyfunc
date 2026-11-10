//! 字母化层：把一个时段的逐笔、撤单、盘口切成格子，每格变成一个字母。
//! 字母 0 表示静默格（这格没有发生相应行为），其余字母由两个属性组合而成。
use crate::align::SIZE;
use crate::fast_csv_reader::{MarketRecord, TradeRecord};

pub const VOLSIG: usize = 0;
pub const MAG5: usize = 1;
pub const MAG5Y: usize = 2;
pub const SIZEL: usize = 3;
pub const RHYTHM: usize = 4;
pub const OID: usize = 5;
pub const PRICE: usize = 6;
pub const BOOKFLOW: usize = 7;
pub const BOOKSHAPE: usize = 8;
pub const CANCEL: usize = 9;
pub const ALPHA_NAMES: [&str; 10] =
    ["volsig", "mag5", "mag5y", "size", "rhythm", "oid", "price", "bookflow", "bookshape", "cancel"];
/// 每个字母表的属性格数（第一属性, 第二属性）
pub const GRID: [(usize, usize); 10] =
    [(3, 3), (5, 1), (5, 1), (3, 3), (3, 3), (3, 3), (3, 3), (3, 3), (3, 3), (3, 3)];

/// 字母之间的计分表：完全相同 +4；一个属性相同（单属性字母表则是相邻档）+1；其余 -2；静默对静默 +1。
pub fn build_score_table(alpha: usize) -> Vec<i32> {
    let (na, nb) = GRID[alpha];
    let mut t = vec![-2i32; SIZE * SIZE];
    t[0] = 1;
    for x in 1..=na * nb {
        for y in 1..=na * nb {
            let (ax, bx) = ((x - 1) / nb, (x - 1) % nb);
            let (ay, by) = ((y - 1) / nb, (y - 1) % nb);
            let s = if x == y { 4 }
                else if nb == 1 { if (ax as i32 - ay as i32).abs() == 1 { 1 } else { -2 } }
                else if ax == ay || bx == by { 1 } else { -2 };
            t[x * SIZE + y] = s;
        }
    }
    t
}

/// 一个时段内逐格的汇总。magnitude 在时间钟下是成交量，在成交量钟下是这一格花掉的时间。
#[derive(Default, Clone)]
pub struct Bars {
    pub vol: Vec<f64>, pub mag: Vec<f64>, pub buy: Vec<f64>, pub sell: Vec<f64>, pub n: Vec<f64>,
    pub n_event: Vec<f64>, pub gap_cv: Vec<f64>, pub switch_rate: Vec<f64>, pub log_age: Vec<f64>,
    pub ret: Vec<f64>, pub range: Vec<f64>,
    pub cancel: Vec<f64>, pub cancel_bid: Vec<f64>, pub cancel_ask: Vec<f64>,
    pub imb: Vec<f64>, pub net_add: Vec<f64>, pub spread: Vec<f64>, pub top_vol: Vec<f64>, pub book_ok: Vec<bool>,
}

impl Bars {
    fn new(l: usize) -> Self {
        let z = vec![0.0; l];
        let nan = vec![f64::NAN; l];
        Bars { vol: z.clone(), mag: z.clone(), buy: z.clone(), sell: z.clone(), n: z.clone(),
            n_event: z.clone(), gap_cv: nan.clone(), switch_rate: nan.clone(), log_age: nan.clone(),
            ret: nan.clone(), range: nan.clone(),
            cancel: z.clone(), cancel_bid: z.clone(), cancel_ask: z,
            imb: nan.clone(), net_add: nan.clone(), spread: nan.clone(), top_vol: nan, book_ok: vec![false; l] }
    }
}

/// 把逐笔聚合成格子。bin_of 给出每笔属于哪一格，且随时间不减，所以顺序扫描即可，不需要分组容器。
fn fill_trades(b: &mut Bars, ts: &[TradeRecord], bin_of: impl Fn(usize, &TradeRecord) -> usize, use_duration: bool) {
    let (mut cur, mut first) = (usize::MAX, true);
    let (mut gs, mut gs2, mut gn, mut last_t, mut first_t) = (0.0f64, 0.0f64, 0.0f64, 0i64, 0i64);
    let (mut sw, mut prev_active, mut age_sum, mut age_w) = (0.0f64, i64::MIN, 0.0f64, 0.0f64);
    let (mut px_open, mut px_last, mut hi, mut lo, mut prev_px_last) = (0.0f64, 0.0f64, f64::MIN, f64::MAX, f64::NAN);
    let close = |b: &mut Bars, c: usize, gs: f64, gs2: f64, gn: f64, sw: f64, age_sum: f64, age_w: f64,
                 px_open: f64, px_last: f64, hi: f64, lo: f64, prev: f64, first_t: i64, last_t: i64| {
        if gn >= 2.0 && gs > 0.0 {
            let m = gs / gn;
            b.gap_cv[c] = ((gs2 / gn - m * m).max(0.0)).sqrt() / m;
        }
        if b.n[c] > 0.0 { b.switch_rate[c] = sw / b.n[c]; }
        if age_w > 0.0 { b.log_age[c] = age_sum / age_w; }
        let base = if prev.is_finite() { prev } else { px_open };
        if base > 0.0 { b.ret[c] = px_last / base - 1.0; }
        if lo > 0.0 && hi >= lo { b.range[c] = hi / lo - 1.0; }
        if use_duration { b.mag[c] = (last_t - first_t) as f64 / 1e6; } else { b.mag[c] = b.vol[c]; }
    };
    for (idx, t) in ts.iter().enumerate() {
        let c = bin_of(idx, t);
        if c != cur {
            if cur != usize::MAX {
                close(b, cur, gs, gs2, gn, sw, age_sum, age_w, px_open, px_last, hi, lo, prev_px_last, first_t, last_t);
                prev_px_last = px_last;
            }
            cur = c; first = true;
            gs = 0.0; gs2 = 0.0; gn = 0.0; sw = 0.0; prev_active = i64::MIN; age_sum = 0.0; age_w = 0.0;
            hi = f64::MIN; lo = f64::MAX;
        }
        b.vol[c] += t.volume;
        b.n[c] += 1.0;
        if first {
            first = false; px_open = t.price; first_t = t.time_us; b.n_event[c] += 1.0;
        } else if t.time_us != last_t {
            let g = (t.time_us - last_t) as f64 / 1e6;
            gs += g; gs2 += g * g; gn += 1.0; b.n_event[c] += 1.0;
        }
        last_t = t.time_us; px_last = t.price;
        hi = hi.max(t.price); lo = lo.min(t.price);
        let (active, passive) = match t.flag {
            66 => { b.buy[c] += t.volume; (t.bid_order, t.ask_order) }
            83 => { b.sell[c] += t.volume; (t.ask_order, t.bid_order) }
            _ => (0, 0),
        };
        if active > 0 {
            if active != prev_active { sw += 1.0; prev_active = active; }
            if passive > 0 { age_sum += t.volume * (((active - passive).max(1)) as f64).ln(); age_w += t.volume; }
        }
    }
    if cur != usize::MAX {
        close(b, cur, gs, gs2, gn, sw, age_sum, age_w, px_open, px_last, hi, lo, prev_px_last, first_t, last_t);
    }
}

/// 时间钟格子。trades/cancels 已按时间排好；start、width 都是当日时钟的微秒。
pub fn time_bars(trades: &[TradeRecord], cancels: &[TradeRecord], mkt: &[MarketRecord],
                 start: i64, width: i64, l: usize, day_us: i64) -> Bars {
    let mut b = Bars::new(l);
    let bin = |t: i64| -> usize { (((t - day_us - start) / width).max(0) as usize).min(l - 1) };
    fill_trades(&mut b, trades, |_, t| bin(t.time_us), false);
    for t in cancels {
        let c = bin(t.time_us);
        b.cancel[c] += t.volume;
        if t.bid_order > 0 { b.cancel_bid[c] += t.volume; } else if t.ask_order > 0 { b.cancel_ask[c] += t.volume; }
    }
    // 盘口：每格取最后一个快照，没有快照的格子沿用前值
    let mut last: Option<(f64, f64, f64, f64)> = None;
    let mut prev_total = f64::NAN;
    let mut p = 0usize;
    for c in 0..l {
        let end = day_us + start + width * (c as i64 + 1);
        let mut seen = false;
        while p < mkt.len() && (mkt[p].time_us < end || c == l - 1) {
            let s = &mkt[p];
            let (bv, av) = (s.total_bid_vol, s.total_ask_vol);
            let mid = (s.ask_prcs[0] + s.bid_prcs[0]) * 0.5;
            if bv + av > 0.0 && mid > 0.0 {
                last = Some(((bv - av) / (bv + av), bv + av, (s.ask_prcs[0] - s.bid_prcs[0]) / mid, s.ask_vols[0] + s.bid_vols[0]));
                seen = true;
            }
            p += 1;
        }
        if let Some((imb, total, spread, top)) = last {
            b.imb[c] = imb; b.spread[c] = spread; b.top_vol[c] = top; b.book_ok[c] = seen;
            if prev_total.is_finite() { b.net_add[c] = total - prev_total + b.vol[c]; }
            prev_total = total;
        }
    }
    b
}

/// 成交量钟格子：每格成交量相同，字母反映“走完这段量”时的行为。
pub fn volume_bars(trades: &[TradeRecord], l: usize) -> Bars {
    let mut b = Bars::new(l);
    let total: f64 = trades.iter().map(|t| t.volume).sum();
    if total <= 0.0 { return b; }
    let mut cum = vec![0.0f64; trades.len()];
    let mut acc = 0.0;
    for (i, t) in trades.iter().enumerate() { cum[i] = (acc + t.volume * 0.5) / total; acc += t.volume; }
    fill_trades(&mut b, trades, |i, _| ((cum[i] * l as f64) as usize).min(l - 1), true);
    b
}

fn cuts(vals: &[f64], active: impl Fn(usize) -> bool, k: usize, buf: &mut Vec<f64>) -> [f64; 4] {
    buf.clear();
    for (i, &v) in vals.iter().enumerate() { if active(i) && v.is_finite() { buf.push(v); } }
    let mut r = [f64::INFINITY; 4];
    if buf.is_empty() { return r; }
    buf.sort_by(|a, b| a.total_cmp(b));
    for q in 1..k { r[q - 1] = buf[(buf.len() * q / k).min(buf.len() - 1)]; }
    r
}
#[inline]
fn tier(v: f64, c: &[f64; 4], k: usize) -> usize {
    if !v.is_finite() { return k / 2; }
    let mut t = 0;
    for q in 0..k - 1 { if v > c[q] { t = q + 1; } }
    t
}
#[inline]
fn sign3(v: f64, th: f64) -> usize { if v > th { 0 } else if v < -th { 2 } else { 1 } }

/// 用这一天这一段的量算五档阈值，供第二天的 mag5y 字母表当基准。
pub fn mag5_ref_cuts(b: &Bars, buf: &mut Vec<f64>) -> [f64; 4] {
    cuts(&b.mag, |i| b.n[i] > 0.0, 5, buf)
}

/// 生成字母序列，返回非静默格数。阈值默认来自这一天这一段自身；mag5y 用昨天的阈值，保留今昨量级差别。
pub fn letters(b: &Bars, ref_cuts: &[f64; 4], alpha: usize, out: &mut Vec<u8>, buf: &mut Vec<f64>) -> usize {
    let l = b.vol.len();
    out.clear(); out.resize(l, 0);
    let (na, nb) = GRID[alpha];
    let mk = |x: usize, y: usize| (1 + x * nb + y) as u8;
    let has_trade = |i: usize| b.n[i] > 0.0;
    let mut nz = 0usize;
    let mut set = |out: &mut Vec<u8>, i: usize, v: u8| { out[i] = v; nz += 1; };
    match alpha {
        VOLSIG => {
            let c = cuts(&b.mag, has_trade, 3, buf);
            for i in 0..l { if has_trade(i) {
                let s = b.buy[i] + b.sell[i];
                let d = if s > 0.0 { (b.buy[i] - b.sell[i]) / s } else { 0.0 };
                set(out, i, mk(sign3(d, 0.2), tier(b.mag[i], &c, 3)));
            } }
        }
        MAG5 | MAG5Y => {
            let c = if alpha == MAG5 { cuts(&b.mag, has_trade, 5, buf) } else { *ref_cuts };
            for i in 0..l { if has_trade(i) { set(out, i, mk(tier(b.mag[i], &c, 5), 0)); } }
        }
        SIZEL => {
            let avg: Vec<f64> = (0..l).map(|i| if b.n[i] > 0.0 { b.vol[i] / b.n[i] } else { f64::NAN }).collect();
            let (c1, c2) = (cuts(&b.n, has_trade, 3, buf), cuts(&avg, has_trade, 3, buf));
            for i in 0..l { if has_trade(i) { set(out, i, mk(tier(b.n[i], &c1, 3), tier(avg[i], &c2, 3))); } }
        }
        RHYTHM => {
            let (c1, c2) = (cuts(&b.n_event, has_trade, 3, buf), cuts(&b.gap_cv, has_trade, 3, buf));
            for i in 0..l { if has_trade(i) { set(out, i, mk(tier(b.n_event[i], &c1, 3), tier(b.gap_cv[i], &c2, 3))); } }
        }
        OID => {
            let (c1, c2) = (cuts(&b.log_age, has_trade, 3, buf), cuts(&b.switch_rate, has_trade, 3, buf));
            for i in 0..l { if has_trade(i) { set(out, i, mk(tier(b.log_age[i], &c1, 3), tier(b.switch_rate[i], &c2, 3))); } }
        }
        PRICE => {
            let c = cuts(&b.range, has_trade, 3, buf);
            for i in 0..l { if has_trade(i) {
                let s = if b.ret[i].is_finite() { sign3(b.ret[i], 0.0) } else { 1 };
                set(out, i, mk(s, tier(b.range[i], &c, 3)));
            } }
        }
        BOOKFLOW => {
            let ok = |i: usize| b.imb[i].is_finite();
            let (c1, c2) = (cuts(&b.imb, ok, 3, buf), cuts(&b.net_add, ok, 3, buf));
            for i in 0..l { if ok(i) { set(out, i, mk(tier(b.imb[i], &c1, 3), tier(b.net_add[i], &c2, 3))); } }
        }
        BOOKSHAPE => {
            let ok = |i: usize| b.spread[i].is_finite();
            let (c1, c2) = (cuts(&b.spread, ok, 3, buf), cuts(&b.top_vol, ok, 3, buf));
            for i in 0..l { if ok(i) { set(out, i, mk(tier(b.spread[i], &c1, 3), tier(b.top_vol[i], &c2, 3))); } }
        }
        _ => { // CANCEL
            let act = |i: usize| b.cancel[i] > 0.0 || b.n[i] > 0.0;
            let share: Vec<f64> = (0..l).map(|i| {
                let s = b.cancel[i] + b.vol[i];
                if s > 0.0 { b.cancel[i] / s } else { f64::NAN }
            }).collect();
            let c = cuts(&share, act, 3, buf);
            for i in 0..l { if act(i) {
                let s = b.cancel_bid[i] + b.cancel_ask[i];
                let d = if s > 0.0 { (b.cancel_bid[i] - b.cancel_ask[i]) / s } else { 0.0 };
                set(out, i, mk(tier(share[i], &c, 3), sign3(d, 0.2)));
            } }
        }
    }
    let _ = na;
    nz
}
