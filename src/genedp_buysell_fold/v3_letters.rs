//! 第三版字母化层：16 种基础字母 × 中性区阈值档 → 38 个字母版本；另含真实挂撤弧的提取。
//! 凡依赖撤单数据的字母名都以 cxl 开头；撤单数据不可用的日期（沪市 20211105 前、深市 20160104 前）一律返回空。
use super::reader::{MarketRecord, TradeRecord};

pub const BASE: [&str; 16] = [
    "flow", "count", "size", "speed", "ordjump", "cxl_vol", "lob", "ret", "bigord", "volbar", "lob1", "lob5", "lobslope",
    "spread", "cxl_cnt", "cxl_rate",
];
/// 基础中性区阈值；0 表示这种字母没有阈值（按中位数或事件定方向）
const THR: [f64; 16] = [0.1, 0.1, 0.1, 0.0, 0.05, 0.1, 0.01, 0.0, 0.0, 0.1, 0.05, 0.02, 0.02, 0.0, 0.1, 0.0];
pub const MULTS: [(&str, f64); 3] = [("d05", 0.5), ("d10", 1.0), ("d20", 2.0)];

pub fn is_cxl(a: usize) -> bool {
    BASE[a].starts_with("cxl")
}

/// 撤单数据是否存在：沪市 20211105 起，深市 20160104 起（实测普查确定）
pub fn cxl_available(code: &str, date: i64) -> bool {
    if code.starts_with('6') { date >= 20211105 } else { date >= 20160104 }
}

pub struct Variant {
    pub name: String,
    pub alpha: usize,
    pub mult: f64,
}

pub fn variants() -> Vec<Variant> {
    let mut v = Vec::new();
    for a in 0..BASE.len() {
        if THR[a] > 0.0 {
            for (t, m) in MULTS {
                v.push(Variant { name: format!("{}_{}", BASE[a], t), alpha: a, mult: m });
            }
        } else {
            v.push(Variant { name: BASE[a].to_string(), alpha: a, mult: 1.0 });
        }
    }
    v
}

/// 每种基础字母的“基准阈值”版本在 variants() 里的下标
pub fn base_variant(a: usize) -> usize {
    let vs = variants();
    vs.iter().position(|v| v.alpha == a && (v.mult - 1.0).abs() < 1e-12).unwrap()
}

#[derive(Clone)]
pub struct Win {
    pub name: &'static str,
    pub raw_lo: u32,
    pub raw_hi: u32,
    pub adj_lo: i64,
    pub adj_hi: i64,
    pub bar: i64,
}

impl Win {
    pub fn n(&self) -> usize {
        ((self.adj_hi - self.adj_lo) / self.bar) as usize
    }
    #[inline]
    pub fn bar_idx(&self, time_us: i64) -> usize {
        let s = (time_us / 1_000_000).rem_euclid(86400);
        (((s - self.adj_lo) / self.bar).max(0) as usize).min(self.n() - 1)
    }
    pub fn with_bar(&self, bar: i64) -> Win {
        let mut w = self.clone();
        w.bar = bar;
        w
    }
}

/// 三时段（读取器已把 13:00 平移成 11:30，并丢弃 14:57 之后）：
/// 全天 09:30-11:30 + 13:00-14:57，下午 13:00-14:57，尾盘 14:27-14:57。
pub fn windows() -> [Win; 3] {
    [
        Win { name: "full", raw_lo: 34200, raw_hi: 53820, adj_lo: 34200, adj_hi: 48420, bar: 30 },
        Win { name: "pm", raw_lo: 46800, raw_hi: 53820, adj_lo: 41400, adj_hi: 48420, bar: 30 },
        Win { name: "tail", raw_lo: 52020, raw_hi: 53820, adj_lo: 46620, adj_hi: 48420, bar: 15 },
    ]
}
/// 每个时段两档粒度（秒），以及 H 型假结用的约 120 字母粒度
pub const TIER_BARS: [[i64; 2]; 3] = [[30, 60], [30, 60], [15, 5]];
pub const HK_BARS: [i64; 3] = [120, 60, 15];
pub const TIER_TAGS: [&str; 2] = ["g1", "g2"];

#[derive(Clone, Default)]
pub struct Seq {
    pub side: Vec<i8>,
    pub strength: Vec<u8>,
    pub vol: Vec<f32>,
    pub price: Vec<f32>,
}

impl Seq {
    pub fn concat(a: &Seq, b: &Seq) -> Seq {
        let mut s = a.clone();
        s.side.extend_from_slice(&b.side);
        s.strength.extend_from_slice(&b.strength);
        s.vol.extend_from_slice(&b.vol);
        s.price.extend_from_slice(&b.price);
        s
    }
}

#[inline]
fn dead(x: f64, thr: f64) -> i8 {
    if !x.is_finite() || x.abs() <= thr { 0 } else if x > 0.0 { 1 } else { -1 }
}

fn median(mut v: Vec<f64>) -> f64 {
    if v.is_empty() { return f64::NAN; }
    let mid = v.len() / 2;
    let (_, m, _) = v.select_nth_unstable_by(mid, |a, b| a.total_cmp(b));
    *m
}

fn make_seq(side: Vec<i8>, mag: &[f64], vol: &[f64], price: &[f32]) -> Seq {
    let m = median(mag.iter().zip(&side).filter(|(_, &s)| s != 0).map(|(&x, _)| x).collect());
    let strength = mag.iter().map(|&x| if x > m { 2 } else { 1 }).collect();
    Seq { side, strength, vol: vol.iter().map(|&x| x as f32).collect(), price: price.to_vec() }
}

/// 一个时段、一个粒度下的分时片统计量
pub struct Bars {
    n: usize,
    bvol: Vec<f64>,
    svol: Vec<f64>,
    bn: Vec<f64>,
    sn: Vec<f64>,
    bc: Vec<f64>,
    sc: Vec<f64>,
    bcn: Vec<f64>,
    scn: Vec<f64>,
    total: Vec<f64>,
    px: Vec<f32>,
    last_px: Vec<f32>,
    bgap: Vec<f64>,
    bgn: Vec<f64>,
    sgap: Vec<f64>,
    sgn: Vec<f64>,
    /// 盘口：每片最后一个快照的 总量不平衡/一档/前五档/前三档集中度差/相对价差，及时段内第一个有效值
    mk_last: [Vec<f64>; 5],
    mk_first: [f64; 5],
    events: Vec<(i8, f64, f64)>,
    trades: Vec<(i8, f64, f64, i64)>,
}

fn snapshot_measures(m: &MarketRecord) -> [f64; 5] {
    let r = |a: f64, b: f64| if a + b > 0.0 { (a - b) / (a + b) } else { f64::NAN };
    let tot = r(m.total_bid_vol, m.total_ask_vol);
    let l1 = r(m.bid_vols[0], m.ask_vols[0]);
    let (b5, a5): (f64, f64) = (m.bid_vols[..5].iter().sum(), m.ask_vols[..5].iter().sum());
    let l5 = r(b5, a5);
    let (b3, a3): (f64, f64) = (m.bid_vols[..3].iter().sum(), m.ask_vols[..3].iter().sum());
    let (b10, a10): (f64, f64) = (m.bid_vols.iter().sum(), m.ask_vols.iter().sum());
    let slope = if b10 > 0.0 && a10 > 0.0 { b3 / b10 - a3 / a10 } else { f64::NAN };
    let (a1, b1) = (m.ask_prcs[0], m.bid_prcs[0]);
    let spread = if a1 > 0.0 && b1 > 0.0 && a1 >= b1 { (a1 - b1) / ((a1 + b1) * 0.5) } else { f64::NAN };
    [tot, l1, l5, slope, spread]
}

impl Bars {
    pub fn new(tr: &[TradeRecord], mk: &[MarketRecord], win: &Win) -> Option<Bars> {
        let n = win.n();
        let z = || vec![0f64; n];
        let mut b = Bars {
            n,
            bvol: z(), svol: z(), bn: z(), sn: z(), bc: z(), sc: z(), bcn: z(), scn: z(), total: z(),
            px: vec![f32::NAN; n], last_px: vec![f32::NAN; n],
            bgap: z(), bgn: z(), sgap: z(), sgn: z(),
            mk_last: [vec![f64::NAN; n], vec![f64::NAN; n], vec![f64::NAN; n], vec![f64::NAN; n], vec![f64::NAN; n]],
            mk_first: [f64::NAN; 5],
            events: Vec::with_capacity(tr.len() / 2),
            trades: Vec::with_capacity(tr.len()),
        };
        let mut amt = z();
        let (mut prev_flag, mut prev_id, mut last_buy, mut last_sell) = (0i32, -1i64, -1i64, -1i64);
        for t in tr {
            if t.raw_sec < win.raw_lo || t.raw_sec > win.raw_hi { continue; }
            let k = win.bar_idx(t.time_us);
            if t.flag == 32 {
                if t.bid_order > 0 && t.ask_order == 0 { b.bc[k] += t.volume; b.bcn[k] += 1.0; }
                else if t.ask_order > 0 && t.bid_order == 0 { b.sc[k] += t.volume; b.scn[k] += 1.0; }
                continue;
            }
            if t.price <= 0.0 || t.volume <= 0.0 || (t.flag != 66 && t.flag != 83) { continue; }
            let is_buy = t.flag == 66;
            amt[k] += t.turnover;
            b.last_px[k] = t.price as f32;
            let id = if is_buy { t.bid_order } else { t.ask_order };
            if is_buy {
                b.bvol[k] += t.volume;
                b.bn[k] += 1.0;
                if id != last_buy {
                    if last_buy >= 0 && id > last_buy { b.bgap[k] += ((id - last_buy) as f64).ln(); b.bgn[k] += 1.0; }
                    last_buy = id;
                }
            } else {
                b.svol[k] += t.volume;
                b.sn[k] += 1.0;
                if id != last_sell {
                    if last_sell >= 0 && id > last_sell { b.sgap[k] += ((id - last_sell) as f64).ln(); b.sgn[k] += 1.0; }
                    last_sell = id;
                }
            }
            if t.flag == prev_flag && id == prev_id {
                let e = b.events.last_mut().unwrap();
                e.1 += t.volume;
                e.2 += t.turnover;
            } else {
                b.events.push((if is_buy { 1 } else { -1 }, t.volume, t.turnover));
                prev_flag = t.flag;
                prev_id = id;
            }
            b.trades.push((if is_buy { 1 } else { -1 }, t.volume, t.turnover, t.time_us));
        }
        if b.trades.is_empty() { return None; }
        for i in 0..n {
            b.total[i] = b.bvol[i] + b.svol[i];
            if b.total[i] > 0.0 { b.px[i] = (amt[i] / b.total[i]) as f32; }
        }
        let mut cur = b.px.iter().copied().find(|x| x.is_finite()).unwrap();
        for p in b.px.iter_mut() {
            if p.is_finite() { cur = *p; } else { *p = cur; }
        }
        for m in mk {
            if m.raw_sec < win.raw_lo || m.raw_sec > win.raw_hi { continue; }
            let k = win.bar_idx(m.time_us);
            let v = snapshot_measures(m);
            for q in 0..5 {
                if v[q].is_finite() {
                    if !b.mk_first[q].is_finite() { b.mk_first[q] = v[q]; }
                    b.mk_last[q][k] = v[q];
                }
            }
        }
        Some(b)
    }

    fn delta(&self, q: usize) -> Option<Vec<f64>> {
        if !self.mk_first[q].is_finite() { return None; }
        let mut prev = self.mk_first[q];
        let mut d = vec![0f64; self.n];
        for i in 0..self.n {
            let x = self.mk_last[q][i];
            if x.is_finite() { d[i] = x - prev; prev = x; }
        }
        Some(d)
    }

    /// 生成一个字母版本；cxl_ok=false 时撤单类字母返回空
    pub fn seq(&self, v: &Variant, cxl_ok: bool) -> Option<Seq> {
        let a = v.alpha;
        if is_cxl(a) && !cxl_ok { return None; }
        let n = self.n;
        let thr = THR[a] * v.mult;
        let ratio = |x: f64, y: f64| if x + y > 0.0 { (x - y) / (x + y) } else { 0.0 };
        let total = &self.total;
        let px = &self.px;
        let by_median = |vals: &[f64], higher_is_plus: bool| -> Seq {
            let med = median(vals.iter().copied().filter(|x| x.is_finite()).collect());
            let side: Vec<i8> = vals.iter().map(|&x| {
                if !x.is_finite() || x == med { 0 } else if (x > med) == higher_is_plus { 1 } else { -1 }
            }).collect();
            let mag: Vec<f64> = vals.iter().map(|&x| if x.is_finite() { (x - med).abs() } else { 0.0 }).collect();
            make_seq(side, &mag, total, px)
        };
        Some(match BASE[a] {
            "flow" => {
                let side = (0..n).map(|i| dead(ratio(self.bvol[i], self.svol[i]), thr)).collect();
                make_seq(side, total, total, px)
            }
            "count" => {
                let side = (0..n).map(|i| dead(ratio(self.bn[i], self.sn[i]), thr)).collect();
                let mag: Vec<f64> = (0..n).map(|i| self.bn[i] + self.sn[i]).collect();
                make_seq(side, &mag, total, px)
            }
            "size" => {
                let xs: Vec<f64> = (0..n).map(|i| {
                    if self.bn[i] > 0.0 && self.sn[i] > 0.0 { ((self.bvol[i] / self.bn[i]) / (self.svol[i] / self.sn[i])).ln() } else { 0.0 }
                }).collect();
                let side = xs.iter().map(|&x| dead(x, thr)).collect();
                let mag: Vec<f64> = xs.iter().map(|x| x.abs()).collect();
                make_seq(side, &mag, total, px)
            }
            "speed" => {
                let cnt: Vec<f64> = (0..n).map(|i| self.bn[i] + self.sn[i]).collect();
                by_median(&cnt, true)
            }
            "ordjump" => {
                let xs: Vec<f64> = (0..n).map(|i| {
                    if self.bgn[i] > 0.0 && self.sgn[i] > 0.0 { self.sgap[i] / self.sgn[i] - self.bgap[i] / self.bgn[i] } else { 0.0 }
                }).collect();
                let side = xs.iter().map(|&x| dead(x, thr)).collect();
                let mag: Vec<f64> = xs.iter().map(|x| x.abs()).collect();
                make_seq(side, &mag, total, px)
            }
            "cxl_vol" => {
                let cv: Vec<f64> = (0..n).map(|i| self.bc[i] + self.sc[i]).collect();
                let side = (0..n).map(|i| dead(ratio(self.sc[i], self.bc[i]), thr)).collect();
                make_seq(side, &cv, &cv, px)
            }
            "cxl_cnt" => {
                let cc: Vec<f64> = (0..n).map(|i| self.bcn[i] + self.scn[i]).collect();
                let cv: Vec<f64> = (0..n).map(|i| self.bc[i] + self.sc[i]).collect();
                let side = (0..n).map(|i| dead(ratio(self.scn[i], self.bcn[i]), thr)).collect();
                make_seq(side, &cc, &cv, px)
            }
            "cxl_rate" => {
                let rate: Vec<f64> = (0..n).map(|i| {
                    let c = self.bc[i] + self.sc[i];
                    if c + total[i] > 0.0 { c / (c + total[i]) } else { f64::NAN }
                }).collect();
                by_median(&rate, true)
            }
            "lob" | "lob1" | "lob5" | "lobslope" => {
                let q = match BASE[a] { "lob" => 0, "lob1" => 1, "lob5" => 2, _ => 3 };
                let d = self.delta(q)?;
                let side = d.iter().map(|&x| dead(x, thr)).collect();
                let mag: Vec<f64> = d.iter().map(|x| x.abs()).collect();
                make_seq(side, &mag, total, px)
            }
            "spread" => {
                if !self.mk_first[4].is_finite() { return None; }
                by_median(&self.mk_last[4], false) // 价差比中位数窄记 +1
            }
            "ret" => {
                let mut prev = f32::NAN;
                let mut xs = vec![0f64; n];
                for i in 0..n {
                    if self.last_px[i].is_finite() {
                        if prev.is_finite() { xs[i] = (self.last_px[i] / prev - 1.0) as f64; }
                        prev = self.last_px[i];
                    }
                }
                let side = xs.iter().map(|&x| dead(x, 0.0)).collect();
                let mag: Vec<f64> = xs.iter().map(|x| x.abs()).collect();
                make_seq(side, &mag, total, px)
            }
            "bigord" => {
                let ev = &self.events;
                let mut idx: Vec<u32> = (0..ev.len() as u32).collect();
                if idx.len() > n {
                    idx.select_nth_unstable_by(n - 1, |&x, &y| {
                        ev[y as usize].1.total_cmp(&ev[x as usize].1).then(x.cmp(&y))
                    });
                    idx.truncate(n);
                }
                idx.sort_unstable();
                let side = idx.iter().map(|&i| ev[i as usize].0).collect();
                let vol: Vec<f64> = idx.iter().map(|&i| ev[i as usize].1).collect();
                let price: Vec<f32> = idx.iter().map(|&i| (ev[i as usize].2 / ev[i as usize].1) as f32).collect();
                make_seq(side, &vol, &vol, &price)
            }
            "volbar" => {
                let v_total: f64 = self.trades.iter().map(|t| t.1).sum();
                let (mut bv, mut sv, mut a2) = (vec![0f64; n], vec![0f64; n], vec![0f64; n]);
                let (mut fst, mut lst) = (vec![i64::MAX; n], vec![i64::MIN; n]);
                let mut cum = 0f64;
                for t in &self.trades {
                    let k = ((cum / v_total * n as f64) as usize).min(n - 1);
                    if t.0 > 0 { bv[k] += t.1; } else { sv[k] += t.1; }
                    a2[k] += t.2;
                    fst[k] = fst[k].min(t.3);
                    lst[k] = lst[k].max(t.3);
                    cum += t.1;
                }
                let tot2: Vec<f64> = (0..n).map(|i| bv[i] + sv[i]).collect();
                let mut px2 = vec![f32::NAN; n];
                for i in 0..n {
                    if tot2[i] > 0.0 { px2[i] = (a2[i] / tot2[i]) as f32; }
                }
                let mut cur = px2.iter().copied().find(|x| x.is_finite()).unwrap();
                for p in px2.iter_mut() {
                    if p.is_finite() { cur = *p; } else { *p = cur; }
                }
                let side = (0..n).map(|i| dead(ratio(bv[i], sv[i]), thr)).collect();
                let mag: Vec<f64> = (0..n).map(|i| if tot2[i] > 0.0 { 1.0 / (1.0 + (lst[i] - fst[i]) as f64 / 1e6) } else { 0.0 }).collect();
                make_seq(side, &mag, &tot2, &px2)
            }
            _ => unreachable!(),
        })
    }
}

/// 一条真实挂撤弧：挂单时刻（由委托编号推算）到撤单时刻
pub struct Arc {
    pub start_us: i64,
    pub end_us: i64,
    pub buy: bool,
    pub vol: f64,
    pub unmapped: bool,
    /// 挂单时刻早于时段开始（开始端被截到时段起点）
    pub clipped: bool,
}

/// 预构建当天主动成交“委托编号 → 时刻”有序表（三个时段共用一次排序，不再重复排序 3 次）。
pub fn build_cancel_map(tr: &[TradeRecord]) -> Vec<(i64, i64)> {
    let mut map: Vec<(i64, i64)> = Vec::with_capacity(tr.len());
    for t in tr {
        if t.flag == 66 || t.flag == 83 {
            let oid = if t.flag == 66 { t.bid_order } else { t.ask_order };
            if oid > 0 {
                map.push((oid, t.time_us));
            }
        }
    }
    map.sort_unstable();
    map
}

pub fn cancel_arcs_with_map(tr: &[TradeRecord], map: &[(i64, i64)], win: &Win) -> Vec<Arc> {
    let lo_us = {
        let t0 = tr.first().map(|t| t.time_us).unwrap_or(0);
        t0 - t0.rem_euclid(86_400_000_000) + win.adj_lo * 1_000_000
    };
    let mut out = Vec::new();
    for t in tr {
        if t.flag != 32 || t.raw_sec < win.raw_lo || t.raw_sec > win.raw_hi { continue; }
        let (buy, id) = if t.bid_order > 0 && t.ask_order == 0 { (true, t.bid_order) }
            else if t.ask_order > 0 && t.bid_order == 0 { (false, t.ask_order) } else { continue };
        let p = map.partition_point(|x| x.0 <= id);
        let (start, unmapped, clipped) = if p == 0 { (lo_us, true, true) } else {
            let raw = map[p - 1].1;
            (raw.max(lo_us).min(t.time_us), false, raw < lo_us)
        };
        out.push(Arc { start_us: start, end_us: t.time_us, buy, vol: t.volume, unmapped, clipped });
    }
    out
}

/// 由同一只股票当天的主动成交建立“委托编号 → 时刻”对照（编号随时间递增），
/// 撤单的挂单时刻取编号不超过它的最后一个主动委托的时刻；编号早于所有主动委托（如集合竞价挂单）记为无法定位。
pub fn cancel_arcs(tr: &[TradeRecord], win: &Win) -> Vec<Arc> {
    let map = build_cancel_map(tr);
    cancel_arcs_with_map(tr, &map, win)
}
