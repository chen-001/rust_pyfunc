//! 最优路径（Dijkstra）横截面因子：把 Level2 逐笔按 27 种口径分桶当节点，
//! 相邻窗口连边跑最短路，从图与路径里取 7422 个连续量。
//!
//! 数据流：rayon 并行读全市场逐笔+盘口 → per-stock 建图求最短路 → 横截面 z-score。
//!
//! 节点口径（27 种）：价格档/毛分位/分的个位/相对开盘位置/相对 VWAP 偏离/价×量复合、
//! 量桶/相对量速/额桶/成交进度、急迫度/方向×量/跳价方向、秒数/毫秒位/间隔/
//! 分钟内位置/间隔×方向、价差/盘口不平衡/可观测挂单比例、委托量规模/近 20 笔收益/
//! 近 20 笔波动、三元状态/价×时/节奏×方向。
//!
//! 两套时间窗口（w8 = 30 分钟一格，w32 = 7.4 分钟一格）。
//! 每个节点每套窗口固定输出 137 个量；价格档与相对开盘位置另加 6 个（共 143）。

use crate::fast_csv_reader::{read_market_fast_inner, read_trade_fast_inner, MarketRecord, TradeRecord};
use pyo3::prelude::*;
use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::{BTreeMap, BinaryHeap, HashMap};

/// 本地 f64 副本：主库 TradeRecord/MarketRecord 用 f32 存价量额，转一次即可。
#[derive(Clone)]
struct Tr {
    time_us: i64,
    price: f64,
    volume: f64,
    turnover: f64,
    flag: i32,
    bid_order: i64,
    ask_order: i64,
}

struct Mk {
    time_us: i64,
    ask_prcs: [f64; 10],
    ask_vols: [f64; 10],
    bid_prcs: [f64; 10],
    bid_vols: [f64; 10],
    total_ask_vol: f64,
    total_bid_vol: f64,
}

fn conv_trade(t: &TradeRecord) -> Tr {
    Tr {
        time_us: t.time_us,
        price: t.price as f64,
        volume: t.volume as f64,
        turnover: t.turnover as f64,
        flag: t.flag,
        bid_order: t.bid_order,
        ask_order: t.ask_order,
    }
}

fn conv_market(m: &MarketRecord) -> Mk {
    let mut ask_prcs = [0.0f64; 10];
    let mut ask_vols = [0.0f64; 10];
    let mut bid_prcs = [0.0f64; 10];
    let mut bid_vols = [0.0f64; 10];
    for i in 0..10 {
        ask_prcs[i] = m.ask_prcs[i] as f64;
        ask_vols[i] = m.ask_vols[i] as f64;
        bid_prcs[i] = m.bid_prcs[i] as f64;
        bid_vols[i] = m.bid_vols[i] as f64;
    }
    Mk {
        time_us: m.time_us,
        ask_prcs,
        ask_vols,
        bid_prcs,
        bid_vols,
        total_ask_vol: m.total_ask_vol as f64,
        total_bid_vol: m.total_bid_vol as f64,
    }
}

const WINDOW_COUNTS: [usize; 2] = [8, 32];
const ALPHA: f64 = 0.1;
const BETA: f64 = 5.0;
const STATIC_LAM: f64 = 0.5;
/// 每个节点每套窗口的固定输出数（不含价格类专属的 6 个）。
const PER_NODE: usize = 137;
/// 价格档与相对开盘位置额外输出 6 个（探路/偏离类）。
const PER_NODE_PRICE: usize = 6;
/// 27 节点 × 2 窗口 × 137 + 2 节点 × 2 窗口 × 6 = 7422
pub const N_FACTORS: usize = 7422;

const SERIES_OPS: [&str; 16] = [
    "mean", "std", "min", "max", "range", "slope", "ratio_half", "q25", "q75", "iqr_r", "skew",
    "cv", "ac1", "centroid", "mono", "dd",
];
const QUANTITIES: [&str; 5] = ["wid", "pop", "ent", "top3", "peak"];
const PATH_FIELDS: [&str; 11] = [
    "cost_norm", "gapcost_share", "stationary", "traj_mean_corr", "traj_disp", "traj_amp",
    "traj_smooth", "move_center", "dev_end", "dev_mean", "dev_std",
];
const ADV_FIELDS: [&str; 3] = ["adv_rand", "adv_hot", "adv_cold"];
const UNIQ_FIELDS: [&str; 2] = ["uniq_gap", "uniq_abs"];
const GRAPH_FIELDS: [&str; 7] = [
    "conn_density", "bottleneck", "diam_proxy", "reach", "deg_hhi", "deg_max", "flow_asym",
];
const PRICE_FIELDS: [&str; 6] = [
    "probe_top", "probe_bot", "gap_mean", "gap_std", "gap_ac1", "traj_price_corr",
];

#[derive(Clone, Copy, PartialEq)]
enum Node {
    P1, P2, P3, P4, P5, P6,
    V1, V2, V3, V4,
    U1, U2, U3,
    T1, T2, T3, T4, T5,
    B1, B2, B3,
    O1, R1, R2,
    C1, C2, C3,
}

const ALL_NODES: [Node; 27] = [
    Node::P1, Node::P2, Node::P3, Node::P4, Node::P5, Node::P6,
    Node::V1, Node::V2, Node::V3, Node::V4,
    Node::U1, Node::U2, Node::U3,
    Node::T1, Node::T2, Node::T3, Node::T4, Node::T5,
    Node::B1, Node::B2, Node::B3,
    Node::O1, Node::R1, Node::R2,
    Node::C1, Node::C2, Node::C3,
];

fn node_name(n: Node) -> &'static str {
    match n {
        Node::P1 => "p1price", Node::P2 => "p2dime", Node::P3 => "p3cent",
        Node::P4 => "p4rel", Node::P5 => "p5vwap", Node::P6 => "p6pxvol",
        Node::V1 => "v1vol", Node::V2 => "v2rvol", Node::V3 => "v3amt", Node::V4 => "v4cum",
        Node::U1 => "u1urg", Node::U2 => "u2dirvol", Node::U3 => "u3jump",
        Node::T1 => "t1sec", Node::T2 => "t2ms", Node::T3 => "t3gap",
        Node::T4 => "t4inmin", Node::T5 => "t5gapdir",
        Node::B1 => "b1spread", Node::B2 => "b2imb", Node::B3 => "b3obs",
        Node::O1 => "o1order", Node::R1 => "r1ret", Node::R2 => "r2rvol",
        Node::C1 => "c1dvs", Node::C2 => "c2pxsec", Node::C3 => "c3rhythm",
    }
}

fn is_ordinal(n: Node) -> bool {
    matches!(n,
        Node::P1 | Node::P2 | Node::P3 | Node::P4 | Node::P5 | Node::P6 |
        Node::V1 | Node::V2 | Node::V3 | Node::V4 |
        Node::U1 | Node::T3 | Node::T4 | Node::T5 |
        Node::B1 | Node::B2 | Node::B3 | Node::O1 | Node::R1 | Node::R2)
}

fn needs_market(n: Node) -> bool { matches!(n, Node::B1 | Node::B2 | Node::B3) }

fn is_price_node(n: Node) -> bool { matches!(n, Node::P1 | Node::P4) }

// ============================================================
// 因子名（单一来源，顺序即输出顺序）
// ============================================================

pub fn dijkstra_names() -> Vec<String> {
    let mut names = Vec::with_capacity(N_FACTORS);
    for &nw in WINDOW_COUNTS.iter() {
        for &node in ALL_NODES.iter() {
            let pfx = format!("{}_{}", node_name(node), format!("w{}", nw));
            for q in QUANTITIES.iter() {
                for op in SERIES_OPS.iter() {
                    names.push(format!("{}_{}_{}", pfx, q, op));
                }
            }
            for op in SERIES_OPS.iter() {
                names.push(format!("{}_ret_{}", pfx, op));
            }
            names.push(format!("{}_churn_pulse", pfx));
            names.push(format!("{}_churn_center", pfx));
            for f in PATH_FIELDS.iter() {
                names.push(format!("{}_{}", pfx, f));
            }
            for op in SERIES_OPS.iter() {
                names.push(format!("{}_step_{}", pfx, op));
            }
            for f in ADV_FIELDS.iter() {
                names.push(format!("{}_{}", pfx, f));
            }
            for f in UNIQ_FIELDS.iter() {
                names.push(format!("{}_{}", pfx, f));
            }
            for f in GRAPH_FIELDS.iter() {
                names.push(format!("{}_{}", pfx, f));
            }
            if is_price_node(node) {
                for f in PRICE_FIELDS.iter() {
                    names.push(format!("{}_{}", pfx, f));
                }
            }
        }
    }
    debug_assert_eq!(names.len(), N_FACTORS, "因子名数量与 N_FACTORS 不一致");
    names
}

// ============================================================
// 统计工具
// ============================================================

fn mean(xs: &[f64]) -> f64 { if xs.is_empty() { 0.0 } else { xs.iter().sum::<f64>() / xs.len() as f64 } }

fn stddev(xs: &[f64]) -> f64 {
    if xs.len() < 2 { return 0.0; }
    let m = mean(xs);
    (xs.iter().map(|x| (x - m).powi(2)).sum::<f64>() / xs.len() as f64).sqrt()
}

fn entropy(counts: &[f64]) -> f64 {
    let tot: f64 = counts.iter().sum();
    if tot <= 0.0 { return 0.0; }
    counts.iter().filter(|&&c| c > 0.0).map(|&c| { let p = c / tot; -p * p.ln() }).sum()
}

fn slope(xs: &[f64]) -> f64 {
    let n = xs.len();
    if n < 2 { return 0.0; }
    let mx = (n - 1) as f64 / 2.0;
    let my = xs.iter().sum::<f64>() / n as f64;
    let (mut num, mut den) = (0.0, 0.0);
    for (i, &v) in xs.iter().enumerate() {
        let dx = i as f64 - mx;
        num += dx * (v - my);
        den += dx * dx;
    }
    if den < 1e-12 { 0.0 } else { num / den }
}

fn skewness(xs: &[f64]) -> f64 {
    if xs.len() < 3 { return 0.0; }
    let (m, s) = (mean(xs), stddev(xs));
    if s < 1e-12 { return 0.0; }
    xs.iter().map(|x| ((x - m) / s).powi(3)).sum::<f64>() / xs.len() as f64
}

fn corr(a: &[f64], b: &[f64]) -> f64 {
    if a.len() < 3 || a.len() != b.len() { return f64::NAN; }
    let (ma, mb) = (mean(a), mean(b));
    let (mut num, mut sa, mut sb) = (0.0, 0.0, 0.0);
    for i in 0..a.len() {
        num += (a[i] - ma) * (b[i] - mb);
        sa += (a[i] - ma).powi(2);
        sb += (b[i] - mb).powi(2);
    }
    if sa < 1e-12 || sb < 1e-12 { f64::NAN } else { num / (sa * sb).sqrt() }
}

fn autocorr1(xs: &[f64]) -> f64 {
    if xs.len() < 4 { return f64::NAN; }
    corr(&xs[..xs.len() - 1], &xs[1..])
}

fn quantile_sorted(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() { return 0.0; }
    let pos = (sorted.len() - 1) as f64 * q;
    let lo = pos.floor() as usize;
    let hi = (pos.ceil() as usize).min(sorted.len() - 1);
    let w = pos - lo as f64;
    sorted[lo] * (1.0 - w) + sorted[hi] * w
}

fn max_drawdown(xs: &[f64]) -> f64 {
    let (mut peak, mut dd) = (f64::NEG_INFINITY, 0.0f64);
    for &v in xs {
        if v > peak { peak = v; }
        if peak > 1e-12 { dd = dd.max((peak - v) / peak); }
    }
    dd
}

/// 把一个逐层序列压成 16 个连续算子，按 SERIES_OPS 顺序追加。
fn push_series(out: &mut Vec<f64>, s: &[f64]) {
    let n = s.len();
    if n < 3 || s.iter().all(|x| !x.is_finite()) {
        out.extend(std::iter::repeat(f64::NAN).take(SERIES_OPS.len()));
        return;
    }
    let mut sorted: Vec<f64> = s.iter().copied().filter(|v| v.is_finite()).collect();
    if sorted.len() < 3 {
        out.extend(std::iter::repeat(f64::NAN).take(SERIES_OPS.len()));
        return;
    }
    sorted.sort_by(|a, b| a.total_cmp(b));
    let m = mean(s);
    let sd = stddev(s);
    out.push(m);
    out.push(sd);
    out.push(sorted[0]);
    out.push(sorted[n - 1]);
    out.push(sorted[n - 1] - sorted[0]);
    out.push(slope(s));
    let half = n / 2;
    let (h1, h2) = (mean(&s[..half]), mean(&s[half..]));
    out.push(if h2.abs() > 1e-9 { h1 / h2 } else { f64::NAN });
    out.push(quantile_sorted(&sorted, 0.25));
    out.push(quantile_sorted(&sorted, 0.75));
    out.push(if m.abs() > 1e-9 {
        (quantile_sorted(&sorted, 0.75) - quantile_sorted(&sorted, 0.25)) / m
    } else { f64::NAN });
    out.push(skewness(s));
    out.push(if m.abs() > 1e-9 { sd / m } else { f64::NAN });
    out.push(autocorr1(s));
    let tot: f64 = s.iter().sum();
    out.push(if tot.abs() > 1e-12 {
        s.iter().enumerate().map(|(i, &v)| i as f64 * v).sum::<f64>() / tot
    } else { f64::NAN });
    let idx: Vec<f64> = (0..n).map(|i| i as f64).collect();
    out.push(corr(s, &idx));
    out.push(max_drawdown(s));
}

struct HeapItem { d: f64, u: usize }
impl PartialEq for HeapItem { fn eq(&self, o: &Self) -> bool { self.d == o.d && self.u == o.u } }
impl Eq for HeapItem {}
impl PartialOrd for HeapItem { fn partial_cmp(&self, o: &Self) -> Option<Ordering> { Some(self.cmp(o)) } }
impl Ord for HeapItem {
    fn cmp(&self, o: &Self) -> Ordering {
        o.d.total_cmp(&self.d).then(self.u.cmp(&o.u))
    }
}

fn transition_cost(trans: &HashMap<(usize, usize), u64>, nb: usize, src: usize) -> (f64, f64) {
    let mut adj: Vec<Vec<(usize, f64)>> = vec![Vec::new(); nb];
    for (&(x, y), &c) in trans {
        if x != y {
            let w = STATIC_LAM + 1.0 / c as f64;
            adj[x].push((y, w));
            adj[y].push((x, w));
        }
    }
    let mut dist = vec![f64::INFINITY; nb];
    dist[src] = 0.0;
    let mut pq = BinaryHeap::new();
    pq.push(HeapItem { d: 0.0, u: src });
    while let Some(HeapItem { d, u }) = pq.pop() {
        if d > dist[u] { continue; }
        for &(v, w) in &adj[u] {
            if d + w < dist[v] {
                dist[v] = d + w;
                pq.push(HeapItem { d: d + w, u: v });
            }
        }
    }
    let fin: Vec<f64> = dist.iter().filter(|x| x.is_finite()).copied().collect();
    (fin.iter().cloned().fold(0.0, f64::max), fin.len() as f64 / nb.max(1) as f64)
}

// ============================================================
// 分桶
// ============================================================

struct Ctx {
    rel_px: Vec<f64>,
    rel_vwap: Vec<f64>,
    cum_frac: Vec<f64>,
    ret20: Vec<f64>,
    rvol20: Vec<f64>,
}

fn build_ctx(tr: &[Tr]) -> Ctx {
    let n = tr.len();
    let open = tr[0].price;
    let total_turnover: f64 = tr.iter().map(|t| t.turnover).sum();
    let total_vol: f64 = tr.iter().map(|t| t.volume).sum();
    let vwap = if total_vol > 0.0 { total_turnover / total_vol } else { open };

    let mut cum = vec![0.0f64; n];
    let mut acc = 0.0;
    for i in 0..n {
        acc += tr[i].volume;
        cum[i] = if total_vol > 0.0 { acc / total_vol } else { 0.0 };
    }

    let mut ret20 = vec![0.0f64; n];
    let mut rvol20 = vec![0.0f64; n];
    for i in 0..n {
        let j = i.saturating_sub(20);
        if i > j {
            ret20[i] = (tr[i].price - tr[j].price) / tr[j].price;
            let mut s = 0.0;
            let mut c = 0;
            for k in (j + 1)..=i {
                let r = (tr[k].price - tr[k - 1].price) / tr[k - 1].price;
                s += r * r;
                c += 1;
            }
            rvol20[i] = if c > 0 { (s / c as f64).sqrt() } else { 0.0 };
        }
    }

    Ctx {
        rel_px: tr.iter().map(|t| (t.price - open) / open).collect(),
        rel_vwap: tr.iter().map(|t| (t.price - vwap) / vwap).collect(),
        cum_frac: cum,
        ret20,
        rvol20,
    }
}

fn quantile_buckets(vals: &[f64], nb: usize) -> Vec<i64> {
    let mut idx: Vec<usize> = (0..vals.len()).collect();
    idx.sort_by(|&a, &b| vals[a].total_cmp(&vals[b]));
    let mut out = vec![0i64; vals.len()];
    for (r, &i) in idx.iter().enumerate() {
        out[i] = (r * nb / vals.len().max(1)).min(nb - 1) as i64;
    }
    out
}

fn align_market(tr: &[Tr], mk: &[Mk], f: &dyn Fn(&Mk) -> i64) -> Vec<i64> {
    let mut out = vec![-1i64; tr.len()];
    let mut j = 0usize;
    for (i, t) in tr.iter().enumerate() {
        while j < mk.len() && mk[j].time_us <= t.time_us { j += 1; }
        if j > 0 { out[i] = f(&mk[j - 1]); }
    }
    out
}

fn make_buckets(node: Node, tr: &[Tr], mk: &[Mk], ctx: &Ctx) -> Vec<i64> {
    let n = tr.len();
    match node {
        Node::P1 => tr.iter().map(|t| (t.price * 100.0).round() as i64).collect(),
        Node::P2 => tr.iter().map(|t| ((t.price * 100.0).round() as i64 / 10) % 10).collect(),
        Node::P3 => tr.iter().map(|t| (t.price * 100.0).round() as i64 % 10).collect(),
        Node::P4 => quantile_buckets(&ctx.rel_px, 25),
        Node::P5 => quantile_buckets(&ctx.rel_vwap, 25),
        Node::P6 => {
            let mut ps: Vec<f64> = tr.iter().map(|t| t.price).collect();
            ps.sort_by(|a, b| a.total_cmp(b));
            let (lo, hi) = (ps[0], ps[n - 1]);
            let span = (hi - lo).max(1e-9);
            tr.iter().map(|t| {
                let pb = (((t.price - lo) / span) * 9.99) as i64;
                let vb: i64 = if t.volume < 1000.0 { 0 } else if t.volume < 10000.0 { 1 } else { 2 };
                pb * 3 + vb
            }).collect()
        }
        Node::V1 => tr.iter().map(|t| {
            let v = t.volume;
            if v < 500.0 { 0 } else if v < 1000.0 { 1 } else if v < 3000.0 { 2 }
            else if v < 10000.0 { 3 } else if v < 50000.0 { 4 } else { 5 }
        }).collect(),
        Node::V2 => {
            let mut out = vec![0f64; n];
            let (mut j, mut acc) = (0usize, 0.0f64);
            for i in 0..n {
                while tr[j].time_us + 60_000_000 < tr[i].time_us {
                    acc += tr[j].volume;
                    j += 1;
                }
                out[i] = if acc > 0.0 { tr[i].volume / acc } else { 0.0 };
            }
            quantile_buckets(&out, 20)
        }
        Node::V3 => tr.iter().map(|t| {
            let v = t.turnover;
            if v < 5e4 { 0 } else if v < 2e5 { 1 } else if v < 5e5 { 2 }
            else if v < 2e6 { 3 } else if v < 1e7 { 4 } else { 5 }
        }).collect(),
        Node::V4 => quantile_buckets(&ctx.cum_frac, 20),
        Node::U1 => {
            let v: Vec<f64> = tr.iter().map(|t| {
                let d = (t.ask_order + t.bid_order) as f64;
                if d > 0.0 { (t.ask_order - t.bid_order) as f64 / d } else { 0.0 }
            }).collect();
            quantile_buckets(&v, 40)
        }
        Node::U2 => tr.iter().map(|t| {
            let dir = if t.flag == 66 { 0i64 } else { 1 };
            let vb: i64 = if t.volume < 500.0 { 0 } else if t.volume < 3000.0 { 1 } else { 2 };
            dir * 3 + vb
        }).collect(),
        Node::U3 => {
            let mut out = vec![0i64; n];
            for i in 1..n {
                let d = tr[i].price - tr[i - 1].price;
                out[i] = if d > 0.0 { 2 } else if d < 0.0 { 0 } else { 1 };
            }
            out
        }
        Node::T1 => tr.iter().map(|t| ((t.time_us / 1_000_000) % 60) as i64).collect(),
        Node::T2 => tr.iter().map(|t| ((t.time_us / 1_000) % 1000 / 100) as i64).collect(),
        Node::T3 => {
            let mut out = vec![3i64; n];
            for i in 1..n {
                let d = tr[i].time_us - tr[i - 1].time_us;
                out[i] = if d < 10_000 { 0 } else if d < 100_000 { 1 }
                    else if d < 1_000_000 { 2 } else if d < 10_000_000 { 3 } else { 4 };
            }
            out
        }
        Node::T4 => tr.iter().map(|t| {
            let sec = (t.time_us / 1_000_000) % 60;
            if sec < 20 { 0 } else if sec < 40 { 1 } else { 2 }
        }).collect(),
        Node::T5 => {
            let mut out = vec![0i64; n];
            for i in 1..n {
                let d = tr[i].time_us - tr[i - 1].time_us;
                let g: i64 = if d < 100_000 { 0 } else if d < 1_000_000 { 1 } else { 2 };
                let dir: i64 = if tr[i].flag == 66 { 0 } else { 1 };
                out[i] = dir * 3 + g;
            }
            out
        }
        Node::B1 => align_market(tr, mk, &|s: &Mk| {
            let sp = s.ask_prcs[0] - s.bid_prcs[0];
            if sp <= 0.011 { 0 } else if sp <= 0.021 { 1 } else { 2 }
        }),
        Node::B2 => align_market(tr, mk, &|s: &Mk| {
            let d = s.bid_vols[0] + s.ask_vols[0];
            if d <= 0.0 { 4 } else {
                let imb = (s.bid_vols[0] - s.ask_vols[0]) / d;
                ((imb + 1.0) * 4.0).clamp(0.0, 7.99) as i64
            }
        }),
        Node::B3 => align_market(tr, mk, &|s: &Mk| {
            let (mut ba, mut bb) = (0.0, 0.0);
            for i in 0..10 {
                ba += s.ask_vols[i];
                bb += s.bid_vols[i];
            }
            let ra = if s.total_ask_vol > 0.0 { ba / s.total_ask_vol } else { 0.0 };
            let rb = if s.total_bid_vol > 0.0 { bb / s.total_bid_vol } else { 0.0 };
            (((ra + rb) / 2.0) * 10.0).clamp(0.0, 9.99) as i64
        }),
        Node::O1 => {
            let v: Vec<f64> = tr.iter().map(|t| (t.bid_order + t.ask_order) as f64).collect();
            quantile_buckets(&v, 20)
        }
        Node::R1 => quantile_buckets(&ctx.ret20, 20),
        Node::R2 => quantile_buckets(&ctx.rvol20, 20),
        Node::C1 => tr.iter().map(|t| {
            let dir = if t.flag == 66 { 0i64 } else { 1 };
            let vb: i64 = if t.volume < 500.0 { 0 } else if t.volume < 3000.0 { 1 } else { 2 };
            let d = (t.ask_order + t.bid_order) as f64;
            let u = if d > 0.0 { ((t.ask_order - t.bid_order) as f64 / d + 1.0) * 2.5 } else { 2.5 };
            dir * 30 + vb * 10 + (u.clamp(0.0, 9.99) as i64)
        }).collect(),
        Node::C2 => tr.iter().map(|t| {
            let cents = ((t.price * 10.0).round() as i64) % 100;
            let sec = (t.time_us / 1_000_000) % 60;
            cents * 60 + sec
        }).collect(),
        Node::C3 => {
            let mut out = vec![0i64; n];
            for i in 1..n {
                let dt = tr[i].time_us - tr[i - 1].time_us;
                let r: i64 = if dt < 100_000 { 0 } else if dt < 1_000_000 { 1 } else { 2 };
                let d = if tr[i].flag == 66 { 0i64 } else { 1 };
                out[i] = d * 3 + r;
            }
            out
        }
    }
}

// ============================================================
// 单股 × 单节点 × 单窗口：固定 137（价格类 143）个量，顺序对齐 dijkstra_names
// ============================================================

fn node_values(out: &mut Vec<f64>, node: Node, tr: &[Tr], win: &[usize], nw: usize, bucket: &[i64]) {
    let start = out.len();
    let need = PER_NODE + if is_price_node(node) { PER_NODE_PRICE } else { 0 };
    let ntr = tr.len();
    macro_rules! pad {
        () => {{
            while out.len() - start < need { out.push(f64::NAN); }
            return;
        }};
    }
    if ntr < 50 { pad!(); }

    let mut map: BTreeMap<i64, usize> = BTreeMap::new();
    for &b in bucket {
        if b >= 0 {
            let l = map.len();
            map.entry(b).or_insert(l);
        }
    }
    let nb = map.len();
    if nb < 2 { pad!(); }
    let bid: Vec<i64> = bucket.iter().map(|&b| *map.get(&b).unwrap_or(&usize::MAX) as i64).collect();

    let mut pop: Vec<Vec<f64>> = vec![vec![0.0; nb]; nw];
    let mut layer_sum = vec![0.0; nw];
    let mut layer_cnt = vec![0.0; nw];
    let mut day_cnt = vec![0.0; nb];
    for i in 0..ntr {
        let b = bid[i];
        if b < 0 || b as usize >= nb { continue; }
        let w = win[i];
        pop[w][b as usize] += 1.0;
        layer_sum[w] += b as f64;
        layer_cnt[w] += 1.0;
        day_cnt[b as usize] += 1.0;
    }
    for w in 0..nw {
        if layer_cnt[w] > 0.0 { layer_sum[w] /= layer_cnt[w]; }
    }

    let ordinal = is_ordinal(node);

    // ---- 逐层量 ----
    let wid: Vec<f64> = pop.iter().map(|p| p.iter().filter(|&&c| c > 0.0).count() as f64).collect();
    let poptot: Vec<f64> = pop.iter().map(|p| p.iter().sum()).collect();
    let top3: Vec<f64> = pop.iter().map(|p| {
        let tot: f64 = p.iter().sum();
        if tot <= 0.0 { return 0.0; }
        let mut s = p.clone();
        s.sort_by(|a, b| b.total_cmp(a));
        s.iter().take(3).sum::<f64>() / tot
    }).collect();
    let hent: Vec<f64> = pop.iter().map(|p| entropy(p)).collect();
    let peak_ratio: Vec<f64> = pop.iter().map(|p| {
        let tot: f64 = p.iter().sum();
        if tot <= 0.0 { return 0.0; }
        let mx = p.iter().cloned().fold(0.0, f64::max);
        let k = p.iter().filter(|&&c| c > 0.0).count().max(1) as f64;
        let av = tot / k;
        if av > 0.0 { mx / av } else { 0.0 }
    }).collect();

    push_series(out, &wid);
    push_series(out, &poptot);
    push_series(out, &hent);
    push_series(out, &top3);
    push_series(out, &peak_ratio);

    // ---- 层间迁移 ----
    let mut ret = vec![0.0; nw - 1];
    for k in 0..nw - 1 {
        let keep: f64 = pop[k].iter().zip(&pop[k + 1]).map(|(&a, &b)| a.min(b)).sum();
        let tot: f64 = pop[k].iter().sum();
        ret[k] = if tot > 0.0 { keep / tot } else { 0.0 };
    }
    push_series(out, &ret);
    let churn: Vec<f64> = ret.iter().map(|r| 1.0 - r).collect();
    out.push(churn.iter().cloned().fold(f64::INFINITY, f64::min));
    let churn_sum: f64 = churn.iter().sum();
    out.push(if churn_sum > 1e-9 {
        churn.iter().enumerate().map(|(i, &c)| i as f64 * c).sum::<f64>() / churn_sum
    } else { f64::NAN });

    // ---- 时间扩展图最短路（最优 + 次优）----
    let first_nonempty = (0..nw).find(|&k| pop[k].iter().any(|&c| c > 0.0));
    let mut dist: Vec<HashMap<usize, f64>> = vec![HashMap::new(); nw];
    let mut prev: Vec<HashMap<usize, (usize, usize)>> = vec![HashMap::new(); nw];
    let mut second_dist: Vec<HashMap<usize, f64>> = vec![HashMap::new(); nw];
    let mut first_k = usize::MAX;
    if let Some(fk) = first_nonempty {
        for (b, &c) in pop[fk].iter().enumerate() {
            if c > 0.0 { dist[fk].insert(b, BETA / c); }
        }
        first_k = fk;
        let mut last = fk;
        for k in (fk + 1)..nw {
            if !pop[k].iter().any(|&c| c > 0.0) { continue; }
            for (y, &cy) in pop[k].iter().enumerate() {
                if cy <= 0.0 { continue; }
                let (mut b1, mut b2) = (f64::INFINITY, f64::INFINITY);
                let mut bx = usize::MAX;
                for (&x, &dx) in &dist[last] {
                    let dd = if ordinal {
                        (x as f64 - y as f64).abs() / (nb - 1).max(1) as f64
                    } else if x == y { 0.0 } else { 1.0 };
                    let v = dx + ALPHA * dd + BETA / pop[last][x].min(cy);
                    if v < b1 { b2 = b1; b1 = v; bx = x; }
                    else if v < b2 { b2 = v; }
                }
                if b1.is_finite() {
                    dist[k].insert(y, b1);
                    prev[k].insert(y, (last, bx));
                }
                if b2.is_finite() { second_dist[k].insert(y, b2); }
            }
            last = k;
        }
    }
    let mut opt_cost = f64::NAN;
    let mut traj: Vec<(usize, usize)> = Vec::new();
    if first_k != usize::MAX {
        if let Some((&ye, &de)) = dist[nw - 1].iter()
            .min_by(|a, b| f64::total_cmp(a.1, b.1)) {
            opt_cost = de;
            let (mut ck, mut cy) = (nw - 1, ye);
            loop {
                traj.push((ck, cy));
                if ck == first_k { break; }
                let &(pj, px) = &prev[ck][&cy];
                ck = pj;
                cy = px;
            }
            traj.reverse();
        }
    }
    let mut opt2 = f64::INFINITY;
    if !traj.is_empty() {
        let best_end = traj.last().unwrap().1;
        for (&y, &d) in &dist[nw - 1] {
            if y != best_end { opt2 = opt2.min(d); }
            if let Some(&s2) = second_dist[nw - 1].get(&y) { opt2 = opt2.min(s2); }
        }
    }

    let layer_cost = |nodes: &[usize]| -> f64 {
        let mut c = 0.0;
        for k in 0..nw - 1 {
            let (x, y) = (nodes[k], nodes[k + 1]);
            let dd = if ordinal {
                (x as f64 - y as f64).abs() / (nb - 1).max(1) as f64
            } else if x == y { 0.0 } else { 1.0 };
            c += ALPHA * dd + BETA / pop[k][x].min(pop[k + 1][y]);
        }
        c
    };
    let hot: Vec<usize> = pop.iter().map(|p|
        (0..nb).filter(|&b| p[b] > 0.0)
            .max_by(|&a, &b| p[a].total_cmp(&p[b]))
            .unwrap_or(0)).collect();
    let cold: Vec<usize> = pop.iter().map(|p|
        (0..nb).filter(|&b| p[b] > 0.0)
            .min_by(|&a, &b| p[a].total_cmp(&p[b]))
            .unwrap_or(0)).collect();
    let cost_hot = layer_cost(&hot);
    let cost_cold = layer_cost(&cold);
    let mut rng: u64 = 0x9e3779b9;
    let (mut rand_sum, mut rand_n) = (0.0, 0);
    for _ in 0..48 {
        let nodes: Vec<usize> = pop.iter().map(|p| {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let cand: Vec<usize> = (0..nb).filter(|&b| p[b] > 0.0).collect();
            if cand.is_empty() { usize::MAX } else { cand[(rng >> 33) as usize % cand.len()] }
        }).collect();
        if nodes.iter().all(|&x| x != usize::MAX) {
            rand_sum += layer_cost(&nodes);
            rand_n += 1;
        }
    }
    let cost_rand = if rand_n > 0 { rand_sum / rand_n as f64 } else { f64::NAN };

    // ---- 路径行为（名字顺序：path 11 个 + step 16 个，位于 adv 之前）----
    push_path(out, &traj, opt_cost, nw, nb, &pop, &layer_sum, ordinal);

    // ---- 图自由度 / 路径唯一性 ----
    out.push(if opt_cost > 1e-12 { cost_rand / opt_cost } else { f64::NAN });
    out.push(if cost_hot > 1e-12 { opt_cost / cost_hot } else { f64::NAN });
    out.push(if cost_cold > 1e-12 { opt_cost / cost_cold } else { f64::NAN });
    out.push(if opt_cost > 1e-12 && opt2.is_finite() { opt2 / opt_cost - 1.0 } else { f64::NAN });
    out.push(if opt2.is_finite() { opt2 - opt_cost } else { f64::NAN });

    // ---- 图形状 + 方向性 ----
    let mut trans: HashMap<(usize, usize), u64> = HashMap::new();
    let mut dir_net: HashMap<(usize, usize), u64> = HashMap::new();
    for i in 1..ntr {
        let (x, y) = (bid[i - 1], bid[i]);
        if x >= 0 && y >= 0 && (x as usize) < nb && (y as usize) < nb && x != y {
            let (a, b) = (x.min(y) as usize, x.max(y) as usize);
            *trans.entry((a, b)).or_insert(0) += 1;
            *dir_net.entry((x as usize, y as usize)).or_insert(0) += 1;
        }
    }
    let poss = nb * (nb - 1) / 2;
    out.push(trans.len() as f64 / poss.max(1) as f64);
    out.push(wid.iter().cloned().fold(f64::INFINITY, f64::min) / mean(&wid).max(1e-9));
    let hot0 = day_cnt.iter().enumerate()
        .max_by(|a, b| f64::total_cmp(a.1, b.1))
        .map(|(i, _)| i).unwrap_or(0);
    let (far, reach) = transition_cost(&trans, nb, hot0);
    out.push(far);
    out.push(reach);
    let mut deg = vec![0f64; nb];
    for (&(x, y), &c) in &trans {
        deg[x] += c as f64;
        deg[y] += c as f64;
    }
    let deg_tot: f64 = deg.iter().sum();
    if deg_tot > 0.0 {
        out.push(deg.iter().map(|&d| (d / deg_tot).powi(2)).sum::<f64>());
        out.push(deg.iter().cloned().fold(0.0, f64::max) / deg_tot);
    } else {
        out.push(f64::NAN);
        out.push(f64::NAN);
    }
    let net: f64 = dir_net.iter()
        .map(|(&(x, y), &c)| {
            let rev = *dir_net.get(&(y, x)).unwrap_or(&0);
            c as f64 - rev as f64
        })
        .sum::<f64>().abs();
    let all: f64 = dir_net.values().map(|&v| v as f64).sum();
    out.push(if all > 0.0 { net / all } else { f64::NAN });

    // ---- 价格类专属：探路深度与重心偏离 ----
    if is_price_node(node) {
        push_price(out, node, tr, win, nw, &traj, &map);
    }
    pad!();
}

fn push_path(out: &mut Vec<f64>, traj: &[(usize, usize)], opt_cost: f64, nw: usize, nb: usize,
             pop: &[Vec<f64>], layer_sum: &[f64], ordinal: bool) {
    let start = out.len();
    let need = PATH_FIELDS.len() + SERIES_OPS.len();
    out.push(opt_cost / nw as f64);
    if traj.len() < 2 {
        while out.len() - start < need { out.push(f64::NAN); }
        return;
    }
    let (mut gc, mut tc) = (0.0, 0.0);
    for w in 0..traj.len() - 1 {
        let (kx, x) = traj[w];
        let (ky, y) = traj[w + 1];
        let dd = if ordinal {
            (x as f64 - y as f64).abs() / (nb - 1).max(1) as f64
        } else if x == y { 0.0 } else { 1.0 };
        gc += ALPHA * dd;
        tc += ALPHA * dd + BETA / pop[kx][x].min(pop[ky][y]);
    }
    out.push(if tc > 1e-12 { gc / tc } else { 0.0 });
    let stay = traj.windows(2).filter(|w| w[0].1 == w[1].1).count() as f64;
    out.push(stay / (traj.len() - 1) as f64);
    let tv: Vec<f64> = traj.iter().map(|&(_, b)| b as f64).collect();
    let mv: Vec<f64> = traj.iter().map(|&(k, _)| layer_sum[k]).collect();
    out.push(corr(&tv, &mv));
    let disp: f64 = tv.windows(2).map(|w| (w[1] - w[0]).abs()).sum();
    let amp = tv.iter().cloned().fold(0.0, f64::max) - tv.iter().cloned().fold(f64::INFINITY, f64::min);
    let scale = (nb - 1).max(1) as f64;
    out.push(disp / scale);
    out.push(amp / scale);
    out.push(if amp > 1e-9 { disp / amp } else { f64::NAN });
    let num: f64 = traj.windows(2)
        .map(|w| (w[1].1 as f64 - w[0].1 as f64).abs() * (w[0].0 + w[1].0) as f64 / 2.0)
        .sum();
    out.push(num / disp.max(1e-9));
    let (lk, lb) = *traj.last().unwrap();
    out.push((lb as f64 - layer_sum[lk]).abs() / scale);
    let devs: Vec<f64> = traj.iter().map(|&(k, b)| (b as f64 - layer_sum[k]).abs()).collect();
    out.push(mean(&devs) / scale);
    out.push(stddev(&devs) / scale);
    let steps: Vec<f64> = traj.windows(2).map(|w| (w[1].1 as f64 - w[0].1 as f64).abs() / scale).collect();
    push_series(out, &steps);
}

fn push_price(out: &mut Vec<f64>, node: Node, tr: &[Tr], win: &[usize], nw: usize,
              traj: &[(usize, usize)], map: &BTreeMap<i64, usize>) {
    let need = PRICE_FIELDS.len();
    if traj.len() < 2 {
        out.extend(std::iter::repeat(f64::NAN).take(need));
        return;
    }
    let real: Vec<f64> = if node == Node::P1 {
        tr.iter().map(|t| t.price).collect()
    } else {
        let open = tr[0].price;
        tr.iter().map(|t| (t.price - open) / open).collect()
    };
    let mut layer_real: Vec<Vec<f64>> = vec![Vec::new(); nw];
    for i in 0..tr.len() { layer_real[win[i]].push(real[i]); }
    let lmax: Vec<f64> = layer_real.iter().map(|v| v.iter().cloned().fold(f64::NEG_INFINITY, f64::max)).collect();
    let lmin: Vec<f64> = layer_real.iter().map(|v| v.iter().cloned().fold(f64::INFINITY, f64::min)).collect();
    let day_max = lmax.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let day_min = lmin.iter().cloned().fold(f64::INFINITY, f64::min);
    let day_range = (day_max - day_min).max(1e-12);
    let traj_real: Vec<f64> = traj.iter().map(|&(_, b)| {
        let key = map.iter().find(|(_, &v)| v == b).map(|(k, _)| *k).unwrap_or(0);
        if node == Node::P1 { key as f64 / 100.0 } else { key as f64 }
    }).collect();
    let tmax = traj_real.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let tmin = traj_real.iter().cloned().fold(f64::INFINITY, f64::min);
    out.push((day_max - tmax) / day_range);
    out.push((tmin - day_min) / day_range);
    let lm: Vec<f64> = layer_real.iter().map(|v| mean(v)).collect();
    let lm_at: Vec<f64> = traj.iter().map(|&(k, _)| lm[k]).collect();
    let gaps: Vec<f64> = traj_real.iter().zip(&lm_at).map(|(&a, &b)| (a - b) / day_range).collect();
    out.push(mean(&gaps));
    out.push(stddev(&gaps));
    out.push(autocorr1(&gaps));
    out.push(corr(&traj_real, &lm_at));
}

// ============================================================
// 单股全量
// ============================================================

fn per_stock(raw: &[Tr], mk: &[Mk]) -> Option<Vec<f64>> {
    // 脏数据防御：价格必须为正且有限，量额必须非负；否则下游除零会产出 NaN/inf，
    // 进而让排序比较函数失去全序（Rust 会直接 panic）。
    let cleaned: Vec<Tr> = raw.iter()
        .filter(|t| t.price.is_finite() && t.price > 0.0
                 && t.volume.is_finite() && t.volume >= 0.0
                 && t.turnover.is_finite() && t.turnover >= 0.0)
        .cloned()
        .collect();
    let tr: &[Tr] = &cleaned;
    if tr.len() < 50 { return None; }
    let ctx = build_ctx(tr);
    let tmin = tr[0].time_us;
    let tmax = tr[tr.len() - 1].time_us;
    let span = (tmax - tmin).max(1);
    let mut out: Vec<f64> = Vec::with_capacity(N_FACTORS);
    for &nw in WINDOW_COUNTS.iter() {
        let win: Vec<usize> = tr.iter()
            .map(|t| (((t.time_us - tmin) as f64 / span as f64 * nw as f64) as usize).min(nw - 1))
            .collect();
        for &node in ALL_NODES.iter() {
            if needs_market(node) && mk.is_empty() {
                let need = PER_NODE + if is_price_node(node) { PER_NODE_PRICE } else { 0 };
                out.extend(std::iter::repeat(f64::NAN).take(need));
                continue;
            }
            let bucket = make_buckets(node, tr, mk, &ctx);
            node_values(&mut out, node, tr, &win, nw, &bucket);
        }
    }
    debug_assert_eq!(out.len(), N_FACTORS, "单股输出列数与 N_FACTORS 不一致");
    Some(out)
}

// ============================================================
// 横截面入口
// ============================================================

fn zscore_column(vals: &mut [f64]) {
    let finite: Vec<f64> = vals.iter().filter(|v| v.is_finite()).copied().collect();
    if finite.len() < 2 { return; }
    let mean = finite.iter().sum::<f64>() / finite.len() as f64;
    let var = finite.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / finite.len() as f64;
    let std = var.sqrt();
    if std < 1e-12 { return; }
    for v in vals.iter_mut() {
        if v.is_finite() { *v = (*v - mean) / std; }
    }
}

/// 核心：读一天全市场 → per-stock 建图求最短路 → 横截面 z-score → (codes, vals)。
pub fn compute_dijkstra_full(date: i64) -> std::io::Result<(Vec<String>, Vec<f32>)> {
    let codes = crate::data_paths::list_stock_codes(date, "transaction");
    if codes.is_empty() {
        return Err(std::io::Error::new(std::io::ErrorKind::NotFound,
                                       format!("{date} 无股票范围")));
    }

    let features: Vec<Option<Vec<f64>>> = codes
        .par_iter()
        .map(|code| {
            let raw_trades = read_trade_fast_inner(code, date, false, true, usize::MAX).ok()?;
            let raw_market = read_market_fast_inner(code, date, false, true, usize::MAX).unwrap_or_default();
            let trades: Vec<Tr> = raw_trades.iter().map(conv_trade).collect();
            let market: Vec<Mk> = raw_market.iter().map(conv_market).collect();
            per_stock(&trades, &market)
        })
        .collect();

    let mut valid_codes: Vec<String> = Vec::new();
    let mut matrix: Vec<Vec<f64>> = Vec::new();
    for (code, f) in codes.iter().zip(features.into_iter()) {
        if let Some(row) = f {
            valid_codes.push(code.clone());
            matrix.push(row);
        }
    }
    if matrix.is_empty() {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidData,
                                       format!("{date} 无有效股票")));
    }

    for j in 0..N_FACTORS {
        let mut col: Vec<f64> = matrix.iter().map(|r| r[j]).collect();
        zscore_column(&mut col);
        for (i, &v) in col.iter().enumerate() { matrix[i][j] = v; }
    }

    let mut vals: Vec<f32> = Vec::with_capacity(valid_codes.len() * N_FACTORS);
    for r in &matrix {
        vals.extend(r.iter().map(|&v| v as f32));
    }
    Ok((valid_codes, vals))
}

// ============================================================
// Python 入口
// ============================================================

/// Python 单日：返回 (codes, vals)，每股 N_FACTORS 个值。
#[pyfunction]
pub fn py_dijkstra(py: Python<'_>, date: i64) -> PyResult<(Vec<String>, Vec<f32>)> {
    compute_dijkstra_full(date)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("{e:?}")))
}

/// Python 拿因子名（与 N_FACTORS 严格对齐）。
#[pyfunction]
pub fn py_dijkstra_names() -> Vec<String> {
    dijkstra_names()
}
