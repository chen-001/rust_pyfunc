//! 逐笔跳跃DAG最短路因子族（dagpath）：序列×边算子×子序列×K×λ 全网格 + 路径交互。
//! 只含 O(n) 边：abs / sgn / hop（up/dn/msg/mix 四种 O(n*K) 边留 sandbox 版本）。
//! 单股单日输入 → 固定 OUT_LEN 列输出，缺数据的格子填 NaN。
use crate::fast_csv_reader::{read_trade_fast_inner, TradeRecord};
use pyo3::prelude::*;
use std::collections::{BTreeMap, BTreeSet, VecDeque};

// ---------- 网格定义 ----------
const SEQ_TAGS: [&str; 8] = ["prc", "vol", "amt", "urg", "sdir", "svol", "tim", "cvl"];
const EDGE_TAGS: [&str; 3] = ["abs", "sgn", "hop"];
const SUB_TAGS: [&str; 7] = ["all", "buy", "sel", "big", "sml", "am", "pm"];
const KS: [usize; 2] = [200, 1000];
const LAMS: [f64; 2] = [0.5, 2.0];

#[derive(Clone, Copy, PartialEq)]
enum EdgeKind { Abs, Sgn, Hop }

fn edge_of(et: usize) -> EdgeKind {
    match et { 0 => EdgeKind::Abs, 1 => EdgeKind::Sgn, _ => EdgeKind::Hop }
}

fn run_key(s: usize, et: usize, u: usize, k_req: usize, lam: f64) -> String {
    format!("{}-{}_sub{}_K{}_L{}",
        SEQ_TAGS[s], EDGE_TAGS[et], SUB_TAGS[u], k_req, (lam * 10.0) as i32)
}

// ---------- 工具 ----------
fn stddev(x: &[f64]) -> f64 {
    if x.len() < 2 { return 1.0; }
    let m = x.iter().sum::<f64>() / x.len() as f64;
    (x.iter().map(|v| (v - m).powi(2)).sum::<f64>() / x.len() as f64).sqrt().max(1e-12)
}
fn qtile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() { return 0.0; }
    sorted[((sorted.len() as f64 - 1.0) * q).round() as usize]
}
fn mean(x: &[f64]) -> f64 { if x.is_empty() { 0.0 } else { x.iter().sum::<f64>() / x.len() as f64 } }

// ---------- 单调队列 ----------
#[derive(Default)]
struct MonoDq { dq: VecDeque<(usize, f64)> }
impl MonoDq {
    fn push_max(&mut self, i: usize, v: f64) {
        while let Some(&(_, b)) = self.dq.back() { if b <= v { self.dq.pop_back(); } else { break; } }
        self.dq.push_back((i, v));
    }
    fn push_min(&mut self, i: usize, v: f64) {
        while let Some(&(_, b)) = self.dq.back() { if b >= v { self.dq.pop_back(); } else { break; } }
        self.dq.push_back((i, v));
    }
    fn evict(&mut self, lo: usize) {
        while let Some(&(i, _)) = self.dq.front() { if i < lo { self.dq.pop_front(); } else { break; } }
    }
    fn front(&self) -> Option<(usize, f64)> { self.dq.front().copied() }
}

// ---------- DP 内核 ----------
// ---------- 按 s_i 坐压 + 每坐标 BTreeMap<下标,值> + 每棵树存 (该点min值, 该下标) ----------
// 滑动窗口删除时只删对应下标的条目，同值其他下标不受影响。
use std::collections::BTreeMap as Map;

struct WinSeg {
    // 每个压缩坐标一个 multiset：下标→值；树里只放该坐标的当前最小 (值, 下标)
    pts: Vec<Map<usize, f64>>,
    n: usize,
    tree: Vec<(f64, usize)>,
}
impl WinSeg {
    fn new(m: usize) -> Self {
        let size = m.next_power_of_two().max(1);
        WinSeg {
            pts: vec![Map::new(); m],
            n: size,
            tree: vec![(f64::INFINITY, usize::MAX); size * 2],
        }
    }
    fn pull(&mut self, p: usize) {
        // 重新算叶子 p 的 (min, argmin) 并上推
        let leaf = self.pts[p].iter().next().map(|(&i, &v)| (v, i));
        let mut x = p + self.n;
        self.tree[x] = leaf.unwrap_or((f64::INFINITY, usize::MAX));
        x /= 2;
        while x >= 1 {
            self.tree[x] = if self.tree[2 * x].0 <= self.tree[2 * x + 1].0 { self.tree[2 * x] } else { self.tree[2 * x + 1] };
            if x == 1 { break; }
            x /= 2;
        }
    }
    fn insert(&mut self, p: usize, idx: usize, v: f64) {
        self.pts[p].insert(idx, v);
        self.pull(p);
    }
    fn remove(&mut self, p: usize, idx: usize) {
        if self.pts[p].remove(&idx).is_some() { self.pull(p); }
    }
    fn query(&self, mut l: usize, mut r: usize) -> (f64, usize) {
        let mut res = (f64::INFINITY, usize::MAX);
        l += self.n; r += self.n;
        while l <= r {
            if l % 2 == 1 {
                let v = self.tree[l];
                if v.0 < res.0 || (v.0 == res.0 && v.1 < res.1) { res = v; }
                l += 1;
            }
            if r % 2 == 0 {
                let v = self.tree[r];
                if v.0 < res.0 || (v.0 == res.0 && v.1 < res.1) { res = v; }
                if r == 0 { break; }
                r -= 1;
            }
            l /= 2; r /= 2;
        }
        res
    }
}

fn dp_abs(s: &[f64], k: usize, lam: f64) -> (Vec<f64>, Vec<i64>) {
    let n = s.len();
    let mut cost = vec![f64::INFINITY; n]; cost[0] = 0.0;
    let mut prev = vec![-1i64; n];
    let mut xs: Vec<f64> = s.to_vec();
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    xs.dedup();
    let m = xs.len();
    let pos = |v: f64| -> usize { xs.partition_point(|&x| x < v) };
    let mut t1 = WinSeg::new(m); // s_i <= s_j 查 min(cost_i - s_i)
    let mut t2 = WinSeg::new(m); // s_i >= s_j 查 min(cost_i + s_i)
    for j in 0..n {
        let lo = if j > k { j - k } else { 0 };
        if j > k {
            let out = j - k - 1;
            if cost[out].is_finite() {
                let p = pos(s[out]);
                t1.remove(p, out);
                t2.remove(p, out);
            }
        }
        if lo < j {
            let pj = pos(s[j]).min(m - 1);
            let (v1, i1) = t1.query(0, pj);
            let (v2, i2) = t2.query(pj, m - 1);
            let mut best = f64::INFINITY; let mut bi = -1i64;
            if v1.is_finite() { let c = s[j] + v1; if c < best { best = c; bi = i1 as i64; } }
            if v2.is_finite() { let c = -s[j] + v2; if c < best { best = c; bi = i2 as i64; } }
            if best.is_finite() { cost[j] = best + lam; prev[j] = bi; }
        }
        if cost[j].is_finite() {
            let p = pos(s[j]);
            t1.insert(p, j, cost[j] - s[j]);
            t2.insert(p, j, cost[j] + s[j]);
        }
    }
    (cost, prev)
}

fn dp_sgn(s: &[f64], k: usize, lam: f64) -> (Vec<f64>, Vec<i64>) {
    let n = s.len();
    let mut cost = vec![f64::INFINITY; n]; cost[0] = 0.0;
    let mut prev = vec![-1i64; n];
    let mut dm = MonoDq::default();
    for j in 0..n {
        let lo = if j > k { j - k } else { 0 };
        dm.evict(lo);
        if lo < j {
            if let Some((i, cm)) = dm.front() {
                cost[j] = s[j] + cm + lam; prev[j] = i as i64;
            }
        }
        dm.push_min(j, cost[j] - s[j]);
    }
    (cost, prev)
}

fn dp_hop(s: &[f64], k: usize) -> (Vec<f64>, Vec<i64>) {
    let n = s.len();
    let mut cost = vec![f64::INFINITY; n]; cost[0] = 0.0;
    let mut prev = vec![-1i64; n];
    let mut dm = MonoDq::default();
    for j in 0..n {
        let lo = if j > k { j - k } else { 0 };
        dm.evict(lo);
        if lo < j {
            if let Some((i, cm)) = dm.front() {
                cost[j] = cm + 1.0; prev[j] = i as i64;
            }
        }
        dm.push_min(j, cost[j]);
    }
    (cost, prev)
}

// ---------- 输出统计 ----------
fn run_stats(key: &str, s: &[f64], idx: &[usize], tr: &[TradeRecord],
             cost: &[f64], prev: &[i64], urg_all: &[f64],
             pvol: &[f64], pamt: &[f64], purg: &[f64],
             out: &mut BTreeMap<String, f64>) -> Vec<usize> {
    let n = s.len();
    let nan = f64::NAN;
    let mut bb: Vec<usize> = vec![0];
    let mut j = n - 1;
    while j != 0 && prev[j] >= 0 { bb.push(j); j = prev[j] as usize; }
    bb.reverse();
    let mut mask = vec![false; n];
    for &b in &bb { mask[b] = true; }

    let c = cost[n - 1];
    let real: f64 = (1..n).map(|i| (s[i] - s[i - 1]).abs()).sum();
    out.insert(format!("{}_cost", key), c);
    out.insert(format!("{}_save", key), if real > 1e-12 { 1.0 - c / real } else { nan });
    out.insert(format!("{}_hops", key), (bb.len() - 1) as f64);
    out.insert(format!("{}_bbn", key), bb.len() as f64 / n as f64);
    out.insert(format!("{}_netd", key), s[n - 1] - s[0]);

    // 分段统计：前缀和 O(骨架长) 版本
    let mut span = Vec::new(); let mut d = Vec::new();
    let mut vol = Vec::new(); let mut urg = Vec::new(); let mut amt = Vec::new();
    for w in 1..bb.len() {
        let (a, b) = (bb[w - 1], bb[w]);
        let sp = (b - a) as f64;
        span.push(sp);
        d.push(s[b] - s[a]);
        vol.push(pvol[b + 1] - pvol[a + 1]);
        amt.push(pamt[b + 1] - pamt[a + 1]);
        urg.push((purg[b + 1] - purg[a + 1]) / sp);
    }
    let up: Vec<usize> = (0..d.len()).filter(|&i| d[i] > 0.0).collect();
    let dn: Vec<usize> = (0..d.len()).filter(|&i| d[i] < 0.0).collect();
    let pick = |v: &Vec<f64>, ids: &Vec<usize>| ids.iter().map(|&i| v[i]).collect::<Vec<f64>>();
    let (du, dd) = (pick(&d, &up), pick(&d, &dn));
    let (su, sd) = (pick(&span, &up), pick(&span, &dn));
    let (vu, vd) = (pick(&vol, &up), pick(&vol, &dn));
    let (uu, ud) = (pick(&urg, &up), pick(&urg, &dn));
    let (au, ad) = (pick(&amt, &up), pick(&amt, &dn));

    out.insert(format!("{}_span", key), mean(&span));
    out.insert(format!("{}_smax", key), span.iter().cloned().fold(0.0, f64::max));
    out.insert(format!("{}_uN", key), up.len() as f64);
    out.insert(format!("{}_dN", key), dn.len() as f64);
    out.insert(format!("{}_udN", key), up.len() as f64 - dn.len() as f64);
    out.insert(format!("{}_uDisp", key), du.iter().sum::<f64>());
    out.insert(format!("{}_dDisp", key), dd.iter().sum::<f64>());
    out.insert(format!("{}_uDispM", key), mean(&du));
    out.insert(format!("{}_dDispM", key), mean(&dd));
    out.insert(format!("{}_uSpanM", key), mean(&su));
    out.insert(format!("{}_dSpanM", key), mean(&sd));
    out.insert(format!("{}_spanD", key), mean(&su) - mean(&sd));
    out.insert(format!("{}_uVolM", key), mean(&vu));
    out.insert(format!("{}_dVolM", key), mean(&vd));
    out.insert(format!("{}_volD", key), mean(&vu) - mean(&vd));
    out.insert(format!("{}_uAmtM", key), mean(&au));
    out.insert(format!("{}_dAmtM", key), mean(&ad));
    out.insert(format!("{}_uUrgM", key), mean(&uu));
    out.insert(format!("{}_dUrgM", key), mean(&ud));
    out.insert(format!("{}_urgD", key), mean(&uu) - mean(&ud));

    // 被跳过笔的统计（前缀和）
    let tv = pvol[n] - pvol[0];
    let ta = pamt[n] - pamt[0];
    let mut sv = 0.0; let mut sa = 0.0; let mut sflag = 0.0; let mut sc = 0u64;
    for t in 0..n {
        if !mask[t] {
            sv += tr[idx[t]].volume as f64; sa += tr[idx[t]].turnover as f64; sc += 1;
            if tr[idx[t]].flag == 66 { sflag += 1.0; }
        }
    }
    out.insert(format!("{}_svol", key), if tv > 0.0 { sv / tv } else { nan });
    out.insert(format!("{}_samt", key), if ta > 0.0 { sa / ta } else { nan });
    out.insert(format!("{}_sbuy", key), if sc > 0 { sflag / sc as f64 } else { nan });
    bb.iter().map(|&b| idx[b]).collect()
}

// ---------- 子序列 ----------
fn sub_idx(tr: &[TradeRecord], sub: usize) -> Vec<usize> {
    let n = tr.len();
    match sub {
        1 => (0..n).filter(|&i| tr[i].flag == 66).collect(),
        2 => (0..n).filter(|&i| tr[i].flag == 83).collect(),
        3 | 4 => {
            let mut v: Vec<f64> = tr.iter().map(|t| t.volume as f64).collect();
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let (th, tl) = (qtile(&v, 0.9), qtile(&v, 0.5));
            if sub == 3 { (0..n).filter(|&i| tr[i].volume as f64 >= th).collect() }
            else { (0..n).filter(|&i| tr[i].volume as f64 <= tl).collect() }
        }
        5 | 6 => {
            let mid = (tr[0].time_us + tr[n - 1].time_us) / 2;
            if sub == 5 { (0..n).filter(|&i| tr[i].time_us < mid).collect() }
            else { (0..n).filter(|&i| tr[i].time_us >= mid).collect() }
        }
        _ => (0..n).collect(),
    }
}

fn iou(a: &[usize], b: &[usize]) -> f64 {
    let (mut i, mut j, mut inter) = (0, 0, 0usize);
    while i < a.len() && j < b.len() {
        if a[i] == b[j] { inter += 1; i += 1; j += 1; }
        else if a[i] < b[j] { i += 1; } else { j += 1; }
    }
    let uni = a.len() + b.len() - inter;
    if uni > 0 { inter as f64 / uni as f64 } else { f64::NAN }
}

// ---------- 单股计算 → BTreeMap ----------
fn per_stock(tr: &[TradeRecord]) -> BTreeMap<String, f64> {
    let mut out = BTreeMap::new();
    let n = tr.len();
    if n < 200 { return out; }

    let urg_all: Vec<f64> = tr.iter().map(|t| (t.ask_order - t.bid_order) as f64).collect();
    let lp: Vec<f64> = tr.iter().map(|t| t.price as f64 / 0.01).collect();
    let lvol: Vec<f64> = tr.iter().map(|t| (t.volume as f64 + 1.0).ln()).collect();
    let lamt: Vec<f64> = tr.iter().map(|t| (t.turnover as f64 + 1.0).ln()).collect();
    let sdir: Vec<f64> = tr.iter().map(|t| if t.flag == 66 { 1.0 } else { -1.0 }).collect();
    let svol: Vec<f64> = tr.iter().enumerate()
        .map(|(i, t)| (t.volume as f64 + 1.0).ln() * sdir[i]).collect();
    let tim: Vec<f64> = tr.iter().map(|t| (t.time_us - tr[0].time_us) as f64 / 1e6).collect();
    let cvl: Vec<f64> = {
        let tot: f64 = tr.iter().map(|t| t.volume as f64).sum();
        let mut acc = 0.0;
        tr.iter().map(|t| { acc += t.volume as f64; acc / tot.max(1.0) }).collect()
    };
    let norm = |x: &[f64]| -> Vec<f64> {
        let d: Vec<f64> = (1..x.len()).map(|i| x[i] - x[i - 1]).collect();
        let sd = stddev(&d);
        x.iter().map(|v| v / sd).collect()
    };
    let seqs: Vec<Vec<f64>> = vec![
        lp.clone(), norm(&lvol), norm(&lamt), norm(&urg_all),
        sdir.clone(), norm(&svol), norm(&tim), norm(&cvl),
    ];

    // 前缀和（跨 idx 映射到原 tr 下标）
    let mut pvol = vec![0.0f64; n + 1]; let mut pamt = vec![0.0f64; n + 1];
    let mut purg = vec![0.0f64; n + 1];
    for i in 0..n {
        pvol[i + 1] = pvol[i] + tr[i].volume as f64;
        pamt[i + 1] = pamt[i] + tr[i].turnover as f64;
        purg[i + 1] = purg[i] + urg_all[i];
    }

    let subs: Vec<Vec<usize>> = (0..7).map(|u| sub_idx(tr, u)).collect();
    let mut results: BTreeMap<String, (Vec<usize>, f64)> = BTreeMap::new();

    for s in 0..SEQ_TAGS.len() {
        for u in 0..SUB_TAGS.len() {
            let idx = &subs[u];
            if idx.len() < 60 { continue; }
            // 子序列上的前缀和（leg 统计用）
            let sn = idx.len();
            let mut pvol_s = vec![0.0f64; sn + 1]; let mut pamt_s = vec![0.0f64; sn + 1];
            let mut purg_s = vec![0.0f64; sn + 1];
            for (t, &i) in idx.iter().enumerate() {
                pvol_s[t + 1] = pvol_s[t] + tr[i].volume as f64;
                pamt_s[t + 1] = pamt_s[t] + tr[i].turnover as f64;
                purg_s[t + 1] = purg_s[t] + urg_all[i];
            }
            let seq: Vec<f64> = idx.iter().map(|&i| seqs[s][i]).collect();
            for &kw in &KS {
                let k = kw.min(idx.len() - 1);
                for &lam in &LAMS {
                    for et in 0..EDGE_TAGS.len() {
                        let e = edge_of(et);
                        if e == EdgeKind::Hop && lam != LAMS[0] { continue; }
                        let key = run_key(s, et, u, kw, lam);
                        let (cost, prev) = match e {
                            EdgeKind::Abs => dp_abs(&seq, k, lam),
                            EdgeKind::Sgn => dp_sgn(&seq, k, lam),
                            EdgeKind::Hop => dp_hop(&seq, k),
                        };
                        if !cost[idx.len() - 1].is_finite() { continue; }
                        let bb = run_stats(&key, &seq, idx, tr, &cost, &prev, &urg_all,
                                           &pvol_s, &pamt_s, &purg_s, &mut out);
                        results.insert(key, (bb, cost[idx.len() - 1]));
                    }
                }
            }
        }
    }

    // ---- 交互层 ----
    // I1: 同(sub,K,λ)下，prc-abs 骨架 vs 其他 (seq,edge) 骨架的重合度与成本差
    for u in 0..SUB_TAGS.len() {
        for &kw in &KS {
            for &lam in &LAMS {
                let base = run_key(0, 0, u, kw, lam);
                let Some((m0, c0)) = results.get(&base) else { continue; };
                for s in 0..SEQ_TAGS.len() {
                    for et in 0..EDGE_TAGS.len() {
                        if s == 0 && et == 0 { continue; }
                        let kk = run_key(s, et, u, kw, lam);
                        let Some((m1, c1)) = results.get(&kk) else { continue; };
                        out.insert(format!("int_{}_x_{}", base, kk), iou(m0, m1));
                        out.insert(format!("intC_{}_x_{}", base, kk), c1 - c0);
                    }
                }
            }
        }
    }
    // I2: 同(seq,edge,K,λ)下，互补子序列对骨架重合与成本差
    for s in 0..SEQ_TAGS.len() {
        for et in 0..EDGE_TAGS.len() {
            for &kw in &KS {
                for &lam in &LAMS {
                    for &(a, b) in &[(1usize, 2usize), (5, 6), (0, 1), (0, 2)] {
                        let ka = run_key(s, et, a, kw, lam);
                        let kb = run_key(s, et, b, kw, lam);
                        let (Some((ma, ca)), Some((mb, cb))) =
                            (results.get(&ka), results.get(&kb)) else { continue; };
                        let tag = format!("{}-{}_x_{}-{}_K{}_L{}",
                            SEQ_TAGS[s], EDGE_TAGS[et], SUB_TAGS[a], SUB_TAGS[b], kw,
                            (lam * 10.0) as i32);
                        out.insert(format!("intS_{}", tag), iou(ma, mb));
                        out.insert(format!("intCS_{}", tag), cb - ca);
                    }
                }
            }
        }
    }
    let _ = (pvol, pamt, purg);
    out
}

// ---------- 全量因子名（确定性生成，与 BTreeMap 排序一致）----------
fn all_base_stat_names() -> Vec<String> {
    vec![
        "cost", "save", "hops", "bbn", "netd", "span", "smax",
        "uN", "dN", "udN", "uDisp", "dDisp",
        "uDispM", "dDispM", "uSpanM", "dSpanM", "spanD",
        "uVolM", "dVolM", "volD",
        "uAmtM", "dAmtM", "uUrgM", "dUrgM", "urgD",
        "svol", "samt", "sbuy",
    ].iter().map(|s| s.to_string()).collect()
}

/// 生成全部可能的因子名（含交互名全集，与实际 BTreeMap.keys() 并集一致）
pub fn dagpath_names() -> Vec<String> {
    let mut names: BTreeSet<String> = BTreeSet::new();
    let stats = all_base_stat_names();
    // 基础统计名：所有 (seq, edge, sub, K, λ) 组合；hop 只算 LAMS[0]
    for s in 0..SEQ_TAGS.len() {
        for u in 0..SUB_TAGS.len() {
            for &kw in &KS {
                for &lam in &LAMS {
                    for et in 0..EDGE_TAGS.len() {
                        if edge_of(et) == EdgeKind::Hop && lam != LAMS[0] { continue; }
                        let key = run_key(s, et, u, kw, lam);
                        for st in &stats { names.insert(format!("{}_{}", key, st)); }
                    }
                }
            }
        }
    }
    // I1 交互名全集：u × K × λ × (s',et')（除去 s'=0&et'=0）；hop 只存在于 λ=LAMS[0]
    for u in 0..SUB_TAGS.len() {
        for &kw in &KS {
            for &lam in &LAMS {
                let base = run_key(0, 0, u, kw, lam);
                for s in 0..SEQ_TAGS.len() {
                    for et in 0..EDGE_TAGS.len() {
                        if s == 0 && et == 0 { continue; }
                        if edge_of(et) == EdgeKind::Hop && lam != LAMS[0] { continue; }
                        let kk = run_key(s, et, u, kw, lam);
                        names.insert(format!("int_{}_x_{}", base, kk));
                        names.insert(format!("intC_{}_x_{}", base, kk));
                    }
                }
            }
        }
    }
    // I2 交互名全集；hop 只存在于 λ=LAMS[0]
    for s in 0..SEQ_TAGS.len() {
        for et in 0..EDGE_TAGS.len() {
            for &kw in &KS {
                for &lam in &LAMS {
                    if edge_of(et) == EdgeKind::Hop && lam != LAMS[0] { continue; }
                    for &(a, b) in &[(1usize, 2usize), (5, 6), (0, 1), (0, 2)] {
                        let tag = format!("{}-{}_x_{}-{}_K{}_L{}",
                            SEQ_TAGS[s], EDGE_TAGS[et], SUB_TAGS[a], SUB_TAGS[b], kw,
                            (lam * 10.0) as i32);
                        names.insert(format!("intS_{}", tag));
                        names.insert(format!("intCS_{}", tag));
                    }
                }
            }
        }
    }
    names.into_iter().collect()
}

pub const OUT_LEN: usize = 17384;

/// 核心唯一真相源：单股单日 → 固定 OUT_LEN 列
pub fn compute_dagpath_full(code: &str, date: i64) -> std::io::Result<Vec<f32>> {
    let tr = read_trade_fast_inner(code, date, false, true, usize::MAX)?;
    let feats = per_stock(&tr);
    let names = dagpath_names();
    let mut out = vec![f32::NAN; OUT_LEN];
    for (i, n) in names.iter().enumerate() {
        if let Some(&v) = feats.get(n) { out[i] = v as f32; }
    }
    Ok(out)
}

#[pyfunction]
pub fn py_dagpath(_py: Python<'_>, code: &str, date: i64) -> PyResult<Vec<f32>> {
    compute_dagpath_full(code, date)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("{:?}", e)))
}

#[pyfunction]
pub fn py_dagpath_names() -> Vec<String> { dagpath_names() }
