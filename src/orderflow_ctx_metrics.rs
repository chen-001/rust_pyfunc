//! orderflow_ctx：三时段 × (逐笔流 + 订单级流 + 盘口流) × 上下文预测算子工厂。
//!
//! 从 sandbox_orderflow_ctx 移植为正式库横截面因子。每股每时段把逐笔、盘口、
//! 订单聚合（按 bid_order 分组的 fill 事件）各自的符号/节奏/结构二值流，
//! 统一刷同一套"上下文预测"算子（acc_k/词典熵/motif/游程/漂移），再对每股每个
//! 因子做全市场 z-score。
//!
//! # 分层
//! - 时段层：f_ 全天 / a_ 下午（调整后 11:30 起）/ t_ 尾盘 30 分钟（12:57 起）
//! - 流层：逐笔 34 条 / 订单 12 条 / 盘口 19 条
//! - 算子层：逐笔流 62 个（ops）、订单/盘口流 30 个（ops_lite）
//! - 散装：n_tr / xt_imb_mi / xt_imb_lead / to_dec_dom（每时段各一）+ n_trades
//!
//! # 与 sandbox 的对应
//! - 因子名 = "{scope}_{stream}_{op}"，scope ∈ {f_, a_, t_}
//! - 输出列数 = 1 + 3*(1 + 34*62 + 12*30 + 19*30 + 3) = 10030
//! - sandbox 把缺时段的列整列不写；正式库改为整行对齐 10030，缺的时段补 NaN，
//!   股票只要全天段满 2000 笔就保留（不因为尾盘不足 500 笔而丢行）。

use crate::fast_csv_reader::{read_market_fast_inner, read_trade_fast_inner, MarketRecord, TradeRecord};
use pyo3::prelude::*;
use rayon::prelude::*;
use std::collections::HashMap;

// ---------- 常量布局（columns 固定为 10030） ----------

const N_TRADE_STREAMS: usize = 34;
const N_ORDER_STREAMS: usize = 12;
const N_MKT_STREAMS: usize = 19;
const OPS_FULL: usize = 59;
const OPS_LITE: usize = 43;
const SCATTER_PER_SCOPE: usize = 4; // n_tr, xt_imb_mi, xt_imb_lead, to_dec_dom
const SCOPE_COLS: usize = SCATTER_PER_SCOPE
    + N_TRADE_STREAMS * OPS_FULL
    + (N_ORDER_STREAMS + N_MKT_STREAMS) * OPS_LITE;
/// 每股输出列数：n_trades + 3 个时段。
pub const N_FACTORS: usize = 1 + 3 * SCOPE_COLS;

const TRADE_LIST: [&str; N_TRADE_STREAMS] = [
    "sign", "tick", "tick_d", "tick_big", "to50", "to60", "to70", "to80",
    "to90", "to95", "dir_b70", "dir_b80", "dir_b90", "to_chg", "sign_big", "big_in_buy",
    "gap30", "gap50", "gap70", "gap80", "gap90", "gap_burst", "gap_chg",
    "same_bid", "same_ask", "seq_bid", "seq_ask", "ord_diff_pos", "odf50", "odf80",
    "pos100", "pos500", "sec1s", "sec3s",
];
const ORDER_LIST: [&str; N_ORDER_STREAMS] = [
    "ofills50", "ofills80", "oto50", "oto80", "odur50", "odur80",
    "opw", "odir", "ofill1", "oarr50", "oarr80", "odec_dom",
];
const MKT_LIST: [&str; N_MKT_STREAMS] = [
    "imb_pos", "imb_chg", "mid_up", "spread_up", "bv0_up", "bv2_up", "bv4_up",
    "bv9_up", "av0_up", "av2_up", "av4_up", "av9_up", "tb_up", "ta_up", "imb_big",
    "imb3_pos", "imb10_pos", "dep_b", "mid_pos50",
];
const SCOPES: [&str; 3] = ["f_", "a_", "t_"];

// ---------- 序列基础 ----------

fn mean_f(xs: &[u8]) -> f64 {
    if xs.is_empty() { f64::NAN } else { xs.iter().map(|&x| x as f64).sum::<f64>() / xs.len() as f64 }
}

fn ac_lag(s: &[u8], lag: usize) -> f64 {
    let n = s.len();
    if n <= lag + 1 { return f64::NAN; }
    let m: f64 = s.iter().map(|&x| 2.0 * x as f64 - 1.0).sum::<f64>() / n as f64;
    let (mut num, mut den) = (0.0, 0.0);
    for i in 0..n {
        let x = 2.0 * s[i] as f64 - 1.0 - m;
        den += x * x;
        if i + lag < n { num += x * (2.0 * s[i + lag] as f64 - 1.0 - m); }
    }
    if den < 1e-12 { f64::NAN } else { num / den }
}

fn ac_decay(s: &[u8]) -> f64 {
    let mut xs = Vec::new();
    for l in 1..=10usize {
        let a = ac_lag(s, l);
        if a.is_finite() { xs.push((l as f64, a)); }
    }
    if xs.len() < 5 { return f64::NAN; }
    let n = xs.len() as f64;
    let mx = xs.iter().map(|x| x.0).sum::<f64>() / n;
    let my = xs.iter().map(|x| x.1).sum::<f64>() / n;
    let (mut c, mut v) = (0.0, 0.0);
    for &(x, y) in &xs { c += (x - mx) * (y - my); v += (x - mx).powi(2); }
    if v < 1e-12 { f64::NAN } else { c / v }
}

/// hurst 粗估：聚合块方差的 log-log 斜率，>0.5 长程相关
fn hurst(s: &[u8]) -> f64 {
    let n = s.len();
    if n < 64 { return f64::NAN; }
    let x: Vec<f64> = s.iter().map(|&v| 2.0 * v as f64 - 1.0).collect();
    let mut pts = Vec::new();
    for k in [2usize, 4, 8, 16, 32] {
        let nb = n / k;
        if nb < 8 { continue; }
        let mut means = Vec::with_capacity(nb);
        for b in 0..nb {
            means.push(x[b * k..(b + 1) * k].iter().sum::<f64>() / k as f64);
        }
        let m = means.iter().sum::<f64>() / nb as f64;
        let var = means.iter().map(|v| (v - m).powi(2)).sum::<f64>() / nb as f64;
        if var > 0.0 { pts.push(((k as f64).ln(), var.ln() / 2.0)); }
    }
    if pts.len() < 3 { return f64::NAN; }
    let n2 = pts.len() as f64;
    let mx = pts.iter().map(|p| p.0).sum::<f64>() / n2;
    let my = pts.iter().map(|p| p.1).sum::<f64>() / n2;
    let (mut c, mut v) = (0.0, 0.0);
    for &(x_, y_) in &pts { c += (x_ - mx) * (y_ - my); v += (x_ - mx).powi(2); }
    if v < 1e-12 { f64::NAN } else { c / v }
}

fn ctx_index(s: &[u8], k: usize) -> Vec<usize> {
    let n = s.len();
    if n <= k { return Vec::new(); }
    let mask = (1usize << k) - 1;
    let mut out = Vec::with_capacity(n - k);
    let mut cur = 0usize;
    for i in 0..k { cur = (cur << 1) | s[i] as usize; }
    for i in 0..(n - k) {
        out.push(cur & mask);
        cur = ((cur << 1) | s[i + k] as usize) & mask;
    }
    out
}

/// 一趟滚动扫描同时给出 k ∈ {2,3,4,8} 的上下文索引。ctxk[j] 对应下一位 s[j+k]。
fn ctx_all(s: &[u8]) -> (Vec<usize>, Vec<usize>, Vec<usize>, Vec<usize>) {
    let n = s.len();
    let mut c2 = Vec::with_capacity(n.saturating_sub(2));
    let mut c3 = Vec::with_capacity(n.saturating_sub(3));
    let mut c4 = Vec::with_capacity(n.saturating_sub(4));
    let mut c8 = Vec::with_capacity(n.saturating_sub(8));
    let mut cur = 0usize;
    for i in 0..n {
        cur = ((cur << 1) | s[i] as usize) & 0xFF;
        // 位置 j=i-k 的上下文 k 位窗口 (s[j..i])，下一拍 s[i+1] 必须存在 → i <= n-2
        if i >= 1 && i + 1 < n { c2.push(cur & 3); }
        if i >= 2 && i + 1 < n { c3.push(cur & 7); }
        if i >= 3 && i + 1 < n { c4.push(cur & 15); }
        if i >= 7 && i + 1 < n { c8.push(cur & 255); }
    }
    (c2, c3, c4, c8)
}

fn count_table(ctx: &[usize], next: &[u8], nc: usize) -> (Vec<u64>, Vec<u64>) {
    let mut c0 = vec![0u64; nc];
    let mut c1 = vec![0u64; nc];
    for (i, &c) in ctx.iter().enumerate() {
        if next[i] == 0 { c0[c] += 1; } else { c1[c] += 1; }
    }
    (c0, c1)
}

fn oos_acc(ctx: &[usize], next: &[u8], nc: usize) -> f64 {
    let m = ctx.len();
    if m < 100 { return f64::NAN; }
    let h = m / 2;
    let (c0, c1) = count_table(&ctx[..h], &next[..h], nc);
    let mut ok = 0u64;
    for i in h..m {
        if (c1[ctx[i]] > c0[ctx[i]]) as u8 == next[i] { ok += 1; }
    }
    ok as f64 / (m - h) as f64
}

fn cond_acc_bit(ctx: &[usize], next: &[u8], nc: usize, bit: u8) -> f64 {
    let m = ctx.len();
    if m < 100 { return f64::NAN; }
    let h = m / 2;
    let (c0, c1) = count_table(&ctx[..h], &next[..h], nc);
    let (mut ok, mut tot) = (0u64, 0u64);
    for i in h..m {
        if next[i] != bit { continue; }
        tot += 1;
        if (c1[ctx[i]] > c0[ctx[i]]) as u8 == next[i] { ok += 1; }
    }
    if tot == 0 { f64::NAN } else { ok as f64 / tot as f64 }
}

fn cond_entropy(ctx: &[usize], next: &[u8], nc: usize) -> f64 {
    let (c0, c1) = count_table(ctx, next, nc);
    let m = ctx.len() as f64;
    let mut h = 0.0;
    for c in 0..nc {
        let t = (c0[c] + c1[c]) as f64;
        if t <= 0.0 { continue; }
        for v in [c0[c], c1[c]] {
            if v > 0 {
                let p = v as f64 / t;
                h -= (t / m) * p * p.ln();
            }
        }
    }
    h
}

fn binary_entropy(p: f64) -> f64 {
    if p <= 0.0 || p >= 1.0 { 0.0 } else { -(p * p.ln() + (1.0 - p) * (1.0 - p).ln()) }
}

fn run_stats(s: &[u8]) -> (f64, f64, f64, u64, u64, u64, u64) {
    if s.is_empty() { return (f64::NAN, f64::NAN, f64::NAN, 0, 0, 0, 0); }
    let (mut r1, mut r0, mut c1, mut c0) = (0u64, 0u64, 0u64, 0u64);
    let (mut mx1, mut mx0, mut ms1, mut ms0) = (0u64, 0u64, 0u64, 0u64);
    let mut cur = s[0];
    let mut len = 1u64;
    macro_rules! flush {
        () => {{
            if cur == 1 {
                r1 += len; c1 += 1; mx1 = mx1.max(len);
                if len >= 5 { ms1 += len; }
            } else {
                r0 += len; c0 += 1; mx0 = mx0.max(len);
                if len >= 5 { ms0 += len; }
            }
        }};
    }
    for &b in &s[1..] {
        if b == cur { len += 1; }
        else { flush!(); cur = b; len = 1; }
    }
    flush!();
    let rr = (c1 + c0) as f64 / s.len() as f64;
    (if c1 > 0 { r1 as f64 / c1 as f64 } else { 0.0 },
     if c0 > 0 { r0 as f64 / c0 as f64 } else { 0.0 }, rr, mx1, mx0, ms1, ms0)
}

fn motif_cond(ctx: &[usize], next: &[u8], nc: usize) -> Vec<f64> {
    let (c0, c1) = count_table(ctx, next, nc);
    (0..nc).map(|c| {
        let t = c0[c] + c1[c];
        if t >= 30 { c1[c] as f64 / t as f64 } else { f64::NAN }
    }).collect()
}

fn motif_share(ctx: &[usize], nc: usize) -> Vec<f64> {
    let mut cnt = vec![0u64; nc];
    for &c in ctx { cnt[c] += 1; }
    let t = ctx.len() as f64;
    cnt.iter().map(|&v| v as f64 / t).collect()
}

fn motif_entropy(ctx: &[usize], nc: usize) -> f64 {
    let sh = motif_share(ctx, nc);
    let mut h = 0.0;
    for p in sh {
        if p > 0.0 { h -= p * p.ln(); }
    }
    h
}

// ---------- 流构造 ----------

fn qtile(v: &[f64], q: f64) -> f64 {
    if v.is_empty() { return f64::NAN; }
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    s[((s.len() - 1) as f64 * q).round() as usize]
}

fn binarize_by(v: &[f64], thr: f64, above: bool) -> Vec<u8> {
    v.iter().map(|&x| if above { (x > thr) as u8 } else { (x <= thr) as u8 }).collect()
}

fn trade_streams(trades: &[TradeRecord]) -> HashMap<String, Vec<u8>> {
    let n = trades.len();
    let mut m = HashMap::new();
    if n < 300 { return m; }
    let sign: Vec<u8> = trades.iter().map(|t| (t.flag == 66) as u8).collect();
    m.insert("sign".into(), sign.clone());

    let mut tick = Vec::with_capacity(n - 1);
    let mut tick_d = Vec::with_capacity(n - 1);
    let mut ups = Vec::with_capacity(n - 1);
    for i in 1..n {
        let d = trades[i].price - trades[i - 1].price;
        if d != 0.0 { tick.push((d > 0.0) as u8); ups.push(d as f64); }
        tick_d.push((d > 0.0) as u8);
    }
    m.insert("tick".into(), tick);
    m.insert("tick_d".into(), tick_d);
    if ups.len() > 200 {
        let m_up = qtile(&ups, 0.5);
        m.insert("tick_big".into(), ups.iter().map(|&d| (d.abs() > m_up) as u8).collect());
    }

    let tos: Vec<f64> = trades.iter().map(|t| t.turnover as f64).collect();
    for q in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95] {
        let thr = qtile(&tos, q);
        m.insert(format!("to{}", (q * 100.0) as u32), binarize_by(&tos, thr, true));
    }
    for q in [0.7, 0.8, 0.9] {
        let key = format!("to{}", (q * 100.0) as u32);
        if let Some(b) = m.get(&key) {
            m.insert(format!("dir_b{}", (q * 100.0) as u32),
                     sign.iter().zip(b).map(|(&s, &g)| s & g).collect());
        }
    }
    let mut to_c = Vec::with_capacity(n - 1);
    for i in 1..n { to_c.push((tos[i] > tos[i - 1]) as u8); }
    m.insert("to_chg".into(), to_c);
    if let Some(b) = m.get("to80") {
        let g: Vec<u8> = sign.iter().zip(b).filter(|(_, &g)| g == 1).map(|(&x, _)| x).collect();
        if g.len() >= 300 { m.insert("sign_big".into(), g); }
    }
    let to80_thr = qtile(&tos, 0.8);
    let dir1: Vec<u8> = tos.iter().zip(&sign).filter(|(_, &g)| g == 1)
        .map(|(&t, _)| (t > to80_thr) as u8).collect();
    if dir1.len() >= 300 { m.insert("big_in_buy".into(), dir1); }

    let mut gaps = Vec::with_capacity(n - 1);
    for i in 1..n { gaps.push((trades[i].time_us - trades[i - 1].time_us) as f64 / 1000.0); }
    for q in [0.3, 0.5, 0.7, 0.8, 0.9] {
        let thr = qtile(&gaps, q);
        m.insert(format!("gap{}", (q * 100.0) as u32), binarize_by(&gaps, thr, false));
    }
    m.insert("gap_burst".into(), binarize_by(&gaps, qtile(&gaps, 0.5) * 5.0, true));
    let mut g_c = Vec::with_capacity(gaps.len().saturating_sub(1));
    for i in 1..gaps.len() { g_c.push((gaps[i] > gaps[i - 1]) as u8); }
    m.insert("gap_chg".into(), g_c);

    let mut same_bid = vec![0u8; n];
    let mut same_ask = vec![0u8; n];
    let mut seq_bid = vec![0u8; n];
    let mut seq_ask = vec![0u8; n];
    for i in 1..n {
        same_bid[i] = (trades[i].bid_order == trades[i - 1].bid_order) as u8;
        same_ask[i] = (trades[i].ask_order == trades[i - 1].ask_order) as u8;
        seq_bid[i] = (trades[i].bid_order == trades[i - 1].bid_order + 1) as u8;
        seq_ask[i] = (trades[i].ask_order == trades[i - 1].ask_order + 1) as u8;
    }
    m.insert("same_bid".into(), same_bid);
    m.insert("same_ask".into(), same_ask);
    m.insert("seq_bid".into(), seq_bid);
    m.insert("seq_ask".into(), seq_ask);
    m.insert("ord_diff_pos".into(),
             trades.iter().map(|t| (t.ask_order > t.bid_order) as u8).collect());
    let odf: Vec<f64> = trades.iter().map(|t| (t.ask_order - t.bid_order).abs() as f64).collect();
    for q in [0.5, 0.8] {
        let thr = qtile(&odf, q);
        m.insert(format!("odf{}", (q * 100.0) as u32), binarize_by(&odf, thr, true));
    }

    // 滚动价格位置 100/500（单调栈，头指针出队 O(1)）
    for w in [100usize, 500] {
        let mut hi = Vec::with_capacity(n);
        let mut lo = Vec::with_capacity(n);
        let mut dq_hi: Vec<usize> = Vec::new();
        let mut dq_lo: Vec<usize> = Vec::new();
        let (mut h0, mut l0) = (0usize, 0usize);
        for i in 0..n {
            while dq_hi.len() > h0 && trades[*dq_hi.last().unwrap()].price <= trades[i].price { dq_hi.pop(); }
            dq_hi.push(i);
            while dq_hi[h0] + w < i { h0 += 1; }
            hi.push(trades[dq_hi[h0]].price as f64);
            while dq_lo.len() > l0 && trades[*dq_lo.last().unwrap()].price >= trades[i].price { dq_lo.pop(); }
            dq_lo.push(i);
            while dq_lo[l0] + w < i { l0 += 1; }
            lo.push(trades[dq_lo[l0]].price as f64);
        }
        m.insert(format!("pos{w}"), (0..n).map(|i| {
            if hi[i] > lo[i] { (((trades[i].price as f64) - lo[i]) / (hi[i] - lo[i]) > 0.5) as u8 } else { 0 }
        }).collect());
    }

    // 秒级聚合桶
    let t0 = trades[0].time_us;
    for span in [1_000_000i64, 3_000_000] {
        let mut bucket: HashMap<i64, (u64, u64)> = HashMap::new();
        for (i, t) in trades.iter().enumerate() {
            let s = (t.time_us - t0) / span;
            let e = bucket.entry(s).or_insert((0, 0));
            if sign[i] == 1 { e.1 += 1; } else { e.0 += 1; }
        }
        let mut secs: Vec<i64> = bucket.keys().copied().collect();
        secs.sort_unstable();
        let bs: Vec<u8> = secs.iter().map(|s| { let (a, b) = bucket[s]; (b > a) as u8 }).collect();
        if bs.len() >= 200 { m.insert(format!("sec{}s", span / 1_000_000), bs); }
    }
    m
}

/// 订单聚合流：按订单号把逐笔聚合成 fill 事件序列
fn order_streams(trades: &[TradeRecord]) -> HashMap<String, Vec<u8>> {
    let mut m = HashMap::new();
    let n = trades.len();
    if n < 300 { return m; }
    let mut groups: std::collections::BTreeMap<i64, Vec<usize>> = std::collections::BTreeMap::new();
    for (i, t) in trades.iter().enumerate() {
        groups.entry(t.bid_order).or_default().push(i);
    }
    if groups.len() < 100 { return m; }
    let mut recs: Vec<(i64, u64, f64, i64, i8, i64, usize)> = Vec::with_capacity(groups.len());
    for (oid, idx) in &groups {
        let fills = idx.len() as u64;
        let to: f64 = idx.iter().map(|&i| trades[i].turnover as f64).sum();
        let dur = trades[*idx.last().unwrap()].time_us - trades[idx[0]].time_us;
        let pw = (trades[*idx.last().unwrap()].price - trades[idx[0]].price).signum() as i8;
        let (mut b, mut a) = (0u64, 0u64);
        for &i in idx { if trades[i].flag == 66 { b += 1; } else { a += 1; } }
        let dir = if b >= a { 1i8 } else { -1i8 };
        recs.push((*oid, fills, to, dur, pw, dir as i64, idx[0]));
    }
    recs.sort_by_key(|r| r.6);
    let nos = recs.len();
    let fills: Vec<f64> = recs.iter().map(|r| r.1 as f64).collect();
    let tos: Vec<f64> = recs.iter().map(|r| r.2).collect();
    let durs: Vec<f64> = recs.iter().map(|r| r.3 as f64 / 1000.0).collect();
    let pws: Vec<u8> = recs.iter().map(|r| (r.4 > 0) as u8).collect();
    let odir: Vec<u8> = recs.iter().map(|r| (r.5 > 0) as u8).collect();
    let mut arrivals = Vec::with_capacity(nos.saturating_sub(1));
    for i in 1..nos {
        arrivals.push((trades[recs[i].6].time_us - trades[recs[i - 1].6].time_us) as f64 / 1000.0);
    }
    for q in [0.5, 0.8] {
        m.insert(format!("ofills{}", (q * 100.0) as u32), binarize_by(&fills, qtile(&fills, q), true));
        m.insert(format!("oto{}", (q * 100.0) as u32), binarize_by(&tos, qtile(&tos, q), true));
        m.insert(format!("odur{}", (q * 100.0) as u32), binarize_by(&durs, qtile(&durs, q), true));
    }
    m.insert("opw".into(), pws);
    m.insert("odir".into(), odir);
    m.insert("ofill1".into(), fills.iter().map(|&f| (f <= 1.0) as u8).collect());
    if arrivals.len() >= 100 {
        m.insert("oarr50".into(), binarize_by(&arrivals, qtile(&arrivals, 0.5), false));
        m.insert("oarr80".into(), binarize_by(&arrivals, qtile(&arrivals, 0.8), false));
    }
    let edges: Vec<f64> = (1..10).map(|i| qtile(&tos, i as f64 / 10.0)).collect();
    let mut dec = vec![0u64; 10];
    for &t in &tos {
        let mut d = 9;
        for i in 0..9 { if t <= edges[i] { d = i; break; } }
        dec[d] += 1;
    }
    m.insert("odec_dom".into(), dec.iter().map(|&v| (v as f64 / nos as f64) > 0.5).map(|x| x as u8).collect());
    m
}

fn market_streams(mkt: &[MarketRecord]) -> HashMap<String, Vec<u8>> {
    let n = mkt.len();
    let mut m = HashMap::new();
    if n < 200 { return m; }
    let imb: Vec<f64> = (0..n).map(|i| {
        let b = mkt[i].bid_vols[0] as f64; let a = mkt[i].ask_vols[0] as f64;
        if a + b > 0.0 { (b - a) / (a + b) } else { 0.0 }
    }).collect();
    m.insert("imb_pos".into(), imb.iter().map(|&x| (x > 0.0) as u8).collect());
    let mut ic = Vec::with_capacity(n - 1);
    for i in 1..n { ic.push((imb[i] > imb[i - 1]) as u8); }
    m.insert("imb_chg".into(), ic);
    let mid: Vec<f64> = (0..n).map(|i| (mkt[i].ask_prcs[0] as f64 + mkt[i].bid_prcs[0] as f64) / 2.0).collect();
    let mut up = Vec::with_capacity(n - 1);
    for i in 1..n { let d = mid[i] - mid[i - 1]; if d != 0.0 { up.push((d > 0.0) as u8); } }
    m.insert("mid_up".into(), up);
    let spr: Vec<f64> = (0..n).map(|i| {
        let a = mkt[i].ask_prcs[0] as f64; let b = mkt[i].bid_prcs[0] as f64;
        if a > 0.0 && b > 0.0 { (a - b) / b } else { 0.0 }
    }).collect();
    let mut spc = Vec::with_capacity(n - 1);
    for i in 1..n { spc.push((spr[i] > spr[i - 1]) as u8); }
    m.insert("spread_up".into(), spc);
    for d in [0usize, 2, 4, 9] {
        let mut v = Vec::with_capacity(n - 1);
        for i in 1..n { v.push((mkt[i].bid_vols[d] > mkt[i - 1].bid_vols[d]) as u8); }
        m.insert(format!("bv{d}_up"), v);
        let mut v2 = Vec::with_capacity(n - 1);
        for i in 1..n { v2.push((mkt[i].ask_vols[d] > mkt[i - 1].ask_vols[d]) as u8); }
        m.insert(format!("av{d}_up"), v2);
    }
    let mut tb = Vec::with_capacity(n - 1);
    let mut ta = Vec::with_capacity(n - 1);
    for i in 1..n {
        tb.push((mkt[i].total_bid_vol > mkt[i - 1].total_bid_vol) as u8);
        ta.push((mkt[i].total_ask_vol > mkt[i - 1].total_ask_vol) as u8);
    }
    m.insert("tb_up".into(), tb);
    m.insert("ta_up".into(), ta);
    let imb_abs: Vec<f64> = imb.iter().map(|x| x.abs()).collect();
    m.insert("imb_big".into(), binarize_by(&imb_abs, qtile(&imb_abs, 0.7), true));
    let imb3: Vec<f64> = (0..n).map(|i| {
        let b: f64 = mkt[i].bid_vols[..3].iter().map(|&x| x as f64).sum();
        let a: f64 = mkt[i].ask_vols[..3].iter().map(|&x| x as f64).sum();
        if a + b > 0.0 { (b - a) / (a + b) } else { 0.0 }
    }).collect();
    m.insert("imb3_pos".into(), imb3.iter().map(|&x| (x > 0.0) as u8).collect());
    let imb10: Vec<f64> = (0..n).map(|i| {
        let b: f64 = mkt[i].bid_vols.iter().map(|&x| x as f64).sum();
        let a: f64 = mkt[i].ask_vols.iter().map(|&x| x as f64).sum();
        if a + b > 0.0 { (b - a) / (a + b) } else { 0.0 }
    }).collect();
    m.insert("imb10_pos".into(), imb10.iter().map(|&x| (x > 0.0) as u8).collect());
    m.insert("dep_b".into(), (0..n).map(|i| {
        let sum_b: f64 = mkt[i].bid_vols.iter().map(|&x| x as f64).sum();
        ((mkt[i].total_bid_vol as f64) > sum_b * 1.5) as u8
    }).collect());
    // 中间价相对滚动 50 窗的位置
    let w = 50usize;
    let mut pos = Vec::with_capacity(n);
    for i in 0..n {
        let lo = i.saturating_sub(w);
        let (mut mn, mut mx) = (f64::MAX, f64::MIN);
        for j in lo..=i { let p = mid[j]; if p < mn { mn = p; } if p > mx { mx = p; } }
        pos.push(if mx > mn { ((mid[i] - mn) / (mx - mn) > 0.5) as u8 } else { 0 });
    }
    m.insert("mid_pos50".into(), pos);
    m
}

// ---------- 发射（push 的两种用途：起名 / 填值） ----------

/// Emit sink：起名模式把"前缀+算子名"拼进名字列表，计算模式把值写进固定偏移的行。
trait Emit {
    fn push(&mut self, pre_nm: &str, v: f64);
}

/// 起名模式：pre_nm 已经是完整名（含 scope/流名前缀）。
struct NameEmit<'a> { names: &'a mut Vec<String> }
impl<'a> Emit for NameEmit<'a> {
    fn push(&mut self, pre_nm: &str, _v: f64) { self.names.push(pre_nm.to_string()); }
}

/// 计算模式：名字忽略，只按顺序写值。
struct ValEmit<'a> { row: &'a mut [f64], pos: usize }
impl<'a> Emit for ValEmit<'a> {
    fn push(&mut self, _pre_nm: &str, v: f64) {
        self.row[self.pos] = v;
        self.pos += 1;
    }
}

/// 完整版算子（59 个）：逐笔主流用。pre = "{scope}{stream}_"。
fn ops<E: Emit>(s: &[u8], pre: &str, e: &mut E) {
    macro_rules! psh { ($nm:expr, $v:expr) => { e.push(&format!("{}{}", pre, $nm), $v) } }
    psh!("p", mean_f(s));
    for l in [1usize, 2, 3, 5, 10, 20] {
        psh!(&format!("ac{l}"), ac_lag(s, l));
    }
    psh!("ac_decay", ac_decay(s));
    psh!("hurst", hurst(s));
    let (c2v, c3v, c4v, c8v) = ctx_all(s);
    let mut accs = Vec::new();
    for &k in &[1usize, 2, 3, 4, 6] {
        if s.len() <= k {
            psh!(&format!("acc{k}"), f64::NAN);
            accs.push(f64::NAN);
            continue;
        }
        let ctx = ctx_index(s, k);
        let next = &s[k..];
        let acc = if ctx.len() >= 100 { oos_acc(&ctx, next, 1usize << k) } else { f64::NAN };
        psh!(&format!("acc{k}"), acc);
        accs.push(acc);
    }
    let acc8 = if c8v.len() >= 100 { oos_acc(&c8v, &s[8..], 256) } else { f64::NAN };
    psh!("acc8", acc8);
    accs.push(acc8);
    psh!("acc_gap82", accs[5] - accs[1]);
    let next8 = if s.len() > 8 { &s[8..] } else { &[][..] };
    if c8v.len() >= 100 {
        let (c0, c1) = count_table(&c8v, next8, 256);
        let m = c8v.len() as f64;
        let (mut used, mut top, mut ent) = (0u64, 0u64, 0.0);
        for c in 0..256 {
            let t = c0[c] + c1[c];
            if t == 0 { continue; }
            used += 1; top = top.max(t);
            let pc = t as f64 / m; ent -= pc * pc.ln();
        }
        let h = cond_entropy(&c8v, next8, 256);
        psh!("h_norm", h / binary_entropy(mean_f(s)).max(1e-9));
        psh!("dict_fill", used as f64 / 256.0);
        psh!("ctx_top1", top as f64 / m);
        psh!("ctx_ent", ent);
        psh!("acc8_b", cond_acc_bit(&c8v, next8, 256, 1));
        psh!("acc8_s", cond_acc_bit(&c8v, next8, 256, 0));
    } else {
        for nm in ["h_norm", "dict_fill", "ctx_top1", "ctx_ent", "acc8_b", "acc8_s"] {
            psh!(nm, f64::NAN);
        }
    }
    let next4 = if s.len() > 4 { &s[4..] } else { &[][..] };
    if c4v.len() >= 100 {
        let (c0, c1) = count_table(&c4v, next4, 16);
        let m = c4v.len() as f64;
        let (mut used, mut top) = (0u64, 0u64);
        for c in 0..16 { let t = c0[c] + c1[c]; if t > 0 { used += 1; top = top.max(t); } }
        psh!("dict4_fill", used as f64 / 16.0);
        psh!("ctx4_top1", top as f64 / m);
        psh!("ment4", motif_entropy(&c4v, 16));
    } else {
        psh!("dict4_fill", f64::NAN);
        psh!("ctx4_top1", f64::NAN);
        psh!("ment4", f64::NAN);
    }
    let next2 = if s.len() > 2 { &s[2..] } else { &[][..] };
    if c2v.len() >= 100 {
        let mc = motif_cond(&c2v, next2, 4);
        for (i, nm) in ["t11", "t10", "t01", "t00"].iter().enumerate() {
            psh!(nm, mc[i]);
        }
    } else {
        for nm in ["t11", "t10", "t01", "t00"] { psh!(nm, f64::NAN); }
    }
    let next3 = if s.len() > 3 { &s[3..] } else { &[][..] };
    if c3v.len() >= 100 {
        let mc = motif_cond(&c3v, next3, 8);
        let sh = motif_share(&c3v, 8);
        for i in 0..8 { psh!(&format!("m{i}"), mc[i]); }
        for i in 0..8 { psh!(&format!("ms{i}"), sh[i]); }
        psh!("ment", motif_entropy(&c3v, 8));
        let same_all = sh[0] + sh[7];
        let alt = sh[2] + sh[5];
        psh!("motif_dom", same_all - alt);
        psh!("per_asym", mc[7] - mc[0]);
    } else {
        for i in 0..8 { psh!(&format!("m{i}"), f64::NAN); }
        for i in 0..8 { psh!(&format!("ms{i}"), f64::NAN); }
        psh!("ment", f64::NAN);
        psh!("motif_dom", f64::NAN);
        psh!("per_asym", f64::NAN);
    }
    let (r1, r0, rr, mx1, mx0, ms1, ms0) = run_stats(s);
    psh!("run1", r1);
    psh!("run0", r0);
    psh!("run_rate", rr);
    psh!("run_max1", mx1 as f64);
    psh!("run_max0", mx0 as f64);
    psh!("streak1", ms1 as f64 / s.len() as f64);
    psh!("streak0", ms0 as f64 / s.len() as f64);
    let mut sw = 0u64;
    for i in 1..s.len() { if s[i] != s[i - 1] { sw += 1; } }
    psh!("switch", if s.len() > 1 { sw as f64 / (s.len() - 1) as f64 } else { f64::NAN });
    let nb = 8usize;
    let bn = s.len() / nb;
    if bn >= 30 {
        let mut ps = Vec::with_capacity(nb);
        for b in 0..nb { ps.push(mean_f(&s[b * bn..(b + 1) * bn])); }
        let m = ps.iter().sum::<f64>() / nb as f64;
        let var = ps.iter().map(|p| (p - m).powi(2)).sum::<f64>() / nb as f64;
        psh!("wcv", if m > 1e-9 { var.sqrt() / m } else { f64::NAN });
        psh!("wspread", ps.iter().cloned().fold(f64::MIN, f64::max) - ps.iter().cloned().fold(f64::MAX, f64::min));
        psh!("drift", ps[nb - 1] - ps[0]);
    } else {
        psh!("wcv", f64::NAN);
        psh!("wspread", f64::NAN);
        psh!("drift", f64::NAN);
    }
}

/// 轻量算子（43 个）：订单流、盘口流用。不刷 acc/词典/条件 acc。
fn ops_lite<E: Emit>(s: &[u8], pre: &str, e: &mut E) {
    macro_rules! psh { ($nm:expr, $v:expr) => { e.push(&format!("{}{}", pre, $nm), $v) } }
    psh!("p", mean_f(s));
    for l in [1usize, 2, 3, 5, 10, 20] {
        psh!(&format!("ac{l}"), ac_lag(s, l));
    }
    psh!("ac_decay", ac_decay(s));
    psh!("hurst", hurst(s));
    let (c2v, c3v, _, _) = ctx_all(s);
    let next2 = if s.len() > 2 { &s[2..] } else { &[][..] };
    if c2v.len() >= 100 {
        let mc = motif_cond(&c2v, next2, 4);
        for (i, nm) in ["t11", "t10", "t01", "t00"].iter().enumerate() {
            psh!(nm, mc[i]);
        }
    } else {
        for nm in ["t11", "t10", "t01", "t00"] { psh!(nm, f64::NAN); }
    }
    let next3 = if s.len() > 3 { &s[3..] } else { &[][..] };
    if c3v.len() >= 100 {
        let mc = motif_cond(&c3v, next3, 8);
        let sh = motif_share(&c3v, 8);
        for i in 0..8 { psh!(&format!("m{i}"), mc[i]); }
        for i in 0..8 { psh!(&format!("ms{i}"), sh[i]); }
        psh!("ment", motif_entropy(&c3v, 8));
        let same_all = sh[0] + sh[7];
        let alt = sh[2] + sh[5];
        psh!("motif_dom", same_all - alt);
        psh!("per_asym", mc[7] - mc[0]);
    } else {
        for i in 0..8 { psh!(&format!("m{i}"), f64::NAN); }
        for i in 0..8 { psh!(&format!("ms{i}"), f64::NAN); }
        psh!("ment", f64::NAN);
        psh!("motif_dom", f64::NAN);
        psh!("per_asym", f64::NAN);
    }
    let (r1, r0, rr, mx1, mx0, ms1, ms0) = run_stats(s);
    psh!("run1", r1);
    psh!("run0", r0);
    psh!("run_rate", rr);
    psh!("run_max1", mx1 as f64);
    psh!("run_max0", mx0 as f64);
    psh!("streak1", ms1 as f64 / s.len() as f64);
    psh!("streak0", ms0 as f64 / s.len() as f64);
    let mut sw = 0u64;
    for i in 1..s.len() { if s[i] != s[i - 1] { sw += 1; } }
    psh!("switch", if s.len() > 1 { sw as f64 / (s.len() - 1) as f64 } else { f64::NAN });
    let nb = 8usize;
    let bn = s.len() / nb;
    if bn >= 30 {
        let mut ps = Vec::with_capacity(nb);
        for b in 0..nb { ps.push(mean_f(&s[b * bn..(b + 1) * bn])); }
        let m = ps.iter().sum::<f64>() / nb as f64;
        let var = ps.iter().map(|p| (p - m).powi(2)).sum::<f64>() / nb as f64;
        psh!("wcv", if m > 1e-9 { var.sqrt() / m } else { f64::NAN });
        psh!("wspread", ps.iter().cloned().fold(f64::MIN, f64::max) - ps.iter().cloned().fold(f64::MAX, f64::min));
        psh!("drift", ps[nb - 1] - ps[0]);
    } else {
        psh!("wcv", f64::NAN);
        psh!("wspread", f64::NAN);
        psh!("drift", f64::NAN);
    }
}

/// 一个时段的发射序列。scope ∈ {"f_","a_","t_"}。
fn compute_scope<E: Emit>(
    trades: &[TradeRecord], mkt: Option<&[MarketRecord]>, scope: &str, e: &mut E,
) {
    e.push(&format!("{scope}n_tr"), trades.len() as f64);
    let s_map = trade_streams(trades);
    let o_map = order_streams(trades);
    let m_map = mkt.map(|mk| market_streams(mk)).unwrap_or_default();

    for s in TRADE_LIST {
        let pre = format!("{scope}{s}_");
        if let Some(v) = s_map.get(s) { ops(v, &pre, e); }
        else { skip_ops::<E>(e, OPS_FULL); }
    }
    for s in ORDER_LIST {
        let pre = format!("{scope}{s}_");
        if let Some(v) = o_map.get(s) { ops_lite(v, &pre, e); }
        else { skip_ops::<E>(e, OPS_LITE); }
    }
    for s in MKT_LIST {
        let pre = format!("{scope}{s}_");
        if let Some(v) = m_map.get(s) { ops_lite(v, &pre, e); }
        else { skip_ops::<E>(e, OPS_LITE); }
    }
    if let (Some(sign), Some(imb)) = (s_map.get("sign"), m_map.get("imb_pos")) {
        let ln = sign.len().min(imb.len());
        if ln >= 300 {
            let s = &sign[..ln]; let t = &imb[..ln];
            let (mut a, mut b, mut c, mut d) = (0u64, 0u64, 0u64, 0u64);
            for i in 0..ln {
                match (s[i], t[i]) {
                    (1, 1) => a += 1, (1, 0) => b += 1, (0, 1) => c += 1, (0, 0) => d += 1, _ => {}
                }
            }
            let mi = {
                let p = |x: u64| x as f64 / ln as f64;
                let mut mi = 0.0;
                for (v, ps, pt) in [(a, p(a + b), p(a + c)), (b, p(a + b), p(b + d)),
                                     (c, p(c + d), p(a + c)), (d, p(c + d), p(b + d))] {
                    let pj = p(v);
                    if pj > 0.0 && ps > 0.0 && pt > 0.0 { mi += pj * (pj / (ps * pt)).ln(); }
                }
                mi
            };
            e.push(&format!("{scope}xt_imb_mi"), mi);
            let ms = mean_f(s); let mt = mean_f(t);
            let (mut num, mut den) = (0.0, 0.0);
            for i in 0..ln - 1 {
                num += (s[i] as f64 - ms) * (t[i + 1] as f64 - mt);
                den += (s[i] as f64 - ms).powi(2);
            }
            e.push(&format!("{scope}xt_imb_lead"), if den > 0.0 { num / den } else { f64::NAN });
        } else {
            e.push(&format!("{scope}xt_imb_mi"), f64::NAN);
            e.push(&format!("{scope}xt_imb_lead"), f64::NAN);
        }
    } else {
        e.push(&format!("{scope}xt_imb_mi"), f64::NAN);
        e.push(&format!("{scope}xt_imb_lead"), f64::NAN);
    }
    let tos: Vec<f64> = trades.iter().map(|t| t.turnover as f64).collect();
    let edges: Vec<f64> = (1..10).map(|i| qtile(&tos, i as f64 / 10.0)).collect();
    let mut dec = vec![0u64; 10];
    for t in trades {
        let mut d = 9;
        for i in 0..9 { if t.turnover as f64 <= edges[i] { d = i; break; } }
        dec[d] += 1;
    }
    e.push(&format!("{scope}to_dec_dom"), dec.iter().max().copied().unwrap_or(0) as f64 / trades.len() as f64);
}

/// 缺流时按 emit 顺序补位：值写 NaN（计算模式）；起名模式把 "__nan" 拼进名字占位。
fn skip_ops<E: Emit>(e: &mut E, n: usize) {
    for _ in 0..n { e.push("__nan", f64::NAN); }
}

/// canonical 因子名：发射顺序固定，长度 = N_FACTORS。
/// 起名时用一支满足最小长度的 dummy 流走完整 ops/ops_lite；
/// ops 内部缺格分支也 push，名字个数不受数据长度影响。
pub fn orderflow_ctx_names() -> Vec<String> {
    let mut names = Vec::with_capacity(N_FACTORS);
    names.push("n_trades".into());
    let dummy: Vec<u8> = vec![0; 500];
    for scope in SCOPES {
        let mut e = NameEmit { names: &mut names };
        e.push(&format!("{scope}n_tr"), 0.0);
        for s in TRADE_LIST {
            let pre = format!("{scope}{s}_");
            ops::<NameEmit>(&dummy, &pre, &mut e);
        }
        for s in ORDER_LIST {
            let pre = format!("{scope}{s}_");
            ops_lite::<NameEmit>(&dummy, &pre, &mut e);
        }
        for s in MKT_LIST {
            let pre = format!("{scope}{s}_");
            ops_lite::<NameEmit>(&dummy, &pre, &mut e);
        }
        e.push(&format!("{scope}xt_imb_mi"), 0.0);
        e.push(&format!("{scope}xt_imb_lead"), 0.0);
        e.push(&format!("{scope}to_dec_dom"), 0.0);
    }
    names
}

// ---------- per-stock 与横截面 ----------

/// 每股一天：行宽 N_FACTORS，缺的时段/流补 NaN。返回 None 表示连全天 2000 笔都不到。
fn per_stock_row(trades: &[TradeRecord], mkt: Option<&[MarketRecord]>) -> Option<Vec<f64>> {
    if trades.len() < 2000 { return None; }
    let mut row = vec![f64::NAN; N_FACTORS];
    row[0] = trades.len() as f64;
    for (si, scope) in SCOPES.iter().enumerate() {
        let base = 1 + si * SCOPE_COLS;
        let lo = match si {
            0 => i64::MIN,
            1 => 11 * 3600 * 1_000_000i64 + 30 * 60 * 1_000_000,
            _ => 12 * 3600 * 1_000_000i64 + 57 * 60 * 1_000_000,
        };
        let scope_trades: Vec<TradeRecord> = trades.iter().cloned()
            .filter(|t| (t.time_us / 1_000_000).rem_euclid(86_400) * 1_000_000 >= lo)
            .collect();
        if scope_trades.len() < 500 { continue; }
        let scope_mkt: Option<Vec<MarketRecord>> = mkt.map(|mk| mk.iter().cloned()
            .filter(|m| (m.time_us / 1_000_000).rem_euclid(86_400) * 1_000_000 >= lo)
            .collect());
        let mut e = ValEmit { row: &mut row[base..base + SCOPE_COLS], pos: 0 };
        compute_scope(&scope_trades, scope_mkt.as_deref(), scope, &mut e);
        debug_assert_eq!(e.pos, SCOPE_COLS);
    }
    Some(row)
}

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

fn compute_from_rows(codes: &[String], rows: Vec<Option<Vec<f64>>>) -> (Vec<String>, Vec<f32>) {
    let mut valid_codes = Vec::new();
    let mut mat: Vec<Vec<f64>> = Vec::new();
    for (c, r) in codes.iter().zip(rows.iter()) {
        if let Some(v) = r {
            valid_codes.push(c.clone());
            mat.push(v.clone());
        }
    }
    let n = valid_codes.len();
    if n == 0 { return (Vec::new(), Vec::new()); }
    // ORDERFLOW_CTX_RAW=1：调试开关，输出原始值不 z-score（与 sandbox 对账用）。
    let raw = std::env::var("ORDERFLOW_CTX_RAW").ok().as_deref() == Some("1");
    if !raw {
        for j in 0..N_FACTORS {
            let mut col: Vec<f64> = mat.iter().map(|r| r[j]).collect();
            zscore_column(&mut col);
            for (i, &v) in col.iter().enumerate() { mat[i][j] = v; }
        }
    }
    let mut vals = Vec::with_capacity(n * N_FACTORS);
    for r in &mat { vals.extend(r.iter().map(|&v| v as f32)); }
    (valid_codes, vals)
}

/// 横截面核心：读全市场 → per-stock 三时段流×算子 → 每列全市场 z-score。
pub fn compute_orderflow_ctx_full(date: i64) -> std::io::Result<(Vec<String>, Vec<f32>)> {
    let codes = crate::data_paths::list_stock_codes(date, "transaction");
    let rows: Vec<Option<Vec<f64>>> = codes
        .par_iter()
        .map(|code| {
            let t = read_trade_fast_inner(code, date, false, true, usize::MAX).ok()?;
            let mk = read_market_fast_inner(code, date, false, true, usize::MAX).ok();
            per_stock_row(&t, mk.as_deref())
        })
        .collect();
    Ok(compute_from_rows(&codes, rows))
}

/// Python 单日调试：返回 (codes, vals)。
#[pyfunction]
pub fn py_orderflow_ctx(py: Python<'_>, date: i64) -> PyResult<(Vec<String>, Vec<f32>)> {
    compute_orderflow_ctx_full(date)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("{e:?}")))
}

/// Python 拿因子名（长度 = N_FACTORS）。
#[pyfunction]
pub fn py_orderflow_ctx_names() -> Vec<String> {
    orderflow_ctx_names()
}
