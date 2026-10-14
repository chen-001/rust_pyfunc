//! 订单群体分布研究（sandbox v8）。
//! 思路：把一天的全部订单按「买/卖」分成两个群体，每个订单是一个带 (成交笔数, 成交量, 价差位置, 主被动)
//! 的对象；对两个群体各算分布形状、尾部、集中度、2D 联合结构，再算两群体之间的分布距离与不对称。
//! 订单身份的语义（照 src/order_pair_metrics.rs:360 的口径）：flag=66 主动买，bid_order 是主动方、
//! ask_order 是被动方；flag=83 主动卖，ask_order 主动、bid_order 被动。所以 bid_order 是买方单号、
//! ask_order 是卖方单号，两侧都同时包含主动单与被动单，用 flag 区分。
//! 构建: cd /home/chenzongwei/rust_pyfunc/sandbox_oidflow && cargo build --release
//! 运行: ./target/release/oidflow_sandbox 20220819 > out_v8_20220819.json
mod fast_csv_reader;
use rayon::prelude::*;
use serde::Serialize;
use std::collections::HashMap;

// 输出名顺序见 names()。diag: [n_trade, n_order_buy, n_order_sell, span_min]
#[derive(Serialize)]
struct Out {
    names: Vec<String>,
    codes: Vec<String>,
    vals: Vec<Vec<f32>>,
    diag: Vec<Vec<f32>>,
}

#[derive(Clone, Copy, Default)]
struct Ord {
    side: u8,       // 0 买 1 卖
    fills: f64,     // 被吃掉的成交笔数
    vol: f64,       // 成交量
    first: f64,     // 首笔时刻（秒）
    last: f64,      // 末笔时刻（秒）
    pmin: f64,
    pmax: f64,
    aggr: bool,     // 至少有一笔是主动方吃别人
}

fn stock_codes_for_day(
    dir: &std::path::Path,
    symbol_map: &std::path::Path,
    date: i64,
    subdir: &str,
) -> std::io::Result<Vec<String>> {
    use std::io::{Error, ErrorKind};
    let invalid = |e: &str| Error::new(ErrorKind::InvalidData, e.to_string());
    let text = std::fs::read_to_string(symbol_map).map_err(|e| invalid(&e.to_string()))?;
    let mut lines = text.lines();
    let header = lines.next().ok_or_else(|| invalid("股票表为空"))?;
    let col = header
        .split(',')
        .position(|h| h.trim() == "symbol")
        .ok_or_else(|| invalid("股票表缺少 symbol 列"))?;
    let mut stocks = std::collections::BTreeSet::new();
    for line in lines {
        let code = line.split(',').nth(col).unwrap_or("").trim();
        if code.is_empty() {
            continue;
        }
        if code.len() > 6 || !code.bytes().all(|b| b.is_ascii_digit()) {
            return Err(invalid("股票表含无效代码"));
        }
        stocks.insert(format!("{code:0>6}"));
    }
    if stocks.is_empty() || !dir.is_dir() {
        return Err(invalid("股票白名单为空或原始数据目录不存在"));
    }
    Ok(stocks
        .into_iter()
        .filter(|code| dir.join(format!("{code}_{date}_{subdir}.csv")).is_file())
        .collect())
}

fn list_codes(date: i64) -> std::io::Result<Vec<String>> {
    let root = std::env::var("RUST_PYFUNC_LEVEL2_ROOT").unwrap_or("/ssd_data/stock".into());
    let basic =
        std::env::var("RUST_PYFUNC_BASIC_INFO_DIR").unwrap_or("/ssd_data/data/basic_info".into());
    let dir = std::path::Path::new(&root).join(date.to_string()).join("transaction");
    let map = std::path::Path::new(&basic).join("symbol_map.csv");
    stock_codes_for_day(&dir, &map, date, "transaction")
}

// ── 统计工具 ────────────────────────────────────────────────────────────────
fn mean(x: &[f64]) -> f64 {
    if x.is_empty() { return f64::NAN; }
    x.iter().sum::<f64>() / x.len() as f64
}
fn std(x: &[f64]) -> f64 {
    if x.len() < 2 { return f64::NAN; }
    let m = mean(x);
    (x.iter().map(|v| (v - m).powi(2)).sum::<f64>() / (x.len() as f64 - 1.0)).sqrt()
}
fn skew(x: &[f64]) -> f64 {
    let n = x.len();
    if n < 3 { return f64::NAN; }
    let m = mean(x);
    let s = (x.iter().map(|v| (v - m).powi(2)).sum::<f64>() / n as f64).sqrt();
    if s <= 0.0 { return 0.0; }
    x.iter().map(|v| ((v - m) / s).powi(3)).sum::<f64>() / n as f64
}
fn kurt(x: &[f64]) -> f64 {
    let n = x.len();
    if n < 4 { return f64::NAN; }
    let m = mean(x);
    let s = (x.iter().map(|v| (v - m).powi(2)).sum::<f64>() / n as f64).sqrt();
    if s <= 0.0 { return 0.0; }
    x.iter().map(|v| ((v - m) / s).powi(4)).sum::<f64>() / n as f64 - 3.0
}
fn quantile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() { return f64::NAN; }
    let pos = (sorted.len() - 1) as f64 * q;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    if lo == hi { return sorted[lo]; }
    sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo as f64)
}
fn gini(x: &[f64]) -> f64 {
    let n = x.len();
    if n < 2 { return f64::NAN; }
    let mut s = x.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let tot: f64 = s.iter().sum();
    if tot <= 0.0 { return f64::NAN; }
    let mut acc = 0.0;
    for (i, v) in s.iter().enumerate() {
        acc += (2.0 * (i as f64 + 1.0) - n as f64 - 1.0) * v;
    }
    acc / (n as f64 * tot)
}
fn shannon_entropy(p: &[f64]) -> f64 {
    let mut h = 0.0;
    for &v in p {
        if v > 0.0 {
            h -= v * v.ln();
        }
    }
    h
}
/// 均匀抽样（保序），把大数组缩到 <= k，保持形状。
fn downsample(sorted: &[f64], k: usize) -> Vec<f64> {
    if sorted.len() <= k { return sorted.to_vec(); }
    let step = sorted.len() as f64 / k as f64;
    (0..k).map(|i| sorted[((i as f64) * step) as usize]).collect()
}
/// 两个经验分布的 Wasserstein-1（等权，取同长度抽样后按排名配对）。
fn wasserstein(a: &[f64], b: &[f64], k: usize) -> f64 {
    if a.is_empty() || b.is_empty() { return f64::NAN; }
    let sa = downsample(a, k);
    let sb = downsample(b, k);
    let n = sa.len().min(sb.len());
    let mut d = 0.0;
    for i in 0..n {
        d += (sa[i] - sb[i]).abs();
    }
    d / n as f64
}
/// 分布之间的 KS 统计量。
fn ks_stat(a: &[f64], b: &[f64]) -> f64 {
    if a.is_empty() || b.is_empty() { return f64::NAN; }
    let (mut i, mut j, mut d, mut best) = (0usize, 0usize, 0.0f64, 0.0f64);
    let (na, nb) = (a.len() as f64, b.len() as f64);
    while i < a.len() && j < b.len() {
        if a[i] <= b[j] {
            d += 1.0 / na;
            i += 1;
        } else {
            d -= 1.0 / nb;
            j += 1;
        }
        best = best.max(d.abs());
    }
    best
}
/// 直方分布的 Jensen-Shannon 散度（自然对数）。
fn js_divergence(a: &[f64], b: &[f64], lo: f64, hi: f64, bins: usize) -> f64 {
    if a.is_empty() || b.is_empty() || hi <= lo { return f64::NAN; }
    let w = (hi - lo) / bins as f64;
    let idx = |v: f64| ((v - lo) / w).floor().max(0.0).min((bins - 1) as f64) as usize;
    let mut ha = vec![0.0f64; bins];
    let mut hb = vec![0.0f64; bins];
    for &v in a { ha[idx(v)] += 1.0; }
    for &v in b { hb[idx(v)] += 1.0; }
    let (na, nb) = (a.len() as f64, b.len() as f64);
    for v in ha.iter_mut() { *v /= na; }
    for v in hb.iter_mut() { *v /= nb; }
    let mut m = vec![0.0f64; bins];
    for i in 0..bins { m[i] = 0.5 * (ha[i] + hb[i]); }
    let mut js = 0.0;
    for i in 0..bins {
        if ha[i] > 0.0 && m[i] > 0.0 { js += 0.5 * ha[i] * (ha[i] / m[i]).ln(); }
        if hb[i] > 0.0 && m[i] > 0.0 { js += 0.5 * hb[i] * (hb[i] / m[i]).ln(); }
    }
    js
}
/// 单侧分布的一整套形状指标，返回固定顺序的 12 个值。
#[derive(Default, Clone, Copy)]
struct SideStats {
    n: f64,
    vol: f64,
    fills_mean: f64,
    fills_p50: f64,
    fills_p90: f64,
    fills_max: f64,
    logvol_mean: f64,
    logvol_std: f64,
    logvol_skew: f64,
    logvol_kurt: f64,
    gini_vol: f64,
    top10_vol: f64,
    one_fill_share: f64,
    aggr_share: f64,
    aggr_vol_share: f64,
    off_mean: f64,
    off_vw: f64,
    lvl_mean: f64,
    cxl0_share: f64,
}

fn side_stats(orders: &[Ord], side: u8, total_vol: f64, vwap: f64) -> SideStats {
    let v: Vec<&Ord> = orders.iter().filter(|o| o.side == side).collect();
    let mut s = SideStats { n: v.len() as f64, ..Default::default() };
    if v.is_empty() { return s; }
    let vols: Vec<f64> = v.iter().map(|o| o.vol).collect();
    s.vol = vols.iter().sum();
    let mut fills: Vec<f64> = v.iter().map(|o| o.fills).collect();
    let mut logvol: Vec<f64> = v.iter().map(|o| (o.vol + 1.0).ln()).collect();
    let mut logvol_sorted = logvol.clone();
    logvol_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    fills.sort_by(|a, b| a.partial_cmp(b).unwrap());
    logvol.sort_by(|a, b| a.partial_cmp(b).unwrap());
    s.fills_mean = mean(&fills);
    s.fills_p50 = quantile(&fills, 0.5);
    s.fills_p90 = quantile(&fills, 0.9);
    s.fills_max = *fills.last().unwrap();
    s.logvol_mean = mean(&logvol);
    s.logvol_std = std(&logvol);
    s.logvol_skew = skew(&logvol);
    s.logvol_kurt = kurt(&logvol);
    s.gini_vol = gini(&vols);
    let mut vs = vols.clone();
    vs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let k = ((vs.len() as f64 * 0.9) as usize).min(vs.len().saturating_sub(1));
    let top: f64 = vs[k..].iter().sum();
    s.top10_vol = if total_vol > 0.0 { top / total_vol } else { f64::NAN };
    let one: f64 = v.iter().filter(|o| o.fills <= 1.5).map(|o| o.vol).sum();
    s.one_fill_share = if s.vol > 0.0 { one / s.vol } else { f64::NAN };
    let ag: Vec<&Ord> = v.iter().copied().filter(|o| o.aggr).collect();
    s.aggr_share = ag.len() as f64 / s.n;
    s.aggr_vol_share = if s.vol > 0.0 { ag.iter().map(|o| o.vol).sum::<f64>() / s.vol } else { f64::NAN };
    if vwap > 0.0 {
        let off: Vec<f64> = v.iter()
            .map(|o| ((o.pmin + o.pmax) / 2.0 - vwap) / vwap)
            .collect();
        s.off_mean = mean(&off);
        s.off_vw = if s.vol > 0.0 {
            v.iter().zip(off.iter()).map(|(o, d)| o.vol * d).sum::<f64>() / s.vol
        } else { f64::NAN };
    }
    s.lvl_mean = mean(&v.iter().map(|o| (o.pmax > o.pmin) as i32 as f64 + 1.0).collect::<Vec<f64>>());
    s
}

/// 2D 联合结构：把 (成交笔数档, 成交量十分位) 做成 4x10 直方图，返回 (互信息, 对角集中度, 熵)
fn joint_stats(v: &[&Ord], vols_all: &[f64]) -> (f64, f64, f64) {
    if v.len() < 50 || vols_all.len() < 20 { return (f64::NAN, f64::NAN, f64::NAN); }
    let bins_f = 4usize; // fills: 1, 2, 3, >=4
    let bins_v = 10usize;
    let mut vs = vols_all.to_vec();
    vs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let q = |p: f64| quantile(&vs, p);
    let cuts: Vec<f64> = (1..bins_v).map(|i| q(i as f64 / bins_v as f64)).collect();
    let vb = |x: f64| -> usize {
        let mut b = 0;
        while b < cuts.len() && x > cuts[b] { b += 1; }
        b
    };
    let mut h = vec![0.0f64; bins_f * bins_v];
    for o in v {
        let f = ((o.fills.round() as usize).min(4)).saturating_sub(1).min(bins_f - 1);
        let b = vb(o.vol).min(bins_v - 1);
        h[f * bins_v + b] += 1.0;
    }
    let tot: f64 = h.iter().sum();
    if tot <= 0.0 { return (f64::NAN, f64::NAN, f64::NAN); }
    for x in h.iter_mut() { *x /= tot; }
    let mut row = vec![0.0f64; bins_f];
    let mut col = vec![0.0f64; bins_v];
    for i in 0..bins_f { for j in 0..bins_v { row[i] += h[i * bins_v + j]; col[j] += h[i * bins_v + j]; } }
    let mut mi = 0.0;
    for i in 0..bins_f {
        for j in 0..bins_v {
            let p = h[i * bins_v + j];
            if p > 0.0 && row[i] > 0.0 && col[j] > 0.0 {
                mi += p * (p / (row[i] * col[j])).ln();
            }
        }
    }
    // 归一化互信息（除以 min(熵)）
    let hx = shannon_entropy(&row);
    let hy = shannon_entropy(&col);
    let denom = hx.min(hy);
    let nmi = if denom > 1e-9 { mi / denom } else { f64::NAN };
    // 对角集中度：同档内成交量十分位与笔数分位一致的程度（这里用「最大行占比」代替）
    let diag: f64 = (0..bins_f.min(bins_v)).map(|i| h[i * bins_v + i]).sum();
    (nmi, diag, shannon_entropy(&h))
}

// ── 单股计算 ────────────────────────────────────────────────────────────────
const K_SAMPLE: usize = 2000;
fn per_stock(code: &str, date: i64) -> Option<([f64; 48], [f64; 4])> {
    let root = std::env::var("RUST_PYFUNC_LEVEL2_ROOT").unwrap_or("/ssd_data/stock".into());
    let path = format!("{}/{}/transaction/{}_{}_transaction.csv", root, date, code, date);
    let text = std::fs::read_to_string(&path).ok()?;

    let mut m: HashMap<i64, Ord> = HashMap::new();
    let mut n_trade = 0.0f64;
    let (mut vol_sum, mut turn_sum) = (0.0f64, 0.0f64);
    let (mut bid0, mut ask0) = (0.0f64, 0.0f64);
    for line in text.lines().skip(1) {
        let f: Vec<&str> = line.split(',').collect();
        if f.len() < 15 { continue; }
        let flag: i32 = match f[10].parse() { Ok(v) => v, Err(_) => continue };
        if flag != 66 && flag != 83 { continue; }
        let vol: f64 = match f[8].parse() { Ok(v) => v, Err(_) => continue };
        let price: f64 = match f[7].parse() { Ok(v) => v, Err(_) => continue };
        let t: f64 = match f[4].parse::<i64>() { Ok(v) => v as f64 / 1e6, Err(_) => continue };
        let turn: f64 = f[9].parse().unwrap_or(0.0);
        if !(vol > 0.0) || !(price > 0.0) { continue; }
        let bid: i64 = f[14].parse().unwrap_or(0);
        let ask: i64 = f[13].parse().unwrap_or(0);
        n_trade += 1.0;
        vol_sum += vol;
        turn_sum += turn;
        if bid <= 0 { bid0 += 1.0; }
        if ask <= 0 { ask0 += 1.0; }
        let is_buy = flag == 66;
        // 两侧各登记一笔（同一订单号只出现在一侧）
        let mut upd = |oid: i64, side: u8, aggr: bool| {
            if oid <= 0 { return; }
            let e = m.entry(oid).or_insert(Ord {
                side, fills: 0.0, vol: 0.0, first: t, last: t,
                pmin: price, pmax: price, aggr: false,
            });
            e.fills += 1.0;
            e.vol += vol;
            if t < e.first { e.first = t; }
            if t > e.last { e.last = t; }
            if price < e.pmin { e.pmin = price; }
            if price > e.pmax { e.pmax = price; }
            if aggr { e.aggr = true; }
        };
        if is_buy { upd(bid, 0, true); upd(ask, 1, false); }
        else { upd(ask, 1, true); upd(bid, 0, false); }
    }
    if n_trade < 500.0 || m.len() < 200 { return None; }
    let vwap = if vol_sum > 0.0 { turn_sum / vol_sum } else { f64::NAN };
    if !vwap.is_finite() || vwap <= 0.0 { return None; }
    let orders: Vec<Ord> = m.values().copied().collect();

    let bs = side_stats(&orders, 0, vol_sum, vwap);
    let ss = side_stats(&orders, 1, vol_sum, vwap);

    // 两群体的分布距离
    let mut bv: Vec<f64> = orders.iter().filter(|o| o.side == 0).map(|o| (o.vol + 1.0).ln()).collect();
    let mut sv: Vec<f64> = orders.iter().filter(|o| o.side == 1).map(|o| (o.vol + 1.0).ln()).collect();
    bv.sort_by(|a, b| a.partial_cmp(b).unwrap());
    sv.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let w_full = wasserstein(&bv, &sv, K_SAMPLE);
    let lo_ = if bv.is_empty() || sv.is_empty() { 0.0 } else { bv[0].min(sv[0]) };
    let hi_ = if bv.is_empty() || sv.is_empty() { 1.0 } else { bv[bv.len() - 1].max(sv[sv.len() - 1]) };
    let js_full = js_divergence(&bv, &sv, lo_, hi_, 40);
    let ks_full = ks_stat(&bv, &sv);
    // 尾部（volume 上四分位以上）的分布距离
    let tail = |v: &mut Vec<f64>| {
        let q3 = quantile(v, 0.75);
        v.retain(|x| *x >= q3);
    };
    let mut bvt = bv.clone();
    let mut svt = sv.clone();
    tail(&mut bvt);
    tail(&mut svt);
    let w_tail = wasserstein(&bvt, &svt, K_SAMPLE);
    let js_tail = js_divergence(&bvt, &svt,
        if bvt.is_empty() || svt.is_empty() { 0.0 } else { bvt[0].min(svt[0]) },
        if bvt.is_empty() || svt.is_empty() { 1.0 } else { bvt[bvt.len()-1].max(svt[svt.len()-1]) }, 30);

    // 2D 联合结构
    let all_v: Vec<f64> = orders.iter().map(|o| o.vol).collect();
    let bo: Vec<&Ord> = orders.iter().filter(|o| o.side == 0).collect();
    let so: Vec<&Ord> = orders.iter().filter(|o| o.side == 1).collect();
    let (nmi_b, diag_b, ent_b) = joint_stats(&bo, &all_v);
    let (nmi_s, diag_s, ent_s) = joint_stats(&so, &all_v);

    // 主动侧与被动侧的对比（同一侧内部再分人群）
    let aggr_dists: Vec<f64> = orders.iter().filter(|o| o.aggr).map(|o| (o.vol + 1.0).ln()).collect();
    let pasv_dists: Vec<f64> = orders.iter().filter(|o| !o.aggr).map(|o| (o.vol + 1.0).ln()).collect();
    let mut a_s = aggr_dists.clone();
    let mut p_s = pasv_dists.clone();
    a_s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    p_s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let w_ap = wasserstein(&a_s, &p_s, K_SAMPLE);
    let js_ap = js_divergence(&a_s, &p_s,
        if a_s.is_empty() || p_s.is_empty() { 0.0 } else { a_s[0].min(p_s[0]) },
        if a_s.is_empty() || p_s.is_empty() { 1.0 } else { a_s[a_s.len()-1].max(p_s[p_s.len()-1]) }, 40);

    // 价位条件切割：单号均价高于 vwap 的订单 vs 低于的
    let mut hi_o: Vec<f64> = Vec::new();
    let mut lo_o: Vec<f64> = Vec::new();
    for o in orders.iter() {
        let mid = (o.pmin + o.pmax) / 2.0;
        if mid > vwap { hi_o.push((o.vol + 1.0).ln()); } else { lo_o.push((o.vol + 1.0).ln()); }
    }
    hi_o.sort_by(|a, b| a.partial_cmp(b).unwrap());
    lo_o.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let w_price = wasserstein(&hi_o, &lo_o, K_SAMPLE);

    // 订单时长（首末成交间隔）与「一单撞几人」
    let mut dura: Vec<f64> = orders.iter().filter(|o| o.fills >= 2.0).map(|o| (o.last - o.first).max(0.0)).collect();
    dura.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let dura_p50 = quantile(&dura, 0.5);
    let dura_p90 = quantile(&dura, 0.9);
    let frac_multi = orders.iter().filter(|o| o.fills >= 2.0).count() as f64 / orders.len() as f64;

    let v = [
        bs.n, ss.n,
        bs.vol / vol_sum, ss.vol / vol_sum,
        bs.fills_mean, bs.fills_p50, bs.fills_p90, bs.fills_max,
        ss.fills_mean, ss.fills_p50, ss.fills_p90, ss.fills_max,
        bs.logvol_mean, bs.logvol_std, bs.logvol_skew, bs.logvol_kurt,
        ss.logvol_mean, ss.logvol_std, ss.logvol_skew, ss.logvol_kurt,
        bs.gini_vol, bs.top10_vol, bs.one_fill_share,
        ss.gini_vol, ss.top10_vol, ss.one_fill_share,
        bs.aggr_share, bs.aggr_vol_share, ss.aggr_share, ss.aggr_vol_share,
        bs.off_mean, bs.off_vw, ss.off_mean, ss.off_vw,
        bs.lvl_mean, ss.lvl_mean,
        w_full, js_full, ks_full,
        w_tail, js_tail,
        nmi_b, nmi_s,
        diag_b, diag_s,
        ent_b, ent_s,
        w_ap, js_ap, w_price,
    ];
    let mut out = [0.0f64; 48];
    let take = out.len().min(v.len());
    out[..take].copy_from_slice(&v[..take]);
    // 收尾：把几个额外统计塞进剩余槽位（时长与多笔占比）
    out[45] = dura_p50;
    out[46] = dura_p90;
    out[47] = frac_multi;
    for x in out.iter() {
        if !x.is_finite() { return None; }
    }
    let diag = [n_trade, bs.n, ss.n, (vol_sum.max(0.0)).min(1e18)];
    Some((out, diag))
}

fn names() -> Vec<String> {
    [
        "n_ord_buy", "n_ord_sell", "vol_share_buy", "vol_share_sell",
        "fills_mean_buy", "fills_p50_buy", "fills_p90_buy", "fills_max_buy",
        "fills_mean_sell", "fills_p50_sell", "fills_p90_sell", "fills_max_sell",
        "logvol_mean_buy", "logvol_std_buy", "logvol_skew_buy", "logvol_kurt_buy",
        "logvol_mean_sell", "logvol_std_sell", "logvol_skew_sell", "logvol_kurt_sell",
        "gini_buy", "top10_vol_share_buy", "one_fill_share_buy",
        "gini_sell", "top10_vol_share_sell", "one_fill_share_sell",
        "aggr_share_buy", "aggr_vol_share_buy", "aggr_share_sell", "aggr_vol_share_sell",
        "price_off_mean_buy", "price_off_vw_buy", "price_off_mean_sell", "price_off_vw_sell",
        "level_mean_buy", "level_mean_sell",
        "wass_ordersize_bs", "js_ordersize_bs", "ks_ordersize_bs",
        "wass_tail_bs", "js_tail_bs",
        "nmi_fills_vol_buy", "nmi_fills_vol_sell",
        "diag_share_buy", "diag_share_sell",
        "wass_aggr_passive", "js_aggr_passive",
        "dura_p90_multi", "frac_multi_fill",
    ].iter().map(|s| s.to_string()).collect()
}

fn main() {
    let date: i64 = std::env::args()
        .nth(1)
        .expect("用法: oidflow_sandbox <date>")
        .parse()
        .expect("date 必须是整数");
    let codes = list_codes(date).expect("股票范围或原始目录检查失败");
    eprintln!("codes = {}", codes.len());

    let rows: Vec<(String, [f64; 48], [f64; 4])> = codes
        .par_iter()
        .filter_map(|c| {
            let (v, d) = per_stock(c, date)?;
            Some((c.clone(), v, d))
        })
        .collect();

    let mut out = Out {
        names: names(),
        codes: Vec::with_capacity(rows.len()),
        vals: Vec::with_capacity(rows.len()),
        diag: Vec::with_capacity(rows.len()),
    };
    for (c, v, d) in rows {
        out.codes.push(c);
        out.vals.push(v.iter().map(|&x| x as f32).collect());
        out.diag.push(d.iter().map(|&x| x as f32).collect());
    }
    eprintln!("valid = {}", out.codes.len());
    println!("{}", serde_json::to_string(&out).unwrap());
}
