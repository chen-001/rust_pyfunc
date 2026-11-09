//! 字母化层：把一个时段内的逐笔和盘口变成 10 种“字母序列”。
//! 每个字母有方向 side（+1 / -1 / 0，0 表示中性、不参与配对）、强度 strength（1 弱 / 2 强）、
//! 量 vol（配对权重用）、价 price（价格条件用）。
use crate::fast_csv_reader::{MarketRecord, TradeRecord};

pub const ALPHA_NAMES: [&str; 10] = [
    "flow", "count", "size", "speed", "ordjump", "cancel", "lob", "ret", "bigord", "volbar",
];
pub const N_ALPHA: usize = ALPHA_NAMES.len();

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
    fn bar_idx(&self, time_us: i64) -> usize {
        let s = (time_us / 1_000_000).rem_euclid(86400);
        (((s - self.adj_lo) / self.bar).max(0) as usize).min(self.n() - 1)
    }
}

/// 三个时段。下午时间已由读取器平移 90 分钟（13:00 记作 11:30）；读取器丢弃 14:57 之后，
/// 因此“下午”实际是 13:00-14:57（117 分钟），“尾盘”实际是 14:30-14:57（27 分钟），都不含收盘集合竞价。
pub fn windows(full_bar: i64, pm_bar: i64, tail_bar: i64) -> Vec<Win> {
    vec![
        Win { name: "full", raw_lo: 34200, raw_hi: 53820, adj_lo: 34200, adj_hi: 48420, bar: full_bar },
        Win { name: "pm", raw_lo: 46800, raw_hi: 53820, adj_lo: 41400, adj_hi: 48420, bar: pm_bar },
        Win { name: "tail", raw_lo: 52200, raw_hi: 53820, adj_lo: 46800, adj_hi: 48420, bar: tail_bar },
    ]
}

#[derive(Clone, Default)]
pub struct Seq {
    pub side: Vec<i8>,
    pub strength: Vec<u8>,
    pub vol: Vec<f32>,
    pub price: Vec<f32>,
}

#[inline]
fn dead(x: f64, thr: f64) -> i8 {
    if !x.is_finite() || x.abs() <= thr { 0 } else if x > 0.0 { 1 } else { -1 }
}

fn median(mut v: Vec<f64>) -> f64 {
    if v.is_empty() { return f64::NAN; }
    v.sort_by(|a, b| a.total_cmp(b));
    v[v.len() / 2]
}

fn strengths(mag: &[f64], side: &[i8]) -> Vec<u8> {
    let m = median(mag.iter().zip(side).filter(|(_, &s)| s != 0).map(|(&x, _)| x).collect());
    mag.iter().map(|&x| if x > m { 2 } else { 1 }).collect()
}

fn make_seq(side: Vec<i8>, mag: &[f64], vol: &[f64], price: &[f32]) -> Seq {
    let strength = strengths(mag, &side);
    Seq { side, strength, vol: vol.iter().map(|&x| x as f32).collect(), price: price.to_vec() }
}

/// 生成该时段的 10 条序列；时段内没有任何成交时返回全 None。
pub fn build_sequences(tr: &[TradeRecord], mk: &[MarketRecord], win: &Win) -> Vec<Option<Seq>> {
    let n = win.n();
    let mut bvol = vec![0f64; n];
    let mut svol = vec![0f64; n];
    let mut bn = vec![0f64; n];
    let mut sn = vec![0f64; n];
    let mut bc = vec![0f64; n];
    let mut sc = vec![0f64; n];
    let mut amt = vec![0f64; n];
    let mut last_px = vec![f32::NAN; n];
    let (mut bgap, mut bgn, mut sgap, mut sgn) = (vec![0f64; n], vec![0f64; n], vec![0f64; n], vec![0f64; n]);
    // 按“同一主动委托连续成交”合并成一个事件
    struct Ev { side: i8, vol: f64, amt: f64 }
    let mut events: Vec<Ev> = Vec::new();
    let (mut prev_flag, mut prev_id) = (0i32, -1i64);
    let (mut last_buy, mut last_sell) = (-1i64, -1i64);
    // 量钟要用到的成交明细
    let mut trades: Vec<(i8, f64, f64, i64)> = Vec::new();

    for t in tr {
        if t.raw_sec < win.raw_lo || t.raw_sec > win.raw_hi { continue; }
        let b = win.bar_idx(t.time_us);
        if t.flag == 32 {
            if t.bid_order > 0 && t.ask_order == 0 { bc[b] += t.volume; }
            else if t.ask_order > 0 && t.bid_order == 0 { sc[b] += t.volume; }
            continue;
        }
        if t.price <= 0.0 || t.volume <= 0.0 || (t.flag != 66 && t.flag != 83) { continue; }
        let is_buy = t.flag == 66;
        amt[b] += t.turnover;
        last_px[b] = t.price as f32;
        let id = if is_buy { t.bid_order } else { t.ask_order };
        if is_buy {
            bvol[b] += t.volume;
            bn[b] += 1.0;
            if id != last_buy {
                if last_buy >= 0 && id > last_buy { bgap[b] += ((id - last_buy) as f64).ln(); bgn[b] += 1.0; }
                last_buy = id;
            }
        } else {
            svol[b] += t.volume;
            sn[b] += 1.0;
            if id != last_sell {
                if last_sell >= 0 && id > last_sell { sgap[b] += ((id - last_sell) as f64).ln(); sgn[b] += 1.0; }
                last_sell = id;
            }
        }
        if t.flag == prev_flag && id == prev_id {
            let e = events.last_mut().unwrap();
            e.vol += t.volume;
            e.amt += t.turnover;
        } else {
            events.push(Ev { side: if is_buy { 1 } else { -1 }, vol: t.volume, amt: t.turnover });
            prev_flag = t.flag;
            prev_id = id;
        }
        trades.push((if is_buy { 1 } else { -1 }, t.volume, t.turnover, t.time_us));
    }
    if trades.is_empty() {
        return (0..N_ALPHA).map(|_| None).collect();
    }

    let total: Vec<f64> = (0..n).map(|i| bvol[i] + svol[i]).collect();
    let mut px = vec![f32::NAN; n];
    for i in 0..n {
        if total[i] > 0.0 { px[i] = (amt[i] / total[i]) as f32; }
    }
    let first = px.iter().copied().find(|x| x.is_finite()).unwrap();
    let mut cur = first;
    for p in px.iter_mut() {
        if p.is_finite() { cur = *p; } else { *p = cur; }
    }

    let mut out: Vec<Option<Seq>> = Vec::with_capacity(N_ALPHA);
    let ratio = |a: f64, b: f64| if a + b > 0.0 { (a - b) / (a + b) } else { 0.0 };

    // flow：净主动量
    let side: Vec<i8> = (0..n).map(|i| dead(ratio(bvol[i], svol[i]), 0.1)).collect();
    out.push(Some(make_seq(side, &total, &total, &px)));

    // count：净主动笔数
    let side: Vec<i8> = (0..n).map(|i| dead(ratio(bn[i], sn[i]), 0.1)).collect();
    let mag: Vec<f64> = (0..n).map(|i| bn[i] + sn[i]).collect();
    out.push(Some(make_seq(side, &mag, &total, &px)));

    // size：主买和主卖的平均每笔大小之比
    let xs: Vec<f64> = (0..n)
        .map(|i| if bn[i] > 0.0 && sn[i] > 0.0 { ((bvol[i] / bn[i]) / (svol[i] / sn[i])).ln() } else { 0.0 })
        .collect();
    let side: Vec<i8> = xs.iter().map(|&x| dead(x, 0.1)).collect();
    let mag: Vec<f64> = xs.iter().map(|x| x.abs()).collect();
    out.push(Some(make_seq(side, &mag, &total, &px)));

    // speed：这里配对的是“快”和“慢”，+1 表示笔数高于该时段中位数，-1 表示低于
    let cnt: Vec<f64> = (0..n).map(|i| bn[i] + sn[i]).collect();
    let med = median(cnt.clone());
    let side: Vec<i8> = cnt.iter().map(|&c| if c > med { 1 } else if c < med { -1 } else { 0 }).collect();
    let mag: Vec<f64> = cnt.iter().map(|&c| (c - med).abs()).collect();
    out.push(Some(make_seq(side, &mag, &total, &px)));

    // ordjump：买卖双方主动委托的编号间隔（取对数）谁更密
    let xs: Vec<f64> = (0..n)
        .map(|i| if bgn[i] > 0.0 && sgn[i] > 0.0 { sgap[i] / sgn[i] - bgap[i] / bgn[i] } else { 0.0 })
        .collect();
    let side: Vec<i8> = xs.iter().map(|&x| dead(x, 0.05)).collect();
    let mag: Vec<f64> = xs.iter().map(|x| x.abs()).collect();
    out.push(Some(make_seq(side, &mag, &total, &px)));

    // cancel：卖撤量减买撤量（卖撤偏多方信号）
    let cv: Vec<f64> = (0..n).map(|i| bc[i] + sc[i]).collect();
    let side: Vec<i8> = (0..n).map(|i| dead(ratio(sc[i], bc[i]), 0.1)).collect();
    out.push(Some(make_seq(side, &cv, &cv, &px)));

    // lob：盘口总买卖量不平衡的变化
    let mut imb_last = vec![f64::NAN; n];
    let mut first_imb = f64::NAN;
    for m in mk {
        if m.raw_sec < win.raw_lo || m.raw_sec > win.raw_hi { continue; }
        let d = m.total_bid_vol + m.total_ask_vol;
        if d <= 0.0 { continue; }
        let x = (m.total_bid_vol - m.total_ask_vol) / d;
        if !first_imb.is_finite() { first_imb = x; }
        imb_last[win.bar_idx(m.time_us)] = x;
    }
    let mut prev = first_imb;
    let mut deltas = vec![0f64; n];
    for i in 0..n {
        if imb_last[i].is_finite() {
            deltas[i] = imb_last[i] - prev;
            prev = imb_last[i];
        }
    }
    if first_imb.is_finite() {
        let side: Vec<i8> = deltas.iter().map(|&x| dead(x, 0.01)).collect();
        let mag: Vec<f64> = deltas.iter().map(|x| x.abs()).collect();
        out.push(Some(make_seq(side, &mag, &total, &px)));
    } else {
        out.push(None);
    }

    // ret：相邻两个时间片最后成交价的涨跌（收益率只作为多种输入之一）
    let mut prev_px = f32::NAN;
    let mut xs = vec![0f64; n];
    for i in 0..n {
        if last_px[i].is_finite() {
            if prev_px.is_finite() { xs[i] = (last_px[i] / prev_px - 1.0) as f64; }
            prev_px = last_px[i];
        }
    }
    let side: Vec<i8> = xs.iter().map(|&x| dead(x, 0.0)).collect();
    let mag: Vec<f64> = xs.iter().map(|x| x.abs()).collect();
    out.push(Some(make_seq(side, &mag, &total, &px)));

    // bigord：只取该时段最大的 n 笔主动委托，按时间排列
    let mut idx: Vec<u32> = (0..events.len() as u32).collect();
    idx.sort_unstable_by(|&a, &b| events[b as usize].vol.total_cmp(&events[a as usize].vol).then(a.cmp(&b)));
    idx.truncate(n);
    idx.sort_unstable();
    let mut side = Vec::with_capacity(idx.len());
    let mut vol = Vec::with_capacity(idx.len());
    let mut price = Vec::with_capacity(idx.len());
    for &i in &idx {
        let e = &events[i as usize];
        side.push(e.side);
        vol.push(e.vol);
        price.push((e.amt / e.vol) as f32);
    }
    out.push(Some(make_seq(side, &vol, &vol, &price)));

    // volbar：等成交量切成 n 段（成交量时钟），成交更快的段算强字母
    let v_total: f64 = trades.iter().map(|t| t.1).sum();
    let (mut bv, mut sv, mut a2, mut fst, mut lst) =
        (vec![0f64; n], vec![0f64; n], vec![0f64; n], vec![i64::MAX; n], vec![i64::MIN; n]);
    let mut cum = 0f64;
    for t in &trades {
        let b = ((cum / v_total * n as f64) as usize).min(n - 1);
        if t.0 > 0 { bv[b] += t.1; } else { sv[b] += t.1; }
        a2[b] += t.2;
        fst[b] = fst[b].min(t.3);
        lst[b] = lst[b].max(t.3);
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
    let side: Vec<i8> = (0..n).map(|i| dead(ratio(bv[i], sv[i]), 0.1)).collect();
    let mag: Vec<f64> = (0..n)
        .map(|i| if tot2[i] > 0.0 { 1.0 / (1.0 + (lst[i] - fst[i]) as f64 / 1e6) } else { 0.0 })
        .collect();
    out.push(Some(make_seq(side, &mag, &tot2, &px2)));

    out
}
