//! 数据预处理：盘口对齐 + 特征构造 + 4时段筛选（移植 Python data_prep）
use crate::fast_csv_reader::{TradeRecord, MarketRecord};
use crate::Window;

const SEG_LATE30_START: f64 = 12600.0;
const SEG_LATE30_END: f64 = 14220.0;

fn safe_div(a: f64, b: f64) -> f64 {
    if b.abs() < 1e-12 { 0.0 } else { a / b }
}

fn rolling_mean(x: &[f64], k: usize) -> Vec<f64> {
    let n = x.len();
    let mut csum = vec![0f64; n + 1];
    for i in 0..n { csum[i + 1] = csum[i] + x[i]; }
    let mut out = vec![0f64; n];
    for i in 0..n {
        let lo = if i + 1 >= k { i + 1 - k } else { 0 };
        out[i] = (csum[i + 1] - csum[lo]) / (i - lo + 1) as f64;
    }
    out
}

fn rolling_std(x: &[f64], k: usize) -> Vec<f64> {
    // pandas rolling std, ddof=1, min_periods=1（窗口<2 → 0）
    let n = x.len();
    let mut out = vec![0f64; n];
    for i in 0..n {
        let lo = if i + 1 >= k { i + 1 - k } else { 0 };
        let cnt = i - lo + 1;
        if cnt < 2 { out[i] = 0.0; continue; }
        let mean = (lo..=i).map(|j| x[j]).sum::<f64>() / cnt as f64;
        let var = (lo..=i).map(|j| (x[j] - mean).powi(2)).sum::<f64>() / (cnt - 1) as f64;
        out[i] = var.sqrt();
    }
    out
}

fn count_within(time_sec: &[f64], window: f64) -> Vec<f64> {
    let n = time_sec.len();
    let mut out = vec![0f64; n];
    for i in 0..n {
        let lo = time_sec[i] - window;
        let j = time_sec.partition_point(|&t| t < lo);
        out[i] = (i - j + 1) as f64;
    }
    out
}

fn sameprice_runlen(price: &[f64]) -> Vec<f64> {
    let n = price.len();
    let mut out = vec![0f64; n];
    if n == 0 { return out; }
    let mut grp = vec![0i64; n];
    let mut g = 0i64;
    grp[0] = 0;
    for i in 1..n {
        if price[i] != price[i - 1] { g += 1; }
        grp[i] = g;
    }
    let maxg = (g + 1) as usize;
    let mut counts = vec![0f64; maxg];
    for &gg in &grp { counts[gg as usize] += 1.0; }
    for i in 0..n { out[i] = counts[grp[i] as usize]; }
    out
}

fn depth_slope(prc: &[f64], vol: &[f64], mid: &[f64], is_bid: bool) -> Vec<f64> {
    let n = prc.len() / 5; // prc 是扁平 n*5
    let mut out = vec![0f64; n];
    for i in 0..n {
        let p = &prc[i * 5..i * 5 + 5];
        let v = &vol[i * 5..i * 5 + 5];
        let dist: Vec<f64> = if is_bid {
            (0..5).map(|j| mid[i] - p[j]).collect()
        } else {
            (0..5).map(|j| p[j] - mid[i]).collect()
        };
        let mut cumv = [0f64; 5];
        cumv[0] = v[0];
        for j in 1..5 { cumv[j] = cumv[j - 1] + v[j]; }
        // mask dist>0
        let xs: Vec<f64> = (0..5).filter(|&j| dist[j] > 0.0).map(|j| dist[j]).collect();
        let ys: Vec<f64> = (0..5).filter(|&j| dist[j] > 0.0).map(|j| cumv[j]).collect();
        if xs.len() >= 2 {
            let mx = xs.iter().sum::<f64>() / xs.len() as f64;
            let my = ys.iter().sum::<f64>() / ys.len() as f64;
            let num: f64 = xs.iter().zip(ys.iter()).map(|(x, y)| (x - mx) * (y - my)).sum();
            let den: f64 = xs.iter().map(|x| (x - mx).powi(2)).sum();
            if den > 1e-12 { out[i] = num / den; }
        }
    }
    out
}

pub fn align_and_build(trade: &[TradeRecord], market: &[MarketRecord]) -> Window {
    let n = trade.len();
    // 按 time_us 稳定排序（匹配 Python read_trade 的 exchtime 排序）
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| trade[a].time_us.cmp(&trade[b].time_us).then(trade[b].index.cmp(&trade[a].index)));
    let trade: Vec<TradeRecord> = order.iter().map(|&i| trade[i]).collect();
    let trade = trade.as_slice();
    // market 按 time_us 有序（CSV 时间序）
    let m_times: Vec<i64> = market.iter().map(|m| m.time_us).collect();

    // 对齐盘口
    let mut bid1 = vec![0f64; n];
    let mut ask1 = vec![0f64; n];
    let mut bvol1 = vec![0f64; n];
    let mut avol1 = vec![0f64; n];
    let mut bvol5 = vec![0f64; n];
    let mut avol5 = vec![0f64; n];
    let mut bvol10 = vec![0f64; n];
    let mut avol10 = vec![0f64; n];
    let mut bp5 = vec![0f64; n * 5];
    let mut ap5 = vec![0f64; n * 5];
    let mut bv5 = vec![0f64; n * 5];
    let mut av5 = vec![0f64; n * 5];
    for i in 0..n {
        let tu = trade[i].time_us;
        let idx = m_times.partition_point(|&t| t <= tu);
        let idx = if idx > 0 { idx - 1 } else { 0 };
        let m = &market[idx];
        bid1[i] = m.bid_prcs[0] as f64;
        ask1[i] = m.ask_prcs[0] as f64;
        bvol1[i] = m.bid_vols[0] as f64;
        avol1[i] = m.ask_vols[0] as f64;
        let (b5, a5) = (0..5).fold((0f64, 0f64), |(b, a), j| (b + m.bid_vols[j] as f64, a + m.ask_vols[j] as f64));
        bvol5[i] = b5; avol5[i] = a5;
        let (b10, a10) = (0..10).fold((0f64, 0f64), |(b, a), j| (b + m.bid_vols[j] as f64, a + m.ask_vols[j] as f64));
        bvol10[i] = b10; avol10[i] = a10;
        for j in 0..5 {
            bp5[i * 5 + j] = m.bid_prcs[j] as f64;
            ap5[i * 5 + j] = m.ask_prcs[j] as f64;
            bv5[i * 5 + j] = m.bid_vols[j] as f64;
            av5[i * 5 + j] = m.ask_vols[j] as f64;
        }
    }
    let mid: Vec<f64> = (0..n).map(|i| (bid1[i] + ask1[i]) / 2.0).collect();
    let spread: Vec<f64> = (0..n).map(|i| ask1[i] - bid1[i]).collect();
    let imb1: Vec<f64> = (0..n).map(|i| safe_div(bvol1[i] - avol1[i], bvol1[i] + avol1[i])).collect();
    let imb5: Vec<f64> = (0..n).map(|i| safe_div(bvol5[i] - avol5[i], bvol5[i] + avol5[i])).collect();
    let imb10: Vec<f64> = (0..n).map(|i| safe_div(bvol10[i] - avol10[i], bvol10[i] + avol10[i])).collect();
    let microprice: Vec<f64> = (0..n).map(|i| safe_div(bid1[i] * avol1[i] + ask1[i] * bvol1[i], bvol1[i] + avol1[i])).collect();
    let micro_off: Vec<f64> = (0..n).map(|i| microprice[i] - mid[i]).collect();
    let slope_bid = depth_slope(&bp5, &bv5, &mid, true);
    let slope_ask = depth_slope(&ap5, &av5, &mid, false);

    // 主动方向
    let price: Vec<f64> = trade.iter().map(|t| t.price as f64).collect();
    let flag: Vec<i32> = trade.iter().map(|t| t.flag).collect();
    let mut side = vec![0i8; n];
    for i in 0..n {
        if flag[i] == 66 { side[i] = 1; }
        else if flag[i] == 83 { side[i] = -1; }
    }
    // 未知按中间价
    let mut last = 0i8;
    for i in 0..n {
        if side[i] != 0 { last = side[i]; continue; }
        let prev_price = if i > 0 { price[i - 1] } else { f64::NAN };
        let cond_buy = price[i] > mid[i] || (price[i] == mid[i] && price[i] > prev_price);
        let cond_sell = price[i] < mid[i] || (price[i] == mid[i] && price[i] < prev_price);
        if cond_buy { side[i] = 1; last = 1; }
        else if cond_sell { side[i] = -1; last = -1; }
        else { side[i] = last; }
    }
    let active_order: Vec<i64> = (0..n).map(|i| if side[i] == 1 { trade[i].bid_order } else if side[i] == -1 { trade[i].ask_order } else { -1 }).collect();
    let passive_order: Vec<i64> = (0..n).map(|i| if side[i] == 1 { trade[i].ask_order } else if side[i] == -1 { trade[i].bid_order } else { -1 }).collect();

    // 滚动特征
    let signed_vol: Vec<f64> = (0..n).map(|i| trade[i].volume as f64 * side[i] as f64).collect();
    let signed_imb10 = rolling_mean(&signed_vol, 10);
    let signed_imb50 = rolling_mean(&signed_vol, 50);
    let time_sec: Vec<f64> = (0..n).map(|i| (trade[i].time_us - trade[0].time_us) as f64 / 1_000_000.0).collect();
    let mut dt_prev = vec![0f64; n];
    for i in 1..n { dt_prev[i] = time_sec[i] - time_sec[i - 1]; }
    let a_dt_mean8 = rolling_mean(&dt_prev, 8);
    let a_dt_mean32 = rolling_mean(&dt_prev, 32);
    let rs_dt32 = rolling_std(&dt_prev, 32);
    let a_dt_cv32: Vec<f64> = (0..n).map(|i| safe_div(rs_dt32[i], a_dt_mean32[i])).collect();
    let a_cnt1s = count_within(&time_sec, 1.0);
    let a_cnt5s = count_within(&time_sec, 5.0);
    let mut mid_ret = vec![0f64; n];
    for i in 1..n { mid_ret[i] = mid[i] - mid[i - 1]; }
    let mid_ret10 = rolling_mean(&mid_ret, 10);
    let mid_vol50 = rolling_std(&mid_ret, 50);

    // 队列特征
    let mut q_bid1_chg = vec![0f64; n];
    let mut q_ask1_chg = vec![0f64; n];
    for i in 1..n { q_bid1_chg[i] = bvol1[i] - bvol1[i - 1]; q_ask1_chg[i] = avol1[i] - avol1[i - 1]; }
    let q_bid1_consume: Vec<f64> = (0..n).map(|i| {
        let prev = if i > 0 { bvol1[i - 1] } else { bvol1[i] };
        safe_div((prev - bvol1[i]).max(0.0), prev)
    }).collect();
    let q_ask1_consume: Vec<f64> = (0..n).map(|i| {
        let prev = if i > 0 { avol1[i - 1] } else { avol1[i] };
        safe_div((prev - avol1[i]).max(0.0), prev)
    }).collect();
    let q_sameprice = sameprice_runlen(&price);

    // 组装特征矩阵（顺序匹配 Python BOOK/QUEUE/ARRIVAL_FEATS）
    let mut book_feats = Vec::with_capacity(n * 14);
    for i in 0..n {
        book_feats.push(spread[i]); book_feats.push(imb1[i]); book_feats.push(imb5[i]); book_feats.push(imb10[i]);
        book_feats.push(bvol5[i]); book_feats.push(avol5[i]); book_feats.push(micro_off[i]);
        book_feats.push(slope_bid[i]); book_feats.push(slope_ask[i]);
        book_feats.push(signed_imb10[i]); book_feats.push(signed_imb50[i]);
        book_feats.push(dt_prev[i]); book_feats.push(mid_ret10[i]); book_feats.push(mid_vol50[i]);
    }
    let mut queue_feats = Vec::with_capacity(n * 10);
    for i in 0..n {
        queue_feats.push(bvol1[i]); queue_feats.push(avol1[i]);
        queue_feats.push(q_bid1_chg[i]); queue_feats.push(q_ask1_chg[i]);
        queue_feats.push(q_bid1_consume[i]); queue_feats.push(q_ask1_consume[i]);
        queue_feats.push(imb1[i]); queue_feats.push(spread[i]);
        queue_feats.push(side[i] as f64); queue_feats.push(q_sameprice[i]);
    }
    let mut arrival_feats = Vec::with_capacity(n * 6);
    for i in 0..n {
        arrival_feats.push(dt_prev[i]); arrival_feats.push(a_dt_mean8[i]);
        arrival_feats.push(a_dt_mean32[i]); arrival_feats.push(a_dt_cv32[i]);
        arrival_feats.push(a_cnt1s[i]); arrival_feats.push(a_cnt5s[i]);
    }

    Window {
        n,
        volume: trade.iter().map(|t| t.volume as i64).collect(),
        price, flag,
        bid_order: trade.iter().map(|t| t.bid_order).collect(),
        ask_order: trade.iter().map(|t| t.ask_order).collect(),
        time_sec,
        active_side: side,
        active_order, passive_order,
        book_feats, queue_feats, arrival_feats,
    }
}

fn percentile(x: &[f64], q: f64) -> f64 {
    let mut s = x.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = s.len();
    if n == 0 { return 0.0; }
    let pos = (n - 1) as f64 * q / 100.0;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    if lo == hi { s[lo] } else { s[lo] + (pos - lo as f64) * (s[hi] - s[lo]) }
}

pub fn select_segment(win: &Window, seg: usize) -> Window {
    let mut idx: Vec<usize> = vec![];
    let vols: Vec<f64> = win.volume.iter().map(|&v| v as f64).collect();
    for i in 0..win.n {
        let is_active = win.active_side[i] != 0;
        let is_buy = win.active_side[i] == 1;
        match seg {
            1 => { if is_active { idx.push(i); } }
            2 => { if is_buy { idx.push(i); } }
            3 => { if is_active { idx.push(i); } }
            4 => {
                let ts = win.time_sec[i];
                if is_buy && ts >= SEG_LATE30_START && ts <= SEG_LATE30_END { idx.push(i); }
            }
            _ => {}
        }
    }
    // 按成交量筛选
    let sub_vols: Vec<f64> = idx.iter().map(|&i| vols[i]).collect();
    let (lo_thr, hi_thr) = match seg {
        1 | 2 => (percentile(&sub_vols, 40.0), f64::INFINITY),
        3 => (percentile(&sub_vols, 40.0), percentile(&sub_vols, 90.0)),
        _ => (f64::NEG_INFINITY, f64::INFINITY),
    };
    if seg <= 3 {
        let mx = sub_vols.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    }
    let filtered: Vec<usize> = idx.into_iter().filter(|&i| {
        if seg == 3 { vols[i] > lo_thr && vols[i] <= hi_thr }
        else if seg == 4 { true }
        else { vols[i] <= lo_thr }
    }).collect();
    slice_window(win, &filtered)
}

/// 固定时间跨度滑窗（方案 A）：窗口 [start, start+tw)，步长 step，返回每个窗口（N 自适应）。
/// 窗口时间戳 = 窗口末笔时间。N < n_min 的窗口丢弃（树无意义）。
pub fn slide_time_windows(seg: &Window, tw: f64, step: f64, n_min: usize, max_windows: usize) -> Vec<(f64, Window)> {
    if seg.n == 0 { return vec![]; }
    let t0 = seg.time_sec[0];
    let t1 = seg.time_sec[seg.n - 1];
    let mut result = vec![];
    let mut start = t0;
    while start + tw <= t1 + 1e-6 {
        let end = start + tw;
        let mut idx = vec![];
        for i in 0..seg.n {
            if seg.time_sec[i] >= start && seg.time_sec[i] < end { idx.push(i); }
            else if seg.time_sec[i] >= end { break; }
        }
        if idx.len() >= n_min {
            let win = slice_window(seg, &idx);
            let ts = win.time_sec[win.n - 1];
            result.push((ts, win));
        }
        start += step;
        if result.len() >= max_windows { break; }
    }
    result
}

pub fn slice_window(win: &Window, idx: &[usize]) -> Window {
    let n = idx.len();
    let get_f = |arr: &[f64], nf: usize| -> Vec<f64> {
        let mut out = Vec::with_capacity(n * nf);
        for &i in idx { out.extend_from_slice(&arr[i * nf..(i + 1) * nf]); }
        out
    };
    Window {
        n,
        volume: idx.iter().map(|&i| win.volume[i]).collect(),
        price: idx.iter().map(|&i| win.price[i]).collect(),
        flag: idx.iter().map(|&i| win.flag[i]).collect(),
        bid_order: idx.iter().map(|&i| win.bid_order[i]).collect(),
        ask_order: idx.iter().map(|&i| win.ask_order[i]).collect(),
        time_sec: idx.iter().map(|&i| win.time_sec[i]).collect(),
        active_side: idx.iter().map(|&i| win.active_side[i]).collect(),
        active_order: idx.iter().map(|&i| win.active_order[i]).collect(),
        passive_order: idx.iter().map(|&i| win.passive_order[i]).collect(),
        book_feats: get_f(&win.book_feats, 14),
        queue_feats: get_f(&win.queue_feats, 10),
        arrival_feats: get_f(&win.arrival_feats, 6),
    }
}
