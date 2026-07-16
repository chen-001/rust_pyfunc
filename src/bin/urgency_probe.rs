//! 第二阶段完整 probe（优化版）：迫切传播因子。
//!
//! 关键优化：把 both_urg 的属性预提取成连续 SoA（both_time/abratio/vol/dir/cid），
//! 遍历窗口时连续访问，避免 urg[u]→j→全局soa[j] 的随机跳转（每事件省3次随机内存访问）。
//! 其他优化：lag_bin 用 if-else 链；分位/排序并行。
use rust_pyfunc::fast_csv_reader::read_trade_fast_inner;
use rust_pyfunc::features::get_features_factors_rust_full;
use rayon::prelude::*;
use ndarray::Array2;
use std::fs;
use std::time::Instant;

const WIN_US: i64 = 5_000_000;
const EPS: f32 = 1e-9;
const NMETRICS: usize = 37;
const NCOLS: usize = NMETRICS * 3;

fn list_codes(date: i64) -> Vec<String> {
    let dir = format!("/ssd_data/stock/{date}/transaction");
    let mut set = std::collections::BTreeSet::new();
    if let Ok(entries) = fs::read_dir(&dir) {
        for e in entries.flatten() {
            let name = e.file_name().into_string().unwrap_or_default();
            if let Some(code) = name.split('_').next() {
                if code.chars().all(|c| c.is_ascii_digit()) {
                    set.insert(code.to_string());
                }
            }
        }
    }
    set.into_iter().collect()
}

#[inline]
fn lag_bin(dt: f32) -> usize {
    if dt <= 0.1 { 0 }
    else if dt <= 0.25 { 1 }
    else if dt <= 0.5 { 2 }
    else if dt <= 1.0 { 3 }
    else if dt <= 2.0 { 4 }
    else { 5 }
}

fn metric_names() -> Vec<String> {
    let base = [
        "log_ntrades", "n_urg", "urg_ratio", "urg_mass", "mean_sev", "max_sev",
        "vol_share", "same_mass", "opp_mass", "persistence", "purity", "peak_r",
        "peak_lag", "front_load", "decay", "time_ent", "time_hhi", "first_lag",
        "self_share", "breadth", "stock_hhi", "stock_ent", "n_eff", "top1_share",
        "urg_time_mean", "urg_time_std", "dir_flips", "first_same_lag", "first_opp_lag",
        "t_10_resp", "same_vol_share", "amplification",
        "excess", "z_score", "lift",
        "price_ret", "price_impact",
    ];
    let mut out = Vec::with_capacity(NCOLS);
    for tag in ["pre", "post", "diff"] {
        for b in base {
            out.push(format!("{b}_{tag}"));
        }
    }
    out
}

/// 单窗口指标。遍历 both 连续 SoA [ua..ub)，cache-friendly。
#[allow(clippy::too_many_arguments)]
fn window_metrics(
    is_post: bool, t_seed: i64, d_seed: i8, seed_cid: u32, m_seed: f32, p0: f32,
    pref_vol: &[f64], lo: usize, hi: usize,
    both_time: &[i64], both_abratio: &[f32], both_vol: &[f32], both_dir: &[i8], both_cid: &[u32],
    ua: usize, ub: usize,
    stamp: &mut [u32], mstock: &mut [f32], active: &mut Vec<u32>, counter: &mut u32,
) -> [f32; NMETRICS] {
    let mut out = [0.0f32; NMETRICS];
    let n_trades = (hi - lo) as f32;
    let sum_vol = (pref_vol[hi] - pref_vol[lo]) as f32;
    let mut umass = 0.0f32;
    let mut urg_vol = 0.0f32;
    let mut same = 0.0f32;
    let mut opp = 0.0f32;
    let mut selfm = 0.0f32;
    let mut maxsev = 0.0f32;
    let mut n_urg = 0u32;
    let mut first_lag_us = -1i64;
    let mut lag_m = [0.0f32; 6];
    let mut lag_n = [0u32; 6];
    *counter = counter.wrapping_add(1);
    let cnt = *counter;
    active.clear();
    let mut m_dt_sum = 0.0f64;
    let mut m_dt2_sum = 0.0f64;
    let mut same_vol = 0.0f64;
    let mut prev_dir: i8 = 0;
    let mut n_flips = 0u32;
    let mut first_same_us = -1i64;
    let mut first_opp_us = -1i64;
    let mut t10_us = -1i64;
    for u in ua..ub {
        let bt = unsafe { *both_time.get_unchecked(u) };
        let dt_us = if is_post { bt - t_seed } else { t_seed - bt };
        if dt_us <= 0 { continue; }
        let dt = dt_us as f32 / 1e6;
        let m = unsafe { *both_abratio.get_unchecked(u) };
        let vv = unsafe { *both_vol.get_unchecked(u) };
        let dd = unsafe { *both_dir.get_unchecked(u) };
        let c = unsafe { *both_cid.get_unchecked(u) };
        umass += m;
        urg_vol += vv;
        n_urg += 1;
        if dd == d_seed { same += m; same_vol += vv as f64; if first_same_us < 0 { first_same_us = dt_us; } }
        else { opp += m; if first_opp_us < 0 { first_opp_us = dt_us; } }
        if c == seed_cid { selfm += m; }
        if m > maxsev { maxsev = m; }
        if first_lag_us < 0 { first_lag_us = dt_us; }
        let bk = lag_bin(dt);
        if bk < 6 { lag_m[bk] += m; lag_n[bk] += 1; }
        m_dt_sum += m as f64 * dt as f64;
        m_dt2_sum += m as f64 * dt as f64 * dt as f64;
        if n_urg >= 2 && dd != prev_dir { n_flips += 1; }
        prev_dir = dd;
        if n_urg == 10 { t10_us = dt_us; }
        let ci = c as usize;
        if stamp[ci] != cnt {
            stamp[ci] = cnt;
            mstock[ci] = m;
            active.push(c);
        } else {
            mstock[ci] += m;
        }
    }
    let n_urgf = n_urg as f32;
    out[0] = if n_trades > 1.0 { n_trades.ln() } else { 0.0 };
    out[1] = n_urgf;
    out[2] = if n_trades > EPS { n_urgf / n_trades } else { 0.0 };
    out[3] = umass;
    out[4] = if n_urgf > EPS { umass / n_urgf } else { 0.0 };
    out[5] = maxsev;
    out[6] = if sum_vol > EPS { urg_vol / sum_vol } else { 0.0 };
    out[7] = same;
    out[8] = opp;
    let so = same + opp;
    out[9] = if so > EPS { (same - opp) / (so + EPS) } else { 0.0 };
    out[10] = if so > EPS { same.max(opp) / (so + EPS) } else { 0.0 };
    let mut peak_r = 0.0f32;
    let mut peak_lag = 0.0f32;
    for k in 0..6 {
        if lag_n[k] > 0 {
            let r = lag_m[k] / lag_n[k] as f32;
            if r > peak_r { peak_r = r; peak_lag = [0.1f32, 0.25, 0.5, 1.0, 2.0, 5.0][k]; }
        }
    }
    out[11] = peak_r;
    out[12] = peak_lag;
    let m01 = lag_m[0] + lag_m[1] + lag_m[2];
    let m15 = lag_m[3] + lag_m[4] + lag_m[5];
    let mtot = m01 + m15;
    out[13] = if mtot > EPS { m01 / (mtot + EPS) } else { 0.0 };
    out[14] = if m01 > EPS && m15 > EPS { (m01 / (m15 + EPS)).ln() }
        else if m01 > EPS { 8.0 } else if m15 > EPS { -8.0 } else { 0.0 };
    let mut ent = 0.0f32;
    let mut hhi = 0.0f32;
    if mtot > EPS {
        for k in 0..6 {
            if lag_m[k] > EPS {
                let p = lag_m[k] / mtot;
                ent -= p * p.ln();
                hhi += p * p;
            }
        }
        ent /= 6f32.ln();
    }
    out[15] = ent;
    out[16] = hhi;
    out[17] = if first_lag_us > 0 { first_lag_us as f32 / 1e6 } else { -1.0 };
    out[18] = if umass > EPS { selfm / (umass + EPS) } else { 0.0 };
    let breadth = active.len() as f32;
    let mut mstock_tot = 0.0f32;
    let mut top1 = 0.0f32;
    let mut entropy_raw = 0.0f32;
    let mut stock_hhi = 0.0f32;
    for &c in active.iter() {
        let m = mstock[c as usize];
        mstock_tot += m;
        if m > top1 { top1 = m; }
    }
    if mstock_tot > EPS {
        for &c in active.iter() {
            let p = mstock[c as usize] / mstock_tot;
            if p > EPS {
                entropy_raw -= p * p.ln();
                stock_hhi += p * p;
            }
        }
    }
    out[19] = breadth;
    out[20] = stock_hhi;
    out[21] = if breadth > 1.0 { entropy_raw / breadth.ln() } else { 0.0 };
    out[22] = entropy_raw.exp();
    out[23] = if mstock_tot > EPS { top1 / mstock_tot } else { 0.0 };
    out[24] = if umass > EPS { (m_dt_sum as f32) / umass } else { 0.0 };
    let mean_t = out[24] as f64;
    out[25] = if umass > EPS { (m_dt2_sum / umass as f64 - mean_t * mean_t).max(0.0).sqrt() as f32 } else { 0.0 };
    out[26] = n_flips as f32;
    out[27] = if first_same_us > 0 { first_same_us as f32 / 1e6 } else { -1.0 };
    out[28] = if first_opp_us > 0 { first_opp_us as f32 / 1e6 } else { -1.0 };
    out[29] = if t10_us > 0 { t10_us as f32 / 1e6 } else { -1.0 };
    out[30] = if urg_vol > EPS { same_vol as f32 / urg_vol } else { 0.0 };
    out[31] = if m_seed > EPS { umass / m_seed } else { 0.0 };
    let p0d = p0 as f64;
    let ntr = n_trades as f64;
    let excess = n_urgf as f64 - ntr * p0d;
    out[32] = excess as f32;
    out[33] = (excess / (ntr * p0d * (1.0 - p0d)).max(1e-9).sqrt()) as f32;
    out[34] = if p0d > 1e-9 { (n_urgf as f64 / (ntr + 1e-9) / p0d) as f32 } else { 0.0 };
    out
}

/// 单股票单版本：seeds 是该版本种子的有序索引，seed_dirs/seed_cids 对应方向与股票。
#[allow(clippy::too_many_arguments)]
fn stock_version_factors(
    seeds: &[u32], seed_dirs: &[i8], seed_cids: &[u32],
    time_s: &[i64], pref_vol: &[f64], abratio_s: &[f32], p0: f32,
    both_time: &[i64], both_abratio: &[f32], both_vol: &[f32], both_dir: &[i8], both_cid: &[u32],
    price_s: &[f32], stock_pos: &[Vec<u32>], stock_cum_pv: &[Vec<f64>], stock_cum_v: &[Vec<f64>],
    col_names: &[String], n_stocks: usize,
) -> Vec<f32> {
    if seeds.is_empty() { return Vec::new(); }
    let n = seeds.len();
    let mut flat = vec![0f32; n * NCOLS];
    let mut stamp = vec![0u32; n_stocks];
    let mut mstock = vec![0f32; n_stocks];
    let mut active: Vec<u32> = Vec::with_capacity(512);
    let mut counter: u32 = 0;
    for si in 0..n {
        let k = seeds[si] as usize;
        let t_seed = time_s[k];
        let d_seed = seed_dirs[si];
        let seed_cid = seed_cids[si];
        let m_seed = abratio_s[k];
        let lo = time_s.partition_point(|&t| t < t_seed - WIN_US);
        let hi = time_s.partition_point(|&t| t <= t_seed + WIN_US);
        let mid = time_s.partition_point(|&t| t < t_seed);
        let ua_lo = both_time.partition_point(|&t| t < t_seed - WIN_US);
        let ua_mid = both_time.partition_point(|&t| t < t_seed);
        let ua_hi = both_time.partition_point(|&t| t <= t_seed + WIN_US);
        let pre = window_metrics(false, t_seed, d_seed, seed_cid, m_seed, p0, pref_vol, lo, mid,
            both_time, both_abratio, both_vol, both_dir, both_cid, ua_lo, ua_mid,
            &mut stamp, &mut mstock, &mut active, &mut counter);
        let post = window_metrics(true, t_seed, d_seed, seed_cid, m_seed, p0, pref_vol, mid, hi,
            both_time, both_abratio, both_vol, both_dir, both_cid, ua_mid, ua_hi,
            &mut stamp, &mut mstock, &mut active, &mut counter);
        let base = si * NCOLS;
        for j in 0..NMETRICS {
            flat[base + j] = pre[j];
            flat[base + NMETRICS + j] = post[j];
            flat[base + NMETRICS * 2 + j] = post[j] - pre[j];
        }
        // 价格冲击（覆盖占位 35..36）：种子股前/后窗口 VWAP（前缀和 O(1)）
        let spx = price_s[k] as f64;
        let positions = &stock_pos[seed_cid as usize];
        let cpv = &stock_cum_pv[seed_cid as usize];
        let cv = &stock_cum_v[seed_cid as usize];
        let pp_lo = positions.partition_point(|&p| time_s[p as usize] < t_seed - WIN_US);
        let pp_mid = positions.partition_point(|&p| time_s[p as usize] < t_seed);
        let pp_hi = positions.partition_point(|&p| time_s[p as usize] <= t_seed + WIN_US);
        let vwap_pre = if pp_mid > pp_lo {
            let dv = cv[pp_mid] - cv[pp_lo];
            if dv > 1e-9 { (cpv[pp_mid] - cpv[pp_lo]) / dv } else { spx }
        } else { spx };
        let vwap_post = if pp_hi > pp_mid {
            let dv = cv[pp_hi] - cv[pp_mid];
            if dv > 1e-9 { (cpv[pp_hi] - cpv[pp_mid]) / dv } else { spx }
        } else { spx };
        let ret_pre = if spx > 0.0 { (vwap_pre - spx) / spx } else { 0.0 };
        let ret_post = if spx > 0.0 { (vwap_post - spx) / spx } else { 0.0 };
        let dsgn = -(d_seed as f32); // 主卖+1→-1, 主买-1→+1，使 impact>0 表示价格沿主动方移动
        let pre_impact = dsgn * ret_pre as f32;
        let post_impact = dsgn * ret_post as f32;
        flat[base + 35] = ret_pre as f32;
        flat[base + 36] = pre_impact;
        flat[base + NMETRICS + 35] = ret_post as f32;
        flat[base + NMETRICS + 36] = post_impact;
        flat[base + NMETRICS * 2 + 35] = (ret_post - ret_pre) as f32;
        flat[base + NMETRICS * 2 + 36] = post_impact - pre_impact;
    }
    let arr = Array2::from_shape_vec((n, NCOLS), flat)
        .unwrap_or_else(|_| Array2::zeros((0, NCOLS)));
    let (vals, _) = get_features_factors_rust_full(&arr.view(), col_names, false);
    vals
}

fn main() {
    let date = 20251231;
    let t0 = Instant::now();
    let codes = list_codes(date);
    eprintln!("{} codes", codes.len());

    let per_stock: Vec<(u32, String, Vec<_>)> = codes
        .par_iter()
        .enumerate()
        .filter_map(|(i, code)| {
            let mut trades = read_trade_fast_inner(code, date, false, true, usize::MAX).ok()?;
            trades.retain(|t| t.flag != 32);
            if trades.is_empty() { return None; }
            Some((i as u32, code.clone(), trades))
        })
        .collect();
    let n_stocks = per_stock.len();
    let mut code_of_cid: Vec<String> = Vec::with_capacity(n_stocks);
    let total: usize = per_stock.iter().map(|(_, _, t)| t.len()).sum();
    let mut time_us = Vec::with_capacity(total);
    let mut ratio = Vec::with_capacity(total);
    let mut vol = Vec::with_capacity(total);
    let mut price = Vec::with_capacity(total);
    let mut dir = Vec::with_capacity(total);
    let mut cid = Vec::with_capacity(total);
    for (i, code, trades) in &per_stock {
        code_of_cid.push(code.clone());
        for t in trades {
            let sum = (t.ask_order + t.bid_order) as f64;
            let r = ((t.ask_order - t.bid_order) as f64 / sum) as f32;
            time_us.push(t.time_us);
            ratio.push(r);
            vol.push(t.volume);
            price.push(t.price);
            dir.push(if t.flag == 83 { 1i8 } else { -1i8 });
            cid.push(*i);
        }
    }
    drop(per_stock);
    let n = time_us.len();
    eprintln!("read {n_stocks} stocks, {:.1}M trades in {:?}", n as f64 / 1e6, t0.elapsed());

    let t1 = Instant::now();
    let qidx = |q: f64| ((n as f64 * q) as usize).min(n - 1);
    let mut rs = ratio.clone();
    rs.par_sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    let q95 = rs[qidx(0.95)];
    let q5 = rs[qidx(0.05)];
    for v in rs.iter_mut() { *v = v.abs(); }
    rs.par_sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    let q95_abs = rs[qidx(0.95)];
    eprintln!("q95={q95:+.5} q5={q5:+.5} q95_abs={q95_abs:.5} in {:?}", t1.elapsed());

    let t2 = Instant::now();
    let mut idx: Vec<u32> = (0..n as u32).collect();
    idx.par_sort_unstable_by_key(|&i| time_us[i as usize]);
    let time_s: Vec<i64> = idx.iter().map(|&i| time_us[i as usize]).collect();
    let vol_s: Vec<f32> = idx.iter().map(|&i| vol[i as usize]).collect();
    let price_s: Vec<f32> = idx.iter().map(|&i| price[i as usize]).collect();
    let dir_s: Vec<i8> = idx.iter().map(|&i| dir[i as usize]).collect();
    let cid_s: Vec<u32> = idx.iter().map(|&i| cid[i as usize]).collect();
    let ratio_s: Vec<f32> = idx.iter().map(|&i| ratio[i as usize]).collect();
    let abratio_s: Vec<f32> = ratio_s.iter().map(|r| r.abs()).collect();
    let mut pref_vol = vec![0f64; n + 1];
    for i in 0..n { pref_vol[i + 1] = pref_vol[i] + vol_s[i] as f64; }
    // per-stock 在 time_s 中的位置索引（有序），供价格冲击查种子股窗口价格
    let mut stock_pos: Vec<Vec<u32>> = vec![Vec::new(); n_stocks];
    for i in 0..n { stock_pos[cid_s[i] as usize].push(i as u32); }
    // per-stock price·vol / vol 前缀和（价格冲击 O(1) 查询，避免遍历该股窗口的随机访问）
    let stock_cum_pv: Vec<Vec<f64>> = (0..n_stocks).into_par_iter().map(|c| {
        let mut acc = 0.0; let mut v = Vec::with_capacity(stock_pos[c].len()+1); v.push(0.0);
        for &p in &stock_pos[c] { acc += price_s[p as usize] as f64 * vol_s[p as usize] as f64; v.push(acc); }
        v
    }).collect();
    let stock_cum_v: Vec<Vec<f64>> = (0..n_stocks).into_par_iter().map(|c| {
        let mut acc = 0.0; let mut v = Vec::with_capacity(stock_pos[c].len()+1); v.push(0.0);
        for &p in &stock_pos[c] { acc += vol_s[p as usize] as f64; v.push(acc); }
        v
    }).collect();
    drop(idx);
    drop(time_us);
    drop(ratio);
    drop(vol);
    drop(price);
    drop(dir);
    drop(cid);
    eprintln!("sort+reindex in {:?}", t2.elapsed());

    // 三版本种子（按股票分组）+ both 连续 SoA（预提取，cache-friendly）
    let t3 = Instant::now();
    let mut sell_by_stock: Vec<(Vec<u32>, Vec<i8>, Vec<u32>)> = vec![(Vec::new(),Vec::new(),Vec::new()); n_stocks];
    let mut buy_by_stock = sell_by_stock.clone();
    let mut both_by_stock = sell_by_stock.clone();
    let mut both_time = Vec::new();
    let mut both_abratio = Vec::new();
    let mut both_vol = Vec::new();
    let mut both_dir = Vec::new();
    let mut both_cid = Vec::new();
    for i in 0..n {
        let r = ratio_s[i];
        let ab = r.abs();
        let c = cid_s[i] as usize;
        let d = dir_s[i];
        if r > q95 { sell_by_stock[c].0.push(i as u32); sell_by_stock[c].1.push(d); sell_by_stock[c].2.push(c as u32); }
        if r < q5 { buy_by_stock[c].0.push(i as u32); buy_by_stock[c].1.push(d); buy_by_stock[c].2.push(c as u32); }
        if ab > q95_abs {
            both_by_stock[c].0.push(i as u32); both_by_stock[c].1.push(d); both_by_stock[c].2.push(c as u32);
            both_time.push(time_s[i]); both_abratio.push(ab); both_vol.push(vol_s[i]);
            both_dir.push(d); both_cid.push(c as u32);
        }
    }
    eprintln!("urg grouped in {:?}: sell/buy/both = {}/{}/{}",
        t3.elapsed(),
        sell_by_stock.iter().map(|v| v.0.len()).sum::<usize>(),
        buy_by_stock.iter().map(|v| v.0.len()).sum::<usize>(),
        both_time.len());
    let p0 = both_time.len() as f32 / n as f32;

    let col_names = metric_names();
    let tid = code_of_cid.iter().position(|c| c == "000001").unwrap();
    // 验证打印
    {
        let (s, sd, sc) = &sell_by_stock[tid];
        if !s.is_empty() {
            let mut stamp = vec![0u32; n_stocks]; let mut mstock = vec![0f32; n_stocks];
            let mut active = Vec::new(); let mut counter = 0u32;
            let t_seed = time_s[s[0] as usize];
            let mid = time_s.partition_point(|&t| t < t_seed);
            let hi = time_s.partition_point(|&t| t <= t_seed + WIN_US);
            let ua_mid = both_time.partition_point(|&t| t < t_seed);
            let ua_hi = both_time.partition_point(|&t| t <= t_seed + WIN_US);
            let post = window_metrics(true, t_seed, sd[0], sc[0], abratio_s[s[0] as usize], p0, &pref_vol, mid, hi,
                &both_time, &both_abratio, &both_vol, &both_dir, &both_cid, ua_mid, ua_hi,
                &mut stamp, &mut mstock, &mut active, &mut counter);
            eprintln!("验证 000001 seed0 后窗口: breadth={:.0} hhi={:.4} ent={:.4} n_eff={:.1} top1={:.4} self={:.4} pers={:.4} first={:.4}s",
                post[19], post[20], post[21], post[22], post[23], post[18], post[9], post[17]);
            let seed_px = price_s[s[0] as usize];
            let positions = &stock_pos[sc[0] as usize];
            let pp_mid = positions.partition_point(|&p| time_s[p as usize] < t_seed);
            let pp_hi = positions.partition_point(|&p| time_s[p as usize] <= t_seed + WIN_US);
            let (mut amt, mut vol) = (0f64, 0f64);
            for pi in pp_mid..pp_hi {
                let pp = positions[pi] as usize;
                amt += price_s[pp] as f64 * vol_s[pp] as f64;
                vol += vol_s[pp] as f64;
            }
            let vwap_post = if vol > 0.0 { amt / vol } else { seed_px as f64 };
            let ret = if seed_px > 0.0 { (vwap_post - seed_px as f64) / seed_px as f64 } else { 0.0 };
            eprintln!("  价格冲击: seed_px={:.3} vwap_post={:.3} ret={:+.5} impact={:+.5} (excess={:.0} z={:.2} lift={:.3})",
                seed_px, vwap_post, ret, -(sd[0] as f32) * ret as f32, post[32], post[33], post[34]);
        }
    }
    let probe_vals = stock_version_factors(&sell_by_stock[tid].0, &sell_by_stock[tid].1, &sell_by_stock[tid].2,
        &time_s, &pref_vol, &abratio_s, p0, &both_time, &both_abratio, &both_vol, &both_dir, &both_cid,
        &price_s, &stock_pos, &stock_cum_pv, &stock_cum_v, &col_names, n_stocks);
    let feat_len = probe_vals.len();
    eprintln!("feat_len = {feat_len} (NCOLS={NCOLS}), OUT_LEN = {}", feat_len * 3);

    let t4 = Instant::now();
    let cv: Vec<(usize, usize, Vec<f32>)> = (0..n_stocks)
        .flat_map(|c| [(c, 0usize), (c, 1), (c, 2)])
        .collect::<Vec<_>>()
        .into_par_iter()
        .map(|(c, ver)| {
            let by = match ver { 0 => &sell_by_stock[c], 1 => &buy_by_stock[c], _ => &both_by_stock[c] };
            let v = stock_version_factors(&by.0, &by.1, &by.2,
                &time_s, &pref_vol, &abratio_s, p0, &both_time, &both_abratio, &both_vol, &both_dir, &both_cid,
                &price_s, &stock_pos, &stock_cum_pv, &stock_cum_v, &col_names, n_stocks);
            (c, ver, v)
        })
        .collect();
    eprintln!("全市场三版本×降维 in {:?} ({} tasks)", t4.elapsed(), n_stocks * 3);
    // 聚合按 c
    let mut by_stock_out: Vec<[Option<Vec<f32>>; 3]> = vec![[None, None, None]; n_stocks];
    for (c, ver, v) in cv {
        by_stock_out[c][ver] = if v.is_empty() { None } else { Some(v) };
    }
    let valid = by_stock_out.iter().filter(|arr| {
        arr.iter().any(|o| o.as_ref().map_or(false, |v| v.iter().any(|x| x.is_finite())))
    }).count();
    eprintln!("有效股票 {valid}, 总耗时 {:?}, vals_len={}", t0.elapsed(), valid * feat_len * 3);
}
