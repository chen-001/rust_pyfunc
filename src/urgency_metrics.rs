//! 迫切传播因子（urgency_propagation）—— 跨股票微观结构传播因子。
//!
//! # 核心思想
//! 读一天全市场逐笔成交 → 算每笔的买卖订单编号差值比例 (ask-bid)/(ask+bid) →
//! 全市场 q95/q5 分位定阈值 → 标记三版本迫切交易（sell/buy/both）作为种子事件 →
//! 每笔种子事件划前后5秒全市场窗口 → 算37个传播响应指标（前/后/差=111列）→
//! get_features_factors_rust_full 降维 → 三版本拼接 → per-stock 因子。
//!
//! 窗口内"迫切响应"统一用双向迫切 |ratio|>q95abs（含买含卖，方向指标才有 same/opp 区分）。
//!
//! # 37 指标（前/后/差 = 111 列）
//! 响应曲线(peak_r/lag/front_load/decay/time_ent/time_hhi/urg_time_mean/urg_time_std) +
//! 首次响应(first_lag/t_10_resp) + 方向(persistence/purity/dir_flips/first_same/first_opp) +
//! 传播广度(breadth/stock_hhi/stock_ent/n_eff/top1/self_share) + 超额迫切(excess/z/lift) +
//! 质量(urg_mass/mean_sev/max_sev/amplification) + 量价(vol_share/same_vol_share) +
//! 价格冲击(price_ret/price_impact) + 规模(log_ntrades/n_urg/urg_ratio)
//!
//! # 性能（20251231 实测）
//! 单进程 50线程 ~99s/天（CPU ~4930s）。框架 n_jobs=500(10进程×50线程) 约 6.8h/2500天。
//! 关键优化：both 连续 SoA 预提取（消除随机访问）、par_iter over 股票×版本（负载均衡）、
//! unsafe get_unchecked、价格冲击 per-stock 前缀和。
use crate::fast_csv_reader::read_trade_fast_inner;
use crate::features::get_features_factors_rust_full;
use ndarray::Array2;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::fs;

const WIN_US: i64 = 5_000_000; // 5 秒窗口
const EPS: f32 = 1e-9;
const NMETRICS: usize = 37;
const NCOLS: usize = NMETRICS * 3; // pre/post/diff = 111
pub const FEAT_PER_GROUP: usize = 21 * NCOLS + NCOLS * (NCOLS - 1) / 2; // 8436
pub const N_FACTORS: usize = 3 * FEAT_PER_GROUP; // 25308

/// 列出某天全市场股票代码（横截面枚举）。
pub fn list_codes(date: i64) -> Vec<String> {
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

/// 37 个原始指标名（× pre/post/diff = 111 列名）。
pub fn metric_names() -> Vec<String> {
    let base = [
        "log_ntrades", "n_urg", "urg_ratio", "urg_mass", "mean_sev", "max_sev",
        "vol_share", "same_mass", "opp_mass", "persistence", "purity", "peak_r",
        "peak_lag", "front_load", "decay", "time_ent", "time_hhi", "first_lag",
        "self_share", "breadth", "stock_hhi", "stock_ent", "n_eff", "top1_share",
        "urg_time_mean", "urg_time_std", "dir_flips", "first_same_lag", "first_opp_lag",
        "t_10_resp", "same_vol_share", "amplification",
        "excess", "z_score", "lift", "price_ret", "price_impact",
    ];
    let mut out = Vec::with_capacity(NCOLS);
    for tag in ["pre", "post", "diff"] {
        for b in base {
            out.push(format!("{b}_{tag}"));
        }
    }
    out
}

/// 单窗口指标（前/后通用）。遍历 both 连续 SoA [ua..ub)，cache-friendly。
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
    let mut m_dt_sum = 0.0f64;
    let mut m_dt2_sum = 0.0f64;
    let mut same_vol = 0.0f64;
    let mut prev_dir: i8 = 0;
    let mut n_flips = 0u32;
    let mut first_same_us = -1i64;
    let mut first_opp_us = -1i64;
    let mut t10_us = -1i64;
    *counter = counter.wrapping_add(1);
    let cnt = *counter;
    active.clear();
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
    // price_ret/price_impact 占位（由 stock_version_factors 用前缀和覆盖）
    out[35] = 0.0;
    out[36] = 0.0;
    out
}

/// 单股票单版本：所有种子 → 2D (n_seeds × NCOLS) → 降维 → 1D 因子。
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
        let dsgn = -(d_seed as f32);
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

/// 核心唯一真相源：读全市场 → 迫切标记 → 窗口指标 → 降维 → (codes, vals)。
pub fn compute_urgency_full(date: i64) -> std::io::Result<(Vec<String>, Vec<f32>)> {
    let codes = list_codes(date);
    // ① rayon 并行读全市场（过滤撤单）
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
    if n_stocks == 0 {
        return Ok((Vec::new(), Vec::new()));
    }
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

    // ② 全市场分位
    let qidx = |q: f64| ((n as f64 * q) as usize).min(n - 1);
    let mut rs = ratio.clone();
    rs.par_sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    let q95 = rs[qidx(0.95)];
    let q5 = rs[qidx(0.05)];
    for v in rs.iter_mut() { *v = v.abs(); }
    rs.par_sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    let q95_abs = rs[qidx(0.95)];

    // ③ 时间排序 → 有序 SoA
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
    let mut stock_pos: Vec<Vec<u32>> = vec![Vec::new(); n_stocks];
    for i in 0..n { stock_pos[cid_s[i] as usize].push(i as u32); }
    let stock_cum_pv: Vec<Vec<f64>> = (0..n_stocks).into_par_iter().map(|c| {
        let mut acc = 0.0; let mut v = Vec::with_capacity(stock_pos[c].len() + 1); v.push(0.0);
        for &p in &stock_pos[c] { acc += price_s[p as usize] as f64 * vol_s[p as usize] as f64; v.push(acc); }
        v
    }).collect();
    let stock_cum_v: Vec<Vec<f64>> = (0..n_stocks).into_par_iter().map(|c| {
        let mut acc = 0.0; let mut v = Vec::with_capacity(stock_pos[c].len() + 1); v.push(0.0);
        for &p in &stock_pos[c] { acc += vol_s[p as usize] as f64; v.push(acc); }
        v
    }).collect();
    drop(idx); drop(time_us); drop(ratio); drop(vol); drop(price); drop(dir); drop(cid);

    // ④ 三版本种子（按股票分组）+ both 连续 SoA
    let mut sell_by_stock: Vec<(Vec<u32>, Vec<i8>, Vec<u32>)> = vec![(Vec::new(), Vec::new(), Vec::new()); n_stocks];
    let mut buy_by_stock = sell_by_stock.clone();
    let mut both_by_stock = sell_by_stock.clone();
    let mut both_time = Vec::new();
    let mut both_abratio = Vec::new();
    let mut both_vol = Vec::new();
    let mut both_dir = Vec::new();
    let mut both_cid = Vec::new();
    for i in 0..n {
        let r = ratio_s[i];
        let ab = abratio_s[i];
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
    let p0 = both_time.len() as f32 / n as f32;
    let col_names = metric_names();

    // ⑤ 三版本×全股票（par_iter over 股票×版本，负载均衡）
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

    // ⑥ 聚合 + 扁平化输出
    let mut by_stock_out: Vec<[Option<Vec<f32>>; 3]> = vec![[None, None, None]; n_stocks];
    for (c, ver, v) in cv {
        by_stock_out[c][ver] = if v.is_empty() { None } else { Some(v) };
    }
    // feat_len 从第一个有效版本取（= FEAT_PER_GROUP）
    let feat_len = by_stock_out.iter().flatten()
        .next().and_then(|o| o.as_ref()).map(|v| v.len()).unwrap_or(FEAT_PER_GROUP);
    let mut out_codes = Vec::new();
    let mut out_vals = Vec::with_capacity(n_stocks * N_FACTORS);
    for c in 0..n_stocks {
        let arr = &by_stock_out[c];
        let valid = arr.iter().any(|o| o.as_ref().map_or(false, |v| v.iter().any(|x| x.is_finite())));
        if !valid { continue; }
        out_codes.push(code_of_cid[c].clone());
        for ver in 0..3 {
            match &arr[ver] {
                Some(v) if v.len() == feat_len => out_vals.extend(v),
                _ => out_vals.extend(std::iter::repeat(f32::NAN).take(feat_len)),
            }
        }
    }
    Ok((out_codes, out_vals))
}

/// 因子名（N_FACTORS 个：sell_/buy_/both_ 前缀 × 降维名）。
pub fn urgency_names() -> Vec<String> {
    let col_names = metric_names();
    let dummy = Array2::zeros((2, NCOLS));
    let (_, names) = get_features_factors_rust_full(&dummy.view(), &col_names, false);
    let mut out = Vec::with_capacity(N_FACTORS);
    for ver in ["sell", "buy", "both"] {
        for n in &names {
            out.push(format!("{ver}_{n}"));
        }
    }
    out
}

// ============================================================
// Python 调试入口
// ============================================================
#[pyfunction]
pub fn py_urgency(py: Python<'_>, date: i64) -> PyResult<(Vec<String>, Vec<f32>)> {
    compute_urgency_full(date)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("{e:?}")))
}

#[pyfunction]
pub fn py_urgency_names() -> Vec<String> {
    urgency_names()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_names_count() {
        let names = urgency_names();
        assert_eq!(names.len(), N_FACTORS);
    }
    #[test]
    fn test_metric_names_count() {
        assert_eq!(metric_names().len(), NCOLS);
    }
}
