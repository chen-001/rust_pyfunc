//! 一呼百应「行业头部宇宙」因子层面测试（sandbox）。
//!
//! # 做什么
//! 把 Multica COR-39 评论里的拓展想法「标的局限到行业头部（31 申万一级 × 总市值前 10 = 310 只）」
//! 完整实现为**每股每日耦合因子**，并做 IC 测试：
//!   ① 事件层：移植 yhyb_metrics.rs 核心检测逻辑，6 种代表性事件
//!      （big_buy_m 中单买 / big_buy_s 主动小单买 / big_sell_l 大单卖 /
//!        sweep_buy 扫单买 / ice 冰山同价 / jump 价格跳变），每事件每日 500 笔上限；
//!   ② 配对层：310 只全配对（310×309/2 ≈ 4.8 万对），双向最近邻：
//!      fwd = A 事件之后 B 的下一事件（B 响应 A），bwd = A 事件之前 B 的上一事件（A 响应 B）；
//!      hit = 距离 ≤ 60s 占比（分母 = 有匹配的 A 事件数，正式版口径）；med 为精确中位距离（秒）；
//!   ③ 零模型（均匀解析式，见 yhyb_metrics.rs 注释）：B 的 m 个事件在 [LO,HI] 均匀分布时，
//!      A 事件 t 出发的下一次 B 事件距离分布 P(d>y)=((x−y)/x)^m；
//!      fwd 取 x = HI−t、bwd 取 x = t−LO（比正式版两方向共用 x=HI−t 更精确）；
//!      null_med = geo_mean(x)·c(m)（c(m)=1−2^(−1/m)）；
//!      null_hit = mean over 匹配 A 事件 of 1 − clamp((x−T)/x,0,1)^m（逐点精确）；
//!      rmed = med/null_med（<1 净更快）、rhit = hit/null_hit（>1 净更强）；
//!   ④ 配对强度（旧 agent 口径）：rhit_score = 0.5·(rhit_f+rhit_b)，
//!      comp = 0.5·(null_med_f/med_f + null_med_b/med_b) = 0.5·(1/rmed_f + 1/rmed_b)；
//!      pair_strength = 事件内跨对所有 rhit_score 与 comp 的百分位均值（[0,1]）；
//!      特异性：spec_ab = strength_ab − 0.5·(marg_A + marg_B)，
//!      marg = 该股对宇宙内其他所有股票的平均 strength；
//!   ⑤ 因子聚合（每股·每事件）：rate / strength_raw_mean（不过滤原始均值）/
//!      top5·top10（特异性 top-k 同伴 strength 均值）/ spec_positive_frac /
//!      same_ind_strong_ratio（同行业强耦合占比）/ rhit·rmed 双向均值 / hit_f_mean、med_b_mean / n_partners。
//!
//! 构建: cd sandbox_yhyb_topcap && cargo build --release
//! 运行: ./target/release/yhyb_topcap_sandbox 20240104 > out_20240104.json
//! 日志走 stderr，结果走 stdout。
mod fast_csv_reader;
use fast_csv_reader::read_trade_fast;
use rayon::prelude::*;
use serde::Serialize;
use std::io::{BufRead, BufReader};

// ------------------------- 常量 -------------------------
const N_EV: usize = 6;
const EV_NAMES: [&str; N_EV] = [
    "big_buy_m",
    "big_buy_s",
    "big_sell_l",
    "sweep_buy",
    "ice",
    "jump",
];
const CAP_EV: usize = 500;
const LO_S: i64 = 5400; // 09:30
const HI_S: i64 = 19620; // 14:57
const T_US: u64 = 60 * 1_000_000; // 60s 命中窗口
const NF: usize = 16;

const FACTOR_NAMES: [&str; NF] = [
    "rate",
    "strength_raw_mean",
    "top5_strength",
    "top10_strength",
    "top5_spec_mean",
    "spec_positive_frac",
    "same_ind_strong_ratio",
    "rhit_f_mean",
    "rhit_b_mean",
    "rmed_f_mean",
    "rmed_b_mean",
    "hit_f_mean",
    "med_b_mean",
    "n_partners",
    "same_ind_top5_strength",
    "cross_ind_top5_strength",
];

#[derive(Clone, Default)]
struct EvStream {
    t: Vec<i64>,
}

fn push_capped(out: &mut EvStream, cand: &mut Vec<(i64, f64)>) {
    if cand.len() > CAP_EV {
        cand.select_nth_unstable_by(CAP_EV, |a, b| b.1.total_cmp(&a.1));
        cand.truncate(CAP_EV);
    }
    cand.sort_unstable_by(|a, b| a.0.cmp(&b.0));
    for &(tt, _) in cand.iter() {
        out.t.push(tt);
    }
}

/// 6 种事件检测（移植 yhyb_metrics.rs::detect_trade_cols，thr=NaN 内部阈值路径）。
fn detect6(t: &[i64], p: &[f64], amt: &[f64], f: &[i32]) -> [EvStream; N_EV] {
    let mut out: [EvStream; N_EV] = Default::default();
    let n = t.len();
    if n < 2 {
        return out;
    }
    let mut amt_sorted = amt.to_vec();
    amt_sorted.sort_unstable_by(|a, b| a.total_cmp(b));
    let inner_m = amt_sorted[((n as f64) * 0.40) as usize];
    let inner_l = amt_sorted[((n as f64) * 0.90) as usize];
    let is_l: Vec<bool> = amt.iter().map(|&x| x >= inner_l).collect();
    let is_m: Vec<bool> = amt.iter().map(|&x| x >= inner_m && x < inner_l).collect();
    let is_s: Vec<bool> = amt.iter().map(|&x| x < inner_m).collect();

    // big_buy_m / big_buy_s / big_sell_l
    let mut cand: [Vec<(i64, f64)>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    for i in 0..n {
        if f[i] != 66 && f[i] != 83 {
            continue;
        }
        if is_m[i] && f[i] == 66 {
            cand[0].push((t[i], amt[i]));
        }
        if is_s[i] && f[i] == 66 {
            cand[1].push((t[i], amt[i]));
        }
        if is_l[i] && f[i] == 83 {
            cand[2].push((t[i], amt[i]));
        }
    }
    for (i, c) in cand.iter_mut().enumerate() {
        push_capped(&mut out[i], c);
    }

    // sweep_buy：1200s 窗口内同向大单（flag 66 且 is_l）≥ 2
    {
        let sw_us = 1200 * 1_000_000i64;
        let mut tb: Vec<(i64, f64)> = Vec::new();
        for i in 0..n {
            if is_l[i] && f[i] == 66 {
                tb.push((t[i], amt[i]));
            }
        }
        if tb.len() >= 2 {
            let mut cand_s: Vec<(i64, f64)> = Vec::new();
            let mut j = 0usize;
            for i in 0..tb.len() {
                while tb[j].0 <= tb[i].0 - sw_us {
                    j += 1;
                }
                if i - j + 1 >= 2 {
                    cand_s.push((tb[i].0, tb[i].1));
                }
            }
            push_capped(&mut out[3], &mut cand_s);
        }
    }

    // ice：180s 窗口同价位大单（is_l）≥ 2
    {
        let ice_us = 180 * 1_000_000i64;
        let mut idx: Vec<usize> = (0..n).filter(|&i| is_l[i]).collect();
        idx.sort_unstable_by(|&a, &b| p[a].total_cmp(&p[b]).then(t[a].cmp(&t[b])));
        let mut ice_ev: Vec<(i64, f64)> = Vec::new();
        let mut s = 0usize;
        while s < idx.len() {
            let mut e = s + 1;
            while e < idx.len() && p[idx[e]] == p[idx[s]] {
                e += 1;
            }
            if e - s >= 2 {
                let mut j = s;
                for i in s..e {
                    while t[idx[j]] <= t[idx[i]] - ice_us {
                        j += 1;
                    }
                    if i - j + 1 >= 2 {
                        ice_ev.push((t[idx[i]], amt[idx[i]]));
                    }
                }
            }
            s = e;
        }
        push_capped(&mut out[4], &mut ice_ev);
    }

    // jump：|Δp| > 当日 |Δp| 的 99% 分位
    if n > 2 {
        let mut dp_abs: Vec<f64> = (0..n - 1).map(|i| (p[i + 1] - p[i]).abs()).collect();
        let k = ((n as f64) * 0.99).round().max(1.0) as usize;
        let k = k.min(dp_abs.len() - 1);
        dp_abs.select_nth_unstable_by(k, |a, b| a.total_cmp(b));
        let thr = dp_abs[k];
        let mut cand_j: Vec<(i64, f64)> = Vec::new();
        for i in 0..n - 1 {
            let dp = p[i + 1] - p[i];
            if dp.abs() > thr {
                cand_j.push((t[i + 1], dp.abs()));
            }
        }
        push_capped(&mut out[5], &mut cand_j);
    }
    assert!(out.iter().all(|s| s.t.windows(2).all(|w| w[0] <= w[1])));
    out
}

fn day_base(t: i64) -> i64 {
    (t / 86_400_000_000) * 86_400_000_000 + 28_800_000_000
}

fn period_slice(t: &[i64], base: i64) -> (usize, usize) {
    let lo = base + LO_S * 1_000_000;
    let hi = base + HI_S * 1_000_000;
    let s = t.partition_point(|&x| x < lo);
    let e = t.partition_point(|&x| x < hi);
    (s, e)
}

fn c_of_m(m: u32) -> f64 {
    1.0 - 2f64.powf(-1.0 / m.max(1) as f64)
}

/// 对级 8 维，按序：rhit_f, rhit_b, rmed_f, rmed_b, hit_f, hit_b, med_f, med_b
const M: usize = 8;
const IDX_RHIT_F: usize = 0;
const IDX_RHIT_B: usize = 1;
const IDX_RMED_F: usize = 2;
const IDX_RMED_B: usize = 3;
const IDX_HIT_F: usize = 4;
const IDX_HIT_B: usize = 5;
const IDX_MED_F: usize = 6;
const IDX_MED_B: usize = 7;

/// 单对单事件统计（有序 (A,B)：fwd=B 响应 A，bwd=A 响应 B）。
/// 返回 None 当 A 无事件或两侧均无匹配。
fn pair_stats(ta: &[i64], tb: &[i64], base: i64) -> Option<[f64; M]> {
    if ta.is_empty() || tb.is_empty() {
        return None;
    }
    let hi_us = (base + HI_S * 1_000_000) as f64;
    let lo_us = (base + LO_S * 1_000_000) as f64;
    let m_b = tb.len() as u32;
    let c = c_of_m(m_b);

    // 每方向收集 (距离µs, xµs) 与几何和
    let mut fwd_d: Vec<(u64, f64)> = Vec::with_capacity(ta.len());
    let mut bwd_d: Vec<(u64, f64)> = Vec::with_capacity(ta.len());
    let mut lnx_f = 0.0f64;
    let mut lnx_b = 0.0f64;
    let mut j = 0usize; // 指向首个 > a 的 B
    for &a in ta {
        while j < tb.len() && tb[j] <= a {
            j += 1;
        }
        if j < tb.len() {
            let d = (tb[j] - a) as u64;
            let x = (hi_us - a as f64).max(1.0);
            fwd_d.push((d, x));
            lnx_f += x.ln();
        }
        if j > 0 {
            let d = (a - tb[j - 1]) as u64;
            let x = (a as f64 - lo_us).max(1.0);
            bwd_d.push((d, x));
            lnx_b += x.ln();
        }
    }
    if fwd_d.is_empty() && bwd_d.is_empty() {
        return None;
    }
    let mut out = [f64::NAN; M];
    for (k, arr, lnx) in [(0usize, &fwd_d, lnx_f), (1, &bwd_d, lnx_b)] {
        if arr.is_empty() {
            continue;
        }
        // 精确中位
        let mut s: Vec<u64> = arr.iter().map(|&(d, _)| d).collect();
        s.sort_unstable();
        let med_s = s[s.len() / 2] as f64 / 1e6;
        // hit / null_hit（同匹配集）
        let n_x = arr.len() as f64;
        let mut hit = 0.0f64;
        let mut null_hit = 0.0f64;
        for &(d, x) in arr.iter() {
            if d <= T_US {
                hit += 1.0;
            }
            let base_p = ((x - T_US as f64) / x).clamp(0.0, 1.0);
            null_hit += 1.0 - base_p.powf(m_b as f64);
        }
        hit /= n_x;
        null_hit /= n_x;
        let null_med = (lnx / n_x).exp() * c / 1e6; // 秒
        let (i_rhit, i_rmed, i_hit, i_med) = if k == 0 {
            (IDX_RHIT_F, IDX_RMED_F, IDX_HIT_F, IDX_MED_F)
        } else {
            (IDX_RHIT_B, IDX_RMED_B, IDX_HIT_B, IDX_MED_B)
        };
        out[i_rhit] = hit / null_hit.max(1e-9);
        out[i_rmed] = med_s / null_med.max(1e-9);
        out[i_hit] = hit;
        out[i_med] = med_s;
    }
    Some(out)
}

#[derive(Serialize)]
struct TopPair {
    event: usize,
    a: String,
    b: String,
    ia: usize,
    ib: usize,
    strength: f32,
    spec: f32,
    rhit_f: f32,
    rhit_b: f32,
    rmed_f: f32,
    rmed_b: f32,
    hit_f: f32,
    hit_b: f32,
}

#[derive(Serialize)]
struct Out {
    date: i64,
    codes: Vec<String>,
    inds: Vec<u32>,
    events: Vec<String>,
    factor_names: Vec<String>,
    vals: Vec<Vec<f32>>, // [股票][6 事件 × 14 因子]
    top_pairs: Vec<TopPair>,
}

fn main() {
    let date: i64 = std::env::args().nth(1).expect("用法: yhyb_topcap_sandbox <date>").parse().unwrap();
    let n_threads = std::env::var("RAYON_NUM_THREADS").ok().and_then(|v| v.parse().ok()).unwrap_or(64);
    let _ = rayon::ThreadPoolBuilder::new().num_threads(n_threads).build_global();

    let univ_path = format!("universe_{date}.txt");
    let f = std::fs::File::open(&univ_path).unwrap_or_else(|_| panic!("缺少 universe 文件 {univ_path}"));
    let mut codes: Vec<String> = Vec::new();
    let mut inds: Vec<u32> = Vec::new();
    for line in BufReader::new(f).lines() {
        let line = line.unwrap();
        let mut it = line.split_whitespace();
        let c = it.next().unwrap().to_string();
        let i: u32 = it.next().unwrap().parse().unwrap();
        codes.push(c);
        inds.push(i);
    }
    let n = codes.len();
    eprintln!("date={date} universe={n}");

    let streams: Vec<Option<[EvStream; N_EV]>> = codes
        .par_iter()
        .map(|c| {
            let trades = read_trade_fast(c, date).ok()?;
            let nt = trades.len();
            let mut t = Vec::with_capacity(nt);
            let mut p = Vec::with_capacity(nt);
            let mut amt = Vec::with_capacity(nt);
            let mut f = Vec::with_capacity(nt);
            for tr in &trades {
                t.push(tr.time_us);
                p.push(tr.price);
                amt.push(tr.turnover);
                f.push(tr.flag);
            }
            Some(detect6(&t, &p, &amt, &f))
        })
        .collect();

    let base = streams
        .iter()
        .filter_map(|s| s.as_ref())
        .find_map(|s| s.iter().find(|e| !e.t.is_empty()).map(|e| day_base(e.t[0])))
        .unwrap_or(0);

    let mut slices: Vec<Option<Vec<Vec<i64>>>> = Vec::with_capacity(n);
    for s in &streams {
        let Some(s) = s else { slices.push(None); continue };
        let mut per_ev: Vec<Vec<i64>> = Vec::with_capacity(N_EV);
        for e in s.iter() {
            let (lo, hi) = period_slice(&e.t, base);
            per_ev.push(e.t[lo..hi].to_vec());
        }
        slices.push(Some(per_ev));
    }

    let mut vals: Vec<Vec<f32>> = vec![vec![f64::NAN as f32; N_EV * NF]; n];
    let mut top_pairs: Vec<TopPair> = Vec::new();

    // 先把所有事件的对级矩阵算出来（内存：6 事件 × n² × 8 f32 ≈ 6×310²×8×4B ≈ 18MB，可行）
    // 步骤：对每个事件并行算所有 (i<j) 对 → pm_ev[ev][i*n+j]
    let mut all_pm: Vec<Vec<[f32; M]>> = Vec::with_capacity(N_EV);
    for ev in 0..N_EV {
        let mut mat = vec![[f32::NAN; M]; n * n];
        let si: Vec<Option<&Vec<i64>>> = slices.iter().map(|s| s.as_ref().map(|se| &se[ev])).collect();
        let idxs: Vec<(usize, usize)> = (0..n).flat_map(|i| (i + 1..n).map(move |j| (i, j))).collect();
        let res: Vec<((usize, usize), Option<[f64; M]>, Option<[f64; M]>)> = idxs
            .par_iter()
            .filter_map(|&(i, j)| {
                let (a, b) = (si[i]?, si[j]?);
                Some(((i, j), pair_stats(a, b, base), pair_stats(b, a, base)))
            })
            .collect();
        for (ij, p_ab, p_ba) in res {
            let (i, j) = ij;
            // 有序对 (A,B) 存 p_ab（A 事件 vs B 事件：rhit_f=B响应A, rhit_b=A响应B）；
            // 有序对 (B,A) 存 p_ba；一侧无匹配时用对侧同语义方向补充。
            let m_ab = match (p_ab, p_ba) {
                (Some(x), _) => x,
                (None, Some(y)) => [y[1], y[0], y[3], y[2], y[5], y[4], y[7], y[6]],
                _ => continue,
            };
            let m_ba = match (p_ba, p_ab) {
                (Some(x), _) => x,
                (None, Some(y)) => [y[1], y[0], y[3], y[2], y[5], y[4], y[7], y[6]],
                _ => m_ab,
            };
            mat[i * n + j] = m_ab.map(|x| x as f32);
            mat[j * n + i] = m_ba.map(|x| x as f32);
        }
        all_pm.push(mat);
    }
    eprintln!("pair matrices done");

    for ev in 0..N_EV {
        let mat = &all_pm[ev];
        // 跨对百分位：rhit_score 与 comp
        let mut rhit_scores: Vec<f64> = Vec::new();
        let mut comps: Vec<f64> = Vec::new();
        for i in 0..n {
            for j in i + 1..n {
                let m = mat[i * n + j];
                let rs = if m[0].is_finite() && m[1].is_finite() {
                    0.5 * (m[0] as f64 + m[1] as f64)
                } else if m[0].is_finite() {
                    m[0] as f64
                } else if m[1].is_finite() {
                    m[1] as f64
                } else {
                    f64::NAN
                };
                let cp = if m[2].is_finite() && m[2] > 0.0 && m[3].is_finite() && m[3] > 0.0 {
                    0.5 * (1.0 / m[2] as f64 + 1.0 / m[3] as f64)
                } else if m[2].is_finite() && m[2] > 0.0 {
                    1.0 / m[2] as f64
                } else if m[3].is_finite() && m[3] > 0.0 {
                    1.0 / m[3] as f64
                } else {
                    f64::NAN
                };
                if rs.is_finite() {
                    rhit_scores.push(rs);
                }
                if cp.is_finite() {
                    comps.push(cp);
                }
            }
        }
        rhit_scores.sort_unstable_by(|a, b| a.total_cmp(b));
        comps.sort_unstable_by(|a, b| a.total_cmp(b));
        let rank_pct = |v: f64, arr: &[f64]| -> f64 {
            let lo = arr.partition_point(|&x| x < v - 1e-12);
            let hi = arr.partition_point(|&x| x <= v + 1e-12);
            if arr.is_empty() {
                return f64::NAN;
            }
            (0.5 * (lo + hi) as f64) / arr.len() as f64
        };
        // strength 矩阵
        let mut strength = vec![[f64::NAN; 2]; n * n]; // [0]=strength, [1]=rhit_score
        for i in 0..n {
            for j in i + 1..n {
                let m = mat[i * n + j];
                let rs = if m[0].is_finite() && m[1].is_finite() {
                    0.5 * (m[0] as f64 + m[1] as f64)
                } else if m[0].is_finite() {
                    m[0] as f64
                } else if m[1].is_finite() {
                    m[1] as f64
                } else {
                    f64::NAN
                };
                let cp = if m[2].is_finite() && m[2] > 0.0 && m[3].is_finite() && m[3] > 0.0 {
                    0.5 * (1.0 / m[2] as f64 + 1.0 / m[3] as f64)
                } else if m[2].is_finite() && m[2] > 0.0 {
                    1.0 / m[2] as f64
                } else if m[3].is_finite() && m[3] > 0.0 {
                    1.0 / m[3] as f64
                } else {
                    f64::NAN
                };
                let st = match (rs.is_finite(), cp.is_finite()) {
                    (true, true) => 0.5 * (rank_pct(rs, &rhit_scores) + rank_pct(cp, &comps)),
                    (true, false) => rank_pct(rs, &rhit_scores),
                    (false, true) => rank_pct(cp, &comps),
                    _ => f64::NAN,
                };
                strength[i * n + j][0] = st;
                strength[j * n + i][0] = st;
                strength[i * n + j][1] = rs;
                strength[j * n + i][1] = rs;
            }
        }
        // marg & spec
        let mut row_valid = vec![0usize; n];
        let mut row_sum = vec![0.0f64; n];
        for i in 0..n {
            for j in 0..n {
                if i != j && strength[i * n + j][0].is_finite() {
                    row_valid[i] += 1;
                    row_sum[i] += strength[i * n + j][0];
                }
            }
        }
        let marg: Vec<f64> = (0..n)
            .map(|i| if row_valid[i] > 0 { row_sum[i] / row_valid[i] as f64 } else { f64::NAN })
            .collect();
        let mut spec = vec![f64::NAN; n * n];
        for i in 0..n {
            for j in i + 1..n {
                let st = strength[i * n + j][0];
                if st.is_finite() && marg[i].is_finite() && marg[j].is_finite() {
                    let sp = st - 0.5 * (marg[i] + marg[j]);
                    spec[i * n + j] = sp;
                    spec[j * n + i] = sp;
                }
            }
        }

        // 每股聚合
        for a in 0..n {
            let mut arr = [f64::NAN; NF];
            let mut nv = 0usize;
            let mut sum_st = 0.0f64;
            let mut sum_rhit_f = 0.0f64;
            let mut sum_rhit_b = 0.0f64;
            let mut sum_rmed_f = 0.0f64;
            let mut sum_rmed_b = 0.0f64;
            let mut sum_hit_f = 0.0f64;
            let mut sum_med_b = 0.0f64;
            let mut strong = 0usize;
            let mut n_same = 0usize;
            let mut n_same_strong = 0usize;
            let mut pair_list: Vec<(f64, f64, bool)> = Vec::new(); // (strength, spec, same_ind)
            for b in 0..n {
                if a == b {
                    continue;
                }
                let st = strength[a * n + b][0];
                if !st.is_finite() {
                    continue;
                }
                nv += 1;
                sum_st += st;
                let sp = spec[a * n + b];
                if sp.is_finite() && sp > 0.0 {
                    strong += 1;
                }
                let m = mat[a * n + b];
                if m[0].is_finite() {
                    sum_rhit_f += m[0] as f64;
                }
                if m[1].is_finite() {
                    sum_rhit_b += m[1] as f64;
                }
                if m[2].is_finite() {
                    sum_rmed_f += m[2] as f64;
                }
                if m[3].is_finite() {
                    sum_rmed_b += m[3] as f64;
                }
                if m[4].is_finite() {
                    sum_hit_f += m[4] as f64;
                }
                if m[7].is_finite() {
                    sum_med_b += m[7] as f64;
                }
                let same = inds[a] == inds[b];
                if same {
                    n_same += 1;
                    if sp.is_finite() && sp > 0.0 {
                        n_same_strong += 1;
                    }
                }
                pair_list.push((st, sp, same));
            }
            if nv == 0 {
                continue;
            }
            let rate = slices[a].as_ref().map(|s| s[ev].len() as f64).unwrap_or(0.0);
            arr[0] = rate;
            arr[1] = sum_st / nv as f64;
            pair_list.sort_by(|x, y| y.1.total_cmp(&x.1));
            let topk = |k: usize, arr: &mut [f64; NF], idx: usize| {
                let cnt = k.min(pair_list.len());
                if cnt > 0 {
                    arr[idx] = pair_list.iter().take(cnt).map(|x| x.0).sum::<f64>() / cnt as f64;
                }
            };
            topk(5, &mut arr, 2);
            topk(10, &mut arr, 3);
            let cnt5 = 5.min(pair_list.len());
            if cnt5 > 0 {
                arr[4] = pair_list.iter().take(cnt5).map(|x| x.1).sum::<f64>() / cnt5 as f64;
            }
            // 同行业 top5（按 spec 排序后取同行业的前 5）与跨行业 top5
            let same_list: Vec<(f64, f64)> =
                pair_list.iter().filter(|x| x.2).map(|x| (x.0, x.1)).collect();
            let cross_list: Vec<(f64, f64)> =
                pair_list.iter().filter(|x| !x.2).map(|x| (x.0, x.1)).collect();
            let pick5 = |lst: &[(f64, f64)]| -> f64 {
                let cnt = 5.min(lst.len());
                if cnt > 0 {
                    lst.iter().take(cnt).map(|x| x.0).sum::<f64>() / cnt as f64
                } else {
                    f64::NAN
                }
            };
            arr[14] = pick5(&same_list);
            arr[15] = pick5(&cross_list);
            arr[5] = strong as f64 / nv as f64;
            arr[6] = if n_same > 0 { n_same_strong as f64 / n_same as f64 } else { f64::NAN };
            arr[7] = sum_rhit_f / nv as f64;
            arr[8] = sum_rhit_b / nv as f64;
            arr[9] = sum_rmed_f / nv as f64;
            arr[10] = sum_rmed_b / nv as f64;
            arr[11] = sum_hit_f / nv as f64;
            arr[12] = sum_med_b / nv as f64;
            arr[13] = nv as f64;
            for k in 0..NF {
                vals[a][ev * NF + k] = arr[k] as f32;
            }
        }

        // 该事件 top200 对
        let mut cands: Vec<(usize, usize, f64, f64)> = Vec::new();
        for i in 0..n {
            for j in i + 1..n {
                let st = strength[i * n + j][0];
                if !st.is_finite() {
                    continue;
                }
                let sp = spec[i * n + j];
                if sp.is_finite() {
                    cands.push((i, j, st, sp));
                }
            }
        }
        cands.sort_by(|a, b| b.3.total_cmp(&a.3));
        for &(i, j, _st, sp) in cands.iter().take(200) {
            let m = mat[i * n + j];
            top_pairs.push(TopPair {
                event: ev,
                a: codes[i].clone(),
                b: codes[j].clone(),
                ia: i,
                ib: j,
                strength: strength[i * n + j][0] as f32,
                spec: sp as f32,
                rhit_f: m[0],
                rhit_b: m[1],
                rmed_f: m[2],
                rmed_b: m[3],
                hit_f: m[4],
                hit_b: m[5],
            });
        }
        eprintln!("event {} done", EV_NAMES[ev]);
    }

    let out = Out {
        date,
        codes,
        inds,
        events: EV_NAMES.iter().map(|s| s.to_string()).collect(),
        factor_names: FACTOR_NAMES.iter().map(|s| s.to_string()).collect(),
        vals,
        top_pairs,
    };
    println!("{}", serde_json::to_string(&out).unwrap());
}
