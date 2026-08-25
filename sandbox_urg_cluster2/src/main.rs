//! 迫切交易时序聚类程度因子探索（sandbox_urg_cluster2，复算 + 超越旧 agent 版）。
//!
//! 口径与正式实现 urgency_metrics.rs 完全一致：
//!   ratio = (ask_order - bid_order) / (ask_order + bid_order)  全市场 q5/q95 定阈值
//!   sell 种子: ratio > q95（卖出急切的，卖方订单号很大）
//!   buy  种子: ratio < q5 （买入急切的）
//!   both 种子: |ratio| > q95_abs（双向迫切）
//! 时间用 time_us（已做下午时段平移 ajust_afternoon，午休无间隙）。
//!
//! 每只股票对三类种子序列分别算 26 个聚类指标（含旧 agent 的 gini_gap/gap_cv/gap_mean/
//! bucket_ent/dens300/span 复现 + 新指标）+ 3 个活跃度对照（总笔数/成交额/总量）。
//!
//! 构建: cd sandbox_urg_cluster2 && cargo build --release
//! 运行: RAYON_NUM_THREADS=64 ./target/release/urg_cluster2_sandbox 20240104 > out_20240104.json
//! 日志走 stderr，结果走 stdout（NaN 序列化为 null）。

mod fast_csv_reader;
use fast_csv_reader::{read_trade_fast, TradeRecord};
use rayon::prelude::*;
use serde::Serialize;
use std::collections::BTreeSet;
use std::fs;

/// adjust 后连续交易日：9:30 开盘（日内秒 34200），收盘 13:27（= 15:00 - 90min）
const SESSION0_SEC: i64 = 34200;
const SESSION_LEN_SEC: i64 = (3 * 3600 + 57 * 60) as i64; // 14220 秒 = 237 分钟
const SEC_US: i64 = 1_000_000;
const N_FEAT_PER_VER: usize = 29;
const N_GLOB: usize = 3;

#[derive(Serialize)]
struct Out {
    names: Vec<String>,
    codes: Vec<String>,
    vals: Vec<Vec<Option<f64>>>,
}

fn list_codes(date: i64) -> Vec<String> {
    for root in ["/ssd_data/stock", "/nas197/binary/stock/sz_alpha/stock"] {
        let dir = format!("{root}/{date}/transaction");
        if let Ok(entries) = fs::read_dir(&dir) {
            let mut set = BTreeSet::new();
            for e in entries.flatten() {
                if let Some(code) = e.file_name().to_str().and_then(|n| n.split('_').next()) {
                    if code.bytes().all(|b| b.is_ascii_digit()) {
                        set.insert(code.to_string());
                    }
                }
            }
            if !set.is_empty() {
                return set.into_iter().collect();
            }
        }
    }
    Vec::new()
}

#[inline]
fn ratio_of(t: &TradeRecord) -> f32 {
    let s = (t.ask_order + t.bid_order) as f64;
    if s > 0.0 {
        ((t.ask_order - t.bid_order) as f64 / s) as f32
    } else {
        0.0
    }
}

/// 通用统计工具
fn mean_std(x: &[f64]) -> (f64, f64) {
    let n = x.len();
    if n == 0 {
        return (f64::NAN, f64::NAN);
    }
    let m = x.iter().sum::<f64>() / n as f64;
    if n < 2 {
        return (m, f64::NAN);
    }
    let v = x.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / n as f64;
    (m, v.sqrt())
}

fn gini_gap(g_sorted: &[f64]) -> f64 {
    let n = g_sorted.len();
    let s: f64 = g_sorted.iter().sum();
    if n < 2 || s <= 0.0 {
        return f64::NAN;
    }
    let mut acc = 0.0f64;
    for (i, &x) in g_sorted.iter().enumerate() {
        acc += (i as f64 + 1.0) * x;
    }
    2.0 * acc / (n as f64 * s) - (n as f64 + 1.0) / n as f64
}

fn kurt(x: &[f64]) -> f64 {
    let n = x.len();
    if n < 4 {
        return f64::NAN;
    }
    let m = x.iter().sum::<f64>() / n as f64;
    let m2 = x.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / n as f64;
    if m2 <= 0.0 {
        return f64::NAN;
    }
    let m4 = x.iter().map(|v| ((v - m) * (v - m)).powi(2)).sum::<f64>() / n as f64;
    m4 / (m2 * m2) - 3.0
}

/// 间隔序列 vs 指数分布（率 = 1/mean）的 KS 距离
fn ks_exp(g_sorted: &[f64]) -> f64 {
    let n = g_sorted.len();
    let m = g_sorted.iter().sum::<f64>() / n as f64;
    if n < 2 || m <= 0.0 {
        return f64::NAN;
    }
    let mut dmax = 0.0f64;
    for (i, &x) in g_sorted.iter().enumerate() {
        let cdf = 1.0 - (-x / m).exp();
        let d1 = ((i as f64 + 1.0) / n as f64 - cdf).abs();
        let d2 = (i as f64 / n as f64 - cdf).abs();
        if d1 > dmax {
            dmax = d1;
        }
        if d2 > dmax {
            dmax = d2;
        }
    }
    dmax
}

/// lag-1 自相关（相邻间隔对）
fn ac1(gaps: &[f64]) -> f64 {
    let m = gaps.len();
    if m < 3 {
        return f64::NAN;
    }
    let mu = gaps.iter().sum::<f64>() / m as f64;
    let mut num = 0.0;
    let mut den = 0.0;
    for i in 0..m - 1 {
        num += (gaps[i] - mu) * (gaps[i + 1] - mu);
        den += (gaps[i] - mu) * (gaps[i] - mu);
    }
    if den <= 0.0 {
        f64::NAN
    } else {
        num / den
    }
}

/// 秒桶计数（日内 0..SESSION_LEN_SEC），越界忽略
fn count_sec(times: &[i64]) -> Vec<u32> {
    let mut c = vec![0u32; SESSION_LEN_SEC as usize];
    for &t in times {
        let sec = (t / SEC_US) % 86400 - SESSION0_SEC;
        if sec >= 0 && sec < SESSION_LEN_SEC {
            c[sec as usize] += 1;
        }
    }
    c
}

/// 种子时间序列（已排序，微秒）→ 聚类指标；masses 为同序迫切强度 |ratio|。
fn cluster_feats(times: &[i64], masses: &[f64]) -> Vec<f64> {
    let n = times.len();
    let mut out = vec![f64::NAN; N_FEAT_PER_VER];
    if n == 0 {
        return out;
    }
    out[0] = n as f64;
    out[1] = (times[n - 1] - times[0]) as f64 / SEC_US as f64; // span 秒

    // 间隔序列
    let mut gaps: Vec<f64> = Vec::with_capacity(n - 1);
    for i in 1..n {
        gaps.push((times[i] - times[i - 1]) as f64 / SEC_US as f64);
    }
    let mut gs = gaps.clone();
    gs.sort_by(|a, b| a.total_cmp(b));
    let (gm, gsd) = mean_std(&gaps);
    out[2] = gini_gap(&gs); // gini_gap
    out[3] = gm; // gap_mean
    out[4] = if gm > 0.0 && gsd.is_finite() { gsd / gm } else { f64::NAN }; // gap_cv
    out[5] = if gm > 0.0 && gsd.is_finite() {
        (gsd - gm) / (gsd + gm) // burstiness
    } else {
        f64::NAN
    };
    out[6] = if gm > 0.0 && gsd.is_finite() {
        gsd * gsd / gm // fano factor
    } else {
        f64::NAN
    };
    out[7] = kurt(&gaps); // 峰度（超额）
    out[8] = ks_exp(&gs); // 指数分布 KS 距离
    out[9] = ac1(&gaps); // lag-1 自相关

    // 桶熵（固定桶归一化）
    let sec_c = count_sec(times);
    let (ent_all, ent_nz, nz_cnt) = {
        let nb = (SESSION_LEN_SEC / 60) as usize;
        let mut bc = vec![0u32; nb];
        for s in 0..SESSION_LEN_SEC as usize {
            bc[s / 60] += sec_c[s];
        }
        let mut ent = 0.0f64;
        let mut nz = 0usize;
        let mut ent_nzv = 0.0f64;
        for &c in &bc {
            if c > 0 {
                let p = c as f64 / n as f64;
                ent -= p * p.ln();
                nz += 1;
            }
        }
        if nz > 1 {
            for &c in &bc {
                if c > 0 {
                    let p = c as f64 / n as f64;
                    ent_nzv -= p * p.ln(); // 非零归一化分母在下面算
                }
            }
            ent_nzv /= (nz as f64).ln();
        } else {
            ent_nzv = f64::NAN;
        }
        ent /= (nb as f64).ln();
        (ent, ent_nzv, nz)
    };
    out[10] = ent_all; // 60s 桶熵（含空桶，ln(237) 归一）
    out[11] = ent_nz; // 60s 桶熵（仅非零桶归一化）
    let _ = nz_cnt;

    // 变宽桶熵
    for (k, bw) in [(1usize, 2i64), (2, 30), (3, 120)] {
        let nb = ((SESSION_LEN_SEC + bw - 1) / bw) as usize;
        let mut bc = vec![0u32; nb];
        for s in 0..SESSION_LEN_SEC as usize {
            bc[s / bw as usize] += sec_c[s];
        }
        let mut ent = 0.0f64;
        for &c in &bc {
            if c > 0 {
                let p = c as f64 / n as f64;
                ent -= p * p.ln();
            }
        }
        ent /= (nb as f64).ln();
        out[10 + 1 + k] = ent; // out[12]=2s, out[13]=30s, out[14]=120s
    }

    // dens300：最密集 300 秒滑动窗口内种子占比
    {
        let w = 300usize;
        let mut cur: u64 = 0;
        for s in 0..w.min(SESSION_LEN_SEC as usize) {
            cur += sec_c[s] as u64;
        }
        let mut best = cur;
        for s in w..SESSION_LEN_SEC as usize {
            cur += sec_c[s] as u64 - sec_c[s - w] as u64;
            if cur > best {
                best = cur;
            }
        }
        out[15] = best as f64 / n as f64;
    }

    // 局部强度峰：1 分钟桶计数 vs 均值 + 2σ（泊松基线 σ=sqrt(μ)）
    {
        let nb = (SESSION_LEN_SEC / 60) as usize;
        let mut mc = vec![0u32; nb];
        for s in 0..SESSION_LEN_SEC as usize {
            mc[s / 60] += sec_c[s];
        }
        let mu = n as f64 / nb as f64;
        let sig = mu.max(1e-12).sqrt();
        let thr = mu + 2.0 * sig;
        let mut n_peaks = 0usize;
        let mut in_peak = false;
        let mut mx = 0.0f64;
        for &c in &mc {
            if c as f64 > thr {
                if !in_peak {
                    n_peaks += 1;
                    in_peak = true;
                }
                mx = mx.max(c as f64 / mu);
            } else {
                in_peak = false;
            }
        }
        out[16] = n_peaks as f64;
        out[17] = if mu > 0.0 { mx } else { f64::NAN };
    }

    // 三段（开盘/盘中/尾盘，各 1/3）：段内间隔 CV + 段内占比
    {
        let seg_sec = SESSION_LEN_SEC / 3;
        let bounds = [0i64, seg_sec, 2 * seg_sec, SESSION_LEN_SEC];
        let mut seg_cv = [f64::NAN; 3];
        let mut seg_n = [0usize; 3];
        for &t in times {
            let sec = (t / SEC_US) % 86400 - SESSION0_SEC;
            if sec >= 0 && sec < SESSION_LEN_SEC {
                let k = (sec / seg_sec).min(2) as usize;
                seg_n[k] += 1;
            }
        }
        for k in 0..3 {
            let mut sg: Vec<f64> = Vec::new();
            for &t in times {
                let sec = (t / SEC_US) % 86400 - SESSION0_SEC;
                if sec >= bounds[k] && sec < bounds[k + 1] {
                    sg.push(t as f64);
                }
            }
            if sg.len() >= 2 {
                let mut gg: Vec<f64> = Vec::with_capacity(sg.len() - 1);
                for i in 1..sg.len() {
                    gg.push((sg[i] - sg[i - 1]) as f64 / SEC_US as f64);
                }
                let (m, sd) = mean_std(&gg);
                seg_cv[k] = if m > 0.0 && sd.is_finite() { sd / m } else { f64::NAN };
            }
            out[18 + k] = seg_cv[k];
            let _ = bounds;
        }
        // seg_cv_diff = max - min
        let vals: Vec<f64> = seg_cv.iter().filter(|v| v.is_finite()).copied().collect();
        out[21] = if vals.len() == 3 {
            vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
                - vals.iter().cloned().fold(f64::INFINITY, f64::min)
        } else {
            f64::NAN
        };
        for k in 0..3 {
            out[22 + k] = seg_n[k] as f64 / n as f64; // seg_share1/2/3
        }
    }

    // 质量加权（|ratio| 权重）桶熵族：mass 在 60s 桶内的集中度
    {
        let nb = (SESSION_LEN_SEC / 60) as usize;
        let mut mcnt = vec![0.0f64; nb];
        for (i, &t) in times.iter().enumerate() {
            let sec = (t / SEC_US) % 86400 - SESSION0_SEC;
            if sec >= 0 && sec < SESSION_LEN_SEC {
                mcnt[sec as usize / 60] += masses[i];
            }
        }
        let mtot: f64 = mcnt.iter().sum();
        let mut ent = 0.0f64;
        let mut hhi = 0.0f64;
        let mut nz = 0usize;
        let mut ent_nz = 0.0f64;
        if mtot > 1e-12 {
            for &c in &mcnt {
                if c > 1e-12 {
                    let p = c / mtot;
                    ent -= p * p.ln();
                    hhi += p * p;
                    nz += 1;
                }
            }
            ent /= (nb as f64).ln();
            if nz > 1 {
                for &c in &mcnt {
                    if c > 1e-12 {
                        let p = c / mtot;
                        ent_nz -= p * p.ln();
                    }
                }
                ent_nz /= (nz as f64).ln();
            } else {
                ent_nz = f64::NAN;
            }
        } else {
            ent = f64::NAN;
            ent_nz = f64::NAN;
            hhi = f64::NAN;
        }
        out[25] = ent; // ent60_mass
        out[26] = ent_nz; // ent60_mass_nz
        out[27] = hhi; // hhi60_mass
        // dens300_mass：5 分钟质量集中度（逐秒质量桶 + 300s 滑动和）
        mut_slide(&sec_c, &times, &masses, &mut out);
    }
    out
}

/// dens300_mass（赋值给 out[28]）
fn mut_slide(sec_c: &[u32], times: &[i64], masses: &[f64], out: &mut [f64]) {
    let w = 300usize;
    let mut msec = vec![0.0f64; SESSION_LEN_SEC as usize];
    for (i, &t) in times.iter().enumerate() {
        let sec = (t / SEC_US) % 86400 - SESSION0_SEC;
        if sec >= 0 && sec < SESSION_LEN_SEC {
            msec[sec as usize] += masses[i];
        }
    }
    let mut cur = 0.0f64;
    for s in 0..w.min(SESSION_LEN_SEC as usize) {
        cur += msec[s];
    }
    let mut best = cur;
    for s in w..SESSION_LEN_SEC as usize {
        cur += msec[s] - msec[s - w];
        if cur > best {
            best = cur;
        }
    }
    let _ = sec_c;
    out[28] = best;
}

fn compute_stock(trades: &[TradeRecord], q95: f32, q5: f32, q95abs: f32) -> Vec<f64> {
    let mut sell_t = Vec::new();
    let mut sell_m = Vec::new();
    let mut buy_t = Vec::new();
    let mut buy_m = Vec::new();
    let mut both_t = Vec::new();
    let mut both_m = Vec::new();
    let mut ntr = 0usize;
    let mut to = 0.0f64;
    let mut vv = 0.0f64;
    for t in trades {
        ntr += 1;
        to += t.turnover;
        vv += t.volume;
        let r = ratio_of(t);
        if r > q95 {
            sell_t.push(t.time_us);
            sell_m.push(r as f64);
        }
        if r < q5 {
            buy_t.push(t.time_us);
            buy_m.push((-r) as f64);
        }
        if r.abs() > q95abs {
            both_t.push(t.time_us);
            both_m.push(r.abs() as f64);
        }
    }
    sell_t.sort_unstable();
    buy_t.sort_unstable();
    both_t.sort_unstable();
    let mut out = Vec::with_capacity(N_GLOB + 3 * N_FEAT_PER_VER);
    out.push(ntr as f64);
    out.push(to);
    out.push(vv);
    for (tt, mm) in [(&sell_t, &sell_m), (&buy_t, &buy_m), (&both_t, &both_m)] {
        out.extend(cluster_feats(tt, mm));
    }
    out
}

fn main() {
    let date: i64 = std::env::args()
        .nth(1)
        .expect("用法: urg_cluster2_sandbox <date>")
        .parse()
        .unwrap();
    let codes = list_codes(date);
    eprintln!("date={date} codes={}", codes.len());

    // ① 第一遍：全市场 ratio 收集 → 分位数阈值（与正式实现同口径）
    // 注：部分日期 flag 列带 '.0'（如 "32.0"），read_trade_fast 的字节串比对
    // 过滤失效，这里按解析后的 i32 flag 补过滤撤单行（与 compute_urgency_full 一致）。
    let per_ratios: Vec<(String, Vec<f32>)> = codes
        .par_iter()
        .map(|c| {
            let mut rs = Vec::new();
            if let Ok(mut ts) = read_trade_fast(c, date) {
                ts.retain(|t| t.flag != 32);
                rs.reserve(ts.len());
                for t in &ts {
                    rs.push(ratio_of(t));
                }
            }
            (c.clone(), rs)
        })
        .collect();
    let total_n: usize = per_ratios.iter().map(|(_, r)| r.len()).sum();
    eprintln!("total trades = {total_n}");
    let mut all_ratio: Vec<f32> = Vec::with_capacity(total_n);
    for (_, r) in &per_ratios {
        all_ratio.extend_from_slice(r);
    }
    drop(per_ratios);
    all_ratio.par_sort_unstable_by(|a, b| a.total_cmp(b));
    let qidx = |q: f64| ((total_n as f64 * q) as usize).min(total_n - 1);
    let q95 = all_ratio[qidx(0.95)];
    let q5 = all_ratio[qidx(0.05)];
    let q95abs = {
        let mut absv: Vec<f32> = all_ratio.iter().map(|v| v.abs()).collect();
        absv.par_sort_unstable_by(|a, b| a.total_cmp(b));
        absv[qidx(0.95)]
    };
    drop(all_ratio);
    eprintln!("q95={q95} q5={q5} q95abs={q95abs}");

    // ② 第二遍：标记种子 → per-stock 聚类指标
    let feats: Vec<(String, Vec<f64>)> = codes
        .par_iter()
        .filter_map(|c| {
            let mut ts = read_trade_fast(c, date).ok()?;
            ts.retain(|t| t.flag != 32);
            if ts.is_empty() {
                return None;
            }
            let f = compute_stock(&ts, q95, q5, q95abs);
            Some((c.clone(), f))
        })
        .collect();
    eprintln!("valid stocks = {}", feats.len());

    // ③ 列名 + 输出
    let ver_tags = ["sell", "buy", "both"];
    let base_names = [
        "n_seeds",
        "span_sec",
        "gini_gap",
        "gap_mean",
        "gap_cv",
        "burst",
        "fano",
        "gap_kurt",
        "ks_exp",
        "ac1",
        "ent60",
        "ent60_nz",
        "ent2",
        "ent30",
        "ent120",
        "dens300",
        "n_peaks",
        "max_peak",
        "seg_cv1",
        "seg_cv2",
        "seg_cv3",
        "seg_cv_diff",
        "seg_share1",
        "seg_share2",
        "seg_share3",
        "ent60_mass",
        "ent60_mass_nz",
        "hhi60_mass",
        "dens300_mass",
    ];
    let mut names = vec!["n_trades".to_string(), "turnover".to_string(), "vol".to_string()];
    for tag in ver_tags {
        for b in base_names {
            names.push(format!("{tag}_{b}"));
        }
    }
    let mut codes_out = Vec::with_capacity(feats.len());
    let mut vals_out = Vec::with_capacity(feats.len());
    for (c, f) in feats {
        codes_out.push(c);
        vals_out.push(f.into_iter().map(|v| if v.is_finite() { Some(v) } else { None }).collect());
    }
    let out = Out {
        names,
        codes: codes_out,
        vals: vals_out,
    };
    println!("{}", serde_json::to_string(&out).unwrap());
}
