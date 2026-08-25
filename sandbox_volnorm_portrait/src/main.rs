//! 想法B:分时段量标准化画像(vol-norm portrait)——单日全市场横截面探索(sandbox)。
//!
//! 目标序列:当天分桶量 / 历史同桶均量(非累计,围绕1.0波动),tick 继承所在桶比值。
//! 历史均量来源:volume.h5 目标日前 20 个交易日的 1 分钟 nanmean(Python 预计算 CSV)。
//!
//! 分桶配置(3 种):
//!   - 1m    : 1 分钟桶(上午120 + 下午117 = 237 桶,与 volume.h5 有效行一致)
//!   - 10s   : 10 秒桶(1422 桶),历史均量按分钟桶内均匀假设缩放 /6
//!   - sm    : 1 分钟 ratio 的 3 点中心平滑(解决阶跃函数问题)
//!
//! 挑点标准(3 种):V=单笔量 / DevA=|ratio-1| / DevV=|vol*(ratio-1)|
//! 拟合 lin/int,比较 cor/dev,N=5/10/20。
//! 价格版对照:目标=价格,标准=V,同框架。
//!
//! 构建: cd sandbox_volnorm_portrait && cargo build --release
//! 运行: ./target/release/volnorm_portrait_sandbox <date> <hm_csv> > out.json
mod fast_csv_reader;
use fast_csv_reader::{read_trade_fast, TradeRecord};
use rayon::prelude::*;
use serde::Serialize;
use std::collections::{BTreeSet, HashMap};

const N_BUCKET_1M: usize = 237;
const N_BUCKET_10S: usize = 1422;
const NS: [usize; 3] = [5, 10, 20];
const KINDS: [&str; 3] = ["1m", "10s", "sm"];
const CRITS: [&str; 3] = ["V", "DevA", "DevV"];
const FITS: [&str; 2] = ["lin", "int"];
const CMPS: [&str; 2] = ["cor", "dev"];
const N_VOL_COLS: usize = KINDS.len() * CRITS.len() * NS.len() * FITS.len() * CMPS.len(); // 108
const N_PRICE_COLS: usize = NS.len() * FITS.len() * CMPS.len(); // 12
const N_COLS: usize = N_VOL_COLS + N_PRICE_COLS;

#[derive(Serialize)]
struct Out {
    names: Vec<String>,
    codes: Vec<String>,
    vals: Vec<Vec<f32>>,
}

/// 枚举当日全市场代码。
fn list_codes(date: i64) -> Vec<String> {
    for root in ["/ssd_data/stock", "/nas197/binary/stock/sz_alpha/stock"] {
        let dir = format!("{root}/{date}/transaction");
        if let Ok(entries) = std::fs::read_dir(&dir) {
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

/// 读历史均量 CSV(code,v0..v239)。
fn load_hist(path: &str) -> HashMap<String, Vec<f64>> {
    let s = std::fs::read_to_string(path).expect("read hist csv");
    let mut m = HashMap::new();
    for line in s.lines().skip(1) {
        let mut it = line.split(',');
        let code = it.next().unwrap().to_string();
        let vals: Vec<f64> = it
            .map(|x| {
                if x == "nan" {
                    f64::NAN
                } else {
                    x.parse::<f64>().unwrap_or(f64::NAN)
                }
            })
            .collect();
        if vals.len() >= 240 {
            m.insert(code, vals);
        }
    }
    m
}

/// 反推真实钟表秒:time_sec = exchtime + 8h(exchtime 为"北京钟表-8h"的假 unix 秒),
/// 故 time_sec mod 86400 直接得到北京钟表秒;reader 已把下午(13:00-14:57)
/// 平移 -90 分钟(钟表 11:30-13:27),钟表秒 > 11:30 的再加回 90 分钟。
#[inline]
fn real_time(t_sec_abs: f64) -> f64 {
    let clock = t_sec_abs.rem_euclid(86400.0);
    if clock > 41400.0 {
        clock + 5400.0
    } else {
        clock
    }
}

/// 1 分钟桶(9:30 起第 k 个 60s;上午 0..119,下午 120..236;>=237 丢弃,<0 丢弃)。
/// 输入为 reader 的 time_sec(绝对 Unix 秒)。
#[inline]
fn bucket_1m(t_sec_abs: f64) -> i64 {
    let t = real_time(t_sec_abs);
    if t < 46800.0 {
        // 上午 [9:30, 11:30)
        let k = ((t - 34200.0) / 60.0).floor() as i64;
        k.clamp(0, 119)
    } else {
        // 下午 [13:00, 15:00)
        let k = 120 + ((t - 46800.0) / 60.0).floor() as i64;
        k.clamp(120, 236)
    }
}

/// 10 秒桶(上午 0..719,下午 720..1421;>=1422 丢弃)。
#[inline]
fn bucket_10s(t_sec_abs: f64) -> i64 {
    let t = real_time(t_sec_abs);
    if t < 46800.0 {
        let k = ((t - 34200.0) / 10.0).floor() as i64;
        k.clamp(0, 719)
    } else {
        let k = 720 + ((t - 46800.0) / 10.0).floor() as i64;
        k.clamp(720, 1421)
    }
}

// ===== 拟合(与正式库 extreme_point_fit_metrics.rs 逐字对齐)=====

#[inline]
fn linear_coeffs(pts: &[(f32, f32)]) -> (f32, f32) {
    let n = pts.len() as f64;
    let sum_t: f64 = pts.iter().map(|(t, _)| *t as f64).sum();
    let sum_y: f64 = pts.iter().map(|(_, y)| *y as f64).sum();
    let sum_tt: f64 = pts.iter().map(|(t, _)| (*t as f64) * (*t as f64)).sum();
    let sum_ty: f64 = pts.iter().map(|(t, y)| (*t as f64) * (*y as f64)).sum();
    let denom = n * sum_tt - sum_t * sum_t;
    if denom.abs() < 1e-30 {
        (0.0, (sum_y / n) as f32)
    } else {
        let a = (n * sum_ty - sum_t * sum_y) / denom;
        let b = (sum_y - a * sum_t) / n;
        (a as f32, b as f32)
    }
}

fn fit_interp_into(pts: &[(f32, f32)], eval_t: &[f32], out: &mut [f32]) {
    let mut sorted: Vec<(f32, f32)> = pts
        .iter()
        .filter(|(t, y)| t.is_finite() && y.is_finite())
        .copied()
        .collect();
    sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let np = sorted.len();
    if np == 0 {
        out.fill(f32::NAN);
        return;
    }
    if np == 1 {
        out.fill(sorted[0].1);
        return;
    }
    let mut seg = 0usize;
    for (i, &t) in eval_t.iter().enumerate() {
        if t <= sorted[0].0 {
            out[i] = sorted[0].1;
        } else if t >= sorted[np - 1].0 {
            out[i] = sorted[np - 1].1;
        } else {
            while seg < np - 2 && sorted[seg + 1].0 <= t {
                seg += 1;
            }
            let (t0, y0) = sorted[seg];
            let (t1, y1) = sorted[seg + 1];
            let dt = t1 - t0;
            out[i] = if dt > 0.0 {
                y0 + (t - t0) / dt * (y1 - y0)
            } else {
                y0
            };
        }
    }
}

/// 与正式库一致:按 score 降序预排序(要求 score/target 有限)。
fn presort_desc(scores: &[f32], target: &[f32], times: &[f32]) -> Vec<(f32, f32, usize)> {
    let mut pairs: Vec<(f32, f32, usize)> = scores
        .iter()
        .zip(target.iter())
        .zip(times.iter())
        .enumerate()
        .filter(|(_, ((s, y), _))| s.is_finite() && y.is_finite())
        .map(|(i, ((s, _), &t))| (*s, t, i))
        .collect();
    pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    pairs
}

/// 贪心去邻:归一化时间距离 < 0.5/N 的较弱候选舍弃。
fn greedy_select_dedup(sorted: &[(f32, f32, usize)], k: usize, min_gap: f32) -> Vec<usize> {
    let mut selected: Vec<usize> = Vec::with_capacity(k);
    let mut sel_times: Vec<f32> = Vec::with_capacity(k);
    for &(_, t, idx) in sorted {
        if selected.len() >= k {
            break;
        }
        let mut ok = true;
        for &st in &sel_times {
            if (t - st).abs() < min_gap {
                ok = false;
                break;
            }
        }
        if ok {
            selected.push(idx);
            sel_times.push(t);
        }
    }
    selected
}

/// 单次遍历算 (cor, dev);与正式库 compare 逻辑一致。
fn compare_lin_int(lin: &[f32], int: &[f32], target: &[f32]) -> [(f32, f32); 2] {
    let mut stats = [[0f64; 3]; 2]; // [fit][sa,saa,sab]
    let mut sb = 0f64;
    let mut sbb = 0f64;
    let mut abs_diff = [0f64; 2];
    let mut cnt = 0u32;
    for i in 0..target.len() {
        let b = target[i];
        if b.is_finite() {
            let bf = b as f64;
            sb += bf;
            sbb += bf * bf;
            let fits = [lin[i], int[i]];
            for j in 0..2 {
                let af = fits[j] as f64;
                stats[j][0] += af;
                stats[j][1] += af * af;
                stats[j][2] += af * bf;
                abs_diff[j] += (af - bf).abs();
            }
            cnt += 1;
        }
    }
    let mut result = [(f32::NAN, f32::NAN); 2];
    if cnt >= 3 {
        let nf = cnt as f64;
        for j in 0..2 {
            let cov = stats[j][2] - stats[j][0] * sb / nf;
            let vara = stats[j][1] - stats[j][0] * stats[j][0] / nf;
            let varb = sbb - sb * sb / nf;
            let denom = (vara * varb).sqrt();
            let cor = if denom < 1e-30 {
                f32::NAN
            } else {
                (cov / denom) as f32
            };
            result[j] = (cor, (abs_diff[j] / nf) as f32);
        }
    }
    result
}

/// 每股核心:返回 120 个因子值(108 量版 + 12 价格版)。
fn per_stock(trades: &[TradeRecord], hm: &[f64]) -> Vec<f32> {
    let n = trades.len();
    let mut out = vec![f32::NAN; N_COLS];
    if n < 2 {
        return out;
    }

    // 归一化时间
    let t0 = trades[0].time_sec;
    let t1 = trades[n - 1].time_sec;
    let trange = (t1 - t0).max(1e-6);
    let tn: Vec<f32> = trades
        .iter()
        .map(|t| ((t.time_sec - t0) / trange) as f32)
        .collect();

    // 桶聚合(1m + 10s 一次遍历)
    let mut ka1 = vec![0.0f64; N_BUCKET_1M];
    let mut ka10 = vec![0.0f64; N_BUCKET_10S];
    let mut b1 = vec![usize::MAX; n];
    let mut b10 = vec![usize::MAX; n];
    let mut valid = 0usize;
    for (i, t) in trades.iter().enumerate() {
        let k1 = bucket_1m(t.time_sec);
        let k10 = bucket_10s(t.time_sec);
        if (0..N_BUCKET_1M as i64).contains(&k1) {
            let k = k1 as usize;
            b1[i] = k;
            ka1[k] += t.volume;
            if k10 >= 0 && k10 < N_BUCKET_10S as i64 {
                let kk = k10 as usize;
                b10[i] = kk;
                ka10[kk] += t.volume;
            }
        }
        valid += 1;
    }
    if valid == 0 {
        return out;
    }

    // 历史均量有效桶
    let hmok: Vec<bool> = (0..N_BUCKET_1M)
        .map(|k| k < hm.len() && hm[k].is_finite() && hm[k] > 0.0)
        .collect();

    // ratio 序列(1m / 10s / 平滑)
    let r1m: Vec<f32> = (0..N_BUCKET_1M)
        .map(|k| {
            if hmok[k] {
                (ka1[k] / hm[k]) as f32
            } else {
                f32::NAN
            }
        })
        .collect();
    let r10: Vec<f32> = (0..N_BUCKET_10S)
        .map(|k| {
            let km = (k / 6).min(N_BUCKET_1M - 1);
            if hmok[km] {
                (ka10[k] / (hm[km] / 6.0)) as f32
            } else {
                f32::NAN
            }
        })
        .collect();
    let rsm: Vec<f32> = (0..N_BUCKET_1M)
        .map(|k| {
            if !hmok[k] {
                return f32::NAN;
            }
            let lo = k.saturating_sub(1);
            let hi = (k + 1).min(N_BUCKET_1M - 1);
            let mut s = 0.0f64;
            let mut c = 0u32;
            for j in lo..=hi {
                if hmok[j] {
                    s += r1m[j] as f64;
                    c += 1;
                }
            }
            if c > 0 {
                (s / c as f64) as f32
            } else {
                f32::NAN
            }
        })
        .collect();

    let r1m_ok = hmok.iter().any(|&x| x);
    let r10_ok = (0..N_BUCKET_10S).any(|k| hmok[(k / 6).min(N_BUCKET_1M - 1)]);
    let rsm_ok = rsm.iter().any(|x| x.is_finite());

    // 目标序列(tick 继承)
    let tgt_1m: Vec<f32> = b1.iter().map(|&b| if b == usize::MAX { f32::NAN } else { r1m[b] }).collect();
    let tgt_10s: Vec<f32> = b10.iter().map(|&b| if b == usize::MAX { f32::NAN } else { r10[b] }).collect();
    let tgt_sm: Vec<f32> = b1.iter().map(|&b| if b == usize::MAX { f32::NAN } else { rsm[b] }).collect();
    let tgt_p: Vec<f32> = trades.iter().map(|t| t.price as f32).collect();
    let vol: Vec<f32> = trades.iter().map(|t| t.volume as f32).collect();

    // 每 (kind, crit):presort 一次,3 个 N × 2 拟合
    let kinds: [(&str, &[f32], bool, bool); 3] = [
        ("1m", &tgt_1m, r1m_ok, false),
        ("10s", &tgt_10s, r10_ok, false),
        ("sm", &tgt_sm, rsm_ok, false),
    ];
    for (ki, (kname, tgt, tok, _is_price)) in kinds.iter().enumerate() {
        if !*tok {
            continue;
        }
        for (ci, cname) in CRITS.iter().enumerate() {
            let scores: Vec<f32> = match *cname {
                "V" => vol.clone(),
                "DevA" => tgt.iter().map(|&r| (r - 1.0).abs()).collect(),
                _ => vol.iter().zip(tgt.iter()).map(|(&v, &r)| (v * (r - 1.0)).abs()).collect(),
            };
            let sorted = presort_desc(&scores, tgt, &tn);
            if sorted.len() < 2 {
                continue;
            }
            let mut buf_lin = vec![0f32; n];
            let mut buf_int = vec![0f32; n];
            for (ni, &npts) in NS.iter().enumerate() {
                let min_gap = 0.5 / npts as f32;
                let sel = greedy_select_dedup(&sorted, npts, min_gap);
                if sel.len() < 2 {
                    continue;
                }
                let pts: Vec<(f32, f32)> = sel.iter().map(|&idx| (tn[idx], tgt[idx])).collect();
                let (la, lb) = linear_coeffs(&pts);
                for i in 0..n {
                    buf_lin[i] = la * tn[i] + lb;
                }
                fit_interp_into(&pts, &tn, &mut buf_int);
                let res = compare_lin_int(&buf_lin, &buf_int, tgt);
                for (fi, (cor, dev)) in res.iter().enumerate() {
                    let col = ki * (CRITS.len() * NS.len() * FITS.len() * CMPS.len())
                        + ci * (NS.len() * FITS.len() * CMPS.len())
                        + ni * (FITS.len() * CMPS.len())
                        + fi * CMPS.len();
                    out[col] = *cor;
                    out[col + 1] = *dev;
                }
            }
        }
    }

    // 价格版:目标=价格,标准=V
    {
        let sorted = presort_desc(&vol, &tgt_p, &tn);
        let mut buf_lin = vec![0f32; n];
        let mut buf_int = vec![0f32; n];
        for (ni, &npts) in NS.iter().enumerate() {
            let min_gap = 0.5 / npts as f32;
            let sel = greedy_select_dedup(&sorted, npts, min_gap);
            if sel.len() < 2 {
                continue;
            }
            let pts: Vec<(f32, f32)> = sel.iter().map(|&idx| (tn[idx], tgt_p[idx])).collect();
            let (la, lb) = linear_coeffs(&pts);
            for i in 0..n {
                buf_lin[i] = la * tn[i] + lb;
            }
            fit_interp_into(&pts, &tn, &mut buf_int);
            let res = compare_lin_int(&buf_lin, &buf_int, &tgt_p);
            for (fi, (cor, dev)) in res.iter().enumerate() {
                let col = N_VOL_COLS + ni * (FITS.len() * CMPS.len()) + fi * CMPS.len();
                out[col] = *cor;
                out[col + 1] = *dev;
            }
        }
    }

    out
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let date: i64 = args[1].parse().expect("用法: <date> <hm_csv>");
    let hm_csv = &args[2];
    let hm = load_hist(hm_csv);
    eprintln!("hist codes = {}", hm.len());

    let codes = list_codes(date);
    eprintln!("codes = {}", codes.len());

    let feats: Vec<Option<Vec<f32>>> = codes
        .par_iter()
        .map(|c| {
            read_trade_fast(c, date)
                .ok()
                .map(|tr| per_stock(&tr, hm.get(c).map(|v| &v[..]).unwrap_or(&[])))
        })
        .collect();

    let mut valid = Vec::new();
    let mut vals = Vec::new();
    for (c, f) in codes.iter().zip(feats.iter()) {
        if let Some(v) = f {
            if v.iter().any(|x| x.is_finite()) {
                valid.push(c.clone());
                vals.push(v.clone());
            }
        }
    }
    eprintln!("valid = {}", valid.len());

    // 列名
    let mut names = Vec::with_capacity(N_COLS);
    for k in KINDS {
        for cr in CRITS {
            for &n_ in &NS {
                for f in FITS {
                    for c in CMPS {
                        names.push(format!("{k}-{cr}-N{n_}-{f}-{c}"));
                    }
                }
            }
        }
    }
    for &n_ in &NS {
        for f in FITS {
            for c in CMPS {
                names.push(format!("price-V-N{n_}-{f}-{c}"));
            }
        }
    }
    assert_eq!(names.len(), N_COLS);

    let out = Out {
        names,
        codes: valid,
        vals,
    };
    println!("{}", serde_json::to_string(&out).unwrap());
}
