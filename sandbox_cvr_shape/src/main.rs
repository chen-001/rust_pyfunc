//! 单日全市场横截面探索：CVR/CTR 曲线形状因子（sandbox）。
//! 读取一天全市场逐笔成交 → 237 个 1 分钟桶 → 累计占比曲线 → 形状特征 →
//! 输出 JSON 给 Python 做 IC 评估。
//!
//! 构建: cd sandbox_cvr_shape && cargo build --release
//! 运行: ./target/release/cvr_shape_sandbox 20240104 > out_20240104.json
mod fast_csv_reader;
use fast_csv_reader::{read_trade_fast, TradeRecord};
use rayon::prelude::*;
use serde::Serialize;
use std::collections::BTreeSet;

const N_BUCKET: usize = 237; // 上午120 + 下午117 = 237 个 1 分钟桶
const DAY_START_SEC: f64 = 34200.0; // 9:30（adjust_afternoon 平移后）
const N_FITS: usize = 2; // 直接拟合对照: lin / int（N=10, 挑点标准=单笔量 V）
const N_PICK: usize = 10;

#[derive(Serialize)]
struct Out {
    names: Vec<String>,
    codes: Vec<String>,
    vals: Vec<Vec<f64>>,
}

fn list_codes(date: i64) -> Vec<String> {
    let dir = format!("/ssd_data/stock/{}/transaction", date);
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
    Vec::new()
}

/// 桶索引：time_us（绝对本地 epoch 微秒）→ 日内秒 → [34200, 48420) 每分钟一桶。
#[inline]
fn bucket_of(time_us: i64) -> Option<usize> {
    let t = time_us as f64 / 1e6;
    let day_sec = (t as i64).rem_euclid(86400) as f64;
    let k = ((day_sec - DAY_START_SEC) / 60.0) as i64;
    if k >= 0 && (k as usize) < N_BUCKET {
        Some(k as usize)
    } else {
        None
    }
}

/// tick 级拟合 corr：挑点标准=单笔量 V，贪心去邻（min_gap = 0.5/N），
/// 拟合 lin（最小二乘直线）/ int（分段线性插值），与 tick 级 CVR 求 Pearson。
fn fit_corrs(trades: &[TradeRecord], pick: usize) -> Option<(f64, f64)> {
    let n = trades.len();
    if n < 30 {
        return None;
    }
    let t_min = trades[0].time_sec;
    let t_max = trades[n - 1].time_sec;
    let t_range = (t_max - t_min).max(1e-6);

    // 目标序列：tick 级 CVR（累计量占比）
    let mut cum = 0.0f64;
    let mut total = 0.0f64;
    for t in trades {
        total += t.volume;
    }
    if total <= 0.0 {
        return None;
    }
    // 单次遍历构造 (x=归一化时间, y=CVR, score=volume)
    let mut items: Vec<(f64, f64, f64)> = Vec::with_capacity(n);
    for t in trades {
        cum += t.volume;
        let x = (t.time_sec - t_min) / t_range;
        items.push((x, cum / total, t.volume));
    }
    // 按 score 降序预排序（去邻用）
    let mut order: Vec<usize> = (0..n).collect();
    order.retain(|&i| items[i].2.is_finite() && items[i].1.is_finite());
    order.sort_by(|&i, &j| items[j].2.partial_cmp(&items[i].2).unwrap_or(std::cmp::Ordering::Equal));
    let min_gap = 0.5 / pick as f64;
    let mut selected: Vec<(f64, f64)> = Vec::with_capacity(pick);
    let mut sel_t: Vec<f64> = Vec::with_capacity(pick);
    for &i in &order {
        if selected.len() >= pick {
            break;
        }
        let x = items[i].0;
        let mut ok = true;
        for &st in &sel_t {
            if (x - st).abs() < min_gap {
                ok = false;
                break;
            }
        }
        if ok {
            selected.push((x, items[i].1));
            sel_t.push(x);
        }
    }
    if selected.len() < 2 {
        return None;
    }
    let pts = &selected;

    // lin: 最小二乘直线 y = a + b*x
    let np = pts.len() as f64;
    let (sx, sy, sxx, sxy): (f64, f64, f64, f64) =
        pts.iter().fold((0.0, 0.0, 0.0, 0.0), |acc, p| {
            (acc.0 + p.0, acc.1 + p.1, acc.2 + p.0 * p.0, acc.3 + p.0 * p.1)
        });
    let denom = np * sxx - sx * sx;
    let b = if denom.abs() < 1e-20 { 0.0 } else { (np * sxy - sx * sy) / denom };
    let a = (sy - b * sx) / np;

    // 内插 pts 分段线性所需的排序（按 x 升序）
    let mut sp: Vec<(f64, f64)> = pts.to_vec();
    sp.sort_by(|p, q| p.0.partial_cmp(&q.0).unwrap());

    // 一次遍历同时算 lin/int 的 corr
    let mut cnt = 0.0f64;
    let (mut sa, mut saa, mut sab, mut sb, mut sbb) = (0.0, 0.0, 0.0, 0.0, 0.0);
    let (mut sa2, mut saa2, mut sab2) = (0.0, 0.0, 0.0);
    let mut seg = 0usize;
    let m = sp.len();
    for (x, y) in items.iter().map(|it| (it.0, it.1)) {
        let fit_lin = a + b * x;
        let fit_int = if x <= sp[0].0 {
            sp[0].1
        } else if x >= sp[m - 1].0 {
            sp[m - 1].1
        } else {
            while seg < m - 2 && sp[seg + 1].0 <= x {
                seg += 1;
            }
            let (x0, y0) = sp[seg];
            let (x1, y1) = sp[seg + 1];
            let dt = x1 - x0;
            if dt > 0.0 {
                y0 + (x - x0) / dt * (y1 - y0)
            } else {
                y0
            }
        };
        sa += fit_lin;
        saa += fit_lin * fit_lin;
        sab += fit_lin * y;
        sa2 += fit_int;
        saa2 += fit_int * fit_int;
        sab2 += fit_int * y;
        sb += y;
        sbb += y * y;
        cnt += 1.0;
    }
    let corr = |s_a: f64, s_aa: f64, s_ab: f64| -> f64 {
        let cov = s_ab - s_a * sb / cnt;
        let va = s_aa - s_a * s_a / cnt;
        let vb = sbb - sb * sb / cnt;
        let den = (va * vb).sqrt();
        if den < 1e-30 {
            f64::NAN
        } else {
            cov / den
        }
    };
    Some((corr(sa, saa, sab), corr(sa2, saa2, sab2)))
}

/// CVR 的 tick 级 corr 与桶级形状因子同步计算。
/// 返回 24 个因子：0..23 见 names()。
fn per_stock(trades: &[TradeRecord]) -> Option<Vec<f64>> {
    if trades.len() < 100 {
        return None;
    }
    let mut vol_b = vec![0.0f64; N_BUCKET];
    let mut amt_b = vec![0.0f64; N_BUCKET];
    let mut total_vol = 0.0f64;
    let mut total_amt = 0.0f64;
    for t in trades {
        if let Some(k) = bucket_of(t.time_us) {
            vol_b[k] += t.volume;
            amt_b[k] += t.turnover;
            total_vol += t.volume;
            total_amt += t.turnover;
        }
    }
    if total_vol <= 0.0 || total_amt <= 0.0 {
        return None;
    }

    // 桶级累计占比曲线
    let mut cvr = vec![0.0f64; N_BUCKET];
    let mut ctr = vec![0.0f64; N_BUCKET];
    let mut cv = 0.0f64;
    let mut ca = 0.0f64;
    for k in 0..N_BUCKET {
        cv += vol_b[k] / total_vol;
        ca += amt_b[k] / total_amt;
        cvr[k] = cv;
        ctr[k] = ca;
    }

    // 对角线（均匀假设）：(k+1)/237
    let mut area_dev = 0.0f64;
    let mut mean_dev = 0.0f64;
    let mut max_dev_s = f64::NEG_INFINITY;
    let mut max_dev_a = 0.0f64;
    let mut dcv_mean = 0.0f64;
    let mut dcv_abs = 0.0f64;
    let mut dcv_max = f64::NEG_INFINITY;
    let mut dcv_min = f64::INFINITY;
    let mut ctr_area = 0.0f64;
    for k in 0..N_BUCKET {
        let diag = (k + 1) as f64 / N_BUCKET as f64;
        let dev = cvr[k] - diag;
        area_dev += dev.abs();
        mean_dev += dev;
        if dev > max_dev_s {
            max_dev_s = dev;
        }
        if dev.abs() > max_dev_a {
            max_dev_a = dev.abs();
        }
        let d = cvr[k] - ctr[k];
        dcv_mean += d;
        dcv_abs += d.abs();
        if d > dcv_max {
            dcv_max = d;
        }
        if d < dcv_min {
            dcv_min = d;
        }
        ctr_area += (ctr[k] - diag).abs();
    }
    area_dev /= N_BUCKET as f64;
    mean_dev /= N_BUCKET as f64;
    dcv_mean /= N_BUCKET as f64;
    dcv_abs /= N_BUCKET as f64;
    ctr_area /= N_BUCKET as f64;

    // 分段累计占比
    let cum_open30 = cvr[29];
    let cum_open60 = cvr[59];
    let cum_morning = cvr[119];
    let cum_afternoon = 1.0 - cvr[119];
    let cum_close10 = 1.0 - cvr[226];
    let cum_close30 = 1.0 - cvr[206];

    // 完成时刻（桶内线性插值，归一化到 [0,1]）
    let pct_time = |p: f64| -> f64 {
        let mut prev = 0.0f64;
        for k in 0..N_BUCKET {
            if cvr[k] >= p {
                let frac = cvr[k] - prev;
                let t = if frac > 1e-12 { (k as f64 + (p - prev) / frac) / N_BUCKET as f64 } else { (k + 1) as f64 / N_BUCKET as f64 };
                return t.min(1.0);
            }
            prev = cvr[k];
        }
        1.0
    };
    let t50 = pct_time(0.5);
    let t90 = pct_time(0.9);

    // 集中度
    let mut sh = vec![0.0f64; N_BUCKET];
    for k in 0..N_BUCKET {
        sh[k] = vol_b[k] / total_vol;
    }
    let hhi: f64 = sh.iter().map(|&x| x * x).sum();
    let mut sh_sorted = sh.clone();
    sh_sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
    let top1 = sh_sorted[0];
    let top5: f64 = sh_sorted.iter().take(5).sum();

    // tick 级直接拟合对照（CVR）
    let (cvr_lin, cvr_int) = fit_corrs(trades, N_PICK)?;
    // CTR 的 tick 级拟合（同挑点标准 V）
    let (ctr_lin, ctr_int) = fit_corrs_amt(trades, N_PICK)?;

    let mut out = Vec::with_capacity(26);
    out.push(area_dev);
    out.push(mean_dev);
    out.push(max_dev_s);
    out.push(max_dev_a);
    out.push(cum_open30);
    out.push(cum_open60);
    out.push(cum_morning);
    out.push(cum_afternoon);
    out.push(cum_close10);
    out.push(cum_close30);
    out.push(t50);
    out.push(t90);
    out.push(hhi);
    out.push(top1);
    out.push(top5);
    out.push(dcv_mean);
    out.push(dcv_abs);
    out.push(dcv_max);
    out.push(dcv_min);
    out.push(ctr_area);
    out.push(cvr_lin);
    out.push(cvr_int);
    out.push(ctr_lin);
    out.push(ctr_int);
    Some(out)
}

/// CTR 的 tick 级拟合 corr（挑选标准同样用单笔量 V，目标=累计成交额占比）。
fn fit_corrs_amt(trades: &[TradeRecord], pick: usize) -> Option<(f64, f64)> {
    let n = trades.len();
    if n < 30 {
        return None;
    }
    let t_min = trades[0].time_sec;
    let t_max = trades[n - 1].time_sec;
    let t_range = (t_max - t_min).max(1e-6);
    let total: f64 = trades.iter().map(|t| t.turnover).sum();
    if total <= 0.0 {
        return None;
    }
    let mut cum = 0.0f64;
    let mut items: Vec<(f64, f64, f64)> = Vec::with_capacity(n);
    for t in trades {
        cum += t.turnover;
        let x = (t.time_sec - t_min) / t_range;
        items.push((x, cum / total, t.volume));
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.retain(|&i| items[i].1.is_finite());
    order.sort_by(|&i, &j| items[j].2.partial_cmp(&items[i].2).unwrap_or(std::cmp::Ordering::Equal));
    let min_gap = 0.5 / pick as f64;
    let mut selected: Vec<(f64, f64)> = Vec::with_capacity(pick);
    let mut sel_t: Vec<f64> = Vec::with_capacity(pick);
    for &i in &order {
        if selected.len() >= pick {
            break;
        }
        let x = items[i].0;
        let mut ok = true;
        for &st in &sel_t {
            if (x - st).abs() < min_gap {
                ok = false;
                break;
            }
        }
        if ok {
            selected.push((x, items[i].1));
            sel_t.push(x);
        }
    }
    if selected.len() < 2 {
        return None;
    }
    let pts = &selected;
    let np = pts.len() as f64;
    let (sx, sy, sxx, sxy): (f64, f64, f64, f64) =
        pts.iter().fold((0.0, 0.0, 0.0, 0.0), |acc, p| {
            (acc.0 + p.0, acc.1 + p.1, acc.2 + p.0 * p.0, acc.3 + p.0 * p.1)
        });
    let denom = np * sxx - sx * sx;
    let b = if denom.abs() < 1e-20 { 0.0 } else { (np * sxy - sx * sy) / denom };
    let a = (sy - b * sx) / np;
    let mut sp = pts.to_vec();
    sp.sort_by(|p, q| p.0.partial_cmp(&q.0).unwrap());
    let mut cnt = 0.0f64;
    let (mut sa, mut saa, mut sab, mut sb, mut sbb) = (0.0, 0.0, 0.0, 0.0, 0.0);
    let (mut sa2, mut saa2, mut sab2) = (0.0, 0.0, 0.0);
    let mut seg = 0usize;
    let m = sp.len();
    for (x, y) in items.iter().map(|it| (it.0, it.1)) {
        let fit_lin = a + b * x;
        let fit_int = if x <= sp[0].0 {
            sp[0].1
        } else if x >= sp[m - 1].0 {
            sp[m - 1].1
        } else {
            while seg < m - 2 && sp[seg + 1].0 <= x {
                seg += 1;
            }
            let (x0, y0) = sp[seg];
            let (x1, y1) = sp[seg + 1];
            let dt = x1 - x0;
            if dt > 0.0 {
                y0 + (x - x0) / dt * (y1 - y0)
            } else {
                y0
            }
        };
        sa += fit_lin;
        saa += fit_lin * fit_lin;
        sab += fit_lin * y;
        sa2 += fit_int;
        saa2 += fit_int * fit_int;
        sab2 += fit_int * y;
        sb += y;
        sbb += y * y;
        cnt += 1.0;
    }
    let corr = |s_a: f64, s_aa: f64, s_ab: f64| -> f64 {
        let cov = s_ab - s_a * sb / cnt;
        let va = s_aa - s_a * s_a / cnt;
        let vb = sbb - sb * sb / cnt;
        let den = (va * vb).sqrt();
        if den < 1e-30 {
            f64::NAN
        } else {
            cov / den
        }
    };
    Some((corr(sa, saa, sab), corr(sa2, saa2, sab2)))
}

fn names() -> Vec<String> {
    vec![
        "area_dev".into(),
        "mean_dev".into(),
        "max_dev".into(),
        "max_dev_abs".into(),
        "cum_open30".into(),
        "cum_open60".into(),
        "cum_morning".into(),
        "cum_afternoon".into(),
        "cum_close10".into(),
        "cum_close30".into(),
        "t50".into(),
        "t90".into(),
        "hhi".into(),
        "top1".into(),
        "top5".into(),
        "dcv_mean".into(),
        "dcv_abs".into(),
        "dcv_max".into(),
        "dcv_min".into(),
        "ctr_area".into(),
        "cvr_lin_cor".into(),
        "cvr_int_cor".into(),
        "ctr_lin_cor".into(),
        "ctr_int_cor".into(),
    ]
}

fn main() {
    let date: i64 = std::env::args()
        .nth(1)
        .expect("用法: cvr_shape_sandbox <date>")
        .parse()
        .unwrap();
    let codes = list_codes(date);
    eprintln!("codes = {}", codes.len());

    let feats: Vec<Option<Vec<f64>>> = codes
        .par_iter()
        .map(|c| read_trade_fast(c, date).ok().and_then(|t| per_stock(&t)))
        .collect();

    let mut valid = Vec::new();
    let mut mat = Vec::new();
    for (c, f) in codes.iter().zip(feats.iter()) {
        if let Some(v) = f {
            valid.push(c.clone());
            mat.push(v.clone());
        }
    }
    eprintln!("valid = {}", valid.len());

    let out = Out {
        names: names(),
        codes: valid,
        vals: mat,
    };
    println!("{}", serde_json::to_string(&out).unwrap());
}
