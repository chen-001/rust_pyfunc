//! 单日全市场「高峰-小峰」三方向因子探索（sandbox_cs_peak v3）
//!
//! 同一次全市场读取，对每只股票同时计算 5 套高峰-小峰定义（局部极大值判定共用）：
//!   S0 个股版  ：高峰线 = 每股内部 top1% 分位，小峰线 = 每股内部 top10%（= hm64 原版）
//!   C1 截面1%  ：高峰线 = 全市场成交量 top1%，小峰线 = top10%
//!   C2 截面额1%：高峰线 = 全市场成交额 top1%，小峰线 = top10%（资金冲击视角，消除价格偏置）
//!   C3 截面0.1%：高峰线 = 全市场成交量 top0.1%，小峰线 = top1%（极端事件金字塔顶层）
//!   C4 截面0.5%：高峰线 = 全市场成交量 top0.5%，小峰线 = top5%（金字塔中层）
//! 每套输出 n_peaks + get_res 110 统计量（与 hm64 的 pandas get_res 同构）。
//!
//! 交互量（截面版 × 单股版逻辑融合）：
//!   n_both            共振高峰数（S0 ∩ C1）
//!   diff_stats[51]    共振高峰上「个股尺度小峰 17 特征 − 市场尺度小峰 17 特征」的 mean/std/skew
//!   rel_strength[3]   C1 高峰量 / 每股内部 top1% 阈值 的 mean/std/skew（冲击的相对突兀程度）
//!   imp_vol_ratio     C1 高峰量总和 / 当日总成交量（规模中性化冲击强度）
//!   imp_turn_ratio    C1 高峰额总和 / 当日总成交额
//!
//! 市场状态量（顶层输出）：全市场笔数/量/额、主买主卖资金、volume/turnover 多级分位族。
//!
//! 构建: cd sandbox_cs_peak && cargo build --release
//! 运行: ./target/release/cs_peak_sandbox 20260717 > out.json
//! 日志走 stderr，结果走 stdout，重定向只收 JSON。

mod fast_csv_reader;
use fast_csv_reader::{read_trade_fast, TradeRecord};
use rayon::prelude::*;
use serde::Serialize;
use std::collections::BTreeSet;
use std::sync::atomic::{AtomicU64, Ordering};

const SECONDS: i64 = 30;             // 时间窗口（秒），与 hm64 一致
const MAX_BUCKETS: u64 = 2_000_000;  // 直方图桶数上限（16MB）
const TURN_BUCKET: u64 = 1_000;      // 成交额桶宽（元），覆盖 20 亿元
const SEC_US: i64 = SECONDS * 1_000_000;

// ---------------- 输出结构 ----------------
#[derive(Serialize, Clone)]
struct SetOut {
    n_peaks: usize,
    s: Vec<f64>, // get_res 110 统计量
}

#[derive(Serialize)]
struct StockOut {
    code: String,
    n_trades: usize,
    tot_vol: f64,
    tot_turnover: f64,
    ps_tier1: f64, // 每股内部 top1% 阈值
    ps_tier2: f64, // 每股内部 top10% 阈值
    s0: SetOut,
    c1: SetOut,
    c2: SetOut,
    c3: SetOut,
    c4: SetOut,
    n_both: usize,
    diff_stats: Vec<f64>, // 51 = 17 差异 × (mean,std,skew)
    rel_strength: Vec<f64>, // 3
    imp_vol_ratio: f64,
    imp_turn_ratio: f64,
}

#[derive(Serialize)]
struct Out {
    date: i64,
    n_stocks: usize,
    n_with_data: usize,
    n_total_trades: u64,
    tot_vol: f64,
    tot_turnover: f64,
    buy_vol: f64, // 主买(66)手数
    sell_vol: f64, // 主卖(83)手数
    quants_vol: Vec<f64>,  // [0.001,0.005,0.01,0.02,0.05,0.1]
    quants_turn: Vec<f64>, // [0.01, 0.1]
    stocks: Vec<StockOut>,
}

// ---------------- 枚举当日全市场代码 ----------------
fn list_codes(date: i64, subdir: &str) -> Vec<String> {
    for root in ["/ssd_data/stock", "/nas197/binary/stock/sz_alpha/stock"] {
        let dir = format!("{root}/{date}/{subdir}");
        if let Ok(entries) = std::fs::read_dir(&dir) {
            let mut set = BTreeSet::new();
            for e in entries.flatten() {
                if let Some(name) = e.file_name().to_str() {
                    if let Some(code) = name.split('_').next() {
                        if code.bytes().all(|b| b.is_ascii_digit()) && code.len() == 6 {
                            set.insert(code.to_string());
                        }
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

// ---------------- 第一遍：全市场直方图（volume + turnover） ----------------
fn fill_histograms(
    code: &str, date: i64,
    hv: &[AtomicU64], ht: &[AtomicU64],
    n_total: &AtomicU64, buy_vol: &AtomicU64, sell_vol: &AtomicU64,
) {
    let filename = format!("{code}_{date}_transaction.csv");
    let path = resolve_any_path(date, &filename);
    let content = match std::fs::read(&path) {
        Ok(c) => c,
        Err(_) => return,
    };
    let bytes = &content;
    let mut start = 0usize;
    let mut first = true;
    for i in 0..bytes.len() {
        if bytes[i] == b'\n' {
            let line = &bytes[start..i];
            let line = if line.last() == Some(&b'\r') { &line[..line.len() - 1] } else { line };
            if !first {
                if let Some((v, t, flag)) = parse_line_vt(line) {
                    let bv = (v.round() as u64).min(MAX_BUCKETS - 1);
                    hv[bv as usize].fetch_add(1, Ordering::Relaxed);
                    let bt = ((t / TURN_BUCKET as f64).floor() as u64).min(MAX_BUCKETS - 1);
                    ht[bt as usize].fetch_add(1, Ordering::Relaxed);
                    n_total.fetch_add(1, Ordering::Relaxed);
                    match flag {
                        66 => { buy_vol.fetch_add(v.round() as u64, Ordering::Relaxed); }
                        83 => { sell_vol.fetch_add(v.round() as u64, Ordering::Relaxed); }
                        _ => {}
                    }
                }
            }
            first = false;
            start = i + 1;
        }
    }
}

fn resolve_any_path(date: i64, filename: &str) -> String {
    for root in ["/ssd_data/stock", "/nas197/binary/stock/sz_alpha/stock"] {
        let p = format!("{root}/{date}/transaction/{filename}");
        if std::path::Path::new(&p).exists() {
            return p;
        }
    }
    format!("/ssd_data/stock/{date}/transaction/{filename}")
}

/// 轻量解析一行：过滤 flag==32 与非盘中，返回 (volume, turnover, flag)（与 parse_line 同口径）
fn parse_line_vt(line: &[u8]) -> Option<(f64, f64, i32)> {
    if line.is_empty() || line[0] == b's' {
        return None; // 表头
    }
    let mut fields: [&[u8]; 15] = [&[][..]; 15];
    let mut col = 0;
    let mut s = 0;
    for (i, &b) in line.iter().enumerate() {
        if b == b',' {
            if col < 15 { fields[col] = &line[s..i]; }
            col += 1;
            s = i + 1;
        }
    }
    if col < 15 { fields[col] = &line[s..]; }
    if fields[10] == b"32" {
        return None; // 撤单
    }
    let exchtime_us = parse_i64_fast(fields[4]);
    let day_offset = ((exchtime_us / 1_000_000) + 8 * 3600).rem_euclid(86400);
    let in_morning = day_offset >= 9 * 3600 + 30 * 60 && day_offset <= 11 * 3600 + 30 * 60;
    let in_afternoon = day_offset >= 13 * 3600 && day_offset <= 14 * 3600 + 57 * 60;
    if !in_morning && !in_afternoon {
        return None;
    }
    let flag = parse_i64_fast(fields[10]) as i32;
    Some((parse_f64_fast(fields[8]), parse_f64_fast(fields[9]), flag))
}

#[inline]
fn parse_i64_fast(bytes: &[u8]) -> i64 {
    let mut neg = false;
    let mut i = 0;
    while i < bytes.len() && bytes[i] == b' ' { i += 1; }
    if i < bytes.len() && (bytes[i] == b'-' || bytes[i] == b'+') { neg = bytes[i] == b'-'; i += 1; }
    let mut val: i64 = 0;
    while i < bytes.len() && bytes[i] >= b'0' && bytes[i] <= b'9' {
        val = val * 10 + (bytes[i] - b'0') as i64;
        i += 1;
    }
    if neg { -val } else { val }
}

#[inline]
fn parse_f64_fast(bytes: &[u8]) -> f64 {
    let mut int_part = 0.0f64;
    let mut i = 0;
    let mut neg = false;
    if i < bytes.len() && bytes[i] == b'-' { neg = true; i += 1; }
    while i < bytes.len() && bytes[i] >= b'0' && bytes[i] <= b'9' {
        int_part = int_part * 10.0 + (bytes[i] - b'0') as f64;
        i += 1;
    }
    let mut frac = 0.0f64;
    let mut scale = 1.0f64;
    if i < bytes.len() && bytes[i] == b'.' {
        i += 1;
        while i < bytes.len() && bytes[i] >= b'0' && bytes[i] <= b'9' {
            frac = frac * 10.0 + (bytes[i] - b'0') as f64;
            scale *= 10.0;
            i += 1;
        }
    }
    let mut r = int_part + frac / scale;
    if neg { r = -r; }
    r
}

/// 从直方图求 q 分位（从高到低累计，桶宽 bw）
fn quantile(hist: &[AtomicU64], n_total: u64, q: f64, bw: u64) -> f64 {
    let target = n_total as f64 * q;
    let mut cum = 0u64;
    for b in (0..MAX_BUCKETS as usize).rev() {
        cum += hist[b].load(Ordering::Relaxed);
        if cum as f64 >= target {
            return (b as u64 * bw) as f64;
        }
    }
    0.0
}

// ---------------- 第二遍：per-stock 5 套高峰 ----------------
fn local_max_flags(trades: &[TradeRecord], key: fn(&TradeRecord) -> f64) -> (Vec<bool>, Vec<bool>) {
    let n = trades.len();
    let mut left = vec![false; n];
    let mut right = vec![false; n];
    let mut dq: Vec<usize> = Vec::with_capacity(1024);
    for i in 0..n {
        let t_i = trades[i].time_us;
        while let Some(&f) = dq.first() {
            if t_i - trades[f].time_us > SEC_US { dq.remove(0); } else { break; }
        }
        if let Some(&j) = dq.first() {
            if key(&trades[j]) > key(&trades[i]) { left[i] = true; }
        }
        while let Some(&b) = dq.last() {
            if key(&trades[b]) <= key(&trades[i]) { dq.pop(); } else { break; }
        }
        dq.push(i);
    }
    dq.clear();
    for i in (0..n).rev() {
        let t_i = trades[i].time_us;
        while let Some(&f) = dq.first() {
            if trades[f].time_us - t_i > SEC_US { dq.remove(0); } else { break; }
        }
        if let Some(&j) = dq.first() {
            if key(&trades[j]) > key(&trades[i]) { right[i] = true; }
        }
        while let Some(&b) = dq.last() {
            if key(&trades[b]) <= key(&trades[i]) { dq.pop(); } else { break; }
        }
        dq.push(i);
    }
    (left, right)
}

fn vol_key(t: &TradeRecord) -> f64 { t.volume }
fn turn_key(t: &TradeRecord) -> f64 { t.turnover }

/// 单套高峰累加器
struct SetAcc {
    rows: Vec<Vec<f64>>,
    n_peaks: usize,
}
impl SetAcc {
    fn new() -> Self { SetAcc { rows: Vec::new(), n_peaks: 0 } }
    fn push(&mut self, feats: Vec<f64>) {
        self.rows.push(feats);
        self.n_peaks += 1;
    }
    fn finish(self) -> SetOut {
        SetOut { n_peaks: self.n_peaks, s: get_res(&self.rows) }
    }
}

fn per_stock(
    trades: &[TradeRecord],
    t01_v: f64, t05_v: f64, t1_v: f64, t5_v: f64, t10_v: f64,
    t1_t: f64, t10_t: f64,
) -> Option<StockOut> {
    let n = trades.len();
    if n < 30 {
        return None;
    }

    // 每股内部阈值（与 hm64 相同的降序分位取法）
    let mut vols: Vec<f64> = trades.iter().map(|t| t.volume).collect();
    let k1 = ((n as f64 * 0.01) as usize).min(n - 1);
    let idx1 = n - 1 - k1;
    vols.select_nth_unstable_by(idx1, |a, b| a.total_cmp(b));
    let ps_tier1 = vols[idx1];
    let k2 = ((n as f64 * 0.10) as usize).min(n - 1);
    let idx2 = n - 1 - k2;
    vols.select_nth_unstable_by(idx2, |a, b| a.total_cmp(b));
    let ps_tier2 = vols[idx2];

    // 局部极大值（volume 尺度与 turnover 尺度各一套，互不干扰）
    let (vl, vr) = local_max_flags(trades, vol_key);
    let (tl, tr) = local_max_flags(trades, turn_key);

    let mut tot_vol = 0.0f64;
    let mut tot_turn = 0.0f64;
    for t in trades {
        tot_vol += t.volume;
        tot_turn += t.turnover;
    }

    let mut s0 = SetAcc::new();
    let mut c1 = SetAcc::new();
    let mut c2 = SetAcc::new();
    let mut c3 = SetAcc::new();
    let mut c4 = SetAcc::new();
    let mut n_both = 0usize;
    let mut diff_rows: Vec<Vec<f64>> = Vec::new(); // 共振高峰：个股尺度 17 − 市场尺度 17
    let mut rel_vals: Vec<f64> = Vec::new();       // C1 高峰量 / ps_tier1
    let mut c1_vol_sum = 0.0f64;
    let mut c1_turn_sum = 0.0f64;

    for i in 0..n {
        let v = trades[i].volume;
        let t = trades[i].turnover;
        let is_vmax = !vl[i] && !vr[i];
        let is_tmax = !tl[i] && !tr[i];
        let hit_s0 = is_vmax && v >= ps_tier1;
        let hit_c1 = is_vmax && v >= t1_v;
        let hit_c2 = is_tmax && t >= t1_t;
        let hit_c3 = is_vmax && v >= t01_v;
        let hit_c4 = is_vmax && v >= t05_v;
        if !hit_s0 && !hit_c1 && !hit_c2 && !hit_c3 && !hit_c4 {
            continue;
        }

        // 一次扫描收集窗口内所有点 (vol, turn, dt_sec)
        let t_i = trades[i].time_us;
        let mut ws: Vec<(f64, f64, f64)> = Vec::new();
        for j in (i + 1)..n {
            let dt = trades[j].time_us - t_i;
            if dt > SEC_US { break; }
            ws.push((trades[j].volume, trades[j].turnover, dt as f64 / 1_000_000.0));
        }

        // 各套按自己的阈值过滤小峰
        if hit_s0 {
            let mut mp: Vec<f64> = Vec::new();
            let mut td: Vec<f64> = Vec::new();
            for &(wv, _, dt) in &ws {
                if wv >= ps_tier2 { mp.push(wv); td.push(dt); }
            }
            s0.push(calculate_features(v, &mp, &td));
        }
        if hit_c1 {
            let mut mp: Vec<f64> = Vec::new();
            let mut td: Vec<f64> = Vec::new();
            for &(wv, _, dt) in &ws {
                if wv >= t10_v { mp.push(wv); td.push(dt); }
            }
            c1.push(calculate_features(v, &mp, &td));
            c1_vol_sum += v;
            c1_turn_sum += t;
            rel_vals.push(v / ps_tier1);
        }
        if hit_c2 {
            let mut mp: Vec<f64> = Vec::new();
            let mut td: Vec<f64> = Vec::new();
            for &(_, wt, dt) in &ws {
                if wt >= t10_t { mp.push(wt); td.push(dt); }
            }
            c2.push(calculate_features(t, &mp, &td)); // 资金尺度：高峰量用成交额
        }
        if hit_c3 {
            let mut mp: Vec<f64> = Vec::new();
            let mut td: Vec<f64> = Vec::new();
            for &(wv, _, dt) in &ws {
                if wv >= t1_v { mp.push(wv); td.push(dt); }
            }
            c3.push(calculate_features(v, &mp, &td));
        }
        if hit_c4 {
            let mut mp: Vec<f64> = Vec::new();
            let mut td: Vec<f64> = Vec::new();
            for &(wv, _, dt) in &ws {
                if wv >= t5_v { mp.push(wv); td.push(dt); }
            }
            c4.push(calculate_features(v, &mp, &td));
        }

        // 共振高峰（S0 ∩ C1）：双尺度跟随差异
        if hit_s0 && hit_c1 {
            n_both += 1;
            let mut mp_s: Vec<f64> = Vec::new();
            let mut td_s: Vec<f64> = Vec::new();
            let mut mp_c: Vec<f64> = Vec::new();
            let mut td_c: Vec<f64> = Vec::new();
            for &(wv, _, dt) in &ws {
                if wv >= ps_tier2 { mp_s.push(wv); td_s.push(dt); }
                if wv >= t10_v { mp_c.push(wv); td_c.push(dt); }
            }
            let fs = calculate_features(v, &mp_s, &td_s);
            let fc = calculate_features(v, &mp_c, &td_c);
            diff_rows.push(fs.iter().zip(fc.iter()).map(|(a, b)| a - b).collect());
        }
    }

    let code = String::new();
    Some(StockOut {
        code,
        n_trades: n,
        tot_vol,
        tot_turnover: tot_turn,
        ps_tier1,
        ps_tier2,
        s0: s0.finish(),
        c1: c1.finish(),
        c2: c2.finish(),
        c3: c3.finish(),
        c4: c4.finish(),
        n_both,
        diff_stats: col_stats_3(&diff_rows), // 51
        rel_strength: stats3(&rel_vals),      // 3
        imp_vol_ratio: if tot_vol > 0.0 { c1_vol_sum / tot_vol } else { 0.0 },
        imp_turn_ratio: if tot_turn > 0.0 { c1_turn_sum / tot_turn } else { 0.0 },
    })
}

/// 每列 mean/std/skew（3 个统计 × 列数）
fn col_stats_3(rows: &[Vec<f64>]) -> Vec<f64> {
    if rows.is_empty() {
        return vec![f64::NAN; 51];
    }
    let ncols = rows[0].len();
    let mut out = Vec::with_capacity(ncols * 3);
    for j in 0..ncols {
        let col: Vec<f64> = rows.iter().map(|r| r[j]).collect();
        let mean = col.iter().sum::<f64>() / col.len() as f64;
        let std = calculate_std(&col, mean);
        out.push(mean);
        out.push(std);
        out.push(calculate_skewness(&col, mean, std));
    }
    out
}

/// 一列数的 mean/std/skew
fn stats3(x: &[f64]) -> Vec<f64> {
    if x.is_empty() {
        return vec![f64::NAN; 3];
    }
    let mean = x.iter().sum::<f64>() / x.len() as f64;
    let std = calculate_std(x, mean);
    vec![mean, std, calculate_skewness(x, mean, std)]
}

// ---------------- get_res：N×17 矩阵 → 110 统计量/股 ----------------
fn get_res(m: &[Vec<f64>]) -> Vec<f64> {
    let mut out = vec![f64::NAN; 110];
    if m.is_empty() {
        return out;
    }
    let pairs: [(usize, usize); 8] = [(0, 3), (1, 3), (2, 3), (6, 11), (7, 12), (8, 13), (9, 14), (10, 15)];
    for (k, &(a, b)) in pairs.iter().enumerate() {
        let x: Vec<f64> = m.iter().map(|r| r[a]).collect();
        let y: Vec<f64> = m.iter().map(|r| r[b]).collect();
        out[k] = correlation(&x, &y);
    }
    for j in 0..17 {
        let col: Vec<f64> = m.iter().map(|r| r[j]).collect();
        let mean = col.iter().sum::<f64>() / col.len() as f64;
        let std = calculate_std(&col, mean);
        out[8 + j] = mean;
        out[8 + 17 + j] = std;
        out[8 + 34 + j] = calculate_skewness(&col, mean, std);
        out[8 + 51 + j] = calculate_kurtosis(&col, mean, std);
        out[8 + 68 + j] = calculate_autocorr(&col);
        out[8 + 85 + j] = calculate_trend(&col);
    }
    out
}

// ---------------- 17 特征计算（与正式库 trade_peak_analysis.rs 逐字一致） ----------------
fn calculate_features(peak_volume: f64, minor_peaks: &[f64], time_diffs: &[f64]) -> Vec<f64> {
    let mut features = vec![0.0; 17];
    if minor_peaks.is_empty() {
        return features;
    }
    let n = minor_peaks.len();
    let minor_sum: f64 = minor_peaks.iter().sum();
    let minor_mean = minor_sum / n as f64;
    features[0] = minor_sum / peak_volume;
    features[1] = minor_mean / peak_volume;
    features[2] = n as f64;
    let time_mean: f64 = time_diffs.iter().sum::<f64>() / n as f64;
    features[3] = time_mean;
    features[4] = correlation(minor_peaks, time_diffs);
    features[5] = dtw_distance_simple(minor_peaks, time_diffs);
    let minor_std = calculate_std(minor_peaks, minor_mean);
    features[6] = if minor_mean.abs() > f64::EPSILON { minor_std / minor_mean } else { 0.0 };
    features[7] = calculate_skewness(minor_peaks, minor_mean, minor_std);
    features[8] = calculate_kurtosis(minor_peaks, minor_mean, minor_std);
    features[9] = calculate_trend(minor_peaks);
    features[10] = calculate_autocorr(minor_peaks);
    let time_std = calculate_std(time_diffs, time_mean);
    features[11] = if time_mean.abs() > f64::EPSILON { time_std / time_mean } else { 0.0 };
    features[12] = calculate_skewness(time_diffs, time_mean, time_std);
    features[13] = calculate_kurtosis(time_diffs, time_mean, time_std);
    features[14] = calculate_trend(time_diffs);
    features[15] = calculate_autocorr(time_diffs);
    features[16] = calculate_volume_weighted_time_distance(minor_peaks, time_diffs);
    features
}

fn correlation(x: &[f64], y: &[f64]) -> f64 {
    if x.len() != y.len() || x.len() < 2 {
        return 0.0;
    }
    let n = x.len() as f64;
    let x_mean = x.iter().sum::<f64>() / n;
    let y_mean = y.iter().sum::<f64>() / n;
    let (mut cov, mut var_x, mut var_y) = (0.0, 0.0, 0.0);
    for i in 0..x.len() {
        let dx = x[i] - x_mean;
        let dy = y[i] - y_mean;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }
    if var_x.abs() < f64::EPSILON || var_y.abs() < f64::EPSILON { 0.0 } else { cov / (var_x.sqrt() * var_y.sqrt()) }
}

fn dtw_distance_simple(s1: &[f64], s2: &[f64]) -> f64 {
    if s1.is_empty() || s2.is_empty() { return 0.0; }
    let (len1, len2) = (s1.len(), s2.len());
    if len1 > 100 || len2 > 100 {
        let min_len = len1.min(len2);
        let sum: f64 = (0..min_len).map(|i| (s1[i] - s2[i]).powi(2)).sum();
        return sum.sqrt();
    }
    let mut dp = vec![vec![f64::INFINITY; len2]; len1];
    dp[0][0] = (s1[0] - s2[0]).abs();
    for i in 1..len1 { dp[i][0] = dp[i - 1][0] + (s1[i] - s2[0]).abs(); }
    for j in 1..len2 { dp[0][j] = dp[0][j - 1] + (s1[0] - s2[j]).abs(); }
    for i in 1..len1 {
        for j in 1..len2 {
            let cost = (s1[i] - s2[j]).abs();
            dp[i][j] = cost + dp[i - 1][j].min(dp[i][j - 1]).min(dp[i - 1][j - 1]);
        }
    }
    dp[len1 - 1][len2 - 1]
}

fn calculate_std(data: &[f64], mean: f64) -> f64 {
    if data.len() < 2 { return 0.0; }
    let variance: f64 = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (data.len() - 1) as f64;
    variance.sqrt()
}

fn calculate_skewness(data: &[f64], mean: f64, std: f64) -> f64 {
    if data.len() < 3 || std.abs() < f64::EPSILON { return 0.0; }
    let n = data.len() as f64;
    let skew: f64 = data.iter().map(|&x| ((x - mean) / std).powi(3)).sum::<f64>() / n;
    if data.len() > 2 {
        let adj_factor = (n * (n - 1.0)).sqrt() / (n - 2.0);
        skew * adj_factor
    } else {
        skew
    }
}

fn calculate_kurtosis(data: &[f64], mean: f64, std: f64) -> f64 {
    if data.len() < 4 || std.abs() < f64::EPSILON { return 0.0; }
    let n = data.len() as f64;
    let kurt: f64 = data.iter().map(|&x| ((x - mean) / std).powi(4)).sum::<f64>() / n;
    if data.len() > 3 {
        ((n - 1.0) / ((n - 2.0) * (n - 3.0))) * ((n + 1.0) * kurt - 3.0 * (n - 1.0))
    } else {
        kurt - 3.0
    }
}

fn calculate_trend(data: &[f64]) -> f64 {
    if data.len() < 2 { return 0.0; }
    let indices: Vec<f64> = (0..data.len()).map(|i| i as f64).collect();
    correlation(data, &indices)
}

fn calculate_autocorr(data: &[f64]) -> f64 {
    if data.len() < 2 { return 0.0; }
    correlation(&data[0..data.len() - 1], &data[1..data.len()])
}

fn calculate_volume_weighted_time_distance(minor_peaks: &[f64], time_diffs: &[f64]) -> f64 {
    if minor_peaks.is_empty() || minor_peaks.len() != time_diffs.len() { return 0.0; }
    let total_volume: f64 = minor_peaks.iter().sum();
    if total_volume.abs() < f64::EPSILON { return 0.0; }
    let weighted_sum: f64 = minor_peaks.iter().zip(time_diffs.iter()).map(|(&v, &t)| v * t).sum();
    weighted_sum / total_volume
}

// ---------------- main ----------------
fn main() {
    let date: i64 = std::env::args()
        .nth(1)
        .expect("用法: cs_peak_sandbox <date>")
        .parse()
        .expect("date 必须为 8 位整数");
    let codes = list_codes(date, "transaction");
    eprintln!("[1/3] 枚举股票数: {}", codes.len());

    // ---- 第一遍：全市场直方图（volume + turnover）+ 主买主卖 ----
    let hv: Vec<AtomicU64> = (0..MAX_BUCKETS).map(|_| AtomicU64::new(0)).collect();
    let ht: Vec<AtomicU64> = (0..MAX_BUCKETS).map(|_| AtomicU64::new(0)).collect();
    let n_total = AtomicU64::new(0);
    let buy_vol = AtomicU64::new(0);
    let sell_vol = AtomicU64::new(0);
    codes.par_iter().for_each(|c| fill_histograms(c, date, &hv, &ht, &n_total, &buy_vol, &sell_vol));
    let n_total = n_total.load(Ordering::Relaxed);
    let buy_vol = buy_vol.load(Ordering::Relaxed) as f64;
    let sell_vol = sell_vol.load(Ordering::Relaxed) as f64;
    eprintln!("[2/3] 全市场盘中逐笔: {} 笔, 主买 {:.0} 手 / 主卖 {:.0} 手", n_total, buy_vol, sell_vol);

    let qv: Vec<f64> = [0.001, 0.005, 0.01, 0.02, 0.05, 0.10]
        .iter().map(|&q| quantile(&hv, n_total, q, 1)).collect();
    let qt: Vec<f64> = [0.01, 0.10]
        .iter().map(|&q| quantile(&ht, n_total, q, TURN_BUCKET)).collect();
    eprintln!(
        "      volume分位 [0.1%,0.5%,1%,2%,5%,10%] = {:?} 手",
        qv.iter().map(|v| *v as i64).collect::<Vec<_>>()
    );
    eprintln!("      turnover分位 [1%,10%] = {:?} 元", qt.iter().map(|v| *v as i64).collect::<Vec<_>>());

    let (t01_v, t05_v, t1_v, t5_v, t10_v) = (qv[0], qv[1], qv[2], qv[3], qv[4]);
    let (t1_t, t10_t) = (qt[0], qt[1]);

    // ---- 第二遍：per-stock 5 套高峰 ----
    let results: Vec<Option<StockOut>> = codes
        .par_iter()
        .map(|c| {
            read_trade_fast(c, date)
                .ok()
                .and_then(|t| per_stock(&t, t01_v, t05_v, t1_v, t5_v, t10_v, t1_t, t10_t))
                .map(|mut s| {
                    s.code = c.clone();
                    s
                })
        })
        .collect();
    let n_with_data = results.iter().filter(|r| r.is_some()).count();
    eprintln!("[3/3] 有数据股票: {n_with_data}");

    // 全市场总量（第二遍已读数据，直接从 results 归约）
    let mut tot_vol = 0.0f64;
    let mut tot_turn = 0.0f64;
    let mut stocks = Vec::new();
    for r in results.into_iter().flatten() {
        tot_vol += r.tot_vol;
        tot_turn += r.tot_turnover;
        stocks.push(r);
    }

    let out = Out {
        date,
        n_stocks: codes.len(),
        n_with_data,
        n_total_trades: n_total,
        tot_vol,
        tot_turnover: tot_turn,
        buy_vol,
        sell_vol,
        quants_vol: qv,
        quants_turn: qt,
        stocks,
    };
    println!("{}", serde_json::to_string(&out).unwrap());
}
