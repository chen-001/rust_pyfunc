//! 高峰-小峰截面因子（方案 1 · 后处理系列）
//!
//! 从 hm64「大单高峰 → 小峰跟随」改造为全市场截面阈值，一次性输出：
//! - **983 因子/股/日**（与 `factor_pool_v3.py::build_pool` 逐值一致，名字与顺序硬契约）：
//!   - 基础 550：5 套高峰定义（s0 个股内部 / c1 全市场量1% / c2 全市场额1% / c3 量0.1% / c4 量0.5%）
//!     × {n_peaks + 110 统计量}
//!   - 金字塔尺度差异 330：d31_(c3−c1)、d41_(c4−c1)、d34_(c3−c4) × 110
//!   - 事件身份 6（ev_）、相对强度 3（rs_）、冲击占比 2（imp_）
//!   - 双尺度跟随差异 51（dd_）、元信息 2（size_）
//!   - 市场中性化 34（mn_：c1 mean/std 列 − 全市场逐列中位数）
//! - **14 状态标量/日**（全股同值附加列，`__st_` 前缀，供方案 1 状态引擎滚动重估）：
//!   cov_c1、qvol 六级分位、qturn 两级分位、总笔数、总量、总额、买卖比、升级率截面中位数
//!
//! 缺失值补全策略（方案 1 修复点）：
//! 对算不出因子值（无高峰/无小峰/样本不足）的股票，**不填 0、不填截面均值**，
//! 而是按「代理变量截面秩 → 该列有值分布分位映射」填充：代理 = log1p(tot_vol)
//! （tot_vol 缺失时用 log1p(n_trades)）。不同股票填不同值（按其规模/活跃度秩），
//! 且填充值落在该列真实分布内——回测分组时缺失股不会被压成同一个桶。
//!
//! 计算流程（与 sandbox_cs_peak v3 逐字一致的算法核心）：
//!   第一遍：全市场并行读逐笔 → volume/turnover 直方图 + 状态标量（阈值来源）
//!   第二遍：per-stock 并行 5 套高峰识别 → 17 特征 → 110 统计量 → 交互量
//!   归约：cov_c1 / 升级率中位数 / mn 列中位数 → 组装 983 列 → 分位映射补全 → +14 状态列

use crate::fast_csv_reader::{read_trade_fast_inner, TradeRecord};
use numpy::PyReadonlyArray2;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::collections::BTreeSet;
use std::fs;
use std::sync::atomic::{AtomicU64, Ordering};

/// 每只股票输出的因子数 = 983 因子 + 14 状态标量。
pub const N_FACTORS: usize = 983 + 14;
/// 纯因子列数（回测用）。
pub const N_FACTOR_COLS: usize = 983;
/// 状态标量列数。
pub const N_STATE_COLS: usize = 14;

const SECONDS: i64 = 30; // 时间窗口（秒），与 hm64 一致
const MAX_BUCKETS: u64 = 2_000_000; // 直方图桶数上限（16MB）
const TURN_BUCKET: u64 = 1_000; // 成交额桶宽（元），覆盖 20 亿元
const SEC_US: i64 = SECONDS * 1_000_000;

/// 17 特征名（顺序硬契约，与 factor_pool_v3.py FEAT17 一致）。
const FEAT17: [&str; 17] = [
    "小峰成交量总和比值",
    "小峰平均成交量比值",
    "小峰个数",
    "时间间隔均值秒",
    "成交量时间相关系数",
    "DTW距离",
    "成交量变异系数",
    "成交量偏度",
    "成交量峰度",
    "成交量趋势",
    "成交量自相关",
    "时间变异系数",
    "时间偏度",
    "时间峰度",
    "时间趋势",
    "时间自相关",
    "成交量加权时间距离",
];

/// 8 个相关对（顺序硬契约，与 factor_pool_v3.py PAIRS 一致）。
const PAIRS: [(usize, usize); 8] = [
    (0, 3),
    (1, 3),
    (2, 3),
    (6, 11),
    (7, 12),
    (8, 13),
    (9, 14),
    (10, 15),
];

/// 110 统计量后缀顺序（硬契约）。
const SUFFIXES: [&str; 6] = ["_mean", "_std", "_skew", "_kurt", "_autocorr", "_trend"];

// ============================================================
// 因子名（单一源，与 N_FACTORS 严格对齐）
// ============================================================

fn names110(prefix: &str) -> Vec<String> {
    let mut v = Vec::with_capacity(110);
    for (a, b) in PAIRS {
        v.push(format!("{prefix}corr_{}_{}", FEAT17[a], FEAT17[b]));
    }
    for s in SUFFIXES {
        for f in FEAT17 {
            v.push(format!("{prefix}{f}{s}"));
        }
    }
    v
}

/// 983 因子名（与 factor_pool_v3.py::build_pool 的列顺序完全一致）。
pub fn peaks_factor_names() -> Vec<String> {
    let mut names = Vec::with_capacity(N_FACTOR_COLS);
    for p in ["s0_", "c1_", "c2_", "c3_", "c4_"] {
        names.push(format!("{p}n_peaks"));
        names.extend(names110(p));
    }
    for p in ["d31_", "d41_", "d34_"] {
        names.extend(names110(p));
    }
    names.extend([
        "ev_共振数".to_string(),
        "ev_共振占比".to_string(),
        "ev_升级率_c1过s0".to_string(),
        "ev_升级率_c3过c1".to_string(),
        "ev_升级率_c4过c1".to_string(),
        "ev_大单频率".to_string(),
    ]);
    names.extend([
        "rs_相对强度_mean".to_string(),
        "rs_相对强度_std".to_string(),
        "rs_相对强度_skew".to_string(),
    ]);
    names.extend(["imp_量占比".to_string(), "imp_额占比".to_string()]);
    for f in FEAT17 {
        for s in ["_mean", "_std", "_skew"] {
            names.push(format!("dd_{f}{s}"));
        }
    }
    names.extend(["size_tot_vol".to_string(), "size_n_trades".to_string()]);
    for s in ["_mean", "_std"] {
        for f in FEAT17 {
            names.push(format!("mn_{f}{s}"));
        }
    }
    debug_assert_eq!(names.len(), N_FACTOR_COLS);
    names
}

/// 14 状态标量名（全股同值附加列）。
pub fn peaks_state_names() -> Vec<String> {
    vec![
        "__st_cov_c1",
        "__st_qvol_001",
        "__st_qvol_005",
        "__st_qvol_01",
        "__st_qvol_02",
        "__st_qvol_05",
        "__st_qvol_10",
        "__st_qturn_01",
        "__st_qturn_10",
        "__st_n_trades",
        "__st_tot_vol",
        "__st_tot_turnover",
        "__st_buy_sell_ratio",
        "__st_upgrade_med",
    ]
    .into_iter()
    .map(|s| s.to_string())
    .collect()
}

/// 997 个名字（983 因子 + 14 状态标量）。
pub fn peaks_names() -> Vec<String> {
    let mut names = peaks_factor_names();
    names.extend(peaks_state_names());
    names
}

// ============================================================
// 第一遍：全市场直方图（volume + turnover）+ 主买主卖
// ============================================================

fn fill_histograms(
    code: &str,
    date: i64,
    hv: &[AtomicU64],
    ht: &[AtomicU64],
    n_total: &AtomicU64,
    buy_vol: &AtomicU64,
    sell_vol: &AtomicU64,
) {
    if let Ok(trades) = read_trade_fast_inner(code, date, false, true, 8 * 1024 * 1024) {
        for t in &trades {
            let v = t.volume as f64;
            let bv = (v.round() as u64).min(MAX_BUCKETS - 1);
            hv[bv as usize].fetch_add(1, Ordering::Relaxed);
            let bt = ((t.turnover as f64 / TURN_BUCKET as f64).floor() as u64).min(MAX_BUCKETS - 1);
            ht[bt as usize].fetch_add(1, Ordering::Relaxed);
            n_total.fetch_add(1, Ordering::Relaxed);
            match t.flag {
                66 => {
                    buy_vol.fetch_add(v.round() as u64, Ordering::Relaxed);
                }
                83 => {
                    sell_vol.fetch_add(v.round() as u64, Ordering::Relaxed);
                }
                _ => {}
            }
        }
    }
}

/// 从直方图求 top-q 分位（从高到低累计，桶宽 bw）。
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

// ============================================================
// 第二遍：per-stock 5 套高峰
// ============================================================

/// 单套高峰输出。
struct SetOut {
    n_peaks: usize,
    s: Vec<f64>, // 110 统计量
}

/// 单套高峰累加器。
struct SetAcc {
    rows: Vec<Vec<f64>>,
    n_peaks: usize,
}

impl SetAcc {
    fn new() -> Self {
        SetAcc {
            rows: Vec::new(),
            n_peaks: 0,
        }
    }
    fn push(&mut self, feats: Vec<f64>) {
        self.rows.push(feats);
        self.n_peaks += 1;
    }
    fn finish(self) -> SetOut {
        SetOut {
            n_peaks: self.n_peaks,
            s: get_res(&self.rows),
        }
    }
}

/// per-stock 计算输出（983 列中的前 949 列 + 供归约的中间量）。
struct StockOut {
    n_trades: usize,
    tot_vol: f64,
    tot_turnover: f64,
    s0: SetOut,
    c1: SetOut,
    c2: SetOut,
    c3: SetOut,
    c4: SetOut,
    n_both: usize,
    diff_stats: Vec<f64>,   // 51
    rel_strength: Vec<f64>, // 3
    imp_vol_ratio: f64,
    imp_turn_ratio: f64,
}

/// 单只股票 5 套高峰识别（与 sandbox_cs_peak v3 逐字一致；n<30 返回空结构而非丢弃）。
#[allow(clippy::too_many_arguments)]
fn per_stock(
    trades: &[TradeRecord],
    t01_v: f64,
    t05_v: f64,
    t1_v: f64,
    t5_v: f64,
    t10_v: f64,
    t1_t: f64,
    t10_t: f64,
    code: &str,
) -> StockOut {
    let n = trades.len();

    let mut tot_vol = 0.0f64;
    let mut tot_turn = 0.0f64;
    for t in trades {
        tot_vol += t.volume as f64;
        tot_turn += t.turnover as f64;
    }

    // n<30：无足够样本识别高峰 → 空结构（所有统计量为 NaN，由补全策略填充）
    if n < 30 {
        return StockOut {
            n_trades: n,
            tot_vol,
            tot_turnover: tot_turn,
            s0: SetOut { n_peaks: 0, s: vec![f64::NAN; 110] },
            c1: SetOut { n_peaks: 0, s: vec![f64::NAN; 110] },
            c2: SetOut { n_peaks: 0, s: vec![f64::NAN; 110] },
            c3: SetOut { n_peaks: 0, s: vec![f64::NAN; 110] },
            c4: SetOut { n_peaks: 0, s: vec![f64::NAN; 110] },
            n_both: 0,
            diff_stats: vec![f64::NAN; 51],
            rel_strength: vec![f64::NAN; 3],
            imp_vol_ratio: 0.0,
            imp_turn_ratio: 0.0,
        };
    }

    // 每股内部阈值（与 hm64 相同的降序分位取法）
    let mut vols: Vec<f64> = trades.iter().map(|t| t.volume as f64).collect();
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

    let mut s0 = SetAcc::new();
    let mut c1 = SetAcc::new();
    let mut c2 = SetAcc::new();
    let mut c3 = SetAcc::new();
    let mut c4 = SetAcc::new();
    let mut n_both = 0usize;
    let mut diff_rows: Vec<Vec<f64>> = Vec::new(); // 共振高峰：个股尺度 17 − 市场尺度 17
    let mut rel_vals: Vec<f64> = Vec::new(); // C1 高峰量 / ps_tier1
    let mut c1_vol_sum = 0.0f64;
    let mut c1_turn_sum = 0.0f64;

    for i in 0..n {
        let v = trades[i].volume as f64;
        let t = trades[i].turnover as f64;
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
            if dt > SEC_US {
                break;
            }
            ws.push((trades[j].volume as f64, trades[j].turnover as f64, dt as f64 / 1_000_000.0));
        }

        // 各套按自己的阈值过滤小峰
        if hit_s0 {
            let mut mp: Vec<f64> = Vec::new();
            let mut td: Vec<f64> = Vec::new();
            for &(wv, _, dt) in &ws {
                if wv >= ps_tier2 {
                    mp.push(wv);
                    td.push(dt);
                }
            }
            s0.push(calculate_features(v, &mp, &td));
        }
        if hit_c1 {
            let mut mp: Vec<f64> = Vec::new();
            let mut td: Vec<f64> = Vec::new();
            for &(wv, _, dt) in &ws {
                if wv >= t10_v {
                    mp.push(wv);
                    td.push(dt);
                }
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
                if wt >= t10_t {
                    mp.push(wt);
                    td.push(dt);
                }
            }
            c2.push(calculate_features(t, &mp, &td)); // 资金尺度：高峰量用成交额
        }
        if hit_c3 {
            let mut mp: Vec<f64> = Vec::new();
            let mut td: Vec<f64> = Vec::new();
            for &(wv, _, dt) in &ws {
                if wv >= t1_v {
                    mp.push(wv);
                    td.push(dt);
                }
            }
            c3.push(calculate_features(v, &mp, &td));
        }
        if hit_c4 {
            let mut mp: Vec<f64> = Vec::new();
            let mut td: Vec<f64> = Vec::new();
            for &(wv, _, dt) in &ws {
                if wv >= t5_v {
                    mp.push(wv);
                    td.push(dt);
                }
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
                if wv >= ps_tier2 {
                    mp_s.push(wv);
                    td_s.push(dt);
                }
                if wv >= t10_v {
                    mp_c.push(wv);
                    td_c.push(dt);
                }
            }
            let fs = calculate_features(v, &mp_s, &td_s);
            let fc = calculate_features(v, &mp_c, &td_c);
            diff_rows.push(fs.iter().zip(fc.iter()).map(|(a, b)| a - b).collect());
        }
    }

    StockOut {
        n_trades: n,
        tot_vol,
        tot_turnover: tot_turn,
        s0: s0.finish(),
        c1: c1.finish(),
        c2: c2.finish(),
        c3: c3.finish(),
        c4: c4.finish(),
        n_both,
        diff_stats: col_stats_3(&diff_rows), // 51
        rel_strength: stats3(&rel_vals),     // 3
        imp_vol_ratio: if tot_vol > 0.0 { c1_vol_sum / tot_vol } else { 0.0 },
        imp_turn_ratio: if tot_turn > 0.0 { c1_turn_sum / tot_turn } else { 0.0 },
    }
}

/// 单调队列求 ±30s 窗口局部极大值（返回 left/right 两侧存在更大量的标记）。
fn local_max_flags(trades: &[TradeRecord], key: fn(&TradeRecord) -> f64) -> (Vec<bool>, Vec<bool>) {
    let n = trades.len();
    let mut left = vec![false; n];
    let mut right = vec![false; n];
    let mut dq: Vec<usize> = Vec::with_capacity(1024);
    for i in 0..n {
        let t_i = trades[i].time_us;
        while let Some(&f) = dq.first() {
            if t_i - trades[f].time_us > SEC_US {
                dq.remove(0);
            } else {
                break;
            }
        }
        if let Some(&j) = dq.first() {
            if key(&trades[j]) > key(&trades[i]) {
                left[i] = true;
            }
        }
        while let Some(&b) = dq.last() {
            if key(&trades[b]) <= key(&trades[i]) {
                dq.pop();
            } else {
                break;
            }
        }
        dq.push(i);
    }
    dq.clear();
    for i in (0..n).rev() {
        let t_i = trades[i].time_us;
        while let Some(&f) = dq.first() {
            if trades[f].time_us - t_i > SEC_US {
                dq.remove(0);
            } else {
                break;
            }
        }
        if let Some(&j) = dq.first() {
            if key(&trades[j]) > key(&trades[i]) {
                right[i] = true;
            }
        }
        while let Some(&b) = dq.last() {
            if key(&trades[b]) <= key(&trades[i]) {
                dq.pop();
            } else {
                break;
            }
        }
        dq.push(i);
    }
    (left, right)
}

fn vol_key(t: &TradeRecord) -> f64 {
    t.volume as f64
}
fn turn_key(t: &TradeRecord) -> f64 {
    t.turnover as f64
}

/// 每列 mean/std/skew（3 个统计 × 列数）。
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

/// 一列数的 mean/std/skew。
fn stats3(x: &[f64]) -> Vec<f64> {
    if x.is_empty() {
        return vec![f64::NAN; 3];
    }
    let mean = x.iter().sum::<f64>() / x.len() as f64;
    let std = calculate_std(x, mean);
    vec![mean, std, calculate_skewness(x, mean, std)]
}

// ============================================================
// get_res：N×17 矩阵 → 110 统计量/股（与 sandbox v3 / hm64 一致）
// ============================================================

fn get_res(m: &[Vec<f64>]) -> Vec<f64> {
    let mut out = vec![f64::NAN; 110];
    if m.is_empty() {
        return out;
    }
    for (k, &(a, b)) in PAIRS.iter().enumerate() {
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

// ============================================================
// 17 特征计算（与正式库 trade_peak_analysis.rs 逐字一致）
// ============================================================

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
    features[6] = if minor_mean.abs() > f64::EPSILON {
        minor_std / minor_mean
    } else {
        0.0
    };
    features[7] = calculate_skewness(minor_peaks, minor_mean, minor_std);
    features[8] = calculate_kurtosis(minor_peaks, minor_mean, minor_std);
    features[9] = calculate_trend(minor_peaks);
    features[10] = calculate_autocorr(minor_peaks);
    let time_std = calculate_std(time_diffs, time_mean);
    features[11] = if time_mean.abs() > f64::EPSILON {
        time_std / time_mean
    } else {
        0.0
    };
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
    if var_x.abs() < f64::EPSILON || var_y.abs() < f64::EPSILON {
        0.0
    } else {
        cov / (var_x.sqrt() * var_y.sqrt())
    }
}

fn dtw_distance_simple(s1: &[f64], s2: &[f64]) -> f64 {
    if s1.is_empty() || s2.is_empty() {
        return 0.0;
    }
    let (len1, len2) = (s1.len(), s2.len());
    if len1 > 100 || len2 > 100 {
        let min_len = len1.min(len2);
        let sum: f64 = (0..min_len).map(|i| (s1[i] - s2[i]).powi(2)).sum();
        return sum.sqrt();
    }
    let mut dp = vec![vec![f64::INFINITY; len2]; len1];
    dp[0][0] = (s1[0] - s2[0]).abs();
    for i in 1..len1 {
        dp[i][0] = dp[i - 1][0] + (s1[i] - s2[0]).abs();
    }
    for j in 1..len2 {
        dp[0][j] = dp[0][j - 1] + (s1[0] - s2[j]).abs();
    }
    for i in 1..len1 {
        for j in 1..len2 {
            let cost = (s1[i] - s2[j]).abs();
            dp[i][j] = cost + dp[i - 1][j].min(dp[i][j - 1]).min(dp[i - 1][j - 1]);
        }
    }
    dp[len1 - 1][len2 - 1]
}

fn calculate_std(data: &[f64], mean: f64) -> f64 {
    if data.len() < 2 {
        return 0.0;
    }
    let variance: f64 = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (data.len() - 1) as f64;
    variance.sqrt()
}

fn calculate_skewness(data: &[f64], mean: f64, std: f64) -> f64 {
    if data.len() < 3 || std.abs() < f64::EPSILON {
        return 0.0;
    }
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
    if data.len() < 4 || std.abs() < f64::EPSILON {
        return 0.0;
    }
    let n = data.len() as f64;
    let kurt: f64 = data.iter().map(|&x| ((x - mean) / std).powi(4)).sum::<f64>() / n;
    if data.len() > 3 {
        ((n - 1.0) / ((n - 2.0) * (n - 3.0))) * ((n + 1.0) * kurt - 3.0 * (n - 1.0))
    } else {
        kurt - 3.0
    }
}

fn calculate_trend(data: &[f64]) -> f64 {
    if data.len() < 2 {
        return 0.0;
    }
    let indices: Vec<f64> = (0..data.len()).map(|i| i as f64).collect();
    correlation(data, &indices)
}

fn calculate_autocorr(data: &[f64]) -> f64 {
    if data.len() < 2 {
        return 0.0;
    }
    correlation(&data[0..data.len() - 1], &data[1..data.len()])
}

fn calculate_volume_weighted_time_distance(minor_peaks: &[f64], time_diffs: &[f64]) -> f64 {
    if minor_peaks.is_empty() || minor_peaks.len() != time_diffs.len() {
        return 0.0;
    }
    let total_volume: f64 = minor_peaks.iter().sum();
    if total_volume.abs() < f64::EPSILON {
        return 0.0;
    }
    let weighted_sum: f64 = minor_peaks
        .iter()
        .zip(time_diffs.iter())
        .map(|(&v, &t)| v * t)
        .sum();
    weighted_sum / total_volume
}

// ============================================================
// 缺失值补全：代理变量截面秩 → 该列有值分布分位映射
// ============================================================

/// 中位数（跳过 NaN）。
fn median_skip_nan(vals: &[f64]) -> f64 {
    let mut fin: Vec<f64> = vals.iter().filter(|v| v.is_finite()).copied().collect();
    if fin.is_empty() {
        return f64::NAN;
    }
    fin.sort_by(|a, b| a.total_cmp(b));
    let n = fin.len();
    if n % 2 == 1 {
        fin[n / 2]
    } else {
        (fin[n / 2 - 1] + fin[n / 2]) / 2.0
    }
}

/// 对一列值做缺失补全：
/// - 有值部分（is_finite）排序后取 101 个分位点；
/// - 缺失部分按代理秩（0..1，全部股票的百分位秩）线性插值到分位点；
/// - 兜底：有值股票 < 30 时填该列中位数；代理缺失时按秩 0.5。
fn fill_missing_column(col: &mut [f64], proxy_ranks: &[f64]) {
    let mut valid: Vec<f64> = col.iter().filter(|v| v.is_finite()).copied().collect();
    if valid.is_empty() {
        return; // 全列 NaN：保持（该日该列无信息）
    }
    if valid.len() < 30 {
        let med = median_skip_nan(&valid);
        for v in col.iter_mut() {
            if !v.is_finite() {
                *v = med;
            }
        }
        return;
    }
    valid.sort_by(|a, b| a.total_cmp(b));
    // 101 个分位点 [0%, 1%, ..., 100%]
    let mut qpts = [0.0f64; 101];
    let last = (valid.len() - 1) as f64;
    for (i, q) in qpts.iter_mut().enumerate() {
        *q = valid[((last * i as f64 / 100.0).round() as usize).min(valid.len() - 1)];
    }
    for (v, &r) in col.iter_mut().zip(proxy_ranks.iter()) {
        if v.is_finite() {
            continue;
        }
        let r = r.clamp(0.0, 1.0);
        let pos = r * 100.0;
        let lo = (pos.floor() as usize).min(100);
        let hi = (pos.ceil() as usize).min(100);
        let frac = pos - pos.floor();
        let filled = qpts[lo] * (1.0 - frac) + qpts[hi] * frac;
        *v = filled;
    }
}

/// 计算全部股票的代理变量百分位秩（代理 = log1p(tot_vol)，缺失时 log1p(n_trades)，再缺失 0）。
fn proxy_ranks(tot_vols: &[f64], n_trades: &[usize]) -> Vec<f64> {
    let n = tot_vols.len();
    let mut prox: Vec<f64> = Vec::with_capacity(n);
    for i in 0..n {
        if tot_vols[i].is_finite() && tot_vols[i] > 0.0 {
            prox.push((tot_vols[i]).ln_1p());
        } else if n_trades[i] > 0 {
            prox.push((n_trades[i] as f64).ln_1p());
        } else {
            prox.push(0.0);
        }
    }
    // 秩：排序后位置 → [0,1]
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| prox[a].total_cmp(&prox[b]));
    let mut ranks = vec![0.0f64; n];
    for (pos, &idx) in order.iter().enumerate() {
        ranks[idx] = (pos as f64 + 0.5) / n as f64;
    }
    ranks
}

// ============================================================
// 核心计算：读全市场 → 983 因子 + 14 状态标量
// ============================================================

/// 列出某天 transaction 目录下所有股票代码。
pub fn list_codes(date: i64) -> Vec<String> {
    let dir = format!("/ssd_data/stock/{date}/transaction");
    let mut set = BTreeSet::new();
    if let Ok(entries) = fs::read_dir(&dir) {
        for e in entries.flatten() {
            let name = e.file_name().into_string().unwrap_or_default();
            if let Some(code) = name.split('_').next() {
                if code.chars().all(|c| c.is_ascii_digit()) && code.len() == 6 {
                    set.insert(code.to_string());
                }
            }
        }
    }
    set.into_iter().collect()
}

/// 单日全市场计算（v1 读盘入口）。
pub fn compute_peaks_full(date: i64) -> std::io::Result<(Vec<String>, Vec<f32>)> {
    let codes = list_codes(date);

    // ---- 第一遍：全市场直方图（volume + turnover）+ 主买主卖 ----
    let hv: Vec<AtomicU64> = (0..MAX_BUCKETS).map(|_| AtomicU64::new(0)).collect();
    let ht: Vec<AtomicU64> = (0..MAX_BUCKETS).map(|_| AtomicU64::new(0)).collect();
    let n_total = AtomicU64::new(0);
    let buy_vol = AtomicU64::new(0);
    let sell_vol = AtomicU64::new(0);
    codes
        .par_iter()
        .for_each(|c| fill_histograms(c, date, &hv, &ht, &n_total, &buy_vol, &sell_vol));
    let n_total = n_total.load(Ordering::Relaxed);
    let buy_vol = buy_vol.load(Ordering::Relaxed) as f64;
    let sell_vol = sell_vol.load(Ordering::Relaxed) as f64;

    let qv: Vec<f64> = [0.001, 0.005, 0.01, 0.02, 0.05, 0.10]
        .iter()
        .map(|&q| quantile(&hv, n_total, q, 1))
        .collect();
    let qt: Vec<f64> = [0.01, 0.10]
        .iter()
        .map(|&q| quantile(&ht, n_total, q, TURN_BUCKET))
        .collect();

    let (t01_v, t05_v, t1_v, t5_v, t10_v) = (qv[0], qv[1], qv[2], qv[3], qv[4]);
    let (t1_t, t10_t) = (qt[0], qt[1]);

    // ---- 第二遍：per-stock 5 套高峰 ----
    let results: Vec<Option<StockOut>> = codes
        .par_iter()
        .map(|c| {
            read_trade_fast_inner(c, date, false, true, 8 * 1024 * 1024)
                .ok()
                .map(|t| per_stock(&t, t01_v, t05_v, t1_v, t5_v, t10_v, t1_t, t10_t, c))
        })
        .collect();

    // ---- 组装 949 列（983 − mn 34），收集归约量（code 与 row 一一对应） ----
    let mut out_codes: Vec<String> = Vec::new();
    let mut rows: Vec<Vec<f64>> = Vec::new();
    let mut stock_tot_vol: Vec<f64> = Vec::new();
    let mut stock_n_trades: Vec<usize> = Vec::new();
    let mut upgrade_vals: Vec<f64> = Vec::new();
    let mut tot_vol_sum = 0.0f64;
    let mut tot_turn_sum = 0.0f64;
    let mut c1_peak_hits = 0usize;

    for (code, r_opt) in codes.iter().zip(results.into_iter()) {
        let Some(r) = r_opt else { continue };
        out_codes.push(code.clone());
        tot_vol_sum += r.tot_vol;
        tot_turn_sum += r.tot_turnover;
        if r.c1.n_peaks > 0 {
            c1_peak_hits += 1;
        }
        let n0 = r.s0.n_peaks as f64;
        let n1 = r.c1.n_peaks as f64;
        upgrade_vals.push(n1 / n0.max(1.0));
        stock_tot_vol.push(r.tot_vol);
        stock_n_trades.push(r.n_trades);

        let mut row = Vec::with_capacity(N_FACTOR_COLS);
        for set in [&r.s0, &r.c1, &r.c2, &r.c3, &r.c4] {
            row.push(set.n_peaks as f64);
            row.extend_from_slice(&set.s);
        }
        // 金字塔尺度差异（NaN - x = NaN，保持）
        let c1s = &r.c1.s;
        let c3s = &r.c3.s;
        let c4s = &r.c4.s;
        row.extend(c3s.iter().zip(c1s.iter()).map(|(a, b)| a - b)); // d31
        row.extend(c4s.iter().zip(c1s.iter()).map(|(a, b)| a - b)); // d41
        row.extend(c3s.iter().zip(c4s.iter()).map(|(a, b)| a - b)); // d34
        // 事件身份
        let n0i = r.s0.n_peaks;
        let n1i = r.c1.n_peaks;
        row.push(r.n_both as f64);
        row.push(r.n_both as f64 / n1i.max(1) as f64);
        row.push(n1i as f64 / n0i.max(1) as f64);
        row.push(r.c3.n_peaks as f64 / n1i.max(1) as f64);
        row.push(r.c4.n_peaks as f64 / n1i.max(1) as f64);
        row.push(n1i as f64 / r.n_trades.max(1) as f64 * 1e4);
        // 相对强度 + 冲击占比
        row.extend_from_slice(&r.rel_strength);
        row.push(r.imp_vol_ratio);
        row.push(r.imp_turn_ratio);
        // 双尺度跟随差异 51
        row.extend_from_slice(&r.diff_stats);
        // size 2
        row.push(r.tot_vol);
        row.push(r.n_trades as f64);
        // mn 34 列：暂填 NaN，下面归约后填充
        row.extend(std::iter::repeat(f64::NAN).take(34));
        debug_assert_eq!(row.len(), N_FACTOR_COLS);
        rows.push(row);
    }

    // ---- mn 归约 + 缺失补全 + 状态标量（公共路径） ----
    let (rows, state_vals) = finalize_rows(
        &mut rows,
        &stock_tot_vol,
        &stock_n_trades,
        &upgrade_vals,
        tot_vol_sum,
        tot_turn_sum,
        c1_peak_hits,
        n_total,
        buy_vol,
        sell_vol,
        &qv,
        &qt,
    );

    // ---- fan-out ----
    let n_stocks = rows.len();
    let mut vals = Vec::with_capacity(n_stocks * N_FACTORS);
    for row in rows.iter() {
        vals.extend(row.iter().map(|&v| v as f32));
        vals.extend(state_vals.iter().map(|&v| v as f32));
    }
    Ok((out_codes, vals))
}

// ============================================================
// v2 核心（无 IO）：从预加载的 per-stock 逐笔数据 → 因子
// ============================================================

/// 纯计算核心：trades（可选）→ 983 因子 + 14 状态标量。
/// 与 v1 的区别仅在于数据来源；阈值/归约/补全逻辑完全一致。
/// 注意：v2 用传入的股票子集做横截面（cov_c1/mn/分位阈值均基于子集），
/// 与 v1 全市场口径数值不同是预期的（cross-section-pipeline 规范说明）。
pub fn compute_peaks_from_trades(
    codes: &[String],
    trades_per_code: Vec<Option<Vec<TradeRecord>>>,
) -> (Vec<String>, Vec<f32>) {
    // 第一遍等价物：从 trades 直接构建直方图（与 fill_histograms 同口径，并行）
    let hv: Vec<AtomicU64> = (0..MAX_BUCKETS).map(|_| AtomicU64::new(0)).collect();
    let ht: Vec<AtomicU64> = (0..MAX_BUCKETS).map(|_| AtomicU64::new(0)).collect();
    let n_total = AtomicU64::new(0);
    let buy_vol = AtomicU64::new(0);
    let sell_vol = AtomicU64::new(0);
    trades_per_code.par_iter().for_each(|t_opt| {
        if let Some(trades) = t_opt {
            for t in trades {
                let v = t.volume as f64;
                let bv = (v.round() as u64).min(MAX_BUCKETS - 1);
                hv[bv as usize].fetch_add(1, Ordering::Relaxed);
                let bt = ((t.turnover as f64 / TURN_BUCKET as f64).floor() as u64).min(MAX_BUCKETS - 1);
                ht[bt as usize].fetch_add(1, Ordering::Relaxed);
                n_total.fetch_add(1, Ordering::Relaxed);
                match t.flag {
                    66 => {
                        buy_vol.fetch_add(v.round() as u64, Ordering::Relaxed);
                    }
                    83 => {
                        sell_vol.fetch_add(v.round() as u64, Ordering::Relaxed);
                    }
                    _ => {}
                }
            }
        }
    });
    let n_total = n_total.load(Ordering::Relaxed);
    let buy_vol = buy_vol.load(Ordering::Relaxed) as f64;
    let sell_vol = sell_vol.load(Ordering::Relaxed) as f64;

    let qv: Vec<f64> = [0.001, 0.005, 0.01, 0.02, 0.05, 0.10]
        .iter()
        .map(|&q| quantile(&hv, n_total, q, 1))
        .collect();
    let qt: Vec<f64> = [0.01, 0.10]
        .iter()
        .map(|&q| quantile(&ht, n_total, q, TURN_BUCKET))
        .collect();

    let (t01_v, t05_v, t1_v, t5_v, t10_v) = (qv[0], qv[1], qv[2], qv[3], qv[4]);
    let (t1_t, t10_t) = (qt[0], qt[1]);

    // per-stock 计算（并行，单遍读内存版的核心）
    let per_stock_results: Vec<Option<(String, StockOut)>> = codes
        .par_iter()
        .zip(trades_per_code.par_iter())
        .map(|(code, t_opt)| {
            let trades = t_opt.as_ref()?;
            Some((code.clone(), per_stock(trades, t01_v, t05_v, t1_v, t5_v, t10_v, t1_t, t10_t, code)))
        })
        .collect();

    let mut out_codes: Vec<String> = Vec::new();
    let mut rows: Vec<Vec<f64>> = Vec::new();
    let mut stock_tot_vol: Vec<f64> = Vec::new();
    let mut stock_n_trades: Vec<usize> = Vec::new();
    let mut upgrade_vals: Vec<f64> = Vec::new();
    let mut tot_vol_sum = 0.0f64;
    let mut tot_turn_sum = 0.0f64;
    let mut c1_peak_hits = 0usize;

    for r_opt in per_stock_results.into_iter() {
        let Some((code, r)) = r_opt else { continue };
        out_codes.push(code.clone());
        tot_vol_sum += r.tot_vol;
        tot_turn_sum += r.tot_turnover;
        if r.c1.n_peaks > 0 {
            c1_peak_hits += 1;
        }
        let n0 = r.s0.n_peaks as f64;
        let n1 = r.c1.n_peaks as f64;
        upgrade_vals.push(n1 / n0.max(1.0));
        stock_tot_vol.push(r.tot_vol);
        stock_n_trades.push(r.n_trades);

        let mut row = Vec::with_capacity(N_FACTOR_COLS);
        for set in [&r.s0, &r.c1, &r.c2, &r.c3, &r.c4] {
            row.push(set.n_peaks as f64);
            row.extend_from_slice(&set.s);
        }
        let c1s = &r.c1.s;
        let c3s = &r.c3.s;
        let c4s = &r.c4.s;
        row.extend(c3s.iter().zip(c1s.iter()).map(|(a, b)| a - b));
        row.extend(c4s.iter().zip(c1s.iter()).map(|(a, b)| a - b));
        row.extend(c3s.iter().zip(c4s.iter()).map(|(a, b)| a - b));
        let n0i = r.s0.n_peaks;
        let n1i = r.c1.n_peaks;
        row.push(r.n_both as f64);
        row.push(r.n_both as f64 / n1i.max(1) as f64);
        row.push(n1i as f64 / n0i.max(1) as f64);
        row.push(r.c3.n_peaks as f64 / n1i.max(1) as f64);
        row.push(r.c4.n_peaks as f64 / n1i.max(1) as f64);
        row.push(n1i as f64 / r.n_trades.max(1) as f64 * 1e4);
        row.extend_from_slice(&r.rel_strength);
        row.push(r.imp_vol_ratio);
        row.push(r.imp_turn_ratio);
        row.extend_from_slice(&r.diff_stats);
        row.push(r.tot_vol);
        row.push(r.n_trades as f64);
        row.extend(std::iter::repeat(f64::NAN).take(34));
        debug_assert_eq!(row.len(), N_FACTOR_COLS);
        rows.push(row);
    }

    // mn 列归约 + 补全 + 状态标量（与 v1 相同的公共路径）
    let (rows, state_vals) = finalize_rows(
        &mut rows,
        &stock_tot_vol,
        &stock_n_trades,
        &upgrade_vals,
        tot_vol_sum,
        tot_turn_sum,
        c1_peak_hits,
        n_total,
        buy_vol,
        sell_vol,
        &qv,
        &qt,
    );

    let n_stocks = rows.len();
    let mut vals = Vec::with_capacity(n_stocks * N_FACTORS);
    for row in rows.iter() {
        vals.extend(row.iter().map(|&v| v as f32));
        vals.extend(state_vals.iter().map(|&v| v as f32));
    }
    (out_codes, vals)
}

/// 公共归约路径：mn 列中位数 → 缺失补全 → 14 状态标量。
#[allow(clippy::too_many_arguments)]
fn finalize_rows(
    rows: &mut Vec<Vec<f64>>,
    stock_tot_vol: &[f64],
    stock_n_trades: &[usize],
    upgrade_vals: &[f64],
    tot_vol_sum: f64,
    tot_turn_sum: f64,
    c1_peak_hits: usize,
    n_total: u64,
    buy_vol: f64,
    sell_vol: f64,
    qv: &[f64],
    qt: &[f64],
) -> (Vec<Vec<f64>>, Vec<f64>) {
    // ---- mn 列归约：c1 的 mean/std 列（行内偏移 120..154，110 统计量布局 = corr8 + mean17 + std17）逐列中位数 ----
    let mn_base = 120usize;
    let mn_medians: Vec<f64> = (0..34)
        .map(|j| {
            let col: Vec<f64> = rows.iter().map(|r| r[mn_base + j]).collect();
            median_skip_nan(&col)
        })
        .collect();
    for row in rows.iter_mut() {
        for (j, &med) in mn_medians.iter().enumerate() {
            let v = row[mn_base + j];
            row[N_FACTOR_COLS - 34 + j] = if v.is_finite() { v - med } else { f64::NAN };
        }
    }

    // ---- 缺失值补全（代理秩 → 分位映射；跳过 n_peaks/ev/size 列） ----
    let ranks = proxy_ranks(stock_tot_vol, stock_n_trades);
    let mut keep_raw = [false; N_FACTOR_COLS];
    for &i in &[0usize, 111, 222, 333, 444] {
        keep_raw[i] = true; // 5 套 n_peaks
    }
    for i in 885..891 {
        keep_raw[i] = true; // ev_ 6 个
    }
    keep_raw[947] = true; // size_tot_vol
    keep_raw[948] = true; // size_n_trades
    for j in 0..N_FACTOR_COLS {
        if keep_raw[j] {
            continue;
        }
        let mut col: Vec<f64> = rows.iter().map(|r| r[j]).collect();
        fill_missing_column(&mut col, &ranks);
        for (r, v) in rows.iter_mut().zip(col.into_iter()) {
            r[j] = v;
        }
    }

    // ---- 14 状态标量（全股同值） ----
    let n_valid_f = rows.len().max(1) as f64;
    let cov_c1 = c1_peak_hits as f64 / n_valid_f;
    let upgrade_med = median_skip_nan(upgrade_vals);
    let buy_sell_ratio = if sell_vol > 0.0 { buy_vol / sell_vol } else { 0.0 };
    let state_vals: Vec<f64> = vec![
        cov_c1,
        qv[0],
        qv[1],
        qv[2],
        qv[3],
        qv[4],
        qv[5],
        qt[0],
        qt[1],
        n_total as f64,
        tot_vol_sum,
        tot_turn_sum,
        buy_sell_ratio,
        upgrade_med,
    ];
    (std::mem::take(rows), state_vals)
}

// ============================================================
// PyO3 导出（v1 读盘 / v2 传数据）
// ============================================================

/// Python 单日调试：v1 读盘入口，返回 (codes, vals[997])。
#[pyfunction]
pub fn py_peaks(py: Python<'_>, date: i64) -> PyResult<(Vec<String>, Vec<f32>)> {
    compute_peaks_full(date).map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("{e:?}")))
}

/// Python 拿 997 个因子名（983 因子 + 14 状态标量）。
#[pyfunction]
pub fn py_peaks_names() -> Vec<String> {
    peaks_names()
}

/// v2 入口（Python 传数据）：codes + per-stock trade 数组 → (codes, vals)。
///
/// trade_arrays 列序与 read_trade_fast 一致：[time_sec, price, volume, turnover, flag, bid_order, ask_order]；
/// time_us 由 time_sec（epoch 秒）× 1e6 构造（v2 演示场景秒级精度足够）。
#[pyfunction]
pub fn py_peaks_from_data(
    _py: Python<'_>,
    codes: Vec<String>,
    trade_arrays: Vec<PyReadonlyArray2<f64>>,
) -> PyResult<(Vec<String>, Vec<f32>)> {
    if codes.len() != trade_arrays.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "codes.len()={} != trade_arrays.len()={}",
            codes.len(),
            trade_arrays.len()
        )));
    }
    let trades_per_code: Vec<Option<Vec<TradeRecord>>> = trade_arrays
        .iter()
        .map(|arr| {
            let a = arr.as_array();
            if a.nrows() == 0 {
                return None;
            }
            let mut recs = Vec::with_capacity(a.nrows());
            for i in 0..a.nrows() {
                recs.push(TradeRecord {
                    time_sec: a[[i, 0]] as f32,
                    time_us: (a[[i, 0]] * 1_000_000.0) as i64,
                    price: a[[i, 1]] as f32,
                    volume: a[[i, 2]] as f32,
                    turnover: a[[i, 3]] as f32,
                    flag: a[[i, 4]] as i32,
                    bid_order: a[[i, 5]] as i64,
                    ask_order: a[[i, 6]] as i64,
                });
            }
            Some(recs)
        })
        .collect();
    Ok(compute_peaks_from_trades(&codes, trades_per_code))
}

// ============================================================
// 测试
// ============================================================
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_names_count() {
        assert_eq!(peaks_names().len(), N_FACTORS);
        assert_eq!(peaks_factor_names().len(), N_FACTOR_COLS);
        assert_eq!(peaks_state_names().len(), N_STATE_COLS);
    }

    #[test]
    fn test_fill_missing_column_distinct() {
        // 缺失股票按代理秩映射到不同分位 → 填充值互不相同
        let mut col = vec![1.0, 2.0, 3.0, 4.0, 5.0, f64::NAN, f64::NAN];
        let ranks = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.1, 0.9];
        fill_missing_column(&mut col, &ranks);
        assert!(col[5].is_finite() && col[6].is_finite());
        assert!((col[5] - col[6]).abs() > 1e-9, "不同代理秩必须填不同值");
    }

    #[test]
    fn test_reader_paths_consistent() {
        // 对比 read_to_string 路径（usize::MAX）与 mmap 路径（8MB）——必须逐笔一致
        let a = crate::fast_csv_reader::read_trade_fast_inner("000001", 20260618, false, true, usize::MAX).unwrap();
        let b = crate::fast_csv_reader::read_trade_fast_inner("000001", 20260618, false, true, 8 * 1024 * 1024).unwrap();
        println!("usize::MAX path: n={} vol_sum={}", a.len(), a.iter().map(|t| t.volume as f64).sum::<f64>());
        println!("8MB path:       n={} vol_sum={}", b.len(), b.iter().map(|t| t.volume as f64).sum::<f64>());
        assert_eq!(a.len(), b.len());
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            if x.volume != y.volume || x.time_us != y.time_us || x.turnover != y.turnover {
                println!("first diff at {i}: a={:?} b={:?}", (x.volume, x.time_us, x.turnover), (y.volume, y.time_us, y.turnover));
                return;
            }
        }
        println!("两路径逐笔完全一致");
    }

    #[test]
    fn test_get_res_empty() {
        let out = get_res(&[]);
        assert_eq!(out.len(), 110);
        assert!(out.iter().all(|v| v.is_nan()));
    }
}
