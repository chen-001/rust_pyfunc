//! 高峰-小峰截面因子 · 方案 2（计算自适应系列）核心模块
//!
//! 与方案 1（后处理系列）完全独立：方案 2 的因子计算按日期**顺序推进**，
//! 每天先读自己的标量表存储 [t−W, t−1]，聚类判定当日市场状态，再用
//! 状态化参数计算当天因子——状态进入**计算过程**而非后处理。
//!
//! 本模块包含：
//! - `RegimeParams`：当日状态参数（聚类标签 + 连续标量 + 计算层参数预留）
//! - `fit_regime`：滚动 PCA + k=4 聚类 + 修正阈值 + 卖压软编码（纯 Rust，KB 级输入，毫秒级）
//! - `state_store`：标量表每日读写（14 个 f64，binary 文件，为 t+1 备料）
//! - `compute_peaks_regime_full`：当日因子计算入口（状态化参数生效点，先基线版）
//!
//! 调度框架见 `factor_pipeline.rs::run_factor_pipeline_regime`。

use crate::fast_csv_reader::{read_trade_fast_inner, TradeRecord};
use crate::peaks_metrics;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::fs;

/// 状态标量列数（与 peaks_metrics 的 14 状态列一致）。
pub const N_STATE: usize = 14;

/// 状态标量列索引（与 peaks_metrics::peaks_state_names 顺序一致）。
#[derive(Clone, Copy)]
pub struct StateIdx {
    pub cov_c1: usize,
    pub qvol_001: usize,
    pub qvol_005: usize,
    pub qvol_01: usize,
    pub qvol_02: usize,
    pub qvol_05: usize,
    pub qvol_10: usize,
    pub qturn_01: usize,
    pub qturn_10: usize,
    pub n_trades: usize,
    pub tot_vol: usize,
    pub tot_turnover: usize,
    pub buy_sell_ratio: usize,
    pub upgrade_med: usize,
}

impl StateIdx {
    pub fn new() -> Self {
        StateIdx {
            cov_c1: 0,
            qvol_001: 1,
            qvol_005: 2,
            qvol_01: 3,
            qvol_02: 4,
            qvol_05: 5,
            qvol_10: 6,
            qturn_01: 7,
            qturn_10: 8,
            n_trades: 9,
            tot_vol: 10,
            tot_turnover: 11,
            buy_sell_ratio: 12,
            upgrade_med: 13,
        }
    }
}

/// 当日状态参数（fit_regime 输出，供计算层使用）。
#[derive(Clone, Debug)]
pub struct RegimeParams {
    /// k=4 聚类标签（0..4，无语义，语义由阈值/连续标量判定）。
    pub cluster: i32,
    /// PC1「尾部极端强度」得分（当日）。
    pub pc1: f64,
    /// 当日是否为极端尾部（修正阈值：qvol_01≥2万 或 qvol_001≥18万手）。
    pub extreme: bool,
    /// 卖压软编码 bsr/rolling_med20(bsr)−1（无历史时为 0）。
    pub sell_pressure: f64,
    /// 当日 cov_c1（层 1 门控用）。
    pub cov_c1: f64,
    /// 当日全部 14 标量（原样传递，供计算层/元数据用）。
    pub scalars: Vec<f64>,
    /// 计算层参数（方案 2 计算代码阶段填充：阈值缩放/窗口/特征开关等）。
    pub threshold_scale: f64,
    pub window_sec: i64,
    pub agg_mode: i32,
    /// 该参数是否为基线（预热期/无历史）。
    pub is_baseline: bool,
}

impl RegimeParams {
    /// 基线参数（预热期或窗口不足时使用）。
    pub fn baseline(scalars: Vec<f64>) -> Self {
        RegimeParams {
            cluster: -1,
            pc1: 0.0,
            extreme: false,
            sell_pressure: 0.0,
            cov_c1: scalars.get(0).copied().unwrap_or(0.0),
            scalars,
            threshold_scale: 1.0,
            window_sec: 30,
            agg_mode: 0,
            is_baseline: true,
        }
    }
}

// ============================================================
// 标量表存储（state_store）：每日一个 binary 文件（14 × f64）
// ============================================================

/// 写入某日标量（14 个 f64，binary，小端）。文件: {dir}/{date}.bin
pub fn state_write_row(dir: &str, date: i64, scalars: &[f64]) -> std::io::Result<()> {
    fs::create_dir_all(dir)?;
    let path = format!("{dir}/{date}.bin");
    let mut buf = Vec::with_capacity(N_STATE * 8);
    for v in scalars.iter().take(N_STATE) {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    fs::write(&path, buf)
}

/// 读取某日标量（无文件 → None）。
pub fn state_read_row(dir: &str, date: i64) -> Option<Vec<f64>> {
    let path = format!("{dir}/{date}.bin");
    let buf = fs::read(&path).ok()?;
    if buf.len() < N_STATE * 8 {
        return None;
    }
    let mut out = Vec::with_capacity(N_STATE);
    for i in 0..N_STATE {
        let mut b = [0u8; 8];
        b.copy_from_slice(&buf[i * 8..i * 8 + 8]);
        out.push(f64::from_le_bytes(b));
    }
    Some(out)
}

/// 读取 [t−W, t−1] 窗口内的标量表（按日期升序；缺失日期跳过）。
/// 返回 (dates, rows)，rows 每行 14 列。
pub fn state_read_window(dir: &str, t: i64, w: usize, trading_days: &[i64]) -> (Vec<i64>, Vec<Vec<f64>>) {
    // 从 trading_days 中取 t 之前最近的 W 个交易日
    let mut dates: Vec<i64> = Vec::new();
    for &d in trading_days.iter().rev() {
        if d >= t {
            continue;
        }
        dates.push(d);
        if dates.len() >= w {
            break;
        }
    }
    dates.reverse();
    let mut rows = Vec::new();
    let mut kept = Vec::new();
    for &d in &dates {
        if let Some(r) = state_read_row(dir, d) {
            kept.push(d);
            rows.push(r);
        }
    }
    (kept, rows)
}

// ============================================================
// fit_regime：滚动 PCA + k=4 聚类 + 修正阈值 + 卖压软编码
// ============================================================

/// 列均值/标准差（跨行）。
fn col_mean_std(rows: &[Vec<f64>], j: usize) -> (f64, f64) {
    let n = rows.len();
    let mean = rows.iter().map(|r| r[j]).sum::<f64>() / n as f64;
    let var = rows.iter().map(|r| (r[j] - mean).powi(2)).sum::<f64>() / n as f64;
    (mean, var.sqrt())
}

/// 对称矩阵雅可比特征分解（14×14，确定性，返回特征值降序 + 特征向量列）。
/// 仅用于小矩阵（状态维度 ≤ 16），不做通用 QR。
fn jacobi_eigen(matrix: &[[f64; N_STATE]; N_STATE]) -> (Vec<f64>, Vec<[f64; N_STATE]>) {
    const MAX_SWEEP: usize = 50;
    let n = N_STATE;
    let mut a = *matrix;
    let mut v: Vec<[f64; N_STATE]> = (0..n)
        .map(|i| {
            let mut row = [0.0f64; N_STATE];
            row[i] = 1.0;
            row
        })
        .collect();

    for _ in 0..MAX_SWEEP {
        // 找最大非对角元
        let mut off = 0.0f64;
        let (mut p, mut q) = (0usize, 1usize);
        for i in 0..n {
            for j in (i + 1)..n {
                let x = a[i][j].abs();
                if x > off {
                    off = x;
                    p = i;
                    q = j;
                }
            }
        }
        if off < 1e-12 {
            break;
        }
        let theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]).max(1e-300);
        let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
        let c = 1.0 / (1.0 + t * t).sqrt();
        let s = t * c;
        for k in 0..n {
            let akp = a[k][p];
            let akq = a[k][q];
            a[k][p] = c * akp - s * akq;
            a[k][q] = s * akp + c * akq;
        }
        for k in 0..n {
            let apk = a[p][k];
            let aqk = a[q][k];
            a[p][k] = c * apk - s * aqk;
            a[q][k] = s * apk + c * aqk;
        }
        for k in 0..n {
            let vkp = v[k][p];
            let vkq = v[k][q];
            v[k][p] = c * vkp - s * vkq;
            v[k][q] = s * vkp + c * vkq;
        }
    }
    // 特征值 = 对角元；排序（降序），特征向量同步排列
    let mut eig: Vec<(f64, usize)> = (0..n).map(|i| (a[i][i], i)).collect();
    eig.sort_by(|x, y| y.0.total_cmp(&x.0));
    let vals: Vec<f64> = eig.iter().map(|(v, _)| *v).collect();
    let mut vecs = Vec::with_capacity(n);
    for (_, col) in eig {
        let mut row = [0.0f64; N_STATE];
        for k in 0..n {
            row[k] = v[k][col];
        }
        vecs.push(row);
    }
    (vals, vecs)
}

/// 1D k-means（k=4，固定种子确定性，10 次迭代）。
/// 输入一维得分，返回每点标签。
fn kmeans_1d(scores: &[f64], k: usize) -> Vec<i32> {
    let n = scores.len();
    if n == 0 {
        return Vec::new();
    }
    let mut sorted: Vec<(usize, f64)> = scores.iter().enumerate().map(|(i, &s)| (i, s)).collect();
    sorted.sort_by(|a, b| a.1.total_cmp(&b.1));
    // 初始中心：等分位（确定性）
    let mut centers: Vec<f64> = (0..k)
        .map(|c| {
            let idx = ((n - 1) as f64 * (c as f64 + 0.5) / k as f64).round() as usize;
            sorted[idx].1
        })
        .collect();
    let mut labels = vec![0i32; n];
    for _ in 0..10 {
        // 分配
        for (i, &s) in scores.iter().enumerate() {
            let mut best = 0usize;
            let mut bd = f64::INFINITY;
            for (c, &ctr) in centers.iter().enumerate() {
                let d = (s - ctr).abs();
                if d < bd {
                    bd = d;
                    best = c;
                }
            }
            labels[i] = best as i32;
        }
        // 更新中心
        let mut sums = vec![0.0f64; k];
        let mut cnt = vec![0usize; k];
        for (i, &s) in scores.iter().enumerate() {
            let c = labels[i] as usize;
            sums[c] += s;
            cnt[c] += 1;
        }
        let mut moved = false;
        for c in 0..k {
            if cnt[c] > 0 {
                let nc = sums[c] / cnt[c] as f64;
                if (nc - centers[c]).abs() > 1e-12 {
                    moved = true;
                }
                centers[c] = nc;
            }
        }
        if !moved {
            break;
        }
    }
    labels
}

/// 状态拟合：输入窗口标量表（W×14，日期升序）→ 当日 params。
/// 窗口不足 W 行时退化为基线（is_baseline=true）。
pub fn fit_regime(window: &[Vec<f64>], min_rows: usize) -> RegimeParams {
    let n = window.len();
    if n < min_rows {
        return RegimeParams::baseline(Vec::new());
    }
    let idx = StateIdx::new();

    // ---- 1. 标准化 + 协方差 PCA ----
    // 特征取 log 化：qvol/turnover/规模列取 log1p（长尾），cov_c1/bsr/upgrade 原值
    let mut z = vec![vec![0.0f64; N_STATE]; n];
    let mut means = [0.0f64; N_STATE];
    let mut stds = [0.0f64; N_STATE];
    for j in 0..N_STATE {
        let log_col = matches!(j, 1..=11); // qvol 6 + qturn 2 + n_trades + tot_vol + tot_turnover
        let col: Vec<f64> = window.iter().map(|r| if log_col { r[j].ln_1p() } else { r[j] }).collect();
        let m = col.iter().sum::<f64>() / n as f64;
        let v = col.iter().map(|x| (x - m).powi(2)).sum::<f64>() / n as f64;
        means[j] = m;
        stds[j] = v.sqrt().max(1e-12);
        for (i, &x) in col.iter().enumerate() {
            z[i][j] = (x - m) / stds[j];
        }
    }
    let mut cov = [[0.0f64; N_STATE]; N_STATE];
    for a in 0..N_STATE {
        for b in 0..N_STATE {
            cov[a][b] = (0..n).map(|i| z[i][a] * z[i][b]).sum::<f64>() / n as f64;
        }
    }
    let (_vals, vecs) = jacobi_eigen(&cov);
    let pc1_load = vecs[0]; // 第一主成分载荷
    let pc1_scores: Vec<f64> = z.iter().map(|row| row.iter().zip(pc1_load.iter()).map(|(a, b)| a * b).sum()).collect();

    // ---- 2. k=4 聚类（PC1 上 1D k-means，确定性） ----
    let labels = kmeans_1d(&pc1_scores, 4);

    // ---- 3. 修正阈值：极端尾部（当日标量判定，qvol 阈值来自定案） ----
    let last = &window[n - 1];
    let extreme = last[idx.qvol_01] >= 20_000.0 || last[idx.qvol_001] >= 180_000.0;

    // ---- 4. 卖压软编码：bsr / rolling_med20(bsr) − 1 ----
    let bsr_col: Vec<f64> = window.iter().map(|r| r[idx.buy_sell_ratio]).collect();
    let sell_pressure = if n >= 20 {
        let mut med = bsr_col[n - 20..n].to_vec();
        med.sort_by(|a, b| a.total_cmp(b));
        let m20 = med[9] + (med[10] - med[9]) / 2.0; // 中位数
        if m20 > 0.0 {
            last[idx.buy_sell_ratio] / m20 - 1.0
        } else {
            0.0
        }
    } else {
        0.0
    };

    RegimeParams {
        cluster: *labels.last().unwrap_or(&-1),
        pc1: *pc1_scores.last().unwrap_or(&0.0),
        extreme,
        sell_pressure,
        cov_c1: last[idx.cov_c1],
        scalars: last.clone(),
        threshold_scale: 1.0,
        window_sec: 30,
        agg_mode: 0,
        is_baseline: false,
    }
}

// ============================================================
// 当日因子计算（状态化参数生效点）
// ============================================================

/// 方案 2 当日计算入口：v0 先复用方案 1 的基线计算（params 记录但不生效），
/// 返回 (codes, vals[997])。状态化计算（阈值/窗口/聚合随 params 调整）
/// 在方案 2 计算代码阶段实现。
pub fn compute_peaks_regime_full(
    date: i64,
    _params: &RegimeParams,
) -> std::io::Result<(Vec<String>, Vec<f32>)> {
    // v0：基线计算（两遍读全在 peaks_metrics 内）
    peaks_metrics::compute_peaks_full(date)
}

/// 从 compute 输出提取当日 14 状态标量（vals 尾部 14 列，全股同值）。
/// 无股票时返回 None（该日无数据，不写标量存储）。
pub fn extract_state_from_vals(vals: &[f32], n_factors: usize) -> Option<Vec<f64>> {
    let n_stocks = vals.len() / n_factors;
    if n_stocks == 0 {
        return None;
    }
    let base = (n_stocks - 1) * n_factors + n_factors - peaks_metrics::N_STATE_COLS;
    let mut out = Vec::with_capacity(N_STATE);
    for i in 0..N_STATE {
        out.push(vals[base + i] as f64);
    }
    Some(out)
}

/// 供调度框架复用的当日全流程：读盘 → 计算 → 返回 (codes, vals)。
/// 框架 v0 阶段 compute 内部完成两遍读。
pub fn compute_day(date: i64, params: &RegimeParams) -> std::io::Result<(Vec<String>, Vec<f32>)> {
    compute_peaks_regime_full(date, params)
}

/// 单遍读（预读流水线用）：读一天全市场 → (codes, trades)。
/// 读盘与状态无关，可提前并行执行；内存块由调用方持有。
pub fn read_all_market(date: i64) -> (Vec<String>, Vec<Option<Vec<TradeRecord>>>) {
    let codes = peaks_metrics::list_codes(date);
    let trades: Vec<Option<Vec<TradeRecord>>> = codes
        .par_iter()
        .map(|c| read_trade_fast_inner(c, date, false, true, 8 * 1024 * 1024).ok())
        .collect();
    (codes, trades)
}

/// 从状态参数派生当日计算参数（窗口 + 阈值缩放）。
///
/// v1 启发性规则（软过渡，参数可调；后续以敏感性实验收敛）：
/// - 极端尾部：聚焦更极端事件（volume 阈值上调 30%）、窗口缩短到 20s
/// - 低迷（cov_c1 < 0.80）：阈值下调捕获更多事件（补足稀疏跟随）、窗口拉长到 50s
/// - 活跃（cov_c1 > 0.95）：阈值上调聚焦大单、窗口缩短到 20s
/// - 常态/预热：基线（30s、不缩放）——方案 1 行为
pub fn derive_compute_params(p: &RegimeParams) -> (i64, [f64; 4], [f64; 4]) {
    let mut window = 30i64;
    let mut peak = [1.0f64; 4]; // [c1, c2, c3, c4] 高峰线缩放
    let mut valley = [1.0f64; 4]; // [c1, c2, c3, c4] 小峰线缩放
    if p.is_baseline {
        return (window, peak, valley);
    }
    let cov = p.cov_c1;
    if p.extreme {
        peak = [1.3, 1.0, 1.5, 1.3];
        valley = [1.0, 1.0, 1.1, 1.0];
        window = 20;
    } else if cov < 0.80 {
        let w = ((0.80 - cov) / 0.20).clamp(0.0, 1.0);
        peak = [1.0 - 0.40 * w, 1.0 - 0.20 * w, 1.0 - 0.30 * w, 1.0 - 0.35 * w];
        valley = [1.0 - 0.30 * w, 1.0 - 0.10 * w, 1.0 - 0.20 * w, 1.0 - 0.25 * w];
        window = 30 + (20.0 * w) as i64;
    } else if cov > 0.95 {
        let w = ((cov - 0.95) / 0.05).clamp(0.0, 1.0);
        peak = [1.0 + 0.20 * w, 1.0, 1.0 + 0.30 * w, 1.0 + 0.25 * w];
        valley = [1.0, 1.0, 1.0, 1.0];
        window = 30 - (10.0 * w) as i64;
    }
    (window, peak, valley)
}

/// 从内存计算（状态化）：直方图 → 状态化阈值 → per-stock 并行 → 归约/补全/状态列。
/// 窗口与阈值缩放由 params 派生（方案 2 计算代码的核心生效点）。
pub fn compute_from_memory(
    codes: &[String],
    trades: Vec<Option<Vec<TradeRecord>>>,
    params: &RegimeParams,
) -> (Vec<String>, Vec<f32>) {
    let (window, peak, valley) = derive_compute_params(params);
    peaks_metrics::compute_peaks_state_from_trades(codes, trades, window, &peak, &valley)
}

// ============================================================
// PyO3 导出（调试用：单日状态拟合）
// ============================================================

/// 调试：给定 state_store_dir 与 date，拟合当日状态参数（返回可读 dict）。
#[pyfunction]
pub fn py_peaks_regime_fit(
    state_store_dir: String,
    date: i64,
    window: usize,
    trading_days: Vec<i64>,
) -> PyResult<PyObject> {
    let (_dates, rows) = state_read_window(&state_store_dir, date, window, &trading_days);
    let p = fit_regime(&rows, (window as f64 * 0.8) as usize);
    Python::with_gil(|py| {
        let d = pyo3::types::PyDict::new(py);
        d.set_item("cluster", p.cluster)?;
        d.set_item("pc1", p.pc1)?;
        d.set_item("extreme", p.extreme)?;
        d.set_item("sell_pressure", p.sell_pressure)?;
        d.set_item("cov_c1", p.cov_c1)?;
        d.set_item("is_baseline", p.is_baseline)?;
        d.set_item("scalars", p.scalars)?;
        Ok(d.into())
    })
}

/// 调试：读某日标量行。
#[pyfunction]
pub fn py_state_read_row(state_store_dir: String, date: i64) -> PyResult<Option<Vec<f64>>> {
    Ok(state_read_row(&state_store_dir, date))
}

// ============================================================
// 测试
// ============================================================
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fit_regime_baseline_when_empty() {
        let p = fit_regime(&[], 5);
        assert!(p.is_baseline);
    }

    #[test]
    fn test_jacobi_identity() {
        let mut m = [[0.0f64; N_STATE]; N_STATE];
        for i in 0..N_STATE {
            m[i][i] = 1.0;
        }
        let (vals, _vecs) = jacobi_eigen(&m);
        assert!((vals[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_state_roundtrip() {
        let dir = std::env::temp_dir().join("peaks_state_test");
        let dir = dir.to_str().unwrap();
        let scalars = vec![1.0f64; N_STATE];
        state_write_row(dir, 20240101, &scalars).unwrap();
        let r = state_read_row(dir, 20240101).unwrap();
        assert_eq!(r.len(), N_STATE);
        let _ = fs::remove_dir_all(dir);
    }
}
