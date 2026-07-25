//! 盘口快照挂单失衡模拟退火恢复因子。
//!
//! 读盘口快照 → 构造 4 种失衡序列（一档差/五档差/十档差/总差）→
//! 每序列按 6 窗口 × 4 分位切 24 段 → 每段跑确定性模拟退火 → 提取 25 因子。
//! 另有逐分钟（237 分钟）× 4 种失衡 × 25 因子 = 237×100 矩阵，
//! 经 get_features_factors_rust_full 降维。
//!
//! 退火引擎复用的 anneal_volume_metrics::anneal（确定性 xorshift64）。
//! 同股同日反复运算结果逐比特相同。

use crate::anneal_volume_metrics::{
    adaptive_m_max, anneal, AnnealBuf, Quantile, FACTOR_NAMES, M_MAX_MINUTE, M_MAX_SCALAR,
    N_FACTORS, N_MINUTES, WINDOW_BOUNDS, WINDOW_NAMES,
};
use crate::fast_csv_reader::{read_market_fast_inner, MarketRecord};
use crate::features;
use ndarray::Array2;
use pyo3::prelude::*;

// ============================================================================
// 失衡类型定义
// ============================================================================

#[derive(Clone, Copy, PartialEq)]
enum ImbType {
    Level1,
    Level5,
    Level10,
    Total,
}

impl ImbType {
    fn all() -> [ImbType; 4] {
        [
            ImbType::Level1,
            ImbType::Level5,
            ImbType::Level10,
            ImbType::Total,
        ]
    }

    fn as_str(&self) -> &'static str {
        match self {
            ImbType::Level1 => "imb1",
            ImbType::Level5 => "imb5",
            ImbType::Level10 => "imb10",
            ImbType::Total => "totimb",
        }
    }

    /// 从一条快照提取该类型的失衡值（bid - ask，纯差）。
    #[inline]
    fn extract(&self, m: &MarketRecord) -> f32 {
        match self {
            ImbType::Level1 => m.bid_vols[0] - m.ask_vols[0],
            ImbType::Level5 => {
                m.bid_vols[..5].iter().sum::<f32>() - m.ask_vols[..5].iter().sum::<f32>()
            }
            ImbType::Level10 => {
                m.bid_vols.iter().sum::<f32>() - m.ask_vols.iter().sum::<f32>()
            }
            ImbType::Total => m.total_bid_vol - m.total_ask_vol,
        }
    }
}

// ============================================================================
// 常量
// ============================================================================

pub const N_IMB_TYPES: usize = 4;
pub const N_QUANTILES: usize = 4; // all, top10, mid50, bot40
pub const N_WINDOWS: usize = 6;
pub const N_SEGMENTS_PER_TYPE: usize = N_WINDOWS * N_QUANTILES; // 24
pub const N_SCALAR: usize = N_IMB_TYPES * N_SEGMENTS_PER_TYPE * N_FACTORS; // 2400
pub const N_MINUTE_COLS: usize = N_IMB_TYPES * N_FACTORS; // 100
/// 降维后列数 = 21 * 100 + C(100,2) = 2100 + 4950
pub const N_REDUCED: usize = 21 * N_MINUTE_COLS + N_MINUTE_COLS * (N_MINUTE_COLS - 1) / 2;
pub const EXPECTED_LEN: usize = N_SCALAR + N_REDUCED; // 2400 + 7050 = 9450

// ============================================================================
// 片段定义 = 4 失衡 × 6 窗口 × 4 分位
// ============================================================================

fn segment_defs() -> Vec<(ImbType, usize, Quantile)> {
    let quantiles = [Quantile::All, Quantile::Top10, Quantile::Mid50, Quantile::Bot40];
    let mut segs = Vec::with_capacity(N_IMB_TYPES * N_WINDOWS * N_QUANTILES);
    for &imb in ImbType::all().iter() {
        for win_idx in 0..N_WINDOWS {
            for &q in &quantiles {
                segs.push((imb, win_idx, q));
            }
        }
    }
    debug_assert_eq!(segs.len(), N_SEGMENTS_PER_TYPE * N_IMB_TYPES);
    segs
}

// ============================================================================
// 核心计算
// ============================================================================

/// 核心唯一真相源：pipeline 和 Python 入口的共同调用点。
/// 读盘口快照 → 4 种失衡 → 窗口/分位/退火 → 逐分钟退火 → 降维。
pub fn compute_anneal_volume_market_full(code: &str, date: i64) -> std::io::Result<Vec<f32>> {
    use std::time::Instant;
    let t_total = Instant::now();

    let t_read = Instant::now();
    let market = read_market_fast_inner(code, date, false, false, usize::MAX)?;
    eprintln!(
        "[prof] {} {} market_read: {:?}  n_snap={}",
        code,
        date,
        t_read.elapsed(),
        market.len()
    );

    if market.is_empty() {
        return Ok(vec![f32::NAN; EXPECTED_LEN]);
    }

    let t_open = market.first().unwrap().time_sec;

    // ── 标量片段 ──
    let t_scalar = Instant::now();
    let segs = segment_defs();
    let mut buf = AnnealBuf::new();
    let mut out: Vec<f32> = Vec::with_capacity(EXPECTED_LEN);

    // 缓存：预提取每个 (imb_type, window) 的失衡序列
    let cache_len = N_IMB_TYPES * N_WINDOWS;
    let mut window_seqs: Vec<Option<Vec<f32>>> = (0..cache_len).map(|_| None).collect();
    let mut quantile_orders: Vec<Option<Vec<(f32, usize)>>> =
        (0..cache_len).map(|_| None).collect();

    for &(imb, win_idx, quantile) in &segs {
        let cache_idx = imb_type_index(imb) * N_WINDOWS + win_idx;
        if window_seqs[cache_idx].is_none() {
            let (sec_lo, sec_hi) = WINDOW_BOUNDS[win_idx];
            let lo_time = t_open + sec_lo;
            let hi_time = t_open + sec_hi;
            let lo = market.partition_point(|m| m.time_sec < lo_time);
            let hi = market.partition_point(|m| m.time_sec < hi_time);
            let vals: Vec<f32> = market[lo..hi].iter().map(|m| imb.extract(m)).collect();
            window_seqs[cache_idx] = Some(vals);
        }
        let vals = window_seqs[cache_idx].as_ref().unwrap();

        let filtered = if quantile == Quantile::All {
            vals.clone()
        } else {
            if quantile_orders[cache_idx].is_none() {
                let mut order: Vec<(f32, usize)> =
                    vals.iter().enumerate().map(|(i, &v)| (v, i)).collect();
                order.sort_by(|a, b| {
                    b.0.partial_cmp(&a.0)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                quantile_orders[cache_idx] = Some(order);
            }
            quantile_filter_from_sorted(
                vals,
                quantile_orders[cache_idx].as_ref().unwrap(),
                quantile,
            )
        };

        let m_adapt = adaptive_m_max(filtered.len(), M_MAX_SCALAR);
        let factors = anneal(&filtered, m_adapt, &mut buf);
        out.extend_from_slice(&factors);
    }
    eprintln!(
        "[prof] {} {} {}_scalar: {:?}",
        code,
        date,
        N_SCALAR,
        t_scalar.elapsed()
    );

    // ── 逐分钟矩阵 237 × 100 ──
    let t_minute = Instant::now();
    let mut matrix = Array2::zeros((N_MINUTES, N_MINUTE_COLS));
    let mut n_nonempty = 0usize;

    for m_idx in 0..N_MINUTES {
        let lo_time = t_open + (m_idx as f32) * 60.0;
        let hi_time = t_open + ((m_idx + 1) as f32) * 60.0;
        let lo = market.partition_point(|m| m.time_sec < lo_time);
        let hi = market.partition_point(|m| m.time_sec < hi_time);

        if lo >= hi {
            continue;
        }
        n_nonempty += 1;

        for (ti, &imb) in ImbType::all().iter().enumerate() {
            let vals: Vec<f32> = market[lo..hi].iter().map(|m| imb.extract(m)).collect();
            let m_adapt = adaptive_m_max(vals.len(), M_MAX_MINUTE);
            let factors = anneal(&vals, m_adapt, &mut buf);
            for (fi, &val) in factors.iter().enumerate() {
                matrix[[m_idx, ti * N_FACTORS + fi]] = val;
            }
        }
    }
    eprintln!(
        "[prof] {} {} 237_minute: {:?}  n_nonempty={}",
        code,
        date,
        t_minute.elapsed(),
        n_nonempty
    );

    // ── 降维 ──
    let t_reduce = Instant::now();
    let minute_col_names = build_minute_col_names();
    let (reduced_vals, _) =
        features::get_features_factors_rust_full(&matrix.view(), &minute_col_names, false);
    eprintln!(
        "[prof] {} {} dim_reduce: {:?}",
        code,
        date,
        t_reduce.elapsed()
    );
    out.extend_from_slice(&reduced_vals);

    // 长度校准
    if out.len() < EXPECTED_LEN {
        out.resize(EXPECTED_LEN, f32::NAN);
    } else if out.len() > EXPECTED_LEN {
        out.truncate(EXPECTED_LEN);
    }

    eprintln!(
        "[prof] {} {} TOTAL: {:?}",
        code,
        date,
        t_total.elapsed()
    );

    Ok(out)
}

#[inline]
fn imb_type_index(imb: ImbType) -> usize {
    match imb {
        ImbType::Level1 => 0,
        ImbType::Level5 => 1,
        ImbType::Level10 => 2,
        ImbType::Total => 3,
    }
}

/// 使用已按值降序排列的索引做分位数过滤，保留原始时间序。
fn quantile_filter_from_sorted(
    volumes: &[f32],
    sorted: &[(f32, usize)],
    q: Quantile,
) -> Vec<f32> {
    let n = volumes.len();
    if n == 0 || q == Quantile::All {
        return volumes.to_vec();
    }
    let (start, end) = match q {
        Quantile::Top10 => {
            let k = ((n as f64) * 0.1).ceil() as usize;
            (0, k.max(1).min(n))
        }
        Quantile::Mid50 => {
            let start = ((n as f64) * 0.1) as usize;
            let end = ((n as f64) * 0.6) as usize;
            (start, end.min(n).max(start))
        }
        Quantile::Bot40 => {
            let k = ((n as f64) * 0.4) as usize;
            (n - k, n)
        }
        Quantile::All => unreachable!(),
    };
    let mut keep = vec![false; n];
    for &(_, idx) in &sorted[start..end] {
        keep[idx] = true;
    }
    volumes
        .iter()
        .enumerate()
        .filter(|(i, _)| keep[*i])
        .map(|(_, &v)| v)
        .collect()
}

// ============================================================================
// 名字生成
// ============================================================================

fn build_minute_col_names() -> Vec<String> {
    let mut names = Vec::with_capacity(N_MINUTE_COLS);
    for imb in ImbType::all().iter() {
        for factor in FACTOR_NAMES {
            names.push(format!("min_{}_{}", imb.as_str(), factor));
        }
    }
    names
}

/// 生成全部 9450 个因子名（与 compute_anneal_volume_market_full 输出严格对齐）。
pub fn anneal_volume_market_names() -> Vec<String> {
    let mut names: Vec<String> = Vec::with_capacity(EXPECTED_LEN);

    // 标量片段名
    let segs = segment_defs();
    for &(imb, win_idx, quantile) in &segs {
        let seg_prefix = format!(
            "{}_{}_{}",
            imb.as_str(),
            WINDOW_NAMES[win_idx],
            quantile.as_str()
        );
        for factor in FACTOR_NAMES {
            names.push(format!("{}_{}", seg_prefix, factor));
        }
    }

    // 逐分钟降维名
    let minute_col_names = build_minute_col_names();
    let dummy = Array2::ones((1, N_MINUTE_COLS));
    let (_, reduced_names) =
        features::get_features_factors_rust_full(&dummy.view(), &minute_col_names, false);
    names.extend(reduced_names);

    while names.len() < EXPECTED_LEN {
        names.push(format!("extra_{}", names.len()));
    }
    names.truncate(EXPECTED_LEN);

    names
}

// ============================================================================
// PyO3 接口
// ============================================================================

#[pyfunction]
pub fn py_anneal_volume_market(code: &str, date: i64) -> PyResult<Vec<f32>> {
    compute_anneal_volume_market_full(code, date)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("{:?}", e)))
}

#[pyfunction]
pub fn py_anneal_volume_market_names() -> Vec<String> {
    anneal_volume_market_names()
}
