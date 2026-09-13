// ============================================================================
// sandbox_engine_opt: tail_backtest_engine 回测速度瓶颈分析与优化验证
//
// BASELINE: 从生产代码原样复刻 (tail_v2_rank_roll_factor.rs 的 rank/rolling,
//   tail_v5_pipeline.rs 的 preflight/backtest 段, factor_neutralize_std.rs 的
//   中性化), 数值应与生产引擎逐位一致。
// OPT:
//   O1 backtest: 收益秩预排序 (per-date ret_sum 全行排序, 因子无关) + 子集
//      walk 替代逐 slot 子集排序; 信号侧 radix 秩替代 sort_by。
//   O2 neutralize: 行业排序预计算 (等价 C'') + per-date X'X 预计算 (valid
//      集合与因子无关, 见 RANKIC_NEUTRALIZATION_REPORT 4.4)。
//   O3 rolling: 行主序滚动统计 (消除按列访问 stride cache miss)。
//   O4 结构: slot 级 rayon 并行。
// ============================================================================
use ndarray::{Array1, Array2, Array3, ArrayView1, ArrayView2, ArrayView3, s};
use nalgebra::{Cholesky, DMatrix};
use numpy::{IntoPyArray, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::prelude::*;
use rayon::prelude::*;
use std::collections::VecDeque;

// ---------------- A. rank (tail_v2_rank_roll_factor.rs 原样) ----------------
fn rank_average_row(row: &[f32]) -> Vec<f32> {
    let mut indexed = row
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, value)| !value.is_nan())
        .collect::<Vec<_>>();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    let mut ranked = vec![f32::NAN; row.len()];
    let mut start = 0usize;
    while start < indexed.len() {
        let value = indexed[start].1;
        let mut end = start + 1;
        while end < indexed.len() && indexed[end].1 == value {
            end += 1;
        }
        let avg_rank = ((start + 1 + end) as f64 / 2.0) as f32;
        for item in indexed.iter().take(end).skip(start) {
            ranked[item.0] = avg_rank;
        }
        start = end;
    }
    ranked
}

fn rank_axis1_average_f32_serial(data: &Array2<f32>) -> Array2<f32> {
    let (n_rows, n_cols) = data.dim();
    let mut flat = vec![f32::NAN; n_rows * n_cols];
    for row_idx in 0..n_rows {
        let row = data.row(row_idx);
        let ranked = rank_average_row(row.as_slice().unwrap_or(&[]));
        let start = row_idx * n_cols;
        flat[start..start + n_cols].copy_from_slice(&ranked);
    }
    Array2::from_shape_vec((n_rows, n_cols), flat).unwrap()
}

// ---------------- B. rolling (tail_v2_rank_roll_factor.rs 原样 + 行主序版) ----------------
fn rolling_stats_for_column(
    ranked: &Array2<f32>,
    col_idx: usize,
    window: usize,
    min_periods: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let n_rows = ranked.nrows();
    let mut mean_out = vec![f32::NAN; n_rows];
    let mut max_out = vec![f32::NAN; n_rows];
    let mut min_out = vec![f32::NAN; n_rows];
    let mut std_out = vec![f32::NAN; n_rows];
    let mut sum = 0.0f64;
    let mut sumsq = 0.0f64;
    let mut count = 0usize;
    let mut max_deque = VecDeque::<(usize, f32)>::new();
    let mut min_deque = VecDeque::<(usize, f32)>::new();
    for row_idx in 0..n_rows {
        let value = ranked[[row_idx, col_idx]];
        if !value.is_nan() {
            let value64 = value as f64;
            sum += value64;
            sumsq += value64 * value64;
            count += 1;
            while let Some((_, tail_val)) = max_deque.back() {
                if *tail_val <= value {
                    max_deque.pop_back();
                } else {
                    break;
                }
            }
            max_deque.push_back((row_idx, value));
            while let Some((_, tail_val)) = min_deque.back() {
                if *tail_val >= value {
                    min_deque.pop_back();
                } else {
                    break;
                }
            }
            min_deque.push_back((row_idx, value));
        }
        if row_idx >= window {
            let leave_idx = row_idx - window;
            let leave_value = ranked[[leave_idx, col_idx]];
            if !leave_value.is_nan() {
                let leave64 = leave_value as f64;
                sum -= leave64;
                sumsq -= leave64 * leave64;
                count -= 1;
            }
        }
        let valid_start = (row_idx + 1).saturating_sub(window);
        while let Some((idx, _)) = max_deque.front() {
            if *idx < valid_start {
                max_deque.pop_front();
            } else {
                break;
            }
        }
        while let Some((idx, _)) = min_deque.front() {
            if *idx < valid_start {
                min_deque.pop_front();
            } else {
                break;
            }
        }
        if count >= min_periods {
            let mean = sum / count as f64;
            mean_out[row_idx] = mean as f32;
            max_out[row_idx] = max_deque.front().map(|item| item.1).unwrap_or(f32::NAN);
            min_out[row_idx] = min_deque.front().map(|item| item.1).unwrap_or(f32::NAN);
            if count > 1 {
                let variance =
                    ((sumsq - (sum * sum) / count as f64) / (count as f64 - 1.0)).max(0.0);
                std_out[row_idx] = variance.sqrt() as f32;
            }
        }
    }
    (mean_out, max_out, min_out, std_out)
}

fn rolling_stats_f32_serial(
    ranked: &Array2<f32>,
    window: usize,
    min_periods: usize,
) -> (Array2<f32>, Array2<f32>, Array2<f32>, Array2<f32>) {
    let n_rows = ranked.nrows();
    let n_cols = ranked.ncols();
    let mut mean = Array2::<f32>::from_elem((n_rows, n_cols), f32::NAN);
    let mut max = Array2::<f32>::from_elem((n_rows, n_cols), f32::NAN);
    let mut min = Array2::<f32>::from_elem((n_rows, n_cols), f32::NAN);
    let mut std = Array2::<f32>::from_elem((n_rows, n_cols), f32::NAN);
    for col_idx in 0..n_cols {
        let (mean_col, max_col, min_col, std_col) =
            rolling_stats_for_column(ranked, col_idx, window, min_periods);
        for row_idx in 0..n_rows {
            mean[[row_idx, col_idx]] = mean_col[row_idx];
            max[[row_idx, col_idx]] = max_col[row_idx];
            min[[row_idx, col_idx]] = min_col[row_idx];
            std[[row_idx, col_idx]] = std_col[row_idx];
        }
    }
    (mean, max, min, std)
}

/// O3: 行主序滚动统计 — 每列独立滑动窗口状态, 逐行推进 (内存顺序访问)。
/// 每个输出元素与 rolling_stats_f32_serial 数学一致 (同一窗口/同一 NaN 语义)。
fn rolling_stats_rowmajor(
    ranked: &Array2<f32>,
    window: usize,
    min_periods: usize,
) -> (Array2<f32>, Array2<f32>, Array2<f32>, Array2<f32>) {
    let n_rows = ranked.nrows();
    let n_cols = ranked.ncols();
    let mut mean = Array2::<f32>::from_elem((n_rows, n_cols), f32::NAN);
    let mut max = Array2::<f32>::from_elem((n_rows, n_cols), f32::NAN);
    let mut min = Array2::<f32>::from_elem((n_rows, n_cols), f32::NAN);
    let mut std = Array2::<f32>::from_elem((n_rows, n_cols), f32::NAN);
    let mut sum = vec![0.0f64; n_cols];
    let mut sumsq = vec![0.0f64; n_cols];
    let mut count = vec![0usize; n_cols];
    let mut max_deq: Vec<VecDeque<(usize, f32)>> = vec![VecDeque::new(); n_cols];
    let mut min_deq: Vec<VecDeque<(usize, f32)>> = vec![VecDeque::new(); n_cols];
    let row = ranked.as_slice().unwrap();
    for row_idx in 0..n_rows {
        let base = row_idx * n_cols;
        for col_idx in 0..n_cols {
            let value = row[base + col_idx];
            if !value.is_nan() {
                let value64 = value as f64;
                sum[col_idx] += value64;
                sumsq[col_idx] += value64 * value64;
                count[col_idx] += 1;
                {
                    let mq = &mut max_deq[col_idx];
                    while let Some((_, tail_val)) = mq.back() {
                        if *tail_val <= value {
                            mq.pop_back();
                        } else {
                            break;
                        }
                    }
                    mq.push_back((row_idx, value));
                }
                {
                    let mq = &mut min_deq[col_idx];
                    while let Some((_, tail_val)) = mq.back() {
                        if *tail_val >= value {
                            mq.pop_back();
                        } else {
                            break;
                        }
                    }
                    mq.push_back((row_idx, value));
                }
            }
        }
        if row_idx >= window {
            let leave_base = (row_idx - window) * n_cols;
            for col_idx in 0..n_cols {
                let leave_value = row[leave_base + col_idx];
                if !leave_value.is_nan() {
                    let leave64 = leave_value as f64;
                    sum[col_idx] -= leave64;
                    sumsq[col_idx] -= leave64 * leave64;
                    count[col_idx] -= 1;
                }
            }
        }
        let valid_start = (row_idx + 1).saturating_sub(window);
        for col_idx in 0..n_cols {
            while let Some((idx, _)) = max_deq[col_idx].front() {
                if *idx < valid_start {
                    max_deq[col_idx].pop_front();
                } else {
                    break;
                }
            }
            while let Some((idx, _)) = min_deq[col_idx].front() {
                if *idx < valid_start {
                    min_deq[col_idx].pop_front();
                } else {
                    break;
                }
            }
            if count[col_idx] >= min_periods {
                let m = sum[col_idx] / count[col_idx] as f64;
                mean[[row_idx, col_idx]] = m as f32;
                max[[row_idx, col_idx]] =
                    max_deq[col_idx].front().map(|it| it.1).unwrap_or(f32::NAN);
                min[[row_idx, col_idx]] =
                    min_deq[col_idx].front().map(|it| it.1).unwrap_or(f32::NAN);
                if count[col_idx] > 1 {
                    let variance = ((sumsq[col_idx]
                        - (sum[col_idx] * sum[col_idx]) / count[col_idx] as f64)
                        / (count[col_idx] as f64 - 1.0))
                        .max(0.0);
                    std[[row_idx, col_idx]] = variance.sqrt() as f32;
                }
            }
        }
    }
    (mean, max, min, std)
}

// ---------------- C. fill missing (tail_v5_pipeline.rs 原样) ----------------
const UNIVERSE_LOOKBACK: usize = 20;

fn fill_missing_rank_with_cross_sectional_median(ranked: &mut Array2<f32>, active: &Array2<f32>) {
    let n_dates = ranked.nrows();
    let n_stocks = ranked.ncols();
    let mut last_valid: Vec<Option<usize>> = vec![None; n_stocks];
    for date_idx in 0..n_dates {
        let mut valid_count = 0usize;
        for stock_idx in 0..n_stocks {
            if ranked[[date_idx, stock_idx]].is_finite() {
                valid_count += 1;
            }
            if active[[date_idx, stock_idx]].is_finite() {
                last_valid[stock_idx] = Some(date_idx);
            }
        }
        if valid_count == 0 {
            continue;
        }
        let median_rank = ((valid_count + 1) as f32) / 2.0;
        for stock_idx in 0..n_stocks {
            if !ranked[[date_idx, stock_idx]].is_finite() {
                let is_missing = last_valid[stock_idx]
                    .map(|last| date_idx - last <= UNIVERSE_LOOKBACK)
                    .unwrap_or(false);
                if is_missing {
                    ranked[[date_idx, stock_idx]] = median_rank;
                }
            }
        }
    }
}

// ---------------- D. preflight (tail_v5_pipeline.rs 原样) ----------------
struct PreflightReport {
    passed: bool,
    majority_count_mean: f64,
    zero_ratio_mean: f64,
    nan_ratio_mean: f64,
}

fn preflight_quality_check(
    raw_values: &ArrayView2<f32>,
    restrict: &ArrayView2<f32>,
    majority_count_threshold: f64,
    zero_max_threshold: f64,
    nan_max_threshold: f64,
) -> PreflightReport {
    let n_dates = raw_values.shape()[0];
    let n_stocks = raw_values.shape()[1];
    let mut majority_sum: f64 = 0.0;
    let mut nan_ratio_sum: f64 = 0.0;
    let mut zero_ratio_sum: f64 = 0.0;
    let mut valid_date_count: usize = 0;
    for t in 0..n_dates {
        let mut value_counts: std::collections::HashMap<u32, usize> = std::collections::HashMap::new();
        let mut free_count: usize = 0;
        let mut nan_count: usize = 0;
        let mut zero_count: usize = 0;
        for s in 0..n_stocks {
            let val = raw_values[[t, s]];
            let is_free = restrict[[t, s]].is_finite() && restrict[[t, s]] == 0.0;
            if is_free {
                free_count += 1;
                if !val.is_finite() {
                    nan_count += 1;
                } else if val == 0.0 {
                    zero_count += 1;
                }
            }
            if val.is_finite() {
                *value_counts.entry(val.to_bits()).or_insert(0) += 1;
            }
        }
        let max_count = value_counts.values().max().copied().unwrap_or(0);
        majority_sum += max_count as f64;
        if free_count > 0 {
            nan_ratio_sum += nan_count as f64 / free_count as f64;
            zero_ratio_sum += zero_count as f64 / free_count as f64;
            valid_date_count += 1;
        }
    }
    let majority_count_mean = if n_dates > 0 {
        majority_sum / n_dates as f64
    } else {
        0.0
    };
    let nan_ratio_mean = if valid_date_count > 0 {
        nan_ratio_sum / valid_date_count as f64
    } else {
        0.0
    };
    let zero_ratio_mean = if valid_date_count > 0 {
        zero_ratio_sum / valid_date_count as f64
    } else {
        0.0
    };
    PreflightReport {
        passed: majority_count_mean <= majority_count_threshold
            && zero_ratio_mean < zero_max_threshold
            && nan_ratio_mean < nan_max_threshold,
        majority_count_mean,
        zero_ratio_mean,
        nan_ratio_mean,
    }
}

// ---------------- E. backtest (tail_v5_pipeline.rs 原样 + O1) ----------------
const EPS: f64 = 1e-12;

fn nanmean_f64(values: &[f64]) -> f64 {
    let mut sum = 0.0;
    let mut count = 0usize;
    for &v in values {
        if v.is_nan() {
            continue;
        }
        sum += v;
        count += 1;
    }
    if count == 0 {
        f64::NAN
    } else {
        sum / count as f64
    }
}

fn nanstd_population(values: &[f64]) -> f64 {
    let mut sum = 0.0;
    let mut sumsq = 0.0;
    let mut count = 0usize;
    for &v in values {
        if v.is_nan() {
            continue;
        }
        sum += v;
        sumsq += v * v;
        count += 1;
    }
    if count == 0 {
        f64::NAN
    } else {
        ((sumsq - sum * sum / count as f64) / count as f64)
            .max(0.0)
            .sqrt()
    }
}

fn annualized_sharpe_sample(values: &[f64]) -> f64 {
    let mut sum = 0.0;
    let mut sumsq = 0.0;
    let mut count = 0usize;
    for &v in values {
        if v.is_nan() {
            continue;
        }
        sum += v;
        sumsq += v * v;
        count += 1;
    }
    if count < 2 {
        return f64::NAN;
    }
    let mean = sum / count as f64;
    let variance = ((sumsq - sum * sum / count as f64) / (count as f64 - 1.0)).max(0.0);
    let std = variance.sqrt();
    if std <= EPS {
        f64::NAN
    } else {
        mean / std * 250.0f64.sqrt()
    }
}

fn max_drawdown_from_returns(values: &[f64]) -> f64 {
    let mut peak = f64::NEG_INFINITY;
    let mut max_dd = 0.0f64;
    for &v in values {
        if v.is_nan() {
            continue;
        }
        if v > peak {
            peak = v;
        }
        let dd = peak - v;
        if dd > max_dd {
            max_dd = dd;
        }
    }
    max_dd
}

fn average_ranks(values: &[f32]) -> Vec<f64> {
    let mut indexed = values
        .iter()
        .copied()
        .enumerate()
        .collect::<Vec<(usize, f32)>>();
    indexed.sort_by(|lhs, rhs| {
        lhs.1
            .partial_cmp(&rhs.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| lhs.0.cmp(&rhs.0))
    });
    let mut ranks = vec![f64::NAN; values.len()];
    let mut start = 0usize;
    while start < indexed.len() {
        let value = indexed[start].1;
        let mut end = start + 1;
        while end < indexed.len() && indexed[end].1 == value {
            end += 1;
        }
        let avg_rank = (start + 1 + end) as f64 / 2.0;
        for item in indexed.iter().take(end).skip(start) {
            ranks[item.0] = avg_rank;
        }
        start = end;
    }
    ranks
}

fn ordinal_ranks(values: &[f32]) -> Vec<i64> {
    let mut indexed = values
        .iter()
        .copied()
        .enumerate()
        .collect::<Vec<(usize, f32)>>();
    indexed.sort_by(|lhs, rhs| match (lhs.1.is_nan(), rhs.1.is_nan()) {
        (true, true) => lhs.0.cmp(&rhs.0),
        (true, false) => std::cmp::Ordering::Greater,
        (false, true) => std::cmp::Ordering::Less,
        (false, false) => lhs
            .1
            .partial_cmp(&rhs.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| lhs.0.cmp(&rhs.0)),
    });
    let mut ranks = vec![0i64; values.len()];
    for (rank, (idx, _)) in indexed.iter().enumerate() {
        ranks[*idx] = rank as i64;
    }
    ranks
}

fn legacy_spearman_correlation(x: &[f32], y: &[f32]) -> f64 {
    if x.len() != y.len() || x.len() < 2 {
        return f64::NAN;
    }
    let xx = ordinal_ranks(x);
    let yy = ordinal_ranks(y);
    let n = x.len() as f64;
    let mut diff_sq_sum = 0.0;
    for idx in 0..x.len() {
        let diff = xx[idx] - yy[idx];
        diff_sq_sum += (diff * diff) as f64;
    }
    1.0 - 6.0 * diff_sq_sum / (n * (n * n - 1.0))
}

#[inline]
fn mono_key32(v: f32) -> u32 {
    let v = if v == 0.0 { 0.0 } else { v };
    let bits = v.to_bits();
    if bits >> 31 == 0 {
        bits ^ 0x8000_0000
    } else {
        !bits
    }
}

/// 稳定 radix: 按 (mono_key32(v), index) 排序, 与 sort_by(partial_cmp then index) 同序
/// (调用方保证无 NaN; -0.0 已规范化)。
fn radix_sort_idx_keys(keys: &mut [u64], order: &mut Vec<usize>, tmp: &mut Vec<usize>) {
    let n = order.len();
    if n < 2 {
        return;
    }
    tmp.clear();
    tmp.resize(n, 0);
    let mut count = [0usize; 256];
    for shift in (0..64).step_by(8) {
        count.fill(0);
        for &i in order.iter() {
            count[((keys[i] >> shift) & 0xff) as usize] += 1;
        }
        let mut acc = 0usize;
        for c in count.iter_mut() {
            let t = *c;
            *c = acc;
            acc += t;
        }
        for &i in order.iter() {
            let b = ((keys[i] >> shift) & 0xff) as usize;
            tmp[count[b]] = i;
            count[b] += 1;
        }
        std::mem::swap(order, tmp);
    }
}

fn ordinal_ranks_radix(values: &[f32]) -> Vec<i64> {
    let n = values.len();
    let mut keys = vec![0u64; n];
    for (i, &v) in values.iter().enumerate() {
        keys[i] = (mono_key32(v) as u64) << 32 | i as u64;
    }
    let mut order: Vec<usize> = (0..n).collect();
    let mut tmp: Vec<usize> = Vec::new();
    radix_sort_idx_keys(&mut keys, &mut order, &mut tmp);
    let mut ranks = vec![0i64; n];
    for (rank, &idx) in order.iter().enumerate() {
        ranks[idx] = rank as i64;
    }
    ranks
}

fn average_ranks_radix(values: &[f32]) -> Vec<f64> {
    let n = values.len();
    let mut keys = vec![0u64; n];
    for (i, &v) in values.iter().enumerate() {
        keys[i] = (mono_key32(v) as u64) << 32 | i as u64;
    }
    let mut order: Vec<usize> = (0..n).collect();
    let mut tmp: Vec<usize> = Vec::new();
    radix_sort_idx_keys(&mut keys, &mut order, &mut tmp);
    let mut ranks = vec![f64::NAN; n];
    let mut start = 0usize;
    while start < n {
        let value = values[order[start]];
        let mut end = start + 1;
        while end < n && values[order[end]] == value {
            end += 1;
        }
        let avg_rank = (start + 1 + end) as f64 / 2.0;
        for &idx in order.iter().take(end).skip(start) {
            ranks[idx] = avg_rank;
        }
        start = end;
    }
    ranks
}

fn count_open_symbols(restrict_row: ArrayView1<'_, f32>) -> usize {
    restrict_row
        .iter()
        .filter(|&&value| value.is_finite() && value == 0.0)
        .count()
}

fn precompute_open_symbol_counts(restrict: &ArrayView2<'_, f32>) -> Vec<usize> {
    (0..restrict.shape()[0])
        .map(|row_idx| count_open_symbols(restrict.row(row_idx)))
        .collect()
}

fn has_enough_unique_values(factor: &ArrayView3<'_, f32>, slot_idx: usize, min_unique: usize) -> bool {
    let mut seen = std::collections::HashSet::<u32>::new();
    for raw_idx in 0..factor.shape()[0].saturating_sub(1) {
        for stock_idx in 0..factor.shape()[1] {
            let value = factor[[raw_idx, stock_idx, slot_idx]];
            if value.is_finite() {
                seen.insert(value.to_bits());
                if seen.len() >= min_unique {
                    return true;
                }
            }
        }
    }
    false
}

fn effective_raw_indices_for_slot(
    factor: &ArrayView3<'_, f32>,
    dates: &[i32],
    backtest_start: i32,
    slot_idx: usize,
) -> Vec<usize> {
    let n_dates = factor.shape()[0];
    let n_stocks = factor.shape()[1];
    let mut effective_raw_indices = Vec::<usize>::new();
    for raw_eff_idx in 1..n_dates {
        if dates[raw_eff_idx] <= backtest_start {
            continue;
        }
        let signal_row_idx = raw_eff_idx - 1;
        let mut all_nan = true;
        for stock_idx in 0..n_stocks {
            if factor[[signal_row_idx, stock_idx, slot_idx]].is_finite() {
                all_nan = false;
                break;
            }
        }
        if !all_nan {
            effective_raw_indices.push(raw_eff_idx);
        }
    }
    effective_raw_indices
}

#[derive(Clone)]
struct LegacyBacktestResult {
    summary: [f64; 10],
    ic_dates: Vec<i32>,
    ic_values: Vec<f32>,
}

fn default_legacy_backtest_result() -> LegacyBacktestResult {
    LegacyBacktestResult {
        summary: [f64::NAN; 10],
        ic_dates: Vec::new(),
        ic_values: Vec::new(),
    }
}

fn finish_summary(
    ic_values_f64: &[f64],
    ratio_values: &[f64],
    group_returns: &Vec<Vec<f64>>,
    effective_raw_indices: &[usize],
    index: &Array1<f32>,
    gap: usize,
    date_size: usize,
    ic_only: bool,
) -> [f64; 10] {
    let ic_mean = nanmean_f64(ic_values_f64);
    let ic_std = nanstd_population(ic_values_f64);
    let ir = if ic_std.is_nan() || ic_std <= EPS {
        f64::NAN
    } else {
        ic_mean.abs() / ic_std * (250.0 / gap as f64).sqrt()
    };
    if ic_only {
        return [
            ic_mean,
            ir,
            0.0,
            0.0,
            0.0,
            date_size as f64,
            nanmean_f64(ratio_values),
            0.0,
            0.0,
            0.0,
        ];
    }
    let portf_num = group_returns.len();
    let first_leg_cum = group_returns[0].iter().sum::<f64>();
    let last_leg_cum = group_returns[portf_num - 1].iter().sum::<f64>();
    let (long_idx, short_idx) = if first_leg_cum > last_leg_cum {
        (0usize, portf_num - 1)
    } else {
        (portf_num - 1, 0usize)
    };
    let mut ls_returns = vec![0.0_f64; date_size];
    let mut hedge_returns = vec![0.0_f64; date_size];
    for (local_t, &raw_eff_idx) in effective_raw_indices.iter().enumerate() {
        let long_ret = group_returns[long_idx][local_t];
        let short_ret = group_returns[short_idx][local_t];
        ls_returns[local_t] = long_ret - short_ret;
        hedge_returns[local_t] = long_ret - index[raw_eff_idx] as f64;
    }
    [
        ic_mean,
        ir,
        nanmean_f64(&ls_returns) * 250.0,
        annualized_sharpe_sample(&ls_returns),
        max_drawdown_from_returns(&ls_returns),
        date_size as f64,
        nanmean_f64(ratio_values),
        nanmean_f64(&hedge_returns) * 250.0,
        annualized_sharpe_sample(&hedge_returns),
        max_drawdown_from_returns(&hedge_returns),
    ]
}

/// BASELINE: 生产 legacy_backtest_single_factor_with_effective 原样。
#[allow(clippy::too_many_arguments)]
fn legacy_backtest_single_factor_with_effective(
    factor: &ArrayView3<'_, f32>,
    ret: &Array2<f32>,
    ret_sum: &Array2<f32>,
    restrict: &Array2<f32>,
    index: &Array1<f32>,
    dates: &[i32],
    slot_idx: usize,
    gap: usize,
    portf_num: usize,
    effective_raw_indices: &[usize],
    open_symbol_counts: &[usize],
    ic_only: bool,
) -> LegacyBacktestResult {
    if effective_raw_indices.is_empty() {
        return default_legacy_backtest_result();
    }
    let n_stocks = factor.shape()[1];
    let date_size = effective_raw_indices.len();
    let mut group_returns = if ic_only {
        Vec::new()
    } else {
        vec![vec![0.0_f64; date_size]; portf_num]
    };
    let mut ratio_values = vec![f64::NAN; date_size];
    let mut ic_dates = Vec::<i32>::new();
    let mut ic_values_f64 = Vec::<f64>::new();
    let mut ic_values_f32 = Vec::<f32>::new();
    let mut filtered_signal = Vec::<f32>::with_capacity(n_stocks);
    let mut filtered_ret = Vec::<f32>::with_capacity(n_stocks);
    let mut filtered_future = Vec::<f32>::with_capacity(n_stocks);
    let mut group_sums = vec![0.0_f64; portf_num];
    let mut group_counts = vec![0usize; portf_num];
    let mut held_signal_row_idx = effective_raw_indices[0] - 1;
    let mut held_restrict_row_idx = effective_raw_indices[0] - 1;

    for (local_t, &raw_eff_idx) in effective_raw_indices.iter().enumerate() {
        if local_t % gap == 0 {
            held_signal_row_idx = raw_eff_idx - 1;
            held_restrict_row_idx = raw_eff_idx - 1;
        }
        filtered_signal.clear();
        filtered_ret.clear();
        filtered_future.clear();
        for stock_idx in 0..n_stocks {
            let signal_value = factor[[held_signal_row_idx, stock_idx, slot_idx]];
            let ret_value = ret[[raw_eff_idx, stock_idx]];
            let is_open = restrict[[held_restrict_row_idx, stock_idx]].is_finite()
                && restrict[[held_restrict_row_idx, stock_idx]] == 0.0;
            if signal_value.is_finite() && ret_value.is_finite() && is_open {
                filtered_signal.push(signal_value);
                filtered_ret.push(ret_value);
                filtered_future.push(ret_sum[[raw_eff_idx, stock_idx]]);
            }
        }
        if (local_t + 1) % gap == 0 {
            let ic_value = legacy_spearman_correlation(&filtered_future, &filtered_signal);
            ic_dates.push(dates[raw_eff_idx]);
            ic_values_f64.push(ic_value);
            ic_values_f32.push(ic_value as f32);
        }
        let stocks_num = filtered_signal.len();
        if stocks_num < portf_num {
            continue;
        }
        let valid_symbol_num = open_symbol_counts
            .get(raw_eff_idx - 1)
            .copied()
            .unwrap_or(0);
        if valid_symbol_num > 0 {
            ratio_values[local_t] = stocks_num as f64 / valid_symbol_num as f64;
        }
        if ic_only {
            continue;
        }
        group_sums.fill(0.0);
        group_counts.fill(0);
        let ranks = average_ranks(&filtered_signal);
        for idx in 0..stocks_num {
            let pct = ranks[idx] / stocks_num as f64;
            let mut bucket = (pct * portf_num as f64).floor() as usize;
            if bucket >= portf_num {
                bucket = portf_num - 1;
            }
            group_sums[bucket] += filtered_ret[idx] as f64;
            group_counts[bucket] += 1;
        }
        for bucket in 0..portf_num {
            group_returns[bucket][local_t] = if group_counts[bucket] == 0 {
                0.0
            } else {
                group_sums[bucket] / group_counts[bucket] as f64
            };
        }
    }
    LegacyBacktestResult {
        summary: finish_summary(
            &ic_values_f64,
            &ratio_values,
            &group_returns,
            effective_raw_indices,
            index,
            gap,
            date_size,
            ic_only,
        ),
        ic_dates,
        ic_values: ic_values_f32,
    }
}

/// O1 预计算 (因子无关): 收益秩全行排序。
/// 注意: "开市/收益有效候选列表" 曾按 (date, held_h_off) 预计算, 但 held 行随
/// effective 序列的段边界变化 (低覆盖率因子存在跳天), h_off 可能超出固定范围,
/// 导致取错 restrict 行 (col=1000 因子实测 maxdiff=2e-3)。已回退为与生产相同的
/// 全扫描过滤, 保留收益秩预排序 walk 与 radix 秩 (这两项才是大头)。
struct BtPrecomputed {
    orders: [Vec<Vec<u32>>; 2],          // [g][date] -> 按 (ret_sum 值, index) 排序的股票索引
}

fn build_bt_precomputed(
    ret_sum_g1: &Array2<f32>,
    ret_sum_g5: &Array2<f32>,
    _ret_g1: &Array2<f32>,
    _ret_g5: &Array2<f32>,
    _restrict: &Array2<f32>,
) -> BtPrecomputed {
    let n_dates = ret_sum_g1.nrows();
    let n_stocks = ret_sum_g1.ncols();
    let mut orders: [Vec<Vec<u32>>; 2] = [Vec::new(), Vec::new()];
    for (g, ret_sum) in [(0usize, ret_sum_g1), (1usize, ret_sum_g5)] {
        let mut ords = Vec::with_capacity(n_dates);
        for d in 0..n_dates {
            let mut keys: Vec<u64> = Vec::with_capacity(n_stocks);
            for (j, &v) in ret_sum.row(d).iter().enumerate() {
                let k = if v.is_nan() { u32::MAX } else { mono_key32(v) };
                keys.push((k as u64) << 32 | j as u64);
            }
            let mut order: Vec<usize> = (0..n_stocks).collect();
            let mut tmp: Vec<usize> = Vec::new();
            radix_sort_idx_keys(&mut keys, &mut order, &mut tmp);
            ords.push(order.into_iter().map(|x| x as u32).collect());
        }
        orders[g] = ords;
    }
    BtPrecomputed { orders }
}

#[allow(clippy::too_many_arguments)]
fn finish_ic_from_parts(
    ic_dates: &mut Vec<i32>,
    ic_values_f64: &mut Vec<f64>,
    ic_values_f32: &mut Vec<f32>,
    future_ranks: &[i64],
    signal_ranks: &[i64],
    count: usize,
    date: i32,
) {
    let n = count as f64;
    let mut diff_sq_sum = 0.0;
    for idx in 0..count {
        let diff = future_ranks[idx] - signal_ranks[idx];
        diff_sq_sum += (diff * diff) as f64;
    }
    let ic_value = if n < 2.0 {
        f64::NAN
    } else {
        1.0 - 6.0 * diff_sq_sum / (n * (n * n - 1.0))
    };
    ic_dates.push(date);
    ic_values_f64.push(ic_value);
    ic_values_f32.push(ic_value as f32);
}

/// O1 backtest: 收益秩 = 全行预排序 walk 出的子集 ordinal 秩; 信号秩 = radix。
/// 与 BASELINE 数值一致 (同 tie-break/同 NaN 语义/同累积顺序)。
#[allow(clippy::too_many_arguments)]
fn backtest_opt(
    factor: &ArrayView3<'_, f32>,
    ret: &Array2<f32>,
    ret_sum: &Array2<f32>,
    restrict: &Array2<f32>,
    index: &Array1<f32>,
    dates: &[i32],
    slot_idx: usize,
    gap: usize,
    portf_num: usize,
    effective_raw_indices: &[usize],
    open_symbol_counts: &[usize],
    ic_only: bool,
    pre: &BtPrecomputed,
) -> LegacyBacktestResult {
    if effective_raw_indices.is_empty() {
        return default_legacy_backtest_result();
    }
    let n_stocks = factor.shape()[1];
    let g = if gap == 1 { 0 } else { 1 };
    let date_size = effective_raw_indices.len();
    let mut group_returns = if ic_only {
        Vec::new()
    } else {
        vec![vec![0.0_f64; date_size]; portf_num]
    };
    let mut ratio_values = vec![f64::NAN; date_size];
    let mut ic_dates = Vec::<i32>::new();
    let mut ic_values_f64 = Vec::<f64>::new();
    let mut ic_values_f32 = Vec::<f32>::new();
    let mut filtered_signal = Vec::<f32>::with_capacity(n_stocks);
    let mut filtered_ret = Vec::<f32>::with_capacity(n_stocks);
    let mut filtered_stock_idx = Vec::<u32>::with_capacity(n_stocks);
    let mut group_sums = vec![0.0_f64; portf_num];
    let mut group_counts = vec![0usize; portf_num];
    let mut held_signal_row_idx = effective_raw_indices[0] - 1;
    let mut held_restrict_row_idx = effective_raw_indices[0] - 1;
    let mut gen = vec![0u32; n_stocks];
    let mut stamp = vec![0u32; n_stocks];
    let mut walk_buf = Vec::<i64>::with_capacity(n_stocks);
    let mut gen_id: u32 = 0;

    for (local_t, &raw_eff_idx) in effective_raw_indices.iter().enumerate() {
        if local_t % gap == 0 {
            held_signal_row_idx = raw_eff_idx - 1;
            held_restrict_row_idx = raw_eff_idx - 1;
        }
        filtered_signal.clear();
        filtered_ret.clear();
        filtered_stock_idx.clear();
        // 候选过滤与生产完全相同 (全扫描 n_stocks, restrict[held] 判定);
        // 不再使用 (date, held_off) 预计算列表——held 随 effective 段边界变化,
        // 低覆盖率因子跳天会越过预计算窗口导致取错行 (详见 BtPrecomputed 注释)。
        for stock_idx in 0..n_stocks {
            let signal_value = factor[[held_signal_row_idx, stock_idx, slot_idx]];
            let ret_value = ret[[raw_eff_idx, stock_idx]];
            let is_open = restrict[[held_restrict_row_idx, stock_idx]].is_finite()
                && restrict[[held_restrict_row_idx, stock_idx]] == 0.0;
            if signal_value.is_finite() && ret_value.is_finite() && is_open {
                filtered_signal.push(signal_value);
                filtered_ret.push(ret_value);
                filtered_stock_idx.push(stock_idx as u32);
            }
        }
        if (local_t + 1) % gap == 0 {
            // 收益秩: walk 预排序全行 (gen 代标记避免跨日期串扰)
            gen_id += 1;
            let order = &pre.orders[g][raw_eff_idx];
            for (pos, &stk) in filtered_stock_idx.iter().enumerate() {
                gen[stk as usize] = gen_id;
                stamp[stk as usize] = (pos + 1) as u32;
            }
            walk_buf.clear();
            walk_buf.resize(filtered_stock_idx.len(), 0);
            let mut counter = 0usize;
            for &stk in order {
                if gen[stk as usize] == gen_id {
                    walk_buf[stamp[stk as usize] as usize - 1] = counter as i64;
                    counter += 1;
                }
            }
            let xx = ordinal_ranks_radix(&filtered_signal);
            finish_ic_from_parts(
                &mut ic_dates,
                &mut ic_values_f64,
                &mut ic_values_f32,
                &walk_buf,
                &xx,
                filtered_signal.len(),
                dates[raw_eff_idx],
            );
        }
        let stocks_num = filtered_signal.len();
        if stocks_num < portf_num {
            continue;
        }
        let valid_symbol_num = open_symbol_counts
            .get(raw_eff_idx - 1)
            .copied()
            .unwrap_or(0);
        if valid_symbol_num > 0 {
            ratio_values[local_t] = stocks_num as f64 / valid_symbol_num as f64;
        }
        if ic_only {
            continue;
        }
        group_sums.fill(0.0);
        group_counts.fill(0);
        let ranks = average_ranks_radix(&filtered_signal);
        for idx in 0..stocks_num {
            let pct = ranks[idx] / stocks_num as f64;
            let mut bucket = (pct * portf_num as f64).floor() as usize;
            if bucket >= portf_num {
                bucket = portf_num - 1;
            }
            group_sums[bucket] += filtered_ret[idx] as f64;
            group_counts[bucket] += 1;
        }
        for bucket in 0..portf_num {
            group_returns[bucket][local_t] = if group_counts[bucket] == 0 {
                0.0
            } else {
                group_sums[bucket] / group_counts[bucket] as f64
            };
        }
    }
    LegacyBacktestResult {
        summary: finish_summary(
            &ic_values_f64,
            &ratio_values,
            &group_returns,
            effective_raw_indices,
            index,
            gap,
            date_size,
            ic_only,
        ),
        ic_dates,
        ic_values: ic_values_f32,
    }
}

/// u32 key 稳定 LSD radix (4-pass); 初始 order 按 index 升序 → 与 (key,index) 总序一致。
fn radix_sort_u32_keys(keys: &[u32], order: &mut Vec<u32>, tmp: &mut Vec<u32>) {
    let n = order.len();
    if n < 2 {
        return;
    }
    tmp.clear();
    tmp.resize(n, 0);
    let mut count = [0usize; 256];
    for shift in (0..32).step_by(8) {
        count.fill(0);
        for &i in order.iter() {
            count[((keys[i as usize] >> shift) & 0xff) as usize] += 1;
        }
        let mut acc = 0usize;
        for c in count.iter_mut() {
            let t = *c;
            *c = acc;
            acc += t;
        }
        for &i in order.iter() {
            let b = ((keys[i as usize] >> shift) & 0xff) as usize;
            tmp[count[b]] = i;
            count[b] += 1;
        }
        std::mem::swap(order, tmp);
    }
}

/// 单次排序同时产出 ordinal 秩与平均秩 (4-pass u32; 与 ordinal_ranks_radix + average_ranks_radix
/// 的 (value,index) 总序逐位一致)。
fn rank_both_radix(values: &[f32]) -> (Vec<i64>, Vec<f64>) {
    let n = values.len();
    let mut keys = vec![0u32; n];
    for (i, &v) in values.iter().enumerate() {
        keys[i] = mono_key32(v);
    }
    let mut order: Vec<u32> = (0..n as u32).collect();
    let mut tmp: Vec<u32> = Vec::new();
    radix_sort_u32_keys(&keys, &mut order, &mut tmp);
    let mut ordinal = vec![0i64; n];
    let mut avg = vec![f64::NAN; n];
    for (rank, &i) in order.iter().enumerate() {
        ordinal[i as usize] = rank as i64;
    }
    let mut start = 0usize;
    while start < n {
        let v = values[order[start] as usize];
        let mut end = start + 1;
        while end < n && values[order[end] as usize] == v {
            end += 1;
        }
        let avg_rank = (start + 1 + end) as f64 / 2.0;
        for &i in order[start..end].iter() {
            avg[i as usize] = avg_rank;
        }
        start = end;
    }
    (ordinal, avg)
}

/// 4-pass 版 bt_pre 构建 (与 build_bt_precomputed 同序)。
fn build_bt_precomputed4(
    ret_sum_g1: &Array2<f32>,
    ret_sum_g5: &Array2<f32>,
    _ret_g1: &Array2<f32>,
    _ret_g5: &Array2<f32>,
    _restrict: &Array2<f32>,
) -> BtPrecomputed {
    let n_dates = ret_sum_g1.nrows();
    let n_stocks = ret_sum_g1.ncols();
    let mut orders: [Vec<Vec<u32>>; 2] = [Vec::new(), Vec::new()];
    for (g, ret_sum) in [(0usize, ret_sum_g1), (1usize, ret_sum_g5)] {
        let mut ords = Vec::with_capacity(n_dates);
        for d in 0..n_dates {
            let mut keys: Vec<u32> = Vec::with_capacity(n_stocks);
            for (j, &v) in ret_sum.row(d).iter().enumerate() {
                let k = if v.is_nan() { u32::MAX } else { mono_key32(v) };
                keys.push(k);
            }
            let mut order: Vec<u32> = (0..n_stocks as u32).collect();
            let mut tmp: Vec<u32> = Vec::new();
            radix_sort_u32_keys(&keys, &mut order, &mut tmp);
            ords.push(order);
        }
        orders[g] = ords;
    }
    BtPrecomputed { orders }
}

fn backtest_opt2(
    factor: &ArrayView3<'_, f32>,
    ret: &Array2<f32>,
    ret_sum: &Array2<f32>,
    restrict: &Array2<f32>,
    index: &Array1<f32>,
    dates: &[i32],
    slot_idx: usize,
    gap: usize,
    portf_num: usize,
    effective_raw_indices: &[usize],
    open_symbol_counts: &[usize],
    ic_only: bool,
    pre: &BtPrecomputed,
) -> LegacyBacktestResult {
    if effective_raw_indices.is_empty() {
        return default_legacy_backtest_result();
    }
    let n_stocks = factor.shape()[1];
    let g = if gap == 1 { 0 } else { 1 };
    let date_size = effective_raw_indices.len();
    let mut group_returns = if ic_only {
        Vec::new()
    } else {
        vec![vec![0.0_f64; date_size]; portf_num]
    };
    let mut ratio_values = vec![f64::NAN; date_size];
    let mut ic_dates = Vec::<i32>::new();
    let mut ic_values_f64 = Vec::<f64>::new();
    let mut ic_values_f32 = Vec::<f32>::new();
    let mut filtered_signal = Vec::<f32>::with_capacity(n_stocks);
    let mut filtered_ret = Vec::<f32>::with_capacity(n_stocks);
    let mut filtered_stock_idx = Vec::<u32>::with_capacity(n_stocks);
    let mut group_sums = vec![0.0_f64; portf_num];
    let mut group_counts = vec![0usize; portf_num];
    let mut held_signal_row_idx = effective_raw_indices[0] - 1;
    let mut held_restrict_row_idx = effective_raw_indices[0] - 1;
    let mut gen = vec![0u32; n_stocks];
    let mut stamp = vec![0u32; n_stocks];
    let mut walk_buf = Vec::<i64>::with_capacity(n_stocks);
    let mut gen_id: u32 = 0;

    for (local_t, &raw_eff_idx) in effective_raw_indices.iter().enumerate() {
        if local_t % gap == 0 {
            held_signal_row_idx = raw_eff_idx - 1;
            held_restrict_row_idx = raw_eff_idx - 1;
        }
        filtered_signal.clear();
        filtered_ret.clear();
        filtered_stock_idx.clear();
        // 候选过滤与生产完全相同 (全扫描 n_stocks, restrict[held] 判定);
        // 不再使用 (date, held_off) 预计算列表——held 随 effective 段边界变化,
        // 低覆盖率因子跳天会越过预计算窗口导致取错行 (详见 BtPrecomputed 注释)。
        for stock_idx in 0..n_stocks {
            let signal_value = factor[[held_signal_row_idx, stock_idx, slot_idx]];
            let ret_value = ret[[raw_eff_idx, stock_idx]];
            let is_open = restrict[[held_restrict_row_idx, stock_idx]].is_finite()
                && restrict[[held_restrict_row_idx, stock_idx]] == 0.0;
            if signal_value.is_finite() && ret_value.is_finite() && is_open {
                filtered_signal.push(signal_value);
                filtered_ret.push(ret_value);
                filtered_stock_idx.push(stock_idx as u32);
            }
        }
        if (local_t + 1) % gap == 0 {
            // 收益秩: walk 预排序全行 (gen 代标记避免跨日期串扰)
            gen_id += 1;
            let order = &pre.orders[g][raw_eff_idx];
            for (pos, &stk) in filtered_stock_idx.iter().enumerate() {
                gen[stk as usize] = gen_id;
                stamp[stk as usize] = (pos + 1) as u32;
            }
            walk_buf.clear();
            walk_buf.resize(filtered_stock_idx.len(), 0);
            let mut counter = 0usize;
            for &stk in order {
                if gen[stk as usize] == gen_id {
                    walk_buf[stamp[stk as usize] as usize - 1] = counter as i64;
                    counter += 1;
                }
            }
            let (xx, _avg0) = rank_both_radix(&filtered_signal);
            finish_ic_from_parts(
                &mut ic_dates,
                &mut ic_values_f64,
                &mut ic_values_f32,
                &walk_buf,
                &xx,
                filtered_signal.len(),
                dates[raw_eff_idx],
            );
        }
        let stocks_num = filtered_signal.len();
        if stocks_num < portf_num {
            continue;
        }
        let valid_symbol_num = open_symbol_counts
            .get(raw_eff_idx - 1)
            .copied()
            .unwrap_or(0);
        if valid_symbol_num > 0 {
            ratio_values[local_t] = stocks_num as f64 / valid_symbol_num as f64;
        }
        if ic_only {
            continue;
        }
        group_sums.fill(0.0);
        group_counts.fill(0);
        let (_ord1, ranks) = rank_both_radix(&filtered_signal);
        for idx in 0..stocks_num {
            let pct = ranks[idx] / stocks_num as f64;
            let mut bucket = (pct * portf_num as f64).floor() as usize;
            if bucket >= portf_num {
                bucket = portf_num - 1;
            }
            group_sums[bucket] += filtered_ret[idx] as f64;
            group_counts[bucket] += 1;
        }
        for bucket in 0..portf_num {
            group_returns[bucket][local_t] = if group_counts[bucket] == 0 {
                0.0
            } else {
                group_sums[bucket] / group_counts[bucket] as f64
            };
        }
    }
    LegacyBacktestResult {
        summary: finish_summary(
            &ic_values_f64,
            &ratio_values,
            &group_returns,
            effective_raw_indices,
            index,
            gap,
            date_size,
            ic_only,
        ),
        ic_dates,
        ic_values: ic_values_f32,
    }
}


// ---------------- F. neutralize (factor_neutralize_std.rs 原样 + O2) ----------------
fn cmp_f64(a: &f64, b: &f64) -> std::cmp::Ordering {
    match (a.is_nan(), b.is_nan()) {
        (true, true) => std::cmp::Ordering::Equal,
        (true, false) => std::cmp::Ordering::Greater,
        (false, true) => std::cmp::Ordering::Less,
        (false, false) => a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal),
    }
}

fn mono_key(v: f64) -> u64 {
    let bits = v.to_bits();
    if bits >> 63 == 0 {
        bits ^ 0x8000_0000_0000_0000
    } else {
        !bits
    }
}

fn radix_sort_order(keys: &[u64], order: &mut Vec<usize>, tmp: &mut Vec<usize>) {
    let n = order.len();
    if n < 2 {
        return;
    }
    tmp.clear();
    tmp.resize(n, 0);
    let mut count = [0usize; 256];
    for shift in (0..64).step_by(8) {
        count.fill(0);
        for &i in order.iter() {
            count[((keys[i] >> shift) & 0xff) as usize] += 1;
        }
        let mut acc = 0usize;
        for c in count.iter_mut() {
            let t = *c;
            *c = acc;
            acc += t;
        }
        for &i in order.iter() {
            let b = ((keys[i] >> shift) & 0xff) as usize;
            tmp[count[b]] = i;
            count[b] += 1;
        }
        std::mem::swap(order, tmp);
    }
}

fn ols2(ys: &[f64], bs: &[f64]) -> (f64, f64) {
    let n = ys.len() as f64;
    let mut s1 = 0.0;
    let mut s2 = 0.0;
    let mut sy = 0.0;
    let mut sby = 0.0;
    for i in 0..ys.len() {
        let y = ys[i];
        let b = bs[i];
        s1 += b;
        s2 += b * b;
        sy += y;
        sby += b * y;
    }
    let det = n * s2 - s1 * s1;
    let c0 = (s2 * sy - s1 * sby) / det;
    let c1 = (n * sby - s1 * sy) / det;
    (c0, c1)
}

fn rank_pct_row_into(
    vals: &[f64],
    ranks: &mut Vec<f64>,
    idxs: &mut Vec<usize>,
    tmp: &mut Vec<usize>,
    keys: &mut Vec<u64>,
) {
    let n = vals.len();
    ranks.clear();
    ranks.resize(n, f64::NAN);
    idxs.clear();
    keys.clear();
    keys.resize(n, 0);
    for (i, &v) in vals.iter().enumerate() {
        if !v.is_nan() {
            idxs.push(i);
            keys[i] = mono_key(v);
        }
    }
    let n_valid = idxs.len();
    if n_valid == 0 {
        return;
    }
    radix_sort_order(keys, idxs, tmp);
    let mut i = 0;
    while i < n_valid {
        let mut j = i;
        while j + 1 < n_valid && vals[idxs[j + 1]] == vals[idxs[i]] {
            j += 1;
        }
        let avg_rank = ((i + 1) + (j + 1)) as f64 / 2.0;
        let pct = avg_rank / n_valid as f64;
        for item in &idxs[i..=j] {
            ranks[*item] = pct;
        }
        i = j + 1;
    }
}

fn rank_pct_all(values: &mut Array2<f64>) {
    let mut ranks: Vec<f64> = Vec::with_capacity(values.ncols());
    let mut idxs: Vec<usize> = Vec::with_capacity(values.ncols());
    let mut tmp: Vec<usize> = Vec::with_capacity(values.ncols());
    let mut keys: Vec<u64> = Vec::with_capacity(values.ncols());
    for mut row in values.rows_mut() {
        rank_pct_row_into(row.as_slice().unwrap(), &mut ranks, &mut idxs, &mut tmp, &mut keys);
        for (j, &r) in ranks.iter().enumerate() {
            row[j] = r;
        }
    }
}

fn quickselect(v: &mut [f64], k: usize) -> f64 {
    let mut lo = 0usize;
    let mut hi = v.len();
    while lo + 1 < hi {
        let mid = lo + (hi - lo) / 2;
        let a = v[lo];
        let b = v[mid];
        let c = v[hi - 1];
        let pivot = if (a <= b && b <= c) || (c <= b && b <= a) {
            b
        } else if (b <= a && a <= c) || (c <= a && a <= b) {
            a
        } else {
            c
        };
        let pi = if v[lo] == pivot {
            lo
        } else if v[mid] == pivot {
            mid
        } else {
            hi - 1
        };
        v.swap(pi, hi - 1);
        let mut store = lo;
        for i in lo..hi - 1 {
            if v[i] < pivot {
                v.swap(i, store);
                store += 1;
            }
        }
        v.swap(store, hi - 1);
        if k < store {
            hi = store;
        } else if k > store {
            lo = store + 1;
        } else {
            return v[store];
        }
    }
    v[lo]
}

fn median_inplace(v: &mut [f64]) -> f64 {
    let n = v.len();
    if n == 0 {
        return f64::NAN;
    }
    if n % 2 == 1 {
        quickselect(v, n / 2)
    } else {
        let a = quickselect(v, n / 2 - 1);
        let b = quickselect(v, n / 2);
        (a + b) / 2.0
    }
}

fn has_ge_n_unique(vals: &[f64], need: usize) -> bool {
    let mut seen: Vec<f64> = Vec::with_capacity(need + 1);
    let mut nan_seen = false;
    for &v in vals {
        if v.is_nan() {
            nan_seen = true;
        } else if !seen.contains(&v) {
            seen.push(v);
        }
        if seen.len() + (nan_seen as usize) >= need {
            return true;
        }
    }
    false
}

fn fill_ind_reg(fv: &mut Array2<f64>, base_list: &[Array2<f64>], ind3: &Array2<f64>) {
    let (t, n) = fv.dim();
    let n_base = base_list.len();
    debug_assert_eq!(n_base, 1, "行业填充 base 只支持单个因子 (生产为 size)");
    let ind2 = ind3.map(|&v| (v / 100.0).floor());
    let ind1 = ind3.map(|&v| (v / 10000.0).floor());
    let ind0 = ind1.map(|&v| if v.is_nan() { 0.0 } else { 1.0 });
    let mut order: Vec<usize> = Vec::with_capacity(n);
    let mut order_tmp: Vec<usize> = Vec::with_capacity(n);
    let mut order_keys: Vec<u64> = Vec::with_capacity(n);
    let mut cols: Vec<usize> = Vec::with_capacity(n);
    let mut f_masked: Vec<f64> = Vec::with_capacity(n);
    let mut b_masked: Vec<f64> = Vec::with_capacity(n);
    let mut not_nan: Vec<bool> = Vec::with_capacity(n);
    let mut ys: Vec<f64> = Vec::with_capacity(n);
    let mut bs: Vec<f64> = Vec::with_capacity(n);
    for ind_level in [&ind2, &ind1, &ind0] {
        for idx in 0..t {
            let mut fv_row: Vec<f64> = fv.row(idx).iter().copied().collect();
            if !has_ge_n_unique(&fv_row, 10) {
                continue;
            }
            let ind_row = ind_level.row(idx);
            let base_row = base_list[0].row(idx);
            order.clear();
            order_keys.clear();
            for j in 0..n {
                order.push(j);
                order_keys.push(mono_key(ind_row[j]));
            }
            radix_sort_order(&order_keys, &mut order, &mut order_tmp);
            let mut seg_start = 0usize;
            while seg_start < n {
                let code = ind_row[order[seg_start]];
                if code.is_nan() {
                    break;
                }
                let mut seg_end = seg_start + 1;
                while seg_end < n && ind_row[order[seg_end]] == code {
                    seg_end += 1;
                }
                let m_count = seg_end - seg_start;
                cols.clear();
                f_masked.clear();
                b_masked.clear();
                not_nan.clear();
                for &ci in &order[seg_start..seg_end] {
                    cols.push(ci);
                    f_masked.push(fv_row[ci]);
                    let bv = base_row[ci];
                    b_masked.push(bv);
                    not_nan.push(!f_masked.last().unwrap().is_nan() && !bv.is_nan());
                }
                let n_obs = not_nan.iter().filter(|&&b| b).count();
                if n_obs >= 10 {
                    ys.clear();
                    bs.clear();
                    for i in 0..m_count {
                        if not_nan[i] {
                            ys.push(f_masked[i]);
                            bs.push(b_masked[i]);
                        }
                    }
                    let (c0, c1) = ols2(&ys, &bs);
                    for i in 0..m_count {
                        if !not_nan[i] {
                            f_masked[i] = c0 + c1 * b_masked[i];
                        }
                    }
                    for (mi, &ci) in cols.iter().enumerate() {
                        fv_row[ci] = f_masked[mi];
                    }
                }
                seg_start = seg_end;
            }
            for j in 0..n {
                fv[[idx, j]] = fv_row[j];
            }
        }
    }
}

fn fill_by_group_median_inplace(
    values: &mut Array2<f64>,
    codes: &Array2<f64>,
    valid_mask: Option<&Array2<f64>>,
) {
    let (t, n) = values.dim();
    let mut order: Vec<usize> = Vec::with_capacity(n);
    let mut order_tmp: Vec<usize> = Vec::with_capacity(n);
    let mut order_keys: Vec<u64> = Vec::with_capacity(n);
    let mut sv: Vec<f64> = Vec::with_capacity(n);
    for idx in 0..t {
        let mut row = values.row(idx).to_vec();
        let mut nan_mask: Vec<bool> = Vec::with_capacity(n);
        let mut has_nan = false;
        for (j, &v) in row.iter().enumerate() {
            let nn = v.is_nan();
            nan_mask.push(nn);
            if nn {
                has_nan = true;
            }
        }
        if !has_nan {
            continue;
        }
        let codes_row = codes.row(idx);
        order.clear();
        order_keys.clear();
        for j in 0..n {
            order.push(j);
            order_keys.push(mono_key(codes_row[j]));
        }
        radix_sort_order(&order_keys, &mut order, &mut order_tmp);
        let mut seg_start = 0usize;
        while seg_start < n {
            let code = codes_row[order[seg_start]];
            if code.is_nan() {
                break;
            }
            let mut seg_end = seg_start + 1;
            while seg_end < n && codes_row[order[seg_end]] == code {
                seg_end += 1;
            }
            sv.clear();
            for &ci in &order[seg_start..seg_end] {
                let valid = valid_mask.map_or(true, |vm| vm[[idx, ci]] == 1.0);
                if valid && !row[ci].is_nan() {
                    sv.push(row[ci]);
                }
            }
            if !sv.is_empty() {
                let med = median_inplace(&mut sv);
                for &ci in &order[seg_start..seg_end] {
                    let valid = valid_mask.map_or(true, |vm| vm[[idx, ci]] == 1.0);
                    if nan_mask[ci] && valid {
                        row[ci] = med;
                    }
                }
            }
            seg_start = seg_end;
        }
        for j in 0..n {
            values[[idx, j]] = row[j];
        }
    }
}

fn get_residual(
    fv: &Array2<f64>,
    bench: &[Array2<f64>],
    industry: Option<&Array2<f64>>,
) -> Array2<f64> {
    let (t, n) = fv.dim();
    let mut resid = Array2::<f64>::from_elem((t, n), f64::NAN);
    let k = bench.len();
    let mut valid: Vec<bool> = Vec::with_capacity(n);
    let mut rows: Vec<[f64; 11]> = Vec::with_capacity(n);
    let mut cur = [0.0_f64; 11];
    let mut ind_cols: Vec<i32> = Vec::with_capacity(n);
    let mut ind_codes: Vec<f64> = Vec::with_capacity(40);
    let mut xtx: Vec<f64> = Vec::with_capacity(42 * 42);
    let mut xty: Vec<f64> = Vec::with_capacity(42);
    for idx in 0..t {
        let row = fv.row(idx);
        if !row.iter().any(|v| !v.is_nan()) {
            continue;
        }
        if let Some(ind) = industry {
            ind_codes.clear();
            for j in 0..n {
                let c = ind[[idx, j]];
                if !c.is_nan() && !ind_codes.contains(&c) {
                    ind_codes.push(c);
                }
            }
            ind_codes.sort_by(cmp_f64);
        }
        let n_ind = if industry.is_some() { ind_codes.len() } else { 0 };
        let p = if industry.is_some() { k + n_ind } else { k + 1 };
        valid.clear();
        rows.clear();
        ind_cols.clear();
        for j in 0..n {
            let ok = row[j].is_finite() && bench.iter().all(|b| b[[idx, j]].is_finite());
            valid.push(ok);
            if ok {
                cur[0] = row[j];
                for c in 0..k {
                    cur[c + 1] = bench[c][[idx, j]];
                }
                rows.push(cur);
                if let Some(ind) = industry {
                    let c = ind[[idx, j]];
                    let col = if c.is_nan() {
                        -1
                    } else {
                        match ind_codes.binary_search_by(|x| cmp_f64(x, &c)) {
                            Ok(pos) => pos as i32,
                            Err(_) => -1,
                        }
                    };
                    ind_cols.push(col);
                } else {
                    ind_cols.push(-1);
                }
            }
        }
        let n_valid = rows.len();
        if n_valid <= 10 {
            continue;
        }
        let mut resid_row = vec![f64::NAN; n];
        let mut uniq: Vec<f64> = rows.iter().map(|r| r[0]).collect();
        uniq.sort_by(cmp_f64);
        uniq.dedup_by(|a, b| (a.is_nan() && b.is_nan()) || a == b);
        if uniq.len() == 1 {
            for j in 0..n {
                if valid[j] {
                    resid_row[j] = 0.5;
                }
            }
        } else {
            xtx.clear();
            xtx.resize(p * p, 0.0);
            xty.clear();
            xty.resize(p, 0.0);
            for (i, r) in rows.iter().enumerate() {
                let yv = r[0];
                if industry.is_none() {
                    xty[0] += yv;
                    xtx[0] += 1.0;
                    for c in 0..k {
                        let b = r[c + 1];
                        xty[c + 1] += b * yv;
                        xtx[c + 1] += b;
                        xtx[(c + 1) * p] += b;
                    }
                    for c1 in 0..k {
                        let b1 = r[c1 + 1];
                        xtx[(c1 + 1) * p + (c1 + 1)] += b1 * b1;
                        for c2 in (c1 + 1)..k {
                            let v = b1 * r[c2 + 1];
                            xtx[(c1 + 1) * p + (c2 + 1)] += v;
                            xtx[(c2 + 1) * p + (c1 + 1)] += v;
                        }
                    }
                } else {
                    for c in 0..k {
                        let b = r[c + 1];
                        xty[c] += b * yv;
                        xtx[c * p + c] += b * b;
                        for c2 in (c + 1)..k {
                            let v = b * r[c2 + 1];
                            xtx[c * p + c2] += v;
                            xtx[c2 * p + c] += v;
                        }
                    }
                    let ic = ind_cols[i];
                    if ic >= 0 {
                        let col = (k as i32 + ic) as usize;
                        xty[col] += yv;
                        xtx[col * p + col] += 1.0;
                        for c in 0..k {
                            let b = r[c + 1];
                            xtx[c * p + col] += b;
                            xtx[col * p + c] += b;
                        }
                    }
                }
            }
            let m = DMatrix::from_row_slice(p, p, &xtx);
            let rhs = DMatrix::from_column_slice(p, 1, &xty);
            let use_svd = n_valid <= 40 || Cholesky::new(m.clone()).is_none();
            if use_svd && idx < 5 {
                eprintln!("[diag] date {idx} use SVD: n_valid={n_valid} p={p}");
            }
            let coef: Vec<f64> = if !use_svd {
                let chol = Cholesky::new(m).expect("chol");
                chol.solve(&rhs).column(0).iter().copied().collect()
            } else {
                // SVD 伪逆 (生产路径)
                let n_r = rows.len();
                let xm = DMatrix::from_fn(n_r, p, |r_i, c| {
                    if industry.is_none() {
                        if c == 0 {
                            1.0
                        } else {
                            rows[r_i][c]
                        }
                    } else {
                        let ic = ind_cols[r_i];
                        if c < k {
                            rows[r_i][c + 1]
                        } else if ic >= 0 && c == k as usize + ic as usize {
                            1.0
                        } else {
                            0.0
                        }
                    }
                });
                let svd = xm.clone().svd(true, true);
                let u = svd.u.expect("svd u");
                let vt = svd.v_t.expect("svd vt");
                let sv = svd.singular_values;
                let s_max = sv.iter().cloned().fold(0.0_f64, f64::max);
                let rcond = s_max * (n_r.max(p) as f64) * 2.22e-16;
                let ym = DMatrix::from_fn(n_r, 1, |r_i, _| rows[r_i][0]);
                let uty = u.transpose() * ym;
                let mut coef = DMatrix::zeros(p, 1);
                for i in 0..p {
                    if sv[i] > rcond {
                        coef[(i, 0)] = uty[(i, 0)] / sv[i];
                    }
                }
                (vt.transpose() * coef).column(0).iter().copied().collect()
            };
            let mut vi = 0;
            for (i, r) in rows.iter().enumerate() {
                let mut pred = 0.0;
                if industry.is_none() {
                    pred = coef[0];
                    for c in 0..k {
                        pred += coef[c + 1] * r[c + 1];
                    }
                } else {
                    for c in 0..k {
                        pred += coef[c] * r[c + 1];
                    }
                    let ic = ind_cols[i];
                    if ic >= 0 {
                        pred += coef[k + ic as usize];
                    }
                }
                while !valid[vi] {
                    vi += 1;
                }
                resid_row[vi] = r[0] - pred;
                vi += 1;
            }
        }
        for j in 0..n {
            resid[[idx, j]] = resid_row[j];
        }
    }
    resid
}

fn neutralize_std_section_owned(
    mut fv_ranked: Array2<f64>,
    ind: &Array2<f64>,
    restrict: &Array2<f32>,
    barra_ranked: &[Array2<f64>],
    industry_neutralize: bool,
) -> Array2<f64> {
    let (t, n) = fv_ranked.dim();
    let size_ranked = &barra_ranked[2];
    rank_pct_all(&mut fv_ranked);
    fill_ind_reg(&mut fv_ranked, std::slice::from_ref(size_ranked), ind);
    let ind1 = ind.map(|&v| (v / 10000.0).floor());
    let ind2 = ind.map(|&v| (v / 100.0).floor());
    for i in 0..(t * n) {
        if ind1.as_slice().unwrap()[i].is_nan() {
            fv_ranked.as_slice_mut().unwrap()[i] = f64::NAN;
        }
    }
    let mut fv_filled = fv_ranked.clone();
    fill_by_group_median_inplace(&mut fv_filled, &ind2, None);
    fill_by_group_median_inplace(&mut fv_filled, &ind1, None);
    let zeros = Array2::<f64>::zeros((t, n));
    let ind1_mask = ind1.map(|&v| if v.is_nan() { 0.0 } else { 1.0 });
    fill_by_group_median_inplace(&mut fv_filled, &zeros, Some(&ind1_mask));
    drop(fv_ranked);
    for i in 0..(t * n) {
        if restrict.as_slice().unwrap()[i] != 0.0 {
            fv_filled.as_slice_mut().unwrap()[i] = f64::NAN;
        }
    }
    rank_pct_all(&mut fv_filled);
    // 注意: 生产 get_residual 行业列用 ind1 (一级行业分组), 不是原始行业码
    let resid = if industry_neutralize {
        get_residual(&fv_filled, barra_ranked, Some(&ind1))
    } else {
        get_residual(&fv_filled, barra_ranked, None)
    };
    let mut resid_rank = resid;
    rank_pct_all(&mut resid_rank);
    resid_rank
}

// ---- O2 neutralize 优化 ----
struct NeuPrecomputed {
    /// 5 个 (T,N) 排序: 0=ind2 1=ind1 2=ind0 (fill_ind_reg), 3=ind2 4=ind1 (中位填充)
    orders: Vec<Array2<usize>>,
    /// 逐日: (p, 有效股票索引, 每股行业列号, X'X) —— 行业 one-hot 列 = 10+ic
    per_date: Vec<(usize, Vec<u32>, Vec<i32>, Vec<f64>)>,
    pub ind1: Array2<f64>,
    pub ind2: Array2<f64>,
    pub ind1_mask: Array2<f64>,
    pub zeros: Array2<f64>,
}

fn precompute_neu(
    industry: &Array2<f64>,
    restrict: &Array2<f32>,
    barra_ranked: &[Array2<f64>],
) -> NeuPrecomputed {
    let (t, n) = industry.dim();
    let ind1 = industry.map(|&v| (v / 10000.0).floor());
    let ind2 = industry.map(|&v| (v / 100.0).floor());
    let ind0 = ind1.map(|&v| if v.is_nan() { 0.0 } else { 1.0 });
    let levels = [&ind2, &ind1, &ind0, &ind2, &ind1];
    let mut orders = Vec::new();
    for codes in levels {
        let mut ord = Array2::<usize>::zeros((t, n));
        for idx in 0..t {
            let mut order: Vec<usize> = (0..n).collect();
            let mut keys: Vec<u64> = codes.row(idx).iter().map(|&v| mono_key(v)).collect();
            let mut tmp: Vec<usize> = Vec::new();
            radix_sort_order(&keys, &mut order, &mut tmp);
            for j in 0..n {
                ord[[idx, j]] = order[j];
            }
        }
        orders.push(ord);
    }
    let ind1_mask = ind1.map(|&v| if v.is_nan() { 0.0 } else { 1.0 });
    let zeros = Array2::<f64>::zeros((t, n));
    let mut per_date = Vec::with_capacity(t);
    for idx in 0..t {
        let mut ind_codes: Vec<f64> = Vec::new();
        for j in 0..n {
            let c = ind1[[idx, j]];
            if !c.is_nan() && !ind_codes.contains(&c) {
                ind_codes.push(c);
            }
        }
        ind_codes.sort_by(cmp_f64);
        let n_ind = ind_codes.len();
        let p = 10 + n_ind;
        let mut valid_idx: Vec<u32> = Vec::new();
        let mut valid_cols: Vec<i32> = Vec::new();
        let mut valid_nan_y = false;
        for j in 0..n {
            let ok = restrict[[idx, j]].is_finite()
                && restrict[[idx, j]] == 0.0
                && barra_ranked.iter().all(|b| b[[idx, j]].is_finite());
            if ok {
                valid_idx.push(j as u32);
                let c_ind = ind1[[idx, j]];
                let col = if c_ind.is_nan() {
                    -1
                } else {
                    match ind_codes.binary_search_by(|x| cmp_f64(x, &c_ind)) {
                        Ok(pos) => pos as i32,
                        Err(_) => -1,
                    }
                };
                valid_cols.push(col);
            }
        }
        let n_valid = valid_idx.len();
        if n_valid <= 10 {
            per_date.push((0, valid_idx, valid_cols, Vec::new()));
            continue;
        }
        let mut xtx = vec![0.0f64; p * p];
        for (pos, &j) in valid_idx.iter().enumerate() {
            let ji = j as usize;
            let k = 10usize;
            for c in 0..k {
                let b = barra_ranked[c][[idx, ji]];
                xtx[c * p + c] += b * b;
                for c2 in (c + 1)..k {
                    let v = b * barra_ranked[c2][[idx, ji]];
                    xtx[c * p + c2] += v;
                    xtx[c2 * p + c] += v;
                }
            }
            let ic = valid_cols[pos];
            if ic >= 0 {
                let col = 10 + ic as usize;
                xtx[col * p + col] += 1.0;
                for c in 0..k {
                    let b = barra_ranked[c][[idx, ji]];
                    xtx[c * p + col] += b;
                    xtx[col * p + c] += b;
                }
            }
        }
        let _ = valid_nan_y;
        per_date.push((p, valid_idx, valid_cols, xtx));
    }
    NeuPrecomputed { orders, per_date, ind1, ind2, ind1_mask, zeros }
}

fn fill_ind_reg_pre(
    fv: &mut Array2<f64>,
    ind2: &Array2<f64>,
    ind1: &Array2<f64>,
    size_ranked: &Array2<f64>,
    orders: &[Array2<usize>],
) -> (u64, u64) {
    // 返回 (跳过天数, 处理天数) 供诊断
    let (t, n) = fv.dim();
    let ind0 = ind1.map(|&v| if v.is_nan() { 0.0 } else { 1.0 });
    let levels = [ind2, ind1, &ind0];
    let mut ys: Vec<f64> = Vec::with_capacity(n);
    let mut bs: Vec<f64> = Vec::with_capacity(n);
    let mut skip = 0u64;
    let mut done = 0u64;
    for (li, level) in levels.iter().enumerate() {
        for idx in 0..t {
            let mut row = fv.row(idx).to_vec();
            if !has_ge_n_unique(&row, 10) {
                skip += 1;
                continue;
            }
            done += 1;
            let order_arr = orders[li].row(idx);
            let order = order_arr.as_slice().unwrap();
            let mut seg_start = 0usize;
            while seg_start < n {
                let code = level[[idx, order[seg_start]]];
                if code.is_nan() {
                    break;
                }
                let mut seg_end = seg_start + 1;
                while seg_end < n && level[[idx, order[seg_end]]] == code {
                    seg_end += 1;
                }
                ys.clear();
                bs.clear();
                let mut obs: Vec<bool> = Vec::with_capacity(seg_end - seg_start);
                for &ci in &order[seg_start..seg_end] {
                    let ok = !row[ci].is_nan() && !size_ranked[[idx, ci]].is_nan();
                    obs.push(ok);
                    if ok {
                        ys.push(row[ci]);
                        bs.push(size_ranked[[idx, ci]]);
                    }
                }
                if ys.len() >= 10 {
                    let (c0, c1) = ols2(&ys, &bs);
                    for (mi, &ci) in order[seg_start..seg_end].iter().enumerate() {
                        if !obs[mi] {
                            row[ci] = c0 + c1 * size_ranked[[idx, ci]];
                        }
                    }
                }
                seg_start = seg_end;
            }
            for j in 0..n {
                fv[[idx, j]] = row[j];
            }
        }
    }
    (skip, done)
}

fn group_median_fill_pre(
    values: &mut Array2<f64>,
    codes: &Array2<f64>,
    valid_mask: Option<&Array2<f64>>,
    orders: &Array2<usize>,
) -> u64 {
    // 返回跳过天数诊断
    let (t, n) = values.dim();
    let mut sv: Vec<f64> = Vec::with_capacity(n);
    let mut skipped = 0u64;
    for idx in 0..t {
        let mut row = values.row(idx).to_vec();
        let mut nan_mask: Vec<bool> = Vec::with_capacity(n);
        let mut has_nan = false;
        for j in 0..n {
            let nn = row[j].is_nan();
            nan_mask.push(nn);
            if nn {
                has_nan = true;
            }
        }
        if !has_nan {
            skipped += 1;
            continue;
        }
        let order_arr = orders.row(idx);
        let order = order_arr.as_slice().unwrap();
        let mut seg_start = 0usize;
        while seg_start < n {
            let code = codes[[idx, order[seg_start]]];
            if code.is_nan() {
                break;
            }
            let mut seg_end = seg_start + 1;
            while seg_end < n && codes[[idx, order[seg_end]]] == code {
                seg_end += 1;
            }
            sv.clear();
            for &ci in &order[seg_start..seg_end] {
                let valid = valid_mask.map_or(true, |vm| vm[[idx, ci]] == 1.0);
                if valid && !row[ci].is_nan() {
                    sv.push(row[ci]);
                }
            }
            if !sv.is_empty() {
                let med = median_inplace(&mut sv);
                for &ci in &order[seg_start..seg_end] {
                    let valid = valid_mask.map_or(true, |vm| vm[[idx, ci]] == 1.0);
                    if nan_mask[ci] && valid {
                        row[ci] = med;
                    }
                }
            }
            seg_start = seg_end;
        }
        for j in 0..n {
            values[[idx, j]] = row[j];
        }
    }
    skipped
}

/// O2 中性化: 预计算 orders + 逐日 X'X; per-day 有效集含因子 NaN 时回退生产路径。
fn neutralize_v2_core(
    factor: &Array2<f64>,
    pre: &NeuPrecomputed,
    restrict: &Array2<f32>,
    barra_ranked: &[Array2<f64>],
) -> (Array2<f64>, u64) {
    let (t, n) = factor.dim();
    let mut fv_ranked = factor.clone();
    rank_pct_all(&mut fv_ranked);
    let (_, _) = fill_ind_reg_pre(
        &mut fv_ranked,
        &pre.ind2,
        &pre.ind1,
        &barra_ranked[2],
        &pre.orders[0..3],
    );
    for i in 0..(t * n) {
        if pre.ind1.as_slice().unwrap()[i].is_nan() {
            fv_ranked.as_slice_mut().unwrap()[i] = f64::NAN;
        }
    }
    let mut fv_filled = fv_ranked.clone();
    let _ = group_median_fill_pre(&mut fv_filled, &pre.ind2, None, &pre.orders[3]);
    let _ = group_median_fill_pre(&mut fv_filled, &pre.ind1, None, &pre.orders[4]);
    let _ = group_median_fill_pre(&mut fv_filled, &pre.zeros, Some(&pre.ind1_mask), &pre.orders[0]);
    drop(fv_ranked);
    for i in 0..(t * n) {
        if restrict.as_slice().unwrap()[i] != 0.0 {
            fv_filled.as_slice_mut().unwrap()[i] = f64::NAN;
        }
    }
    rank_pct_all(&mut fv_filled);
    // OLS: 用预计算 X'X; 若某日 valid 集合中存在 y NaN -> 回退该日为生产路径
    let mut fallback_days = 0u64;
    let resid = get_residual_v2(&fv_filled, barra_ranked, &pre.ind1, &pre.per_date, &mut fallback_days);
    let mut resid_rank = resid;
    rank_pct_all(&mut resid_rank);
    let _ = fallback_days;
    (resid_rank, fallback_days)
}

fn get_residual_v2(
    fv_filled: &Array2<f64>,
    barra_ranked: &[Array2<f64>],
    ind1: &Array2<f64>,
    per_date: &[(usize, Vec<u32>, Vec<i32>, Vec<f64>)],
    fallback_days: &mut u64,
) -> Array2<f64> {
    let (t, n) = fv_filled.dim();
    let mut resid = Array2::<f64>::from_elem((t, n), f64::NAN);
    let k = 10usize;
    for idx in 0..t {
        let (p, valid_idx, valid_cols, xtx_pre) = &per_date[idx];
        if *p == 0 {
            continue;
        }
        // y unique 检查 + y NaN 检查
        let mut uniq: Vec<f64> = Vec::with_capacity(valid_idx.len());
        let mut any_nan = false;
        for &j in valid_idx {
            let y = fv_filled[[idx, j as usize]];
            if !y.is_finite() {
                any_nan = true;
                break;
            }
            uniq.push(y);
        }
        if any_nan {
            *fallback_days += 1;
            // 回退: 生产 get_residual 单日路径
            let mut fv1 = Array2::<f64>::from_elem((1, n), f64::NAN);
            for j in 0..n {
                fv1[[0, j]] = fv_filled[[idx, j]];
            }
            let mut bench1: Vec<Array2<f64>> = Vec::new();
            for b in barra_ranked {
                let mut b1 = Array2::<f64>::from_elem((1, n), f64::NAN);
                for j in 0..n {
                    b1[[0, j]] = b[[idx, j]];
                }
                bench1.push(b1);
            }
            let mut ind1_1 = Array2::<f64>::from_elem((1, n), f64::NAN);
            for j in 0..n {
                ind1_1[[0, j]] = ind1[[idx, j]];
            }
            let r = get_residual(&fv1, &bench1, Some(&ind1_1));
            for j in 0..n {
                resid[[idx, j]] = r[[0, j]];
            }
            continue;
        }
        uniq.sort_by(cmp_f64);
        uniq.dedup_by(|a, b| (a.is_nan() && b.is_nan()) || a == b);
        if uniq.len() == 1 {
            for &j in valid_idx {
                resid[[idx, j as usize]] = 0.5;
            }
            continue;
        }
        let mut xty = vec![0.0f64; *p];
        for (pos, &j) in valid_idx.iter().enumerate() {
            let ji = j as usize;
            let yv = fv_filled[[idx, ji]];
            for c in 0..k {
                xty[c] += barra_ranked[c][[idx, ji]] * yv;
            }
            let ic = valid_cols[pos];
            if ic >= 0 {
                xty[k + ic as usize] += yv;
            }
        }
        // 生产: X'X 已预计算 (bitwise 相同累积); 直接 Cholesky 求解
        let m = DMatrix::from_row_slice(*p, *p, xtx_pre);
        let rhs = DMatrix::from_column_slice(*p, 1, &xty);
        let use_svd = valid_idx.len() <= 40 || Cholesky::new(m.clone()).is_none();
        let coef: Vec<f64> = if !use_svd {
            let chol = Cholesky::new(m).expect("chol");
            chol.solve(&rhs).column(0).iter().copied().collect()
        } else {
            // 回退生产 SVD
            let n_r = valid_idx.len();
            let rows: Vec<[f64; 11]> = valid_idx
                .iter()
                .map(|&j| {
                    let ji = j as usize;
                    let mut cur = [0.0_f64; 11];
                    cur[0] = fv_filled[[idx, ji]];
                    for c in 0..k {
                        cur[c + 1] = barra_ranked[c][[idx, ji]];
                    }
                    cur
                })
                .collect();
            let xm = DMatrix::from_fn(n_r, *p, |r_i, c| {
                let ic = valid_cols[r_i];
                if c < k {
                    rows[r_i][c + 1]
                } else if ic >= 0 && c == k + ic as usize {
                    1.0
                } else {
                    0.0
                }
            });
            let svd = xm.clone().svd(true, true);
            let u = svd.u.expect("svd u");
            let vt = svd.v_t.expect("svd vt");
            let sv = svd.singular_values;
            let s_max = sv.iter().cloned().fold(0.0_f64, f64::max);
            let rcond = s_max * (n_r.max(*p) as f64) * 2.22e-16;
            let ym = DMatrix::from_fn(n_r, 1, |r_i, _| rows[r_i][0]);
            let uty = u.transpose() * ym;
            let mut coef = DMatrix::zeros(*p, 1);
            for i in 0..*p {
                if sv[i] > rcond {
                    coef[(i, 0)] = uty[(i, 0)] / sv[i];
                }
            }
            (vt.transpose() * coef).column(0).iter().copied().collect()
        };
        // resid
        let mut vi = 0usize;
        for (pos, &j) in valid_idx.iter().enumerate() {
            let ji = j as usize;
            while vi < ji {
                vi += 1;
            }
            let yv = fv_filled[[idx, ji]];
            let mut pred = 0.0;
            for c in 0..k {
                pred += coef[c] * barra_ranked[c][[idx, ji]];
            }
            let ic = valid_cols[pos];
            if ic >= 0 {
                pred += coef[k + ic as usize];
            }
            resid[[idx, ji]] = yv - pred;
            let _ = vi;
        }
    }
    resid
}

// ======================= pyfunctions =======================
fn to2_f32(a: &PyReadonlyArray2<f32>) -> Array2<f32> {
    Array2::from_shape_vec(a.as_array().dim(), a.as_array().iter().copied().collect()).unwrap()
}

#[pyfunction]
fn bench_rank_py<'py>(py: Python<'py>, raw: PyReadonlyArray2<'py, f32>) -> PyResult<(f64, Py<PyArray2<f32>>)> {
    let data = to2_f32(&raw);
    let (ms, out) = py.allow_threads(move || {
        let t0 = std::time::Instant::now();
        let out = rank_axis1_average_f32_serial(&data);
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        (ms, out)
    });
    Ok((ms, out.into_pyarray(py).to_owned()))
}

#[pyfunction]
fn bench_rolling_py<'py>(
    py: Python<'py>,
    ranked: PyReadonlyArray2<'py, f32>,
    window: usize,
) -> PyResult<(f64, f64, Vec<Py<PyArray2<f32>>>)> {
    let data = to2_f32(&ranked);
    let min_periods = std::cmp::max(1, window / 2);
    let (ms_serial, ms_rowmajor, outs) = py.allow_threads(move || {
        let t0 = std::time::Instant::now();
        let (m1, x1, n1, s1) = rolling_stats_f32_serial(&data, window, min_periods);
        let ms_serial = t0.elapsed().as_secs_f64() * 1000.0;
        let t0 = std::time::Instant::now();
        let (m2, x2, n2, s2) = rolling_stats_rowmajor(&data, window, min_periods);
        let ms_rowmajor = t0.elapsed().as_secs_f64() * 1000.0;
        (ms_serial, ms_rowmajor, vec![m1, x1, n1, s1, m2, x2, n2, s2])
    });
    Ok((ms_serial, ms_rowmajor, outs.into_iter().map(|o| o.into_pyarray(py).to_owned()).collect()))
}


#[pyfunction]
fn bench_backtest_opt2_py<'py>(
    py: Python<'py>,
    slot: PyReadonlyArray2<'py, f32>,
    ret_gap1: PyReadonlyArray2<'py, f32>,
    ret_sum_gap1: PyReadonlyArray2<'py, f32>,
    ret_gap5: PyReadonlyArray2<'py, f32>,
    ret_sum_gap5: PyReadonlyArray2<'py, f32>,
    restrict: PyReadonlyArray2<'py, f32>,
    index_ret: PyReadonlyArray1<'py, f32>,
    dates: PyReadonlyArray1<'py, i32>,
    backtest_start: i32,
    ic_only: bool,
) -> PyResult<(f64, f64, Vec<f64>, Vec<f32>)> {
    let slot = to2_f32(&slot);
    let ret_g1 = to2_f32(&ret_gap1);
    let ret_s1 = to2_f32(&ret_sum_gap1);
    let ret_g5 = to2_f32(&ret_gap5);
    let ret_s5 = to2_f32(&ret_sum_gap5);
    let restr = to2_f32(&restrict);
    let idx = Array1::from_vec(index_ret.as_array().iter().copied().collect());
    let dts: Vec<i32> = dates.as_array().iter().copied().collect();
    let (pre_ms, opt2_ms, sums, ics) = py.allow_threads(move || {
        let t0 = std::time::Instant::now();
        let pre = build_bt_precomputed4(&ret_s1, &ret_s5, &ret_g1, &ret_g5, &restr);
        let pre_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let open_cnts = precompute_open_symbol_counts(&restr.view());
        let block = slot.view().insert_axis(ndarray::Axis(2));
        let ef = effective_raw_indices_for_slot(&block.view(), &dts, backtest_start, 0);
        let t0 = std::time::Instant::now();
        let o1 = backtest_opt2(
            &block, &ret_g1, &ret_s1, &restr, &idx, &dts, 0, 1, 10, &ef, &open_cnts, ic_only, &pre);
        let o5 = backtest_opt2(
            &block, &ret_g5, &ret_s5, &restr, &idx, &dts, 0, 5, 10, &ef, &open_cnts, ic_only, &pre);
        let opt2_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let mut sums = Vec::new();
        for s in [o1.summary, o5.summary] {
            sums.extend_from_slice(&s);
        }
        let mut ics = Vec::new();
        ics.extend(o1.ic_values);
        ics.extend(o5.ic_values);
        (pre_ms, opt2_ms, sums, ics)
    });
    Ok((pre_ms, opt2_ms, sums, ics))
}


#[pyfunction]
fn bench_backtest_py<'py>(
    py: Python<'py>,
    slot: PyReadonlyArray2<'py, f32>,
    ret_gap1: PyReadonlyArray2<'py, f32>,
    ret_sum_gap1: PyReadonlyArray2<'py, f32>,
    ret_gap5: PyReadonlyArray2<'py, f32>,
    ret_sum_gap5: PyReadonlyArray2<'py, f32>,
    restrict: PyReadonlyArray2<'py, f32>,
    index_ret: PyReadonlyArray1<'py, f32>,
    dates: PyReadonlyArray1<'py, i32>,
    backtest_start: i32,
    ic_only: bool,
) -> PyResult<(f64, f64, f64, Vec<f64>, Vec<f32>, Vec<f32>)> {
    let slot = to2_f32(&slot);
    let ret_g1 = to2_f32(&ret_gap1);
    let ret_s1 = to2_f32(&ret_sum_gap1);
    let ret_g5 = to2_f32(&ret_gap5);
    let ret_s5 = to2_f32(&ret_sum_gap5);
    let restr = to2_f32(&restrict);
    let idx = Array1::from_vec(index_ret.as_array().iter().copied().collect());
    let dts: Vec<i32> = dates.as_array().iter().copied().collect();
    let (pre_ms, base_ms, opt_ms, sums, ic_base, ic_opt) = py.allow_threads(move || {
        let t0 = std::time::Instant::now();
        let pre = build_bt_precomputed(&ret_s1, &ret_s5, &ret_g1, &ret_g5, &restr);
        let pre_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let open_cnts = precompute_open_symbol_counts(&restr.view());
        let block = slot.view().insert_axis(ndarray::Axis(2));
        let ef = effective_raw_indices_for_slot(&block.view(), &dts, backtest_start, 0);
        let t0 = std::time::Instant::now();
        let r1 = legacy_backtest_single_factor_with_effective(
            &block, &ret_g1, &ret_s1, &restr, &idx, &dts, 0, 1, 10, &ef, &open_cnts, ic_only);
        let r5 = legacy_backtest_single_factor_with_effective(
            &block, &ret_g5, &ret_s5, &restr, &idx, &dts, 0, 5, 10, &ef, &open_cnts, ic_only);
        let base_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let t0 = std::time::Instant::now();
        let o1 = backtest_opt(
            &block, &ret_g1, &ret_s1, &restr, &idx, &dts, 0, 1, 10, &ef, &open_cnts, ic_only, &pre);
        let o5 = backtest_opt(
            &block, &ret_g5, &ret_s5, &restr, &idx, &dts, 0, 5, 10, &ef, &open_cnts, ic_only, &pre);
        let opt_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let mut sums = Vec::new();
        for s in [r1.summary, r5.summary, o1.summary, o5.summary] {
            sums.extend_from_slice(&s);
        }
        let mut ic_base = Vec::new();
        ic_base.extend(r1.ic_values);
        ic_base.extend(r5.ic_values);
        let mut ic_opt = Vec::new();
        ic_opt.extend(o1.ic_values);
        ic_opt.extend(o5.ic_values);
        (pre_ms, base_ms, opt_ms, sums, ic_base, ic_opt)
    });
    Ok((pre_ms, base_ms, opt_ms, sums, ic_base, ic_opt))
}

#[pyfunction]
fn pre_neu_py<'py>(
    py: Python<'py>,
    industry: PyReadonlyArray2<'py, f64>,
    restrict: PyReadonlyArray2<'py, f32>,
    barra_ranked: PyReadonlyArray3<'py, f64>,
) -> PyResult<(f64, Vec<Py<PyArray2<usize>>>)> {
    let (t, n) = industry.as_array().dim();
    let ind = Array2::from_shape_vec(industry.as_array().dim(), industry.as_array().iter().copied().collect()).unwrap();
    let restr = to2_f32(&restrict);
    let mut barra = Vec::new();
    for c in 0..10 {
        let mut m = Array2::<f64>::from_elem((t, n), f64::NAN);
        for i in 0..t {
            for j in 0..n {
                m[[i, j]] = barra_ranked.as_array()[[i, j, c]];
            }
        }
        barra.push(m);
    }
    let (ms, orders) = py.allow_threads(move || {
        let t0 = std::time::Instant::now();
        let pre = precompute_neu(&ind, &restr, &barra);
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        (ms, pre.orders)
    });
    Ok((ms, orders.into_iter().map(|o| o.into_pyarray(py).to_owned()).collect()))
}

#[pyfunction]
fn bench_neutralize_py<'py>(
    py: Python<'py>,
    slot: PyReadonlyArray2<'py, f32>,
    industry: PyReadonlyArray2<'py, f64>,
    restrict: PyReadonlyArray2<'py, f32>,
    barra_ranked: PyReadonlyArray3<'py, f64>,
    orders: Vec<PyReadonlyArray2<'py, usize>>,
    mode: &str,
) -> PyResult<(f64, Py<PyArray2<f32>>, u64, f64)> {
    let (t, n) = slot.as_array().dim();
    let slotf = to2_f32(&slot);
    let ind = Array2::from_shape_vec(industry.as_array().dim(), industry.as_array().iter().copied().collect()).unwrap();
    let restr = to2_f32(&restrict);
    let mut barra = Vec::new();
    for c in 0..10 {
        let mut m = Array2::<f64>::from_elem((t, n), f64::NAN);
        for i in 0..t {
            for j in 0..n {
                m[[i, j]] = barra_ranked.as_array()[[i, j, c]];
            }
        }
        barra.push(m);
    }
    let ords: Vec<Array2<usize>> = orders
        .iter()
        .map(|o| Array2::from_shape_vec(o.as_array().dim(), o.as_array().iter().copied().collect()).unwrap())
        .collect();
    let mode_owned = mode.to_string();
    let (ms, out32, fallback, _pre_ms) = py.allow_threads(move || {
        // 预计算 (因子无关, 全 run 一次): 计时单列返回, 不混入 per-slot 耗时
        let pre = if mode_owned == "baseline" {
            None
        } else {
            Some(precompute_neu(&ind, &restr, &barra))
        };
        let t0 = std::time::Instant::now();
        let slot64 = slotf.map(|&v| v as f64);
        let (out64, fallback) = if mode_owned == "baseline" {
            (neutralize_std_section_owned(slot64, &ind, &restr, &barra, true), 0u64)
        } else {
            // v2: 预计算 orders + 逐日 X'X (若 pre 为空则回退 baseline)
            let pre = pre.expect("v2 需要预计算");
            let (out, fb) = neutralize_v2_core(&slot64, &pre, &restr, &barra);
            (out, fb)
        };
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        let out32 = out64.map(|&v| if v.is_nan() { f32::NAN } else { v as f32 });
        let _ = ords;
        (ms, out32, fallback, 0.0)
    });
    Ok((ms, out32.into_pyarray(py).to_owned(), fallback, 0.0))
}

// ==================== 新实验: preflight 变体与 radix 秩 (2026-09-01) ====================

/// 变体R: radix 排序数连 — 与 HashMap 分组计数逐组相等 (按 bits 分组), 无哈希开销。
fn preflight_quality_check_radix(
    raw_values: &ArrayView2<f32>,
    restrict: &ArrayView2<f32>,
    majority_count_threshold: f64,
    zero_max_threshold: f64,
    nan_max_threshold: f64,
) -> PreflightReport {
    let n_dates = raw_values.shape()[0];
    let n_stocks = raw_values.shape()[1];
    let mut majority_sum: f64 = 0.0;
    let mut nan_ratio_sum: f64 = 0.0;
    let mut zero_ratio_sum: f64 = 0.0;
    let mut valid_date_count: usize = 0;
    let mut keys: Vec<u32> = Vec::with_capacity(n_stocks);
    let mut order: Vec<u32> = Vec::with_capacity(n_stocks);
    let mut tmp: Vec<u32> = Vec::with_capacity(n_stocks);
    let mut count = [0usize; 256];
    for t in 0..n_dates {
        keys.clear();
        let mut free_count: usize = 0;
        let mut nan_count: usize = 0;
        let mut zero_count: usize = 0;
        for s in 0..n_stocks {
            let val = raw_values[[t, s]];
            let is_free = restrict[[t, s]].is_finite() && restrict[[t, s]] == 0.0;
            if is_free {
                free_count += 1;
                if !val.is_finite() {
                    nan_count += 1;
                } else if val == 0.0 {
                    zero_count += 1;
                }
            }
            if val.is_finite() {
                keys.push(val.to_bits());
            }
        }
        let n = keys.len();
        order.clear();
        order.extend(0..n as u32);
        if n >= 2 {
            tmp.clear();
            tmp.resize(n, 0);
            for shift in (0..32).step_by(8) {
                count.fill(0);
                for &i in order.iter() {
                    count[((keys[i as usize] >> shift) & 0xff) as usize] += 1;
                }
                let mut acc = 0usize;
                for c in count.iter_mut() {
                    let t = *c;
                    *c = acc;
                    acc += t;
                }
                for &i in order.iter() {
                    let b = ((keys[i as usize] >> shift) & 0xff) as usize;
                    tmp[count[b]] = i;
                    count[b] += 1;
                }
                std::mem::swap(&mut order, &mut tmp);
            }
        }
        let mut max_count = 0usize;
        let mut start = 0usize;
        while start < n {
            let key = keys[order[start] as usize];
            let mut end = start + 1;
            while end < n && keys[order[end] as usize] == key {
                end += 1;
            }
            let c = end - start;
            if c > max_count {
                max_count = c;
            }
            start = end;
        }
        majority_sum += max_count as f64;
        if free_count > 0 {
            nan_ratio_sum += nan_count as f64 / free_count as f64;
            zero_ratio_sum += zero_count as f64 / free_count as f64;
            valid_date_count += 1;
        }
    }
    let majority_count_mean = if n_dates > 0 {
        majority_sum / n_dates as f64
    } else {
        0.0
    };
    let nan_ratio_mean = if valid_date_count > 0 {
        nan_ratio_sum / valid_date_count as f64
    } else {
        0.0
    };
    let zero_ratio_mean = if valid_date_count > 0 {
        zero_ratio_sum / valid_date_count as f64
    } else {
        0.0
    };
    PreflightReport {
        passed: majority_count_mean <= majority_count_threshold
            && zero_ratio_mean < zero_max_threshold
            && nan_ratio_mean < nan_max_threshold,
        majority_count_mean,
        zero_ratio_mean,
        nan_ratio_mean,
    }
}

/// 变体N: majority_count_threshold >= n_stocks 时 majority 判定恒真 (max 计数 <= n_stocks),
/// 跳过值计数。可观测量 (passed/zero_mean/nan_mean/preflight 计数器) 与生产完全一致。
fn preflight_quality_check_nohash(
    raw_values: &ArrayView2<f32>,
    restrict: &ArrayView2<f32>,
    majority_count_threshold: f64,
    zero_max_threshold: f64,
    nan_max_threshold: f64,
) -> PreflightReport {
    let n_dates = raw_values.shape()[0];
    let n_stocks = raw_values.shape()[1];
    let mut nan_ratio_sum: f64 = 0.0;
    let mut zero_ratio_sum: f64 = 0.0;
    let mut valid_date_count: usize = 0;
    for t in 0..n_dates {
        let mut free_count: usize = 0;
        let mut nan_count: usize = 0;
        let mut zero_count: usize = 0;
        for s in 0..n_stocks {
            let val = raw_values[[t, s]];
            let is_free = restrict[[t, s]].is_finite() && restrict[[t, s]] == 0.0;
            if is_free {
                free_count += 1;
                if !val.is_finite() {
                    nan_count += 1;
                } else if val == 0.0 {
                    zero_count += 1;
                }
            }
        }
        if free_count > 0 {
            nan_ratio_sum += nan_count as f64 / free_count as f64;
            zero_ratio_sum += zero_count as f64 / free_count as f64;
            valid_date_count += 1;
        }
    }
    let nan_ratio_mean = if valid_date_count > 0 {
        nan_ratio_sum / valid_date_count as f64
    } else {
        0.0
    };
    let zero_ratio_mean = if valid_date_count > 0 {
        zero_ratio_sum / valid_date_count as f64
    } else {
        0.0
    };
    PreflightReport {
        // majority 恒真 (threshold >= n_stocks); majority_count_mean 不参与下游任何
        // 可观测输出 (仅用于判定与计数), 这里返回 n_stocks 供校验。
        passed: zero_ratio_mean < zero_max_threshold && nan_ratio_mean < nan_max_threshold,
        majority_count_mean: n_stocks as f64,
        zero_ratio_mean,
        nan_ratio_mean,
    }
}

/// 变体R: 每行 radix 平均秩 (mono_key32 稳定排序, 初始顺序=index; 与 rank_average_row 同组同秩)。
fn rank_average_row_radix(row: &[f32]) -> Vec<f32> {
    let n = row.len();
    let mut keys: Vec<u32> = vec![0u32; n];
    let mut order: Vec<u32> = Vec::with_capacity(n);
    for (i, &v) in row.iter().enumerate() {
        if !v.is_nan() {
            keys[i] = mono_key32(v);
            order.push(i as u32);
        }
    }
    let m = order.len();
    let mut tmp = vec![0u32; m];
    let mut count = [0usize; 256];
    for shift in (0..32).step_by(8) {
        count.fill(0);
        for &i in order.iter() {
            count[((keys[i as usize] >> shift) & 0xff) as usize] += 1;
        }
        let mut acc = 0usize;
        for c in count.iter_mut() {
            let t = *c;
            *c = acc;
            acc += t;
        }
        for &i in order.iter() {
            let b = ((keys[i as usize] >> shift) & 0xff) as usize;
            tmp[count[b]] = i;
            count[b] += 1;
        }
        std::mem::swap(&mut order, &mut tmp);
    }
    let mut ranked = vec![f32::NAN; n];
    let mut start = 0usize;
    while start < m {
        let key = keys[order[start] as usize];
        let mut end = start + 1;
        while end < m && keys[order[end] as usize] == key {
            end += 1;
        }
        let avg_rank = ((start + 1 + end) as f64 / 2.0) as f32;
        for &i in order.iter().take(end).skip(start) {
            ranked[i as usize] = avg_rank;
        }
        start = end;
    }
    ranked
}

fn rank_axis1_average_f32_radix(data: &Array2<f32>) -> Array2<f32> {
    let (n_rows, n_cols) = data.dim();
    let mut flat = vec![f32::NAN; n_rows * n_cols];
    for row_idx in 0..n_rows {
        let row = data.row(row_idx);
        let ranked = rank_average_row_radix(row.as_slice().unwrap_or(&[]));
        let start = row_idx * n_cols;
        flat[start..start + n_cols].copy_from_slice(&ranked);
    }
    Array2::from_shape_vec((n_rows, n_cols), flat).unwrap()
}

#[pyfunction]
fn bench_preflight_py<'py>(
    py: Python<'py>,
    slot: PyReadonlyArray2<'py, f32>,
    restrict: PyReadonlyArray2<'py, f32>,
    majority_count_threshold: f64,
    zero_max_threshold: f64,
    nan_max_threshold: f64,
    mode: &str,
) -> PyResult<(f64, bool, f64, f64, f64)> {
    let s = to2_f32(&slot);
    let r = to2_f32(&restrict);
    let mode_owned = mode.to_string();
    let (ms, rep) = py.allow_threads(move || {
        let t0 = std::time::Instant::now();
        let rep = match mode_owned.as_str() {
            "radix" => preflight_quality_check_radix(
                &s.view(),
                &r.view(),
                majority_count_threshold,
                zero_max_threshold,
                nan_max_threshold,
            ),
            "nohash" => preflight_quality_check_nohash(
                &s.view(),
                &r.view(),
                majority_count_threshold,
                zero_max_threshold,
                nan_max_threshold,
            ),
            _ => preflight_quality_check(
                &s.view(),
                &r.view(),
                majority_count_threshold,
                zero_max_threshold,
                nan_max_threshold,
            ),
        };
        (t0.elapsed().as_secs_f64() * 1000.0, rep)
    });
    Ok((ms, rep.passed, rep.majority_count_mean, rep.zero_ratio_mean, rep.nan_ratio_mean))
}

#[pyfunction]
fn bench_rank_radix_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, f32>,
    mode: &str,
) -> PyResult<(f64, Py<PyArray2<f32>>)> {
    let d = to2_f32(&data);
    let mode_owned = mode.to_string();
    let (ms, out) = py.allow_threads(move || {
        let t0 = std::time::Instant::now();
        let out = if mode_owned == "radix" {
            rank_axis1_average_f32_radix(&d)
        } else {
            rank_axis1_average_f32_serial(&d)
        };
        (t0.elapsed().as_secs_f64() * 1000.0, out)
    });
    Ok((ms, out.into_pyarray(py).to_owned()))
}

// ==================== 新实验: neutralize 分阶段计时 + 微优化包 (2026-09-01) ====================
// 语义基准: 引擎当前路径 neutralize_std_section_owned_v2_resid (C': OLS 后无最终 rank_pct)。
// "cur" = 逐位复刻当前实现 (含 clone/逐级 to_vec/每级独立 nan_mask);
// "opt" = 微优化包: ①首 rank 直接在 f32 源上做 (u32 key, 4-pass)  ②fill_ind_reg 就地无 to_vec
//           ③去掉 fv_filled clone (就地继续)  ④三级中位填充融合为单次行拷贝+三次走段+一次写回。

/// 第一阶段 rank_pct: 输入 f32 源, 输出 f64 pct 秩 (与 f64 版分组/数值逐位一致)。
fn rank_pct_row_f32_to_f64(vals_f32: &[f32], out_ranks: &mut [f64]) {
    let n = vals_f32.len();
    let mut keys: Vec<u32> = vec![0u32; n];
    let mut idxs: Vec<usize> = Vec::with_capacity(n);
    for (i, &v) in vals_f32.iter().enumerate() {
        if !v.is_nan() {
            idxs.push(i);
            keys[i] = mono_key32(v);
        }
    }
    for r in out_ranks.iter_mut() {
        *r = f64::NAN;
    }
    let n_valid = idxs.len();
    if n_valid == 0 {
        return;
    }
    // u32 key 稳定 LSD (初始顺序 = index 升序, 与 (key,index) 总序一致)
    let mut order: Vec<u32> = idxs.iter().map(|&i| i as u32).collect();
    let mut tmp = vec![0u32; n_valid];
    let mut count = [0usize; 256];
    for shift in (0..32).step_by(8) {
        count.fill(0);
        for &i in order.iter() {
            count[((keys[i as usize] >> shift) & 0xff) as usize] += 1;
        }
        let mut acc = 0usize;
        for c in count.iter_mut() {
            let t = *c;
            *c = acc;
            acc += t;
        }
        for &i in order.iter() {
            let b = ((keys[i as usize] >> shift) & 0xff) as usize;
            tmp[count[b]] = i;
            count[b] += 1;
        }
        std::mem::swap(&mut order, &mut tmp);
    }
    let mut i = 0usize;
    while i < n_valid {
        let mut j = i;
        while j + 1 < n_valid && vals_f32[order[j + 1] as usize] == vals_f32[order[i] as usize] {
            j += 1;
        }
        let avg_rank = ((i + 1) + (j + 1)) as f64 / 2.0;
        let pct = avg_rank / n_valid as f64;
        for item in order[i..=j].iter() {
            out_ranks[*item as usize] = pct;
        }
        i = j + 1;
    }
}

fn rank_pct_all_f32_to_f64(values: &Array2<f32>) -> Array2<f64> {
    let (t, n) = values.dim();
    let mut out = Array2::<f64>::from_elem((t, n), f64::NAN);
    let mut ranks: Vec<f64> = Vec::with_capacity(n);
    for idx in 0..t {
        ranks.resize(n, f64::NAN);
        let row = values.row(idx);
        rank_pct_row_f32_to_f64(row.as_slice().unwrap(), &mut ranks);
        for j in 0..n {
            out[[idx, j]] = ranks[j];
        }
    }
    out
}

/// fill_ind_reg_pre 就地版: 无逐级 to_vec, 直接按行段操作 (段不重叠, 语义不变)。
fn fill_ind_reg_pre_inplace(
    fv: &mut Array2<f64>,
    ind2: &Array2<f64>,
    ind1: &Array2<f64>,
    size_ranked: &Array2<f64>,
    orders: &[Array2<usize>],
) -> (u64, u64) {
    let (t, n) = fv.dim();
    let ind0 = ind1.map(|&v| if v.is_nan() { 0.0 } else { 1.0 });
    let levels = [ind2, ind1, &ind0];
    let mut ys: Vec<f64> = Vec::with_capacity(n);
    let mut bs: Vec<f64> = Vec::with_capacity(n);
    let mut skip = 0u64;
    let mut done = 0u64;
    for (li, level) in levels.iter().enumerate() {
        for idx in 0..t {
            {
                let row_slice = fv.row(idx).as_slice().unwrap().to_vec();
                if !has_ge_n_unique(&row_slice, 10) {
                    skip += 1;
                    continue;
                }
            }
            done += 1;
            let order_arr = orders[li].row(idx);
            let order = order_arr.as_slice().unwrap();
            let mut row = fv.row_mut(idx);
            let row_slice = row.as_slice_mut().unwrap(); // 借用: 段内读写 (段不重叠)
            let mut seg_start = 0usize;
            while seg_start < n {
                let code = level[[idx, order[seg_start]]];
                if code.is_nan() {
                    break;
                }
                let mut seg_end = seg_start + 1;
                while seg_end < n && level[[idx, order[seg_end]]] == code {
                    seg_end += 1;
                }
                ys.clear();
                bs.clear();
                let mut obs: Vec<bool> = Vec::with_capacity(seg_end - seg_start);
                for &ci in &order[seg_start..seg_end] {
                    let ok = !row_slice[ci].is_nan() && !size_ranked[[idx, ci]].is_nan();
                    obs.push(ok);
                    if ok {
                        ys.push(row_slice[ci]);
                        bs.push(size_ranked[[idx, ci]]);
                    }
                }
                if ys.len() >= 10 {
                    let (c0, c1) = ols2(&ys, &bs);
                    for (mi, &ci) in order[seg_start..seg_end].iter().enumerate() {
                        if !obs[mi] {
                            row_slice[ci] = c0 + c1 * size_ranked[[idx, ci]];
                        }
                    }
                }
                seg_start = seg_end;
            }
        }
    }
    (skip, done)
}

/// 三级中位填充融合: 每日期一次行拷贝 + 一次 nan_mask, 三次走段, 一次写回。
fn group_median_fill_fused(
    values: &mut Array2<f64>,
    ind2: &Array2<f64>,
    ind1: &Array2<f64>,
    ind1_mask: &Array2<f64>,
    zeros: &Array2<f64>,
    orders: &[Array2<usize>],
) -> u64 {
    let (t, n) = values.dim();
    let mut sv: Vec<f64> = Vec::with_capacity(n);
    let mut skipped = 0u64;
    let levels: [(&Array2<f64>, Option<&Array2<f64>>, usize); 3] = [
        (ind2, None, 3),
        (ind1, None, 4),
        (zeros, Some(ind1_mask), 0),
    ];
    for idx in 0..t {
        let mut row = values.row(idx).to_vec();
        let mut nan_mask = Vec::<bool>::with_capacity(n);
        let mut has_nan = false;
        for j in 0..n {
            let nn = row[j].is_nan();
            nan_mask.push(nn);
            if nn {
                has_nan = true;
            }
        }
        if !has_nan {
            skipped += 1;
            continue;
        }
        for (codes, valid_mask, oi) in levels.iter() {
            let order_arr = orders[*oi].row(idx);
            let order = order_arr.as_slice().unwrap();
            let mut seg_start = 0usize;
            while seg_start < n {
                let code = codes[[idx, order[seg_start]]];
                if code.is_nan() {
                    break;
                }
                let mut seg_end = seg_start + 1;
                while seg_end < n && codes[[idx, order[seg_end]]] == code {
                    seg_end += 1;
                }
                sv.clear();
                for &ci in &order[seg_start..seg_end] {
                    let valid = valid_mask.map_or(true, |vm| vm[[idx, ci]] == 1.0);
                    if valid && !row[ci].is_nan() {
                        sv.push(row[ci]);
                    }
                }
                if !sv.is_empty() {
                    let med = median_inplace(&mut sv);
                    for &ci in &order[seg_start..seg_end] {
                        let valid = valid_mask.map_or(true, |vm| vm[[idx, ci]] == 1.0);
                        // 语义与逐级分调一致: nan_mask 基于"当前行"状态,
                        // 已被前级填充的位置不再是 NaN, 后级不再覆盖。
                        if nan_mask[ci] && valid {
                            row[ci] = med;
                            nan_mask[ci] = false;
                        }
                    }
                }
                seg_start = seg_end;
            }
        }
        for j in 0..n {
            values[[idx, j]] = row[j];
        }
    }
    skipped
}

/// 分阶段执行引擎当前 C' 路径; mode: "cur" 逐位复刻当前实现; "opt1/opt2/opt3/opt" 逐步启用
/// 微优化 (opt1=首rank f32 4-pass; opt2=+fill 就地; opt3=+去clone+融合中位; opt=全量)。
/// 返回 (每阶段毫秒数组[9], fallback天数, 输出 f32 矩阵)。
fn neutralize_slot_v2_resid_bench(
    slot32: &Array2<f32>,
    pre: &NeuPrecomputed,
    restr: &Array2<f32>,
    barra_ranked: &[Array2<f64>],
    mode: &str,
) -> (Vec<f64>, u64, Array2<f32>) {
    let (t, n) = slot32.dim();
    let mut st = [0u64; 9];
    let use_r1 = mode != "cur";
    let use_fill = matches!(mode, "opt2" | "opt3" | "opt");
    let use_med = matches!(mode, "opt3" | "opt");
    let mut fv;
    if use_r1 {
        // S0+S1: 直接在 f32 源上 rank (u32 key 4-pass), 产出 f64 pct 矩阵
        let t0 = std::time::Instant::now();
        fv = rank_pct_all_f32_to_f64(slot32);
        st[0] += t0.elapsed().as_nanos() as u64;
    } else {
        let t0 = std::time::Instant::now();
        fv = Array2::<f64>::from_shape_vec(
            (t, n),
            slot32.iter().map(|&v| v as f64).collect(),
        )
        .expect("shape");
        st[0] += t0.elapsed().as_nanos() as u64;
        let t0 = std::time::Instant::now();
        rank_pct_all(&mut fv);
        st[1] += t0.elapsed().as_nanos() as u64;
    }
    let t0 = std::time::Instant::now();
    if use_fill {
        fill_ind_reg_pre_inplace(&mut fv, &pre.ind2, &pre.ind1, &barra_ranked[2], &pre.orders[0..3]);
    } else {
        fill_ind_reg_pre(&mut fv, &pre.ind2, &pre.ind1, &barra_ranked[2], &pre.orders[0..3]);
    }
    st[2] += t0.elapsed().as_nanos() as u64;
    let t0 = std::time::Instant::now();
    for i in 0..(t * n) {
        if pre.ind1.as_slice().unwrap()[i].is_nan() {
            fv.as_slice_mut().unwrap()[i] = f64::NAN;
        }
    }
    if !use_med {
        // 保留引擎 clone (基准语义), 除非 use_med 意味着就地继续
        let fv2 = fv.clone();
        fv = fv2;
    }
    st[3] += t0.elapsed().as_nanos() as u64;
    let t0 = std::time::Instant::now();
    if use_med {
        let skipped = group_median_fill_fused(
            &mut fv, &pre.ind2, &pre.ind1, &pre.ind1_mask, &pre.zeros, &pre.orders,
        );
        let _ = skipped;
    } else {
        let _ = group_median_fill_pre(&mut fv, &pre.ind2, None, &pre.orders[3]);
        let _ = group_median_fill_pre(&mut fv, &pre.ind1, None, &pre.orders[4]);
        let _ = group_median_fill_pre(&mut fv, &pre.zeros, Some(&pre.ind1_mask), &pre.orders[0]);
    }
    st[4] += t0.elapsed().as_nanos() as u64;
    let t0 = std::time::Instant::now();
    for i in 0..(t * n) {
        if restr.as_slice().unwrap()[i] != 0.0 {
            fv.as_slice_mut().unwrap()[i] = f64::NAN;
        }
    }
    st[5] += t0.elapsed().as_nanos() as u64;
    let t0 = std::time::Instant::now();
    rank_pct_all(&mut fv);
    st[6] += t0.elapsed().as_nanos() as u64;
    let t0 = std::time::Instant::now();
    let mut fallback = 0u64;
    let resid = get_residual_v2(&fv, barra_ranked, &pre.ind1, &pre.per_date, &mut fallback);
    st[7] += t0.elapsed().as_nanos() as u64;
    let t0 = std::time::Instant::now();
    let out = Array2::<f32>::from_shape_vec(
        (t, n),
        resid
            .iter()
            .map(|&v| if v.is_nan() { f32::NAN } else { v as f32 })
            .collect(),
    )
    .expect("shape");
    st[8] += t0.elapsed().as_nanos() as u64;
    (
        st.iter().map(|&x| x as f64 / 1e6).collect(),
        fallback,
        out,
    )
}

#[pyfunction]
fn bench_neutralize_stages_py<'py>(
    py: Python<'py>,
    slot: PyReadonlyArray2<'py, f32>,
    industry: PyReadonlyArray2<'py, f64>,
    restrict: PyReadonlyArray2<'py, f32>,
    barra_ranked: PyReadonlyArray3<'py, f64>,
    mode: &str,
) -> PyResult<(Vec<f64>, u64, Py<PyArray2<f32>>, f64)> {
    let (t, n) = slot.as_array().dim();
    let slotf = to2_f32(&slot);
    let ind = Array2::from_shape_vec(
        industry.as_array().dim(),
        industry.as_array().iter().copied().collect(),
    )
    .unwrap();
    let restr = to2_f32(&restrict);
    let mut barra = Vec::new();
    for c in 0..10 {
        let mut m = Array2::<f64>::from_elem((t, n), f64::NAN);
        for i in 0..t {
            for j in 0..n {
                m[[i, j]] = barra_ranked.as_array()[[i, j, c]];
            }
        }
        barra.push(m);
    }
    let mode_owned = mode.to_string();
    let (stages, fb, out, pre_ms) = py.allow_threads(move || {
        let t0 = std::time::Instant::now();
        let pre = precompute_neu(&ind, &restr, &barra);
        let pre_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let (stages, fb, out) =
            neutralize_slot_v2_resid_bench(&slotf, &pre, &restr, &barra, &mode_owned);
        (stages, fb, out, pre_ms)
    });
    Ok((stages, fb, out.into_pyarray(py).to_owned(), pre_ms))
}

#[pymodule]
fn dev_sandbox_engine_opt(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(bench_rank_py, m)?)?;
    m.add_function(wrap_pyfunction!(bench_rolling_py, m)?)?;
    m.add_function(wrap_pyfunction!(bench_backtest_py, m)?)?;
    m.add_function(wrap_pyfunction!(bench_backtest_opt2_py, m)?)?;
    m.add_function(wrap_pyfunction!(pre_neu_py, m)?)?;
    m.add_function(wrap_pyfunction!(bench_neutralize_py, m)?)?;
    m.add_function(wrap_pyfunction!(bench_preflight_py, m)?)?;
    m.add_function(wrap_pyfunction!(bench_rank_radix_py, m)?)?;
    m.add_function(wrap_pyfunction!(bench_neutralize_stages_py, m)?)?;
    Ok(())
}
