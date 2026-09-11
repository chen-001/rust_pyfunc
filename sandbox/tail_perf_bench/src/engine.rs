//! tail_pipeline_engine 单因子计算核心的逐函数复刻（与生产 tail_v5_pipeline.rs
//! 完全一致，仅去掉 pyo3/numpy 依赖），用于 sandbox 性能基准与优化对照。
//! 各步骤用 Instant 计时；optimized 变体在 opt.rs。
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet, VecDeque};
use std::time::Instant;

use ndarray::{Array1, Array2, Array3, ArrayView1, ArrayView2, ArrayView3, Axis};

pub const EPS: f64 = 1e-12;
pub const UNIVERSE_LOOKBACK: usize = 20;

// ---------- 计时器 ----------
#[derive(Default, Clone)]
pub struct StepTimes {
    pub raw_cover: f64,
    pub rank_fill: f64,
    pub rolling: f64,
    pub preflight: f64,
    pub bt_raw: f64,
    pub fold: f64,
    pub slots: usize,
    pub bt_calls: usize,
}

// ---------- 统计 ----------
pub fn nanmean_f64(values: &[f64]) -> f64 {
    let mut sum = 0.0;
    let mut count = 0usize;
    for &value in values {
        if !value.is_nan() {
            sum += value;
            count += 1;
        }
    }
    if count == 0 {
        f64::NAN
    } else {
        sum / count as f64
    }
}

pub fn nanstd_population(values: &[f64]) -> f64 {
    let mean = nanmean_f64(values);
    if mean.is_nan() {
        return f64::NAN;
    }
    let mut sq_sum = 0.0;
    let mut count = 0usize;
    for &value in values {
        if !value.is_nan() {
            let delta = value - mean;
            sq_sum += delta * delta;
            count += 1;
        }
    }
    if count == 0 {
        f64::NAN
    } else {
        (sq_sum / count as f64).sqrt()
    }
}

pub fn sample_std(values: &[f64]) -> f64 {
    let mut count = 0usize;
    let mut sum = 0.0;
    for &value in values {
        if !value.is_nan() {
            sum += value;
            count += 1;
        }
    }
    if count < 2 {
        return f64::NAN;
    }
    let mean = sum / count as f64;
    let mut sq_sum = 0.0;
    for &value in values {
        if !value.is_nan() {
            let delta = value - mean;
            sq_sum += delta * delta;
        }
    }
    (sq_sum / (count as f64 - 1.0)).sqrt()
}

pub fn annualized_sharpe_sample(values: &[f64]) -> f64 {
    let std = sample_std(values);
    if std.is_nan() || std <= EPS {
        return f64::NAN;
    }
    nanmean_f64(values) / std * 250.0_f64.sqrt()
}

pub fn max_drawdown_from_returns(values: &[f64]) -> f64 {
    let mut cumulative = 0.0;
    let mut peak = 0.0;
    let mut max_drawdown = 0.0;
    for &value in values {
        if !value.is_nan() {
            cumulative += value;
        }
        if cumulative > peak {
            peak = cumulative;
        }
        let drawdown = peak - cumulative;
        if drawdown > max_drawdown {
            max_drawdown = drawdown;
        }
    }
    max_drawdown
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
            .unwrap_or(Ordering::Equal)
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
        (true, false) => Ordering::Greater,
        (false, true) => Ordering::Less,
        (false, false) => lhs
            .1
            .partial_cmp(&rhs.1)
            .unwrap_or(Ordering::Equal)
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

// ---------- rank / rolling / fill ----------
fn rank_average_row(row: &[f32]) -> Vec<f32> {
    let mut indexed = row
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, value)| !value.is_nan())
        .collect::<Vec<_>>();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));

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

pub fn rank_axis1_average_f32_serial(data: &Array2<f32>) -> Array2<f32> {
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

pub fn rolling_stats_f32_serial(
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

// 优化版 rolling：按行（日期）主序遍历 + 每股票独立累加器，cache 友好。
pub fn rolling_stats_f32_rowmajor(
    ranked: &Array2<f32>,
    window: usize,
    min_periods: usize,
) -> (Array2<f32>, Array2<f32>, Array2<f32>, Array2<f32>) {
    let (n_rows, n_cols) = ranked.dim();
    let mut mean = Array2::<f32>::from_elem((n_rows, n_cols), f32::NAN);
    let mut max = Array2::<f32>::from_elem((n_rows, n_cols), f32::NAN);
    let mut min = Array2::<f32>::from_elem((n_rows, n_cols), f32::NAN);
    let mut std = Array2::<f32>::from_elem((n_rows, n_cols), f32::NAN);

    let mut sum = vec![0.0f64; n_cols];
    let mut sumsq = vec![0.0f64; n_cols];
    let mut count = vec![0usize; n_cols];
    let mut max_deque = vec![VecDeque::<(usize, f32)>::new(); n_cols];
    let mut min_deque = vec![VecDeque::<(usize, f32)>::new(); n_cols];

    for row_idx in 0..n_rows {
        let row = ranked.row(row_idx);
        for (col_idx, &value) in row.iter().enumerate() {
            let md = &mut max_deque[col_idx];
            let nd = &mut min_deque[col_idx];
            if !value.is_nan() {
                let value64 = value as f64;
                sum[col_idx] += value64;
                sumsq[col_idx] += value64 * value64;
                count[col_idx] += 1;

                while let Some((_, tail_val)) = md.back() {
                    if *tail_val <= value {
                        md.pop_back();
                    } else {
                        break;
                    }
                }
                md.push_back((row_idx, value));
                while let Some((_, tail_val)) = nd.back() {
                    if *tail_val >= value {
                        nd.pop_back();
                    } else {
                        break;
                    }
                }
                nd.push_back((row_idx, value));
            }

            if row_idx >= window {
                let leave_value = ranked[[row_idx - window, col_idx]];
                if !leave_value.is_nan() {
                    let leave64 = leave_value as f64;
                    sum[col_idx] -= leave64;
                    sumsq[col_idx] -= leave64 * leave64;
                    count[col_idx] -= 1;
                }
            }

            let valid_start = (row_idx + 1).saturating_sub(window);
            while let Some((idx, _)) = md.front() {
                if *idx < valid_start {
                    md.pop_front();
                } else {
                    break;
                }
            }
            while let Some((idx, _)) = nd.front() {
                if *idx < valid_start {
                    nd.pop_front();
                } else {
                    break;
                }
            }

            if count[col_idx] >= min_periods {
                let c = count[col_idx] as f64;
                mean[[row_idx, col_idx]] = (sum[col_idx] / c) as f32;
                max[[row_idx, col_idx]] = md.front().map(|i| i.1).unwrap_or(f32::NAN);
                min[[row_idx, col_idx]] = nd.front().map(|i| i.1).unwrap_or(f32::NAN);
                if count[col_idx] > 1 {
                    let variance = ((sumsq[col_idx] - (sum[col_idx] * sum[col_idx]) / c)
                        / (c - 1.0))
                        .max(0.0);
                    std[[row_idx, col_idx]] = variance.sqrt() as f32;
                }
            }
        }
    }
    (mean, max, min, std)
}

pub fn fill_missing_rank_with_cross_sectional_median(ranked: &mut Array2<f32>, restrict: &Array2<f32>) {
    // 生产口径 (tail_v5_pipeline.rs:5997)：只填"缺失"——当日 Restrict==0 可交易、
    // 仅当天缺值的位置。不可交易(停牌/涨跌停)与未上市保持 NaN。
    let n_dates = ranked.nrows();
    let n_stocks = ranked.ncols();
    for date_idx in 0..n_dates {
        let mut valid_count = 0usize;
        for stock_idx in 0..n_stocks {
            if ranked[[date_idx, stock_idx]].is_finite() {
                valid_count += 1;
            }
        }
        if valid_count == 0 {
            continue;
        }
        let median_rank = ((valid_count + 1) as f32) / 2.0;
        for stock_idx in 0..n_stocks {
            if !ranked[[date_idx, stock_idx]].is_finite() {
                if restrict[[date_idx, stock_idx]] == 0.0 {
                    ranked[[date_idx, stock_idx]] = median_rank;
                }
            }
        }
    }
}

pub fn rank_and_fill_missing_cross_sectional_median(
    variant_values: &Array2<f32>,
    restrict: &Array2<f32>,
) -> Array2<f32> {
    let mut ranked = rank_axis1_average_f32_serial(variant_values);
    fill_missing_rank_with_cross_sectional_median(&mut ranked, restrict);
    ranked
}

// ---------- preflight / cover ----------
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
    let mut seen = HashSet::<u32>::new();
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

pub struct PreflightReport {
    pub passed: bool,
    pub majority_count_mean: f64,
    pub zero_ratio_mean: f64,
    pub nan_ratio_mean: f64,
}

pub fn preflight_quality_check(
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
        let mut value_counts: HashMap<u32, usize> = HashMap::new();
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

pub fn compute_raw_cover_rate(
    raw_values: &ArrayView2<f32>,
    restrict: &ArrayView2<f32>,
    ret: &ArrayView2<f32>,
    min_stocks: usize,
) -> f64 {
    let n_dates = raw_values.shape()[0];
    let n_stocks = raw_values.shape()[1];
    let mut ratios: Vec<f64> = Vec::new();

    for t in 0..n_dates {
        let mut free_count: usize = 0;
        let mut valid_count: usize = 0;
        for s in 0..n_stocks {
            let is_free = restrict[[t, s]].is_finite() && restrict[[t, s]] == 0.0;
            if is_free {
                free_count += 1;
                if !raw_values[[t, s]].is_nan() && ret[[t, s]].is_finite() {
                    valid_count += 1;
                }
            }
        }
        if free_count > 0 && valid_count >= min_stocks {
            ratios.push(valid_count as f64 / free_count as f64);
        }
    }

    if ratios.is_empty() {
        1.0
    } else {
        ratios.iter().sum::<f64>() / ratios.len() as f64
    }
}

// ---------- backtest ----------
#[derive(Default, Clone)]
pub struct LegacyBacktestResult {
    pub summary: [f64; 10],
    pub ic_dates: Vec<i32>,
    pub ic_values: Vec<f32>,
}

fn default_legacy_backtest_result() -> LegacyBacktestResult {
    LegacyBacktestResult {
        summary: [f64::NAN; 10],
        ic_dates: Vec::new(),
        ic_values: Vec::new(),
    }
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

pub fn legacy_backtest_single_factor_with_effective(
    factor: &ArrayView3<'_, f32>,
    ret: &ArrayView2<'_, f32>,
    ret_sum: &ArrayView2<'_, f32>,
    restrict: &ArrayView2<'_, f32>,
    index: &ArrayView1<'_, f32>,
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

    let ic_mean = nanmean_f64(&ic_values_f64);
    let ic_std = nanstd_population(&ic_values_f64);
    let ir = if ic_std.is_nan() || ic_std <= EPS {
        f64::NAN
    } else {
        ic_mean.abs() / ic_std * (250.0 / gap as f64).sqrt()
    };
    let summary = if ic_only {
        [
            ic_mean,
            ir,
            0.0,
            0.0,
            0.0,
            date_size as f64,
            nanmean_f64(&ratio_values),
            0.0,
            0.0,
            0.0,
        ]
    } else {
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
            nanmean_f64(&ratio_values),
            nanmean_f64(&hedge_returns) * 250.0,
            annualized_sharpe_sample(&hedge_returns),
            max_drawdown_from_returns(&hedge_returns),
        ]
    };
    LegacyBacktestResult {
        summary,
        ic_dates,
        ic_values: ic_values_f32,
    }
}

pub fn legacy_backtest_gap1_gap5_single_slot(
    slot: ArrayView2<'_, f32>,
    ret_gap1: ArrayView2<'_, f32>,
    ret_sum_gap1: ArrayView2<'_, f32>,
    ret_gap5: ArrayView2<'_, f32>,
    ret_sum_gap5: ArrayView2<'_, f32>,
    restrict: ArrayView2<'_, f32>,
    index: ArrayView1<'_, f32>,
    dates: &[i32],
    backtest_start: i32,
    portf_num: usize,
    open_symbol_counts: &[usize],
    ic_only: bool,
) -> (LegacyBacktestResult, LegacyBacktestResult) {
    let n_dates = slot.nrows();
    let slot_block = slot.insert_axis(Axis(2));
    if n_dates < 2 || !has_enough_unique_values(&slot_block, 0, 10) {
        return (
            default_legacy_backtest_result(),
            default_legacy_backtest_result(),
        );
    }
    let effective_raw_indices =
        effective_raw_indices_for_slot(&slot_block, dates, backtest_start, 0);
    (
        legacy_backtest_single_factor_with_effective(
            &slot_block,
            &ret_gap1,
            &ret_sum_gap1,
            &restrict,
            &index,
            dates,
            0,
            1,
            portf_num,
            &effective_raw_indices,
            open_symbol_counts,
            ic_only,
        ),
        legacy_backtest_single_factor_with_effective(
            &slot_block,
            &ret_gap5,
            &ret_sum_gap5,
            &restrict,
            &index,
            dates,
            0,
            5,
            portf_num,
            &effective_raw_indices,
            open_symbol_counts,
            ic_only,
        ),
    )
}

pub fn build_fold_values(raw_values: &Array2<f32>) -> Array2<f32> {
    let n_dates = raw_values.nrows();
    let n_stocks = raw_values.ncols();
    let mut folded = Array2::<f32>::from_elem((n_dates, n_stocks), f32::NAN);
    for date_idx in 0..n_dates {
        let mut sum = 0.0_f64;
        let mut count = 0usize;
        for stock_idx in 0..n_stocks {
            let value = raw_values[[date_idx, stock_idx]];
            if value.is_finite() {
                sum += value as f64;
                count += 1;
            }
        }
        if count == 0 {
            continue;
        }
        let mean = (sum / count as f64) as f32;
        for stock_idx in 0..n_stocks {
            let value = raw_values[[date_idx, stock_idx]];
            if value.is_finite() {
                folded[[date_idx, stock_idx]] = (value - mean).abs();
            }
        }
    }
    folded
}

// ---------- 共享输入（复刻 SharedInputs 的最小集） ----------
pub struct Shared {
    pub cover_rate: f64,
    pub dates: Vec<i32>,
    pub windows: Vec<usize>,
    pub fold: bool,
    pub backtest_start: i32,
    pub ret_gap1: Array2<f32>,
    pub ret_sum_gap1: Array2<f32>,
    pub ret_gap5: Array2<f32>,
    pub ret_sum_gap5: Array2<f32>,
    pub restrict: Array2<f32>,
    pub index_ret: Array1<f32>,
    pub majority_count_threshold: f64,
    pub zero_max_threshold: f64,
    pub nan_max_threshold: f64,
}

// ---------- per-variant 流水线（与 process_v7_variant/process_v7_slot 对齐，
// 中性化步骤用外部计时器占位，neu 回测按同一函数复算） ----------
pub struct VariantStats {
    pub times: StepTimes,
    pub raw_summaries: Vec<(String, i32, [f64; 10])>, // (derived_name, gap, summary)
    pub neu_summaries: Vec<(String, i32, [f64; 10])>,
}

pub fn run_variant(
    variant_name: &str,
    ranked: Array2<f32>,
    shared: &Shared,
    open_symbol_counts: &[usize],
    run_neu: bool,
) -> VariantStats {
    let mut times = StepTimes::default();
    let (n_dates, n_stocks) = ranked.dim();
    assert_eq!(shared.ret_gap1.dim(), (n_dates, n_stocks));

    let mut derived_names: Vec<String> = vec![format!("{}_smooth_1", variant_name)];
    for &window in &shared.windows {
        derived_names.push(format!("{}_mean_smooth_{}", variant_name, window));
        derived_names.push(format!("{}_max_smooth_{}", variant_name, window));
        derived_names.push(format!("{}_min_smooth_{}", variant_name, window));
        derived_names.push(format!("{}_std_smooth_{}", variant_name, window));
    }

    let mut raw_summaries = Vec::new();
    let mut neu_summaries = Vec::new();
    let mut slot_idx = 0usize;

    // 逐 slot：preflight → raw 回测 →（中性化 + neu 回测）
    let mut process_slot = |slot: ArrayView2<'_, f32>, times: &mut StepTimes| -> bool {
        let name = derived_names[slot_idx].clone();
        slot_idx += 1;
        times.slots += 1;

        let t = Instant::now();
        let pre = preflight_quality_check(
            &slot,
            &shared.restrict.view(),
            shared.majority_count_threshold,
            shared.zero_max_threshold,
            shared.nan_max_threshold,
        );
        times.preflight += t.elapsed().as_secs_f64();
        if !pre.passed {
            return false;
        }

        let t = Instant::now();
        let (r1, r5) = legacy_backtest_gap1_gap5_single_slot(
            slot,
            shared.ret_gap1.view(),
            shared.ret_sum_gap1.view(),
            shared.ret_gap5.view(),
            shared.ret_sum_gap5.view(),
            shared.restrict.view(),
            shared.index_ret.view(),
            &shared.dates,
            shared.backtest_start,
            10,
            open_symbol_counts,
            false,
        );
        times.bt_raw += t.elapsed().as_secs_f64();
        times.bt_calls += 2;
        raw_summaries.push((name.clone(), 1, r1.summary));
        raw_summaries.push((name.clone(), 5, r5.summary));

        if run_neu {
            // 中性化耗时在 main 中用公共 API 单独测量；这里用同一回测函数模拟 neu 侧。
            let t = Instant::now();
            let (n1, n5) = legacy_backtest_gap1_gap5_single_slot(
                slot,
                shared.ret_gap1.view(),
                shared.ret_sum_gap1.view(),
                shared.ret_gap5.view(),
                shared.ret_sum_gap5.view(),
                shared.restrict.view(),
                shared.index_ret.view(),
                &shared.dates,
                shared.backtest_start,
                10,
                open_symbol_counts,
                false,
            );
            times.bt_raw += t.elapsed().as_secs_f64(); // 计入 bt（与 neu 同量级）
            times.bt_calls += 2;
            neu_summaries.push((name.clone(), 1, n1.summary));
            neu_summaries.push((name, 5, n5.summary));
        }
        true
    };

    // slot 0: _smooth_1
    process_slot(ranked.view(), &mut times);

    // window 派生 slot
    for &window in &shared.windows {
        let min_periods = std::cmp::max(1, window / 2);
        let t = Instant::now();
        let (mean, max, min, std) =
            rolling_stats_f32_serial(&ranked, window, min_periods);
        times.rolling += t.elapsed().as_secs_f64();
        process_slot(mean.view(), &mut times);
        drop(mean);
        process_slot(max.view(), &mut times);
        drop(max);
        process_slot(min.view(), &mut times);
        drop(min);
        process_slot(std.view(), &mut times);
        drop(std);
    }

    VariantStats {
        times,
        raw_summaries,
        neu_summaries,
    }
}

// ---------- 单因子整体（复刻 process_task_with_values_v7 的 raw 部分） ----------
pub struct TaskStats {
    pub times: StepTimes,
    pub raw_cover_before_fill: f64,
    pub raw_cover_after_fill: f64,
    pub eliminated_by_raw_cover: bool,
    pub raw_summaries: Vec<(String, i32, [f64; 10])>,
}

pub fn run_factor_v7(
    source_factor: &str,
    raw_values: Array2<f32>,
    shared: &Shared,
    run_neu: bool,
) -> TaskStats {
    let mut times = StepTimes::default();

    let t = Instant::now();
    let raw_cover_before_fill = compute_raw_cover_rate(
        &raw_values.view(),
        &shared.restrict.view(),
        &shared.ret_gap1.view(),
        10,
    );
    times.raw_cover += t.elapsed().as_secs_f64();

    let t = Instant::now();
    let ranked_raw = rank_and_fill_missing_cross_sectional_median(&raw_values, &shared.restrict);
    let raw_cover_rate = compute_raw_cover_rate(
        &ranked_raw.view(),
        &shared.restrict.view(),
        &shared.ret_gap1.view(),
        10,
    );
    times.rank_fill += t.elapsed().as_secs_f64();

    if raw_cover_rate < shared.cover_rate {
        return TaskStats {
            times,
            raw_cover_before_fill,
            raw_cover_after_fill: raw_cover_rate,
            eliminated_by_raw_cover: true,
            raw_summaries: Vec::new(),
        };
    }

    let open_symbol_counts = precompute_open_symbol_counts(&shared.restrict.view());

    let raw_stats = run_variant(
        source_factor,
        ranked_raw,
        shared,
        &open_symbol_counts,
        run_neu,
    );
    times.slots += raw_stats.times.slots;
    times.bt_calls += raw_stats.times.bt_calls;
    times.bt_raw += raw_stats.times.bt_raw;
    times.preflight += raw_stats.times.preflight;
    times.rolling += raw_stats.times.rolling;
    times.rank_fill += raw_stats.times.rank_fill;
    times.raw_cover += raw_stats.times.raw_cover;
    let mut raw_summaries = raw_stats.raw_summaries;

    if shared.fold {
        let t = Instant::now();
        let folded = build_fold_values(&raw_values);
        let ranked_fold = rank_and_fill_missing_cross_sectional_median(&folded, &shared.restrict);
        times.fold += t.elapsed().as_secs_f64();
        drop(raw_values);
        drop(folded);
        let fold_stats = run_variant(
            &format!("{}_fold", source_factor),
            ranked_fold,
            shared,
            &open_symbol_counts,
            run_neu,
        );
        times.slots += fold_stats.times.slots;
        times.bt_calls += fold_stats.times.bt_calls;
        times.bt_raw += fold_stats.times.bt_raw;
        times.preflight += fold_stats.times.preflight;
        times.rolling += fold_stats.times.rolling;
        times.rank_fill += fold_stats.times.rank_fill;
        raw_summaries.extend(fold_stats.raw_summaries);
    }

    TaskStats {
        times,
        raw_cover_before_fill,
        raw_cover_after_fill: raw_cover_rate,
        eliminated_by_raw_cover: false,
        raw_summaries,
    }
}
