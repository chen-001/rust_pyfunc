//! 优化变体（sandbox 实验用，未改动生产代码）：
//! - fused：每日过滤一次 + 信号只排序一次（同时供 spearman 序数与十分组平均秩用）
//! - prerank：把 ret_sum 的横截面序数秩在整轮回测前一次性预计算（跨 factor/slot 复用）
//!   ——这是近似优化：预计算秩定义在全「open+ret 有限」集合上，而生产 spearman 秩
//!   定义在「当日过滤子集」上，两者在信号缺失位置附近会有微小差异，需实测偏差量。
use std::cmp::Ordering;

use ndarray::{ArrayView1, ArrayView2, ArrayView3};

use crate::engine::{self, LegacyBacktestResult};

/// 与生产 ordinal_ranks 完全一致的序数秩。
/// sorted 已按 (value, pos) 升序排好（pos = 过滤后向量中的位置），
/// 返回 ranks[pos] = 序数秩。
fn ordinal_ranks_positional(sorted: &[(usize, f32)]) -> Vec<i64> {
    let mut ranks = vec![0i64; sorted.len()];
    for (rank, &(pos, _)) in sorted.iter().enumerate() {
        ranks[pos] = rank as i64;
    }
    ranks
}

/// 与生产 average_ranks 完全一致的平均秩（stable sort by (value, idx)，ties 平均）。
/// sorted 必须已按 (value, idx) 升序排好。
fn average_ranks_from_sorted(values: &[f32], sorted: &[(usize, f32)]) -> Vec<f64> {
    let mut ranks = vec![f64::NAN; values.len()];
    let mut start = 0usize;
    while start < sorted.len() {
        let value = sorted[start].1;
        let mut end = start + 1;
        while end < sorted.len() && sorted[end].1 == value {
            end += 1;
        }
        let avg_rank = (start + 1 + end) as f64 / 2.0;
        for item in sorted.iter().take(end).skip(start) {
            ranks[item.0] = avg_rank;
        }
        start = end;
    }
    ranks
}

/// fused 版回测：语义与 legacy_backtest_single_factor_with_effective 完全一致
/// （逐位可对账），但每日过滤一次 + 信号排序一次复用。
#[allow(clippy::too_many_arguments)]
pub fn backtest_fused(
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
) -> LegacyBacktestResult {
    if effective_raw_indices.is_empty() {
        return LegacyBacktestResult::default();
    }
    let n_stocks = factor.shape()[1];
    let date_size = effective_raw_indices.len();
    let mut group_returns = vec![vec![0.0_f64; date_size]; portf_num];
    let mut ratio_values = vec![f64::NAN; date_size];
    let mut ic_dates = Vec::<i32>::new();
    let mut ic_values_f64 = Vec::<f64>::new();
    let mut ic_values_f32 = Vec::<f32>::new();
    let mut sig_idx: Vec<(usize, f32)> = Vec::with_capacity(n_stocks);
    let mut sig_val: Vec<f32> = Vec::with_capacity(n_stocks);
    let mut fut_val: Vec<f32> = Vec::with_capacity(n_stocks);
    let mut ret_val: Vec<f32> = Vec::with_capacity(n_stocks);
    let mut fut_sorted: Vec<(usize, f32)> = Vec::with_capacity(n_stocks);
    let mut group_sums = vec![0.0_f64; portf_num];
    let mut group_counts = vec![0usize; portf_num];
    let mut held_signal_row_idx = effective_raw_indices[0] - 1;
    let mut held_restrict_row_idx = effective_raw_indices[0] - 1;

    for (local_t, &raw_eff_idx) in effective_raw_indices.iter().enumerate() {
        if local_t % gap == 0 {
            held_signal_row_idx = raw_eff_idx - 1;
            held_restrict_row_idx = raw_eff_idx - 1;
        }

        sig_idx.clear();
        sig_val.clear();
        fut_val.clear();
        ret_val.clear();
        for stock_idx in 0..n_stocks {
            let signal_value = factor[[held_signal_row_idx, stock_idx, slot_idx]];
            let ret_value = ret[[raw_eff_idx, stock_idx]];
            let is_open = restrict[[held_restrict_row_idx, stock_idx]].is_finite()
                && restrict[[held_restrict_row_idx, stock_idx]] == 0.0;
            if signal_value.is_finite() && ret_value.is_finite() && is_open {
                sig_idx.push((sig_val.len(), signal_value));
                sig_val.push(signal_value);
                ret_val.push(ret_value);
                fut_val.push(ret_sum[[raw_eff_idx, stock_idx]]);
            }
        }
        // 信号排序一次：stable by (value, idx) —— 与 ordinal_ranks/average_ranks 同序
        sig_idx.sort_by(|a, b| {
            a.1.partial_cmp(&b.1)
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });

        if (local_t + 1) % gap == 0 {
            // spearman：x 序数秩直接来自 sig_idx 的排序位置；y 仍需一次排序
            let n = sig_val.len();
            if n < 2 {
                ic_dates.push(dates[raw_eff_idx]);
                ic_values_f64.push(f64::NAN);
                ic_values_f32.push(f32::NAN);
            } else {
                let xx = ordinal_ranks_positional(&sig_idx);
                fut_sorted.clear();
                fut_sorted.extend(
                    fut_val
                        .iter()
                        .copied()
                        .enumerate()
                        .collect::<Vec<(usize, f32)>>(),
                );
                fut_sorted.sort_by(|a, b| {
                    a.1.partial_cmp(&b.1)
                        .unwrap_or(Ordering::Equal)
                        .then_with(|| a.0.cmp(&b.0))
                });
                let yy = ordinal_ranks_positional(&fut_sorted);
                let nf = n as f64;
                let mut diff_sq_sum = 0.0;
                for idx in 0..n {
                    let diff = xx[idx] - yy[idx];
                    diff_sq_sum += (diff * diff) as f64;
                }
                let ic = 1.0 - 6.0 * diff_sq_sum / (nf * (nf * nf - 1.0));
                ic_dates.push(dates[raw_eff_idx]);
                ic_values_f64.push(ic);
                ic_values_f32.push(ic as f32);
            }
        }

        let stocks_num = sig_val.len();
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

        group_sums.fill(0.0);
        group_counts.fill(0);
        let ranks = average_ranks_from_sorted(&sig_val, &sig_idx);
        for idx in 0..stocks_num {
            let pct = ranks[idx] / stocks_num as f64;
            let mut bucket = (pct * portf_num as f64).floor() as usize;
            if bucket >= portf_num {
                bucket = portf_num - 1;
            }
            group_sums[bucket] += ret_val[idx] as f64;
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

    let ic_mean = engine::nanmean_f64(&ic_values_f64);
    let ic_std = engine::nanstd_population(&ic_values_f64);
    let ir = if ic_std.is_nan() || ic_std <= engine::EPS {
        f64::NAN
    } else {
        ic_mean.abs() / ic_std * (250.0 / gap as f64).sqrt()
    };
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

    LegacyBacktestResult {
        summary: [
            ic_mean,
            ir,
            engine::nanmean_f64(&ls_returns) * 250.0,
            engine::annualized_sharpe_sample(&ls_returns),
            engine::max_drawdown_from_returns(&ls_returns),
            date_size as f64,
            engine::nanmean_f64(&ratio_values),
            engine::nanmean_f64(&hedge_returns) * 250.0,
            engine::annualized_sharpe_sample(&hedge_returns),
            engine::max_drawdown_from_returns(&hedge_returns),
        ],
        ic_dates,
        ic_values: ic_values_f32,
    }
}

/// 预计算每日 ret_sum 在「open 且 ret 有限」全集上的序数秩（f32 序数 + 平均秩两套）。
/// 一次 O(T·N·logN) 计算，跨所有 factor/slot 复用。
pub struct PrecomputedRetRanks {
    /// ordinal_rank[t][s] = 序数秩（0-based），非 open/非有限为 -1
    pub ordinal: Vec<Vec<i32>>,
    /// avg_rank[t][s] = 平均秩（1-based），非 open/非有限为 NaN
    pub avg: Vec<Vec<f32>>,
}

pub fn precompute_ret_ranks(ret_sum: &ArrayView2<'_, f32>, restrict: &ArrayView2<'_, f32>) -> PrecomputedRetRanks {
    let (t, n) = ret_sum.dim();
    let mut ordinal = Vec::with_capacity(t);
    let mut avg = Vec::with_capacity(t);
    let mut row: Vec<(usize, f32)> = Vec::with_capacity(n);
    for day in 0..t {
        row.clear();
        for s in 0..n {
            let is_open = restrict[[day, s]].is_finite() && restrict[[day, s]] == 0.0;
            let v = ret_sum[[day, s]];
            if is_open && v.is_finite() {
                row.push((s, v));
            }
        }
        row.sort_by(|a, b| {
            a.1.partial_cmp(&b.1)
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        let mut ord = vec![-1i32; n];
        let mut av = vec![f32::NAN; n];
        let mut start = 0usize;
        while start < row.len() {
            let value = row[start].1;
            let mut end = start + 1;
            while end < row.len() && row[end].1 == value {
                end += 1;
            }
            let avg_rank = (start + 1 + end) as f32 / 2.0;
            for (pos, &(idx, _)) in row.iter().enumerate().take(end).skip(start) {
                ord[idx] = pos as i32;
                av[idx] = avg_rank;
            }
            start = end;
        }
        ordinal.push(ord);
        avg.push(av);
    }
    PrecomputedRetRanks { ordinal, avg }
}

/// prerank 版回测（近似）：IC 用「全集预计算序数秩」代替过滤子集序数秩，
/// 与生产结果存在微小偏差（偏差量由 bench 输出 max/mean abs diff 量化）。
#[allow(clippy::too_many_arguments)]
pub fn backtest_prerank(
    factor: &ArrayView3<'_, f32>,
    ret: &ArrayView2<'_, f32>,
    restrict: &ArrayView2<'_, f32>,
    index: &ArrayView1<'_, f32>,
    dates: &[i32],
    slot_idx: usize,
    gap: usize,
    portf_num: usize,
    effective_raw_indices: &[usize],
    open_symbol_counts: &[usize],
    prerank: &PrecomputedRetRanks,
) -> LegacyBacktestResult {
    if effective_raw_indices.is_empty() {
        return LegacyBacktestResult::default();
    }
    let n_stocks = factor.shape()[1];
    let date_size = effective_raw_indices.len();
    let mut group_returns = vec![vec![0.0_f64; date_size]; portf_num];
    let mut ratio_values = vec![f64::NAN; date_size];
    let mut ic_dates = Vec::<i32>::new();
    let mut ic_values_f64 = Vec::<f64>::new();
    let mut ic_values_f32 = Vec::<f32>::new();
    let mut sig_idx: Vec<(usize, f32)> = Vec::with_capacity(n_stocks);
    let mut sig_val: Vec<f32> = Vec::with_capacity(n_stocks);
    let mut ret_val: Vec<f32> = Vec::with_capacity(n_stocks);
    let mut position_of: Vec<usize> = vec![usize::MAX; n_stocks];
    let mut xx: Vec<i64> = Vec::with_capacity(n_stocks);
    let mut y_ranks: Vec<i64> = Vec::with_capacity(n_stocks);
    let mut group_sums = vec![0.0_f64; portf_num];
    let mut group_counts = vec![0usize; portf_num];
    let mut held_signal_row_idx = effective_raw_indices[0] - 1;
    let mut held_restrict_row_idx = effective_raw_indices[0] - 1;

    for (local_t, &raw_eff_idx) in effective_raw_indices.iter().enumerate() {
        if local_t % gap == 0 {
            held_signal_row_idx = raw_eff_idx - 1;
            held_restrict_row_idx = raw_eff_idx - 1;
        }

        sig_idx.clear();
        sig_val.clear();
        ret_val.clear();
        for stock_idx in 0..n_stocks {
            let signal_value = factor[[held_signal_row_idx, stock_idx, slot_idx]];
            let ret_value = ret[[raw_eff_idx, stock_idx]];
            let is_open = restrict[[held_restrict_row_idx, stock_idx]].is_finite()
                && restrict[[held_restrict_row_idx, stock_idx]] == 0.0;
            if signal_value.is_finite() && ret_value.is_finite() && is_open {
                position_of[stock_idx] = sig_val.len();
                sig_idx.push((stock_idx, signal_value));
                sig_val.push(signal_value);
                ret_val.push(ret_value);
            }
        }
        sig_idx.sort_by(|a, b| {
            a.1.partial_cmp(&b.1)
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });

        if (local_t + 1) % gap == 0 {
            let n = sig_val.len();
            if n < 2 {
                ic_dates.push(dates[raw_eff_idx]);
                ic_values_f64.push(f64::NAN);
                ic_values_f32.push(f32::NAN);
            } else {
                xx.clear();
                xx.resize(n, 0);
                y_ranks.clear();
                y_ranks.resize(n, 0);
                for (rank, &(stock_idx, _)) in sig_idx.iter().enumerate() {
                    let pos = position_of[stock_idx];
                    xx[pos] = rank as i64;
                    y_ranks[pos] = prerank.ordinal[raw_eff_idx][stock_idx] as i64;
                }
                let nf = n as f64;
                let mut diff_sq_sum = 0.0;
                for idx in 0..n {
                    let diff = xx[idx] - y_ranks[idx];
                    diff_sq_sum += (diff * diff) as f64;
                }
                let ic = 1.0 - 6.0 * diff_sq_sum / (nf * (nf * nf - 1.0));
                ic_dates.push(dates[raw_eff_idx]);
                ic_values_f64.push(ic);
                ic_values_f32.push(ic as f32);
            }
        }

        let stocks_num = sig_val.len();
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

        group_sums.fill(0.0);
        group_counts.fill(0);
        // 平均秩：sorted 已按 (value, stock_idx) 排序；秩落在过滤后位置空间
        let ranks: Vec<f64> = {
            let mut ranks = vec![f64::NAN; stocks_num];
            let mut start = 0usize;
            while start < sig_idx.len() {
                let value = sig_idx[start].1;
                let mut end = start + 1;
                while end < sig_idx.len() && sig_idx[end].1 == value {
                    end += 1;
                }
                let avg_rank = (start + 1 + end) as f64 / 2.0;
                for item in sig_idx.iter().take(end).skip(start) {
                    ranks[position_of[item.0]] = avg_rank;
                }
                start = end;
            }
            ranks
        };
        for idx in 0..stocks_num {
            let pct = ranks[idx] / stocks_num as f64;
            let mut bucket = (pct * portf_num as f64).floor() as usize;
            if bucket >= portf_num {
                bucket = portf_num - 1;
            }
            group_sums[bucket] += ret_val[idx] as f64;
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

    let ic_mean = engine::nanmean_f64(&ic_values_f64);
    let ic_std = engine::nanstd_population(&ic_values_f64);
    let ir = if ic_std.is_nan() || ic_std <= engine::EPS {
        f64::NAN
    } else {
        ic_mean.abs() / ic_std * (250.0 / gap as f64).sqrt()
    };
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

    LegacyBacktestResult {
        summary: [
            ic_mean,
            ir,
            engine::nanmean_f64(&ls_returns) * 250.0,
            engine::annualized_sharpe_sample(&ls_returns),
            engine::max_drawdown_from_returns(&ls_returns),
            date_size as f64,
            engine::nanmean_f64(&ratio_values),
            engine::nanmean_f64(&hedge_returns) * 250.0,
            engine::annualized_sharpe_sample(&hedge_returns),
            engine::max_drawdown_from_returns(&hedge_returns),
        ],
        ic_dates,
        ic_values: ic_values_f32,
    }
}
