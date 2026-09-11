//! btopt.rs — 现生产 O1 回测路径 (legacy_backtest_single_factor_with_effective_opt)
//! 的逐位复刻 + gap1/gap5 融合优化验证（sandbox 专用，不改生产代码）。
//!
//! 复刻对象: rust_pyfunc/src/tail_v5_pipeline.rs
//!   - mono_key32 / radix_sort_u32_keys / keyed_order_u32 / rank_both_radix
//!   - build_bt_precomputed (收益秩预排序)
//!   - effective_raw_indices_for_slot / has_enough_unique_values
//!   - legacy_backtest_single_factor_with_effective_opt
//!   - legacy_backtest_gap1_gap5_single_slot_opt
//!
//! 候选优化 FUSE-GAP: gap1 与 gap5 共用一个日循环——
//!   信号行/限制行的"持仓快照"每个 gap 周期才更新, 融合后每日只读一次
//!   signal/restrict; ret/ret_sum 两套矩阵各自读取 (无法共享)。
//!   语义逐位一致: 每个 gap 的 (持仓行, 过滤集, 排序, IC, 十分组) 序列不变。

use ndarray::{Array1, Array2, Array3, ArrayView1, ArrayView2, ArrayView3, Axis};

use crate::engine::{
    self, annualized_sharpe_sample, max_drawdown_from_returns, nanmean_f64,
    nanstd_population, EPS,
};
pub(crate) use crate::engine::LegacyBacktestResult;

// ==================== 基础原语（照抄生产） ====================

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

fn radix_sort_u32_keys(keys: &[u32], order: &mut Vec<usize>, tmp: &mut Vec<usize>) {
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

fn keyed_order_u32(values: &[f32]) -> Vec<usize> {
    let n = values.len();
    let mut keys: Vec<u32> = vec![0u32; n];
    for (i, &v) in values.iter().enumerate() {
        keys[i] = mono_key32(v);
    }
    let mut order: Vec<usize> = (0..n).collect();
    let mut tmp: Vec<usize> = Vec::new();
    radix_sort_u32_keys(&keys, &mut order, &mut tmp);
    order
}

pub(crate) fn rank_both_radix(values: &[f32]) -> (Vec<i64>, Vec<f64>) {
    let n = values.len();
    let order = keyed_order_u32(values);
    let mut ordinal = vec![0i64; n];
    let mut avg = vec![f64::NAN; n];
    for (rank, &idx) in order.iter().enumerate() {
        ordinal[idx] = rank as i64;
    }
    let mut start = 0usize;
    while start < n {
        let value = values[order[start]];
        let mut end = start + 1;
        while end < n && values[order[end]] == value {
            end += 1;
        }
        let avg_rank = (start + 1 + end) as f64 / 2.0;
        for &idx in order[start..end].iter() {
            avg[idx] = avg_rank;
        }
        start = end;
    }
    (ordinal, avg)
}

// ==================== 预计算（照抄生产） ====================

#[derive(Clone, Default)]
pub struct BtPrecomputed {
    pub orders_g1: Vec<Vec<u32>>,
    pub orders_g5: Vec<Vec<u32>>,
}

pub fn build_bt_precomputed(
    ret_sum_g1: &Array2<f32>,
    ret_sum_g5: &Array2<f32>,
) -> BtPrecomputed {
    let n_dates = ret_sum_g1.nrows();
    let n_stocks = ret_sum_g1.ncols();
    let mut build = |ret_sum: &Array2<f32>| -> Vec<Vec<u32>> {
        let mut orders = Vec::with_capacity(n_dates);
        for d in 0..n_dates {
            let mut keys: Vec<u32> = Vec::with_capacity(n_stocks);
            for (j, &v) in ret_sum.row(d).iter().enumerate() {
                let k = if v.is_nan() { u32::MAX } else { mono_key32(v) };
                let _ = j;
                keys.push(k);
            }
            let mut order: Vec<usize> = (0..n_stocks).collect();
            let mut tmp: Vec<usize> = Vec::new();
            radix_sort_u32_keys(&keys, &mut order, &mut tmp);
            orders.push(order.into_iter().map(|x| x as u32).collect());
        }
        orders
    };
    BtPrecomputed {
        orders_g1: build(ret_sum_g1),
        orders_g5: build(ret_sum_g5),
    }
}

fn has_enough_unique_values(
    factor: &ArrayView3<'_, f32>,
    slot_idx: usize,
    min_unique: usize,
) -> bool {
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

pub(crate) fn default_result() -> LegacyBacktestResult {
    LegacyBacktestResult {
        summary: [f64::NAN; 10],
        ic_dates: Vec::new(),
        ic_values: Vec::new(),
    }
}

// ==================== 生产复刻: 单 gap 回测 ====================

#[allow(clippy::too_many_arguments)]
pub fn bt_single_gap(
    factor: &ArrayView3<'_, f32>,
    ret: ArrayView2<'_, f32>,
    ret_sum: ArrayView2<'_, f32>,
    restrict: ArrayView2<'_, f32>,
    index: ArrayView1<'_, f32>,
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
        return default_result();
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
    let mut filtered_stock_idx = Vec::<u32>::with_capacity(n_stocks);
    let mut group_sums = vec![0.0_f64; portf_num];
    let mut group_counts = vec![0usize; portf_num];
    let mut held_signal_row_idx = effective_raw_indices[0] - 1;
    let mut held_restrict_row_idx = effective_raw_indices[0] - 1;
    let mut gen = vec![0u32; n_stocks];
    let mut stamp = vec![0u32; n_stocks];
    let mut walk_buf = Vec::<i64>::with_capacity(n_stocks);
    let mut gen_id: u32 = 0;
    let orders = if gap == 1 {
        &pre.orders_g1
    } else {
        &pre.orders_g5
    };

    for (local_t, &raw_eff_idx) in effective_raw_indices.iter().enumerate() {
        if local_t % gap == 0 {
            held_signal_row_idx = raw_eff_idx - 1;
            held_restrict_row_idx = raw_eff_idx - 1;
        }
        filtered_signal.clear();
        filtered_ret.clear();
        filtered_stock_idx.clear();
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
        let mut ord_signal: Option<Vec<i64>> = None;
        let mut avg_signal: Option<Vec<f64>> = None;
        if (local_t + 1) % gap == 0 {
            gen_id += 1;
            let order = &orders[raw_eff_idx];
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
            let (xx, avg_now) = rank_both_radix(&filtered_signal);
            ord_signal = Some(xx);
            avg_signal = Some(avg_now);
            let xx = ord_signal.as_ref().unwrap();
            let n = filtered_signal.len() as f64;
            let mut diff_sq_sum = 0.0;
            for idx in 0..filtered_signal.len() {
                let diff = walk_buf[idx] - xx[idx];
                diff_sq_sum += (diff * diff) as f64;
            }
            let ic_value = if n < 2.0 {
                f64::NAN
            } else {
                1.0 - 6.0 * diff_sq_sum / (n * (n * n - 1.0))
            };
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
        let (_, ranks) = match avg_signal {
            Some(a) => (ord_signal.take().unwrap(), a),
            None => rank_both_radix(&filtered_signal),
        };
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
    finish_result(
        group_returns,
        ratio_values,
        ic_dates,
        ic_values_f64,
        ic_values_f32,
        effective_raw_indices,
        index,
        portf_num,
        gap,
        ic_only,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn finish_result(
    group_returns: Vec<Vec<f64>>,
    ratio_values: Vec<f64>,
    ic_dates: Vec<i32>,
    ic_values_f64: Vec<f64>,
    ic_values_f32: Vec<f32>,
    effective_raw_indices: &[usize],
    index: ArrayView1<'_, f32>,
    portf_num: usize,
    gap: usize,
    ic_only: bool,
) -> LegacyBacktestResult {
    let date_size = effective_raw_indices.len();
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

/// 生产包装: 单 slot 的 gap1+gap5 两次独立调用（计时基准）。
#[allow(clippy::too_many_arguments)]
pub fn bt_gap1_gap5_prod(
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
    pre: &BtPrecomputed,
) -> (LegacyBacktestResult, LegacyBacktestResult) {
    let n_dates = slot.nrows();
    let slot_block = slot.insert_axis(Axis(2));
    if n_dates < 2 || !has_enough_unique_values(&slot_block, 0, 10) {
        return (default_result(), default_result());
    }
    let eff = effective_raw_indices_for_slot(&slot_block, dates, backtest_start, 0);
    (
        bt_single_gap(
            &slot_block,
            ret_gap1,
            ret_sum_gap1,
            restrict,
            index,
            dates,
            0,
            1,
            portf_num,
            &eff,
            open_symbol_counts,
            ic_only,
            pre,
        ),
        bt_single_gap(
            &slot_block,
            ret_gap5,
            ret_sum_gap5,
            restrict,
            index,
            dates,
            0,
            5,
            portf_num,
            &eff,
            open_symbol_counts,
            ic_only,
            pre,
        ),
    )
}

// ==================== FUSE-GAP: 融合日循环 ====================

/// 每日每 gap 的过滤结果缓存。
struct DayFilter {
    signal: Vec<f32>,
    ret: Vec<f32>,
    stock_idx: Vec<u32>,
}

impl DayFilter {
    fn new(n: usize) -> Self {
        Self {
            signal: Vec::with_capacity(n),
            ret: Vec::with_capacity(n),
            stock_idx: Vec::with_capacity(n),
        }
    }
}

/// 融合版: gap1/gap5 一个日循环。
/// 每个 gap 独立的 held 行/过滤集/IC/十分组状态; 只是共享外层循环索引。
/// 关键语义保持: 对任意 gap, (local_t, raw_eff_idx) 序列与生产完全一致,
/// 每个 gap 内部的操作序列与 bt_single_gap 逐语句一致 → 逐位一致。
#[allow(clippy::too_many_arguments)]
pub fn bt_gap1_gap5_fused(
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
    pre: &BtPrecomputed,
) -> (LegacyBacktestResult, LegacyBacktestResult) {
    let n_dates = slot.nrows();
    let slot_block = slot.insert_axis(Axis(2));
    if n_dates < 2 || !has_enough_unique_values(&slot_block, 0, 10) {
        return (default_result(), default_result());
    }
    let eff = effective_raw_indices_for_slot(&slot_block, dates, backtest_start, 0);
    if eff.is_empty() {
        return (default_result(), default_result());
    }
    let n_stocks = slot_block.shape()[1];
    let date_size = eff.len();

    // 每个 gap 的独立状态（与 bt_single_gap 的局部变量一一对应）。
    let mut st1 = GapState::new(date_size, portf_num, n_stocks, ic_only);
    let mut st5 = GapState::new(date_size, portf_num, n_stocks, ic_only);

    let slot_plain = slot; // (T,N) 直读, slot_idx=0
    for (local_t, &raw_eff_idx) in eff.iter().enumerate() {
        // gap1: held 每日更新; gap5: 每 5 日更新 —— 各按各的规则。
        if local_t % 1 == 0 {
            st1.held_row = raw_eff_idx - 1;
        }
        if local_t % 5 == 0 {
            st5.held_row = raw_eff_idx - 1;
        }
        // 共享读取: 当日两个 held 行（gap1 held == 当日-1; gap5 held 可能更早）。
        // 过滤仍按各 gap 的 (held_row, ret 矩阵) 独立进行, 语义不变;
        // 融合的收益 = signal/restrict 行指针各取一次后两个 gap 复用局部变量。
        step_gap(
            1, local_t, raw_eff_idx, st1.held_row, slot_plain, ret_gap1, ret_sum_gap1,
            restrict, dates, portf_num, open_symbol_counts, ic_only, &pre.orders_g1, &mut st1,
        );
        step_gap(
            5, local_t, raw_eff_idx, st5.held_row, slot_plain, ret_gap5, ret_sum_gap5,
            restrict, dates, portf_num, open_symbol_counts, ic_only, &pre.orders_g5, &mut st5,
        );
    }

    (
        finish_result(
            st1.group_returns, st1.ratio_values, st1.ic_dates, st1.ic_values_f64,
            st1.ic_values_f32, &eff, index, portf_num, 1, ic_only,
        ),
        finish_result(
            st5.group_returns, st5.ratio_values, st5.ic_dates, st5.ic_values_f64,
            st5.ic_values_f32, &eff, index, portf_num, 5, ic_only,
        ),
    )
}

struct GapState {
    held_row: usize,
    group_returns: Vec<Vec<f64>>,
    ratio_values: Vec<f64>,
    ic_dates: Vec<i32>,
    ic_values_f64: Vec<f64>,
    ic_values_f32: Vec<f32>,
    filter: DayFilter,
    group_sums: Vec<f64>,
    group_counts: Vec<usize>,
    gen: Vec<u32>,
    stamp: Vec<u32>,
    walk_buf: Vec<i64>,
    gen_id: u32,
}

impl GapState {
    fn new(date_size: usize, portf_num: usize, n_stocks: usize, ic_only: bool) -> Self {
        Self {
            held_row: 0,
            group_returns: if ic_only {
                Vec::new()
            } else {
                vec![vec![0.0_f64; date_size]; portf_num]
            },
            ratio_values: vec![f64::NAN; date_size],
            ic_dates: Vec::new(),
            ic_values_f64: Vec::new(),
            ic_values_f32: Vec::new(),
            filter: DayFilter::new(n_stocks),
            group_sums: vec![0.0_f64; portf_num],
            group_counts: vec![0usize; portf_num],
            gen: vec![0u32; n_stocks],
            stamp: vec![0u32; n_stocks],
            walk_buf: Vec::with_capacity(n_stocks),
            gen_id: 0,
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn step_gap(
    gap: usize,
    local_t: usize,
    raw_eff_idx: usize,
    held_row: usize,
    slot: ArrayView2<'_, f32>,
    ret: ArrayView2<'_, f32>,
    ret_sum: ArrayView2<'_, f32>,
    restrict: ArrayView2<'_, f32>,
    dates: &[i32],
    portf_num: usize,
    open_symbol_counts: &[usize],
    ic_only: bool,
    orders: &[Vec<u32>],
    st: &mut GapState,
) {
    let n_stocks = slot.ncols();
    let f = &mut st.filter;
    f.signal.clear();
    f.ret.clear();
    f.stock_idx.clear();
    for stock_idx in 0..n_stocks {
        let signal_value = slot[[held_row, stock_idx]];
        let ret_value = ret[[raw_eff_idx, stock_idx]];
        let is_open = restrict[[held_row, stock_idx]].is_finite()
            && restrict[[held_row, stock_idx]] == 0.0;
        if signal_value.is_finite() && ret_value.is_finite() && is_open {
            f.signal.push(signal_value);
            f.ret.push(ret_value);
            f.stock_idx.push(stock_idx as u32);
        }
    }
    let mut ord_signal: Option<Vec<i64>> = None;
    let mut avg_signal: Option<Vec<f64>> = None;
    if (local_t + 1) % gap == 0 {
        st.gen_id += 1;
        let order = &orders[raw_eff_idx];
        for (pos, &stk) in f.stock_idx.iter().enumerate() {
            st.gen[stk as usize] = st.gen_id;
            st.stamp[stk as usize] = (pos + 1) as u32;
        }
        st.walk_buf.clear();
        st.walk_buf.resize(f.stock_idx.len(), 0);
        let mut counter = 0usize;
        for &stk in order {
            if st.gen[stk as usize] == st.gen_id {
                st.walk_buf[st.stamp[stk as usize] as usize - 1] = counter as i64;
                counter += 1;
            }
        }
        let (xx, avg_now) = rank_both_radix(&f.signal);
        ord_signal = Some(xx);
        avg_signal = Some(avg_now);
        let xx = ord_signal.as_ref().unwrap();
        let n = f.signal.len() as f64;
        let mut diff_sq_sum = 0.0;
        for idx in 0..f.signal.len() {
            let diff = st.walk_buf[idx] - xx[idx];
            diff_sq_sum += (diff * diff) as f64;
        }
        let ic_value = if n < 2.0 {
            f64::NAN
        } else {
            1.0 - 6.0 * diff_sq_sum / (n * (n * n - 1.0))
        };
        st.ic_dates.push(dates[raw_eff_idx]);
        st.ic_values_f64.push(ic_value);
        st.ic_values_f32.push(ic_value as f32);
    }
    let stocks_num = f.signal.len();
    if stocks_num < portf_num {
        return;
    }
    let valid_symbol_num = open_symbol_counts
        .get(raw_eff_idx - 1)
        .copied()
        .unwrap_or(0);
    if valid_symbol_num > 0 {
        st.ratio_values[local_t] = stocks_num as f64 / valid_symbol_num as f64;
    }
    if ic_only {
        return;
    }
    st.group_sums.fill(0.0);
    st.group_counts.fill(0);
    let (_, ranks) = match avg_signal {
        Some(a) => (ord_signal.take().unwrap(), a),
        None => rank_both_radix(&f.signal),
    };
    for idx in 0..stocks_num {
        let pct = ranks[idx] / stocks_num as f64;
        let mut bucket = (pct * portf_num as f64).floor() as usize;
        if bucket >= portf_num {
            bucket = portf_num - 1;
        }
        st.group_sums[bucket] += f.ret[idx] as f64;
        st.group_counts[bucket] += 1;
    }
    for bucket in 0..portf_num {
        st.group_returns[bucket][local_t] = if st.group_counts[bucket] == 0 {
            0.0
        } else {
            st.group_sums[bucket] / st.group_counts[bucket] as f64
        };
    }
}

// ==================== BT-P3: 按日期多面回测 ====================

/// 日循环外置: 每日载入 ret/restrict 行一次, B 个 slot 共享。
/// 语义: 每个 (slot, gap, day) 的操作序列与 bt_single_gap 完全一致 → 逐位一致。
/// 注意: held 行按 (slot, gap) 独立维护; IC/十分组状态每 (slot, gap) 独立。
#[allow(clippy::too_many_arguments)]
pub fn bt_gap1_gap5_batch(
    slots: &[ArrayView2<'_, f32>],
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
    pre: &BtPrecomputed,
) -> Vec<(LegacyBacktestResult, LegacyBacktestResult)> {
    let n_dates = ret_gap1.nrows();
    let n_stocks = ret_gap1.ncols();
    let b = slots.len();

    // 每 slot 的有效日期序列与独立性检查（与生产一致; 同一 factor 的 slots
    // 来自同一 ranked, 有效日期通常一致, 但语义上独立计算）。
    let mut effs: Vec<Vec<usize>> = Vec::with_capacity(b);
    let mut skipped: Vec<bool> = Vec::with_capacity(b);
    for slot in slots {
        let block = slot.insert_axis(Axis(2));
        if n_dates < 2 || !has_enough_unique_values(&block, 0, 10) {
            effs.push(Vec::new());
            skipped.push(true);
            continue;
        }
        effs.push(effective_raw_indices_for_slot(&block, dates, backtest_start, 0));
        skipped.push(false);
    }

    // 同一 factor 的 slots 有效日期一致时走共享日循环; 否则逐 slot 回退。
    let all_same = effs.iter().all(|e| *e == effs[0]);
    if !all_same {
        return slots
            .iter()
            .enumerate()
            .map(|(i, slot)| {
                if skipped[i] {
                    return (default_result(), default_result());
                }
                let block = slot.insert_axis(Axis(2));
                (
                    bt_single_gap(
                        &block, ret_gap1, ret_sum_gap1, restrict, index, dates, 0, 1,
                        portf_num, &effs[i], open_symbol_counts, ic_only, pre,
                    ),
                    bt_single_gap(
                        &block, ret_gap5, ret_sum_gap5, restrict, index, dates, 0, 5,
                        portf_num, &effs[i], open_symbol_counts, ic_only, pre,
                    ),
                )
            })
            .collect();
    }

    let eff = &effs[0];
    if eff.is_empty() {
        return slots.iter().map(|_| (default_result(), default_result())).collect();
    }
    let date_size = eff.len();

    // 状态: 每 slot × 每 gap
    let mut st1: Vec<GapState> = (0..b)
        .map(|_| GapState::new(date_size, portf_num, n_stocks, ic_only))
        .collect();
    let mut st5: Vec<GapState> = (0..b)
        .map(|_| GapState::new(date_size, portf_num, n_stocks, ic_only))
        .collect();

    let rg1 = ret_gap1.as_slice().unwrap();
    let rs1 = ret_sum_gap1.as_slice().unwrap();
    let rg5 = ret_gap5.as_slice().unwrap();
    let rs5 = ret_sum_gap5.as_slice().unwrap();
    let rst = restrict.as_slice().unwrap();
    let slot_slices: Vec<&[f32]> = slots.iter().map(|s| s.as_slice().unwrap()).collect();

    for (local_t, &raw_eff_idx) in eff.iter().enumerate() {
        let r1 = raw_eff_idx * n_stocks;
        // gap1 held 每日更新为 raw_eff_idx-1; gap5 每 5 日更新
        let h1 = (raw_eff_idx - 1) * n_stocks;
        let update5 = local_t % 5 == 0;
        // 当日行（只读一次, 全 slot 共享）
        let ret1_row = &rg1[r1..r1 + n_stocks];
        let rets1_row = &rs1[r1..r1 + n_stocks];
        let ret5_row = &rg5[r1..r1 + n_stocks];
        let rets5_row = &rs5[r1..r1 + n_stocks];
        for i in 0..b {
            let ss = slot_slices[i];
            st1[i].held_row = raw_eff_idx - 1;
            if update5 {
                st5[i].held_row = raw_eff_idx - 1;
            }
            let h5 = st5[i].held_row * n_stocks;
            step_gap_rows(
                1, local_t, raw_eff_idx, ss, h1, ret1_row, rets1_row, rst,
                dates, n_stocks, portf_num, open_symbol_counts, ic_only,
                &pre.orders_g1, &mut st1[i],
            );
            step_gap_rows(
                5, local_t, raw_eff_idx, ss, h5, ret5_row, rets5_row, rst,
                dates, n_stocks, portf_num, open_symbol_counts, ic_only,
                &pre.orders_g5, &mut st5[i],
            );
        }
    }

    (0..b)
        .map(|i| {
            let s1 = std::mem::replace(
                &mut st1[i],
                GapState::new(0, portf_num, 0, true),
            );
            let s5 = std::mem::replace(
                &mut st5[i],
                GapState::new(0, portf_num, 0, true),
            );
            (
                finish_result(
                    s1.group_returns, s1.ratio_values, s1.ic_dates, s1.ic_values_f64,
                    s1.ic_values_f32, eff, index, portf_num, 1, ic_only,
                ),
                finish_result(
                    s5.group_returns, s5.ratio_values, s5.ic_dates, s5.ic_values_f64,
                    s5.ic_values_f32, eff, index, portf_num, 5, ic_only,
                ),
            )
        })
        .collect()
}

/// 与 step_gap 逐语句一致, 但行以 slice 传入（held/当前行基址外置）。
#[allow(clippy::too_many_arguments)]
fn step_gap_rows(
    gap: usize,
    local_t: usize,
    raw_eff_idx: usize,
    slot: &[f32],
    held_base: usize,
    ret_row: &[f32],
    ret_sum_row: &[f32],
    restrict_flat: &[f32],
    dates: &[i32],
    n_stocks: usize,
    portf_num: usize,
    open_symbol_counts: &[usize],
    ic_only: bool,
    orders: &[Vec<u32>],
    st: &mut GapState,
) {
    let f = &mut st.filter;
    f.signal.clear();
    f.ret.clear();
    f.stock_idx.clear();
    for stock_idx in 0..n_stocks {
        let signal_value = slot[held_base + stock_idx];
        let ret_value = ret_row[stock_idx];
        let rv = restrict_flat[held_base + stock_idx];
        let is_open = rv.is_finite() && rv == 0.0;
        if signal_value.is_finite() && ret_value.is_finite() && is_open {
            f.signal.push(signal_value);
            f.ret.push(ret_value);
            f.stock_idx.push(stock_idx as u32);
        }
    }
    let mut ord_signal: Option<Vec<i64>> = None;
    let mut avg_signal: Option<Vec<f64>> = None;
    if (local_t + 1) % gap == 0 {
        st.gen_id += 1;
        let order = &orders[raw_eff_idx];
        for (pos, &stk) in f.stock_idx.iter().enumerate() {
            st.gen[stk as usize] = st.gen_id;
            st.stamp[stk as usize] = (pos + 1) as u32;
        }
        st.walk_buf.clear();
        st.walk_buf.resize(f.stock_idx.len(), 0);
        let mut counter = 0usize;
        for &stk in order {
            if st.gen[stk as usize] == st.gen_id {
                st.walk_buf[st.stamp[stk as usize] as usize - 1] = counter as i64;
                counter += 1;
            }
        }
        let (xx, avg_now) = rank_both_radix(&f.signal);
        ord_signal = Some(xx);
        avg_signal = Some(avg_now);
        let xx = ord_signal.as_ref().unwrap();
        let n = f.signal.len() as f64;
        let mut diff_sq_sum = 0.0;
        for idx in 0..f.signal.len() {
            let diff = st.walk_buf[idx] - xx[idx];
            diff_sq_sum += (diff * diff) as f64;
        }
        let ic_value = if n < 2.0 {
            f64::NAN
        } else {
            1.0 - 6.0 * diff_sq_sum / (n * (n * n - 1.0))
        };
        st.ic_dates.push(dates[raw_eff_idx]);
        st.ic_values_f64.push(ic_value);
        st.ic_values_f32.push(ic_value as f32);
    }
    let stocks_num = f.signal.len();
    if stocks_num < portf_num {
        return;
    }
    let valid_symbol_num = open_symbol_counts
        .get(raw_eff_idx - 1)
        .copied()
        .unwrap_or(0);
    if valid_symbol_num > 0 {
        st.ratio_values[local_t] = stocks_num as f64 / valid_symbol_num as f64;
    }
    if ic_only {
        return;
    }
    st.group_sums.fill(0.0);
    st.group_counts.fill(0);
    let (_, ranks) = match avg_signal {
        Some(a) => (ord_signal.take().unwrap(), a),
        None => rank_both_radix(&f.signal),
    };
    for idx in 0..stocks_num {
        let pct = ranks[idx] / stocks_num as f64;
        let mut bucket = (pct * portf_num as f64).floor() as usize;
        if bucket >= portf_num {
            bucket = portf_num - 1;
        }
        st.group_sums[bucket] += f.ret[idx] as f64;
        st.group_counts[bucket] += 1;
    }
    for bucket in 0..portf_num {
        st.group_returns[bucket][local_t] = if st.group_counts[bucket] == 0 {
            0.0
        } else {
            st.group_sums[bucket] / st.group_counts[bucket] as f64
        };
    }
}

// ==================== 基准入口 ====================

/// 对一个 slot 分别跑生产复刻与融合版, 校验逐位一致并输出耗时。
#[allow(clippy::too_many_arguments)]
pub fn bench_one_slot(
    tag: &str,
    slot: ArrayView2<'_, f32>,
    shared: &engine::Shared,
    pre: &BtPrecomputed,
    open_counts: &[usize],
) {
    let t = std::time::Instant::now();
    let (p1, p5) = bt_gap1_gap5_prod(
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
        open_counts,
        false,
        pre,
    );
    let t_prod = t.elapsed().as_secs_f64();

    let t = std::time::Instant::now();
    let (f1, f5) = bt_gap1_gap5_fused(
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
        open_counts,
        false,
        pre,
    );
    let t_fused = t.elapsed().as_secs_f64();

    let eq = result_bitwise_eq(&p1, &f1) && result_bitwise_eq(&p5, &f5);
    println!(
        "BTSLOT {tag}: prod={t_prod:.3}s fused={t_fused:.3}s speedup={:.2}x bitwise_eq={eq}",
        t_prod / t_fused
    );
}

pub fn result_bitwise_eq(a: &LegacyBacktestResult, b: &LegacyBacktestResult) -> bool {
    let arr_eq = |x: &[f64], y: &[f64]| {
        x.len() == y.len()
            && x.iter().zip(y.iter()).all(|(u, v)| {
                (u.is_nan() && v.is_nan()) || u.to_bits() == v.to_bits()
            })
    };
    arr_eq(&a.summary, &b.summary)
        && a.ic_dates == b.ic_dates
        && a.ic_values.len() == b.ic_values.len()
        && a.ic_values
            .iter()
            .zip(b.ic_values.iter())
            .all(|(u, v)| (u.is_nan() && v.is_nan()) || u.to_bits() == v.to_bits())
}

pub(crate) fn rank_both_radix_pub(values: &[f32]) -> (Vec<i64>, Vec<f64>) { rank_both_radix(values) }
pub(crate) fn default_result_pub() -> LegacyBacktestResult { default_result() }
pub(crate) fn finish_result_pub(
    ratio_values: Vec<f64>,
    ic_dates: Vec<i32>,
    ic_values_f64: Vec<f64>,
    ic_values_f32: Vec<f32>,
    effective_raw_indices: &[usize],
    gap: usize,
) -> LegacyBacktestResult {
    finish_result(
        Vec::new(),
        ratio_values,
        ic_dates,
        ic_values_f64,
        ic_values_f32,
        effective_raw_indices,
        ndarray::ArrayView1::from(&[0.0f32][..]),
        10,
        gap,
        true,
    )
}
