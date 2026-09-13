// IC-only 回测复刻 (与 tail_v5_pipeline.rs 的 O1 opt 路径语义一致, 只保留 IC 序列)。
use std::collections::HashSet;

#[inline]
fn b_mono_key32(v: f32) -> u32 {
    let v = if v == 0.0 { 0.0 } else { v };
    let bits = v.to_bits();
    if bits >> 31 == 0 { bits ^ 0x8000_0000 } else { !bits }
}

fn radix_sort_u32_keys(keys: &[u32], order: &mut Vec<usize>, tmp: &mut Vec<usize>) {
    let n = order.len();
    if n < 2 { return; }
    tmp.clear();
    tmp.resize(n, 0);
    let mut count = [0usize; 256];
    for shift in (0..32).step_by(8) {
        count.fill(0);
        for &i in order.iter() { count[((keys[i] >> shift) & 0xff) as usize] += 1; }
        let mut acc = 0usize;
        for c in count.iter_mut() { let t = *c; *c = acc; acc += t; }
        for &i in order.iter() {
            let b = ((keys[i] >> shift) & 0xff) as usize;
            tmp[count[b]] = i; count[b] += 1;
        }
        std::mem::swap(order, tmp);
    }
}

fn keyed_order_u32(values: &[f32]) -> Vec<usize> {
    let n = values.len();
    let mut keys: Vec<u32> = vec![0u32; n];
    for (i, &v) in values.iter().enumerate() { keys[i] = b_mono_key32(v); }
    let mut order: Vec<usize> = (0..n).collect();
    let mut tmp: Vec<usize> = Vec::new();
    radix_sort_u32_keys(&keys, &mut order, &mut tmp);
    order
}

fn rank_both_radix(values: &[f32]) -> (Vec<i64>, Vec<f64>) {
    let n = values.len();
    let order = keyed_order_u32(values);
    let mut ordinal = vec![0i64; n];
    let mut avg = vec![f64::NAN; n];
    for (rank, &idx) in order.iter().enumerate() { ordinal[idx] = rank as i64; }
    let mut start = 0usize;
    while start < n {
        let value = values[order[start]];
        let mut end = start + 1;
        while end < n && values[order[end]] == value { end += 1; }
        let avg_rank = (start + 1 + end) as f64 / 2.0;
        for &idx in order[start..end].iter() { avg[idx] = avg_rank; }
        start = end;
    }
    (ordinal, avg)
}

#[derive(Clone, Default)]
pub struct BtPre {
    pub orders_g1: Vec<Vec<u32>>,
    pub orders_g5: Vec<Vec<u32>>,
}

fn build_bt_precomputed(ret_sum_g1: &Array2<f32>, ret_sum_g5: &Array2<f32>) -> BtPre {
    let n_dates = ret_sum_g1.nrows();
    let n_stocks = ret_sum_g1.ncols();
    let mut build = |ret_sum: &Array2<f32>| -> Vec<Vec<u32>> {
        let mut orders = Vec::with_capacity(n_dates);
        for d in 0..n_dates {
            let mut keys: Vec<u32> = Vec::with_capacity(n_stocks);
            for &v in ret_sum.row(d).iter() {
                keys.push(if v.is_nan() { u32::MAX } else { b_mono_key32(v) });
            }
            let mut order: Vec<usize> = (0..n_stocks).collect();
            let mut tmp: Vec<usize> = Vec::new();
            radix_sort_u32_keys(&keys, &mut order, &mut tmp);
            orders.push(order.into_iter().map(|x| x as u32).collect());
        }
        orders
    };
    BtPre { orders_g1: build(ret_sum_g1), orders_g5: build(ret_sum_g5) }
}

fn precompute_open_symbol_counts(restrict: &Array2<f32>) -> Vec<usize> {
    (0..restrict.nrows()).map(|r| restrict.row(r).iter().filter(|&&v| v.is_finite() && v == 0.0).count()).collect()
}

fn has_enough_unique_values(slot: &Array2<f32>, min_unique: usize) -> bool {
    let mut seen = HashSet::<u32>::new();
    for r in 0..slot.nrows().saturating_sub(1) {
        for &v in slot.row(r).iter() {
            if v.is_finite() { seen.insert(v.to_bits()); if seen.len() >= min_unique { return true; } }
        }
    }
    false
}

fn effective_raw_indices_for_slot(slot: &Array2<f32>, dates: &[i32], backtest_start: i32) -> Vec<usize> {
    let n_dates = slot.nrows();
    let n_stocks = slot.ncols();
    let mut out = Vec::new();
    for raw_eff_idx in 1..n_dates {
        if dates[raw_eff_idx] <= backtest_start { continue; }
        let signal_row_idx = raw_eff_idx - 1;
        let mut all_nan = true;
        for s in 0..n_stocks {
            if slot[[signal_row_idx, s]].is_finite() { all_nan = false; break; }
        }
        if !all_nan { out.push(raw_eff_idx); }
    }
    out
}

/// 复刻 legacy_backtest_single_factor_with_effective_opt 的 ic_only 分支。
fn backtest_ic_slot(
    slot: &Array2<f32>,
    ret: &Array2<f32>,
    ret_sum: &Array2<f32>,
    restrict: &Array2<f32>,
    dates: &[i32],
    backtest_start: i32,
    gap: usize,
    pre: &BtPre,
    open_symbol_counts: &[usize],
) -> (Vec<i32>, Vec<f32>) {
    let n_stocks = slot.ncols();
    if slot.nrows() < 2 || !has_enough_unique_values(slot, 10) {
        return (Vec::new(), Vec::new());
    }
    let eff = effective_raw_indices_for_slot(slot, dates, backtest_start);
    if eff.is_empty() { return (Vec::new(), Vec::new()); }
    let orders = if gap == 1 { &pre.orders_g1 } else { &pre.orders_g5 };
    let mut ic_dates = Vec::<i32>::new();
    let mut ic_values = Vec::<f32>::new();
    let mut filtered_signal = Vec::<f32>::with_capacity(n_stocks);
    let mut filtered_ret = Vec::<f32>::with_capacity(n_stocks);
    let mut filtered_stock_idx = Vec::<u32>::with_capacity(n_stocks);
    let mut held_signal_row_idx = eff[0] - 1;
    let mut held_restrict_row_idx = eff[0] - 1;
    let mut gen = vec![0u32; n_stocks];
    let mut stamp = vec![0u32; n_stocks];
    let mut walk_buf = Vec::<i64>::with_capacity(n_stocks);
    let mut gen_id: u32 = 0;
    for (local_t, &raw_eff_idx) in eff.iter().enumerate() {
        if local_t % gap == 0 {
            held_signal_row_idx = raw_eff_idx - 1;
            held_restrict_row_idx = raw_eff_idx - 1;
        }
        filtered_signal.clear();
        filtered_ret.clear();
        filtered_stock_idx.clear();
        for s in 0..n_stocks {
            let signal_value = slot[[held_signal_row_idx, s]];
            let ret_value = ret[[raw_eff_idx, s]];
            let is_open = restrict[[held_restrict_row_idx, s]].is_finite() && restrict[[held_restrict_row_idx, s]] == 0.0;
            if signal_value.is_finite() && ret_value.is_finite() && is_open {
                filtered_signal.push(signal_value);
                filtered_ret.push(ret_value);
                filtered_stock_idx.push(s as u32);
            }
        }
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
            let (xx, _avg) = rank_both_radix(&filtered_signal);
            let n = filtered_signal.len() as f64;
            let mut diff_sq_sum = 0.0;
            for idx in 0..filtered_signal.len() {
                let diff = walk_buf[idx] - xx[idx];
                diff_sq_sum += (diff * diff) as f64;
            }
            let ic_value = if n < 2.0 { f64::NAN } else { 1.0 - 6.0 * diff_sq_sum / (n * (n * n - 1.0)) };
            ic_dates.push(dates[raw_eff_idx]);
            ic_values.push(ic_value as f32);
        }
        let _ = open_symbol_counts.get(raw_eff_idx - 1);
    }
    (ic_dates, ic_values)
}

#[pyclass]
pub struct BtHandle {
    pre: std::sync::Arc<BtPre>,
    open_counts: std::sync::Arc<Vec<usize>>,
}

#[pyfunction]
pub fn sandbox_make_bt_handle(
    ret_sum_g1: PyReadonlyArray2<'_, f32>,
    ret_sum_g5: PyReadonlyArray2<'_, f32>,
    restrict: PyReadonlyArray2<'_, f32>,
) -> PyResult<BtHandle> {
    let r1 = ret_sum_g1.as_array().to_owned();
    let r5 = ret_sum_g5.as_array().to_owned();
    let restr = restrict.as_array().to_owned();
    Ok(BtHandle {
        pre: std::sync::Arc::new(build_bt_precomputed(&r1, &r5)),
        open_counts: std::sync::Arc::new(precompute_open_symbol_counts(&restr)),
    })
}

#[pyfunction]
pub fn sandbox_backtest_ic_batch(
    slots: PyRef<SlotsHandle>,
    ret_g1: PyReadonlyArray2<'_, f32>,
    ret_sum_g1: PyReadonlyArray2<'_, f32>,
    ret_g5: PyReadonlyArray2<'_, f32>,
    ret_sum_g5: PyReadonlyArray2<'_, f32>,
    restrict: PyReadonlyArray2<'_, f32>,
    dates: Vec<i32>,
    backtest_start: i32,
    bt: PyRef<BtHandle>,
) -> PyResult<(Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<Vec<i32>>, Vec<Vec<i32>>)> {
    let r1 = ret_g1.as_array().to_owned();
    let rs1 = ret_sum_g1.as_array().to_owned();
    let r5 = ret_g5.as_array().to_owned();
    let rs5 = ret_sum_g5.as_array().to_owned();
    let restr = restrict.as_array().to_owned();
    let b = slots.slots.len();
    let mut v1s = Vec::with_capacity(b);
    let mut v5s = Vec::with_capacity(b);
    let mut d1s = Vec::with_capacity(b);
    let mut d5s = Vec::with_capacity(b);
    for slot in slots.slots.iter() {
        let (d1, v1) = backtest_ic_slot(slot, &r1, &rs1, &restr, &dates, backtest_start, 1, &bt.pre, &bt.open_counts);
        let (d5, v5) = backtest_ic_slot(slot, &r5, &rs5, &restr, &dates, backtest_start, 5, &bt.pre, &bt.open_counts);
        v1s.push(v1); v5s.push(v5); d1s.push(d1); d5s.push(d5);
    }
    Ok((v1s, v5s, d1s, d5s))
}
