use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use nalgebra::Dyn;
use numpy::PyArray2;

// ---------------------------------------------------------------------------
// precompute: 直接用 barra_raw/industry/restrict 构建 NeutralizeStdShared。
// 与生产 neutralize_std_precompute 的数值/顺序一致 (barra rank, orders, X'X, Cholesky)。
// ---------------------------------------------------------------------------
fn build_shared_from_raw(
    industry: &Array2<f64>,
    restrict: &Array2<f32>,
    barra_raw: &ArrayView3<f64>,
) -> Result<NeutralizeStdShared, String> {
    let (n_dates, n_stocks) = industry.dim();
    if restrict.dim() != (n_dates, n_stocks) {
        return Err("restrict 形状不匹配".into());
    }
    if barra_raw.dim() != (n_dates, n_stocks, 10) {
        return Err(format!("barra_raw 形状 {:?} != ({},{},10)", barra_raw.dim(), n_dates, n_stocks));
    }
    // barra 10 风格 rank pct (生产顺序: 先复制原始值再逐列 rank_pct_all)
    let mut barra_ranked: Vec<Array2<f64>> = Vec::with_capacity(10);
    for c in 0..10 {
        let mut m = Array2::<f64>::from_elem((n_dates, n_stocks), f64::NAN);
        for i in 0..n_dates {
            for j in 0..n_stocks {
                m[[i, j]] = barra_raw[[i, j, c]];
            }
        }
        rank_pct_all(&mut m);
        barra_ranked.push(m);
    }
    // size_ranked = rank_pct(raw value_2) = barra_ranked[2] (同一函数、同一输入)
    let size_ranked = barra_ranked[2].clone();

    let restrict_f64 = Array2::<f64>::from_shape_vec(
        (n_dates, n_stocks),
        restrict.iter().map(|&v| v as f64).collect(),
    )
    .map_err(|e| e.to_string())?;
    let ind1 = industry.map(|&v| (v / 10000.0).floor());
    let ind2 = industry.map(|&v| (v / 100.0).floor());
    let zeros = Array2::<f64>::zeros((n_dates, n_stocks));
    let ind1_mask = ind1.map(|&v| if v.is_nan() { 0.0 } else { 1.0 });

    // ---- 预计算 5 张行业分组排序 ----
    let ind0 = ind1.map(|&v| if v.is_nan() { 0.0 } else { 1.0 });
    let levels = [&ind2, &ind1, &ind0, &ind2, &ind1];
    let mut orders: Vec<Array2<usize>> = Vec::with_capacity(5);
    for codes in levels {
        let mut ord = Array2::<usize>::zeros((n_dates, n_stocks));
        for date_idx in 0..n_dates {
            let mut order: Vec<usize> = (0..n_stocks).collect();
            let mut keys: Vec<u64> = codes.row(date_idx).iter().map(|&v| mono_key(v)).collect();
            let mut tmp: Vec<usize> = Vec::new();
            radix_sort_order(&keys, &mut order, &mut tmp);
            for j in 0..n_stocks {
                ord[[date_idx, j]] = order[j];
            }
        }
        orders.push(ord);
    }

    // ---- 逐日 X'X / Cholesky / 连续 X (严格照抄生产 neutralize_std_precompute) ----
    let mut per_date: Vec<(usize, Vec<u32>, Vec<i32>, Vec<f64>)> = Vec::with_capacity(n_dates);
    let mut chols: Vec<Option<nalgebra::Cholesky<f64, nalgebra::Dyn>>> = Vec::with_capacity(n_dates);
    let mut xdays: Vec<Vec<f64>> = Vec::with_capacity(n_dates);
    for date_idx in 0..n_dates {
        let mut ind_codes: Vec<f64> = Vec::new();
        for j in 0..n_stocks {
            let c = ind1[[date_idx, j]];
            if !c.is_nan() && !ind_codes.contains(&c) {
                ind_codes.push(c);
            }
        }
        ind_codes.sort_by(cmp_f64);
        let p = 10 + ind_codes.len();
        let mut valid_idx: Vec<u32> = Vec::new();
        let mut valid_cols: Vec<i32> = Vec::new();
        for j in 0..n_stocks {
            let ok = restrict_f64[[date_idx, j]].is_finite()
                && restrict_f64[[date_idx, j]] == 0.0
                && barra_ranked.iter().all(|b| b[[date_idx, j]].is_finite());
            if ok {
                valid_idx.push(j as u32);
                let c_ind = ind1[[date_idx, j]];
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
            chols.push(None);
            xdays.push(Vec::new());
            continue;
        }
        let mut xtx = vec![0.0f64; p * p];
        let k = 10usize;
        let mut xd = vec![0.0f64; n_valid * k];
        for (pos, &j) in valid_idx.iter().enumerate() {
            let ji = j as usize;
            for c in 0..k {
                let b = barra_ranked[c][[date_idx, ji]];
                xd[pos * k + c] = b;
                xtx[c * p + c] += b * b;
                for c2 in (c + 1)..k {
                    let v = b * barra_ranked[c2][[date_idx, ji]];
                    xtx[c * p + c2] += v;
                    xtx[c2 * p + c] += v;
                }
            }
            let ic = valid_cols[pos];
            if ic >= 0 {
                let col = 10 + ic as usize;
                xtx[col * p + col] += 1.0;
                for c in 0..k {
                    let b = barra_ranked[c][[date_idx, ji]];
                    xtx[c * p + col] += b;
                    xtx[col * p + c] += b;
                }
            }
        }
        chols.push(nalgebra::Cholesky::new(DMatrix::from_row_slice(p, p, &xtx)));
        per_date.push((p, valid_idx, valid_cols, xtx));
        xdays.push(xd);
    }

    Ok(NeutralizeStdShared {
        industry: industry.clone(),
        restrict_f64,
        ind1,
        ind2,
        zeros,
        ind1_mask,
        barra_ranked,
        size_ranked,
        orders,
        per_date,
        chols,
        xdays,
    })
}

// ---------------------------------------------------------------------------
// fast path helpers
// ---------------------------------------------------------------------------

/// 仅对 restrict==0 的位置做 rank_pct (f32 -> f64), 语义等于生产链路在
/// "free 位置全部有限且无填充副作用" 条件下的 x1。
fn rank_pct_free_from_f32(
    in_row: &[f32],
    restrict_row: &[f64],
    out_row: &mut [f64],
    idxs: &mut Vec<usize>,
    tmp: &mut Vec<usize>,
    keys: &mut Vec<u32>,
) {
    let n = in_row.len();
    for r in out_row.iter_mut() {
        *r = f64::NAN;
    }
    idxs.clear();
    for i in 0..n {
        if restrict_row[i] == 0.0 && !in_row[i].is_nan() {
            idxs.push(i);
            keys[i] = mono_key32(in_row[i]);
        }
    }
    let n_valid = idxs.len();
    if n_valid == 0 {
        return;
    }
    radix_sort_order32(keys, idxs, tmp);
    let mut i = 0usize;
    while i < n_valid {
        let mut j = i;
        while j + 1 < n_valid && in_row[idxs[j + 1]] == in_row[idxs[i]] {
            j += 1;
        }
        let avg_rank = ((i + 1) + (j + 1)) as f64 / 2.0;
        let pct = avg_rank / n_valid as f64;
        for item in &idxs[i..=j] {
            out_row[*item] = pct;
        }
        i = j + 1;
    }
}

/// 该日是否满足 fast path 条件: 所有 restrict==0 的位置 slot/size/ind1 都有限。
fn fast_day_ok(slot_row: &[f32], restrict_row: &[f64], size_row: &[f64], ind1_row: &[f64]) -> bool {
    for j in 0..slot_row.len() {
        if restrict_row[j] == 0.0 {
            if !slot_row[j].is_finite() || !size_row[j].is_finite() || !ind1_row[j].is_finite() {
                return false;
            }
        }
    }
    true
}

/// 单 slot 的 fast/hybrid 中性化, 输出 f32 residual (与生产 C' 语义一致)。
/// force_fast=true 时无条件走快路径 (仅用于速度上界, 不保证等价)。
fn neutralize_slot_hybrid(
    slot: ArrayView2<'_, f32>,
    shared: &NeutralizeStdShared,
    force_fast: bool,
) -> Result<Array2<f32>, String> {
    let (t, n) = slot.dim();
    if shared.industry.dim() != (t, n) || shared.restrict_f64.dim() != (t, n) {
        return Err("形状不匹配".into());
    }
    let mut out = Array2::<f32>::from_elem((t, n), f32::NAN);
    // 复用缓冲
    let mut pct: Vec<f64> = vec![0.0; n];
    let mut filled: Vec<f64> = vec![0.0; n];
    let mut idxs: Vec<usize> = Vec::with_capacity(n);
    let mut tmp: Vec<usize> = Vec::with_capacity(n);
    let mut keys32: Vec<u32> = vec![0u32; n];
    let mut keys64: Vec<u64> = vec![0u64; n];
    let mut ranks64: Vec<f64> = Vec::with_capacity(n);
    let mut ys: Vec<f64> = Vec::with_capacity(n);
    let mut bs: Vec<f64> = Vec::with_capacity(n);
    let mut obs: Vec<bool> = Vec::with_capacity(n);
    let mut sv: Vec<f64> = Vec::with_capacity(n);
    let mut nan_mask: Vec<bool> = Vec::with_capacity(n);
    let mut y_buf: Vec<f64> = Vec::with_capacity(n);
    let mut xty: Vec<f64> = Vec::with_capacity(64);
    let mut ind0_row: Vec<f64> = vec![0.0; n];
    let mut fast_days = 0u64;
    let mut slow_days = 0u64;

    for idx in 0..t {
        let slot_row = slot.row(idx);
        let slot_r = slot_row.as_slice().unwrap();
        let restrict_v = shared.restrict_f64.row(idx);
        let restrict_r = restrict_v.as_slice().unwrap();
        let ind2_v = shared.ind2.row(idx);
        let ind1_v = shared.ind1.row(idx);
        let size_v = shared.size_ranked.row(idx);
        let zeros_v = shared.zeros.row(idx);
        let mask_v = shared.ind1_mask.row(idx);
        let o0_v = shared.orders[0].row(idx);
        let o1_v = shared.orders[1].row(idx);
        let o2_v = shared.orders[2].row(idx);
        let o3_v = shared.orders[3].row(idx);
        let o4_v = shared.orders[4].row(idx);
        let ind2_r = ind2_v.as_slice().unwrap();
        let ind1_r = ind1_v.as_slice().unwrap();
        let size_r = size_v.as_slice().unwrap();
        let zeros_r = zeros_v.as_slice().unwrap();
        let mask_r = mask_v.as_slice().unwrap();
        let o0 = o0_v.as_slice().unwrap();
        let o1 = o1_v.as_slice().unwrap();
        let o2 = o2_v.as_slice().unwrap();
        let o3 = o3_v.as_slice().unwrap();
        let o4 = o4_v.as_slice().unwrap();
        ind0_row.clear();
        ind0_row.extend(ind1_r.iter().map(|&v| if v.is_nan() { 0.0 } else { 1.0 }));

        let is_fast = force_fast || fast_day_ok(slot_r, restrict_r, size_r, ind1_r);
        if is_fast {
            fast_days += 1;
            rank_pct_free_from_f32(slot_r, restrict_r, &mut pct, &mut idxs, &mut tmp, &mut keys32);
            let mut out_row = out.row_mut(idx);
            let out_r = out_row.as_slice_mut().unwrap();
            // pct 已对 free 位置给出 rank_pct, 其余 NaN
            let _ = ols_day_row_fast(&pct, shared, idx, &mut y_buf, &mut xty, out_r);
        } else {
            slow_days += 1;
            rank_pct_row_from_f32_in(slot_r, &mut pct, &mut idxs, &mut tmp, &mut keys32);
            fill_ind_reg_row(
                &mut pct,
                [ind2_r, ind1_r, &ind0_row],
                size_r,
                [o0, o1, o2],
                &mut ys,
                &mut bs,
                &mut obs,
            );
            for (j, &v) in ind1_r.iter().enumerate() {
                if v.is_nan() {
                    pct[j] = f64::NAN;
                }
            }
            filled.copy_from_slice(&pct);
            nan_mask.clear();
            for &v in filled.iter() {
                nan_mask.push(v.is_nan());
            }
            if nan_mask.iter().any(|&x| x) {
                median_fill_level_row(&mut filled, ind2_r, None, o3, &nan_mask, &mut sv);
            }
            nan_mask.clear();
            for &v in filled.iter() {
                nan_mask.push(v.is_nan());
            }
            if nan_mask.iter().any(|&x| x) {
                median_fill_level_row(&mut filled, ind1_r, None, o4, &nan_mask, &mut sv);
            }
            nan_mask.clear();
            for &v in filled.iter() {
                nan_mask.push(v.is_nan());
            }
            if nan_mask.iter().any(|&x| x) {
                median_fill_level_row(&mut filled, zeros_r, Some(mask_r), o0, &nan_mask, &mut sv);
            }
            for (j, &v) in restrict_r.iter().enumerate() {
                if v != 0.0 {
                    filled[j] = f64::NAN;
                }
            }
            rank_pct_row_f64_in_place(&mut filled, &mut ranks64, &mut idxs, &mut tmp, &mut keys64);
            let mut out_row = out.row_mut(idx);
            let out_r = out_row.as_slice_mut().unwrap();
            let _ = ols_day_row_fast(&filled, shared, idx, &mut y_buf, &mut xty, out_r);
        }
    }
    // 把统计值存到返回对象的属性? 这里用 eprintln 一次
    // eprintln!("[sandbox_neu_fast] fast_days={} slow_days={}", fast_days, slow_days);
    Ok(out)
}

// ---------------------------------------------------------------------------
// pyo3 handles
// ---------------------------------------------------------------------------

#[pyclass]
pub struct SharedHandle {
    inner: Arc<NeutralizeStdShared>,
    inv_orders: Arc<Vec<Array2<i32>>>,
}

#[pyclass]
pub struct SlotsHandle {
    slots: Vec<Array2<f32>>,
}

#[pyfunction]
#[pyo3(signature = (industry, restrict, barra_raw))]
pub fn sandbox_precompute<'py>(
    py: Python<'py>,
    industry: PyReadonlyArray2<'py, f64>,
    restrict: PyReadonlyArray2<'py, f32>,
    barra_raw: PyReadonlyArray3<'py, f64>,
) -> PyResult<SharedHandle> {
    let industry = industry.as_array().to_owned();
    let restrict = restrict.as_array().to_owned();
    let barra_raw = barra_raw.as_array().to_owned();
    let shared = build_shared_from_raw(&industry, &restrict, &barra_raw.view())
        .map_err(PyValueError::new_err)?;
    // 每个 order 的逆映射: inv[date, order[date,pos]] = pos
    let (t, n) = industry.dim();
    let mut inv_orders: Vec<Array2<i32>> = Vec::with_capacity(shared.orders.len());
    for ord in shared.orders.iter() {
        let mut inv = Array2::<i32>::zeros((t, n));
        for d in 0..t {
            let ord_row = ord.row(d);
            let mut inv_row = inv.row_mut(d);
            for pos in 0..n {
                inv_row[ord_row[pos]] = pos as i32;
            }
        }
        inv_orders.push(inv);
    }
    Ok(SharedHandle {
        inner: Arc::new(shared),
        inv_orders: Arc::new(inv_orders),
    })
}

#[pyfunction]
pub fn sandbox_make_slots(slots: Vec<PyReadonlyArray2<'_, f32>>) -> PyResult<SlotsHandle> {
    let mut owned = Vec::with_capacity(slots.len());
    for s in slots {
        owned.push(s.as_array().to_owned());
    }
    Ok(SlotsHandle { slots: owned })
}

#[pyfunction]
pub fn sandbox_baseline_batch(
    py: Python<'_>,
    slots: PyRef<SlotsHandle>,
    shared: PyRef<SharedHandle>,
) -> PyResult<Vec<Py<PyArray2<f32>>>> {
    let views: Vec<ArrayView2<'_, f32>> = slots.slots.iter().map(|s| s.view()).collect();
    let out = neutralize_std_slots_f32_v2_resid_batch(&views, &shared.inner, true)
        .map_err(PyValueError::new_err)?;
    Ok(out
        .into_iter()
        .map(|a| a.into_pyarray(py).to_owned())
        .collect())
}

#[pyfunction]
#[pyo3(signature = (slots, shared, force_fast=false))]
pub fn sandbox_fast_batch(
    py: Python<'_>,
    slots: PyRef<SlotsHandle>,
    shared: PyRef<SharedHandle>,
    force_fast: bool,
) -> PyResult<Vec<Py<PyArray2<f32>>>> {
    let views: Vec<ArrayView2<'_, f32>> = slots.slots.iter().map(|s| s.view()).collect();
    let mut outs = Vec::with_capacity(views.len());
    for v in views.iter() {
        outs.push(neutralize_slot_hybrid(v.clone(), &shared.inner, force_fast).map_err(PyValueError::new_err)?);
    }
    let out = outs;
    Ok(out
        .into_iter()
        .map(|a| a.into_pyarray(py).to_owned())
        .collect())
}


// ---------------------------------------------------------------------------
// 分阶段计时 (仅 benchmark 用)
// ---------------------------------------------------------------------------

#[pyfunction]
pub fn sandbox_baseline_profile(
    slots: PyRef<SlotsHandle>,
    shared: PyRef<SharedHandle>,
) -> PyResult<(Vec<f64>, Vec<u64>)> {
    let slots_ref = &slots.slots;
    let base = &*shared.inner;
    let (t, n) = slots_ref[0].dim();
    let b = slots_ref.len();
    let mut st = [0u64; 9];
    let mut cnt = [0u64; 9];
    let mut pct: Vec<Vec<f64>> = vec![vec![0.0; n]; b];
    let mut filled: Vec<Vec<f64>> = vec![vec![0.0; n]; b];
    let mut keys32 = vec![0u32; n];
    let mut idxs: Vec<usize> = Vec::with_capacity(n);
    let mut tmp: Vec<usize> = Vec::with_capacity(n);
    let mut ranks64: Vec<f64> = Vec::with_capacity(n);
    let mut keys64: Vec<u64> = vec![0u64; n];
    let mut ys: Vec<f64> = Vec::with_capacity(n);
    let mut bs: Vec<f64> = Vec::with_capacity(n);
    let mut obs: Vec<bool> = Vec::with_capacity(n);
    let mut sv: Vec<f64> = Vec::with_capacity(n);
    let mut nan_mask: Vec<bool> = Vec::with_capacity(n);
    let mut y_buf: Vec<f64> = Vec::with_capacity(n);
    let mut xty: Vec<f64> = Vec::with_capacity(64);
    let mut ind0_row = Vec::<f64>::with_capacity(n);
    for idx in 0..t {
        let (ind2_v, ind1_v, size_v) = (base.ind2.row(idx), base.ind1.row(idx), base.size_ranked.row(idx));
        let (zeros_v, mask_v, restrict_v) = (base.zeros.row(idx), base.ind1_mask.row(idx), base.restrict_f64.row(idx));
        let (o0_v, o1_v, o2_v) = (base.orders[0].row(idx), base.orders[1].row(idx), base.orders[2].row(idx));
        let (o3_v, o4_v) = (base.orders[3].row(idx), base.orders[4].row(idx));
        let ind2_r = ind2_v.as_slice().unwrap();
        let ind1_r = ind1_v.as_slice().unwrap();
        let size_r = size_v.as_slice().unwrap();
        let zeros_r = zeros_v.as_slice().unwrap();
        let mask_r = mask_v.as_slice().unwrap();
        let restrict_r = restrict_v.as_slice().unwrap();
        let o0 = o0_v.as_slice().unwrap();
        let o1 = o1_v.as_slice().unwrap();
        let o2 = o2_v.as_slice().unwrap();
        let o3 = o3_v.as_slice().unwrap();
        let o4 = o4_v.as_slice().unwrap();
        ind0_row.clear();
        ind0_row.extend(ind1_r.iter().map(|&v| if v.is_nan() { 0.0 } else { 1.0 }));
        for (f, slot) in slots_ref.iter().enumerate() {
            let slot_row = slot.row(idx);
            let t0 = std::time::Instant::now();
            rank_pct_row_from_f32_in(slot_row.as_slice().unwrap(), &mut pct[f], &mut idxs, &mut tmp, &mut keys32);
            st[1] += t0.elapsed().as_nanos() as u64; cnt[1]+=1;
            let t0 = std::time::Instant::now();
            fill_ind_reg_row(&mut pct[f], [ind2_r, ind1_r, &ind0_row], size_r, [o0, o1, o2], &mut ys, &mut bs, &mut obs);
            st[2] += t0.elapsed().as_nanos() as u64; cnt[2]+=1;
            let t0 = std::time::Instant::now();
            for (j, &v) in ind1_r.iter().enumerate() { if v.is_nan() { pct[f][j] = f64::NAN; } }
            st[3] += t0.elapsed().as_nanos() as u64; cnt[3]+=1;
            filled[f].copy_from_slice(&pct[f]);
        }
        for f in 0..b {
            let t0 = std::time::Instant::now();
            nan_mask.clear();
            for &v in filled[f].iter() { nan_mask.push(v.is_nan()); }
            if nan_mask.iter().any(|&x| x) { median_fill_level_row(&mut filled[f], ind2_r, None, o3, &nan_mask, &mut sv); }
            nan_mask.clear();
            for &v in filled[f].iter() { nan_mask.push(v.is_nan()); }
            if nan_mask.iter().any(|&x| x) { median_fill_level_row(&mut filled[f], ind1_r, None, o4, &nan_mask, &mut sv); }
            nan_mask.clear();
            for &v in filled[f].iter() { nan_mask.push(v.is_nan()); }
            if nan_mask.iter().any(|&x| x) { median_fill_level_row(&mut filled[f], zeros_r, Some(mask_r), o0, &nan_mask, &mut sv); }
            st[4] += t0.elapsed().as_nanos() as u64; cnt[4]+=1;
            let t0 = std::time::Instant::now();
            for (j, &v) in restrict_r.iter().enumerate() { if v != 0.0 { filled[f][j] = f64::NAN; } }
            rank_pct_row_f64_in_place(&mut filled[f], &mut ranks64, &mut idxs, &mut tmp, &mut keys64);
            st[5] += t0.elapsed().as_nanos() as u64; cnt[5]+=1;
            let mut out_row = Array2::<f32>::from_elem((1,n), f32::NAN);
            let out_r = out_row.as_slice_mut().unwrap();
            let t0 = std::time::Instant::now();
            let _ = ols_day_row_fast(&filled[f], base, idx, &mut y_buf, &mut xty, out_r);
            st[6] += t0.elapsed().as_nanos() as u64; cnt[6]+=1;
        }
    }
    Ok((st.iter().map(|&x| x as f64/1e6).collect(), cnt.to_vec()))
}

#[pyfunction]
#[pyo3(signature = (slots, shared, force_fast=false))]
pub fn sandbox_fast_profile(
    slots: PyRef<SlotsHandle>,
    shared: PyRef<SharedHandle>,
    force_fast: bool,
) -> PyResult<(Vec<f64>, Vec<u64>)> {
    let base = &*shared.inner;
    let (t, n) = slots.slots[0].dim();
    let mut st = [0u64; 5];
    let mut cnt = [0u64; 5];
    let mut pct: Vec<f64> = vec![0.0; n];
    let mut idxs: Vec<usize> = Vec::with_capacity(n);
    let mut tmp: Vec<usize> = Vec::with_capacity(n);
    let mut keys32: Vec<u32> = vec![0u32; n];
    let mut y_buf: Vec<f64> = Vec::with_capacity(n);
    let mut xty: Vec<f64> = Vec::with_capacity(64);
    for slot in slots.slots.iter() {
        for idx in 0..t {
            let slot_row = slot.row(idx);
            let restrict_v = base.restrict_f64.row(idx);
            let size_v = base.size_ranked.row(idx);
            let ind1_v = base.ind1.row(idx);
            let restrict_r = restrict_v.as_slice().unwrap();
            let size_r = size_v.as_slice().unwrap();
            let ind1_r = ind1_v.as_slice().unwrap();
            let ok = force_fast || fast_day_ok(slot_row.as_slice().unwrap(), restrict_r, size_r, ind1_r);
            if ok {
                let t0 = std::time::Instant::now();
                rank_pct_free_from_f32(slot_row.as_slice().unwrap(), restrict_r, &mut pct, &mut idxs, &mut tmp, &mut keys32);
                st[1]+=t0.elapsed().as_nanos() as u64; cnt[1]+=1;
                let t0 = std::time::Instant::now();
                let mut out_row = Array2::<f32>::from_elem((1,n), f32::NAN);
                let _ = ols_day_row_fast(&pct, base, idx, &mut y_buf, &mut xty, out_row.as_slice_mut().unwrap());
                st[2]+=t0.elapsed().as_nanos() as u64; cnt[2]+=1;
            } else {
                st[3]+=1; cnt[3]+=1;
            }
        }
    }
    Ok((st.iter().map(|&x| x as f64/1e6).collect(), cnt.to_vec()))
}

#[pyfunction]
pub fn sandbox_fast_profile_full(
    slots: PyRef<SlotsHandle>,
    shared: PyRef<SharedHandle>,
) -> PyResult<(Vec<f64>, Vec<u64>)> {
    let base = &*shared.inner;
    let (t, n) = slots.slots[0].dim();
    let mut st = [0u64; 10];
    let mut cnt = [0u64; 10];
    let mut pct: Vec<f64> = vec![0.0; n];
    let mut filled: Vec<f64> = vec![0.0; n];
    let mut idxs: Vec<usize> = Vec::with_capacity(n);
    let mut tmp: Vec<usize> = Vec::with_capacity(n);
    let mut keys32: Vec<u32> = vec![0u32; n];
    let mut keys64: Vec<u64> = vec![0u64; n];
    let mut ranks64: Vec<f64> = Vec::with_capacity(n);
    let mut ys: Vec<f64> = Vec::with_capacity(n);
    let mut bs: Vec<f64> = Vec::with_capacity(n);
    let mut obs: Vec<bool> = Vec::with_capacity(n);
    let mut sv: Vec<f64> = Vec::with_capacity(n);
    let mut nan_mask: Vec<bool> = Vec::with_capacity(n);
    let mut y_buf: Vec<f64> = Vec::with_capacity(n);
    let mut xty: Vec<f64> = Vec::with_capacity(64);
    let mut ind0_row: Vec<f64> = vec![0.0; n];
    for slot in slots.slots.iter() {
        for idx in 0..t {
            let slot_row = slot.row(idx);
            let slot_r = slot_row.as_slice().unwrap();
            let ind2_v = base.ind2.row(idx); let ind1_v = base.ind1.row(idx); let size_v = base.size_ranked.row(idx);
            let zeros_v = base.zeros.row(idx); let mask_v = base.ind1_mask.row(idx); let restrict_v = base.restrict_f64.row(idx);
            let o0_v = base.orders[0].row(idx); let o1_v = base.orders[1].row(idx); let o2_v = base.orders[2].row(idx);
            let o3_v = base.orders[3].row(idx); let o4_v = base.orders[4].row(idx);
            let ind2_r=ind2_v.as_slice().unwrap(); let ind1_r=ind1_v.as_slice().unwrap(); let size_r=size_v.as_slice().unwrap();
            let zeros_r=zeros_v.as_slice().unwrap(); let mask_r=mask_v.as_slice().unwrap(); let restrict_r=restrict_v.as_slice().unwrap();
            let o0=o0_v.as_slice().unwrap(); let o1=o1_v.as_slice().unwrap(); let o2=o2_v.as_slice().unwrap();
            let o3=o3_v.as_slice().unwrap(); let o4=o4_v.as_slice().unwrap();
            ind0_row.clear();
            ind0_row.extend(ind1_r.iter().map(|&v| if v.is_nan() { 0.0 } else { 1.0 }));
            if fast_day_ok(slot_r, restrict_r, size_r, ind1_r) {
                let t0=std::time::Instant::now();
                rank_pct_free_from_f32(slot_r, restrict_r, &mut pct, &mut idxs, &mut tmp, &mut keys32);
                st[1]+=t0.elapsed().as_nanos() as u64; cnt[1]+=1;
                let t0=std::time::Instant::now();
                let mut out_row = Array2::<f32>::from_elem((1,n), f32::NAN);
                let _ = ols_day_row_fast(&pct, base, idx, &mut y_buf, &mut xty, out_row.as_slice_mut().unwrap());
                st[2]+=t0.elapsed().as_nanos() as u64; cnt[2]+=1;
            } else {
                let t0=std::time::Instant::now();
                rank_pct_row_from_f32_in(slot_r, &mut pct, &mut idxs, &mut tmp, &mut keys32);
                st[3]+=t0.elapsed().as_nanos() as u64; cnt[3]+=1;
                let t0=std::time::Instant::now();
                fill_ind_reg_row(&mut pct, [ind2_r, ind1_r, &ind0_row], size_r, [o0,o1,o2], &mut ys,&mut bs,&mut obs);
                for (j,&v) in ind1_r.iter().enumerate(){ if v.is_nan(){pct[j]=f64::NAN;} }
                st[4]+=t0.elapsed().as_nanos() as u64; cnt[4]+=1;
                filled.copy_from_slice(&pct);
                let t0=std::time::Instant::now();
                nan_mask.clear(); for &v in filled.iter(){nan_mask.push(v.is_nan());}
                if nan_mask.iter().any(|&x|x){median_fill_level_row(&mut filled,ind2_r,None,o3,&nan_mask,&mut sv);}
                nan_mask.clear(); for &v in filled.iter(){nan_mask.push(v.is_nan());}
                if nan_mask.iter().any(|&x|x){median_fill_level_row(&mut filled,ind1_r,None,o4,&nan_mask,&mut sv);}
                nan_mask.clear(); for &v in filled.iter(){nan_mask.push(v.is_nan());}
                if nan_mask.iter().any(|&x|x){median_fill_level_row(&mut filled,zeros_r,Some(mask_r),o0,&nan_mask,&mut sv);}
                st[5]+=t0.elapsed().as_nanos() as u64; cnt[5]+=1;
                let t0=std::time::Instant::now();
                for (j,&v) in restrict_r.iter().enumerate(){ if v!=0.0{filled[j]=f64::NAN;} }
                rank_pct_row_f64_in_place(&mut filled,&mut ranks64,&mut idxs,&mut tmp,&mut keys64);
                st[6]+=t0.elapsed().as_nanos() as u64; cnt[6]+=1;
                let t0=std::time::Instant::now();
                let mut out_row = Array2::<f32>::from_elem((1,n), f32::NAN);
                let _ = ols_day_row_fast(&filled, base, idx, &mut y_buf, &mut xty, out_row.as_slice_mut().unwrap());
                st[7]+=t0.elapsed().as_nanos() as u64; cnt[7]+=1;
            }
        }
    }
    Ok((st.iter().map(|&x| x as f64/1e6).collect(), cnt.to_vec()))
}

/// 日期外层、slot 内层批处理版 hybrid (共享当日行业/size/orders 行, 改善 cache)。
fn neutralize_slots_hybrid_batched(
    slots: &[Array2<f32>],
    shared: &NeutralizeStdShared,
    force_fast: bool,
) -> Result<Vec<Array2<f32>>, String> {
    let b = slots.len();
    if b == 0 { return Ok(Vec::new()); }
    let (t, n) = slots[0].dim();
    if shared.industry.dim() != (t, n) { return Err("形状不匹配".into()); }
    let mut outs: Vec<Array2<f32>> = (0..b).map(|_| Array2::<f32>::from_elem((t,n), f32::NAN)).collect();
    let mut pct: Vec<f64> = vec![0.0; n];
    let mut filled: Vec<f64> = vec![0.0; n];
    let mut idxs: Vec<usize> = Vec::with_capacity(n);
    let mut tmp: Vec<usize> = Vec::with_capacity(n);
    let mut keys32: Vec<u32> = vec![0u32; n];
    let mut keys64: Vec<u64> = vec![0u64; n];
    let mut ranks64: Vec<f64> = Vec::with_capacity(n);
    let mut ys: Vec<f64> = Vec::with_capacity(n);
    let mut bs: Vec<f64> = Vec::with_capacity(n);
    let mut obs: Vec<bool> = Vec::with_capacity(n);
    let mut sv: Vec<f64> = Vec::with_capacity(n);
    let mut nan_mask: Vec<bool> = Vec::with_capacity(n);
    let mut y_buf: Vec<f64> = Vec::with_capacity(n);
    let mut xty: Vec<f64> = Vec::with_capacity(64);
    let mut ind0_row: Vec<f64> = vec![0.0; n];
    for idx in 0..t {
        let ind2_v = shared.ind2.row(idx); let ind1_v = shared.ind1.row(idx); let size_v = shared.size_ranked.row(idx);
        let zeros_v = shared.zeros.row(idx); let mask_v = shared.ind1_mask.row(idx); let restrict_v = shared.restrict_f64.row(idx);
        let o0_v = shared.orders[0].row(idx); let o1_v = shared.orders[1].row(idx); let o2_v = shared.orders[2].row(idx);
        let o3_v = shared.orders[3].row(idx); let o4_v = shared.orders[4].row(idx);
        let ind2_r=ind2_v.as_slice().unwrap(); let ind1_r=ind1_v.as_slice().unwrap(); let size_r=size_v.as_slice().unwrap();
        let zeros_r=zeros_v.as_slice().unwrap(); let mask_r=mask_v.as_slice().unwrap(); let restrict_r=restrict_v.as_slice().unwrap();
        let o0=o0_v.as_slice().unwrap(); let o1=o1_v.as_slice().unwrap(); let o2=o2_v.as_slice().unwrap();
        let o3=o3_v.as_slice().unwrap(); let o4=o4_v.as_slice().unwrap();
        ind0_row.clear();
        ind0_row.extend(ind1_r.iter().map(|&v| if v.is_nan() { 0.0 } else { 1.0 }));
        for f in 0..b {
            let slot_row = slots[f].row(idx);
            let slot_r = slot_row.as_slice().unwrap();
            let mut out_row = outs[f].row_mut(idx);
            let out_r = out_row.as_slice_mut().unwrap();
            if force_fast || fast_day_ok(slot_r, restrict_r, size_r, ind1_r) {
                rank_pct_free_from_f32(slot_r, restrict_r, &mut pct, &mut idxs, &mut tmp, &mut keys32);
                let _ = ols_day_row_fast(&pct, shared, idx, &mut y_buf, &mut xty, out_r);
            } else {
                rank_pct_row_from_f32_in(slot_r, &mut pct, &mut idxs, &mut tmp, &mut keys32);
                fill_ind_reg_row(&mut pct, [ind2_r, ind1_r, &ind0_row], size_r, [o0,o1,o2], &mut ys,&mut bs,&mut obs);
                for (j,&v) in ind1_r.iter().enumerate(){ if v.is_nan(){pct[j]=f64::NAN;} }
                filled.copy_from_slice(&pct);
                nan_mask.clear(); for &v in filled.iter(){nan_mask.push(v.is_nan());}
                if nan_mask.iter().any(|&x|x){median_fill_level_row(&mut filled,ind2_r,None,o3,&nan_mask,&mut sv);}
                nan_mask.clear(); for &v in filled.iter(){nan_mask.push(v.is_nan());}
                if nan_mask.iter().any(|&x|x){median_fill_level_row(&mut filled,ind1_r,None,o4,&nan_mask,&mut sv);}
                nan_mask.clear(); for &v in filled.iter(){nan_mask.push(v.is_nan());}
                if nan_mask.iter().any(|&x|x){median_fill_level_row(&mut filled,zeros_r,Some(mask_r),o0,&nan_mask,&mut sv);}
                for (j,&v) in restrict_r.iter().enumerate(){ if v!=0.0{filled[j]=f64::NAN;} }
                rank_pct_row_f64_in_place(&mut filled,&mut ranks64,&mut idxs,&mut tmp,&mut keys64);
                let _ = ols_day_row_fast(&filled, shared, idx, &mut y_buf, &mut xty, out_r);
            }
        }
    }
    Ok(outs)
}

#[pyfunction]
#[pyo3(signature = (slots, shared, force_fast=false))]
pub fn sandbox_fast_batch_batched(
    py: Python<'_>,
    slots: PyRef<SlotsHandle>,
    shared: PyRef<SharedHandle>,
    force_fast: bool,
) -> PyResult<Vec<Py<PyArray2<f32>>>> {
    let out = neutralize_slots_hybrid_batched(&slots.slots, &shared.inner, force_fast)
        .map_err(PyValueError::new_err)?;
    Ok(out.into_iter().map(|a| a.into_pyarray(py).to_owned()).collect())
}

// ===========================================================================
// Targeted slow-row path: 当慢路径 target 数较少时, 只处理受影响分组, 精确等价。
// ===========================================================================
const TARGETED_MAX_TARGETS: usize = 256;

#[inline]
fn segment_bounds(
    order: &[usize],
    inv: &[i32],
    codes: &[f64],
    j: usize,
) -> Option<(usize, usize)> {
    let code = codes[j];
    if code.is_nan() {
        return None;
    }
    let n = order.len();
    let p = inv[j] as usize;
    let mut s = p;
    while s > 0 && codes[order[s - 1]] == code {
        s -= 1;
    }
    let mut e = p + 1;
    while e < n && codes[order[e]] == code {
        e += 1;
    }
    Some((s, e))
}

/// targeted fill_ind_reg (3 级); 仅处理 targets 所在分组。
fn fill_ind_reg_row_targeted(
    row: &mut [f64],
    size_row: &[f64],
    level_rows: [&[f64]; 3],
    order_rows: [&[usize]; 3],
    inv_rows: [&[i32]; 3],
    targets: &mut Vec<usize>,
    ys: &mut Vec<f64>,
    bs: &mut Vec<f64>,
) {
    let n = row.len();
    if targets.is_empty() {
        return;
    }
    let mut groups: Vec<(usize, Vec<usize>)> = Vec::new();
    for li in 0..3 {
        if targets.is_empty() {
            break;
        }
        if !has_ge_n_unique(row, 10) {
            continue;
        }
        let codes = level_rows[li];
        let order = order_rows[li];
        let inv = inv_rows[li];
        groups.clear();
        for &j in targets.iter() {
            if let Some((s, e)) = segment_bounds(order, inv, codes, j) {
                if let Some(g) = groups.iter_mut().find(|(gs, _)| *gs == s) {
                    g.1.push(j);
                } else {
                    groups.push((s, vec![j]));
                }
            }
        }
        for (s, js) in groups.iter() {
            let e = {
                let code = codes[order[*s]];
                let mut ee = *s + 1;
                while ee < n && codes[order[ee]] == code {
                    ee += 1;
                }
                ee
            };
            // 该组是否有 size 有限的 target (决定 OLS 结果是否会被用到)
            let mut any_size_finite = false;
            for &j in js.iter() {
                if size_row[j].is_finite() {
                    any_size_finite = true;
                    break;
                }
            }
            if !any_size_finite {
                continue;
            }
            ys.clear();
            bs.clear();
            for &ci in &order[*s..e] {
                let ok = !row[ci].is_nan() && !size_row[ci].is_nan();
                if ok {
                    ys.push(row[ci]);
                    bs.push(size_row[ci]);
                }
            }
            if ys.len() >= 10 {
                let (c0, c1) = ols2(ys, bs);
                for &j in js.iter() {
                    let ok = !row[j].is_nan() && !size_row[j].is_nan();
                    if !ok {
                        row[j] = c0 + c1 * size_row[j];
                    }
                }
            }
        }
        targets.retain(|&j| row[j].is_nan() || size_row[j].is_nan());
    }
}

/// targeted 三级中位填充; valid_mask 仅市场级使用。
fn median_fill_row_targeted(
    row: &mut [f64],
    level_rows: [&[f64]; 3],
    order_rows: [&[usize]; 3],
    inv_rows: [&[i32]; 3],
    valid_masks: [Option<&[f64]>; 3],
    targets: &mut Vec<usize>,
    sv: &mut Vec<f64>,
) {
    let n = row.len();
    if targets.is_empty() {
        return;
    }
    let mut groups: Vec<(usize, Vec<usize>)> = Vec::new();
    for li in 0..3 {
        if targets.is_empty() {
            break;
        }
        let codes = level_rows[li];
        let order = order_rows[li];
        let inv = inv_rows[li];
        let valid_mask = valid_masks[li];
        groups.clear();
        for &j in targets.iter() {
            if let Some(vm) = valid_mask {
                if vm[j] != 1.0 {
                    continue;
                }
            }
            if let Some((s, _)) = segment_bounds(order, inv, codes, j) {
                if let Some(g) = groups.iter_mut().find(|(gs, _)| *gs == s) {
                    g.1.push(j);
                } else {
                    groups.push((s, vec![j]));
                }
            }
        }
        for (s, js) in groups.iter() {
            let code = codes[order[*s]];
            let mut e = *s + 1;
            while e < n && codes[order[e]] == code {
                e += 1;
            }
            sv.clear();
            for &ci in &order[*s..e] {
                let valid = valid_mask.map_or(true, |vm| vm[ci] == 1.0);
                if valid && !row[ci].is_nan() {
                    sv.push(row[ci]);
                }
            }
            if !sv.is_empty() {
                let med = median_inplace(sv);
                for &j in js.iter() {
                    let valid = valid_mask.map_or(true, |vm| vm[j] == 1.0);
                    if row[j].is_nan() && valid {
                        row[j] = med;
                    }
                }
            }
        }
        targets.retain(|&j| row[j].is_nan());
    }
}

/// 对 restrict==0 的位置做 f64 rank_pct (等价于生产 restrict 置空后再 rank_pct)。
fn rank_pct_free_f64_in_place(
    row: &mut [f64],
    restrict_row: &[f64],
    ranks: &mut Vec<f64>,
    idxs: &mut Vec<usize>,
    tmp: &mut Vec<usize>,
    keys: &mut Vec<u64>,
) {
    let n = row.len();
    let mut vals: Vec<f64> = Vec::with_capacity(n);
    for i in 0..n {
        if restrict_row[i] == 0.0 {
            vals.push(row[i]);
        } else {
            vals.push(f64::NAN);
        }
    }
    rank_pct_row_into(&vals, ranks, idxs, tmp, keys);
    row.copy_from_slice(ranks);
}

/// 慢路径 targeted 版: 初始 target 少时只处理受影响分组。
fn neutralize_slow_targeted(
    slot_r: &[f32],
    shared: &NeutralizeStdShared,
    inv_orders: &[Array2<i32>],
    idx: usize,
    out_r: &mut [f32],
    pct: &mut Vec<f64>,
    filled: &mut Vec<f64>,
    targets: &mut Vec<usize>,
    idxs: &mut Vec<usize>,
    tmp: &mut Vec<usize>,
    keys32: &mut Vec<u32>,
    keys64: &mut Vec<u64>,
    ranks64: &mut Vec<f64>,
    ys: &mut Vec<f64>,
    bs: &mut Vec<f64>,
    sv: &mut Vec<f64>,
    y_buf: &mut Vec<f64>,
    xty: &mut Vec<f64>,
    ind0_row: &mut Vec<f64>,
) {
    let n = slot_r.len();
    let restrict_v = shared.restrict_f64.row(idx);
    let restrict_r = restrict_v.as_slice().unwrap();
    let ind2_v = shared.ind2.row(idx); let ind1_v = shared.ind1.row(idx); let size_v = shared.size_ranked.row(idx);
    let zeros_v = shared.zeros.row(idx); let mask_v = shared.ind1_mask.row(idx);
    let ind2_r=ind2_v.as_slice().unwrap(); let ind1_r=ind1_v.as_slice().unwrap(); let size_r=size_v.as_slice().unwrap();
    let zeros_r=zeros_v.as_slice().unwrap(); let mask_r=mask_v.as_slice().unwrap();
    let o0_v = shared.orders[0].row(idx); let o1_v = shared.orders[1].row(idx); let o2_v = shared.orders[2].row(idx);
    let o3_v = shared.orders[3].row(idx); let o4_v = shared.orders[4].row(idx);
    let o0=o0_v.as_slice().unwrap(); let o1=o1_v.as_slice().unwrap(); let o2=o2_v.as_slice().unwrap();
    let o3=o3_v.as_slice().unwrap(); let o4=o4_v.as_slice().unwrap();
    let i0_v = inv_orders[0].row(idx); let i1_v = inv_orders[1].row(idx); let i2_v = inv_orders[2].row(idx);
    let i3_v = inv_orders[3].row(idx); let i4_v = inv_orders[4].row(idx);
    let i0=i0_v.as_slice().unwrap(); let i1=i1_v.as_slice().unwrap(); let i2=i2_v.as_slice().unwrap();
    let i3=i3_v.as_slice().unwrap(); let i4=i4_v.as_slice().unwrap();
    ind0_row.clear();
    ind0_row.extend(ind1_r.iter().map(|&v| if v.is_nan() { 0.0 } else { 1.0 }));

    // 第一层 rank pct (生产语义: 对全部有限值)
    rank_pct_row_from_f32_in(slot_r, pct, idxs, tmp, keys32);

    // 初始 targets (只保留可能影响 free 结果的):
    //   free: row NaN 或 size NaN
    //   outside: row NaN 且 size 有限 (会被 OLS 填成有限值, 后续 level 可能成为 observed)
    targets.clear();
    for j in 0..n {
        let rnan = pct[j].is_nan();
        let snan = size_r[j].is_nan();
        let is_free = restrict_r[j] == 0.0;
        if (is_free && (rnan || snan)) || (!is_free && rnan && !snan) {
            targets.push(j);
        }
    }
    fill_ind_reg_row_targeted(
        pct, size_r,
        [ind2_r, ind1_r, ind0_row],
        [o0, o1, o2],
        [i0, i1, i2],
        targets, ys, bs,
    );
    // ind1 NaN 置空, 并加入 median targets
    for j in 0..n {
        if ind1_r[j].is_nan() {
            if !pct[j].is_nan() { targets.push(j); }
            pct[j] = f64::NAN;
        }
    }
    filled.copy_from_slice(pct);
    median_fill_row_targeted(
        filled,
        [ind2_r, ind1_r, zeros_r],
        [o3, o4, o0],
        [i3, i4, i0],
        [None, None, Some(mask_r)],
        targets, sv,
    );
    rank_pct_free_f64_in_place(filled, restrict_r, ranks64, idxs, tmp, keys64);
    let _ = ols_day_row_fast(filled, shared, idx, y_buf, xty, out_r);
}

/// 新 hybrid: fast day 走 free rank; slow day target 少走 targeted, 否则走生产 full row。
#[pyfunction]
#[pyo3(signature = (slots, shared, force_fast=false, targeted_max=256usize))]
pub fn sandbox_fast_batch_v2(
    py: Python<'_>,
    slots: PyRef<SlotsHandle>,
    shared: PyRef<SharedHandle>,
    force_fast: bool,
    targeted_max: usize,
) -> PyResult<Vec<Py<PyArray2<f32>>>> {
    let base = &*shared.inner;
    let inv = &*shared.inv_orders;
    let b = slots.slots.len();
    let (t, n) = slots.slots[0].dim();
    let mut outs: Vec<Array2<f32>> = (0..b).map(|_| Array2::<f32>::from_elem((t,n), f32::NAN)).collect();
    let mut pct: Vec<f64> = vec![0.0; n];
    let mut filled: Vec<f64> = vec![0.0; n];
    let mut targets: Vec<usize> = Vec::with_capacity(256);
    let mut idxs: Vec<usize> = Vec::with_capacity(n);
    let mut tmp: Vec<usize> = Vec::with_capacity(n);
    let mut keys32: Vec<u32> = vec![0u32; n];
    let mut keys64: Vec<u64> = vec![0u64; n];
    let mut ranks64: Vec<f64> = Vec::with_capacity(n);
    let mut ys: Vec<f64> = Vec::with_capacity(n);
    let mut bs: Vec<f64> = Vec::with_capacity(n);
    let mut sv: Vec<f64> = Vec::with_capacity(n);
    let mut nan_mask: Vec<bool> = Vec::with_capacity(n);
    let mut obs: Vec<bool> = Vec::with_capacity(n);
    let mut y_buf: Vec<f64> = Vec::with_capacity(n);
    let mut xty: Vec<f64> = Vec::with_capacity(64);
    let mut ind0_row: Vec<f64> = vec![0.0; n];
    let mut t_fast = 0u64; let mut t_targ = 0u64; let mut t_full = 0u64;
    for f in 0..b {
        for idx in 0..t {
            let slot_v = slots.slots[f].row(idx);
            let slot_r = slot_v.as_slice().unwrap();
            let restrict_v = base.restrict_f64.row(idx);
            let size_v0 = base.size_ranked.row(idx);
            let ind1_v0 = base.ind1.row(idx);
            let restrict_r = restrict_v.as_slice().unwrap();
            let size_r = size_v0.as_slice().unwrap();
            let ind1_r = ind1_v0.as_slice().unwrap();
            let mut out_row = outs[f].row_mut(idx);
            let out_r = out_row.as_slice_mut().unwrap();
            if force_fast || fast_day_ok(slot_r, restrict_r, size_r, ind1_r) {
                let _t = std::time::Instant::now();
                rank_pct_free_from_f32(slot_r, restrict_r, &mut pct, &mut idxs, &mut tmp, &mut keys32);
                let _ = ols_day_row_fast(&pct, base, idx, &mut y_buf, &mut xty, out_r);
                t_fast += _t.elapsed().as_nanos() as u64;
            } else {
                // 初始 target 数 (只统计会影响 free 结果的 target)
                let mut nt = 0usize;
                for j in 0..n {
                    let rnan = slot_r[j].is_nan();
                    let snan = size_r[j].is_nan();
                    let is_free = restrict_r[j] == 0.0;
                    if (is_free && (rnan || snan)) || (!is_free && rnan && !snan) { nt += 1; }
                }
                if nt <= targeted_max {
                    let _t = std::time::Instant::now();
                    neutralize_slow_targeted(
                        slot_r, base, inv, idx, out_r,
                        &mut pct, &mut filled, &mut targets,
                        &mut idxs, &mut tmp, &mut keys32, &mut keys64, &mut ranks64,
                        &mut ys, &mut bs, &mut sv, &mut y_buf, &mut xty, &mut ind0_row,
                    );
                    t_targ += _t.elapsed().as_nanos() as u64;
                } else {
                    let _t = std::time::Instant::now();
                    // 生产 full row 路径
                    let ind2_v=base.ind2.row(idx); let ind1_v=base.ind1.row(idx); let size_v=base.size_ranked.row(idx);
                    let zeros_v=base.zeros.row(idx); let mask_v=base.ind1_mask.row(idx);
                    let ind2_r=ind2_v.as_slice().unwrap(); let ind1_r2=ind1_v.as_slice().unwrap(); let size_r2=size_v.as_slice().unwrap();
                    let zeros_r=zeros_v.as_slice().unwrap(); let mask_r=mask_v.as_slice().unwrap();
                    let o0_v=base.orders[0].row(idx); let o1_v=base.orders[1].row(idx); let o2_v=base.orders[2].row(idx);
                    let o3_v=base.orders[3].row(idx); let o4_v=base.orders[4].row(idx);
                    let o0=o0_v.as_slice().unwrap(); let o1=o1_v.as_slice().unwrap(); let o2=o2_v.as_slice().unwrap();
                    let o3=o3_v.as_slice().unwrap(); let o4=o4_v.as_slice().unwrap();
                    ind0_row.clear();
                    ind0_row.extend(ind1_r2.iter().map(|&v| if v.is_nan() { 0.0 } else { 1.0 }));
                    rank_pct_row_from_f32_in(slot_r, &mut pct, &mut idxs, &mut tmp, &mut keys32);
                    fill_ind_reg_row(&mut pct, [ind2_r, ind1_r2, &ind0_row], size_r2, [o0,o1,o2], &mut ys,&mut bs,&mut obs);
                    for (j,&v) in ind1_r2.iter().enumerate(){ if v.is_nan(){pct[j]=f64::NAN;} }
                    filled.copy_from_slice(&pct);
                    nan_mask.clear(); for &v in filled.iter(){nan_mask.push(v.is_nan());}
                    if nan_mask.iter().any(|&x|x){median_fill_level_row(&mut filled,ind2_r,None,o3,&nan_mask,&mut sv);}
                    nan_mask.clear(); for &v in filled.iter(){nan_mask.push(v.is_nan());}
                    if nan_mask.iter().any(|&x|x){median_fill_level_row(&mut filled,ind1_r2,None,o4,&nan_mask,&mut sv);}
                    nan_mask.clear(); for &v in filled.iter(){nan_mask.push(v.is_nan());}
                    if nan_mask.iter().any(|&x|x){median_fill_level_row(&mut filled,zeros_r,Some(mask_r),o0,&nan_mask,&mut sv);}
                    for (j,&v) in restrict_r.iter().enumerate(){ if v!=0.0{filled[j]=f64::NAN;} }
                    rank_pct_row_f64_in_place(&mut filled,&mut ranks64,&mut idxs,&mut tmp,&mut keys64);
                    let _ = ols_day_row_fast(&filled, base, idx, &mut y_buf, &mut xty, out_r);
                    t_full += _t.elapsed().as_nanos() as u64;
                }
            }
        }
    }
    if std::env::var("NEU_FAST_PROFILE").is_ok() {
        eprintln!("[v2 prof] fast={:.1}ms targeted={:.1}ms full={:.1}ms", t_fast as f64/1e6, t_targ as f64/1e6, t_full as f64/1e6);
    }
    Ok(outs.into_iter().map(|a| a.into_pyarray(py).to_owned()).collect())
}

/// 近似快路径专用 OLS: 当 y 在理想 valid 集内有缺失时, 从预计算 X'X 中减去缺失
/// 位置贡献并现场 Cholesky; 避免生产 ols_day_row_fast 的 SVD 回退。
fn ols_day_adjusted_fast(
    y_row: &[f64],
    shared: &NeutralizeStdShared,
    idx: usize,
    out_f32_row: &mut [f32],
) -> bool {
    let k = 10usize;
    let (p, valid_idx, valid_cols, xtx_pre) = &shared.per_date[idx];
    if *p == 0 {
        return false;
    }
    let n_valid = valid_idx.len();
    let xd = &shared.xdays[idx];
    // 检查缺失
    let mut n_missing = 0usize;
    let mut mn = f64::INFINITY;
    let mut mx = f64::NEG_INFINITY;
    for (pos, &j) in valid_idx.iter().enumerate() {
        let y = y_row[j as usize];
        if !y.is_finite() {
            n_missing += 1;
        } else {
            if y < mn { mn = y; }
            if y > mx { mx = y; }
        }
    }
    if n_missing == 0 {
        let mut y_buf = Vec::with_capacity(n_valid);
        let mut xty = vec![0f64; *p];
        return ols_day_row_fast(y_row, shared, idx, &mut y_buf, &mut xty, out_f32_row);
    }
    if n_valid - n_missing <= 10 {
        return false;
    }
    if mn == mx && n_missing == 0 {
        for &j in valid_idx.iter() { out_f32_row[j as usize] = 0.5; }
        return false;
    }
    // 调整 X'X
    let mut xtx = xtx_pre.clone();
    for (pos, &j) in valid_idx.iter().enumerate() {
        if y_row[j as usize].is_finite() { continue; }
        let xrow = &xd[pos * k..pos * k + k];
        let ic = valid_cols[pos];
        for c in 0..k {
            let b = xrow[c];
            xtx[c * *p + c] -= b * b;
            for c2 in (c + 1)..k {
                let v = b * xrow[c2];
                xtx[c * *p + c2] -= v;
                xtx[c2 * *p + c] -= v;
            }
        }
        if ic >= 0 {
            let col = k + ic as usize;
            xtx[col * *p + col] -= 1.0;
            for c in 0..k {
                xtx[c * *p + col] -= xrow[c];
                xtx[col * *p + c] -= xrow[c];
            }
        }
    }
    // xty
    let mut xty = vec![0f64; *p];
    for (pos, &j) in valid_idx.iter().enumerate() {
        let y = y_row[j as usize];
        if !y.is_finite() { continue; }
        let xrow = &xd[pos * k..pos * k + k];
        for c in 0..k { xty[c] += xrow[c] * y; }
        let ic = valid_cols[pos];
        if ic >= 0 { xty[k + ic as usize] += y; }
    }
    let m = DMatrix::from_row_slice(*p, *p, &xtx);
    let chol = match Cholesky::new(m) {
        Some(c) => c,
        None => return false,
    };
    let rhs = DMatrix::from_column_slice(*p, 1, &xty);
    let coef: Vec<f64> = chol.solve(&rhs).column(0).iter().copied().collect();
    for (pos, &j) in valid_idx.iter().enumerate() {
        let ji = j as usize;
        let y = y_row[ji];
        if !y.is_finite() { continue; }
        let xrow = &xd[pos * k..pos * k + k];
        let mut pred = 0.0;
        for c in 0..k { pred += coef[c] * xrow[c]; }
        let ic = valid_cols[pos];
        if ic >= 0 { pred += coef[k + ic as usize]; }
        out_f32_row[ji] = (y - pred) as f32;
    }
    true
}

/// 近似模式: 所有日期都用 fast rank free + adjusted OLS (跳过全部 fill/median/rank2)。
#[pyfunction]
pub fn sandbox_fast_approx_batch(
    py: Python<'_>,
    slots: PyRef<SlotsHandle>,
    shared: PyRef<SharedHandle>,
) -> PyResult<Vec<Py<PyArray2<f32>>>> {
    let base = &*shared.inner;
    let b = slots.slots.len();
    let (t, n) = slots.slots[0].dim();
    let mut outs: Vec<Array2<f32>> = (0..b).map(|_| Array2::<f32>::from_elem((t,n), f32::NAN)).collect();
    let mut pct: Vec<f64> = vec![0.0; n];
    let mut idxs: Vec<usize> = Vec::with_capacity(n);
    let mut tmp: Vec<usize> = Vec::with_capacity(n);
    let mut keys32: Vec<u32> = vec![0u32; n];
    for f in 0..b {
        for idx in 0..t {
            let slot_v = slots.slots[f].row(idx);
            let restrict_v = base.restrict_f64.row(idx);
            let slot_r = slot_v.as_slice().unwrap();
            let restrict_r = restrict_v.as_slice().unwrap();
            rank_pct_free_from_f32(slot_r, restrict_r, &mut pct, &mut idxs, &mut tmp, &mut keys32);
            let mut out_row = outs[f].row_mut(idx);
            let _ = ols_day_adjusted_fast(&pct, base, idx, out_row.as_slice_mut().unwrap());
        }
    }
    Ok(outs.into_iter().map(|a| a.into_pyarray(py).to_owned()).collect())
}

#[pyfunction]
pub fn sandbox_fast_approx_batch_batched(
    py: Python<'_>,
    slots: PyRef<SlotsHandle>,
    shared: PyRef<SharedHandle>,
) -> PyResult<Vec<Py<PyArray2<f32>>>> {
    let base = &*shared.inner;
    let b = slots.slots.len();
    let (t, n) = slots.slots[0].dim();
    let mut outs: Vec<Array2<f32>> = (0..b).map(|_| Array2::<f32>::from_elem((t,n), f32::NAN)).collect();
    let mut pct: Vec<f64> = vec![0.0; n];
    let mut idxs: Vec<usize> = Vec::with_capacity(n);
    let mut tmp: Vec<usize> = Vec::with_capacity(n);
    let mut keys32: Vec<u32> = vec![0u32; n];
    for idx in 0..t {
        let restrict_v = base.restrict_f64.row(idx);
        let restrict_r = restrict_v.as_slice().unwrap();
        for f in 0..b {
            let slot_v = slots.slots[f].row(idx);
            let slot_r = slot_v.as_slice().unwrap();
            rank_pct_free_from_f32(slot_r, restrict_r, &mut pct, &mut idxs, &mut tmp, &mut keys32);
            let mut out_row = outs[f].row_mut(idx);
            let _ = ols_day_adjusted_fast(&pct, base, idx, out_row.as_slice_mut().unwrap());
        }
    }
    Ok(outs.into_iter().map(|a| a.into_pyarray(py).to_owned()).collect())
}
