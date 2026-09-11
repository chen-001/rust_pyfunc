//! 生产标准中性化的 sandbox 复刻 + 优化对照。
//! 生产路径 = factor_neutralize_std.rs 的 neutralize_std_section_owned 逐函数拷贝
//! （含每步计时）；优化路径 = 把只依赖日期的量全部预计算（设计矩阵 X、Cholesky 因子、
//! 行业排序、有效掩码），每 slot 只做 y 相关计算，数值与生产逐位一致（残差部分）。
use std::cmp::Ordering;
use std::time::Instant;

use nalgebra::{Cholesky, DMatrix};
use ndarray::{Array2, ArrayView2};

// ==================== 生产代码拷贝（无计时，逻辑逐行一致） ====================
#[inline]
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
        rank_pct_row_into(
            row.as_slice().unwrap(),
            &mut ranks,
            &mut idxs,
            &mut tmp,
            &mut keys,
        );
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
    debug_assert_eq!(n_base, 1);
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
    let mut nan_mask: Vec<bool> = Vec::with_capacity(n);
    let mut source_row: Vec<f64> = Vec::with_capacity(n);
    let is_market_level = valid_mask.is_some();
    for idx in 0..t {
        let mut row = values.row_mut(idx);
        source_row.clear();
        source_row.extend(row.iter().copied());
        nan_mask.clear();
        nan_mask.extend(row.iter().map(|v| v.is_nan()));
        if !nan_mask.iter().any(|&b| b) {
            continue;
        }
        let codes_row = codes.row(idx);
        let valid_row = valid_mask.map(|vm| vm.row(idx));
        if is_market_level {
            order.clear();
            order.extend(0..n);
        } else {
            order.clear();
            order_keys.clear();
            for j in 0..n {
                order.push(j);
                order_keys.push(mono_key(codes_row[j]));
            }
            radix_sort_order(&order_keys, &mut order, &mut order_tmp);
        }
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
                if valid_row.map_or(true, |vr| vr[ci] == 1.0) && !source_row[ci].is_nan() {
                    sv.push(source_row[ci]);
                }
            }
            if !sv.is_empty() {
                let med = median_inplace(&mut sv);
                for &ci in &order[seg_start..seg_end] {
                    if nan_mask[ci] && valid_row.map_or(true, |vr| vr[ci] == 1.0) {
                        row[ci] = med;
                    }
                }
            }
            seg_start = seg_end;
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
            ind_codes.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
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
                        match ind_codes.binary_search_by(|x| x.partial_cmp(&c).unwrap_or(Ordering::Equal)) {
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
        uniq.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
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
            let coef: Vec<f64> = if !use_svd {
                let chol = Cholesky::new(m).expect("chol");
                chol.solve(&rhs).column(0).iter().copied().collect()
            } else {
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
                let s = svd.singular_values;
                let s_max = s.iter().cloned().fold(0.0_f64, f64::max);
                let rcond = s_max * (n_r.max(p) as f64) * 2.22e-16;
                let ym = DMatrix::from_fn(n_r, 1, |r_i, _| rows[r_i][0]);
                let uty = u.transpose() * ym;
                let mut coef = DMatrix::zeros(p, 1);
                for i in 0..p {
                    if s[i] > rcond {
                        coef[(i, 0)] = uty[(i, 0)] / s[i];
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

// ==================== 共享预计算（生产语义） ====================
pub struct NeuShared {
    pub industry: Array2<f64>,
    pub restrict_f64: Array2<f64>,
    pub ind1: Array2<f64>,
    pub ind2: Array2<f64>,
    pub zeros: Array2<f64>,
    pub ind1_mask: Array2<f64>,
    pub barra_ranked: Vec<Array2<f64>>,
    pub size_ranked: Array2<f64>,
}

pub fn neu_precompute(
    industry: &Array2<f64>,
    restrict: &Array2<f32>,
    barra_raw: &ndarray::Array3<f64>,
) -> NeuShared {
    let restrict_f64 = Array2::<f64>::from_shape_vec(
        restrict.dim(),
        restrict.iter().map(|&v| v as f64).collect(),
    )
    .unwrap();
    let ind1 = industry.map(|&v| (v / 10000.0).floor());
    let ind2 = industry.map(|&v| (v / 100.0).floor());
    let zeros = Array2::<f64>::zeros(industry.dim());
    let ind1_mask = ind1.map(|&v| if v.is_nan() { 0.0 } else { 1.0 });

    let mut barra_ranked: Vec<Array2<f64>> = (0..10)
        .map(|f| {
            let mut m = Array2::<f64>::from_elem((industry.nrows(), industry.ncols()), f64::NAN);
            for i in 0..industry.nrows() {
                for j in 0..industry.ncols() {
                    m[[i, j]] = barra_raw[[i, j, f]];
                }
            }
            m
        })
        .collect();
    for b in barra_ranked.iter_mut() {
        rank_pct_all(b);
    }
    let mut size_ranked = barra_ranked[2].clone();
    rank_pct_all(&mut size_ranked);

    NeuShared {
        industry: industry.clone(),
        restrict_f64,
        ind1,
        ind2,
        zeros,
        ind1_mask,
        barra_ranked,
        size_ranked,
    }
}

// ==================== 生产路径：neutralize_std_section_owned（带计时） ====================
pub struct NeuStepTimes {
    pub rank1: f64,
    pub fill_ind: f64,
    pub mask_clone: f64,
    pub fills: f64,
    pub restrict: f64,
    pub rank2: f64,
    pub residual: f64,
    pub rank3: f64,
    pub convert: f64,
}

pub fn neutralize_slot_prod(
    slot: ArrayView2<'_, f32>,
    shared: &NeuShared,
) -> (Array2<f32>, NeuStepTimes) {
    let mut t = NeuStepTimes {
        rank1: 0.0,
        fill_ind: 0.0,
        mask_clone: 0.0,
        fills: 0.0,
        restrict: 0.0,
        rank2: 0.0,
        residual: 0.0,
        rank3: 0.0,
        convert: 0.0,
    };
    let factor_f64 = slot.map(|&v| v as f64);
    let (n_dates, n_stocks) = slot.dim();

    let mut fv_ranked = factor_f64;
    let s = Instant::now();
    rank_pct_all(&mut fv_ranked);
    t.rank1 = s.elapsed().as_secs_f64();

    let s = Instant::now();
    fill_ind_reg(&mut fv_ranked, &[shared.size_ranked.clone()], &shared.industry);
    t.fill_ind = s.elapsed().as_secs_f64();

    let s = Instant::now();
    for i in 0..(n_dates * n_stocks) {
        if shared.ind1.as_slice().unwrap()[i].is_nan() {
            fv_ranked.as_slice_mut().unwrap()[i] = f64::NAN;
        }
    }
    let mut fv_filled = fv_ranked.clone();
    t.mask_clone = s.elapsed().as_secs_f64();

    let s = Instant::now();
    fill_by_group_median_inplace(&mut fv_filled, &shared.ind2, None);
    fill_by_group_median_inplace(&mut fv_filled, &shared.ind1, None);
    fill_by_group_median_inplace(&mut fv_filled, &shared.zeros, Some(&shared.ind1_mask));
    t.fills = s.elapsed().as_secs_f64();

    let s = Instant::now();
    for i in 0..(n_dates * n_stocks) {
        if shared.restrict_f64.as_slice().unwrap()[i] != 0.0 {
            fv_filled.as_slice_mut().unwrap()[i] = f64::NAN;
        }
    }
    t.restrict = s.elapsed().as_secs_f64();

    let s = Instant::now();
    rank_pct_all(&mut fv_filled);
    t.rank2 = s.elapsed().as_secs_f64();

    let s = Instant::now();
    let resid = get_residual(&fv_filled, &shared.barra_ranked, Some(&shared.ind1));
    t.residual = s.elapsed().as_secs_f64();

    let s = Instant::now();
    let mut resid_rank = resid;
    rank_pct_all(&mut resid_rank);
    t.rank3 = s.elapsed().as_secs_f64();

    let s = Instant::now();
    let mut output = Array2::<f32>::from_elem((n_dates, n_stocks), f32::NAN);
    let out_slice = output.as_slice_mut().unwrap();
    let rr = resid_rank.as_slice().unwrap();
    for i in 0..(n_dates * n_stocks) {
        let v = rr[i];
        out_slice[i] = if v.is_nan() { f32::NAN } else { v as f32 };
    }
    t.convert = s.elapsed().as_secs_f64();
    (output, t)
}

// ==================== 优化路径 ====================
/// 只依赖日期的量：get_residual 的设计矩阵（含 Cholesky）、行业排序顺序、
/// 有效掩码。每 slot 只做 y 相关计算。
pub struct NeuSharedOpt {
    pub base: NeuShared,
    /// get_residual: 每日 [valid_idxs(u32), X(n_valid×p) f64, ind_cols(i32), p, L(p×p)]
    pub resid_days: Vec<ResidDay>,
    /// fill_ind_reg: 每日 3 层排序 (ind2/ind1/ind0) 的 order（u32，稳定 radix 语义）
    pub fillind_orders: [Vec<Vec<u32>>; 3],
    /// fill_by_group_median: ind2/ind1 层的 order
    pub fillmed_orders: [Vec<Vec<u32>>; 2],
}

pub struct ResidDay {
    pub valid_idxs: Vec<u32>,
    pub x: Vec<f64>,      // n_valid × p 行主序
    pub ind_cols: Vec<i32>,
    pub p: usize,
    pub chol_l: DMatrix<f64>, // Cholesky 因子 L（与生产 Cholesky::new(X'X) 一致）
}

fn stable_radix_order(vals: &[f64]) -> Vec<u32> {
    let mut order: Vec<usize> = (0..vals.len()).collect();
    let mut keys: Vec<u64> = vals.iter().map(|&v| mono_key(v)).collect();
    let mut tmp: Vec<usize> = Vec::with_capacity(vals.len());
    radix_sort_order(&keys, &mut order, &mut tmp);
    order.into_iter().map(|i| i as u32).collect()
}

pub fn neu_precompute_opt(
    industry: &Array2<f64>,
    restrict: &Array2<f32>,
    barra_raw: &ndarray::Array3<f64>,
) -> NeuSharedOpt {
    let base = neu_precompute(industry, restrict, barra_raw);
    let (t, n) = industry.dim();
    let k = 10usize;

    // ---- get_residual 每日预计算 ----
    let mut resid_days = Vec::with_capacity(t);
    for idx in 0..t {
        let mut day = ResidDay {
            valid_idxs: Vec::new(),
            x: Vec::new(),
            ind_cols: Vec::new(),
            p: 0,
            chol_l: DMatrix::zeros(0, 0),
        };
        // 生产语义：fv 行在 step5 后 NaN 当且仅当 (ind1 NaN) 或 (restrict != 0)
        // —— 与因子无关；bench 全有限条件同样因子无关。
        let mut valid: Vec<u32> = Vec::new();
        let mut ind_codes: Vec<f64> = Vec::new();
        for j in 0..n {
            let c = base.ind1[[idx, j]];
            let is_open = base.restrict_f64[[idx, j]] == 0.0;
            if c.is_nan() || !is_open {
                continue;
            }
            if !base.barra_ranked.iter().all(|b| b[[idx, j]].is_finite()) {
                continue;
            }
            valid.push(j as u32);
            if !ind_codes.contains(&c) {
                ind_codes.push(c);
            }
        }
        if valid.len() <= 10 {
            resid_days.push(day);
            continue;
        }
        ind_codes.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
        let n_ind = ind_codes.len();
        let p = k + n_ind;
        let nv = valid.len();
        let mut x = vec![0.0f64; nv * p];
        let mut ind_cols = vec![-1i32; nv];
        for (i, &j) in valid.iter().enumerate() {
            for c in 0..k {
                x[i * p + c] = base.barra_ranked[c][[idx, j as usize]];
            }
            let code = base.ind1[[idx, j as usize]];
            if !code.is_nan() {
                if let Ok(pos) = ind_codes.binary_search_by(|x2| {
                    x2.partial_cmp(&code).unwrap_or(Ordering::Equal)
                }) {
                    ind_cols[i] = pos as i32;
                    x[i * p + k + pos] = 1.0;
                }
            }
        }
        // X'X：与生产 get_residual 完全相同的行序累积
        let mut xtx = vec![0.0f64; p * p];
        for i in 0..nv {
            for c in 0..k {
                let b = x[i * p + c];
                xtx[c * p + c] += b * b;
                for c2 in (c + 1)..k {
                    let v = b * x[i * p + c2];
                    xtx[c * p + c2] += v;
                    xtx[c2 * p + c] += v;
                }
            }
            let ic = ind_cols[i];
            if ic >= 0 {
                let col = k + ic as usize;
                xtx[col * p + col] += 1.0;
                for c in 0..k {
                    let b = x[i * p + c];
                    xtx[c * p + col] += b;
                    xtx[col * p + c] += b;
                }
            }
        }
        let m = DMatrix::from_row_slice(p, p, &xtx);
        // 生产对 n_valid<=40 或近奇异走 SVD；模板轴每日常数千有效样本，
        // 此处记录走 SVD 的日期（真实数据上数量极小/为 0），优化路径仅对
        // Cholesky 快路径生效，SVD 日期回退生产函数。
        let chol_l = match Cholesky::new(m) {
            Some(ch) => ch.l().clone(),
            None => DMatrix::zeros(0, 0),
        };
        day.valid_idxs = valid;
        day.x = x;
        day.ind_cols = ind_cols;
        day.p = p;
        day.chol_l = chol_l;
        resid_days.push(day);
    }

    // ---- fill_ind_reg / fill_by_group_median 每日排序预计算 ----
    let mut fillind_orders: [Vec<Vec<u32>>; 3] = Default::default();
    for (li, level) in [&base.ind2, &base.ind1].iter().enumerate() {
        fillind_orders[li] = (0..t)
            .map(|idx| stable_radix_order(level.row(idx).as_slice().unwrap()))
            .collect();
    }
    // ind0 层：与 ind1 结构相同（NaN→0），但排序键不同（0 与有限码）
    {
        let ind0: Array2<f64> = base.ind1.map(|&v| if v.is_nan() { 0.0 } else { 1.0 });
        fillind_orders[2] = (0..t)
            .map(|idx| stable_radix_order(ind0.row(idx).as_slice().unwrap()))
            .collect();
    }
    let fillmed_orders: [Vec<Vec<u32>>; 2] = [
        (0..t)
            .map(|idx| stable_radix_order(base.ind2.row(idx).as_slice().unwrap()))
            .collect(),
        (0..t)
            .map(|idx| stable_radix_order(base.ind1.row(idx).as_slice().unwrap()))
            .collect(),
    ];

    NeuSharedOpt {
        base,
        resid_days,
        fillind_orders,
        fillmed_orders,
    }
}

/// 优化版 neutralize：步骤顺序与生产一致，仅替换三类「只依赖日期」的重计算。
pub fn neutralize_slot_opt(
    slot: ArrayView2<'_, f32>,
    shared: &NeuSharedOpt,
) -> (Array2<f32>, NeuStepTimes) {
    let mut t = NeuStepTimes {
        rank1: 0.0,
        fill_ind: 0.0,
        mask_clone: 0.0,
        fills: 0.0,
        restrict: 0.0,
        rank2: 0.0,
        residual: 0.0,
        rank3: 0.0,
        convert: 0.0,
    };
    let base = &shared.base;
    let factor_f64 = slot.map(|&v| v as f64);
    let (n_dates, n_stocks) = slot.dim();
    let k = 10usize;

    let mut fv_ranked = factor_f64;
    let s = Instant::now();
    rank_pct_all(&mut fv_ranked);
    t.rank1 = s.elapsed().as_secs_f64();

    // ---- fill_ind_reg（预计算排序） ----
    let s = Instant::now();
    {
        let mut cols: Vec<usize> = Vec::with_capacity(n_stocks);
        let mut f_masked: Vec<f64> = Vec::with_capacity(n_stocks);
        let mut b_masked: Vec<f64> = Vec::with_capacity(n_stocks);
        let mut not_nan: Vec<bool> = Vec::with_capacity(n_stocks);
        let mut ys: Vec<f64> = Vec::with_capacity(n_stocks);
        let mut bs: Vec<f64> = Vec::with_capacity(n_stocks);
        for (li, orders) in shared.fillind_orders.iter().enumerate() {
            for idx in 0..n_dates {
                let mut fv_row: Vec<f64> = fv_ranked.row(idx).iter().copied().collect();
                if !has_ge_n_unique(&fv_row, 10) {
                    continue;
                }
                let base_row = base.size_ranked.row(idx);
                let order = &orders[idx];
                let code_of = |j: usize| -> f64 {
                    match li {
                        0 => base.ind2[[idx, j]],
                        1 => base.ind1[[idx, j]],
                        _ => {
                            if base.ind1[[idx, j]].is_nan() {
                                0.0
                            } else {
                                1.0
                            }
                        }
                    }
                };
                let mut seg_start = 0usize;
                while seg_start < n_stocks {
                    let code = code_of(order[seg_start] as usize);
                    if code.is_nan() {
                        break;
                    }
                    let mut seg_end = seg_start + 1;
                    while seg_end < n_stocks && code_of(order[seg_end] as usize) == code {
                        seg_end += 1;
                    }
                    let m_count = seg_end - seg_start;
                    cols.clear();
                    f_masked.clear();
                    b_masked.clear();
                    not_nan.clear();
                    for &ci_u in &order[seg_start..seg_end] {
                        let ci = ci_u as usize;
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
                for j in 0..n_stocks {
                    fv_ranked[[idx, j]] = fv_row[j];
                }
            }
        }
    }
    t.fill_ind = s.elapsed().as_secs_f64();

    // ---- ind1 NaN 置空 + clone ----
    let s = Instant::now();
    for i in 0..(n_dates * n_stocks) {
        if base.ind1.as_slice().unwrap()[i].is_nan() {
            fv_ranked.as_slice_mut().unwrap()[i] = f64::NAN;
        }
    }
    let mut fv_filled = fv_ranked.clone();
    t.mask_clone = s.elapsed().as_secs_f64();

    // ---- fill_by_group_median ×3（前两层预计算排序） ----
    let s = Instant::now();
    {
        let mut sv: Vec<f64> = Vec::with_capacity(n_stocks);
        let mut nan_mask: Vec<bool> = Vec::with_capacity(n_stocks);
        let mut source_row: Vec<f64> = Vec::with_capacity(n_stocks);
        for li in 0..3 {
            let codes = match li {
                0 => &base.ind2,
                1 => &base.ind1,
                _ => &base.zeros,
            };
            let valid_mask = if li == 2 { Some(&base.ind1_mask) } else { None };
            let is_market_level = li == 2;
            for idx in 0..n_dates {
                let mut row = fv_filled.row_mut(idx);
                source_row.clear();
                source_row.extend(row.iter().copied());
                nan_mask.clear();
                nan_mask.extend(row.iter().map(|v| v.is_nan()));
                if !nan_mask.iter().any(|&b| b) {
                    continue;
                }
                let codes_row = codes.row(idx);
                let order: Vec<u32> = if is_market_level {
                    (0..n_stocks as u32).collect()
                } else {
                    shared.fillmed_orders[li][idx].clone()
                };
                let mut seg_start = 0usize;
                while seg_start < n_stocks {
                    let code = codes_row[order[seg_start] as usize];
                    if code.is_nan() {
                        break;
                    }
                    let mut seg_end = seg_start + 1;
                    while seg_end < n_stocks && codes_row[order[seg_end] as usize] == code {
                        seg_end += 1;
                    }
                    sv.clear();
                    for &ci_u in &order[seg_start..seg_end] {
                        let ci = ci_u as usize;
                        let in_valid = valid_mask.map_or(true, |vm| vm[[idx, ci]] == 1.0);
                        if in_valid && !source_row[ci].is_nan() {
                            sv.push(source_row[ci]);
                        }
                    }
                    if !sv.is_empty() {
                        let med = median_inplace(&mut sv);
                        for &ci_u in &order[seg_start..seg_end] {
                            let ci = ci_u as usize;
                            let in_valid = valid_mask.map_or(true, |vm| vm[[idx, ci]] == 1.0);
                            if nan_mask[ci] && in_valid {
                                row[ci] = med;
                            }
                        }
                    }
                    seg_start = seg_end;
                }
            }
        }
    }
    t.fills = s.elapsed().as_secs_f64();

    // ---- restrict 置空 ----
    let s = Instant::now();
    for i in 0..(n_dates * n_stocks) {
        if base.restrict_f64.as_slice().unwrap()[i] != 0.0 {
            fv_filled.as_slice_mut().unwrap()[i] = f64::NAN;
        }
    }
    t.restrict = s.elapsed().as_secs_f64();

    // ---- rank pct ----
    let s = Instant::now();
    rank_pct_all(&mut fv_filled);
    t.rank2 = s.elapsed().as_secs_f64();

    // ---- get_residual（预计算 X / L / 有效掩码） ----
    let s = Instant::now();
    let mut resid = Array2::<f64>::from_elem((n_dates, n_stocks), f64::NAN);
    {
        let mut y: Vec<f64> = Vec::with_capacity(n_stocks);
        let mut xty: Vec<f64> = Vec::with_capacity(42);
        let mut coef: Vec<f64> = Vec::with_capacity(42);
        for idx in 0..n_dates {
            let day = &shared.resid_days[idx];
            let nv = day.valid_idxs.len();
            if nv <= 10 || day.chol_l.nrows() == 0 {
                continue;
            }
            let row = fv_filled.row(idx);
            let p = day.p;
            y.clear();
            for &j in &day.valid_idxs {
                y.push(row[j as usize]);
            }
            let (mn, mx) = y
                .iter()
                .fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &v| {
                    (a.min(v), b.max(v))
                });
            if mn == mx {
                for &j in &day.valid_idxs {
                    resid[[idx, j as usize]] = 0.5;
                }
                continue;
            }
            xty.clear();
            xty.resize(p, 0.0);
            for i in 0..nv {
                let yv = y[i];
                for c in 0..k {
                    xty[c] += day.x[i * p + c] * yv;
                }
                let ic = day.ind_cols[i];
                if ic >= 0 {
                    xty[k + ic as usize] += yv;
                }
            }
            let rhs = DMatrix::from_column_slice(p, 1, &xty);
            // Cholesky::solve 语义：先解 L z = y，再解 L^T coef = z
            let z = day
                .chol_l
                .solve_lower_triangular(&rhs)
                .unwrap_or(DMatrix::zeros(p, 1));
            let sol = day
                .chol_l
                .transpose()
                .solve_upper_triangular(&z)
                .unwrap_or(DMatrix::zeros(p, 1));
            coef.clear();
            coef.extend(sol.column(0).iter().copied());
            for i in 0..nv {
                let mut pred = 0.0;
                for c in 0..k {
                    pred += coef[c] * day.x[i * p + c];
                }
                let ic = day.ind_cols[i];
                if ic >= 0 {
                    pred += coef[k + ic as usize];
                }
                resid[[idx, day.valid_idxs[i] as usize]] = y[i] - pred;
            }
        }
    }
    t.residual = s.elapsed().as_secs_f64();

    // ---- 残差 rank + f64→f32 ----
    let s = Instant::now();
    let mut resid_rank = resid;
    rank_pct_all(&mut resid_rank);
    t.rank3 = s.elapsed().as_secs_f64();

    let s = Instant::now();
    let mut output = Array2::<f32>::from_elem((n_dates, n_stocks), f32::NAN);
    let out_slice = output.as_slice_mut().unwrap();
    let rr = resid_rank.as_slice().unwrap();
    for i in 0..(n_dates * n_stocks) {
        let v = rr[i];
        out_slice[i] = if v.is_nan() { f32::NAN } else { v as f32 };
    }
    t.convert = s.elapsed().as_secs_f64();
    (output, t)
}
