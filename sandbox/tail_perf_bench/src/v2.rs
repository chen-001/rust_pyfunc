//! v2.rs — 现生产 v2_resid 中性化路径的逐位复刻 + 候选优化验证（sandbox 专用）。
//!
//! 复刻对象: rust_pyfunc/src/factor_neutralize_std.rs 的
//!   neutralize_std_slot_f32_v2_resid（含 fill_ind_reg_pre / group_median_fill_pre /
//!   get_residual_v2 / ols_day_inline 全链路）。
//! 候选优化（全部要求与复刻版输出逐位一致）:
//!   O3a: get_residual_v2 的 y 全等检查 sort+dedup → 线性 min==max 扫描
//!   O3b: 每日 Cholesky 分解只做一次（生产每 slot 每日做两次）
//!   O3c: 每日 Cholesky 预计算共享（跨 slot / 跨因子摊销为 0）
//!   O3d: xty/pred 用每日连续内存 X（10 风格列连续，替代跨 10 张大矩阵取数）
//!   O3e: 第一次 rank 用 32 位 radix（f32→f64 是保序单射，秩逐位一致）

use std::cmp::Ordering;
use std::sync::Arc;
use std::time::Instant;

use nalgebra::{Cholesky, DMatrix, Dyn};
use ndarray::{Array2, ArrayView2};

use crate::neu::{neu_precompute, NeuShared};

// ==================== 基础原语（照抄生产） ====================

fn cmp_f64(a: &f64, b: &f64) -> Ordering {
    match (a.is_nan(), b.is_nan()) {
        (true, true) => Ordering::Equal,
        (true, false) => Ordering::Greater,
        (false, true) => Ordering::Less,
        (false, false) => a.partial_cmp(b).unwrap(),
    }
}

fn mono_key(v: f64) -> u64 {
    let b = v.to_bits();
    if v.is_nan() {
        u64::MAX
    } else if b >> 63 == 0 {
        b | (1u64 << 63)
    } else {
        !b
    }
}

pub(crate) fn mono_key32(v: f32) -> u32 {
    let b = v.to_bits();
    if v.is_nan() {
        u32::MAX
    } else if b >> 31 == 0 {
        b | (1u32 << 31)
    } else {
        !b
    }
}

pub(crate) fn radix_sort_order(keys: &[u64], order: &mut Vec<usize>, tmp: &mut Vec<usize>) {
    let n = order.len();
    if n <= 1 {
        return;
    }
    tmp.clear();
    tmp.resize(n, 0);
    let mut counts = [0usize; 256];
    for pass in 0..8 {
        let shift = pass * 8;
        counts.fill(0);
        for &i in order.iter() {
            counts[((keys[i] >> shift) & 0xff) as usize] += 1;
        }
        if counts.iter().any(|&c| c == n) {
            continue;
        }
        let mut sum = 0usize;
        for c in counts.iter_mut() {
            let t = *c;
            *c = sum;
            sum += t;
        }
        for &i in order.iter() {
            let d = ((keys[i] >> shift) & 0xff) as usize;
            tmp[counts[d]] = i;
            counts[d] += 1;
        }
        std::mem::swap(order, tmp);
    }
}

pub(crate) fn radix_sort_order32(keys: &[u32], order: &mut Vec<usize>, tmp: &mut Vec<usize>) {
    let n = order.len();
    if n <= 1 {
        return;
    }
    tmp.clear();
    tmp.resize(n, 0);
    let mut counts = [0usize; 256];
    for pass in 0..4 {
        let shift = pass * 8;
        counts.fill(0);
        for &i in order.iter() {
            counts[((keys[i] >> shift) & 0xff) as usize] += 1;
        }
        if counts.iter().any(|&c| c == n) {
            continue;
        }
        let mut sum = 0usize;
        for c in counts.iter_mut() {
            let t = *c;
            *c = sum;
            sum += t;
        }
        for &i in order.iter() {
            let d = ((keys[i] >> shift) & 0xff) as usize;
            tmp[counts[d]] = i;
            counts[d] += 1;
        }
        std::mem::swap(order, tmp);
    }
}

pub(crate) fn rank_pct_row_into(
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

/// O3e: 第一次 rank 的 32 位 radix 版。输入 fv 是 f32 slot 的 f64 精确映像，
/// f32→f64 保序单射 ⇒ 排序/并列组完全一致 ⇒ 秩逐位一致。
fn rank_pct_all_from_f32(slot: &ArrayView2<'_, f32>, out: &mut Array2<f64>) {
    let (t, n) = slot.dim();
    let mut idxs: Vec<usize> = Vec::with_capacity(n);
    let mut tmp: Vec<usize> = Vec::with_capacity(n);
    let mut keys: Vec<u32> = vec![0; n];
    for ti in 0..t {
        let mut out_row = out.row_mut(ti);
        let in_row = slot.row(ti);
        idxs.clear();
        for (i, &v) in in_row.iter().enumerate() {
            out_row[i] = f64::NAN;
            if !v.is_nan() {
                idxs.push(i);
                keys[i] = mono_key32(v);
            }
        }
        let n_valid = idxs.len();
        if n_valid == 0 {
            continue;
        }
        radix_sort_order32(&keys, &mut idxs, &mut tmp);
        let mut i = 0;
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

pub(crate) fn median_inplace(v: &mut [f64]) -> f64 {
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

pub(crate) fn has_ge_n_unique(vals: &[f64], need: usize) -> bool {
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

pub(crate) fn ols2(ys: &[f64], bs: &[f64]) -> (f64, f64) {
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

// ==================== 生产 O2 预计算（orders + per_date X'X）====================

pub struct V2Shared {
    pub base: NeuShared,
    /// 5 张 (T,N) 排序: 0=ind2 1=ind1 2=ind0 (fill_ind_reg), 3=ind2 4=ind1 (中位填充)
    pub orders: Vec<Array2<usize>>,
    /// 逐日 (p, valid_idx, valid_cols, xtx)
    pub per_date: Vec<(usize, Vec<u32>, Vec<i32>, Vec<f64>)>,
    /// O3c: 每日 Cholesky 预分解（与生产 Cholesky::new(X'X) 同位）
    pub chols: Vec<Option<Cholesky<f64, Dyn>>>,
    /// O3d: 每日连续 X（valid 顺序 × 10 风格列，行主序）
    pub xdays: Vec<Vec<f64>>,
}

pub fn v2_precompute(base: NeuShared) -> V2Shared {
    let (t, n) = base.industry.dim();
    let ind0 = base.ind1.map(|&v| if v.is_nan() { 0.0 } else { 1.0 });
    let levels = [&base.ind2, &base.ind1, &ind0, &base.ind2, &base.ind1];
    let mut orders = Vec::with_capacity(5);
    for codes in levels {
        let mut ord = Array2::<usize>::zeros((t, n));
        for idx in 0..t {
            let mut order: Vec<usize> = (0..n).collect();
            let keys: Vec<u64> = codes.row(idx).iter().map(|&v| mono_key(v)).collect();
            let mut tmp: Vec<usize> = Vec::new();
            radix_sort_order(&keys, &mut order, &mut tmp);
            for j in 0..n {
                ord[[idx, j]] = order[j];
            }
        }
        orders.push(ord);
    }
    let k = 10usize;
    let mut per_date = Vec::with_capacity(t);
    let mut chols = Vec::with_capacity(t);
    let mut xdays = Vec::with_capacity(t);
    for idx in 0..t {
        let mut ind_codes: Vec<f64> = Vec::new();
        for j in 0..n {
            let c = base.ind1[[idx, j]];
            if !c.is_nan() && !ind_codes.contains(&c) {
                ind_codes.push(c);
            }
        }
        ind_codes.sort_by(cmp_f64);
        let p = k + ind_codes.len();
        let mut valid_idx: Vec<u32> = Vec::new();
        let mut valid_cols: Vec<i32> = Vec::new();
        for j in 0..n {
            let ok = base.restrict_f64[[idx, j]].is_finite()
                && base.restrict_f64[[idx, j]] == 0.0
                && base.barra_ranked.iter().all(|b| b[[idx, j]].is_finite());
            if ok {
                valid_idx.push(j as u32);
                let c_ind = base.ind1[[idx, j]];
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
        let mut xd = vec![0.0f64; n_valid * k];
        for (pos, &j) in valid_idx.iter().enumerate() {
            let ji = j as usize;
            for c in 0..k {
                let b = base.barra_ranked[c][[idx, ji]];
                xd[pos * k + c] = b;
                xtx[c * p + c] += b * b;
                for c2 in (c + 1)..k {
                    let v = b * base.barra_ranked[c2][[idx, ji]];
                    xtx[c * p + c2] += v;
                    xtx[c2 * p + c] += v;
                }
            }
            let ic = valid_cols[pos];
            if ic >= 0 {
                let col = k + ic as usize;
                xtx[col * p + col] += 1.0;
                for c in 0..k {
                    let b = base.barra_ranked[c][[idx, ji]];
                    xtx[c * p + col] += b;
                    xtx[col * p + c] += b;
                }
            }
        }
        let m = DMatrix::from_row_slice(p, p, &xtx);
        chols.push(Cholesky::new(m));
        per_date.push((p, valid_idx, valid_cols, xtx));
        xdays.push(xd);
    }
    V2Shared {
        base,
        orders,
        per_date,
        chols,
        xdays,
    }
}

// ==================== 生产填充（照抄 v2 pre 版）====================

fn fill_ind_reg_pre(
    fv: &mut Array2<f64>,
    ind2: &Array2<f64>,
    ind1: &Array2<f64>,
    size_ranked: &Array2<f64>,
    orders: &[Array2<usize>],
    skip_clean_segments: bool,
) {
    let (t, n) = fv.dim();
    let ind0 = ind1.map(|&v| if v.is_nan() { 0.0 } else { 1.0 });
    let levels = [ind2, ind1, &ind0];
    let mut ys: Vec<f64> = Vec::with_capacity(n);
    let mut bs: Vec<f64> = Vec::with_capacity(n);
    for (li, level) in levels.iter().enumerate() {
        for idx in 0..t {
            let mut row = fv.row(idx).to_vec();
            if !has_ge_n_unique(&row, 10) {
                continue;
            }
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
                // O3f: 段内无填充目标（所有位置 row 与 size 都有限）时，
                // 回归结果不会被使用，跳过收集与回归（逐位一致）。
                if skip_clean_segments {
                    let mut has_target = false;
                    for &ci in &order[seg_start..seg_end] {
                        if row[ci].is_nan() || size_ranked[[idx, ci]].is_nan() {
                            has_target = true;
                            break;
                        }
                    }
                    if !has_target {
                        seg_start = seg_end;
                        continue;
                    }
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
}

/// O3g: 无整行拷贝 + 缓冲复用版（与 pre 版逐位一致：分组分段互不重叠，
/// 行内直接读写与先拷贝后回写等价；obs/nan_mask/sv 缓冲复用不改变数值）。
fn fill_ind_reg_fast(
    fv: &mut Array2<f64>,
    ind2: &Array2<f64>,
    ind1: &Array2<f64>,
    size_ranked: &Array2<f64>,
    orders: &[Array2<usize>],
) {
    let (t, n) = fv.dim();
    let ind0 = ind1.map(|&v| if v.is_nan() { 0.0 } else { 1.0 });
    let levels = [ind2, ind1, &ind0];
    let mut ys: Vec<f64> = Vec::with_capacity(n);
    let mut bs: Vec<f64> = Vec::with_capacity(n);
    let mut obs: Vec<bool> = Vec::with_capacity(n);
    for (li, level) in levels.iter().enumerate() {
        for idx in 0..t {
            let mut row_view = fv.row_mut(idx);
            let row = row_view.as_slice_mut().unwrap();
            if !has_ge_n_unique(row, 10) {
                continue;
            }
            let order_arr = orders[li].row(idx);
            let order = order_arr.as_slice().unwrap();
            let size_row = size_ranked.row(idx);
            let size_row = size_row.as_slice().unwrap();
            let level_row = level.row(idx);
            let level_row = level_row.as_slice().unwrap();
            let mut seg_start = 0usize;
            while seg_start < n {
                let code = level_row[order[seg_start]];
                if code.is_nan() {
                    break;
                }
                let mut seg_end = seg_start + 1;
                while seg_end < n && level_row[order[seg_end]] == code {
                    seg_end += 1;
                }
                let mut has_target = false;
                for &ci in &order[seg_start..seg_end] {
                    if row[ci].is_nan() || size_row[ci].is_nan() {
                        has_target = true;
                        break;
                    }
                }
                if !has_target {
                    seg_start = seg_end;
                    continue;
                }
                ys.clear();
                bs.clear();
                obs.clear();
                for &ci in &order[seg_start..seg_end] {
                    let ok = !row[ci].is_nan() && !size_row[ci].is_nan();
                    obs.push(ok);
                    if ok {
                        ys.push(row[ci]);
                        bs.push(size_row[ci]);
                    }
                }
                if ys.len() >= 10 {
                    let (c0, c1) = ols2(&ys, &bs);
                    for (mi, &ci) in order[seg_start..seg_end].iter().enumerate() {
                        if !obs[mi] {
                            row[ci] = c0 + c1 * size_row[ci];
                        }
                    }
                }
                seg_start = seg_end;
            }
        }
    }
}

/// O3g: group_median_fill 的无整行拷贝 + 干净段跳过版。
fn group_median_fill_fast(
    values: &mut Array2<f64>,
    codes: &Array2<f64>,
    valid_mask: Option<&Array2<f64>>,
    orders: &Array2<usize>,
) {
    let (t, n) = values.dim();
    let mut sv: Vec<f64> = Vec::with_capacity(n);
    let mut nan_mask: Vec<bool> = Vec::with_capacity(n);
    for idx in 0..t {
        let mut row_view = values.row_mut(idx);
        let row = row_view.as_slice_mut().unwrap();
        nan_mask.clear();
        let mut has_nan = false;
        for j in 0..n {
            let nn = row[j].is_nan();
            nan_mask.push(nn);
            if nn {
                has_nan = true;
            }
        }
        if !has_nan {
            continue;
        }
        let order_arr = orders.row(idx);
        let order = order_arr.as_slice().unwrap();
        let codes_row = codes.row(idx);
        let codes_row = codes_row.as_slice().unwrap();
        let valid_row = valid_mask.map(|vm| vm.row(idx));
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
            let mut has_target = false;
            for &ci in &order[seg_start..seg_end] {
                let valid = valid_row.as_ref().map_or(true, |vr| vr[ci] == 1.0);
                if nan_mask[ci] && valid {
                    has_target = true;
                    break;
                }
            }
            if !has_target {
                seg_start = seg_end;
                continue;
            }
            sv.clear();
            for &ci in &order[seg_start..seg_end] {
                let valid = valid_row.as_ref().map_or(true, |vr| vr[ci] == 1.0);
                if valid && !row[ci].is_nan() {
                    sv.push(row[ci]);
                }
            }
            if !sv.is_empty() {
                let med = median_inplace(&mut sv);
                for &ci in &order[seg_start..seg_end] {
                    let valid = valid_row.as_ref().map_or(true, |vr| vr[ci] == 1.0);
                    if nan_mask[ci] && valid {
                        row[ci] = med;
                    }
                }
            }
            seg_start = seg_end;
        }
    }
}

fn group_median_fill_pre(
    values: &mut Array2<f64>,
    codes: &Array2<f64>,
    valid_mask: Option<&Array2<f64>>,
    orders: &Array2<usize>,
    skip_clean_segments: bool,
) {
    let (t, n) = values.dim();
    let mut sv: Vec<f64> = Vec::with_capacity(n);
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
            // O3f: 段内没有 (NaN 且 valid) 的填充目标时，中位数不会被使用，
            // 跳过收集与 quickselect（逐位一致）。
            if skip_clean_segments {
                let mut has_target = false;
                for &ci in &order[seg_start..seg_end] {
                    let valid = valid_mask.map_or(true, |vm| vm[[idx, ci]] == 1.0);
                    if nan_mask[ci] && valid {
                        has_target = true;
                        break;
                    }
                }
                if !has_target {
                    seg_start = seg_end;
                    continue;
                }
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

// ==================== 残差回归：生产复刻 vs 优化 ====================

/// 生产 ols_day_inline 照抄（any_nan 回退路径）。
fn ols_day_inline(
    fv_filled: &Array2<f64>,
    barra_ranked: &[Array2<f64>],
    ind1: &Array2<f64>,
    idx: usize,
) -> Vec<f64> {
    let n = fv_filled.ncols();
    let k = barra_ranked.len();
    let mut resid_row = vec![f64::NAN; n];
    let mut ind_codes: Vec<f64> = Vec::with_capacity(40);
    for j in 0..n {
        let c = ind1[[idx, j]];
        if !c.is_nan() && !ind_codes.contains(&c) {
            ind_codes.push(c);
        }
    }
    ind_codes.sort_by(cmp_f64);
    let n_ind = ind_codes.len();
    let p = k + n_ind;
    let mut valid: Vec<bool> = Vec::with_capacity(n);
    let mut rows: Vec<[f64; 11]> = Vec::with_capacity(n);
    let mut ind_cols: Vec<i32> = Vec::with_capacity(n);
    for j in 0..n {
        let ok = fv_filled[[idx, j]].is_finite()
            && barra_ranked.iter().all(|b| b[[idx, j]].is_finite());
        valid.push(ok);
        if ok {
            let mut cur = [0.0_f64; 11];
            cur[0] = fv_filled[[idx, j]];
            for c in 0..k {
                cur[c + 1] = barra_ranked[c][[idx, j]];
            }
            rows.push(cur);
            let c = ind1[[idx, j]];
            ind_cols.push(if c.is_nan() {
                -1
            } else {
                match ind_codes.binary_search_by(|x| cmp_f64(x, &c)) {
                    Ok(pos) => pos as i32,
                    Err(_) => -1,
                }
            });
        }
    }
    let n_valid = rows.len();
    if n_valid <= 10 {
        return resid_row;
    }
    let mut uniq: Vec<f64> = rows.iter().map(|r| r[0]).collect();
    uniq.sort_by(cmp_f64);
    uniq.dedup_by(|a, b| (a.is_nan() && b.is_nan()) || a == b);
    if uniq.len() == 1 {
        for j in 0..n {
            if valid[j] {
                resid_row[j] = 0.5;
            }
        }
        return resid_row;
    }
    let mut xtx: Vec<f64> = vec![0.0f64; p * p];
    let mut xty: Vec<f64> = vec![0.0f64; p];
    for (i, r) in rows.iter().enumerate() {
        let yv = r[0];
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
    let m = DMatrix::from_row_slice(p, p, &xtx);
    let rhs = DMatrix::from_column_slice(p, 1, &xty);
    let use_svd = n_valid <= 40 || Cholesky::new(m.clone()).is_none();
    let coef: Vec<f64> = if !use_svd {
        let chol = Cholesky::new(m).expect("chol");
        chol.solve(&rhs).column(0).iter().copied().collect()
    } else {
        let n_r = rows.len();
        let xm = DMatrix::from_fn(n_r, p, |r_i, c| {
            let ic = ind_cols[r_i];
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
        for c in 0..k {
            pred += coef[c] * r[c + 1];
        }
        let ic = ind_cols[i];
        if ic >= 0 {
            pred += coef[k + ic as usize];
        }
        while !valid[vi] {
            vi += 1;
        }
        resid_row[vi] = r[0] - pred;
        vi += 1;
    }
    resid_row
}

#[derive(Default, Clone, Copy)]
pub struct ResidTimes {
    pub uniq: f64,
    pub xty: f64,
    pub chol: f64,
    pub pred: f64,
}

/// 生产 get_residual_v2 照抄（带子阶段计时）。
fn get_residual_v2_replica(
    fv_filled: &Array2<f64>,
    barra_ranked: &[Array2<f64>],
    ind1: &Array2<f64>,
    per_date: &[(usize, Vec<u32>, Vec<i32>, Vec<f64>)],
    times: &mut ResidTimes,
) -> Array2<f64> {
    let (t, n) = fv_filled.dim();
    let mut resid = Array2::<f64>::from_elem((t, n), f64::NAN);
    let k = 10usize;
    for idx in 0..t {
        let (p, valid_idx, valid_cols, xtx_pre) = &per_date[idx];
        if *p == 0 {
            continue;
        }
        let s = Instant::now();
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
            let row = ols_day_inline(fv_filled, barra_ranked, ind1, idx);
            for j in 0..n {
                resid[[idx, j]] = row[j];
            }
            times.uniq += s.elapsed().as_secs_f64();
            continue;
        }
        uniq.sort_by(cmp_f64);
        uniq.dedup_by(|a, b| (a.is_nan() && b.is_nan()) || a == b);
        if uniq.len() == 1 {
            for &j in valid_idx {
                resid[[idx, j as usize]] = 0.5;
            }
            times.uniq += s.elapsed().as_secs_f64();
            continue;
        }
        times.uniq += s.elapsed().as_secs_f64();
        let s = Instant::now();
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
        times.xty += s.elapsed().as_secs_f64();
        let s = Instant::now();
        let m = DMatrix::from_row_slice(*p, *p, xtx_pre);
        let rhs = DMatrix::from_column_slice(*p, 1, &xty);
        let use_svd = valid_idx.len() <= 40 || Cholesky::new(m.clone()).is_none();
        let coef: Vec<f64> = if !use_svd {
            let chol = Cholesky::new(m).expect("chol");
            chol.solve(&rhs).column(0).iter().copied().collect()
        } else {
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
        times.chol += s.elapsed().as_secs_f64();
        let s = Instant::now();
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
        }
        times.pred += s.elapsed().as_secs_f64();
    }
    resid
}

/// 优化残差: O3a(min==max) + O3c(预分解 Cholesky) + O3d(连续 X)。
/// any_nan / valid<=40 / Cholesky 失败日回退生产路径，保证逐位一致。
fn get_residual_v2_opt(
    fv_filled: &Array2<f64>,
    barra_ranked: &[Array2<f64>],
    ind1: &Array2<f64>,
    shared: &V2Shared,
    times: &mut ResidTimes,
) -> Array2<f64> {
    let (t, n) = fv_filled.dim();
    let mut resid = Array2::<f64>::from_elem((t, n), f64::NAN);
    let k = 10usize;
    let mut y_buf: Vec<f64> = Vec::with_capacity(n);
    let mut xty: Vec<f64> = Vec::with_capacity(64);
    for idx in 0..t {
        let (p, valid_idx, valid_cols, _xtx_pre) = &shared.per_date[idx];
        if *p == 0 {
            continue;
        }
        let s = Instant::now();
        y_buf.clear();
        let mut any_nan = false;
        let mut mn = f64::INFINITY;
        let mut mx = f64::NEG_INFINITY;
        for &j in valid_idx {
            let y = fv_filled[[idx, j as usize]];
            if !y.is_finite() {
                any_nan = true;
                break;
            }
            if y < mn {
                mn = y;
            }
            if y > mx {
                mx = y;
            }
            y_buf.push(y);
        }
        if any_nan {
            let row = ols_day_inline(fv_filled, barra_ranked, ind1, idx);
            for j in 0..n {
                resid[[idx, j]] = row[j];
            }
            times.uniq += s.elapsed().as_secs_f64();
            continue;
        }
        if mn == mx {
            for &j in valid_idx {
                resid[[idx, j as usize]] = 0.5;
            }
            times.uniq += s.elapsed().as_secs_f64();
            continue;
        }
        times.uniq += s.elapsed().as_secs_f64();
        let s = Instant::now();
        xty.clear();
        xty.resize(*p, 0.0);
        let xd = &shared.xdays[idx];
        for (pos, &yv) in y_buf.iter().enumerate() {
            let xrow = &xd[pos * k..pos * k + k];
            for c in 0..k {
                xty[c] += xrow[c] * yv;
            }
            let ic = valid_cols[pos];
            if ic >= 0 {
                xty[k + ic as usize] += yv;
            }
        }
        times.xty += s.elapsed().as_secs_f64();
        let s = Instant::now();
        let use_svd = valid_idx.len() <= 40 || shared.chols[idx].is_none();
        let coef: Vec<f64> = if !use_svd {
            let chol = shared.chols[idx].as_ref().unwrap();
            let rhs = DMatrix::from_column_slice(*p, 1, &xty);
            chol.solve(&rhs).column(0).iter().copied().collect()
        } else {
            // 回退生产 SVD 路径
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
        times.chol += s.elapsed().as_secs_f64();
        let s = Instant::now();
        let xd = &shared.xdays[idx];
        for (pos, &j) in valid_idx.iter().enumerate() {
            let ji = j as usize;
            let yv = y_buf[pos];
            let xrow = &xd[pos * k..pos * k + k];
            let mut pred = 0.0;
            for c in 0..k {
                pred += coef[c] * xrow[c];
            }
            let ic = valid_cols[pos];
            if ic >= 0 {
                pred += coef[k + ic as usize];
            }
            resid[[idx, ji]] = yv - pred;
        }
        times.pred += s.elapsed().as_secs_f64();
    }
    resid
}

// ==================== slot 级入口 ====================

#[derive(Default, Clone, Copy)]
pub struct V2Times {
    pub cast: f64,
    pub rank1: f64,
    pub fill_ind: f64,
    pub mask_clone: f64,
    pub fills: f64,
    pub restrict: f64,
    pub rank2: f64,
    pub resid: ResidTimes,
    pub convert: f64,
}

impl V2Times {
    pub fn total(&self) -> f64 {
        self.cast
            + self.rank1
            + self.fill_ind
            + self.mask_clone
            + self.fills
            + self.restrict
            + self.rank2
            + self.resid.uniq
            + self.resid.xty
            + self.resid.chol
            + self.resid.pred
            + self.convert
    }
}

/// mode: "replica"=生产 v2_resid 照抄; "opt"=O3a/c/d; "opt_rank32"=再加 O3e
pub fn v2_slot(
    slot: ArrayView2<'_, f32>,
    shared: &V2Shared,
    mode: &str,
) -> (Array2<f32>, V2Times) {
    let mut tt = V2Times::default();
    let (n_dates, n_stocks) = slot.dim();
    let base = &shared.base;

    let s = Instant::now();
    let mut fv_ranked;
    if mode == "opt_rank32" || mode == "opt2" {
        fv_ranked = Array2::<f64>::zeros((n_dates, n_stocks));
        tt.cast = s.elapsed().as_secs_f64();
        let s = Instant::now();
        rank_pct_all_from_f32(&slot, &mut fv_ranked);
        tt.rank1 = s.elapsed().as_secs_f64();
    } else {
        fv_ranked = slot.map(|&v| v as f64);
        tt.cast = s.elapsed().as_secs_f64();
        let s = Instant::now();
        rank_pct_all(&mut fv_ranked);
        tt.rank1 = s.elapsed().as_secs_f64();
    }

    let fast = mode == "opt2";
    let s = Instant::now();
    if fast {
        fill_ind_reg_fast(
            &mut fv_ranked,
            &base.ind2,
            &base.ind1,
            &base.size_ranked,
            &shared.orders[0..3],
        );
    } else {
        fill_ind_reg_pre(
            &mut fv_ranked,
            &base.ind2,
            &base.ind1,
            &base.size_ranked,
            &shared.orders[0..3],
            false,
        );
    }
    tt.fill_ind = s.elapsed().as_secs_f64();

    let s = Instant::now();
    for i in 0..(n_dates * n_stocks) {
        if base.ind1.as_slice().unwrap()[i].is_nan() {
            fv_ranked.as_slice_mut().unwrap()[i] = f64::NAN;
        }
    }
    let mut fv_filled = fv_ranked.clone();
    tt.mask_clone = s.elapsed().as_secs_f64();

    let s = Instant::now();
    if fast {
        group_median_fill_fast(&mut fv_filled, &base.ind2, None, &shared.orders[3]);
        group_median_fill_fast(&mut fv_filled, &base.ind1, None, &shared.orders[4]);
        group_median_fill_fast(
            &mut fv_filled,
            &base.zeros,
            Some(&base.ind1_mask),
            &shared.orders[0],
        );
    } else {
        group_median_fill_pre(&mut fv_filled, &base.ind2, None, &shared.orders[3], false);
        group_median_fill_pre(&mut fv_filled, &base.ind1, None, &shared.orders[4], false);
        group_median_fill_pre(
            &mut fv_filled,
            &base.zeros,
            Some(&base.ind1_mask),
            &shared.orders[0],
            false,
        );
    }
    tt.fills = s.elapsed().as_secs_f64();
    drop(fv_ranked);

    let s = Instant::now();
    for i in 0..(n_dates * n_stocks) {
        if base.restrict_f64.as_slice().unwrap()[i] != 0.0 {
            fv_filled.as_slice_mut().unwrap()[i] = f64::NAN;
        }
    }
    tt.restrict = s.elapsed().as_secs_f64();

    let s = Instant::now();
    rank_pct_all(&mut fv_filled);
    tt.rank2 = s.elapsed().as_secs_f64();

    let resid = if mode == "replica" {
        get_residual_v2_replica(
            &fv_filled,
            &base.barra_ranked,
            &base.ind1,
            &shared.per_date,
            &mut tt.resid,
        )
    } else {
        get_residual_v2_opt(
            &fv_filled,
            &base.barra_ranked,
            &base.ind1,
            shared,
            &mut tt.resid,
        )
    };

    let s = Instant::now();
    let mut output = Array2::<f32>::from_elem((n_dates, n_stocks), f32::NAN);
    let out_slice = output.as_slice_mut().unwrap();
    let rr = resid.as_slice().unwrap();
    for i in 0..(n_dates * n_stocks) {
        let v = rr[i];
        out_slice[i] = if v.is_nan() { f32::NAN } else { v as f32 };
    }
    tt.convert = s.elapsed().as_secs_f64();
    (output, tt)
}

pub fn bitwise_equal(a: &Array2<f32>, b: &Array2<f32>) -> (bool, usize) {
    let mut mismatch = 0usize;
    for (x, y) in a.iter().zip(b.iter()) {
        if x.to_bits() != y.to_bits() && !(x.is_nan() && y.is_nan()) {
            mismatch += 1;
        }
    }
    (mismatch == 0, mismatch)
}

// ==================== 多线程扩展性测试 ====================

/// W 个线程并发做 slot 中性化（每线程 rounds 次），返回 (每 slot 平均耗时, 总吞吐 slot/s)。
pub fn mt_bench(
    shared: &Arc<V2Shared>,
    slots: &Arc<Vec<Array2<f32>>>,
    workers: usize,
    rounds: usize,
    mode: &'static str,
) -> (f64, f64) {
    let t0 = Instant::now();
    let mut handles = Vec::with_capacity(workers);
    for w in 0..workers {
        let sh = shared.clone();
        let sl = slots.clone();
        handles.push(std::thread::spawn(move || {
            let mut acc = 0.0f64;
            for r in 0..rounds {
                let slot = &sl[(w + r) % sl.len()];
                let s = Instant::now();
                let (out, _) = v2_slot(slot.view(), &sh, mode);
                acc += s.elapsed().as_secs_f64();
                std::hint::black_box(&out);
            }
            acc / rounds as f64
        }));
    }
    let per_slot: Vec<f64> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    let wall = t0.elapsed().as_secs_f64();
    let mean = per_slot.iter().sum::<f64>() / per_slot.len() as f64;
    let throughput = (workers * rounds) as f64 / wall;
    (mean, throughput)
}

// ==================== 数据加载 + 驱动 ====================

pub fn build_shared(data_dir: &str) -> V2Shared {
    let barra_raw = crate::npy::as_f64_3d(crate::npy::load(&format!("{data_dir}/barra_raw.npy")));
    let industry = crate::npy::as_f64_mat(crate::npy::load(&format!("{data_dir}/industry.npy")));
    let restrict = crate::npy::as_f32_mat(crate::npy::load(&format!("{data_dir}/restrict.npy")));
    let base = neu_precompute(&industry, &restrict, &barra_raw);
    v2_precompute(base)
}

fn fmt_times(t: &V2Times) -> String {
    format!(
        "total={:.3}s [cast={:.3} rank1={:.3} fill_ind={:.3} mask_clone={:.3} fills={:.3} restrict={:.3} rank2={:.3} resid(uniq={:.3} xty={:.3} chol={:.3} pred={:.3}) convert={:.3}]",
        t.total(), t.cast, t.rank1, t.fill_ind, t.mask_clone, t.fills, t.restrict, t.rank2,
        t.resid.uniq, t.resid.xty, t.resid.chol, t.resid.pred, t.convert,
    )
}

pub fn run_v2_bench(data_dir: &str) {
    let s = Instant::now();
    let shared = build_shared(data_dir);
    println!("v2 precompute (orders+X'X+chol+X): {:.2}s", s.elapsed().as_secs_f64());

    let names = std::fs::read_to_string(format!("{data_dir}/sample_names.txt")).unwrap();
    let restrict_m = crate::npy::as_f32_mat(crate::npy::load(&format!("{data_dir}/restrict.npy")));
    for nm in names.lines() {
        let raw = crate::npy::as_f32_mat(crate::npy::load(&format!("{data_dir}/factor_{nm}.npy")));
        let ranked = crate::engine::rank_and_fill_missing_cross_sectional_median(&raw, &restrict_m);
        // slot 集合: ranked 本身 + w=5 的 4 个滚动派生（代表 13 slot 的形态多样性）
        let (mean5, max5, min5, std5) = crate::engine::rolling_stats_f32_serial(&ranked, 5, 2);
        let slots: Vec<(&str, &Array2<f32>)> = vec![
            ("smooth_1", &ranked),
            ("mean_5", &mean5),
            ("max_5", &max5),
            ("min_5", &min5),
            ("std_5", &std5),
        ];
        for (tag, slot) in &slots {
            let (out_rep, t_rep) = v2_slot(slot.view(), &shared, "replica");
            let (out_opt, t_opt) = v2_slot(slot.view(), &shared, "opt");
            let (out_o32, t_o32) = v2_slot(slot.view(), &shared, "opt_rank32");
            let (out_o2, t_o2) = v2_slot(slot.view(), &shared, "opt2");
            let (eq1, mm1) = bitwise_equal(&out_rep, &out_opt);
            let (eq2, mm2) = bitwise_equal(&out_rep, &out_o32);
            let (eq3, mm3) = bitwise_equal(&out_rep, &out_o2);
            println!("SLOT {nm}::{tag}");
            println!("  replica    {}", fmt_times(&t_rep));
            println!("  opt        {}  speedup={:.2}x bitwise={}({})", fmt_times(&t_opt), t_rep.total() / t_opt.total(), eq1, mm1);
            println!("  opt_rank32 {}  speedup={:.2}x bitwise={}({})", fmt_times(&t_o32), t_rep.total() / t_o32.total(), eq2, mm2);
            println!("  opt2       {}  speedup={:.2}x bitwise={}({})", fmt_times(&t_o2), t_rep.total() / t_o2.total(), eq3, mm3);
        }
    }
}

pub fn run_v2_mt(data_dir: &str, workers_list: &[usize], rounds: usize) {
    let restrict_m = crate::npy::as_f32_mat(crate::npy::load(&format!("{data_dir}/restrict.npy")));
    let s = Instant::now();
    let shared = Arc::new(build_shared(data_dir));
    println!("v2 precompute: {:.2}s", s.elapsed().as_secs_f64());

    let names = std::fs::read_to_string(format!("{data_dir}/sample_names.txt")).unwrap();
    let mut slots: Vec<Array2<f32>> = Vec::new();
    for nm in names.lines() {
        let raw = crate::npy::as_f32_mat(crate::npy::load(&format!("{data_dir}/factor_{nm}.npy")));
        let ranked = crate::engine::rank_and_fill_missing_cross_sectional_median(&raw, &restrict_m);
        let (mean5, _max5, _min5, std5) = crate::engine::rolling_stats_f32_serial(&ranked, 5, 2);
        slots.push(ranked);
        slots.push(mean5);
        slots.push(std5);
    }
    let slots = Arc::new(slots);

    for &w in workers_list {
        for mode in ["replica", "opt2"] {
            let (per_slot, tp) = mt_bench(&shared, &slots, w, rounds, mode);
            println!(
                "MT workers={w:3} mode={mode:7} per_slot={per_slot:.3}s throughput={tp:.2} slot/s"
            );
        }
    }
}

// ==================== P3: 按日期小批量多面 (2026-09-01) ====================
// 设计: 一个批次取 B 个面, 外层循环日期, 内层循环面。
// 每日装载一次共享行 (行业码/排序/风格/restrict/有效集), B 个面共用;
// 面的 fv 行按日期流式处理, 每日工作集 B×~72KB×2 + 共享 ~1.5MB, 全部驻留缓存。
// 语义与 per-face 逐位一致: 每个 (面, 日期) 的操作序列与生产完全相同,
// 位序无关性由"每日序列不变"保证 (见注释)。
// 内存流量/面: ~4.9GB → ~2.4GB (B=4); 共享行加载摊销 B 倍。

/// 行级 rank (f32 源, u32 4-pass radix)。与 rank_pct_all_from_f32 逐位一致。
pub(crate) fn rank_pct_row_from_f32_in(
    in_row: &[f32],
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
    for (i, &v) in in_row.iter().enumerate() {
        if !v.is_nan() {
            idxs.push(i);
            keys[i] = mono_key32(v);
        }
    }
    let n_valid = idxs.len();
    if n_valid == 0 {
        return;
    }
    radix_sort_order32(&keys, idxs, tmp);
    let mut i = 0;
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

/// 行级 fill_ind_reg (3 级顺序就地执行, 每级判断唯一值/跳过干净段)。
/// 与生产 fill_ind_reg_pre 对同一行状态的操作序列一致 (级别内段不重叠→就地等价)。
#[allow(clippy::too_many_arguments)]
pub(crate) fn fill_ind_reg_row(
    row: &mut [f64],
    level_rows: [&[f64]; 3],
    size_row: &[f64],
    order_rows: [&[usize]; 3],
    ys: &mut Vec<f64>,
    bs: &mut Vec<f64>,
    obs: &mut Vec<bool>,
) {
    let n = row.len();
    for li in 0..3 {
        if !has_ge_n_unique(row, 10) {
            continue;
        }
        let level_row = level_rows[li];
        let order = order_rows[li];
        let mut seg_start = 0usize;
        while seg_start < n {
            let code = level_row[order[seg_start]];
            if code.is_nan() {
                break;
            }
            let mut seg_end = seg_start + 1;
            while seg_end < n && level_row[order[seg_end]] == code {
                seg_end += 1;
            }
            let mut has_target = false;
            for &ci in &order[seg_start..seg_end] {
                if row[ci].is_nan() || size_row[ci].is_nan() {
                    has_target = true;
                    break;
                }
            }
            if !has_target {
                seg_start = seg_end;
                continue;
            }
            ys.clear();
            bs.clear();
            obs.clear();
            for &ci in &order[seg_start..seg_end] {
                let ok = !row[ci].is_nan() && !size_row[ci].is_nan();
                obs.push(ok);
                if ok {
                    ys.push(row[ci]);
                    bs.push(size_row[ci]);
                }
            }
            if ys.len() >= 10 {
                let (c0, c1) = ols2(&ys, &bs);
                for (mi, &ci) in order[seg_start..seg_end].iter().enumerate() {
                    if !obs[mi] {
                        row[ci] = c0 + c1 * size_row[ci];
                    }
                }
            }
            seg_start = seg_end;
        }
    }
}

/// 行级中位填充 (单级; nan_mask 由调用方按"当前行状态"构建, 与生产每 call 重建一致)。
pub(crate) fn median_fill_level_row(
    row: &mut [f64],
    codes_row: &[f64],
    valid_mask_row: Option<&[f64]>,
    order: &[usize],
    nan_mask: &[bool],
    sv: &mut Vec<f64>,
) {
    let n = row.len();
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
        let mut has_target = false;
        for &ci in &order[seg_start..seg_end] {
            let valid = valid_mask_row.map_or(true, |vr| vr[ci] == 1.0);
            if nan_mask[ci] && valid {
                has_target = true;
                break;
            }
        }
        if !has_target {
            seg_start = seg_end;
            continue;
        }
        sv.clear();
        for &ci in &order[seg_start..seg_end] {
            let valid = valid_mask_row.map_or(true, |vr| vr[ci] == 1.0);
            if valid && !row[ci].is_nan() {
                sv.push(row[ci]);
            }
        }
        if !sv.is_empty() {
            let med = median_inplace(sv);
            for &ci in &order[seg_start..seg_end] {
                let valid = valid_mask_row.map_or(true, |vr| vr[ci] == 1.0);
                if nan_mask[ci] && valid {
                    row[ci] = med;
                }
            }
        }
        seg_start = seg_end;
    }
}

/// 生产 ols_day_inline 的行级移植 (回退路径, 逐位一致)。
fn ols_day_inline_row(
    fv_row: &[f64],
    barra_row: &[&[f64]],
    ind1_row: &[f64],
) -> Vec<f64> {
    let n = fv_row.len();
    let k = barra_row.len();
    let mut resid_row = vec![f64::NAN; n];
    let mut ind_codes: Vec<f64> = Vec::with_capacity(40);
    for j in 0..n {
        let c = ind1_row[j];
        if !c.is_nan() && !ind_codes.contains(&c) {
            ind_codes.push(c);
        }
    }
    ind_codes.sort_by(cmp_f64);
    let n_ind = ind_codes.len();
    let p = k + n_ind;
    let mut valid: Vec<bool> = Vec::with_capacity(n);
    let mut rows: Vec<[f64; 11]> = Vec::with_capacity(n);
    let mut ind_cols: Vec<i32> = Vec::with_capacity(n);
    for j in 0..n {
        let ok = fv_row[j].is_finite() && barra_row.iter().all(|b| b[j].is_finite());
        valid.push(ok);
        if ok {
            let mut cur = [0.0_f64; 11];
            cur[0] = fv_row[j];
            for c in 0..k {
                cur[c + 1] = barra_row[c][j];
            }
            rows.push(cur);
            let c = ind1_row[j];
            ind_cols.push(if c.is_nan() {
                -1
            } else {
                match ind_codes.binary_search_by(|x| cmp_f64(x, &c)) {
                    Ok(pos) => pos as i32,
                    Err(_) => -1,
                }
            });
        }
    }
    let n_valid = rows.len();
    if n_valid <= 10 {
        return resid_row;
    }
    let mut uniq: Vec<f64> = rows.iter().map(|r| r[0]).collect();
    uniq.sort_by(cmp_f64);
    uniq.dedup_by(|a, b| (a.is_nan() && b.is_nan()) || a == b);
    if uniq.len() == 1 {
        for j in 0..n {
            if valid[j] {
                resid_row[j] = 0.5;
            }
        }
        return resid_row;
    }
    let mut xtx: Vec<f64> = vec![0.0f64; p * p];
    let mut xty: Vec<f64> = vec![0.0f64; p];
    for (i, r) in rows.iter().enumerate() {
        let yv = r[0];
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
    let m = DMatrix::from_row_slice(p, p, &xtx);
    let rhs = DMatrix::from_column_slice(p, 1, &xty);
    let use_svd = n_valid <= 40 || Cholesky::new(m.clone()).is_none();
    let coef: Vec<f64> = if !use_svd {
        let chol = Cholesky::new(m).expect("chol");
        chol.solve(&rhs).column(0).iter().copied().collect()
    } else {
        let n_r = rows.len();
        let xm = DMatrix::from_fn(n_r, p, |r_i, c| {
            let ic = ind_cols[r_i];
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
    let mut vi = 0usize;
    for (i, r) in rows.iter().enumerate() {
        let mut pred = 0.0;
        for c in 0..k {
            pred += coef[c] * r[c + 1];
        }
        let ic = ind_cols[i];
        if ic >= 0 {
            pred += coef[k + ic as usize];
        }
        while !valid[vi] {
            vi += 1;
        }
        resid_row[vi] = r[0] - pred;
        vi += 1;
    }
    resid_row
}

/// OLS 行级快路径: 与 get_residual_v2 (O3a/c/d) 完全一致的运算, 残差直接写 f32 行。
/// 返回该行是否走了回退路径。
#[allow(clippy::too_many_arguments)]
pub(crate) fn ols_day_row_fast(
    fv_row: &[f64],
    shared: &V2Shared,
    idx: usize,
    y_buf: &mut Vec<f64>,
    xty: &mut Vec<f64>,
    out_f32_row: &mut [f32],
) -> bool {
    let k = 10usize;
    let (p, valid_idx, valid_cols, _xtx) = &shared.per_date[idx];
    if *p == 0 {
        return false;
    }
    y_buf.clear();
    let mut any_nan = false;
    let mut mn = f64::INFINITY;
    let mut mx = f64::NEG_INFINITY;
    for &j in valid_idx {
        let y = fv_row[j as usize];
        if !y.is_finite() {
            any_nan = true;
            break;
        }
        if y < mn {
            mn = y;
        }
        if y > mx {
            mx = y;
        }
        y_buf.push(y);
    }
    if any_nan {
        let barra_v: Vec<_> = (0..k).map(|c| shared.base.barra_ranked[c].row(idx)).collect();
        let barra_row: Vec<&[f64]> = barra_v.iter().map(|r| r.as_slice().unwrap()).collect();
        let ind1_v = shared.base.ind1.row(idx);
        let ind1_row = ind1_v.as_slice().unwrap();
        let row = ols_day_inline_row(fv_row, &barra_row, ind1_row);
        for (j, &v) in row.iter().enumerate() {
            out_f32_row[j] = if v.is_nan() { f32::NAN } else { v as f32 };
        }
        return true;
    }
    ols_day_core(y_buf, shared, idx, out_f32_row);
    false
}

/// OLS 核心：生产 ols_day_row_fast 的 "mn==mx 之后" 段原样抽出，
/// 供行级路径与 V3 压缩路径共用（同一份代码 → 同一份 codegen → 逐位一致）。
pub(crate) fn ols_day_core(
    y_buf: &[f64],
    shared: &V2Shared,
    idx: usize,
    out_f32_row: &mut [f32],
) {
    let k = 10usize;
    let (p, valid_idx, valid_cols, _xtx) = &shared.per_date[idx];
    if *p == 0 {
        return;
    }
    let mut mn = f64::INFINITY;
    let mut mx = f64::NEG_INFINITY;
    for &y in y_buf {
        if y < mn {
            mn = y;
        }
        if y > mx {
            mx = y;
        }
    }
    if mn == mx {
        for &j in valid_idx {
            out_f32_row[j as usize] = 0.5;
        }
        return;
    }
    let mut xty = vec![0.0f64; *p];
    let xd = &shared.xdays[idx];
    for (pos, &yv) in y_buf.iter().enumerate() {
        let xrow = &xd[pos * k..pos * k + k];
        for c in 0..k {
            xty[c] += xrow[c] * yv;
        }
        let ic = valid_cols[pos];
        if ic >= 0 {
            xty[k + ic as usize] += yv;
        }
    }
    let use_svd = valid_idx.len() <= 40 || shared.chols[idx].is_none();
    let coef: Vec<f64> = if !use_svd {
        let chol = shared.chols[idx].as_ref().unwrap();
        let rhs = DMatrix::from_column_slice(*p, 1, &xty);
        chol.solve(&rhs).column(0).iter().copied().collect()
    } else {
        let n_r = valid_idx.len();
        let barra_v: Vec<_> = (0..k).map(|c| shared.base.barra_ranked[c].row(idx)).collect();
        let barra_row: Vec<&[f64]> = barra_v.iter().map(|r| r.as_slice().unwrap()).collect();
        let rows: Vec<[f64; 11]> = valid_idx
            .iter()
            .map(|&j| {
                let ji = j as usize;
                let mut cur = [0.0_f64; 11];
                cur[0] = y_buf[valid_idx.iter().position(|&x| x == j).unwrap()];
                for c in 0..k {
                    cur[c + 1] = barra_row[c][ji];
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
    for (pos, &j) in valid_idx.iter().enumerate() {
        let ji = j as usize;
        let yv = y_buf[pos];
        let xrow = &xd[pos * k..pos * k + k];
        let mut pred = 0.0;
        for c in 0..k {
            pred += coef[c] * xrow[c];
        }
        let ic = valid_cols[pos];
        if ic >= 0 {
            pred += coef[k + ic as usize];
        }
        out_f32_row[ji] = (yv - pred) as f32;
    }
}

/// B 个面按日期批处理的中性化 (C' 语义: 无最终残差 rank, 输出 f32)。
/// 与逐面 v2_slot(opt2) 逐位一致。
pub fn v2_slot_batch(slots: &[ArrayView2<f32>], shared: &V2Shared) -> (Vec<Array2<f32>>, u64) {
    let b = slots.len();
    let (t, n) = slots[0].dim();
    let mut outs: Vec<Array2<f32>> = (0..b).map(|_| Array2::from_elem((t, n), f32::NAN)).collect();
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
    let mut fallback_days = 0u64;
    let base = &shared.base;
    let mut ind0_row = Vec::<f64>::with_capacity(n);
    for idx in 0..t {
        // ---- 每日共享行 (B 面共用, 一次装载) ----
        let (ind2_v, ind1_v, size_v) = (base.ind2.row(idx), base.ind1.row(idx), base.size_ranked.row(idx));
        let (zeros_v, mask_v, restrict_v) = (base.zeros.row(idx), base.ind1_mask.row(idx), base.restrict_f64.row(idx));
        let (o0_v, o1_v, o2_v) = (shared.orders[0].row(idx), shared.orders[1].row(idx), shared.orders[2].row(idx));
        let (o3_v, o4_v) = (shared.orders[3].row(idx), shared.orders[4].row(idx));
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
        // ---- 每面: rank1 → fill(3级) → scrub (row 流式) ----
        for (f, slot) in slots.iter().enumerate() {
            let slot_row = slot.row(idx);
            rank_pct_row_from_f32_in(
                slot_row.as_slice().unwrap(),
                &mut pct[f],
                &mut idxs,
                &mut tmp,
                &mut keys32,
            );
            fill_ind_reg_row(
                &mut pct[f],
                [ind2_r, ind1_r, &ind0_row],
                size_r,
                [o0, o1, o2],
                &mut ys,
                &mut bs,
                &mut obs,
            );
            for (j, &v) in ind1_r.iter().enumerate() {
                if v.is_nan() {
                    pct[f][j] = f64::NAN;
                }
            }
            filled[f].copy_from_slice(&pct[f]);
        }
        // ---- 每面: 三级中位 (每级按当前行状态重建 nan_mask) → restrict → rank2 → OLS ----
        for f in 0..b {
            // 级1: ind2
            nan_mask.clear();
            for &v in filled[f].iter() {
                nan_mask.push(v.is_nan());
            }
            if nan_mask.iter().any(|&x| x) {
                median_fill_level_row(&mut filled[f], ind2_r, None, o3, &nan_mask, &mut sv);
            }
            // 级2: ind1
            nan_mask.clear();
            for &v in filled[f].iter() {
                nan_mask.push(v.is_nan());
            }
            if nan_mask.iter().any(|&x| x) {
                median_fill_level_row(&mut filled[f], ind1_r, None, o4, &nan_mask, &mut sv);
            }
            // 级3: 全市场 (valid_mask=ind1_mask, 单段)
            nan_mask.clear();
            for &v in filled[f].iter() {
                nan_mask.push(v.is_nan());
            }
            if nan_mask.iter().any(|&x| x) {
                median_fill_level_row(&mut filled[f], zeros_r, Some(mask_r), o0, &nan_mask, &mut sv);
            }
            for (j, &v) in restrict_r.iter().enumerate() {
                if v != 0.0 {
                    filled[f][j] = f64::NAN;
                }
            }
            rank_pct_row_f64_in_place(
                &mut filled[f],
                &mut ranks64,
                &mut idxs,
                &mut tmp,
                &mut keys64,
            );
            let mut out_row = outs[f].row_mut(idx);
            let fb = ols_day_row_fast(
                &filled[f],
                shared,
                idx,
                &mut y_buf,
                &mut xty,
                out_row.as_slice_mut().unwrap(),
            );
            if fb {
                fallback_days += 1;
            }
        }
    }
    (outs, fallback_days)
}

/// 行级 f64 rank (就地写回), 与 rank_pct_all 的逐行语义一致。
pub(crate) fn rank_pct_row_f64_in_place(
    row: &mut [f64],
    ranks: &mut Vec<f64>,
    idxs: &mut Vec<usize>,
    tmp: &mut Vec<usize>,
    keys: &mut Vec<u64>,
) {
    // 生产 rank_pct_row_into 从 vals 读入 idxs/keys 后写 ranks, 再拷回 row;
    // 就地时先取副本防读写交错 —— 直接复用 rank_pct_row_into 的读语义。
    let vals: Vec<f64> = row.to_vec();
    rank_pct_row_into(&vals, ranks, idxs, tmp, keys);
    row.copy_from_slice(ranks);
}

/// 批处理基准: 逐位对账 (vs per-face opt2) + 单线程计时。
pub fn run_v2_batch_bench(data_dir: &str) {
    let restrict_m = crate::npy::as_f32_mat(crate::npy::load(&format!("{data_dir}/restrict.npy")));
    let shared = build_shared(data_dir);
    let names = std::fs::read_to_string(format!("{data_dir}/sample_names.txt")).unwrap();
    let mut slots: Vec<Array2<f32>> = Vec::new();
    let mut slot_tags: Vec<String> = Vec::new();
    for nm in names.lines() {
        let raw = crate::npy::as_f32_mat(crate::npy::load(&format!("{data_dir}/factor_{nm}.npy")));
        let ranked = crate::engine::rank_and_fill_missing_cross_sectional_median(&raw, &restrict_m);
        let (mean5, max5, min5, std5) = crate::engine::rolling_stats_f32_serial(&ranked, 5, 2);
        for (tag, s) in [("smooth_1", &ranked), ("mean_5", &mean5), ("max_5", &max5), ("min_5", &min5), ("std_5", &std5)] {
            slots.push(s.clone());
            slot_tags.push(format!("{nm}::{tag}"));
        }
    }
    println!("pool slots: {}", slots.len());
    // per-face 参考
    let views: Vec<ArrayView2<f32>> = slots.iter().map(|s| s.view()).collect();
    let refs: Vec<Array2<f32>> = views
        .iter()
        .map(|v| {
            let (o, _) = v2_slot(v.clone(), &shared, "opt2");
            o
        })
        .collect();
    // 分组合并 per-face 计时 (近似: 取 group 内平均)
    let t0 = Instant::now();
    let mut perface = 0.0f64;
    for v in &views {
        let s = Instant::now();
        let _ = v2_slot(v.clone(), &shared, "opt2");
        perface += s.elapsed().as_secs_f64();
    }
    let perface_avg = perface / views.len() as f64;
    println!("per-face opt2 avg: {perface_avg:.3}s/face");

    for &b in &[2usize, 4, 13] {
        let mut ok = true;
        let mut mismatch = 0usize;
        let t0 = Instant::now();
        for (g, chunk) in views.chunks(b).enumerate() {
            let (outs, fb) = v2_slot_batch(chunk, &shared);
            for (i, o) in outs.iter().enumerate() {
                let (eq, mm) = bitwise_equal(&o, &refs[g * b + i]);
                if !eq {
                    ok = false;
                    mismatch += mm;
                }
            }
            let _ = fb;
        }
        let dt = t0.elapsed().as_secs_f64();
        let nfaces = views.len();
        println!(
            "batch B={b:2}: total={dt:.3}s per_face={:.3}s speedup_vs_perface={:.2}x bitwise_ok={ok} mismatch={mismatch}",
            dt / nfaces as f64,
            perface_avg / (dt / nfaces as f64)
        );
    }
}

/// 批处理多线程: workers 个线程, 每个线程 rounds 轮处理 batch 个面; 输出 slot/s。
pub fn run_v2_batch_mt(data_dir: &str, workers_list: &[usize], rounds: usize, batch: usize) {
    let restrict_m = crate::npy::as_f32_mat(crate::npy::load(&format!("{data_dir}/restrict.npy")));
    let shared = Arc::new(build_shared(data_dir));
    let names = std::fs::read_to_string(format!("{data_dir}/sample_names.txt")).unwrap();
    let mut slots: Vec<Array2<f32>> = Vec::new();
    for nm in names.lines() {
        let raw = crate::npy::as_f32_mat(crate::npy::load(&format!("{data_dir}/factor_{nm}.npy")));
        let ranked = crate::engine::rank_and_fill_missing_cross_sectional_median(&raw, &restrict_m);
        let (mean5, _max5, _min5, std5) = crate::engine::rolling_stats_f32_serial(&ranked, 5, 2);
        slots.push(ranked);
        slots.push(mean5);
        slots.push(std5);
    }
    let slots = Arc::new(slots);
    let n = slots.len();

    for &w in workers_list {
        for mode in ["perface", "batch"] {
            let t0 = Instant::now();
            let mut handles = Vec::with_capacity(w);
            for wk in 0..w {
                let sh = shared.clone();
                let sl = slots.clone();
                handles.push(std::thread::spawn(move || {
                    let mut acc = 0.0f64;
                    for r in 0..rounds {
                        let base = (wk * rounds + r) % n;
                        if mode == "perface" {
                            let s = Instant::now();
                            let (o, _) = v2_slot(sl[base].view(), &sh, "opt2");
                            acc += s.elapsed().as_secs_f64();
                            std::hint::black_box(&o);
                        } else {
                            let mut views = Vec::with_capacity(batch);
                            for b in 0..batch {
                                views.push(sl[(base + b) % n].view());
                            }
                            let s = Instant::now();
                            let (outs, _) = v2_slot_batch(&views, &sh);
                            acc += s.elapsed().as_secs_f64();
                            std::hint::black_box(&outs);
                        }
                    }
                    acc / rounds as f64
                }));
            }
            let per_work: Vec<f64> = handles.into_iter().map(|h| h.join().unwrap()).collect();
            let wall = t0.elapsed().as_secs_f64();
            let mean = per_work.iter().sum::<f64>() / per_work.len() as f64;
            let throughput = if mode == "perface" {
                (w * rounds) as f64 / wall
            } else {
                (w * rounds * batch) as f64 / wall
            };
            println!(
                "MT workers={w:3} mode={mode:7} batch={batch:2} per_slot={mean:.3}s throughput={throughput:.2} slot/s"
            );
        }
    }
}

// ==================== opt3: 归并式 rank2 (2026-09 沙箱原型) ====================
//
// 结论 (tail_perf_bench v2m / v2mmt, 真实数据 15 个 slot):
//   - 与生产 opt2 批量路径 **逐位一致** (bitwise=true, mismatch=0), 单线程与 192 线程下均验证
//   - neutralize 端到端加速: 单线程 1.20~1.28x; 48/96/192 线程 1.11~1.50x
//   - 该步在 IC-only 回测里占 ~81%, 折算整体约 1.15~1.20x
//   - 顺带消掉 rank2 路径上每 (slot,日) 的 3 次堆分配 (vals.to_vec / ranks / keys64)
// 实现要点: 用 stamp(每 slot 一份) 标记被填充过的位置, 归并时把 rank1 的有序值表与
//   少量填充值二路归并; 等值段取平均秩, 输出 avg_rank/n_valid, 与 rank_pct_row_into 同语义。
//
// 观察（真实数据实测, data/ 三个因子）: rank2 的输入 = rank1 产出的 pct 值
// （本身按值升序，只是散落在股票下标上）+ 少量"填充值"。生产数据里每天只有
// 约 44/5400 个位置被填充且存活到 rank2（其余 NaN 都在不可交易股上，会被
// restrict mask 掉）。也就是说 rank2 用 8-pass u64 基数排序重排 ~3900 个元素，
// 其中 ~99% 本来就有序。
//
// 归并方案: A = rank1 原值（沿 rank1 的升序位置表扫描，天然升序）,
//           B = 填充值（小数组，单独排序）,
//           二路归并 + 等值段取平均秩 → 与 rank_pct_row_into 逐位一致。

/// 与生产 fill_ind_reg_row 逐位一致，额外把"被写入的位置"记进 stamp / filled_idx。
#[allow(clippy::too_many_arguments)]
fn fill_ind_reg_row_track(
    row: &mut [f64],
    level_rows: [&[f64]; 3],
    size_row: &[f64],
    order_rows: [&[usize]; 3],
    ys: &mut Vec<f64>,
    bs: &mut Vec<f64>,
    obs: &mut Vec<bool>,
    stamp: &mut [u32],
    gen: u32,
    filled_idx: &mut Vec<u32>,
) {
    let n = row.len();
    for li in 0..3 {
        if !has_ge_n_unique(row, 10) {
            continue;
        }
        let level_row = level_rows[li];
        let order = order_rows[li];
        let mut seg_start = 0usize;
        while seg_start < n {
            let code = level_row[order[seg_start]];
            if code.is_nan() {
                break;
            }
            let mut seg_end = seg_start + 1;
            while seg_end < n && level_row[order[seg_end]] == code {
                seg_end += 1;
            }
            let mut has_target = false;
            for &ci in &order[seg_start..seg_end] {
                if row[ci].is_nan() || size_row[ci].is_nan() {
                    has_target = true;
                    break;
                }
            }
            if !has_target {
                seg_start = seg_end;
                continue;
            }
            ys.clear();
            bs.clear();
            obs.clear();
            for &ci in &order[seg_start..seg_end] {
                let ok = !row[ci].is_nan() && !size_row[ci].is_nan();
                obs.push(ok);
                if ok {
                    ys.push(row[ci]);
                    bs.push(size_row[ci]);
                }
            }
            if ys.len() >= 10 {
                let (c0, c1) = ols2(&ys, &bs);
                for (mi, &ci) in order[seg_start..seg_end].iter().enumerate() {
                    if !obs[mi] {
                        row[ci] = c0 + c1 * size_row[ci];
                        stamp[ci] = gen;
                        filled_idx.push(ci as u32);
                    }
                }
            }
            seg_start = seg_end;
        }
    }
}

/// 与生产 median_fill_level_row 逐位一致，额外记录被填充的位置。
#[allow(clippy::too_many_arguments)]
fn median_fill_level_row_track(
    row: &mut [f64],
    codes_row: &[f64],
    valid_mask_row: Option<&[f64]>,
    order: &[usize],
    nan_mask: &[bool],
    sv: &mut Vec<f64>,
    stamp: &mut [u32],
    gen: u32,
    filled_idx: &mut Vec<u32>,
) {
    let n = row.len();
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
        let mut has_target = false;
        for &ci in &order[seg_start..seg_end] {
            let valid = valid_mask_row.map_or(true, |vr| vr[ci] == 1.0);
            if nan_mask[ci] && valid {
                has_target = true;
                break;
            }
        }
        if !has_target {
            seg_start = seg_end;
            continue;
        }
        sv.clear();
        for &ci in &order[seg_start..seg_end] {
            let valid = valid_mask_row.map_or(true, |vr| vr[ci] == 1.0);
            if valid && !row[ci].is_nan() {
                sv.push(row[ci]);
            }
        }
        if !sv.is_empty() {
            let med = median_inplace(sv);
            for &ci in &order[seg_start..seg_end] {
                let valid = valid_mask_row.map_or(true, |vr| vr[ci] == 1.0);
                if nan_mask[ci] && valid {
                    row[ci] = med;
                    stamp[ci] = gen;
                    filled_idx.push(ci as u32);
                }
            }
        }
        seg_start = seg_end;
    }
}

/// 归并式 rank2（就地）。与 rank_pct_row_f64_in_place 逐位一致。
/// rank1_order: rank1 的升序位置表; stamp/filled_idx: 填充痕迹; gen: 本次 (slot,日) 的代标记。
#[allow(clippy::too_many_arguments)]
fn rank_pct_row_merge_in_place(
    row: &mut [f64],
    rank1_order: &[usize],
    stamp: &mut [u32],
    gen: u32,
    filled_idx: &[u32],
    a_pos: &mut Vec<u32>,
    a_val: &mut Vec<f64>,
    b_pos: &mut Vec<u32>,
    b_val: &mut Vec<f64>,
) {
    // A: rank1 原值（沿升序位置表 → 天然非降序）
    a_pos.clear();
    a_val.clear();
    for &j in rank1_order {
        if stamp[j] == gen {
            continue; // 已被填充 → 归 B
        }
        let v = row[j];
        if v.is_nan() {
            continue; // 被 ind1 scrub / restrict mask
        }
        a_pos.push(j as u32);
        a_val.push(v);
    }
    // B: 填充值（stamp 去重后收集）
    b_pos.clear();
    b_val.clear();
    for &j in filled_idx {
        let ju = j as usize;
        if stamp[ju] != gen {
            continue; // 重复记录
        }
        stamp[ju] = gen.wrapping_add(1); // 标记已收集, 防重复
        let v = row[ju];
        if v.is_nan() {
            continue; // 填充结果是 NaN（size 缺失）→ 不参与排序
        }
        b_pos.push(j);
        b_val.push(v);
    }
    // B 升序（插入排序, 稳定; |B| 每天仅几十个）
    let nb = b_val.len();
    for i in 1..nb {
        let mut k = i;
        while k > 0 && b_val[k - 1] > b_val[k] {
            b_val.swap(k - 1, k);
            b_pos.swap(k - 1, k);
            k -= 1;
        }
    }
    // 二路归并 + 等值段平均秩
    let na = a_val.len();
    let n_valid = (na + nb) as f64;
    let mut i = 0usize;
    let mut j = 0usize;
    let mut rank = 1usize;
    while i < na || j < nb {
        let va = if i < na { a_val[i] } else { f64::INFINITY };
        let vb = if j < nb { b_val[j] } else { f64::INFINITY };
        let v = if va <= vb { va } else { vb };
        let i0 = i;
        let j0 = j;
        while i < na && a_val[i] == v {
            i += 1;
        }
        while j < nb && b_val[j] == v {
            j += 1;
        }
        let cnt = (i - i0) + (j - j0);
        // 与 rank_pct_row_into 一致: 输出 avg_rank / n_valid (pct, 非原始秩)
        let avg = ((rank + rank + cnt - 1) as f64 / 2.0) / n_valid;
        for k in i0..i {
            row[a_pos[k] as usize] = avg;
        }
        for k in j0..j {
            row[b_pos[k] as usize] = avg;
        }
        rank += cnt;
    }
}


use std::cell::Cell;
thread_local! {
    static T_RANK1: Cell<f64> = Cell::new(0.0);
    static T_FILLIND: Cell<f64> = Cell::new(0.0);
    static T_FILLS: Cell<f64> = Cell::new(0.0);
    static T_RANK2: Cell<f64> = Cell::new(0.0);
    static T_OLS: Cell<f64> = Cell::new(0.0);
}
thread_local! {
}
pub fn take_timers() -> (f64, f64, f64, f64, f64) {
    let g = |c: &'static std::thread::LocalKey<Cell<f64>>| c.with(|x| { let v = x.get(); x.set(0.0); v });
    (g(&T_RANK1), g(&T_FILLIND), g(&T_FILLS), g(&T_RANK2), g(&T_OLS))
}

/// opt3 批量中性化: 与 v2_slot_batch 逐位一致, 只把 rank2 换成归并。
pub fn v2_slot_batch_merge(slots: &[ArrayView2<'_, f32>], shared: &V2Shared) -> (Vec<Array2<f32>>, u64) {
    let b = slots.len();
    let (t, n) = slots[0].dim();
    let mut outs: Vec<Array2<f32>> = (0..b).map(|_| Array2::from_elem((t, n), f32::NAN)).collect();
    let mut pct: Vec<Vec<f64>> = vec![vec![0.0; n]; b];
    let mut filled: Vec<Vec<f64>> = vec![vec![0.0; n]; b];
    let mut rank1_orders: Vec<Vec<usize>> = vec![Vec::with_capacity(n); b];
    let mut filled_idx: Vec<Vec<u32>> = vec![Vec::with_capacity(64); b];
    // 每个 slot 一份 stamp: 同一天内多个 slot 会填充同一批位置, 共用一份会互相覆盖
    let mut stamp: Vec<Vec<u32>> = vec![vec![0u32; n]; b];
    let mut keys32 = vec![0u32; n];
    let mut tmp: Vec<usize> = Vec::with_capacity(n);
    let mut ys: Vec<f64> = Vec::with_capacity(n);
    let mut bs: Vec<f64> = Vec::with_capacity(n);
    let mut obs: Vec<bool> = Vec::with_capacity(n);
    let mut sv: Vec<f64> = Vec::with_capacity(n);
    let mut nan_mask: Vec<bool> = Vec::with_capacity(n);
    let mut y_buf: Vec<f64> = Vec::with_capacity(n);
    let mut xty: Vec<f64> = Vec::with_capacity(64);
    let mut a_pos: Vec<u32> = Vec::with_capacity(n);
    let mut a_val: Vec<f64> = Vec::with_capacity(n);
    let mut b_pos: Vec<u32> = Vec::with_capacity(256);
    let mut b_val: Vec<f64> = Vec::with_capacity(256);
    let mut fallback_days = 0u64;
    let base = &shared.base;
    let mut ind0_row = Vec::<f64>::with_capacity(n);
    let mut gen: u32 = 0;
    let mut selfcheck_reported = 0usize;
    for idx in 0..t {
        let (ind2_v, ind1_v, size_v) = (base.ind2.row(idx), base.ind1.row(idx), base.size_ranked.row(idx));
        let (zeros_v, mask_v, restrict_v) = (base.zeros.row(idx), base.ind1_mask.row(idx), base.restrict_f64.row(idx));
        let (o0_v, o1_v, o2_v) = (shared.orders[0].row(idx), shared.orders[1].row(idx), shared.orders[2].row(idx));
        let (o3_v, o4_v) = (shared.orders[3].row(idx), shared.orders[4].row(idx));
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
        // ---- 每面: rank1 → fill(3级) → scrub ----
        for (f, slot) in slots.iter().enumerate() {
            gen += 2;
            let slot_row = slot.row(idx);
            let _t = Instant::now();
            rank_pct_row_from_f32_in(
                slot_row.as_slice().unwrap(),
                &mut pct[f],
                &mut rank1_orders[f],
                &mut tmp,
                &mut keys32,
            );
            T_RANK1.with(|c| c.set(c.get() + _t.elapsed().as_secs_f64()));
            let _t = Instant::now();
            filled_idx[f].clear();
            fill_ind_reg_row_track(
                &mut pct[f],
                [ind2_r, ind1_r, &ind0_row],
                size_r,
                [o0, o1, o2],
                &mut ys,
                &mut bs,
                &mut obs,
                &mut stamp[f],
                gen,
                &mut filled_idx[f],
            );
            T_FILLIND.with(|c| c.set(c.get() + _t.elapsed().as_secs_f64()));
            for (j, &v) in ind1_r.iter().enumerate() {
                if v.is_nan() {
                    pct[f][j] = f64::NAN;
                }
            }
            filled[f].copy_from_slice(&pct[f]);
        }
        // ---- 每面: 三级中位 → restrict → rank2(归并) → OLS ----
        for f in 0..b {
            let fgen = gen - 2 * (b - 1 - f) as u32;
            let _t = Instant::now();
            nan_mask.clear();
            for &v in filled[f].iter() {
                nan_mask.push(v.is_nan());
            }
            if nan_mask.iter().any(|&x| x) {
                median_fill_level_row_track(
                    &mut filled[f], ind2_r, None, o3, &nan_mask, &mut sv,
                    &mut stamp[f], fgen, &mut filled_idx[f],
                );
            }
            nan_mask.clear();
            for &v in filled[f].iter() {
                nan_mask.push(v.is_nan());
            }
            if nan_mask.iter().any(|&x| x) {
                median_fill_level_row_track(
                    &mut filled[f], ind1_r, None, o4, &nan_mask, &mut sv,
                    &mut stamp[f], fgen, &mut filled_idx[f],
                );
            }
            nan_mask.clear();
            for &v in filled[f].iter() {
                nan_mask.push(v.is_nan());
            }
            if nan_mask.iter().any(|&x| x) {
                median_fill_level_row_track(
                    &mut filled[f], zeros_r, Some(mask_r), o0, &nan_mask, &mut sv,
                    &mut stamp[f], fgen, &mut filled_idx[f],
                );
            }
            T_FILLS.with(|c| c.set(c.get() + _t.elapsed().as_secs_f64()));
            for (j, &v) in restrict_r.iter().enumerate() {
                if v != 0.0 {
                    filled[f][j] = f64::NAN;
                }
            }
            let _t = Instant::now();
            rank_pct_row_merge_in_place(
                &mut filled[f],
                &rank1_orders[f],
                &mut stamp[f],
                fgen,
                &filled_idx[f],
                &mut a_pos,
                &mut a_val,
                &mut b_pos,
                &mut b_val,
            );
            if std::env::var("V2_SELFCHECK").is_ok() {
                // 自检: 与基数排序版逐位对账
                let mut ref_row = filled[f].clone();
                let mut r_ranks: Vec<f64> = Vec::with_capacity(n);
                let mut r_idxs: Vec<usize> = Vec::with_capacity(n);
                let mut r_tmp: Vec<usize> = Vec::with_capacity(n);
                let mut r_keys: Vec<u64> = vec![0u64; n];
                rank_pct_row_f64_in_place(&mut ref_row, &mut r_ranks, &mut r_idxs, &mut r_tmp, &mut r_keys);
                let mut bad = 0usize;
                let mut first = None;
                for j in 0..n {
                    let a = filled[f][j];
                    let bb = ref_row[j];
                    let eq = (a.is_nan() && bb.is_nan()) || (a == bb);
                    if !eq {
                        bad += 1;
                        if first.is_none() {
                            first = Some((j, a, bb));
                        }
                    }
                }
                if bad > 0 && selfcheck_reported < 3 {
                    selfcheck_reported += 1;
                    let nan_in = 0usize;
                    let _ = nan_in;
                    eprintln!(
                        "[SELFCHECK] day={idx} f={f} bad={bad}/{} |A|={} |B|={} n_rank1={} n_filled_idx={} first={:?}",
                        n, a_val.len(), b_val.len(), rank1_orders[f].len(), filled_idx[f].len(), first
                    );
                    // 打印 A 是否升序
                    let mut unsorted = 0usize;
                    for k in 1..a_val.len() {
                        if a_val[k] < a_val[k - 1] { unsorted += 1; }
                    }
                    let mut bunsorted = 0usize;
                    for k in 1..b_val.len() {
                        if b_val[k] < b_val[k - 1] { bunsorted += 1; }
                    }
                    eprintln!("[SELFCHECK] a_unsorted={unsorted} b_unsorted={bunsorted}");
                    let mut ref_nonnan = 0usize;
                    for j in 0..n { if !ref_row[j].is_nan() { ref_nonnan += 1; } }
                    eprintln!("[SELFCHECK] ref_nonnan={ref_nonnan} a+b={}", a_val.len() + b_val.len());
                    // 找出"非 NaN 但既不在 A 也不在 B"的位置
                    let mut in_a = vec![false; n];
                    for k in 0..a_val.len() { in_a[a_pos[k] as usize] = true; }
                    let mut in_b = vec![false; n];
                    for k in 0..b_val.len() { in_b[b_pos[k] as usize] = true; }
                    let mut orphan = 0usize;
                    for j in 0..n {
                        if !ref_row[j].is_nan() && !in_a[j] && !in_b[j] {
                            orphan += 1;
                            if orphan <= 4 {
                                let in_r1 = rank1_orders[f].binary_search(&j).is_ok();
                                eprintln!("[SELFCHECK] orphan j={j} val={} stamp={} gen={} in_rank1={} in_filled_idx={}",
                                    ref_row[j], stamp[f][j], fgen, in_r1, filled_idx[f].contains(&(j as u32)));
                            }
                        }
                    }
                }
            }
            T_RANK2.with(|c| c.set(c.get() + _t.elapsed().as_secs_f64()));
            let _t = Instant::now();
            let mut out_row = outs[f].row_mut(idx);
            let fb = ols_day_row_fast(
                &filled[f],
                shared,
                idx,
                &mut y_buf,
                &mut xty,
                out_row.as_slice_mut().unwrap(),
            );
            T_OLS.with(|c| c.set(c.get() + _t.elapsed().as_secs_f64()));
            if fb {
                fallback_days += 1;
            }
        }
    }
    (outs, fallback_days)
}

/// opt3 对账 + 计时（vs opt2 批量路径）。
pub fn run_v2_merge_bench(data_dir: &str) {
    let restrict_m = crate::npy::as_f32_mat(crate::npy::load(&format!("{data_dir}/restrict.npy")));
    let shared = build_shared(data_dir);
    let names = std::fs::read_to_string(format!("{data_dir}/sample_names.txt")).unwrap();
    let mut slots: Vec<Array2<f32>> = Vec::new();
    let mut slot_tags: Vec<String> = Vec::new();
    for nm in names.lines() {
        let raw = crate::npy::as_f32_mat(crate::npy::load(&format!("{data_dir}/factor_{nm}.npy")));
        let ranked = crate::engine::rank_and_fill_missing_cross_sectional_median(&raw, &restrict_m);
        let (mean5, max5, min5, std5) = crate::engine::rolling_stats_f32_serial(&ranked, 5, 2);
        for (tag, s) in [("smooth_1", &ranked), ("mean_5", &mean5), ("max_5", &max5), ("min_5", &min5), ("std_5", &std5)] {
            slots.push(s.clone());
            slot_tags.push(format!("{nm}::{tag}"));
        }
    }
    let views: Vec<ArrayView2<f32>> = slots.iter().map(|s| s.view()).collect();
    println!("pool slots: {}", slots.len());
    for &b in &[4usize, 13] {
        let t0 = Instant::now();
        let mut refs: Vec<Array2<f32>> = Vec::new();
        for chunk in views.chunks(b) {
            let (outs, _) = v2_slot_batch(chunk, &shared);
            refs.extend(outs);
        }
        let t_ref = t0.elapsed().as_secs_f64();
        let _ = take_timers();
        let t0 = Instant::now();
        let mut news: Vec<Array2<f32>> = Vec::new();
        let mut fb_total = 0u64;
        for chunk in views.chunks(b) {
            let (outs, fb) = v2_slot_batch_merge(chunk, &shared);
            fb_total += fb;
            news.extend(outs);
        }
        let t_new = t0.elapsed().as_secs_f64();
        let (tr1, tfi, tfl, tr2, tol) = take_timers();
        println!(
            "  opt3 breakdown: rank1={tr1:.3}s fill_ind={tfi:.3}s fills={tfl:.3}s rank2(merge)={tr2:.3}s ols={tol:.3}s fallback_days={fb_total}"
        );
        let mut ok = true;
        let mut mismatch = 0usize;
        let mut first_bad: Option<String> = None;
        for (i, (a, c)) in refs.iter().zip(news.iter()).enumerate() {
            let (eq, mm) = bitwise_equal(a, c);
            if !eq {
                ok = false;
                mismatch += mm;
                if first_bad.is_none() {
                    first_bad = Some(slot_tags[i].clone());
                }
            }
        }
        println!(
            "B={b:2} opt2={t_ref:.3}s opt3={t_new:.3}s speedup={:.3}x bitwise={ok} mismatch={mismatch} first_bad={first_bad:?}",
            t_ref / t_new
        );
    }
}

/// opt2 vs opt3 的多线程吞吐对照（生产是 200 线程同时跑不同因子, 内存带宽争抢严重）。
pub fn run_v2_merge_mt(data_dir: &str, workers_list: &[usize], rounds: usize, batch: usize) {
    let restrict_m = crate::npy::as_f32_mat(crate::npy::load(&format!("{data_dir}/restrict.npy")));
    let shared = Arc::new(build_shared(data_dir));
    let names = std::fs::read_to_string(format!("{data_dir}/sample_names.txt")).unwrap();
    let mut slots: Vec<Array2<f32>> = Vec::new();
    for nm in names.lines() {
        let raw = crate::npy::as_f32_mat(crate::npy::load(&format!("{data_dir}/factor_{nm}.npy")));
        let ranked = crate::engine::rank_and_fill_missing_cross_sectional_median(&raw, &restrict_m);
        let (mean5, max5, min5, std5) = crate::engine::rolling_stats_f32_serial(&ranked, 5, 2);
        for s in [&ranked, &mean5, &max5, &min5, &std5] {
            slots.push(s.clone());
        }
    }
    let slots = Arc::new(slots);
    let n_slots = slots.len();
    println!("pool slots: {n_slots}, batch={batch}, rounds={rounds}");
    for &w in workers_list {
        for mode in ["opt2", "opt3"] {
            let t0 = Instant::now();
            let mut handles = Vec::with_capacity(w);
            for _ in 0..w {
                let sh = shared.clone();
                let sl = slots.clone();
                handles.push(std::thread::spawn(move || {
                    let mut acc = 0u64;
                    for _ in 0..rounds {
                        let views: Vec<ArrayView2<f32>> = sl.iter().map(|s| s.view()).collect();
                        for chunk in views.chunks(batch) {
                            let (outs, fb) = if mode == "opt2" {
                                v2_slot_batch(chunk, &sh)
                            } else {
                                v2_slot_batch_merge(chunk, &sh)
                            };
                            acc += fb + outs.len() as u64;
                            std::hint::black_box(&outs);
                        }
                    }
                    acc
                }));
            }
            for h in handles {
                let _ = h.join().unwrap();
            }
            let wall = t0.elapsed().as_secs_f64();
            let total_slots = (w * rounds * n_slots) as f64;
            println!(
                "MT workers={w:3} mode={mode} wall={wall:7.2}s throughput={:.1} slot/s",
                total_slots / wall
            );
        }
    }
}

pub(crate) fn has_ge_n_unique_pub(vals: &[f64], need: usize) -> bool { has_ge_n_unique(vals, need) }
pub(crate) fn ols2_pub(ys: &[f64], bs: &[f64]) -> (f64, f64) { ols2(ys, bs) }
pub(crate) fn median_inplace_pub(v: &mut [f64]) -> f64 { median_inplace(v) }
