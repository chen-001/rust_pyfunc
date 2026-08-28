use ndarray::Array2;
use numpy::{IntoPyArray, PyReadonlyArray2, PyReadonlyArray3, PyArray2, PyArray3};
use pyo3::prelude::*;

// ============================================================================
//  Sandbox: 收益率中性化 IC 设计验证
//
//  目标: 验证「共享收益秩 + 因子侧轻量残差秩」是否与现状
//        「因子中性化 (rank -> OLS 残差 -> rank) + Spearman」逐位一致,
//        并给出提速对比。
//
//  口径复刻: 生产 neutralize_std_section (行业模式) + legacy_spearman_correlation
//           (ordinal 秩 + 1 - 6*sum(d^2)/(n(n^2-1)))
// ============================================================================

fn rank_pct_row(vals: &[f64]) -> Vec<f64> {
    let n = vals.len();
    let mut idxs: Vec<usize> = (0..n).filter(|&i| !vals[i].is_nan()).collect();
    idxs.sort_by(|&a, &b| vals[a].total_cmp(&vals[b]));
    let nv = idxs.len();
    let mut ranks = vec![f64::NAN; n];
    if nv == 0 {
        return ranks;
    }
    let mut i = 0;
    while i < nv {
        let mut j = i;
        while j + 1 < nv && vals[idxs[j + 1]] == vals[idxs[i]] {
            j += 1;
        }
        let avg_rank = ((i + 1) + (j + 1)) as f64 / 2.0;
        let pct = avg_rank / nv as f64;
        for item in &idxs[i..=j] {
            ranks[*item] = pct;
        }
        i = j + 1;
    }
    ranks
}

/// ordinal 秩 (0..n-1; NaN 排最后; tie 按出现顺序), 语义同生产 ordinal_ranks
fn ordinal_ranks(vals: &[f64]) -> Vec<i64> {
    let mut indexed: Vec<(usize, f64)> = vals.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| match (a.1.is_nan(), b.1.is_nan()) {
        (true, true) => a.0.cmp(&b.0),
        (true, false) => std::cmp::Ordering::Greater,
        (false, true) => std::cmp::Ordering::Less,
        (false, false) => a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)),
    });
    let mut out = vec![0i64; vals.len()];
    for (i, (idx, _)) in indexed.iter().enumerate() {
        out[*idx] = i as i64;
    }
    out
}

/// 复刻生产 legacy_spearman_correlation(ordinal 秩差公式)
fn spearman_ordinal(x: &[f64], y: &[f64]) -> f64 {
    let xr = ordinal_ranks(x);
    let yr = ordinal_ranks(y);
    let n = x.len() as f64;
    let mut d2 = 0.0f64;
    for i in 0..x.len() {
        let d = (xr[i] - yr[i]) as f64;
        d2 += d * d;
    }
    1.0 - 6.0 * d2 / (n * (n * n - 1.0))
}

/// 逐期 IC 计算 (复刻 neutralize_std_section 行业模式 + legacy_spearman):
///   1) rank_pct(x) -> 2) restrict 置空 -> 3) OLS 残差(barra 10 + 一级行业 one-hot,
///      Cholesky, n_valid>40; 否则 LU fallback) -> 4) 残差 rank_pct
///      -> 5) 过滤集 F = {open & ret finite & 残差 finite} -> 6) spearman(残差秩, 收益秩)
/// `ret_ord_global`: 若提供且 F == O(open&ret finite 全集), 直接用预计算的收益 ordinal
/// 序 (全局共享, 与因子无关), 否则对 F 内的收益重新排序 (与现状完全一致)。
fn compute_ic(
    x: &[f64],
    barra: &[f64], // (N, 10)
    ind: &[f64],   // 行业码 (原始 SW 码)
    ret: &[f64],
    restrict: &[f64],
    ret_ord_global: Option<&[i64]>,
) -> f64 {
    let n = x.len();
    // 1. rank pct
    let mut xr = rank_pct_row(x);
    // 2. restrict 置空: restrict != 0 (非 open) -> NaN
    for j in 0..n {
        if restrict[j].is_nan() || restrict[j] != 0.0 {
            xr[j] = f64::NAN;
        }
    }
    // 3. 有效集与回归 (行业模式)
    let mut ind1 = vec![f64::NAN; n];
    let mut valid = vec![false; n];
    for j in 0..n {
        if xr[j].is_finite() && (0..10).all(|c| barra[j * 10 + c].is_finite()) {
            valid[j] = true;
        }
        if ind[j].is_nan() {
            ind1[j] = f64::NAN;
        } else {
            ind1[j] = (ind[j] / 10000.0).floor();
        }
    }
    let n_valid = valid.iter().filter(|&&v| v).count();
    let mut resid = vec![f64::NAN; n];
    if n_valid <= 10 || n_valid == 0 {
        return f64::NAN;
    }
    // 当日 unique 一级行业码 (排序)
    let mut ind_codes: Vec<f64> = Vec::new();
    for j in 0..n {
        let c = ind1[j];
        if !c.is_nan() && !ind_codes.contains(&c) {
            ind_codes.push(c);
        }
    }
    ind_codes.sort_by(|a, b| a.total_cmp(b));
    let n_ind = ind_codes.len();
    let p = 10 + n_ind;
    // 累积 X'X / X'y (行业模式: 风格列 + 行业 one-hot, 无截距)
    let mut xtx = vec![0.0f64; p * p];
    let mut xty = vec![0.0f64; p];
    for j in 0..n {
        if !valid[j] {
            continue;
        }
        let yv = xr[j];
        for c in 0..10 {
            let b = barra[j * 10 + c];
            xty[c] += b * yv;
            xtx[c * p + c] += b * b;
            for c2 in (c + 1)..10 {
                let v = b * barra[j * 10 + c2];
                xtx[c * p + c2] += v;
                xtx[c2 * p + c] += v;
            }
        }
        let ic = if ind1[j].is_nan() { -1 } else {
            match ind_codes.binary_search_by(|x| x.total_cmp(&ind1[j])) {
                Ok(pos) => pos as i32,
                Err(_) => -1,
            }
        };
        if ic >= 0 {
            let col = (10 + ic) as usize;
            xty[col] += yv;
            xtx[col * p + col] += 1.0;
            for c in 0..10 {
                let b = barra[j * 10 + c];
                xtx[c * p + col] += b;
                xtx[col * p + c] += b;
            }
        }
    }
    // 求解 (Cholesky; 主路径 n_valid>40 与生产一致; 小样本 LU fallback)
    let m = nalgebra::DMatrix::from_row_slice(p, p, &xtx);
    let rhs = nalgebra::DMatrix::from_column_slice(p, 1, &xty);
    let coef: Vec<f64> = if p > 0 && nalgebra::Cholesky::new(m.clone()).is_some() {
        let chol = nalgebra::Cholesky::new(m).expect("chol");
        chol.solve(&rhs).column(0).iter().copied().collect()
    } else {
        let lu = m.lu();
        lu.solve(&rhs).map(|v| v.column(0).iter().copied().collect()).unwrap_or_default()
    };
    if coef.len() != p {
        return f64::NAN;
    }
    for j in 0..n {
        if !valid[j] {
            continue;
        }
        let mut pred = 0.0f64;
        for c in 0..10 {
            pred += coef[c] * barra[j * 10 + c];
        }
        let ic = if ind1[j].is_nan() { -1 } else {
            match ind_codes.binary_search_by(|x| x.total_cmp(&ind1[j])) {
                Ok(pos) => pos as i32,
                Err(_) => -1,
            }
        };
        if ic >= 0 {
            pred += coef[(10 + ic) as usize];
        }
        resid[j] = xr[j] - pred;
    }
    // 4. 残差 rank pct (NaN 保持 NaN)
    let resp = rank_pct_row(&resid);
    // 5. 过滤集 F = {open & ret finite & 残差有限}
    let mut fx: Vec<f64> = Vec::new();
    let mut fy: Vec<f64> = Vec::new();
    let mut o_mask = vec![false; n];
    for j in 0..n {
        let is_open = !restrict[j].is_nan() && restrict[j] == 0.0;
        if is_open && ret[j].is_finite() {
            o_mask[j] = true;
        }
        if is_open && ret[j].is_finite() && resp[j].is_finite() {
            fx.push(resp[j]);
            fy.push(ret[j]);
        }
    }
    if fx.len() < 10 {
        return f64::NAN;
    }
    // 6. IC: 现状 = 对 F 内收益重新排序; 新设计 = 全局预计算序 (F == O 时等价)
    if let Some(ord) = ret_ord_global {
        if fx.len() == o_mask.iter().filter(|&&v| v).count() {
            // F == O: 直接用全局序 (零排序)
            let xr2 = ordinal_ranks(&fx);
            let nf = fx.len() as f64;
            let mut d2 = 0.0f64;
            let mut k = 0usize;
            for j in 0..n {
                if o_mask[j] {
                    let d = (xr2[k] - ord[j]) as f64;
                    d2 += d * d;
                    k += 1;
                }
            }
            1.0 - 6.0 * d2 / (nf * (nf * nf - 1.0))
        } else {
            spearman_ordinal(&fx, &fy)
        }
    } else {
        spearman_ordinal(&fx, &fy)
    }
}

/// 预计算收益 ordinal 序 (基于 O = open & ret finite 全集, 与因子无关): (T, N)
fn precompute_ret_ord(ret: &Array2<f64>, restrict: &Array2<f64>) -> Vec<i64> {
    let (t, n) = ret.dim();
    let mut out = vec![0i64; t * n];
    for i in 0..t {
        let mut vals = vec![f64::NAN; n];
        for j in 0..n {
            let is_open = !restrict[[i, j]].is_nan() && restrict[[i, j]] == 0.0;
            if is_open && ret[[i, j]].is_finite() {
                vals[j] = ret[[i, j]];
            }
        }
        let ord = ordinal_ranks(&vals);
        for j in 0..n {
            out[i * n + j] = ord[j];
        }
    }
    out
}

/// 现状算法层逐期复刻: 每 slot 每期都重新排收益
#[pyfunction]
fn sandbox_ic_old(
    py: Python<'_>,
    blocks: PyReadonlyArray3<'_, f32>,   // (F, T, N) 因子 slot 值 (已 rank+fill)
    barra: PyReadonlyArray3<'_, f64>,    // (T, N, 10) 风格 rank pct
    ind: PyReadonlyArray2<'_, f64>,      // (T, N) 行业码
    ret: PyReadonlyArray2<'_, f32>,      // (T, N) 未来收益 (已与信号对齐)
    restrict: PyReadonlyArray2<'_, f32>, // (T, N) 0=open (信号日)
) -> PyResult<Py<PyArray2<f64>>> {
    let blocks = blocks.as_array().to_owned();
    let barra = barra.as_array().to_owned();
    let ind = ind.as_array().to_owned();
    let ret = ret.as_array().to_owned();
    let restrict = restrict.as_array().to_owned();
    let (f, t, n) = blocks.dim();
    let mut out = Array2::<f64>::from_elem((f, t), f64::NAN);
    for fi in 0..f {
        for ti in 0..t {
            let x: Vec<f64> = (0..n).map(|j| blocks[[fi, ti, j]] as f64).collect();
            let barra_flat: Vec<f64> = (0..n * 10).map(|k| barra[[ti, k / 10, k % 10]]).collect();
            let ind_t: Vec<f64> = (0..n).map(|j| ind[[ti, j]]).collect();
            let ret_t: Vec<f64> = (0..n).map(|j| ret[[ti, j]] as f64).collect();
            let restr_t: Vec<f64> = (0..n).map(|j| restrict[[ti, j]] as f64).collect();
            out[[fi, ti]] = compute_ic(&x, &barra_flat, &ind_t, &ret_t, &restr_t, None);
        }
    }
    Ok(out.into_pyarray(py).to_owned())
}

/// 新设计: 收益序全局预计算 (与因子无关, 一次) + 每 slot 只做轻量残差秩
#[pyfunction]
fn sandbox_ic_new(
    py: Python<'_>,
    blocks: PyReadonlyArray3<'_, f32>,
    barra: PyReadonlyArray3<'_, f64>,
    ind: PyReadonlyArray2<'_, f64>,
    ret: PyReadonlyArray2<'_, f32>,
    restrict: PyReadonlyArray2<'_, f32>,
) -> PyResult<Py<PyArray2<f64>>> {
    let blocks = blocks.as_array().to_owned();
    let barra = barra.as_array().to_owned();
    let ind = ind.as_array().to_owned();
    let ret = ret.as_array().to_owned();
    let restrict = restrict.as_array().to_owned();
    let (f, t, n) = blocks.dim();
    // 预计算: 收益 ordinal 序 (T, N) —— 一次, 所有因子/slot 共享
    let ret_f = ret.mapv(|v| v as f64);
    let restr_f = restrict.mapv(|v| v as f64);
    let ord = precompute_ret_ord(&ret_f, &restr_f);
    let mut out = Array2::<f64>::from_elem((f, t), f64::NAN);
    for fi in 0..f {
        for ti in 0..t {
            let x: Vec<f64> = (0..n).map(|j| blocks[[fi, ti, j]] as f64).collect();
            let barra_flat: Vec<f64> = (0..n * 10).map(|k| barra[[ti, k / 10, k % 10]]).collect();
            let ind_t: Vec<f64> = (0..n).map(|j| ind[[ti, j]]).collect();
            let ret_t: Vec<f64> = (0..n).map(|j| ret[[ti, j]] as f64).collect();
            let restr_t: Vec<f64> = (0..n).map(|j| restrict[[ti, j]] as f64).collect();
            let ord_t: &[i64] = &ord[ti * n..(ti + 1) * n];
            out[[fi, ti]] = compute_ic(&x, &barra_flat, &ind_t, &ret_t, &restr_t, Some(ord_t));
        }
    }
    Ok(out.into_pyarray(py).to_owned())
}


// ============================================================================
//  Pearson 口径版本: 收益侧一次性中性化 + 闭式恒等式
//   IC_A(pearson) = corr(M_B x, r)  ≡  <x, M_B r> / (|M_B x| * |r|)
//   -> 收益残差 M_B r 与 |r| 所有 slot 共享; 每 slot 只做 O(NK) 轻量投影
// ============================================================================

/// 仅返回 OLS 残差 (行业模式, 与 compute_ic 相同口径) — 供 Pearson 方案复用
fn resid_only(x: &[f64], barra: &[f64], ind: &[f64], restrict: &[f64]) -> Vec<f64> {
    let n = x.len();
    let mut xr = x.to_vec();
    for j in 0..n {
        if restrict[j].is_nan() || restrict[j] != 0.0 {
            xr[j] = f64::NAN;
        }
    }
    let mut ind1 = vec![f64::NAN; n];
    let mut valid = vec![false; n];
    for j in 0..n {
        if xr[j].is_finite() && (0..10).all(|c| barra[j * 10 + c].is_finite()) {
            valid[j] = true;
        }
        if ind[j].is_nan() {
            ind1[j] = f64::NAN;
        } else {
            ind1[j] = (ind[j] / 10000.0).floor();
        }
    }
    let n_valid = valid.iter().filter(|&&v| v).count();
    let mut resid = vec![f64::NAN; n];
    if n_valid <= 10 {
        return resid;
    }
    let mut ind_codes: Vec<f64> = Vec::new();
    for j in 0..n {
        let c = ind1[j];
        if !c.is_nan() && !ind_codes.contains(&c) {
            ind_codes.push(c);
        }
    }
    ind_codes.sort_by(|a, b| a.total_cmp(b));
    let n_ind = ind_codes.len();
    let p = 10 + n_ind;
    let mut xtx = vec![0.0f64; p * p];
    let mut xty = vec![0.0f64; p];
    for j in 0..n {
        if !valid[j] {
            continue;
        }
        let yv = xr[j];
        for c in 0..10 {
            let b = barra[j * 10 + c];
            xty[c] += b * yv;
            xtx[c * p + c] += b * b;
            for c2 in (c + 1)..10 {
                let v = b * barra[j * 10 + c2];
                xtx[c * p + c2] += v;
                xtx[c2 * p + c] += v;
            }
        }
        let ic = if ind1[j].is_nan() { -1 } else {
            match ind_codes.binary_search_by(|x| x.total_cmp(&ind1[j])) {
                Ok(pos) => pos as i32,
                Err(_) => -1,
            }
        };
        if ic >= 0 {
            let col = (10 + ic) as usize;
            xty[col] += yv;
            xtx[col * p + col] += 1.0;
            for c in 0..10 {
                let b = barra[j * 10 + c];
                xtx[c * p + col] += b;
                xtx[col * p + c] += b;
            }
        }
    }
    let m = nalgebra::DMatrix::from_row_slice(p, p, &xtx);
    let rhs = nalgebra::DMatrix::from_column_slice(p, 1, &xty);
    let coef: Vec<f64> = if nalgebra::Cholesky::new(m.clone()).is_some() {
        let chol = nalgebra::Cholesky::new(m).expect("chol");
        chol.solve(&rhs).column(0).iter().copied().collect()
    } else {
        let lu = m.lu();
        lu.solve(&rhs).map(|v| v.column(0).iter().copied().collect()).unwrap_or_default()
    };
    if coef.len() != p {
        return resid;
    }
    for j in 0..n {
        if !valid[j] {
            continue;
        }
        let mut pred = 0.0f64;
        for c in 0..10 {
            pred += coef[c] * barra[j * 10 + c];
        }
        let ic = if ind1[j].is_nan() { -1 } else {
            match ind_codes.binary_search_by(|x| x.total_cmp(&ind1[j])) {
                Ok(pos) => pos as i32,
                Err(_) => -1,
            }
        };
        if ic >= 0 {
            pred += coef[(10 + ic) as usize];
        }
        resid[j] = xr[j] - pred;
    }
    resid
}

/// Pearson 现状版: 每 slot 因子侧完整中性化 -> corr(残差, 收益)
#[pyfunction]
fn sandbox_ic_pearson_old(
    py: Python<'_>,
    blocks: PyReadonlyArray3<'_, f32>,
    barra: PyReadonlyArray3<'_, f64>,
    ind: PyReadonlyArray2<'_, f64>,
    ret: PyReadonlyArray2<'_, f32>,
    restrict: PyReadonlyArray2<'_, f32>,
) -> PyResult<Py<PyArray2<f64>>> {
    let blocks = blocks.as_array().to_owned();
    let barra = barra.as_array().to_owned();
    let ind = ind.as_array().to_owned();
    let ret = ret.as_array().to_owned();
    let restrict = restrict.as_array().to_owned();
    let (f, t, n) = blocks.dim();
    let mut out = Array2::<f64>::from_elem((f, t), f64::NAN);
    for fi in 0..f {
        for ti in 0..t {
            let x: Vec<f64> = (0..n).map(|j| blocks[[fi, ti, j]] as f64).collect();
            let barra_flat: Vec<f64> = (0..n * 10).map(|k| barra[[ti, k / 10, k % 10]]).collect();
            let ind_t: Vec<f64> = (0..n).map(|j| ind[[ti, j]]).collect();
            let ret_t: Vec<f64> = (0..n).map(|j| ret[[ti, j]] as f64).collect();
            let restr_t: Vec<f64> = (0..n).map(|j| restrict[[ti, j]] as f64).collect();
            let resid = resid_only(&x, &barra_flat, &ind_t, &restr_t);
            // corr(resid, ret) 在有效集上
            let mut sv = Vec::new();
            let mut rv = Vec::new();
            for j in 0..n {
                if resid[j].is_finite() && ret_t[j].is_finite() {
                    sv.push(resid[j]);
                    rv.push(ret_t[j]);
                }
            }
            if sv.len() < 10 {
                continue;
            }
            let nv = sv.len() as f64;
            let ms: f64 = sv.iter().sum::<f64>() / nv;
            let mr: f64 = rv.iter().sum::<f64>() / nv;
            let mut cov = 0.0;
            let mut vs = 0.0;
            let mut vr = 0.0;
            for i in 0..sv.len() {
                let a = sv[i] - ms;
                let b = rv[i] - mr;
                cov += a * b;
                vs += a * a;
                vr += b * b;
            }
            out[[fi, ti]] = cov / (vs.sqrt() * vr.sqrt());
        }
    }
    Ok(out.into_pyarray(py).to_owned())
}

/// Pearson 新设计版: 收益残差与开一次预计算, 因子侧只 O(NK) 投影
#[pyfunction]
fn sandbox_ic_pearson_new(
    py: Python<'_>,
    blocks: PyReadonlyArray3<'_, f32>,
    barra: PyReadonlyArray3<'_, f64>,
    ind: PyReadonlyArray2<'_, f64>,
    ret: PyReadonlyArray2<'_, f32>,
    restrict: PyReadonlyArray2<'_, f32>,
) -> PyResult<Py<PyArray2<f64>>> {
    let blocks = blocks.as_array().to_owned();
    let barra = barra.as_array().to_owned();
    let ind = ind.as_array().to_owned();
    let ret = ret.as_array().to_owned();
    let restrict = restrict.as_array().to_owned();
    let (f, t, n) = blocks.dim();
    // ---- 收益侧一次性: 每期收益残差 r_res 与 |r| (与因子无关) ----
    let mut r_res = Array2::<f64>::from_elem((t, n), f64::NAN);
    let mut r_norm = vec![f64::NAN; t];
    for ti in 0..t {
        let ret_t: Vec<f64> = (0..n).map(|j| ret[[ti, j]] as f64).collect();
        let barra_flat: Vec<f64> = (0..n * 10).map(|k| barra[[ti, k / 10, k % 10]]).collect();
        let ind_t: Vec<f64> = (0..n).map(|j| ind[[ti, j]]).collect();
        let restr_t: Vec<f64> = (0..n).map(|j| restrict[[ti, j]] as f64).collect();
        let rr = resid_only(&ret_t, &barra_flat, &ind_t, &restr_t);
        for j in 0..n {
            r_res[[ti, j]] = rr[j];
        }
        let mut s = 0.0;
        let mut c = 0.0;
        for j in 0..n {
            if rr[j].is_finite() {
                s += rr[j];
                c += 1.0;
            }
        }
        if c > 0.0 {
            r_norm[ti] = s / c;
        }
    }
    let mut out = Array2::<f64>::from_elem((f, t), f64::NAN);
    for fi in 0..f {
        for ti in 0..t {
            let x: Vec<f64> = (0..n).map(|j| blocks[[fi, ti, j]] as f64).collect();
            let barra_flat: Vec<f64> = (0..n * 10).map(|k| barra[[ti, k / 10, k % 10]]).collect();
            let ind_t: Vec<f64> = (0..n).map(|j| ind[[ti, j]]).collect();
            let restr_t: Vec<f64> = (0..n).map(|j| restrict[[ti, j]] as f64).collect();
            let rrt: Vec<f64> = (0..n).map(|j| r_res[[ti, j]]).collect();
            let ret_t: Vec<f64> = (0..n).map(|j| ret[[ti, j]] as f64).collect();
            // 分子 <x, r_res> (在 F 有效集上); 分母 |M_B x| * |r|
            // 其中 |M_B x| 用逐日 O(NK) 投影 (与分子同批 valid)
            let mut f_valid = vec![false; n];
            for j in 0..n {
                f_valid[j] = x[j].is_finite() && rrt[j].is_finite() && !restr_t[j].is_nan() && restr_t[j] == 0.0;
            }
            let mut vs: f64 = 0.0;
            let mut vr: f64 = 0.0;
            let mut cov: f64 = 0.0;
            let mut cnt: f64 = 0.0;
            let mut ms = 0.0;
            let mut mr = 0.0;
            let mut mr_ret = 0.0;
            for j in 0..n {
                if f_valid[j] {
                    ms += x[j];
                    mr += rrt[j];
                    mr_ret += ret_t[j];
                    cnt += 1.0;
                }
            }
            if cnt <= 10.0 {
                continue;
            }
            ms /= cnt;
            mr /= cnt;
            mr_ret /= cnt;
            let residx = resid_only(&x, &barra_flat, &ind_t, &restr_t);
            let mut ms_r = 0.0;
            let mut cnt_r = 0.0;
            for j in 0..n {
                if residx[j].is_finite() {
                    ms_r += residx[j];
                    cnt_r += 1.0;
                }
            }
            if cnt_r <= 10.0 {
                continue;
            }
            ms_r /= cnt_r;
            // IC = cov(x, r_res) / (|M_B x_centered| * |r_centered|)
            // (恒等式: corr(M_B x, r) = cov(x, M_B r) / (|M_B x| * |r|))
            for j in 0..n {
                if f_valid[j] {
                    let a = x[j] - ms;
                    let b = rrt[j] - mr;
                    cov += a * b;
                    vs += a * a;
                    vr += (ret_t[j] - mr_ret) * (ret_t[j] - mr_ret);
                }
            }
            let mut res_s = 0.0;
            for j in 0..n {
                if residx[j].is_finite() {
                    let a = residx[j] - ms_r;
                    res_s += a * a;
                }
            }
            out[[fi, ti]] = cov / (res_s.sqrt() * vr.sqrt());
        }
    }
    Ok(out.into_pyarray(py).to_owned())
}


/// Pearson 新设计-快路径: 收益侧 & X'X 一次预计算, 因子侧仅 O(NK) 投影+O(N) 相关
/// (理论: 每 slot 从 O(N*K^2) X'X 累积 + Cholesky + 残差物化 降到 O(N*K) 向量乘加)
#[pyfunction]
fn sandbox_ic_pearson_fast(
    py: Python<'_>,
    blocks: PyReadonlyArray3<'_, f32>,
    barra: PyReadonlyArray3<'_, f64>,
    ind: PyReadonlyArray2<'_, f64>,
    ret: PyReadonlyArray2<'_, f32>,
    restrict: PyReadonlyArray2<'_, f32>,
) -> PyResult<Py<PyArray2<f64>>> {
    let blocks = blocks.as_array().to_owned();
    let barra = barra.as_array().to_owned();
    let ind = ind.as_array().to_owned();
    let ret = ret.as_array().to_owned();
    let restrict = restrict.as_array().to_owned();
    let (f, t, n) = blocks.dim();

    // ---------- 收益侧/风格侧一次性预计算 (与因子无关) ----------
    let mut valid_masks: Vec<Vec<bool>> = Vec::with_capacity(t);
    let mut ind_cols_saved: Vec<Vec<i32>> = Vec::with_capacity(t);
    let mut xtx_saved: Vec<Vec<f64>> = Vec::with_capacity(t);
    let mut n_ind_saved: Vec<usize> = Vec::with_capacity(t);
    let mut r_res = Array2::<f64>::from_elem((t, n), f64::NAN);
    let mut ret_ss = vec![0.0f64; t]; // 有效集上原始收益中心化平方和
    for ti in 0..t {
        let mut ind1 = vec![f64::NAN; n];
        let mut valid = vec![false; n];
        for j in 0..n {
            let is_open = !restrict[[ti, j]].is_nan() && restrict[[ti, j]] == 0.0;
            if is_open && (0..10).all(|c| barra[[ti, j, c]].is_finite()) {
                valid[j] = true;
            }
            if ind[[ti, j]].is_nan() {
                ind1[j] = f64::NAN;
            } else {
                ind1[j] = (ind[[ti, j]] / 10000.0).floor();
            }
        }
        let mut ind_codes: Vec<f64> = Vec::new();
        for j in 0..n {
            let c = ind1[j];
            if !c.is_nan() && valid[j] && !ind_codes.contains(&c) {
                ind_codes.push(c);
            }
        }
        ind_codes.sort_by(|a, b| a.total_cmp(b));
        let n_ind = ind_codes.len();
        let p = 10 + n_ind;
        let mut ind_cols = vec![-1i32; n];
        for j in 0..n {
            if valid[j] && !ind1[j].is_nan() {
                if let Ok(pos) = ind_codes.binary_search_by(|x| x.total_cmp(&ind1[j])) {
                    ind_cols[j] = pos as i32;
                }
            }
        }
        let n_valid = valid.iter().filter(|&&v| v).count();
        if n_valid <= 10 || n_ind == 0 {
            valid_masks.push(valid);
            ind_cols_saved.push(ind_cols);
            xtx_saved.push(Vec::new());
            n_ind_saved.push(0);
            ret_ss[ti] = f64::NAN;
            continue;
        }
        // X'X (基于 barra 有效+open 的因子无关集) 与收益残差
        let mut xtx = vec![0.0f64; p * p];
        let mut xty_ret = vec![0.0f64; p];
        for j in 0..n {
            if !valid[j] {
                continue;
            }
            for c in 0..10 {
                let b = barra[[ti, j, c]];
                xty_ret[c] += b * ret[[ti, j]] as f64;
                xtx[c * p + c] += b * b;
                for c2 in (c + 1)..10 {
                    let v = b * barra[[ti, j, c2]];
                    xtx[c * p + c2] += v;
                    xtx[c2 * p + c] += v;
                }
            }
            let ic = ind_cols[j];
            if ic >= 0 {
                let col = (10 + ic) as usize;
                xty_ret[col] += ret[[ti, j]] as f64;
                xtx[col * p + col] += 1.0;
                for c in 0..10 {
                    let b = barra[[ti, j, c]];
                    xtx[c * p + col] += b;
                    xtx[col * p + c] += b;
                }
            }
        }
        // 收益残差 (一次): r_res = r - X * beta_ret
        let m = nalgebra::DMatrix::from_row_slice(p, p, &xtx);
        let rhs = nalgebra::DMatrix::from_column_slice(p, 1, &xty_ret);
        if let Some(chol) = nalgebra::Cholesky::new(m.clone()) {
            let beta = chol.solve(&rhs).column(0).iter().copied().collect::<Vec<f64>>();
            if beta.len() == p {
                for j in 0..n {
                    if valid[j] {
                        let mut pred = 0.0f64;
                        for c in 0..10 {
                            pred += beta[c] * barra[[ti, j, c]];
                        }
                        let ic = ind_cols[j];
                        if ic >= 0 {
                            pred += beta[(10 + ic) as usize];
                        }
                        r_res[[ti, j]] = ret[[ti, j]] as f64 - pred;
                    }
                }
            }
        }
        // 收益中心化平方和 (有效集)
        let mut mr = 0.0;
        let mut cnt = 0.0;
        for j in 0..n {
            if valid[j] && ret[[ti, j]].is_finite() {
                mr += ret[[ti, j]] as f64;
                cnt += 1.0;
            }
        }
        if cnt > 0.0 {
            mr /= cnt;
            let mut ss = 0.0;
            for j in 0..n {
                if valid[j] && ret[[ti, j]].is_finite() {
                    let dv = ret[[ti, j]] as f64 - mr;
                    ss += dv * dv;
                }
            }
            ret_ss[ti] = ss;
        } else {
            ret_ss[ti] = f64::NAN;
        }
        valid_masks.push(valid);
        ind_cols_saved.push(ind_cols);
        xtx_saved.push(xtx);
        n_ind_saved.push(n_ind);
    }

    // ---------- 因子侧: 每 slot 每期仅 O(N*K) 投影 + O(N) 相关 ----------
    let mut out = Array2::<f64>::from_elem((f, t), f64::NAN);
    for fi in 0..f {
        for ti in 0..t {
            let xtx = &xtx_saved[ti];
            let n_ind = n_ind_saved[ti];
            if xtx.is_empty() {
                continue;
            }
            let p = 10 + n_ind;
            let valid = &valid_masks[ti];
            let ind_cols = &ind_cols_saved[ti];
            // X'y_x
            let mut xty_x = vec![0.0f64; p];
            let mut sum_x = 0.0;
            let mut cnt = 0.0;
            for j in 0..n {
                if valid[j] && blocks[[fi, ti, j]].is_finite() {
                    let y = blocks[[fi, ti, j]] as f64;
                    for c in 0..10 {
                        xty_x[c] += barra[[ti, j, c]] * y;
                    }
                    let ic = ind_cols[j];
                    if ic >= 0 {
                        xty_x[(10 + ic) as usize] += y;
                    }
                    sum_x += y;
                    cnt += 1.0;
                }
            }
            if cnt <= 10.0 {
                continue;
            }
            let m = nalgebra::DMatrix::from_row_slice(p, p, xtx);
            let rhs = nalgebra::DMatrix::from_column_slice(p, 1, &xty_x);
            let Some(chol) = nalgebra::Cholesky::new(m) else { continue; };
            let beta = chol.solve(&rhs).column(0).iter().copied().collect::<Vec<f64>>();
            if beta.len() != p {
                continue;
            }
            // |M_B x|^2 = sum(y^2) - beta' * X'y  (残差自动零均值)
            let mut yy = 0.0;
            let mut beta_xtx = 0.0;
            for c in 0..p {
                beta_xtx += beta[c] * xty_x[c];
            }
            let mut sum_x2 = 0.0;
            for j in 0..n {
                if valid[j] && blocks[[fi, ti, j]].is_finite() {
                    let y = blocks[[fi, ti, j]] as f64;
                    yy += y * y;
                    sum_x2 += y;
                }
            }
            let res_ss = yy - beta_xtx; // 残差平方和 (M_B 含截距 -> 自动零均值; 与 old 的 vs 同口径)
            if res_ss <= 0.0 || ret_ss[ti].is_nan() || ret_ss[ti] <= 0.0 {
                continue;
            }
            // 分子 cov(x, r_res) 与 去均值
            let x_bar = sum_x / cnt;
            let mut cov = 0.0;
            for j in 0..n {
                if valid[j] && blocks[[fi, ti, j]].is_finite() && r_res[[ti, j]].is_finite() {
                    cov += (blocks[[fi, ti, j]] as f64 - x_bar) * r_res[[ti, j]];
                }
            }
            // IC_A(pearson) = cov(x, r_res) / (sqrt(res_ss) * sqrt(ret_ss))
            out[[fi, ti]] = cov / (res_ss.sqrt() * ret_ss[ti].sqrt());
        }
    }
    Ok(out.into_pyarray(py).to_owned())
}

#[pymodule]
fn dev_sandbox(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sandbox_ic_old, m)?)?;
    m.add_function(wrap_pyfunction!(sandbox_ic_new, m)?)?;
    m.add_function(wrap_pyfunction!(sandbox_ic_pearson_old, m)?)?;
    m.add_function(wrap_pyfunction!(sandbox_ic_pearson_new, m)?)?;
    m.add_function(wrap_pyfunction!(sandbox_ic_pearson_fast, m)?)?;
    Ok(())
}

