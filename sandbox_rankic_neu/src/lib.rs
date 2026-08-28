// ============================================================================
// sandbox_rankic_neu: RankIC 中性化方案独立验证
//
// 从零实现（不引用生产代码文本）:
//   1. 现状链路复刻 neutralize_full  (rank_pct -> 行业OLS填充 -> 中位填充
//      -> restrict置空 -> rank_pct -> OLS残差 -> rank_pct)，输出与生产
//      rp.neutralize_std_block_py 逐位对齐。
//   2. 方案 C : 省掉残差最终 rank_pct（数学上 Spearman 对保序变换不变）。
//   3. 方案 C3: 跳过填充的快速路径（仅当因子无 NaN 且 size 无 NaN 时与现状等价）。
//   4. 方案 B : 收益侧中性化（同一链路作用在 ret 上）。
//   5. legacy_spearman_correlation 复刻 + 回测 IC 序列复刻。
//
// 数值路径与生产一致: f32->f64 输入, 行业模式下 X=[10风格, ind1 one-hot],
// Cholesky 解正规方程 (n_valid>40), 残差 f64->f32 cast。
// ============================================================================
use nalgebra::{Cholesky, DMatrix};
use ndarray::{Array2, Array3};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::prelude::*;

// ---------- 基础: pandas rank(axis=1, pct=True) 语义 ----------

fn cmp_f64(a: &f64, b: &f64) -> std::cmp::Ordering {
    match (a.is_nan(), b.is_nan()) {
        (true, true) => std::cmp::Ordering::Equal,
        (true, false) => std::cmp::Ordering::Greater,
        (false, true) => std::cmp::Ordering::Less,
        (false, false) => a
            .partial_cmp(b)
            .unwrap_or(std::cmp::Ordering::Equal),
    }
}

/// 行内非 NaN 平均秩, pct = avg_rank / n_non_nan; NaN 保持。
/// tie 用 f64 == 判等 (与生产一致, 含 -0.0/+0.0 合并)。
/// sort_unstable: tie 组位置只由组边界决定, 与组内顺序无关, 结果确定。
fn rank_pct_row(vals: &[f64], out: &mut [f64]) {
    let n = vals.len();
    for v in out.iter_mut() {
        *v = f64::NAN;
    }
    let mut idxs: Vec<usize> = (0..n).filter(|&i| !vals[i].is_nan()).collect();
    let nv = idxs.len();
    if nv == 0 {
        return;
    }
    idxs.sort_unstable_by(|&a, &b| cmp_f64(&vals[a], &vals[b]));
    let mut i = 0;
    while i < nv {
        let mut j = i;
        while j + 1 < nv && vals[idxs[j + 1]] == vals[idxs[i]] {
            j += 1;
        }
        let avg = ((i + 1) + (j + 1)) as f64 / 2.0;
        let pct = avg / nv as f64;
        for &k in &idxs[i..=j] {
            out[k] = pct;
        }
        i = j + 1;
    }
}

fn rank_pct_all(a: &mut Array2<f64>) {
    let n = a.ncols();
    let mut out = vec![0.0; n];
    for mut row in a.rows_mut() {
        rank_pct_row(row.as_slice().unwrap(), &mut out);
        for (j, &v) in out.iter().enumerate() {
            row[j] = v;
        }
    }
}

/// 2 参数 OLS 闭式 (y ~ 1 + b)。
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

/// 行业 OLS 填充: 3 层 (ind2 -> ind1 -> ind0), 每层对当日行业段做 2 参数 OLS
/// (base = size rank pct), 填充因子 NaN 位置 (含 base NaN -> 结果 NaN)。
/// 当日因子 unique < 10 跳过; 段内观测 < 10 跳过。
fn fill_ind_reg(fv: &mut Array2<f64>, ind: &Array2<f64>, size_ranked: &Array2<f64>) {
    let (t, n) = fv.dim();
    let ind1 = ind.map(|&v| (v / 10000.0).floor());
    let ind2 = ind.map(|&v| (v / 100.0).floor());
    let ind0 = ind1.map(|&v| if v.is_nan() { 0.0 } else { 1.0 });
    let levels = [ind2, ind1, ind0];
    let mut order: Vec<usize> = (0..n).collect();
    let mut ys: Vec<f64> = Vec::with_capacity(n);
    let mut bs: Vec<f64> = Vec::with_capacity(n);
    for level in &levels {
        for idx in 0..t {
            let mut row = fv.row(idx).to_vec();
            if !has_ge_n_unique(&row, 10) {
                continue;
            }
            let codes = level.row(idx).to_vec();
            order.sort_by(|&a, &b| cmp_f64(&codes[a], &codes[b]));
            let mut seg_start = 0usize;
            while seg_start < n {
                let code = codes[order[seg_start]];
                if code.is_nan() {
                    break;
                }
                let mut seg_end = seg_start + 1;
                while seg_end < n && codes[order[seg_end]] == code {
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

/// 分组中位填充: median 固定来自 source。
fn group_median_fill(
    values: &mut Array2<f64>,
    codes: &Array2<f64>,
    valid_mask: Option<&Array2<f64>>,
    source: &Array2<f64>,
) {
    let (t, n) = values.dim();
    let mut order: Vec<usize> = (0..n).collect();
    let mut sv: Vec<f64> = Vec::with_capacity(n);
    for idx in 0..t {
        let mut row = values.row(idx).to_vec();
        let src = source.row(idx).to_vec();
        let nan_mask: Vec<bool> = row.iter().map(|v| v.is_nan()).collect();
        if !nan_mask.iter().any(|&b| b) {
            continue;
        }
        let codes_row = codes.row(idx).to_vec();
        order.sort_unstable_by(|&a, &b| cmp_f64(&codes_row[a], &codes_row[b]));
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
                if valid && !src[ci].is_nan() {
                    sv.push(src[ci]);
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

/// 逐日 OLS 残差。industry=true: X=[10风格, ind1 one-hot] (行业NaN哑变量全0);
/// industry=false: X=[1, 10风格]。n_valid<=10 跳过; y unique==1 -> 残差 0.5。
/// 与生产同序的 X'X/X'y 累加 + Cholesky (n_valid>40), 否则 SVD 伪逆。
fn get_residual(
    fv: &Array2<f64>,
    barra_ranked: &[Array2<f64>],
    ind: &Array2<f64>,
    industry: bool,
) -> Array2<f64> {
    let (t, n) = fv.dim();
    let k = barra_ranked.len();
    let ind1 = ind.map(|&v| (v / 10000.0).floor());
    let mut resid = Array2::<f64>::from_elem((t, n), f64::NAN);
    let mut valid: Vec<bool> = Vec::with_capacity(n);
    let mut rows: Vec<[f64; 11]> = Vec::with_capacity(n);
    let mut ind_cols: Vec<i32> = Vec::with_capacity(n);
    for idx in 0..t {
        let row = fv.row(idx);
        if !row.iter().any(|v| !v.is_nan()) {
            continue;
        }
        let mut ind_codes: Vec<f64> = Vec::new();
        if industry {
            for j in 0..n {
                let c = ind1[[idx, j]];
                if !c.is_nan() && !ind_codes.contains(&c) {
                    ind_codes.push(c);
                }
            }
            ind_codes.sort_by(cmp_f64);
        }
        let n_ind = if industry { ind_codes.len() } else { 0 };
        let p = if industry { k + n_ind } else { k + 1 };
        valid.clear();
        rows.clear();
        ind_cols.clear();
        for j in 0..n {
            let ok = row[j].is_finite()
                && barra_ranked.iter().all(|b| b[[idx, j]].is_finite());
            valid.push(ok);
            if ok {
                let mut cur = [0.0_f64; 11];
                cur[0] = row[j];
                for c in 0..k {
                    cur[c + 1] = barra_ranked[c][[idx, j]];
                }
                rows.push(cur);
                if industry {
                    let c = ind1[[idx, j]];
                    ind_cols.push(if c.is_nan() {
                        -1
                    } else {
                        match ind_codes.binary_search_by(|x| cmp_f64(x, &c)) {
                            Ok(pos) => pos as i32,
                            Err(_) => -1,
                        }
                    });
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
            let mut xtx = vec![0.0_f64; p * p];
            let mut xty = vec![0.0_f64; p];
            for (i, r) in rows.iter().enumerate() {
                let yv = r[0];
                if !industry {
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
                let xm = DMatrix::from_fn(n_valid, p, |r_i, c| {
                    if !industry {
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
                let rcond = s_max * (n_valid.max(p) as f64) * 2.22e-16;
                let ym = DMatrix::from_fn(n_valid, 1, |r_i, _| rows[r_i][0]);
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
                if !industry {
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

/// 完整链路复刻 (barra_ranked 已预计算, 与生产 neutralize_std_section 对齐)。
/// 返回 (x_neu = resid rank_pct, resid, a = OLS 输入 rank_pct)。
fn neutralize_core_s(
    factor: &Array2<f64>,
    ind: &Array2<f64>,
    restrict: &Array2<f64>,
    barra_ranked: &[Array2<f64>],
    industry: bool,
) -> (Array2<f64>, Array2<f64>, Array2<f64>) {
    let (t, n) = factor.dim();
    let size_ranked = barra_ranked[2].clone();

    // 1. factor rank pct
    let mut fv_ranked = factor.clone();
    rank_pct_all(&mut fv_ranked);

    // 2. 行业 OLS 填充
    fill_ind_reg(&mut fv_ranked, ind, &size_ranked);

    // 3. 三级中位填充 (median_source 固定)
    let ind1 = ind.map(|&v| (v / 10000.0).floor());
    let ind2 = ind.map(|&v| (v / 100.0).floor());
    let zeros = Array2::<f64>::zeros((t, n));
    let ind1_mask = ind1.map(|&v| if v.is_nan() { 0.0 } else { 1.0 });
    let mut fv_filled = fv_ranked.clone();
    for i in 0..(t * n) {
        if ind1.as_slice().unwrap()[i].is_nan() {
            fv_filled.as_slice_mut().unwrap()[i] = f64::NAN;
        }
    }
    let median_source = fv_filled.clone();
    group_median_fill(&mut fv_filled, &ind2, None, &median_source);
    group_median_fill(&mut fv_filled, &ind1, None, &median_source);
    group_median_fill(&mut fv_filled, &zeros, Some(&ind1_mask), &median_source);

    // 4. restrict 置空
    for i in 0..(t * n) {
        if restrict.as_slice().unwrap()[i] != 0.0 {
            fv_filled.as_slice_mut().unwrap()[i] = f64::NAN;
        }
    }

    // 5. rank pct
    rank_pct_all(&mut fv_filled);

    // 6. 残差
    let resid = get_residual(&fv_filled, &barra_ranked, ind, industry);

    // 7. 残差 rank pct
    let mut x_neu = resid.clone();
    rank_pct_all(&mut x_neu);
    (x_neu, resid, fv_filled)
}

/// 完整链路复刻 (barra_raw 原始值输入, 内部先 rank)。
fn neutralize_core(
    factor: &Array2<f64>,
    ind: &Array2<f64>,
    restrict: &Array2<f64>,
    barra_raw: &Array3<f64>,
    industry: bool,
) -> (Array2<f64>, Array2<f64>, Array2<f64>) {
    let (t, n) = factor.dim();
    let barra_ranked: Vec<Array2<f64>> = (0..10)
        .map(|f| {
            let mut m = Array2::<f64>::zeros((t, n));
            for i in 0..t {
                for j in 0..n {
                    m[[i, j]] = barra_raw[[i, j, f]];
                }
            }
            rank_pct_all(&mut m);
            m
        })
        .collect();
    neutralize_core_s(factor, ind, restrict, &barra_ranked, industry)
}

fn to_f32(a: &Array2<f64>) -> Array2<f32> {
    a.map(|&v| if v.is_nan() { f32::NAN } else { v as f32 })
}

// ---------- legacy_spearman + 回测 IC ----------

fn ordinal_ranks(values: &[f32]) -> Vec<i64> {
    let mut indexed: Vec<(usize, f32)> = values.iter().copied().enumerate().collect();
    indexed.sort_by(|l, r| match (l.1.is_nan(), r.1.is_nan()) {
        (true, true) => l.0.cmp(&r.0),
        (true, false) => std::cmp::Ordering::Greater,
        (false, true) => std::cmp::Ordering::Less,
        (false, false) => l
            .1
            .partial_cmp(&r.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| l.0.cmp(&r.0)),
    });
    let mut ranks = vec![0i64; values.len()];
    for (rank, (idx, _)) in indexed.iter().enumerate() {
        ranks[*idx] = rank as i64;
    }
    ranks
}

/// f64 域 ordinal rank（用于 resid 直接算秩，tie 按索引，与生产的
/// rank_pct(resid) 再 ordinal 的秩完全一致——rank_pct 保序且 tie 保持）。
fn ordinal_ranks_f64(values: &[f64]) -> Vec<i64> {
    let mut indexed: Vec<(usize, f64)> = values.iter().copied().enumerate().collect();
    indexed.sort_by(|l, r| match (l.1.is_nan(), r.1.is_nan()) {
        (true, true) => l.0.cmp(&r.0),
        (true, false) => std::cmp::Ordering::Greater,
        (false, true) => std::cmp::Ordering::Less,
        (false, false) => l
            .1
            .partial_cmp(&r.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| l.0.cmp(&r.0)),
    });
    let mut ranks = vec![0i64; values.len()];
    for (rank, (idx, _)) in indexed.iter().enumerate() {
        ranks[*idx] = rank as i64;
    }
    ranks
}

/// 信号用 f64 域秩、收益用 f32 域秩的 Spearman（方案 C' 核心）。
fn spearman_mixed(x: &[f64], y: &[f32]) -> f64 {
    if x.len() != y.len() || x.len() < 2 {
        return f64::NAN;
    }
    let xx = ordinal_ranks_f64(x);
    let yy = ordinal_ranks(y);
    let n = x.len() as f64;
    let mut d2 = 0.0;
    for i in 0..x.len() {
        let d = xx[i] - yy[i];
        d2 += (d * d) as f64;
    }
    1.0 - 6.0 * d2 / (n * (n * n - 1.0))
}

fn legacy_spearman(x: &[f32], y: &[f32]) -> f64 {
    if x.len() != y.len() || x.len() < 2 {
        return f64::NAN;
    }
    let xx = ordinal_ranks(x);
    let yy = ordinal_ranks(y);
    let n = x.len() as f64;
    let mut d2 = 0.0;
    for i in 0..x.len() {
        let d = xx[i] - yy[i];
        d2 += (d * d) as f64;
    }
    1.0 - 6.0 * d2 / (n * (n * n - 1.0))
}

/// 回测 IC 序列 (复刻 legacy_backtest_single_factor_with_effective 的 IC 部分)。
fn ic_series(
    sig: ndarray::ArrayView2<'_, f32>,
    ret: ndarray::ArrayView2<'_, f32>,
    restrict: ndarray::ArrayView2<'_, f32>,
    dates: &[i32],
    backtest_start: i32,
    gap: usize,
) -> Vec<f32> {
    let n_dates = sig.nrows();
    let n_stocks = sig.ncols();
    let mut eff: Vec<usize> = Vec::new();
    for t in 1..n_dates {
        if dates[t] <= backtest_start {
            continue;
        }
        let mut all_nan = true;
        for s in 0..n_stocks {
            if sig[[t - 1, s]].is_finite() {
                all_nan = false;
                break;
            }
        }
        if !all_nan {
            eff.push(t);
        }
    }
    let mut out: Vec<f32> = Vec::new();
    if eff.is_empty() {
        return out;
    }
    let mut held = eff[0] - 1;
    let mut fs: Vec<f32> = Vec::with_capacity(n_stocks);
    let mut fr: Vec<f32> = Vec::with_capacity(n_stocks);
    for (local_t, &t) in eff.iter().enumerate() {
        if local_t % gap == 0 {
            held = t - 1;
        }
        if (local_t + 1) % gap != 0 {
            continue;
        }
        fs.clear();
        fr.clear();
        for s in 0..n_stocks {
            let sv = sig[[held, s]];
            let rv = ret[[t, s]];
            let is_open = restrict[[held, s]].is_finite() && restrict[[held, s]] == 0.0;
            if sv.is_finite() && rv.is_finite() && is_open {
                fs.push(sv);
                fr.push(rv);
            }
        }
        out.push(legacy_spearman(&fr, &fs) as f32);
    }
    out
}

/// 方案 C' 的 IC 序列：信号为 resid(f64) 直接 f64 域求秩，与现状 A 逐位一致。
fn ic_series_resid64(
    resid: ndarray::ArrayView2<'_, f64>,
    ret: ndarray::ArrayView2<'_, f32>,
    restrict: ndarray::ArrayView2<'_, f32>,
    dates: &[i32],
    backtest_start: i32,
    gap: usize,
) -> Vec<f32> {
    let n_dates = resid.nrows();
    let n_stocks = resid.ncols();
    let mut eff: Vec<usize> = Vec::new();
    for t in 1..n_dates {
        if dates[t] <= backtest_start {
            continue;
        }
        let mut all_nan = true;
        for s in 0..n_stocks {
            if resid[[t - 1, s]].is_finite() {
                all_nan = false;
                break;
            }
        }
        if !all_nan {
            eff.push(t);
        }
    }
    let mut out: Vec<f32> = Vec::new();
    if eff.is_empty() {
        return out;
    }
    let mut held = eff[0] - 1;
    let mut fs: Vec<f64> = Vec::with_capacity(n_stocks);
    let mut fr: Vec<f32> = Vec::with_capacity(n_stocks);
    for (local_t, &t) in eff.iter().enumerate() {
        if local_t % gap == 0 {
            held = t - 1;
        }
        if (local_t + 1) % gap != 0 {
            continue;
        }
        fs.clear();
        fr.clear();
        for s in 0..n_stocks {
            let sv = resid[[held, s]];
            let rv = ret[[t, s]];
            let is_open = restrict[[held, s]].is_finite() && restrict[[held, s]] == 0.0;
            if sv.is_finite() && rv.is_finite() && is_open {
                fs.push(sv);
                fr.push(rv);
            }
        }
        out.push(spearman_mixed(&fs, &fr) as f32);
    }
    out
}

// ---------- 方案 C'': 预计算行业码排序 (与因子无关, 逐位一致) ----------

/// 预计算 fill_ind_reg / group_median_fill 需要的全部行排序:
/// 返回 [ind2, ind1, ind0, ind2, ind1] 五个 (T,N) 排序索引矩阵
/// (fill_ind_reg 3 层 + 中位填充 2 层; zeros 层免排序)。
fn precompute_ind_orders(ind: &Array2<f64>) -> Vec<Array2<usize>> {
    let (t, n) = ind.dim();
    let ind1 = ind.map(|&v| (v / 10000.0).floor());
    let ind2 = ind.map(|&v| (v / 100.0).floor());
    let ind0 = ind1.map(|&v| if v.is_nan() { 0.0 } else { 1.0 });
    let levels = [&ind2, &ind1, &ind0, &ind2, &ind1];
    levels
        .iter()
        .map(|codes| {
            let mut orders = Array2::<usize>::zeros((t, n));
            for idx in 0..t {
                let mut order: Vec<usize> = (0..n).collect();
                let codes_row = codes.row(idx).to_vec();
                order.sort_by(|&a, &b| cmp_f64(&codes_row[a], &codes_row[b]));
                for j in 0..n {
                    orders[[idx, j]] = order[j];
                }
            }
            orders
        })
        .collect()
}

/// 带预计算排序的行业 OLS 填充。
fn fill_ind_reg_pre(
    fv: &mut Array2<f64>,
    ind: &Array2<f64>,
    size_ranked: &Array2<f64>,
    orders: &[Array2<usize>],
) {
    let (t, n) = fv.dim();
    let ind1 = ind.map(|&v| (v / 10000.0).floor());
    let ind2 = ind.map(|&v| (v / 100.0).floor());
    let ind0 = ind1.map(|&v| if v.is_nan() { 0.0 } else { 1.0 });
    let levels = [&ind2, &ind1, &ind0];
    let mut ys: Vec<f64> = Vec::with_capacity(n);
    let mut bs: Vec<f64> = Vec::with_capacity(n);
    for (li, level) in levels.iter().enumerate() {
        for idx in 0..t {
            let mut row = fv.row(idx).to_vec();
            if !has_ge_n_unique(&row, 10) {
                continue;
            }
            let order = orders[li].row(idx).to_vec();
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
}

/// 带预计算排序的分组中位填充。
fn group_median_fill_pre(
    values: &mut Array2<f64>,
    codes: &Array2<f64>,
    valid_mask: Option<&Array2<f64>>,
    source: &Array2<f64>,
    orders: &Array2<usize>,
) {
    let (t, n) = values.dim();
    let mut sv: Vec<f64> = Vec::with_capacity(n);
    for idx in 0..t {
        let mut row = values.row(idx).to_vec();
        let src = source.row(idx).to_vec();
        let nan_mask: Vec<bool> = row.iter().map(|v| v.is_nan()).collect();
        if !nan_mask.iter().any(|&b| b) {
            continue;
        }
        let order = orders.row(idx).to_vec();
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
                if valid && !src[ci].is_nan() {
                    sv.push(src[ci]);
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

/// 方案 C'': 完整链路 + 预计算行业排序 + 省残差 rank_pct。
/// 输出 resid(f64), 与现状 IC 逐位一致。barra_ranked 预计算共享。
#[allow(clippy::too_many_arguments)]
fn neutralize_cpp_core(
    factor: &Array2<f64>,
    ind: &Array2<f64>,
    restrict: &Array2<f64>,
    barra_ranked: &[Array2<f64>],
    industry: bool,
    orders: &[Array2<usize>],
) -> Array2<f64> {
    let (t, n) = factor.dim();
    let size_ranked = barra_ranked[2].clone();

    let mut fv_ranked = factor.clone();
    rank_pct_all(&mut fv_ranked);
    fill_ind_reg_pre(&mut fv_ranked, ind, &size_ranked, &orders[..3]);

    let ind1 = ind.map(|&v| (v / 10000.0).floor());
    let ind2 = ind.map(|&v| (v / 100.0).floor());
    let zeros = Array2::<f64>::zeros((t, n));
    let ind1_mask = ind1.map(|&v| if v.is_nan() { 0.0 } else { 1.0 });
    let mut fv_filled = fv_ranked.clone();
    for i in 0..(t * n) {
        if ind1.as_slice().unwrap()[i].is_nan() {
            fv_filled.as_slice_mut().unwrap()[i] = f64::NAN;
        }
    }
    let median_source = fv_filled.clone();
    group_median_fill_pre(&mut fv_filled, &ind2, None, &median_source, &orders[3]);
    group_median_fill_pre(&mut fv_filled, &ind1, None, &median_source, &orders[4]);
    group_median_fill(&mut fv_filled, &zeros, Some(&ind1_mask), &median_source);

    for i in 0..(t * n) {
        if restrict.as_slice().unwrap()[i] != 0.0 {
            fv_filled.as_slice_mut().unwrap()[i] = f64::NAN;
        }
    }
    rank_pct_all(&mut fv_filled);
    get_residual(&fv_filled, &barra_ranked, ind, industry)
}

/// 方案 C3-exact: 因子无 NaN 时的精确快路径。
/// = 抹 size-NaN 位置 -> 三级中位填充(预计算排序) -> restrict -> rank -> OLS。
/// 仅当因子无 NaN 时与现状逐位一致 (fill_ind_reg 的副作用只有抹 size-NaN,
/// 与完整链路逐位相同)。
fn neutralize_c3_exact_core(
    factor: &Array2<f64>,
    ind: &Array2<f64>,
    restrict: &Array2<f64>,
    barra_ranked: &[Array2<f64>],
    industry: bool,
    orders: &[Array2<usize>],
) -> Array2<f64> {
    let (t, n) = factor.dim();
    let size_ranked = &barra_ranked[2];

    // 因子无 NaN: rank_pct 是仿射 (avg_rank/n), 直接跳过 (affine 不影响残差秩)。
    // fill_ind_reg 的副作用: 抹 size-NaN 位置 (因子有值但 base NaN -> NaN)。
    let mut fv = factor.clone();
    for i in 0..(t * n) {
        if size_ranked.as_slice().unwrap()[i].is_nan() {
            fv.as_slice_mut().unwrap()[i] = f64::NAN;
        }
    }
    // 三级中位填充 (median_source = 抹后矩阵)
    let ind1 = ind.map(|&v| (v / 10000.0).floor());
    let ind2 = ind.map(|&v| (v / 100.0).floor());
    let zeros = Array2::<f64>::zeros((t, n));
    let ind1_mask = ind1.map(|&v| if v.is_nan() { 0.0 } else { 1.0 });
    for i in 0..(t * n) {
        if ind1.as_slice().unwrap()[i].is_nan() {
            fv.as_slice_mut().unwrap()[i] = f64::NAN;
        }
    }
    let source = fv.clone();
    group_median_fill_pre(&mut fv, &ind2, None, &source, &orders[3]);
    group_median_fill_pre(&mut fv, &ind1, None, &source, &orders[4]);
    group_median_fill(&mut fv, &zeros, Some(&ind1_mask), &source);

    for i in 0..(t * n) {
        if restrict.as_slice().unwrap()[i] != 0.0 {
            fv.as_slice_mut().unwrap()[i] = f64::NAN;
        }
    }
    rank_pct_all(&mut fv);
    get_residual(&fv, &barra_ranked, ind, industry)
}

// ---------- pyo3 绑定 ----------

#[pyfunction]
#[pyo3(signature = (factor, ind, restrict, barra_raw, industry=true))]
fn neutralize_full<'py>(
    py: Python<'py>,
    factor: PyReadonlyArray2<'py, f32>,
    ind: PyReadonlyArray2<'py, f64>,
    restrict: PyReadonlyArray2<'py, f32>,
    barra_raw: PyReadonlyArray3<'py, f64>,
    industry: bool,
) -> PyResult<(Py<PyArray2<f32>>, Py<PyArray2<f32>>)> {
    let f = factor.as_array().map(|&v| v as f64);
    let i = ind.as_array().to_owned();
    let r = restrict.as_array().map(|&v| v as f64);
    let b = barra_raw.as_array().to_owned();
    let (x_neu, resid, _a) = neutralize_core(&f, &i, &r, &b, industry);
    Ok((to_f32(&x_neu).into_pyarray(py).to_owned(), to_f32(&resid).into_pyarray(py).to_owned()))
}

/// 方案 C: 完整链路但省残差最终 rank_pct (返回 resid, f64 保留全精度)。
#[pyfunction]
#[pyo3(signature = (factor, ind, restrict, barra_raw, industry=true))]
fn neutralize_c<'py>(
    py: Python<'py>,
    factor: PyReadonlyArray2<'py, f32>,
    ind: PyReadonlyArray2<'py, f64>,
    restrict: PyReadonlyArray2<'py, f32>,
    barra_raw: PyReadonlyArray3<'py, f64>,
    industry: bool,
) -> PyResult<Py<PyArray2<f64>>> {
    let f = factor.as_array().map(|&v| v as f64);
    let i = ind.as_array().to_owned();
    let r = restrict.as_array().map(|&v| v as f64);
    let b = barra_raw.as_array().to_owned();
    let (_, resid, _a) = neutralize_core(&f, &i, &r, &b, industry);
    Ok(resid.into_pyarray(py).to_owned())
}

/// 方案 C3: 跳过填充, 直接 restrict 置空 + rank_pct + OLS 残差 (无最终 rank)。
/// 仅当因子无 NaN 且 size(value_2) 无 NaN 时与现状逐位一致。
#[pyfunction]
#[pyo3(signature = (factor, ind, restrict, barra_raw, industry=true))]
fn neutralize_c3<'py>(
    py: Python<'py>,
    factor: PyReadonlyArray2<'py, f32>,
    ind: PyReadonlyArray2<'py, f64>,
    restrict: PyReadonlyArray2<'py, f32>,
    barra_raw: PyReadonlyArray3<'py, f64>,
    industry: bool,
) -> PyResult<Py<PyArray2<f32>>> {
    let f = factor.as_array().map(|&v| v as f64);
    let i = ind.as_array().to_owned();
    let r = restrict.as_array().map(|&v| v as f64);
    let b = barra_raw.as_array().to_owned();
    let (t, n) = f.dim();
    let mut barra_ranked: Vec<Array2<f64>> = (0..10)
        .map(|f_i| {
            let mut m = Array2::<f64>::zeros((t, n));
            for i2 in 0..t {
                for j2 in 0..n {
                    m[[i2, j2]] = b[[i2, j2, f_i]];
                }
            }
            rank_pct_all(&mut m);
            m
        })
        .collect();
    let _ = &barra_ranked;
    let mut fv = f.clone();
    for i2 in 0..(t * n) {
        if r.as_slice().unwrap()[i2] != 0.0 {
            fv.as_slice_mut().unwrap()[i2] = f64::NAN;
        }
    }
    rank_pct_all(&mut fv);
    let resid = get_residual(&fv, &barra_ranked, &i, industry);
    Ok(to_f32(&resid).into_pyarray(py).to_owned())
}

/// 收益侧中性化 (方案 B): 同一完整链路作用在 ret 上。
#[pyfunction]
#[pyo3(signature = (ret, ind, restrict, barra_raw, industry=true))]
fn neutralize_ret<'py>(
    py: Python<'py>,
    ret: PyReadonlyArray2<'py, f32>,
    ind: PyReadonlyArray2<'py, f64>,
    restrict: PyReadonlyArray2<'py, f32>,
    barra_raw: PyReadonlyArray3<'py, f64>,
    industry: bool,
) -> PyResult<(Py<PyArray2<f32>>, Py<PyArray2<f32>>)> {
    let f = ret.as_array().map(|&v| v as f64);
    let i = ind.as_array().to_owned();
    let r = restrict.as_array().map(|&v| v as f64);
    let b = barra_raw.as_array().to_owned();
    let (x_neu, resid, _a) = neutralize_core(&f, &i, &r, &b, industry);
    Ok((to_f32(&x_neu).into_pyarray(py).to_owned(), to_f32(&resid).into_pyarray(py).to_owned()))
}

#[pyfunction]
fn spearman_ordinal(x: PyReadonlyArray1<'_, f32>, y: PyReadonlyArray1<'_, f32>) -> PyResult<f64> {
    Ok(legacy_spearman(x.as_slice()?, y.as_slice()?))
}

#[pyfunction]
#[pyo3(signature = (sig, ret, restrict, dates, backtest_start, gap))]
fn ic_series_py<'py>(
    py: Python<'py>,
    sig: PyReadonlyArray2<'py, f32>,
    ret: PyReadonlyArray2<'py, f32>,
    restrict: PyReadonlyArray2<'py, f32>,
    dates: Vec<i32>,
    backtest_start: i32,
    gap: usize,
) -> PyResult<Py<PyArray1<f32>>> {
    let out = ic_series(sig.as_array(), ret.as_array(), restrict.as_array(), &dates, backtest_start, gap);
    Ok(PyArray1::from_vec(py, out).to_owned())
}

#[pyfunction]
#[pyo3(signature = (resid, ret, restrict, dates, backtest_start, gap))]
fn ic_series_resid64_py<'py>(
    py: Python<'py>,
    resid: PyReadonlyArray2<'py, f64>,
    ret: PyReadonlyArray2<'py, f32>,
    restrict: PyReadonlyArray2<'py, f32>,
    dates: Vec<i32>,
    backtest_start: i32,
    gap: usize,
) -> PyResult<Py<PyArray1<f32>>> {
    let out = ic_series_resid64(resid.as_array(), ret.as_array(), restrict.as_array(), &dates, backtest_start, gap);
    Ok(PyArray1::from_vec(py, out).to_owned())
}

#[pyfunction]
fn precompute_orders<'py>(
    py: Python<'py>,
    ind: PyReadonlyArray2<'py, f64>,
) -> PyResult<Vec<Py<PyArray2<usize>>>> {
    let i = ind.as_array().to_owned();
    let orders = precompute_ind_orders(&i);
    Ok(orders
        .into_iter()
        .map(|o| o.into_pyarray(py).to_owned())
        .collect())
}

/// 预计算 10 风格 rank pct (与因子无关, 全任务共享)。
#[pyfunction]
fn precompute_barra<'py>(
    py: Python<'py>,
    barra_raw: PyReadonlyArray3<'py, f64>,
) -> PyResult<Py<PyArray3<f64>>> {
    let b = barra_raw.as_array().to_owned();
    let (t, n, _) = b.dim();
    let mut out = Array3::<f64>::zeros((t, n, 10));
    for f in 0..10 {
        let mut m = Array2::<f64>::zeros((t, n));
        for i in 0..t {
            for j in 0..n {
                m[[i, j]] = b[[i, j, f]];
            }
        }
        rank_pct_all(&mut m);
        for i in 0..t {
            for j in 0..n {
                out[[i, j, f]] = m[[i, j]];
            }
        }
    }
    Ok(out.into_pyarray(py).to_owned())
}

fn barra_slice(ranked: &Array3<f64>) -> Vec<Array2<f64>> {
    let (t, n, _) = ranked.dim();
    (0..10)
        .map(|f| {
            let mut m = Array2::<f64>::zeros((t, n));
            for i in 0..t {
                for j in 0..n {
                    m[[i, j]] = ranked[[i, j, f]];
                }
            }
            m
        })
        .collect()
}

/// 方案 C'': 预计算排序 + 省残差 rank, 输出 resid(f64)。barra_ranked 共享。
#[pyfunction]
#[pyo3(signature = (factor, ind, restrict, barra_ranked, orders, industry=true))]
fn neutralize_cpp<'py>(
    py: Python<'py>,
    factor: PyReadonlyArray2<'py, f32>,
    ind: PyReadonlyArray2<'py, f64>,
    restrict: PyReadonlyArray2<'py, f32>,
    barra_ranked: PyReadonlyArray3<'py, f64>,
    orders: Vec<PyReadonlyArray2<'py, usize>>,
    industry: bool,
) -> PyResult<Py<PyArray2<f64>>> {
    let f = factor.as_array().to_owned().map(|&v| v as f64);
    let i = ind.as_array().to_owned();
    let r = restrict.as_array().to_owned().map(|&v| v as f64);
    let b = barra_ranked.as_array().to_owned();
    let br = barra_slice(&b);
    let ords: Vec<Array2<usize>> = orders.iter().map(|o| o.as_array().to_owned()).collect();
    let resid = neutralize_cpp_core(&f, &i, &r, &br, industry, &ords);
    Ok(resid.into_pyarray(py).to_owned())
}

/// 方案 C3-exact: 因子无 NaN 时的精确快路径, 输出 resid(f64)。barra_ranked 共享。
#[pyfunction]
#[pyo3(signature = (factor, ind, restrict, barra_ranked, orders, industry=true))]
fn neutralize_c3_exact<'py>(
    py: Python<'py>,
    factor: PyReadonlyArray2<'py, f32>,
    ind: PyReadonlyArray2<'py, f64>,
    restrict: PyReadonlyArray2<'py, f32>,
    barra_ranked: PyReadonlyArray3<'py, f64>,
    orders: Vec<PyReadonlyArray2<'py, usize>>,
    industry: bool,
) -> PyResult<Py<PyArray2<f64>>> {
    let f = factor.as_array().to_owned().map(|&v| v as f64);
    let i = ind.as_array().to_owned();
    let r = restrict.as_array().to_owned().map(|&v| v as f64);
    let b = barra_ranked.as_array().to_owned();
    let br = barra_slice(&b);
    let ords: Vec<Array2<usize>> = orders.iter().map(|o| o.as_array().to_owned()).collect();
    let resid = neutralize_c3_exact_core(&f, &i, &r, &br, industry, &ords);
    Ok(resid.into_pyarray(py).to_owned())
}


/// 方案 C': 共享 barra_ranked + 省残差 rank_pct, 输出 resid(f64)。
#[pyfunction]
#[pyo3(signature = (factor, ind, restrict, barra_ranked, industry=true))]
fn neutralize_c_s<'py>(
    py: Python<'py>,
    factor: PyReadonlyArray2<'py, f32>,
    ind: PyReadonlyArray2<'py, f64>,
    restrict: PyReadonlyArray2<'py, f32>,
    barra_ranked: PyReadonlyArray3<'py, f64>,
    industry: bool,
) -> PyResult<Py<PyArray2<f64>>> {
    let f = factor.as_array().to_owned().map(|&v| v as f64);
    let i = ind.as_array().to_owned();
    let r = restrict.as_array().to_owned().map(|&v| v as f64);
    let b = barra_ranked.as_array().to_owned();
    let br = barra_slice(&b);
    let (_, resid, _a) = neutralize_core_s(&f, &i, &r, &br, industry);
    Ok(resid.into_pyarray(py).to_owned())
}


/// 测速用: 仅做 barra (T,N,10) -> 10×(T,N) 切片拷贝。
#[pyfunction]
fn barra_slice_py<'py>(
    py: Python<'py>,
    barra_ranked: PyReadonlyArray3<'py, f64>,
) -> PyResult<Vec<Py<PyArray2<f64>>>> {
    let b = barra_ranked.as_array().to_owned();
    let br = barra_slice(&b);
    Ok(br.into_iter().map(|m| m.into_pyarray(py).to_owned()).collect())
}

/// 阶段计时版: 返回 (resid, [barra_rank, rank1, fill_ind, fill_med, restrict_rank, ols, resid_rank])。
#[pyfunction]
#[pyo3(signature = (factor, ind, restrict, barra_raw, industry=true))]
fn neutralize_prof<'py>(
    py: Python<'py>,
    factor: PyReadonlyArray2<'py, f32>,
    ind: PyReadonlyArray2<'py, f64>,
    restrict: PyReadonlyArray2<'py, f32>,
    barra_raw: PyReadonlyArray3<'py, f64>,
    industry: bool,
) -> PyResult<(Py<PyArray2<f64>>, Vec<f64>)> {
    let f = factor.as_array().to_owned().map(|&v| v as f64);
    let i = ind.as_array().to_owned();
    let r = restrict.as_array().to_owned().map(|&v| v as f64);
    let b = barra_raw.as_array().to_owned();
    let (t, n) = f.dim();
    let mut times = vec![0.0f64; 7];
    let mut t0 = std::time::Instant::now();
    let mut barra_ranked: Vec<Array2<f64>> = (0..10)
        .map(|fi| {
            let mut m = Array2::<f64>::zeros((t, n));
            for i2 in 0..t {
                for j2 in 0..n {
                    m[[i2, j2]] = b[[i2, j2, fi]];
                }
            }
            rank_pct_all(&mut m);
            m
        })
        .collect();
    times[0] = t0.elapsed().as_secs_f64();
    let size_ranked = barra_ranked[2].clone();

    t0 = std::time::Instant::now();
    let mut fv_ranked = f.clone();
    rank_pct_all(&mut fv_ranked);
    times[1] = t0.elapsed().as_secs_f64();

    t0 = std::time::Instant::now();
    fill_ind_reg(&mut fv_ranked, &i, &size_ranked);
    times[2] = t0.elapsed().as_secs_f64();

    t0 = std::time::Instant::now();
    let ind1 = i.map(|&v| (v / 10000.0).floor());
    let ind2 = i.map(|&v| (v / 100.0).floor());
    let zeros = Array2::<f64>::zeros((t, n));
    let ind1_mask = ind1.map(|&v| if v.is_nan() { 0.0 } else { 1.0 });
    let mut fv_filled = fv_ranked.clone();
    for i2 in 0..(t * n) {
        if ind1.as_slice().unwrap()[i2].is_nan() {
            fv_filled.as_slice_mut().unwrap()[i2] = f64::NAN;
        }
    }
    let median_source = fv_filled.clone();
    group_median_fill(&mut fv_filled, &ind2, None, &median_source);
    group_median_fill(&mut fv_filled, &ind1, None, &median_source);
    group_median_fill(&mut fv_filled, &zeros, Some(&ind1_mask), &median_source);
    times[3] = t0.elapsed().as_secs_f64();

    t0 = std::time::Instant::now();
    for i2 in 0..(t * n) {
        if r.as_slice().unwrap()[i2] != 0.0 {
            fv_filled.as_slice_mut().unwrap()[i2] = f64::NAN;
        }
    }
    rank_pct_all(&mut fv_filled);
    times[4] = t0.elapsed().as_secs_f64();

    t0 = std::time::Instant::now();
    let resid = get_residual(&fv_filled, &barra_ranked, &i, industry);
    times[5] = t0.elapsed().as_secs_f64();

    t0 = std::time::Instant::now();
    let mut x_neu = resid.clone();
    rank_pct_all(&mut x_neu);
    times[6] = t0.elapsed().as_secs_f64();
    Ok((resid.into_pyarray(py).to_owned(), times))
}

#[pymodule]
fn dev_sandbox_rankic(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(neutralize_full, m)?)?;
    m.add_function(wrap_pyfunction!(neutralize_c, m)?)?;
    m.add_function(wrap_pyfunction!(neutralize_c3, m)?)?;
    m.add_function(wrap_pyfunction!(neutralize_ret, m)?)?;
    m.add_function(wrap_pyfunction!(spearman_ordinal, m)?)?;
    m.add_function(wrap_pyfunction!(ic_series_py, m)?)?;
    m.add_function(wrap_pyfunction!(ic_series_resid64_py, m)?)?;
    m.add_function(wrap_pyfunction!(precompute_orders, m)?)?;
    m.add_function(wrap_pyfunction!(neutralize_cpp, m)?)?;
    m.add_function(wrap_pyfunction!(neutralize_c3_exact, m)?)?;
    m.add_function(wrap_pyfunction!(neutralize_prof, m)?)?;
    m.add_function(wrap_pyfunction!(precompute_barra, m)?)?;
    m.add_function(wrap_pyfunction!(neutralize_c_s, m)?)?;
    m.add_function(wrap_pyfunction!(barra_slice_py, m)?)?;
    Ok(())
}
