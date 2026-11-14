//! 冷门中性化加速内核（Plan 2 `project_out` + Plan 3 `build_heat`）
//!
//! 严格复刻 `design_whatever/cold_neutral.py` 的数学与边界口径：
//! - `cold_project_out_2d` / `cold_project_out_3d`: 逐日逐列有效集 `pct=True` 平均秩 + 冷得分单变量回归残差
//! - `cold_build_heat_core`: 8 个热度指标滚动统计 + 截面平均秩标准化 + 10 风格截面 OLS 残差合成

use nalgebra::{DMatrix, SVD};
use ndarray::{Array2, Array3, ArrayView2, ArrayView3};
use numpy::{IntoPyArray, PyArray2, PyArray3, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

#[inline]
fn f64_mono_key(v: f64) -> u64 {
    let b = if v == 0.0 { 0.0f64.to_bits() } else { v.to_bits() };
    if (b >> 63) != 0 {
        !b
    } else {
        b ^ (1u64 << 63)
    }
}

fn radix_sort_u64(keys: &[u64], order: &mut [usize], tmp: &mut [usize]) {
    let n = order.len();
    if n <= 64 {
        order.sort_by_key(|&i| keys[i]);
        return;
    }
    let mut counts = [[0usize; 256]; 8];
    for &i in order.iter() {
        let k = keys[i];
        for pass in 0..8 {
            counts[pass][((k >> (pass * 8)) & 0xff) as usize] += 1;
        }
    }
    let mut swapped = false;
    for pass in 0..8 {
        let c = &mut counts[pass];
        if c.iter().any(|&cnt| cnt == n) {
            continue;
        }
        let mut sum = 0usize;
        for slot in c.iter_mut() {
            let v = *slot;
            *slot = sum;
            sum += v;
        }
        let shift = pass * 8;
        if !swapped {
            for &idx in order.iter() {
                let bucket = ((keys[idx] >> shift) & 0xff) as usize;
                tmp[c[bucket]] = idx;
                c[bucket] += 1;
            }
        } else {
            for &idx in tmp.iter() {
                let bucket = ((keys[idx] >> shift) & 0xff) as usize;
                order[c[bucket]] = idx;
                c[bucket] += 1;
            }
        }
        swapped = !swapped;
    }
    if swapped {
        order.copy_from_slice(tmp);
    }
}

/// 计算无 NaN 切片 `vals` 的平均秩（`pct=true` 时除以 `n`，否则返回 1..=n 平均秩）。
fn rank_dense_f64_into(
    vals: &[f64],
    pct: bool,
    out: &mut [f64],
    keys: &mut Vec<u64>,
    order: &mut Vec<usize>,
    tmp: &mut Vec<usize>,
) {
    let n = vals.len();
    if n == 0 {
        return;
    }
    keys.clear();
    keys.extend(vals.iter().map(|&v| f64_mono_key(v)));
    order.clear();
    order.extend(0..n);
    tmp.resize(n, 0);
    radix_sort_u64(keys, order, tmp);

    let inv_n = if pct { 1.0 / (n as f64) } else { 1.0 };
    let mut s = 0usize;
    while s < n {
        let v0 = vals[order[s]];
        let mut e = s + 1;
        while e < n && vals[order[e]] == v0 {
            e += 1;
        }
        let avg_rank = (((s + 1) + e) as f64 * 0.5) * inv_n;
        for idx in s..e {
            out[order[idx]] = avg_rank;
        }
        s = e;
    }
}

#[inline]
fn mean_and_pop_std(vals: &[f64]) -> (f64, f64) {
    let n = vals.len();
    if n == 0 {
        return (f64::NAN, f64::NAN);
    }
    let nf = n as f64;
    let mean = vals.iter().sum::<f64>() / nf;
    let var = vals
        .iter()
        .map(|&x| {
            let d = x - mean;
            d * d
        })
        .sum::<f64>()
        / nf;
    (mean, var.sqrt())
}

pub fn project_out_core_3d(
    mat: ArrayView3<'_, f64>,
    cold: ArrayView2<'_, f64>,
    min_valid: usize,
) -> Result<Array3<f32>, String> {
    let (d, s, l) = mat.dim();
    if cold.dim() != (d, s) {
        return Err(format!(
            "cold 形状 {:?} 与 mat 前两维 {:?} 不匹配",
            cold.dim(),
            (d, s)
        ));
    }
    let row_stride = s * l;
    let mut out_flat = vec![f32::NAN; d * row_stride];

    out_flat
        .par_chunks_mut(row_stride.max(1))
        .enumerate()
        .for_each(|(t, out_day)| {
            if t >= d || row_stride == 0 {
                return;
            }
            let mut ok_c = vec![false; s];
            let mut n_ok_c = 0usize;
            for j in 0..s {
                let f = cold[[t, j]].is_finite();
                ok_c[j] = f;
                if f {
                    n_ok_c += 1;
                }
            }
            if n_ok_c < min_valid {
                return;
            }

            let mut keys = Vec::with_capacity(s);
            let mut order = Vec::with_capacity(s);
            let mut tmp = Vec::with_capacity(s);
            let mut y_sub = Vec::with_capacity(s);
            let mut c_sub = Vec::with_capacity(s);
            let mut idx_sub = Vec::with_capacity(s);
            let mut yy = vec![0.0f64; s];
            let mut rr = vec![0.0f64; s];

            // 若当日 cold 全列有效，按需惰性准备一次 full 分支的归一化 cr
            let mut full_cr: Option<Option<Vec<f64>>> = None;

            for k in 0..l {
                idx_sub.clear();
                y_sub.clear();
                c_sub.clear();
                for j in 0..s {
                    if ok_c[j] {
                        let yv = mat[[t, j, k]];
                        if yv.is_finite() {
                            idx_sub.push(j);
                            y_sub.push(yv);
                            c_sub.push(cold[[t, j]]);
                        }
                    }
                }
                let n = idx_sub.len();
                if n < min_valid {
                    continue;
                }
                if n == s {
                    let cr_opt = full_cr.get_or_insert_with(|| {
                        let mut cr_vec = vec![0.0f64; s];
                        rank_dense_f64_into(&c_sub, true, &mut cr_vec, &mut keys, &mut order, &mut tmp);
                        for v in cr_vec.iter_mut() {
                            *v -= 0.5;
                        }
                        let (_m, sd) = mean_and_pop_std(&cr_vec);
                        if sd.is_finite() && sd > 0.0 {
                            let inv = 1.0 / sd;
                            for v in cr_vec.iter_mut() {
                                *v *= inv;
                            }
                            Some(cr_vec)
                        } else {
                            None
                        }
                    });
                    if let Some(cr_norm) = cr_opt.as_ref() {
                        rank_dense_f64_into(&y_sub, true, &mut yy[..s], &mut keys, &mut order, &mut tmp);
                        let sf = s as f64;
                        let ym_mean = yy[..s].iter().sum::<f64>() / sf;
                        let mut dot = 0.0f64;
                        for j in 0..s {
                            dot += yy[j] * cr_norm[j];
                        }
                        let b = dot / sf;
                        for j in 0..s {
                            out_day[j * l + k] = (yy[j] - (ym_mean + b * cr_norm[j])) as f32;
                        }
                    }
                } else {
                    rank_dense_f64_into(&y_sub, true, &mut yy[..n], &mut keys, &mut order, &mut tmp);
                    rank_dense_f64_into(&c_sub, true, &mut rr[..n], &mut keys, &mut order, &mut tmp);
                    let nf = n as f64;
                    let rr_mean = rr[..n].iter().sum::<f64>() / nf;
                    let mut rr_var = 0.0f64;
                    for pos in 0..n {
                        rr[pos] -= rr_mean;
                        rr_var += rr[pos] * rr[pos];
                    }
                    let sd = (rr_var / nf).sqrt();
                    if !sd.is_finite() || sd <= 0.0 {
                        continue;
                    }
                    let inv_sd = 1.0 / sd;
                    let yy_mean = yy[..n].iter().sum::<f64>() / nf;
                    let mut dot = 0.0f64;
                    for pos in 0..n {
                        rr[pos] *= inv_sd;
                        dot += yy[pos] * rr[pos];
                    }
                    let b = dot / nf;
                    for pos in 0..n {
                        let j = idx_sub[pos];
                        out_day[j * l + k] = (yy[pos] - (yy_mean + b * rr[pos])) as f32;
                    }
                }
            }
        });

    Array3::from_shape_vec((d, s, l), out_flat).map_err(|e| e.to_string())
}

#[pyfunction]
#[pyo3(signature = (mat, cold, min_valid=500))]
pub fn cold_project_out_2d<'py>(
    py: Python<'py>,
    mat: PyReadonlyArray2<'py, f64>,
    cold: PyReadonlyArray2<'py, f64>,
    min_valid: usize,
) -> PyResult<&'py PyArray2<f32>> {
    let m2 = mat.as_array();
    let (d, s) = m2.dim();
    let m3 = m2
        .into_shape((d, s, 1))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let c2 = cold.as_array();
    let out3 = py
        .allow_threads(|| project_out_core_3d(m3, c2, min_valid))
        .map_err(PyValueError::new_err)?;
    let out2 = out3
        .into_shape((d, s))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(out2.into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (mat, cold, min_valid=500))]
pub fn cold_project_out_3d<'py>(
    py: Python<'py>,
    mat: PyReadonlyArray3<'py, f64>,
    cold: PyReadonlyArray2<'py, f64>,
    min_valid: usize,
) -> PyResult<&'py PyArray3<f32>> {
    let m3 = mat.as_array();
    let c2 = cold.as_array();
    let out3 = py
        .allow_threads(|| project_out_core_3d(m3, c2, min_valid))
        .map_err(PyValueError::new_err)?;
    Ok(out3.into_pyarray(py))
}

fn rolling_max_col(col: &[f64], w: usize, min_p: usize, out: &mut [f64]) {
    let d = col.len();
    for t in 0..d {
        let start = (t + 1).saturating_sub(w);
        let mut cnt = 0usize;
        let mut mx = f64::NEG_INFINITY;
        for i in start..=t {
            let v = col[i];
            if !v.is_nan() {
                cnt += 1;
                if v > mx {
                    mx = v;
                }
            }
        }
        out[t] = if cnt >= min_p { mx } else { f64::NAN };
    }
}

fn rolling_mean_col(col: &[f64], w: usize, min_p: usize, out: &mut [f64]) {
    let d = col.len();
    for t in 0..d {
        let start = (t + 1).saturating_sub(w);
        let mut cnt = 0usize;
        let mut sum = 0.0f64;
        for i in start..=t {
            let v = col[i];
            if !v.is_nan() {
                cnt += 1;
                sum += v;
            }
        }
        out[t] = if cnt >= min_p { sum / (cnt as f64) } else { f64::NAN };
    }
}

fn rolling_std_col(col: &[f64], w: usize, min_p: usize, out: &mut [f64]) {
    let d = col.len();
    let need = min_p.max(2);
    for t in 0..d {
        let start = (t + 1).saturating_sub(w);
        let mut cnt = 0usize;
        let mut sum = 0.0f64;
        for i in start..=t {
            let v = col[i];
            if !v.is_nan() {
                cnt += 1;
                sum += v;
            }
        }
        if cnt < need {
            out[t] = f64::NAN;
            continue;
        }
        let mean = sum / (cnt as f64);
        let mut sq = 0.0f64;
        for i in start..=t {
            let v = col[i];
            if !v.is_nan() {
                let diff = v - mean;
                sq += diff * diff;
            }
        }
        out[t] = (sq / ((cnt - 1) as f64)).sqrt();
    }
}

/// 输入已对齐并滞后到 `(D, S)` 的 `close`, `turn` 与 `(D, S, 10)` 的 `styles`，返回 `(D, S)` f32 冷得分。
#[pyfunction]
pub fn cold_build_heat_core<'py>(
    py: Python<'py>,
    close: PyReadonlyArray2<'py, f64>,
    turn: PyReadonlyArray2<'py, f64>,
    styles: PyReadonlyArray3<'py, f64>,
) -> PyResult<&'py PyArray2<f32>> {
    let close_a = close.as_array();
    let turn_a = turn.as_array();
    let styles_a = styles.as_array();
    let (d, s) = close_a.dim();
    if turn_a.dim() != (d, s) || styles_a.dim() != (d, s, 10) {
        return Err(PyValueError::new_err(format!(
            "cold_build_heat_core 形状不匹配: close={:?}, turn={:?}, styles={:?}",
            close_a.dim(),
            turn_a.dim(),
            styles_a.dim()
        )));
    }

    let out = py.allow_threads(|| {
        // 1) 按股票列并行计算 8 个热度量，结果存为 8 张 (D * S) 行主序表
        let per_stock: Vec<[Vec<f64>; 8]> = (0..s)
            .into_par_iter()
            .map(|j| {
                let mut c_col = vec![f64::NAN; d];
                let mut t_col = vec![f64::NAN; d];
                for t in 0..d {
                    c_col[t] = close_a[[t, j]];
                    t_col[t] = turn_a[[t, j]];
                }
                let mut ret = vec![f64::NAN; d];
                for t in 1..d {
                    ret[t] = c_col[t] / c_col[t - 1] - 1.0;
                }
                let mut r5 = vec![f64::NAN; d];
                if d > 5 {
                    for t in 5..d {
                        r5[t] = c_col[t] / c_col[t - 5] - 1.0;
                    }
                }
                let mut r20 = vec![f64::NAN; d];
                if d > 20 {
                    for t in 20..d {
                        r20[t] = c_col[t] / c_col[t - 20] - 1.0;
                    }
                }
                let mut mx20 = vec![f64::NAN; d];
                rolling_max_col(&ret, 20, 10, &mut mx20);

                let mut tm20 = vec![f64::NAN; d];
                rolling_mean_col(&t_col, 20, 10, &mut tm20);
                let mut t20 = vec![f64::NAN; d];
                for t in 0..d {
                    t20[t] = tm20[t].ln();
                }

                let mut tm5 = vec![f64::NAN; d];
                let mut tm60 = vec![f64::NAN; d];
                rolling_mean_col(&t_col, 5, 3, &mut tm5);
                rolling_mean_col(&t_col, 60, 30, &mut tm60);
                let mut te = vec![f64::NAN; d];
                for t in 0..d {
                    te[t] = (tm5[t] / tm60[t]).ln();
                }

                let mut v20 = vec![f64::NAN; d];
                rolling_std_col(&ret, 20, 10, &mut v20);

                let mut cmx60 = vec![f64::NAN; d];
                rolling_max_col(&c_col, 60, 30, &mut cmx60);
                let mut dh = vec![f64::NAN; d];
                for t in 0..d {
                    dh[t] = c_col[t] / cmx60[t];
                }

                [ret, r5, r20, mx20, t20, te, v20, dh]
            })
            .collect();

        // 2) 按日并行：对 8 个热度量分别做 _zrank，随后对 [1, 10 styles] 做截面 OLS 取标准化残差均值
        let mut cold_flat = vec![f32::NAN; d * s];
        cold_flat
            .par_chunks_mut(s.max(1))
            .enumerate()
            .for_each(|(t, out_row)| {
                if t >= d || s == 0 {
                    return;
                }
                let mut keys = Vec::with_capacity(s);
                let mut order = Vec::with_capacity(s);
                let mut tmp = Vec::with_capacity(s);
                let mut sub_v = Vec::with_capacity(s);
                let mut sub_i = Vec::with_capacity(s);
                let mut ranks = vec![0.0f64; s];
                let mut z_day = vec![f64::NAN; s * 8];

                for h in 0..8 {
                    sub_v.clear();
                    sub_i.clear();
                    for j in 0..s {
                        let v = per_stock[j][h][t];
                        if v.is_finite() {
                            sub_v.push(v);
                            sub_i.push(j);
                        }
                    }
                    let m = sub_v.len();
                    if m > 10 {
                        rank_dense_f64_into(&sub_v, false, &mut ranks[..m], &mut keys, &mut order, &mut tmp);
                        let (rmean, rstd) = mean_and_pop_std(&ranks[..m]);
                        let inv_std = 1.0 / rstd;
                        for pos in 0..m {
                            z_day[sub_i[pos] * 8 + h] = (ranks[pos] - rmean) * inv_std;
                        }
                    }
                }

                // ok = np.isfinite(Z[t]).all(1) & np.isfinite(styles[t]).all(1)
                let mut ok_idx = Vec::with_capacity(s);
                for j in 0..s {
                    let z_ok = (0..8).all(|h| z_day[j * 8 + h].is_finite());
                    if !z_ok {
                        continue;
                    }
                    let st_ok = (0..10).all(|c| styles_a[[t, j, c]].is_finite());
                    if st_ok {
                        ok_idx.push(j);
                    }
                }
                let n_ok = ok_idx.len();
                if n_ok < 500 {
                    return;
                }

                // X: (n_ok, 11), Y: (n_ok, 8) — 与 np.linalg.lstsq(X, Y, rcond=None) 等价的 SVD 最小二乘
                let p = 11usize;
                let xm = DMatrix::from_fn(n_ok, p, |r_i, c| {
                    if c == 0 {
                        1.0
                    } else {
                        styles_a[[t, ok_idx[r_i], c - 1]]
                    }
                });
                let ym = DMatrix::from_fn(n_ok, 8, |r_i, h| z_day[ok_idx[r_i] * 8 + h]);
                let svd = SVD::new(xm.clone(), true, true);
                let u = svd.u.expect("svd u");
                let vt = svd.v_t.expect("svd vt");
                let sv = &svd.singular_values;
                let s_max = sv.iter().cloned().fold(0.0f64, f64::max);
                let rcond = s_max * (n_ok.max(p) as f64) * f64::EPSILON;
                let uty = u.transpose() * &ym;
                let mut scaled = DMatrix::zeros(p, 8);
                for i in 0..p {
                    if sv[i] > rcond {
                        let inv = 1.0 / sv[i];
                        for h in 0..8 {
                            scaled[(i, h)] = uty[(i, h)] * inv;
                        }
                    }
                }
                let beta = vt.transpose() * scaled;
                let resid = ym - xm * beta; // (n_ok, 8)

                let nf = n_ok as f64;
                let mut inv_sds = [f64::NAN; 8];
                for h in 0..8 {
                    let mut mean = 0.0f64;
                    for r_i in 0..n_ok {
                        mean += resid[(r_i, h)];
                    }
                    mean /= nf;
                    let mut var = 0.0f64;
                    for r_i in 0..n_ok {
                        let df = resid[(r_i, h)] - mean;
                        var += df * df;
                    }
                    let sd = (var / nf).sqrt();
                    if sd > 0.0 {
                        inv_sds[h] = 1.0 / sd;
                    }
                }

                for (r_i, &j) in ok_idx.iter().enumerate() {
                    let mut acc = 0.0f64;
                    let mut cnt = 0usize;
                    for h in 0..8 {
                        let v = resid[(r_i, h)] * inv_sds[h];
                        if !v.is_nan() {
                            acc += v;
                            cnt += 1;
                        }
                    }
                    if cnt > 0 {
                        out_row[j] = (-(acc / (cnt as f64))) as f32;
                    }
                }
            });

        Array2::from_shape_vec((d, s), cold_flat).expect("shape valid")
    });

    Ok(out.into_pyarray(py))
}
