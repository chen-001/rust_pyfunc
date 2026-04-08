use ndarray::ArrayView2;
use numpy::PyReadonlyArray2;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

const EPS: f64 = 1e-12;

#[inline]
fn pairwise_abs_corr(x: &[f32], y: &[f32]) -> f64 {
    let mut n = 0usize;
    let mut sum_x = 0.0_f64;
    let mut sum_y = 0.0_f64;
    let mut sum_x2 = 0.0_f64;
    let mut sum_y2 = 0.0_f64;
    let mut sum_xy = 0.0_f64;

    for idx in 0..x.len() {
        let xv = x[idx] as f64;
        let yv = y[idx] as f64;
        if !xv.is_finite() || !yv.is_finite() {
            continue;
        }
        n += 1;
        sum_x += xv;
        sum_y += yv;
        sum_x2 += xv * xv;
        sum_y2 += yv * yv;
        sum_xy += xv * yv;
    }

    if n < 2 {
        return f64::NAN;
    }

    let nf = n as f64;
    let cov = sum_xy - (sum_x * sum_y) / nf;
    let var_x = sum_x2 - (sum_x * sum_x) / nf;
    let var_y = sum_y2 - (sum_y * sum_y) / nf;
    if var_x <= EPS || var_y <= EPS {
        return f64::NAN;
    }
    (cov / (var_x.sqrt() * var_y.sqrt())).abs()
}

fn greedy_select_impl(ic_by_factor: ArrayView2<'_, f32>, threshold: f64) -> Vec<usize> {
    let n_factors = ic_by_factor.nrows();
    if n_factors == 0 {
        return Vec::new();
    }

    let mut selected: Vec<usize> = Vec::with_capacity(n_factors.min(64));
    selected.push(0);

    for candidate_idx in 1..n_factors {
        let candidate = ic_by_factor.row(candidate_idx);
        let candidate_slice = candidate
            .as_slice()
            .expect("ic_by_factor must be row-major contiguous");
        let mut too_similar = false;
        for &chosen_idx in &selected {
            let chosen = ic_by_factor.row(chosen_idx);
            let chosen_slice = chosen
                .as_slice()
                .expect("ic_by_factor must be row-major contiguous");
            let corr_val = pairwise_abs_corr(candidate_slice, chosen_slice);
            if corr_val.is_finite() && corr_val >= threshold {
                too_similar = true;
                break;
            }
        }
        if !too_similar {
            selected.push(candidate_idx);
        }
    }
    selected
}

#[pyfunction]
#[pyo3(signature = (ic_by_factor, threshold))]
pub fn tail_v2_select_by_ic_corr_abs_f32(
    ic_by_factor: PyReadonlyArray2<f32>,
    threshold: f32,
) -> PyResult<Vec<usize>> {
    let view = ic_by_factor.as_array();
    if view.ndim() != 2 {
        return Err(PyValueError::new_err("ic_by_factor 必须是二维数组"));
    }
    if threshold.is_nan() {
        return Err(PyValueError::new_err("threshold 不能是 NaN"));
    }
    if !view.is_standard_layout() {
        return Err(PyValueError::new_err(
            "ic_by_factor 必须是 C contiguous；请在 Python 侧先做 np.ascontiguousarray",
        ));
    }
    Ok(greedy_select_impl(view, threshold as f64))
}

