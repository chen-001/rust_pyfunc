use pyo3::prelude::*;

/// Formats the sum of two numbers as string.
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}




fn sakoe_chiba_window(i: usize, j: usize, radius: usize, len_s1: usize, len_s2: usize) -> bool {
    (i.saturating_sub(radius) <= j) && (j <= i + radius)
}

fn set_c(b: Option<i32>) -> i32 {
    match b {
        Some(value) => value, // 如果b不是None，则c等于b的值加1
        None => 2,                // 如果b是None，则c等于1
    }
}

// #[pyo3(signature = (s1, s2, radius=None))]
#[pyfunction]
fn dtw_distance(s1: Vec<f64>, s2: Vec<f64>, radius: Option<usize>) -> PyResult<f64> {
    // let radius_after_default = set_c(radius);
    let len_s1 = s1.len();
    let len_s2 = s2.len();
    let mut warp_dist_mat = vec![vec![f64::INFINITY; len_s2 + 1]; len_s1 + 1];
    warp_dist_mat[0][0] = 0.0;

    for i in 1..=len_s1 {
        for j in 1..=len_s2 {
            match radius {
                Some(radius_after_default) => {
                    if !sakoe_chiba_window(i, j, radius.unwrap(), len_s1, len_s2) {
                        continue;
                    }
                    },
                None => {},
            }
            let cost = (s1[i - 1] - s2[j - 1]).abs() as f64;
            warp_dist_mat[i][j] = cost + warp_dist_mat[i - 1][j].min(
                warp_dist_mat[i][j - 1].min(warp_dist_mat[i - 1][j - 1])
            );
        }
    }
    Ok(warp_dist_mat[len_s1][len_s2])
}


/// A Python module implemented in Rust.
#[pymodule]
fn rust_pyfunc(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    m.add_function(wrap_pyfunction!(dtw_distance, m)?)?;
    Ok(())
}