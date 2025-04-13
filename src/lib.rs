#[allow(unused_imports)]
use pyo3::prelude::*;

pub mod text;
pub mod sequence;
pub mod statistics;
pub mod time_series;
pub mod pandas_ext;
pub mod tree;

/// Formats the sum of two numbers as string.
#[pyfunction]
#[pyo3(signature = (a, b))]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

/// A Python module implemented in Rust.
#[pymodule]
fn rust_pyfunc(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    m.add_function(wrap_pyfunction!(time_series::dtw_distance, m)?)?;
    m.add_function(wrap_pyfunction!(time_series::transfer_entropy, m)?)?;
    m.add_function(wrap_pyfunction!(statistics::ols, m)?)?;
    m.add_function(wrap_pyfunction!(statistics::ols_predict, m)?)?;
    m.add_function(wrap_pyfunction!(statistics::min_range_loop, m)?)?;
    m.add_function(wrap_pyfunction!(statistics::max_range_loop, m)?)?;
    m.add_function(wrap_pyfunction!(text::vectorize_sentences, m)?)?;
    m.add_function(wrap_pyfunction!(text::vectorize_sentences_list, m)?)?;
    m.add_function(wrap_pyfunction!(text::jaccard_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(sequence::identify_segments, m)?)?;
    m.add_function(wrap_pyfunction!(time_series::trend, m)?)?;
    m.add_function(wrap_pyfunction!(time_series::trend_fast, m)?)?;
    m.add_function(wrap_pyfunction!(sequence::find_max_range_product, m)?)?;
    m.add_function(wrap_pyfunction!(text::min_word_edit_distance, m)?)?;
    m.add_function(wrap_pyfunction!(time_series::find_local_peaks_within_window, m)?)?;
    m.add_function(wrap_pyfunction!(pandas_ext::rolling_window_stat, m)?)?;
    m.add_class::<tree::PriceTree>()?;
    // m.add_function(wrap_pyfunction!(sequence::compute_top_eigenvalues, m)?)?;
    m.add_function(wrap_pyfunction!(sequence::compute_max_eigenvalue, m)?)?;
    m.add_function(wrap_pyfunction!(time_series::find_follow_volume_sum_same_price, m)?)?;
    m.add_function(wrap_pyfunction!(time_series::find_follow_volume_sum_same_price_and_flag, m)?)?;
    m.add_function(wrap_pyfunction!(time_series::mark_follow_groups, m)?)?;
    m.add_function(wrap_pyfunction!(time_series::mark_follow_groups_with_flag, m)?)?;
    m.add_function(wrap_pyfunction!(time_series::find_half_energy_time, m)?)?;
    m.add_function(wrap_pyfunction!(sequence::calculate_shannon_entropy_change, m)?)?;
    m.add_function(wrap_pyfunction!(sequence::calculate_shannon_entropy_change_at_low, m)?)?;
    m.add_function(wrap_pyfunction!(sequence::brachistochrone_curve, m)?)?;
    // m.add_function(wrap_pyfunction!(text::normalized_diff, m)?)?;
    Ok(())
}