#[allow(unused_imports)]
use pyo3::prelude::*;

pub mod text;
pub mod sequence;
pub mod statistics;
pub mod time_series;
pub mod pandas_ext;
pub mod tree;
pub mod error;
pub mod grouping;
pub mod parallel_computing;



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
    m.add_function(wrap_pyfunction!(time_series::fast_dtw_distance, m)?)?;
    m.add_function(wrap_pyfunction!(time_series::super_dtw_distance, m)?)?;
    m.add_function(wrap_pyfunction!(time_series::transfer_entropy, m)?)?;
    m.add_function(wrap_pyfunction!(statistics::ols, m)?)?;
    m.add_function(wrap_pyfunction!(statistics::ols_predict, m)?)?;
    m.add_function(wrap_pyfunction!(statistics::ols_residuals, m)?)?;
    m.add_function(wrap_pyfunction!(statistics::min_range_loop, m)?)?;
    m.add_function(wrap_pyfunction!(statistics::max_range_loop, m)?)?;
    m.add_function(wrap_pyfunction!(statistics::rolling_volatility, m)?)?;
    m.add_function(wrap_pyfunction!(statistics::rolling_cv, m)?)?;
    m.add_function(wrap_pyfunction!(statistics::rolling_qcv, m)?)?;
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
    m.add_function(wrap_pyfunction!(time_series::find_half_extreme_time, m)?)?;
    m.add_function(wrap_pyfunction!(time_series::fast_extreme::fast_find_half_extreme_time, m)?)?;
    m.add_function(wrap_pyfunction!(time_series::super_extreme::super_find_half_extreme_time, m)?)?;
    m.add_function(wrap_pyfunction!(time_series::calculate_large_order_nearby_small_order_time_gap, m)?)?;
    m.add_function(wrap_pyfunction!(sequence::calculate_shannon_entropy_change, m)?)?;
    m.add_function(wrap_pyfunction!(sequence::calculate_shannon_entropy_change_at_low, m)?)?;
    m.add_function(wrap_pyfunction!(sequence::calculate_base_entropy, m)?)?;
    m.add_function(wrap_pyfunction!(sequence::calculate_window_entropy, m)?)?;
    m.add_function(wrap_pyfunction!(sequence::brachistochrone_curve, m)?)?;
    m.add_function(wrap_pyfunction!(sequence::brachistochrone_curve_v2, m)?)?;
    m.add_function(wrap_pyfunction!(statistics::dataframe_corrwith, m)?)?;
    m.add_function(wrap_pyfunction!(time_series::rolling_dtw_distance, m)?)?;
    m.add_function(wrap_pyfunction!(sequence::segment_and_correlate, m)?)?;
    m.add_function(wrap_pyfunction!(sequence::test_function, m)?)?;
    m.add_function(wrap_pyfunction!(time_series::retreat_advance::analyze_retreat_advance, m)?)?;
    m.add_function(wrap_pyfunction!(time_series::retreat_advance_v2::analyze_retreat_advance_v2, m)?)?;
    m.add_function(wrap_pyfunction!(pandas_ext::rank_axis1, m)?)?;
    m.add_function(wrap_pyfunction!(pandas_ext::fast_merge, m)?)?;
    m.add_function(wrap_pyfunction!(pandas_ext::fast_merge_mixed, m)?)?;
    m.add_function(wrap_pyfunction!(pandas_ext::fast_inner_join_dataframes, m)?)?;
    m.add_function(wrap_pyfunction!(grouping::factor_grouping, m)?)?;
    m.add_function(wrap_pyfunction!(grouping::factor_correlation_by_date, m)?)?;
    m.add_function(wrap_pyfunction!(parallel_computing::run_pools_queue, m)?)?;
    m.add_function(wrap_pyfunction!(parallel_computing::query_backup, m)?)?;
    m.add_function(wrap_pyfunction!(parallel_computing::query_backup_fast, m)?)?;
    // m.add_function(wrap_pyfunction!(text::normalized_diff, m)?)?;
    Ok(())
}