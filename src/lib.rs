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
pub mod backup_reader;
pub mod order_contamination;
pub mod trade_peak_analysis;
pub mod order_neighborhood;
pub mod trade_records_ultra_sorted;
pub mod order_records_ultra_sorted;
pub mod market_correlation;
pub mod entropy_analysis;

pub mod factor_neutralization_io_optimized;


pub mod permutation_analysis_v0816_fixed;
pub mod abnormal_asks_analyzer;
pub mod series_rank;
pub mod price_breakthrough_stats;
pub mod lagged_regression;
pub mod lagged_regression_optimized;
pub mod lagged_regression_simd;
pub mod lagged_regression_incremental;



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
    m.add_function(wrap_pyfunction!(time_series::transfer_entropy_safe, m)?)?;
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
    m.add_function(wrap_pyfunction!(time_series::trend_2d, m)?)?;
    m.add_function(wrap_pyfunction!(sequence::find_max_range_product, m)?)?;
    m.add_function(wrap_pyfunction!(text::min_word_edit_distance, m)?)?;
    m.add_function(wrap_pyfunction!(text::check_string_proximity, m)?)?;
    m.add_function(wrap_pyfunction!(text::check_string_proximity_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(text::check_string_proximity_with_tolerance, m)?)?;
    m.add_function(wrap_pyfunction!(text::check_string_proximity_matrix_with_tolerance, m)?)?;
    m.add_function(wrap_pyfunction!(time_series::find_local_peaks_within_window, m)?)?;
    m.add_function(wrap_pyfunction!(pandas_ext::rolling_window_stat, m)?)?;
    m.add_function(wrap_pyfunction!(pandas_ext::rolling_window_stat_backward, m)?)?;
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
    m.add_function(wrap_pyfunction!(statistics::dataframe_corrwith_single_thread, m)?)?;
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
    m.add_function(wrap_pyfunction!(backup_reader::query_backup, m)?)?;
    m.add_function(wrap_pyfunction!(backup_reader::query_backup_fast, m)?)?;
    m.add_function(wrap_pyfunction!(backup_reader::query_backup_single_column, m)?)?;
    m.add_function(wrap_pyfunction!(backup_reader::query_backup_single_column_with_filter, m)?)?;
    m.add_function(wrap_pyfunction!(backup_reader::query_backup_columns_range_with_filter, m)?)?;
    m.add_function(wrap_pyfunction!(backup_reader::query_backup_factor_only, m)?)?;
    m.add_function(wrap_pyfunction!(backup_reader::query_backup_factor_only_with_filter, m)?)?;
    m.add_function(wrap_pyfunction!(backup_reader::query_backup_factor_only_ultra_fast, m)?)?;
    m.add_function(wrap_pyfunction!(order_contamination::order_contamination, m)?)?;
    m.add_function(wrap_pyfunction!(order_contamination::order_contamination_parallel, m)?)?;
    m.add_function(wrap_pyfunction!(order_contamination::order_contamination_bilateral, m)?)?;
    m.add_function(wrap_pyfunction!(trade_peak_analysis::trade_peak_analysis, m)?)?;
    m.add_function(wrap_pyfunction!(order_neighborhood::order_neighborhood_analysis, m)?)?;
    m.add_function(wrap_pyfunction!(trade_records_ultra_sorted::calculate_trade_time_gap_and_price_percentile_ultra_sorted, m)?)?;
    m.add_function(wrap_pyfunction!(order_records_ultra_sorted::calculate_order_time_gap_and_price_percentile_ultra_sorted, m)?)?;
    m.add_function(wrap_pyfunction!(order_records_ultra_sorted::calculate_order_time_gap_and_price_percentile_ultra_sorted_v2, m)?)?;
    m.add_function(wrap_pyfunction!(time_series::lyapunov::calculate_lyapunov_exponent, m)?)?;
    m.add_function(wrap_pyfunction!(statistics::local_correlation::local_correlation, m)?)?;
    m.add_function(wrap_pyfunction!(statistics::eigenvalue_analysis::matrix_eigenvalue_analysis, m)?)?;
    m.add_function(wrap_pyfunction!(statistics::eigenvalue_analysis::matrix_eigenvalue_analysis_optimized, m)?)?;
    m.add_function(wrap_pyfunction!(statistics::eigenvalue_analysis_modified::matrix_eigenvalue_analysis_modified, m)?)?;
    m.add_function(wrap_pyfunction!(statistics::eigenvalue_analysis_modified::matrix_eigenvalue_analysis_modified_ultra, m)?)?;
    m.add_function(wrap_pyfunction!(statistics::fast_correlation::fast_correlation_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(statistics::fast_correlation_v2::fast_correlation_matrix_v2, m)?)?;
    m.add_function(wrap_pyfunction!(statistics::rolling_correlation_mean::rolling_correlation_mean, m)?)?;
    m.add_function(wrap_pyfunction!(statistics::rolling_correlation_mean::rolling_correlation_skew, m)?)?;
    m.add_function(wrap_pyfunction!(statistics::rolling_window_core_feature::rolling_window_core_feature, m)?)?;
    m.add_function(wrap_pyfunction!(statistics::rolling_window_core_feature_optimized::rolling_window_core_feature_optimized, m)?)?;
    m.add_function(wrap_pyfunction!(statistics::rolling_window_core_feature_simd::rolling_window_core_feature_simd, m)?)?;
    m.add_function(wrap_pyfunction!(statistics::rolling_window_core_feature_ultra::rolling_window_core_feature_ultra, m)?)?;
    m.add_function(wrap_pyfunction!(market_correlation::price_volume_orderbook_correlation, m)?)?;
    m.add_function(wrap_pyfunction!(entropy_analysis::calculate_entropy_1d, m)?)?;
    m.add_function(wrap_pyfunction!(entropy_analysis::calculate_entropy_2d, m)?)?;
    m.add_function(wrap_pyfunction!(entropy_analysis::calculate_entropy_discrete_1d, m)?)?;
    m.add_function(wrap_pyfunction!(entropy_analysis::calculate_entropy_discrete_2d, m)?)?;
    m.add_function(wrap_pyfunction!(entropy_analysis::calculate_binned_entropy_1d, m)?)?;
    m.add_function(wrap_pyfunction!(entropy_analysis::calculate_binned_entropy_2d, m)?)?;
    m.add_function(wrap_pyfunction!(factor_neutralization_io_optimized::batch_factor_neutralization_io_optimized, m)?)?;
    m.add_function(wrap_pyfunction!(permutation_analysis_v0816_fixed::analyze_sequence_permutations_v0816_fixed, m)?)?;
    m.add_function(wrap_pyfunction!(statistics::hmm_trend_prediction::hmm_trend_prediction, m)?)?;
    m.add_function(wrap_pyfunction!(abnormal_asks_analyzer::analyze_asks, m)?)?;
    m.add_function(wrap_pyfunction!(series_rank::pandas_series_rank, m)?)?;
    m.add_function(wrap_pyfunction!(price_breakthrough_stats::compute_non_breakthrough_stats, m)?)?;
    m.add_function(wrap_pyfunction!(lagged_regression::rolling_lagged_regression, m)?)?;
    m.add_function(wrap_pyfunction!(lagged_regression::rolling_lagged_regression_ridge, m)?)?;
    m.add_function(wrap_pyfunction!(lagged_regression_optimized::rolling_lagged_regression_ridge_fast, m)?)?;
    m.add_function(wrap_pyfunction!(lagged_regression_simd::rolling_lagged_regression_ridge_simd, m)?)?;
    m.add_function(wrap_pyfunction!(lagged_regression_incremental::rolling_lagged_regression_ridge_incremental, m)?)?;
    // m.add_function(wrap_pyfunction!(text::normalized_diff, m)?)?;
    Ok(())
}