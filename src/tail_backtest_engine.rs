//! tail_backtest_engine：全新高性能回测引擎。
//!
//! 核心设计：回归 v4 的 thread-per-factor 模型（N 线程各自独立处理整个流水线），
//! 消灭 v5 的 IO/CPU 分离 + bounded(16) 瓶颈。直接读 colblk 列式存储（scatter_map 优化），
//! 每个线程自带 Reader + scatter_maps，从 unbounded channel 取因子独立处理。
//!
//! 与 v4 的区别：直接读 colblk（不走 parquet 中间格式）。
//! 与 v5 的区别：无 IO/CPU 分离，unbounded channel 不限制并行度。

use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
use std::thread;
use std::time::Instant;

use crossbeam::channel::{unbounded, Receiver, Sender};
use ndarray::Array2;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::tail_v5_pipeline::{
    self, build_shared_inputs, build_selection_config, factor_result_path,
    append_completed_source, read_task_result, write_task_result, write_aggregated_outputs,
    format_hms, init_status_line, update_status_line, reset_status_line, is_terminal,
    AggregatedCandidates, ProcessStats, SharedInputs, TailTask, TailTaskResult,
};

/// 单个因子的完整处理（IO + 计算 + 写结果），在一个线程内串行执行。
/// 复用 v5 的 process_task_with_values_v7（含 selected_slots + gap1/gap5 合并优化）。
fn process_single_factor(
    task: &TailTask,
    raw_values: Array2<f32>,
    shared: &SharedInputs,
) -> Result<TailTaskResult, String> {
    tail_v5_pipeline::process_task_with_values_v7(task, raw_values, shared)
}

/// 解析 col_idx（从 "store_dir::col_idx" 格式的 factor_path 中提取）
fn parse_col_idx(factor_path: &str) -> Option<usize> {
    factor_path.splitn(2, "::").nth(1)?.parse::<usize>().ok()
}

#[pyfunction]
#[pyo3(signature = (
    colblk_store_dir,
    factor_names,
    factor_paths,
    dates,
    stocks,
    windows,
    fold,
    n_jobs,
    min_valid,
    cache_root,
    style_data_path,
    ret_gap1_path,
    ret_sum_gap1_path,
    ret_gap5_path,
    ret_sum_gap5_path,
    restrict_path,
    index_ret_path,
    backtest_start,
    cover_rate,
    ret_point_neu_gap5,
    ret_point_neu_gap1,
    ic_point_neu_gap5,
    ic_point_neu_gap1,
    ret_point_gap5,
    ret_point_gap1,
    ic_point_gap5,
    ic_point_gap1,
    ic_more_important_gap5,
    ic_more_important_gap1,
    majority_count_threshold,
    zero_max_threshold,
    nan_max_threshold,
    industry_neutralize=true,
))]
#[allow(clippy::too_many_arguments)]
pub fn tail_backtest_engine<'py>(
    py: Python<'py>,
    colblk_store_dir: String,
    factor_names: Vec<String>,
    factor_paths: Vec<String>,
    dates: Vec<i32>,
    stocks: Vec<String>,
    windows: Vec<usize>,
    fold: bool,
    n_jobs: usize,
    min_valid: usize,
    cache_root: String,
    style_data_path: String,
    ret_gap1_path: String,
    ret_sum_gap1_path: String,
    ret_gap5_path: String,
    ret_sum_gap5_path: String,
    restrict_path: String,
    index_ret_path: String,
    backtest_start: i32,
    cover_rate: f64,
    ret_point_neu_gap5: f64,
    ret_point_neu_gap1: f64,
    ic_point_neu_gap5: f64,
    ic_point_neu_gap1: f64,
    ret_point_gap5: f64,
    ret_point_gap1: f64,
    ic_point_gap5: f64,
    ic_point_gap1: f64,
    ic_more_important_gap5: Option<f64>,
    ic_more_important_gap1: Option<f64>,
    majority_count_threshold: f64,
    zero_max_threshold: f64,
    nan_max_threshold: f64,
    industry_neutralize: bool,
) -> PyResult<PyObject> {
    if factor_names.len() != factor_paths.len() {
        return Err(PyValueError::new_err("factor_names 和 factor_paths 长度必须一致"));
    }
    if n_jobs == 0 {
        return Err(PyValueError::new_err("n_jobs 必须大于 0"));
    }

    let output = py.allow_threads(|| -> Result<(usize, usize, HashMap<String, usize>), String> {
        let started = Instant::now();
        let cache_root_path = PathBuf::from(&cache_root);
        let task_results_dir = cache_root_path.join("task_results");
        let logs_dir = cache_root_path.join("logs");
        let completed_log_path = logs_dir.join("completed_sources.txt");
        fs::create_dir_all(&task_results_dir).map_err(|e| format!("创建 task_results 目录失败: {}", e))?;
        fs::create_dir_all(&logs_dir).map_err(|e| format!("创建 logs 目录失败: {}", e))?;

        // 构建共享输入
        let config = build_selection_config(
            cover_rate, ret_point_neu_gap5, ret_point_neu_gap1,
            ic_point_neu_gap5, ic_point_neu_gap1,
            ret_point_gap5, ret_point_gap1,
            ic_point_gap5, ic_point_gap1,
            ic_more_important_gap5, ic_more_important_gap1,
            majority_count_threshold, zero_max_threshold, nan_max_threshold,
        );
        let shared = build_shared_inputs(
            dates, stocks, windows, fold, min_valid, backtest_start,
            industry_neutralize,
            &style_data_path,
            &ret_gap1_path, &ret_sum_gap1_path,
            &ret_gap5_path, &ret_sum_gap5_path,
            &restrict_path, &index_ret_path,
            config,
        )?;

        // ---- 断点续算：恢复已完成因子 ----
        let mut aggregated = AggregatedCandidates::default();
        let mut completed_sources = HashSet::<String>::new();
        let mut stats = ProcessStats::default();
        for (source_factor, _factor_path) in factor_names.iter().zip(factor_paths.iter()) {
            let result_path = factor_result_path(&task_results_dir, source_factor);
            if result_path.exists() {
                if let Ok(task_result) = read_task_result(&result_path) {
                    let passed = task_result.passed;
                    let raw_cov = task_result.eliminated_by_raw_cover;
                    let any_window = task_result.any_window_passed_preflight;
                    let has_data = !task_result.raw_summary_gap1.is_empty()
                        || !task_result.raw_summary_gap5.is_empty()
                        || !task_result.neu_summary_gap1.is_empty()
                        || !task_result.neu_summary_gap5.is_empty();
                    if passed {
                        stats.restored_pass += 1;
                    } else if raw_cov {
                        stats.restored_raw_cov += 1;
                    } else if !any_window && !has_data {
                        if task_result.preflight_maj_failed_windows > 0
                            || task_result.preflight_zero_failed_windows > 0
                            || task_result.preflight_nan_failed_windows > 0
                        {
                            stats.restored_preflight += 1;
                        } else {
                            stats.restored_unknown += 1;
                        }
                    } else {
                        stats.restored_ret_ic += 1;
                    }
                    stats.preflight_maj_windows += task_result.preflight_maj_failed_windows;
                    stats.preflight_zero_windows += task_result.preflight_zero_failed_windows;
                    stats.preflight_nan_windows += task_result.preflight_nan_failed_windows;
                    aggregated.merge_task(task_result);
                    completed_sources.insert(source_factor.clone());
                }
            }
        }
        let restored_sources = stats.restored_pass + stats.restored_raw_cov
            + stats.restored_preflight + stats.restored_ret_ic + stats.restored_unknown;

        // ---- 构建待处理任务 ----
        let pending_tasks: Vec<TailTask> = factor_names
            .iter()
            .zip(factor_paths.iter())
            .filter_map(|(source_factor, factor_path)| {
                if completed_sources.contains(source_factor) {
                    None
                } else {
                    Some(TailTask::new(source_factor.clone(), factor_path.clone()))
                }
            })
            .collect();

        let total_pending = pending_tasks.len();
        if total_pending > 0 {
            init_status_line();
            let current_time = chrono::Local::now().format("%Y-%m-%d %H:%M:%S");
            let total = factor_names.len();
            let l1 = format!("[{}] Backtest Engine 启动，待处理 {}/{} 个原始因子",
                current_time, total_pending, total);
            let l2 = format!("累计通过 {} | 淘汰 raw_cov={} preflight={} ret_ic={} 未知={}",
                stats.restored_pass,
                stats.restored_raw_cov, stats.restored_preflight,
                stats.restored_ret_ic, stats.restored_unknown);
            let l3 = format!("恢复 {} 个 | 即将开始处理（{} 线程，thread-per-factor）...",
                restored_sources, n_jobs);
            update_status_line(&l1, &l2, &l3);
        }

        // ---- 核心：unbounded channel + N 线程各自独立处理 ----
        let (task_sender, task_receiver): (Sender<TailTask>, Receiver<TailTask>) = unbounded();
        let (result_sender, result_receiver) =
            unbounded::<Result<TailTaskResult, (String, String)>>();

        for task in pending_tasks {
            task_sender.send(task).map_err(|e| format!("发送任务失败: {}", e))?;
        }
        drop(task_sender);

        let shared_arc = Arc::new(shared);
        let processed_count = Arc::new(AtomicUsize::new(0));

        // 启动 N 个 worker 线程（thread-per-factor，无 IO/CPU 分离）
        let mut handles = Vec::with_capacity(n_jobs);
        for _ in 0..n_jobs {
            let rx = task_receiver.clone();
            let tx = result_sender.clone();
            let shared_clone = shared_arc.clone();
            let store_dir = colblk_store_dir.clone();
            let dates_arc = shared_clone.dates.clone();
            let stocks_arc = shared_clone.stocks.clone();
            handles.push(thread::spawn(move || {
                // 每个线程独立打开 Reader（pread 线程安全）
                let reader = match crate::factor_store_v5::FactorStoreReader::open(&store_dir) {
                    Ok(r) => r,
                    Err(_) => return,
                };
                // 预计算 scatter_maps（1 次，所有因子复用）
                let scatter_maps = reader.precompute_scatter_maps(
                    dates_arc.as_slice(),
                    stocks_arc.as_slice(),
                );

                while let Ok(task) = rx.recv() {
                    // 1. 解析 col_idx
                    let col_idx = match parse_col_idx(&task.factor_path) {
                        Some(v) => v,
                        None => continue,
                    };

                    // 2. 快速读因子（scatter_map 优化）
                    let raw_values = match reader.read_factor_to_matrix_fast(
                        col_idx,
                        dates_arc.as_slice(),
                        stocks_arc.as_slice(),
                        &scatter_maps,
                    ) {
                        Ok(m) => m,
                        Err(_) => continue,
                    };

                    // 3. 完整计算流水线（rank_roll → preflight → backtest → neutralize → backtest）
                    let task_name = task.source_factor.clone();
                    let outcome = process_single_factor(&task, raw_values, &shared_clone)
                        .map_err(|err| (task_name, err));

                    if tx.send(outcome).is_err() {
                        break;
                    }
                }
            }));
        }
        drop(result_sender);

        // ---- 主线程收集结果 ----
        let mut processed_sources = 0usize;
        while let Ok(task_outcome) = result_receiver.recv() {
            match task_outcome {
                Ok(task_result) => {
                    let result_path = factor_result_path(&task_results_dir, &task_result.source_factor);
                    let is_passed = task_result.passed;
                    let is_raw_cov = task_result.eliminated_by_raw_cover;
                    let any_window = task_result.any_window_passed_preflight;
                    let preflight_maj = task_result.preflight_maj_failed_windows;
                    let preflight_zero = task_result.preflight_zero_failed_windows;
                    let preflight_nan = task_result.preflight_nan_failed_windows;
                    write_task_result(&result_path, &task_result)?;
                    append_completed_source(&completed_log_path, &task_result.source_factor)?;
                    aggregated.merge_task(task_result);
                    processed_sources += 1;
                    processed_count.store(processed_sources, AtomicOrdering::Relaxed);

                    if is_passed {
                        stats.done_pass += 1;
                    } else if is_raw_cov {
                        stats.done_raw_cov += 1;
                    } else if !any_window {
                        stats.done_preflight += 1;
                    } else {
                        stats.done_ret_ic += 1;
                    }
                    stats.done = processed_sources;
                    stats.preflight_maj_windows += preflight_maj;
                    stats.preflight_zero_windows += preflight_zero;
                    stats.preflight_nan_windows += preflight_nan;

                    // 进度显示
                    if total_pending > 0 {
                        let elapsed = started.elapsed();
                        let progress = processed_sources as f64 / total_pending as f64;
                        let estimated_total_secs = if progress > 0.0 {
                            elapsed.as_secs_f64() / progress
                        } else {
                            elapsed.as_secs_f64()
                        };
                        let remaining_secs = ((estimated_total_secs - elapsed.as_secs_f64()).max(0.0)) as u64;
                        let (eh, em, es) = format_hms(elapsed.as_secs());
                        let (rh, rm, rs) = format_hms(remaining_secs);
                        let current_time = chrono::Local::now().format("%Y-%m-%d %H:%M:%S");

                        let cum_pass = stats.restored_pass + stats.done_pass;
                        let cum_total = restored_sources + processed_sources;
                        let cum_raw_cov = stats.restored_raw_cov + stats.done_raw_cov;
                        let cum_preflight = stats.restored_preflight + stats.done_preflight;
                        let cum_ret_ic = stats.restored_ret_ic + stats.done_ret_ic;
                        let cum_unknown = stats.restored_unknown + stats.done_unknown;

                        let l1 = format!(
                            "[{}] Engine 进度 {}/{} ({:.1}%)，已用{}h{}m{}s，预计剩余{}h{}m{}s",
                            current_time, processed_sources, total_pending,
                            progress * 100.0, eh, em, es, rh, rm, rs,
                        );
                        let l2 = format!(
                            "累计通过 {} ({:.0}%) | 淘汰 raw_cov={} preflight={} ret_ic={} 未知={}",
                            cum_pass,
                            if cum_total > 0 { cum_pass as f64 * 100.0 / cum_total as f64 } else { 0.0 },
                            cum_raw_cov, cum_preflight, cum_ret_ic, cum_unknown,
                        );
                        let l3 = format!(
                            "maj={}w zero={}w nan={}w | 本次 {}(通过{}) | 恢复 {}(通过{})",
                            stats.preflight_maj_windows, stats.preflight_zero_windows,
                            stats.preflight_nan_windows,
                            stats.done, stats.done_pass,
                            restored_sources, stats.restored_pass,
                        );
                        update_status_line(&l1, &l2, &l3);
                    }
                }
                Err((task_name, err)) => {
                    reset_status_line();
                    return Err(format!("处理因子 {} 失败: {}", task_name, err));
                }
            }
        }

        if total_pending > 0 {
            println!();
            reset_status_line();
        }

        for handle in handles {
            let _ = handle.join();
        }

        write_aggregated_outputs(&cache_root_path, &aggregated)?;

        let mut candidate_counts = HashMap::new();
        candidate_counts.insert("rolled_gap1".to_string(), aggregated.raw_summary_gap1.len());
        candidate_counts.insert("rolled_gap5".to_string(), aggregated.raw_summary_gap5.len());
        candidate_counts.insert("neu_gap1".to_string(), aggregated.neu_summary_gap1.len());
        candidate_counts.insert("neu_gap5".to_string(), aggregated.neu_summary_gap5.len());
        Ok((processed_sources, restored_sources, candidate_counts))
    }).map_err(PyRuntimeError::new_err)?;

    let info = PyDict::new(py);
    info.set_item("processed_sources", output.0)?;
    info.set_item("restored_sources", output.1)?;
    let candidate_counts = PyDict::new(py);
    for (key, value) in output.2 {
        candidate_counts.set_item(key, value)?;
    }
    info.set_item("candidate_counts", candidate_counts)?;
    Ok(info.into())
}
